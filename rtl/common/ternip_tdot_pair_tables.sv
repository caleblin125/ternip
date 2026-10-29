// ternip_tdot_pair_tables
//
// Table-lookup (mu = 2) ternary dot product for ternip_pkg::TERNARY_PAIR4 weights. Replaces
// tmatmul's import buffer: instead of the activations, each lane pair (2q, 2q+1) keeps the
// four values a, b, a+b, a-b of its activations (a = lane 2q, b = lane 2q+1) for each of
// the ROWS import rows, in one distributed RAM of 4*ROWS x (ACT_WIDTH+1) bits (32 deep at
// ROWS = 8: the LUTRAM depth is fully used). A weight pair arrives as a 4-bit pair code
// {nonzero, sign, entry}: entry 0..3 selects (1,0), (0,1), (1,1), (1,-1), sign negates the
// pair, so every one of the 9 trit pairs is one table read, with the sign applied as a
// ones'-complement operand (plus the injected carry) and the zero pair gated. One CSA row per
// pair, so the reduction tree has NUM_LANES/2 rows instead of NUM_LANES.
//
// Import: IMPORT_WIDTH activations per beat (the vector-register chunk), lanes in order, rows
// in order. Each beat completes IMPORT_WIDTH/2 pairs; a pair's four entries are written over
// four cycles by one of 2*IMPORT_WIDTH write buses (pair q uses bus q % (2*IMPORT_WIDTH),
// which is reused every fourth beat at the earliest). A bus holds only a and b; a+b and a-b
// come from one adder-subtractor per bus. imp_busy_o is high while a bus is still writing.
//
// GO: one beat of pair codes per cycle (ROW_PARALLELISM rows of NUM_LANES/2 codes) with its
// import row; registered, read from the tables, registered again, then one
// ternip_tdot_accumulator per row (ones'-complement CSA / adder tree / GPC). The two input
// stages advance together whenever the accumulator takes a beat or stage 2 is empty.
//
// Storage per pair vs the activation buffer: 4*(ACT_WIDTH+1) bits per row instead of
// 2*ACT_WIDTH (+6% bits / weight-row more than 2x), but the activation buffer's pipeline
// registers (three TP-wide words in ternip_pipelined_mem) and the import gearbox (one
// TP-wide word) are gone.

`define SAFE_CLOG2(x) ( (((x)==1) || ((x)==0))? 1 : $clog2((x)))

module ternip_tdot_pair_tables #(
    parameter int NUM_LANES = 128,          // lanes per row datapath (even)
    parameter int ROW_PARALLELISM = 1,      // row datapaths sharing the tables
    parameter int ROWS = 8,                 // import rows (beats per output row)
    parameter int ACT_WIDTH = 16,
    parameter int IMPORT_WIDTH = 2,         // activations per import beat (even)
    parameter type result_t = logic signed [15:0],
    parameter ternip_pkg::tdot_arith_e Arith = ternip_pkg::TDOT_ONES_CSA,
    parameter ternip_pkg::tdot_correction_e Correction = ternip_pkg::TDOT_CORR_INJECT_BITS,
    parameter int GPC_K = 6,
    parameter int GPC_FINAL_ROWS = 2,
    parameter bit GPC_ADD = 0,
    parameter int D = 1,

    localparam int Pairs = NUM_LANES / 2,
    localparam type act_t = logic signed [ACT_WIDTH-1:0],
    localparam type row_index_t = logic [`SAFE_CLOG2(ROWS)-1:0]
) (
    input  logic                                          clk_i,
    input  logic                                          rst_ni,

    // import (imp_clear_i holds the import position at lane 0, row 0)
    input  logic                                          imp_clear_i,
    input  logic                                          imp_valid_i,
    input  act_t [IMPORT_WIDTH-1:0]                       imp_act_i,
    output logic                                          imp_busy_o,

    // GO beats
    output logic                                          go_ready_o,
    input  logic                                          go_valid_i,
    input  logic                                          go_final_i,
    input  row_index_t                                    go_row_i,
    input  logic [ROW_PARALLELISM-1:0][Pairs-1:0][3:0]    go_codes_i,

    input  logic                                          out_ready_i,
    output logic [ROW_PARALLELISM-1:0]                    out_valid_o,
    output result_t [ROW_PARALLELISM-1:0]                 out_result_o
);

localparam int PairsPerBeat = IMPORT_WIDTH / 2;
localparam int NumBuses = 4 * PairsPerBeat;
localparam int BeatsPerRow = NUM_LANES / IMPORT_WIDTH;
localparam int SlotsPerRow = Pairs / NumBuses;      // pairs per row on one bus
localparam int OpWidth = ACT_WIDTH + 1;
typedef logic signed [OpWidth-1:0] op_t;
typedef logic [`SAFE_CLOG2(SlotsPerRow)-1:0] slot_t;

`ifndef SYNTHESIS
initial begin
    assert (IMPORT_WIDTH % 2 == 0) else $fatal(1, "IMPORT_WIDTH (VectorParallelism) must be even");
    // pair q writes through bus q % NumBuses, so rows must hold whole bus rounds
    assert (Pairs % NumBuses == 0)
        else $fatal(1, "NUM_LANES (%0d) must be a multiple of 4*IMPORT_WIDTH (%0d)", NUM_LANES, 4*IMPORT_WIDTH);
    assert (Arith != ternip_pkg::TDOT_TWOS_ADDER_TREE)
        else $fatal(1, "pair tables need a ones'-complement Arith");
end
`endif

// ---- import position ------------------------------------------------------------------------
logic [`SAFE_CLOG2(BeatsPerRow)-1:0] imp_beat_q;
row_index_t imp_row_q;
always_ff @(posedge clk_i) begin
    if (!rst_ni || imp_clear_i) begin
        imp_beat_q <= '0;
        imp_row_q <= '0;
    end else if (imp_valid_i) begin
        imp_beat_q <= imp_beat_q + 1'b1;
        if (imp_beat_q == BeatsPerRow-1) begin
            imp_beat_q <= '0;
            imp_row_q <= imp_row_q + 1'b1;
        end
    end
end

// ---- table write buses: bus m = (beat % 4) * PairsPerBeat + k for pair k of the beat --------
logic [NumBuses-1:0] bus_active_q;
logic [NumBuses-1:0][1:0] bus_phase_q;
act_t [NumBuses-1:0] bus_a_q, bus_b_q;
slot_t [NumBuses-1:0] bus_slot_q;
row_index_t [NumBuses-1:0] bus_row_q;
op_t [NumBuses-1:0] bus_w_data;

wire logic [1:0] imp_group = 2'(imp_beat_q);
wire slot_t imp_slot = slot_t'(imp_beat_q >> 2);

always_ff @(posedge clk_i) begin
    for (int m = 0; m < NumBuses; m++) begin
        if (bus_active_q[m]) bus_phase_q[m] <= bus_phase_q[m] + 1'b1;
        if (bus_active_q[m] && bus_phase_q[m] == 2'd3) bus_active_q[m] <= 1'b0;
        if (imp_valid_i && imp_group == 2'(m / PairsPerBeat)) begin
            bus_active_q[m] <= 1'b1;
            bus_phase_q[m] <= '0;
            bus_a_q[m] <= imp_act_i[2 * (m % PairsPerBeat)];
            bus_b_q[m] <= imp_act_i[2 * (m % PairsPerBeat) + 1];
            bus_slot_q[m] <= imp_slot;
            bus_row_q[m] <= imp_row_q;
        end
    end
    if (!rst_ni) bus_active_q <= '0;
end

for (genvar m_GEN = 0; m_GEN < NumBuses; m_GEN++) begin : bus
    wire op_t a = op_t'(bus_a_q[m_GEN]);
    wire op_t b = op_t'(bus_b_q[m_GEN]);
    always_comb begin
        unique case (bus_phase_q[m_GEN])
            2'd0: bus_w_data[m_GEN] = a;
            2'd1: bus_w_data[m_GEN] = b;
            2'd2: bus_w_data[m_GEN] = a + b;
            default: bus_w_data[m_GEN] = a - b;
        endcase
    end
end

assign imp_busy_o = |bus_active_q;

// ---- GO stage 1: codes and row ------------------------------------------------------------
logic go_valid_q, go_final_q;
row_index_t go_row_q;
logic [ROW_PARALLELISM-1:0][Pairs-1:0][3:0] go_codes_q;

logic acc_in_ready;
logic op_valid_q, op_final_q;
op_t [ROW_PARALLELISM-1:0][Pairs-1:0] op_d, op_q;
logic [ROW_PARALLELISM-1:0][Pairs-1:0] pos_d, neg_d, pos_q, neg_q;

assign go_ready_o = acc_in_ready || !op_valid_q;

always_ff @(posedge clk_i) begin
    if (!rst_ni) go_valid_q <= 1'b0;
    else if (go_ready_o) go_valid_q <= go_valid_i;
    if (go_ready_o) begin
        go_final_q <= go_final_i;
        go_row_q <= go_row_i;
        go_codes_q <= go_codes_i;
    end
end

// ---- pair tables -----------------------------------------------------------------------------
for (genvar q_GEN = 0; q_GEN < Pairs; q_GEN++) begin : pair
    localparam int M = q_GEN % NumBuses;
    localparam int Slot = q_GEN / NumBuses;
    (* ram_style = "distributed" *) op_t mem [4*ROWS];
    always_ff @(posedge clk_i)
        if (bus_active_q[M] && bus_slot_q[M] == slot_t'(Slot))
            mem[4*bus_row_q[M] + bus_phase_q[M]] <= bus_w_data[M];
    for (genvar r_GEN = 0; r_GEN < ROW_PARALLELISM; r_GEN++) begin : row
        wire logic [3:0] code = go_codes_q[r_GEN][q_GEN];
        assign op_d[r_GEN][q_GEN] = mem[4*go_row_q + code[1:0]];
        assign pos_d[r_GEN][q_GEN] = code[3] && !code[2];
        assign neg_d[r_GEN][q_GEN] = code[3] && code[2];
    end
end

// ---- GO stage 2: operands ------------------------------------------------------------------
always_ff @(posedge clk_i) begin
    if (!rst_ni) op_valid_q <= 1'b0;
    else if (go_ready_o) op_valid_q <= go_valid_q;
    if (go_ready_o) begin
        op_final_q <= go_final_q;
        op_q <= op_d;
        pos_q <= pos_d;
        neg_q <= neg_d;
    end
end

logic [ROW_PARALLELISM-1:0] acc_in_ready_rows;
assign acc_in_ready = acc_in_ready_rows[0];

for (genvar r_GEN = 0; r_GEN < ROW_PARALLELISM; r_GEN++) begin : row
    ternip_tdot_accumulator #(
        .NUM_LANES(Pairs),
        .ACT_WIDTH(OpWidth),
        .result_t(result_t),
        .Arith(Arith),
        .Correction(Correction),
        .GPC_K(GPC_K),
        .GPC_FINAL_ROWS(GPC_FINAL_ROWS),
        .GPC_ADD(GPC_ADD),
        .D(D)
    ) tdot_accumulator (
        .clk_i,
        .rst_ni,
        .in_ready_o(acc_in_ready_rows[r_GEN]),
        .in_valid_i(op_valid_q),
        .in_final_i(op_final_q),
        .in_pos_i(pos_q[r_GEN]),
        .in_neg_i(neg_q[r_GEN]),
        .in_packed_i('0),
        .in_act_i(op_q[r_GEN]),
        .out_ready_i,
        .out_valid_o(out_valid_o[r_GEN]),
        .out_result_o(out_result_o[r_GEN])
    );
end

endmodule

`undef SAFE_CLOG2
