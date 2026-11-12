// ternip_tdot_group_tables
//
// Table-lookup (mu = N = 4 or 5) ternary dot product for ternip_pkg::TERNARY_GROUP4 /
// TERNARY_GROUP5 weights, in block RAM / UltraRAM. Same role as ternip_tdot_pair_tables
// (it replaces tmatmul's import buffer), with one table read per group of N lanes:
//
//   - Lanes (Ng .. Ng+N-1) form group g (the last group is padded with zero activations).
//     For each import row, group g's table holds the activation sum of every sign-folded
//     nonzero trit pattern: (3^N - 1) / 2 = 40 (N = 4) or 121 (N = 5) entries, at index
//     1..entries in the order of ternip_pkg::group_pattern; index 0 is the zero pattern.
//   - A weight group arrives as a code {sign, index} (ternip_pkg::group_code_bits(N) = 7 / 8
//     bits). The table entry is applied as a ones'-complement operand (sign) with the
//     injected carry, and index 0 is gated, so every one of the 3^N patterns is one read.
//   - Entries are ACT_WIDTH + clog2(N) bits (18 / 19 at 16-bit activations): exact.
//   - The reduction tree has ceil(NUM_LANES/N) rows (32 / 26 at 128 lanes) instead of
//     NUM_LANES/2 (64) for pair tables.
//
// Storage: two groups per RAM unit, one port each (true dual port), depth 2*ROWS*2^idx_bits
// (1K x 18 at N = 4, ROWS = 8: one RAMB18; 2K x 19 at N = 5: one URAM, or 1.5 RAMB36).
// Units 0 .. URAM_UNITS-1 are UltraRAM, the rest block RAM. A port writes its group's table
// during IMPORT and reads it during GO (tmatmul never overlaps the two here).
//
// Build: each group has one walker that fills the table with one add/sub per entry, in the
// order of ternip_pkg::group_walk_step (each index differs from the previous one in one trit
// by +-1), starting as soon as the group's last lane has arrived. That is 40 / 121 cycles per
// group and row. imp_ready_o stalls an import beat that would overwrite the activations of a
// group whose walker is still busy: never at N = 4 with VP 2 (64 cycles per 128-lane row),
// ~8 cycles per row at N = 4 VP 4, ~57 cycles per row at N = 5 VP 2.
//
// GO: one beat of group codes per cycle with its import row: registered (stage 1), RAM read
// (stage 2: the RAM's output latch), optional RAM output register (OUT_REG, stage 3), then
// one ternip_tdot_accumulator. All stages advance together whenever the accumulator takes a
// beat or the last stage is empty (TdotRegisteredReady: that enable is a register).

`define SAFE_CLOG2(x) ( (((x)==1) || ((x)==0))? 1 : $clog2((x)))

module ternip_tdot_group_tables #(
    parameter int GROUP_TRITS = 4,          // N: 4 or 5
    parameter int NUM_LANES = 128,          // lanes per row datapath
    parameter int ROW_PARALLELISM = 1,      // must be 1
    parameter int ROWS = 8,                 // import rows (beats per output row)
    parameter int ACT_WIDTH = 16,
    parameter int IMPORT_WIDTH = 2,         // activations per import beat
    parameter type result_t = logic signed [15:0],
    parameter ternip_pkg::tdot_arith_e Arith = ternip_pkg::TDOT_ONES_CSA,
    parameter ternip_pkg::tdot_correction_e Correction = ternip_pkg::TDOT_CORR_INJECT_BITS,
    parameter int GPC_K = 6,
    parameter int GPC_FINAL_ROWS = 2,
    parameter bit GPC_ADD = 0,
    parameter int D = 1,
    parameter int URAM_UNITS = 0,           // two-group RAM units mapped to UltraRAM
    parameter bit OUT_REG = 1,              // RAM output register (GO stage 3)
    parameter int STREAM_WIDTH = 8 * ((ternip_pkg::group_code_bits(GROUP_TRITS)
                                       * ((NUM_LANES + GROUP_TRITS - 1) / GROUP_TRITS) + 7) / 8),

    localparam int N = GROUP_TRITS,
    localparam type act_t = logic signed [ACT_WIDTH-1:0],
    localparam type row_index_t = logic [`SAFE_CLOG2(ROWS)-1:0]
) (
    input  logic                                          clk_i,
    input  logic                                          rst_ni,

    // import (imp_clear_i holds the import position at lane 0, row 0)
    input  logic                                          imp_clear_i,
    output logic                                          imp_ready_o,
    input  logic                                          imp_valid_i,
    input  act_t [IMPORT_WIDTH-1:0]                       imp_act_i,
    output logic                                          imp_busy_o,

    // GO beats
    output logic                                          go_ready_o,
    input  logic                                          go_valid_i,
    input  logic                                          go_final_i,
    input  row_index_t                                    go_row_i,
    input  logic [STREAM_WIDTH-1:0]                       go_codes_i,

    input  logic                                          out_ready_i,
    output logic [ROW_PARALLELISM-1:0]                    out_valid_o,
    output result_t [ROW_PARALLELISM-1:0]                 out_result_o
);

localparam int Groups = (NUM_LANES + N - 1) / N;
localparam int Units = (Groups + 1) / 2;
localparam int Entries = ternip_pkg::group_entries(N);
localparam int IdxBits = $clog2(Entries + 1);
localparam int CodeBits = 1 + IdxBits;
localparam int Stride = 1 << IdxBits;
localparam int Depth = 2 * ROWS * Stride;
localparam int AddrBits = $clog2(Depth);
localparam int OpWidth = ACT_WIDTH + $clog2(N);
localparam int BeatsPerRow = NUM_LANES / IMPORT_WIDTH;
typedef logic signed [OpWidth-1:0] op_t;
typedef logic [IdxBits-1:0] idx_t;
typedef logic [AddrBits-1:0] addr_t;

`ifndef SYNTHESIS
initial begin
    assert (N == 4 || N == 5) else $fatal(1, "GROUP_TRITS must be 4 or 5");
    assert (ROW_PARALLELISM == 1) else $fatal(1, "group tables need ROW_PARALLELISM 1 (TmatmulParallelism <= D)");
    assert (NUM_LANES % IMPORT_WIDTH == 0) else $fatal(1, "NUM_LANES must be a multiple of IMPORT_WIDTH");
    assert (STREAM_WIDTH >= Groups * CodeBits) else $fatal(1, "STREAM_WIDTH too narrow for %0d codes", Groups);
    assert (Arith != ternip_pkg::TDOT_TWOS_ADDER_TREE)
        else $fatal(1, "group tables need a ones'-complement Arith");
end
`endif

// address of index idx of import row row in port p's half of a unit
function automatic addr_t table_addr(int p, row_index_t row, idx_t idx);
    return addr_t'(p * ROWS * Stride) + addr_t'(row) * addr_t'(Stride) + addr_t'(idx);
endfunction

// ---- import position ------------------------------------------------------------------------
logic [`SAFE_CLOG2(BeatsPerRow)-1:0] imp_beat_q;
row_index_t imp_row_q;
wire imp_fire = imp_valid_i && imp_ready_o;
always_ff @(posedge clk_i) begin
    if (!rst_ni || imp_clear_i) begin
        imp_beat_q <= '0;
        imp_row_q <= '0;
    end else if (imp_fire) begin
        imp_beat_q <= imp_beat_q + 1'b1;
        if (imp_beat_q == BeatsPerRow-1) begin
            imp_beat_q <= '0;
            imp_row_q <= imp_row_q + 1'b1;
        end
    end
end

// ---- per-group activations and table walkers -----------------------------------------------
logic [Groups-1:0] walk_busy_q;
logic [Groups-1:0] beat_touches;     // groups with a lane in the current import beat
logic [Groups-1:0] walk_we;
addr_t [Groups-1:0] walk_addr;
op_t [Groups-1:0] walk_data;

for (genvar g_GEN = 0; g_GEN < Groups; g_GEN++) begin : group
    localparam int FirstLane = N * g_GEN;
    localparam int LastLane = (N * g_GEN + N - 1 < NUM_LANES) ? (N * g_GEN + N - 1) : (NUM_LANES - 1);
    localparam int FirstBeat = FirstLane / IMPORT_WIDTH;
    localparam int LastBeat = LastLane / IMPORT_WIDTH;

    act_t [N-1:0] act_q;
    for (genvar k_GEN = 0; k_GEN < N; k_GEN++) begin : lane
        localparam int L = N * g_GEN + k_GEN;
        if (L < NUM_LANES) begin : real_lane
            always_ff @(posedge clk_i)
                if (imp_fire && imp_beat_q == (L / IMPORT_WIDTH)) act_q[k_GEN] <= imp_act_i[L % IMPORT_WIDTH];
        end else begin : pad_lane
            assign act_q[k_GEN] = '0;
        end
    end

    assign beat_touches[g_GEN] = ((FirstBeat == 0) || (int'(imp_beat_q) >= FirstBeat))
                              && ((LastBeat == BeatsPerRow-1) || (int'(imp_beat_q) <= LastBeat));

    // walker: index s (1..Entries) = index s-1 +- act[k]
    idx_t step_q;
    op_t sum_q;
    row_index_t row_q;
    wire logic [3:0] step = ternip_pkg::group_walk_step(N, int'(step_q));
    // explicit sign extension (sv2v flattens act_q; a part-select would zero-extend)
    wire act_t a_k = act_q[step[2:0]];
    wire op_t a = op_t'({{(OpWidth-ACT_WIDTH){a_k[ACT_WIDTH-1]}}, a_k});
    wire op_t next = step[3] ? (sum_q + a) : (sum_q - a);

    always_ff @(posedge clk_i) begin
        if (walk_busy_q[g_GEN]) begin
            sum_q <= next;
            step_q <= step_q + 1'b1;
            if (step_q == idx_t'(Entries)) walk_busy_q[g_GEN] <= 1'b0;
        end
        if (imp_fire && imp_beat_q == LastBeat) begin
            walk_busy_q[g_GEN] <= 1'b1;
            step_q <= idx_t'(1);
            sum_q <= '0;
            row_q <= imp_row_q;
        end
        if (!rst_ni) walk_busy_q[g_GEN] <= 1'b0;
    end

    assign walk_we[g_GEN] = walk_busy_q[g_GEN];
    assign walk_addr[g_GEN] = table_addr(g_GEN % 2, row_q, step_q);
    assign walk_data[g_GEN] = next;
end

assign imp_ready_o = !(|(walk_busy_q & beat_touches));
assign imp_busy_o = |walk_busy_q;

// ---- GO stage 1: codes and row ------------------------------------------------------------
logic go_valid_q, go_final_q;
row_index_t go_row_q;
logic [Groups-1:0][CodeBits-1:0] go_codes_q;

logic acc_in_ready, acc_in_ready_next;
logic go_ready; // clock enable of every GO stage
logic rd_valid_q, rd_final_q;           // stage 2 (RAM output latch)
logic [Groups-1:0] rd_pos_q, rd_neg_q;
op_t [Groups-1:0] rd_q;
logic op_valid, op_final;               // last stage
logic [Groups-1:0] op_pos, op_neg;
op_t [Groups-1:0] op;

wire go_ready_now = acc_in_ready || !op_valid;

if (ternip_pkg::TdotRegisteredReady) begin : g_registered_ready
    wire last_valid_prev = OUT_REG ? rd_valid_q : go_valid_q;
    wire op_valid_next = go_ready_now ? last_valid_prev : op_valid;
    (* max_fanout = 64 *) logic go_ready_q;
    always_ff @(posedge clk_i) begin
        if (!rst_ni) go_ready_q <= 1'b1;
        else         go_ready_q <= acc_in_ready_next || !op_valid_next;
    end
    assign go_ready = go_ready_q;
    assert property (@(posedge clk_i) disable iff (!rst_ni) go_ready_q == go_ready_now)
        else $fatal(0, "registered go_ready disagrees with the accumulator");
end else begin : g_comb_ready
    assign go_ready = go_ready_now;
end
assign go_ready_o = go_ready;

always_ff @(posedge clk_i) begin
    if (!rst_ni) go_valid_q <= 1'b0;
    else if (go_ready) go_valid_q <= go_valid_i;
    if (go_ready) begin
        go_final_q <= go_final_i;
        go_row_q <= go_row_i;
        for (int g = 0; g < Groups; g++) go_codes_q[g] <= go_codes_i[CodeBits*g +: CodeBits];
    end
end

// ---- GO stage 2: RAM read ------------------------------------------------------------------
always_ff @(posedge clk_i) begin
    if (!rst_ni) rd_valid_q <= 1'b0;
    else if (go_ready) rd_valid_q <= go_valid_q;
    if (go_ready) begin
        rd_final_q <= go_final_q;
        for (int g = 0; g < Groups; g++) begin
            rd_pos_q[g] <= (go_codes_q[g][IdxBits-1:0] != '0) && !go_codes_q[g][CodeBits-1];
            rd_neg_q[g] <= (go_codes_q[g][IdxBits-1:0] != '0) && go_codes_q[g][CodeBits-1];
        end
    end
end

// two groups per unit, port p = group 2u+p: write while its walker runs, else read
for (genvar u_GEN = 0; u_GEN < Units; u_GEN++) begin : unit
    localparam int G0 = 2 * u_GEN;
    localparam bit HasB = (2 * u_GEN + 1 < Groups);
    localparam int G1 = HasB ? (2 * u_GEN + 1) : G0;

    wire addr_t addr_a = walk_we[G0] ? walk_addr[G0] : table_addr(0, go_row_q, go_codes_q[G0][IdxBits-1:0]);
    wire addr_t addr_b = walk_we[G1] ? walk_addr[G1] : table_addr(1, go_row_q, go_codes_q[G1][IdxBits-1:0]);
    wire we_b = HasB && walk_we[G1];
    // port enables: UltraRAM inference needs the UG901 no-change form `if (en) if (we) .. else ..`
    wire en_a = walk_we[G0] || go_ready;
    wire en_b = we_b || go_ready;
    op_t rd_a, rd_b;

    // no-change single-clock true dual port (one process per port)
    if (u_GEN < URAM_UNITS) begin : g_uram
        (* ram_style = "ultra" *) op_t mem [Depth];
        /* verilator lint_off MULTIDRIVEN */
        always_ff @(posedge clk_i) if (en_a) begin
            if (walk_we[G0]) mem[addr_a] <= walk_data[G0];
            else rd_a <= mem[addr_a];
        end
        always_ff @(posedge clk_i) if (en_b) begin
            if (we_b) mem[addr_b] <= walk_data[G1];
            else rd_b <= mem[addr_b];
        end
        /* verilator lint_on MULTIDRIVEN */
    end else begin : g_bram
        (* ram_style = "block" *) op_t mem [Depth];
        /* verilator lint_off MULTIDRIVEN */
        always_ff @(posedge clk_i) if (en_a) begin
            if (walk_we[G0]) mem[addr_a] <= walk_data[G0];
            else rd_a <= mem[addr_a];
        end
        always_ff @(posedge clk_i) if (en_b) begin
            if (we_b) mem[addr_b] <= walk_data[G1];
            else rd_b <= mem[addr_b];
        end
        /* verilator lint_on MULTIDRIVEN */
    end
    assign rd_q[G0] = rd_a;
    if (HasB) begin : g_b
        assign rd_q[G1] = rd_b;
    end
end

// ---- GO stage 3: RAM output register (optional) ---------------------------------------------
if (OUT_REG) begin : g_out_reg
    logic op_valid_q, op_final_q;
    logic [Groups-1:0] op_pos_q, op_neg_q;
    op_t [Groups-1:0] op_q;
    always_ff @(posedge clk_i) begin
        if (!rst_ni) op_valid_q <= 1'b0;
        else if (go_ready) op_valid_q <= rd_valid_q;
        if (go_ready) begin
            op_final_q <= rd_final_q;
            op_pos_q <= rd_pos_q;
            op_neg_q <= rd_neg_q;
            op_q <= rd_q;
        end
    end
    assign op_valid = op_valid_q;
    assign op_final = op_final_q;
    assign op_pos = op_pos_q;
    assign op_neg = op_neg_q;
    assign op = op_q;
end else begin : g_no_out_reg
    assign op_valid = rd_valid_q;
    assign op_final = rd_final_q;
    assign op_pos = rd_pos_q;
    assign op_neg = rd_neg_q;
    assign op = rd_q;
end

ternip_tdot_accumulator #(
    .NUM_LANES(Groups),
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
    .in_ready_o(acc_in_ready),
    .in_ready_next_o(acc_in_ready_next),
    .in_valid_i(op_valid),
    .in_final_i(op_final),
    .in_pos_i(op_pos),
    .in_neg_i(op_neg),
    .in_packed_i('0),
    .in_act_i(op),
    .out_ready_i,
    .out_valid_o(out_valid_o[0]),
    .out_result_o(out_result_o[0])
);

endmodule

`undef SAFE_CLOG2
