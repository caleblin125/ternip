// ternip_tmatmul_overlap
//
// ternip_tmatmul with IMPORT and EXPORT running beside GO (ternip_pkg::TmatmulOverlap,
// TERNARY_PAIR4 only). Same ports and command semantics as ternip_tmatmul: every GO
// multiplies the vector of the most recent IMPORT before it, every EXPORT writes the
// result of the most recent GO before it, exactly as if the commands ran one at a time.
//
// Two engines share the unit:
//  - the I/O engine runs IMPORT (vector registers -> pair tables) and EXPORT
//    (exportvector -> vector registers), one at a time, in command order;
//  - the GO engine streams a matrix through the pair tables into an exportvector.
// The pair tables have two banks and there are two exportvectors, so an IMPORT fills the
// bank the running GO is not reading and an EXPORT drains the exportvector the running GO
// is not writing. Each IMPORT takes the bank after the last IMPORT's, each GO the
// exportvector after the last GO's; the waits below only trigger on the third command
// back to the same bank while the first still uses it.
//
// Command acceptance: in_ready_o (any command) while the I/O engine is idle and no GO is
// waiting to start, so the vector-register port stays with one unit at a time (the core
// issues other units' instructions only when every unit is ready). go_in_ready_o (a GO
// only) while no GO is waiting, so the core can issue the next GO while an IMPORT or
// EXPORT still runs. An accepted GO waits in one slot until the GO engine is free, its
// IMPORT is complete and its exportvector is not being exported.
//
// Cost over ternip_tmatmul: the pair tables' LUTRAM doubles (64 deep instead of 32) and a
// second exportvector.

`define SAFE_CLOG2(x) ( (((x)==1) || ((x)==0))? 1 : $clog2((x)))

module ternip_tmatmul_overlap #(
    parameter ternip_pkg::ternip_cfg_t Cfg = `TERNIP_CFG,
    parameter ternip_pkg::tdot_arith_e       Arith      = ternip_pkg::TmatmulArith,
    parameter ternip_pkg::tdot_correction_e  Correction = ternip_pkg::TmatmulCorrection,

    localparam int D                   = Cfg.D,
    localparam int TmatmulParallelism  = Cfg.TmatmulParallelism,
    localparam int FixedPointPrecision = Cfg.FixedPointPrecision,
    localparam int VectorParallelism   = Cfg.VectorParallelism,
    localparam int NumVectorRegisters  = Cfg.NumVectorRegisters,
    localparam int NumChunksPerVector  = Cfg.D / Cfg.VectorParallelism,
    localparam int DdrAddressWidth     = Cfg.DdrAddressWidth,
    localparam int StreamWidth         = ternip_pkg::tmatmul_stream_width(ternip_pkg::TERNARY_PAIR4, TmatmulParallelism),
    localparam int MatrixSizeInBytes   = ((D * D) / TmatmulParallelism) * (StreamWidth / 8),
    localparam type fixed_point_t      = logic signed [FixedPointPrecision-1:0],
    localparam type vector_chunk_t     = fixed_point_t [VectorParallelism-1:0],
    localparam type vector_offset_t    = logic [$clog2(NumChunksPerVector)-1:0],
    localparam type vector_select_t    = logic [$clog2(NumVectorRegisters)-1:0],
    localparam type ddr_address_t      = logic [DdrAddressWidth-1:0]
) (
    input logic                     clk_i,
    input logic                     rst_ni,

    output logic                    in_ready_o,
    output logic                    go_in_ready_o,
    input  logic                    in_valid_i,
    input  ternip_pkg::tmatmul_op_e in_operation_i,
    input  vector_select_t          in_vector_select_i,
    input  ddr_address_t            in_go_matrix_address_i,

    // vector request interface
    input  logic                    vector_request_ready_i,
    output logic                    vector_request_valid_o,
    output logic                    vector_request_write_not_read_o,
    output vector_select_t          vector_request_vector_select_o,
    output vector_offset_t          vector_request_vector_addr_o,
    output vector_chunk_t           vector_request_w_data_o,

    // vector read interface
    output logic                    vector_read_ready_o,
    input  logic                    vector_read_valid_i,
    input  vector_offset_t          vector_read_addr_i,
    input  vector_chunk_t           vector_read_data_i,

    // ddr stream start config
    input  logic                    ddr_stream_ready_i,
    output logic                    ddr_stream_valid_o,
    output ddr_address_t            ddr_stream_address_o,
    output logic [31:0]             ddr_stream_length_o,

    // read data is streamed in sequentially
    output logic                    ddr_r_ready_o,
    input  logic                    ddr_r_valid_i,
    input  logic [StreamWidth-1:0]  ddr_r_data_i
);

localparam int RowParallelism = (TmatmulParallelism < D) ? (1) : (TmatmulParallelism / D);
localparam int DdrReadsPerRow = (TmatmulParallelism > D) ? (1) : (D / TmatmulParallelism);
localparam int DdrReadsPerMatrix = (D*D) / TmatmulParallelism;
localparam int ImportVectorRowWidth = ternip_pkg::min_int(TmatmulParallelism, D);

typedef logic [`SAFE_CLOG2(DdrReadsPerRow)-1:0] row_t;
typedef logic [`SAFE_CLOG2(NumChunksPerVector)-1:0] chunk_addr_t;
typedef logic [$clog2(NumChunksPerVector+1)-1:0] chunk_count_t;

assign ddr_stream_length_o = MatrixSizeInBytes;

// ---- bank bookkeeping (command order) ------------------------------------------------------
logic tbank_last_q; // pair-table bank of the last accepted IMPORT (the bank a new GO reads)
logic ebank_last_q; // exportvector of the last accepted GO (the one a new EXPORT drains)

// ---- I/O engine ------------------------------------------------------------------------------
typedef enum logic [2:0] {
    IO_IDLE,
    IO_IMPORT_WAIT,   // the running GO still reads this IMPORT's bank
    IO_IMPORT,
    IO_EXPORT_WAIT,   // the GO writing this EXPORT's exportvector has not finished
    IO_EXPORT
} io_state_e;
io_state_e io_state_q, io_state_d;
vector_select_t io_vsel_q;
logic io_bank_q; // IMPORT: pair-table bank; EXPORT: exportvector
chunk_count_t io_req_q, io_req_d;       // vector chunks requested (IMPORT) / exportvector reads requested (EXPORT)
chunk_count_t io_chunks_q, io_chunks_d; // IMPORT: chunks handed to the pair tables
logic import_last_q, import_last_d;

// ---- GO engine ----------------------------------------------------------------------------
typedef enum logic [1:0] {
    GO_IDLE,
    GO_WAIT_DDR,
    GO_RUN
} go_state_e;
go_state_e go_state_q, go_state_d;
ddr_address_t go_addr_q;
logic go_tbank_q, go_ebank_q;
logic [`SAFE_CLOG2(DdrReadsPerMatrix):0] go_beats_q, go_beats_d;
row_t go_row_q, go_row_d;
chunk_count_t go_chunks_q, go_chunks_d; // exportvector chunks written

// the accepted GO waiting for the GO engine
logic pend_valid_q;
ddr_address_t pend_addr_q;
logic pend_tbank_q, pend_ebank_q;

wire io_importing = (io_state_q == IO_IMPORT_WAIT) || (io_state_q == IO_IMPORT);
wire io_exporting = (io_state_q == IO_EXPORT_WAIT) || (io_state_q == IO_EXPORT);
wire go_busy = (go_state_q != GO_IDLE);

assign in_ready_o = (io_state_q == IO_IDLE) && !pend_valid_q;
assign go_in_ready_o = !pend_valid_q;

wire accept_import = in_valid_i && in_ready_o && (in_operation_i == ternip_pkg::IMPORT);
wire accept_export = in_valid_i && in_ready_o && (in_operation_i == ternip_pkg::EXPORT);
wire accept_go     = in_valid_i && go_in_ready_o && (in_operation_i == ternip_pkg::GO);

// a waiting GO starts once its IMPORT is in and nothing exports its exportvector
wire go_start = pend_valid_q && !go_busy && !io_importing
             && !(io_exporting && (io_bank_q == pend_ebank_q));

// ---- import skid + pair tables (as in ternip_tmatmul) ---------------------------------------
logic          import_skid_in_ready;
logic          import_skid_in_valid;
logic          import_skid_out_valid;
vector_chunk_t import_skid_out_data;

bsg_two_fifo #(
    .width_p($bits(vector_chunk_t))
) import_skid (
    .clk_i,
    .reset_i(!rst_ni),

    .ready_param_o(import_skid_in_ready),
    .data_i(vector_read_data_i),
    .v_i(import_skid_in_valid),

    .v_o(import_skid_out_valid),
    .data_o(import_skid_out_data),
    .yumi_i(import_skid_out_valid)   // the pair tables take every chunk as it comes
);

logic tlut_imp_busy;
logic tlut_go_ready, tlut_go_valid, tlut_go_final;
logic accumulator_out_ready;
logic [RowParallelism-1:0] accumulator_out_valid;
fixed_point_t [RowParallelism-1:0] accumulator_result;

ternip_tdot_pair_tables #(
    .NUM_LANES(ImportVectorRowWidth),
    .ROW_PARALLELISM(RowParallelism),
    .ROWS(DdrReadsPerRow),
    .ACT_WIDTH(FixedPointPrecision),
    .IMPORT_WIDTH(VectorParallelism),
    .result_t(fixed_point_t),
    .Arith(Arith),
    .Correction(Correction),
    .GPC_K(ternip_pkg::TmatmulGpcK),
    .GPC_FINAL_ROWS(ternip_pkg::TmatmulGpcFinalRows),
    .GPC_ADD(ternip_pkg::TmatmulGpcAdd),
    .D(D),
    .BANKS(2)
) tables (
    .clk_i,
    .rst_ni,
    .imp_clear_i(!io_importing),
    .imp_valid_i(import_skid_out_valid),
    .imp_act_i(import_skid_out_data),
    .imp_bank_i(io_bank_q),
    .imp_busy_o(tlut_imp_busy),
    .go_ready_o(tlut_go_ready),
    .go_valid_i(tlut_go_valid),
    .go_final_i(tlut_go_final),
    .go_bank_i(go_tbank_q),
    .go_row_i(go_row_q),
    .go_codes_i(ddr_r_data_i),
    .out_ready_i(accumulator_out_ready),
    .out_valid_o(accumulator_out_valid),
    .out_result_o(accumulator_result)
);

// ---- export gearbox: results -> vector chunks (GO engine) ----------------------------------
logic gbfifo_export_in_ready;
logic gbfifo_export_out_ready, gbfifo_export_out_valid;
vector_chunk_t gbfifo_export_out_data;

ternip_gearbox_fifo #(
    .InDataWidth($bits(accumulator_result)),
    .OutDataWidth($bits(vector_chunk_t))
) gbfifo_export (
    .clk_i,
    .rst_ni,

    .in_ready_o(gbfifo_export_in_ready),
    .in_valid_i(accumulator_out_valid[0]),
    .in_data_i(accumulator_result),

    .out_ready_i(gbfifo_export_out_ready),
    .out_valid_o(gbfifo_export_out_valid),
    .out_data_o(gbfifo_export_out_data)
);
assign accumulator_out_ready = gbfifo_export_in_ready;

// ---- two exportvectors: GO writes go_ebank_q, EXPORT reads io_bank_q -------------------------
logic        [1:0] ev_request_ready, ev_request_valid, ev_request_write_not_read;
chunk_addr_t [1:0] ev_request_addr;
logic        [1:0] ev_read_ready, ev_read_valid;
chunk_addr_t [1:0] ev_read_addr;
vector_chunk_t [1:0] ev_read_data;

for (genvar b_GEN = 0; b_GEN < 2; b_GEN++) begin : exportvector
    ternip_pipelined_mem #(
        .DATA_WIDTH($bits(vector_chunk_t)),
        .NUM_ENTRIES(NumChunksPerVector)
    ) mem (
        .clk_i,
        .rst_ni,

        .request_ready_o(ev_request_ready[b_GEN]),
        .request_valid_i(ev_request_valid[b_GEN]),
        .request_write_not_read_i(ev_request_write_not_read[b_GEN]),
        .request_addr_i(ev_request_addr[b_GEN]),
        .request_w_data_i(gbfifo_export_out_data),

        .read_ready_i(ev_read_ready[b_GEN]),
        .read_valid_o(ev_read_valid[b_GEN]),
        .read_addr_o(ev_read_addr[b_GEN]),
        .read_data_o(ev_read_data[b_GEN])
    );
end

// ---- control ---------------------------------------------------------------------------------
wire go_writes_ev  = (go_state_q == GO_RUN) && gbfifo_export_out_valid;
wire io_reads_ev   = (io_state_q == IO_EXPORT) && (io_req_q < NumChunksPerVector);

always_comb begin
    io_state_d = io_state_q;
    io_req_d = io_req_q;
    io_chunks_d = io_chunks_q;
    import_last_d = import_last_q;

    go_state_d = go_state_q;
    go_beats_d = go_beats_q;
    go_row_d = go_row_q;
    go_chunks_d = go_chunks_q;

    vector_request_valid_o = 0;
    vector_request_write_not_read_o = 'x;
    vector_request_vector_select_o = 'x;
    vector_request_vector_addr_o = 'x;
    vector_request_w_data_o = 'x;
    vector_read_ready_o = 0;
    import_skid_in_valid = 0;

    ddr_stream_valid_o = 0;
    ddr_stream_address_o = 'x;
    ddr_r_ready_o = 0;
    tlut_go_valid = 0;
    tlut_go_final = 0;
    gbfifo_export_out_ready = 0;

    ev_request_valid = '0;
    ev_request_write_not_read = 'x;
    ev_request_addr = 'x;
    ev_read_ready = '0;

    // -- I/O engine
    unique case (io_state_q)
        IO_IDLE: begin
            io_req_d = '0;
            io_chunks_d = '0;
            if (accept_import) io_state_d = IO_IMPORT_WAIT;
            if (accept_export) io_state_d = IO_EXPORT_WAIT;
        end
        IO_IMPORT_WAIT: begin
            if (!(go_busy && (go_tbank_q == io_bank_q))) io_state_d = IO_IMPORT;
        end
        IO_IMPORT: begin
            if (io_req_q < NumChunksPerVector) begin
                vector_request_valid_o = 1;
                vector_request_write_not_read_o = 0;
                vector_request_vector_select_o = io_vsel_q;
                vector_request_vector_addr_o = vector_offset_t'(io_req_q);
                if (vector_request_ready_i) io_req_d = io_req_q + 1'b1;
            end
            vector_read_ready_o = import_skid_in_ready;
            import_skid_in_valid = vector_read_valid_i;
            if (import_skid_out_valid) begin
                io_chunks_d = io_chunks_q + 1'b1;
                if (io_chunks_q >= NumChunksPerVector-1) import_last_d = 1;
            end
            // done once the last chunk is in and the table write buses are idle
            if (import_last_q && !tlut_imp_busy) begin
                import_last_d = 0;
                io_state_d = IO_IDLE;
            end
        end
        IO_EXPORT_WAIT: begin
            if (!((go_busy && (go_ebank_q == io_bank_q)) || (pend_valid_q && (pend_ebank_q == io_bank_q))))
                io_state_d = IO_EXPORT;
        end
        IO_EXPORT: begin
            if (io_reads_ev) begin
                ev_request_valid[io_bank_q] = 1;
                ev_request_write_not_read[io_bank_q] = 0;
                ev_request_addr[io_bank_q] = chunk_addr_t'(io_req_q);
                if (ev_request_ready[io_bank_q]) io_req_d = io_req_q + 1'b1;
            end
            ev_read_ready[io_bank_q] = vector_request_ready_i;
            if (vector_request_ready_i && ev_read_valid[io_bank_q]) begin
                vector_request_valid_o = 1;
                vector_request_write_not_read_o = 1;
                vector_request_vector_select_o = io_vsel_q;
                vector_request_vector_addr_o = vector_offset_t'(ev_read_addr[io_bank_q]);
                vector_request_w_data_o = ev_read_data[io_bank_q];
                if (ev_read_addr[io_bank_q] >= NumChunksPerVector-1) io_state_d = IO_IDLE;
            end
        end
        default: ;
    endcase

    // -- GO engine
    unique case (go_state_q)
        GO_IDLE: begin
            go_beats_d = '0;
            go_row_d = '0;
            go_chunks_d = '0;
            if (go_start) go_state_d = GO_WAIT_DDR;
        end
        GO_WAIT_DDR: begin
            ddr_stream_valid_o = 1;
            ddr_stream_address_o = go_addr_q;
            if (ddr_stream_ready_i) go_state_d = GO_RUN;
        end
        GO_RUN: begin
            // one DDR beat of pair codes per table lookup, rows in import order
            if (go_beats_q < DdrReadsPerMatrix) begin
                ddr_r_ready_o = tlut_go_ready;
                if (ddr_r_valid_i && tlut_go_ready) begin
                    tlut_go_valid = 1;
                    tlut_go_final = (go_row_q == row_t'(DdrReadsPerRow-1));
                    go_beats_d = go_beats_q + 1'b1;
                    go_row_d = (go_row_q == row_t'(DdrReadsPerRow-1)) ? row_t'(0) : row_t'(go_row_q + 1'b1);
                end
            end
            // results into this GO's exportvector
            gbfifo_export_out_ready = ev_request_ready[go_ebank_q];
            if (go_writes_ev) begin
                ev_request_valid[go_ebank_q] = 1;
                ev_request_write_not_read[go_ebank_q] = 1;
                ev_request_addr[go_ebank_q] = chunk_addr_t'(go_chunks_q);
                if (ev_request_ready[go_ebank_q]) begin
                    go_chunks_d = go_chunks_q + 1'b1;
                    if (go_chunks_q >= NumChunksPerVector-1) go_state_d = GO_IDLE;
                end
            end
        end
        default: ;
    endcase
end

always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
        io_state_q <= IO_IDLE;
        go_state_q <= GO_IDLE;
        pend_valid_q <= 1'b0;
        import_last_q <= 1'b0;
        tbank_last_q <= 1'b0;
        ebank_last_q <= 1'b0;
    end else begin
        io_state_q <= io_state_d;
        go_state_q <= go_state_d;
        import_last_q <= import_last_d;

        if (accept_import) tbank_last_q <= !tbank_last_q;
        if (accept_go) ebank_last_q <= !ebank_last_q;

        if (accept_go) pend_valid_q <= 1'b1;
        else if (go_start) pend_valid_q <= 1'b0;
    end
end

always_ff @(posedge clk_i) begin
    io_req_q <= io_req_d;
    io_chunks_q <= io_chunks_d;
    go_beats_q <= go_beats_d;
    go_row_q <= go_row_d;
    go_chunks_q <= go_chunks_d;

    if (accept_import) begin
        io_vsel_q <= in_vector_select_i;
        io_bank_q <= !tbank_last_q;
    end
    if (accept_export) begin
        io_vsel_q <= in_vector_select_i;
        io_bank_q <= ebank_last_q;
    end
    if (accept_go) begin
        pend_addr_q <= in_go_matrix_address_i;
        pend_tbank_q <= tbank_last_q;
        pend_ebank_q <= !ebank_last_q;
    end
    if (go_start) begin
        go_addr_q <= pend_addr_q;
        go_tbank_q <= pend_tbank_q;
        go_ebank_q <= pend_ebank_q;
    end
end

`ifndef SYNTHESIS
always @(posedge clk_i) if (rst_ni) begin
    // accept_go and go_start together would need two waiting slots
    assert (!(accept_go && pend_valid_q)) else $fatal(1, "tmatmul overlap: GO accepted while one waits");
    // the I/O engine and the GO engine never share a bank
    assert (!((io_state_q == IO_IMPORT) && go_busy && (go_tbank_q == io_bank_q)))
        else $fatal(1, "tmatmul overlap: IMPORT writes the bank GO reads");
    assert (!((io_state_q == IO_EXPORT) && go_busy && (go_ebank_q == io_bank_q)))
        else $fatal(1, "tmatmul overlap: EXPORT reads the exportvector GO writes");
end
`endif

endmodule

`undef SAFE_CLOG2
