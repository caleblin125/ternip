// Copyright (c) 2026 Ethan Sifferman
//
// Redistribution and use in source and binary forms, with or without modification, are permitted
// provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
//    conditions and the following disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to
//    endorse or promote products derived from this software without specific prior written
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// ternip_tmatmul
//
// Functional unit for ternary matrix/vector multiplication.
//
// Ternary matmul uses three required commands:
//
// IMPORT loads the input vector into the internal importvector.
// GO streams the ternary matrix from DDR, multiplies it by importvector, and
// stores the result in the internal exportvector.
// EXPORT writes exportvector to the selected vector register.
//
// The unit accepts one command on in_* and then drives the vector-register and
// DDR read interfaces until that command finishes. Use the commands in order:
// IMPORT, then GO, then EXPORT.
//
// Encoding selects the DDR weight format (see ternip_pkg::ternary_encoding_e),
// Decoder the base-3 unpacker for TERNARY_PACKED5, and Arith/Correction the
// dot-product datapath (see ternip_tdot_accumulator). The defaults from
// ternip_pkg (TERNARY_2BIT, TDOT_TWOS_ADDER_TREE) are the original datapath.

`define SAFE_CLOG2(x) ( (((x)==1) || ((x)==0))? 1 : $clog2((x)))

module ternip_tmatmul #(
    parameter ternip_pkg::ternip_cfg_t Cfg = `TERNIP_CFG,
    parameter ternip_pkg::ternary_encoding_e Encoding   = ternip_pkg::TmatmulEncoding,
    parameter ternip_pkg::trit5_decoder_e    Decoder    = ternip_pkg::TmatmulDecoder,
    parameter ternip_pkg::tdot_arith_e       Arith      = ternip_pkg::TmatmulArith,
    parameter ternip_pkg::tdot_correction_e  Correction = ternip_pkg::TmatmulCorrection,

    localparam int D                   = Cfg.D,
    localparam int TmatmulParallelism  = Cfg.TmatmulParallelism,
    localparam int FixedPointPrecision = Cfg.FixedPointPrecision,
    localparam int VectorParallelism   = Cfg.VectorParallelism,
    localparam int NumVectorRegisters  = Cfg.NumVectorRegisters,
    localparam int NumChunksPerVector  = Cfg.D / Cfg.VectorParallelism,
    localparam int DdrAddressWidth     = Cfg.DdrAddressWidth,
    localparam int StreamWidth         = ternip_pkg::tmatmul_stream_width(Encoding, TmatmulParallelism),
    localparam int MatrixSizeInBytes   = ((D * D) / TmatmulParallelism) * (StreamWidth / 8),
    localparam type fixed_point_t      = logic signed [FixedPointPrecision-1:0],
    localparam type vector_chunk_t     = fixed_point_t [VectorParallelism-1:0],
    localparam type ternary_t          = logic signed [1:0],
    localparam type vector_offset_t    = logic [$clog2(NumChunksPerVector)-1:0],
    localparam type vector_select_t    = logic [$clog2(NumVectorRegisters)-1:0],
    localparam type ddr_address_t      = logic [DdrAddressWidth-1:0]
) (
    input logic                     clk_i,
    input logic                     rst_ni,

    output logic                    in_ready_o,
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
    output logic                              ddr_r_ready_o,
    input  logic                              ddr_r_valid_i,
    input  logic [StreamWidth-1:0]            ddr_r_data_i
);

localparam int RowParallelism = (TmatmulParallelism < D) ? (1) : (TmatmulParallelism / D);
localparam int DdrReadsPerRow = (TmatmulParallelism > D) ? (1) : (D / TmatmulParallelism);
localparam int ImportVectorRowWidth = ternip_pkg::min_int(TmatmulParallelism, D);

`ifndef SYNTHESIS
initial begin
    if (TmatmulParallelism < D) begin
        assert(DdrReadsPerRow * TmatmulParallelism == D) else begin
            $fatal("TmatmulParallelism (%0d) must divide evenly into D (%0d)", TmatmulParallelism, D);
        end
    end else begin
        assert(RowParallelism * D == TmatmulParallelism) else begin
            $fatal("D (%0d) must divide evenly into TmatmulParallelism (%0d)", D, TmatmulParallelism);
        end
    end
end
`endif

typedef logic signed [FixedPointPrecision:0] tmul_result_t;

function automatic tmul_result_t ternary_mul(ternary_t ternary, fixed_point_t fixed_point);
    unique case (ternary)
        0: return '0;
        1: return fixed_point;
        -1: return -fixed_point;
        default: return 'x;
    endcase
endfunction

enum logic [1:0] {
    WAITING_FOR_IN,
    WAITING_FOR_DDR_STREAM_READY,
    WORKING
} state_d, state_q = WAITING_FOR_IN; // for assertions at time=0

ternip_pkg::tmatmul_op_e tmatmul_operation_d, tmatmul_operation_q;
// =========================================================================
// Manual forward retiming of state_q and tmatmul_operation_q to avoid high fanout in the FSM.
// The flags are set directly by each transition in the FSM below (not decoded from
// state_d / tmatmul_operation_d), so each flag's next value only depends on the inputs of
// the transitions that can change it. Decoding pulled the shared weight-stream ready
// (ddr_stream_ready_i, from another core) into in_state_working_import_d, which only a
// dispatch from WAITING_FOR_IN can set: the 300 MHz critical path at 156 cores.
logic in_state_working_import_d, in_state_working_import_q;
logic in_state_working_go_d, in_state_working_go_q;
logic in_state_working_export_d, in_state_working_export_q;
logic in_state_waiting_for_in_d, in_state_waiting_for_in_q;
logic in_state_waiting_for_ddr_go_d, in_state_waiting_for_ddr_go_q;

// IMPORT ends one cycle after its last importvector write, so the import gearbox's
// all-lanes-full valid only reaches this register, not the replicated flags.
logic import_last_d, import_last_q;

`ifndef SYNTHESIS
always @(posedge clk_i) if (rst_ni) begin
    assert ({in_state_working_import_q, in_state_working_go_q, in_state_working_export_q,
             in_state_waiting_for_in_q, in_state_waiting_for_ddr_go_q} ==
            {(state_q == WORKING) && (tmatmul_operation_q == ternip_pkg::IMPORT),
             (state_q == WORKING) && (tmatmul_operation_q == ternip_pkg::GO),
             (state_q == WORKING) && (tmatmul_operation_q == ternip_pkg::EXPORT),
             (state_q == WAITING_FOR_IN),
             (state_q == WAITING_FOR_DDR_STREAM_READY) && (tmatmul_operation_q == ternip_pkg::GO)})
        else $fatal(1, "tmatmul state flags disagree with state_q / tmatmul_operation_q");
end
`endif

always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
        in_state_working_import_q     <= '0;
        in_state_working_go_q         <= '0;
        in_state_working_export_q     <= '0;
        in_state_waiting_for_in_q     <= '1;
        in_state_waiting_for_ddr_go_q <= '0;
        import_last_q                 <= '0;
    end else begin
        in_state_working_import_q     <= in_state_working_import_d;
        in_state_working_go_q         <= in_state_working_go_d;
        in_state_working_export_q     <= in_state_working_export_d;
        in_state_waiting_for_in_q     <= in_state_waiting_for_in_d;
        in_state_waiting_for_ddr_go_q <= in_state_waiting_for_ddr_go_d;
        import_last_q                 <= import_last_d;
    end
end
// =========================================================================

vector_select_t vector_select_d, vector_select_q;

localparam int DdrReadsPerMatrix = (D*D) / TmatmulParallelism;
assign ddr_stream_length_o = MatrixSizeInBytes;

logic [`SAFE_CLOG2(DdrReadsPerMatrix):0] importvector_counter_d, importvector_counter_q;
logic [`SAFE_CLOG2(DdrReadsPerMatrix):0] exportvector_counter_d, exportvector_counter_q;
ddr_address_t ddr_stream_address_d, ddr_stream_address_q;
logic [`SAFE_CLOG2(DdrReadsPerRow):0] importvector_request_addr_counter_d, importvector_request_addr_counter_q;

// AXI stream adapter for IMPORT: vector_chunk_t -> fixed_point_t
logic gbfifo_import_in_ready;
logic gbfifo_import_in_valid;
vector_chunk_t gbfifo_import_in_data;

logic gbfifo_import_out_ready;
logic gbfifo_import_out_valid;
fixed_point_t [ImportVectorRowWidth-1:0] gbfifo_import_out_data;

// AXI stream adapter for GO (export): fixed_point_t -> vector_chunk_t
logic gbfifo_export_in_ready;
logic gbfifo_export_in_valid;
fixed_point_t [RowParallelism-1:0] gbfifo_export_in_data;

logic gbfifo_export_out_ready;
logic gbfifo_export_out_valid;
vector_chunk_t gbfifo_export_out_data;

// IMPORT BRAM write tracking
logic [`SAFE_CLOG2(DdrReadsPerRow):0] import_bram_addr_d, import_bram_addr_q;

// Queued Instruction
logic queued_valid_d, queued_valid_q;
ternip_pkg::tmatmul_op_e queued_operation_d, queued_operation_q;
vector_select_t queued_vector_select_d, queued_vector_select_q;
ddr_address_t queued_go_matrix_address_d, queued_go_matrix_address_q;

// Weight stream decode: one +1/-1 flag pair per lane, except TRIT5_DECODE_DIRECT,
// where each row datapath builds its operands from the packed bytes.
localparam bit DirectPacked = (Encoding == ternip_pkg::TERNARY_PACKED5)
                           && (Decoder == ternip_pkg::TRIT5_DECODE_DIRECT);
localparam int AccumulatorPackedBytes = DirectPacked ? (StreamWidth / 8) : 0;
localparam int AccumulatorPackedWidth = DirectPacked ? StreamWidth : 1;

logic [TmatmulParallelism-1:0] weight_pos, weight_neg;
ternary_t [TmatmulParallelism-1:0] ddr_r_trits;
logic [AccumulatorPackedWidth-1:0] accumulator_packed;

`ifndef SYNTHESIS
if (DirectPacked && (Arith == ternip_pkg::TDOT_TWOS_ADDER_TREE))
    $fatal(0, "TRIT5_DECODE_DIRECT needs a ones'-complement Arith");
`endif

if (DirectPacked) begin : packed5_direct
    assign accumulator_packed = ddr_r_data_i;
    assign weight_pos = '0;
    assign weight_neg = '0;
    assign ddr_r_trits = '0;
end else if (Encoding == ternip_pkg::TERNARY_PACKED5) begin : packed5
    assign accumulator_packed = '0;
    localparam int NumBytes = StreamWidth / 8;
    logic [5*NumBytes-1:0] unpacked_pos, unpacked_neg;
    for (genvar b_GEN = 0; b_GEN < NumBytes; b_GEN++) begin : byte_decode
        ternip_trit5_decode #(
            .Decoder(Decoder)
        ) decode (
            .packed_i(ddr_r_data_i[8*b_GEN +: 8]),
            .pos_o(unpacked_pos[5*b_GEN +: 5]),
            .neg_o(unpacked_neg[5*b_GEN +: 5]),
            .invalid_o()
        );
    end
    // trits beyond TmatmulParallelism in the last byte are padding
    assign weight_pos = unpacked_pos[TmatmulParallelism-1:0];
    assign weight_neg = unpacked_neg[TmatmulParallelism-1:0];
    for (genvar i_GEN = 0; i_GEN < TmatmulParallelism; i_GEN++) begin : lane
        assign ddr_r_trits[i_GEN] = weight_pos[i_GEN] ? ternary_t'(1) : (weight_neg[i_GEN] ? ternary_t'(-1) : ternary_t'(0));
    end
end else begin : trit2
    assign accumulator_packed = '0;
    assign ddr_r_trits = ddr_r_data_i;
    for (genvar i_GEN = 0; i_GEN < TmatmulParallelism; i_GEN++) begin : lane
        assign weight_pos[i_GEN] = (ddr_r_trits[i_GEN] == ternary_t'(1));
        assign weight_neg[i_GEN] = (ddr_r_trits[i_GEN] == ternary_t'(-1));
    end
end

// Multioperand Accumulator
logic accumulator_in_valid, accumulator_in_final, accumulator_out_ready;
tmul_result_t [RowParallelism-1:0][ImportVectorRowWidth-1:0] accumulator_operands;
logic [RowParallelism-1:0] accumulator_in_ready, accumulator_out_valid;
fixed_point_t [RowParallelism-1:0] accumulator_result;
fixed_point_t [ImportVectorRowWidth-1:0] importvector_read_data;

for (genvar i_GEN = 0; i_GEN < RowParallelism; i_GEN++) begin : row

    if (Arith == ternip_pkg::TDOT_TWOS_ADDER_TREE) begin : twos
        ternip_multioperand_accumulator #(
            .Cfg(Cfg),
            .operand_t(tmul_result_t),
            .result_t(fixed_point_t),
            .NUM_OPERANDS(ImportVectorRowWidth),
            .NEXT_STAGE_FANIN(2),
            .D(D)
        ) multioperand_accumulator (
            .clk_i,
            .rst_ni,
            .in_ready_o(accumulator_in_ready[i_GEN]),
            .in_valid_i(accumulator_in_valid),
            .in_final_i(accumulator_in_final),
            .in_operands_i(accumulator_operands[i_GEN]),
            .out_ready_i(accumulator_out_ready),
            .out_valid_o(accumulator_out_valid[i_GEN]),
            .out_result_o(accumulator_result[i_GEN])
        );
    end else begin : ones
        ternip_tdot_accumulator #(
            .NUM_LANES(ImportVectorRowWidth),
            .ACT_WIDTH(FixedPointPrecision),
            .result_t(fixed_point_t),
            .Arith(Arith),
            .Correction(Correction),
            .GPC_K(ternip_pkg::TmatmulGpcK),
            .GPC_FINAL_ROWS(ternip_pkg::TmatmulGpcFinalRows),
            .GPC_ADD(ternip_pkg::TmatmulGpcAdd),
            .D(D),
            .PACKED_BYTES(AccumulatorPackedBytes),
            .LANE_OFFSET(i_GEN*ImportVectorRowWidth)
        ) tdot_accumulator (
            .clk_i,
            .rst_ni,
            .in_ready_o(accumulator_in_ready[i_GEN]),
            .in_valid_i(accumulator_in_valid),
            .in_final_i(accumulator_in_final),
            .in_pos_i(weight_pos[i_GEN*ImportVectorRowWidth +: ImportVectorRowWidth]),
            .in_neg_i(weight_neg[i_GEN*ImportVectorRowWidth +: ImportVectorRowWidth]),
            .in_packed_i(accumulator_packed),
            .in_act_i(importvector_read_data),
            .out_ready_i(accumulator_out_ready),
            .out_valid_o(accumulator_out_valid[i_GEN]),
            .out_result_o(accumulator_result[i_GEN])
        );
    end

end

// Import Vector
logic importvector_request_ready;
logic importvector_request_valid;
logic importvector_request_write_not_read;
logic [`SAFE_CLOG2(DdrReadsPerRow)-1:0] importvector_request_addr;
fixed_point_t [ImportVectorRowWidth-1:0] importvector_request_w_data;

logic importvector_read_ready;
logic importvector_read_valid;
logic [`SAFE_CLOG2(DdrReadsPerRow)-1:0] importvector_read_addr;

ternip_pipelined_mem #(
    .DATA_WIDTH($bits(importvector_read_data)),
    .NUM_ENTRIES(DdrReadsPerRow)
) importvector (
    .clk_i,
    .rst_ni,

    .request_ready_o(importvector_request_ready),
    .request_valid_i(importvector_request_valid),
    .request_write_not_read_i(importvector_request_write_not_read),
    .request_addr_i(importvector_request_addr),
    .request_w_data_i(importvector_request_w_data),

    .read_ready_i(importvector_read_ready),
    .read_valid_o(importvector_read_valid),
    .read_addr_o(importvector_read_addr),
    .read_data_o(importvector_read_data)
);

// Export Vector
logic exportvector_request_ready;
logic exportvector_request_valid;
logic exportvector_request_write_not_read;
logic [`SAFE_CLOG2(NumChunksPerVector)-1:0] exportvector_request_addr;
vector_chunk_t exportvector_request_w_data;

logic exportvector_read_ready;
logic exportvector_read_valid;
logic [`SAFE_CLOG2(NumChunksPerVector)-1:0] exportvector_read_addr;
vector_chunk_t exportvector_read_data;

ternip_pipelined_mem #(
    .DATA_WIDTH($bits(exportvector_read_data)),
    .NUM_ENTRIES(NumChunksPerVector)
) exportvector (
    .clk_i,
    .rst_ni,

    .request_ready_o(exportvector_request_ready),
    .request_valid_i(exportvector_request_valid),
    .request_write_not_read_i(exportvector_request_write_not_read),
    .request_addr_i(exportvector_request_addr),
    .request_w_data_i(exportvector_request_w_data),

    .read_ready_i(exportvector_read_ready),
    .read_valid_o(exportvector_read_valid),
    .read_addr_o(exportvector_read_addr),
    .read_data_o(exportvector_read_data)
);

// IMPORT skid: a two-entry FIFO in front of the import gearbox, so
// vector_read_ready_o is a register (not full) instead of the gearbox's
// round-robin fullness logic, which reached the vector-register read buffer's
// clock enable through the core's ready OR (the 300 MHz critical path).
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
    .yumi_i(import_skid_out_valid && gbfifo_import_in_ready)
);

assign gbfifo_import_in_valid = import_skid_out_valid;
assign gbfifo_import_in_data = import_skid_out_data;

// IMPORT gearbox_fifo: vector_chunk_t -> fixed_point_t [TmatmulParallelism-1:0]
ternip_gearbox_fifo #(
    .InDataWidth($bits(gbfifo_import_in_data)),
    .OutDataWidth($bits(gbfifo_import_out_data))
) gbfifo_import (
    .clk_i,
    .rst_ni,

    .in_ready_o(gbfifo_import_in_ready),
    .in_valid_i(gbfifo_import_in_valid),
    .in_data_i(gbfifo_import_in_data),

    .out_ready_i(gbfifo_import_out_ready),
    .out_valid_o(gbfifo_import_out_valid),
    .out_data_o(gbfifo_import_out_data)
);

// EXPORT (GO) gearbox_fifo: fixed_point_t -> vector_chunk_t
ternip_gearbox_fifo #(
    .InDataWidth($bits(gbfifo_export_in_data)),
    .OutDataWidth($bits(gbfifo_export_out_data))
) gbfifo_export (
    .clk_i,
    .rst_ni,

    .in_ready_o(gbfifo_export_in_ready),
    .in_valid_i(gbfifo_export_in_valid),
    .in_data_i(gbfifo_export_in_data),

    .out_ready_i(gbfifo_export_out_ready),
    .out_valid_o(gbfifo_export_out_valid),
    .out_data_o(gbfifo_export_out_data)
);

always_comb begin
    state_d = state_q;
    in_ready_o = 0;

    in_state_working_import_d     = in_state_working_import_q;
    in_state_working_go_d         = in_state_working_go_q;
    in_state_working_export_d     = in_state_working_export_q;
    in_state_waiting_for_in_d     = in_state_waiting_for_in_q;
    in_state_waiting_for_ddr_go_d = in_state_waiting_for_ddr_go_q;
    import_last_d                 = import_last_q;

    queued_valid_d = queued_valid_q;
    queued_operation_d = queued_operation_q;
    queued_vector_select_d = queued_vector_select_q;
    queued_go_matrix_address_d = queued_go_matrix_address_q;

    tmatmul_operation_d = tmatmul_operation_q;
    exportvector_counter_d = exportvector_counter_q;
    ddr_stream_address_d = ddr_stream_address_q;
    importvector_request_addr_counter_d = importvector_request_addr_counter_q;

    importvector_counter_d = importvector_counter_q;
    vector_select_d = vector_select_q;
    import_bram_addr_d = import_bram_addr_q;

    vector_request_valid_o = 0;
    vector_request_write_not_read_o = 'x;
    vector_request_vector_select_o = 'x;
    vector_request_vector_addr_o = 'x;
    vector_request_w_data_o = 'x;
    vector_read_ready_o = 0;

    ddr_stream_address_o = 'x;

    ddr_stream_valid_o = 0;
    ddr_r_ready_o = 0;

    accumulator_in_valid = 0;
    accumulator_in_final = 0;
    accumulator_operands = 'x;
    accumulator_out_ready = 0;

    importvector_request_valid = '0;
    importvector_request_write_not_read = 'x;
    importvector_request_addr = 'x;
    importvector_request_w_data = 'x;
    importvector_read_ready = '0;

    exportvector_request_valid = 0;
    exportvector_request_write_not_read = 'x;
    exportvector_request_addr = 'x;
    exportvector_request_w_data = 'x;
    exportvector_read_ready = 0;

    import_skid_in_valid = 0;
    gbfifo_import_out_ready = 0;

    gbfifo_export_in_valid = 0;
    gbfifo_export_in_data = 'x;
    gbfifo_export_out_ready = 0;

    // if (state_q == WAITING_FOR_IN) begin // manual forward retiming
    if (in_state_waiting_for_in_q) begin

        in_ready_o = !queued_valid_q;

        importvector_counter_d = 0;
        exportvector_counter_d = 0;
        importvector_request_addr_counter_d = 0;
        import_bram_addr_d = 0;

        if (queued_valid_q) begin
            queued_valid_d = 0;
            queued_operation_d = ternip_pkg::NO_TMATMUL_OP;
            queued_vector_select_d = 'x;
            queued_go_matrix_address_d = 'x;

            tmatmul_operation_d = queued_operation_q;
            vector_select_d = queued_vector_select_q;
            ddr_stream_address_d = queued_go_matrix_address_q;

            state_d = (tmatmul_operation_d == ternip_pkg::GO) ? (WAITING_FOR_DDR_STREAM_READY) : (WORKING);

        end else if (in_valid_i) begin
            tmatmul_operation_d = in_operation_i;
            vector_select_d = in_vector_select_i;
            ddr_stream_address_d = in_go_matrix_address_i;

            state_d = (tmatmul_operation_d == ternip_pkg::GO) ? (WAITING_FOR_DDR_STREAM_READY) : (WORKING);
        end

        if (queued_valid_q || in_valid_i) begin
            in_state_waiting_for_in_d     = 0;
            in_state_waiting_for_ddr_go_d = (tmatmul_operation_d == ternip_pkg::GO);
            in_state_working_import_d     = (tmatmul_operation_d == ternip_pkg::IMPORT);
            in_state_working_export_d     = (tmatmul_operation_d == ternip_pkg::EXPORT);
        end

    // end else if ((state_q == WAITING_FOR_DDR_STREAM_READY) && (tmatmul_operation_q == ternip_pkg::GO)) begin // manual forward retiming
    end else if (in_state_waiting_for_ddr_go_q) begin

        ddr_stream_valid_o = 1;
        if (ddr_stream_ready_i) begin
            ddr_stream_address_o = ddr_stream_address_q;
            state_d = WORKING;
            in_state_waiting_for_ddr_go_d = 0;
            in_state_working_go_d = 1;
        end

    // end else if ((state_q == WORKING) && (tmatmul_operation_q == ternip_pkg::IMPORT)) begin // manual forward retiming
    end if (in_state_working_import_q) begin
        // Request chunks from vector registers
        if (importvector_counter_q < NumChunksPerVector) begin
            vector_request_valid_o = 1;
            vector_request_write_not_read_o = 0;
            vector_request_vector_select_o = vector_select_q;
            vector_request_vector_addr_o = importvector_counter_q;
            if (vector_request_ready_i) importvector_counter_d++;
        end

        // Feed vector chunks into adapter
        vector_read_ready_o = import_skid_in_ready;
        import_skid_in_valid = vector_read_valid_i;

        // Receive deserialized elements from adapter and write to BRAMs
        gbfifo_import_out_ready = importvector_request_ready && !import_last_q;
        if (gbfifo_import_out_ready && gbfifo_import_out_valid) begin
            importvector_request_valid = 1;
            importvector_request_write_not_read = 1;
            importvector_request_addr = import_bram_addr_q;
            importvector_request_w_data = gbfifo_import_out_data;

            import_bram_addr_d++;
            if (import_bram_addr_q >= DdrReadsPerRow-1) import_last_d = 1;
        end

        if (import_last_q) begin
            import_last_d = 0;
            state_d = WAITING_FOR_IN;
            tmatmul_operation_d = ternip_pkg::NO_TMATMUL_OP;
            in_state_working_import_d = 0;
            in_state_waiting_for_in_d = 1;
        end

    // end else if ((state_q == WORKING) && (tmatmul_operation_q == ternip_pkg::GO)) begin // manual forward retiming
    end else if (in_state_working_go_q) begin

        // allow instruction queueing
        in_ready_o = !queued_valid_q;
        if (in_ready_o && in_valid_i) begin
            queued_operation_d = in_operation_i;
            queued_go_matrix_address_d = in_go_matrix_address_i;
            queued_vector_select_d = in_vector_select_i;
            queued_valid_d = 1;
        end

        // Request read from importvector
        if (importvector_counter_q < DdrReadsPerMatrix) begin
            importvector_request_valid = '1;
            importvector_request_write_not_read = '0;
            importvector_request_addr = importvector_request_addr_counter_q;
            if (importvector_request_ready) begin
                importvector_counter_d++;
                importvector_request_addr_counter_d++;
                if (importvector_request_addr_counter_q == DdrReadsPerRow-1) importvector_request_addr_counter_d = 0;
            end
        end

        // Receive read response from importvector, and fetch from ddr
        // add product to accumulator
        if (importvector_read_valid) begin
            ddr_r_ready_o = accumulator_in_ready[0];
            if (ddr_r_ready_o && ddr_r_valid_i) begin
                importvector_read_ready = '1;

                accumulator_in_valid = 1;
                for (int i = 0; i < TmatmulParallelism; i++) begin
                    accumulator_operands[i / D][i % D] = ternary_mul(ddr_r_trits[i], importvector_read_data[i % D]);
                end
                if (importvector_read_addr >= DdrReadsPerRow-1) begin
                    accumulator_in_final = 1;
                end
            end
        end

        // Feed accumulator results into adapter
        accumulator_out_ready = gbfifo_export_in_ready;
        gbfifo_export_in_valid = accumulator_out_valid[0];
        gbfifo_export_in_data = accumulator_result;

        // Write serialized chunks to exportvector
        gbfifo_export_out_ready = exportvector_request_ready;
        if (gbfifo_export_out_ready && gbfifo_export_out_valid) begin
            exportvector_request_valid = 1;
            exportvector_request_write_not_read = 1;
            exportvector_request_addr = exportvector_counter_q;
            exportvector_request_w_data = gbfifo_export_out_data;

            exportvector_counter_d++;
            if (exportvector_counter_q >= NumChunksPerVector-1) begin
                state_d = WAITING_FOR_IN;
                tmatmul_operation_d = ternip_pkg::NO_TMATMUL_OP;
                exportvector_counter_d = 0;
                in_state_working_go_d = 0;
                in_state_waiting_for_in_d = 1;
            end
        end

    // end else if ((state_q == WORKING) && (tmatmul_operation_q == ternip_pkg::EXPORT)) begin // manual forward retiming
    end else if (in_state_working_export_q) begin

        // Read chunks from exportvector
        if (exportvector_counter_q < NumChunksPerVector) begin
            exportvector_request_valid = 1;
            exportvector_request_write_not_read = 0;
            exportvector_request_addr = exportvector_counter_q;
            if (exportvector_request_ready) exportvector_counter_d++;
        end

        // Write chunks to vector registers
        exportvector_read_ready = vector_request_ready_i;
        if (exportvector_read_ready && exportvector_read_valid) begin
            vector_request_valid_o = 1;
            vector_request_write_not_read_o = '1;
            vector_request_vector_select_o = vector_select_q;
            vector_request_vector_addr_o = exportvector_read_addr;
            vector_request_w_data_o = exportvector_read_data;
            if (exportvector_read_addr >= NumChunksPerVector-1) begin
                state_d = WAITING_FOR_IN;
                tmatmul_operation_d = ternip_pkg::NO_TMATMUL_OP;
                in_state_working_export_d = 0;
                in_state_waiting_for_in_d = 1;
            end
        end

    end
end

always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
        queued_valid_q <= '0;
        state_q <= WAITING_FOR_IN;
    end else begin
        queued_valid_q <= queued_valid_d;
        state_q <= state_d;
    end
end
always_ff @(posedge clk_i) begin
    vector_select_q <= vector_select_d;
    tmatmul_operation_q <= tmatmul_operation_d;

    queued_operation_q <= queued_operation_d;
    queued_vector_select_q <= queued_vector_select_d;
    queued_go_matrix_address_q <= queued_go_matrix_address_d;

    importvector_counter_q <= importvector_counter_d;
    importvector_request_addr_counter_q <= importvector_request_addr_counter_d;
    exportvector_counter_q <= exportvector_counter_d;
    import_bram_addr_q <= import_bram_addr_d;
    `ifndef SYNTHESIS
    if (!rst_ni) begin
        vector_select_q <= 'x;
        tmatmul_operation_q <= ternip_pkg::NO_TMATMUL_OP;

        queued_operation_q <= ternip_pkg::NO_TMATMUL_OP;
        queued_vector_select_q <= 'x;
        queued_go_matrix_address_q <= 'x;

        importvector_counter_q <= 'x;
        importvector_request_addr_counter_q <= 'x;
        exportvector_counter_q <= 'x;
        import_bram_addr_q <= 'x;
    end
    `endif
end
always_ff @(posedge clk_i) begin
    ddr_stream_address_q <= ddr_stream_address_d;
    `ifndef SYNTHESIS
    if (!rst_ni) begin
        ddr_stream_address_q <= 'x;
    end
    `endif
end

endmodule

`undef SAFE_CLOG2
