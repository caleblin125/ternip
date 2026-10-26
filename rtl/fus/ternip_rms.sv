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

// ternip_rms
//
// Functional unit for RMS normalization over vector registers.
//
// CLEAR resets the internal accumulated square-sum register.
// ACCUMULATE adds the squared elements of one vector into that register.
// FINISH_ACCUMULATE computes the reciprocal RMS scale factor.
// NORM multiplies a vector by that scale and writes the normalized result.
//
// Use CLEAR before a new RMS group. Then issue one or more ACCUMULATE commands,
// one FINISH_ACCUMULATE command, and one or more NORM commands. The unit accepts
// one command on in_* and sequences the vector-register and math handshakes
// until that command finishes.
//
// Datapath width: vector chunks are VectorParallelism (VP) elements, but the squarers
// and normalise multipliers are RmsParallelism (RP, ternip_pkg; 0 = VP) wide, so each
// chunk takes VP/RP cycles. The scheduler overlaps RMS with tmatmul streaming, which
// hides most of that (model: VP4 -> RP2 costs ~1% tok/s). Read responses land in a
// small chunk FIFO and a read is only requested when its response has room, so a
// narrow RMS never stalls the shared vector-register pipeline. The scale factor is
// sqrt(rms_length / accumulator) from ternip_rsqrt (no divider).

module ternip_rms #(
    parameter ternip_pkg::ternip_cfg_t Cfg = `TERNIP_CFG
) (
    input  logic                                      clk_i,
    input  logic                                      rst_ni,

    output logic                                      in_ready_o,
    input  logic                                      in_valid_i,
    input  ternip_pkg::rms_op_e                       in_rms_op_i,
    input  ternip_types#(Cfg)::vector_select_t        in_vector1_select_i,
    input  ternip_types#(Cfg)::vector_select_t        in_vector2_select_i,
    input  ternip_types#(Cfg)::immediate_t            in_rms_length_i,

    input  logic                                      vector_request_ready_i,
    output logic                                      vector_request_valid_o,
    output logic                                      vector_request_write_not_read_o,
    output ternip_types#(Cfg)::vector_select_t        vector_request_vector_select_o,
    output ternip_types#(Cfg)::vector_offset_t        vector_request_vector_addr_o,
    output ternip_types#(Cfg)::vector_chunk_t         vector_request_w_data_o,

    output logic                                      vector_read_ready_o,
    input  logic                                      vector_read_valid_i,
    input  ternip_types#(Cfg)::vector_offset_t        vector_read_addr_i,
    input  ternip_types#(Cfg)::vector_chunk_t         vector_read_data_i,

    // debug ports
    output logic                                      accumulator_out_valid_o,
    output ternip_types#(Cfg)::rms_accumulator_t      accumulator_out_result_o,
    output ternip_types#(Cfg)::rms_value_reciprocal_t rms_value_reciprocal_o,
    output logic                                      rms_value_reciprocal_valid_o
);

// Same values as ternip_types#(Cfg) (local copies: Verilator rejects dotted class
// parameters in parameter positions, which kept RMS out of simulation).
localparam int RmsSqaSumPrecision          = 2 * Cfg.FixedPointPrecision;
localparam int RmsSqaSumExponent           = 2 * Cfg.FixedPointExponent;
localparam int RmsValueReciprocalPrecision = 2 * (Cfg.FixedPointPrecision + 1);
localparam int RmsValueReciprocalExponent  = -int'(Cfg.FixedPointPrecision);
localparam int RmsAccumulatorWidth         = 2 * Cfg.FixedPointPrecision + $clog2(Cfg.D) + 1;
localparam int NumChunksPerVector          = Cfg.D / Cfg.VectorParallelism;

localparam int VP = Cfg.VectorParallelism;
localparam int RP = (ternip_pkg::RmsParallelism <= 0 || ternip_pkg::RmsParallelism > VP)
                    ? VP : ternip_pkg::RmsParallelism;
localparam int Beats = VP / RP;            // cycles per chunk
localparam int ChunkFifoDepth = 8;         // read credits

`ifndef SYNTHESIS
initial assert ((VP % RP) == 0) else $fatal(1, "RmsParallelism (%0d) must divide VectorParallelism (%0d)", RP, VP);
`endif

typedef logic signed [Cfg.FixedPointPrecision-1:0]  fixed_point_t;
typedef fixed_point_t [VP-1:0]                      chunk_t;
typedef fixed_point_t [RP-1:0]                      lanes_t;
typedef logic signed [RmsSqaSumPrecision-1:0]       sqa_t;
typedef logic signed [RmsAccumulatorWidth-1:0]      accumulator_t;
typedef logic signed [RmsValueReciprocalPrecision-1:0] reciprocal_t;

ternip_pkg::rms_op_e                rms_op_d, rms_op_q;
ternip_types#(Cfg)::vector_select_t in_vector1_select_d, in_vector1_select_q;
ternip_types#(Cfg)::vector_select_t in_vector2_select_d, in_vector2_select_q;
ternip_types#(Cfg)::immediate_t     in_rms_length_d, in_rms_length_q;

enum logic [2:0] {
    WAITING_FOR_IN,
    WORKING,
    SENDING_FINAL_TO_ACCUMULATOR,
    WAITING_FOR_ACCUMULATOR,
    SENDING_TO_RSQRT
} state_d, state_q = WAITING_FOR_IN;

logic [$clog2(NumChunksPerVector):0]         read_counter_d, read_counter_q;       // chunks requested
logic [$clog2(NumChunksPerVector):0]         write_counter_d, write_counter_q;     // chunks written (NORM)
logic [$clog2(NumChunksPerVector * Beats):0] beat_counter_d, beat_counter_q;       // beats accumulated
logic [$clog2(ChunkFifoDepth):0]             outstanding_d, outstanding_q;         // requested, not yet in the PISO


// Read responses -> chunk FIFO -> parallel-to-serial (RP lanes per cycle)

logic   fifo_in_ready, fifo_out_valid, fifo_out_yumi;
chunk_t fifo_out_data;

bsg_fifo_1r1w_small #(
    .width_p($bits(chunk_t)),
    .els_p(ChunkFifoDepth),
    .harden_p(0),
    .ready_THEN_valid_p(0)
) chunk_fifo (
    .clk_i,
    .reset_i(!rst_ni),
    .v_i(vector_read_valid_i && vector_read_ready_o),
    .ready_param_o(fifo_in_ready),
    .data_i(vector_read_data_i),
    .v_o(fifo_out_valid),
    .data_o(fifo_out_data),
    .yumi_i(fifo_out_yumi)
);

`ifndef SYNTHESIS
assert property (@(posedge clk_i) disable iff (!rst_ni) (vector_read_valid_i && outstanding_q != 0) |-> fifo_in_ready)
    else $fatal(1, "ternip_rms: read response with no FIFO room (credit bug)");
`endif

logic                       piso_valid_d, piso_valid_q;
chunk_t                     piso_data_d, piso_data_q;
logic [$clog2(Beats+1)-1:0] piso_beat_d, piso_beat_q;
logic                       piso_take;   // consumer accepts the current RP lanes this cycle
lanes_t                     piso_lanes;

// piso_lanes = piso_data_q[piso_beat_q*RP +: RP], held in a register: the beat
// select runs on the _d side, so the squarers / normalise multipliers see a
// flop output (the select fed the multiplier's input conversion directly).
always_ff @(posedge clk_i) begin
    for (int r = 0; r < RP; r++)
        if (int'(piso_beat_d) * RP + r < $bits(chunk_t) / $bits(piso_lanes[0]))
            piso_lanes[r] <= piso_data_d[int'(piso_beat_d) * RP + r];
end

wire logic piso_last = (int'(piso_beat_q) == Beats - 1);


// Accumulate: RP squarers -> RP-operand accumulator

logic [RP-1:0] square_in_ready, square_in_valid, square_out_ready, square_out_valid;
sqa_t [RP-1:0] square_result_out;

for (genvar i_GEN = 0; i_GEN < RP; i_GEN++) begin : parallel_squares
    ternip_mul #(
        .InAPrecision(Cfg.FixedPointPrecision),
        .InAExponent(Cfg.FixedPointExponent),
        .InBPrecision(Cfg.FixedPointPrecision),
        .InBExponent(Cfg.FixedPointExponent),
        .OutPrecision(RmsSqaSumPrecision),
        .OutExponent(RmsSqaSumExponent),
        .FinalConversionNumPipelineStages(2)
    ) square (
        .clk_i,
        .rst_ni,
        .in_ready_o(square_in_ready[i_GEN]),
        .in_valid_i(square_in_valid[i_GEN]),
        .a_i(piso_lanes[i_GEN]),
        .b_i(piso_lanes[i_GEN]),
        .out_ready_i(square_out_ready[i_GEN]),
        .out_valid_o(square_out_valid[i_GEN]),
        .y_o(square_result_out[i_GEN])
    );
end

logic         accumulator_in_ready, accumulator_in_valid, accumulator_in_final;
sqa_t [RP-1:0] accumulator_in_operand;
logic         accumulator_out_ready, accumulator_out_valid;
accumulator_t accumulator_out_result;

always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
        accumulator_out_valid_o <= 0;
    end else if (accumulator_out_valid) begin
        accumulator_out_valid_o <= 1;
    end
end
always_ff @(posedge clk_i) begin
    accumulator_out_result_o <= accumulator_out_result;
    `ifndef SYNTHESIS
    if (!rst_ni) begin
        accumulator_out_result_o <= 'x;
    end
    `endif
end

ternip_multioperand_accumulator #(
    .operand_t(sqa_t),
    .result_t(accumulator_t),
    .NUM_OPERANDS(RP),
    .NEXT_STAGE_FANIN(2),
    .D(Cfg.D)
) multioperand_accumulator (
    .clk_i,
    .rst_ni,
    .in_ready_o(accumulator_in_ready),
    .in_valid_i(accumulator_in_valid),
    .in_final_i(accumulator_in_final),
    .in_operands_i(accumulator_in_operand),
    .out_ready_i(accumulator_out_ready),
    .out_valid_o(accumulator_out_valid),
    .out_result_o(accumulator_out_result)
);


// Finish accumulate: rms_value_reciprocal = sqrt(rms_length / accumulator)

reciprocal_t rms_value_reciprocal_d, rms_value_reciprocal_q;
logic        rms_value_reciprocal_valid_d, rms_value_reciprocal_valid_q;

assign rms_value_reciprocal_o       = rms_value_reciprocal_q;
assign rms_value_reciprocal_valid_o = rms_value_reciprocal_valid_q;

// Registered operands break the MOA -> rsqrt combinational path.
accumulator_t                   rsqrt_acc_d, rsqrt_acc_q;
ternip_types#(Cfg)::immediate_t rsqrt_len_d, rsqrt_len_q;
logic                           rsqrt_in_ready, rsqrt_in_valid, rsqrt_out_ready, rsqrt_out_valid;
reciprocal_t                    rsqrt_y;

ternip_rsqrt #(
    .AccWidth(RmsAccumulatorWidth),
    .AccExponent(RmsSqaSumExponent),
    .LenWidth(Cfg.ImmediateWidth),
    .OutPrecision(RmsValueReciprocalPrecision),
    .OutExponent(RmsValueReciprocalExponent)
) rsqrt (
    .clk_i,
    .rst_ni,
    .in_ready_o(rsqrt_in_ready),
    .in_valid_i(rsqrt_in_valid),
    .acc_i(rsqrt_acc_q),
    .len_i(rsqrt_len_q),
    .out_ready_i(rsqrt_out_ready),
    .out_valid_o(rsqrt_out_valid),
    .y_o(rsqrt_y)
);


// Norm: RP multipliers -> serial-to-parallel -> one chunk write

logic [RP-1:0] norm_mul_in_ready, norm_mul_in_valid, norm_mul_out_ready, norm_mul_out_valid;
lanes_t        norm_mul_out_result;

for (genvar i_GEN = 0; i_GEN < RP; i_GEN++) begin : parallel_norm_mul
    ternip_mul #(
        .InAPrecision(Cfg.FixedPointPrecision),
        .InAExponent(Cfg.FixedPointExponent),
        .InBPrecision(RmsValueReciprocalPrecision),
        .InBExponent(RmsValueReciprocalExponent),
        .OutPrecision(Cfg.FixedPointPrecision),
        .OutExponent(Cfg.FixedPointExponent),
        .FinalConversionNumPipelineStages(2)
    ) norm_mul (
        .clk_i,
        .rst_ni,
        .in_ready_o(norm_mul_in_ready[i_GEN]),
        .in_valid_i(norm_mul_in_valid[i_GEN]),
        .a_i(piso_lanes[i_GEN]),
        .b_i(rms_value_reciprocal_q),
        .out_ready_i(norm_mul_out_ready[i_GEN]),
        .out_valid_o(norm_mul_out_valid[i_GEN]),
        .y_o(norm_mul_out_result[i_GEN])
    );
end

chunk_t                     sipo_data_d, sipo_data_q;
logic [$clog2(Beats+1)-1:0] sipo_beat_d, sipo_beat_q;
logic                       sipo_full_d, sipo_full_q;

`ifndef SYNTHESIS
// All lanes of an RP group move together, so lane 0's handshakes stand for all.
always_comb begin
    assert (!rst_ni || ((&square_in_ready) == (|square_in_ready)));
    assert (!rst_ni || ((&square_out_valid) == (|square_out_valid)));
    assert (!rst_ni || ((&norm_mul_in_ready) == (|norm_mul_in_ready)));
    assert (!rst_ni || ((&norm_mul_out_valid) == (|norm_mul_out_valid)));
end
// squarer outputs are not back-pressured during ACCUMULATE, so the accumulator must take them
assert property (@(posedge clk_i) disable iff (!rst_ni)
    (state_q == WORKING && rms_op_q == ternip_pkg::ACCUMULATE && accumulator_in_valid) |-> accumulator_in_ready)
    else $fatal(1, "ternip_rms: accumulator not ready for a square result");
`endif


// State machine

wire logic read_credit = (outstanding_q < ChunkFifoDepth) && (read_counter_q < NumChunksPerVector);

always_comb begin
    in_ready_o = 0;
    rms_op_d = rms_op_q;
    in_vector1_select_d = in_vector1_select_q;
    in_vector2_select_d = in_vector2_select_q;
    in_rms_length_d = in_rms_length_q;
    state_d = state_q;

    vector_request_valid_o = 0;
    vector_request_write_not_read_o = 'x;
    vector_request_vector_addr_o = 'x;
    vector_request_vector_select_o = 'x;
    vector_request_w_data_o = 'x;
    // The core shares one read-response bus between FUs: only claim it while reading.
    vector_read_ready_o = fifo_in_ready && (state_q == WORKING)
                          && (rms_op_q == ternip_pkg::ACCUMULATE || rms_op_q == ternip_pkg::NORM);

    read_counter_d = read_counter_q;
    write_counter_d = write_counter_q;
    beat_counter_d = beat_counter_q;
    outstanding_d = outstanding_q;

    piso_valid_d = piso_valid_q;
    piso_data_d = piso_data_q;
    piso_beat_d = piso_beat_q;
    piso_take = 0;
    fifo_out_yumi = 0;

    accumulator_in_valid = 0;
    accumulator_in_final = 'x;
    accumulator_in_operand = 'x;
    accumulator_out_ready = 0;

    square_in_valid = '0;
    square_out_ready = '0;

    rsqrt_acc_d = rsqrt_acc_q;
    rsqrt_len_d = rsqrt_len_q;
    rsqrt_in_valid = 0;
    rsqrt_out_ready = 0;

    rms_value_reciprocal_d = rms_value_reciprocal_q;
    rms_value_reciprocal_valid_d = rms_value_reciprocal_valid_q;

    norm_mul_in_valid = '0;
    norm_mul_out_ready = '0;
    sipo_data_d = sipo_data_q;
    sipo_beat_d = sipo_beat_q;
    sipo_full_d = sipo_full_q;

    if (state_q == WAITING_FOR_IN) begin
        in_ready_o = 1;
        if (in_ready_o && in_valid_i) begin
            rms_op_d = in_rms_op_i;
            in_vector1_select_d = in_vector1_select_i;
            in_vector2_select_d = in_vector2_select_i;
            in_rms_length_d = in_rms_length_i;
            read_counter_d = 0;
            write_counter_d = 0;
            beat_counter_d = 0;
            if (in_rms_op_i == ternip_pkg::CLEAR) begin
                state_d = SENDING_FINAL_TO_ACCUMULATOR;
                rms_value_reciprocal_d = 'x;
                rms_value_reciprocal_valid_d = 0;
            end else if (in_rms_op_i == ternip_pkg::ACCUMULATE) begin
                state_d = WORKING;
                rms_value_reciprocal_d = 'x;
                rms_value_reciprocal_valid_d = 0;
            end else if (in_rms_op_i == ternip_pkg::FINISH_ACCUMULATE) begin
                state_d = SENDING_FINAL_TO_ACCUMULATOR;
                rms_value_reciprocal_d = 'x;
                rms_value_reciprocal_valid_d = 0;
            end else if (in_rms_op_i == ternip_pkg::NORM) begin
                state_d = WORKING;
            end
        end
    end else if ((state_q == SENDING_FINAL_TO_ACCUMULATOR) && (rms_op_q == ternip_pkg::CLEAR)) begin
        accumulator_in_valid = 1;
        accumulator_in_final = 1;
        accumulator_in_operand = 'x;
        accumulator_out_ready = 1; // drain any stale results
        if (accumulator_in_ready) begin
            state_d = WAITING_FOR_ACCUMULATOR;
        end
    end else if ((state_q == WAITING_FOR_ACCUMULATOR) && (rms_op_q == ternip_pkg::CLEAR)) begin
        accumulator_out_ready = 1;
        if (accumulator_out_valid) begin
            state_d = WAITING_FOR_IN;
            rms_op_d = ternip_pkg::NO_RMS_OP;
        end
    end else if ((state_q == WORKING) && (rms_op_q == ternip_pkg::ACCUMULATE)) begin
        // read requests, limited by FIFO credits
        if (read_credit) begin
            vector_request_valid_o = 1;
            vector_request_write_not_read_o = 0;
            vector_request_vector_addr_o = read_counter_q;
            vector_request_vector_select_o = in_vector1_select_q;
            if (vector_request_ready_i) begin
                read_counter_d = read_counter_q + 1;
                outstanding_d = outstanding_d + 1;
            end
        end
        // PISO -> squarers
        square_in_valid = {RP{piso_valid_q}};
        piso_take = piso_valid_q && square_in_ready[0];
        // squarers -> accumulator
        square_out_ready = '1;
        accumulator_in_valid = square_out_valid[0];
        accumulator_in_final = 0;
        accumulator_in_operand = square_result_out;
        if (accumulator_in_ready && accumulator_in_valid) begin
            beat_counter_d = beat_counter_q + 1;
            if (int'(beat_counter_q) == NumChunksPerVector * Beats - 1) begin
                state_d = WAITING_FOR_IN;
                rms_op_d = ternip_pkg::NO_RMS_OP;
            end
        end
    end else if ((state_q == SENDING_FINAL_TO_ACCUMULATOR) && (rms_op_q == ternip_pkg::FINISH_ACCUMULATE)) begin
        accumulator_in_valid = 1;
        accumulator_in_final = 1;
        accumulator_in_operand = '0;
        accumulator_out_ready = 1;
        if (accumulator_in_ready) begin
            state_d = WAITING_FOR_ACCUMULATOR;
        end
    end else if ((state_q == WAITING_FOR_ACCUMULATOR) && (rms_op_q == ternip_pkg::FINISH_ACCUMULATE)) begin
        accumulator_out_ready = 1;
        if (accumulator_out_valid) begin
            state_d = SENDING_TO_RSQRT;
            `ifndef SYNTHESIS
            assert(!rst_ni || $isunknown(in_rms_length_q) || (in_rms_length_q > 0));
            assert(!rst_ni || $isunknown(accumulator_out_result) || (accumulator_out_result >= 0)) else $fatal(0, "accumulator_out_result=%b", accumulator_out_result);
            `endif
            rsqrt_acc_d = accumulator_out_result;   // <= 0 is clamped to one LSB inside ternip_rsqrt
            rsqrt_len_d = in_rms_length_q;
        end
    end else if ((state_q == SENDING_TO_RSQRT) && (rms_op_q == ternip_pkg::FINISH_ACCUMULATE)) begin
        rsqrt_in_valid = 1;
        if (rsqrt_in_ready) begin
            state_d = WORKING;
        end
    end else if ((state_q == WORKING) && (rms_op_q == ternip_pkg::FINISH_ACCUMULATE)) begin
        rsqrt_out_ready = 1;
        if (rsqrt_out_valid) begin
            rms_value_reciprocal_d = rsqrt_y;
            rms_value_reciprocal_valid_d = 1;
            state_d = WAITING_FOR_IN;
            rms_op_d = ternip_pkg::NO_RMS_OP;
        end
    end else if ((state_q == WORKING) && (rms_op_q == ternip_pkg::NORM)) begin
        // a completed chunk is written first; otherwise request a read (credit-limited)
        if (sipo_full_q) begin
            vector_request_valid_o = 1;
            vector_request_write_not_read_o = 1;
            vector_request_vector_addr_o = write_counter_q;
            vector_request_w_data_o = sipo_data_q;
            vector_request_vector_select_o = in_vector2_select_q;
            if (vector_request_ready_i) begin
                sipo_full_d = 0;
                write_counter_d = write_counter_q + 1;
                if (int'(write_counter_q) == NumChunksPerVector - 1) begin
                    state_d = WAITING_FOR_IN;
                    rms_op_d = ternip_pkg::NO_RMS_OP;
                end
            end
        end else if (read_credit) begin
            vector_request_valid_o = 1;
            vector_request_write_not_read_o = 0;
            vector_request_vector_addr_o = read_counter_q;
            vector_request_vector_select_o = in_vector1_select_q;
            if (vector_request_ready_i) begin
                read_counter_d = read_counter_q + 1;
                outstanding_d = outstanding_d + 1;
            end
        end
        // PISO -> multipliers
        norm_mul_in_valid = {RP{piso_valid_q}};
        piso_take = piso_valid_q && norm_mul_in_ready[0];
        // multipliers -> SIPO (accept while the SIPO has room, or as its chunk is written)
        norm_mul_out_ready = {RP{ !sipo_full_q || (vector_request_valid_o && vector_request_write_not_read_o && vector_request_ready_i) }};
        if (norm_mul_out_ready[0] && norm_mul_out_valid[0]) begin
            for (int r = 0; r < RP; r++) sipo_data_d[int'(sipo_beat_q) * RP + r] = norm_mul_out_result[r];
            if (int'(sipo_beat_q) == Beats - 1) begin
                sipo_beat_d = 0;
                sipo_full_d = 1;
            end else begin
                sipo_beat_d = sipo_beat_q + 1;
            end
        end
    end

    // PISO advance / refill from the chunk FIFO
    if (piso_take && !piso_last) begin
        piso_beat_d = piso_beat_q + 1;
    end else if (!piso_valid_q || (piso_take && piso_last)) begin
        piso_valid_d = fifo_out_valid;
        piso_beat_d = 0;
        if (fifo_out_valid) begin
            fifo_out_yumi = 1;
            piso_data_d = fifo_out_data;
            outstanding_d = outstanding_d - 1;
        end
    end
end

always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
        state_q <= WAITING_FOR_IN;
        rms_value_reciprocal_valid_q <= 0;
        outstanding_q <= '0;
        piso_valid_q <= 0;
        piso_beat_q <= '0;
        sipo_full_q <= 0;
        sipo_beat_q <= '0;
    end else begin
        state_q <= state_d;
        rms_value_reciprocal_valid_q <= rms_value_reciprocal_valid_d;
        outstanding_q <= outstanding_d;
        piso_valid_q <= piso_valid_d;
        piso_beat_q <= piso_beat_d;
        sipo_full_q <= sipo_full_d;
        sipo_beat_q <= sipo_beat_d;
    end
end
always_ff @(posedge clk_i) begin
    rms_op_q <= rms_op_d;
    read_counter_q <= read_counter_d;
    write_counter_q <= write_counter_d;
    beat_counter_q <= beat_counter_d;

    in_vector1_select_q <= in_vector1_select_d;
    in_vector2_select_q <= in_vector2_select_d;
    in_rms_length_q <= in_rms_length_d;

    rms_value_reciprocal_q <= rms_value_reciprocal_d;
    piso_data_q <= piso_data_d;
    sipo_data_q <= sipo_data_d;

    rsqrt_acc_q <= rsqrt_acc_d;
    rsqrt_len_q <= rsqrt_len_d;
    `ifndef SYNTHESIS
    if (!rst_ni) begin
        rms_op_q <= ternip_pkg::NO_RMS_OP;
        read_counter_q <= 'x;
        write_counter_q <= 'x;
        beat_counter_q <= 'x;

        in_vector1_select_q <= 'x;
        in_vector2_select_q <= 'x;
        in_rms_length_q <= 'x;

        rms_value_reciprocal_q <= 'x;
    end
    `endif
end

endmodule
