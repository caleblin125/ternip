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

// ternip_gearbox_fifo
//
// Ready/valid width converter between arbitrary input and output word sizes.
//
// If one width is an integer multiple of the other, this wraps the appropriate
// BaseJump PISO or SIPO primitive. For non-multiple widths it collects input
// words into a staging register, emits OutDataWidth chunks, and preserves little
// endian bit order by shifting low bits out first.
//
// Use this to bridge stream widths such as byte streams to vector chunks. Set
// FastPush when the non-multiple path should immediately move newly staged data
// toward the output in the same control step.

`define SAFE_CLOG2(x) ( (((x)==1) || ((x)==0))? 1 : $clog2((x)))

module ternip_gearbox_fifo #(
    parameter int InDataWidth = 8,
    parameter int OutDataWidth = 8,
    parameter bit FastPush = 0,
    // Widths that do not divide each other: 1 = two chained gearboxes through
    // gcd(InDataWidth, OutDataWidth) bits (each a cheap PISO / SIPO, gcd bits per
    // cycle), 0 = the staging-register shifter (full width per cycle, but wide
    // variable shifts that fail timing at high utilization).
    parameter bit NonMultipleViaGcd = 0
) (
    input  logic                    clk_i,
    input  logic                    rst_ni,

    output logic                    in_ready_o,
    input  logic                    in_valid_i,
    input  logic [InDataWidth-1:0]  in_data_i,

    input  logic                    out_ready_i,
    output logic                    out_valid_o,
    output logic [OutDataWidth-1:0] out_data_o
);

function automatic int gcd(int a, int b);
    while (b != 0) begin
    int t = a % b;
    a = b;
    b = t;
    end
    return a;
endfunction

if (gcd(InDataWidth, OutDataWidth) == OutDataWidth) begin : piso

    // https://github.com/bespoke-silicon-group/basejump_stl/blob/a43571d2/bsg_dataflow/bsg_parallel_in_serial_out.sv
    bsg_parallel_in_serial_out #(
        .width_p(OutDataWidth),
        .els_p(InDataWidth / OutDataWidth),
        .hi_to_lo_p(0)
    ) piso (
        .clk_i,
        .reset_i(!rst_ni),

        .ready_and_o(in_ready_o),
        .valid_i(in_valid_i),
        .data_i(in_data_i),

        .yumi_i(out_ready_i && out_valid_o),
        .valid_o(out_valid_o),
        .data_o(out_data_o)
    );

end else if (gcd(InDataWidth, OutDataWidth) == InDataWidth) begin : sipo

    // https://github.com/bespoke-silicon-group/basejump_stl/blob/a43571d2/bsg_dataflow/bsg_serial_in_parallel_out_full.sv
    bsg_serial_in_parallel_out_full #(
        .width_p(InDataWidth),
        .els_p(OutDataWidth / InDataWidth),
        .hi_to_lo_p(0)
    ) sipo (
        .clk_i,
        .reset_i(!rst_ni),

        .ready_and_o(in_ready_o),
        .v_i(in_valid_i),
        .data_i(in_data_i),

        .yumi_i(out_ready_i && out_valid_o),
        .v_o(out_valid_o),
        .data_o(out_data_o)
    );

end else if (NonMultipleViaGcd) begin : non_multiple_via_gcd

    localparam int MidWidth = gcd(InDataWidth, OutDataWidth);

    logic                mid_ready;
    logic                mid_valid;
    logic [MidWidth-1:0] mid_data;

    ternip_gearbox_fifo #(
        .InDataWidth(InDataWidth),
        .OutDataWidth(MidWidth),
        .FastPush(FastPush)
    ) to_mid (
        .clk_i,
        .rst_ni,
        .in_ready_o,
        .in_valid_i,
        .in_data_i,
        .out_ready_i(mid_ready),
        .out_valid_o(mid_valid),
        .out_data_o(mid_data)
    );

    ternip_gearbox_fifo #(
        .InDataWidth(MidWidth),
        .OutDataWidth(OutDataWidth),
        .FastPush(FastPush)
    ) from_mid (
        .clk_i,
        .rst_ni,
        .in_ready_o(mid_ready),
        .in_valid_i(mid_valid),
        .in_data_i(mid_data),
        .out_ready_i,
        .out_valid_o,
        .out_data_o
    );

end else begin : non_multiple

    // smallest multiple of InDataWidth that is greater than OutDataWidth
    localparam int StagingWidth =
        ((OutDataWidth + InDataWidth - 1) / InDataWidth) * InDataWidth;

    logic in_staging_ready;
    logic in_staging_valid;
    logic [StagingWidth-1:0] in_staging_data;

    // https://github.com/bespoke-silicon-group/basejump_stl/blob/a43571d2/bsg_dataflow/bsg_serial_in_parallel_out_full.sv
    bsg_serial_in_parallel_out_full #(
        .width_p(InDataWidth),
        .els_p(StagingWidth / InDataWidth),
        .hi_to_lo_p(0)
    ) sipo (
        .clk_i,
        .reset_i(!rst_ni),

        .ready_and_o(in_ready_o),
        .v_i(in_valid_i),
        .data_i(in_data_i),

        .yumi_i(in_staging_ready && in_staging_valid),
        .v_o(in_staging_valid),
        .data_o(in_staging_data)
    );

    localparam int UnitWidth = gcd(StagingWidth, OutDataWidth);
    localparam int StagingFull = StagingWidth/UnitWidth;
    localparam int OutFull = OutDataWidth/UnitWidth;

    logic [StagingWidth-1:0] staging_data_d, staging_data_q;
    logic [`SAFE_CLOG2(StagingFull+1)-1:0] staging_state_d, staging_state_q;
    logic [OutDataWidth-1:0] out_data_d, out_data_q;
    logic [`SAFE_CLOG2(OutFull+1)-1:0] out_state_d, out_state_q;

    logic [`SAFE_CLOG2(StagingFull)-1:0] shift_amount;

    always_comb begin
        staging_data_d = staging_data_q;
        staging_state_d = staging_state_q;
        out_data_d = out_data_q;
        out_state_d = out_state_q;

        shift_amount = 'x;
        if (out_ready_i && out_valid_o) begin // can output
            out_state_d = 0;
            out_data_d = '0;
            if (staging_state_q >= OutFull) begin
                shift_amount = OutFull;
            end else begin
                shift_amount = staging_state_q;
            end
            out_data_d = staging_data_q;
            staging_data_d >>= (UnitWidth*shift_amount);
            out_state_d += shift_amount;
            staging_state_d -= shift_amount;
        end else if (!out_valid_o) begin // cannot output, shift anyways
            if (staging_state_q+out_state_q > OutFull) begin
                shift_amount = OutFull-out_state_q;
            end else begin
                shift_amount = staging_state_q;
            end
            out_data_d = out_data_q | (staging_data_q<<(UnitWidth*out_state_q));
            staging_data_d >>= (UnitWidth*shift_amount);
            out_state_d += shift_amount;
            staging_state_d -= shift_amount;
        end

        in_staging_ready = (staging_state_d == 0);

        if (in_staging_ready && in_staging_valid) begin
            staging_data_d = in_staging_data;
            staging_state_d = StagingFull;

            if (FastPush) begin
                if (StagingFull+out_state_d > OutFull) begin
                    shift_amount = OutFull-out_state_d;
                end else begin
                    shift_amount = StagingFull;
                end
                out_data_d = out_data_d | (staging_data_d<<(UnitWidth*out_state_d));
                staging_data_d >>= (UnitWidth*shift_amount);
                out_state_d += shift_amount;
                staging_state_d -= shift_amount;
            end
        end
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            staging_data_q <= '0;
            staging_state_q <= '0;
            out_data_q <= '0;
            out_state_q <= '0;
        end else begin
            staging_data_q <= staging_data_d;
            staging_state_q <= staging_state_d;
            out_data_q <= out_data_d;
            out_state_q <= out_state_d;
        end
    end

    assign out_valid_o = (out_state_q == OutFull);
    assign out_data_o = out_data_q;

end

endmodule

`undef SAFE_CLOG2
