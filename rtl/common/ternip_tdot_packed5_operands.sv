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

// ternip_tdot_packed5_operands
//
// Ones'-complement dot-product operands generated straight from base-3 packed
// weight bytes (ternip_pkg::TRIT5_DECODE_DIRECT). Lane i uses digit
// (LANE_OFFSET+i) % 5 of byte (LANE_OFFSET+i) / 5, with
// byte = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4 and digit 0=-1, 1=0, 2=+1.
//
// No per-lane weight flags are formed: every operand bit is a single lookup on
// {activation bit, byte},
//   q[j] = act[j] (digit 2), ~act[j] (digit 0), 0 (digit 1 or invalid byte),
// and the sign-extension bits repeat q[ACT_WIDTH-1]. neg_o[i] is the +1
// ones'-complement correction of a negative weight, also one lookup on the byte.
// Invalid bytes (243..255) give five zero weights and set invalid_o, exactly
// like ternip_trit5_decode.

module ternip_tdot_packed5_operands #(
    parameter int NUM_LANES   = 16,
    parameter int ACT_WIDTH   = 8,
    parameter int ROW_WIDTH   = ACT_WIDTH + 1,
    parameter int NUM_BYTES   = (NUM_LANES + 4) / 5,
    parameter int LANE_OFFSET = 0,

    localparam type act_t = logic signed [ACT_WIDTH-1:0],
    localparam type row_t = logic [ROW_WIDTH-1:0]
) (
    input  logic [8*NUM_BYTES-1:0] packed_i,
    input  act_t [NUM_LANES-1:0]   act_i,
    output row_t [NUM_LANES-1:0]   operands_o,
    output logic [NUM_LANES-1:0]   neg_o,
    output logic [NUM_BYTES-1:0]   invalid_o
);

localparam int NumValidCodes = 243;

`ifndef SYNTHESIS
initial begin
    assert ((LANE_OFFSET + NUM_LANES + 4) / 5 <= NUM_BYTES)
        else $fatal(0, "ternip_tdot_packed5_operands: lanes exceed packed bytes");
end
`endif

// digit k of code, or -1 for an invalid code
function automatic int digit_of(int code, int k);
    int rest = code;
    if (code >= NumValidCodes) return -1;
    for (int i = 0; i < k; i++) rest = rest / 3;
    return rest % 3;
endfunction

// Bit-major tables (one 2^n-entry table per digit) so every lookup stays a plain
// n-input function after sv2v.
function automatic logic [4:0][511:0] build_operand_tables();
    logic [4:0][511:0] tables = '0;
    for (int k = 0; k < 5; k++) begin
        for (int code = 0; code < 256; code++) begin
            tables[k][256 + code] = (digit_of(code, k) == 2); // act bit 1
            tables[k][code]       = (digit_of(code, k) == 0); // act bit 0 -> ~act bit 1
        end
    end
    return tables;
endfunction

function automatic logic [4:0][255:0] build_neg_tables();
    logic [4:0][255:0] tables = '0;
    for (int k = 0; k < 5; k++) begin
        for (int code = 0; code < 256; code++) tables[k][code] = (digit_of(code, k) == 0);
    end
    return tables;
endfunction

function automatic logic [255:0] build_invalid_table();
    logic [255:0] table_out = '0;
    for (int code = NumValidCodes; code < 256; code++) table_out[code] = 1'b1;
    return table_out;
endfunction

localparam logic [4:0][511:0] OperandTables = build_operand_tables();
localparam logic [4:0][255:0] NegTables = build_neg_tables();
localparam logic [255:0] InvalidTable = build_invalid_table();

for (genvar b_GEN = 0; b_GEN < NUM_BYTES; b_GEN++) begin : byte_flag
    assign invalid_o[b_GEN] = InvalidTable[packed_i[8*b_GEN +: 8]];
end

for (genvar i_GEN = 0; i_GEN < NUM_LANES; i_GEN++) begin : lane
    localparam int Byte = (LANE_OFFSET + i_GEN) / 5;
    localparam int Digit = (LANE_OFFSET + i_GEN) % 5;

    wire logic [7:0] code = packed_i[8*Byte +: 8];
    logic [ACT_WIDTH-1:0] q;

    for (genvar j_GEN = 0; j_GEN < ACT_WIDTH; j_GEN++) begin : bit_gen
        assign q[j_GEN] = OperandTables[Digit][{act_i[i_GEN][j_GEN], code}];
    end
    assign operands_o[i_GEN] = row_t'({ {ROW_WIDTH{q[ACT_WIDTH-1]}}, q });
    assign neg_o[i_GEN] = NegTables[Digit][code];
end

endmodule
