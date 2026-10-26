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

// ternip_trit5_decode
//
// Combinational decoder for one base-3 packed byte holding five ternary weights.
//
// packed_i = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4, with digit 0=-1, 1=0, 2=+1.
// Trit k drives pos_o[k] (weight +1) and neg_o[k] (weight -1); both low means 0.
// Bytes 243..255 are not valid encodings: they decode to five zero weights and
// assert invalid_o.
//
// Decoder selects the implementation:
//  TRIT5_DECODE_ARITH:   repeated divide-by-3 using an exact reciprocal multiply.
//  TRIT5_DECODE_RESIDUE: packed_i = 27*t + r; t = floor(packed_i/27) from constant
//                        compares gives d3, d4; the 5-bit residue r gives d0..d2.
//  TRIT5_DECODE_LUT (and TRIT5_DECODE_DIRECT, whose datapath does not use this
//  module): 256-entry constant table.

module ternip_trit5_decode #(
    parameter ternip_pkg::trit5_decoder_e Decoder = ternip_pkg::TRIT5_DECODE_LUT
) (
    input  logic [7:0] packed_i,
    output logic [4:0] pos_o,
    output logic [4:0] neg_o,
    output logic       invalid_o
);

localparam logic [7:0] NumValidCodes = 8'd243;

typedef struct packed {
    logic       invalid;
    logic [4:0] neg;
    logic [4:0] pos;
} decoded_t;

function automatic decoded_t decode_reference(int unsigned code);
    decoded_t out = '0;
    int unsigned rest = code;
    if (code >= NumValidCodes) begin
        out.invalid = 1'b1;
        return out;
    end
    for (int k = 0; k < 5; k++) begin
        out.neg[k] = ((rest % 3) == 0);
        out.pos[k] = ((rest % 3) == 2);
        rest = rest / 3;
    end
    return out;
endfunction

// One 256-entry truth table per output bit, so each output is a direct
// 8-input lookup (a code-major table becomes a multiply-and-shift after sv2v).
function automatic logic [$bits(decoded_t)-1:0][255:0] build_table();
    logic [$bits(decoded_t)-1:0][255:0] table_out;
    for (int code = 0; code < 256; code++) begin
        decoded_t d = decode_reference(code);
        for (int j = 0; j < $bits(decoded_t); j++) table_out[j][code] = d[j];
    end
    return table_out;
endfunction

// Digit k of a value below 3^NumDigits as +1/-1 flags, one table per flag and digit.
function automatic logic [1:0][2:0][31:0] build_digit_tables(int num_digits, int num_values);
    logic [1:0][2:0][31:0] tables = '0;
    for (int v = 0; v < num_values; v++) begin
        int unsigned rest = v;
        for (int k = 0; k < num_digits; k++) begin
            tables[0][k][v] = ((rest % 3) == 2);
            tables[1][k][v] = ((rest % 3) == 0);
            rest = rest / 3;
        end
    end
    return tables;
endfunction

if (Decoder == ternip_pkg::TRIT5_DECODE_RESIDUE) begin : residue

    // packed_i - 27*t == (packed_i[4:0] + 5*t) mod 32 because -27 == 5 (mod 32),
    // and the residue is below 27 < 32.
    localparam logic [1:0][2:0][31:0] LowTables = build_digit_tables(3, 27);
    localparam logic [1:0][2:0][31:0] HighTables = build_digit_tables(2, 9);

    logic [3:0] t;
    logic [4:0] r;

    always_comb begin
        invalid_o = (packed_i >= NumValidCodes);
        t = '0;
        for (int m = 1; m <= 8; m++) begin
            if (packed_i >= 8'(27 * m)) t = 4'(m);
        end
        r = packed_i[4:0] + 5'(5 * t);
        for (int k = 0; k < 3; k++) begin
            pos_o[k] = !invalid_o && LowTables[0][k][r];
            neg_o[k] = !invalid_o && LowTables[1][k][r];
        end
        for (int k = 0; k < 2; k++) begin
            pos_o[3+k] = !invalid_o && HighTables[0][k][t];
            neg_o[3+k] = !invalid_o && HighTables[1][k][t];
        end
    end

end else if (Decoder != ternip_pkg::TRIT5_DECODE_ARITH) begin : lut

    localparam logic [$bits(decoded_t)-1:0][255:0] Table = build_table();

    decoded_t decoded;
    for (genvar j_GEN = 0; j_GEN < $bits(decoded_t); j_GEN++) begin : output_bit
        assign decoded[j_GEN] = Table[j_GEN][packed_i];
    end
    assign pos_o = decoded.pos;
    assign neg_o = decoded.neg;
    assign invalid_o = decoded.invalid;

end else begin : arith

    // floor(x/3) == (x*171) >> 9 for every 0 <= x <= 255:
    // 171/512 = 1/3 + 1/1536, and frac(x/3) + x/1536 <= 2/3 + 255/1536 < 1.
    function automatic logic [7:0] div3(logic [7:0] x);
        logic [16:0] product = 17'(x) * 17'd171;
        return product[16:9];
    endfunction

    logic [5:0][7:0] quotient;
    logic [4:0][1:0] digit;

    always_comb begin
        quotient[0] = packed_i;
        for (int k = 0; k < 5; k++) begin
            quotient[k+1] = div3(quotient[k]);
            digit[k] = 2'(quotient[k] - 8'(3 * quotient[k+1]));
        end
        invalid_o = (packed_i >= NumValidCodes);
        for (int k = 0; k < 5; k++) begin
            neg_o[k] = !invalid_o && (digit[k] == 2'd0);
            pos_o[k] = !invalid_o && (digit[k] == 2'd2);
        end
    end

end

endmodule
