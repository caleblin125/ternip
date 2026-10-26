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

// ternip_tdot_popcount
//
// Free-running pipelined population count with a fixed latency of NUM_STAGES
// cycles (0 = combinational). A fan-in-2 adder tree is spread over the stages;
// any stages beyond the tree depth become a delay line.

module ternip_tdot_popcount #(
    parameter int NUM_BITS = 8,
    parameter int NUM_STAGES = 1,
    localparam int CountWidth = $clog2(NUM_BITS+1)
) (
    input  logic                  clk_i,
    input  logic [NUM_BITS-1:0]   bits_i,
    output logic [CountWidth-1:0] count_o
);

typedef logic [CountWidth-1:0] count_t;

localparam int NumLevels = (NUM_BITS > 1) ? $clog2(NUM_BITS) : 0;
localparam int NumTreeRegs = (NUM_STAGES < NumLevels) ? NUM_STAGES : NumLevels;
localparam int NumDelayRegs = NUM_STAGES - NumTreeRegs;

function automatic int rows_at(int level);
    int n = NUM_BITS;
    for (int l = 0; l < level; l++) n = (n + 1) / 2;
    return n;
endfunction

function automatic bit reg_after(int level);
    return (((level + 1) * NumTreeRegs) / NumLevels) != ((level * NumTreeRegs) / NumLevels);
endfunction

count_t tree_out;

// Rows between levels live at module scope (no cross-generate-block
// references, which some synthesis front ends mis-resolve).
count_t [NumLevels:0][NUM_BITS-1:0] level_rows;

for (genvar i = 0; i < NUM_BITS; i++) begin : g_bit
    assign level_rows[0][i] = count_t'(bits_i[i]);
end

for (genvar l = 0; l < NumLevels; l++) begin : level
    localparam int NIn = rows_at(l);
    localparam int NOut = rows_at(l + 1);

    count_t [NOut-1:0] rows_comb, rows_out;

    always_comb begin
        for (int j = 0; j < NOut; j++) begin
            rows_comb[j] = level_rows[l][2*j];
            if (2*j + 1 < NIn) rows_comb[j] += level_rows[l][2*j + 1];
        end
    end

    if (reg_after(l)) begin : g_reg
        always_ff @(posedge clk_i) rows_out <= rows_comb;
    end else begin : g_comb
        assign rows_out = rows_comb;
    end

    assign level_rows[l+1][NOut-1:0] = rows_out;
    if (NOut < NUM_BITS) begin : g_pad
        assign level_rows[l+1][NUM_BITS-1:NOut] = '0;
    end
end

assign tree_out = level_rows[NumLevels][0];

if (NumDelayRegs == 0) begin : g_no_delay
    assign count_o = tree_out;
end else begin : g_delay
    count_t [NumDelayRegs-1:0] delay_q;
    always_ff @(posedge clk_i) begin
        delay_q[0] <= tree_out;
        for (int k = 1; k < NumDelayRegs; k++) delay_q[k] <= delay_q[k-1];
    end
    assign count_o = delay_q[NumDelayRegs-1];
end

endmodule
