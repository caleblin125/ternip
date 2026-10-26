// ternip_tdot_dsp_adds
//
// One level of the TDOT_ONES_DSP_TREE reduction: NOUT registered pairwise sums of
// NIN rows (an odd last row passes through registered). Marked for DSP48 SIMD so
// Vivado packs two W <= 24 bit adds per DSP (TWO24) with the register as PREG,
// moving the reduction out of LUTs.

(* use_dsp = "simd" *)
module ternip_tdot_dsp_adds #(
    parameter int W    = 24,
    parameter int NIN  = 2,
    parameter int NOUT = 1
) (
    input  logic                     clk_i,
    input  logic [NIN-1:0][W-1:0]    rows_i,
    output logic [NOUT-1:0][W-1:0]   sums_o
);

(* use_dsp = "simd" *) logic [NOUT-1:0][W-1:0] sums_q;

always_ff @(posedge clk_i) begin
    for (int j = 0; j < NOUT; j++) begin
        if (2 * j + 1 < NIN) sums_q[j] <= rows_i[2 * j] + rows_i[2 * j + 1];
        else                 sums_q[j] <= rows_i[2 * j];
    end
end

assign sums_o = sums_q;

endmodule
