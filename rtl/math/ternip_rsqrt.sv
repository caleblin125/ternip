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

// ternip_rsqrt
//
// RMS scale factor without a divider: y = round( sqrt(len / acc) ), where acc is a
// non-negative fixed-point sum of squares (exponent AccExponent, must be even) and
// len an unsigned integer, returned as a signed fixed-point value (OutPrecision,
// OutExponent), saturated. acc <= 0 is treated as one LSB (as the old divider path
// did); len == 0 returns 0.
//
// Both operands are normalised to m * 4^k with m in [1,4) (30 fraction bits), so
//     sqrt(len / acc) = m_len * rsqrt(m_len) * rsqrt(m_acc) * 2^(k_len - k_acc)
// and each rsqrt is a 192-entry seed (top 8 bits of m) plus two Newton steps
// r <- r * (3 - m r^2) / 2. All 14 multiplies run sequentially on one pipelined
// 32x32 unsigned multiplier (~85 cycles per result). A bit-accurate model of this
// arithmetic is within 1 LSB of exact sqrt(len/acc) over the full input range.

module ternip_rsqrt #(
    parameter int AccWidth     = 43,
    parameter int AccExponent  = -10,
    parameter int LenWidth     = 16,
    parameter int OutPrecision = 34,
    parameter int OutExponent  = -16
) (
    input  logic                           clk_i,
    input  logic                           rst_ni,

    output logic                           in_ready_o,
    input  logic                           in_valid_i,
    input  logic signed [AccWidth-1:0]     acc_i,
    input  logic        [LenWidth-1:0]     len_i,

    input  logic                           out_ready_i,
    output logic                           out_valid_o,
    output logic signed [OutPrecision-1:0] y_o
);

localparam int F = 30;            // mantissa / rsqrt fraction bits
localparam int MulLatency = 4;    // issue -> product
localparam int Shift0 = -AccExponent / 2 - OutExponent;
localparam logic [63:0] OutMax = (64'd1 << (OutPrecision - 1)) - 1;

`ifndef SYNTHESIS
initial begin
    assert ((AccExponent % 2) == 0) else $fatal(1, "ternip_rsqrt: AccExponent must be even");
    assert (LenWidth <= 32) else $fatal(1, "ternip_rsqrt: LenWidth must be <= 32");
end
`endif

typedef logic [31:0] mant_t;    // m in [1,4): 2.30
typedef logic [30:0] recip_t;   // r in (0.5,1]: 1.30

function automatic recip_t seed(input logic [7:0] idx);
    case (idx)
        8'd64: seed = 31'd1069571937;
        8'd65: seed = 31'd1061375863;
        8'd66: seed = 31'd1053365364;
        8'd67: seed = 31'd1045533543;
        8'd68: seed = 31'd1037873853;
        8'd69: seed = 31'd1030380081;
        8'd70: seed = 31'd1023046322;
        8'd71: seed = 31'd1015866961;
        8'd72: seed = 31'd1008836655;
        8'd73: seed = 31'd1001950318;
        8'd74: seed = 31'd995203101;
        8'd75: seed = 31'd988590382;
        8'd76: seed = 31'd982107753;
        8'd77: seed = 31'd975751001;
        8'd78: seed = 31'd969516107;
        8'd79: seed = 31'd963399225;
        8'd80: seed = 31'd957396679;
        8'd81: seed = 31'd951504951;
        8'd82: seed = 31'd945720673;
        8'd83: seed = 31'd940040618;
        8'd84: seed = 31'd934461692;
        8'd85: seed = 31'd928980931;
        8'd86: seed = 31'd923595489;
        8'd87: seed = 31'd918302635;
        8'd88: seed = 31'd913099745;
        8'd89: seed = 31'd907984300;
        8'd90: seed = 31'd902953878;
        8'd91: seed = 31'd898006148;
        8'd92: seed = 31'd893138870;
        8'd93: seed = 31'd888349887;
        8'd94: seed = 31'd883637122;
        8'd95: seed = 31'd878998575;
        8'd96: seed = 31'd874432318;
        8'd97: seed = 31'd869936492;
        8'd98: seed = 31'd865509306;
        8'd99: seed = 31'd861149030;
        8'd100: seed = 31'd856853995;
        8'd101: seed = 31'd852622592;
        8'd102: seed = 31'd848453263;
        8'd103: seed = 31'd844344506;
        8'd104: seed = 31'd840294869;
        8'd105: seed = 31'd836302947;
        8'd106: seed = 31'd832367382;
        8'd107: seed = 31'd828486860;
        8'd108: seed = 31'd824660110;
        8'd109: seed = 31'd820885902;
        8'd110: seed = 31'd817163045;
        8'd111: seed = 31'd813490383;
        8'd112: seed = 31'd809866800;
        8'd113: seed = 31'd806291212;
        8'd114: seed = 31'd802762568;
        8'd115: seed = 31'd799279851;
        8'd116: seed = 31'd795842072;
        8'd117: seed = 31'd792448274;
        8'd118: seed = 31'd789097526;
        8'd119: seed = 31'd785788926;
        8'd120: seed = 31'd782521599;
        8'd121: seed = 31'd779294692;
        8'd122: seed = 31'd776107379;
        8'd123: seed = 31'd772958857;
        8'd124: seed = 31'd769848346;
        8'd125: seed = 31'd766775087;
        8'd126: seed = 31'd763738342;
        8'd127: seed = 31'd760737394;
        8'd128: seed = 31'd757771545;
        8'd129: seed = 31'd754840116;
        8'd130: seed = 31'd751942447;
        8'd131: seed = 31'd749077893;
        8'd132: seed = 31'd746245831;
        8'd133: seed = 31'd743445649;
        8'd134: seed = 31'd740676754;
        8'd135: seed = 31'd737938568;
        8'd136: seed = 31'd735230528;
        8'd137: seed = 31'd732552083;
        8'd138: seed = 31'd729902700;
        8'd139: seed = 31'd727281856;
        8'd140: seed = 31'd724689043;
        8'd141: seed = 31'd722123764;
        8'd142: seed = 31'd719585535;
        8'd143: seed = 31'd717073885;
        8'd144: seed = 31'd714588353;
        8'd145: seed = 31'd712128489;
        8'd146: seed = 31'd709693855;
        8'd147: seed = 31'd707284021;
        8'd148: seed = 31'd704898571;
        8'd149: seed = 31'd702537095;
        8'd150: seed = 31'd700199195;
        8'd151: seed = 31'd697884480;
        8'd152: seed = 31'd695592571;
        8'd153: seed = 31'd693323095;
        8'd154: seed = 31'd691075688;
        8'd155: seed = 31'd688849996;
        8'd156: seed = 31'd686645670;
        8'd157: seed = 31'd684462372;
        8'd158: seed = 31'd682299768;
        8'd159: seed = 31'd680157534;
        8'd160: seed = 31'd678035353;
        8'd161: seed = 31'd675932912;
        8'd162: seed = 31'd673849909;
        8'd163: seed = 31'd671786046;
        8'd164: seed = 31'd669741030;
        8'd165: seed = 31'd667714578;
        8'd166: seed = 31'd665706409;
        8'd167: seed = 31'd663716251;
        8'd168: seed = 31'd661743836;
        8'd169: seed = 31'd659788901;
        8'd170: seed = 31'd657851191;
        8'd171: seed = 31'd655930453;
        8'd172: seed = 31'd654026442;
        8'd173: seed = 31'd652138916;
        8'd174: seed = 31'd650267639;
        8'd175: seed = 31'd648412378;
        8'd176: seed = 31'd646572906;
        8'd177: seed = 31'd644749001;
        8'd178: seed = 31'd642940445;
        8'd179: seed = 31'd641147024;
        8'd180: seed = 31'd639368526;
        8'd181: seed = 31'd637604748;
        8'd182: seed = 31'd635855486;
        8'd183: seed = 31'd634120543;
        8'd184: seed = 31'd632399724;
        8'd185: seed = 31'd630692839;
        8'd186: seed = 31'd628999701;
        8'd187: seed = 31'd627320126;
        8'd188: seed = 31'd625653934;
        8'd189: seed = 31'd624000948;
        8'd190: seed = 31'd622360996;
        8'd191: seed = 31'd620733905;
        8'd192: seed = 31'd619119510;
        8'd193: seed = 31'd617517646;
        8'd194: seed = 31'd615928151;
        8'd195: seed = 31'd614350868;
        8'd196: seed = 31'd612785640;
        8'd197: seed = 31'd611232315;
        8'd198: seed = 31'd609690743;
        8'd199: seed = 31'd608160777;
        8'd200: seed = 31'd606642271;
        8'd201: seed = 31'd605135082;
        8'd202: seed = 31'd603639073;
        8'd203: seed = 31'd602154103;
        8'd204: seed = 31'd600680040;
        8'd205: seed = 31'd599216749;
        8'd206: seed = 31'd597764100;
        8'd207: seed = 31'd596321965;
        8'd208: seed = 31'd594890217;
        8'd209: seed = 31'd593468733;
        8'd210: seed = 31'd592057391;
        8'd211: seed = 31'd590656069;
        8'd212: seed = 31'd589264652;
        8'd213: seed = 31'd587883021;
        8'd214: seed = 31'd586511063;
        8'd215: seed = 31'd585148667;
        8'd216: seed = 31'd583795720;
        8'd217: seed = 31'd582452115;
        8'd218: seed = 31'd581117744;
        8'd219: seed = 31'd579792502;
        8'd220: seed = 31'd578476286;
        8'd221: seed = 31'd577168993;
        8'd222: seed = 31'd575870523;
        8'd223: seed = 31'd574580778;
        8'd224: seed = 31'd573299660;
        8'd225: seed = 31'd572027073;
        8'd226: seed = 31'd570762923;
        8'd227: seed = 31'd569507118;
        8'd228: seed = 31'd568259565;
        8'd229: seed = 31'd567020175;
        8'd230: seed = 31'd565788860;
        8'd231: seed = 31'd564565531;
        8'd232: seed = 31'd563350103;
        8'd233: seed = 31'd562142492;
        8'd234: seed = 31'd560942613;
        8'd235: seed = 31'd559750385;
        8'd236: seed = 31'd558565727;
        8'd237: seed = 31'd557388559;
        8'd238: seed = 31'd556218802;
        8'd239: seed = 31'd555056379;
        8'd240: seed = 31'd553901213;
        8'd241: seed = 31'd552753230;
        8'd242: seed = 31'd551612356;
        8'd243: seed = 31'd550478516;
        8'd244: seed = 31'd549351640;
        8'd245: seed = 31'd548231656;
        8'd246: seed = 31'd547118494;
        8'd247: seed = 31'd546012085;
        8'd248: seed = 31'd544912362;
        8'd249: seed = 31'd543819257;
        8'd250: seed = 31'd542732704;
        8'd251: seed = 31'd541652637;
        8'd252: seed = 31'd540578994;
        8'd253: seed = 31'd539511710;
        8'd254: seed = 31'd538450722;
        8'd255: seed = 31'd537395969;
        default: seed = 31'd1073741823;
    endcase
endfunction

// msb index of a positive value (0 for 0)
function automatic int unsigned msb_acc(input logic [AccWidth-1:0] x);
    msb_acc = 0;
    for (int i = 0; i < AccWidth; i++) if (x[i]) msb_acc = i;
endfunction

function automatic int unsigned msb_len(input logic [LenWidth-1:0] x);
    msb_len = 0;
    for (int i = 0; i < LenWidth; i++) if (x[i]) msb_len = i;
endfunction

enum logic [2:0] {IDLE, NORMALISE, SEED, ISSUE, WAIT, SHIFT, SATURATE, DONE} state_q;

logic [AccWidth-1:0] acc_q;
logic [LenWidth-1:0] len_q;
logic                len_zero_q;
logic [5:0]          k_acc_q, k_len_q;
mant_t               m_acc_q, m_len_q;
recip_t              r_acc_q, r_len_q;
logic [31:0]         t_q;
logic [3:0]          step_q;       // 0..13, see below
logic signed [OutPrecision-1:0] y_q;
logic signed [15:0]  sh_q;         // k_len - k_acc + Shift0 - F, set in SEED
logic [63:0]         y_wide_q;

// One pipelined unsigned 32x32 multiplier.
(* use_dsp = "yes" *) logic [63:0] mul_p [MulLatency];
logic [31:0] mul_a_q, mul_b_q;
logic [MulLatency:0] mul_v;      // mul_v[i] set => operands issued i cycles ago; product valid at [MulLatency]

// Steps 0..5: two Newton steps on acc, 6..11: on len (phase = step % 3);
// 12: m_len * r_len, 13: (that) * r_acc.
wire logic       step_on_len = (step_q >= 6) && (step_q < 12);
wire logic [1:0] phase = 2'(step_q % 3);
wire mant_t      cur_m = step_on_len ? m_len_q : m_acc_q;
wire recip_t     cur_r = step_on_len ? r_len_q : r_acc_q;

logic [31:0] op_a, op_b;
always_comb begin
    op_a = 'x;
    op_b = 'x;
    if (step_q == 12) begin
        op_a = m_len_q;
        op_b = 32'(r_len_q);
    end else if (step_q == 13) begin
        op_a = t_q;
        op_b = 32'(r_acc_q);
    end else if (phase == 0) begin
        op_a = 32'(cur_r);
        op_b = 32'(cur_r);
    end else if (phase == 1) begin
        op_a = cur_m;
        op_b = t_q;
    end else begin
        op_a = 32'(cur_r);
        op_b = t_q;
    end
end

wire logic [63:0] product = mul_p[MulLatency-1];

// final shift: y = p2 * 2^(k_len - k_acc + Shift0 - F), rounded to nearest, saturated
logic [63:0] y_wide;
always_comb begin
    automatic int sh = int'(sh_q);
    y_wide = '0;
    if (sh >= 0) begin
        y_wide = (sh >= 32) ? 64'hFFFF_FFFF_FFFF_FFFF : (64'(t_q) << sh);
        if (sh >= 32 || (y_wide >> sh) != 64'(t_q)) y_wide = 64'hFFFF_FFFF_FFFF_FFFF;
    end else if (-sh < 63) begin
        y_wide = (64'(t_q) + (64'd1 << (-sh - 1))) >> (-sh);
    end
end

always_ff @(posedge clk_i) begin
    mul_a_q <= op_a;
    mul_b_q <= op_b;
    mul_p[0] <= 64'(mul_a_q) * 64'(mul_b_q);
    for (int i = 1; i < MulLatency; i++) mul_p[i] <= mul_p[i-1];
end

always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
        state_q <= IDLE;
        mul_v <= '0;
    end else begin
        mul_v <= {mul_v[MulLatency-1:0], (state_q == ISSUE)};
        case (state_q)
        IDLE: if (in_valid_i) begin
            acc_q <= (acc_i <= 0) ? AccWidth'(1) : acc_i;
            len_q <= len_i;
            len_zero_q <= (len_i == '0);
            state_q <= NORMALISE;
        end
        NORMALISE: begin
            automatic int unsigned ka = msb_acc(acc_q) >> 1;
            automatic int unsigned kl = msb_len(len_q) >> 1;
            k_acc_q <= 6'(ka);
            k_len_q <= 6'(kl);
            m_acc_q <= mant_t'(({acc_q, F'(0)}) >> (2 * ka));
            m_len_q <= mant_t'(({len_q, F'(0)}) >> (2 * kl));
            state_q <= SEED;
        end
        SEED: begin
            r_acc_q <= seed(m_acc_q[31:24]);
            r_len_q <= seed(m_len_q[31:24]);
            sh_q <= 16'(int'(k_len_q) - int'(k_acc_q) + Shift0 - F);
            step_q <= '0;
            state_q <= ISSUE;
        end
        ISSUE: state_q <= WAIT;    // operands captured into mul_a_q/mul_b_q this cycle
        WAIT: if (mul_v[MulLatency]) begin
            if (step_q >= 12) begin
                t_q <= 32'(product >> F);
            end else if (phase == 0) begin
                t_q <= 32'(product >> F);
            end else if (phase == 1) begin
                t_q <= (32'd3 << F) - 32'(product >> F);
            end else if (step_on_len) begin
                r_len_q <= recip_t'(product >> (F + 1));
            end else begin
                r_acc_q <= recip_t'(product >> (F + 1));
            end
            if (step_q == 13) begin
                state_q <= SHIFT;
            end else begin
                step_q <= step_q + 1;
                state_q <= ISSUE;
            end
        end
        SHIFT: begin                // t_q holds p2: shift / round into y_wide_q
            y_wide_q <= y_wide;
            state_q <= SATURATE;
        end
        SATURATE: begin
            y_q <= len_zero_q ? '0
                 : (y_wide_q > OutMax) ? OutPrecision'(OutMax) : OutPrecision'(y_wide_q);
            state_q <= DONE;
        end
        DONE: begin
            if (out_ready_i) state_q <= IDLE;
        end
        default: state_q <= IDLE;
        endcase
    end
end

assign in_ready_o  = (state_q == IDLE);
assign out_valid_o = (state_q == DONE);
assign y_o = y_q;

endmodule
