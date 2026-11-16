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

// ternip_pkg
//
// Ternip enums, the configuration struct, and helper functions.
//
// Import this package anywhere Ternip modules need the common enums or helpers.
// The ternip_assertions module at the bottom provides elaboration-time checks
// for supported configuration combinations.

package ternip_pkg;

// ======================================================= //
// Helper functions for fixed-point and integer operations //
// ======================================================= //
function automatic integer abs_int(integer a);
    return ((a<0) ? -a : a);
endfunction

function automatic integer max_int(integer a, integer b);
    return ((a>b) ? a : b);
endfunction

function automatic integer min_int(integer a, integer b);
    return ((a<b) ? a : b);
endfunction

function automatic integer clamp_int(integer lo, integer x, integer hi);
    return max_int(lo, min_int(x, hi));
endfunction

function automatic real clamp_real(real lo, real x, real hi);
    return (x < lo) ? lo : ((x > hi) ? hi : x);
endfunction

function automatic longint fixed_point_min(integer precision);
    if ((precision < 1) || (precision > $bits(longint)))
        $fatal(1, "fixed_point_min: precision %0d outside representable range [1, %0d]", precision, $bits(longint));
    return -(64'sd1 <<< (precision-1));
endfunction

function automatic longint fixed_point_max(integer precision);
    if ((precision < 1) || (precision > $bits(longint)))
        $fatal(1, "fixed_point_max: precision %0d outside representable range [1, %0d]", precision, $bits(longint));
    return (64'sd1 <<< (precision-1)) - 1;
endfunction

function automatic integer fixed_point_one(integer exponent);
    if ((-exponent < 0) || (-exponent > $bits(integer)-1))
        $fatal(1, "fixed_point_one: exponent %0d yields a shift outside representable range [0, %0d]", exponent, $bits(integer)-1);
    return (1 <<< -exponent);
endfunction

function automatic real fixed_point2real(longint value, integer exponent);
    return value * (2.0 ** exponent);
endfunction

function automatic longint real2fixed_point(real value, integer exponent, integer precision);
    real lowest  = fixed_point2real(fixed_point_min(precision), exponent);
    real highest = fixed_point2real(fixed_point_max(precision), exponent);
    return longint'(clamp_real(lowest, value, highest) * (2.0 ** -exponent));
endfunction

// =========================== //
// Implementation-select enums //
// =========================== //
typedef enum logic [1:0] {
    MUL_BSG,
    MUL_ROUNDROBIN,
    MUL_STAR,
    MUL_NONE
} mul_impl_e;

// Piecewise-linear sigmoid approximations, symmetric about (0, 1/2). Each is the
// minimax fit for its segment count; the POWER2_SLOPE variants restrict every slope to a
// power of two so the multiply degrades to a shift.
typedef enum logic [2:0] {
    SIGMOID_LUT,                                // Max |error|: exact
    SIGMOID_APPROXIMATE_1ST_ORDER,              // Max |error|: 0.056050
    SIGMOID_APPROXIMATE_3RD_ORDER,              // Max |error|: 0.017376
    SIGMOID_APPROXIMATE_5TH_ORDER,              // Max |error|: 0.008362
    SIGMOID_APPROXIMATE_POWER2_SLOPE_1ST_ORDER, // Max |error|: 0.119203  ("hard sigmoid")
    SIGMOID_APPROXIMATE_POWER2_SLOPE_3RD_ORDER, // Max |error|: 0.034857
    SIGMOID_APPROXIMATE_POWER2_SLOPE_5TH_ORDER  // Max |error|: 0.015848
} sigmoid_model_e;

typedef enum logic [1:0] {
    DIV_BSG,
    DIV_ROUNDROBIN,
    DIV_NONE
} div_impl_e;

// Ternary weight storage format of the tmatmul DDR stream.
//  TERNARY_2BIT:    4 trits per byte, ternary_t encoding (01=+1, 11=-1, 00=0).
//  TERNARY_PACKED5: 5 trits per byte, base 3: byte = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4,
//                   digit 0=-1, 1=0, 2=+1. Each beat of TmatmulParallelism trits is
//                   padded to ceil(TmatmulParallelism/5) bytes with digit 1.
//  TERNARY_PAIR4:   2 bits per trit as one 4-bit pair code per lane pair (2q, 2q+1), in the
//                   bits TERNARY_2BIT gives that pair: {nonzero, sign, entry[1:0]}, entry
//                   0..3 = (1,0), (0,1), (1,1), (1,-1), sign negates, (0,0) = 4'b0000. Read
//                   by the pair-table datapath (ternip_tdot_pair_tables), no trit decode.
//  TERNARY_GROUP4 / TERNARY_GROUP5: lanes (Ng .. Ng+N-1) of a beat (N = 4 / 5) as one group code
//                   {sign, index} of group_code_bits(N) = 7 / 8 bits at [bits*g +: bits]. index
//                   selects the sign-folded trit pattern group_pattern(N, index) (0 = all zero),
//                   sign negates it. A beat holds ceil(TP/N) codes padded to whole bytes;
//                   lanes past TP in the last group are 0. Read by ternip_tdot_group_tables.
//  TERNARY_GROUP3:  the same with N = 3: 5-bit codes {sign, index 0..13} (1.67 bits per weight),
//                   13-entry tables (LUTRAM by default, see TmatmulGroupLutram).
typedef enum logic [2:0] {
    TERNARY_2BIT,
    TERNARY_PACKED5,
    TERNARY_PAIR4,
    TERNARY_GROUP4,
    TERNARY_GROUP5,
    TERNARY_GROUP3
} ternary_encoding_e;

// Base-3 unpacking for TERNARY_PACKED5.
typedef enum logic [1:0] {
    TRIT5_DECODE_ARITH,
    TRIT5_DECODE_LUT,
    TRIT5_DECODE_RESIDUE,
    TRIT5_DECODE_DIRECT
} trit5_decoder_e;

// Ternary dot-product arithmetic.
typedef enum logic [2:0] {
    TDOT_TWOS_ADDER_TREE,
    TDOT_ONES_ADDER_TREE,
    TDOT_ONES_CSA,
    TDOT_ONES_DSP_TREE,  // ones'-complement operands, pairwise adds mapped to DSP48 SIMD, popcount correction
    TDOT_ONES_GPC        // ones'-complement operands, K:clog2(K+1) column-counter tree (TmatmulGpcK, TmatmulGpcFinalRows)
} tdot_arith_e;

// Where the ones'-complement correction (+1 per negative weight) is added.
typedef enum logic [1:0] {
    TDOT_CORR_AFTER,
    TDOT_CORR_INJECT_COUNT,
    TDOT_CORR_INJECT_BITS
} tdot_correction_e;

// How ternary weight beats are delivered to BatchSize cores (parent ternip_batched).
//  WEIGHT_DIST_COMB: combinational fanout of the shared DDR stream (baseline).
//  WEIGHT_DIST_PIPE: per-core ready/valid pipeline (local registered copies).
//  WEIGHT_DIST_BRAM: per-core BRAM FIFO holding a full matrix copy.
//  WEIGHT_DIST_TREE: broadcast tree, per-level fanouts WeightTreeShape (default binary).
//  WEIGHT_DIST_RING / RING2 / RING2M / LONG: HPL panel-broadcast topologies
//                    (increasing ring, increasing 2-ring, modified 2-ring, and
//                    bandwidth-reducing scatter + bidirectional roll).
typedef enum logic [2:0] {
    WEIGHT_DIST_COMB,
    WEIGHT_DIST_PIPE,
    WEIGHT_DIST_BRAM,
    WEIGHT_DIST_TREE,
    WEIGHT_DIST_RING,
    WEIGHT_DIST_RING2,
    WEIGHT_DIST_RING2M,
    WEIGHT_DIST_LONG
} weight_dist_e;

// ================= //
// Ternip parameters //
// ================= //
typedef struct packed {
    int unsigned    D;
    int unsigned    TmatmulParallelism;
    int unsigned    VectorParallelism;
    int unsigned    LutParallelism;
    int unsigned    FixedPointPrecision;
    int             FixedPointExponent;
    sigmoid_model_e SigmoidModel;
    int unsigned    BatchSize;
    int unsigned    NumVectorRegisters;
    int unsigned    ImmediateWidth;
    int unsigned    DdrAddressWidth;
    int unsigned    InstructionWidth;
    int unsigned    DdrDataWidth;
    int unsigned    AxiAuxDataWidth;
    int unsigned    InstrFetchWidth;
    int unsigned    CoreInterconnectNumStages;
    mul_impl_e      MultiplicationImplementation;
    div_impl_e      DivisionImplementation;
    bit             EnableRms;
    bit             EnableRowwiseOperation;
} ternip_cfg_t;

// ================================================================================= //
// Instruction fields and widths                                                     //
// The instruction_t type is derived from the user base struct in ternip_types#(Cfg) //
// ================================================================================= //
typedef enum logic [3:0] {
    NOP,
    ADD,
    SUB,
    MUL,
    DIV,
    SIG,
    CSIG,
    SILU
} rowwise_op_e;

typedef enum logic [1:0] {
    NO_LS_OP,
    LDV,
    SV
} loadstore_op_e;

typedef enum logic [1:0] {
    NO_TMATMUL_OP,
    IMPORT,
    GO,
    EXPORT
} tmatmul_op_e;

typedef enum logic [2:0] {
    NO_RMS_OP,
    CLEAR,
    ACCUMULATE,
    FINISH_ACCUMULATE,
    NORM
} rms_op_e;

typedef enum logic [2:0] {
    NO_FU,
    LOADSTORE,
    ROWWISE_OPERATION,
    TMATMUL,
    RMS,
    STALL
} fu_e;

// ==================================================== //
// 2-bit ternary type for ternary matrix multiplication //
// ==================================================== //
typedef logic signed [1:0] ternary_t;

localparam int TritsPerPackedByte = 5;

function automatic integer tmatmul_stream_width(ternary_encoding_e encoding, integer parallelism);
    if (encoding == TERNARY_PACKED5)
        return 8 * ((parallelism + TritsPerPackedByte - 1) / TritsPerPackedByte);
    if (encoding == TERNARY_GROUP4)
        return 8 * ((7 * ((parallelism + 3) / 4) + 7) / 8);
    if (encoding == TERNARY_GROUP5)
        return 8 * ((parallelism + 4) / 5);
    if (encoding == TERNARY_GROUP3)
        return 8 * ((5 * ((parallelism + 2) / 3) + 7) / 8);
    return parallelism * $bits(ternary_t);
endfunction

// lanes per group code of a group-table encoding (0 = not a group encoding)
function automatic integer group_trits(ternary_encoding_e encoding);
    return (encoding == TERNARY_GROUP3) ? 3 : (encoding == TERNARY_GROUP4) ? 4
         : (encoding == TERNARY_GROUP5) ? 5 : 0;
endfunction

// Datapath / weight-distribution knobs (parent passes -D macros; defaults =
// original baseline). Structural config comes from `TERNIP_CFG (JSON → -D).
`ifndef TERNIP_TMATMUL_ENCODING
`define TERNIP_TMATMUL_ENCODING TERNARY_2BIT
`endif
`ifndef TERNIP_TMATMUL_DECODER
`define TERNIP_TMATMUL_DECODER TRIT5_DECODE_LUT
`endif
`ifndef TERNIP_TMATMUL_ARITH
`define TERNIP_TMATMUL_ARITH TDOT_TWOS_ADDER_TREE
`endif
`ifndef TERNIP_TMATMUL_CORRECTION
`define TERNIP_TMATMUL_CORRECTION TDOT_CORR_AFTER
`endif
localparam ternary_encoding_e TmatmulEncoding = `TERNIP_TMATMUL_ENCODING;
localparam trit5_decoder_e    TmatmulDecoder  = `TERNIP_TMATMUL_DECODER;
localparam tdot_arith_e       TmatmulArith    = `TERNIP_TMATMUL_ARITH;
localparam tdot_correction_e  TmatmulCorrection = `TERNIP_TMATMUL_CORRECTION;
// TDOT_ONES_GPC: rows per column counter (3 = the 3:2 CSA) and rows left for the final adder (2 or 3)
`ifndef TERNIP_TMATMUL_GPC_K
`define TERNIP_TMATMUL_GPC_K 6
`endif
`ifndef TERNIP_TMATMUL_GPC_FINAL_ROWS
`define TERNIP_TMATMUL_GPC_FINAL_ROWS 2
`endif
// 1: each group of K rows is one carry-propagate add (K = 3: ternary-adder tree)
`ifndef TERNIP_TMATMUL_GPC_ADD
`define TERNIP_TMATMUL_GPC_ADD 0
`endif
localparam int                TmatmulGpcK         = `TERNIP_TMATMUL_GPC_K;
localparam int                TmatmulGpcFinalRows = `TERNIP_TMATMUL_GPC_FINAL_ROWS;
localparam bit                TmatmulGpcAdd       = `TERNIP_TMATMUL_GPC_ADD;
// 1: the tdot accumulator's in_ready is a register (its credit compare is done a cycle
// early) and the pair tables drive their GO-stage clock enables from replicated
// registers instead of one LUT. Cycle-for-cycle identical to 0.
`ifndef TERNIP_TDOT_REGISTERED_READY
`define TERNIP_TDOT_REGISTERED_READY 0
`endif
localparam bit                TdotRegisteredReady = `TERNIP_TDOT_REGISTERED_READY;
// 1: tmatmul runs IMPORT / EXPORT beside GO (two pair-table banks, two exportvector
// banks; TERNARY_PAIR4 only). Results are those of running the commands in order.
`ifndef TERNIP_TMATMUL_OVERLAP
`define TERNIP_TMATMUL_OVERLAP 0
`endif
localparam bit                TmatmulOverlap      = `TERNIP_TMATMUL_OVERLAP;
// TERNARY_GROUP4 / GROUP5: how many of the group tables' two-group RAM units are UltraRAM
// (the rest are block RAM). Lets a build balance BRAM against URAM.
`ifndef TERNIP_TMATMUL_GROUP_URAM_UNITS
`define TERNIP_TMATMUL_GROUP_URAM_UNITS 0
`endif
localparam int                TmatmulGroupUramUnits = `TERNIP_TMATMUL_GROUP_URAM_UNITS;
// TERNARY_GROUP3: 1 = one 16-entry LUTRAM table per group (simple dual port), 0 = block RAM
// like GROUP4/5. GROUP4/5 always use block RAM / UltraRAM.
`ifndef TERNIP_TMATMUL_GROUP_LUTRAM
`define TERNIP_TMATMUL_GROUP_LUTRAM 1
`endif
localparam bit                TmatmulGroupLutram  = `TERNIP_TMATMUL_GROUP_LUTRAM;

`ifndef TERNIP_WEIGHT_DIST
`define TERNIP_WEIGHT_DIST WEIGHT_DIST_COMB
`endif
`ifndef TERNIP_WEIGHT_PIPE_STAGES
`define TERNIP_WEIGHT_PIPE_STAGES 2
`endif
localparam weight_dist_e WeightDistribution = `TERNIP_WEIGHT_DIST;

// Load/store gearboxes for VP x BatchSize x 16 not a multiple of the DDR width: 1 = gcd chain
// (moves gcd(width, DDR width) bits per cycle; slow when that gcd is small), 0 = staging shifter.
`ifndef TERNIP_LOADSTORE_GEARBOX_GCD
`define TERNIP_LOADSTORE_GEARBOX_GCD 1
`endif

// Elements per cycle in ternip_rms (squarers / normalise multipliers); 0 = VectorParallelism.
// Must divide VectorParallelism. Narrower RMS overlaps with tmatmul streaming (see ternip_rms).
`ifndef TERNIP_RMS_PARALLELISM
`define TERNIP_RMS_PARALLELISM 0
`endif
localparam int RmsParallelism = `TERNIP_RMS_PARALLELISM;
localparam int WeightPipeStages = `TERNIP_WEIGHT_PIPE_STAGES;
// WEIGHT_DIST_TREE shape: per-level fanouts from the root, one hex digit each (1..15, up to 7
// levels), root in the low digit ('h42 = root fans to 2, each of those to 4; pass it to -D as the
// decimal, 66). 0 = binary, clog2(BatchSize) levels.
`ifndef TERNIP_WEIGHT_TREE_SHAPE
`define TERNIP_WEIGHT_TREE_SHAPE 0
`endif
localparam longint unsigned WeightTreeShape = `TERNIP_WEIGHT_TREE_SHAPE;

// ---- BEGIN generated by bench/tdot/group/gen_group_walk.py (do not edit) ----
// N-trit group tables (TERNARY_GROUP4 / TERNARY_GROUP5, ternip_tdot_group_tables).
// Table index i holds the activation sum of pattern i; index 0 is the zero pattern.
// Step s (1..entries) builds index s from index s-1: field [4*(s-1) +: 4] = {up, k[2:0]},
// entry(s) = entry(s-1) + (up ? act[k] : -act[k]). Pattern i: [2*N*i +: 2*N], trit k
// (lane k of the group) as a ternary_t at bits [2k +: 2].
localparam int GroupEntries3 = 13;
localparam logic [51:0] GroupWalk3 = 52'h89001188a0098;
localparam logic [83:0] GroupPattern3 = 84'h4137dc7515545c7105040;
localparam int GroupEntries4 = 40;
localparam logic [159:0] GroupWalk4 = 160'h0188921009898a002288110098b89001188a0098;
localparam logic [327:0] GroupPattern4 = 328'h7071757477734347444541404c4f7f7c7d4d5d5c5f53575455515010131f1c1d111514170704050100;
localparam int GroupEntries5 = 121;
localparam logic [483:0] GroupWalk5 = 484'h01889900a88110098b01889900288110098201889900388110098301889a1009898200aa88110098c0188921009898a002288110098b89001188a0098;
localparam logic [1219:0] GroupPattern5 = 1220'h401014150441d0343d0c4353d4f13f4cd374d1354c5307c1f17d5f47ddf37fdfc7f5cd731cf70dc7711c5705c0741d1755d475dd377ddc7751d4711f44d174511544510541515555455d5350d4751145505405314f57d5c5754d5f57c5fd735dd745d5715c0701c4751d0771cc4311c44114411004c13c7f1f07d1345d1705f14c5715055144500401307c1c074110541405c070100500400;
function automatic int group_entries(int n);
    return (n == 3) ? GroupEntries3 : (n == 4) ? GroupEntries4 : GroupEntries5;
endfunction
// bits per group code {sign, index}
function automatic int group_code_bits(int n);
    return 1 + $clog2(group_entries(n) + 1);
endfunction
// {up, k} of build step s (1..group_entries(n))
function automatic logic [3:0] group_walk_step(int n, int s);
    return (n == 3) ? GroupWalk3[4*(s-1) +: 4] : (n == 4) ? GroupWalk4[4*(s-1) +: 4] : GroupWalk5[4*(s-1) +: 4];
endfunction
// trits of table index i, lane k at [2k +: 2]
function automatic logic [9:0] group_pattern(int n, int i);
    return (n == 3) ? 10'(GroupPattern3[6*i +: 6]) : (n == 4) ? 10'(GroupPattern4[8*i +: 8]) : GroupPattern5[10*i +: 10];
endfunction
// ---- END generated group walk ----

endpackage : ternip_pkg

/* verilator lint_save */
/* verilator lint_off DECLFILENAME */
module ternip_assertions; import ternip_pkg::*;

// The instruction_t width check lives in ternip_types_assertions: instruction_t
// now lives only in ternip_types#(Cfg), which is declared after this package.
localparam ternip_cfg_t Cfg = `TERNIP_CFG;
if (!(Cfg.FixedPointPrecision inside {8, 16})) $fatal(0, "Invalid value for FixedPointPrecision: %0d.", Cfg.FixedPointPrecision);

endmodule
/* verilator lint_restore */
