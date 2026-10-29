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
typedef enum logic [1:0] {
    TERNARY_2BIT,
    TERNARY_PACKED5,
    TERNARY_PAIR4
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
    return parallelism * $bits(ternary_t);
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

`ifndef TERNIP_WEIGHT_DIST
`define TERNIP_WEIGHT_DIST WEIGHT_DIST_COMB
`endif
`ifndef TERNIP_WEIGHT_PIPE_STAGES
`define TERNIP_WEIGHT_PIPE_STAGES 2
`endif
localparam weight_dist_e WeightDistribution = `TERNIP_WEIGHT_DIST;

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
