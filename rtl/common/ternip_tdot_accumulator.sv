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

// ternip_tdot_accumulator
//
// Pipelined ternary dot product plus running accumulator.
//
// Each accepted beat computes sum_i w_i * act_i, where lane i has weight +1
// (in_pos_i[i]), -1 (in_neg_i[i]) or 0, and adds it into an accumulator. The
// handshake, accumulator width, per-beat accumulator saturation and final clip to
// result_t match ternip_multioperand_accumulator with NEXT_STAGE_FANIN=2 and
// operand_t = logic signed [ACT_WIDTH:0], so every Arith setting is a drop-in
// replacement for the ternip_tmatmul baseline datapath. The output FIFO is one
// entry deeper (see output_fifo), and CSA / GPC pipelines may be shallower.
//
// Arith:
//  TDOT_TWOS_ADDER_TREE: +/-act two's-complement products into
//      ternip_multioperand_accumulator (the existing datapath).
//  TDOT_ONES_ADDER_TREE / TDOT_ONES_CSA: operand q_i = act_i, ~act_i or 0
//      after sign extension to the reduction width W, reduced with a fan-in-2
//      adder tree or a 3:2 carry-save tree. Since -x == ~x + 1 modulo 2^W,
//      sum_i w_i*act_i == sum_i q_i + (number of negative weights).
//  TDOT_ONES_GPC: the CSA generalised to GPC_K rows per group. Each group is
//      reduced column by column with a GPC_K-input counter whose M = clog2(GPC_K+1)
//      output bits become M rows, row t shifted left by t (a K:M compressor; K=3
//      is the 3:2 CSA). A leftover group of r >= 3 rows uses an r-input counter,
//      fewer rows pass through. Levels repeat until GPC_FINAL_ROWS rows remain,
//      which one carry-propagate adder (2 rows) or ternary adder (3 rows) sums.
//      GPC_ADD = 1 sums each group with a carry-propagate add instead (one row per
//      group, a leftover of >= 2 rows summed too); GPC_K = 3 is a ternary-adder tree.
//
// Correction (ones'-complement modes only) chooses where that +1 per negative
// weight enters:
//  TDOT_CORR_AFTER:        pipelined popcount added in the final stage.
//  TDOT_CORR_INJECT_COUNT: pipelined popcount injected as an extra CSA row (CSA only).
//  TDOT_CORR_INJECT_BITS:  each negative bit fills a free slot: an adder carry-in,
//                          the empty LSB of a carry-save carry vector, the final
//                          carry-propagate carry-in, or the accumulator carry-in.
//                          GPC: the empty LSBs of counter rows 1..M-1 (GPC_ADD: each
//                          group add's carry-in); negative bits left without a slot
//                          enter as extra 1-bit operand rows.
//
// PACKED_BYTES > 0 (ternip_pkg::TRIT5_DECODE_DIRECT, ones'-complement modes only)
// takes the weights as base-3 packed bytes on in_packed_i instead of in_pos_i/in_neg_i:
// lane i uses digit (LANE_OFFSET+i)%5 of byte (LANE_OFFSET+i)/5, and each operand bit
// comes straight from {activation bit, byte} (see ternip_tdot_packed5_operands).
//
// Reduction width W = ACT_WIDTH + 1 + NumStages (NumStages = clog2(NUM_LANES)):
// |sum_i w_i*act_i| <= NUM_LANES * 2^(ACT_WIDTH-1) <= 2^(W-2), so the beat
// result is exact in W bits, and all intermediate sums are taken modulo 2^W.

`define SIGN_EXTEND(X, FROM, TO) (TO'({ {$bits(TO){X[$bits(FROM)-1]}}, X }))

module ternip_tdot_accumulator #(
    parameter int NUM_LANES = 16,
    parameter int ACT_WIDTH = 8,
    parameter type result_t = logic signed [7:0],
    parameter ternip_pkg::tdot_arith_e Arith = ternip_pkg::TDOT_ONES_CSA,
    parameter ternip_pkg::tdot_correction_e Correction = ternip_pkg::TDOT_CORR_INJECT_BITS,
    parameter int GPC_K = 6,
    parameter int GPC_FINAL_ROWS = 2,
    parameter bit GPC_ADD = 0,
    parameter int D = 1,
    parameter int PACKED_BYTES = 0,
    parameter int LANE_OFFSET = 0,

    localparam type act_t = logic signed [ACT_WIDTH-1:0],
    localparam int PackedWidth = (PACKED_BYTES > 0) ? 8*PACKED_BYTES : 1
) (
    input  logic                     clk_i,
    input  logic                     rst_ni,

    output logic                     in_ready_o,
    input  logic                     in_valid_i,
    input  logic                     in_final_i,
    input  logic [NUM_LANES-1:0]     in_pos_i,
    input  logic [NUM_LANES-1:0]     in_neg_i,
    input  logic [PackedWidth-1:0]   in_packed_i,
    input  act_t [NUM_LANES-1:0]     in_act_i,

    input  logic                     out_ready_i,
    output logic                     out_valid_o,
    output result_t                  out_result_o
);

// Elaboration-time structure of the ones'-complement datapath

localparam bit IsCsa = (Arith == ternip_pkg::TDOT_ONES_CSA);
localparam bit IsGpc = (Arith == ternip_pkg::TDOT_ONES_GPC);
// DSP tree: DSP48 SIMD lanes have no per-lane carry-in to inject into, so the
// correction is always the pipelined popcount added at the end.
localparam bit IsDsp = (Arith == ternip_pkg::TDOT_ONES_DSP_TREE);
localparam bit CorrAfter = IsDsp || (Correction == ternip_pkg::TDOT_CORR_AFTER);
localparam bit CorrCount = !IsDsp && (Correction == ternip_pkg::TDOT_CORR_INJECT_COUNT);
localparam bit CorrBits = !IsDsp && (Correction == ternip_pkg::TDOT_CORR_INJECT_BITS);

localparam int N = NUM_LANES;
localparam int BaselineStages = (N > 1) ? $clog2(N) : 1;
localparam int W = ACT_WIDTH + 1 + BaselineStages;
localparam int AccumulatorWidth = W + $clog2(D);
localparam int CountWidth = $clog2(N+1);

typedef logic [W-1:0] row_t;
typedef logic signed [W-1:0] pipeline_data_t;
typedef logic signed [AccumulatorWidth-1:0] accumulator_t;
typedef logic [CountWidth-1:0] count_t;

// Reduction levels: combinational steps between and within pipeline stages.
typedef enum logic [2:0] {
    LT_PASS,      // 1 row -> 1 row
    LT_ADD,       // pairwise add, odd row passes through
    LT_CSA,       // 3:2 compression of row triples, leftovers pass through
    LT_CSA_COUNT, // append popcount row, then 3:2 if three rows
    LT_CPA,       // final carry-propagate add of the carry-save pair
    LT_ADD_COUNT, // add popcount to the single row
    LT_GPC        // GPC_K-row column counters, leftovers counted (>= 3) or passed through
} level_type_e;

function automatic int num_csa_levels();
    int n = N;
    int c = 0;
    while (n > 2) begin
        n = 2*(n/3) + (n%3);
        c++;
    end
    return c;
endfunction

// One GPC level on n rows: groups of GPC_K rows -> clog2(GPC_K+1) rows each
function automatic int gpc_rows_after(int n);
    int r = n % GPC_K;
    if (GPC_ADD) return (n / GPC_K) + ((r >= 2) ? 1 : r);
    return (n / GPC_K) * $clog2(GPC_K + 1) + ((r >= 3) ? $clog2(r + 1) : r);
endfunction

// free carry-row LSBs of one GPC level (rows 1..M-1 of every counter)
function automatic int gpc_slots(int n);
    int r = n % GPC_K;
    if (GPC_ADD) return (n / GPC_K) + ((r >= 2) ? 1 : 0);
    return (n / GPC_K) * ($clog2(GPC_K + 1) - 1) + ((r >= 3) ? $clog2(r + 1) - 1 : 0);
endfunction

function automatic int num_gpc_levels(int n);
    int c = 0;
    while (n > GPC_FINAL_ROWS) begin
        n = gpc_rows_after(n);
        c++;
    end
    return c;
endfunction

// final adder slot: the carry-in of a 2-row add (a ternary add keeps none)
function automatic int final_add_slots(int n);
    return (n == 3) ? 0 : 1;
endfunction

// GPC slots for n operand rows: counter LSBs, final adder, accumulator carry-in
function automatic int gpc_total_slots(int n);
    int s = 1;
    while (n > GPC_FINAL_ROWS) begin
        s += gpc_slots(n);
        n = gpc_rows_after(n);
    end
    return s + final_add_slots(n);
endfunction

// TDOT_CORR_INJECT_BITS with GPC: negative bits beyond the slots become 1-bit rows
// appended to the operands (they use bits [0, ExtraRows), the slots the rest)
function automatic int gpc_extra_rows();
    if (!IsGpc || !CorrBits) return 0;
    for (int e = 0; e < N; e++) if (gpc_total_slots(N + e) + e >= N) return e;
    return N;
endfunction

localparam int ExtraRows = gpc_extra_rows();
localparam int NumRows0 = N + ExtraRows;

localparam int NumAddLevels = (N > 1) ? $clog2(N) : 0;
localparam int NumCsaLevels = num_csa_levels() + (CorrCount ? 1 : 0);
localparam int NumGpcLevels = num_gpc_levels(NumRows0);
localparam int GpcSpreadLevels = (NumGpcLevels > 1) ? NumGpcLevels - 1 : 1;

function automatic int num_levels();
    if (IsCsa) return NumCsaLevels + 1 + (CorrAfter ? 1 : 0);
    if (IsGpc) return NumGpcLevels + 1 + (CorrAfter ? 1 : 0);
    if (NumAddLevels + (CorrAfter ? 1 : 0) == 0) return 1;
    return NumAddLevels + (CorrAfter ? 1 : 0);
endfunction

localparam int NumLevels = num_levels();

function automatic level_type_e level_type(int level);
    if (IsCsa) begin
        if (CorrCount && level == NumCsaLevels-1) return LT_CSA_COUNT;
        if (level < NumCsaLevels) return LT_CSA;
        if (level == NumCsaLevels) return LT_CPA;
        return LT_ADD_COUNT;
    end
    if (IsGpc) begin
        if (level < NumGpcLevels) return LT_GPC;
        if (level == NumGpcLevels) return LT_CPA;
        return LT_ADD_COUNT;
    end
    if (level < NumAddLevels) return LT_ADD;
    if (CorrAfter) return LT_ADD_COUNT;
    return LT_PASS;
endfunction

function automatic int rows_after(level_type_e t, int n);
    case (t)
        LT_ADD:       return (n + 1) / 2;
        LT_CSA:       return 2*(n/3) + (n%3);
        LT_CSA_COUNT: return 2;
        LT_GPC:       return gpc_rows_after(n);
        default:      return 1;
    endcase
endfunction

function automatic int rows_at(int level);
    int n = NumRows0;
    for (int l = 0; l < level; l++) n = rows_after(level_type(l), n);
    return n;
endfunction

// carry-in / carry-LSB slots offered by one level
function automatic int slots_in(level_type_e t, int n);
    case (t)
        LT_ADD: return n / 2;
        LT_CSA: return n / 3;
        LT_GPC: return gpc_slots(n);
        LT_CPA: return final_add_slots(n);
        default: return 0;
    endcase
endfunction

function automatic int slot_base(int level);
    int s = 0;
    for (int l = 0; l < level; l++) s += slots_in(level_type(l), rows_at(l));
    return s;
endfunction

localparam int AccumulatorSlot = slot_base(NumLevels);

function automatic int max_rows();
    int m = 0;
    for (int l = 0; l <= NumLevels; l++) if (rows_at(l) > m) m = rows_at(l);
    return m;
endfunction

localparam int MaxRows = max_rows();

// Pipeline registers: adder tree registers after every add (the popcount add
// shares the last stage); CSA / GPC spread their compressor levels over
// BaselineStages-1 stages and give the final add (plus popcount add) the last one.
function automatic bit reg_after(int level);
    level_type_e t = level_type(level);
    if (level == NumLevels-1) return 1'b1;
    if (t == LT_ADD) return IsDsp || !(CorrAfter && level == NumAddLevels-1);
    if (t == LT_CSA || t == LT_CSA_COUNT)
        return (((level + 1) * (BaselineStages - 1)) / NumCsaLevels)
            != ((level * (BaselineStages - 1)) / NumCsaLevels);
    // GPC: the first level (the widest row set) always shares a stage with the
    // second; the registers spread over levels 1..NumGpcLevels-1
    if (t == LT_GPC && GPC_ADD)
        return (((level + 1) * (BaselineStages - 1)) / NumGpcLevels)
            != ((level * (BaselineStages - 1)) / NumGpcLevels);
    if (t == LT_GPC)
        return (level > 0) && (((level * (BaselineStages - 1)) / GpcSpreadLevels)
            != (((level - 1) * (BaselineStages - 1)) / GpcSpreadLevels));
    return 1'b0;
endfunction

function automatic int stage_of_level(int level);
    int s = 0;
    for (int l = 0; l < level; l++) s += int'(reg_after(l));
    return s;
endfunction

function automatic int count_level();
    for (int l = 0; l < NumLevels; l++) begin
        if (level_type(l) == LT_CSA_COUNT || level_type(l) == LT_ADD_COUNT) return l;
    end
    return 0;
endfunction

localparam int NumStages = stage_of_level(NumLevels);
localparam int CountStages = stage_of_level(count_level());

function automatic logic slot_bit(logic [N-1:0] bits, int index);
    if (!CorrBits || ExtraRows + index >= N) return 1'b0;
    return bits[ExtraRows + index];
endfunction

if (Arith == ternip_pkg::TDOT_TWOS_ADDER_TREE) begin : twos

    `ifndef SYNTHESIS
    if (PACKED_BYTES > 0)
        $fatal(0, "PACKED_BYTES (TRIT5_DECODE_DIRECT) needs a ones'-complement Arith");
    `endif

    typedef logic signed [ACT_WIDTH:0] product_t;
    product_t [NUM_LANES-1:0] products;

    always_comb begin
        for (int i = 0; i < NUM_LANES; i++) begin
            products[i] = '0;
            if (in_pos_i[i]) products[i] = `SIGN_EXTEND(in_act_i[i], act_t, product_t);
            else if (in_neg_i[i]) products[i] = -`SIGN_EXTEND(in_act_i[i], act_t, product_t);
        end
    end

    ternip_multioperand_accumulator #(
        .operand_t(product_t),
        .result_t(result_t),
        .NUM_OPERANDS(NUM_LANES),
        .NEXT_STAGE_FANIN(2),
        .D(D)
    ) multioperand_accumulator (
        .clk_i,
        .rst_ni,
        .in_ready_o,
        .in_valid_i,
        .in_final_i,
        .in_operands_i(products),
        .out_ready_i,
        .out_valid_o,
        .out_result_o
    );

end else begin : ones

    `ifndef SYNTHESIS
    if (!IsCsa && CorrCount)
        $fatal(0, "TDOT_CORR_INJECT_COUNT is only supported with TDOT_ONES_CSA");
    if (IsGpc && (GPC_K < (GPC_ADD ? 2 : 3) || GPC_FINAL_ROWS < 2 || GPC_FINAL_ROWS > 3))
        $fatal(0, "TDOT_ONES_GPC needs GPC_K >= 3 (2 with GPC_ADD) and GPC_FINAL_ROWS of 2 or 3");
    `endif

    // q_i = act_i, ~act_i or 0, sign extended to W bits; lane_neg = negative weights
    row_t [N-1:0] operands;
    logic [N-1:0] lane_neg;
    if (PACKED_BYTES > 0) begin : g_packed_operands
        ternip_tdot_packed5_operands #(
            .NUM_LANES(N),
            .ACT_WIDTH(ACT_WIDTH),
            .ROW_WIDTH(W),
            .NUM_BYTES(PACKED_BYTES),
            .LANE_OFFSET(LANE_OFFSET)
        ) packed_operands (
            .packed_i(in_packed_i),
            .act_i(in_act_i),
            .operands_o(operands),
            .neg_o(lane_neg),
            .invalid_o()
        );
    end else begin : g_flag_operands
        always_comb begin
            for (int i = 0; i < N; i++) begin
                operands[i] = '0;
                if (in_pos_i[i]) operands[i] = `SIGN_EXTEND(in_act_i[i], act_t, row_t);
                else if (in_neg_i[i]) operands[i] = ~`SIGN_EXTEND(in_act_i[i], act_t, row_t);
            end
        end
        assign lane_neg = in_neg_i;
    end

    // Negative-weight count for TDOT_CORR_AFTER / TDOT_CORR_INJECT_COUNT
    count_t count;
    if (CorrAfter || CorrCount) begin : g_popcount
        ternip_tdot_popcount #(
            .NUM_BITS(N),
            .NUM_STAGES(CountStages)
        ) popcount (
            .clk_i,
            .bits_i(lane_neg),
            .count_o(count)
        );
    end else begin : g_no_popcount
        assign count = '0;
    end

    // Rows between levels live at module scope (no cross-generate-block
    // references, which some synthesis front ends mis-resolve).
    row_t [NumLevels:0][MaxRows-1:0] level_rows;
    logic [NumLevels:0][N-1:0] level_neg;

    assign level_rows[0][N-1:0] = operands;
    for (genvar e = 0; e < ExtraRows; e++) begin : g_extra_rows
        assign level_rows[0][N + e] = row_t'(lane_neg[e]);
    end
    if (MaxRows > NumRows0) begin : g_pad_operands
        assign level_rows[0][MaxRows-1:NumRows0] = '0;
    end
    assign level_neg[0] = lane_neg;

    // Reduction datapath — always advances, no stalling
    for (genvar l = 0; l < NumLevels; l++) begin : level
        localparam level_type_e Type = level_type(l);
        localparam int NIn = rows_at(l);
        localparam int NOut = rows_at(l + 1);
        localparam int SlotBase = slot_base(l);

        row_t [NIn-1:0] rows_in;
        row_t [NOut-1:0] rows_comb, rows_out;
        logic [N-1:0] neg_in, neg_out;

        assign rows_in = level_rows[l][NIn-1:0];
        assign neg_in = level_neg[l];
        assign level_rows[l+1][NOut-1:0] = rows_out;
        if (NOut < MaxRows) begin : g_pad
            assign level_rows[l+1][MaxRows-1:NOut] = '0;
        end
        assign level_neg[l+1] = neg_out;

        if (Type == LT_ADD && IsDsp) begin : g_dsp_add
            // registered pairwise adds, one DSP48 SIMD lane each (rows_comb is the register)
            ternip_tdot_dsp_adds #(
                .W(W),
                .NIN(NIn),
                .NOUT(NOut)
            ) adds (
                .clk_i,
                .rows_i(rows_in),
                .sums_o(rows_comb)
            );
        end else if (Type == LT_ADD) begin : g_add
            always_comb begin
                for (int j = 0; j < NOut; j++) begin
                    rows_comb[j] = rows_in[2*j];
                    if (2*j + 1 < NIn)
                        rows_comb[j] = rows_in[2*j] + rows_in[2*j + 1] + row_t'(slot_bit(neg_in, SlotBase + j));
                end
            end
        end else if (Type == LT_CSA || Type == LT_CSA_COUNT) begin : g_csa
            localparam int NRows = (Type == LT_CSA_COUNT) ? (NIn + 1) : NIn;
            localparam int NGroups = NRows / 3;
            row_t [NRows-1:0] rows;
            always_comb begin
                for (int r = 0; r < NIn; r++) rows[r] = rows_in[r];
                if (Type == LT_CSA_COUNT) rows[NRows-1] = row_t'(count);
                for (int g = 0; g < NGroups; g++) begin
                    rows_comb[2*g] = rows[3*g] ^ rows[3*g + 1] ^ rows[3*g + 2];
                    rows_comb[2*g + 1] = {
                        (rows[3*g][W-2:0] & rows[3*g + 1][W-2:0])
                        | (rows[3*g][W-2:0] & rows[3*g + 2][W-2:0])
                        | (rows[3*g + 1][W-2:0] & rows[3*g + 2][W-2:0]),
                        slot_bit(neg_in, SlotBase + g)
                    };
                end
                for (int r = 0; r < NRows % 3; r++) rows_comb[2*NGroups + r] = rows[3*NGroups + r];
            end
        end else if (Type == LT_GPC && GPC_ADD) begin : g_gpc_add
            // group g (and a leftover of >= 2 rows, g == NGroups) -> row g, carry-in last
            localparam int NGroups = NIn / GPC_K;
            localparam int Rem = NIn % GPC_K;
            localparam int NAdds = NGroups + ((Rem >= 2) ? 1 : 0);
            always_comb begin
                for (int g = 0; g < NAdds; g++) begin
                    automatic int k = (g < NGroups) ? GPC_K : Rem;
                    rows_comb[g] = rows_in[g*GPC_K];
                    for (int i = 1; i < k; i++) rows_comb[g] += rows_in[g*GPC_K + i];
                    rows_comb[g] += row_t'(slot_bit(neg_in, SlotBase + g));
                end
                if (Rem == 1) rows_comb[NOut - 1] = rows_in[NIn - 1];
            end
        end else if (Type == LT_GPC) begin : g_gpc
            // group g (and the leftover counter, g == NGroups) writes rows g*M .. g*M+m-1
            localparam int M = $clog2(GPC_K + 1);
            localparam int NGroups = NIn / GPC_K;
            localparam int Rem = NIn % GPC_K;
            localparam int NCounters = NGroups + ((Rem >= 3) ? 1 : 0);
            always_comb begin
                rows_comb = '0;
                for (int g = 0; g < NCounters; g++) begin
                    automatic int k = (g < NGroups) ? GPC_K : Rem;
                    automatic int m = $clog2(k + 1);
                    for (int c = 0; c < W; c++) begin
                        automatic logic [M-1:0] count_c = '0;
                        for (int i = 0; i < k; i++) count_c += M'(rows_in[g*GPC_K + i][c]);
                        for (int t = 0; t < m; t++)
                            if (c + t < W) rows_comb[g*M + t][c + t] = count_c[t];
                    end
                    for (int t = 1; t < m; t++)
                        rows_comb[g*M + t][0] = slot_bit(neg_in, SlotBase + g*(M-1) + t - 1);
                end
                if (Rem < 3)
                    for (int r = 0; r < Rem; r++) rows_comb[NGroups*M + r] = rows_in[NGroups*GPC_K + r];
            end
        end else if (Type == LT_CPA && NIn == 3) begin : g_ternary
            assign rows_comb[0] = rows_in[0] + rows_in[1] + rows_in[NIn - 1];
        end else if (Type == LT_CPA) begin : g_cpa
            always_comb begin
                rows_comb[0] = rows_in[0] + row_t'(slot_bit(neg_in, SlotBase));
                if (NIn > 1) rows_comb[0] += rows_in[NIn > 1 ? 1 : 0];
            end
        end else if (Type == LT_ADD_COUNT) begin : g_add_count
            assign rows_comb[0] = rows_in[0] + row_t'(count);
        end else begin : g_pass
            assign rows_comb[0] = rows_in[0];
        end

        if (reg_after(l) && Type == LT_ADD && IsDsp) begin : g_dsp_reg
            // the add module already registered the rows
            assign rows_out = rows_comb;
            always_ff @(posedge clk_i) neg_out <= neg_in;
        end else if (reg_after(l)) begin : g_reg
            always_ff @(posedge clk_i) begin
                rows_out <= rows_comb;
                neg_out <= neg_in;
            end
        end else begin : g_comb
            assign rows_out = rows_comb;
            assign neg_out = neg_in;
        end
    end

    wire pipeline_data_t pipeline_data = pipeline_data_t'(level_rows[NumLevels][0]);
    wire logic pipeline_carry_in = slot_bit(level_neg[NumLevels], AccumulatorSlot);

    // final_pipeline: NumStages+1 bits — tracks in_final_i; extra bit lets last data accumulate
    // valid_pipeline: NumStages bits — tracks in_valid_i, aligned with the last level output
    logic [NumStages:0] final_pipeline_d, final_pipeline_q;
    logic [NumStages-1:0] valid_pipeline_d, valid_pipeline_q;
    accumulator_t accumulator_d, accumulator_q;

    localparam accumulator_t AccumulatorMin = (1 << (AccumulatorWidth-1));
    localparam accumulator_t AccumulatorMax = (AccumulatorMin - 1);

    enum logic [1:0] {
        STARTING_FLUSH,
        FLUSHING,
        WORKING
    } state_d, state_q;

    wire pipeline_data_valid = valid_pipeline_q[NumStages-1];

    // Output FIFO signals
    localparam logic ResultIsSigned = ( result_t'('1) < 0 );
    localparam result_t ResultMin = ResultIsSigned ? (1 << ($bits(result_t)-1)) : ('0);
    localparam result_t ResultMax = ResultMin-1;

    result_t clipped_result;
    always_comb begin
        clipped_result = accumulator_q;
        if (accumulator_q > ResultMax) clipped_result = ResultMax;
        if (accumulator_q < ResultMin) clipped_result = ResultMin;
    end

    logic                       fifo_in_ready;
    logic                       fifo_in_valid;
    logic [$bits(result_t)-1:0] fifo_in_data;

    logic                       fifo_out_ready;
    logic                       fifo_out_valid;
    logic [$bits(result_t)-1:0] fifo_out_data;

    assign fifo_in_data = clipped_result;

    logic    out_valid_d, out_valid_q;
    result_t out_result_d, out_result_q;

    assign fifo_out_ready = !out_valid_q || out_ready_i;

    assign out_valid_o = out_valid_q;
    always_comb begin
        out_result_o = out_result_q;
        if (!out_valid_q) out_result_o = 'x;
    end

    always_comb begin
        out_valid_d = out_valid_q;
        if (fifo_out_ready)
            out_valid_d = fifo_out_valid;
    end
    always_comb begin
        out_result_d = out_result_q;
        if (fifo_out_ready && fifo_out_valid)
            out_result_d = result_t'(fifo_out_data);
    end

    // State and control
    always_comb begin
        state_d = state_q;
        in_ready_o = 0;
        fifo_in_valid = 0;
        valid_pipeline_d = {valid_pipeline_q, 1'b0};
        final_pipeline_d = {final_pipeline_q, 1'b0};

        if (state_q == STARTING_FLUSH) begin
            final_pipeline_d = {final_pipeline_q, 1'b1};
            state_d = FLUSHING;
        end else if (state_q == FLUSHING) begin
            if (final_pipeline_q[NumStages])
                state_d = WORKING;
        end else if (state_q == WORKING) begin
            in_ready_o = !out_valid_q;
            fifo_in_valid = final_pipeline_q[NumStages];
            valid_pipeline_d = {valid_pipeline_q, (in_ready_o && in_valid_i)};
            final_pipeline_d = {final_pipeline_q, (in_ready_o && in_valid_i && in_final_i)};
        end
    end

    // Accumulator — gated by pipeline_data_valid at the narrow output
    typedef logic signed [AccumulatorWidth:0] unclipped_t;
    always_comb begin
        automatic pipeline_data_t pipeline_addend = pipeline_data_valid ? pipeline_data : '0;
        automatic unclipped_t unclipped = `SIGN_EXTEND(pipeline_addend, pipeline_data_t, unclipped_t)
                                        + unclipped_t'(pipeline_data_valid && pipeline_carry_in);
        // final_pipeline_q[NumStages] clears accumulator_q for the next accumulation
        if (!final_pipeline_q[NumStages]) unclipped += `SIGN_EXTEND(accumulator_q, accumulator_t, unclipped_t);
        accumulator_d = unclipped;
        if (unclipped > AccumulatorMax) accumulator_d = AccumulatorMax;
        if (unclipped < AccumulatorMin) accumulator_d = AccumulatorMin;
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            final_pipeline_q <= '0;
            valid_pipeline_q <= '0;
            state_q <= STARTING_FLUSH;
        end else begin
            final_pipeline_q <= final_pipeline_d;
            valid_pipeline_q <= valid_pipeline_d;
            state_q <= state_d;
        end
    end
    always_ff @(posedge clk_i) begin
        accumulator_q <= accumulator_d;
        `ifndef SYNTHESIS
        if (!rst_ni) accumulator_q <= 'x;
        `endif
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni)
            out_valid_q <= 1'b0;
        else
            out_valid_q <= out_valid_d;
    end
    always_ff @(posedge clk_i) begin
        out_result_q <= out_result_d;
        `ifndef SYNTHESIS
        if (!rst_ni) out_result_q <= 'x;
        `endif
    end

    // https://github.com/bespoke-silicon-group/basejump_stl/blob/a43571d2/bsg_dataflow/bsg_fifo_1r1w_small.sv
    // in_ready_o falls NumStages+3 cycles after the first accepted final beat, so
    // back-to-back one-beat groups against a stalled output leave NumStages+3
    // results in flight: NumStages+2 here plus the output register.
    // (ternip_multioperand_accumulator uses NumStages+1, one short.)
    bsg_fifo_1r1w_small #(
        .width_p($bits(result_t)),
        .els_p(NumStages + 2),
        .harden_p(0),
        .ready_THEN_valid_p(0)
    ) output_fifo (
        .clk_i,
        .reset_i(!rst_ni),

        .v_i(fifo_in_valid),
        .ready_param_o(fifo_in_ready),
        .data_i(fifo_in_data),

        .v_o(fifo_out_valid),
        .data_o(fifo_out_data),
        .yumi_i(fifo_out_ready && fifo_out_valid)
    );

    assert property (@(posedge clk_i) disable iff (!rst_ni) fifo_in_valid |-> fifo_in_ready)
        else $fatal(0, "Output FIFO overflow");

end

endmodule

`undef SIGN_EXTEND
