//! The prompt path's kernel switches against the baseline kernels, on the real kernels and random data: every switch's
//! launches must give the baseline bytes, and a control (one input changed) must differ.
//!   - TF_DS_PF_SPLIT: mx.gemm of the split FP8 GEMMs' shapes (wq_a, wkv, wq_b, wo_b, the indexer's wq_b) with each
//!     rows, fp32 and accumulated outputs at four row counts, weight scales on the fast path and off it; rows 37 ..
//!     1,036 of one launch against a launch of those rows alone; the fold kernel against the slab kernels and their
//!     reduce (TF_DS_PF_TILES=0's path); a shape whose split a configuration does not divide takes the fold kernel.
//!   - TF_DS_WOA_W8: mx.woA8Prefill (wo_a from its FP8 rows) against mx.woA (o's quant-dequant rows through the
//!     admitted (code, scale byte) pair whose product is finite through the kernel's conversion against
//!     weights.woABf16 (one-hot activation rows: each output is one converted weight); one o value changed differs;
//!     a soak of 200 launches on one input, each output the first's.
//!   - TF_DS_QUANT_ONCE: prompt.frontLinears over a chunk's rows (x quantized once, wq_a and wkv on the codes)
//!     against mx.linear of each (each quantizing x) at 1 .. 2,048 rows, the scratch both read poisoned before each:
//!     qa and y byte-equal; one x value set to 64 differs.
//!   - TF_DS_PF_GEMM5: mx.gemm of Engram's wkv (3,072 x 12,800, no K split) on tf_dsn_pf_gemm5 against tf_dsn_pf_gemm,
//!   - TF_DS_GROUP_TILES: mx.grouping of two skewed groupings in turn on one scratch poisoned with 0x7f bytes (experts
//!     of 0, 1, 63, 64, 65 and 300 picks, the shared expert in every row, ids of another node), the second's outputs
//!     against the baseline (tf_dsn_group_scan, then mx.tileList): off, act, nact, fill, the tile list and its length equal,
//!     each expert's entries and rows equal as sets (place's order within an expert is its atomics'); a changed pick
//!     differs; a soak of 200 launches on one input, each output the first's.
//!   - TF_DS_DEC_GROUP2: mx.decodeGrouping (a decode launch's grouping) with the scan-and-rank kernel and with the
//!     one-CTA kernel, each against the grouping computed on the host (off, nact, act, and each expert's entries and
//!     rows in entry order, exactly) for 2 .. 585 rows of 7 picks (14 .. 4,095 entries; three experts holding about a
//!     third of them, ids of another node among them), the scratch poisoned with 0x7f bytes before each launch; one
//!     pick moved differs.
//!   - TF_DS_MERGE_D: tri_attn.sparseAttn of a decode round's shapes (1, 2, 6 and 16 rows of 16 heads, 8 splits, the pool's
//!     ring with per-row bases) with the single-program merge against the merge's column twin at 128 and 64 columns a
//!     program: OUT and the split partials byte-equal; a ring value changed differs; every call found its twin.
//!   - TF_DS_TOPK_TILES: mx.topkTiles against the exact top-k on the host, and tri_index.topkSelectPruned (the kit's
//!     _topk_sel / _gather_tiles / _topk_finish and ops.topkI64) against it as a second reference, on keys made as the
//!     scorer makes them (an fp32 score's ordered bits above, the inverted position below; tile maxima of 64 keys), k
//!     512, 32,769 .. 524,288 keys a row, each row its own vis (every key, past n, fewer than k, inside the last tile,
//!     none, half), each row that sees every key holding its largest (but a head's) at n - 1 (a partial last tile
//!     among the picks at 32,769 and 262,143 keys). The rows a tile past their last tile apart, with padding keys
//!     that outscore every real one; then production's packed rows (a stride of n) at 32,769, 65,536 and 262,143
//!     keys, each row's keys 0 .. 62 outscoring its others (what a read past the row before's n reaches): the picks
//!     equal. Controls: row 0's largest key lowered (its tile's maximum recomputed) gives the changed keys' exact
//!     top-k; at a partial tile, the launch told the tiles' whole length (nt x 64 keys) reads past n and picks
//!     position n where it is visible. A soak of 200 launches on one input at 262,143 packed keys, each output the
//!     first's. A chain that cannot run fails; the kernel's lines run without it.
//!   - TF_DS_PASS_TAIL: the pass rule at the served prompt lengths, served, D1 and with a merged final pass (its counts
//!     and every plan's whole non-final passes on the resume grid), and the indexer at a merged final pass's compressed
//!     key count against the count of the chunk it replaces: the same keys and top-k picks; one changed code byte differs.
//!   - TF_DS_IDX_PROJ_OWNED: the indexer's projections of 2,048 rows (wq_b through mx.linear, then prompt.idxQueries:
//!     the RoPE and fp4_qd, then weights_proj through mx.bf16Gemm) against the same launches at every owned block of
//!     512, 256 and 1,024 rows at the rows' offsets (wq_b also on each TF_DS_PF_SPLIT configuration): iq, iq4 and iw
//!     byte-equal; one qr value set to 64 differs.
//!   - TF_DS_BF16_ROWS: mx.bf16Gemm of weights_proj (x [M, 5,120] bf16 through [32, 5,120], rounded first, times the
//!     one x value set to 64 differs.
//!   - TF_DS_PF_RUN128: mx.gateUp and mx.down of a 2D node's experts (384 MXFP4 routed experts over 8 weight sets, the
//!     FP8 shared expert, a pack of 640 columns with a 128-column rest and a random owner of each rest, down 2,560 x
//!     fp32 entry outputs and the combined fp32 rows byte-equal; one x value and one down input value set to 64 differ;
//!     every staged piece is read: one byte of a weight set changed in each K block of a pair (down's tail pair too)
//!     and in each stage word's scales changes the twin's own output; a soak of the twins, 200 launches on one input at
//!     512 rows, each output the first's.
//!   - TF_DS_HC_STRIDE: a quarter-HC mix on the chunk's own gathered planes (plane 0 at the owned rows' offset,
//!     cap * D elements between planes) against the tight repack it reads in the baseline (rows * D between them): PART and
//!     h_out byte-equal at two caps and owned ranges past row 0; one plane value changed differs.
//!   - TF_DS_HC_FUSED: tri_hc.hcPrePf and tri_hc.hcPre with the switch off (the baseline _hc_mix_pf / _hc_partial and
//!     _hc_finish) and on (the extra set's _hc_pre_pf_fused), at 1,024 and 4,096 rows, hidden 5,120, two planes, the
//!     post fused in and without it: out, pre, post, comb and the posted streams byte-equal; one stream value changed
//!     differs; every launch found the set's variant. On the owner sums' two bf16 blocks (blk_rs set; the arena layout) the
//!     extra set's _hc_pre_pf_fused_2b against _hc_mix_pf_2b then _hc_finish, at 1,024 and 4,096 rows: the same outputs.
//!   - TF_DS_ATTN_EPI_FUSED: a prompt chunk's attention with the epilogue in its final store (the extra set's
//!     _sparse_attn_part_ue) against the part's twin then the epilogue launch, 256 rows at 5 .. 512 picks: the codes and
//!     scales, and wo_a's fake rows, byte-equal; one picked row's code byte changed differs; every launch found the variant.
//!   - TF_DS_GEMM_R2: TF_DS_MMA_SF_EXACT's arithmetic on its R2 expert kernels (the activation rows staged as there, a
//!     padding row's copies left out; down on 128-row tiles): the experts' pack, entry outputs and combined rows at
//!     37 .. 4,096 rows against its kernels, their route asserted, and one dense shape's outputs with the switch on
//!     against off (the dense split GEMMs stay on the exact kernels); one changed x value differs.
//!   - TF_DS_ATTN_EPILOGUE: a prompt chunk's attention output takes the inverse RoPE and wo_a's quant in one launch,
//!     against rope_heads followed by the quant, at 1, 17 and 4,096 rows of 16 heads of 512: the roped o, the codes
//!     and scales, and wo_a's fake rows byte-equal; one o value in the roped columns changed differs.
//!   - TF_DS_ASSEMBLE_QUANT: a parity chunk's assembly of the expert intermediate and down's quant of it in one
//!     launch against assemble_rest followed by the quant, at 4,096 rows of 7 slots and the node's 1,152 columns: the
//!     assembled rows, down's codes and scales byte-equal; one pack value changed differs.
//!   - TF_DS_ATTN_U: tri_attn.sparseAttn of packed latents with and without `u`, the base set's _sparse_attn_part
//!     against the extra set's twin _sparse_attn_part_u (AOT_DIR/../aot-extra), prompt chunks at the window's end of
//!     5 .. 512 picks (NBLK 5, 6, 8, 12, 20) and a small chunk on the ring (8 splits): the outputs byte-equal; one
//!     picked row's code byte changed differs.
//!   - TF_DS_IDX_U: tri_index.indexScore of a prompt chunk's row block with and without the switch, the base set's
//!     _index_score_rows against the extra set's twin _index_score_rows_u (AOT_DIR/../aot-extra), at RB 2 and RB 1
//!     (TREE), keys with tile maxima and plain scores, on packed keys of every code byte and every finite E8M0 scale
//!     byte: byte-equal; one visible key's code byte changed differs; every call found its twin.
//!   - TF_DS_IDX_V: the same calls against the extra set's twin _index_score_rows_v (one row's dots at a time, its head
//!     sum written out, four warps): byte-equal; one visible key's code byte changed differs; every call found its twin.
//!   - TF_DS_ATTN_U_DECODE: tri_attn.sparseAttn of a decode round's shapes with and without `u` (the pool's ring, 1 ..
//!     16 rows split 8 ways, each row's ring and compressed bases or none, 512 and 20 picks), the part against its
//!     twin: OUT and the partials byte-equal; one picked row's code byte changed differs; every call found its twin.
//!   - TF_DS_DRAFT_MAIN_4WAY: the drafter's main_proj (FP8, 5,120 x 15,360) on each quarter of its output columns (the
//!     whole weight's rows and their scale blocks, as a node loads them) through mx.linearsDecode, the absorb's call,
//!     against the whole GEMV at 1 .. 128 rows: each quarter the whole output's columns byte for byte; a changed taps
//!     value differs.
//!   - TF_DS_MMA_SF: mx.gemm of the dense FP8 GEMMs' shapes (the split ones and Engram's wkv) and mx.gateUp / mx.down of
//!     a 2D node's experts (pf_run128's) on the block-scaled kernels (native_sf.cu; their own seed): the exact
//!     arithmetic (each K block's MMA from zero, then an fp32 add), each GEMM's key proven first, against the baseline
//!     the shape has a K split; Engram's wkv, none, the baseline at every row count); the switch's chained arithmetic (a
//!     split's K blocks accumulated in the MMA, not the baseline bits) gives a row the same bits at every row count and run
//!     to run, and every fp32 output stays within the baseline distance to fp64
//!     plus (a split's K blocks + the splits) x 2^-23 of the sum of its products' magnitudes (16 rows of each dense
//!     GEMM, 35 expert entries; gate / up's bf16 pack normwise within 2^-6 of the baseline), each line with the largest
//!     differences and the share that differ against the baseline; one changed code or x value differs.
//!   - TF_DS_MMA_SF_EXACT: the same GEMMs on the block-scaled kernels' exact arithmetic (its own seed): the dense shapes'
//!     asserted; its guard's instrument (tf_dsn_sf_ranges over packed tensors and every expert's three scale tensors)
//!     matches the host's; at the proven range's edges (FP8 41 and 127, MXFP4 33 and 137) the kernels give the baseline
//!     bytes, and a byte outside it keeps the GEMM on the baseline kernel; the guard's coverage over two layers (every dense
//!     field each in turn, the experts with an owned rest): the planted tensor's GEMM alone kept, by its key and in
//!     words, every other proven, the bytes it saw; a layer whose loader did not tie its scale tensors to the guard's
//!     lengths: none of its bytes read, its experts kept apart; one changed code differs.
//!   - TF_DS_GEMM_W16: the exact GEMMs' 16-warp (512-thread) kernels against the baseline 8-warp ones on the same shapes
//!     (their own seed): the dense FP8 GEMMs of production's shapes at 64 .. 4,096 rows (a partial 64-row tile at 67,
//!     1,000 and 2,048 and a partial 128-row tile at 1,025) in bf16 and fp32, a 2D node's experts at 37 .. 4,096 rows
//!     (the FP8 shared expert among them) and wo_a's bf16 rows at 1 .. 4,096 rows: every output byte-equal, each
//!     launch's kernel asserted (the twin loaded, the shape's split the twin's); one changed input differs in each.
//!   - TF_DS_SPLIT64N: the exact split GEMMs of wkv (N 512) and wq_a (N 1,280) on the 64 x 64 tile against the 64 x 128 one at 1,
//!     37, 1,000, 1,024 and 4,096 rows, bf16 and fp32 outputs, every byte equal and each launch's route asserted; one code byte
//!     changed differs.
//!   - TF_DS_PF_RASTER: mx.gemm on the dense FP8 shapes a prompt pass runs (wq_a, wkv, wo_b) on the prefill tile kernels, with
//!     TF_DS_PF_SPLIT=table, and on the block-scaled kernels (wq_a, wo_b), and mx.bf16Gemm of the indexer's weights_proj, at 4,096,
//!     8,192, 16,384 and 8,200 rows (a partial last row tile) with each group of row tiles (2, 4, 8, 16) against the default
//!     order, and with TF_DS_GEMM_HALVES on beside it: every output byte equal; the gate (a 1,024-row launch keeps the default
//!     order, a launch of 14 MiB of activation takes the group) asserted; one code byte changed differs; the GB10 timing lines
//!     (default, each group, halves) at 8,192 and 16,384 rows on wq_a, wo_b and the block-scaled wq_a.
//!   - TF_DS_RING_WC: a ring's data slots from cudaHostAlloc(Mapped | WriteCombined) against the baseline mmap +
//!     cuMemHostRegister, on the real staging, publish and collecting kernels: the staging kernel's slot and the
//!     publish kernel's fixed slot read back from the host byte for byte; the doorbell, size, slot name and last
//!     sequence from the baseline memory; a copy-out of a write-combined landing ring against the baseline; one source byte
//!     changed differs.
//!   - TF_DS_RING_INPLACE: the in-place publish's copy-out (the producer's bytes in a write-combined send slot, the
//!     waiting and publishing kernels, the landing in a write-combined receive ring) against the baseline stage and copy-out
//!     on the same bytes: every destination byte equal; the await holds its stream while the previous send's sequence
//!     is outstanding and ends when it completes; the publish's sequence, slot name and size; one source or landing
//!     byte changed differs.
//!   - TF_DS_RING_WORD: the generation-word collecting kernel (dsv41_rdma_collect_into_roci) on a receive ring's
//!     slots against the baseline dsv41_rdma_collect_into on the same slots, a host thread writing the lanes' words after
//!     the launch: the word signal (the sequence inline) and the atomic signal (one fetch-and-add a lane) give the baseline
//!     bytes (also out of write-combined landing slots); one landing byte changed differs; words written 20 ms late
//!     still collect under the deadline; a lane's
//!     word that never arrives ends in the failure record within the deadline (the ring's abort word set, nothing
//!     copied); the abort word alone ends a wait with the record's abort reason, far under the deadline.
//!   - TF_DS_ROCI's two-block readers (_hc_mix_pf_2b, _hc_post_2b, _engram_gate_2b): the blocks as the transport lands them against
//!     widenPairs + the baseline readers (the fp32 planes), byte for byte, at 1 to 1,024 rows with NaN, +-0, subnormals and infinities
//!     in the sums; a sum changed must change each output.
//!   - TF_DS_ROCI's whole-line GEMM stores (the _st kernels): a dense FP8 GEMM into write-combined host memory against the same into
//!     device memory, byte for byte, with the time of both; a changed activation code must change the bytes.
//!   - TF_DS_ROCI_COLLECT's copy (Ops.collect2): two blocks from device memory into write-combined host memory, the bytes equal, the time
//!     of the launch printed; a changed source must differ.
//!   - the cost of splitting wo_b's GEMM and the MoE combine into four row quarters (TF_DS_ROCI sends two of four): kernel-only times, device y
//!     and arena y, 4,096 rows as one launch and as four of 1,024.
//!   - TF_DS_ROCI_COMBINE_FIRST's routed sum (the quarters an owner sum's first part sends, a host step, the quarters the node keeps)
//!     against one tf_dsn_combine launch of 4,096 rows, into device memory and into write-combined host memory, byte for byte; the
//!     host step sees the sent quarters complete and the kept ones unwritten; a kept quarter left out must differ.
//!   - the kernel classes the owner path moves onto the arena (roci_wc_classes: act_quant in planar and in records, hcPrePf, ffnRoute,
//!     rmsnorm, pf_gu and down from codes, the block-scaled GEMM from codes, sparseAttn from cidx), kernel only at the 4,096-row shapes,
//!     into or out of cudaMalloc, a second cudaMalloc set (the null control, within 2 %) and write-combined host memory: ms a launch and a pass.
//!   - TF_DS_ROCI_GATHERS' boundary staging (roci_boundary): four nodes' cache arrays staged into the arena's boundary region, the slabs
//!     written to the other three regions, copied out to their rows, against the in-place gather's layout, with the region in cudaMalloc, in
//!     write-combined host memory and in locked registered pages; a node that stages nothing must leave its rows missing; the ms of the
//!     16 copies.
//!   - TF_DS_ROCI_CODES' two exchanges on two column partners (roci_codes): wo_b's part quantized into the arena's send region, written to
//!     the partner's land region, read by the four copies that assemble wo_b's input; the experts' pack quantized, written, read by the
//!     assembly into down's input; the regions in cudaMalloc, write-combined host memory and locked registered pages: every consumer byte
//!     equal to the run in device memory; a pack that never lands must differ; the ms of the quantizers, the copies and the assembly.
//!   - TF_DS_ROCI_STAGGER's two lanes (roci_pair): two passes' exchanges in flight at once on two streams a node, each lane's bytes in its own
//!     lane; a late peer on one lane holds that lane's wait and not the other's streams; the cache gate (pass B's read behind pass A's event is
//!     A's write, without the event it is not).
//!   - TF_DS_ROCI_STAGGER_LANES=3 (roci_triple): three passes' exchanges in flight at once on three lanes, each lane's bytes in its own lane; the cache
//!     gate chain (pass C's read behind B's event behind A's event is A's write, with B's wait on A's left out it is not).
//!   - The decoder half on owned rows (roci_dec_rows, roci_dec_mixes, roci_dec_kset): at every decoder window of d = 1 to 128 rows, each
//!     launch class layers 20-39 make at a node's slab of the plan (slab_plan: q(d) = 16 or 32 rows a rank) against the window's one
//!     launch, the baseline decoder half's: act_quant, wq_a / wkv / wq_b, the q RMSNorm, the router's logits and routing, hcPre; the posted
//!     mixes and posts from the transport's two blocks at the plan's stride against _hc_mix_pf / _hc_post on the quarters' planes; a
//!     tap's mean; the experts' gate / up from the slabs' codes against gateUp's own quantization. Every byte equal; a slab one row short
//!     and blocks read one row short of their stride must differ. And the decoder half's start check on the kit (every layer's launches at
//!     every slab of a window, the two-block readers in the extra set), with a control the check must refuse.
//!   - TF_DS_QGATE_CANARY 2 / 3 (the body canary, mx.qgateBody) on layer 0's MoE partials' shapes (1, 3, 17, 48 and
//!     2,048 rows of 2,560 and 5,120 fp32 columns; every group at its own power-of-two magnitude, zero groups,
//!     negative zeros, a value far above its group): mode 2 in place against act_quant's quant-dequant of the same
//!     values (mx.fake, bf16, widened) byte-equal, and the values it moved; mode 3 against each value's bits plus one
//!     (zeros, infinities, NaN and the largest finite values kept) byte-equal.
//!   tf-dsv41-kernels-replay AOT_DIR
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const mx = dsv41.mx;
const weights = dsv41.weights;
const tri_attn = dsv41.tri_attn;
const tri_hc = dsv41.tri_hc;
const tri_norm = dsv41.tri_norm;
const tri_basic = dsv41.tri_basic;
const tri = dsv41.tri;

/// CUDA's __float2bfloat16_rn of a non-NaN value: round to nearest, ties to even.
fn bf16Rne(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn upload(d: *const cuda.Driver, bytes: []const u8) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(d, bytes.len);
    return b;
}

/// A device buffer's first n bytes, read back (the caller frees them with std.heap.page_allocator).
fn get(buf: u64, n: usize, d: *const cuda.Driver) ![]u8 {
    const x = try std.heap.page_allocator.alloc(u8, n);
    errdefer std.heap.page_allocator.free(x);
    return x;
}

/// A soak's launches: one input, each output against the first's (a race or a read of stale memory shows as a launch

fn poison(buf: u64, n: usize, d: *const cuda.Driver, s: cuda.Stream) !void {
    try d.check(d.api.cuMemsetD8Async(buf, 0xa5, n, s.handle), "cuMemsetD8Async");
}

/// bf16 values, normal * scale (bf16-rounded).
fn bf16s(n: usize, rng: std.Random, scale: f32, a: std.mem.Allocator) ![]u8 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = bf16Rne(rng.floatNorm(f32) * scale);
    return std.mem.sliceAsBytes(v);
}

/// fp32 values, normal.
fn f32s(n: usize, rng: std.Random, a: std.mem.Allocator) ![]u8 {
    const v = try a.alloc(f32, n);
    for (v) |*x| x.* = rng.floatNorm(f32);
    return std.mem.sliceAsBytes(v);
}

/// Weight scales: `fast` every UE8M0 byte in 118 .. 127 (2^-9 .. 2^0: the tile kernels' fast scaling); else a quarter
/// of the blocks outside [23, 127] (2^1 .. 2^13 and 2^-112 .. 2^-105), so every tile runs the slow scaling.
const Scales = enum { fast, slow };

/// An FP8 linear [n, k]: E4M3 codes (no NaN) and UE8M0 scales a 32 x 32 block.
fn fp8Linear(d: *const cuda.Driver, n: usize, k: usize, sc_kind: Scales, rng: std.Random, a: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer)) !weights.Linear {
    const w = try a.alloc(u8, n * k);
    for (w) |*x| {
        x.* = rng.int(u8);
        if (x.* & 0x7f == 0x7f) x.* -= 1;
    }
    const sc = try a.alloc(u8, (n / 32) * (k / 32));
    for (sc) |*x| {
        x.* = 118 + rng.uintLessThan(u8, 10);
        if (sc_kind == .slow and rng.uintLessThan(u8, 4) == 0) x.* = if (rng.boolean()) 128 + rng.uintLessThan(u8, 13) else 15 + rng.uintLessThan(u8, 8);
    }
    const wb = try upload(d, w);
    const sb = try upload(d, sc);
    try keep.append(a, wb);
    try keep.append(a, sb);
    return .{ .words = 0, .k = @intCast(k), .n = @intCast(n), .fmt = .fp8, .w = wb.ptr, .s = sb.ptr };
}

fn differs(x: []const u8, y: []const u8) usize {
    var n: usize = 0;
    for (x, y) |p, q| n += @intFromBool(p != q);
    return n;
}

const Run = struct {
    d: *const cuda.Driver,
    dev: cuda.abi.Device = 0,
    s: cuda.Stream,
    t: dsv41.tri.Tri,
    k: *mx.Kernels,
    a: std.mem.Allocator,
    ops: ?*const dsv41.ops.Ops = null,
    ok: bool = true,

    fn report(r: *Run, what: []const u8, n: usize, want_equal: bool) void {
        const pass = (n == 0) == want_equal;
        if (!pass) r.ok = false;
        std.debug.print("{s}: {d} bytes differ: {s}\n", .{ what, n, if (pass) "as it must" else "WRONG" });
    }

    fn note(r: *Run, what: []const u8, holds: bool) void {
        if (!holds) r.ok = false;
        std.debug.print("{s}: {s}\n", .{ what, if (holds) "as it must" else "WRONG" });
    }
};

// -- TF_DS_PF_SPLIT --------------------------------------------------------------------------------------------------

const Shape = struct { name: []const u8, k: usize, n: usize };
/// The split FP8 GEMMs of a prompt chunk on a 2D node (K, N).
const split_shapes = [_]Shape{
    .{ .name = "wq_a", .k = 5120, .n = 1280 },
    .{ .name = "wkv", .k = 5120, .n = 512 },
    .{ .name = "wq_b", .k = 1280, .n = 8192 },
    .{ .name = "wo_b", .k = 4096, .n = 2560 },
    .{ .name = "idx_wq_b", .k = 1280, .n = 4096 },
};
const configs = [_]mx.PfSplit{ .a, .b, .c6, .c7 };
const config_names = [_][]const u8{ "A", "B", "C6", "C7" };

/// The output modes a launch is checked in: bf16, fp32, fp32 accumulated onto values already there.
const Mode = enum { bf16, f32, acc };

/// One launch of mx.gemm with TF_DS_PF_SPLIT `sw` in effect (pf_off: the slab kernels) on rows [r0, r0 + m) of the
/// quantized activation, into y (poisoned, or holding `init` for an accumulated launch); its bytes (get's).
fn gemmOnce(r: *Run, sw: mx.PfSplit, pf_off: bool, l: weights.Linear, q: u64, s: u64, z: u64, r0: usize, m: usize, y: u64, mode: Mode, init: ?[]const u8) ![]u8 {
    const esz: usize = if (mode == .bf16) 2 else 4;
    const bytes = m * l.n * esz;
    r.k.split = sw;
    r.k.pf_off = pf_off;
    defer {
        r.k.split = .off;
        r.k.pf_off = false;
    }
    try mx.gemm(r.k, r.s, l, q + r0 * l.k, l.k, s + r0 * (l.k / 32) * 4, l.k / 32, m, y, l.n, if (mode == .bf16) .bf16 else .f32, mode == .acc, z);
    try r.s.synchronize();
    return get(y, bytes, r.d);
}

fn free(x: []u8) void {
    std.heap.page_allocator.free(x);
}

fn pfSplit(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var line: [256]u8 = undefined;
    for (split_shapes) |sh| for ([_]Scales{ .fast, .slow }) |scales| {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const l = try fp8Linear(d, sh.n, sh.k, scales, rng, ha, &keep);
        // the activation rows' codes and scales (act_quant), as every split GEMM reads them
        const x = try upload(d, try bf16s(split_max_rows * sh.k, rng, 0.5, ha));
        try keep.append(ha, x);
        const q = try cuda.DeviceBuffer.alloc(d, split_max_rows * sh.k);
        try keep.append(ha, q);
        const s = try cuda.DeviceBuffer.alloc(d, split_max_rows * (sh.k / 32) * 4);
        try keep.append(ha, s);
        try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, split_max_rows, sh.k, q.ptr, sh.k, s.ptr, sh.k / 32);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, split_max_rows), 1) * 4);
        try keep.append(ha, z);
        const y = try cuda.DeviceBuffer.alloc(d, split_max_rows * sh.n * 4);
        try keep.append(ha, y);
        var n_diff: [4]usize = @splat(0);
        var launches: [4]usize = @splat(0);
        for ([_]Mode{ .bf16, .f32, .acc }) |mode| {
            const rows: []const usize = if (mode == .bf16) &split_rows else &split_rows_f32;
            for (rows) |m| {
                var iarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                defer iarena.deinit();
                const init: ?[]const u8 = if (mode == .acc) try f32s(m * sh.n, rng, iarena.allocator()) else null;
                const want = try gemmOnce(r, .off, false, l, q.ptr, s.ptr, z.ptr, 0, m, y.ptr, mode, init);
                defer free(want);
                for (configs, 0..) |sw, ci| {
                    if (mx.pfSplitCfg(sw, m, sh.k, sh.n) == null) continue;
                    const got = try gemmOnce(r, sw, false, l, q.ptr, s.ptr, z.ptr, 0, m, y.ptr, mode, init);
                    defer free(got);
                    n_diff[ci] += differs(want, got);
                    launches[ci] += 1;
                }
            }
        }
        for (configs, config_names, 0..) |sw, name, ci| {
            if (mx.pfSplitCfg(sw, 2048, sh.k, sh.n) == null) {
                // a configuration that does not divide the shape's split: mx.gemm takes the fold kernel
                const fold = mx.pfGemmOf(sw, 2048, sh.k, sh.n) == .fold;
                if (scales == .fast) r.note(try std.fmt.bufPrint(&line, "pf_split {s} {s} ({d} x {d}): not this shape's configuration (its split's K blocks), the fold kernel", .{ name, sh.name, sh.k, sh.n }), fold);
                continue;
            }
        }
        if (scales == .fast) {
            // the fold kernel against the slab kernels and their reduce (TF_DS_PF_TILES=0), bf16 at three row counts
            var slab: usize = 0;
            for ([_]usize{ 64, 129, 1000 }) |m| {
                const want = try gemmOnce(r, .off, true, l, q.ptr, s.ptr, z.ptr, 0, m, y.ptr, .bf16, null);
                defer free(want);
                const got = try gemmOnce(r, .off, false, l, q.ptr, s.ptr, z.ptr, 0, m, y.ptr, .bf16, null);
                defer free(got);
                slab += differs(want, got);
            }
            r.report(try std.fmt.bufPrint(&line, "pf_split {s}: the fold kernel against the slab kernels and their reduce at 64, 129, 1000 rows", .{sh.name}), slab, true);
            // a row's bits at another place in the launch: rows 37 .. 1036 of a 1,037-row launch against those rows alone
            for (configs, config_names) |sw, name| {
                if (mx.pfSplitCfg(sw, 1000, sh.k, sh.n) == null) continue;
                const whole = try gemmOnce(r, sw, false, l, q.ptr, s.ptr, z.ptr, 0, 1037, y.ptr, .bf16, null);
                defer free(whole);
                const part = try gemmOnce(r, sw, false, l, q.ptr, s.ptr, z.ptr, 37, 1000, y.ptr, .bf16, null);
                defer free(part);
                r.report(try std.fmt.bufPrint(&line, "pf_split {s} {s}: rows 37 .. 1036 of one launch against a launch of those rows", .{ name, sh.name }), differs(whole[37 * sh.n * 2 ..], part), true);
            }
            // the control: one code of row 700 changed, against the unchanged fold kernel's bytes
            const want = try gemmOnce(r, .off, false, l, q.ptr, s.ptr, z.ptr, 0, 1024, y.ptr, .bf16, null);
            defer free(want);
            var code: [1]u8 = undefined;
            const at = q.ptr + 700 * sh.k + 5;
            const changed = [1]u8{code[0] ^ 0x08};
            const cfg_on: mx.PfSplit = if (mx.pfSplitCfg(.b, 1024, sh.k, sh.n) != null) .b else .c7;
            const got = try gemmOnce(r, cfg_on, false, l, q.ptr, s.ptr, z.ptr, 0, 1024, y.ptr, .bf16, null);
            defer free(got);
            r.report(try std.fmt.bufPrint(&line, "pf_split {s} control: one code of row 700 changed ({s})", .{ sh.name, config_names[@backingInt(cfg_on) - 2] }), differs(want, got), false);
        }
    };
}

// -- TF_DS_PF_GEMM5 --------------------------------------------------------------------------------------------------

fn pfGemm5(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var line: [256]u8 = undefined;
    const K: usize = 3072;
    const N: usize = 12800;
    defer r.k.gemm5 = false;
    for ([_]Scales{ .fast, .slow }) |scales| {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const l = try fp8Linear(d, N, K, scales, rng, ha, &keep);
        try keep.append(ha, x);
        try keep.append(ha, q);
        try keep.append(ha, s);
        try keep.append(ha, y);
        var diff: usize = 0;
        var launches: usize = 0;
        for ([_]Mode{ .f32, .bf16 }) |mode| for (rows) |m| {
            r.k.gemm5 = false;
            const want = try gemmOnce(r, .off, false, l, q.ptr, s.ptr, 0, 0, m, y.ptr, mode, null);
            defer free(want);
            r.k.gemm5 = true;
            const got = try gemmOnce(r, .off, false, l, q.ptr, s.ptr, 0, 0, m, y.ptr, mode, null);
            defer free(got);
            diff += differs(want, got);
            launches += 1;
        };
        if (scales == .fast) {
            r.k.gemm5 = false;
            const want = try gemmOnce(r, .off, false, l, q.ptr, s.ptr, 0, 0, 1000, y.ptr, .f32, null);
            defer free(want);
            var code: [1]u8 = undefined;
            const at = q.ptr + 700 * K + 9;
            const changed = [1]u8{code[0] ^ 0x08};
            r.k.gemm5 = true;
            const got = try gemmOnce(r, .off, false, l, q.ptr, s.ptr, 0, 0, 1000, y.ptr, .f32, null);
            defer free(got);
            r.report("pf_gemm5 control: one code of row 700 changed", differs(want, got), false);
        }
    }
}

// -- TF_DS_GROUP_TILES -----------------------------------------------------------------------------------------------

const G_ROWS: usize = 2048;
const G_SLOTS: usize = 7; // 6 routed picks and the shared expert
const G_E: usize = 385; // 384 routed experts and the shared one (id 384)

/// A grouping's picks [G_ROWS, G_SLOTS]: experts 10 .. 15 hold 1, 63, 64, 65, 300 and 0 picks, the others random
/// (a row's routed picks distinct), ids of another node (>= G_E) among them, the shared expert every row's last slot.
fn picksOf(rng: std.Random, a: std.mem.Allocator) ![]i32 {
    const p = try a.alloc(i32, G_ROWS * G_SLOTS);
    const quota = [_]usize{ 1, 63, 64, 65, 300, 0 };
    var left = quota;
    for (0..G_ROWS) |row| {
        var used: [6]i32 = undefined;
        for (0..6) |sl| {
            var e: i32 = undefined;
            while (true) {
                // the skewed experts first, while they have picks to place, then any other id
                const forced = for (left, 0..) |q, i| {
                    if (q > 0 and rng.uintLessThan(usize, G_ROWS - row) < q) break @as(?usize, i);
                } else null;
                if (forced) |i| {
                    e = @intCast(10 + i);
                } else {
                    e = @intCast(rng.uintLessThan(usize, G_E + 40)); // past 384: another node's
                    if (e >= 10 and e <= 15) continue;
                    if (e == G_E - 1) continue; // the shared expert has its own slot
                }
                if (std.mem.indexOfScalar(i32, used[0..sl], e) == null) break;
            }
            if (e >= 10 and e <= 15) left[@intCast(e - 10)] -= 1;
            used[sl] = e;
            p[row * G_SLOTS + sl] = e;
        }
        p[row * G_SLOTS + 6] = G_E - 1;
    }
    return p;
}

/// A grouping's outputs read back: off, act (nact of them), nact, fill, tiles (ntiles), ntiles, ent and erow.
const Grouping = struct { off: []i32, act: []i32, nact: i32, fill: []i32, tiles: []i32, ntiles: i32, ent: []i32, erow: []i32 };

fn readGrouping(r: *Run, sc: mx.ExpertScratch, a: std.mem.Allocator) !Grouping {
    const d = r.d;
    const ints = struct {
        fn of(dd: *const cuda.Driver, ptr: u64, n: usize, aa: std.mem.Allocator) ![]i32 {
            const x = try aa.alloc(i32, n);
            return x;
        }
    }.of;
    const nact = (try ints(d, sc.nact, 1, a))[0];
    const ntiles = (try ints(d, sc.ntiles, 1, a))[0];
    const off = try ints(d, sc.off, G_E + 1, a);
    const p: usize = @intCast(@max(off[G_E], 0));
    return .{
        .off = off,
        .act = try ints(d, sc.act, @intCast(std.math.clamp(nact, 0, G_E)), a),
        .nact = nact,
        .fill = try ints(d, sc.fill, G_E, a),
        .tiles = try ints(d, sc.tiles, 2 * @as(usize, @intCast(std.math.clamp(ntiles, 0, G_ROWS * G_SLOTS / 64 + G_E))), a),
        .ntiles = ntiles,
        .ent = try ints(d, sc.ent, @min(p, G_ROWS * G_SLOTS), a),
        .erow = try ints(d, sc.erow, @min(p, G_ROWS * G_SLOTS), a),
    };
}

/// Ints that differ between two groupings: the lists exactly, each expert's entries and rows as sorted sets.
fn groupingDiff(x: Grouping, y: Grouping) usize {
    var n: usize = 0;
    n += @intFromBool(x.nact != y.nact) + @intFromBool(x.ntiles != y.ntiles);
    for ([_][2][]const i32{ .{ x.off, y.off }, .{ x.act, y.act }, .{ x.fill, y.fill }, .{ x.tiles, y.tiles } }) |pair| {
        if (pair[0].len != pair[1].len) {
            n += 1;
            continue;
        }
        for (pair[0], pair[1]) |u, v| n += @intFromBool(u != v);
    }
    if (x.ent.len != y.ent.len or n > 0) return n + @intFromBool(x.ent.len != y.ent.len);
    for (0..G_E) |e| {
        const lo: usize = @intCast(x.off[e]);
        const hi: usize = @intCast(x.off[e + 1]);
        for ([_][2][]i32{ .{ x.ent, y.ent }, .{ x.erow, y.erow } }) |pair| {
            std.mem.sort(i32, pair[0][lo..hi], {}, std.sort.asc(i32));
            std.mem.sort(i32, pair[1][lo..hi], {}, std.sort.asc(i32));
            for (pair[0][lo..hi], pair[1][lo..hi]) |u, v| n += @intFromBool(u != v);
        }
    }
    return n;
}

fn groupTiles(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const p = G_ROWS * G_SLOTS;
    const sizes = [_]usize{ 4 * G_E, 4 * (G_E + 1), 4 * G_E, 4 * G_E, 4, 4 * p, 4 * p, mx.ExpertScratch.tileBytes(G_ROWS, G_SLOTS, G_E), 4 };
    var bufs: [sizes.len]cuda.DeviceBuffer = undefined;
    for (&bufs, sizes) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b.*);
    }
    var sc = std.mem.zeroes(mx.ExpertScratch);
    sc.rows = G_ROWS;
    sc.slots = G_SLOTS;
    sc.cnt = bufs[0].ptr;
    sc.off = bufs[1].ptr;
    sc.fill = bufs[2].ptr;
    sc.act = bufs[3].ptr;
    sc.nact = bufs[4].ptr;
    sc.ent = bufs[5].ptr;
    sc.erow = bufs[6].ptr;
    sc.tiles = bufs[7].ptr;
    sc.ntiles = bufs[8].ptr;
    const pa = try upload(d, std.mem.sliceAsBytes(try picksOf(rng, ha)));
    try keep.append(ha, pa);
    const pb_host = try picksOf(rng, ha);
    const pb = try upload(d, std.mem.sliceAsBytes(pb_host));
    try keep.append(ha, pb);
    // grouping A then B on scratch poisoned with 0x7f bytes, B's outputs: the baseline (scan, then tileList) or the one launch
    const Two = struct {
        fn run(rr: *Run, s: mx.ExpertScratch, picks: [2]u64, once: bool, bs: []const cuda.DeviceBuffer, aa: std.mem.Allocator) !Grouping {
            for (bs) |b| try rr.d.check(rr.d.api.cuMemsetD8Async(b.ptr, 0x7f, b.len, rr.s.handle), "cuMemsetD8Async");
            rr.k.group_tiles = once;
            defer rr.k.group_tiles = false;
            for (picks) |pk| {
                try mx.grouping(rr.k, rr.s, rr.d, G_E, s, pk, G_ROWS);
                if (!once) try mx.tileList(rr.k, rr.s, s);
            }
            try rr.s.synchronize();
            return readGrouping(rr, s, aa);
        }
    };
    const want = try Two.run(r, sc, .{ pa.ptr, pb.ptr }, false, &bufs, ha);
    const got = try Two.run(r, sc, .{ pa.ptr, pb.ptr }, true, &bufs, ha);
    var line: [256]u8 = undefined;
    r.report(try std.fmt.bufPrint(&line, "group_tiles: the second of two groupings ({d} active experts, {d} tiles) against the scan and the tile list's (ints)", .{ want.nact, want.ntiles }), groupingDiff(want, got), true);
    // the skewed experts' tiles: 1, 63, 64, 65, 300 and 0 picks take 1, 1, 1, 2, 5 and 0 tiles in the one launch's list
    var counts: [6]usize = @splat(0);
    for (0..@intCast(got.ntiles)) |t| {
        const e = got.tiles[2 * t];
        if (e >= 10 and e <= 15) counts[@intCast(e - 10)] += 1;
    }
    r.note(try std.fmt.bufPrint(&line, "group_tiles: the skewed experts' tiles {any} (1, 1, 1, 2, 5, 0)", .{counts}), std.mem.eql(usize, &counts, &.{ 1, 1, 1, 2, 5, 0 }));
    // the soak: the two groupings again and again, the one launch's outputs against the first's
    {
        var bad: usize = 0;
        for (0..soak_launches) |_| {
            var sa = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer sa.deinit();
            bad += @intFromBool(groupingDiff(got, try Two.run(r, sc, .{ pa.ptr, pb.ptr }, true, &bufs, sa.allocator())) != 0);
        }
        r.note(try std.fmt.bufPrint(&line, "group_tiles soak: {d} launches of the two groupings on one input, {d} differ from the first (ints)", .{ soak_launches, bad }), bad == 0);
    }
    // the control: one pick of B changed (expert 12's first to expert 15, which had none)
    const at = std.mem.indexOfScalar(i32, pb_host, 12).?;
    const changed = [1]i32{15};
    const moved = try Two.run(r, sc, .{ pa.ptr, pb.ptr }, true, &bufs, ha);
    r.report("group_tiles control: one pick moved from expert 12 to expert 15 (ints)", groupingDiff(want, moved), false);
}

// -- TF_DS_DEC_GROUP2 --------------------------------------------------------------------------------------------------

/// mx.decodeGrouping (a decode launch's grouping in one CTA) with the scan-and-rank kernel (TF_DS_DEC_GROUP2) and with the
/// one-CTA loop, each against the grouping computed on the host from the picks: off, nact, act and, for each expert, its
/// entries in entry order and their rows, exactly (the entries' places are fixed, not a set). Picks of 14 .. 4,095
/// entries over a layer's 385 experts (ids past 384 and -1 are another node's), a few experts holding most of them.
fn decGroup2(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    const e_n: usize = G_E;
    const slots: usize = G_SLOTS;
    const max_rs: usize = 4096;
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const sizes = [_]usize{ 4 * e_n, 4 * (e_n + 1), 4 * e_n, 4, 4 * max_rs, 4 * max_rs, 4 * max_rs };
    var bufs: [sizes.len]cuda.DeviceBuffer = undefined;
    for (&bufs, sizes) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b.*);
    }
    var sc = std.mem.zeroes(mx.ExpertScratch);
    sc.rows = max_rs / slots;
    sc.slots = slots;
    sc.cnt = bufs[0].ptr;
    sc.off = bufs[1].ptr;
    sc.fill = bufs[2].ptr;
    sc.nact = bufs[3].ptr;
    sc.act = bufs[4].ptr;
    sc.ent = bufs[5].ptr;
    sc.erow = bufs[6].ptr;
    const pick_dev = try cuda.DeviceBuffer.alloc(d, 4 * max_rs);
    try keep.append(ha, pick_dev);
    const Want = struct { off: []i32, nact: i32, act: []i32, ent: []i32, erow: []i32 };
    const Host = struct {
        /// The grouping of picks [rs] as the one-CTA kernel defines it.
        fn want(picks: []const i32, a: std.mem.Allocator) !Want {
            const off = try a.alloc(i32, e_n + 1);
            @memset(off, 0);
            for (picks) |e| {
                if (e >= 0 and e < e_n) off[@intCast(e)] += 1;
            }
            var run: i32 = 0;
            var act: std.ArrayList(i32) = .empty;
            for (0..e_n) |e| {
                const c = off[e];
                off[e] = run;
                run += c;
                if (c != 0) try act.append(a, @intCast(e));
            }
            off[e_n] = run;
            const ent = try a.alloc(i32, @intCast(run));
            const erow = try a.alloc(i32, @intCast(run));
            const fill = try a.alloc(i32, e_n);
            @memset(fill, 0);
            for (picks, 0..) |e, i| {
                if (e < 0 or e >= e_n) continue;
                const at: usize = @intCast(off[@intCast(e)] + fill[@intCast(e)]);
                fill[@intCast(e)] += 1;
                ent[at] = @intCast(i);
                erow[at] = @intCast(i / slots);
            }
            return .{ .off = off, .nact = @intCast(act.items.len), .act = act.items, .ent = ent, .erow = erow };
        }
        /// The device's grouping read back, differences from `w` counted (every array, entry for entry).
        fn diff(rr: *Run, s: mx.ExpertScratch, w: Want, a: std.mem.Allocator) !usize {
            const dd = rr.d;
            const off = try a.alloc(i32, e_n + 1);
            var nact: [1]i32 = undefined;
            var n: usize = @intFromBool(nact[0] != w.nact);
            for (off, w.off) |u, v| n += @intFromBool(u != v);
            const act = try a.alloc(i32, w.act.len);
            const ent = try a.alloc(i32, w.ent.len);
            const erow = try a.alloc(i32, w.ent.len);
            if (ent.len > 0) {
            }
            for (act, w.act) |u, v| n += @intFromBool(u != v);
            for (ent, w.ent) |u, v| n += @intFromBool(u != v);
            for (erow, w.erow) |u, v| n += @intFromBool(u != v);
            return n;
        }
    };
    var line: [256]u8 = undefined;
    const row_counts = [_]usize{ 2, 7, 20, 64, 200, 585 };
    var last: []i32 = &.{};
    var last_want: Want = undefined;
    for (row_counts) |rows| {
        const rs = rows * slots;
        const picks = try ha.alloc(i32, rs);
        for (picks, 0..) |*p, i| {
            const roll = rng.uintLessThan(usize, 100);
            p.* = if (i % slots == slots - 1) @intCast(e_n - 1) // the shared expert every row's last slot
                else if (roll < 30) @intCast(10 + rng.uintLessThan(usize, 3)) // three experts hold the skew
                else if (roll < 36) -1 // another node's
                else @intCast(rng.uintLessThan(usize, e_n + 40)); // past 384: another node's
        }
        const w = try Host.want(picks, ha);
        for ([_]bool{ false, true }) |g2| {
            for (bufs) |b| try d.check(d.api.cuMemsetD8Async(b.ptr, 0x7f, b.len, r.s.handle), "cuMemsetD8Async");
            r.k.group2 = g2;
            defer r.k.group2 = false;
            try mx.decodeGrouping(r.k, r.s, e_n, sc, pick_dev.ptr, rows);
            try r.s.synchronize();
            r.report(try std.fmt.bufPrint(&line, "dec_group2: {s}, {d} rows x {d} slots ({d} active experts, {d} entries) against the host's grouping (every array, entry for entry)", .{ if (g2) "the scan-and-rank kernel" else "the one-CTA kernel", rows, slots, w.nact, w.ent.len }), try Host.diff(r, sc, w, ha), true);
        }
        last = picks;
        last_want = w;
    }
    // the control: one pick of the last case moved to another expert; the scan-and-rank kernel's output differs from the unchanged host grouping
    last[0] = if (last[0] == 100) 101 else 100;
    for (bufs) |b| try d.check(d.api.cuMemsetD8Async(b.ptr, 0x7f, b.len, r.s.handle), "cuMemsetD8Async");
    r.k.group2 = true;
    defer r.k.group2 = false;
    try mx.decodeGrouping(r.k, r.s, e_n, sc, pick_dev.ptr, row_counts[row_counts.len - 1]);
    try r.s.synchronize();
    r.report("dec_group2 control: one pick moved to another expert", try Host.diff(r, sc, last_want, ha), false);
}

// -- TF_DS_TOPK_TILES ------------------------------------------------------------------------------------------------

/// The replay's rows at each width, and k (the indexer's).
const topk_rows: usize = 16;
const topk_k: usize = 512;

/// The scorer's key of a score at position t (kernels.py: the score's bits in total order above, ~t below).
fn scoreKey(sc: f32, t: usize) i64 {
    const b: i32 = if (sc == 0) 0 else @bitCast(sc);
    const o: i32 = if (b < 0) b ^ 0x7FFFFFFF else b;
    const hi: u64 = @as(u64, @bitCast(@as(i64, o))) << 32;
    return @bitCast(hi | (0xFFFFFFFF - @as(u64, t)));
}

fn topkI64Run(ctx: ?*anyopaque, _: dsv41.tri.Tri, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) anyerror!void {
    const rr: *Run = @ptrCast(@alignCast(ctx.?));
    try rr.ops.?.topkI64(rr.s, keys, ks, rows, n, k, top);
}

/// The exact top-k of a row's n keys (unique) on the host: the kk largest keys' positions ascending, each at or past
/// vis written -1 (tf_dsn_topk_tiles' and _topk_finish's output).
fn topkExact(keys: []const i64, kk: usize, vis: i64, out: []i64, a: std.mem.Allocator) !void {
    const tmp = try a.dupe(i64, keys);
    defer a.free(tmp);
    std.mem.sort(i64, tmp, {}, std.sort.desc(i64));
    const pos = try a.alloc(i64, kk);
    defer a.free(pos);
    for (tmp[0..kk], pos) |key, *p| p.* = @intCast(0xFFFFFFFF - (@as(u64, @bitCast(key)) & 0xFFFFFFFF));
    std.mem.sort(i64, pos, {}, std.sort.asc(i64));
    for (pos, out[0..kk]) |p, *o| o.* = if (p < vis) p else -1;
}

/// One width's keys: topk_rows rows of n keys at a stride of ks keys (ks >= n), then one tile past the last row; each
/// row's tile maxima over its n keys. Keys as the scorer makes them: a few thousand score values (ties broken by
/// position), some negative, zeros, -inf at or past the row's vis (by its index mod 8: every key, past n, fewer than k,
/// inside the last tile, none, half, two random). The keys past a row's n (ks > n) and the tile past the last row
/// outscore every row's keys, so any read at or past n changes the picks; a row that sees every key holds its largest
/// key but a head's at n - 1, so its last tile, partial or not, is among the k tiles. `head`: each row's keys 0 .. 62
/// outscore its others, each row's above the row before's (at ks == n, what a read past the row before's n reaches;
/// every key one launch can read is unique).
const TopkKeys = struct { keys: []i64, tmax: []i64, vis: []i64 };

fn topkKeys(n: usize, ks: usize, head: bool, rng: std.Random, a: std.mem.Allocator) !TopkKeys {
    const nt = (n + 63) / 64;
    const keys = try a.alloc(i64, topk_rows * ks + 64);
    const tmax = try a.alloc(i64, topk_rows * nt);
    const vis = try a.alloc(i64, topk_rows);
    for (0..topk_rows) |row| {
        const v: usize = switch (row % 8) {
            0 => n,
            1 => n + 9,
            2 => 256, // fewer than k visible
            3 => n - 3, // inside the last tile
            4 => 0, // none visible: every score -inf (but a head's)
            5 => n / 2,
            else => rng.uintLessThan(usize, n),
        };
        vis[row] = @intCast(v);
        const kr = keys[row * ks ..][0..ks];
        for (0..n) |t| {
            var sc: f32 = @as(f32, @floatFromInt(rng.uintLessThan(u32, 4000))) * 0.0625 - 30;
            if (t % 13 == 0) sc = 0;
            kr[t] = scoreKey(if (t < v) sc else -std.math.inf(f32), t);
        }
        for (n..ks) |t| kr[t] = scoreKey(1e30, t);
        if (v >= n) kr[n - 1] = scoreKey(1e20, n - 1);
        if (head) {
            const hs: f32 = 1e21 * @as(f32, @floatFromInt(row + 1));
            for (kr[0..63], 0..) |*k, t| k.* = scoreKey(hs, t);
        }
        for (0..nt) |b| tmax[row * nt + b] = std.mem.max(i64, kr[b * 64 .. @min(b * 64 + 64, n)]);
    }
    for (keys[topk_rows * ks ..], 0..) |*k, t| k.* = scoreKey(1e30, t);
    return .{ .keys = keys, .tmax = tmax, .vis = vis };
}

/// One width's checks at a row stride of ks keys (a tile past each row's last tile, or n: production's packed rows),
/// every output poisoned before its launch. mx.topkTiles' picks against the exact top-k; with `chain`, the pruned
/// chain's (tri_index.topkSelectPruned) against it as well, a chain that cannot run failing. At a partial last tile:
/// the exact picks reach it, and a launch told nt x 64 keys a row reads past n (the row's padding, or the next row's
/// head) and picks position n in every row that sees it. Row 0's largest key set to a score of -1e30, its tile's
/// maximum recomputed: the picks are the changed keys' exact top-k.
fn topkWidth(r: *Run, n: usize, ks: usize, head: bool, chain: bool, soak: bool, rng: std.Random) !void {
    const d = r.d;
    const R = topk_rows;
    const KK = topk_k;
    const ob = R * KK * 8;
    var line: [384]u8 = undefined;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const nt = (n + 63) / 64;
    const h = try topkKeys(n, ks, head, rng, ha);
    const kb = try upload(d, std.mem.sliceAsBytes(h.keys));
    const tb = try upload(d, std.mem.sliceAsBytes(h.tmax));
    const vb = try upload(d, std.mem.sliceAsBytes(h.vis));
    const every = try ha.alloc(i64, R);
    @memset(every, dsv41.tri_index.every_vis);
    const eb = try upload(d, std.mem.sliceAsBytes(every));
    var bufs: [4]cuda.DeviceBuffer = undefined; // tpos, cand, top, out
    for (&bufs, [_]usize{ R * KK * 8, R * KK * 64 * 8, R * KK * 8, ob }) |*b, nb| b.* = try cuda.DeviceBuffer.alloc(d, nb);
    for ([_]cuda.DeviceBuffer{ kb, tb, vb, eb } ++ bufs) |b| try keep.append(ha, b);
    const exact = try ha.alloc(i64, R * KK);
    for (0..R) |row| try topkExact(h.keys[row * ks ..][0..n], KK, h.vis[row], exact[row * KK ..][0..KK], ha);
    const xb = std.mem.sliceAsBytes(exact);
    r.k.topk_tiles = true;
    try poison(bufs[3].ptr, ob, d, r.s);
    try mx.topkTiles(r.k, r.s, kb.ptr, ks, tb.ptr, n, KK, vb.ptr, bufs[3].ptr, R);
    try r.s.synchronize();
    const got = try get(bufs[3].ptr, ob, d);
    defer free(got);
    r.report(try std.fmt.bufPrint(&line, "topk_tiles: {d} keys a row, rows {d} keys apart ({d} tiles{s}), {d} rows of every vis: the picks against the exact top-k", .{ n, ks, nt, if (head) ", each row's keys 0 .. 62 its largest" else "", R }), differs(got, xb), true);
    if (soak) {
        var bad: usize = 0;
        for (0..soak_launches) |_| {
            try poison(bufs[3].ptr, ob, d, r.s);
            try mx.topkTiles(r.k, r.s, kb.ptr, ks, tb.ptr, n, KK, vb.ptr, bufs[3].ptr, R);
            try r.s.synchronize();
            const again = try get(bufs[3].ptr, ob, d);
            defer free(again);
            bad += @intFromBool(differs(got, again) != 0);
        }
        r.note(try std.fmt.bufPrint(&line, "topk_tiles soak: {d} keys a row, rows {d} keys apart: {d} launches on one input, {d} differ from the first", .{ n, ks, soak_launches, bad }), bad == 0);
    }
    if (chain) {
        const pb: dsv41.tri_index.PrunedBufs = .{ .tpos = bufs[0].ptr, .cand = bufs[1].ptr, .every = eb.ptr };
        const top: dsv41.tri_index.TopK = .{ .top = bufs[2].ptr, .ctx = @ptrCast(r), .run = topkI64Run };
        try poison(bufs[3].ptr, ob, d, r.s);
        if (dsv41.tri_index.topkSelectPruned(r.t, kb.ptr, ks, tb.ptr, KK, vb.ptr, bufs[3].ptr, pb, top, R, n)) |_| {
            try r.s.synchronize();
            const want = try get(bufs[3].ptr, ob, d);
            defer free(want);
            r.report(try std.fmt.bufPrint(&line, "topk_tiles: {d} keys a row, rows {d} keys apart: the pruned chain's picks against the exact top-k", .{ n, ks }), differs(want, xb), true);
        } else |e| r.note(try std.fmt.bufPrint(&line, "topk_tiles: {d} keys a row, rows {d} keys apart: the pruned chain runs (it failed: {s})", .{ n, ks, @errorName(e) }), false);
    }
    if (n % 64 != 0) {
        // every row that sees all n keys picks position n - 1
        var reached = true;
        for (0..R) |row| {
            if (h.vis[row] >= n and exact[row * KK + KK - 1] != @as(i64, @intCast(n - 1))) reached = false;
        }
        r.note(try std.fmt.bufPrint(&line, "topk_tiles: {d} keys a row, rows {d} keys apart: the partial last tile among the exact picks of every row that sees it", .{ n, ks }), reached);
        // the control: the launch told nt * 64 keys a row reads past n, whose keys win
        try poison(bufs[3].ptr, ob, d, r.s);
        try mx.topkTiles(r.k, r.s, kb.ptr, ks, tb.ptr, nt * 64, KK, vb.ptr, bufs[3].ptr, R);
        try r.s.synchronize();
        const over = try get(bufs[3].ptr, ob, d);
        defer free(over);
        const ov: []const i64 = @alignCast(std.mem.bytesAsSlice(i64, over));
        var seen = true;
        for (0..R) |row| {
            if (h.vis[row] > n and std.mem.indexOfScalar(i64, ov[row * KK ..][0..KK], @intCast(n)) == null) seen = false;
        }
        const nd = differs(over, xb);
        r.note(try std.fmt.bufPrint(&line, "topk_tiles control: {d} keys a row, rows {d} keys apart, read as {d} (past n: {s}): {d} bytes differ, position {d} among the picks of every row that sees it", .{ n, ks, nt * 64, if (ks == n) "the next row's first keys" else "the padding", nd, n }), nd > 0 and seen);
    }
    // the control: row 0's largest key set to a score of -1e30 and its tile's maximum recomputed (a pick moves)
    const k0 = try ha.dupe(i64, h.keys[0..n]);
    const mx_t = std.mem.indexOfMax(i64, k0);
    k0[mx_t] = scoreKey(-1e30, mx_t);
    const tile = mx_t / 64;
    const low = [1]i64{k0[mx_t]};
    const m0 = [1]i64{std.mem.max(i64, k0[tile * 64 .. @min(tile * 64 + 64, n)])};
    const changed = try ha.dupe(i64, exact);
    try topkExact(k0, KK, h.vis[0], changed[0..KK], ha);
    try poison(bufs[3].ptr, ob, d, r.s);
    try mx.topkTiles(r.k, r.s, kb.ptr, ks, tb.ptr, n, KK, vb.ptr, bufs[3].ptr, R);
    try r.s.synchronize();
    const moved = try get(bufs[3].ptr, ob, d);
    defer free(moved);
    const md = differs(moved, std.mem.sliceAsBytes(changed));
    const cd = differs(std.mem.sliceAsBytes(changed), xb);
    r.note(try std.fmt.bufPrint(&line, "topk_tiles control: {d} keys a row, rows {d} keys apart, row 0's largest key (position {d}) set to a score of -1e30, its tile's maximum recomputed: {d} bytes differ from the changed keys' exact top-k, {d} from the unchanged keys'", .{ n, ks, mx_t, md, cd }), md == 0 and cd > 0);
}

/// mx.topkTiles against the exact top-k at 32,769 .. 524,288 keys a row: the rows a tile past their last tile apart,
/// with the pruned chain as a second reference; then production's packed rows (ks == n: an odd width leaves a row
/// 8-byte aligned, a read past n reaches the next row's head) at two odd widths and one even, on a random stream of
/// their own (the later sections draw what they drew before).
fn topkTiles(r: *Run, rng: std.Random) !void {
    defer r.k.topk_tiles = false;
    r.note("topk_tiles: 32768 keys a row (512 tiles, not pruned) take the chain's own selection", !dsv41.prompt.prunedOnTiles(true, 32768, topk_k));
    for ([_]usize{ 32769, 32832, 65536, 131072, 262143, 262144, 524288 }) |n| try topkWidth(r, n, (n + 63) / 64 * 64 + 64, false, true, false, rng);
    var packed_rng = std.Random.DefaultPrng.init(0x4b5332);
    for ([_]usize{ 32769, 65536, 262143 }) |n| try topkWidth(r, n, n, true, false, n == 262143, packed_rng.random());
}

// -- TF_DS_PASS_TAIL -------------------------------------------------------------------------------------------------

/// Whether a plan tiles its prompt on the resume grid with whole units but its final pass, which ends at the prompt
/// within the cap: the rule checkPasses enforces, over a plan already built.
fn planHolds(p: *const dsv41.lanes.Passes, len: usize, rows: usize) bool {
    if (p.len == 0) return false;
    var at: usize = 0;
    for (p.slice(), 0..) |q, i| {
        if (q.start != at or q.start % rows != 0) return false;
        if (i + 1 < p.len and q.n % rows != 0) return false;
        at = q.start + q.n;
    }
    const last = p.list[p.len - 1];
    return at == len and last.n <= 2 * rows;
}

/// Whether two plans are the same passes (first row and rows).
fn samePlan(a: *const dsv41.lanes.Passes, b: *const dsv41.lanes.Passes) bool {
    if (a.len != b.len) return false;
    for (a.slice(), b.slice()) |x, y| if (x.start != y.start or x.n != y.n) return false;
    return true;
}

/// The pass rule at the served prompt lengths: the served, D1 and merged-tail pass counts, and every merged plan's whole
/// non-final passes on the resume grid. A prompt whose last grid remainder is decode-sized keeps the baseline passes (the
/// control plan with a pass a row longer fails the rule).
fn passRule(r: *Run, _: std.Random) !void {
    const rows = dsv41.model.chunk_rows;
    const cap: u32 = @intCast(2 * rows);
    const Case = struct { len: usize, served: usize, d1: usize, d2a: usize };
    const cases = [_]Case{
        .{ .len = 546, .served = 1, .d1 = 1, .d2a = 1 },
        .{ .len = 1036, .served = 1, .d1 = 1, .d2a = 1 },
        .{ .len = 1133, .served = 1, .d1 = 1, .d2a = 1 },
        .{ .len = 2039, .served = 1, .d1 = 1, .d2a = 1 },
        .{ .len = 2225, .served = 2, .d1 = 2, .d2a = 1 },
        .{ .len = 3149, .served = 2, .d1 = 2, .d2a = 1 },
        .{ .len = 3229, .served = 2, .d1 = 2, .d2a = 1 },
        .{ .len = 4288, .served = 3, .d1 = 2, .d2a = 2 },
        .{ .len = 4703, .served = 3, .d1 = 2, .d2a = 2 },
        .{ .len = 6517, .served = 4, .d1 = 3, .d2a = 2 },
        .{ .len = 6767, .served = 4, .d1 = 3, .d2a = 2 },
        .{ .len = 8364, .served = 5, .d1 = 3, .d2a = 3 },
        .{ .len = 9180, .served = 5, .d1 = 3, .d2a = 3 },
        .{ .len = 12841, .served = 6, .d1 = 4, .d2a = 4 },
        .{ .len = 14524, .served = 7, .d1 = 5, .d2a = 4 },
        .{ .len = 17992, .served = 7, .d1 = 5, .d2a = 5 },
        .{ .len = 18797, .served = 8, .d1 = 6, .d2a = 5 },
        .{ .len = 37334, .served = 12, .d1 = 10, .d2a = 10 },
        .{ .len = 37588, .served = 12, .d1 = 10, .d2a = 10 },
        .{ .len = 156889, .served = 41, .d1 = 39, .d2a = 39 },
    };
    var counts = true;
    var plans = true;
    for (cases) |c| {
        const served = dsv41.lanes.checkPasses(c.len, 0, 2 * rows, 4 * rows, &.{cap}, &.{}, false) catch {
            counts = false;
            continue;
        };
        const d1 = dsv41.lanes.checkPasses(c.len, 0, 2 * rows, 0, &.{cap}, &.{}, false) catch {
            counts = false;
            continue;
        };
        const d2a = dsv41.lanes.checkPasses(c.len, 0, 2 * rows, 0, &.{cap}, &.{}, true) catch {
            counts = false;
            continue;
        };
        counts = counts and served.len == c.served and d1.len == c.d1 and d2a.len == c.d2a;
        plans = plans and planHolds(&d2a, c.len, rows);
    }
    // 4,097 rows: the 1-row tail is decode-sized and keeps the baseline passes, never the merged 2,049-row pass
    const ref = dsv41.lanes.checkPasses(2 * rows + 1, 0, 2 * rows, 0, &.{cap}, &.{}, false) catch return r.note("pass_rule: the 4,097-row plan runs", false);
    const tailed = dsv41.lanes.checkPasses(2 * rows + 1, 0, 2 * rows, 0, &.{cap}, &.{}, true) catch return r.note("pass_rule: the 4,097-row merged plan runs", false);
    plans = plans and samePlan(&ref, &tailed);
    var line: [320]u8 = undefined;
    r.note(try std.fmt.bufPrint(&line, "pass_rule: {d} prompt lengths' served, D1 and merged-tail pass counts are the expected ones", .{cases.len}), counts);
    r.note(try std.fmt.bufPrint(&line, "pass_rule: every merged-tail plan starts on the {d}-row grid and runs whole units but its final pass, which ends at the prompt within the {d}-row cap; a decode-sized remainder keeps the baseline passes", .{ rows, 2 * rows }), plans);
    const good = dsv41.lanes.checkPasses(14524, 0, 2 * rows, 0, &.{cap}, &.{}, true) catch return r.note("pass_rule: the merged-tail plan at 14,524 rows runs", false);
    var bad = good;
    bad.list[1].n += 1;
    r.note(try std.fmt.bufPrint(&line, "pass_rule control: the 14,524-row plan with its second pass a row longer ({d} rows) is refused", .{bad.list[1].n}), !planHolds(&bad, 14524, rows) and planHolds(&good, 14524, rows));
}

/// The indexer at a merged final pass's compressed key count against the count of the chunk it replaces: the same keys
/// and the same top-k picks. Each case is the changed pass's first row, its rows in the baseline and merged (compress ratio 2:
/// N = (start + n) / 2), and the query rows; each row's vis is its own position's count, with rows that see every key of
/// the baseline pass, its last three, half and none. The control: one visible key's code byte changed changes the keys.
fn passTail(r: *Run, rng: std.Random) !void {
    const d = r.d;
    const IH: usize = 32;
    const ID: usize = 128;
    const KK: usize = topk_k;
    const Case = struct { start: usize, today: usize, merged: usize, rows: usize };
    const cases = [_]Case{
        .{ .start = 0, .today = 1024, .merged = 1112, .rows = 64 }, // 2,225 prompt rows: 2,048 then 2,225
        .{ .start = 0, .today = 1024, .merged = 1614, .rows = 64 }, // 3,229
        .{ .start = 12288, .today = 7168, .merged = 7262, .rows = 64 }, // 14,524: the last 2,048-row chunk merged to 2,236
    };
    const keep_u = dsv41.tri_index.idx_u;
    defer dsv41.tri_index.idx_u = keep_u;
    dsv41.tri_index.idx_u = false;
    var line: [320]u8 = undefined;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var control: usize = 0;
    for (cases, 0..) |c, ci| {
        const nmax = @max(c.today, c.merged);
        const codes = try a.alloc(u8, nmax * ID / 2);
        rng.bytes(codes);
        for (codes[0..256], 0..) |*x, i| x.* = @intCast(i);
        const scales = try a.alloc(u8, nmax * ID / 32);
        for (scales) |*x| x.* = 112 + rng.uintLessThan(u8, 14);
        const vis = try a.alloc(i64, c.rows);
        for (vis, 0..) |*v, i| v.* = @intCast(@min(c.today, (c.start + i + 1) / 2));
        vis[c.rows - 4] = @intCast(c.today);
        vis[c.rows - 3] = @intCast(c.today - 3);
        vis[c.rows - 2] = @intCast(c.today / 2);
        vis[c.rows - 1] = 0;
        const q = try upload(d, try bf16s(c.rows * IH * ID, rng, 0.5, a));
        const w = try upload(d, try bf16s(c.rows * IH, rng, 0.25, a));
        const codes_b = try upload(d, codes);
        const scales_b = try upload(d, scales);
        const vis_b = try upload(d, std.mem.sliceAsBytes(vis));
        const keys_today = try cuda.DeviceBuffer.alloc(d, c.rows * c.today * 8);
        const keys_merged = try cuda.DeviceBuffer.alloc(d, c.rows * c.merged * 8);
        const tmax_today = try cuda.DeviceBuffer.alloc(d, c.rows * ((c.today + 63) / 64) * 8);
        const tmax_merged = try cuda.DeviceBuffer.alloc(d, c.rows * ((c.merged + 63) / 64) * 8);
        const out_today = try cuda.DeviceBuffer.alloc(d, c.rows * KK * 8);
        const out_merged = try cuda.DeviceBuffer.alloc(d, c.rows * KK * 8);
        const top_today = try cuda.DeviceBuffer.alloc(d, c.rows * KK * 8);
        const top_merged = try cuda.DeviceBuffer.alloc(d, c.rows * KK * 8);
        for ([_]cuda.DeviceBuffer{ q, w, codes_b, scales_b, vis_b, keys_today, keys_merged, tmax_today, tmax_merged, out_today, out_merged, top_today, top_merged }) |b| try keep.append(a, b);
        const k: dsv41.tri_index.IndexK = .{ .fp4 = .{ .codes = codes_b.ptr, .scales = scales_b.ptr } };
        try dsv41.tri_index.indexScore(r.t, q.ptr, k, w.ptr, vis_b.ptr, c.today, keys_today.ptr, null, true, tmax_today.ptr, false, c.rows, IH, ID);
        try dsv41.tri_index.indexScore(r.t, q.ptr, k, w.ptr, vis_b.ptr, c.merged, keys_merged.ptr, null, true, tmax_merged.ptr, false, c.rows, IH, ID);
        try r.s.synchronize();
        const kt = try get(keys_today.ptr, keys_today.len, d);
        defer free(kt);
        const km = try get(keys_merged.ptr, keys_merged.len, d);
        defer free(km);
        var keys_diff: usize = 0;
        for (0..c.rows) |row| keys_diff += differs(kt[row * c.today * 8 ..][0 .. c.today * 8], km[row * c.merged * 8 ..][0 .. c.today * 8]);
        const top_t: dsv41.tri_index.TopK = .{ .top = top_today.ptr, .ctx = @ptrCast(r), .run = topkI64Run };
        try dsv41.tri_index.topkSelect(r.t, keys_today.ptr, c.today, KK, vis_b.ptr, out_today.ptr, top_t, c.rows, c.today);
        const top_m: dsv41.tri_index.TopK = .{ .top = top_merged.ptr, .ctx = @ptrCast(r), .run = topkI64Run };
        try dsv41.tri_index.topkSelect(r.t, keys_merged.ptr, c.merged, KK, vis_b.ptr, out_merged.ptr, top_m, c.rows, c.merged);
        try r.s.synchronize();
        const ot = try get(out_today.ptr, out_today.len, d);
        defer free(ot);
        const om = try get(out_merged.ptr, out_merged.len, d);
        defer free(om);
        r.report(try std.fmt.bufPrint(&line, "pass_tail {d} rows from {d}: the indexer's keys at {d} and {d} compressed keys, the first {d} of each row", .{ c.merged, c.start, c.today, c.merged, c.today }), keys_diff, true);
        r.report(try std.fmt.bufPrint(&line, "pass_tail {d} rows from {d}: the top-k picks at {d} and {d} compressed keys", .{ c.merged, c.start, c.today, c.merged }), differs(ot, om), true);
        if (ci == 0) {
            // one code byte of the key at position 300, visible to the rows with vis past it
            const at = codes_b.ptr + 300 * (ID / 2) + 7;
            var was: [1]u8 = undefined;
            const changed = [1]u8{was[0] ^ 0x40};
            try dsv41.tri_index.indexScore(r.t, q.ptr, k, w.ptr, vis_b.ptr, c.merged, keys_merged.ptr, null, true, tmax_merged.ptr, false, c.rows, IH, ID);
            try r.s.synchronize();
            const km2 = try get(keys_merged.ptr, keys_merged.len, d);
            defer free(km2);
            for (0..c.rows) |row| control += differs(km[row * c.merged * 8 ..][0 .. c.merged * 8], km2[row * c.merged * 8 ..][0 .. c.merged * 8]);
        }
    }
    r.report("pass_tail control: one visible key's code byte changed at 1,112 compressed keys: the keys", control, false);
}

// -- TF_DS_ATTN_U ----------------------------------------------------------------------------------------------------

/// tri_attn.sparseAttn with and without `u` on random data with packed latents: the base set's _sparse_attn_part against
/// the extra set's twin _sparse_attn_part_u. Prompt chunks at the window's end (positions up to 1,048,575) of 5 .. 512
/// picks over 262,144 compressed rows (one split, FINAL; NBLK 5, 6, 8, 12, 20), two of them with their positions 8 bytes
/// into the buffer (the unaligned variants); small chunks on the ring (8 splits, then the merge) of 512, 16, 3 and 1
/// picks (a prompt of up to 16 tokens: NBLK 5, the pick count folded or not 16-divisible); and a chunk with q zero over
/// the full range of scale bytes (0x01 .. 0x7e: uniform attention, OUT the values' mean): OUT and the split partials
/// byte-equal; one picked row's code byte changed differs.
fn attnU(r: *Run, rng: std.Random) !void {
    const d = r.d;
    const H: usize = 16;
    const HD: usize = 512;
    const WIN: usize = 128;
    const RING: usize = WIN + 16;
    const end: usize = 1 << 20; // the window: positions below it
    const ncomp: usize = end / 4; // compressed rows at ratio 4 (int64 offsets either way)
    var line: [256]u8 = undefined;
    const tag = @tagName(tri_attn.attn_twin); // u the baseline twin; w, z the sections attn_w / attn_z compare against the same part
    if (r.t.extra == null) return r.note(try std.fmt.bufPrint(&line, "attn_{s}: the kit's extra set (aot-extra/) loads", .{tag}), false);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    // the latents: random code bytes (every value), E4M3 scales 2^-4 .. 1.75 (a soft attention a code byte can move)
    const codes = try ha.alloc(u8, ncomp * HD / 2);
    rng.bytes(codes);
    const scales = try ha.alloc(u8, ncomp * HD / 16);
    for (scales) |*x| x.* = 0x18 + rng.uintLessThan(u8, 0x3f - 0x18);
    const cb = try upload(d, codes);
    try keep.append(ha, cb);
    const sb = try upload(d, scales);
    try keep.append(ha, sb);
    const Case = struct { rows: usize, n_idx: usize, ring: bool, pos_off: usize = 0, flat: bool = false };
    const cases = [_]Case{
        .{ .rows = 256, .n_idx = 5, .ring = false },                 .{ .rows = 256, .n_idx = 40, .ring = false },
        .{ .rows = 256, .n_idx = 70, .ring = false },                .{ .rows = 256, .n_idx = 200, .ring = false },
        .{ .rows = 256, .n_idx = 512, .ring = false },               .{ .rows = 6, .n_idx = 512, .ring = true },
        .{ .rows = 6, .n_idx = 16, .ring = true },                   .{ .rows = 6, .n_idx = 3, .ring = true },
        .{ .rows = 6, .n_idx = 1, .ring = true },                    .{ .rows = 256, .n_idx = 200, .ring = false, .pos_off = 8 },
        .{ .rows = 256, .n_idx = 512, .ring = false, .pos_off = 8 }, .{ .rows = 256, .n_idx = 512, .ring = false, .flat = true },
    };
    for (cases) |cs| {
        const R = cs.rows;
        const start = end - R;
        const split = tri_attn.splits(R, false) > 1;
        const n_pad = if (split) cs.n_idx else (cs.n_idx + 15) / 16 * 16;
        const wrows: usize = if (cs.ring) RING else R + WIN;
        // (q zero: every score 0, the attention uniform; drawn last, so the cases before draw what they drew)
        const q = if (cs.flat) try upload(d, blk: {
            const z = try ha.alloc(u8, R * H * HD * 2);
            @memset(z, 0);
            break :blk z;
        }) else try upload(d, try bf16s(R * H * HD, rng, 0.5, ha));
        const wsrc = try upload(d, try bf16s(wrows * HD, rng, 0.5, ha));
        const idx = try ha.alloc(i64, R * n_pad);
        for (0..R) |i| for (0..n_pad) |j| {
            idx[i * n_pad + j] = if (j >= cs.n_idx or (j + i) % 29 == 7) -1 else @intCast(rng.uintLessThan(usize, (start + i + 1) / 4));
        };
        const ib = try upload(d, std.mem.sliceAsBytes(idx));
        // the positions, pos_off bytes into their buffer
        const pos = try ha.alloc(i64, R + cs.pos_off / 8);
        for (pos[cs.pos_off / 8 ..], 0..) |*x, i| x.* = @intCast(start + i);
        const pb = try upload(d, std.mem.sliceAsBytes(pos));
        var wlo_v = [1]i64{@intCast(start - WIN)};
        const wb = try upload(d, std.mem.sliceAsBytes(&wlo_v));
        const sink = try upload(d, try f32s(H, rng, ha));
        const out = try cuda.DeviceBuffer.alloc(d, R * H * HD * 2);
        const sp: usize = if (split) tri_attn.attn_splits else 1;
        const pm = try cuda.DeviceBuffer.alloc(d, R * H * sp * 4);
        const pl = try cuda.DeviceBuffer.alloc(d, R * H * sp * 4);
        const po = try cuda.DeviceBuffer.alloc(d, R * H * sp * HD * 4);
        for ([_]cuda.DeviceBuffer{ q, wsrc, ib, pb, wb, sink, out, pm, pl, po }) |b| try keep.append(ha, b);
        // (with q zero: scale bytes over the whole range but the E4M3 NaN, 0x01 .. 0x7e)
        const csb = if (cs.flat) blk: {
            const wide = try ha.alloc(u8, ncomp * HD / 16);
            for (wide) |*x| x.* = 1 + rng.uintLessThan(u8, 0x7e);
            const w = try upload(d, wide);
            try keep.append(ha, w);
            break :blk w;
        } else sb;
        var a: tri_attn.Attn = .{ .q = q.ptr, .out = out.ptr, .rows = R, .h = H, .hd = HD, .sink = sink.ptr, .wsrc = wsrc.ptr, .wsrc_rows = wrows, .wlo = wb.ptr, .ring = cs.ring, .comp = .{ .fp4 = .{ .codes = cb.ptr, .scales = csb.ptr } }, .idx = ib.ptr, .n_idx = cs.n_idx, .pos = pb.ptr + cs.pos_off, .scale = dsv41.prompt.scale(HD), .window = WIN, .parts = .{ .pm = pm.ptr, .pl = pl.ptr, .po = po.ptr } };
        // the outputs of a call: OUT, then the split partials
        const Got = struct {
            fn of(rr: *Run, x: tri_attn.Attn, bufs: [4]cuda.DeviceBuffer, sizes: [4]usize) ![]u8 {
                for (bufs, sizes) |b, n| try poison(b.ptr, n, rr.d, rr.s);
                try tri_attn.sparseAttn(rr.t, x);
                try rr.s.synchronize();
                var all: std.ArrayList(u8) = .empty;
                for (bufs, sizes) |b, n| {
                    const g = try get(b.ptr, n, rr.d);
                    defer free(g);
                    try all.appendSlice(std.heap.page_allocator, g);
                }
                return all.toOwnedSlice(std.heap.page_allocator);
            }
        };
        const bufs = [4]cuda.DeviceBuffer{ out, pm, pl, po };
        const sizes = [4]usize{ R * H * HD * 2, if (split) R * H * sp * 4 else 0, if (split) R * H * sp * 4 else 0, if (split) R * H * sp * HD * 4 else 0 };
        const want = try Got.of(r, a, bufs, sizes);
        defer std.heap.page_allocator.free(want);
        a.u = true;
        const got = try Got.of(r, a, bufs, sizes);
        defer std.heap.page_allocator.free(got);
        const nblk = WIN / 32 + if (split) tri.cdiv(n_pad, 32) else tri.pow2(tri.cdiv(n_pad, 32));
        r.report(try std.fmt.bufPrint(&line, "attn_{s}: {d} rows{s}, {d} picks (NBLK {d}{s}){s}: the twin's {s} against the part's", .{ tag, R, if (cs.ring) " on the ring" else " at the window's end", cs.n_idx, nblk, if (split) ", 8 splits" else ", one split", if (cs.pos_off != 0) ", positions 8 bytes into their buffer" else if (cs.flat) ", q zero, scale bytes 0x01 .. 0x7e" else "", if (split) "OUT and partials" else "OUT" }), differs(want, got), true);
        if (cs.n_idx == 512 and !cs.ring and cs.pos_off == 0 and !cs.flat) {
            // the control: a code byte of the first row's first pick changed
            const row: usize = @intCast(idx[0]);
            var one = [1]u8{codes[row * HD / 2] ^ 0x11};
            const moved = try Got.of(r, a, bufs, sizes);
            defer std.heap.page_allocator.free(moved);
            r.report(try std.fmt.bufPrint(&line, "attn_{s} control: a code byte of row 0's first pick changed", .{tag}), differs(want, moved), false);
        }
    }
    // (a call the extra set has no twin for would have run the part itself: the comparisons above would hold vacuously)
    r.note(try std.fmt.bufPrint(&line, "attn_{s}: every call found its twin in the extra set (no call sent to the part)", .{tag}), tri.twin_misses.load(.monotonic) == 0);
}

// -- TF_DS_IDX_PROJ_OWNED --------------------------------------------------------------------------------------------

const I_CAP: usize = 2048;
const I_QW: usize = 1280; // q_lora: wq_b's input
const I_D: usize = 5120; // hidden: weights_proj's input
const I_IH: usize = 32; // index heads
const I_ID: usize = 128; // index head dim
const I_RD: usize = 64; // rope dim
const I_START: usize = 70000; // the chunk's first position

const IdxBufs = struct { iq: cuda.DeviceBuffer, iq4: cuda.DeviceBuffer, iw: cuda.DeviceBuffer };

/// The indexer's projections of rows r (prompt.indexer's launches: mx.linear of wq_b, idxQueries, weights_proj) into
/// the buffers' rows r.
fn idxProject(r: *Run, sc: mx.Scratch, wqb: weights.Linear, qr: u64, x: u64, proj: u64, cos: u64, sin: u64, pos: u64, rows: dsv41.prompt.QRows, o: IdxBufs) !void {
    try mx.linear(r.k, r.s, sc, wqb, qr + rows.start * I_QW * 2, .bf16, I_QW, rows.n, o.iq.ptr + rows.start * I_IH * I_ID * 2, I_IH * I_ID, .bf16, false, null);
    try dsv41.prompt.idxQueries(r.t, o.iq.ptr, o.iq4.ptr, cos, sin, pos, rows, I_RD, I_IH, I_ID);
    try mx.bf16Gemm(r.k, r.s, x + rows.start * I_D * 2, I_D, proj, I_D, rows.n, I_IH, I_D, o.iw.ptr + rows.start * I_IH * 2, I_IH, .bf16, dsv41.exact.indexScale(I_ID, I_IH), true, 1, 0, 0, 0, null);
}

fn idxProjOwned(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var line: [256]u8 = undefined;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const wqb = try fp8Linear(d, I_IH * I_ID, I_QW, .fast, rng, ha, &keep);
    const qr = try upload(d, try bf16s(I_CAP * I_QW, rng, 0.5, ha));
    const x = try upload(d, try bf16s(I_CAP * I_D, rng, 0.5, ha));
    const proj = try upload(d, try bf16s(I_IH * I_D, rng, 0.05, ha));
    const tab = try ha.alloc(f32, (I_START + I_CAP) * (I_RD / 2));
    for (tab) |*v| v.* = rng.float(f32) * 2 - 1;
    const cos = try upload(d, std.mem.sliceAsBytes(tab));
    for (tab) |*v| v.* = rng.float(f32) * 2 - 1;
    const sin = try upload(d, std.mem.sliceAsBytes(tab));
    const ph = try ha.alloc(i64, I_CAP);
    for (ph, 0..) |*p, i| p.* = @intCast(I_START + i);
    const pos = try upload(d, std.mem.sliceAsBytes(ph));
    for ([_]cuda.DeviceBuffer{ qr, x, proj, cos, sin, pos }) |b| try keep.append(ha, b);
    var sb: [4]cuda.DeviceBuffer = undefined;
    for (&sb, mx.Scratch.sizes(I_CAP, I_QW, @max(mx.zFloats(wqb, I_CAP), 1))) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b.*);
    }
    const sc: mx.Scratch = .{ .q = sb[0].ptr, .s = sb[1].ptr, .z = sb[2].ptr, .fake = sb[3].ptr, .rows = I_CAP, .max_k = I_QW };
    var bufs: [2]IdxBufs = undefined; // the whole chunk's, the owned blocks'
    for (&bufs) |*b| {
        b.* = .{ .iq = try cuda.DeviceBuffer.alloc(d, I_CAP * I_IH * I_ID * 2), .iq4 = try cuda.DeviceBuffer.alloc(d, I_CAP * I_IH * I_ID * 2), .iw = try cuda.DeviceBuffer.alloc(d, I_CAP * I_IH * 2) };
        for ([_]cuda.DeviceBuffer{ b.iq, b.iq4, b.iw }) |x_| try keep.append(ha, x_);
    }
    const Read = struct {
        fn all(rr: *Run, b: IdxBufs) ![3][]u8 {
            try rr.s.synchronize();
            return .{ try get(b.iq.ptr, b.iq.len, rr.d), try get(b.iq4.ptr, b.iq4.len, rr.d), try get(b.iw.ptr, b.iw.len, rr.d) };
        }
    };
    for (bufs[0..1]) |b| for ([_]cuda.DeviceBuffer{ b.iq, b.iq4, b.iw }) |x_| try poison(x_.ptr, x_.len, d, r.s);
    try idxProject(r, sc, wqb, qr.ptr, x.ptr, proj.ptr, cos.ptr, sin.ptr, pos.ptr, .{ .start = 0, .n = I_CAP }, bufs[0]);
    const want = try Read.all(r, bufs[0]);
    defer for (want) |w| free(w);
    const names = [_][]const u8{ "iq", "iq4", "iw" };
    for ([_]mx.PfSplit{ .off, .a, .c6, .c7 }) |sw| for ([_]usize{ 512, 256, 1024 }) |q| {
        if (sw != .off and q != 512) continue; // the split configurations at the owned quarter
        for ([_]cuda.DeviceBuffer{ bufs[1].iq, bufs[1].iq4, bufs[1].iw }) |x_| try poison(x_.ptr, x_.len, d, r.s);
        r.k.split = sw;
        defer r.k.split = .off;
        var k0: usize = 0;
        while (k0 < I_CAP) : (k0 += q) try idxProject(r, sc, wqb, qr.ptr, x.ptr, proj.ptr, cos.ptr, sin.ptr, pos.ptr, .{ .start = k0, .n = q }, bufs[1]);
        const got = try Read.all(r, bufs[1]);
        defer for (got) |g| free(g);
        for (names, want, got) |nm, w, g| r.report(try std.fmt.bufPrint(&line, "idx_proj_owned q={d} (pf_split {s}): {s} at every owned block against the whole chunk's", .{ q, mx.PfSplit.name(@backingInt(sw)).?, nm }), differs(w, g), true);
    };
    // the control: one qr value of row 700 set to 64, the owned quarter holding it
    const at = qr.ptr + (700 * I_QW + 17) * 2;
    var was: [2]u8 = undefined;
    const big = [2]u8{ 0x80, 0x42 };
    try idxProject(r, sc, wqb, qr.ptr, x.ptr, proj.ptr, cos.ptr, sin.ptr, pos.ptr, .{ .start = 512, .n = 512 }, bufs[1]);
    const got = try Read.all(r, bufs[1]);
    defer for (got) |g| free(g);
    r.report("idx_proj_owned control: iq with one qr value of row 700 set to 64", differs(want[0][512 * I_IH * I_ID * 2 .. 1024 * I_IH * I_ID * 2], got[0][512 * I_IH * I_ID * 2 .. 1024 * I_IH * I_ID * 2]), false);
}

// -- TF_DS_BF16_ROWS -------------------------------------------------------------------------------------------------

fn bf16Rows(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const x = try upload(d, try bf16s(M * I_D, rng, 0.5, ha));
    const w = try upload(d, try bf16s(I_IH * I_D, rng, 0.05, ha));
    const y = try cuda.DeviceBuffer.alloc(d, M * I_IH * 2);
    for ([_]cuda.DeviceBuffer{ x, w, y }) |b| try keep.append(ha, b);
    const Launch = struct {
        fn run(rr: *Run, xp: u64, wp: u64, yb: cuda.DeviceBuffer, m: usize, rows_on: bool) ![]u8 {
            try poison(yb.ptr, yb.len, rr.d, rr.s);
            rr.k.bf16_rows = rows_on;
            defer rr.k.bf16_rows = false;
            try mx.bf16Gemm(rr.k, rr.s, xp, I_D, wp, I_D, m, I_IH, I_D, yb.ptr, I_IH, .bf16, dsv41.exact.indexScale(I_ID, I_IH), true, 1, 0, 0, 0, null);
            try rr.s.synchronize();
            return get(yb.ptr, m * I_IH * 2, rr.d);
        }
    };
    var diff: usize = 0;
        const want = try Launch.run(r, x.ptr, w.ptr, y, m, false);
        defer free(want);
        const got = try Launch.run(r, x.ptr, w.ptr, y, m, true);
        defer free(got);
        diff += differs(want, got);
    }
    // the control: one x value of row 3000 set to 64
    const want = try Launch.run(r, x.ptr, w.ptr, y, M, false);
    defer free(want);
    const at = x.ptr + (3000 * I_D + 77) * 2;
    var was: [2]u8 = undefined;
    const big = [2]u8{ 0x80, 0x42 };
    const got = try Launch.run(r, x.ptr, w.ptr, y, M, true);
    defer free(got);
    r.report("bf16_rows control: one x value of row 3000 set to 64", differs(want, got), false);
}

// -- TF_DS_PF_RUN128 -------------------------------------------------------------------------------------------------

const X_E: usize = 385; // 384 routed experts and the shared one (the last id)
const X_SETS: usize = 8; // distinct routed weight sets (expert e takes set e % 8)
const X_D: usize = 5120; // gate / up's K
const X_TILES: usize = 20; // the pack's 32-column tiles: 512 main, 128 rest
const X_REST: usize = 4;
const X_W: usize = X_TILES * 32; // 640: gate / up's columns a node computes
const X_MAIN: usize = (X_TILES - X_REST) * 32; // 512
const X_I: usize = 1152; // down's K: the rank's whole intermediate
const X_N: usize = 2560; // down's output columns a node
const X_SL: usize = 7; // a row's slots: 6 routed, the shared expert last

fn bytesOf(n: usize, rng: std.Random, a: std.mem.Allocator, comptime fp8: bool) ![]u8 {
    const v = try a.alloc(u8, n);
    for (v) |*x| {
        x.* = rng.int(u8);
        if (fp8 and x.* & 0x7f == 0x7f) x.* -= 1; // no E4M3 NaN
    }
    return v;
}

fn scalesOf(n: usize, rng: std.Random, a: std.mem.Allocator) ![]u8 {
    const v = try a.alloc(u8, n);
    for (v) |*x| x.* = 116 + rng.uintLessThan(u8, 12);
    return v;
}

    return b.ptr;
}
    var sets: [X_SETS][6]u64 = undefined;
    for (&sets) |*st| {
    const shared = [6]u64{
    };
    var tables: [6][]u64 = undefined;
    for (&tables, 0..) |*t, j| {
        t.* = try ha.alloc(u64, X_E);
        for (t.*, 0..) |*p, e| p.* = if (e == X_E - 1) shared[j] else sets[e % X_SETS][j];
    }
    var tp: [6]u64 = undefined;
    const owner = try ha.alloc(i32, X_E + 1);
    for (owner) |*o| o.* = @intFromBool(rng.boolean());
    owner[X_E] = 0;
    var ex = std.mem.zeroes(weights.Experts);
    ex.count = X_E;
    ex.dims = X_D;
    ex.width = X_MAIN;
    ex.down_k = X_I;
    ex.down_n = X_N;
    // (scales_match: every expertScaleLens range lies inside its tensor here)
    ex.native = .{ .w1 = tp[0], .w3 = tp[1], .s1 = tp[2], .s3 = tp[3], .w2 = tp[4], .s2 = tp[5], .owner = try upKeep(d, std.mem.sliceAsBytes(owner), keep, ha), .tiles = X_TILES, .rest_tiles = X_REST, .scales_match = true };
    var sb: [sz.len]u64 = undefined;
    for (&sb, sz) |*p, n| {
        const b = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b);
        p.* = b.ptr;
    }
        const nt = try cuda.DeviceBuffer.alloc(d, 4);
        try keep.append(ha, tl);
        try keep.append(ha, nt);
        var used: [6]i32 = undefined;
        for (0..6) |sl| {
            var e: i32 = undefined;
            while (true) {
                e = @intCast(rng.uintLessThan(usize, X_E - 1));
                if (std.mem.indexOfScalar(i32, used[0..sl], e) == null) break;
            }
            used[sl] = e;
            pick[row * X_SL + sl] = e;
            wts[row * X_SL + sl] = rng.float(f32) * 0.5;
        }
        pick[row * X_SL + 6] = X_E - 1;
        wts[row * X_SL + 6] = 1;
    }
    const h = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
    const out = try cuda.DeviceBuffer.alloc(d, R * X_N * 4);
    for ([_]cuda.DeviceBuffer{ x, hdn, pb, wb, h, out }) |b| try keep.append(ha, b);
    const Moe = struct {
        /// gate / up of rows m into h, then down of hdn's entries: h, down's entry outputs, the combined rows
        fn run(rr: *Run, e_: weights.Experts, s_: mx.ExpertScratch, xp: u64, hd: u64, pk: u64, wt: u64, hb: cuda.DeviceBuffer, ob: cuda.DeviceBuffer, m: usize, run128: bool) ![3][]u8 {
            rr.k.run128 = run128;
            defer rr.k.run128 = false;
            try poison(hb.ptr, hb.len, rr.d, rr.s);
            try poison(s_.out, m * X_SL * X_N * 4, rr.d, rr.s);
            try poison(ob.ptr, ob.len, rr.d, rr.s);
            try mx.gateUp(rr.k, rr.s, rr.d, e_, s_, xp, X_D, pk, wt, m, 7.0, hb.ptr, X_MAIN, false);
            try rr.s.synchronize();
            const hh = try get(hb.ptr, m * X_SL * X_W * 2, rr.d);
            errdefer free(hh);
            try mx.down(rr.k, rr.s, e_, s_, hd, X_I, pk, m, ob.ptr, X_N, false, false);
            try rr.s.synchronize();
            return .{ hh, try get(s_.out, m * X_SL * X_N * 4, rr.d), try get(ob.ptr, m * X_N * 4, rr.d) };
        }
    };
    const names = [_][]const u8{ "gate / up's bf16 pack", "down's fp32 entry outputs", "the combined fp32 rows" };
            const want = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, false);
            defer for (want) |w| free(w);
            const got = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, true);
            defer for (got) |g| free(g);
            for (names, want, got) |nm, w, g| r.report(try std.fmt.bufPrint(&line, "pf_run128 rows={d}: {s} on the 128-byte-run tiles against the baseline's", .{ m, nm }), differs(w, g), true);
        }
    // the controls: one x value of row 30 and one down input value of entry 30 x 7 set to 64
    const want = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, false);
    defer for (want) |w| free(w);
    const big = [2]u8{ 0x80, 0x42 };
    var was: [2]u8 = undefined;
    for ([_]u64{ x.ptr + (30 * X_D + 300) * 2, hdn.ptr + (30 * X_SL * X_I + 70) * 2 }, [_]usize{ 0, 1 }) |at, which| {
        const got = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, true);
        defer for (got) |g| free(g);
        r.report(if (which == 0) "pf_run128 control: gate / up's pack with one x value of row 30 set to 64" else "pf_run128 control: down's entry outputs with one input value of row 30's first slot set to 64", differs(want[which], got[which]), false);
    }
    // the staged pieces: one byte of weight set 0's (experts 0, 8, .. 376) changed at a time, each launch against the
    // twin's own unchanged output, which every one must change: a weight byte in each K block of a pair (and down's
    // K block 35, its tail pair's last), a scale byte in each stage word (and down's tail word)
    const nx = ex.native.?;
    var set0: [4]u64 = undefined; // gate's rows and scales, down's rows and scales
    // gate / up's pack is bf16 and clamped: column 5's scales of the pair's K blocks at 2^0 first, so that one changed
    // byte moves its sums by more than a pack value's rounding (down's entry outputs are fp32)
    const gsc = set0[1] + 5 * (X_D / 32);
    var gsc_was: [8]u8 = undefined;
    const unit: [8]u8 = @splat(127);
    const own = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, true);
    defer for (own) |w| free(w);
    const Piece = struct { at: u64, out: usize };
    var pieces: [22]Piece = undefined;
    var np: usize = 0;
    // gate / up: output column 5 (the pack's main block), rows of 2,560 bytes, scale rows of 160
    for (0..8) |kb| {
        pieces[np] = .{ .at = set0[0] + 5 * (X_D / 2) + kb * 16 + 3, .out = 0 };
        np += 1;
    }
    for ([_]usize{ 1, 6 }) |kb| {
        pieces[np] = .{ .at = set0[1] + 5 * (X_D / 32) + kb, .out = 0 };
        np += 1;
    }
    const gu_n = np;
    // down: output column 9, rows of 576 bytes, scale rows of 36; K blocks 32 .. 35 the tail pair's one stage
    for ([_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 35 }) |kb| {
        pieces[np] = .{ .at = set0[2] + 9 * (X_I / 2) + kb * 16 + 3, .out = 1 };
        np += 1;
    }
    for ([_]usize{ 1, 6, 34 }) |kb| {
        pieces[np] = .{ .at = set0[3] + 9 * (X_I / 32) + kb, .out = 1 };
        np += 1;
    }
    var fewest = [2]usize{ std.math.maxInt(usize), std.math.maxInt(usize) };
    for (pieces[0..np], 0..) |pc, i| {
        var b: [1]u8 = undefined;
        const was0 = b[0];
        // a weight byte: both codes' magnitudes; a scale byte: its block's scale doubled
        b[0] = if ((i < gu_n and i < 8) or (i >= gu_n and i < gu_n + 9)) b[0] ^ 0x77 else b[0] + 1;
        const got = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, true);
        defer for (got) |g| free(g);
        b[0] = was0;
        fewest[pc.out] = @min(fewest[pc.out], differs(own[pc.out], got[pc.out]));
    }
    r.note(try std.fmt.bufPrint(&line, "pf_run128 control: gate / up's pack with one byte of weight set 0 changed, in each K block of a pair (8 launches) and each stage word's scales (2): each differs from the twin's own (fewest {d} bytes)", .{fewest[0]}), fewest[0] > 0);
    r.note(try std.fmt.bufPrint(&line, "pf_run128 control: down's entry outputs with one byte of weight set 0 changed, in each K block of a pair and K block 35 of the tail pair (9 launches) and each stage word's scales, the tail's too (3): each differs from the twin's own (fewest {d} bytes)", .{fewest[1]}), fewest[1] > 0);
    // the soak: the twins at 512 rows, each launch's three outputs against the first's
    {
        const first = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 512, true);
        defer for (first) |f| free(f);
        var bad: usize = 0;
        for (1..soak_launches) |_| {
            const again = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 512, true);
            defer for (again) |g| free(g);
            var n: usize = 0;
            for (first, again) |f, g| n += differs(f, g);
            bad += @intFromBool(n != 0);
        }
        r.note(try std.fmt.bufPrint(&line, "pf_run128 soak: the twins at 512 rows, {d} launches on one input, {d} differ from the first (the pack, down's entry outputs, the combined rows)", .{ soak_launches, bad }), bad == 0);
    }
}

// -- TF_DS_HC_STRIDE -------------------------------------------------------------------------------------------------

/// A quarter-HC mix on the chunk's own planes (plane 0 at the owned rows' offset, `cap * D` elements between planes)
/// against the tight repack the baseline path reads (`rows * D` between them): PART and h_out byte-equal, at a cap other
/// than the launch's rows and an owned range past row 0; one plane value changed differs.
fn hcStride(r: *Run, rng: std.Random) !void {
    const D: usize = 5120;
    const HC: usize = 4;
    const WORLD: usize = 2;
    const BLOCKS: usize = 40;
    var line: [256]u8 = undefined;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const d = r.d;
    const fn_w = try upload(d, try f32s(24 * HC * D, rng, ha));
    const h = try upload(d, try bf16s(4096 * HC * D, rng, 0.5, ha));
    const post = try upload(d, try f32s(4096 * HC, rng, ha));
    const comb = try upload(d, try f32s(4096 * HC * HC, rng, ha));
    for ([_]cuda.DeviceBuffer{ fn_w, h, post, comb }) |b| try keep.append(ha, b);
    var diffs: [2]usize = @splat(0);
    for ([_]struct { cap: usize, rows: usize, start: usize }{ .{ .cap = 2048, .rows = 1024, .start = 1024 }, .{ .cap = 4096, .rows = 768, .start = 512 } }, 0..) |cs, ci| {
        const plane = cs.cap * D;
        const full_host = try f32s(WORLD * plane, rng, ha);
        const full = try upload(d, full_host);
        const tight_host = try ha.alloc(u8, WORLD * cs.rows * D * 4);
        for (0..WORLD) |w| @memcpy(tight_host[w * cs.rows * D * 4 ..][0 .. cs.rows * D * 4], full_host[w * plane * 4 + cs.start * D * 4 ..][0 .. cs.rows * D * 4]);
        const tight = try upload(d, tight_host);
        const P = cs.rows * BLOCKS * 32;
        var part: [2]cuda.DeviceBuffer = undefined;
        var out: [2]cuda.DeviceBuffer = undefined;
        part[0] = try cuda.DeviceBuffer.alloc(d, P * 4);
        part[1] = try cuda.DeviceBuffer.alloc(d, P * 4);
        out[0] = try cuda.DeviceBuffer.alloc(d, cs.rows * HC * D * 2);
        out[1] = try cuda.DeviceBuffer.alloc(d, cs.rows * HC * D * 2);
        for (part ++ out) |b| try keep.append(ha, b);
        _ = try tri_hc.hcMixPf(r.t, h.ptr, fn_w.ptr, part[0].ptr, .{ .gathered = tight.ptr, .world = WORLD, .h_out = out[0].ptr }, post.ptr, comb.ptr, cs.rows, D);
        _ = try tri_hc.hcMixPf(r.t, h.ptr, fn_w.ptr, part[1].ptr, .{ .gathered = full.ptr + cs.start * D * 4, .rs = plane, .world = WORLD, .h_out = out[1].ptr }, post.ptr, comb.ptr, cs.rows, D);
        try r.s.synchronize();
        const pa = try get(part[0].ptr, P * 4, d);
        defer free(pa);
        const pb = try get(part[1].ptr, P * 4, d);
        defer free(pb);
        const oa = try get(out[0].ptr, cs.rows * HC * D * 2, d);
        defer free(oa);
        const ob = try get(out[1].ptr, cs.rows * HC * D * 2, d);
        defer free(ob);
        diffs[ci] = differs(pa, pb) + differs(oa, ob);
        r.report(try std.fmt.bufPrint(&line, "hc_stride cap={d} rows={d} start={d}: the mix on the chunk's planes at their stride against the tight repack (PART and h_out)", .{ cs.cap, cs.rows, cs.start }), diffs[ci], true);
    }
    // the control: plane 0's first owned row of the second case, one column set to 64
    {
        const cs = .{ .cap = 4096, .rows = 768, .start = 512 };
        const plane = cs.cap * D;
        const full_host = try f32s(WORLD * plane, rng, ha);
        const full = try upload(d, full_host);
        const P = cs.rows * BLOCKS * 32;
        const part = try cuda.DeviceBuffer.alloc(d, P * 4);
        const out = try cuda.DeviceBuffer.alloc(d, cs.rows * HC * D * 2);
        for ([_]cuda.DeviceBuffer{ full, part, out }) |b| try keep.append(ha, b);
        _ = try tri_hc.hcMixPf(r.t, h.ptr, fn_w.ptr, part.ptr, .{ .gathered = full.ptr + cs.start * D * 4, .rs = plane, .world = WORLD, .h_out = out.ptr }, post.ptr, comb.ptr, cs.rows, D);
        try r.s.synchronize();
        const want_p = try get(part.ptr, P * 4, d);
        defer free(want_p);
        const want_o = try get(out.ptr, cs.rows * HC * D * 2, d);
        defer free(want_o);
        const at = full.ptr + (cs.start * D + 40) * 4;
        const big = [_]u8{ 0, 0, 0x80, 0x42 }; // 64.0f
        var was: [4]u8 = undefined;
        _ = try tri_hc.hcMixPf(r.t, h.ptr, fn_w.ptr, part.ptr, .{ .gathered = full.ptr + cs.start * D * 4, .rs = plane, .world = WORLD, .h_out = out.ptr }, post.ptr, comb.ptr, cs.rows, D);
        try r.s.synchronize();
        const got_p = try get(part.ptr, P * 4, d);
        defer free(got_p);
        const got_o = try get(out.ptr, cs.rows * HC * D * 2, d);
        defer free(got_o);
        r.report("hc_stride control: plane 0's first owned row, one column set to 64", differs(want_p, got_p) + differs(want_o, got_o), false);
    }
}

// -- TF_DS_HC_FUSED -------------------------------------------------------------------------------------------------

fn copyBuf(d: *const cuda.Driver, s: cuda.Stream, dst: u64, src: u64, n: usize) !void {
    try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, n, s.handle), "cuMemcpyDtoDAsync");
}

/// Two buffers' differing bytes.
fn cmpBytes(a: u64, b: u64, n: usize, d: *const cuda.Driver) !usize {
    const x = try get(a, n, d);
    defer free(x);
    const y = try get(b, n, d);
    defer free(y);
    return differs(x, y);
}

/// A prompt chunk's mHC mixes and finish in one launch (tri_hc.hcPrePf with TF_DS_HC_FUSED in effect, the extra set's
/// _hc_pre_pf_fused) against the baseline two kernels, at the engine's shapes: a quarter's 1,024 owned rows and a full
/// 4,096-row chunk, hidden 5,120 and two gathered planes; the normed rows, pre, post, comb and the posted streams
/// byte-equal, with the post fused in and without it; one stream value changed differs; every launch found the set's
/// variant.
fn hcFused(r: *Run, rng: std.Random) !void {
    const D: usize = 5120;
    const HC: usize = 4;
    const WORLD: usize = 2;
    const NB: usize = 40;
    var line: [256]u8 = undefined;
    const was = tri_hc.hc_fused;
    defer tri_hc.hc_fused = was;
    if (r.t.extra == null) {
        for (0..6) |_| r.note("hc_fused: the kit's extra set (aot-extra/) holds _hc_pre_pf_fused", false);
        return;
    }
    const misses0 = tri_hc.hc_fused_misses.load(.monotonic);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const d = r.d;
    const fn_w = try upload(d, try f32s(24 * HC * D, rng, ha));
    const scale = try upload(d, try f32s(3, rng, ha));
    const base = try upload(d, try f32s(24, rng, ha));
    const norm_w = try upload(d, try bf16s(D, rng, 0.5, ha));
    for ([_]cuda.DeviceBuffer{ fn_w, scale, base, norm_w }) |b| try keep.append(ha, b);
    for ([_]struct { rows: usize, posted: bool, label: []const u8 }{
        .{ .rows = 1024, .posted = true, .label = "a quarter's owned rows (1,024), the post fused in" },
        .{ .rows = 4096, .posted = true, .label = "a full chunk (4,096), the post fused in" },
        .{ .rows = 1024, .posted = false, .label = "1,024 rows, no post" },
        .{ .rows = 4096, .posted = false, .label = "4,096 rows, no post" },
    }) |cs| {
        const rows = cs.rows;
        const h = try upload(d, try bf16s(rows * HC * D, rng, 0.5, ha));
        const g = try upload(d, try f32s(WORLD * rows * D, rng, ha));
        const post0 = try upload(d, try f32s(rows * HC, rng, ha));
        const comb0 = try upload(d, try f32s(rows * HC * HC, rng, ha));
        const pre_in = try upload(d, try f32s(rows * HC, rng, ha));
        const part = try cuda.DeviceBuffer.alloc(d, rows * NB * 32 * 4);
        const out_a = try cuda.DeviceBuffer.alloc(d, rows * D * 2);
        const pre_a = try cuda.DeviceBuffer.alloc(d, rows * HC * 4);
        const post_a = try cuda.DeviceBuffer.alloc(d, rows * HC * 4);
        const comb_a = try cuda.DeviceBuffer.alloc(d, rows * HC * HC * 4);
        const hout_a = try cuda.DeviceBuffer.alloc(d, rows * HC * D * 2);
        for ([_]cuda.DeviceBuffer{ h, g, post0, comb0, pre_in, part, out_a, pre_a, post_a, comb_a, hout_a }) |b| try keep.append(ha, b);
        // the baseline: the posted mixes (or the plain partials), then the finish
        tri_hc.hc_fused = false;
        try copyBuf(d, r.s, post_a.ptr, post0.ptr, rows * HC * 4);
        try copyBuf(d, r.s, comb_a.ptr, comb0.ptr, rows * HC * HC * 4);
        if (cs.posted) {
            _ = try tri_hc.hcPrePf(r.t, h.ptr, fn_w.ptr, scale.ptr, base.ptr, pre_in.ptr, norm_w.ptr, 1e-20, 1e-6, 20, out_a.ptr, pre_a.ptr, post_a.ptr, comb_a.ptr, part.ptr, .{ .gathered = g.ptr, .world = WORLD, .h_out = hout_a.ptr }, rows, D);
        } else try tri_hc.hcPre(r.t, h.ptr, fn_w.ptr, scale.ptr, base.ptr, pre_in.ptr, norm_w.ptr, 1e-20, 1e-6, 20, out_a.ptr, pre_a.ptr, post_a.ptr, comb_a.ptr, part.ptr, rows, D);
        try r.s.synchronize();
        // the switch: one launch a row
        tri_hc.hc_fused = true;
        const post_b = try cuda.DeviceBuffer.alloc(d, rows * HC * 4);
        const comb_b = try cuda.DeviceBuffer.alloc(d, rows * HC * HC * 4);
        const out_b = try cuda.DeviceBuffer.alloc(d, rows * D * 2);
        const pre_b = try cuda.DeviceBuffer.alloc(d, rows * HC * 4);
        const hout_b = try cuda.DeviceBuffer.alloc(d, rows * HC * D * 2);
        for ([_]cuda.DeviceBuffer{ post_b, comb_b, out_b, pre_b, hout_b }) |b| try keep.append(ha, b);
        try copyBuf(d, r.s, post_b.ptr, post0.ptr, rows * HC * 4);
        try copyBuf(d, r.s, comb_b.ptr, comb0.ptr, rows * HC * HC * 4);
        if (cs.posted) {
            _ = try tri_hc.hcPrePf(r.t, h.ptr, fn_w.ptr, scale.ptr, base.ptr, pre_in.ptr, norm_w.ptr, 1e-20, 1e-6, 20, out_b.ptr, pre_b.ptr, post_b.ptr, comb_b.ptr, part.ptr, .{ .gathered = g.ptr, .world = WORLD, .h_out = hout_b.ptr }, rows, D);
        } else try tri_hc.hcPre(r.t, h.ptr, fn_w.ptr, scale.ptr, base.ptr, pre_in.ptr, norm_w.ptr, 1e-20, 1e-6, 20, out_b.ptr, pre_b.ptr, post_b.ptr, comb_b.ptr, part.ptr, rows, D);
        try r.s.synchronize();
        var n: usize = 0;
        n += try cmpBytes(out_a.ptr, out_b.ptr, rows * D * 2, d);
        n += try cmpBytes(pre_a.ptr, pre_b.ptr, rows * HC * 4, d);
        n += try cmpBytes(post_a.ptr, post_b.ptr, rows * HC * 4, d);
        n += try cmpBytes(comb_a.ptr, comb_b.ptr, rows * HC * HC * 4, d);
        if (cs.posted) n += try cmpBytes(hout_a.ptr, hout_b.ptr, rows * HC * D * 2, d);
        r.report(try std.fmt.bufPrint(&line, "hc_fused {s}: out, pre, post, comb{s}", .{ cs.label, if (cs.posted) ", the posted streams" else "" }), n, true);
        // the control: one stream value of row 3 set to 32
        if (cs.posted and rows == 4096) {
            const at = h.ptr + @as(u64, @intCast((3 * HC + 2) * D + 17)) * 2;
            var old: [2]u8 = undefined;
            const big = [2]u8{ 0x00, 0x42 };
            _ = try tri_hc.hcPrePf(r.t, h.ptr, fn_w.ptr, scale.ptr, base.ptr, pre_in.ptr, norm_w.ptr, 1e-20, 1e-6, 20, out_b.ptr, pre_b.ptr, post_b.ptr, comb_b.ptr, part.ptr, .{ .gathered = g.ptr, .world = WORLD, .h_out = hout_b.ptr }, rows, D);
            try r.s.synchronize();
            r.report("hc_fused control: one stream value of row 3 set to 32", try cmpBytes(out_a.ptr, out_b.ptr, rows * D * 2, d), false);
        }
    }
    r.note("hc_fused: every launch found the extra set's variant (none kept the baseline two kernels)", tri_hc.hc_fused_misses.load(.monotonic) == misses0);
}

// -- TF_DS_GEMM_R2 ---------------------------------------------------------------------------------------------------

/// mx.gemm of rows [0, m) with TF_DS_MMA_SF_EXACT's arithmetic in effect, TF_DS_GEMM_R2 `on` or not, outputs of `mode`
/// into y (poisoned first); their bytes (get's).
fn gemmR2Run(r: *Run, on: bool, l: weights.Linear, q: u64, s: u64, z: u64, m: usize, y: u64, mode: Mode) ![]u8 {
    r.k.sf = .exact;
    r.k.gemm_r2 = on;
    defer {
        r.k.sf = .off;
        r.k.gemm_r2 = false;
    }
    return gemmOnce(r, .off, false, l, q, s, z, 0, m, y, mode, null);
}

/// TF_DS_GEMM_R2's expert kernels against TF_DS_MMA_SF_EXACT's on the replay's shapes: the experts' gate / up pack,
/// down entry outputs and combined rows at 37 .. 4,096 rows (down on its 128-row tiles), their route asserted; one
/// dense shape's outputs with the switch on against off (the dense split GEMMs stay on the exact kernels); a changed x
/// value differs.
fn gemmR2(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x72a5);
    const rng = prng.random();
    const d = r.d;
    var line: [512]u8 = undefined;
    const carrying = if (r.k.sfk) |x| x.r2.gu != null and x.r2.dn != null else false;
    r.note("gemm_r2: the exact arithmetic's R2 kernels (native_sf.cu's _r2) loaded on this GPU", carrying);
    if (!carrying) return;
    defer {
        r.k.sf = .off;
        r.k.gemm_r2 = false;
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
    }
    {
        // the dense split GEMMs stay on TF_DS_MMA_SF_EXACT's kernels under the switch: the exact path holds two
        // accumulator arrays (the split's partial and the running sum) and is at the register limit, so a wider tile
        // or a vector fragment load spills. This line is one shape's bytes with the switch on against off.
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const sh = sf_shapes[0];
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
        const max_m = 4096;
        const x = try upload(d, try bf16s(max_m * sh.k, rng, 0.5, ha));
        try keep.append(ha, x);
        const q = try cuda.DeviceBuffer.alloc(d, max_m * sh.k);
        try keep.append(ha, q);
        const s = try cuda.DeviceBuffer.alloc(d, max_m * (sh.k / 32) * 4);
        try keep.append(ha, s);
        try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, max_m, sh.k, q.ptr, sh.k, s.ptr, sh.k / 32);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, max_m), 1) * 4);
        try keep.append(ha, z);
        const y = try cuda.DeviceBuffer.alloc(d, max_m * sh.n * 4);
        try keep.append(ha, y);
        r.k.sf = .exact;
        try mx.sfProve(r.k, l.s);
        var diff: usize = 0;
        for ([_]usize{ 64, 1024, 1025, 4096 }) |m| {
            for ([_]Mode{ .bf16, .f32 }) |mode| {
                const t = try gemmR2Run(r, false, l, q.ptr, s.ptr, z.ptr, m, y.ptr, mode);
                defer free(t);
                const e = try gemmR2Run(r, true, l, q.ptr, s.ptr, z.ptr, m, y.ptr, mode);
                defer free(e);
                diff += differs(t, e);
            }
        }
        r.report(try std.fmt.bufPrint(&line, "gemm_r2 {s} ({d} x {d}): the dense split GEMMs stay on TF_DS_MMA_SF_EXACT's kernels (their exact path's two accumulator arrays leave no register room for a wider tile or a vector fragment load): bf16 and fp32 outputs at 64, 1,024, 1,025 and 4,096 rows with the switch on against off", .{ sh.name, sh.k, sh.n }), diff, true);
    }
    {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const ex = try xExperts(d, rng, ha, &keep);
        const R: usize = sf_max_rows;
        const P = R * X_SL;
        const sc = try xScratch(d, R, true, ha, &keep);
        const x = try upload(d, try bf16s(R * X_D, rng, 0.5, ha));
        const hdn = try upload(d, try bf16s(P * X_I, rng, 0.5, ha));
        const pw = try xPicks(R, rng, ha);
        const pb = try upload(d, std.mem.sliceAsBytes(pw.pick));
        const wb = try upload(d, std.mem.sliceAsBytes(pw.wts));
        const h = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
        const out = try cuda.DeviceBuffer.alloc(d, R * X_N * 4);
        for ([_]cuda.DeviceBuffer{ x, hdn, pb, wb, h, out }) |b| try keep.append(ha, b);
        const Moe = struct {
            fn run(rr: *Run, e_: weights.Experts, s_: mx.ExpertScratch, xp: u64, hd: u64, pk: u64, wt: u64, hb: cuda.DeviceBuffer, ob: cuda.DeviceBuffer, m: usize, on: bool) ![3][]u8 {
                rr.k.run128 = true;
                rr.k.sf = .exact;
                rr.k.gemm_r2 = on;
                defer {
                    rr.k.run128 = false;
                    rr.k.sf = .off;
                    rr.k.gemm_r2 = false;
                }
                try poison(hb.ptr, hb.len, rr.d, rr.s);
                try poison(s_.out, m * X_SL * X_N * 4, rr.d, rr.s);
                try poison(ob.ptr, ob.len, rr.d, rr.s);
                try mx.gateUp(rr.k, rr.s, rr.d, e_, s_, xp, X_D, pk, wt, m, 7.0, hb.ptr, X_MAIN, false);
                try rr.s.synchronize();
                const hh = try get(hb.ptr, m * X_SL * X_W * 2, rr.d);
                errdefer free(hh);
                try mx.down(rr.k, rr.s, e_, s_, hd, X_I, pk, m, ob.ptr, X_N, false, false);
                try rr.s.synchronize();
                return .{ hh, try get(s_.out, m * X_SL * X_N * 4, rr.d), try get(ob.ptr, m * X_N * 4, rr.d) };
            }
        };
        const nx = ex.native.?;
        try mx.sfProve(r.k, nx.s1);
        r.k.sf = .exact;
        r.k.gemm_r2 = true;
        const gu_r2 = mx.sfExperts(r.k, nx.s1, false);
        const dn_r2 = mx.sfExperts(r.k, nx.s1, true);
        r.k.gemm_r2 = false;
        const route_bad = @intFromBool(!(gu_r2 != null and dn_r2 != null and gu_r2.?.f.handle == r.k.sfk.?.r2.gu.?.handle and dn_r2.?.f.handle == r.k.sfk.?.r2.dn.?.handle));
        const names = [_][]const u8{ "gate / up's bf16 pack", "down's fp32 entry outputs", "the combined fp32 rows" };
        var diff: [3]usize = .{ route_bad, 0, 0 };
        for ([_]usize{ 37, 64, 512, 2048, 4096 }) |m| {
            const want = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, false);
            defer for (want) |w| free(w);
            const got = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, true);
            defer for (got) |g| free(g);
            for (&diff, want, got) |*n, w, g| n.* += differs(w, g);
        }
        for (names, diff) |nm, n| r.report(try std.fmt.bufPrint(&line, "gemm_r2 experts: {s} at 37 .. 4,096 rows against TF_DS_MMA_SF_EXACT's 128-byte-run tiles, down on its 128-row tiles, the route asserted ({d} wrong)", .{ nm, route_bad }), n, true);
        // the control: one x value of row 30 changed, gate / up's pack differing
        const base = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, true);
        defer for (base) |b| free(b);
        const at = x.ptr + (30 * X_D + 300) * 2;
        var was: [2]u8 = undefined;
        const big = [2]u8{ 0x80, 0x42 };
        const ctl = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, true);
        defer for (ctl) |g| free(g);
        r.report("gemm_r2 experts control: gate / up's pack with one x value of row 30 set to 64", differs(base[0], ctl[0]), false);
    }
}

// -- TF_DS_ATTN_EPILOGUE ----------------------------------------------------------------------------------------------

/// The attention epilogue (inverse RoPE and wo_a's quant in one launch) against rope_heads followed by the quant, on
/// o [rows, 16, 512] at 1, 17 and 4,096 rows: o (roped), the codes and scales, and the fake rows byte-equal; one o
/// value in the roped columns changed differs.
fn attnEpilogue(r: *Run, rng: std.Random) !void {
    const H: usize = 16;
    const HD: usize = 512;
    const RD: usize = 64;
    const K = H * HD;
    const MAX: usize = 4096;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const d = r.d;
    const o = try upload(d, try bf16s(MAX * K, rng, 0.5, ha));
    const o_ref = try cuda.DeviceBuffer.alloc(d, MAX * K * 2);
    const o_fused = try cuda.DeviceBuffer.alloc(d, MAX * K * 2);
    const pos: []i64 = try ha.alloc(i64, MAX);
    for (pos, 0..) |*p, i| p.* = @intCast(i);
    const pb = try upload(d, std.mem.sliceAsBytes(pos));
    const cos = try upload(d, try f32s(MAX * (RD / 2), rng, ha));
    const sin = try upload(d, try f32s(MAX * (RD / 2), rng, ha));
    const qr = try cuda.DeviceBuffer.alloc(d, MAX * K);
    const sr = try cuda.DeviceBuffer.alloc(d, MAX * K / 32 * 4);
    const qf = try cuda.DeviceBuffer.alloc(d, MAX * K);
    const sf = try cuda.DeviceBuffer.alloc(d, MAX * K / 32 * 4);
    const fr = try cuda.DeviceBuffer.alloc(d, MAX * K * 2);
    const ff = try cuda.DeviceBuffer.alloc(d, MAX * K * 2);
    for ([_]cuda.DeviceBuffer{ o, o_ref, o_fused, pb, cos, sin, qr, sr, qf, sf, fr, ff }) |b| try keep.append(ha, b);
    var code_diff: usize = 0;
    var fake_diff: usize = 0;
    for ([_]usize{ 1, 17, 4096 }) |rows| {
        const bytes = rows * K * 2;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(o_ref.ptr, o.ptr, bytes, r.s.handle), "cuMemcpyDtoDAsync");
        try d.check(d.api.cuMemcpyDtoDAsync_v2(o_fused.ptr, o.ptr, bytes, r.s.handle), "cuMemcpyDtoDAsync");
        try tri_norm.ropeHeads(r.t, o_ref.ptr, cos.ptr, sin.ptr, pb.ptr, RD, true, rows, H, HD);
        try mx.quant(r.k, r.s, o_ref.ptr, .bf16, K, rows, K, qr.ptr, K, sr.ptr, K / 32);
        try mx.fake(r.k, r.s, o_ref.ptr, .bf16, K, rows, K, fr.ptr, K);
        try mx.attnEpilogue(r.k, r.s, o_fused.ptr, cos.ptr, sin.ptr, pb.ptr, rows, H, HD, RD, qf.ptr, sf.ptr, 0, true);
        try r.s.synchronize();
        const want_o = try get(o_ref.ptr, bytes, d);
        defer free(want_o);
        const got_o = try get(o_fused.ptr, bytes, d);
        defer free(got_o);
        code_diff += differs(want_o, got_o);
        const want_q = try get(qr.ptr, rows * K, d);
        defer free(want_q);
        const got_q = try get(qf.ptr, rows * K, d);
        defer free(got_q);
        code_diff += differs(want_q, got_q);
        const want_s = try get(sr.ptr, rows * K / 32 * 4, d);
        defer free(want_s);
        const got_s = try get(sf.ptr, rows * K / 32 * 4, d);
        defer free(got_s);
        code_diff += differs(want_s, got_s);
        // the fake rows from a fresh o (the fused launch writes o too)
        try d.check(d.api.cuMemcpyDtoDAsync_v2(o_fused.ptr, o.ptr, bytes, r.s.handle), "cuMemcpyDtoDAsync");
        try mx.attnEpilogue(r.k, r.s, o_fused.ptr, cos.ptr, sin.ptr, pb.ptr, rows, H, HD, RD, 0, 0, ff.ptr, true);
        try r.s.synchronize();
        const want_f = try get(fr.ptr, bytes, d);
        defer free(want_f);
        const got_f = try get(ff.ptr, bytes, d);
        defer free(got_f);
        fake_diff += differs(want_f, got_f);
    }
    r.report("attn_epilogue 1, 17 and 4096 rows of 16 heads of 512: the inverse RoPE and the quant in one launch against rope_heads then the quant (o, codes and scales)", code_diff, true);
    r.report("attn_epilogue: wo_a's fake rows in the same launch against rope_heads then the fake quant", fake_diff, true);
    // the control: o's row 3, column 500 (inside the roped columns) set to 64, both arms on a fresh o
    const rows: usize = 4;
    const bytes = rows * K * 2;
    try d.check(d.api.cuMemcpyDtoDAsync_v2(o_ref.ptr, o.ptr, bytes, r.s.handle), "cuMemcpyDtoDAsync");
    try tri_norm.ropeHeads(r.t, o_ref.ptr, cos.ptr, sin.ptr, pb.ptr, RD, true, rows, H, HD);
    try mx.quant(r.k, r.s, o_ref.ptr, .bf16, K, rows, K, qr.ptr, K, sr.ptr, K / 32);
    try r.s.synchronize();
    const want = try get(qr.ptr, rows * K, d);
    defer free(want);
    const at = o.ptr + (3 * K + 500) * 2;
    const big = [2]u8{ 0x80, 0x42 };
    var was: [2]u8 = undefined;
    try d.check(d.api.cuMemcpyDtoDAsync_v2(o_fused.ptr, o.ptr, bytes, r.s.handle), "cuMemcpyDtoDAsync");
    try mx.attnEpilogue(r.k, r.s, o_fused.ptr, cos.ptr, sin.ptr, pb.ptr, rows, H, HD, RD, qf.ptr, sf.ptr, 0, true);
    try r.s.synchronize();
    const got = try get(qf.ptr, rows * K, d);
    defer free(got);
    r.report("attn_epilogue control: o's row 3, column 500 set to 64", differs(want, got), false);
}

// -- TF_DS_ASSEMBLE_QUANT ---------------------------------------------------------------------------------------------

/// A prompt chunk's parity assembly (28,672 slot rows of a 1,152-column intermediate) with down's quant of it in one
/// launch against assemble_rest and the separate quant: the assembled rows, down's codes and scales byte-equal; one
/// value of pair 1's main rows set to 64 differs.
fn assembleQuant(r: *Run, rng: std.Random) !void {
    const ops = r.ops orelse return error.NoOps;
    const d = r.d;
    const R: usize = 4096;
    const P = R * X_SL;
    const ig = X_MAIN;
    const ir = X_W - X_MAIN;
    const w = 2 * ig + ir;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const part0 = try upload(d, try bf16s(P * (ig + ir), rng, 0.5, ha));
    const part1 = try upload(d, try bf16s(P * (ig + ir), rng, 0.5, ha));
    const pick = try ha.alloc(i32, P);
    for (pick) |*e| e.* = @intCast(rng.uintLessThan(usize, X_E));
    const owner = try ha.alloc(i32, X_E + 1);
    for (owner) |*o| o.* = @intFromBool(rng.boolean());
    owner[X_E] = 0;
    const pb = try upload(d, std.mem.sliceAsBytes(pick));
    const ob = try upload(d, std.mem.sliceAsBytes(owner));
    var xd: [2]cuda.DeviceBuffer = undefined;
    var hq: [2]cuda.DeviceBuffer = undefined;
    var hs: [2]cuda.DeviceBuffer = undefined;
    for (0..2) |i| {
        xd[i] = try cuda.DeviceBuffer.alloc(d, P * w * 2);
        hq[i] = try cuda.DeviceBuffer.alloc(d, P * w);
        hs[i] = try cuda.DeviceBuffer.alloc(d, P * w / 32 * 4);
    }
    for ([_]cuda.DeviceBuffer{ part0, part1, pb, ob } ++ xd ++ hq ++ hs) |b| try keep.append(ha, b);
    const s1: mx.ExpertScratch = .{ .rows = R, .slots = X_SL, .q = 0, .s = 0, .cnt = 0, .off = 0, .fill = 0, .act = 0, .nact = 0, .ent = 0, .erow = 0, .hq = hq[1].ptr, .hs = hs[1].ptr, .out = 0 };
    // baseline: assemble_rest, then down's quant of the assembled rows
    try poison(xd[0].ptr, xd[0].len, d, r.s);
    try ops.assembleRest(r.s, part0.ptr, part1.ptr, pb.ptr, ob.ptr, xd[0].ptr, P, ig, ir);
    try mx.quant(r.k, r.s, xd[0].ptr, .bf16, w, P, w, hq[0].ptr, w, hs[0].ptr, w / 32);
    try r.s.synchronize();
    const want_xd = try get(xd[0].ptr, P * w * 2, d);
    defer free(want_xd);
    const want_q = try get(hq[0].ptr, P * w, d);
    defer free(want_q);
    const want_s = try get(hs[0].ptr, P * w / 32 * 4, d);
    defer free(want_s);
    // the switch: the assembly and the quant in one launch
    try poison(xd[1].ptr, xd[1].len, d, r.s);
    try poison(hq[1].ptr, hq[1].len, d, r.s);
    try poison(hs[1].ptr, hs[1].len, d, r.s);
    try mx.assembleQuant(r.k, r.s, part0.ptr, part1.ptr, pb.ptr, ob.ptr, xd[1].ptr, P, ig, ir, s1);
    try r.s.synchronize();
    const got_xd = try get(xd[1].ptr, P * w * 2, d);
    defer free(got_xd);
    const got_q = try get(hq[1].ptr, P * w, d);
    defer free(got_q);
    const got_s = try get(hs[1].ptr, P * w / 32 * 4, d);
    defer free(got_s);
    r.report("assemble_quant 4096 rows x 7 slots, 1,152 columns: the assembly with down's quant in one launch against assemble_rest and the quant (rows, codes and scales)", differs(want_xd, got_xd) + differs(want_q, got_q) + differs(want_s, got_s), true);
    // the control: one value of pair 1's main row 3 set to 64 (that row's second half and its code)
    const at = part1.ptr + (3 * ig + 40) * 2;
    const big = [2]u8{ 0x80, 0x42 };
    var was: [2]u8 = undefined;
    try mx.assembleQuant(r.k, r.s, part0.ptr, part1.ptr, pb.ptr, ob.ptr, xd[1].ptr, P, ig, ir, s1);
    try r.s.synchronize();
    const moved_q = try get(hq[1].ptr, P * w, d);
    defer free(moved_q);
    r.report("assemble_quant control: down's codes with one value of pair 1's main row 3 set to 64", differs(want_q, moved_q), false);
}
// -- TF_DS_WOA_W8 ----------------------------------------------------------------------------------------------------

const W8_GROUPS: usize = 2; // a 2D node's wo_a groups
const W8_GN: usize = 1024; // o_lora
const W8_GK: usize = 4096; // a group's inputs

/// A layer's stacked FP8 wo_a (codes without NaN, UE8M0 scales `lo` ..) and the bf16 rows the loader would make of it
/// (weights.woABf16), a Linear a group.
fn w8Layer(d: *const cuda.Driver, groups: usize, gn: usize, gk: usize, codes: []const u8, scales: []const u8, a: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer)) !struct { w8: weights.Linear, rows: [4]weights.Linear } {
    const wb = try upload(d, codes);
    const sb = try upload(d, scales);
    try keep.append(a, wb);
    try keep.append(a, sb);
    var rows: [4]weights.Linear = undefined;
    for (0..groups) |g| {
        const v = try a.alloc(u16, gn * gk);
        for (0..gn) |i| for (0..gk) |j| {
            const r = g * gn + i;
            v[i * gk + j] = weights.woABf16(codes[r * gk + j], scales[(r / 32) * (gk / 32) + j / 32]);
        };
        const b = try upload(d, std.mem.sliceAsBytes(v));
        try keep.append(a, b);
        rows[g] = .{ .words = b.ptr, .w = b.ptr, .k = @intCast(gk), .n = @intCast(gn), .fmt = .bf16 };
    }
    return .{ .w8 = .{ .words = wb.ptr, .w = wb.ptr, .s = sb.ptr, .k = @intCast(gk), .n = @intCast(groups * gn), .fmt = .fp8 }, .rows = rows };
}

fn woaW8(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const n_all = W8_GROUPS * W8_GN;
    const kk = W8_GROUPS * W8_GK;
    const max_m = w8_rows[w8_rows.len - 1];
    // the layer: random codes (no NaN), scales 2^-17 .. 2^0
    const codes = try ha.alloc(u8, n_all * W8_GK);
    for (codes) |*c| {
        c.* = rng.int(u8);
        if (c.* & 0x7f == 0x7f) c.* -= 1;
    }
    const scales = try ha.alloc(u8, (n_all / 32) * (W8_GK / 32));
    for (scales) |*e| e.* = 110 + rng.uintLessThan(u8, 18);
    const lay = try w8Layer(d, W8_GROUPS, W8_GN, W8_GK, codes, scales, ha, &keep);
    const o = try upload(d, try bf16s(max_m * kk, rng, 0.5, ha));
    try keep.append(ha, o);
    var sb: [4]cuda.DeviceBuffer = undefined;
    for (&sb, mx.Scratch.sizes(max_m, kk, 1)) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b.*);
    }
    const sc: mx.Scratch = .{ .q = sb[0].ptr, .s = sb[1].ptr, .z = sb[2].ptr, .fake = sb[3].ptr, .rows = max_m, .max_k = kk };
    const u = try cuda.DeviceBuffer.alloc(d, max_m * n_all * 2);
    try keep.append(ha, u);
    var diff: usize = 0;
    for (w8_rows) |m| {
        try poison(u.ptr, m * n_all * 2, d, r.s);
        try mx.woA(r.k, r.s, sc, lay.rows[0..W8_GROUPS], o.ptr, kk, m, u.ptr, n_all, null);
        try r.s.synchronize();
        const want = try get(u.ptr, m * n_all * 2, d);
        defer free(want);
        try poison(u.ptr, m * n_all * 2, d, r.s);
        try mx.woA8Prefill(r.k, r.s, sc, lay.w8, W8_GROUPS, o.ptr, kk, m, u.ptr, n_all);
        try r.s.synchronize();
        const got = try get(u.ptr, m * n_all * 2, d);
        defer free(got);
        diff += differs(want, got);
    }
    // the control: one o value changed (its group's codes and scale another)
    {
        const m: usize = 256;
        try mx.woA(r.k, r.s, sc, lay.rows[0..W8_GROUPS], o.ptr, kk, m, u.ptr, n_all, null);
        try r.s.synchronize();
        const want = try get(u.ptr, m * n_all * 2, d);
        defer free(want);
        const at = o.ptr + (100 * kk + 4100) * 2;
        var was: [2]u8 = undefined;
        const big = [2]u8{ 0x80, 0x42 }; // 64.0
        try mx.woA8Prefill(r.k, r.s, sc, lay.w8, W8_GROUPS, o.ptr, kk, m, u.ptr, n_all);
        try r.s.synchronize();
        const got = try get(u.ptr, m * n_all * 2, d);
        defer free(got);
        r.report("woa_w8 control: one o value of row 100 set to 64", differs(want, got), false);
    }
    // the soak, at 1,024 rows
    {
        const m: usize = 1024;
        var first: ?[]u8 = null;
        defer if (first) |f| free(f);
        var bad: usize = 0;
        for (0..soak_launches) |_| {
            try poison(u.ptr, m * n_all * 2, d, r.s);
            try mx.woA8Prefill(r.k, r.s, sc, lay.w8, W8_GROUPS, o.ptr, kk, m, u.ptr, n_all);
            try r.s.synchronize();
            const got = try get(u.ptr, m * n_all * 2, d);
            if (first) |f| {
                bad += @intFromBool(differs(f, got) != 0);
                free(got);
            } else first = got;
        }
        var line: [128]u8 = undefined;
        r.note(try std.fmt.bufPrint(&line, "woa_w8 soak: {d} launches at {d} rows on one input, {d} differ from the first", .{ soak_launches, m, bad }), bad == 0);
    }
    // the conversion of every admitted (code, scale byte) pair whose product is finite: one group of 128 outputs and
    // 2,048 inputs (4 x 64 scale blocks), the activation's row i one-hot at input i (act_quant makes it exactly 1.0), so
    // u[i, n] is the converted weight [n, i] itself
    {
        const gn: usize = 128;
        const gk: usize = 2048;
        var admitted: [254]u8 = undefined;
        var na: usize = 0;
        for (0..256) |c| if (c & 0x7f != 0x7f) {
            admitted[na] = @intCast(c);
            na += 1;
        };
        var checked: usize = 0;
        var bad: usize = 0;
        for ([_][2]u8{ .{ 0, 246 }, .{ 247, 254 } }) |range| {
            const cw = try ha.alloc(u8, gn * gk);
            const cs = try ha.alloc(u8, (gn / 32) * (gk / 32));
            for (cs, 0..) |*e, b| e.* = range[0] + @as(u8, @intCast(b % (@as(usize, range[1]) - range[0] + 1)));
            for (0..gn) |i| for (0..gk) |j| {
                var c = admitted[(i * gk + j) % na];
                const e = cs[(i / 32) * (gk / 32) + j / 32];
                // a product past f32's range (scale bytes from 247 on): code 0 in its place
                if (!std.math.isFinite(@as(f32, @bitCast(@as(u32, weights.woABf16(c, e)) << 16)))) c = 0;
                cw[i * gk + j] = c;
            };
            var kp: std.ArrayList(cuda.DeviceBuffer) = .empty;
            defer for (kp.items) |*b| b.free();
            const one = try w8Layer(d, 1, gn, gk, cw, cs, ha, &kp);
            const eye = try ha.alloc(u16, gk * gk);
            @memset(eye, 0);
            for (0..gk) |i| eye[i * gk + i] = 0x3f80;
            const ob = try upload(d, std.mem.sliceAsBytes(eye));
            try kp.append(ha, ob);
            const ub = try cuda.DeviceBuffer.alloc(d, gk * gn * 2);
            try kp.append(ha, ub);
            try poison(ub.ptr, gk * gn * 2, d, r.s);
            try mx.woA8Prefill(r.k, r.s, sc, one.w8, 1, ob.ptr, gk, gk, ub.ptr, gn);
            try r.s.synchronize();
            const got = try get(ub.ptr, gk * gn * 2, d);
            defer free(got);
            const v = std.mem.bytesAsSlice(u16, got);
            for (0..gk) |i| for (0..gn) |n| {
                const want = weights.woABf16(cw[n * gk + i], cs[(n / 32) * (gk / 32) + i / 32]);
                const have = std.mem.littleToNative(u16, v[i * gn + n]);
                // a sum from +0 makes -0 +0: zeros compare without their sign here (woA's line compares them)
                bad += @intFromBool(have != want and !(have & 0x7fff == 0 and want & 0x7fff == 0));
                checked += 1;
            };
        }
    }
}

// -- TF_DS_QUANT_ONCE -------------------------------------------------------------------------------------------------

fn quantOnce(r: *Run, rng: std.Random) !void {
    const d = r.d;
    const D: usize = 5120;
    const QW: usize = 1280;
    const HD: usize = 512;
    const CAP: usize = 2048;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var lay: weights.Layer = .{ .idx = 0, .ratio = 0 };
    lay.wq_a = try fp8Linear(d, QW, D, .fast, rng, ha, &keep);
    lay.wkv = try fp8Linear(d, HD, D, .fast, rng, ha, &keep);
    const x = try upload(d, try bf16s(CAP * D, rng, 0.5, ha));
    try keep.append(ha, x);
    var sb: [4]cuda.DeviceBuffer = undefined;
    for (&sb, mx.Scratch.sizes(CAP, D, @max(mx.zFloats(lay.wq_a, CAP), mx.zFloats(lay.wkv, CAP)))) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b.*);
    }
    const sc: mx.Scratch = .{ .q = sb[0].ptr, .s = sb[1].ptr, .z = sb[2].ptr, .fake = sb[3].ptr, .rows = CAP, .max_k = D };
    var out: [2]cuda.DeviceBuffer = undefined; // qa, y
    for (&out, [_]usize{ QW, HD }) |*b, w| {
        b.* = try cuda.DeviceBuffer.alloc(d, CAP * w * 2);
        try keep.append(ha, b.*);
    }
    // the baseline front: mx.linear of each (mm's), then the quant-once front
    const Front = struct {
        fn today(rr: *Run, l: weights.Layer, s: mx.Scratch, xp: u64, n: usize, o: [2]cuda.DeviceBuffer) !void {
            try mx.linear(rr.k, rr.s, s, l.wq_a, xp, .bf16, 5120, n, o[0].ptr, 1280, .bf16, false, null);
            try mx.linear(rr.k, rr.s, s, l.wkv, xp, .bf16, 5120, n, o[1].ptr, 512, .bf16, false, null);
        }
        fn once(rr: *Run, l: weights.Layer, s: mx.Scratch, xp: u64, n: usize, o: [2]cuda.DeviceBuffer) !void {
            try dsv41.prompt.frontLinears(rr.k, rr.s, s, l, xp, 5120, .{ .start = 0, .n = n }, o[0].ptr, o[1].ptr, 512);
        }
    };
    // every arm on poisoned outputs and a poisoned scratch: an arm that quantizes part of x reads poison where the
    // other arm left its codes
    var diff = [2]usize{ 0, 0 };
    for ([_]usize{ 1, 17, 63, 64, 100, 128, 129, 1000, 2048 }) |n| {
        var want: [2][]u8 = undefined;
        for (out ++ sb) |b| try poison(b.ptr, b.len, d, r.s);
        try Front.today(r, lay, sc, x.ptr, n, out);
        try r.s.synchronize();
        for (&want, out, [_]usize{ QW, HD }) |*h, b, w| h.* = try get(b.ptr, n * w * 2, d);
        defer for (want) |h| free(h);
        for (out ++ sb) |b| try poison(b.ptr, b.len, d, r.s);
        try Front.once(r, lay, sc, x.ptr, n, out);
        try r.s.synchronize();
        for (&diff, want, out) |*df, h, b| {
            const got = try get(b.ptr, h.len, d);
            defer free(got);
            df.* += differs(h, got);
        }
    }
    r.report("quant_once: qa (wq_a) on x quantized once against mx.linear's, at 1, 17, 63, 64, 100, 128, 129, 1000, 2048 rows", diff[0], true);
    r.report("quant_once: y (wkv) on x quantized once against mx.linear's, at the same row counts", diff[1], true);
    // the control: one x value of row 700 set to 64 (its group's scale another, every code of the group)
    {
        const n = CAP;
        for (out ++ sb) |b| try poison(b.ptr, b.len, d, r.s);
        try Front.today(r, lay, sc, x.ptr, n, out);
        try r.s.synchronize();
        const want = try get(out[0].ptr, n * QW * 2, d);
        defer free(want);
        const at = x.ptr + (700 * D + 33) * 2;
        var was: [2]u8 = undefined;
        const big = [2]u8{ 0x80, 0x42 }; // 64.0
        for (out ++ sb) |b| try poison(b.ptr, b.len, d, r.s);
        try Front.once(r, lay, sc, x.ptr, n, out);
        try r.s.synchronize();
        const got = try get(out[0].ptr, n * QW * 2, d);
        defer free(got);
        r.report("quant_once control: qa with one x value of row 700 set to 64", differs(want, got), false);
    }
}

// -- TF_DS_ATTN_U_DECODE ---------------------------------------------------------------------------------------------

/// tri_attn.sparseAttn of a decode round's shapes with and without `u`: the base set's _sparse_attn_part against the
/// extra set's twin _sparse_attn_part_u on packed latents, on the pool's ring (as round.zig calls it; a round is at
/// most 16 rows: 8 splits, then the merge): 1, 6 and 16 rows with each row's ring and compressed bases and 512 picks
/// (NBLK 20), 6 rows without bases, and 1 row of 20 picks (NBLK 5): OUT and the split partials byte-equal; one picked
/// row's code byte changed differs; every call found its twin.
fn attnUDecode(r: *Run, rng: std.Random) !void {
    const d = r.d;
    const H: usize = 16;
    const HD: usize = 512;
    const WIN: usize = 128;
    const RING: usize = WIN + 16; // a stream's ring rows
    const STREAMS: usize = 8;
    const ncomp: usize = 1 << 16; // the pool's compressed rows
    var line: [256]u8 = undefined;
    const tag = @tagName(tri_attn.attn_twin);
    if (r.t.extra == null) return r.note(try std.fmt.bufPrint(&line, "attn_{s}_decode: the kit's extra set (aot-extra/) loads", .{tag}), false);
    const misses = tri.twin_misses.load(.monotonic);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    // the latents: random code bytes (every value), E4M3 scales 2^-4 .. 1.875; the pool's ring of 8 streams
    const codes = try ha.alloc(u8, ncomp * HD / 2);
    rng.bytes(codes);
    const scales = try ha.alloc(u8, ncomp * HD / 16);
    for (scales) |*x| x.* = 0x18 + rng.uintLessThan(u8, 0x3f - 0x18);
    const cb = try upload(d, codes);
    const sb = try upload(d, scales);
    const ring = try upload(d, try bf16s(STREAMS * RING * HD, rng, 0.5, ha));
    const zbytes: [8]u8 = @splat(0);
    const zero = try upload(d, &zbytes);
    for ([_]cuda.DeviceBuffer{ cb, sb, ring, zero }) |b| try keep.append(ha, b);
    const Case = struct { rows: usize, n_idx: usize, based: bool };
    for ([_]Case{ .{ .rows = 1, .n_idx = 512, .based = true }, .{ .rows = 6, .n_idx = 512, .based = true }, .{ .rows = 16, .n_idx = 512, .based = true }, .{ .rows = 6, .n_idx = 512, .based = false }, .{ .rows = 1, .n_idx = 20, .based = false } }) |cs| {
        const R = cs.rows;
        const split = tri_attn.splits(R, false) > 1;
        const n_pad = if (split) cs.n_idx else (cs.n_idx + 15) / 16 * 16;
        const q = try upload(d, try bf16s(R * H * HD, rng, 0.5, ha));
        // each row: a stream's ring base and compressed base, a position past the window, picks below its visible count
        const wbase = try ha.alloc(i64, R);
        const cbase = try ha.alloc(i64, R);
        const pos = try ha.alloc(i64, R);
        const idx = try ha.alloc(i64, R * n_pad);
        for (0..R) |i| {
            wbase[i] = @intCast(rng.uintLessThan(usize, STREAMS) * RING);
            cbase[i] = @intCast(rng.uintLessThan(usize, ncomp / 2));
            pos[i] = @intCast(4000 + rng.uintLessThan(usize, 100000));
            for (0..n_pad) |j| idx[i * n_pad + j] = if (j >= cs.n_idx or (j + i) % 29 == 7) -1 else @intCast(rng.uintLessThan(usize, ncomp / 2));
        }
        const ib = try upload(d, std.mem.sliceAsBytes(idx));
        const pb = try upload(d, std.mem.sliceAsBytes(pos));
        const wbb = try upload(d, std.mem.sliceAsBytes(wbase));
        const cbb = try upload(d, std.mem.sliceAsBytes(cbase));
        const sink = try upload(d, try f32s(H, rng, ha));
        const out = try cuda.DeviceBuffer.alloc(d, R * H * HD * 2);
        const sp: usize = if (split) tri_attn.attn_splits else 1;
        const pm = try cuda.DeviceBuffer.alloc(d, R * H * sp * 4);
        const pl = try cuda.DeviceBuffer.alloc(d, R * H * sp * 4);
        const po = try cuda.DeviceBuffer.alloc(d, R * H * sp * HD * 4);
        for ([_]cuda.DeviceBuffer{ q, ib, pb, wbb, cbb, sink, out, pm, pl, po }) |b| try keep.append(ha, b);
        var a: tri_attn.Attn = .{ .q = q.ptr, .out = out.ptr, .rows = R, .h = H, .hd = HD, .sink = sink.ptr, .wsrc = ring.ptr, .wsrc_rows = if (cs.based) STREAMS * RING else RING, .wlo = zero.ptr, .ring = true, .comp = .{ .fp4 = .{ .codes = cb.ptr, .scales = sb.ptr } }, .idx = ib.ptr, .n_idx = cs.n_idx, .pos = pb.ptr, .scale = dsv41.prompt.scale(HD), .window = WIN, .wbase = if (cs.based) wbb.ptr else null, .cbase = if (cs.based) cbb.ptr else null, .ring_rows = RING, .parts = .{ .pm = pm.ptr, .pl = pl.ptr, .po = po.ptr } };
        const Got = struct {
            fn of(rr: *Run, x: tri_attn.Attn, bufs: [4]cuda.DeviceBuffer, sizes: [4]usize) ![]u8 {
                for (bufs, sizes) |b, n| try poison(b.ptr, n, rr.d, rr.s);
                try tri_attn.sparseAttn(rr.t, x);
                try rr.s.synchronize();
                var all: std.ArrayList(u8) = .empty;
                for (bufs, sizes) |b, n| {
                    const g = try get(b.ptr, n, rr.d);
                    defer free(g);
                    try all.appendSlice(std.heap.page_allocator, g);
                }
                return all.toOwnedSlice(std.heap.page_allocator);
            }
        };
        const bufs = [4]cuda.DeviceBuffer{ out, pm, pl, po };
        const sizes = [4]usize{ R * H * HD * 2, if (split) R * H * sp * 4 else 0, if (split) R * H * sp * 4 else 0, if (split) R * H * sp * HD * 4 else 0 };
        const want = try Got.of(r, a, bufs, sizes);
        defer std.heap.page_allocator.free(want);
        a.u = true;
        const got = try Got.of(r, a, bufs, sizes);
        defer std.heap.page_allocator.free(got);
        r.report(try std.fmt.bufPrint(&line, "attn_{s}_decode: {d} rows on the pool's ring{s}, {d} picks ({s}): the twin's {s} against the part's", .{ tag, R, if (cs.based) " with ring and compressed bases" else "", cs.n_idx, if (split) "8 splits" else "one split", if (split) "OUT and partials" else "OUT" }), differs(want, got), true);
        if (R == 6 and cs.based) {
            // the control: a code byte of row 0's first pick changed (its pool row: the pick past the row's base)
            const row: usize = @intCast(idx[0] + cbase[0]);
            var one = [1]u8{codes[row * HD / 2] ^ 0x11};
            const moved = try Got.of(r, a, bufs, sizes);
            defer std.heap.page_allocator.free(moved);
            r.report(try std.fmt.bufPrint(&line, "attn_{s}_decode control: a code byte of row 0's first pick changed", .{tag}), differs(want, moved), false);
        }
    }
    r.note(try std.fmt.bufPrint(&line, "attn_{s}_decode: every call found its twin in the extra set (no call sent to the part)", .{tag}), tri.twin_misses.load(.monotonic) == misses);
}

// -- TF_DS_MERGE_D ---------------------------------------------------------------------------------------------------

/// tri_attn.sparseAttn of a decode round's shapes (1, 2, 6 and 16 rows of 16 heads, 8 splits, a window and no picks, on the
/// pool's ring with per-row bases) with the single-program merge (tri_attn.merge_db 0) against the merge's column twin at 128
/// at 64 and at 16 columns a program: OUT and the split partials byte-equal; a ring value changed differs; every call found its
/// twin in the extra set.
fn mergeD(r: *Run, rng: std.Random) !void {
    const d = r.d;
    const H: usize = 16;
    const HD: usize = 512;
    const WIN: usize = 128;
    const RING: usize = WIN + 16;
    const STREAMS: usize = 8;
    var line: [256]u8 = undefined;
    if (r.t.extra == null) return r.note("merge_d: the kit's extra set (aot-extra/) loads", false);
    const was_db = tri_attn.merge_db;
    defer tri_attn.merge_db = was_db;
    const misses = tri.twin_misses.load(.monotonic);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const ring_host = try bf16s(STREAMS * RING * HD, rng, 0.5, ha);
    const ring = try upload(d, ring_host);
    const zbytes: [8]u8 = @splat(0);
    const zero = try upload(d, &zbytes);
    for ([_]cuda.DeviceBuffer{ ring, zero }) |b| try keep.append(ha, b);
    const Got = struct {
        fn of(rr: *Run, x: tri_attn.Attn, bufs: [4]cuda.DeviceBuffer, sizes: [4]usize, db: usize) ![]u8 {
            tri_attn.merge_db = db;
            for (bufs, sizes) |b, n| try poison(b.ptr, n, rr.d, rr.s);
            try tri_attn.sparseAttn(rr.t, x);
            try rr.s.synchronize();
            var all: std.ArrayList(u8) = .empty;
            for (bufs, sizes) |b, n| {
                const g = try get(b.ptr, n, rr.d);
                defer free(g);
                try all.appendSlice(std.heap.page_allocator, g);
            }
            return all.toOwnedSlice(std.heap.page_allocator);
        }
    };
    var control_done = false;
    for ([_]usize{ 1, 2, 6, 16 }) |R| {
        const q = try upload(d, try bf16s(R * H * HD, rng, 0.5, ha));
        const wbase = try ha.alloc(i64, R);
        const pos = try ha.alloc(i64, R);
        for (0..R) |i| {
            wbase[i] = @intCast(rng.uintLessThan(usize, STREAMS) * RING);
            pos[i] = @intCast(4000 + rng.uintLessThan(usize, 100000));
        }
        const pb = try upload(d, std.mem.sliceAsBytes(pos));
        const wbb = try upload(d, std.mem.sliceAsBytes(wbase));
        const sink = try upload(d, try f32s(H, rng, ha));
        const out = try cuda.DeviceBuffer.alloc(d, R * H * HD * 2);
        const sp: usize = tri_attn.attn_splits;
        const pm = try cuda.DeviceBuffer.alloc(d, R * H * sp * 4);
        const pl = try cuda.DeviceBuffer.alloc(d, R * H * sp * 4);
        const po = try cuda.DeviceBuffer.alloc(d, R * H * sp * HD * 4);
        for ([_]cuda.DeviceBuffer{ q, pb, wbb, sink, out, pm, pl, po }) |b| try keep.append(ha, b);
        const a: tri_attn.Attn = .{ .q = q.ptr, .out = out.ptr, .rows = R, .h = H, .hd = HD, .sink = sink.ptr, .wsrc = ring.ptr, .wsrc_rows = STREAMS * RING, .wlo = zero.ptr, .ring = true, .comp = .none, .idx = pb.ptr, .n_idx = 0, .pos = pb.ptr, .scale = dsv41.prompt.scale(HD), .window = WIN, .wbase = wbb.ptr, .cbase = wbb.ptr, .ring_rows = RING, .parts = .{ .pm = pm.ptr, .pl = pl.ptr, .po = po.ptr } };
        const bufs = [4]cuda.DeviceBuffer{ out, pm, pl, po };
        const sizes = [4]usize{ R * H * HD * 2, R * H * sp * 4, R * H * sp * 4, R * H * sp * HD * 4 };
        const want = try Got.of(r, a, bufs, sizes, 0);
        defer std.heap.page_allocator.free(want);
        for ([_]usize{ 128, 64, 16 }) |db| {
            const got = try Got.of(r, a, bufs, sizes, db);
            defer std.heap.page_allocator.free(got);
            r.report(try std.fmt.bufPrint(&line, "merge_d: {d} rows x {d} heads, 8 splits, {d} columns a program: OUT and the partials against the single-program merge's", .{ R, H, db }), differs(want, got), true);
            if (R == 6 and db == 128 and !control_done) {
                control_done = true;
                // the control: a value of the ring row the first row's window reads last, changed
                const slot: usize = @intCast(wbase[0] + @mod(pos[0], @as(i64, RING)));
                const one = [2]u8{ 0x80, 0x42 }; // 64.0
                const moved = try Got.of(r, a, bufs, sizes, db);
                defer std.heap.page_allocator.free(moved);
                r.report("merge_d control: a value of the first row's current ring slot set to 64", differs(want, moved), false);
            }
        }
    }
    r.note("merge_d: every call found its twin in the extra set (no call sent to the single-program merge)", tri.twin_misses.load(.monotonic) == misses);
}

// -- TF_DS_WOA_MERGE --------------------------------------------------------------------------------------------------------------------

/// TF_DS_WOA_MERGE: mx.woA8Prefill with wo_a's groups in ONE launch of tf_dsn_pf_bf16w8 (grid.z = groups) against one launch a group, u byte for byte at 1 .. 16,384 rows (the woa_w8 section's layer: 2 groups
/// of 1,024 outputs x 4,096 K), again with the 16-warp twin where the image has it; the control: one o value changed, the merged launch's u against the group launches' of the original.
fn woaMerge(r: *Run, rng: std.Random) !void {
    const d = r.d;
    const have = r.k.pf_bf16w8 != null;
    r.note("woa_merge: tf_dsn_pf_bf16w8 is in this image", have);
    if (!have) return;
    const keep_merge = mx.woa_merge;
    defer mx.woa_merge = keep_merge;
    const keep_w16 = r.k.gemm_w16;
    defer r.k.gemm_w16 = keep_w16;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const n_all = W8_GROUPS * W8_GN;
    const kk = W8_GROUPS * W8_GK;
    const max_m = w8_rows[w8_rows.len - 1];
    const codes = try ha.alloc(u8, n_all * W8_GK);
    for (codes) |*c| {
        c.* = rng.int(u8);
        if (c.* & 0x7f == 0x7f) c.* -= 1;
    }
    const scales = try ha.alloc(u8, (n_all / 32) * (W8_GK / 32));
    for (scales) |*e| e.* = 110 + rng.uintLessThan(u8, 18);
    const lay = try w8Layer(d, W8_GROUPS, W8_GN, W8_GK, codes, scales, ha, &keep);
    const o = try upload(d, try bf16s(max_m * kk, rng, 0.5, ha));
    try keep.append(ha, o);
    var sb: [4]cuda.DeviceBuffer = undefined;
    for (&sb, mx.Scratch.sizes(max_m, kk, 1)) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b.*);
    }
    const sc: mx.Scratch = .{ .q = sb[0].ptr, .s = sb[1].ptr, .z = sb[2].ptr, .fake = sb[3].ptr, .rows = max_m, .max_k = kk };
    const u = try cuda.DeviceBuffer.alloc(d, max_m * n_all * 2);
    try keep.append(ha, u);
    var diff: usize = 0;
    for (w8_rows) |m| {
        mx.woa_merge = false;
        try poison(u.ptr, m * n_all * 2, d, r.s);
        try mx.woA8Prefill(r.k, r.s, sc, lay.w8, W8_GROUPS, o.ptr, kk, m, u.ptr, n_all);
        try r.s.synchronize();
        const want = try get(u.ptr, m * n_all * 2, d);
        defer free(want);
        mx.woa_merge = true;
        try poison(u.ptr, m * n_all * 2, d, r.s);
        try mx.woA8Prefill(r.k, r.s, sc, lay.w8, W8_GROUPS, o.ptr, kk, m, u.ptr, n_all);
        try r.s.synchronize();
        const got = try get(u.ptr, m * n_all * 2, d);
        defer free(got);
        diff += differs(want, got);
    }
    r.report("woa_merge: u from one launch over the groups against one launch a group at 1 .. 16384 rows, 2 groups of 1024 x 4096", diff, true);
    {
        const m: usize = 256;
        mx.woa_merge = false;
        try poison(u.ptr, m * n_all * 2, d, r.s);
        try mx.woA8Prefill(r.k, r.s, sc, lay.w8, W8_GROUPS, o.ptr, kk, m, u.ptr, n_all);
        try r.s.synchronize();
        const want = try get(u.ptr, m * n_all * 2, d);
        defer free(want);
        const at = o.ptr + (100 * kk + 4100) * 2;
        var was: [2]u8 = undefined;
        try take(d, &was, at, 2);
        const big = [2]u8{ 0x80, 0x42 }; // 64.0
        try put(d, at, &big, 2);
        mx.woa_merge = true;
        try mx.woA8Prefill(r.k, r.s, sc, lay.w8, W8_GROUPS, o.ptr, kk, m, u.ptr, n_all);
        try r.s.synchronize();
        try put(d, at, &was, 2);
        const got = try get(u.ptr, m * n_all * 2, d);
        defer free(got);
        r.report("woa_merge control: one o value of row 100 set to 64, the merged launch against the group launches' of the original", differs(want, got), false);
    }
    var diff16: usize = 0;
    if (r.k.pf_bf16w8_w16 != null) {
        r.k.gemm_w16 = true;
        for (w8_rows) |m| {
            mx.woa_merge = false;
            try poison(u.ptr, m * n_all * 2, d, r.s);
            try mx.woA8Prefill(r.k, r.s, sc, lay.w8, W8_GROUPS, o.ptr, kk, m, u.ptr, n_all);
            try r.s.synchronize();
            const want = try get(u.ptr, m * n_all * 2, d);
            defer free(want);
            mx.woa_merge = true;
            try poison(u.ptr, m * n_all * 2, d, r.s);
            try mx.woA8Prefill(r.k, r.s, sc, lay.w8, W8_GROUPS, o.ptr, kk, m, u.ptr, n_all);
            try r.s.synchronize();
            const got = try get(u.ptr, m * n_all * 2, d);
            defer free(got);
            diff16 += differs(want, got);
        }
    }
    r.report("woa_merge with the 16-warp twin (0 when the image has none): merged against one launch a group", diff16, true);
}

// -- TF_DS_AQ_VEC -----------------------------------------------------------------------------------------------------------------------

/// One bf16 value of act_quant's pattern `pat`: the typical range, zeros, -0, NaN, infinities, subnormals, huge and tiny exponents (the scale's edges), a lone NaN
/// or a lone value among zeros in a group, near 1, any 16 bits.
fn aqValue(pat: u32, rng: std.Random) u16 {
    const r = rng.int(u32);
    const sign: u16 = @truncate(r & 0x8000);
    const man: u16 = @truncate(r & 0x7f);
    const E = struct {
        fn of(x: u32, lo: u32, span: u32) u16 {
            return @intCast(lo + (x >> 4) % span);
        }
    };
    return switch (pat) {
        1 => sign | (E.of(r, 118, 10) << 7) | man,
        2 => 0,
        3 => 0x8000,
        4 => 0x7fc0 | (man & 0x3f),
        5 => 0x7f80,
        6 => if ((r & 3) == 0) 0xff80 else sign | (E.of(r, 120, 6) << 7) | man,
        7 => sign | man,
        8 => sign | (E.of(r, 250, 5) << 7) | man,
        9 => sign | (E.of(r, 1, 20) << 7) | man,
        11 => if ((r % 32) == 0) 0x7fc0 else sign | (E.of(r, 124, 6) << 7) | man,
        12 => if ((r % 32) < 31) 0 else sign | (127 << 7) | man,
        13 => sign | (E.of(r, 100, 40) << 7) | man,
        14 => sign | (E.of(r, 126, 3) << 7) | man,
        else => @truncate(r),
    };
}

/// x filled by 32-value groups: mode 0 every group its own pattern of the 16 (aqValue), 1 the typical range, 2 the wide range (amax 2^-27 .. 2^13), 3 any 16 bits.
fn aqFill(x: []u16, mode: usize, rng: std.Random) void {
    var pat: u32 = 10;
    for (x, 0..) |*v, i| {
        if (i % 32 == 0) pat = switch (mode) {
            0 => rng.uintLessThan(u32, 16),
            1 => 1,
            2 => 13,
            else => 10,
        };
        v.* = aqValue(pat, rng);
    }
}

const AqOut = struct { q: []u8, s: []u8, f: []u8 };

fn aqFree(o: AqOut) void {
    free(o.q);
    free(o.s);
    free(o.f);
}

/// act_quant of x's rows (`vec`: the four-lane twin where the call fits, else the baseline kernel): the codes, the scales and the fake rows, read back (aqFree).
fn aqRun(r: *Run, vec: bool, x: u64, ldx: usize, rows: usize, kk: usize) !AqOut {
    const d = r.d;
    mx.aq_vec_on = vec;
    var qb = try cuda.DeviceBuffer.alloc(d, rows * kk);
    defer qb.free();
    var sb = try cuda.DeviceBuffer.alloc(d, rows * (kk / 32) * 4);
    defer sb.free();
    var fb = try cuda.DeviceBuffer.alloc(d, rows * kk * 2);
    defer fb.free();
    try poison(qb.ptr, qb.len, d, r.s);
    try poison(sb.ptr, sb.len, d, r.s);
    try poison(fb.ptr, fb.len, d, r.s);
    try mx.quant(r.k, r.s, x, .bf16, ldx, rows, kk, qb.ptr, kk, sb.ptr, kk / 32);
    try mx.fake(r.k, r.s, x, .bf16, ldx, rows, kk, fb.ptr, kk);
    try r.s.synchronize();
    const q = try get(qb.ptr, qb.len, d);
    errdefer free(q);
    const s = try get(sb.ptr, sb.len, d);
    errdefer free(s);
    const f = try get(fb.ptr, fb.len, d);
    return .{ .q = q, .s = s, .f = f };
}

/// TF_DS_AQ_VEC: tf_dsn_act_quant_vec_kernel (four lanes a 32-value group, 16-byte loads) against tf_dsn_act_quant_kernel (a warp a group) over the prompt pass's shapes
/// (the experts' pack 28,672 x 640, the front 1,024 x 5,120, wo_b's input 1,024 x 8,192, a 4,096-row pass, odd group counts, one group, a tail, a strided row) x 4 fills
/// (aqFill), every output byte compared: the codes, the scales and the fake rows (wo_a's input). Controls: one input value changed (the twin's output must differ from
/// the baseline of the unchanged input), and a row stride off 8 elements (no twin: both settings run the baseline kernel and must agree).
fn actQuant(r: *Run, rng: std.Random) !void {
    const d = r.d;
    const have = r.k.act_quant_vec != null;
    r.note("act_quant: tf_dsn_act_quant_vec_kernel is in this image", have);
    if (!have) return;
    const keep = mx.aq_vec_on;
    defer mx.aq_vec_on = keep;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    const shapes = [_][3]usize{ .{ 28672, 640, 640 }, .{ 1024, 5120, 5120 }, .{ 1024, 8192, 8192 }, .{ 4096, 5120, 5120 }, .{ 5, 96, 96 }, .{ 1, 32, 32 }, .{ 4095, 640, 640 }, .{ 777, 1280, 2048 } };
    var diff_q: usize = 0;
    var diff_s: usize = 0;
    var diff_f: usize = 0;
    var control: usize = 0;
    var fits_all = true;
    for (shapes) |sh| {
        const rows = sh[0];
        const kk = sh[1];
        const ldx = sh[2];
        const x = try ha.alloc(u16, rows * ldx);
        for (0..4) |mode| {
            aqFill(x, mode, rng);
            var xb = try upload(d, std.mem.sliceAsBytes(x));
            defer xb.free();
            fits_all = fits_all and mx.aqVecFits(.bf16, xb.ptr, ldx, kk, 0, kk, 0, 0) and mx.aqVecFits(.bf16, xb.ptr, ldx, kk, 0, 0, 0x100, kk);
            const base = try aqRun(r, false, xb.ptr, ldx, rows, kk);
            defer aqFree(base);
            const twin = try aqRun(r, true, xb.ptr, ldx, rows, kk);
            defer aqFree(twin);
            diff_q += differs(base.q, twin.q);
            diff_s += differs(base.s, twin.s);
            diff_f += differs(base.f, twin.f);
            if (rows == 4095 and mode == 1) {
                x[0] = 0x4280; // 64.0 in the first group of a typical-range input: its scale and codes move
                var yb = try upload(d, std.mem.sliceAsBytes(x));
                defer yb.free();
                const changed = try aqRun(r, true, yb.ptr, ldx, rows, kk);
                defer aqFree(changed);
                control += differs(base.q, changed.q);
            }
        }
    }
    r.report("act_quant: the twin's codes and scales against the baseline kernel's (8 shapes x 4 fills)", diff_q + diff_s, true);
    r.report("act_quant: the twin's fake rows (wo_a's input) against the baseline kernel's", diff_f, true);
    r.note("act_quant: every shape's call fits the twin (so the twin is the kernel the lines above compare)", fits_all);
    r.report("act_quant control: one input value changed, the twin's codes against the baseline kernel's of the unchanged input", control, false);
    // a row stride off 8 elements: the twin's loads need it, so the call runs the baseline kernel under either setting
    const orows: usize = 33;
    const odd = try ha.alloc(u16, orows * 644);
    aqFill(odd, 0, rng);
    var ob = try upload(d, std.mem.sliceAsBytes(odd));
    defer ob.free();
    const a0 = try aqRun(r, false, ob.ptr, 644, orows, 640);
    defer aqFree(a0);
    const a1 = try aqRun(r, true, ob.ptr, 644, orows, 640);
    defer aqFree(a1);
    const strayed = differs(a0.q, a1.q) + differs(a0.s, a1.s) + differs(a0.f, a1.f) + @as(usize, @intFromBool(mx.aqVecFits(.bf16, ob.ptr, 644, 640, 0, 640, 0, 0)));
    r.report("act_quant: a row stride of 644 (not a multiple of 8) runs the baseline kernel under either setting", strayed, true);
}

// -- TF_DS_IDX_U / TF_DS_IDX_V ---------------------------------------------------------------------------------------

/// Which of the extra set's index score twins a run asks for (none: the baseline kernel).
const Which = enum { none, u, v, t };

/// tri_index.indexScore of a prompt chunk's row block with and without idx_u: the base set's _index_score_rows against
/// the extra set's twin _index_score_rows_u. See idxTwin.
fn idxU(r: *Run, rng: std.Random) !void {
    try idxTwin(r, rng, .u);
}

/// tri_index.indexScore of a prompt chunk's row block with and without idx_v: the base set's _index_score_rows against
/// the extra set's twin _index_score_rows_v (one row's dots at a time, its head sum written out, four warps). See
/// idxTwin.
fn idxV(r: *Run, rng: std.Random) !void {
    try idxTwin(r, rng, .v);
}

/// idxTwin with TF_DS_IDX_TPP=4: the extra set's _index_score_rows_t (4 key tiles a program, grid y = cdiv(nt, 4), the next tile prefetched into
/// L1) against the base set's _index_score_rows: the same cases, the control and the line count.
fn idxT(r: *Run, rng: std.Random) !void {
    try idxTwin(r, rng, .t);
}

/// The index score twins' shared body: a prompt chunk's row block with and without the twin, at RB 2 and RB 1 (TREE),
/// keys with tile maxima and plain scores, 64, 512 and 37 rows over 4,100, 33,000 and 131,072 packed keys (key 0 .. 3
/// every code byte, keys 4 .. 257 every finite E8M0 scale byte, the rest random scales 2^-17 .. 2^0), each row its own
/// vis (all, none, random): byte-equal; one visible key's code byte changed differs; every call found its twin.
fn idxTwin(r: *Run, rng: std.Random, which: Which) !void {
    const d = r.d;
    const IH: usize = 32;
    const ID: usize = 128;
    const label: []const u8 = switch (which) {
        .u => "idx_u",
        .v => "idx_v",
        .t => "idx_t",
        .none => "idx",
    };
    var line: [256]u8 = undefined;
    if (r.t.extra == null) return r.note(try std.fmt.bufPrint(&line, "{s}: the kit's extra set (aot-extra/) loads", .{label}), false);
    const keep_u = dsv41.tri_index.idx_u;
    defer dsv41.tri_index.idx_u = keep_u;
    const keep_v = dsv41.tri_index.idx_v;
    defer dsv41.tri_index.idx_v = keep_v;
    const keep_t = dsv41.tri_index.idx_tpp;
    defer dsv41.tri_index.idx_tpp = keep_t;
    const keep_rb = dsv41.tri_index.idx_rb;
    defer dsv41.tri_index.idx_rb = keep_rb;
    const misses = tri.twin_misses.load(.monotonic);
    const IdxShape = struct { rows: usize, n: usize };
    const shapes = [_]IdxShape{ .{ .rows = 64, .n = 4100 }, .{ .rows = 512, .n = 33000 }, .{ .rows = 37, .n = 131072 } };
    const In = struct { q: cuda.DeviceBuffer, w: cuda.DeviceBuffer, codes: cuda.DeviceBuffer, scales: cuda.DeviceBuffer, vis: cuda.DeviceBuffer, out: cuda.DeviceBuffer, tmax: cuda.DeviceBuffer };
    const Idx = struct {
        fn make(rr: *Run, sh: IdxShape, rg: std.Random, aa: std.mem.Allocator, kp: *std.ArrayList(cuda.DeviceBuffer)) !In {
            const codes = try aa.alloc(u8, sh.n * ID / 2);
            rg.bytes(codes);
            for (codes[0..256], 0..) |*c, i| c.* = @intCast(i);
            const scales = try aa.alloc(u8, sh.n * ID / 32);
            for (scales) |*x| x.* = 110 + rg.uintLessThan(u8, 18);
            for (1..255) |e| @memset(scales[(4 + e - 1) * (ID / 32) ..][0 .. ID / 32], @intCast(e));
            const vis = try aa.alloc(i64, sh.rows);
            for (vis) |*v| v.* = @intCast(rg.uintAtMost(usize, sh.n));
            vis[0] = @intCast(sh.n);
            vis[1] = 0;
            const in: In = .{
                .q = try upload(rr.d, try bf16s(sh.rows * IH * ID, rg, 0.5, aa)),
                .w = try upload(rr.d, try bf16s(sh.rows * IH, rg, 0.25, aa)),
                .codes = try upload(rr.d, codes),
                .scales = try upload(rr.d, scales),
                .vis = try upload(rr.d, std.mem.sliceAsBytes(vis)),
                .out = try cuda.DeviceBuffer.alloc(rr.d, sh.rows * sh.n * 8),
                .tmax = try cuda.DeviceBuffer.alloc(rr.d, sh.rows * ((sh.n + 63) / 64) * 8),
            };
            for ([_]cuda.DeviceBuffer{ in.q, in.w, in.codes, in.scales, in.vis, in.out, in.tmax }) |b| try kp.append(aa, b);
            return in;
        }
        /// the row block's keys (with their tile maxima) or scores, poisoned first: the bytes, the maxima after the keys
        fn run(rr: *Run, in: In, sh: IdxShape, keyed: bool, w: Which) ![]u8 {
            dsv41.tri_index.idx_u = w == .u;
            dsv41.tri_index.idx_v = w == .v;
            dsv41.tri_index.idx_tpp = if (w == .t) 4 else 0;
            const ob = sh.rows * sh.n * @as(usize, if (keyed) 8 else 4);
            const tb = sh.rows * ((sh.n + 63) / 64) * 8;
            try poison(in.out.ptr, ob, rr.d, rr.s);
            try poison(in.tmax.ptr, tb, rr.d, rr.s);
            try dsv41.tri_index.indexScore(rr.t, in.q.ptr, .{ .fp4 = .{ .codes = in.codes.ptr, .scales = in.scales.ptr } }, in.w.ptr, in.vis.ptr, sh.n, in.out.ptr, null, keyed, if (keyed) in.tmax.ptr else null, false, sh.rows, IH, ID);
            try rr.s.synchronize();
            const all = try std.heap.page_allocator.alloc(u8, ob + if (keyed) tb else 0);
            errdefer std.heap.page_allocator.free(all);
            return all;
        }
    };
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var ins: [shapes.len]In = undefined;
    for (&ins, shapes) |*in, sh| in.* = try Idx.make(r, sh, rng, ha, &keep);
    for ([_]usize{ 2, 1 }) |rb| for ([_]bool{ true, false }) |keyed| {
        dsv41.tri_index.idx_rb = rb;
        var diff: usize = 0;
        for (ins, shapes) |in, sh| {
            const want = try Idx.run(r, in, sh, keyed, .none);
            defer free(want);
            const got = try Idx.run(r, in, sh, keyed, which);
            defer free(got);
            diff += differs(want, got);
        }
        r.report(try std.fmt.bufPrint(&line, "{s}: RB {d}{s}, {s} at 64 x 4100, 512 x 33000 and 37 x 131072 (rows x keys): the twin's against _index_score_rows'", .{ label, rb, if (rb == 1) " (TREE)" else "", if (keyed) "keys and tile maxima" else "scores" }), diff, true);
    };
    // the control: one visible key's code byte changed (key 300 of the first block), the twin's keys against the baseline
    {
        dsv41.tri_index.idx_rb = 2;
        const want = try Idx.run(r, ins[0], shapes[0], true, .none);
        defer free(want);
        const at = ins[0].codes.ptr + 300 * (ID / 2) + 7;
        var was: [1]u8 = undefined;
        const changed = [1]u8{was[0] ^ 0x11};
        const got = try Idx.run(r, ins[0], shapes[0], true, which);
        defer free(got);
        r.report(try std.fmt.bufPrint(&line, "{s} control: RB 2 keys with one visible key's code byte changed", .{label}), differs(want, got), false);
    }
    // (a call the extra set has no twin for would have run the baseline kernel: the comparisons above would hold vacuously)
    r.note(try std.fmt.bufPrint(&line, "{s}: every call found its twin in the extra set (no call sent to _index_score_rows)", .{label}), tri.twin_misses.load(.monotonic) == misses);
}

/// TF_DS_DRAFT_MAIN_4WAY: the drafter's main_proj (FP8, 5,120 x 15,360: the taps' three hidden rows) on each quarter of
/// its output columns (the rows of its whole weight and of its 32-row scale blocks a node loads: mx.linearsDecode, the
/// absorb's call) against the whole GEMV: each quarter's rows are the whole output's columns of that quarter, byte for
/// byte, at an absorb's 1 .. 128 rows; one taps value changed differs.
fn draftMain4(r: *Run, rng: std.Random) !void {
    const d = r.d;
    const N: usize = 5120;
    const K: usize = 3 * 5120;
    const Q = N / 4;
    const CAP: usize = 128;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const full = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    const x = try upload(d, try bf16s(CAP * K, rng, 0.5, ha));
    try keep.append(ha, x);
    var sb: [4]cuda.DeviceBuffer = undefined;
    for (&sb, mx.Scratch.sizes(CAP, K, 1)) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b.*);
    }
    const sc: mx.Scratch = .{ .q = sb[0].ptr, .s = sb[1].ptr, .z = sb[2].ptr, .fake = sb[3].ptr, .rows = CAP, .max_k = K };
    const z: mx.Z = .{ .ptr = sb[2].ptr, .floats = 1 };
    const yw = try cuda.DeviceBuffer.alloc(d, CAP * N * 2);
    try keep.append(ha, yw);
    const yq = try cuda.DeviceBuffer.alloc(d, CAP * Q * 2);
    try keep.append(ha, yq);
    const Gemv = struct {
        fn run(rr: *Run, scc: mx.Scratch, zz: mx.Z, l: weights.Linear, xp: u64, m: usize, out: u64, ldo: usize) ![]u8 {
            try poison(out, m * ldo * 2, rr.d, rr.s);
            try mx.linearsDecode(rr.k, rr.s, scc, &.{l}, &.{xp}, &.{3 * 5120}, m, &.{out}, &.{ldo}, &.{.bf16}, zz);
            try rr.s.synchronize();
            return get(out, m * ldo * 2, rr.d);
        }
    };
    // the quarter: the whole weight's rows q .. q + Q and their scale blocks' rows (32 rows a block)
    const Part = struct {
        fn of(l: weights.Linear, q: usize) weights.Linear {
            var p = l;
            p.w = l.w + q * Q * K;
            p.s = l.s + (q * Q / 32) * (K / 32);
            p.n = @intCast(Q);
            return p;
        }
    };
    var line: [256]u8 = undefined;
    for ([_]usize{ 1, 5, 16, 17, 48, 80, 128 }) |m| {
        const want = try Gemv.run(r, sc, z, full, x.ptr, m, yw.ptr, N);
        defer free(want);
        var bad: usize = 0;
        for (0..4) |q| {
            const got = try Gemv.run(r, sc, z, Part.of(full, q), x.ptr, m, yq.ptr, Q);
            defer free(got);
            for (0..m) |i| bad += differs(want[(i * N + q * Q) * 2 ..][0 .. Q * 2], got[i * Q * 2 ..][0 .. Q * 2]);
        }
        r.report(try std.fmt.bufPrint(&line, "draft_main4: {d} rows: main_proj's four quarters (1,280 columns each, the whole weight's rows) against the whole GEMV's columns", .{m}), bad, true);
    }
    // the control: one taps value of row 0 set to 64, quarter 0 against the unchanged whole output
    {
        const m: usize = 16;
        const want = try Gemv.run(r, sc, z, full, x.ptr, m, yw.ptr, N);
        defer free(want);
        var was: [2]u8 = undefined;
        const big = [2]u8{ 0x80, 0x42 }; // 64.0
        const got = try Gemv.run(r, sc, z, Part.of(full, 0), x.ptr, m, yq.ptr, Q);
        defer free(got);
        var n: usize = 0;
        for (0..m) |i| n += differs(want[i * N * 2 ..][0 .. Q * 2], got[i * Q * 2 ..][0 .. Q * 2]);
        r.report("draft_main4 control: one taps value of row 0 set to 64, quarter 0 against the unchanged whole output", n, false);
    }
}

// -- TF_DS_MMA_SF ----------------------------------------------------------------------------------------------------

/// The dense FP8 GEMMs of a prompt chunk on a 2D node (split_shapes) and Engram's wkv (no K split).
const sf_shapes = split_shapes ++ [_]Shape{.{ .name = "engram_wkv", .k = 3072, .n = 12800 }};
/// Row counts: one split a CTA below 64 rows, the 64-row tiles to 1,024 rows, the 128-row ones past them.
/// The bound on a block-scaled output's distance to fp64 beyond the baseline own, of the sum of its products' magnitudes:
/// each MMA of a chain lands within one ulp of its exact sum (the tensor core aligns the products and the running sum
/// and drops the bits past its width), at most 2^-23 of that magnitude, and each split's partial meets the others in an
/// fp32 add (half an ulp): (the K blocks of a split + the splits) x 2^-23.
fn sfBound(kblocks: usize, nsplit: usize) f64 {
    return @as(f64, @floatFromInt(kblocks + nsplit)) / 8388608.0;
}
/// The gate / up pack's bound: its normwise distance to fp64 at most 2^-6 past the baseline (both are bf16 roundings).
const sf_pack_bound: f64 = 1.0 + 1.0 / 64.0;

/// E4M3 codes' values (no NaN code reaches a GEMM here).
const e4m3_lut: [256]f64 = blk: {
    @setEvalBranchQuota(10000);
    var t: [256]f64 = undefined;
    for (&t, 0..) |*v, q| {
        const e: i32 = @intCast((q >> 3) & 15);
        const m: f64 = @floatFromInt(q & 7);
        const a = if (e == 0) std.math.ldexp(m / 8.0, -6) else std.math.ldexp(1.0 + m / 8.0, e - 7);
        v.* = if (q & 0x80 != 0) -a else a;
    }
    break :blk t;
};

fn e2m1Val(c: u8) f64 {
    const t = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return if (c & 8 != 0) -t[c & 7] else t[c & 7];
}

fn ue8m0(b: u8) f64 {
    return std.math.ldexp(@as(f64, 1), @as(i32, b) - 127);
}

/// The fp32 value i of little-endian bytes.
fn f32At(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
}

/// The bf16 value i of little-endian bytes.
fn bf16At(bytes: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
}

/// Two outputs' distance: the largest |a - b|, the largest |a - b| / |b| (|b| at least 2^-10 of the outputs' rms), the
/// count that differ in any bit, of n.
const Dist = struct {
    max_abs: f64 = 0,
    max_rel: f64 = 0,
    differ: usize = 0,
    n: usize = 0,

    fn of(got: []const u8, want: []const u8, bf16: bool) Dist {
        const esz: usize = if (bf16) 2 else 4;
        const n = want.len / esz;
        var ss: f64 = 0;
        for (0..n) |i| {
            const w: f64 = if (bf16) bf16At(want, i) else f32At(want, i);
            ss += w * w;
        }
        const floor = @sqrt(ss / @as(f64, @floatFromInt(@max(n, 1)))) / 1024.0;
        var x: Dist = .{ .n = n };
        for (0..n) |i| {
            const g: f64 = if (bf16) bf16At(got, i) else f32At(got, i);
            const w: f64 = if (bf16) bf16At(want, i) else f32At(want, i);
            if (!std.mem.eql(u8, got[i * esz ..][0..esz], want[i * esz ..][0..esz])) x.differ += 1;
            const d = @abs(g - w);
            x.max_abs = @max(x.max_abs, d);
            if (@abs(w) >= floor) x.max_rel = @max(x.max_rel, d / @abs(w));
        }
        return x;
    }

    fn share(x: Dist) f64 {
        return 100.0 * @as(f64, @floatFromInt(x.differ)) / @as(f64, @floatFromInt(@max(x.n, 1)));
    }
};

/// Outputs against fp64 references: each |err| / sum|products| at its largest for the baseline and the block-scaled
/// outputs, the normwise errors, and whether every block-scaled one is within the baseline + bound x sum|products|.
const Fp64 = struct {
    today_max: f64 = 0,
    sf_max: f64 = 0,
    today_ss: f64 = 0,
    sf_ss: f64 = 0,
    ref_ss: f64 = 0,
    bound: f64,
    within: bool = true,

    fn add(x: *Fp64, today: f64, sf: f64, ref: f64, mag: f64) void {
        const et = @abs(today - ref);
        const es = @abs(sf - ref);
        if (mag > 0) {
            x.today_max = @max(x.today_max, et / mag);
            x.sf_max = @max(x.sf_max, es / mag);
        }
        x.today_ss += et * et;
        x.sf_ss += es * es;
        x.ref_ss += ref * ref;
        if (!(es <= et + x.bound * mag)) x.within = false;
    }

    fn normToday(x: Fp64) f64 {
        return @sqrt(x.today_ss / @max(x.ref_ss, 1e-300));
    }

    fn normSf(x: Fp64) f64 {
        return @sqrt(x.sf_ss / @max(x.ref_ss, 1e-300));
    }
};

/// mx.gemm of rows [0, m) with TF_DS_MMA_SF `sw` in effect (off: the baseline kernels, the fold kernel or the slab
/// kernels), fp32 outputs into y (poisoned first); their bytes (get's).
fn gemmSf(r: *Run, sw: mx.MmaSf, l: weights.Linear, q: u64, s: u64, z: u64, m: usize, y: u64) ![]u8 {
    r.k.sf = sw;
    defer r.k.sf = .off;
    return gemmOnce(r, .off, false, l, q, s, z, 0, m, y, .f32, null);
}

/// Whether mx.gemm takes the block-scaled kernel for a dense GEMM of m rows on these operands with `sw` in effect
/// (mx.sfRoute, as gemmOnce's launch passes them).
fn sfTaken(r: *Run, sw: mx.MmaSf, l: weights.Linear, q: u64, s: u64, y: u64, m: usize) bool {
    r.k.sf = sw;
    defer r.k.sf = .off;
    return mx.sfRoute(r.k, l, q, l.k, s, l.k / 32, y, l.n, false, m) != null;
}

/// fp64 of a dense FP8 GEMM's output (row m, column n) from the codes and scales the GEMM read (q [rows, k] and s
/// [rows, k / 32] fp32, w [n, k] and ws [n / 32, k / 32] UE8M0), and the sum of its products' magnitudes.
fn refDense(qh: []const u8, sh: []const u8, wh: []const u8, wsh: []const u8, k: usize, m: usize, n: usize) [2]f64 {
    const nkb = k / 32;
    var sum: f64 = 0;
    var mag: f64 = 0;
    for (0..nkb) |kb| {
        const sc = @as(f64, f32At(sh, m * nkb + kb)) * ue8m0(wsh[(n / 32) * nkb + kb]);
        var dot: f64 = 0;
        var dm: f64 = 0;
        const qa = qh[m * k + kb * 32 ..][0..32];
        const wa = wh[n * k + kb * 32 ..][0..32];
        for (qa, wa) |a, b| {
            const p = e4m3_lut[a] * e4m3_lut[b];
            dot += p;
            dm += @abs(p);
        }
        sum += dot * sc;
        mag += dm * sc;
    }
    return .{ sum, mag };
}

/// An expert's weight rows as the replay's experts hold them: MXFP4 (fp4: [n, k / 2], low nibble first, scales [n,
/// k / 32]) or FP8 ([n, k], scales [n / 32, k / 32]); column c of K block kb's value i.
const Rows = struct {
    w: []const u8,
    s: []const u8,
    k: usize,
    fp4: bool,

    fn val(x: Rows, c: usize, kk: usize) f64 {
        const nkb = x.k / 32;
        if (x.fp4) {
            const byte = x.w[c * (x.k / 2) + kk / 2];
            return e2m1Val(if (kk & 1 == 0) byte & 15 else byte >> 4) * ue8m0(x.s[c * nkb + kk / 32]);
        }
        return e4m3_lut[x.w[c * x.k + kk]] * ue8m0(x.s[(c / 32) * nkb + kk / 32]);
    }
};

/// fp64 of activation row a (codes [k] and their fp32 scales) through column c of `rows`, and the sum of the products'
/// magnitudes.
fn refRow(codes: []const u8, scales: []const u8, rows: Rows, c: usize) [2]f64 {
    var sum: f64 = 0;
    var mag: f64 = 0;
    for (0..rows.k) |kk| {
        const p = e4m3_lut[codes[kk]] * @as(f64, f32At(scales, kk / 32)) * rows.val(c, kk);
        sum += p;
        mag += @abs(p);
    }
    return .{ sum, mag };
}

/// An expert's weights by id as the device's pointer tables give them (table: u64 [E] on the device), each set read
/// back once.
fn expertRows(r: *Run, table: u64, e: usize, bytes: usize, ha: std.mem.Allocator) ![]u8 {
    var p: [8]u8 = undefined;
    const x = try ha.alloc(u8, bytes);
    return x;
}

fn mmaSf(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    // its own seed: TF_DS_REPLAY_ONLY=mma_sf runs the inputs the whole replay runs
    var prng = std.Random.DefaultPrng.init(0x5f30b);
    const rng = prng.random();
    const d = r.d;
    var line: [512]u8 = undefined;
    r.note("mma_sf: the block-scaled kernels (native_sf.cu) loaded on this GPU", r.k.sfk != null);
    if (r.k.sfk == null) return;
    defer r.k.sf = .off;
    for (sf_shapes) |sh| {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
        const x = try upload(d, try bf16s(sf_max_rows * sh.k, rng, 0.5, ha));
        try keep.append(ha, x);
        const q = try cuda.DeviceBuffer.alloc(d, sf_max_rows * sh.k);
        try keep.append(ha, q);
        const s = try cuda.DeviceBuffer.alloc(d, sf_max_rows * (sh.k / 32) * 4);
        try keep.append(ha, s);
        try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, sf_max_rows, sh.k, q.ptr, sh.k, s.ptr, sh.k / 32);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, sf_max_rows), 1) * 4);
        try keep.append(ha, z);
        const y = try cuda.DeviceBuffer.alloc(d, sf_max_rows * sh.n * 4);
        try keep.append(ha, y);
        // the exact arithmetic (the GEMM's key proven, as its guard would) against the baseline kernels at every row count,
        // its route asserted: the block-scaled kernel from 64 rows where the shape has a K split, the baseline else; the
        try mx.sfProve(r.k, l.s);
        defer {
            r.k.sf_nproven = 0;
            r.k.sf_nkept = 0;
        }
        const split = mx.sfExactShape(sh.k, sh.n);
        var route_bad: usize = 0;
        const today = try gemmSf(r, .off, l, q.ptr, s.ptr, z.ptr, sf_max_rows, y.ptr);
        defer free(today);
        const full = try gemmSf(r, .chained, l, q.ptr, s.ptr, z.ptr, sf_max_rows, y.ptr);
        defer free(full);
        var exact_diff: usize = 0;
        var rows_diff: usize = 0;
        {
            const again = try gemmSf(r, .chained, l, q.ptr, s.ptr, z.ptr, sf_max_rows, y.ptr);
            defer free(again);
            rows_diff += differs(full, again);
        }
        for (sf_rows) |m| {
            const t = try gemmSf(r, .off, l, q.ptr, s.ptr, z.ptr, m, y.ptr);
            defer free(t);
            route_bad += @intFromBool(sfTaken(r, .exact, l, q.ptr, s.ptr, y.ptr, m) != (split and m >= mx.pf_min_rows));
            const e = try gemmSf(r, .exact, l, q.ptr, s.ptr, z.ptr, m, y.ptr);
            defer free(e);
            exact_diff += differs(t, e);
            if (m < sf_max_rows) {
                const c = try gemmSf(r, .chained, l, q.ptr, s.ptr, z.ptr, m, y.ptr);
                defer free(c);
                rows_diff += differs(c, full[0 .. m * sh.n * 4]);
            }
        }
        if (split) {
            r.report(try std.fmt.bufPrint(&line, "mma_sf exact {s} ({d} x {d}): its key proven, fp32 outputs at 1 .. 8,192 rows against the baseline kernels, the block-scaled kernel's route asserted from 64 rows (the slab kernels below; {d} routes wrong)", .{ sh.name, sh.k, sh.n, route_bad }), exact_diff + route_bad, true);
        } else r.report(try std.fmt.bufPrint(&line, "mma_sf exact {s} ({d} x {d}): no K split, so with its key proven the exact arithmetic keeps the baseline kernel at 1 .. 8,192 rows (its route asserted not taken; {d} wrong), against the baseline's", .{ sh.name, sh.k, sh.n, route_bad }), exact_diff + route_bad, true);
        // against the baseline, and both against fp64 on 16 rows spread over the launch
        const dist = Dist.of(full, today, false);
        const qh = try get(q.ptr, sf_max_rows * sh.k, d);
        defer free(qh);
        const shost = try get(s.ptr, sf_max_rows * (sh.k / 32) * 4, d);
        defer free(shost);
        const wh = try get(l.w, sh.n * sh.k, d);
        defer free(wh);
        const wsh = try get(l.s, (sh.n / 32) * (sh.k / 32), d);
        defer free(wsh);
        const sp = mx.splits(sh.k, sh.n);
        var f: Fp64 = .{ .bound = sfBound((sh.k / 32 + sp - 1) / sp, sp) };
        for (0..16) |i| {
            const m = i * (sf_max_rows - 1) / 15;
            for (0..sh.n) |n| {
                const ref = refDense(qh, shost, wh, wsh, sh.k, m, n);
                f.add(f32At(today, m * sh.n + n), f32At(full, m * sh.n + n), ref[0], ref[1]);
            }
        }
        r.note(try std.fmt.bufPrint(&line, "mma_sf {s} at 8,192 rows: against the baseline max |d| {e:.3}, max rel {e:.3}, {d:.3}% of outputs differ; against fp64 on 16 rows, |err| / sum|p| the baseline {e:.3} (normwise {e:.3}), block-scaled {e:.3} (normwise {e:.3}); every output within the baseline + {e:.3} sum|p| ({d} K blocks a split, {d} splits)", .{ sh.name, dist.max_abs, dist.max_rel, dist.share(), f.today_max, f.normToday(), f.sf_max, f.normSf(), f.bound, (sh.k / 32 + sp - 1) / sp, sp }), f.within);
        // the control: one code of row 700 changed
        var code: [1]u8 = undefined;
        const at = q.ptr + 700 * sh.k + 5;
        const changed = [1]u8{code[0] ^ 0x08};
        const ctl = try gemmSf(r, .chained, l, q.ptr, s.ptr, z.ptr, 1024, y.ptr);
        defer free(ctl);
        r.report(try std.fmt.bufPrint(&line, "mma_sf {s} control: one code of row 700 changed", .{sh.name}), differs(full[0 .. 1024 * sh.n * 4], ctl), false);
    }
    try mmaSfExperts(r, rng, &line);
}

/// TF_DS_MMA_SF's experts (gate / up and down of a 2D node's experts, pfRun128's): the exact arithmetic against
/// the baseline 128-byte-run tiles; the chained one's rows at every row count, against the baseline and against fp64.
fn mmaSfExperts(r: *Run, rng: std.Random, line: []u8) !void {
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const ex = try xExperts(d, rng, ha, &keep);
    const R: usize = sf_max_rows;
    const P = R * X_SL;
    const sc = try xScratch(d, R, true, ha, &keep);
    const x = try upload(d, try bf16s(R * X_D, rng, 0.5, ha));
    const hdn = try upload(d, try bf16s(P * X_I, rng, 0.5, ha));
    const pw = try xPicks(R, rng, ha);
    const pb = try upload(d, std.mem.sliceAsBytes(pw.pick));
    const wb = try upload(d, std.mem.sliceAsBytes(pw.wts));
    const h = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
    const out = try cuda.DeviceBuffer.alloc(d, R * X_N * 4);
    for ([_]cuda.DeviceBuffer{ x, hdn, pb, wb, h, out }) |b| try keep.append(ha, b);
    const Moe = struct {
        /// gate / up of rows m into h, then down of hdn's entries: h, down's entry outputs, the combined rows (the baseline:
        /// the 128-byte-run tiles, as served)
        fn run(rr: *Run, e_: weights.Experts, s_: mx.ExpertScratch, xp: u64, hd: u64, pk: u64, wt: u64, hb: cuda.DeviceBuffer, ob: cuda.DeviceBuffer, m: usize, sw: mx.MmaSf) ![3][]u8 {
            rr.k.run128 = true;
            rr.k.sf = sw;
            defer {
                rr.k.run128 = false;
                rr.k.sf = .off;
            }
            try poison(hb.ptr, hb.len, rr.d, rr.s);
            try poison(s_.out, m * X_SL * X_N * 4, rr.d, rr.s);
            try poison(ob.ptr, ob.len, rr.d, rr.s);
            try mx.gateUp(rr.k, rr.s, rr.d, e_, s_, xp, X_D, pk, wt, m, 7.0, hb.ptr, X_MAIN, false);
            try rr.s.synchronize();
            const hh = try get(hb.ptr, m * X_SL * X_W * 2, rr.d);
            errdefer free(hh);
            try mx.down(rr.k, rr.s, e_, s_, hd, X_I, pk, m, ob.ptr, X_N, false, false);
            try rr.s.synchronize();
            return .{ hh, try get(s_.out, m * X_SL * X_N * 4, rr.d), try get(ob.ptr, m * X_N * 4, rr.d) };
        }
    };
    // route asserted: the block-scaled gate / up and down kernels
    const names = [_][]const u8{ "gate / up's bf16 pack", "down's fp32 entry outputs", "the combined fp32 rows" };
    try mx.sfProve(r.k, ex.native.?.s1);
    var exact_diff: [3]usize = @splat(0);
    {
        defer {
            r.k.sf_nproven = 0;
            r.k.sf_nkept = 0;
        }
        r.k.sf = .exact;
        const taken = mx.sfExperts(r.k, ex.native.?.s1, false) != null and mx.sfExperts(r.k, ex.native.?.s1, true) != null;
        r.k.sf = .off;
        exact_diff[0] += @intFromBool(!taken);
            const want = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, .off);
            defer for (want) |w| free(w);
            const got = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, .exact);
            defer for (got) |g| free(g);
            for (&exact_diff, want, got) |*n, w, g| n.* += differs(w, g);
        }
    }
    for (names, exact_diff) |nm, n| r.report(try std.fmt.bufPrint(line, "mma_sf exact experts: the layer's experts proven, their block-scaled kernels' route asserted: {s} at 37 .. 8,192 rows against the baseline 128-byte-run tiles", .{nm}), n, true);
    // the chained arithmetic: a row's bits at every row count and run to run (the pack's main block of each entry, its
    const today = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, R, .off);
    defer for (today) |w| free(w);
    const full = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, R, .chained);
    defer for (full) |w| free(w);
    var rows_diff: usize = 0;
    {
        const again = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, R, .chained);
        defer for (again) |w| free(w);
        for (full, again) |f, g| rows_diff += differs(f, g);
    }
    const rest = X_REST * 32;
    for ([_]usize{ 37, 64, 512, 2048 }) |m| {
        const got = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, .chained);
        defer for (got) |g| free(g);
        const pm = m * X_SL;
        rows_diff += differs(got[0][0 .. pm * X_MAIN * 2], full[0][0 .. pm * X_MAIN * 2]);
        rows_diff += differs(got[0][pm * X_MAIN * 2 ..][0 .. pm * rest * 2], full[0][P * X_MAIN * 2 ..][0 .. pm * rest * 2]);
        rows_diff += differs(got[1], full[1][0 .. pm * X_N * 4]);
        rows_diff += differs(got[2], full[2][0 .. m * X_N * 4]);
    }
    // down's entry outputs against the baseline and against fp64 (35 entries: rows 0, 1, 1000, 2047, 4095, every slot); the
    // inputs as the launch read them: h's codes and scales, each expert's down rows
    const nx = ex.native.?;
    const hq = try get(sc.hq, P * X_I, d);
    defer free(hq);
    const hs = try get(sc.hs, P * (X_I / 32) * 4, d);
    defer free(hs);
    const sample = [_]usize{ 0, 1, 1000, 2047, 4095 };
    var fd: Fp64 = .{ .bound = sfBound(X_I / 32, 1) };
    for (sample) |row| for (0..X_SL) |slot| {
        const ent = row * X_SL + slot;
        const e: usize = @intCast(pw.pick[ent]);
        const fp4 = e != X_E - 1;
        const rows: Rows = .{
            .w = try expertRows(r, nx.w2, e, if (fp4) X_N * X_I / 2 else X_N * X_I, ha),
            .s = try expertRows(r, nx.s2, e, if (fp4) X_N * (X_I / 32) else (X_N / 32) * (X_I / 32), ha),
            .k = X_I,
            .fp4 = fp4,
        };
        for (0..X_N) |n| {
            const ref = refRow(hq[ent * X_I ..][0..X_I], hs[ent * (X_I / 32) * 4 ..][0 .. (X_I / 32) * 4], rows, n);
            fd.add(f32At(today[1], ent * X_N + n), f32At(full[1], ent * X_N + n), ref[0], ref[1]);
        }
    };
    const dd = Dist.of(full[1], today[1], false);
    const dy = Dist.of(full[2], today[2], false);
    r.note(try std.fmt.bufPrint(line, "mma_sf experts' down at 8,192 rows: entry outputs against the baseline max |d| {e:.3}, max rel {e:.3}, {d:.3}% differ (the combined rows: {e:.3}, {e:.3}, {d:.3}%); against fp64 on 35 entries, |err| / sum|p| the baseline {e:.3} (normwise {e:.3}), block-scaled {e:.3} (normwise {e:.3}); every output within the baseline + {e:.3} sum|p| (36 K blocks, no split)", .{ dd.max_abs, dd.max_rel, dd.share(), dy.max_abs, dy.max_rel, dy.share(), fd.today_max, fd.normToday(), fd.sf_max, fd.normSf(), fd.bound }), fd.within);
    // gate / up's pack against the baseline and against fp64 (its epilogue in fp64 on the exact sums, no bf16 rounding):
    // the main block of the same 35 entries
    const qa = try get(sc.q, R * X_D, d);
    defer free(qa);
    const sa = try get(sc.s, R * (X_D / 32) * 4, d);
    defer free(sa);
    var t_ss: f64 = 0;
    var s_ss: f64 = 0;
    var r_ss: f64 = 0;
    for (sample) |row| for (0..X_SL) |slot| {
        const ent = row * X_SL + slot;
        const e: usize = @intCast(pw.pick[ent]);
        const fp4 = e != X_E - 1;
        const wbytes = if (fp4) X_W * X_D / 2 else X_W * X_D;
        const sbytes = if (fp4) X_W * (X_D / 32) else (X_W / 32) * (X_D / 32);
        const g_rows: Rows = .{ .w = try expertRows(r, nx.w1, e, wbytes, ha), .s = try expertRows(r, nx.s1, e, sbytes, ha), .k = X_D, .fp4 = fp4 };
        const u_rows: Rows = .{ .w = try expertRows(r, nx.w3, e, wbytes, ha), .s = try expertRows(r, nx.s3, e, sbytes, ha), .k = X_D, .fp4 = fp4 };
        const codes = qa[row * X_D ..][0..X_D];
        const scales = sa[row * (X_D / 32) * 4 ..][0 .. (X_D / 32) * 4];
        for (0..X_MAIN) |col| {
            const g = @min(refRow(codes, scales, g_rows, col)[0], 7.0);
            const u = std.math.clamp(refRow(codes, scales, u_rows, col)[0], -7.0, 7.0);
            const want = g / (1.0 + @exp(-g)) * u * pw.wts[ent];
            const et = @as(f64, bf16At(today[0], ent * X_MAIN + col)) - want;
            const es = @as(f64, bf16At(full[0], ent * X_MAIN + col)) - want;
            t_ss += et * et;
            s_ss += es * es;
            r_ss += want * want;
        }
    };
    const dp = Dist.of(full[0][0 .. P * X_MAIN * 2], today[0][0 .. P * X_MAIN * 2], true);
    const nt = @sqrt(t_ss / r_ss);
    const ns = @sqrt(s_ss / r_ss);
    r.note(try std.fmt.bufPrint(line, "mma_sf experts' gate / up at 8,192 rows: the pack's main block against the baseline max |d| {e:.3}, max rel {e:.3}, {d:.4}% of values differ; against fp64 (the epilogue on the exact sums) on 35 entries, normwise the baseline {e:.4}, block-scaled {e:.4}: within 2^-6 of the baseline's", .{ dp.max_abs, dp.max_rel, dp.share(), nt, ns }), ns <= nt * sf_pack_bound);
    // the control: one x value of row 30 set to 64
    const big = [2]u8{ 0x80, 0x42 };
    var was: [2]u8 = undefined;
    const at = x.ptr + (30 * X_D + 300) * 2;
    const base = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, .chained);
    defer for (base) |w| free(w);
    const got = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, .chained);
    defer for (got) |g| free(g);
    r.report("mma_sf experts control: gate / up's pack with one x value of row 30 set to 64", differs(base[0], got[0]), false);
}

// -- TF_DS_MMA_SF_EXACT ----------------------------------------------------------------------------------------------

/// mx.gemm of rows [0, m) with TF_DS_MMA_SF_EXACT's arithmetic in effect or not (`on`), outputs of `mode` into y
/// (poisoned first); their bytes (get's).
fn gemmExact(r: *Run, on: bool, l: weights.Linear, q: u64, s: u64, z: u64, m: usize, y: u64, mode: Mode) ![]u8 {
    r.k.sf = if (on) .exact else .off;
    defer r.k.sf = .off;
    return gemmOnce(r, .off, false, l, q, s, z, 0, m, y, mode, null);
}

/// The smallest and largest byte of host bytes (255 and 0 for none).
fn hostRange(b: []const u8) [2]u8 {
    var x = [2]u8{ 255, 0 };
    for (b) |v| x = .{ @min(x[0], v), @max(x[1], v) };
    return x;
}

/// A dense FP8 GEMM against the baseline with TF_DS_MMA_SF_EXACT: its codes' activations x (bf16 normal x `xs`), every
/// weight scale byte `sbyte` (null: fp8Linear's fast scales), bf16 and fp32 outputs at `rows`; the bytes that differ.
/// The guard's verdict on a layer of this linear: proven (the block-scaled kernel), kept (by its key, in words), or not
/// covered (no K split: neither, the baseline kernel); `keep_scales` false proves its key whatever the verdict (unguarded).
/// The route at each row count is asserted (route_ok): the block-scaled kernel from 64 rows for a proven key with a K
/// split, the baseline kernel else.
fn exactDense(r: *Run, rng: std.Random, sh: Shape, xs: f32, sbyte: ?u8, rows: []const usize, keep_scales: bool) !struct { diff: usize, proven: bool, covered: bool, route_ok: bool } {
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
    const max_m = rows[rows.len - 1];
    const x = try upload(d, try bf16s(max_m * sh.k, rng, xs, ha));
    try keep.append(ha, x);
    const q = try cuda.DeviceBuffer.alloc(d, max_m * sh.k);
    try keep.append(ha, q);
    const s = try cuda.DeviceBuffer.alloc(d, max_m * (sh.k / 32) * 4);
    try keep.append(ha, s);
    try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, max_m, sh.k, q.ptr, sh.k, s.ptr, sh.k / 32);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, max_m), 1) * 4);
    try keep.append(ha, z);
    const y = try cuda.DeviceBuffer.alloc(d, max_m * sh.n * 4);
    try keep.append(ha, y);
    // the guard's verdict (model.sfGuard on a layer of this linear): mx.gemm takes the block-scaled kernel only for a
    // proven key; keep_scales false proves it whatever the verdict (the kernel unguarded)
    var lay = std.mem.zeroes(weights.Layer);
    lay.wq_a = l;
    defer {
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
    }
    var kb: [320]u8 = undefined;
    const kept = try dsv41.model.sfGuard(std.heap.page_allocator, d, r.s, r.k, &.{lay}, &kb);
    const covered = mx.sfExactShape(sh.k, sh.n);
    const proven = mx.sfProvenKey(r.k, l.s);
    const words = if (!covered)
        !proven and !mx.sfKept(r.k, l.s) and std.mem.eql(u8, kept, "none")
    else if (proven)
        r.k.sf_nkept == 0 and std.mem.eql(u8, kept, "none")
    else
        mx.sfKept(r.k, l.s) and r.k.sf_nkept == 1 and std.mem.eql(u8, kept, "wq_a in 1 layer");
    if (!words) return error.GuardWords;
    if (!keep_scales) try mx.sfProve(r.k, l.s);
    const takes = covered and mx.sfProvenKey(r.k, l.s);
    var route_ok = true;
    for (rows) |m| route_ok = route_ok and sfTaken(r, .exact, l, q.ptr, s.ptr, y.ptr, m) == (takes and m >= mx.pf_min_rows);
    var diff: usize = 0;
    for ([_]Mode{ .bf16, .f32 }) |mode| for (rows) |m| {
        const t = try gemmExact(r, false, l, q.ptr, s.ptr, z.ptr, m, y.ptr, mode);
        defer free(t);
        const e = try gemmExact(r, true, l, q.ptr, s.ptr, z.ptr, m, y.ptr, mode);
        defer free(e);
        diff += differs(t, e);
    };
    return .{ .diff = diff, .proven = proven, .covered = covered, .route_ok = route_ok };
}

fn mmaSfExact(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    // its own seed: TF_DS_REPLAY_ONLY=mma_sf_exact runs the inputs the whole replay runs
    var prng = std.Random.DefaultPrng.init(0xe5ac7);
    const rng = prng.random();
    const d = r.d;
    var line: [512]u8 = undefined;
    r.note("mma_sf_exact: the block-scaled kernels (native_sf.cu) loaded on this GPU", r.k.sfk != null);
    if (r.k.sfk == null) return;
    defer {
        r.k.sf = .off;
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
    }
    // production's shapes at a chunk's rows: the baseline bytes, bf16 and fp32 (wo_b and Engram's wkv write fp32); a shape
    // without a K split (Engram's wkv) stays on the baseline kernel, its guard proving it no key
    // (512, 1,000 and 1,024: the 64-row tiles' kernel past one tile, a part tile at the end, its last row count; a
    // node's owned quarter of a 2,048-row chunk is 512 rows of idx_wq_b)
    for (sf_shapes) |sh| {
        const x = try exactDense(r, rng, sh, 0.5, null, &rows, true);
        if (x.covered) {
            r.report(try std.fmt.bufPrint(&line, "mma_sf_exact {s} ({d} x {d}): bf16 and fp32 outputs at 1, 63, 64, 512, 1,000, 1,024, 1,025, 2,048, 4,096 and 8,192 rows against the baseline kernels, the block-scaled kernel's route asserted from 64 rows{s}", .{ sh.name, sh.k, sh.n, if (!x.proven) " (its scales outside the proven range: WRONG input)" else if (!x.route_ok) " (a route WRONG)" else "" }), if (x.proven and x.route_ok) x.diff else 1, true);
        } else {
            r.report(try std.fmt.bufPrint(&line, "mma_sf_exact {s} ({d} x {d}): no K split, so its guard proves it no key and mx.gemm keeps it on the baseline kernel (its route asserted not taken{s}): bf16 and fp32 outputs at 1, 63, 64, 512, 1,000, 1,024, 1,025, 2,048, 4,096 and 8,192 rows against the baseline's", .{ sh.name, sh.k, sh.n, if (x.route_ok) "" else ": WRONG" }), if (x.proven or !x.route_ok) 1 else x.diff, true);
        }
    }
    try mmaSfExactExperts(r, rng, &line);
    // the guard's instrument: tensors packed with gaps of 0x00 and 0xff between them (a range one byte too long shows),
    // each a constant fill with one smallest and one largest byte planted where no other tensor has its own: every
    // word of a 16-byte load (x, y, z, w), the tail past the last whole load, an unaligned start; and an empty range
    {
        const T = struct { align16: bool, len: usize, lo_at: usize, hi_at: usize };
        const ts = [_]T{
            .{ .align16 = true, .len = 1024, .lo_at = 0, .hi_at = 1023 }, // x, w
            .{ .align16 = true, .len = 1024, .lo_at = 4, .hi_at = 520 }, // y, z
            .{ .align16 = true, .len = 1024, .lo_at = 8, .hi_at = 13 }, // z, w
            .{ .align16 = true, .len = 1024, .lo_at = 12, .hi_at = 6 }, // w, y
            .{ .align16 = true, .len = 1000, .lo_at = 995, .hi_at = 2 }, // the tail, x
            .{ .align16 = false, .len = 4111, .lo_at = 4000, .hi_at = 17 }, // an unaligned start
            .{ .align16 = false, .len = 3, .lo_at = 1, .hi_at = 2 },
            .{ .align16 = true, .len = 0, .lo_at = 0, .hi_at = 0 }, // empty: 255 and 0
        };
        var host: std.ArrayList(u8) = .empty;
        defer host.deinit(std.heap.page_allocator);
        var rs: [ts.len]mx.ByteRange = undefined;
        var want: [ts.len][2]u8 = undefined;
        for (ts, 0..) |t, i| {
            // the gap up to the tensor's start (16-aligned, or 3 past it): its last byte 0x00 before even tensors, 0xff
            // before odd ones, the first byte after the tensor the other (a read one byte early or late moves a range)
            const before: u8 = if (i % 2 == 0) 0x00 else 0xff;
            const target: usize = if (t.align16) 0 else 3;
            while ((host.items.len + 1) % 16 != target) try host.append(std.heap.page_allocator, ~before);
            try host.append(std.heap.page_allocator, before);
            const at = host.items.len;
            for (0..t.len) |_| try host.append(std.heap.page_allocator, 80);
            if (t.len > 0) {
                host.items[at + t.lo_at] = 41;
                host.items[at + t.hi_at] = 127;
            }
            want[i] = if (t.len > 0) .{ 41, 127 } else .{ 255, 0 };
            rs[i] = .{ .ptr = at, .len = t.len };
            try host.append(std.heap.page_allocator, ~before);
        }
        const buf = try upload(d, host.items);
        defer {
            var b = buf;
            b.free();
        }
        for (&rs) |*x| x.ptr += buf.ptr;
        var got: [ts.len][2]u8 = undefined;
        try mx.scaleRanges(r.k, d, r.s, std.heap.page_allocator, &rs, &got);
        var bad: usize = 0;
        for (want, got) |w, g| bad += @intFromBool(w[0] != g[0] or w[1] != g[1]);
        r.report("mma_sf_exact guard: tf_dsn_sf_ranges' smallest and largest bytes of packed tensors (each planted where no other has its own: every word of a 16-byte load, the tail, an unaligned start; an empty range) against the host's, the 0x00 / 0xff gaps between them never read", bad, true);
    }
    // the proven range's edges: every FP8 weight scale byte 41 (2^-86) with activations at act_quant's floor (2^-22),
    // and 127 (2^0) with large ones: the guard keeps neither, the baseline bytes
    const edge = [_]usize{ 64, 2048 };
    const lo = try exactDense(r, rng, sf_shapes[0], 1e-5, 41, &edge, true);
    r.report("mma_sf_exact guard's lower edge: wq_a with every weight scale byte 41 (2^-86) and activations at act_quant's floor (2^-22), proven and on the block-scaled kernel, bf16 and fp32 at 64 and 2,048 rows against the baseline's", if (lo.proven and lo.route_ok) lo.diff else 1, true);
    const hi = try exactDense(r, rng, sf_shapes[0], 1e3, 127, &edge, true);
    r.report("mma_sf_exact guard's upper edge: wq_a with every weight scale byte 127 (2^0) and large activations, proven and on the block-scaled kernel, against the baseline's", if (hi.proven and hi.route_ok) hi.diff else 1, true);
    // outside it: 40 and 128 are kept on the baseline kernel (the outputs the baseline); the kernel unguarded on them, as information
    const out40 = try exactDense(r, rng, sf_shapes[0], 1e-5, 40, &edge, true);
    const out128 = try exactDense(r, rng, sf_shapes[0], 1e3, 128, &edge, true);
    const raw40 = try exactDense(r, rng, sf_shapes[0], 1e-5, 40, &edge, false);
    r.report(try std.fmt.bufPrint(&line, "mma_sf_exact guard: wq_a with every weight scale byte 40, then 128, outside the proven range and kept on the baseline kernel, against the baseline (unguarded at 40 the block-scaled kernel's bytes differ in {d})", .{raw40.diff}), if (out40.proven or out128.proven or !out40.route_ok or !out128.route_ok or !raw40.route_ok) 1 else out40.diff + out128.diff, true);
    try exactCoverage(r, rng);
    // the control: the block-scaled kernel against the baseline with one code of row 700 changed for it alone
    {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const sh = sf_shapes[0];
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
        const x = try upload(d, try bf16s(1024 * sh.k, rng, 0.5, ha));
        try keep.append(ha, x);
        const q = try cuda.DeviceBuffer.alloc(d, 1024 * sh.k);
        try keep.append(ha, q);
        const s = try cuda.DeviceBuffer.alloc(d, 1024 * (sh.k / 32) * 4);
        try keep.append(ha, s);
        try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, 1024, sh.k, q.ptr, sh.k, s.ptr, sh.k / 32);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, 1024), 1) * 4);
        try keep.append(ha, z);
        const y = try cuda.DeviceBuffer.alloc(d, 1024 * sh.n * 4);
        try keep.append(ha, y);
        const t = try gemmExact(r, false, l, q.ptr, s.ptr, z.ptr, 1024, y.ptr, .bf16);
        defer free(t);
        // the key proven (as its guard would) and the block-scaled kernel's route asserted, then one code changed
        try mx.sfProve(r.k, l.s);
        defer {
            r.k.sf_nproven = 0;
            r.k.sf_nkept = 0;
        }
        const taken = sfTaken(r, .exact, l, q.ptr, s.ptr, y.ptr, 1024);
        var code: [1]u8 = undefined;
        const at = q.ptr + 700 * sh.k + 5;
        const changed = [1]u8{code[0] ^ 0x08};
        const e = try gemmExact(r, true, l, q.ptr, s.ptr, z.ptr, 1024, y.ptr, .bf16);
        defer free(e);
        r.report("mma_sf_exact control: one code of row 700 changed for the block-scaled kernel (its key proven, its route asserted)", if (taken) differs(t, e) else 0, false);
    }
}

/// A device byte set on the replay's stream (stream-ordered: the guard's kernel runs on that stream).
fn pokeByte(r: *Run, at: u64, v: u8) !void {
    const b = [1]u8{v};
}

/// Whether the guard saw these extremes (mx.Kernels.sf_seen: FP8 dense, MXFP4, the shared expert's; 255, 0: none).
fn seenIs(got: [3][2]u8, want: [3][2]u8) bool {
    for (got, want) |g, w| if (g[0] != w[0] or g[1] != w[1]) return false;
    return true;
}

/// The guard's coverage (model.sfGuard over two layers). Dense: every field of sf_fields on both layers at a shape the
/// exact form takes and, (k, n) swapped, does not (K 1,472, N 1,280), scale bytes 120: every key proven; one byte 40 in
/// layer 1's field, each in turn: that field of layer 1 alone kept, by its key and in words, every other key proven.
/// Experts: two layers of scale tensors alone (two routed experts, the first owning its rest, then the shared one),
/// bytes 120: both layers' keys (s1) proven; one byte 32 in layer 1's first up tensor's rest rows: layer 1's experts
/// alone kept; a second bad tensor there (a down byte 140): still one layer; the shared expert of layer 0 bad too (an
/// FP8 gate byte 128): two layers, both keys kept.
fn exactCoverage(r: *Run, rng: std.Random) !void {
    _ = rng;
    const d = r.d;
    var kb: [320]u8 = undefined;
    defer {
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
    }
    {
        const fields = .{ "wq_a", "wkv", "wq_b", "wo_b", "comp_wkv", "comp_wgate", "idx_wq_b", "idx_wk", "engram_wkv" };
        const ck: u32 = 1472;
        const cn: u32 = 1280;
        const per: usize = (cn / 32) * (ck / 32);
        var lays: [2]weights.Layer = .{ std.mem.zeroes(weights.Layer), std.mem.zeroes(weights.Layer) };
        var sc: [2][fields.len]cuda.DeviceBuffer = undefined;
        for (0..2) |li| inline for (fields, 0..) |f, fi| {
            sc[li][fi] = try cuda.DeviceBuffer.alloc(d, per);
            try d.check(d.api.cuMemsetD8Async(sc[li][fi].ptr, 120, per, r.s.handle), "cuMemsetD8Async");
            @field(lays[li], f) = .{ .words = 0, .k = ck, .n = cn, .fmt = .fp8, .s = sc[li][fi].ptr };
        };
        defer for (&sc) |*row| for (row) |*b| b.free();
        var fails: usize = 0;
        const none = try dsv41.model.sfGuard(std.heap.page_allocator, d, r.s, r.k, &lays, &kb);
        fails += @intFromBool(!(std.mem.eql(u8, none, "none") and r.k.sf_nkept == 0 and r.k.sf_nproven == 2 * fields.len));
        fails += @intFromBool(!seenIs(r.k.sf_seen, .{ .{ 120, 120 }, .{ 255, 0 }, .{ 255, 0 } }));
        inline for (fields, 0..) |f, fi| {
            try pokeByte(r, sc[1][fi].ptr + 7, 40);
            const words = try dsv41.model.sfGuard(std.heap.page_allocator, d, r.s, r.k, &lays, &kb);
            var want: [64]u8 = undefined;
            const w = try std.fmt.bufPrint(&want, "{s} in 1 layer", .{f});
            fails += @intFromBool(!(std.mem.eql(u8, words, w) and r.k.sf_nkept == 1 and mx.sfKept(r.k, sc[1][fi].ptr) and !mx.sfKept(r.k, sc[0][fi].ptr) and
                !mx.sfProvenKey(r.k, sc[1][fi].ptr) and mx.sfProvenKey(r.k, sc[0][fi].ptr) and r.k.sf_nproven == 2 * fields.len - 1));
            // the bytes it saw: FP8 40 .. 120, nothing of the experts' classes
            fails += @intFromBool(!seenIs(r.k.sf_seen, .{ .{ 40, 120 }, .{ 255, 0 }, .{ 255, 0 } }));
            try pokeByte(r, sc[1][fi].ptr + 7, 120);
        }
        r.report("mma_sf_exact guard's coverage, dense: two layers with every field at a shape the exact form takes (K 1,472, N 1,280: swapped, one it does not), all in range: every key proven; one scale byte 40 in layer 1's field, each in turn: that field of layer 1 alone kept, by its key and in words; the bytes it saw FP8 40 .. 120", fails, true);
    }
    {
        // two routed experts (the first owning its rest) and the shared one: rows 64 (+32), dims 256, down 256 x 64
        var ex = std.mem.zeroes(weights.Experts);
        ex.count = 3;
        ex.width = 64;
        ex.dims = 256;
        ex.down_n = 256;
        ex.down_k = 64;
        const owner = [_]i32{ 1, 0, 0 };
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        var lays: [2]weights.Layer = .{ std.mem.zeroes(weights.Layer), std.mem.zeroes(weights.Layer) };
        var tens: [2][3][3]u64 = undefined; // [layer][expert][gate, up, down]
        for (0..2) |li| {
            var x = ex;
            x.native = .{ .w1 = 0, .w3 = 0, .s1 = 0, .s3 = 0, .w2 = 0, .s2 = 0, .tiles = 3, .rest_tiles = 1, .scales_match = true };
            var tabs: [3][3]u64 = undefined; // [s1, s3, s2][expert]
            for (0..3) |e| {
                const lens = weights.expertScaleLens(x, e, owner[e] == 1);
                for (0..3) |j| {
                    const b = try cuda.DeviceBuffer.alloc(d, lens[j]);
                    try keep.append(std.heap.page_allocator, b);
                    try d.check(d.api.cuMemsetD8Async(b.ptr, 120, lens[j], r.s.handle), "cuMemsetD8Async");
                    tabs[j][e] = b.ptr;
                    tens[li][e][j] = b.ptr;
                }
            }
            var tp: [4]u64 = undefined;
            for (0..3) |j| {
                const b = try cuda.DeviceBuffer.alloc(d, 3 * 8);
                try keep.append(std.heap.page_allocator, b);
                tp[j] = b.ptr;
            }
            const ob = try cuda.DeviceBuffer.alloc(d, 3 * 4);
            try keep.append(std.heap.page_allocator, ob);
            tp[3] = ob.ptr;
            try r.s.synchronize();
            x.native.?.s1 = tp[0];
            x.native.?.s3 = tp[1];
            x.native.?.s2 = tp[2];
            x.native.?.owner = tp[3];
            lays[li].experts = x;
        }
        const k0 = lays[0].experts.native.?.s1;
        const k1 = lays[1].experts.native.?.s1;
        var fails: usize = 0;
        const none = try dsv41.model.sfGuard(std.heap.page_allocator, d, r.s, r.k, &lays, &kb);
        fails += @intFromBool(!(std.mem.eql(u8, none, "none") and mx.sfProvenKey(r.k, k0) and mx.sfProvenKey(r.k, k1) and r.k.sf_nkept == 0));
        // layer 1, expert 0's up tensor, in its rest rows (row 64 on: 64 x 8 bytes in)
        try pokeByte(r, tens[1][0][1] + 64 * 8 + 3, 32);
        const one = try dsv41.model.sfGuard(std.heap.page_allocator, d, r.s, r.k, &lays, &kb);
        fails += @intFromBool(!(std.mem.eql(u8, one, "the experts in 1 layer") and r.k.sf_nkept == 1 and mx.sfKept(r.k, k1) and !mx.sfKept(r.k, k0) and mx.sfProvenKey(r.k, k0) and !mx.sfProvenKey(r.k, k1)));
        // a second bad tensor in layer 1 (expert 1's down, 140): still one layer
        try pokeByte(r, tens[1][1][2] + 5, 140);
        const still = try dsv41.model.sfGuard(std.heap.page_allocator, d, r.s, r.k, &lays, &kb);
        fails += @intFromBool(!(std.mem.eql(u8, still, "the experts in 1 layer") and r.k.sf_nkept == 1));
        // layer 0's shared expert's gate (FP8) 128: two layers, both keys
        try pokeByte(r, tens[0][2][0] + 1, 128);
        const two = try dsv41.model.sfGuard(std.heap.page_allocator, d, r.s, r.k, &lays, &kb);
        fails += @intFromBool(!(std.mem.eql(u8, two, "the experts in 2 layers") and r.k.sf_nkept == 2 and mx.sfKept(r.k, k0) and mx.sfKept(r.k, k1) and !mx.sfProvenKey(r.k, k0) and !mx.sfProvenKey(r.k, k1)));
        // the bytes it saw: MXFP4 32 .. 140 (the routed experts' pokes), the shared expert's 120 .. 128, no dense FP8
        fails += @intFromBool(!seenIs(r.k.sf_seen, .{ .{ 255, 0 }, .{ 32, 140 }, .{ 120, 128 } }));
        r.report("mma_sf_exact guard's coverage, experts: two layers, all in range: both keys proven; one byte 32 in layer 1's owned rest rows: layer 1 alone kept by its key; a second bad tensor there: still one layer; layer 0's shared expert at 128 too: two layers, both keys; the bytes it saw MXFP4 32 .. 140, the shared expert's 120 .. 128", fails, true);
    }
    {
        // a layer whose scale tensors the loader did not tie to the guard's lengths (scales_match false): two layers,
        // layer 1 untied, its bytes 99 and its last tensor (the shared expert's down) half the guard's length: none of
        // its bytes read, its experts kept apart (k.sf_apart, not the scale words), layer 0 proven
        var ex = std.mem.zeroes(weights.Experts);
        ex.count = 3;
        ex.width = 64;
        ex.dims = 256;
        ex.down_n = 256;
        ex.down_k = 64;
        const owner = [_]i32{ 1, 0, 0 };
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        var lays: [2]weights.Layer = .{ std.mem.zeroes(weights.Layer), std.mem.zeroes(weights.Layer) };
        for (0..2) |li| {
            var x = ex;
            x.native = .{ .w1 = 0, .w3 = 0, .s1 = 0, .s3 = 0, .w2 = 0, .s2 = 0, .tiles = 3, .rest_tiles = 1, .scales_match = li == 0 };
            var tabs: [3][3]u64 = undefined;
            for (0..3) |e| {
                const lens = weights.expertScaleLens(x, e, owner[e] == 1);
                for (0..3) |j| {
                    const n = if (li == 1 and e == 2 and j == 2) lens[j] / 2 else lens[j];
                    const b = try cuda.DeviceBuffer.alloc(d, n);
                    try keep.append(std.heap.page_allocator, b);
                    try d.check(d.api.cuMemsetD8Async(b.ptr, if (li == 1) 99 else 120, n, r.s.handle), "cuMemsetD8Async");
                    tabs[j][e] = b.ptr;
                }
            }
            var tp: [4]u64 = undefined;
            for (0..3) |j| {
                const b = try cuda.DeviceBuffer.alloc(d, 3 * 8);
                try keep.append(std.heap.page_allocator, b);
                tp[j] = b.ptr;
            }
            const ob = try cuda.DeviceBuffer.alloc(d, 3 * 4);
            try keep.append(std.heap.page_allocator, ob);
            tp[3] = ob.ptr;
            try r.s.synchronize();
            x.native.?.s1 = tp[0];
            x.native.?.s3 = tp[1];
            x.native.?.s2 = tp[2];
            x.native.?.owner = tp[3];
            lays[li].experts = x;
        }
        const k0 = lays[0].experts.native.?.s1;
        const k1 = lays[1].experts.native.?.s1;
        const words = try dsv41.model.sfGuard(std.heap.page_allocator, d, r.s, r.k, &lays, &kb);
        const ok = std.mem.eql(u8, words, "none") and r.k.sf_apart == 1 and r.k.sf_nkept == 1 and mx.sfKept(r.k, k1) and !mx.sfProvenKey(r.k, k1) and mx.sfProvenKey(r.k, k0) and
            seenIs(r.k.sf_seen, .{ .{ 255, 0 }, .{ 120, 120 }, .{ 120, 120 } });
        var lb: [320]u8 = undefined;
        r.note(try std.fmt.bufPrint(&lb, "mma_sf_exact guard's coverage, a layer packed at other lengths: layer 1 untied (its bytes 99, its last tensor half the guard's length), none of its bytes read and its experts kept apart (\"{s}\" for a scale, {d} apart), layer 0 proven", .{ words, r.k.sf_apart }), ok);
    }
}

/// TF_DS_MMA_SF_EXACT's experts (pf_run128's node, the baseline 128-byte-run tiles as served): the baseline bytes at a chunk's row
/// counts; the guard over every expert's three scale tensors (expertScales with the owner table) against the host's;
/// the proven range's edges on weight set 0, and a byte outside it keeping the layer's experts on the baseline kernels.
fn mmaSfExactExperts(r: *Run, rng: std.Random, line: []u8) !void {
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const ex = try xExperts(d, rng, ha, &keep);
    const nx = ex.native.?;
    const P = R * X_SL;
    const sc = try xScratch(d, R, true, ha, &keep);
    const x = try upload(d, try bf16s(R * X_D, rng, 0.5, ha));
    const hdn = try upload(d, try bf16s(P * X_I, rng, 0.5, ha));
    const pw = try xPicks(R, rng, ha);
    const pb = try upload(d, std.mem.sliceAsBytes(pw.pick));
    const wb = try upload(d, std.mem.sliceAsBytes(pw.wts));
    const h = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
    const out = try cuda.DeviceBuffer.alloc(d, R * X_N * 4);
    for ([_]cuda.DeviceBuffer{ x, hdn, pb, wb, h, out }) |b| try keep.append(ha, b);
    const Moe = struct {
        fn run(rr: *Run, e_: weights.Experts, s_: mx.ExpertScratch, xp: u64, hd: u64, pk: u64, wt: u64, hb: cuda.DeviceBuffer, ob: cuda.DeviceBuffer, m: usize, on: bool) ![3][]u8 {
            rr.k.run128 = true;
            rr.k.sf = if (on) .exact else .off;
            defer {
                rr.k.run128 = false;
                rr.k.sf = .off;
            }
            try poison(hb.ptr, hb.len, rr.d, rr.s);
            try poison(s_.out, m * X_SL * X_N * 4, rr.d, rr.s);
            try poison(ob.ptr, ob.len, rr.d, rr.s);
            try mx.gateUp(rr.k, rr.s, rr.d, e_, s_, xp, X_D, pk, wt, m, 7.0, hb.ptr, X_MAIN, false);
            try rr.s.synchronize();
            const hh = try get(hb.ptr, m * X_SL * X_W * 2, rr.d);
            errdefer free(hh);
            try mx.down(rr.k, rr.s, e_, s_, hd, X_I, pk, m, ob.ptr, X_N, false, false);
            try rr.s.synchronize();
            return .{ hh, try get(s_.out, m * X_SL * X_N * 4, rr.d), try get(ob.ptr, m * X_N * 4, rr.d) };
        }
    };
    // the guard over the layer's experts: every tensor's range against the host's of the same bytes read back
    const tabs = try ha.alloc(u64, 3 * X_E);
    const owner = try ha.alloc(i32, X_E);
    const rs = try ha.alloc(mx.ByteRange, 3 * X_E);
    mx.expertScales(ex, tabs[0..X_E], tabs[X_E .. 2 * X_E], tabs[2 * X_E ..], owner, rs);
    const got = try ha.alloc([2]u8, rs.len);
    try mx.scaleRanges(r.k, d, r.s, std.heap.page_allocator, rs, got);
    // the guard's verdict: model.sfGuard on a layer of these experts (none kept, or the layer's experts in words)
    var lay = std.mem.zeroes(weights.Layer);
    lay.experts = ex;
    const Guard = struct {
        fn verdict(rr: *Run, l: weights.Layer) !?[]const u8 {
            rr.k.sf_nkept = 0;
            var kb: [320]u8 = undefined;
            const kept = try dsv41.model.sfGuard(std.heap.page_allocator, rr.d, rr.s, rr.k, &.{l}, &kb);
            if (rr.k.sf_nkept == 0 and std.mem.eql(u8, kept, "none")) return null;
            return if (mx.sfKept(rr.k, l.experts.native.?.s1) and rr.k.sf_nkept == 1 and std.mem.eql(u8, kept, "the experts in 1 layer")) "kept" else "WRONG words";
        }
    };
    const proven = (try Guard.verdict(r, lay)) == null;
    r.k.sf_nkept = 0;
    var bad: usize = 0;
    for (rs, got) |rg, g| {
        const b = try get(rg.ptr, rg.len, d);
        defer free(b);
        const w = hostRange(b);
        bad += @intFromBool(w[0] != g[0] or w[1] != g[1]);
    }
    // the baseline bytes at a chunk's row counts, the layer's scales proven
    var diff: [3]usize = @splat(0);
        const want = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, false);
        defer for (want) |w| free(w);
        const g = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, true);
        defer for (g) |w| free(w);
        for (&diff, want, g) |*n, w, gg| n.* += differs(w, gg);
    }
    const names = [_][]const u8{ "gate / up's bf16 pack", "down's fp32 entry outputs", "the combined fp32 rows" };
    for (names, diff) |nm, n| r.report(try std.fmt.bufPrint(line, "mma_sf_exact experts: {s} at 37, 64, 512, 2,048, 4,096 and 8,192 rows against the baseline 128-byte-run tiles", .{nm}), if (proven) n else 1, true);
    r.report("mma_sf_exact guard over the experts: every expert's three scale tensors (expertScales: the owner table's rest rows, the shared expert's FP8 blocks) against the host's of the same bytes", bad, true);
    // the edges on weight set 0 (experts 0, 8, .. 376): gate 33, up 137, down 33; the shared expert's 41 and 127
    const set0: [3]u64 = .{ tabs[0], tabs[X_E], tabs[2 * X_E] };
    const shared: [3]u64 = .{ tabs[X_E - 1], tabs[2 * X_E - 1], tabs[3 * X_E - 1] };
    const fills = [_]struct { at: u64, len: u64, b: u8 }{
        .{ .at = set0[0], .len = X_W * (X_D / 32), .b = 33 },           .{ .at = set0[1], .len = X_W * (X_D / 32), .b = 137 },
        .{ .at = set0[2], .len = X_N * (X_I / 32), .b = 33 },           .{ .at = shared[0], .len = (X_W / 32) * (X_D / 32), .b = 41 },
        .{ .at = shared[1], .len = (X_W / 32) * (X_D / 32), .b = 127 }, .{ .at = shared[2], .len = (X_N / 32) * (X_I / 32), .b = 41 },
    };
    const edge_proven = (try Guard.verdict(r, lay)) == null;
    r.k.sf_nkept = 0;
    var edge_diff: usize = 0;
    {
        const want = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 512, false);
        defer for (want) |w| free(w);
        const g = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 512, true);
        defer for (g) |w| free(w);
        for (want, g) |w, gg| edge_diff += differs(w, gg);
    }
    r.report("mma_sf_exact guard's edges on the experts: weight set 0's MXFP4 scale bytes 33 (2^-94) and 137 (2^10), the shared expert's FP8 41 and 127, proven and on the block-scaled kernels, at 512 rows against the baseline's", if (edge_proven) edge_diff else 1, true);
    // outside it: one MXFP4 scale byte 32 keeps the layer's experts on the baseline kernels
    const one = [1]u8{32};
    const out_verdict = try Guard.verdict(r, lay);
    defer r.k.sf_nkept = 0;
    const out_proven = !(out_verdict != null and std.mem.eql(u8, out_verdict.?, "kept"));
    var out_diff: usize = 0;
    {
        const want = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 512, false);
        defer for (want) |w| free(w);
        const g = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 512, true);
        defer for (g) |w| free(w);
        for (want, g) |w, gg| out_diff += differs(w, gg);
    }
    r.report("mma_sf_exact guard over the experts: one MXFP4 scale byte 32 keeps the layer's experts on the baseline kernels (model.sfGuard: \"the experts in 1 layer\"), at 512 rows against the baseline's", if (out_proven) 1 else out_diff, true);
    // the shared expert's FP8 rule: one of its scale bytes 128 (inside MXFP4's range, outside FP8's) keeps them too
    const back = [1]u8{33};
    const big = [1]u8{128};
    const sh_verdict = try Guard.verdict(r, lay);
    const sh_kept = sh_verdict != null and std.mem.eql(u8, sh_verdict.?, "kept");
    var sh_diff: usize = 0;
    {
        const want = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 512, false);
        defer for (want) |w| free(w);
        const g = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 512, true);
        defer for (g) |w| free(w);
        for (want, g) |w, gg| sh_diff += differs(w, gg);
    }
    r.report("mma_sf_exact guard over the experts: the shared expert's FP8 scale byte 128 (inside MXFP4's range, outside FP8's) keeps the layer's experts on the baseline kernels, at 512 rows against the baseline's", if (sh_kept) sh_diff else 1, true);
}

// -- TF_DS_GEMM_W16 ---------------------------------------------------------------------------------------------------

/// mx.gemm of rows [0, m) with the exact arithmetic (TF_DS_MMA_SF_EXACT) under TF_DS_GEMM_W16 on or off, its output of
/// `mode` into y (poisoned first); its bytes (get's).
fn gemmW16Once(r: *Run, w16: bool, l: weights.Linear, q: u64, s: u64, z: u64, m: usize, y: u64, mode: Mode) ![]u8 {
    r.k.sf = .exact;
    r.k.gemm_w16 = w16;
    defer {
        r.k.sf = .off;
        r.k.gemm_w16 = false;
    }
    return gemmOnce(r, .off, false, l, q, s, z, 0, m, y, mode, null);
}

/// Whether TF_DS_GEMM_W16 launches this GEMM kind's 16-warp kernel: the switch on, the exact arithmetic, the twin loaded.
fn w16Taken(r: *const Run, kind: mx.SfGemm) bool {
    const sk = r.k.sfk orelse return false;
    if (!r.k.gemm_w16 or r.k.sf != .exact) return false;
    return switch (kind) {
        .split64 => sk.w16.split64 != null,
        .split128 => sk.w16.split128 != null,
        .gemm, .zsplit => false,
    };
}

/// TF_DS_GEMM_W16's 16-warp kernels against the baseline 8-warp ones: the dense FP8 GEMMs of production's shapes (bf16 and
/// fp32 outputs at 64 .. 4,096 rows, a partial 64-row tile at 67, 1,000 and 2,048 and a partial 128-row tile at 1,025),
/// a 2D node's experts at 37 .. 4,096 rows (the FP8 shared expert among them) and wo_a's bf16 rows at 1 .. 4,096 rows:
/// every output's bytes equal, each launch's kernel asserted (the twin loaded, the shape's split the twin's); one
/// changed input differs in each. The twins' shared memory is the 8-warp kernels' (mx.sfExperts' 512-thread launch and
/// mx.sfGemm's are the tiles' own dynamic bytes).
fn gemmW16(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var line: [512]u8 = undefined;
    const sk = r.k.sfk;
    r.note("gemm_w16: the exact arithmetic's 16-warp kernels loaded (split64, split128, gate / up, down) and wo_a's", sk != null and sk.?.w16.split64 != null and sk.?.w16.split128 != null and sk.?.w16.gu != null and sk.?.w16.dn != null and r.k.pf_bf16w8_w16 != null);
    if (sk == null or r.k.pf_bf16w8_w16 == null) return;
    defer {
        r.k.sf = .off;
        r.k.gemm_w16 = false;
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
    }
    // the dense shapes: the same operands and launches on each side of the switch
    const rows = [_]usize{ 64, 67, 512, 1000, 1025, 2048, 4096 };
    for (sf_shapes) |sh| {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
        const max_m = rows[rows.len - 1];
        const x = try upload(d, try bf16s(max_m * sh.k, rng, 0.5, ha));
        try keep.append(ha, x);
        const q = try cuda.DeviceBuffer.alloc(d, max_m * sh.k);
        try keep.append(ha, q);
        const s = try cuda.DeviceBuffer.alloc(d, max_m * (sh.k / 32) * 4);
        try keep.append(ha, s);
        try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, max_m, sh.k, q.ptr, sh.k, s.ptr, sh.k / 32);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, max_m), 1) * 4);
        try keep.append(ha, z);
        const y = try cuda.DeviceBuffer.alloc(d, max_m * sh.n * 4);
        try keep.append(ha, y);
        // the exact GEMMs take a shape only with its key proven (TF_DS_MMA_SF_EXACT's guard over the scales)
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
        try mx.sfProve(r.k, l.s);
        const sp = mx.splits(sh.k, sh.n);
        var route = true;
        var twin = false;
        for (rows) |m| {
            const kind = mx.sfGemmOf(.exact, m, sp) orelse return error.TestUnexpectedResult;
            const has = kind == .split64 or kind == .split128;
            twin = twin or has;
            r.k.sf = .exact;
            r.k.gemm_w16 = true;
            route = route and w16Taken(r, kind) == has;
            r.k.sf = .off;
            r.k.gemm_w16 = false;
        }
        var diff: usize = 0;
        for ([_]Mode{ .bf16, .f32 }) |mode| for (rows) |m| {
            const off = try gemmW16Once(r, false, l, q.ptr, s.ptr, z.ptr, m, y.ptr, mode);
            defer free(off);
            const on = try gemmW16Once(r, true, l, q.ptr, s.ptr, z.ptr, m, y.ptr, mode);
            defer free(on);
            diff += differs(off, on);
        };
        r.report(try std.fmt.bufPrint(&line, "gemm_w16 {s} ({d} x {d}): bf16 and fp32 outputs at 64, 67, 512, 1,000, 1,025, 2,048 and 4,096 rows against the baseline 8-warp kernels ({s}, its route asserted{s})", .{ sh.name, sh.k, sh.n, if (twin) "the 16-warp twins" else "no K split, so no twin: the baseline kernel", if (route) "" else ": WRONG" }), if (route) diff else 1, true);
    }
    // the experts: gate / up, down's entry outputs and the combined rows, the switch on each side
    {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const ex = try xExperts(d, rng, ha, &keep);
        const nx = ex.native.?;
        const R: usize = 4096;
        const P = R * X_SL;
        const sc = try xScratch(d, R, true, ha, &keep);
        const x = try upload(d, try bf16s(R * X_D, rng, 0.5, ha));
        const hdn = try upload(d, try bf16s(P * X_I, rng, 0.5, ha));
        const pw = try xPicks(R, rng, ha);
        const pb = try upload(d, std.mem.sliceAsBytes(pw.pick));
        const wb = try upload(d, std.mem.sliceAsBytes(pw.wts));
        const h = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
        const out = try cuda.DeviceBuffer.alloc(d, R * X_N * 4);
        for ([_]cuda.DeviceBuffer{ x, hdn, pb, wb, h, out }) |b| try keep.append(ha, b);
        const Moe = struct {
            fn run(rr: *Run, e_: weights.Experts, s_: mx.ExpertScratch, xp: u64, hd: u64, pk: u64, wt: u64, hb: cuda.DeviceBuffer, ob: cuda.DeviceBuffer, m: usize, w16: bool) ![3][]u8 {
                rr.k.run128 = true;
                rr.k.sf = .exact;
                rr.k.gemm_w16 = w16;
                defer {
                    rr.k.run128 = false;
                    rr.k.sf = .off;
                    rr.k.gemm_w16 = false;
                }
                try poison(hb.ptr, hb.len, rr.d, rr.s);
                try poison(s_.out, m * X_SL * X_N * 4, rr.d, rr.s);
                try poison(ob.ptr, ob.len, rr.d, rr.s);
                try mx.gateUp(rr.k, rr.s, rr.d, e_, s_, xp, X_D, pk, wt, m, 7.0, hb.ptr, X_MAIN, false);
                try rr.s.synchronize();
                const hh = try get(hb.ptr, m * X_SL * X_W * 2, rr.d);
                errdefer free(hh);
                try mx.down(rr.k, rr.s, e_, s_, hd, X_I, pk, m, ob.ptr, X_N, false, false);
                try rr.s.synchronize();
                return .{ hh, try get(s_.out, m * X_SL * X_N * 4, rr.d), try get(ob.ptr, m * X_N * 4, rr.d) };
            }
        };
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
        try mx.sfProve(r.k, nx.s1);
        // the route: the exact twins at 512 threads, or the baseline kernels where the layer was not proven
        r.k.sf = .exact;
        r.k.gemm_w16 = true;
        const gu_l = mx.sfExperts(r.k, nx.s1, false);
        const dn_l = mx.sfExperts(r.k, nx.s1, true);
        r.k.sf = .off;
        r.k.gemm_w16 = false;
        const route = if (gu_l) |a| a.threads == 512 else false;
        const route_dn = if (dn_l) |a| a.threads == 512 else false;
        var diff: [3]usize = @splat(0);
        for ([_]usize{ 37, 64, 512, 2048, 4096 }) |m| {
            const off = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, false);
            defer for (off) |o| free(o);
            const on = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, m, true);
            defer for (on) |o| free(o);
            for (&diff, off, on) |*n, o, w| n.* += differs(o, w);
        }
        const names = [_][]const u8{ "gate / up's bf16 pack", "down's fp32 entry outputs", "the combined fp32 rows" };
        for (names, diff) |nm, n| r.report(try std.fmt.bufPrint(&line, "gemm_w16 experts: {s} at 37, 64, 512, 2,048 and 4,096 rows (the FP8 shared expert among them) against the baseline 8-warp kernels", .{nm}), if (route and route_dn) n else 1, true);
        // the control: one x value of row 30 set to 64
        const base = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, true);
        defer for (base) |o| free(o);
        const at = x.ptr + (30 * X_D + 300) * 2;
        var was: [2]u8 = undefined;
        const big = [2]u8{ 0x80, 0x42 };
        const got = try Moe.run(r, ex, sc, x.ptr, hdn.ptr, pb.ptr, wb.ptr, h, out, 64, true);
        defer for (got) |o| free(o);
        r.report("gemm_w16 experts control: gate / up's pack with one x value of row 30 set to 64", differs(base[0], got[0]), false);
    }
    // wo_a's bf16 rows on the 16-warp kernel
    {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const n_all = W8_GROUPS * W8_GN;
        const kk = W8_GROUPS * W8_GK;
        const max_m = w8_rows[w8_rows.len - 1];
        const codes = try ha.alloc(u8, n_all * W8_GK);
        for (codes) |*c| {
            c.* = rng.int(u8);
            if (c.* & 0x7f == 0x7f) c.* -= 1;
        }
        const scales = try ha.alloc(u8, (n_all / 32) * (W8_GK / 32));
        for (scales) |*e| e.* = 110 + rng.uintLessThan(u8, 18);
        const lay = try w8Layer(d, W8_GROUPS, W8_GN, W8_GK, codes, scales, ha, &keep);
        const o = try upload(d, try bf16s(max_m * kk, rng, 0.5, ha));
        try keep.append(ha, o);
        var sb: [4]cuda.DeviceBuffer = undefined;
        for (&sb, mx.Scratch.sizes(max_m, kk, 1)) |*b, n| {
            b.* = try cuda.DeviceBuffer.alloc(d, n);
            try keep.append(ha, b.*);
        }
        const sc: mx.Scratch = .{ .q = sb[0].ptr, .s = sb[1].ptr, .z = sb[2].ptr, .fake = sb[3].ptr, .rows = max_m, .max_k = kk };
        const u = try cuda.DeviceBuffer.alloc(d, max_m * n_all * 2);
        try keep.append(ha, u);
        const W = struct {
            fn run(rr: *Run, s_: mx.Scratch, w8: weights.Linear, op: u64, nk: usize, m: usize, y: u64, n: usize, w16: bool) ![]u8 {
                rr.k.gemm_w16 = w16;
                defer rr.k.gemm_w16 = false;
                try poison(y, m * n * 2, rr.d, rr.s);
                try mx.woA8Prefill(rr.k, rr.s, s_, w8, W8_GROUPS, op, nk, m, y, n);
                try rr.s.synchronize();
                return get(y, m * n * 2, rr.d);
            }
        };
        var diff: usize = 0;
        for (w8_rows) |m| {
            const off = try W.run(r, sc, lay.w8, o.ptr, kk, m, u.ptr, n_all, false);
            defer free(off);
            const on = try W.run(r, sc, lay.w8, o.ptr, kk, m, u.ptr, n_all, true);
            defer free(on);
            diff += differs(off, on);
        }
        r.report("gemm_w16 wo_a: u from its FP8 rows on 16 warps a block against the baseline 8-warp kernel at 1, 17, 63, 64, 65, 256, 512, 1,000, 2,048 and 4,096 rows, 2 groups of 1024 x 4096", diff, true);
        const m: usize = 256;
        const base = try W.run(r, sc, lay.w8, o.ptr, kk, m, u.ptr, n_all, true);
        defer free(base);
        const at = o.ptr + (100 * kk + 4100) * 2;
        var was: [2]u8 = undefined;
        const big = [2]u8{ 0x80, 0x42 };
        const got = try W.run(r, sc, lay.w8, o.ptr, kk, m, u.ptr, n_all, true);
        defer free(got);
        r.report("gemm_w16 wo_a control: one o value of row 100 set to 64", differs(base, got), false);
    }
}

    r.report("collect_peers all4 quarters (2,560 fp32 a node) at 1, 2, 3, 5, 16, 26 and 34 rows and rows 5 .. 10 of 34, every rank: against the baseline copy-out, and the baseline against the slots' bytes", all4, true);
    r.report("collect_peers head quarters (32,384 / 32,256 fp32 a node) at 1, 2 and 34 rows, every rank: against the baseline's, and the baseline against the slots' bytes", head, true);
    r.report("collect_peers column ring: catRank (2,048 bf16 a rank), the parity pack (8,960-byte rows, the partner's alone) and the pair gather (2,560 fp32) at 1, 2 and 16 rows, both ranks: against the baseline's, and the baseline against the slots' bytes", col, true);
    r.report("collect_peers odd widths (1, 3, 255, 1,023, 1,024, 1,025, 2,049 float4) at 1, 2 and 5 rows, a gap of 3 float4 between rows, rank 0 without a destination, on rank 2: against the baseline's, and the baseline against the slots' bytes", odd, true);
    r.report("collect_peers zero-width destination: rank 1's destination has a pointer and row4 0 in the 4-rank gather at 2 rows on rank 0: against the baseline's, and the baseline against the slots' bytes", zw, true);
    r.note("collect_peers: the switch runs dsv41_rdma_collect_peers (the ring's probe counts the baseline launch, 1, and not the switch's, 0)", counts[0] == 1 and counts[1] == 0);
        r.note(try std.fmt.bufPrint(&line, "head_cand {d} rows, count {d} (each node's {d}): the baseline kernel flags none; the nodes' parts merged into the baseline candidates, values' bits and ids", .{ R, c, dsv41.head_cand.kFor(c) }), x.flagged == 0 and x.merged and x.equal);
        r.note(try std.fmt.bufPrint(&line, "head_cand 2 rows, count {d} (each node's {d}): the baseline kernel flags none; the nodes' parts merged into the baseline candidates, values' bits and ids", .{ c, dsv41.head_cand.kFor(c) }), x.flagged == 0 and x.merged and x.equal);
        r.note("head_cand 2,000 ties at the cut, all in node g0: the baseline kernel flags none; merged into the baseline candidates", x.flagged == 0 and x.merged and x.equal);
        r.note(try std.fmt.bufPrint(&line, "head_cand ties across the nodes (grids of 1/8, 1, 2), 24 rounds of 2 rows, count 28: the baseline kernel flags {d} rows; {d} rounds merged into the baseline candidates ({d} not), {d} fell back (a cut value some node holds at its k-th and two nodes hold)", .{ flagged, merged, bad, fell }), flagged == 0 and merged > 0 and fell > 0 and bad == 0);
        r.note("head_cand 2,400 ties at the cut, 600 in each node: the baseline kernel flags the row; the merge falls back to the whole head", x.flagged == 1 and !x.merged);
        r.note("head_cand a NaN in node g1's columns: the baseline kernel flags the row; the merge falls back to the whole head", x.flagged == 1 and !x.merged);
        r.note("head_cand signed zeros at the cut over the nodes: the baseline kernel flags none; merged into the baseline candidates, each zero's sign bit kept", x.flagged == 0 and x.merged and x.equal and x.neg_zero);
        r.note(try std.fmt.bufPrint(&line, "head_cand timing {d} rows, count 28 (each node's {d}), us (a mean of 20 after 3; the engine's trace decides): the baseline kernel on 129,280 columns {d:.1}, a node's on 32,384 {d:.1} (x{d:.3}), the host merge {d:.2}; its lists merged as the baseline's", .{ R, k, today, one, one / today, merge_us }), today > 0 and one > 0 and ok and x.merged and x.equal and std.mem.eql(i64, ids, x.ids));
        r.note(try std.fmt.bufPrint(&line, "head_key {d} rows: the nodes' bests merged into the baseline argmax tokens", .{R}), x.equal);
        r.note("head_key the largest value twice in node g2 and in g1 and g3: the lowest id among them, the baseline tokens", x.equal and x.tokens[0] == hc_off[1] + 101);
        r.note("head_key a NaN at the vocabulary's first column (token 0) and at node g2's first (never larger): the baseline tokens", x.equal and x.tokens[0] == 0 and x.tokens[1] != 0);
        r.note("head_key every node's layout: the largest at each node's first and last column, ties across each boundary, NaNs at nodes' edges, nodes without a number, +inf in two nodes, signed zeros: the baseline tokens", x.equal and std.mem.eql(u32, &want, x.tokens));
        r.note("head_key a row of NaNs and a row of -inf alone: the baseline tokens (0, 0)", x.equal and x.tokens[0] == 0 and x.tokens[1] == 0);
        r.note(try std.fmt.bufPrint(&line, "head_key timing {d} rows, us (a mean of 20 after 3; the engine's trace decides): the baseline argmax on 129,280 columns {d:.1}, a node's part on 32,384 {d:.1} (x{d:.3}), the host merge {d:.2}; its bests merged as the baseline's", .{ R, today, one, one / today, merge_us }), today > 0 and one > 0 and x.equal and std.mem.eql(u32, tokens, x.tokens));
// -- the rings' data slots: write-combined and in-place --------------------------------------------------------------

/// A ring's two data regions and its metadata without the NIC: enough to run the staging, publish, await and collecting
/// kernels on the memory kinds the live rings use. The rotating slots are the engine's; user 0's in-place send slot
/// follows them, as rdma.Ring lays them out.
const RingRig = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    io: std.Io,
    hr: dsv41.rdma.HostReg,
    module: cuda.Module,
    k: dsv41.rdma.Kernels,
    send: dsv41.rdma.Pinned, // mmap + cuMemHostRegister (baseline)
    send_wc: dsv41.rdma.Pinned, // cudaHostAlloc(Mapped | WriteCombined)
    recv: dsv41.rdma.Pinned,
    recv_wc: dsv41.rdma.Pinned,
    meta: dsv41.rdma.Pinned, // the baseline memory: every word the CPU reads
    seq: cuda.DeviceBuffer,
    staged: cuda.DeviceBuffer,

    const slots: usize = 4;
    const world: usize = 2;
    const max_bytes: usize = 4096;
    const rank: usize = 0;
    const q: u64 = 1; // the first published sequence: rotating slot 1
    const user: usize = 0;

    const doorbell_w = slots * world;
    const sizes_w = doorbell_w + 1;
    const sent_w = sizes_w + slots;
    const abort_w = sent_w + slots;
    const send_q_w = abort_w + 1;
    const send_slot_w = send_q_w + slots;
    const send_last_w = send_slot_w + slots;
    const send_done_w = send_last_w + dsv41.rdma.max_users;
    const meta_words = send_done_w + dsv41.rdma.max_users;

    fn init(d: *const cuda.Driver, s: cuda.Stream, io: std.Io) !RingRig {
        var hr = try dsv41.rdma.HostReg.open();
        errdefer hr.lib.close();
        var send = try dsv41.rdma.Pinned.alloc(d, &hr, slots * max_bytes);
        errdefer send.free(&hr);
        var send_wc = try dsv41.rdma.Pinned.allocHost(d, (slots + dsv41.rdma.max_users) * max_bytes, true);
        errdefer send_wc.free(&hr);
        var recv = try dsv41.rdma.Pinned.alloc(d, &hr, slots * world * max_bytes);
        errdefer recv.free(&hr);
        var recv_wc = try dsv41.rdma.Pinned.allocHost(d, slots * world * max_bytes, true);
        errdefer recv_wc.free(&hr);
        var meta = try dsv41.rdma.Pinned.alloc(d, &hr, meta_words * 8);
        errdefer meta.free(&hr);
        var seq = try cuda.DeviceBuffer.alloc(d, 8);
        errdefer seq.free();
        var staged = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer staged.free();
        try seq.fill8(0, null);
        try staged.fill8(0, null);
        var module = try cuda.Module.load(d, cuda.kernels.dsv41_rdma);
        errdefer module.unload();
        return .{ .d = d, .s = s, .io = io, .hr = hr, .module = module, .k = try dsv41.rdma.Kernels.load(module), .send = send, .send_wc = send_wc, .recv = recv, .recv_wc = recv_wc, .meta = meta, .seq = seq, .staged = staged };
    }

    fn deinit(g: *RingRig) void {
        g.module.unload();
        g.staged.free();
        g.seq.free();
        g.meta.free(&g.hr);
        g.recv_wc.free(&g.hr);
        g.recv.free(&g.hr);
        g.send_wc.free(&g.hr);
        g.send.free(&g.hr);
        g.hr.lib.close();
    }

    fn metaDev(g: *const RingRig, word: usize) u64 {
        return g.meta.dev + word * 8;
    }

    fn words(g: *RingRig) []u64 {
        const p: [*]u64 = @ptrCast(@alignCast(g.meta.mem.ptr));
        return p[0..meta_words];
    }

    fn reset(g: *RingRig) void {
        @memset(g.meta.mem, 0);
    }

    fn setSeq(g: *RingRig, v: u64) !void {
        try g.s.synchronize();
    }

    /// dsv41_rdma_stage as rdma.Ring.stage launches it: no own slice, fence mode 1.
    fn stage(g: *RingRig, src: u64, n4: usize, send_ring: u64, own: u64, own_row4: usize, own_stride: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_int, @intCast(n4)));
        a.add(send_ring);
        a.add(g.metaDev(sizes_w));
        a.add(g.metaDev(doorbell_w));
        a.add(g.metaDev(sent_w));
        a.add(g.seq.ptr);
        a.add(@as(c_int, slots));
        a.add(@as(u64, max_bytes));
        a.add(@as(u64, 0));
        a.add(g.metaDev(abort_w));
        a.add(g.staged.ptr);
        a.add(@as(c_int, 1));
        a.add(own);
        a.add(@as(c_int, @intCast(own_row4)));
        a.add(@as(c_longlong, @intCast(own_stride)));
        a.add(@as(c_int, 1));
        try cuda.launch.launch(g.k.stage, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 512, .y = 1, .z = 1 }, .pdl = true }, g.s, &a);
    }

    /// dsv41_rdma_publish as rdma.Ring.publish launches it: user 0's fixed slot, no own copy (0).
    fn publish(g: *RingRig, nbytes: usize, own: u64, own_row4: usize, own_stride: usize) !void {
        var a: cuda.Args = .{};
        a.add(g.metaDev(sizes_w));
        a.add(g.metaDev(doorbell_w));
        a.add(g.seq.ptr);
        a.add(g.metaDev(send_q_w));
        a.add(g.metaDev(send_slot_w));
        a.add(g.metaDev(send_last_w));
        a.add(@as(c_int, slots));
        a.add(@as(c_int, user));
        a.add(g.metaDev(sent_w));
        a.add(g.metaDev(abort_w));
        // own_src: the fixed send slot this rig's producer filled; own_dst: the caller's layout
        const slot_src: u64 = g.send_wc.dev + (RingRig.slots + RingRig.user) * RingRig.max_bytes;
        a.add(if (own != 0) slot_src else 0);
        a.add(own);
        a.add(@as(c_int, @intCast(own_row4)));
        a.add(@as(c_longlong, @intCast(own_stride)));
        a.add(@as(u64, nbytes));
        a.add(@as(u64, 0));
        a.add(@as(c_int, 1));
        if (a.count != 17) return error.BadInfo; // dsv41_rdma_publish's 17 parameters
        try cuda.launch.launch(g.k.publish.?, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 512, .y = 1, .z = 1 }, .pdl = true }, g.s, &a);
    }

    fn awaitSend(g: *RingRig) !void {
        var a: cuda.Args = .{};
        a.add(g.metaDev(send_done_w));
        a.add(g.metaDev(send_last_w));
        a.add(@as(c_int, user));
        a.add(g.metaDev(abort_w));
        a.add(@as(c_int, 1));
        try cuda.launch.launch(g.k.await_send.?, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 }, .pdl = true }, g.s, &a);
    }

    /// The copy-out (dsv41_rdma_collect_into) of both ranks' slots into `dst` [2, n4 float4], as gather_into's.
    fn collect(g: *RingRig, recv_dev: u64, dst: u64, n4: usize) !void {
        var dd: dsv41.rdma.Dsts = .{};
        for (0..world) |p| {
            dd.ptr[p] = dst + p * n4 * 16;
            dd.row4[p] = @intCast(n4);
            dd.stride4[p] = @intCast(n4);
        }
        var copy4: u64 = 0;
        for (0..world) |p| if (p != rank) {
            copy4 += n4;
        };
        const ring: dsv41.rdma.CollectRing = .{ .recv = recv_dev, .flags = g.metaDev(0), .seq = g.seq.ptr, .abort = g.metaDev(abort_w), .rank = rank, .world = world };
        try dsv41.rdma.launchCollect(g.k, .{ .max_bytes = max_bytes, .slots = slots }, g.s, dd, 1, ring, copy4);
    }

    /// The generation-word copy-out (dsv41_rdma_collect_into_roci) of both ranks' slots into `dst`, the signal the
    /// waiting kernel sees and the failure record it writes, through the same launch the rings use.
    fn collectRoci(g: *RingRig, recv_dev: u64, dst: u64, n4: usize, gen: u64, lanes: u64, fault: u64, deadline_ns: u64, sig: dsv41.rdma.Roci) !void {
        var dd: dsv41.rdma.Dsts = .{};
        for (0..world) |p| {
            dd.ptr[p] = dst + p * n4 * 16;
            dd.row4[p] = @intCast(n4);
            dd.stride4[p] = @intCast(n4);
        }
        var copy4: u64 = 0;
        for (0..world) |p| if (p != rank) {
            copy4 += n4;
        };
        const ring: dsv41.rdma.CollectRing = .{ .recv = recv_dev, .flags = g.metaDev(0), .seq = g.seq.ptr, .abort = g.metaDev(abort_w), .rank = rank, .world = world, .gen = gen, .lanes = lanes, .fault = fault, .deadline_ns = deadline_ns, .roci = sig };
        try dsv41.rdma.launchCollect(g.k, .{ .max_bytes = max_bytes, .slots = slots, .roci = sig }, g.s, dd, 1, ring, copy4);
    }

    /// The NIC's landing of `nbytes` from a send slot at byte `off` of `send_mem` into this rank's slot, and every flag
    /// at q (the data before the flag, as the proxy publishes it).
    fn land(g: *RingRig, send_mem: []const u8, off: usize, recv_mem: []u8, nbytes: usize) void {
        const at = (@as(usize, @intCast(q % slots)) * world + rank) * max_bytes;
        @memcpy(recv_mem[at..][0..nbytes], send_mem[off..][0..nbytes]);
        const w = g.words();
        for (0..world) |p| w[@as(usize, @intCast(q % slots)) * world + p] = q;
    }
};

fn ringWc(r: *Run, rng_in: std.Random) !void {
    var prng = std.Random.DefaultPrng.init(0x77c1);
    const rng = prng.random();
    _ = rng_in;
    var g = try RingRig.init(r.d, r.s, r.io orelse return error.NoIo);
    defer g.deinit();
    r.note("ring_wc: the rings' fatbin has the in-place kernels (publish, await)", g.k.hasInplace());
    if (!g.k.hasInplace()) {
        for (0..5) |_| r.note("ring_wc: no publish/await kernel to run", false);
        return;
    }
    const n4: usize = 192; // 3,072 bytes
    const nbytes = n4 * 16;
    const host = try r.a.alloc(u8, nbytes);
    for (host) |*b| b.* = rng.int(u8);
    var src = try upload(r.d, host);
    defer src.free();
    const want = try get(src.ptr, nbytes, r.d);
    defer free(want);

    // the baseline slot, then the write-combined one: the same staging kernel, the same bytes read from the host
    g.reset();
    try g.setSeq(0);
    try g.stage(src.ptr, n4, g.send.dev, 0, n4, n4);
    try r.s.synchronize();
    const staging_today = differs(want, g.send.mem[@as(usize, @intCast(RingRig.q % RingRig.slots)) * RingRig.max_bytes ..][0..nbytes]);
    g.reset();
    try g.setSeq(0);
    try g.stage(src.ptr, n4, g.send_wc.dev, 0, n4, n4);
    try r.s.synchronize();
    const staging_wc = differs(want, g.send_wc.mem[@as(usize, @intCast(RingRig.q % RingRig.slots)) * RingRig.max_bytes ..][0..nbytes]);
    r.report("ring_wc: the staging kernel's send slot, cudaHostAlloc(Mapped | WriteCombined) and the baseline mmap + cuMemHostRegister, each against the source's bytes", staging_today + staging_wc, true);

    // the publish kernel writes the user's fixed slot and keeps every CPU-read word in the baseline memory
    g.reset();
    try g.setSeq(0);
    const slot_off = (RingRig.slots + RingRig.user) * RingRig.max_bytes;
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(g.send_wc.dev + slot_off, src.ptr, nbytes, r.s.handle), "cuMemcpyDtoDAsync");
    try g.awaitSend();
    try g.publish(nbytes, 0, n4, n4);
    try r.s.synchronize();
    const w = g.words();
    const meta_ok = w[RingRig.doorbell_w] == RingRig.q and w[RingRig.sizes_w + RingRig.q % RingRig.slots] == nbytes and
        w[RingRig.send_q_w + RingRig.q % RingRig.slots] == RingRig.q and
        w[RingRig.send_slot_w + RingRig.q % RingRig.slots] == RingRig.user and
        w[RingRig.send_last_w + RingRig.user] == RingRig.q;
    r.report("ring_wc: the publish kernel's write-combined slot against the source's bytes", differs(want, g.send_wc.mem[slot_off..][0..nbytes]), true);
    r.note("ring_wc: the publish's doorbell, size, slot name and last sequence read from the baseline (non-write-combined) metadata", meta_ok);

    // the collecting kernel out of a write-combined landing ring gives the baseline bytes
    const peer = try r.a.alloc(u8, nbytes);
    defer r.a.free(peer);
    for (peer) |*b| b.* = rng.int(u8);
    @memcpy(g.recv.mem[(RingRig.q % RingRig.slots) * RingRig.world * RingRig.max_bytes + RingRig.max_bytes ..][0..nbytes], peer);
    @memcpy(g.recv_wc.mem[(RingRig.q % RingRig.slots) * RingRig.world * RingRig.max_bytes + RingRig.max_bytes ..][0..nbytes], peer);
    for (0..2) |p| {
        g.words()[@as(usize, @intCast(RingRig.q % RingRig.slots)) * RingRig.world + p] = RingRig.q;
    }
    var dst_today = try cuda.DeviceBuffer.alloc(r.d, 2 * nbytes);
    defer dst_today.free();
    var dst_wc = try cuda.DeviceBuffer.alloc(r.d, 2 * nbytes);
    defer dst_wc.free();
    try r.d.check(r.d.api.cuMemsetD8Async(dst_today.ptr, 0x5a, 2 * nbytes, r.s.handle), "cuMemsetD8Async");
    try r.d.check(r.d.api.cuMemsetD8Async(dst_wc.ptr, 0x5a, 2 * nbytes, r.s.handle), "cuMemsetD8Async");
    try g.collect(g.recv.dev, dst_today.ptr, n4);
    try g.collect(g.recv_wc.dev, dst_wc.ptr, n4);
    try r.s.synchronize();
    const today = try get(dst_today.ptr, 2 * nbytes, r.d);
    defer free(today);
    const wc = try get(dst_wc.ptr, 2 * nbytes, r.d);
    defer free(wc);
    r.report("ring_wc: the collecting kernel out of write-combined landing slots against the baseline mmap slots, both ranks' bytes", differs(today, wc), true);

    // a control: one source byte changed moves the published slot
    var changed = try r.a.dupe(u8, host);
    defer r.a.free(changed);
    changed[0] ^= 0xff;
    var src2 = try upload(r.d, changed);
    defer src2.free();
    g.reset();
    try g.setSeq(0);
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(g.send_wc.dev + slot_off, src2.ptr, nbytes, r.s.handle), "cuMemcpyDtoDAsync");
    try g.awaitSend();
    try g.publish(nbytes, 0, n4, n4);
    try r.s.synchronize();
    r.report("ring_wc: one source byte changed moves the write-combined slot's bytes", differs(want, g.send_wc.mem[slot_off..][0..nbytes]), false);
}

fn ringInplace(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var g = try RingRig.init(r.d, r.s, r.io orelse return error.NoIo);
    defer g.deinit();
    r.note("ring_inplace: the rings' fatbin has the in-place kernels (publish, await)", g.k.hasInplace());
    if (!g.k.hasInplace()) {
        for (0..7) |_| r.note("ring_inplace: no publish/await kernel to run", false);
        return;
    }
    var prng = std.Random.DefaultPrng.init(0x1d1a);
    const rng = prng.random();
    const n4: usize = 192;
    const nbytes = n4 * 16;
    const host = try r.a.alloc(u8, nbytes);
    defer r.a.free(host);
    for (host) |*b| b.* = rng.int(u8);
    var src = try upload(r.d, host);
    defer src.free();
    const peer = try r.a.alloc(u8, nbytes);
    defer r.a.free(peer);
    for (peer) |*b| b.* = rng.int(u8);

    var dst_today = try cuda.DeviceBuffer.alloc(r.d, 2 * nbytes);
    defer dst_today.free();
    var dst_inplace = try cuda.DeviceBuffer.alloc(r.d, 2 * nbytes);
    defer dst_inplace.free();

    // baseline: stage into the rotating slot, land it, collect
    g.reset();
    try g.setSeq(0);
    try g.stage(src.ptr, n4, g.send.dev, dst_today.ptr, n4, n4);
    try r.s.synchronize();
    g.land(g.send.mem, @as(usize, @intCast(RingRig.q % RingRig.slots)) * RingRig.max_bytes, g.recv.mem, nbytes);
    @memcpy(g.recv.mem[(RingRig.q % RingRig.slots) * RingRig.world * RingRig.max_bytes + RingRig.max_bytes ..][0..nbytes], peer);
    for (0..2) |p| g.words()[@as(usize, @intCast(RingRig.q % RingRig.slots)) * RingRig.world + p] = RingRig.q;
    try g.collect(g.recv.dev, dst_today.ptr, n4);
    try r.s.synchronize();

    // in place: the producer wrote the user slot, await then publish, land it, collect from the write-combined ring
    g.reset();
    try g.setSeq(0);
    const slot_off = (RingRig.slots + RingRig.user) * RingRig.max_bytes;
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(g.send_wc.dev + slot_off, src.ptr, nbytes, r.s.handle), "cuMemcpyDtoDAsync");
    try g.awaitSend();
    try g.publish(nbytes, dst_inplace.ptr, n4, n4);
    try r.s.synchronize();
    g.land(g.send_wc.mem, slot_off, g.recv_wc.mem, nbytes);
    @memcpy(g.recv_wc.mem[(RingRig.q % RingRig.slots) * RingRig.world * RingRig.max_bytes + RingRig.max_bytes ..][0..nbytes], peer);
    for (0..2) |p| g.words()[@as(usize, @intCast(RingRig.q % RingRig.slots)) * RingRig.world + p] = RingRig.q;
    try g.collect(g.recv_wc.dev, dst_inplace.ptr, n4);
    try r.s.synchronize();

    const today = try get(dst_today.ptr, 2 * nbytes, r.d);
    defer free(today);
    const mine = try get(dst_inplace.ptr, 2 * nbytes, r.d);
    defer free(mine);
    const expect = try r.a.alloc(u8, 2 * nbytes);
    defer r.a.free(expect);
    @memcpy(expect[0..nbytes], host);
    @memcpy(expect[nbytes..], peer);
    r.report("ring_inplace: the in-place publish's copy-out against the baseline staged copy-out (both ranks' bytes, the sender's from a write-combined slot)", differs(today, mine), true);
    r.report("ring_inplace: the in-place copy-out against the bytes sent (this rank's partial and the peer's slot)", differs(mine, expect), true);

    // the await waits for the previous send: not done with last 5 and done 0, done once done reaches 5
    g.reset();
    g.words()[RingRig.send_last_w + RingRig.user] = 5;
    g.words()[RingRig.send_done_w + RingRig.user] = 0;
    try g.awaitSend();
    var waited: usize = 0;
    while (!(try r.s.done())) : (waited += 1) {
        if (waited == 2000) break;
        std.Io.sleep(g.io, .fromMilliseconds(1), .awake) catch {};
    }
    const blocked = !(try r.s.done());
    g.words()[RingRig.send_done_w + RingRig.user] = 5;
    if (blocked) g.words()[RingRig.abort_w] = 1; // (a host write that never arrives must not hang the replay)
    try r.s.synchronize();
    r.note("ring_inplace: the await holds the stream while the previous send is outstanding and ends when its sequence completes", blocked);

    // the publish's sequence and its slot name
    g.reset();
    try g.setSeq(0);
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(g.send_wc.dev + slot_off, src.ptr, nbytes, r.s.handle), "cuMemcpyDtoDAsync");
    try g.awaitSend();
    try g.publish(nbytes, 0, n4, n4);
    try r.s.synchronize();
    const w = g.words();
    const meta_ok = w[RingRig.doorbell_w] == RingRig.q and w[RingRig.sizes_w + RingRig.q % RingRig.slots] == nbytes and
        w[RingRig.send_q_w + RingRig.q % RingRig.slots] == RingRig.q and
        w[RingRig.send_slot_w + RingRig.q % RingRig.slots] == RingRig.user and
        w[RingRig.send_last_w + RingRig.user] == RingRig.q;
    r.note("ring_inplace: the publish names user 0's fixed send slot, its size and sequence, and rings the doorbell at q", meta_ok);

    // controls: one source byte and one landing byte changed differ
    var changed = try r.a.dupe(u8, host);
    defer r.a.free(changed);
    changed[0] ^= 0xff;
    var src2 = try upload(r.d, changed);
    defer src2.free();
    g.reset();
    try g.setSeq(0);
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(g.send_wc.dev + slot_off, src2.ptr, nbytes, r.s.handle), "cuMemcpyDtoDAsync");
    try g.awaitSend();
    try g.publish(nbytes, dst_inplace.ptr, n4, n4);
    try r.s.synchronize();
    @memcpy(g.recv_wc.mem[(RingRig.q % RingRig.slots) * RingRig.world * RingRig.max_bytes + RingRig.max_bytes ..][0..nbytes], peer);
    for (0..2) |p| g.words()[@as(usize, @intCast(RingRig.q % RingRig.slots)) * RingRig.world + p] = RingRig.q;
    g.land(g.send_wc.mem, slot_off, g.recv_wc.mem, nbytes);
    try g.collect(g.recv_wc.dev, dst_inplace.ptr, n4);
    try r.s.synchronize();
    const ctl = try get(dst_inplace.ptr, 2 * nbytes, r.d);
    defer free(ctl);
    r.report("ring_inplace: one source byte changed differs", differs(mine, ctl), false);

    var peer2 = try r.a.dupe(u8, peer);
    defer r.a.free(peer2);
    peer2[0] ^= 0xff;
    @memcpy(g.recv_wc.mem[(RingRig.q % RingRig.slots) * RingRig.world * RingRig.max_bytes + RingRig.max_bytes ..][0..nbytes], peer2);
    for (0..2) |p| g.words()[@as(usize, @intCast(RingRig.q % RingRig.slots)) * RingRig.world + p] = RingRig.q;
    try g.collect(g.recv_wc.dev, dst_inplace.ptr, n4);
    try r.s.synchronize();
    const ctl2 = try get(dst_inplace.ptr, 2 * nbytes, r.d);
    defer free(ctl2);
    r.report("ring_inplace: one landing byte changed differs", differs(ctl, ctl2), false);
    r.note("ring_inplace: user 0's send slot begins after the rotating slots, at slots x max_bytes", (RingRig.slots + RingRig.user) * RingRig.max_bytes == RingRig.slots * RingRig.max_bytes + RingRig.user * RingRig.max_bytes);
}

// -- TF_DS_RING_WORD: the generation word wakes the collecting kernel ----------------------------------------------

/// A host writer, standing in for the sending NIC: it stores (or fetch-adds) one lane's 64-bit generation word after
/// the collecting kernel is queued, so the kernel's wait is what the test exercises. A separate thread, because the
/// kernel only returns once the word is in.
const WordWriter = struct {
    fn store(io: std.Io, p: *u64, delay_ms: i64, value: u64) void {
        std.Io.sleep(io, .fromMilliseconds(delay_ms), .awake) catch {};
        @atomicStore(u64, p, value, .release);
    }
    fn add(io: std.Io, p: *u64, delay_ms: i64) void {
        std.Io.sleep(io, .fromMilliseconds(delay_ms), .awake) catch {};
        _ = @atomicRmw(u64, p, .Add, 1, .release);
    }
    fn abort(io: std.Io, p: *u64, delay_ms: i64) void {
        std.Io.sleep(io, .fromMilliseconds(delay_ms), .awake) catch {};
        @atomicStore(u64, p, 1, .release);
    }
};

/// TF_DS_RING_WORD's collecting kernel against the baseline on the same slots: the word signal and the atomic signal
/// (the host thread writes the words after the launch), a changed landing byte, a late word, a lane that never arrives
/// (the failure record, the poison and the untouched destination, within the deadline) and the abort word ending a wait
/// early. The bytes are the baseline collect kernel's from the same receive ring.
fn ringWord(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var g = try RingRig.init(r.d, r.s, r.io orelse return error.NoIo);
    defer g.deinit();
    r.note("ring_word: the rings' fatbin has the generation-word kernels (dsv41_rdma_collect_roci, _into_roci, _peers_roci)", g.k.hasRoci());
    if (!g.k.hasRoci()) {
        for (0..7) |_| r.note("ring_word: no generation-word kernel to run", false);
        return;
    }
    var prng = std.Random.DefaultPrng.init(0x0c1dec);
    const rng = prng.random();
    const n4: usize = 192;
    const nbytes = n4 * 16;
    const peer = try r.a.alloc(u8, nbytes);
    defer r.a.free(peer);
    for (peer) |*b| b.* = rng.int(u8);
    const slot = RingRig.q % RingRig.slots;
    const peer_off = @as(usize, @intCast(slot)) * RingRig.world * RingRig.max_bytes + RingRig.max_bytes;

    // the generation words (ordinary pinned memory: the CPU and the peers' NICs read-modify-write them), the lanes
    // toward each peer, and the first expired wait's record
    var gen = try dsv41.rdma.Pinned.alloc(r.d, &g.hr, dsv41.rdma.rociWords() * 8);
    defer gen.free(&g.hr);
    var lanes_buf = try cuda.DeviceBuffer.alloc(r.d, 2 * 4);
    defer lanes_buf.free();
    const lane_counts = [2]u32{ 0, 2 };
    var fault = try cuda.DeviceBuffer.alloc(r.d, 64);
    defer fault.free();
    try fault.fill8(0, null);
    const words: [*]u64 = @ptrCast(@alignCast(gen.mem.ptr));
    const lane_at = 1 * dsv41.rdma.max_devices; // the sender is peer 1; rank 0 reads its words
    var dst_today = try cuda.DeviceBuffer.alloc(r.d, 2 * nbytes);
    defer dst_today.free();
    var dst = try cuda.DeviceBuffer.alloc(r.d, 2 * nbytes);
    defer dst.free();

    // the baseline collecting kernel: the peer's slot in, its flag at q, into the reference bytes
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    g.words()[@as(usize, @intCast(slot)) * RingRig.world + 1] = RingRig.q;
    try poison(dst_today.ptr, 2 * nbytes, r.d, r.s);
    try g.collect(g.recv.dev, dst_today.ptr, n4);
    try r.s.synchronize();
    const want = try get(dst_today.ptr, 2 * nbytes, r.d);
    defer free(want);

    const deadline_ns: u64 = 500 * std.time.ns_per_ms;

    // the word signal: the same slot, the host thread writing both lanes' words after the launch
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .word);
    var w0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 2, RingRig.q });
    var w1 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at + 1], 2, RingRig.q });
    w0.join();
    w1.join();
    try r.s.synchronize();
    const got = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(got);
    r.report("ring_word: the word signal (each lane's sequence written by a host thread after the launch) against the baseline collecting kernel on the same slots", differs(want, got), true);

    // the same signal out of write-combined landing slots (TF_DS_RING_WC's memory), against the baseline collect there
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv_wc.mem[peer_off..][0..nbytes], peer);
    g.words()[@as(usize, @intCast(slot)) * RingRig.world + 1] = RingRig.q;
    try poison(dst_today.ptr, 2 * nbytes, r.d, r.s);
    try g.collect(g.recv_wc.dev, dst_today.ptr, n4);
    try r.s.synchronize();
    const want_wc = try get(dst_today.ptr, 2 * nbytes, r.d);
    defer free(want_wc);
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv_wc.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv_wc.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .word);
    var wc0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 2, RingRig.q });
    var wc1 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at + 1], 2, RingRig.q });
    wc0.join();
    wc1.join();
    try r.s.synchronize();
    const got_wc = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(got_wc);
    r.report("ring_word: the word signal out of write-combined landing slots against the baseline collect on the same slots", differs(want_wc, got_wc), true);

    // the atomic signal: the host fetch-adds one per lane; the word the kernel waits on counts to the sequence
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .atomic);
    var a0 = try std.Thread.spawn(.{}, WordWriter.add, .{ g.io, &words[lane_at], 2 });
    var a1 = try std.Thread.spawn(.{}, WordWriter.add, .{ g.io, &words[lane_at + 1], 2 });
    a0.join();
    a1.join();
    try r.s.synchronize();
    const atomic_got = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(atomic_got);
    r.report("ring_word: the atomic signal (one fetch-and-add a lane, the word counting to the sequence) against the baseline collecting kernel on the same slots", differs(want, atomic_got), true);

    // the control: one landing byte changed differs
    var peer2 = try r.a.dupe(u8, peer);
    defer r.a.free(peer2);
    peer2[0] ^= 0xff;
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer2);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .word);
    var c0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 2, RingRig.q });
    var c1 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at + 1], 2, RingRig.q });
    c0.join();
    c1.join();
    try r.s.synchronize();
    const ctl = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(ctl);
    r.report("ring_word: one landing byte changed differs the collected bytes", differs(want, ctl), false);

    // a late word (20 ms) still collects under a 500 ms deadline: the wait is bounded, not a fixed sleep
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .word);
    var l0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 20, RingRig.q });
    var l1 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at + 1], 20, RingRig.q });
    l0.join();
    l1.join();
    try r.s.synchronize();
    const late = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(late);
    r.report("ring_word: lanes' words written 20 ms after the launch still collect the baseline bytes under the 500 ms deadline", differs(want, late), true);

    // one lane's word never arrives: the deadline ends the wait with the record, the ring poisoned, nothing copied
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try fault.fill8(0, null);
    const short_deadline: u64 = 50 * std.time.ns_per_ms;
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, short_deadline, .word);
    var n0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 2, RingRig.q });
    n0.join();
    try r.s.synchronize();
    var rec: [8]u64 = undefined;
    const dead = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(dead);
    var untouched: usize = 0;
    for (dead) |b| untouched += @intFromBool(b != 0xa5);
    const record_ok = rec[0] == 0x524f_4349_0000_0001 and rec[1] == RingRig.q and (rec[3] >> 32) == 1 and
        (rec[3] & 0xffff_ffff) == 1 and rec[4] == 0 and rec[7] == 2 and g.words()[RingRig.abort_w] == 1;
    r.note("ring_word: a lane's word that never arrives ends in the failure record (magic, sequence, peer 1 lane 1, the word seen 0, the deadline) with the ring's abort word set and the destination untouched", record_ok and untouched == 0);
    r.note("ring_word: the expired wait ended within its deadline and well before the replay would give up", rec[5] >= 45 * std.time.ns_per_ms and rec[5] < 5 * std.time.ns_per_s);

    // the abort word alone ends a wait early (why 1), with a deadline far away
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try fault.fill8(0, null);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, std.time.ns_per_s, .word);
    var ab = try std.Thread.spawn(.{}, WordWriter.abort, .{ g.io, &g.words()[RingRig.abort_w], 2 });
    ab.join();
    try r.s.synchronize();
    r.note("ring_word: the abort word ends a waiting kernel well before its 1 s deadline (the record's why is the abort)", rec[0] == 0x524f_4349_0000_0001 and rec[7] == 1 and rec[5] < 900 * std.time.ns_per_ms);
}

// -- TF_DS_ATTN_EPI_FUSED --------------------------------------------------------------------------------------------

/// attnEpiFused with _sparse_attn_part_xe (TF_DS_ATTN_TWIN=x) as the fused launch: the same cases, control and line count; its bytes are
/// the baseline part then epilogue, so _xe is _ue's.
fn attnEpiXe(r: *Run, rng: std.Random) !void {
    const was = tri_attn.attn_epi_xe;
    defer tri_attn.attn_epi_xe = was;
    tri_attn.attn_epi_xe = true;
    try attnEpiFused(r, rng);
}

/// attnEpiFused with TF_DS_ATTN_TWIN's _sparse_attn_part_we as the fused launch (tri_attn.attn_epi_we): the same cases, control and line count;
/// its bytes are the baseline part then epilogue, so _we is _ue's (and _w's mainloop is _u's).
fn attnEpiWe(r: *Run, rng: std.Random) !void {
    const was = tri_attn.attn_epi_we;
    defer tri_attn.attn_epi_we = was;
    tri_attn.attn_epi_we = true;
    try attnEpiFused(r, rng);
}

/// A prompt chunk's attention with the epilogue in its final store (tri_attn.sparseAttn with `epi`, the extra set's
/// _sparse_attn_part_ue) against the part kernel's twin then mx.attnEpilogue (rope_heads' inverse RoPE and the quant in one
/// launch), at the engine's shapes (16 heads of 512, 256 rows at the window's end of a 1M context, 5 .. 512 picks): the
/// codes and scales, and wo_a's fake rows, byte-equal (o is never written by the fused launch); one picked row's code byte
/// changed differs; every launch found the set's variant.
fn attnEpiFused(r: *Run, rng: std.Random) !void {
    const d = r.d;
    const H: usize = 16;
    const HD: usize = 512;
    const RD: usize = 64;
    const K = H * HD;
    const WIN: usize = 128;
    const end: usize = 1 << 20;
    const ncomp: usize = end / 4;
    const R: usize = 256;
    const start = end - R;
    var line: [256]u8 = undefined;
    const was = tri_attn.attn_epi_fused;
    defer tri_attn.attn_epi_fused = was;
    const tag: []const u8 = if (tri_attn.attn_epi_xe) "attn_epi_xe" else if (tri_attn.attn_epi_we) "attn_epi_we" else "attn_epi_fused"; // (_we: the fused part with _w's mainloop, against the baseline part then epilogue)
    const we_misses0 = tri_attn.attn_we_misses.load(.monotonic);
    if (r.t.extra == null) {
        for (0..7) |_| r.note(try std.fmt.bufPrint(&line, "{s}: the kit's extra set (aot-extra/) holds _sparse_attn_part_ue", .{tag}), false);
        return;
    }
    const misses0 = tri_attn.attn_epi_misses.load(.monotonic);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const codes = try ha.alloc(u8, ncomp * HD / 2);
    rng.bytes(codes);
    const scales = try ha.alloc(u8, ncomp * HD / 16);
    for (scales) |*x| x.* = 0x18 + rng.uintLessThan(u8, 0x3f - 0x18);
    const cb = try upload(d, codes);
    const sb = try upload(d, scales);
    const cos = try upload(d, try f32s(end * (RD / 2), rng, ha));
    const sin = try upload(d, try f32s(end * (RD / 2), rng, ha));
    for ([_]cuda.DeviceBuffer{ cb, sb, cos, sin }) |b| try keep.append(ha, b);
    for ([_]usize{ 5, 40, 70, 200, 512 }) |n_idx| {
        const n_pad = (n_idx + 15) / 16 * 16;
        const q = try upload(d, try bf16s(R * H * HD, rng, 0.5, ha));
        const wsrc = try upload(d, try bf16s((R + WIN) * HD, rng, 0.5, ha));
        const idx = try ha.alloc(i64, R * n_pad);
        for (0..R) |i| for (0..n_pad) |j| {
            idx[i * n_pad + j] = if (j >= n_idx or (j + i) % 29 == 7) -1 else @intCast(rng.uintLessThan(usize, (start + i + 1) / 4));
        };
        const ib = try upload(d, std.mem.sliceAsBytes(idx));
        const pos = try ha.alloc(i64, R);
        for (pos, 0..) |*x, i| x.* = @intCast(start + i);
        const pb = try upload(d, std.mem.sliceAsBytes(pos));
        var wlo_v = [1]i64{@intCast(start - WIN)};
        const wb = try upload(d, std.mem.sliceAsBytes(&wlo_v));
        const sink = try upload(d, try f32s(H, rng, ha));
        const out = try cuda.DeviceBuffer.alloc(d, R * K * 2);
        const qr = try cuda.DeviceBuffer.alloc(d, R * K);
        const sr = try cuda.DeviceBuffer.alloc(d, R * K / 32 * 4);
        const fr = try cuda.DeviceBuffer.alloc(d, R * K * 2);
        const qf = try cuda.DeviceBuffer.alloc(d, R * K);
        const sf = try cuda.DeviceBuffer.alloc(d, R * K / 32 * 4);
        const ff = try cuda.DeviceBuffer.alloc(d, R * K * 2);
        for ([_]cuda.DeviceBuffer{ q, wsrc, ib, pb, wb, sink, out, qr, sr, fr, qf, sf, ff }) |b| try keep.append(ha, b);
        var a: tri_attn.Attn = .{ .q = q.ptr, .out = out.ptr, .rows = R, .h = H, .hd = HD, .sink = sink.ptr, .wsrc = wsrc.ptr, .wsrc_rows = R + WIN, .wlo = wb.ptr, .ring = false, .comp = .{ .fp4 = .{ .codes = cb.ptr, .scales = sb.ptr } }, .idx = ib.ptr, .n_idx = n_idx, .pos = pb.ptr, .scale = 1.0 / @sqrt(@as(f32, HD)), .window = WIN, .u = true };
        // the baseline: the part (twin), then the epilogue launch (codes and scales; then the fake rows from a fresh o)
        tri_attn.attn_epi_fused = false;
        try tri_attn.sparseAttn(r.t, a);
        try mx.attnEpilogue(r.k, r.s, out.ptr, cos.ptr, sin.ptr, pb.ptr, R, H, HD, RD, qr.ptr, sr.ptr, 0, true);
        try tri_attn.sparseAttn(r.t, a);
        try mx.attnEpilogue(r.k, r.s, out.ptr, cos.ptr, sin.ptr, pb.ptr, R, H, HD, RD, 0, 0, fr.ptr, true);
        try r.s.synchronize();
        // the switch: the epilogue in the part's final store (poisoned outputs: a store not made shows)
        tri_attn.attn_epi_fused = true;
        for ([_]cuda.DeviceBuffer{ qf, sf, ff }) |b| try poison(b.ptr, b.len, d, r.s);
        var ran = false;
        a.epi_ran = &ran;
        a.epi = .{ .cos = cos.ptr, .sin = sin.ptr, .rd = RD, .codes = qf.ptr, .scales = sf.ptr, .fake = 0 };
        try tri_attn.sparseAttn(r.t, a);
        var ran_fake = false;
        a.epi_ran = &ran_fake;
        a.epi = .{ .cos = cos.ptr, .sin = sin.ptr, .rd = RD, .codes = 0, .scales = 0, .fake = ff.ptr };
        try tri_attn.sparseAttn(r.t, a);
        try r.s.synchronize();
        var n: usize = 0;
        n += try cmpBytes(qr.ptr, qf.ptr, R * K, d);
        n += try cmpBytes(sr.ptr, sf.ptr, R * K / 32 * 4, d);
        n += try cmpBytes(fr.ptr, ff.ptr, R * K * 2, d);
        if (!ran or !ran_fake) n += 1;
        r.report(try std.fmt.bufPrint(&line, "{s}: {d} rows at the window's end, {d} picks: codes, scales and wo_a's fake rows from the part's final store against the part then the epilogue launch", .{ tag, R, n_idx }), n, true);
        if (n_idx == 512) {
            // the control: a code byte of the first row's first pick changed
            const row: usize = @intCast(idx[0]);
            var one = [1]u8{codes[row * HD / 2] ^ 0x11};
            a.epi_ran = null;
            a.epi = .{ .cos = cos.ptr, .sin = sin.ptr, .rd = RD, .codes = qf.ptr, .scales = sf.ptr, .fake = 0 };
            try tri_attn.sparseAttn(r.t, a);
            try r.s.synchronize();
            r.report(try std.fmt.bufPrint(&line, "{s} control: a code byte of row 0's first pick changed", .{tag}), try cmpBytes(qr.ptr, qf.ptr, R * K, d), false);
        }
    }
    r.note(try std.fmt.bufPrint(&line, "{s}: every launch found the extra set's variant (none kept the baseline part and epilogue, none took _ue for _we)", .{tag}), tri_attn.attn_epi_misses.load(.monotonic) == misses0 and tri_attn.attn_we_misses.load(.monotonic) == we_misses0);
}

// -- TF_DS_HC_FUSED on the two-block layout --------------------------------------------------------------------------

/// A prompt chunk's mHC mixes and finish in one launch on the owner sums' two bf16 blocks (tri_hc.hcPrePf with
/// TF_DS_HC_FUSED in effect and `blk_rs` set, the extra set's _hc_pre_pf_fused_2b) against the baseline _hc_mix_pf_2b then
/// _hc_finish, at 1,024 and 4,096 rows: the normed rows, pre, post, comb and the posted streams byte-equal; one stream
/// value changed differs; every launch found the set's variant.
fn hcFused2b(r: *Run, rng: std.Random) !void {
    const D: usize = 5120;
    const HC: usize = 4;
    const NB: usize = 40;
    var line: [256]u8 = undefined;
    const was = tri_hc.hc_fused;
    defer tri_hc.hc_fused = was;
    if (r.t.extra == null) {
        for (0..3) |_| r.note("hc_fused_2b: the kit's extra set (aot-extra/) holds _hc_pre_pf_fused_2b", false);
        return;
    }
    const misses0 = tri_hc.hc_fused_misses.load(.monotonic);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const d = r.d;
    const fn_w = try upload(d, try f32s(24 * HC * D, rng, ha));
    const scale = try upload(d, try f32s(3, rng, ha));
    const base = try upload(d, try f32s(24, rng, ha));
    const norm_w = try upload(d, try bf16s(D, rng, 0.5, ha));
    for ([_]cuda.DeviceBuffer{ fn_w, scale, base, norm_w }) |b| try keep.append(ha, b);
    for ([_]usize{ 1024, 4096 }) |rows| {
        const h = try upload(d, try bf16s(rows * HC * D, rng, 0.5, ha));
        const g = try upload(d, try bf16s(2 * rows * (D / 2), rng, 0.5, ha));
        const post0 = try upload(d, try f32s(rows * HC, rng, ha));
        const comb0 = try upload(d, try f32s(rows * HC * HC, rng, ha));
        const pre_in = try upload(d, try f32s(rows * HC, rng, ha));
        const part = try cuda.DeviceBuffer.alloc(d, rows * NB * 32 * 4);
        const out_a = try cuda.DeviceBuffer.alloc(d, rows * D * 2);
        const pre_a = try cuda.DeviceBuffer.alloc(d, rows * HC * 4);
        const post_a = try cuda.DeviceBuffer.alloc(d, rows * HC * 4);
        const comb_a = try cuda.DeviceBuffer.alloc(d, rows * HC * HC * 4);
        const hout_a = try cuda.DeviceBuffer.alloc(d, rows * HC * D * 2);
        const post_b = try cuda.DeviceBuffer.alloc(d, rows * HC * 4);
        const comb_b = try cuda.DeviceBuffer.alloc(d, rows * HC * HC * 4);
        const out_b = try cuda.DeviceBuffer.alloc(d, rows * D * 2);
        const pre_b = try cuda.DeviceBuffer.alloc(d, rows * HC * 4);
        const hout_b = try cuda.DeviceBuffer.alloc(d, rows * HC * D * 2);
        for ([_]cuda.DeviceBuffer{ h, g, post0, comb0, pre_in, part, out_a, pre_a, post_a, comb_a, hout_a, post_b, comb_b, out_b, pre_b, hout_b }) |b| try keep.append(ha, b);
        const posted_a: tri_hc.Posted = .{ .gathered = g.ptr, .world = 2, .h_out = hout_a.ptr, .blk_rs = rows * (D / 2) };
        const posted_b: tri_hc.Posted = .{ .gathered = g.ptr, .world = 2, .h_out = hout_b.ptr, .blk_rs = rows * (D / 2) };
        tri_hc.hc_fused = false;
        try copyBuf(d, r.s, post_a.ptr, post0.ptr, rows * HC * 4);
        try copyBuf(d, r.s, comb_a.ptr, comb0.ptr, rows * HC * HC * 4);
        _ = try tri_hc.hcPrePf(r.t, h.ptr, fn_w.ptr, scale.ptr, base.ptr, pre_in.ptr, norm_w.ptr, 1e-20, 1e-6, 20, out_a.ptr, pre_a.ptr, post_a.ptr, comb_a.ptr, part.ptr, posted_a, rows, D);
        try r.s.synchronize();
        tri_hc.hc_fused = true;
        try copyBuf(d, r.s, post_b.ptr, post0.ptr, rows * HC * 4);
        try copyBuf(d, r.s, comb_b.ptr, comb0.ptr, rows * HC * HC * 4);
        _ = try tri_hc.hcPrePf(r.t, h.ptr, fn_w.ptr, scale.ptr, base.ptr, pre_in.ptr, norm_w.ptr, 1e-20, 1e-6, 20, out_b.ptr, pre_b.ptr, post_b.ptr, comb_b.ptr, part.ptr, posted_b, rows, D);
        try r.s.synchronize();
        var n: usize = 0;
        n += try cmpBytes(out_a.ptr, out_b.ptr, rows * D * 2, d);
        n += try cmpBytes(pre_a.ptr, pre_b.ptr, rows * HC * 4, d);
        n += try cmpBytes(post_a.ptr, post_b.ptr, rows * HC * 4, d);
        n += try cmpBytes(comb_a.ptr, comb_b.ptr, rows * HC * HC * 4, d);
        n += try cmpBytes(hout_a.ptr, hout_b.ptr, rows * HC * D * 2, d);
        r.report(try std.fmt.bufPrint(&line, "hc_fused_2b {d} rows, the sums in two bf16 blocks: out, pre, post, comb and the posted streams", .{rows}), n, true);
        if (rows == 4096) {
            const at = h.ptr + @as(u64, @intCast((3 * HC + 2) * D + 17)) * 2;
            var old: [2]u8 = undefined;
            const big = [2]u8{ 0x00, 0x42 };
            _ = try tri_hc.hcPrePf(r.t, h.ptr, fn_w.ptr, scale.ptr, base.ptr, pre_in.ptr, norm_w.ptr, 1e-20, 1e-6, 20, out_b.ptr, pre_b.ptr, post_b.ptr, comb_b.ptr, part.ptr, posted_b, rows, D);
            try r.s.synchronize();
            r.report("hc_fused_2b control: one stream value of row 3 set to 32", try cmpBytes(out_a.ptr, out_b.ptr, rows * D * 2, d), false);
        }
    }
    r.note("hc_fused_2b: every launch found the extra set's variant (none kept the two-block mix and the finish)", tri_hc.hc_fused_misses.load(.monotonic) == misses0);
}

// -- TF_DS_HC_2S -----------------------------------------------------------------------------------------------------

/// The two-block mHC mix with the four streams of a K block in one program (tri_hc.hcMixPf with TF_DS_HC_2S in effect, the extra
/// set's _hc_mix_pf_2s at RB 2) against _hc_mix_pf_2b, at 1,024 and 4,096 rows and 17 rows (a chunk past the last program's
/// rows): the posted streams and the block partials byte-equal; one sum changed differs; every launch found the variant.
fn hc2s(r: *Run, rng: std.Random) !void {
    const D: usize = 5120;
    const HC: usize = 4;
    const NB: usize = 40;
    var line: [256]u8 = undefined;
    const was = tri_hc.hc_2s;
    defer tri_hc.hc_2s = was;
    if (r.t.extra == null) {
        for (0..4) |_| r.note("hc_2s: the kit's extra set (aot-extra/) holds _hc_mix_pf_2s", false);
        return;
    }
    const misses0 = tri_hc.hc_2s_misses.load(.monotonic);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const d = r.d;
    const fn_w = try upload(d, try f32s(24 * HC * D, rng, ha));
    try keep.append(ha, fn_w);
    for ([_]usize{ 1024, 4096, 17 }) |rows| {
        const h = try upload(d, try bf16s(rows * HC * D, rng, 0.5, ha));
        const g = try upload(d, try bf16s(2 * rows * (D / 2), rng, 0.5, ha));
        const post = try upload(d, try f32s(rows * HC, rng, ha));
        const comb = try upload(d, try f32s(rows * HC * HC, rng, ha));
        const part_a = try cuda.DeviceBuffer.alloc(d, rows * NB * 32 * 4);
        const part_b = try cuda.DeviceBuffer.alloc(d, rows * NB * 32 * 4);
        const hout_a = try cuda.DeviceBuffer.alloc(d, rows * HC * D * 2);
        const hout_b = try cuda.DeviceBuffer.alloc(d, rows * HC * D * 2);
        for ([_]cuda.DeviceBuffer{ h, g, post, comb, part_a, part_b, hout_a, hout_b }) |b| try keep.append(ha, b);
        try poison(part_a.ptr, rows * NB * 32 * 4, d, r.s);
        try poison(part_b.ptr, rows * NB * 32 * 4, d, r.s);
        tri_hc.hc_2s = false;
        _ = try tri_hc.hcMixPf(r.t, h.ptr, fn_w.ptr, part_a.ptr, .{ .gathered = g.ptr, .world = 2, .h_out = hout_a.ptr, .blk_rs = rows * (D / 2) }, post.ptr, comb.ptr, rows, D);
        tri_hc.hc_2s = true;
        _ = try tri_hc.hcMixPf(r.t, h.ptr, fn_w.ptr, part_b.ptr, .{ .gathered = g.ptr, .world = 2, .h_out = hout_b.ptr, .blk_rs = rows * (D / 2) }, post.ptr, comb.ptr, rows, D);
        try r.s.synchronize();
        var n: usize = 0;
        n += try cmpBytes(part_a.ptr, part_b.ptr, rows * NB * 32 * 4, d);
        n += try cmpBytes(hout_a.ptr, hout_b.ptr, rows * HC * D * 2, d);
        r.report(try std.fmt.bufPrint(&line, "hc_2s {d} rows: the block partials and the posted streams, four streams a program against one", .{rows}), n, true);
        if (rows == 4096) {
            const at = g.ptr + @as(u64, @intCast(rows * (D / 2) + 100)) * 2;
            var old: [2]u8 = undefined;
            const big = [2]u8{ 0x00, 0x42 };
            _ = try tri_hc.hcMixPf(r.t, h.ptr, fn_w.ptr, part_b.ptr, .{ .gathered = g.ptr, .world = 2, .h_out = hout_b.ptr, .blk_rs = rows * (D / 2) }, post.ptr, comb.ptr, rows, D);
            try r.s.synchronize();
            r.report("hc_2s control: one sum of block 1, row 0 set to 32", try cmpBytes(hout_a.ptr, hout_b.ptr, rows * HC * D * 2, d), false);
        }
    }
    r.note("hc_2s: every launch found the extra set's variant (none kept _hc_mix_pf_2b)", tri_hc.hc_2s_misses.load(.monotonic) == misses0);
}

// -- TF_DS_ROCI's two-block readers ----------------------------------------------------------------------------------

/// bf16 [2, rows, w] sums with the special values the identity argument names scattered in (quiet NaN, +-0, subnormals,
/// +-inf, the largest finite).
fn rociBlocks(rows: usize, w: usize, rng: std.Random, a: std.mem.Allocator) ![]u16 {
    const v = try a.alloc(u16, 2 * rows * w);
    for (v) |*x| x.* = bf16Rne(rng.floatNorm(f32) * 2.0);
    const special = [_]u16{ 0x7fc0, 0x0000, 0x8000, 0x7f80, 0xff80, 0x7f7f, 0x0001, 0x8001, 0xff7f };
    for (0..@min(v.len / 4, 3000)) |_| v[rng.uintLessThan(usize, v.len)] = special[rng.uintLessThan(usize, special.len)];
    return v;
}

const RociIn = struct { blk: cuda.DeviceBuffer, planes: cuda.DeviceBuffer, rows: usize };

/// The baseline input of the readers from the blocks: widenPairs into plane 0 of the tight fp32 buffer, plane 1 -0.0.
fn rociPlanes(r: *Run, blk: cuda.DeviceBuffer, rows: usize, w: usize) !cuda.DeviceBuffer {
    const planes = try cuda.DeviceBuffer.alloc(r.d, 2 * rows * 2 * w * 4);
    try r.ops.?.widenPairs(r.s, blk.ptr, blk.ptr + rows * w * 2, planes.ptr, rows, w);
    try r.d.check(r.d.api.cuMemsetD32Async(planes.ptr + rows * 2 * w * 4, dsv41.prompt2d.plane1_word, rows * 2 * w, r.s.handle), "cuMemsetD32Async");
    return planes;
}

fn rociReaders(r: *Run, rng: std.Random) !void {
    const D: usize = 5120;
    const W: usize = D / 2;
    const EW: usize = 12800;
    var line: [256]u8 = undefined;
    if (r.t.extra == null or r.ops == null) return r.note("roci_readers: the kit's extra set (aot-extra/) loads and the ops are available", false);
    r.note("roci_readers: the kit's extra set (aot-extra/) loads", true);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    const hc_blocks = dsv41.tri_basic.hc_blocks;
    for ([_]usize{ 1, 5, 13, 16, 100, 512, 1024 }) |rows| {
        const sums = try rociBlocks(rows, W, rng, ha);
        var blk = try upload(r.d, std.mem.sliceAsBytes(sums));
        defer blk.free();
        var planes = try rociPlanes(r, blk, rows, W);
        defer planes.free();
        var h = try upload(r.d, try bf16s(rows * 4 * D, rng, 1.0, ha));
        defer h.free();
        var post = try upload(r.d, try f32s(rows * 4, rng, ha));
        defer post.free();
        var comb = try upload(r.d, try f32s(rows * 16, rng, ha));
        defer comb.free();
        var fnw = try upload(r.d, try f32s(24 * 4 * D, rng, ha));
        defer fnw.free();
        var outs: [2]cuda.DeviceBuffer = undefined;
        var parts: [2]cuda.DeviceBuffer = undefined;
        for (&outs, &parts, 0..) |*o, *pt, i| {
            o.* = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            pt.* = try cuda.DeviceBuffer.alloc(r.d, rows * hc_blocks * 32 * 4);
            try poison(o.ptr, rows * 4 * D * 2, r.d, r.s);
            try poison(pt.ptr, rows * hc_blocks * 32 * 4, r.d, r.s);
            const posted: dsv41.tri_hc.Posted = if (i == 0) .{ .gathered = planes.ptr, .world = 2, .h_out = o.ptr } else .{ .gathered = blk.ptr, .world = 2, .h_out = o.ptr, .blk_rs = rows * W };
            _ = try dsv41.tri_hc.hcMixPf(r.t, h.ptr, fnw.ptr, pt.ptr, posted, post.ptr, comb.ptr, rows, D);
        }
        try r.s.synchronize();
        const xo = [2][]u8{ try get(outs[0].ptr, rows * 4 * D * 2, r.d), try get(outs[1].ptr, rows * 4 * D * 2, r.d) };
        const pp = [2][]u8{ try get(parts[0].ptr, rows * hc_blocks * 32 * 4, r.d), try get(parts[1].ptr, rows * hc_blocks * 32 * 4, r.d) };
        r.report(try std.fmt.bufPrint(&line, "roci_readers {d} rows: _hc_mix_pf_2b's streams and partial dots against widenPairs + _hc_mix_pf (NaN, +-0, subnormals, +-inf in the sums)", .{rows}), differs(xo[0], xo[1]) + differs(pp[0], pp[1]), true);
        for (&outs) |*o| o.free();
        for (&parts) |*pt| pt.free();
        // hc_post in place on a copy of the streams
        var hp: [2]cuda.DeviceBuffer = undefined;
        for (&hp, 0..) |*b, i| {
            b.* = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(b.ptr, h.ptr, rows * 4 * D * 2, r.s.handle), "cuMemcpyDtoDAsync");
            if (i == 0) try dsv41.tri_basic.hcPost(r.t, planes.ptr, b.ptr, post.ptr, comb.ptr, b.ptr, 2, rows, D) else try dsv41.tri_basic.hcPost2b(r.t, blk.ptr, rows * W, b.ptr, post.ptr, comb.ptr, b.ptr, rows, D);
        }
        try r.s.synchronize();
        const ph = [2][]u8{ try get(hp[0].ptr, rows * 4 * D * 2, r.d), try get(hp[1].ptr, rows * 4 * D * 2, r.d) };
        r.report(try std.fmt.bufPrint(&line, "roci_readers {d} rows: _hc_post_2b (the tile at column 2,560 straddles the blocks) against widenPairs + _hc_post", .{rows}), differs(ph[0], ph[1]), true);
        for (&hp) |*b| b.free();
        // the Engram gate: the key and value row [2 blocks of 12,800] as one row of 25,600 against the blocks as they lie
        const eb = try rociBlocks(rows, EW, rng, ha);
        var ekb = try upload(r.d, std.mem.sliceAsBytes(eb));
        defer ekb.free();
        var kv = try cuda.DeviceBuffer.alloc(r.d, rows * 2 * EW * 2);
        defer kv.free();
        for (0..2) |b| try r.ops.?.copyRows(r.s, ekb.ptr + b * rows * EW * 2, EW * 2, kv.ptr + b * EW * 2, 2 * EW * 2, EW * 2, rows);
        var qk = try upload(r.d, try f32s(4 * D, rng, ha));
        defer qk.free();
        var eo: [2]cuda.DeviceBuffer = undefined;
        for (&eo, 0..) |*o, i| {
            o.* = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            try poison(o.ptr, rows * 4 * D * 2, r.d, r.s);
            if (i == 0) try dsv41.tri_basic.engramGate(r.t, h.ptr, kv.ptr, qk.ptr, o.ptr, 1e-20, rows, D) else try dsv41.tri_basic.engramGate2b(r.t, h.ptr, ekb.ptr, rows * EW, qk.ptr, o.ptr, 1e-20, rows, D);
        }
        try r.s.synchronize();
        const eg = [2][]u8{ try get(eo[0].ptr, rows * 4 * D * 2, r.d), try get(eo[1].ptr, rows * 4 * D * 2, r.d) };
        r.report(try std.fmt.bufPrint(&line, "roci_readers {d} rows: _engram_gate_2b against the key and value rows assembled + _engram_gate", .{rows}), differs(eg[0], eg[1]), true);
        for (&eo) |*o| o.free();
        // controls, once: one sum changed in the last row's last column must change each reader's output
        if (rows == 512) {
            const bad = try ha.dupe(u16, sums);
            bad[2 * rows * W - 1] = if (bad[2 * rows * W - 1] == 0x4100) 0x4200 else 0x4100;
            var blk2 = try upload(r.d, std.mem.sliceAsBytes(bad));
            defer blk2.free();
            var c1 = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            defer c1.free();
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(c1.ptr, h.ptr, rows * 4 * D * 2, r.s.handle), "cuMemcpyDtoDAsync");
            try dsv41.tri_basic.hcPost2b(r.t, blk2.ptr, rows * W, c1.ptr, post.ptr, comb.ptr, c1.ptr, rows, D);
            try r.s.synchronize();
            r.report("roci_readers control: _hc_post_2b with one sum changed (the last row's last column, block 1)", differs(ph[0], try get(c1.ptr, rows * 4 * D * 2, r.d)), false);
            const bad_e = try ha.dupe(u16, eb);
            bad_e[2 * rows * EW - 1] = if (bad_e[2 * rows * EW - 1] == 0x4100) 0x4200 else 0x4100;
            var ekb2 = try upload(r.d, std.mem.sliceAsBytes(bad_e));
            defer ekb2.free();
            var c2 = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            defer c2.free();
            try poison(c2.ptr, rows * 4 * D * 2, r.d, r.s);
            try dsv41.tri_basic.engramGate2b(r.t, h.ptr, ekb2.ptr, rows * EW, qk.ptr, c2.ptr, 1e-20, rows, D);
            try r.s.synchronize();
            r.report("roci_readers control: _engram_gate_2b with the value's last column changed", differs(eg[0], try get(c2.ptr, rows * 4 * D * 2, r.d)), false);
            var c3 = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            defer c3.free();
            var c3p = try cuda.DeviceBuffer.alloc(r.d, rows * hc_blocks * 32 * 4);
            defer c3p.free();
            try poison(c3.ptr, rows * 4 * D * 2, r.d, r.s);
            try poison(c3p.ptr, rows * hc_blocks * 32 * 4, r.d, r.s);
            _ = try dsv41.tri_hc.hcMixPf(r.t, h.ptr, fnw.ptr, c3p.ptr, .{ .gathered = blk2.ptr, .world = 2, .h_out = c3.ptr, .blk_rs = rows * W }, post.ptr, comb.ptr, rows, D);
            try r.s.synchronize();
            r.report("roci_readers control: _hc_mix_pf_2b with one sum changed", differs(xo[0], try get(c3.ptr, rows * 4 * D * 2, r.d)), false);
        }
    }
}

// -- TF_DS_ROCI's whole-line GEMM stores -------------------------------------------------------------------------------

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A dense FP8 GEMM into y in write-combined host memory (the transport's arena), which takes the _st kernels, against the
/// same GEMM into device memory, byte for byte: wo_b's shape and an Engram-width one, the fold / split A / split C7 tiles and
/// the exact block-scaled kernel (its key proven), at 64 to 4,096 rows. The time of each, in ms a launch, is printed beside.
fn rociStage(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x57a9e);
    const rng = prng.random();
    const d = r.d;
    var line: [320]u8 = undefined;
    const shapes = [_]Shape{ .{ .name = "wo_b", .k = 4096, .n = 2560 }, .{ .name = "engram", .k = 5120, .n = 12800 } };
    const rows_list = [_]usize{ 64, 129, 1000, 4096 };
    const max_m: usize = 4096;
    var arena_mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&arena_mem, max_m * 12800 * 4, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(arena_mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem, 0), "cuMemHostGetDevicePointer");
    try r.k.useStaged(wc, wc + max_m * 12800 * 4);
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
        r.k.sf_nproven = 0;
    }
    var last_control: []u8 = &.{};
    for (shapes) |sh| {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
        const x = try upload(d, try bf16s(max_m * sh.k, rng, 0.5, ha));
        try keep.append(ha, x);
        const q = try cuda.DeviceBuffer.alloc(d, max_m * sh.k);
        try keep.append(ha, q);
        const sc = try cuda.DeviceBuffer.alloc(d, max_m * (sh.k / 32) * 4);
        try keep.append(ha, sc);
        try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, max_m, sh.k, q.ptr, sh.k, sc.ptr, sh.k / 32);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, max_m), 1) * 4);
        try keep.append(ha, z);
        const y = try cuda.DeviceBuffer.alloc(d, max_m * sh.n * 4);
        try keep.append(ha, y);
        const names = [_][]const u8{ "fold / tile", "split A", "split C7", "block-scaled exact" };
        for (names, 0..) |nm, v| {
            var diff: usize = 0;
            var t_dev: u64 = 0;
            var t_arena: u64 = 0;
            for (rows_list) |m| {
                const run = struct {
                    fn go(rr: *Run, vv: usize, ll: weights.Linear, qq: u64, ss: u64, zz: u64, mm: usize, yy: u64) ![]u8 {
                        return switch (vv) {
                            0 => gemmOnce(rr, .off, false, ll, qq, ss, zz, 0, mm, yy, .f32, null),
                            1 => gemmOnce(rr, .a, false, ll, qq, ss, zz, 0, mm, yy, .f32, null),
                            2 => gemmOnce(rr, .c7, false, ll, qq, ss, zz, 0, mm, yy, .f32, null),
                            else => gemmExact(rr, true, ll, qq, ss, zz, mm, yy, .f32),
                        };
                    }
                };
                if (v == 3) {
                    if (r.k.sfk == null) continue;
                    try mx.sfProve(r.k, l.s);
                }
                r.k.wc = .{ 0, 0 };
                const a = try run.go(r, v, l, q.ptr, sc.ptr, z.ptr, m, y.ptr);
                defer free(a);
                r.k.wc = .{ wc, wc + max_m * 12800 * 4 };
                const b = try run.go(r, v, l, q.ptr, sc.ptr, z.ptr, m, wc);
                defer free(b);
                diff += differs(a, b);
                if (v == 0 and m == 4096 and sh.n == 2560) last_control = try std.heap.page_allocator.dupe(u8, a);
                if (m == 4096) {
                    for ([_]u1{ 0, 1 }) |on| {
                        r.k.wc = if (on == 1) .{ wc, wc + max_m * 12800 * 4 } else .{ 0, 0 };
                        const yy = if (on == 1) wc else y.ptr;
                        const t0 = nowNs();
                        for (0..10) |_| free(try run.go(r, v, l, q.ptr, sc.ptr, z.ptr, m, yy));
                        if (on == 1) t_arena = nowNs() - t0 else t_dev = nowNs() - t0;
                    }
                }
            }
            if (v == 3 and r.k.sfk == null) {
                r.note(try std.fmt.bufPrint(&line, "roci_stage {s} ({d} x {d}) {s}: the block-scaled kernels (native_sf.cu) are not loaded on this GPU", .{ sh.name, sh.k, sh.n, nm }), false);
                continue;
            }
            std.debug.print("roci_stage {s} {s} at 4,096 rows (ms a launch incl. the readback): device y {d:.2}, arena y {d:.2}\n", .{ sh.name, nm, @as(f64, @floatFromInt(t_dev)) / 1e7, @as(f64, @floatFromInt(t_arena)) / 1e7 });
            r.report(try std.fmt.bufPrint(&line, "roci_stage {s} ({d} x {d}), {s}: y in write-combined host memory against y in device memory at 64, 129, 1,000 and 4,096 rows", .{ sh.name, sh.k, sh.n, nm }), diff, true);
        }
    }
    // control: one activation code changed in row 700 changes the arena GEMM's bytes (the comparison is live)
    r.k.wc = .{ wc, wc + max_m * 12800 * 4 };
    defer if (last_control.len != 0) free(last_control);
    {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const sh = shapes[0];
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
        const qb = try ha.alloc(u8, 1024 * sh.k);
        @memset(qb, 0x38);
        const sb = try ha.alloc(f32, 1024 * (sh.k / 32));
        @memset(sb, 1.0);
        const q = try upload(d, qb);
        try keep.append(ha, q);
        const sc = try upload(d, std.mem.sliceAsBytes(sb));
        try keep.append(ha, sc);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, 1024), 1) * 4);
        try keep.append(ha, z);
        const a = try gemmOnce(r, .off, false, l, q.ptr, sc.ptr, z.ptr, 0, 1000, wc, .f32, null);
        defer free(a);
        const b = try gemmOnce(r, .off, false, l, q.ptr, sc.ptr, z.ptr, 0, 1000, wc, .f32, null);
        defer free(b);
        r.report("roci_stage control: one activation code of row 700 changed, the arena GEMM's bytes", differs(a, b), false);
    }
}

// -- TF_DS_ROCI_COLLECT's copy ---------------------------------------------------------------------------------------

/// Ops.collect2: two blocks from device memory into write-combined host memory (the transport's arena) in one launch, at the
/// owner sums' part 1 sizes (wo_b's and MoE's psum blocks 10,485,760 bytes, Engram's 52,428,800): the bytes equal, and the
/// launch's time printed (ms, 20 launches). A source changed after the copy must differ from the copy.
fn rociCollect(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const d = r.d;
    var prng = std.Random.DefaultPrng.init(0xc011ec7);
    const rng = prng.random();
    var line: [256]u8 = undefined;
    const ops = r.ops orelse return r.note("roci_collect: the ops are available", false);
    const big: usize = 52_428_800;
    var mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&mem, 3 * big, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, mem, 0), "cuMemHostGetDevicePointer");
    var src = try cuda.DeviceBuffer.alloc(d, 2 * big);
    defer src.free();
    const host = try std.heap.page_allocator.alloc(u8, 2 * big);
    defer std.heap.page_allocator.free(host);
    rng.bytes(host);
    var kept: []u8 = &.{};
    defer if (kept.len != 0) free(kept);
    for ([_]usize{ 10_485_760, big }) |bytes| {
        // the blocks sit at different offsets of the source and of the arena, as the plan's rows do
        try poison(wc, 3 * big, d, r.s);
        try ops.collect2(r.s, src.ptr, wc + big, src.ptr + big, wc, bytes);
        try r.s.synchronize();
        const a = try get(wc + big, bytes, d);
        defer free(a);
        const b = try get(wc, bytes, d);
        defer free(b);
        r.report(try std.fmt.bufPrint(&line, "roci_collect {d} bytes a block: both blocks in the arena equal their sources", .{bytes}), differs(a, host[0..bytes]) + differs(b, host[big..][0..bytes]), true);
        const t0 = nowNs();
        for (0..20) |_| try ops.collect2(r.s, src.ptr, wc + big, src.ptr + big, wc, bytes);
        try r.s.synchronize();
        std.debug.print("roci_collect {d} bytes a block, two blocks: {d:.3} ms a launch\n", .{ bytes, @as(f64, @floatFromInt(nowNs() - t0)) / 20e6 });
        if (bytes == big) kept = try std.heap.page_allocator.dupe(u8, a);
    }
    // control: the first block's source changed after the copy no longer equals the copy
    host[7] ^= 0x55;
    r.report("roci_collect control: the source changed after the copy", differs(kept, host[0..big]), false);
}

// -- the cost of splitting wo_b and combine into row quarters (TF_DS_ROCI sends only two of four row blocks) ----------

/// Milliseconds a launch of `f` takes, kernel only: `reps` launches between two timing events, no readback, the mean.
fn kernelMs(r: *Run, comptime Ctx: type, ctx: Ctx, comptime f: fn (Ctx) anyerror!void, reps: usize) !f32 {
    var t0 = try cuda.Event.init(r.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(r.d, true);
    defer t1.deinit();
    for (0..3) |_| try f(ctx); // warm
    try t0.record(r.s);
    for (0..reps) |_| try f(ctx);
    try t1.record(r.s);
    try t1.synchronize();
    return (try cuda.Event.elapsedMs(t0, t1)) / @as(f32, @floatFromInt(reps));
}

/// wo_b's block-scaled exact GEMM (K 4096, N 2560) and the MoE combine (7 slots, 2,560 columns), each as one launch of 4,096 rows
/// and as four of 1,024 (the sent quarters are not adjacent), into device memory and into the arena's write-combined host memory:
/// kernel time only (events around the launches, no readback). The combine indexes rows relative to its arguments, so a row start
/// is a pointer offset.
fn rociSplitCost(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x59117);
    const rng = prng.random();
    const d = r.d;
    var line: [640]u8 = undefined;
    const K: usize = 4096;
    const N: usize = 2560;
    const M: usize = 4096;
    var arena_mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&arena_mem, M * N * 4, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(arena_mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem, 0), "cuMemHostGetDevicePointer");
    try r.k.useStaged(wc, wc + M * N * 4);
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    const x = try upload(d, try bf16s(M * K, rng, 0.5, ha));
    try keep.append(ha, x);
    const q = try cuda.DeviceBuffer.alloc(d, M * K);
    try keep.append(ha, q);
    const sc = try cuda.DeviceBuffer.alloc(d, M * (K / 32) * 4);
    try keep.append(ha, sc);
    try mx.quant(r.k, r.s, x.ptr, .bf16, K, M, K, q.ptr, K, sc.ptr, K / 32);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, M), 1) * 4);
    try keep.append(ha, z);
    const y = try cuda.DeviceBuffer.alloc(d, M * N * 4);
    try keep.append(ha, y);
    if (r.k.sfk == null) return r.note("roci_split_cost: the block-scaled kernels (native_sf.cu) are loaded on this GPU", false);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    const G = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        sc: u64,
        z: u64,
        y: u64,
        parts: usize,
        fn go(g: @This()) anyerror!void {
            const m = M / g.parts;
            for (0..g.parts) |i| try mx.gemm(g.r.k, g.r.s, g.l, g.q + i * m * K, K, g.sc + i * m * (K / 32) * 4, K / 32, m, g.y + i * m * N * 4, N, .f32, false, g.z);
        }
    };
    var g_ms: [2][2]f32 = undefined; // [y: device, arena][parts: 1, 4]
    for ([_]u64{ y.ptr, wc }, 0..) |yy, yi| {
        r.k.wc = if (yi == 1) .{ wc, wc + M * N * 4 } else .{ 0, 0 };
        for ([_]usize{ 1, 4 }, 0..) |parts, pi| g_ms[yi][pi] = try kernelMs(r, G, .{ .r = r, .l = l, .q = q.ptr, .sc = sc.ptr, .z = z.ptr, .y = yy, .parts = parts }, G.go, 20);
    }
    const txt = try std.fmt.bufPrint(&line, "roci_split_cost wo_b (K {d}, N {d}) block-scaled exact, kernel only, ms for 4,096 rows: device y {d:.3} as 1 launch, {d:.3} as 4 x 1,024; arena y (whole-line stores) {d:.3} as 1, {d:.3} as 4 x 1,024", .{ K, N, g_ms[0][0], g_ms[0][1], g_ms[1][0], g_ms[1][1] });
    r.note(txt, true);
    std.debug.print("{s}\n", .{txt});
    // combine: 4,096 rows x 7 slots, expert ids distinct in a row
    const S: usize = 7;
    const E: usize = 384;
    const picks = try ha.alloc(i32, M * S);
    for (0..M) |row| {
        var seen: [S]i32 = undefined;
        var k: usize = 0;
        while (k < S) {
            const e: i32 = @intCast(rng.uintLessThan(usize, E));
            if (std.mem.indexOfScalar(i32, seen[0..k], e) != null) continue;
            seen[k] = e;
            picks[row * S + k] = e;
            k += 1;
        }
    }
    const pk = try upload(d, std.mem.sliceAsBytes(picks));
    try keep.append(ha, pk);
    const outs = try cuda.DeviceBuffer.alloc(d, M * S * N * 4);
    try keep.append(ha, outs);
    try d.check(d.api.cuMemsetD32Async(outs.ptr, 0x3f800000, M * S * N, r.s.handle), "cuMemsetD32Async");
    const C = struct {
        r: *Run,
        out: u64,
        pick: u64,
        y: u64,
        parts: usize,
        fn go(c: @This()) anyerror!void {
            const m = M / c.parts;
            for (0..c.parts) |i| {
                var ca: cuda.Args = .{};
                ca.add(c.out + i * m * S * N * 4);
                ca.add(@as(i64, @intCast(N)));
                ca.add(c.pick + i * m * S * 4);
                ca.add(@as(c_int, @intCast(m)));
                ca.add(@as(c_int, @intCast(S)));
                ca.add(@as(c_int, @intCast(E)));
                ca.add(@as(c_int, @intCast(N)));
                ca.add(c.y + i * m * N * 4);
                ca.add(@as(i64, @intCast(N)));
                ca.add(@as(c_int, 0));
                try cuda.launch.launch(c.r.k.combine, .{ .grid = .{ .x = @intCast((m * N + 255) / 256) }, .block = .{ .x = 256 } }, c.r.s, &ca);
            }
        }
    };
    var c_ms: [2][2]f32 = undefined;
    for ([_]u64{ y.ptr, wc }, 0..) |yy, yi| for ([_]usize{ 1, 4 }, 0..) |parts, pi| {
        c_ms[yi][pi] = try kernelMs(r, C, .{ .r = r, .out = outs.ptr, .pick = pk.ptr, .y = yy, .parts = parts }, C.go, 20);
    };
    const txt2 = try std.fmt.bufPrint(&line, "roci_split_cost combine (7 slots, N {d}), kernel only, ms for 4,096 rows: device y {d:.3} as 1 launch, {d:.3} as 4 x 1,024; arena y {d:.3} as 1, {d:.3} as 4 x 1,024", .{ N, c_ms[0][0], c_ms[0][1], c_ms[1][0], c_ms[1][1] });
    r.note(txt2, true);
    std.debug.print("{s}\n", .{txt2});
}

// -- TF_DS_ROCI_COMBINE_FIRST ------------------------------------------------------------------------------------------

/// What the host step between the combine's two stages sees (mx.CombineFirst.between): the stream drained, the first stage's quarters
/// equal to the reference's and the last stage's quarters not written yet.
const CombineStep = struct {
    r: *Run,
    y: u64,
    ref: []const u8,
    cf: mx.CombineFirst,
    row_bytes: usize,
    called: usize = 0,
    first_wrong: usize = 0,
    last_written: usize = 0,

    fn between(p: *anyopaque) anyerror!void {
        const c: *CombineStep = @ptrCast(@alignCast(p));
        c.called += 1;
        try c.r.s.synchronize();
        const blk = c.cf.rows * c.row_bytes;
        for (c.cf.first) |row0| {
            const got = try get(c.y + row0 * c.row_bytes, blk, c.r.d);
            defer free(got);
            c.first_wrong += differs(got, c.ref[row0 * c.row_bytes ..][0..blk]);
        }
        for (c.cf.last) |row0| {
            const got = try get(c.y + row0 * c.row_bytes, blk, c.r.d);
            defer free(got);
            for (got) |b| c.last_written += @intFromBool(b != 0xa5);
        }
    }
};

/// TF_DS_ROCI_COMBINE_FIRST's routed sum (mx.combineFirst: the quarters an owner sum's first part sends, a host step, the quarters the
/// node keeps) against tf_dsn_combine over all 4,096 rows in one launch (7 slots, 2,560 columns, random entries' outputs, picks
/// of distinct expert ids), the rows into device memory and into the arena's write-combined host memory: every output byte equal.
/// The host step sees the sent quarters complete and the kept ones unwritten. Controls: a kept quarter left out must differ.
fn rociCombineFirst(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xc0b1);
    const rng = prng.random();
    const d = r.d;
    var line: [320]u8 = undefined;
    const S: usize = 7;
    const E: usize = 384;
    const N: usize = 2560;
    const M: usize = 4096;
    const row_bytes = N * 4;
    var arena_mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&arena_mem, M * row_bytes, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(arena_mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem, 0), "cuMemHostGetDevicePointer");
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const picks = try ha.alloc(i32, M * S);
    for (0..M) |row| {
        var seen: [S]i32 = undefined;
        var k: usize = 0;
        while (k < S) {
            const e: i32 = @intCast(rng.uintLessThan(usize, E));
            if (std.mem.indexOfScalar(i32, seen[0..k], e) != null) continue;
            seen[k] = e;
            picks[row * S + k] = e;
            k += 1;
        }
    }
    const pk = try upload(d, std.mem.sliceAsBytes(picks));
    try keep.append(ha, pk);
    const outs = try upload(d, try f32s(M * S * N, rng, ha));
    try keep.append(ha, outs);
    const ydev = try cuda.DeviceBuffer.alloc(d, M * row_bytes);
    try keep.append(ha, ydev);
    var sc: mx.ExpertScratch = undefined;
    sc.slots = S;
    sc.out = outs.ptr;
    // the reference: one launch over every row
    try mx.combine(r.k, r.s, sc, pk.ptr, M, E, N, ydev.ptr, N, false);
    try r.s.synchronize();
    const ref = try get(ydev.ptr, M * row_bytes, d);
    defer free(ref);
    // a node of the second pair, its ranks in an order that is not the identity
    const pl = dsv41.prompt2d.ownerPlan(1, .{ 2, 0, 3, 1 }, M / 4);
    var step: CombineStep = .{ .r = r, .y = 0, .ref = ref, .cf = undefined, .row_bytes = row_bytes };
    const cf: mx.CombineFirst = .{ .rows = M / 4, .first = pl.send1, .last = pl.recv1, .ctx = &step, .between = CombineStep.between };
    step.cf = cf;
    const names = [_][]const u8{ "device memory", "the arena's write-combined host memory" };
    for ([_]u64{ ydev.ptr, wc }, 0..) |y, yi| {
        step.y = y;
        step.called = 0;
        step.first_wrong = 0;
        step.last_written = 0;
        try poison(y, M * row_bytes, d, r.s);
        try mx.combineFirst(r.k, r.s, sc, pk.ptr, M, E, N, y, N, false, &cf);
        try r.s.synchronize();
        const got = try get(y, M * row_bytes, d);
        defer free(got);
        r.report(try std.fmt.bufPrint(&line, "roci_combine_first y in {s}: the quarters sent first, a host step, the quarters kept, against one launch of 4,096 rows", .{names[yi]}), differs(got, ref), true);
        r.note(try std.fmt.bufPrint(&line, "roci_combine_first y in {s}: the host step ran once and saw the sent quarters equal to the reference ({d} bytes differ) and the kept ones unwritten ({d} bytes written)", .{ names[yi], step.first_wrong, step.last_written }), step.called == 1 and step.first_wrong == 0 and step.last_written == 0);
    }
    // control: the second kept quarter never launched leaves its rows unwritten, so the bytes differ from the reference
    try poison(wc, M * row_bytes, d, r.s);
    for (cf.first) |row0| try mx.combineRows(r.k, r.s, sc, pk.ptr, row0, cf.rows, E, N, wc, N, false);
    try mx.combineRows(r.k, r.s, sc, pk.ptr, cf.last[0], cf.rows, E, N, wc, N, false);
    try r.s.synchronize();
    const miss = try get(wc, M * row_bytes, d);
    defer free(miss);
    r.report("roci_combine_first control: one kept quarter left out, the arena's bytes", differs(miss, ref), false);
}

// -- the kernel classes on the arena's host memory, in every page type a NIC can register -----------------------------------------

/// Where a measured buffer lives: cudaMalloc, or host memory of one page type, all mapped into the GPU's address space:
/// `arena` cuMemHostAlloc portable|mapped|write-combined (the arena in the baseline, (b)); `cached` the same without write-combined (e);
/// `mmapreg` anonymous pages (2 MiB aligned, huge pages when the kernel gives them), locked, cuMemHostRegister'd mapped|portable
/// (the decode rings' type, (f)); `mmaphuge` the same on MAP_HUGETLB 2 MiB pages only (f2: refused when none are reserved); `vmm` cuMemCreate on the host (a NUMA node, else the host), cuMemMap'd and made accessible to the
/// CPU and the device (g).
const Where = enum { device, arena, cached, mmapreg, mmaphuge, vmm };

/// The driver's virtual memory management entry points, which the runtime's driver table does not hold.
const Vmm = struct {
    lib: std.DynLib,
    create: *const fn (*u64, usize, *const VmmProp, u64) callconv(.c) c_int,
    reserve: *const fn (*u64, usize, usize, u64, u64) callconv(.c) c_int,
    map: *const fn (u64, usize, usize, u64, u64) callconv(.c) c_int,
    set_access: *const fn (u64, usize, [*]const VmmAccess, usize) callconv(.c) c_int,
    gran: *const fn (*usize, *const VmmProp, c_int) callconv(.c) c_int,
    unmap: *const fn (u64, usize) callconv(.c) c_int,
    addr_free: *const fn (u64, usize) callconv(.c) c_int,
    release: *const fn (u64) callconv(.c) c_int,
    dev_attr: *const fn (*c_int, c_int, c_int) callconv(.c) c_int,
    err_name: *const fn (c_int, *[*:0]const u8) callconv(.c) c_int,

    fn open() !Vmm {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.NoLibcuda;
        errdefer lib.close();
        var v: Vmm = undefined;
        v.lib = lib;
        inline for (.{ .{ "create", "cuMemCreate" }, .{ "reserve", "cuMemAddressReserve" }, .{ "map", "cuMemMap" }, .{ "set_access", "cuMemSetAccess" }, .{ "gran", "cuMemGetAllocationGranularity" }, .{ "unmap", "cuMemUnmap" }, .{ "addr_free", "cuMemAddressFree" }, .{ "release", "cuMemRelease" }, .{ "dev_attr", "cuDeviceGetAttribute" }, .{ "err_name", "cuGetErrorName" } }) |p| {
            @field(v, p[0]) = v.lib.lookup(@TypeOf(@field(v, p[0])), p[1]) orelse return error.NoVmmSymbol;
        }
        return v;
    }

    fn name(v: *const Vmm, res: c_int) []const u8 {
        var s: [*:0]const u8 = undefined;
        if (v.err_name(res, &s) != 0) return "?";
        return std.mem.span(s);
    }
};

/// CUmemAllocationProp and CUmemAccessDesc (cuda.h): 32 and 12 bytes.
const VmmProp = extern struct { kind: c_int, handle_types: c_int, loc_type: c_int, loc_id: c_int, win32: ?*anyopaque = null, compression: u8 = 0, rdma: u8 = 0, usage: u16 = 0, reserved: [4]u8 = @splat(0) };
const VmmAccess = extern struct { loc_type: c_int, loc_id: c_int, flags: c_int };

extern "c" fn mlock(addr: *const anyopaque, len: usize) c_int;

/// A block of one page type: the GPU's address of it, the CPU's (what ibv_reg_mr would be given), and what frees it.
const WcBlock = struct { kind: Where, dev: u64, host: ?*anyopaque, len: usize, map: []align(std.heap.page_size_min) u8 = &.{}, vmm_handle: u64 = 0 };

const huge: usize = 2 << 20;

const WcPool = struct {
    d: *const cuda.Driver,
    blocks: std.ArrayList(WcBlock) = .empty,
    vmm: ?Vmm = null,
    /// What the last cuMemCreate attempts said (for the page-type line).
    vmm_note: [320]u8 = undefined,
    vmm_note_len: usize = 0,
    /// How the mmap type got its pages: "2 MiB huge pages (MAP_HUGETLB)", "4 KiB pages with MADV_HUGEPAGE", ...
    mmap_how: []const u8 = "",

    fn noteVmm(p: *WcPool, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(p.vmm_note[p.vmm_note_len..], fmt, args) catch return;
        p.vmm_note_len += s.len;
    }

    fn hostAlloc(p: *WcPool, kind: Where, bytes: usize, flags: c_uint) !WcBlock {
        var mem: ?*anyopaque = null;
        try p.d.check(p.d.api.cuMemHostAlloc(&mem, bytes, flags), "cuMemHostAlloc");
        errdefer _ = p.d.api.cuMemFreeHost(mem);
        var dev: u64 = 0;
        try p.d.check(p.d.api.cuMemHostGetDevicePointer_v2(&dev, mem, 0), "cuMemHostGetDevicePointer");
        return .{ .kind = kind, .dev = dev, .host = mem, .len = bytes };
    }

    fn mmapReg(p: *WcPool, bytes: usize, huge_only: bool) !WcBlock {
        const len = std.mem.alignForward(usize, bytes, huge);
        const prot: std.posix.PROT = .{ .READ = true, .WRITE = true };
        var map: []align(std.heap.page_size_min) u8 = undefined;
        var at: usize = 0;
        if (std.posix.mmap(null, len, prot, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .POPULATE = true, .HUGETLB = true }, -1, 0)) |m| {
            map = m;
            p.mmap_how = "2 MiB huge pages (MAP_HUGETLB)";
        } else |err| {
            if (huge_only) {
                p.mmap_how = @errorName(err);
                return error.HugePagesRefused;
            }
            map = try std.posix.mmap(null, len + huge, prot, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .POPULATE = true }, -1, 0);
            at = std.mem.alignForward(usize, @intFromPtr(map.ptr), huge) - @intFromPtr(map.ptr);
            std.posix.madvise(@alignCast(map.ptr + at), len, std.posix.MADV.HUGEPAGE) catch {};
            p.mmap_how = "4 KiB pages, 2 MiB aligned, MADV_HUGEPAGE (MAP_HUGETLB refused)";
        }
        errdefer std.posix.munmap(map);
        const base = map.ptr + at;
        @memset(base[0..len], 0);
        _ = mlock(base, len);
        var hr = try dsv41.rdma.HostReg.open();
        defer hr.lib.close();
        try p.d.check(hr.register(base, len, 0x01 | 0x02), "cuMemHostRegister");
        var dev: u64 = 0;
        try p.d.check(p.d.api.cuMemHostGetDevicePointer_v2(&dev, base, 0), "cuMemHostGetDevicePointer");
        return .{ .kind = if (huge_only) .mmaphuge else .mmapreg, .dev = dev, .host = base, .len = len, .map = map };
    }

    fn vmmAlloc(p: *WcPool, bytes: usize) !WcBlock {
        if (p.vmm == null) p.vmm = try Vmm.open();
        const v = &p.vmm.?;
        var numa: c_int = 0;
        _ = v.dev_attr(&numa, 134, 0); // CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
        var supported: c_int = 0;
        _ = v.dev_attr(&supported, 141, 0); // ..._HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
        p.vmm_note_len = 0;
        p.noteVmm("device attribute HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED {d}, host NUMA id {d}", .{ supported, numa });
        // the NUMA location first (the docs' EGM recipe), then the plain host location
        const tries = [_]struct { loc: c_int, id: c_int, label: []const u8 }{ .{ .loc = 3, .id = if (numa < 0) 0 else numa, .label = "HOST_NUMA" }, .{ .loc = 2, .id = 0, .label = "HOST" } };
        for (tries) |t| {
            var prop: VmmProp = .{ .kind = 1, .handle_types = 0, .loc_type = t.loc, .loc_id = t.id };
            var g: usize = 0;
            var res = v.gran(&g, &prop, 0);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemGetAllocationGranularity {s}", .{ t.label, v.name(res) });
                continue;
            }
            const len = std.mem.alignForward(usize, bytes, @max(g, 1));
            var h: u64 = 0;
            res = v.create(&h, len, &prop, 0);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemCreate {s}", .{ t.label, v.name(res) });
                continue;
            }
            var va: u64 = 0;
            res = v.reserve(&va, len, 0, 0, 0);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemAddressReserve {s}", .{ t.label, v.name(res) });
                _ = v.release(h);
                continue;
            }
            res = v.map(va, len, 0, h, 0);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemMap {s}", .{ t.label, v.name(res) });
                _ = v.addr_free(va, len);
                _ = v.release(h);
                continue;
            }
            const acc = [2]VmmAccess{ .{ .loc_type = t.loc, .loc_id = t.id, .flags = 3 }, .{ .loc_type = 1, .loc_id = 0, .flags = 3 } };
            res = v.set_access(va, len, &acc, 2);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemSetAccess (CPU + device read/write) {s}", .{ t.label, v.name(res) });
                _ = v.unmap(va, len);
                _ = v.addr_free(va, len);
                _ = v.release(h);
                continue;
            }
            p.noteVmm("; created with the {s} location (granularity {d} B)", .{ t.label, g });
            return .{ .kind = .vmm, .dev = va, .host = @ptrFromInt(va), .len = len, .vmm_handle = h };
        }
        return error.VmmRefused;
    }

    fn alloc(p: *WcPool, kind: Where, bytes: usize) !WcBlock {
        const b: WcBlock = switch (kind) {
            .arena => try p.hostAlloc(.arena, bytes, 0x01 | 0x02 | 0x04),
            .cached => try p.hostAlloc(.cached, bytes, 0x01 | 0x02),
            .mmapreg => try p.mmapReg(bytes, false),
            .mmaphuge => try p.mmapReg(bytes, true),
            .vmm => try p.vmmAlloc(bytes),
            .device => unreachable,
        };
        try p.blocks.append(std.heap.page_allocator, b);
        return b;
    }

    fn release(p: *WcPool, b: WcBlock) void {
        switch (b.kind) {
            .arena, .cached => _ = p.d.api.cuMemFreeHost(b.host),
            .mmapreg, .mmaphuge => {
                if (dsv41.rdma.HostReg.open()) |hr_| {
                    var hr = hr_;
                    _ = hr.unregister(b.host);
                    hr.lib.close();
                } else |_| {}
                std.posix.munmap(b.map);
            },
            .vmm => {
                const v = &p.vmm.?;
                _ = v.unmap(b.dev, b.len);
                _ = v.addr_free(b.dev, b.len);
                _ = v.release(b.vmm_handle);
            },
            .device => {},
        }
    }

    /// Frees every block still held.
    fn freeAll(p: *WcPool) void {
        for (p.blocks.items) |b| p.release(b);
        p.blocks.clearRetainingCapacity();
    }

    fn deinit(p: *WcPool) void {
        p.freeAll();
        p.blocks.deinit(std.heap.page_allocator);
        if (p.vmm) |*v| v.lib.close();
    }
};

/// A variant of a class: where its buffers live and what the line calls it.
const WcVar = struct { w: Where, label: []const u8 };

/// The class measurements' shared state: the arena pool, the device buffers kept, the allocator for host-side inputs, and the
/// variants to run (cudaMalloc, the null control, then every page type this machine allocates).
const WcEnv = struct {
    r: *Run,
    pool: *WcPool,
    keep: *std.ArrayList(cuda.DeviceBuffer),
    ha: std.mem.Allocator,
    vars: [7]WcVar,
    nvars: usize,

    fn list(e: *const WcEnv) []const WcVar {
        return e.vars[0..e.nvars];
    }

    fn buf(e: WcEnv, w: Where, bytes: usize) !u64 {
        if (w != .device) return (try e.pool.alloc(w, bytes)).dev;
        const b = try cuda.DeviceBuffer.alloc(e.r.d, bytes);
        try e.keep.append(e.ha, b);
        return b.ptr;
    }

    fn up(e: WcEnv, bytes: []const u8) !u64 {
        const b = try upload(e.r.d, bytes);
        try e.keep.append(e.ha, b);
        return b.ptr;
    }
};

const wc_reps: usize = 20;
/// The launches of a kernel class in a 4,096-row pass: one a layer of the model's 40, the ceiling (the owner path carries
/// layers 0-19, so about half of it); the hc mixes are two a layer.
const wc_layers: usize = 40;
/// Under this many ms a launch a kernel is launch-bound: its launches are repeated until a measurement lasts about 20 ms, and a
/// null control still outside 2 % is reported as too short to time rather than as wrong.
const wc_short_ms: f32 = 0.05;
/// A null control outside 2 % of a long kernel but within this many % is NOISY (printed, not failed); beyond it the line fails.
const wc_noisy_pct: f32 = 5.0;

/// A line of text built in pieces.
const WcText = struct {
    buf: [2048]u8 = undefined,
    n: usize = 0,

    fn add(t: *WcText, comptime fmt: []const u8, args: anytype) !void {
        t.n += (try std.fmt.bufPrint(t.buf[t.n..], fmt, args)).len;
    }
};

/// The variants' times of a class.
const WcRes = struct {
    labels: [8][]const u8 = undefined,
    ms: [8]f32 = undefined,
    n: usize = 0,

    fn add(s: *WcRes, label: []const u8, ms: f32) void {
        s.labels[s.n] = label;
        s.ms[s.n] = ms;
        s.n += 1;
    }
};

/// ms a launch: the best of three measurements of `reps` launches (kernelMs), `reps` raised for a launch-bound kernel so that a
/// measurement lasts about 20 ms.
fn wcTime(r: *Run, comptime Ctx: type, ctx: Ctx, comptime f: fn (Ctx) anyerror!void) !f32 {
    var reps: usize = wc_reps;
    var best = try kernelMs(r, Ctx, ctx, f, reps);
    if (best < wc_short_ms * 2) {
        reps = @min(4000, @max(reps, @as(usize, @intFromFloat(20.0 / @max(best, 0.001)))));
        best = try kernelMs(r, Ctx, ctx, f, reps);
    }
    for (0..2) |_| best = @min(best, try kernelMs(r, Ctx, ctx, f, reps));
    return best;
}

/// One line for a class: ms a launch of each variant (variant 0 is cudaMalloc, variant 1 a second cudaMalloc set: the null
/// control, which must read within 2 % of variant 0; the others are the page types), each page type over variant 0, and ms a
fn wcLine(r: *Run, name: []const u8, res: *const WcRes, launches: usize) !void {
    const ms = res.ms[0..res.n];
    const labels = res.labels[0..res.n];
    var t: WcText = .{};
    try t.add("roci_wc_classes {s}: ms a launch", .{name});
    for (labels, ms) |l, v| try t.add(", {s} {d:.4}", .{ l, v });
    const null_pct = (ms[1] / ms[0] - 1.0) * 100.0;
    const vd = dsv41.wc_verdict.verdict(null_pct, ms[0]);
    const short = vd == .too_short;
    const noisy = vd == .noisy or vd == .noisy_short;
    if (short) {
        try t.add("; null control {d:.1} % on a kernel under {d:.0} us: TOO SHORT TO TIME, the percentages below are launch noise", .{ null_pct, wc_short_ms * 1000 });
    } else if (noisy) {
        try t.add("; null control {d:.1} %: NOISY (outside 2 %, within {d:.0} %{s}), read the percentages below with that error", .{ null_pct, if (vd == .noisy_short) dsv41.wc_verdict.short_noisy_pct else wc_noisy_pct, if (vd == .noisy_short) ", a kernel under 100 us" else "" });
    } else try t.add("; null control {d:.1} %", .{null_pct});
    for (labels[2..], ms[2..]) |l, v| try t.add("; {s} over cudaMalloc {d:.1} %", .{ l, (v / ms[0] - 1.0) * 100.0 });
    try t.add("; ms a 4,096-row pass at {d} launches:", .{launches});
    for (labels, ms) |l, v| try t.add(" {s} {d:.2},", .{ l, v * @as(f32, @floatFromInt(launches)) });
}

/// mx.quant of bf16 rows into each variant planar, and (the arena type of the baseline) in records: a row's `k` codes then its scales as
/// floats in one record of `rec` bytes (the scales at +k, lds = rec / 4 floats).
fn wcActQuant(r: *Run, env: WcEnv, name: []const u8, rng: std.Random, rows: usize, k: usize, rec: usize) !void {
    const x = try env.up(try bf16s(rows * k, rng, 1.0, env.ha));
    const C = struct {
        r: *Run,
        x: u64,
        rows: usize,
        k: usize,
        q: u64,
        ldq: usize,
        s: u64,
        lds: usize,
        fn go(c: @This()) anyerror!void {
            try mx.quant(c.r.k, c.r.s, c.x, .bf16, c.k, c.rows, c.k, c.q, c.ldq, c.s, c.lds);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{ .r = r, .x = x, .rows = rows, .k = k, .q = try env.buf(vr.w, rows * k), .ldq = k, .s = try env.buf(vr.w, rows * (k / 32) * 4), .lds = k / 32 };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    const base = try env.buf(.arena, rows * rec);
    const c: C = .{ .r = r, .x = x, .rows = rows, .k = k, .q = base, .ldq = rec, .s = base + k, .lds = rec / 4 };
    res.add("(c) WC records", try wcTime(r, C, c, C.go));
    try wcLine(r, name, &res, wc_layers);
}

fn wcActQuantWob(r: *Run, env: WcEnv, rng: std.Random) !void {
    // wo_b's input a node sends: 4,096 rows of 2,048 codes (+ 64 scales: a 2,304-byte record)
    try wcActQuant(r, env, "act_quant of wo_b's input (4,096 rows x 2,048)", rng, 4096, 2048, 2304);
}

fn wcActQuantPack(r: *Run, env: WcEnv, rng: std.Random) !void {
    // the experts' pack: 28,672 rows (4,096 x 7 slots) of 640 codes (+ 20 scales: a 720-byte record)
    try wcActQuant(r, env, "act_quant of the experts' pack (28,672 rows x 640)", rng, 4096 * 7, 640, 720);
}

/// _hc_mix_pf_2b then _hc_finish (tri_hc.hcPrePf, the owner path's attention-side mix): h_out (the streams) and the normalised
/// rows `x` written into each variant.
fn wcHcPre(r: *Run, env: WcEnv, rng: std.Random) !void {
    if (r.t.extra == null) return r.note("roci_wc_classes hcPrePf: the kit's extra set (aot-extra/) loads", false);
    const D: usize = 5120;
    const W: usize = D / 2;
    const rows: usize = 4096;
    const hc_blocks = dsv41.tri_basic.hc_blocks;
    const blk = try env.up(std.mem.sliceAsBytes(try rociBlocks(rows, W, rng, env.ha)));
    const h = try env.up(try bf16s(rows * 4 * D, rng, 1.0, env.ha));
    const fnw = try env.up(try f32s(24 * 4 * D, rng, env.ha));
    const scale = try env.up(try f32s(3, rng, env.ha));
    const base = try env.up(try f32s(24, rng, env.ha));
    const pre_in = try env.up(try f32s(rows * 4, rng, env.ha));
    const nw = try env.up(try bf16s(D, rng, 1.0, env.ha));
    const C = struct {
        r: *Run,
        h: u64,
        blk: u64,
        fnw: u64,
        scale: u64,
        base: u64,
        pre_in: u64,
        nw: u64,
        x: u64,
        h_out: u64,
        pre_out: u64,
        post: u64,
        comb: u64,
        part: u64,
        fn go(c: @This()) anyerror!void {
            _ = try dsv41.tri_hc.hcPrePf(c.r.t, c.h, c.fnw, c.scale, c.base, c.pre_in, c.nw, 1e-20, 1e-6, 20, c.x, c.pre_out, c.post, c.comb, c.part, .{ .gathered = c.blk, .world = 2, .h_out = c.h_out, .blk_rs = rows * W }, rows, D);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{
            .r = r,
            .h = h,
            .blk = blk,
            .fnw = fnw,
            .scale = scale,
            .base = base,
            .pre_in = pre_in,
            .nw = nw,
            .x = try env.buf(vr.w, rows * D * 2),
            .h_out = try env.buf(vr.w, rows * 4 * D * 2),
            .pre_out = try env.buf(.device, rows * 4 * 4),
            .post = try env.buf(.device, rows * 4 * 4),
            .comb = try env.buf(.device, rows * 16 * 4),
            .part = try env.buf(.device, rows * hc_blocks * 32 * 4),
        };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    try wcLine(r, "hcPrePf into x and the streams (4,096 rows x 4 x 5,120)", &res, 2 * wc_layers);
}

/// tri_norm.route (the router's top-6 and the shared expert, one warp a row): pick and wts into each variant.
fn wcRoute(r: *Run, env: WcEnv, rng: std.Random) !void {
    const rows: usize = 4096;
    const E: usize = 384;
    const logits = try env.up(try f32s(rows * E, rng, env.ha));
    const bias = try env.up(try f32s(E, rng, env.ha));
    const C = struct {
        r: *Run,
        logits: u64,
        bias: u64,
        pick: u64,
        wts: u64,
        fn go(c: @This()) anyerror!void {
            try dsv41.tri_norm.route(c.r.t, c.logits, 0, c.bias, 6, 1.5, E, c.pick, c.wts, rows, E, 7);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{ .r = r, .logits = logits, .bias = bias, .pick = try env.buf(vr.w, rows * 7 * 4), .wts = try env.buf(vr.w, rows * 7 * 4) };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    try wcLine(r, "ffnRoute into pick and wts (4,096 rows)", &res, wc_layers);
}

/// tri_basic.rmsnorm of the q_a projection (1,280 columns) into qr, into each variant.
fn wcRmsnorm(r: *Run, env: WcEnv, rng: std.Random) !void {
    const rows: usize = 4096;
    const Dq: usize = 1280;
    const x = try env.up(try bf16s(rows * Dq, rng, 1.0, env.ha));
    const w_ = try env.up(try bf16s(Dq, rng, 1.0, env.ha));
    const C = struct {
        r: *Run,
        x: u64,
        w: u64,
        out: u64,
        fn go(c: @This()) anyerror!void {
            try dsv41.tri_basic.rmsnorm(c.r.t, c.x, Dq, c.w, c.out, Dq, 1e-20, rows, Dq);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{ .r = r, .x = x, .w = w_, .out = try env.buf(vr.w, rows * Dq * 2) };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    try wcLine(r, "rmsnorm into qr (4,096 rows x 1,280)", &res, wc_layers);
}

/// The experts' consumers of codes: gate / up from the activation codes q / s (the loader reads them by erow) and down from the
/// intermediate's codes hq / hs, the codes in each variant (planar). The codes are made once by the quantising calls and the timed
/// calls only read them (mx.gateUpCodes, mx.downCodes; down's closing tf_dsn_combine reads and writes cudaMalloc in every variant).
fn wcExperts(r: *Run, env: WcEnv, rng: std.Random, comptime down: bool) !void {
    const R: usize = 4096;
    const P = R * X_SL;
    const ex = try xExperts(r.d, rng, env.ha, env.keep);
    const x = try env.up(try bf16s(R * X_D, rng, 0.5, env.ha));
    const hdn = try env.up(try bf16s(P * X_I, rng, 0.5, env.ha));
    const pw = try xPicks(R, rng, env.ha);
    const pb = try env.up(std.mem.sliceAsBytes(pw.pick));
    const wb = try env.up(std.mem.sliceAsBytes(pw.wts));
    const hb = try env.buf(.device, P * X_W * 2);
    const ob = try env.buf(.device, R * X_N * 4);
    const base = try xScratch(r.d, R, true, env.ha, env.keep);
    const C = struct {
        r: *Run,
        ex: weights.Experts,
        sc: mx.ExpertScratch,
        pk: u64,
        wt: u64,
        hb: u64,
        ob: u64,
        fn go(c: @This()) anyerror!void {
            if (down) try mx.downCodes(c.r.k, c.r.s, c.ex, c.sc, c.pk, R, c.ob, X_N, false) else try mx.gateUpCodes(c.r.k, c.r.s, c.r.d, c.ex, c.sc, c.pk, c.wt, R, 7.0, c.hb, X_MAIN);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        var sc = base;
        if (down) {
            sc.hq = try env.buf(vr.w, P * X_I);
            sc.hs = try env.buf(vr.w, P * (X_I / 32) * 4);
        } else {
            sc.q = try env.buf(vr.w, R * X_D);
            sc.s = try env.buf(vr.w, R * (X_D / 32) * 4);
        }
        // the codes: the quantising call writes them where this variant keeps them
        if (down) try mx.down(r.k, r.s, ex, sc, hdn, X_I, pb, R, ob, X_N, false, false) else try mx.gateUp(r.k, r.s, r.d, ex, sc, x, X_D, pb, wb, R, 7.0, hb, X_MAIN, false);
        try r.s.synchronize();
        const c: C = .{ .r = r, .ex = ex, .sc = sc, .pk = pb, .wt = wb, .hb = hb, .ob = ob };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    if (down) {
        try wcLine(r, "sf_dn / pf_dn from the intermediate's codes (4,096 rows x 7 slots)", &res, wc_layers);
    } else try wcLine(r, "pf_gu from the activation codes by erow (4,096 rows)", &res, wc_layers);
}

fn wcPfGu(r: *Run, env: WcEnv, rng: std.Random) !void {
    return wcExperts(r, env, rng, false);
}

fn wcPfDn(r: *Run, env: WcEnv, rng: std.Random) !void {
    return wcExperts(r, env, rng, true);
}

/// The block-scaled exact GEMM (sf_split128 family, wo_b's shape: K 2,048, N 2,560, 4,096 rows) reading its codes and scales from
/// each variant planar, and from the arena type of the baseline in 2,304-byte records.
fn wcGemmCodes(r: *Run, env: WcEnv, rng: std.Random) !void {
    const K: usize = 2048;
    const N: usize = 2560;
    const M: usize = 4096;
    if (r.k.sfk == null) return r.note("roci_wc_classes sf GEMM: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    const l = try fp8Linear(r.d, N, K, .fast, rng, env.ha, env.keep);
    const x = try env.up(try bf16s(M * K, rng, 0.5, env.ha));
    const z = try env.buf(.device, @max(mx.zFloats(l, M), 1) * 4);
    const y = try env.buf(.device, M * N * 4);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    const C = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        ldq: usize,
        s: u64,
        lds: usize,
        z: u64,
        y: u64,
        fn go(c: @This()) anyerror!void {
            try mx.gemm(c.r.k, c.r.s, c.l, c.q, c.ldq, c.s, c.lds, M, c.y, N, .f32, false, c.z);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{ .r = r, .l = l, .q = try env.buf(vr.w, M * K), .ldq = K, .s = try env.buf(vr.w, M * (K / 32) * 4), .lds = K / 32, .z = z, .y = y };
        // the codes where this variant keeps them (the quantising call writes the layout it reads)
        try mx.quant(r.k, r.s, x, .bf16, K, M, K, c.q, c.ldq, c.s, c.lds);
        try r.s.synchronize();
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    const b = try env.buf(.arena, M * 2304);
    const c: C = .{ .r = r, .l = l, .q = b, .ldq = 2304, .s = b + K, .lds = 2304 / 4, .z = z, .y = y };
    try mx.quant(r.k, r.s, x, .bf16, K, M, K, c.q, c.ldq, c.s, c.lds);
    try r.s.synchronize();
    res.add("(c) WC records", try wcTime(r, C, c, C.go));
    try wcLine(r, "sf_split128 / wo_b GEMM from the codes (4,096 rows, K 2,048, N 2,560)", &res, wc_layers);
}

/// tri_attn.sparseAttn (the base set's _sparse_attn_part: 4,096 rows, 16 heads, 512 picks over 262,144 compressed rows, one split)
/// with the picks' index rows (cidx) in each variant.
fn wcSparseAttn(r: *Run, env: WcEnv, rng: std.Random) !void {
    const H: usize = 16;
    const HD: usize = 512;
    const WIN: usize = 128;
    const end: usize = 1 << 20;
    const ncomp: usize = end / 4;
    const R: usize = 4096;
    const n_idx: usize = 512;
    const start = end - R;
    if (tri_attn.splits(R, false) > 1) return r.note("roci_wc_classes sparseAttn: 4,096 rows take one split", false);
    const codes = try env.ha.alloc(u8, ncomp * HD / 2);
    rng.bytes(codes);
    const scales = try env.ha.alloc(u8, ncomp * HD / 16);
    for (scales) |*s| s.* = 0x18 + rng.uintLessThan(u8, 0x3f - 0x18);
    const cb = try env.up(codes);
    const sb = try env.up(scales);
    const q = try env.up(try bf16s(R * H * HD, rng, 0.5, env.ha));
    const wsrc = try env.up(try bf16s((R + WIN) * HD, rng, 0.5, env.ha));
    const idx = try env.ha.alloc(i64, R * n_idx);
    for (0..R) |i| for (0..n_idx) |j| {
        idx[i * n_idx + j] = if ((j + i) % 29 == 7) -1 else @intCast(rng.uintLessThan(usize, (start + i + 1) / 4));
    };
    const pos = try env.ha.alloc(i64, R);
    for (pos, 0..) |*p, i| p.* = @intCast(start + i);
    const pb = try env.up(std.mem.sliceAsBytes(pos));
    var wlo = [1]i64{@intCast(start - WIN)};
    const wb = try env.up(std.mem.sliceAsBytes(&wlo));
    const sink = try env.up(try f32s(H, rng, env.ha));
    const out = try env.buf(.device, R * H * HD * 2);
    const C = struct {
        r: *Run,
        a: tri_attn.Attn,
        fn go(c: @This()) anyerror!void {
            try tri_attn.sparseAttn(c.r.t, c.a);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const ib = try env.buf(vr.w, R * n_idx * 8);
        const a: tri_attn.Attn = .{ .q = q, .out = out, .rows = R, .h = H, .hd = HD, .sink = sink, .wsrc = wsrc, .wsrc_rows = R + WIN, .wlo = wb, .ring = false, .comp = .{ .fp4 = .{ .codes = cb, .scales = sb } }, .idx = ib, .n_idx = n_idx, .pos = pb, .scale = dsv41.prompt.scale(HD), .window = WIN, .parts = .{ .pm = 0, .pl = 0, .po = 0 } };
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .a = a }, C.go));
    }
    try wcLine(r, "sparseAttn reading cidx (4,096 rows, 512 picks)", &res, wc_layers);
}

/// Points the staged (whole-line) GEMM kernels at a variant's y: the arena types take the _st kernels (the engine's choice for a y in the
/// arena), cudaMalloc takes the baseline.
fn wcStaged(r: *Run, w: Where, y: u64, bytes: usize) !void {
    if (w == .device) r.k.wc = .{ 0, 0 } else try r.k.useStaged(y, y + bytes);
}

/// wo_b's block-scaled exact GEMM (sf_split128_exact; _st into the arena), K 4,096, N 2,560, 4,096 rows, its fp32 y (psum) in each variant.
fn wcWobPsum(r: *Run, env: WcEnv, rng: std.Random) !void {
    const K: usize = 4096;
    const N: usize = 2560;
    const M: usize = 4096;
    if (r.k.sfk == null) return r.note("roci_wc_classes wo_b into psum: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    const l = try fp8Linear(r.d, N, K, .fast, rng, env.ha, env.keep);
    const x = try env.up(try bf16s(M * K, rng, 0.5, env.ha));
    const q = try env.buf(.device, M * K);
    const s = try env.buf(.device, M * (K / 32) * 4);
    const z = try env.buf(.device, @max(mx.zFloats(l, M), 1) * 4);
    try mx.quant(r.k, r.s, x, .bf16, K, M, K, q, K, s, K / 32);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer {
        r.k.sf = .off;
        r.k.wc = .{ 0, 0 };
    }
    const C = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        z: u64,
        y: u64,
        fn go(c: @This()) anyerror!void {
            try mx.gemm(c.r.k, c.r.s, c.l, c.q, K, c.s, K / 32, M, c.y, N, .f32, false, c.z);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const y = try env.buf(vr.w, M * N * 4);
        try wcStaged(r, vr.w, y, M * N * 4);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .l = l, .q = q, .s = s, .z = z, .y = y }, C.go));
    }
    try wcLine(r, "wo_b sf_split128_exact(_st) into psum (4,096 rows, K 4,096, N 2,560)", &res, wc_layers);
}

/// tf_dsn_combine (7 slots, 2,560 columns, 4,096 rows) into pm, in each variant; the entries' outputs are cudaMalloc.
fn wcCombinePm(r: *Run, env: WcEnv, rng: std.Random) !void {
    const M: usize = 4096;
    const S: usize = 7;
    const E: usize = 384;
    const N: usize = 2560;
    const picks = try env.ha.alloc(i32, M * S);
    for (0..M) |row| {
        var seen: [S]i32 = undefined;
        var n: usize = 0;
        while (n < S) {
            const e: i32 = @intCast(rng.uintLessThan(usize, E));
            if (std.mem.indexOfScalar(i32, seen[0..n], e) != null) continue;
            seen[n] = e;
            picks[row * S + n] = e;
            n += 1;
        }
    }
    const pk = try env.up(std.mem.sliceAsBytes(picks));
    const outs = try env.buf(.device, M * S * N * 4);
    try r.d.check(r.d.api.cuMemsetD32Async(outs, 0x3f800000, M * S * N, r.s.handle), "cuMemsetD32Async");
    var sc: mx.ExpertScratch = undefined;
    sc.slots = S;
    sc.out = outs;
    const C = struct {
        r: *Run,
        sc: mx.ExpertScratch,
        pk: u64,
        y: u64,
        fn go(c: @This()) anyerror!void {
            try mx.combine(c.r.k, c.r.s, c.sc, c.pk, M, E, N, c.y, N, false);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const y = try env.buf(vr.w, M * N * 4);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .sc = sc, .pk = pk, .y = y }, C.go));
    }
    try wcLine(r, "combine (7 slots) into pm (4,096 rows x 2,560)", &res, wc_layers);
}

/// Engram's projection (a dense FP8 GEMM, K 5,120, N 12,800, 4,096 rows; the fold / tile configuration, _st into the arena) into ek.
fn wcPfGemmEk(r: *Run, env: WcEnv, rng: std.Random) !void {
    const K: usize = 5120;
    const N: usize = 12800;
    const M: usize = 4096;
    const l = try fp8Linear(r.d, N, K, .fast, rng, env.ha, env.keep);
    const x = try env.up(try bf16s(M * K, rng, 0.5, env.ha));
    const q = try env.buf(.device, M * K);
    const s = try env.buf(.device, M * (K / 32) * 4);
    const z = try env.buf(.device, @max(mx.zFloats(l, M), 1) * 4);
    try mx.quant(r.k, r.s, x, .bf16, K, M, K, q, K, s, K / 32);
    r.k.sf = .off;
    defer r.k.wc = .{ 0, 0 };
    const C = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        z: u64,
        y: u64,
        fn go(c: @This()) anyerror!void {
            try mx.gemm(c.r.k, c.r.s, c.l, c.q, K, c.s, K / 32, M, c.y, N, .f32, false, c.z);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const y = try env.buf(vr.w, M * N * 4);
        try wcStaged(r, vr.w, y, M * N * 4);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .l = l, .q = q, .s = s, .z = z, .y = y }, C.go));
    }
    try wcLine(r, "pf_gemm(_st) into ek (4,096 rows, K 5,120, N 12,800)", &res, 2);
}

/// add2_bf16 of an owner sum's part 2 (one quarter, 1,024 x 2,560): this node's fp32 partial plus the landed stage block, rounded to
/// bf16 into the block, all three in each variant.
fn wcAdd2(r: *Run, env: WcEnv, rng: std.Random) !void {
    const ops = r.ops orelse return r.note("roci_wc_classes add2: the ops are available", false);
    const n: usize = 1024 * 2560;
    const a_host = try f32s(n, rng, env.ha);
    const b_host = try f32s(n, rng, env.ha);
    const C = struct {
        ops: *const dsv41.ops.Ops,
        s: cuda.Stream,
        a: u64,
        b: u64,
        out: u64,
        fn go(c: @This()) anyerror!void {
            try c.ops.add2Bf16(c.s, c.a, c.b, c.out, n);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const a = try env.buf(vr.w, n * 4);
        const b = try env.buf(vr.w, n * 4);
        const out = try env.buf(vr.w, n * 2);
        res.add(vr.label, try wcTime(r, C, .{ .ops = ops, .s = r.s, .a = a, .b = b, .out = out }, C.go));
    }
    try wcLine(r, "add2_bf16 reading the partial and the landed stage block, into the sum block (1,024 x 2,560)", &res, 83);
    try wcAdd2Vec(r, env, ops, a_host, b_host, n);
}

/// TF_DS_ADD2_VEC: the four-wide add2 (tf_ds_add2_bf16_v4_kernel) against the scalar kernel in every variant: every output byte equal, and the ms of
/// each (the page cost of the vector loads), then TF_DS_STAGE_WC's question: the partial in locked pages and the landed stage block in write-combined
/// host memory, against both in locked pages (the arena's kind in the baseline), with the scalar kernel.
fn wcAdd2Vec(r: *Run, env: WcEnv, ops: *const dsv41.ops.Ops, a_host: []const u8, b_host: []const u8, n: usize) !void {
    const C = struct {
        ops: *const dsv41.ops.Ops,
        s: cuda.Stream,
        a: u64,
        b: u64,
        out: u64,
        vec: bool,
        n: usize,
        fn go(c: @This()) anyerror!void {
            dsv41.ops.Ops.add2_vec_on = c.vec;
            defer dsv41.ops.Ops.add2_vec_on = true;
            try c.ops.add2Bf16(c.s, c.a, c.b, c.out, c.n);
        }
    };
    if (ops.add2_v4 == null) {
        r.note("add2_vec: tf_ds_add2_bf16_v4_kernel is in this build's module", false);
        r.note("add2_vec page-cost line: not run without the kernel", false);
        r.note("add2_vec stage in write-combined memory: not run without the kernel", false);
        return;
    }
    var t: WcText = .{};
    try t.add("roci_wc_classes add2_vec: ms a launch scalar / four-wide", .{});
    var bad: usize = 0;
    var locked: ?struct { w: Where, a: u64, b: u64 } = null;
    var wc: ?Where = null;
    for (env.list()) |vr| {
        const a = try env.buf(vr.w, n * 4);
        const b = try env.buf(vr.w, n * 4);
        const out_s = try env.buf(vr.w, n * 2);
        const out_v = try env.buf(vr.w, n * 2);
        const ms_s = try wcTime(r, C, .{ .ops = ops, .s = r.s, .a = a, .b = b, .out = out_s, .vec = false, .n = n }, C.go);
        const ms_v = try wcTime(r, C, .{ .ops = ops, .s = r.s, .a = a, .b = b, .out = out_v, .vec = true, .n = n }, C.go);
        try r.s.synchronize();
        const gs = try get(out_s, n * 2, r.d);
        defer free(gs);
        const gv = try get(out_v, n * 2, r.d);
        defer free(gv);
        bad += differs(gs, gv);
        try t.add(", {s} {d:.4} / {d:.4}", .{ vr.label, ms_s, ms_v });
        if (std.mem.indexOf(u8, vr.label, "mmap") != null) locked = .{ .w = vr.w, .a = a, .b = b };
        if (std.mem.indexOf(u8, vr.label, "WC") != null) wc = vr.w;
    }
    r.report("add2_vec: the four-wide kernel's bytes against the scalar kernel's in every page type", bad, true);
    r.note(t.buf[0..t.n], true);
    // stage in write-combined memory, partial in locked pages
    if (locked) |l| if (wc) |ww| {
        const b_wc = try env.buf(ww, n * 4);
        const out = try env.buf(l.w, n * 2);
        const both = try wcTime(r, C, .{ .ops = ops, .s = r.s, .a = l.a, .b = l.b, .out = out, .vec = false, .n = n }, C.go);
        const split = try wcTime(r, C, .{ .ops = ops, .s = r.s, .a = l.a, .b = b_wc, .out = out, .vec = false, .n = n }, C.go);
        var u: WcText = .{};
        try u.add("roci_wc_classes add2 with the partial in locked pages: stage locked {d:.4} ms, stage write-combined {d:.4} ms a launch ({d:.2} against {d:.2} ms a pass at 83 launches)", .{ both, split, both * 83, split * 83 });
        r.note(u.buf[0..u.n], true);
        return;
    };
    r.note("add2_vec stage in write-combined memory: this machine allocates both locked and write-combined pages", false);
}

/// _hc_mix_pf_2b (tri_hc.hcMixPf with the blocks as they land) reading the two bf16 sum blocks from each variant: 4,096 rows x 5,120.
fn wcHcMix2b(r: *Run, env: WcEnv, rng: std.Random) !void {
    if (r.t.extra == null) return r.note("roci_wc_classes _hc_mix_pf_2b: the kit's extra set (aot-extra/) loads", false);
    const D: usize = 5120;
    const W: usize = D / 2;
    const rows: usize = 4096;
    const hc_blocks = dsv41.tri_basic.hc_blocks;
    const blk_host = try bf16s(2 * rows * W, rng, 1.0, env.ha);
    const h = try env.up(try bf16s(rows * 4 * D, rng, 1.0, env.ha));
    const fnw = try env.up(try f32s(24 * 4 * D, rng, env.ha));
    const post = try env.buf(.device, rows * 4 * 4);
    const comb = try env.buf(.device, rows * 16 * 4);
    const part = try env.buf(.device, rows * hc_blocks * 32 * 4);
    const h_out = try env.buf(.device, rows * 4 * D * 2);
    const C = struct {
        r: *Run,
        h: u64,
        fnw: u64,
        blk: u64,
        part: u64,
        h_out: u64,
        post: u64,
        comb: u64,
        fn go(c: @This()) anyerror!void {
            _ = try dsv41.tri_hc.hcMixPf(c.r.t, c.h, c.fnw, c.part, .{ .gathered = c.blk, .world = 2, .h_out = c.h_out, .blk_rs = rows * W }, c.post, c.comb, rows, D);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const blk = try env.buf(vr.w, 2 * rows * W * 2);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .h = h, .fnw = fnw, .blk = blk, .part = part, .h_out = h_out, .post = post, .comb = comb }, C.go));
    }
    try wcLine(r, "_hc_mix_pf_2b reading the landed sum blocks (4,096 rows x 5,120)", &res, 2 * wc_layers);
}

/// _engram_gate_2b reading Engram's two bf16 sum blocks (12,800 columns each) from each variant: 4,096 rows.
fn wcEngramGate2b(r: *Run, env: WcEnv, rng: std.Random) !void {
    if (r.t.extra == null) return r.note("roci_wc_classes _engram_gate_2b: the kit's extra set (aot-extra/) loads", false);
    const D: usize = 5120;
    const EW: usize = 12800;
    const rows: usize = 4096;
    const blk_host = try bf16s(2 * rows * EW, rng, 1.0, env.ha);
    const h = try env.up(try bf16s(rows * 4 * D, rng, 1.0, env.ha));
    const qk = try env.up(try f32s(4 * D, rng, env.ha));
    const out = try env.buf(.device, rows * 4 * D * 2);
    const C = struct {
        r: *Run,
        h: u64,
        blk: u64,
        qk: u64,
        out: u64,
        fn go(c: @This()) anyerror!void {
            try dsv41.tri_basic.engramGate2b(c.r.t, c.h, c.blk, rows * EW, c.qk, c.out, 1e-20, rows, D);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const blk = try env.buf(vr.w, 2 * rows * EW * 2);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .h = h, .blk = blk, .qk = qk, .out = out }, C.go));
    }
    try wcLine(r, "_engram_gate_2b reading the landed sum blocks (4,096 rows, 12,800 columns)", &res, 2);
}

const wc_labels = [_][]const u8{ "(a) cudaMalloc", "null (a) again", "(b) WC hostalloc (the arena)", "(e) cached hostalloc", "(f) mmap+cuMemHostRegister", "(f2) mmap MAP_HUGETLB", "(g) cuMemCreate host" };

/// Each page type this machine allocates (a 64 MiB block): whether it allocates, how, and whether ibv_reg_mr accepts it. With no verbs
/// device to ask (this workstation) the registration answer is the construction's: (b) (e) cudaHostAlloc and (f) mlock'd anonymous
/// pages are ordinary pinned host memory, which the driver registers for DMA; (g) is a CUDA-managed mapping, and CUDA's documentation
/// (Virtual Memory Management, Extended GPU Memory) says nothing about registering its CPU mapping with a NIC.
fn wcPageTypes(r: *Run, pool: *WcPool, avail: *[7]bool) !void {
    var t: WcText = .{};
    try t.add("roci_wc_classes page types (64 MiB blocks)", .{});
    const dev_env = std.c.getenv("TF_DS_RDMA_DEVICES");
    const first: []const u8 = if (dev_env) |v| std.mem.sliceTo(std.mem.span(v), ',') else "mlx5_0";
    var probe: ?dsv41.rocinante_verbs.RegProbe = dsv41.rocinante_verbs.RegProbe.open(first) catch null;
    defer if (probe) |*p| p.close();
    if (probe == null) try t.add("; ibv_reg_mr not probed (no verbs device {s} to open here)", .{first}) else try t.add("; ibv_reg_mr probed on {s}", .{first});
    const kinds = [_]Where{ .arena, .cached, .mmapreg, .mmaphuge, .vmm };
    const names = [_][]const u8{ "(b) WC hostalloc", "(e) cached hostalloc", "(f) mmap+register", "(f2) mmap MAP_HUGETLB", "(g) cuMemCreate host" };
    const construction = [_][]const u8{ "registerable by construction (pinned host memory)", "registerable by construction (pinned host memory)", "registerable by construction (locked anonymous pages, the decode rings' registration)", "registerable by construction (locked huge pages)", "registration of its CPU mapping is not documented by CUDA" };
    for (kinds, names, construction, 0..) |k, nm, why, i| {
        const b = pool.alloc(k, 64 << 20) catch |err| {
            try t.add("; {s}: NOT allocated ({s}{s}{s})", .{ nm, @errorName(err), if (k == .vmm or k == .mmaphuge) ": " else "", if (k == .vmm) pool.vmm_note[0..pool.vmm_note_len] else if (k == .mmaphuge) pool.mmap_how else "" });
            continue;
        };
        avail[i + 2] = true;
        try t.add("; {s}: allocated", .{nm});
        if (k == .mmapreg or k == .mmaphuge) try t.add(" with {s}", .{pool.mmap_how});
        if (k == .vmm) try t.add(" ({s})", .{pool.vmm_note[0..pool.vmm_note_len]});
        if (probe) |*p| {
            if (p.refused(b.host.?, b.len)) |errno| {
                try t.add(", ibv_reg_mr REFUSED (errno {d})", .{errno});
            } else try t.add(", ibv_reg_mr ACCEPTED", .{});
        } else try t.add(", {s}", .{why});
    }
    pool.freeAll();
    r.note(t.buf[0..t.n], avail[2]);
}

/// U8, the gate before the steps that move more producers and consumers to the arena: each kernel class the owner path moves
/// (act_quant in planar and in records, hcPrePf, ffnRoute, rmsnorm, pf_gu and down from the codes, the block-scaled GEMM from the
/// codes, sparseAttn from cidx) at the real shapes of a 4,096-row pass, kernel only (events, best of three runs of 20 launches or
/// more, no readback), into or out of cudaMalloc, a second cudaMalloc set (the null control, within 2 %) and every page type of
/// host memory the machine allocates. This changes no engine path. A class that cannot run says why on its own line.
fn rociWcClasses(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x0c1a55e5);
    const rng = prng.random();
    var pool: WcPool = .{ .d = r.d };
    defer pool.deinit();
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var avail: [7]bool = .{ true, true, false, false, false, false, false };
    try wcPageTypes(r, &pool, &avail);
    var env: WcEnv = .{ .r = r, .pool = &pool, .keep = &keep, .ha = ha, .vars = undefined, .nvars = 0 };
    const wheres = [_]Where{ .device, .device, .arena, .cached, .mmapreg, .mmaphuge, .vmm };
    for (wheres, wc_labels, avail) |w, l, ok| if (ok) {
        env.vars[env.nvars] = .{ .w = w, .label = l };
        env.nvars += 1;
    };
    const classes = [_]struct { name: []const u8, f: *const fn (*Run, WcEnv, std.Random) anyerror!void }{
        .{ .name = "act_quant of wo_b's input", .f = wcActQuantWob },
        .{ .name = "act_quant of the experts' pack", .f = wcActQuantPack },
        .{ .name = "hcPrePf", .f = wcHcPre },
        .{ .name = "ffnRoute", .f = wcRoute },
        .{ .name = "rmsnorm", .f = wcRmsnorm },
        .{ .name = "pf_gu", .f = wcPfGu },
        .{ .name = "pf_dn", .f = wcPfDn },
        .{ .name = "sf GEMM", .f = wcGemmCodes },
        .{ .name = "sparseAttn", .f = wcSparseAttn },
        .{ .name = "wo_b into psum", .f = wcWobPsum },
        .{ .name = "combine into pm", .f = wcCombinePm },
        .{ .name = "pf_gemm into ek", .f = wcPfGemmEk },
        .{ .name = "add2", .f = wcAdd2 },
        .{ .name = "_hc_mix_pf_2b", .f = wcHcMix2b },
        .{ .name = "_engram_gate_2b", .f = wcEngramGate2b },
    };
    for (classes) |cl| {
        const before = r.lines;
        cl.f(r, env, rng) catch |err| {
            var line: [256]u8 = undefined;
            if (r.lines == before) r.note(try std.fmt.bufPrint(&line, "roci_wc_classes {s}: could not run ({s})", .{ cl.name, @errorName(err) }), false);
        };
        // each class's buffers are freed before the next (the device buffers kept and the host blocks)
        for (keep.items) |*b| b.free();
        keep.clearRetainingCapacity();
        pool.freeAll();
    }
    r.note("roci_wc_classes act_quant (d), the record writer through shared memory: no such kernel exists in the engine tree, so it is not measured", true);
}

// -- the cost of a local-first tile order (a consumer starts on the rows this node holds, then waits for the others') --------------

/// `rows` rows' picks as a skewed router makes them: the routed experts in a random fixed order, the one of rank i drawn with
/// weight 1 / (i + 1)^0.8, six distinct a row, the shared expert last.
fn xPicksSkew(rows: usize, rng: std.Random, ha: std.mem.Allocator) !struct { pick: []i32, wts: []f32 } {
    const n = X_E - 1;
    const order = try ha.alloc(i32, n);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    rng.shuffle(i32, order);
    const cdf = try ha.alloc(f64, n);
    var total: f64 = 0;
    for (cdf, 0..) |*c, i| {
        total += std.math.pow(f64, @as(f64, @floatFromInt(i + 1)), -0.8);
        c.* = total;
    }
    const pick = try ha.alloc(i32, rows * X_SL);
    const wts = try ha.alloc(f32, rows * X_SL);
    for (0..rows) |row| {
        var used: [6]i32 = undefined;
        for (0..6) |sl| {
            var e: i32 = undefined;
            while (true) {
                const u = rng.float(f64) * total;
                var i: usize = 0;
                while (i + 1 < n and cdf[i] < u) i += 1;
                e = order[i];
                if (std.mem.indexOfScalar(i32, used[0..sl], e) == null) break;
            }
            used[sl] = e;
            pick[row * X_SL + sl] = e;
            wts[row * X_SL + sl] = rng.float(f32) * 0.5;
        }
        pick[row * X_SL + 6] = X_E - 1;
        wts[row * X_SL + 6] = 1;
    }
    return .{ .pick = pick, .wts = wts };
}

/// One kernel launch ready to repeat: the function, its arguments and grid (256 threads a CTA).
const Lau = struct {
    f: cuda.Function,
    args: cuda.Args,
    gx: usize,
    gy: usize,
    smem: u32,
};

/// Launches `ls` in order on the run's stream; with kernelMs the milliseconds of the whole sequence.
const LauRun = struct {
    r: *Run,
    ls: []const Lau,

    fn go(c: @This()) anyerror!void {
        for (c.ls) |l| {
            var a = l.args;
            try cuda.launch.launch(l.f, .{ .grid = .{ .x = @intCast(l.gx), .y = @intCast(l.gy), .z = 1 }, .block = .{ .x = 256 }, .shared = l.smem }, c.r.s, &a);
        }
    }
};

fn lauMs(r: *Run, ls: []const Lau) !f32 {
    return kernelBest(r, LauRun, .{ .r = r, .ls = ls }, LauRun.go);
}

/// The ints of a device array, read back (memory from `a`).
fn getInts(d: *const cuda.Driver, ptr: u64, n: usize, a: std.mem.Allocator) ![]i32 {
    const v = try a.alloc(i32, n);
    return v;
}

/// The tile list a layout launches: local tiles then rest tiles in one device array, and their two counts in another.
const LfDevice = struct {
    off: u64,
    ent: u64,
    erow: u64,
    tiles: u64,
    counts: u64,
    nlocal: usize,
    nrest: usize,

    fn make(d: *const cuda.Driver, l: dsv41.local_first.Layout, ha: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer)) !LfDevice {
        const both = try ha.alloc([2]i32, l.local.len + l.rest.len);
        @memcpy(both[0..l.local.len], l.local);
        @memcpy(both[l.local.len..], l.rest);
        const cnt = [2]i32{ @intCast(l.local.len), @intCast(l.rest.len) };
        const o = try upload(d, std.mem.sliceAsBytes(l.off));
        const en = try upload(d, std.mem.sliceAsBytes(l.ent));
        const er = try upload(d, std.mem.sliceAsBytes(l.erow));
        const tl = try upload(d, std.mem.sliceAsBytes(both));
        const ct = try upload(d, std.mem.sliceAsBytes(&cnt));
        for ([_]cuda.DeviceBuffer{ o, en, er, tl, ct }) |b| try keep.append(ha, b);
        return .{ .off = o.ptr, .ent = en.ptr, .erow = er.ptr, .tiles = tl.ptr, .counts = ct.ptr, .nlocal = l.local.len, .nrest = l.rest.len };
    }
};

/// The grouped experts' kernel of one phase over one set of operands: gate / up (`down` false) or down, on `h` / `out`.
const LfKernel = struct {
    r: *Run,
    down: bool,
    gu: mx.GuArgs,
    dn: mx.DnArgs,
    f: cuda.Function,
    gx: usize,
    smem: u32,

    /// A launch of `tiles` (device list) counted by *ntiles over grouping arrays (off, ent, erow), `gy` tile rows of CTAs, into `dst`
    /// (the pack h or the entries' outputs).
    fn lau(k: LfKernel, off: u64, ent: u64, erow: u64, tiles: u64, ntiles: u64, gy: usize, dst: u64) Lau {
        var args: cuda.Args = .{};
        if (k.down) {
            var a = k.dn;
            a.off = off;
            a.ent = ent;
            a.out = dst;
            args.add(a);
        } else {
            var a = k.gu;
            a.off = off;
            a.ent = ent;
            a.erow = erow;
            a.h = dst;
            args.add(a);
        }
        args.add(tiles);
        args.add(ntiles);
        return .{ .f = k.f, .args = args, .gx = k.gx, .gy = gy, .smem = k.smem };
    }
};

fn pct(x: f32, base: f32) f32 {
    return (x / base - 1.0) * 100.0;
}

/// What the local-first order costs the grouped expert kernels at a 4,096-row pass's real shapes, kernel only (events, 20 launches
/// averaged after 3): gate / up (tf_dsn_sf_gu_exact) and down (tf_dsn_sf_dn_exact) in one launch of the baseline tile list (a) against
/// the same rows laid out so a tile's rows come from one part of the chunk (dsv41.local_first), the node's own quarter's tiles
/// launched first and the rest in a second launch, with the rest cut into the three other quarters' tiles (four segments) and
/// kept whole (two): tile counts, ms of each launch, the pair against (a), the share of the work in the first launch. Rows 0 ..
/// 1,023 are the node's own. Routing: the replay's uniform draw of six experts a row, and a skewed draw (a stand-in for a router's
/// hot experts, not a capture). Then the attention front's dense GEMM (wq_a) on the gathered rows as one 4,096-row launch against
/// 1,024 then 3,072. Outputs byte for byte, each with a control.
fn rociLocalFirst(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x10ca1f);
    const rng = prng.random();
    const d = r.d;
    var line: [1024]u8 = undefined;
    const sk = r.k.sfk orelse return r.note("roci_local_first: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const R: usize = 4096;
    const P = R * X_SL;
    const quarter = R / 4;
    const ex = try xExperts(d, rng, ha, &keep);
    const nx = ex.native.?;
    const x = try upload(d, try bf16s(R * X_D, rng, 0.5, ha));
    const hdn = try upload(d, try bf16s(P * X_I, rng, 0.5, ha));
    const h_a = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
    const h_n = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
    const h_b = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
    const o_a = try cuda.DeviceBuffer.alloc(d, P * X_N * 4);
    const o_n = try cuda.DeviceBuffer.alloc(d, P * X_N * 4);
    const o_b = try cuda.DeviceBuffer.alloc(d, P * X_N * 4);
    for ([_]cuda.DeviceBuffer{ x, hdn, h_a, h_n, h_b, o_a, o_n, o_b }) |b| try keep.append(ha, b);
    const names = [_][]const u8{ "uniform routing", "skewed routing" };
    for (names, 0..) |rname, ri| {
        var picks: []i32 = undefined;
        var weights_: []f32 = undefined;
        if (ri == 0) {
            const p = try xPicks(R, rng, ha);
            picks = p.pick;
            weights_ = p.wts;
        } else {
            const p = try xPicksSkew(R, rng, ha);
            picks = p.pick;
            weights_ = p.wts;
        }
        const pb = try upload(d, std.mem.sliceAsBytes(picks));
        const wb = try upload(d, std.mem.sliceAsBytes(weights_));
        try keep.append(ha, pb);
        try keep.append(ha, wb);
        const sc = try xScratch(d, R, true, ha, &keep);
        // the codes, the grouping and the baseline tile list, once: what gate / up and down read
        try mx.quant(r.k, r.s, x.ptr, .bf16, X_D, R, X_D, sc.q, X_D, sc.s, X_D / 32);
        try mx.grouping(r.k, r.s, d, X_E, sc, pb.ptr, R);
        try mx.tileList(r.k, r.s, sc);
        try mx.quant(r.k, r.s, hdn.ptr, .bf16, X_I, P, X_I, sc.hq, X_I, sc.hs, X_I / 32);
        try r.s.synchronize();
        const off = try getInts(d, sc.off, X_E + 1, ha);
        const ent = try getInts(d, sc.ent, P, ha);
        const erow = try getInts(d, sc.erow, P, ha);
        const nt_a: usize = @intCast((try getInts(d, sc.ntiles, 1, ha))[0]);
        const want = try dsv41.local_first.todays(ha, off, X_E);
        const lay2 = try dsv41.local_first.segmented(ha, off, ent, erow, X_E, 2, quarter);
        const lay4 = try dsv41.local_first.segmented(ha, off, ent, erow, X_E, 4, quarter);
        const g2 = try LfDevice.make(d, lay2, ha, &keep);
        const g4 = try LfDevice.make(d, lay4, ha, &keep);
        const holds = want.len == nt_a and dsv41.local_first.holds(lay2, off, ent, erow, X_E, 2, quarter) and dsv41.local_first.holds(lay4, off, ent, erow, X_E, 4, quarter);
        const grid_a: usize = P / mx.pf_bm + X_E; // the baseline grid: the most tiles a grouping can make
        for ([_]bool{ false, true }) |down| {
            const kname: []const u8 = if (down) "sf_dn_exact" else "sf_gu_exact";
            const k: LfKernel = .{
                .r = r,
                .down = down,
                .gu = .{ .q = sc.q, .s = sc.s, .ldq = X_D, .lds = X_D / 32, .ent = sc.ent, .erow = sc.erow, .off = sc.off, .act = sc.act, .nact = sc.nact, .w1 = nx.w1, .w3 = nx.w3, .s1 = nx.s1, .s3 = nx.s3, .ldw = X_D / 2, .ldws = X_D / 32, .fmt = 1, .D = X_D, .shared = X_E - 1, .sldw = X_D, .sldws = X_D / 32, .owner = nx.owner, .n_tiles = X_TILES, .rest_tiles = X_REST, .wts = wb.ptr, .limit = 7.0, .h = h_a.ptr, .ldh = X_MAIN, .P = @intCast(P) },
                .dn = .{ .q = sc.hq, .s = sc.hs, .ldq = X_I, .lds = X_I / 32, .ent = sc.ent, .off = sc.off, .act = sc.act, .nact = sc.nact, .w2 = nx.w2, .s2 = nx.s2, .ldw = X_I / 2, .ldws = X_I / 32, .fmt = 1, .shared = X_E - 1, .sldw = X_I, .sldws = X_I / 32, .I = X_I, .N = X_N, .round_bf16 = 0, .out = o_a.ptr, .ldo = X_N },
                .f = if (down) sk.dn[1] else sk.gu[1],
                .gx = if (down) X_N / 128 else X_TILES * 32 / 128,
                .smem = if (down) mx.pf_dn_r128_smem else mx.pf_gu_r128_smem,
            };
            const dst_a: u64 = if (down) o_a.ptr else h_a.ptr;
            const dst_n: u64 = if (down) o_n.ptr else h_n.ptr;
            const dst_b: u64 = if (down) o_b.ptr else h_b.ptr;
            const bytes: usize = if (down) P * X_N * 4 else P * X_W * 2;
            const la = [1]Lau{k.lau(sc.off, sc.ent, sc.erow, sc.tiles, sc.ntiles, grid_a, dst_a)};
            const ln = [1]Lau{k.lau(sc.off, sc.ent, sc.erow, sc.tiles, sc.ntiles, grid_a, dst_n)};
            const lx = [1]Lau{k.lau(sc.off, sc.ent, sc.erow, sc.tiles, sc.ntiles, nt_a, dst_a)};
            const gs = [2]LfDevice{ g2, g4 };
            var pair: [2][2]Lau = undefined;
            for (gs, 0..) |g, gi| {
                pair[gi][0] = k.lau(g.off, g.ent, g.erow, g.tiles, g.counts, g.nlocal, dst_b);
                pair[gi][1] = k.lau(g.off, g.ent, g.erow, g.tiles + g.nlocal * 8, g.counts + 4, g.nrest, dst_b);
            }
            const ms_a = try lauMs(r, &la);
            const ms_n = try lauMs(r, &ln);
            const ms_x = try lauMs(r, &lx);
            var ms_l: [2]f32 = undefined;
            var ms_r: [2]f32 = undefined;
            var ms_p: [2]f32 = undefined;
            for (0..2) |gi| {
                ms_l[gi] = try lauMs(r, pair[gi][0..1]);
                ms_r[gi] = try lauMs(r, pair[gi][1..2]);
                ms_p[gi] = try lauMs(r, &pair[gi]);
            }
            var t: WcText = .{};
            try t.add("roci_local_first {s} {s} (4,096 rows x 7 slots, kernel only): tiles (a) {d}; two parts local {d} + rest {d} = {d}; four quarters local {d} + rest {d} = {d}; ", .{ rname, kname, nt_a, g2.nlocal, g2.nrest, g2.nlocal + g2.nrest, g4.nlocal, g4.nrest, g4.nlocal + g4.nrest });
            try t.add("ms: (a) {d:.3} (null control, (a) again, {d:.1} %: {s}; grid of the exact tile count {d:.1} %); ", .{ ms_a, pct(ms_n, ms_a), resolved(pct(ms_n, ms_a)), pct(ms_x, ms_a) });
            for (0..2) |gi| try t.add("{s} local {d:.3} + rest {d:.3}, back to back {d:.3} = {d:.1} % over (a), the local launch is {d:.0} % of the tiles and {d:.0} % of the ms{s} ", .{ if (gi == 0) "two parts" else "four quarters", ms_l[gi], ms_r[gi], ms_p[gi], pct(ms_p[gi], ms_a), 100.0 * @as(f32, @floatFromInt(gs[gi].nlocal)) / @as(f32, @floatFromInt(gs[gi].nlocal + gs[gi].nrest)), 100.0 * ms_l[gi] / (ms_l[gi] + ms_r[gi]), if (gi == 0) ";" else "." });
            try t.add("The research's kill line (more than 1.5 % slower): two parts {s}, four quarters {s}.", .{ if (pct(ms_p[0], ms_a) > 1.5) "crossed" else "not crossed", if (pct(ms_p[1], ms_a) > 1.5) "crossed" else "not crossed" });
            r.note(t.buf[0..t.n], @abs(pct(ms_n, ms_a)) <= null_bar and holds);
            // the bytes: (a), then each layout's two launches, every output the same
            try poison(dst_a, bytes, d, r.s);
            try LauRun.go(.{ .r = r, .ls = &la });
            try r.s.synchronize();
            const ref = try get(dst_a, bytes, d);
            defer free(ref);
            var bad: [2]usize = undefined;
            for (0..2) |gi| {
                try poison(dst_b, bytes, d, r.s);
                try LauRun.go(.{ .r = r, .ls = &pair[gi] });
                try r.s.synchronize();
                const got = try get(dst_b, bytes, d);
                defer free(got);
                bad[gi] = differs(ref, got);
            }
            r.report(try std.fmt.bufPrint(&line, "roci_local_first {s} {s}: the two-part layout's {s} and the four-quarter layout's output, the local launch then the rest, against the baseline single launch", .{ rname, kname, if (down) "entry outputs" else "pack" }), bad[0] + bad[1], true);
            // control: the last tile of the rest launch never runs
            const cut = [2]Lau{ pair[1][0], blk: {
                var c = pair[1][1];
                c.gy -= 1;
                break :blk c;
            } };
            try poison(dst_b, bytes, d, r.s);
            try LauRun.go(.{ .r = r, .ls = &cut });
            try r.s.synchronize();
            const got = try get(dst_b, bytes, d);
            defer free(got);
            r.report(try std.fmt.bufPrint(&line, "roci_local_first {s} {s} control: the four-quarter layout's last rest tile not launched", .{ rname, kname }), differs(ref, got), false);
        }
    }
    // the attention front's dense GEMM (wq_a: K 5120, N 1280) on the gathered rows
    const K: usize = 5120;
    const N: usize = 1280;
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    const q = try cuda.DeviceBuffer.alloc(d, R * K);
    const qs = try cuda.DeviceBuffer.alloc(d, R * (K / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, R), 1) * 4);
    const y_a = try cuda.DeviceBuffer.alloc(d, R * N * 4);
    const y_n = try cuda.DeviceBuffer.alloc(d, R * N * 4);
    const y_b = try cuda.DeviceBuffer.alloc(d, R * N * 4);
    for ([_]cuda.DeviceBuffer{ q, qs, z, y_a, y_n, y_b }) |b| try keep.append(ha, b);
    try mx.quant(r.k, r.s, x.ptr, .bf16, X_D, R, K, q.ptr, K, qs.ptr, K / 32);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    const G = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        z: u64,
        y: u64,
        /// 0: one launch of all rows; 1: rows 0 .. 1,023 then 1,024 ..; 2: rows 0 .. 1,023 then 1,025 ..
        mode: u8,

        fn rows(g: @This(), r0: usize, m: usize) anyerror!void {
            try mx.gemm(g.r.k, g.r.s, g.l, g.q + r0 * K, K, g.s + r0 * (K / 32) * 4, K / 32, m, g.y + r0 * N * 4, N, .f32, false, g.z);
        }

        fn go(g: @This()) anyerror!void {
            switch (g.mode) {
                0 => try g.rows(0, R),
                1 => {
                    try g.rows(0, quarter);
                    try g.rows(quarter, R - quarter);
                },
                else => {
                    try g.rows(0, quarter);
                    try g.rows(quarter + 1, R - quarter - 1);
                },
            }
        }
    };
    const one: G = .{ .r = r, .l = l, .q = q.ptr, .s = qs.ptr, .z = z.ptr, .y = y_a.ptr, .mode = 0 };
    var two = one;
    two.y = y_b.ptr;
    two.mode = 1;
    var again = one;
    again.y = y_n.ptr;
    const ms1 = try kernelBest(r, G, one, G.go);
    const msn = try kernelBest(r, G, again, G.go);
    const ms2 = try kernelBest(r, G, two, G.go);
    const ybytes = R * N * 4;
    try poison(y_a.ptr, ybytes, d, r.s);
    try one.go();
    try r.s.synchronize();
    const ref = try get(y_a.ptr, ybytes, d);
    defer free(ref);
    try poison(y_b.ptr, ybytes, d, r.s);
    try two.go();
    try r.s.synchronize();
    const got = try get(y_b.ptr, ybytes, d);
    defer free(got);
    var cut = two;
    cut.mode = 2;
    try poison(y_b.ptr, ybytes, d, r.s);
    try cut.go();
    try r.s.synchronize();
    const got_cut = try get(y_b.ptr, ybytes, d);
    defer free(got_cut);
    const kinds = [3]?mx.SfGemm{ mx.sfRoute(r.k, l, q.ptr, K, qs.ptr, K / 32, y_a.ptr, N, false, R), mx.sfRoute(r.k, l, q.ptr, K, qs.ptr, K / 32, y_a.ptr, N, false, quarter), mx.sfRoute(r.k, l, q.ptr, K, qs.ptr, K / 32, y_a.ptr, N, false, R - quarter) };
    var t: WcText = .{};
    try t.add("roci_local_first wq_a (K {d}, N {d}) block-scaled exact on the gathered rows, kernel only, ms: one launch of 4,096 rows {d:.3} ({s}), null control {d:.1} % ({s}); 1,024 rows then 3,072 rows {d:.3} = {d:.1} % over one launch ({s} then {s}). The research's kill line (more than 1.5 % slower): {s}", .{
        K,                                                     N, ms1, if (kinds[0]) |kd| @tagName(kd) else "the baseline kernels", pct(msn, ms1), resolved(pct(msn, ms1)), ms2, pct(ms2, ms1), if (kinds[1]) |kd| @tagName(kd) else "the baseline kernels", if (kinds[2]) |kd| @tagName(kd) else "the baseline kernels",
        if (pct(ms2, ms1) > 1.5) "crossed" else "not crossed",
    });
    r.report("roci_local_first wq_a: 1,024 rows then 3,072 rows against one launch of 4,096", differs(ref, got), true);
    r.report("roci_local_first wq_a control: row 1,024 not computed", differs(ref, got_cut), false);
}

// -- wq_b after the qr gather: the node's rows first -------------------------------------------------------------------------------

/// The cost of running wq_b (K 1,280, N 8,192, bf16 out) over the rows a node holds before the other nodes' qr rows have landed: the
/// owned quarter's launch first (a node of the second row block: rows 1,024 .. 2,047), then the rows before it, then the rows after it
/// (the three launches the gather's wait would sit before), against one launch of 4,096 rows, kernel only, and the bytes of the output
/// equal. The research's kill line: more than 1.5 % slower than one launch. A row left out must differ.
fn rociLocalFirstWqb(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x7b0b);
    const rng = prng.random();
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const R: usize = 4096;
    const quarter = R / 4;
    const K: usize = 1280;
    const N: usize = 8192;
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    const x = try upload(d, try bf16s(R * K, rng, 0.5, ha));
    const q = try cuda.DeviceBuffer.alloc(d, R * K);
    const qs = try cuda.DeviceBuffer.alloc(d, R * (K / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, R), 1) * 4);
    const y_a = try cuda.DeviceBuffer.alloc(d, R * N * 2);
    const y_n = try cuda.DeviceBuffer.alloc(d, R * N * 2);
    const y_b = try cuda.DeviceBuffer.alloc(d, R * N * 2);
    for ([_]cuda.DeviceBuffer{ x, q, qs, z, y_a, y_n, y_b }) |b| try keep.append(ha, b);
    try mx.quant(r.k, r.s, x.ptr, .bf16, K, R, K, q.ptr, K, qs.ptr, K / 32);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    const G = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        z: u64,
        y: u64,
        /// 0: one launch; 1: the owned quarter, the rows before it, the rows after it; 2: as 1 with the row after the owned quarter left out
        mode: u8,

        fn rows(g: @This(), r0: usize, m: usize) anyerror!void {
            if (m == 0) return;
            try mx.gemm(g.r.k, g.r.s, g.l, g.q + r0 * K, K, g.s + r0 * (K / 32) * 4, K / 32, m, g.y + r0 * N * 2, N, .bf16, false, g.z);
        }

        fn go(g: @This()) anyerror!void {
            switch (g.mode) {
                0 => try g.rows(0, R),
                1 => {
                    try g.rows(quarter, quarter);
                    try g.rows(0, quarter);
                    try g.rows(2 * quarter, 2 * quarter);
                },
                else => {
                    try g.rows(quarter, quarter);
                    try g.rows(0, quarter);
                    try g.rows(2 * quarter + 1, 2 * quarter - 1);
                },
            }
        }
    };
    const one: G = .{ .r = r, .l = l, .q = q.ptr, .s = qs.ptr, .z = z.ptr, .y = y_a.ptr, .mode = 0 };
    var three = one;
    three.y = y_b.ptr;
    three.mode = 1;
    var again = one;
    again.y = y_n.ptr;
    const ms1 = try kernelBest(r, G, one, G.go);
    const msn = try kernelBest(r, G, again, G.go);
    const ms3 = try kernelBest(r, G, three, G.go);
    const owned_only = try kernelBest(r, G, .{ .r = r, .l = l, .q = q.ptr, .s = qs.ptr, .z = z.ptr, .y = y_n.ptr, .mode = 3 }, struct {
        fn go(g: G) anyerror!void {
            try g.rows(quarter, quarter);
        }
    }.go);
    const ybytes = R * N * 2;
    try poison(y_a.ptr, ybytes, d, r.s);
    try one.go();
    try r.s.synchronize();
    const ref = try get(y_a.ptr, ybytes, d);
    defer free(ref);
    try poison(y_b.ptr, ybytes, d, r.s);
    try three.go();
    try r.s.synchronize();
    const got = try get(y_b.ptr, ybytes, d);
    defer free(got);
    var cut = three;
    cut.mode = 2;
    try poison(y_b.ptr, ybytes, d, r.s);
    try cut.go();
    try r.s.synchronize();
    const got_cut = try get(y_b.ptr, ybytes, d);
    defer free(got_cut);
    var t: WcText = .{};
    try t.add("roci_local_first wq_b (K {d}, N {d}) block-scaled exact on the gathered qr rows, kernel only, ms: one launch of 4,096 rows {d:.3} (null control {d:.1} % {s}); the owned quarter alone {d:.3}, then the rows before and after it {d:.3} = {d:.1} % over one launch. Kill line (more than 1.5 % slower): {s}", .{
        K, N, ms1, pct(msn, ms1), resolved(pct(msn, ms1)), owned_only, ms3, pct(ms3, ms1), if (pct(ms3, ms1) > 1.5) "crossed" else "not crossed",
    });
    r.report("roci_local_first wq_b: the owned quarter, the rows before it, the rows after it, against one launch of 4,096", differs(ref, got), true);
    r.report("roci_local_first wq_b control: the row after the owned quarter not computed", differs(ref, got_cut), false);
}

// -- wo_b on the two pairs' parts where they lie ---------------------------------------------------------------------------------

/// wo_b's block-scaled exact GEMM (K 4,096, N 2,560, 4,096 rows, y in the arena's write-combined memory) reading its activation rows from the
/// two pairs' codes parts where they lie (mx.gemmTwo: K blocks 0 .. 63 from the node's part, 64 .. 127 from the partner's) against the baseline path:
/// the four copies that assemble the input (prompt2d.catCodesCopies) and the GEMM on it. Every output byte equal; one code byte of the
/// partner's part changed must differ. The ms are the GEMM on the parts, and the copies plus the GEMM on the assembled input.
fn rociTwoBase(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x2ba5e);
    const rng = prng.random();
    const d = r.d;
    var line: [640]u8 = undefined;
    const n: usize = 4096;
    const uw: usize = 2048;
    const K: usize = 2 * uw;
    const N: usize = 2560;
    const ops = r.ops orelse return r.note("roci_two_base: the ops are available", false);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const ybytes = n * N * 4;
    var arena_mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&arena_mem, 2 * ybytes, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(arena_mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem, 0), "cuMemHostGetDevicePointer");
    try r.k.useStaged(wc, wc + 2 * ybytes);
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    const part_bytes = dsv41.prompt2d.catCodesBytes(n, uw);
    var parts: [2]u64 = undefined;
    for (0..2) |i| {
        const x = try upload(d, try bf16s(n * uw, rng, 0.5, ha));
        try keep.append(ha, x);
        const pb = try cuda.DeviceBuffer.alloc(d, part_bytes);
        try keep.append(ha, pb);
        parts[i] = pb.ptr;
        try mx.quant(r.k, r.s, x.ptr, .bf16, uw, n, uw, pb.ptr, uw, pb.ptr + n * uw, uw / 32);
    }
    const q = try cuda.DeviceBuffer.alloc(d, n * K);
    const qs = try cuda.DeviceBuffer.alloc(d, n * (K / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, n), 1) * 4);
    for ([_]cuda.DeviceBuffer{ q, qs, z }) |b| try keep.append(ha, b);
    try mx.sfProve(r.k, l.s);
    r.k.sf = replay_sf;
    const y_ref = wc;
    const y_two = wc + ybytes;
    // node 1 (pair 0): its part is the first, the partner's the second
    const pl = dsv41.prompt2d.catCodesPlan(1, .{ 0, 1, 2, 3 }, n, .{ uw, uw }, 0, parts[0], parts[1], q.ptr, qs.ptr);
    if (!mx.twoTakes(r.k, l, parts[0], uw, parts[0] + n * uw, uw / 32, y_two, N, n)) return r.note("roci_two_base: the kernels take the two-base GEMM (exact, split128, y in the arena)", false);
    const C = struct {
        r: *Run,
        l: weights.Linear,
        pl: dsv41.prompt2d.CatCodes,
        p: [2]u64,
        z: u64,
        y: u64,
        two: bool,
        fn go(c: @This()) anyerror!void {
            if (c.two) {
                try mx.gemmTwo(c.r.k, c.r.s, c.l, c.p[0], uw, c.p[0] + n * uw, uw / 32, .{ .q1 = c.p[1], .s1 = c.p[1] + n * uw, .kb_split = uw / 32 }, n, c.y, N, c.z);
            } else {
                try dsv41.prompt2d.catCodesCopies(c.r.ops.?, c.r.s, c.pl);
                try mx.gemm(c.r.k, c.r.s, c.l, c.pl.q, c.pl.k, c.pl.s, c.pl.k / 32, n, c.y, N, .f32, false, c.z);
            }
        }
    };
    _ = ops;
    const today: C = .{ .r = r, .l = l, .pl = pl, .p = parts, .z = z.ptr, .y = y_ref, .two = false };
    var two = today;
    two.y = y_two;
    two.two = true;
    try poison(y_ref, ybytes, d, r.s);
    try today.go();
    try r.s.synchronize();
    try poison(y_two, ybytes, d, r.s);
    try two.go();
    try r.s.synchronize();
    const ref = try get(y_ref, ybytes, d);
    defer free(ref);
    const got = try get(y_two, ybytes, d);
    defer free(got);
    const ms_today = try kernelBest(r, C, today, C.go);
    const ms_two = try kernelBest(r, C, two, C.go);
    // the control: one code byte of the partner's part changed
    var one: [1]u8 = undefined;
    const flipped = [1]u8{one[0] ^ 0x08};
    try poison(y_two, ybytes, d, r.s);
    try two.go();
    try r.s.synchronize();
    const cut = try get(y_two, ybytes, d);
    defer free(cut);
    var t: WcText = .{};
    try t.add("roci_two_base wo_b (K {d}, N {d}) block-scaled exact, y in write-combined host memory, kernel only, ms for 4,096 rows: the four copies and the GEMM on the assembled input {d:.3}; the GEMM on the two parts {d:.3} ({d:.1} % against the baseline path)", .{ K, N, ms_today, ms_two, pct(ms_two, ms_today) });
    r.note(t.buf[0..t.n], true);
    r.report(try std.fmt.bufPrint(&line, "roci_two_base wo_b: the GEMM on the node's part and the partner's part against the copies and the GEMM on the assembled input", .{}), differs(ref, got), true);
    r.report("roci_two_base wo_b control: one code byte of the partner's part changed", differs(ref, cut), false);
}

// -- TF_DS_MMA_SF's twins of the transport-integrated kernels ------------------------------------------------------------------------

/// The block-scaled arithmetic the roci_two_base and roci_three_base replays run (their chained wrappers set it for one call).
var replay_sf: mx.MmaSf = .exact;

fn rociTwoBaseChained(r: *Run, rng_in: std.Random) !void {
    replay_sf = .chained;
    defer replay_sf = .exact;
    try rociTwoBase(r, rng_in);
}

fn rociThreeBaseChained(r: *Run, rng_in: std.Random) !void {
    replay_sf = .chained;
    defer replay_sf = .exact;
    try rociThreeBase(r, rng_in);
}

/// wo_b's door kernel in TF_DS_MMA_SF's arithmetic (tf_dsn_sf_split128_st_door) in each node's row order against the plain chained launch
/// (mx.gemm, the whole-line kernel): every output byte equal; a row order that runs a quarter twice must differ.
fn chainedDoor(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xc4a1d00);
    const rng = prng.random();
    const d = r.d;
    _ = r.k.sfk orelse return r.note("chained_door: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var arena_mem: ?*anyopaque = null;
    defer if (arena_mem != null) {
        _ = d.api.cuMemFreeHost(arena_mem);
    };
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    var rigd = try DoorRig.init(d);
    defer rigd.deinit();
    const ybytes = sf_m * sf_n * 4;
    const rig = (try sfRig(r, rng, ha, &keep, &arena_mem)) orelse return r.note("chained_door: wo_b's 4,096 rows take tf_dsn_sf_split128 here", false);
    r.k.sf = .chained;
    const route_ok = mx.sentFirstTakes(r.k, rig.l, rig.q, sf_k, rig.s, sf_k / 32, rig.y_new, sf_n, sf_m);
    r.report("chained_door: sentFirstTakes takes wo_b in the chained arithmetic", @intFromBool(!route_ok), true);
    const today: TodayRun = .{ .r = r, .l = rig.l, .q = rig.q, .s = rig.s, .z = rig.z, .y = rig.y_ref };
    try poison(rig.y_ref, ybytes, d, r.s);
    try today.go();
    try r.s.synchronize();
    const ref = try get(rig.y_ref, ybytes, d);
    defer free(ref);
    const q = sf_m / 4;
    var bad: usize = 0;
    var orders: [4]mx.RowOrder = undefined;
    for (0..4) |g| orders[g] = mx.rowOrder(.{ (g ^ 1) * q, (g ^ 3) * q }, .{ g * q, (g ^ 2) * q }, q, sf_ct).?;
    for (orders) |ro| {
        const run: SentRun = .{ .r = r, .f = f, .a = rig.margs, .ro = ro, .door = rigd.arm() };
        try poison(rig.y_new, ybytes, d, r.s);
        try run.go();
        try r.s.synchronize();
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(ref, got);
    }
    r.report("chained_door wo_b: the door kernel in each of the four nodes' row orders against the plain chained launch", bad, true);
    var dup = orders[0];
    dup.quarters = (dup.quarters & 0x00ffffff) | (@as(u32, 0) << 24);
    const cut: SentRun = .{ .r = r, .f = f, .a = rig.margs, .ro = dup, .door = rigd.arm() };
    try poison(rig.y_new, ybytes, d, r.s);
    try cut.go();
    try r.s.synchronize();
    const got = try get(rig.y_new, ybytes, d);
    defer free(got);
    r.report("chained_door wo_b control: a row order that runs one quarter twice and another never", differs(ref, got), false);
}

// -- wo_b on the two pairs' parts: where each base lives -------------------------------------------------------------------------

/// roci_two_base in the engine's own memory: the node's part and the landed part, and wo_b's y, in the same block kind (the engine's arena is one
/// kind: write-combined host memory, locked anonymous 4 KiB pages registered with the driver as rdma.Pinned makes them, or those with
/// MADV_HUGEPAGE before first touch), against the replay's earlier setup (the parts in device memory, y in write-combined host memory).
/// For each kind: the exact whole-line kernel on the two bases (mx.gemmTwo) and the path it replaced, four copies into device memory and the plain
/// GEMM on them, kernel only, ms for 4,096 rows; the outputs byte for byte against the device-parts reference.
fn rociTwoBasePages(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x2ba5f);
    const rng = prng.random();
    const d = r.d;
    var line: [640]u8 = undefined;
    const n: usize = 4096;
    const uw: usize = 2048;
    const K: usize = 2 * uw;
    const N: usize = 2560;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const ybytes = n * N * 4;
    const part_bytes = dsv41.prompt2d.catCodesBytes(n, uw);
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    var src: [2]cuda.DeviceBuffer = undefined;
    for (0..2) |i| {
        const x = try upload(d, try bf16s(n * uw, rng, 0.5, ha));
        try keep.append(ha, x);
        src[i] = try cuda.DeviceBuffer.alloc(d, part_bytes);
        try keep.append(ha, src[i]);
        try mx.quant(r.k, r.s, x.ptr, .bf16, uw, n, uw, src[i].ptr, uw, src[i].ptr + n * uw, uw / 32);
    }
    const q = try cuda.DeviceBuffer.alloc(d, n * K);
    const qs = try cuda.DeviceBuffer.alloc(d, n * (K / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, n), 1) * 4);
    for ([_]cuda.DeviceBuffer{ q, qs, z }) |b| try keep.append(ha, b);
    try r.s.synchronize();
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    var hr = try dsv41.rdma.HostReg.open();
    defer hr.lib.close();
    var pool: WcPool = .{ .d = d };
    defer pool.deinit();
    const names = [_][]const u8{ "parts in device memory, y in write-combined host memory (the earlier replay)", "all in write-combined host memory", "all in locked 4 KiB pages (rdma.Pinned: no madvise)", "all in locked pages with MADV_HUGEPAGE before first touch" };
    var ref: ?[]u8 = null;
    defer if (ref) |x| free(x);
    var ms_all: [4][2]f32 = .{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } };
    var control_diff: usize = 0;
    for (0..4) |v| {
        // the three blocks of this kind: the two parts and y
        var p: [2]u64 = undefined;
        var y: u64 = undefined;
        var pinned: [3]?dsv41.rdma.Pinned = .{ null, null, null };
        defer for (&pinned) |*x| if (x.*) |*pp| pp.free(&hr);
        const sizes = [3]usize{ part_bytes, part_bytes, 2 * ybytes };
        var addr: [3]u64 = undefined;
        for (0..3) |bi| {
            addr[bi] = switch (v) {
                0 => if (bi < 2) src[bi].ptr else (try pool.alloc(.arena, sizes[bi])).dev,
                1 => (try pool.alloc(.arena, sizes[bi])).dev,
                2 => blk: {
                    pinned[bi] = try dsv41.rdma.Pinned.alloc(d, &hr, sizes[bi]);
                    break :blk pinned[bi].?.dev;
                },
                else => (try pool.alloc(.mmapreg, sizes[bi])).dev,
            };
        }
        p = .{ addr[0], addr[1] };
        y = addr[2];
        try r.s.synchronize();
        try r.k.useStaged(y, y + 2 * ybytes);
        const pl = dsv41.prompt2d.catCodesPlan(1, .{ 0, 1, 2, 3 }, n, .{ uw, uw }, 0, p[0], p[1], q.ptr, qs.ptr);
        if (!mx.twoTakes(r.k, l, p[0], uw, p[0] + n * uw, uw / 32, y + ybytes, N, n)) return r.note("roci_two_base_pages: the kernels take the two-base GEMM", false);
        const C = struct {
            r: *Run,
            l: weights.Linear,
            pl: dsv41.prompt2d.CatCodes,
            p: [2]u64,
            z: u64,
            y: u64,
            two: bool,
            fn go(c: @This()) anyerror!void {
                if (c.two) {
                    try mx.gemmTwo(c.r.k, c.r.s, c.l, c.p[0], uw, c.p[0] + n * uw, uw / 32, .{ .q1 = c.p[1], .s1 = c.p[1] + n * uw, .kb_split = uw / 32 }, n, c.y, N, c.z);
                } else {
                    try dsv41.prompt2d.catCodesCopies(c.r.ops.?, c.r.s, c.pl);
                    try mx.gemm(c.r.k, c.r.s, c.l, c.pl.q, c.pl.k, c.pl.s, c.pl.k / 32, n, c.y, N, .f32, false, c.z);
                }
            }
        };
        const today: C = .{ .r = r, .l = l, .pl = pl, .p = p, .z = z.ptr, .y = y, .two = false };
        var two = today;
        two.y = y + ybytes;
        two.two = true;
        try poison(y, 2 * ybytes, d, r.s);
        try today.go();
        try two.go();
        try r.s.synchronize();
        const a = try get(y, ybytes, d);
        const b = try get(y + ybytes, ybytes, d);
        defer free(b);
        if (ref == null) ref = a else free(a);
        ms_all[v][0] = try kernelBest(r, C, two, C.go);
        ms_all[v][1] = try kernelBest(r, C, today, C.go);
        r.note(try std.fmt.bufPrint(&line, "roci_two_base_pages {s}: wo_b on the two bases {d:.3} ms, the four copies and the plain GEMM {d:.3} ms ({d:.1} % on the two bases)", .{ names[v], ms_all[v][0], ms_all[v][1], pct(ms_all[v][0], ms_all[v][1]) }), true);
        r.report(try std.fmt.bufPrint(&line, "roci_two_base_pages {s}: the two-base output against the device-parts reference", .{names[v]}), differs(ref.?, b), true);
        if (v == 3) {
            var one: [1]u8 = undefined;
            const flipped = [1]u8{one[0] ^ 0x08};
            try poison(y + ybytes, ybytes, d, r.s);
            try two.go();
            try r.s.synchronize();
            const cut = try get(y + ybytes, ybytes, d);
            defer free(cut);
            control_diff = differs(ref.?, cut);
        }
    }
    r.report("roci_two_base_pages control: one code byte of the second part changed, in the last kind's block", control_diff, false);
}

// -- down on the pack's two parts where they lie ----------------------------------------------------------------------------------

/// Down of the grouped experts (the exact block-scaled kernel, 4,096 rows x 7 slots, the replay's experts with a random owner of each rest block)
/// reading its activation rows from the two pairs' pack parts where they lie (tf_dsn_sf_dn_exact_three_kernel: K blocks 0 .. 15 from the first
/// part's main codes, 16 .. 31 from the second's, 32 .. 35 from the part of the pair that computed the tile's expert's rest) against the baseline path:
/// tf_ds_assemble_rest in codes and in scales (prompt2d.packAssemble) into hq / hs and the exact down kernel on them. Every output byte equal; one
/// code byte of the second part changed must differ. The ms are the assembly and down, and down on the parts.
fn rociThreeBase(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x3ba5e);
    const rng = prng.random();
    const d = r.d;
    const sk = r.k.sfk orelse return r.note("roci_three_base: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    const f3 = sk.three(replay_sf) orelse return r.note("roci_three_base: tf_dsn_sf_dn_exact_three_kernel loads", false);
    const ops = r.ops orelse return r.note("roci_three_base: the ops are available", false);
    if (X_I != 1152 or X_MAIN != 512 or X_W != 640) return r.note("roci_three_base: the pack is 512 + 512 main columns and a 128-column rest", false);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const R: usize = 4096;
    const P = R * X_SL;
    const ex = try xExperts(d, rng, ha, &keep);
    const nx = ex.native.?;
    const x = try upload(d, try bf16s(R * X_D, rng, 0.5, ha));
    const p = try xPicks(R, rng, ha);
    const pb = try upload(d, std.mem.sliceAsBytes(p.pick));
    try keep.append(ha, x);
    try keep.append(ha, pb);
    const sc = try xScratch(d, R, true, ha, &keep);
    try mx.quant(r.k, r.s, x.ptr, .bf16, X_D, R, X_D, sc.q, X_D, sc.s, X_D / 32);
    try mx.grouping(r.k, r.s, d, X_E, sc, pb.ptr, R);
    try mx.tileList(r.k, r.s, sc);
    const pc = try dsv41.prompt2d.packCodesPlan(P, 512, 128);
    var parts: [2]u64 = undefined;
    for (0..2) |i| {
        const xd = try upload(d, try bf16s(P * X_W, rng, 0.5, ha));
        const pbuf = try cuda.DeviceBuffer.alloc(d, pc.bytes);
        try keep.append(ha, xd);
        try keep.append(ha, pbuf);
        parts[i] = pbuf.ptr;
        try dsv41.prompt2d.packQuant(r.k, r.s, xd.ptr, pbuf.ptr, pc);
    }
    const o_a = try cuda.DeviceBuffer.alloc(d, P * X_N * 4);
    const o_b = try cuda.DeviceBuffer.alloc(d, P * X_N * 4);
    try keep.append(ha, o_a);
    try keep.append(ha, o_b);
    const grid_y: usize = P / mx.pf_bm + X_E;
    const C = struct {
        r: *Run,
        sc: mx.ExpertScratch,
        nx: weights.NativeExperts,
        f: cuda.Function,
        f3: cuda.Function,
        parts: [2]u64,
        pc: dsv41.prompt2d.PackCodes,
        pick: u64,
        out: u64,
        three: bool,
        grid_y: usize,

        fn dnArgs(c: @This()) mx.DnArgs {
            return .{ .q = c.sc.hq, .s = c.sc.hs, .ldq = X_I, .lds = X_I / 32, .ent = c.sc.ent, .off = c.sc.off, .act = c.sc.act, .nact = c.sc.nact, .w2 = c.nx.w2, .s2 = c.nx.s2, .ldw = X_I / 2, .ldws = X_I / 32, .fmt = 1, .shared = X_E - 1, .sldw = X_I, .sldws = X_I / 32, .I = X_I, .N = X_N, .round_bf16 = 0, .out = c.out, .ldo = X_N };
        }

        fn go(c: @This()) anyerror!void {
            var args: cuda.Args = .{};
            var a = c.dnArgs();
            if (c.three) {
                // a part: its main block [P, 512] of codes then its rest block [P, 128]; the scales after the codes, main [P, 16] then rest [P, 4]
                const b = mx.packBlocks(c.pc.p, 512, 128);
                a.q = c.parts[0] + b.main_q;
                a.s = c.parts[0] + b.main_s;
                a.ldq = 512;
                a.lds = 16;
                args.add(a);
                args.add(c.sc.tiles);
                args.add(c.sc.ntiles);
                const p3: mx.Part3 = .{ .q1 = c.parts[1] + b.main_q, .s1 = c.parts[1] + b.main_s, .qr0 = c.parts[0] + b.rest_q, .sr0 = c.parts[0] + b.rest_s, .qr1 = c.parts[1] + b.rest_q, .sr1 = c.parts[1] + b.rest_s, .owner = c.nx.owner, .ldq_rest = 128, .lds_rest = 4, .kb_main = 16 };
                args.add(p3);
                try cuda.launch.launch(c.f3, .{ .grid = .{ .x = X_N / 128, .y = @intCast(c.grid_y), .z = 1 }, .block = .{ .x = 256 }, .shared = mx.pf_dn_r128_smem }, c.r.s, &args);
            } else {
                try dsv41.prompt2d.packAssemble(c.r.ops.?, c.r.s, c.parts[0], c.parts[1], c.pick, c.nx.owner, c.sc, c.pc);
                args.add(a);
                args.add(c.sc.tiles);
                args.add(c.sc.ntiles);
                try cuda.launch.launch(c.f, .{ .grid = .{ .x = X_N / 128, .y = @intCast(c.grid_y), .z = 1 }, .block = .{ .x = 256 }, .shared = mx.pf_dn_r128_smem }, c.r.s, &args);
            }
        }
    };
    _ = ops;
    const today: C = .{ .r = r, .sc = sc, .nx = nx, .f = sk.dn[@backingInt(replay_sf) - 1], .f3 = f3, .parts = parts, .pc = pc, .pick = pb.ptr, .out = o_a.ptr, .three = false, .grid_y = grid_y };
    var three = today;
    three.out = o_b.ptr;
    three.three = true;
    const bytes = P * X_N * 4;
    try poison(o_a.ptr, bytes, d, r.s);
    try today.go();
    try r.s.synchronize();
    const ref = try get(o_a.ptr, bytes, d);
    defer free(ref);
    try poison(o_b.ptr, bytes, d, r.s);
    try three.go();
    try r.s.synchronize();
    const got = try get(o_b.ptr, bytes, d);
    defer free(got);
    const ms_today = try kernelBest(r, C, today, C.go);
    const ms_three = try kernelBest(r, C, three, C.go);
    // the control: a main code of the second part changed
    var one: [1]u8 = undefined;
    const flipped = [1]u8{one[0] ^ 0x10};
    try poison(o_b.ptr, bytes, d, r.s);
    try three.go();
    try r.s.synchronize();
    const cut = try get(o_b.ptr, bytes, d);
    defer free(cut);
    var t: WcText = .{};
    try t.add("roci_three_base down (4,096 rows x 7 slots, I {d}, N {d}) block-scaled exact, kernel only, ms: the two assemblies and down on hq / hs {d:.3}; down on the two parts {d:.3} ({d:.1} % against the baseline path)", .{ X_I, X_N, ms_today, ms_three, pct(ms_three, ms_today) });
    // where the outputs first differ, and how many entries differ, so a failing run says where to look
    var first: ?usize = null;
    var bad_entries: usize = 0;
    {
        const row_bytes = X_N * 4;
        var last_bad: usize = std.math.maxInt(usize);
        for (ref, got, 0..) |rb, gb, i| if (rb != gb) {
            if (first == null) first = i;
            if (i / row_bytes != last_bad) {
                last_bad = i / row_bytes;
                bad_entries += 1;
            }
        };
    }
    if (first) |f| {
        const ent = f / (X_N * 4);
        try t.add("; FIRST DIFFERENCE at entry {d} (row {d}, slot {d}), column {d}; {d} of {d} entries differ", .{ ent, ent / X_SL, ent % X_SL, (f % (X_N * 4)) / 4, bad_entries, P });
    }
    r.note(t.buf[0..t.n], true);
    r.report("roci_three_base down: the entries' outputs from the pack's parts where they lie against the assembled input", differs(ref, got), true);
    r.report("roci_three_base down control: one main code of the second part changed", differs(ref, cut), false);
}

/// roci_three_base in the engine's own memory kinds: the two parts of the pack in device memory (the earlier replay), in write-combined host memory,
/// in locked 4 KiB pages as rdma.Pinned makes them, and in locked pages with MADV_HUGEPAGE before first touch. Down on the two parts (the three-base
/// kernel, kernel only, ms for 4,096 rows x 7 slots) in each kind; the outputs byte for byte against the baseline path on the device parts; a control
/// changes one main code of the second part in the last kind.
fn rociThreeBasePages(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x3ba5f);
    const rng = prng.random();
    const d = r.d;
    const sk = r.k.sfk orelse return r.note("roci_three_base_pages: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    const f3 = sk.dn_three orelse return r.note("roci_three_base_pages: tf_dsn_sf_dn_exact_three_kernel loads", false);
    if (X_I != 1152 or X_MAIN != 512 or X_W != 640) return r.note("roci_three_base_pages: the pack is 512 + 512 main columns and a 128-column rest", false);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const R: usize = 4096;
    const P = R * X_SL;
    const ex = try xExperts(d, rng, ha, &keep);
    const nx = ex.native.?;
    const x = try upload(d, try bf16s(R * X_D, rng, 0.5, ha));
    const p = try xPicks(R, rng, ha);
    const pb = try upload(d, std.mem.sliceAsBytes(p.pick));
    try keep.append(ha, x);
    try keep.append(ha, pb);
    const sc = try xScratch(d, R, true, ha, &keep);
    try mx.quant(r.k, r.s, x.ptr, .bf16, X_D, R, X_D, sc.q, X_D, sc.s, X_D / 32);
    try mx.grouping(r.k, r.s, d, X_E, sc, pb.ptr, R);
    try mx.tileList(r.k, r.s, sc);
    const pc = try dsv41.prompt2d.packCodesPlan(P, 512, 128);
    var dev: [2]u64 = undefined;
    for (0..2) |i| {
        const xd = try upload(d, try bf16s(P * X_W, rng, 0.5, ha));
        const pbuf = try cuda.DeviceBuffer.alloc(d, pc.bytes);
        try keep.append(ha, xd);
        try keep.append(ha, pbuf);
        dev[i] = pbuf.ptr;
        try dsv41.prompt2d.packQuant(r.k, r.s, xd.ptr, pbuf.ptr, pc);
    }
    const bytes = P * X_N * 4;
    const o_a = try cuda.DeviceBuffer.alloc(d, bytes);
    try keep.append(ha, o_a);
    const grid_y: usize = P / mx.pf_bm + X_E;
    const C = struct {
        r: *Run,
        sc: mx.ExpertScratch,
        nx: weights.NativeExperts,
        f3: cuda.Function,
        parts: [2]u64,
        pc: dsv41.prompt2d.PackCodes,
        out: u64,
        grid_y: usize,

        fn go(c: @This()) anyerror!void {
            var args: cuda.Args = .{};
            const b = mx.packBlocks(c.pc.p, 512, 128);
            const a: mx.DnArgs = .{ .q = c.parts[0] + b.main_q, .s = c.parts[0] + b.main_s, .ldq = 512, .lds = 16, .ent = c.sc.ent, .off = c.sc.off, .act = c.sc.act, .nact = c.sc.nact, .w2 = c.nx.w2, .s2 = c.nx.s2, .ldw = X_I / 2, .ldws = X_I / 32, .fmt = 1, .shared = X_E - 1, .sldw = X_I, .sldws = X_I / 32, .I = X_I, .N = X_N, .round_bf16 = 0, .out = c.out, .ldo = X_N };
            args.add(a);
            args.add(c.sc.tiles);
            args.add(c.sc.ntiles);
            const p3: mx.Part3 = .{ .q1 = c.parts[1] + b.main_q, .s1 = c.parts[1] + b.main_s, .qr0 = c.parts[0] + b.rest_q, .sr0 = c.parts[0] + b.rest_s, .qr1 = c.parts[1] + b.rest_q, .sr1 = c.parts[1] + b.rest_s, .owner = c.nx.owner, .ldq_rest = 128, .lds_rest = 4, .kb_main = 16 };
            args.add(p3);
            try cuda.launch.launch(c.f3, .{ .grid = .{ .x = X_N / 128, .y = @intCast(c.grid_y), .z = 1 }, .block = .{ .x = 256 }, .shared = mx.pf_dn_r128_smem }, c.r.s, &args);
        }
    };
    var hr = try dsv41.rdma.HostReg.open();
    defer hr.lib.close();
    var pool: WcPool = .{ .d = d };
    defer pool.deinit();
    const names = [_][]const u8{ "parts in device memory (the earlier replay)", "parts in write-combined host memory", "parts in locked 4 KiB pages (rdma.Pinned: no madvise)", "parts in locked pages with MADV_HUGEPAGE before first touch" };
    var ref: ?[]u8 = null;
    defer if (ref) |rf| free(rf);
    var control_diff: usize = 0;
    var line: [640]u8 = undefined;
    for (0..4) |v| {
        var parts: [2]u64 = dev;
        var pinned: [2]?dsv41.rdma.Pinned = .{ null, null };
        defer for (&pinned) |*pq| if (pq.*) |*pp| pp.free(&hr);
        if (v != 0) for (0..2) |i| {
            parts[i] = switch (v) {
                1 => (try pool.alloc(.arena, pc.bytes)).dev,
                2 => blk: {
                    pinned[i] = try dsv41.rdma.Pinned.alloc(d, &hr, pc.bytes);
                    break :blk pinned[i].?.dev;
                },
                else => (try pool.alloc(.mmapreg, pc.bytes)).dev,
            };
        };
        try r.s.synchronize();
        const c: C = .{ .r = r, .sc = sc, .nx = nx, .f3 = f3, .parts = parts, .pc = pc, .out = o_a.ptr, .grid_y = grid_y };
        try poison(o_a.ptr, bytes, d, r.s);
        try c.go();
        try r.s.synchronize();
        const got = try get(o_a.ptr, bytes, d);
        if (ref == null) ref = got else {
            defer free(got);
            r.report(try std.fmt.bufPrint(&line, "roci_three_base_pages {s}: down's output against the device-parts output", .{names[v]}), differs(ref.?, got), true);
        }
        const ms = try kernelBest(r, C, c, C.go);
        r.note(try std.fmt.bufPrint(&line, "roci_three_base_pages {s}: down on the two parts {d:.3} ms (4,096 rows x 7 slots, kernel only)", .{ names[v], ms }), true);
        if (v == 3) {
            var one: [1]u8 = undefined;
            const flipped = [1]u8{one[0] ^ 0x10};
            try poison(o_a.ptr, bytes, d, r.s);
            try c.go();
            try r.s.synchronize();
            const cut = try get(o_a.ptr, bytes, d);
            defer free(cut);
            control_diff = differs(ref.?, cut);
        }
    }
    r.report("roci_three_base_pages control: one main code of the second part changed, in the last kind's block", control_diff, false);
}

// -- T-ANY: the owned path's launch classes on a node's slab against the whole chunk's launch ----------------------------------------

/// The chunk row counts T-ANY runs: below a block, around the 64-row variants, a tail of the 4,096 pass, and the sizes the lead measures.
const any_ns = [_]usize{ 13, 63, 64, 65, 500, 1087, 1956, 3000 };
const any_max: usize = 3000;

/// The row counts a T-ANY class runs at and how its lines name them.
const AnyPlan = struct {
    /// the chunks' rows (a decoder window's for the decoder half)
    ns: []const usize,
    /// the most of them: the rows the buffers hold
    max: usize,
    /// how the lines name the row counts, and the chunks
    text: []const u8,
    unit: []const u8,
};
const any_plan: AnyPlan = .{ .ns = &any_ns, .max = any_max, .text = "n = 13, 63, 64, 65, 500, 1,087, 1,956, 3,000", .unit = "chunks" };

/// The decoder half's windows: every row count a chunk's decoder half runs at (1 row to the window's 128, prompt.decoderRows), whose slabs
/// (q(d) = 16 or 32 rows) hold every row count from 1 to 32.
const dec_max: usize = 128;
const dec_ds = blk: {
    var ds: [dec_max]usize = undefined;
    for (&ds, 1..) |*x, i| x.* = i;
    break :blk ds;
};
const dec_plan: AnyPlan = .{ .ns = &dec_ds, .max = dec_max, .text = "every decoder window d = 1 to 128 (slabs of q(d) = 16 or 32 rows)", .unit = "windows" };

/// One launch class of the owned path over rows [start, start + m): `out` is its output buffer (every row-indexed pointer offset by the
/// start, as the engine launches a slab), `row` the output bytes a row.
const AnyClass = struct {
    name: []const u8,
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, start: usize, m: usize) anyerror!void,
    out: u64,
    row: usize,
    /// The section's name, the prefix of its lines and of a failing step's message.
    section: []const u8 = "roci_any_rows",
    /// The row counts it runs at.
    plan: *const AnyPlan = &any_plan,
};

/// The control's short slab of a chunk of `n` rows (replay_runner.shortSlab).
const anyShort = dsv41.replay_runner.shortSlab;

comptime {
    // the outputs are poisoned with the byte the null control reads as "not written"
    if (dsv41.replay_runner.poison_byte != 0xa5) @compileError("poison() and the runner's poison byte differ");
}

/// What a class's run over its chunks found: the bytes that differ (a chunk whose null control failed counts every byte, uncompared),
/// the chunks whose control differed, the chunks whose null control held, and the first that did not (its rows, and whether the one
/// launch or the slabs left their output unwritten).
const AnyRes = struct { bad: usize = 0, controls_ok: usize = 0, null_ok: usize = 0, first_null: ?struct { n: usize, slabs: bool } = null };

/// Whole launch of n rows against the four ranks' slabs (slab_plan.slabs), the rows of the output equal; before the comparison the null
/// control, each run's output written (replay_runner.wrotePoisoned: at least half its bytes no longer the poison, not all zero); and the
/// control, one row of the last slab that has more than one row left out (anyShort), which must differ. A failing step names itself: the
/// class, the chunk row count, the slab and the step.
fn anyClassRun(r: *Run, c: AnyClass) !AnyRes {
    var res: AnyRes = .{};
    const max = c.plan.max;
    for (c.plan.ns) |n| {
        const bytes = n * c.row;
        poison(c.out, max * c.row, r.d, r.s) catch |e| return anyFail(c, n, null, "poison", e);
        c.run(c.ctx, 0, n) catch |e| return anyFail(c, n, null, "the one launch of the chunk's rows", e);
        r.s.synchronize() catch |e| return anyFail(c, n, null, "synchronize after the one launch", e);
        const ref = get(c.out, bytes, r.d) catch |e| return anyFail(c, n, null, "read back the one launch", e);
        defer free(ref);
        poison(c.out, max * c.row, r.d, r.s) catch |e| return anyFail(c, n, null, "poison", e);
        const sl = dsv41.slab_plan.slabs(n);
        for (sl, 0..) |sb, k| if (sb.n > 0) c.run(c.ctx, sb.start, sb.n) catch |e| return anyFail(c, n, .{ k, sb.start, sb.n }, "the slab's launch", e);
        r.s.synchronize() catch |e| return anyFail(c, n, null, "synchronize after the slabs", e);
        const got = get(c.out, bytes, r.d) catch |e| return anyFail(c, n, null, "read back the slabs", e);
        defer free(got);
        const ref_wrote = dsv41.replay_runner.wrotePoisoned(ref);
        if (ref_wrote and dsv41.replay_runner.wrotePoisoned(got)) {
            res.null_ok += 1;
            res.bad += differs(ref, got);
        } else {
            res.bad += bytes;
            if (res.first_null == null) res.first_null = .{ .n = n, .slabs = ref_wrote };
        }
        // the control: the last slab with more than one row, one row short
        poison(c.out, max * c.row, r.d, r.s) catch |e| return anyFail(c, n, null, "poison", e);
        const short = anyShort(n);
        for (sl, 0..) |sb, k| if (sb.n > 0) {
            const m = if (short != null and k == short.?) sb.n - 1 else sb.n;
            if (m > 0) c.run(c.ctx, sb.start, m) catch |e| return anyFail(c, n, .{ k, sb.start, m }, "the control's slab launch", e);
        };
        r.s.synchronize() catch |e| return anyFail(c, n, null, "synchronize after the control", e);
        const cut = get(c.out, bytes, r.d) catch |e| return anyFail(c, n, null, "read back the control", e);
        defer free(cut);
        res.controls_ok += @intFromBool(differs(ref, cut) > 0);
    }
    return res;
}

fn anyFail(c: AnyClass, n: usize, slab: ?[3]usize, step: []const u8, e: anyerror) anyerror {
    if (slab) |s| {
        std.debug.print("{s} {s}: FAILED at chunk n = {d}, slab {d} (start {d}, {d} rows), {s}: {s}\n", .{ c.section, c.name, n, s[0], s[1], s[2], step, @errorName(e) });
    } else std.debug.print("{s} {s}: FAILED at chunk n = {d}, {s}: {s}\n", .{ c.section, c.name, n, step, @errorName(e) });
    return e;
}

fn anyReport(r: *Run, c: AnyClass) !void {
    var line: [320]u8 = undefined;
    const res = try anyClassRun(r, c);
    r.report(try std.fmt.bufPrint(&line, "{s} {s}: the node's slab launches against the chunk's one launch at {s}", .{ c.section, c.name, c.plan.text }), res.bad, true);
    if (res.first_null) |f| {
        r.note(try std.fmt.bufPrint(&line, "{s} {s} null control: the one launch and the slabs wrote their outputs in {d} of {d} {s}; first at {d} rows: {s} did not", .{ c.section, c.name, res.null_ok, c.plan.ns.len, c.plan.unit, f.n, if (f.slabs) "the slabs' launches" else "the chunk's one launch" }), false);
    } else r.note(try std.fmt.bufPrint(&line, "{s} {s} null control: the one launch and the slabs wrote their outputs, in all {d} {s}", .{ c.section, c.name, c.plan.ns.len, c.plan.unit }), res.null_ok == c.plan.ns.len);
    r.report(try std.fmt.bufPrint(&line, "{s} {s} control: the last slab one row short, in all {d} {s}", .{ c.section, c.name, c.plan.ns.len, c.plan.unit }), if (res.controls_ok == c.plan.ns.len) 1 else 0, false);
}

/// TF_DS_ROCI_ANY's gate (T-ANY): each launch class the owned path makes at a slab's rows (act_quant, the dense FP8 GEMMs wq_a / wkv / wq_b on the
/// block-scaled exact kernels, the q RMSNorm, the router's logits and routing, the hyper-connection mix) at n_k rows of the slab plan against the one
/// launch of the chunk's n rows, at chunk row counts 13 to 3,000: every row's bytes equal. No engine path runs; this is the kernels' row-count
/// independence the ownership rule rests on.
fn rociAnyRows(r: *Run, rng_in: std.Random) !void {
    return rociRowsAt(r, rng_in, "roci_any_rows", &any_plan);
}

/// The decoder half on owned rows (layers 20-39 at a window of d decoder rows, the gate of moving its gathers onto the transport): the same
/// launch classes on each node's slab of every window d = 1 to 128 against the window's one launch, the baseline decoder half's.
fn rociDecRows(r: *Run, rng_in: std.Random) !void {
    return rociRowsAt(r, rng_in, "roci_dec_rows", &dec_plan);
}

fn rociRowsAt(r: *Run, rng_in: std.Random, section: []const u8, plan: *const AnyPlan) !void {
    var stage: []const u8 = "start";
    return rociAnyRowsBody(r, rng_in, &stage, section, plan) catch |e| {
        std.debug.print("{s}: FAILED in setup step \"{s}\": {s}\n", .{ section, stage, @errorName(e) });
        return e;
    };
}

fn rociAnyRowsBody(r: *Run, rng_in: std.Random, stage: *[]const u8, section: []const u8, plan: *const AnyPlan) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xa17);
    const rng = prng.random();
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    stage.* = "the activation of the plan's most rows and its act_quant";
    const M = plan.max;
    const D: usize = 5120;
    const x = try upload(d, try bf16s(M * D, rng, 0.5, ha));
    try keep.append(ha, x);
    const q5 = try cuda.DeviceBuffer.alloc(d, M * D);
    const s5 = try cuda.DeviceBuffer.alloc(d, M * (D / 32) * 4);
    try keep.append(ha, q5);
    try keep.append(ha, s5);
    try mx.quant(r.k, r.s, x.ptr, .bf16, D, M, D, q5.ptr, D, s5.ptr, D / 32);
    stage.* = "the q_a activation and its act_quant";
    const qa = try upload(d, try bf16s(M * 1280, rng, 1.0, ha));
    try keep.append(ha, qa);
    const q1 = try cuda.DeviceBuffer.alloc(d, M * 1280);
    const s1 = try cuda.DeviceBuffer.alloc(d, M * (1280 / 32) * 4);
    try keep.append(ha, q1);
    try keep.append(ha, s1);
    try mx.quant(r.k, r.s, qa.ptr, .bf16, 1280, M, 1280, q1.ptr, 1280, s1.ptr, 1280 / 32);
    stage.* = "the FP8 linears wq_a, wkv, wq_b";
    const wa = try fp8Linear(d, 1280, 5120, .fast, rng, ha, &keep);
    const wk = try fp8Linear(d, 512, 5120, .fast, rng, ha, &keep);
    const wb = try fp8Linear(d, 8192, 1280, .fast, rng, ha, &keep);
    stage.* = "the scale guards of the three linears (mx.sfProve)";
    try mx.sfProve(r.k, wa.s);
    try mx.sfProve(r.k, wk.s);
    try mx.sfProve(r.k, wb.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    stage.* = "the output, split-partial and router buffers";
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(wb, M), mx.zFloats(wa, M), mx.zFloats(wk, M), 1) * 4);
    try keep.append(ha, z);
    const out_big = try cuda.DeviceBuffer.alloc(d, M * 8192 * 2);
    try keep.append(ha, out_big);
    const nw = try upload(d, try bf16s(1280, rng, 1.0, ha));
    try keep.append(ha, nw);
    const gate_w = try upload(d, try bf16s(384 * D, rng, 0.5, ha));
    const gate_b = try upload(d, try f32s(384, rng, ha));
    try keep.append(ha, gate_w);
    try keep.append(ha, gate_b);
    const logits_in = try upload(d, try f32s(M * 384, rng, ha));
    try keep.append(ha, logits_in);

    stage.* = "the launch classes";
    const Quant = struct {
        r: *Run,
        x: u64,
        q: u64,
        s: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try mx.quant(c.r.k, c.r.s, c.x + start * D * 2, .bf16, D, m, D, c.q + start * D, D, c.s + start * (D / 32) * 4, D / 32);
        }
    };
    var qc: Quant = .{ .r = r, .x = x.ptr, .q = out_big.ptr, .s = out_big.ptr + M * D };
    // (quant's two outputs are checked through one buffer: codes then scales; the scales are compared by the GEMMs' use of them as well)
    try anyReport(r, .{ .name = "act_quant", .ctx = &qc, .run = Quant.run, .out = out_big.ptr, .row = D, .section = section, .plan = plan });

    const Gemm = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        k: usize,
        z: u64,
        y: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try mx.gemm(c.r.k, c.r.s, c.l, c.q + start * c.k, c.k, c.s + start * (c.k / 32) * 4, c.k / 32, m, c.y + start * c.l.n * 2, c.l.n, .bf16, false, c.z);
        }
    };
    var ga: Gemm = .{ .r = r, .l = wa, .q = q5.ptr, .s = s5.ptr, .k = 5120, .z = z.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "wq_a GEMM", .ctx = &ga, .run = Gemm.run, .out = out_big.ptr, .row = 1280 * 2, .section = section, .plan = plan });
    var gk: Gemm = .{ .r = r, .l = wk, .q = q5.ptr, .s = s5.ptr, .k = 5120, .z = z.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "wkv GEMM", .ctx = &gk, .run = Gemm.run, .out = out_big.ptr, .row = 512 * 2, .section = section, .plan = plan });
    var gb: Gemm = .{ .r = r, .l = wb, .q = q1.ptr, .s = s1.ptr, .k = 1280, .z = z.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "wq_b GEMM", .ctx = &gb, .run = Gemm.run, .out = out_big.ptr, .row = 8192 * 2, .section = section, .plan = plan });

    const Norm = struct {
        r: *Run,
        x: u64,
        w: u64,
        y: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try dsv41.tri_basic.rmsnorm(c.r.t, c.x + start * 1280 * 2, 1280, c.w, c.y + start * 1280 * 2, 1280, 1e-20, m, 1280);
        }
    };
    var nc: Norm = .{ .r = r, .x = qa.ptr, .w = nw.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "q RMSNorm", .ctx = &nc, .run = Norm.run, .out = out_big.ptr, .row = 1280 * 2, .section = section, .plan = plan });

    const Logits = struct {
        r: *Run,
        x: u64,
        w: u64,
        y: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try mx.bf16Gemm(c.r.k, c.r.s, c.x + start * D * 2, D, c.w, D, m, 384, D, c.y + start * 384 * 4, 384, .f32, 1, false, 1, 0, 0, 0, null);
        }
    };
    var lc: Logits = .{ .r = r, .x = x.ptr, .w = gate_w.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "router logits", .ctx = &lc, .run = Logits.run, .out = out_big.ptr, .row = 384 * 4, .section = section, .plan = plan });

    const Route = struct {
        r: *Run,
        logits: u64,
        bias: u64,
        pick: u64,
        wts: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try dsv41.tri_norm.route(c.r.t, c.logits + start * 384 * 4, 0, c.bias, 6, 1.5, 384, c.pick + start * 7 * 4, c.wts + start * 7 * 4, m, 384, 7);
        }
    };
    var rc: Route = .{ .r = r, .logits = logits_in.ptr, .bias = gate_b.ptr, .pick = out_big.ptr, .wts = out_big.ptr + M * 7 * 4 };
    // picks and weights are compared through the pick rows; the weights lie in the buffer after them
    try anyReport(r, .{ .name = "ffnRoute", .ctx = &rc, .run = Route.run, .out = out_big.ptr, .row = 7 * 4, .section = section, .plan = plan });

    if (r.t.extra == null) {
        var nl: [128]u8 = undefined;
        r.note(try std.fmt.bufPrint(&nl, "{s} hcPre: the kit's extra set loads", .{section}), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcPre null control: the kit's extra set loads", .{section}), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcPre control: the kit's extra set loads", .{section}), false);
        return;
    }
    stage.* = "the hyper-connection mix buffers";
    const hc_blocks = dsv41.tri_basic.hc_blocks;
    const h = try upload(d, try bf16s(M * 4 * D, rng, 1.0, ha));
    const fnw = try upload(d, try f32s(24 * 4 * D, rng, ha));
    const scale = try upload(d, try f32s(3, rng, ha));
    const base = try upload(d, try f32s(24, rng, ha));
    const pre_in = try upload(d, try f32s(M * 4, rng, ha));
    const nw5 = try upload(d, try bf16s(D, rng, 1.0, ha));
    const part = try cuda.DeviceBuffer.alloc(d, M * hc_blocks * 32 * 4);
    const post = try cuda.DeviceBuffer.alloc(d, M * 4 * 4);
    const comb = try cuda.DeviceBuffer.alloc(d, M * 16 * 4);
    const pre_o = try cuda.DeviceBuffer.alloc(d, M * 4 * 4);
    for ([_]cuda.DeviceBuffer{ h, fnw, scale, base, pre_in, nw5, part, post, comb, pre_o }) |b| try keep.append(ha, b);
    const Hc = struct {
        r: *Run,
        h: u64,
        fnw: u64,
        scale: u64,
        base: u64,
        pre_in: u64,
        nw: u64,
        x: u64,
        pre_o: u64,
        post: u64,
        comb: u64,
        part: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try dsv41.tri_basic.hcPre(c.r.t, c.h + start * 4 * D * 2, c.fnw, c.scale, c.base, c.pre_in + start * 16, c.nw, 1e-20, 1e-6, 20, c.x + start * D * 2, c.pre_o + start * 16, c.post + start * 16, c.comb + start * 64, c.part + start * hc_blocks * 32 * 4, m, D);
        }
    };
    var hcc: Hc = .{ .r = r, .h = h.ptr, .fnw = fnw.ptr, .scale = scale.ptr, .base = base.ptr, .pre_in = pre_in.ptr, .nw = nw5.ptr, .x = out_big.ptr, .pre_o = pre_o.ptr, .post = post.ptr, .comb = comb.ptr, .part = part.ptr };
    try anyReport(r, .{ .name = "hcPre (x)", .ctx = &hcc, .run = Hc.run, .out = out_big.ptr, .row = D * 2, .section = section, .plan = plan });
}

// -- T-ANY, the rest of the owned path's launch classes: the grouped experts, the index queries, the Engram gate ---------------------------

/// The grouped experts at a chunk of any row count: gate / up from the activation codes the nodes' slabs quantized into the scratch
/// (mx.quant of each slab's rows at the slab's row offsets, as ffnRows does; the codes of the whole chunk gathered), mx.gateUpCodes at the
/// chunk's n rows, against mx.gateUp's own quantization of the n rows: every byte of the pack equal, at chunk row counts 13 to 3,000.
/// The control leaves the last row of the last slab with more than one row unquantized: the pack must differ.
fn rociAnyExperts(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xa1e);
    const rng = prng.random();
    const d = r.d;
    var line: [256]u8 = undefined;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const M = any_max;
    const ex = try xExperts(d, rng, ha, &keep);
    const sc = try xScratch(d, M, true, ha, &keep);
    const hb_bytes = M * X_SL * X_W * 2;
    const hb_ref = try cuda.DeviceBuffer.alloc(d, hb_bytes);
    const hb_got = try cuda.DeviceBuffer.alloc(d, hb_bytes);
    try keep.append(ha, hb_ref);
    try keep.append(ha, hb_got);
    var bad: usize = 0;
    var controls_ok: usize = 0;
    for (any_ns) |n| {
        const x = try upload(d, try bf16s(n * X_D, rng, 0.5, ha));
        const pw = try xPicks(n, rng, ha);
        const pb = try upload(d, std.mem.sliceAsBytes(pw.pick));
        const wb = try upload(d, std.mem.sliceAsBytes(pw.wts));
        for ([_]cuda.DeviceBuffer{ x, pb, wb }) |b| try keep.append(ha, b);
        const bytes = n * X_SL * X_W * 2;
        try poison(hb_ref.ptr, hb_bytes, d, r.s);
        try mx.gateUp(r.k, r.s, d, ex, sc, x.ptr, X_D, pb.ptr, wb.ptr, n, 7.0, hb_ref.ptr, X_MAIN, false);
        try r.s.synchronize();
        const ref = try get(hb_ref.ptr, bytes, d);
        defer free(ref);
        const sl = dsv41.slab_plan.slabs(n);
        var short: ?usize = null;
        for (sl, 0..) |sb, k| if (sb.n > 1) {
            short = k;
        };
        for ([_]bool{ false, true }) |control| {
            try poison(sc.q, M * X_D, d, r.s);
            try poison(sc.s, M * (X_D / 32) * 4, d, r.s);
            try poison(hb_got.ptr, hb_bytes, d, r.s);
            for (sl, 0..) |sb, k| if (sb.n > 0) {
                const m = if (control and short != null and k == short.?) sb.n - 1 else sb.n;
                try mx.quant(r.k, r.s, x.ptr + sb.start * X_D * 2, .bf16, X_D, m, X_D, sc.q + sb.start * X_D, X_D, sc.s + sb.start * (X_D / 32) * 4, X_D / 32);
            };
            try mx.gateUpCodes(r.k, r.s, d, ex, sc, pb.ptr, wb.ptr, n, 7.0, hb_got.ptr, X_MAIN);
            try r.s.synchronize();
            const got = try get(hb_got.ptr, bytes, d);
            defer free(got);
            if (control) controls_ok += @intFromBool(differs(ref, got) > 0) else bad += differs(ref, got);
        }
    }
    r.report(try std.fmt.bufPrint(&line, "roci_any_experts: gate / up from the slabs' codes at the chunk's n rows against mx.gateUp's own quantization at n = 13, 63, 64, 65, 500, 1,087, 1,956, 3,000", .{}), bad, true);
    r.report(try std.fmt.bufPrint(&line, "roci_any_experts control: the last slab one row unquantized, in all {d} chunks", .{any_ns.len}), if (controls_ok == any_ns.len) 1 else 0, false);
}

const AI_QW: usize = 1280;
const AI_D: usize = 5120;
const AI_IH: usize = 32;
const AI_ID: usize = 128;
const AI_RD: usize = 64;
const AI_START: usize = 70000;

/// The indexer's per-row launches on a slab (prompt.indexer with TF_DS_IDX_SHARD): wq_b through mx.linear into iq, idxQueries (the RoPE of iq's
/// rows and fp4_qd's bytes into iq4), weights_proj into iw, each at the slab's row offsets, against the one launch of the chunk's n rows.
const IdxCtx = struct {
    r: *Run,
    sc: mx.Scratch,
    wqb: weights.Linear,
    qr: u64,
    x: u64,
    proj: u64,
    cos: u64,
    sin: u64,
    pos: u64,
    iq: u64,
    iq4: u64,
    iw: u64,
    fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
        const c: *@This() = @ptrCast(@alignCast(p));
        try idxProjectRows(c.r, c.sc, c.wqb, c.qr, c.x, c.proj, c.cos, c.sin, c.pos, .{ .start = start, .n = m }, c.iq, c.iq4, c.iw);
    }
};

fn idxProjectRows(r: *Run, sc: mx.Scratch, wqb: weights.Linear, qr: u64, x: u64, proj: u64, cos: u64, sin: u64, pos: u64, rows: dsv41.prompt.QRows, iq: u64, iq4: u64, iw: u64) !void {
    try mx.linear(r.k, r.s, sc, wqb, qr + rows.start * AI_QW * 2, .bf16, AI_QW, rows.n, iq + rows.start * AI_IH * AI_ID * 2, AI_IH * AI_ID, .bf16, false, null);
    try dsv41.prompt.idxQueries(r.t, iq, iq4, cos, sin, pos, rows, AI_RD, AI_IH, AI_ID);
    try mx.bf16Gemm(r.k, r.s, x + rows.start * AI_D * 2, AI_D, proj, AI_D, rows.n, AI_IH, AI_D, iw + rows.start * AI_IH * 2, AI_IH, .bf16, dsv41.exact.indexScale(AI_ID, AI_IH), true, 1, 0, 0, 0, null);
}

fn rociAnyIndex(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xa1d);
    const rng = prng.random();
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const M = any_max;
    const wqb = try fp8Linear(d, AI_IH * AI_ID, AI_QW, .fast, rng, ha, &keep);
    const qr = try upload(d, try bf16s(M * AI_QW, rng, 0.5, ha));
    const x = try upload(d, try bf16s(M * AI_D, rng, 0.5, ha));
    const proj = try upload(d, try bf16s(AI_IH * AI_D, rng, 0.05, ha));
    const tab = try ha.alloc(f32, (AI_START + M) * (AI_RD / 2));
    for (tab) |*v| v.* = rng.float(f32) * 2 - 1;
    const cos = try upload(d, std.mem.sliceAsBytes(tab));
    for (tab) |*v| v.* = rng.float(f32) * 2 - 1;
    const sin = try upload(d, std.mem.sliceAsBytes(tab));
    const ph = try ha.alloc(i64, M);
    for (ph, 0..) |*pv, i| pv.* = @intCast(AI_START + i);
    const pos = try upload(d, std.mem.sliceAsBytes(ph));
    for ([_]cuda.DeviceBuffer{ qr, x, proj, cos, sin, pos }) |b| try keep.append(ha, b);
    var sb: [4]cuda.DeviceBuffer = undefined;
    for (&sb, mx.Scratch.sizes(M, AI_QW, @max(mx.zFloats(wqb, M), 1))) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b.*);
    }
    try mx.sfProve(r.k, wqb.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    const iq = try cuda.DeviceBuffer.alloc(d, M * AI_IH * AI_ID * 2);
    const iq4 = try cuda.DeviceBuffer.alloc(d, M * AI_IH * AI_ID * 2);
    const iw = try cuda.DeviceBuffer.alloc(d, M * AI_IH * 2);
    for ([_]cuda.DeviceBuffer{ iq, iq4, iw }) |b| try keep.append(ha, b);
    var ctx: IdxCtx = .{ .r = r, .sc = .{ .q = sb[0].ptr, .s = sb[1].ptr, .z = sb[2].ptr, .fake = sb[3].ptr, .rows = M, .max_k = AI_QW }, .wqb = wqb, .qr = qr.ptr, .x = x.ptr, .proj = proj.ptr, .cos = cos.ptr, .sin = sin.ptr, .pos = pos.ptr, .iq = iq.ptr, .iq4 = iq4.ptr, .iw = iw.ptr };
    try anyReport(r, .{ .name = "index queries (fp4_qd bytes)", .ctx = &ctx, .run = IdxCtx.run, .out = iq4.ptr, .row = AI_IH * AI_ID * 2, .section = "roci_any_index" });
    try anyReport(r, .{ .name = "index weights (weights_proj)", .ctx = &ctx, .run = IdxCtx.run, .out = iw.ptr, .row = AI_IH * 2, .section = "roci_any_index" });
}

/// The Engram gate on a slab: _engram_gate on h and the assembled key and value rows at the slab's row offsets against the chunk's one launch; and
/// _engram_gate_2b reading the transport's two blocks with a block stride of q rows (the slab plan's stride, the last slab shorter than q) against
/// the assembled rows and _engram_gate.
const EgCtx = struct {
    r: *Run,
    h: u64,
    kv: u64,
    qk: u64,
    out: u64,
    fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
        const c: *@This() = @ptrCast(@alignCast(p));
        try dsv41.tri_basic.engramGate(c.r.t, c.h + start * 4 * 5120 * 2, c.kv + start * 5 * 5120 * 2, c.qk, c.out + start * 4 * 5120 * 2, 1e-20, m, 5120);
    }
};

fn rociAnyEngram(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xa1c);
    const rng = prng.random();
    const d = r.d;
    const D: usize = 5120;
    const EW: usize = 12800;
    var line: [256]u8 = undefined;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const M = any_max;
    const h = try upload(d, try bf16s(M * 4 * D, rng, 1.0, ha));
    const kv = try upload(d, try bf16s(M * 5 * D, rng, 1.0, ha));
    const qk = try upload(d, try f32s(4 * D, rng, ha));
    const out = try cuda.DeviceBuffer.alloc(d, M * 4 * D * 2);
    for ([_]cuda.DeviceBuffer{ h, kv, qk, out }) |b| try keep.append(ha, b);
    var ctx: EgCtx = .{ .r = r, .h = h.ptr, .kv = kv.ptr, .qk = qk.ptr, .out = out.ptr };
    try anyReport(r, .{ .name = "_engram_gate", .ctx = &ctx, .run = EgCtx.run, .out = out.ptr, .row = 4 * D * 2, .section = "roci_any_engram" });
    if (r.t.extra == null) return r.note("roci_any_engram _engram_gate_2b: the kit's extra set loads", false);
    // the two blocks at the slab plan's stride: the slab's rows only are read, q rows apart
    var bad: usize = 0;
    var controls_ok: usize = 0;
    var tested: usize = 0;
    for ([_]usize{ 13, 63, 65, 500, 1087 }) |n| {
        const q = dsv41.slab_plan.q(n);
        const m = dsv41.slab_plan.slab(n, 3).n; // the last slab: m <= q rows, the stride q
        if (m == 0 or m == q) continue;
        tested += 1;
        const eb = try rociBlocks(q, EW, rng, ha);
        var ekb = try upload(d, std.mem.sliceAsBytes(eb));
        defer ekb.free();
        var kva = try cuda.DeviceBuffer.alloc(d, m * 2 * EW * 2);
        defer kva.free();
        for (0..2) |b| try r.ops.?.copyRows(r.s, ekb.ptr + b * q * EW * 2, EW * 2, kva.ptr + b * EW * 2, 2 * EW * 2, EW * 2, m);
        var o: [2]cuda.DeviceBuffer = undefined;
        for (&o, 0..) |*ob, i| {
            ob.* = try cuda.DeviceBuffer.alloc(d, m * 4 * D * 2);
            try poison(ob.ptr, m * 4 * D * 2, d, r.s);
            if (i == 0) try dsv41.tri_basic.engramGate(r.t, h.ptr, kva.ptr, qk.ptr, ob.ptr, 1e-20, m, D) else try dsv41.tri_basic.engramGate2b(r.t, h.ptr, ekb.ptr, q * EW, qk.ptr, ob.ptr, 1e-20, m, D);
        }
        try r.s.synchronize();
        const g0 = try get(o[0].ptr, m * 4 * D * 2, d);
        const g1 = try get(o[1].ptr, m * 4 * D * 2, d);
        bad += differs(g0, g1);
        free(g1);
        for (&o) |*ob| ob.free();
        // the control: the stride one row short reads the second block from the wrong place
        var oc = try cuda.DeviceBuffer.alloc(d, m * 4 * D * 2);
        defer oc.free();
        try poison(oc.ptr, m * 4 * D * 2, d, r.s);
        try dsv41.tri_basic.engramGate2b(r.t, h.ptr, ekb.ptr, (q - 1) * EW, qk.ptr, oc.ptr, 1e-20, m, D);
        try r.s.synchronize();
        const gc = try get(oc.ptr, m * 4 * D * 2, d);
        controls_ok += @intFromBool(differs(g0, gc) > 0);
        free(gc);
        free(g0);
    }
    r.report(try std.fmt.bufPrint(&line, "roci_any_engram: _engram_gate_2b on the last slab at the plan's block stride q against the assembled rows and _engram_gate ({d} chunk row counts)", .{tested}), bad, true);
    r.report("roci_any_engram control: the block stride one row short in every chunk", if (controls_ok == tested and tested != 0) 1 else 0, false);
}

// -- The decoder half on owned rows: the launch classes it adds to the encoder half's ---------------------------------------------

/// The hidden width, a pair's columns, a row of the streams.
const DM_D: usize = 5120;
const DM_W: usize = DM_D / 2;
const DM_RH: usize = 4 * DM_D * 2;

/// fp32 values, normal * scale.
fn f32sAt(n: usize, rng: std.Random, scale: f32, a: std.mem.Allocator) ![]u8 {
    const v = try a.alloc(f32, n);
    for (v) |*x| x.* = rng.floatNorm(f32) * scale;
    return std.mem.sliceAsBytes(v);
}

/// The device bytes of `parts` (address, bytes) read back one after another (freed with free).
fn getParts(d: *const cuda.Driver, parts: []const [2]u64) ![]u8 {
    var n: usize = 0;
    for (parts) |pt| n += @intCast(pt[1]);
    const x = try std.heap.page_allocator.alloc(u8, n);
    errdefer std.heap.page_allocator.free(x);
    var at: usize = 0;
    for (parts) |pt| {
        const len: usize = @intCast(pt[1]);
        at += len;
    }
    return x;
}

/// One bf16 value at `addr` in device memory changed (an exponent bit flipped, `saved` its old bits) or put back: the null control's change
/// of an input value. Synchronous: the runner has synchronized the stream before it.
fn flipBf16(r: *Run, addr: u64, on: bool, saved: *u16) !void {
    if (on) {
        const v: u16 = saved.* ^ 0x4000;
}

/// The sums a mix posts for a window of d rows in the two forms its two paths read: the baseline fp32 planes [2, d, D] over the window (plane 0
/// the bf16 sums widened, plane 1 -0.0: the readers' input from the quarters' gather) and, for each slab of the plan, the two bf16 blocks
/// [2, q, D / 2] the transport lands for the slab's owner (block 0 pair 0's columns, block 1 pair 1's; the slab's rows first, the rest
/// poisoned), q = slab_plan.q(d).
const DecSums = struct {
    r: *Run,
    sums: u64, // bf16 [2, dec_max, D / 2]: block b's row i
    planes: u64, // fp32 [2, dec_max, D]
    blk: [4]u64, // bf16 [2, 32, D / 2] a slab

    fn prep(c: *DecSums, d: usize) !void {
        const ops = c.r.ops.?;
        try ops.widenPairs(c.r.s, c.sums, c.sums + dec_max * DM_W * 2, c.planes, d, DM_W);
        try c.r.d.check(c.r.d.api.cuMemsetD32Async(c.planes + d * DM_D * 4, dsv41.prompt2d.plane1_word, d * DM_D, c.r.s.handle), "cuMemsetD32Async");
        const q = dsv41.slab_plan.q(d);
        for (dsv41.slab_plan.slabs(d), 0..) |sb, k| {
            try poison(c.blk[k], 2 * 32 * DM_W * 2, c.r.d, c.r.s);
            for (0..2) |b| try ops.copyRows(c.r.s, c.sums + b * dec_max * DM_W * 2 + sb.start * DM_W * 2, DM_W * 2, c.blk[k] + b * q * DM_W * 2, DM_W * 2, DM_W * 2, sb.n);
        }
    }
};

/// A mix with the previous sublayer's sums posted (attnMixes' and ffnMixes' hcPrePf): the baseline _hc_mix_pf on the planes and _hc_finish over the
/// window's d rows; the owned path's _hc_mix_pf_2b on the slab's blocks at the plan's stride q (`short` rows less: the stride control) and
/// _hc_finish at the slab's rows. post and comb are read by the mix and written by the finish: restored before each run.
const DecMix = struct {
    r: *Run,
    sums: *DecSums,
    h: u64,
    fnw: u64,
    scale: u64,
    base: u64,
    pre_in: u64,
    nw: u64,
    post_in: u64,
    comb_in: u64,
    h_out: u64,
    x: u64,
    pre_o: u64,
    post: u64,
    comb: u64,
    part: u64,
    short: usize = 0,
    saved: u16 = 0,

    pub const strided = true;
    const part_row = dsv41.tri_basic.hc_blocks * 32 * 4;

    /// The window's last row's first sum (block 0), the source prep builds the planes and the blocks from.
    pub fn perturb(c: *DecMix, d: usize, on: bool) !void {
        try flipBf16(c.r, c.sums.sums + (d - 1) * DM_W * 2, on, &c.saved);
    }

    pub fn prep(c: *DecMix, d: usize) !void {
        try c.sums.prep(d);
    }
    pub fn reset(c: *DecMix) !void {
        const r = c.r;
        try poison(c.h_out, dec_max * DM_RH, r.d, r.s);
        try poison(c.x, dec_max * DM_D * 2, r.d, r.s);
        try poison(c.pre_o, dec_max * 16, r.d, r.s);
        try poison(c.part, dec_max * part_row, r.d, r.s);
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(c.post, c.post_in, dec_max * 16, r.s.handle), "cuMemcpyDtoDAsync");
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(c.comb, c.comb_in, dec_max * 64, r.s.handle), "cuMemcpyDtoDAsync");
    }
    pub fn whole(c: *DecMix, d: usize) !void {
        _ = try dsv41.tri_hc.hcPrePf(c.r.t, c.h, c.fnw, c.scale, c.base, c.pre_in, c.nw, 1e-20, 1e-6, 20, c.x, c.pre_o, c.post, c.comb, c.part, .{ .gathered = c.sums.planes, .world = 2, .h_out = c.h_out }, d, DM_D);
    }
    pub fn slab(c: *DecMix, d: usize, k: usize, s: usize, m: usize) !void {
        const q = dsv41.slab_plan.q(d);
        _ = try dsv41.tri_hc.hcPrePf(c.r.t, c.h + s * DM_RH, c.fnw, c.scale, c.base, c.pre_in + s * 16, c.nw, 1e-20, 1e-6, 20, c.x + s * DM_D * 2, c.pre_o + s * 16, c.post + s * 16, c.comb + s * 64, c.part + s * part_row, .{ .gathered = c.sums.blk[k], .world = 2, .h_out = c.h_out + s * DM_RH, .blk_rs = (q - c.short) * DM_W }, m, DM_D);
    }
    pub fn after(c: *DecMix, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn read(c: *DecMix, d: usize) ![]u8 {
        return getParts(c.r.d, &.{ .{ c.h_out, d * DM_RH }, .{ c.x, d * DM_D * 2 }, .{ c.pre_o, d * 16 }, .{ c.post, d * 16 }, .{ c.comb, d * 64 } });
    }
};

/// A pending sum posted without a mix (a DSpark tap's and the head's hc_post), in place on the streams: the baseline _hc_post on the planes over
/// the window's d rows; the owned path's _hc_post_2b on the slab's blocks at the plan's stride q (`short` rows less: the stride control).
const DecPost = struct {
    r: *Run,
    sums: *DecSums,
    h_in: u64,
    h: u64,
    post: u64,
    comb: u64,
    short: usize = 0,
    saved: u16 = 0,

    pub const strided = true;

    /// The window's last row's first sum (block 0), the source prep builds the planes and the blocks from.
    pub fn perturb(c: *DecPost, d: usize, on: bool) !void {
        try flipBf16(c.r, c.sums.sums + (d - 1) * DM_W * 2, on, &c.saved);
    }
    pub fn prep(c: *DecPost, d: usize) !void {
        try c.sums.prep(d);
    }
    pub fn reset(c: *DecPost) !void {
        try c.r.d.check(c.r.d.api.cuMemcpyDtoDAsync_v2(c.h, c.h_in, dec_max * DM_RH, c.r.s.handle), "cuMemcpyDtoDAsync");
    }
    pub fn whole(c: *DecPost, d: usize) !void {
        try dsv41.tri_basic.hcPost(c.r.t, c.sums.planes, c.h, c.post, c.comb, c.h, 2, d, DM_D);
    }
    pub fn slab(c: *DecPost, d: usize, k: usize, s: usize, m: usize) !void {
        const q = dsv41.slab_plan.q(d);
        try dsv41.tri_basic.hcPost2b(c.r.t, c.sums.blk[k], (q - c.short) * DM_W, c.h + s * DM_RH, c.post + s * 16, c.comb + s * 64, c.h + s * DM_RH, m, DM_D);
    }
    pub fn after(c: *DecPost, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn read(c: *DecPost, d: usize) ![]u8 {
        return get(c.h, d * DM_RH, c.r.d);
    }
};

/// A DSpark tap's mean of the streams (hcMean4) into the tap's rows: the window's d rows in one launch, or each slab's.
const DecMean = struct {
    r: *Run,
    ex: *const dsv41.exact.Exact,
    h: u64,
    out: u64,
    saved: u16 = 0,

    pub const strided = false;

    /// The window's last row's first stream value.
    pub fn perturb(c: *DecMean, d: usize, on: bool) !void {
        try flipBf16(c.r, c.h + (d - 1) * DM_RH, on, &c.saved);
    }

    pub fn prep(c: *DecMean, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn reset(c: *DecMean) !void {
        try poison(c.out, dec_max * DM_D * 2, c.r.d, c.r.s);
    }
    pub fn whole(c: *DecMean, d: usize) !void {
        try c.ex.hcMean4(c.r.s, c.h, c.out, DM_D, d, DM_D);
    }
    pub fn slab(c: *DecMean, d: usize, k: usize, s: usize, m: usize) !void {
        _ = d;
        _ = k;
        try c.ex.hcMean4(c.r.s, c.h + s * DM_RH, c.out + s * DM_D * 2, DM_D, m, DM_D);
    }
    pub fn after(c: *DecMean, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn read(c: *DecMean, d: usize) ![]u8 {
        return get(c.out, d * DM_D * 2, c.r.d);
    }
};

/// The experts' gate / up at a window of d rows: the baseline mx.gateUp quantizing the d rows itself; the owned path's act_quant of each slab's rows
/// into the scratch (ffnRows) and mx.gateUpCodes on the codes the slabs left.
const DecExperts = struct {
    r: *Run,
    ex: weights.Experts,
    sc: mx.ExpertScratch,
    x: u64,
    pick: u64,
    wts: u64,
    hb: u64,
    saved: u16 = 0,

    pub const strided = false;

    /// The window's last row's first activation value (its codes and its block's scale change).
    pub fn perturb(c: *DecExperts, d: usize, on: bool) !void {
        try flipBf16(c.r, c.x + (d - 1) * X_D * 2, on, &c.saved);
    }

    pub fn prep(c: *DecExperts, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn reset(c: *DecExperts) !void {
        try poison(c.sc.q, dec_max * X_D, c.r.d, c.r.s);
        try poison(c.sc.s, dec_max * (X_D / 32) * 4, c.r.d, c.r.s);
        try poison(c.hb, dec_max * X_SL * X_W * 2, c.r.d, c.r.s);
    }
    pub fn whole(c: *DecExperts, d: usize) !void {
        try mx.gateUp(c.r.k, c.r.s, c.r.d, c.ex, c.sc, c.x, X_D, c.pick, c.wts, d, 7.0, c.hb, X_MAIN, false);
    }
    pub fn slab(c: *DecExperts, d: usize, k: usize, s: usize, m: usize) !void {
        _ = d;
        _ = k;
        try mx.quant(c.r.k, c.r.s, c.x + s * X_D * 2, .bf16, X_D, m, X_D, c.sc.q + s * X_D, X_D, c.sc.s + s * (X_D / 32) * 4, X_D / 32);
    }
    pub fn after(c: *DecExperts, d: usize) !void {
        try mx.gateUpCodes(c.r.k, c.r.s, c.r.d, c.ex, c.sc, c.pick, c.wts, d, 7.0, c.hb, X_MAIN);
    }
    pub fn read(c: *DecExperts, d: usize) ![]u8 {
        return get(c.hb, d * X_SL * X_W * 2, c.r.d);
    }
};

// Every decoder class's interface (replay_runner.zig), public: the runner calls each method on every class (a private method is invisible
// to @hasDecl, and a dispatch by it skips the method silently).
comptime {
    for ([_]type{ DecMix, DecPost, DecMean, DecExperts }) |T| for ([_][]const u8{ "prep", "reset", "whole", "slab", "after", "read", "perturb", "strided" }) |m| {
        if (!@hasDecl(T, m)) @compileError("the decoder class " ++ @typeName(T) ++ " has no public " ++ m);
    };
}

/// A decoder class at every window of dec_ds through the runner (replay_runner.runClass: the null control before the comparison, the
/// controls); a failing step names the section, the class, the window, the slab and the step.
fn decClassRun(r: *Run, section: []const u8, name: []const u8, c: anytype) !dsv41.replay_runner.Result {
    var fail: dsv41.replay_runner.Failure = .{};
    return dsv41.replay_runner.runClass(r.s, c, &dec_ds, &fail) catch |e| {
        if (fail.slab) |sl| {
            std.debug.print("{s} {s}: FAILED at window d = {d}, slab {d} (start {d}, {d} rows), {s}: {s}\n", .{ section, name, fail.d, sl[0], sl[1], sl[2], fail.step, @errorName(e) });
        } else std.debug.print("{s} {s}: FAILED at window d = {d}, {s}: {s}\n", .{ section, name, fail.d, fail.step, @errorName(e) });
        return e;
    };
}

/// A decoder class's lines: the comparison, the null control (named failure: which run, at which window), the control and, strided, the
/// stride control.
fn decReport(r: *Run, section: []const u8, what: []const u8, res: dsv41.replay_runner.Result, stride: bool) !void {
    var line: [512]u8 = undefined;
    const name = what[0..std.mem.indexOfScalar(u8, what, ':').?];
    r.report(try std.fmt.bufPrint(&line, "{s} {s}, at {s}", .{ section, what, dec_plan.text }), res.bad, true);
    if (res.first_null) |f| {
        r.note(try std.fmt.bufPrint(&line, "{s} {s} null control: both runs wrote their outputs and the baseline launch reads the window's inputs, in {d} of {d} windows; first at d = {d}: {s}", .{ section, name, res.null_ok, res.windows, f.d, dsv41.replay_runner.nullText(f.why) }), false);
    } else r.note(try std.fmt.bufPrint(&line, "{s} {s} null control: both runs wrote their outputs and the baseline launch reads the window's inputs, in all {d} windows", .{ section, name, res.windows }), res.null_ok == res.windows);
    r.report(try std.fmt.bufPrint(&line, "{s} {s} control: the last slab one row short, in all {d} windows", .{ section, name, dec_ds.len }), if (res.cut_ok == dec_ds.len) 1 else 0, false);
    if (stride) r.report(try std.fmt.bufPrint(&line, "{s} {s} stride control: the slabs' blocks read q - 1 rows apart, in all {d} windows", .{ section, name, dec_ds.len }), if (res.stride_ok == dec_ds.len) 1 else 0, false);
}

/// The decoder half on owned rows, the launch classes the encoder half's sections do not cover (roci_dec_rows runs those at the windows):
/// the posted mixes and the posts from the transport's two blocks at a window's plan (stride q(d), the last slab short) against the baseline
/// readers of the quarters' planes over the window's rows, a DSpark tap's mean, and the experts' gate / up from the slabs' codes, at every
/// window d = 1 to 128; each with its control.
fn rociDecMixes(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const section = "roci_dec_mixes";
    var prng = std.Random.DefaultPrng.init(0xdec0);
    const rng = prng.random();
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const M = dec_max;
    const h = try upload(d, try bf16s(M * 4 * DM_D, rng, 1.0, ha));
    try keep.append(ha, h);
    const mix_what = "hcPrePf: _hc_mix_pf_2b + _hc_finish on each slab's two blocks at the plan's stride q against _hc_mix_pf + _hc_finish on the planes over the window's d rows (the streams, x, pre, post, comb)";
    const post_what = "hcPost: _hc_post_2b in place on each slab's rows from its two blocks against _hc_post on the planes over the window's d rows";
    if (r.t.extra == null or r.ops == null) {
        for ([_][]const u8{ "hcPrePf", "hcPost" }) |nm| for ([_][]const u8{ "", " null control", " control", " stride control" }) |sfx| {
            var nl: [160]u8 = undefined;
            r.note(try std.fmt.bufPrint(&nl, "{s} {s}{s}: the kit's extra set (aot-extra/) loads and the ops are available", .{ section, nm, sfx }), false);
        };
    } else {
        var sums: DecSums = .{ .r = r, .sums = 0, .planes = 0, .blk = undefined };
        const sb = try upload(d, try bf16s(2 * M * DM_W, rng, 1.0, ha));
        try keep.append(ha, sb);
        sums.sums = sb.ptr;
        const pl = try cuda.DeviceBuffer.alloc(d, 2 * M * DM_D * 4);
        try keep.append(ha, pl);
        sums.planes = pl.ptr;
        for (&sums.blk) |*b| {
            const bb = try cuda.DeviceBuffer.alloc(d, 2 * 32 * DM_W * 2);
            try keep.append(ha, bb);
            b.* = bb.ptr;
        }
        // the mixes' weights small enough that the Sinkhorn finish stays finite
        const fnw = try upload(d, try f32sAt(24 * 4 * DM_D, rng, 0.01, ha));
        const scale = try upload(d, try f32sAt(3, rng, 1.0, ha));
        const base = try upload(d, try f32sAt(24, rng, 1.0, ha));
        const pre_in = try upload(d, try f32sAt(M * 4, rng, 0.5, ha));
        const nw = try upload(d, try bf16s(DM_D, rng, 1.0, ha));
        const post_in = try upload(d, try f32sAt(M * 4, rng, 0.5, ha));
        const comb_in = try upload(d, try f32sAt(M * 16, rng, 0.5, ha));
        for ([_]cuda.DeviceBuffer{ fnw, scale, base, pre_in, nw, post_in, comb_in }) |b| try keep.append(ha, b);
        var outs: [6]u64 = undefined;
        for (&outs, [_]usize{ M * DM_RH, M * DM_D * 2, M * 16, M * 16, M * 64, M * DecMix.part_row }) |*o, n| {
            const b = try cuda.DeviceBuffer.alloc(d, n);
            try keep.append(ha, b);
            o.* = b.ptr;
        }
        var mix: DecMix = .{ .r = r, .sums = &sums, .h = h.ptr, .fnw = fnw.ptr, .scale = scale.ptr, .base = base.ptr, .pre_in = pre_in.ptr, .nw = nw.ptr, .post_in = post_in.ptr, .comb_in = comb_in.ptr, .h_out = outs[0], .x = outs[1], .pre_o = outs[2], .post = outs[3], .comb = outs[4], .part = outs[5] };
        try decReport(r, section, mix_what, try decClassRun(r, section, "hcPrePf", &mix), true);
        const hw = try cuda.DeviceBuffer.alloc(d, M * DM_RH);
        try keep.append(ha, hw);
        var post: DecPost = .{ .r = r, .sums = &sums, .h_in = h.ptr, .h = hw.ptr, .post = post_in.ptr, .comb = comb_in.ptr };
        try decReport(r, section, post_what, try decClassRun(r, section, "hcPost", &post), true);
    }
    // a tap's mean
    const mean_what = "hcMean4 (a DSpark tap's mean): each slab's rows against the window's one launch";
    if (dsv41.exact.Exact.load(d, cuda.kernels.dsv41_torch)) |ex_loaded| {
        var ex = ex_loaded;
        defer ex.unload();
        const out = try cuda.DeviceBuffer.alloc(d, M * DM_D * 2);
        try keep.append(ha, out);
        var mean: DecMean = .{ .r = r, .ex = &ex, .h = h.ptr, .out = out.ptr };
        try decReport(r, section, mean_what, try decClassRun(r, section, "hcMean4", &mean), false);
    } else |e| {
        var nl: [160]u8 = undefined;
        r.note(try std.fmt.bufPrint(&nl, "{s} hcMean4: the torch kernels load ({s})", .{ section, @errorName(e) }), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcMean4 null control: the torch kernels load", .{section}), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcMean4 control: the torch kernels load", .{section}), false);
    }
    // the experts' gate / up
    const ex = try xExperts(d, rng, ha, &keep);
    const sc = try xScratch(d, M, true, ha, &keep);
    const x = try upload(d, try bf16s(M * X_D, rng, 0.5, ha));
    const pw = try xPicks(M, rng, ha);
    const pb = try upload(d, std.mem.sliceAsBytes(pw.pick));
    const wb = try upload(d, std.mem.sliceAsBytes(pw.wts));
    const hb = try cuda.DeviceBuffer.alloc(d, M * X_SL * X_W * 2);
    for ([_]cuda.DeviceBuffer{ x, pb, wb, hb }) |b| try keep.append(ha, b);
    var gu: DecExperts = .{ .r = r, .ex = ex, .sc = sc, .x = x.ptr, .pick = pb.ptr, .wts = wb.ptr, .hb = hb.ptr };
    try decReport(r, section, "gate / up: act_quant at each slab's rows and mx.gateUpCodes on the slabs' codes against mx.gateUp's own quantization of the window's d rows", try decClassRun(r, section, "gate / up", &gu), false);
}

/// The kit's base and extra sets, looked up as a launch finds its variant (Set.lookup: find's variant, without its log); `hide` a name the
/// extra set is taken not to hold (the control).
const KitFinds = struct {
    set: *const cuda.aot.Set,
    extra: ?*const cuda.aot.Set,
    hide: ?[]const u8 = null,

    pub fn has(f: KitFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        return f.set.lookup(name, args, consts) != null;
    }
    pub fn hasExtra(f: KitFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        if (f.hide) |x| if (std.mem.eql(u8, x, name)) return false;
        const set = f.extra orelse return false;
        return set.lookup(name, args, consts) != null;
    }
};

/// The decoder half's start check on this kit (any_check.decoderOff: every launch of layers 0-39 at every slab of a decoder window of 1 to 128
/// rows, the two-block readers in the extra set), TF_DS_ROCI_ANY's (layers 0-19, chunks of 1 to 4,096 rows) with the readers looked up as well,
/// and the control: the extra set taken to lack _hc_post_2b, which the decoder half's check must name.
fn rociDecKset(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const section = "roci_dec_kset";
    var line: [512]u8 = undefined;
    const gpa = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var why: dsv41.config.Why = .{};
    const cfg = try dsv41.config.parse(a, dsv41.config.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    dsv41.any_check.layers(&lays, cfg);
    const finds: KitFinds = .{ .set = r.t.set.?, .extra = r.t.extra };
    const dec = try dsv41.any_check.decoderOff(gpa, a, finds, cfg, &lays);
    r.note(try std.fmt.bufPrint(&line, "{s}: the decoder half's start check on this kit (every launch of layers 0-39 at every slab of a decoder window of 1 to 128 rows, the two-block readers in the extra set): {s}", .{ section, dec orelse "every variant found" }), dec == null);
    const enc = try dsv41.any_check.anyOwnedOff(gpa, a, finds, cfg, &lays, cfg.layers / 2, true, 4096);
    r.note(try std.fmt.bufPrint(&line, "{s}: TF_DS_ROCI_ANY's start check on this kit (layers 0-19, chunks of 1 to 4,096 rows) with the two-block readers looked up: {s}", .{ section, enc orelse "every variant found" }), enc == null);
    const ctl = try dsv41.any_check.decoderOff(gpa, a, KitFinds{ .set = r.t.set.?, .extra = r.t.extra, .hide = "_hc_post_2b" }, cfg, &lays);
    r.note(try std.fmt.bufPrint(&line, "{s} control: the extra set taken to lack _hc_post_2b, the decoder half's check names it", .{section}), ctl != null and std.mem.indexOf(u8, ctl.?, "_hc_post_2b") != null);
}

// -- wo_b and the MoE's routed sum with the sent quarters first, in one launch ------------------------------------------------

/// The noise bar of a null control: the rig's measured noise floor. A time line is "as it must" (its instrument holds) while the null
/// control, the same launch again, reads within it; a verdict (a kill line crossed, a difference outside the null) is in the text.
const null_bar: f32 = 5.0;

/// ms a launch: the best of three measurements of 20 launches.
fn kernelBest(r: *Run, comptime Ctx: type, ctx: Ctx, comptime f: fn (Ctx) anyerror!void) !f32 {
    var best = try kernelMs(r, Ctx, ctx, f, 20);
    for (0..2) |_| best = @min(best, try kernelMs(r, Ctx, ctx, f, 20));
    return best;
}

/// "within" / "outside" the null control's reading, for a text.
fn resolved(null_pct: f32) []const u8 {
    return if (@abs(null_pct) <= 2.0) "resolved" else if (@abs(null_pct) <= null_bar) "noisy: differences under the null control's reading are not resolved" else "NOT resolved";
}

const sf_m: usize = 4096;
const sf_k: usize = 4096;
const sf_n: usize = 2560;
const sf_rt: u32 = sf_m / 128; // row tiles
const sf_ct: u32 = sf_n / 128; // column tiles

/// A launch of wo_b's block-scaled exact GEMM (K 4,096, N 2,560, 4,096 rows, y in the arena) in `ro`'s order: `f` the kernel that
/// takes (MxArgs, RowOrder), plus the Door when `door` and the stamps when `ts`.
const SentRun = struct {
    r: *Run,
    f: cuda.Function,
    a: mx.MxArgs,
    ro: mx.RowOrder,
    door: ?mx.Door = null,
    /// The timed twins' stamps [CTA in launch order][begin, end] (u64 ns) and, past them, the ring's; 0: untimed.
    ts: u64 = 0,

    fn go(c: @This()) anyerror!void {
        var args: cuda.Args = .{};
        args.add(c.a);
        args.add(c.ro);
        if (c.door) |dr| args.add(dr);
        if (c.ts != 0) args.add(c.ts);
        try cuda.launch.launch(c.f, .{ .grid = .{ .x = sf_rt, .y = sf_ct, .z = 1 }, .block = .{ .x = 256 }, .shared = mx.sf_split128_smem }, c.r.s, &args);
    }
};

/// The baseline launch: mx.gemm with y in the arena (the whole-line kernel).
const TodayRun = struct {
    r: *Run,
    l: weights.Linear,
    q: u64,
    s: u64,
    z: u64,
    y: u64,

    fn go(c: @This()) anyerror!void {
        try mx.gemm(c.r.k, c.r.s, c.l, c.q, sf_k, c.s, sf_k / 32, sf_m, c.y, sf_n, .f32, false, c.z);
    }
};

/// The span of a timed launch (stamps [begin, end] per CTA in launch order, and past them the ring's) in %: the CTA `n` of them
/// that ended last gives 100; `upto` is the time to place (a CTA set's last end, or the ring's stamp). Null when a CTA did not stamp.
fn stampSpan(ts: []const u64, n: usize, upto: u64) ?f64 {
    var t0: u64 = std.math.maxInt(u64);
    var t1: u64 = 0;
    for (0..n) |i| {
        const b = ts[2 * i];
        const e = ts[2 * i + 1];
        if (b == 0 or e < b) return null;
        t0 = @min(t0, b);
        t1 = @max(t1, e);
    }
    if (upto < t0 or t1 <= t0) return null;
    return 100.0 * @as(f64, @floatFromInt(upto - t0)) / @as(f64, @floatFromInt(t1 - t0));
}

/// The last end among the CTAs `sent[i]` names.
fn lastEnd(ts: []const u64, sent: []const bool) u64 {
    var e: u64 = 0;
    for (sent, 0..) |s, i| {
        if (s) e = @max(e, ts[2 * i + 1]);
    }
    return e;
}

fn u64s(raw: []u8) []const u64 {
    return @alignCast(std.mem.bytesAsSlice(u64, raw));
}

/// The wo_b launch's operands: the arena y blocks and the quantized operand, built once for the two sections.
const SfRig = struct {
    l: weights.Linear,
    q: u64,
    s: u64,
    z: u64,
    margs: mx.MxArgs,
    wc: u64,
    y_ref: u64,
    y_new: u64,
    y_null: u64,
};

fn sfRig(r: *Run, rng: std.Random, ha: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer), arena_mem: *?*anyopaque) !?SfRig {
    const d = r.d;
    const ybytes = sf_m * sf_n * 4;
    try d.check(d.api.cuMemHostAlloc(arena_mem, 3 * ybytes, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem.*, 0), "cuMemHostGetDevicePointer");
    try r.k.useStaged(wc, wc + 3 * ybytes);
    const l = try fp8Linear(d, sf_n, sf_k, .fast, rng, ha, keep);
    const x = try upload(d, try bf16s(sf_m * sf_k, rng, 0.5, ha));
    const q = try cuda.DeviceBuffer.alloc(d, sf_m * sf_k);
    const qs = try cuda.DeviceBuffer.alloc(d, sf_m * (sf_k / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, sf_m), 1) * 4);
    for ([_]cuda.DeviceBuffer{ x, q, qs, z }) |b| try keep.append(ha, b);
    try mx.quant(r.k, r.s, x.ptr, .bf16, sf_k, sf_m, sf_k, q.ptr, sf_k, qs.ptr, sf_k / 32);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    const kind = mx.sfRoute(r.k, l, q.ptr, sf_k, qs.ptr, sf_k / 32, wc, sf_n, false, sf_m);
    if (kind == null or kind.? != .split128) return null;
    const sp = mx.splits(sf_k, sf_n);
    const margs: mx.MxArgs = .{ .q = q.ptr, .s = qs.ptr, .ldq = sf_k, .lds = sf_k / 32, .w = l.w, .ws = l.s, .ldw = sf_k, .ldws = sf_k / 32, .fmt = 0, .M = sf_m, .N = sf_n, .K = sf_k, .splits = @intCast(sp), .kblocks = @intCast((sf_k / 32 + sp - 1) / sp), .z = z.ptr, .y = wc + ybytes, .ldy = sf_n, .y_dtype = @backingInt(mx.DType.f32), .accumulate = 0 };
    return .{ .l = l, .q = q.ptr, .s = qs.ptr, .z = z.ptr, .margs = margs, .wc = wc, .y_ref = wc, .y_new = wc + ybytes, .y_null = wc + 2 * ybytes };
}

/// wo_b's block-scaled exact GEMM (K 4,096, N 2,560, 4,096 rows, y in the arena's write-combined host memory: the whole-line
/// kernel) in one launch whose CTAs take the row tiles of the two quarters a node's first-part send takes before the other two's
/// (tf_dsn_sf_split128_exact_st_first: a row order, every column of the sent quarters first), against the baseline launch: ms, bytes,
/// and when the sent quarters are done, from each CTA's %globaltimer at its begin and end (the timed twin).
fn rociSentFirst(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x5e17f1);
    const rng = prng.random();
    const d = r.d;
    const sk = r.k.sfk orelse return r.note("roci_sent_first: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    const f_first = try sk.module.function("tf_dsn_sf_split128_exact_st_first_kernel");
    try f_first.allowDynamicShared(mx.sf_split128_smem);
    const f_time = try sk.module.function("tf_dsn_sf_split128_exact_st_first_t_kernel");
    try f_time.allowDynamicShared(mx.sf_split128_smem);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var arena_mem: ?*anyopaque = null;
    defer if (arena_mem != null) {
        _ = d.api.cuMemFreeHost(arena_mem);
    };
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    const ybytes = sf_m * sf_n * 4;
    const rig = (try sfRig(r, rng, ha, &keep, &arena_mem)) orelse return r.note("roci_sent_first: wo_b's 4,096 rows take tf_dsn_sf_split128 here", false);
    const stamps = try cuda.DeviceBuffer.alloc(d, (sf_rt * sf_ct * 2 + 1) * 8);
    try keep.append(ha, stamps);
    const q = sf_m / 4;
    const ident = mx.identityRowOrder(sf_m, sf_ct).?;
    var orders: [4]mx.RowOrder = undefined;
    for (0..4) |g| orders[g] = mx.rowOrder(.{ (g ^ 1) * q, (g ^ 3) * q }, .{ g * q, (g ^ 2) * q }, q, sf_ct).?;
    // a permutation that is not one: quarter g twice, quarter g ^ 2 never
    var dup = orders[0];
    dup.quarters = (dup.quarters & 0x00ffffff) | (@as(u32, 0) << 24);
    const today: TodayRun = .{ .r = r, .l = rig.l, .q = rig.q, .s = rig.s, .z = rig.z, .y = rig.y_ref };
    var today_n = today;
    today_n.y = rig.y_null;
    const sent_run: SentRun = .{ .r = r, .f = f_first, .a = rig.margs, .ro = orders[0] };
    var id_run = sent_run;
    id_run.ro = ident;
    var t_sent = sent_run;
    t_sent.f = f_time;
    t_sent.ts = stamps.ptr;
    var t_id = id_run;
    t_id.f = f_time;
    t_id.ts = stamps.ptr;
    const ms_t = try kernelBest(r, TodayRun, today, TodayRun.go);
    const ms_n = try kernelBest(r, TodayRun, today_n, TodayRun.go);
    const ms_s = try kernelBest(r, SentRun, sent_run, SentRun.go);
    const ms_i = try kernelBest(r, SentRun, id_run, SentRun.go);
    const ms_ts = try kernelBest(r, SentRun, t_sent, SentRun.go);
    const ms_ti = try kernelBest(r, SentRun, t_id, SentRun.go);
    const null_pct = pct(ms_n, ms_t);
    const bar = @max(1.0, @abs(null_pct));
    var t: WcText = .{};
    try t.add("roci_sent_first wo_b (K {d}, N {d}, 4,096 rows, y in the arena) block-scaled exact, kernel only, best of three, ms: the baseline launch {d:.3} (null control, again, {d:.1} %: {s}); ", .{ sf_k, sf_n, ms_t, null_pct, resolved(null_pct) });
    try t.add("one launch with the sent quarters' row tiles first {d:.3} ({d:.1} % over the baseline's, {s} the null control's bar of {d:.1} %); the same kernel in the baseline order {d:.3} ({d:.1} %); ", .{ ms_s, pct(ms_s, ms_t), if (@abs(pct(ms_s, ms_t)) <= bar) "within" else "OUTSIDE", bar, ms_i, pct(ms_i, ms_t) });
    try t.add("the timed twins (a %globaltimer read at a CTA's begin and end): sent-first {d:.3} ({d:.1} % over the untimed), the baseline order {d:.3} ({d:.1} %)", .{ ms_ts, pct(ms_ts, ms_s), ms_ti, pct(ms_ti, ms_i) });
    r.note(t.buf[0..t.n], @abs(null_pct) <= null_bar);
    // when the sent quarters are done: 5 launches of each order, the stamps read back
    var sent_first_l: [sf_rt * sf_ct]bool = undefined;
    var sent_id_l: [sf_rt * sf_ct]bool = undefined;
    for (0..sf_rt * sf_ct) |i| {
        sent_first_l[i] = i < ident.span / 2;
        const quarter = (i % sf_rt) / 8;
        sent_id_l[i] = quarter == 1 or quarter == 3;
    }
    var span: [2][5]f64 = undefined;
    var sane = true;
    var res_ns: u64 = std.math.maxInt(u64);
    for ([_]SentRun{ t_sent, t_id }, 0..) |run, ki| for (0..5) |rep| {
        try d.check(d.api.cuMemsetD8Async(stamps.ptr, 0, sf_rt * sf_ct * 16, r.s.handle), "cuMemsetD8Async");
        try run.go();
        try r.s.synchronize();
        const raw = try get(stamps.ptr, sf_rt * sf_ct * 16, d);
        defer free(raw);
        const ts = u64s(raw);
        const sent = if (ki == 0) &sent_first_l else &sent_id_l;
        const sc_v = stampSpan(ts, sf_rt * sf_ct, lastEnd(ts, sent));
        sane = sane and sc_v != null;
        span[ki][rep] = sc_v orelse 0;
        // the stamps' resolution: the smallest difference between two different stamps
        var sorted: [sf_rt * sf_ct * 2]u64 = undefined;
        @memcpy(&sorted, ts[0 .. sf_rt * sf_ct * 2]);
        std.mem.sort(u64, &sorted, {}, std.sort.asc(u64));
        for (sorted[1..], sorted[0 .. sorted.len - 1]) |hi, lo| if (hi > lo) {
            res_ns = @min(res_ns, hi - lo);
        };
    };
    for (&span) |*s_| std.mem.sort(f64, s_, {}, std.sort.asc(f64));
    var t2: WcText = .{};
    try t2.add("roci_sent_first wo_b: the last CTA of the two sent quarters ends at {d:.1} % of the launch (median of 5, {d:.1} to {d:.1}) with the sent quarters first; with the baseline order it ends at {d:.1} % ({d:.1} to {d:.1}); ", .{ span[0][2], span[0][0], span[0][4], span[1][2], span[1][0], span[1][4] });
    try t2.add("the stamps: {d} CTAs, smallest step between two stamps {d} ns. The send could start there, less the stores' completion and the fence", .{ sf_rt * sf_ct, res_ns });
    r.note(t2.buf[0..t2.n], sane and span[1][2] >= 90.0);
    // the bytes
    try poison(rig.y_ref, ybytes, d, r.s);
    try today.go();
    try r.s.synchronize();
    const ref = try get(rig.y_ref, ybytes, d);
    defer free(ref);
    var bad: usize = 0;
    for (0..4) |g| {
        var run = sent_run;
        run.ro = orders[g];
        try poison(rig.y_new, ybytes, d, r.s);
        try run.go();
        try r.s.synchronize();
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(ref, got);
    }
    r.report("roci_sent_first wo_b: the sent-first launch of each of the four nodes (quarters g ^ 1, g ^ 3 first) against the baseline launch", bad, true);
    bad = 0;
    for ([_]SentRun{ id_run, t_sent, t_id }) |run| {
        try poison(rig.y_new, ybytes, d, r.s);
        try run.go();
        try r.s.synchronize();
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(ref, got);
    }
    r.report("roci_sent_first wo_b: the map kernel with the baseline order, and the timed twins in both orders, against the baseline launch", bad, true);
    var cut = sent_run;
    cut.ro = dup;
    try poison(rig.y_new, ybytes, d, r.s);
    try cut.go();
    try r.s.synchronize();
    const got = try get(rig.y_new, ybytes, d);
    defer free(got);
    r.report("roci_sent_first wo_b control: a row order that runs one quarter twice and another never", differs(ref, got), false);
}

/// The doorbell line a test kernel rings (host-mapped memory, not write-combined, as the transport's control block), the
/// arrival count (a device word, zero between launches) and the observer's two result words.
const DoorRig = struct {
    d: *const cuda.Driver,
    mem: ?*anyopaque = null,
    dev: u64 = 0,
    cnt: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    ticket: u64 = 0,

    fn init(d: *const cuda.Driver) !DoorRig {
        var g: DoorRig = .{ .d = d, .cnt = try cuda.DeviceBuffer.alloc(d, 256), .out = try cuda.DeviceBuffer.alloc(d, 256) };
        errdefer g.cnt.free();
        errdefer g.out.free();
        try d.check(d.api.cuMemHostAlloc(&g.mem, 4096, 0x01 | 0x02), "cuMemHostAlloc");
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&g.dev, g.mem, 0), "cuMemHostGetDevicePointer");
        @memset(@as([*]u8, @ptrCast(g.mem.?))[0..4096], 0);
        return g;
    }

    fn deinit(g: *DoorRig) void {
        _ = g.d.api.cuMemFreeHost(g.mem);
        g.cnt.free();
        g.out.free();
    }

    fn line(g: *const DoorRig) *volatile u64 {
        return @ptrCast(@alignCast(g.mem.?));
    }

    /// The next ticket, the line cleared; the Door a producer rings.
    fn arm(g: *DoorRig) mx.Door {
        g.ticket += 1;
        g.line().* = 0;
        return .{ .cnt = g.cnt.ptr, .door = g.dev, .ticket = g.ticket };
    }

    /// Whether the line shows the ticket and the arrival count is zero again: the producer rang once and left the count clean.
    fn rang(g: *const DoorRig) !bool {
        var c: u32 = 1;
        return g.line().* == g.ticket and c == 0;
    }
};

/// One producer launch, observed: the observer kernel (a CTA on `sw`) is started and given a moment to be waiting; then the
/// producer `P.go(p)` runs on the run's stream (it must ring `rig.ticket`); once both have ended the observer's count of words that
/// differed from their references at the moment the doorbell showed, and whether it showed.
fn observed(r: *Run, rig: *DoorRig, wf: cuda.Function, sw: cuda.Stream, ref: [2]u64, got: [2]u64, words: u64, comptime P: type, p: P) !struct { wrong: u32, seen: bool } {
    try r.d.check(r.d.api.cuMemsetD8Async(rig.out.ptr, 0, 256, sw.handle), "cuMemsetD8Async");
    var a: cuda.Args = .{};
    a.add(rig.dev);
    a.add(rig.ticket);
    a.add(ref[0]);
    a.add(got[0]);
    a.add(ref[1]);
    a.add(got[1]);
    a.add(words);
    a.add(rig.out.ptr);
    a.add(@as(u64, 3_000_000_000));
    try cuda.launch.launch(wf, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024 } }, sw, &a);
    std.Io.sleep(r.io.?, .fromMilliseconds(3), .awake) catch {};
    try P.go(p);
    try r.s.synchronize();
    try sw.synchronize();
    var o: [2]u32 = undefined;
    return .{ .wrong = o[0], .seen = o[1] != 0 };
}

/// wo_b as two launches of 2,048 rows (mx.gemm at row offsets).
const HalfRun = struct {
    r: *Run,
    l: weights.Linear,
    q: u64,
    s: u64,
    z: u64,
    y: u64,

    fn go(c: @This()) anyerror!void {
        for ([_]usize{ 0, sf_m / 2 }) |r0| try mx.gemm(c.r.k, c.r.s, c.l, c.q + r0 * sf_k, sf_k, c.s + r0 * (sf_k / 32) * 4, sf_k / 32, sf_m / 2, c.y + r0 * sf_n * 4, sf_n, .f32, false, c.z);
    }
};

/// tf_dsn_combine over all rows: one launch of 4,096 rows, and four of 1,024 (mx.combineRows) in the quarter order of a node.
const CombRun = struct {
    r: *Run,
    sc: mx.ExpertScratch,
    pick: u64,
    y: u64,
    /// 0: one launch; 1: four launches in the order `first`, `last`
    mode: u8,
    first: [2]usize = .{ 0, 0 },
    last: [2]usize = .{ 0, 0 },

    fn go(c: @This()) anyerror!void {
        if (c.mode == 0) return mx.combine(c.r.k, c.r.s, c.sc, c.pick, sf_m, comb_e, sf_n, c.y, sf_n, false);
        for ([_]usize{ c.first[0], c.first[1], c.last[0], c.last[1] }) |row0| try mx.combineRows(c.r.k, c.r.s, c.sc, c.pick, row0, sf_m / 4, comb_e, sf_n, c.y, sf_n, false);
    }
};
const comb_e: usize = 384;
const comb_s: usize = 7;

/// tf_dsn_combine_door over all rows in `qo`'s order (variants: the plain kernel, the early-ring control, the timed twin).
const CombDoor = struct {
    r: *Run,
    f: cuda.Function,
    sc: mx.ExpertScratch,
    pick: u64,
    y: u64,
    qo: mx.QuarterOrder,
    door: mx.Door,
    ts: u64 = 0,

    fn go(c: @This()) anyerror!void {
        var ca: cuda.Args = .{};
        ca.add(c.sc.out);
        ca.add(@as(i64, @intCast(sf_n)));
        ca.add(c.pick);
        ca.add(@as(c_int, @intCast(sf_m)));
        ca.add(@as(c_int, @intCast(comb_s)));
        ca.add(@as(c_int, @intCast(comb_e)));
        ca.add(@as(c_int, @intCast(sf_n)));
        ca.add(c.y);
        ca.add(@as(i64, @intCast(sf_n)));
        ca.add(@as(c_int, 0));
        ca.add(c.qo);
        ca.add(c.door);
        if (c.ts != 0) ca.add(c.ts);
        try cuda.launch.launch(c.f, .{ .grid = .{ .x = @intCast(sf_m * sf_n / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, c.r.s, &ca);
    }
};

/// The doorbell rung from inside the producer (TF_DS_ROCI_SENT_FIRST): wo_b's launch with the sent quarters first
/// (tf_dsn_sf_split128_exact_st_door: the last CTA of the sent quarters rings) and the MoE's routed sum in one launch in the same
/// order (tf_dsn_combine_door), each against the baseline launch (and the combine against the four launches of the quarter order):
/// ms, the ring's place in the launch, bytes for the four nodes, and an observer kernel on a second stream that waits for the
/// doorbell and then reads the sent quarters against the reference: zero words wrong at the ring. Controls: the early-ring kernels
/// (the arrival counted before the CTA's stores) must show words wrong at the ring.
fn rociDoor(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xd007b311);
    const rng = prng.random();
    const d = r.d;
    const sk = r.k.sfk orelse return r.note("roci_door: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    var fns: [6]cuda.Function = undefined;
    for (&fns, [_][:0]const u8{ "tf_dsn_sf_split128_exact_st_door_kernel", "tf_dsn_sf_split128_exact_st_door_early_kernel", "tf_dsn_sf_split128_exact_st_door_t_kernel", "tf_dsn_sf_split128_exact_st_door_b_kernel", "tf_dsn_sf_split128_exact_st_door_b_t_kernel", "tf_dsn_sf_split128_exact_st_door_g_kernel" }) |*f, name| {
        f.* = try sk.module.function(name);
        try f.allowDynamicShared(mx.sf_split128_smem);
    }
    const wf = try r.k.module.function("tf_dsn_door_watch_kernel");
    const cf = try r.k.module.function("tf_dsn_combine_door_kernel");
    const cf_early = try r.k.module.function("tf_dsn_combine_door_early_kernel");
    const cf_time = try r.k.module.function("tf_dsn_combine_door_t_kernel");
    const cf_pt = try r.k.module.function("tf_dsn_combine_door_pt_kernel");
    const cf_g = try r.k.module.function("tf_dsn_combine_door_g_kernel");
    const cf_g_early = try r.k.module.function("tf_dsn_combine_door_g_early_kernel");
    var sw = try cuda.Stream.init(d, true);
    defer sw.deinit();
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var arena_mem: ?*anyopaque = null;
    defer if (arena_mem != null) {
        _ = d.api.cuMemFreeHost(arena_mem);
    };
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    var rigd = try DoorRig.init(d);
    defer rigd.deinit();
    const ybytes = sf_m * sf_n * 4;
    const rig = (try sfRig(r, rng, ha, &keep, &arena_mem)) orelse return r.note("roci_door: wo_b's 4,096 rows take tf_dsn_sf_split128 here", false);
    const q = sf_m / 4;
    const qwords: u64 = q * sf_n; // a quarter's 32-bit words
    const ref_dev = try cuda.DeviceBuffer.alloc(d, ybytes);
    const stamps = try cuda.DeviceBuffer.alloc(d, (sf_rt * sf_ct * 2 + 1) * 8);
    try keep.append(ha, ref_dev);
    try keep.append(ha, stamps);
    // ---- wo_b
    const today: TodayRun = .{ .r = r, .l = rig.l, .q = rig.q, .s = rig.s, .z = rig.z, .y = rig.y_ref };
    var today_n = today;
    today_n.y = rig.y_null;
    try poison(rig.y_ref, ybytes, d, r.s);
    try today.go();
    try r.s.synchronize();
    const ref = try get(ref_dev.ptr, ybytes, d);
    defer free(ref);
    var orders: [4]mx.RowOrder = undefined;
    for (0..4) |g| orders[g] = mx.rowOrder(.{ (g ^ 1) * q, (g ^ 3) * q }, .{ g * q, (g ^ 2) * q }, q, sf_ct).?;
    const mk = struct {
        fn run(rr: *Run, f: cuda.Function, a: mx.MxArgs, ro: mx.RowOrder, door: mx.Door, ts: u64) SentRun {
            return .{ .r = rr, .f = f, .a = a, .ro = ro, .door = door, .ts = ts };
        }
    }.run;
    // ms: the baseline, the null, the plain door kernel (a ring each launch: the tickets only need to be distinct per launch for the
    // timing, so one armed Door serves the 60 launches), the timed twin
    const door0 = rigd.arm();
    const ms_t = try kernelBest(r, TodayRun, today, TodayRun.go);
    const ms_n = try kernelBest(r, TodayRun, today_n, TodayRun.go);
    const ms_d = try kernelBest(r, SentRun, mk(r, fns[0], rig.margs, orders[0], door0, 0), SentRun.go);
    const ms_dt = try kernelBest(r, SentRun, mk(r, fns[2], rig.margs, orders[0], door0, stamps.ptr), SentRun.go);
    const ms_db = try kernelBest(r, SentRun, mk(r, fns[3], rig.margs, orders[0], door0, 0), SentRun.go);
    const ms_dg = try kernelBest(r, SentRun, mk(r, fns[5], rig.margs, orders[0], door0, 0), SentRun.go);
    var halves: HalfRun = .{ .r = r, .l = rig.l, .q = rig.q, .s = rig.s, .z = rig.z, .y = rig.y_new };
    const ms_h = try kernelBest(r, HalfRun, halves, HalfRun.go);
    halves.y = rig.y_null;
    const ms_hn = try kernelBest(r, HalfRun, halves, HalfRun.go);
    {
        var th: WcText = .{};
        try th.add("roci_door wo_b as two launches of 2,048 rows (rows 0 .. 2,047 then 2,048 .. 4,095: the sent half then the kept half in time, whatever rows they are), kernel only, best of three, ms: one launch of 4,096 rows {d:.3}; two of 2,048 rows {d:.3} ({d:.1} % over one launch; null control, the pair again, {d:.1} %: {s}); for comparison the sent-first order in one launch with the door {d:.3} ({d:.1} %), with the count released at GPU scope {d:.3} ({d:.1} %)", .{ ms_t, ms_h, pct(ms_h, ms_t), pct(ms_hn, ms_h), resolved(pct(ms_hn, ms_h)), ms_d, pct(ms_d, ms_t), ms_dg, pct(ms_dg, ms_t) });
        r.note(th.buf[0..th.n], @abs(pct(ms_hn, ms_h)) <= null_bar);
    }
    // where the ring falls: 5 timed launches, the ring's stamp past the CTAs'
    var ring_pct: [5]f64 = undefined;
    var ring_b: [5]f64 = undefined;
    var sane = true;
    for (0..5) |rep| {
        try d.check(d.api.cuMemsetD8Async(stamps.ptr, 0, (sf_rt * sf_ct * 2 + 1) * 8, r.s.handle), "cuMemsetD8Async");
        const dr = rigd.arm();
        try mk(r, fns[4], rig.margs, orders[0], dr, stamps.ptr).go();
        try r.s.synchronize();
        {
            const raw = try get(stamps.ptr, (sf_rt * sf_ct * 2 + 1) * 8, d);
            defer free(raw);
            const ts = u64s(raw);
            ring_b[rep] = stampSpan(ts, sf_rt * sf_ct, ts[sf_rt * sf_ct * 2]) orelse 0;
            sane = sane and ring_b[rep] != 0 and try rigd.rang();
        }
        try d.check(d.api.cuMemsetD8Async(stamps.ptr, 0, (sf_rt * sf_ct * 2 + 1) * 8, r.s.handle), "cuMemsetD8Async");
        const dr2 = rigd.arm();
        try mk(r, fns[2], rig.margs, orders[0], dr2, stamps.ptr).go();
        try r.s.synchronize();
        const raw = try get(stamps.ptr, (sf_rt * sf_ct * 2 + 1) * 8, d);
        defer free(raw);
        const ts = u64s(raw);
        const v = stampSpan(ts, sf_rt * sf_ct, ts[sf_rt * sf_ct * 2]);
        sane = sane and v != null and try rigd.rang();
        ring_pct[rep] = v orelse 0;
    }
    std.mem.sort(f64, &ring_pct, {}, std.sort.asc(f64));
    std.mem.sort(f64, &ring_b, {}, std.sort.asc(f64));
    const null_pct = pct(ms_n, ms_t);
    var t: WcText = .{};
    try t.add("roci_door wo_b (K {d}, N {d}, 4,096 rows, y in the arena), kernel only, best of three, ms: the baseline launch {d:.3} (null control {d:.1} %: {s}); the door kernel (sent quarters first, the ring from inside) {d:.3} ({d:.1} % over the baseline's); its timed twin {d:.3} ({d:.1} % over the plain one); the door kernel with the block-level release {d:.3} ({d:.1} % over the baseline's, {d:.1} % over the per-thread one); with the count released at GPU scope and the system fence by the last arrival alone {d:.3} ({d:.1} % over the baseline's); ", .{ sf_k, sf_n, ms_t, null_pct, resolved(null_pct), ms_d, pct(ms_d, ms_t), ms_dt, pct(ms_dt, ms_d), ms_db, pct(ms_db, ms_t), pct(ms_db, ms_d), ms_dg, pct(ms_dg, ms_t) });
    try t.add("the ring falls at {d:.1} % of the launch (per-thread release, median of 5, {d:.1} to {d:.1}), {d:.1} % with the block-level release ({d:.1} to {d:.1}); the doorbell showed the ticket and the count was zero after each", .{ ring_pct[2], ring_pct[0], ring_pct[4], ring_b[2], ring_b[0], ring_b[4] });
    r.note(t.buf[0..t.n], @abs(null_pct) <= null_bar and sane);
    // bytes and the observer, the four nodes
    var bad: usize = 0;
    var wrong: u32 = 0;
    var unseen: usize = 0;
    var unclean: usize = 0;
    for (0..4) |g| {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const ro = mk(r, fns[0], rig.margs, orders[g], dr, 0);
        const rows: [2]usize = .{ (g ^ 1) * q, (g ^ 3) * q };
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + rows[0] * sf_n * 4, ref_dev.ptr + rows[1] * sf_n * 4 }, .{ rig.y_new + rows[0] * sf_n * 4, rig.y_new + rows[1] * sf_n * 4 }, qwords, SentRun, ro);
        wrong += o.wrong;
        unseen += @intFromBool(!o.seen);
        unclean += @intFromBool(!(try rigd.rang()));
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(ref, got);
    }
    var line: [512]u8 = undefined;
    r.report(try std.fmt.bufPrint(&line, "roci_door wo_b: the door kernel of each of the four nodes against the baseline launch (the doorbell showed {d} of 4 times, the count clean {d} of 4)", .{ 4 - unseen, 4 - unclean }), bad + unseen + unclean, true);
    r.report("roci_door wo_b: words of the two sent quarters that differed from the reference when the observer saw the doorbell, four nodes", wrong, true);
    // the block-level release and the GPU-scope release, the same four nodes
    for ([_]usize{ 3, 5 }, [_][]const u8{ "the block-level release (thread 0 fences once per CTA)", "the count released at GPU scope and the system fence by the last arrival alone" }) |fi, what| {
        var badb: usize = 0;
        var wrongb: u32 = 0;
        var offb: usize = 0;
        for (0..4) |g| {
            const dr = rigd.arm();
            try poison(rig.y_new, ybytes, d, r.s);
            const rows: [2]usize = .{ (g ^ 1) * q, (g ^ 3) * q };
            const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + rows[0] * sf_n * 4, ref_dev.ptr + rows[1] * sf_n * 4 }, .{ rig.y_new + rows[0] * sf_n * 4, rig.y_new + rows[1] * sf_n * 4 }, qwords, SentRun, mk(r, fns[fi], rig.margs, orders[g], dr, 0));
            wrongb += o.wrong;
            offb += @intFromBool(!o.seen) + @intFromBool(!(try rigd.rang()));
            const got = try get(rig.y_new, ybytes, d);
            defer free(got);
            badb += differs(ref, got);
        }
        r.report(try std.fmt.bufPrint(&line, "roci_door wo_b with {s}: output bytes against the baseline launch, words wrong at the ring, and doorbells not seen or counts not clean, summed over the four nodes", .{what}), badb + wrongb + offb, true);
    }
    // control: the arrival counted before the CTA's stores
    var early_wrong: u32 = 0;
    {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + q * sf_n * 4, ref_dev.ptr + 3 * q * sf_n * 4 }, .{ rig.y_new + q * sf_n * 4, rig.y_new + 3 * q * sf_n * 4 }, qwords, SentRun, mk(r, fns[1], rig.margs, orders[0], dr, 0));
        early_wrong = o.wrong;
        if (!o.seen) early_wrong = 0;
    }
    r.report("roci_door wo_b control: the early-ring kernel (the arrival counted before the CTA's stores), words of the sent quarters that differed when the observer saw the doorbell", early_wrong, false);
    // ---- the routed sum
    const pk = try ha.alloc(i32, sf_m * comb_s);
    for (0..sf_m) |row| {
        var seen: [comb_s]i32 = undefined;
        var n: usize = 0;
        while (n < comb_s) {
            const e: i32 = @intCast(rng.uintLessThan(usize, comb_e));
            if (std.mem.indexOfScalar(i32, seen[0..n], e) != null) continue;
            seen[n] = e;
            pk[row * comb_s + n] = e;
            n += 1;
        }
    }
    // the entries' outputs: a value that differs by entry and column, so a row put in the wrong place shows
    const ev = try ha.alloc(f32, sf_m * comb_s * sf_n);
    for (ev, 0..) |*x, i| x.* = @as(f32, @floatFromInt((i *% 2654435761) >> 12 & 0xffff)) / 256.0;
    const pick = try upload(d, std.mem.sliceAsBytes(pk));
    const outs = try upload(d, std.mem.sliceAsBytes(ev));
    const cstamps = try cuda.DeviceBuffer.alloc(d, (sf_m * sf_n / 256 * 2 + 1) * 8);
    try keep.append(ha, pick);
    try keep.append(ha, outs);
    try keep.append(ha, cstamps);
    var csc = std.mem.zeroes(mx.ExpertScratch);
    csc.out = outs.ptr;
    csc.slots = comb_s;
    const one: CombRun = .{ .r = r, .sc = csc, .pick = pick.ptr, .y = rig.y_ref, .mode = 0 };
    var one_n = one;
    one_n.y = rig.y_null;
    try poison(rig.y_ref, ybytes, d, r.s);
    try one.go();
    try r.s.synchronize();
    const cref = try get(ref_dev.ptr, ybytes, d);
    defer free(cref);
    var qos: [4]mx.QuarterOrder = undefined;
    for (0..4) |g| qos[g] = mx.quarterOrder(.{ (g ^ 1) * q, (g ^ 3) * q }, .{ g * q, (g ^ 2) * q }, q, sf_n).?;
    const four: CombRun = .{ .r = r, .sc = csc, .pick = pick.ptr, .y = rig.y_new, .mode = 1, .first = .{ q, 3 * q }, .last = .{ 0, 2 * q } };
    const cdoor0 = rigd.arm();
    const cd = struct {
        fn make(rr: *Run, f: cuda.Function, sc_: mx.ExpertScratch, pk_: u64, y: u64, qo: mx.QuarterOrder, door: mx.Door, ts: u64) CombDoor {
            return .{ .r = rr, .f = f, .sc = sc_, .pick = pk_, .y = y, .qo = qo, .door = door, .ts = ts };
        }
    }.make;
    const cms_1 = try kernelBest(r, CombRun, one, CombRun.go);
    const cms_n = try kernelBest(r, CombRun, one_n, CombRun.go);
    const cms_4 = try kernelBest(r, CombRun, four, CombRun.go);
    const cms_d = try kernelBest(r, CombDoor, cd(r, cf, csc, pick.ptr, rig.y_new, qos[0], cdoor0, 0), CombDoor.go);
    const cms_pt = try kernelBest(r, CombDoor, cd(r, cf_pt, csc, pick.ptr, rig.y_new, qos[0], cdoor0, 0), CombDoor.go);
    const cms_g = try kernelBest(r, CombDoor, cd(r, cf_g, csc, pick.ptr, rig.y_new, qos[0], cdoor0, 0), CombDoor.go);
    const cms_t = try kernelBest(r, CombDoor, cd(r, cf_time, csc, pick.ptr, rig.y_new, qos[0], cdoor0, cstamps.ptr), CombDoor.go);
    const ncta = sf_m * sf_n / 256;
    var cring: [5]f64 = undefined;
    var csane = true;
    for (0..5) |rep| {
        try d.check(d.api.cuMemsetD8Async(cstamps.ptr, 0, (ncta * 2 + 1) * 8, r.s.handle), "cuMemsetD8Async");
        const dr = rigd.arm();
        try cd(r, cf_time, csc, pick.ptr, rig.y_new, qos[0], dr, cstamps.ptr).go();
        try r.s.synchronize();
        const raw = try get(cstamps.ptr, (ncta * 2 + 1) * 8, d);
        defer free(raw);
        const ts = u64s(raw);
        const v = stampSpan(ts, ncta, ts[ncta * 2]);
        csane = csane and v != null and try rigd.rang();
        cring[rep] = v orelse 0;
    }
    std.mem.sort(f64, &cring, {}, std.sort.asc(f64));
    const cnull = pct(cms_n, cms_1);
    var t3: WcText = .{};
    try t3.add("roci_door combine (7 slots, N {d}, 4,096 rows, y in the arena), kernel only, best of three, ms: one launch (the baseline's) {d:.3} (null control {d:.1} %: {s}); the four launches of the quarter order (the sent quarters first) {d:.3} ({d:.1} % over one); ", .{ sf_n, cms_1, cnull, resolved(cnull), cms_4, pct(cms_4, cms_1) });
    try t3.add("the door kernel (one launch, sent quarters first, the ring from inside, the block-level release) {d:.3} ({d:.1} % over one launch, {d:.1} % over the four); with every thread fencing its own stores {d:.3} ({d:.1} % over one launch); the GPU-scope release with the last arrival's system fence {d:.3} ({d:.1} % over one launch); the block-level one's timed twin {d:.3}; ", .{ cms_d, pct(cms_d, cms_1), pct(cms_d, cms_4), cms_pt, pct(cms_pt, cms_1), cms_g, pct(cms_g, cms_1), cms_t });
    try t3.add("the ring falls at {d:.1} % of the launch (median of 5, {d:.1} to {d:.1}); the doorbell showed the ticket and the count was zero after each of the 5", .{ cring[2], cring[0], cring[4] });
    r.note(t3.buf[0..t3.n], @abs(cnull) <= null_bar and csane);
    bad = 0;
    wrong = 0;
    unseen = 0;
    unclean = 0;
    for (0..4) |g| {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const rows: [2]usize = .{ (g ^ 1) * q, (g ^ 3) * q };
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + rows[0] * sf_n * 4, ref_dev.ptr + rows[1] * sf_n * 4 }, .{ rig.y_new + rows[0] * sf_n * 4, rig.y_new + rows[1] * sf_n * 4 }, qwords, CombDoor, cd(r, cf, csc, pick.ptr, rig.y_new, qos[g], dr, 0));
        wrong += o.wrong;
        unseen += @intFromBool(!o.seen);
        unclean += @intFromBool(!(try rigd.rang()));
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(cref, got);
    }
    r.report(try std.fmt.bufPrint(&line, "roci_door combine: the door kernel of each of the four nodes against the one launch (the doorbell showed {d} of 4 times, the count clean {d} of 4)", .{ 4 - unseen, 4 - unclean }), bad + unseen + unclean, true);
    r.report("roci_door combine: words of the two sent quarters that differed from the reference when the observer saw the doorbell, four nodes", wrong, true);
    early_wrong = 0;
    {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + q * sf_n * 4, ref_dev.ptr + 3 * q * sf_n * 4 }, .{ rig.y_new + q * sf_n * 4, rig.y_new + 3 * q * sf_n * 4 }, qwords, CombDoor, cd(r, cf_early, csc, pick.ptr, rig.y_new, qos[0], dr, 0));
        early_wrong = if (o.seen) o.wrong else 0;
    }
    r.report("roci_door combine control: the early-ring kernel (the arrival counted before the CTA's stores), words of the sent quarters that differed when the observer saw the doorbell", early_wrong, false);
    // the engine's kernel: the count released at GPU scope, the system fence by the last arrival alone
    bad = 0;
    wrong = 0;
    unseen = 0;
    unclean = 0;
    for (0..4) |g| {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const rows: [2]usize = .{ (g ^ 1) * q, (g ^ 3) * q };
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + rows[0] * sf_n * 4, ref_dev.ptr + rows[1] * sf_n * 4 }, .{ rig.y_new + rows[0] * sf_n * 4, rig.y_new + rows[1] * sf_n * 4 }, qwords, CombDoor, cd(r, cf_g, csc, pick.ptr, rig.y_new, qos[g], dr, 0));
        wrong += o.wrong;
        unseen += @intFromBool(!o.seen);
        unclean += @intFromBool(!(try rigd.rang()));
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(cref, got);
    }
    r.report("roci_door combine with the count released at GPU scope and the system fence by the last arrival alone (the engine's kernel): output bytes against the one launch, words wrong at the ring, and doorbells not seen or counts not clean, summed over the four nodes", bad + wrong + unseen + unclean, true);
    {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + q * sf_n * 4, ref_dev.ptr + 3 * q * sf_n * 4 }, .{ rig.y_new + q * sf_n * 4, rig.y_new + 3 * q * sf_n * 4 }, qwords, CombDoor, cd(r, cf_g_early, csc, pick.ptr, rig.y_new, qos[0], dr, 0));
        early_wrong = if (o.seen) o.wrong else 0;
    }
    r.report("roci_door combine control for the GPU-scope release: its early-ring kernel (the arrival counted before the CTA's stores, delayed stores), words of the sent quarters that differed when the observer saw the doorbell", early_wrong, false);
}

// -- TF_DS_ROCI_GATHERS' boundary staging -----------------------------------------------------------------------------------------

/// TF_DS_ROCI_GATHERS' gather of the boundary layer's cache rows on four nodes in one process: each node's rows of the four cache
/// arrays (device memory) are staged by prompt.boundaryOut into its slab of the arena's boundary region (cudaMalloc, the write-combined
/// host memory of the arena in the baseline, or the locked anonymous pages registered with the driver: TF_DS_ROCI_PAGES=mmap), each slab is
/// written to the same place in the other three regions (the NIC's WRITEs, as device-to-device copies), and prompt.boundaryIn copies the
/// three that landed out to their rows: every array of every node must hold the bytes the in-place gather leaves, the groups around
/// the chunk's untouched. The ms are the 16 copies of one gather (4 out and 12 in) at the real sizes (1,024 rows an owner). A node that
/// stages nothing must leave its rows missing.
fn rociBoundary(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xb0da);
    const rng = prng.random();
    const d = r.d;
    var line: [400]u8 = undefined;
    const world = 4;
    const q: usize = 1024;
    const start: usize = 8192; // the chunk's first group
    const hd: usize = 512;
    const id: usize = 128;
    const rows = [4]usize{ hd / 2, hd / 16, id / 2, id / 32 };
    const total_rows = start + world * q + 16;
    const slab = q * dsv41.rocinante.gb_row;
    var pool: WcPool = .{ .d = d };
    defer pool.deinit();
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    // the whole update's bytes of each array: the rows of the chunk
    var whole: [4][]u8 = undefined;
    for (&whole, rows) |*w, row| {
        w.* = try ha.alloc(u8, total_rows * row);
        rng.bytes(w.*);
    }
    const kinds = [_]Where{ .device, .arena, .mmapreg };
    const names = [_][]const u8{ "cudaMalloc", "write-combined host memory", "locked anonymous pages (cuMemHostRegister)" };
    var ms_out: [3]f64 = .{ 0, 0, 0 };
    for (kinds, names, 0..) |kind, nm, ki| {
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        var arrays: [world][4]u64 = undefined;
        var parts: [world][4]dsv41.comm.InPlace = undefined;
        var regions: [world]u64 = undefined;
        var ok = true;
        for (0..world) |k| {
            for (0..4) |a| {
                const b = try cuda.DeviceBuffer.alloc(d, total_rows * rows[a]);
                try keep.append(ha, b);
                try d.check(d.api.cuMemsetD8Async(b.ptr, 0xee, b.len, r.s.handle), "cuMemsetD8Async");
                // this node's own groups hold the update's bytes
                const lo = (start + k * q) * rows[a];
                arrays[k][a] = b.ptr;
            }
            var cs: dsv41.prompt.Caches = undefined;
            cs.comp_codes[20] = arrays[k][0];
            cs.comp_scales[20] = arrays[k][1];
            cs.idx_codes[20] = arrays[k][2];
            cs.idx_scales[20] = arrays[k][3];
            parts[k] = dsv41.prompt.boundaryGathers(&cs, 20, start, .{ .start = k * q, .n = q }, hd, id);
            if (kind == .device) {
                const b = try cuda.DeviceBuffer.alloc(d, world * slab);
                try keep.append(ha, b);
                regions[k] = b.ptr;
            } else regions[k] = (pool.alloc(kind, world * slab) catch |err| {
                r.note(try std.fmt.bufPrint(&line, "roci_boundary {s}: could not allocate the region ({s})", .{ nm, @errorName(err) }), false);
                ok = false;
                break;
            }).dev;
        }
        if (!ok) continue;
        try r.s.synchronize();
        // (the staging out, the wire, the staging in) as many times as the timing wants; the first run is the checked one
        var t_total: u64 = 0;
        for (0..11) |rep| {
            const t0 = nowNs();
            for (0..world) |k| for (dsv41.prompt.boundaryOut(&parts[k], regions[k], slab, k)) |cp| try d.check(d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, r.s.handle), "cuMemcpyDtoDAsync");
            for (0..world) |k| for (0..world) |dd| {
                if (dd == k) continue;
                const at = dsv41.prompt.boundarySlab(0, slab, k);
                try d.check(d.api.cuMemcpyDtoDAsync_v2(regions[dd] + at, regions[k] + at, slab, r.s.handle), "cuMemcpyDtoDAsync");
            };
            for (0..world) |dd| for (0..world) |k| {
                if (dd == k) continue;
                for (dsv41.prompt.boundaryIn(&parts[dd], regions[dd], slab, k)) |cp| try d.check(d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, r.s.handle), "cuMemcpyDtoDAsync");
            };
            try r.s.synchronize();
            if (rep > 0) t_total += nowNs() - t0;
            if (rep == 0) {
                var bad: usize = 0;
                for (0..world) |k| for (0..4) |a| {
                    const got = try get(arrays[k][a], total_rows * rows[a], d);
                    defer free(got);
                    const lo = start * rows[a];
                    const hi = (start + world * q) * rows[a];
                    bad += differs(got[lo..hi], whole[a][lo..hi]);
                    for (got[0..lo]) |b| bad += @intFromBool(b != 0xee);
                    for (got[hi..]) |b| bad += @intFromBool(b != 0xee);
                };
                r.report(try std.fmt.bufPrint(&line, "roci_boundary {s}: four nodes' cache arrays after the staged gather against the in-place gather's layout (rows around the chunk untouched)", .{nm}), bad, true);
            }
        }
        ms_out[ki] = @as(f64, @floatFromInt(t_total)) / 10.0 / 1e6;
        if (kind == .mmapreg) {
            // the control: node 2 stages nothing, so its rows never reach the other three (the regions hold the checked runs' bytes: poisoned first)
            for (regions) |rg| try poison(rg, world * slab, d, r.s);
            for (0..world) |k| for (0..4) |a| {
                const lo = start * rows[a];
                try d.check(d.api.cuMemsetD8Async(arrays[k][a] + lo, 0xee, world * q * rows[a], r.s.handle), "cuMemsetD8Async");
                const own = (start + k * q) * rows[a];
            };
            try r.s.synchronize();
            for (0..world) |k| {
                if (k == 2) continue;
                for (dsv41.prompt.boundaryOut(&parts[k], regions[k], slab, k)) |cp| try d.check(d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, r.s.handle), "cuMemcpyDtoDAsync");
            }
            for (0..world) |k| for (0..world) |dd| {
                if (dd == k) continue;
                const at = dsv41.prompt.boundarySlab(0, slab, k);
                try d.check(d.api.cuMemcpyDtoDAsync_v2(regions[dd] + at, regions[k] + at, slab, r.s.handle), "cuMemcpyDtoDAsync");
            };
            for (0..world) |dd| for (0..world) |k| {
                if (dd == k) continue;
                for (dsv41.prompt.boundaryIn(&parts[dd], regions[dd], slab, k)) |cp| try d.check(d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, r.s.handle), "cuMemcpyDtoDAsync");
            };
            try r.s.synchronize();
            var bad: usize = 0;
            for (0..world) |k| for (0..4) |a| {
                const got = try get(arrays[k][a], total_rows * rows[a], d);
                defer free(got);
                const lo = start * rows[a];
                const hi = (start + world * q) * rows[a];
                bad += differs(got[lo..hi], whole[a][lo..hi]);
            };
            r.report("roci_boundary control: node 2 stages nothing, the arrays of the four nodes", bad, false);
        }
    }
    r.note(try std.fmt.bufPrint(&line, "roci_boundary: ms for one gather's copies (4 out, 12 in) and its 12 slab writes at 1,024 rows an owner: cudaMalloc {d:.3}, write-combined {d:.3}, locked pages {d:.3}; a pass makes one", .{ ms_out[0], ms_out[1], ms_out[2] }), true);
}

// -- TF_DS_ROCI_CODES -------------------------------------------------------------------------------------------------------------

/// The inputs of TF_DS_ROCI_CODES' two exchanges for two column partners (nodes 1 and 3: pairs 0 and 1): wo_b's part of each (u, bf16
/// [4,096, 2,048]), the experts' pack of each (xd, bf16 [28,672, 640]), the picks and the owner of each expert's rest.
const CodesIn = struct { u: [2]u64, xd: [2]u64, pick: u64, owner: u64 };

const codes_n: usize = 4096;
const codes_uw: usize = 2048;
const codes_slots: usize = 7;
const codes_experts: usize = 384;

/// What the two nodes' consumers hold after the exchange: wo_b's input (codes [4,096, 4,096], scales [4,096, 128] fp32) and down's
/// (hq [28,672, 1,152], hs [28,672, 36] fp32), each node's, read back.
const CodesOut = struct { wq: [2][]u8, ws: [2][]u8, hq: [2][]u8, hs: [2][]u8 };

/// Both exchanges as the engine runs them with TF_DS_ROCI_CODES, the wire (the NIC's WRITE of a send region into the partner's land
/// region) a device-to-device copy, the send and land regions in `kind` memory (device memory: the reference). `skip_pack_wire`: the
/// pack's land regions never written (the control). ms: the producer's launches, the wo_b copies and the assembly, each alone.
fn codesRun(r: *Run, pool: *WcPool, kind: Where, in: CodesIn, skip_pack_wire: bool, ms: ?*[3]f32) !CodesOut {
    const d = r.d;
    const k = r.k;
    const ops = r.ops.?;
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (keep.items) |*b| b.free();
        keep.deinit(std.heap.page_allocator);
    }
    const wob_bytes = dsv41.prompt2d.catCodesBytes(codes_n, codes_uw);
    const pc = try dsv41.prompt2d.packCodesPlan(codes_n * codes_slots, 512, 128);
    var out: CodesOut = undefined;
    var wob_send: [2]u64 = undefined;
    var wob_land: [2]u64 = undefined;
    var pk_send: [2]u64 = undefined;
    var pk_land: [2]u64 = undefined;
    var wq: [2]u64 = undefined;
    var ws: [2]u64 = undefined;
    var hq: [2]u64 = undefined;
    var hs: [2]u64 = undefined;
    const region = struct {
        fn at(rr: *Run, pl: *WcPool, kp: *std.ArrayList(cuda.DeviceBuffer), kd: Where, bytes: usize) !u64 {
            if (kd == .device) {
                const b = try cuda.DeviceBuffer.alloc(rr.d, bytes);
                try kp.append(std.heap.page_allocator, b);
                return b.ptr;
            }
            return (try pl.alloc(kd, bytes)).dev;
        }
        fn dev(rr: *Run, kp: *std.ArrayList(cuda.DeviceBuffer), bytes: usize) !u64 {
            const b = try cuda.DeviceBuffer.alloc(rr.d, bytes);
            try kp.append(std.heap.page_allocator, b);
            return b.ptr;
        }
    };
    for (0..2) |i| {
        wob_send[i] = try region.at(r, pool, &keep, kind, wob_bytes);
        wob_land[i] = try region.at(r, pool, &keep, kind, wob_bytes);
        pk_send[i] = try region.at(r, pool, &keep, kind, pc.bytes);
        pk_land[i] = try region.at(r, pool, &keep, kind, pc.bytes);
        wq[i] = try region.dev(r, &keep, codes_n * 2 * codes_uw);
        ws[i] = try region.dev(r, &keep, codes_n * (2 * codes_uw / 32) * 4);
        hq[i] = try region.dev(r, &keep, codes_n * codes_slots * (2 * 512 + 128));
        hs[i] = try region.dev(r, &keep, codes_n * codes_slots * ((2 * 512 + 128) / 32) * 4);
        try poison(wob_land[i], wob_bytes, d, r.s);
        try poison(pk_land[i], pc.bytes, d, r.s);
        try poison(wq[i], codes_n * 2 * codes_uw, d, r.s);
        try poison(hq[i], codes_n * codes_slots * (2 * 512 + 128), d, r.s);
    }
    const plans = [2]dsv41.prompt2d.CatCodes{
        dsv41.prompt2d.catCodesPlan(1, .{ 0, 1, 2, 3 }, codes_n, .{ codes_uw, codes_uw }, in.u[0], wob_send[0], wob_land[0], wq[0], ws[0]),
        dsv41.prompt2d.catCodesPlan(3, .{ 0, 1, 2, 3 }, codes_n, .{ codes_uw, codes_uw }, in.u[1], wob_send[1], wob_land[1], wq[1], ws[1]),
    };
    for (plans) |pl| try dsv41.prompt2d.catCodesQuant(k, r.s, pl);
    for (0..2) |i| try dsv41.prompt2d.packQuant(k, r.s, in.xd[i], pk_send[i], pc);
    // the wire
    for (0..2) |i| {
        try d.check(d.api.cuMemcpyDtoDAsync_v2(wob_land[1 - i], wob_send[i], wob_bytes, r.s.handle), "cuMemcpyDtoDAsync");
        if (!skip_pack_wire) try d.check(d.api.cuMemcpyDtoDAsync_v2(pk_land[1 - i], pk_send[i], pc.bytes, r.s.handle), "cuMemcpyDtoDAsync");
    }
    // the consumers
    for (plans) |pl| for (pl.copies) |cp| try ops.copyRows(r.s, cp.src, cp.src_stride, cp.dst, cp.dst_stride, cp.width, cp.rows);
    for (0..2) |i| {
        var sc: mx.ExpertScratch = undefined;
        sc.hq = hq[i];
        sc.hs = hs[i];
        const part0 = if (i == 0) pk_send[0] else pk_land[1];
        const part1 = if (i == 1) pk_send[1] else pk_land[0];
        try dsv41.prompt2d.packAssemble(ops, r.s, part0, part1, in.pick, in.owner, sc, pc);
    }
    try r.s.synchronize();
    if (ms) |m| {
        // each stage alone, kernel only (the data stays as the checked run left it)
        const Q = struct {
            r: *Run,
            pl: dsv41.prompt2d.CatCodes,
            pc: dsv41.prompt2d.PackCodes,
            xd: u64,
            send: u64,
            fn quant(c: @This()) anyerror!void {
                try dsv41.prompt2d.catCodesQuant(c.r.k, c.r.s, c.pl);
                try dsv41.prompt2d.packQuant(c.r.k, c.r.s, c.xd, c.send, c.pc);
            }
            fn copies(c: @This()) anyerror!void {
                for (c.pl.copies) |cp| try c.r.ops.?.copyRows(c.r.s, cp.src, cp.src_stride, cp.dst, cp.dst_stride, cp.width, cp.rows);
            }
        };
        const q: Q = .{ .r = r, .pl = plans[0], .pc = pc, .xd = in.xd[0], .send = pk_send[0] };
        m[0] = try wcTime(r, Q, q, Q.quant);
        m[1] = try wcTime(r, Q, q, Q.copies);
        const A = struct {
            r: *Run,
            p0: u64,
            p1: u64,
            pick: u64,
            owner: u64,
            sc: mx.ExpertScratch,
            pc: dsv41.prompt2d.PackCodes,
            fn go(c: @This()) anyerror!void {
                try dsv41.prompt2d.packAssemble(c.r.ops.?, c.r.s, c.p0, c.p1, c.pick, c.owner, c.sc, c.pc);
            }
        };
        var sc: mx.ExpertScratch = undefined;
        sc.hq = hq[0];
        sc.hs = hs[0];
        m[2] = try wcTime(r, A, .{ .r = r, .p0 = pk_send[0], .p1 = pk_land[0], .pick = in.pick, .owner = in.owner, .sc = sc, .pc = pc }, A.go);
    }
    for (0..2) |i| {
        out.wq[i] = try get(wq[i], codes_n * 2 * codes_uw, d);
        out.ws[i] = try get(ws[i], codes_n * (2 * codes_uw / 32) * 4, d);
        out.hq[i] = try get(hq[i], codes_n * codes_slots * (2 * 512 + 128), d);
        out.hs[i] = try get(hs[i], codes_n * codes_slots * ((2 * 512 + 128) / 32) * 4, d);
    }
    return out;
}

fn codesFree(o: CodesOut) void {
    for (0..2) |i| {
        free(o.wq[i]);
        free(o.ws[i]);
        free(o.hq[i]);
        free(o.hs[i]);
    }
}

fn codesDiffer(a: CodesOut, b: CodesOut) [2]usize {
    var wob: usize = 0;
    var pack: usize = 0;
    for (0..2) |i| {
        wob += differs(a.wq[i], b.wq[i]) + differs(a.ws[i], b.ws[i]);
        pack += differs(a.hq[i], b.hq[i]) + differs(a.hs[i], b.hs[i]);
    }
    return .{ wob, pack };
}

/// TF_DS_ROCI_CODES' two exchanges on two column partners in one process: wo_b's part quantized into the send region of each page type
/// (device memory, write-combined host memory, locked registered pages), written to the partner's land region, read by the four copies
/// that assemble wo_b's input; the experts' pack quantized into its send region, written to the land region, read by the assembly into
/// down's input. Every consumer byte must equal the run in device memory throughout. A pack that never reaches the land region must
/// leave the assembly's bytes different. ms a launch of the quantizers, the four copies and the assembly in each page type.
fn rociCodes(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xc0de5);
    const rng = prng.random();
    const d = r.d;
    var line: [512]u8 = undefined;
    if (r.ops == null) {
        r.note("roci_codes: the ops are available", false);
        return;
    }
    var pool: WcPool = .{ .d = d };
    defer pool.deinit();
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var in: CodesIn = undefined;
    for (0..2) |i| {
        const ub = try upload(d, try bf16s(codes_n * codes_uw, rng, 1.0, ha));
        try keep.append(ha, ub);
        in.u[i] = ub.ptr;
        const xb = try upload(d, try bf16s(codes_n * codes_slots * 640, rng, 1.0, ha));
        try keep.append(ha, xb);
        in.xd[i] = xb.ptr;
    }
    const picks = try ha.alloc(i32, codes_n * codes_slots);
    for (picks) |*p| p.* = @intCast(rng.uintLessThan(usize, codes_experts));
    const owner = try ha.alloc(i32, codes_experts + 1);
    for (owner) |*o| o.* = @intFromBool(rng.boolean());
    owner[codes_experts] = 0;
    const pb = try upload(d, std.mem.sliceAsBytes(picks));
    try keep.append(ha, pb);
    const ob = try upload(d, std.mem.sliceAsBytes(owner));
    try keep.append(ha, ob);
    in.pick = pb.ptr;
    in.owner = ob.ptr;
    var ms: [3][3]f32 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    const ref = try codesRun(r, &pool, .device, in, false, &ms[0]);
    defer codesFree(ref);
    pool.freeAll();
    const kinds = [_]Where{ .arena, .mmapreg };
    const names = [_][]const u8{ "write-combined host memory", "locked anonymous pages (cuMemHostRegister)" };
    for (kinds, names, 1..) |kind, nm, ki| {
        const got = codesRun(r, &pool, kind, in, false, &ms[ki]) catch |err| {
            r.note(try std.fmt.bufPrint(&line, "roci_codes {s}: could not run ({s})", .{ nm, @errorName(err) }), false);
            r.note(try std.fmt.bufPrint(&line, "roci_codes {s}: could not run ({s})", .{ nm, @errorName(err) }), false);
            pool.freeAll();
            continue;
        };
        defer codesFree(got);
        const df = codesDiffer(ref, got);
        r.report(try std.fmt.bufPrint(&line, "roci_codes {s}: wo_b's input (codes and scales) on both partners, the send and land regions in this memory, against device memory", .{nm}), df[0], true);
        r.report(try std.fmt.bufPrint(&line, "roci_codes {s}: down's input (hq and hs) on both partners, the pack's regions in this memory, against device memory", .{nm}), df[1], true);
        pool.freeAll();
    }
    // the control: the pack never written into the land regions
    const ctl = try codesRun(r, &pool, .mmapreg, in, true, null);
    defer codesFree(ctl);
    pool.freeAll();
    r.report("roci_codes control: the pack's land regions never written, down's input on both partners", codesDiffer(ref, ctl)[1], false);
    r.note(try std.fmt.bufPrint(&line, "roci_codes: ms a pass-launch (one of the 20 layers' each) cudaMalloc / write-combined / locked pages: the quantizers of wo_b's part and of the pack {d:.3} / {d:.3} / {d:.3}, wo_b's four copies from the land region {d:.3} / {d:.3} / {d:.3}, the pack's assembly {d:.3} / {d:.3} / {d:.3}", .{ ms[0][0], ms[1][0], ms[2][0], ms[0][1], ms[1][1], ms[2][1], ms[0][2], ms[1][2], ms[2][2] }), true);
}

// -- TF_DS_ROCI_STAGGER ----------------------------------------------------------------------------------------------

const PairNode = struct {
    arena: []align(64) u8,
    arena_dev: u64,
    ctl: *dsv41.rocinante.Control,
    ctl_dev: u64,
    t: *dsv41.rocinante.Transport = undefined,
    peer: *PairNode = undefined,
    me: u8 = 0,
    hold: std.atomic.Value(bool) = .init(false),
};

fn pairHostAlloc(d: *const cuda.Driver, len: usize, flags: c_uint) !struct { p: [*]align(64) u8, dev: u64 } {
    var p: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&p, len, flags), "cuMemHostAlloc");
    var dev: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, p, 0), "cuMemHostGetDevicePointer");
    return .{ .p = @ptrCast(@alignCast(p.?)), .dev = dev };
}

fn pairNap(ns: u64) void {
    const ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}

fn pairPost(ctx: *anyopaque, rail: u8, peer: u8, class: dsv41.rocinante.Class, chain: *const dsv41.rocinante.Chain, word: u64) dsv41.rocinante.PostError!void {
    const me: *PairNode = @ptrCast(@alignCast(ctx));
    _ = peer;
    while (me.hold.load(.acquire)) pairNap(100_000);
    for (chain.slice()) |p| @memcpy(me.peer.arena[p.dst..][0..p.len], me.arena[p.src..][0..p.len]);
    @atomicStore(u64, &me.peer.ctl.recv[@backingInt(class)][me.me][rail].v, word, .release);
}
fn pairFail(_: *anyopaque) void {}
fn pairReap(_: *anyopaque, _: u8) dsv41.rocinante.PostError!void {}

const PairLane = struct {
    nodes: [2]PairNode = undefined,

    fn open(l: *PairLane, d: *const cuda.Driver) !void {
        for (&l.nodes, 0..) |*n, i| {
            const a = try pairHostAlloc(d, 64 << 20, 0x01 | 0x02 | 0x04);
            const c = try pairHostAlloc(d, @sizeOf(dsv41.rocinante.Control), 0x01 | 0x02);
            n.* = .{ .arena = a.p[0 .. 64 << 20], .arena_dev = a.dev, .ctl = @ptrCast(c.p), .ctl_dev = c.dev, .me = @intCast(i) };
            n.ctl.* = .{};
        }
        for (&l.nodes, 0..) |*n, i| {
            n.peer = &l.nodes[1 - i];
            n.t = try dsv41.rocinante.Transport.init(std.heap.page_allocator, @intCast(i), dsv41.rocinante.identity, n.ctl, n.ctl_dev, .{ .ctx = n, .post_fn = pairPost, .fail_fn = pairFail, .reap_fn = pairReap }, .{ .nap = false });
            try n.t.start(null);
        }
    }

    fn close(l: *PairLane, d: *const cuda.Driver) void {
        for (&l.nodes) |*n| n.t.deinit();
        for (&l.nodes) |*n| {
            _ = d.api.cuMemFreeHost(n.arena.ptr);
            _ = d.api.cuMemFreeHost(@ptrCast(n.ctl));
        }
    }
};

fn pairWord(lane: u32, node: u32, block: u32, gen: u32) u32 {
    return (lane << 30) | (node << 28) | (block << 20) | gen;
}

fn pairWrite(d: *const cuda.Driver, nd: *PairNode, s: cuda.Stream, lane: u32, gen: u32) !void {
    const l = dsv41.rocinante.layout();
    const blk: usize = 1024 * 2560 * 4;
    for (0..4) |b| try d.check(d.api.cuMemsetD32Async(nd.arena_dev + l.at(.psum) + b * blk, pairWord(lane, nd.me, @intCast(b), gen), blk / 4, s.handle), "memset");
}

/// Words of node `nd`'s stage regions that differ from the peer's lane `from` words.
fn pairStageBad(nd: *PairNode, from: u32, gen: u32) usize {
    const l = dsv41.rocinante.layout();
    const blk: usize = 1024 * 2560 * 4;
    const peer: u32 = 1 - nd.me;
    const w = [2]u32{ pairWord(from, peer, nd.me, gen), pairWord(from, peer, nd.me ^ 2, gen) };
    var bad: usize = 0;
    for (0..2) |k| {
        const got = std.mem.bytesAsSlice(u32, nd.arena[l.at(.stage) + k * blk ..][0..blk]);
        for (got) |x| bad += @intFromBool(x != w[k]);
    }
    return bad;
}

fn pairStreamDone(d: *const cuda.Driver, s: cuda.Stream, ms: u64) bool {
    var waited: u64 = 0;
    while (waited < ms) : (waited += 1) {
        if (d.api.cuStreamQuery(s.handle) == 0) return true;
        pairNap(1_000_000);
    }
    return false;
}

/// Two passes of a pair on one rank's two lanes (each lane its own arena, control block, transport; two nodes a lane, one stream a node): both
/// lanes' exchanges in flight at once land every byte in their own lane; a lane-1 peer that is late holds lane 1's wait while lane 0's
/// streams finish; lane 1 miswired into lane 0's arena must differ; pass B's cache read behind the event sees A's write and, without the event,
/// does not.
fn rociPair(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const d = r.d;
    var lanes: [2]PairLane = undefined;
    try lanes[0].open(d);
    try lanes[1].open(d);
    defer for (&lanes) |*l| l.close(d);
    var st: [2][2]cuda.Stream = undefined;
    for (&st) |*row| {
        for (row) |*x| x.* = try cuda.Stream.init(d, true);
    }
    defer for (&st) |*row| {
        for (row) |*x| x.deinit();
    };
    var bad: usize = 0;
    for (1..5) |gen| {
        for (0..2) |lane| {
            for (0..2) |n| {
                try pairWrite(d, &lanes[lane].nodes[n], st[lane][n], @intCast(lane), @intCast(gen));
                try lanes[lane].nodes[n].t.send(st[lane][n], .own1, 0);
            }
        }
        for (0..2) |lane| {
            for (0..2) |n| try lanes[lane].nodes[n].t.wait(st[lane][n], .own1);
        }
        for (0..2) |lane| {
            for (0..2) |n| try st[lane][n].synchronize();
        }
        for (0..2) |lane| {
            for (0..2) |n| bad += pairStageBad(&lanes[lane].nodes[n], @intCast(lane), @intCast(gen));
        }
    }
    r.report("roci_pair: two lanes in flight at once, four generations: every word of each lane's stage regions is its own lane's", bad, true);
    // lane 1's node 1 is late; lane 0's streams finish meanwhile and lane 1's node 0 still waits
    lanes[1].nodes[1].hold.store(true, .release);
    for (0..2) |n| {
        try pairWrite(d, &lanes[1].nodes[n], st[1][n], 1, 9);
        try lanes[1].nodes[n].t.send(st[1][n], .own1, 0);
    }
    for (0..2) |n| {
        try pairWrite(d, &lanes[0].nodes[n], st[0][n], 0, 9);
        try lanes[0].nodes[n].t.send(st[0][n], .own1, 0);
        try lanes[0].nodes[n].t.wait(st[0][n], .own1);
    }
    try lanes[1].nodes[0].t.wait(st[1][0], .own1);
    const live = pairStreamDone(d, st[0][0], 3000) and pairStreamDone(d, st[0][1], 3000);
    const pending = !pairStreamDone(d, st[1][0], 150);
    r.note("roci_pair: lane 0's streams finish while lane 1's wait for its late peer is pending (a pending wait on one stream holds the other)", live and pending);
    r.report("roci_pair control: the wait for lane 1's words not yet released, lane 1's stage holds lane 1's bytes", pairStageBad(&lanes[1].nodes[0], 1, 9), false);
    lanes[1].nodes[1].hold.store(false, .release);
    try lanes[1].nodes[1].t.wait(st[1][1], .own1);
    try st[1][0].synchronize();
    try st[1][1].synchronize();
    r.report("roci_pair: after the late peer posts, lane 1's stage holds its bytes", pairStageBad(&lanes[1].nodes[0], 1, 9) + pairStageBad(&lanes[0].nodes[0], 0, 9), true);
    // the cache gate
    const flag = try pairHostAlloc(d, 4096, 0x01 | 0x02);
    defer _ = d.api.cuMemFreeHost(flag.p);
    const flag_p: *u64 = @ptrCast(@alignCast(flag.p));
    var cache = try cuda.DeviceBuffer.alloc(d, 4096);
    defer cache.free();
    var out = try cuda.DeviceBuffer.alloc(d, 8192);
    defer out.free();
    var ev = try cuda.Event.init(d, false);
    defer ev.deinit();
    var diff: [2]usize = .{ 0, 0 };
    for ([_]bool{ true, false }, 0..) |gated, gi| {
        flag_p.* = 0;
        const base: u64 = if (gated) 0 else 4096;
        const val: u32 = if (gated) 0x22222222 else 0x33333333;
        try st[0][0].waitValues64(&.{flag.dev}, &.{1});
        try d.check(d.api.cuMemsetD32Async(cache.ptr, val, 1024, st[0][0].handle), "memset");
        try ev.record(st[0][0]);
        if (gated) try st[1][0].wait(ev);
        try d.check(d.api.cuMemcpyDtoDAsync_v2(out.ptr + base, cache.ptr, 4096, st[1][0].handle), "copy");
        pairNap(100_000_000);
        @atomicStore(u64, flag_p, 1, .release);
        try st[0][0].synchronize();
        try st[1][0].synchronize();
        var got: [1024]u32 = undefined;
        for (got) |x| diff[gi] += @intFromBool(x != val);
    }
    r.report("roci_pair: pass B's read of a layer's cache behind pass A's event is A's write", diff[0], true);
    r.report("roci_pair control: the same read without the event", diff[1], false);
}

/// TF_DS_ROCI_STAGGER_LANES=3: three passes on one rank's three lanes (each its own arena, control block, transport; two nodes a lane, one stream a
/// node): all three lanes' exchanges in flight at once land every byte in their own lane; and the cache gate chains, pass C's read of a layer's
/// cache behind B's event behind A's event is A's write, and with B's wait left out is not.
fn rociTriple(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const d = r.d;
    var lanes: [3]PairLane = undefined;
    for (&lanes) |*l| try l.open(d);
    defer for (&lanes) |*l| l.close(d);
    var st: [3][2]cuda.Stream = undefined;
    for (&st) |*row| {
        for (row) |*x| x.* = try cuda.Stream.init(d, true);
    }
    defer for (&st) |*row| {
        for (row) |*x| x.deinit();
    };
    var bad: usize = 0;
    for (1..5) |gen| {
        for (0..3) |lane| {
            for (0..2) |n| {
                try pairWrite(d, &lanes[lane].nodes[n], st[lane][n], @intCast(lane), @intCast(gen));
                try lanes[lane].nodes[n].t.send(st[lane][n], .own1, 0);
            }
        }
        for (0..3) |lane| {
            for (0..2) |n| try lanes[lane].nodes[n].t.wait(st[lane][n], .own1);
        }
        for (0..3) |lane| {
            for (0..2) |n| try st[lane][n].synchronize();
        }
        for (0..3) |lane| {
            for (0..2) |n| bad += pairStageBad(&lanes[lane].nodes[n], @intCast(lane), @intCast(gen));
        }
    }
    r.report("roci_triple: three lanes in flight at once, four generations: every word of each lane's stage regions is its own lane's", bad, true);
    // the cache gate chain A -> B -> C
    const flag = try pairHostAlloc(d, 4096, 0x01 | 0x02);
    defer _ = d.api.cuMemFreeHost(flag.p);
    const flag_p: *u64 = @ptrCast(@alignCast(flag.p));
    var cache = try cuda.DeviceBuffer.alloc(d, 4096);
    defer cache.free();
    var mid = try cuda.DeviceBuffer.alloc(d, 8192);
    defer mid.free();
    var out = try cuda.DeviceBuffer.alloc(d, 8192);
    defer out.free();
    var ev_a = try cuda.Event.init(d, false);
    defer ev_a.deinit();
    var ev_b = try cuda.Event.init(d, false);
    defer ev_b.deinit();
    var diff: [2]usize = .{ 0, 0 };
    for ([_]bool{ true, false }, 0..) |gated, gi| {
        flag_p.* = 0;
        const base: u64 = if (gated) 0 else 4096;
        const val: u32 = if (gated) 0x22222222 else 0x33333333;
        // stale contents everywhere, so a read that runs early sees a value that is not `val`
        try d.check(d.api.cuMemsetD32Async(cache.ptr, 0x11111111, 1024, st[0][0].handle), "memset");
        try d.check(d.api.cuMemsetD32Async(mid.ptr + base, 0x11111111, 1024, st[0][0].handle), "memset");
        try st[0][0].synchronize();
        try st[0][0].waitValues64(&.{flag.dev}, &.{1});
        try d.check(d.api.cuMemsetD32Async(cache.ptr, val, 1024, st[0][0].handle), "memset");
        try ev_a.record(st[0][0]);
        if (gated) try st[1][0].wait(ev_a);
        try d.check(d.api.cuMemcpyDtoDAsync_v2(mid.ptr + base, cache.ptr, 4096, st[1][0].handle), "copy");
        try ev_b.record(st[1][0]);
        try st[2][0].wait(ev_b); // C always waits B: the chain breaks only where B skips A
        try d.check(d.api.cuMemcpyDtoDAsync_v2(out.ptr + base, mid.ptr + base, 4096, st[2][0].handle), "copy");
        pairNap(100_000_000);
        @atomicStore(u64, flag_p, 1, .release);
        try st[0][0].synchronize();
        try st[1][0].synchronize();
        try st[2][0].synchronize();
        var got: [1024]u32 = undefined;
        for (got) |x| diff[gi] += @intFromBool(x != val);
    }
    r.report("roci_triple: pass C's read of a layer's cache behind B's event behind A's event is A's write", diff[0], true);
    r.report("roci_triple control: the same chain with B's wait on A's event left out", diff[1], false);
}

/// TF_DS_PACK_EPILOGUE: gate/up's twin (tf_dsn_sf_gu_exact_q_kernel) against gate/up then packQuant, on random data at the pack's real shape
/// (a main block of X_MAIN columns and a rest block of the remaining ones): every byte of the codes, of the scales and of the bf16 pack equal at
/// n = 64, 500, 1,087 rows. The control leaves the twin out: the poisoned codes must differ. Without the twin in the module it says so, as a
/// failed line (the section comes alive when the kernel is in the build).
fn packEpilogue(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xf5);
    const rng = prng.random();
    const d = r.d;
    var line: [256]u8 = undefined;
    mx.usePackEpilogue(r.k) catch {
        r.note("pack_epilogue: tf_dsn_sf_gu_exact_q_kernel is in this build's module", false);
        r.note("pack_epilogue control: not run without the twin", false);
        return;
    };
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const M = any_max;
    const ex = try xExperts(d, rng, ha, &keep);
    // the twin is the exact block-scaled gate/up's: both launches (gate / up then packQuant, and the twin) run on that arithmetic
    try mx.sfProve(r.k, ex.native.?.s1);
    r.k.sf = .exact;
    r.k.run128 = true;
    defer {
        r.k.sf = .off;
        r.k.run128 = false;
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
    }
    const sc = try xScratch(d, M, true, ha, &keep);
    const hb_bytes = M * X_SL * X_W * 2;
    const hb_ref = try cuda.DeviceBuffer.alloc(d, hb_bytes);
    const hb_got = try cuda.DeviceBuffer.alloc(d, hb_bytes);
    const pc_max = try dsv41.prompt2d.packCodesPlan(M * X_SL, X_MAIN, X_W - X_MAIN);
    const q_ref = try cuda.DeviceBuffer.alloc(d, pc_max.bytes);
    const q_got = try cuda.DeviceBuffer.alloc(d, pc_max.bytes);
    for ([_]cuda.DeviceBuffer{ hb_ref, hb_got, q_ref, q_got }) |b| try keep.append(ha, b);
    // which expert's rest columns this node computes (the node's owner table): the other entries' rest bytes are never written by gate / up
    // and never read by down (it takes a slot's rest block from its owner's part), so what the two launches leave there is not compared
    const owner_raw = try get(ex.native.?.owner, (X_E + 1) * 4, d);
    defer free(owner_raw);
    const owner = std.mem.bytesAsSlice(i32, owner_raw);
    // the differing bytes by region: [codes, scales, bf16] x [main, rest owned, rest not owned]
    var diffs: [3][3]usize = @splat(@splat(0));
    var first: [3][3]?[4]usize = @splat(@splat(null)); // n, entry, byte in the region, (ref << 8 | got)
    var bad: usize = 0;
    var control_differs = true;
    for ([_]usize{ 64, 500, 1087 }) |n| {
        const x = try upload(d, try bf16s(n * X_D, rng, 0.5, ha));
        const pw = try xPicks(n, rng, ha);
        const pb = try upload(d, std.mem.sliceAsBytes(pw.pick));
        const wb = try upload(d, std.mem.sliceAsBytes(pw.wts));
        for ([_]cuda.DeviceBuffer{ x, pb, wb }) |b| try keep.append(ha, b);
        const pc = try dsv41.prompt2d.packCodesPlan(n * X_SL, X_MAIN, X_W - X_MAIN);
        try poison(q_ref.ptr, pc.bytes, d, r.s);
        try mx.gateUp(r.k, r.s, d, ex, sc, x.ptr, X_D, pb.ptr, wb.ptr, n, 7.0, hb_ref.ptr, X_MAIN, false);
        try dsv41.prompt2d.packQuant(r.k, r.s, hb_ref.ptr, q_ref.ptr, pc);
        try poison(q_got.ptr, pc.bytes, d, r.s);
        try poison(hb_got.ptr, hb_bytes, d, r.s);
        try r.s.synchronize();
        const ref_q = try get(q_ref.ptr, pc.bytes, d);
        defer free(ref_q);
        const ref_h = try get(hb_ref.ptr, n * X_SL * X_W * 2, d);
        defer free(ref_h);
        const unwritten = try get(q_got.ptr, pc.bytes, d);
        defer free(unwritten);
        control_differs = control_differs and differs(ref_q, unwritten) > 0;
        try mx.gateUpPack(r.k, r.s, d, ex, sc, x.ptr, X_D, pb.ptr, wb.ptr, n, 7.0, hb_got.ptr, X_MAIN, false, .{ .codes = q_got.ptr, .scales = q_got.ptr + pc.scales, .ig = @intCast(pc.ig), .ir = @intCast(pc.ir), .P = @intCast(pc.p) });
        try r.s.synchronize();
        const got_q = try get(q_got.ptr, pc.bytes, d);
        defer free(got_q);
        const got_h = try get(hb_got.ptr, n * X_SL * X_W * 2, d);
        defer free(got_h);
        const P = n * X_SL;
        for (0..P) |e| {
            const owned = owner[@intCast(pw.pick[e])] != 0;
            // [region][block]: block 0 main, 1 rest of an expert this node owns, 2 rest of another's
            const rb: usize = if (owned) 1 else 2;
            const cm = e * X_MAIN; // main codes, bytes
            const cr = P * X_MAIN + e * (X_W - X_MAIN); // rest codes
            const sm = pc.scales + (e * (X_MAIN / 32)) * 4; // main scales
            const sr = pc.scales + (P * (X_MAIN / 32) + e * ((X_W - X_MAIN) / 32)) * 4; // rest scales
            const hm = e * X_MAIN * 2; // bf16 main
            const hr = (P * X_MAIN + e * (X_W - X_MAIN)) * 2; // bf16 rest
            const spans = [_][4]usize{ .{ 0, 0, cm, X_MAIN }, .{ 0, rb, cr, X_W - X_MAIN }, .{ 1, 0, sm, (X_MAIN / 32) * 4 }, .{ 1, rb, sr, ((X_W - X_MAIN) / 32) * 4 }, .{ 2, 0, hm, X_MAIN * 2 }, .{ 2, rb, hr, (X_W - X_MAIN) * 2 } };
            for (spans, 0..) |sp, si| {
                const a = if (sp[0] == 2) ref_h else ref_q;
                const b = if (sp[0] == 2) got_h else got_q;
                for (sp[2]..sp[2] + sp[3]) |i| if (a[i] != b[i]) {
                    diffs[sp[0]][sp[1]] += 1;
                    if (first[sp[0]][sp[1]] == null) first[sp[0]][sp[1]] = .{ n, e, i - sp[2], @as(usize, a[i]) << 8 | b[i] };
                };
                _ = si;
            }
        }
        // what this comparison covers: every byte of the three regions
        _ = differs(ref_q, got_q) + differs(ref_h, got_h);
    }
    {
        const names = [_][]const u8{ "codes", "scales", "bf16 pack" };
        var t: WcText = .{};
        try t.add("pack_epilogue differing bytes (twin against gate / up then packQuant; the rest of an expert this node does not own is written by neither launch and read by nothing): ", .{});
        for (names, 0..) |nm, ri| {
            try t.add("{s}: main {d}, rest owned {d}, rest not owned {d}; ", .{ nm, diffs[ri][0], diffs[ri][1], diffs[ri][2] });
            for (0..2) |bi| if (first[ri][bi]) |f| try t.add("FIRST {s} {s} at n {d} entry {d} byte {d}: gate/up+packQuant 0x{x:0>2} twin 0x{x:0>2}; ", .{ nm, if (bi == 0) "main" else "rest owned", f[0], f[1], f[2], f[3] >> 8, f[3] & 0xff });
        }
        r.note(t.buf[0..t.n], true);
        for (0..3) |ri| bad += diffs[ri][0] + diffs[ri][1];
    }
    r.report(try std.fmt.bufPrint(&line, "pack_epilogue: the twin's codes, scales and bf16 pack against gate / up then packQuant at n = 64, 500, 1,087 (every byte down or the transport reads)", .{}), bad, true);
    r.report("pack_epilogue control: the pack's codes before the twin ran (poisoned) differ from packQuant's", if (control_differs) 1 else 0, false);
}

// -- TF_DS_ATTN_TWIN (w, z) and TF_DS_ROPE_W -------------------------------------------------------------------------

/// attnU / attnUDecode with the twin TF_DS_ATTN_TWIN names: the same calls, cases, controls and line counts, the twin's
/// bytes against the part's (and so against u's: both equal the part's).
fn withTwin(r: *Run, rng: std.Random, twin: tri_attn.Twin, decode: bool) !void {
    const was = tri_attn.attn_twin;
    tri_attn.attn_twin = twin;
    defer tri_attn.attn_twin = was;
    if (decode) try attnUDecode(r, rng) else try attnU(r, rng);
}
fn attnW(r: *Run, rng: std.Random) !void {
    try withTwin(r, rng, .w, false);
}
fn attnX(r: *Run, rng: std.Random) !void {
    try withTwin(r, rng, .x, false);
}
fn attnZ(r: *Run, rng: std.Random) !void {
    try withTwin(r, rng, .z, false);
}
fn attnWDecode(r: *Run, rng: std.Random) !void {
    try withTwin(r, rng, .w, true);
}
fn attnXDecode(r: *Run, rng: std.Random) !void {
    try withTwin(r, rng, .x, true);
}
fn attnZDecode(r: *Run, rng: std.Random) !void {
    try withTwin(r, rng, .z, true);
}

/// tri_norm.ropeHeads with and without `rope_w` on random bf16 heads with NaN, infinity, minus zero and subnormals planted:
/// _rope_heads against the extra set's twin _rope_heads_w, in place, forward and inverse, at the kit's shapes (H 16 and 32,
/// HD 512 and 128, 4,096 / 17 / 1 rows): every output byte equal; and a control: the rotation changed the data.
fn ropeW(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var line: [200]u8 = undefined;
    if (r.t.extra == null) return r.note("rope_w: the kit's extra set (aot-extra/) loads", false);
    const misses = tri.twin_misses.load(.monotonic);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const RD: usize = 64;
    const maxpos: usize = 8192;
    const cos = try upload(d, try f32s(maxpos * RD / 2, rng, ha));
    const sin = try upload(d, try f32s(maxpos * RD / 2, rng, ha));
    try keep.append(ha, cos);
    try keep.append(ha, sin);
    const Case = struct { h: usize, hd: usize, inv: bool, rows: usize };
    const shapes = [_]Case{ .{ .h = 16, .hd = 512, .inv = false, .rows = 0 }, .{ .h = 16, .hd = 512, .inv = true, .rows = 0 }, .{ .h = 32, .hd = 128, .inv = false, .rows = 0 } };
    const specials = [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x8000, 0x0000, 0x0001, 0x8001, 0x7f7f };
    var changed_all = true;
    for (shapes) |sh| for ([_]usize{ 4096, 17, 1 }) |R| {
        const n = R * sh.h * sh.hd;
        const x0 = try bf16s(n, rng, 3.0, ha);
        const words = std.mem.bytesAsSlice(u16, x0);
        for (0..@max(n / 1000, 8)) |_| words[rng.uintLessThan(usize, n)] = specials[rng.uintLessThan(usize, specials.len)];
        const pos = try ha.alloc(i64, R);
        for (pos) |*q| q.* = @intCast(rng.uintLessThan(usize, maxpos));
        const pb = try upload(d, std.mem.sliceAsBytes(pos));
        const xa = try upload(d, x0);
        const xb = try upload(d, x0);
        try keep.append(ha, pb);
        try keep.append(ha, xa);
        try keep.append(ha, xb);
        tri_norm.rope_w = false;
        try tri_norm.ropeHeads(r.t, xa.ptr, cos.ptr, sin.ptr, pb.ptr, RD, sh.inv, R, sh.h, sh.hd);
        tri_norm.rope_w = true;
        defer tri_norm.rope_w = false;
        try tri_norm.ropeHeads(r.t, xb.ptr, cos.ptr, sin.ptr, pb.ptr, RD, sh.inv, R, sh.h, sh.hd);
        try r.s.synchronize();
        const want = try get(xa.ptr, n * 2, d);
        defer free(want);
        const got = try get(xb.ptr, n * 2, d);
        defer free(got);
        if (differs(want, x0) == 0) changed_all = false;
        r.report(try std.fmt.bufPrint(&line, "rope_w: {d} rows, H {d}, HD {d}{s}: the twin's heads against rope_heads'", .{ R, sh.h, sh.hd, if (sh.inv) ", inverse" else "" }), differs(want, got), true);
    };
    r.note("rope_w control: the rotation changed the data in every case", changed_all);
    r.note("rope_w: every call found its twin in the extra set (no call sent to rope_heads)", tri.twin_misses.load(.monotonic) == misses);
}

/// TF_DS_QKV_PLAIN: tri_norm.qKvNormPlain (the extra set's _q_kv_norm without rotations: q's RMSNorm and the window KV's norm / RoPE /
/// quant in one launch) against tri_basic.rmsnorm then tri_norm.kvNormRope, on random bf16 rows with NaN, infinity, minus zero and
/// subnormals planted, D 1280 and DK 512, RD 64, 1 / 3 / 17 / 64 rows, slots mixing ring rows and -1: q's normed rows, the window KV
/// and the whole ring equal byte for byte; a control: one q value changed changes q's rows.
fn qkvPlain(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var line: [200]u8 = undefined;
    if (r.t.extra == null) return r.note("qkv_plain: the kit's extra set (aot-extra/) loads", false);
    const misses = tri.fuse_misses.load(.monotonic);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const D: usize = 1280;
    const DK: usize = 512;
    const RD: usize = 64;
    const maxpos: usize = 8192;
    const ring_rows: usize = 256;
    const specials = [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x8000, 0x0000, 0x0001, 0x8001, 0x7f7f };
    const cos = try upload(d, try f32s(maxpos * RD / 2, rng, ha));
    const sin = try upload(d, try f32s(maxpos * RD / 2, rng, ha));
    const wq = try upload(d, try bf16s(D, rng, 1.5, ha));
    const wk = try upload(d, try bf16s(DK, rng, 1.5, ha));
    try keep.append(ha, cos);
    try keep.append(ha, sin);
    try keep.append(ha, wq);
    try keep.append(ha, wk);
    var changed_all = true;
    for ([_]usize{ 1, 3, 17, 64 }) |R| {
        const qa0 = try bf16s(R * D, rng, 3.0, ha);
        const yk0 = try bf16s(R * DK, rng, 3.0, ha);
        for ([_][]u8{ qa0, yk0 }) |x| {
            const words = std.mem.bytesAsSlice(u16, x);
            for (0..@max(words.len / 1000, 4)) |_| words[rng.uintLessThan(usize, words.len)] = specials[rng.uintLessThan(usize, specials.len)];
        }
        const pos = try ha.alloc(i64, R);
        const slots = try ha.alloc(i64, R);
        for (pos, slots, 0..) |*q, *s, i| {
            q.* = @intCast(rng.uintLessThan(usize, maxpos));
            s.* = if (i % 5 == 4) -1 else @intCast((i * 7 + rng.uintLessThan(usize, 3)) % ring_rows);
        }
        const pb = try upload(d, std.mem.sliceAsBytes(pos));
        const sb = try upload(d, std.mem.sliceAsBytes(slots));
        const qa = try upload(d, qa0);
        const yk = try upload(d, yk0);
        try keep.append(ha, pb);
        try keep.append(ha, sb);
        try keep.append(ha, qa);
        try keep.append(ha, yk);
        const ring0 = try bf16s(ring_rows * DK, rng, 1.0, ha);
        var outs: [2]struct { qr: cuda.DeviceBuffer, kout: cuda.DeviceBuffer, ring: cuda.DeviceBuffer } = undefined;
        for (&outs) |*o| {
            o.* = .{ .qr = try cuda.DeviceBuffer.alloc(d, R * D * 2), .kout = try cuda.DeviceBuffer.alloc(d, R * DK * 2), .ring = try upload(d, ring0) };
            try o.qr.fill8(0x5a, null);
            try o.kout.fill8(0x5a, null);
            try keep.append(ha, o.qr);
            try keep.append(ha, o.kout);
            try keep.append(ha, o.ring);
        }
        try tri_basic.rmsnorm(r.t, qa.ptr, D, wq.ptr, outs[0].qr.ptr, D, 1e-6, R, D);
        try tri_norm.kvNormRope(r.t, yk.ptr, wk.ptr, cos.ptr, sin.ptr, pb.ptr, outs[0].ring.ptr, ring_rows, sb.ptr, 1e-6, true, RD, outs[0].kout.ptr, R, DK);
        const ok = try tri_norm.qKvNormPlain(r.t, qa.ptr, D, wq.ptr, 1e-6, yk.ptr, wk.ptr, cos.ptr, sin.ptr, pb.ptr, outs[1].ring.ptr, ring_rows, sb.ptr, true, RD, outs[1].qr.ptr, outs[1].kout.ptr, R, D, DK);
        if (!ok) return r.note("qkv_plain: the extra set holds the _q_kv_norm variant without rotations", false);
        try r.s.synchronize();
        const names = [_][]const u8{ "q's rows", "the window KV", "the ring" };
        const sizes = [_]usize{ R * D * 2, R * DK * 2, ring_rows * DK * 2 };
        for (names, sizes, 0..) |nm, sz, k| {
            const a = switch (k) {
                0 => outs[0].qr.ptr,
                1 => outs[0].kout.ptr,
                else => outs[0].ring.ptr,
            };
            const b = switch (k) {
                0 => outs[1].qr.ptr,
                1 => outs[1].kout.ptr,
                else => outs[1].ring.ptr,
            };
            const want = try get(a, sz, d);
            defer free(want);
            const got = try get(b, sz, d);
            defer free(got);
            r.report(try std.fmt.bufPrint(&line, "qkv_plain: {d} rows, {s}: the fused launch against rmsnorm then kv_norm_rope", .{ R, nm }), differs(want, got), true);
        }
        // control: one q value changed changes q's normed rows (the instrument sees a difference). Over a row 0 that is finite (finiteControl): a random
        // row that carries a NaN / inf / huge word norms to the same all-NaN row with and without the change (qa0 does at 1 row), and sees nothing
        const pair = try rio.finiteControl(ha, qa0, D * 2);
        const qa2 = try upload(d, pair.clean);
        const qa3 = try upload(d, pair.changed);
        try keep.append(ha, qa2);
        try keep.append(ha, qa3);
        const qr2 = try cuda.DeviceBuffer.alloc(d, R * D * 2);
        const qr3 = try cuda.DeviceBuffer.alloc(d, R * D * 2);
        try keep.append(ha, qr2);
        try keep.append(ha, qr3);
        try tri_basic.rmsnorm(r.t, qa2.ptr, D, wq.ptr, qr2.ptr, D, 1e-6, R, D);
        try tri_basic.rmsnorm(r.t, qa3.ptr, D, wq.ptr, qr3.ptr, D, 1e-6, R, D);
        try r.s.synchronize();
        const w0 = try get(qr2.ptr, R * D * 2, d);
        defer free(w0);
        const w3 = try get(qr3.ptr, R * D * 2, d);
        defer free(w3);
        if (differs(w0, w3) == 0) changed_all = false;
    }
    r.note("qkv_plain control: one changed q value changed q's rows in every case", changed_all);
    r.note("qkv_plain: every call found the fused variant in the extra set (none made the two launches)", tri.fuse_misses.load(.monotonic) == misses);
}

// -- TF_DS_QGATE_CANARY 2 / 3 ---------------------------------------------------------------------------------------

/// The body canary's values: every 32-value group normal values at its own power of two (2^-30 .. 2^30), every 97th
/// group zeros, a negative zero in every 89th, a value 5,000 times its group's magnitude in every 83rd.
fn canaryValues(n: usize, rng: std.Random, a: std.mem.Allocator) ![]f32 {
    const v = try a.alloc(f32, n);
    for (0..n / 32) |g| {
        const sc = std.math.ldexp(@as(f32, 1.0), rng.intRangeAtMost(i32, -30, 30));
        const grp = v[g * 32 ..][0..32];
        for (grp) |*x| x.* = rng.floatNorm(f32) * sc;
        if (g % 97 == 5) @memset(grp, 0);
        if (g % 89 == 7) grp[3] = -0.0;
        if (g % 83 == 11) grp[9] = 5000 * sc;
    }
    return v;
}

fn qgateBody(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    const shapes = [_][2]usize{ .{ 1, 2560 }, .{ 3, 5120 }, .{ 17, 2560 }, .{ 48, 2560 }, .{ 2048, 5120 } };
    var diff2: usize = 0;
    var moved: usize = 0;
    var diff3: usize = 0;
    for (shapes) |sh| {
        const rows = sh[0];
        const w = sh[1];
        const n = rows * w;
        const x = try canaryValues(n, rng, ha);
        // (the copies through replay_io: in the stream's order)
        var xb = try cuda.DeviceBuffer.alloc(d, n * 4);
        defer xb.free();
        try rio.write(xb, 0, std.mem.sliceAsBytes(x));
        var fb = try cuda.DeviceBuffer.alloc(d, n * 2);
        defer fb.free();
        // mode 2: act_quant's own quant-dequant of the same rows (bf16), then the canary in place
        try poison(fb.ptr, fb.len, d, r.s);
        try mx.fake(r.k, r.s, xb.ptr, .f32, w, rows, w, fb.ptr, w);
        try mx.qgateBody(r.k, r.s, xb.ptr, n, 2);
        try r.s.synchronize();
        const fake = try get(fb.ptr, n * 2, d);
        defer free(fake);
        const got2 = try get(xb.ptr, n * 4, d);
        defer free(got2);
        const want2 = try ha.alloc(u32, n);
        for (want2, 0..) |*o, i| o.* = @as(u32, std.mem.readInt(u16, fake[i * 2 ..][0..2], .little)) << 16;
        diff2 += differs(std.mem.sliceAsBytes(want2), got2);
        moved += differs(std.mem.sliceAsBytes(x), got2);
        // mode 3: the specials first, then each value's bits plus one
        const special = [_]u32{ 0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x7f7fffff, 0xff7fffff, 0x7f7ffffe, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x3f800000, 0xbf800000 };
        const bits = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(x));
        @memcpy(bits[0..special.len], &special);
        try rio.write(xb, 0, std.mem.sliceAsBytes(x));
        try mx.qgateBody(r.k, r.s, xb.ptr, n, 3);
        try r.s.synchronize();
        const got3 = try get(xb.ptr, n * 4, d);
        defer free(got3);
        const want3 = try ha.alloc(u32, n);
        for (want3, bits) |*o, u| {
            const mag = u & 0x7fffffff;
            o.* = if (mag != 0 and mag < 0x7f7fffff) u + 1 else u;
        }
        diff3 += differs(std.mem.sliceAsBytes(want3), got3);
    }
    r.report("qgate_body 2: layer 0's partials (1 .. 2,048 rows of 2,560 and 5,120 columns) one FP8 step in place against act_quant's quant-dequant of the same values (mx.fake, widened)", diff2, true);
    r.report("qgate_body 2 control: the partials against their FP8 step", moved, false);
    r.report("qgate_body 3: each nonzero finite value's bits plus one, zeros, infinities, NaN and the largest finite values kept", diff3, true);
}

    .{ "dec_group2", decGroup2, 13 },
    .{ "pass_rule", passRule, 3 },
    .{ "pass_tail", passTail, 7 },
    .{ "attn_epilogue", attnEpilogue, 3 },
    .{ "hc_stride", hcStride, 3 },
    .{ "hc_fused", hcFused, 6 },
    .{ "gemm_r2", gemmR2, 6 },
    .{ "chained_door", chainedDoor, 3 },
    .{ "chained_two_base", rociTwoBaseChained, 3 },
    .{ "chained_three_base", rociThreeBaseChained, 3 },
    .{ "assemble_quant", assembleQuant, 2 },
    .{ "qgate_body", qgateBody, 3 },
    .{ "idx_v", idxV, 6 },
    .{ "idx_t", idxT, 6 },
    .{ "act_quant", actQuant, 6 },
    .{ "woa_merge", woaMerge, 4 },
    .{ "merge_d", mergeD, 14 }, // 10 + the 16-column width at rows 1, 2, 6 and 16 (TF_DS_MERGE_D=3)
    .{ "attn_w", attnW, 14 },
    .{ "attn_x", attnX, 14 },
    .{ "attn_z", attnZ, 14 },
    .{ "attn_w_decode", attnWDecode, 7 },
    .{ "attn_x_decode", attnXDecode, 7 },
    .{ "attn_z_decode", attnZDecode, 7 },
    .{ "rope_w", ropeW, 9 + 2 },
    .{ "qkv_plain", qkvPlain, 4 * 3 + 2 },
    .{ "draft_main4", draftMain4, 8 },
    .{ "mma_sf", mmaSf, 32 },
    .{ "mma_sf_exact", mmaSfExact, 22 },
    .{ "gemm_w16", gemmW16, 13 },
    .{ "split64n", split64n, 6 },
    .{ "pf_t64", pfT64, 4 },
    .{ "ring_wc", ringWc, 6 },
    .{ "ring_inplace", ringInplace, 8 },
    .{ "ring_word", ringWord, 9 },
    .{ "roci_readers", rociReaders, 1 + 7 * 3 + 3 },
    .{ "roci_stage", rociStage, 2 * 4 + 1 },
    .{ "roci_collect", rociCollect, 3 },
    .{ "roci_split_cost", rociSplitCost, 2 },
    .{ "roci_combine_first", rociCombineFirst, 5 },
    .{ "roci_wc_classes", rociWcClasses, 20 },
    .{ "roci_local_first", rociLocalFirst, 15 },
    .{ "roci_sent_first", rociSentFirst, 5 },
    .{ "roci_door", rociDoor, 13 },
    .{ "roci_boundary", rociBoundary, 3 + 1 + 1 },
    .{ "roci_codes", rociCodes, 2 * 2 + 1 + 1 },
    .{ "roci_local_first_wqb", rociLocalFirstWqb, 3 },
    .{ "roci_two_base", rociTwoBase, 3 },
    .{ "roci_three_base", rociThreeBase, 3 },
    .{ "roci_any_rows", rociAnyRows, 3 * 8 },
    .{ "roci_two_base_pages", rociTwoBasePages, 4 * 2 + 1 },
    .{ "roci_three_base_pages", rociThreeBasePages, 4 * 2 },
    .{ "roci_any_experts", rociAnyExperts, 2 },
    .{ "roci_any_index", rociAnyIndex, 6 },
    .{ "roci_any_engram", rociAnyEngram, 5 },
    .{ "roci_dec_rows", rociDecRows, 3 * 8 },
    .{ "roci_dec_mixes", rociDecMixes, 4 + 4 + 3 + 3 },
    .{ "roci_dec_kset", rociDecKset, 3 },
    .{ "roci_pair", rociPair, 6 },
    .{ "roci_triple", rociTriple, 3 },
    .{ "pack_epilogue", packEpilogue, 3 },
    .{ "attn_epi_fused", attnEpiFused, 5 + 1 + 1 },
    .{ "attn_epi_we", attnEpiWe, 5 + 1 + 1 },
    .{ "attn_epi_xe", attnEpiXe, 5 + 1 + 1 },
    .{ "hc_fused_2b", hcFused2b, 2 + 1 + 1 },
    .{ "hc_2s", hc2s, 3 + 1 + 1 },
const replay_lines: usize = (3 + 356 + 9 + 3 + 2 + 5 + 23 + 15 + 5 + 13 + 5 + 6 + 3 + 3 + 3 + 24 + 9 + 8 + 2 + 6 + 5 + 24 + 14 + 3 + 6 + 6 + 13 + 2 + 3 + 3 + 6 + 6 + 6 + 8 + 9 + 10 + 18 + 3 + 3 + 2 + 13 + 10 + 53 + 5 + 21 + 9 + 11 + 5 + 2 + 5 + 1 + 14 + 5 + 1 + 1 + 5 + 1 + 1 + 6 + 4 + 6 + 6 + 4 + 6 + 4) - loadonly_lines + (if (instruments) loadonly_lines else 0); // + 4: merge_d's 16-column width; + 6: idx_t (TF_DS_IDX_TPP=4); + 6: split64n (4 byte lines, 2 kernel time lines); + 4: pf_t64 (loaded, bytes, kernel time, control)
pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("usage: tf-dsv41-kernels-replay AOT_DIR\n", .{});
        return 2;
    }
    if (!cuda.kernels.available) {
        std.debug.print("tf-dsv41-kernels-replay: this build has no kernel images: run the image's binary\n", .{});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var set = try cuda.aot.Set.load(init.gpa, io, &driver, ctx.device, args[1]);
    defer set.deinit();
    // the kit's extra set beside it (aot-extra/: TF_DS_ATTN_U's twins); without one its lines fail
    var extra: ?cuda.aot.Set = cuda.aot.Set.load(init.gpa, io, &driver, ctx.device, try std.fs.path.join(a, &.{ args[1], "..", "aot-extra" })) catch null;
    defer if (extra) |*x| x.deinit();
    var k = try mx.Kernels.load(&driver, cuda.kernels.dsv41_native);
    defer k.unload();
    try k.useSplit(.a); // the configurations' kernels; each check sets the switch it runs
    k.split = .off;
    try k.useW8();
    try k.useGemm5();
    k.gemm5 = false;
    try k.useGroupTiles();
    k.group_tiles = false;
    try k.useTopkTiles();
    k.topk_tiles = false;
    try k.useRun128();
    k.run128 = false;
    try k.useQgateBody();
    // TF_DS_MMA_SF's kernels (sm_121a: another GPU leaves them unloaded, and the section says so); each check sets the
    // arithmetic it runs
    k.useMmaSf(&driver, cuda.kernels.dsv41_native_sf, .chained) catch {};
    k.sf = .off;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var ops = try dsv41.ops.Ops.load(&driver, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    defer ops.unload();
    var r: Run = .{ .d = &driver, .dev = ctx.device, .s = stream, .t = .{ .set = &set, .extra = if (extra) |*x| x else null, .stream = stream, .pdl = true }, .k = &k, .a = arena.allocator(), .ops = &ops, .io = io };
    var prng = std.Random.DefaultPrng.init(0x6b32);
    if (r.ok) {
        std.debug.print("kernels-replay: every same-bits switch's launches give the baseline bytes, TF_DS_MMA_SF's within its bound of fp64 beyond the baseline's; every control as it must ({d} lines, {d} sections)\n", .{ r.lines, ran });
        return 0;
    }
    std.debug.print("kernels-replay: FAILED (the lines above)\n", .{});
    return 1;
}
// -- TF_DS_SPLIT64N -------------------------------------------------------------------------------------------------------

/// mx.gemm of rows [0, m) with the exact arithmetic (TF_DS_MMA_SF_EXACT) under TF_DS_SPLIT64N on or off, its output of `mode` into y
/// (poisoned first); its bytes (get's). The key must be proven (mx.sfProve) before.
fn split64nOnce(r: *Run, on: bool, l: weights.Linear, q: u64, s: u64, z: u64, m: usize, y: u64, mode: Mode) ![]u8 {
    r.k.sf = .exact;
    r.k.split64n = on;
    defer {
        r.k.sf = .off;
        r.k.split64n = false;
    }
    return gemmOnce(r, .off, false, l, q, s, z, 0, m, y, mode, null);
}

/// TF_DS_SPLIT64N's 64 x 64 tile against the 64 x 128 one: wkv (N 512) and wq_a (N 1,280) at 1, 37, 1,000, 1,024 and 4,096 rows
/// (the split64 tile's ragged row blocks and its last row count, the 128-row tile past it, and the rows below the tile GEMMs'
/// own), bf16 and fp32 outputs: every output byte equal, each launch's route asserted (the 64-column kernel exactly where the
/// split64 tile runs), the kernel's grid covering every column (an output column it left unwritten would still hold the
/// poison). One code byte of row 700 changed differs. Then each shape's kernel time at 1,024 rows (a 4,096-row pass's owned rows): the
/// 64 x 128 tile, the same again (the null control), the 64 x 64 tile, best of three of 20 launches, bf16 out; a time line, its
/// reading in its text (the prompt test cannot see a difference this size: x0.998 against two readings 0.2 % apart).
fn split64n(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var line: [512]u8 = undefined;
    const sk = r.k.sfk;
    r.note("split64n: the 64-column split64 kernel is loaded", sk != null and sk.?.split64n != null);
    if (sk == null or sk.?.split64n == null) return;
    defer {
        r.k.sf = .off;
        r.k.split64n = false;
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
    }
    const rows = [_]usize{ 1, 37, 1000, 1024, 4096 };
    for ([_]Shape{ split_shapes[1], split_shapes[0] }) |sh| {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
        const max_m = rows[rows.len - 1];
        const x = try upload(d, try bf16s(max_m * sh.k, rng, 0.5, ha));
        try keep.append(ha, x);
        const q = try cuda.DeviceBuffer.alloc(d, max_m * sh.k);
        try keep.append(ha, q);
        const s = try cuda.DeviceBuffer.alloc(d, max_m * (sh.k / 32) * 4);
        try keep.append(ha, s);
        try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, max_m, sh.k, q.ptr, sh.k, s.ptr, sh.k / 32);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, max_m), 1) * 4);
        try keep.append(ha, z);
        const y = try cuda.DeviceBuffer.alloc(d, max_m * sh.n * 4);
        try keep.append(ha, y);
        r.k.sf_nkept = 0;
        r.k.sf_nproven = 0;
        try mx.sfProve(r.k, l.s);
        const sp = mx.splits(sh.k, sh.n);
        var route = true;
        var taken: usize = 0;
        for (rows) |m| {
            const kind = mx.sfGemmOf(.exact, m, sp);
            const want = kind != null and kind.? == .split64;
            taken += @intFromBool(want);
            route = route and (sk.?.split64nOf(true, 1, sh.n, false, false) != null) and (sk.?.split64nOf(false, 1, sh.n, false, false) == null) and (want == (m >= 64 and m <= 1024));
        }
        var diff: usize = 0;
        for ([_]Mode{ .bf16, .f32 }) |mode| for (rows) |m| {
            const off = try split64nOnce(r, false, l, q.ptr, s.ptr, z.ptr, m, y.ptr, mode);
            defer free(off);
            const on = try split64nOnce(r, true, l, q.ptr, s.ptr, z.ptr, m, y.ptr, mode);
            defer free(on);
            diff += differs(off, on);
        };
        r.report(try std.fmt.bufPrint(&line, "split64n {s} ({d} x {d}): bf16 and fp32 outputs at 1, 37, 1,000, 1,024 and 4,096 rows against the 64 x 128 tile (the 64-column kernel at {d} of the 5 row counts, its route asserted{s})", .{ sh.name, sh.k, sh.n, taken, if (route) "" else ": WRONG" }), if (route) diff else 1, true);
        {
            // the kernel time at the owned rows of a 4,096-row pass
            const T = struct {
                r: *Run,
                l: weights.Linear,
                q: u64,
                s: u64,
                z: u64,
                y: u64,
                on: bool,
                fn go(t: @This()) anyerror!void {
                    t.r.k.sf = .exact;
                    t.r.k.split64n = t.on;
                    try mx.gemm(t.r.k, t.r.s, t.l, t.q, t.l.k, t.s, t.l.k / 32, 1024, t.y, t.l.n, .bf16, false, t.z);
                }
            };
            const base: T = .{ .r = r, .l = l, .q = q.ptr, .s = s.ptr, .z = z.ptr, .y = y.ptr, .on = false };
            var tiled = base;
            tiled.on = true;
            const ms128 = try kernelBest(r, T, base, T.go);
            const msn = try kernelBest(r, T, base, T.go);
            const ms64 = try kernelBest(r, T, tiled, T.go);
            r.k.sf = .off;
            r.k.split64n = false;
            const nul = (msn / ms128 - 1.0) * 100.0;
            const txt = try std.fmt.bufPrint(&line, "split64n {s} ({d} x {d}) kernel only at 1,024 rows, bf16 out, ms a launch: 64 x 128 tile {d:.4}, null control {d:.4} ({d:.1} %, {s}), 64 x 64 tile {d:.4}: {d:.1} %; at one launch a layer of 40: {d:.2} against {d:.2} ms a pass", .{ sh.name, sh.k, sh.n, ms128, msn, nul, resolved(nul), ms64, (ms64 / ms128 - 1.0) * 100.0, ms128 * 40, ms64 * 40 });
            r.note(txt, true); // a time line: its bytes are the line above, the null control's reading is in its text
        }
        if (sh.n == 1280) {
            // the control: one code byte of row 700, the 1,000-row launch
            const base = try split64nOnce(r, true, l, q.ptr, s.ptr, z.ptr, 1000, y.ptr, .bf16);
            defer free(base);
            var code: [1]u8 = undefined;
            const at = q.ptr + 700 * sh.k + 7;
            try take(d, &code, at, 1);
            const changed = [1]u8{code[0] ^ 0x08};
            try put(d, at, &changed, 1);
            const moved = try split64nOnce(r, true, l, q.ptr, s.ptr, z.ptr, 1000, y.ptr, .bf16);
            defer free(moved);
            try put(d, at, &code, 1);
            r.report("split64n control: one code byte of row 700 changed, wq_a's 1,000-row launch on the 64 x 64 tile", differs(base, moved), false);
        }
    }
}

// -- TF_DS_PF_T64 ---------------------------------------------------------------------------------------------------------

/// The router's bf16 GEMM (mx.bf16Gemm: K 5,120, N 384, one group) of `m` rows with TF_DS_PF_T64 on or off, its `y_dtype` output into y
/// (poisoned first); its bytes (get's).
fn pfT64Once(r: *Run, on: bool, x: u64, w: u64, m: usize, y: u64, y_dtype: mx.DType) ![]u8 {
    const esz: usize = if (y_dtype == .f32) 4 else 2;
    const bytes = m * 384 * esz;
    try poison(y, bytes, r.d, r.s);
    r.k.pf_t64 = on;
    defer r.k.pf_t64 = false;
    try mx.bf16Gemm(r.k, r.s, x, 5120, w, 5120, m, 384, 5120, y, 384, y_dtype, 1, false, 1, 0, 0, 0, null);
    try r.s.synchronize();
    return get(y, bytes, r.d);
}

/// TF_DS_PF_T64's 64 x 64 tile against the 128 x 128 one: the router's logits GEMM (K 5,120, N 384) at 1, 37, 1,000, 1,024 and 4,096 rows
/// (below the tile GEMMs' own 64 rows both arms run the same kernel), fp32 and bf16 outputs: every output byte equal, the route asserted
/// (the 64 x 64 kernel for N 384 of one group only, from 64 rows), the grid covering every column and row (one left unwritten would still
/// hold the poison). One code byte of row 700 changed differs. Then the kernel time at 1,024 rows (a 4,096-row pass's owned rows): the
/// 128 x 128 tile, the same again (the null control), the 64 x 64 tile, best of three of 20 launches, fp32 out; a time line, the null
/// control's reading in its text, never failing the byte proof.
fn pfT64(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var line: [512]u8 = undefined;
    const have = r.k.usePfT64(true);
    r.k.pf_t64 = false;
    r.note("pf_t64: the 64 x 64 bf16 tile is loaded", have);
    if (!have) return;
    const rows = [_]usize{ 1, 37, 1000, 1024, 4096 };
    const max_m = rows[rows.len - 1];
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const x = try upload(d, try bf16s(max_m * 5120, rng, 0.5, ha));
    try keep.append(ha, x);
    const w = try upload(d, try bf16s(384 * 5120, rng, 0.05, ha));
    try keep.append(ha, w);
    const y = try cuda.DeviceBuffer.alloc(d, max_m * 384 * 4);
    try keep.append(ha, y);
    var route = true;
    var taken: usize = 0;
    for (rows) |m| {
        const want = m >= mx.pf_min_rows;
        taken += @intFromBool(want);
        route = route and (mx.pfT64Of(r.k, true, 384, 1) != null) and (mx.pfT64Of(r.k, false, 384, 1) == null) and (mx.pfT64Of(r.k, true, 128, 1) == null) and (mx.pfT64Of(r.k, true, 384, 2) == null);
    }
    var diff: usize = 0;
    for ([_]mx.DType{ .f32, .bf16 }) |dt| for (rows) |m| {
        const off = try pfT64Once(r, false, x.ptr, w.ptr, m, y.ptr, dt);
        defer free(off);
        const on = try pfT64Once(r, true, x.ptr, w.ptr, m, y.ptr, dt);
        defer free(on);
        diff += differs(off, on);
    };
    r.report(try std.fmt.bufPrint(&line, "pf_t64 router (5120 x 384): fp32 and bf16 outputs at 1, 37, 1,000, 1,024 and 4,096 rows against the 128 x 128 tile (the 64 x 64 kernel at {d} of the 5 row counts, its route asserted{s})", .{ taken, if (route) "" else ": WRONG" }), if (route) diff else 1, true);
    {
        const T = struct {
            r: *Run,
            x: u64,
            w: u64,
            y: u64,
            on: bool,
            fn go(t: @This()) anyerror!void {
                t.r.k.pf_t64 = t.on;
                try mx.bf16Gemm(t.r.k, t.r.s, t.x, 5120, t.w, 5120, 1024, 384, 5120, t.y, 384, .f32, 1, false, 1, 0, 0, 0, null);
            }
        };
        const base: T = .{ .r = r, .x = x.ptr, .w = w.ptr, .y = y.ptr, .on = false };
        var tiled = base;
        tiled.on = true;
        const ms128 = try kernelBest(r, T, base, T.go);
        const msn = try kernelBest(r, T, base, T.go);
        const ms64 = try kernelBest(r, T, tiled, T.go);
        r.k.pf_t64 = false;
        const nul = (msn / ms128 - 1.0) * 100.0;
        const txt = try std.fmt.bufPrint(&line, "pf_t64 router (5120 x 384) kernel only at 1,024 rows, fp32 out, ms a launch: 128 x 128 tile {d:.4}, null control {d:.4} ({d:.1} %, {s}), 64 x 64 tile {d:.4}: {d:.1} %; at 18.8 launches a pass: {d:.2} against {d:.2} ms", .{ ms128, msn, nul, resolved(nul), ms64, (ms64 / ms128 - 1.0) * 100.0, ms128 * 18.8, ms64 * 18.8 });
        r.note(txt, true); // a time line: its bytes are the line above, the null control's reading is in its text
    }
    {
        // the control: one value of row 700 changed, the 1,000-row launch on the 64 x 64 tile
        const base = try pfT64Once(r, true, x.ptr, w.ptr, 1000, y.ptr, .f32);
        defer free(base);
        var v: [2]u8 = undefined;
        const at = x.ptr + (700 * 5120 + 7) * 2;
        try take(d, &v, at, 2);
        const changed = [2]u8{ v[0], v[1] ^ 0x08 };
        try put(d, at, &changed, 2);
        const moved = try pfT64Once(r, true, x.ptr, w.ptr, 1000, y.ptr, .f32);
        defer free(moved);
        try put(d, at, &v, 2);
        r.report("pf_t64 control: one value of row 700 changed, the 1,000-row launch on the 64 x 64 tile", differs(base, moved), false);
    }
}


// -- TF_DS_PF_RASTER -------------------------------------------------------------------------------------------------

const raster_groups = [_]u32{ 2, 4, 8, 16 };
const raster_rows = [_]usize{ 4096, 8192, 16384, 8200 };
const raster_max_rows: usize = 16384;

/// One launch of mx.gemm of rows [0, m) with TF_DS_PF_RASTER `g` (0: default order), TF_DS_GEMM_HALVES `halves`, TF_DS_MMA_SF
/// `sw` and TF_DS_PF_SPLIT `split` in effect (each put back).
fn rasterGemm(r: *Run, g: u32, halves: bool, sw: mx.MmaSf, split: mx.PfSplit, l: weights.Linear, q: u64, s: u64, z: u64, m: usize, y: u64, fp32: bool) !void {
    const was = .{ mx.pf_raster, mx.gemm_halves, r.k.sf, r.k.split };
    defer {
        mx.pf_raster = was[0];
        mx.gemm_halves = was[1];
        r.k.sf = was[2];
        r.k.split = was[3];
    }
    mx.pf_raster = g;
    mx.gemm_halves = halves;
    r.k.sf = sw;
    r.k.split = split;
    try mx.gemm(r.k, r.s, l, q, l.k, s, l.k / 32, m, y, l.n, if (fp32) .f32 else .bf16, false, z);
}

/// rasterGemm's bytes: y poisoned first, read back after the launch.
fn rasterBytes(r: *Run, g: u32, halves: bool, sw: mx.MmaSf, split: mx.PfSplit, l: weights.Linear, q: u64, s: u64, z: u64, m: usize, y: u64, fp32: bool) ![]u8 {
    const n = m * l.n * @as(usize, if (fp32) 4 else 2);
    try poison(y, n, r.d, r.s);
    try rasterGemm(r, g, halves, sw, split, l, q, s, z, m, y, fp32);
    try r.s.synchronize();
    return get(y, n, r.d);
}

/// A dense FP8 shape's operands for rasterGemm at raster_max_rows rows (the activation quantized once).
const RasterOps = struct {
    l: weights.Linear,
    q: cuda.DeviceBuffer,
    s: cuda.DeviceBuffer,
    z: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    fp32: bool,

    fn make(r: *Run, rng: std.Random, ha: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer), sh: Shape, fp32: bool) !RasterOps {
        const d = r.d;
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, keep);
        const x = try upload(d, try bf16s(raster_max_rows * sh.k, rng, 0.5, ha));
        try keep.append(ha, x);
        const q = try cuda.DeviceBuffer.alloc(d, raster_max_rows * sh.k);
        const s = try cuda.DeviceBuffer.alloc(d, raster_max_rows * (sh.k / 32) * 4);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, raster_max_rows), 1) * 4);
        const y = try cuda.DeviceBuffer.alloc(d, raster_max_rows * sh.n * @as(usize, if (fp32) 4 else 2));
        for ([_]cuda.DeviceBuffer{ q, s, z, y }) |b| try keep.append(ha, b);
        try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, raster_max_rows, sh.k, q.ptr, sh.k, s.ptr, sh.k / 32);
        return .{ .l = l, .q = q, .s = s, .z = z, .y = y, .fp32 = fp32 };
    }
};

/// Bytes that differ between the default order and each group (and the group beside TF_DS_GEMM_HALVES) at the rows of
/// raster_rows, on the kernels `sw` and `split` select.
fn rasterDiff(r: *Run, o: RasterOps, sw: mx.MmaSf, split: mx.PfSplit) !usize {
    var diff: usize = 0;
    for (raster_rows) |m| {
        const base = try rasterBytes(r, 0, false, sw, split, o.l, o.q.ptr, o.s.ptr, o.z.ptr, m, o.y.ptr, o.fp32);
        defer free(base);
        for (raster_groups) |g| {
            const got = try rasterBytes(r, g, false, sw, split, o.l, o.q.ptr, o.s.ptr, o.z.ptr, m, o.y.ptr, o.fp32);
            defer free(got);
            diff += differs(base, got);
        }
        // beside TF_DS_GEMM_HALVES (a block past 4,096 rows is a launch of its own: the group applies to each)
        const both = try rasterBytes(r, 8, true, sw, split, o.l, o.q.ptr, o.s.ptr, o.z.ptr, m, o.y.ptr, o.fp32);
        defer free(both);
        diff += differs(base, both);
    }
    return diff;
}

const RasterLaunch = struct {
    r: *Run,
    g: u32,
    halves: bool,
    sw: mx.MmaSf,
    o: RasterOps,
    m: usize,

    fn go(self: RasterLaunch) anyerror!void {
        try rasterGemm(self.r, self.g, self.halves, self.sw, .off, self.o.l, self.o.q.ptr, self.o.s.ptr, self.o.z.ptr, self.m, self.o.y.ptr, self.o.fp32);
    }
};

/// TF_DS_PF_RASTER's grouped tile order against the default order (see the section's header), every byte equal; the timing
/// lines are GB10's information (a mean of 5 launches after 3 warm ones, kernel time between two events).
fn pfRaster(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var line: [1024]u8 = undefined;
    // the gate: a launch below 14 MiB of activation keeps the default order, one at it takes the group
    {
        const wq_a = split_shapes[0];
    }
    const dense = [_]Shape{ split_shapes[0], split_shapes[1], split_shapes[3] }; // wq_a, wkv, wo_b (fp32)
    var ops: [3]RasterOps = undefined;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const ha = arena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    for (dense, 0..) |sh, i| ops[i] = try RasterOps.make(r, rng, ha, &keep, sh, i == 2);
    for (dense, ops) |sh, o| {
        const diff = try rasterDiff(r, o, .off, .off);
        r.report(try std.fmt.bufPrint(&line, "pf_raster {s} ({d} x {d}, {s}), the prefill tile kernels: 4,096, 8,192, 16,384 and 8,200 rows, groups 2, 4, 8 and 16 and group 8 beside TF_DS_GEMM_HALVES against the default order", .{ sh.name, sh.k, sh.n, if (o.fp32) "fp32" else "bf16" }), diff, true);
    }
    {
        const sh = dense[0];
        const was = r.k.split;
        defer r.k.split = was;
        var diff: usize = 0;
        for ([_]usize{ 8192, 16384, 8200 }) |m| {
            const base = try rasterBytes(r, 0, false, .off, .table, ops[0].l, ops[0].q.ptr, ops[0].s.ptr, ops[0].z.ptr, m, ops[0].y.ptr, false);
            defer free(base);
            const got = try rasterBytes(r, 8, false, .off, .table, ops[0].l, ops[0].q.ptr, ops[0].s.ptr, ops[0].z.ptr, m, ops[0].y.ptr, false);
            defer free(got);
            diff += differs(base, got);
        }
        r.report(try std.fmt.bufPrint(&line, "pf_raster TF_DS_PF_SPLIT=table {s} ({d} x {d}): 8,192, 16,384 and 8,200 rows, group 8 against the default order", .{ sh.name, sh.k, sh.n }), diff, true);
    }
    // the block-scaled kernels (chained: the same tile template as the exact form's; the exact form needs a proven key, the
    // order is the same code): wq_a bf16 and wo_b fp32
    for ([_]usize{ 0, 2 }) |i| {
        const sh = dense[i];
        const taken = r.k.sfk != null and sfTaken(r, .chained, ops[i].l, ops[i].q.ptr, ops[i].s.ptr, ops[i].y.ptr, 8192);
        const diff = if (taken) try rasterDiff(r, ops[i], .chained, .off) else 1;
        r.report(try std.fmt.bufPrint(&line, "pf_raster {s} ({d} x {d}), the block-scaled kernels{s}: 4,096, 8,192, 16,384 and 8,200 rows, groups 2, 4, 8 and 16 and group 8 beside TF_DS_GEMM_HALVES against the default order", .{ sh.name, sh.k, sh.n, if (taken) "" else " (NOT TAKEN: WRONG)" }), diff, true);
    }
    // the bf16 tile GEMM: the indexer's weights_proj (5,120 -> 4,096)
    {
        const K: usize = 5120;
        const N: usize = 4096;
        const x = try upload(d, try bf16s(raster_max_rows * K, rng, 0.5, ha));
        const w = try upload(d, try bf16s(N * K, rng, 0.05, ha));
        const y = try cuda.DeviceBuffer.alloc(d, raster_max_rows * N * 2);
        for ([_]cuda.DeviceBuffer{ x, w, y }) |b| try keep.append(ha, b);
        const B = struct {
            fn run(rr: *Run, g: u32, halves: bool, xp: u64, wp: u64, yp: u64, m: usize) ![]u8 {
                try poison(yp, m * N * 2, rr.d, rr.s);
                const was = .{ mx.pf_raster, mx.gemm_halves };
                defer {
                    mx.pf_raster = was[0];
                    mx.gemm_halves = was[1];
                }
                mx.pf_raster = g;
                mx.gemm_halves = halves;
                try mx.bf16Gemm(rr.k, rr.s, xp, K, wp, K, m, N, K, yp, N, .bf16, 1, false, 1, 0, 0, 0, null);
                try rr.s.synchronize();
                return get(yp, m * N * 2, rr.d);
            }
        };
        var diff: usize = 0;
        for (raster_rows) |m| {
            const base = try B.run(r, 0, false, x.ptr, w.ptr, y.ptr, m);
            defer free(base);
            for (raster_groups) |g| {
                const got = try B.run(r, g, false, x.ptr, w.ptr, y.ptr, m);
                defer free(got);
                diff += differs(base, got);
            }
            const both = try B.run(r, 8, true, x.ptr, w.ptr, y.ptr, m);
            defer free(both);
            diff += differs(base, both);
        }
        r.report("pf_raster weights_proj (5120 x 4096, bf16 tile GEMM): 4,096, 8,192, 16,384 and 8,200 rows, groups 2, 4, 8 and 16 and group 8 beside TF_DS_GEMM_HALVES against the default order", diff, true);
    }
    // the control: one activation code changed (row 5,000) differs under the group; the comparison reads the data
    {
        const o = ops[0];
        const base = try rasterBytes(r, 0, false, .off, .off, o.l, o.q.ptr, o.s.ptr, o.z.ptr, 8192, o.y.ptr, false);
        defer free(base);
        var code: [1]u8 = undefined;
        const at = o.q.ptr + 5000 * dense[0].k + 7;
        const changed = [1]u8{code[0] ^ 0x08};
        const moved = try rasterBytes(r, 8, false, .off, .off, o.l, o.q.ptr, o.s.ptr, o.z.ptr, 8192, o.y.ptr, false);
        defer free(moved);
        r.report("pf_raster control: one code byte of row 5000 changed, group 8, wq_a at 8,192 rows", differs(base, moved), false);
    }
    // GB10's timing: default, each group and halves, per shape at 8,192 and 16,384 rows
    const timed = [_]struct { name: []const u8, i: usize, sw: mx.MmaSf, what: []const u8 }{
        .{ .name = "wq_a", .i = 0, .sw = .off, .what = "the prefill tile kernels" },
        .{ .name = "wo_b", .i = 2, .sw = .off, .what = "the prefill tile kernels" },
        .{ .name = "wq_a", .i = 0, .sw = .chained, .what = "the block-scaled kernels" },
    };
    for (timed) |t| for ([_]usize{ 8192, 16384 }) |m| {
        const o = ops[t.i];
        const sh = dense[t.i];
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(sh.n)) * @as(f64, @floatFromInt(sh.k));
        var ms: [6]f32 = undefined;
        const cfg = [_]struct { g: u32, halves: bool }{ .{ .g = 0, .halves = false }, .{ .g = 2, .halves = false }, .{ .g = 4, .halves = false }, .{ .g = 8, .halves = false }, .{ .g = 16, .halves = false }, .{ .g = 0, .halves = true } };
        var live = t.sw == .off or r.k.sfk != null;
        for (cfg, 0..) |c, ci| ms[ci] = if (live) try kernelMs(r, RasterLaunch, .{ .r = r, .g = c.g, .halves = c.halves, .sw = t.sw, .o = o, .m = m }, RasterLaunch.go, 5) else 0;
        for (ms) |v| live = live and v > 0;
        var tf: [6]f64 = undefined;
        for (ms, 0..) |v, ci| tf[ci] = if (v > 0) flops / @as(f64, v) / 1e9 else 0;
        r.note(try std.fmt.bufPrint(&line, "pf_raster timing {s} ({d} x {d}), {s}, {d} rows, ms (TFLOPS): default {d:.3} ({d:.1}), group 2 {d:.3} ({d:.1}), 4 {d:.3} ({d:.1}), 8 {d:.3} ({d:.1}), 16 {d:.3} ({d:.1}), TF_DS_GEMM_HALVES {d:.3} ({d:.1}) (GB10's information)", .{ t.name, sh.k, sh.n, t.what, m, ms[0], tf[0], ms[1], tf[1], ms[2], tf[2], ms[3], tf[3], ms[4], tf[4], ms[5], tf[5] }), live);
    };
}
