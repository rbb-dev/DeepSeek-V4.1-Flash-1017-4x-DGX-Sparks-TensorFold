//! A prompt chunk's forward on one TP rank, the served build's Model._forward_k (model.py) step by step on the served
//! build's kernels: the streams of the chunk's tokens (embed_init), then a layer at a time the attention mixes (hc_pre, or
//! hc_pre_pf with the previous layer's gathered MoE partials posted first), attention (attention_k), the gather of its
//! partials, the FFN mixes with that post fused in (hc_pre_pf), the MoE and its gather, whose post goes into the next
//! layer's mixes (switch "hc_pf2"); before an Engram layer's mixes, its Engram (engram_apply). Compressed layers
//! (ratio 1, 2): a kv source's compressor and index keys into the caches (kv_source_update), the indexer's top-k
//! (prompt_keys: scored keys and the selection; from the candidate source on: fp32 scores and topk_indices), and the
//! sparse attention over the window (a later chunk's first keys from the ring) and the selected latents. Decode-sized
//! rows (1..128) take the grouped EXL3 linears. With replay (bounded replay prefill) the decoder's first layer cuts the
//! chunk to its rows at or past replay (replayCut); DSpark taps (tap) and the head on the last row (head) end it.
//! Not yet: ring mode (verify windows), a pool that leaves blocks out, a chunk starting inside a compressor group;
//! those return error.NotPortedYet.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const config_mod = @import("config.zig");
const aot = cuda.aot;
const weights = @import("weights.zig");
const tri = @import("tri.zig");
const tri_basic = @import("tri_basic.zig");
const tri_hc = @import("tri_hc.zig");
const tri_norm = @import("tri_norm.zig");
const tri_attn = @import("tri_attn.zig");
const tri_markov = @import("tri_markov.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const ops = @import("ops.zig");
const cublas = @import("cublas.zig");
const comm = @import("comm.zig");
const graph_fallback = @import("graph_fallback.zig");
const where = @import("where.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const engram_rows = @import("engram_rows.zig");
const wait_bell = @import("wait_bell.zig");
const exact = @import("exact.zig");
const tri_index = @import("tri_index.zig");
const idx_spread = @import("idx_spread.zig");
const roci_engine = @import("roci_engine.zig");
const roci = @import("rocinante.zig");
const slab_plan = @import("slab_plan.zig");
const exl3_linear = @import("exl3_linear.zig");
const mx = @import("mx.zig");
const prompt2d = @import("prompt2d.zig");
const keep_ahead = @import("keep_ahead.zig");
const rdma = @import("rdma.zig");
const round = @import("round.zig");
const lanes = @import("lanes"); // the pass rule (stream.passRows), for the worker's per-pass read-ahead window

/// A profile's (--profile 2) chunk timer (model.zig sets it while it times a chunk): the attention's and the MoE's
/// steps marked inside their phases (a_in, a_q, a_idx, a_core, a_woa; m_rt, m_gu, m_x), the rest of each in its own.
pub var ptimer: ?*round.PhaseTimer = null;

/// The most rows the attention front, wo_a and the linear matmul take the decode kernels with; above it they run the
/// prompt kernels (the router and the indexer's projection switch at the kernel set's own decode row count, 16). A
/// final pass at or under it takes a decode-class kernel a longer pass does not.
pub const decode_rows: usize = 128;

/// The monotonic clock (ns), for the engramWorker thread: no
/// std.Io handle reaches that thread, so this calls libc directly rather than going through Model.now's std.Io path.
fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

fn pmark(e: *const Engine, p: round.PhaseTimer.Phase) void {
    if (ptimer) |t| t.mark(e.s, p) catch {};
}

/// The indexer's [rows, n_comp] keys a row block holds at most (model.py: rb * n_comp <= 2^26 / 4 past 16 rows).
const key_elems: usize = 1 << 24;
/// A row block holds at least this many rows (the kit's index kernels were captured for 16-row-aligned views).
const key_rows_floor: usize = 16;

/// The most [rows, n_comp] keys a chunk's row block holds, for a context whose longest compressed length is `max_comp`:
/// key_elems (2^24, the 1,048,576-token bound) while the floor's 16 rows fit it, else the floor's 16 rows of the longest
/// n_comp. At the served 1M context this is exactly key_elems, so no buffer and no refusal changes there.
fn keyCapFor(max_comp: usize) usize {
    return @max(key_elems, key_rows_floor * max_comp);
}

/// Whether a row block of `rb` rows over `n_comp_end` compressed entries exceeds the chunk's key buffer.
fn keysRefused(rb: usize, n_comp_end: usize, key_cap: usize) bool {
    return rb * n_comp_end > key_cap;
}
/// A row block's rows at most where the layers before the candidate source prune (prunes: n_comp > 32,768, so rb =
/// 2^26 / (4 n_comp) < 512).
const pruned_rows: usize = 512;

/// model.py's RING_EXTRA: window ring slots beyond the window (a verify window never clobbers a key it reads).
pub const ring_extra = 16;
/// model.py's RAW: per-position compressor inputs kept (a ratio-2 group's earlier row for the next chunk).
pub const raw_rows = 64;

/// Device memory carved in 256-byte aligned pieces from blocks allocated as it grows, all freed together: the first
/// block init's size, then blocks of `block` bytes; a piece of `own` bytes or more takes a block of its own (the
/// pool's caches), so a pool of any size costs what its pieces take plus at most one block. `used`: the pieces'
/// bytes; `reserved`: the blocks'.
pub const Arena = struct {
    d: *const cuda.Driver,
    blocks: [max_blocks]cuda.DeviceBuffer = undefined,
    n: usize = 0,
    cur: ?usize = null, // the block small pieces come from
    at: usize = 0, // its next free byte
    used: usize = 0,
    reserved: usize = 0,

    const max_blocks = 1024;
    const block: usize = 256 << 20;
    const own: usize = 64 << 20;

    pub fn init(d: *const cuda.Driver, bytes: usize) !Arena {
        var a: Arena = .{ .d = d };
        if (bytes > 0) {
            a.cur = try a.grow(bytes);
            a.at = 0;
        }
        return a;
    }

    fn grow(a: *Arena, bytes: usize) !usize {
        if (a.n == max_blocks) return error.ArenaFull;
        a.blocks[a.n] = try cuda.DeviceBuffer.alloc(a.d, bytes);
        a.reserved += bytes;
        a.n += 1;
        return a.n - 1;
    }

    pub fn deinit(a: *Arena) void {
        for (a.blocks[0..a.n]) |*b| b.free();
        a.n = 0;
    }

    pub fn take(a: *Arena, bytes: usize) !u64 {
        if (bytes >= own) {
            const i = try a.grow(std.mem.alignForward(usize, bytes, 256));
            a.used += bytes;
            return a.blocks[i].ptr;
        }
        var at = std.mem.alignForward(usize, a.at, 256);
        if (a.cur == null or at + bytes > a.blocks[a.cur.?].len) {
            a.cur = try a.grow(block);
            at = 0;
        }
        a.at = at + bytes;
        a.used += bytes;
        return a.blocks[a.cur.?].ptr + at;
    }
};

/// The kinds of CUDA graphs: a round's stretches (round.Graphs), the drafter's passes (draft.PassGraphs).
pub const CaptureKind = enum { round, pass };

/// TF_DS_GRAPH_FAIL_TEST=RANK:N (round graphs) or RANK:N:pass (the drafter's pass graphs): a test of G0.3, never in
/// service: rank RANK's Nth capture of that kind reports a failed instantiate (its graph dropped). Unset: never.
pub fn injectedCaptureFailure(kind: CaptureKind, rank: u32, nth: u64) bool {
    const v = std.mem.span(std.c.getenv("TF_DS_GRAPH_FAIL_TEST") orelse return false);
    var it = std.mem.splitScalar(u8, v, ':');
    const r = std.fmt.parseInt(u32, it.next() orelse return false, 10) catch return false;
    const n = std.fmt.parseInt(u64, it.next() orelse return false, 10) catch return false;
    const want: CaptureKind = if (it.next()) |x| (if (std.mem.eql(u8, x, "pass")) .pass else return false) else .round;
    return r == rank and n == nth and want == kind;
}

/// The ranks' side of a capture (graph_fallback.Lazy.run): their votes, this rank's collective calls so far, a test's
/// injected failure.
pub const CaptureRanks = struct {
    e: *const Engine,
    kind: CaptureKind,

    pub fn agreeAll(r: CaptureRanks, ok: bool) !bool {
        return r.e.agreeAll(ok);
    }

    pub fn agreeCapture(r: CaptureRanks, stage: graph_fallback.Stage, recorded: u32) !graph_fallback.Outcome {
        return r.e.agreeCapture(stage, recorded);
    }

    /// The collectives a capture recorded that a rank could leave unmatched: none (the exchanges in a graph are the rings' kernels), so
    /// every rank's count is 0.
    pub fn calls(_: CaptureRanks) u32 {
        return 0;
    }

    pub fn injected(r: CaptureRanks, nth: u64) bool {
        return injectedCaptureFailure(r.kind, r.e.comm.rank, nth);
    }
};

/// A RoPE kind's tables on the device: cos, sin fp32 [rows, rope_dim / 2] (ops.RopeTables' rows).
pub const Rope = struct { cos: u64, sin: u64 };

/// What a chunk's steps run on: one rank's weights, kernels, stream and gather.
pub const Engine = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    t: tri.Tri,
    blas: *const cublas.Blas,
    comm: *const comm.Comm,
    pf: *const exl3_prefill.Kernels,
    lin: *const exl3_linear.Kernels,
    ex: *const exl3_experts.Kernels,
    ops: *const ops.Ops,
    exact: *const exact.Exact,
    c: *const Config,
    w: *const weights.Weights,
    world: usize,
    plain: Rope,
    compressed: Rope,
    round_rows: usize = 16, // a concurrent round's rows at most (TF_DS_ROUND_ROWS; round.max_rows at most)
    two: ?*const prompt2d.Two = null, // a node of the four-node 2D split (prompt2d.zig); `world` stays TP2's
    ring: ?*rdma.Ring = null, // the decode-size all-gathers over the RDMA ring (the served lane's: unused on the 2D split, whose rings are Two.rings)
    par: ?*const exl3_experts2d.ParKernels = null, // the parity split's gate / up in one launch (null: the served two)
    mx: ?*const mx.Kernels = null, // Format.native: DeepSeek's own FP8 / MXFP4 checkpoint's kernels (mx.zig)
    agree_flag: u64 = 0, // device bytes for agreeAll and agreeCapture (64: Model.init, layer_main); 0 at world > 1: they refuse

    /// G0.3 (docs/UNIFIED-DECODE-IMPROVEMENTS.md): whether `ok` holds on every rank, after a graph capture and
    /// instantiate and before any launch of it: one byte a rank set on the stream, the ranks' largest over the host links (1: a
    /// failure), read after a synchronize. Called in the same order on every rank (a new graph key comes up at the same
    /// round and stretch everywhere). World 1: this rank's own. A failure in the middle of recording on one rank only
    /// is not covered: the others' recorded collectives stay unmatched (no out-of-band abort exists).
    pub fn agreeAll(e: *const Engine, ok: bool) !bool {
        if (e.comm.world <= 1) return ok;
        if (e.agree_flag == 0) return error.NoAgreeFlag;
        try e.d.check(e.d.api.cuMemsetD8Async(e.agree_flag, @intFromBool(!ok), 1, e.s.handle), "cuMemsetD8Async");
        try e.comm.maxHost(e.d, e.agree_flag, 1, e.s);
        try e.s.synchronize();
        var v: [1]u8 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&v, e.agree_flag, 1), "cuMemcpyDtoH");
        return v[0] == 0;
    }

    /// The ranks' vote on a capture, after it and before any launch (graph_fallback: keep the graphs, run the work
    /// eagerly on every rank, or fail the round): this rank's ballot (how its capture went, the collective calls it
    /// recorded) made the ranks' byte-wise largest over the host links in the agreement buffer, read after a synchronize, so every
    /// rank decides the same. Called in the same order on every rank, as agreeAll. World 1: this rank's own ballot.
    pub fn agreeCapture(e: *const Engine, stage: graph_fallback.Stage, recorded: u32) !graph_fallback.Outcome {
        const prev = where.enter(.capture_vote, @backingInt(stage), recorded);
        defer where.leave(prev);
        const mine = graph_fallback.ballot(stage, recorded);
        if (e.comm.world <= 1) return graph_fallback.decide(mine);
        if (e.agree_flag == 0) return error.NoAgreeFlag;
        try e.d.copyHtoDAsync(@src(), e.agree_flag, &mine, mine.len, e.s.handle);
        try e.comm.maxHost(e.d, e.agree_flag, mine.len, e.s);
        try e.s.synchronize(); // (also the end of the copy from `mine`, on this stack)
        var v: [graph_fallback.ballot_len]u8 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&v, e.agree_flag, v.len), "cuMemcpyDtoH");
        return graph_fallback.decide(v);
    }

    /// A decode-size all-gather of `count` fp32 a rank into `dst`, rank after rank, over the RDMA ring (whole float4s that fit a
    /// slot: the start's decodeOff holds every decode exchange to that).
    pub fn gatherF32(e: *const Engine, src: u64, dst: u64, count: usize) !void {
        const r = e.ring orelse return error.DecodeNeedsRings;
        if (count % 4 != 0 or count * 4 > r.s.max_bytes) return error.RingRefused;
        return r.gather(e.s, src, @intCast(count / 4), dst);
    }

    /// The device address of the fixed send slot this stream's producer writes when the in-place switch is on, or
    /// null (no ring, the switch off, or a buffer larger than a slot: the caller allocates its own). The first call
    /// gives the stream its slot; every later gather on the same stream finds it.
    pub fn ringSendSlot(e: *const Engine, stream: cuda.Stream, bytes: usize) !?u64 {
        if (e.two) |tw| if (tw.rings) |rs| return rs.all4.sendSlotDev(stream, bytes) catch |err| switch (err) {
            error.BadInfo => return error.TooManyRingStreams,
            else => return err,
        };
        if (e.ring) |r| return r.sendSlotDev(stream, bytes) catch |err| switch (err) {
            error.BadInfo => return error.TooManyRingStreams,
            else => return err,
        };
        return null;
    }

    /// Before a producer that writes `ringSendSlot`'s buffer: wait (on the GPU) for the previous send from that slot to
    /// complete, so its bytes are not replaced while the NIC reads them. Nothing when the ring is not in place.
    pub fn prepareRingSend(e: *const Engine, stream: cuda.Stream) !void {
        if (e.two) |tw| if (tw.rings) |rs| {
            _ = try rs.all4.prepareSend(stream);
            return;
        };
        if (e.ring) |r| _ = try r.prepareSend(stream);
    }

    pub fn heads(e: *const Engine) usize {
        if (e.two) |t| return t.heads;
        return e.c.heads / e.world;
    }

    pub fn slots(e: *const Engine) usize {
        return e.c.top_k + 1;
    }

    /// A sequence's window ring for one layer: bf16 [window + RING_EXTRA, head_dim].
    pub fn ringBytes(e: *const Engine) usize {
        return (e.c.window + ring_extra) * e.c.head_dim * 2;
    }
};

/// A prompt chunk's buffers for up to `cap` rows (model.py _forward_k's and its steps' torch.empty's).
/// An image span of the prompt: positions [at, at + len), its rows from row `row` of the image rows buffer.
pub const Span = struct { at: usize, len: usize, row: usize };

/// The rows [r0, r1) of the chunk (relative to its start) a span covers, if any.
fn overlap(sp: Span, start: usize, n: usize) ?[2]usize {
    const lo = @max(sp.at, start);
    const hi = @min(sp.at + sp.len, start + n);
    return if (lo < hi) .{ lo - start, hi - start } else null;
}

pub const Chunk = struct {
    lane: u2 = 0, // the Rocinante lane this chunk's pass runs on (1: the second pass of a pair, 2: the third of a group of three)
    cap: usize, // the pass's view (Chunk.view): its owned-row paths' and plane strides' rows, at most rows_max
    rows_max: usize = 0, // the rows every buffer holds (init's cap)
    spans: []const Span = &.{}, // the prompt's image spans (absolute positions)
    img_rows: u64 = 0, // bf16 [span tokens, D]: their rows
    pick_vl: u64 = 0, // int32 [cap, slots]: the routing of span rows with the image bias, before it joins pick
    wts_vl: u64 = 0, // fp32 [cap, slots]
    n: usize = 0,
    start: usize = 0,
    ids: u64, // int64 [cap]
    pos: u64, // int64 [n]: start .. start + n (after a replay cut a view into pos0, pos[first:])
    pos0: u64, // int64 [cap]: the chunk's positions as begin() wrote them
    pos1: u64 = 0, // int64 [cap]: a replay cut's positions moved to a 16-byte boundary (replayCut)
    h: u64, // bf16 [cap, hc, D]: the streams
    h_alt: u64, // their second buffer (hc_pre_pf writes the posted streams there)
    x: u64, // bf16 [cap, D]: a block's input rows
    part: u64, // fp32 [cap * HC_BLOCKS * 32]
    pre: u64, // fp32 [cap, hc]: the layer's input pre (the previous layer's FFN pre_out)
    pre_a: u64, // fp32 [cap, hc]
    pre_f: u64, // fp32 [cap, hc]
    post: u64, // fp32 [cap, hc]
    comb: u64, // fp32 [cap, hc, hc]
    // attention
    qa: u64, // bf16 [cap, q_lora]
    y: u64, // bf16 [cap, head_dim]
    qr: u64, // bf16 [cap, q_lora]
    q: u64, // bf16 [cap, heads, head_dim]
    wsrc: u64, // bf16 [window - 1 + cap, head_dim]
    wlo: u64, // int64 [1]
    o: u64, // bf16 [cap, heads, head_dim]
    u: u64, // bf16 [cap, groups * o_lora]
    pa: u64, // fp32 [cap, D]: the attention partial
    ga: u64, // fp32 [world, cap, D]: its gather
    neg: u64, // int64 [cap], -1: no ring slot (model.py _neg)
    ring_idx: u64, // int64 [ring]
    // TF_DS_WINDOW_ONCE: the window indices a chunk's every layer uploads alike, made once per chunk (windowHave says which are in)
    ridx_g: u64 = 0, // int64 [ring]: the window keys' gather indices
    ridx_s: u64 = 0, // int64 [ring]: the ring write's scatter indices
    win_key: [4]usize = .{ std.math.maxInt(usize), 0, 0, 0 }, // (start, n, lo, ring mode) the uploads below were made for
    win_have: u8 = 0,
    rslot: u64 = 0, // int64 [RING_EXTRA]: ring mode's slots, pos % R
    pm_: u64 = 0, // fp32 [16, heads, 8]: the split attention's partials (ring mode's rows)
    pl_: u64 = 0,
    po_: u64 = 0, // fp32 [16, heads, 8, head_dim]
    invalid: u64, // uint32 [1]
    // compressed layers
    kvc: u64 = 0, // fp32 [cap, head_dim]: the compressor's kv projection
    scc: u64 = 0, // fp32 [cap, head_dim]: its gate scores
    lat: u64 = 0, // bf16 [cap, head_dim]: the compressed latents
    lat2: u64 = 0, // bf16 [cap, head_dim]: their RoPE'd copy (the cache's)
    ik: u64 = 0, // bf16 [cap, index_head_dim]: the index keys
    groups: u64 = 0, // int64 [cap]: the new latents' groups
    gpos: u64 = 0, // int64 [cap]: their positions (group * ratio)
    raw_idx: u64 = 0, // int64 [RAW]
    iq: u64 = 0, // bf16 [cap, index_heads * index_head_dim]
    iq4: u64 = 0, // bf16 [cap, index_heads * index_head_dim]: fp4_qd's bytes
    wl: u64 = 0, // fp32 [cap, index_heads]
    iw: u64 = 0, // bf16 [cap, index_heads]: the heads' weights
    vis: u64 = 0, // int64 [cap]: compressed entries a row sees
    keys: u64 = 0, // int64 [rows of a block, n_comp]: at most key_cap
    score: u64 = 0, // fp32 [rows of a block, n_comp]: the candidate pool's layers' index scores
    tmax: u64 = 0, // int64 [rows of a block, cdiv(n_comp, 64)]
    cidx: u64 = 0, // int64 [cap, index_topk]: the selected latents
    max_comp: usize = 0,
    key_cap: usize = key_elems, // the most keys a row block holds: keyCapFor(max_comp)
    // the candidate pool past candidate_blocks blocks (model.py _candidates, apply_candidates): the chunk's block mask
    // (u8 [dec_rows, pool_ms]), a row block's block maxima (fp32) and their top blocks (int64 [dec_rows,
    // candidate_blocks]): the candidate source and the layers after it run on the decoder half's rows alone; a
    // row's every_vis (no mask); the layers before the candidate source's pruned selection (tpos, cand: up to
    // pruned_rows rows)
    pool: u64 = 0,
    pool_ms: usize = 0,
    bmax: u64 = 0,
    pidx: u64 = 0,
    every: u64 = 0,
    pbufs: tri_index.PrunedBufs = .{ .tpos = 0, .cand = 0, .every = 0 },
    // MoE
    xf: u64, // fp32 [cap, D]
    gate_f: u64, // fp32 [experts, D]
    logits: u64, // fp32 [cap, experts]
    pick: u64, // int32 [cap, slots]
    wts: u64, // fp32 [cap, slots]
    pm: u64, // fp32 [cap, D]: the MoE partial
    gm: u64, // fp32 [world, cap, D]: its gather (the pending post)
    // WP4 (docs/prefill-plan.md, quarter-HC): a repack of ga's or gm's owned-row slice, tightly strided [world, q, D]
    // (q = cap / world) -- _hc_mix_pf / _hc_post take the gathered planes' per-plane stride as a runtime argument
    // (RS), so a reader at owned rows reads either the chunk's own cap-strided planes where they lie (TF_DS_HC_STRIDE) or
    // this tight copy. Zero size (and never touched) when there is no Two.
    qhc_g: u64 = 0,
    // WP4: `part` is pure intermediate scratch for the Sinkhorn/mix iteration (written by _hc_mix_pf, read back by
    // _hc_finish within the SAME call, never carrying state between calls) whose layout is documented only as a
    // flat element count, not a dimension order (unlike h/pre/post/comb, which ARE plainly per-row and safe to
    // pointer-offset -- tf-dsv41-hc-replay's own working test proves it for exactly those). A pointer offset into
    // the full-cap ch.part assumes a row-major layout that was never established; give it its own buffer, sized
    // tight for q = cap / world rows, exactly as the real engine would size ch.part for a q-row chunk.
    qhc_part: u64 = 0,
    // the gathers' word to their readers (gatherStep sets it, ownedPartials reads it) and whether plane 1 of qhc_g
    // holds -0.0 written in this call: host-side, reset by begin (every fillChunk call)
    ga_how: Gathered = .{},
    gm_how: Gathered = .{},
    tight_neg: bool = false,
    // the chunk runs no decoder row (replay >= start + n: it stops at the boundary layer), set by fillChunk before
    // layer 0 from the frame's replay; host-side, reset by begin
    encoder_only: bool = false,
    /// TF_DS_ROCI_ANY: the boundary layer's owned update and the handoff of the decoder rows are done; the layers after it are not owned
    boundary_done: bool = false,
    // TF_DS_OWNER_YX in effect: the owner sums' stage (prompt2d.OwnStage for cap / world rows); 0 otherwise
    own_stage: u64 = 0,
    /// TF_DS_ROCI_COMBINE_FIRST: the MoE's partials' part-1 send was queued inside moe (behind the rows it sends); the gather that follows
    /// takes it as sent, once.
    own1_sent: bool = false,
    // TF_DS_CAT_PARTNER in effect: the column partner's part of wo_b's input, bf16 [cap, uw[1 - p]] (TF_DS_CAT_CODES:
    // its codes and scales, prompt2d.catCodesBytes, fewer bytes); 0 otherwise
    cat_stage: u64 = 0,
    // TF_DS_CAT_CODES in effect: this node's part of wo_b's input as codes and scales (prompt2d.catCodesBytes for cap
    // rows of uw[p]), and wo_b's input assembled from both pairs' (codes [cap, uw0 + uw1], scales [cap, (uw0 + uw1) /
    // 32] fp32); 0 otherwise
    cat_codes: u64 = 0,
    wob_q: u64 = 0,
    wob_s: u64 = 0,
    // TF_DS_PACK_CODES in effect: this node's expert pack as codes and scales, and its column partner's
    // (prompt2d.PackCodes for cap * slots slot rows each); 0 otherwise
    pack_codes: u64 = 0,
    peer_codes: u64 = 0,
    // TF_DS_ROCI_GATHERS in effect: x of the layers whose x is gathered lives in the transport's arena (x_g) and every other
    // layer's x in the device buffer (x_dev); `x` is the one the current layer reads and writes. 0 otherwise.
    x_dev: u64 = 0,
    x_g: u64 = 0,
    xs: exl3_experts.Scratch,
    // Model._moe_scratch's decode scratch: windows under 64 rows (a prompt's small chunk, every decode round), one for
    // every layer of the same slots and experts, its epoch carried from call to call
    xsd: exl3_experts.DecodeScratch,
    gl: u64, // fp32 [16, D / 256, experts]: rowmm_gate's chunk sums (chunks of up to 16 rows)
    cidxp: u64 = 0, // int64 [cap, index_topk + 16]: the selection padded with -1 to whole 16-column tiles
    // Engram (zero when no layer has it)
    eb: u64 = 0, // bf16 [cap, cols * engram_head_dim]: the rows read
    ek: u64 = 0, // fp32 [cap, hc * D + D]: their projection, this rank's columns
    ekg: u64 = 0, // fp32 [world, cap, hc * D + D]: its gather
    kv: u64 = 0, // bf16 [cap, hc * D + D]: the ranks' sum
    ws: exl3_prefill.Workspace,
    blas_ws: u64, // cuBLAS workspace
    // the grouped EXL3 linears of decode-sized rows (1..128): rotated rows, split-K partials, each layer's counters
    gxh: u64 = 0, // fp16 [128, the largest group's sum of K]
    gz: u64 = 0, // fp32 [the largest group's split-K partials at 128 rows]
    counters: [640]Counter = undefined,
    n_counters: usize = 0,
    ktop: u64 = 0, // int64 [cap, index_topk]: torch's top-k values for _topk_finish
    // the DSpark taps and the head
    taps: u64 = 0, // bf16 [taps, dec_rows, D]: each tap layer's mean of the streams (Model.forward's taps)
    dec_rows: usize = 0, // the rows the decoder half reads at most (decoderRows): the taps', the pool's and its picks
    head_x: u64 = 0, // bf16 [1, D]: collapse_norm of the last row
    head_l: u64 = 0, // fp32 [1, this rank's vocabulary columns]: its logits
    head_g: u64 = 0, // fp32 [world, 1, columns]: their gather (the prompt's logits, rank after rank)
    pending: bool = false, // gm holds a MoE gather whose post is not in h yet
    // Format.native (mx.zig): the linears' quantized rows and split partials, and the experts' scratch (the
    // intermediate's rows: xs.xd for a prompt chunk's, xsd.xd for a decode window's)
    mxs: mx.Scratch = undefined,
    mxe: mx.ExpertScratch = undefined,

    /// The view of a pass of n rows (viewRows: its own rows when whole, the resume rows under them), within the rows
    /// every buffer holds.
    pub fn view(ch: *Chunk, n: usize, resume_rows: usize) !void {
        const v = viewRows(n, resume_rows);
        if (v > ch.rows_max) return error.ChunkTooLong;
        ch.cap = v;
    }

    /// Every buffer for `cap` rows from `a`, the compressed layers' for chunks ending by position `tokens`; the scratch
    /// the served build zeros (or fills with -1) is set the same.
    pub fn init(e: *const Engine, a: *Arena, cap: usize, tokens: usize) !Chunk {
        const c = e.c;
        const d = c.hidden;
        const hc = c.hc;
        const l0 = e.w.layers[0];
        const hl = e.heads();
        const hd = c.head_dim;
        const groups = l0.groups;
        const o_lora = l0.wo_a[0].n;
        const ex = l0.experts;
        const sl = e.slots();
        var ch: Chunk = undefined;
        ch.spans = &.{};
        ch.img_rows = 0;
        ch.cap = cap;
        ch.rows_max = cap;
        ch.dec_rows = decoderRows(cap, c.window);
        ch.n = 0;
        ch.start = 0;
        ch.pending = false;
        ch.ids = try a.take(cap * 8);
        ch.pos0 = try a.take(cap * 8);
        ch.pos1 = try a.take(cap * 8);
        ch.pos = ch.pos0;
        ch.h = try a.take(cap * hc * d * 2);
        ch.h_alt = try a.take(cap * hc * d * 2);
        ch.x = try a.take(cap * d * 2);
        ch.x_dev = ch.x;
        // TF_DS_ROCI_GATHERS: the gathers' tensors are the arena's regions (the rows each rank produces in place, the others'
        // landed by the transport), not buffers of this arena
        const ga_rc: ?*roci_engine.Roci = if (e.two) |t| t.rc else null;
        ch.x_g = if (ga_rc) |r| r.addr(.g_x) else 0;
        // TF_DS_ROCI_DEC: the taps and the head's row are the arena's regions (each rank writes its own rows, the transport lands the others)
        const dec_rc: ?*roci_engine.Roci = if (e.two) |t| t.rc else null;
        if (dec_rc != null and ch.dec_rows > roci.hand_rows) return error.ChunkTooLong;
        ch.part = try a.take(cap * tri_basic.hc_blocks * 32 * 4);
        ch.pre = try a.take(cap * hc * 4);
        ch.pre_a = try a.take(cap * hc * 4);
        ch.pre_f = try a.take(cap * hc * 4);
        ch.post = try a.take(cap * hc * 4);
        ch.comb = try a.take(cap * hc * hc * 4);
        ch.qa = try a.take(cap * l0.wq_a.n * 2);
        ch.y = if (ga_rc) |r| r.addr(.g_y) else try a.take(cap * hd * 2);
        ch.qr = if (ga_rc) |r| r.addr(.g_qr) else try a.take(cap * l0.wq_a.n * 2);
        ch.q = try a.take(cap * hl * hd * 2);
        ch.wsrc = try a.take((c.window - 1 + cap) * hd * 2);
        ch.wlo = try a.take(8);
        ch.o = try a.take(cap * hl * hd * 2);
        ch.u = try a.take(cap * groups * o_lora * 2);
        // the tensors that cross the wire live in the transport's arena, not in this one
        const rc: ?*roci_engine.Roci = if (e.two) |t| t.rc else null;
        ch.pa = if (rc) |r| r.addr(.psum) else try a.take(cap * d * 4);
        ch.ga = try a.take(e.world * cap * d * 4);
        ch.neg = try a.take(cap * 8);
        ch.ring_idx = try a.take((c.window + ring_extra) * 8);
        ch.ridx_g = try a.take((c.window + ring_extra) * 8);
        ch.ridx_s = try a.take((c.window + ring_extra) * 8);
        ch.win_key = .{ std.math.maxInt(usize), 0, 0, 0 };
        ch.win_have = 0;
        ch.rslot = try a.take(ring_extra * 8);
        ch.pm_ = try a.take(ring_extra * hl * tri_attn.attn_splits * 4);
        ch.pl_ = try a.take(ring_extra * hl * tri_attn.attn_splits * 4);
        ch.po_ = try a.take(ring_extra * hl * tri_attn.attn_splits * hd * 4);
        ch.invalid = try a.take(4);
        ch.xf = try a.take(cap * d * 4);
        ch.gate_f = try a.take(c.experts * d * 4);
        ch.logits = try a.take(cap * c.experts * 4);
        ch.pick = if (ga_rc) |r| r.addr(.g_pick) else try a.take(cap * sl * 4);
        ch.wts = if (ga_rc) |r| r.addr(.g_wts) else try a.take(cap * sl * 4);
        ch.pick_vl = try a.take(cap * sl * 4);
        ch.wts_vl = try a.take(cap * sl * 4);
        ch.pm = if (rc) |r| r.addr(.psum) else try a.take(cap * d * 4); // (attention's and the MoE's partials never coexist: one buffer)
        ch.gm = try a.take(e.world * cap * d * 4);
        // WP4 quarter-HC scratch: only when a Two exists and cap divides evenly by the physical world (today always
        // 4): sized for the owned-row-slice repack, world * q * d, q = cap / world.
        if (e.two != null and e.comm.world != 0 and cap % e.comm.world == 0) {
            ch.qhc_g = try a.take(e.world * (cap / e.comm.world) * d * 4);
            ch.qhc_part = try a.take((cap / e.comm.world) * tri_basic.hc_blocks * 32 * 4);
        }
        resetGathers(&ch);
        ch.encoder_only = false;
        ch.boundary_done = false;
        ch.own_stage = 0;
        ch.wob_q = 0;
        ch.wob_s = 0;
        if (e.two) |t| {
            const k = t.uw[0] + t.uw[1];
            ch.wob_q = try a.take(cap * k);
            ch.wob_s = try a.take(cap * (k / 32) * 4);
        }
        // compressed layers: the smallest ratio sets the most compressed entries a row can see
        var min_ratio: usize = 0;
        for (e.w.layers) |lay| {
            if (lay.ratio != 0 and (min_ratio == 0 or lay.ratio < min_ratio)) min_ratio = lay.ratio;
        }
        if (min_ratio != 0) {
            const ih = c.index_heads * c.index_head_dim;
            ch.max_comp = tokens / min_ratio;
            ch.key_cap = keyCapFor(ch.max_comp);
            ch.kvc = try a.take(cap * c.head_dim * 4);
            ch.scc = try a.take(cap * c.head_dim * 4);
            ch.lat = try a.take(cap * c.head_dim * 2);
            ch.lat2 = try a.take(cap * c.head_dim * 2);
            ch.ik = try a.take(cap * c.index_head_dim * 2);
            ch.groups = try a.take(cap * 8);
            ch.gpos = try a.take(cap * 8);
            ch.raw_idx = try a.take(raw_rows * 8);
            ch.iq = try a.take(cap * ih * 2);
            ch.iq4 = try a.take(cap * ih * 2);
            ch.wl = try a.take(cap * c.index_heads * 4);
            ch.iw = try a.take(cap * c.index_heads * 2);
            ch.vis = try a.take(cap * 8);
            // the indexer's row blocks keep rows * n_comp under 2^24 (model.py rb; a context past 1,048,576 keeps the 16-row floor's rows, keyCapFor), so their keys under key_cap
            const ke: usize = @min(cap * ch.max_comp, ch.key_cap);
            ch.keys = try a.take(ke * 8);
            ch.score = try a.take(ke * 4);
            ch.tmax = try a.take((ke / tri_index.tile + cap) * 8);
            ch.cidx = if (ga_rc) |r| r.addr(.g_cidx) else try a.take(cap * c.index_topk * 8);
            ch.ktop = try a.take(cap * c.index_topk * 8);
            ch.cidxp = try a.take(cap * (c.index_topk + 16) * 8);
            // the pool: only past candidate_blocks blocks, so a row block holds at most key_elems / 2048 rows there
            const cb = c.candidate_block;
            ch.pool_ms = (ch.max_comp + cb - 1) / cb;
            if (ch.pool_ms > c.candidate_blocks) {
                ch.pool = try a.take(ch.dec_rows * ch.pool_ms);
                ch.bmax = try a.take(@max(16 * ch.pool_ms, ch.key_cap / cb + cap) * 4);
                ch.pidx = try a.take(ch.dec_rows * c.candidate_blocks * 8);
            }
            ch.every = try a.take(cap * 8);
            {
                const ev = try std.heap.page_allocator.alloc(i64, cap);
                defer std.heap.page_allocator.free(ev);
                @memset(ev, tri_index.every_vis);
                try upload(e, ch.every, ev.ptr, cap * 8);
                try e.s.synchronize(); // (the host copy is freed next)
            }
            if (tri_index.prunes(ch.max_comp, c.index_topk)) {
                ch.pbufs = .{ .tpos = try a.take(pruned_rows * c.index_topk * 8), .cand = try a.take(pruned_rows * c.index_topk * tri_index.tile * 8), .every = ch.every };
            }
        }
        for (e.w.layers) |lay| {
            const ew = lay.engram_wkv orelse continue;
            const en = if (e.two) |t| t.engramWidth() else ew.n; // the summed projection's width (2D: both pairs')
            ch.eb = try a.take(cap * ew.k * 2);
            ch.ek = if (rc) |r| r.addr(.ek) else try a.take(cap * ew.n * 4);
            ch.ekg = try a.take(ekgBytes(cap, e.world, en, if (e.two) |t| .{ .ew = t.ew, .p = t.p, .world = e.comm.world } else null));
            ch.kv = try a.take(cap * en * 2);
            break;
        }
        const sz = exl3_experts.Scratch.sizes(cap, sl, ex.dims, ex.width + if (ex.rest) |r| r.width else 0, ex.count); // parity: the pack
        var xs: exl3_experts.Scratch = undefined;
        xs.rows = cap;
        xs.slots = sl;
        if (e.mx != null) {
            // Format.native: only the intermediate's rows (bf16 [cap * slots, pack]), xs.xd for chunks and xsd.xd for
            // decode windows; the grouping and the quantized rows live in ch.mxe
            const pack = ex.width + if (ex.rest) |r| r.width else 0;
            xs = std.mem.zeroes(exl3_experts.Scratch);
            xs.rows = cap;
            xs.slots = sl;
            xs.xd = try a.take(cap * sl * pack * 2);
            ch.xs = xs;
            ch.xsd = std.mem.zeroes(exl3_experts.DecodeScratch);
            ch.xsd.rows = exl3_experts.exact_rows;
            ch.xsd.slots = sl;
            ch.xsd.xd = xs.xd;
            try mxScratch(e, &ch, a, cap);
        } else {
            inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
                @field(xs, f) = try a.take(sz[j]);
            }
            ch.xs = xs;
            ch.xsd = try decodeScratch(e, a, ex, sl);
            if (e.two) |t| try t.clearRest(e); // parity: the rest launches' grouping starts empty
        }
        ch.gl = try a.take(16 * (d / 256) * c.experts * 4);
        // prefill.Workspace: the largest prompt GEMM's rotated input and W_q (fp16) over every layer, and the Hadamard;
        // none on Format.native (mmRows takes mx.linear: no EXL3 GEMM runs)
        var max_xk: usize = 0;
        var max_kn: usize = 0;
        if (e.mx == null) for (e.w.layers) |lay| {
            var ls: [16]?weights.Linear = @splat(null);
            ls[0] = lay.wq_a;
            ls[1] = lay.wkv;
            ls[2] = lay.wq_b;
            ls[3] = lay.wo_b;
            for (lay.wo_a[0..lay.groups], 0..) |wo, g| ls[4 + g] = wo;
            ls[8] = lay.comp_wkv;
            ls[9] = lay.comp_wgate;
            ls[10] = lay.idx_wq_b;
            ls[11] = lay.idx_wk;
            ls[12] = lay.engram_wkv;
            for (ls) |ol| {
                const l = ol orelse continue;
                max_xk = @max(max_xk, l.k);
                max_kn = @max(max_kn, @as(usize, l.k) * l.n);
            }
        };
        ch.ws = if (e.mx == null) .{ .xh = try a.take(cap * max_xk * 2), .w = try a.take(max_kn * 2), .h = try a.take(128 * 128 * 2) } else .{ .xh = 0, .w = 0, .h = 0 };
        ch.blas_ws = try a.take(cublas.Blas.workspace_bytes);
        // grouped linears: every layer's counters (int32 [8 N / 128], zeros, left zero by each launch), the largest
        // group's rotated rows and partials at 128 rows
        var max_gk: usize = 0;
        var max_gz: usize = 0;
        ch.n_counters = 0;
        for (e.w.layers) |lay| try layerLinears(e, &ch, a, lay, &max_gk, &max_gz);
        {
            // the head: one row through its own group
            const l = e.w.head;
            try addLinear(e, &ch, a, l, &max_gk, &max_gz);
            ch.head_x = if (dec_rc) |r| r.addr(.head_x) else try a.take(d * 2);
            ch.head_l = try a.take(l.n * 4);
            const hn = if (e.two) |t| t.hw[0] + t.hw[1] else l.n; // a rank's vocabulary half (2D: both pairs' parts)
            ch.head_g = try a.take(e.world * hn * 4);
        }
        if (e.w.dspark) |ds| {
            // the drafter (draft.zig): its stages' linears as a layer's, main_proj, and the stages' wkv as one group
            // (Drafter.stage_kv)
            var kv_k: usize = 0;
            var kv: [8]weights.Linear = undefined;
            if (ds.blocks.len > kv.len) return error.TooManyStages;
            for (ds.blocks, 0..) |lay, j| {
                try layerLinears(e, &ch, a, lay, &max_gk, &max_gz);
                kv[j] = lay.wkv;
                kv_k += lay.wkv.k;
            }
            max_gk = @max(max_gk, kv_k);
            max_gz = @max(max_gz, exl3_linear.zFloats(kv[0..ds.blocks.len], 128));
            try addLinear(e, &ch, a, ds.main_proj, &max_gk, &max_gz);
        }
        ch.taps = if (dec_rc) |r| r.addr(.g_taps) else try a.take(@max(c.dspark_taps.slice().len, 1) * ch.dec_rows * d * 2);
        ch.gxh = try a.take(128 * max_gk * 2);
        ch.gz = try a.take(@max(max_gz, 1) * 4);
        // the scratch as experts.py makes it: zeros, the member lists -1
        if (e.mx == null) {
            for ([_]u64{ xs.xg, xs.xu, xs.xd, xs.z, xs.no_y, xs.ids, xs.count, xs.counts }, [_]usize{ sz[0], sz[1], sz[2], sz[3], sz[4], sz[5], sz[6], sz[7] }) |p, n| try fill(e, p, 0, n);
            try fill32(e, xs.members, 0xffffffff, sz[8] / 4);
        }
        try fill(e, ch.neg, 0xff, cap * 8);
        if (e.mx == null) {
            var had: [128 * 128]u16 = undefined;
            exl3_prefill.hadamard(&had);
            try upload(e, ch.ws.h, &had, had.len * 2);
        }
        return ch;
    }
};

/// A pass's view of its chunk: the cap its owned-row paths and plane strides read (a whole pass of n rows: n, n == cap;
/// a pass under the resume rows: the resume rows, so n < cap and the unsharded path runs, as at the grid's own chunks).
pub fn viewRows(n: usize, resume_rows: usize) usize {
    return @max(n, resume_rows);
}

/// The rows a chunk's decoder half reads at most (the DSpark taps, the candidate pool and its picks): the window
/// (replayCut keeps the rows from the prompt's length less the window), or the chunk's own rows when fewer.
pub fn decoderRows(cap: usize, window: usize) usize {
    return @min(cap, window);
}

/// The most rows a prompt pass runs (TF_DS_PASS_ROWS=16384: eight resume units). Declared here so this module's host row tables
/// size without importing model (which imports this); model asserts they agree. What bounds it: the host row tables on the stack
/// (three of this many i64 in a layer), the Engram worker's ring (twice the units, engram_rows.max_wk), and the memory a pass takes
/// (the start check names the shortfall); a larger pass raises this one constant and the replay tables.
pub const max_pass_rows = 16384;

/// A host row table for a pass's `n` rows over stack storage for the engine's largest pass (`max_pass_rows`): the
/// chunk's view and the storage both cap it. The table is filled to n only; the storage is the ceiling, not the count.
pub fn hostRows(ch: *const Chunk, store: []i64, n: usize) ![]i64 {
    if (n > ch.cap or n > store.len) return error.ChunkTooLong;
    return store[0..n];
}

/// A 2D split's Engram widths for ekgBytes: this node's pair's and the other's columns, its pair, the nodes.
pub const EkgSplit = struct { ew: [2]usize, p: usize, world: usize };

/// Engram's gather buffer (Chunk.ekg) for chunks of `cap` rows: on a 2D split the largest of engramSum's n (6 ew[p]
/// + 2 ew[1 - p]) bytes, the quarters' [2, n, ew0 + ew1] fp32 under 64 rows (add2Bf16 reads them after) and the owner
/// sums' stage at the owned rows; else the `tp` ranks' projections [tp, cap, en] fp32 (the all-gather).
pub fn ekgBytes(cap: usize, tp: usize, en: usize, two: ?EkgSplit) usize {
    const t = two orelse return tp * cap * en * 4;
    const small: usize = @min(cap, 63); // (typed: @min with a comptime bound narrows)
    var b = @max(cap * (6 * t.ew[t.p] + 2 * t.ew[1 - t.p]), 2 * small * en * 4);
    if (t.world != 0 and cap % t.world == 0) b = @max(b, prompt2d.OwnStage.bytes(cap / t.world, t.ew[t.p]));
    return b;
}

/// Format.native's scratch for chunks of `cap` rows (and rounds and drafter passes, fewer): the widest K any native
/// linear reads (wo_a's groups together), the largest split partials, and the experts' grouping for the routed rows.
fn mxScratch(e: *const Engine, ch: *Chunk, a: *Arena, cap: usize) !void {
    var max_k: usize = 0;
    var zf: usize = 0;
    var e_max: usize = 0;
    var dn_k: usize = 0;
    var dn_n: usize = 0;
    const blocks: []const weights.Layer = if (e.w.dspark) |ds| ds.blocks else &.{};
    for ([_][]const weights.Layer{ e.w.layers, blocks }) |ls| for (ls) |lay| {
        var lin: [16]?weights.Linear = @splat(null);
        lin[0] = lay.wq_a;
        lin[1] = lay.wkv;
        lin[2] = lay.wq_b;
        lin[3] = lay.wo_b;
        lin[4] = lay.comp_wkv;
        lin[5] = lay.comp_wgate;
        lin[6] = lay.idx_wq_b;
        lin[7] = lay.idx_wk;
        lin[8] = lay.engram_wkv;
        for (lin) |ol| {
            const l = ol orelse continue;
            max_k = @max(max_k, l.k);
            zf = @max(zf, mx.zFloats(l, cap));
        }
        var wk: usize = 0;
        for (lay.wo_a[0..lay.groups]) |wo| wk += wo.k;
        max_k = @max(max_k, wk);
        // the 2D split's wo_b reads the rank's whole wo_a output
        if (e.two) |t| max_k = @max(max_k, t.uw[0] + t.uw[1]);
        e_max = @max(e_max, lay.experts.count);
        dn_k = @max(dn_k, lay.experts.down_k);
        dn_n = @max(dn_n, lay.experts.down_n);
        max_k = @max(max_k, lay.experts.down_k);
    };
    if (e.w.dspark) |ds| {
        max_k = @max(max_k, ds.main_proj.k);
        zf = @max(zf, mx.zFloats(ds.main_proj, cap));
    }
    const ls = mx.Scratch.sizes(cap, max_k, zf);
    ch.mxs = .{ .q = try a.take(ls[0]), .s = try a.take(ls[1]), .z = try a.take(ls[2]), .fake = try a.take(ls[3]), .rows = cap, .max_k = max_k };
    const sl = e.slots();
    const es = mx.ExpertScratch.sizes(cap, sl, e_max, e.c.hidden, dn_k, dn_n);
    var x: mx.ExpertScratch = undefined;
    x.rows = cap;
    x.slots = sl;
    // TF_DS_ROCI_GATHERS: the routing's codes and scales are the arena's regions, which the transport gathers
    const ga_rc: ?*roci_engine.Roci = if (e.two) |t| t.rc else null;
    inline for (.{ "q", "s", "cnt", "off", "fill", "act", "nact", "ent", "erow", "hq", "hs", "out" }, 0..) |f, j| {
        @field(x, f) = if (ga_rc) |r| (if (j == 0) r.addr(.g_q) else if (j == 1) r.addr(.g_s) else try a.take(es[j])) else try a.take(es[j]);
    }
    // the prefill tiles' list (mx.gateUp / down of a prompt chunk)
    x.tiles = try a.take(mx.ExpertScratch.tileBytes(cap, sl, e_max));
    x.ntiles = try a.take(4);
    x.z = .{ .ptr = 0, .floats = 0 };
    // decode launches' split partials (round.zig and the drafter's own: mx.decode_rows rows at most)
    var zf_dec: usize = 0;
    for ([_][]const weights.Layer{ e.w.layers, blocks }) |dls| for (dls) |lay| {
        zf_dec = @max(zf_dec, mx.decodeZFloats(lay, e.c.hidden, e.c.experts, e.c.index_heads, mx.decode_rows, sl));
    };
    if (e.w.head.fmt == .bf16) zf_dec = @max(zf_dec, mx.bfSplits(e.w.head.k, e.w.head.n) * mx.decode_rows * e.w.head.n);
    if (zf_dec > 0) x.z = .{ .ptr = try a.take(zf_dec * 4), .floats = zf_dec };
    ch.mxe = x;
}

/// A layer's linears in the chunk's grouped-linear sizes (its attention input group, its wo_a group, each alone at 128
/// rows) and each one's split-K counters (Chunk.init).
fn layerLinears(e: *const Engine, ch: *Chunk, a: *Arena, lay: weights.Layer, max_gk: *usize, max_gz: *usize) !void {
    const in_group = [_]weights.Linear{ lay.wq_a, lay.wkv };
    max_gk.* = @max(max_gk.*, lay.wq_a.k + lay.wkv.k);
    max_gz.* = @max(max_gz.*, exl3_linear.zFloats(&in_group, 128));
    var wo_k: usize = 0;
    for (lay.wo_a[0..lay.groups]) |wo| wo_k += wo.k;
    max_gk.* = @max(max_gk.*, wo_k);
    max_gz.* = @max(max_gz.*, exl3_linear.zFloats(lay.wo_a[0..lay.groups], 128));
    var ls: [16]?weights.Linear = @splat(null);
    ls[0] = lay.wq_a;
    ls[1] = lay.wkv;
    ls[2] = lay.wq_b;
    ls[3] = lay.wo_b;
    for (lay.wo_a[0..lay.groups], 0..) |wo, g| ls[4 + g] = wo;
    ls[8] = lay.comp_wkv;
    ls[9] = lay.comp_wgate;
    ls[10] = lay.idx_wq_b;
    ls[11] = lay.idx_wk;
    ls[12] = lay.engram_wkv;
    for (ls) |ol| {
        const l = ol orelse continue;
        try addLinear(e, ch, a, l, max_gk, max_gz);
    }
}

/// One linear alone at 128 rows in the chunk's grouped-linear sizes, and its split-K counters (int32 [8 N / 128], zeros,
/// left zero by each launch).
fn addLinear(e: *const Engine, ch: *Chunk, a: *Arena, l: weights.Linear, max_gk: *usize, max_gz: *usize) !void {
    max_gk.* = @max(max_gk.*, l.k);
    max_gz.* = @max(max_gz.*, exl3_linear.zFloats(&[_]weights.Linear{l}, 128));
    if (ch.n_counters == ch.counters.len) return error.TooManyLinears;
    const bytes = 8 * (l.n / 128) * 4;
    const ptr = try a.take(bytes);
    try fill(e, ptr, 0, bytes);
    ch.counters[ch.n_counters] = .{ .words = l.words, .ptr = ptr };
    ch.n_counters += 1;
}

/// Model._moe_scratch's decode scratch: experts.py Scratch(rows=max(n, 64), prompt=False), zeros but its member lists'
/// -1 (made at the first window under 64 rows and never replaced).
pub fn decodeScratch(e: *const Engine, a: *Arena, ex: weights.Experts, slots: usize) !exl3_experts.DecodeScratch {
    const rows = exl3_experts.exact_rows;
    const sz = if (e.two != null) try exl3_experts2d.decodeSizes(rows, slots, ex) else try exl3_experts.DecodeScratch.sizes(rows, slots, ex.dims, ex.width, ex.count); // 2D: this node's widths
    var sc: exl3_experts.DecodeScratch = undefined;
    sc.rows = rows;
    sc.slots = slots;
    inline for (.{ "xg", "xu", "xd", "z", "y", "cnt_gu", "cnt_d", "epoch", "ready", "ready_cnt", "ids", "count", "members" }, 0..) |f, j| {
        const ptr = try a.take(sz[j]);
        @field(sc, f) = ptr;
        if (j == 12) try fill32(e, ptr, 0xffffffff, sz[j] / 4) else try fill(e, ptr, 0, sz[j]);
    }
    return sc;
}

/// Host bytes to the device in the stream's order (the stream is non-blocking: the legacy stream's synchronous copies
/// would not wait for its kernels). Pageable host memory is staged before the call returns.
pub fn upload(e: *const Engine, dst: u64, src: *const anyopaque, bytes: usize) !void {
    try e.d.copyHtoDAsync(@src(), dst, src, bytes, e.s.handle);
}

/// cuMemsetD8 in the stream's order.
pub fn fill(e: *const Engine, dst: u64, value: u8, bytes: usize) !void {
    try e.d.check(e.d.api.cuMemsetD8Async(dst, value, bytes, e.s.handle), "cuMemsetD8Async");
}

/// cuMemsetD32 in the stream's order.
pub fn fill32(e: *const Engine, dst: u64, value: u32, words: usize) !void {
    try e.d.check(e.d.api.cuMemsetD32Async(dst, value, words, e.s.handle), "cuMemsetD32Async");
}

/// A layer's split-K counters (Exl3Linear.counters) by its trellis words.
pub const Counter = struct { words: u64, ptr: u64 };

pub fn counterOf(ch: *const Chunk, l: weights.Linear) !u64 {
    for (ch.counters[0..ch.n_counters]) |c| if (c.words == l.words) return c.ptr;
    return error.NoCounters;
}

/// Exl3Group(ls) on m (1..128) rows (linear.py): one rot_many launch rotating every layer's input rows xs[i] (bf16,
/// row stride ldxs[i]) into the group's buffer, then a glinear launch for each (bits, warps) set, outputs into outs[i]
/// (row stride ldos[i], dtype types[i]); programmatic dependent launches, split-K partials dropped from L2 (the served
/// TF_EXL3_PDL and TF_EXL3_L2_DISCARD).
pub fn grouped(e: *const Engine, ch: *const Chunk, m: usize, ls: []const weights.Linear, xs: []const u64, ldxs: []const usize, outs: []const u64, ldos: []const usize, types: []const exl3_linear.DType) !void {
    if (m == 0 or m > decode_rows or ls.len > 8) return error.NotADecodeWindow;
    if (e.mx) |k| {
        // Format.native: each linear on its own (its rows quantized, the MX or bf16 GEMM): the bits of any row count
        for (ls, 0..) |l, i| try mx.linear(k, e.s, ch.mxs, l, xs[i], .bf16, ldxs[i], m, outs[i], ldos[i], types[i], false, null);
        return;
    }
    var calls: [8]exl3_linear.Call = undefined;
    var off: usize = 0;
    for (ls, 0..) |l, i| {
        calls[i] = .{ .layer = l, .x = xs[i], .ldx = @intCast(ldxs[i]), .x_dtype = .bf16, .xh = ch.gxh + off * 2, .y = outs[i], .ldy = @intCast(ldos[i]), .y_dtype = types[i], .counters = try counterOf(ch, l) };
        off += m * l.k;
    }
    try exl3_linear.rotMany(e.lin, e.s, calls[0..ls.len], m, true);
    try exl3_linear.glinear(e.lin, e.s, calls[0..ls.len], m, ch.gz, true, true);
}

/// Exl3Group.rotated on m rows already rotated into each call's xh (a kernel that folds the rotation in wrote them):
/// the glinear launches alone, each call's counters and folds as given.
/// grouped for a decode launch (a round's rows, the drafter's): Format.native's FP8 / FP4 linears through the GEMV
/// (mx.linear decode), each row its own bits at any row count; EXL3 as grouped.
pub fn groupedDecode(e: *const Engine, ch: *const Chunk, m: usize, ls: []const weights.Linear, xs: []const u64, ldxs: []const usize, outs: []const u64, ldos: []const usize, types: []const exl3_linear.DType) !void {
    const k = e.mx orelse return grouped(e, ch, m, ls, xs, ldxs, outs, ldos, types);
    // past mx.decode_rows rows (the drafter's pass of 13+ streams) linearsDecode runs slabs of the same kernels
    if (m == 0 or m > decode_rows or ls.len > 8) return error.NotADecodeWindow;
    try mx.linearsDecode(k, e.s, ch.mxs, ls, xs, ldxs, m, outs, ldos, types, ch.mxe.z);
}

pub fn groupedRotated(e: *const Engine, ch: *const Chunk, m: usize, calls: []exl3_linear.Call) !void {
    if (m == 0 or m > decode_rows or calls.len > exl3_linear.gmax) return error.NotADecodeWindow;
    if (e.mx != null) return error.RotatedRowsOnNative; // Format.native: the callers pass plain rows (grouped)
    for (calls) |*c| c.counters = try counterOf(ch, c.layer);
    try exl3_linear.glinear(e.lin, e.s, calls, m, ch.gz, true, true);
}

/// The pool's compressed-attention caches (model.py PoolCache: comp, index_k, comp_raw), by kv-source layer: the
/// compressed latents as packed FP4 (codes [rows, head_dim / 2], E8M0 scales [rows, head_dim / 16]), the index keys
/// (codes [rows, index_head_dim / 2], scales [rows, index_head_dim / 32], 127 when unwritten) over `cap` positions of
/// streams' extents (rows = cap / ratio + 2), and, at ratio 2, each slot's positional store of the compressor's raw kv
/// and scores (fp32 [RAW, head_dim] each, `slots` of them). One stream's caches (SeqCache) are a view.
pub const Caches = struct {
    comp_codes: [64]u64 = @splat(0),
    comp_scales: [64]u64 = @splat(0),
    idx_codes: [64]u64 = @splat(0),
    idx_scales: [64]u64 = @splat(0),
    raw_kv: [64]u64 = @splat(0),
    raw_score: [64]u64 = @splat(0),
    cap: usize = 0, // positions
    slots: usize = 1, // the positional stores' slots

    /// The device bytes of one layer's caches for a pool of `cap` positions and `slots` streams (zero where the layer keeps none):
    /// the sizes init takes.
    pub const LayerBytes = struct { comp_codes: usize = 0, comp_scales: usize = 0, idx_codes: usize = 0, idx_scales: usize = 0, raw_kv: usize = 0, raw_score: usize = 0 };

    pub fn layerBytes(c: *const Config, lay: weights.Layer, cap: usize, slots: usize) LayerBytes {
        var b: LayerBytes = .{};
        if (lay.comp_wkv == null) return b;
        const r: usize = lay.ratio;
        const rows = cap / r + 2;
        b.comp_codes = rows * c.head_dim / 2;
        b.comp_scales = rows * c.head_dim / 16;
        if (lay.idx_wk != null) {
            b.idx_codes = rows * c.index_head_dim / 2;
            b.idx_scales = rows * c.index_head_dim / 32;
        }
        if (r > 1) {
            b.raw_kv = slots * raw_rows * c.head_dim * 4;
            b.raw_score = slots * raw_rows * c.head_dim * 4;
        }
        return b;
    }

    /// The device bytes of the whole pool's caches (what init takes, before the arena's 256-byte alignment): the KV pool a Spark holds.
    pub fn poolBytes(c: *const Config, layers: []const weights.Layer, cap: usize, slots: usize) usize {
        var total: usize = 0;
        for (layers) |lay| {
            const b = layerBytes(c, lay, cap, slots);
            total += b.comp_codes + b.comp_scales + b.idx_codes + b.idx_scales + b.raw_kv + b.raw_score;
        }
        return total;
    }

    pub fn init(e: *const Engine, a: *Arena, cap: usize, slots: usize) !Caches {
        const Src = struct {
            a: *Arena,
            fn take(s: @This(), bytes: usize, _: usize) !u64 {
                return s.a.take(bytes);
            }
        };
        return initFrom(e, Src{ .a = a }, cap, cap, slots);
    }

    /// The pool in virtual memory (TF_DS_KV_VMM=1, kv_vmm.zig): each plane takes a stretch of `vp`'s reservation sized for
    /// `cap_max` positions and has its first `cap` positions' bytes backed. Same planes, same order, same bytes as `init`.
    pub fn initVmm(e: *const Engine, vp: anytype, cap: usize, cap_max: usize, slots: usize) !Caches {
        const Src = struct {
            vp: @TypeOf(vp),
            fn take(s: @This(), bytes: usize, max: usize) !u64 {
                return s.vp.addPlane(max, bytes);
            }
        };
        return initFrom(e, Src{ .vp = vp }, cap, cap_max, slots);
    }

    /// The address space a pool of up to `cap_max` positions takes in a reservation of granularity `gran`: each plane's stretch.
    pub fn vaBytes(c: *const Config, layers: []const weights.Layer, cap_max: usize, slots: usize, gran: usize) usize {
        var total: usize = 0;
        for (layers) |lay| {
            const b = layerBytes(c, lay, cap_max, slots);
            inline for (.{ b.comp_codes, b.comp_scales, b.idx_codes, b.idx_scales, b.raw_kv, b.raw_score }) |n| {
                if (n != 0) total += std.mem.alignForward(usize, n, gran);
            }
        }
        return total;
    }

    pub const PlaneKind = enum { comp_codes, comp_scales, idx_codes, idx_scales, raw_kv, raw_score };
    pub const Plane = struct { kind: PlaneKind, bytes: usize };
    /// The most planes a pool has: six a layer.
    pub const max_planes = 64 * 6;

    /// Each plane's bytes for a pool of `cap` positions, in the order init takes them (the VMM pool's plane numbers); how many.
    pub fn planeSizes(c: *const Config, layers: []const weights.Layer, cap: usize, slots: usize, out: *[max_planes]Plane) usize {
        var n: usize = 0;
        for (layers) |lay| {
            const b = layerBytes(c, lay, cap, slots);
            if (b.comp_codes == 0) continue;
            inline for (.{ .{ PlaneKind.comp_codes, b.comp_codes }, .{ PlaneKind.comp_scales, b.comp_scales }, .{ PlaneKind.idx_codes, b.idx_codes }, .{ PlaneKind.idx_scales, b.idx_scales }, .{ PlaneKind.raw_kv, b.raw_kv }, .{ PlaneKind.raw_score, b.raw_score } }) |pl| {
                if (pl[1] != 0) {
                    out[n] = .{ .kind = pl[0], .bytes = pl[1] };
                    n += 1;
                }
            }
        }
        return n;
    }

    /// The device bytes a VMM pool of `cap` positions has mapped: each plane's bytes rounded up to the granularity.
    pub fn vmmBytes(c: *const Config, layers: []const weights.Layer, cap: usize, slots: usize, gran: usize) usize {
        var pl: [max_planes]Plane = undefined;
        const n = planeSizes(c, layers, cap, slots, &pl);
        var total: usize = 0;
        for (pl[0..n]) |x| total += std.mem.alignForward(usize, x.bytes, gran);
        return total;
    }

    /// The VMM pool moves to `new_cap` positions: every plane grows or shrinks at its tail, its address unchanged. A grow clears the
    /// rows it adds (zeros, the index keys' scales 127: a new pool's contents) and, failing part way, gives back what it took and
    /// leaves the pool as it was. A shrink gives back whole tail segments and keeps the rest mapped (the pool's size is then `new_cap`
    /// all the same). The device must be idle: the caller synchronised.
    pub fn resizeVmm(cs: *Caches, e: *const Engine, vp: anytype, new_cap: usize) !void {
        var old: [max_planes]Plane = undefined;
        var neu: [max_planes]Plane = undefined;
        const n = planeSizes(e.c, e.w.layers, cs.cap, cs.slots, &old);
        if (n != planeSizes(e.c, e.w.layers, new_cap, cs.slots, &neu) or n != vp.planes.items.len) return error.BadPool;
        if (new_cap > cs.cap) {
            var done: usize = 0;
            errdefer for (0..done) |i| {
                _ = vp.shrink(@intCast(i), old[i].bytes) catch 0;
            };
            for (0..n) |i| {
                try vp.grow(@intCast(i), neu[i].bytes);
                done = i + 1;
            }
            for (0..n) |i| if (neu[i].bytes > old[i].bytes) {
                const value: u8 = if (neu[i].kind == .idx_scales) 127 else 0;
                try fill(e, vp.addressOf(@intCast(i)) + old[i].bytes, value, neu[i].bytes - old[i].bytes);
            };
        } else {
            for (0..n) |i| _ = try vp.shrink(@intCast(i), neu[i].bytes);
        }
        cs.cap = new_cap;
    }

    /// `src.take(bytes, max_bytes)` gives one plane: `bytes` now, room for `max_bytes` (the planes at `cap_max` positions).
    fn initFrom(e: *const Engine, src: anytype, cap: usize, cap_max: usize, slots: usize) !Caches {
        const c = e.c;
        var cs: Caches = .{ .cap = cap, .slots = slots };
        for (e.w.layers, 0..) |lay, i| {
            const b = layerBytes(c, lay, cap, slots);
            if (b.comp_codes == 0) continue;
            const m = layerBytes(c, lay, cap_max, slots);
            cs.comp_codes[i] = try src.take(b.comp_codes, m.comp_codes);
            cs.comp_scales[i] = try src.take(b.comp_scales, m.comp_scales);
            if (b.idx_codes != 0) {
                cs.idx_codes[i] = try src.take(b.idx_codes, m.idx_codes);
                cs.idx_scales[i] = try src.take(b.idx_scales, m.idx_scales);
            }
            if (b.raw_kv != 0) {
                cs.raw_kv[i] = try src.take(b.raw_kv, m.raw_kv);
                cs.raw_score[i] = try src.take(b.raw_score, m.raw_score);
            }
        }
        try cs.clear(e);
        return cs;
    }

    /// A new pool's caches, as PoolCache makes them: zeros, the index keys' E8M0 scales 127 (1.0).
    pub fn clear(cs: *const Caches, e: *const Engine) !void {
        const c = e.c;
        for (e.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            const rows = cs.cap / r + 2;
            try fill(e, cs.comp_codes[i], 0, rows * c.head_dim / 2);
            try fill(e, cs.comp_scales[i], 0, rows * c.head_dim / 16);
            if (lay.idx_wk != null) {
                try fill(e, cs.idx_codes[i], 0, rows * c.index_head_dim / 2);
                try fill(e, cs.idx_scales[i], 127, rows * c.index_head_dim / 32);
            }
            if (r > 1) {
                try fill(e, cs.raw_kv[i], 0, cs.slots * raw_rows * c.head_dim * 4);
                try fill(e, cs.raw_score[i], 0, cs.slots * raw_rows * c.head_dim * 4);
            }
        }
    }

    /// The stream in pool slot `slot` with its extent from position `base` (MultiDecoder: pool_view): its compressed
    /// rows from base / ratio, its slot's positional stores (one slot's view: its prompt chunks read positions from 0).
    pub fn view(cs: *const Caches, e: *const Engine, slot: usize, base: usize) !Caches {
        const c = e.c;
        if (slot >= cs.slots or base >= cs.cap) return error.BadView;
        var v = cs.*;
        v.slots = 1;
        v.cap = cs.cap - base;
        for (e.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (base % r != 0) return error.BadView;
            const g = base / r;
            v.comp_codes[i] += g * c.head_dim / 2;
            v.comp_scales[i] += g * c.head_dim / 16;
            if (lay.idx_wk != null) {
                v.idx_codes[i] += g * c.index_head_dim / 2;
                v.idx_scales[i] += g * c.index_head_dim / 32;
            }
            if (r > 1) {
                v.raw_kv[i] += slot * raw_rows * c.head_dim * 4;
                v.raw_score[i] += slot * raw_rows * c.head_dim * 4;
            }
        }
        return v;
    }
};

/// What one forward carries from layer to layer (model.py _forward_k's `shared`): the latest kv-source layer, and the
/// latest indexer's selection (in the chunk's cidx, kk a row).
pub const Shared = struct { kv_layer: ?usize = null, kk: ?usize = null, pool: bool = false };

/// The chunk's rows: token ids at positions start .., their streams h [n, hc, D] (every stream the token's embedding
/// row) and pre [n, hc] = (1, 0, 0, 0): embed_init, the bytes of forward()'s embedding rows expanded over the streams.
pub fn begin(e: *const Engine, ch: *Chunk, ids: []const i64, start: usize, host_pos: []i64) !void {
    const n = ids.len;
    if (n > ch.cap or host_pos.len < n) return error.ChunkTooLong;
    beginHost(ch, n, start);
    for (host_pos[0..n], 0..) |*p, i| p.* = @intCast(start + i);
    try upload(e, ch.ids, ids.ptr, n * 8);
    ch.pos = ch.pos0;
    try upload(e, ch.pos, host_pos.ptr, n * 8);
    try tri_basic.embedInit(e.t, e.w.embed, ch.ids, ch.h, ch.pre, n, e.c.hidden, e.c.hc);
}

/// merge_image_embeddings on the chunk: each span row's embedding (in every one of the hc streams) replaced by its
/// image row (the delimiters' learned rows and the aligner's), before the first layer.
pub fn embedSpans(e: *const Engine, ch: *Chunk) !void {
    const c = e.c;
    const row = c.hidden * 2;
    for (ch.spans) |sp| {
        const o = overlap(sp, ch.start, ch.n) orelse continue;
        const src = ch.img_rows + (sp.row + (ch.start + o[0] - sp.at)) * row;
        for (0..c.hc) |s| try e.ops.copyRows(e.s, src, row, ch.h + (o[0] * c.hc + s) * row, c.hc * row, row, o[1] - o[0]);
    }
}

/// The streams' second buffer becomes the streams (hc_pre_pf wrote the posted ones there) and the old one the spare.
fn swapStreams(ch: *Chunk, src: u64) void {
    std.debug.assert(src == ch.h_alt);
    ch.h_alt = ch.h;
    ch.h = src;
}

/// A row range: `start`, the count `n`. WP4's qhcRange and WP5's row-sharded gather() both key owned-row
/// transport off the identical range, so it is named once here rather than as two matching anonymous structs.
pub const QRows = struct { start: usize, n: usize };

/// Which sublayer's partials a gather assembles: the attention's (into ch.ga) or the MoE's (into ch.gm).
pub const Sub = enum { attn, moe };

/// What a gather left for its readers: `owned` when it filled the tight buffer ch.qhc_g itself for this node's owned
/// rows (plane 0 the TP2 ranks' sums, plane 1 -0.0: the readers' G0 + G1 gives the same bits); else the planes of
/// ga or gm, which a reader at owned rows repacks into the tight buffer first.
pub const Gathered = struct { owned: bool = false, blocks: bool = false }; // blocks: the sums lie in the transport's two bf16 blocks (TF_DS_ROCI)

/// A gather as gatherStep decided it on the host: its path, the owned rows its readers take, and on the owner path
/// whether this call's plane 1 of the tight buffer is still to be written (`neg`).
pub const GatherStep = struct { owned: QRows };

/// WP4 (quarter-HC, docs/prefill-plan.md): this layer's owned row range on this node, when quarter-HC applies --
/// a full-cap 2D chunk, strictly before the replay-cut boundary layer (c.layers / 2, which needs every row's h /
/// pre / x globally and is never sharded: replayCut truncates ch.n there for the decoder's own, unchanged path).
/// node g of `world` owns rows [g * q, (g + 1) * q), q = cap / world. A short chunk, a resumed chunk under cap
/// rows, a partial final pass (a cap off the resume grid), a non-2D engine or the flag off (agreeQuarterHc) all fall
/// through to the baseline unsharded path (null).
/// The owned rows of this rank: its slab of the chunk (slab_plan.slab), empty for a rank beyond the chunk's rows.
fn anySlab(e: *const Engine, ch: *const Chunk) QRows {
    const s = slab_plan.slab(ch.n, e.comm.rank);
    return .{ .start = s.start, .n = s.n };
}

/// The rows between the starts of two ranks' slabs: a quarter of the chunk's own rows (slab_plan.q). The stride of every owned block in the arena, whatever rows the last slab holds.
pub fn ownQ(e: *const Engine, ch: *const Chunk) usize {
    _ = e;
    return slab_plan.q(ch.n);
}

/// The rows of rank `k`'s slab (ownQ rows when every slab is whole).
pub fn slabN(e: *const Engine, ch: *const Chunk, k: usize) usize {
    _ = e;
    return slab_plan.slab(ch.n, k).n;
}

/// The transport's plan for this chunk's owned exchanges, `x` the event's extra number (rocinante.buildEventX).
pub fn rociShape(e: *const Engine, ch: *const Chunk, x: usize) !roci_engine.Shape {
    _ = e;
    return roci_engine.Roci.shapeFor(ch.n, ch.cap, true, x);
}

fn qhcRange(e: *const Engine, ch: *const Chunk, li: usize) ?QRows {
    if (e.two == null) return null;
    // the decoder half and the head (li = layers) on this rank's slab of the decoder rows, once the boundary has handed them over
    // (cutRows: ch.n is then the window's rows, boundary_done set); before the handoff the boundary is boundaryRange's
    if (li >= e.c.layers / 2) return if (ch.boundary_done) anySlab(e, ch) else null;
    return anySlab(e, ch);
}

/// Layer `li`'s owned rows on this node (qhcRange; at the boundary layer of an encoder-only chunk with
/// TF_DS_BOUNDARY_OWNED, boundaryRange): the rows a gather for li's readers produces at the owner path and the rows
/// those readers take, one function for both.
pub fn ownedRange(e: *const Engine, ch: *const Chunk, li: usize) ?QRows {
    return qhcRange(e, ch, li) orelse boundaryRange(e, ch, li);
}

/// Whether a chunk of `n` rows from `start` runs no decoder row: the prompt's replay (its length less the window) at or
/// past the chunk's end, the chunk replayCut stops at the boundary layer.
pub fn encoderOnly(start: usize, n: usize, replay: usize) bool {
    return replay >= start + n;
}

/// The boundary layer's owned rows on this node (TF_DS_BOUNDARY_OWNED in effect, an encoder-only full quarter-HC
/// chunk): its mixes and kv-source update run there (boundaryUpdate), and the gather before it (layer layers / 2 - 1's
/// MoE partials) produces them.
pub fn boundaryRange(e: *const Engine, ch: *const Chunk, li: usize) ?QRows {
    if (li != e.c.layers / 2) return null;
    if (e.two == null) return null;
    // every chunk's boundary is on owned rows (a chunk with decoder rows hands them their rows after it: handoff)
    return if (!ch.boundary_done) anySlab(e, ch) else null;
}

/// Whether a layer's attention front can run on owned rows (TF_DS_FRONT_OWNED): without a compressor (kvSourceUpdate)
/// and an indexer (its weights_proj), only wq_a and wkv read x's rows.
pub fn frontOwnable(has_compressor: bool, has_indexer: bool) bool {
    return !has_compressor and !has_indexer;
}

/// Layer `li`'s owned rows when its attention front runs on them (TF_DS_FRONT_OWNED in effect, a frontOwnable layer,
/// owned rows): attnMixes gathers no x then, and attention's front gathers qr and y (frontOwned).
pub fn frontRange(e: *const Engine, ch: *const Chunk, li: usize) ?QRows {
    if (e.two == null) return null;
    const lay = e.w.layers[li];
    if (!frontOwnable(lay.comp_wkv != null, lay.idx_wq_b != null)) return null;
    return ownedRange(e, ch, li);
}

/// Layer `li`'s owned rows when its router, routing and MoE input codes run on them (TF_DS_FFN_CODES in effect, the
/// checkpoint's own weights, owned rows): ffnMixes gathers no x then, and moe gathers the codes, picks and weights
/// (ffnOwned).
pub fn ffnRange(e: *const Engine, ch: *const Chunk, li: usize) ?QRows {
    if (e.two == null or e.mx == null) return null;
    return ownedRange(e, ch, li);
}

/// Layer `li`'s owned rows when its Engram sums and gate run on them (owned rows): engramApply's gate and image-span
/// copies then take those rows alone.
pub fn engramOwned(e: *const Engine, ch: *const Chunk, li: usize) ?QRows {
    if (e.two == null) return null;
    return ownedRange(e, ch, li);
}

/// The gathered planes of `sub` (ga or gm).
fn fullOf(ch: *const Chunk, sub: Sub) u64 {
    return switch (sub) {
        .attn => ch.ga,
        .moe => ch.gm,
    };
}

/// What the last gather of `sub` left for its readers in this call.
pub fn howOf(ch: *Chunk, sub: Sub) *Gathered {
    return switch (sub) {
        .attn => &ch.ga_how,
        .moe => &ch.gm_how,
    };
}

/// begin's host side: a chunk of `n` rows from position `start`, nothing pending, no gather yet in this call.
pub fn beginHost(ch: *Chunk, n: usize, start: usize) void {
    ch.n = n;
    ch.start = start;
    ch.pending = false;
    ch.encoder_only = false;
    ch.boundary_done = false;
    ch.win_key = .{ std.math.maxInt(usize), 0, 0, 0 }; // (TF_DS_WINDOW_ONCE: one chunk, one stream: its window uploads are made again)
    ch.win_have = 0;
    resetGathers(ch);
}

/// A fillChunk call's start: no gather yet, plane 1 of the tight buffer not yet written in it.
pub fn resetGathers(ch: *Chunk) void {
    ch.ga_how = .{};
    ch.gm_how = .{};
    ch.tight_neg = false;
    ch.own1_sent = false;
}

/// Whether plane 1 of the tight buffer is still to be written in this call (the first owner gather's), marked written.
pub fn takeTightNeg(ch: *Chunk) bool {
    if (ch.tight_neg) return false;
    ch.tight_neg = true;
    return true;
}

/// The plan shape of a pass whose gathers ride the transport (TF_DS_ROCI_GATHERS): a whole chunk of a size the transport has
/// plans for. Null: the chunk has no plan (an error at the call). Every rank decides alike (the chunk and its cap are the same on all).
pub fn gathersShape(e: *const Engine, ch: *const Chunk) ?roci_engine.Shape {
    const t = e.two orelse return null;
    if (t.rc == null) return null;
    return rociShape(e, ch, 0) catch null;
}

/// gathersShape of a chunk the engine runs: the transport has a plan for every chunk, so none ends the chunk by name (there is
/// no other transport).
fn needShape(e: *const Engine, ch: *const Chunk) !roci_engine.Shape {
    return gathersShape(e, ch) orelse error.RociNoShape;
}

/// The plan shape of a pass whose codes ride the transport (TF_DS_ROCI_CODES): a whole chunk of a size the transport has plans for,
/// and a pack of exactly the rows the arena's regions hold. Null: the chunk has no plan (an error at the call). Every rank decides alike.
pub fn codesShape(e: *const Engine, ch: *const Chunk) ?roci_engine.Shape {
    const t = e.two orelse return null;
    if (t.rc == null) return null;
    return rociShape(e, ch, 0) catch null;
}

/// One family's gather of a full pass over the transport: each node's slab of the family's tensors, produced in place in the
/// arena, goes to the three other nodes as RDMA WRITEs, and the consumers wait for the three slabs that land.
fn rociGather(e: *const Engine, class: roci_engine.Class, shape: roci_engine.Shape) !void {
    const rc = e.two.?.rc.?;
    try rc.send(e.s, class, shape);
    try rc.wait(e.s, class);
}

/// Whether layer `li`'s attention gathers x (at its owned rows): its front is not on owned rows (frontRange) and it is not the
/// boundary layer on owned rows.
fn xGathered(e: *const Engine, ch: *const Chunk, li: usize) bool {
    return frontRange(e, ch, li) == null and boundaryRange(e, ch, li) == null;
}

/// TF_DS_HC_STRIDE in effect (Model.open: asked, with quarter-HC): a reader at owned rows takes the gathered planes
/// where they lie in ga / gm (`full + r.start * D`, `cap * D` elements between planes) instead of the tight repack.
pub var hc_stride = false;

/// Where a reader at owned rows takes a sublayer's gathered planes: `ptr` plane 0's first owned row (fp32), and the
/// planes' stride in elements; null is the tight buffer's, the launch's own rows apart.
pub const Partials = struct { ptr: u64, rs: ?usize = null };

/// The chunk's own planes (`full`, fp32 [world, cap, D]) as a reader at owned rows `start ..` takes them: plane 0 at
/// its first row, `cap * D` elements between planes.
pub fn planeRead(full: u64, cap: usize, hidden: usize, start: usize) Partials {
    return .{ .ptr = full + start * hidden * 4, .rs = cap * hidden };
}

/// Where a reader at owned rows `r` takes `sub`'s partials: the tight buffer [e.world, r.n, D] as the gather left it
/// (owner sums), else the chunk's own planes at their stride (TF_DS_HC_STRIDE), else the tight buffer filled by the repack
/// of ga's or gm's owned rows (qhcRepackG) first.
pub fn ownedPartials(e: *const Engine, ch: *Chunk, sub: Sub, r: QRows) !Partials {
    if (repackSource(ch, sub)) |full| {
        if (hc_stride) return planeRead(full, ch.cap, e.c.hidden, r.start);
        try qhcRepackG(e, ch, full, r);
    }
    return .{ .ptr = ch.qhc_g };
}

/// What a reader at owned rows repacks into the tight buffer first: `sub`'s gathered planes (ga or gm), or nothing
/// when the gather filled the tight buffer itself (howOf(sub).owned).
pub fn repackSource(ch: *Chunk, sub: Sub) ?u64 {
    return if (howOf(ch, sub).owned) null else fullOf(ch, sub);
}

/// WP4: `full` (fp32 [e.world, ch.cap, D], i.e. ch.ga or ch.gm) repacked at rows [r.start, r.start + r.n) of every
/// plane into ch.qhc_g, tightly strided [e.world, r.n, D] -- see ch.qhc_g's own comment for why. A local
/// device-to-device copy, one per plane.
fn qhcRepackG(e: *const Engine, ch: *Chunk, full: u64, r: anytype) !void {
    const row_bytes = e.c.hidden * 4;
    for (0..e.world) |w| {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(
            ch.qhc_g + w * r.n * row_bytes,
            full + w * ch.cap * row_bytes + r.start * row_bytes,
            r.n * row_bytes,
            e.s.handle,
        ), "cuMemcpyDtoDAsync");
    }
}

/// Layer `li`'s attention mixes: x = RMSNorm of the streams mixed by pre, pre_a / post / comb from their own mixes;
/// with a pending MoE gather, hc_pre_pf posts it into the streams first.
pub fn attnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    // TF_DS_ROCI_GATHERS: the layers whose x the transport gathers keep it in the arena, the others in the device buffer
    if (ch.x_g != 0) ch.x = ch.x_dev;
    if (ownedRange(e, ch, li)) |r| {
        if (ch.x_g != 0 and xGathered(e, ch, li) and gathersShape(e, ch) != null) ch.x = ch.x_g;
        const row_h = c.hc * c.hidden * 2;
        const row_p = c.hc * 4;
        const row_c = c.hc * c.hc * 4;
        if (ch.pending) {
            // (a rank beyond the chunk's rows under TF_DS_ROCI_ANY owns none: no launch, the streams swap and the gathers go on)
            if (r.n != 0) {
                const posted = try ownedPosted(e, ch, .moe, r, ch.h_alt + r.start * row_h);
                _ = try tri_hc.hcPrePf(e.t, ch.h + r.start * row_h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre + r.start * row_p, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x + r.start * c.hidden * 2, ch.pre_a + r.start * row_p, ch.post + r.start * row_p, ch.comb + r.start * row_c, ch.qhc_part, posted, r.n, c.hidden);
            }
            // swapStreams asserts its argument is exactly ch.h_alt (the whole buffer); the kernel's `h_out` here was
            // offset to this node's own rows (correct for the write it actually did), so swap the base pointers
            // directly instead of going through that helper's assert.
            const old_h = ch.h;
            ch.h = ch.h_alt;
            ch.h_alt = old_h;
            ch.pending = false;
        } else if (r.n != 0) {
            try tri_hc.hcPre(e.t, ch.h + r.start * row_h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre + r.start * row_p, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x + r.start * c.hidden * 2, ch.pre_a + r.start * row_p, ch.post + r.start * row_p, ch.comb + r.start * row_c, ch.qhc_part, r.n, c.hidden);
        }
        // the owned front (frontRange) reads x's own rows alone
        // (the boundary on owned rows, boundaryRange, reads them alone as well)
        if (xGathered(e, ch, li)) {
            try rociGather(e, .ga, try needShape(e, ch));
        }
    } else if (ch.pending) {
        const src = try tri_hc.hcPrePf(e.t, ch.h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_a, ch.post, ch.comb, ch.part, .{ .gathered = ch.gm, .world = e.world, .h_out = ch.h_alt }, ch.n, c.hidden);
        swapStreams(ch, src);
        ch.pending = false;
    } else {
        try tri_hc.hcPre(e.t, ch.h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_a, ch.post, ch.comb, ch.part, ch.n, c.hidden);
    }
}

/// TF_DS_DUMP=DIR (a debugging aid, off in service): TF_DS_DUMP_LAYER's layers' (a comma list, default just "0")
/// intermediates of the first prompt chunk, each written as DIR/n<node>-<name>.bin (raw device bytes) after the
/// stream drains. WP1's exit gate (a tensor dump at layers 0, 8 and 19) is TF_DS_DUMP_LAYER=0,8,19.
pub var dump_dir: ?[:0]const u8 = null;
pub var dump_layers: [8]i64 = .{ 0, -1, -1, -1, -1, -1, -1, -1 };
pub var dump_done: bool = false;
pub var dump_node: u32 = 0;

fn wantLayer(li: usize) bool {
    const l: i64 = @intCast(li);
    for (dump_layers) |d| {
        if (d == l) return true;
    }
    return false;
}

pub fn dump(e: *const Engine, li: usize, name: []const u8, ptr: u64, bytes: usize) void {
    const dir = dump_dir orelse return;
    if (!wantLayer(li) or dump_done or bytes == 0) return;
    e.s.synchronize() catch return;
    const host = std.heap.page_allocator.alloc(u8, bytes) catch return;
    defer std.heap.page_allocator.free(host);
    e.d.check(e.d.api.cuMemcpyDtoH_v2(host.ptr, ptr, bytes), "cuMemcpyDtoH") catch return;
    var buf: [512]u8 = undefined;
    const path = std.mem.printSentinel(&buf, "{s}/n{d}-{s}.bin", .{ dir, dump_node, name }, 0) catch return;
    const f = std.c.fopen(path.ptr, "wb") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(host.ptr, 1, bytes, f);
}

/// mm(layer, x, out_dtype) of m rows: the prompt GEMM beyond 128 rows, the layer's own group (GROUPED) up to 128.
pub fn mmRows(e: *const Engine, ch: *const Chunk, m: usize, l: weights.Linear, x: u64, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize) !void {
    const dt: exl3_linear.DType = switch (out_type) {
        .bf16 => .bf16,
        .fp32 => .f32,
    };
    // Format.native: one kernel for every row count (a row's bits at any m)
    if (e.mx) |k| return mx.linear(k, e.s, ch.mxs, l, x, .bf16, ldx, m, out, os, dt, false, null);
    if (m > decode_rows) return exl3_prefill.matmul(e.pf, e.t, ch.ws, l, x, .bf16, ldx, out, out_type, os, m);
    try grouped(e, ch, m, &.{l}, &.{x}, &.{ldx}, &.{out}, &.{os}, &.{dt});
}

/// mm of the chunk's rows.
pub fn mm(e: *const Engine, ch: *const Chunk, l: weights.Linear, x: u64, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize) !void {
    try mmRows(e, ch, ch.n, l, x, ldx, out, out_type, os);
}

/// TF_DS_REPLAY_FLOOR=1 (model.zig sets it on every rank): a small chunk whose window reaches below the floor reads it
/// by the gather path (off: ring mode, as the served lane reads, the slot's previous request's keys below the floor).
pub var replay_floor = false;

/// How attention reads a chunk's window keys (chunkMode).
pub const ChunkMode = struct {
    /// ring mode: the rows' keys into the ring first, the attention reads the ring
    ring: bool,
    /// a chunk of decode_rows rows or fewer off ring mode: the prompt chunks' one-split variant (tri_attn.Attn.one_split)
    one_split: bool,
};

/// ring mode (a chunk of RING_EXTRA rows or fewer): the rows' keys into the ring first, the attention reads the ring.
/// With replay_floor (`on`), not when the window reaches below `floor` (the replay's cut: below it the ring holds the slot's
/// previous request's keys): such a chunk reads by the gather path from the floor, as one large chunk's rows would.
pub fn chunkMode(n: usize, start: usize, floor: usize, window: usize, on: bool) ChunkMode {
    const below = floor > start -| (window - 1);
    const ring = n <= ring_extra and !(on and below);
    return .{ .ring = ring, .one_split = !ring and n <= tri.decode_rows };
}

/// TF_DS_WINDOW_ONCE (Model.open): the gather indices, the window's low position and the ring write's indices depend only on the chunk
/// (start, rows, the window's floor), not on the layer, so a chunk uploads each once, into buffers of their own, instead of in every
/// layer (3 of the ~111 pageable uploads a 4,096-row pass makes are these, 59 of the 111 in all). The bytes the kernels read are the same.
pub var window_once: bool = false;

/// TF_DS_WINDOW_ONCE's text: on unless set otherwise (a bit-identical speed change): unset or "1" is on, "0" or any other text is off
/// (a typo takes the old path). Surrounding white space is ignored.
pub fn windowOnceWanted(text: ?[]const u8) bool {
    const v = text orelse return true;
    return std.mem.eql(u8, std.mem.trim(u8, v, " \t\r\n"), "1");
}
pub const win_g: u8 = 1;
pub const win_lo: u8 = 2;
pub const win_s: u8 = 4;

/// Whether the upload `bit` is due for this chunk's (start, n, lo, ring mode): true once per key (and marks it made), false after.
pub fn windowFresh(ch: *Chunk, start: usize, n: usize, lo: usize, ring_mode: bool, bit: u8) bool {
    const key: [4]usize = .{ start, n, lo, @intFromBool(ring_mode) };
    if (!std.mem.eql(usize, &key, &ch.win_key)) {
        ch.win_key = key;
        ch.win_have = 0;
    }
    if (ch.win_have & bit != 0) return false;
    ch.win_have |= bit;
    return true;
}

/// attention_k: the partial pa [n, D] fp32 of this rank's heads; the chunk's keys go into the layer's window ring
/// (bf16 [window + RING_EXTRA, head_dim]). A compressed layer also attends to its kv source's latents the indexer picks
/// (a kv source first compresses the chunk into them). `floor`: no window key before this position.
pub fn attention(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, ring: u64, floor: usize, kv_done: bool) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const start = ch.start;
    const hd = c.head_dim;
    const rd = c.rope_dim;
    const hl = e.heads();
    // _cs: the layer's RoPE kind (compressed layers rotate by the compressed table) for every rotation it makes
    const rope = if (lay.ratio != 0) e.compressed else e.plain;
    const ring_rows = c.window + ring_extra;
    // ring mode, or the gather path from the floor (replay_floor: a small chunk whose window reaches below it): chunkMode
    const mode = chunkMode(n, start, floor, c.window, replay_floor);
    const ring_mode = mode.ring;
    const one_split = mode.one_split;
    // attn_in: wq_a and wkv of x, one prompt GEMM each, or one group up to 128 rows; TF_DS_FRONT_OWNED (frontRange):
    // on owned rows with the q norm, then qr and y gathered (frontOwned); TF_DS_QUANT_ONCE: x quantized once for both
    const front = frontRange(e, ch, li);
    switch (attnFront(front != null, quant_once, e.mx != null, frontCodes(lay), n)) {
        .owned => try frontOwned(e, ch, lay, front.?),
        .once => try frontLinears(e.mx.?, e.s, ch.mxs, lay, ch.x, c.hidden, .{ .start = 0, .n = n }, ch.qa, ch.y, hd),
        .linears => {
            try mm(e, ch, lay.wq_a, ch.x, c.hidden, ch.qa, .bf16, lay.wq_a.n);
            try mm(e, ch, lay.wkv, ch.x, c.hidden, ch.y, .bf16, hd);
        },
        .grouped => try grouped(e, ch, n, &.{ lay.wq_a, lay.wkv }, &.{ ch.x, ch.x }, &.{ c.hidden, c.hidden }, &.{ ch.qa, ch.y }, &.{ lay.wq_a.n, hd }, &.{ .bf16, .bf16 }),
    }
    pmark(e, .a_in);
    dump(e, li, "x", ch.x, n * c.hidden * 2);
    dump(e, li, "qa", ch.qa, n * lay.wq_a.n * 2);
    dump(e, li, "y", ch.y, n * hd * 2);
    if (front == null) try tri_basic.rmsnorm(e.t, ch.qa, lay.wq_a.n, lay.q_norm, ch.qr, lay.wq_a.n, c.eps, n, lay.wq_a.n);
    dump(e, li, "qr", ch.qr, n * lay.wq_a.n * 2);
    try mm(e, ch, lay.wq_b, ch.qr, lay.wq_a.n, ch.q, .bf16, hl * hd);
    dump(e, li, "q_norope", ch.q, n * hl * hd * 2);
    try tri_norm.ropeHeads(e.t, ch.q, rope.cos, rope.sin, ch.pos, rd, false, n, hl, hd);
    pmark(e, .a_q);
    // the window keys: the ring's last window - 1 positions before start, then this chunk's rows
    const lo = @max(floor, start -| (c.window - 1));
    const wsrc_rows = if (ring_mode) ring_rows else start - lo + n;
    if (ring_mode) {
        // kv_norm_rope into ring rows pos % R (and its rows out), the attention's window keys the ring, WLO zero
        var slots: [ring_extra]i64 = undefined;
        for (0..n) |j| slots[j] = @intCast((start + j) % ring_rows);
        try upload(e, ch.rslot, &slots, n * 8);
        try tri_norm.kvNormRope(e.t, ch.y, lay.kv_norm, rope.cos, rope.sin, ch.pos, ring, ring_rows, ch.rslot, c.eps, true, rd, ch.wsrc, n, hd);
    } else if (start > lo) {
        // wsrc[:start - lo] = ring[arange(lo, start) % R]
        var ridx: [256]i64 = undefined;
        if (start - lo > ridx.len) return error.WindowTooLong;
        var gidx = ch.ring_idx;
        if (window_once) {
            gidx = ch.ridx_g;
            if (windowFresh(ch, start, n, lo, ring_mode, win_g)) {
                for (0..start - lo) |j| ridx[j] = @intCast((lo + j) % ring_rows);
                try upload(e, gidx, &ridx, (start - lo) * 8);
            }
        } else {
            for (0..start - lo) |j| ridx[j] = @intCast((lo + j) % ring_rows);
            try upload(e, gidx, &ridx, (start - lo) * 8);
        }
        try fill(e, ch.invalid, 0, 4);
        try e.ops.gatherRows(e.s, ring, ring_rows, gidx, ch.wsrc, hd * 2, start - lo, ch.invalid);
    }
    var lo64: i64 = if (ring_mode) 0 else @intCast(lo);
    if (!window_once or windowFresh(ch, start, n, lo, ring_mode, win_lo)) try upload(e, ch.wlo, &lo64, 8);
    const kv = ch.wsrc + (start - lo) * hd * 2;
    // slots -1: the keys go to wsrc only (ring_mode is off); the ring is written after the attention
    if (!ring_mode) try tri_norm.kvNormRope(e.t, ch.y, lay.kv_norm, rope.cos, rope.sin, ch.pos, ring, ring_rows, ch.neg, c.eps, true, rd, kv, n, hd);
    var comp: tri_attn.Comp = .none;
    var n_idx: usize = 0;
    var padded = false;
    if (lay.ratio != 0) {
        const r: usize = lay.ratio;
        if (lay.comp_wkv != null and !kv_done) try kvSourceUpdate(e, ch, cs, sh, li);
        const src = sh.kv_layer orelse return error.NoKvSource;
        const n_comp_end = (start + n) / r;
        // vis = (pos + 1) // ratio: the compressed entries row i may see (host-made: the same integers)
        var vis_store: [max_pass_rows]i64 = undefined;
        const vis_host = try hostRows(ch, &vis_store, n);
        for (vis_host, 0..) |*p, i| p.* = @intCast((start + i + 1) / r);
        try upload(e, ch.vis, vis_host.ptr, n * 8);
        if (lay.idx_wq_b) |wqb| try indexer(e, ch, cs, sh, li, wqb, src, n_comp_end, rope);
        const kk = sh.kk orelse return error.NoIndexerSelection;
        comp = .{ .fp4 = .{ .codes = cs.comp_codes[src], .scales = cs.comp_scales[src] } };
        n_idx = kk;
        if (kk % 16 != 0 and tri_attn.splits(n, one_split) == 1) {
            // sparse_attn's one pass (a prompt chunk, or a small chunk run as one): its pick list padded with -1 to whole
            // 16-column tiles (F.pad), read from there
            const kp = (kk + 15) / 16 * 16;
            try fill(e, ch.cidxp, 0xff, n * kp * 8);
            try e.ops.copyRows(e.s, ch.cidx, kk * 8, ch.cidxp, kp * 8, kk * 8, n);
            padded = true;
        }
    }
    pmark(e, .a_idx);
    const kk = hl * hd;
    const epi = epilogueOn(e, ch, n, kk, hd, rd);
    // TF_DS_ATTN_EPI_FUSED: the epilogue in the part kernel's final store (the kit's _sparse_attn_part_ue); epi_ran says it did
    var epi_ran = false;
    try tri_attn.sparseAttn(e.t, .{
        .q = ch.q,
        .out = ch.o,
        .rows = n,
        .h = hl,
        .hd = hd,
        .sink = lay.sink,
        .wsrc = if (ring_mode) ring else ch.wsrc,
        .wsrc_rows = wsrc_rows,
        .wlo = ch.wlo,
        .ring = ring_mode,
        .parts = .{ .pm = ch.pm_, .pl = ch.pl_, .po = ch.po_ },
        .comp = comp,
        .idx = if (n_idx > 0) (if (padded) ch.cidxp else ch.cidx) else 0,
        .n_idx = n_idx,
        .pos = ch.pos,
        .scale = scale(hd),
        .window = c.window,
        .one_split = one_split,
        .u = tri_attn.attn_u,
        .epi = if (epi and tri_attn.attn_epi_fused) (if (lay.wo_a_rows)
            tri_attn.Epi{ .cos = rope.cos, .sin = rope.sin, .rd = rd, .codes = 0, .scales = 0, .fake = ch.mxs.fake }
        else
            tri_attn.Epi{ .cos = rope.cos, .sin = rope.sin, .rd = rd, .codes = ch.mxs.q, .scales = ch.mxs.s, .fake = 0 }) else null,
        .epi_ran = &epi_ran,
    });
    if (epi_ran) {
        // the part kernel did the inverse RoPE and the quant (o was never written)
    } else if (epi) {
        if (lay.wo_a_rows) try mx.attnEpilogue(e.mx.?, e.s, ch.o, rope.cos, rope.sin, ch.pos, n, hl, hd, rd, 0, 0, ch.mxs.fake, dump_dir != null) else try mx.attnEpilogue(e.mx.?, e.s, ch.o, rope.cos, rope.sin, ch.pos, n, hl, hd, rd, ch.mxs.q, ch.mxs.s, 0, dump_dir != null);
    } else try tri_norm.ropeHeads(e.t, ch.o, rope.cos, rope.sin, ch.pos, rd, true, n, hl, hd);
    // ring[pos[-keep:] % R] = kv[-keep:] (ring mode wrote them before the attention)
    const keep: usize = if (ring_mode) 0 else @min(n, ring_rows);
    if (keep > 0) {
        var idx: [256]i64 = undefined;
        const sidx = if (window_once) ch.ridx_s else ch.ring_idx;
        if (!window_once or windowFresh(ch, start, n, lo, ring_mode, win_s)) {
            for (0..keep) |j| idx[j] = @intCast((start + n - keep + j) % ring_rows);
            try upload(e, sidx, &idx, keep * 8);
        }
        try e.ops.scatterRows(e.s, kv + (n - keep) * hd * 2, hd * 2, sidx, ring, hd * 2, hd * 2, keep);
    }
    pmark(e, .a_core);
    dump(e, li, "o", ch.o, n * hl * hd * 2);
    // wo_a: each group's column block of o read in place, written into its column block of u; then wo_b to fp32
    const groups = lay.groups;
    const gk = hl * hd / groups;
    const uw = groups * lay.wo_a[0].n;
    var col: usize = 0;
    if (e.mx) |k| {
        if (lay.wo_a_rows) {
            // Format.native: o quant-dequantized, then each group's bf16 einsum (Attention.forward); TF_DS_ATTN_EPILOGUE
            // wrote those rows already
            if (epi) try mx.woAFromFake(k, e.s, ch.mxs, lay.wo_a[0..groups], n, ch.u, uw, null) else try mx.woA(k, e.s, ch.mxs, lay.wo_a[0..groups], ch.o, hl * hd, n, ch.u, uw, null);
            dump(e, li, "o_fake", ch.mxs.fake, n * hl * hd * 2);
            dump(e, li, "wo_a0", lay.wo_a[0].w, @as(usize, lay.wo_a[0].n) * lay.wo_a[0].k * 2);
        } else {
            // a layer loaded without the bf16 rows (TF_DS_WOA_W8): the same values from the FP8 rows, woA's bits
            const w8 = lay.wo_a8 orelse return error.NoWoA8;
            if (epi) try mx.woA8PrefillCodes(k, e.s, ch.mxs, w8, groups, n, ch.u, uw) else try mx.woA8Prefill(k, e.s, ch.mxs, w8, groups, ch.o, hl * hd, n, ch.u, uw);
            dump(e, li, "wo_a8", w8.w, @as(usize, w8.n) * w8.k);
        }
    } else if (n > decode_rows) {
        for (lay.wo_a[0..groups], 0..) |wo, g| {
            try mm(e, ch, wo, ch.o + g * gk * 2, hl * hd, ch.u + col * 2, .bf16, uw);
            col += wo.n;
        }
    } else {
        // one group: the column blocks of o read in place, written into u's column blocks
        var xs: [4]u64 = undefined;
        var outs: [4]u64 = undefined;
        for (lay.wo_a[0..groups], 0..) |wo, g| {
            xs[g] = ch.o + g * gk * 2;
            outs[g] = ch.u + col * 2;
            col += wo.n;
        }
        const lds = [_]usize{ hl * hd, hl * hd, hl * hd, hl * hd };
        const oss = [_]usize{ uw, uw, uw, uw };
        const dts = [_]exl3_linear.DType{ .bf16, .bf16, .bf16, .bf16 };
        try grouped(e, ch, n, lay.wo_a[0..groups], xs[0..groups], lds[0..groups], outs[0..groups], oss[0..groups], dts[0..groups]);
    }
    pmark(e, .a_woa);
    dump(e, li, "u", ch.u, n * uw * 2);
    if (e.two) |t| {
        // 2D: the column partner's groups first, then this pair's columns; with TF_DS_ROCI_SENT_FIRST the GEMM that rings the first part's doorbell
        if (attnSentFirst(e, ch, li)) _ = try t.woBSentFirst(e, ch, lay) else try t.woB(e, ch, lay);
        dump(e, li, "u_full", t.u_full, n * (t.uw[0] + t.uw[1]) * 2);
        dump(e, li, "pa", ch.pa, n * lay.wo_b.n * 4);
        return;
    }
    try mm(e, ch, lay.wo_b, ch.u, uw, ch.pa, .fp32, c.hidden);
}

/// TF_DS_FRONT_OWNED's attention front at owned rows `r` (frontRange): frontRows, then every rank's rows of qr and y
/// gathered over the transport (rociGather). qr and y then hold the whole chunk's front, the bytes
/// the baseline front leaves (every kernel row-wise, the gathers move bytes); x and qa hold rows r alone.
fn frontOwned(e: *const Engine, ch: *Chunk, lay: weights.Layer, r: QRows) !void {
    if (r.n != 0) try frontRows(e.mx.?, e.t, e.s, ch.mxs, lay, ch.x, e.c.hidden, r, ch.qa, ch.y, ch.qr, e.c.head_dim, e.c.eps);
    return rociGather(e, .gf, try needShape(e, ch));
}

/// The owned front's kernels at rows `r` of x [*, ldx] bf16: x's rows quantized once (act_quant: a row's 32-value
/// groups each on its own) into the linears' scratch, wq_a and wkv on the codes (mx.linear's GEMM, a row's bits at any
/// row count) into qa's rows r [*, wq_a.n] and y's [*, hd], then the q RMSNorm of qa's rows r into qr's (frontNorm).
/// The owned rows of every rank when each owns `q` of a chunk: rank k's [k q, (k + 1) q), for each q of `qs` (the start
/// checks' row sets). Caller frees.
pub fn ownedRows(gpa: std.mem.Allocator, qs: []const usize, world: usize) ![]QRows {
    const out = try gpa.alloc(QRows, qs.len * world);
    for (qs, 0..) |q, i| for (0..world) |k| {
        out[i * world + k] = .{ .start = k * q, .n = q };
    };
    return out;
}

/// wq_a and wkv FP8 / FP4 on one K (frontOwnedOff checks it at the start).
pub fn frontRows(k: *const mx.Kernels, t: tri.Tri, s: cuda.Stream, sc: mx.Scratch, lay: weights.Layer, x: u64, ldx: usize, r: QRows, qa: u64, y: u64, qr: u64, hd: usize, eps: f32) !void {
    try frontLinears(k, s, sc, lay, x, ldx, r, qa, y, hd);
    try frontNorm(t, lay, qa, qr, r, eps);
}

/// wq_a and wkv of x's rows `r` [*, ldx] bf16 on one quant: the rows quantized once (act_quant into the linears'
/// scratch, a row's 32-value groups each on its own), then each linear's GEMM on those codes (mx.linear's GEMM) into
/// qa's rows r [*, wq_a.n] and y's [*, hd]: mx.linear's launches for each, the second quant (the same codes) left out.
/// wq_a and wkv FP8 / FP4 on one K (frontCodes).
pub fn frontLinears(k: *const mx.Kernels, s: cuda.Stream, sc: mx.Scratch, lay: weights.Layer, x: u64, ldx: usize, r: QRows, qa: u64, y: u64, hd: usize) !void {
    if (!frontCodes(lay)) return error.NotAnMxLinear;
    const kd: usize = lay.wq_a.k;
    if (r.n > sc.rows or kd > sc.max_k) return error.ScratchTooSmall;
    const qw: usize = lay.wq_a.n;
    try mx.quant(k, s, x + r.start * ldx * 2, .bf16, ldx, r.n, kd, sc.q, kd, sc.s, kd / 32);
    try mx.gemm(k, s, lay.wq_a, sc.q, kd, sc.s, kd / 32, r.n, qa + r.start * qw * 2, qw, .bf16, false, sc.z);
    try mx.gemm(k, s, lay.wkv, sc.q, kd, sc.s, kd / 32, r.n, y + r.start * hd * 2, hd, .bf16, false, sc.z);
}

/// TF_DS_QUANT_ONCE in effect (Model.open: asked, with the checkpoint's own weights): attention's front not on owned
/// rows quantizes x once for wq_a and wkv (attnFront's .once).
pub var quant_once = false;

/// TF_DS_ATTN_EPILOGUE in effect (Model.open: asked, with the checkpoint's own weights): a prompt chunk's attention
/// output takes the inverse RoPE and wo_a's quant in one launch, and wo_a reads its codes (or fake rows).
pub var attn_epilogue = false;

/// Whether a layer's attention output takes the epilogue over `n` rows: the kernel's shapes (the last rd columns and
/// the rest whole 32-value groups) and the chunk's scratch for kk codes or fake rows.
pub fn epilogueOn(e: *const Engine, ch: *const Chunk, n: usize, kk: usize, hd: usize, rd: usize) bool {
    if (!attn_epilogue or e.mx == null) return false;
    if (n == 0 or n > ch.mxs.rows or kk > ch.mxs.max_k) return false;
    return kk % 32 == 0 and hd % 32 == 0 and rd > 0 and rd <= hd and rd % 32 == 0 and (hd - rd) % 32 == 0;
}

/// How attention computes wq_a and wkv of x: on owned rows (TF_DS_FRONT_OWNED's frontOwned), on x quantized once
/// (TF_DS_QUANT_ONCE: frontLinears over the chunk's rows), or the baseline: each linear on its own (mm, which quantizes x for
/// each FP8 linear) above 128 rows, the layer's group (grouped) up to 128. `codes`: frontCodes of the layer.
pub const Front = enum { owned, once, linears, grouped };
pub fn attnFront(owned: bool, once: bool, native: bool, codes: bool, n: usize) Front {
    if (owned) return .owned;
    if (once and native and codes) return .once;
    return if (n > decode_rows) .linears else .grouped;
}

/// Whether one quant of x's rows feeds both of a layer's front linears: wq_a and wkv FP8 / FP4 on one K.
fn frontCodes(lay: weights.Layer) bool {
    const mxl = struct {
        fn is(l: weights.Linear) bool {
            return l.fmt == .fp8 or l.fmt == .fp4;
        }
    }.is;
    return mxl(lay.wq_a) and mxl(lay.wkv) and lay.wq_a.k == lay.wkv.k;
}

/// The owned front's q RMSNorm: qa's rows r [*, wq_a.n] into qr's (one _rmsnorm program a row, the whole chunk's
/// launch's arguments at the rows' offsets).
pub fn frontNorm(t: tri.Tri, lay: weights.Layer, qa: u64, qr: u64, r: QRows, eps: f32) !void {
    const qw: usize = lay.wq_a.n;
    try tri_basic.rmsnorm(t, qa + r.start * qw * 2, qw, lay.q_norm, qr + r.start * qw * 2, qw, eps, r.n, qw);
}

/// TF_DS_FRONT_OWNED's start check on this rank (null: it can run; else the reason, for Two.agreeFrontOwned):
/// quarter-HC, the checkpoint's own FP8 / MXFP4 weights (`native`), every front layer below `half` with wq_a and wkv
/// on one quant (frontCodes), and every q RMSNorm launch frontNorm makes at each of the `world` ranks' owned rows of
/// each row count in `qs` found in this rank's kernel set: `finds.has(name, args, consts)` (the engine's: aot.Set.find,
/// as a launch finds its variant).
pub fn frontOwnedOff(gpa: std.mem.Allocator, finds: anytype, native: bool, layers: []const weights.Layer, half: usize, eps: f32, qs: []const usize, world: usize) !?[]const u8 {
    const rows = try ownedRows(gpa, qs, world);
    defer gpa.free(rows);
    return frontOwnedOffAt(gpa, finds, native, layers, half, eps, rows);
}

pub fn frontOwnedOffAt(gpa: std.mem.Allocator, finds: anytype, native: bool, layers: []const weights.Layer, half: usize, eps: f32, rows: []const QRows) !?[]const u8 {
    if (!native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    var log: tri.Log = .{ .gpa = gpa };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    // qa and qr as the arena aligns a chunk's buffers (256 bytes)
    const qa: u64 = 1 << 32;
    const qr: u64 = 2 << 32;
    for (layers[0..@min(half, layers.len)]) |lay| {
        if (!frontOwnable(lay.comp_wkv != null, lay.idx_wq_b != null)) continue;
        if (!frontCodes(lay)) return "wq_a and wkv are not FP8 / FP4 on one K";
        for (rows) |r| try frontNorm(t, lay, qa, qr, r, eps);
    }
    for (log.items.items) |l| if (!l.foundIn(finds)) return "this rank's kernel set has no _rmsnorm variant for the owned rows";
    return null;
}

/// kv_source_update of a kv-source layer on the chunk (no earlier pending row: a chunk starting on a group boundary):
/// _compress's new latents (ratio 1: RMSNorm of the bf16 projection; ratio 2: of the pairs' softmax-weighted kv), its
/// positional store's last RAW rows, then the index keys (RMSNorm of idx_wk's projection, RoPE) and the RoPE'd latents
/// into the caches as FP4 (fp4_store).
fn kvSourceUpdate(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize) !void {
    return kvSourceAt(e, ch, cs, sh, li, ch.x, ch.start, ch.n);
}

/// kvSourceUpdate of `n` rows of x [*, hidden] bf16 from position `start`: the compressor on those rows, their groups
/// into the layer's caches (the whole chunk's: ch.x, ch.start, ch.n; the boundary's owned rows: theirs).
pub fn kvSourceAt(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, x: u64, start: usize, n: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const hd = c.head_dim;
    const rd = c.rope_dim;
    const r: usize = lay.ratio;
    const cwkv = lay.comp_wkv.?;
    if (start % r != 0) return notPorted(@src()); // the group's earlier rows from the positional store
    const full = n / r;
    if (r == 1) {
        try mmRows(e, ch, n, cwkv, x, c.hidden, ch.lat2, .bf16, hd);
        try e.exact.rmsNorm(e.s, ch.lat2, hd, lay.comp_norm, ch.lat, hd, n, hd, c.eps);
    } else {
        if (r != 2) return notPorted(@src());
        try mmRows(e, ch, n, cwkv, x, c.hidden, ch.kvc, .fp32, hd);
        try mmRows(e, ch, n, lay.comp_wgate.?, x, c.hidden, ch.scc, .fp32, hd);
        // rk[pw] = kv[-keep:], rs[pw] = score[-keep:]: pw = positions start + n - keep .. % RAW
        const keep: usize = @min(n, raw_rows);
        var idx: [raw_rows]i64 = undefined;
        for (0..keep) |j| idx[j] = @intCast((start + n - keep + j) % raw_rows);
        try upload(e, ch.raw_idx, &idx, keep * 8);
        try e.ops.scatterRows(e.s, ch.kvc + (n - keep) * hd * 4, hd * 4, ch.raw_idx, cs.raw_kv[li], hd * 4, hd * 4, keep);
        try e.ops.scatterRows(e.s, ch.scc + (n - keep) * hd * 4, hd * 4, ch.raw_idx, cs.raw_score[li], hd * 4, hd * 4, keep);
        if (full == 0) { // a one-row chunk (a prompt one past a chunk end): its row waits in the store, no new group
            sh.kv_layer = li;
            return;
        }
        try e.exact.compress2(e.s, ch.kvc, ch.scc, ch.lat2, full, hd);
        try e.exact.rmsNorm(e.s, ch.lat2, hd, lay.comp_norm, ch.lat, hd, full, hd, c.eps);
    }
    sh.kv_layer = li;
    // groups first // r .. and their positions group * r (the rotations _rot reads)
    var g_store: [max_pass_rows]i64 = undefined;
    var p_store: [max_pass_rows]i64 = undefined;
    const g_host = try hostRows(ch, &g_store, full);
    const p_host = try hostRows(ch, &p_store, full);
    for (g_host, p_host, 0..) |*g, *p, j| {
        g.* = @intCast(start / r + j);
        p.* = @intCast((start / r + j) * r);
    }
    try upload(e, ch.groups, g_host.ptr, full * 8);
    try upload(e, ch.gpos, p_host.ptr, full * 8);
    const rot = e.compressed;
    if (lay.idx_wk) |iwk| {
        const id = c.index_head_dim;
        try mmRows(e, ch, full, iwk, ch.lat, hd, ch.lat2, .bf16, id);
        try e.exact.rmsNorm(e.s, ch.lat2, id, lay.idx_k_norm, ch.ik, id, full, id, c.eps);
        try e.exact.rope(e.s, ch.ik, id, id - rd, rot.cos, rot.sin, ch.gpos, full, rd / 2, false);
        // switch "comp": the packed cache's rows in one launch (fp4_store: store_rows' bytes)
        try tri_attn.fp4Store(e.t, ch.ik, id, cs.idx_codes[li], cs.idx_scales[li], ch.groups, full, id, 32, false);
    }
    try e.ops.copyRows(e.s, ch.lat, hd * 2, ch.lat2, hd * 2, hd * 2, full);
    try e.exact.rope(e.s, ch.lat2, hd, hd - rd, rot.cos, rot.sin, ch.gpos, full, rd / 2, false);
    try tri_attn.fp4Store(e.t, ch.lat2, hd, cs.comp_codes[li], cs.comp_scales[li], ch.groups, full, hd, 16, true);
}

/// TF_DS_BOUNDARY_OWNED's boundary for an encoder-only full chunk at owned rows `r` (boundaryRange), in place of the
/// resync and replayCut: the layer's mixes on r (attnMixes' owned branch: the gathered MoE partials of the layer
/// before posted there, no x gather), its kv-source update of x's rows r (kvSourceAt: the compressor, the index keys,
/// both caches' groups start + r.start ..), then every rank's rows of the layer's four cache arrays all-gathered in
/// place (boundaryGathers): the caches hold the whole chunk's groups, the bytes replayCut leaves; the chunk stops here.
pub fn boundaryUpdate(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, r: QRows) !void {
    try attnMixes(e, ch, li);
    if (e.w.layers[li].comp_wkv == null) return error.NoKvSource;
    const c = e.c;
    if (r.n != 0) try kvSourceAt(e, ch, cs, sh, li, ch.x + r.start * c.hidden * 2, ch.start + r.start, r.n) else sh.kv_layer = li;
    return boundaryRociAny(e, cs, li, ch, try needShape(e, ch));
}

/// One device-to-device copy of the boundary's staging.
pub const BoundaryCopy = struct { dst: u64, src: u64, bytes: usize };

/// The boundary's slab of node `k` in the arena's boundary region at `base`: `slab` bytes (r.n rows of the four arrays).
pub fn boundarySlab(base: u64, slab: usize, k: usize) u64 {
    return base + k * slab;
}

/// The copies that stage this node's rows of the four cache arrays (parts: boundaryGathers, the in-place layout: rank k's part
/// of an array is at buf + k * bytes) into its slab of the arena, array after array.
pub fn boundaryOut(parts: *const [4]comm.InPlace, base: u64, slab: usize, rank: usize) [4]BoundaryCopy {
    var out: [4]BoundaryCopy = undefined;
    var at = boundarySlab(base, slab, rank);
    for (&out, parts) |*cp, pt| {
        cp.* = .{ .dst = at, .src = pt.buf + rank * pt.bytes, .bytes = pt.bytes };
        at += pt.bytes;
    }
    return out;
}

/// The copies that put the slab of node `k` (landed in the arena) into its rows of the four arrays.
pub fn boundaryIn(parts: *const [4]comm.InPlace, base: u64, slab: usize, k: usize) [4]BoundaryCopy {
    var out: [4]BoundaryCopy = undefined;
    var at = boundarySlab(base, slab, k);
    for (&out, parts) |*cp, pt| {
        cp.* = .{ .dst = pt.buf + k * pt.bytes, .src = at, .bytes = pt.bytes };
        at += pt.bytes;
    }
    return out;
}

/// The four cache arrays' rows (bytes a row) of the boundary layer, array after array: compressed codes and scales, index codes and scales.
pub fn boundaryRowBytes(hd: usize, id: usize) [4]usize {
    return .{ hd / 2, hd / 16, id / 2, id / 32 };
}

/// The copies between rank `k`'s `nk` rows of the four arrays (`bufs`: each array at the chunk's first row; rank k's rows start k q rows in)
/// and its slab of the arena's boundary region at `base`, each array `q` rows apart inside it (the slab's `q` rows are the stride of any
/// chunk's slabs, the last one short): out stages them, else they are copied back. An empty slab has no copy.
pub fn boundaryCopiesAny(bufs: [4]u64, rb: [4]usize, base: u64, q: usize, k: usize, nk: usize, out: bool) [4]BoundaryCopy {
    var res: [4]BoundaryCopy = undefined;
    var at = base + k * q * roci.gb_row;
    for (&res, bufs, rb) |*cp, buf, row| {
        const arr = buf + k * q * row;
        cp.* = if (out) .{ .dst = at, .src = arr, .bytes = nk * row } else .{ .dst = arr, .src = at, .bytes = nk * row };
        at += q * row;
    }
    return res;
}

/// boundaryRoci for a chunk of any row count (TF_DS_ROCI_ANY): each rank's slab is `q` rows apart in the region, the last one short.
fn boundaryRociAny(e: *const Engine, cs: *const Caches, li: usize, ch: *const Chunk, shape: roci_engine.Shape) !void {
    const rc = e.two.?.rc.?;
    const c = e.c;
    const rb = boundaryRowBytes(c.head_dim, c.index_head_dim);
    var bufs: [4]u64 = .{ cs.comp_codes[li], cs.comp_scales[li], cs.idx_codes[li], cs.idx_scales[li] };
    for (&bufs, rb) |*b, row| b.* += ch.start * row;
    const q = slab_plan.q(ch.n);
    const base = rc.addr(.g_b);
    const mine: usize = e.comm.rank;
    const nk = slab_plan.slab(ch.n, mine).n;
    if (nk != 0) for (boundaryCopiesAny(bufs, rb, base, q, mine, nk, true)) |cp| try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, e.s.handle), "cuMemcpyDtoDAsync");
    try rociGather(e, .gb, shape);
    for (0..e.comm.world) |k| {
        if (k == mine) continue;
        const nj = slab_plan.slab(ch.n, k).n;
        if (nj == 0) continue;
        for (boundaryCopiesAny(bufs, rb, base, q, k, nj, false)) |cp| try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, e.s.handle), "cuMemcpyDtoDAsync");
    }
}

/// The boundary's gathers: the layer's compressed codes [*, hd / 2] and scales [*, hd / 16] and index codes [*, id / 2]
/// and scales [*, id / 32] (bytes a group: a group a row at compress ratio 1), every rank's groups start + r.start ..
/// in place over the chunk's groups from `start` (rank k's part at buf + k * bytes: its owned rows, r.start = k r.n).
pub fn boundaryGathers(cs: *const Caches, li: usize, start: usize, r: QRows, hd: usize, id: usize) [4]comm.InPlace {
    const arrays = [4]u64{ cs.comp_codes[li], cs.comp_scales[li], cs.idx_codes[li], cs.idx_scales[li] };
    const rows = [4]usize{ hd / 2, hd / 16, id / 2, id / 32 };
    var parts: [4]comm.InPlace = undefined;
    for (&parts, arrays, rows) |*pt, buf, row| pt.* = .{ .buf = buf + start * row, .bytes = r.n * row };
    return parts;
}

/// Why TF_DS_BOUNDARY_OWNED, asked for, stays off (null: it runs): owned rows need quarter-HC, and the owned update is
/// the boundary layer `lay`'s at compress ratio 1 with its index keys (one group a row, no positional store: its four
/// cache arrays are the layer's whole kept state from an encoder-only chunk).
pub fn boundaryOwnedOff(lay: weights.Layer) ?[]const u8 {
    if (lay.ratio != 1 or lay.comp_wkv == null or lay.idx_wk == null) return "needs a boundary layer of compress ratio 1 with its index keys";
    return null;
}

/// WP6(b) (docs/prefill-plan.md, indexer row-sharding): this layer's owned row range on this node, when
/// TF_DS_IDX_SHARD applies -- a full-cap 2D chunk, strictly before the candidate-source layer (li <
/// c.candidate_source). That bound is not coincidental: it is the indexer's own switch point, after which the
/// per-row "keyed" scoring this shards stops and a cross-layer pool (ch.pool, written once at li ==
/// candidate_source over every row, read by every later layer) takes over -- excluded here by construction, so
/// no boundary resync like WP4's is needed. A short chunk, a resumed chunk under cap rows, a partial final pass (a cap
/// off the resume grid), a non-2D engine, an unaligned quarter (the kit's index kernels need every block's rows to
/// start on a 16-row boundary) or the flag off (agreeIdxShard) all fall through to the baseline unsharded path (null).
fn idxShardRange(e: *const Engine, ch: *const Chunk, li: usize) ?QRows {
    const t = e.two orelse return null;
    if (!t.idx_shard) return null;
    if (li >= e.c.candidate_source) return null;
    return anySlab(e, ch);
}

/// The indexer of a layer with index queries (switch "prompt_keys", layers before the candidate source): the
/// queries (idx_wq_b of q's latent, RoPE, fp4_qd's bytes), the heads' weights (x.float() @ idx_proj.t() through cuBLAS,
/// to bf16, times idx_dim ** -0.5 * idx_heads ** -0.5), each row's int64 keys of the kv source's index keys and the
/// top-k selection into cidx.
/// WP6(b): when idxShardRange applies, the row-block loop below covers only this node's owned quarter (the same
/// pointer-offset-into-the-full-buffer pattern it already uses for block 2, 3, ... of any multi-block call --
/// see the loop's own "a row's scores and picks are its own" comment, and the offline proof,
/// tf-dsv41-wp6b-idx-shard-replay, for a nonzero FIRST block), and ch.cidx is all-gathered back to full width
/// before this returns, so every caller sees the same fully-populated cidx either way.
fn indexer(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, wqb: weights.Linear, src: usize, n_comp_end: usize, rope: Rope) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const ih = c.index_heads;
    const id = c.index_head_dim;
    if (n_comp_end == 0) { // cidx [n, 0]: no compressed entry yet (a one-token prompt), the attention reads none
        sh.kk = 0;
        return;
    }
    if (n_comp_end > ch.max_comp) return error.ChunkTooLong;
    const idxr = idxShardRange(e, ch, li);
    // the queries and the heads' weights on the rows the scorer reads: TF_DS_IDX_PROJ_OWNED with idx_shard's owned rows,
    // else every row (projRows); each launch a row's own arithmetic, at the rows' offsets
    const pr = projRows(idxr, if (e.two) |t| t.idx_proj_owned else false, n);
    const qw: usize = lay.wq_a.n;
    if (pr.n != 0) {
        try mmRows(e, ch, pr.n, wqb, ch.qr + pr.start * qw * 2, qw, ch.iq + pr.start * ih * id * 2, .bf16, ih * id);
        try idxQueries(e.t, ch.iq, ch.iq4, rope.cos, rope.sin, ch.pos, pr, c.rope_dim, ih, id);
    }
    if (pr.n == 0) {
        // (an empty owned slab: nothing to project, the scorer below reads none either)
    } else if (e.mx) |k| {
        // Format.native: weights_proj, a bf16 Linear, times the scale (Indexer.forward), at any row count
        try mx.bf16Gemm(k, e.s, ch.x + pr.start * c.hidden * 2, c.hidden, lay.idx_proj, c.hidden, pr.n, ih, c.hidden, ch.iw + pr.start * ih * 2, ih, .bf16, exact.indexScale(id, ih), true, 1, 0, 0, 0, null);
    } else {
        if (n <= tri.decode_rows) {
            // a decode-sized chunk: the row-invariant matmul of the fp16 projection (rowmm2)
            try tri_norm.rowmm2(e.t, ch.x, c.hidden, lay.idx_proj_h, ch.wl, n, c.hidden, ih);
        } else {
            try e.ops.toF32(e.s, ch.x, ch.xf, n * c.hidden);
            try e.blas.xwT(ch.xf, lay.idx_proj, ch.wl, n, c.hidden, ih);
        }
        try e.exact.bf16Scale(e.s, ch.wl, ch.iw, n * ih, exact.indexScale(id, ih));
    }
    const kk = @min(c.index_topk, n_comp_end);
    // rows in blocks so the [rows, n_comp] keys stay bounded at long contexts; a block a multiple of 16 rows, so every
    // block's views (vis + r0 * 8 and the rest) start on a 16-byte boundary, the alignment the kit's index kernels were
    // captured for (a 137,612-token prompt gave 121-row blocks: MissingTritonVariant at the second). A row's scores and
    // picks are its own, so the block size changes no bits.
    const rb = @max(16, @min(n, (@as(usize, 1) << 26) / (4 * @max(n_comp_end, 1)) / 16 * 16));
    if (keysRefused(rb, n_comp_end, ch.key_cap)) return error.ChunkTooLong;
    const keyed = li != c.candidate_source and !(c.candidate_source < li) and kk & (kk - 1) == 0;
    const k: tri_index.IndexK = .{ .fp4 = .{ .codes = cs.idx_codes[src], .scales = cs.idx_scales[src] } };
    const torch_topk: tri_index.TopK = .{ .top = ch.ktop, .ctx = @ptrCast(@constCast(e)), .run = torchTopk };
    if (keyed and tri_index.prunes(n_comp_end, kk) and (rb > pruned_rows or ch.pbufs.cand == 0)) return error.ChunkTooLong;
    // the candidate source and the layers after it: fp32 scores, the pool (_candidates at the source, apply_candidates
    // after it), topk_indices and -1 past vis. While every block of the scores fits in the pool (cand_blocks of them)
    // the pool keeps each block with a finite score (the newest pinned: it holds the row's last visible key) and
    // apply_candidates rewrites only scores already -inf: no launch changes a byte. Past it the source's block maxima
    // (the newest block pinned to +inf) pick the pool's cand_blocks blocks (topk_indices: ties to the lower block), the
    // picked blocks with a finite maximum make the chunk's mask, and the later layers' scores outside it become -inf.
    const cb = c.candidate_block;
    const nbk = (n_comp_end + cb - 1) / cb;
    const pooled = !keyed and nbk > c.candidate_blocks;
    if (pooled and (ch.pool == 0 or nbk > ch.pool_ms or n > ch.dec_rows)) return error.ChunkTooLong;
    const pk = @min(c.candidate_blocks, nbk);
    dump(e, li, "iq", ch.iq, n * ih * id * 2);
    dump(e, li, "iq4", ch.iq4, n * ih * id * 2);
    dump(e, li, "iw", ch.iw, n * ih * 2);
    var r0: usize = if (idxr) |r| r.start else 0;
    const r_end: usize = if (idxr) |r| r.start + r.n else n;
    while (r0 < r_end) : (r0 += rb) {
        // the block's rows: iq[r0:r1], wts[r0:r1] and vis[r0:r1] as views (their offsets the served pointers'), its
        // selection into cidx[r0:r1]
        const m = @min(r_end, r0 + rb) - r0;
        const q = ch.iq4 + r0 * ih * id * 2;
        const wb = ch.iw + r0 * ih * 2;
        const vr = ch.vis + r0 * 8;
        const out = ch.cidx + r0 * kk * 8;
        if (idx_spread.enabled and ih == 32 and id == 128) try idx_spread.observe(e.d, e.s, e.mx.?, ch.iq + r0 * ih * id * 2, cs.idx_scales[src], vr, m, n_comp_end);
        if (keyed) {
            try tri_index.indexScore(e.t, q, k, wb, vr, n_comp_end, ch.keys, null, true, ch.tmax, false, m, ih, id);
            if (prunedOnTiles(if (e.mx) |x| x.topk_tiles else false, n_comp_end, kk)) {
                try mx.topkTiles(e.mx.?, e.s, ch.keys, n_comp_end, ch.tmax, n_comp_end, kk, vr, out, m);
            } else try tri_index.topkSelectPruned(e.t, ch.keys, n_comp_end, ch.tmax, kk, vr, out, ch.pbufs, torch_topk, m, n_comp_end);
        } else {
            try tri_index.indexScore(e.t, q, k, wb, vr, n_comp_end, ch.score, null, false, null, false, m, ih, id);
            const mrows = ch.pool + r0 * ch.pool_ms;
            if (pooled and li == c.candidate_source) {
                try e.ops.blockMax(e.s, ch.score, n_comp_end, n_comp_end, cb, vr, ch.bmax, nbk, m);
                try e.ops.topkIndices(e.s, ch.bmax, nbk, m, nbk, pk, ch.every, ch.pidx);
                try e.ops.poolPick(e.s, ch.bmax, nbk, nbk, ch.pidx, pk, mrows, ch.pool_ms, 0, m);
            } else if (pooled and c.candidate_source < li) {
                if (!sh.pool) return error.NoCandidatePool;
                try e.ops.applyPool(e.s, ch.score, n_comp_end, n_comp_end, cb, mrows, ch.pool_ms, m);
            }
            try e.ops.topkIndices(e.s, ch.score, n_comp_end, m, n_comp_end, kk, vr, out);
        }
    }
    if (pooled and li == c.candidate_source) sh.pool = true;
    // WP6(b): this rank's owned-row slice of cidx, all-gathered to every rank -- a plain world-4 all-gather, in
    // place (in place: send == recv + rank * count), the same pattern qhcGatherRows already uses
    // for ch.x. Bytes, not a typed count: cidx's per-row width (kk int64 keys) is uniform across rows for this
    // call, so a raw byte gather needs no element-type opinion.
    if (idxr) |r| {
        // (the transport's plan holds the top-k at index_topk keys a row, the rows' stride in the arena; fewer keys, an error at the call; with
        // TF_DS_ROCI_ANY the plan carries the rows' width in keys, any)
        _ = r;
        try rociGather(e, .gi, try rociShape(e, ch, kk));
    }
    dump(e, li, "cidx", ch.cidx, n * kk * 8);
    sh.kk = kk;
}

/// The rows the indexer's projections run on: idx_shard's owned rows when TF_DS_IDX_PROJ_OWNED is in effect (`on`)
/// and the layer has them (`idxr`: the scorer reads only those), else all n.
pub fn projRows(idxr: ?QRows, on: bool, n: usize) QRows {
    if (on) if (idxr) |r| return r;
    return .{ .start = 0, .n = n };
}

/// The index queries' launches at rows r: the RoPE of iq's rows r (POS from row r.start) and fp4_qd's bytes of them
/// into iq4's rows r (switch "idx": one launch), the whole chunk's launches at the rows' offsets.
pub fn idxQueries(t: tri.Tri, iq: u64, iq4: u64, cos: u64, sin: u64, pos: u64, r: QRows, rd: usize, ih: usize, id: usize) !void {
    const off = r.start * ih * id * 2;
    try tri_norm.ropeHeads(t, iq + off, cos, sin, pos + r.start * 8, rd, false, r.n, ih, id);
    try tri_attn.fp4QdP2(t, iq + off, iq4 + off, r.n * ih * id);
}

/// TF_DS_IDX_PROJ_OWNED's start check on this rank (null: it can run; else the reason, for Two.agreeIdxProjOwned):
/// idx_shard in effect, the checkpoint's own weights, and every index query launch idxQueries makes at each of the
/// `world` ranks' owned rows of each row count in `qs`, on every index layer before the candidate source, found in this
/// rank's kernel set (`finds.has`: aot.Set.find, as a launch finds its variant).
pub fn idxProjOwnedOff(gpa: std.mem.Allocator, finds: anytype, native: bool, idx_shard: bool, layers: []const weights.Layer, candidate_source: usize, rd: usize, ih: usize, id: usize, qs: []const usize, world: usize) !?[]const u8 {
    const rows = try ownedRows(gpa, qs, world);
    defer gpa.free(rows);
    return idxProjOwnedOffAt(gpa, finds, native, idx_shard, layers, candidate_source, rd, ih, id, rows);
}

pub fn idxProjOwnedOffAt(gpa: std.mem.Allocator, finds: anytype, native: bool, idx_shard: bool, layers: []const weights.Layer, candidate_source: usize, rd: usize, ih: usize, id: usize, rows: []const QRows) !?[]const u8 {
    if (!idx_shard) return "needs TF_DS_IDX_SHARD=1";
    if (!native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    var log: tri.Log = .{ .gpa = gpa };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    // the buffers as the arena aligns a chunk's (256 bytes)
    const iq: u64 = 1 << 32;
    const iq4: u64 = 2 << 32;
    const pos: u64 = 3 << 32;
    const cos: u64 = 4 << 32;
    const sin: u64 = 5 << 32;
    for (layers[0..@min(candidate_source, layers.len)]) |lay| {
        if (lay.idx_wq_b == null) continue;
        for (rows) |r| try idxQueries(t, iq, iq4, cos, sin, pos, r, rd, ih, id);
    }
    for (log.items.items) |l| if (!l.foundIn(finds)) return "this rank's kernel set has no _rope_heads / _fp4qd_p2 variant for the owned rows";
    return null;
}

/// Whether the indexer's keyed top-k of a row block over n keys takes tf_dsn_topk_tiles (TF_DS_TOPK_TILES `on`): where
/// the chain prunes (tri_index.prunes: past TOPK_FUSED_MAX keys and more tiles than k) and k fits the kernel; elsewhere
/// topkSelectPruned's own choice (topkSelect) as in the baseline.
pub fn prunedOnTiles(on: bool, n: usize, k: usize) bool {
    return on and tri_index.prunes(n, k) and k <= mx.topk_tiles_max;
}

/// topk_select's torch step, keys.topk(k).values: the top-k set of the unique int64 keys (ops.topkI64).
pub fn torchTopk(ctx: ?*anyopaque, _: tri.Tri, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) anyerror!void {
    const e: *const Engine = @ptrCast(@alignCast(ctx.?));
    try e.ops.topkI64(e.s, keys, ks, rows, n, k, top);
}

/// hd ** -0.5 as Triton passes the Python float: rounded to fp32.
pub fn scale(hd: usize) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
}

/// Comm.gather: [world, n, D] fp32 of every rank's partial, in rank order: the owner sums over the Rocinante transport at the owned
/// rows (gatherStep), the readers' tight buffer filled by the gather itself. Every rank holds the same chunk rows, so all take the
/// same path.
/// `consumer_li` is the layer index whose attnMixes/ffnMixes/tap/engramApply call will actually read the sums back (the SAME li for
/// the attention gather; li + 1 for the MoE gather, which the next layer reads as its "pending" post). ownedRange(consumer_li) is
/// exactly the range that reader will pull back out, so it keeps the producer and every consumer locked to the identical row range
/// by construction. `sub` names the partials (src: ch.pa or ch.pm).
pub fn gather(e: *const Engine, ch: *Chunk, src: u64, sub: Sub, consumer_li: usize) !void {
    const st = try gatherStep(e, ch, sub, consumer_li);
    const rc = e.two.?.rc orelse return error.RociNoShape;
    try ownerGatherRoci(e, ch, rc, src, st.owned, sub);
}

/// gather's host side: the owned rows of `sub`'s gather of a chunk of ch.n rows whose readers are layer `consumer_li`'s, recorded for
/// them (howOf).
pub fn gatherStep(e: *const Engine, ch: *Chunk, sub: Sub, consumer_li: usize) !GatherStep {
    const how = howOf(ch, sub);
    how.* = .{};
    if (e.two == null) return error.NotA2DSplit;
    const r = ownedRange(e, ch, consumer_li) orelse return error.NoOwnedRows;
    how.owned = true;
    return .{ .owned = r };
}

/// The owner path over the Rocinante transport (TF_DS_ROCI) at owned rows `r`: the partials in the arena's psum
/// go to the row partner as two blocks, one a rail (one stream write behind their producer); once its blocks have landed
/// in the stage, two adds make this pair's sums, this node's rows in its own block of `blk` and the column partner's
/// rows in `send2`, which goes to the column partner into its `blk`. The readers take the two blocks as they lie
/// (howOf.blocks): nothing is copied, widened or checked between a producer and its consumer.
fn ownerGatherRoci(e: *const Engine, ch: *Chunk, rc: *roci_engine.Roci, src: u64, r: QRows, sub: Sub) !void {
    _ = r;
    const t = e.two.?;
    const w = t.ow[t.p];
    if (src != rc.addr(.psum)) return error.RociNotInArena;
    const shape = try rociShape(e, ch, 0);
    howOf(ch, sub).blocks = true;
    if (ch.own1_sent) ch.own1_sent = false else try rc.send(e.s, .own1, shape);
    try rc.wait(e.s, .own1);
    try rociPart2(e, rc, src, ownQ(e, ch), ownRows(e, ch), w, shape, .stage, .send2, .blk, .own2);
}

/// The rows of the two blocks an owner sum's part 2 adds: this node's slab, then its column partner's (equal to ownQ unless a chunk's
/// last slab is short, TF_DS_ROCI_ANY).
fn ownRows(e: *const Engine, ch: *const Chunk) [2]usize {
    const t = e.two.?;
    const g: u32 = @intCast(t.r + 2 * t.p);
    return .{ slabN(e, ch, t.rank_of[g]), slabN(e, ch, t.rank_of[g ^ 2]) };
}

/// Part 2 of an owner sum on the arena: this node's partial at the plan's rows plus the stage's block, rounded to bf16
/// (add2Bf16: one fp32 add), k = 0 into this pair's block of `blk` (this node's rows), k = 1 into `send2` (the column partner's).
fn rociAdd(e: *const Engine, rc: *roci_engine.Roci, src: u64, q: usize, rows: usize, w: usize, k: usize, comptime stage: roci_engine.Region, comptime send2: roci_engine.Region, comptime blk: roci_engine.Region) !void {
    const t = e.two.?;
    const pl = prompt2d.ownerPlan(@intCast(t.r + 2 * t.p), t.rank_of, q);
    const out = if (k == 0) rc.addr(blk) + t.p * q * w * 2 else rc.addr(send2);
    if (rows == 0) return; // (a slab beyond the chunk's rows)
    try e.ops.add2Bf16(e.s, src + pl.recv1[k] * w * 4, rc.addr(stage) + k * q * w * 4, out, rows * w);
}

/// An owner sum's parts 2 and 3 on the arena once part 1's blocks have landed: the adds, the second part's send and wait.
/// With TF_DS_ROCI_P2_FIRST the add that makes the second part (k = 1) runs first and the send follows it at once, ahead of
/// the add of this node's own block (k = 0), which only the readers need: the same two adds on the same operands into the same
/// outputs (their order moves no bit: they read the partial and the stage, write disjoint buffers, and the second
/// part's source is no operand of the own block's add), and the send reads `send2` alone, which add k = 0 does not write.
fn rociPart2(e: *const Engine, rc: *roci_engine.Roci, src: u64, q: usize, rows: [2]usize, w: usize, shape: roci_engine.Shape, comptime stage: roci_engine.Region, comptime send2: roci_engine.Region, comptime blk: roci_engine.Region, comptime class: roci_engine.Class) !void {
    if (e.two.?.p2First()) {
        try rociAdd(e, rc, src, q, rows[1], w, 1, stage, send2, blk);
        try rc.send(e.s, class, shape);
        try rociAdd(e, rc, src, q, rows[0], w, 0, stage, send2, blk);
    } else {
        try rociAdd(e, rc, src, q, rows[0], w, 0, stage, send2, blk);
        try rociAdd(e, rc, src, q, rows[1], w, 1, stage, send2, blk);
        try rc.send(e.s, class, shape);
    }
    try rc.wait(e.s, class);
}

/// Engram's part 1 over the Rocinante transport: the projection's blocks to the row partner (one stream write behind
/// the projection, after the collect copy with TF_DS_ROCI_COLLECT).
fn engramSend1Roci(e: *const Engine, ch: *const Chunk, rc: *roci_engine.Roci, r: QRows) !void {
    _ = r;
    if (ch.ek != rc.addr(.ek)) return error.RociNotInArena;
    const shape = try rociShape(e, ch, 0);
    try rc.send(e.s, .eng1, shape);
}

/// Engram's owner sums over the Rocinante transport: `ek` (the projection's fp32 partials, in the arena) the
/// same way as ownerGatherRoci. The gate reads the two blocks of `eng_blk` where they lie.
fn engramSumsRoci(e: *const Engine, ch: *const Chunk, rc: *roci_engine.Roci, r: QRows, sent1: bool) !void {
    const t = e.two.?;
    const w = t.ew[t.p];
    if (!sent1) try engramSend1Roci(e, ch, rc, r);
    const shape = try rociShape(e, ch, 0);
    try rc.wait(e.s, .eng1);
    try rociPart2(e, rc, ch.ek, ownQ(e, ch), ownRows(e, ch), w, shape, .eng_stage, .eng_send2, .eng_blk, .eng2);
}

/// `sub`'s sums at owned rows `r` as a mix posts them into the streams: the transport's two blocks when the gather left
/// them there, else the tight buffer of fp32 planes.
fn ownedPosted(e: *const Engine, ch: *Chunk, sub: Sub, r: QRows, h_out: u64) !tri_hc.Posted {
    if (howOf(ch, sub).blocks) return .{ .gathered = e.two.?.rc.?.addr(.blk), .world = e.world, .h_out = h_out, .blk_rs = ownQ(e, ch) * e.two.?.ow[e.two.?.p] };
    const g = try ownedPartials(e, ch, sub, r);
    return .{ .gathered = g.ptr, .world = e.world, .h_out = h_out, .rs = g.rs };
}

/// hcPost of the pending gather `sub` at owned rows `r`, in place on the streams.
fn hcPostOwned(e: *const Engine, ch: *Chunk, sub: Sub, r: QRows) !void {
    if (r.n == 0) return;
    const c = e.c;
    const row_h = c.hc * c.hidden * 2;
    const row_p = c.hc * 4;
    const row_c = c.hc * c.hc * 4;
    if (howOf(ch, sub).blocks) {
        const t = e.two.?;
        return tri_basic.hcPost2b(e.t, t.rc.?.addr(.blk), ownQ(e, ch) * t.ow[t.p], ch.h + r.start * row_h, ch.post + r.start * row_p, ch.comb + r.start * row_c, ch.h + r.start * row_h, r.n, c.hidden);
    }
    const g = try ownedPartials(e, ch, sub, r);
    try tri_basic.hcPostAt(e.t, g.ptr, g.rs orelse r.n * c.hidden, ch.h + r.start * row_h, ch.post + r.start * row_p, ch.comb + r.start * row_c, ch.h + r.start * row_h, e.world, r.n, c.hidden);
}

/// Layer `li`'s FFN mixes: hc_pre_pf with the gathered attention partials posted into the streams first; x the MoE's
/// input rows, pre_f / post / comb the FFN's.
pub fn ffnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    if (ch.x_g != 0) ch.x = ch.x_dev; // (TF_DS_ROCI_GATHERS: the MoE reads the device x; attention's was the arena's on a gathered layer)
    if (ownedRange(e, ch, li)) |r| {
        const row_h = c.hc * c.hidden * 2;
        const row_p = c.hc * 4;
        const row_c = c.hc * c.hc * 4;
        if (r.n != 0) {
            const posted = try ownedPosted(e, ch, .attn, r, ch.h_alt + r.start * row_h);
            _ = try tri_hc.hcPrePf(e.t, ch.h + r.start * row_h, lay.hc_ffn[0], lay.hc_ffn[1], lay.hc_ffn[2], ch.pre_a + r.start * row_p, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x + r.start * c.hidden * 2, ch.pre_f + r.start * row_p, ch.post + r.start * row_p, ch.comb + r.start * row_c, ch.qhc_part, posted, r.n, c.hidden);
        }
        const old_h = ch.h;
        ch.h = ch.h_alt;
        ch.h_alt = old_h;
        // the owned routing (ffnRange) reads x's own rows alone
        if (ffnRange(e, ch, li) == null) return error.NotNative; // (the owned routing is the only one: the start requires the checkpoint's own weights)
        return;
    }
    const src = try tri_hc.hcPrePf(e.t, ch.h, lay.hc_ffn[0], lay.hc_ffn[1], lay.hc_ffn[2], ch.pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_f, ch.post, ch.comb, ch.part, .{ .gathered = ch.ga, .world = e.world, .h_out = ch.h_alt }, ch.n, c.hidden);
    swapStreams(ch, src);
}

/// Whether layer li's MoE partials take TF_DS_ROCI_COMBINE_FIRST (or TF_DS_ROCI_COMBINE_DOOR, whose one launch rings the doorbell itself):
/// it is in effect, the transport carries the partials' owner sums (gatherStep's owner path at the owned rows of the layer that reads
/// them, whole pass quarters) and the partials are not collected into the arena by a copy that would have to follow the whole sum.
fn fixedPlan(ch: *const Chunk) bool {
    _ = roci_engine.Roci.shapeOf(ch.cap) catch return false;
    return true;
}

fn moeSendFirst(e: *const Engine, ch: *const Chunk, li: usize) bool {
    const t = e.two orelse return false;
    if (!t.combineDoor() or t.rc == null) return false;
    const r = ownedRange(e, ch, li + 1) orelse return false;
    // (a whole pass the transport has a fixed plan for: the kernels' quarters are q = cap / world rows, the same on every rank; under
    // TF_DS_ROCI_ANY a chunk of fewer rows takes the plain path, whatever its slab holds on this rank)
    const whole_quarters = r.n * e.comm.world == ch.cap and ch.n == ch.cap and fixedPlan(ch);
    if (!whole_quarters) return false;
    if (t.combineDoor()) {
        const k = e.mx orelse return false;
        const q = ch.cap / e.comm.world;
        const pl = prompt2d.ownerPlan(@intCast(t.r + 2 * t.p), t.rank_of, q);
        return k.combine_door != null and ch.n == ch.cap and mx.quarterOrder(pl.send1, pl.recv1, q, e.w.layers[li].experts.down_n) != null;
    }
    return true;
}

/// The part-1 send of the MoE's owner sum, queued from inside moe once the rows it sends are summed.
const MoeRing = struct {
    e: *const Engine,
    ch: *Chunk,

    fn send(ctx: *anyopaque) anyerror!void {
        const self: *MoeRing = @ptrCast(@alignCast(ctx));
        const t = self.e.two.?;
        try t.rc.?.send(self.e.s, .own1, try roci_engine.Roci.shapeOf(self.ch.cap));
        self.ch.own1_sent = true;
    }

    /// TF_DS_ROCI_SENT_FIRST: the ticket of the part-1 send, reserved right before the launch that rings it.
    fn door(ctx: *anyopaque) anyerror!mx.Door {
        const self: *MoeRing = @ptrCast(@alignCast(ctx));
        const rc = self.e.two.?.rc.?;
        const rs = try rc.reserve(.own1, try roci_engine.Roci.shapeOf(self.ch.cap));
        self.ch.own1_sent = true;
        return .{ .cnt = rs.cnt, .door = rs.door, .ticket = rs.ticket };
    }
};

/// The routed sum's two stages for the owner sums over `q`-row quarters: the quarters part 1 sends (the row partner's and its
/// column partner's rows, ownerPlan's send1) first, the quarters this node keeps (recv1) last.
fn moeCombineFirst(e: *const Engine, q: usize, ring: *MoeRing) mx.CombineFirst {
    const t = e.two.?;
    const pl = prompt2d.ownerPlan(@intCast(t.r + 2 * t.p), t.rank_of, q);
    return .{ .rows = q, .first = pl.send1, .last = pl.recv1, .ctx = ring, .between = MoeRing.send, .door = if (t.combineDoor()) MoeRing.door else null };
}

/// Whether layer li's attention partial takes TF_DS_ROCI_SENT_FIRST's wo_b: gatherStep's owner path on the arena at the layer's owned
/// rows, a full pass.
fn attnSentFirst(e: *const Engine, ch: *const Chunk, li: usize) bool {
    const t = e.two orelse return false;
    const rc = t.rc orelse return false;
    if (!t.sentFirst() or ch.n != ch.cap or ch.pa != rc.addr(.psum) or !fixedPlan(ch)) return false;
    const r = ownedRange(e, ch, li) orelse return false;
    return r.n * e.comm.world == ch.cap;
}

/// wo_b's GEMM (`l` on x, the bf16 rows of width ldx) into the arena's psum in one launch that runs the row quarters part 1 of the owner
/// sum sends first and rings that part's doorbell from inside, once their tiles are stored (mx.linearSentFirst): the host reserves the
/// ticket and queues the launch directly behind the quantization, so the proxies see tickets in the order the stream reaches them.
/// The gather that follows sends nothing for the part (own1_sent). False, with nothing launched or reserved, when the kernels do not
/// take the shape (mx.sentFirstTakes).
pub fn mmSentFirst(e: *const Engine, ch: *Chunk, t: *const prompt2d.Two, l: weights.Linear, x: u64, ldx: usize, out: u64) !bool {
    const k = e.mx orelse return false;
    const rc = t.rc orelse return false;
    if (!mx.sentFirstTakes(k, l, ch.mxs.q, l.k, ch.mxs.s, l.k / 32, out, l.n, ch.n)) return false;
    const q = ch.cap / e.comm.world;
    const pl = prompt2d.ownerPlan(@intCast(t.r + 2 * t.p), t.rank_of, q);
    const ro = mx.rowOrder(pl.send1, pl.recv1, q, l.n / 128) orelse return false;
    const shape = try roci_engine.Roci.shapeOf(ch.cap);
    const rs = try rc.reserve(.own1, shape);
    try mx.linearSentFirst(k, e.s, ch.mxs, l, x, .bf16, ldx, ch.n, out, l.n, ro, .{ .cnt = rs.cnt, .door = rs.door, .ticket = rs.ticket });
    ch.own1_sent = true;
    return true;
}

/// mmSentFirst's launch on the codes TF_DS_ROCI_CODES landed: wo_b's GEMM (`l` on the input codes `q` [n, k] and scales `s`
/// [n, k / 32], both pairs' columns in place) into the arena's psum in the one launch that runs the row quarters part 1 of the
/// owner sum sends first and rings that part's doorbell itself. The host reserves the ticket and queues the launch directly behind
/// the copies, as mmSentFirst does. False, with nothing launched or reserved, when the kernels do not take the shape.
pub fn codesGemmSentFirst(e: *const Engine, ch: *Chunk, t: *const prompt2d.Two, l: weights.Linear, q: u64, s: u64) !bool {
    const k = e.mx orelse return false;
    const rc = t.rc orelse return false;
    const out = ch.pa;
    if (!mx.sentFirstTakes(k, l, q, l.k, s, l.k / 32, out, l.n, ch.n)) return false;
    const qq = ch.cap / e.comm.world;
    const pl = prompt2d.ownerPlan(@intCast(t.r + 2 * t.p), t.rank_of, qq);
    const ro = mx.rowOrder(pl.send1, pl.recv1, qq, l.n / 128) orelse return false;
    const shape = try roci_engine.Roci.shapeOf(ch.cap);
    const rs = try rc.reserve(.own1, shape);
    try mx.gemmSentFirst(k, e.s, l, q, l.k, s, l.k / 32, ch.n, out, l.n, ch.mxs.z, ro, .{ .cnt = rs.cnt, .door = rs.door, .ticket = rs.ticket });
    ch.own1_sent = true;
    return true;
}

/// Model.moe of a prompt chunk: the gate's logits in fp32 (x.float() @ gate_w.float().t(), one cuBLAS GEMM as torch
/// issues it), the routing (the shared expert in every row's last slot) and the routed experts: the partial pm [n, D].
pub fn moe(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const d = c.hidden;
    const sl = e.slots();
    const shared_id = lay.experts.count - 1;
    var kc: usize = 0;
    if (e.mx) |k| {
        // Format.native: Gate.forward's fp32 logits of the bf16 rows and weight (exact products, fp32 sums), any rows;
        // TF_DS_FFN_CODES (ffnRange): on owned rows with x's codes, then gathered (ffnOwned)
        const own = ffnRange(e, ch, li);
        if (own) |r| {
            try ffnOwned(e, ch, lay, r, shared_id, sl);
        } else {
            try mx.bf16Gemm(k, e.s, ch.x, d, lay.gate_w, d, n, c.experts, d, ch.logits, c.experts, .f32, 1, false, 1, 0, 0, 0, null);
            try tri_norm.route(e.t, ch.logits, 0, lay.gate_b, c.top_k, c.routed_scaling, shared_id, ch.pick, ch.wts, n, c.experts, sl);
            try routeSpans(e, ch, lay, shared_id, sl, 0);
        }
        pmark(e, .m_rt);
        dump(e, li, "mx", ch.x, n * d * 2);
        dump(e, li, "logits", ch.logits, n * c.experts * 4);
        dump(e, li, "pick", ch.pick, n * sl * 4);
        dump(e, li, "wts", ch.wts, n * sl * 4);
        if (e.two) |t| {
            var ring: MoeRing = undefined;
            var first: mx.CombineFirst = undefined;
            const cf: ?*const mx.CombineFirst = if (moeSendFirst(e, ch, li)) blk: {
                ring = .{ .e = e, .ch = ch };
                first = moeCombineFirst(e, ch.cap / e.comm.world, &ring);
                break :blk &first;
            } else null;
            try t.experts(e, ch, lay, own != null, cf);
            const pack = lay.experts.width + t.ir;
            dump(e, li, "xd", ch.xs.xd, n * sl * pack * 2);
            dump(e, li, "xd_full", t.xd_full, n * sl * lay.experts.down_k * 2);
            dump(e, li, "pm", ch.pm, n * lay.experts.down_n * 4);
            return;
        }
        try mx.gateUp(k, e.s, e.d, lay.experts, ch.mxe, ch.x, d, ch.pick, ch.wts, n, c.swiglu_limit, ch.xs.xd, lay.experts.width, false);
        return mx.down(k, e.s, lay.experts, ch.mxe, ch.xs.xd, lay.experts.width, ch.pick, n, ch.pm, d, false, false);
    }
    if (n <= tri.decode_rows) {
        // a decode-sized chunk: the row-invariant gate (rowmm_gate's chunk sums) and its routing
        kc = try tri_norm.rowmmGate(e.t, ch.x, d, lay.gate_w, ch.gl, n, d, c.experts);
        try tri_norm.route(e.t, ch.gl, kc, lay.gate_b, c.top_k, c.routed_scaling, shared_id, ch.pick, ch.wts, n, c.experts, sl);
    } else {
        try e.ops.toF32(e.s, ch.x, ch.xf, n * d);
        try e.ops.f16ToF32(e.s, lay.gate_w, ch.gate_f, c.experts * d);
        try e.blas.xwT(ch.xf, ch.gate_f, ch.logits, n, d, c.experts);
        try tri_norm.route(e.t, ch.logits, 0, lay.gate_b, c.top_k, c.routed_scaling, shared_id, ch.pick, ch.wts, n, c.experts, sl);
    }
    try routeSpans(e, ch, lay, shared_id, sl, kc);
    if (e.two) |t| return t.experts(e, ch, lay, false, null); // 2D: gate / up here, the intermediate's exchange, down here
    // routed(): a chunk of fewer than EXACT_ROWS rows takes the decode window's fused path and scratch
    if (n < exl3_experts.exact_rows) return exl3_experts.decode(e.ex, e.s, lay.experts, ch.xsd, ch.x, d, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
    try exl3_experts.prompt(e.ex, e.s, lay.experts, ch.xs, ch.x, d, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
}

/// The VL routing (Gate.forward's bias = where(image_mask, bias_vl, bias)) as the served build runs it (recording v0's
/// launches): every row routed with the text bias, then each span's rows of the gate's output (the logits, or a
/// decode-sized chunk's chunk sums [n, kc, NE]) routed again with the image bias in a launch of their own, their picks and
/// weights put over the text ones. Row by row the routing is independent, so the text rows keep theirs.
fn routeSpans(e: *const Engine, ch: *Chunk, lay: weights.Layer, shared_id: usize, sl: usize, kc: usize) !void {
    return routeSpansIn(e, ch, lay, shared_id, sl, kc, .{ .start = 0, .n = ch.n });
}

/// routeSpans over the chunk's rows `within` alone: each span's overlap with them (rowsIn), routed and put as
/// routeSpans routes and puts it. The owned routing's (ffnOwned): a node routes its owned rows' span rows.
fn routeSpansIn(e: *const Engine, ch: *Chunk, lay: weights.Layer, shared_id: usize, sl: usize, kc: usize, within: QRows) !void {
    if (ch.spans.len == 0) return;
    if (lay.gate_b_vl == 0) return error.NoImageRoutingBias;
    const c = e.c;
    const n = ch.n;
    const src = if (n <= tri.decode_rows and kc != 0) ch.gl else ch.logits;
    const row = @max(kc, 1) * c.experts * 4;
    for (ch.spans) |sp| {
        const o = rowsIn(overlap(sp, ch.start, n) orelse continue, within) orelse continue;
        const rows = o[1] - o[0];
        try tri_norm.route(e.t, src + o[0] * row, kc, lay.gate_b_vl, c.top_k, c.routed_scaling, shared_id, ch.pick_vl, ch.wts_vl, rows, c.experts, sl);
        const dd = e.d;
        try dd.check(dd.api.cuMemcpyDtoDAsync_v2(ch.pick + o[0] * sl * 4, ch.pick_vl, rows * sl * 4, e.s.handle), "cuMemcpyDtoDAsync");
        try dd.check(dd.api.cuMemcpyDtoDAsync_v2(ch.wts + o[0] * sl * 4, ch.wts_vl, rows * sl * 4, e.s.handle), "cuMemcpyDtoDAsync");
    }
}

/// A span's chunk rows `o` [lo, hi) within `rows` (null: none).
pub fn rowsIn(o: [2]usize, rows: QRows) ?[2]usize {
    const lo = @max(o[0], rows.start);
    const hi = @min(o[1], rows.start + rows.n);
    return if (lo < hi) .{ lo, hi } else null;
}

/// TF_DS_FFN_CODES's routing at owned rows `r` (ffnRange): ffnRows, the image spans' rows within r routed with their
/// bias (routeSpansIn), then every rank's rows of x's codes and scales, the picks and the weights all-gathered in
/// place over the transport (rociGather). The MoE then reads the bytes the baseline routing and group() leave (every kernel
/// row-wise, the gathers move bytes); x and the logits hold rows r alone.
fn ffnOwned(e: *const Engine, ch: *Chunk, lay: weights.Layer, r: QRows, shared_id: usize, sl: usize) !void {
    const c = e.c;
    if (r.n != 0) {
        try ffnRows(e.mx.?, e.t, e.s, lay, ch.x, c.hidden, r, c.experts, c.top_k, c.routed_scaling, shared_id, sl, ch.logits, ch.pick, ch.wts, ch.mxe);
        try routeSpansIn(e, ch, lay, shared_id, sl, 0, r);
    }
    return rociGather(e, .gm, try needShape(e, ch));
}

/// The owned routing's kernels at rows `r` of x [*, ldx] bf16: the router's fp32 logits (moe's bf16 GEMM, K whole, a
/// row's bits at any row count) into logits' rows r [*, ne], the routing (ffnRoute) into pick's and wts's rows r
/// [*, sl], and x's rows quantized into sc.q / sc.s rows r [*, dims] (group()'s act_quant).
pub fn ffnRows(k: *const mx.Kernels, t: tri.Tri, s: cuda.Stream, lay: weights.Layer, x: u64, ldx: usize, r: QRows, ne: usize, top_k: usize, scale_: f32, shared_id: usize, sl: usize, logits: u64, pick: u64, wts: u64, sc: mx.ExpertScratch) !void {
    const dd: usize = lay.experts.dims;
    if (r.start + r.n > sc.rows) return error.ScratchTooSmall;
    const xr = x + r.start * ldx * 2;
    try mx.bf16Gemm(k, s, xr, ldx, lay.gate_w, ldx, r.n, ne, ldx, logits + r.start * ne * 4, ne, .f32, 1, false, 1, 0, 0, 0, null);
    try ffnRoute(t, lay, logits, pick, wts, r, ne, top_k, scale_, shared_id, sl);
    try mx.quant(k, s, xr, .bf16, ldx, r.n, dd, sc.q + r.start * dd, dd, sc.s + r.start * (dd / 32) * 4, dd / 32);
}

/// The owned routing's text-bias routing: logits' rows r [*, ne] into pick's and wts's rows r [*, sl] (one _route
/// program a row, the whole chunk's launch's arguments at the rows' offsets).
pub fn ffnRoute(t: tri.Tri, lay: weights.Layer, logits: u64, pick: u64, wts: u64, r: QRows, ne: usize, top_k: usize, scale_: f32, shared_id: usize, sl: usize) !void {
    try tri_norm.route(t, logits + r.start * ne * 4, 0, lay.gate_b, top_k, scale_, shared_id, pick + r.start * sl * 4, wts + r.start * sl * 4, r.n, ne, sl);
}

/// TF_DS_FFN_CODES's start check on this rank (null: it can run; else the reason, for Two.agreeFfnCodes): quarter-HC,
/// the checkpoint's own weights (`native`), and every routing launch ffnRoute makes at each of the `world` ranks'
/// owned rows of each row count in `qs`, for every layer below `half`, found in this rank's kernel set
/// (`finds.has`, as frontOwnedOff's).
pub fn ffnCodesOff(gpa: std.mem.Allocator, finds: anytype, native: bool, layers: []const weights.Layer, half: usize, ne: usize, top_k: usize, scale_: f32, sl: usize, qs: []const usize, world: usize) !?[]const u8 {
    const rows = try ownedRows(gpa, qs, world);
    defer gpa.free(rows);
    return ffnCodesOffAt(gpa, finds, native, layers, half, ne, top_k, scale_, sl, rows);
}

pub fn ffnCodesOffAt(gpa: std.mem.Allocator, finds: anytype, native: bool, layers: []const weights.Layer, half: usize, ne: usize, top_k: usize, scale_: f32, sl: usize, rows: []const QRows) !?[]const u8 {
    if (!native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    var log: tri.Log = .{ .gpa = gpa };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    // logits, pick and wts as the arena aligns a chunk's buffers (256 bytes)
    const logits: u64 = 1 << 32;
    const pick: u64 = 2 << 32;
    const wts: u64 = 3 << 32;
    for (layers[0..@min(half, layers.len)]) |lay| {
        for (rows) |r| try ffnRoute(t, lay, logits, pick, wts, r, ne, top_k, scale_, ne, sl);
    }
    for (log.items.items) |l| if (!l.foundIn(finds)) return "this rank's kernel set has no _route variant for the owned rows";
    return null;
}

/// Engram on the host: the hasher (the compressed token map, the multipliers), this rank's hash columns [lo, hi) of
/// each Engram layer's, its tables and readers, and host buffers for a chunk's rows (model.py's Engram).
/// The worker's slots whose rows are still being copied to the device, by Engram layer: a slot is armed with the `used`
/// count it frees (wk_used) once its copy's event is done, and freed in that order. At most two are pending a layer (the
/// worker holds two slots).
pub const PinSlots = struct {
    live: [2][engram_rows.max_wk]bool = @splat(@splat(false)),
    used: [2][engram_rows.max_wk]usize = @splat(@splat(0)),
    ring: usize = 2, // the worker's chunk slots (engram_rows.workerSlots): the pending copies a layer at most
    next: [2]usize = .{ 0, 0 }, // the ring position of the next arm, by layer

    /// The ring position the next arm of layer j takes, or null when both are pending (the caller waits for the oldest).
    pub fn room(s: *const PinSlots, j: usize) ?usize {
        const p = s.next[j] % s.ring;
        return if (s.live[j][p]) null else p;
    }

    /// Layer j's slot `p` (room's) holds a copy that frees `used` slots of the worker when done.
    pub fn arm(s: *PinSlots, j: usize, p: usize, used: usize) void {
        std.debug.assert(!s.live[j][p]);
        s.live[j][p] = true;
        s.used[j][p] = used;
        s.next[j] += 1;
    }

    /// The largest `used` pending for layer j (0: none), over the whole ring.
    pub fn highest(s: *const PinSlots, j: usize) usize {
        var u: usize = 0;
        for (0..s.ring) |p| {
            if (s.live[j][p]) u = @max(u, s.used[j][p]);
        }
        return u;
    }

    /// The pending position of layer j with the fewest `used` (the oldest), or null.
    pub fn oldest(s: *const PinSlots, j: usize) ?usize {
        var best: ?usize = null;
        for (0..s.ring) |p| {
            if (s.live[j][p] and (best == null or s.used[j][p] < s.used[j][best.?])) best = p;
        }
        return best;
    }

    /// Frees, oldest first, layer j's slots whose copies are done (`ctx.done(j, p)`, then `ctx.release(j, used)`); true while
    /// one is still pending.
    pub fn poll(s: *PinSlots, j: usize, ctx: anytype) !bool {
        while (s.oldest(j)) |p| {
            if (!try ctx.done(j, p)) return true;
            ctx.release(j, s.used[j][p]);
            s.live[j][p] = false;
        }
        return false;
    }
};

pub const EngramHost = struct {
    hasher: engram.Hasher,
    tables: *const engram_io.Tables,
    pool: *engram_io.Pool,
    lo: usize,
    hi: usize,
    hashes: []i64, // [cap][layers][cols]
    flat: []i64, // [cap * (hi - lo)]
    w: []u8, // FP8 rows
    s: []u8, // their E8M0 scales
    rows: []u16, // bf16 [cap, (hi - lo) * head_dim]
    aio: ?*engram_aio.Aio = null, // a round's reads by Linux AIO on the tables' O_DIRECT descriptors (else the pool)
    pk: ?*engram_rows.Rows = null, // TF_DS_ENGRAM_PACKED: rows staged as packed records and decoded on the GPU
    paio: ?*engram_aio.Aio = null, // a prompt chunk's reads the same way, on a ring of its own (else the pool)
    // the rounds' Engram host time (ns, summed; a profile reads it): hashing, the table reads, the decode, the upload
    io: ?std.Io = null,
    t_hash: u64 = 0,
    t_read: u64 = 0,
    t_decode: u64 = 0,
    t_upload: u64 = 0,
    calls: u64 = 0,
    // a prompt chunk's rows read ahead on a thread (engramReadAhead): Engram layer j's rows of the chunk at
    // [ahead_start, ahead_start + ahead_n), ready when ahead_ready[j]
    ahead_w: [2][]u8 = .{ &.{}, &.{} },
    ahead_s: [2][]u8 = .{ &.{}, &.{} },
    ahead_rows: [2][]u16 = .{ &.{}, &.{} },
    ahead_flat: [2][]i64 = .{ &.{}, &.{} },
    ahead_ready: [2]bool = .{ false, false },
    ahead_start: usize = 0,
    ahead_n: usize = 0,
    ahead_err: ?anyerror = null,
    // a profile's prompt Engram split (ns summed, the stream synchronized at each cut): the rows ready on the host,
    // their upload, the projection, the quarters' exchange, the sum and gate
    p_rows: u64 = 0,
    p_upload: u64 = 0,
    p_mm: u64 = 0,
    p_x: u64 = 0,
    p_gate: u64 = 0,
    p_calls: u64 = 0,
    // the prompt's read-ahead worker (engramWorker; model.zig runs it through a fill): chunk i's rows of Engram layer j
    // in wk_rows[i % 2][j], chunk i ready once wk_done[j] > i, its slot free again once wk_used[j] > i (the rows
    // uploaded), so the worker runs up to a chunk ahead of the layers: a chunk's table reads overlap the chunk before
    wk_rows: [engram_rows.max_wk][2][]u16 = @splat(.{ &.{}, &.{} }),
    // the chunk slots in use (engram_rows.workerSlots: a pass's resume units, or twice them for two passes in flight, at least 2)
    wk_slots: usize = 2,
    // the worker's row ids by chunk parity: a pass-major read leaves one chunk's layer-1 rows to read after the next
    // chunk's layer 0 has hashed its ids, so each slot keeps its own (the chunk-major order read them back to back)
    wk_hashes: [engram_rows.max_wk][]i64 = @splat(&.{}),
    wk_flat: []i64 = &.{},
    wk_w: []u8 = &.{},
    wk_s: []u8 = &.{},
    wk_done: [2]std.atomic.Value(usize) = .{ .init(0), .init(0) },
    wk_used: [2]std.atomic.Value(usize) = .{ .init(0), .init(0) },
    // TF_DS_ENGRAM_PINNED: the worker's slots in pinned host memory, uploaded without the host waiting for the stream; a slot
    // goes back to the worker (wk_used) when the event recorded behind its copy is done (PinSlots)
    pinned: bool = false,
    pin_ev: [2][engram_rows.max_wk]?cuda.Event = @splat(@splat(null)),
    pin: PinSlots = .{},
    pin_host: [engram_rows.max_wk][2]?cuda.HostBuffer = @splat(.{ null, null }), // the unpacked rows' pinned slots
    // the engine thread's time in the pass (TF_DS_ROCI_PAIR_LOG's line): waiting for the worker's rows, and in the upload calls (a blocked copy is here)
    inst_wait_ns: u64 = 0,
    ka: keep_ahead.Stats = .{}, // TF_DS_ENGRAM_KEEP_AHEAD: the worker's starts this prompt and the first layer's wait after a round (the engine thread alone)
    inst_apply_ns: u64 = 0,
    // the worker's own time and reads (its thread adds, the engine thread's pair line reads the difference from `inst_snap`): steps, hashing,
    // the rows' gather (the packed path's cache lookups and reads), decode on the host, rows asked, rows read (cache misses), bytes read
    inst_steps: std.atomic.Value(u64) = .init(0),
    inst_hash_ns: std.atomic.Value(u64) = .init(0),
    inst_gather_ns: std.atomic.Value(u64) = .init(0),
    inst_decode_ns: std.atomic.Value(u64) = .init(0),
    inst_rows: std.atomic.Value(u64) = .init(0),
    inst_miss: std.atomic.Value(u64) = .init(0),
    inst_bytes: std.atomic.Value(u64) = .init(0),
    inst_snap: [23]u64 = @splat(0),
    // where the worker's time went (its thread adds; the pair line reads the differences): its CPU time, its involuntary context
    // switches, and the cores it was on at each step's start and end, as counts on the GB10's big cores (X925: 5-9, 15-19) and small
    // ones (A725: 0-4, 10-14), and a mask of the cores seen since the last line
    inst_cpu_ns: std.atomic.Value(u64) = .init(0),
    inst_nivcsw: std.atomic.Value(u64) = .init(0),
    inst_big: std.atomic.Value(u64) = .init(0),
    inst_small: std.atomic.Value(u64) = .init(0),
    inst_cores: std.atomic.Value(u64) = .init(0),
    // sleeping waits on the counters above and the flags below (wait_bell): the engine thread rings `wk_go` after it gives a slot back or
    // stops the worker, the worker rings `wk_ready` after a chunk is done or it failed
    wk_go: wait_bell.Bell = .{},
    wk_ready: wait_bell.Bell = .{},
    wk_stop: std.atomic.Value(bool) = .init(false),
    wk_failed: std.atomic.Value(bool) = .init(false),
    wk_err: ?anyerror = null,
    wk_thread: ?std.Thread = null,
    wk_seq: []const i32 = &.{}, // the prompt it reads (its whole ids)
    wk_start: usize = 0, // its first chunk's start
    wk_ahead: bool = false, // TF_DS_ENGRAM_AHEAD_PASS in effect for this worker
    wk_group: usize = 0, // TF_DS_ROCI_WORKER_PAIR_MAJOR with two passes in flight: 4 chunks a group (WorkerOrder.group)
    wk_pass_from: usize = 0, // the pass rule's pass_from (prompt2d.pass_from_asked)
    wk_pass_rows: usize = 0, // the pass rule's pass_rows cap (Model.pass_rows)
    // engramApply's rows when the worker read them: the pass's resume units in order (use_n of them, use_unit rows each
    // but the last), or their records instead with packed shards (engramWorkerStage)
    use_rows: [engram_rows.max_units]?[]u16 = @splat(null),
    use_stage: [engram_rows.max_units]?*engram_rows.Stage = @splat(null),
    use_n: usize = 0,
    use_unit: usize = 0,

    pub fn init(gpa: std.mem.Allocator, c: *const Config, hasher: engram.Hasher, tables: *const engram_io.Tables, pool: *engram_io.Pool, rank: usize, world: usize, cap: usize, unit: usize, wk_slots: usize) !EngramHost {
        var why: [320]u8 = undefined;
        if (engram_rows.slotsRefusal(&why, wk_slots, @max(1, cap / unit))) |w| {
            std.log.err("{s}", .{w});
            return error.BadSlots;
        }
        const cols = hasher.cols();
        const lo = rank * cols / world;
        const hi = (rank + 1) * cols / world;
        const k = hi - lo;
        const t = tables.layers.get(@intCast(c.engram_layers.slice()[0])) orelse return error.MissingEngramTable;
        var wr: [engram_rows.max_wk][2][]u16 = @splat(.{ &.{}, &.{} });
        var max_w: usize = 0;
        var max_s: usize = 0;
        for (0..@min(2, c.engram_layers.slice().len)) |j| {
            const tj = tables.layers.get(@intCast(c.engram_layers.slice()[j])) orelse return error.MissingEngramTable;
            for (0..wk_slots) |slot| wr[slot][j] = try gpa.alloc(u16, unit * k * tj.row_w);
            max_w = @max(max_w, tj.row_w);
            max_s = @max(max_s, tj.row_s);
        }
        var aw: [2][]u8 = .{ &.{}, &.{} };
        var as: [2][]u8 = .{ &.{}, &.{} };
        var ar: [2][]u16 = .{ &.{}, &.{} };
        var af: [2][]i64 = .{ &.{}, &.{} };
        for (0..@min(2, c.engram_layers.slice().len)) |j| {
            const tj = tables.layers.get(@intCast(c.engram_layers.slice()[j])) orelse return error.MissingEngramTable;
            aw[j] = try gpa.alloc(u8, cap * k * tj.row_w);
            as[j] = try gpa.alloc(u8, cap * k * tj.row_s);
            ar[j] = try gpa.alloc(u16, cap * k * tj.row_w);
            af[j] = try gpa.alloc(i64, cap * k);
        }
        return .{
            .wk_rows = wr,
            .wk_slots = wk_slots,
            .wk_hashes = blk: {
                var h: [engram_rows.max_wk][]i64 = @splat(&.{});
                for (0..wk_slots) |i| h[i] = try gpa.alloc(i64, unit * hasher.layers * cols);
                break :blk h;
            },
            .wk_flat = try gpa.alloc(i64, unit * k),
            .wk_w = try gpa.alloc(u8, unit * k * max_w),
            .wk_s = try gpa.alloc(u8, unit * k * max_s),
            .ahead_w = aw,
            .ahead_s = as,
            .ahead_rows = ar,
            .ahead_flat = af,
            .hasher = hasher,
            .tables = tables,
            .pool = pool,
            .lo = lo,
            .hi = hi,
            .hashes = try gpa.alloc(i64, cap * hasher.layers * cols),
            .flat = try gpa.alloc(i64, cap * k),
            .w = try gpa.alloc(u8, cap * k * t.row_w),
            .s = try gpa.alloc(u8, cap * k * t.row_s),
            .rows = try gpa.alloc(u16, cap * k * t.row_w),
        };
    }
};

/// A prompt chunk's Engram layer j rows ahead of its layer, on a thread of the caller's while the GPU runs the layers
/// before it: the n-grams hashed (j 0: every layer's at once), the rows read and decoded into ahead_rows[j], exactly
/// as engramApply would at the layer. Errors land in ahead_err.
pub fn engramReadAhead(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize) void {
    engramAhead1(eh, c, seq, start, n, j) catch |err| {
        eh.ahead_err = err;
    };
}

/// The next chunk's rows of both Engram layers (fillChunk starts it once this chunk's last Engram layer is applied).
pub fn engramReadAheadAll(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize) void {
    for (0..@min(2, c.engram_layers.slice().len)) |j| {
        engramAhead1(eh, c, seq, start, n, j) catch |err| {
            eh.ahead_err = err;
            return;
        };
    }
}

fn engramAhead1(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize) !void {
    const li = c.engram_layers.slice()[j];
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    if (j == 0) eh.hasher.hashes(seq, start, n, eh.hashes[0 .. n * eh.hasher.layers * cols]);
    if (eh.pk) |pr| {
        // packed shards: the records staged in this layer's read-ahead buffer (decoded on the GPU when applied)
        try pr.chunkRead(&pr.ahead[j], j, eh.hashes[0 .. n * eh.hasher.layers * cols], n, eh.hasher.layers);
        eh.ahead_ready[j] = true;
        return;
    }
    for (0..n) |r| {
        for (0..k) |q| eh.ahead_flat[j][r * k + q] = eh.hashes[(r * eh.hasher.layers + j) * cols + eh.lo + q];
    }
    const m = n * k;
    // by AIO on the prompt's ring when it is open (the main thread reads by it only after joining this one)
    if (eh.paio) |aio| try aioGather(aio, t, eh.ahead_flat[j][0..m], eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s]) else try eh.pool.gather(t, eh.ahead_flat[j][0..m], eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s]);
    engram.decodeRows(eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s], eh.ahead_rows[j][0 .. m * t.row_w], m, t.row_w, t.row_s, 8);
    eh.ahead_ready[j] = true;
}

/// TF_DS_ENGRAM_AHEAD_PASS=1: a fill's Engram worker schedules a prompt pass's chunks a layer at a time, so a 4,096-row
/// pass's second chunk has its layer's rows read with the first chunk's and not behind the first chunk's later layer.
/// Unset (the default): the worker keeps the baseline chunk-major order, every layer of a chunk before the next chunk.
pub var ahead_pass_on: bool = false;
pub const ahead_pass_line = "TF_DS_ENGRAM_AHEAD_PASS=1: the Engram worker reads a prompt pass's chunks a layer at a time (both chunks of a 4,096-row pass ready when it starts; a 2,048-row pass keeps the chunk order)";

/// Read TF_DS_ENGRAM_AHEAD_PASS once at start (model.readWorkerEnv, from Model.open) and print its start line once.
pub fn readAheadPassEnv() void {
    if (std.c.getenv("TF_DS_ENGRAM_AHEAD_PASS")) |v| ahead_pass_on = !std.mem.eql(u8, std.mem.span(v), "0");
    if (ahead_pass_on) std.log.info("{s}", .{ahead_pass_line});
}

/// The pass rule's rows at a fill's chunk `at`, from the host's quiet facts: while this worker reads, nothing decodes
/// and no other prompt fills; a round that interrupts the fill joins the worker, so the next fill starts it afresh.
/// `filling_pass` is the rule's filling switch, which changes nothing under these facts.
fn workerRuleRows(at: usize, pass_from: usize, pass_rows: usize, chunk: usize, filling_pass: bool) usize {
    const quiet: lanes.stream.Facts = .{ .decoding = false, .filling = false, .waiting = false, .step_rows = std.math.maxInt(u32) };
    return lanes.stream.passRows(quiet, at, pass_from, @intCast(pass_rows), @intCast(chunk), filling_pass);
}

/// One worker read: a chunk of `chunk` rows and one Engram layer, exactly the workerRows call of the chunk-major loop.
pub const WorkerStep = struct { chunk: usize, layer: usize };

/// The order the fill's Engram worker reads its chunks: the baseline chunk-major order, or with TF_DS_ENGRAM_AHEAD_PASS each
/// pass's chunks a layer at a time, the pass's size from the pass rule at that chunk's position (stream.passRows).
pub const WorkerOrder = struct {
    start: usize, // the fill's first chunk start, the pass rule's absolute position
    chunks: usize,
    chunk: usize,
    layers: usize,
    pass_from: usize,
    pass_rows: usize, // the pass rule's cap (<= chunk: the baseline chunk-major order)
    filling_pass: bool = false, // the pass rule's filling switch (TF_DS_PASS_FILLING)
    group: usize = 0, // chunks a pass-major group takes from pass_from on, whatever the pass rule says (0: the rule): two passes in flight read both passes' layer 0, then both layer 1
    next_chunk: usize = 0,
    next_layer: usize = 0,
    base: usize = 0, // the current pass's first chunk
    end: usize = 0, // and its end chunk (exclusive)

    pub fn next(self: *WorkerOrder) ?WorkerStep {
        if (self.next_chunk >= self.chunks or self.layers == 0 or self.chunk == 0) return null;
        if (self.next_chunk == self.end) {
            // a new pass: its rows from the pass rule at this chunk's absolute position (the fill's start + the chunk)
            const at = self.start + self.next_chunk * self.chunk;
            const rows = if (self.group != 0 and at >= self.pass_from) self.group * self.chunk else if (self.pass_rows <= self.chunk) self.chunk else workerRuleRows(at, self.pass_from, self.pass_rows, self.chunk, self.filling_pass);
            self.base = self.next_chunk;
            self.end = @min(self.base + @max(@as(usize, 1), rows / self.chunk), self.chunks);
            self.next_layer = 0;
        }
        const step: WorkerStep = .{ .chunk = self.next_chunk, .layer = self.next_layer };
        self.next_chunk += 1;
        if (self.next_chunk == self.end) {
            self.next_layer += 1;
            self.next_chunk = self.base;
            if (self.next_layer == self.layers) {
                self.next_layer = 0;
                self.next_chunk = self.end;
            }
        }
        return step;
    }
};

/// A thread's CPU time, involuntary switches and core now (a worker step's start and end).
const ThreadProbe = struct { cpu_ns: u64, nivcsw: u64, core: usize };

fn threadProbe() ThreadProbe {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.THREAD_CPUTIME_ID, &ts);
    var ru: std.os.linux.rusage = undefined;
    var inv: u64 = 0;
    if (std.os.linux.getrusage(std.os.linux.rusage.THREAD, &ru) == 0) inv = @intCast(ru.nivcsw);
    var cpu: usize = 0;
    _ = std.os.linux.getcpu(&cpu, null);
    return .{ .cpu_ns = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec)), .nivcsw = inv, .core = cpu };
}

/// GB10: X925 cores 5-9 and 15-19, A725 cores 0-4 and 10-14.
pub fn coreIsBig(core: usize) bool {
    return (core >= 5 and core <= 9) or (core >= 15 and core <= 19);
}

/// Adds a step's thread time, switches and the cores at its two ends to the worker's counters.
fn workerProbe(eh: *EngramHost, start: ThreadProbe) void {
    const end = threadProbe();
    _ = eh.inst_cpu_ns.fetchAdd(end.cpu_ns -% start.cpu_ns, .monotonic);
    _ = eh.inst_nivcsw.fetchAdd(end.nivcsw -% start.nivcsw, .monotonic);
    for ([_]usize{ start.core, end.core }) |core| {
        _ = (if (coreIsBig(core)) &eh.inst_big else &eh.inst_small).fetchAdd(1, .monotonic);
        if (core < 64) _ = eh.inst_cores.fetchOr(@as(u64, 1) << @intCast(core), .monotonic);
    }
}

/// The fill's read-ahead worker (a thread of model.zig's fill): from the chunk at `start`, chunk after chunk of
/// `chunk` rows to the end of `seq` (the prompt's whole ids), both Engram layers' rows hashed, read and decoded as
/// engramApply would, into the chunk's slot once the layers freed it. Errors land in wk_err (wk_failed set).
/// TF_DS_ENGRAM_FAULT=R:C (a test: the drill of a failure on one rank alone): rank R's Engram worker fails its row read once,
/// with the error the disk would give, at the C-th layer-0 step (0 is the first) it takes after the start; the switch is then
/// spent. Every rank parses the same text and acts only when it is the named one. Unset or empty: nothing.
pub const EngramFault = struct {
    rank: u32 = 0,
    at: u32 = 0,
    asked: bool = false,

    pub const Error = error{BadFault};

    pub fn parse(text: ?[]const u8, world: u32) Error!EngramFault {
        const t = text orelse return .{};
        if (t.len == 0) return .{};
        const colon = std.mem.indexOfScalar(u8, t, ':') orelse return error.BadFault;
        const rank = std.fmt.parseInt(u32, t[0..colon], 10) catch return error.BadFault;
        const at = std.fmt.parseInt(u32, t[colon + 1 ..], 10) catch return error.BadFault;
        if (rank >= world) return error.BadFault;
        return .{ .rank = rank, .at = at, .asked = true };
    }

    /// The start line's words.
    pub fn describe(f: EngramFault, buf: []u8, rank: u32) []const u8 {
        if (!f.asked) return "off";
        return std.fmt.bufPrint(buf, "{d}:{d} (this rank {s})", .{ f.rank, f.at, if (rank == f.rank) "fails its worker's row read once" else "carries on" }) catch "on";
    }
};

/// The parsed switch, set at the start by the model; `engram_fault_seen` counts this rank's layer-0 worker steps and
/// `engram_fault_spent` is set by the step that fired.
pub var engram_fault: EngramFault = .{};
pub var engram_fault_rank: u32 = 0; // this rank, for the worker thread
pub var engram_fault_seen: std.atomic.Value(u32) = .init(0);
pub var engram_fault_spent: std.atomic.Value(bool) = .init(false);

/// Whether the worker step of layer `j` on rank `rank` is the one the switch names; true at most once.
pub fn engramFaultFires(f: EngramFault, rank: u32, j: usize) bool {
    if (!f.asked or f.rank != rank or j != 0) return false;
    if (engram_fault_spent.load(.acquire)) return false;
    const n = engram_fault_seen.fetchAdd(1, .acq_rel);
    if (n != f.at) return false;
    engram_fault_spent.store(true, .release);
    return true;
}

/// How long the engine thread sleeps for the worker's rows before it looks again for the work that no ring announces (a copy
/// of its own finishing, rank 0's next round frame); TF_DS_ENGRAM_WAIT_US (readWaitEnv). A frame waits at most this long.
pub var engram_wait_ns: u64 = 100_000;
/// Read TF_DS_ENGRAM_WAIT_US once at start (model.readWorkerEnv, from Model.open).
pub fn readWaitEnv() void {
    if (std.c.getenv("TF_DS_ENGRAM_WAIT_US")) |v| engram_wait_ns = @as(u64, std.fmt.parseInt(u32, std.mem.span(v), 10) catch 100) *| 1000;
}

/// The worker's longest sleep for a free slot: a ring ends it at once, this only bounds a ring that never came.
const worker_backstop_ns: u64 = 10_000_000;

/// Waits (the worker's thread) until chunk `chunk` has a free slot: the layer's uploads (`used`) within `slots` of it.
/// False when `stop` ended the wait.
fn waitSlot(go: *wait_bell.Bell, used: *const std.atomic.Value(usize), slots: usize, chunk: usize, stop: *const std.atomic.Value(bool)) bool {
    while (chunk >= used.load(.acquire) + slots) {
        if (stop.load(.acquire)) return false;
        const at = go.seen(); // (read before the check below: a ring after it makes the sleep return at once)
        if (chunk < used.load(.acquire) + slots or stop.load(.acquire)) continue;
        go.sleep(at, worker_backstop_ns);
    }
    return !stop.load(.acquire);
}

/// What the engine thread does between its sleeps for the worker's rows: its copies' slots given back (the worker may be
/// waiting for one).
const EngramDuties = struct {
    eh: *EngramHost,
    pub fn run(d: EngramDuties) !void {
        try engramPoll(d.eh);
    }
};

/// Waits (the engine thread) until the worker has finished chunk `ci` of a layer (`done` > ci), running `duties` between sleeps.
/// False when `failed` ended the wait.
fn waitDone(ready: *wait_bell.Bell, done: *const std.atomic.Value(usize), ci: usize, failed: *const std.atomic.Value(bool), duties: anytype, timeout_ns: u64) !bool {
    while (done.load(.acquire) <= ci) {
        if (failed.load(.acquire)) return false;
        const at = ready.seen(); // (read before the duties and the check: a ring after it makes the sleep return at once)
        try duties.run();
        if (done.load(.acquire) > ci or failed.load(.acquire)) continue;
        ready.sleep(at, timeout_ns);
    }
    return true;
}

pub fn engramWorker(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, chunk: usize) void {
    const nl = @min(2, c.engram_layers.slice().len);
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    if (chunk == 0) return;
    if (std.c.getenv("TF_DS_ROCI_PAIR_LOG") != null) std.log.info("the Engram worker of this fill is thread {d} (core {d})", .{ std.os.linux.gettid(), threadProbe().core });
    const chunks = if (seq.len > start) (seq.len - start + chunk - 1) / chunk else 0;
    // The ring and its per-slot hash buffers hold wk_slots chunks of one resume unit (engram_rows.workerSlots: a pass's
    // units at the least, twice them with two passes in flight); the guard below keeps the worker wk_slots chunks ahead of
    // the layers' uploads.
    var order: WorkerOrder = .{
        .start = start,
        .chunks = chunks,
        .chunk = chunk,
        .layers = nl,
        .pass_from = eh.wk_pass_from,
        .pass_rows = if (eh.wk_ahead) @min(eh.wk_pass_rows, max_pass_rows) else chunk,
        .filling_pass = prompt2d.pass_filling_asked,
        .group = eh.wk_group,
    };
    while (order.next()) |step| {
        const at = start + step.chunk * chunk;
        const n = @min(chunk, seq.len - at);
        const j = step.layer;
        if (!waitSlot(&eh.wk_go, &eh.wk_used[j], eh.wk_slots, step.chunk, &eh.wk_stop)) return;
        const probe0 = threadProbe();
        defer workerProbe(eh, probe0);
        if (engramFaultFires(engram_fault, engram_fault_rank, j)) {
            std.log.err("TF_DS_ENGRAM_FAULT: rank {d}: the worker's row read of layer {d} at position {d} fails once (a drill)", .{ engram_fault_rank, j, at });
            eh.wk_err = error.InputOutput;
            eh.wk_failed.store(true, .release);
            eh.wk_ready.ring();
            return;
        }
        workerRows(eh, c, seq, at, n, j, cols, k, eh.wk_rows[step.chunk % eh.wk_slots][j], step.chunk % eh.wk_slots) catch |err| {
            eh.wk_err = err;
            eh.wk_failed.store(true, .release);
            eh.wk_ready.ring();
            return;
        };
        eh.wk_done[j].store(step.chunk + 1, .release);
        eh.wk_ready.ring();
    }
}

fn workerRows(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize, cols: usize, k: usize, out: []u16, slot: usize) !void {
    const li = c.engram_layers.slice()[j];
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    _ = eh.inst_steps.fetchAdd(1, .monotonic);
    if (j == 0) {
        const th = nowNs();
        eh.hasher.hashes(seq, start, n, eh.wk_hashes[slot][0 .. n * eh.hasher.layers * cols]);
        _ = eh.inst_hash_ns.fetchAdd(@intCast(nowNs() - th), .monotonic);
    }
    // packed shards: the chunk's records staged in the worker's slot, decoded on the GPU when applied
    if (eh.pk) |pr| {
        pr.step_tag.store(((start / 2048) << 8) | j, .monotonic);
        const tg = nowNs();
        try pr.chunkReadSplit(&pr.wk[slot][j], j, eh.wk_hashes[slot][0 .. n * eh.hasher.layers * cols], n, eh.hasher.layers, ((start / 2048) << 8) | j);
        _ = eh.inst_gather_ns.fetchAdd(@intCast(nowNs() - tg), .monotonic);
        const st = &pr.wk[slot][j];
        _ = eh.inst_rows.fetchAdd(n * pr.slots(), .monotonic);
        _ = eh.inst_miss.fetchAdd(st.misses, .monotonic);
        _ = eh.inst_bytes.fetchAdd(st.misses * pr.rec, .monotonic);
        return;
    }
    for (0..n) |r| {
        for (0..k) |q| eh.wk_flat[r * k + q] = eh.wk_hashes[slot][(r * eh.hasher.layers + j) * cols + eh.lo + q];
    }
    const m = n * k;
    const tg = nowNs();
    if (eh.paio) |aio| try aioGather(aio, t, eh.wk_flat[0..m], eh.wk_w[0 .. m * t.row_w], eh.wk_s[0 .. m * t.row_s]) else try eh.pool.gather(t, eh.wk_flat[0..m], eh.wk_w[0 .. m * t.row_w], eh.wk_s[0 .. m * t.row_s]);
    const td = nowNs();
    engram.decodeRows(eh.wk_w[0 .. m * t.row_w], eh.wk_s[0 .. m * t.row_s], out[0 .. m * t.row_w], m, t.row_w, t.row_s, 8);
    _ = eh.inst_gather_ns.fetchAdd(@intCast(td - tg), .monotonic);
    _ = eh.inst_decode_ns.fetchAdd(@intCast(nowNs() - td), .monotonic);
    _ = eh.inst_rows.fetchAdd(m, .monotonic);
    _ = eh.inst_miss.fetchAdd(m, .monotonic);
    _ = eh.inst_bytes.fetchAdd(m * (t.row_w + t.row_s), .monotonic);
}

/// Start the worker for a fill of `seq` from chunk `start` (any worker before it stopped first). `pass_from` and
/// `pass_rows` are the pass rule's setting (prompt2d.pass_from_asked, Model.pass_rows): with TF_DS_ENGRAM_AHEAD_PASS
/// the worker takes each pass's size from that rule at the pass's chunk position; unset, it keeps the baseline order.
pub fn engramWorkerStart(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, chunk: usize, pass_from: usize, pass_rows: usize) !void {
    engramWorkerStop(eh);
    try engramFlushPins(eh); // (a copy still reading a slot the new worker's first rows would overwrite)
    eh.pin = .{ .ring = eh.wk_slots };
    for (&eh.wk_done) |*v| v.store(0, .release);
    for (&eh.wk_used) |*v| v.store(0, .release);
    eh.wk_stop.store(false, .release);
    eh.wk_failed.store(false, .release);
    eh.wk_err = null;
    eh.wk_seq = seq;
    eh.wk_start = start;
    eh.wk_ahead = ahead_pass_on;
    eh.wk_pass_from = pass_from;
    eh.wk_pass_rows = pass_rows;
    eh.wk_thread = try std.Thread.spawn(.{}, engramWorker, .{ eh, c, seq, start, chunk });
    eh.ka.workerStarted();
}

/// Stop and join the worker (a fill's end, or its prompt about to change).
pub fn engramWorkerStop(eh: *EngramHost) void {
    const th = eh.wk_thread orelse return;
    eh.wk_stop.store(true, .release);
    eh.wk_go.ring();
    where.joinThread(th);
    eh.wk_thread = null;
    eh.wk_seq = &.{};
}

/// The pinned slots' events and, without packed shards, the pinned rows (TF_DS_ENGRAM_PINNED, the worker on): the worker's
/// rows land in page-locked memory, so the copy to the device is asynchronous. Returns the pinned bytes this rank holds
/// (packed shards: their stages are pinned by engram_rows, counted here).
pub fn engramPin(eh: *EngramHost, d: *const cuda.Driver) !usize {
    var bytes: usize = 0;
    for (&eh.pin_ev) |*row| for (row) |*ev| {
        ev.* = try cuda.Event.init(d, false);
    };
    if (eh.pk) |pr| {
        bytes = pr.pinnedBytes();
    } else {
        for (0..eh.wk_slots) |slot| for (0..2) |j| {
            const n = eh.wk_rows[slot][j].len;
            if (n == 0) continue;
            var hb = try cuda.HostBuffer.alloc(d, n * 2);
            eh.pin_host[slot][j] = hb;
            eh.wk_rows[slot][j] = hb.slice(u16)[0..n];
            bytes += n * 2;
        };
    }
    eh.pin = .{ .ring = eh.wk_slots };
    eh.pinned = true;
    return bytes;
}

/// Frees what engramPin made (the events and pinned rows); the worker is stopped.
pub fn engramUnpin(eh: *EngramHost) void {
    for (&eh.pin_ev) |*row| for (row) |*ev| {
        if (ev.*) |*x| x.deinit();
        ev.* = null;
    };
    for (&eh.pin_host) |*row| for (row) |*hb| {
        if (hb.*) |*x| x.free();
        hb.* = null;
    };
    eh.pinned = false;
}

const PinCtx = struct {
    eh: *EngramHost,

    fn done(c: PinCtx, j: usize, p: usize) !bool {
        return try c.eh.pin_ev[j][p].?.done();
    }

    fn release(c: PinCtx, j: usize, used: usize) void {
        c.eh.wk_used[j].store(used, .release);
        c.eh.wk_go.ring();
    }
};

/// Gives back, oldest first, the worker's slots whose copies are done (no wait).
pub fn engramPoll(eh: *EngramHost) !void {
    if (!eh.pinned) return;
    for (0..2) |j| _ = try eh.pin.poll(j, PinCtx{ .eh = eh });
}

/// The worker's `used` slots of layer j (up to `used`) copied by the stream `s`: the slots go back at once when the rows were
/// pageable (the call returned with them staged), else when the event recorded here is done. A full ring waits for its oldest.
pub fn engramRelease(eh: *EngramHost, s: cuda.Stream, j: usize, used: usize) !void {
    if (!eh.pinned) {
        eh.wk_used[j].store(used, .release);
        eh.wk_go.ring();
        return;
    }
    const held = where.enter(.engram_release, @intCast(j), 0);
    defer where.leave(held);
    while (eh.pin.room(j) == null) {
        const p = eh.pin.oldest(j).?;
        try eh.pin_ev[j][p].?.synchronize();
        _ = try eh.pin.poll(j, PinCtx{ .eh = eh });
    }
    const p = eh.pin.room(j).?;
    try eh.pin_ev[j][p].?.record(s);
    eh.pin.arm(j, p, used);
    try engramPoll(eh);
}

/// The worker's chunks of layer j the engine has uploaded: the slots given back (wk_used) and those whose copies are still
/// pending. A pageable upload made wk_used at once; the engine thread's test that the running worker is still on this fill's
/// chunks reads it (a pair's second pass counts too: its release names the larger count).
pub fn engramUploaded(eh: *EngramHost, j: usize) usize {
    return @max(eh.wk_used[j].load(.acquire), eh.pin.highest(j));
}

/// Waits for every pending copy and gives its slots back: before a worker starts afresh, whose first rows overwrite them.
pub fn engramFlushPins(eh: *EngramHost) !void {
    if (!eh.pinned) return;
    const prev = where.enter(.engram_flush, 0, 0);
    defer where.leave(prev);
    for (0..2) |j| while (eh.pin.oldest(j)) |p| {
        try eh.pin_ev[j][p].?.synchronize();
        _ = try eh.pin.poll(j, PinCtx{ .eh = eh });
    };
}

/// The worker's rows of Engram layer j for the chunk at `start` (waiting for them), for engramApply.
pub fn engramWorkerRows(eh: *EngramHost, j: usize, start: usize, chunk: usize) ![]u16 {
    const ci = (start - eh.wk_start) / chunk;
    const prev_place = where.enter(.engram_rows, @intCast(j), @intCast(ci));
    defer where.leave(prev_place);
    const t_in = nowNs();
    defer {
        const waited: u64 = @intCast(nowNs() - t_in);
        eh.inst_wait_ns += waited;
        if (j == 0) eh.ka.firstLayerWait(waited);
    }
    if (!try waitDone(&eh.wk_ready, &eh.wk_done[j], ci, &eh.wk_failed, EngramDuties{ .eh = eh }, engram_wait_ns)) return eh.wk_err orelse error.EngramWorkerFailed;
    return eh.wk_rows[ci % eh.wk_slots][j];
}

/// With packed shards: the worker's records of Engram layer j for the chunk at `start` (waiting for them), for
/// engramApply, as engramWorkerRows gives its rows.
pub fn engramWorkerStage(eh: *EngramHost, j: usize, start: usize, chunk: usize) !*engram_rows.Stage {
    const pr = eh.pk orelse return error.NoPackedRows;
    _ = try engramWorkerRows(eh, j, start, chunk);
    return &pr.wk[((start - eh.wk_start) / chunk) % eh.wk_slots][j];
}

/// Layer li's Engram on the chunk (engram_apply, no image span), as _forward_k runs it before the attention mixes: a
/// pending MoE post into the streams first (hc_post), then the rows of each position's n-grams (hashed on the host from
/// `seq`, the whole sequence's ids so far; read from the tables; decoded to bf16 as Engram._decode does), their
/// projection summed over the ranks in rank order, rounded to bf16, and the gate into the streams.
pub fn engramApply(e: *const Engine, ch: *Chunk, eh: *EngramHost, li: usize, seq: []const i32) !void {
    try engramPending(e, ch, li);
    try engramProject(e, ch, eh, li, seq);
    try engramFinish(e, ch, eh, li, false);
}

/// Layer li's pending MoE post into the streams (engramApply's first step).
pub fn engramPending(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    if (lay.engram_wkv == null) return;
    const c = e.c;
    const n = ch.n;
    if (ch.pending) {
        if (ownedRange(e, ch, li)) |r| {
            // WP4: the same row-sharded flush as attnMixes's pending branch (hcPost alone is _hc_mix_pf's "posted"
            // finisher without the RMSNorm finish; same G-buffer stride gotcha, same repack).
            try hcPostOwned(e, ch, .moe, r);
        } else {
            try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, n, c.hidden);
        }
        ch.pending = false;
    }
}

/// The Engram rows of layer li on the GPU and their projection into ek (engramApply's second step: the n-grams' rows, read or
/// taken from the read-ahead, uploaded, one GEMM). Nothing here reads the streams.
pub fn engramProject(e: *const Engine, ch: *Chunk, eh: *EngramHost, li: usize, seq: []const i32) !void {
    const lay = e.w.layers[li];
    const ew = lay.engram_wkv orelse return;
    const c = e.c;
    const n = ch.n;
    const li32: u16 = @intCast(li);
    const l = std.mem.indexOfScalar(u16, c.engram_layers.slice(), li32) orelse return error.NotAnEngramLayer;
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    const m = n * k;
    const prof = eh.io != null;
    var tq = if (prof) pstamp(e, eh) else 0;
    if (m * t.row_w != n * ew.k) return error.EngramShape;
    if (eh.use_n > 0) {
        // the fill's worker's resume units of the pass in order, each at its rows: its decoded rows, or its records
        // (packed shards) decoded on the GPU
        if (prof) eh.p_rows += lap(e, eh, &tq);
        for (0..eh.use_n) |u| {
            const r0 = u * eh.use_unit;
            const nu = @min(eh.use_unit, n - r0);
            const at = ch.eb + r0 * ew.k * 2;
            if (eh.pk) |pr| try pr.chunkApplyOn(eh.use_stage[u].?, nu, e.s, at, ch.lane) else try upload(e, at, eh.use_rows[u].?.ptr, nu * ew.k * 2);
        }
        if (prof) eh.p_upload += lap(e, eh, &tq);
    } else if (eh.pk) |pr| {
        // packed shards: the chunk's records (read ahead when they are this chunk's, else read here), decoded on the GPU
        const st = if (l < 2 and eh.ahead_ready[l] and eh.ahead_start == ch.start and eh.ahead_n == n) blk: {
            eh.ahead_ready[l] = false;
            break :blk &pr.ahead[l];
        } else blk: {
            eh.hasher.hashes(seq, ch.start, n, eh.hashes[0 .. n * eh.hasher.layers * cols]);
            try pr.chunkRead(&pr.chunk, l, eh.hashes[0 .. n * eh.hasher.layers * cols], n, eh.hasher.layers);
            break :blk &pr.chunk;
        };
        if (prof) eh.p_rows += lap(e, eh, &tq);
        try pr.chunkApply(st, n, e.s, ch.eb);
        if (prof) eh.p_upload += lap(e, eh, &tq);
    } else {
        // the rows read ahead (fillChunk's thread) when they are this chunk's, else read here
        var rows: []u16 = eh.rows;
        if (l < 2 and eh.ahead_ready[l] and eh.ahead_start == ch.start and eh.ahead_n == n) {
            rows = eh.ahead_rows[l];
            eh.ahead_ready[l] = false;
        } else {
            eh.hasher.hashes(seq, ch.start, n, eh.hashes[0 .. n * eh.hasher.layers * cols]);
            for (0..n) |r| {
                for (0..k) |j| eh.flat[r * k + j] = eh.hashes[(r * eh.hasher.layers + l) * cols + eh.lo + j];
            }
            if (eh.paio) |aio| try aioGather(aio, t, eh.flat[0..m], eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s]) else try eh.pool.gather(t, eh.flat[0..m], eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s]);
            engram.decodeRows(eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s], eh.rows[0 .. m * t.row_w], m, t.row_w, t.row_s, 8); // the rows on 8 threads
        }
        if (prof) eh.p_rows += lap(e, eh, &tq);
        try upload(e, ch.eb, rows.ptr, m * t.row_w * 2);
        if (prof) eh.p_upload += lap(e, eh, &tq);
    }
    try mm(e, ch, ew, ch.eb, ew.k, ch.ek, .fp32, ew.n);
    if (prof) eh.p_mm += lap(e, eh, &tq);
}

/// Whether layer li's Engram sums run on the transport at owned rows (TF_DS_ROCI_ENGRAM_AHEAD hoists those alone).
pub fn engramOnRoci(e: *const Engine, ch: *const Chunk, li: usize) bool {
    const t = e.two orelse return false;
    return t.rc != null and engramOwned(e, ch, li) != null;
}

/// Layer li's part 1 sent right behind its projection (TF_DS_ROCI_ENGRAM_AHEAD): the projection is done a layer ahead.
pub fn engramSendAhead(e: *const Engine, ch: *const Chunk, li: usize) !void {
    const r = engramOwned(e, ch, li) orelse return;
    try engramSend1Roci(e, ch, e.two.?.rc.?, r);
}

/// Layer li's Engram from its projection on: the sums (part 1 already sent when `sent1`), the gate into the streams.
pub fn engramFinish(e: *const Engine, ch: *Chunk, eh: *EngramHost, li: usize, sent1: bool) !void {
    const lay = e.w.layers[li];
    const ew = lay.engram_wkv orelse return;
    const c = e.c;
    const n = ch.n;
    const prof = eh.io != null;
    var tq = if (prof) pstamp(e, eh) else 0;
    // 2D (model.py esum): this node's columns of its rank's projection; the quarters give both ranks' whole ones
    const en = if (e.two) |two| two.engramWidth() else ew.n;
    // the sums at this node's rows alone, over the transport (engramSumsRoci)
    const own = engramOwned(e, ch, li);
    const r = own orelse return error.NoOwnedRows;
    try engramSumsRoci(e, ch, e.two.?.rc orelse return error.RociNoShape, r, sent1);
    if (prof) eh.p_x += lap(e, eh, &tq);
    // the gate on every row, or with the owned sums on those rows alone (no other row of the streams is read before
    // the boundary's resync)
    const rows: QRows = own orelse .{ .start = 0, .n = n };
    const rowb = c.hc * c.hidden * 2;
    if (own != null and e.two.?.rc != null) {
        // the key and value rows lie in the transport's two blocks: block stride = the slab's rows x a pair's width
        if (rows.n != 0) try tri_basic.engramGate2b(e.t, ch.h + rows.start * rowb, e.two.?.rc.?.addr(.eng_blk), ownQ(e, ch) * e.two.?.ew[e.two.?.p], lay.engram_qk, ch.h_alt + rows.start * rowb, c.eps, rows.n, c.hidden);
    } else try engramGateRows(e.t, lay.engram_qk, ch.h, ch.kv, ch.h_alt, rows, c.eps, c.hidden, rowb, en * 2);
    swapStreams(ch, ch.h_alt);
    // image spans take no Engram contribution: their rows keep the streams they came with (engram_mask)
    for (ch.spans) |sp| {
        const o = rowsIn(overlap(sp, ch.start, n) orelse continue, rows) orelse continue;
        try e.ops.copyRows(e.s, ch.h_alt + o[0] * rowb, rowb, ch.h + o[0] * rowb, rowb, rowb, o[1] - o[0]);
    }
    if (prof) {
        eh.p_gate += lap(e, eh, &tq);
        eh.p_calls += 1;
    }
}

/// The Engram gate on rows `r` of the streams h [*, hc, d] bf16 (`h_row` bytes a row) with kv [*, (hc + 1) d] bf16
/// (`kv_row` bytes a row) into out's rows r: the whole chunk's _engram_gate launch at the rows' offsets (one program a
/// row and stream).
pub fn engramGateRows(t: tri.Tri, qk: u64, h: u64, kv: u64, out: u64, r: QRows, eps: f32, d: usize, h_row: usize, kv_row: usize) !void {
    try tri_basic.engramGate(t, h + r.start * h_row, kv + r.start * kv_row, qk, out + r.start * h_row, eps, r.n, d);
}

/// TF_DS_ENGRAM_OWNER_SUM's start check on this rank (null: it can run; else the reason, for
/// Two.agreeEngramOwnerSum): quarter-HC, Engram layers of equal pair widths `ew` (the owner sums' parts take one width),
/// no ring links (the owner sums' exchanges are the world communicator's), and every gate launch engramGateRows makes at
/// each of the `world` ranks' owned rows of each row count in `qs`, for every Engram layer below `half`, found in this
/// rank's kernel set (`finds.has`, as frontOwnedOff's).
pub fn engramOwnerSumOff(gpa: std.mem.Allocator, finds: anytype, ew: [2]usize, layers: []const weights.Layer, half: usize, eps: f32, d: usize, hc: usize, qs: []const usize, world: usize) !?[]const u8 {
    const rows = try ownedRows(gpa, qs, world);
    defer gpa.free(rows);
    return engramOwnerSumOffAt(gpa, finds, ew, layers, half, eps, d, hc, rows);
}

pub fn engramOwnerSumOffAt(gpa: std.mem.Allocator, finds: anytype, ew: [2]usize, layers: []const weights.Layer, half: usize, eps: f32, d: usize, hc: usize, rows: []const QRows) !?[]const u8 {
    if (ew[0] == 0 or ew[0] != ew[1]) return "needs Engram layers of equal pair widths";
    var log: tri.Log = .{ .gpa = gpa };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    // h, kv and out as the arena aligns a chunk's buffers (256 bytes)
    const h: u64 = 1 << 32;
    const kv: u64 = 2 << 32;
    const out: u64 = 3 << 32;
    for (layers[0..@min(half, layers.len)]) |lay| {
        if (lay.engram_wkv == null) continue;
        for (rows) |r| try engramGateRows(t, lay.engram_qk, h, kv, out, r, eps, d, hc * d * 2, (ew[0] + ew[1]) * 2);
    }
    for (log.items.items) |l| if (!l.foundIn(finds)) return "this rank's kernel set has no _engram_gate variant for the owned rows";
    return null;
}

/// A profile's clock after the stream drains (prompt Engram split).
fn pstamp(e: *const Engine, eh: *const EngramHost) u64 {
    e.s.synchronize() catch {};
    return @intCast(std.Io.Timestamp.now(eh.io.?, .awake).nanoseconds);
}

fn lap(e: *const Engine, eh: *const EngramHost, t: *u64) u64 {
    const now = pstamp(e, eh);
    defer t.* = now;
    return now - t.*;
}

/// A chunk's rows `ids` of table `t` by AIO (the pool's bytes, at ~0.07 s for 24,576 rows where the pool's preads take
/// ~0.5 s on one GB10 node): batches of a quarter of the ring's slots (two reads a row), all submitted (a submit reaps while the
/// ring is full), then waited for in order.
fn aioGather(aio: *engram_aio.Aio, t: engram_io.Table, ids: []const i64, out_w: []u8, out_s: []u8) !void {
    const per = @max(aio.slots / 4, 1);
    var batch: [64]u64 = undefined;
    var nb: usize = 0;
    var at: usize = 0;
    errdefer for (batch[0..nb]) |id| aio.wait(id) catch {};
    while (at < ids.len) : (at += per) {
        const n = @min(per, ids.len - at);
        if (nb == batch.len) return error.TooManyRows;
        batch[nb] = try aio.start(t, ids[at..][0..n], out_w[at * t.row_w ..][0 .. n * t.row_w], out_s[at * t.row_s ..][0 .. n * t.row_s]);
        nb += 1;
    }
    for (batch[0..nb], 0..) |id, i| {
        aio.wait(id) catch |err| {
            for (batch[i + 1 .. nb]) |rest| aio.wait(rest) catch {};
            nb = 0;
            return err;
        };
    }
    nb = 0;
}

/// The decoder's bounded replay at its first layer (_forward_k's `replay`, CED's prefill): the layer's attention mixes
/// (the pending MoE post fused in) and its kv-source update over every row of the chunk, then the chunk cut to its rows
/// at positions >= replay, the streams and their pre copied down (h[first:], pre[first:]). The decoder layers then run
/// on those rows with no window key before replay and this layer's update done. False: an encoder-only chunk (no
/// row at or past replay), which ends here.
pub fn replayCut(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, replay: usize, host_pos: []i64) !bool {
    _ = host_pos;
    const first = @max(ch.start, replay) - ch.start;
    // the boundary on owned rows (the mixes, the kv-source update of the owned rows, the cache rows gathered), then the rows the
    // decoder half reads handed to every node (handoff) in place of the all-row mixes and update
    const r = boundaryRange(e, ch, li).?;
    try boundaryUpdate(e, ch, cs, sh, li, r);
    if (first >= ch.n) return false;
    try handoff(e, ch, first);
    try cutRows(e, ch, first, true);
    return true;
}

/// The decoder rows' handoff (TF_DS_ROCI_ANY): each node's owned rows from `first` on of the streams h and of pre are copied into the
/// arena's handoff regions at row (row - first), and the node's window goes to the three other nodes; empty windows send their word.
pub fn handoff(e: *const Engine, ch: *Chunk, first: usize) !void {
    const rc = e.two.?.rc.?;
    const c = e.c;
    const d = ch.n - first;
    if (d > roci.hand_rows) return error.ChunkTooLong;
    const s = slab_plan.slab(ch.n, e.comm.rank);
    const lo = @max(first, s.start);
    const hi = @min(ch.n, s.start + s.n);
    if (hi > lo) {
        const row_h = c.hc * c.hidden * 2;
        const row_p = c.hc * 4;
        try e.ops.copyRows(e.s, ch.h + lo * row_h, row_h, rc.addr(.hand_h) + (lo - first) * row_h, row_h, row_h, hi - lo);
        try e.ops.copyRows(e.s, ch.pre + lo * row_p, row_p, rc.addr(.hand_pre) + (lo - first) * row_p, row_p, row_p, hi - lo);
    }
    try rc.sendX(e.s, .gh, ch.n, d);
    try rc.wait(e.s, .gh);
}

/// The cut of replayCut: the chunk's rows from `first` on only, with their streams and pre (h[first:], pre[first:]) copied down, from
/// the chunk's own buffers or, `handed` (TF_DS_ROCI_ANY), from the handoff regions where every node's rows have landed. The decoder layers
/// then run on those rows with no window key before replay.
fn cutRows(e: *const Engine, ch: *Chunk, first: usize, handed: bool) !void {
    const c = e.c;
    const n = ch.n - first;
    const row_h = c.hc * c.hidden * 2;
    const row_p = c.hc * 4;
    if (handed) {
        const rc = e.two.?.rc.?;
        try e.ops.copyRows(e.s, rc.addr(.hand_h), row_h, ch.h_alt, row_h, row_h, n);
        try e.ops.copyRows(e.s, rc.addr(.hand_pre), row_p, ch.pre_f, row_p, row_p, n);
        ch.boundary_done = true;
    } else {
        try e.ops.copyRows(e.s, ch.h + first * row_h, row_h, ch.h_alt, row_h, row_h, n);
        try e.ops.copyRows(e.s, ch.pre + first * row_p, row_p, ch.pre_f, row_p, row_p, n);
    }
    std.mem.swap(u64, &ch.h, &ch.h_alt);
    std.mem.swap(u64, &ch.pre, &ch.pre_f);
    ch.start += first;
    ch.n = n;
    // pos[first:]: a view, first * 8 bytes in (its alignment is the one the served kernels were specialized for). An odd
    // `first` leaves it off a 16-byte boundary, and the kit has no variant for some launches with such a view (a short
    // chunk's split attention: _sparse_attn_part with SPLITS > 1 and no row bases); the same positions copied, in the
    // stream's order, to a 16-byte boundary then: the kernels read the same values from an address they were captured for.
    if ((first * 8) % 16 != 0) {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(ch.pos1, ch.pos + first * 8, n * 8, e.s.handle), "cuMemcpyDtoDAsync");
        ch.pos = ch.pos1;
    } else ch.pos += first * 8;
}

/// DSpark tap j (_forward_k at a tap layer): a pending MoE gather posted into the streams first (hc_post, not fused),
/// then h.to(fp32).mean(1).to(bf16) into tap j's rows.
pub fn tap(e: *const Engine, ch: *Chunk, j: usize, li: usize) !void {
    const c = e.c;
    const r = ownedRange(e, ch, li);
    if (ch.pending) {
        if (r) |rr| {
            try hcPostOwned(e, ch, .moe, rr);
        } else {
            try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, ch.n, c.hidden);
        }
        ch.pending = false;
    }
    // WP4: hcMean4 is a per-row mean (no cross-row reduction), so it is safe to run over the full ch.n even with
    // this node's don't-care garbage in the rows it does not own (that garbage produces garbage only in those same
    // rows' tap output, which the gather below overwrites with every rank's own correct rows next) -- no row-shard
    // needed for the mean itself, only for what already needed it above (hcPost) and for making ch.taps globally
    // correct afterward, since DSpark drafting reads every row of it on every node.
    if (ch.n > ch.dec_rows) return error.ChunkTooLong;
    return tapOwned(e, ch, j, r orelse return error.NoOwnedRows);
}

/// TF_DS_ROCI_DEC's tap j on this rank's slab `r` of the decoder rows: the mean of the slab's streams into its rows of the taps region (the
/// three taps side by side a row; the other ranks' rows are the transport's to land, never written here), and at the last tap the slab's
/// rows of all three to the other three ranks in one event (gt). The drafter's read of the taps waits for it (tapsLanded).
fn tapOwned(e: *const Engine, ch: *Chunk, j: usize, r: QRows) !void {
    const c = e.c;
    const row_h = c.hc * c.hidden * 2;
    const ld = tapsLd(e);
    if (r.n != 0) try e.exact.hcMean4(e.s, ch.h + r.start * row_h, tapRows(e, ch, j) + r.start * ld * 2, ld, r.n, c.hidden);
    if (j + 1 == c.dspark_taps.slice().len) try e.two.?.rc.?.sendX(e.s, .gt, ch.n, 0);
}

/// TF_DS_ROCI_DEC: the taps' rows of the decoder window landed from the other ranks, before the drafter reads them (a stream wait; nothing
/// without the switch).
pub fn tapsLanded(e: *const Engine) !void {
    try e.two.?.rc.?.wait(e.s, .gt);
}

/// The values between two rows of a tap: the model's width, or with TF_DS_ROCI_DEC the three taps side by side (the arena's taps region,
/// the drafter's own row).
pub fn tapsLd(e: *const Engine) usize {
    return e.c.dspark_taps.slice().len * e.c.hidden;
}

/// Tap j's rows (bf16 [n, D], n <= dec_rows; rows tapsLd apart).
pub fn tapRows(e: *const Engine, ch: *const Chunk, j: usize) u64 {
    return ch.taps + j * e.c.hidden * 2;
}

/// TF_DS_QGATE_CANARY 2 or 3 in effect (Model.open: the same canary asked on every rank, with the checkpoint's own
/// kernels): layer body_layer's MoE partials, each rank's own before their gather, perturbed on purpose in every prompt
/// chunk and every round (mx.qgateBody): one more FP8 step, act_quant's quant-dequant of each 32-value group (body_fp8),
/// or each nonzero finite value one fp32 ulp away from zero (body_ulp: a reduction order's size). A quality gate's floor,
/// never a serving setting. The gather sums the perturbed partials, so every rank reads the same streams. Off: nothing
/// launches.
pub var qgate_body: prompt2d.Qgate = .off;

/// The layer whose MoE partials the body canary perturbs: the first, so every later layer, the keys they store and
/// the head read perturbed streams.
pub const body_layer: usize = 0;

/// The body canary at layer `li`'s MoE partials `pm` (fp32 [rows, moeWidth], this rank's, before their gather): nothing
/// unless it is in effect (qgate_body) and li is body_layer. A chunk whose owner sum's part 1 already left in the MoE's
/// own launch (own1_sent: TF_DS_ROCI_COMBINE_DOOR / TF_DS_ROCI_SENT_FIRST) cannot be perturbed after the fact: refused,
/// never a canary that perturbs nothing.
pub fn bodyCanary(e: *const Engine, ch: ?*const Chunk, pm: u64, rows: usize, li: usize) !void {
    if (qgate_body == .off or li != body_layer) return;
    const k = e.mx orelse return error.NotAnMxCheckpoint;
    if (ch) |c| if (c.own1_sent) return error.CanaryAfterSend;
    return mx.qgateBody(k, e.s, pm, rows * moeWidth(e.two, e.c.hidden), @backingInt(qgate_body));
}

/// The columns of a rank's MoE partials (the experts' down output, its row stride): its node's down columns on a 2D
/// node (Two.dw), the hidden size otherwise.
pub fn moeWidth(two: ?*const prompt2d.Two, hidden: usize) usize {
    return if (two) |t| t.dw[t.p] else hidden;
}

/// The head on the chunk's last row (_forward_k without all_logits): the pending MoE post, collapse_norm of the row with
/// its pre, the head's columns (one row through the head's own group, fp32) and their gather: head_g holds the
/// prompt's logits, the ranks' columns in rank order.
pub fn head(e: *const Engine, ch: *Chunk) !void {
    const c = e.c;
    try headOwned(e, ch);
    const hl = e.w.head;
    dump(e, 0, "head_l", ch.head_l, hl.n * 4);
    try e.two.?.quarters(e, ch.head_l, ch.head_g, 1, e.two.?.hw, 4); // the vocabulary quarters, over the decode rings
}

/// TF_DS_ROCI_DEC's head row: the rank whose slab holds the window's last row (rocinante.headOwner) posts the pending MoE sums into its slab's
/// streams (its two blocks, hcPostOwned) and collapses the last row into the head_x region, which goes to the other three ranks (hx); every
/// rank waits for it before its head columns. The other ranks post nothing: no other row reaches the head.
fn headOwned(e: *const Engine, ch: *Chunk) !void {
    const c = e.c;
    const n = ch.n;
    const rc = e.two.?.rc.?;
    const row_h = c.hc * c.hidden * 2;
    if (roci.headOwner(n) == e.comm.rank) {
        const r = ownedRange(e, ch, c.layers) orelse return error.NoOwnedRows;
        if (ch.pending) try hcPostOwned(e, ch, .moe, r);
        try tri_basic.collapseNorm(e.t, ch.h + (n - 1) * row_h, ch.pre + (n - 1) * c.hc * 4, e.w.norm, ch.head_x, c.eps, 1, c.hidden);
    }
    ch.pending = false;
    try rc.sendX(e.s, .hx, n, 0);
    try rc.wait(e.s, .hx);
}

/// A layer's end: the MoE's gather is pending (the next layer's attention mixes post it) and the FFN's pre_out is the
/// next layer's pre (_forward_k's `pre, pre_f = pre_f, pre`).
pub fn endLayer(ch: *Chunk) void {
    ch.pending = true;
    std.mem.swap(u64, &ch.pre, &ch.pre_f);
}

test "a small chunk at the replay's cut reads its window from the ring with the floor off, from the floor with it on" {
    const window = 128; // config.json's sliding_window
    for ([_]bool{ false, true }) |on| {
        // replayCut leaves 1 .. 16 rows from the floor (start = floor = replay): its window reaches below the floor
        for (1..ring_extra + 1) |n| {
            const m = chunkMode(n, 4096 - 16, 4096 - 16, window, on);
            try std.testing.expectEqual(!on, m.ring);
            try std.testing.expectEqual(on, m.one_split);
        }
        // a last chunk of 2 .. 16 rows whose window reaches below the floor (replay = the prompt's length - window)
        for (2..ring_extra + 1) |n| {
            const m = chunkMode(n, 6144, 6144 + n - window, window, on);
            try std.testing.expectEqual(!on, m.ring);
            try std.testing.expectEqual(on, m.one_split);
        }
        // a window that stays at or above the floor: ring mode either way (a last chunk of one row, the floor 0)
        for ([_][2]usize{ .{ 1, 6144 - window + 1 }, .{ 16, 0 }, .{ 7, 6144 - window + 1 } }) |nf| {
            const m = chunkMode(nf[0], 6144, nf[1], window, on);
            try std.testing.expect(m.ring and !m.one_split);
        }
        // past RING_EXTRA rows: never ring mode, never one split
        for ([_]usize{ 17, 112, 2048 }) |n| {
            for ([_]usize{ 0, 6144 - 16, 6144 }) |floor| {
                const m = chunkMode(n, 6144, floor, window, on);
                try std.testing.expect(!m.ring and !m.one_split);
            }
        }
    }
}

test "the replay floor on: no prompt length's decoder chunk reads a key below the floor from the ring; off: exactly two classes do" {
    // fillChunk's chunks start at the prompt's multiples of model.chunk_rows; the decoder layers run each one's rows at
    // positions >= replay (replayCut), with replay = the prompt's length - window and no window key before it
    const window = 128;
    const chunk = 2048;
    var stale_lengths: usize = 0;
    for (1..64 * chunk + 128) |len| {
        const replay = len -| window;
        var stale = [2]bool{ false, false }; // the floor off, on
        var start = replay / chunk * chunk; // the chunks before it are encoder-only
        while (start < len) : (start += chunk) {
            const at = @max(start, replay);
            const n = @min(start + chunk, len) - at;
            for ([_]bool{ false, true }, 0..) |on, k| {
                const m = chunkMode(n, at, replay, window, on);
                // ring mode reads the ring's last window - 1 positions before the chunk, below the floor too
                if (m.ring and replay > at -| (window - 1)) stale[k] = true;
                try std.testing.expect(!m.ring or n <= ring_extra);
                try std.testing.expect(!m.one_split or (!m.ring and n <= tri.decode_rows));
                if (!on) try std.testing.expect(m.ring == (n <= ring_extra) and !m.one_split); // off: the lane's bits
            }
        }
        try std.testing.expect(!stale[1]);
        // the two classes: 2048k + 112 .. 127 tokens (the cut leaves 1 .. 16 rows) and a last chunk of 2 .. 16 rows
        const r = len % chunk;
        const class = len > chunk and ((r >= 112 and r <= 127) or (r >= 2 and r <= 16));
        try std.testing.expectEqual(class, stale[0]);
        stale_lengths += @intFromBool(stale[0]);
    }
    try std.testing.expectEqual(@as(usize, 64 * 31), stale_lengths);
}

test "a 4,097-row prompt's tail pass runs the decode kernels, a merged pass the prompt kernels" {
    // 4,097 rows: the baseline third pass is the one row at 4,096 (a ring window, eight splits, the grouped linears). A
    // merged 2,049-row pass runs that row non-ring with one split and the prefill linears: different bits.
    const tail = chunkMode(1, 4096, 0, 128, false);
    try std.testing.expect(tail.ring);
    try std.testing.expectEqual(tri_attn.attn_splits, tri_attn.splits(1, tail.one_split));
    const whole = chunkMode(2049, 2048, 0, 128, false);
    try std.testing.expect(!whole.ring);
    try std.testing.expectEqual(@as(usize, 1), tri_attn.splits(2049, whole.one_split));
    try std.testing.expectEqual(Front.grouped, attnFront(false, false, true, true, 1));
    try std.testing.expectEqual(Front.linears, attnFront(false, false, true, true, 2049));
}

test "hd ** -0.5 as the served build passes it" {
    try std.testing.expectEqual(@as(u32, 0x3d3504f3), @as(u32, @bitCast(scale(512))));
}

/// error.NotPortedYet with its site on stderr (a request that reaches a path the port lacks names it in the log).
fn notPorted(src: std.builtin.SourceLocation) error{NotPortedYet} {
    std.debug.print("{{\"not_ported\": \"{s}:{d}\"}}\n", .{ src.file, src.line });
    return error.NotPortedYet;
}

/// 40 layers whose compressor and indexer follow the checkpoint's config (weights.zig loads comp_wkv on a kv source,
/// idx_wq_b on an index source), the front linears FP8 on one K.
fn checkpointLayers(lays: *[40]weights.Layer) !void {
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(std.testing.allocator, config_mod.test_config, &why);
    const one: weights.Linear = .{ .words = 0, .k = 1, .n = 1 };
    for (lays, 0..) |*l, i| {
        l.* = .{ .idx = @intCast(i), .ratio = cfg.ratios[i] };
        l.comp_wkv = if (cfg.kv_sources.has(i)) one else null;
        l.idx_wq_b = if (cfg.index_sources.has(i)) one else null;
        l.wq_a = .{ .words = 0, .k = 5120, .n = 1280, .fmt = .fp8 };
        l.wkv = .{ .words = 0, .k = 5120, .n = 512, .fmt = .fp8 };
        l.q_norm = 0x7000 + i * 0x1000;
    }
}

test "TF_DS_QUANT_ONCE: attention's front quantizes x once where the owned front does not run, with the checkpoint's own weights; unset, the baseline front" {
    for ([_]usize{ 1, 17, 63, 64, 128, 129, 1000, 2048 }) |n| {
        const today: Front = if (n > decode_rows) .linears else .grouped;
        // unset: the baseline front whatever else holds
        for ([_]bool{ false, true }) |native| for ([_]bool{ false, true }) |codes| try std.testing.expectEqual(today, attnFront(false, false, native, codes, n));
        // set: once on the checkpoint's own FP8 / FP4 front linears, else the baseline
        try std.testing.expectEqual(Front.once, attnFront(false, true, true, true, n));
        try std.testing.expectEqual(today, attnFront(false, true, false, true, n));
        try std.testing.expectEqual(today, attnFront(false, true, true, false, n));
        // the owned front first (TF_DS_FRONT_OWNED quantizes its owned rows once already)
        for ([_]bool{ false, true }) |once| try std.testing.expectEqual(Front.owned, attnFront(true, once, true, true, n));
    }
    try std.testing.expect(!quant_once); // Model.open's starting value
}

test "attention takes its front through attnFront, and the quant-once front through frontLinears at the chunk's rows" {
    const src = @embedFile("prompt.zig");
    const at = std.mem.indexOf(u8, src, "pub fn attention(").?;
    const end = at + std.mem.indexOf(u8, src[at..], "\n}\n").?;
    const body = src[at..end];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "switch (attnFront(front != null, quant_once, e.mx != null, frontCodes(lay), n))"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, ".once => try frontLinears(e.mx.?, e.s, ch.mxs, lay, ch.x, c.hidden, .{ .start = 0, .n = n }, ch.qa, ch.y, hd)"));
    // frontRows (the owned front) is frontLinears then the norm
    const fr = std.mem.indexOf(u8, src, "pub fn frontRows(").?;
    const fr_body = src[fr .. fr + std.mem.indexOf(u8, src[fr..], "\n}\n").?];
    try std.testing.expect(std.mem.indexOf(u8, fr_body, "try frontLinears(k, s, sc, lay, x, ldx, r, qa, y, hd);\n    try frontNorm(t, lay, qa, qr, r, eps);") != null);
}

test "TF_DS_TOPK_TILES: the indexer's keyed top-k takes the one-launch kernel exactly where the chain prunes and k fits it; unset, the chain" {
    // the chain prunes past TOPK_FUSED_MAX keys and more tiles than k (tri_index.prunes)
    for ([_]usize{ 64, 512 }) |k| for ([_]usize{ 4096, 4097, 32768, 32769, 65536, 524288 }) |n| {
        try std.testing.expect(!prunedOnTiles(false, n, k));
        try std.testing.expectEqual(tri_index.prunes(n, k), prunedOnTiles(true, n, k));
    };
    try std.testing.expect(!prunedOnTiles(true, 32768, 512)); // 512 tiles: not pruned
    try std.testing.expect(prunedOnTiles(true, 32769, 512));
    try std.testing.expect(!prunedOnTiles(true, 1 << 20, 1024)); // past the kernel's k
}

test "TF_DS_IDX_PROJ_OWNED: the indexer's projections run on idx_shard's owned rows only with the switch and those rows; else every row" {
    const own: QRows = .{ .start = 1024, .n = 512 };
    try std.testing.expectEqual(own, projRows(own, true, 2048));
    try std.testing.expectEqual(QRows{ .start = 0, .n = 2048 }, projRows(own, false, 2048)); // unset: the baseline rows
    try std.testing.expectEqual(QRows{ .start = 0, .n = 1000 }, projRows(null, true, 1000)); // a partial chunk
    try std.testing.expectEqual(QRows{ .start = 0, .n = 2048 }, projRows(null, false, 2048));
}

test "the index queries at owned rows are the whole chunk's launches at the rows' offsets: RoPE (POS from the first row) and fp4_qd" {
    const gpa = std.testing.allocator;
    var whole: tri.Log = .{ .gpa = gpa };
    defer whole.deinit();
    var part: tri.Log = .{ .gpa = gpa };
    defer part.deinit();
    const iq: u64 = 0x1000_0000;
    const iq4: u64 = 0x2000_0000;
    const pos: u64 = 0x3000_0000;
    try idxQueries(.{ .log = &whole }, iq, iq4, 0x4000_0000, 0x5000_0000, pos, .{ .start = 0, .n = 2048 }, 64, 32, 128);
    const r: QRows = .{ .start = 1536, .n = 512 };
    try idxQueries(.{ .log = &part }, iq, iq4, 0x4000_0000, 0x5000_0000, pos, r, 64, 32, 128);
    try std.testing.expectEqual(@as(usize, 2), whole.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), part.items.items.len);
    const row = 32 * 128 * 2; // iq / iq4 bytes a row
    for (whole.items.items, part.items.items) |w, p| {
        try std.testing.expectEqualStrings(w.name, p.name);
        try std.testing.expectEqual(w.consts.len, p.consts.len);
        for (w.consts, p.consts) |a, b| try std.testing.expectEqual(a, b);
        for (w.args, p.args) |a, b| {
            const name = a.name;
            if (std.mem.eql(u8, name, "X") or std.mem.eql(u8, name, "OUT")) {
                try std.testing.expectEqual(a.value.ptr.addr + r.start * row, b.value.ptr.addr);
            } else if (std.mem.eql(u8, name, "POS")) {
                try std.testing.expectEqual(a.value.ptr.addr + r.start * 8, b.value.ptr.addr);
            } else if (std.mem.eql(u8, name, "nblk")) {
                try std.testing.expectEqual(@as(i32, @intCast(r.n * 32 * 128 / 32)), b.value.i32);
            } else try std.testing.expectEqualDeep(a, b);
        }
    }
    // the RoPE's rows are its grid's
    try std.testing.expectEqual(@as(u32, 2048), whole.items.items[0].grid[0]);
    try std.testing.expectEqual(@as(u32, 512), part.items.items[0].grid[0]);
}

test "TF_DS_IDX_PROJ_OWNED's start check: idx_shard, native weights, every owned-row query launch of every index layer found" {
    var lays: [40]weights.Layer = undefined;
    try checkpointLayers(&lays);
    const gpa = std.testing.allocator;
    const qs = [_]usize{ 256, 512, 1024, 512 };
    var n: usize = 0;
    try std.testing.expectEqual(@as(?[]const u8, null), try idxProjOwnedOff(gpa, FakeFinds{ .asked = &n }, true, true, &lays, 20, 64, 32, 128, &qs, 4));
    // index layers 2, 8, 14 before the candidate source: two launches at every rank's owned rows of every row count
    try std.testing.expectEqual(@as(usize, 3 * qs.len * 4 * 2), n);
    try std.testing.expectEqualStrings("needs TF_DS_IDX_SHARD=1", (try idxProjOwnedOff(gpa, FakeFinds{ .asked = &n }, true, false, &lays, 20, 64, 32, 128, &qs, 4)) orelse "can");
    try std.testing.expectEqualStrings("needs the checkpoint's own FP8 / MXFP4 weights", (try idxProjOwnedOff(gpa, FakeFinds{ .asked = &n }, false, true, &lays, 20, 64, 32, 128, &qs, 4)) orelse "can");
    for ([_][]const u8{ "_rope_heads", "_fp4qd_p2" }) |missing|
        try std.testing.expectEqualStrings("this rank's kernel set has no _rope_heads / _fp4qd_p2 variant for the owned rows", (try idxProjOwnedOff(gpa, FakeFinds{ .asked = &n, .missing = missing }, true, true, &lays, 20, 64, 32, 128, &qs, 4)) orelse "can");
}

test "the indexer projects through projRows and idxQueries: wq_b, the queries and weights_proj on the same rows" {
    const src = @embedFile("prompt.zig");
    const at = std.mem.indexOf(u8, src, "fn indexer(").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const pr = projRows(idxr, if (e.two) |t| t.idx_proj_owned else false, n);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "try mmRows(e, ch, pr.n, wqb, ch.qr + pr.start * qw * 2, qw, ch.iq + pr.start * ih * id * 2, .bf16, ih * id);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "try idxQueries(e.t, ch.iq, ch.iq4, rope.cos, rope.sin, ch.pos, pr, c.rope_dim, ih, id);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "ch.x + pr.start * c.hidden * 2, c.hidden, lay.idx_proj, c.hidden, pr.n, ih, c.hidden, ch.iw + pr.start * ih * 2"));
}

test "the owned front's q norm is the whole chunk's launch at the owned rows' offsets: one _rmsnorm program a row" {
    var log: tri.Log = .{ .gpa = std.testing.allocator };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    var lay: weights.Layer = .{ .idx = 0, .ratio = 0 };
    lay.wq_a = .{ .words = 0, .k = 5120, .n = 1280, .fmt = .fp8 };
    lay.q_norm = 0x7000;
    const qa: u64 = 0x100000;
    const qr: u64 = 0x900000;
    try tri_basic.rmsnorm(t, qa, 1280, lay.q_norm, qr, 1280, 1e-20, 2048, 1280); // attention's whole-chunk launch
    try frontNorm(t, lay, qa, qr, .{ .start = 512, .n = 512 }, 1e-20);
    try std.testing.expectEqual(@as(usize, 2), log.items.items.len);
    const all = log.items.items[0];
    const own = log.items.items[1];
    try std.testing.expectEqualStrings(all.name, own.name);
    try std.testing.expectEqual([3]u32{ 512, 1, 1 }, own.grid);
    try std.testing.expectEqualDeep(all.consts, own.consts);
    try std.testing.expectEqual(all.args.len, own.args.len);
    for (all.args, own.args) |a, b| {
        try std.testing.expectEqualStrings(a.name, b.name);
        const shift: u64 = if (std.mem.eql(u8, a.name, "X") or std.mem.eql(u8, a.name, "OUT")) 512 * 1280 * 2 else 0;
        switch (a.value) {
            .ptr => |pa| try std.testing.expectEqual(pa.addr + shift, b.value.ptr.addr),
            else => try std.testing.expectEqualDeep(a.value, b.value),
        }
    }
}

/// A kernel set for start checks: every launch found but those of `missing`; counts the lookups.
const FakeFinds = struct {
    missing: ?[]const u8 = null,
    asked: *usize,

    pub fn has(f: FakeFinds, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) bool {
        _ = .{ args, consts };
        f.asked.* += 1;
        const m = f.missing orelse return true;
        return !std.mem.eql(u8, m, name);
    }

    /// No extra set (a runTwin launch then counts as found when the base set has its fallback, as Tri runs it).
    pub fn hasExtra(f: FakeFinds, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) bool {
        _ = .{ f, name, args, consts };
        return false;
    }
};

test "TF_DS_ROPE_W on, a kit with no _rope_heads_w at the owned rows: the owned-rows check passes on _rope_heads, as the launch falls back to it" {
    const keep = tri_norm.rope_w;
    defer tri_norm.rope_w = keep;
    tri_norm.rope_w = true;
    var lays: [40]weights.Layer = undefined;
    try checkpointLayers(&lays);
    const gpa = std.testing.allocator;
    const qs = [_]usize{ 256, 512, 1024, 512 };
    var n: usize = 0;
    try std.testing.expectEqual(@as(?[]const u8, null), try idxProjOwnedOff(gpa, FakeFinds{ .asked = &n }, true, true, &lays, 20, 64, 32, 128, &qs, 4));
    // the fallback itself missing still refuses
    try std.testing.expectEqualStrings("this rank's kernel set has no _rope_heads / _fp4qd_p2 variant for the owned rows", (try idxProjOwnedOff(gpa, FakeFinds{ .asked = &n, .missing = "_rope_heads" }, true, true, &lays, 20, 64, 32, 128, &qs, 4)) orelse "can");
}

test "the owned routing's image spans: every span row routed with the image bias by its owner alone, as routeSpans routes it" {
    const cap = 64;
    const q = cap / 4;
    // spans over the chunk's rows (chunk start 100): inside one owner, across one boundary, across three, at the ends
    const spans = [_]Span{ .{ .at = 102, .len = 5, .row = 0 }, .{ .at = 110, .len = 12, .row = 0 }, .{ .at = 120, .len = 40, .row = 0 }, .{ .at = 90, .len = 12, .row = 0 }, .{ .at = 160, .len = 10, .row = 0 } };
    var whole: [cap]u8 = @splat(0);
    var owned: [cap]u8 = @splat(0);
    for (spans) |sp| {
        const o = overlap(sp, 100, cap) orelse continue;
        for (o[0]..o[1]) |i| whole[i] += 1;
        for (0..4) |k| {
            const r: QRows = .{ .start = k * q, .n = q };
            const in = rowsIn(o, r) orelse continue;
            try std.testing.expect(in[0] >= r.start and in[1] <= r.start + r.n and in[0] < in[1]);
            for (in[0]..in[1]) |i| owned[i] += 1;
        }
    }
    try std.testing.expectEqualSlices(u8, &whole, &owned);
    try std.testing.expect(std.mem.indexOfScalar(u8, &whole, 1) != null); // the spans reach the chunk
    // the whole chunk's rows give the span's own rows back
    try std.testing.expectEqual([2]usize{ 2, 7 }, rowsIn(.{ 2, 7 }, .{ .start = 0, .n = cap }).?);
    try std.testing.expect(rowsIn(.{ 2, 7 }, .{ .start = 16, .n = 16 }) == null);
}

test "the owned routing's text-bias routing is the whole chunk's _route launch at the owned rows' offsets" {
    var log: tri.Log = .{ .gpa = std.testing.allocator };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    var lay: weights.Layer = .{ .idx = 0, .ratio = 2 };
    lay.gate_b = 0x5000;
    const logits: u64 = 0x100000;
    const pick: u64 = 0x900000;
    const wts: u64 = 0xa00000;
    try tri_norm.route(t, logits, 0, lay.gate_b, 6, 1.5, 384, pick, wts, 2048, 384, 7); // moe's whole-chunk launch
    try ffnRoute(t, lay, logits, pick, wts, .{ .start = 1536, .n = 512 }, 384, 6, 1.5, 384, 7);
    try std.testing.expectEqual(@as(usize, 2), log.items.items.len);
    const all = log.items.items[0];
    const own = log.items.items[1];
    try std.testing.expectEqualStrings(all.name, own.name);
    try std.testing.expectEqual([3]u32{ 512, 1, 1 }, own.grid);
    try std.testing.expectEqualDeep(all.consts, own.consts);
    try std.testing.expectEqual(all.args.len, own.args.len);
    for (all.args, own.args) |a, b| {
        try std.testing.expectEqualStrings(a.name, b.name);
        const row: u64 = if (std.mem.eql(u8, a.name, "L")) 384 * 4 else if (std.mem.eql(u8, a.name, "PICK") or std.mem.eql(u8, a.name, "WTS")) 7 * 4 else 0;
        switch (a.value) {
            .ptr => |pa| try std.testing.expectEqual(pa.addr + 1536 * row, b.value.ptr.addr),
            else => try std.testing.expectEqualDeep(a.value, b.value),
        }
    }
}

test "the Engram gate at a row range is the whole chunk's _engram_gate launch at the rows' offsets, and the baseline at every row" {
    var log: tri.Log = .{ .gpa = std.testing.allocator };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    const d = 5120;
    const h_row = 4 * d * 2;
    const kv_row = 5 * d * 2;
    const h: u64 = 0x100000;
    const kv: u64 = 0x4000000;
    const out: u64 = 0x9000000;
    const qk: u64 = 0x7000;
    try tri_basic.engramGate(t, h, kv, qk, out, 1e-20, 2048, d); // engramApply's whole-chunk launch
    try engramGateRows(t, qk, h, kv, out, .{ .start = 0, .n = 2048 }, 1e-20, d, h_row, kv_row);
    try engramGateRows(t, qk, h, kv, out, .{ .start = 1536, .n = 512 }, 1e-20, d, h_row, kv_row);
    try std.testing.expectEqual(@as(usize, 3), log.items.items.len);
    const all = log.items.items[0];
    // every row: the baseline launch, argument for argument
    try std.testing.expectEqualDeep(all, log.items.items[1]);
    // owned rows: the same kernel and constexprs, grid 512, H, KV and OUT moved by 1,536 rows
    const own = log.items.items[2];
    try std.testing.expectEqualStrings(all.name, own.name);
    try std.testing.expectEqual([3]u32{ 512, 4, 1 }, own.grid);
    try std.testing.expectEqualDeep(all.consts, own.consts);
    try std.testing.expectEqual(all.args.len, own.args.len);
    for (all.args, own.args) |a, b| {
        try std.testing.expectEqualStrings(a.name, b.name);
        const row: u64 = if (std.mem.eql(u8, a.name, "H") or std.mem.eql(u8, a.name, "OUT")) h_row else if (std.mem.eql(u8, a.name, "KV")) kv_row else 0;
        switch (a.value) {
            .ptr => |pa| try std.testing.expectEqual(pa.addr + 1536 * row, b.value.ptr.addr),
            else => try std.testing.expectEqualDeep(a.value, b.value),
        }
    }
}

test "the boundary's in-place gathers put every rank's owned groups of the four cache arrays where the whole chunk's update puts them" {
    const n = 16;
    const world = 4;
    const q = n / world;
    const hd = 64;
    const id = 32;
    const start = 8; // the chunk's first group (a group a row at compress ratio 1)
    const li = 20;
    var cs: Caches = undefined;
    cs.comp_codes[li] = 0;
    cs.comp_scales[li] = 4096;
    cs.idx_codes[li] = 8192;
    cs.idx_scales[li] = 12288;
    const arrays = [_]u64{ cs.comp_codes[li], cs.comp_scales[li], cs.idx_codes[li], cs.idx_scales[li] };
    const rows = [_]usize{ hd / 2, hd / 16, id / 2, id / 32 }; // bytes a group
    const whole = struct {
        fn byte(buf: u64, i: usize) u8 {
            return @truncate(buf / 3 + i * 17 + 1);
        }
    }.byte;
    // rank k holds the whole update's bytes at its own groups, nothing else
    var mem: [world][16384]u8 = undefined;
    var parts: [world][4]comm.InPlace = undefined;
    for (&mem, &parts, 0..) |*m, *pk, k| {
        @memset(m, 0xee);
        for (arrays, rows) |buf, row| for ((start + k * q) * row..(start + k * q + q) * row) |i| {
            m[buf + i] = whole(buf, i);
        };
        pk.* = boundaryGathers(&cs, li, start, .{ .start = k * q, .n = q }, hd, id);
    }
    for (0..4) |j| {
        for (parts) |pk| try std.testing.expectEqualDeep(parts[0][j], pk[j]);
        const pt = parts[0][j];
        var recv: [16384]u8 = undefined;
        for (0..world) |k| @memcpy(recv[k * pt.bytes ..][0..pt.bytes], mem[k][pt.buf + k * pt.bytes ..][0..pt.bytes]);
        for (&mem) |*m| @memcpy(m[pt.buf..][0 .. world * pt.bytes], recv[0 .. world * pt.bytes]);
    }
    // every rank: the chunk's groups of every array hold the whole update's bytes, the groups around them untouched
    for (mem) |m| for (arrays, rows) |buf, row| {
        for (start * row..(start + n) * row) |i| try std.testing.expectEqual(whole(buf, i), m[buf + i]);
        for (0..start * row) |i| try std.testing.expectEqual(@as(u8, 0xee), m[buf + i]);
        for ((start + n) * row..(start + n + 4) * row) |i| try std.testing.expectEqual(@as(u8, 0xee), m[buf + i]);
    };
}

test "Engram's gather buffer holds engramSum's parts, a small chunk's quarters and the owner sums' stage" {
    const s: EkgSplit = .{ .ew = .{ 12800, 12800 }, .p = 0, .world = 4 };
    // a 2D node at 2,048 rows: engramSum's 2,048 x (6 + 2) x 12,800 bytes, half the baseline [2, 2,048, 25,600] fp32
    try std.testing.expectEqual(@as(usize, 209_715_200), ekgBytes(2048, 2, 25600, s));
    // under 64 rows the quarters' [2, 63, 25,600] fp32 decide
    try std.testing.expectEqual(@as(usize, 2 * 63 * 25600 * 4), ekgBytes(64, 2, 25600, s));
    // unequal pairs: this node's own share of the six and the two
    try std.testing.expectEqual(@as(usize, 2048 * (6 * 12928 + 2 * 12672)), ekgBytes(2048, 2, 25600, .{ .ew = .{ 12928, 12672 }, .p = 0, .world = 4 }));
    try std.testing.expectEqual(@as(usize, 2048 * (6 * 12672 + 2 * 12928)), ekgBytes(2048, 2, 25600, .{ .ew = .{ 12928, 12672 }, .p = 1, .world = 4 }));
    // the owner sums' stage fits: 512 owned rows of 12,800 at 14 bytes
    try std.testing.expect(prompt2d.OwnStage.bytes(512, 12800) <= ekgBytes(2048, 2, 25600, s));
    // TP2: both ranks' projections, as in the baseline
    try std.testing.expectEqual(@as(usize, 2 * 2048 * 12800 * 4), ekgBytes(2048, 2, 12800, null));
}

test "the indexer's key bound follows the context: 1M is the baseline's, 2M passes where 1M refuses" {
    // served 1M context: max_comp 1,048,576 -> exactly key_elems, the same buffers and the same refusals as before
    try std.testing.expectEqual(key_elems, keyCapFor(1_048_576));
    try std.testing.expectEqual(key_elems, keyCapFor(524_288));
    try std.testing.expectEqual(key_elems, keyCapFor(0));
    // a ratio-1 layer's last chunk at n_comp 1,048,577: the 16-row floor needs 16 * n_comp keys
    const floor = key_rows_floor;
    try std.testing.expect(!keysRefused(floor, 1_048_576, keyCapFor(1_048_576)));
    try std.testing.expect(keysRefused(floor, 1_048_577, keyCapFor(1_048_576))); // the 1M chunk refuses past 2^20
    try std.testing.expect(keysRefused(floor, 2_097_152, keyCapFor(1_048_576))); // the baseline bound refuses at 2M
    try std.testing.expect(!keysRefused(floor, 2_097_152, keyCapFor(2_097_152))); // the 2M chunk takes it
    try std.testing.expectEqual(@as(usize, 1) << 25, keyCapFor(2_097_152));
    // the 2^26 / 4 bound still sizes bigger blocks where they fit: rb at n_comp 600,000 is 16 * floor(2^26 / (4 * 600000) / 16)
    const n: usize = 600_000;
    const rb = @max(floor, (@as(usize, 1) << 26) / (4 * n) / 16 * 16);
    try std.testing.expect(!keysRefused(rb, n, keyCapFor(1_048_576)));
}

test "a chunk's decoder half reads the window's rows at most" {
    try std.testing.expectEqual(@as(usize, 128), decoderRows(2048, 128));
    try std.testing.expectEqual(@as(usize, 128), decoderRows(4096, 128));
    try std.testing.expectEqual(@as(usize, 64), decoderRows(64, 128));
}

test "a chunk's view of a pass stays within the rows its buffers hold" {
    var ch: Chunk = undefined;
    ch.rows_max = 4096;
    ch.cap = 4096;
    try ch.view(2048, 2048);
    try std.testing.expectEqual(@as(usize, 2048), ch.cap);
    try ch.view(4096, 2048);
    try std.testing.expectEqual(@as(usize, 4096), ch.cap);
    try ch.view(17, 2048);
    try std.testing.expectEqual(@as(usize, 2048), ch.cap);
    try std.testing.expectError(error.ChunkTooLong, ch.view(6144, 2048));
    // a chunk of the resume rows alone (TF_DS_PASS_ROWS unset): every pass's view is the baseline cap
    ch.rows_max = 2048;
    try ch.view(2048, 2048);
    try std.testing.expectEqual(@as(usize, 2048), ch.cap);
    try std.testing.expectError(error.ChunkTooLong, ch.view(4096, 2048));
}

test "a pass's view: its own rows when whole, the resume rows under them" {
    try std.testing.expectEqual(@as(usize, 2048), viewRows(2048, 2048));
    try std.testing.expectEqual(@as(usize, 4096), viewRows(4096, 2048));
    try std.testing.expectEqual(@as(usize, 2048), viewRows(2047, 2048));
    try std.testing.expectEqual(@as(usize, 2048), viewRows(1, 2048));
}

test "a prompt chunk's host row tables take an 8,192-row pass: no 4,096-row stack caps a pass" {
    const src = @embedFile("prompt.zig");
    // the needle split so this test's own text does not match
    const cap: []const u8 = "[4096]i64";
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, src, "var vis_host: " ++ cap));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, src, "var g_host: " ++ cap));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, src, "var p_host: " ++ cap));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, src, "[" ++ "max_pass_rows]i64 = undefined"));
    try std.testing.expect(std.mem.indexOf(u8, src, "hostRows(ch,") != null);
}

test "TF_DS_ATTN_U moves a prompt chunk's sparse attention only: its call asks for the twin, a round's and a pass's never" {
    const src = @embedFile("prompt.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "        .one_split = one_split,\n        .u = tri_attn.attn_u,\n        .epi = if (epi and tri_attn.attn_epi_fused)"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, @embedFile("round.zig"), "tri_attn.attn_u,"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, @embedFile("draft.zig"), "attn_u"));
    // TF_DS_ATTN_U_DECODE: a round's call alone (the drafter's passes read bf16 compressed keys)
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, @embedFile("round.zig"), "        .parts = .{ .pm = rd.pm_, .pl = rd.pl_, .po = rd.po_ },\n        .u = tri_attn.attn_u_decode,\n    });"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, @embedFile("round.zig"), "attn_u_decode"));
}

test "the pruned top-k in one launch reads tile maxima its chunk's index score wrote from the keys: unique, no dead tile padded" {
    // indexer: the keyed index score of a row block, then tf_dsn_topk_tiles on the same keys and maxima
    const src = @embedFile("prompt.zig");
    const at = std.mem.indexOf(u8, src, "\nfn indexer(e: *const Engine").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
    const score = std.mem.indexOf(u8, body, "try tri_index.indexScore(e.t, q, k, wb, vr, n_comp_end, ch.keys, null, true, ch.tmax, false, m, ih, id);").?;
    const tiles = std.mem.indexOf(u8, body, "try mx.topkTiles(e.mx.?, e.s, ch.keys, n_comp_end, ch.tmax, n_comp_end, kk, vr, out, m);").?;
    try std.testing.expect(score < tiles);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "mx.topkTiles("));
    // index_score's launches for a prompt chunk: _index_score with DEAD_PAD off (on, every dead tile's maximum one
    // constant: tied maxima), _index_score_rows with none
    const ti = @embedFile("tri_index.zig");
    const fs = std.mem.indexOf(u8, ti, "pub fn indexScore(").?;
    const fb = ti[fs .. fs + std.mem.indexOf(u8, ti[fs..], "\n}\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, fb, "cb(\"DEAD_PAD\","));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, fb, "cb(\"DEAD_PAD\", false)"));
}

fn expectWorkerOrder(order: *WorkerOrder, expected: []const WorkerStep) !void {
    var i: usize = 0;
    while (order.next()) |step| : (i += 1) {
        try std.testing.expect(i < expected.len);
        try std.testing.expectEqual(expected[i].chunk, step.chunk);
        try std.testing.expectEqual(expected[i].layer, step.layer);
    }
    try std.testing.expectEqual(expected.len, i);
}

test "TF_DS_ENGRAM_AHEAD_PASS=1: a 4,096-row pass reads both chunks of a layer before the pass's next layer" {
    var order: WorkerOrder = .{ .start = 0, .chunks = 4, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 4096 };
    try expectWorkerOrder(&order, &.{
        .{ .chunk = 0, .layer = 0 }, .{ .chunk = 1, .layer = 0 },
        .{ .chunk = 0, .layer = 1 }, .{ .chunk = 1, .layer = 1 },
        .{ .chunk = 2, .layer = 0 }, .{ .chunk = 3, .layer = 0 },
        .{ .chunk = 2, .layer = 1 }, .{ .chunk = 3, .layer = 1 },
    });
}

test "TF_DS_ENGRAM_AHEAD_PASS off (the default): a 2,048-row pass keeps the baseline chunk order" {
    var order: WorkerOrder = .{ .start = 0, .chunks = 2, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 2048 };
    try expectWorkerOrder(&order, &.{
        .{ .chunk = 0, .layer = 0 }, .{ .chunk = 0, .layer = 1 },
        .{ .chunk = 1, .layer = 0 }, .{ .chunk = 1, .layer = 1 },
    });
}

test "TF_DS_ENGRAM_AHEAD_PASS=1: a tail pass of one chunk keeps that chunk's two layers together" {
    var order: WorkerOrder = .{ .start = 0, .chunks = 3, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 4096 };
    try expectWorkerOrder(&order, &.{
        .{ .chunk = 0, .layer = 0 }, .{ .chunk = 1, .layer = 0 },
        .{ .chunk = 0, .layer = 1 }, .{ .chunk = 1, .layer = 1 },
        .{ .chunk = 2, .layer = 0 }, .{ .chunk = 2, .layer = 1 },
    });
    var empty: WorkerOrder = .{ .start = 0, .chunks = 0, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 4096 };
    try std.testing.expectEqual(@as(?WorkerStep, null), empty.next());
}

test "TF_DS_PASS_FROM=8192, TF_DS_PASS_ROWS=4096: the fill's first four 2,048-row chunks are chunk-major, then 4,096-row passes" {
    var order: WorkerOrder = .{ .start = 0, .chunks = 10, .chunk = 2048, .layers = 2, .pass_from = 8192, .pass_rows = 4096 };
    try expectWorkerOrder(&order, &.{
        .{ .chunk = 0, .layer = 0 }, .{ .chunk = 0, .layer = 1 },
        .{ .chunk = 1, .layer = 0 }, .{ .chunk = 1, .layer = 1 },
        .{ .chunk = 2, .layer = 0 }, .{ .chunk = 2, .layer = 1 },
        .{ .chunk = 3, .layer = 0 }, .{ .chunk = 3, .layer = 1 },
        .{ .chunk = 4, .layer = 0 }, .{ .chunk = 5, .layer = 0 },
        .{ .chunk = 4, .layer = 1 }, .{ .chunk = 5, .layer = 1 },
        .{ .chunk = 6, .layer = 0 }, .{ .chunk = 7, .layer = 0 },
        .{ .chunk = 6, .layer = 1 }, .{ .chunk = 7, .layer = 1 },
        .{ .chunk = 8, .layer = 0 }, .{ .chunk = 9, .layer = 0 },
        .{ .chunk = 8, .layer = 1 }, .{ .chunk = 9, .layer = 1 },
    });
}

test "TF_DS_PASS_FROM=8192, TF_DS_PASS_ROWS=8192: 8,192-row passes read four chunks a layer at a time" {
    var order: WorkerOrder = .{ .start = 0, .chunks = 12, .chunk = 2048, .layers = 2, .pass_from = 8192, .pass_rows = 8192 };
    try expectWorkerOrder(&order, &.{
        .{ .chunk = 0, .layer = 0 },  .{ .chunk = 0, .layer = 1 },
        .{ .chunk = 1, .layer = 0 },  .{ .chunk = 1, .layer = 1 },
        .{ .chunk = 2, .layer = 0 },  .{ .chunk = 2, .layer = 1 },
        .{ .chunk = 3, .layer = 0 },  .{ .chunk = 3, .layer = 1 },
        .{ .chunk = 4, .layer = 0 },  .{ .chunk = 5, .layer = 0 },
        .{ .chunk = 6, .layer = 0 },  .{ .chunk = 7, .layer = 0 },
        .{ .chunk = 4, .layer = 1 },  .{ .chunk = 5, .layer = 1 },
        .{ .chunk = 6, .layer = 1 },  .{ .chunk = 7, .layer = 1 },
        .{ .chunk = 8, .layer = 0 },  .{ .chunk = 9, .layer = 0 },
        .{ .chunk = 10, .layer = 0 }, .{ .chunk = 11, .layer = 0 },
        .{ .chunk = 8, .layer = 1 },  .{ .chunk = 9, .layer = 1 },
        .{ .chunk = 10, .layer = 1 }, .{ .chunk = 11, .layer = 1 },
    });
}

test "TF_DS_PASS_ROWS at every size: a pass of u resume units reads its u chunks a layer at a time, layer 0's then layer 1's, pass after pass" {
    for (1..engram_rows.max_units + 1) |u| {
        var order: WorkerOrder = .{ .start = 0, .chunks = 3 * u, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = u * 2048 };
        for (0..3) |pass| for (0..2) |layer| for (0..u) |k| {
            const step = order.next() orelse return error.TestUnexpectedResult;
            try std.testing.expectEqual(pass * u + k, step.chunk);
            try std.testing.expectEqual(layer, step.layer);
        };
        try std.testing.expect(order.next() == null);
    }
}

test "TF_DS_ROCI_STAGGER at every size: the pair's group is both passes' 2u chunks, layer 0's then layer 1's, pair after pair" {
    for (1..engram_rows.max_units + 1) |u| {
        var order: WorkerOrder = .{ .start = 0, .chunks = 4 * u, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = u * 2048, .group = 2 * u };
        for (0..2) |pair| for (0..2) |layer| for (0..2 * u) |k| {
            const step = order.next() orelse return error.TestUnexpectedResult;
            try std.testing.expectEqual(pair * 2 * u + k, step.chunk);
            try std.testing.expectEqual(layer, step.layer);
        };
        try std.testing.expect(order.next() == null);
    }
}

test "TF_DS_PASS_FILLING: two fills alternating 4,096-row passes read each pass two chunks a layer, its own slots" {
    // a fill from the prompt's first row and one resumed at 4,096 (an alternating prompt's next pass): each pass is
    // two units, layer-major from its own start, and a pass's two chunks take the worker's two ring slots (wk_rows[2])
    for ([_]usize{ 0, 4096 }) |start| {
        var order: WorkerOrder = .{ .start = start, .chunks = 4, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 4096, .filling_pass = true };
        var seen: [2][2]bool = @splat(@splat(false));
        var i: usize = 0;
        while (i < 8) : (i += 1) {
            const step = order.next().?;
            const pass = i / 4;
            const layer = (i % 4) / 2;
            const k = i % 2;
            try std.testing.expectEqual(layer, step.layer);
            try std.testing.expectEqual(pass * 2 + k, step.chunk);
            // a pass's two chunks sit in the two slots, per layer
            const slot = step.chunk % 2;
            try std.testing.expect(!seen[step.layer][slot]);
            seen[step.layer][slot] = true;
            if (k == 1) seen[step.layer] = @splat(false);
        }
        try std.testing.expect(order.next() == null);
    }
}

/// The worker (greedy, blocked by its slot guard) against the host consuming a pair's units in the pair's issue order: A's layer-1
/// Engram units, B's, A's layer-14, B's. Returns how many of the worker's steps were done when the host first consumed B's layer-14
/// units, and whether the host ever waited on a step the worker could not take (a deadlock).
fn pairWorkerRun(slots: usize, pairs: usize) struct { ok: bool, ahead_at_b14: usize } {
    var order: WorkerOrder = .{ .start = 0, .chunks = pairs * 4, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 4096 };
    var done = [2]usize{ 0, 0 };
    var used = [2]usize{ 0, 0 };
    var steps: usize = 0;
    var ahead: usize = 0;
    var have_ahead = false;
    var pending: ?WorkerStep = null;
    const total = pairs * 8;
    var host: usize = 0; // consumptions: 4 a pair (A j0, B j0, A j1, B j1), each 2 chunks
    while (host < pairs * 4) {
        // the worker takes every step its guard allows
        while (steps < total) {
            const st = pending orelse (order.next() orelse break);
            if (st.chunk >= used[st.layer] + slots) {
                pending = st;
                break;
            }
            pending = null;
            done[st.layer] = st.chunk + 1;
            steps += 1;
        }
        const pair = host / 4;
        const which = host % 4; // 0 A j0, 1 B j0, 2 A j1, 3 B j1
        const j: usize = which / 2;
        const first = pair * 4 + (which % 2) * 2;
        if (done[j] < first + 2) return .{ .ok = false, .ahead_at_b14 = ahead };
        used[j] = first + 2;
        if (which == 3 and !have_ahead) {
            ahead = steps;
            have_ahead = true;
        }
        host += 1;
    }
    return .{ .ok = true, .ahead_at_b14 = ahead };
}

test "two passes in flight: the worker's four chunk slots serve a pair's four units in the issue order without a wait on a step it cannot take" {
    const four = pairWorkerRun(4, 8);
    try std.testing.expect(four.ok);
    // the first pair's B layer-14 units are consumed when the worker is already into the next pair
    try std.testing.expect(four.ahead_at_b14 >= 8);
}

test "control: two slots hold the worker back to the pair's own units (the guard is live)" {
    const two = pairWorkerRun(2, 8);
    const four = pairWorkerRun(4, 8);
    try std.testing.expect(two.ahead_at_b14 < four.ahead_at_b14 or !two.ok);
}

test "TF_DS_ROCI_WORKER_PAIR_MAJOR: a group of four chunks reads layer 0 of all four, then layer 1 of all four, so the second pass's layer 0 comes after four steps, not six" {
    // the baseline order for 4,096-row passes: pass A's layers, then pass B's
    var today: WorkerOrder = .{ .start = 0, .chunks = 4, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 4096 };
    try expectWorkerOrder(&today, &.{
        .{ .chunk = 0, .layer = 0 }, .{ .chunk = 1, .layer = 0 }, .{ .chunk = 0, .layer = 1 }, .{ .chunk = 1, .layer = 1 },
        .{ .chunk = 2, .layer = 0 }, .{ .chunk = 3, .layer = 0 }, .{ .chunk = 2, .layer = 1 }, .{ .chunk = 3, .layer = 1 },
    });
    var major: WorkerOrder = .{ .start = 0, .chunks = 8, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 4096, .group = 4 };
    try expectWorkerOrder(&major, &.{
        .{ .chunk = 0, .layer = 0 }, .{ .chunk = 1, .layer = 0 }, .{ .chunk = 2, .layer = 0 }, .{ .chunk = 3, .layer = 0 },
        .{ .chunk = 0, .layer = 1 }, .{ .chunk = 1, .layer = 1 }, .{ .chunk = 2, .layer = 1 }, .{ .chunk = 3, .layer = 1 },
        .{ .chunk = 4, .layer = 0 }, .{ .chunk = 5, .layer = 0 }, .{ .chunk = 6, .layer = 0 }, .{ .chunk = 7, .layer = 0 },
        .{ .chunk = 4, .layer = 1 }, .{ .chunk = 5, .layer = 1 }, .{ .chunk = 6, .layer = 1 }, .{ .chunk = 7, .layer = 1 },
    });
    // the steps done when the host has its first pass's and its second pass's layer-0 rows
    var o1: WorkerOrder = .{ .start = 0, .chunks = 8, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 4096 };
    var o2 = o1;
    o2.group = 4;
    var needs: [2]usize = .{ 0, 0 };
    for ([_]*WorkerOrder{ &o1, &o2 }, 0..) |o, i| {
        var n: usize = 0;
        while (o.next()) |st| {
            n += 1;
            if (st.chunk == 3 and st.layer == 0) break;
        }
        needs[i] = n;
    }
    try std.testing.expectEqual(@as(usize, 6), needs[0]);
    try std.testing.expectEqual(@as(usize, 4), needs[1]);
    // a short last group takes what is left; below pass_from the rule's chunk-major order stays
    var tail: WorkerOrder = .{ .start = 0, .chunks = 6, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = 4096, .group = 4 };
    var seen: usize = 0;
    while (tail.next()) |_| seen += 1;
    try std.testing.expectEqual(@as(usize, 12), seen);
    var below: WorkerOrder = .{ .start = 0, .chunks = 4, .chunk = 2048, .layers = 2, .pass_from = 8192, .pass_rows = 4096, .group = 4 };
    try expectWorkerOrder(&below, &.{
        .{ .chunk = 0, .layer = 0 }, .{ .chunk = 0, .layer = 1 }, .{ .chunk = 1, .layer = 0 }, .{ .chunk = 1, .layer = 1 },
        .{ .chunk = 2, .layer = 0 }, .{ .chunk = 2, .layer = 1 }, .{ .chunk = 3, .layer = 0 }, .{ .chunk = 3, .layer = 1 },
    });
}

/// The worker's order (groups of `per` passes of `units` resume units, layer by layer) against the engine's uploads on a ring of
/// `slots` chunk slots, as engramWorker's guard and engramApply's releases run them: a pass's layer-j units are uploaded together
/// (wk_used[j] = the pass's end) once the worker has read them all. True when the worker and the uploads both reach the end;
/// false when neither can move (the worker waits for a slot the uploads have not freed, the uploads for rows the worker has not read).
fn ringRuns(units: usize, per: usize, slots: usize, groups: usize) bool {
    const chunk = 2048;
    var order: WorkerOrder = .{ .start = 0, .chunks = groups * per * units, .chunk = chunk, .layers = 2, .pass_from = 0, .pass_rows = units * chunk, .group = per * units };
    var done = [2]usize{ 0, 0 };
    var used = [2]usize{ 0, 0 };
    var next = order.next();
    var g: usize = 0; // the group the uploads are at, its layer and its pass
    var j: usize = 0;
    var q: usize = 0;
    while (g < groups or next != null) {
        var moved = false;
        if (next) |st| if (st.chunk < used[st.layer] + slots) {
            done[st.layer] = st.chunk + 1;
            next = order.next();
            moved = true;
        };
        if (g < groups) {
            const end = g * per * units + (q + 1) * units;
            if (done[j] >= end) {
                used[j] = end;
                q += 1;
                if (q == per) {
                    q = 0;
                    j += 1;
                    if (j == 2) {
                        j = 0;
                        g += 1;
                    }
                }
                moved = true;
            }
        }
        if (!moved) return false;
    }
    return true;
}

test "the worker's ring: a pass's units are the fewest slots that do not hang it, twice them serve two passes in flight, at every pass size from 2,048 to 16,384 rows" {
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8 }) |units| {
        for ([_]usize{ 1, 2 }) |per| {
            // fewer slots than a pass's units never lets a pass finish (a silent hang in the engine)
            for (1..units) |slots| try std.testing.expect(!ringRuns(units, per, slots, 3));
            // a pass's units and more, up to the largest ring, run
            for (units..engram_rows.max_wk + 1) |slots| try std.testing.expect(ringRuns(units, per, slots, 3));
        }
        // the slots the engine boots: without the stagger and with it
        try std.testing.expect(ringRuns(units, 1, engram_rows.workerSlots(units, false), 3));
        try std.testing.expect(ringRuns(units, 2, engram_rows.workerSlots(units, true), 3));
        var why: [320]u8 = undefined;
        try std.testing.expectEqual(@as(?[]const u8, null), engram_rows.slotsRefusal(&why, engram_rows.workerSlots(units, true), units));
        try std.testing.expectEqual(@as(?[]const u8, null), engram_rows.slotsRefusal(&why, engram_rows.workerSlots(units, false), units));
    }
}

test "TF_DS_ROCI_STAGGER_LANES=3: a group of three passes' 3u chunks, layer 0's then layer 1's, and any ring of a pass's units or more runs it against the uploads (no deadlock); the booted ring is three passes' units" {
    for (1..engram_rows.max_units + 1) |u| {
        var order: WorkerOrder = .{ .start = 0, .chunks = 6 * u, .chunk = 2048, .layers = 2, .pass_from = 0, .pass_rows = u * 2048, .group = 3 * u };
        for (0..2) |group| for (0..2) |layer| for (0..3 * u) |k| {
            const step = order.next() orelse return error.TestUnexpectedResult;
            try std.testing.expectEqual(group * 3 * u + k, step.chunk);
            try std.testing.expectEqual(layer, step.layer);
        };
        try std.testing.expect(order.next() == null);
        const slots = engram_rows.workerSlotsN(u, 3);
        try std.testing.expectEqual(@max(2, 3 * u), slots);
        var why: [320]u8 = undefined;
        if (slots <= engram_rows.max_wk) {
            try std.testing.expect(ringRuns(u, 3, slots, 3));
            try std.testing.expectEqual(@as(?[]const u8, null), engram_rows.slotsRefusal(&why, slots, u));
        } else {
            // the ring would be larger than the worker's largest: the start refuses it (units 6 to 8 at three lanes)
            try std.testing.expect(engram_rows.slotsRefusal(&why, slots, u) != null);
        }
        // a ring under a pass's units never runs (the worker waits for a slot no upload frees), from the pass's units up to the largest it does
        for (1..u) |sl| try std.testing.expect(!ringRuns(u, 3, sl, 3));
        for (u..engram_rows.max_wk + 1) |sl| try std.testing.expect(ringRuns(u, 3, sl, 3));
    }
    try std.testing.expectEqual(@as(usize, 6), engram_rows.workerSlotsN(2, 3)); // 4,096-row passes: six chunk slots
    try std.testing.expectEqual(engram_rows.workerSlots(2, true), engram_rows.workerSlotsN(2, 2));
    try std.testing.expectEqual(engram_rows.workerSlots(2, false), engram_rows.workerSlotsN(2, 1));
}

test "TF_DS_PASS_FROM is tested at the fill's own start: a fill resumed at 8,192 starts pass-major" {
    var order: WorkerOrder = .{ .start = 8192, .chunks = 4, .chunk = 2048, .layers = 2, .pass_from = 8192, .pass_rows = 4096 };
    try expectWorkerOrder(&order, &.{
        .{ .chunk = 0, .layer = 0 }, .{ .chunk = 1, .layer = 0 },
        .{ .chunk = 0, .layer = 1 }, .{ .chunk = 1, .layer = 1 },
        .{ .chunk = 2, .layer = 0 }, .{ .chunk = 3, .layer = 0 },
        .{ .chunk = 2, .layer = 1 }, .{ .chunk = 3, .layer = 1 },
    });
    // a fill resumed at 4,096 is chunk-major for its two chunks below pass_from and pass-major from chunk 2
    var mid: WorkerOrder = .{ .start = 4096, .chunks = 4, .chunk = 2048, .layers = 2, .pass_from = 8192, .pass_rows = 4096 };
    try expectWorkerOrder(&mid, &.{
        .{ .chunk = 0, .layer = 0 }, .{ .chunk = 0, .layer = 1 },
        .{ .chunk = 1, .layer = 0 }, .{ .chunk = 1, .layer = 1 },
        .{ .chunk = 2, .layer = 0 }, .{ .chunk = 3, .layer = 0 },
        .{ .chunk = 2, .layer = 1 }, .{ .chunk = 3, .layer = 1 },
    });
}

test "TF_DS_ENGRAM_AHEAD_PASS: the environment sets the switch and prints its start line when on" {
    // show the start line in the test's output when the switch is on (the engine's log level does so at start)
    const old = std.testing.log_level;
    std.testing.log_level = .info;
    defer std.testing.log_level = old;
    readAheadPassEnv();
    const v = std.c.getenv("TF_DS_ENGRAM_AHEAD_PASS");
    const on = if (v) |s| !std.mem.eql(u8, std.mem.span(s), "0") else false;
    try std.testing.expectEqual(on, ahead_pass_on);
    if (on) try std.testing.expect(std.mem.startsWith(u8, ahead_pass_line, "TF_DS_ENGRAM_AHEAD_PASS=1: "));
    try std.testing.expect(std.mem.startsWith(u8, ahead_pass_line, "TF_DS_ENGRAM_AHEAD_PASS=1: "));
}

/// Whether the kit's extra set has the two-block reader variant of every launch the owned path makes with TF_DS_ROCI at
/// each row count of `rows` (a slab of a full pass): null when it has, else the reason. The launches are logged, not
/// run (`finds.has`, as frontOwnedOff's).
pub fn rociReadersOff(gpa: std.mem.Allocator, finds: anytype, rows: []const usize, d: usize, ew: usize, eps: f32, hc_eps: f32, iters: usize) !?[]const u8 {
    var log: tri.Log = .{ .gpa = gpa };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    const base_h: u64 = 1 << 32;
    for (rows) |n| {
        const rs = n * (d / 2);
        _ = try tri_hc.hcPrePf(t, base_h, 2 << 32, 3 << 32, 4 << 32, 5 << 32, 6 << 32, eps, hc_eps, iters, 7 << 32, 8 << 32, 9 << 32, 10 << 32, 11 << 32, .{ .gathered = 12 << 32, .world = 2, .h_out = 13 << 32, .blk_rs = rs }, n, d);
        try tri_basic.hcPost2b(t, 12 << 32, rs, base_h, 9 << 32, 10 << 32, base_h, n, d);
        try tri_basic.engramGate2b(t, base_h, 14 << 32, n * ew, 15 << 32, 16 << 32, eps, n, d);
    }
    for (log.items.items) |l| {
        if (!l.extra) continue;
        if (!finds.has(l.name, l.args, l.consts)) return "this rank's extra kernel set has no two-block reader variant";
    }
    return null;
}

test "TF_DS_ROCI's readers check: every two-block reader launch of a full pass's slab found in the extra set, else named" {
    const gpa = std.testing.allocator;
    var n: usize = 0;
    try std.testing.expectEqual(@as(?[]const u8, null), try rociReadersOff(gpa, FakeFinds{ .asked = &n }, &[_]usize{ 512, 1024 }, 5120, 12800, 1e-20, 1e-6, 20));
    // two slab sizes, each: the mix (one extra launch), the post, the Engram gate
    try std.testing.expectEqual(@as(usize, 2 * 3), n);
    try std.testing.expect((try rociReadersOff(gpa, FakeFinds{ .asked = &n, .missing = "_hc_post_2b" }, &[_]usize{1024}, 5120, 12800, 1e-20, 1e-6, 20)) != null);
    try std.testing.expect((try rociReadersOff(gpa, FakeFinds{ .asked = &n, .missing = "_engram_gate_2b" }, &[_]usize{1024}, 5120, 12800, 1e-20, 1e-6, 20)) != null);
    try std.testing.expect((try rociReadersOff(gpa, FakeFinds{ .asked = &n, .missing = "_hc_mix_pf_2b" }, &[_]usize{1024}, 5120, 12800, 1e-20, 1e-6, 20)) != null);
}

test "TF_DS_ROCI_P2_FIRST: the second part's add and send come before the own block's add; off, both adds then the send" {
    const src = @embedFile("prompt.zig");
    const body = src[std.mem.indexOf(u8, src, "fn rociPart2(").?..];
    const on = body[std.mem.indexOf(u8, body, "if (e.two.?.p2First()) {").?..];
    const k1 = std.mem.indexOf(u8, on, "try rociAdd(e, rc, src, q, rows[1], w, 1,").?;
    const sd = std.mem.indexOf(u8, on, "try rc.send(e.s, class, shape);").?;
    const k0 = std.mem.indexOf(u8, on, "try rociAdd(e, rc, src, q, rows[0], w, 0,").?;
    try std.testing.expect(k1 < sd and sd < k0);
    const off = on[std.mem.indexOf(u8, on, "} else {").?..];
    const o0 = std.mem.indexOf(u8, off, "try rociAdd(e, rc, src, q, rows[0], w, 0,").?;
    const o1 = std.mem.indexOf(u8, off, "try rociAdd(e, rc, src, q, rows[1], w, 1,").?;
    const os = std.mem.indexOf(u8, off, "try rc.send(e.s, class, shape);").?;
    try std.testing.expect(o0 < o1 and o1 < os);
    // the wait comes after both
    try std.testing.expect(std.mem.indexOf(u8, body, "try rc.wait(e.s, class);").? > std.mem.indexOf(u8, body, "try rociAdd(e, rc, src, q, rows[0], w, 0,").?);
}

test "TF_DS_ROCI_COMBINE_FIRST: the sent quarters come first for every node and node order, the kept ones last, all four once" {
    const orders = [_][4]u32{ .{ 0, 1, 2, 3 }, .{ 2, 0, 3, 1 }, .{ 3, 2, 1, 0 } };
    var dummy: u8 = 0;
    const F = struct {
        fn f(_: *anyopaque) anyerror!void {}
    };
    for (orders) |rank_of| for (0..4) |g| {
        const pl = prompt2d.ownerPlan(@intCast(g), rank_of, 1024);
        const cf: mx.CombineFirst = .{ .rows = 1024, .first = pl.send1, .last = pl.recv1, .ctx = &dummy, .between = F.f };
        try std.testing.expect(cf.covers(4096));
        // what part 1 sends is the first stage's, what the adds read from this node's own partial is the last's
        try std.testing.expectEqual(@as(usize, rank_of[g ^ 1]) * 1024, cf.first[0]);
        try std.testing.expectEqual(@as(usize, rank_of[g ^ 3]) * 1024, cf.first[1]);
        try std.testing.expectEqual(@as(usize, rank_of[g]) * 1024, cf.last[0]);
        try std.testing.expectEqual(@as(usize, rank_of[g ^ 2]) * 1024, cf.last[1]);
    };
}

test "TF_DS_ROCI_ENGRAM_AHEAD: engramApply is the pending post, the projection, then the rest, in the baseline order; the early part sends only on the transport" {
    const src = @embedFile("prompt.zig");
    const ap = src[std.mem.indexOf(u8, src, "pub fn engramApply(").?..];
    const a = std.mem.indexOf(u8, ap, "try engramPending(e, ch, li);").?;
    const b = std.mem.indexOf(u8, ap, "try engramProject(e, ch, eh, li, seq);").?;
    const c = std.mem.indexOf(u8, ap, "try engramFinish(e, ch, eh, li, false);").?;
    try std.testing.expect(a < b and b < c and c < 400);
    // the projection reads no stream: it names neither the streams' buffers nor a gather
    const pr = src[std.mem.indexOf(u8, src, "pub fn engramProject(").?..std.mem.indexOf(u8, src, "pub fn engramOnRoci(").?];
    try std.testing.expect(std.mem.indexOf(u8, pr, "ch.h") == null and std.mem.indexOf(u8, pr, "ch.pending") == null);
    // the early send exists only where the sums are on the transport at owned rows
    const sa = src[std.mem.indexOf(u8, src, "pub fn engramSendAhead(").?..];
    try std.testing.expect(std.mem.indexOf(u8, sa, "engramOwned(e, ch, li) orelse return") != null);
}

/// A device copy of a pinned slot, as the stream runs it later than the host call: the host buffer is read when `run` is
/// called, not when `copy` is. The worker's rows are written into the slot whenever the slot is free (wk_used).
const FakeCopies = struct {
    host: [2]u32 = .{ 0, 0 }, // the slots' pinned bytes
    queued: [8]struct { slot: usize, at: usize, done: bool } = undefined,
    n: usize = 0,
    device: [8]u32 = @splat(0), // what each queued copy read
    wk_used: usize = 0,

    fn copy(f: *FakeCopies, slot: usize, chunk: usize) void {
        f.queued[f.n] = .{ .slot = slot, .at = chunk, .done = false };
        f.n += 1;
    }

    fn run(f: *FakeCopies, i: usize) void {
        f.device[f.queued[i].at] = f.host[f.queued[i].slot];
        f.queued[i].done = true;
    }

    // PinSlots' context: layer 0 only
    fn done(f: *FakeCopies, _: usize, p: usize) !bool {
        return f.queued[p].done;
    }

    fn release(f: *FakeCopies, _: usize, used: usize) void {
        f.wk_used = used;
    }
};

test "TF_DS_ENGRAM_PINNED: a slot goes back to the worker only after its copy has run, and one freed early is overwritten before the copy reads it" {
    // two slots, chunks 0..3: the worker writes chunk i into slot i % 2 once wk_used + 2 > i; the stream runs a chunk's copy
    // late (here: after the worker has had every chance to write)
    for ([_]bool{ false, true }) |early| {
        var f: FakeCopies = .{};
        var pins: PinSlots = .{};
        var chunk: usize = 0; // the worker's next chunk
        var uploaded: usize = 0;
        var runs: usize = 0;
        while (uploaded < 4) {
            // the worker: every chunk its slot allows
            while (chunk < 4 and chunk < f.wk_used + 2) : (chunk += 1) f.host[chunk % 2] = @intCast(100 + chunk);
            // the engine thread uploads the next chunk the worker has written and arms its slot
            if (uploaded < chunk) {
                const c = uploaded;
                f.copy(c % 2, c);
                if (early) f.wk_used = c + 1 else pins.arm(0, pins.room(0).?, c + 1);
                uploaded += 1;
                // (the queue index of a chunk's copy is its position in `queued`; PinSlots' position p is the same here)
                continue;
            }
            // the stream: runs its oldest copy not yet run, then the engine thread polls
            if (runs < f.n) {
                f.run(runs);
                runs += 1;
            }
            if (!early) _ = try pins.poll(0, &f);
        }
        while (runs < f.n) : (runs += 1) f.run(runs);
        var bad: usize = 0;
        for (0..4) |c| bad += @intFromBool(f.device[c] != 100 + c);
        // the proper release hands every chunk's bytes over; the early one lets the worker overwrite a slot before its copy
        if (early) try std.testing.expect(bad > 0) else try std.testing.expectEqual(@as(usize, 0), bad);
    }
}

test "PinSlots.highest: the largest count pending over a ring of four, so a pair's second pass uploaded but not yet given back is still the running worker's" {
    var s: PinSlots = .{ .ring = 4 };
    try std.testing.expectEqual(@as(usize, 0), s.highest(0));
    s.arm(0, s.room(0).?, 2); // pass A's units
    s.arm(0, s.room(0).?, 4); // pass B's
    try std.testing.expectEqual(@as(usize, 4), s.highest(0));
    try std.testing.expectEqual(@as(usize, 0), s.highest(1));
    // control: a ring of two sees only two positions
    var two: PinSlots = .{};
    two.arm(0, 0, 2);
    two.arm(0, 1, 4);
    try std.testing.expectEqual(@as(usize, 4), two.highest(0));
}

test "PinSlots: two pending a layer, the oldest first, a full ring has no room, a done copy frees only up to itself" {
    var s: PinSlots = .{};
    var f: FakeCopies = .{};
    try std.testing.expectEqual(@as(?usize, 0), s.room(0));
    f.copy(0, 0);
    s.arm(0, 0, 2);
    try std.testing.expectEqual(@as(?usize, 1), s.room(0));
    f.copy(1, 1);
    s.arm(0, 1, 4);
    try std.testing.expectEqual(@as(?usize, null), s.room(0)); // both pending
    try std.testing.expect(try s.poll(0, &f)); // none done
    try std.testing.expectEqual(@as(usize, 0), f.wk_used);
    f.run(1); // the newer copy done first (it cannot be, on one stream; the free is still in order)
    try std.testing.expect(try s.poll(0, &f));
    try std.testing.expectEqual(@as(usize, 0), f.wk_used);
    f.run(0);
    try std.testing.expect(!(try s.poll(0, &f)));
    try std.testing.expectEqual(@as(usize, 4), f.wk_used);
    try std.testing.expectEqual(@as(?usize, 0), s.room(0)); // (both free again; the ring position is the arm count mod 2)
}

test "TF_DS_ROCI_ANY: the transport's plan of a chunk is the fixed plan at a whole pass and the rows' plan otherwise, with the top-k width riding along" {
    try std.testing.expectEqual(@as(u8, 0), (try roci_engine.Roci.shapeFor(4096, 4096, true, 0)).id);
    try std.testing.expectEqual(@as(u8, 1), (try roci_engine.Roci.shapeFor(2048, 2048, true, 0)).id);
    const s = try roci_engine.Roci.shapeFor(500, 2048, true, 100);
    try std.testing.expectEqual(roci.any_shape, s.id);
    try std.testing.expectEqual(@as(u16, 500), s.n);
    try std.testing.expectEqual(@as(u16, 100), s.x);
    // off: the cap's fixed plan (the callers check that the chunk is a whole pass); no plan for a cap the transport does not plan
    try std.testing.expectEqual(@as(u8, 1), (try roci_engine.Roci.shapeFor(500, 2048, false, 0)).id);
    try std.testing.expectError(error.NoPlanForRows, roci_engine.Roci.shapeFor(500, 3000, false, 0));
    try std.testing.expectError(error.NoPlanForRows, roci_engine.Roci.shapeFor(0, 2048, true, 0));
}

test "TF_DS_ROCI_ANY: the boundary's copies stage each rank's rows in a q-row stride per array and stay inside the slab and the region" {
    const rb = boundaryRowBytes(512, 128);
    try std.testing.expectEqual(@as(usize, roci.gb_row), rb[0] + rb[1] + rb[2] + rb[3]);
    for ([_]usize{ 13, 500, 1087, 3000, 4096 }) |n| {
        const q = slab_plan.q(n);
        for (0..4) |k| {
            const nk = slab_plan.slab(n, k).n;
            const bufs = [4]u64{ 0x1000_0000, 0x2000_0000, 0x3000_0000, 0x4000_0000 };
            const outs = boundaryCopiesAny(bufs, rb, 0x9000_0000, q, k, nk, true);
            const ins = boundaryCopiesAny(bufs, rb, 0x9000_0000, q, k, nk, false);
            var at: u64 = 0x9000_0000 + k * q * roci.gb_row;
            for (outs, ins, 0..) |o, i, a| {
                try std.testing.expectEqual(at, o.dst);
                try std.testing.expectEqual(bufs[a] + k * q * rb[a], o.src);
                try std.testing.expectEqual(nk * rb[a], o.bytes);
                try std.testing.expectEqual(o.dst, i.src);
                try std.testing.expectEqual(o.src, i.dst);
                at += q * rb[a];
            }
            try std.testing.expectEqual(0x9000_0000 + (k + 1) * q * roci.gb_row, at);
        }
    }
}

test "TF_DS_ROCI_ANY with SENT_FIRST and COMBINE_DOOR: only a whole pass of a planned size takes the door launches, on every rank alike" {
    // a chunk of 2,040 rows in a 2,048-row pass: three ranks own 512 rows (= cap / 4), the fourth 504; the rank-local test r.n * 4 == cap
    // would split the ranks (three send from inside the GEMM, one not): the whole-pass test is the chunk's alone
    var ch: Chunk = undefined;
    ch.cap = 2048;
    ch.n = 2040;
    var split: usize = 0;
    for (0..4) |k| split += @intFromBool(slab_plan.slab(ch.n, k).n * 4 == ch.cap);
    try std.testing.expectEqual(@as(usize, 3), split);
    try std.testing.expect(!(ch.n == ch.cap and fixedPlan(&ch)));
    ch.n = 2048;
    try std.testing.expect(ch.n == ch.cap and fixedPlan(&ch));
    ch.cap = 4096;
    ch.n = 4096;
    try std.testing.expect(fixedPlan(&ch));
    // a tail pass (cap 2,236) is a whole chunk with no fixed plan: the plain path
    ch.cap = 2236;
    ch.n = 2236;
    try std.testing.expect(!fixedPlan(&ch));
    // and its owner exchanges take the rows' plan
    try std.testing.expectEqual(roci.any_shape, (try roci_engine.Roci.shapeFor(2236, 2236, true, 0)).id);
    const src = @embedFile("prompt.zig");
    const a = src[std.mem.indexOf(u8, src, "fn attnSentFirst(").?..std.mem.indexOf(u8, src, "pub fn mmSentFirst(").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, a, "!fixedPlan(ch)"));
    const m = src[std.mem.indexOf(u8, src, "fn moeSendFirst(").?..std.mem.indexOf(u8, src, "const MoeRing = struct").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, m, "ch.n == ch.cap and fixedPlan(ch)"));
}

test "TF_DS_ENGRAM_FAULT: R:C parses; anything else, or a rank past the world, is refused; unset is nothing" {
    const f = try EngramFault.parse("2:5", 4);
    try std.testing.expect(f.asked and f.rank == 2 and f.at == 5);
    try std.testing.expect(!(try EngramFault.parse(null, 4)).asked);
    try std.testing.expect(!(try EngramFault.parse("", 4)).asked);
    for ([_][]const u8{ "2", "2:", ":5", "x:5", "2:y", "4:0", "2:5:1", "-1:0" }) |bad| try std.testing.expectError(error.BadFault, EngramFault.parse(bad, 4));
    var b: [96]u8 = undefined;
    try std.testing.expectEqualStrings("off", (EngramFault{}).describe(&b, 0));
    try std.testing.expect(std.mem.indexOf(u8, f.describe(&b, 2), "fails its worker's row read once") != null);
    try std.testing.expect(std.mem.indexOf(u8, f.describe(&b, 1), "carries on") != null);
}

test "TF_DS_ENGRAM_FAULT: the named rank fails at its C-th layer-0 step once and is then spent; other ranks and other layers never fail" {
    engram_fault_seen.store(0, .release);
    engram_fault_spent.store(false, .release);
    defer {
        engram_fault_seen.store(0, .release);
        engram_fault_spent.store(false, .release);
    }
    const f = try EngramFault.parse("2:3", 4);
    // another rank: never, and it does not use up the count
    for (0..10) |_| try std.testing.expect(!engramFaultFires(f, 1, 0));
    // layer 1 steps are not counted
    for (0..10) |_| try std.testing.expect(!engramFaultFires(f, 2, 1));
    var fired: u32 = 0;
    for (0..12) |k| if (engramFaultFires(f, 2, 0)) {
        fired += 1;
        try std.testing.expectEqual(@as(usize, 3), k);
    };
    try std.testing.expectEqual(@as(u32, 1), fired);
    // off: nothing
    for (0..4) |_| try std.testing.expect(!engramFaultFires(.{}, 0, 0));
    // the worker takes the fault where it would take the disk's error
    const src = @embedFile("prompt.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "if (engramFaultFires(engram_fault, engram_fault_rank, j)) {") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "eh.wk_err = error.InputOutput;") != null);
}

test "TF_DS_ATTN_EPILOGUE: the switch, the kernel's shapes and the chunk's scratch decide the epilogue" {
    var two: prompt2d.Two = undefined;
    var cfg: Config = undefined;
    var cm: comm.Comm = undefined;
    var e = hostEngine(&two, &cfg, &cm);
    var k: mx.Kernels = undefined;
    var ch: Chunk = undefined;
    ch.mxs = .{ .q = 0, .s = 0, .z = 0, .fake = 0, .rows = 4096, .max_k = 8192 };
    const was = attn_epilogue;
    defer attn_epilogue = was;
    attn_epilogue = true;
    e.mx = null;
    try std.testing.expect(!epilogueOn(&e, &ch, 4096, 8192, 512, 64));
    e.mx = &k;
    try std.testing.expect(epilogueOn(&e, &ch, 4096, 8192, 512, 64));
    // the scratch's bounds
    try std.testing.expect(!epilogueOn(&e, &ch, 4097, 8192, 512, 64));
    try std.testing.expect(!epilogueOn(&e, &ch, 4096, 8193, 512, 64));
    try std.testing.expect(!epilogueOn(&e, &ch, 0, 8192, 512, 64));
    // the roped columns and the rest whole 32-value groups
    try std.testing.expect(!epilogueOn(&e, &ch, 4096, 8192, 512, 48));
    try std.testing.expect(!epilogueOn(&e, &ch, 4096, 8192, 512, 0));
    try std.testing.expect(!epilogueOn(&e, &ch, 4096, 8192, 512, 513));
    try std.testing.expect(!epilogueOn(&e, &ch, 4096, 8190, 512, 64));
    // the switch off
    attn_epilogue = false;
    try std.testing.expect(!epilogueOn(&e, &ch, 4096, 8192, 512, 64));
}

test "TF_DS_FRONT_OWNED's start check: quarter-HC, native weights, one quant for wq_a and wkv, every owned-row launch found" {
    var lays: [40]weights.Layer = undefined;
    try checkpointLayers(&lays);
    const gpa = std.testing.allocator;
    const qs = [_]usize{ 256, 512, 1024, 512 };
    var n: usize = 0;
    try std.testing.expectEqual(@as(?[]const u8, null), try frontOwnedOff(gpa, FakeFinds{ .asked = &n }, true, &lays, 20, 1e-20, &qs, 4));
    // every front layer's launch at every rank's owned rows of every row count was looked up
    try std.testing.expectEqual(@as(usize, 17 * qs.len * 4), n);
    try std.testing.expectEqualStrings("needs the checkpoint's own FP8 / MXFP4 weights", (try frontOwnedOff(gpa, FakeFinds{ .asked = &n }, false, &lays, 20, 1e-20, &qs, 4)).?);
    try std.testing.expectEqualStrings("this rank's kernel set has no _rmsnorm variant for the owned rows", (try frontOwnedOff(gpa, FakeFinds{ .asked = &n, .missing = "_rmsnorm" }, true, &lays, 20, 1e-20, &qs, 4)).?);
    lays[3].wkv.fmt = .bf16;
    try std.testing.expectEqualStrings("wq_a and wkv are not FP8 / FP4 on one K", (try frontOwnedOff(gpa, FakeFinds{ .asked = &n }, true, &lays, 20, 1e-20, &qs, 4)).?);
    // a layer with a compressor is not a front layer: its linears are not asked about
    lays[3].wkv.fmt = .fp8;
    lays[2].wkv.fmt = .bf16;
    try std.testing.expectEqual(@as(?[]const u8, null), try frontOwnedOff(gpa, FakeFinds{ .asked = &n }, true, &lays, 20, 1e-20, &qs, 4));
}

test "TF_DS_FFN_CODES's start check: quarter-HC, native weights, every owned-row routing launch found" {
    var lays: [40]weights.Layer = undefined;
    try checkpointLayers(&lays);
    for (&lays) |*l| l.gate_b = 0x5000;
    const gpa = std.testing.allocator;
    const qs = [_]usize{ 256, 512, 1024, 512 };
    var n: usize = 0;
    try std.testing.expectEqual(@as(?[]const u8, null), try ffnCodesOff(gpa, FakeFinds{ .asked = &n }, true, &lays, 20, 384, 6, 1.5, 7, &qs, 4));
    // every quarter-HC layer's launch at every rank's owned rows of every row count was looked up
    try std.testing.expectEqual(@as(usize, 20 * qs.len * 4), n);
    try std.testing.expectEqualStrings("needs the checkpoint's own FP8 / MXFP4 weights", (try ffnCodesOff(gpa, FakeFinds{ .asked = &n }, false, &lays, 20, 384, 6, 1.5, 7, &qs, 4)).?);
    try std.testing.expectEqualStrings("this rank's kernel set has no _route variant for the owned rows", (try ffnCodesOff(gpa, FakeFinds{ .asked = &n, .missing = "_route" }, true, &lays, 20, 384, 6, 1.5, 7, &qs, 4)).?);
}

test "TF_DS_ENGRAM_OWNER_SUM's start check: quarter-HC, equal Engram pair widths, no ring links, every owned-row gate launch found" {
    var lays: [40]weights.Layer = undefined;
    try checkpointLayers(&lays);
    const one: weights.Linear = .{ .words = 0, .k = 1, .n = 1 };
    for ([_]usize{ 1, 14 }) |li| { // the checkpoint's Engram layers (engram_layer_ids)
        lays[li].engram_wkv = one;
        lays[li].engram_qk = 0x5000 + li * 0x1000;
    }
    const gpa = std.testing.allocator;
    const qs = [_]usize{ 256, 512, 1024, 512 };
    const ew = [2]usize{ 12800, 12800 };
    var n: usize = 0;
    try std.testing.expectEqual(@as(?[]const u8, null), try engramOwnerSumOff(gpa, FakeFinds{ .asked = &n }, ew, &lays, 20, 1e-20, 5120, 4, &qs, 4));
    // both Engram layers' gate at every rank's owned rows of every row count was looked up
    try std.testing.expectEqual(@as(usize, 2 * qs.len * 4), n);
    try std.testing.expectEqualStrings("needs Engram layers of equal pair widths", (try engramOwnerSumOff(gpa, FakeFinds{ .asked = &n }, .{ 12800, 12288 }, &lays, 20, 1e-20, 5120, 4, &qs, 4)).?);
    try std.testing.expectEqualStrings("needs Engram layers of equal pair widths", (try engramOwnerSumOff(gpa, FakeFinds{ .asked = &n }, .{ 0, 0 }, &lays, 20, 1e-20, 5120, 4, &qs, 4)).?);
    try std.testing.expectEqualStrings("this rank's kernel set has no _engram_gate variant for the owned rows", (try engramOwnerSumOff(gpa, FakeFinds{ .asked = &n, .missing = "_engram_gate" }, ew, &lays, 20, 1e-20, 5120, 4, &qs, 4)).?);
}

test "TF_DS_BOUNDARY_OWNED's conditions: quarter-HC and a ratio-1 boundary layer with its index keys" {
    var lays: [40]weights.Layer = undefined;
    try checkpointLayers(&lays);
    lays[20].idx_wk = .{ .words = 0, .k = 1, .n = 1 }; // weights.zig loads it on a kv and index source
    try std.testing.expectEqual(@as(?[]const u8, null), boundaryOwnedOff(lays[20]));
    const why = "needs a boundary layer of compress ratio 1 with its index keys";
    var l = lays[20];
    l.ratio = 2;
    try std.testing.expectEqualStrings(why, boundaryOwnedOff(l).?);
    l = lays[20];
    l.idx_wk = null;
    try std.testing.expectEqualStrings(why, boundaryOwnedOff(l).?);
    l = lays[20];
    l.comp_wkv = null;
    try std.testing.expectEqualStrings(why, boundaryOwnedOff(l).?);
}

test "the four ranks' owned rows of a chunk of any row count cover it once, the encoder half only, the boundary until its handoff, and the stride is slab_plan.q" {
    var two: prompt2d.Two = undefined;
    two.idx_shard = true;
    var cfg: Config = undefined;
    var cm: comm.Comm = undefined;
    var e = hostEngine(&two, &cfg, &cm);
    cfg.candidate_source = 3;
    var ch: Chunk = undefined;
    ch.cap = 4096;
    var n: usize = 1;
    while (n <= 4096) : (n += if (n < 140 or n > 4000) 1 else 29) {
        beginHost(&ch, n, 8192);
        ch.encoder_only = (n % 2 == 0);
        try std.testing.expectEqual(slab_plan.q(n), ownQ(&e, &ch));
        var covered: usize = 0;
        var next: usize = 0;
        for (0..4) |k| {
            cm.rank = @intCast(k);
            // every layer of the encoder half has this rank's slab, the decoder half none
            for ([_]usize{ 0, 1, 19 }) |li| {
                const r = ownedRange(&e, &ch, li).?;
                try std.testing.expectEqual(slab_plan.slab(n, k).start, r.start);
                try std.testing.expectEqual(slab_plan.slab(n, k).n, r.n);
                try std.testing.expectEqual(slab_plan.slab(n, k).n, slabN(&e, &ch, k));
            }
            for ([_]usize{ 21, 39 }) |li| try std.testing.expect(ownedRange(&e, &ch, li) == null);
            // the boundary layer: owned for every chunk (decoder rows or none) until the handoff has cut the chunk
            const b = boundaryRange(&e, &ch, 20).?;
            try std.testing.expectEqual(slab_plan.slab(n, k).n, b.n);
            try std.testing.expect(boundaryRange(&e, &ch, 19) == null);
            const ix = idxShardRange(&e, &ch, 1).?;
            try std.testing.expectEqual(@as(usize, 0), ix.start % 16);
            try std.testing.expect(idxShardRange(&e, &ch, 3) == null); // (the candidate source and after: the pool)
            if (b.n != 0) {
                try std.testing.expectEqual(next, b.start);
                next += b.n;
                covered += b.n;
            }
        }
        try std.testing.expectEqual(n, covered);
        cm.rank = 1;
        ch.boundary_done = true;
        try std.testing.expect(boundaryRange(&e, &ch, 20) == null);
        // (the decoder half then takes the handoff slab: the next test)
        try std.testing.expect(ownedRange(&e, &ch, 20) != null);
    }
}

test "after the handoff every layer from the boundary to the head has this rank's slab of the decoder window and the four slabs cover it once; before it, and with the switch off, the decoder half has none" {
    var two: prompt2d.Two = undefined;
    two.idx_shard = true;
    var cfg: Config = undefined;
    var cm: comm.Comm = undefined;
    var e = hostEngine(&two, &cfg, &cm);
    var ch: Chunk = undefined;
    ch.cap = 4096;
    for ([_]usize{ 4096, 2048, 1000, 128, 13, 1 }) |n| for (1..@min(n, roci.hand_rows) + 1) |d| {
        beginHost(&ch, n, 8192);
        // before the cut: the boundary is the chunk's slab, the decoder half has none
        cm.rank = 3;
        try std.testing.expectEqual(slab_plan.slab(n, 3).n, ownedRange(&e, &ch, 20).?.n);
        for ([_]usize{ 21, 39, 40 }) |li| try std.testing.expect(ownedRange(&e, &ch, li) == null);
        // the cut: the window's d rows, the boundary done
        ch.n = d;
        ch.boundary_done = true;
        var next: usize = 0;
        for (0..4) |k| {
            cm.rank = @intCast(k);
            const s = slab_plan.slab(d, k);
            for ([_]usize{ 20, 21, 24, 36, 37, 39, 40 }) |li| {
                const r = ownedRange(&e, &ch, li).?;
                try std.testing.expectEqual(s.start, r.start);
                try std.testing.expectEqual(s.n, r.n);
            }
            try std.testing.expect(boundaryRange(&e, &ch, 20) == null);
            try std.testing.expectEqual(slab_plan.q(d), ownQ(&e, &ch));
            if (s.n != 0) {
                try std.testing.expectEqual(next, s.start);
                next += s.n;
            }
            // the head's row: this rank's slab holds the window's last row exactly when it is roci.headOwner's
            try std.testing.expectEqual(s.n != 0 and d - 1 >= s.start and d - 1 < s.start + s.n, roci.headOwner(d) == k);
        }
        try std.testing.expectEqual(d, next);
    };
}

test "an encoder-only chunk is known before layer 0 (replayCut's rule)" {
    // encoderOnly is replayCut's rule: the chunk's first decoder row (max(start, replay) - start) at or past its end
    for ([_]usize{ 0, 2048, 4096, 6000 }) |start| for ([_]usize{ 1, 1000, 2048 }) |n| for ([_]usize{ 0, start, start + n - 1, start + n, start + n + 1, start + 5000 }) |replay| {
        try std.testing.expectEqual(@max(start, replay) - start >= n, encoderOnly(start, n, replay));
    };
}

fn hostEngine(two: *const prompt2d.Two, cfg: *Config, cm: *comm.Comm) Engine {
    cfg.* = undefined;
    cfg.layers = 40;
    cm.* = comm.Comm.init(1, 4);
    var e: Engine = undefined;
    e.two = two;
    e.c = cfg;
    e.comm = cm;
    e.world = 2;
    return e;
}

const WaitFix = struct {
    go: wait_bell.Bell = .{},
    ready: wait_bell.Bell = .{},
    used: std.atomic.Value(usize) = .init(0),
    done: std.atomic.Value(usize) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    cpu_ns: std.atomic.Value(u64) = .init(0),
    result: std.atomic.Value(u8) = .init(0), // 1 true, 2 false once the wait ended
    duties: std.atomic.Value(u32) = .init(0),

    const Duties = struct {
        f: *WaitFix,
        pub fn run(d: @This()) !void {
            _ = d.f.duties.fetchAdd(1, .monotonic);
        }
    };

    fn cpu() u64 {
        var ts: std.os.linux.timespec = undefined;
        _ = std.os.linux.clock_gettime(.THREAD_CPUTIME_ID, &ts);
        return @intCast(@as(i128, ts.sec) * 1_000_000_000 + ts.nsec);
    }
    fn slot(f: *WaitFix) void {
        const c0 = cpu();
        const r = waitSlot(&f.go, &f.used, 2, 5, &f.stop);
        f.cpu_ns.store(cpu() -% c0, .release);
        f.result.store(if (r) 1 else 2, .release);
    }
    fn rows(f: *WaitFix, timeout_ns: u64) void {
        const c0 = cpu();
        const r = waitDone(&f.ready, &f.done, 3, &f.failed, Duties{ .f = f }, timeout_ns) catch false;
        f.cpu_ns.store(cpu() -% c0, .release);
        f.result.store(if (r) 1 else 2, .release);
    }
    fn nap(ms: i64) !void {
        try std.Io.sleep(std.testing.io, .fromMilliseconds(ms), .awake);
    }
};

test "the worker's wait for a free slot sleeps, wakes on the upload, ends on stop" {
    // sleeps: 200 ms waited, a yield loop would burn ~200 ms of CPU
    var f: WaitFix = .{};
    var th = try std.Thread.spawn(.{}, WaitFix.slot, .{&f});
    try WaitFix.nap(200);
    try std.testing.expectEqual(@as(u8, 0), f.result.load(.acquire));
    f.used.store(4, .release); // 5 < 4 + 2: a slot is free
    f.go.ring();
    th.join();
    try std.testing.expectEqual(@as(u8, 1), f.result.load(.acquire));
    try std.testing.expect(f.cpu_ns.load(.acquire) < 20_000_000);
    try std.testing.expect(f.go.sleeps.load(.monotonic) >= 1);
    // stop
    var g: WaitFix = .{};
    th = try std.Thread.spawn(.{}, WaitFix.slot, .{&g});
    try WaitFix.nap(50);
    g.stop.store(true, .release);
    g.go.ring();
    th.join();
    try std.testing.expectEqual(@as(u8, 2), g.result.load(.acquire));
}

test "the engine thread's wait for rows sleeps, runs its duties between sleeps, wakes on done and on failure" {
    var f: WaitFix = .{};
    var th = try std.Thread.spawn(.{}, WaitFix.rows, .{ &f, @as(u64, 100_000) });
    try WaitFix.nap(200);
    try std.testing.expectEqual(@as(u8, 0), f.result.load(.acquire));
    // a frame waiting at the socket is served within the timeout: the duties ran many times while it waited
    try std.testing.expect(f.duties.load(.monotonic) > 100);
    f.done.store(4, .release);
    f.ready.ring();
    th.join();
    try std.testing.expectEqual(@as(u8, 1), f.result.load(.acquire));
    try std.testing.expect(f.cpu_ns.load(.acquire) < 40_000_000);
    // failure
    var g: WaitFix = .{};
    th = try std.Thread.spawn(.{}, WaitFix.rows, .{ &g, @as(u64, 100_000) });
    try WaitFix.nap(30);
    g.failed.store(true, .release);
    g.ready.ring();
    th.join();
    try std.testing.expectEqual(@as(u8, 2), g.result.load(.acquire));
}

test "a long timeout still ends at once on a ring, and a ring is not needed for the duties" {
    var f: WaitFix = .{};
    const th = try std.Thread.spawn(.{}, WaitFix.rows, .{ &f, @as(u64, 5_000_000_000) });
    try WaitFix.nap(100);
    f.done.store(4, .release);
    f.ready.ring();
    th.join(); // would take 5 s if the ring did not wake it
    try std.testing.expectEqual(@as(u8, 1), f.result.load(.acquire));
}

test "TF_DS_WINDOW_ONCE: a chunk's window uploads are due once per (start, rows, floor, mode) and again when the key changes" {
    var ch: Chunk = undefined;
    ch.win_key = .{ std.math.maxInt(usize), 0, 0, 0 };
    ch.win_have = 0;
    for ([_]u8{ win_g, win_lo, win_s }) |b| {
        try std.testing.expect(windowFresh(&ch, 8192, 4096, 8064, false, b));
        for (0..39) |_| try std.testing.expect(!windowFresh(&ch, 8192, 4096, 8064, false, b));
    }
    // the next chunk: all three due again, once
    try std.testing.expect(windowFresh(&ch, 12288, 4096, 12160, false, win_lo));
    try std.testing.expect(!windowFresh(&ch, 12288, 4096, 12160, false, win_lo));
    try std.testing.expect(windowFresh(&ch, 12288, 4096, 12160, false, win_g));
    // a different floor or mode under the same start is a different key
    try std.testing.expect(windowFresh(&ch, 12288, 4096, 12000, false, win_g));
    try std.testing.expect(windowFresh(&ch, 12288, 4096, 12000, true, win_g));
}

test "TF_DS_WINDOW_ONCE: attention uploads the window indices through windowFresh into the chunk's own buffers when the switch is on, as before when off" {
    const code = @embedFile("prompt.zig");
    const at = std.mem.indexOf(u8, code, "pub fn attention(").?;
    const body = code[at .. std.mem.indexOf(u8, code[at..], "pmark(e, .a_core);").? + at];
    for ([_][]const u8{ "windowFresh(ch, start, n, lo, ring_mode, win_g)", "windowFresh(ch, start, n, lo, ring_mode, win_lo)", "windowFresh(ch, start, n, lo, ring_mode, win_s)", "gidx = ch.ridx_g", "ch.ridx_s", "try e.ops.gatherRows(e.s, ring, ring_rows, gidx," }) |needle|
        try std.testing.expect(std.mem.indexOf(u8, body, needle) != null);
    // off: the same three uploads in every layer, into ring_idx and wlo
    try std.testing.expect(std.mem.indexOf(u8, body, "var gidx = ch.ring_idx;") != null);
}

test "TF_DS_WINDOW_ONCE is on unless set otherwise: unset and 1 are on, 0 and any other text off" {
    try std.testing.expect(windowOnceWanted(null));
    try std.testing.expect(windowOnceWanted("1"));
    try std.testing.expect(windowOnceWanted(" 1\n"));
    try std.testing.expect(!windowOnceWanted("0"));
    try std.testing.expect(!windowOnceWanted(""));
    try std.testing.expect(!windowOnceWanted("on"));
}

test "layer 0's MoE partials take the body canary right before their gather, in a prompt chunk and in a round; off, nothing launches" {
    const p = @embedFile("prompt.zig");
    const m = @embedFile("model.zig");
    const r = @embedFile("round.zig");
    try std.testing.expectEqual(@as(usize, 0), body_layer);
    // the off path returns before any launch: not in effect, or another layer
    const at = std.mem.indexOf(u8, p, "pub fn bodyCanary(").?;
    const body = p[at .. at + std.mem.indexOf(u8, p[at..], "\n}\n").?];
    try std.testing.expect(std.mem.indexOf(u8, body, "{\n    if (qgate_body == .off or li != body_layer) return;\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "mx.qgateBody(k, e.s, pm, rows * moeWidth(e.two, e.c.hidden), @backingInt(qgate_body))") != null);
    // a part 1 already sent by the MoE's launch is refused before the launch
    const sent = std.mem.indexOf(u8, body, "if (c.own1_sent) return error.CanaryAfterSend;").?;
    const launch = std.mem.indexOf(u8, body, "mx.qgateBody(").?;
    try std.testing.expect(sent < launch);
    // both prompt layer loops: the MoE, then the canary on its partials, then their gather
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, m, "try prompt.moe(e, ch, li);\n            if (pt) |t| try t.mark(e.s, .moe);\n            try prompt.bodyCanary(e, ch, ch.pm, ch.n, li);\n            try prompt.gather(e, ch, ch.pm, .moe, li + 1);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, m, "try prompt.moe(e, ch, li);\n        try prompt.bodyCanary(e, ch, ch.pm, ch.n, li);\n        try prompt.gather(e, ch, ch.pm, .moe, li + 1);"));
    // a round's (every verify row, the drafted ones too): the MoE, its probe, the canary, the gather
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, r, "try Probe.check(probe, .moe_out, l, rd.pm);\n        try prompt.bodyCanary(e, null, rd.pm, R, l);\n        if (e.two) |tw| try tw.quarters(e, rd.pm, rd.gm, R, tw.dw, 4) else try e.gatherF32(rd.pm, rd.gm, R * d);"));
    // no other caller, and the switch is read by bodyCanary alone (round.zig never names it; model.zig's useQgateCanary sets it)
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, m, "prompt.bodyCanary("));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, r, "prompt.bodyCanary("));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, r, "qgate_body"));
}

test "a rank's MoE partials are its node's down columns on a 2D node, the hidden size otherwise" {
    var t: prompt2d.Two = undefined;
    t.dw = .{ 2560, 2048 };
    t.p = 0;
    try std.testing.expectEqual(@as(usize, 2560), moeWidth(&t, 5120));
    t.p = 1;
    try std.testing.expectEqual(@as(usize, 2048), moeWidth(&t, 5120));
    try std.testing.expectEqual(@as(usize, 5120), moeWidth(null, 5120));
}
