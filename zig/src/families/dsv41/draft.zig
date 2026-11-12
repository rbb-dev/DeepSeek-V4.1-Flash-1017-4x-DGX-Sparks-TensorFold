//! The DSpark drafter of the served lane's concurrent decoder (dspark.py's Drafter, DraftPool and BatchDraftGraph as
//! multi.py runs them) on one TP rank: the streams' window rings in one pool, the target's taps absorbed into them
//! (absorb after a prompt chunk; absorb_many of every drafting stream's window rows after a round's forward, the served
//! TF_DS_EAGER_ABSORB), and N streams' drafts and confidences from one pass: BatchDraftGraph._body with
//! TF_DS_DRAFT_FUSED and the hc switch (the three stages, each sublayer's post fused into the next mixes; q's RMSNorm
//! writing wq_b's rotated rows and q's RoPE in wq_b's epilogue; the block rows' keys normed and roped, stored nowhere;
//! the sparse attention over the stream's 128 newest absorbed positions and its own block rows; wo_a's epilogue
//! rotating wo_b's rows; the MoE at top 3 of the routed experts and the shared one), the Markov loop (markov.py: the
//! vocabulary split over the ranks, the bias rows of 256 frequent tokens cached, each step's bests gathered) and the
//! confidence head (one fp32 GEMV as torch issues it). The served L2 prefetch forks move no bytes and are left out; the
//! RDMA gathers are the rings' gathers (the same bytes).
const std = @import("std");
const cuda = @import("cuda");
const ring_wait = @import("ring_wait.zig");
const prompt = @import("prompt.zig");
const comm_mod = @import("comm.zig");
const weights = @import("weights.zig");
const plan = @import("plan.zig");
const ring2d = @import("ring2d.zig");
const tri_basic = @import("tri_basic.zig");
const tri_hc = @import("tri_hc.zig");
const tri_norm = @import("tri_norm.zig");
const tri = @import("tri.zig");
const tri_attn = @import("tri_attn.zig");
const tri_markov = @import("tri_markov.zig");
const exl3_linear = @import("exl3_linear.zig");
const mx = @import("mx.zig");
const exl3_experts = @import("exl3_experts.zig");
const markov_tokens = @import("markov_tokens.zig");
const draft2d = @import("draft2d.zig");
const draft_vocab = @import("draft_vocab.zig");
const land_mod = @import("land.zig");
const graph_fallback = @import("graph_fallback.zig");

const Engine = prompt.Engine;
const Chunk = prompt.Chunk;
const Config = @import("config.zig").Config;

/// MultiDecoder's stream slots in the served lane (--parallel 4): the pool's streams.
pub const max_streams = 32; // the most streams any stream setting (Model.streams) gives the pool
/// The served pool's streams when unset (--parallel 4); the four-node lane serves 16.
pub const default_streams = 4;
/// A pass's rows at most: every stream's block (four streams: 20). Past kernels.decode_rows() (16 rows: four streams) the MoE gate is a
/// cuBLAS matmul with the plain routing and the attention one split (its pick list padded to 16 columns of -1).
pub const max_rows = max_streams * 5; // every stream of the pool, a five-row block each
/// kernels.decode_rows(): the row-invariant decode kernels' rows at most.
/// sparse_attn's pick list of a single-split call: padded with -1 to a multiple of 16 entries.
const pick_pad = 16;
/// dspark.py: ring_size = window + 16.
pub const ring_extra = 16;
/// The stages (DSpark blocks) and a stream's block rows at most.
pub const max_stages = 4;
pub const max_block = 8;
/// absorb_many's rows at most: every drafting stream's verify window (16 rows a round).
pub const max_absorb = 64; // a round's drafting rows at most (round.max_rows)

/// DraftPool: one ring plane a stage, bf16 [slots * ring, head_dim]; slot s's rows [s * ring, (s + 1) * ring). Positions
/// below `absorbed[s]` are in slot s's rings.
pub const Pool = struct {
    rings: [max_stages]u64 = @splat(0),
    stages: usize,
    slots: usize,
    ring: usize,
    hd: usize,
    absorbed: [max_streams]usize = @splat(0),

    /// Zeros (torch.zeros), as the served pool starts.
    pub fn init(e: *const Engine, a: *prompt.Arena, slots: usize) !Pool {
        const ds = e.w.dspark orelse return error.NoDrafter;
        if (ds.blocks.len > max_stages or slots > max_streams) return error.DrafterTooLarge;
        var p: Pool = .{ .stages = ds.blocks.len, .slots = slots, .ring = e.c.window + ring_extra, .hd = e.c.head_dim };
        for (0..p.stages) |j| {
            p.rings[j] = try a.take(p.planeBytes());
            try prompt.fill(e, p.rings[j], 0, p.planeBytes());
        }
        return p;
    }

    pub fn planeBytes(p: *const Pool) usize {
        return p.slots * p.ring * p.hd * 2;
    }

    /// Slot s's rows of stage j's plane (DraftPool.views[s].rings[j]).
    pub fn view(p: *const Pool, j: usize, s: usize) u64 {
        return p.rings[j] + s * p.ring * p.hd * 2;
    }
};

/// The points of a pass a checker sees, in the order the served build's recorder takes them (stage j: layer
/// layers + j): each stage's attention input rows and its stage's ring plane, its partial and gather, the MoE's input
/// rows, partial and gather; the streams and pre after the stages; the head's logits (this rank's columns); the drafts.
pub const Point = enum { attn_in, attn_ring, attn_out, attn_gather, moe_in, moe_out, moe_gather, stages_h, stages_pre, local, markov };

/// A checker of a pass's buffers between its steps (synchronized first by the checker): false stops the pass.
pub const Probe = struct {
    ctx: *anyopaque,
    at: *const fn (ctx: *anyopaque, what: Point, stage: usize, dev: u64) anyerror!bool,

    fn check(pb: ?Probe, what: Point, stage: usize, dev: u64) !void {
        const p = pb orelse return;
        if (!try p.at(p.ctx, what, stage, dev)) return error.DraftMismatch;
    }
};

/// TF_DS_DRAFT_DUMP=DIR (a debugging aid, off in service): node 0 appends every absorbed taps row (DIR/taps.bin bf16
/// [n, taps * D], DIR/tpos.bin int64 positions) until pass TF_DS_DRAFT_DUMP_PASS (default 2), then runs that pass
/// eagerly and writes its intermediates (DIR/<name>-<stage>.bin, raw device bytes) and DIR/meta.txt.
pub var ddump: ?[:0]const u8 = null;
pub var ddump_pass: usize = 2;
pub var ddump_node: u32 = 0;
var ddump_count: usize = 0;
var ddump_on: bool = false; // inside the dumped pass

fn ddWrite(e: *const Engine, name: []const u8, stage: usize, ptr: u64, bytes: usize, append: bool) void {
    const dir = ddump orelse return;
    if (ddump_node != 0 or bytes == 0) return;
    e.s.synchronize() catch return;
    const host = std.heap.page_allocator.alloc(u8, bytes) catch return;
    defer std.heap.page_allocator.free(host);
    e.d.check(e.d.api.cuMemcpyDtoH_v2(host.ptr, ptr, bytes), "cuMemcpyDtoH") catch return;
    var buf: [512]u8 = undefined;
    const path = std.mem.printSentinel(&buf, "{s}/{s}-{d}.bin", .{ dir, name, stage }, 0) catch return;
    const f = std.c.fopen(path.ptr, if (append) "ab" else "wb") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(host.ptr, 1, bytes, f);
}

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// A host buffer of the dumped pass.
fn ddHost(name: []const u8, bytes: []const u8) void {
    const dir = ddump orelse return;
    if (ddump_node != 0) return;
    var buf: [512]u8 = undefined;
    const path = std.mem.printSentinel(&buf, "{s}/{s}-0.bin", .{ dir, name }, 0) catch return;
    const f = std.c.fopen(path.ptr, "wb") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(bytes.ptr, 1, bytes.len, f);
}

/// A dumped pass's buffer (only inside the dumped pass).
fn dd(e: *const Engine, name: []const u8, stage: usize, ptr: u64, bytes: usize) void {
    if (ddump_on) ddWrite(e, name, stage, ptr, bytes, false);
}

/// An absorb's main_proj rows (whole, after a four-way absorb's gather), appended until the dumped pass.
fn ddMain(e: *const Engine, ay: u64, bytes: usize) void {
    if (ddump == null or ddump_count > ddump_pass) return;
    ddWrite(e, "main_y", 0, ay, bytes, true);
}

/// Absorbed taps rows and their positions (host), appended until the dumped pass.
fn ddTaps(e: *const Engine, dr: *const Drafter, taps: u64, pos: []const i64) void {
    if (ddump == null or ddump_count > ddump_pass) return;
    ddWrite(e, "taps", 0, taps, pos.len * dr.taps_w * 2, true);
    const dir = ddump.?;
    if (ddump_node != 0) return;
    var buf: [512]u8 = undefined;
    const path = std.mem.printSentinel(&buf, "{s}/tpos-0.bin", .{dir}, 0) catch return;
    const f = std.c.fopen(path.ptr, "ab") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(std.mem.sliceAsBytes(pos).ptr, 8, pos.len, f);
}

const DumpProbe = struct {
    e: *const Engine,
    dr: *const Drafter,
    pool: *const Pool,
    R: usize,
    N: usize,
    steps: usize,

    fn at(ctx: *anyopaque, what: Point, stage: usize, dev: u64) anyerror!bool {
        const p: *DumpProbe = @ptrCast(@alignCast(ctx));
        const e = p.e;
        const d = e.c.hidden;
        const R = p.R;
        const bytes: usize = switch (what) {
            .attn_in, .moe_in => R * d * 2,
            .attn_ring => p.pool.slots * p.pool.ring * p.pool.hd * 2,
            .attn_out, .moe_out => R * d * 4,
            .attn_gather, .moe_gather => e.world * R * d * 4,
            .stages_h => R * e.c.hc * d * 2,
            .stages_pre => R * e.c.hc * 4,
            .local => R * p.dr.headWidth(e) * 4,
            .markov => p.N * (p.steps + 1) * 8,
        };
        ddWrite(e, @tagName(what), stage, dev, bytes, false);
        return true;
    }
};

/// Debug (TF_DS_DRAFT_DUMP): a round's tap rows and the streams they came from (the first rounds), appended.
var tapcheck_n: usize = 0;
pub fn ddTapCheck(e: *const Engine, h: u64, tap: u64, ld: usize, rows: usize) void {
    if (ddump == null or tapcheck_n >= 12) return;
    tapcheck_n += 1;
    const d = e.c.hidden;
    ddWrite(e, "tc_h", 0, h, rows * e.c.hc * d * 2, true);
    // the tap rows: [rows, d] of a [rows, ld] buffer
    for (0..rows) |r| ddWrite(e, "tc_t", 0, tap + r * ld * 2, d * 2, true);
}

/// TF_DS_DRAFT_PROF=1 (a profile, off in service): passes run eagerly with a synchronize at each mark; each mark's
/// host ms since the previous one summed by name (printed by printProf).
pub var dprof: bool = false;
/// TF_DS_DRAFT_LOG=FILE (a measurement, off in service): node 0 appends each pass's drafts and confidences.
pub var dlog: ?[:0]const u8 = null;
const prof_names = [_][]const u8{ "start", "embed", "mix_a", "qkv_a", "q_b", "kv_rope", "sattn", "wo_a", "wo_b", "gather_a", "mix_f", "gate", "experts_gu", "experts_d", "gather_m", "post", "head", "markov", "conf", "conf_a", "conf_b", "gather_a0", "gather_m0" };
const ProfMark = enum { start, embed, mix_a, qkv_a, q_b, kv_rope, sattn, wo_a, wo_b, gather_a, mix_f, gate, experts_gu, experts_d, gather_m, post, head, markov, conf, conf_a, conf_b, gather_a0, gather_m0 };
var prof_ns: [prof_names.len]u64 = @splat(0);
var prof_last: u64 = 0;
var prof_passes: u64 = 0;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn pmark(e: *const Engine, m: ProfMark) void {
    if (!dprof) return;
    e.s.synchronize() catch {};
    const t = nowNs();
    if (m != .start) prof_ns[@backingInt(m)] += t - prof_last;
    prof_last = t;
}

pub fn printProf() void {
    if (!dprof or prof_passes == 0) return;
    var buf: [2048]u8 = undefined;
    var n: usize = 0;
    n += (std.fmt.bufPrint(buf[n..], "{{\"follow\": \"draft\", \"draft_profile_ms\": {{\"passes\": {d}", .{prof_passes}) catch return).len;
    var tot: u64 = 0;
    for (prof_names, 0..) |nm, i| {
        if (i == 0) continue;
        tot += prof_ns[i];
        n += (std.fmt.bufPrint(buf[n..], ", \"{s}\": {d:.3}", .{ nm, @as(f64, @floatFromInt(prof_ns[i])) / @as(f64, @floatFromInt(prof_passes)) / 1e6 }) catch return).len;
    }
    n += (std.fmt.bufPrint(buf[n..], ", \"total\": {d:.3}}}}}\n", .{@as(f64, @floatFromInt(tot)) / @as(f64, @floatFromInt(prof_passes)) / 1e6}) catch return).len;
    std.debug.print("{s}", .{buf[0..n]});
}

/// An absorb_many item: a drafting stream's slot, its window rows' taps (bf16 [n, taps * D], contiguous rows) and the
/// window's first position.
pub const Item = struct { slot: usize, taps: u64, n: usize, start: usize };

/// The drafter's sublayer gather: fp32 [R, width] partials -> dst [world, R, width] in TP2 rank order (a 2D node: its pair's two,
/// over the row pair's ring).
fn pairGather(e: *const Engine, src: u64, dst: u64, R: usize, width: usize) !void {
    const tw = e.two orelse return error.DecodeNeedsRings;
    return tw.pairGather(e, src, dst, R, width, 4);
}

/// The per-step gather of the Markov loop's bests: [world, rows, 4] fp32 into dst.
const MarkovGather = struct {
    e: *const Engine,
    dst: u64,

    fn run(ctx: *anyopaque, send: u64, rows: usize) anyerror!u64 {
        const g: *MarkovGather = @ptrCast(@alignCast(ctx));
        // the four nodes' ring carries the bests (rows x 16 bytes a node), in the communicator's rank order
        const tw = g.e.two orelse return error.DecodeNeedsRings;
        try tw.nodesGather(g.e, send, g.dst, rows);
        return g.dst;
    }
};

/// TF_DS_DRAFT_VOCAB: a step's picked list positions to their ids (ops.Ops.listIds), on the pass's stream.
const ListIds = struct {
    e: *const Engine,
    ids: u64,

    fn run(ctx: *anyopaque, out: u64, stride: usize, rows: usize) anyerror!void {
        const l: *ListIds = @ptrCast(@alignCast(ctx));
        try l.e.ops.listIds(l.e.s, out, stride, rows, l.ids, draft_vocab.count);
    }
};

/// The kit's base set as draft_vocab.kitOff's lookups.
const SetFinds = struct {
    set: *const cuda.aot.Set,

    pub fn has(f: SetFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        return f.set.lookup(name, args, consts) != null;
    }
};

/// TF_DS_DRAFT_VOCAB's state on a node: the list's head rows of this node (FP4 [8,192, D], the list's order), the
/// Markov loop on its positions (its head rows [count, rank] fp16 and cached bias rows of its own), the ids (int64
/// [count]: the picks' map), the start's exchange plan and the rows it took from other nodes.
pub const Vocab = struct {
    head: weights.Linear,
    mk: tri_markov.Markov,
    ids: u64,
    flag: u64, // u32: a gather's row past its table (the start's checks)
    plan: draft_vocab.Plan,
    idx: u64, // int64 [nodes, 8,192]: each send's (and the node's own) rows of its quarter
    quarters: [draft_vocab.nodes][2]usize, // every node's head rows of the vocabulary
    blob: usize, // the exchange's export in bytes: its rounds' chunks, the staging's first part
    from_others: usize,
    bits: draft_vocab.Bits, // the list's ids, a bit an id (rank 0's guard)
};

/// TF_DS_DRAFT_VOCAB's start: the rows of this node's quarter `q` each node takes (pl.send), as row indices of the
/// quarter's head, into out int64 [nodes, 8,192] (node j's from j * 8,192; the rest 0).
pub fn vocabRows(pl: draft_vocab.Plan, q: [2]usize, out: []i64) void {
    @memset(out, 0);
    for (pl.send, 0..) |x, j| for (x[0]..x[1], 0..) |p, i| {
        out[j * draft_vocab.per_node + i] = @as(i64, draft_vocab.id(p)) - @as(i64, @intCast(q[0]));
    };
}

/// The rows of this node's quarter node j takes (pl.send[j]), gathered from the FP4 head h8 by the uploaded vocabRows
/// table `idx` into dst_w (codes, k / 2 bytes a row) and dst_s (scales, k / 32): how many.
pub fn vocabPack(ops: anytype, s: cuda.Stream, h8: weights.Linear, idx: u64, pl: draft_vocab.Plan, j: usize, dst_w: u64, dst_s: u64, flag: u64) !usize {
    const x = pl.send[j];
    const r = x[1] - x[0];
    if (r == 0) return 0;
    const at = idx + j * draft_vocab.per_node * 8;
    try ops.gatherRows(s, h8.w, h8.n, at, dst_w, h8.k / 2, r, flag);
    try ops.gatherRows(s, h8.s, h8.n, at, dst_s, h8.k / 32, r, flag);
    return r;
}

/// Where the rows node j sends node `me` (pl.recv[j] of me's plan) land in me's list head: its codes' and scales'
/// spans; null when j sends none.
pub fn vocabLand(pl: draft_vocab.Plan, me: usize, j: usize, head: weights.Linear) ?[2]land_mod.Span {
    const x = pl.recv[j];
    if (x[1] == x[0]) return null;
    const at = x[0] - draft_vocab.nodeRange(me)[0];
    const cb: usize = head.k / 2;
    const sb: usize = head.k / 32;
    return .{ .{ .addr = head.w + at * cb, .len = (x[1] - x[0]) * cb }, .{ .addr = head.s + at * sb, .len = (x[1] - x[0]) * sb } };
}

/// The drafter pass's CUDA graphs (BatchDraftGraph), one a (streams, steps): every pointer it reads is a fixed
/// buffer and every per-pass value (ids, positions, slots, the pick lists) is uploaded ahead of the replay.
pub const PassGraphs = struct {
    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(Key, cuda.graph.Exec) = .empty,
    lazy: Lazy = .{}, // the passes held eager after a failed capture, the captures attempted (TF_DS_GRAPH_FAIL_TEST=RANK:N:pass)

    pub const Key = struct { streams: u32, steps: u32, list: bool = false }; // list: on TF_DS_DRAFT_VOCAB's list
    const Lazy = graph_fallback.Lazy(graph_fallback.Cuda, Key);

    pub fn init(gpa: std.mem.Allocator) PassGraphs {
        return .{ .gpa = gpa };
    }

    pub fn deinit(g: *PassGraphs) void {
        var it = g.map.valueIterator();
        while (it.next()) |x| {
            ring_wait.forget(@intFromPtr(x.handle));
            x.deinit();
        }
        g.map.deinit(g.gpa);
        g.lazy.deinit(g.gpa);
        g.* = undefined;
    }

    pub fn count(g: *const PassGraphs) usize {
        return g.map.count();
    }

    fn run(g: *PassGraphs, dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, N: usize, steps: usize) !void {
        const key: Key = .{ .streams = @intCast(N), .steps = @intCast(steps), .list = dr.onList() };
        if (g.map.getPtr(key)) |x| return ring_wait.launchGraph(x.*, e.s);
        // G0.3, as round.Graphs.run: every rank captures and instantiates, then all agree before any launch
        // (graph_fallback: launch and keep, every rank eager, or the pass failed on every rank)
        var site: Site = .{ .dr = dr, .e = e, .ch = ch, .pool = pool, .N = N, .steps = steps };
        try g.lazy.run(g.gpa, e.s, key, &site, Store{ .g = g }, prompt.CaptureRanks{ .e = e, .kind = .pass });
    }

    /// The map as Lazy's store: room for one more pass before the ranks vote, and a launched one kept.
    const Store = struct {
        g: *PassGraphs,

        pub fn reserve(st: Store, gpa: std.mem.Allocator) !void {
            try st.g.map.ensureUnusedCapacity(gpa, 1);
        }

        pub fn keep(st: Store, key: Key, x: cuda.graph.Exec) void {
            st.g.map.putAssumeCapacity(key, x);
        }
    };

    /// A pass as Lazy's site. It changes no host state as it records (its inputs are uploaded before it; it has no
    /// side stream), so a failed capture has nothing to put back.
    const Site = struct {
        dr: *Drafter,
        e: *const Engine,
        ch: *const Chunk,
        pool: *const Pool,
        N: usize,
        steps: usize,

        pub fn body(s: *Site) !void {
            return s.dr.body(s.e, s.ch, s.pool, s.N, s.steps, null);
        }

        pub fn abort(_: *Site) void {}

        pub fn undo(_: *Site) void {}

        pub fn value(_: *Site, exec: cuda.graph.Exec) cuda.graph.Exec {
            return exec;
        }

        pub fn fellBack(s: *Site, err: ?anyerror, left: u32) void {
            std.log.warn("dsv41: the drafter pass graph of {d} streams, {d} steps failed ({s}): every rank runs it eagerly for its next {d} uses, then captures it again (logged once a pass)", .{ s.N, s.steps, if (err) |x| @errorName(x) else "on another rank", left });
        }

        pub fn failed(s: *Site, err: ?anyerror) void {
            if (err) |x| {
                std.log.err("drafter pass graph of {d} streams, {d} steps failed on rank {d} while it was recorded: {s}; the pass fails on every rank (restart the engine if replies stop: the ranks' graph state may differ)", .{ s.N, s.steps, s.e.comm.rank, @errorName(x) });
            } else std.log.err("drafter pass graph of {d} streams, {d} steps failed on another rank while it was recorded; the pass fails on every rank (restart the engine if replies stop: the ranks' graph state may differ)", .{ s.N, s.steps });
        }
    };
};

pub const Drafter = struct {
    ds: weights.DSpark,
    n: usize, // a stream's block rows: [token, noise x (n - 1)]
    noise: i64,
    topk: usize,
    slots_moe: usize, // the MoE's slots a row: top-k routed and the shared expert
    taps_w: usize, // the taps' row width: taps * D
    // the pass's index rows (int64, max_rows a row: ids, positions, window positions, ring bases, zeros, -1), bidx
    // [max_rows, n] (each row's stream's block rows; a single-split pass's padded to 16 columns of -1), wlo int64 [1] =
    // 0 (Model._zero)
    rows: u64,
    bidx: u64,
    wlo: u64,
    // the streams and the mHC state
    h: u64,
    h_alt: u64,
    x: u64,
    part: u64,
    pre: u64,
    pre_a: u64,
    pre_f: u64,
    post: u64,
    comb: u64,
    // attention
    qa: u64,
    ykv: u64,
    qr: u64,
    xhq: u64, // fp16 [R, q_lora]: wq_b's rotated input rows (the RMSNorm writes them)
    q: u64,
    kvb: u64, // bf16 [R, head_dim]: the block rows' keys
    o: u64,
    pm_: u64,
    pl_: u64,
    po_: u64,
    u: u64,
    xb: u64, // fp16 [R, wo_b.k]: wo_b's rotated input rows (wo_a's epilogue writes them)
    pa: u64,
    ga: u64,
    // MoE
    gl: u64,
    xf: u64, // fp32 [R, D]: a wide pass's rows for the gate's cuBLAS matmul
    gate_f: [max_stages]u64, // fp32 [experts, D]: each stage's gate widened (gate_w.float())
    pick: u64,
    mw: u64,
    pm: u64,
    gm: u64,
    xsd: exl3_experts.DecodeScratch,
    xs: exl3_experts.Scratch, // a pass of EXACT_ROWS rows or more (13+ streams): the chunk path on a scratch of its own (d48)
    mxe: mx.ExpertScratch = undefined, // Format.native: the stages' experts' grouping and rows (mx.zig)
    nxh: u64 = 0, // Format.native: the stages' intermediate rows, bf16 [max_rows * slots, width]
    // the head
    xc: u64, // bf16 [R, D]: the collapsed streams (pre-norm: the confidence head reads them)
    xn: u64,
    local: u64, // fp32 [R, this rank's vocabulary columns]
    // the Markov loop
    mk: tri_markov.Markov,
    msc: tri_markov.Scratch,
    mg: u64, // fp32 [world, max_streams, 4]: a step's gathered bests
    out: u64, // int64 [max_streams, n + 1]: each stream's last token, then its drafts
    // the confidence head
    embs: u64, // bf16 [max_rows, rank]: the Markov embedding rows of the tokens each step read
    xcf: u64, // fp32 [max_rows, D]
    ef: u64, // fp32 [max_rows, rank]
    conf_in: u64, // fp32 [max_rows, D + rank]
    confw: u64, // fp32 [D + rank]: the head's weight widened (conf.float())
    conf: u64, // fp32 [max_rows]
    confw_host: []f32 = &.{}, // the head's fp32 weight on the host: the confidence dot runs there (pass)
    xc_host: []u16 = &.{}, // bf16 [max_rows, D]: the collapsed rows read back
    em_host: []u16 = &.{}, // bf16 [max_rows, rank]: the embedding rows read back
    // absorbs: the taps rows (one item's or the items' cat), main_proj's rows, main_x, each stage's wkv rows, the
    // keys' normed rows (kv_norm_rope's out, not read), positions and ring rows
    at: u64,
    ay: u64,
    ax: u64,
    ak: [max_stages]u64,
    ao: u64,
    apos: u64,
    arows: u64,
    // TF_DS_DRAFT_4WAY (weights.draft_4way): the stages split four ways like the backbone (the loaded blocks hold the
    // node's 2D parts); their exchanges the backbone's (catRank, the parity packs, quartersDecode)
    four: bool = false,
    // TF_DS_DRAFT_MAIN_4WAY (weights.draft_main_4way): main_proj holds the node's quarter of its columns
    // (weights.mainProjCols); an absorb's main_proj rows go to aq, the four quarters gathered into ag ([2, n, hidden / 2]:
    // quartersDecode's layout), each plane copied into its half of ay's rows
    main4: bool = false,
    aq: u64 = 0,
    ag: u64 = 0,
    // TF_DS_DRAFT_LOG: each stage's active experts (the grouping's count) copied out by the pass, and the passes so far
    nacts: u64 = 0,
    passes: u64 = 0,
    // the pass's graphs (BatchDraftGraph: one a streams and steps; null: eager)
    graphs: ?*PassGraphs = null,
    // TF_DS_DRAFT_VOCAB (vocabPrepare, vocabExchange): the pruned list's head, Markov loop and ids; null: the whole
    // vocabulary's
    vocab: ?Vocab = null,
    // with the list in effect, the next pass on the whole head (the guard's call, every rank alike: the pass frame)
    vocab_full: bool = false,
    // the last pass on the host: each stream's drafts and confidences (`steps` of them)
    streams: usize = 0,
    steps: usize = 0,
    drafts: [max_streams][max_block]i64 = undefined,
    confs: [max_streams][max_block]f32 = undefined,

    /// Every buffer for passes of up to max_rows rows and absorbs of up to a ring of rows; the Markov loop's cached
    /// bias rows (Markov.fill, at load as served) for rank s.rank of s.world.
    pub fn init(e: *const Engine, a: *prompt.Arena, sp: plan.Split) !Drafter {
        const c = e.c;
        const ds = e.w.dspark orelse return error.NoDrafter;
        if (ds.blocks.len > max_stages or c.dspark_block > max_block or c.dspark_block == 0) return error.DrafterTooLarge;
        const d = c.hidden;
        const hc = c.hc;
        const hd = c.head_dim;
        const R = max_rows;
        const b0 = ds.blocks[0];
        const hl = b0.wq_b.n / hd; // the blocks' heads (TP2's on a 2D node, the node's own four ways)
        var dr: Drafter = undefined;
        dr.four = if (e.two) |tw| hl == tw.heads else false;
        // TF_DS_DRAFT_MAIN_4WAY: main_proj's quarter, gathered over the 2D view's quarters
        dr.main4 = ds.main_proj.n * 4 == c.hidden and e.two != null;
        if (ds.main_proj.n != c.hidden and !dr.main4) return error.WeightsNotThisSplit;
        if (dr.four) {
            // four ways: every stage's parts are the backbone's widths (its exchanges move them), Format.native only
            const tw = e.two.?;
            if (e.mx == null) return error.WeightsNotThisSplit;
            for (ds.blocks) |blk| {
                const x = blk.experts;
                const ir: usize = if (x.rest) |r| r.width else 0;
                if (blk.groups * blk.wo_a[0].n != tw.uw[tw.p] or blk.wo_b.n != tw.ow[tw.p] or x.width != tw.gu[tw.p] or
                    x.down_n != tw.dw[tw.p] or ir != tw.ir) return error.WeightsNotThisSplit;
            }
        }
        dr.passes = 0;
        dr.vocab = null;
        dr.vocab_full = false;
        dr.nacts = if (dlog != null and e.mx != null) try a.take(max_stages * 4) else 0;
        dr.ds = ds;
        dr.n = c.dspark_block;
        dr.noise = c.dspark_noise;
        dr.topk = c.draft_top_k;
        dr.slots_moe = c.draft_top_k + 1;
        dr.taps_w = c.dspark_taps.slice().len * d;
        dr.streams = 0;
        dr.steps = 0;
        dr.graphs = null;
        dr.rows = try a.take(6 * R * 8);
        dr.bidx = try a.take(R * pick_pad * 8);
        dr.wlo = try a.take(8);
        try prompt.fill(e, dr.wlo, 0, 8);
        dr.h = try a.take(R * hc * d * 2);
        dr.h_alt = try a.take(R * hc * d * 2);
        dr.x = try a.take(R * d * 2);
        dr.part = try a.take(R * tri_basic.hc_blocks * 32 * 4);
        dr.pre = try a.take(R * hc * 4);
        dr.pre_a = try a.take(R * hc * 4);
        dr.pre_f = try a.take(R * hc * 4);
        dr.post = try a.take(R * hc * 4);
        dr.comb = try a.take(R * hc * hc * 4);
        dr.qa = try a.take(R * b0.wq_a.n * 2);
        dr.ykv = try a.take(R * hd * 2);
        dr.qr = try a.take(R * b0.wq_a.n * 2);
        dr.xhq = try a.take(R * b0.wq_b.k * 2);
        dr.q = try a.take(R * hl * hd * 2);
        dr.kvb = try a.take(R * hd * 2);
        dr.o = try a.take(R * hl * hd * 2);
        dr.pm_ = try a.take(R * hl * tri_attn.attn_splits * 4);
        dr.pl_ = try a.take(R * hl * tri_attn.attn_splits * 4);
        dr.po_ = try a.take(R * hl * tri_attn.attn_splits * hd * 4);
        dr.u = try a.take(R * b0.groups * b0.wo_a[0].n * 2);
        dr.xb = try a.take(R * b0.wo_b.k * 2);
        dr.pa = try a.take(R * d * 4);
        dr.ga = try a.take(e.world * R * d * 4);
        const ne: usize = b0.experts.count - 1;
        dr.gl = try a.take(@max(16 * (d / 256), R) * ne * 4);
        dr.xf = try a.take(R * d * 4);
        dr.gate_f = @splat(0);
        if (e.mx == null) for (ds.blocks, 0..) |lay, j| {
            dr.gate_f[j] = try a.take(ne * d * 4);
            try e.ops.f16ToF32(e.s, lay.gate_w, dr.gate_f[j], ne * d);
        };
        dr.pick = try a.take(R * dr.slots_moe * 4);
        dr.mw = try a.take(R * dr.slots_moe * 4);
        dr.pm = try a.take(R * d * 4);
        dr.gm = try a.take(e.world * R * d * 4);
        // Model._moe_scratch(("moe", top-k + 1, the stages' experts, decode)): one decode scratch of their own
        if (e.mx != null) {
            const ex = b0.experts;
            const es = mx.ExpertScratch.sizes(R, dr.slots_moe, ex.count, ex.dims, ex.down_k, ex.down_n);
            dr.mxe.rows = R;
            dr.mxe.slots = dr.slots_moe;
            inline for (.{ "q", "s", "cnt", "off", "fill", "act", "nact", "ent", "erow", "hq", "hs", "out" }, 0..) |f, j| {
                @field(dr.mxe, f) = try a.take(es[j]);
            }
            // its decode launches' split partials (mx.decodeZFloats: the stages' experts, router and wo_a)
            var zf: usize = 0;
            for (ds.blocks) |blk| zf = @max(zf, mx.decodeZFloats(blk, c.hidden, ne, c.index_heads, R, dr.slots_moe));
            dr.mxe.z = .{ .ptr = try a.take(zf * 4), .floats = zf };
            // gate / up's pack (a parity split's four-way stage: every entry's main columns, then every entry's rest)
            const pack = ex.width + if (ex.rest) |r| r.width else 0;
            dr.nxh = try a.take(R * dr.slots_moe * pack * 2);
            dr.xsd = std.mem.zeroes(exl3_experts.DecodeScratch);
            dr.xs = std.mem.zeroes(exl3_experts.Scratch);
        } else dr.xsd = try prompt.decodeScratch(e, a, b0.experts, dr.slots_moe);
        if (e.mx == null) {
            const ex = b0.experts;
            const sz = exl3_experts.Scratch.sizes(R, dr.slots_moe, ex.dims, ex.width, ex.count);
            dr.xs.rows = R;
            dr.xs.slots = dr.slots_moe;
            inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
                @field(dr.xs, f) = try a.take(sz[j]);
            }
        }
        dr.xc = try a.take(R * d * 2);
        dr.xn = try a.take(R * d * 2);
        dr.local = try a.take(R * e.w.head.n * 4);
        // the Markov loop: every token's slot -1 (none), then the cached rows of the first 256 frequent tokens
        const none = try a.take(c.vocab * 4);
        try e.d.check(e.d.api.cuMemsetD32Async(none, 0xffffffff, c.vocab, e.s.handle), "cuMemsetD32Async");
        // the Markov loop's kernels read the head as fp16 (the EXL3 lane's checkpoint); DeepSeek's own holds it in bf16
        var mhead = ds.markov_head;
        if (e.mx) |mk| {
            const n = c.vocab * c.markov_rank;
            mhead = try a.take(n * 2);
            try mx.bf16ToF16(mk, e.s, ds.markov_head, mhead, n);
        }
        dr.mk = if (e.two) |tw| try draft2d.markov(c.*, tw, mhead, ds.markov_embed, none) else try tri_markov.Markov.init(c.*, sp, mhead, ds.markov_embed, none);
        const k = markov_tokens.cached.len;
        const tok = try a.take(k * 8);
        try prompt.upload(e, tok, &markov_tokens.cached, k * 8);
        const cache = try a.take(k * dr.mk.cols * 2);
        const slot = try a.take(c.vocab * 4);
        {
            const host = try std.heap.page_allocator.alloc(i32, c.vocab);
            defer std.heap.page_allocator.free(host);
            @memset(host, -1);
            for (markov_tokens.cached, 0..) |t, j| host[@intCast(t)] = @intCast(j);
            try prompt.upload(e, slot, host.ptr, c.vocab * 4);
            try e.s.synchronize(); // (the host table is freed next)
        }
        try dr.mk.fill(e.t, tok, k, cache, slot);
        dr.msc = .{
            .pv = try a.take(max_streams * dr.mk.n_part * 4),
            .pi = try a.take(max_streams * dr.mk.n_part * 4),
            .send = try a.take(max_streams * 4 * 4),
            .stage = try a.take(max_streams * dr.mk.cols * 2),
        };
        try prompt.fill(e, dr.msc.send, 0, max_streams * 4 * 4);
        dr.mg = try a.take(dr.mk.world * max_streams * 4 * 4);

        dr.out = try a.take(max_streams * (max_block + 1) * 8);
        // the confidence head
        const rank = c.markov_rank;
        dr.embs = try a.take(R * rank * 2);
        dr.xcf = try a.take(R * d * 4);
        dr.ef = try a.take(R * rank * 4);
        dr.conf_in = try a.take(R * (d + rank) * 4);
        dr.confw = try a.take((d + rank) * 4);
        // the confidence head's fp32 weights: the EXL3 lane's checkpoint holds them in fp16, DeepSeek's own in bf16,
        // widened to fp32 when loaded (plan.zig)
        if (e.mx != null) {
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dr.confw, ds.conf, (d + rank) * 4, e.s.handle), "cuMemcpyDtoDAsync");
        } else try e.ops.f16ToF32(e.s, ds.conf, dr.confw, d + rank);
        dr.conf = try a.take(R * 4);
        // the confidence head runs on the host after the pass's one read (cuBLAS's 5 x 5376 fp32 product took ~2 ms of
        // a ~11 ms pass on a GB10): its weight read back once
        dr.confw_host = &.{};
        dr.xc_host = &.{};
        dr.em_host = &.{};
        errdefer dr.deinit();
        dr.confw_host = try std.heap.page_allocator.alloc(f32, d + rank);
        // (the parts start at multiples of 16 in a 16-aligned block: the casts check what the types cannot see)
        dr.ho_host = @alignCast(std.mem.bytesAsSlice(i64, pb[0..ho_bytes]));
        dr.xc_host = @alignCast(std.mem.bytesAsSlice(u16, pb[ho_bytes..][0 .. R * d * 2]));
        dr.em_host = @alignCast(std.mem.bytesAsSlice(u16, pb[ho_bytes + xc_bytes ..][0 .. R * rank * 2]));
        try e.s.synchronize();
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(dr.confw_host.ptr, dr.confw, (d + rank) * 4), "cuMemcpyDtoH");
        // absorbs
        const ring = c.window + ring_extra;
        dr.at = try a.take(ring * dr.taps_w * 2);
        dr.ay = try a.take(ring * d * 2);
        dr.ax = try a.take(ring * d * 2);
        dr.aq = 0;
        dr.ag = 0;
        if (dr.main4) {
            dr.aq = try a.take(ring * (d / 4) * 2);
            dr.ag = try a.take(ring * d * 2);
        }
        dr.ak = @splat(0);
        for (0..ds.blocks.len) |j| dr.ak[j] = try a.take(ring * hd * 2);
        dr.ao = try a.take(ring * hd * 2);
        dr.apos = try a.take(ring * 8);
        dr.arows = try a.take(ring * 8);
        return dr;
    }

    /// The host buffers (the device ones are the arena's).
    pub fn deinit(dr: *Drafter) void {
        const pa = std.heap.page_allocator;
        if (dr.confw_host.len > 0) pa.free(dr.confw_host);
        dr.confw_host = &.{};
        dr.xc_host = &.{};
        dr.em_host = &.{};
        dr.dropVocab();
    }

    fn row(dr: *const Drafter, i: usize) u64 {
        return dr.rows + i * max_rows * 8;
    }

    /// TF_DS_DRAFT_VOCAB off (refused, or the drafter closing): its host table freed, every pass the whole vocabulary's
    /// (its device buffers are the arena's).
    pub fn dropVocab(dr: *Drafter) void {
        if (dr.vocab) |*v| v.bits.deinit(std.heap.page_allocator);
        dr.vocab = null;
        dr.vocab_full = false;
    }

    /// The next pass on TF_DS_DRAFT_VOCAB's list (in effect, and the guard not asking for the whole head).
    pub fn onList(dr: *const Drafter) bool {
        return dr.vocab != null and !dr.vocab_full;
    }

    /// The head's columns the next pass computes on this node: the list's 8,192, else the quarter's.
    pub fn headWidth(dr: *const Drafter, e: *const Engine) usize {
        return if (dr.onList()) draft_vocab.per_node else (dr.ds.head8 orelse e.w.head).n;
    }

    /// TF_DS_DRAFT_VOCAB's checks on this rank, before anything is made: why it cannot run (null: it can). The
    /// four-node split is the 2D one: `two` set and the engine's comm the four nodes' (its `world` is TP2's, 2 inside
    /// a pair). `quarters`: every node's head rows of the vocabulary (Model.open's all-gather), `me` this node's.
    pub fn vocabWhy(dr: *const Drafter, e: *const Engine, quarters: [draft_vocab.nodes][2]usize, me: u32) !?[]const u8 {
        const c = e.c;
        if (draft_vocab.check(c.vocab)) |why| return why;
        if (e.two == null or e.comm.world != draft_vocab.nodes or me >= draft_vocab.nodes) return "needs the four-node split";
        const h8 = dr.ds.head8 orelse return "needs TF_DS_DRAFT_HEAD=fp4";
        if (h8.fmt != .fp4) return "needs TF_DS_DRAFT_HEAD=fp4";
        const q = quarters[me];
        if (q[0] != e.w.vocab_lo or h8.n != q[1] - q[0]) return "its head rows are not its vocabulary quarter";
        _ = draft_vocab.plan(me, quarters, c.vocab) catch return "the nodes' head rows do not tile the vocabulary";
        if (e.t.set) |set| if (try draft_vocab.kitOff(std.heap.page_allocator, SetFinds{ .set = set }, c.*)) |why| return why;
        return null;
    }

    /// TF_DS_DRAFT_VOCAB, this rank's part before the agreement: why it cannot (vocabWhy; null: it can), else the
    /// list's buffers made, its Markov head rows and this node's own head rows in them, its cached bias rows filled, and
    /// the exchange's staging (`staging`, the caller's to free) holding each send's rows of this node's quarter.
    pub fn vocabPrepare(dr: *Drafter, e: *const Engine, a: *prompt.Arena, quarters: [draft_vocab.nodes][2]usize, me: u32, staging: *?cuda.DeviceBuffer) !?[]const u8 {
        const c = e.c;
        dr.vocab = null;
        if (try dr.vocabWhy(e, quarters, me)) |why| return why;
        const h8 = dr.ds.head8.?;
        const q = quarters[me];
        const pl = try draft_vocab.plan(me, quarters, c.vocab);
        const n = draft_vocab.per_node;
        const cb: usize = h8.k / 2; // an FP4 row's codes
        const sb: usize = h8.k / 32; // its scales
        const rank = c.markov_rank;
        var v: Vocab = undefined;
        v.plan = pl;
        v.from_others = pl.fromOthers(me);
        v.flag = try a.take(4);
        try prompt.fill(e, v.flag, 0, 4);
        // the list's ids, and its rows of the Markov head
        v.ids = try a.take(draft_vocab.count * 8);
        const pa = std.heap.page_allocator;
        const ids = try pa.alloc(i64, draft_vocab.count);
        defer pa.free(ids);
        for (ids, 0..) |*x, i| x.* = draft_vocab.id(i);
        try prompt.upload(e, v.ids, ids.ptr, draft_vocab.count * 8);
        const mhead = try a.take(draft_vocab.count * rank * 2);
        try e.ops.gatherRows(e.s, dr.mk.head, c.vocab, v.ids, mhead, rank * 2, draft_vocab.count, v.flag);
        // the staging: the export blob (every round's chunk of it, zeros past its end), then each node's rows of this
        // quarter (int64 [nodes, n])
        v.quarters = quarters;
        v.blob = try draft_vocab.rounds(quarters, cb, sb) * draft_vocab.chunk;
        staging.* = try cuda.DeviceBuffer.alloc(e.d, v.blob + draft_vocab.nodes * n * 8);
        try prompt.fill(e, staging.*.?.ptr, 0, v.blob);
        v.idx = staging.*.?.ptr + v.blob;
        const rows = try pa.alloc(i64, draft_vocab.nodes * n);
        defer pa.free(rows);
        vocabRows(pl, q, rows);
        try prompt.upload(e, v.idx, rows.ptr, rows.len * 8);
        try e.s.synchronize(); // (the host tables are freed next)
        // the list's head rows: this node's own now, the others' in the exchange
        const w = try a.take(n * cb);
        const sc = try a.take(n * sb);
        v.head = .{ .words = w, .w = w, .s = sc, .k = h8.k, .n = n, .fmt = .fp4 };
        if (vocabLand(pl, me, me, v.head)) |own| _ = try vocabPack(e.ops, e.s, h8, v.idx, pl, me, own[0].addr, own[1].addr, v.flag);
        // the Markov loop on the list, with its own cached bias rows of the 256 tokens (the slot table is the same)
        v.mk = draft_vocab.markov(c.*, me, mhead, dr.mk.emb, dr.mk.none);
        const kt = markov_tokens.cached.len;
        const tok = try a.take(kt * 8);
        try prompt.upload(e, tok, &markov_tokens.cached, kt * 8);
        try v.mk.fill(e.t, tok, kt, try a.take(kt * n * 2), dr.mk.slot);
        try e.s.synchronize();
        var bad: u32 = 0;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(@ptrCast(&bad), v.flag, 4), "cuMemcpyDtoH");
        if (bad != 0) return "a row of the list is past its table";
        v.bits = try draft_vocab.Bits.init(pa, c.vocab);
        dr.vocab = v;
        return null;
    }

    /// TF_DS_DRAFT_VOCAB's start exchange, on every rank after the agreement: the rows each other node takes from this
    /// node's quarter gathered into `staging` (the export blob, draft_vocab.blobPart), then the ranks gather their blobs
    /// over the host links one chunk a round (comm.gatherHost, in step) and this node copies each sender's part for it
    /// from the gathered chunk into its list's head (draft_vocab.landings).
    pub fn vocabExchange(dr: *Drafter, e: *const Engine, me: u32, staging: *const cuda.DeviceBuffer) !void {
        const v = &dr.vocab.?;
        const h8 = dr.ds.head8.?;
        const cb: usize = h8.k / 2;
        const sb: usize = h8.k / 32;
        for (0..draft_vocab.nodes) |m| if (m != me) {
            const part = try draft_vocab.blobPart(v.quarters, me, m, cb, sb);
            _ = try vocabPack(e.ops, e.s, h8, v.idx, v.plan, m, staging.ptr + part.codes, staging.ptr + part.scales, v.flag);
        };
        var recv = try cuda.DeviceBuffer.alloc(e.d, draft_vocab.nodes * draft_vocab.chunk);
        defer recv.free();
        for (0..v.blob / draft_vocab.chunk) |round| {
            try e.comm.gatherHost(e.d, staging.ptr + round * draft_vocab.chunk, recv.ptr, draft_vocab.chunk, e.s);
            for (0..draft_vocab.nodes) |j| for (try draft_vocab.landings(v.quarters, me, j, round, cb, sb)) |cp| if (cp) |x| {
                const dst = (if (x.codes) v.head.w else v.head.s) + x.dst;
                try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, recv.ptr + j * draft_vocab.chunk + x.src, x.len, e.s.handle), "cuMemcpyDtoDAsync");
            };
        }
        try e.s.synchronize();
        var bad: u32 = 0;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(@ptrCast(&bad), v.flag, 4), "cuMemcpyDtoH");
        if (bad != 0) return error.RowPastTable;
    }

    /// Drafter.absorb into slot `slot`'s view: the target's taps rows of positions start .. start + n - 1 (bf16
    /// [n, taps * D], contiguous) into every stage's ring, the last ring rows of them: main_x = RMSNorm(main_proj @
    /// taps), each stage's wkv of it normed and roped into its ring row (position % ring).
    pub fn absorb(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *Pool, slot: usize, taps: u64, n: usize, start: usize) !void {
        if (n == 0 or slot >= pool.slots) return error.BadAbsorb;
        const keep = @min(n, pool.ring);
        if (keep > 128) return notPorted(@src()); // main_proj's prompt GEMM (the served lane's replay keeps 128 rows)
        const first = start + n - keep;
        var pos: [max_absorb * 9]i64 = undefined;
        var rows: [max_absorb * 9]i64 = undefined;
        for (0..keep) |r| {
            pos[r] = @intCast(first + r);
            rows[r] = @intCast((first + r) % pool.ring);
        }
        try prompt.upload(e, dr.apos, &pos, keep * 8);
        try prompt.upload(e, dr.arows, &rows, keep * 8);
        ddTaps(e, dr, taps + (n - keep) * dr.taps_w * 2, pos[0..keep]);
        try dr.mainKeys(e, ch, taps + (n - keep) * dr.taps_w * 2, keep);
        for (dr.ds.blocks, 0..) |lay, j| {
            try tri_norm.kvNormRope(e.t, dr.ak[j], lay.kv_norm, e.plain.cos, e.plain.sin, dr.apos, pool.view(j, slot), pool.ring, dr.arows, e.c.eps, true, e.c.rope_dim, dr.ao, keep, e.c.head_dim);
        }
        pool.absorbed[slot] = start + n;
    }

    /// Drafter.absorb_many: every item's rows in one pass over the pool's planes (ring row slot * ring + position %
    /// ring); the caller sets each slot's `absorbed` (eager: the kept length, known after sampling).
    pub fn absorbMany(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, items: []const Item) !void {
        var n: usize = 0;
        var pos: [max_absorb]i64 = undefined;
        var rows: [max_absorb]i64 = undefined;
        for (items) |it| {
            if (it.slot >= pool.slots or n + it.n > max_absorb) return error.BadAbsorb;
            try e.ops.copyRows(e.s, it.taps, dr.taps_w * 2, dr.at + n * dr.taps_w * 2, dr.taps_w * 2, dr.taps_w * 2, it.n);
            for (0..it.n) |r| {
                const p = it.start + r;
                pos[n + r] = @intCast(p);
                rows[n + r] = @intCast(it.slot * pool.ring + p % pool.ring);
            }
            n += it.n;
        }
        if (n == 0) return;
        try prompt.upload(e, dr.apos, &pos, n * 8);
        try prompt.upload(e, dr.arows, &rows, n * 8);
        ddTaps(e, dr, dr.at, pos[0..n]);
        try dr.mainKeys(e, ch, dr.at, n);
        for (dr.ds.blocks, 0..) |lay, j| {
            try tri_norm.kvNormRope(e.t, dr.ak[j], lay.kv_norm, e.plain.cos, e.plain.sin, dr.apos, pool.rings[j], pool.slots * pool.ring, dr.arows, e.c.eps, true, e.c.rope_dim, dr.ao, n, e.c.head_dim);
        }
    }

    /// main_x = RMSNorm(main_proj @ taps) of n rows, then every stage's wkv of it (stage_kv: one group).
    fn mainKeys(dr: *Drafter, e: *const Engine, ch: *const Chunk, taps: u64, n: usize) !void {
        const c = e.c;
        const d = c.hidden;
        const hd = c.head_dim;
        // (decode-size absorbs, a round's window rows: the GEMV; a prompt chunk's: the GEMM)
        if (dr.main4) {
            // the node's quarter of the columns, the four gathered as the stages' partials are, each plane into its
            // half of the rows (plane r: quarters 2r and 2r + 1, weights.mainProjCols)
            const tw = e.two.?;
            const q = d / 4;
            try prompt.groupedDecode(e, ch, n, &.{dr.ds.main_proj}, &.{taps}, &.{dr.taps_w}, &.{dr.aq}, &.{q}, &.{.bf16});
            try tw.quartersDecode(e, dr.aq, dr.ag, n, .{ q, q }, 2);
            for (0..2) |r| try e.ops.copyRows(e.s, dr.ag + r * n * 2 * q * 2, 2 * q * 2, dr.ay + r * 2 * q * 2, d * 2, 2 * q * 2, n);
        } else try prompt.groupedDecode(e, ch, n, &.{dr.ds.main_proj}, &.{taps}, &.{dr.taps_w}, &.{dr.ay}, &.{d}, &.{.bf16});
        ddMain(e, dr.ay, n * d * 2);
        try tri_basic.rmsnorm(e.t, dr.ay, d, dr.ds.main_norm, dr.ax, d, c.eps, n, d);
        const S = dr.ds.blocks.len;
        var ls: [max_stages]weights.Linear = undefined;
        var xs: [max_stages]u64 = undefined;
        var ldxs: [max_stages]usize = undefined;
        var ldos: [max_stages]usize = undefined;
        var types: [max_stages]exl3_linear.DType = undefined;
        for (dr.ds.blocks, 0..) |lay, j| {
            ls[j] = lay.wkv;
            xs[j] = dr.ax;
            ldxs[j] = d;
            ldos[j] = hd;
            types[j] = .bf16;
        }
        try prompt.groupedDecode(e, ch, n, ls[0..S], xs[0..S], ldxs[0..S], dr.ak[0..S], ldos[0..S], types[0..S]);
    }

    /// BatchDraftGraph.run for N streams (tokens at positions q0 in slots): each stream's `steps` drafts and their
    /// confidences into drafts / confs. Each slot's rings hold its positions before q0 (absorbed).
    pub fn pass(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize, probe: ?Probe) !void {
        const N = tokens.len;
        const n = dr.n;
        const R = N * n;
        if (N == 0 or N > max_streams or R > max_rows or q0.len != N or slots.len != N) return notPorted(@src());
        if (steps == 0 or steps > n) return error.BadDraftPass;
        // _body's index rows: ids [token, noise ...], positions q0 + j, window positions q0 - 1, ring bases
        var hv: [6 * max_rows]i64 = @splat(0);
        var hb: [max_rows * pick_pad]i64 = @splat(-1);
        const bw: usize = if (R > tri.decode_rows) pick_pad else n; // the pick list's columns
        for (0..N) |i| {
            if (slots[i] < 0 or slots[i] >= pool.slots) return error.BadSlot;
            for (0..n) |j| {
                const r = i * n + j;
                hv[0 * max_rows + r] = if (j == 0) tokens[i] else dr.noise;
                hv[1 * max_rows + r] = q0[i] + @as(i64, @intCast(j));
                hv[2 * max_rows + r] = q0[i] - 1;
                hv[3 * max_rows + r] = slots[i] * @as(i64, @intCast(pool.ring));
                hv[4 * max_rows + r] = 0;
                hv[5 * max_rows + r] = -1;
                for (0..n) |k| hb[r * bw + k] = @intCast(i * n + k);
            }
        }
        try prompt.upload(e, dr.rows, &hv, hv.len * 8);
        try prompt.upload(e, dr.bidx, &hb, R * bw * 8);
        // the Markov loop's out [N, steps + 1]: each stream's last token first
        const ts = steps + 1;
        var ho: [max_streams * (max_block + 1)]i64 = @splat(0);
        for (0..N) |i| ho[i * ts] = tokens[i];
        try prompt.upload(e, dr.out, &ho, N * ts * 8);
        var dprobe: DumpProbe = .{ .e = e, .dr = dr, .pool = pool, .R = R, .N = N, .steps = steps };
        const dumping = ddump != null and ddump_count == ddump_pass and probe == null;
        if (ddump != null) ddump_count += 1;
        if (dumping) {
            ddump_on = true;
            defer ddump_on = false;
            dd(e, "rows", 0, dr.rows, hv.len * 8);
            try dr.body(e, ch, pool, N, steps, .{ .ctx = &dprobe, .at = DumpProbe.at });
            dd(e, "out", 0, dr.out, N * ts * 8);
            if (ddump_node == 0) {
                var mb: [512]u8 = undefined;
                const path = std.mem.printSentinel(&mb, "{s}/meta.txt", .{ddump.?}, 0) catch unreachable;
                if (std.c.fopen(path.ptr, "wb")) |f| {
                    defer _ = std.c.fclose(f);
                    var tb: [1024]u8 = undefined;
                    const s = std.fmt.bufPrint(&tb, "N {d}\nsteps {d}\nn {d}\nslots {d}\nring {d}\nworld {d}\nhead_n {d}\nvocab_lo {d}\ntoken {d}\nq0 {d}\nslot {d}\n", .{ N, steps, n, pool.slots, pool.ring, e.world, e.w.head.n, e.w.vocab_lo, tokens[0], q0[0], slots[0] }) catch unreachable;
                    _ = std.c.fwrite(s.ptr, 1, s.len, f);
                }
            }
        } else if (dprof and probe == null) {
            pmark(e, .start);
            try dr.body(e, ch, pool, N, steps, null);
            prof_passes += 1;
        } else if (dr.graphs != null and probe == null) try dr.graphs.?.run(dr, e, ch, pool, N, steps) else try dr.body(e, ch, pool, N, steps, probe);
        var hc: [max_streams * max_block]f32 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&ho, dr.out, N * ts * 8), "cuMemcpyDtoH");
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&hc, dr.conf, N * steps * 4), "cuMemcpyDtoH");
        if (dumping) ddHost("conf", std.mem.sliceAsBytes(hc[0 .. N * steps]));
        for (0..N) |i| for (0..steps) |j| {
            dr.drafts[i][j] = ho[i * ts + 1 + j];
            dr.confs[i][j] = hc[i * steps + j];
        };
        dr.streams = N;
        dr.steps = steps;
        dr.passes += 1;
        if (dlog) |path| if (ddump_node == 0) {
            // TF_DS_DRAFT_LOG: a line a stream of the pass: slot q0 token, its drafts, its confidences, then the pass's
            // number, its streams and steps, and each stage's active experts (passes matched by their composition)
            var na: [max_stages]i32 = @splat(0);
            const S = dr.ds.blocks.len;
            if (dr.nacts != 0) try e.d.check(e.d.api.cuMemcpyDtoH_v2(&na, dr.nacts, S * 4), "cuMemcpyDtoH");
            if (std.c.fopen(path.ptr, "ab")) |f| {
                defer _ = std.c.fclose(f);
                for (0..N) |i| {
                    var lb: [640]u8 = undefined;
                    var at: usize = 0;
                    at += (std.fmt.bufPrint(lb[at..], "{d} {d} {d}", .{ slots[i], q0[i], tokens[i] }) catch break).len;
                    for (0..steps) |j| at += (std.fmt.bufPrint(lb[at..], " {d}", .{dr.drafts[i][j]}) catch break).len;
                    for (0..steps) |j| at += (std.fmt.bufPrint(lb[at..], " {d:.4}", .{dr.confs[i][j]}) catch break).len;
                    at += (std.fmt.bufPrint(lb[at..], " | pass {d} streams {d} steps {d} experts", .{ dr.passes, N, steps }) catch break).len;
                    for (na[0..S]) |x| at += (std.fmt.bufPrint(lb[at..], " {d}", .{x}) catch break).len;
                    lb[at] = '\n';
                    _ = std.c.fwrite(&lb, 1, at + 1, f);
                }
            }
        };
    }

    /// DSparkConfidenceHead on the host: conf[i * steps + j] = [xc row (i, j), the Markov embedding row of the token step
    /// j read] . w, fp32 products summed in fp64 (the collapsed rows and the embedding rows read back as bf16).
        const d = e.c.hidden;
        const rank = e.c.markov_rank;
        const n = dr.n;
        const xc = dr.xc_host;
        const em = dr.em_host;
        const wv = dr.confw_host;
        for (0..N) |i| for (0..steps) |j| {
            var acc: f64 = 0;
            const xr = xc[(i * n + j) * d ..][0..d];
            for (xr, wv[0..d]) |x, wk| acc += @as(f64, bf(x) * wk);
            const er = em[(i * steps + j) * rank ..][0..rank];
            for (er, wv[d..][0..rank]) |x, wk| acc += @as(f64, bf(x) * wk);
            out[i * steps + j] = @floatCast(acc);
        };
    }

    /// The pass's device work for N streams of `steps` drafts, from the rows' embeddings to the confidence head (its
    /// inputs uploaded: the index rows, the pick lists, the Markov loop's first column): what a pass graph captures.
    fn body(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, N: usize, steps: usize, probe: ?Probe) !void {
        const c = e.c;
        const t = e.t;
        const n = dr.n;
        const R = N * n;
        const d = c.hidden;
        // the streams: every row's token embedding in each stream, pre (1, 0, 0, 0)
        try tri_basic.embedInit(t, e.w.embed, dr.row(0), dr.h, dr.pre, R, d, c.hc);
        pmark(e, .embed);
        // _stages_fused
        var h = dr.h;
        var spare = dr.h_alt;
        var pre = dr.pre;
        var pre_f = dr.pre_f;
        var pending: ?u64 = null;
        for (dr.ds.blocks, 0..) |lay, j| {
            h = try dr.mix(e, h, &spare, &pending, lay.hc_attn, pre, lay.attn_norm, dr.pre_a, R);
            pmark(e, .mix_a);
            dd(e, "h_a", j, h, R * c.hc * d * 2);
            dd(e, "pre_in", j, pre, R * c.hc * 4);
            dd(e, "pre_a", j, dr.pre_a, R * c.hc * 4);
            try Probe.check(probe, .attn_in, j, dr.x);
            try Probe.check(probe, .attn_ring, j, pool.rings[j]);
            try dr.attention(e, ch, pool, j, R);
            try Probe.check(probe, .attn_out, j, dr.pa);
            try dr.stageGather(e, dr.pa, dr.ga, R, .attn);
            pmark(e, if (j == 0) .gather_a0 else .gather_a);
            try Probe.check(probe, .attn_gather, j, dr.ga);
            pending = dr.ga;
            h = try dr.mix(e, h, &spare, &pending, lay.hc_ffn, dr.pre_a, lay.ffn_norm, pre_f, R);
            pmark(e, .mix_f);
            dd(e, "h_f", j, h, R * c.hc * d * 2);
            dd(e, "pre_f", j, pre_f, R * c.hc * 4);
            try Probe.check(probe, .moe_in, j, dr.x);
            try dr.moe(e, j, R);
            try Probe.check(probe, .moe_out, j, dr.pm);
            if (dr.nacts != 0) try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dr.nacts + j * 4, dr.mxe.nact, 4, e.s.handle), "cuMemcpyDtoDAsync");
            try dr.stageGather(e, dr.pm, dr.gm, R, .moe);
            pmark(e, if (j == 0) .gather_m0 else .gather_m);
            try Probe.check(probe, .moe_gather, j, dr.gm);
            pending = dr.gm;
            std.mem.swap(u64, &pre, &pre_f);
        }
        try tri_basic.hcPost(t, dr.gm, h, dr.post, dr.comb, h, e.world, R, d);
        pmark(e, .post);
        const S = dr.ds.blocks.len;
        try Probe.check(probe, .stages_h, S, h);
        try Probe.check(probe, .stages_pre, S, pre);
        // the head on every row (this rank's columns: the Markov loop scores them without a gather)
        try tri_basic.collapse(t, h, pre, dr.xc, R, d);
        try tri_basic.rmsnorm(t, dr.xc, d, dr.ds.norm, dr.xn, d, c.eps, R, d);
        dd(e, "xc", 0, dr.xc, R * d * 2);
        dd(e, "xn", 0, dr.xn, R * d * 2);
        // the list's head rows (TF_DS_DRAFT_VOCAB), else the drafter's FP8 / FP4 copy (weights.DSpark.head8), else the
        // backbone's
        const list: ?*const Vocab = if (dr.onList()) &dr.vocab.? else null;
        const head = if (list) |v| v.head else dr.ds.head8 orelse e.w.head;
        try prompt.groupedDecode(e, ch, R, &.{head}, &.{dr.xn}, &.{d}, &.{dr.local}, &.{head.n}, &.{.f32});
        try Probe.check(probe, .local, S, dr.local);
        pmark(e, .head);
        // the Markov loop: out [N, steps + 1], each stream's last token first
        const ts = steps + 1;
        var g: MarkovGather = .{ .e = e, .dst = dr.mg };
        if (list) |v| {
            var ids: ListIds = .{ .e = e, .ids = v.ids };
            try draft_vocab.steps(v.mk, t, dr.local, dr.out, ts, N, n, steps, dr.msc, .{ .ctx = &g, .run = MarkovGather.run }, .{ .ctx = &ids, .run = ListIds.run });
        } else try dr.mk.steps(t, dr.local, dr.out, ts, N, n, steps, dr.msc, .{ .ctx = &g, .run = MarkovGather.run });
        try Probe.check(probe, .markov, S, dr.out);
        pmark(e, .markov);
        // the confidence head: [xc (fp32), the embedding rows the steps read (fp32)] @ conf^T, a stream's first steps rows
        const rank = c.markov_rank;
        const w = d + rank;
        try e.d.check(e.d.api.cuMemsetD8Async(ch.invalid, 0, 4, e.s.handle), "cuMemsetD8Async");
        // (the product itself on the host: pass)
        _ = w;
        for (0..N) |i| {
            try e.ops.gatherRows(e.s, dr.ds.markov_embed, c.vocab, dr.out + i * ts * 8, dr.embs + i * steps * rank * 2, rank * 2, steps, ch.invalid);
        }
        pmark(e, .conf);
    }

    /// A stage's sublayer partials gathered into dst [2, R, D] in TP2 rank order: four ways, the backbone's quarters of
    /// a decode exchange (the node's columns, wo_b's or down's, from every node); else the pair gather.
    fn stageGather(dr: *const Drafter, e: *const Engine, src: u64, dst: u64, R: usize, part: enum { attn, moe }) !void {
        if (dr.four) {
            const tw = e.two.?;
            return tw.quartersDecode(e, src, dst, R, if (part == .attn) tw.ow else tw.dw, 4);
        }
        return pairGather(e, src, dst, R, e.c.hidden);
    }

    /// pre_mix of _stages_fused: hc_pre2 of the streams h with the pending gathered partials posted in (into the spare
    /// buffer, which becomes the streams; no side stream in the drafter); returns the streams.
    fn mix(dr: *Drafter, e: *const Engine, h: u64, spare: *u64, pending: *?u64, params: [3]u64, pre_in: u64, norm: u64, pre_out: u64, R: usize) !u64 {
        const c = e.c;
        if (pending.*) |gp| {
            const out = try tri_hc.hcPre2(e.t, h, params[0], params[1], params[2], pre_in, norm, c.eps, c.hc_eps, c.hc_iters, dr.x, pre_out, dr.post, dr.comb, dr.part, .{ .gathered = gp, .world = e.world, .h_out = spare.* }, null, R, c.hidden);
            pending.* = null;
            spare.* = h;
            return out;
        }
        return tri_hc.hcPre2(e.t, h, params[0], params[1], params[2], pre_in, norm, c.eps, c.hc_eps, c.hc_iters, dr.x, pre_out, dr.post, dr.comb, dr.part, null, null, R, c.hidden);
    }

    /// BatchDraftGraph._attention (fused): stage j's attention partial pa [R, D] fp32 from its mixed rows x.
    /// sparseAttn's arguments for stage j's R rows of hl heads (attention): each row sees its stream's 128 newest
    /// absorbed positions (to q0 - 1) and every block row of its stream; four ways the key-split part's variant from the
    /// kit's extra set (H = the node's heads, which the base set lacks).
    fn sattnArgs(dr: *const Drafter, c: *const Config, sink: u64, pool: *const Pool, j: usize, R: usize, hl: usize) tri_attn.Attn {
        return .{
            .q = dr.q,
            .out = dr.o,
            .rows = R,
            .h = hl,
            .hd = c.head_dim,
            .sink = sink,
            .wsrc = pool.rings[j],
            .wsrc_rows = pool.slots * pool.ring,
            .wlo = dr.wlo,
            .ring = true,
            .comp = .{ .bf16 = dr.kvb },
            .idx = dr.bidx,
            .n_idx = dr.n,
            .pos = dr.row(2),
            .scale = prompt.scale(c.head_dim),
            .window = c.window,
            .wbase = dr.row(3),
            .cbase = dr.row(4),
            .ring_rows = pool.ring,
            .parts = .{ .pm = dr.pm_, .pl = dr.pl_, .po = dr.po_ },
            .extra = dr.four,
        };
    }

    fn attention(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, j: usize, R: usize) !void {
        const c = e.c;
        const t = e.t;
        const lay = dr.ds.blocks[j];
        const hd = c.head_dim;
        const rd = c.rope_dim;
        const hl = lay.wq_b.n / hd; // the block's heads (TP2's on a 2D node, the node's own four ways)
        const rope = e.plain; // a window-only stage (Model._cs: ratio 0)
        const pos = dr.row(1);
        // attn_in(comp=False): wq_a and wkv, one group
        try prompt.groupedDecode(e, ch, R, &.{ lay.wq_a, lay.wkv }, &.{ dr.x, dr.x }, &.{ c.hidden, c.hidden }, &.{ dr.qa, dr.ykv }, &.{ lay.wq_a.n, hd }, &.{ .bf16, .bf16 });
        pmark(e, .qkv_a);
        if (e.mx != null) {
            // Format.native: q's RMSNorm as plain rows, wq_b on them, q's RoPE
            try tri_basic.rmsnorm(t, dr.qa, lay.wq_a.n, lay.q_norm, dr.qr, lay.wq_a.n, c.eps, R, lay.wq_a.n);
            try prompt.groupedDecode(e, ch, R, &.{lay.wq_b}, &.{dr.qr}, &.{lay.wq_a.n}, &.{dr.q}, &.{lay.wq_b.n}, &.{.bf16});
            try tri_norm.ropeHeads(t, dr.q, rope.cos, rope.sin, pos, rd, false, R, hl, hd);
            pmark(e, .q_b);
            dd(e, "qa", j, dr.qa, R * lay.wq_a.n * 2);
            dd(e, "ykv", j, dr.ykv, R * hd * 2);
            dd(e, "qr", j, dr.qr, R * lay.wq_a.n * 2);
            dd(e, "q", j, dr.q, R * hl * hd * 2);
        } else {
            // q_proj with rope (rot_q): q's RMSNorm writes wq_b's rotated rows; wq_b's epilogue applies q's RoPE
            const rot = [_]tri_norm.Rot{.{ .suh = lay.wq_b.suh, .h = dr.xhq }};
            try tri_norm.rmsnormRot(t, dr.qa, lay.wq_a.n, lay.q_norm, c.eps, &rot, dr.qr, lay.wq_a.n, R, lay.wq_a.n);
            var qc = [_]exl3_linear.Call{.{ .layer = lay.wq_b, .x = 0, .ldx = 0, .x_dtype = .bf16, .xh = dr.xhq, .y = dr.q, .ldy = @intCast(lay.wq_b.n), .y_dtype = .bf16, .counters = 0, .rope = .{ .cos = rope.cos, .sin = rope.sin, .pos = pos, .hd = @intCast(hd), .rd = @intCast(rd) } }};
            try prompt.groupedRotated(e, ch, R, &qc);
        }
        // the block rows' keys: normed and roped, stored nowhere (ring row -1)
        const plane_rows = pool.slots * pool.ring;
        try tri_norm.kvNormRope(t, dr.ykv, lay.kv_norm, rope.cos, rope.sin, pos, pool.rings[j], plane_rows, dr.row(5), c.eps, true, rd, dr.kvb, R, hd);
        pmark(e, .kv_rope);
        try tri_attn.sparseAttn(t, dr.sattnArgs(c, lay.sink, pool, j, R, hl));
        pmark(e, .sattn);
        dd(e, "kvb", j, dr.kvb, R * hd * 2);
        dd(e, "o_rope", j, dr.o, R * hl * hd * 2);
        try tri_norm.ropeHeads(t, dr.o, rope.cos, rope.sin, pos, rd, true, R, hl, hd);
        dd(e, "o", j, dr.o, R * hl * hd * 2);
        const uw = lay.groups * lay.wo_a[0].n;
        if (e.mx) |k| {
            // Format.native: wo_a's groups (o quant-dequantized, bf16 einsum; the FP8 rows (weights.draftWoAFp8): each
            // group's act_quant(o) through its FP8 GEMV, the same products), then wo_b (fp32)
            if (if (k.dec_on) lay.wo_a8 else null) |w8| {
                // the decode kernels' FP8 copy of wo_a (weights.zig wo_a8): one launch for the groups
                try mx.woA8(k, e.s, ch.mxs, w8, lay.groups, dr.o, hl * hd, R, dr.u, uw, dr.mxe.z);
            } else if (lay.wo_a[0].fmt == .fp8) {
                const gk: usize = lay.wo_a[0].k;
                for (lay.wo_a[0..lay.groups], 0..) |wo, g| try mx.linear(k, e.s, ch.mxs, wo, dr.o + g * gk * 2, .bf16, hl * hd, R, dr.u + g * wo.n * 2, uw, .bf16, false, dr.mxe.z);
            } else try mx.woA(k, e.s, ch.mxs, lay.wo_a[0..lay.groups], dr.o, hl * hd, R, dr.u, uw, dr.mxe.z);
            dd(e, "u", j, dr.u, R * uw * 2);
            pmark(e, .wo_a);
            if (dr.four) {
                // four ways: the rank's four groups from the column partner, then wo_b for this node's columns over all
                // of them (round.zig attention)
                const tw = e.two.?;
                try tw.catRank(e, dr.u, tw.u_full, R, tw.uw, 2);
                try prompt.groupedDecode(e, ch, R, &.{lay.wo_b}, &.{tw.u_full}, &.{tw.uw[0] + tw.uw[1]}, &.{dr.pa}, &.{lay.wo_b.n}, &.{.f32});
            } else try prompt.groupedDecode(e, ch, R, &.{lay.wo_b}, &.{dr.u}, &.{uw}, &.{dr.pa}, &.{lay.wo_b.n}, &.{.f32});
            pmark(e, .wo_b);
            return;
        }
        // wo_ab: wo_a's inputs rotated from o's column blocks, its epilogue writing wo_b's rotated rows; wo_b (fp32)
        var calls: [exl3_linear.gmax]exl3_linear.Call = undefined;
        const gw = hl * hd / lay.groups;
        var col: usize = 0;
        var off: usize = 0;
        for (lay.wo_a[0..lay.groups], 0..) |wo, gi| {
            calls[gi] = .{ .layer = wo, .x = dr.o + gi * gw * 2, .ldx = @intCast(hl * hd), .x_dtype = .bf16, .xh = ch.gxh + off * 2, .y = dr.u + col * 2, .ldy = @intCast(uw), .y_dtype = .bf16, .counters = try prompt.counterOf(ch, wo), .rot = .{ .suh = lay.wo_b.suh, .xh = dr.xb, .ldr = @intCast(lay.wo_b.k), .off = @intCast(col) } };
            col += wo.n;
            off += R * wo.k;
        }
        try exl3_linear.rotMany(e.lin, e.s, calls[0..lay.groups], R, true);
        try exl3_linear.glinear(e.lin, e.s, calls[0..lay.groups], R, ch.gz, true, true);
        var cb = [_]exl3_linear.Call{.{ .layer = lay.wo_b, .x = 0, .ldx = 0, .x_dtype = .bf16, .xh = dr.xb, .y = dr.pa, .ldy = @intCast(lay.wo_b.n), .y_dtype = .f32, .counters = 0 }};
        try prompt.groupedRotated(e, ch, R, &cb);
    }

    /// Model.moe(lay, x, topk) of a pass's rows: the gate's chunk sums (rowmm_gate; past decode_rows rows the gate's fp32
    /// matmul), the routing at the stages' top-k with the shared expert in every row's last slot, the routed experts'
    /// decode path on the stages' own scratch.
    fn moe(dr: *Drafter, e: *const Engine, j: usize, R: usize) !void {
        const c = e.c;
        const lay = dr.ds.blocks[j];
        const ne: usize = lay.experts.count - 1;
        if (e.mx) |k| {
            // Format.native: Gate.forward's fp32 logits, the routing at the stages' top-k, the experts
            try mx.bf16Gemm(k, e.s, dr.x, c.hidden, lay.gate_w, c.hidden, R, ne, c.hidden, dr.gl, ne, .f32, 1, false, 1, 0, 0, 0, dr.mxe.z);
            try tri_norm.route(e.t, dr.gl, 0, lay.gate_b, dr.topk, c.routed_scaling, lay.experts.count - 1, dr.pick, dr.mw, R, ne, dr.slots_moe);
            dd(e, "gl", j, dr.gl, R * ne * 4);
            dd(e, "pick", j, dr.pick, R * dr.slots_moe * 4);
            dd(e, "mw", j, dr.mw, R * dr.slots_moe * 4);
            pmark(e, .gate);
            try mx.gateUp(k, e.s, e.d, lay.experts, dr.mxe, dr.x, c.hidden, dr.pick, dr.mw, R, c.swiglu_limit, dr.nxh, lay.experts.width, true);
            pmark(e, .experts_gu);
            if (dr.four) {
                // four ways: the packs exchanged and assembled, down over the node's output columns (round.zig moe)
                const tw = e.two.?;
                const ex = lay.experts;
                const rows = R * dr.slots_moe;
                if (tw.ir != 0 and k.decode_fuse) {
                    try tw.assembleParityCodes(e, k, ex, dr.nxh, dr.pick, rows, dr.mxe);
                    try mx.downDecCodes(k, e.s, ex, dr.mxe, dr.pick, R, dr.pm, tw.dw[tw.p], false);
                } else {
                    if (tw.ir != 0) try tw.assembleParity(e, ex, dr.nxh, dr.pick, rows) else try tw.catRank(e, dr.nxh, tw.xd_full, rows, tw.gu, 2);
                    try mx.down(k, e.s, ex, dr.mxe, tw.xd_full, ex.down_k, dr.pick, R, dr.pm, tw.dw[tw.p], false, true);
                }
            } else try mx.down(k, e.s, lay.experts, dr.mxe, dr.nxh, lay.experts.width, dr.pick, R, dr.pm, c.hidden, false, true);
            pmark(e, .experts_d);
            return;
        }
        var kc: usize = 0;
        if (R <= tri.decode_rows) {
            kc = try tri_norm.rowmmGate(e.t, dr.x, c.hidden, lay.gate_w, dr.gl, R, c.hidden, ne);
        } else {
            // x.float() @ gate_w.float().t() through cuBLAS, then the plain routing
            try e.ops.toF32(e.s, dr.x, dr.xf, R * c.hidden);
            try e.blas.xwT(dr.xf, dr.gate_f[j], dr.gl, R, c.hidden, ne);
        }
        try tri_norm.route(e.t, dr.gl, kc, lay.gate_b, dr.topk, c.routed_scaling, lay.experts.count - 1, dr.pick, dr.mw, R, ne, dr.slots_moe);
        // routed(): fewer than EXACT_ROWS rows take the decode window's fused path, more the chunk's (13+ streams)
        if (R < exl3_experts.exact_rows) return exl3_experts.decode(e.ex, e.s, lay.experts, dr.xsd, dr.x, c.hidden, dr.pick, dr.mw, dr.pm, R, c.swiglu_limit);
        try exl3_experts.prompt(e.ex, e.s, lay.experts, dr.xs, dr.x, c.hidden, dr.pick, dr.mw, dr.pm, R, c.swiglu_limit);
    }
};

/// MultiDecoder._depth: the most drafts a stream verifies in a round of `live` streams (TF_DS_PARALLEL_DEPTH unset:
/// 5, 5, 3, 3), at most the engine's drafts and the rows a stream may take of a round (max_rows / live - 1).
pub fn depth(live: usize, drafts: usize, round_rows: usize) usize {
    const by = [_]usize{ 5, 5, 3, 3 };
    const k = by[@min(live, by.len) - 1];
    return @min(@min(k, drafts), (round_rows / @max(live, 1)) -| 1);
}

/// The batched pass's Markov steps for N drafting streams (BatchDraftGraph's `steps` under the served "conf" policy):
/// as deep as any round with N or more live streams may verify.
pub fn passSteps(n_streams: usize, slots: usize, drafts: usize, round_rows: usize, block: usize) usize {
    var s: usize = 0;
    var live = n_streams;
    while (live <= slots) : (live += 1) s = @max(s, depth(live, drafts, round_rows));
    return @max(1, @min(block, s));
}

/// The served ROUND_MS table (TF_DS_ROUND_MS unset): a round's forward ms by its rows, and DRAFT_MS.
pub const round_ms = [_]f64{ 30.0, 35.2, 39.9, 44.9, 48.6, 52.1, 55.6, 59.1, 62.7, 66.3, 69.6, 72.9, 75.5, 78.2, 81.3, 84.4, 87.5, 90.6, 93.7, 96.8, 99.9, 103.0, 106.1, 109.2 };
pub const draft_ms: f64 = 3.5;

extern "c" fn exp(x: f64) f64;

/// TF_DS_CONF_CAL="a,b": a draft's landing chance (once the earlier drafts landed) is sigmoid(a conf + b) (a 1, b 0:
/// the served rule, the confidence's own sigmoid). Unset: a 1, b 0.19, the logistic fit of 399 logged rounds of the
/// four-node native lane (prose and code, all five drafts verified; the head is well calibrated, a shade low).
pub var conf_cal: [2]f64 = .{ 1.0, 0.19 };
/// TF_DS_DRAFT_MS: a round's ms past its forward that chooseK adds (unset: the four-node native lane's measured pass
/// and absorb, ~6.0 + 0.55 ms after this work; the served EXL3 lane's DRAFT_MS was 3.5).
pub var cost_ms: f64 = 6.6;

/// A draft's landing chance from its confidence (conf_cal).
pub fn landing(c: f32) f64 {
    return 1.0 / (1.0 + exp(-(conf_cal[0] * @as(f64, c) + conf_cal[1])));
}

/// MultiDecoder._choose_k: the k in 0 .. kmax with the most expected tokens a millisecond (1 + the prefix survivals'
/// sum, survival the sigmoid of each confidence, over the round's table cost with the other streams at their table
/// depth), in Python's float64 arithmetic (libm's exp, as Python's math.exp calls it); the first best wins.
pub fn chooseK(conf: []const f32, kmax: usize, live: usize, drafts: usize, round_rows: usize) usize {
    return chooseKWith(conf, kmax, live, drafts, round_rows, &round_ms, draft_ms);
}

/// chooseK over a round table by rows (`table`, the lane's TF_DS_ROUND_MS) and a round's ms past its forward `dms`,
/// each draft's survival its landing chance (conf_cal).
pub fn chooseKWith(conf: []const f32, kmax: usize, live: usize, drafts: usize, round_rows: usize, table: []const f64, dms: f64) usize {
    const others = (live - 1) * (depth(live, drafts, round_rows) + 1);
    var best: usize = 0;
    var best_v: f64 = -1.0;
    var surv: f64 = 1.0;
    var exp_tokens: f64 = 1.0;
    for (0..kmax + 1) |kk| {
        if (kk > 0) {
            surv *= landing(conf[kk - 1]);
            exp_tokens += surv;
        }
        const rows = @min(table.len, others + kk + 1);
        const fwd = table[rows - 1];
        const v = exp_tokens * @as(f64, @floatFromInt(live)) / (fwd + dms);
        if (v > best_v) {
            best = kk;
            best_v = v;
        }
    }
    return best;
}

test "draft depths and pass steps as multi.py takes them" {
    // 16-row rounds, 5 drafts: 5 alone or beside one other stream, 3 at three or four
    try std.testing.expectEqual(@as(usize, 5), depth(1, 5, 16));
    try std.testing.expectEqual(@as(usize, 5), depth(2, 5, 16));
    try std.testing.expectEqual(@as(usize, 3), depth(3, 5, 16));
    try std.testing.expectEqual(@as(usize, 3), depth(4, 5, 16));
    try std.testing.expectEqual(@as(usize, 5), passSteps(1, 4, 5, 16, 5));
    try std.testing.expectEqual(@as(usize, 5), passSteps(2, 4, 5, 16, 5));
    try std.testing.expectEqual(@as(usize, 3), passSteps(3, 4, 5, 16, 5));
    try std.testing.expectEqual(@as(usize, 3), passSteps(4, 4, 5, 16, 5));
}

/// A stage's attention launches (attention's Triton calls on Format.native, in its order: q's RoPE, the keys' norm and
/// RoPE, the sparse attention's part and merge, o's inverse RoPE) for R rows of hl heads at the served arguments (the
/// arena's 256-byte aligned buffers, the lane's 16-stream pool), `four` ways or TP2, into `log`.
pub fn logStageAttn(log: *tri.Log, c: *const Config, hl: usize, R: usize, four: bool) !void {
    const at = struct {
        fn buf(i: u64) u64 {
            return 0x7f00_0000_0000 + i * 0x0100_0000;
        }
    }.buf;
    const pool: Pool = .{ .rings = .{ at(1), at(2), at(3), 0 }, .stages = 3, .slots = max_streams, .ring = c.window + ring_extra, .hd = c.head_dim };
    var dr: Drafter = undefined;
    dr.q = at(4);
    dr.o = at(5);
    dr.kvb = at(6);
    dr.ykv = at(7);
    dr.bidx = at(8);
    dr.wlo = at(9);
    dr.rows = at(10);
    dr.pm_ = at(11);
    dr.pl_ = at(12);
    dr.po_ = at(13);
    dr.n = c.dspark_block;
    dr.four = four;
    const cos = at(14);
    const sin = at(15);
    const pos = dr.row(1);
    const t: tri.Tri = .{ .log = log };
    try tri_norm.ropeHeads(t, dr.q, cos, sin, pos, c.rope_dim, false, R, hl, c.head_dim);
    try tri_norm.kvNormRope(t, dr.ykv, at(16), cos, sin, pos, pool.rings[0], pool.slots * pool.ring, dr.row(5), c.eps, true, c.rope_dim, dr.kvb, R, c.head_dim);
    try tri_attn.sparseAttn(t, dr.sattnArgs(c, at(17), &pool, 0, R, hl));
    try tri_norm.ropeHeads(t, dr.o, cos, sin, pos, c.rope_dim, true, R, hl, c.head_dim);
}

/// TF_DS_DRAFT_4WAY's kit condition: null when `finds` (the kit's extra set: has(name, args, consts)) holds every launch
/// a four-way stage of hl heads takes from it at the served arguments (a pass of at most decode_rows rows splits the
/// keys, a larger one not), else why not.
pub fn fourWayKitOff(gpa: std.mem.Allocator, finds: anytype, c: *const Config, hl: usize) !?[]const u8 {
    for ([_]usize{ c.dspark_block, max_rows }) |R| {
        var log: tri.Log = .{ .gpa = gpa };
        defer log.deinit();
        try logStageAttn(&log, c, hl, R, true);
        for (log.items.items) |k| if (k.extra and !finds.has(k.name, k.args, k.consts)) return "the kit's extra set has no variant of the four-way stages' sparse attention";
    }
    return null;
}

test "a four-way stage's Triton launches resolve in the kit" {
    // fixtures/kit-drafter.json and kit-drafter-extra.json (tools/dsv41-zig/kit/kit_fixture.py): the kit's base and
    // extra entries of the stage's attention functions. Each launch of a stage at the served arguments (logStageAttn),
    // resolved as the engine resolves it (aot.zig: exact, then general): four ways the part in the extra set alone and
    // every other launch in the base set; TP2's launches (H=32) all in the base set.
    const gpa = std.testing.allocator;
    var base = try cuda.aot.Manifest.parse(gpa, @embedFile("fixtures/kit-drafter.json"));
    defer base.deinit();
    var extra = try cuda.aot.Manifest.parse(gpa, @embedFile("fixtures/kit-drafter-extra.json"));
    defer extra.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var why: @import("config.zig").Why = .{};
    const c = try @import("config.zig").parse(arena.allocator(), @import("config.zig").test_config, &why);
    const keep = tri.decode_rows;
    defer tri.decode_rows = keep;
    tri.decode_rows = 16; // (a pass's: no round's bound in force)
    var kept: [2]usize = .{ 0, 0 }; // four ways: launches from the extra set, from the base set
    for ([_]bool{ true, false }) |four| {
        const s: plan.Split = if (four) .{ .rank = 1, .world = 2, .pair = 1, .parity = true } else .{ .rank = 1, .world = 2 };
        const hl = s.headSpan(c)[1];
        try std.testing.expectEqual(@as(usize, if (four) 16 else 32), hl);
        for ([_]usize{ 5, 10, 15, 20, 40, 80 }) |R| {
            var log: tri.Log = .{ .gpa = gpa };
            defer log.deinit();
            try logStageAttn(&log, &c, hl, R, four);
            try std.testing.expectEqual(@as(usize, if (R <= 16) 5 else 4), log.items.items.len);
            for (log.items.items) |k| {
                const part = std.mem.eql(u8, k.name, "_sparse_attn_part");
                try std.testing.expectEqual(four and part, k.extra);
                if (k.extra) {
                    if (extra.find(k.name, k.args, k.consts) == null) {
                        std.debug.print("four ways, {d} rows: no variant of {s} in the kit's extra set\n", .{ R, k.name });
                        return error.TestUnexpectedResult;
                    }
                    // (the base set has none: the reason the part searches the extra set)
                    try std.testing.expectEqual(@as(?usize, null), base.find(k.name, k.args, k.consts));
                    kept[0] += 1;
                } else {
                    if (base.find(k.name, k.args, k.consts) == null) {
                        std.debug.print("{s}, {d} rows: no variant of {s} in the kit's base set\n", .{ if (four) "four ways" else "TP2", R, k.name });
                        return error.TestUnexpectedResult;
                    }
                    if (four) kept[1] += 1;
                }
            }
        }
    }
    try std.testing.expectEqual([2]usize{ 6, 21 }, kept);
    // the start's check: on with this kit's extra set, off without its variants
    const Finds = struct {
        m: *const cuda.aot.Manifest,
        pub fn has(f: @This(), name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
            return f.m.find(name, args, consts) != null;
        }
    };
    try std.testing.expectEqual(@as(?[]const u8, null), try fourWayKitOff(gpa, Finds{ .m = &extra }, &c, 16));
    try std.testing.expect(try fourWayKitOff(gpa, Finds{ .m = &base }, &c, 16) != null);
}

test "four ways a stage takes the backbone's exchanges and its blocks' parts; a TP2 stage the baseline's" {
    const src = @embedFile("draft.zig");
    const S = struct {
        fn body(text: []const u8, head: []const u8) []const u8 {
            const at = std.mem.indexOf(u8, text, head).?;
            return text[at .. at + std.mem.indexOf(u8, text[at..], "\n    }\n").?];
        }
        fn once(text: []const u8, line: []const u8) !void {
            try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, line));
        }
    };
    // four ways when the blocks hold the node's heads, their widths the backbone's 2D parts
    const init = S.body(src, "    pub fn init(e: *const Engine, a: *prompt.Arena, sp: plan.Split) !Drafter {");
    try S.once(init, "dr.four = if (e.two) |tw| hl == tw.heads else false;");
    try S.once(init, "x.down_n != tw.dw[tw.p] or ir != tw.ir) return error.WeightsNotThisSplit;");
    try S.once(init, "const pack = ex.width + if (ex.rest) |r| r.width else 0;");
    // attention (round.zig's): the rank's four groups from the column partner, then wo_b for the node's columns
    const att = S.body(src, "    fn attention(");
    try S.once(att, "try tw.catRank(e, dr.u, tw.u_full, R, tw.uw, 2);");
    try S.once(att, "try prompt.groupedDecode(e, ch, R, &.{lay.wo_b}, &.{tw.u_full}, &.{tw.uw[0] + tw.uw[1]}, &.{dr.pa}, &.{lay.wo_b.n}, &.{.f32});");
    try S.once(att, "} else try prompt.groupedDecode(e, ch, R, &.{lay.wo_b}, &.{dr.u}, &.{uw}, &.{dr.pa}, &.{lay.wo_b.n}, &.{.f32});");
    try S.once(att, "try tri_attn.sparseAttn(t, dr.sattnArgs(c, lay.sink, pool, j, R, hl));");
    try S.once(S.body(src, "    fn sattnArgs("), ".extra = dr.four,");
    // the MoE (round.zig's): the packs exchanged and assembled (TF_DS_DECODE_FUSE: with down's codes), down over the
    // node's output columns
    const moe = S.body(src, "    fn moe(");
    try S.once(moe, "if (tw.ir != 0 and k.decode_fuse) {");
    try S.once(moe, "try tw.assembleParityCodes(e, k, ex, dr.nxh, dr.pick, rows, dr.mxe);");
    try S.once(moe, "try mx.downDecCodes(k, e.s, ex, dr.mxe, dr.pick, R, dr.pm, tw.dw[tw.p], false);");
    try S.once(moe, "if (tw.ir != 0) try tw.assembleParity(e, ex, dr.nxh, dr.pick, rows) else try tw.catRank(e, dr.nxh, tw.xd_full, rows, tw.gu, 2);");
    try S.once(moe, "try mx.down(k, e.s, ex, dr.mxe, tw.xd_full, ex.down_k, dr.pick, R, dr.pm, tw.dw[tw.p], false, true);");
    try S.once(moe, "} else try mx.down(k, e.s, lay.experts, dr.mxe, dr.nxh, lay.experts.width, dr.pick, R, dr.pm, c.hidden, false, true);");
    // the gathers: four ways the backbone's quarters (the node's columns from every node), else the pair gather
    const sg = S.body(src, "    fn stageGather(");
    try S.once(sg, "return tw.quartersDecode(e, src, dst, R, if (part == .attn) tw.ow else tw.dw, 4);");
    try S.once(sg, "return pairGather(e, src, dst, R, e.c.hidden);");
    const pass = S.body(src, "    fn body(dr: *Drafter");
    try S.once(pass, "try dr.stageGather(e, dr.pa, dr.ga, R, .attn);");
    try S.once(pass, "try dr.stageGather(e, dr.pm, dr.gm, R, .moe);");
}

test "with main_proj four ways an absorb's rows are each node's quarter, gathered as quarters, each plane into its half" {
    const src = @embedFile("draft.zig");
    const S = struct {
        fn body(text: []const u8, head: []const u8) []const u8 {
            const at = std.mem.indexOf(u8, text, head).?;
            return text[at .. at + std.mem.indexOf(u8, text[at..], "\n    }\n").?];
        }
        fn once(text: []const u8, line: []const u8) !void {
            try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, line));
        }
    };
    const init = S.body(src, "    pub fn init(e: *const Engine, a: *prompt.Arena, sp: plan.Split) !Drafter {");
    try S.once(init, "dr.main4 = ds.main_proj.n * 4 == c.hidden and e.two != null;");
    try S.once(init, "if (ds.main_proj.n != c.hidden and !dr.main4) return error.WeightsNotThisSplit;");
    try S.once(init, "dr.aq = try a.take(ring * (d / 4) * 2);");
    try S.once(init, "dr.ag = try a.take(ring * d * 2);");
    const mk = S.body(src, "    fn mainKeys(");
    try S.once(mk, "try prompt.groupedDecode(e, ch, n, &.{dr.ds.main_proj}, &.{taps}, &.{dr.taps_w}, &.{dr.aq}, &.{q}, &.{.bf16});");
    try S.once(mk, "try tw.quartersDecode(e, dr.aq, dr.ag, n, .{ q, q }, 2);");
    try S.once(mk, "for (0..2) |r| try e.ops.copyRows(e.s, dr.ag + r * n * 2 * q * 2, 2 * q * 2, dr.ay + r * 2 * q * 2, d * 2, 2 * q * 2, n);");
    try S.once(mk, "} else try prompt.groupedDecode(e, ch, n, &.{dr.ds.main_proj}, &.{taps}, &.{dr.taps_w}, &.{dr.ay}, &.{d}, &.{.bf16});");
    try S.once(mk, "ddMain(e, dr.ay, n * d * 2);");
    // a host model: each node's quarter (weights.mainProjCols) landed where quarters' ring lands it (ring2d.quartersDsts:
    // plane r = TP2 rank, its pair's columns), each plane's rows copied as mainKeys copies them: the whole rows again
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(arena.allocator(), @import("config.zig").test_config, &why);
    c.format = .native;
    c.hidden = 64; // (a small hidden: 16 columns a quarter)
    const d = c.hidden;
    const q = d / 4;
    const n: usize = 3;
    const keep = weights.draft_main_4way;
    defer weights.draft_main_4way = keep;
    weights.draft_main_4way = true;
    var whole: [n * 64]u16 = undefined;
    for (&whole, 0..) |*x, i| x.* = @intCast(i + 1);
    var gathered: [2 * n * 32]u16 = @splat(0);
    const base: u64 = 0;
    for (0..4) |node| {
        const s: plan.Split = .{ .rank = @intCast(node % 2), .world = 2, .pair = @intCast(node / 2), .parity = true };
        const cols = weights.mainProjCols(c, s).?;
        const dsts = try ring2d.quartersDsts(base, n, 0, .{ q, q }, 2);
        const dst = dsts[node];
        for (0..n) |i| for (0..q) |k| {
            const at = (dst.ptr - base) / 2 + i * (dst.stride4 * 16 / 2) + k;
            gathered[at] = whole[i * d + cols[0] + k];
        };
    }
    var ay: [n * 64]u16 = @splat(0);
    for (0..2) |r| for (0..n) |i| for (0..2 * q) |k| {
        ay[i * d + r * 2 * q + k] = gathered[r * n * 2 * q + i * 2 * q + k];
    };
    try std.testing.expectEqualSlices(u16, &whole, &ay);
}

/// error.NotPortedYet with its site on stderr (a request that reaches a path the port lacks names it in the log).
fn notPorted(src: std.builtin.SourceLocation) error{NotPortedYet} {
    std.debug.print("{{\"not_ported\": \"{s}:{d}\"}}\n", .{ src.file, src.line });
    return error.NotPortedYet;
}

test "TF_DS_DRAFT_VOCAB's checks pass on every node of the four-node split as Model.open builds its engine, and refuse TP2's" {
    const config = @import("config.zig");
    const prompt2d = @import("prompt2d.zig");
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, config.test_config, &why);
    // Model.open on four nodes: the split prompt2d.split(g, 4) (TP2 inside a pair: world 2), the engine's `world` the
    // split's, its comm the four nodes' (m.comm), `two` set; each node's head rows the split's head columns
    // (weights.load: headCols), the four-way drafter's FP4 head those rows (TF_DS_DRAFT_HEAD=fp4)
    const model = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, model, "m.eng = .{ .d = d,") orelse return error.TestUnexpectedResult;
    const eng = model[at .. at + std.mem.indexOfScalar(u8, model[at..], '\n').?];
    for ([_][]const u8{ ".comm = &m.comm,", ".world = sp.world,", ".two = if (m.two) |*t| t else null };" }) |pin| {
        if (std.mem.indexOf(u8, eng, pin) == null) {
            std.debug.print("Model.open's engine lacks: {s}\n", .{pin});
            return error.TestUnexpectedResult;
        }
    }
    try std.testing.expect(std.mem.indexOf(u8, model, "dr.vocabPrepare(&m.eng, &m.arena, quarters, m.comm.rank, &staging)") != null);
    var quarters: [draft_vocab.nodes][2]usize = undefined;
    for (0..draft_vocab.nodes) |g| quarters[g] = (try prompt2d.split(@intCast(g), 4)).headCols(c);
    try std.testing.expectEqual([2]usize{ 64_640, 97_024 }, quarters[1]);
    var two: prompt2d.Two = undefined;
    for (0..draft_vocab.nodes) |g| {
        const sp = try prompt2d.split(@intCast(g), 4);
        try std.testing.expectEqual(@as(u32, 2), sp.world);
        var comm: comm_mod.Comm = undefined;
        comm.world = 4;
        comm.rank = @intCast(g);
        var w: weights.Weights = undefined;
        w.vocab_lo = quarters[g][0];
        w.vocab_hi = quarters[g][1];
        var e: Engine = undefined;
        e.c = &c;
        e.w = &w;
        e.comm = &comm;
        e.world = sp.world;
        e.two = &two;
        e.t = .{};
        var dr: Drafter = undefined;
        dr.ds.head8 = .{ .words = 0, .k = @intCast(c.hidden), .n = @intCast(quarters[g][1] - quarters[g][0]), .fmt = .fp4 };
        try std.testing.expectEqual(@as(?[]const u8, null), try dr.vocabWhy(&e, quarters, @intCast(g)));
        // the start line each node then logs: its list's rows from the other nodes' quarters
        const from = (try draft_vocab.plan(g, quarters, c.vocab)).fromOthers(g);
        try std.testing.expectEqual(([_]usize{ 0, 8192, 806, 4637 })[g], from);
        var lb: [512]u8 = undefined;
        var want: [512]u8 = undefined;
        var db: [8]u8 = undefined;
        try std.testing.expectEqualStrings(try std.fmt.bufPrint(&want, "TF_DS_DRAFT_VOCAB=1: the drafter's head and Markov loop on 8192 ids a node (32768 of 129280, list 82fa3034), on every rank; {d} of this node's head rows from the other nodes; the whole head kept: a pass takes it while more than 2% of the last 4096 committed tokens are outside the list (back under 1%)", .{from}), draft_vocab.line(&lb, 0, null, "", from, c.vocab, draft_vocab.digest(&db)));
        // the drafter's FP8 head, and a node's head rows other than its quarter
        dr.ds.head8.?.fmt = .fp8;
        try std.testing.expectEqualStrings("needs TF_DS_DRAFT_HEAD=fp4", (try dr.vocabWhy(&e, quarters, @intCast(g))).?);
        dr.ds.head8.?.fmt = .fp4;
        w.vocab_lo += 128;
        try std.testing.expectEqualStrings("its head rows are not its vocabulary quarter", (try dr.vocabWhy(&e, quarters, @intCast(g))).?);
        w.vocab_lo -= 128;
        // TP2 (two nodes, no 2D split): its engine's comm is the pair's
        var tp2 = e;
        tp2.two = null;
        var pair = comm;
        pair.world = 2;
        tp2.comm = &pair;
        try std.testing.expectEqualStrings("needs the four-node split", (try dr.vocabWhy(&tp2, quarters, @intCast(g))).?);
        // four ranks without the 2D split, and a rank past the four nodes
        var plain = e;
        plain.two = null;
        try std.testing.expectEqualStrings("needs the four-node split", (try dr.vocabWhy(&plain, quarters, @intCast(g))).?);
        try std.testing.expectEqualStrings("needs the four-node split", (try dr.vocabWhy(&e, quarters, draft_vocab.nodes)).?);
    }
}

test "TF_DS_DRAFT_VOCAB's top ids: the list's head, built by the start's exchange, ranks the kept ids as the whole head does (top 1 and top 5, ties to the lower id), and a misplaced row is caught" {
    // The head's columns are independent dot products (the decode kernel's K split ignores N: replay section draft_vocab
    // proves a list column's fp32 logit byte for byte the whole head's), so what is left to prove on the host is the
    // PLACEMENT: that the bytes the exchange puts at each list position are the row of that position's id. Each node's
    // list head is built here by the engine's own functions on host memory: the senders' blobs (vocabRows: each
    // receiver's gather indices into the sender's quarter; draft_vocab.blobPart: where its parts sit), the rounds' chunks
    // (draft_vocab.landings: what each receiver copies from each sender's chunk, and where), its own rows (vocabLand).
    // The reference reads the whole table by id, never through the placement.
    const gpa = std.testing.allocator;
    const quarters = [draft_vocab.nodes][2]usize{ .{ 0, 32_384 }, .{ 64_640, 97_024 }, .{ 32_384, 64_640 }, .{ 97_024, 129_280 } };
    const V: usize = 129_280;
    const K: usize = 32;
    const cb = K / 2; // an FP4 row's codes
    const sb = K / 32; // its scales: one
    const n = draft_vocab.per_node;
    const codes = try gpa.alloc(u8, V * cb); // the whole head by id (a node holds its quarter's rows of it)
    defer gpa.free(codes);
    const scales = try gpa.alloc(u8, V * sb);
    defer gpa.free(scales);
    var prng = std.Random.DefaultPrng.init(0xd4af7);
    const rng = prng.random();
    rng.bytes(codes);
    rng.bytes(scales);
    // equal logits whatever the activation: three ids' rows are copies of a fourth's, on different nodes, in the list
    const tie_ids = [_]u32{ draft_vocab.id(100), draft_vocab.id(9_000), draft_vocab.id(17_000), draft_vocab.id(30_000) };
    for (tie_ids[1..]) |v| {
        @memcpy(codes[v * cb ..][0..cb], codes[tie_ids[0] * cb ..][0..cb]);
        scales[v] = scales[tie_ids[0]];
    }
    const nr = try draft_vocab.rounds(quarters, cb, sb);
    // each node's export blob, zeros past its end
    const blobs = try gpa.alloc(u8, draft_vocab.nodes * nr * draft_vocab.chunk);
    defer gpa.free(blobs);
    @memset(blobs, 0);
    const ix = try gpa.alloc(i64, draft_vocab.nodes * n);
    defer gpa.free(ix);
    for (0..draft_vocab.nodes) |j| {
        const pj = try draft_vocab.plan(j, quarters, V);
        vocabRows(pj, quarters[j], ix);
        const blob = blobs[j * nr * draft_vocab.chunk ..];
        for (0..draft_vocab.nodes) |m| if (m != j) {
            const part = try draft_vocab.blobPart(quarters, j, m, cb, sb);
            for (0..part.rows) |i| { // vocabPack: the rows of this quarter the receiver takes
                const row = quarters[j][0] + @as(usize, @intCast(ix[m * n + i]));
                @memcpy(blob[part.codes + i * cb ..][0..cb], codes[row * cb ..][0..cb]);
                @memcpy(blob[part.scales + i * sb ..][0..sb], scales[row * sb ..][0..sb]);
            }
        };
    }
    // each node's list head: its own rows, then every round's landings
    const hw = try gpa.alloc(u8, draft_vocab.nodes * n * cb);
    defer gpa.free(hw);
    const hs = try gpa.alloc(u8, draft_vocab.nodes * n * sb);
    defer gpa.free(hs);
    @memset(hw, 0xee);
    @memset(hs, 0xee);
    for (0..draft_vocab.nodes) |me| {
        const pl = try draft_vocab.plan(me, quarters, V);
        vocabRows(pl, quarters[me], ix);
        const lin: weights.Linear = .{ .words = 0, .w = 0, .s = 0, .k = K, .n = n, .fmt = .fp4 };
        if (vocabLand(pl, me, me, lin)) |own| {
            for (0..pl.send[me][1] - pl.send[me][0]) |i| {
                const row = quarters[me][0] + @as(usize, @intCast(ix[me * n + i]));
                @memcpy(hw[me * n * cb + own[0].addr + i * cb ..][0..cb], codes[row * cb ..][0..cb]);
                @memcpy(hs[me * n * sb + own[1].addr + i * sb ..][0..sb], scales[row * sb ..][0..sb]);
            }
        }
        for (0..nr) |r| for (0..draft_vocab.nodes) |j| for (try draft_vocab.landings(quarters, me, j, r, cb, sb)) |cp| if (cp) |x| {
            const chunk_j = blobs[(j * nr + r) * draft_vocab.chunk ..][0..draft_vocab.chunk];
            const dst = if (x.codes) hw[me * n * cb + x.dst ..] else hs[me * n * sb + x.dst ..];
            @memcpy(dst[0..x.len], chunk_j[x.src..][0..x.len]);
        };
    }
    // the head the nodes built holds, at every list position, the bytes of that position's id
    for (0..draft_vocab.count) |p| {
        const v = draft_vocab.id(p);
        try std.testing.expectEqualSlices(u8, codes[v * cb ..][0..cb], hw[p * cb ..][0..cb]);
        try std.testing.expectEqualSlices(u8, scales[v * sb ..][0..sb], hs[p * sb ..][0..sb]);
    }

    // the logits: a row's codes (two FP4-like nibbles a byte) scaled, dotted with an activation: the same function on
    // the table's row and on the list head's row
    const Dot = struct {
        fn of(c: []const u8, s: u8, x: *const [K]f32) f32 {
            var acc: f32 = 0;
            for (0..K) |k| {
                const nib: u8 = if (k % 2 == 0) c[k / 2] & 15 else c[k / 2] >> 4;
                acc += (@as(f32, @floatFromInt(nib)) - 8) * 0.25 * (1 + @as(f32, @floatFromInt(s)) / 16) * x[k];
            }
            return acc;
        }
    };
    const Hit = struct { v: f32, id: u32 };
    const better = struct {
        fn f(_: void, a: Hit, b: Hit) bool {
            return a.v > b.v or (a.v == b.v and a.id < b.id);
        }
    }.f;
    const top = 5;
    // the control: node 2's own rows landing a row late (row i holds row i - 1's bytes)
    const bad_w = try gpa.dupe(u8, hw);
    defer gpa.free(bad_w);
    const bad_s = try gpa.dupe(u8, hs);
    defer gpa.free(bad_s);
    {
        var i: usize = n - 1;
        while (i > 0) : (i -= 1) {
            @memcpy(bad_w[(2 * n + i) * cb ..][0..cb], hw[(2 * n + i - 1) * cb ..][0..cb]);
            bad_s[(2 * n + i) * sb] = hs[(2 * n + i - 1) * sb];
        }
    }
    var control_caught = false;
    for (0..40) |trial| {
        var x: [K]f32 = undefined;
        for (&x) |*e| e.* = rng.float(f32) * 2 - 1;
        var ref: [top]Hit = @splat(.{ .v = -std.math.inf(f32), .id = std.math.maxInt(u32) });
        var got = ref;
        var bad = ref;
        for (0..draft_vocab.count) |p| {
            const v = draft_vocab.id(p);
            insertTop(&ref, .{ .v = Dot.of(codes[v * cb ..][0..cb], scales[v], &x), .id = v }, better);
            insertTop(&got, .{ .v = Dot.of(hw[p * cb ..][0..cb], hs[p], &x), .id = v }, better);
            insertTop(&bad, .{ .v = Dot.of(bad_w[p * cb ..][0..cb], bad_s[p], &x), .id = v }, better);
        }
        if (trial % 8 == 0) { // the tied rows rank first: ties to the lower id on both
            for (&x, 0..) |*e, k| e.* = if (k % 2 == 0) 1 else -1;
            var r2: [top]Hit = @splat(.{ .v = -std.math.inf(f32), .id = std.math.maxInt(u32) });
            var g2 = r2;
            for (0..draft_vocab.count) |p| {
                const v = draft_vocab.id(p);
                insertTop(&r2, .{ .v = Dot.of(codes[v * cb ..][0..cb], scales[v], &x), .id = v }, better);
                insertTop(&g2, .{ .v = Dot.of(hw[p * cb ..][0..cb], hs[p], &x), .id = v }, better);
            }
            for (r2, g2) |a, b| try std.testing.expectEqual(a.id, b.id);
        }
        for (ref, got) |a, b| {
            try std.testing.expectEqual(a.id, b.id);
            try std.testing.expectEqual(a.v, b.v);
        }
        for (ref, bad) |a, b| if (a.id != b.id) {
            control_caught = true;
        };
    }
    // the instrument sees a misplaced row: node 2's rows a row late change some trial's top 5
    try std.testing.expect(control_caught);

    // an id outside the list that the whole head ranks first is the list's miss: the list's top 1 is then the best kept
    // id, not the whole head's (the guard counts the miss)
    const out_id: u32 = 128_000;
    try std.testing.expect(draft_vocab.find(out_id) == null);
    @memset(codes[out_id * cb ..][0..cb], 0xff);
    scales[out_id] = 255;
    var x: [K]f32 = @splat(1);
    var best_all: Hit = .{ .v = -std.math.inf(f32), .id = 0 };
    var best_kept = best_all;
    for (0..V) |v| {
        const h: Hit = .{ .v = Dot.of(codes[v * cb ..][0..cb], scales[v], &x), .id = @intCast(v) };
        if (better({}, h, best_all)) best_all = h;
        if (draft_vocab.find(@intCast(v)) != null and better({}, h, best_kept)) best_kept = h;
    }
    try std.testing.expectEqual(out_id, best_all.id);
    try std.testing.expect(best_kept.id != out_id);
}

fn insertTop(list: anytype, h: @typeInfo(@TypeOf(list.*)).array.child, less: anytype) void {
    var i: usize = list.len;
    while (i > 0 and less({}, h, list[i - 1])) : (i -= 1) {}
    if (i >= list.len) return;
    var j: usize = list.len - 1;
    while (j > i) : (j -= 1) list[j] = list[j - 1];
    list[i] = h;
}

test "TF_DS_DRAFT_VOCAB: a pass takes the list's head and Markov loop unless the guard asks for the whole head, its graphs apart" {
    var dr: Drafter = undefined;
    dr.vocab = null;
    dr.vocab_full = false;
    try std.testing.expect(!dr.onList());
    dr.vocab = std.mem.zeroes(Vocab);
    try std.testing.expect(dr.onList());
    dr.vocab_full = true;
    try std.testing.expect(!dr.onList());
    // the graph of a pass on the list is not the whole head's, nor the baseline key
    const k: PassGraphs.Key = .{ .streams = 4, .steps = 5 };
    try std.testing.expect(!std.meta.eql(k, PassGraphs.Key{ .streams = 4, .steps = 5, .list = true }));
    try std.testing.expect(std.meta.eql(k, PassGraphs.Key{ .streams = 4, .steps = 5, .list = false }));
    const src = @embedFile("draft.zig");
    const code = src[0..std.mem.indexOf(u8, src, "\ntest \"").?];
    // the body: the head and the loop by onList, the list's loop mapping its picks
    const at = std.mem.indexOf(u8, code, "    fn body(dr: *Drafter").?;
    const body = code[at .. at + std.mem.indexOf(u8, code[at..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const list: ?*const Vocab = if (dr.onList()) &dr.vocab.? else null;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const head = if (list) |v| v.head else dr.ds.head8 orelse e.w.head;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "try draft_vocab.steps(v.mk, t, dr.local, dr.out, ts, N, n, steps, dr.msc, .{ .ctx = &g, .run = MarkovGather.run }, .{ .ctx = &ids, .run = ListIds.run });"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "} else try dr.mk.steps(t, dr.local, dr.out, ts, N, n, steps, dr.msc, .{ .ctx = &g, .run = MarkovGather.run });"));
    // the pass graph's key carries it
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "const key: Key = .{ .streams = @intCast(N), .steps = @intCast(steps), .list = dr.onList() };"));
}

test "TF_DS_DRAFT_VOCAB's start: this rank's checks in order, its own rows and Markov rows before the agreement, the others' rows a chunk a round over the host links after it" {
    const src = @embedFile("draft.zig");
    const code = src[0..std.mem.indexOf(u8, src, "\ntest \"").?];
    const S = struct {
        fn body(text: []const u8, head: []const u8) []const u8 {
            const at = std.mem.indexOf(u8, text, head).?;
            return text[at .. at + std.mem.indexOf(u8, text[at..], "\n    }\n").?];
        }
        fn order(text: []const u8, lines: []const []const u8) !void {
            var last: usize = 0;
            for (lines) |l| {
                const at = std.mem.indexOf(u8, text, l) orelse {
                    std.debug.print("missing: {s}\n", .{l});
                    return error.TestUnexpectedResult;
                };
                if (at < last) {
                    std.debug.print("out of order: {s}\n", .{l});
                    return error.TestUnexpectedResult;
                }
                last = at;
            }
        }
    };
    const checks = S.body(code, "    pub fn vocabWhy(");
    try S.order(checks, &.{
        "if (draft_vocab.check(c.vocab)) |why| return why;",
        "return \"needs the four-node split\";",
        "const h8 = dr.ds.head8 orelse return \"needs TF_DS_DRAFT_HEAD=fp4\";",
        "if (h8.fmt != .fp4) return \"needs TF_DS_DRAFT_HEAD=fp4\";",
        "if (q[0] != e.w.vocab_lo or h8.n != q[1] - q[0]) return \"its head rows are not its vocabulary quarter\";",
        "_ = draft_vocab.plan(me, quarters, c.vocab) catch return \"the nodes' head rows do not tile the vocabulary\";",
        "draft_vocab.kitOff(std.heap.page_allocator, SetFinds{ .set = set }, c.*)",
        "return null;",
    });
    const prep = S.body(code, "    pub fn vocabPrepare(");
    try S.order(prep, &.{
        "dr.vocab = null;",
        "\n        if (try dr.vocabWhy(e, quarters, me)) |why| return why;\n",
        "const pl = try draft_vocab.plan(me, quarters, c.vocab);",
        "try e.ops.gatherRows(e.s, dr.mk.head, c.vocab, v.ids, mhead, rank * 2, draft_vocab.count, v.flag);",
        "vocabRows(pl, q, rows);",
        "if (vocabLand(pl, me, me, v.head)) |own| _ = try vocabPack(e.ops, e.s, h8, v.idx, pl, me, own[0].addr, own[1].addr, v.flag);",
        "v.mk = draft_vocab.markov(c.*, me, mhead, dr.mk.emb, dr.mk.none);",
        "try v.mk.fill(e.t, tok, kt, try a.take(kt * n * 2), dr.mk.slot);",
        "if (bad != 0) return \"a row of the list is past its table\";",
        "dr.vocab = v;",
    });
    const ex = S.body(code, "    pub fn vocabExchange(");
    try S.order(ex, &.{
        "const part = try draft_vocab.blobPart(v.quarters, me, m, cb, sb);",
        "_ = try vocabPack(e.ops, e.s, h8, v.idx, v.plan, m, staging.ptr + part.codes, staging.ptr + part.scales, v.flag);",
        "for (0..v.blob / draft_vocab.chunk) |round| {",
        "try e.comm.gatherHost(e.d, staging.ptr + round * draft_vocab.chunk, recv.ptr, draft_vocab.chunk, e.s);",
        "try draft_vocab.landings(v.quarters, me, j, round, cb, sb)",
        "cuMemcpyDtoDAsync_v2(dst, recv.ptr + j * draft_vocab.chunk + x.src, x.len, e.s.handle)",
        "try e.s.synchronize();",
        "if (bad != 0) return error.RowPastTable;",
    });
}

test "TF_DS_DRAFT_VOCAB's row placement: each node's rows of a quarter as its head's rows, and where a sender's rows land" {
    const quarters = [draft_vocab.nodes][2]usize{ .{ 0, 32_384 }, .{ 64_640, 97_024 }, .{ 32_384, 64_640 }, .{ 97_024, 129_280 } };
    const gpa = std.testing.allocator;
    const n = draft_vocab.per_node;
    const out = try gpa.alloc(i64, draft_vocab.nodes * n);
    defer gpa.free(out);
    const head: weights.Linear = .{ .words = 0, .w = 0x1000_0000, .s = 0x2000_0000, .k = 5120, .n = n, .fmt = .fp4 };
    for (0..draft_vocab.nodes) |me| {
        const pl = try draft_vocab.plan(me, quarters, 129_280);
        vocabRows(pl, quarters[me], out);
        for (pl.send, 0..) |x, j| {
            for (0..n) |i| {
                const want: i64 = if (i < x[1] - x[0]) @as(i64, draft_vocab.id(x[0] + i)) - @as(i64, @intCast(quarters[me][0])) else 0;
                try std.testing.expectEqual(want, out[j * n + i]);
                if (i < x[1] - x[0]) try std.testing.expect(out[j * n + i] >= 0 and out[j * n + i] < quarters[me][1] - quarters[me][0]);
            }
        }
        // the lands tile me's head, in order, codes 2,560 bytes a row and scales 160
        var next: usize = 0;
        var rows: usize = 0;
        var order: [draft_vocab.nodes]usize = .{ 0, 1, 2, 3 };
        std.mem.sort(usize, &order, pl, struct {
            fn lt(p: draft_vocab.Plan, a: usize, b: usize) bool {
                return p.recv[a][0] < p.recv[b][0];
            }
        }.lt);
        for (order) |j| if (vocabLand(pl, me, j, head)) |l| {
            try std.testing.expectEqual(head.w + next * 2560, l[0].addr);
            try std.testing.expectEqual(head.s + next * 160, l[1].addr);
            const r = pl.recv[j][1] - pl.recv[j][0];
            try std.testing.expectEqual(r * 2560, l[0].len);
            try std.testing.expectEqual(r * 160, l[1].len);
            next += r;
            rows += r;
        } else try std.testing.expectEqual(pl.recv[j][0], pl.recv[j][1]);
        try std.testing.expectEqual(n, rows);
    }
}
