//! One TP rank of DeepSeek-V4.1-Flash loaded for serving: the weights, kernels, caches and buffers the layer gate sets
//! up (layer_main.zig), for a pool of draft.max_streams streams (MultiDecoder's slots: each stream's window rings,
//! positional stores and drafter rings in its slot, its compressed rows in its extent of one shared plane), and the
//! primitives a lane round runs on them. Both ranks run the same primitives in the same order (lanes.zig sends them
//! to rank 1): a prompt's chunks into its stream's slot and extent (fill), one round over every stream's window and
//! the windows' taps absorbed (verify), and the drafter's batched pass (pass). Every primitive is the gates' own code
//! path (prompt.zig, round.zig, draft.zig) with no checks.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const rank_cache = @import("rank_cache.zig");

/// The words of the start's refusal when a retired name is set in the environment (null: none is): "TF_DS_ROCI_DECODE no longer exists:
/// Rocinante is the only transport".
pub fn retiredSwitch(buf: []u8) ?[]const u8 {
    return engine_api.settings.firstRetired(ProcessEnv{}, buf);
}

/// The process environment as a lookup (`get(name) ?[]const u8`), what settings.firstRetired reads.
const ProcessEnv = struct {
    pub fn get(_: ProcessEnv, name: []const u8) ?[]const u8 {
        var buf: [128:0]u8 = undefined;
        if (name.len >= buf.len) return null;
        @memcpy(buf[0..name.len], name);
        buf[name.len] = 0;
        const v = std.c.getenv(&buf) orelse return null;
        return std.mem.span(v);
    }
};

const weights = @import("weights.zig");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const round = @import("round.zig");
const sampling = @import("sampling.zig");
const draft = @import("draft.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const mx = @import("mx.zig");
const mx5 = @import("mlx5_devx.zig");
const ops_mod = @import("ops.zig");
const exact = @import("exact.zig");
const cublas = @import("cublas.zig");
const comm_mod = @import("comm.zig");
const link = @import("link.zig");
const ring2d = @import("ring2d.zig");
const roci_engine = @import("roci_engine.zig");
const pair_spans = @import("pair_spans.zig");
const rdma = @import("rdma.zig");
const tri_index = @import("tri_index.zig");

/// TF_DS_ENGRAM_WORKER=0: a fill's Engram rows read by a thread a chunk (the chunk before's end to this one's first
/// Engram layer) instead of the fill's worker.
var wk_on: bool = true;
pub fn readWorkerEnv() void {
    if (std.c.getenv("TF_DS_ENGRAM_WORKER")) |v| wk_on = !std.mem.eql(u8, std.mem.span(v), "0");
}
/// The pool's streams (--parallel 4).
pub const max_streams = draft.max_streams;
/// The pool's streams when unset (--parallel 4).
pub const max_streams_default = draft.default_streams;
/// The RDMA ring's slot (the served 4.25 MiB: a 16-row round's logits half fits).
pub const ring_bytes = 4456448;

pub const Options = struct {
    model_dir: []const u8, // config.json
    cache_dir: []const u8, // the lane's per-rank weight file (TF_DS_RANK_CACHE)
    kit_dir: []const u8, // a gate dir's layout: aot/, cubins/, rope-{plain,compressed}-{cos,sin}.f32, engram.json
    aot_dir: ?[]const u8 = null, // the recorded Triton set (aot.json and its cubins; null: kit_dir/aot)
    rank: u32,
    world: u32,
    master: [4]u8,
    listen: ?[3][4]u8 = null, // rank 0 at world 4: the address to listen on for each follower (TF_TP_LISTEN), else master
    port: u16,
    engram_dir: ?[]const u8 = null, // the Engram tables (none: a checkpoint without Engram layers)
    token_map: ?[]const u8 = null, // the compressed token map (the lane's JSON cache)
    pool: usize = 1 << 18, // positions of the streams' shared plane (their extents together)
    context: usize = 0, // a stream's extent at most (its prompt, budget and a round's rows; 0: the pool)
    drafts: bool = true, // load the DSpark drafter
    rdma_devices: ?[]const u8 = null, // the decode gathers over RDMA on these devices (comma separated): TP2's
    // ring (rdma.zig), or at world 4 the 2D split's decode-size exchanges over its rings (ring2d.zig)
    rdma_kernels: ?[]const u8 = null, // world 4: the rings' kernels' image (rdma_gather.cu's fatbin; null: the embedded one)
    graphs: bool = false, // the rounds' stretches as CUDA graphs (round.Graphs)
    graph_cap: usize = 0, // the captured stretches kept at most, least recently replayed out first (TF_DS_GRAPH_CAP; 0: all)
    side: bool = false, // the mixes' side work on a stream of its own (round.Round.useSide)
    prefetch: bool = false, // the paced L2 prefetch (round.Round.usePrefetch)
    engram_aio: bool = false, // a round's Engram reads by Linux AIO on O_DIRECT (else the reader pool)
    arena_bytes: usize = 0, // the device arena's first block (0: it grows from 256 MiB blocks as the buffers ask)
    round_rows: usize = round.default_rows, // a round's rows at most (TF_DS_ROUND_ROWS; the four-node lane's 48)
    streams: usize = draft.default_streams, // the pool's streams (--parallel; the four-node lane's 16)
    round_ms: ?[]const f64 = null, // the lane core's round costs by rows (TF_DS_ROUND_MS; null: the served ROUND_MS)
    vision_bias: ?[]const u8 = null, // gate_bias_vl.safetensors (null: the kit's vision/): the image-span routing bias
};

/// TF_DS_2D_GU=parity (the served switch): the 2D split's experts with half the gate / up blocks a pair and the ninth by
/// expert parity (plan.Split.parity); the rank cache must be one written for it.
pub fn parityGU() bool {
    const v = std.c.getenv("TF_DS_2D_GU") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "parity");
}

/// The parity split's decode gate / up in one launch (experts_par.cu, the same bits); TF_DS_PAR_ONE=0: the served
/// build's launches (main, then the rest's grouping and gate / up, beside it on a stream of its own with --side).
pub fn parOne() bool {
    const v = std.c.getenv("TF_DS_PAR_ONE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    host: std.heap.ArenaAllocator, // allocations that live as long as the model
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    cfg: Config,
    rank: u32,
    world: u32,
    streams: usize, // the pool's streams (Options.streams)
    round_ms: []const f64, // the lane core's round costs by rows (Options.round_ms, else draft.round_ms)
    pool_cap: usize,
    ctx_cap: usize, // a stream's extent at most (Options.context, else the pool)
    cache_dir: std.Io.Dir,
    cache_file: std.Io.File,
    ix: rank_cache.Index,
    w: weights.Weights,
    fan: prompt2d.Fan,
    comm: comm_mod.Comm,
    rdma_mod: ?cuda.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*rdma.Ring = null,
    set: cuda.aot.Set,
    pf: exl3_prefill.Kernels,
    lg: exl3_linear.Kernels,
    exk: exl3_experts.Kernels,
    ops: ops_mod.Ops,
    ex: exact.Exact,
    par: ?exl3_experts2d.ParKernels, // the parity split's one-launch gate / up (parOne)
    rope_buf: cuda.DeviceBuffer,
    arena: prompt.Arena,
    eng: prompt.Engine,
    ch: prompt.Chunk,
    // TF_DS_WAVES: the first wave's chunk (ch is the second's), the gathers' stream, communicator, 2D buffers (the
    // upper half of the parts) and engine, and each wave's events (pa, ga, pm, gm ready)
    waves: bool,
    ch2: prompt.Chunk,
    cs: ?cuda.Stream,
    comm2: ?comm_mod.Comm,
    two_c: prompt2d.Two,
    ec: prompt.Engine,
    wev: [2][4]cuda.Event,
    caches: prompt.Caches,
    blas: cublas.Blas,
    rings: []u64, // each layer's window rings, max_streams slots
    ring_view: []u64, // a filling stream's slot of them
    rd: round.Round,
    graphs: ?round.Graphs = null,
    pgraphs: ?draft.PassGraphs = null, // the drafter pass as CUDA graphs (with graphs)
    dpool: ?draft.Pool = null,
    dr: ?draft.Drafter = null,
    tables: ?engram_io.Tables = null,
    epool: ?*engram_io.Pool = null,
    aio: ?*engram_aio.Aio = null,
    paio: ?*engram_aio.Aio = null, // the prompt chunks' Engram reads, on a ring of their own
    eh: ?prompt.EngramHost = null,
    host_pos: []i64,
    ids64: []i64,
    vocab: usize,
    logits: []f32, // round.max_rows rows of logits on the host
    amax: u64, // u32 [round.max_rows]: a round's greedy tokens on the device (roundArgmax)
    amax_host: [round.max_rows]u32 = undefined,
    // the stream a fill is running: its slot and its view of the caches (its extent from base)
    fill_slot: usize = 0,
    // the prompt a fill's chunks come from (the caller's, set for the fill) and the thread reading the next chunk's
    // Engram rows during this one (fillChunk), joined before the prompt can change (joinAhead)
    fill_seq: []const i32 = &.{},
    ahead: ?std.Thread = null,
    // the next prompt pass's rows at most (the lanes' pass rule): the Engram read-ahead reads that many, no more
    next_rows: usize = chunk_rows,
    view: prompt.Caches = undefined,
    // the filling prompt's image spans and their rows (setImages; cleared by fillBegin)
    spans: std.ArrayList(prompt.Span) = .empty,
    img_buf: ?cuda.DeviceBuffer = null,
    // profiling (lanes' --profile): the last round's forward and absorb, synchronized apart (ns)
    prof: bool = false,
    timer: ?*round.PhaseTimer = null, // a profile's GPU time by phase (eager rounds)
    ptimer: ?*round.PhaseTimer = null, // the same for prompt chunks (fills), a chunk a 'round'
    t_enqueue: u64 = 0, // the host's time to issue the forward (the GPU idle before it)
    // Model.useClock: the rounds' GPU clock gaps summed (ns; gap i = clock i + 1 - clock i: the forward's start to
    // the first stretch, then each stretch, then the wait before the next), and the rounds counted
    clk_ph: [256]u64 = @splat(0), // by the phase of the clock that ends it
    clk_cnt: [256]u64 = @splat(0),
    clk_rounds: u64 = 0,
    t_forward: u64 = 0,
    t_absorb: u64 = 0,
    // world 4: the 2D split's exchanges (prompt2d.zig, TP4's), with the RDMA rings when asked
    two: ?prompt2d.Two = null,
    rdma_rings: ?ring2d.Rings = null,
    neighbor_gather: ?neighbor.Gather = null,
    // watchPeers: a peer rank that stops stops this one too (link.Watch)
    watch: link.Watch = .{},

    /// Loads rank o.rank's weights from the lane's rank cache, links the ranks (rank 0 listens on o.port), opens NCCL
    /// and sets up every buffer. Needs `ctx` current on this thread.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: Options) !*Model {
        if (o.world != 2 and o.world != 4) return error.NotPortedYet; // TP2, or the four-node 2D split (prompt2d.zig)
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        if (std.c.getenv("TF_DS_REPLAY_FLOOR")) |v| prompt.replay_floor = std.mem.eql(u8, std.mem.span(v), "1"); // every rank alike
        try prompt.splitEnv(); // TF_DS_IDX_SPLIT: every rank alike too
        if (std.c.getenv("TF_DS_2D_YX")) |v| prompt2d.y_sums = std.mem.eql(u8, std.mem.span(v), "1"); // every rank alike
        if (std.c.getenv("TF_DS_TWO_HOP")) |v| comm_mod.two_hop = std.mem.eql(u8, std.mem.span(v), "1"); // a ring of four: every rank alike
        m.spans = .empty;
        m.img_buf = null;
        m.host = std.heap.ArenaAllocator.init(gpa);
        errdefer m.host.deinit();
        const a = m.host.allocator();
        m.io = io;
        m.ctx = ctx;
        m.rank = o.rank;
        m.world = o.world;
        if (o.streams == 0 or o.streams > max_streams) return error.BadStreams;
        m.streams = o.streams;
        if (std.c.getenv("TF_DS_L2_SKIP")) |v| round.l2_skip = std.fmt.parseInt(u8, std.mem.span(v), 10) catch 0;
        for ([_][:0]const u8{ "TF_DS_L2_ATTN_MB", "TF_DS_L2_MOE_MB", "TF_DS_L2_NEXT_MB", "TF_DS_L2_STRETCH_END_MB" }, 0..) |name, i| {
            if (std.c.getenv(name)) |v| round.l2_mb[i] = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_mb[i];
        }
        if (std.c.getenv("TF_EXL3_L2_DELAY_NS")) |v| ops_mod.l2_delay_ns = std.fmt.parseInt(c_longlong, std.mem.span(v), 10) catch ops_mod.l2_delay_ns;
        if (std.c.getenv("TF_EXL3_L2_RATE_GBPS")) |v| ops_mod.l2_rate_gbps = std.fmt.parseFloat(f64, std.mem.span(v)) catch ops_mod.l2_rate_gbps;
        m.round_ms = o.round_ms orelse &draft.round_ms;
        if (m.round_ms.len == 0) return error.BadRoundMs;
        m.pool_cap = o.pool;
        m.ctx_cap = if (o.context != 0) @min(o.context, o.pool) else o.pool;
        m.fill_slot = 0;
        m.prof = false;
        m.timer = null;
        m.ptimer = null;
        m.t_enqueue = 0;
        m.t_forward = 0;
        m.t_absorb = 0;
        m.dpool = null;
        m.dr = null;
        m.tables = null;
        m.epool = null;
        m.aio = null;
        m.paio = null;
        m.eh = null;
        m.rdma_mod = null;
        m.ring = null;
        m.graphs = null;
        m.pgraphs = null;
        m.two = null;
        m.waves = wavesOn();
        prompt.waves = m.waves;
        m.cs = null;
        m.comm2 = null;
        m.rdma_rings = null;
        m.neighbor_gather = null;
        m.watch = .{};
        const d = ctx.d;
        m.cfg = try Config.read(a, io, o.model_dir);
        const c = &m.cfg;
        m.stream = try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();

    /// The base set alone: the extra set's launches (the two-block readers) are not looked up by the checks that take this lookup.
    pub fn hasExtra(f: SetFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        _ = f;
        _ = name;
        _ = args;
        _ = consts;
        return true;
    }
};

/// SetFinds with the kit's extra set as well: a launch of the extra set (the two-block readers, runExtra) is looked up there, as it runs;
/// without an extra set it has no variant.
const KitFinds = struct {
    set: *const cuda.aot.Set,
    extra: ?*const cuda.aot.Set,

    pub fn has(f: KitFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        _ = f.set.find(name, args, consts) catch return false;
        return true;
    }

    pub fn hasExtra(f: KitFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        const x = f.extra orelse return false;
        return x.lookup(name, args, consts) != null;
    }

        // the RoPE tables (the served build's bits: torch CPU pow and polar), every position a stream reaches
        const half = c.rope_dim / 2;
        const rope_rows = m.ctx_cap + round.max_rows + 1;
        m.rope_buf = try cuda.DeviceBuffer.alloc(d, 4 * rope_rows * half * 4);
        errdefer m.rope_buf.free();
        {
            const part = try gpa.alloc(u8, rope_rows * half * 4);
            defer gpa.free(part);
            for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
                var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ o.kit_dir, name }), .{});
                defer f.close(io);
                if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable; // a table of fewer rows than a stream's context
                try m.rope_buf.upload(j * part.len, part);
            }
        }
        const tbl = rope_rows * half * 4;
        const rope: prompt.Rope = .{ .cos = m.rope_buf.ptr, .sin = m.rope_buf.ptr + tbl };
        const rope_c: prompt.Rope = .{ .cos = m.rope_buf.ptr + 2 * tbl, .sin = m.rope_buf.ptr + 3 * tbl };

        m.arena = try prompt.Arena.init(d, o.arena_bytes);
        errdefer m.arena.deinit();
        // world 4 (layer_main's setup): the 2D exchanges' buffers, and with rdma_devices their rings (whose infos go
        // round over NCCL world 4)
        errdefer if (m.neighbor_gather) |*ng| ng.close(gpa);
        errdefer if (m.rdma_rings) |*x| x.close(gpa);
        if (sp.pair != null) {
            m.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, chunk_rows);
            if (neighbor_rdma) {
                m.rdma_mod = try cuda.Module.load(d, cuda.kernels.dsv41_rdma);
                m.neighbor_gather = try neighbor.Gather.open(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), o.rank, &m.comm, m.stream, neighbor_gid);
                m.two.?.neighbor_gather = &m.neighbor_gather.?;
            }
            if (o.rdma_devices) |devs| {
                const image: []const u8 = if (o.rdma_kernels) |path|
                    try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 26), .@"16", null)
                else if (cuda.kernels.available) cuda.kernels.dsv41_rdma else return error.NoRdmaKernels;
                m.rdma_mod = try cuda.Module.load(d, image);
                var devices: std.ArrayList([]const u8) = .empty;
                var dit = std.mem.splitScalar(u8, devs, ',');
                while (dit.next()) |x| try devices.append(a, x);
                m.rdma_rings = try ring2d.Rings.open(gpa, io, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, .{ .max_bytes = ring_bytes, .gid_index = rdma.rdmaGid() }, &m.comm, m.stream);
                m.two.?.rings = &m.rdma_rings.?;
            }
        }
        m.eng = .{ .d = d, .s = m.stream, .t = .{ .set = &m.set, .stream = m.stream }, .blas = undefined, .comm = &m.comm, .pf = &m.pf, .lin = &m.lg, .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = c, .w = &m.w, .world = sp.world, .plain = rope, .compressed = rope_c, .ring = m.ring, .two = if (m.two) |*t| t else null };
        m.eng.round_rows = o.round_rows;
        if (m.par) |*x| m.eng.par = x;
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, if (m.waves) wave_rows else chunk_rows, m.ctx_cap); // a stream's keys at most
        if (m.waves) m.ch2 = try prompt.Chunk.init(&m.eng, &m.arena, wave_rows, m.ctx_cap);
        m.caches = try prompt.Caches.init(&m.eng, &m.arena, o.pool, m.streams);
        m.blas = try cublas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer m.blas.close();
        m.eng.blas = &m.blas;
        if (m.waves) {
            if (2 * wave_rows != chunk_rows) return error.WavesNeedTwoWaveRowsAPass;
            const t = if (m.two) |*x| x else return error.WavesNeedThe2DSplit;
            if (t.ow[0] != t.ow[1]) return error.WavesNeedEqualWidths; // the gathers on cs never take t.pad
            m.cs = try cuda.Stream.init(d, true);
            // the compute stream's exchanges stay in the parts' lower half (a wave's rows at most), cs's in the upper
            m.two_c = t.*;
            m.two_c.part_bytes = t.part_bytes / 2;
            m.two_c.parts = t.parts + 2 * t.part_bytes;
            m.two_c.rings = null;
            // the bf16-sum gathers (TF_DS_2D_YX) exchange through a pad of cs's own
            m.two_c.pad = try m.arena.take(m.two_c.part_bytes);
            m.ec = m.eng;
            m.ec.s = m.cs.?;
            m.ec.t.stream = m.cs.?;
            m.ec.comm = &m.comm2.?;
            m.ec.two = &m.two_c;
            m.ec.ring = null;
            for (&m.wev) |*w| for (w) |*ev| {
                ev.* = try cuda.Event.init(d, false);
            };
        }
        m.rings = try a.alloc(u64, c.layers);
        m.ring_view = try a.alloc(u64, c.layers);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.streams * m.eng.ringBytes());
            try d.check(d.api.cuMemsetD8Async(r.*, 0, m.streams * m.eng.ringBytes(), m.stream.handle), "cuMemsetD8Async");
        }
        m.rd = try round.Round.init(&m.eng, &m.arena, a, m.ctx_cap); // a round's deepest bucket: a stream's context
        m.amax = try m.arena.take(round.max_rows * 4);
        if (o.graphs) m.graphs = round.Graphs.init(gpa);
        if (o.side) try m.rd.useSide(&m.eng);
        if (o.prefetch) try m.rd.usePrefetch(&m.eng);
        if (m.two) |t| if (t.ir != 0 and o.side and m.par == null) try m.rd.usePar(&m.eng); // parity in two launches: the rest beside the main gate / up
        if (o.drafts) {
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, m.streams);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);
            if (o.graphs) {
                m.pgraphs = draft.PassGraphs.init(gpa);
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        try m.synchronize();
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize"); // the setup's legacy-stream copies too
    if (!is_default or text == null or !std.mem.eql(u8, text.?, "4096")) return null;

        // Engram on the host: the compressed token map, the multipliers, the tables and 64 readers
        if (o.engram_dir) |edir| {
            const map_text = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
            const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
            const EngramJson = struct { multipliers: [][]i64 };
            const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit_dir, "engram.json" }), a, .limited(1 << 26));
            const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
            var mult: [engram.max_layers][engram.max_ngram]i64 = @splat(@splat(0));
            for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
                mult[l][k] = v;
            };
            m.tables = try engram_io.Tables.open(gpa, io, edir);
            m.epool = try engram_io.Pool.init(gpa, io, 64);
            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, chunk_rows);
            if (o.engram_aio) {
                // a round's reads, 2 x 12 x its rows at most: 1,024 slots for each 16 rows of the round-row setting
                m.aio = try engram_aio.Aio.init(gpa, io, 1024 * ((o.round_rows + 15) / 16));
                m.eh.?.aio = m.aio;
                // a prompt chunk's reads (2 x 12 x 2,048 at most): a ring of 4,096 slots (32 MiB), 1,024 rows a batch
                m.paio = try engram_aio.Aio.init(gpa, io, 4096);
                m.eh.?.paio = m.paio;
            }
        } else if (c.engram_layers.slice().len > 0) return error.NoEngramTables;

        // a vision checkpoint's image routing bias (without it a prompt with images is refused; text runs as ever)
        if (c.vision) {
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            m.loadVisionBias(path) catch |err| if (o.vision_bias != null or err != error.FileNotFound) return err;
        }

        m.host_pos = try a.alloc(i64, chunk_rows);
        m.ids64 = try a.alloc(i64, chunk_rows);
        m.vocab = sp.world * (if (m.two) |t| t.hw[0] + t.hw[1] else m.w.head.n); // 2D: both pairs' vocabulary parts
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        return m;
    }

    pub fn close(m: *Model) void {
        m.joinAhead();
        m.stream.synchronize() catch {};
        if (m.img_buf) |*b| b.free();
        m.spans.deinit(m.gpa);
        if (m.graphs) |*g| g.deinit();
        if (m.pgraphs) |*g| g.deinit();
        if (m.timer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        if (m.ptimer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        m.rd.dropSide(&m.eng);
        m.rd.dropPrefetch();
        m.rd.dropPar();
        if (m.aio) |x| x.deinit();
        if (m.paio) |x| x.deinit();
        if (m.epool) |p| p.deinit(m.gpa);
        if (m.tables) |*t| t.close();
        if (m.neighbor_gather) |*ng| ng.close(m.gpa);
        if (m.rdma_rings) |*r| r.close(m.gpa);
        m.blas.close();
        m.arena.deinit();
        m.rope_buf.free();
        if (m.par) |*x| x.unload();
        m.ex.unload();
        m.ops.unload();
        m.exk.unload();
        m.lg.unload();
        m.pf.unload();
        m.set.deinit();
        if (m.ring) |r| {
            r.stop();
            r.destroy(m.gpa);
        }
        if (m.rdma_mod) |*x| x.unload();
        if (m.waves) {
            for (&m.wev) |*w| for (w) |*ev| ev.deinit();
            if (m.cs) |*x| x.deinit();
        }
        if (m.comm2) |*x| x.deinit();
        m.comm.deinit();
        m.watch.stop(); // (still watching up to here: a wait above on a rank that went ends this process)
        m.fan.close();
        m.w.deinit();
        m.cache_file.close(m.io);
        m.ix.deinit();
        m.cache_dir.close(m.io);
        m.stream.deinit();
        m.host.deinit();
        m.gpa.destroy(m);
    }

    fn synchronize(m: *Model) !void {
        try m.stream.synchronize();
        if (m.neighbor_gather) |*ng| try ng.check();
    }

    /// TP2's RDMA ring for the decode gathers: created on both ranks, the queue-pair infos exchanged over NCCL,
    /// connected, and started once both are (an all-gather as the barrier).
    pub fn openRing(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const comm_mod.Comm, stream: cuda.Stream) !*rdma.Ring {
        if (comm.world != 2) return error.NotTwoRanks;
        const r = try rdma.Ring.create(gpa, d, k, devices, rank, 2, .{ .max_bytes = ring_bytes, .gid_index = rdma.rdmaGid() });
        errdefer r.destroy(gpa);
        const per = @sizeOf(rdma.Info);
        const mine = [2]rdma.Info{ r.info(0), r.info(1) };
        var dev = try cuda.DeviceBuffer.alloc(d, 3 * 2 * per);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 2 * per, .u8, stream);
        try stream.synchronize();
        var all: [2][2]rdma.Info = undefined;
        try dev.download(2 * per, std.mem.sliceAsBytes(&all));
        try r.connect(&.{ all[0][rank], all[1][rank] });
        // every queue pair ready on both ranks before either sends
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 1, .u8, stream);
        try stream.synchronize();
        try r.start();
        return r;
    }

    fn readKit(a: std.mem.Allocator, io: std.Io, kit: []const u8, name: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ kit, "cubins", name }), a, .limited(1 << 28), .@"16", null);
    }

    /// The link to the other rank (rank 0's to rank 1, rank 1's to rank 0).
    pub fn peer(m: *const Model) link.Link {
        return m.fan.links[0].?;
    }

    /// Rank 0's links to its followers: rank 1's, or ranks 1-3's on the four-node split.
    pub fn followers(m: *const Model) []const ?link.Link {
        return m.fan.links[0 .. m.world - 1];
    }

    /// From here on a peer rank that stops stops this one too, with a line that names it (link.Watch): rank 0 watches
    /// its followers, a follower rank 0. The server's rank 0 and the lane followers watch; rank 0's done (Lanes.deinit),
    /// a follower reading it (follow) and close end the watch.
    pub fn watchPeers(m: *Model) !void {
        if (m.world < 2 or m.watch.thread != null) return;
        m.watch = .{ .rank = m.rank };
        if (m.rank == 0) {
            for (m.followers(), 1..) |l, r| if (l) |x| try m.watch.add(x, @intCast(r));
        } else try m.watch.add(m.peer(), 0);
        try m.watch.start();
    }

    /// Ends watchPeers' watch: this process ends its links itself now.
    pub fn unwatch(m: *Model) void {
        m.watch.stop();
    }

    /// Whether this rank's GPU context has failed for good: after a fault in a kernel (an illegal address, a trap)
    /// every later call returns that fault, the stream's query too (which logs it); a context that survived a failed
    /// call answers done or busy.
    pub fn contextFailed(m: *Model) bool {
        _ = m.stream.done() catch return true;
        return false;
    }

    pub fn drafting(m: *const Model) bool {
        return m.dr != null;
    }

    // -- the primitives (both ranks, in the same order) ----------------------------------------------------------

    /// A stream's prompt is about to fill pool slot `slot`, its compressed rows in its extent from position `base`.
    /// Wait for the next chunk's Engram read-ahead, if one runs (before the prompt it reads changes or goes).
    pub fn joinAhead(m: *Model) void {
        if (m.ahead) |th| th.join();
        m.ahead = null;
    }

    /// A kept prompt state's own bytes (the prompt cache's snapshots; the compressed plane stays in the stream's
    /// extent): one slot's window ring of every layer and positional stores of every ratio-2 layer.
    pub fn snapBytes(m: *const Model) usize {
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var n: usize = m.rings.len * m.eng.ringBytes();
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            if (k != 0) n += pb;
            if (sc != 0) n += pb;
        }
        return n;
    }

    /// Slot `slot`'s own state into `buf` (save: true) or back from it, in the stream's order (no wait): with the
    /// extent's compressed rows, the state a prompt pass stands in at a chunk end.
    pub fn snapCopy(m: *Model, slot: usize, buf: u64, save: bool) !void {
        const rb = m.eng.ringBytes();
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var at = buf;
        for (m.rings) |r| {
            try m.copyState(r + slot * rb, at, rb, save);
            at += rb;
        }
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            for ([_]u64{ k, sc }) |live| if (live != 0) {
                try m.copyState(live + slot * pb, at, pb, save);
                at += pb;
            };
        }
    }

    /// The compressed rows of positions [0, at) of the extent at `from` copied into the extent at `to` (a kept prompt
    /// resumed in another extent while a live stream holds its own), in the stream's order: every kv-source layer's
    /// codes and scales and its index keys, the rows view() places at those positions.
    pub fn copyExtent(m: *Model, from: usize, to: usize, at: usize) !void {
        const c = m.eng.c;
        const cs = &m.caches;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (from % r != 0 or to % r != 0 or at % r != 0) return error.BadView;
            const n = at / r;
            try m.copySpan(cs.comp_codes[i], from / r, to / r, n, c.head_dim / 2);
            try m.copySpan(cs.comp_scales[i], from / r, to / r, n, c.head_dim / 16);
            if (lay.idx_wk != null) {
                try m.copySpan(cs.idx_codes[i], from / r, to / r, n, c.index_head_dim / 2);
                try m.copySpan(cs.idx_scales[i], from / r, to / r, n, c.index_head_dim / 32);
            }
        }
    }

    fn copySpan(m: *Model, plane: u64, from: usize, to: usize, n: usize, row: usize) !void {
        const d = m.ctx.d;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(plane + to * row, plane + from * row, n * row, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    fn copyState(m: *Model, live: u64, kept: u64, bytes: usize, save: bool) !void {
        const d = m.ctx.d;
        const dst = if (save) kept else live;
        const src = if (save) live else kept;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    /// Each layer's gate_bias_vl (f32 [experts], the checkpoint's extra file) on the device: Gate.forward's bias at
    /// image-span positions.
    fn loadVisionBias(m: *Model, path: []const u8) !void {
        const core = @import("core");
        var f = try core.safetensors.File.open(m.gpa, m.io, path);
        defer f.close(m.io);
        var nb: [64]u8 = undefined;
        for (m.w.layers, 0..) |*lay, i| {
            const t = f.get(try std.fmt.bufPrint(&nb, "layers.{d}.ffn.gate.bias_vl", .{i})) orelse return error.MissingVisionBias;
            if (!t.is(.f32, &.{m.cfg.experts})) return error.BadVisionBias;
            const dst = try m.arena.take(t.bytes.len);
            const tmp = try m.gpa.dupe(u8, t.bytes); // (a copy out of the file mapping first: see vit.zig)
            defer m.gpa.free(tmp);
            try m.ctx.d.check(m.ctx.d.api.cuMemcpyHtoD_v2(dst, tmp.ptr, tmp.len), "cuMemcpyHtoD");
            lay.gate_b_vl = dst;
        }
    }

    /// The filling prompt's image spans (absolute positions, in order) and their rows bf16 [tokens, hidden] from the
    /// host, after fillBegin and before its chunks.
    pub fn setImages(m: *Model, spans: []const prompt.Span, rows: []const u8) !void {
        if (rows.len > 0) {
            if (m.img_buf == null or m.img_buf.?.len < rows.len) {
                if (m.img_buf) |*b| b.free();
                m.img_buf = null;
                m.img_buf = try cuda.DeviceBuffer.alloc(m.ctx.d, rows.len);
            }
            const d = m.ctx.d;
            try d.check(d.api.cuMemcpyHtoDAsync_v2(m.img_buf.?.ptr, rows.ptr, rows.len, m.stream.handle), "cuMemcpyHtoDAsync");
            try m.synchronize(); // (the caller's rows may go once this returns)
        }
        m.spans.clearRetainingCapacity();
        try m.spans.appendSlice(m.gpa, spans);
        m.ch.spans = m.spans.items;
        m.ch.img_rows = if (m.img_buf) |b| b.ptr else 0;
        if (m.waves) {
            m.ch2.spans = m.ch.spans;
            m.ch2.img_rows = m.ch.img_rows;
        }
    }

    pub fn fillBegin(m: *Model, slot: usize, base: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        if (m.eh) |*eh| { // rows read ahead for another prompt are not this one's, even at the same positions
            eh.ahead_start = std.math.maxInt(usize);
            eh.ahead_ready = .{ false, false };
        }
        m.spans.clearRetainingCapacity();
        m.ch.spans = &.{};
        if (m.waves) m.ch2.spans = &.{};
        m.view = try m.caches.view(&m.eng, slot, base);
        for (m.rings, m.ring_view) |r, *v| v.* = r + slot * m.eng.ringBytes();
        m.fill_slot = slot;
    }

    /// The prompt's chunk of n rows from position `start` (seq: the stream's ids through the chunk's end) through every
    /// layer with the served bounded replay (replay = the prompt's length - the window: an encoder-only chunk stops at
    /// the decoder's first layer). A chunk that reaches the head leaves the logits of its last row in ch.head_g and,
    /// drafting, its DSpark taps absorbed into the slot's drafter rings. True when it reached the head.
    pub fn fillChunk(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        if (n == 0 or n > chunk_rows or seq.len != start + n) return error.BadChunk;
        if (m.waves and n > wave_rows) return m.fillPair(seq, start, n, replay);
        const e = &m.eng;
/// TF_DS_NACT_LOG=1 (log only, off by default): the device ring the decode groupings append to (mx.nact_ring; none when off: no launch, no copy),
/// written by flushDiag to nact-rank<R>.txt in TF_DS_NACT_DIR (default the folder's log directory): per decode launch, the rows, the distinct
/// experts, how many of them this node owns and whether the shared expert is among them: what a launch's gate / up and down weight bytes are.
var nact_buf: ?cuda.DeviceBuffer = null;
var diag_calls: u32 = 0;
var diag_last_ns: u64 = 0;
fn nactOpen(d: *const cuda.Driver) void {
    if (nact_buf != null) return;
    const b = cuda.DeviceBuffer.alloc(d, 64 + mx.nact_records * 32) catch |e| return std.log.warn("TF_DS_NACT_LOG=1: off: no device memory for the log ({s})", .{@errorName(e)});
    b.fill8(0, null) catch {};
    nact_buf = b;
    mx.nact_ring = b.ptr;
    std.log.info("TF_DS_NACT_LOG=1: each decode launch's rows, distinct experts and owned experts are logged ({d} records) to nact-rank<R>.txt", .{mx.nact_records});
}

fn nactWrite(m: *Model, b: cuda.DeviceBuffer) void {
    const gpa = std.heap.page_allocator;
    const words = gpa.alloc(u64, 8 + mx.nact_records * 4) catch return;
    defer gpa.free(words);
    b.download(0, std.mem.sliceAsBytes(words)) catch return;
    const recs = gpa.alloc(mx.NactRec, mx.nact_records) catch return;
    defer gpa.free(recs);
    const n = mx.nactDecode(words, recs);
    const text = mx.nactRender(gpa, m.rank, m.world, recs[0..n]) catch return;
    defer gpa.free(text);
    var name: [48]u8 = undefined;
    const file = std.fmt.bufPrint(&name, "nact-rank{d}.txt", .{m.rank}) catch return;
    if (rdma.diagWriteFile("TF_DS_NACT_DIR", file, text)) std.log.info("TF_DS_NACT_LOG: {d} launches written ({s})", .{ n, file });
}

/// Why the decode exchanges cannot all ride the rings on this rank (null: they can): the rings are open and every decode exchange fits
/// them (prompt2d.decodeOff).
pub fn decodeOff(rings: bool, shape: prompt2d.DecodeShape) ?[]const u8 {
    return std.fmt.bufPrint(&S.buf, "Rocinante decode: every decode round and drafter pass exchange rides the RDMA rings (the drafter's per-step bests over the four nodes' ring in rank order {s}); a ring that cannot take an exchange ends it by name; on every rank (this one: {d})", .{ order, rank }) catch S.buf[0..0];
fn rankOrderText(order: *const [4]u32) []const u8 {
pub fn startSettings(o: Options) prompt2d.Settings {
    return .{
        .world = o.world,
        .streams = @intCast(o.streams),
        .round_rows = @intCast(o.round_rows),
        .rails = if (o.rdma_devices) |list| @intCast(std.mem.count(u8, list, ",") + 1) else 0,
        .drafts = @intFromBool(o.drafts),
        .graphs = @intFromBool(o.graphs),
        .graph_cap = @intCast(@min(o.graph_cap, std.math.maxInt(u32))),
        .pool = o.pool,
        .replay_floor = @intFromBool(prompt.replay_floor),
    };
}

        why = "needs the four-node 2D split with Rocinante";
    const rc = try roci_engine.Roci.openLane(gpa, d, m.comm.rank, node, m.comm.world, m.two.?.rank_of, &m.fan, rails, Model.rdmaGid() orelse 3, 1, m.two.?.rc.?, .host);
        const is_default = speed_defaults.defaulted("TF_DS_ROCI_STAGGER");
    m.two_b = try prompt2d.Two.init(c, &m.w, &m.arena, sp, m.pass_rows);
    m.two_b.?.rc = try roci_engine.Roci.openLane(gpa, d, m.comm.rank, @intCast(m.two.?.node()), m.comm.world, m.two.?.rank_of, &m.fan, rails, Model.rdmaGid() orelse 3, 2, m.two.?.rc.?, .prompt);
    if (m.pair_log) {
        // the span log's events: a pair of timing events per outstanding pair (pair_spans.cap of them), made here and only here
        m.span_t0 = try m.gpa.alloc(cuda.Event, pair_spans.cap);
        m.span_join = try m.gpa.alloc(cuda.Event, pair_spans.cap);
        for (0..pair_spans.cap) |i| {
            m.span_t0[i] = try cuda.Event.init(d, true);
            m.span_join[i] = try cuda.Event.init(d, true);
        }
    }
    mxk: ?mx.Kernels = null, // Format.native: DeepSeek's own FP8 / MXFP4 checkpoint's kernels (mx.zig)
    rope_buf: cuda.DeviceBuffer,
    arena: prompt.Arena,
    graph_fallbacks: std.atomic.Value(u64) = .init(0), // either kind's work run eagerly after a failed capture (/metrics)
    // Track B (TF_DS_SAMPLE_CAND=1): a round's candidates on the device (vals f32, ids i64, flags i32; max_rows rows of
    // sampling.max_candidates) and on the host
    cand: u64 = 0,
    cand_vals: [round.max_rows * sampling.max_candidates]f32 = undefined,
    cand_ids: [round.max_rows * sampling.max_candidates]i64 = undefined,
    cand_flags: [round.max_rows]i32 = undefined,
    span_ring: pair_spans.Ring = .{}, // TF_DS_ROCI_PAIR_LOG: the pairs whose span is not printed yet
    span_t0: []cuda.Event = &.{},
    span_join: []cuda.Event = &.{},
    span_slot: ?usize = null, // the slot of the pair being issued
    // digests by rank (rank 0 holds every rank's, a follower its own; prompt2d.no_digest: none)
    /// Loads rank o.rank's weights from the lane's rank cache, links the ranks (rank 0 listens on o.port)
    /// and sets up every buffer. Needs `ctx` current on this thread.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: Options) !*Model {
        readWorkerEnv();
        {
            // a name that no longer exists, in the environment, ends the start by name (every rank reads the same environment: every rank refuses alike)
            var why: [512]u8 = undefined;
            if (retiredSwitch(&why)) |text| {
                std.log.err("{s}", .{text});
                return error.RetiredSwitch;
            }
        }
        if (o.world != 4) {
            std.log.err("WORLD {d}: the engine runs on the four-node 2D split only (Rocinante carries every exchange between the Sparks; the two-Spark TP2 split needed NCCL)", .{o.world});
            return error.NotPortedYet;
        }
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.spans = .empty;
        m.img_buf = null;
        // WP6 (docs/prefill-plan.md): TF_DS_IDX_RB=2 batches the indexer's prompt-chunk scoring kernel to the
        // kit's own captured RB=2 variant (dequantizing a 64-key tile once for 2 rows instead of 1); the kit
        // never captured RB above 2. Off (RB=1, today's behavior) unless set; validate on your own ring first.
        if (std.c.getenv("TF_DS_IDX_RB")) |v| tri_index.idx_rb = std.fmt.parseInt(usize, std.mem.span(v), 10) catch tri_index.idx_rb;
        if (std.c.getenv("TF_DS_REPLAY_FLOOR")) |v| prompt.replay_floor = std.mem.eql(u8, std.mem.span(v), "1"); // every rank alike
        const ring_wait_text: ?[]const u8 = if (std.c.getenv("TF_DS_RING_WAIT")) |v| std.mem.span(v) else null;
        prompt2d.ring_wait_asked = prompt2d.switchDefaultOn(ring_wait_text); // on unless set to something but 1; every rank alike (Settings)
        if (!prompt2d.ring_wait_asked) std.log.info("TF_DS_RING_WAIT: off (set to something but 1): the rings run their proxy threads", .{});
            prompt2d.ring_wait_asked = false;
        }
        prompt2d.ring_wait_explicit = ring_wait_text != null;
        if (std.c.getenv("TF_DS_ROCI_XLOG")) |v| rdma.xlog = std.mem.eql(u8, std.mem.span(v), "1");
        if (std.c.getenv("TF_DS_L2_DN_MB")) |v| round.l2_dn_mb = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_dn_mb;
        if (std.c.getenv("TF_DS_L2_DEC_MB")) |v| round.l2_dec_mb = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_dec_mb;
        m.graph_fallbacks = .init(0);
        m.span_ring = .{};
        m.span_t0 = &.{};
        m.span_join = &.{};
        m.span_slot = null;
        // rank 0's ports for its followers listen before its weights load: a follower that loads first connects and
        // waits in its port's backlog instead of retrying until rank 0 and the followers before it are in
        m.fan = try prompt2d.Fan.listenAt(o.master, o.listen, o.port, o.rank, o.world);
        m.fan.share_digests = o.world > 1; // a rank's store is named from all four weight files
        errdefer m.fan.close();
        m.vfan = try prompt2d.Fan.listenAt(o.master, o.listen, o.port + prompt2d.host_links_port_offset, o.rank, o.world);
        // world 4: the 2D node this rank serves (TF_DS_2D_ORDER; its weight file is that node's)
        const ord = if (o.world == 4) try prompt2d.order() else [4]u32{ 0, 1, 2, 3 };
        const role: u32 = if (o.world == 4) ord[o.rank] else o.rank;
        const cache_name = try rank_cache.find(a, io, m.cache_dir, role, o.world);
        m.ix = try rank_cache.open(gpa, io, m.cache_dir, cache_name);
        errdefer m.ix.deinit();
        m.cache_file = try m.cache_dir.openFile(io, cache_name, .{});
        errdefer m.cache_file.close(io);
        var sp = try prompt2d.split(role, o.world);
        if (std.c.getenv("TF_DS_DUMP")) |dd| {
            prompt.dump_dir = std.mem.span(dd);
            prompt.dump_node = role;
            // WP1's exit gate wants several layers' tensors in one dump (x, pa, pm, h at layers 0, 8, 19): accept a
            // comma list, not just one layer, so one instrumented restart covers all of them.
            if (std.c.getenv("TF_DS_DUMP_LAYER")) |dl| {
                var it = std.mem.splitScalar(u8, std.mem.span(dl), ',');
                var i: usize = 0;
                while (it.next()) |tok| : (i += 1) {
                    if (i >= prompt.dump_layers.len) break;
                    prompt.dump_layers[i] = std.fmt.parseInt(i64, tok, 10) catch -1;
                }
            }
        }
        if (std.c.getenv("TF_DS_DRAFT_PROF")) |v| draft.dprof = v[0] == '1';
        if (std.c.getenv("TF_DS_CONF_CAL")) |v| {
            var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
            draft.conf_cal[0] = std.fmt.parseFloat(f64, it.next() orelse "1") catch return error.BadConfCal;
            draft.conf_cal[1] = std.fmt.parseFloat(f64, it.next() orelse "0") catch return error.BadConfCal;
        }
        if (std.c.getenv("TF_DS_DRAFT_MS")) |v| draft.cost_ms = std.fmt.parseFloat(f64, std.mem.span(v)) catch return error.BadDraftMs;
        if (std.c.getenv("TF_DS_DRAFT_LOG")) |v| {
            draft.dlog = std.mem.span(v);
            draft.ddump_node = role;
        }
        if (std.c.getenv("TF_DS_DRAFT_DUMP")) |dd| {
            draft.ddump = std.mem.span(dd);
            draft.ddump_node = role;
            if (std.c.getenv("TF_DS_DRAFT_DUMP_PASS")) |dp| draft.ddump_pass = std.fmt.parseInt(usize, std.mem.span(dp), 10) catch 2;
        }
        sp.parity = sp.pair != null and parityGU(); // TF_DS_2D_GU=parity: the balanced experts split
        // the ranks' links (rank 0 takes its followers in order; a follower retries its connect for TF_DS_CONNECT_S)

        try m.fan.join(io, o.master, o.port, o.rank, o.world, prompt2d.connectSeconds());
        try m.vfan.join(io, o.master, o.port + prompt2d.host_links_port_offset, o.rank, o.world, prompt2d.connectSeconds());
        m.comm = try m.fan.settingsChecked(o.rank, o.world, mine, rankDigest(), &heard);
        errdefer m.comm.deinit();
        m.comm.host = .{ .star = .{ .links = &m.vfan.links, .rank = o.rank, .world = o.world, .busy = &m.star_busy } };
        m.par = if (sp.parity and parOne() and c.format == .exl3) try exl3_experts2d.ParKernels.load(d, cuda.kernels.dsv41_experts_par) else null;
        errdefer if (m.par) |*x| x.unload();
        m.mxk = if (c.format == .native) try mx.Kernels.load(d, cuda.kernels.dsv41_native) else null;
        if (std.c.getenv("TF_DS_NACT_LOG")) |v| if (std.mem.eql(u8, std.mem.span(v), "1")) nactOpen(d);
        errdefer if (m.mxk) |*x| x.unload();
        // round over the host links)
        errdefer if (m.rdma_rings) |*x| x.close(gpa);
            m.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, m.pass_rows);
            m.two.?.rank_of = prompt2d.rankOf(ord);
            try m.two.?.agreeTopology(d, &m.comm, m.stream);
            try m.two.?.agreeIdxShard(d, &m.comm, m.stream);
            // Rocinante carries every exchange between the Sparks. Each family's start check is every rank's own (its kernel set included),
            // then on every rank or the start ends naming the rank and its reason: there is no other transport to fall back to.
            const Two = prompt2d.Two;
            const rows4 = [_]usize{ 256, 512, 1024, m.pass_rows / m.comm.world };
            try Two.require(d, &m.comm, m.stream, "the front on owned rows (qr and y gathered)", try prompt.frontOwnedOff(gpa, SetFinds{ .set = &m.set }, m.mxk != null, m.w.layers, c.layers / 2, c.eps, &rows4, m.comm.world));
            try Two.require(d, &m.comm, m.stream, "the routing on owned rows (codes, picks and weights gathered)", try prompt.ffnCodesOff(gpa, SetFinds{ .set = &m.set }, m.mxk != null, m.w.layers, c.layers / 2, c.experts, c.top_k, c.routed_scaling, c.top_k + 1, &rows4, m.comm.world));
            try Two.require(d, &m.comm, m.stream, "the expert pack as codes", prompt2d.packCodesOff(m.mxk != null, m.two.?.gu[m.two.?.p], m.two.?.ir));
            try Two.require(d, &m.comm, m.stream, "the Engram sums at owned rows", try prompt.engramOwnerSumOff(gpa, SetFinds{ .set = &m.set }, m.two.?.ew, m.w.layers, c.layers / 2, c.eps, c.hidden, c.hc, &rows4, m.comm.world));
            try Two.require(d, &m.comm, m.stream, "the boundary layer on owned rows", prompt.boundaryOwnedOff(m.w.layers[c.layers / 2]));
            try Two.require(d, &m.comm, m.stream, "wo_b's input as codes", prompt2d.catCodesOff(m.mxk != null, m.w.layers, m.two.?.uw));
            {
                // the owner sums over the transport: each rank's start check (the kit's two-block readers included)
                const readers: bool = if (m.extra) |*x| try prompt.rociReadersOff(gpa, SetLookups{ .set = x }, &[_]usize{m.pass_rows / m.comm.world}, c.hidden, m.two.?.ew[m.two.?.p], c.eps, c.hc_eps, c.hc_iters) == null else false;
                const why = prompt2d.rociOff(m.two.?.ow, m.pass_rows, readers) orelse if (o.rdma_devices == null) @as(?[]const u8, "needs the rails' devices (TF_DS_RDMA_DEVICES)") else null;
                try Two.require(d, &m.comm, m.stream, "the owner sums over Rocinante", why);
                if (m.two.?.sentFirst()) std.log.info("TF_DS_ROCI_SENT_FIRST=1: wo_b runs in one launch whose first CTAs are the row quarters an owner sum's first part sends, and the last of those CTAs rings the part's doorbell itself (needs TF_DS_MMA_SF_EXACT), on every rank", .{});
                const devs = o.rdma_devices.?;
                var pit = std.mem.splitScalar(u8, devs, ',');
                const rails: [2][]const u8 = .{ pit.next() orelse return error.RociRails, pit.next() orelse return error.RociRails };
                // the producers that write the arena (the dense GEMMs) store whole lines there
                if (m.mxk) |*x| try x.useStaged(m.two.?.rc.?.range()[0], m.two.?.rc.?.range()[1]);
            }
            try Two.require(d, &m.comm, m.stream, "wo_b's and the expert pack's codes over Rocinante", prompt2d.rociCodesOff(m.two.?.uw, m.two.?.gu, m.two.?.ir, c.top_k + 1));
            std.log.info("Rocinante carries wo_b's and the expert pack's codes and scales: quantized into the transport's arena, moved to the column partner as RDMA WRITEs, on every rank", .{});
            try Two.require(d, &m.comm, m.stream, "the gathers over Rocinante", prompt2d.rociGathersOff(m.mxk != null, c.hidden, m.w.layers[0].wq_a.n, c.head_dim, c.index_topk, c.index_head_dim, m.w.layers[0].experts.dims, c.top_k + 1));
            std.log.info("Rocinante carries the gathers (qr and y, x, the routing's codes and picks, the indexer's top-k, the boundary's cache rows): their tensors are in the transport's arena and move as RDMA WRITEs to the three other nodes, on every rank", .{});
            // each rank's start check of every launch of every slab (its kernel sets, the two-block readers in the extra set)
            try Two.require(d, &m.comm, m.stream, "chunks of any row count over Rocinante", try any_check.anyOwnedOff(gpa, a, KitFinds{ .set = &m.set, .extra = if (m.extra) |*x| x else null }, c.*, m.w.layers, c.layers / 2, true, m.pass_rows));
            std.log.info("Rocinante carries the owned path and its exchanges for a chunk of any row count (a quarter of the rows, rounded up to 16, a rank), the decoder rows' boundary and handoff, on every rank", .{});
            // each rank's start check (its settings, then every launch of every layer at every slab of a decoder window in its kernel sets)
            try Two.require(d, &m.comm, m.stream, "the decoder half over Rocinante", prompt2d.rociDecOff(prompt.quant_once, c.dspark_taps.slice().len, c.hidden) orelse try any_check.decoderOff(gpa, a, KitFinds{ .set = &m.set, .extra = if (m.extra) |*x| x else null }, c.*, m.w.layers));
            std.log.info("Rocinante carries the decoder half (layers {d} to {d}) on owned rows of the decoder rows, its sums and gathers, the DSpark taps and the head's row, on every rank", .{ c.layers / 2, c.layers - 1 });
            const proj_why: ?[]const u8 = if (prompt2d.idx_proj_owned_asked) try prompt.idxProjOwnedOff(gpa, SetFinds{ .set = &m.set }, m.mxk != null, m.two.?.idx_shard, m.w.layers, c.candidate_source, c.rope_dim, c.index_heads, c.index_head_dim, &rows4, m.comm.world) else null;
                const ring_wait_on = ringWaitOn(rk.hasWait(), devices.items);
                m.rdma_rings = try ring2d.Rings.open(gpa, d, rk, devices.items, role, ord, .{ .max_bytes = ring_bytes, .gid_index = rdmaGid(), .pdl = rdmaPdl(), .collect_peers = peers and rk.collect_peers != null, .wc = sw.wc, .inplace = sw.inplace, .roci = ringWordOf(), .roci_deadline_ns = ringWordDeadlineNs(), .wait = ring_wait_on, .wait_explicit = prompt2d.ring_wait_explicit }, &m.comm, m.stream, !weights.draft_4way);
                m.two.?.rings = &m.rdma_rings.?;
            }
            // the decode exchanges: each rank's start check (the rings, the shapes), then on every rank or the start ends
            {
                try Two.require(d, &m.comm, m.stream, "the decode exchanges on the rings", decodeOff(m.two.?.rings != null, shape));
                std.log.info("{s}", .{decodeLine(o.rank, rankOrderText(&m.two.?.rank_of))});
        prompt.hc_stride = prompt2d.hc_stride_asked and m.two != null;
        logPdlControls(m.comm.rank);
        std.log.info("transport: Rocinante carries every exchange between the Sparks; no NCCL is loaded or made (rank {d})", .{o.rank});
        if (m.mxk) |*x| m.eng.mx = x;
        m.cand = try m.arena.take(round.max_rows * (sampling.max_candidates * 12 + 4));
        m.eng.agree_flag = try m.arena.take(64); // G0.3: the graph captures' agreement bytes (agreeAll, agreeCapture)
        if (o.graphs) {
            m.graphs = round.Graphs.init(gpa, o.graph_cap);
            m.graphs.?.lazy.fallbacks = &m.graph_fallbacks;
        }
        if (o.side) try m.rd.useSide(&m.eng);
                m.pgraphs.?.lazy.fallbacks = &m.graph_fallbacks;
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
                const owned = false; // (TF_DS_ENGRAM_OWNED is a retired name: the whole tables are on every node)
        if (c.vision and c.format == .exl3) { // (Format.native: the checkpoint's own gate.bias_vl, loaded with the weights)
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            m.loadVisionBias(path) catch |err| if (o.vision_bias != null or err != error.FileNotFound) return err;
        }

        for (m.span_t0) |*ev| ev.deinit();
        for (m.span_join) |*ev| ev.deinit();
        if (m.span_t0.len != 0) m.gpa.free(m.span_t0);
        if (m.span_join.len != 0) m.gpa.free(m.span_join);
        m.span_t0 = &.{};
        m.span_join = &.{};
    /// The diagnostic logs (TF_DS_ROCI_XLOG, TF_DS_NACT_LOG) to their files: every 256th call and, forced, at close. Called after a round's work, with
    /// the stream synchronized here: never inside a round. Nothing runs when both logs are off.
    pub fn flushDiag(m: *Model, force: bool) void {
        if (!rdma.xlog and nact_buf == null) return;
        diag_calls +%= 1;
        // every 256th call, or 3 s after the last write (a phase of a few hundred rounds then lands in the file whole but for its last seconds)
        const t_now = m.now();
        if (!force and diag_calls % 256 != 0 and t_now -% diag_last_ns < 3 * std.time.ns_per_s) return;
        diag_last_ns = t_now;
        m.stream.synchronize() catch return;
        if (m.ring) |r| r.xlogWriteFile();
        if (m.rdma_rings) |rs| for ([_]?*rdma.Ring{ rs.all4, rs.col, rs.pair }) |y| if (y) |r| r.xlogWriteFile(); // (a TP4 run's rings: m.ring is its single-ring twin)
        if (nact_buf) |b| nactWrite(m, b);
    }

        m.flushDiag(true);
        if (m.dr) |*dr| dr.deinit();
        if (m.timer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        if (m.ptimer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        m.rd.dropSide(&m.eng);
        m.rd.dropPrefetch();
        m.rd.dropPar();
        if (m.mxk) |*x| x.unload();
        m.ex.unload();
        m.ops.unload();
        m.exk.unload();
        m.lg.unload();
        m.pf.unload();
        m.set.deinit();
        if (m.two) |*t| if (t.rc) |rc| rc.close();
        m.comm.deinit();
    /// The decode plan's G0.2 controls (TF_DS_DEC_PDL, TF_DS_RDMA_PDL, TF_DS_TRITON_PDL_ATTR, TF_DS_PDL), one line on each
    /// rank when any is off, so a run can show that every rank has the same ones.
    fn logPdlControls(rank: usize) void {
        const off = struct {
            fn f(name: [*:0]const u8) bool {
                return if (std.c.getenv(name)) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
            }
        }.f;
        const native = off("TF_DS_DEC_PDL");
        const ring = !rdmaPdl();
        const tri_attr = off("TF_DS_TRITON_PDL_ATTR");
        const all = off("TF_DS_PDL");
        if (!(native or ring or tri_attr or all)) return;
        const w = struct {
            fn f(o: bool) []const u8 {
                return if (o) "off" else "on";
            }
        }.f;
        std.log.info("decode PDL controls on rank {d}: native decode kernels {s}, RDMA kernels {s}, Triton consumer launch attribute {s}, every launch attribute {s}", .{ rank, w(native or all), w(ring or all), w(tri_attr or all), w(all) });
    }

    /// TF_DS_RDMA_PDL=0 (or the Python build's TF_RDMA_PDL=0): the RDMA rings' kernels as ordinary launches with their
    /// pdl argument 0 (no griddepcontrol in them): the decode plan's G0.2 control (b). Unset or any other value: as served.
    pub fn rdmaPdl() bool {
        for ([_][*:0]const u8{ "TF_DS_RDMA_PDL", "TF_RDMA_PDL" }) |name| if (std.c.getenv(name)) |v| return !std.mem.eql(u8, std.mem.span(v), "0");
        return true;
    }

    /// Why the card's WAIT cannot run on the rings when it is wanted, for the fallback to the proxy threads when TF_DS_RING_WAIT is unset (the
    /// default); null when it can. Asked for with 1 it is never a fallback: Ring.create refuses by name.
    pub fn ringWaitFallback(explicit: bool, has_kernels: bool, mlx_ok: bool, atomic_why: ?[]const u8) ?[]const u8 {
        if (explicit) return null;
        if (atomic_why) |why| return why; // (TF_DS_RING_WORD_SIGNAL=atomic on a card that cannot do ATOMICs)
        if (!has_kernels) return "the rings' kernels have no dsv41_rdma_stage_wait / publish_wait / await_send_wait";
        if (!mlx_ok) return "libmlx5.so.1 does not load (the card's WAIT needs its DEVX entry points)";
        return null;
    }

    /// Whether the rings' queue pairs wait in the card: the switch after its checks (ring_wait_asked), and the fallback; each outcome has its line.
    fn ringWaitOn(has_kernels: bool, devices: []const []const u8) bool {
        if (!prompt2d.ring_wait_asked) return false;
        var mlx_ok = false;
        if (mx5.Mlx5.open()) |m0| {
            var m = m0;
            m.close();
            mlx_ok = true;
        } else |_| {}
        var ab: [300]u8 = undefined;
        const atomic_why: ?[]const u8 = if (prompt2d.ring_word_atomic_asked) rdma.probeAtomic(&ab, devices) else null;
        if (ringWaitFallback(prompt2d.ring_wait_explicit, has_kernels, mlx_ok, atomic_why)) |why| {
            std.log.info("TF_DS_RING_WAIT: unset (the default is the card's WAIT) but {s}: the rings run their proxy threads", .{why});
            prompt2d.ring_wait_asked = false;
            return false;
        }
        std.log.info("TF_DS_RING_WAIT: {s}: the rings' queue pairs wait in the card (no proxy threads; =0 for the proxy threads)", .{if (prompt2d.ring_wait_explicit) "1" else "unset (the default)"});
        return true;
    }

    /// TF_DS_RDMA_GID: the RDMA rings' GID index on every device; unset, each device's first RoCE v2 IPv4 GID.
    pub fn rdmaGid() ?u8 {
        const v = std.c.getenv("TF_DS_RDMA_GID") orelse return null;
        return std.fmt.parseInt(u8, std.mem.span(v), 10) catch null;
    }

    fn readKit(a: std.mem.Allocator, io: std.Io, kit: []const u8, name: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ kit, "cubins", name }), a, .limited(1 << 28), .@"16", null);
    }

    /// The link to the other rank (rank 0's to rank 1, rank 1's to rank 0).
    pub fn peer(m: *const Model) link.Link {
        return m.fan.links[0].?;
    }

    /// Rank 0's links to its followers: rank 1's, or ranks 1-3's on the four-node split.
    pub fn followers(m: *const Model) []const ?link.Link {
        return m.fan.links[0 .. m.world - 1];
    }

        if (m.eh) |*eh| prompt.engramWorkerStop(eh);
    }

    /// A kept prompt state's own bytes (the prompt cache's snapshots; the compressed plane stays in the stream's
    /// extent): one slot's window ring of every layer and positional stores of every ratio-2 layer.
    pub fn snapBytes(m: *const Model) usize {
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var n: usize = m.rings.len * m.eng.ringBytes();
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            if (k != 0) n += pb;
            if (sc != 0) n += pb;
        }
        return n;
    }

    /// Slot `slot`'s own state into `buf` (save: true) or back from it, in the stream's order (no wait): with the
    /// extent's compressed rows, the state a prompt pass stands in at a chunk end.
    pub fn snapCopy(m: *Model, slot: usize, buf: u64, save: bool) !void {
        const c = e.c;
        const ch = &m.ch;
        // an image position's id is negative in the sequence (its n-grams dead for Engram): its embedding's id is the
        // image token's, which its span row then replaces
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        var shared: prompt.Shared = .{};
        const pt = m.ptimer;
        prompt.ptimer = pt;
        defer prompt.ptimer = null;
        if (pt) |t| try t.mark(e.s, .start);
        try prompt.begin(e, ch, m.ids64[0..n], start, m.host_pos);
        if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
        if (pt) |t| try t.mark(m.stream, .embed);
        const taps = m.dr != null;
        var floor: usize = 0;
        var kv_done: ?usize = null;
        // Engram rows read ahead while the GPU runs the layers before them. A fill of the prompt (seq a prefix of
        // fill_seq): the fill's worker thread (prompt.engramWorker), up to a chunk ahead of the layers, so a chunk's
        // table reads overlap the chunk before it. Else (a chunk alone): a thread from the chunk's start, the first
        // layer's rows then the next's once the first is applied.
        const use_wk = m.eh != null and wk_on and m.fill_seq.len >= start + n and m.fill_seq.ptr == seq.ptr;
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        defer if (ahead) |th| th.join();
        if (m.eh) |*eh| {
            if (use_wk) {
                ahead = null;
                const same = eh.wk_thread != null and eh.wk_seq.ptr == m.fill_seq.ptr and eh.wk_seq.len == m.fill_seq.len and
                    start >= eh.wk_start and (start - eh.wk_start) % chunk_rows == 0 and
            } else {
                prompt.engramWorkerStop(eh);
                if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
            }
        }
        for (0..c.layers) |li| {
            if (li == c.layers / 2) {
                if (!try prompt.replayCut(e, ch, &m.view, &shared, li, replay, m.host_pos)) {
                    // encoder-only: no synchronize either (as at the chunk's end below), so the next chunk's host steps
                    // run while this one's last layers do; the profile alone waits
                    if (pt) |t| {
                        try m.synchronize();
                        try t.finish();
                    }
                    return false;
                }
                floor = replay;
                kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (ahead) |th| {
                    th.join();
                    ahead = null;
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j, li);
            try prompt.attnMixes(e, ch, li);
            if (pt) |t| try t.mark(e.s, .mix_attn);
            try prompt.attention(e, ch, &m.view, &shared, li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
            if (pt) |t| try t.mark(e.s, .attn);
            // The gathers' own output, both sublayers, at every requested layer of the first chunk of at least 64 rows
            // (prompt.dump_layers; dump_done stops it after that chunk, below). The >= 64 guard keeps a smoke test or
            // validate.py's short prompts from spending the one-shot dump on a chunk too short to compare.
            if (prompt.dump_dir != null and !prompt.dump_done and ch.n >= 64) {
                prompt.dump(e, li, "ga", ch.ga, 2 * ch.n * c.hidden * 4);
                prompt.dump(e, li, "gm", ch.gm, 2 * ch.n * c.hidden * 4);
            }
            prompt.endLayer(ch);
        }
        try prompt.head(e, ch);
        if (ch.n >= 64) prompt.dump_done = prompt.dump_dir != null;
        if (pt) |t| try t.mark(e.s, .head);
        if (m.dr) |*dr| {
            // its taps (the tap layers side by side) into the slot's drafter rings; with TF_DS_ROCI_DEC the arena's taps region holds them
            // side by side already: one copy once the other ranks' rows have landed
            try prompt.tapsLanded(e);
            try e.ops.copyRows(e.s, ch.taps, dr.taps_w * 2, dr.at, dr.taps_w * 2, dr.taps_w * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
        }
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try e.s.synchronize();
            try t.finish();
        }
        return true;
    }

                }
                if (eh.ahead_err) |err| return err;
                try prompt.engramApply(e, ch, eh, li, seq);
                const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
                const ne = @min(2, c.engram_layers.slice().len);
                if (j + 1 < ne) {
                    if (!eh.ahead_ready[j + 1])
                        ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
                } else if (m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr) {
                    // the next chunk's rows while this one's layers run
                    const nn = @min(m.next_rows, m.fill_seq.len - start - n);
                    eh.ahead_start = start + n;
                    eh.ahead_n = nn;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    m.ahead = std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn }) catch null;
    /// Prints, in issue order, every outstanding pair's GPU span whose events are done (TF_DS_ROCI_PAIR_LOG; rank 0 prints, every rank keeps the ring).
    fn sweepSpans(m: *Model) void {
        const Ctx = struct {
            m: *Model,
            pub fn done(c: @This(), slot: usize) bool {
                return c.m.span_join[slot].done() catch true;
            }
            pub fn elapsed(c: @This(), slot: usize) f32 {
                return cuda.Event.elapsedMs(c.m.span_t0[slot], c.m.span_join[slot]) catch -1;
            }
            pub fn emit(c: @This(), pair: u64, ms: f32) void {
                if (c.m.comm.rank == 0) pairLog("ROCI pair {d}: GPU span {d:.1} ms", .{ pair, ms });
            }
        };
        _ = m.span_ring.sweep(Ctx{ .m = m });
    }

        if (m.pair_log) m.sweepSpans();
        _ = t;
        // every finished pair's GPU span (its first queued work to its second pass's last), in order
        if (m.pair_log) m.sweepSpans();
        m.span_slot = if (m.pair_log) m.span_ring.push(m.pair_count + 1) else null;
        if (m.span_slot) |sl| try m.span_t0[sl].record(m.stream);
        if (m.span_slot) |sl| try m.span_join[sl].record(m.passStreamOf(last));
        m.span_slot = null;
                }
                if (pt) |t| try t.mark(m.stream, .engram);
            }
            // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
            try prompt.attnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_attn);
            try prompt.attention(e, ch, &m.view, &shared, li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
            if (pt) |t| try t.mark(m.stream, .attn);
            ch.ga_rs = try prompt.gather(e, ch, ch.pa, ch.ga);
            if (pt) |t| try t.mark(m.stream, .gather_a);
            try prompt.ffnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_ffn);
            try prompt.moe(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .moe);
            ch.gm_rs = try prompt.gather(e, ch, ch.pm, ch.gm);
            if (pt) |t| try t.mark(m.stream, .gather_m);
            prompt.endLayer(ch);
        }
        try prompt.head(e, ch);
        if (pt) |t| try t.mark(m.stream, .head);
        if (m.dr) |*dr| {
            // its taps (the tap layers side by side) into the slot's drafter rings
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
        }
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try m.synchronize();
            try t.finish();
    /// Track B: the round's first `count` candidates a row (sampling.topCandidates' set and order) on the host, from the
    /// device's logits: vals and ids [R, count], flags [R] (non-zero: that row must be read whole, roundLogits).
    }

        if (m.two) |t| if (t.rc) |rc| {
            try rc.t.check(); // a failed transport stops the pass before its tokens are used
            rc.passLine();
        }; // a synchronisation point: every proxy has posted what the stream sent
        return m.logits[0..m.vocab];
    }

    /// The drafter's batched pass for N streams: each one's pending token at position q0 in its slot (its rings hold
    /// its positions below q0), `steps` drafts each (the drafter's drafts and confidences on the host).
    pub fn pass(m: *Model, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize) !void {

        }
        return true;
    const open = std.mem.indexOf(u8, code, "m.rdma_rings = try ring2d.Rings.open(gpa, d, rk, devices.items, role, ord, .{ .max_bytes = ring_bytes, .gid_index = rdmaGid(), .pdl = rdmaPdl(), .collect_peers = peers and rk.collect_peers != null, .wc = sw.wc, .inplace = sw.inplace, .roci = ringWordOf(), .roci_deadline_ns = ringWordDeadlineNs(), .wait = ring_wait_on, .wait_explicit = prompt2d.ring_wait_explicit }, &m.comm, m.stream, !weights.draft_4way);") orelse return error.TestUnexpectedResult;
    }

    /// fillChunk's pass of n > wave_rows rows with TF_DS_WAVES: wave A (rows [start, start + wave_rows), in ch2) and
    /// wave B (the rest, in ch) layer by layer, A's work then B's on the stream, each wave's partial gathers (the
    /// quarters of pa and pm) on cs over comm2 while the stream runs the other wave. A wave's kernels take the rows and
    /// inputs a fillChunk of its rows would: the stream runs A's layer before B's (B's attention reads A's keys of
    /// that layer) and B's before A's next, and a wave's consumers wait for its gathers. The head and the logits are
    /// B's (ch's), as the second of two chunks would leave them.
    fn fillPair(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        if (n > 2 * wave_rows) return error.BadChunk;
        const e = &m.eng;
        const c = e.c;
        const ws = [2]*prompt.Chunk{ &m.ch2, &m.ch };
        const st = [2]usize{ start, start + wave_rows };
        const nw = [2]usize{ wave_rows, n - wave_rows };
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        var hp: [2][]i64 = undefined;
        var shared: [2]prompt.Shared = .{ .{}, .{} };
        var live = [2]bool{ true, true };
        var posted = [2]bool{ false, false }; // a wave's gm gather on cs the stream has not waited for yet
        // whatever happens, the stream waits for every gather on cs before the pass's buffers are used again
        defer for (0..2) |w| if (posted[w]) m.stream.wait(m.wev[w][3]) catch {};
        for (0..2) |w| {
            hp[w] = m.host_pos[w * wave_rows ..];
            try prompt.begin(e, ws[w], m.ids64[w * wave_rows ..][0..nw[w]], st[w], hp[w]);
            if (ws[w].spans.len > 0) try prompt.embedSpans(e, ws[w]);
        }
        const taps = m.dr != null;
        var floor: usize = 0;
        var kv_done: ?usize = null;
        // the Engram rows read ahead for the whole pass (fillChunk's), each wave applying its part of them
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        defer if (ahead) |th| th.join();
        if (m.eh) |*eh| {
            if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                ahead.?.join();
                ahead = null;
            }
            const whole = eh.ahead_start == start and eh.ahead_n == n and eh.ahead_err == null and eh.ahead_ready[0] and (c.engram_layers.slice().len < 2 or eh.ahead_ready[1]);
            if (ahead == null and !whole) {
                eh.ahead_start = start;
                eh.ahead_n = n;
                eh.ahead_ready = .{ false, false };
                eh.ahead_err = null;
                ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
            }
        }
        for (0..c.layers) |li| {
            if (li == c.layers / 2) {
                for (0..2) |w| {
                    try m.waitGm(&posted, w);
                    live[w] = try prompt.replayCut(e, ws[w], &m.view, &shared[w], li, replay, hp[w]);
                }
                if (!live[1]) return false; // encoder-only (wave A's rows too): fillChunk's, no synchronize
                floor = replay;
                kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (ahead) |th| {
                    th.join();
                    ahead = null;
                }
                if (eh.ahead_err) |err| return err;
                const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
                const ne = @min(2, c.engram_layers.slice().len);
                // the next layer's read-ahead takes the hashes of this one's (the whole pass's): only after rows read
                // ahead, never after the waves hashed their own rows here
                const read_ahead = j < 2 and eh.ahead_ready[j] and eh.ahead_start == start and eh.ahead_n == n;
                for (0..2) |w| if (live[w]) {
                    try m.waitGm(&posted, w);
                    try prompt.engramApply(e, ws[w], eh, li, seq[0 .. st[w] + nw[w]]);
                };
                if (j + 1 < ne) {
                    if (!eh.ahead_ready[j + 1] and read_ahead)
                        ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
                } else if (m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr) {
                    const nn = @min(chunk_rows, m.fill_seq.len - start - n);
                    eh.ahead_start = start + n;
                    eh.ahead_n = nn;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    m.ahead = std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn }) catch null;
                }
            }
            for (0..2) |w| if (live[w]) {
                const ch = ws[w];
                try m.waitGm(&posted, w);
                if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
                try prompt.attnMixes(e, ch, li);
                try prompt.attention(e, ch, &m.view, &shared[w], li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
                try m.gatherOnCs(w, 0, ch.pa, ch.ga);
            };
            for (0..2) |w| if (live[w]) {
                const ch = ws[w];
                try m.stream.wait(m.wev[w][1]);
                try prompt.ffnMixes(e, ch, li);
                try prompt.moe(e, ch, li);
                try m.gatherOnCs(w, 2, ch.pm, ch.gm);
                posted[w] = true;
                prompt.endLayer(ch);
            };
        }
        for (0..2) |w| try m.waitGm(&posted, w);
        try prompt.head(e, &m.ch);
        if (m.dr) |*dr| {
            const dw = c.hidden;
            for (0..2) |w| if (live[w]) {
                const ch = ws[w];
                for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
                try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
            };
        }
        return true;
    }

    /// Wave w's partial `src` gathered into `dst` on cs (events k, k + 1: ready on the stream, gathered on cs).
    fn gatherOnCs(m: *Model, w: usize, k: usize, src: u64, dst: u64) !void {
        const ch = if (w == 0) &m.ch2 else &m.ch;
        try m.wev[w][k].record(m.stream);
        try m.cs.?.wait(m.wev[w][k]);
        const rs = try prompt.gather(&m.ec, ch, src, dst);
        if (k == 0) ch.ga_rs = rs else ch.gm_rs = rs;
        try m.wev[w][k + 1].record(m.cs.?);
    }

    /// The stream waits for wave w's MoE gather (on cs) before anything reads its gm.
    fn waitGm(m: *Model, posted: *[2]bool, w: usize) !void {
        if (!posted[w]) return;
        try m.stream.wait(m.wev[w][3]);
        posted[w] = false;
    }

    /// The prompt's logits (its last row's, the ranks' columns in rank order) after a fill's last chunk.
    pub fn promptLogits(m: *const Model) u64 {
        return m.ch.head_g;
    }

    /// One round over every stream's window (rows.windows, each with its stream's slot and extent), then each window
    /// whose stream drafts has its rows' taps absorbed into its slot's drafter rings at its positions (the served
    /// eager absorb: rows past the ones the round keeps sit where the drafter never reads, and the next window
    /// overwrites them). absorb[i]: window i's stream drafts.
    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        const e = &m.eng;
        if (m.prof) try m.synchronize(); // (the GPU idle, so the issue time shows whether the host keeps it fed)
        const t0 = m.now();
        try round.forward(e, &m.rd, &m.ch, &m.caches, m.rings, if (m.eh) |*x| x else null, &.{}, rows, m.pool_cap, null, if (m.graphs) |*g| g else null);
        if (m.prof) {
            m.t_enqueue = m.now() - t0;
            try m.synchronize();
            m.t_forward = m.now() - t0;
            if (m.timer) |t| try t.finish();
            if (m.rd.clk != 0) {
                var h: [1 + 2 * round.Round.max_clocks]u64 = undefined;
                const d = m.ctx.d;
                try d.check(d.api.cuMemcpyDtoH_v2(&h, m.rd.clk, h.len * 8), "cuMemcpyDtoH");
                const n = @min(h[0], round.Round.max_clocks);
                for (1..n) |i| {
                    const tag = h[2 + 2 * i] & 0xff;
                    m.clk_ph[tag] += h[1 + 2 * i] -% h[1 + 2 * (i - 1)];
                    m.clk_cnt[tag] += 1;
                }
                m.clk_rounds += 1;
            }
        }
        const t1 = m.now();
        if (m.dr) |*dr| {
            const wins = rows.windows orelse return error.BadRound;
            var items: [max_streams]draft.Item = undefined;
            var n: usize = 0;
            for (wins, absorb) |win, yes| {
                if (!yes) continue;
                items[n] = .{ .slot = @intCast(rows.slots.?[win.row]), .taps = m.rd.taps + win.row * dr.taps_w * 2, .n = win.n, .start = @intCast(rows.pos[win.row]) };
                n += 1;
            }
            if (n > 0) try dr.absorbMany(e, &m.ch, &m.dpool.?, items[0..n]);
        }
        try m.synchronize();
        if (m.prof) m.t_absorb = m.now() - t1;
    }

    /// The rounds' GPU clocks (a profile): the forward's start and each stretch's ends, for its gaps.
    pub fn useClock(m: *Model, marks: u8) !void {
        m.rd.clk_marks = marks == 1 or marks == 3;
        // TF_DS_CLOCK_RATIO: the compress ratio whose layers mode 3 times (2 unless set)
        const r: u8 = if (std.c.getenv("TF_DS_CLOCK_RATIO")) |v| std.fmt.parseInt(u8, std.mem.span(v), 10) catch 2 else 2;
        m.rd.clk_ratio = if (marks == 3) r else null;
        m.rd.clk_layers = marks == 2 or marks == 3;
        if (m.rd.clk == 0) m.rd.clk = try m.arena.take((1 + 2 * round.Round.max_clocks) * 8);
    }

    /// The rounds' GPU time by phase (eager rounds; --profile 2): a timer on the round.
    pub fn usePhaseTimer(m: *Model) !void {
        if (m.timer != null) return;
        const t = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(t);
        t.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.timer = t;
        const pt = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(pt);
        pt.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.ptimer = pt;
        m.rd.timer = t;
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);

test "this rank's weight digest comes from TF_DS_RANK_CACHE_SHA; unset, empty or not a digest, none (and the start goes on)" {
    const env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    const name = "TF_DS_RANK_CACHE_SHA";
    if (std.c.getenv(name) != null) return error.SkipZigTest; // the caller's own setting: left alone
    defer _ = env.unsetenv(name);
    try std.testing.expectEqual(prompt2d.no_digest, rankDigest());
    const hex = "c0ffee00" ++ "11223344" ++ "55667788" ++ "99aabbcc" ++ "ddeeff00" ++ "01234567" ++ "89abcdef" ++ "fedcba98";
    var want: prompt2d.Digest = undefined;
    _ = try std.fmt.hexToBytes(&want, hex);
    for ([_][:0]const u8{ hex, "sha256:" ++ hex }) |v| {
        _ = env.setenv(name, v, 1);
        try std.testing.expectEqual(want, rankDigest());
    }
    for ([_][:0]const u8{ "", hex[1..], "x" ++ hex, "sha1:" ++ hex[24..] }) |v| {
        _ = env.setenv(name, v, 1);
        try std.testing.expectEqual(prompt2d.no_digest, rankDigest());
    }
}

test "the start-up settings carry this rank's options and the math switches it runs" {
    const o: Options = .{ .model_dir = "", .cache_dir = "", .kit_dir = "", .rank = 2, .world = 4, .master = .{ 10, 255, 0, 33 }, .port = 29600, .pool = 1 << 24, .drafts = true, .rdma_devices = "mlx5_0,mlx5_2", .graphs = true, .graph_cap = 1 << 40, .round_rows = 48, .streams = 16 };
    const was = .{ prompt.replay_floor, tri_index.idx_rb };
    defer {
        prompt.replay_floor = was[0];
        tri_index.idx_rb = was[1];
    }
    prompt.replay_floor = false;
    tri_index.idx_rb = 1;
    const off = startSettings(o);
    try std.testing.expectEqual(prompt2d.Settings{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = std.math.maxInt(u32), .pool = 1 << 24, .replay_floor = 0, .idx_rb = 1, .pass_rows = 2048, .pass_from = 8192 }, off);
    // the pass settings as Model.open read them
    const passes = .{ prompt2d.pass_rows_asked, prompt2d.pass_from_asked };
    defer {
        prompt2d.pass_rows_asked = passes[0];
        prompt2d.pass_from_asked = passes[1];
    }
    prompt2d.pass_rows_asked = 4096;
    prompt2d.pass_from_asked = 16384;
    try std.testing.expectEqual(@as(u32, 4096), startSettings(o).pass_rows);
    try std.testing.expectEqual(@as(u32, 16384), startSettings(o).pass_from);
    prompt2d.pass_rows_asked = 2048;
    prompt2d.pass_from_asked = 8192;
    prompt.replay_floor = true;
    tri_index.idx_rb = 2;
    const on = startSettings(o);
    try std.testing.expectEqual(@as(u32, 1), on.replay_floor);
    try std.testing.expectEqual(@as(u32, 2), on.idx_rb);
    tri_index.idx_rb = 1 << 40; // a count past u32: the largest u32, never a wrap to an agreeing value
    try std.testing.expectEqual(@as(u32, std.math.maxInt(u32)), startSettings(o).idx_rb);
}

    pub fn roundLogits(m: *Model, R: usize) ![]const f32 {
        const d = m.ctx.d;
        try m.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.rd.logits, R * m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0 .. R * m.vocab];

test "TF_DS_RING_WAIT's fallback to the proxy threads is for the default only: unset without the kernels or libmlx5 says why, 1 is left to Ring.create to refuse by name" {
    try std.testing.expect(Model.ringWaitFallback(false, true, true, null) == null);
    try std.testing.expect(std.mem.indexOf(u8, Model.ringWaitFallback(false, false, true, null).?, "stage_wait") != null);
    try std.testing.expect(std.mem.indexOf(u8, Model.ringWaitFallback(false, true, false, null).?, "libmlx5") != null);
    try std.testing.expect(Model.ringWaitFallback(true, false, false, null) == null);
    // a card that cannot do the atomic signal: unset falls back with the card's own reason, 1 is left to refuse by name
    try std.testing.expectEqualStrings("device mlx5_0 reports atomic_cap 0", Model.ringWaitFallback(false, true, true, "device mlx5_0 reports atomic_cap 0").?);
    try std.testing.expect(Model.ringWaitFallback(true, true, true, "device mlx5_0 reports atomic_cap 0") == null);
    // the parse: unset is on, and the start lines name the mode
    const src = @embedFile("model.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "prompt2d.ring_wait_asked = prompt2d.switchDefaultOn(ring_wait_text);") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "unset (the default)") != null);
}
    }

    /// The prompt's logits on the host (after a fill's last chunk).
    pub fn promptLogitsHost(m: *Model) ![]const f32 {
        const d = m.ctx.d;
        try m.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.ch.head_g, m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0..m.vocab];

test "TF_DS_ROCI_XLOG's files are written for the rings a run has: the 2D rings' (Model.rdma_rings), not only the single ring a TP4 run never has; and by a follower between its frames" {
    const code = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, code, "pub fn flushDiag(").?;
    const body = code[at..std.mem.indexOfPos(u8, code, at, "\n    }\n").?];
    try std.testing.expect(std.mem.indexOf(u8, body, "m.rdma_rings") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "rs.all4, rs.col, rs.pair") != null);
    const lanes = @embedFile("lanes.zig");
    const f = std.mem.indexOf(u8, lanes, "pub fn follow(gpa").?;
    const loop = lanes[f..std.mem.indexOfPos(u8, lanes, f, "\n}\n").?];
    try std.testing.expect(std.mem.indexOf(u8, loop, "defer m.flushDiag(false);") != null);
}
    }

    /// The drafter's batched pass for N streams: each one's pending token at position q0 in its slot (its rings hold
    /// its positions below q0), `steps` drafts each (the drafter's drafts and confidences on the host).
    pub fn pass(m: *Model, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize) !void {
        if (m.dr == null) return error.NoDrafter;
        const dr = &m.dr.?;
        const pool = &m.dpool.?;
        for (q0, slots) |q, s| pool.absorbed[@intCast(s)] = @intCast(q);
        try dr.pass(&m.eng, &m.ch, pool, tokens, q0, slots, steps, null);
    const kv = std.mem.indexOf(u8, kb, "speed_defaults.verdict(why, speed_defaults.defaulted(\"TF_DS_ENGRAM_KEEP_AHEAD\")) == .fallback").?;
    const kerr = std.mem.indexOf(u8, kb[kv..], "return error.KeepAheadSharesReader;").?;
    try std.testing.expect(passRowsFallback("2048", true, 64 * gib) == null);
    try std.testing.expect(passRowsFallback(null, true, 64 * gib) == null);
}

    try std.testing.expect(std.mem.indexOf(u8, src, "set TF_DS_DRAFT_MAIN_4WAY=0 on every Spark") != null);
}
    }
};
