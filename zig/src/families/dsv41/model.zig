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
const roci_pair = @import("roci_pair.zig");
const pair_spans = @import("pair_spans.zig");
const pass_meter = @import("pass_meter.zig");
const pair_nap = @import("rdma_nap.zig");
const any_check = @import("any_check.zig");
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

/// A kernel set's answer for a start check: whether a launch of these arguments finds its variant, as a launch looks it
/// up (aot.Set.find).
const SetFinds = struct {
    set: *const cuda.aot.Set,

    pub fn has(f: SetFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        _ = f.set.find(name, args, consts) catch return false;
        return true;
    }

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
};

/// The Spark-class box's memory: a default of 4,096-row passes needs about 1.5 GiB more a node (measured) and, with the stagger, the
/// guard below; a box with less than this much (CUDA's total) keeps 2,048-row passes. The default only: a setting is the operator's.
const pass_rows_default_floor: usize = 96 << 30;

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
    if (total_bytes >= pass_rows_default_floor) return null;
    return "this box's memory is under 96 GiB, too little for the larger pass's buffers";
}

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
    if (passRowsFallback(v, speed_defaults.defaulted("TF_DS_PASS_ROWS"), total_bytes)) |why| {

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

/// TF_DS_QGATE_CANARY's canary (prompt2d.qgateAsked); text that is no canary runs none, said once.
fn readQgate() prompt2d.Qgate {
    const v: ?[]const u8 = if (std.c.getenv("TF_DS_QGATE_CANARY")) |x| std.mem.span(x) else null;
    const q = prompt2d.qgateAsked(v);
    if (q == .off) if (v) |t| if (t.len != 0 and !std.mem.eql(u8, t, "0"))
        std.log.info("TF_DS_QGATE_CANARY={s}: not a canary (1 the head, 2 or 3 the body): none", .{t});
    return q;
}

/// TF_DS_QGATE_CANARY `asked` (any canary on every rank alike: the start-up Settings' kernel bit) the same canary on
/// every rank, else the start is refused, the ranks named. No collective when none is asked.
fn agreeQgateCanary(d: *const cuda.Driver, comm: *const comm_mod.Comm, stream: cuda.Stream, asked: prompt2d.Qgate) !void {
    if (asked == .off) return;
    const differ = try prompt2d.ranksDiffer(d, comm, stream, @backingInt(asked));
    if (differ == 0) return;
    var buf: [64]u8 = undefined;
    std.log.err("TF_DS_QGATE_CANARY={d}: {s} asked another canary than rank 0: the start is refused", .{ @backingInt(asked), prompt2d.rankList(differ, &buf) });
    return error.CanariesDiffer;
}

/// TF_DS_QGATE_CANARY `asked` in effect with the checkpoint's own kernels (`native`), one line when asked: the head's
/// canary (prompt.qgate_canary: every row's logits the head's on rows one more FP8 step away, a quality gate's ruler) or
/// the body's (prompt.qgate_body: layer 0's MoE partials perturbed before their gather, a quality gate's floor), or off
/// and why. Never a serving setting.
fn useQgateCanary(asked: prompt2d.Qgate, native: bool) void {
    prompt.qgate_body = .off;
    if (asked == .off) return;
    if (!native) return std.log.info("TF_DS_QGATE_CANARY={d}: off: needs the checkpoint's own FP8 / MXFP4 kernels (act_quant)", .{@backingInt(asked)});
    switch (asked) {
        .off => {},
        .head => {
        },
        .body_fp8 => {
            prompt.qgate_body = .body_fp8;
            std.log.info("TF_DS_QGATE_CANARY=2: layer 0's MoE partials take one more FP8 step (act_quant's quant-dequant) before their gather, in every prompt chunk and every round: the body perturbed on purpose, a quality gate's floor, not a serving setting", .{});
        },
        .body_ulp => {
            prompt.qgate_body = .body_ulp;
            std.log.info("TF_DS_QGATE_CANARY=3: layer 0's MoE partials move one fp32 ulp away from zero (a reduction order's size) before their gather, in every prompt chunk and every round: the body perturbed on purpose, a quality gate's floor, not a serving setting", .{});
        },
    }
/// Why the decode exchanges cannot all ride the rings on this rank (null: they can): the rings are open and every decode exchange fits
/// them (prompt2d.decodeOff).
pub fn decodeOff(rings: bool, shape: prompt2d.DecodeShape) ?[]const u8 {
    if (!rings) return "needs the rails' devices (TF_DS_RDMA_DEVICES): the decode rings";
    return prompt2d.decodeOff(shape);
}

/// TF_DS_ROCI_DECODE's start line: what rides where.
fn decodeLine(rank: u32, order: []const u8) []const u8 {
    const S = struct {
        var buf: [512]u8 = undefined;
    };
    return std.fmt.bufPrint(&S.buf, "Rocinante decode: every decode round and drafter pass exchange rides the RDMA rings (the drafter's per-step bests over the four nodes' ring in rank order {s}); a ring that cannot take an exchange ends it by name; on every rank (this one: {d})", .{ order, rank }) catch S.buf[0..0];
}

fn rankOrderText(order: *const [4]u32) []const u8 {
    const S = struct {
        var buf: [16]u8 = undefined;
    };
    return std.fmt.bufPrint(&S.buf, "{d},{d},{d},{d}", .{ order[0], order[1], order[2], order[3] }) catch S.buf[0..0];
}

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
        .exch = prompt2d.exchAsked(),
    };
}

const EngramPhase = enum { all, pre, finish };

const pair_guard_bytes: usize = 6 << 30;

    return pairShortN(free, pass_rows, lane_arena, 2);
}

/// `pairShort` for `lanes` passes in flight (TF_DS_ROCI_STAGGER_LANES): every pass after the first takes the estimate.
pub fn pairShortN(free: u64, pass_rows: usize, lane_arena: u64, lanes: usize) u64 {
    const need = (lanes - 1) * pairEstimate(pass_rows, lane_arena) + pair_guard_bytes;
/// What TF_DS_ROCI_STAGGER_LANES comes to at the start: the lanes asked for (`want`, 2 or 3), whether the operator set it, and whether
/// every rank has the memory for them. A box under the floor refuses a lane count the operator set, by name; one that was only the
/// default runs two lanes and warns. Two lanes are never judged here: the pair's own guard runs as it always did.
pub const LanesChoice = struct { lanes: usize, warn: bool = false, refuse: bool = false };
pub fn lanesChoice(want: usize, set: bool, fits_all: bool) LanesChoice {
    if (want <= 2 or fits_all) return .{ .lanes = want };
    if (set) return .{ .lanes = want, .refuse = true };
    return .{ .lanes = 2, .warn = true };
}

fn pairShortText(buf: []u8, free: u64, pass_rows: usize, lane_arena: u64, v: PoolView, lanes: usize) ![]const u8 {
    const what = if (lanes >= 3) "the second and third passes (TF_DS_ROCI_STAGGER_LANES=3)" else "the second pass";
    return memoryRefusal(buf, what, true, pass_rows, (lanes - 1) * pairEstimate(pass_rows, lane_arena) + pair_guard_bytes, free, v);
        why = "needs the four-node 2D split with Rocinante";
        switch (speed_defaults.verdict(why orelse "another rank cannot run it", speed_defaults.defaulted("TF_DS_ENGRAM_PARTNER"))) {
    const rc = try roci_engine.Roci.openLane(gpa, d, m.comm.rank, node, m.comm.world, m.two.?.rank_of, &m.fan, rails, Model.rdmaGid() orelse 3, 1, m.two.?.rc.?, .host);
/// TF_DS_ROCI_STAGGER's start: two whole passes in flight need every exchange of a full pass on the transport (so no collective of
/// another communicator runs beside a stream's memop waits), the Engram uploads from pinned memory (a pageable copy blocks the host thread
/// that issues both passes), a second chunk, node of the split, lane, stream, cuBLAS handle and the per-layer cache events. Every rank's
/// checks agree before anything is allocated; a rank that cannot run the route ends the start naming its reason.
fn openPair(m: *Model, gpa: std.mem.Allocator, d: *const cuda.Driver, o: Options, sp: plan.Split, c: *const Config) !void {
    var why: ?[]const u8 = null;
    var free: usize = 0;
    const two = if (m.two) |*t| t else null;
    const connections: ?usize = if (std.c.getenv("CUDA_DEVICE_MAX_CONNECTIONS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch null else null;
    if (sp.pair == null or two == null) {
        why = "needs the four-node 2D split";
    } else if (c.format != .native) {
        why = "needs the checkpoint's own FP8 / MXFP4 weights";
    } else if (m.pass_rows % chunk_rows != 0 or m.pass_rows > pass_rows_max) {
    } else if (m.pass_rows < 2 * chunk_rows) {
    } else if (connections == null or connections.? < 16) {
        why = "needs CUDA_DEVICE_MAX_CONNECTIONS=16 (the second pass's stream would share the first's queue, and a pending wait on it would block the other)";
    } else if (!pinned) {
        why = "needs TF_DS_ENGRAM_PINNED=1 (a pageable Engram upload blocks the host thread until the GPU reaches it, so the second pass could not be queued ahead)";
    } else if (m.eh == null or !wk_on) {
        why = "needs the Engram worker with its four chunk slots";
    }
    if (why == null and o.rdma_devices == null) why = "needs the rails' devices (TF_DS_RDMA_DEVICES)";
    var lanes: usize = prompt2d.roci_stagger_lanes;
    if (why == null) {
        free = (try m.ctx.memInfo()).free;
        if (lanes >= 3) {
            // the third lane's bytes count: every rank votes on the memory for three, and a box under the floor either refuses (set) or keeps two (default)
            const fits = pairShortN(free, m.pass_rows, lane_arena, 3) == 0;
            const choice = lanesChoice(lanes, prompt2d.roci_stagger_lanes_set, try m.eng.agreeAll(fits));
            if (choice.warn) std.log.warn("TF_DS_ROCI_STAGGER_LANES: rank {d}: three lanes need {d:.2} GiB free and a rank has less: two passes in flight", .{ m.comm.rank, @as(f64, @floatFromInt(2 * pairEstimate(m.pass_rows, lane_arena) + pair_guard_bytes)) / (1 << 30) });
            if (choice.refuse and !fits) why = try pairShortText(&why_buf, free, m.pass_rows, lane_arena, poolView(m, c), 3) else if (choice.refuse) why = "TF_DS_ROCI_STAGGER_LANES=3: another rank has too little free memory for the third lane";
            lanes = choice.lanes;
        }
        if (why == null and pairShort(free, m.pass_rows, lane_arena) != 0) why = try pairShortText(&why_buf, free, m.pass_rows, lane_arena, poolView(m, c), 2);
    }
    if (!try m.eng.agreeAll(why == null)) {
        const is_default = speed_defaults.defaulted("TF_DS_ROCI_STAGGER");
        switch (speed_defaults.verdict(why orelse "another rank cannot run it", is_default)) {
                if (why) |w| std.log.err("TF_DS_ROCI_STAGGER=1: off: {s}", .{w}) else std.log.err("TF_DS_ROCI_STAGGER=1: off: another rank cannot run it (its line names the reason)", .{});
                return error.StaggerRefused;
            },
                std.log.warn("TF_DS_ROCI_STAGGER: off: {s}; without it: {s} (the engine runs one pass at a time; set TF_DS_ROCI_STAGGER=1 to make this a refusal, 0 to silence it)", .{ why orelse "another rank cannot run it (its line names the reason)", speed_defaults.costOf("TF_DS_ROCI_STAGGER") });
                prompt2d.roci_stagger_asked = false;
                return;
            },
        }
    }
    const used = m.arena.used;
    m.two_b = try prompt2d.Two.init(c, &m.w, &m.arena, sp, m.pass_rows);
    m.two_b.?.adoptSettings(&m.two.?);
    var devs_it = std.mem.splitScalar(u8, o.rdma_devices.?, ',');
    const rails: [2][]const u8 = .{ devs_it.next() orelse return error.RociRails, devs_it.next() orelse return error.RociRails };
    m.two_b.?.rc = try roci_engine.Roci.openLane(gpa, d, m.comm.rank, @intCast(m.two.?.node()), m.comm.world, m.two.?.rank_of, &m.fan, rails, Model.rdmaGid() orelse 3, 2, m.two.?.rc.?, .prompt);
    errdefer m.two_b.?.rc.?.close();
    m.stream_b = if (m.pair_prio != .equal) blk: {
        const range = try cuda.Stream.priorityRange(d);
        break :blk try cuda.Stream.initPriority(d, roci_pair.prioritiesOf(m.pair_prio, range.least, range.greatest)[1]);
    } else try cuda.Stream.init(d, true);
    errdefer if (m.stream_b) |*x| x.deinit();
    m.usePass(.b);
    defer m.usePass(.a);
    m.ch_b = try prompt.Chunk.init(&m.eng, &m.arena, m.pass_rows, m.ctx_cap);
    m.ch_b.?.lane = 1;
    m.blas_b = try cublas.Blas.open(m.stream_b.?, m.ch_b.?.blas_ws);
    errdefer if (m.blas_b) |*b| b.close();
    if (lanes >= 3) {
        // the third lane: the same resources as the second's, the transport on the third part of the first lane's block, the three ring-linked
        m.two_c = try prompt2d.Two.init(c, &m.w, &m.arena, sp, m.pass_rows);
        m.two_c.?.adoptSettings(&m.two.?);
        m.two_c.?.rc = try roci_engine.Roci.openLane(gpa, d, m.comm.rank, @intCast(m.two.?.node()), m.comm.world, m.two.?.rank_of, &m.fan, rails, Model.rdmaGid() orelse 3, 3, m.two.?.rc.?, .prompt);
        errdefer m.two_c.?.rc.?.close();
        rocinante_mod.Transport.ring3(m.two.?.rc.?.t, m.two_b.?.rc.?.t, m.two_c.?.rc.?.t);
        m.stream_c = if (m.pair_prio != .equal) blk: {
            const range = try cuda.Stream.priorityRange(d);
            break :blk try cuda.Stream.initPriority(d, roci_pair.prioritiesOf(m.pair_prio, range.least, range.greatest)[1]);
        } else try cuda.Stream.init(d, true);
        errdefer if (m.stream_c) |*x| x.deinit();
        m.usePass(.c);
        m.ch_c = try prompt.Chunk.init(&m.eng, &m.arena, m.pass_rows, m.ctx_cap);
        m.ch_c.?.lane = 2;
        m.blas_c = try cublas.Blas.open(m.stream_c.?, m.ch_c.?.blas_ws);
        errdefer if (m.blas_c) |*b| b.close();
        m.ids64_c = try m.host.allocator().alloc(i64, m.pass_rows);
        m.host_pos_c = try m.host.allocator().alloc(i64, m.pass_rows);
        m.usePass(.b);
    }
    m.pair_lanes = lanes;
    m.pair_ev = try roci_pair.Events.init(d, c.layers / 2, lanes);
    errdefer if (m.pair_ev) |*e| e.deinit();
    m.pair_join = try cuda.Event.init(d, true);
    errdefer if (m.pair_join) |*e| e.deinit();
    m.pair_t0 = try cuda.Event.init(d, true);
    errdefer if (m.pair_t0) |*e| e.deinit();
    m.pair_log = std.c.getenv("TF_DS_ROCI_PAIR_LOG") != null;
    if (m.pair_log) {
        // the span log's events: a pair of timing events per outstanding pair (pair_spans.cap of them), made here and only here
        m.span_t0 = try m.gpa.alloc(cuda.Event, pair_spans.cap);
        m.span_join = try m.gpa.alloc(cuda.Event, pair_spans.cap);
        for (0..pair_spans.cap) |i| {
            m.span_t0[i] = try cuda.Event.init(d, true);
            m.span_join[i] = try cuda.Event.init(d, true);
        }
    }
    m.pair_major = roci_pair.pairMajorOf(if (std.c.getenv("TF_DS_ROCI_WORKER_PAIR_MAJOR")) |v| std.mem.span(v) else null);
    if (m.eh) |*eh| eh.wk_group = if (m.pair_major) roci_pair.groupChunksN(lanes, m.pass_rows, chunk_rows) else 0;
    m.pair_gap = roci_pair.gapOf(if (std.c.getenv("TF_DS_ROCI_STAGGER_GAP")) |v| std.mem.span(v) else null);
    m.ids64_b = try m.host.allocator().alloc(i64, m.pass_rows);
    m.host_pos_b = try m.host.allocator().alloc(i64, m.pass_rows);
    try m.stream_b.?.synchronize();
    if (m.stream_c) |sc| try sc.synchronize();
    m.pair_bytes = m.arena.used - used;
    m.pair = true;
    const after = m.ctx.memInfo() catch cuda.Context.MemInfo{ .free = free, .total = 0 };
    const lay = @import("rocinante.zig").layout();
    std.log.info("TF_DS_ROCI_STAGGER=1: rank {d}: {d} whole passes of {d} rows in flight (TF_DS_ROCI_STAGGER_LANES), the later's layer L after the earlier's layer L caches, issue gap {d} layers (TF_DS_ROCI_STAGGER_GAP), stream priorities A {d} and B {d} (TF_DS_ROCI_STAGGER_PRIO={d}; a lower number is more urgent), the Engram worker's order {s} (TF_DS_ROCI_WORKER_PAIR_MAJOR); the second pass's chunk and node take {d} bytes of the device arena, its transport's arena {d} bytes and control block {d} bytes; {d} bytes free before, {d} after; the Engram worker's ring holds {d} chunk slots of {d} rows, a pass takes {d}; the KV pool holds {d} positions ({d:.2} GiB on this Spark, POOL) and the floor kept free is {d:.1} GiB", .{ m.comm.rank, m.pair_lanes, m.pass_rows, m.pair_gap, m.stream.priority() catch -99, m.stream_b.?.priority() catch -99, @backingInt(m.pair_prio), if (m.pair_major) "pair-major (both passes' layer 0, then both layer 1)" else "pass-major (each pass's layers in turn)", m.pair_bytes, lay.total, @sizeOf(@import("rocinante.zig").Control), free, after.free, m.workerSlots(), chunk_rows, m.pass_rows / chunk_rows, m.pool_cap, @as(f64, @floatFromInt(poolView(m, c).bytes(m.pool_cap))) / (1 << 30), @as(f64, @floatFromInt(pair_guard_bytes)) / (1 << 30) });
}

    return where.report(buf);
/// One line of TF_DS_ROCI_PAIR_LOG: to the log, and appended to the file the variable names when its value is a path.
fn pairLog(comptime fmt: []const u8, args: anytype) void {
    std.log.info(fmt, args);
    const v = std.c.getenv("TF_DS_ROCI_PAIR_LOG") orelse return;
    if (v[0] != '/') return;
    const line = std.fmt.bufPrint(&buf, fmt ++ "\n", args) catch return;
    const fd = std.c.open(v, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    _ = std.c.write(fd, line.ptr, line.len);
    _ = std.c.close(fd);
}

    vfan: prompt2d.Fan, // the host collectives' links (prompt2d.hostLinksAsked), else none
    watch: link.Watch, // watchPeers: a peer rank that stops stops this one too
    star_busy: std.atomic.Value(bool), // star.Star's guard against two threads in one collective
    comm: comm_mod.Comm,
    rdma_mod: ?cuda.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*rdma.Ring = null,
    set: cuda.aot.Set,
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
    // TF_DS_ROCI_STAGGER: the second lane of a pair of passes in flight: its node of the 2D split (own scratch, own transport), chunk,
    // stream, cuBLAS handle, the per-layer cache events and its host id and position arrays.
    pair: bool = false,
    pair_open: std.atomic.Value(bool) = .init(false), // a pair's second pass may still run on stream_b (a group's last on stream_c): the first stream joins it before any other work
    pair_last: usize = 2, // the passes of the group last queued (its last pass's events are what the next group's first waits on)
    pair_lanes: usize = 2, // the passes in flight (TF_DS_ROCI_STAGGER_LANES: 2, or 3 with the third lane below)
    two_b: ?prompt2d.Two = null,
    ch_b: ?prompt.Chunk = null,
    stream_b: ?cuda.Stream = null,
    blas_b: ?cublas.Blas = null,
    two_c: ?prompt2d.Two = null, // the third lane of a group of three (TF_DS_ROCI_STAGGER_LANES=3): the same resources as the second's
    ch_c: ?prompt.Chunk = null,
    stream_c: ?cuda.Stream = null,
    blas_c: ?cublas.Blas = null,
    pair_ev: ?roci_pair.Events = null,
    pair_join: ?cuda.Event = null,
    span_ring: pair_spans.Ring = .{}, // TF_DS_ROCI_PAIR_LOG: the pairs whose span is not printed yet
    span_t0: []cuda.Event = &.{},
    span_join: []cuda.Event = &.{},
    span_slot: ?usize = null, // the slot of the pair being issued
    pair_t0: ?cuda.Event = null, // timing events: the pair's first queued work on stream A, and its second pass's last (pair_join)
    pair_log: bool = false, // TF_DS_ROCI_PAIR_LOG: one line a pair (a value starting with / also appends them to that file on rank 0)
    pair_gap: usize = roci_pair.gap,
    pair_prio: roci_pair.Prio = .equal, // TF_DS_ROCI_STAGGER_PRIO
    pair_major: bool = false, // TF_DS_ROCI_WORKER_PAIR_MAJOR
    fill_t0: u64 = 0, // the fill's admission (fillBegin), for the pair log's first line of a prompt
    pair_in_fill: u64 = 0, // pairs run since fillBegin
    pair_count: u64 = 0,
    pair_prev_pending: bool = false, // the last pair's GPU span has not been printed
    meter: pass_meter.Meter = .{}, // the rows the device has finished of each slot's prompt pass (the lane host's live prefill speed)
    ids64_b: []i64 = &.{},
    host_pos_b: []i64 = &.{},
    ids64_c: []i64 = &.{},
    host_pos_c: []i64 = &.{},
    pair_bytes: usize = 0,
    rdma_rings: ?ring2d.Rings = null,
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
        prompt2d.roci_p2_first_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ROCI_P2_FIRST")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.roci_engram_ahead_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ROCI_ENGRAM_AHEAD")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.roci_sent_first_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ROCI_SENT_FIRST")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.roci_combine_door_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ROCI_COMBINE_DOOR")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.roci_stagger_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ROCI_STAGGER")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        {
            const text: ?[]const u8 = if (std.c.getenv("TF_DS_ROCI_STAGGER_LANES")) |v| std.mem.span(v) else null;
            prompt2d.roci_stagger_lanes = roci_pair.lanesOf(text) orelse {
                std.log.err("TF_DS_ROCI_STAGGER_LANES is 2 or 3 (the passes in flight), not \"{s}\"", .{text.?});
                return error.StaggerRefused;
            };
            prompt2d.roci_stagger_lanes_set = text != null;
            if (prompt2d.roci_stagger_lanes == 3 and !prompt2d.roci_stagger_asked) {
                std.log.err("TF_DS_ROCI_STAGGER_LANES=3 needs TF_DS_ROCI_STAGGER=1 (it is the stagger's third lane)", .{});
                return error.StaggerRefused;
            }
            prompt2d.roci_stagger3_asked = prompt2d.roci_stagger_lanes == 3; // every rank alike (Settings)
        }
        rocinante_mod.setCap(prompt2d.pass_rows_asked); // (the transport's arenas and plans follow the pass rows, before any layout is read)
        prompt2d.ring_wc_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_RING_WC")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.ring_inplace_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_RING_INPLACE")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        const ring_word_text: ?[]const u8 = if (std.c.getenv("TF_DS_RING_WORD")) |v| std.mem.span(v) else null;
        prompt2d.ring_word_asked = prompt2d.switchDefaultOn(ring_word_text); // on unless set to something but 1; every rank alike (Settings)
        prompt2d.ring_word_explicit = ring_word_text != null;
        if (!prompt2d.ring_word_asked) std.log.info("TF_DS_RING_WORD: off (set to something but 1): each lane's arrival is the proxy's flag, not a generation word", .{});
        const ring_sig_text: ?[]const u8 = if (std.c.getenv("TF_DS_RING_WORD_SIGNAL")) |v| std.mem.span(v) else null;
        var ring_sig: ?rdma.Roci = ringWordSignal(ring_sig_text);
        if (prompt2d.ring_word_asked and ring_sig == null) {
            std.log.info("TF_DS_RING_WORD: off: TF_DS_RING_WORD_SIGNAL={s} is neither word nor atomic", .{ring_sig_text.?});
            prompt2d.ring_word_asked = false;
        }
        if (prompt2d.ring_word_asked and ring_sig == .atomic and o.rdma_devices != null) {
            // the signal is the atomic (the default): the cards must do ATOMICs. Unset falls back to the word with a warning; set outright refuses.
            var names: [8][]const u8 = undefined;
            var nn: usize = 0;
            var it = std.mem.splitScalar(u8, o.rdma_devices.?, ',');
            while (it.next()) |x| if (nn < names.len) {
                names[nn] = std.mem.trim(u8, x, " ");
                nn += 1;
            };
            var ab: [300]u8 = undefined;
            const sm = ringWordSignalMode(ring_sig_text, rdma.probeAtomic(&ab, names[0..nn])).?;
            if (sm.refuse) {
                std.log.err("TF_DS_RING_WORD_SIGNAL=atomic: {s}: set TF_DS_RING_WORD_SIGNAL=word for the inline word", .{sm.why.?});
                return error.AtomicUnsupported;
            }
            if (sm.warn) std.log.warn("TF_DS_RING_WORD_SIGNAL: unset (the default is the atomic) but {s}: the rings' arrival word is the inline WRITE", .{sm.why.?});
            ring_sig = sm.roci;
        }
        prompt2d.ring_word_atomic_asked = prompt2d.ring_word_asked and ring_sig == .atomic; // every rank alike (Settings)
        const ring_wait_text: ?[]const u8 = if (std.c.getenv("TF_DS_RING_WAIT")) |v| std.mem.span(v) else null;
        prompt2d.ring_wait_asked = prompt2d.switchDefaultOn(ring_wait_text); // on unless set to something but 1; every rank alike (Settings)
        if (!prompt2d.ring_wait_asked) std.log.info("TF_DS_RING_WAIT: off (set to something but 1): the rings run their proxy threads", .{});
        if (prompt2d.ring_wait_asked and !prompt2d.ring_word_asked) {
            // the chains end in the generation word's signal (the inline WRITE, or the atomic when TF_DS_RING_WORD_SIGNAL asks for it): without
            // TF_DS_RING_WORD the card's WAIT cannot run, so the proxies do
            std.log.info("TF_DS_RING_WAIT: {s}: off: it needs TF_DS_RING_WORD=1 (each chain ends in the generation word's signal), so the rings run their proxy threads", .{if (ring_wait_text == null) "unset (the default is the card's WAIT)" else "1"});
            prompt2d.ring_wait_asked = false;
        }
        prompt2d.ring_wait_explicit = ring_wait_text != null;
        prompt2d.qgate_canary_asked = readQgate(); // every rank alike (Settings: any canary; the canary itself, agreeQgateCanary)
        if (std.c.getenv("TF_DS_ROCI_XLOG")) |v| rdma.xlog = std.mem.eql(u8, std.mem.span(v), "1");
        if (std.c.getenv("TF_DS_L2_DN_MB")) |v| round.l2_dn_mb = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_dn_mb;
        if (std.c.getenv("TF_DS_L2_DEC_MB")) |v| round.l2_dec_mb = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_dec_mb;
        ops_mod.Ops.add2_vec_on = prompt2d.switchDefaultOn(if (std.c.getenv("TF_DS_ADD2_VEC")) |v| std.mem.span(v) else null); // the bytes are the scalar kernel's: a rank may differ
        if (std.c.getenv("TF_EXL3_L2_DELAY_NS")) |v| ops_mod.l2_delay_ns = std.fmt.parseInt(c_longlong, std.mem.span(v), 10) catch ops_mod.l2_delay_ns;
        if (std.c.getenv("TF_EXL3_L2_RATE_GBPS")) |v| ops_mod.l2_rate_gbps = std.fmt.parseFloat(f64, std.mem.span(v)) catch ops_mod.l2_rate_gbps;
        m.graph_fallbacks = .init(0);
        m.pair = false;
        m.pair_open = .init(false);
        m.pair_lanes = 2;
        m.pair_last = 2;
        m.two_b = null;
        m.ch_b = null;
        m.stream_b = null;
        m.blas_b = null;
        m.two_c = null;
        m.ch_c = null;
        m.stream_c = null;
        m.blas_c = null;
        m.pair_ev = null;
        m.pair_join = null;
        m.pair_t0 = null;
        m.span_ring = .{};
        m.span_t0 = &.{};
        m.span_join = &.{};
        m.span_slot = null;
        m.pair_log = false;
        m.pair_gap = roci_pair.gap;
        m.pair_prio = .equal;
        m.pair_major = false;
        m.fill_t0 = 0;
        m.pair_in_fill = 0;
        m.pair_count = 0;
        m.pair_prev_pending = false;
        m.meter = .{};
        m.ids64_b = &.{};
        m.host_pos_b = &.{};
        m.ids64_c = &.{};
        m.host_pos_c = &.{};
        m.pair_bytes = 0;
        m.rdma_rings = null;
        m.vfan = .{};
        m.watch = .{};
        m.star_busy = .init(false);
        const d = ctx.d;
        m.cfg = try Config.read(a, io, o.model_dir);
        const c = &m.cfg;
        // TF_DS_ROCI_STAGGER_PRIO: pass A's stream (this one, which every other piece of work also runs on) at the priority the mode gives it
        m.pair_prio = roci_pair.prioOf(if (std.c.getenv("TF_DS_ROCI_STAGGER_PRIO")) |v| std.mem.span(v) else null);
        m.stream = if (prompt2d.roci_stagger_asked and m.pair_prio != .equal) blk: {
            const range = try cuda.Stream.priorityRange(d);
            break :blk try cuda.Stream.initPriority(d, roci_pair.prioritiesOf(m.pair_prio, range.least, range.greatest)[0]);
        } else try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();
        // rank 0's ports for its followers listen before its weights load: a follower that loads first connects and
        // waits in its port's backlog instead of retrying until rank 0 and the followers before it are in
        m.fan = try prompt2d.Fan.listenAt(o.master, o.listen, o.port, o.rank, o.world);
        m.fan.share_digests = o.world > 1; // a rank's store is named from all four weight files
        errdefer m.fan.close();
        // the host collectives' own links (TF_DS_ROCI_STARTUP / TF_DS_ROCI_SSD), listening the same way: they never carry a
        // step frame, so a vote or a table can never be read in place of one
        m.vfan = try prompt2d.Fan.listenAt(o.master, o.listen, o.port + prompt2d.host_links_port_offset, o.rank, o.world);
        errdefer m.vfan.close();
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
        // the first lane's proxy threads (Roci.openLane, below) run from its start: a later failure stops them before the arena and the
        // context go (at the function's scope: inside the `if` this would be dropped when the block ends well)
        errdefer if (m.two) |*t| if (t.rc) |rc| rc.close();
        if (sp.pair != null) {
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
                if (m.two.?.combineDoor()) std.log.info("TF_DS_ROCI_COMBINE_DOOR=1: the MoE's routed sum runs in one launch whose first CTAs are the row quarters an owner sum's first part sends, and the last of those CTAs rings the part's doorbell itself, on every rank", .{});
                if (m.two.?.sentFirst()) std.log.info("TF_DS_ROCI_SENT_FIRST=1: wo_b runs in one launch whose first CTAs are the row quarters an owner sum's first part sends, and the last of those CTAs rings the part's doorbell itself (needs TF_DS_MMA_SF_EXACT), on every rank", .{});
                if (m.two.?.engramAhead()) std.log.info("TF_DS_ROCI_ENGRAM_AHEAD=1: an Engram layer's rows, projection and part-1 send run at the start of the layer before it, on every rank", .{});
                if (m.two.?.p2First()) std.log.info("TF_DS_ROCI_P2_FIRST=1: an owner sum's second part is sent as soon as its add has made it, before the add of this node's own block, on every rank", .{});
                const devs = o.rdma_devices.?;
                var pit = std.mem.splitScalar(u8, devs, ',');
                const rails: [2][]const u8 = .{ pit.next() orelse return error.RociRails, pit.next() orelse return error.RociRails };
                m.two.?.rc = try roci_engine.Roci.openLane(gpa, d, m.comm.rank, @intCast(m.two.?.node()), m.comm.world, m.two.?.rank_of, &m.fan, rails, rdmaGid() orelse 3, if (prompt2d.roci_stagger_asked) @intCast(prompt2d.roci_stagger_lanes) else 1, null, .prompt);
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
                const sw = ringSwitches(prompt2d.ring_wc_asked, prompt2d.ring_inplace_asked, rk.hasInplace());
                if (prompt2d.ring_wc_asked) std.log.info("{s}", .{ringWcLine(sw.wc)});
                if (prompt2d.ring_inplace_asked) {
                    var lb: [256]u8 = undefined;
                    std.log.info("{s}", .{ringInplaceLine(&lb, sw)});
                }
                const ring_wait_on = ringWaitOn(rk.hasWait(), devices.items);
                m.rdma_rings = try ring2d.Rings.open(gpa, d, rk, devices.items, role, ord, .{ .max_bytes = ring_bytes, .gid_index = rdmaGid(), .pdl = rdmaPdl(), .collect_peers = peers and rk.collect_peers != null, .wc = sw.wc, .inplace = sw.inplace, .roci = ringWordOf(), .roci_deadline_ns = ringWordDeadlineNs(), .wait = ring_wait_on, .wait_explicit = prompt2d.ring_wait_explicit }, &m.comm, m.stream, !weights.draft_4way);
                m.two.?.rings = &m.rdma_rings.?;
            }
            // the decode exchanges: each rank's start check (the rings, the shapes), then on every rank or the start ends
            {
                const hc = head_cand.layout(round.max_rows, head_cand.kFor(sampling.max_candidates)).bytes;
                const shape = m.two.?.decodeShape(c.hidden, o.round_rows, c.top_k + 1, ring_bytes, hc);
                try Two.require(d, &m.comm, m.stream, "the decode exchanges on the rings", decodeOff(m.two.?.rings != null, shape));
                std.log.info("{s}", .{decodeLine(o.rank, rankOrderText(&m.two.?.rank_of))});
            }
        }
        if (o.rdma_devices == null) {
            // asked without an RDMA ring: say the switch changed nothing
            // (the switches are on by default: only one that was set says it changed nothing)
            if (ringNoRingNote(prompt2d.ring_word_asked, prompt2d.ring_word_explicit)) std.log.info("TF_DS_RING_WORD=1: off: no RDMA ring (TF_RDMA_DEVICES unset)", .{});
            if (ringNoRingNote(prompt2d.ring_wait_asked, prompt2d.ring_wait_explicit)) std.log.info("TF_DS_RING_WAIT=1: off: no RDMA ring (TF_RDMA_DEVICES unset)", .{});
            if (prompt2d.ring_wc_asked) std.log.info("{s}", .{ringWcLine(false)});
            if (prompt2d.ring_inplace_asked) {
                var lb: [256]u8 = undefined;
                std.log.info("{s}", .{ringInplaceLine(&lb, .{ .wc = false, .inplace = false, .why = null })});
            }
        }
        prompt.hc_stride = prompt2d.hc_stride_asked and m.two != null;
        logPdlControls(m.comm.rank);
        std.log.info("transport: Rocinante carries every exchange between the Sparks; no NCCL is loaded or made (rank {d})", .{o.rank});
        if (m.mxk) |*x| m.eng.mx = x;
        try agreeQgateCanary(d, &m.comm, m.stream, prompt2d.qgate_canary_asked);
        if (prompt.qgate_body != .off) try m.mxk.?.useQgateBody();
        errdefer m.rd.dropPar();
        errdefer m.rd.dropPrefetch();
        errdefer m.rd.dropSide(&m.eng);
        m.amax = try m.arena.take(round.max_rows * 4);
        m.cand = try m.arena.take(round.max_rows * (sampling.max_candidates * 12 + 4));
        m.eng.agree_flag = try m.arena.take(64); // G0.3: the graph captures' agreement bytes (agreeAll, agreeCapture)
        errdefer if (m.dr) |*dr| dr.deinit();
        errdefer if (m.pgraphs) |*g| g.deinit();
        errdefer if (m.graphs) |*g| g.deinit();
        if (o.graphs) {
            m.graphs = round.Graphs.init(gpa, o.graph_cap);
            m.graphs.?.lazy.fallbacks = &m.graph_fallbacks;
        }
        if (o.side) try m.rd.useSide(&m.eng);
                m.pgraphs.?.lazy.fallbacks = &m.graph_fallbacks;
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        // (unwound in close()'s order: the pins, the packed rows, the rings, the pool, the tables)
        errdefer if (m.tables) |*t| t.close();
        errdefer if (m.epool) |p| p.deinit(gpa);
        errdefer if (m.paio) |x| x.deinit();
        errdefer if (m.aio) |x| x.deinit();
        errdefer if (m.erows) |r| r.close();
        errdefer if (m.eh) |*eh| prompt.engramUnpin(eh);
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
                const owned = false; // (TF_DS_ENGRAM_OWNED is a retired name: the whole tables are on every node)
                    .chunk_dev_b = if (prompt2d.roci_stagger_asked) try m.arena.take(m.pass_rows * sl * rec) else 0,
                    .chunk_dev_c = if (prompt2d.roci_stagger_asked and prompt2d.roci_stagger_lanes >= 3) try m.arena.take(m.pass_rows * sl * rec) else 0,
        if (c.vision and c.format == .exl3) { // (Format.native: the checkpoint's own gate.bias_vl, loaded with the weights)
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            m.loadVisionBias(path) catch |err| if (o.vision_bias != null or err != error.FileNotFound) return err;
        }

        return engram_rows.workerSlotsN(m.pass_rows / chunk_rows, if (prompt2d.roci_stagger_asked) prompt2d.roci_stagger_lanes else 1);
    /// A pair's second pass: its lane before the first's (it borrows the first's arena), then its events, handle and stream.
    fn closePair(m: *Model) void {
        if (m.two_c) |*t| if (t.rc) |rc| rc.close(); // (the third lane before the second)
        if (m.two_b) |*t| if (t.rc) |rc| rc.close();
        if (m.pair_join) |*ev| ev.deinit();
        if (m.pair_t0) |*ev| ev.deinit();
        for (m.span_t0) |*ev| ev.deinit();
        for (m.span_join) |*ev| ev.deinit();
        if (m.span_t0.len != 0) m.gpa.free(m.span_t0);
        if (m.span_join.len != 0) m.gpa.free(m.span_join);
        m.span_t0 = &.{};
        m.span_join = &.{};
        if (m.pair_ev) |*ev| ev.deinit();
        if (m.blas_c) |*b| b.close();
        if (m.stream_c) |*x| x.deinit();
        if (m.blas_b) |*b| b.close();
        if (m.stream_b) |*x| x.deinit();
    }

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

    pub fn close(m: *Model) void {
        m.joinAhead();
        m.stream.synchronize() catch {};
        m.flushDiag(true);
        if (m.stream_b) |s| s.synchronize() catch {};
        if (m.stream_c) |s| s.synchronize() catch {};
        m.closePair(); // the second lane before the first: it borrows the first's arena
        m.ilead_ev.free(LeadEvents{ .d = m.ctx.d });
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
        m.watch.stop(); // (still watching up to here: a wait above on a rank that went ends this process)
        m.vfan.close();
        m.fan.close();
        m.w.deinit();
        m.cache_file.close(m.io);
        m.ix.deinit();
        m.cache_dir.close(m.io);
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

    /// TF_DS_RING_WORD_SIGNAL's text when TF_DS_RING_WORD runs: unset or `atomic`: the RDMA ATOMIC fetch-and-add (D79, the guaranteed arrival),
    /// `word`/`1`: the inline 8-byte RDMA WRITE; any other text: null, a value this build does not know (the switch then
    /// stays off and its line says so).
    pub fn ringWordSignal(text: ?[]const u8) ?rdma.Roci {
        const v = text orelse return .atomic;
        if (std.mem.eql(u8, v, "word") or std.mem.eql(u8, v, "1")) return .word;
        if (std.mem.eql(u8, v, "atomic")) return .atomic;
        return null;
    }

    pub const RingSignal = struct { roci: rdma.Roci, refuse: bool = false, warn: bool = false, why: ?[]const u8 = null };

    /// The rings' arrival signal and why: unset or atomic is the fetch-and-add where the cards can (`atomic_why` null), unset on a card that cannot is
    /// the inline word with a warning, atomic set outright refuses there; word is always the word. Null for text no signal has.
    pub fn ringWordSignalMode(text: ?[]const u8, atomic_why: ?[]const u8) ?RingSignal {
        const sig = ringWordSignal(text) orelse return null;
        if (sig != .atomic) return .{ .roci = sig };
        if (atomic_why) |why| {
            if (text != null) return .{ .roci = .atomic, .refuse = true, .why = why };
            return .{ .roci = .word, .warn = true, .why = why };
        }
        return .{ .roci = .atomic };
    }

    /// The rings' generation-word mode after the start's checks.
    pub fn ringWordOf() rdma.Roci {
        if (!prompt2d.ring_word_asked) return .off;
        return if (prompt2d.ring_word_atomic_asked) .atomic else .word;
    }

    /// Whether a start without an RDMA ring says a ring switch changed nothing: only when it was set (they are on by default).
    pub fn ringNoRingNote(asked: bool, explicit: bool) bool {
        return asked and explicit;
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

    /// TF_DS_RING_WORD_DEADLINE_MS: a lane's generation-word wait's bound in ms (default 120,000). 0 or unparsable: the
    /// default, because a wait that never ends is the hang the switch avoids.
    pub fn ringWordDeadlineNs() u64 {
        const ms: u64 = if (std.c.getenv("TF_DS_RING_WORD_DEADLINE_MS")) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch 0 else 0;
        return (if (ms == 0) 120_000 else ms) * std.time.ns_per_ms;
    }

    /// The rings' switches after their own checks: the in-place path needs the write-combined slots and both of its
    /// kernels in the fatbin. `why` says why a requested switch is off (none: it is off because it was not asked).
    pub const RingSwitch = struct { wc: bool, inplace: bool, why: ?[]const u8 };

    pub fn ringSwitches(asked_wc: bool, asked_inplace: bool, have_inplace: bool) RingSwitch {
        if (!asked_inplace) return .{ .wc = asked_wc, .inplace = false, .why = null };
        if (rdma.inplaceOff(asked_wc, true)) |why| return .{ .wc = asked_wc, .inplace = false, .why = why };
        if (!have_inplace) return .{ .wc = asked_wc, .inplace = false, .why = "the rings' kernels have no dsv41_rdma_publish and dsv41_rdma_await_send" };
        return .{ .wc = true, .inplace = true, .why = null };
    }

    /// TF_DS_RING_WC's start line.
    pub fn ringWcLine(on: bool) []const u8 {
        return if (on) "TF_DS_RING_WC=1: the RDMA rings' send and landing data slots come from cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocWriteCombined) and are registered with ibv_reg_mr as the baseline; the flags, sizes, sent words, sequence and doorbell stay in the baseline memory: the same bytes" else "TF_DS_RING_WC=1: off: no RDMA ring (TF_RDMA_DEVICES unset)";
    }

    /// TF_DS_RING_INPLACE's start line: on, or off and why.
    pub fn ringInplaceLine(buf: []u8, sw: RingSwitch) []const u8 {
        if (sw.inplace) return "TF_DS_RING_INPLACE=1: the decode exchanges' producers write their partial straight into a fixed send slot and publish it, so the staging copy is gone; the collecting copy-out stays: the same bytes";
        return if (sw.why) |why| std.fmt.bufPrint(buf, "TF_DS_RING_INPLACE=1: off: {s}", .{why}) catch "TF_DS_RING_INPLACE=1: off" else "TF_DS_RING_INPLACE=1: off: no RDMA ring (TF_RDMA_DEVICES unset)";
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

    /// From here on a peer rank that stops stops this one too, with a line that names it (link.Watch): rank 0 watches its
    /// followers' links, a follower rank 0's. The server's rank 0 (native.open) and the lane followers (lanes_main) watch; rank
    /// 0's done (Lanes.finish), a follower reading it (follow) and close end the watch. A peer's link that has been closed for
    /// the grace fails the Rocinante lanes first (`peerGone`), so the ranks still up have it by the abort and not by their
    /// watchdogs, then ends the process.
    pub fn watchPeers(m: *Model) !void {
        if (m.world < 2 or m.watch.thread != null) return;
        m.watch = .{ .rank = m.rank, .stop_hook = peerGone, .ctx = m };
        if (m.rank == 0) {
            for (m.followers(), 1..) |l, r| if (l) |x| try m.watch.add(x, @intCast(r));
        } else try m.watch.add(m.peer(), 0);
        try m.watch.start();
    }

    /// The watch's hook, run on its thread once a peer has been gone for the grace: the Rocinante lanes fail with the rank that went
    /// (a lane that waits in the card as well as one with proxies: Transport.fail), and so do the decode rings.
    fn peerGone(ctx: ?*anyopaque, rank: u32) void {
        const m: *Model = @ptrCast(@alignCast(ctx.?));
        m.failLanes(.peer, @intCast(rank));
        m.abortRings("a peer rank's link closed: the rank stopped");
    }

    /// Every decode ring gives up its waits from here on (Ring.abort: a ring that waits in the card has its queue pairs taken
    /// from the card first, then the abort word set, which releases the kernels' waits).
    fn abortRings(m: *Model, why: []const u8) void {
        if (m.ring) |r| r.abort(why);
        if (m.rdma_rings) |rs| for ([_]?*rdma.Ring{ rs.all4, rs.col, rs.pair }) |y| if (y) |r| r.abort(why);
    }

    /// Ends watchPeers' watch: this process ends its links itself now.
    pub fn unwatch(m: *Model) void {
        m.watch.stop();
    }

    /// Whether this rank's GPU context has failed for good: after a fault in a kernel (an illegal address, a trap) every
    /// later call returns that fault, the stream's query too; a context that survived a failed call answers done or busy.
    pub fn contextFailed(m: *Model) bool {
        _ = m.stream.done() catch return true;
        return false;
    }

    pub fn drafting(m: *const Model) bool {
        return m.dr != null;
    }

    // -- the primitives (both ranks, in the same order) ----------------------------------------------------------

    /// Every decode ring's exchange waits since the last call (instr.WaitHist), summed into `acc`.
    pub fn ringWaits(m: *Model, acc: *instr_mod.WaitHist.Acc) void {
        if (m.ring) |r| r.waits.drain(acc);
        if (m.rdma_rings) |rs| for ([_]?*rdma.Ring{ rs.all4, rs.col, rs.pair }) |y| if (y) |r| r.waits.drain(acc);
    }

    pub fn joinAhead(m: *Model) void {
        if (m.ahead) |th| where.joinThread(th);
        m.ahead = null;
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
        try m.pairJoin();
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
        try m.pairJoin(); // (the copy reads the cache planes a pair's second pass may still be writing)
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

        // (every copy of the tier's reads or writes the caches on the model's stream: behind a pair's second pass first)
            try dv.m.pairJoin();
            try dv.m.pairJoin();
            try dv.m.pairJoin();
            try dv.m.pairJoin();
        try m.pairJoin(); // (a pair's second pass may still read the image rows this replaces)
        if (rows.len > 0) {
            if (m.img_buf == null or m.img_buf.?.len < rows.len) {
                if (m.img_buf) |*b| b.free();
                m.img_buf = null;
                m.img_buf = try cuda.DeviceBuffer.alloc(m.ctx.d, rows.len);
            }
            const d = m.ctx.d;
                // the default: one WARN naming why and what it costs, the round joins the worker as before; set to 1 by the operator: the refusal
                if (speed_defaults.verdict(why, speed_defaults.defaulted("TF_DS_ENGRAM_KEEP_AHEAD")) == .fallback) {
                    std.log.warn("TF_DS_ENGRAM_KEEP_AHEAD: rank {d}: off: the Engram reads are not asynchronous (TF_DS_ENGRAM_AIO=0), so the round and the worker would share one reader pool; without it: {s} (set TF_DS_ENGRAM_KEEP_AHEAD=1 to make this a refusal, 0 to silence it)", .{ m.rank, speed_defaults.costOf("TF_DS_ENGRAM_KEEP_AHEAD") });
                    return;
                }
        try m.pairJoin();
        m.fill_t0 = m.now();
        m.pair_in_fill = 0;
    /// Layer li's Engram on the chunk, in phases: all (the baseline order: the pending MoE post, the rows and their projection, the
    /// sums and the gate), pre (TF_DS_ROCI_ENGRAM_AHEAD: the rows and the projection, then the part-1 send, a layer early) and finish
    /// (the pending post, the sums, the gate).
    fn engramLayer(m: *Model, e: *const prompt.Engine, ch: *prompt.Chunk, seq: []const i32, start: usize, n: usize, li: usize, use_wk: bool, ahead: *?std.Thread, phase: EngramPhase) !void {
        const c = e.c;
        const eh = if (m.eh) |*x| x else return error.NoEngramTables;
        if (phase == .finish) {
            try prompt.engramPending(e, ch, li);
            return prompt.engramFinish(e, ch, eh, li, true);
        }
        const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
        if (use_wk and j < 2) {
            // the worker's resume units of the pass in order: its rows, or its records with packed shards
            // (engram_rows.zig)
            const first = (start - eh.wk_start) / chunk_rows;
            const units = (n + chunk_rows - 1) / chunk_rows;
            if (units > eh.use_rows.len) return error.BadChunk;
            defer {
                eh.use_n = 0;
            }
            for (0..units) |u| {
                if (eh.pk != null) {
                    eh.use_stage[u] = try prompt.engramWorkerStage(eh, j, start + u * chunk_rows, chunk_rows);
                } else eh.use_rows[u] = try prompt.engramWorkerRows(eh, j, start + u * chunk_rows, chunk_rows);
            }
            eh.use_n = units;
            eh.use_unit = chunk_rows;
            const t_apply = m.now();
            if (phase == .pre) {
                try prompt.engramProject(e, ch, eh, li, seq);
                try prompt.engramSendAhead(e, ch, li);
            } else try prompt.engramApply(e, ch, eh, li, seq);
            eh.inst_apply_ns += m.now() - t_apply;
        } else {
            if (ahead.*) |th| {
                where.joinThread(th);
                ahead.* = null;
            }
            if (eh.ahead_err) |err| return err;
            if (phase == .pre) {
                try prompt.engramProject(e, ch, eh, li, seq);
                try prompt.engramSendAhead(e, ch, li);
            } else try prompt.engramApply(e, ch, eh, li, seq);
            const ne = @min(2, c.engram_layers.slice().len);
            if (j + 1 < ne) {
                if (!eh.ahead_ready[j + 1])
                    ahead.* = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
            }
        }
    }

    /// The prompt's chunk of n rows from position `start` (seq: the stream's ids through the chunk's end) through every
    /// layer with the served bounded replay (replay = the prompt's length - the window: an encoder-only chunk stops at
    /// the decoder's first layer). A chunk that reaches the head leaves the logits of its last row in ch.head_g and,
    /// drafting, its DSpark taps absorbed into the slot's drafter rings. True when it reached the head.
    pub fn fillChunk(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        try m.pairJoin();
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
        if (pt) |t| try t.mark(e.s, .embed);
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
        defer if (ahead) |th| where.joinThread(th);
        if (m.eh) |*eh| {
            if (use_wk) {
                if (ahead) |th| where.joinThread(th);
                ahead = null;
                const same = eh.wk_thread != null and eh.wk_seq.ptr == m.fill_seq.ptr and eh.wk_seq.len == m.fill_seq.len and
                    start >= eh.wk_start and (start - eh.wk_start) % chunk_rows == 0 and
            } else {
                prompt.engramWorkerStop(eh);
                if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                    where.joinThread(ahead.?);
                    ahead = null;
                }
                if (ahead == null) {
                    eh.ahead_start = start;
                    eh.ahead_n = n;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
                }
            }
        }
        var engram_pre: ?usize = null;
        for (0..c.layers) |li| {
            try m.issueLead(e.s);
            // TF_DS_ROCI_ENGRAM_AHEAD: the next layer's Engram, when it runs its sums on the transport at owned rows, starts here
            if (m.two) |*t| if (t.engramAhead() and li + 1 < c.layers and e.w.layers[li + 1].engram_wkv != null and prompt.engramOnRoci(e, ch, li + 1)) {
                try m.engramLayer(e, ch, seq, start, n, li + 1, use_wk, &ahead, .pre);
                engram_pre = li + 1;
            };
            if (li == c.layers / 2) {
                if (if (prompt.boundaryRange(e, ch, li)) |r| (if (ch.encoder_only) r else null) else null) |r| {
                if (!try prompt.replayCut(e, ch, &m.view, &shared, li, replay, m.host_pos)) {
                    // encoder-only: no synchronize either (as at the chunk's end below), so the next chunk's host steps
                    // run while this one's last layers do; the profile alone waits
                    if (pt) |t| {
                        try e.s.synchronize();
                        try t.finish();
                    }
                    return false;
                }
                floor = replay;
                kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                // TF_DS_ROCI_ENGRAM_AHEAD: the rows, projection and part-1 send were done at the start of the layer before
                const phase: EngramPhase = if (engram_pre == li) .finish else .all;
                try m.engramLayer(e, ch, seq, start, n, li, use_wk, &ahead, phase);
                if (pt) |t| try t.mark(e.s, .engram);
            }
            // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j, li);
            try prompt.attnMixes(e, ch, li);
            if (pt) |t| try t.mark(e.s, .mix_attn);
            try prompt.attention(e, ch, &m.view, &shared, li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
            if (pt) |t| try t.mark(e.s, .attn);
            try prompt.gather(e, ch, ch.pa, .attn, li);
            if (pt) |t| try t.mark(e.s, .gather_a);
            try prompt.ffnMixes(e, ch, li);
            if (pt) |t| try t.mark(e.s, .mix_ffn);
            try prompt.moe(e, ch, li);
            if (pt) |t| try t.mark(e.s, .moe);
            try prompt.bodyCanary(e, ch, ch.pm, ch.n, li);
            try prompt.gather(e, ch, ch.pm, .moe, li + 1);
            if (pt) |t| try t.mark(e.s, .gather_m);
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
        if (m.two_c) |*t| if (t.rc) |rc| try rc.t.check();
    /// The stream, cuBLAS handle and node of the split of one pass of a pair: A is the model's, B the second lane's. Every launch and
    /// exchange a layer makes reads `Engine.s`, `Engine.t.stream`, `Engine.blas` and `Engine.two` (whose transport is the lane), so
    /// swapping them runs the same serial calls on the pass's own stream and lane.
    fn usePass(m: *Model, p: roci_pair.PairPass) void {
        m.eng.s = m.passStreamOf(p);
        m.eng.t.stream = m.eng.s;
        m.eng.blas = switch (p) {
            .a => &m.blas,
            .b => if (m.blas_b) |*b| b else &m.blas, // (the lane's handle is opened after its chunk, which is made on the lane's own stream)
            .c => if (m.blas_c) |*b| b else &m.blas,
        };
        m.eng.two = switch (p) {
            .a => if (m.two) |*t| t else null,
            .b => if (m.two_b) |*t| t else null,
            .c => if (m.two_c) |*t| t else null,
        };
    }

    /// The stream of a pass of a group: A the model's, B and C the lanes' own.
    fn passStreamOf(m: *const Model, p: roci_pair.PairPass) cuda.Stream {
        return switch (p) {
            .a => m.stream,
            .b => m.stream_b.?,
            .c => m.stream_c.?,
        };
    }

    /// The chunk of a pass of a group.
    fn passChunkOf(m: *Model, p: roci_pair.PairPass) *prompt.Chunk {
        return switch (p) {
            .a => &m.ch,
            .b => &m.ch_b.?,
            .c => &m.ch_c.?,
        };
    }

    /// A new prompt pass on `slot`: the cell its progress is gated by and the generation it counts under (null: a slot beyond the table).
    pub fn meterArm(m: *Model, slot: usize) ?pass_meter.Armed {
        return m.meter.arm(slot);
    }

    /// The serial chunk just issued ends at `rows` of `slot`'s prompt: they count once the first stream has run it.
    pub fn meterChunk(m: *Model, slot: usize, rows: usize) !void {
        try m.meter.mark(m.stream, slot, rows);
    }

    /// The pair just issued: pass A ends at `rows_a`, pass B at `rows_b`; each counts once its own stream has run it.
    pub fn meterPair(m: *Model, slot: usize, rows_a: usize, rows_b: usize) !void {
        try m.meterGroup(slot, &.{ rows_a, rows_b });
    }

    /// The group just issued: pass k ends at `ends[k]`; each counts once its own stream has run it.
    pub fn meterGroup(m: *Model, slot: usize, ends: []const usize) !void {
        for (ends, 0..) |rows, k| try m.meter.mark(m.passStreamOf(@fromBackingInt(@intCast(k))), slot, rows);
    }

    /// The first stream waits for the pair's second pass to finish its work: before anything that is not the next pair (a serial pass,
    /// a decode round, a snapshot) reads or writes what the pair did. Stream-ordered: the host does not wait.
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

    pub fn pairJoin(m: *Model) !void {
        if (m.pair_log) m.sweepSpans();
        if (!m.pair_open.swap(false, .acq_rel)) return;
        try m.stream.wait(m.pair_join.?);
    }

    /// Whether the passes [start_a, start_a + n_a) and [start_b, start_b + n_b) run as a pair: the route, whole 4,096-row encoder-only passes
    /// back to back (the pass rule gave `pass_rows_now` rows), the Engram worker holding both, and the boundary on owned rows.
    pub fn pairEligible(m: *Model, seq: []const i32, start_a: usize, n_a: usize, start_b: usize, n_b: usize, replay: usize, pass_rows_now: usize) bool {
        _ = start_b;
        return m.groupEligible(seq, start_a, &.{ n_a, n_b }, replay, pass_rows_now);
    }

    /// `pairEligible` for the passes of a group, back to back from `start_a` (`ns` their rows: two, or three with TF_DS_ROCI_STAGGER_LANES=3 and
    /// the third lane open): the route, whole encoder-only passes of the booted rows, the Engram worker holding all of them.
    pub fn groupEligible(m: *Model, seq: []const i32, start_a: usize, ns: []const usize, replay: usize, pass_rows_now: usize) bool {
        if (!m.pair or ns.len < 2 or ns.len > m.pair_lanes) return false;
        var end = start_a;
        for (ns) |n| end += n;
        const worker = m.eh == null or (wk_on and m.fill_seq.len >= end and m.fill_seq.ptr == seq.ptr);
        if (!roci_pair.eligibleGroup(true, ns.len, m.pass_rows, pass_rows_now, start_a, ns, replay, worker)) return false;
        if (seq.len < end) return false;
        const t = if (m.two) |*x| x else return false;
        _ = t;
        return (ns[0] % chunk_rows == 0) and ns[0] % m.comm.world == 0;
    }

    /// A pass's start, as the serial chunk's: its view, ids, positions and pre, and its image spans. A and B have their own host id and
    /// position arrays, so the two starts do not overwrite each other.
    fn preparePass(m: *Model, ch: *prompt.Chunk, seq: []const i32, start: usize, n: usize, replay: usize, p: roci_pair.PairPass) !void {
        const e = &m.eng;
        const c = e.c;
        m.usePass(p);
        try ch.view(n, chunk_rows);
        ch.spans = m.ch.spans;
        ch.img_rows = m.ch.img_rows;
        const ids = switch (p) {
            .a => m.ids64[0..n],
            .b => m.ids64_b[0..n],
            .c => m.ids64_c[0..n],
        };
        const pos = switch (p) {
            .a => m.host_pos[0..n],
            .b => m.host_pos_b[0..n],
            .c => m.host_pos_c[0..n],
        };
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| ids[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        try prompt.begin(e, ch, ids, start, pos);
        ch.encoder_only = prompt.encoderOnly(start, n, replay);
        if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
    }

    const PairState = struct { sh: prompt.Shared = .{}, engram_pre: ?usize = null };

    /// One layer of one pass of a pair, the serial chunk's own calls in the same order (fillChunk's loop body, an encoder-only pass). With `cache_ev`
    /// the pass records it after the layer's attention: that is where its compressed caches and window ring are written, the state the
    /// next pass reads at the same layer.
    fn passLayer(m: *Model, ch: *prompt.Chunk, seq: []const i32, start: usize, n: usize, li: usize, st: *PairState, taps: bool, p: roci_pair.PairPass, cache_ev: ?*cuda.Event) !void {
        const e = &m.eng;
        const c = e.c;
        m.usePass(p);
        var no_ahead: ?std.Thread = null;
        if (m.two_b) |*t| _ = t;
        const t = e.two.?;
        if (t.engramAhead() and li + 1 < c.layers and e.w.layers[li + 1].engram_wkv != null and prompt.engramOnRoci(e, ch, li + 1)) {
            try m.engramLayer(e, ch, seq, start, n, li + 1, true, &no_ahead, .pre);
            st.engram_pre = li + 1;
        }
        if (e.w.layers[li].engram_wkv != null) {
            const phase: EngramPhase = if (st.engram_pre == li) .finish else .all;
            try m.engramLayer(e, ch, seq, start, n, li, true, &no_ahead, phase);
        }
        // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
        if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j, li);
        try prompt.attnMixes(e, ch, li);
        try prompt.attention(e, ch, &m.view, &st.sh, li, m.ring_view[li], 0, false);
        // the pass's caches and ring of this layer are written: the other pass's layer may run
        if (cache_ev) |ev| try ev.*.record(e.s);
        try prompt.gather(e, ch, ch.pa, .attn, li);
        try prompt.ffnMixes(e, ch, li);
        try prompt.moe(e, ch, li);
        try prompt.bodyCanary(e, ch, ch.pm, ch.n, li);
        try prompt.gather(e, ch, ch.pm, .moe, li + 1);
        prompt.endLayer(ch);
    }

    /// A pass's boundary layer on owned rows (fillChunk's `boundaryRange` branch); the cache event follows it.
    fn passBoundary(m: *Model, ch: *prompt.Chunk, li: usize, st: *PairState, p: roci_pair.PairPass, cache_ev: *cuda.Event) !void {
        const e = &m.eng;
        m.usePass(p);
        const r = prompt.boundaryRange(e, ch, li) orelse return error.BadPair;
        try prompt.boundaryUpdate(e, ch, &m.view, &st.sh, li, r);
        try cache_ev.*.record(e.s);
    }

    /// TF_DS_ROCI_STAGGER: two whole encoder-only passes in flight, pass A on the model's chunk, stream and lane, B on the second's. B's
    /// layer L waits A's cache event for L; a later pair's A waits the previous pair's B (nothing waits the host). The issue order is
    /// `roci_pair.pairSchedule`: A's layer L, then B's layer L - gap, then B's remaining layers and the boundaries; every rank runs this
    /// same list. The call never synchronizes: the host queues the whole pair, and the next one, ahead of the GPU.
    pub fn fillPair(m: *Model, seq: []const i32, start_a: usize, n_a: usize, start_b: usize, n_b: usize, replay: usize) !void {
        if (start_b != start_a + n_a) return error.BadPair;
        return m.fillGroup(seq, start_a, &.{ n_a, n_b }, replay);
    }

    /// `fillPair` for a group of `ns.len` passes (two, or three with TF_DS_ROCI_STAGGER_LANES=3), back to back from `start_a`: pass k on
    /// lane k, each pass's layer L after its predecessor's layer L caches (`roci_pair.groupSchedule`), the first after the previous group's last.
    pub fn fillGroup(m: *Model, seq: []const i32, start_a: usize, rows: []const usize, replay: usize) !void {
        const ns = rows;
        if (!m.pair) return error.BadPair;
        if (ns.len < 2 or ns.len > m.pair_lanes) return error.BadPair;
        if (!roci_pair.eligibleGroup(true, ns.len, m.pass_rows, m.pass_rows, start_a, ns, replay, m.eh == null or wk_on)) return error.BadPair;
        var starts: [roci_pair.max_lanes]usize = undefined;
        var end = start_a;
        for (ns, 0..) |n, k| {
            starts[k] = end;
            end += n;
        }
        if (seq.len < end) return error.BadChunk;
        const e = &m.eng;
        const c = e.c;
        const half = c.layers / 2;
        const first = !m.pair_open.load(.acquire);
        const host_t0 = m.now();
        const cpu_t0 = pair_nap.cpuNow();
        // every finished pair's GPU span (its first queued work to its second pass's last), in order
        if (m.pair_log) m.sweepSpans();
        if (m.ahead) |th| {
            where.joinThread(th);
            m.ahead = null;
        }
        defer {
            m.usePass(.a);
            prompt.ptimer = null;
        }
        prompt.ptimer = null;
        const taps = m.dr != null;
        if (m.eh) |*eh| {
            const same = eh.wk_thread != null and eh.wk_seq.ptr == m.fill_seq.ptr and eh.wk_seq.len == m.fill_seq.len and
                start_a >= eh.wk_start and (start_a - eh.wk_start) % chunk_rows == 0 and
            if (!same) try prompt.engramWorkerStart(eh, c, m.fill_seq, start_a, chunk_rows, prompt2d.pass_from_asked, m.pass_rows);
        }
        var chs: [roci_pair.max_lanes]*prompt.Chunk = undefined;
        var sts: [roci_pair.max_lanes]PairState = .{ .{}, .{}, .{} };
        for (ns, 0..) |n, k| {
            const which: roci_pair.PairPass = @fromBackingInt(@intCast(k));
            chs[k] = m.passChunkOf(which);
            try m.preparePass(chs[k], seq, starts[k], n, replay, which);
        }
        try m.pair_t0.?.record(m.stream);
        m.span_slot = if (m.pair_log) m.span_ring.push(m.pair_count + 1) else null;
        if (m.span_slot) |sl| try m.span_t0[sl].record(m.stream);
        if (m.pair_log and m.comm.rank == 0 and m.pair_in_fill == 0)
            pairLog("ROCI pair {d} is the fill's first at {d}: admission to the first launch {d:.1} ms", .{ m.pair_count + 1, start_a, @as(f64, @floatFromInt(m.now() - m.fill_t0)) / 1e6 });
        var buf: [roci_pair.maxSteps(roci_pair.max_half)]roci_pair.Step = undefined;
        const steps = roci_pair.groupSchedule(ns.len, half, m.pair_gap, first, &buf);
        for (steps) |step| switch (step) {
            .wait => |w| {
                // a pass waits its predecessor's events: A the previous group's last pass's, B A's, C B's
                m.usePass(w.pass);
                const pred = roci_pair.predecessor(@backingInt(w.pass), m.pair_last);
                try m.eng.s.wait(m.pair_ev.?.at(pred, w.li).*);
            },
            .layer => |l| {
                const k: usize = @backingInt(l.pass);
                const s0 = starts[k];
                const n0 = ns[k];
                try m.issueLead(m.passStreamOf(l.pass));
                try m.passLayer(chs[k], seq[0 .. s0 + n0], s0, n0, l.li, &sts[k], taps, l.pass, m.pair_ev.?.at(k, l.li));
            },
            .boundary => |q| {
                const k: usize = @backingInt(q);
                try m.passBoundary(chs[k], half, &sts[k], q, m.pair_ev.?.at(k, half));
            },
        };
        // everything the group queued on the last stream (each pass's end is behind its predecessor's boundary event), for whoever is not the next group
        const last: roci_pair.PairPass = @fromBackingInt(@intCast(ns.len - 1));
        m.usePass(last);
        try m.pair_join.?.record(m.passStreamOf(last));
        if (m.span_slot) |sl| try m.span_join[sl].record(m.passStreamOf(last));
        m.span_slot = null;
        m.pair_last = ns.len;
        m.pair_open.store(true, .release);
        m.pair_count += 1;
        m.pair_prev_pending = true;
        m.pair_in_fill += 1;
            const eh = &m.eh.?;
            for (&d, cur, eh.inst_snap) |*x, y, z| x.* = y -% z;
            eh.inst_snap = cur;
            const ms = struct {
                fn of(t_ns: u64) f64 {
                    return @as(f64, @floatFromInt(t_ns)) / 1e6;
                }
            }.of;
        }
        if (m.eh) |*eh| {
            eh.inst_wait_ns = 0;
            eh.inst_apply_ns = 0;
        }
    }

    /// The prompt's logits (its last row's, the ranks' columns in rank order) after a fill's last chunk.
    pub fn promptLogits(m: *const Model) u64 {
        return m.ch.head_g;
    }

        try m.pairJoin();
        try m.pairJoin();
                }
                m.clk_rounds += 1;
            }
        }
        try prompt.head(e, ch);
        if (pt) |t| try t.mark(m.stream, .head);
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
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try m.synchronize();
            try t.finish();
    /// A primitive that carries exchanges failed on this rank alone: its peers wait for parts this rank will never send. Fails every
    /// Rocinante lane at once (the abort reaches the peers' queue pairs and releases their waits), so each rank says so within
    /// a second instead of at the watchdog's limit.
    pub fn failExchanges(m: *Model) void {
        m.failLanes(.refused, 0);
    }

    /// failExchanges for any reason: every Rocinante lane fails at once, whoever's fault it is.
    fn failLanes(m: *Model, why: rocinante_mod.FailReason, gone: u8) void {
        if (m.two) |*t| if (t.rc) |rc| rc.t.fail(why, .own1, gone);
        if (m.two_b) |*t| if (t.rc) |rc| rc.t.fail(why, .own1, gone);
        if (m.two_c) |*t| if (t.rc) |rc| rc.t.fail(why, .own1, gone);
        if (m.partner_rc) |rc| rc.t.fail(why, .own1, gone);
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);
    }

    /// The round's logits, R rows of the vocabulary, on the host (after verify).
    /// The round's greedy tokens, R rows (the device's argmax of the round's logits, as sampling.argmax), on the host.
    pub fn roundArgmax(m: *Model, R: usize) ![]const u32 {
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
        try m.pairJoin();

    for ([_]prompt2d.Qgate{ .head, .body_fp8, .body_ulp }) |q| {
        prompt2d.qgate_canary_asked = q;
    }
    prompt2d.qgate_canary_asked = .off;
        }
        return true;
    const keep_b = prompt.qgate_body;
    defer prompt.qgate_body = keep_b;
    for ([_]prompt2d.Qgate{ .off, .head, .body_fp8, .body_ulp }) |q| {
        // not asked, or no FP8 kernels (an EXL3 checkpoint): no canary
        prompt.qgate_body = .body_ulp;
        useQgateCanary(q, false);
        try std.testing.expect(!prompt.qgate_canary);
        try std.testing.expectEqual(prompt2d.Qgate.off, prompt.qgate_body);
        // with them: the head's (1) or the body's (2, 3), never both
        useQgateCanary(q, true);
        try std.testing.expectEqual(q == .head, prompt.qgate_canary);
        try std.testing.expectEqual(if (q == .body_fp8 or q == .body_ulp) q else prompt2d.Qgate.off, prompt.qgate_body);
    }
    // Model.open: the switch read with the others; once the checkpoint's kernels are the engine's, every rank's canary
    // the same (else the start is refused), in effect, and the body's kernel loaded when it is the body's
    const read = std.mem.indexOf(u8, body, "prompt2d.qgate_canary_asked = readQgate();").?;
    const agree = std.mem.indexOf(u8, body, "try agreeQgateCanary(d, &m.comm, m.stream, prompt2d.qgate_canary_asked);").?;
    const load = std.mem.indexOf(u8, body, "if (prompt.qgate_body != .off) try m.mxk.?.useQgateBody();").?;
    try std.testing.expect(read < kernels and kernels < agree and agree < use and use < load);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "useQgateBody("));
}

test "TF_DS_QGATE_CANARY: the text read once (none asked, no collective); another canary on a rank refuses the start" {
    const src = @embedFile("model.zig");
    var at = std.mem.indexOf(u8, src, "fn readQgate() prompt2d.Qgate {").?;
    var body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
    try std.testing.expect(std.mem.indexOf(u8, body, "prompt2d.qgateAsked(v)") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "std.c.getenv(\"TF_DS_QGATE_CANARY\")") != null);
    at = std.mem.indexOf(u8, src, "fn agreeQgateCanary(").?;
    body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
    const off = std.mem.indexOf(u8, body, "    if (asked == .off) return;\n").?;
    const gather = std.mem.indexOf(u8, body, "try prompt2d.ranksDiffer(d, comm, stream, @backingInt(asked))").?;
    const same = std.mem.indexOf(u8, body, "    if (differ == 0) return;\n").?;
    const refuse = std.mem.indexOf(u8, body, "return error.CanariesDiffer;").?;
    try std.testing.expect(off < gather and gather < same and same < refuse);
    const open = std.mem.indexOf(u8, code, "m.rdma_rings = try ring2d.Rings.open(gpa, d, rk, devices.items, role, ord, .{ .max_bytes = ring_bytes, .gid_index = rdmaGid(), .pdl = rdmaPdl(), .collect_peers = peers and rk.collect_peers != null, .wc = sw.wc, .inplace = sw.inplace, .roci = ringWordOf(), .roci_deadline_ns = ringWordDeadlineNs(), .wait = ring_wait_on, .wait_explicit = prompt2d.ring_wait_explicit }, &m.comm, m.stream, !weights.draft_4way);") orelse return error.TestUnexpectedResult;
test "TF_DS_RING_WORD_SIGNAL: unset and atomic name the fetch-and-add (the guaranteed arrival), word the inline word, other text is refused" {
    try std.testing.expectEqual(@as(?rdma.Roci, .atomic), Model.ringWordSignal(null));
    try std.testing.expectEqual(@as(?rdma.Roci, .word), Model.ringWordSignal("word"));
    try std.testing.expectEqual(@as(?rdma.Roci, .word), Model.ringWordSignal("1"));
    try std.testing.expectEqual(@as(?rdma.Roci, .atomic), Model.ringWordSignal("atomic"));
    try std.testing.expectEqual(@as(?rdma.Roci, null), Model.ringWordSignal("atomics"));
    try std.testing.expectEqual(@as(?rdma.Roci, null), Model.ringWordSignal("0"));
    // without TF_DS_RING_WORD the signal changes nothing
    const keep = prompt2d.ring_word_asked;
    defer prompt2d.ring_word_asked = keep;
    prompt2d.ring_word_asked = false;
    try std.testing.expectEqual(rdma.Roci.off, Model.ringWordOf());
}

test "TF_DS_RING_WC and TF_DS_RING_INPLACE: the settings after their checks, and their start lines" {
    // in place needs the write-combined slots and both kernels; asked without either, it is off with the reason
    const off = Model.ringSwitches(false, false, true);
    try std.testing.expect(!off.wc and !off.inplace and off.why == null);
    const wc = Model.ringSwitches(true, false, false);
    try std.testing.expect(wc.wc and !wc.inplace and wc.why == null);
    const nowc = Model.ringSwitches(false, true, true);
    try std.testing.expect(!nowc.wc and !nowc.inplace);
    try std.testing.expectEqualStrings("needs TF_DS_RING_WC=1 (the send slot is a write-combined data slot its producer writes)", nowc.why.?);
    const nokernel = Model.ringSwitches(true, true, false);
    try std.testing.expect(nokernel.wc and !nokernel.inplace);
    try std.testing.expectEqualStrings("the rings' kernels have no dsv41_rdma_publish and dsv41_rdma_await_send", nokernel.why.?);
    const on = Model.ringSwitches(true, true, true);
    try std.testing.expect(on.wc and on.inplace and on.why == null);
    try std.testing.expectEqualStrings("TF_DS_RING_WC=1: the RDMA rings' send and landing data slots come from cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocWriteCombined) and are registered with ibv_reg_mr as the baseline; the flags, sizes, sent words, sequence and doorbell stay in the baseline memory: the same bytes", Model.ringWcLine(true));
    try std.testing.expectEqualStrings("TF_DS_RING_WC=1: off: no RDMA ring (TF_RDMA_DEVICES unset)", Model.ringWcLine(false));
    var lb: [256]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_RING_INPLACE=1: the decode exchanges' producers write their partial straight into a fixed send slot and publish it, so the staging copy is gone; the collecting copy-out stays: the same bytes", Model.ringInplaceLine(&lb, on));
    try std.testing.expectEqualStrings("TF_DS_RING_INPLACE=1: off: needs TF_DS_RING_WC=1 (the send slot is a write-combined data slot its producer writes)", Model.ringInplaceLine(&lb, nowc));
    try std.testing.expectEqualStrings("TF_DS_RING_INPLACE=1: off: no RDMA ring (TF_RDMA_DEVICES unset)", Model.ringInplaceLine(&lb, .{ .wc = false, .inplace = false, .why = null }));
    // both switches are read once in Model.open, before the settings exchange
    const src = @embedFile("model.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "prompt2d.ring_wc_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_RING_WC\")) |v| std.mem.span(v) else null);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "prompt2d.ring_inplace_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_RING_INPLACE\")) |v| std.mem.span(v) else null);"));
}


test "TF_DS_ROCI_STAGGER: every write to state a later pass reads is issued before the cache event that gates it, and the tier's copies join the pair first" {
    const model_src = @embedFile("model.zig");
    const code = model_src[0 .. std.mem.indexOf(u8, model_src, "\ntest \"") orelse return error.TestUnexpectedResult];
    // the layer's event follows the whole attention call (the ring, the compressor's store, the index keys and the compressed groups are written inside it)
    const layer = code[std.mem.indexOf(u8, code, "fn passLayer(").?..std.mem.indexOf(u8, code, "fn passBoundary(").?];
    const att = std.mem.indexOf(u8, layer, "try prompt.attention(").?;
    const rec = std.mem.indexOf(u8, layer, "ev.*.record(e.s)").?;
    try std.testing.expect(att < rec);
    // and no cache write after it in the layer: the calls after the event are the partials' gathers, the mixes and the experts
    for ([_][]const u8{ "kvSourceAt", "fp4Store", "scatterRows", "boundaryUpdate" }) |w| try std.testing.expect(std.mem.indexOf(u8, layer[rec..], w) == null);
    // the boundary's event follows its cache writes (the owned update and the gather that lands the other ranks' rows)
    const bnd = code[std.mem.indexOf(u8, code, "fn passBoundary(").?..std.mem.indexOf(u8, code, "pub fn fillPair(").?];
    try std.testing.expect(std.mem.indexOf(u8, bnd, "prompt.boundaryUpdate(").? < std.mem.indexOf(u8, bnd, "cache_ev.*.record(e.s)").?);
    // the cache planes are written only by kvSourceAt (attention's compressor update, and the boundary's owned update)
    const p = @embedFile("prompt.zig");
    const pcode = p[0 .. std.mem.indexOf(u8, p, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, pcode, "tri_attn.fp4Store("));
    const kv_at = pcode[std.mem.indexOf(u8, pcode, "pub fn kvSourceAt(").?..std.mem.indexOf(u8, pcode, "pub fn boundaryUpdate(").?];
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, kv_at, "tri_attn.fp4Store("));
    for ([_][]const u8{ "cs.raw_kv[li]", "cs.raw_score[li]" }) |w| {
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, pcode, w));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, kv_at, w));
    }
    // every copy the spill tier makes on the model's stream joins the pair's second pass first
    const dev = model_src[std.mem.indexOf(u8, model_src, "pub const SpillDev = struct").?..std.mem.indexOf(u8, model_src, "pub fn done(_: SpillDev").?];
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, dev, "try dv.m.pairJoin();"));
}

test "R6 and R7: lane 1, chunk B, node B and stream B are pass B's alone; a round, a verify and a pass use pass A's scratch" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    for ([_][]const u8{ "pub fn verifyHead(", "pub fn verify(", "pub fn pass(" }) |name| {
        const at = std.mem.indexOf(u8, code, name) orelse return error.TestUnexpectedResult;
        const end = std.mem.indexOfPos(u8, code, at, "\n    }\n") orelse return error.TestUnexpectedResult;
        for ([_][]const u8{ "ch_b", "stream_b", "blas_b", "node_b" }) |w|
            try std.testing.expect(std.mem.indexOf(u8, code[at..end], w) == null);
    }
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, ".lane = 1;"));
    try std.testing.expect(std.mem.indexOf(u8, code, "m.ch_b.?.lane = 1;") != null);
}

test "a pair's pass B: every entry that touches the model's stream starts with pairJoin, and the two that did not now do" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    for ([_][]const u8{ "pub fn snapCopy(", "pub fn copyExtent(", "pub fn setImages(", "pub fn fillBegin(", "pub fn fillChunk(", "pub fn verifyHead(", "pub fn verify(", "pub fn pass(" }) |name| {
        const at = std.mem.indexOf(u8, code, name) orelse return error.TestUnexpectedResult;
        const end = std.mem.indexOfPos(u8, code, at, "\n    }\n") orelse return error.TestUnexpectedResult;
        const first = std.mem.indexOfPos(u8, code, at, "try m.pairJoin();") orelse return error.TestUnexpectedResult;
        try std.testing.expect(first < end);
        // before any other call that is not a bare local: the join is within the function's first statements
        try std.testing.expect(first - at < 600);
    }
}

test "TF_DS_ROCI_PAIR_LOG: the pair line names the engine thread's CPU time, the worker's hash, gather and decode time, rows and bytes, and a fill's first pair names the admission to its first launch" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    for ([_][]const u8{ "engine thread CPU", "hash {d:.1} ms, gather {d:.1} ms, decode {d:.1} ms", "rows asked", "admission to the first launch", "m.fill_t0 = m.now();" }) |w|
        try std.testing.expect(std.mem.indexOf(u8, code, w) != null);
    // the worker adds its time where it spends it
    const p = @embedFile("prompt.zig");
    for ([_][]const u8{ "eh.inst_hash_ns.fetchAdd", "eh.inst_gather_ns.fetchAdd", "eh.inst_decode_ns.fetchAdd", "eh.inst_bytes.fetchAdd" }) |w|
        try std.testing.expect(std.mem.count(u8, p, w) >= 1);
}
    const pair = code[std.mem.indexOf(u8, code, "pub fn fillGroup(").?..];
    const g_at = std.mem.indexOf(u8, steps, "try m.issueLead(m.passStreamOf(l.pass));").?;
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, steps[0..std.mem.indexOf(u8, steps, "// everything the group queued").?], "issueLead("));
    try std.testing.expect(std.mem.indexOf(u8, lcb, "if (m.two_c) |*t| if (t.rc) |rc| try rc.t.check();") != null);

test "Model.open: a refusal or failure after partial setup unwinds what already runs (the transport lanes' proxy threads, the Engram readers, the round's side state, the pair), before the arena and the context go" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    const open = code[std.mem.indexOf(u8, code, "    pub fn open(gpa: std.mem.Allocator").?..std.mem.indexOf(u8, code, "    pub fn close(m: *Model) void {").?];
    const at = struct {
        fn f(s: []const u8, needle: []const u8) !usize {
            return std.mem.indexOf(u8, s, needle) orelse error.TestUnexpectedResult;
        }
    }.f;
    // 1. the first lane's proxy threads start inside Model.open (Roci.openLane); an errdefer at the function's scope (one inside the
    // `if` would be dropped when the block ends well) stops them, registered after the arena's so that it runs before the arena is freed
    // (a refused TF_DS_ROCI_STAGGER otherwise ended rank 0 with SIGSEGV, the lane's threads still running when the context went)
    const arena = try at(open, "errdefer m.arena.deinit();");
    const lane = try at(open, "errdefer if (m.two) |*t| if (t.rc) |rc| rc.close();");
    const first_lane = try at(open, "m.two.?.rc = try roci_engine.Roci.openLane(");
    try std.testing.expect(arena < lane and lane < first_lane);
    // 2. the Engram host's tables, pool, rings, packed rows and pins: unwound in close()'s order, registered before the block that makes them
    const tables_open = try at(open, "m.tables = try engram_io.Tables.open(");
    const names = [_][]const u8{
        "errdefer if (m.tables) |*t| t.close();",
        "errdefer if (m.epool) |p| p.deinit(gpa);",
        "errdefer if (m.paio) |x| x.deinit();",
        "errdefer if (m.aio) |x| x.deinit();",
        "errdefer if (m.erows) |r| r.close();",
        "errdefer if (m.eh) |*eh| prompt.engramUnpin(eh);",
    };
    var last: usize = 0;
    for (names) |n| {
        const i = try at(open, n);
        try std.testing.expect(i > last and i < tables_open);
        last = i;
    }
    // 3. the round's side state and the drafter / graph caches
    const round_init = try at(open, "m.rd = try round.Round.init(");
    try std.testing.expect(round_init < try at(open, "errdefer m.rd.dropSide(&m.eng);"));
    try std.testing.expect(round_init < try at(open, "errdefer m.rd.dropPrefetch();"));
    try std.testing.expect(round_init < try at(open, "errdefer m.rd.dropPar();"));
    try std.testing.expect((try at(open, "errdefer if (m.graphs) |*g| g.deinit();")) < try at(open, "if (o.graphs) {"));
    try std.testing.expect((try at(open, "errdefer if (m.dr) |*dr| dr.deinit();")) < try at(open, "m.dr = try draft.Drafter.init("));
    // 4. the pair's events are unwound inside openPair
    const pf = code[(try at(code, "fn openPair("))..(try at(code, "/// The host's place for a transport's failure line."))];
    try std.testing.expect(std.mem.indexOf(u8, pf, "errdefer if (m.pair_ev) |*e| e.deinit();") != null);
    try std.testing.expect(std.mem.indexOf(u8, pf, "errdefer if (m.pair_join) |*e| e.deinit();") != null);
    try std.testing.expect(std.mem.indexOf(u8, pf, "errdefer if (m.pair_t0) |*e| e.deinit();") != null);
    // 5. close() uses the same pair teardown
    const close = code[try at(code, "pub fn close(m: *Model) void {")..];
    try std.testing.expect(std.mem.indexOf(u8, close, "m.closePair();") != null);
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
    try std.testing.expect(std.mem.indexOf(u8, code, "pairShort(free, m.pass_rows, lane_arena) != 0) why = try pairShortText(") != null);

test "TF_DS_ROCI_STAGGER_LANES: the guard counts every lane after the first, a box under the floor refuses a set count and keeps two for a default, and two lanes are judged as they always were" {
    const keep_caps = rocinante_mod.Transport.shape_caps;
    defer {
        rocinante_mod.setCap(4096);
        rocinante_mod.Transport.shape_caps = keep_caps;
    }
    var p: usize = 2048;
    while (p <= pass_rows_max) : (p += 2048) {
        rocinante_mod.setCap(p);
        const lane = rocinante_mod.layoutOf(.prompt).total;
        const est = pairEstimate(p, lane);
        // two lanes are the pair's guard
        for ([_]u64{ 0, est, est + pair_guard_bytes, est + pair_guard_bytes - 1, 1 << 40 }) |free| try std.testing.expectEqual(pairShort(free, p, lane), pairShortN(free, p, lane, 2));
        // three lanes: two estimates and the floor
        try std.testing.expectEqual(2 * est + pair_guard_bytes, pairShortN(0, p, lane, 3));
        try std.testing.expectEqual(@as(u64, 0), pairShortN(2 * est + pair_guard_bytes, p, lane, 3));
        try std.testing.expectEqual(@as(u64, 1), pairShortN(2 * est + pair_guard_bytes - 1, p, lane, 3));
        // a box that fits two and not three
        try std.testing.expectEqual(@as(u64, 0), pairShortN(est + pair_guard_bytes, p, lane, 2));
        try std.testing.expectEqual(est, pairShortN(est + pair_guard_bytes, p, lane, 3));
    }
    // the 4,096-row passes on a Spark with 12 GiB free: a pair fits, three lanes are short by the third pass's estimate and name it
    rocinante_mod.setCap(4096);
    const lane4096 = rocinante_mod.layoutOf(.prompt).total;
    try std.testing.expectEqual(@as(u64, 0), pairShortN(12 << 30, 4096, lane4096, 2));
    try std.testing.expect(pairShortN(12 << 30, 4096, lane4096, 3) > 0);
    var buf: [1000]u8 = undefined;
    const c: Config = undefined;
    const v: PoolView = .{ .c = &c, .layers = &.{}, .slots = 16, .pool = 1 << 24, .floor = 1 << 20 };
    _ = v;
    // the decision, by what every rank says
    try std.testing.expectEqual(LanesChoice{ .lanes = 2 }, lanesChoice(2, false, true));
    try std.testing.expectEqual(LanesChoice{ .lanes = 2 }, lanesChoice(2, true, false)); // (two lanes: the pair's own guard decides)
    try std.testing.expectEqual(LanesChoice{ .lanes = 3 }, lanesChoice(3, true, true));
    try std.testing.expectEqual(LanesChoice{ .lanes = 3 }, lanesChoice(3, false, true));
    try std.testing.expectEqual(LanesChoice{ .lanes = 3, .refuse = true }, lanesChoice(3, true, false)); // set: refused, by name
    try std.testing.expectEqual(LanesChoice{ .lanes = 2, .warn = true }, lanesChoice(3, false, false)); // the default: two lanes and a warning
    // the source: every rank's memory votes before the choice, the refusal names the setting and the third pass, the third lane opens only for three
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    const open = code[std.mem.indexOf(u8, code, "fn openPair(").?..std.mem.indexOf(u8, code, "/// How the issue lead's events are made and destroyed").?];
    const vote = std.mem.indexOf(u8, open, "lanesChoice(lanes, prompt2d.roci_stagger_lanes_set, try m.eng.agreeAll(fits))").?;
    const agree = std.mem.indexOf(u8, open, "if (!try m.eng.agreeAll(why == null))").?;
    const lane_c = std.mem.indexOf(u8, open, "if (lanes >= 3) {\n        // the third lane").?;
    try std.testing.expect(vote < agree and agree < lane_c);
    try std.testing.expect(std.mem.indexOf(u8, open, "pairShortText(&why_buf, free, m.pass_rows, lane_arena, poolView(m, c), 3)") != null);
    try std.testing.expect(std.mem.indexOf(u8, open, "rocinante_mod.Transport.ring3(m.two.?.rc.?.t, m.two_b.?.rc.?.t, m.two_c.?.rc.?.t);") != null);
    try std.testing.expect(std.mem.indexOf(u8, open, "roci_pair.Events.init(d, c.layers / 2, lanes)") != null);
    try std.testing.expect(std.mem.indexOf(u8, open, "roci_pair.groupChunksN(lanes, m.pass_rows, chunk_rows)") != null);
    // the setting is read once, before the settings are compared, and three lanes without the stagger end the start by name
    const read = std.mem.indexOf(u8, code, "prompt2d.roci_stagger_lanes = roci_pair.lanesOf(text) orelse {").?;
    const asked = std.mem.indexOf(u8, code, "prompt2d.roci_stagger_asked = prompt2d.switchAsked(").?;
    try std.testing.expect(asked < read);
    try std.testing.expect(std.mem.indexOf(u8, code, "TF_DS_ROCI_STAGGER_LANES=3 needs TF_DS_ROCI_STAGGER=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "TF_DS_ROCI_STAGGER_LANES is 2 or 3 (the passes in flight), not") != null);
    _ = &buf;
}

test "TF_DS_ROCI_STAGGER_LANES: a group of three is issued by groupSchedule, each pass waits its predecessor's events (the first the last group's last pass), and the join is the last pass's stream" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    const g = code[std.mem.indexOf(u8, code, "pub fn fillGroup(").?..];
    try std.testing.expect(std.mem.indexOf(u8, g, "roci_pair.groupSchedule(ns.len, half, m.pair_gap, first, &buf)") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "const pred = roci_pair.predecessor(@backingInt(w.pass), m.pair_last);") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "m.pair_ev.?.at(k, l.li)") != null and std.mem.indexOf(u8, g, "m.pair_ev.?.at(k, half)") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "try m.pair_join.?.record(m.passStreamOf(last));") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "m.pair_last = ns.len;") != null);
    // the first pass of a group asks nothing of events the host has not recorded: `first` is the pair_open flag read before anything is queued
    try std.testing.expect(std.mem.indexOf(u8, g, "const first = !m.pair_open.load(.acquire);") != null);
    // every lane is made, closed and reset where the second's are
    try std.testing.expect(std.mem.indexOf(u8, code, "if (m.two_c) |*t| if (t.rc) |rc| rc.close(); // (the third lane before the second)") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "if (m.stream_c) |s| s.synchronize() catch {};") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "        m.ch_c = null;\n        m.stream_c = null;\n        m.blas_c = null;") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "        m.ids64_c = &.{};\n        m.host_pos_c = &.{};") != null);
    // the group's ids and positions are the lane's own: no two passes write one array
    try std.testing.expect(std.mem.indexOf(u8, code, ".c => m.ids64_c[0..n],") != null and std.mem.indexOf(u8, code, ".c => m.host_pos_c[0..n],") != null);
    // the chunk's lane number is the Engram records' set
    try std.testing.expect(std.mem.indexOf(u8, code, "m.ch_c.?.lane = 2;") != null);
}

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
    try std.testing.expect(std.mem.indexOf(u8, code, "return memoryRefusal(buf, what, true, pass_rows, (lanes - 1) * pairEstimate(pass_rows, lane_arena) + pair_guard_bytes, free, v);") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "const what = if (lanes >= 3) \"the second and third passes (TF_DS_ROCI_STAGGER_LANES=3)\" else \"the second pass\";") != null);
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);
    m.two_c = null;
    m.partner_rc = null;
    m.ring = null;
    m.rdma_rings = null;
    try m.checkTransport(); // (nothing configured: nothing failed)
    // a decode ring that lost a peer: every wait on its rank is released by the abort word, and the round after it computes on
    // whatever the buffers hold
    var r: rdma.Ring = undefined;
    r.failed = .init(true);
    r.why = @splat(0);
    @memcpy(r.why[0..4], "lost");
    m.ring = &r;
    try std.testing.expectError(error.RingFailed, m.checkTransport());
    m.ring = null;
    const rings: ring2d.Rings = .{ .g = 0, .r = 0, .p = 0, .all4 = &r, .col = &r, .pair = null };
    m.rdma_rings = rings;
    try std.testing.expectError(error.RingFailed, m.checkTransport());
    // the call sites on rank 0
    const lsrc = @embedFile("lanes.zig");
    const vat = std.mem.indexOf(u8, lsrc, "    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {").?;
    const vbody = lsrc[vat .. vat + std.mem.indexOf(u8, lsrc[vat..], "\n    }\n").?];
    const first = std.mem.indexOf(u8, vbody, "try m.checkTransport();").?;
    try std.testing.expect(first < std.mem.indexOf(u8, vbody, "if (try self.begin(.verify)) {").?);
    const second = std.mem.indexOfPos(u8, vbody, first + 1, "try m.checkTransport();").?;
    try std.testing.expect(second > std.mem.indexOf(u8, vbody, "if (m.cand_first) try m.verifyHead(").? and second < std.mem.indexOf(u8, vbody, "const keyed = head == head_cand.key;").?);
    // the drafter pass: asked once before the passes' frames go out (draftFn sends one frame a group of streams), and again after each pass
    const pat = std.mem.indexOf(u8, lsrc, "    fn draftFn(ptr: *anyopaque").?;
    const pbody = lsrc[pat .. pat + std.mem.indexOf(u8, lsrc[pat..], "\n    }\n").?];
    try std.testing.expect(std.mem.indexOf(u8, pbody, "try m.checkTransport();").? < std.mem.indexOf(u8, pbody, "if (try self.begin(.pass)) {").?);
    try std.testing.expect(std.mem.indexOfPos(u8, pbody, std.mem.indexOf(u8, pbody, "try m.pass(").?, "try m.checkTransport();") != null);
}


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

test "TF_DS_RING_WORD is on unless set to something but 1; a start without an RDMA ring says nothing of a default, and of a switch only when it was set" {
    const src = @embedFile("model.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "prompt2d.ring_word_asked = prompt2d.switchDefaultOn(ring_word_text);") != null);
    try std.testing.expect(!Model.ringNoRingNote(true, false)); // the default, no ring: quiet
    try std.testing.expect(Model.ringNoRingNote(true, true)); // set to 1, no ring: the switch changed nothing
    try std.testing.expect(!Model.ringNoRingNote(false, true)); // set to 0: nothing to say
    try std.testing.expect(!Model.ringNoRingNote(false, false));
    // the explicit flag follows the environment's presence
    try std.testing.expect(std.mem.indexOf(u8, src, "prompt2d.ring_word_explicit = ring_word_text != null;") != null);
}

test "TF_DS_RING_WORD_SIGNAL's decision: unset is the atomic where the cards can, else the word with a warning; atomic set outright refuses there; word is always the word" {
    const why = "device mlx5_0 reports atomic_cap 0: it does no ATOMIC operations";
    const d = Model.ringWordSignalMode(null, null).?;
    try std.testing.expect(d.roci == .atomic and !d.refuse and !d.warn);
    try std.testing.expect(Model.ringWordSignalMode("atomic", null).?.roci == .atomic);
    const f = Model.ringWordSignalMode(null, why).?;
    try std.testing.expect(f.roci == .word and !f.refuse and f.warn and std.mem.eql(u8, f.why.?, why));
    try std.testing.expect(Model.ringWordSignalMode("atomic", why).?.refuse);
    const w = Model.ringWordSignalMode("word", why).?;
    try std.testing.expect(w.roci == .word and !w.refuse and !w.warn);
    try std.testing.expect(Model.ringWordSignalMode("atomics", null) == null);
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
    // the partner split: the refusal is the operator's branch, the warning and the cleared flag the default's, and no lane is opened before either
    const refused = std.mem.indexOf(u8, body[v..], "return error.PartnerRefused;").?;
    const warn = std.mem.indexOf(u8, body[v..], "std.log.warn(\"TF_DS_ENGRAM_PARTNER: off:").?;
    try std.testing.expect(refuse < refused and refused < warn and warn < cleared);
    const kv = std.mem.indexOf(u8, kb, "speed_defaults.verdict(why, speed_defaults.defaulted(\"TF_DS_ENGRAM_KEEP_AHEAD\")) == .fallback").?;
    const kwarn = std.mem.indexOf(u8, kb[kv..], "std.log.warn(\"TF_DS_ENGRAM_KEEP_AHEAD: rank").?;
    const kerr = std.mem.indexOf(u8, kb[kv..], "return error.KeepAheadSharesReader;").?;
    // both are in the defaults table, so the speed report lists a fallback with its reason
    const gib: usize = 1 << 30;
    try std.testing.expect(passRowsFallback("2048", true, 64 * gib) == null);
    try std.testing.expect(passRowsFallback(null, true, 64 * gib) == null);
}

    const refused = std.mem.indexOf(u8, body[v..], "return error.StaggerRefused;").?;
    const warn = std.mem.indexOf(u8, body[v..], "std.log.warn(\"TF_DS_ROCI_STAGGER: off:").?;
    try std.testing.expect(refuse < refused and refused < warn and warn < off); // the refusal is the operator's branch, the fallback the other
    // the fallback names what it costs and runs before any allocation of the second pass
    try std.testing.expect(std.mem.indexOf(u8, body[v..], "speed_defaults.costOf(\"TF_DS_ROCI_STAGGER\")") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "needs passes of two resume units (TF_DS_PASS_ROWS=4096)") != null);
}

test "the four-way drafter's mismatch between ranks still refuses, naming the cause and the way out" {
    try std.testing.expect(std.mem.indexOf(u8, src, "set TF_DS_DRAFT_MAIN_4WAY=0 on every Spark") != null);
}
    }
};
