//! One TP rank of DeepSeek-V4.1-Flash loaded for serving: the weights, kernels, caches and buffers the layer gate sets
//! up (layer_main.zig), for a pool of draft.max_streams streams (MultiDecoder's slots: each stream's window rings,
//! positional stores and drafter rings in its slot, its compressed rows in its extent of one shared plane), and the
//! primitives a lane round runs on them. Both ranks run the same primitives in the same order (lanes.zig sends them
//! to rank 1): a prompt's chunks into its stream's slot and extent (fill), one round over every stream's window and
//! the windows' taps absorbed (verify), and the drafter's batched pass (pass). Every primitive is the gates' own code
//! path (prompt.zig, round.zig, draft.zig) with no checks.
const std = @import("std");
const cuda = @import("cuda");
const engine_api = @import("engine_api");
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
const kv_vmm = @import("kv_vmm.zig");
const land = @import("land.zig");
const prompt2d = @import("prompt2d.zig");
const spill_sync = @import("spill_sync.zig");
const round = @import("round.zig");
const sampling = @import("sampling.zig");
const head_cand = @import("head_cand.zig");
const draft = @import("draft.zig");
const draft_vocab = @import("draft_vocab.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const engram_rows = @import("engram_rows.zig");
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
const api = @import("engine_api");
const link = @import("link.zig");
const ring2d = @import("ring2d.zig");
const roci_engine = @import("roci_engine.zig");
const roci_pair = @import("roci_pair.zig");
const pair_spans = @import("pair_spans.zig");
const keep_ahead = @import("keep_ahead.zig");
const pack_epilogue = @import("pack_epilogue.zig");
const pass_meter = @import("pass_meter.zig");
const engram_partner = @import("engram_partner.zig");
const pair_nap = @import("rdma_nap.zig");
const any_check = @import("any_check.zig");
const rdma = @import("rdma.zig");
const tri_index = @import("tri_index.zig");
const tri_attn = @import("tri_attn.zig");
const tri_norm = @import("tri_norm.zig");
const tri_hc = @import("tri_hc.zig");
const plan = @import("plan.zig");

/// The served prompt chunk (TF_DS_PREFILL_CHUNK), the resume grid: a prompt's passes start at its multiples, and every
/// kept state, mark and frame of the grid stands on one. A pass runs one unit, or with TF_DS_PASS_ROWS up to
/// pass_rows_max rows of whole units (lanes.passLen), where the facts allow (lanes.stream.passRows); with
/// TF_DS_PASS_TAIL the final pass takes the remainder up to the pass rows.
pub const chunk_rows = prompt2d.chunk_rows;

/// The most rows a prompt pass runs (TF_DS_PASS_ROWS=16384: eight resume units).
pub const pass_rows_max = 8 * chunk_rows;

comptime {
    if (pass_rows_max != prompt.max_pass_rows)
        @compileError("pass_rows_max and prompt.max_pass_rows disagree: the host row tables would cap a pass");
}

/// TF_DS_ENGRAM_WORKER=0: a fill's Engram rows read by a thread a chunk (the chunk before's end to this one's first
/// Engram layer) instead of the fill's worker.
var wk_on: bool = true;
/// TF_DS_ENGRAM_PINNED (a pass's Engram worker stages its rows in pinned host memory, the upload waiting for nothing).
var pinned_on: bool = false;
/// TF_DS_ENGRAM_KEEP_AHEAD=1 (every rank alike): the decode round between a prompt's chunks leaves the fill's Engram worker running.
var keep_ahead_asked: bool = false;
pub fn readWorkerEnv() void {
    if (std.c.getenv("TF_DS_ENGRAM_KEEP_AHEAD")) |v| keep_ahead_asked = std.mem.eql(u8, std.mem.span(v), "1");
    if (std.c.getenv("TF_DS_ENGRAM_WORKER")) |v| wk_on = !std.mem.eql(u8, std.mem.span(v), "0");
    if (std.c.getenv("TF_DS_ENGRAM_PINNED")) |v| pinned_on = !std.mem.eql(u8, std.mem.span(v), "0");
    prompt.readAheadPassEnv(); // TF_DS_ENGRAM_AHEAD_PASS (its start line printed here, once)
    prompt.readWaitEnv(); // TF_DS_ENGRAM_WAIT_US
}
/// The KV pool's vote and ledger (core/kv_pool.zig), where the followers' flags parse their numbers.
pub const kv_pool = engine_api.kv_pool;
pub const live_config = engine_api.live_config;
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
    land_slots: u32 = 0, // rank 0: its SSD tier's landing buffers on every rank (TF_DS_SPILL_LAND_SLOTS; 0: the tier off)
    side: bool = false, // the mixes' side work on a stream of its own (round.Round.useSide)
    prefetch: bool = false, // the paced L2 prefetch (round.Round.usePrefetch)
    engram_aio: bool = false, // a round's Engram reads by Linux AIO on O_DIRECT (else the reader pool)
    arena_bytes: usize = 0, // the device arena's first block (0: it grows from 256 MiB blocks as the buffers ask)
    round_rows: usize = round.default_rows, // a round's rows at most (TF_DS_ROUND_ROWS; the four-node lane's 48)
    streams: usize = draft.default_streams, // the pool's streams (--parallel; the four-node lane's 16)
    round_ms: ?[]const f64 = null, // the lane core's round costs by rows (TF_DS_ROUND_MS; null: the served ROUND_MS)
    vision_bias: ?[]const u8 = null, // gate_bias_vl.safetensors (null: the kit's vision/): the image-span routing bias
    kv_target_bytes: u64 = 0, // TF_DS_KV_FREE_TARGET_GIB (--kv-free-target-gib): the free memory to keep; 0: the pool is `pool`'s, as ever
    kv_growth_bytes: u64 = 0, // TF_DS_KV_GROWTH_MIB (--kv-growth-mib): left for what the engine takes after the pool, when kv_growth_set
    kv_growth_set: bool = false, // the operator gave it: else the role's measured default (kv_pool.defaultGrowth)
    kv_vmm: bool = false, // TF_DS_KV_VMM=1 (--kv-vmm): the pool's planes in a virtual-memory reservation (kv_vmm.zig)
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

/// Why the defaulted 4,096-row passes do not apply on this box (null: they do, or the operator set the rows).
pub fn passRowsFallback(text: ?[]const u8, is_default: bool, total_bytes: usize) ?[]const u8 {
    if (!is_default or text == null or !std.mem.eql(u8, text.?, "4096")) return null;
    if (total_bytes >= pass_rows_default_floor) return null;
    return "this box's memory is under 96 GiB, too little for the larger pass's buffers";
}

/// TF_DS_PASS_ROWS's setting (prompt2d.passRowsSetting): any whole number of resume units up to pass_rows_max. Text that is no
/// setting refuses the start, and says so; a setting of more rows says the rule it serves under.
fn readPassRows(from: u32, total_bytes: usize) !u32 {
    var v: ?[]const u8 = if (std.c.getenv("TF_DS_PASS_ROWS")) |x| std.mem.span(x) else null;
    if (passRowsFallback(v, speed_defaults.defaulted("TF_DS_PASS_ROWS"), total_bytes)) |why| {
        std.log.warn("TF_DS_PASS_ROWS: 2048 instead of the default 4096: {s}; without 4,096-row passes: {s}", .{ why, speed_defaults.costOf("TF_DS_PASS_ROWS") });
        v = "2048";
    }
    const rows = prompt2d.passRowsSetting(v, chunk_rows, pass_rows_max) orelse {
        var buf: [320]u8 = undefined;
        std.log.err("{s}", .{prompt2d.passRowsRefusal(&buf, v, chunk_rows, pass_rows_max)});
        return error.BadPassRows;
    };
    if (rows > chunk_rows) std.log.info("TF_DS_PASS_ROWS={d}: a prompt's passes of up to {d} rows on the {d}-row grid from position {d} (TF_DS_PASS_FROM) where nothing decodes, fills or waits", .{ rows, rows, chunk_rows, from });
    return rows;
}

/// TF_DS_PASS_BESIDE's setting (prompt2d.passBesideSetting) under the booted pass rows: the rows a prompt's pass may run
/// while a stream decodes. Text that is no setting, or rows above TF_DS_PASS_ROWS (the passes' buffers are sized by it),
/// refuses the start, and says so.
fn readPassBeside(pass_rows: u32) !u32 {
    const v: ?[]const u8 = if (std.c.getenv("TF_DS_PASS_BESIDE")) |x| std.mem.span(x) else null;
    return prompt2d.passBesideSetting(v, chunk_rows, pass_rows) orelse {
        std.log.err("TF_DS_PASS_BESIDE={s}: not a setting (a whole number of {d}-row units, and not above TF_DS_PASS_ROWS={d})", .{ v.?, chunk_rows, pass_rows });
        return error.BadPassBeside;
    };
}

/// TF_DS_ROUNDS_BESIDE's setting (engine_api.share.roundsFromText): a whole number of rounds from 1, or 0 for unset. Text
/// that is no number of rounds, or any setting beside TENSORFOLD_PREFILL_SHARE below 1 (one control, not two fighting),
/// refuses the start, and says so, before any weight loads.
fn readRoundsBeside() !u32 {
    const v: ?[]const u8 = if (std.c.getenv("TF_DS_ROUNDS_BESIDE")) |x| std.mem.span(x) else null;
    const rounds = engine_api.share.roundsFromText(v) orelse {
        std.log.err("TF_DS_ROUNDS_BESIDE={s}: not a setting (a whole number of Token Generation rounds, 1 or more)", .{v.?});
        return error.BadRoundsBeside;
    };
    var s = engine_api.share.Share.fromEnv();
    s.rounds = rounds;
    if (engine_api.share.Share.refusal(s)) |why| {
        std.log.err("{s}", .{why});
        return error.RoundsBesideWithShare;
    }
    return rounds;
}

/// TF_DS_PASS_FROM's setting (prompt2d.passFromSetting); text that is no setting runs the default, said once.
fn readPassFrom() u32 {
    const v: ?[]const u8 = if (std.c.getenv("TF_DS_PASS_FROM")) |x| std.mem.span(x) else null;
    return prompt2d.passFromSetting(v, chunk_rows) orelse {
        std.log.info("TF_DS_PASS_FROM={s}: not a setting (a multiple of {d}, 0 included): {d}", .{ v.?, chunk_rows, 4 * chunk_rows });
        return 4 * chunk_rows;
    };
}

/// A loaded kit set's variants (Set.lookup: the same match, no word in the log for a missing one).
const SetLookups = struct {
    set: *const cuda.aot.Set,

    pub fn has(f: SetLookups, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        return f.set.lookup(name, args, consts) != null;
    }
};

/// A kit set's aot.json read without its cubins (aot.Manifest: the entries Set.find would take), before any load.
const ManifestFinds = struct {
    m: *const cuda.aot.Manifest,

    pub fn has(f: ManifestFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        return f.m.find(name, args, consts) != null;
    }
};

/// TF_DS_DRAFT_4WAY's start check on this rank, before the weights load (the stages' parts are loaded or not, so a
/// check after it would cost a load): null when it can run (a drafter, the checkpoint's own weights, a 2D node, the
/// decode kernels, the kit's extra set at KIT_DIR/aot-extra holding the four-way stages' attention variants: its
/// aot.json read, no cubin loaded), else why not.
fn draftFourWayOff(gpa: std.mem.Allocator, io: std.Io, c: *const Config, sp: plan.Split, kit_dir: []const u8, drafts: bool, dec_on: bool) !?[]const u8 {
    if (!drafts) return "this start loads no drafter";
    if (c.format != .native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    if (sp.pair == null) return "the four-way stages are the four-node 2D split's";
    if (!dec_on) return "needs the decode tensor-core kernels (TF_DS_DECODE_DEC=0 turns them off)";
    const dir = try std.fs.path.join(gpa, &.{ kit_dir, "aot-extra" });
    defer gpa.free(dir);
    var man = cuda.aot.Manifest.read(gpa, io, dir) catch return "the kit has no extra set (aot-extra/aot.json)";
    defer man.deinit();
    return try draft.fourWayKitOff(gpa, ManifestFinds{ .m = &man }, c, sp.headSpan(c.*)[1]);
}

/// Every node's head rows of the vocabulary, [lo, hi) by rank: one world all-gather of each rank's (`lo`, `hi`).
fn headQuarters(d: *const cuda.Driver, comm: *const comm_mod.Comm, stream: cuda.Stream, lo: usize, hi: usize) ![draft_vocab.nodes][2]usize {
    var q: [draft_vocab.nodes][2]usize = @splat(.{ 0, 0 });
    const w: usize = comm.world;
    var dev = try cuda.DeviceBuffer.alloc(d, (1 + w) * 16);
    defer dev.free();
    const mine = [2]u64{ lo, hi };
    try dev.uploadAsync(0, std.mem.asBytes(&mine), stream.handle);
    try comm.gatherHost(d, dev.ptr, dev.ptr + 16, 16, stream);
    try stream.synchronize();
    var all: [16][2]u64 = undefined;
    if (w > all.len) return error.WorldTooLarge;
    try dev.download(16, std.mem.sliceAsBytes(all[0..w]));
    for (all[0..@min(w, draft_vocab.nodes)], 0..) |x, r| q[r] = .{ @intCast(x[0]), @intCast(x[1]) };
    return q;
}

/// TF_DS_DRAFT_VOCAB `asked` in effect: on every rank or none. Every node's head rows first (headQuarters); then this
/// rank's check and the list's buffers (Drafter.vocabPrepare: the list, the four-node split, the FP4 head, the kit's
/// variants), one world all-gather of whether every rank can, and on agreement the head rows' exchange on every rank
/// (Drafter.vocabExchange). Refused, every rank's passes keep the whole vocabulary. One line says which.
fn useDraftVocab(m: *Model, asked: bool) !void {
    if (!asked) return;
    const quarters = try headQuarters(m.eng.d, &m.comm, m.stream, m.w.vocab_lo, m.w.vocab_hi);
    var staging: ?cuda.DeviceBuffer = null;
    defer if (staging) |*b| b.free();
    var why: ?[]const u8 = "needs the drafter";
    if (m.dr) |*dr| why = dr.vocabPrepare(&m.eng, &m.arena, quarters, m.comm.rank, &staging) catch |e| @errorName(e);
    const cannot = try prompt2d.ranksThatCannot(m.eng.d, &m.comm, m.stream, why == null);
    const agreed = prompt2d.agreedOn(asked, cannot);
    if (!agreed) if (m.dr) |*dr| dr.dropVocab();
    if (agreed) try m.dr.?.vocabExchange(&m.eng, m.comm.rank, &staging.?);
    var rb: [64]u8 = undefined;
    var db: [8]u8 = undefined;
    var buf: [512]u8 = undefined;
    const from = if (agreed) m.dr.?.vocab.?.from_others else 0;
    std.log.info("{s}", .{draft_vocab.line(&buf, cannot, why, prompt2d.rankList(cannot, &rb), from, m.cfg.vocab, draft_vocab.digest(&db))});
}

/// TF_DS_DRAFT_4WAY after the load (each rank decided before its own, which cannot be undone): every rank's drafter
/// split the same way and, four ways, every rank's loaded extra set holding the stages' variants (`why`: this rank's
/// reason it cannot launch them, null when it can); else the start is refused on every rank, named. One line says which.
fn agreeDraftFourWay(d: *const cuda.Driver, comm: *const comm_mod.Comm, stream: cuda.Stream, asked: bool, four: bool, why: ?[]const u8) !void {
    if (!asked) return;
    const tp2 = try prompt2d.ranksThatCannot(d, comm, stream, four);
    const cannot = try prompt2d.ranksThatCannot(d, comm, stream, why == null);
    var buf: [64]u8 = undefined;
    switch (fourWayVerdict(comm.world, tp2, cannot)) {
        .off => std.log.info("TF_DS_DRAFT_4WAY=1: off on every rank: the TP2 drafter", .{}),
        .on => std.log.info("TF_DS_DRAFT_4WAY=1: the drafter's stages four ways on every rank (their sublayer gathers the backbone's quarters; no row pair ring)", .{}),
        .differ => {
            std.log.err("TF_DS_DRAFT_4WAY=1: {s} loaded the TP2 drafter and the others four ways: the start is refused (the ranks' kits differ: give every Spark the same kit, or set TF_DS_DRAFT_4WAY=0)", .{prompt2d.rankList(tp2, &buf)});
            return error.DraftersDiffer;
        },
        .cannot => {
            std.log.err("TF_DS_DRAFT_4WAY=1: {s} cannot launch the four-way stages' attention (this rank: {s}): the start is refused (give every Spark the same kit, or set TF_DS_DRAFT_4WAY=0)", .{ prompt2d.rankList(cannot, &buf), why orelse "can" });
            return error.DraftersDiffer;
        },
    }
}

/// TF_DS_DRAFT_MAIN_4WAY's start check on this rank, before the weights load: null when it can run (a drafter, the
/// checkpoint's own weights, a 2D node, the decode kernels: the quarter's GEMV is the whole one's launch on fewer
/// columns, each column the same sum), else why not.
fn draftMainFourWayOff(c: *const Config, sp: plan.Split, drafts: bool, dec_on: bool) ?[]const u8 {
    if (!drafts) return "this start loads no drafter";
    if (c.format != .native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    if (sp.pair == null) return "main_proj's quarters are the four-node 2D split's";
    if (!dec_on) return "needs the decode tensor-core kernels (TF_DS_DECODE_DEC=0 turns them off)";
    if (c.hidden % 128 != 0) return "a quarter of the hidden columns is not whole 32-row scale blocks";
    return null;
}

/// TF_DS_DRAFT_MAIN_4WAY after the load (each rank decided before its own): every rank's main_proj loaded the same
/// way (quarters on all, or whole on all); else the start is refused on every rank, named. One line says which.
fn agreeDraftMainFourWay(d: *const cuda.Driver, comm: *const comm_mod.Comm, stream: cuda.Stream, asked: bool, four: bool) !void {
    if (!asked) return;
    const whole = try prompt2d.ranksThatCannot(d, comm, stream, four);
    var buf: [64]u8 = undefined;
    switch (fourWayVerdict(comm.world, whole, 0)) {
        .off => std.log.info("TF_DS_DRAFT_MAIN_4WAY=1: off on every rank: main_proj whole on every node", .{}),
        .on => std.log.info("TF_DS_DRAFT_MAIN_4WAY=1: the drafter's main_proj four ways on every rank (a node's quarter of the columns, the absorb's rows gathered by the backbone's quarters)", .{}),
        .differ, .cannot => {
            std.log.err("TF_DS_DRAFT_MAIN_4WAY=1: {s} loaded main_proj whole and the others four ways: the start is refused (the ranks differ: set TF_DS_DRAFT_MAIN_4WAY=0 on every Spark, or fix the one)", .{prompt2d.rankList(whole, &buf)});
            return error.DraftersDiffer;
        },
    }
}

/// TF_DS_DRAFT_4WAY's verdict from the ranks' masks (a bit a rank of `world`): `tp2` the ranks that loaded the TP2
/// drafter, `cannot` the ranks that cannot launch the four-way stages. Off when every rank loaded TP2's, on when
/// every rank loaded four ways and can launch them; a mix of the two drafters, or a four-way rank that cannot, refuses.
const FourWay = enum { off, on, differ, cannot };
fn fourWayVerdict(world: u32, tp2: u32, cannot: u32) FourWay {
    const all = (@as(u32, 1) << @intCast(world)) - 1;
    if (tp2 == all) return .off;
    if (tp2 != 0) return .differ;
    if (cannot != 0) return .cannot;
    return .on;
}

/// TF_DS_PF_SPLIT's setting (mx.PfSplit.parse); text that is no setting runs off, said once.
/// TF_DS_PF_RASTER's setting (prompt2d.rasterSetting): the group of row tiles, 0 unset. Text that is no setting refuses the
/// start, and says so.
fn readPfRaster() !u32 {
    const v: ?[]const u8 = if (std.c.getenv("TF_DS_PF_RASTER")) |x| std.mem.span(x) else null;
    return prompt2d.rasterSetting(v) orelse {
        std.log.err("TF_DS_PF_RASTER={s}: not a setting (2, 4, 8 or 16 row tiles, or unset)", .{v.?});
        return error.BadPfRaster;
    };
}

fn readPfSplit() mx.PfSplit {
    const v: ?[]const u8 = if (std.c.getenv("TF_DS_PF_SPLIT")) |x| std.mem.span(x) else null;
    return mx.PfSplit.parse(v) orelse {
        std.log.info("TF_DS_PF_SPLIT={s}: not a setting (0, 1, A, B, C6 or C7): off", .{v.?});
        return .off;
    };
}

/// TF_DS_PF_SPLIT `asked` in effect on this rank's kernels (none: not the checkpoint's own weights), one line when
/// asked: its configuration, or off and why. Its bits are the fold kernel's either way.
fn usePfSplit(k: ?*mx.Kernels, asked: mx.PfSplit) void {
    if (asked == .off) return;
    const name = mx.PfSplit.name(@backingInt(asked)).?;
    if (mx.splitOff(k != null, if (k) |x| x.pf_off else false)) |why| return std.log.info("TF_DS_PF_SPLIT={s}: off: {s}", .{ name, why });
    k.?.useSplit(asked) catch |e| return std.log.info("TF_DS_PF_SPLIT={s}: off: its kernels: {s}", .{ name, @errorName(e) });
    std.log.info("TF_DS_PF_SPLIT={s}: a prompt chunk's split FP8 GEMMs (wq_a, wkv, wq_b, wo_b, the indexer's wq_b) on {s}", .{ name, if (asked == .table) "the per-shape table's configurations" else "that configuration where it divides the shape's split" });
}

/// TF_DS_PF_GEMM5 `asked` in effect on this rank's kernels, one line when asked: on, or off and why (as TF_DS_PF_SPLIT:
/// the checkpoint's own weights and the tile kernels). Its bits are tf_dsn_pf_gemm's either way.
fn usePfGemm5(k: ?*mx.Kernels, asked: bool) void {
    if (!asked) return;
    if (mx.splitOff(k != null, if (k) |x| x.pf_off else false)) |why| return std.log.info("TF_DS_PF_GEMM5=1: off: {s}", .{why});
    k.?.useGemm5() catch |e| return std.log.info("TF_DS_PF_GEMM5=1: off: its kernel: {s}", .{@errorName(e)});
    std.log.info("TF_DS_PF_GEMM5=1: a prompt chunk's dense FP8 GEMMs without a K split (Engram's wkv) at 5 stages", .{});
}

/// TF_DS_GROUP_TILES `asked` in effect on this rank's kernels, one line when asked: on, or off and why (the
/// checkpoint's own weights and the tile kernels, whose list it builds). The grouping's outputs are the baseline either way.
fn useGroupTiles(k: ?*mx.Kernels, asked: bool) void {
    if (!asked) return;
    if (mx.splitOff(k != null, if (k) |x| x.pf_off else false)) |why| return std.log.info("TF_DS_GROUP_TILES=1: off: {s}", .{why});
    k.?.useGroupTiles() catch |e| return std.log.info("TF_DS_GROUP_TILES=1: off: its kernel: {s}", .{@errorName(e)});
    std.log.info("TF_DS_GROUP_TILES=1: a prompt chunk's grouping builds the experts' tile list once (gate / up and down read it)", .{});
}

/// TF_DS_TOPK_TILES `asked` in effect on this rank's kernels, one line when asked: on, or off and why (the
/// checkpoint's own weights; the indexer's k, config index_topk, at most the kernel's). Its picks are the chain's.
fn useTopkTiles(k: ?*mx.Kernels, asked: bool, index_topk: usize) void {
    if (!asked) return;
    const why: ?[]const u8 = if (k == null) "needs the checkpoint's own FP8 / MXFP4 weights" else if (index_topk > mx.topk_tiles_max) "the indexer's k (index_topk) is past the kernel's 512" else null;
    if (why) |w| return std.log.info("TF_DS_TOPK_TILES=1: off: {s}", .{w});
    k.?.useTopkTiles() catch |e| return std.log.info("TF_DS_TOPK_TILES=1: off: its kernel: {s}", .{@errorName(e)});
    std.log.info("TF_DS_TOPK_TILES=1: a prompt chunk's pruned indexer top-k (past 4,096 keys a row) in one launch a row block", .{});
}

/// TF_DS_GEMM_W16 `asked` in effect: on every rank or none (each rank's check: the checkpoint's own weights and the
/// 16-warp kernels loaded beside the baseline, mx.Kernels.useGemmW16; one world all-gather when asked), one line when
/// asked. A prompt chunk's exact GEMMs and wo_a's bf16 rows then launch 16 warps a block (512 threads): the 8-warp
/// kernels' tiles, stages and shared memory, the same arithmetic in the same order, so the replies' bits without it.
fn useGemmW16(d: *const cuda.Driver, comm: *const comm_mod.Comm, stream: cuda.Stream, k: ?*mx.Kernels, asked: bool) !void {
    if (!asked) return;
    var why: ?[]const u8 = if (k == null) "needs the checkpoint's own FP8 / MXFP4 weights" else null;
    if (why == null) if (k) |x| {
        if (!x.useGemmW16(true)) why = "its 16-warp kernels are not in this image or did not load on this GPU";
    };
    const cannot = try prompt2d.ranksThatCannot(d, comm, stream, why == null);
    if (!prompt2d.agreedOn(asked, cannot)) {
        if (k) |x| x.gemm_w16 = false;
    }
    var buf: [512]u8 = undefined;
    std.log.info("{s}", .{gemmW16Line(&buf, cannot, why, if (k) |x| x.sf == .exact else false)});
}

/// TF_DS_SPLIT64N `asked` in effect (mx.Kernels.useSplit64n): one line when asked, on, or off and why. wkv's and wq_a's exact split64 GEMMs
/// then run on a 64-column tile, two blocks an SM; each output element's K chain is the 128-column tile's, so the replies' bits without it.
/// A rank's own launches: ranks may differ.
fn useSplit64n(k: ?*mx.Kernels, asked: bool) void {
    if (!asked) return;
    const x = k orelse return std.log.info("TF_DS_SPLIT64N=1: off: needs the checkpoint's own FP8 / MXFP4 weights", .{});
    if (!x.useSplit64n(true)) return std.log.info("TF_DS_SPLIT64N=1: off: the 64-column tile is not in this image or did not load on this GPU", .{});
    std.log.info("TF_DS_SPLIT64N=1: wkv's (N 512) and wq_a's (N 1,280) exact split GEMMs of a prompt chunk on a 64 x 64 tile (128 threads, two blocks an SM), the 128-column tile's bytes{s}", .{if (x.sf == .exact) "" else " (they take it with TF_DS_MMA_SF_EXACT)"});
}

/// TF_DS_PF_T64 `asked` in effect (mx.Kernels.usePfT64): one line when asked, on, or off and why. The router's bf16 GEMM of a prompt chunk
/// (384 columns) then runs on a 64 x 64 tile, two CTAs an SM; each output element's K chain is the 128 x 128 tile's, so the replies' bits
/// without it. A rank's own launches: ranks may differ.
fn usePfT64(k: ?*mx.Kernels, asked: bool) void {
    if (!asked) return;
    const x = k orelse return std.log.info("TF_DS_PF_T64=1: off: needs the checkpoint's own FP8 / MXFP4 weights", .{});
    if (!x.usePfT64(true)) return std.log.info("TF_DS_PF_T64=1: off: the 64 x 64 bf16 tile is not in this image or did not load on this GPU", .{});
    std.log.info("TF_DS_PF_T64=1: the router's bf16 GEMM of a prompt chunk (N 384) on a 64 x 64 tile (128 threads, two CTAs an SM), the 128 x 128 tile's bytes", .{});
}

/// TF_DS_GEMM_W16's start line: on every rank with what it covers, or off on every rank, the ranks that cannot and
/// this rank's reason (`why`, null: it can).
fn gemmW16Line(buf: []u8, cannot: u32, why: ?[]const u8, sf_exact: bool) []const u8 {
    if (cannot != 0) {
        var rb: [64]u8 = undefined;
        return std.fmt.bufPrint(buf, "TF_DS_GEMM_W16=1: off on every rank: {s} cannot run it (this rank: {s})", .{ prompt2d.rankList(cannot, &rb), why orelse "can" }) catch buf[0..0];
    }
    return std.fmt.bufPrint(buf, "TF_DS_GEMM_W16=1: a prompt chunk's {s} on 16 warps a block (512 threads), the 8-warp kernels' tiles, stages and shared memory, on every rank", .{if (sf_exact) "exact GEMMs and wo_a's bf16 rows" else "wo_a's bf16 rows alone (the exact GEMMs need TF_DS_MMA_SF_EXACT)"}) catch buf[0..0];
}

/// TF_DS_AQ_VEC in effect (read at Model.open into mx.aq_vec_on, on unless 0): one line, on, or off and why. act_quant's bytes are the baseline either way (the kernels replay's
/// act_quant section compares every one); a call whose x or q is not aligned, or not bf16, runs the baseline kernel in either case.
fn useAqVec(k: ?*mx.Kernels) void {
    if (!mx.aq_vec_on) return std.log.info("TF_DS_AQ_VEC=0: act_quant runs the baseline kernel, a warp a 32-value group (the operator's choice)", .{});
    const x = k orelse return;
    if (x.act_quant_vec == null) return std.log.warn("TF_DS_AQ_VEC: off: this image has no tf_dsn_act_quant_vec_kernel; the baseline act_quant kernel", .{});
    std.log.info("TF_DS_AQ_VEC: on: act_quant on four lanes a 32-value group with 16-byte loads (a call that is not bf16 or not aligned runs the baseline kernel)", .{});
}

/// TF_DS_PF_RUN128 `asked` in effect on this rank's kernels, one line when asked: on, or off and why (the checkpoint's
/// own weights and the tile kernels). Its bits are the baseline tiles' either way.
fn useRun128(k: ?*mx.Kernels, asked: bool) void {
    if (!asked) return;
    if (mx.splitOff(k != null, if (k) |x| x.pf_off else false)) |why| return std.log.info("TF_DS_PF_RUN128=1: off: {s}", .{why});
    k.?.useRun128() catch |e| return std.log.info("TF_DS_PF_RUN128=1: off: its kernels: {s}", .{@errorName(e)});
    std.log.info("TF_DS_PF_RUN128=1: a prompt chunk's MXFP4 expert tiles (gate / up and down) stage each weight row 128 bytes at a time", .{});
}

/// TF_DS_GEMM_R2 `asked` in effect on this rank's kernels, one line when asked: on, or off and why (the checkpoint's
/// own weights, TF_DS_MMA_SF_EXACT's exact arithmetic in effect, the R2 kernels this image carries). Its outputs are
/// the baseline either way.
fn useGemmR2(k: ?*mx.Kernels, asked: bool) void {
    if (!asked) return;
    if (k == null) return std.log.info("TF_DS_GEMM_R2=1: off: needs the checkpoint's own FP8 / MXFP4 weights", .{});
    if (k.?.sf != .exact) return std.log.info("TF_DS_GEMM_R2=1: off: needs TF_DS_MMA_SF_EXACT=1 in effect (its exact kernels)", .{});
    if (!k.?.useGemmR2(true)) return std.log.info("TF_DS_GEMM_R2=1: off: this image has none of its kernels (native_sf.cu's _r2 kernels)", .{});
    std.log.info("TF_DS_GEMM_R2=1: the exact expert GEMMs' padding rows' copies left out, down on 128-row tiles", .{});
}

/// TF_DS_MMA_SF `asked` in effect: on every rank or none (each rank's check: the checkpoint's own weights, the prompt
/// tile kernels, the block-scaled kernels loaded on this GPU, mx.Kernels.useMmaSf; one world all-gather when asked), one
/// line when asked (mmaSfLine, `covers` the GEMMs it takes). Not the baseline bits: a prompt chunk's FP8 GEMMs accumulate
/// their K blocks' scaled sums in the tensor core.
fn useMmaSf(d: *const cuda.Driver, comm: *const comm_mod.Comm, stream: cuda.Stream, k: ?*mx.Kernels, asked: bool, covers: []const u8) !void {
    if (!asked) return;
    var why: ?[]const u8 = mx.splitOff(k != null, if (k) |x| x.pf_off else false);
    if (why == null) if (k) |x| x.useMmaSf(d, cuda.kernels.dsv41_native_sf, .chained) catch |e| {
        why = if (e == error.Invalid) "this build has no block-scaled kernels (native_sf.cu)" else "its block-scaled kernels (sm_121a) did not load on this GPU (the line above says why)";
    };
    const cannot = try prompt2d.ranksThatCannot(d, comm, stream, why == null);
    if (!prompt2d.agreedOn(asked, cannot)) {
        if (k) |x| x.sf = .off;
    }
    var buf: [512]u8 = undefined;
    std.log.info("{s}", .{mmaSfLine(&buf, cannot, why, covers)});
}

/// TF_DS_MMA_SF's start line: on every rank with the GEMMs it covers, or off on every rank, the ranks that cannot and
/// this rank's reason (`why`, null: it can).
fn mmaSfLine(buf: []u8, cannot: u32, why: ?[]const u8, covers: []const u8) []const u8 {
    if (cannot != 0) {
        var rb: [64]u8 = undefined;
        return std.fmt.bufPrint(buf, "TF_DS_MMA_SF=1: off on every rank: {s} cannot run it (this rank: {s})", .{ prompt2d.rankList(cannot, &rb), why orelse "can" }) catch buf[0..0];
    }
    return std.fmt.bufPrint(buf, "TF_DS_MMA_SF=1: a prompt chunk's FP8 GEMMs on the block-scaled MMA, each split's K blocks accumulated in the tensor core (not the baseline bits), on every rank: {s}", .{covers}) catch buf[0..0];
}

/// The GEMMs of these layers TF_DS_MMA_SF's kernels take: each dense FP8 linear whose shape they take (mx.sfShape) in
/// some layer, by its field name in a fixed order, then the experts when a layer has the checkpoint's own.
/// The dense linears' field names TF_DS_MMA_SF's kernels may take (sfCovers' order).
const sf_fields = .{ "wq_a", "wkv", "wq_b", "wo_b", "comp_wkv", "comp_wgate", "idx_wq_b", "idx_wk", "engram_wkv" };

fn sfCovers(layers: []const weights.Layer, exact_form: bool, buf: []u8) []const u8 {
    const fields = sf_fields;
    var w: std.Io.Writer = .fixed(buf);
    var n: usize = 0;
    inline for (fields) |field| {
        var takes = false;
        for (layers) |lay| {
            const l: ?weights.Linear = @field(lay, field);
            if (l) |x| takes = takes or (x.fmt == .fp8 and (if (exact_form) mx.sfExactShape(x.k, x.n) else mx.sfShape(x.k, x.n)));
        }
        if (takes) {
            w.print("{s}{s}", .{ if (n == 0) "" else ", ", field }) catch {};
            n += 1;
        }
    }
    for (layers) |lay| if (lay.experts.native != null) {
        w.print("{s}the experts' gate / up and down", .{if (n == 0) "" else " and "}) catch {};
        break;
    };
    return w.buffered();
}

/// TF_DS_MMA_SF_EXACT `asked` in effect: on every rank or none. Each rank's check: the checkpoint's own weights and the
/// prompt tile kernels; TF_DS_MMA_SF not taking the kernels in its own arithmetic; the block-scaled kernels loaded on
/// this GPU (mx.Kernels.useMmaSf); then its guard over this rank's weight scales (sfGuard: the GEMMs it keeps on the baseline
/// kernels). One world all-gather when asked, one line (mmaSfExactLine, `covers` the GEMMs it takes). The baseline bits.
fn useMmaSfExact(gpa: std.mem.Allocator, d: *const cuda.Driver, comm: *const comm_mod.Comm, stream: cuda.Stream, k: ?*mx.Kernels, asked: bool, layers: []const weights.Layer, covers: []const u8) !void {
    if (!asked) return;
    var why: ?[]const u8 = mx.splitOff(k != null, if (k) |x| x.pf_off else false);
    if (why == null and k.?.sf == .chained) why = "TF_DS_MMA_SF=1 runs the block-scaled kernels in its own arithmetic";
    var kb: [320]u8 = undefined;
    var wb: [160]u8 = undefined; // (a guard error's reason)
    var kept: []const u8 = "none";
    if (why == null) if (k) |x| {
        x.useMmaSf(d, cuda.kernels.dsv41_native_sf, .exact) catch |e| {
            why = if (e == error.Invalid) "this build has no block-scaled kernels (native_sf.cu)" else "its block-scaled kernels (sm_121a) did not load on this GPU (the line above says why)";
        };
        if (why == null) kept = sfGuard(gpa, d, stream, x, layers, &kb) catch |e| blk: {
            why = guardWhy(&wb, e);
            break :blk "none";
        };
    };
    const cannot = try prompt2d.ranksThatCannot(d, comm, stream, why == null);
    const agreed = prompt2d.agreedOn(asked, cannot);
    if (k) |x| {
        // refused: its own arithmetic off; TF_DS_MMA_SF's, which took the kernels first, stays
        x.sf = mx.sfAfterExact(x.sf, agreed);
        if (!agreed) {
            x.sf_nkept = 0;
            x.sf_nproven = 0;
            x.sf_apart = 0;
        }
    }
    var sb: [128]u8 = undefined;
    const seen = if (k) |x| sfSeenText(&sb, x.sf_seen) else "";
    var buf: [768]u8 = undefined;
    std.log.info("{s}", .{mmaSfExactLine(&buf, cannot, why, covers, kept, if (k) |x| x.sf_apart else 0, seen)});
}

/// TF_DS_MMA_SF_EXACT's start line: on every rank with the GEMMs it covers and the ones this rank's guard kept on the baseline
/// kernels, or off on every rank, the ranks that cannot and this rank's reason (`why`, null: it can).
/// TF_DS_MMA_SF_EXACT's reason when its guard failed with `e`: its two limits in words, any other error by its name.
fn guardWhy(buf: []u8, e: anyerror) []const u8 {
    return switch (e) {
        error.TooManyKept => "more GEMMs than its guard keeps have a weight scale outside the proven range",
        error.TooManyProven => "more GEMMs than its guard can prove",
        else => std.fmt.bufPrint(buf, "its guard over the weight scales did not run ({s})", .{@errorName(e)}) catch "its guard over the weight scales did not run",
    };
}

fn mmaSfExactLine(buf: []u8, cannot: u32, why: ?[]const u8, covers: []const u8, kept: []const u8, apart: usize, seen: []const u8) []const u8 {
    if (cannot != 0) {
        var rb: [64]u8 = undefined;
        return std.fmt.bufPrint(buf, "TF_DS_MMA_SF_EXACT=1: off on every rank: {s} cannot run it (this rank: {s})", .{ prompt2d.rankList(cannot, &rb), why orelse "can" }) catch buf[0..0];
    }
    var ab: [128]u8 = undefined;
    const ap = if (apart == 0) "" else std.fmt.bufPrint(&ab, "; kept on the baseline kernels (scale tensors packed at other lengths): the experts in {d} layer{s}", .{ apart, if (apart == 1) "" else "s" }) catch "";
    return std.fmt.bufPrint(buf, "TF_DS_MMA_SF_EXACT=1: a prompt chunk's FP8 GEMMs on the block-scaled MMA, each K block's MMA from zero, then the baseline fp32 add (the baseline bits), on every rank: {s}; kept on the baseline kernels (a weight scale outside the proven range): {s}{s}; weight scale bytes seen: {s}", .{ covers, kept, ap, seen }) catch buf[0..0];
}

/// The weight scale bytes TF_DS_MMA_SF_EXACT's guard saw (mx.Kernels.sf_seen) in words: the smallest and the largest of
/// the dense FP8 linears', the routed experts' MXFP4 and the shared expert's; a class with nothing read, none.
fn sfSeenText(buf: []u8, seen: [3][2]u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    for ([_][]const u8{ "FP8", "MXFP4", "shared expert" }, seen, 0..) |name, r, i| {
        if (r[0] > r[1]) w.print("{s}{s} none", .{ if (i == 0) "" else ", ", name }) catch {} else w.print("{s}{s} {d}..{d}", .{ if (i == 0) "" else ", ", name, r[0], r[1] }) catch {};
    }
    return w.buffered();
}

/// The GEMMs TF_DS_MMA_SF_EXACT's guard kept on the baseline kernels in words: each dense field with its layers' count
/// (`dense`, sf_fields' order), then the experts' (`experts` layers); "none".
fn sfKeptText(buf: []u8, dense: [sf_fields.len]usize, experts: usize) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    var n: usize = 0;
    inline for (sf_fields, 0..) |field, i| if (dense[i] > 0) {
        w.print("{s}{s} in {d} layer{s}", .{ if (n == 0) "" else ", ", field, dense[i], if (dense[i] == 1) "" else "s" }) catch {};
        n += 1;
    };
    if (experts > 0) {
        w.print("{s}the experts in {d} layer{s}", .{ if (n == 0) "" else ", ", experts, if (experts == 1) "" else "s" }) catch {};
        n += 1;
    }
    return if (n == 0) "none" else w.buffered();
}

/// TF_DS_MMA_SF_EXACT's guard over this rank's weights, in one pass on the GPU (mx.scaleRanges): the scale bytes of
/// every dense FP8 linear the exact form takes (mx.sfExactShape) and of every layer's experts (each expert's three
/// tensors, their own bytes from the pointer and owner tables, at expertScaleLens' lengths). A layer whose loader did
/// not tie its tensors to those lengths (NativeExperts.scales_match false) has none of its bytes read: its experts are
/// kept, counted apart (k.sf_apart). It proves (mx.sfProve) a linear, or a layer's experts, whose every byte was read
/// and lies in the proven range (mx.sfProvenRead: FP8 or MXFP4, the shared expert FP8); the block-scaled kernels take
/// only those (mx.sfDense, mx.sfExperts). The rest are kept on the baseline kernels (mx.sfKeep, the words). It starts from
/// empty sets; the bytes it read go to k.sf_seen. The GEMMs kept for a scale outside the range in words (sfKeptText)
/// into `buf`.
pub fn sfGuard(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, k: *mx.Kernels, layers: []const weights.Layer, buf: []u8) ![]const u8 {
    const Tag = struct { layer: usize, field: ?usize, fp4: bool, shared: bool };
    k.sf_nkept = 0;
    k.sf_nproven = 0;
    k.sf_seen = @splat(.{ 255, 0 });
    k.sf_apart = 0;
    var ranges: std.ArrayList(mx.ByteRange) = .empty;
    defer ranges.deinit(gpa);
    var tags: std.ArrayList(Tag) = .empty;
    defer tags.deinit(gpa);
    const layout_ok = try gpa.alloc(bool, layers.len);
    defer gpa.free(layout_ok);
    @memset(layout_ok, true);
    for (layers, 0..) |lay, li| {
        inline for (sf_fields, 0..) |field, fi| {
            const l: ?weights.Linear = @field(lay, field);
            if (l) |x| if (x.fmt == .fp8 and mx.sfExactShape(x.k, x.n)) {
                try ranges.append(gpa, mx.linearScales(x));
                try tags.append(gpa, .{ .layer = li, .field = fi, .fp4 = false, .shared = false });
            };
        }
        if (lay.experts.native) |nx| {
            const e: usize = lay.experts.count;
            const tabs = try gpa.alloc(u64, 3 * e);
            defer gpa.free(tabs);
            for ([_]u64{ nx.s1, nx.s3, nx.s2 }, 0..) |t, j| try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(tabs[j * e ..].ptr), t, e * 8), "cuMemcpyDtoH");
            var owner: ?[]i32 = null;
            defer if (owner) |o| gpa.free(o);
            if (nx.owner != 0) {
                const o = try gpa.alloc(i32, e);
                owner = o;
                try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(o.ptr), nx.owner, e * 4), "cuMemcpyDtoH");
            }
            const out = try ranges.addManyAsSlice(gpa, 3 * e);
            mx.expertScales(lay.experts, tabs[0..e], tabs[e .. 2 * e], tabs[2 * e .. 3 * e], owner, out);
            // a layer packed at other lengths: none of its bytes read (its experts kept, apart)
            if (!nx.scales_match) for (out) |*rg| {
                rg.len = 0;
            };
            layout_ok[li] = nx.scales_match; // the loader packed expertScaleLens' bytes
            for (0..3 * e) |i| try tags.append(gpa, .{ .layer = li, .field = null, .fp4 = i / 3 != e - 1, .shared = i / 3 == e - 1 });
        }
    }
    const mm = try gpa.alloc([2]u8, ranges.items.len);
    defer gpa.free(mm);
    try mx.scaleRanges(k, d, s, gpa, ranges.items, mm);
    var dense: [sf_fields.len]usize = @splat(0);
    var experts: usize = 0;
    var apart: usize = 0;
    const bad = try gpa.alloc(bool, layers.len); // a layer's experts with a tensor not proven
    defer gpa.free(bad);
    @memset(bad, false);
    for (tags.items, mm, ranges.items) |t, r, rg| {
        if (rg.len > 0 and r[0] <= r[1]) {
            const c: usize = if (t.field != null) 0 else if (t.shared) 2 else 1;
            k.sf_seen[c] = .{ @min(k.sf_seen[c][0], r[0]), @max(k.sf_seen[c][1], r[1]) };
        }
        if (mx.sfProvenRead(rg.len, r, t.fp4) and (t.field != null or layout_ok[t.layer])) {
            if (t.field != null) try mx.sfProve(k, rg.ptr); // a dense linear's key: its scales (l.s)
            continue;
        }
        if (t.field) |fi| {
            try mx.sfKeep(k, rg.ptr);
            dense[fi] += 1;
        } else bad[t.layer] = true;
    }
    for (layers, bad) |lay, b| if (lay.experts.native) |nx| {
        if (!nx.scales_match) {
            try mx.sfKeep(k, nx.s1);
            apart += 1;
        } else if (b) {
            try mx.sfKeep(k, nx.s1);
            experts += 1;
        } else try mx.sfProve(k, nx.s1);
    };
    k.sf_apart = apart;
    return sfKeptText(buf, dense, experts);
}

/// This process's anonymous memory in 2 MiB pages now, KiB (/proc/self/smaps_rollup; null: unreadable).
fn hugeKiB(io: std.Io) ?u64 {
    var f = std.Io.Dir.cwd().openFile(io, "/proc/self/smaps_rollup", .{}) catch return null;
    defer f.close(io);
    var buf: [4096]u8 = undefined;
    const n = f.readPositionalAll(io, &buf, 0) catch return null;
    return engram_rows.anonHugeKiB(buf[0..n]);
}

/// TF_DS_ENGRAM_PREFAULT's line: the row cache faulted in at the start, how long it took, and how much of it the kernel
/// gave 2 MiB pages (the process's AnonHugePages before and after the open: THP set to madvise gives them to madvised
/// memory alone, the cache's).
fn logPrefault(r: *const engram_rows.Rows, mib: usize, huge0: ?u64, huge1: ?u64) void {
    var buf: [256]u8 = undefined;
    std.log.info("{s}", .{prefaultLine(&buf, mib, r.fault[0..r.layers], r.fault_ns, huge0, huge1)});
}

pub fn prefaultLine(buf: []u8, mib: usize, faults: []const engram_rows.Fault, ns: u64, huge0: ?u64, huge1: ?u64) []const u8 {
    if (mib == 0) return "TF_DS_ENGRAM_PREFAULT=1: off: no row cache (TF_DS_ENGRAM_CACHE_MIB=0)";
    for (faults) |f| if (f.why) |why| return std.fmt.bufPrint(buf, "TF_DS_ENGRAM_PREFAULT=1: off: the kernel did not fault the row cache in ({s}); its pages fault in as it fills", .{why}) catch "TF_DS_ENGRAM_PREFAULT=1: off";
    const ms = @as(f64, @floatFromInt(ns)) / 1e6;
    const how: []const u8 = if (faults.len > 0 and faults[0].populated) "the kernel's populate" else "a write to each page";
    if (huge0 != null and huge1 != null) return std.fmt.bufPrint(buf, "TF_DS_ENGRAM_PREFAULT=1: the Engram row cache's {d} MiB faulted in at the start in {d:.0} ms ({s}; {d} MiB of it in 2 MiB pages)", .{ mib, ms, how, (huge1.? -| huge0.?) / 1024 }) catch "TF_DS_ENGRAM_PREFAULT=1";
    return std.fmt.bufPrint(buf, "TF_DS_ENGRAM_PREFAULT=1: the Engram row cache's {d} MiB faulted in at the start in {d:.0} ms ({s}; its 2 MiB pages not read)", .{ mib, ms, how }) catch "TF_DS_ENGRAM_PREFAULT=1";
}

/// TF_DS_DECODE_FUSE `asked` in effect on this rank's kernels, one line when asked: on, or off and why (the checkpoint's
/// own weights and the decode kernels). Its bits are the separate launches' either way.
fn useDecodeFuse(k: ?*mx.Kernels, d: *const cuda.Driver, asked: bool) void {
    if (!asked) return;
    const why: ?[]const u8 = if (k == null) "needs the checkpoint's own FP8 / MXFP4 weights" else if (!k.?.dec_on) "needs the decode tensor-core kernels (TF_DS_DECODE_DEC=0 turns them off)" else null;
    if (why) |w| return std.log.info("TF_DS_DECODE_FUSE=1: off: {s}", .{w});
    k.?.useDecodeFuse(d) catch |e| return std.log.info("TF_DS_DECODE_FUSE=1: off: its kernels: {s}", .{@errorName(e)});
    std.log.info("TF_DS_DECODE_FUSE=1: a decode round's routed sum in the down kernel's launch, and a 2D node's assembly of the experts' intermediate writing down's codes", .{});
}

/// TF_DS_HC_FUSED `asked` in effect: a prompt chunk's mHC mixes and their finish in one launch from the kit's extra set
/// (_hc_pre_pf_fused), one line saying so. A launch the set has no variant for keeps the baseline two kernels (tri_hc.hcPrePf
/// counts it). A kit whose extra set (aot-extra/) did not load (`why`) or holds no _hc_pre_pf_fused (`variants`,
/// extraVariants) ends the start by name: asking for the kernel and running the baseline two under its name would measure nothing.
fn useHcFused(asked: bool, loaded: bool, why: ?[]const u8, variants: usize) !void {
    tri_hc.hc_fused = false;
    if (!asked) return;
    var buf: [192]u8 = undefined;
    if (hcFusedRefusal(&buf, loaded, why, variants)) |text| {
        std.log.err("{s}", .{text});
        return error.KitLacksKernel;
    }
    tri_hc.hc_fused = true;
    std.log.info("TF_DS_HC_FUSED=1: a prompt chunk's mHC mixes and their finish in one launch (the same block partials in their order, the posted streams kept in the kernel), {d} variants in the kit; a shape the set has no variant for keeps the baseline two launches", .{variants});
}

/// TF_DS_ATTN_EPI_FUSED `asked` in effect: a prompt chunk's attention epilogue (inverse RoPE and wo_a's quant) in the final
/// store of the part kernel, the kit's _sparse_attn_part_ue (o is never written). It needs TF_DS_ATTN_EPILOGUE in effect
/// (Model.open checks it where that is put into effect; packed latents take the fusion only with TF_DS_ATTN_U); a launch the set has no variant for keeps
/// the baseline part kernel and epilogue (tri_attn counts it). A kit whose extra set did not load (`why`) or holds no
/// _sparse_attn_part_ue (`variants`) keeps the baseline two kernels, said in a warning line the speed report prints (the switch is on by default).
fn useAttnEpiFused(asked: bool, loaded: bool, why: ?[]const u8, variants: usize) !void {
    tri_attn.attn_epi_fused = false;
    if (!asked) return;
    var buf: [192]u8 = undefined;
    if (attnEpiFusedRefusal(&buf, loaded, why, variants)) |text| {
        // on by default: a kit without the kernel is a loud fallback (the speed report prints this line), never a refused start
        std.log.warn("TF_DS_ATTN_EPI_FUSED=1: off: {s}", .{text});
        return;
    }
    tri_attn.attn_epi_fused = true;
    std.log.info("TF_DS_ATTN_EPI_FUSED=1: a prompt chunk's attention epilogue (inverse RoPE, wo_a's quant) in the part kernel's final store, {d} variants in the kit; a shape the set has no variant for keeps the baseline part kernel and epilogue", .{variants});
}

/// TF_DS_ATTN_EPI_TWIN (`epi`: ue, we or xe; we unless set) beside TF_DS_ATTN_EPI_FUSED: the epilogue-fused part kernel from the kit's
/// _sparse_attn_part_we or _xe (the same bytes as _ue). A kit without it (`variants` 0) keeps _ue, said in a warning line the speed report
/// prints; a launch the set has no variant for keeps _ue (tri_attn counts it). TF_DS_ATTN_TWIN (the decode and unfused calls) does not move it.
fn useAttnEpiWe(epi: tri_attn.EpiTwin, fused: bool, variants_we: usize, variants_xe: usize) void {
    tri_attn.attn_epi_we = false;
    tri_attn.attn_epi_xe = false;
    if (!fused or epi == .ue) return;
    const xe = epi == .xe;
    const variants = if (xe) variants_xe else variants_we;
    const name: []const u8 = if (xe) "_sparse_attn_part_xe" else "_sparse_attn_part_we";
    if (variants == 0) return std.log.warn("TF_DS_ATTN_EPI_TWIN={s}: off: the kit's extra set (aot-extra/aot.json) holds no {s}; the fused attention epilogue keeps _sparse_attn_part_ue", .{ @tagName(epi), name });
    if (xe) tri_attn.attn_epi_xe = true else tri_attn.attn_epi_we = true;
    std.log.info("TF_DS_ATTN_EPI_TWIN={s}: the fused attention epilogue from {s} ({d} variants in the kit; _ue's bytes); a launch the set has no variant for keeps _ue", .{ @tagName(epi), name, variants });
}

/// TF_DS_IDX_TPP as asked (`text`): key tiles a program of a prompt chunk's index score. Unset or empty is the default, 4 (the kit's twin _index_score_rows_t); "4" is 4; "0" is the baseline
/// kernel; null for any other text (Model.open refuses the start with it at its call site; this parse is pure and logs nothing).
fn idxTppWanted(text: ?[]const u8) ?usize {
    const t = text orelse return 4;
    if (t.len == 0 or std.mem.eql(u8, t, "4")) return 4;
    if (std.mem.eql(u8, t, "0")) return 0;
    return null;
}

/// TF_DS_IDX_TPP's value (`text`) where a refused one cannot have got (Model.open): 4 or 0.
fn idxTpp(text: ?[]const u8) usize {
    return idxTppWanted(text) orelse 0;
}

/// TF_DS_IDX_TPP (`text`) in effect: a prompt chunk's index score from the kit's twin _index_score_rows_t (a program takes 4 key tiles in
/// turn: queries, weights and visible lengths read once, the next tile's rows prefetched into L1; the same scores, keys and maxima), one
/// line saying so with the variants the kit holds. On unless 0 (the default since S9's kernel time: t/u 0.958-0.980 at the 9 production shapes). An extra set that did not load (`why`)
/// or one that holds no _index_score_rows_t (`variants`) leaves it off, said in a line that says "off:" (never a refused start: the baseline kernel runs). 0 is the operator's choice, said
/// without "off:" (the speed report would read it as "cannot run").
fn useIdxTpp(text: ?[]const u8, loaded: bool, why: ?[]const u8, variants: usize) void {
    tri_index.idx_tpp = 0;
    const v = idxTppWanted(text) orelse return; // (a bad value never gets here: Model.open refused it)
    if (v == 0) return std.log.info("TF_DS_IDX_TPP=0: a prompt chunk's index score runs the baseline kernel (the operator's choice)", .{});
    if (!loaded) return std.log.warn("TF_DS_IDX_TPP=4: off: the kit's extra set (aot-extra/) did not load ({s})", .{why orelse "none"});
    if (variants == 0) return std.log.warn("TF_DS_IDX_TPP=4: off: the kit's extra set (aot-extra/aot.json) holds no _index_score_rows_t", .{});
    tri_index.idx_tpp = 4;
    std.log.info("TF_DS_IDX_TPP=4: a prompt chunk's index score from _index_score_rows_t (4 key tiles a program, the next tile's rows prefetched into L1; the same scores and keys), {d} variants in the kit; a variant without its twin keeps the baseline kernel", .{variants});
}

/// TF_DS_HC_2S `asked` in effect: the two-block mHC mix as the kit's _hc_mix_pf_2s (the four streams of a K block in one program, the
/// same bytes). On by default: a kit without the kernel keeps _hc_mix_pf_2b, said in a warning line the speed report prints; a
/// launch the set has no variant for keeps 2b (tri_hc counts it).
fn useHc2s(asked: bool, loaded: bool, why: ?[]const u8, variants: usize) void {
    tri_hc.hc_2s = false;
    if (!asked) return;
    if (!loaded) return std.log.warn("TF_DS_HC_2S=1: off: the kit's extra set (aot-extra/) did not load ({s}); the two-block mix stays _hc_mix_pf_2b", .{why orelse "none"});
    if (variants == 0) return std.log.warn("TF_DS_HC_2S=1: off: the kit's extra set (aot-extra/aot.json) holds no _hc_mix_pf_2s; the two-block mix stays _hc_mix_pf_2b", .{});
    tri_hc.hc_2s = true;
    std.log.info("TF_DS_HC_2S=1: the two-block mHC mix with the four streams of a K block in one program (_hc_mix_pf_2s, {d} rows a program), {d} variants in the kit; a shape the set has no variant for keeps _hc_mix_pf_2b", .{ tri_hc.hc_2s_rb, variants });
}

/// What ends a start that asked for TF_DS_ATTN_EPI_FUSED: the kit's extra set did not load, or holds no _sparse_attn_part_ue.
fn attnEpiFusedRefusal(buf: []u8, loaded: bool, why: ?[]const u8, variants: usize) ?[]const u8 {
    if (!loaded) return std.fmt.bufPrint(buf, "the kit's extra set (aot-extra/) did not load ({s}); the epilogue stays its own launch", .{why orelse "none"}) catch "the kit's extra set did not load; the epilogue stays its own launch";
    if (variants == 0) return "the kit's extra set (aot-extra/aot.json) holds no _sparse_attn_part_ue; the epilogue stays its own launch";
    return null;
}

/// What ends a start that asked for TF_DS_HC_FUSED: the kit's extra set did not load, or holds no _hc_pre_pf_fused.
fn hcFusedRefusal(buf: []u8, loaded: bool, why: ?[]const u8, variants: usize) ?[]const u8 {
    if (!loaded) return std.fmt.bufPrint(buf, "TF_DS_HC_FUSED=1: the kit's extra set (aot-extra/) did not load ({s}); not started", .{why orelse "none"}) catch "TF_DS_HC_FUSED=1: the kit's extra set did not load; not started";
    if (variants == 0) return "TF_DS_HC_FUSED=1: the kit's extra set (aot-extra/aot.json) holds no _hc_pre_pf_fused; not started";
    return null;
}

/// TF_DS_ATTN_TWIN (`text`: u, w, x or z; z unless set, speed_defaults) in effect: which of the kit's twins of _sparse_attn_part the calls of
/// TF_DS_ATTN_U and TF_DS_ATTN_U_DECODE launch (tri_attn.attn_twin): w the scale bytes read in the code bytes' layout and the
/// code bytes unpacked where they sit in their registers, x w with the picks' index rows read one block ahead, z w with a
/// decode round's picks rows prefetched into L2 before the PDL wait; all three give u's bytes. One line saying which and how many of the kit's variants have it. Another
/// value ends the start by name. The switch without TF_DS_ATTN_U / TF_DS_ATTN_U_DECODE (`any_attn_u`), an extra set that did
/// not load (`why`) or one that holds no twin of that name keep u, said in an "off" line the speed report prints (the switch
/// is on by default).
fn useAttnTwin(a: std.mem.Allocator, io: std.Io, text: ?[]const u8, any_attn_u: bool, loaded: bool, why: ?[]const u8, extra_dir: []const u8) !void {
    tri_attn.attn_twin = .u;
    const t = text orelse return;
    if (t.len == 0) return;
    const twin = tri_attn.Twin.parse(t) orelse {
        std.log.err("TF_DS_ATTN_TWIN={s}: not one of u, w, x, z; not started", .{t});
        return error.BadSetting;
    };
    if (twin == .u) return;
    // on by default (z): what keeps it from running is a loud fallback to u (the speed report prints the line), never a refused start
    if (!any_attn_u) return std.log.warn("TF_DS_ATTN_TWIN={s}: off: neither TF_DS_ATTN_U nor TF_DS_ATTN_U_DECODE is on, so no call would ask for it; the baseline kernel (u)", .{t});
    if (!loaded) return std.log.warn("TF_DS_ATTN_TWIN={s}: off: the kit's extra set (aot-extra/) did not load ({s}); the baseline kernel (u)", .{ t, why orelse "none" });
    const n = extraVariants(a, io, extra_dir, twin.kernel());
    if (n == 0) return std.log.warn("TF_DS_ATTN_TWIN={s}: off: the kit's extra set (aot-extra/aot.json) holds no {s}; the baseline kernel (u)", .{ t, twin.kernel() });
    tri_attn.attn_twin = twin;
    std.log.info("TF_DS_ATTN_TWIN={s}: the sparse attention calls of TF_DS_ATTN_U / TF_DS_ATTN_U_DECODE launch {s} ({d} variants in the kit; u's bytes); a launch of a variant without it takes the baseline kernel", .{ t, twin.kernel(), n });
}

/// TF_DS_ROPE_W `asked` in effect: rope_heads from the kit's twin _rope_heads_w (each (even, odd) bf16 pair read and written
/// as one 32-bit word: _rope_heads's bytes), one line saying so; a launch of a variant without its twin takes the baseline
/// kernel (Tri.runTwin). An extra set that did not load (`why`) or holds no _rope_heads_w (`variants`) keeps _rope_heads, said
/// in an "off" line the speed report prints (the switch is on by default).
fn useRopeW(asked: bool, loaded: bool, why: ?[]const u8, variants: usize) !void {
    tri_norm.rope_w = false;
    if (!asked) return;
    // on by default: a kit without the twin is a loud fallback (the speed report prints the line), never a refused start
    if (!loaded) return std.log.warn("TF_DS_ROPE_W=1: off: the kit's extra set (aot-extra/) did not load ({s}); the baseline _rope_heads", .{why orelse "none"});
    if (variants == 0) return std.log.warn("TF_DS_ROPE_W=1: off: the kit's extra set (aot-extra/aot.json) holds no _rope_heads_w; the baseline _rope_heads", .{});
    tri_norm.rope_w = true;
    std.log.info("TF_DS_ROPE_W=1: RoPE on the heads from _rope_heads_w (the bf16 pairs as 32-bit words: _rope_heads's bytes), {d} variants in the kit; a launch of a variant without its twin takes the baseline kernel", .{variants});
}

/// TF_DS_QKV_PLAIN `asked` in effect: a decode window's q RMSNorm and window-KV norm / RoPE / quant as one launch of the kit's extra set's
/// _q_kv_norm without rotations (`variants`: how many of the extra set's _q_kv_norm variants there are; the base set has that function only
/// with rotations). An extra set that did not load (`why`) or holds no such variant keeps the two launches, said in an "off" line the speed
/// report prints (the switch is on by default): never a refused start.
fn useQkvPlain(asked: bool, loaded: bool, why: ?[]const u8, variants: usize) void {
    round.qkv_plain = false;
    if (!asked) return;
    if (!loaded) return std.log.warn("TF_DS_QKV_PLAIN=1: off: the kit's extra set (aot-extra/) did not load ({s}); the q norm and the window KV stay two launches", .{why orelse "none"});
    if (variants == 0) return std.log.warn("TF_DS_QKV_PLAIN=1: off: the kit's extra set (aot-extra/aot.json) holds no _q_kv_norm variant without rotations (NROT 0, SPLIT 0); the q norm and the window KV stay two launches", .{});
    round.qkv_plain = true;
    std.log.info("TF_DS_QKV_PLAIN=1: a decode window's q RMSNorm and window KV as one launch of the kit's _q_kv_norm ({d} variants in the extra set; the two kernels' bytes); a launch of a shape without the variant makes the two", .{variants});
}

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

/// The line a kit-twin switch logs when a rank cannot run it: off on every rank, the ranks named, this rank's reason.
fn twinOffLine(buf: []u8, name: []const u8, cannot: u32, mine_on: bool) []const u8 {
    var rb: [64]u8 = undefined;
    return std.fmt.bufPrint(buf, "{s}=1: off: {s} cannot run it (this rank: {s}); every rank runs the baseline kernel", .{ name, prompt2d.rankList(cannot, &rb), if (mine_on) "can" else "its kit's extra set does not provide it" }) catch buf[0..0];
}

/// A kit-twin switch (`on`, put into effect by useAttnU / useIdxU / useAttnUDecode from this rank's own extra set) on every
/// rank or off on every rank, said: one gather of every rank's flag (ranksThatCannot), called by every rank together after
/// the switch's use, only for a switch asked for (asked is every rank's alike: Settings). The twins give the baseline bytes, so
/// a mix of ranks would be a mix of kernels (speed, and a replay that cannot tell), never a different reply.
fn agreeTwin(d: *const cuda.Driver, comm: *const comm_mod.Comm, stream: cuda.Stream, name: []const u8, asked: bool, on: *bool) !void {
    if (!asked) return;
    const mine_on = on.*;
    const cannot = try prompt2d.ranksThatCannot(d, comm, stream, mine_on);
    if (cannot == 0) return;
    on.* = false;
    var buf: [256]u8 = undefined;
    std.log.info("{s}", .{twinOffLine(&buf, name, cannot, mine_on)});
}

/// TF_DS_ATTN_U `asked` in effect: a prompt chunk's sparse attention with packed latents from the kit's twin
/// _sparse_attn_part_u (the hardware unpack, the same keys) when the extra set loaded (`loaded`; else `why` it did
/// not), one line saying which and how many of the kit's _sparse_attn_part variants have their twin (read from the two
/// sets' aot.json); a launch of a variant without one takes the baseline kernel (Tri.runTwin).
fn useAttnU(a: std.mem.Allocator, io: std.Io, asked: bool, loaded: bool, why: ?[]const u8, aot_dir: []const u8, extra_dir: []const u8) void {
    tri_attn.attn_u = false;
    if (!asked) return;
    if (!loaded) return std.log.info("TF_DS_ATTN_U=1: off: the kit's extra set (aot-extra/) did not load ({s})", .{why orelse "none"});
    tri_attn.attn_u = true;
    var cov: ?[2]usize = null;
    if (cuda.aot.Manifest.read(a, io, aot_dir)) |b| {
        var base = b;
        defer base.deinit();
        if (cuda.aot.Manifest.read(a, io, extra_dir)) |x| {
            var extra = x;
            defer extra.deinit();
            cov = base.twinCoverage(&extra, "_sparse_attn_part", "_sparse_attn_part_u");
        } else |_| {}
    } else |_| {}
    if (cov) |n| {
        if (n[1] < n[0]) std.log.warn("TF_DS_ATTN_U=1: partial twin coverage: only {d} of the kit's {d} _sparse_attn_part variants have their twin; a launch of any other runs the baseline kernel", .{ n[1], n[0] });
        std.log.info("TF_DS_ATTN_U=1: a prompt chunk's sparse attention with packed latents from _sparse_attn_part_u (the latents unpacked in hardware), {d} of the kit's {d} _sparse_attn_part variants with their twin; another keeps the baseline kernel", .{ n[1], n[0] });
    } else std.log.info("TF_DS_ATTN_U=1: a prompt chunk's sparse attention with packed latents from _sparse_attn_part_u (the latents unpacked in hardware); a variant without its twin keeps the baseline kernel", .{});
}

/// TF_DS_IDX_U `asked` in effect: a prompt chunk's index score from the kit's twin _index_score_rows_u (the hardware
/// unpack of the FP4 keys, the same scores) when the extra set loaded (`loaded`; else `why` it did not), one line saying
/// which and how many of the kit's _index_score_rows variants have their twin (read from the two sets' aot.json); a
/// launch of a variant without one takes the baseline kernel (Tri.runTwin).
fn useIdxU(a: std.mem.Allocator, io: std.Io, asked: bool, loaded: bool, why: ?[]const u8, aot_dir: []const u8, extra_dir: []const u8) void {
    tri_index.idx_u = false;
    if (!asked) return;
    if (!loaded) return std.log.info("TF_DS_IDX_U=1: off: the kit's extra set (aot-extra/) did not load ({s})", .{why orelse "none"});
    tri_index.idx_u = true;
    var cov: ?[2]usize = null;
    if (cuda.aot.Manifest.read(a, io, aot_dir)) |b| {
        var base = b;
        defer base.deinit();
        if (cuda.aot.Manifest.read(a, io, extra_dir)) |x| {
            var extra = x;
            defer extra.deinit();
            cov = base.twinCoverage(&extra, "_index_score_rows", "_index_score_rows_u");
        } else |_| {}
    } else |_| {}
    if (cov) |n| {
        if (n[1] < n[0]) std.log.warn("TF_DS_IDX_U=1: partial twin coverage: only {d} of the kit's {d} _index_score_rows variants have their twin; a launch of any other runs the baseline kernel", .{ n[1], n[0] });
        std.log.info("TF_DS_IDX_U=1: a prompt chunk's indexer scores from _index_score_rows_u (the keys unpacked in hardware), {d} of the kit's {d} _index_score_rows variants with their twin; another keeps the baseline kernel", .{ n[1], n[0] });
    } else std.log.info("TF_DS_IDX_U=1: a prompt chunk's indexer scores from _index_score_rows_u (the keys unpacked in hardware); a variant without its twin keeps the baseline kernel", .{});
}

/// TF_DS_IDX_V `asked` in effect: a prompt chunk's index score from the kit's twin _index_score_rows_v (the key tile
/// dequantized once a program, each of the program's rows dotted in turn, its head sum written out, at four warps: the
/// same scores and keys), one line saying so; it replaces _index_score_rows_u where both are on, and a launch of a variant
/// without its twin falls back to the u twin and then the baseline kernel (tri_index.scoreRows). A kit whose extra set
/// (aot-extra/) did not load (`why`) or holds no _index_score_rows_v (`variants`, extraVariants) ends the start by name:
/// asking for the twin and running the baseline kernel under its name would measure nothing.
fn useIdxV(asked: bool, loaded: bool, why: ?[]const u8, variants: usize) !void {
    tri_index.idx_v = false;
    if (!asked) return;
    var buf: [192]u8 = undefined;
    if (idxVRefusal(&buf, loaded, why, variants)) |text| {
        std.log.err("{s}", .{text});
        return error.KitLacksTwin;
    }
    tri_index.idx_v = true;
    std.log.info("TF_DS_IDX_V=1: a prompt chunk's indexer scores from _index_score_rows_v (the key tile dequantized once a program, a row's dots at a time, four warps: the same scores and keys), {d} variants in the kit; a launch of a variant without its twin falls back to the u twin and then the baseline kernel", .{variants});
}

/// What ends a start that asked for TF_DS_IDX_V: the kit's extra set did not load, or holds no _index_score_rows_v.
fn idxVRefusal(buf: []u8, loaded: bool, why: ?[]const u8, variants: usize) ?[]const u8 {
    if (!loaded) return std.fmt.bufPrint(buf, "TF_DS_IDX_V=1: the kit's extra set (aot-extra/) did not load ({s}); not started", .{why orelse "none"}) catch "TF_DS_IDX_V=1: the kit's extra set did not load; not started";
    if (variants == 0) return "TF_DS_IDX_V=1: the kit's extra set (aot-extra/aot.json) holds no _index_score_rows_v; not started";
    return null;
}

/// How many variants of `function` the kit's extra set at `extra_dir` holds (0 when its aot.json does not read).
fn extraVariants(a: std.mem.Allocator, io: std.Io, extra_dir: []const u8, function: []const u8) usize {
    var m = cuda.aot.Manifest.read(a, io, extra_dir) catch return 0;
    defer m.deinit();
    return m.variants(function);
}

/// TF_DS_HEAD_CAND `asked` in effect: on every rank or none (a rank can on the four-node 2D split; one world all-gather
/// when asked), the round's parts and the host's copy of them made, one line when asked (headCandLine).
fn useHeadCand(m: *Model, a: std.mem.Allocator, d: *const cuda.Driver, asked: bool) !void {
    m.head_cand = false;
    if (!asked) return;
    const why: ?[]const u8 = if (m.two == null) "needs the four-node 2D split" else null;
    const cannot = try prompt2d.ranksThatCannot(d, &m.comm, m.stream, why == null);
    if (prompt2d.agreedOn(asked, cannot)) {
        try headParts(m, a);
        m.head_cand = true;
    }
    var buf: [400]u8 = undefined;
    std.log.info("{s}", .{headCandLine(&buf, cannot, why)});
}

/// The round's parts of a head by the nodes' own columns and the host's copy of the four (once: both switches share
/// them).
fn headParts(m: *Model, a: std.mem.Allocator) !void {
    if (m.cand_parts.len != 0) return;
    try m.rd.useHeadCand(&m.eng, &m.arena);
    m.cand_parts = try a.alloc(u8, 4 * head_cand.layout(round.max_rows, head_cand.kFor(sampling.max_candidates)).bytes);
}

/// TF_DS_HEAD_KEY `asked` in effect: on every rank or none (a rank can on the four-node 2D split; one world all-gather
/// when asked), the parts made (headParts), one line when asked (headKeyLine).
fn useHeadKey(m: *Model, a: std.mem.Allocator, d: *const cuda.Driver, asked: bool) !void {
    m.head_key = false;
    if (!asked) return;
    const why: ?[]const u8 = if (m.two == null) "needs the four-node 2D split" else null;
    const cannot = try prompt2d.ranksThatCannot(d, &m.comm, m.stream, why == null);
    if (prompt2d.agreedOn(asked, cannot)) {
        try headParts(m, a);
        m.head_key = true;
    }
    var buf: [400]u8 = undefined;
    std.log.info("{s}", .{headKeyLine(&buf, cannot, why)});
}

/// TF_DS_HEAD_KEY's start line: on every rank, or off on every rank, the ranks that cannot and this rank's reason
/// (`why`, null: it can).
fn headKeyLine(buf: []u8, cannot: u32, why: ?[]const u8) []const u8 {
    if (cannot != 0) {
        var rb: [64]u8 = undefined;
        return std.fmt.bufPrint(buf, "TF_DS_HEAD_KEY=1: off on every rank: {s} cannot run it (this rank: {s})", .{ prompt2d.rankList(cannot, &rb), why orelse "can" }) catch buf[0..0];
    }
    return std.fmt.bufPrint(buf, "TF_DS_HEAD_KEY=1: a greedy round's tokens from each node's best of each row of its own columns of the head, merged on rank 0 into the baseline argmax; on every rank", .{}) catch buf[0..0];
}

/// TF_DS_HEAD_CAND's start line: on every rank, or off on every rank, the ranks that cannot and this rank's reason
/// (`why`, null: it can).
fn headCandLine(buf: []u8, cannot: u32, why: ?[]const u8) []const u8 {
    if (cannot != 0) {
        var rb: [64]u8 = undefined;
        return std.fmt.bufPrint(buf, "TF_DS_HEAD_CAND=1: off on every rank: {s} cannot run it (this rank: {s})", .{ prompt2d.rankList(cannot, &rb), why orelse "can" }) catch buf[0..0];
    }
    return std.fmt.bufPrint(buf, "TF_DS_HEAD_CAND=1: a sampled round's candidates from each node's own columns of the head (its first 32 or 64 of each row) merged on rank 0 into the baseline's; a round the baseline kernel could flag reads the whole head as the baseline; on every rank", .{}) catch buf[0..0];
}

/// TF_DS_MERGE_D as asked: unset, empty or 3 (16 columns a program: 32 programs a row and head group, measured the fastest of
/// the kit's widths), 1 (128), 2 (64), 0 (off); null for any other text (Model.open refuses the start with it: a typo must not
/// run the default unseen). Pure: the refusal is logged at the call site.
var merge_d_asked: usize = 0;
fn mergeDWanted(v: ?[]const u8) ?usize {
    const t = v orelse return 16;
    if (t.len == 0 or std.mem.eql(u8, t, "3")) return 16;
    if (std.mem.eql(u8, t, "0")) return 0;
    if (std.mem.eql(u8, t, "1")) return 128;
    if (std.mem.eql(u8, t, "2")) return 64;
    return null;
}

/// Which widths the loaded kit's extra set holds a _sparse_attn_merge_d for (a variant compiled with that DB).
const MergeHave = struct { c16: bool = false, c64: bool = false, c128: bool = false };
fn mergeDHave(a: std.mem.Allocator, io: std.Io, extra_dir: []const u8) MergeHave {
    var m = cuda.aot.Manifest.read(a, io, extra_dir) catch return .{};
    defer m.deinit();
    const f = "_sparse_attn_merge_d";
    return .{
        .c16 = m.has(f, &.{cuda.aot.ci("DB", 16)}),
        .c64 = m.has(f, &.{cuda.aot.ci("DB", 64)}),
        .c128 = m.has(f, &.{cuda.aot.ci("DB", 128)}),
    };
}

/// The width a kit that holds `have` runs for `asked`: the asked width when its extra set has it, else 128 (the first kits'
/// width), else 64, else null (none: the single-program merge).
fn mergeDResolve(asked: usize, have: MergeHave) ?usize {
    if (asked == 0) return null;
    const ok = switch (asked) {
        16 => have.c16,
        64 => have.c64,
        128 => have.c128,
        else => false,
    };
    if (ok) return asked;
    if (have.c128) return 128;
    if (have.c64) return 64;
    return null;
}

/// TF_DS_MERGE_D `asked` (the columns a program) in effect on this rank: a decode round's split merge from the kit's
/// twin _sparse_attn_merge_d when the extra set loaded (`loaded`; else `why` it did not), `variants` of its twins there
/// (the kit's two merge variants at each width), `have` the widths it holds. A kit without the asked width runs the next one
/// it has (128, then 64) and says so; one line either way; a launch without its twin keeps the baseline kernel.
fn useMergeD(asked: usize, loaded: bool, why: ?[]const u8, variants: usize, have: MergeHave) void {
    tri_attn.merge_db = 0;
    if (asked == 0) return;
    if (!loaded) return std.log.info("TF_DS_MERGE_D: off: the kit's extra set (aot-extra/) did not load ({s})", .{why orelse "none"});
    if (variants == 0) return std.log.info("TF_DS_MERGE_D: off: the loaded kit's extra set has no _sparse_attn_merge_d: kit a02466e or later", .{});
    const got = mergeDResolve(asked, have) orelse return std.log.info("TF_DS_MERGE_D: off: the loaded kit's extra set has no _sparse_attn_merge_d at 128, 64 or {d} columns a program", .{asked});
    // ("instead: " is the speed report's marker: the feature still runs, the report says which width)
    if (got != asked) std.log.warn("TF_DS_MERGE_D: instead: runs {d} columns a program, not {d}: the loaded kit's extra set has no _sparse_attn_merge_d at {d} columns (it needs the kit regenerated with that width)", .{ got, asked, asked });
    tri_attn.merge_db = got;
    std.log.info("TF_DS_MERGE_D: a decode round's split merge in programs of {d} columns (_sparse_attn_merge_d, the same bits); the extra set holds {d} twins in all widths", .{ got, variants });
}

fn useAttnUDecode(a: std.mem.Allocator, io: std.Io, asked: bool, loaded: bool, why: ?[]const u8, aot_dir: []const u8, extra_dir: []const u8) void {
    tri_attn.attn_u_decode = false;
    if (!asked) return;
    if (!loaded) return std.log.info("TF_DS_ATTN_U_DECODE=1: off: the kit's extra set (aot-extra/) did not load ({s})", .{why orelse "none"});
    tri_attn.attn_u_decode = true;
    var cov: ?[2]usize = null;
    if (cuda.aot.Manifest.read(a, io, aot_dir)) |b| {
        var base = b;
        defer base.deinit();
        if (cuda.aot.Manifest.read(a, io, extra_dir)) |x| {
            var extra = x;
            defer extra.deinit();
            cov = base.twinCoverage(&extra, "_sparse_attn_part", "_sparse_attn_part_u");
        } else |_| {}
    } else |_| {}
    if (cov) |n| {
        if (n[1] < n[0]) std.log.warn("TF_DS_ATTN_U_DECODE=1: partial twin coverage: only {d} of the kit's {d} _sparse_attn_part variants have their twin; a launch of any other runs the baseline kernel", .{ n[1], n[0] });
        std.log.info("TF_DS_ATTN_U_DECODE=1: a decode round's sparse attention with packed latents from _sparse_attn_part_u (the latents unpacked in hardware), {d} of the kit's {d} _sparse_attn_part variants with their twin; another keeps the baseline kernel", .{ n[1], n[0] });
    } else std.log.info("TF_DS_ATTN_U_DECODE=1: a decode round's sparse attention with packed latents from _sparse_attn_part_u (the latents unpacked in hardware); a variant without its twin keeps the baseline kernel", .{});
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
/// TF_DS_HC_LATE_FORK `asked` in effect on this rank's round, one line when asked: on, or off and why (the
/// checkpoint's own weights and the mixes' side stream). The same kernels on the same inputs either way.
fn useLateFork(rd: *round.Round, asked: bool, native: bool) void {
    if (!asked) return;
    const why: ?[]const u8 = if (!native) "needs the checkpoint's own FP8 / MXFP4 weights" else if (rd.side == null) "needs the mixes' side stream (TF_DS_HC_SIDE)" else null;
    if (why) |w| return std.log.info("TF_DS_HC_LATE_FORK=1: off: {s}", .{w});
    rd.useLateFork();
    std.log.info("TF_DS_HC_LATE_FORK=1: a decode mix's side work (the dots and the Sinkhorn) launched after the main stream's next launch, the router or the attention's input projections", .{});
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
        .idx_rb = @intCast(@min(tri_index.idx_rb, std.math.maxInt(u32))),
        .land_slots = o.land_slots,
        .exch = prompt2d.exchAsked(),
        .kern = prompt2d.kernAsked(),
        .kern2 = prompt2d.kern2Asked(),
        .pf_split = @backingInt(prompt2d.pf_split_asked),
        .pf_raster = prompt2d.pf_raster_asked,
        .pass_rows = prompt2d.pass_rows_asked,
        .pass_from = prompt2d.pass_from_asked,
    };
}

/// This rank's weight digest from TF_DS_RANK_CACHE_SHA (the SHA-256 of its weight file): none when it is unset, empty
/// or not a digest (said on stderr). The start goes on either way; without one the SSD tier is off, named.
pub fn rankDigest() prompt2d.Digest {
    const v = std.mem.span(std.c.getenv("TF_DS_RANK_CACHE_SHA") orelse return prompt2d.no_digest);
    if (v.len == 0) return prompt2d.no_digest;
    return prompt2d.weightDigest(v) orelse {
        std.log.warn("TF_DS_RANK_CACHE_SHA is not a SHA-256 (64 lowercase hex digits, \"sha256:\" before them or not): this rank sends no weight digest", .{});
        return prompt2d.no_digest;
    };
}

const EngramPhase = enum { all, pre, finish };

/// The device bytes of a second pass's chunk and node of the 2D split at 4,096 rows, MEASURED on a Spark (the pair line's
/// "chunk and node take", release with the stagger: 3,370,548,688). A pass's buffers are its rows' widths, so another
/// pass size takes this times its rows over 4,096 (ESTIMATED: the guard's floor below is the margin for a fixed part).
const pair_chunk_node_4096: u64 = 3_370_548_688;
/// What the guard keeps free for the rest of the engine (the host side of the pair, the round buffers, the cache growth).
const pair_guard_bytes: usize = 6 << 30;

/// The bytes a second pass takes on a rank at `pass_rows`: its chunk and node (scaled from the measured 4,096-row pass), the
/// second lane's transport arena (`lane_arena`: rocinante.layout().total, which follows the booted rows) and the control block.
pub fn pairEstimate(pass_rows: usize, lane_arena: u64) u64 {
    return (pair_chunk_node_4096 * pass_rows + 4095) / 4096 + lane_arena + @sizeOf(rocinante_mod.Control);
}

/// How many bytes short a start is of running two passes at `pass_rows` with `free` bytes free (0: it fits): the estimate and the
/// guard's floor both come off `free`.
pub fn pairShort(free: u64, pass_rows: usize, lane_arena: u64) u64 {
    return pairShortN(free, pass_rows, lane_arena, 2);
}

/// `pairShort` for `lanes` passes in flight (TF_DS_ROCI_STAGGER_LANES): every pass after the first takes the estimate.
pub fn pairShortN(free: u64, pass_rows: usize, lane_arena: u64, lanes: usize) u64 {
    const need = (lanes - 1) * pairEstimate(pass_rows, lane_arena) + pair_guard_bytes;
    return if (free >= need) 0 else need - free;
}

/// What TF_DS_ROCI_STAGGER_LANES comes to at the start: the lanes asked for (`want`, 2 or 3), whether the operator set it, and whether
/// every rank has the memory for them. A box under the floor refuses a lane count the operator set, by name; one that was only the
/// default runs two lanes and warns. Two lanes are never judged here: the pair's own guard runs as it always did.
pub const LanesChoice = struct { lanes: usize, warn: bool = false, refuse: bool = false };
pub fn lanesChoice(want: usize, set: bool, fits_all: bool) LanesChoice {
    if (want <= 2 or fits_all) return .{ .lanes = want };
    if (set) return .{ .lanes = want, .refuse = true };
    return .{ .lanes = 2, .warn = true };
}

/// Why a pool of `pool` positions cannot serve streams of `context` (null: it can): it must hold one stream's whole context, a full
/// request, or the context would be lowered without a word (a pool is the operator's setting; the context is never lowered).
pub fn poolBelowContext(pool: usize, context: usize) ?[]const u8 {
    if (context == 0 or pool >= context) return null;
    return "the KV pool (POOL: TF_DS_WINDOW on rank 0, --pool on the followers) holds fewer positions than one stream's context (--context): the pool must hold one full request, so the context is never lowered";
}

/// What a rank would give back of its KV pool to make room: the pool's caches (prompt.Caches) by the positions it holds. The
/// pool is the operator's setting (POOL in the recipe: TF_DS_WINDOW on rank 0, --pool on the followers, one value on every rank).
pub const PoolView = struct {
    c: *const Config,
    layers: []const weights.Layer,
    slots: usize, // the pool's streams
    pool: usize, // the positions it holds now
    floor: usize, // the fewest it may hold: one stream's whole context (a request is never cut)

    /// The device bytes of the pool at `positions`.
    pub fn bytes(v: PoolView, positions: usize) usize {
        return prompt.Caches.poolBytes(v.c, v.layers, positions, v.slots);
    }
};

/// The granularity of a pool the refusal names: a multiple of 65,536 positions.
pub const pool_step: usize = 65536;

/// The largest pool (a multiple of pool_step, below the current one) whose caches take at least `short` bytes less, and never one
/// below the view's floor; null when even the floor frees less than `short`.
pub fn poolThatFrees(v: PoolView, short: u64) ?usize {
    if (short == 0) return v.pool;
    const lo = std.mem.alignForward(usize, v.floor, pool_step);
    if (lo >= v.pool) return null;
    const have = v.bytes(v.pool);
    if (have - v.bytes(lo) < short) return null;
    // the freed bytes grow as the pool shrinks: the largest multiple whose freed bytes reach `short`
    var good = lo / pool_step; // (frees enough)
    var bad = (v.pool + pool_step - 1) / pool_step; // (the pool itself frees nothing)
    while (bad - good > 1) {
        const mid = good + (bad - good) / 2;
        if (have - v.bytes(mid * pool_step) >= short) good = mid else bad = mid;
    }
    return good * pool_step;
}

/// The words that end a start for want of free memory: what it needs, what is free, the GiB short, and the pool (POOL) that would
/// make it fit (or that none does), or a smaller TF_DS_PASS_ROWS. `what`: "the second pass" or "the pass"; `need` the bytes the
/// start wants free (the estimate and the floor kept together).
pub fn memoryRefusal(buf: []u8, what: []const u8, estimated: bool, pass_rows: usize, need: u64, free: u64, v: PoolView) ![]const u8 {
    const gib: f64 = 1 << 30;
    const short = if (need > free) need - free else 0;
    const head = try std.fmt.bufPrint(buf, "not enough free memory for {s} at TF_DS_PASS_ROWS={d}: {d:.2} GiB wanted free ({s}the {d:.1} GiB kept), {d:.2} GiB free, {d:.2} GiB short; ", .{ what, pass_rows, @as(f64, @floatFromInt(need)) / gib, if (estimated) "ESTIMATED from the 4,096-row pass measured, with " else "MEASURED after the start's allocations, ", @as(f64, @floatFromInt(pair_guard_bytes)) / gib, @as(f64, @floatFromInt(free)) / gib, @as(f64, @floatFromInt(short)) / gib });
    const tail = if (poolThatFrees(v, short)) |p|
        try std.fmt.bufPrint(buf[head.len..], "POOL={d} frees it (the KV pool holds {d} positions now, {d:.2} GiB on this Spark; a pool is the operator's setting, TF_DS_WINDOW on rank 0 and --pool on the followers, one value on every rank, and the smallest any rank names fits them all), or a smaller TF_DS_PASS_ROWS", .{ p, v.pool, @as(f64, @floatFromInt(v.bytes(v.pool))) / gib })
    else
        try std.fmt.bufPrint(buf[head.len..], "no POOL fits: the smallest pool that holds one full request ({d} positions) frees {d:.2} GiB of the {d:.2} GiB short (a smaller TF_DS_PASS_ROWS, or more free memory)", .{ std.mem.alignForward(usize, v.floor, pool_step), @as(f64, @floatFromInt(v.bytes(v.pool) -| v.bytes(std.mem.alignForward(usize, v.floor, pool_step)))) / gib, @as(f64, @floatFromInt(short)) / gib });
    return buf[0 .. head.len + tail.len];
}

/// The device memory free now (the driver's count; 0 when it cannot say): the ledger's measure of what an allocation took.
fn freeNow(ctx: *const cuda.Context) u64 {
    const i = ctx.memInfo() catch return 0;
    return i.free;
}

/// The KV pool's bytes at a number of positions, for the vote's fit (prompt.Caches.poolBytes over this rank's layers).
const PoolBytes = struct {
    m: *const Model,
    c: *const Config,

    pub fn bytes(s: PoolBytes, positions: usize) usize {
        return prompt.Caches.poolBytes(s.c, s.m.w.layers, positions, s.m.streams);
    }
};

/// TF_DS_KV_VMM=1: the pool's planes in virtual memory. The reservation is sized for the most positions the device could ever
/// hold (every plane's stretch at that many), the first `pool` positions of each plane are backed, and the start line says what
/// was reserved and mapped. The bytes, the order and the layout of the planes are the arena's (Caches.init): only where they live differs.
fn openVmmCaches(m: *Model, ctx: *const cuda.Context, c: *const Config, pool: usize) !prompt.Caches {
    const info = try ctx.memInfo();
    var be: kv_vmm.CudaBackend = .{ .d = ctx.d, .device = ctx.device };
    const gran: u64 = try be.granularity();
    const most = @max(pool, kv_vmm.maxPositions(PoolBytes{ .m = m, .c = c }, info.total, engine_api.kv_pool.pool_step));
    const va = prompt.Caches.vaBytes(c, m.w.layers, most, m.streams, @intCast(gran));
    m.kv_vmm = try kv_vmm.Driver.open(m.gpa, be, gran, kv_vmm.seg_default, va);
    errdefer {
        m.kv_vmm.?.deinit();
        m.kv_vmm = null;
    }
    const cs = try prompt.Caches.initVmm(&m.eng, &m.kv_vmm.?, pool, most, m.streams);
    m.vmm_most = most;
    const v = &m.kv_vmm.?;
    std.log.info("TF_DS_KV_VMM=1: rank {d}: the KV pool's {d} planes are in a virtual-memory reservation of {d:.2} GiB (room for {d} positions, the device's total), {d:.2} GiB mapped in segments of {d} MiB (granularity {d} KiB; the planes' bytes {d:.2} GiB, rounded up per plane to the granularity); the planes' addresses never change, so the graphs' stay valid", .{ m.rank, v.planes.items.len, @as(f64, @floatFromInt(v.va_bytes)) / (1 << 30), most, @as(f64, @floatFromInt(v.mapped)) / (1 << 30), kv_vmm.seg_default >> 20, gran >> 10, @as(f64, @floatFromInt(v.asked)) / (1 << 30) });
    return cs;
}

/// The pool this rank allocates, from the ranks' vote (kv_pool.zig): every rank offers the device memory it has free now, its
/// free target and growth, the positions that fits and the ledger of what its start has taken, over the host star; every rank reads
/// the same table and decides the same. With no free target (TF_DS_KV_FREE_TARGET_GIB) each rank keeps its own pool, as before the
/// vote; with one the pool is the smallest any rank fits, never below the context, and a start that cannot hold the context ends on
/// every rank alike, naming the shortfall and the ledger's largest lines.
fn votePool(m: *Model, o: Options, c: *const Config) !usize {
    const kv = engine_api.kv_pool;
    const info = try m.ctx.memInfo();
    const lin: PoolBytes = .{ .m = m, .c = c };
    const floor: usize = if (o.context != 0) o.context else o.pool;
    var mine = kv.Part.init(m.rank);
    mine.free_bytes = info.free;
    mine.total_bytes = info.total;
    mine.target_bytes = o.kv_target_bytes;
    const growth = kv.growthForPass(m.rank, o.kv_growth_set, o.kv_growth_bytes, m.pass_rows); // larger passes: the pool pays (D98)
    mine.growth_bytes = growth;
    mine.growth_set = @intFromBool(o.kv_growth_set);
    mine.floor = floor;
    mine.asked = o.pool;
    mine.fit = if (o.kv_target_bytes != 0) kv.fit(info.free, o.kv_target_bytes, growth, lin) else 0;
    mine.floor_bytes = lin.bytes(floor);
    mine.b_lo = lin.bytes(kv.pool_step);
    mine.b_hi = lin.bytes(17 * kv.pool_step);
    mine.bytes_per_token = (mine.b_hi - mine.b_lo) / (16 * kv.pool_step);
    var snap = m.ledger; // (a copy: the remainder this adds is the vote's, the running ledger closes at the end)
    snap.close(@src(), info.total -| info.free);
    snap.fill(&mine);
    var table: [4]kv.Part = undefined;
    const world: usize = m.world;
    const star = m.comm.host.star orelse return error.NoHostLinks;
    try star.allGather(std.mem.asBytes(&mine), std.mem.sliceAsBytes(table[0..world]));
    const d = kv.decide(table[0..world]);
    kv.report.world = m.world;
    kv.report.table = table;
    kv.report.decision = d;
    if (d.verdict == .refused) {
        var buf: [1600]u8 = undefined;
        std.log.err("rank {d}: {s}", .{ m.rank, kv.refusalText(&buf, table[0..world], d) });
        return error.PoolRefused;
    }
    const pool: usize = if (d.verdict == .auto) d.pool else o.pool;
    kv.report.pool_active = pool;
    kv.report.streams = @intCast(m.streams);
    var fits: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&fits);
    for (table[0..world], 0..) |p, i| w.print("{s}{d}", .{ if (i > 0) ", " else "", p.fit }) catch {};
    if (d.verdict == .auto) {
        std.log.info("KV pool: rank {d}: {d} positions, the smallest any rank fits (rank {d}); the ranks fit {s} with {d:.2} GiB kept free (TF_DS_KV_FREE_TARGET_GIB) and {d:.2} GiB left for what follows the pool (TF_DS_KV_GROWTH_MIB: {s}); this rank has {d:.2} GiB free before the pool", .{ m.rank, pool, d.limiting_rank, w.buffered(), @as(f64, @floatFromInt(o.kv_target_bytes)) / (1 << 30), @as(f64, @floatFromInt(growth)) / (1 << 30), if (o.kv_growth_set) "the operator's number" else kv.growth_measured, @as(f64, @floatFromInt(info.free)) / (1 << 30) });
    } else {
        std.log.info("KV pool: rank {d}: {d} positions, the pool the settings name (no TF_DS_KV_FREE_TARGET_GIB); this rank has {d:.2} GiB free before the pool and the pool takes about {d:.2} GiB", .{ m.rank, pool, @as(f64, @floatFromInt(info.free)) / (1 << 30), @as(f64, @floatFromInt(lin.bytes(pool))) / (1 << 30) });
    }
    return pool;
}

/// What the pool took, measured, beside what its layout says it takes.
fn logPoolTaken(m: *const Model, c: *const Config, taken: u64) void {
    const predicted = (PoolBytes{ .m = m, .c = c }).bytes(m.pool_cap);
    engine_api.kv_pool.report.pool_taken = taken;
    engine_api.kv_pool.report.pool_predicted = predicted;
    std.log.info("KV pool: rank {d}: {d} positions took {d:.2} GiB of device memory (MEASURED by the driver's free count), the caches' layout takes {d:.2} GiB (the arena rounds up to its blocks)", .{ m.rank, m.pool_cap, @as(f64, @floatFromInt(taken)) / (1 << 30), @as(f64, @floatFromInt(predicted)) / (1 << 30) });
}

/// At the start's end: every rank's whole ledger to every rank over the host star, the remainder (the CUDA context, the kernels,
/// the graphs) named, and the start's own line saying what it took after the pool: the growth TF_DS_KV_GROWTH_MIB is to cover.
fn gatherLedger(m: *Model, free_after_pool: u64) !void {
    const kv = engine_api.kv_pool;
    const info = try m.ctx.memInfo();
    m.ledger.close(@src(), info.total -| info.free);
    var mine = kv.Part.init(m.rank);
    mine.free_bytes = info.free;
    mine.total_bytes = info.total;
    m.ledger.fill(&mine);
    var table: [4]kv.Part = undefined;
    const world: usize = m.world;
    const star = m.comm.host.star orelse return error.NoHostLinks;
    try star.allGather(std.mem.asBytes(&mine), std.mem.sliceAsBytes(table[0..world]));
    kv.report.final = table;
    kv.report.have_final = true;
    kv.report.mine = m.ledger;
    kv.report.after_taken = free_after_pool -| info.free;
    kv.report.free_ready = info.free;
    kv.ready.store(true, .release);
    std.log.info("KV pool: rank {d}: the start took {d:.2} GiB after the pool (MEASURED: TF_DS_KV_GROWTH_MIB is the room kept for this, and for what serving grows by); {d:.2} GiB of {d:.2} GiB are free at its end", .{ m.rank, @as(f64, @floatFromInt(kv.report.after_taken)) / (1 << 30), @as(f64, @floatFromInt(info.free)) / (1 << 30), @as(f64, @floatFromInt(info.total)) / (1 << 30) });
}

/// The pool the start holds, for the refusals above.
fn poolView(m: *const Model, c: *const Config) PoolView {
    return .{ .c = c, .layers = m.w.layers, .slots = m.streams, .pool = m.pool_cap, .floor = m.ctx_cap };
}

/// The refusal line of a start that would leave less than the guard's floor, naming the setting, the GiB short and the pool that fits.
fn pairShortText(buf: []u8, free: u64, pass_rows: usize, lane_arena: u64, v: PoolView, lanes: usize) ![]const u8 {
    const what = if (lanes >= 3) "the second and third passes (TF_DS_ROCI_STAGGER_LANES=3)" else "the second pass";
    return memoryRefusal(buf, what, true, pass_rows, (lanes - 1) * pairEstimate(pass_rows, lane_arena) + pair_guard_bytes, free, v);
}

/// Passes above this many rows check the floor of a single pass (the stagger checks its own, whatever the size): 4,096 rows is the
/// largest the start never checked.
const single_floor_above: usize = 4096;

/// What passes of more than 4,096 rows leave of a Spark's memory, checked once the start has allocated everything (an 8,192-row
/// pass took 3.2-3.9 GiB of MemAvailable more than a 4,096-row one, measured; the pass's buffers are its rows' widths): free
/// memory under the guard's floor ends the start, naming the setting, the GiB short and the pool (POOL) that fits, on every rank
/// alike. The stagger checks its own, before it allocates.
fn passFloor(m: *Model, c: *const Config) !void {
    if (m.pass_rows <= single_floor_above or m.pair) return;
    const free = (try m.ctx.memInfo()).free;
    const ok = free >= pair_guard_bytes;
    if (!try m.eng.agreeAll(ok)) {
        if (!ok) {
            var buf: [900]u8 = undefined;
            std.log.err("{s}", .{memoryRefusal(&buf, "the pass", false, m.pass_rows, pair_guard_bytes, free, poolView(m, c)) catch "not enough free memory for the pass at TF_DS_PASS_ROWS"});
        } else std.log.err("TF_DS_PASS_ROWS={d}: another rank has too little free memory for it (its line names the GiB short and the pool that fits)", .{m.pass_rows});
        return error.PassRowsRefused;
    }
    std.log.info("TF_DS_PASS_ROWS={d}: rank {d}: {d:.2} GiB free after the start's allocations (the floor is {d:.1} GiB); the KV pool holds {d} positions, {d:.2} GiB on this Spark", .{ m.pass_rows, m.comm.rank, @as(f64, @floatFromInt(free)) / (1 << 30), @as(f64, @floatFromInt(pair_guard_bytes)) / (1 << 30), m.pool_cap, @as(f64, @floatFromInt(poolView(m, c).bytes(m.pool_cap))) / (1 << 30) });
}

/// TF_DS_ENGRAM_PARTNER's start: the column partners (nodes g and g ^ 2) share the reads of a prompt step's misses over a third Rocinante lane. Every
/// rank's checks agree before anything is allocated; a rank that cannot run it ends the start naming its reason.
fn openPartner(m: *Model, gpa: std.mem.Allocator, d: *const cuda.Driver, o: Options, sp: plan.Split) !void {
    var why: ?[]const u8 = null;
    const two = if (m.two) |*t| t else null;
    if (sp.pair == null or two == null or two.?.rc == null) {
        why = "needs the four-node 2D split with Rocinante";
    } else if (m.eh == null or !wk_on or m.erows == null) {
        why = "needs the Engram worker with packed shards (TF_DS_ENGRAM_PACKED)";
    } else if (m.erows.?.owned) {
        why = "needs whole tables on every node (TF_DS_ENGRAM_OWNED=0)";
    } else if (m.erows.?.rec != @import("rocinante.zig").host_rec or m.erows.?.slots() * chunk_rows > 2 * @import("rocinante.zig").host_max_recs) {
        why = "needs 264-byte records and at most 12 columns a rank (the host lane's slots are sized for them)";
    } else if (o.rdma_devices == null) {
        why = "needs the rails' devices (TF_DS_RDMA_DEVICES)";
    }
    if (!try m.eng.agreeAll(why == null)) {
        switch (speed_defaults.verdict(why orelse "another rank cannot run it", speed_defaults.defaulted("TF_DS_ENGRAM_PARTNER"))) {
            .refuse => {
                if (why) |w| std.log.err("TF_DS_ENGRAM_PARTNER=1: off: {s}", .{w}) else std.log.err("TF_DS_ENGRAM_PARTNER=1: off: another rank cannot run it (its line names the reason)", .{});
                return error.PartnerRefused;
            },
            // the default: one WARN naming why and what it costs; every rank decided the same (all agreed above), no lane was opened
            else => {
                std.log.warn("TF_DS_ENGRAM_PARTNER: off: {s}; without it: {s} (set TF_DS_ENGRAM_PARTNER=1 to make this a refusal, 0 to silence it)", .{ why orelse "another rank cannot run it (its line names the reason)", speed_defaults.costOf("TF_DS_ENGRAM_PARTNER") });
                prompt2d.engram_partner_asked = false;
                return;
            },
        }
    }
    var devs_it = std.mem.splitScalar(u8, o.rdma_devices.?, ',');
    const rails: [2][]const u8 = .{ devs_it.next() orelse return error.RociRails, devs_it.next() orelse return error.RociRails };
    const node: u8 = @intCast(m.two.?.node());
    const rc = try roci_engine.Roci.openLane(gpa, d, m.comm.rank, node, m.comm.world, m.two.?.rank_of, &m.fan, rails, Model.rdmaGid() orelse 3, 1, m.two.?.rc.?, .host);
    errdefer rc.close();
    const plink = try m.gpa.create(engram_partner.Link);
    errdefer m.gpa.destroy(plink);
    plink.* = .{ .t = rc.t, .arena = rc.arena.bytes, .role = @intCast(node >> 1), .gpa = m.gpa };
    m.partner_rc = rc;
    m.partner_link = plink;
    // TF_DS_ENGRAM_PARTNER_SPLIT=0 (every rank alike): the lane is open and nothing crosses it, to tell what the open lane costs from what the split costs
    const split = if (std.c.getenv("TF_DS_ENGRAM_PARTNER_SPLIT")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    if (split) m.erows.?.partner = plink;
    std.log.info("TF_DS_ENGRAM_PARTNER=1: rank {d} (node {d}): {s}; records are {d} bytes, {d} slots a ring, the host lane's arena {d} bytes", .{ m.comm.rank, node, if (!split) "the host lane is open and the Engram worker does not use it (TF_DS_ENGRAM_PARTNER_SPLIT=0)" else if (node >> 1 == 0) "the Engram worker reads the even positions of each prompt step's misses and takes the other positions from node 2 over the host lane (a header first; the partner waited for as long as this node's read took, then the rest read here)" else "the Engram worker reads the odd positions of each prompt step's misses and takes the other positions from node 0 over the host lane (a header first; the partner waited for as long as this node's read took, then the rest read here)", @import("rocinante.zig").host_rec, @import("rocinante.zig").host_slots, rc.layout.total });
}

/// TF_DS_ROCI_STAGGER's start: two whole passes in flight need every exchange of a full pass on the transport (so no collective of
/// another communicator runs beside a stream's memop waits), the Engram uploads from pinned memory (a pageable copy blocks the host thread
/// that issues both passes), a second chunk, node of the split, lane, stream, cuBLAS handle and the per-layer cache events. Every rank's
/// checks agree before anything is allocated; a rank that cannot run the route ends the start naming its reason.
fn openPair(m: *Model, gpa: std.mem.Allocator, d: *const cuda.Driver, o: Options, sp: plan.Split, c: *const Config) !void {
    var why_buf: [900]u8 = undefined;
    var why: ?[]const u8 = null;
    var free: usize = 0;
    const two = if (m.two) |*t| t else null;
    const connections: ?usize = if (std.c.getenv("CUDA_DEVICE_MAX_CONNECTIONS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch null else null;
    const pinned = pinned_on and wk_on;
    if (sp.pair == null or two == null) {
        why = "needs the four-node 2D split";
    } else if (c.format != .native) {
        why = "needs the checkpoint's own FP8 / MXFP4 weights";
    } else if (m.pass_rows % chunk_rows != 0 or m.pass_rows > pass_rows_max) {
        why = "needs whole passes of whole resume units (TF_DS_PASS_ROWS a whole number of 2048-row units, 2048 to 16384)";
    } else if (m.pass_rows < 2 * chunk_rows) {
        why = "needs passes of two resume units (TF_DS_PASS_ROWS=4096)";
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
        const lane_arena = rocinante_mod.layoutOf(.prompt).total;
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
            .refuse => {
                if (why) |w| std.log.err("TF_DS_ROCI_STAGGER=1: off: {s}", .{w}) else std.log.err("TF_DS_ROCI_STAGGER=1: off: another rank cannot run it (its line names the reason)", .{});
                return error.StaggerRefused;
            },
            // the default: one WARN naming why and what it costs, the single pass runs (every rank decided the same, all agreed above)
            else => {
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
    if (m.erows) |r| r.dup_on = m.pair_log;
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

    /// The next nap in microseconds (0: the setting's slice); `backoff` doubles it each empty wake up to issue_lead.backoff_cap_us.
    cur_us: u32 = 0,
        if (c.cur_us == 0) c.cur_us = c.m.ilead_slice_us;
        const ts: std.c.timespec = .{ .sec = 0, .nsec = @as(isize, c.cur_us) * std.time.ns_per_us };
        if (c.m.ilead_wait == .backoff) c.cur_us = issue_lead.backoffNext(c.cur_us, issue_lead.backoff_cap_us);
    return where.report(buf);
/// One line of TF_DS_ROCI_PAIR_LOG: to the log, and appended to the file the variable names when its value is a path.
fn pairLog(comptime fmt: []const u8, args: anytype) void {
    std.log.info(fmt, args);
    const v = std.c.getenv("TF_DS_ROCI_PAIR_LOG") orelse return;
    if (v[0] != '/') return;
    var buf: [2048]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt ++ "\n", args) catch return;
    const fd = std.c.open(v, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    _ = std.c.write(fd, line.ptr, line.len);
    _ = std.c.close(fd);
}

/// The most ranges either list holds: the worker's, with every slot of the ring (max_wk): the three scratch arrays, a hash
/// buffer and two rows a slot, and three ranges for each of a slot's two record stages. The round's list is shorter (11).
/// Sized by `engram_rows.max_wk`, so a wider ring grows the list with it; a list that is still full is an error, never a
/// silent drop (a range not listed is one the overlap check calls "no shared byte" without having seen it).
const keep_ahead_ranges = 3 + 9 * engram_rows.max_wk;

/// Why keepRanges cannot list every region.
const KeepRangesError = error{KeepAheadRangesFull};

/// The memory a round writes through the Engram host (`round`) and the memory the fill's worker writes (`worker`), as address ranges.
fn keepRanges(eh: *const prompt.EngramHost, round_r: *[keep_ahead_ranges]keep_ahead.Range, worker_r: *[keep_ahead_ranges]keep_ahead.Range) KeepRangesError!void {
    var nr: usize = 0;
    var nw: usize = 0;
    const R = struct {
        fn add(list: *[keep_ahead_ranges]keep_ahead.Range, n: *usize, r: keep_ahead.Range) KeepRangesError!void {
            if (n.* >= list.len) return error.KeepAheadRangesFull;
            list[n.*] = r;
            n.* += 1;
        }
    };
    try R.add(round_r, &nr, keep_ahead.rangeOf(i64, eh.hashes));
    try R.add(round_r, &nr, keep_ahead.rangeOf(i64, eh.flat));
    try R.add(round_r, &nr, keep_ahead.rangeOf(u8, eh.w));
    try R.add(round_r, &nr, keep_ahead.rangeOf(u8, eh.s));
    try R.add(round_r, &nr, keep_ahead.rangeOf(u16, eh.rows));
    try R.add(worker_r, &nw, keep_ahead.rangeOf(i64, eh.wk_flat));
    try R.add(worker_r, &nw, keep_ahead.rangeOf(u8, eh.wk_w));
    try R.add(worker_r, &nw, keep_ahead.rangeOf(u8, eh.wk_s));
    for (eh.wk_hashes) |h| try R.add(worker_r, &nw, keep_ahead.rangeOf(i64, h));
    for (eh.wk_rows) |row| for (row) |x| try R.add(worker_r, &nw, keep_ahead.rangeOf(u16, x));
    if (eh.pk) |pr| {
        for (&pr.round) |*st| {
            try R.add(round_r, &nr, keep_ahead.rangeOf(u8, st.bytes));
            try R.add(round_r, &nr, keep_ahead.rangeOf(i64, st.ids));
            try R.add(round_r, &nr, keep_ahead.rangeOf([*]u8, st.dsts));
        }
        nw = try workerStageRanges(worker_r, nw, &pr.wk, pr.wk_n);
    }
}

/// The worker's record stages as ranges, appended to `list` after its first `n0`; returns the new count. Only the first `wk_n` slots
/// exist (2 without the stagger, max_wk with it): the others are not the worker's memory.
fn workerStageRanges(list: *[keep_ahead_ranges]keep_ahead.Range, n0: usize, wk: *const [engram_rows.max_wk][2]engram_rows.Stage, wk_n: usize) KeepRangesError!usize {
    var n = n0;
    for (wk[0..wk_n]) |*row| for (row) |*st| {
        for ([_]keep_ahead.Range{ keep_ahead.rangeOf(u8, st.bytes), keep_ahead.rangeOf(i64, st.ids), keep_ahead.rangeOf([*]u8, st.dsts) }) |r| {
            if (n >= list.len) return error.KeepAheadRangesFull;
            list[n] = r;
            n += 1;
        }
    };
    return n;
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
    ledger: engine_api.kv_pool.Ledger, // what this start has taken, line by line (kv_pool.zig)
    ctx_cap: usize, // a stream's extent at most (Options.context, else the pool)
    pass_rows: usize = chunk_rows, // a pass's rows at most (TF_DS_PASS_ROWS, every rank alike): the chunk's allocation
    cache_dir: std.Io.Dir,
    cache_file: std.Io.File,
    ix: rank_cache.Index,
    w: weights.Weights,
    fan: prompt2d.Fan,
    vfan: prompt2d.Fan, // the host collectives' links (prompt2d.hostLinksAsked), else none
    watch: link.Watch, // watchPeers: a peer rank that stops stops this one too
    star_busy: std.atomic.Value(bool), // star.Star's guard against two threads in one collective
    comm: comm_mod.Comm,
    rdma_mod: ?cuda.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*rdma.Ring = null,
    set: cuda.aot.Set,
    extra: ?cuda.aot.Set = null, // the kit's extra set (TF_DS_DRAFT_4WAY's stages)
    pf: exl3_prefill.Kernels,
    lg: exl3_linear.Kernels,
    exk: exl3_experts.Kernels,
    ops: ops_mod.Ops,
    ex: exact.Exact,
    par: ?exl3_experts2d.ParKernels, // the parity split's one-launch gate / up (parOne)
    mxk: ?mx.Kernels = null, // Format.native: DeepSeek's own FP8 / MXFP4 checkpoint's kernels (mx.zig)
    rope_buf: cuda.DeviceBuffer,
    arena: prompt.Arena,
    vmm_most: usize = 0, // the positions the VMM reservation has room for
    kv_vmm: ?kv_vmm.Driver = null, // the pool's reservation (TF_DS_KV_VMM=1); the caches' planes live in it
    pool_start: usize = 0, // the pool the start took (a live free target of 0 returns to it)
    pool_ceiling: usize = 0, // the most positions a live resize may take: the reservation's, or the start's when the SSD tier's tables are sized by it
    kv_start_target_mib: u32 = 0, // the start's free target (a live `unset` returns to it; 0: the start had none)
    kv_live_loading: u32 = 0, // rank 0, with it: the disk restores still loading into the pool
    kv_live_floor: usize = 0, // rank 0, set by the lane host before a cfg frame: the end of the highest extent a stream or a kept state holds
    eng: prompt.Engine,
    ch: prompt.Chunk,
    caches: prompt.Caches,
    blas: cublas.Blas,
    rings: []u64, // each layer's window rings, max_streams slots
    ring_view: []u64, // a filling stream's slot of them
    rd: round.Round,
    graphs: ?round.Graphs = null,
    pgraphs: ?draft.PassGraphs = null, // the drafter pass as CUDA graphs (with graphs)
    graph_fallbacks: std.atomic.Value(u64) = .init(0), // either kind's work run eagerly after a failed capture (/metrics)
    vocab_stats: api.DraftVocabCounters = .{}, // TF_DS_DRAFT_VOCAB: its passes and guard on rank 0 (/metrics)
    dpool: ?draft.Pool = null,
    dr: ?draft.Drafter = null,
    tables: ?engram_io.Tables = null,
    epool: ?*engram_io.Pool = null,
    aio: ?*engram_aio.Aio = null,
    paio: ?*engram_aio.Aio = null, // the prompt chunks' Engram reads, on a ring of their own
    eh: ?prompt.EngramHost = null,
    erows: ?*engram_rows.Rows = null, // TF_DS_ENGRAM_PACKED: Engram's rows from packed shards (engram_rows.zig)
    host_pos: []i64,
    ids64: []i64,
    vocab: usize,
    logits: []f32, // round.max_rows rows of logits on the host
    amax: u64, // u32 [round.max_rows]: a round's greedy tokens on the device (roundArgmax)
    amax_host: [round.max_rows]u32 = undefined,
    // Track B (TF_DS_SAMPLE_CAND=1): a round's candidates on the device (vals f32, ids i64, flags i32; max_rows rows of
    // sampling.max_candidates) and on the host
    cand: u64 = 0,
    cand_vals: [round.max_rows * sampling.max_candidates]f32 = undefined,
    cand_ids: [round.max_rows * sampling.max_candidates]i64 = undefined,
    cand_flags: [round.max_rows]i32 = undefined,
    // TF_DS_HEAD_CAND, agreed at the start (useHeadCand): a sampled round's candidates from the four nodes' parts
    // (rd.call), read back into cand_parts and merged here (headCandidates); TF_DS_HEAD_KEY (useHeadKey): a greedy
    // round's tokens from the nodes' bests the same way (headTokens)
    head_cand: bool = false,
    head_key: bool = false,
    cand_parts: []u8 = &.{},
    // the stream a fill is running: its slot and its view of the caches (its extent from base)
    fill_slot: usize = 0,
    // the prompt a fill's chunks come from (the caller's, set for the fill) and the thread reading the next chunk's
    // Engram rows during this one (fillChunk), joined before the prompt can change (joinAhead)
    fill_seq: []const i32 = &.{},
    ahead: ?std.Thread = null,
    view: prompt.Caches = undefined,
    // the filling prompt's image spans and their rows (setImages; cleared by fillBegin)
    spans: std.ArrayList(prompt.Span) = .empty,
    img_buf: ?cuda.DeviceBuffer = null,
    // profiling (lanes' --profile): the last round's forward and absorb, synchronized apart (ns)
    prof: bool = false,
    cand_first: bool = false, // TF_DS_CAND_FIRST: the leader's verifyHead (a round's head reads before the absorb)
    head_ready: ?cuda.Event = null, // verifyHead's event after the head reads
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
    keep_ahead: bool = false, // TF_DS_ENGRAM_KEEP_AHEAD in effect: a round between chunks leaves the Engram worker running (keep_ahead.zig)
    keep_ahead_start: bool = false, // what the start gave it (a live `unset` returns to it)
    keep_ahead_able: bool = false, // the start verified it can run (an Engram worker, its own reader rings, no shared scratch): a live change may turn it on
    pair_log: bool = false, // TF_DS_ROCI_PAIR_LOG: one line a pair (a value starting with / also appends them to that file on rank 0)
    pair_gap: usize = roci_pair.gap,
    pair_prio: roci_pair.Prio = .equal, // TF_DS_ROCI_STAGGER_PRIO
    pair_major: bool = false, // TF_DS_ROCI_WORKER_PAIR_MAJOR
    partner_rc: ?*roci_engine.Roci = null, // TF_DS_ENGRAM_PARTNER: the host lane to the column partner, and the worker's side of the exchange
    partner_link: ?*engram_partner.Link = null,
    partner_seen: engram_partner.Counters = .{}, // the partner counters at the last line of the pair log
    fill_t0: u64 = 0, // the fill's admission (fillBegin), for the pair log's first line of a prompt
    pair_in_fill: u64 = 0, // pairs run since fillBegin
    dup_snap: [2]u64 = .{ 0, 0 }, // the row stages' miss totals at the last pair line, and at the fill's start (the repeats' share per line and per fill)
    dup_fill: [2]u64 = .{ 0, 0 },
    steps_seen: u64 = 0, // the row stages' step records already printed
    other_seen: u64 = 0,
    pair_count: u64 = 0,
    pair_prev_pending: bool = false, // the last pair's GPU span has not been printed
    meter: pass_meter.Meter = .{}, // the rows the device has finished of each slot's prompt pass (the lane host's live prefill speed)
    ids64_b: []i64 = &.{},
    host_pos_b: []i64 = &.{},
    ids64_c: []i64 = &.{},
    host_pos_c: []i64 = &.{},
    pair_bytes: usize = 0,
    rdma_rings: ?ring2d.Rings = null,
    // the start-up exchange (prompt2d.Fan.commChecked): rank 0's Settings (rank 0: its own), and the ranks' weight
    // digests by rank (rank 0 holds every rank's, a follower its own; prompt2d.no_digest: none)
    lead: prompt2d.Settings = undefined,
    weights: [4]prompt2d.Digest = @splat(prompt2d.no_digest),

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
        m.cand_parts = &.{};
        m.head_cand = false;
        m.head_key = false;
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
        prompt2d.pack_epilogue_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_PACK_EPILOGUE")) |v| std.mem.span(v) else null); // every rank alike (Settings)
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
        prompt2d.engram_partner_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ENGRAM_PARTNER")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        {
            const text: ?[]const u8 = if (std.c.getenv("TF_DS_SSD_FAULT")) |v| std.mem.span(v) else null;
            prompt2d.ssd_fault = spill_sync.Fault.parse(text, o.world) catch {
                std.log.err("TF_DS_SSD_FAULT is write:R, corrupt:R or write:R,corrupt:S with each rank below {d}, not \"{s}\"", .{ o.world, text.? });
                return error.BadSsdFault;
            };
            if (prompt2d.ssd_fault.any()) {
                var fb: [48]u8 = undefined;
                std.log.info("TF_DS_SSD_FAULT={s}: the four stores' drill: rank {d} carries out {s}{s}", .{ prompt2d.ssd_fault.text(&fb), o.rank, if (prompt2d.ssd_fault.armed(o.rank).write) "the write fault " else "", if (prompt2d.ssd_fault.armed(o.rank).corrupt) "the corrupt fault" else if (!prompt2d.ssd_fault.armed(o.rank).write) "none (another rank's)" else "" });
            }
        }
        prompt2d.pf_split_asked = readPfSplit(); // every rank alike (Settings)
        prompt2d.pf_raster_asked = try readPfRaster(); // every rank alike (Settings)
        prompt2d.woa_w8_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_WOA_W8")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.quant_once_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_QUANT_ONCE")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.pf_gemm5_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_PF_GEMM5")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.group_tiles_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_GROUP_TILES")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.topk_tiles_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_TOPK_TILES")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.idx_proj_owned_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_IDX_PROJ_OWNED")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.bf16_rows_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_BF16_ROWS")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.pf_run128_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_PF_RUN128")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.pass_from_asked = readPassFrom(); // rank 0's (Settings.pass_from: never compared)
        prompt2d.pass_rows_asked = try readPassRows(prompt2d.pass_from_asked, (ctx.memInfo() catch cuda.Context.MemInfo{ .free = 0, .total = std.math.maxInt(usize) }).total); // every rank alike (Settings)
        prompt2d.pass_beside_asked = try readPassBeside(prompt2d.pass_rows_asked); // rank 0's host decides it (never compared)
        rocinante_mod.setCap(prompt2d.pass_rows_asked); // (the transport's arenas and plans follow the pass rows, before any layout is read)
        prompt2d.rounds_beside_asked = try readRoundsBeside(); // rank 0's host runs the rounds (never compared)
        prompt2d.pass_tail_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_PASS_TAIL")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.readPassFillingEnv(); // rank 0's TF_DS_PASS_FILLING (the pass sizes are its frames'; its start line prints here)
        try prompt2d.readPromptOrderEnv(); // rank 0's TF_DS_PROMPT_ORDER and TF_DS_PROMPT_QOS (the host's prompt-step order; its start line prints here)
        prompt2d.decode_fuse_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_DECODE_FUSE")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.mma_sf_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_MMA_SF")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.mma_sf_exact_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_MMA_SF_EXACT")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.gemm_w16_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_GEMM_W16")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.split64n_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_SPLIT64N")) |v| std.mem.span(v) else null); // this rank's launches alone: its bytes are the 128-column tile's
        prompt2d.pf_t64_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_PF_T64")) |v| std.mem.span(v) else null); // this rank's launches alone: its bytes are the 128 x 128 tile's
        prompt2d.gemm_r2_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_GEMM_R2")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.draft_4way_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_DRAFT_4WAY")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.draft_main_4way_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_DRAFT_MAIN_4WAY")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.hc_late_fork_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_HC_LATE_FORK")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.attn_u_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ATTN_U")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.hc_fused_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_HC_FUSED")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.attn_epi_fused_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ATTN_EPI_FUSED")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.hc_2s_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_HC_2S")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.idx_u_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_IDX_U")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.idx_v_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_IDX_V")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.attn_u_decode_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ATTN_U_DECODE")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        const merge_text: ?[]const u8 = if (std.c.getenv("TF_DS_MERGE_D")) |v| std.mem.span(v) else null; // each rank its own: the bytes are the same
        merge_d_asked = mergeDWanted(merge_text) orelse {
            std.log.err("TF_DS_MERGE_D={s}: not one of 0 (off), 1 (128 columns a program), 2 (64), 3 (16, the default); not started", .{merge_text.?});
            return error.BadSetting;
        };
        prompt2d.attn_twin_text = if (std.c.getenv("TF_DS_ATTN_TWIN")) |v| std.mem.span(v) else null;
        prompt2d.rope_w_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ROPE_W")) |v| std.mem.span(v) else null);
        prompt2d.idx_tpp_text = if (std.c.getenv("TF_DS_IDX_TPP")) |v| std.mem.span(v) else null;
        if (idxTppWanted(prompt2d.idx_tpp_text) == null) {
            std.log.err("TF_DS_IDX_TPP={s}: not 0 or 4; not started", .{prompt2d.idx_tpp_text.?});
            return error.BadSetting;
        }
        prompt2d.assemble_quant_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ASSEMBLE_QUANT")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.attn_epilogue_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_ATTN_EPILOGUE")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.hc_stride_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_HC_STRIDE")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.head_cand_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_HEAD_CAND")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.head_key_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_HEAD_KEY")) |v| std.mem.span(v) else null); // every rank alike (Settings)
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
        prompt2d.gemm_halves_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_GEMM_HALVES")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        prompt2d.qgate_canary_asked = readQgate(); // every rank alike (Settings: any canary; the canary itself, agreeQgateCanary)
        prompt2d.draft_vocab_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_DRAFT_VOCAB")) |v| std.mem.span(v) else null); // every rank alike (Settings)
        if (std.c.getenv("TF_DS_L2_SKIP")) |v| round.l2_skip = std.fmt.parseInt(u8, std.mem.span(v), 10) catch 0;
        for ([_][:0]const u8{ "TF_DS_L2_ATTN_MB", "TF_DS_L2_MOE_MB", "TF_DS_L2_NEXT_MB", "TF_DS_L2_STRETCH_END_MB" }, 0..) |name, i| {
            if (std.c.getenv(name)) |v| round.l2_mb[i] = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_mb[i];
        }
        if (std.c.getenv("TF_DS_ROCI_XLOG")) |v| rdma.xlog = std.mem.eql(u8, std.mem.span(v), "1");
        if (std.c.getenv("TF_DS_L2_NEXT_EARLY")) |v| round.l2_next_early = !std.mem.eql(u8, std.mem.span(v), "0");
        if (std.c.getenv("TF_DS_L2_DN_MB")) |v| round.l2_dn_mb = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_dn_mb;
        if (std.c.getenv("TF_DS_L2_DEC_MB")) |v| round.l2_dec_mb = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_dec_mb;
        ops_mod.Ops.add2_vec_on = prompt2d.switchDefaultOn(if (std.c.getenv("TF_DS_ADD2_VEC")) |v| std.mem.span(v) else null); // the bytes are the scalar kernel's: a rank may differ
        if (std.c.getenv("TF_EXL3_L2_DELAY_NS")) |v| ops_mod.l2_delay_ns = std.fmt.parseInt(c_longlong, std.mem.span(v), 10) catch ops_mod.l2_delay_ns;
        if (std.c.getenv("TF_EXL3_L2_RATE_GBPS")) |v| ops_mod.l2_rate_gbps = std.fmt.parseFloat(f64, std.mem.span(v)) catch ops_mod.l2_rate_gbps;
        const chunk_text: ?[]const u8 = if (std.c.getenv("TF_DS_L2_CHUNK_KIB")) |v| std.mem.span(v) else null;
        ops_mod.l2_chunk_bytes = ops_mod.l2ChunkWanted(chunk_text) orelse {
            std.log.err("TF_DS_L2_CHUNK_KIB={s}: not one of 1, 2, 4, 8, 16 (KiB); not started", .{chunk_text.?});
            return error.BadSetting;
        };
        const aq_text: ?[]const u8 = if (std.c.getenv("TF_DS_AQ_VEC")) |v| std.mem.span(v) else null;
        mx.aq_vec_on = mx.aqVecWanted(aq_text) orelse {
            std.log.err("TF_DS_AQ_VEC={s}: not 0 or 1; not started", .{aq_text.?});
            return error.BadSetting;
        };
        const woam_text: ?[]const u8 = if (std.c.getenv("TF_DS_WOA_MERGE")) |v| std.mem.span(v) else null;
        mx.woa_merge = mx.woaMergeWanted(woam_text) orelse {
            std.log.err("TF_DS_WOA_MERGE={s}: not 0 or 1; not started", .{woam_text.?});
            return error.BadSetting;
        };
        if (mx.woa_merge) std.log.info("TF_DS_WOA_MERGE=1: wo_a's FP8 groups of a prompt chunk run as one launch (grid.z = groups) instead of one a group: the same bytes (kernels replay woa_merge)", .{}) else std.log.info("TF_DS_WOA_MERGE=0: wo_a's FP8 groups of a prompt chunk run one launch a group (the operator's choice)", .{});
        if (ops_mod.l2_chunk_bytes != 8192) std.log.info("TF_DS_L2_CHUNK_KIB={d}: the L2 prefetch's waves are {d} KiB chunks x 128 every {d} ns (the served: 8 KiB, same rate, same bytes)", .{ ops_mod.l2_chunk_bytes / 1024, ops_mod.l2_chunk_bytes / 1024, ops_mod.waveNs(ops_mod.l2_rate_gbps, ops_mod.l2_chunk_bytes) });
        m.round_ms = o.round_ms orelse &draft.round_ms;
        if (m.round_ms.len == 0) return error.BadRoundMs;
        if (poolBelowContext(o.pool, o.context)) |why| {
            std.log.err("{s}", .{why});
            return error.PoolBelowContext;
        }
        m.pool_cap = o.pool;
        m.ledger = .{};
        m.ctx_cap = if (o.context != 0) @min(o.context, o.pool) else o.pool;
        m.pass_rows = prompt2d.pass_rows_asked; // (every rank alike: the start-up exchange refuses a follower that differs)
        if (prompt2d.pass_tail_asked) std.log.info("TF_DS_PASS_TAIL=1: a prompt's final pass runs up to {d} rows, the remainder after whole {d}-row units merged into it; every other pass stays whole units on the grid", .{ m.pass_rows, chunk_rows });
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
        m.erows = null;
        m.rdma_mod = null;
        m.ring = null;
        m.graphs = null;
        m.pgraphs = null;
        m.graph_fallbacks = .init(0);
        m.vocab_stats = .{};
        m.two = null;
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
        m.keep_ahead = false;
        m.pair_gap = roci_pair.gap;
        m.pair_prio = .equal;
        m.pair_major = false;
        m.partner_rc = null;
        m.partner_link = null;
        m.partner_seen = .{};
        m.fill_t0 = 0;
        m.pair_in_fill = 0;
        m.dup_snap = .{ 0, 0 };
        m.dup_fill = .{ 0, 0 };
        m.steps_seen = 0;
        m.other_seen = 0;
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
        {
            // the context against the decode indexer's grid (before the weights load: a refused context costs seconds)
            const config_mod = @import("config.zig");
            var grid_why: config_mod.Why = .{};
            config_mod.checkIndexWindow(c, m.ctx_cap, tri_index.tile, &grid_why) catch |e| {
                std.log.err("{s}; not started", .{grid_why.text()});
                return e;
            };
        }
        // TF_DS_ROCI_STAGGER_PRIO: pass A's stream (this one, which every other piece of work also runs on) at the priority the mode gives it
        m.pair_prio = roci_pair.prioOf(if (std.c.getenv("TF_DS_ROCI_STAGGER_PRIO")) |v| std.mem.span(v) else null);
        m.stream = if (prompt2d.roci_stagger_asked and m.pair_prio != .equal) blk: {
            const range = try cuda.Stream.priorityRange(d);
            break :blk try cuda.Stream.initPriority(d, roci_pair.prioritiesOf(m.pair_prio, range.least, range.greatest)[0]);
        } else try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();
        // (Model.open starts from undefined, so the field's default never applies: unset, null here)
        m.head_ready = null;
        m.cand_first = prompt2d.switchAsked(if (std.c.getenv("TF_DS_CAND_FIRST")) |v| std.mem.span(v) else null);
        if (m.cand_first) {
            m.head_ready = try cuda.Event.init(d, false);
            std.log.info("{s}", .{candFirstLine()});
        }
        errdefer if (m.head_ready) |*x| x.deinit();
        // TF_DS_WINDOW_ONCE (on unless 0): a chunk uploads its window indices and low position once, not in every layer. Bytes unchanged.
        prompt.window_once = prompt.windowOnceWanted(if (std.c.getenv("TF_DS_WINDOW_ONCE")) |v| std.mem.span(v) else null);
        std.log.info("TF_DS_WINDOW_ONCE={s}: rank {d}: a prompt chunk's window gather indices, window low position and ring-write indices are uploaded {s}", .{ if (prompt.window_once) "1" else "0", o.rank, if (prompt.window_once) "once per chunk, into buffers of their own" else "in every layer, as before" });
                    std.log.warn("TF_DS_ISSUE_LEAD_WAIT={s}: not a mode (poll, block or backoff): poll", .{wait_text.?});
                else if (m.ilead_wait == .backoff)
                    std.fmt.bufPrint(&how_buf, "naps that start at {d} us and double each empty wake up to {d} us (TF_DS_ISSUE_LEAD_WAIT=backoff), the transport checked and the Engram worker's slots returned every wake, a wait over {d} ms logged every 5 s (TF_DS_ISSUE_LEAD_WARN_MS, 0: never)", .{ m.ilead_slice_us, issue_lead.backoff_cap_us, m.ilead_warn_ms })
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
        // TF_DS_WOA_W8: the main layers' wo_a from its FP8 rows alone where the prompt path's kernel takes them; the
        // bits are the bf16 rows' either way, so each rank decides on its own
        weights.woa_w8 = false;
        if (prompt2d.woa_w8_asked) {
            if (mx.woaW8Off(c.format == .native, mx.pfOffEnv(), mx.decOnEnv())) |why| std.log.info("TF_DS_WOA_W8=1: off: {s}", .{why}) else weights.woa_w8 = true;
        }
        // TF_DS_DRAFT_4WAY: decided here, before the weights load, by each rank (asked and its conditions); the ranks'
        // drafters compared after the load (agreeDraftFourWay)
        weights.draft_4way = false;
        if (prompt2d.draft_4way_asked) {
            if (try draftFourWayOff(a, io, c, sp, o.kit_dir, o.drafts, mx.decOnEnv())) |why| std.log.info("TF_DS_DRAFT_4WAY=1: off on rank {d}: {s}", .{ o.rank, why }) else weights.draft_4way = true;
        }
        // TF_DS_DRAFT_MAIN_4WAY: likewise decided before the load by each rank; the ranks compared after it
        weights.draft_main_4way = false;
        if (prompt2d.draft_main_4way_asked) {
            if (draftMainFourWayOff(c, sp, o.drafts, mx.decOnEnv())) |why| std.log.info("TF_DS_DRAFT_MAIN_4WAY=1: off on rank {d}: {s}", .{ o.rank, why }) else weights.draft_main_4way = true;
        }
        const free_before_weights = freeNow(ctx);
        m.ledger.record(@src(), .weights, free_before_weights -| freeNow(ctx), 0);
        // the ranks' links (rank 0 takes its followers in order; a follower retries its connect for TF_DS_CONNECT_S)

        try m.fan.join(io, o.master, o.port, o.rank, o.world, prompt2d.connectSeconds());
        try m.vfan.join(io, o.master, o.port + prompt2d.host_links_port_offset, o.rank, o.world, prompt2d.connectSeconds());
        // every rank's streams, pool, round rows, rails, drafter, graphs and math switches checked against rank 0's first;
        // each follower's answer carries its weight digest (TF_DS_RANK_CACHE_SHA) to rank 0
        const mine = startSettings(o);
        var heard: prompt2d.Heard = .{};
        m.comm = try m.fan.settingsChecked(o.rank, o.world, mine, rankDigest(), &heard);
        errdefer m.comm.deinit();
        m.comm.host = .{ .star = .{ .links = &m.vfan.links, .rank = o.rank, .world = o.world, .busy = &m.star_busy } };
        m.lead = heard.lead orelse mine;
        m.weights = heard.weights;
        errdefer if (m.ring) |r| r.destroy(gpa);
        errdefer if (m.rdma_mod) |*x| x.unload();

        // kernels: the recorded Triton set, the served extension cubins, the torch-op images
        m.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }));
        errdefer m.set.deinit();
        // the kit's extra set (aot-extra/), which only the call sites of TF_DS_DRAFT_4WAY's stages, TF_DS_ATTN_U's prompt
        // attention, TF_DS_IDX_U's and TF_DS_IDX_V's prompt index scores and TF_DS_ATTN_U_DECODE's round attention search;
        // then every rank's drafter compared
        m.extra = null;
        var extra_why: ?[]const u8 = null;
        const qkv_plain_asked = prompt2d.switchAsked(if (std.c.getenv("TF_DS_QKV_PLAIN")) |v| std.mem.span(v) else null);
        if (weights.draft_4way or prompt2d.attn_u_asked or prompt2d.idx_u_asked or prompt2d.idx_v_asked or prompt2d.attn_u_decode_asked or prompt2d.hc_fused_asked or merge_d_asked != 0 or prompt2d.rope_w_asked or qkv_plain_asked or prompt2d.attn_epi_fused_asked or prompt2d.hc_2s_asked or idxTpp(prompt2d.idx_tpp_text) != 0) {
            if (cuda.aot.Set.load(gpa, io, d, ctx.device, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }))) |x| {
                m.extra = x;
            } else |e| extra_why = @errorName(e);
        }
        errdefer if (m.extra) |*x| x.deinit();
        const load_why = extra_why;
        if (weights.draft_4way) if (m.extra) |*x| {
            extra_why = try draft.fourWayKitOff(gpa, SetLookups{ .set = x }, c, sp.headSpan(c.*)[1]);
        };
        try agreeDraftFourWay(d, &m.comm, m.stream, prompt2d.draft_4way_asked, weights.draft_4way, extra_why);
        try agreeDraftMainFourWay(d, &m.comm, m.stream, prompt2d.draft_main_4way_asked, weights.draft_main_4way);
        try useAttnTwin(a, io, prompt2d.attn_twin_text, prompt2d.attn_u_asked or prompt2d.attn_u_decode_asked, m.extra != null, load_why, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }));
        useQkvPlain(qkv_plain_asked, m.extra != null, load_why, if (qkv_plain_asked and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_q_kv_norm") else 0);
        try useRopeW(prompt2d.rope_w_asked, m.extra != null, load_why, if (prompt2d.rope_w_asked and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_rope_heads_w") else 0);
        useIdxTpp(prompt2d.idx_tpp_text, m.extra != null, load_why, if (idxTpp(prompt2d.idx_tpp_text) != 0 and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_index_score_rows_t") else 0);
        useAttnU(a, io, prompt2d.attn_u_asked, m.extra != null, load_why, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }), try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }));
        try agreeTwin(d, &m.comm, m.stream, "TF_DS_ATTN_U", prompt2d.attn_u_asked, &tri_attn.attn_u);
        useIdxU(a, io, prompt2d.idx_u_asked, m.extra != null, load_why, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }), try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }));
        try agreeTwin(d, &m.comm, m.stream, "TF_DS_IDX_U", prompt2d.idx_u_asked, &tri_index.idx_u);
        try useIdxV(prompt2d.idx_v_asked, m.extra != null, load_why, if (prompt2d.idx_v_asked and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_index_score_rows_v") else 0);
        try useHcFused(prompt2d.hc_fused_asked, m.extra != null, load_why, if (prompt2d.hc_fused_asked and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_hc_pre_pf_fused") else 0);
        useMergeD(merge_d_asked, m.extra != null, load_why, if (merge_d_asked != 0 and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_sparse_attn_merge_d") else 0, if (merge_d_asked != 0 and m.extra != null) mergeDHave(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" })) else .{});
        try useAttnEpiFused(prompt2d.attn_epi_fused_asked, m.extra != null, load_why, if (prompt2d.attn_epi_fused_asked and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_sparse_attn_part_ue") else 0);
        const epi_text: ?[]const u8 = if (std.c.getenv("TF_DS_ATTN_EPI_TWIN")) |v| std.mem.span(v) else null;
        const epi_twin = tri_attn.epiTwinWanted(epi_text) orelse {
            std.log.err("TF_DS_ATTN_EPI_TWIN={s}: not one of ue, we, xe; not started", .{epi_text.?});
            return error.BadSetting;
        };
        useAttnEpiWe(epi_twin, tri_attn.attn_epi_fused, if (tri_attn.attn_epi_fused and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_sparse_attn_part_we") else 0, if (tri_attn.attn_epi_fused and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_sparse_attn_part_xe") else 0);
        useHc2s(prompt2d.hc_2s_asked, m.extra != null, load_why, if (prompt2d.hc_2s_asked and m.extra != null) extraVariants(a, io, try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }), "_hc_mix_pf_2s") else 0);
        useAttnUDecode(a, io, prompt2d.attn_u_decode_asked, m.extra != null, load_why, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }), try std.fs.path.join(a, &.{ o.kit_dir, "aot-extra" }));
        try agreeTwin(d, &m.comm, m.stream, "TF_DS_ATTN_U_DECODE", prompt2d.attn_u_decode_asked, &tri_attn.attn_u_decode);
        m.pf = try exl3_prefill.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear.cubin"));
        errdefer m.pf.unload();
        m.lg = try exl3_linear.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear_grouped.cubin"));
        errdefer m.lg.unload();
        m.exk = try exl3_experts.Kernels.load(d, try readKit(a, io, o.kit_dir, "experts.cubin"), try readKit(a, io, o.kit_dir, "experts_cb.cubin"), exl3_linear.codebook_mul1);
        errdefer m.exk.unload();
        if (!cuda.kernels.available) return error.NoKernelImages;
        m.ops = try ops_mod.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
        errdefer m.ops.unload();
        m.ex = try exact.Exact.load(d, cuda.kernels.dsv41_torch);
        errdefer m.ex.unload();
        m.par = if (sp.parity and parOne() and c.format == .exl3) try exl3_experts2d.ParKernels.load(d, cuda.kernels.dsv41_experts_par) else null;
        errdefer if (m.par) |*x| x.unload();
        m.mxk = if (c.format == .native) try mx.Kernels.load(d, cuda.kernels.dsv41_native) else null;
        if (std.c.getenv("TF_DS_NACT_LOG")) |v| if (std.mem.eql(u8, std.mem.span(v), "1")) nactOpen(d);
        errdefer if (m.mxk) |*x| x.unload();
        usePfSplit(if (m.mxk) |*x| x else null, prompt2d.pf_split_asked);
        usePfGemm5(if (m.mxk) |*x| x else null, prompt2d.pf_gemm5_asked);
        useGroupTiles(if (m.mxk) |*x| x else null, prompt2d.group_tiles_asked);
        useTopkTiles(if (m.mxk) |*x| x else null, prompt2d.topk_tiles_asked, c.index_topk);
        useRun128(if (m.mxk) |*x| x else null, prompt2d.pf_run128_asked);
        useAqVec(if (m.mxk) |*x| x else null);
        mx.pf_raster = if (m.mxk != null) prompt2d.pf_raster_asked else 0;
        if (prompt2d.pf_raster_asked != 0) {
            if (mx.pf_raster != 0) std.log.info("TF_DS_PF_RASTER={d}: a prompt GEMM whose activation slab is {d} MiB or more (the tile FP8 GEMMs, the block-scaled MMA's and the bf16 tile GEMM) walks its tiles {d} row tiles at a time across all the column tiles; the output bytes are the default order's", .{ mx.pf_raster, mx.raster_slab_bytes >> 20, mx.pf_raster }) else std.log.info("TF_DS_PF_RASTER={d}: off: needs the checkpoint's own FP8 / MXFP4 weights (the EXL3 GEMMs are cuBLAS's)", .{prompt2d.pf_raster_asked});
        }
        mx.gemm_halves = prompt2d.gemm_halves_asked and m.mxk != null;
        if (prompt2d.gemm_halves_asked) {
            if (mx.gemm_halves) std.log.info("TF_DS_GEMM_HALVES=1: a prompt pass of more than {d} rows runs its dense FP8, bf16 and pf GEMMs as row blocks of at most {d} rows, each block the launch a {d}-row pass makes for its rows; the experts keep the whole pass", .{ mx.gemm_halves_rows, mx.gemm_halves_rows, mx.gemm_halves_rows }) else std.log.info("TF_DS_GEMM_HALVES=1: off: needs the checkpoint's own FP8 / MXFP4 weights (the EXL3 GEMMs are cuBLAS's)", .{});
        }
        useDecodeFuse(if (m.mxk) |*x| x else null, d, prompt2d.decode_fuse_asked);
        var sf_buf: [256]u8 = undefined;
        try useMmaSf(d, &m.comm, m.stream, if (m.mxk) |*x| x else null, prompt2d.mma_sf_asked, sfCovers(m.w.layers, false, &sf_buf));
        try useMmaSfExact(gpa, d, &m.comm, m.stream, if (m.mxk) |*x| x else null, prompt2d.mma_sf_exact_asked, m.w.layers, sfCovers(m.w.layers, true, &sf_buf));
        try useGemmW16(d, &m.comm, m.stream, if (m.mxk) |*x| x else null, prompt2d.gemm_w16_asked);
        useSplit64n(if (m.mxk) |*x| x else null, prompt2d.split64n_asked);
        usePfT64(if (m.mxk) |*x| x else null, prompt2d.pf_t64_asked);
        useGemmR2(if (m.mxk) |*x| x else null, prompt2d.gemm_r2_asked);
        if (prompt2d.bf16_rows_asked) {
            if (m.mxk) |*x| {
                x.bf16_rows = true;
                std.log.info("TF_DS_BF16_ROWS=1: a prompt chunk's plain bf16 GEMMs (the indexer's weights_proj) take a CTA a 16-row tile", .{});
            } else std.log.info("TF_DS_BF16_ROWS=1: off: needs the checkpoint's own FP8 / MXFP4 weights", .{});
        }
        prompt.quant_once = prompt2d.quant_once_asked and m.mxk != null;
        if (prompt2d.quant_once_asked) {
            if (prompt.quant_once) std.log.info("TF_DS_QUANT_ONCE=1: attention's front quantizes x once for wq_a and wkv (where TF_DS_FRONT_OWNED does not run it on owned rows)", .{}) else std.log.info("TF_DS_QUANT_ONCE=1: off: needs the checkpoint's own FP8 / MXFP4 weights", .{});
        }
        // a layer loaded without wo_a's bf16 rows runs wo_a on its FP8 rows: that kernel, or no start
        for (m.w.layers) |lay| if (!lay.wo_a_rows) {
            (m.mxk orelse return error.NoW8Kernel).useW8() catch |e| {
                std.log.err("TF_DS_WOA_W8=1: the layers loaded without wo_a's bf16 rows have no kernel ({s}): start without it", .{@errorName(e)});
                return e;
            };
            break;
        };

        // the RoPE tables (the served build's bits: torch CPU pow and polar), every position a stream reaches
        const half = c.rope_dim / 2;
        // the factor the kit's tables were cut for against the factor the engine runs, and the context against the positions it
        // reaches: unset TF_DS_YARN_FACTOR and a kit with no "factor" in rope.json are the checkpoint's own (16, 1,048,576)
        {
            const config_mod = @import("config.zig");
            const yarn_text: ?[]const u8 = if (std.c.getenv("TF_DS_YARN_FACTOR")) |v| std.mem.span(v) else null;
            const asked = config_mod.yarnFactorAsked(yarn_text) catch {
                std.log.err("TF_DS_YARN_FACTOR={s}: not a number from 1 to 1024; not started", .{yarn_text.?});
                return error.BadSetting;
            };
            const rope_json: ?[]const u8 = std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit_dir, "rope.json" }), a, .limited(1 << 16)) catch null;
            const kit_factor = if (rope_json) |t| try config_mod.kitRopeFactor(a, t) else null;
            var rope_why: config_mod.Why = .{};
            const rr = config_mod.checkRope(c, asked, kit_factor, m.ctx_cap, &rope_why) catch |e| {
                std.log.err("{s}; not started", .{rope_why.text()});
                return e;
            };
            if (asked != null) std.log.info("TF_DS_YARN_FACTOR={d}: the RoPE tables are YaRN factor {d} over {d} positions (the checkpoint's own: {d} over {d})", .{ rr.factor, rr.factor, rr.max_position, c.rope_factor, c.max_position });
        }
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

        const free_before_transport = freeNow(ctx);
        m.kv_vmm = null; // (TF_DS_KV_VMM=1 opens it with the caches)
        m.vmm_most = 0;
        m.pool_start = 0;
        m.pool_ceiling = 0;
        m.kv_start_target_mib = @intCast(@min(o.kv_target_bytes >> 20, std.math.maxInt(u32) - 1));
        m.kv_live_floor = 0;
        m.kv_live_loading = 0;
        m.arena = try prompt.Arena.init(d, o.arena_bytes);
        errdefer m.arena.deinit();
        // world 4 (layer_main's setup): the 2D exchanges' buffers, and with rdma_devices their rings (whose infos go
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
            if (prompt2d.pack_epilogue_asked) {
                // TF_DS_PACK_EPILOGUE: every rank's own check, then on every rank or the start ends naming the rank and its reason (never a silent fallback)
                const nx = m.w.layers[c.layers / 2].experts.native;
                const why: ?[]const u8 = if (m.mxk == null) "needs the checkpoint's own FP8 / MXFP4 weights" else if (m.mxk.?.sf != .exact) "needs TF_DS_MMA_SF_EXACT=1 (the twin is the exact block-scaled gate/up's)" else if (m.mxk.?.gemm_r2 or m.mxk.?.gemm_w16) "gate/up runs the R2 / 16-warp kernels (TF_DS_GEMM_R2 / TF_DS_GEMM_W16), which have no quantising epilogue" else if (nx == null) "needs native experts" else blk: {
                    mx.usePackEpilogue(&m.mxk.?) catch break :blk "TF_DS_PACK_EPILOGUE=1 needs tf_dsn_sf_gu_exact_q_kernel, which this build lacks";
                    break :blk pack_epilogue.packEpilogueOff(true, m.two.?.gu[m.two.?.p], m.two.?.ir, m.w.layers[c.layers / 2].experts.width, (nx.?.tiles - nx.?.rest_tiles) * 32, nx.?.rest_tiles * 32);
                };
                // on every rank or off on every rank, said: the epilogue is a speed change with the same bytes, so a build, kit or setting that cannot
                // run it leaves the baseline two launches (gate/up, then packQuant) and the start line says why
                const cannot = try prompt2d.ranksThatCannot(d, &m.comm, m.stream, why == null);
                if (cannot != 0) {
                    var rb: [64]u8 = undefined;
                    prompt2d.pack_epilogue_asked = false;
                    std.log.info("TF_DS_PACK_EPILOGUE=1: off: {s} cannot run it (this rank: {s}); gate/up and packQuant run as two launches", .{ prompt2d.rankList(cannot, &rb), why orelse "can" });
                } else {
                    std.log.info("TF_DS_PACK_EPILOGUE=1: gate/up's twin kernel writes the expert pack's codes and scales into the send region and packQuant does not run, on every rank", .{});
                }
            }
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
            // TF_DS_IDX_PROJ_OWNED: each rank's start check (its kernel set included), then on every rank or none
            const proj_why: ?[]const u8 = if (prompt2d.idx_proj_owned_asked) try prompt.idxProjOwnedOff(gpa, SetFinds{ .set = &m.set }, m.mxk != null, m.two.?.idx_shard, m.w.layers, c.candidate_source, c.rope_dim, c.index_heads, c.index_head_dim, &rows4, m.comm.world) else null;
            try m.two.?.agreeIdxProjOwned(d, &m.comm, m.stream, prompt2d.idx_proj_owned_asked, proj_why);
            if (o.rdma_devices) |devs| {
                const image: []const u8 = if (o.rdma_kernels) |path|
                    try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 26), .@"16", null)
                else if (cuda.kernels.available) cuda.kernels.dsv41_rdma else return error.NoRdmaKernels;
                m.rdma_mod = try cuda.Module.load(d, image);
                var devices: std.ArrayList([]const u8) = .empty;
                var dit = std.mem.splitScalar(u8, devs, ',');
                while (dit.next()) |x| try devices.append(a, x);
                const rk = try rdma.Kernels.load(m.rdma_mod.?);
                const peers = rdmaCollectPeers();
                if (peers) std.log.info("{s}", .{collectPeersLine(rk.collect_peers != null)});
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
        if (sp.pair == null and prompt2d.idx_proj_owned_asked) std.log.info("TF_DS_IDX_PROJ_OWNED=1: off: the indexer's owned rows are the four-node 2D split's (TF_DS_IDX_SHARD)", .{});
        // TF_DS_ASSEMBLE_QUANT: a parity chunk's intermediate assembly writes down's codes in the same launch
        if (prompt2d.assemble_quant_asked) {
            if (m.mxk) |*x| {
                if (m.two != null and m.two.?.ir != 0) {
                    x.useAssembleQuant() catch |e| {
                        std.log.err("TF_DS_ASSEMBLE_QUANT=1: the assembly-and-quant kernel is missing ({s})", .{@errorName(e)});
                        return e;
                    };
                    std.log.info("TF_DS_ASSEMBLE_QUANT=1: a 2D parity chunk's expert assembly writes down's codes in the same launch (the quant's read goes)", .{});
                } else std.log.info("TF_DS_ASSEMBLE_QUANT=1: off: needs the four-node 2D split's parity experts (the rest block)", .{});
            } else std.log.info("TF_DS_ASSEMBLE_QUANT=1: off: needs the checkpoint's own FP8 / MXFP4 weights", .{});
        }
        // TF_DS_ATTN_EPILOGUE: the attention output's inverse RoPE and wo_a's quant in one launch
        prompt.attn_epilogue = prompt2d.attn_epilogue_asked and m.mxk != null;
        if (tri_attn.attn_epi_fused and !prompt.attn_epilogue) {
            tri_attn.attn_epi_fused = false;
            std.log.info("TF_DS_ATTN_EPI_FUSED=1: off: needs TF_DS_ATTN_EPILOGUE=1 (in effect)", .{});
        }
        if (prompt2d.attn_epilogue_asked) {
            if (prompt.attn_epilogue) std.log.info("TF_DS_ATTN_EPILOGUE=1: a prompt chunk's attention output takes its inverse RoPE and wo_a's quant in one launch (the quant's read of o goes)", .{}) else std.log.info("TF_DS_ATTN_EPILOGUE=1: off: needs the checkpoint's own FP8 / MXFP4 weights", .{});
        }
        // TF_DS_HC_STRIDE: a quarter-HC layer's mixes read the chunk's own gathered planes; the tight repack goes
        prompt.hc_stride = prompt2d.hc_stride_asked and m.two != null;
        if (prompt2d.hc_stride_asked) {
            if (prompt.hc_stride) std.log.info("TF_DS_HC_STRIDE=1: a quarter-HC layer's mixes read the gathered planes at the chunk's stride (no repack copy)", .{}) else std.log.info("TF_DS_HC_STRIDE=1: off: needs TF_DS_QUARTER_HC=1 and the four-node 2D split", .{});
        }
        logPdlControls(m.comm.rank);
        std.log.info("transport: Rocinante carries every exchange between the Sparks; no NCCL is loaded or made (rank {d})", .{o.rank});
        m.eng = .{ .d = d, .s = m.stream, .t = .{ .set = &m.set, .extra = if (m.extra) |*x| x else null, .stream = m.stream }, .blas = undefined, .comm = &m.comm, .pf = &m.pf, .lin = &m.lg, .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = c, .w = &m.w, .world = sp.world, .plain = rope, .compressed = rope_c, .ring = m.ring, .two = if (m.two) |*t| t else null };
        m.eng.round_rows = o.round_rows;
        if (m.par) |*x| m.eng.par = x;
        if (m.mxk) |*x| m.eng.mx = x;
        m.ledger.record(@src(), .transport, free_before_transport -| freeNow(ctx), 0); // the arena's first block, the exchanges' buffers and rings
        const free_before_chunk = freeNow(ctx);
        try agreeQgateCanary(d, &m.comm, m.stream, prompt2d.qgate_canary_asked);
        if (prompt.qgate_body != .off) try m.mxk.?.useQgateBody();
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, m.pass_rows, m.ctx_cap); // a stream's keys at most
        m.ledger.record(@src(), .chunk, free_before_chunk -| freeNow(ctx), 0);
        // the pool: the ranks' vote on the free memory they have now (kv_pool.zig), before the caches are taken
        const pool = try votePool(m, o, c);
        m.pool_cap = pool;
        const free_before_pool = freeNow(ctx);
        m.caches = if (o.kv_vmm) try openVmmCaches(m, ctx, c, pool) else try prompt.Caches.init(&m.eng, &m.arena, pool, m.streams);
        errdefer if (m.kv_vmm) |*v| v.deinit();
        m.pool_start = pool;
        m.pool_ceiling = if (m.kv_vmm != null) @min(m.vmm_most, engine_api.kv_pool.max_pool_positions) else pool;
        const free_after_pool = freeNow(ctx);
        m.ledger.record(@src(), .kv_pool, free_before_pool -| free_after_pool, 0);
        logPoolTaken(m, c, free_before_pool -| free_after_pool);
        m.blas = try cublas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer m.blas.close();
        m.eng.blas = &m.blas;
        m.rings = try a.alloc(u64, c.layers);
        m.ring_view = try a.alloc(u64, c.layers);
        const free_before_rings = freeNow(ctx);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.streams * m.eng.ringBytes());
            try d.check(d.api.cuMemsetD8Async(r.*, 0, m.streams * m.eng.ringBytes(), m.stream.handle), "cuMemsetD8Async");
        }
        m.ledger.record(@src(), .window_rings, free_before_rings -| freeNow(ctx), 0);
        const free_before_round = freeNow(ctx);
        m.rd = try round.Round.init(&m.eng, &m.arena, a, m.ctx_cap); // a round's deepest bucket: a stream's context
        errdefer m.rd.dropPar();
        errdefer m.rd.dropPrefetch();
        errdefer m.rd.dropSide(&m.eng);
        m.amax = try m.arena.take(round.max_rows * 4);
        m.cand = try m.arena.take(round.max_rows * (sampling.max_candidates * 12 + 4));
        m.eng.agree_flag = try m.arena.take(64); // G0.3: the graph captures' agreement bytes (agreeAll, agreeCapture)
        m.ledger.record(@src(), .round_buffers, free_before_round -| freeNow(ctx), 0);
        errdefer if (m.dr) |*dr| dr.deinit();
        errdefer if (m.pgraphs) |*g| g.deinit();
        errdefer if (m.graphs) |*g| g.deinit();
        if (o.graphs) {
            m.graphs = round.Graphs.init(gpa, o.graph_cap);
            m.graphs.?.lazy.fallbacks = &m.graph_fallbacks;
        }
        if (o.side) try m.rd.useSide(&m.eng);
        useLateFork(&m.rd, prompt2d.hc_late_fork_asked, c.format == .native);
        if (o.prefetch) try m.rd.usePrefetch(&m.eng);
        if (m.two) |t| if (t.ir != 0 and o.side and m.par == null) try m.rd.usePar(&m.eng); // parity in two launches: the rest beside the main gate / up
        try useHeadCand(m, a, d, prompt2d.head_cand_asked);
        try useHeadKey(m, a, d, prompt2d.head_key_asked);
        if (o.drafts) {
            const free_before_drafter = freeNow(ctx);
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, m.streams);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);
            m.ledger.record(@src(), .drafter, free_before_drafter -| freeNow(ctx), 0);
            if (o.graphs) {
                m.pgraphs = draft.PassGraphs.init(gpa);
                m.pgraphs.?.lazy.fallbacks = &m.graph_fallbacks;
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        try useDraftVocab(m, prompt2d.draft_vocab_asked);
        try m.stream.synchronize();
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize"); // the setup's legacy-stream copies too

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
            // the drill names the engine's own rank among all its Sparks (0-3 on four), never the 2D split's position in a ring of two
            prompt.engram_fault = prompt.EngramFault.parse(if (std.c.getenv("TF_DS_ENGRAM_FAULT")) |v| std.mem.span(v) else null, @intCast(o.world)) catch {
                std.log.err("TF_DS_ENGRAM_FAULT must be R:C, a rank below {d} and the layer-0 step to fail at", .{o.world});
                return error.BadEngramFault;
            };
            prompt.engram_fault_rank = @intCast(o.rank);
            prompt.engram_fault_seen.store(0, .release);
            prompt.engram_fault_spent.store(false, .release);
            if (prompt.engram_fault.asked) {
                var fb: [96]u8 = undefined;
                std.log.info("TF_DS_ENGRAM_FAULT {s}", .{prompt.engram_fault.describe(&fb, @intCast(o.rank))});
            }
            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, m.pass_rows, chunk_rows, m.workerSlots());
            if (o.engram_aio) {
                // a round's reads, 2 x 12 x its rows at most: 1,024 slots for each 16 rows of the round-row setting
                m.aio = try engram_aio.Aio.init(gpa, io, 1024 * ((o.round_rows + 15) / 16));
                m.eh.?.aio = m.aio;
                // a prompt chunk's reads (2 x 12 x 2,048 at most): a ring of 4,096 slots (32 MiB), 1,024 rows a batch
                m.paio = try engram_aio.Aio.init(gpa, io, 4096);
                // TF_DS_ENGRAM_AIO_DEPTH: a cap on the prompt ring's reads in flight, 256 unless set (0: none), submitted a slice at a time
                // (TF_DS_ENGRAM_AIO_SLICE, default 64): the drive loses throughput when over-queued and io_submit then blocks for block-layer tags
                {
                    const cap: usize = if (std.c.getenv("TF_DS_ENGRAM_AIO_DEPTH")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 256 else 256;
                    const slice: usize = if (std.c.getenv("TF_DS_ENGRAM_AIO_SLICE")) |sv| std.fmt.parseInt(usize, std.mem.span(sv), 10) catch 64 else 64;
                    m.paio.?.configureDepth(cap, slice);
                    if (cap != 0) std.log.info("TF_DS_ENGRAM_AIO_DEPTH={d}: rank {d}: the prompt path's AIO ring keeps at most {d} reads in flight, submitted {d} at a time as completions make room (the drive loses throughput when over-queued)", .{ cap, m.comm.rank, cap, m.paio.?.depth_slice });
                }
                // TF_DS_ENGRAM_AIO_SLEEP: the prompt ring's waiters sleep in the kernel for a batch of completions instead of polling
                // (on unless "0": waiters sleep after two polls that find nothing; the decode rounds' ring keeps polling)
                if (if (std.c.getenv("TF_DS_ENGRAM_AIO_SLEEP")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true) {
                    const wake: usize = if (std.c.getenv("TF_DS_ENGRAM_AIO_WAKE")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 32 else 32;
                    const us: u32 = if (std.c.getenv("TF_DS_ENGRAM_AIO_WAKE_US")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1000 else 1000;
                    const spin: usize = if (std.c.getenv("TF_DS_ENGRAM_AIO_SPIN")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 2 else 2;
                    m.paio.?.configureSleep(true, wake, us, spin);
                    std.log.info("TF_DS_ENGRAM_AIO_SLEEP (on unless 0): rank {d}: the prompt path's AIO ring sleeps in io_getevents for {d} completions or {d} us instead of polling, after {d} empty polls (the decode rounds' ring keeps polling)", .{ m.comm.rank, m.paio.?.wake_min, us, spin });
                }
                m.eh.?.paio = m.paio;
            }
            // TF_DS_ENGRAM_PACKED (a folder of tf-dsv41-engram-pack's shards): a row is one read, kept in a row cache
            // (TF_DS_ENGRAM_CACHE_MIB, default 2048, 0 none) and decoded on the GPU; TF_DS_ENGRAM_OWNED=1: each node
            // reads only its range of the rows and a byte max over the nodes fills the rest; TF_DS_ENGRAM_INSTREAM=1:
            // a round's landing as a host step on the stream (with TF_DS_ENGRAM_AIO)
            if (std.c.getenv("TF_DS_ENGRAM_PACKED")) |pd| {
                const owned = false; // (TF_DS_ENGRAM_OWNED is a retired name: the whole tables are on every node)
                const eh = &m.eh.?;
                const cols = eh.hasher.cols();
                const k = eh.hi - eh.lo;
                const sl = if (owned) cols else k;
                const rec = c.engram_head_dim + c.engram_head_dim / 32;
                const mib: usize = if (std.c.getenv("TF_DS_ENGRAM_CACHE_MIB")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 2048;
                // TF_DS_ENGRAM_PREFAULT: the row cache faulted in here, not by its first puts while prompts run
                const prefault = engram_rows.envOn("TF_DS_ENGRAM_PREFAULT");
                const huge0 = if (prefault) hugeKiB(io) else null;
                m.erows = try engram_rows.Rows.open(gpa, d, &m.tables.?, m.epool.?, m.aio, m.paio, .{
                    .dir = std.mem.span(pd),
                    .owned = owned,
                    .node = role,
                    .nodes = o.world,
                    .lo = eh.lo,
                    .k = k,
                    .cols = cols,
                    .dim = c.engram_head_dim,
                    .layer_ids = c.engram_layers.slice(),
                    .round_rows = round.max_rows,
                    .chunk_rows = m.pass_rows,
                    .cache_bytes = mib << 20,
                    .prefault = prefault,
                    .instream = engram_rows.envOn("TF_DS_ENGRAM_INSTREAM"),
                    .pinned = pinned_on and wk_on,
                    .round_dev = try m.arena.take(round.max_rows * sl * rec),
                    .chunk_dev = try m.arena.take(m.pass_rows * sl * rec),
                    .chunk_dev_b = if (prompt2d.roci_stagger_asked) try m.arena.take(m.pass_rows * sl * rec) else 0,
                    .chunk_dev_c = if (prompt2d.roci_stagger_asked and prompt2d.roci_stagger_lanes >= 3) try m.arena.take(m.pass_rows * sl * rec) else 0,
                    .wk_slots = m.workerSlots(),
                    .unit_rows = chunk_rows,
                });
                eh.pk = m.erows;
                if (prefault) logPrefault(m.erows.?, mib, huge0, hugeKiB(io));
            }
            if (pinned_on) {
                if (!wk_on) {
                    std.log.info("TF_DS_ENGRAM_PINNED=1: no effect, the Engram worker is off (TF_DS_ENGRAM_WORKER=0)", .{});
                } else {
                    const bytes = try prompt.engramPin(&m.eh.?, d);
                    std.log.info("TF_DS_ENGRAM_PINNED=1: rank {d} stages the fill worker's Engram {s} in {d:.1} MiB of pinned host memory ({d} chunk slots of {d} rows, two Engram layers each); the upload waits for nothing and a slot goes back to the worker when its copy's event is done", .{ m.rank, if (m.erows != null) "records" else "rows", @as(f64, @floatFromInt(bytes)) / (1 << 20), m.eh.?.wk_slots, chunk_rows });
                }
            }
        } else if (c.engram_layers.slice().len > 0) return error.NoEngramTables;
        if (keep_ahead_asked) try m.useKeepAhead();
        m.keep_ahead_start = m.keep_ahead;
        m.keep_ahead_able = m.keep_ahead or m.keepAheadAble();
        live_config.setAble(.keep_ahead, m.keep_ahead_able);
        live_config.setAble(.kv_free_target, m.kv_vmm != null); // (a live free target resizes the pool only where it is in virtual memory)

        // a vision checkpoint's image routing bias (without it a prompt with images is refused; text runs as ever)
        if (c.vision and c.format == .exl3) { // (Format.native: the checkpoint's own gate.bias_vl, loaded with the weights)
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            m.loadVisionBias(path) catch |err| if (o.vision_bias != null or err != error.FileNotFound) return err;
        }

        m.host_pos = try a.alloc(i64, m.pass_rows);
        m.ids64 = try a.alloc(i64, m.pass_rows);
        m.vocab = sp.world * (if (m.two) |t| t.hw[0] + t.hw[1] else m.w.head.n); // 2D: both pairs' vocabulary parts
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        if (prompt2d.roci_stagger_asked) try openPair(m, gpa, d, o, sp, c) else try passFloor(m, c);
        if (prompt2d.engram_partner_asked) try openPartner(m, gpa, d, o, sp);
        try gatherLedger(m, free_after_pool);
        return m;
    }

    /// The worker's ring slots for the pass rows booted: a pass's resume units, twice them with two passes in flight.
    fn workerSlots(m: *const Model) usize {
        return engram_rows.workerSlotsN(m.pass_rows / chunk_rows, if (prompt2d.roci_stagger_asked) prompt2d.roci_stagger_lanes else 1);
    }

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
        if (m.partner_link) |l| {
            l.deinit();
            m.gpa.destroy(l);
        }
        if (m.partner_rc) |rc| rc.close(); // the host lane before the others: it is paired with the first lane
        m.closePair(); // the second lane before the first: it borrows the first's arena
        m.ilead_ev.free(LeadEvents{ .d = m.ctx.d });
        if (m.eh) |*eh| prompt.engramUnpin(eh);
        if (m.img_buf) |*b| b.free();
        m.spans.deinit(m.gpa);
        if (m.graphs) |*g| g.deinit();
        if (m.pgraphs) |*g| g.deinit();
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
        if (m.erows) |r| r.close();
        if (m.aio) |x| x.deinit();
        if (m.paio) |x| x.deinit();
        if (m.epool) |p| p.deinit(m.gpa);
        if (m.tables) |*t| t.close();
        if (m.rdma_rings) |*r| r.close(m.gpa);
        m.blas.close();
        if (m.kv_vmm) |*v| v.deinit();
        m.arena.deinit();
        m.rope_buf.free();
        if (m.par) |*x| x.unload();
        if (m.mxk) |*x| x.unload();
        m.ex.unload();
        m.ops.unload();
        m.exk.unload();
        m.lg.unload();
        m.pf.unload();
        m.set.deinit();
        if (m.extra) |*x| x.deinit();
        if (m.ring) |r| {
            r.stop();
            r.destroy(m.gpa);
        }
        if (m.rdma_mod) |*x| x.unload();
        if (m.two) |*t| if (t.rc) |rc| rc.close();
        m.comm.deinit();
        m.watch.stop(); // (still watching up to here: a wait above on a rank that went ends this process)
        m.vfan.close();
        m.fan.close();
        m.w.deinit();
        m.cache_file.close(m.io);
        m.ix.deinit();
        m.cache_dir.close(m.io);
        if (m.head_ready) |*x| x.deinit();
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

    /// TF_DS_RDMA_COLLECT_PEERS=1: the four-node rings' gather_into copy-out reads every peer's slot at once
    /// (rdma.Settings.collect_peers); unset or 0: the baseline kernel. The same bytes either way, so each rank's own choice.
    pub fn rdmaCollectPeers() bool {
        return prompt2d.switchAsked(if (std.c.getenv("TF_DS_RDMA_COLLECT_PEERS")) |v| std.mem.span(v) else null);
    }

    /// TF_DS_RDMA_COLLECT_PEERS' start line: on with the kernel in the rings' fatbin (`have`), else off and why.
    pub fn collectPeersLine(have: bool) []const u8 {
        return if (have) "TF_DS_RDMA_COLLECT_PEERS=1: the rings' gather_into copy-out reads every peer's slot at once (a row of blocks a peer, each waiting for its own peer's flag): the same bytes" else "TF_DS_RDMA_COLLECT_PEERS=1: off: the rings' kernels (TF_DS_RDMA_KERNELS, else the build's own) have no dsv41_rdma_collect_peers";
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

    // -- spill v2's checkpoint state (port of wojo's GLM 0110, PR #124: spill_state.zig) ------------------------------
    // A kept point's state on this rank as the spill tier stores it: the descriptor derived from the loaded weights and
    // the live allocation, and the planes' addresses and copies for spill.Packer. Nothing here runs unless the tier
    // calls it from the lanes, so with the tier off the served path never reaches it.

    pub const spill = @import("spill_state.zig");

    comptime {
        std.debug.assert(draft.max_streams <= spill.max_slots);
    }

    /// This rank's state planes as spill.Packer reads and writes them: their device addresses (the whole pool's
    /// compressed planes, a slot's rings and raw stores) and copies in the model's stream's order.
    pub const SpillDev = struct {
        m: *Model,

        pub const Event = cuda.Event;

        /// Row 0 of a block plane over the whole pool (the extent at base b starts b / ratio rows in, as view() puts it).
        pub fn blockPlane(dv: SpillDev, p: spill.Plane) u64 {
            const cs = &dv.m.caches;
            return switch (p.kind) {
                .comp_codes => cs.comp_codes[p.layer],
                .comp_scales => cs.comp_scales[p.layer],
                .idx_codes => cs.idx_codes[p.layer],
                .idx_scales => cs.idx_scales[p.layer],
                .ring, .raw_kv, .raw_score => 0,
            };
        }

        /// Slot `slot`'s rows of a small plane (snapCopy's addresses).
        pub fn slotPlane(dv: SpillDev, p: spill.Plane, slot: usize) u64 {
            const m = dv.m;
            const base: u64 = switch (p.kind) {
                .ring => m.rings[p.layer],
                .raw_kv => m.caches.raw_kv[p.layer],
                .raw_score => m.caches.raw_score[p.layer],
                .comp_codes, .comp_scales, .idx_codes, .idx_scales => 0,
            };
            return if (base == 0) 0 else base + slot * p.slotBytes();
        }

        pub fn capacity(dv: SpillDev) usize {
            return dv.m.caches.cap;
        }

        pub fn slots(dv: SpillDev) usize {
            return dv.m.streams;
        }

        // (every copy of the tier's reads or writes the caches on the model's stream: behind a pair's second pass first)
        pub fn d2h(dv: SpillDev, dst: []u8, src: u64) !void {
            try dv.m.pairJoin();
            const d = dv.m.ctx.d;
            try d.check(d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, dv.m.stream.handle), "cuMemcpyDtoHAsync");
        }

        pub fn h2d(dv: SpillDev, dst: u64, src: []const u8) !void {
            try dv.m.pairJoin();
            const d = dv.m.ctx.d;
        }

        pub fn fill(dv: SpillDev, dst: u64, value: u8, bytes: usize) !void {
            try dv.m.pairJoin();
            const d = dv.m.ctx.d;
            try d.check(d.api.cuMemsetD8Async(dst, value, bytes, dv.m.stream.handle), "cuMemsetD8Async");
        }

        pub fn record(dv: SpillDev, ev: cuda.Event) !void {
            try dv.m.pairJoin();
            try ev.record(dv.m.stream);
        }

        /// Whether the copies before `ev` completed (spill_tier polls it; a device error reads as done, and the next
        /// copy on the stream reports it).
        pub fn done(_: SpillDev, ev: cuda.Event) bool {
            return ev.done() catch true;
        }

        /// The copies before `ev` completed (a cancel, a drop, a shutdown: bounded by what was enqueued).
        pub fn wait(_: SpillDev, ev: cuda.Event) void {
            ev.synchronize() catch {};
        }
    };

    /// The spill state descriptor of this rank, derived from the loaded layers (Caches.init's reading of them) and
    /// checked against the live allocation: snapCopy's bytes, a ring per layer, every plane present. Replicated: every
    /// rank holds the whole pool (no KV split), so rank 0's store holds every rank's bytes.
    pub fn spillLayout(m: *const Model) !spill.StateLayout {
        const layers = m.eng.w.layers;
        if (layers.len > spill.max_layers) return error.SpillLayoutMismatch;
        var shapes: [spill.max_layers]spill.LayerShape = undefined;
        for (layers, shapes[0..layers.len]) |lay, *s| {
            const kv = lay.comp_wkv != null;
            s.* = .{ .ratio = if (kv) lay.ratio else 0, .kv = kv, .idx = kv and lay.idx_wk != null, .raw = kv and lay.ratio > 1 };
        }
        const l = try spill.StateLayout.init(shapes[0..layers.len], spill.Geometry.of(m.eng.c, chunk_rows, prompt.ring_extra, prompt.raw_rows), .replicated);
        if (l.small_bytes != m.snapBytes() or m.rings.len != layers.len) return error.SpillLayoutMismatch;
        const dv: SpillDev = .{ .m = @constCast(m) };
        for (l.blockPlanes()) |p| if (dv.blockPlane(p) == 0) return error.SpillLayoutMismatch;
        for (l.smallPlanes()) |p| {
            if (dv.slotPlane(p, 0) == 0) return error.SpillLayoutMismatch;
            if (p.kind == .ring and p.slotBytes() != m.eng.ringBytes()) return error.SpillLayoutMismatch;
        }
        return l;
    }

    /// The pool's extent and slot generations (one counter a chunk of positions), sized once at setup.
    pub fn spillGenerations(m: *const Model, gpa: std.mem.Allocator) !spill.Generations {
        return spill.Generations.init(gpa, m.caches.cap, chunk_rows);
    }

    /// A packer over this rank's planes, `l` from spillLayout and `g` from spillGenerations (both the caller's, alive
    /// while it is used). Its calls enqueue on the model's stream and record the caller's event: no allocation, no wait.
    pub fn spillPacker(m: *Model, l: *const spill.StateLayout, g: *spill.Generations) spill.Packer(SpillDev) {
        return .{ .dev = .{ .m = m }, .layout = l, .gens = g, .rank = m.rank };
    }

    /// The Engram read-ahead joined and its rows dropped, so the next chunk reads its own rows from its prompt's ids.
    /// Baseline hazard (also without spill): fillChunk takes a joined read-ahead whose (ahead_start, ahead_n) match its
    /// chunk, and a pass that ends between chunks (cancel, error) leaves one standing; a later prompt resuming a kept
    /// point at that position (in memory or restored) would consume rows hashed from the other prompt's ids. The fix
    /// is one call at the start of every pass, in fillBegin after its joinAhead (outside this block: routed to the
    /// coordinator); a restore calls this right before the resumed pass.
    pub fn resetEngramAhead(m: *Model) void {
        m.joinAhead();
        if (m.eh) |*eh| {
            eh.ahead_ready = .{ false, false };
            eh.ahead_n = 0;
            eh.ahead_err = null;
        }
    }

    /// After a restored prompt's fill: the drafter's rings were rebuilt by its replay through the prompt's end
    /// (spill.checkReplayRebuilds' rule, here as the pool records it), so the first draft reads no unwritten row.
    pub fn spillCheckResumed(m: *const Model, slot: usize, len: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        if (m.dpool) |*p| if (p.absorbed[slot] != len) return error.DraftNotRebuilt;
    }

    /// A compressed plane as copyExtent copies it: its address, row bytes (copyExtent's own widths from the config)
    /// and ratio.
    pub const SpillRefPlane = struct { plane: u64, row: usize, ratio: usize };

    /// copyExtent's planes as the landing and the probe walk them (land.Plane: the same fields).
    pub fn landPlanes(m: *const Model, out: *[4 * spill.max_layers]land.Plane) ![]land.Plane {
        var refs: [4 * spill.max_layers]SpillRefPlane = undefined;
        const r = try m.spillRefPlanes(&refs);
        for (r, out[0..r.len]) |x, *o| o.* = .{ .plane = x.plane, .row = x.row, .ratio = x.ratio };
        return out[0..r.len];
    }

    /// copyExtent's planes in its order (every kv layer: codes, scales, then the index keys' codes and scales), read
    /// from the loaded layers and the config, not from the spill layout. The landing and the probe walk them.
    pub fn spillRefPlanes(m: *const Model, out: *[4 * spill.max_layers]SpillRefPlane) ![]SpillRefPlane {
        const c = m.eng.c;
        const cs = &m.caches;
        var k: usize = 0;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            if (k + 4 > out.len) return error.SpillLayoutMismatch;
            const r: usize = lay.ratio;
            out[k] = .{ .plane = cs.comp_codes[i], .row = c.head_dim / 2, .ratio = r };
            out[k + 1] = .{ .plane = cs.comp_scales[i], .row = c.head_dim / 16, .ratio = r };
            k += 2;
            if (lay.idx_wk != null) {
                out[k] = .{ .plane = cs.idx_codes[i], .row = c.index_head_dim / 2, .ratio = r };
                out[k + 1] = .{ .plane = cs.idx_scales[i], .row = c.index_head_dim / 32, .ratio = r };
                k += 2;
            }
        }
        return out[0..k];
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
        try m.pairJoin(); // (a pair's second pass may still read the image rows this replaces)
        if (rows.len > 0) {
            if (m.img_buf == null or m.img_buf.?.len < rows.len) {
                if (m.img_buf) |*b| b.free();
                m.img_buf = null;
                m.img_buf = try cuda.DeviceBuffer.alloc(m.ctx.d, rows.len);
            }
            const d = m.ctx.d;
    /// TF_DS_ENGRAM_KEEP_AHEAD at start (every rank alike): the decision (keep_ahead.decide), the check that the round's Engram
    /// buffers and the worker's share no byte, and the start line. A round reads through eh.hashes/flat/w/s/rows, its own AIO ring
    /// (eh.aio) and its own stages and records (Rows.round); the worker through wk_*, the prompt ring (eh.paio) and Rows.wk. The row
    /// cache is the one thing both touch (a lock a set).
    fn useKeepAhead(m: *Model) !void {
        const d = keep_ahead.decide(true, m.eh != null, wk_on, m.aio != null, m.paio != null);
        switch (d) {
            .off => {},
            .no_effect => |why| std.log.info("TF_DS_ENGRAM_KEEP_AHEAD=1: rank {d}: no effect, {s}", .{ m.rank, why }),
            .refuse => |why| {
                // the default: one WARN naming why and what it costs, the round joins the worker as before; set to 1 by the operator: the refusal
                if (speed_defaults.verdict(why, speed_defaults.defaulted("TF_DS_ENGRAM_KEEP_AHEAD")) == .fallback) {
                    std.log.warn("TF_DS_ENGRAM_KEEP_AHEAD: rank {d}: off: the Engram reads are not asynchronous (TF_DS_ENGRAM_AIO=0), so the round and the worker would share one reader pool; without it: {s} (set TF_DS_ENGRAM_KEEP_AHEAD=1 to make this a refusal, 0 to silence it)", .{ m.rank, speed_defaults.costOf("TF_DS_ENGRAM_KEEP_AHEAD") });
                    return;
                }
                std.log.err("{s}", .{why});
                return error.KeepAheadSharesReader;
            },
            .keep => {
                const eh = &m.eh.?;
                var round_r: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
                var worker_r: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
                try keepRanges(eh, &round_r, &worker_r);
                if (keep_ahead.firstShared(&round_r, &worker_r)) |at| {
                    std.log.err("TF_DS_ENGRAM_KEEP_AHEAD=1: the round's Engram scratch (range {d}) and the worker's (range {d}) share memory", .{ at[0], at[1] });
                    return error.KeepAheadSharesScratch;
                }
                m.keep_ahead = true;
                std.log.info("TF_DS_ENGRAM_KEEP_AHEAD=1: rank {d}: the decode round between a prompt's chunks leaves the Engram worker and its read-ahead running (the round's buffers, AIO ring and records are its own and checked apart; the row cache's per-set locks are the only shared state): 0 MiB of scratch more, the KV pool unchanged", .{m.rank});
            },
        }
    }

    /// Whether TF_DS_ENGRAM_KEEP_AHEAD could run on this rank, silently (useKeepAhead's decision and scratch check without its lines and
    /// refusals): what a live change to it may rely on when the start did not ask.
    fn keepAheadAble(m: *Model) bool {
        if (keep_ahead.decide(true, m.eh != null, wk_on, m.aio != null, m.paio != null) != .keep) return false;
        var round_r: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
        var worker_r: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
        keepRanges(&m.eh.?, &round_r, &worker_r) catch return false;
        return keep_ahead.firstShared(&round_r, &worker_r) == null;
    }

    /// The value a slot of class A has when a live change says `unset`: the start's.
    pub fn cfgStart(m: *const Model, slot: u8) u32 {
        return switch (slot) {
            @backingInt(live_config.Slot.keep_ahead) => @intFromBool(m.keep_ahead_start),
            @backingInt(live_config.Slot.kv_free_target) => m.kv_start_target_mib,
            else => live_config.cfg_unset,
        };
    }

    /// A cfg frame's changes on this rank, between two frames (rank 0 after it sent them, a follower when it read them): the same
    /// values on every rank by construction (rank 0 resolved `unset`, the frames are in order). A rank that could not take a
    /// value keeps what it has and says so; the Engram read-ahead is joined by the caller when the frame says it is touched.
        for (items) |it| switch (it.slot) {
            @backingInt(live_config.Slot.keep_ahead) => {
                const want = it.value != 0 and it.value != live_config.cfg_unset;
                if (want and !m.keep_ahead_able) {
                    std.log.warn("TF_DS_ENGRAM_KEEP_AHEAD: rank {d} cannot run it (cfg version {d}): left off", .{ m.rank, version });
                    continue;
                }
                if (want == m.keep_ahead) continue;
                m.keep_ahead = want;
                std.log.info("TF_DS_ENGRAM_KEEP_AHEAD is now {d} on rank {d} (cfg version {d})", .{ @intFromBool(want), m.rank, version });
            },
            @backingInt(live_config.Slot.kv_free_target) => try m.resizePool(version, it.value),
            else => std.log.warn("a cfg frame names slot {d}, which this build does not know (rank {d})", .{ it.slot, m.rank }),
        };
    }

    /// Every stream that can read or write the KV pool's planes: the model's stream, the pairs' second and third pass streams, and
    /// the round's side, prefetch and parity streams. The SSD tier's copies run on the model's stream (SpillDev). The test below
    /// holds this list to the Stream fields of Model and Round.
    pub const pool_stream_models = [_][]const u8{ "stream", "stream_b", "stream_c" };
    pub const pool_stream_round = [_][]const u8{ "side", "pf", "par" };

    fn poolStreams(m: *Model, out: *[16]cuda.Stream) usize {
        var n: usize = 0;
        out[n] = m.stream;
        n += 1;
        for ([_]?cuda.Stream{ m.stream_b, m.stream_c, m.rd.side, m.rd.pf, m.rd.par }) |x| if (x) |st| {
            out[n] = st;
            n += 1;
        };
        return n;
    }

    /// The device idle for a resize: the context, then each stream of `poolStreams` by itself.
    fn syncPoolStreams(m: *Model) !void {
        try m.ctx.synchronize();
        var st: [16]cuda.Stream = undefined;
        const n = m.poolStreams(&st);
        for (st[0..n]) |x| try x.synchronize();
    }

    /// The KV pool's bytes at a number of positions as the VMM pool maps them (each plane rounded up to the granularity).
    const VmmBytes = struct {
        m: *const Model,
        gran: usize,

        pub fn bytes(s: VmmBytes, positions: usize) usize {
            return prompt.Caches.vmmBytes(s.m.eng.c, s.m.w.layers, positions, s.m.streams, s.gran);
        }
    };

    /// TF_DS_KV_FREE_TARGET_GIB written live (class A, only with TF_DS_KV_VMM=1): every rank, at the same cfg frame, offers the
    /// positions its memory holds with `target_mib` kept free (the pool's mapped bytes plus what is free now, less the target), the ranks'
    /// parts go round the host star, and every rank reads the same table and takes the same decision (kv_pool.decideResize: the smallest
    /// fit, within every ceiling, never under the context and never under the extents rank 0 reports in use). The planes then grow or give
    /// back their tails with their addresses unchanged, so the graphs stay valid. A rank that cannot map what the decision asks makes
    /// every rank undo it (a second ballot). A target of 0 (an unset change with no start target) asks for the start's pool.
    fn resizePool(m: *Model, version: u64, target_mib: u32) !void {
        const kv = engine_api.kv_pool;
        // (a rank without the virtual-memory pool takes part all the same, so no rank waits for it: the decision names it)
        const vp: ?*kv_vmm.Driver = if (m.kv_vmm) |*v| v else null;
        if (vp == null) std.log.warn("TF_DS_KV_FREE_TARGET_GIB: rank {d} has no virtual-memory pool (TF_DS_KV_VMM=1 on every rank): nothing moves (cfg version {d})", .{ m.rank, version });
        const star = m.comm.host.star orelse return error.NoHostLinks;
        const target: u64 = if (target_mib == live_config.cfg_unset) @as(u64, m.kv_start_target_mib) << 20 else @as(u64, target_mib) << 20;
        try m.syncPoolStreams();
        const before = try m.ctx.memInfo();
        const gran: usize = if (vp) |v| @intCast(v.gran) else 1;
        const mapped: u64 = if (vp) |v| v.mapped else 0;
        const fit_pos: usize = if (vp == null) 0 else if (target == 0) m.pool_start else blk: {
            const have = before.free + mapped;
            break :blk if (have <= target) 0 else kv.fitIn(have - target, VmmBytes{ .m = m, .gran = gran });
        };
        var mine: kv.ResizePart = std.mem.zeroes(kv.ResizePart);
        mine.magic = kv.ResizePart.tag;
        mine.rank = m.rank;
        mine.vmm = @intFromBool(vp != null);
        mine.cur = m.pool_cap;
        mine.fit = fit_pos;
        mine.ceiling = m.pool_ceiling;
        mine.in_use = if (m.rank == 0) m.kv_live_floor else 0;
        mine.restoring = if (m.rank == 0) m.kv_live_loading else 0;
        mine.free_bytes = before.free;
        mine.mapped_bytes = mapped;
        mine.target_bytes = target;
        var table: [4]kv.ResizePart = undefined;
        const world: usize = @min(m.comm.world, 4);
        try star.allGather(std.mem.asBytes(&mine), std.mem.sliceAsBytes(table[0..world]));
        const d = kv.decideResize(table[0..world], m.ctx_cap);
        const from = m.pool_cap;
        const gib_f: f64 = 1 << 30;
        var failed: u8 = kv.ballot_ok;
        if (d.positions != from) {
            m.caches.resizeVmm(&m.eng, vp.?, d.positions) catch |e| {
                std.log.err("TF_DS_KV_FREE_TARGET_GIB: rank {d} could not move the pool from {d} to {d} positions ({s}); every rank undoes it (cfg version {d})", .{ m.rank, from, d.positions, @errorName(e), version });
                failed = if (e == error.OutOfDeviceMemory or e == error.OutOfMemory) kv.ballot_memory else kv.ballot_error;
            };
        }
        // every rank's result, by rank: the outcome names the rank that ran out of memory
        var ballots: [4]u8 = undefined;
        try star.allGather(&[1]u8{failed}, ballots[0..world]);
        const out = kv.resizeOutcome(d, ballots[0..world]);
        const after_free = if (m.ctx.memInfo()) |i| i.free else |_| 0;
        if (out.verdict == .refused_memory or out.verdict == .refused_error) {
            // some rank failed: the ones that moved go back (a grow's undo, a shrink's regrow of what it gave back: shrinks cannot fail)
            if (failed == kv.ballot_ok and d.positions != from) m.caches.resizeVmm(&m.eng, vp.?, from) catch {};
            kv.report.resize_version = version;
            kv.report.resize_verdict = out.verdict;
            kv.report.resize_rank = out.rank;
            kv.report.resize_from = from;
            kv.report.resize_wanted = d.wanted;
            std.log.warn("TF_DS_KV_FREE_TARGET_GIB: the pool stays at {d} positions on rank {d}: REFUSED, rank {d} {s} mapping the change (cfg version {d}); every rank undid it", .{ from, m.rank, out.rank, if (out.verdict == .refused_memory) "ran out of memory" else "failed", version });
            return;
        }
        try m.syncPoolStreams();
        m.pool_cap = d.positions;
        kv.report.pool_active = d.positions;
        kv.report.resize_version = version;
        kv.report.resize_verdict = d.verdict;
        kv.report.resize_rank = out.rank;
        kv.report.resize_from = from;
        kv.report.resize_wanted = d.wanted;
        std.log.info("TF_DS_KV_FREE_TARGET_GIB is now {d:.2} GiB on rank {d} (cfg version {d}): {s}; the pool was {d} positions and is {d} ({d:.2} GiB mapped, {d:.2} GiB free before and {d:.2} GiB after on this Spark; the ranks fit {d} positions, the smallest by rank {d}; rank 0 reports extents in use to {d})", .{
            @as(f64, @floatFromInt(target)) / gib_f,      m.rank,                                      version,
            switch (d.verdict) {
                .same => "no change: the ranks' memory holds the pool as it is",
                .grow => "the pool grew",
                .shrink => "the pool gave back its tail",
                .shrink_partial => "the pool shrank as far as it is free: a stream or a kept conversation holds an extent above the memory's size, so the target is not reached",
                .blocked_by_use => "NOT shrunk: a stream or a kept conversation holds an extent to the pool's end, or a disk restore is still loading into the pool",
                .refused_context => "REFUSED: fewer positions fit than the context (the context is never lowered)",
                .refused_disagree => "REFUSED: the ranks' offers do not describe one pool and one target",
                .refused_memory, .refused_error => unreachable, // (returned above)
                .refused_no_vmm => "REFUSED: a rank's pool is not in virtual memory (give TF_DS_KV_VMM=1 to every rank)",
            },
            from,                                         d.positions,                                 @as(f64, @floatFromInt(if (vp) |v| v.mapped else 0)) / gib_f,
            @as(f64, @floatFromInt(before.free)) / gib_f, @as(f64, @floatFromInt(after_free)) / gib_f, d.wanted,
            d.limiting_rank,                              table[0].in_use,
        });
    }

    pub fn fillBegin(m: *Model, slot: usize, base: usize) !void {
        try m.pairJoin();
        m.fill_t0 = m.now();
        m.pair_in_fill = 0;
        if (m.eh) |*eh| eh.ka.newPrompt();
        if (m.erows) |r| m.dup_fill = .{ r.dup_misses.load(.monotonic), r.dup_unique.load(.monotonic) };
        if (slot >= m.streams) return error.BadSlot;
        m.spans.clearRetainingCapacity();
        m.ch.spans = &.{};
        m.view = try m.caches.view(&m.eng, slot, base);
        for (m.rings, m.ring_view) |r, *v| v.* = r + slot * m.eng.ringBytes();
        m.fill_slot = slot;
    }

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
                eh.use_rows = @splat(null);
                eh.use_stage = @splat(null);
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
            // the rows are uploaded (a pageable copy is staged before it returns; a pinned one is done when its event is): the
            // units' slots go back to the worker
            try prompt.engramRelease(eh, e.s, j, first + units);
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
        // a pass: a remainder under the resume rows, whole resume units up to the pass rows, or, with TF_DS_PASS_TAIL, the
        // final pass's remainder up to the pass rows (it starts on the resume grid)
        if (n == 0 or n > m.pass_rows or seq.len != start + n) return error.BadChunk;
        if (n > chunk_rows and n % chunk_rows != 0) {
            if (!prompt2d.pass_tail_asked or start % chunk_rows != 0) return error.BadChunk;
        }
        if (m.eh) |*eh| eh.ka.chunkBegins();
        const e = &m.eng;
        const c = e.c;
        const ch = &m.ch;
        try ch.view(n, chunk_rows);
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
        ch.encoder_only = prompt.encoderOnly(start, n, replay);
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
                    (start - eh.wk_start) / chunk_rows <= prompt.engramUploaded(eh, 0);
                if (!same) try prompt.engramWorkerStart(eh, c, m.fill_seq, start, chunk_rows, prompt2d.pass_from_asked, m.pass_rows);
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
                    // TF_DS_BOUNDARY_OWNED: an encoder-only full chunk's boundary on owned rows, its cache rows
                    // gathered; the chunk stops here, as replayCut stops it
                    try prompt.boundaryUpdate(e, ch, &m.view, &shared, li, r);
                    if (pt) |t| {
                        try e.s.synchronize();
                        try t.finish();
                    }
                    return false;
                }
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

                    .poll, .backoff => slices = try issue_lead.pollWait(gate, &w),
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
        const ka: keep_ahead.Taken = if (m.eh) |*eh| eh.ka.take() else .{ .wait_ns = 0, .chunks = 0 };
        const starts: u64 = if (m.eh) |*eh| eh.ka.starts else 0;
        pairLog("ROCI {s} at {d} (rank {d}): engine thread CPU {d:.1} ms of {d:.1} ms wall ({d:.2}); issue lead {d}: {d} layers, the host asleep {d:.1} ms in {d} sleeps of {d} slices, {d} layers entered with nothing queued inside the pass and {d} at its first layer; Engram worker starts this prompt {d}, first-layer wait after a round {d:.1} ms in {d} chunks", .{ what, at, m.comm.rank, cpu, wall, if (wall > 0) cpu / wall else 0.0, if (m.ilead) |g| g.n else 0, st.layers, ms(st.sleep_ns), st.sleeps, st.slices, st.bare, st.bare_first, starts, @as(f64, @floatFromInt(ka.wait_ns)) / 1e6, ka.chunks });
        if (m.partner_link) |pl| {
            // TF_DS_ENGRAM_PARTNER, what the worker's steps did since the last line (the counts of a pass in flight may fall in the next line)
            const c = pl.c.since(m.partner_seen);
            m.partner_seen = pl.c;
            const mb = struct {
                fn of(records: u64) f64 {
                    return @as(f64, @floatFromInt(records * rocinante_mod.host_rec)) / 1e6;
                }
            }.of;
            pairLog("ROCI {s} at {d} (rank {d}) partner: {d} steps, {d} split, {d} mismatched, {d} header late, {d} half late, {d} skipped, {d} declined; records read here {d}, sent {d} ({d:.1} MB), received {d} ({d:.1} MB)", .{ what, at, m.comm.rank, c.steps, c.split, c.mismatch, c.header_late, c.half_late, c.skipped, c.declined, c.records_read, c.records_sent, mb(c.records_sent), c.records_got, mb(c.records_got) });
            pairLog("ROCI {s} at {d} (rank {d}) partner time: header wait {d:.1} ms, records wait {d:.1} ms in {d} waits ({d} looks, {d} naps, the longest {d:.2} ms); own half read {d:.1} ms (the whole lists at that rate {d:.1} ms), rest read {d:.1} ms, copies {d:.1} ms", .{ what, at, m.comm.rank, ms(c.hdr_wait_ns), ms(c.rec_wait_ns), c.waits, c.wait_looks, c.wait_naps, ms(c.wait_max_ns), ms(c.read_ns), ms(c.full_ns), ms(c.redo_ns), ms(c.copy_ns) });
        }
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
        if (m.eh) |*eh| eh.ka.chunkBegins();
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
                (start_a - eh.wk_start) / chunk_rows <= prompt.engramUploaded(eh, 0);
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
        if (m.pair_log) {
            const eh = &m.eh.?;
            const aio_s: u64 = if (eh.paio) |x| x.st_submit_ns.load(.monotonic) else 0;
            const aio_w: u64 = if (eh.paio) |x| x.st_wait_ns.load(.monotonic) else 0;
            const aio_max: u32 = if (eh.paio) |x| x.st_inflight_max.swap(x.st_inflight.load(.monotonic), .monotonic) else 0;
            const aio_c: u64 = if (eh.paio) |x| x.st_calls.load(.monotonic) else 0;
            const aio_r: u64 = if (eh.paio) |x| x.st_reads.load(.monotonic) else 0;
            const aio_cmax: u64 = if (eh.paio) |x| x.st_call_max_ns.swap(0, .monotonic) else 0;
            const aio_wk: u64 = if (eh.paio) |x| x.st_wakes.load(.monotonic) else 0;
            const aio_we: u64 = if (eh.paio) |x| x.st_wake_events.load(.monotonic) else 0;
            const cur: [23]u64 = .{ eh.inst_steps.load(.monotonic), eh.inst_hash_ns.load(.monotonic), eh.inst_gather_ns.load(.monotonic), eh.inst_decode_ns.load(.monotonic), eh.inst_rows.load(.monotonic), eh.inst_miss.load(.monotonic), eh.inst_bytes.load(.monotonic), eh.inst_cpu_ns.load(.monotonic), eh.inst_nivcsw.load(.monotonic), eh.inst_big.load(.monotonic), eh.inst_small.load(.monotonic), aio_s, aio_w, aio_c, aio_r, aio_wk, aio_we, eh.wk_go.sleeps.load(.monotonic), eh.wk_go.asleep_ns.load(.monotonic), eh.wk_go.rung.load(.monotonic), eh.wk_ready.sleeps.load(.monotonic), eh.wk_ready.asleep_ns.load(.monotonic), eh.wk_ready.rung.load(.monotonic) };
            var d: [23]u64 = undefined;
            for (&d, cur, eh.inst_snap) |*x, y, z| x.* = y -% z;
            eh.inst_snap = cur;
            const cores = eh.inst_cores.swap(0, .monotonic);
            var core_text: [160]u8 = undefined;
            var cw = std.Io.Writer.fixed(&core_text);
            for (0..64) |ci| if (cores >> @intCast(ci) & 1 != 0) cw.print("{s}{d}", .{ if (cw.end == 0) "" else ",", ci }) catch {};
            const ms = struct {
                fn of(t_ns: u64) f64 {
                    return @as(f64, @floatFromInt(t_ns)) / 1e6;
                }
            }.of;
            const dm: u64 = if (m.erows) |r| r.dup_misses.load(.monotonic) else 0;
            const du: u64 = if (m.erows) |r| r.dup_unique.load(.monotonic) else 0;
            const dm_line = dm -% m.dup_snap[0];
            const du_line = du -% m.dup_snap[1];
            const dm_fill = dm -% m.dup_fill[0];
            const du_fill = du -% m.dup_fill[1];
            m.dup_snap = .{ dm, du };
            // the worker's steps' miss lists since the last line: chunk (absolute index) and layer slot, count, order hash, set hash; the
            // plans anything else made (they fill the cache too) since the last line
            if (m.erows) |r| {
                var txt: [420]u8 = undefined;
                var tw = std.Io.Writer.fixed(&txt);
                const n_all = r.steps_n.load(.acquire);
                var from = m.steps_seen;
                if (n_all -% from > roci_step_print) from = n_all - roci_step_print;
                for (from..n_all) |si| {
                    const rec = r.steps[si % engram_rows.step_ring];
                    tw.print("{s}c{d}l{d}:{d}:{x:0>8}:{x:0>8}", .{ if (tw.end == 0) "" else " ", rec.tag >> 8, rec.tag & 0xff, rec.count, rec.ord, rec.set }) catch break;
                }
                m.steps_seen = n_all;
                const other = r.other_plans.load(.monotonic);
                pairLog("ROCI pair {d} (rank {d}) misses: {s}; other plans {d}", .{ m.pair_count, m.comm.rank, tw.buffered(), other -% m.other_seen });
                m.other_seen = other;
            }
            const share = struct {
                fn of(misses: u64, uniq: u64) f64 {
                    return if (misses == 0) 0.0 else 100.0 * @as(f64, @floatFromInt(misses -| uniq)) / @as(f64, @floatFromInt(misses));
                }
            }.of;
            // the sleeping waits (wait_bell) since the last line: the worker's for a free slot (ended by the engine thread's ring or its 10 ms backstop),
            // the engine thread's for rows (ended by the worker's ring or its timeout; the timeouts are the duties' turns)
            pairLog("ROCI pair {d} (rank {d}) Engram waits: worker slept {d} times for {d:.1} ms ({d} woken by a ring); engine thread slept {d} times for {d:.1} ms ({d} woken by a ring, {d} ended by the timeout or an earlier ring)", .{ m.pair_count, m.comm.rank, d[17], ms(d[18]), d[19], d[20], ms(d[21]), d[22], d[20] -| d[22] });
            if (m.comm.rank != 0) {
                // the followers' balance figures: what this rank asked and read in the pair, its repeats and its worker's gather time
                pairLog("ROCI pair {d} (rank {d}): {d} rows asked, {d} read, repeats {d:.1}% of this line's misses ({d:.1}% of the fill's), worker gather {d:.1} ms in {d} steps", .{ m.pair_count, m.comm.rank, d[4], d[5], share(dm_line, du_line), share(dm_fill, du_fill), ms(d[2]), d[0] });
            } else {
                // three lines a pair, each under the log's 512-byte limit: the host's issue, the worker, the AIO ring and the repeats
                pairLog("ROCI pair {d} at {d}: host issue {d:.1} ms (engine thread CPU {d:.1} ms), of it waiting for the Engram worker {d:.1} ms and in the Engram upload calls (copies blocked included) {d:.1} ms", .{ m.pair_count, start_a, ms(m.now() - host_t0), ms(pair_nap.cpuNow() - cpu_t0), ms(eh.inst_wait_ns), ms(eh.inst_apply_ns) });
                pairLog("ROCI pair {d} worker: {d} steps, hash {d:.1} ms, gather {d:.1} ms, decode {d:.1} ms, {d} rows asked, {d} read ({d:.1} MB); thread CPU {d:.1} ms, {d} involuntary switches, core samples X925 {d} / A725 {d}, cores [{s}]", .{ m.pair_count, d[0], ms(d[1]), ms(d[2]), ms(d[3]), d[4], d[5], @as(f64, @floatFromInt(d[6])) / 1e6, ms(d[7]), d[8], d[9], d[10], cw.buffered() });
                pairLog("ROCI pair {d} AIO: io_submit {d:.1} ms in {d} calls of {d:.0} reads on average (longest {d:.2} ms), waiting for completions {d:.1} ms, slept {d} times for {d:.0} completions on average, most reads in flight {d}; repeats {d:.1}% of this line's misses, {d:.1}% of the fill's ({d} different ids in {d} misses)", .{ m.pair_count, ms(d[11]), d[13], if (d[13] == 0) 0.0 else @as(f64, @floatFromInt(d[14])) / @as(f64, @floatFromInt(d[13])), ms(aio_cmax), ms(d[12]), d[15], if (d[15] == 0) 0.0 else @as(f64, @floatFromInt(d[16])) / @as(f64, @floatFromInt(d[15])), aio_max, share(dm_line, du_line), share(dm_fill, du_fill), du_line, dm_line });
            }
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

    /// What the leader reads of a round's logits: the device's argmax, its first `count` candidates a row, or nothing
    /// there (the whole rows: roundLogits).
    pub const Head = union(enum) { none, argmax, candidates: usize };

    /// TF_DS_CAND_FIRST's start line.
    pub fn candFirstLine() []const u8 {
        return "TF_DS_CAND_FIRST=1: a round's argmax or first candidates run before the drafter's absorb; the host waits for them alone and draws while the absorb runs";
    }

    /// TF_DS_CAND_FIRST: verify with the round's head reads (`head`) between the forward and the drafter's absorb on
    /// the stream; the host waits for an event after them alone, and the absorb runs on while it draws and sends the
    /// next round's frames (the next round's work queues behind it). headArgmax / headCandidates read the results. The
    /// reads take only the round's logits and write only their own buffers, and the absorb neither, so every byte is
    /// as before. Head none, or a profiled round: verify, then the reads as before.
    pub fn verifyHead(m: *Model, rows: round.Rows, absorb: []const bool, head: Head) !void {
        try m.pairJoin();
        const R = rows.ids.len;
        // TF_DS_HEAD_CAND: the device's merge needs the count each node keeps (the frame's head byte); without it, 0.
        const hb: u8 = switch (head) {
            .none => 0,
            .argmax => if (m.head_key) head_cand.key else 0,
            .candidates => |count| if (m.head_cand) head_cand.kFor(count) else 0,
        };
        m.rd.head_k = hb;
        if (head == .none or m.prof) {
            try m.verify(rows, absorb, hb);
            if (hb == 0) try m.launchHead(head, R); // (hb != 0: round.forward's last stretch ran it)
            try m.stream.synchronize();
            return;
        }
        try m.forwardRows(rows);
        if (hb == 0) try m.launchHead(head, R); // (hb != 0: round.forward's last stretch ran it)
        try m.head_ready.?.record(m.stream);
        try m.absorbRows(rows, absorb);
        if (m.erows) |r| try r.roundCheck(); // the forward's stream host steps came before the event
    }

    /// The round's head reads on the stream: the device's argmax into m.amax, or its first `count` candidates a row
    /// into m.cand.
    fn launchHead(m: *Model, head: Head, R: usize) !void {
        switch (head) {
            .none => {},
            .argmax => try m.ops.argmaxRows(m.stream, m.rd.logits, m.vocab, m.vocab, R, m.amax),
            .candidates => |count| {
                const p = m.candPtrs();
                try m.ops.topCandRows(m.stream, m.rd.logits, m.vocab, m.vocab, R, count, p.vals, p.ids, p.flags);
            },
        }
    }

    /// m.cand's three arrays: vals [max_rows, max_candidates] fp32, ids [.., ..] i64, flags [max_rows] i32.
    fn candPtrs(m: *const Model) struct { vals: u64, ids: u64, flags: u64 } {
        const mc = sampling.max_candidates;
        const ids = m.cand + round.max_rows * mc * 4;
        return .{ .vals = m.cand, .ids = ids, .flags = ids + round.max_rows * mc * 8 };
    }

    /// The round's greedy tokens, R rows, on the host, once the argmax is done (roundArgmax's, or verifyHead's).
    pub fn headArgmax(m: *Model, R: usize) ![]const u32 {
        const d = m.ctx.d;
        try d.check(d.api.cuMemcpyDtoH_v2(&m.amax_host, m.amax, R * 4), "cuMemcpyDtoH");
        return m.amax_host[0..R];
    }

    /// The round's first `count` candidates a row on the host, once their kernel is done (roundCandidates', or
    /// verifyHead's).
    pub fn headCandidates(m: *Model, R: usize, count: usize) !Cands {
        const d = m.ctx.d;
        const p = m.candPtrs();
        try d.check(d.api.cuMemcpyDtoH_v2(&m.cand_vals, p.vals, R * count * 4), "cuMemcpyDtoH");
        try d.check(d.api.cuMemcpyDtoH_v2(&m.cand_ids, p.ids, R * count * 8), "cuMemcpyDtoH");
        try d.check(d.api.cuMemcpyDtoH_v2(&m.cand_flags, p.flags, R * 4), "cuMemcpyDtoH");
        return .{ .vals = m.cand_vals[0 .. R * count], .ids = m.cand_ids[0 .. R * count], .flags = m.cand_flags[0..R] };
    }

    /// One round over every stream's window (rows.windows, each with its stream's slot and extent), then each window
    /// whose stream drafts has its rows' taps absorbed into its slot's drafter rings at its positions (the served
    /// eager absorb: rows past the ones the round keeps sit where the drafter never reads, and the next window
    /// overwrites them). absorb[i]: window i's stream drafts.
    /// `head`: the round's head (Lanes' frame): 0 the whole rows (rd.logits), head_cand.key TF_DS_HEAD_KEY's bests,
    /// else TF_DS_HEAD_CAND's count a node.
    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool, head: u8) !void {
        try m.pairJoin();
        if (m.eh) |*eh| eh.ka.roundRan();
        if (head != 0 and !(if (head == head_cand.key) m.head_key else m.head_cand)) return error.HeadCandOff;
        m.rd.head_k = head;
        try m.forwardRows(rows);
        try m.absorbRows(rows, absorb);
        if (m.erows) |r| try r.roundCheck(); // a stream host step's failed read fails the round
        if (m.prof) m.t_absorb = m.now() - t1;
    }

    /// verify's forward over the round's rows, on the stream (a profiled round: timed, the stream synchronized).
    fn forwardRows(m: *Model, rows: round.Rows) !void {
        const e = &m.eng;
        if (m.prof) try m.stream.synchronize(); // (the GPU idle, so the issue time shows whether the host keeps it fed)
        try round.forward(e, &m.rd, &m.ch, &m.caches, m.rings, if (m.eh) |*x| x else null, &.{}, rows, m.ctx_cap, null, if (m.graphs) |*g| g else null);
        if (m.prof) {
            m.t_enqueue = m.now() - t0;
            try m.stream.synchronize();
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
    }

    /// verify's absorb: each window whose stream drafts has its rows' taps absorbed into its slot's drafter rings, on
    /// the stream (nothing waited for).
    fn absorbRows(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        const e = &m.eng;
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
    }

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
        try m.launchHead(.argmax, R);
        try m.stream.synchronize();
        return m.headArgmax(R);
    }

    /// A round's candidates on the host: vals and ids [R, count], flags [R] (non-zero: that row must be read whole).
    pub const Candidates = struct { vals: []const f32, ids: []const i64, flags: []const i32 };

    /// Track B: the round's first `count` candidates a row (sampling.topCandidates' set and order) on the host, from the
    /// device's logits: vals and ids [R, count], flags [R] (non-zero: that row must be read whole, roundLogits).
    pub const Cands = Candidates;

    pub fn roundCandidates(m: *Model, R: usize, count: usize) !Candidates {
        try m.launchHead(.{ .candidates = count }, R);
        try m.stream.synchronize();
        return m.headCandidates(R, count);
    }

    /// TF_DS_HEAD_CAND: a round's first `count` candidates a row merged from the four nodes' parts (their first `k`
    /// of their own columns, verify's head): the baseline set and order with no flag; null when the baseline kernel could flag a
    /// row (a node flagged one, or ties at the cut the parts cannot rule out): the round then reads the whole head
    /// (headWhole on every rank, then roundCandidates).
    pub fn headCandidatesMerged(m: *Model, R: usize, count: usize, k: u8) !?Candidates {
        const d = m.ctx.d;
        const tw = m.two orelse return error.HeadCandNeeds2D;
        if (!m.head_cand or m.rd.head_k != k or count == 0 or count > k or R > round.max_rows) return error.HeadCandOff;
        const lay = head_cand.layout(R, k);
        // (no wait on the stream: rd.call's only writer is the last stretch's head exchange, enqueued before the round's head event, which verifyHead's wait or verify's sync has passed; the drafter's absorb behind it never writes it)
        try d.check(d.api.cuMemcpyDtoH_v2(m.cand_parts.ptr, m.rd.call, 4 * lay.bytes), "cuMemcpyDtoH");
        if (!head_cand.merge(m.cand_parts[0 .. 4 * lay.bytes], lay, head_cand.vocabOffsets(tw.hw), count, m.cand_vals[0 .. R * count], m.cand_ids[0 .. R * count])) return null;
        @memset(m.cand_flags[0..R], 0);
        return .{ .vals = m.cand_vals[0 .. R * count], .ids = m.cand_ids[0 .. R * count], .flags = m.cand_flags[0..R] };
    }

    /// TF_DS_HEAD_KEY: a greedy round's tokens, R rows, merged from the four nodes' bests (verify's head
    /// head_cand.key): the baseline argmax on the whole rows, on the host.
    pub fn headTokens(m: *Model, R: usize) ![]const u32 {
        const d = m.ctx.d;
        const tw = m.two orelse return error.HeadCandNeeds2D;
        if (!m.head_key or m.rd.head_k != head_cand.key or R > round.max_rows) return error.HeadCandOff;
        const n = 4 * R * head_cand.key_row;
        // (no wait on the stream: as headCandidatesMerged, the head event ordered rd.call)
        try d.check(d.api.cuMemcpyDtoH_v2(m.cand_parts.ptr, m.rd.call, n), "cuMemcpyDtoH");
        head_cand.mergeKeys(m.cand_parts[0..n], R, head_cand.vocabOffsets(tw.hw), m.amax_host[0..R]);
        return m.amax_host[0..R];
    }

    /// The round's whole head into rd.logits after its candidates fell back (every rank, in the same frame): the
    /// exchange and the copies a whole-row round runs, on the round's own head columns (rd.hl).
    pub fn headWhole(m: *Model, R: usize) !void {
        if (m.rd.head_k == 0 or R != m.rd.r) return error.HeadCandOff;
        try round.headWhole(&m.eng, &m.rd, R, null);
        try m.stream.synchronize();
    }

    pub fn roundLogits(m: *Model, R: usize) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.rd.logits, R * m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0 .. R * m.vocab];
    }

    /// The prompt's logits on the host (after a fill's last chunk).
    pub fn promptLogitsHost(m: *Model) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.ch.head_g, m.vocab * 4), "cuMemcpyDtoH");
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
        if (m.eh) |*eh| eh.ka.roundRan();
        if (m.dr == null) return error.NoDrafter;
        const dr = &m.dr.?;
        const pool = &m.dpool.?;
        for (q0, slots) |q, s| pool.absorbed[@intCast(s)] = @intCast(q);

test "TF_DS_PASS_BESIDE is read after TF_DS_PASS_ROWS, refuses a bad start, and reaches the host as its beside rows" {
    const src = @embedFile("model.zig");
    const rows = std.mem.indexOf(u8, src, "prompt2d.pass_rows_asked = try readPassRows(") orelse return error.TestUnexpectedResult;
    const beside = std.mem.indexOf(u8, src, "prompt2d.pass_beside_asked = try readPassBeside(prompt2d.pass_rows_asked)") orelse return error.TestUnexpectedResult;
    try std.testing.expect(rows < beside);
    try std.testing.expect(std.mem.indexOf(u8, src, "return error.BadPassBeside;") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("native.zig"), ".beside = prompt2d.pass_beside_asked") != null);
}

test "TF_DS_ROUNDS_BESIDE is read after TF_DS_PASS_BESIDE, refuses a bad value or the share beside it at the start, and reaches the host" {
    const src = @embedFile("model.zig");
    const beside = std.mem.indexOf(u8, src, "prompt2d.pass_beside_asked = try readPassBeside(") orelse return error.TestUnexpectedResult;
    const rounds = std.mem.indexOf(u8, src, "prompt2d.rounds_beside_asked = try readRoundsBeside()") orelse return error.TestUnexpectedResult;
    try std.testing.expect(beside < rounds);
    try std.testing.expect(std.mem.indexOf(u8, src, "return error.BadRoundsBeside;") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "return error.RoundsBesideWithShare;") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("native.zig"), ".rounds = prompt2d.rounds_beside_asked") != null);
}

test "the start-up settings carry TF_DS_PF_SPLIT and the kernel switches as Model.open read them: 0 by default (the baseline kernels)" {
    const o: Options = .{ .model_dir = "", .cache_dir = "", .kit_dir = "", .rank = 3, .world = 4, .master = .{ 10, 255, 0, 33 }, .port = 29600, .pool = 1 << 24, .drafts = true, .graphs = true, .round_rows = 48, .streams = 16 };
    const was = .{ prompt2d.pf_split_asked, prompt2d.woa_w8_asked };
    defer {
        prompt2d.pf_split_asked = was[0];
        prompt2d.woa_w8_asked = was[1];
    }
    prompt2d.pf_split_asked = .off;
    prompt2d.woa_w8_asked = false;
    try std.testing.expectEqual(@as(u32, 0), startSettings(o).pf_split);
    try std.testing.expectEqual(@as(u32, 0), startSettings(o).kern);
    prompt2d.woa_w8_asked = true;
    try std.testing.expectEqual(prompt2d.kern_woa_w8, startSettings(o).kern);
    prompt2d.woa_w8_asked = false;
    const was_once = .{ prompt2d.quant_once_asked, prompt2d.pf_gemm5_asked };
    defer {
        prompt2d.quant_once_asked = was_once[0];
        prompt2d.pf_gemm5_asked = was_once[1];
    }
    prompt2d.quant_once_asked = true;
    try std.testing.expectEqual(prompt2d.kern_quant_once, startSettings(o).kern);
    prompt2d.quant_once_asked = false;
    prompt2d.pf_gemm5_asked = true;
    try std.testing.expectEqual(prompt2d.kern_pf_gemm5, startSettings(o).kern);
    prompt2d.pf_gemm5_asked = false;
    const was_gt = prompt2d.group_tiles_asked;
    defer prompt2d.group_tiles_asked = was_gt;
    prompt2d.group_tiles_asked = true;
    try std.testing.expectEqual(prompt2d.kern_group_tiles, startSettings(o).kern);
    prompt2d.group_tiles_asked = false;
    const was_tk = prompt2d.topk_tiles_asked;
    defer prompt2d.topk_tiles_asked = was_tk;
    prompt2d.topk_tiles_asked = true;
    try std.testing.expectEqual(prompt2d.kern_topk_tiles, startSettings(o).kern);
    prompt2d.topk_tiles_asked = false;
    const was_po = prompt2d.idx_proj_owned_asked;
    defer prompt2d.idx_proj_owned_asked = was_po;
    prompt2d.idx_proj_owned_asked = true;
    try std.testing.expectEqual(prompt2d.kern_idx_proj_owned, startSettings(o).kern);
    prompt2d.idx_proj_owned_asked = false;
    const was_br = prompt2d.bf16_rows_asked;
    defer prompt2d.bf16_rows_asked = was_br;
    prompt2d.bf16_rows_asked = true;
    try std.testing.expectEqual(prompt2d.kern_bf16_rows, startSettings(o).kern);
    prompt2d.bf16_rows_asked = false;
    const was_r128 = prompt2d.pf_run128_asked;
    defer prompt2d.pf_run128_asked = was_r128;
    prompt2d.pf_run128_asked = true;
    try std.testing.expectEqual(prompt2d.kern_pf_run128, startSettings(o).kern);
    prompt2d.pf_run128_asked = false;
    const was_df = prompt2d.decode_fuse_asked;
    defer prompt2d.decode_fuse_asked = was_df;
    prompt2d.decode_fuse_asked = true;
    try std.testing.expectEqual(prompt2d.kern_decode_fuse, startSettings(o).kern);
    prompt2d.decode_fuse_asked = false;
    const was_d4 = prompt2d.draft_4way_asked;
    defer prompt2d.draft_4way_asked = was_d4;
    prompt2d.draft_4way_asked = true;
    try std.testing.expectEqual(prompt2d.kern_draft_4way, startSettings(o).kern);
    prompt2d.draft_4way_asked = false;
    const was_dm = prompt2d.draft_main_4way_asked;
    defer prompt2d.draft_main_4way_asked = was_dm;
    prompt2d.draft_main_4way_asked = true;
    try std.testing.expectEqual(prompt2d.kern_draft_main_4way, startSettings(o).kern);
    prompt2d.draft_main_4way_asked = false;
    const was_lf = prompt2d.hc_late_fork_asked;
    defer prompt2d.hc_late_fork_asked = was_lf;
    prompt2d.hc_late_fork_asked = true;
    try std.testing.expectEqual(prompt2d.kern_hc_late_fork, startSettings(o).kern);
    prompt2d.hc_late_fork_asked = false;
    const was_au = prompt2d.attn_u_asked;
    defer prompt2d.attn_u_asked = was_au;
    prompt2d.attn_u_asked = true;
    try std.testing.expectEqual(prompt2d.kern_attn_u, startSettings(o).kern);
    prompt2d.attn_u_asked = false;
    const was_iu = prompt2d.idx_u_asked;
    defer prompt2d.idx_u_asked = was_iu;
    prompt2d.idx_u_asked = true;
    try std.testing.expectEqual(prompt2d.kern_idx_u, startSettings(o).kern);
    prompt2d.idx_u_asked = false;
    const was_iv = prompt2d.idx_v_asked;
    defer prompt2d.idx_v_asked = was_iv;
    prompt2d.idx_v_asked = true;
    try std.testing.expectEqual(prompt2d.kern_idx_v, startSettings(o).kern);
    prompt2d.idx_v_asked = false;
    const was_aud = prompt2d.attn_u_decode_asked;
    defer prompt2d.attn_u_decode_asked = was_aud;
    prompt2d.attn_u_decode_asked = true;
    try std.testing.expectEqual(prompt2d.kern_attn_u_decode, startSettings(o).kern);
    prompt2d.attn_u_decode_asked = false;
    const was_sf = prompt2d.mma_sf_asked;
    defer prompt2d.mma_sf_asked = was_sf;
    prompt2d.mma_sf_asked = true;
    try std.testing.expectEqual(prompt2d.kern_mma_sf, startSettings(o).kern);
    prompt2d.mma_sf_asked = false;
    const was_sfx = prompt2d.mma_sf_exact_asked;
    defer prompt2d.mma_sf_exact_asked = was_sfx;
    prompt2d.mma_sf_exact_asked = true;
    try std.testing.expectEqual(prompt2d.kern_mma_sf_exact, startSettings(o).kern);
    prompt2d.mma_sf_exact_asked = false;
    const was_r2 = prompt2d.gemm_r2_asked;
    defer prompt2d.gemm_r2_asked = was_r2;
    prompt2d.gemm_r2_asked = true;
    try std.testing.expectEqual(prompt2d.kern_gemm_r2, startSettings(o).kern);
    prompt2d.gemm_r2_asked = false;
    const was_hc = prompt2d.head_cand_asked;
    defer prompt2d.head_cand_asked = was_hc;
    prompt2d.head_cand_asked = true;
    try std.testing.expectEqual(prompt2d.kern_head_cand, startSettings(o).kern);
    prompt2d.head_cand_asked = false;
    const was_hk = prompt2d.head_key_asked;
    defer prompt2d.head_key_asked = was_hk;
    prompt2d.head_key_asked = true;
    try std.testing.expectEqual(prompt2d.kern_head_key, startSettings(o).kern);
    prompt2d.head_key_asked = false;
    for ([_]prompt2d.Qgate{ .head, .body_fp8, .body_ulp }) |q| {
        prompt2d.qgate_canary_asked = q;
    }
    prompt2d.qgate_canary_asked = .off;
    // the second kernel word: TF_DS_HC_2S travels in kern2, never in kern
    const was_h2 = prompt2d.hc_2s_asked;
    defer prompt2d.hc_2s_asked = was_h2;
    prompt2d.hc_2s_asked = true;
    try std.testing.expectEqual(prompt2d.kern2_hc_2s, startSettings(o).kern2);
    try std.testing.expectEqual(@as(u32, 0), startSettings(o).kern & ~(prompt2d.kern_attn_epi_fused));
    prompt2d.hc_2s_asked = false;
    try std.testing.expectEqual(@as(u32, 0), startSettings(o).kern2);
    for ([_]mx.PfSplit{ .table, .a, .b, .c6, .c7 }) |sw| {
        prompt2d.pf_split_asked = sw;
        try std.testing.expectEqual(@backingInt(sw), startSettings(o).pf_split);
    }
}

test "the four-way drafter is off by its can bit without a drafter, the checkpoint's own weights, a pair, the decode kernels or the kit's variants, decided before the weights load" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    c.format = .native;
    // three kits: the extra set of the stages' variants (draft.zig's fixture, from the kit), an empty one, none
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "with/aot-extra", "empty/aot-extra", "none/aot" }) |sub| try tmp.dir.createDirPath(io, sub);
    try tmp.dir.writeFile(io, .{ .sub_path = "with/aot-extra/aot.json", .data = @embedFile("fixtures/kit-drafter-extra.json") });
    try tmp.dir.writeFile(io, .{ .sub_path = "empty/aot-extra/aot.json", .data = "{\"kernels\": []}" });
    const root = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    const with = try std.fs.path.join(a, &.{ root, "with" });
    const sp: plan.Split = .{ .rank = 1, .world = 2, .pair = 0, .parity = true };
    try std.testing.expectEqual(@as(?[]const u8, null), try draftFourWayOff(a, io, &c, sp, with, true, true));
    const Why = struct {
        fn says(text: ?[]const u8, part: []const u8) !void {
            try std.testing.expect(std.mem.indexOf(u8, text orelse return error.TestExpectedOff, part) != null);
        }
    };
    try Why.says(try draftFourWayOff(a, io, &c, sp, with, false, true), "no drafter");
    try Why.says(try draftFourWayOff(a, io, &c, sp, with, true, false), "TF_DS_DECODE_DEC");
    try Why.says(try draftFourWayOff(a, io, &c, .{ .rank = 1, .world = 2 }, with, true, true), "2D split");
    try Why.says(try draftFourWayOff(a, io, &c, sp, try std.fs.path.join(a, &.{ root, "empty" }), true, true), "no variant");
    try Why.says(try draftFourWayOff(a, io, &c, sp, try std.fs.path.join(a, &.{ root, "none" }), true, true), "no extra set");
    var ce = c;
    ce.format = .exl3;
    try Why.says(try draftFourWayOff(a, io, &ce, sp, with, true, true), "own FP8");
    // Model.open decides before the weights load (which loads the stages as decided), each rank on its own
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const reset = std.mem.indexOf(u8, body, "weights.draft_4way = false;").?;
    const decide = std.mem.indexOf(u8, body, "if (try draftFourWayOff(a, io, c, sp, o.kit_dir, o.drafts, mx.decOnEnv())) |why|").?;
    const load = std.mem.indexOf(u8, body, "m.w = try weights.load(").?;
    try std.testing.expect(reset < decide and decide < load);
}

test "the drafter's main_proj four ways: off by its can bit without a drafter, the checkpoint's own weights, a pair or the decode kernels, decided before the load, agreed after it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    c.format = .native;
    const sp: plan.Split = .{ .rank = 1, .world = 2, .pair = 0, .parity = true };
    try std.testing.expectEqual(@as(?[]const u8, null), draftMainFourWayOff(&c, sp, true, true));
    const Why = struct {
        fn says(text: ?[]const u8, part: []const u8) !void {
            try std.testing.expect(std.mem.indexOf(u8, text orelse return error.TestExpectedOff, part) != null);
        }
    };
    try Why.says(draftMainFourWayOff(&c, sp, false, true), "no drafter");
    try Why.says(draftMainFourWayOff(&c, sp, true, false), "TF_DS_DECODE_DEC");
    try Why.says(draftMainFourWayOff(&c, .{ .rank = 1, .world = 2 }, true, true), "2D split");
    var ce = c;
    ce.format = .exl3;
    try Why.says(draftMainFourWayOff(&ce, sp, true, true), "own FP8");
    var ch = c;
    ch.hidden = c.hidden + 64;
    try Why.says(draftMainFourWayOff(&ch, sp, true, true), "32-row scale blocks");
    // decided before the weights load, each rank on its own, and the ranks compared after it, beside the stages'
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const reset = std.mem.indexOf(u8, body, "weights.draft_main_4way = false;").?;
    const decide = std.mem.indexOf(u8, body, "if (draftMainFourWayOff(c, sp, o.drafts, mx.decOnEnv())) |why|").?;
    const load = std.mem.indexOf(u8, body, "m.w = try weights.load(").?;
    const stages = std.mem.indexOf(u8, body, "try agreeDraftFourWay(d, &m.comm, m.stream, prompt2d.draft_4way_asked, weights.draft_4way, extra_why);").?;
    const agree = std.mem.indexOf(u8, body, "try agreeDraftMainFourWay(d, &m.comm, m.stream, prompt2d.draft_main_4way_asked, weights.draft_main_4way);").?;
    const rings = std.mem.indexOf(u8, body, "ring2d.Rings.open(").?;
    try std.testing.expect(reset < decide and decide < load and load < stages and stages < agree and agree < rings);
    // the verdict: every rank whole: off; every rank four ways: on; a mix: refused
    try std.testing.expectEqual(FourWay.off, fourWayVerdict(4, 0b1111, 0));
    try std.testing.expectEqual(FourWay.on, fourWayVerdict(4, 0, 0));
    try std.testing.expectEqual(FourWay.differ, fourWayVerdict(4, 0b0010, 0));
}

test "the ranks' drafters agree after the load, else the start is refused; the row pair's ring only for TP2's" {
    // every rank TP2's: off; every rank four ways and able to launch them: on; a mix, or a rank that cannot: refused
    try std.testing.expectEqual(FourWay.off, fourWayVerdict(4, 0b1111, 0));
    try std.testing.expectEqual(FourWay.off, fourWayVerdict(4, 0b1111, 0b0100));
    try std.testing.expectEqual(FourWay.on, fourWayVerdict(4, 0, 0));
    try std.testing.expectEqual(FourWay.differ, fourWayVerdict(4, 0b0100, 0));
    try std.testing.expectEqual(FourWay.differ, fourWayVerdict(4, 0b0111, 0b1000));
    try std.testing.expectEqual(FourWay.cannot, fourWayVerdict(4, 0, 0b0010));
    try std.testing.expectEqual(FourWay.off, fourWayVerdict(2, 0b11, 0));
    try std.testing.expectEqual(FourWay.differ, fourWayVerdict(2, 0b10, 0));
    // the agreement comes after the extra set's load and before the 2D state and its rings; the rings open the row
    // pair's ring only for the TP2 drafter; the engine's Triton launches carry the extra set
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const extra = std.mem.indexOf(u8, body, "try std.fs.path.join(a, &.{ o.kit_dir, \"aot-extra\" })").?;
    const agree = std.mem.indexOf(u8, body, "try agreeDraftFourWay(d, &m.comm, m.stream, prompt2d.draft_4way_asked, weights.draft_4way, extra_why);").?;
    const rings = std.mem.indexOf(u8, body, "ring2d.Rings.open(").?;
    try std.testing.expect(extra < agree and agree < rings);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "&m.comm, m.stream, !weights.draft_4way);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, ".t = .{ .set = &m.set, .extra = if (m.extra) |*x| x else null, .stream = m.stream }"));
}

test "Model.open reads TF_DS_PF_RUN128 into the switch every rank compares, and puts it into effect once the kernels load" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const read = std.mem.indexOf(u8, body, "prompt2d.pf_run128_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_PF_RUN128\")) |v| std.mem.span(v) else null);").?;
    const load = std.mem.indexOf(u8, body, "m.mxk = if (c.format == .native) try mx.Kernels.load(d, cuda.kernels.dsv41_native) else null;").?;
    const use = std.mem.indexOf(u8, body, "useRun128(if (m.mxk) |*x| x else null, prompt2d.pf_run128_asked);").?;
    try std.testing.expect(read < load and load < use);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "useRun128("));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "\"TF_DS_PF_RUN128\""));
}

test "Model.open reads TF_DS_GEMM_HALVES into the switch every rank compares, and puts it into effect once the kernels load" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const read = std.mem.indexOf(u8, body, "prompt2d.gemm_halves_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_GEMM_HALVES\")) |v| std.mem.span(v) else null);").?;
    const load = std.mem.indexOf(u8, body, "m.mxk = if (c.format == .native) try mx.Kernels.load(d, cuda.kernels.dsv41_native) else null;").?;
    const use = std.mem.indexOf(u8, body, "mx.gemm_halves = prompt2d.gemm_halves_asked and m.mxk != null;").?;
    try std.testing.expect(read < load and load < use);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "mx.gemm_halves = prompt2d.gemm_halves_asked"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "std.c.getenv(\"TF_DS_GEMM_HALVES\")"));
    // the start line says so when it is on, and why it is off without the checkpoint's own weights
    try std.testing.expect(std.mem.indexOf(u8, body, "TF_DS_GEMM_HALVES=1: a prompt pass of more than") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "TF_DS_GEMM_HALVES=1: off: needs the checkpoint's own FP8 / MXFP4 weights") != null);
}

test "Model.open reads TF_DS_PF_RASTER before the kernels load, refuses text that is no setting, and puts the group into effect with them" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const read = std.mem.indexOf(u8, body, "prompt2d.pf_raster_asked = try readPfRaster();").?;
    const load = std.mem.indexOf(u8, body, "m.mxk = if (c.format == .native) try mx.Kernels.load(d, cuda.kernels.dsv41_native) else null;").?;
    const use = std.mem.indexOf(u8, body, "mx.pf_raster = if (m.mxk != null) prompt2d.pf_raster_asked else 0;").?;
    try std.testing.expect(read < load and load < use);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "std.c.getenv(\"TF_DS_PF_RASTER\")"));
    try std.testing.expect(std.mem.indexOf(u8, src, "TF_DS_PF_RASTER={s}: not a setting (2, 4, 8 or 16 row tiles, or unset)") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "TF_DS_PF_RASTER={d}: a prompt GEMM whose activation slab is") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "TF_DS_PF_RASTER={d}: off: needs the checkpoint's own FP8 / MXFP4 weights") != null);
    // the start-up settings carry the asked group, the SSD tier's math the one in effect
    try std.testing.expect(std.mem.indexOf(u8, src, ".pf_raster = prompt2d.pf_raster_asked,") != null);
}

test "TF_DS_IDX_TPP: unset or 4 takes _index_score_rows_t when the kit holds it; 0, a kit that did not load and one without the twin leave the baseline kernel" {
    const keep = tri_index.idx_tpp;
    defer tri_index.idx_tpp = keep;
    const Case = struct { text: ?[]const u8, loaded: bool, variants: usize, want: usize };
    for ([_]Case{
        .{ .text = null, .loaded = true, .variants = 102, .want = 4 },
        .{ .text = "", .loaded = true, .variants = 102, .want = 4 },
        .{ .text = "0", .loaded = true, .variants = 102, .want = 0 },
        .{ .text = "4", .loaded = true, .variants = 102, .want = 4 },
        .{ .text = null, .loaded = false, .variants = 0, .want = 0 },
        .{ .text = null, .loaded = true, .variants = 0, .want = 0 },
        .{ .text = "4", .loaded = false, .variants = 0, .want = 0 },
        .{ .text = "4", .loaded = true, .variants = 0, .want = 0 },
    }) |c| {
        tri_index.idx_tpp = 7;
        useIdxTpp(c.text, c.loaded, "no such folder", c.variants);
        try std.testing.expectEqual(c.want, tri_index.idx_tpp);
    }
    try std.testing.expectEqual(@as(usize, 4), idxTpp(null));
    try std.testing.expectEqual(@as(usize, 4), idxTpp("4"));
    try std.testing.expectEqual(@as(usize, 0), idxTpp("0"));
}

test "TF_DS_IDX_TPP: unset, empty and 4 are 4, 0 is the baseline kernel, any other text is null (the start is refused at its call site, the parse is pure)" {
    try std.testing.expectEqual(@as(?usize, 4), idxTppWanted(null));
    try std.testing.expectEqual(@as(?usize, 4), idxTppWanted(""));
    try std.testing.expectEqual(@as(?usize, 4), idxTppWanted("4"));
    try std.testing.expectEqual(@as(?usize, 0), idxTppWanted("0"));
    for ([_][]const u8{ "2", "8", "x", " 4", "04", "-4", "four", "1" }) |bad| try std.testing.expectEqual(@as(?usize, null), idxTppWanted(bad));
    // Model.open refuses by name where it reads the setting; every needle split: a literal of this file counts as the file's own text
    const src = @embedFile("model.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "if (idxTpp" ++ "Wanted(prompt2d.idx_tpp_text) == null) {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_IDX_TPP={s}: not 0 or " ++ "4; not started"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_IDX_TPP=0: a prompt chunk's index score runs the baseline " ++ "kernel (the operator's choice)"));
}

test "TF_DS_ATTN_EPI_TWIN picks the fused attention epilogue's kernel on its own: unset is we, ue keeps _ue, xe takes _xe; a kit without the kernel keeps _ue; a bad value is refused; TF_DS_ATTN_TWIN does not move it" {
    // prompt arms measured _we x1.020 / _xe x1.012 on T6 with byte-equal replies, but through TF_DS_ATTN_TWIN, which also
    // picks the decode twin: the epilogue gets its own switch so decode keeps z and the prompt takes _we by default (bit-identical rule)
    try std.testing.expectEqual(tri_attn.EpiTwin.we, tri_attn.epiTwinWanted(null).?);
    try std.testing.expectEqual(tri_attn.EpiTwin.we, tri_attn.epiTwinWanted("").?);
    try std.testing.expectEqual(tri_attn.EpiTwin.ue, tri_attn.epiTwinWanted("ue").?);
    try std.testing.expectEqual(tri_attn.EpiTwin.xe, tri_attn.epiTwinWanted("xe").?);
    try std.testing.expect(tri_attn.epiTwinWanted("w") == null);
    const keep_we = tri_attn.attn_epi_we;
    const keep_xe = tri_attn.attn_epi_xe;
    defer {
        tri_attn.attn_epi_we = keep_we;
        tri_attn.attn_epi_xe = keep_xe;
    }
    const Case = struct { epi: tri_attn.EpiTwin, fused: bool, we: usize, xe: usize, want_we: bool, want_xe: bool };
    for ([_]Case{
        .{ .epi = .we, .fused = true, .we = 16, .xe = 16, .want_we = true, .want_xe = false }, // (the default)
        .{ .epi = .ue, .fused = true, .we = 16, .xe = 16, .want_we = false, .want_xe = false },
        .{ .epi = .xe, .fused = true, .we = 16, .xe = 16, .want_we = false, .want_xe = true },
        .{ .epi = .we, .fused = true, .we = 0, .xe = 16, .want_we = false, .want_xe = false }, // a kit without _we
        .{ .epi = .xe, .fused = true, .we = 16, .xe = 0, .want_we = false, .want_xe = false }, // a kit without _xe
        .{ .epi = .we, .fused = false, .we = 16, .xe = 16, .want_we = false, .want_xe = false }, // no fused epilogue to take it
    }) |c| {
        tri_attn.attn_epi_we = true;
        tri_attn.attn_epi_xe = true;
        useAttnEpiWe(c.epi, c.fused, c.we, c.xe);
        try std.testing.expectEqual(c.want_we, tri_attn.attn_epi_we);
        try std.testing.expectEqual(c.want_xe, tri_attn.attn_epi_xe);
    }
    // Model.open reads the epilogue's own switch and does not hand it the decode twin
    const src = @embedFile("model.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "std.c.getenv(\"TF_DS_ATTN_EPI_TWIN\")") != null);
}

test "TF_DS_ATTN_U takes effect with the kit's extra set loaded, else stays off; unset, never" {
    const keep = tri_attn.attn_u;
    defer tri_attn.attn_u = keep;
    const a = std.testing.allocator;
    const io = std.testing.io;
    tri_attn.attn_u = true;
    useAttnU(a, io, false, true, null, "/nonexistent/aot", "/nonexistent/aot-extra");
    try std.testing.expect(!tri_attn.attn_u); // not asked
    useAttnU(a, io, true, false, "FileNotFound", "/nonexistent/aot", "/nonexistent/aot-extra");
    try std.testing.expect(!tri_attn.attn_u); // no extra set
    useAttnU(a, io, true, true, null, "/nonexistent/aot", "/nonexistent/aot-extra");
    try std.testing.expect(tri_attn.attn_u); // (its coverage unread: the line says so)
}

test "TF_DS_HC_FUSED takes effect with the kit's kernel loaded; asked of a kit without it, the start ends by name; unset, never" {
    const keep = tri_hc.hc_fused;
    defer tri_hc.hc_fused = keep;
    tri_hc.hc_fused = true;
    try useHcFused(false, true, null, 2);
    try std.testing.expect(!tri_hc.hc_fused); // not asked
    var buf: [192]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_HC_FUSED=1: the kit's extra set (aot-extra/) did not load (FileNotFound); not started", hcFusedRefusal(&buf, false, "FileNotFound", 0).?); // no extra set
    try std.testing.expectEqualStrings("TF_DS_HC_FUSED=1: the kit's extra set (aot-extra/aot.json) holds no _hc_pre_pf_fused; not started", hcFusedRefusal(&buf, true, null, 0).?); // an extra set without the kernel
    try std.testing.expect(hcFusedRefusal(&buf, true, null, 2) == null);
    try useHcFused(true, true, null, 2);
    try std.testing.expect(tri_hc.hc_fused);
    var m = try cuda.aot.Manifest.parse(std.testing.allocator, @embedFile("fixtures/kit-drafter-extra.json"));
    defer m.deinit();
    try std.testing.expectEqual(@as(usize, 0), m.variants("_hc_pre_pf_fused"));
    // Model.open: the switch read, the extra set loaded when it is asked for, then the switch put into effect
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const read = std.mem.indexOf(u8, body, "prompt2d.hc_fused_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_HC_FUSED\")) |v| std.mem.span(v) else null);").?;
    const load = std.mem.indexOf(u8, body, "        if (weights.draft_4way or ").?;
    const cond = body[load .. load + std.mem.indexOfScalar(u8, body[load..], '{').?];
    try std.testing.expect(std.mem.indexOf(u8, cond, "prompt2d.hc_fused_asked") != null);
    const use = std.mem.indexOf(u8, body, "try useHcFused(prompt2d.hc_fused_asked, m.extra != null, load_why,").?;
    try std.testing.expect(read < load and load < use);
}

test "TF_DS_IDX_U takes effect with the kit's extra set loaded, else stays off; unset, never" {
    const keep = tri_index.idx_u;
    defer tri_index.idx_u = keep;
    const a = std.testing.allocator;
    const io = std.testing.io;
    tri_index.idx_u = true;
    useIdxU(a, io, false, true, null, "/nonexistent/aot", "/nonexistent/aot-extra");
    try std.testing.expect(!tri_index.idx_u); // not asked
    useIdxU(a, io, true, false, "FileNotFound", "/nonexistent/aot", "/nonexistent/aot-extra");
    try std.testing.expect(!tri_index.idx_u); // no extra set
    useIdxU(a, io, true, true, null, "/nonexistent/aot", "/nonexistent/aot-extra");
    try std.testing.expect(tri_index.idx_u); // (its coverage unread: the line says so)
    // Model.open: the switch read, the extra set loaded when it is asked for, then the switch put into effect
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const read = std.mem.indexOf(u8, body, "prompt2d.idx_u_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_IDX_U\")) |v| std.mem.span(v) else null);").?;
    const load = std.mem.indexOf(u8, body, "        if (weights.draft_4way or ").?;
    const cond = body[load .. load + std.mem.indexOfScalar(u8, body[load..], '{').?];
    try std.testing.expect(std.mem.indexOf(u8, cond, "prompt2d.idx_u_asked") != null);
    const use = std.mem.indexOf(u8, body, "useIdxU(a, io, prompt2d.idx_u_asked, m.extra != null, load_why,").?;
    try std.testing.expect(read < load and load < use);
}

test "TF_DS_IDX_V takes effect with the kit's twin loaded; asked of a kit without it, the start ends by name; unset, never" {
    const keep = tri_index.idx_v;
    defer tri_index.idx_v = keep;
    tri_index.idx_v = true;
    try useIdxV(false, true, null, 102);
    try std.testing.expect(!tri_index.idx_v); // not asked
    var buf: [192]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_IDX_V=1: the kit's extra set (aot-extra/) did not load (FileNotFound); not started", idxVRefusal(&buf, false, "FileNotFound", 0).?); // no extra set
    try std.testing.expectEqualStrings("TF_DS_IDX_V=1: the kit's extra set (aot-extra/aot.json) holds no _index_score_rows_v; not started", idxVRefusal(&buf, true, null, 0).?); // an extra set without the twin
    try std.testing.expect(idxVRefusal(&buf, true, null, 102) == null);
    try useIdxV(true, true, null, 102);
    try std.testing.expect(tri_index.idx_v);
    // the count: an unreadable set holds none, a manifest's twin variants are counted by name
    try std.testing.expectEqual(@as(usize, 0), extraVariants(std.testing.allocator, std.testing.io, "/nonexistent/aot-extra", "_index_score_rows_v"));
    var m = try cuda.aot.Manifest.parse(std.testing.allocator, @embedFile("fixtures/kit-drafter-extra.json"));
    defer m.deinit();
    try std.testing.expectEqual(@as(usize, 0), m.variants("_index_score_rows_v"));
    try std.testing.expect(m.variants("_index_score_rows_u") > 0);
    // Model.open: the switch read, the extra set loaded when it is asked for, then the switch put into effect
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const read = std.mem.indexOf(u8, body, "prompt2d.idx_v_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_IDX_V\")) |v| std.mem.span(v) else null);").?;
    const load = std.mem.indexOf(u8, body, "        if (weights.draft_4way or ").?;
    const cond = body[load .. load + std.mem.indexOfScalar(u8, body[load..], '{').?];
    try std.testing.expect(std.mem.indexOf(u8, cond, "prompt2d.idx_v_asked") != null);
    const use = std.mem.indexOf(u8, body, "try useIdxV(prompt2d.idx_v_asked, m.extra != null, load_why,").?;
    try std.testing.expect(read < load and load < use);
}

test "TF_DS_ATTN_U_DECODE takes effect with the kit's extra set loaded, else stays off; unset, never" {
    const keep = tri_attn.attn_u_decode;
    defer tri_attn.attn_u_decode = keep;
    const a = std.testing.allocator;
    const io = std.testing.io;
    tri_attn.attn_u_decode = true;
    useAttnUDecode(a, io, false, true, null, "/nonexistent/aot", "/nonexistent/aot-extra");
    try std.testing.expect(!tri_attn.attn_u_decode); // not asked
    useAttnUDecode(a, io, true, false, "FileNotFound", "/nonexistent/aot", "/nonexistent/aot-extra");
    try std.testing.expect(!tri_attn.attn_u_decode); // no extra set
    useAttnUDecode(a, io, true, true, null, "/nonexistent/aot", "/nonexistent/aot-extra");
    try std.testing.expect(tri_attn.attn_u_decode); // (its coverage unread: the line says so)
    // Model.open: the switch read, the extra set loaded when it is asked for, then the switch put into effect
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const read = std.mem.indexOf(u8, body, "prompt2d.attn_u_decode_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_ATTN_U_DECODE\")) |v| std.mem.span(v) else null);").?;
    const load = std.mem.indexOf(u8, body, "        if (weights.draft_4way or ").?;
    const cond = body[load .. load + std.mem.indexOfScalar(u8, body[load..], '{').?];
    try std.testing.expect(std.mem.indexOf(u8, cond, "prompt2d.attn_u_decode_asked") != null);
    const use = std.mem.indexOf(u8, body, "useAttnUDecode(a, io, prompt2d.attn_u_decode_asked, m.extra != null, load_why,").?;
    try std.testing.expect(read < load and load < use);
}

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
test "TF_DS_HC_LATE_FORK takes effect with the checkpoint's own weights and the mixes' side stream, else stays off" {
    var rd: round.Round = undefined;
    rd.late_fork = false;
    rd.side = null;
    useLateFork(&rd, true, true);
    try std.testing.expect(!rd.late_fork); // no side stream
    rd.useLateFork();
    try std.testing.expect(!rd.late_fork); // (the round's own guard)
    rd.side = @as(cuda.Stream, undefined);
    useLateFork(&rd, false, true);
    try std.testing.expect(!rd.late_fork); // not asked
    useLateFork(&rd, true, false);
    try std.testing.expect(!rd.late_fork); // the EXL3 checkpoint
    useLateFork(&rd, true, true);
    try std.testing.expect(rd.late_fork);
}

test "TF_DS_ENGRAM_PREFAULT's line: how long, how, and the 2 MiB pages the kernel gave" {
    var buf: [256]u8 = undefined;
    const pop = [_]engram_rows.Fault{ .{ .huge = true, .populated = true }, .{ .huge = true, .populated = true } };
    try std.testing.expectEqualStrings("TF_DS_ENGRAM_PREFAULT=1: the Engram row cache's 2048 MiB faulted in at the start in 412 ms (the kernel's populate; 2044 MiB of it in 2 MiB pages)", prefaultLine(&buf, 2048, &pop, 412_300_000, 2048, 2048 + 2044 * 1024));
    const wrote = [_]engram_rows.Fault{.{ .written = true }};
    try std.testing.expectEqualStrings("TF_DS_ENGRAM_PREFAULT=1: the Engram row cache's 64 MiB faulted in at the start in 9 ms (a write to each page; its 2 MiB pages not read)", prefaultLine(&buf, 64, &wrote, 9_000_000, null, 7));
    const nomem = [_]engram_rows.Fault{ .{ .populated = true }, .{ .why = "OutOfMemory" } };
    try std.testing.expectEqualStrings("TF_DS_ENGRAM_PREFAULT=1: off: the kernel did not fault the row cache in (OutOfMemory); its pages fault in as it fills", prefaultLine(&buf, 2048, &nomem, 1, 0, 0));
    try std.testing.expectEqualStrings("TF_DS_ENGRAM_PREFAULT=1: off: no row cache (TF_DS_ENGRAM_CACHE_MIB=0)", prefaultLine(&buf, 0, &.{}, 0, null, null));
}

test "this process's 2 MiB pages read from /proc/self/smaps_rollup" {
    try std.testing.expect(hugeKiB(std.testing.io) != null);
}

test "TF_DS_MMA_SF's line: on every rank with the GEMMs it covers, or off on every rank, which ranks and why" {
    var buf: [320]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_MMA_SF=1: a prompt chunk's FP8 GEMMs on the block-scaled MMA, each split's K blocks accumulated in the tensor core (not the baseline bits), on every rank: wq_a, wkv and the experts' gate / up and down", mmaSfLine(&buf, 0, null, "wq_a, wkv and the experts' gate / up and down"));
    try std.testing.expectEqualStrings("TF_DS_MMA_SF=1: off on every rank: rank 2 cannot run it (this rank: can)", mmaSfLine(&buf, 1 << 2, null, "wq_a"));
    try std.testing.expectEqualStrings("TF_DS_MMA_SF=1: off on every rank: ranks 0 and 3 cannot run it (this rank: needs the prompt chunks' tile kernels (TF_DS_PF_TILES=0 turns them off))", mmaSfLine(&buf, 0b1001, "needs the prompt chunks' tile kernels (TF_DS_PF_TILES=0 turns them off)", "wq_a"));
}

test "TF_DS_MMA_SF's line names the layers' dense FP8 GEMMs its kernels take, each once in a fixed order, and the experts" {
    var buf: [320]u8 = undefined;
    var lays: [2]weights.Layer = .{ std.mem.zeroes(weights.Layer), std.mem.zeroes(weights.Layer) };
    const fp8 = struct {
        fn of(kdim: u32, n: u32) weights.Linear {
            return .{ .words = 0, .k = kdim, .n = n, .fmt = .fp8 };
        }
    }.of;
    for (&lays) |*l| {
        l.wq_a = fp8(5120, 1280);
        l.wkv = fp8(5120, 512);
        l.wq_b = fp8(1280, 8192);
        l.wo_b = fp8(4096, 2560);
        l.comp_wkv = .{ .words = 0, .k = 5120, .n = 1024, .fmt = .bf16 }; // bf16: not an FP8 GEMM
    }
    lays[1].idx_wq_b = fp8(1280, 4096); // one layer's indexer
    lays[0].engram_wkv = fp8(3072, 12800);
    lays[1].idx_wk = fp8(5120, 1280 - 64); // FP8 but a shape the kernels do not take
    try std.testing.expectEqualStrings("wq_a, wkv, wq_b, wo_b, idx_wq_b, engram_wkv", sfCovers(&lays, false, &buf));
    lays[0].experts.native = std.mem.zeroes(weights.NativeExperts);
    try std.testing.expectEqualStrings("wq_a, wkv, wq_b, wo_b, idx_wq_b, engram_wkv and the experts' gate / up and down", sfCovers(&lays, false, &buf));
}

test "Model.open reads TF_DS_MMA_SF with the kernel switches and puts it in effect after them, through the agreement" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "prompt2d.mma_sf_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_MMA_SF\")) |v| std.mem.span(v) else null); // every rank alike (Settings)"));
    const fuse = std.mem.indexOf(u8, code, "useDecodeFuse(if (m.mxk) |*x| x else null, d, prompt2d.decode_fuse_asked);") orelse return error.TestUnexpectedResult;
    const sf = std.mem.indexOf(u8, code, "try useMmaSf(d, &m.comm, m.stream, if (m.mxk) |*x| x else null, prompt2d.mma_sf_asked, sfCovers(m.w.layers, false, &sf_buf));") orelse return error.TestUnexpectedResult;
    try std.testing.expect(fuse < sf and sf - fuse < 300);
    // the agreement: this rank's check (the weights and tile kernels, then the block-scaled kernels' load), the
    // all-gather, then on or off on every rank
    const at = std.mem.indexOf(u8, code, "fn useMmaSf(") orelse return error.TestUnexpectedResult;
    const body = code[at .. at + (std.mem.indexOf(u8, code[at..], "\n}\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "var why: ?[]const u8 = mx.splitOff(k != null, if (k) |x| x.pf_off else false);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "x.useMmaSf(d, cuda.kernels.dsv41_native_sf, .chained) catch |e| {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const cannot = try prompt2d.ranksThatCannot(d, comm, stream, why == null);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "if (!prompt2d.agreedOn(asked, cannot)) {\n        if (k) |x| x.sf = .off;\n    }"));
}

test "TF_DS_MMA_SF_EXACT's line: on every rank with the GEMMs it covers and the ones its guard kept, or off, which ranks and why" {
    var buf: [400]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_MMA_SF_EXACT=1: a prompt chunk's FP8 GEMMs on the block-scaled MMA, each K block's MMA from zero, then the baseline fp32 add (the baseline bits), on every rank: wq_a, wkv and the experts' gate / up and down; kept on the baseline kernels (a weight scale outside the proven range): none; weight scale bytes seen: FP8 none, MXFP4 none, shared expert none", mmaSfExactLine(&buf, 0, null, "wq_a, wkv and the experts' gate / up and down", "none", 0, "FP8 none, MXFP4 none, shared expert none"));
    try std.testing.expectEqualStrings("TF_DS_MMA_SF_EXACT=1: off on every rank: rank 3 cannot run it (this rank: can)", mmaSfExactLine(&buf, 1 << 3, null, "wq_a", "none", 0, ""));
    try std.testing.expectEqualStrings("TF_DS_MMA_SF_EXACT=1: off on every rank: ranks 0, 1, 2 and 3 cannot run it (this rank: TF_DS_MMA_SF=1 runs the block-scaled kernels in its own arithmetic)", mmaSfExactLine(&buf, 0b1111, "TF_DS_MMA_SF=1 runs the block-scaled kernels in its own arithmetic", "wq_a", "none", 0, ""));
}

test "TF_DS_MMA_SF_EXACT's kept GEMMs in words: each field with its layers' count in sfCovers' order, then the experts; none" {
    var buf: [256]u8 = undefined;
    var dense: [sf_fields.len]usize = @splat(0);
    try std.testing.expectEqualStrings("none", sfKeptText(&buf, dense, 0));
    dense[3] = 3; // wo_b
    dense[0] = 1; // wq_a
    try std.testing.expectEqualStrings("wq_a in 1 layer, wo_b in 3 layers", sfKeptText(&buf, dense, 0));
    try std.testing.expectEqualStrings("wq_a in 1 layer, wo_b in 3 layers, the experts in 2 layers", sfKeptText(&buf, dense, 2));
    try std.testing.expectEqualStrings("the experts in 1 layer", sfKeptText(&buf, @splat(0), 1));
}

test "Model.open reads TF_DS_MMA_SF_EXACT and puts it in effect after TF_DS_MMA_SF, its guard and then the agreement" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "prompt2d.mma_sf_exact_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_MMA_SF_EXACT\")) |v| std.mem.span(v) else null); // every rank alike (Settings)"));
    const sf = std.mem.indexOf(u8, code, "try useMmaSf(d, &m.comm, m.stream,") orelse return error.TestUnexpectedResult;
    const ex = std.mem.indexOf(u8, code, "try useMmaSfExact(gpa, d, &m.comm, m.stream, if (m.mxk) |*x| x else null, prompt2d.mma_sf_exact_asked, m.w.layers, sfCovers(m.w.layers, true, &sf_buf));") orelse return error.TestUnexpectedResult;
    try std.testing.expect(sf < ex and ex - sf < 400);
    const at = std.mem.indexOf(u8, code, "fn useMmaSfExact(") orelse return error.TestUnexpectedResult;
    const body = code[at .. at + (std.mem.indexOf(u8, code[at..], "\n}\n") orelse return error.TestUnexpectedResult)];
    // TF_DS_MMA_SF takes the kernels first; then this rank's load and guard; the all-gather; off everywhere unless agreed
    const chained = std.mem.indexOf(u8, body, "if (why == null and k.?.sf == .chained) why = \"TF_DS_MMA_SF=1 runs the block-scaled kernels in its own arithmetic\";") orelse return error.TestUnexpectedResult;
    const load = std.mem.indexOf(u8, body, "x.useMmaSf(d, cuda.kernels.dsv41_native_sf, .exact) catch |e| {") orelse return error.TestUnexpectedResult;
    const guard = std.mem.indexOf(u8, body, "sfGuard(gpa, d, stream, x, layers, &kb)") orelse return error.TestUnexpectedResult;
    const cannot = std.mem.indexOf(u8, body, "const cannot = try prompt2d.ranksThatCannot(d, comm, stream, why == null);") orelse return error.TestUnexpectedResult;
    const off = std.mem.indexOf(u8, body, "x.sf = mx.sfAfterExact(x.sf, agreed);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(chained < load and load < guard and guard < cannot and cannot < off);
}

test "the exact form's covered GEMMs: the dense FP8 shapes with a K split (Engram's wkv has none), then the experts" {
    var buf: [320]u8 = undefined;
    var lays: [1]weights.Layer = .{std.mem.zeroes(weights.Layer)};
    const fp8 = struct {
        fn of(kdim: u32, n: u32) weights.Linear {
            return .{ .words = 0, .k = kdim, .n = n, .fmt = .fp8 };
        }
    }.of;
    lays[0].wq_a = fp8(5120, 1280);
    lays[0].wo_b = fp8(4096, 2560);
    lays[0].engram_wkv = fp8(3072, 12800);
    lays[0].experts.native = std.mem.zeroes(weights.NativeExperts);
    try std.testing.expectEqualStrings("wq_a, wo_b, engram_wkv and the experts' gate / up and down", sfCovers(&lays, false, &buf));
    try std.testing.expectEqualStrings("wq_a, wo_b and the experts' gate / up and down", sfCovers(&lays, true, &buf));
}

test "TF_DS_MMA_SF_EXACT's line names the weight scale bytes its guard saw, a class without tensors as none" {
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("FP8 41..127, MXFP4 33..137, shared expert 118..126", sfSeenText(&buf, .{ .{ 41, 127 }, .{ 33, 137 }, .{ 118, 126 } }));
    try std.testing.expectEqualStrings("FP8 none, MXFP4 120..131, shared expert none", sfSeenText(&buf, .{ .{ 255, 0 }, .{ 120, 131 }, .{ 255, 0 } }));
    var lb: [768]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_MMA_SF_EXACT=1: a prompt chunk's FP8 GEMMs on the block-scaled MMA, each K block's MMA from zero, then the baseline fp32 add (the baseline bits), on every rank: wq_a and the experts' gate / up and down; kept on the baseline kernels (a weight scale outside the proven range): none; weight scale bytes seen: FP8 118..127, MXFP4 120..131, shared expert 119..126", mmaSfExactLine(&lb, 0, null, "wq_a and the experts' gate / up and down", "none", 0, "FP8 118..127, MXFP4 120..131, shared expert 119..126"));
}

test "useMmaSfExact refused turns only its own arithmetic off (TF_DS_MMA_SF's stays) and clears its guard's sets" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "fn useMmaSfExact(") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n}\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const agreed = prompt2d.agreedOn(asked, cannot);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "x.sf = mx.sfAfterExact(x.sf, agreed);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "if (!agreed) {\n            x.sf_nkept = 0;\n            x.sf_nproven = 0;\n            x.sf_apart = 0;\n        }"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, body, "x.sf = .off;"));
    // the guard covers the exact form's shapes; its line names what it saw
    const g = std.mem.indexOf(u8, src, "pub fn sfGuard(") orelse return error.TestUnexpectedResult;
    const gb = src[g .. g + (std.mem.indexOf(u8, src[g..], "\n}\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, gb, "if (x.fmt == .fp8 and mx.sfExactShape(x.k, x.n)) {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, gb, "k.sf_nkept = 0;\n    k.sf_nproven = 0;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, gb, "if (mx.sfProvenRead(rg.len, r, t.fp4) and (t.field != null or layout_ok[t.layer])) {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, gb, "layout_ok[li] = nx.scales_match;"));
    // a layer packed at other lengths: none of its bytes read, kept apart from the scale-kept ones
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, gb, "if (!nx.scales_match) for (out) |*rg| {\n                rg.len = 0;\n            };"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, gb, "if (!nx.scales_match) {\n            try mx.sfKeep(k, nx.s1);\n            apart += 1;\n        } else if (b) {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, gb, "k.sf_apart = apart;"));
}

test "TF_DS_MMA_SF_EXACT's line: experts kept for their packing, apart from a scale outside the range; a guard's error by its name" {
    var buf: [768]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_MMA_SF_EXACT=1: a prompt chunk's FP8 GEMMs on the block-scaled MMA, each K block's MMA from zero, then the baseline fp32 add (the baseline bits), on every rank: wq_a and the experts' gate / up and down; kept on the baseline kernels (a weight scale outside the proven range): none; kept on the baseline kernels (scale tensors packed at other lengths): the experts in 40 layers; weight scale bytes seen: FP8 none, MXFP4 120..131, shared expert none", mmaSfExactLine(&buf, 0, null, "wq_a and the experts' gate / up and down", "none", 40, "FP8 none, MXFP4 120..131, shared expert none"));
    try std.testing.expect(std.mem.indexOf(u8, mmaSfExactLine(&buf, 0, null, "wq_a", "wq_a in 1 layer", 1, "x"), "; kept on the baseline kernels (scale tensors packed at other lengths): the experts in 1 layer; ") != null);
    var wb: [160]u8 = undefined;
    try std.testing.expectEqualStrings("its guard over the weight scales did not run (OutOfMemory)", guardWhy(&wb, error.OutOfMemory));
    try std.testing.expectEqualStrings("its guard over the weight scales did not run (CudaFailed)", guardWhy(&wb, error.CudaFailed));
    try std.testing.expectEqualStrings("more GEMMs than its guard keeps have a weight scale outside the proven range", guardWhy(&wb, error.TooManyKept));
    try std.testing.expectEqualStrings("more GEMMs than its guard can prove", guardWhy(&wb, error.TooManyProven));
}

test "TF_DS_RDMA_COLLECT_PEERS: read as a switch (the text 1); its line on with the kernel, off without it" {
    try std.testing.expect(prompt2d.switchAsked("1") and !prompt2d.switchAsked("0") and !prompt2d.switchAsked(null));
    const src = @embedFile("model.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "return prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_RDMA_COLLECT_PEERS\")) |v| std.mem.span(v) else null);"));
    try std.testing.expectEqualStrings("TF_DS_RDMA_COLLECT_PEERS=1: the rings' gather_into copy-out reads every peer's slot at once (a row of blocks a peer, each waiting for its own peer's flag): the same bytes", Model.collectPeersLine(true));
    try std.testing.expectEqualStrings("TF_DS_RDMA_COLLECT_PEERS=1: off: the rings' kernels (TF_DS_RDMA_KERNELS, else the build's own) have no dsv41_rdma_collect_peers", Model.collectPeersLine(false));
}

test "Model.open puts TF_DS_RDMA_COLLECT_PEERS in the four-node rings' settings when the rings' kernels have it, with its line" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    const k = std.mem.indexOf(u8, code, "const rk = try rdma.Kernels.load(m.rdma_mod.?);") orelse return error.TestUnexpectedResult;
    const line = std.mem.indexOf(u8, code, "if (peers) std.log.info(\"{s}\", .{collectPeersLine(rk.collect_peers != null)});") orelse return error.TestUnexpectedResult;
    const open = std.mem.indexOf(u8, code, "m.rdma_rings = try ring2d.Rings.open(gpa, d, rk, devices.items, role, ord, .{ .max_bytes = ring_bytes, .gid_index = rdmaGid(), .pdl = rdmaPdl(), .collect_peers = peers and rk.collect_peers != null, .wc = sw.wc, .inplace = sw.inplace, .roci = ringWordOf(), .roci_deadline_ns = ringWordDeadlineNs(), .wait = ring_wait_on, .wait_explicit = prompt2d.ring_wait_explicit }, &m.comm, m.stream, !weights.draft_4way);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(k < line and line < open);
}

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

test "verify: the forward, the drafter's absorb, then the stream's sync and the round's check (as before)" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn verify(m: *Model,") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    var last: usize = 0;
    for ([_][]const u8{ "try m.forwardRows(rows);", "try m.absorbRows(rows, absorb);", "try m.stream.synchronize();", "if (m.erows) |r| try r.roundCheck();" }) |step| {
        const i = std.mem.indexOf(u8, body, step) orelse return error.TestUnexpectedResult;
        try std.testing.expect(i >= last);
        last = i;
    }
}

test "verifyHead (TF_DS_CAND_FIRST): the forward, the head's reads, their event, the absorb, then the event's wait and the round's check; none or a profiled round: verify, then the reads as before" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn verifyHead(") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    var last: usize = 0;
        const i = std.mem.indexOf(u8, body[last..], step) orelse return error.TestUnexpectedResult;
        last += i + step.len;
    }
    // roundArgmax / roundCandidates: the baseline launch, the stream's sync, then the same reads; the reads launch nothing
    for ([_][2][]const u8{ .{ "    pub fn roundArgmax(", "try m.launchHead(.argmax, R);" }, .{ "    pub fn roundCandidates(", "try m.launchHead(.{ .candidates = count }, R);" } }, [_][]const u8{ "return m.headArgmax(R);", "return m.headCandidates(R, count);" }) |f, read| {
        const fa = std.mem.indexOf(u8, src, f[0]) orelse return error.TestUnexpectedResult;
        const fb = src[fa .. fa + (std.mem.indexOf(u8, src[fa..], "\n    }\n") orelse return error.TestUnexpectedResult)];
        const l = std.mem.indexOf(u8, fb, f[1]) orelse return error.TestUnexpectedResult;
        const y = std.mem.indexOf(u8, fb, "try m.stream.synchronize();") orelse return error.TestUnexpectedResult;
        const r = std.mem.indexOf(u8, fb, read) orelse return error.TestUnexpectedResult;
        try std.testing.expect(l < y and y < r);
    }
    for ([_][]const u8{ "    pub fn headArgmax(", "    pub fn headCandidates(" }) |f| {
        const fa = std.mem.indexOf(u8, src, f) orelse return error.TestUnexpectedResult;
        const fb = src[fa .. fa + (std.mem.indexOf(u8, src[fa..], "\n    }\n") orelse return error.TestUnexpectedResult)];
        try std.testing.expect(std.mem.indexOf(u8, fb, "launch") == null and std.mem.indexOf(u8, fb, "synchronize") == null);
    }
}

test "the merged head reads wait for the head event alone: headCandidatesMerged and headTokens never synchronize the stream (the drafter's absorb runs behind it)" {
    const src = @embedFile("model.zig");
    for ([_][]const u8{ "    pub fn headCandidatesMerged(", "    pub fn headTokens(" }) |f| {
        const fa = std.mem.indexOf(u8, src, f) orelse return error.TestUnexpectedResult;
        const fb = src[fa .. fa + (std.mem.indexOf(u8, src[fa..], "\n    }\n") orelse return error.TestUnexpectedResult)];
        try std.testing.expect(std.mem.indexOf(u8, fb, "cuMemcpyDtoH_v2(m.cand_parts.ptr, m.rd.call") != null);
        try std.testing.expect(std.mem.indexOf(u8, fb, "synchronize") == null);
    }
}

test "TF_DS_CAND_FIRST is read as a switch at Model.open, with its line and its event" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "m.cand_first = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_CAND_FIRST\")) |v| std.mem.span(v) else null);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "if (m.cand_first) {\n            m.head_ready = try cuda.Event.init(d, false);\n            std.log.info(\"{s}\", .{candFirstLine()});\n        }"));
    try std.testing.expectEqualStrings("TF_DS_CAND_FIRST=1: a round's argmax or first candidates run before the drafter's absorb; the host waits for them alone and draws while the absorb runs", Model.candFirstLine());
}

test "TF_DS_HEAD_CAND's line: on every rank, or off on every rank, which ranks and why" {
    var buf: [400]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_HEAD_CAND=1: a sampled round's candidates from each node's own columns of the head (its first 32 or 64 of each row) merged on rank 0 into the baseline's; a round the baseline kernel could flag reads the whole head as the baseline; on every rank", headCandLine(&buf, 0, null));
    try std.testing.expectEqualStrings("TF_DS_HEAD_CAND=1: off on every rank: ranks 0 and 1 cannot run it (this rank: needs the four-node 2D split)", headCandLine(&buf, 0b11, "needs the four-node 2D split"));
    try std.testing.expectEqualStrings("TF_DS_HEAD_CAND=1: off on every rank: rank 3 cannot run it (this rank: can)", headCandLine(&buf, 1 << 3, null));
}

test "Model.open reads TF_DS_HEAD_CAND with the kernel switches and puts it in effect after the round and the 2D node, through the agreement" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    const read = std.mem.indexOf(u8, body, "prompt2d.head_cand_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_HEAD_CAND\")) |v| std.mem.span(v) else null); // every rank alike (Settings)") orelse return error.TestUnexpectedResult;
    const settings = std.mem.indexOf(u8, body, "const mine = startSettings(o);") orelse return error.TestUnexpectedResult;
    const two = std.mem.indexOf(u8, body, "m.eng = .{ .d = d,") orelse return error.TestUnexpectedResult;
    const rd = std.mem.indexOf(u8, body, "m.rd = try round.Round.init(&m.eng, &m.arena, a, m.ctx_cap);") orelse return error.TestUnexpectedResult;
    const use = std.mem.indexOf(u8, body, "try useHeadCand(m, a, d, prompt2d.head_cand_asked);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(read < settings and settings < two and two < rd and rd < use);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "useHeadCand("));
    // the agreement before anything is made; the round's parts only when every rank can
    const f = std.mem.indexOf(u8, src, "fn useHeadCand(") orelse return error.TestUnexpectedResult;
    const fb = src[f .. f + (std.mem.indexOf(u8, src[f..], "\n}\n") orelse return error.TestUnexpectedResult)];
    const agree = std.mem.indexOf(u8, fb, "const cannot = try prompt2d.ranksThatCannot(d, &m.comm, m.stream, why == null);") orelse return error.TestUnexpectedResult;
    const on = std.mem.indexOf(u8, fb, "if (prompt2d.agreedOn(asked, cannot)) {") orelse return error.TestUnexpectedResult;
    const made = std.mem.indexOf(u8, fb, "try headParts(m, a);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(agree < on and on < made);
    // a round's head as rank 0's frame says, refused when the switch is not in effect here; the merge's parts in the
    // vocabulary's order
    const v = std.mem.indexOf(u8, src, "    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool, head: u8) !void {") orelse return error.TestUnexpectedResult;
    const vb = src[v .. v + (std.mem.indexOf(u8, src[v..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    const refuse = std.mem.indexOf(u8, vb, "if (head != 0 and !(if (head == head_cand.key) m.head_key else m.head_cand)) return error.HeadCandOff;") orelse return error.TestUnexpectedResult;
    const set = std.mem.indexOf(u8, vb, "m.rd.head_k = head;") orelse return error.TestUnexpectedResult;
    const fwd = std.mem.indexOf(u8, vb, "try m.forwardRows(rows);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(refuse < set and set < fwd);
    const fr = std.mem.indexOf(u8, src, "    fn forwardRows(") orelse return error.TestUnexpectedResult;
    const frb = src[fr .. fr + (std.mem.indexOf(u8, src[fr..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expect(std.mem.indexOf(u8, frb, "try round.forward(e, &m.rd,") != null);
    // a round's bucket stops at the context Round.init sized its buffers for (m.ctx_cap), never at the pool: a
    // 4,128,768-token context sized them for 4,128,768 keys and a 3.99M-deep round asked for the 4,194,304 bucket
    try std.testing.expect(std.mem.indexOf(u8, frb, "rows, m.ctx_cap, null,") != null);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, src, "head_cand.merge(m.cand_parts[0 .. 4 * lay.bytes], lay, head_cand.vocabOffsets(tw.hw), count, m.cand_vals[0 .. R * count], m.cand_ids[0 .. R * count])")); // headCandidates', and this test's text
}

test "TF_DS_HEAD_KEY's line, its read with the kernel switches, its agreement before the shared parts" {
    var buf: [400]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_HEAD_KEY=1: a greedy round's tokens from each node's best of each row of its own columns of the head, merged on rank 0 into the baseline argmax; on every rank", headKeyLine(&buf, 0, null));
    try std.testing.expectEqualStrings("TF_DS_HEAD_KEY=1: off on every rank: rank 2 cannot run it (this rank: needs the four-node 2D split)", headKeyLine(&buf, 1 << 2, "needs the four-node 2D split"));
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    const read = std.mem.indexOf(u8, body, "prompt2d.head_key_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_HEAD_KEY\")) |v| std.mem.span(v) else null); // every rank alike (Settings)") orelse return error.TestUnexpectedResult;
    const settings = std.mem.indexOf(u8, body, "const mine = startSettings(o);") orelse return error.TestUnexpectedResult;
    const cand = std.mem.indexOf(u8, body, "try useHeadCand(m, a, d, prompt2d.head_cand_asked);") orelse return error.TestUnexpectedResult;
    const use = std.mem.indexOf(u8, body, "try useHeadKey(m, a, d, prompt2d.head_key_asked);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(read < settings and settings < cand and cand < use);
    const f = std.mem.indexOf(u8, src, "fn useHeadKey(") orelse return error.TestUnexpectedResult;
    const fb = src[f .. f + (std.mem.indexOf(u8, src[f..], "\n}\n") orelse return error.TestUnexpectedResult)];
    const agree = std.mem.indexOf(u8, fb, "const cannot = try prompt2d.ranksThatCannot(d, &m.comm, m.stream, why == null);") orelse return error.TestUnexpectedResult;
    const on = std.mem.indexOf(u8, fb, "if (prompt2d.agreedOn(asked, cannot)) {") orelse return error.TestUnexpectedResult;
    const made = std.mem.indexOf(u8, fb, "try headParts(m, a);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(agree < on and on < made);
    // the parts made once for both switches; the tokens merged with the parts' offsets
    const hp = std.mem.indexOf(u8, src, "fn headParts(") orelse return error.TestUnexpectedResult;
    const hb = src[hp .. hp + (std.mem.indexOf(u8, src[hp..], "\n}\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expect(std.mem.indexOf(u8, hb, "if (m.cand_parts.len != 0) return;") != null);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, src, "head_cand.mergeKeys(m.cand_parts[0..n], R, head_cand.vocabOffsets(tw.hw), m.amax_host[0..R]);")); // headTokens', and this test's text
}
test "TF_DS_ENGRAM_PINNED: the worker's slots go back through engramRelease after the uploads, the wait for the worker polls the events, a new worker flushes them first" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expect(std.mem.indexOf(u8, code, "eh.wk_used[j].store(") == null); // (only prompt.engramRelease stores it)
    const up = std.mem.indexOf(u8, code, "} else try prompt.engramApply(e, ch, eh, li, seq);\n            // the rows are uploaded") orelse return error.TestUnexpectedResult;
    const rel = std.mem.indexOf(u8, code[up..], "try prompt.engramRelease(eh, e.s, j, first + units);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(rel > 0);
    const p = @embedFile("prompt.zig");
    const wait = std.mem.indexOf(u8, p, "const EngramDuties = struct {") orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, p[wait .. wait + 400], "try engramPoll(d.eh);") != null);
    const start = std.mem.indexOf(u8, p, "pub fn engramWorkerStart(") orelse return error.TestUnexpectedResult;
    const flush = std.mem.indexOf(u8, p[start..], "try engramFlushPins(eh);") orelse return error.TestUnexpectedResult;
    const spawn = std.mem.indexOf(u8, p[start..], "eh.wk_thread = try std.Thread.spawn(") orelse return error.TestUnexpectedResult;
    try std.testing.expect(flush < spawn);
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

test "TF_DS_ENGRAM_PINNED: the check that the running worker is still on this fill counts the chunks uploaded, given back or pending, in a pass and in a pair" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, code, "chunk_rows <= prompt.engramUploaded(eh, 0);"));
    try std.testing.expect(std.mem.indexOf(u8, code, "eh.wk_used[0]") == null);
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

const roci_step_print = 12;

test "TF_DS_ROCI_PAIR_LOG: every line of a pair is short enough for the log (512 bytes) with 12 characters for each number, and the file writer's buffer holds more" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    var at: usize = 0;
    var lines: usize = 0;
    while (std.mem.indexOfPos(u8, code, at, "pairLog(\"")) |i| {
        const start = i + "pairLog(\"".len;
        const end = std.mem.indexOfPos(u8, code, start, "\", .{") orelse break;
        const fmt = code[start..end];
        const holes = std.mem.count(u8, fmt, "{");
        try std.testing.expect(fmt.len - 5 * holes + 12 * holes < 480);
        lines += 1;
        at = end;
    }
    try std.testing.expect(lines >= 6); // the first-launch line, the span line, the follower's line and the three of rank 0
    try std.testing.expect(std.mem.indexOf(u8, code, "var buf: [2048]u8 = undefined;\n    const line = std.fmt.bufPrint(&buf, fmt ++") != null);
}

test "TF_DS_ENGRAM_AIO_SLEEP: the prompt ring sleeps by default (two empty polls, 32 completions, 1 ms) unless the variable is 0" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expect(std.mem.indexOf(u8, code, "if (std.c.getenv(\"TF_DS_ENGRAM_AIO_SLEEP\")) |v| !std.mem.eql(u8, std.mem.span(v), \"0\") else true") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "catch 2 else 2;") != null); // SPIN
    try std.testing.expect(std.mem.indexOf(u8, code, "catch 32 else 32;") != null); // WAKE
    try std.testing.expect(std.mem.indexOf(u8, code, "catch 1000 else 1000;") != null); // WAKE_US
}
    const pair = code[std.mem.indexOf(u8, code, "pub fn fillGroup(").?..];
    const g_at = std.mem.indexOf(u8, steps, "try m.issueLead(m.passStreamOf(l.pass));").?;
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, steps[0..std.mem.indexOf(u8, steps, "// everything the group queued").?], "issueLead("));
    for ([_][]const u8{ "slices of {d} us, the transport checked and the Engram worker's slots returned every slice (TF_DS_ISSUE_LEAD_WAIT=poll, TF_DS_ISSUE_LEAD_SLICE_US), a wait over {d} ms logged every 5 s (TF_DS_ISSUE_LEAD_WARN_MS, 0: never)", "one blocking-sync wait (TF_DS_ISSUE_LEAD_WAIT=block; the transport is checked and the Engram worker's slots returned before and after it, not during)", "TF_DS_ISSUE_LEAD_WARN_MS={s}: not a setting (0, or 1 to 3600000 milliseconds): {d}", "TF_DS_ISSUE_LEAD_WAIT={s}: not a mode (poll, block or backoff): poll", "TF_DS_ISSUE_LEAD_SLICE_US={s}: not a setting (1 to 100000 microseconds): {d}" }) |w|
    try std.testing.expect(std.mem.indexOf(u8, body, ".poll, .backoff => slices = try issue_lead.pollWait(gate, &w),") != null);
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

test "TF_DS_ENGRAM_KEEP_AHEAD: the switch is read as 1 and nothing else, the decision runs at the start of every rank, and rank 0 and the followers test one flag" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expect(std.mem.indexOf(u8, code, "keep_ahead_asked = std.mem.eql(u8, std.mem.span(v), \"1\");") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "if (keep_ahead_asked) try m.useKeepAhead();") != null);
    // the round counts itself once in a verify and once in a pass, on every rank (the same Model methods), the prompt's fill resets the count
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, code, "if (m.eh) |*eh| eh.ka.roundRan();"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "if (m.eh) |*eh| eh.ka.newPrompt();"));
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, code, "if (m.eh) |*eh| eh.ka.chunkBegins();"));
    // the pass line names the worker's starts and the first-layer wait
    try std.testing.expect(std.mem.indexOf(u8, code, "Engram worker starts this prompt {d}, first-layer wait after a round {d:.1} ms in {d} chunks") != null);
    // the worker's starts and the wait are counted where they happen
    const p = @embedFile("prompt.zig");
    try std.testing.expect(std.mem.indexOf(u8, p, "eh.ka.workerStarted();") != null);
    try std.testing.expect(std.mem.indexOf(u8, p, "if (j == 0) eh.ka.firstLayerWait(waited);") != null);
}

test "TF_DS_ENGRAM_KEEP_AHEAD: the round's buffers and the worker's are separate memory in a built EngramHost's field list, and a shared one is named" {
    const a = std.testing.allocator;
    const hashes = try a.alloc(i64, 64);
    defer a.free(hashes);
    const wk0 = try a.alloc(i64, 64);
    defer a.free(wk0);
    var round_r: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
    var worker_r: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
    round_r[0] = keep_ahead.rangeOf(i64, hashes);
    worker_r[0] = keep_ahead.rangeOf(i64, wk0);
    try std.testing.expect(keep_ahead.firstShared(&round_r, &worker_r) == null);
    worker_r[1] = keep_ahead.rangeOf(i64, hashes[8..16]); // a worker slot placed inside the round's hashes
    try std.testing.expectEqual([2]usize{ 0, 1 }, keep_ahead.firstShared(&round_r, &worker_r).?);
    // every region keepRanges lists is one the round or the worker writes: the EngramHost fields it reads exist under these names
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    const at = std.mem.indexOf(u8, code, "fn keepRanges(") orelse return error.TestUnexpectedResult;
    const body = code[at .. std.mem.indexOfPos(u8, code, at, "\n}\n") orelse return error.TestUnexpectedResult];
    for ([_][]const u8{ "eh.hashes", "eh.flat", "eh.w)", "eh.s)", "eh.rows", "eh.wk_flat", "eh.wk_w", "eh.wk_s", "eh.wk_hashes", "eh.wk_rows", "pr.round", "pr.wk" }) |w|
        try std.testing.expect(std.mem.indexOf(u8, body, w) != null);
}

test "TF_DS_ENGRAM_KEEP_AHEAD: the overlap list holds every region of the widest worker ring, and a full list is an error, never a silent drop" {
    const Stage = engram_rows.Stage;
    const wk: [engram_rows.max_wk][2]Stage = @splat(@splat(Stage.empty));
    // the worker's own fields, a hash list and two row lists a slot, then every slot's record stages: all of them fit
    var list: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
    const fields: usize = 3 + engram_rows.max_wk * (1 + 2);
    try std.testing.expectEqual(fields + engram_rows.max_wk * 6, try workerStageRanges(&list, fields, &wk, engram_rows.max_wk));
    // a list with one entry left refuses the stages instead of dropping the ones that do not fit
    try std.testing.expectError(error.KeepAheadRangesFull, workerStageRanges(&list, keep_ahead_ranges - 1, &wk, engram_rows.max_wk));
    // the capacity follows the ring (the worker's side at max_wk slots fits with the round's 11 ranges beside it)
    try std.testing.expect(keep_ahead_ranges >= 11 and keep_ahead_ranges >= fields + engram_rows.max_wk * 6);
    const src = @embedFile("model.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "const keep_ahead_ranges = 3 + 9 * engram_rows.max_wk;") != null);
    const silent = "if (n.* < list" ++ ".len) {";
    try std.testing.expect(std.mem.indexOf(u8, src, silent) == null);
}

test "TF_DS_ENGRAM_KEEP_AHEAD: with two worker slots (the stagger off) the slots past them are not the worker's memory, whatever they hold" {
    const a = std.testing.allocator;
    const Stage = engram_rows.Stage;
    const hashes = try a.alloc(i64, 64);
    defer a.free(hashes);
    const real = try a.alloc(u8, 32);
    defer a.free(real);
    var wk: [engram_rows.max_wk][2]Stage = @splat(@splat(Stage.empty));
    wk[0][0].bytes = real;
    // slots 2 and 3 as an `undefined` array may leave them: pointing into the round's hashes
    for (wk[2..]) |*row| for (row) |*st| {
        st.bytes = std.mem.sliceAsBytes(hashes[8..16]);
        st.ids = hashes[16..24];
    };
    var round_r: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
    round_r[0] = keep_ahead.rangeOf(i64, hashes);
    var worker_r: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
    _ = try workerStageRanges(&worker_r, 0, &wk, 2);
    try std.testing.expect(keep_ahead.firstShared(&round_r, &worker_r) == null);
    // with four slots they are the worker's and the overlap is named
    var worker4: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
    _ = try workerStageRanges(&worker4, 0, &wk, engram_rows.max_wk);
    try std.testing.expect(keep_ahead.firstShared(&round_r, &worker4) != null);
    // Rows.open leaves the slots it does not make empty, never undefined
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("engram_rows.zig"), "var wk: [max_wk][2]Stage = @splat(@splat(Stage.empty));") != null);
}

test "TF_DS_ENGRAM_PARTNER: the worker reads through chunkReadSplit, the start is agreed before the host lane opens, and the host lane closes before the others" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    const p = @embedFile("prompt.zig");
    try std.testing.expect(std.mem.indexOf(u8, p, "pr.chunkReadSplit(&pr.wk[slot][j]") != null);
    const f = code[std.mem.indexOf(u8, code, "fn openPartner(").?..std.mem.indexOf(u8, code, "fn openPair(").?];
    try std.testing.expect(std.mem.indexOf(u8, f, "agreeAll(why == null)").? < std.mem.indexOf(u8, f, "Roci.openLane(").?);
    const close = code[std.mem.indexOf(u8, code, "pub fn close(m: *Model) void {").?..];
    try std.testing.expect(std.mem.indexOf(u8, close, "m.partner_rc) |rc| rc.close()").? < std.mem.indexOf(u8, close, "m.closePair()").?);
}

test "TF_DS_ENGRAM_PARTNER: each pass's line of the pair log carries the partner's counts and its times since the last line" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    const lr = code[std.mem.indexOf(u8, code, "    fn leadReport(").?..];
    for ([_][]const u8{ "partner: {d} steps, {d} split, {d} mismatched, {d} header late, {d} half late, {d} skipped, {d} declined; records read here", "partner time: header wait", "own half read", "the whole lists at that rate", "pl.c.since(m.partner_seen)" }) |w|
        try std.testing.expect(std.mem.indexOf(u8, lr, w) != null);
}

/// Layers as the pool's caches see them (the test config's compress ratios, KV and index sources): what Caches.layerBytes reads.
fn poolTestLayers(lays: *[40]weights.Layer, cfg: Config) []const weights.Layer {
    const one: weights.Linear = .{ .words = 0, .k = 1, .n = 1 };
    for (lays, 0..) |*l, i| {
        l.* = .{ .idx = @intCast(i), .ratio = cfg.ratios[i] };
        l.comp_wkv = if (cfg.kv_sources.has(i)) one else null;
        l.idx_wk = if (cfg.index_sources.has(i)) one else null;
    }
    return lays;
}

test "the memory guard comes from the pass rows at every size: the estimate scales with the pass and the lane's arena, the short is what the floor leaves out" {
    const keep = rocinante_mod.max_cap;
    const keep_caps = rocinante_mod.Transport.shape_caps;
    defer {
        rocinante_mod.max_cap = keep;
        rocinante_mod.Transport.shape_caps = keep_caps;
    }
    var prev_need: u64 = 0;
    var p: usize = 2048;
    var lane4096: u64 = 0;
    while (p <= pass_rows_max) : (p += 2048) {
        rocinante_mod.setCap(p);
        const lane = rocinante_mod.layoutOf(.prompt).total;
        if (p == 4096) lane4096 = lane;
        const need = pairEstimate(p, lane);
        // with nothing free the short is the estimate and the floor; with that much free, nothing; one byte less, one byte short
        try std.testing.expectEqual(need + pair_guard_bytes, pairShort(0, p, lane));
        try std.testing.expectEqual(@as(u64, 0), pairShort(need + pair_guard_bytes, p, lane));
        try std.testing.expectEqual(@as(u64, 1), pairShort(need + pair_guard_bytes - 1, p, lane));
        // the second pass takes more with every unit of rows
        try std.testing.expect(need > prev_need);
        prev_need = need;
        // never less than the measured 4,096-row pass scaled to the rows
        try std.testing.expect(need >= 3_370_548_688 * p / 4096);
    }
    try std.testing.expectEqual(@as(usize, 16384), pass_rows_max);
    // the measured 4,096-row pass: its chunk and node and its lane's arena
    try std.testing.expect(pairEstimate(4096, lane4096) >= 3_370_548_688 + lane4096);
    // on a Spark with 12 GiB free: 2,048 and 4,096 fit, 8,192 and the larger are short
    const free: u64 = 12 << 30;
    rocinante_mod.setCap(2048);
    try std.testing.expectEqual(@as(u64, 0), pairShort(free, 2048, rocinante_mod.layoutOf(.prompt).total));
    rocinante_mod.setCap(4096);
    try std.testing.expectEqual(@as(u64, 0), pairShort(free, 4096, rocinante_mod.layoutOf(.prompt).total));
    for ([_]usize{ 8192, 10240, 16384 }) |big| {
        rocinante_mod.setCap(big);
        try std.testing.expect(pairShort(free, big, rocinante_mod.layoutOf(.prompt).total) > 0);
    }
    // the source: the guard runs before openPair allocates, the floor of a single pass after the start
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expect(std.mem.indexOf(u8, code, "pairShort(free, m.pass_rows, lane_arena) != 0) why = try pairShortText(") != null);
    // the stagger's start line says the pool and the floor beside the free memory; the single pass's the same
    try std.testing.expect(std.mem.indexOf(u8, code, "the KV pool holds {d} positions ({d:.2} GiB on this Spark, POOL) and the floor kept free is {d:.1} GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "GiB free after the start's allocations (the floor is {d:.1} GiB); the KV pool holds {d} positions, {d:.2} GiB on this Spark") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "try openPair(m, gpa, d, o, sp, c) else try passFloor(m, c);") != null);
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, code, "pair_estimate_bytes"));
}

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

test "TF_DS_KV_VMM: the reservation holds every plane of the pool at the most positions, and the planes' backed bytes are the layout's rounded up per plane" {
    const config_mod = @import("config.zig");
    const gpa = std.testing.allocator;
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    const layers = poolTestLayers(&lays, cfg);
    const pool: usize = 1 << 20;
    const most: usize = 4 * pool;
    const gran: usize = 2 << 20;
    const va = prompt.Caches.vaBytes(&cfg, layers, most, 16, gran);
    var vp = try kv_vmm.Pool(kv_vmm.Fake).open(gpa, .{ .gpa = gpa }, gran, kv_vmm.seg_default, va);
    defer vp.b.live.deinit(gpa);
    defer vp.deinit();
    // the planes as initFrom takes them: each layer's, in its order, `pool` positions now and room for `most`
    var planes: usize = 0;
    var rounded: usize = 0;
    for (layers) |lay| {
        const b = prompt.Caches.layerBytes(&cfg, lay, pool, 16);
        const m = prompt.Caches.layerBytes(&cfg, lay, most, 16);
        inline for (.{ "comp_codes", "comp_scales", "idx_codes", "idx_scales", "raw_kv", "raw_score" }) |f| {
            if (@field(b, f) != 0) {
                _ = try vp.addPlane(@field(m, f), @field(b, f));
                planes += 1;
                rounded += std.mem.alignForward(usize, @field(b, f), gran);
            }
        }
    }
    try std.testing.expect(planes > 0);
    try std.testing.expectEqual(planes, vp.planes.items.len);
    try std.testing.expectEqual(@as(u64, va), vp.carved); // the reservation is exactly the planes' stretches
    try std.testing.expectEqual(@as(u64, rounded), vp.mapped);
    const layout = prompt.Caches.poolBytes(&cfg, layers, pool, 16);
    try std.testing.expect(vp.mapped >= layout and vp.mapped < layout + planes * gran); // under one granule of rounding a plane
    // a pool of the most positions fits the stretches: every plane grows to its full size
    for (0..planes) |i| try vp.grow(@intCast(i), vp.planes.items[i].max);
    try std.testing.expect(vp.mapped >= prompt.Caches.poolBytes(&cfg, layers, most, 16));
}

test "TF_DS_KV_FREE_TARGET_GIB live: the planes grow and shrink at their tails in the plane order, addresses fixed, and the mapped bytes follow vmmBytes" {
    const config_mod = @import("config.zig");
    const gpa = std.testing.allocator;
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    const layers = poolTestLayers(&lays, cfg);
    const small: usize = 1 << 20;
    const big: usize = 4 << 20;
    const gran: usize = 2 << 20;
    const va = prompt.Caches.vaBytes(&cfg, layers, big, 16, gran);
    var vp = try kv_vmm.Pool(kv_vmm.Fake).open(gpa, .{ .gpa = gpa }, gran, kv_vmm.seg_default, va);
    defer vp.b.live.deinit(gpa);
    defer vp.deinit();
    var a: [prompt.Caches.max_planes]prompt.Caches.Plane = undefined;
    var b: [prompt.Caches.max_planes]prompt.Caches.Plane = undefined;
    const n = prompt.Caches.planeSizes(&cfg, layers, small, 16, &a);
    try std.testing.expectEqual(n, prompt.Caches.planeSizes(&cfg, layers, big, 16, &b));
    var at: [prompt.Caches.max_planes]u64 = undefined;
    for (0..n) |i| at[i] = try vp.addPlane(b[i].bytes, a[i].bytes);
    try std.testing.expectEqual(@as(u64, prompt.Caches.vmmBytes(&cfg, layers, small, 16, gran)), vp.mapped);
    for (0..n) |i| try vp.grow(@intCast(i), b[i].bytes); // the pool grows to `big`
    try std.testing.expectEqual(@as(u64, prompt.Caches.vmmBytes(&cfg, layers, big, 16, gran)), vp.mapped);
    for (0..n) |i| try std.testing.expectEqual(at[i], vp.addressOf(@intCast(i))); // nothing moved
    for (0..n) |i| _ = try vp.shrink(@intCast(i), a[i].bytes); // and back to `small`
    const want = prompt.Caches.vmmBytes(&cfg, layers, small, 16, gran);
    try std.testing.expect(vp.mapped >= want and vp.mapped < want + n * kv_vmm.seg_default); // whole segments: under one a plane over
    try std.testing.expectEqual(vp.mapped, vp.b.total());
    // the kinds the grow clears: the index keys' scales are 127, every other plane 0 (a new pool's contents)
    var idx: usize = 0;
    for (a[0..n]) |x| {
        if (x.kind == .idx_scales) idx += 1;
    }
    try std.testing.expect(idx > 0);
}

test "TF_DS_KV_FREE_TARGET_GIB live: a graph holds only plane bases, and every position below the pool stays backed through grows and shrinks" {
    // What the round graphs bake (round.zig: cs.<plane>[layer], the whole pool's Caches): each plane's BASE, plus counts that do not depend
    // on the pool (cs.slots, raw_rows). Which rows a kernel touches comes from device buffers (keys, targets) the lanes fill from positions
    // below pool_cap. So a graph is safe across a resize exactly when (1) the bases never move and (2) every row below the pool is backed.
    const config_mod = @import("config.zig");
    const gpa = std.testing.allocator;
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    const layers = poolTestLayers(&lays, cfg);
    const gran: usize = 2 << 20;
    const caps = [_]usize{ 1 << 20, 3 << 20, 1 << 20, 5 << 20, 2 << 20 }; // the pool's sizes, in turn (start, grow, shrink, grow past, shrink)
    const most: usize = 5 << 20;
    var vp = try kv_vmm.Pool(kv_vmm.Fake).open(gpa, .{ .gpa = gpa }, gran, kv_vmm.seg_default, prompt.Caches.vaBytes(&cfg, layers, most, 16, gran));
    defer vp.b.live.deinit(gpa);
    defer vp.deinit();
    var now: [prompt.Caches.max_planes]prompt.Caches.Plane = undefined;
    var big: [prompt.Caches.max_planes]prompt.Caches.Plane = undefined;
    const n = prompt.Caches.planeSizes(&cfg, layers, caps[0], 16, &now);
    _ = prompt.Caches.planeSizes(&cfg, layers, most, 16, &big);
    var bases: [prompt.Caches.max_planes]u64 = undefined;
    for (0..n) |i| bases[i] = try vp.addPlane(big[i].bytes, now[i].bytes);
    // the graph, captured at the first size: it holds the bases, and (the negative control) one address past the plane's first size
    var past: [prompt.Caches.max_planes]u64 = undefined;
    for (0..n) |i| past[i] = bases[i] + now[i].bytes;
    var cap = caps[0];
    for (caps[1..]) |next| {
        var want: [prompt.Caches.max_planes]prompt.Caches.Plane = undefined;
        _ = prompt.Caches.planeSizes(&cfg, layers, next, 16, &want);
        for (0..n) |i| {
            if (next > cap) try vp.grow(@intCast(i), want[i].bytes) else _ = try vp.shrink(@intCast(i), want[i].bytes);
        }
        cap = next;
        for (0..n) |i| {
            try std.testing.expectEqual(bases[i], vp.addressOf(@intCast(i))); // (1) the baked bases did not move
            try std.testing.expect(vp.b.isMapped(bases[i], want[i].bytes)); // (2) every row of the pool is backed
        }
    }
    // the checker can fail: after the last shrink an address a graph captured at the biggest size past the new end is unbacked, and the
    // gap between planes' stretches is never backed
    var found_unbacked = false;
    for (0..n) |i| {
        var at_max: [prompt.Caches.max_planes]prompt.Caches.Plane = undefined;
        _ = prompt.Caches.planeSizes(&cfg, layers, most, 16, &at_max);
        if (!vp.b.isMapped(bases[i] + at_max[i].bytes - 1, 1)) found_unbacked = true;
    }
    try std.testing.expect(found_unbacked);
    try std.testing.expect(!vp.b.isMapped(past[0], vp.stride(big[0].bytes))); // a pointer past the old end, over the whole stretch
}

test "the round's graphs take plane bases of the whole pool: no pointer arithmetic on a cache plane, no pool end" {
    // (the source half of the argument above: a captured address is cs.<plane>[li] itself, never cs.<plane>[li] + something or cs.cap)
    const src = @embedFile("round.zig");
    const planes = [_][]const u8{ "cs.comp_codes[", "cs.comp_scales[", "cs.idx_codes[", "cs.idx_scales[", "cs.raw_kv[", "cs.raw_score[" };
    var uses: usize = 0;
    for (planes) |w| {
        var from: usize = 0;
        while (std.mem.indexOfPos(u8, src, from, w)) |at| {
            from = at + w.len;
            const close = std.mem.indexOfScalarPos(u8, src, from, ']').?;
            const after = std.mem.trimStart(u8, src[close + 1 ..][0..4], " ");
            try std.testing.expect(after.len == 0 or after[0] == ',' or after[0] == ')' or after[0] == ' ' or after[0] == '}'); // not "+ ..." or "* ..."
            uses += 1;
        }
    }
    try std.testing.expect(uses >= 8);
    try std.testing.expect(std.mem.indexOf(u8, src, "cs.cap") == null); // the round never reads the pool's size
}

test "TF_DS_KV_FREE_TARGET_GIB live: a rank without a virtual-memory pool leaves the pool and says the change applies at the next start; the decision order is fixed" {
    const src = @embedFile("model.zig");
    const f = std.mem.indexOf(u8, src, "    fn resizePool(m: *Model").?;
    const body = src[f .. f + std.mem.indexOf(u8, src[f..], "\n    }\n").?];
    const novmm = std.mem.indexOf(u8, body, "has no virtual-memory pool").?;
    try std.testing.expect(std.mem.indexOf(u8, body, "return;") == null or std.mem.indexOf(u8, body, "return;").? > std.mem.indexOf(u8, body, "try star.allGather(").?); // a rank without the pool never leaves before the gather
    const sync = std.mem.indexOf(u8, body, "try m.syncPoolStreams();").?;
    const gather = std.mem.indexOf(u8, body, "try star.allGather(").?;
    const decide = std.mem.indexOf(u8, body, "kv.decideResize(").?;
    const move = std.mem.indexOf(u8, body, "m.caches.resizeVmm(").?;
    const ballot = std.mem.indexOf(u8, body, "try star.allGather(&[1]u8{failed}, ballots[0..world]);").?;
    const commit = std.mem.indexOf(u8, body, "m.pool_cap = d.positions;").?;
    // idle first, then every rank's offer, one decision, the move, the ranks' ballot on it, and only then the new size
    try std.testing.expect(novmm < sync and sync < gather and gather < decide and decide < move and move < ballot and ballot < commit);
    try std.testing.expect(std.mem.indexOf(u8, body, "m.kv_live_floor") != null); // rank 0's extents in use bound a shrink
    try std.testing.expect(std.mem.indexOf(u8, src, "@backingInt(live_config.Slot.kv_free_target) => try m.resizePool(version, it.value),") != null);
    // lanes sets the floor from the extents in use just before the frame is applied
    const lsrc = @embedFile("lanes.zig");
    const lf = std.mem.indexOf(u8, lsrc, "fn liveCfg(").?;
    const lb = lsrc[lf .. lf + std.mem.indexOf(u8, lsrc[lf..], "\n    }\n").?];
    try std.testing.expect(std.mem.indexOf(u8, lb, "m.kv_live_floor = held.end;").? < std.mem.indexOf(u8, lb, "try m.applyCfg(version").?);
}

test "TF_DS_KV_FREE_TARGET_GIB live: Model.open makes the free target live exactly when the pool is in virtual memory" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const caches = std.mem.indexOf(u8, body, "m.caches = if (o.kv_vmm)").?;
    const able = std.mem.indexOf(u8, body, "live_config.setAble(.kv_free_target, m.kv_vmm != null);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(able > caches); // once the pool exists, so the bit says what the pool is
}

test "TF_DS_KV_FREE_TARGET_GIB live: the streams a resize waits for are every Stream field of the model and the round" {
    const S = struct {
        // the Stream fields of T (a Stream or an optional one), as names
        fn names(comptime T: type) []const []const u8 {
            comptime var out: []const []const u8 = &.{};
            const info = @typeInfo(T).@"struct";
            inline for (info.field_names, info.field_types) |name, FT| {
                const is = FT == cuda.Stream or FT == ?cuda.Stream or FT == [2]cuda.Stream;
                if (is) out = out ++ &[_][]const u8{name};
            }
            return out;
        }
        fn same(comptime want: []const []const u8, comptime got: []const []const u8) !void {
            try std.testing.expectEqual(want.len, got.len);
            inline for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
        }
    };
    // a stream added to either struct fails here until poolStreams waits for it (or says why it cannot touch the pool)
    try S.same(&Model.pool_stream_models, S.names(Model));
    try S.same(&Model.pool_stream_round, S.names(round.Round));
}

test "a start short of memory names the GiB short and the POOL that would make it fit, never one below a full request, and says when none does" {
    const config_mod = @import("config.zig");
    const gpa = std.testing.allocator;
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    const layers = poolTestLayers(&lays, cfg);
    const served_pool: usize = 16 * 1048576;
    const v: PoolView = .{ .c = &cfg, .layers = layers, .slots = 16, .pool = served_pool, .floor = 1048576 };
    // the pool's caches grow with its positions, and the per-position bytes are the code's (Caches.layerBytes)
    const b1 = v.bytes(1 << 20);
    const b16 = v.bytes(served_pool);
    try std.testing.expect(b16 > 15 * (b1 - v.bytes(0)) and b16 < 17 * b1);
    const per_position = @as(f64, @floatFromInt(b16 - b1)) / @as(f64, @floatFromInt(served_pool - (1 << 20)));
    try std.testing.expect(per_position > 100 and per_position < 8192);
    // nothing short: the pool stays; a little short: a pool just under, a multiple of 65,536; the pool frees what was asked
    try std.testing.expectEqual(@as(?usize, served_pool), poolThatFrees(v, 0));
    for ([_]u64{ 1, 1 << 20, 1 << 30, 3 << 30, 5 << 30 }) |short| {
        const got = poolThatFrees(v, short) orelse return error.TestUnexpectedResult;
        try std.testing.expect(got < served_pool and got >= 1048576 and got % pool_step == 0);
        try std.testing.expect(b16 - v.bytes(got) >= short); // it frees enough
        try std.testing.expect(got + pool_step >= served_pool or b16 - v.bytes(got + pool_step) < short); // and no larger pool does
    }
    // more short than the whole pool above a full request holds: none fits
    const most = b16 - v.bytes(1048576);
    try std.testing.expect(poolThatFrees(v, most) != null);
    try std.testing.expectEqual(@as(?usize, null), poolThatFrees(v, most + 1));
    // a pool at the floor has nothing to give
    var at_floor = v;
    at_floor.pool = 1048576;
    try std.testing.expectEqual(@as(?usize, null), poolThatFrees(at_floor, 1));
    // the words of a start that fits with a smaller pool
    var buf: [900]u8 = undefined;
    const text = try memoryRefusal(&buf, "the second pass", true, 16384, 30 << 30, 22 << 30, v);
    try std.testing.expect(std.mem.indexOf(u8, text, "TF_DS_PASS_ROWS=16384") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "8.00 GiB short") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "22.00 GiB free") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "ESTIMATED from the 4,096-row pass measured") != null);
    const want = poolThatFrees(v, 8 << 30).?;
    var pb: [32]u8 = undefined;
    try std.testing.expect(std.mem.indexOf(u8, text, try std.fmt.bufPrint(&pb, "POOL={d} frees it", .{want})) != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "TF_DS_WINDOW on rank 0 and --pool on the followers") != null);
    // the words of a start no pool saves
    const none = try memoryRefusal(&buf, "the pass", false, 16384, 6 << 30 << 10, 0, v);
    try std.testing.expect(std.mem.indexOf(u8, none, "no POOL fits") != null);
    try std.testing.expect(std.mem.indexOf(u8, none, "MEASURED after the start's allocations") != null and std.mem.indexOf(u8, none, "ESTIMATED") == null);
    try std.testing.expect(std.mem.indexOf(u8, none, "1048576 positions") != null);
    try std.testing.expect(std.mem.indexOf(u8, none, "POOL=") == null);
    // the pair's refusal and the single pass's are these words
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expect(std.mem.indexOf(u8, code, "memoryRefusal(&buf, \"the pass\", false, m.pass_rows, pair_guard_bytes, free, poolView(m, c))") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "return memoryRefusal(buf, what, true, pass_rows, (lanes - 1) * pairEstimate(pass_rows, lane_arena) + pair_guard_bytes, free, v);") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "const what = if (lanes >= 3) \"the second and third passes (TF_DS_ROCI_STAGGER_LANES=3)\" else \"the second pass\";") != null);
    // a pool under one stream's context is refused at the start, whatever its size: the served 1,048,576-token context needs a pool of that many
    try std.testing.expectEqual(@as(?[]const u8, null), poolBelowContext(1048576, 1048576));
    try std.testing.expectEqual(@as(?[]const u8, null), poolBelowContext(16 * 1048576, 1048576));
    try std.testing.expectEqual(@as(?[]const u8, null), poolBelowContext(4096, 0)); // (no context asked: the pool is the window)
    for ([_]usize{ 1, 262144, 1048575 }) |small| {
        const w = poolBelowContext(small, 1048576) orelse return error.TestUnexpectedResult;
        try std.testing.expect(std.mem.indexOf(u8, w, "POOL") != null and std.mem.indexOf(u8, w, "the context is never lowered") != null);
    }
    try std.testing.expect(std.mem.indexOf(u8, code, "if (poolBelowContext(o.pool, o.context)) |why| {") != null);
    // the floor is the context the stream holds (never the 1M lowered): the view takes it from ctx_cap
    try std.testing.expect(std.mem.indexOf(u8, code, ".pool = m.pool_cap, .floor = m.ctx_cap") != null);
}

test "the keep-ahead overlap list holds every range of the largest ring: no range is dropped for a long list" {
    const Stage = engram_rows.Stage;
    var wk: [engram_rows.max_wk][2]Stage = @splat(@splat(Stage.empty));
    var list: [keep_ahead_ranges]keep_ahead.Range = @splat(.{});
    // every slot of the largest ring: six ranges (two layers' bytes, ids and destinations); the scratch arrays and the slots'
    // hash and row buffers come to the rest, three a slot and three
    try std.testing.expectEqual(6 * engram_rows.max_wk, try workerStageRanges(&list, 0, &wk, engram_rows.max_wk));
    try std.testing.expect(3 + 3 * engram_rows.max_wk + 6 * engram_rows.max_wk <= keep_ahead_ranges);
    // the formula dl-4 counted: 3 + 9 ranges a slot, 75 at eight slots, 147 at sixteen (the 64-entry list dropped the surplus without a word)
    try std.testing.expectEqual(@as(usize, 147), keep_ahead_ranges);
    // the add never drops: the list has no length check to fall silent on
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, code, "if (n.* < list.len)"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, code, "if (n < list.len)"));
}

test "after a transport's or a decode ring's failure no round delivers tokens: checkTransport says so, and rank 0's verify and drafter pass ask it before they send a frame and again before they read what the GPU computed" {
    var m: Model = undefined;
    m.two = null;
    m.two_b = null;
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

test "TF_DS_ENGRAM_FAULT names the engine's rank among all its Sparks (o.rank below o.world), never the 2D split's ring position" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "prompt.engram_fault = prompt.EngramFault" ++ ".parse(").?; // (split: the test must not find itself)
    const tail = src[at..][0..400];
    try std.testing.expect(std.mem.indexOf(u8, tail, "@intCast(o.world)") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "prompt.engram_fault_rank = @intCast(o." ++ "rank);") != null);
    // on four Sparks the split is a ring of two: rank 2 would be refused and both rings' rank 0 would fire
    const sp = try prompt2d.split(2, 4);
    try std.testing.expectEqual(@as(u32, 2), sp.world);
    try std.testing.expect((try prompt.EngramFault.parse("2:3", 4)).asked);
    try std.testing.expectError(error.BadFault, prompt.EngramFault.parse("2:3", sp.world));
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

test "the KV pool is the ranks' vote: decided after the chunk and before the caches, from the vote only in the free-target mode, and every rank's ledger is gathered at the start's end" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const chunk = std.mem.indexOf(u8, body, "m.ch = try prompt.Chunk.init(").?;
    const vote = std.mem.indexOf(u8, body, "const pool = try votePool(m, o, c);").?;
    const caches = std.mem.indexOf(u8, body, "m.caches = if (o.kv_vmm) try openVmmCaches(m, ctx, c, pool) else try prompt.Caches.init(&m.eng, &m.arena, pool, m.streams);").?;
    try std.testing.expect(chunk < vote and vote < caches);
    try std.testing.expect(std.mem.indexOf(u8, body, "prompt.Caches.init(&m.eng, &m.arena, o.pool") == null); // no other way to size the pool
    try std.testing.expect(std.mem.indexOf(u8, body, "m.pool_cap = pool;").? > vote);
    const gather = std.mem.indexOf(u8, body, "try gatherLedger(m, free_after_pool);").?;
    try std.testing.expect(gather > caches and gather < std.mem.indexOf(u8, body, "        return m;").?);
    // without a free target each rank keeps its own pool (the recipe's rule: a follower's may be larger), as before the vote
    const f = std.mem.indexOf(u8, src, "fn votePool(").?;
    const vbody = src[f .. f + std.mem.indexOf(u8, src[f..], "\n}\n").?];
    try std.testing.expect(std.mem.indexOf(u8, vbody, "if (d.verdict == .auto) d.pool else o.pool") != null);
    try std.testing.expect(std.mem.indexOf(u8, vbody, "m.comm.host.star") != null); // the host star, never the device communicator
}

test "a cfg frame's keep-ahead: on only where the start verified it can run, off always, unset is the start's; the other ranks' rule is the same function" {
    var m: Model = undefined;
    m.rank = 2;
    m.keep_ahead = false;
    m.keep_ahead_start = false;
    m.keep_ahead_able = false;
    const slot = @backingInt(live_config.Slot.keep_ahead);
    // a rank that could not run it keeps it off (the start's check is the only thing that turns it on)
    try std.testing.expect(!m.keep_ahead);
    m.keep_ahead_able = true;
    try std.testing.expect(m.keep_ahead);
    try std.testing.expect(!m.keep_ahead);
    // `unset` is resolved by rank 0 from its start's value before it travels; a rank never reads the marker as "on"
    try std.testing.expectEqual(@as(u32, 0), m.cfgStart(slot));
    m.keep_ahead_start = true;
    try std.testing.expectEqual(@as(u32, 1), m.cfgStart(slot));
    try std.testing.expect(!m.keep_ahead);
    // a slot this build does not know changes nothing
    try std.testing.expect(!m.keep_ahead);
    // off stays possible when the start could not have turned it on (a start that asked, then a live off, then a live on again)
    m.keep_ahead_able = false;
    m.keep_ahead = true;
    try std.testing.expect(!m.keep_ahead);
}

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

test "TF_DS_MERGE_D says in the speed report's words why it does not run: a kit without the twins, or without its extra set" {
    // (the lines start "TF_DS_MERGE_D: off: ": core/speed_report.zig keeps them as the feature's reason)
    const src = @embedFile("model.zig");
    const lead = "TF_DS_MERGE_D: " ++ "off: ";
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, lead ++ "the loaded kit's extra set has no _sparse_attn_merge_d: " ++ "kit a02466e or later\""));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, lead ++ "the kit's extra set (aot-extra/) did not load ({s})\""));
}

test "TF_DS_WOA_MERGE: Model.open refuses a bad value by name at its call site (the parse is pure), and says so when it is on" {
    const src = @embedFile("model.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "mx.woa_merge = mx.woaMerge" ++ "Wanted(woam_text) orelse {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_WOA_MERGE={s}: not 0 or " ++ "1; not started"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_WOA_MERGE=1: wo_a's FP8 groups of a prompt chunk run as one launch (grid.z = " ++ "groups)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_WOA_MERGE=0: wo_a's FP8 groups of a prompt chunk run one launch a " ++ "group (the operator's choice)"));
}

test "TF_DS_AQ_VEC: Model.open refuses a bad value by name at its call site (the parse itself is pure and logs nothing), and says on or off once the kernels load" {
    const src = @embedFile("model.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "mx.aq_vec_on = mx.aqVec" ++ "Wanted(aq_text) orelse {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_AQ_VEC={s}: not 0 or " ++ "1; not started"));
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    const load = std.mem.indexOf(u8, body, "m.mxk = if (c.format == .native) try mx.Kernels.load(d, cuda.kernels.dsv41_native) else null;").?;
    const use = std.mem.indexOf(u8, body, "useAqVec(if (m.mxk) |*x| x else null);").?;
    try std.testing.expect(parse < load and load < use);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "useAqVec("));
    // the start lines: on, off by the switch, off for an image without the twin
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_AQ_VEC: on: act_quant on four " ++ "lanes"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_AQ_VEC=0: act_quant runs the baseline " ++ "kernel"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_AQ_VEC: off: this image has no tf_dsn_act_quant_" ++ "vec_kernel"));
}

test "TF_DS_L2_CHUNK_KIB: Model.open refuses a bad value by name at its call site (the parse itself is pure and logs nothing)" {
    const src = @embedFile("model.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "ops_mod.l2_chunk_bytes = ops_mod.l2ChunkWanted(chunk_text) orelse {") != null);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_L2_CHUNK_KIB={s}: not one of 1, 2, 4, " ++ "8, 16 (KiB); not started"));
}

test "TF_DS_MERGE_D: unset or empty is 16 columns a program; 3 is 16, 1 is 128, 2 is 64, 0 is off; any other text refuses the start" {
    try std.testing.expectEqual(@as(?usize, 16), mergeDWanted(null));
    try std.testing.expectEqual(@as(?usize, 16), mergeDWanted(""));
    try std.testing.expectEqual(@as(?usize, 16), mergeDWanted("3"));
    try std.testing.expectEqual(@as(?usize, 128), mergeDWanted("1"));
    try std.testing.expectEqual(@as(?usize, 64), mergeDWanted("2"));
    try std.testing.expectEqual(@as(?usize, 0), mergeDWanted("0"));
    // a typo, a column count (the allowed values are codes, not widths), a number past the table and a sign are refused
    for ([_][]const u8{ "nonsense", "16", "128", "4", "-1", " 1", "01", "true" }) |bad|
        try std.testing.expectEqual(@as(?usize, null), mergeDWanted(bad)); // (pure: it logs nothing; Model.open logs the refusal and returns error.BadSetting)
    // the call site names the allowed values, once, and returns the error
    const src = @embedFile("model.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_MERGE_D={s}: not one of 0 (off), 1 (128 columns a program), " ++ "2 (64), 3 (16, the default); not started"));
    try std.testing.expect(std.mem.indexOf(u8, src, "merge_d_asked = mergeDWanted(merge_text) orelse {") != null);
}

test "TF_DS_MERGE_D: a kit without the asked width runs 128, else 64, else none, and the start line says it ran another width" {
    const keep = tri_attn.merge_db;
    defer tri_attn.merge_db = keep;
    const all: MergeHave = .{ .c16 = true, .c64 = true, .c128 = true };
    const old_kit: MergeHave = .{ .c64 = true, .c128 = true }; // ec8b9ed: the kit before the 16-column variant
    try std.testing.expectEqual(@as(?usize, 16), mergeDResolve(16, all));
    try std.testing.expectEqual(@as(?usize, 128), mergeDResolve(16, old_kit)); // the fallback
    try std.testing.expectEqual(@as(?usize, 64), mergeDResolve(64, old_kit));
    try std.testing.expectEqual(@as(?usize, 64), mergeDResolve(16, .{ .c64 = true }));
    try std.testing.expectEqual(@as(?usize, null), mergeDResolve(16, .{}));
    try std.testing.expectEqual(@as(?usize, null), mergeDResolve(0, all));
    useMergeD(16, true, null, 4, all);
    try std.testing.expectEqual(@as(usize, 16), tri_attn.merge_db);
    useMergeD(16, true, null, 4, old_kit);
    try std.testing.expectEqual(@as(usize, 128), tri_attn.merge_db); // 128, not off, not 16
    useMergeD(16, true, null, 4, .{});
    try std.testing.expectEqual(@as(usize, 0), tri_attn.merge_db);
    useMergeD(16, false, "x", 4, all);
    try std.testing.expectEqual(@as(usize, 0), tri_attn.merge_db);
    useMergeD(16, true, null, 0, all);
    try std.testing.expectEqual(@as(usize, 0), tri_attn.merge_db);
    // the warn line exists once and names the fallback
    const src = @embedFile("model.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "TF_DS_MERGE_D: instead: runs {d} columns a program, not {d}: " ++ "the loaded kit's extra set has no _sparse_attn_merge_d at {d} columns (it needs the kit regenerated with that width)"));
}

test "TF_DS_ATTN_TWIN and TF_DS_ROPE_W are on by default: no packed-latent call, an extra set that did not load or one without the twin keeps the baseline kernel with an 'off' line, never a refused start" {
    const keep_t = tri_attn.attn_twin;
    defer tri_attn.attn_twin = keep_t;
    const keep_r = tri_norm.rope_w;
    defer tri_norm.rope_w = keep_r;
    const a = std.testing.allocator;
    const io = std.testing.io;
    tri_attn.attn_twin = .z;
    try useAttnTwin(a, io, "z", false, true, null, "/nonexistent/aot-extra");
    try std.testing.expectEqual(tri_attn.Twin.u, tri_attn.attn_twin); // no TF_DS_ATTN_U / _DECODE call would ask for it
    try useAttnTwin(a, io, "z", true, false, "FileNotFound", "/nonexistent/aot-extra");
    try std.testing.expectEqual(tri_attn.Twin.u, tri_attn.attn_twin); // the extra set did not load
    try useAttnTwin(a, io, "w", true, true, null, "/nonexistent/aot-extra");
    try std.testing.expectEqual(tri_attn.Twin.u, tri_attn.attn_twin); // the set holds no twin of that name
    tri_norm.rope_w = true;
    try useRopeW(true, false, "FileNotFound", 0);
    try std.testing.expect(!tri_norm.rope_w);
    try useRopeW(true, true, null, 0);
    try std.testing.expect(!tri_norm.rope_w);
    try useRopeW(true, true, null, 3);
    try std.testing.expect(tri_norm.rope_w);
    try useRopeW(false, true, null, 3);
    try std.testing.expect(!tri_norm.rope_w); // TF_DS_ROPE_W=0
    // the speed report prints each fallback: the lines say "off"
    const src = @embedFile("model.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "\"TF_DS_ATTN_TWIN={s}: " ++ "off: ") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "\"TF_DS_ROPE_W=1: " ++ "off: ") != null);
}

test "TF_DS_QKV_PLAIN is on by default: an extra set that did not load or one without the _q_kv_norm variant keeps the two launches with an 'off' line, never a refused start" {
    const keep = round.qkv_plain;
    defer round.qkv_plain = keep;
    round.qkv_plain = true;
    useQkvPlain(true, false, "FileNotFound", 0);
    try std.testing.expect(!round.qkv_plain); // the extra set did not load
    round.qkv_plain = true;
    useQkvPlain(true, true, null, 0);
    try std.testing.expect(!round.qkv_plain); // a kit without the variant (batch 4: aab01c1)
    useQkvPlain(true, true, null, 1);
    try std.testing.expect(round.qkv_plain);
    useQkvPlain(false, true, null, 1);
    try std.testing.expect(!round.qkv_plain); // TF_DS_QKV_PLAIN=0
    const src = @embedFile("model.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "\"TF_DS_QKV_PLAIN=1: " ++ "off: ") != null);
}

test "a kit twin's off line names the ranks that cannot run it and this rank's own state" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_ATTN_U=1: off: rank 2 cannot run it (this rank: can); every rank runs the baseline kernel", twinOffLine(&buf, "TF_DS_ATTN_U", 1 << 2, true));
    try std.testing.expectEqualStrings("TF_DS_IDX_U=1: off: ranks 0 and 3 cannot run it (this rank: its kit's extra set does not provide it); every rank runs the baseline kernel", twinOffLine(&buf, "TF_DS_IDX_U", 0b1001, false));
}

test "Model.open agrees TF_DS_ATTN_U, TF_DS_IDX_U and TF_DS_ATTN_U_DECODE over the ranks right after each is put into effect" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    const pairs = [_][2][]const u8{
        .{ "useAttnU(a, io, prompt2d.attn_u_asked", "try agreeTwin(d, &m.comm, m.stream, \"TF_DS_ATTN_U\", prompt2d.attn_u_asked, &tri_attn.attn_u);" },
        .{ "useIdxU(a, io, prompt2d.idx_u_asked", "try agreeTwin(d, &m.comm, m.stream, \"TF_DS_IDX_U\", prompt2d.idx_u_asked, &tri_index.idx_u);" },
        .{ "useAttnUDecode(a, io, prompt2d.attn_u_decode_asked", "try agreeTwin(d, &m.comm, m.stream, \"TF_DS_ATTN_U_DECODE\", prompt2d.attn_u_decode_asked, &tri_attn.attn_u_decode);" },
    };
    for (pairs) |p| {
        const use = std.mem.indexOf(u8, body, p[0]) orelse return error.TestUnexpectedResult;
        const agree = std.mem.indexOf(u8, body, p[1]) orelse return error.TestUnexpectedResult;
        try std.testing.expect(use < agree);
        // the agreement follows its own switch's line: no other line between
        const between = body[use..agree];
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, between, "\n"));
    }
    // a switch not asked for is never agreed (asked is every rank's alike), and a rank that cannot turns it off
    const f = std.mem.indexOf(u8, src, "fn agreeTwin(") orelse return error.TestUnexpectedResult;
    const fb = src[f .. f + (std.mem.indexOf(u8, src[f..], "\n}\n") orelse return error.TestUnexpectedResult)];
    const skip = std.mem.indexOf(u8, fb, "if (!asked) return;") orelse return error.TestUnexpectedResult;
    const gather = std.mem.indexOf(u8, fb, "ranksThatCannot(d, comm, stream, mine_on)") orelse return error.TestUnexpectedResult;
    const off = std.mem.indexOf(u8, fb, "on.* = false;") orelse return error.TestUnexpectedResult;
    try std.testing.expect(skip < gather and gather < off);
}

test "Model.open reads TF_DS_DRAFT_VOCAB with the kernel switches and puts it in effect after the drafter, its rows' quarters gathered, its check, the agreement, then the exchange" {
    const src = @embedFile("model.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse return error.TestUnexpectedResult];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "prompt2d.draft_vocab_asked = prompt2d.switchAsked(if (std.c.getenv(\"TF_DS_DRAFT_VOCAB\")) |v| std.mem.span(v) else null); // every rank alike (Settings)"));
    const init = std.mem.indexOf(u8, code, "m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);") orelse return error.TestUnexpectedResult;
    const use = std.mem.indexOf(u8, code, "try useDraftVocab(m, prompt2d.draft_vocab_asked);") orelse return error.TestUnexpectedResult;
    const sync = std.mem.indexOf(u8, code[init..], "try d.check(d.api.cuCtxSynchronize(), \"cuCtxSynchronize\");") orelse return error.TestUnexpectedResult;
    try std.testing.expect(init < use and use < init + sync);
    const at = std.mem.indexOf(u8, code, "fn useDraftVocab(") orelse return error.TestUnexpectedResult;
    const body = code[at .. at + (std.mem.indexOf(u8, code[at..], "\n}\n") orelse return error.TestUnexpectedResult)];
    const q = std.mem.indexOf(u8, body, "const quarters = try headQuarters(m.eng.d, &m.comm, m.stream, m.w.vocab_lo, m.w.vocab_hi);") orelse return error.TestUnexpectedResult;
    const prep = std.mem.indexOf(u8, body, "dr.vocabPrepare(&m.eng, &m.arena, quarters, m.comm.rank, &staging)") orelse return error.TestUnexpectedResult;
    const cannot = std.mem.indexOf(u8, body, "const cannot = try prompt2d.ranksThatCannot(m.eng.d, &m.comm, m.stream, why == null);") orelse return error.TestUnexpectedResult;
    const ex = std.mem.indexOf(u8, body, "try m.dr.?.vocabExchange(&m.eng, m.comm.rank, &staging.?);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(q < prep and prep < cannot and cannot < ex);
    // refused: the list's state dropped on every rank, the passes the whole vocabulary's
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "if (!agreed) if (m.dr) |*dr| dr.dropVocab();"));
}

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

    const ll = @import("logits_log.zig");
        }
    for (speed_defaults.table) |d| {
        }
    }
    for (ll.speed_names) |n| {
        var found = false;
        for (speed_defaults.table) |d| found = found or std.mem.eql(u8, d.env, n);
        try std.testing.expect(found);
    }
}

test "TF_DS_ENGRAM_PARTNER and TF_DS_ENGRAM_KEEP_AHEAD: a failed precondition falls back with one warning when it is the default and refuses by name when the operator set it" {
    const src = @embedFile("model.zig");
    // the partner split: the refusal is the operator's branch, the warning and the cleared flag the default's, and no lane is opened before either
    const at = std.mem.indexOf(u8, src, "fn openPartner(").?;
    const v = std.mem.indexOf(u8, body, "speed_defaults.verdict(why orelse").?;
    const refuse = std.mem.indexOf(u8, body[v..], ".refuse => {").?;
    const refused = std.mem.indexOf(u8, body[v..], "return error.PartnerRefused;").?;
    const warn = std.mem.indexOf(u8, body[v..], "std.log.warn(\"TF_DS_ENGRAM_PARTNER: off:").?;
    const cleared = std.mem.indexOf(u8, body[v..], "prompt2d.engram_partner_asked = false;").?;
    try std.testing.expect(refuse < refused and refused < warn and warn < cleared);
    try std.testing.expect(std.mem.indexOf(u8, body, "openLane(").? > v + cleared);
    // keep-ahead: the shared reader pool refuses only a switch the operator set
    const ka = std.mem.indexOf(u8, src, "fn useKeepAhead(").?;
    const kv = std.mem.indexOf(u8, kb, "speed_defaults.verdict(why, speed_defaults.defaulted(\"TF_DS_ENGRAM_KEEP_AHEAD\")) == .fallback").?;
    const kwarn = std.mem.indexOf(u8, kb[kv..], "std.log.warn(\"TF_DS_ENGRAM_KEEP_AHEAD: rank").?;
    const kerr = std.mem.indexOf(u8, kb[kv..], "return error.KeepAheadSharesReader;").?;
    try std.testing.expect(kwarn < kerr);
    // both are in the defaults table, so the speed report lists a fallback with its reason
    try std.testing.expect(speed_defaults.find("TF_DS_ENGRAM_PARTNER") != null and speed_defaults.find("TF_DS_ENGRAM_KEEP_AHEAD") != null);
}

    const gib: usize = 1 << 30;
    try std.testing.expect(passRowsFallback("4096", true, 64 * gib) != null); // a small box, the default: falls back
    try std.testing.expect(passRowsFallback("4096", true, 121 * gib) == null); // a Spark: runs
    try std.testing.expect(passRowsFallback("4096", false, 64 * gib) == null); // the operator set 4096: theirs
    try std.testing.expect(passRowsFallback("8192", true, 64 * gib) == null); // not the default's value
    try std.testing.expect(passRowsFallback("2048", true, 64 * gib) == null);
    try std.testing.expect(passRowsFallback(null, true, 64 * gib) == null);
}

test "TF_DS_ROCI_STAGGER: a failed precondition falls back with one warning when it is the default and refuses by name when the operator set it" {
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "fn openPair(").?;
    const v = std.mem.indexOf(u8, body, "speed_defaults.verdict(why orelse").?;
    const refuse = std.mem.indexOf(u8, body[v..], ".refuse => {").?;
    const refused = std.mem.indexOf(u8, body[v..], "return error.StaggerRefused;").?;
    const warn = std.mem.indexOf(u8, body[v..], "std.log.warn(\"TF_DS_ROCI_STAGGER: off:").?;
    const off = std.mem.indexOf(u8, body[v..], "prompt2d.roci_stagger_asked = false;").?;
    try std.testing.expect(refuse < refused and refused < warn and warn < off); // the refusal is the operator's branch, the fallback the other
    // the fallback names what it costs and runs before any allocation of the second pass
    try std.testing.expect(std.mem.indexOf(u8, body[v..], "speed_defaults.costOf(\"TF_DS_ROCI_STAGGER\")") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "m.two_b = try prompt2d.Two.init").? > std.mem.indexOf(u8, body, "prompt2d.roci_stagger_asked = false;").?);
    try std.testing.expect(std.mem.indexOf(u8, body, "needs passes of two resume units (TF_DS_PASS_ROWS=4096)") != null);
}

test "the four-way drafter's mismatch between ranks still refuses, naming the cause and the way out" {
    const src = @embedFile("model.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "the ranks' kits differ: give every Spark the same kit, or set TF_DS_DRAFT_4WAY=0") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "set TF_DS_DRAFT_MAIN_4WAY=0 on every Spark") != null);
}
    }
};
