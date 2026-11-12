//! A synthetic store: conversations of turns kept through the store's own write path as the engine keeps them (a keep
//! at a chunk end, the small state captured, the blocks staged between rounds from host memory standing in for the
//! device, a drain), at the tier's DeepSeek geometry by default. tf-spill-fill (fill_main.zig) is its command line.
const std = @import("std");
const format = @import("format.zig");
const fsio = @import("fsio.zig");
const slots = @import("slots.zig");
const store = @import("store.zig");
const Allocator = std.mem.Allocator;
const Store = store.Store;
const Digest = format.Digest;

/// The geometry of the store's points (the DeepSeek tier's by default).
pub const Geometry = struct {
    block_tokens: u32 = 2048,
    block_bytes: u64 = 1_822_720,
    small_bytes: u64 = 6_684_672,
    max_tokens: u32 = 1 << 20,
};

/// What is kept: `sessions` conversations of turns of `turn_tokens` (whole blocks) up to `tokens`; every `fork_every`th
/// one (0: none) also branches after its first turn and abandons the branch: its newest point is on its main path.
pub const Plan = struct {
    sessions: u32,
    tokens: u32,
    turn_tokens: u32,
    fork_every: u32 = 0,
    seed: u64 = 1,
    /// Conversations are numbered from here (a later run adds more without reusing ids).
    first: u32 = 0,
};

/// The namespace the synthetic store lives in: its own compat digest (never an engine's).
pub fn compat(geo: Geometry) Digest {
    var common: Digest = undefined;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("tf-spill-fill: a synthetic store, never an engine's");
    h.final(&common);
    return format.compatDigest(common, .replicated, geo.block_tokens, geo.block_bytes);
}

/// Host staging for a store at `geo`: `bytes` split by slots.plan, gpa-owned.
pub const Staging = struct {
    bufs: [4][]align(slots.alignment) u8 = @splat(&.{}),
    lists: [4][1]slots.Buffer = undefined,

    pub fn init(st: *Staging, gpa: Allocator, geo: Geometry, bytes: u64) !slots.Staging {
        const p = try slots.plan(bytes, geo.block_bytes, geo.small_bytes, 4, 2);
        const n = [4]usize{ p.write * p.block_slot, p.read * p.block_slot, p.small * p.small_slot, p.restore * p.small_slot };
        errdefer st.deinit(gpa);
        for (0..4) |i| {
            st.bufs[i] = try gpa.alignedAlloc(u8, .fromByteUnits(slots.alignment), n[i]);
            st.lists[i] = .{st.bufs[i]};
        }
        return .{ .write = &st.lists[0], .read = &st.lists[1], .small = &st.lists[2], .restore = &st.lists[3] };
    }

    pub fn deinit(st: *Staging, gpa: Allocator) void {
        for (st.bufs) |b| gpa.free(b);
        st.* = .{};
    }
};

/// Conversation `c`'s ids (deterministic from the plan's seed), gpa-owned. A fork's own ids after its first turn come
/// from `c + 0x8000_0000`.
pub fn sessionIds(gpa: Allocator, plan: Plan, c: u64) ![]u32 {
    const ids = try gpa.alloc(u32, plan.tokens);
    var prng = std.Random.DefaultPrng.init(plan.seed *% 0x9e3779b97f4a7c15 +% c);
    for (ids) |*x| x.* = prng.random().intRangeAtMost(u32, 1, 150_000);
    return ids;
}

/// The "device" rows of source `source` from token `first` (deterministic: a staged block is the same bytes whenever
/// it is asked for).
fn rows(dst: []u8, source: u64, first: u32) void {
    var prng = std.Random.DefaultPrng.init(source *% 0x9e3779b97f4a7c15 +% first);
    prng.random().bytes(dst);
}

/// The engine between rounds: every block the store asks for packed from its source and reported, until none waits.
fn pump(s: *Store, io: std.Io) void {
    var reqs: [16]store.StageReq = undefined;
    while (s.pendingStage() > 0) {
        const n = s.stage(&reqs);
        for (reqs[0..n]) |r| {
            rows(r.dst, r.source.id, r.first);
            s.staged(r.ticket, true);
        }
        if (n == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {}; // until the writer frees write slots
    }
}

pub const Filled = struct {
    /// Points queued and handed to the writer, states already kept, keeps the store refused (no room, backlog).
    queued: u64 = 0,
    stored: u64 = 0,
    refused: u64 = 0,
};

/// One keep as the engine makes it: the state queued, its small state captured, its blocks staged, retried while the
/// store is out of staging or past its backlog (the writer catches up).
fn keepOne(s: *Store, io: std.Io, geo: Geometry, ids: []const u32, lineage: format.Lineage, source: u64, out: *Filled) void {
    const sections = [_]store.SectionSpec{.{ .name = "state", .len = geo.small_bytes }};
    for (0..1000) |_| {
        const r = s.keep(.{ .keyed = ids, .lineage = lineage, .sections = &sections, .source = .{ .id = source, .generation = 1 } });
        switch (r) {
            .queued => |q| {
                rows(q.small, source ^ 0x5a5a, @intCast(ids.len));
                s.captured(q.job, true);
                pump(s, io);
                out.queued += 1;
                return;
            },
            .stored => {
                out.stored += 1;
                return;
            },
            .no_staging, .backlog => {
                pump(s, io);
                _ = s.drain(std.time.ns_per_s);
            },
            .short, .invalid, .no_space => {
                out.refused += 1;
                return;
            },
        }
    }
    out.refused += 1;
}

/// Keeps every conversation of `plan`, in order, turn by turn, then drains: the store's write path end to end.
pub fn fill(gpa: Allocator, io: std.Io, s: *Store, geo: Geometry, plan: Plan) !Filled {
    std.debug.assert(plan.turn_tokens > 0 and plan.turn_tokens % geo.block_tokens == 0 and plan.tokens <= geo.max_tokens);
    var out: Filled = .{};
    const boot: [16]u8 = @splat(0xf1);
    for (0..plan.sessions) |k| {
        const c: u64 = plan.first + k;
        const ids = try sessionIds(gpa, plan, c);
        defer gpa.free(ids);
        const lineage = format.Lineage.fresh(boot, c);
        var at: u32 = plan.turn_tokens;
        while (at <= plan.tokens) : (at += plan.turn_tokens) {
            keepOne(s, io, geo, ids[0..at], lineage, c, &out);
            if (at == plan.turn_tokens and plan.fork_every > 0 and k % plan.fork_every == 0 and plan.tokens >= 2 * plan.turn_tokens) {
                // a branch from the first turn with ids of its own, then abandoned
                const other = try sessionIds(gpa, plan, c + 0x8000_0000);
                defer gpa.free(other);
                const ids2 = try gpa.alloc(u32, 2 * plan.turn_tokens);
                defer gpa.free(ids2);
                @memcpy(ids2[0..plan.turn_tokens], ids[0..plan.turn_tokens]);
                @memcpy(ids2[plan.turn_tokens..], other[0..plan.turn_tokens]);
                var forked = try lineage.fork(gpa, @splat(0xf2), plan.turn_tokens);
                defer forked.deinit(gpa);
                keepOne(s, io, geo, ids2, forked, c + 0x8000_0000, &out);
            }
        }
    }
    pump(s, io);
    _ = s.drain(60 * std.time.ns_per_s);
    return out;
}

/// The store's config for a synthetic store under `dir` at `geo`, with direct I/O as the engine's; keep_builds is
/// maxed so other namespaces in the root are never pruned.
pub fn config(dir: [:0]const u8, geo: Geometry, cap_bytes: u64, floor_bytes: u64, turns: u32, staging: slots.Staging) store.Config {
    return .{
        .root = dir,
        .rank = 0,
        .compat = compat(geo),
        .limits = .{ .max_tokens = geo.max_tokens, .block_tokens = geo.block_tokens, .block_bytes = geo.block_bytes, .max_small_bytes = geo.small_bytes },
        .cap_bytes = cap_bytes,
        .floor_bytes = floor_bytes,
        .turns = turns,
        .min_tokens = 1,
        .staging = staging,
        .keep_builds = std.math.maxInt(u32),
    };
}

/// Whether `dir` holds a store namespace other than the synthetic one (an engine's: the filler never runs beside it).
pub fn foreignNamespace(gpa: Allocator, dir: [:0]const u8, geo: Geometry) !bool {
    const fd = try fsio.openDir(dir);
    defer _ = std.c.close(fd);
    var names = try fsio.list(gpa, fd);
    defer fsio.freeList(gpa, &names);
    const mine = format.hex(&compat(geo));
    for (names.items) |n| {
        if (!std.mem.startsWith(u8, n, "native-spill-")) continue;
        if (!std.mem.eql(u8, n["native-spill-".len..], mine[0..16])) return true;
    }
    return false;
}

// -- tests ---------------------------------------------------------------------------------------------------------
const testing = std.testing;

const small_geo: Geometry = .{ .block_tokens = 16, .block_bytes = 4096, .small_bytes = 8192, .max_tokens = 1024 };

fn testStore(gpa: Allocator, root: [:0]const u8, staging: *Staging, cap: u64) !*Store {
    const st = try staging.init(gpa, small_geo, 1 << 20);
    return Store.open(gpa, testing.io, config(root, small_geo, cap, 0, 4, st));
}

test "fill: 20 conversations of 4 turns and 4 forks are kept, recovered whole, and evicted in last-use order" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try fsio.testRoot(&rb, "fill");
    const dfd = try fsio.freshDir(gpa, root);
    _ = std.c.close(dfd);
    defer fsio.dropDir(gpa, root);
    var staging: Staging = .{};
    defer staging.deinit(gpa);
    const plan: Plan = .{ .sessions = 20, .tokens = 64, .turn_tokens = 16, .fork_every = 5, .seed = 7 };
    var s = try testStore(gpa, root, &staging, 64 << 20);
    const got = try fill(gpa, testing.io, s, small_geo, plan);
    try testing.expectEqual(Filled{ .queued = 84, .stored = 0, .refused = 0 }, got);
    try testing.expect(s.drain(5 * std.time.ns_per_s).done);
    // the order room would be made in, by the store's own last-use clock (after a reopen it comes from the manifests'
    // times, which can tie on a coarse-timestamp file system): the four forks, then conversation 0 whole, longest
    // point first, and so on to conversation 19
    const order = try s.evictionOrder();
    defer gpa.free(order);
    try testing.expectEqual(@as(usize, 84), order.len);
    for (order[0..4]) |v| try testing.expect(v.why == .pruned_forks);
    const keyOf = struct {
        fn f(g: Allocator, p: Plan, c: u64, at: u32) !Digest {
            const ids = try sessionIds(g, p, c);
            defer g.free(ids);
            return format.pointKey(ids[0..at], small_geo.block_tokens, &.{});
        }
    }.f;
    for (0..4) |t| {
        try testing.expect(order[4 + t].why == .pruned_conversations);
        try testing.expectEqualSlices(u8, &(try keyOf(gpa, plan, 0, @intCast(64 - 16 * t))), &order[4 + t].key);
        try testing.expectEqualSlices(u8, &(try keyOf(gpa, plan, 19, @intCast(64 - 16 * t))), &order[80 + t].key);
    }
    s.close();
    staging.deinit(gpa);
    s = try testStore(gpa, root, &staging, 64 << 20);
    defer s.close();
    const ct = s.counts();
    try testing.expectEqual(@as(u64, 84), ct.recovered_points);
    try testing.expectEqual(@as(u64, 0), ct.recovered_dropped);
}

test "fill: a keep at the cap evicts the least recently used conversation, not a newer one" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try fsio.testRoot(&rb, "fillcap");
    const dfd = try fsio.freshDir(gpa, root);
    _ = std.c.close(dfd);
    defer fsio.dropDir(gpa, root);
    var staging: Staging = .{};
    defer staging.deinit(gpa);
    const plan: Plan = .{ .sessions = 10, .tokens = 64, .turn_tokens = 16, .seed = 8 };
    const s = try testStore(gpa, root, &staging, 64 << 20);
    defer s.close();
    _ = try fill(gpa, testing.io, s, small_geo, plan);
    const full = s.counts().stored_bytes;
    // the store at its cap, with 2 KiB of slack: a keep makes room by its point's estimate, a little under the file
    // it writes, so with no slack the writer's timing could decide whether one point of the next conversation goes too
    s.cfg.cap_bytes = full + 2048;
    const more = try fill(gpa, testing.io, s, small_geo, .{ .sessions = 1, .tokens = 64, .turn_tokens = 16, .seed = 8, .first = 10 });
    try testing.expectEqual(@as(u64, 4), more.queued);
    const ct = s.counts();
    try testing.expect(ct.pruned_conversations > 0 and ct.stored_bytes <= full + 2048);
    for (0..10) |c| {
        const ids = try sessionIds(gpa, plan, c);
        defer gpa.free(ids);
        const present = s.presence(format.pointKey(ids, small_geo.block_tokens, &.{}));
        if (c == 0) try testing.expect(!present) else try testing.expect(present);
    }
}

test "fill: refuses a folder that holds another namespace (an engine's store is never pruned or filled)" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try fsio.testRoot(&rb, "fillforeign");
    const dfd = try fsio.freshDir(gpa, root);
    defer {
        _ = std.c.close(dfd);
        fsio.dropDir(gpa, root);
    }
    try testing.expect(!try foreignNamespace(gpa, root, small_geo));
    const mine = format.hex(&compat(small_geo));
    var nb: [64]u8 = undefined;
    try fsio.ensureDirAt(dfd, try std.fmt.bufPrintSentinel(&nb, "native-spill-{s}", .{mine[0..16]}, 0), null);
    try testing.expect(!try foreignNamespace(gpa, root, small_geo));
    try fsio.ensureDirAt(dfd, "native-spill-0123456789abcdef", null);
    try testing.expect(try foreignNamespace(gpa, root, small_geo));
}
