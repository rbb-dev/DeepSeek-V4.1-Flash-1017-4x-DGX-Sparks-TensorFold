//! spill_tier's host tests on one rank: the Tier over a real core/spill store in a scratch directory
//! (TF_SPILL_TEST_DIR), the packing over a byte-array device whose copies complete after a set number of event polls.
const std = @import("std");
const core = @import("core");
const tier = @import("spill_tier.zig");
const state = @import("spill_state.zig");

const spill = core.spill;
const format = spill.format;
const pc = tier.prompt_cache;
const testing = std.testing;
const io = testing.io;
const gpa = testing.allocator;

pub const bt = 16; // block tokens
pub const cap = 16 * bt; // pool positions
const max_snaps = 16;

fn testLayout() !state.StateLayout {
    const shapes = [_]state.LayerShape{ .{ .ratio = 2, .kv = true, .idx = true, .raw = true }, .{ .ratio = 1, .kv = true, .idx = true } };
    return state.StateLayout.init(&shapes, .{ .block_tokens = bt, .window = 8, .ring_extra = 4, .raw_rows = 4, .head_dim = 64, .index_head_dim = 64 }, .replicated);
}

/// The owner-sharded twin: two parts of half the pool, each rank's codes and scales its own part's only.
const part_rows = cap / 2;
fn shardedLayout() !state.StateLayout {
    const shapes = [_]state.LayerShape{ .{ .ratio = 2, .kv = true, .idx = true, .raw = true }, .{ .ratio = 1, .kv = true, .idx = true } };
    const l = try state.StateLayout.init(&shapes, .{ .block_tokens = bt, .window = 8, .ring_extra = 4, .raw_rows = 4, .head_dim = 64, .index_head_dim = 64 }, .owner_sharded);
    return l.withParts(part_rows);
}

/// The device: block planes over the pool, then Snap buffers; copies immediate, events complete after `delay` polls.
pub const Sim = struct {
    mem: []u8,
    layout: state.StateLayout,
    planes: [state.max_planes]u64 = @splat(0),
    snap_base: u64 = 0,
    snap_used: [max_snaps]bool = @splat(false),
    delay: u32 = 0,
    pending: [1024]u32 = @splat(0),
    waits: u32 = 0,
    rank: u8,

    const origin: u64 = 4096;

    fn init(rank: u8) !*Sim {
        return initWith(rank, try testLayout());
    }

    /// `layout` owner_sharded: an owner_only plane holds this rank's part's rows only (part_rows / ratio + 2), from its
    /// part's first position (prompt.Caches.initShard).
    fn initWith(rank: u8, layout: state.StateLayout) !*Sim {
        const s = try gpa.create(Sim);
        s.* = .{ .mem = &.{}, .layout = layout, .rank = rank };
        var at: u64 = origin;
        for (s.layout.blockPlanes(), 0..) |p, i| {
            s.planes[i] = at;
            at += (@as(usize, if (s.sharded(p)) part_rows else cap) / p.ratio + 2) * p.row;
        }
        s.snap_base = at;
        at += max_snaps * s.layout.small_bytes;
        s.mem = try gpa.alloc(u8, at - origin);
        for (s.mem, 0..) |*b, i| b.* = @truncate((i *% 2654435761 +% @as(usize, rank) *% 977) >> 5);
        return s;
    }

    pub fn deinit(s: *Sim) void {
        gpa.free(s.mem);
        gpa.destroy(s);
    }

    pub fn bytes(s: *Sim, a: u64, len: usize) []u8 {
        return s.mem[a - origin ..][0..len];
    }

    pub fn allocSnap(s: *Sim) !u64 {
        for (&s.snap_used, 0..) |*u, i| if (!u.*) {
            u.* = true;
            return s.snap_base + i * s.layout.small_bytes;
        };
        return error.NoSnapRoom;
    }

    pub fn freeSnap(s: *Sim, buf: u64) void {
        s.snap_used[(buf - s.snap_base) / s.layout.small_bytes] = false;
    }

    fn sharded(s: *const Sim, p: state.Plane) bool {
        return p.owner_only and s.layout.owner_layout == .owner_sharded;
    }

    /// Whether this rank holds plane i's rows at pool position `pos` (an owner_only plane: its own part's only).
    fn holds(s: *const Sim, i: usize, pos: usize) bool {
        const p = s.layout.blockPlanes()[i];
        return !s.sharded(p) or pos / part_rows == s.rank;
    }

    /// A block plane's rows of pool positions [lo, hi) (an owner_only plane's: its part's local rows).
    pub fn rows(s: *Sim, i: usize, lo: usize, hi: usize) []u8 {
        const p = s.layout.blockPlanes()[i];
        const at = if (s.sharded(p)) lo - lo / part_rows * part_rows else lo;
        return s.bytes(s.planes[i] + at / p.ratio * p.row, (hi - lo) / p.ratio * p.row);
    }

    pub fn sameRows(s: *Sim, a: usize, b: usize, n: usize) bool {
        for (0..s.layout.n_block) |i| if (s.holds(i, a) and s.holds(i, b)) {
            if (!std.mem.eql(u8, s.rows(i, a, a + n), s.rows(i, b, b + n))) return false;
        };
        return true;
    }

    fn clobber(s: *Sim, lo: usize, hi: usize) void {
        for (0..s.layout.n_block) |i| if (s.holds(i, lo)) @memset(s.rows(i, lo, hi), 0xA5);
    }

    fn cleared(s: *Sim, lo: usize, hi: usize) bool {
        for (s.layout.blockPlanes(), 0..) |p, i| if (s.holds(i, lo)) for (s.rows(i, lo, hi)) |b| if (b != p.sentinel) return false;
        return true;
    }
};

pub const Dev = struct {
    sim: *Sim,
    pub const Event = u32;

    pub fn blockPlane(d: Dev, p: state.Plane) u64 {
        for (d.sim.layout.blockPlanes(), 0..) |q, i| if (q.kind == p.kind and q.layer == p.layer) return d.sim.planes[i];
        return 0;
    }
    pub fn slotPlane(d: Dev, p: state.Plane, slot: usize) u64 {
        _ = .{ p, slot };
        return d.sim.snap_base;
    }
    pub fn capacity(_: Dev) usize {
        return cap;
    }
    pub fn slots(_: Dev) usize {
        return 4;
    }
    pub fn d2h(d: Dev, dst: []u8, src: u64) !void {
        @memcpy(dst, d.sim.bytes(src, dst.len));
    }
    pub fn h2d(d: Dev, dst: u64, src: []const u8) !void {
        @memcpy(d.sim.bytes(dst, src.len), src);
    }
    pub fn fill(d: Dev, dst: u64, v: u8, len: usize) !void {
        @memset(d.sim.bytes(dst, len), v);
    }
    pub fn record(d: Dev, ev: u32) !void {
        d.sim.pending[ev] = d.sim.delay;
    }
    pub fn done(d: Dev, ev: u32) bool {
        if (d.sim.pending[ev] == 0) return true;
        d.sim.pending[ev] -= 1;
        return false;
    }
    pub fn wait(d: Dev, ev: u32) void {
        if (d.sim.pending[ev] > 0) d.sim.waits += 1;
        d.sim.pending[ev] = 0;
    }
};

pub const L = tier.Leader(Dev);

pub fn bigFree(_: ?*anyopaque) u64 {
    return 1 << 40;
}

/// One rank's pieces: its device, staging, events and a scratch store directory.
pub const Rank = struct {
    sim: *Sim,
    staging: []align(4096) u8,
    events: []u32,
    path: [256]u8 = undefined,
    root: [:0]const u8 = undefined,
    settings: tier.Settings,

    pub fn init(rank: u8, tag: []const u8, staging_bytes: u64) !*Rank {
        return initWith(rank, tag, staging_bytes, try testLayout());
    }

    pub fn initWith(rank: u8, tag: []const u8, staging_bytes: u64, layout: state.StateLayout) !*Rank {
        const r = try gpa.create(Rank);
        r.* = .{ .sim = try Sim.initWith(rank, layout), .staging = &.{}, .events = &.{}, .settings = undefined };
        var tb: [64]u8 = undefined;
        r.root = try spill.fsio.testRoot(&r.path, try std.fmt.bufPrint(&tb, "tier-{s}-r{d}", .{ tag, rank }));
        const fd = try spill.fsio.freshDir(gpa, r.root);
        _ = std.c.close(fd);
        r.settings = .{ .dir = r.root, .gib = 1, .min_free_gib = 0, .min_tokens = bt, .staging_bytes = if (staging_bytes > 0) staging_bytes else 1 << 20, .readers = 2 };
        const plan = try tier.stagingPlan(r.settings, &r.sim.layout);
        r.staging = try gpa.alignedAlloc(u8, .fromByteUnits(4096), plan.bytes());
        r.events = try gpa.alloc(u32, tier.eventCount(plan));
        for (r.events, 0..) |*e, i| e.* = @intCast(i);
        return r;
    }

    pub fn deinit(r: *Rank) void {
        spill.fsio.dropDir(gpa, r.root);
        gpa.free(r.staging);
        gpa.free(r.events);
        r.sim.deinit();
        gpa.destroy(r);
    }

    fn compat(r: *Rank) format.Digest {
        return tier.storeCompat(test_name, &r.sim.layout);
    }

    pub fn side(r: *Rank, rank: u16) !*L.S {
        const gens = try state.Generations.init(gpa, cap, bt);
        return L.S.create(gpa, io, .{ .sim = r.sim }, r.sim.layout, gens, r.settings, rank, r.compat(), cap, r.staging, r.events, .{ .free = bigFree });
    }

    /// Flips a byte in every block file of this rank's store (damage CRC catches on the next read).
    pub fn corrupt(r: *Rank) !void {
        var root = try std.Io.Dir.cwd().openDir(io, r.root, .{ .iterate = true });
        defer root.close(io);
        var it = root.iterate();
        while (try it.next(io)) |ns| {
            if (ns.kind != .directory) continue;
            var sub_buf: [256]u8 = undefined;
            const sub = try std.fmt.bufPrint(&sub_buf, "{s}/rank{d}/blocks", .{ ns.name, r.sim.rank });
            var bd = root.openDir(io, sub, .{ .iterate = true }) catch continue;
            defer bd.close(io);
            var bit = bd.iterate();
            while (try bit.next(io)) |b| {
                if (b.kind != .file) continue;
                var f = try bd.openFile(io, b.name, .{ .mode = .read_write });
                defer f.close(io);
                var one: [1]u8 = undefined;
                _ = try f.readPositionalAll(io, &one, 7);
                one[0] ^= 0xff;
                try f.writePositionalAll(io, &one, 7);
            }
        }
    }
};

/// A kept state as the test's family holds it (lanes.zig's Snap stands in).
pub const Snap = struct { id: u64, base: usize, at: usize, buf: u64, slot: u32 = 0, installed: bool = false };

/// Rank 0's family: restores get a fresh Snap at the next free base the test sets.
pub const Fam0 = struct {
    sim: *Sim,
    next_id: u64 = 1000,
    next_base: usize = 8 * bt,
    installed: u32 = 0,
    discarded: u32 = 0,
    reserved: u32 = 0,
    last_room: usize = 0,
    land_pending: u32 = 0,
    land_final: tier.LandStep = .landed,
    land_calls: u32 = 0,
    probe_says: tier.ProbeVerdict = .same,
    probed: std.ArrayList(u64) = .empty,
    live: std.ArrayList(*Snap) = .empty,

    pub fn family(f: *Fam0) tier.Family {
        return .{ .ptr = f, .reserve = reserve, .install = install, .discard = discard, .info = info };
    }
    /// With a scripted probe: each probe returns `probe_says`, and the states probed are recorded.
    fn probing(f: *Fam0) tier.Family {
        var fam = f.family();
        fam.probe = probeFn;
        return fam;
    }
    fn probeFn(p: *anyopaque, saved: *anyopaque) tier.ProbeVerdict {
        const f = of(p);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        f.probed.append(gpa, sn.id) catch {};
        return f.probe_says;
    }
    /// With a scripted landing: `land_pending` polls pending, then `land_final`.
    fn landing(f: *Fam0) tier.Family {
        var fam = f.family();
        fam.land = landFn;
        return fam;
    }
    fn landFn(p: *anyopaque, saved: *anyopaque, _: u64) tier.LandStep {
        const f = of(p);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (sn.installed) return .failed; // (never: install comes after the landing)
        f.land_calls += 1;
        if (f.land_calls <= f.land_pending) return .pending;
        return f.land_final;
    }
    fn of(p: *anyopaque) *Fam0 {
        return @ptrCast(@alignCast(p));
    }
    fn reserve(p: *anyopaque, at: usize, room: usize) anyerror!tier.Dest {
        const f = of(p);
        f.reserved += 1;
        f.last_room = room;
        const sn = try gpa.create(Snap);
        sn.* = .{ .id = f.next_id, .base = f.next_base, .at = at, .buf = try f.sim.allocSnap() };
        f.next_id += 1;
        try f.live.append(gpa, sn);
        return .{ .saved = sn, .id = sn.id, .base = sn.base, .buf = sn.buf };
    }
    fn install(p: *anyopaque, saved: *anyopaque) void {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        sn.installed = true;
        of(p).installed += 1;
    }
    fn discard(p: *anyopaque, saved: *anyopaque) void {
        const f = of(p);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        f.discarded += 1;
        f.sim.freeSnap(sn.buf);
        for (f.live.items, 0..) |x, i| if (x == sn) {
            _ = f.live.swapRemove(i);
            break;
        };
        gpa.destroy(sn);
    }
    fn info(_: *anyopaque, saved: *anyopaque) tier.SnapInfo {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return .{ .id = sn.id, .base = sn.base, .at = sn.at, .buf = sn.buf, .slot = sn.slot };
    }
    pub fn deinit(f: *Fam0) void {
        for (f.live.items) |sn| gpa.destroy(sn);
        f.live.deinit(gpa);
        f.probed.deinit(gpa);
    }
};

pub fn testIds(out: []u32, seed: u32) void {
    for (out, 0..) |*x, i| x.* = seed *% 7919 +% @as(u32, @intCast(i)) *% 31 +% 3;
}

/// The tests' store name (rank 0's spill_identity.name in the engine).
pub const test_name: format.Digest = @splat(1);

/// Pumps until `n` writes were published, or ten seconds passed (the bound only matters when a write never comes).
pub fn pumpUntil(ld: *L, n: u64) !void {
    const t = ld.tier();
    var k: usize = 0;
    while (k < 10_000) : (k += 1) {
        t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
        if (ld.side.store.counts().committed >= n) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.TestTimedOut;
}

pub fn pollUntilDone(t: pc.Tier, job: u64) pc.Poll {
    var k: usize = 0;
    while (k < 5000) : (k += 1) {
        const p = t.vtable.poll(t.ptr, job, std.time.ns_per_ms);
        if (p != .pending) return p;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return .{ .failed = error.TestTimedOut };
}

/// Staging back to none (reads still in flight on a reader give their slots back when they end).
fn stagingDrains(st: *spill.Store) !void {
    var k: usize = 0;
    while (k < 10_000) : (k += 1) {
        if (st.counts().staging_bytes == 0) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.StagingHeld;
}

pub fn want(ids: []const u32) pc.Want {
    return .{ .key = ids, .beyond = 0, .max_at = @intCast(ids.len), .grid = bt };
}

/// A kept state at `base` holding `at` positions: its small state in a fresh buffer (the device's own bytes there).
pub fn keptSnap(sim: *Sim, id: u64, base: usize, at: usize) !*Snap {
    const sn = try gpa.create(Snap);
    sn.* = .{ .id = id, .base = base, .at = at, .buf = try sim.allocSnap() };
    return sn;
}

pub fn keep(t: pc.Tier, sn: *Snap, ids: []const u32, lineage: []const pc.Fork) ?u64 {
    return t.vtable.keep(t.ptr, &.{ .saved = sn, .key = ids, .at = @intCast(sn.at), .bytes = 0, .group = 9, .lineage = lineage });
}

test "one rank: a kept state written at keep, found, restored byte for byte into a fresh extent" {
    const r = try Rank.init(0, "one", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt + 5]u32 = undefined;
    testIds(&ids, 1);
    const sn = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(sn);
    const lin = [_]pc.Fork{.{ .name = 5, .at = 0 }};
    const id = keep(t, sn, &ids, &lin) orelse return error.NotKept;
    try testing.expect(t.vtable.busy.?(t.ptr));
    try pumpUntil(ld, 1);
    // found (the store's index, no I/O), at its own length, as the prompt cache asks
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    try testing.expectEqual(id, p.id);
    try testing.expectEqual(@as(u32, 4 * bt), p.at);
    try testing.expectEqual(@as(u64, 9), p.group);
    // a prompt that differs inside the point does not find it
    var other = ids;
    other[3] +%= 1;
    try testing.expect(t.vtable.lookup(t.ptr, &want(&other)) == null);
    // restore: the rows land in the family's fresh extent, the small state in its fresh buffer
    r.sim.delay = 2;
    const job = try t.vtable.begin(t.ptr, &p);
    const done = pollUntilDone(t, job);
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    try testing.expect(got.installed);
    try testing.expect(r.sim.sameRows(0, got.base, 4 * bt));
    try testing.expectEqualSlices(u8, r.sim.bytes(sn.buf, r.sim.layout.small_bytes), r.sim.bytes(got.buf, r.sim.layout.small_bytes));
    // the point's lineage comes back as names that map to the same stored lineage
    try testing.expectEqual(@as(usize, 1), done.ready.lineage.len);
    try testing.expectEqualSlices(u8, &ld.name16(5), &ld.name16(done.ready.lineage[0].name));
    try testing.expectEqual(@as(u32, 1), fam.installed);
    try testing.expectEqual(@as(u64, 1), ld.side.cached.writes_committed);
    try stagingDrains(ld.side.store);
    fam.sim.freeSnap(got.buf);
}

test "one rank: what the store did on its own reaches the counters a second later with no service (a point it dropped)" {
    const r = try Rank.init(0, "counts", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [2 * bt + 5]u32 = undefined;
    testIds(&ids, 7);
    const sn = try keptSnap(r.sim, 1, 0, 2 * bt);
    defer gpa.destroy(sn);
    _ = keep(t, sn, &ids, &.{}) orelse return error.NotKept;
    try pumpUntil(ld, 1);
    ld.side.refresh();
    var out: [1]pc.TierCounts = undefined;
    try testing.expectEqual(@as(usize, 1), t.vtable.counts(t.ptr, &out));
    try testing.expectEqual(@as(u64, 1), out[0].points);
    // the store drops the point with no service after it (as its writer evicts on its own thread)
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    t.vtable.forget(t.ptr, p.id);
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().points);
    // within the second: the copy as it was (a status pass takes no store lock)
    _ = t.vtable.counts(t.ptr, &out);
    try testing.expectEqual(@as(u64, 1), out[0].points);
    // a second later: the store's own count
    ld.side.refreshed_ns -|= std.time.ns_per_s;
    _ = t.vtable.counts(t.ptr, &out);
    try testing.expectEqual(@as(u64, 0), out[0].points);
}

test "one rank: rows that change before they are staged void the write; a guard stages them first" {
    const r = try Rank.init(0, "stale", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 2);
    // 1. a placement takes the rows without a guard (the generation moves): the stage is refused, nothing publishes
    const a = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    ld.bump(0, 4 * bt);
    var k: usize = 0;
    while (k < 200 and t.vtable.busy.?(t.ptr)) : (k += 1) {
        t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().committed);
    try testing.expectEqual(@as(u64, 1), ld.side.store.counts().write_failed);
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids)) == null);
    // 2. the guard before the change: every block staged now, so the point publishes the bytes from before it
    testIds(&ids, 3);
    const b = try keptSnap(r.sim, 2, 4 * bt, 4 * bt);
    defer gpa.destroy(b);
    var before: [2][]u8 = undefined;
    before[0] = try gpa.dupe(u8, r.sim.rows(0, 4 * bt, 8 * bt));
    defer gpa.free(before[0]);
    _ = keep(t, b, &ids, &.{.{ .name = 2, .at = 0 }}) orelse return error.NotKept;
    ld.guardSnap(b.id, 0);
    ld.bump(4 * bt, 8 * bt);
    r.sim.clobber(4 * bt, 8 * bt);
    try pumpUntil(ld, 1);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    fam.next_base = 12 * bt;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    try testing.expectEqualSlices(u8, before[0], r.sim.rows(0, got.base, got.base + 4 * bt));
    try testing.expect(ld.side.store.counts().guard_staged > 0);
    fam.sim.freeSnap(got.buf);
}

test "one rank: a guard without staging room cancels the unpublished write" {
    // staging for 4 block slots: 2 writes, 2 reads (and 6 small slots)
    const plan_small = std.mem.alignForward(u64, (try testLayout()).small_bytes, 4096);
    const r = try Rank.init(0, "guardcancel", 6 * plan_small + 4 * 4096);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    try testing.expectEqual(@as(usize, 2), ld.side.plan.write);
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 4);
    const a = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    ld.guardSnap(a.id, 0); // four blocks left, two write slots: cancelled, not staged half
    ld.dropSnap(a.id);
    var k: usize = 0;
    while (k < 200 and t.vtable.busy.?(t.ptr)) : (k += 1) {
        t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    const c = ld.side.store.counts();
    try testing.expectEqual(@as(u64, 1), c.cancelled_guard);
    try testing.expectEqual(@as(u64, 0), c.committed);
    try stagingDrains(ld.side.store);
}

test "one rank: a damaged block fails the restore, forgets the point and clears the destination" {
    const r = try Rank.init(0, "damage", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [3 * bt]u32 = undefined;
    testIds(&ids, 5);
    const a = try keptSnap(r.sim, 1, 0, 3 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    try pumpUntil(ld, 1);
    try r.corrupt();
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expectEqual(error.SpillDamaged, done.failed);
    try testing.expectEqual(@as(u32, 1), fam.discarded);
    try testing.expect(r.sim.cleared(8 * bt, 11 * bt));
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids)) == null);
    try stagingDrains(ld.side.store);
}

test "one rank: a cancel mid-restore waits for issued copies, clears the destination and releases everything" {
    const r = try Rank.init(0, "cancel", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 6);
    const a = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    try pumpUntil(ld, 1);
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    r.sim.delay = 1000; // copies stay on the "device"
    const job = try t.vtable.begin(t.ptr, &p);
    var k: usize = 0;
    while (k < 1000 and ld.jobs.get(job).?.rx.?.copied == 0) : (k += 1) {
        try testing.expect(t.vtable.poll(t.ptr, job, std.time.ns_per_ms) == .pending);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try testing.expect(ld.jobs.get(job).?.rx.?.inflight > 0);
    const waits = r.sim.waits;
    t.vtable.cancel(t.ptr, job);
    try testing.expect(r.sim.waits > waits);
    try testing.expectEqual(@as(u32, 1), fam.discarded);
    try testing.expect(r.sim.cleared(8 * bt, 12 * bt));
    try stagingDrains(ld.side.store);
    try testing.expect(ld.side.copies.items.len == 0);
    // the point is still there (a cancel is not damage)
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids)) != null);
}

test "one rank: the shutdown flush stages and publishes every queued write" {
    const r = try Rank.init(0, "flush", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [2][4 * bt]u32 = undefined;
    var snaps: [2]*Snap = undefined;
    for (0..2) |i| {
        testIds(&ids[i], @intCast(10 + i));
        snaps[i] = try keptSnap(r.sim, i + 1, i * 4 * bt, 4 * bt);
        _ = keep(t, snaps[i], &ids[i], &.{.{ .name = i + 1, .at = 0 }}) orelse return error.NotKept;
    }
    defer for (snaps) |s| gpa.destroy(s);
    r.sim.delay = 3;
    try testing.expectEqual(@as(u64, 0), t.vtable.flush(t.ptr, 5 * std.time.ns_per_s));
    try testing.expectEqual(@as(u64, 2), ld.side.store.counts().committed);
    try testing.expect(!t.vtable.busy.?(t.ptr));
}

test "one rank: a restore past the cap is refused before anything is placed or read" {
    const r = try Rank.init(0, "toomany", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    try testing.expectEqual(@as(u32, 2), ld.max_restores);
    var ids: [3][2 * bt]u32 = undefined;
    var snaps: [3]*Snap = undefined;
    for (0..3) |i| {
        testIds(&ids[i], @intCast(60 + i));
        snaps[i] = try keptSnap(r.sim, i + 1, i * 2 * bt, 2 * bt);
        _ = keep(t, snaps[i], &ids[i], &.{.{ .name = i + 1, .at = 0 }}) orelse return error.NotKept;
    }
    defer for (snaps) |s| gpa.destroy(s);
    try pumpUntil(ld, 3);
    r.sim.delay = 1000;
    try testing.expect(!t.vtable.full.?(t.ptr)); // (the prompt cache asks before making room: robustness)
    var jobs: [2]u64 = undefined;
    for (0..2) |i| {
        fam.next_base = (8 + 4 * i) * bt;
        jobs[i] = try t.vtable.begin(t.ptr, &(t.vtable.lookup(t.ptr, &want(&ids[i])) orelse return error.NotFound));
    }
    const p3 = t.vtable.lookup(t.ptr, &want(&ids[2])) orelse return error.NotFound;
    try testing.expect(t.vtable.full.?(t.ptr)); // full before begin is asked: no room made for a third
    try testing.expectError(error.TooManyRestores, t.vtable.begin(t.ptr, &p3));
    try testing.expectEqual(@as(u32, 2), fam.reserved); // no third destination placed
    for (jobs) |j| t.vtable.cancel(t.ptr, j);
    try stagingDrains(ld.side.store);
    try testing.expect(!t.vtable.full.?(t.ptr));
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids[2])) != null); // still there for later
}

// -- pictures (9014's content keys, recorded with their full digests) ------------------------------------------

const placeholder: u32 = 129264; // the model's image token

/// A picture of a test prompt: its span and the payload its digest is made from.
const Pic = struct { start: u32, len: u32, payload: []const u8 };

/// A prompt with pictures as the server keys it: text ids from `seed`; each picture's keyed ids made from its
/// payload's SHA-256 (pc.pictureId), its raw ids the placeholder and its sequence positions negative (Lanes.seqs).
const PicPrompt = struct {
    keyed: []u32,
    raw: []u32,
    seq: []i32,
    media: []pc.Media,

    fn init(n: usize, seed: u32, pics: []const Pic) !PicPrompt {
        const p: PicPrompt = .{ .keyed = try gpa.alloc(u32, n), .raw = try gpa.alloc(u32, n), .seq = try gpa.alloc(i32, n), .media = try gpa.alloc(pc.Media, pics.len) };
        testIds(p.keyed, seed);
        @memcpy(p.raw, p.keyed);
        for (pics, p.media) |x, *m| {
            m.* = .{ .start = x.start, .len = x.len, .digest = undefined };
            std.crypto.hash.sha2.Sha256.hash(x.payload, &m.digest, .{});
            for (0..x.len) |j| {
                if (x.start + j >= n) break;
                p.keyed[x.start + j] = pc.pictureId(&m.digest, @intCast(j));
                p.raw[x.start + j] = placeholder;
            }
        }
        for (p.seq, p.raw) |*q, id| q.* = if (id == placeholder) -1 else @intCast(id);
        return p;
    }

    fn deinit(p: PicPrompt) void {
        gpa.free(p.keyed);
        gpa.free(p.raw);
        gpa.free(p.seq);
        gpa.free(p.media);
    }

    fn want(p: PicPrompt) pc.Want {
        return .{ .key = p.keyed, .beyond = 0, .max_at = @intCast(p.keyed.len), .grid = bt, .images = true, .media = p.media };
    }

    /// The point a keep at `at` writes: its key over the pictures before the cut.
    fn key(p: PicPrompt, at: u32) format.Digest {
        var n: usize = 0;
        while (n < p.media.len and p.media[n].start < at) n += 1;
        return format.pointKey(p.keyed[0..at], bt, tier.asFormat(p.media[0..n]));
    }
};

fn keepPics(t: pc.Tier, sn: *Snap, p: PicPrompt, lineage: []const pc.Fork) ?u64 {
    return t.vtable.keep(t.ptr, &.{ .saved = sn, .key = p.keyed[0..sn.at], .at = @intCast(sn.at), .bytes = 0, .group = 9, .images = true, .media = p.media, .raw = p.raw, .lineage = lineage });
}

test "pictures: the server's cache ids are the store's scheme" {
    var prng = std.Random.DefaultPrng.init(6);
    for (0..2000) |_| {
        var d: format.Digest = undefined;
        prng.random().bytes(&d);
        const j = prng.random().int(u32);
        try testing.expectEqual(format.pictureId(&d, j), pc.pictureId(&d, j));
        try testing.expect(pc.pictureId(&d, j) >= pc.picture_ids);
    }
}

test "pictures: identical pictures across a restart hit and restore byte for byte; changed or unnamed pictures miss" {
    const r = try Rank.init(0, "pics", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    // three pictures in a 6-block prompt, the point at 4 blocks: one wholly before the cut, one straddling it, one after
    const pics = [_]Pic{ .{ .start = 5, .len = 9, .payload = "red apple" }, .{ .start = 4 * bt - 6, .len = 12, .payload = "blue sky" }, .{ .start = 5 * bt, .len = 7, .payload = "green" } };
    const p = try PicPrompt.init(6 * bt, 31, &pics);
    defer p.deinit();
    const sn = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(sn);
    {
        const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
        defer ld.destroy();
        const t = ld.tier();
        const id = keepPics(t, sn, p, &.{.{ .name = 5, .at = 0 }}) orelse return error.NotKept;
        try testing.expectEqual(tier.pointId(p.key(4 * bt)), id);
        try testing.expectEqual(@as(u64, 1), ld.pictures_kept);
        try pumpUntil(ld, 1);
        try testing.expect(t.vtable.lookup(t.ptr, &p.want()) != null);
    }
    // the process restarts: a new tier over the same store directory recovers the point with its pictures
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(4));
    defer ld.destroy();
    const t = ld.tier();
    try testing.expectEqual(@as(u64, 1), ld.side.store.counts().recovered_points);
    const hit = t.vtable.lookup(t.ptr, &p.want()) orelse return error.NotFound;
    try testing.expectEqual(@as(u32, 4 * bt), hit.at);
    r.sim.delay = 1;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &hit));
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    try testing.expect(r.sim.sameRows(0, got.base, 4 * bt));
    try testing.expectEqualSlices(u8, r.sim.bytes(sn.buf, r.sim.layout.small_bytes), r.sim.bytes(got.buf, r.sim.layout.small_bytes));
    r.sim.freeSnap(got.buf);
    try stagingDrains(ld.side.store);

    // the prompt's next turn (more text, another picture far past the cut) finds the same point
    const more = [_]Pic{ pics[0], pics[1], pics[2], .{ .start = 7 * bt, .len = 5, .payload = "later" } };
    const p2 = try PicPrompt.init(8 * bt, 31, &more);
    defer p2.deinit();
    try testing.expectEqual(hit.id, (t.vtable.lookup(t.ptr, &p2.want()) orelse return error.NotFound).id);
    // another picture in the same positions (its own ids), or the same ids under another payload's full digest
    // (only bytes past the 8 that seed the ids differ): no hit
    const changed = [_]Pic{ .{ .start = 5, .len = 9, .payload = "red apples" }, pics[1], pics[2] };
    const p3 = try PicPrompt.init(6 * bt, 31, &changed);
    defer p3.deinit();
    try testing.expect(t.vtable.lookup(t.ptr, &p3.want()) == null);
    var forged = p.want();
    const fm = try gpa.dupe(pc.Media, p.media);
    defer gpa.free(fm);
    fm[1].digest[31] ^= 1;
    forged.media = fm;
    try testing.expect(t.vtable.lookup(t.ptr, &forged) == null);
    // the straddling picture's span another length (its positions past the cut changed): no hit
    fm[1] = p.media[1];
    fm[1].len -= 1;
    try testing.expect(t.vtable.lookup(t.ptr, &forged) == null);
    // missing identities force a miss: no media at all, or one picture not named
    var bare = p.want();
    bare.media = &.{};
    try testing.expect(t.vtable.lookup(t.ptr, &bare) == null);
    var partial = p.want();
    partial.media = p.media[1..];
    try testing.expect(t.vtable.lookup(t.ptr, &partial) == null);
    try testing.expect(ld.pictures_refused >= 2);
}

test "pictures: a keep whose ids disagree with its pictures writes nothing; text before every picture is a text point; image cache off" {
    const r = try Rank.init(0, "picbad", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    const pics = [_]Pic{.{ .start = 2 * bt + 3, .len = 6, .payload = "cat" }};
    const p = try PicPrompt.init(4 * bt, 41, &pics);
    defer p.deinit();
    const sn = try keptSnap(r.sim, 1, 0, 3 * bt);
    defer gpa.destroy(sn);
    // no identities: nothing kept (the keyed ids hold a picture the tier cannot name)
    try testing.expect(t.vtable.keep(t.ptr, &.{ .saved = sn, .key = p.keyed[0..sn.at], .at = 3 * bt, .bytes = 0, .images = true, .raw = p.raw }) == null);
    // raw ids that are not the placeholder at the picture, or are missing
    const raw = try gpa.dupe(u32, p.raw);
    defer gpa.free(raw);
    raw[2 * bt + 4] = 7;
    try testing.expect(t.vtable.keep(t.ptr, &.{ .saved = sn, .key = p.keyed[0..sn.at], .at = 3 * bt, .bytes = 0, .images = true, .media = p.media, .raw = raw }) == null);
    try testing.expect(t.vtable.keep(t.ptr, &.{ .saved = sn, .key = p.keyed[0..sn.at], .at = 3 * bt, .bytes = 0, .images = true, .media = p.media }) == null);
    // keyed ids that do not derive from the named digest
    const keyed = try gpa.dupe(u32, p.keyed);
    defer gpa.free(keyed);
    keyed[2 * bt + 5] ^= 1;
    try testing.expect(t.vtable.keep(t.ptr, &.{ .saved = sn, .key = keyed[0..sn.at], .at = 3 * bt, .bytes = 0, .images = true, .media = p.media, .raw = p.raw }) == null);
    try testing.expectEqual(@as(u64, 4), ld.pictures_refused);
    {
        // the one store's counts carry them (TierCounts on /metrics, rank 0's)
        var out: [1]pc.TierCounts = undefined;
        try testing.expectEqual(@as(usize, 1), t.vtable.counts(t.ptr, &out));
        try testing.expectEqual(@as(u64, 4), out[0].pictures_refused);
        try testing.expectEqual(ld.pictures_kept, out[0].pictures_kept);
    }
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().queued);
    // with image cache off nothing is kept for a prompt with pictures, nor found
    ld.image_cache = false;
    try testing.expect(keepPics(t, sn, p, &.{.{ .name = 1, .at = 0 }}) == null);
    ld.image_cache = true;
    // a cut no picture reaches: a text point (no media, no raw ids), found by the text prompt with no pictures too
    const sn2 = try keptSnap(r.sim, 2, 4 * bt, 2 * bt);
    defer gpa.destroy(sn2);
    _ = keepPics(t, sn2, p, &.{.{ .name = 2, .at = 0 }}) orelse return error.NotKept;
    try testing.expectEqual(@as(u64, 0), ld.pictures_kept);
    try pumpUntil(ld, 1);
    try testing.expect(ld.side.store.presence(format.pointKey(p.keyed[0 .. 2 * bt], bt, &.{})));
    try testing.expect(t.vtable.lookup(t.ptr, &want(p.keyed[0 .. 2 * bt + 3])) != null);
    try testing.expect(t.vtable.lookup(t.ptr, &p.want()) != null); // (the picture starts past its cut)
    ld.image_cache = false;
    try testing.expect(t.vtable.lookup(t.ptr, &p.want()) == null); // pictures stay out of the tier
    ld.image_cache = true;
    // a picture point, then the same cut with image cache off is refused and the point not found
    const sn3 = try keptSnap(r.sim, 3, 8 * bt, 3 * bt);
    defer gpa.destroy(sn3);
    _ = keepPics(t, sn3, p, &.{.{ .name = 3, .at = 0 }}) orelse return error.NotKept;
    try pumpUntil(ld, 2);
    try testing.expectEqual(@as(u32, 3 * bt), (t.vtable.lookup(t.ptr, &p.want()) orelse return error.NotFound).at);
    ld.image_cache = false;
    try testing.expect(t.vtable.lookup(t.ptr, &p.want()) == null);
    ld.image_cache = true;
    try stagingDrains(ld.side.store);
}

// -- one store on rank 0 ---------------------------------------------------------------------------------------------

test "one rank: an owner-sharded layout is refused (one store for every rank: the pool replicated)" {
    const rep_l = try testLayout();
    const sh = try shardedLayout();
    try testing.expect(!std.mem.eql(u8, &tier.storeCompat(test_name, &rep_l), &tier.storeCompat(test_name, &sh)));
    const r = try Rank.initWith(0, "sharded", 0, try shardedLayout());
    defer r.deinit();
    var gens = try state.Generations.init(gpa, cap, bt);
    defer gens.deinit(gpa);
    try testing.expectError(error.NotReplicated, L.S.create(gpa, io, .{ .sim = r.sim }, r.sim.layout, gens, r.settings, 0, r.compat(), cap, r.staging, r.events, .{ .free = bigFree }));
}

test "one rank: another store name is another namespace; the same name reopens the same points" {
    const r = try Rank.init(0, "names", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    var ids: [2 * bt]u32 = undefined;
    testIds(&ids, 70);
    const sn = try keptSnap(r.sim, 1, 0, 2 * bt);
    defer gpa.destroy(sn);
    {
        const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
        defer ld.destroy();
        _ = keep(ld.tier(), sn, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
        try pumpUntil(ld, 1);
    }
    {
        // the same name: recovered
        const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(4));
        defer ld.destroy();
        try testing.expectEqual(@as(u64, 1), ld.side.store.counts().recovered_points);
        try testing.expect(ld.tier().vtable.lookup(ld.tier().ptr, &want(&ids)) != null);
    }
    {
        // another name (another build, model, ablation or any rank's weights): an empty namespace of its own
        const gens = try state.Generations.init(gpa, cap, bt);
        var other = test_name;
        other[0] ^= 1;
        const side = try L.S.create(gpa, io, .{ .sim = r.sim }, r.sim.layout, gens, r.settings, 0, tier.storeCompat(other, &r.sim.layout), cap, r.staging, r.events, .{ .free = bigFree });
        const ld = try L.create(gpa, side, fam.family(), r.settings, @splat(5));
        defer ld.destroy();
        try testing.expect(ld.tier().vtable.lookup(ld.tier().ptr, &want(&ids)) == null);
    }
}

test "the tier's settings come from the one parse; TF_DS_SPILL_RESTORES past the tier's bound is refused, not clamped" {
    var buf: [256]u8 = undefined;
    const Env = struct {
        pairs: []const [2][]const u8,
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            for (e.pairs) |q| if (std.mem.eql(u8, q[0], name)) return q[1];
            return null;
        }
    };
    const sp = @import("engine_api").Spill.fromEnv(Env{ .pairs = &.{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "/spill" }, .{ "TF_DS_SPILL_BACKLOG_GIB", "8" }, .{ "TF_DS_SPILL_KEEP_BUILDS", "2" } } }, &buf);
    const s = tier.Settings.fromSpill(sp);
    try testing.expectEqualStrings("/spill", s.dir);
    try testing.expectEqual(@as(f64, 256), s.gib);
    try testing.expectEqual(@as(u32, 15_000), s.flush_ms);
    try testing.expectEqual(@as(u64, 8 << 30), s.backlogBytes());
    try testing.expectEqual(@as(u32, 2), s.keep_builds);
    try testing.expectEqual(@as(u32, 2), s.restores);
    const r = try Rank.init(0, "restores", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    var bad = r.settings;
    bad.restores = tier.max_restores + 1;
    const side = try r.side(0);
    try testing.expectError(error.BadSpillSetting, L.create(gpa, side, fam.family(), bad, @splat(3)));
    side.destroy();
}

test "tier off is a flag: no keep, no lookup, no restore; the reason reported once; the store stays open" {
    const r = try Rank.init(0, "off", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [2][2 * bt]u32 = undefined;
    testIds(&ids[0], 80);
    testIds(&ids[1], 81);
    const a = try keptSnap(r.sim, 1, 0, 2 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids[0], &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    try pumpUntil(ld, 1);
    const p = t.vtable.lookup(t.ptr, &want(&ids[0])) orelse return error.NotFound;
    try testing.expect(t.vtable.off.?(t.ptr) == null);
    ld.setOff("probe: rank 2 differs");
    ld.setOff("a later reason"); // the first one stays
    try testing.expectEqualStrings("probe: rank 2 differs", t.vtable.off.?(t.ptr).?);
    try testing.expect(t.vtable.lookup(t.ptr, &want(&ids[0])) == null);
    try testing.expectError(error.SpillOff, t.vtable.begin(t.ptr, &p));
    try testing.expect(!t.vtable.full.?(t.ptr)); // never "full": nothing defers on an off tier
    const b = try keptSnap(r.sim, 2, 2 * bt, 2 * bt);
    defer gpa.destroy(b);
    try testing.expect(keep(t, b, &ids[1], &.{.{ .name = 2, .at = 0 }}) == null);
    try testing.expectEqual(@as(u64, 1), ld.side.store.counts().committed); // the store holds what it held
    try testing.expectEqual(@as(u32, 0), fam.reserved);
}

test "tier off: a write queued before the switch is still guarded and staged, so it publishes the rows from before the change" {
    const r = try Rank.init(0, "offguard", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 82);
    const before = try gpa.dupe(u8, r.sim.rows(0, 4 * bt, 8 * bt));
    defer gpa.free(before);
    const b = try keptSnap(r.sim, 2, 4 * bt, 4 * bt);
    defer gpa.destroy(b);
    {
        const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
        defer ld.destroy();
        const t = ld.tier();
        _ = keep(t, b, &ids, &.{.{ .name = 2, .at = 0 }}) orelse return error.NotKept;
        ld.setOff("probe: rank 1 differs");
        ld.guardSnap(b.id, 0); // the rows are about to change: staged now, off or not
        ld.bump(4 * bt, 8 * bt);
        r.sim.clobber(4 * bt, 8 * bt);
        try pumpUntil(ld, 1);
        try testing.expect(ld.side.store.counts().guard_staged > 0);
    }
    // the next process (its tier on) restores the rows as they were before the change
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(4));
    defer ld.destroy();
    const t = ld.tier();
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    fam.next_base = 12 * bt;
    const done = pollUntilDone(t, try t.vtable.begin(t.ptr, &p));
    try testing.expect(done == .ready);
    const got: *Snap = @ptrCast(@alignCast(done.ready.saved));
    try testing.expectEqualSlices(u8, before, r.sim.rows(0, got.base, got.base + 4 * bt));
    fam.sim.freeSnap(got.buf);
}

test "tier off: a drop still waits for its state's pending capture, and a restore in flight ends at its next poll" {
    const r = try Rank.init(0, "offdrop", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [2][2 * bt]u32 = undefined;
    testIds(&ids[0], 83);
    testIds(&ids[1], 84);
    const a = try keptSnap(r.sim, 1, 0, 2 * bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids[0], &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    try pumpUntil(ld, 1);
    const p = t.vtable.lookup(t.ptr, &want(&ids[0])) orelse return error.NotFound;
    r.sim.delay = 1000;
    const job = try t.vtable.begin(t.ptr, &p);
    const b = try keptSnap(r.sim, 2, 2 * bt, 2 * bt);
    defer gpa.destroy(b);
    _ = keep(t, b, &ids[1], &.{.{ .name = 2, .at = 0 }}) orelse return error.NotKept; // its capture stays on the "device"
    ld.setOff("probe: rank 3 differs");
    const waits = r.sim.waits;
    ld.dropSnap(b.id);
    try testing.expect(r.sim.waits > waits);
    try testing.expectEqual(error.SpillOff, t.vtable.poll(t.ptr, job, std.time.ns_per_ms).failed);
    try testing.expectEqual(@as(u32, 1), fam.discarded);
    r.sim.delay = 0;
    @memset(&r.sim.pending, 0); // the "device" catches up
    var k: usize = 0; // the dropped state's write, guarded before the drop, still stages and publishes (off or not)
    while (k < 1000 and t.vtable.busy.?(t.ptr)) : (k += 1) {
        t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try stagingDrains(ld.side.store);
}

/// A Leader over a store holding one point of `blocks` blocks, its family scripted to land as set.
const LandRig = struct {
    r: *Rank,
    fam: Fam0,
    ld: *L,
    ids: [4 * bt]u32 = undefined,
    sn: *Snap,

    fn init(rig: *LandRig, tag: []const u8) !void {
        rig.r = try Rank.init(0, tag, 0);
        rig.fam = .{ .sim = rig.r.sim };
        rig.ld = try L.create(gpa, try rig.r.side(0), rig.fam.landing(), rig.r.settings, @splat(3));
        testIds(&rig.ids, 90);
        rig.sn = try keptSnap(rig.r.sim, 1, 0, 4 * bt);
        _ = keep(rig.ld.tier(), rig.sn, &rig.ids, &.{.{ .name = 7, .at = 0 }}) orelse return error.NotKept;
        try pumpUntil(rig.ld, 1);
    }

    fn deinit(rig: *LandRig) void {
        rig.ld.destroy();
        gpa.destroy(rig.sn);
        rig.fam.deinit();
        rig.r.deinit();
    }

    fn begin(rig: *LandRig) !u64 {
        const t = rig.ld.tier();
        const p = t.vtable.lookup(t.ptr, &want(&rig.ids)) orelse return error.NotFound;
        return t.vtable.begin(t.ptr, &p);
    }
};

test "landing: a restore is ready only once every rank holds it; installed then, its lineage intact" {
    var rig: LandRig = undefined;
    try rig.init("land");
    defer rig.deinit();
    rig.fam.land_pending = 3;
    const t = rig.ld.tier();
    const job = try rig.begin();
    var polls: usize = 0;
    var p = t.vtable.poll(t.ptr, job, std.time.ns_per_ms);
    while (p == .pending and polls < 5000) : (polls += 1) {
        try testing.expectEqual(@as(u32, 0), rig.fam.installed); // not usable while it reads or lands
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        p = t.vtable.poll(t.ptr, job, std.time.ns_per_ms);
    }
    if (p == .failed) std.debug.print("failed: {s}\n", .{@errorName(p.failed)});
    try testing.expect(p == .ready);
    try testing.expectEqual(@as(u32, 4), rig.fam.land_calls);
    try testing.expectEqual(@as(u32, 1), rig.fam.installed);
    try testing.expectEqual(@as(usize, 1), p.ready.lineage.len);
    try testing.expectEqualSlices(u8, &rig.ld.name16(7), &rig.ld.name16(p.ready.lineage[0].name));
    const got: *Snap = @ptrCast(@alignCast(p.ready.saved));
    try testing.expect(rig.r.sim.sameRows(0, got.base, 4 * bt));
    // its timeline: each block's read, the read's wall, its copy to the device, its landing, the whole (the store's
    // counters as of the last service: a pump refreshes them)
    t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
    const times = t.vtable.times.?(t.ptr);
    try testing.expectEqual(@as(u64, 4), times.block_read.count);
    try testing.expectEqual(@as(u64, 4), times.block_crc.count);
    try testing.expectEqual(@as(u64, 1), times.read_wall.count);
    try testing.expectEqual(@as(u64, 1), times.restore_copy.count);
    try testing.expectEqual(@as(u64, 1), times.restore_land.count);
    try testing.expectEqual(@as(u64, 1), times.restore_ready.count);
    try testing.expect(times.restore_ready.max_ns >= times.restore_copy.max_ns and times.restore_ready.max_ns >= times.restore_land.max_ns);
    rig.r.sim.freeSnap(got.buf);
    try stagingDrains(rig.ld.side.store);
}

test "landing: every rank skipped it: the restore fails, its destination discarded once, the point stays on disk" {
    var rig: LandRig = undefined;
    try rig.init("landrefused");
    defer rig.deinit();
    rig.fam.land_pending = 1;
    rig.fam.land_final = .refused;
    const t = rig.ld.tier();
    const job = try rig.begin();
    const p = pollUntilDone(t, job);
    try testing.expectEqual(error.SpillLandRefused, p.failed);
    try testing.expectEqual(@as(u32, 1), rig.fam.discarded);
    try testing.expectEqual(@as(u32, 0), rig.fam.installed);
    try testing.expect(t.vtable.lookup(t.ptr, &want(&rig.ids)) != null);
    rig.fam.land_calls = 0;
    rig.fam.land_final = .failed; // a broken landing (an engine fault) fails it the same way, named
    try testing.expectEqual(error.SpillLandFailed, pollUntilDone(t, try rig.begin()).failed);
    try testing.expectEqual(@as(u32, 2), rig.fam.discarded);
    try stagingDrains(rig.ld.side.store);
}

test "landing: a cancel or a shutdown flush mid-landing discards the destination once" {
    var rig: LandRig = undefined;
    try rig.init("landcancel");
    defer rig.deinit();
    rig.fam.land_pending = 1_000_000;
    const t = rig.ld.tier();
    const job = try rig.begin();
    while (rig.fam.land_calls == 0) try testing.expect(t.vtable.poll(t.ptr, job, std.time.ns_per_ms) == .pending);
    t.vtable.cancel(t.ptr, job);
    try testing.expectEqual(@as(u32, 1), rig.fam.discarded);
    const second = try rig.begin();
    const calls = rig.fam.land_calls;
    while (rig.fam.land_calls == calls) try testing.expect(t.vtable.poll(t.ptr, second, std.time.ns_per_ms) == .pending);
    try testing.expectEqual(@as(u64, 0), t.vtable.flush(t.ptr, std.time.ns_per_s));
    try testing.expectEqual(@as(u32, 2), rig.fam.discarded);
    try testing.expectEqual(@as(u32, 0), rig.fam.installed);
    try testing.expect(t.vtable.lookup(t.ptr, &want(&rig.ids)) != null);
    try stagingDrains(rig.ld.side.store);
}

test "the probe: every PROBE_EVERY-th keep is probed at the next pump; 0 probes none" {
    const r = try Rank.init(0, "probe", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    var s = r.settings;
    s.probe_every = 3;
    const ld = try L.create(gpa, try r.side(0), fam.probing(), s, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [7][bt]u32 = undefined;
    var snaps: [7]*Snap = undefined;
    for (0..7) |i| {
        testIds(&ids[i], @intCast(100 + i));
        snaps[i] = try keptSnap(r.sim, i + 1, i * bt, bt);
    }
    defer for (snaps) |x| gpa.destroy(x);
    for (0..7) |i| {
        _ = keep(t, snaps[i], &ids[i], &.{.{ .name = i + 1, .at = 0 }}) orelse return error.NotKept;
        try pumpUntil(ld, i + 1); // (each write published before the next: the store's small slots are few here)
    }
    try testing.expectEqualSlices(u64, &.{ 3, 6 }, fam.probed.items); // the 3rd and the 6th keep
    var out: [1]pc.TierCounts = undefined;
    _ = t.vtable.counts(t.ptr, &out);
    try testing.expectEqual(@as(u64, 2), out[0].probe_runs);
    try testing.expectEqual(@as(u64, 0), out[0].probe_failed);
    try testing.expect(t.vtable.off.?(t.ptr) == null);
    // PROBE_EVERY 0: a store whose keeps are never probed (the start probe alone)
    var fam2: Fam0 = .{ .sim = r.sim };
    defer fam2.deinit();
    const r2 = try Rank.init(0, "probe0", 0);
    defer r2.deinit();
    const ld2 = try L.create(gpa, try r2.side(0), fam2.probing(), r2.settings, @splat(3));
    defer ld2.destroy();
    for (0..4) |i| {
        _ = keep(ld2.tier(), snaps[i], &ids[i], &.{.{ .name = i + 1, .at = 0 }}) orelse return error.NotKept;
        try pumpUntil(ld2, i + 1);
    }
    try testing.expectEqual(@as(usize, 0), fam2.probed.items.len);
}

test "the probe: a rank whose bytes differ turns the tier off, named, and nothing fails; a busy stream probes later" {
    const r = try Rank.init(0, "probediff", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    var s = r.settings;
    s.probe_every = 1;
    const ld = try L.create(gpa, try r.side(0), fam.probing(), s, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [2][bt]u32 = undefined;
    testIds(&ids[0], 120);
    testIds(&ids[1], 121);
    const a = try keptSnap(r.sim, 1, 0, bt);
    defer gpa.destroy(a);
    fam.probe_says = .busy;
    _ = keep(t, a, &ids[0], &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    t.vtable.pump.?(t.ptr, std.time.ns_per_ms); // the stream was busy: still due
    try testing.expectEqual(@as(usize, 1), fam.probed.items.len);
    fam.probe_says = .{ .differs = 2 };
    t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
    try testing.expectEqual(@as(usize, 2), fam.probed.items.len);
    try testing.expectEqualStrings("probe: rank 2 differs", t.vtable.off.?(t.ptr).?);
    var out: [1]pc.TierCounts = undefined;
    _ = t.vtable.counts(t.ptr, &out);
    try testing.expectEqual(@as(u64, 1), out[0].probe_failed);
    const b = try keptSnap(r.sim, 2, bt, bt);
    defer gpa.destroy(b);
    try testing.expect(keep(t, b, &ids[1], &.{.{ .name = 2, .at = 0 }}) == null); // off: nothing more kept
    // a probed state dropped before its probe ran is not probed
    var fam2: Fam0 = .{ .sim = r.sim };
    defer fam2.deinit();
    const r2 = try Rank.init(0, "probedrop", 0);
    defer r2.deinit();
    const ld2 = try L.create(gpa, try r2.side(0), fam2.probing(), s, @splat(3));
    defer ld2.destroy();
    const t2 = ld2.tier();
    const id = keep(t2, a, &ids[0], &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    t2.vtable.release.?(t2.ptr, id, a);
    t2.vtable.pump.?(t2.ptr, std.time.ns_per_ms);
    try testing.expectEqual(@as(usize, 0), fam2.probed.items.len);
}

/// Whether `key`'s restore marker is still in the store's books.
fn marked(st: *spill.Store, key: format.Digest) bool {
    st.mutex.lockUncancelable(io);
    defer st.mutex.unlock(io);
    return st.marks.contains(key);
}

/// Pumps until `key`'s marker left the books (or a second passed).
fn unmarkedSoon(ld: *L, key: format.Digest) bool {
    var k: usize = 0;
    while (k < 1000) : (k += 1) {
        if (!marked(ld.side.store, key)) return true;
        ld.tier().vtable.pump.?(ld.tier().ptr, std.time.ns_per_ms);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return false;
}

test "quarantine: a restored state's marker stays until the state is used, then goes" {
    var rig: LandRig = undefined;
    try rig.init("quarantine");
    defer rig.deinit();
    const t = rig.ld.tier();
    const key = format.pointKey(&rig.ids, bt, &.{});
    const job = try rig.begin();
    const p = pollUntilDone(t, job);
    try testing.expect(p == .ready);
    const got: *Snap = @ptrCast(@alignCast(p.ready.saved));
    defer rig.r.sim.freeSnap(got.buf);
    // ready and installed, not yet used: a crash now (in the first pass over it) would forget the point at the next open
    try testing.expect(marked(rig.ld.side.store, key));
    const id = tier.pointId(key);
    t.vtable.touch.?(t.ptr, id); // the prompt cache's resume of it (its first pass ran)
    try testing.expect(unmarkedSoon(rig.ld, key));
}

test "quarantine: a restored state released before any use, or a clean flush, confirms it: the point stays" {
    var rig: LandRig = undefined;
    try rig.init("quarantine2");
    defer rig.deinit();
    const t = rig.ld.tier();
    const key = format.pointKey(&rig.ids, bt, &.{});
    const id = tier.pointId(key);
    var p = pollUntilDone(t, try rig.begin());
    try testing.expect(p == .ready);
    var got: *Snap = @ptrCast(@alignCast(p.ready.saved));
    t.vtable.release.?(t.ptr, id, got);
    rig.r.sim.freeSnap(got.buf);
    try testing.expect(unmarkedSoon(rig.ld, key));
    p = pollUntilDone(t, try rig.begin());
    try testing.expect(p == .ready);
    got = @ptrCast(@alignCast(p.ready.saved));
    defer rig.r.sim.freeSnap(got.buf);
    try testing.expect(marked(rig.ld.side.store, key));
    _ = t.vtable.flush(t.ptr, std.time.ns_per_s);
    try testing.expect(unmarkedSoon(rig.ld, key));
    try testing.expect(t.vtable.lookup(t.ptr, &want(&rig.ids)) != null);
}

test "the store's own counters and lock times reach the tier's: quarantine, markers, I/O priority, the lock" {
    const r = try Rank.init(0, "storecounts", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [bt]u32 = undefined;
    testIds(&ids, 140);
    const a = try keptSnap(r.sim, 1, 0, bt);
    defer gpa.destroy(a);
    _ = keep(t, a, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    try pumpUntil(ld, 1); // (the store's lock taken and held a few times)
    ld.side.store.counters.quarantined = 3;
    ld.side.store.counters.marker_failed = 2;
    ld.side.store.counters.ioprio_refused = 1;
    t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
    var out: [1]pc.TierCounts = undefined;
    _ = t.vtable.counts(t.ptr, &out);
    try testing.expectEqual(@as(u64, 3), out[0].quarantined);
    try testing.expectEqual(@as(u64, 2), out[0].marker_failed);
    try testing.expectEqual(@as(u64, 1), out[0].ioprio_refused);
    // the tier's copy is the store's as of the last service (reading the store's counts takes its lock once more)
    const times = t.vtable.times.?(t.ptr);
    const lc = ld.side.store.counts();
    try testing.expect(times.lock_hold.count > 0);
    try testing.expect(times.lock_hold.count <= lc.lock_hold.count and times.lock_hold.max_ns <= lc.lock_hold.max_ns);
    try testing.expect(times.lock_wait.count <= lc.lock_wait.count);
    var in_buckets: u64 = 0;
    for (times.lock_hold.buckets) |b| in_buckets += b;
    try testing.expectEqual(times.lock_hold.count, in_buckets);
}

test "TF_DS_SPILL_IOPRIO as the store's ioprio value: best effort and idle classes, none leaves it" {
    var buf: [256]u8 = undefined;
    const Env = struct {
        pairs: []const [2][]const u8,
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            for (e.pairs) |q| if (std.mem.eql(u8, q[0], name)) return q[1];
            return null;
        }
    };
    const base = [_][2][]const u8{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "/spill" } };
    const sp = @import("engine_api").Spill;
    try testing.expectEqual(@as(?u16, (2 << 13) | 7), tier.Settings.fromSpill(sp.fromEnv(Env{ .pairs = &base }, &buf)).io_priority);
    try testing.expectEqual(@as(?u16, (2 << 13) | 3), tier.Settings.fromSpill(sp.fromEnv(Env{ .pairs = &(base ++ [_][2][]const u8{.{ "TF_DS_SPILL_IOPRIO", "be/3" }}) }, &buf)).io_priority);
    try testing.expectEqual(@as(?u16, 3 << 13), tier.Settings.fromSpill(sp.fromEnv(Env{ .pairs = &(base ++ [_][2][]const u8{.{ "TF_DS_SPILL_IOPRIO", "idle" }}) }, &buf)).io_priority);
    try testing.expectEqual(@as(?u16, null), tier.Settings.fromSpill(sp.fromEnv(Env{ .pairs = &(base ++ [_][2][]const u8{.{ "TF_DS_SPILL_IOPRIO", "none" }}) }, &buf)).io_priority);
}

test "a keep hashes its ids once: the key and the store's block names come from one chain" {
    const r = try Rank.init(0, "chain", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt + 5]u32 = undefined;
    testIds(&ids, 41);
    const sn = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(sn);
    const before = format.test_id_passes.load(.monotonic);
    const id = keep(t, sn, &ids, &.{.{ .name = 5, .at = 0 }}) orelse return error.NotKept;
    try testing.expectEqual(@as(u64, 1), format.test_id_passes.load(.monotonic) - before);
    try pumpUntil(ld, 1);
    // the tier's key is the store's own: a lookup finds the point under the id the keep gave
    const p = t.vtable.lookup(t.ptr, &want(&ids)) orelse return error.NotFound;
    try testing.expectEqual(id, p.id);
    try stagingDrains(ld.side.store);
}

/// A disk that holds the store's writer inside one write for 1.5 s once `stuck` is set (a stuck NVMe).
const StuckDisk = struct {
    stuck: std.atomic.Value(bool) = .init(false),
    inside: std.atomic.Value(bool) = .init(false),

    fn pwrite(ctx: ?*anyopaque, fd: std.c.fd_t, buf: []const u8, off: u64) spill.fsio.Error!usize {
        const d: *StuckDisk = @ptrCast(@alignCast(ctx.?));
        if (d.stuck.load(.acquire)) {
            d.inside.store(true, .release);
            std.Io.sleep(io, .fromMilliseconds(1500), .awake) catch {};
        }
        const n = spill.fsio.testPwrite(fd, buf, buf.len, off);
        if (n < 0) return error.Io;
        return @intCast(n);
    }
};

/// Every thread of a store left to them by an unclean close has returned.
fn storeExited(st: *spill.Store) bool {
    st.mutex.lockUncancelable(io);
    defer st.mutex.unlock(io);
    return st.writer_exited and st.readers_exited == st.reader_threads.items.len;
}

test "a stop never waits on a stuck disk: the store closed by the flush's deadline, or within SPILL_FLUSH_MS without a flush" {
    // an unclean close leaves the store allocated to its threads (the process is on its way out): this test opens it
    // with the page allocator and keeps the staging until those threads have returned
    const pa = std.heap.page_allocator;
    for ([_]bool{ true, false }) |flushed| {
        const r = try Rank.init(0, if (flushed) "close-flushed" else "close-unflushed", 0);
        defer r.deinit();
        var fam: Fam0 = .{ .sim = r.sim };
        defer fam.deinit();
        var disk: StuckDisk = .{};
        var s = r.settings;
        s.flush_ms = if (flushed) 600 else 200;
        const gens = try state.Generations.init(pa, cap, bt);
        const side = try L.S.create(pa, io, .{ .sim = r.sim }, r.sim.layout, gens, s, 0, r.compat(), cap, r.staging, r.events, .{ .ctx = &disk, .free = bigFree, .pwrite = StuckDisk.pwrite });
        const ld = try L.create(gpa, side, fam.family(), s, @splat(3));
        const t = ld.tier();
        var ids: [4 * bt]u32 = undefined;
        testIds(&ids, 61);
        const sn = try keptSnap(r.sim, 1, 0, 4 * bt);
        defer gpa.destroy(sn);
        disk.stuck.store(true, .release);
        _ = keep(t, sn, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
        if (flushed) {
            // the stop's flush: the writer is held, so the point stays unpublished at its 100 ms deadline
            try testing.expectEqual(@as(u64, 1), t.vtable.flush(t.ptr, 100 * std.time.ns_per_ms));
        } else {
            var k: usize = 0;
            while (k < 2000 and !disk.inside.load(.acquire)) : (k += 1) {
                t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
                std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
            }
            try testing.expect(disk.inside.load(.acquire));
        }
        const st = ld.side.store;
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        const clean = ld.close();
        const took = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
        if (flushed) {
            // the flush's deadline had passed: the close gives up at once (not another SPILL_FLUSH_MS, 600 ms)
            try testing.expect(took < 300 * std.time.ns_per_ms);
        } else {
            // no flush ran: SPILL_FLUSH_MS (200 ms) from the close, not the write's 1.5 s
            try testing.expect(took >= 150 * std.time.ns_per_ms and took < 1000 * std.time.ns_per_ms);
        }
        try testing.expect(!clean); // the store left to its threads: the caller keeps the staging and events
        var k: usize = 0;
        while (k < 5000 and !storeExited(st)) : (k += 1) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        try testing.expect(storeExited(st));
    }
}

test "the tier's times: each probe that ran (the start probe too) and the family's landing steps reach /metrics' times" {
    const r = try Rank.init(0, "times", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    var s = r.settings;
    s.probe_every = 2;
    var steps: pc.Timing = .{}; // the family's landing steps (lanes.zig times them)
    steps.observe(3_000_000);
    steps.observe(5_000_000);
    var f = fam.probing();
    f.land_time = &steps;
    const ld = try L.create(gpa, try r.side(0), f, s, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    ld.noteProbe(2_500_000_000); // the start probe, timed by the start before the Leader existed
    var ids: [4][bt]u32 = undefined;
    var snaps: [4]*Snap = undefined;
    for (0..4) |i| {
        testIds(&ids[i], @intCast(300 + i));
        snaps[i] = try keptSnap(r.sim, i + 1, i * bt, bt);
    }
    defer for (snaps) |x| gpa.destroy(x);
    fam.probe_says = .busy; // the 2nd keep's probe meets a busy stream first: not a run, not timed
    for (0..2) |i| {
        _ = keep(t, snaps[i], &ids[i], &.{.{ .name = i + 1, .at = 0 }}) orelse return error.NotKept;
        try pumpUntil(ld, i + 1);
    }
    try testing.expectEqual(@as(u64, 1), t.vtable.times.?(t.ptr).probe.count); // the start probe alone so far
    fam.probe_says = .same;
    t.vtable.pump.?(t.ptr, std.time.ns_per_ms); // now it runs
    for (2..4) |i| {
        _ = keep(t, snaps[i], &ids[i], &.{.{ .name = i + 1, .at = 0 }}) orelse return error.NotKept;
        try pumpUntil(ld, i + 1);
    }
    const times = t.vtable.times.?(t.ptr);
    try testing.expectEqual(@as(u64, 3), times.probe.count); // the start probe, the 2nd keep's and the 4th's
    try testing.expect(times.probe.max_ns >= 2_500_000_000);
    try testing.expectEqual(@as(u64, 2), times.land_piece.count);
    try testing.expectEqual(@as(u64, 5_000_000), times.land_piece.max_ns);
    // a tier without followers lands nothing: no landing times
    var fam1: Fam0 = .{ .sim = r.sim };
    defer fam1.deinit();
    const r1 = try Rank.init(0, "times1", 0);
    defer r1.deinit();
    const ld1 = try L.create(gpa, try r1.side(0), fam1.family(), r1.settings, @splat(3));
    defer ld1.destroy();
    const t1 = ld1.tier();
    try testing.expectEqual(@as(u64, 0), t1.vtable.times.?(t1.ptr).land_piece.count);
}

test "quarantine: a restore read and committed whose landing never completed releases its marker (refused, failed, cancelled, the tier off, a flush)" {
    // the point was never used: nothing crashed in it, so the next open must keep it
    for ([_]enum { refused, failed, cancelled, off, flushed }{ .refused, .failed, .cancelled, .off, .flushed }) |how| {
        var rig: LandRig = undefined;
        try rig.init("quarantine-landing");
        defer rig.deinit();
        const t = rig.ld.tier();
        const key = format.pointKey(&rig.ids, bt, &.{});
        rig.fam.land_pending = switch (how) {
            .refused, .failed => 1,
            .cancelled, .off, .flushed => 1 << 20, // still landing when it ends
        };
        rig.fam.land_final = if (how == .failed) .failed else .refused;
        const job = try rig.begin();
        // until its read committed (rank 0 holds the state; the landing has begun)
        var k: usize = 0;
        while (k < 5000 and rig.fam.land_calls == 0) : (k += 1) {
            _ = t.vtable.poll(t.ptr, job, std.time.ns_per_ms);
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        try testing.expect(rig.fam.land_calls > 0);
        switch (how) {
            .refused, .failed => _ = pollUntilDone(t, job),
            .cancelled => t.vtable.cancel(t.ptr, job),
            .off => {
                rig.ld.setOff("probe: rank 2 differs");
                _ = pollUntilDone(t, job);
            },
            .flushed => _ = t.vtable.flush(t.ptr, std.time.ns_per_s),
        }
        try testing.expect(unmarkedSoon(rig.ld, key));
    }
}

test "quarantine: a point restored twice before its first use holds two markers, and both go" {
    var rig: LandRig = undefined;
    try rig.init("quarantine-twice");
    defer rig.deinit();
    const t = rig.ld.tier();
    const key = format.pointKey(&rig.ids, bt, &.{});
    const id = tier.pointId(key);
    const p1 = pollUntilDone(t, try rig.begin());
    try testing.expect(p1 == .ready);
    const got1: *Snap = @ptrCast(@alignCast(p1.ready.saved));
    defer rig.r.sim.freeSnap(got1.buf);
    const p2 = pollUntilDone(t, try rig.begin()); // (its first copy went stale before its waiter resumed it)
    try testing.expect(p2 == .ready);
    const got2: *Snap = @ptrCast(@alignCast(p2.ready.saved));
    defer rig.r.sim.freeSnap(got2.buf);
    try testing.expect(marked(rig.ld.side.store, key));
    t.vtable.touch.?(t.ptr, id); // its first use
    try testing.expect(unmarkedSoon(rig.ld, key)); // every hold released, none left for the next open to quarantine
}

test "the tier's counters: the family's landings (landed, skipped, failed and aborted, pieces, bytes) and the start probe reach /metrics" {
    const r = try Rank.init(0, "landcounts", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    var lc: tier.LandCounts = .{ .landings = 5, .skipped = 2, .aborted = 1, .failed = 3, .pieces = 40, .bytes = 1 << 30 };
    var f = fam.landing();
    f.land_counts = &lc;
    const ld = try L.create(gpa, try r.side(0), f, r.settings, @splat(3));
    defer ld.destroy();
    ld.noteProbe(2_500_000_000); // the start probe ran (it passed: the tier is on)
    const t = ld.tier();
    var out: [1]pc.TierCounts = undefined;
    _ = t.vtable.counts(t.ptr, &out);
    try testing.expectEqual(@as(u64, 5), out[0].landings);
    try testing.expectEqual(@as(u64, 2), out[0].landings_skipped);
    try testing.expectEqual(@as(u64, 4), out[0].landings_failed); // failed + aborted: every landing begun that did not land
    try testing.expectEqual(@as(u64, 40), out[0].landing_pieces);
    try testing.expectEqual(@as(u64, 1 << 30), out[0].landing_bytes);
    try testing.expectEqual(@as(u64, 1), out[0].probe_runs);
    // the family's counts are read live
    lc.landings += 1;
    _ = t.vtable.counts(t.ptr, &out);
    try testing.expectEqual(@as(u64, 6), out[0].landings);
}

test "a round's busy check takes no store lock once nothing waits for the engine" {
    const r = try Rank.init(0, "busylock", 0);
    defer r.deinit();
    var fam: Fam0 = .{ .sim = r.sim };
    defer fam.deinit();
    const ld = try L.create(gpa, try r.side(0), fam.family(), r.settings, @splat(3));
    defer ld.destroy();
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 77);
    const sn = try keptSnap(r.sim, 1, 0, 4 * bt);
    defer gpa.destroy(sn);
    _ = keep(t, sn, &ids, &.{.{ .name = 1, .at = 0 }}) orelse return error.NotKept;
    try testing.expect(t.vtable.busy.?(t.ptr)); // a write waits for the engine: busy without asking the store
    try pumpUntil(ld, 1);
    try stagingDrains(ld.side.store);
    t.vtable.pump.?(t.ptr, std.time.ns_per_ms); // (the service that sees nothing left)
    const st = ld.side.store;
    st.mutex.lockUncancelable(io);
    const before = st.counters.lock_hold.count;
    st.mutex.unlock(io);
    for (0..100) |_| try testing.expect(!t.vtable.busy.?(t.ptr));
    st.mutex.lockUncancelable(io);
    const after = st.counters.lock_hold.count;
    st.mutex.unlock(io);
    try testing.expect(after - before < 10); // (the writer's own takes only: never one a round)
}

test "quarantine: a restored state the prompt cache drops without adopting it releases its marker" {
    var rig: LandRig = undefined;
    try rig.init("quarantine-unadopted");
    defer rig.deinit();
    const t = rig.ld.tier();
    const key = format.pointKey(&rig.ids, bt, &.{});
    const p = pollUntilDone(t, try rig.begin());
    try testing.expect(p == .ready);
    const got: *Snap = @ptrCast(@alignCast(p.ready.saved));
    try testing.expect(marked(rig.ld.side.store, key));
    // the cache keeps a usable twin instead (unadopted): the family drops the restored state, never used
    rig.ld.dropSnap(got.id);
    rig.r.sim.freeSnap(got.buf);
    try testing.expect(unmarkedSoon(rig.ld, key));
}
