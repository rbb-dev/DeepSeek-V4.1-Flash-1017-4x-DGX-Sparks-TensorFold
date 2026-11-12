//! Rocinante's host tests: the arena's layout and every event's chains as plain functions, then four nodes in one
//! process over a loopback NIC (the bytes of a chain copied into the peer's arena, then its word), one thread a node
//! for the engine's program and two a node for the proxies. The negative controls make the loopback misbehave the way
//! the hardware would if the design's ordering rules were broken, and each must be seen.

const std = @import("std");
const roci = @import("rocinante.zig");
const R = roci.Region;
const C = roci.Class;

const gpa = std.testing.allocator;

// -- the layout (T0) -------------------------------------------------------------------------------------------------

test "the arena: 25 regions on pages, 639,893,504 bytes, the offsets of the design" {
    const l = roci.layout();
    try std.testing.expectEqual(@as(u64, 639_893_504), l.total);
    const want = [_]struct { r: R, off: u64, len: u64 }{
        .{ .r = .psum, .off = 0, .len = 41_943_040 },
        .{ .r = .stage, .off = 41_943_040, .len = 20_971_520 },
        .{ .r = .send2, .off = 62_914_560, .len = 5_242_880 },
        .{ .r = .blk, .off = 68_157_440, .len = 10_485_760 },
        .{ .r = .ek, .off = 78_643_200, .len = 209_715_200 },
        .{ .r = .eng_stage, .off = 288_358_400, .len = 104_857_600 },
        .{ .r = .eng_send2, .off = 393_216_000, .len = 26_214_400 },
        .{ .r = .eng_blk, .off = 419_430_400, .len = 52_428_800 },
        .{ .r = .wob_send, .off = 471_859_200, .len = 9_437_184 },
        .{ .r = .wob_land, .off = 481_296_384, .len = 9_437_184 },
        .{ .r = .pack_send, .off = 490_733_568, .len = 20_643_840 },
        .{ .r = .pack_land, .off = 511_377_408, .len = 20_643_840 },
        .{ .r = .g_x, .off = 532_021_248, .len = 41_943_040 },
        .{ .r = .g_qr, .off = 573_964_288, .len = 10_485_760 },
        .{ .r = .g_y, .off = 584_450_048, .len = 4_194_304 },
        .{ .r = .g_cidx, .off = 588_644_352, .len = 16_777_216 },
        .{ .r = .g_q, .off = 605_421_568, .len = 20_971_520 },
        .{ .r = .g_s, .off = 626_393_088, .len = 2_621_440 },
        .{ .r = .g_pick, .off = 629_014_528, .len = 114_688 },
        .{ .r = .g_wts, .off = 629_129_216, .len = 114_688 },
        .{ .r = .g_taps, .off = 635_949_056, .len = 3_932_160 },
        .{ .r = .head_x, .off = 639_881_216, .len = 10_240 },
    };
    var prev_end: u64 = 0;
    for (want) |w| {
        try std.testing.expectEqual(w.off, l.at(w.r));
        try std.testing.expectEqual(w.len, l.bytes(w.r));
        try std.testing.expectEqual(@as(u64, 0), l.at(w.r) % roci.page);
        try std.testing.expect(l.at(w.r) >= prev_end);
        prev_end = l.at(w.r) + std.mem.alignForward(u64, l.bytes(w.r), roci.page);
    }
    try std.testing.expectEqual(prev_end, l.total);
}

test "the layout hash differs when any region moves (control)" {
    var l = roci.layout();
    const h = l.hash();
    try std.testing.expectEqual(h, roci.layout().hash());
    l.off[5] += roci.page;
    try std.testing.expect(l.hash() != h);
}

test "halves: two parts on 64-byte boundaries that cover the part; the design's page-multiple halves" {
    for ([_]u64{ 4_718_592, 10_321_920, 2_621_440, 1_310_720, 327_680, 28_672 / 1, 1, 64, 100, 8192 }) |len| {
        const h = roci.halves(len);
        try std.testing.expectEqual(len, h[0] + h[1]);
        try std.testing.expect(h[0] % roci.line == 0 or h[1] == 0);
    }
    try std.testing.expectEqual([2]u64{ 14_336, 14_336 }, roci.halves(28_672));
    try std.testing.expectEqual([2]u64{ 4_718_592, 4_718_592 }, roci.halves(9_437_184));
    try std.testing.expectEqual(@as(u64, 0), roci.halves(4_718_592)[0] % roci.page);
}

// -- the events (T0) -------------------------------------------------------------------------------------------------

const perm: roci.Order = .{ 2, 0, 3, 1 };

fn eachNode(comptime f: fn (g: usize, order: roci.Order) anyerror!void) !void {
    for ([_]roci.Order{ roci.identity, perm }) |o| for (0..roci.world) |g| try f(g, o);
}

fn srcRegion(c: C) []const R {
    return switch (c) {
        .own1 => &.{.psum},
        .own2 => &.{.send2},
        .eng1 => &.{.ek},
        .eng2 => &.{.eng_send2},
        .wob => &.{.wob_send},
        .pack => &.{.pack_send},
        .gf => &.{ .g_qr, .g_y },
        .ga => &.{.g_x},
        .gm => &.{ .g_q, .g_s, .g_pick, .g_wts },
        .gi => &.{.g_cidx},
        .gt => &.{.g_taps},
        .hx => &.{.head_x},
    };
}

fn dstRegion(c: C) []const R {
    return switch (c) {
        .own1 => &.{.stage},
        .own2 => &.{.blk},
        .eng1 => &.{.eng_stage},
        .eng2 => &.{.eng_blk},
        .wob => &.{.wob_land},
        .pack => &.{.pack_land},
        else => srcRegion(c),
    };
}

fn inside(l: roci.Layout, regs: []const R, off: u64, len: u64) bool {
    for (regs) |r| if (off >= l.at(r) and off + len <= l.at(r) + l.bytes(r)) return true;
    return false;
}

fn eventsAreSound(g: usize, order: roci.Order) !void {
    const l = roci.layout();
    for ([_]usize{ 4096, 2048 }) |cap| for (0..roci.n_classes) |ci| {
        const class: C = @fromBackingInt(@intCast(ci));
        const e = roci.buildEvent(class, cap, g, order);
        try std.testing.expectEqual(class, e.class);
        for (e.peerSlice(), 0..) |peer, k| {
            try std.testing.expect(peer != g and peer < roci.world);
            // the peer expects this node as one of its sources
            var srcs: [roci.world]u8 = undefined;
            try std.testing.expect(std.mem.indexOfScalar(u8, roci.sourcesOf(class, peer, &srcs), @intCast(g)) != null);
            // every piece lies inside its source and its destination regions; no two pieces to one peer overlap
            var seen: std.ArrayList([2]u64) = .empty;
            defer seen.deinit(gpa);
            for (0..roci.rails) |rail| for (e.chains[k][rail].slice()) |p| {
                try std.testing.expect(inside(l, srcRegion(class), p.src, p.len));
                try std.testing.expect(inside(l, dstRegion(class), p.dst, p.len));
                try std.testing.expect(p.len > 0);
                for (seen.items) |s| try std.testing.expect(p.dst + p.len <= s[0] or p.dst >= s[0] + s[1]);
                try seen.append(gpa, .{ p.dst, p.len });
            };
            // both rails carry bytes: a pairwise part's halves, a gather's halves
            try std.testing.expect(e.chains[k][0].n > 0 and e.chains[k][1].n > 0);
            // the halves are equal but for a 64-byte round (a part's own blocks are equal)
            const d = @as(i64, @intCast(e.chains[k][0].bytes())) - @as(i64, @intCast(e.chains[k][1].bytes()));
            try std.testing.expect(@abs(d) <= roci.line * roci.max_pieces);
        }
    };
}

test "every event of every node, both orders and caps: pieces inside their regions, disjoint, both rails, peers expect them" {
    try eachNode(eventsAreSound);
}

    var events: u64 = 0;
    for (0..roci.world) |g| {
        var writes: u64 = 0;
        var words: u64 = 0;
        var bytes: u64 = 0;
        events = 0;
        for (per_pass, 0..) |n, ci| {
            const e = roci.buildEvent(@fromBackingInt(@intCast(ci)), 4096, g, roci.identity);
            events += n;
            for (0..e.n_peers) |k| for (0..roci.rails) |r| {
                writes += n * e.chains[k][r].n;
                words += n;
                bytes += n * e.chains[k][r].bytes();
            };
        }
    }
}

test "owner part 1 on the rails: the receiver's own rows' block on rail A, its column partner's on rail B" {
    const l = roci.layout();
    for (0..roci.world) |g| {
        const e = roci.buildEvent(.own1, 4096, g, perm);
        const blk: u64 = 1024 * 2560 * 4;
        try std.testing.expectEqual(l.at(.psum) + perm[g ^ 1] * blk, e.chains[0][0].pieces[0].src);
        try std.testing.expectEqual(l.at(.psum) + perm[g ^ 3] * blk, e.chains[0][1].pieces[0].src);
        try std.testing.expectEqual(l.at(.stage), e.chains[0][0].pieces[0].dst);
        try std.testing.expectEqual(l.at(.stage) + blk, e.chains[0][1].pieces[0].dst);
        try std.testing.expectEqual(blk, e.chains[0][0].pieces[0].len);
    }
}

test "control: the 2,048-row plan is not a prefix of the 4,096-row plan" {
    const a = roci.buildEvent(.own1, 4096, 0, roci.identity);
    const b = roci.buildEvent(.own1, 2048, 0, roci.identity);
    try std.testing.expect(a.chains[0][1].pieces[0].src != b.chains[0][1].pieces[0].src);
    try std.testing.expectEqual(a.chains[0][0].pieces[0].len, 2 * b.chains[0][0].pieces[0].len);
}

test "the scheduling entry round-trips its ticket, class and shape" {
    const e = roci.schedEntry(0x1_0000_0005, .gm, 1);
    try std.testing.expectEqual(@as(u32, 5), roci.schedTicket(e));
    try std.testing.expectEqual(C.gm, roci.schedClass(e));
    try std.testing.expectEqual(@as(u8, 1), roci.schedShape(e));
}

// -- four nodes over a loopback NIC ----------------------------------------------------------------------------------

const Mode = struct {
    /// Control: the word before its data (the order the design forbids); the consumer can then read stale bytes.
    word_first: bool = false,
    /// Control: a rail's whole chain late (the consumer waiting for the other rail's word only is then wrong).
    late_rail: ?u8 = null,
    late_ns: u64 = 0,
    /// Node whose NIC accepts posts and sends nothing (a dead peer).
    silent: ?u8 = null,
    /// Node whose posts fail.
    broken: ?u8 = null,
    /// Fill the gaps between a chain's bytes and its word with this pause.
    gap_ns: u64 = 0,
    /// Control (with word_first): every chain's bytes held back, the words not, until the test opens the gate (Net.openGate),
    /// so a consumer that trusts the word looks before any byte has landed, whatever else the machine is doing.
    hold_data: bool = false,
    /// Control: a rail's whole chain (word and bytes) held back the same way.
    hold_rail: ?u8 = null,
};

/// A copy a held post has not made yet.
const Held = struct { dst: [*]u8, src: [*]const u8, len: usize, word: ?*u64, value: u64 };

const Node = struct {
    arena: []align(std.heap.page_size_min) u8,
    ctl: *roci.Control,
    t: *roci.Transport = undefined,
    net: *Net = undefined,
    me: u8 = 0,
};

const Net = struct {
    nodes: [roci.world]Node = undefined,
    mode: Mode = .{},
    failed_nic: [roci.world]std.atomic.Value(u32) = @splat(.init(0)),
    gate: std.atomic.Value(u32) = .init(0), // 1: the held posts have been released
    gate_lock: std.atomic.Value(u32) = .init(0),
    held: std.ArrayList(Held) = .empty,

    fn lockGate(n: *Net) void {
        while (n.gate_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn unlockGate(n: *Net) void {
        n.gate_lock.store(0, .release);
    }

    /// Releases every held copy (bytes, then the word of a held rail) and lets later posts through at once. Safe to call twice.
    fn openGate(n: *Net) void {
        n.lockGate();
        defer n.unlockGate();
        n.gate.store(1, .release);
        for (n.held.items) |h| {
            @memcpy(h.dst[0..h.len], h.src[0..h.len]);
            if (h.word) |w| @atomicStore(u64, w, h.value, .release);
        }
        n.held.clearRetainingCapacity();
    }

    fn open(mode: Mode, set: roci.Settings, orders: roci.Order) !*Net {
        const n = try gpa.create(Net);
        n.* = .{ .mode = mode };
        for (&n.nodes, 0..) |*nd, g| {
            nd.arena = try std.posix.mmap(null, total, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
            nd.ctl = try gpa.create(roci.Control);
            nd.ctl.* = .{};
            nd.net = n;
            nd.me = @intCast(g);
            nd.t = try roci.Transport.init(gpa, @intCast(g), orders, nd.ctl, @intFromPtr(nd.ctl), .{ .ctx = nd, .post_fn = post, .fail_fn = failNic, .reap_fn = reap }, set);
        }
        return n;
    }

    fn start(n: *Net) !void {
        for (&n.nodes) |*nd| try nd.t.start(null);
    }

    /// Every node's proxies stopped before any arena goes: a proxy of one node may still be copying into another's (a control's late
    /// data behind its word).
    fn close(n: *Net) void {
        n.openGate();
        n.held.deinit(gpa);
        for (&n.nodes) |*nd| {
            std.posix.munmap(nd.arena);
            gpa.destroy(nd.ctl);
        }
        gpa.destroy(n);
    }

    fn post(ctx: *anyopaque, rail: u8, peer: u8, class: C, chain: *const roci.Chain, word: u64) roci.PostError!void {
        const me: *Node = @ptrCast(@alignCast(ctx));
        const m = me.net.mode;
        if (me.net.failed_nic[me.me].load(.acquire) != 0) return error.QpError;
        if (m.broken == me.me) return error.PostFailed;
        if (m.silent == me.me) return;
        if (m.late_rail == rail) sleep(m.late_ns);
        const w = &dst.ctl.recv[@backingInt(class)][me.me][rail].v;
        if (m.hold_rail == rail) {
            // the rail's bytes and then its word, when the gate opens
            me.net.lockGate();
            defer me.net.unlockGate();
            if (me.net.gate.load(.acquire) == 0) {
                for (chain.slice()) |p| me.net.held.append(gpa, .{ .dst = dst.arena[p.dst..].ptr, .src = me.arena[p.src..].ptr, .len = p.len, .word = null, .value = 0 }) catch @panic("held post");
                me.net.held.append(gpa, .{ .dst = dst.arena.ptr, .src = dst.arena.ptr, .len = 0, .word = w, .value = word }) catch @panic("held post");
                return;
            }
        }
        if (m.word_first) {
            sleep(m.gap_ns);
            if (m.hold_data) {
                me.net.lockGate();
                defer me.net.unlockGate();
                if (me.net.gate.load(.acquire) == 0) {
                    for (chain.slice()) |p| me.net.held.append(gpa, .{ .dst = dst.arena[p.dst..].ptr, .src = me.arena[p.src..].ptr, .len = p.len, .word = null, .value = 0 }) catch @panic("held post");
                    return;
                }
            }
        }
        for (chain.slice()) |p| @memcpy(dst.arena[p.dst..][0..p.len], me.arena[p.src..][0..p.len]);
    }

    fn failNic(ctx: *anyopaque) void {
        const me: *Node = @ptrCast(@alignCast(ctx));
        me.net.failed_nic[me.me].store(1, .release);
    }

    fn reap(ctx: *anyopaque, rail: u8) roci.PostError!void {
        _ = rail;
        const me: *Node = @ptrCast(@alignCast(ctx));
        if (me.net.failed_nic[me.me].load(.acquire) != 0) return error.QpError;
    }
};

fn sleep(ns: u64) void {
    if (ns == 0) return;
    const ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}

/// The stream of a node: an op runs when it is called, which is the order a stream keeps; a wait blocks the caller
/// until its words are there, as the GPU's stream wait blocks the consumer behind it.
const FakeStream = struct {
    only_rail_a: bool = false,
    timeout_ns: u64 = 10 * std.time.ns_per_s,

    pub fn writeValue64(_: FakeStream, addr: u64, v: u64) !void {
        @atomicStore(u64, @as(*u64, @ptrFromInt(addr)), v, .release);
    }

    pub fn waitValues64(s: FakeStream, words: []const u64, values: []const u64) !void {
        const t0 = std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds;
        for (words, values, 0..) |w, v, i| {
            // the controls wait on rail A's words alone: every other word of the batch is rail B's
            if (s.only_rail_a and i % 2 == 1) continue;
            while (true) {
                const have = @atomicLoad(u64, @as(*const u64, @ptrFromInt(w)), .acquire);
                if (@as(i64, @bitCast(have -% v)) >= 0) break;
                if (@as(u64, @intCast(std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds - t0)) > s.timeout_ns) return error.WaitTimeout;
                std.atomic.spinLoopHint();
            }
        }
    }
};

fn pattern(buf: []u8, tag: u64) void {
    const words = std.mem.bytesAsSlice(u64, buf[0 .. buf.len / 8 * 8]);
    for (words, 0..) |*w, i| w.* = tag *% 0x9e3779b97f4a7c15 +% i;
}

/// Fill the source ranges of node `g`'s event `class` with a pattern keyed by (generation, node, offset of the range).
fn fillSources(nd: *Node, class: C, gen: u64, order: roci.Order, cap: usize) void {
    const e = roci.buildEvent(class, cap, nd.me, order);
    for (0..e.n_peers) |k| for (0..roci.rails) |r| for (e.chains[k][r].slice()) |p| {
        pattern(nd.arena[p.src..][0..p.len], gen *% 1000003 +% @as(u64, nd.me) *% 7919 +% p.src);
    };
}

/// Count the bytes of node `nd` that differ from what each source sent for `class` (0: all landed).
fn badBytes(net: *Net, nd: *Node, class: C, gen: u64, order: roci.Order, cap: usize) !u64 {
    var bad: u64 = 0;
    var srcs: [roci.world]u8 = undefined;
    for (roci.sourcesOf(class, nd.me, &srcs)) |s| {
        const e = roci.buildEvent(class, cap, s, order);
        const k = std.mem.indexOfScalar(u8, e.peerSlice(), nd.me).?;
        for (0..roci.rails) |r| for (e.chains[k][r].slice()) |p| {
            const want = try gpa.alloc(u8, p.len);
            defer gpa.free(want);
            pattern(want, gen *% 1000003 +% @as(u64, s) *% 7919 +% p.src);
            const got = net.nodes[nd.me].arena[p.dst..][0..p.len];
            // a part's tail short of a whole word is not written by the pattern: compare whole words
            const n = p.len / 8 * 8;
            if (!std.mem.eql(u8, want[0..n], got[0..n])) {
                for (want[0..n], got[0..n]) |a, b| bad += @intFromBool(a != b);
            }
        };
    }
    return bad;
}

const Barrier = struct {
    n: std.atomic.Value(u32) = .init(0),
    gen: std.atomic.Value(u32) = .init(0),
    fn wait(b: *Barrier, parties: u32) void {
        const g = b.gen.load(.acquire);
        if (b.n.fetchAdd(1, .acq_rel) + 1 == parties) {
            b.n.store(0, .release);
            _ = b.gen.fetchAdd(1, .acq_rel);
        } else while (b.gen.load(.acquire) == g) std.atomic.spinLoopHint();
    }
};

const Run = struct {
    net: *Net,
    bar: *Barrier,
    classes: []const C,
    gens: u64,
    order: roci.Order,
    only_a: bool = false,
    bad: [roci.world]u64 = @splat(0),
    err: [roci.world]?anyerror = @splat(null),
};

fn rankMain(run: *Run, g: usize) void {
    rankBody(run, g) catch |e| {
        run.err[g] = e;
    };
}

fn rankBody(run: *Run, g: usize) !void {
    const nd = &run.net.nodes[g];
        for (run.classes) |class| {
            const cap: usize = if (gen % 2 == 0) 4096 else 2048;
            const shape: u8 = if (gen % 2 == 0) 0 else 1;
            fillSources(nd, class, gen + 1, run.order, cap);
            // a generation's events are separated: the loopback has no GPU, so the test keeps the dataflow's fence
            run.bar.wait(roci.world);
            try nd.t.send(s, class, shape);
            try nd.t.wait(s, class);
            run.bad[g] += try badBytes(run.net, nd, class, gen + 1, run.order, cap);
            run.bar.wait(roci.world);
        }
    }
}

fn runProgram(net: *Net, classes: []const C, gens: u64, order: roci.Order, only_a: bool) !Run {
    var bar: Barrier = .{};
    var th: [roci.world]std.Thread = undefined;
    for (&th, 0..) |*t, g| t.* = try std.Thread.spawn(.{}, rankMain, .{ &run, g });
    for (th) |t| t.join();
    net.openGate(); // (every rank has looked: held bytes land now)
    return run;
}

test "four nodes: owner sums, wo_b codes and a four-region gather land every byte, generation after generation, both caps" {
    for ([_]roci.Order{ roci.identity, perm }) |order| {
        const net = try Net.open(.{}, .{ .nap = false }, order);
        defer net.close();
        try net.start();
        const classes = [_]C{ .own1, .own2, .wob, .gm, .gf };
        const run = try runProgram(net, &classes, 4, order, false);
        for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
        for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
        // the counters: what the plans say, no more
        const sends: u64 = classes.len * 4;
        for (&net.nodes) |*nd| {
            const c = nd.t.counters();
            try std.testing.expectEqual(sends, c.events);
            try std.testing.expectEqual(sends, c.doorbells);
            try std.testing.expectEqual(sends, c.stream_writes);
            try std.testing.expectEqual(sends, c.wait_groups);
            var writes: u64 = 0;
            var words: u64 = 0;
            for (0..4) |gen| for (classes) |cl| {
                const e = roci.buildEvent(cl, if (gen % 2 == 0) 4096 else 2048, nd.me, order);
                for (0..e.n_peers) |k| for (0..roci.rails) |r| {
                    writes += e.chains[k][r].n;
                    words += 1;
                };
            };
            try std.testing.expectEqual(writes, c.data_writes);
            try std.testing.expectEqual(words, c.words);
            try std.testing.expectEqual(words, c.posts);
            try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
        }
    }
}

test "control: a NIC that writes the word before the data lets a consumer read stale bytes" {
    const net = try Net.open(.{ .word_first = true, .hold_data = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    const run = try runProgram(net, &.{.own2}, 1, roci.identity, false);
    var bad: u64 = 0;
    for (run.bad) |b| bad += b;
    try std.testing.expect(bad > 0);
}

test "control: waiting on rail A's word alone while rail B is late reads stale bytes of rail B" {
    const net = try Net.open(.{ .hold_rail = 1 }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    const run = try runProgram(net, &.{.own1}, 1, roci.identity, true);
    var bad: u64 = 0;
    for (run.bad) |b| bad += b;
    try std.testing.expect(bad > 0);
    // and the same late rail with both rails waited is clean
    const net2 = try Net.open(.{ .late_rail = 1, .late_ns = 100_000_000 }, .{ .nap = false }, roci.identity);
    defer net2.close();
    try net2.start();
    const ok = try runProgram(net2, &.{.own1}, 1, roci.identity, false);
    for (ok.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
}

    const net = try Net.open(.{ .word_first = true, .hold_data = true }, .{ .nap = false, .codes = true }, roci.identity);
    const net = try Net.open(.{ .word_first = true, .hold_data = true }, .{ .nap = false, .gathers = true }, roci.identity);
test "the boundary's slab: one node's rows of the four cache arrays, array after array, at its rank's place, in halves on the rails" {
    /// The control's order: the other ranks start their program once the runaway has posted every event of every layer, so
    /// each landed piece they check has been written over by a later event, whatever the machine is doing.
    ahead_done: std.atomic.Value(u32) = .init(0),
    if (r.runaway != null and !run_ahead) {
        var waited: u64 = 0;
        while (r.ahead_done.load(.acquire) == 0) : (waited += 1) {
            if (waited > 20_000) return error.RunawayNeverFinished; // (10 s)
            sleep(500_000);
        }
    }
    defer if (run_ahead) r.ahead_done.store(1, .release);
        if (class == .gh or class == .gt or class == .hx) continue; // (no full-cap plan: their rows are a decoder window's)
// -- failure ---------------------------------------------------------------------------------------------------------

test "a peer that never answers: the watchdog names it, every queued wait is released, the next enqueue refuses" {
    const net = try Net.open(.{ .silent = 1 }, .{ .nap = false, .dead_ms = 150 }, roci.identity);
    defer net.close();
    try net.start();
    const nd = &net.nodes[0];
    const s: FakeStream = .{};
    try nd.t.send(s, .own1, 0);
    try nd.t.wait(s, .own1); // returns: the abort value released it (the peer's words never came)
    const f = nd.t.failure().?;
    try std.testing.expectEqual(roci.FailReason.watchdog, f.reason);
    try std.testing.expectEqual(C.own1, f.class);
    try std.testing.expectEqual(@as(u8, 1), f.peer);
    try std.testing.expectError(error.RociFailed, nd.t.check());
    try std.testing.expectError(error.RociFailed, nd.t.send(s, .own2, 0));
    // every awaited word holds the abort value, above any generation and below 2^63
    for (&nd.ctl.recv) |*c| for (c) |*sw| for (sw) |*w| try std.testing.expectEqual(roci.abort_value, w.v);
    // the queue pairs were moved to error first
    try std.testing.expectEqual(@as(u32, 1), net.failed_nic[0].load(.acquire));
    // the abort value passes every wait's cyclic compare: a want of any generation is below it
    try std.testing.expect(@as(i64, @bitCast(roci.abort_value -% 1_000_000)) >= 0);
}

test "a post that fails stops the transport naming the class and the peer" {
    const net = try Net.open(.{ .broken = 0 }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    const nd = &net.nodes[0];
    try nd.t.send(FakeStream{}, .wob, 0);
    var tries: u32 = 0;
    while (nd.t.failure() == null and tries < 2000) : (tries += 1) sleep(1_000_000);
    const f = nd.t.failure().?;
    try std.testing.expectEqual(roci.FailReason.post, f.reason);
    try std.testing.expectEqual(C.wob, f.class);
    try std.testing.expectEqual(@as(u8, 2), f.peer);
}

test "a doorbell with no scheduling entry behind it stops the proxies loudly (stamp)" {
    const net = try Net.open(.{}, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    const nd = &net.nodes[2];
    try (FakeStream{}).writeValue64(@intFromPtr(&nd.ctl.doorbell.v), 1);
    var tries: u32 = 0;
    while (nd.t.failure() == null and tries < 2000) : (tries += 1) sleep(1_000_000);
    try std.testing.expectEqual(roci.FailReason.stamp, nd.t.failure().?.reason);
}

test "the counter line names every counter the gates read" {
    const net = try Net.open(.{}, .{ .nap = false }, roci.identity);
    defer net.close();
    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try net.nodes[0].t.counterLine(&w, "pass 1");
    const text = w.buffered();
        try std.testing.expect(std.mem.indexOf(u8, text, k) != null);
}

// -- the device chain on a real driver (skipped without one) ---------------------------------------------------------

const cuda = @import("cuda");

const GpuNode = struct {
    arena: []align(64) u8, // cudaHostAlloc(Portable | Mapped | WriteCombined)
    arena_dev: u64,
    ctl: *roci.Control, // cudaHostAlloc(Portable | Mapped)
    ctl_dev: u64,
    t: *roci.Transport = undefined,
    peer: *GpuNode = undefined,
    me: u8 = 0,
};

fn hostAlloc(d: *const cuda.Driver, len: usize, flags: c_uint) !struct { p: [*]align(64) u8, dev: u64 } {
    var p: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&p, len, flags), "cuMemHostAlloc");
    var dev: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, p, 0), "cuMemHostGetDevicePointer");
    return .{ .p = @ptrCast(@alignCast(p.?)), .dev = dev };
}

fn gpuPost(ctx: *anyopaque, rail: u8, peer: u8, class: C, chain: *const roci.Chain, word: u64) roci.PostError!void {
    const me: *GpuNode = @ptrCast(@alignCast(ctx));
    _ = peer;
    for (chain.slice()) |p| @memcpy(me.peer.arena[p.dst..][0..p.len], me.arena[p.src..][0..p.len]);
    @atomicStore(u64, &me.peer.ctl.recv[@backingInt(class)][me.me][rail].v, word, .release);
}
fn gpuFail(_: *anyopaque) void {}
fn gpuReap(_: *anyopaque, _: u8) roci.PostError!void {}

test "the device chain: a stream memset into write-combined host memory, the doorbell stream write, a proxy post, a batched stream wait on both rails' words, the bytes landed" {
    var driver = cuda.Driver.open() catch return error.SkipZigTest;
    defer driver.close();
    var cx = cuda.Context.init(&driver, 0) catch return error.SkipZigTest;
    defer cx.deinit();
    const ok = driver.api.cuDeviceGetAttribute;
    _ = ok;
    const arena_len: usize = 64 << 20;
    var nodes: [2]GpuNode = undefined;
    for (&nodes, 0..) |*n, i| {
        const a = try hostAlloc(&driver, arena_len, 0x01 | 0x02 | 0x04);
        const c = try hostAlloc(&driver, @sizeOf(roci.Control), 0x01 | 0x02);
        n.* = .{ .arena = a.p[0..arena_len], .arena_dev = a.dev, .ctl = @ptrCast(c.p), .ctl_dev = c.dev, .me = @intCast(i) };
        n.ctl.* = .{};
    }
    defer for (&nodes) |*n| {
        _ = driver.api.cuMemFreeHost(n.arena.ptr);
        _ = driver.api.cuMemFreeHost(@ptrCast(n.ctl));
    };
    for (&nodes, 0..) |*n, i| {
        n.peer = &nodes[1 - i];
        n.t = try roci.Transport.init(gpa, @intCast(i), roci.identity, n.ctl, n.ctl_dev, .{ .ctx = n, .post_fn = gpuPost, .fail_fn = gpuFail, .reap_fn = gpuReap }, .{ .nap = false });
        try n.t.start(null);
    }
    defer for (&nodes) |*n| n.t.deinit();
    var s0 = try cuda.Stream.init(&driver, true);
    defer s0.deinit();
    var s1 = try cuda.Stream.init(&driver, true);
    defer s1.deinit();
    const l = roci.layout();
    const blk: usize = 1024 * 2560 * 4;
    for (1..6) |gen| {
        // each node's four psum blocks get a value naming (node, block, generation), written by the stream
        for ([_]struct { n: *GpuNode, s: *cuda.Stream }{ .{ .n = &nodes[0], .s = &s0 }, .{ .n = &nodes[1], .s = &s1 } }) |x| {
            for (0..4) |b| {
                const v: u32 = @intCast((@as(u32, x.n.me) << 28) | (@as(u32, @intCast(b)) << 20) | gen);
                try driver.check(driver.api.cuMemsetD32Async(x.n.arena_dev + l.at(.psum) + b * blk, v, blk / 4, x.s.handle), "memset");
            }
            try x.n.t.send(x.s.*, .own1, 0);
        }
        for ([_]struct { n: *GpuNode, s: *cuda.Stream }{ .{ .n = &nodes[0], .s = &s0 }, .{ .n = &nodes[1], .s = &s1 } }) |x| {
            try x.n.t.wait(x.s.*, .own1);
            try x.s.synchronize();
            // stage[0] holds the peer's block of this node's rows, stage[1] the peer's block of its column partner's
            const peer: u32 = 1 - x.n.me;
            const w0: u32 = (peer << 28) | (@as(u32, x.n.me) << 20) | @as(u32, @intCast(gen));
            const w1: u32 = (peer << 28) | (@as(u32, x.n.me ^ 2) << 20) | @as(u32, @intCast(gen));
            const st = x.n.arena[l.at(.stage)..];
            for ([_]usize{ 0, blk / 4 - 1, blk / 8 }) |i| {
                try std.testing.expectEqual(w0, std.mem.readInt(u32, st[i * 4 ..][0..4], .little));
                try std.testing.expectEqual(w1, std.mem.readInt(u32, st[blk + i * 4 ..][0..4], .little));
            }
        }
    }
    for (&nodes) |*n| try std.testing.expectEqual(@as(?@TypeOf(n.t.failure().?), null), n.t.failure());
}

test "the start line names the Rocinante transport, the arena and the layout hash" {
    var w = std.Io.Writer.fixed(&buf);
    const text = w.buffered();
test "the start line says what the transport carries: the owner sums, and the codes and the gathers where the arena holds their regions; the event counts follow" {
    try std.testing.expectEqualStrings("the owner sums of full passes (attention, MoE, Engram)", roci.carriesText(&buf, false, false));
    try std.testing.expectEqualStrings("the owner sums of full passes (attention, MoE, Engram), the gathers (qr and y, x, the routing's codes and picks, the indexer's top-k, the boundary's cache rows)", g);
    try std.testing.expectEqualStrings("the owner sums of full passes (attention, MoE, Engram), the codes families (wo_b's and the expert pack's)", c);
    try std.testing.expect(std.mem.indexOf(u8, both, "NCCL") == null and std.mem.indexOf(u8, both, "the codes families") != null and std.mem.endsWith(u8, both, "the boundary's cache rows)"));
}

test "a batched stream wait of six words (a gather's three peers on both rails) holds the stream until the last one arrives" {
    var driver = cuda.Driver.open() catch return error.SkipZigTest;
    defer driver.close();
    var cx = cuda.Context.init(&driver, 0) catch return error.SkipZigTest;
    defer cx.deinit();
    const c = try hostAlloc(&driver, @sizeOf(roci.Control), 0x01 | 0x02);
    defer _ = driver.api.cuMemFreeHost(c.p);
    const ctl: *roci.Control = @ptrCast(c.p);
    ctl.* = .{};
    var s = try cuda.Stream.init(&driver, true);
    defer s.deinit();
    var words: [6]u64 = undefined;
    var vals: [6]u64 = undefined;
    for (&words, &vals, 0..) |*w, *v, i| {
        w.* = c.dev + @offsetOf(roci.Control, "recv") + i * @sizeOf(roci.Word);
        v.* = 7;
        @atomicStore(u64, &ctl.recv[0][i / 2][i % 2].v, if (i == 5) 6 else 7, .release);
    }
    // the sixth word is one short: the wait, then a write behind it, must not have run
    try s.waitValues64(&words, &vals);
    try s.writeValue64(c.dev + @offsetOf(roci.Control, "abort"), 0x77);
    sleep(150_000_000);
    try std.testing.expectEqual(@as(u64, 0), @atomicLoad(u64, &ctl.abort.v, .acquire));
    @atomicStore(u64, &ctl.recv[0][2][1].v, 7, .release);
    try s.synchronize();
    try std.testing.expectEqual(@as(u64, 0x77), @atomicLoad(u64, &ctl.abort.v, .acquire));
}

test "a pass of 84 owner events: each rail's proxy prints its line and times every class from its post to the peer's word" {
    const net = try Net.open(.{}, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    const Prog = struct {
        fn main(nd: *Node, err: *?anyerror) void {
            body(nd) catch |e| {
                err.* = e;
            };
        }
        fn body(nd: *Node) !void {
            const s: FakeStream = .{};
            for (0..42) |_| {
                try nd.t.send(s, .own1, 0);
                try nd.t.wait(s, .own1);
                try nd.t.send(s, .own2, 0);
                try nd.t.wait(s, .own2);
            }
        }
    };
    var errs: [roci.world]?anyerror = @splat(null);
    var th: [roci.world]std.Thread = undefined;
    for (&th, 0..) |*t, g| t.* = try std.Thread.spawn(.{}, Prog.main, .{ &net.nodes[g], &errs[g] });
    for (th) |t| t.join();
    for (errs) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (&net.nodes) |*nd| {
        // the proxies look at the words on their next pass of the loop: wait for every counter checked below, on both rails
        var tries: u32 = 0;
        while (tries < 2000) : (tries += 1) {
            var done = true;
            for (0..roci.rails) |r| {
                if (nd.t.wire[r][@backingInt(C.own1)].n < 42 or nd.t.wire[r][@backingInt(C.own2)].n < 42) done = false;
                if (nd.t.rail_prev[r].posts < 84 or nd.t.rail_prev[r].words < 84) done = false;
            }
            if (done) break;
            sleep(1_000_000);
        }
        for (0..roci.rails) |r| {
            try std.testing.expectEqual(@as(u64, 42), nd.t.wire[r][@backingInt(C.own1)].n);
            try std.testing.expectEqual(@as(u64, 42), nd.t.wire[r][@backingInt(C.own2)].n);
            try std.testing.expect(nd.t.wire[r][@backingInt(C.own1)].max_ns >= nd.t.wire[r][@backingInt(C.own1)].sum_ns / 42);
            // the rail's line was made once, at event 84, from its own counters
            try std.testing.expectEqual(@as(u64, 84), nd.t.rail_prev[r].posts);
            try std.testing.expectEqual(@as(u64, 84), nd.t.rail_prev[r].words);
        }
    }
}

// -- a decoder window's taps and head row (classes gt, hx) ----------------------------------------------------------------

/// How many times each row of a decoder window lands whole in receiver `d`'s taps region over the three senders' gt events of a window of
/// `plan_n` rows (index i: row i; the rails' halves meet on a 64-byte boundary inside a row, so a row counts once all its bytes landed),
/// and that every piece lies inside the region at the same offset in both arenas.
fn tapsCover(d: usize, order: roci.Order, plan_n: usize) ![roci.hand_rows]u8 {
    const l = roci.layout();
    var got: [roci.hand_rows]u64 = @splat(0);
    var srcs: [roci.world]u8 = undefined;
    for (roci.sourcesOf(.gt, d, &srcs)) |src| {
        const e = roci.buildEventX(.gt, plan_n, 0, src, order);
        try std.testing.expectEqual(@as(u8, 3), e.n_peers);
        const k = std.mem.indexOfScalar(u8, e.peerSlice(), @intCast(d)).?;
        for (0..roci.rails) |r| for (e.chains[k][r].slice()) |p| {
            try std.testing.expect(p.src == p.dst);
            try std.testing.expect(inside(l, &.{.g_taps}, p.src, p.len));
            try std.testing.expectEqual(@as(u64, 0), (p.src - l.at(.g_taps)) % roci.line);
            var b = p.src - l.at(.g_taps);
            const end = b + p.len;
            while (b < end) {
                const row = b / roci.taps_row;
                const upto = @min(end, (row + 1) * roci.taps_row);
                got[row] += upto - b;
                b = upto;
            }
        };
    }
    var cover: [roci.hand_rows]u8 = @splat(0);
    for (got, &cover) |g, *c| {
        try std.testing.expectEqual(@as(u64, 0), g % roci.taps_row);
        c.* = @intCast(g / roci.taps_row);
    }
    return cover;
}

test "a decoder window's taps, every window of 1 to 128 rows: each other rank's slab of the three taps lands once at its rows, the receiver's own rows not at all, empty slabs send their word alone" {
    for ([_]roci.Order{ roci.identity, perm }) |order| for (1..roci.hand_rows + 1) |n| for (0..roci.world) |d| {
        const cover = try tapsCover(d, order, n);
        const own = slabp.slab(n, order[d]);
        for (cover, 0..) |c, row| {
            const want: u8 = if (row < n and !(row >= own.start and row < own.start + own.n)) 1 else 0;
            try std.testing.expectEqual(want, c);
        }
        for (0..roci.world) |g| if (slabp.slab(n, order[g]).n == 0) {
            const e = roci.buildEventX(.gt, n, 0, g, order);
            for (0..3) |k| for (0..roci.rails) |r| try std.testing.expectEqual(@as(u8, 0), e.chains[k][r].n);
        };
    };
}

test "control: the taps' plan of a window one row shorter leaves the window's last row unlanded somewhere" {
    var missed = false;
    for (0..roci.world) |d| {
        const cover = try tapsCover(d, roci.identity, 16);
        const own = slabp.slab(17, d);
        if (!(16 >= own.start and 16 < own.start + own.n) and cover[16] == 0) missed = true;
    }
    try std.testing.expect(missed);
}

test "the head's row, every window of 1 to 128 rows: from the rank whose slab holds the window's last row to the other three, whole; the other ranks send their word alone" {
    const l = roci.layout();
    for ([_]roci.Order{ roci.identity, perm }) |order| for (1..roci.hand_rows + 1) |n| {
        const owner = roci.headOwner(n);
        const sl = slabp.slab(n, owner);
        try std.testing.expect(n - 1 >= sl.start and n - 1 < sl.start + sl.n);
        var senders: usize = 0;
        for (0..roci.world) |g| {
            const e = roci.buildEventX(.hx, n, 0, g, order);
            try std.testing.expectEqual(@as(u8, 3), e.n_peers);
            var srcs: [roci.world]u8 = undefined;
            for (e.peerSlice()) |peer| try std.testing.expect(std.mem.indexOfScalar(u8, roci.sourcesOf(.hx, peer, &srcs), @intCast(g)) != null);
            var bytes: u64 = 0;
            for (0..3) |k| {
                var to_peer: u64 = 0;
                for (0..roci.rails) |r| for (e.chains[k][r].slice()) |p| {
                    try std.testing.expect(p.src == p.dst and inside(l, &.{.head_x}, p.src, p.len));
                    to_peer += p.len;
                };
                // a halves' second piece starts where the first ends
                if (to_peer != 0) {
                    try std.testing.expectEqual(l.at(.head_x), e.chains[k][0].pieces[0].src);
                    try std.testing.expectEqual(e.chains[k][0].pieces[0].src + e.chains[k][0].pieces[0].len, e.chains[k][1].pieces[0].src);
                }
                try std.testing.expect(to_peer == 0 or to_peer == roci.head_row);
                bytes += to_peer;
            }
            if (order[g] == owner) {
                senders += 1;
                try std.testing.expectEqual(@as(u64, 3 * roci.head_row), bytes);
            } else try std.testing.expectEqual(@as(u64, 0), bytes);
        }
        try std.testing.expectEqual(@as(usize, 1), senders);
    };
}

test "control: a head plan that takes the slab after (n / q) misses the last row of every window that fills its slabs" {
    var wrong: usize = 0;
    for (1..roci.hand_rows + 1) |n| {
        const k = n / slabp.q(n);
        if (k >= roci.world) {
            wrong += 1;
            continue;
        }
        const sl = slabp.slab(n, k);
        if (!(n - 1 >= sl.start and n - 1 < sl.start + sl.n)) wrong += 1;
    }
    // n = 16, 32, 48, 64 (q 16) and 64 .. 128's multiples of 32 at least
    try std.testing.expect(wrong >= 6);
}

test "four nodes: a decoder window's taps and head row land every byte at windows of 1 to 128 rows, the words keep their order" {
    for ([_]roci.Order{ roci.identity, perm }) |order| {
        const net = try Net.open(.{}, .{ .nap = false, .gathers = true, .codes = true }, order);
        defer net.close();
        try net.start();
        const classes = [_]C{ .gt, .hx, .gm };
        const ns = [_]usize{ 1, 13, 16, 17, 33, 64, 65, 100, 127, 128 };
        const run = try runAny(net, &classes, &ns, order);
        for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
        for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
        for (&net.nodes) |*nd| try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
    }
}