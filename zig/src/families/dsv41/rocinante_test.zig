//! Rocinante's host tests: the arena's layout and every event's chains as plain functions, then four nodes in one
//! process over a loopback NIC (the bytes of a chain copied into the peer's arena, then its word), one thread a node
//! for the engine's program and two a node for the proxies. The negative controls make the loopback misbehave the way
//! the hardware would if the design's ordering rules were broken, and each must be seen.

const std = @import("std");
const roci = @import("rocinante.zig");
const ti = @import("test_instruments.zig");
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
        .{ .r = .g_b, .off = 629_243_904, .len = 1_458_176 },
        .{ .r = .hand_h, .off = 630_702_080, .len = 5_242_880 },
        .{ .r = .hand_pre, .off = 635_944_960, .len = 2_048 },
        .{ .r = .g_taps, .off = 635_949_056, .len = 3_932_160 },
        .{ .r = .head_x, .off = 639_881_216, .len = 10_240 },
    };
    // (the host lane's four regions are empty in the prompt lanes' layout)
    try std.testing.expectEqual(want.len + 4, roci.n_regions);
    for ([_]R{ .rec_send, .rec_land, .rec_hs, .rec_hl }) |r| try std.testing.expectEqual(@as(u64, 0), l.bytes(r));
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
        .gb => &.{.g_b},
        .gh => &.{ .hand_h, .hand_pre },
        .gt => &.{.g_taps},
        .hx => &.{.head_x},
        .rech, .rec => &.{},
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

test "the per-pass counts of the design: 974 data writes, 512 words, 2,590,060,544 bytes each way" {
    const per_pass = [roci.n_classes]u64{ 40, 40, 2, 2, 20, 20, 17, 3, 20, 3, 1, 0, 0, 0, 0, 0 };
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
        try std.testing.expectEqual(@as(u64, 168), events);
        try std.testing.expectEqual(@as(u64, 974), writes);
        try std.testing.expectEqual(@as(u64, 512), words);
        try std.testing.expectEqual(@as(u64, 2_590_060_544), bytes);
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
    /// Control: this net's chains and words land in another net's nodes (two lanes wired into one).
    cross_to: ?*Net = null,
    /// The nodes' arenas are the host lane's (the partner exchange of the Engram records), not a pass's.
    host: bool = false,
    /// Control (with word_first): every chain's bytes held back, the words not, until the test opens the gate (Net.openGate),
    /// so a consumer that trusts the word looks before any byte has landed, whatever else the machine is doing.
    hold_data: bool = false,
    /// Control: a rail's whole chain (word and bytes) held back the same way.
    hold_rail: ?u8 = null,
    /// How the arrival is signalled: the word written whole, or the card's ATOMIC fetch-and-add of 1 on it.
    signal: roci.Signal = .write,
    /// The atomic's read-modify-write is torn open: the card reads the word, takes this long, then writes the old value + 1 back (an
    /// atomic of the card that a CPU store in between does not see: atomic_cap 1), if the card still has the control block.
    torn_ns: u64 = 0,
    /// The torn atomic writes back only after the word has been stored the abort value (the failure's store came first), `torn_ns` later.
    torn_after_abort: bool = false,
    /// Control: a card that is never fenced off the control block (fail_fn returns with its atomics still landing).
    no_fence: bool = false,
    /// The fence takes this long (the queue pairs to error, the registration gone): atomics that land meanwhile still land, before fail_fn returns.
    fence_ns: u64 = 0,
};

/// A copy a held post has not made yet.
const Held = struct { dst: [*]u8, src: [*]const u8, len: usize, word: ?*u64, value: u64 };

const Node = struct {
    arena: []align(std.heap.page_size_min) u8,
    ctl: *roci.Control,
    t: *roci.Transport = undefined,
    net: *Net = undefined,
    me: u8 = 0,
    gone: bool = false, // the test closed this node's transport itself
};

const Net = struct {
    nodes: [roci.world]Node = undefined,
    mode: Mode = .{},
    failed_nic: [roci.world]std.atomic.Value(u32) = @splat(.init(0)),
    /// Whether each node's card still has its control block (fail_fn fences it off).
    card_open: [roci.world]std.atomic.Value(u32) = @splat(.init(1)),
    /// Atomics whose returned old value was not the signal's count minus one (a signal sent twice or lost), and atomics posted.
    atomic_off: std.atomic.Value(u32) = .init(0),
    atomics: std.atomic.Value(u32) = .init(0),
    /// Posts in progress on each node (a proxy inside its NIC call), and whether one was when that node's fail_fn ran.
    in_post: [roci.world]std.atomic.Value(u32) = @splat(.init(0)),
    post_at_fail: [roci.world]std.atomic.Value(u32) = @splat(.init(0)),
    gate: std.atomic.Value(u32) = .init(0), // 1: the held posts have been released
    gate_lock: std.atomic.Value(u32) = .init(0),
    held: std.ArrayList(Held) = .empty,
    /// The card's unfinished read-modify-writes (Mode.torn_ns), joined before the arenas go.
    torn: std.ArrayList(std.Thread) = .empty,

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
        const total = roci.layoutOf(if (mode.host) .host else .prompt).total;
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
        for (&n.nodes) |*nd| if (!nd.gone) nd.t.deinit();
        for (n.torn.items) |th| th.join();
        n.torn.deinit(gpa);
        n.held.deinit(gpa);
        for (&n.nodes) |*nd| {
            std.posix.munmap(nd.arena);
            gpa.destroy(nd.ctl);
        }
        gpa.destroy(n);
    }

    /// The arrival signal on `dst`'s word `w` as the card does it: a whole-word store of `word`, or an ATOMIC fetch-and-add of 1 whose
    /// returned old value must be `word - 1` (the count of signals before it).
    fn signalWord(me: *Node, dst: *Node, w: *u64, word: u64) void {
        const m = me.net.mode;
        if (m.signal == .write) {
            @atomicStore(u64, w, word, .release);
            return;
        }
        if (m.torn_ns == 0) {
            _ = me.net.atomics.fetchAdd(1, .monotonic);
            const old = @atomicRmw(u64, w, .Add, 1, .acq_rel);
            if (old != word - 1) _ = me.net.atomic_off.fetchAdd(1, .monotonic);
            return;
        }
        // the card reads the word now and writes it back later, on its own, after the post call has returned (as a card does)
        const old = @atomicLoad(u64, w, .acquire);
        _ = me.net.atomics.fetchAdd(1, .monotonic); // (counted once the card has read: a failure that waits for it stores after the read)
        const th = std.Thread.spawn(.{}, tornWrite, .{ dst, w, old, m.torn_ns, m.torn_after_abort }) catch @panic("torn atomic");
        me.net.lockGate();
        defer me.net.unlockGate();
        me.net.torn.append(gpa, th) catch @panic("torn atomic");
    }

    fn tornWrite(dst: *Node, w: *u64, old: u64, ns: u64, after_abort: bool) void {
        if (after_abort) {
            var spins: u32 = 0;
            while (@atomicLoad(u64, w, .acquire) != roci.abort_value and spins < 1_000_000) : (spins += 1) sleep(10_000);
        }
        sleep(ns);
        if (dst.net.card_open[dst.me].load(.acquire) != 0) @atomicStore(u64, w, old +% 1, .release);
    }

    fn post(ctx: *anyopaque, rail: u8, peer: u8, class: C, chain: *const roci.Chain, word: u64) roci.PostError!void {
        const me: *Node = @ptrCast(@alignCast(ctx));
        const m = me.net.mode;
        _ = me.net.in_post[me.me].fetchAdd(1, .acq_rel);
        defer _ = me.net.in_post[me.me].fetchSub(1, .acq_rel);
        if (me.net.failed_nic[me.me].load(.acquire) != 0) return error.QpError;
        if (m.broken == me.me) return error.PostFailed;
        if (m.silent == me.me) return;
        const dst = if (m.cross_to) |c| &c.nodes[peer] else &me.net.nodes[peer];
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
            signalWord(me, dst, w, word);
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
        if (!m.word_first) signalWord(me, dst, w, word);
    }

    fn failNic(ctx: *anyopaque) void {
        const me: *Node = @ptrCast(@alignCast(ctx));
        me.net.failed_nic[me.me].store(1, .release);
        _ = me.net.post_at_fail[me.me].fetchAdd(me.net.in_post[me.me].load(.acquire), .acq_rel);
        // the contract: fail_fn returns when the card can no longer write the control block
        if (me.net.mode.no_fence) return;
        sleep(me.net.mode.fence_ns);
        me.net.card_open[me.me].store(0, .release);
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
    /// Added to every generation's pattern tag: two lanes' bytes differ, so bytes that cross lanes are seen.
    gen_base: u64 = 0,
    timeout_ns: u64 = 10 * std.time.ns_per_s,
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
    const s: FakeStream = .{ .only_rail_a = run.only_a, .timeout_ns = run.timeout_ns };
    for (0..run.gens) |gen_| {
        const gen = gen_ + run.gen_base;
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
    return runProgramOn(net, classes, gens, order, only_a, 0, 10 * std.time.ns_per_s);
}

fn runProgramOn(net: *Net, classes: []const C, gens: u64, order: roci.Order, only_a: bool, gen_base: u64, timeout_ns: u64) !Run {
    var bar: Barrier = .{};
    var run: Run = .{ .net = net, .bar = &bar, .classes = classes, .gens = gens, .order = order, .only_a = only_a, .gen_base = gen_base, .timeout_ns = timeout_ns };
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

// -- the gathers -----------------------------------------------------------------------------------------------------

test "four nodes: a layer's gathers (qr and y, x, the routing's four tensors, the top-k, the boundary's rows) land every slab at its place on the three other nodes, both caps and orders" {
    for ([_]roci.Order{ roci.identity, perm }) |order| {
        const net = try Net.open(.{}, .{ .nap = false, .gathers = true }, order);
        defer net.close();
        try net.start();
        const classes = [_]C{ .gf, .ga, .gm, .gi, .gb };
        const run = try runProgram(net, &classes, 4, order, false);
        for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
        for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
        for (&net.nodes) |*nd| {
            const c = nd.t.counters();
            try std.testing.expectEqual(@as(u64, classes.len * 4), c.wait_groups);
            try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
        }
    }
}

test "four nodes: wo_b's and the pack's codes land every byte of the column partner's part, generation after generation, both caps and orders" {
    for ([_]roci.Order{ roci.identity, perm }) |order| {
        const net = try Net.open(.{}, .{ .nap = false, .codes = true }, order);
        defer net.close();
        try net.start();
        const run = try runProgram(net, &.{ .wob, .pack }, 4, order, false);
        for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
        for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
        for (&net.nodes) |*nd| {
            const c = nd.t.counters();
            try std.testing.expectEqual(@as(u64, 8), c.wait_groups);
            try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
        }
    }
}

test "control: a NIC that writes the pack's word before its codes lets a consumer read stale codes" {
    const net = try Net.open(.{ .word_first = true, .hold_data = true }, .{ .nap = false, .codes = true }, roci.identity);
    defer net.close();
    try net.start();
    const run = try runProgram(net, &.{.pack}, 1, roci.identity, false);
    var bad: u64 = 0;
    for (run.bad) |b| bad += b;
    try std.testing.expect(bad > 0);
}

test "the codes' parts: the send region's bytes to the same offset of the land region, in two halves on the rails, to the column partner alone" {
    for ([_]roci.Order{ roci.identity, perm }) |order| for (0..roci.world) |g| {
        const l = roci.layout();
        for ([_]struct { c: C, s: roci.Region, d: roci.Region, bytes: u64 }{ .{ .c = .wob, .s = .wob_send, .d = .wob_land, .bytes = 4096 * 2304 }, .{ .c = .pack, .s = .pack_send, .d = .pack_land, .bytes = 7 * 4096 * 720 } }) |x| {
            const e = roci.buildEvent(x.c, 4096, g, order);
            try std.testing.expectEqual(@as(u8, 1), e.n_peers);
            try std.testing.expectEqual(@as(u8, @intCast(g ^ 2)), e.peers[0]);
            const a = e.chains[0][0].pieces[0];
            const b = e.chains[0][1].pieces[0];
            try std.testing.expectEqual(l.at(x.s), a.src);
            try std.testing.expectEqual(l.at(x.d), a.dst);
            try std.testing.expectEqual(a.src + a.len, b.src);
            try std.testing.expectEqual(x.bytes, a.len + b.len);
            try std.testing.expectEqual(x.bytes, l.bytes(x.s));
            try std.testing.expectEqual(@as(u64, 0), a.len % roci.page);
        }
    };
}

test "control: a NIC that writes the boundary's word before its rows lets a consumer read stale rows" {
    const net = try Net.open(.{ .word_first = true, .hold_data = true }, .{ .nap = false, .gathers = true }, roci.identity);
    defer net.close();
    try net.start();
    const run = try runProgram(net, &.{.gb}, 1, roci.identity, false);
    var bad: u64 = 0;
    for (run.bad) |b| bad += b;
    try std.testing.expect(bad > 0);
}

test "the boundary's slab: one node's rows of the four cache arrays, array after array, at its rank's place, in halves on the rails" {
    for ([_]roci.Order{ roci.identity, perm }) |order| for (0..roci.world) |g| {
        const l = roci.layout();
        const e = roci.buildEvent(.gb, 4096, g, order);
        try std.testing.expectEqual(@as(u8, 3), e.n_peers);
        const slab: u64 = 1024 * roci.gb_row;
        try std.testing.expectEqual(@as(u64, 364_544), slab);
        for (0..3) |k| {
            try std.testing.expectEqual(@as(u8, 1), e.chains[k][0].n);
            try std.testing.expectEqual(@as(u8, 1), e.chains[k][1].n);
            const a = e.chains[k][0].pieces[0];
            const b = e.chains[k][1].pieces[0];
            try std.testing.expectEqual(l.at(.g_b) + order[g] * slab, a.src);
            try std.testing.expectEqual(a.src, a.dst);
            try std.testing.expectEqual(a.src + a.len, b.src);
            try std.testing.expectEqual(slab, a.len + b.len);
        }
    };
}

/// A word of a stamp: the tag names the event instance and its sender, the index the word.
fn stampWord(tag: u64, i: usize) u64 {
    return tag *% 0x9e3779b97f4a7c15 +% @as(u64, i);
}

const stamp_stride: usize = 509;

/// Writes the stamp's words at every 509th word of `buf` and at its last (the other words are never read).
fn stampPiece(buf: []u8, tag: u64) void {
    const words = std.mem.bytesAsSlice(u64, buf[0 .. buf.len / 8 * 8]);
    var i: usize = 0;
    while (i < words.len) : (i += stamp_stride) words[i] = stampWord(tag, i);
    if (words.len > 0) words[words.len - 1] = stampWord(tag, words.len - 1);
}

/// The stamped words of `buf` that do not hold `tag`'s.
fn badPiece(buf: []const u8, tag: u64) u64 {
    const words = std.mem.bytesAsSlice(u64, buf[0 .. buf.len / 8 * 8]);
    var bad: u64 = 0;
    var i: usize = 0;
    while (i < words.len) : (i += stamp_stride) bad += @intFromBool(words[i] != stampWord(tag, i));
    if (words.len > 0) bad += @intFromBool(words[words.len - 1] != stampWord(tag, words.len - 1));
    return bad;
}

/// The tag of the `n`th event of the program sent by node `s`, for the piece starting at source offset `src`.
fn eventTag(n: u64, s: usize, src: u64) u64 {
    return (n << 40) ^ (@as(u64, s) << 36) ^ src;
}

/// The layers' program of the gathers, the codes and the owner sums over shared regions, with the readers where the engine has them: each
/// rank stamps its sources before a send, checks the landed pieces at the wait and again where a later reader reads them (the
/// top-k six layers on), while ranks sleep random times between events so any rank can run ahead of the others.
const Layers = struct {
    net: *Net,
    layers: usize,
    /// A rank that waits for nothing and never sleeps: it runs ahead of the fences (the control).
    runaway: ?usize = null,
    /// The control's order: the other ranks start their program once the runaway has posted every event of every layer, so
    /// each landed piece they check has been written over by a later event, whatever the machine is doing.
    ahead_done: std.atomic.Value(u32) = .init(0),
    bad: [roci.world]std.atomic.Value(u64) = @splat(.init(0)),
    err: [roci.world]?anyerror = @splat(null),

    const a_layers = [_]usize{ 2, 8 };

    fn isA(l: usize) bool {
        return std.mem.indexOfScalar(usize, &a_layers, l) != null;
    }
};

const Seen = struct {
    /// The sequence number of the latest event of each class this rank sent.
    n: [roci.n_classes]u64 = @splat(0),
};

fn landed(r: *Layers, nd: *Node, class: C, n: u64, cap: usize, order: roci.Order) u64 {
    var bad: u64 = 0;
    var srcs: [roci.world]u8 = undefined;
    for (roci.sourcesOf(class, nd.me, &srcs)) |s| {
        const e = roci.buildEvent(class, cap, s, order);
        const k = std.mem.indexOfScalar(u8, e.peerSlice(), nd.me).?;
        for (0..roci.rails) |rail| for (e.chains[k][rail].slice()) |p| {
            bad += badPiece(nd.arena[p.dst..][0..p.len], eventTag(n, s, p.src));
        };
    }
    _ = r;
    return bad;
}

fn layersMain(r: *Layers, g: usize) void {
    layersBody(r, g) catch |e| {
        r.err[g] = e;
    };
}

fn layersBody(r: *Layers, g: usize) !void {
    const nd = &r.net.nodes[g];
    const s: FakeStream = .{};
    const cap: usize = 2048;
    const order = perm;
    var rng = std.Random.DefaultPrng.init(0x5eed0000 + g);
    var seen: Seen = .{};
    var seq: u64 = 0;
    var last_i: ?u64 = null; // the latest top-k event's sequence number
    var last_i_layer: usize = 0;
    const run_ahead = r.runaway != null and r.runaway.? == g;
    if (r.runaway != null and !run_ahead) {
        var waited: u64 = 0;
        while (r.ahead_done.load(.acquire) == 0) : (waited += 1) {
            if (waited > 20_000) return error.RunawayNeverFinished; // (10 s)
            sleep(500_000);
        }
    }
    defer if (run_ahead) r.ahead_done.store(1, .release);
    const events = struct {
        fn send(rr: *Layers, nn: *Node, ss: FakeStream, class: C, sq: *u64, sn: *Seen, rn: *std.Random.DefaultPrng, ahead: bool, c: usize, o: roci.Order) !void {
            _ = rr;
            sq.* += 1;
            sn.n[@backingInt(class)] = sq.*;
            if (!ahead and rn.random().uintLessThan(u32, 3) == 0) sleep(rn.random().uintLessThan(u64, 1_500_000));
            const e = roci.buildEvent(class, c, nn.me, o);
            for (0..e.n_peers) |k| for (0..roci.rails) |rail| for (e.chains[k][rail].slice()) |p| stampPiece(nn.arena[p.src..][0..p.len], eventTag(sq.*, nn.me, p.src));
            try nn.t.send(ss, class, 1);
        }
    };
    for (0..r.layers) |l| {
        // attention's front: x gathered at an index layer, else qr and y; the top-k at an index layer
        if (Layers.isA(l)) {
            try events.send(r, nd, s, .ga, &seq, &seen, &rng, run_ahead, cap, order);
            if (!run_ahead) try nd.t.wait(s, .ga);
            if (!run_ahead) _ = r.bad[g].fetchAdd(landed(r, nd, .ga, seen.n[@backingInt(C.ga)], cap, order), .monotonic);
            try events.send(r, nd, s, .gi, &seq, &seen, &rng, run_ahead, cap, order);
            if (!run_ahead) try nd.t.wait(s, .gi);
            last_i = seen.n[@backingInt(C.gi)];
            last_i_layer = l;
        } else {
            try events.send(r, nd, s, .gf, &seq, &seen, &rng, run_ahead, cap, order);
            if (!run_ahead) try nd.t.wait(s, .gf);
        }
        if (!run_ahead) {
            if (!Layers.isA(l)) _ = r.bad[g].fetchAdd(landed(r, nd, .gf, seen.n[@backingInt(C.gf)], cap, order), .monotonic);
            // the top-k's readers: this layer and the five after it
            if (last_i) |n| if (l - last_i_layer < 6) {
                _ = r.bad[g].fetchAdd(landed(r, nd, .gi, n, cap, order), .monotonic);
            };
        }
        // wo_b's codes, then the attention's owner sum
        try events.send(r, nd, s, .wob, &seq, &seen, &rng, run_ahead, cap, order);
        if (!run_ahead) try nd.t.wait(s, .wob);
        if (!run_ahead) _ = r.bad[g].fetchAdd(landed(r, nd, .wob, seen.n[@backingInt(C.wob)], cap, order), .monotonic);
        for ([_]C{ .own1, .own2 }) |c| {
            try events.send(r, nd, s, c, &seq, &seen, &rng, run_ahead, cap, order);
            if (!run_ahead) try nd.t.wait(s, c);
            if (!run_ahead) _ = r.bad[g].fetchAdd(landed(r, nd, c, seen.n[@backingInt(c)], cap, order), .monotonic);
        }
        // the routing's gather, then the MoE's owner sum
        try events.send(r, nd, s, .gm, &seq, &seen, &rng, run_ahead, cap, order);
        if (!run_ahead) try nd.t.wait(s, .gm);
        if (!run_ahead) _ = r.bad[g].fetchAdd(landed(r, nd, .gm, seen.n[@backingInt(C.gm)], cap, order), .monotonic);
        // the experts' pack, then the MoE's owner sum
        try events.send(r, nd, s, .pack, &seq, &seen, &rng, run_ahead, cap, order);
        if (!run_ahead) try nd.t.wait(s, .pack);
        if (!run_ahead) _ = r.bad[g].fetchAdd(landed(r, nd, .pack, seen.n[@backingInt(C.pack)], cap, order), .monotonic);
        for ([_]C{ .own1, .own2 }) |c| {
            try events.send(r, nd, s, c, &seq, &seen, &rng, run_ahead, cap, order);
            if (!run_ahead) try nd.t.wait(s, c);
            if (!run_ahead) _ = r.bad[g].fetchAdd(landed(r, nd, c, seen.n[@backingInt(c)], cap, order), .monotonic);
        }
        // the routing's tensors are read by the experts until the MoE's sum: checked again where the combine reads them
        if (!run_ahead) _ = r.bad[g].fetchAdd(landed(r, nd, .gm, seen.n[@backingInt(C.gm)], cap, order), .monotonic);
    }
}

fn runLayers(layers: usize, runaway: ?usize) !u64 {
    const net = try Net.open(.{}, .{ .nap = false, .gathers = true }, perm);
    defer net.close();
    try net.start();
    var r: Layers = .{ .net = net, .layers = layers, .runaway = runaway };
    var th: [roci.world]std.Thread = undefined;
    for (&th, 0..) |*t, g| t.* = try std.Thread.spawn(.{}, layersMain, .{ &r, g });
    for (th) |t| t.join();
    var bad: u64 = 0;
    for (r.err, 0..) |e, g| if (runaway == null or runaway.? != g) try std.testing.expectEqual(@as(?anyerror, null), e);
    for (&r.bad, 0..) |*b, g| if (runaway == null or runaway.? != g) {
        bad += b.load(.monotonic);
    };
    return bad;
}

test "the dataflow of the gathers and the codes: shared regions across layers, ranks skewed by random sleeps, no landed slab is overwritten before its last reader" {
    try std.testing.expectEqual(@as(u64, 0), try runLayers(11, null));
}

test "control: a rank that waits for no fence and never sleeps runs ahead and overwrites slabs the others still read" {
    // rank 2 posts every event of every layer at once; the others, delayed by their sleeps, read the top-k of an index layer
    // after the next index layer's slabs have landed on top of it
    const bad = try runLayers(11, 2);
    try std.testing.expect(bad > 0);
}

// -- events of any row count -----------------------------------------------------------------------------------------------------

const slabp = @import("slab_plan.zig");

fn sameEvent(a: roci.Event, b: roci.Event) !void {
    try std.testing.expectEqual(a.class, b.class);
    try std.testing.expectEqual(a.n_peers, b.n_peers);
    try std.testing.expectEqualSlices(u8, a.peerSlice(), b.peerSlice());
    for (0..a.n_peers) |k| for (0..roci.rails) |r| {
        try std.testing.expectEqual(a.chains[k][r].n, b.chains[k][r].n);
        for (a.chains[k][r].slice(), b.chains[k][r].slice()) |x, y| try std.testing.expectEqual(x, y);
    };
}

test "the plan of any row count is the two fixed plans at 4,096 and 2,048 rows, piece for piece, for every class, node and row order" {
    for ([_]roci.Order{ roci.identity, perm }) |order| for (0..roci.world) |g| for (0..roci.n_classes) |ci| {
        const class: C = @fromBackingInt(@intCast(ci));
        if (class == .gh or class == .gt or class == .hx) continue; // (no full-cap plan: their rows are a decoder window's)
        for ([_]usize{ 4096, 2048 }) |cap| try sameEvent(roci.buildEvent(class, cap, g, order), roci.buildEventN(class, cap, g, order));
    };
}

test "control: the plan of 4,095 rows is not the plan of 4,096" {
    const a = roci.buildEvent(.own1, 4096, 0, roci.identity);
    const b = roci.buildEventN(.own1, 4095, 0, roci.identity);
    try std.testing.expect(a.chains[0][1].pieces[0].len != b.chains[0][1].pieces[0].len or a.chains[0][0].pieces[0].len != b.chains[0][0].pieces[0].len);
}

/// What a node receives of an event class, summed over its sources: bytes into each region, and every piece inside its regions.
fn receivedBytes(class: C, n: usize, d: usize, order: roci.Order) !struct { bytes: u64, pieces: usize } {
    const l = roci.layout();
    var bytes: u64 = 0;
    var pieces: usize = 0;
    var srcs: [roci.world]u8 = undefined;
    for (roci.sourcesOf(class, d, &srcs)) |s| {
        const e = roci.buildEventN(class, n, s, order);
        const k = std.mem.indexOfScalar(u8, e.peerSlice(), @intCast(d)).?;
        for (0..roci.rails) |r| for (e.chains[k][r].slice()) |p| {
            try std.testing.expect(inside(l, srcRegion(class), p.src, p.len));
            try std.testing.expect(inside(l, dstRegion(class), p.dst, p.len));
            bytes += p.len;
            pieces += 1;
        };
    }
    return .{ .bytes = bytes, .pieces = pieces };
}

test "every row count from 1 to 4,096: the slabs of a gather land every row of every other node once, the owner parts carry the partners' rows, the codes all rows, empty slabs send no data" {
    const rows_of = [_]struct { c: C, row: u64 }{ .{ .c = .gf, .row = 2560 + 1024 }, .{ .c = .ga, .row = 10240 }, .{ .c = .gm, .row = 5120 + 640 + 28 + 28 }, .{ .c = .gi, .row = 4096 }, .{ .c = .gb, .row = roci.gb_row } };
    var n: usize = 1;
    while (n <= 4096) : (n += if (n < 130 or n > 4000) 1 else 7) {
        for ([_]roci.Order{ roci.identity, perm }) |order| for (0..roci.world) |d| {
            for (rows_of) |x| {
                const got = try receivedBytes(x.c, n, d, order);
                // every row the other three nodes own, once
                try std.testing.expectEqual((n - slabp.slab(n, order[d]).n) * x.row, got.bytes);
            }
            // the owner sums: part 1 brings this node's rows and its column partner's from the row partner, part 2 the pair's sums at this node's rows
            const o1 = try receivedBytes(.own1, n, d, order);
            try std.testing.expectEqual((slabp.slab(n, order[d]).n + slabp.slab(n, order[d ^ 2]).n) * 2560 * 4, o1.bytes);
            const o2 = try receivedBytes(.own2, n, d, order);
            try std.testing.expectEqual(slabp.slab(n, order[d]).n * 2560 * 2, o2.bytes);
            const w = try receivedBytes(.wob, n, d, order);
            try std.testing.expectEqual(n * 2304, w.bytes);
            const p = try receivedBytes(.pack, n, d, order);
            try std.testing.expectEqual(7 * n * 720, p.bytes);
            // a node with no rows (n small) sends no data WRITE of a gather, only its words
            for (0..roci.world) |g| if (slabp.slab(n, order[g]).n == 0) {
                const e = roci.buildEventN(.gm, n, g, order);
                try std.testing.expectEqual(@as(u8, 3), e.n_peers);
                for (0..3) |k| for (0..roci.rails) |r| try std.testing.expectEqual(@as(u8, 0), e.chains[k][r].n);
            };
        };
    }
}

const Run2 = struct {
    net: *Net,
    bar: *Barrier,
    classes: []const C,
    ns: []const usize,
    order: roci.Order,
    bad: [roci.world]u64 = @splat(0),
    err: [roci.world]?anyerror = @splat(null),
};

fn runAny(net: *Net, classes: []const C, ns: []const usize, order: roci.Order) !Run2 {
    var bar: Barrier = .{};
    var run: Run2 = .{ .net = net, .bar = &bar, .classes = classes, .ns = ns, .order = order };
    var th: [roci.world]std.Thread = undefined;
    for (&th, 0..) |*t, g| t.* = try std.Thread.spawn(.{}, rankAny2, .{ &run, g });
    for (th) |t| t.join();
    return run;
}

fn rankAny2(run: *Run2, g: usize) void {
    rankAny2Body(run, g) catch |e| {
        run.err[g] = e;
    };
}

fn rankAny2Body(run: *Run2, g: usize) !void {
    const nd = &run.net.nodes[g];
    const s: FakeStream = .{};
    for (run.ns) |n| for (run.classes) |class| {
        const e = roci.buildEventN(class, n, nd.me, run.order);
        for (0..e.n_peers) |k| for (0..roci.rails) |r| for (e.chains[k][r].slice()) |p| pattern(nd.arena[p.src..][0..p.len], n *% 1000003 +% @as(u64, nd.me) *% 7919 +% p.src);
        run.bar.wait(roci.world);
        try nd.t.sendN(s, class, n);
        try nd.t.wait(s, class);
        var srcs: [roci.world]u8 = undefined;
        for (roci.sourcesOf(class, nd.me, &srcs)) |src| {
            const se = roci.buildEventN(class, n, src, run.order);
            const k = std.mem.indexOfScalar(u8, se.peerSlice(), nd.me).?;
            for (0..roci.rails) |r| for (se.chains[k][r].slice()) |p| {
                const want = try gpa.alloc(u8, p.len);
                defer gpa.free(want);
                pattern(want, n *% 1000003 +% @as(u64, src) *% 7919 +% p.src);
                const m = p.len / 8 * 8;
                if (!std.mem.eql(u8, want[0..m], nd.arena[p.dst..][0..m])) run.bad[g] += 1;
            };
        }
        run.bar.wait(roci.world);
    };
}

test "four nodes: every class at row counts from 1 to 4,096 in turn (zero-row slabs included) lands every byte, the words keep their order" {
    for ([_]roci.Order{ roci.identity, perm }) |order| {
        const net = try Net.open(.{}, .{ .nap = false, .gathers = true, .codes = true }, order);
        defer net.close();
        try net.start();
        const classes = [_]C{ .own1, .own2, .wob, .pack, .gf, .ga, .gm, .gi, .gb, .eng1, .eng2 };
        const ns = [_]usize{ 13, 1, 500, 4096, 63, 64, 65, 2048, 1087, 17, 3000 };
        const run = try runAny(net, &classes, &ns, order);
        for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
        for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
        for (&net.nodes) |*nd| try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
    }
}

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

test "a rank whose link closed: the watch fails the lane it waits on, the queued wait is released and the failure names the peer (the watchdog is off)" {
    const link = @import("link.zig");
    // the watchdog's limit is far off: only the watch can end the wait inside the test's bound
    const net = try Net.open(.{ .silent = 1 }, .{ .nap = false, .dead_ms = 100_000 }, roci.identity);
    defer net.close();
    try net.start();
    const nd = &net.nodes[0];
    var fds: [2]std.c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &fds));
    defer _ = std.c.close(fds[0]);
    const Hook = struct {
        fn gone(ctx: ?*anyopaque, rank: u32) void {
            const t: *roci.Transport = @ptrCast(@alignCast(ctx.?));
            t.fail(.peer, .own1, @intCast(rank));
        }
        fn exit(_: *link.Watch, _: u32) void {} // (the test's stand-in for the process's end)
    };
    var w: link.Watch = .{ .rank = 0, .grace_ms = 100, .stop_hook = Hook.gone, .ctx = nd.t, .on_gone = Hook.exit };
    try w.add(.{ .fd = fds[0] }, 1);
    try w.start();
    defer w.stop();
    const s: FakeStream = .{ .timeout_ns = 8 * std.time.ns_per_s };
    try nd.t.send(s, .own1, 0);
    const t0 = std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds;
    _ = std.c.close(fds[1]); // rank 1 is gone: the link closes
    try nd.t.wait(s, .own1); // returns: the stop's abort released it (rank 1's words never come)
    const took_ms = @divTrunc(std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds - t0, std.time.ns_per_ms);
    try std.testing.expect(took_ms < 4000);
    const f = nd.t.failure().?;
    try std.testing.expectEqual(roci.FailReason.peer, f.reason);
    try std.testing.expectEqual(@as(u8, 1), f.peer);
    try std.testing.expectError(error.RociFailed, nd.t.check());
    for (&nd.ctl.recv) |*c| for (c) |*sw| for (sw) |*x| try std.testing.expectEqual(roci.abort_value, x.v);
}

// -- the atomic arrival signal --------------------------------------------------------------------------------------

test "TF_DS_ROCI_SIGNAL's text: unset, empty and atomic are the fetch-and-add (the guaranteed arrival signal), write the second WRITE, anything else is refused" {
    try std.testing.expectEqual(@as(?roci.Signal, .atomic), roci.signalFrom(null));
    try std.testing.expectEqual(@as(?roci.Signal, .atomic), roci.signalFrom(""));
    try std.testing.expectEqual(@as(?roci.Signal, .write), roci.signalFrom("write"));
    try std.testing.expectEqual(@as(?roci.Signal, .atomic), roci.signalFrom("atomic"));
    try std.testing.expectEqual(@as(?roci.Signal, .atomic), roci.signalFrom(" atomic "));
    for ([_][]const u8{ "Atomic", "word", "1", "atomics", "cas" }) |bad| try std.testing.expectEqual(@as(?roci.Signal, null), roci.signalFrom(bad));
    // what a start without the setting runs is the atomic (D79); the Settings structs' own default stays the WRITE (the host tests' fake cards)
    try std.testing.expectEqual(roci.Signal.atomic, roci.env_default_signal);
    try std.testing.expectEqual(roci.Signal.write, roci.default_signal);
}

test "the start line names the arrival signal" {
    var buf: [1024]u8 = undefined;
    for ([_]roci.Signal{ .write, .atomic }) |sig| {
        var w = std.Io.Writer.fixed(&buf);
        try roci.startLine(&w, 2, "mlx5_0,mlx5_2", 3, .{}, sig);
        const want = if (sig == .write) "arrival signal an 8-byte RDMA WRITE after the data" else "arrival signal an RDMA ATOMIC fetch-and-add after the data";
        try std.testing.expect(std.mem.indexOf(u8, w.buffered(), want) != null);
    }
}

test "atomic signal, four nodes: every byte lands, each word advances by one a use (the old value an atomic returns is the count before it), both caps and orders" {
    for ([_]roci.Order{ roci.identity, perm }) |order| {
        const net = try Net.open(.{ .signal = .atomic }, .{ .nap = false }, order);
        defer net.close();
        try net.start();
        const classes = [_]C{ .own1, .own2, .wob, .gm, .gf };
        const gens: u64 = 4;
        const run = try runProgram(net, &classes, gens, order, false);
        for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
        for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
        var words: u64 = 0;
        for (&net.nodes) |*nd| {
            const c = nd.t.counters();
            words += c.words;
            try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
            // a word is the number of uses of its class: exactly one atomic each, none lost, none twice
            for (classes) |cl| {
                var srcs: [roci.world]u8 = undefined;
                for (roci.sourcesOf(cl, nd.me, &srcs)) |src| for (0..roci.rails) |r| try std.testing.expectEqual(gens, nd.ctl.recv[@backingInt(cl)][src][r].v);
            }
        }
        // every word a proxy counted went out as an atomic, and each returned the count before it
        try std.testing.expectEqual(words, net.atomics.load(.acquire));
        try std.testing.expectEqual(@as(u32, 0), net.atomic_off.load(.acquire));
    }
}

test "atomic signal, six generations of an owner sum and a gather: the words keep their order and the bytes land" {
    const net = try Net.open(.{ .signal = .atomic }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    const run = try runProgram(net, &.{ .own1, .gm }, 6, roci.identity, false);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    try std.testing.expectEqual(@as(u32, 0), net.atomic_off.load(.acquire));
}

test "control: an atomic the card does before the data lets a consumer read stale bytes" {
    const net = try Net.open(.{ .signal = .atomic, .word_first = true, .hold_data = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    const run = try runProgram(net, &.{.own2}, 1, roci.identity, false);
    var bad: u64 = 0;
    for (run.bad) |b| bad += b;
    try std.testing.expect(bad > 0);
}

/// Node 0 sends class own1 to its partner while the partner's card has the atomic open (read, `torn_ns`, write back old + 1);
/// once both cards have read their word, the partner fails (its fence takes `fence_ns`: the atomic lands inside it). Returns the partner's words of that class after the card has finished.
fn failDuringAtomic(mode: Mode, cap_ns: u64) ![roci.rails]u64 {
    // (no look at the words after the store: these two tests are about the fence alone)
    const net = try Net.open(mode, .{ .nap = false, .dead_ms = 100_000, .abort_rounds = 0 }, roci.identity);
    defer net.close();
    try net.start();
    const e = roci.buildEvent(.own1, 4096, 0, roci.identity);
    const p = e.peerSlice()[0];
    const s: FakeStream = .{};
    try net.nodes[0].t.send(s, .own1, 0);
    var spins: u32 = 0; // the atomics are open (torn_ns) on both rails: each card has read its word
    while (net.atomics.load(.acquire) < roci.rails and spins < 5_000_000) : (spins += 1) std.atomic.spinLoopHint();
    try std.testing.expect(net.atomics.load(.acquire) >= roci.rails);
    net.nodes[p].t.fail(.refused, .own1, 0);
    sleep(cap_ns); // the card finishes what it had begun
    var out: [roci.rails]u64 = undefined;
    for (&out, 0..) |*o, r| o.* = @atomicLoad(u64, &net.nodes[p].ctl.recv[@backingInt(C.own1)][0][r].v, .acquire);
    return out;
}

test "a failure with atomics in flight: the card is fenced off before the abort values are stored, so none lands over them and every wait is released" {
    const got = try failDuringAtomic(.{ .signal = .atomic, .torn_ns = 100 * std.time.ns_per_ms, .fence_ns = 100 * std.time.ns_per_ms }, 300 * std.time.ns_per_ms);
    for (got) |v| try std.testing.expectEqual(roci.abort_value, v);
}

test "control: a card that is never fenced finishes its atomic over the abort value, and the wait of the failed rank is not released" {
    const got = try failDuringAtomic(.{ .signal = .atomic, .torn_ns = 100 * std.time.ns_per_ms, .fence_ns = 100 * std.time.ns_per_ms, .no_fence = true }, 300 * std.time.ns_per_ms);
    for (got) |v| try std.testing.expectEqual(@as(u64, 1), v); // the atomic's old + 1 over the abort value: a count of one, below the generations a later wait names
}

/// Node 0 sends own1 to its partner whose card is NOT fenced and finishes an atomic it had begun (read, `torn_ns`, write back old + 1) just after
/// the partner's failure has stored the abort values. Returns the partner's words and how many the failure's look put back.
fn failBehindAtomic(set: roci.Settings) !struct { words: [roci.rails]u64, restored: u64 } {
    const net = try Net.open(.{ .signal = .atomic, .torn_ns = 200 * std.time.ns_per_us, .torn_after_abort = true, .no_fence = true }, set, roci.identity);
    defer net.close();
    try net.start();
    const e = roci.buildEvent(.own1, 4096, 0, roci.identity);
    const p = e.peerSlice()[0];
    try net.nodes[0].t.send(FakeStream{}, .own1, 0);
    var spins: u32 = 0;
    while (net.atomics.load(.acquire) < roci.rails and spins < 5_000_000) : (spins += 1) std.atomic.spinLoopHint(); // both rails' cards have read their word
    net.nodes[p].t.fail(.refused, .own1, 0);
    sleep(30 * std.time.ns_per_ms); // the card has finished
    var out: [roci.rails]u64 = undefined;
    for (&out, 0..) |*o, r| o.* = @atomicLoad(u64, &net.nodes[p].ctl.recv[@backingInt(C.own1)][0][r].v, .acquire);
    return .{ .words = out, .restored = net.nodes[p].t.abort_restored };
}

test "an atomic the card finishes after the abort store is put right: the failure looks at the words again and stores the abort value back, so the wait is released" {
    const got = try failBehindAtomic(.{ .nap = false, .dead_ms = 100_000, .abort_settle_us = 5_000 });
    for (got.words) |v| try std.testing.expectEqual(roci.abort_value, v);
    try std.testing.expect(got.restored >= 1);
}

test "control: without the failure's second look the same atomic leaves a count of one over the abort value" {
    const got = try failBehindAtomic(.{ .nap = false, .dead_ms = 100_000, .abort_rounds = 0 });
    for (got.words) |v| try std.testing.expectEqual(@as(u64, 1), v);
}

test "a failure raised by another thread waits for the proxies to be out of their posts before the NIC takes the queue pairs away" {
    // node 0's rail A proxy is inside a slow post (50 ms in the NIC) when node 0's host fails the transport
    const net = try Net.open(.{ .late_rail = 0, .late_ns = 50 * std.time.ns_per_ms }, .{ .nap = false, .dead_ms = 100_000 }, roci.identity);
    defer net.close();
    try net.start();
    try net.nodes[0].t.send(FakeStream{}, .own1, 0);
    var spins: u32 = 0;
    while (net.in_post[0].load(.acquire) == 0 and spins < 5_000_000) : (spins += 1) std.atomic.spinLoopHint();
    try std.testing.expect(net.in_post[0].load(.acquire) != 0);
    net.nodes[0].t.fail(.refused, .own1, 0);
    try std.testing.expectEqual(@as(u32, 0), net.post_at_fail[0].load(.acquire)); // no post was in the NIC when the queue pairs went
    try std.testing.expect(net.failed_nic[0].load(.acquire) != 0);
}

test "a failure the proxy raises itself does not wait for itself" {
    const net = try Net.open(.{ .broken = 0 }, .{ .nap = false, .dead_ms = 100_000 }, roci.identity);
    defer net.close();
    try net.start();
    const t0 = std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds;
    try net.nodes[0].t.send(FakeStream{}, .wob, 0);
    var tries: u32 = 0;
    while (net.nodes[0].t.failure() == null and tries < 2000) : (tries += 1) sleep(100_000);
    try std.testing.expectEqual(roci.FailReason.post, net.nodes[0].t.failure().?.reason);
    // the failure's reaction (its look at the words included) is over well inside the 100 ms a wait for a proxy would cost
    while (net.nodes[0].t.failing.load(.acquire) != 0) sleep(100_000);
    const took: u64 = @intCast(std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds - t0);
    try std.testing.expect(took < 90 * std.time.ns_per_ms);
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
    for ([_][]const u8{ "events", "doorbells", "stream_writes", "wait_groups", "data_writes", "words", "posts", "bytes_out", "backstop_wakes", "wake_bell", "wake_deadline", "wake_early", "parks", "park_timeouts", "post_ns_max", "watchdog_max_age_ms", "abort" }) |k|
        try std.testing.expect(std.mem.indexOf(u8, text, k) != null);
}

// -- the proxies' idle park, their wake causes and their loop's clock reads ---------------------------------------------

/// Every node sends and waits `rounds` times with `gap_ns` of nothing between rounds (the proxies' idle).
const GapRun = struct {
    net: *Net,
    rounds: u32,
    gap_ns: u64,
    err: [roci.world]?anyerror = @splat(null),

    fn main(r: *GapRun, g: usize) void {
        r.body(g) catch |e| {
            r.err[g] = e;
        };
    }
    fn body(r: *GapRun, g: usize) !void {
        const nd = &r.net.nodes[g];
        const s: FakeStream = .{};
        for (0..r.rounds) |_| {
            try nd.t.send(s, .own1, 0);
            try nd.t.wait(s, .own1);
            sleep(r.gap_ns);
        }
    }
};

fn runGaps(net: *Net, rounds: u32, gap_ns: u64) !void {
    var run: GapRun = .{ .net = net, .rounds = rounds, .gap_ns = gap_ns };
    var th: [roci.world]std.Thread = undefined;
    for (&th, 0..) |*t, g| t.* = try std.Thread.spawn(.{}, GapRun.main, .{ &run, g });
    for (th) |t| t.join();
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
}

fn allParked(net: *Net) bool {
    for (&net.nodes) |*nd| for (0..roci.rails) |r| if (nd.t.asleep[r].load(.acquire) == 0) return false;
    return true;
}

fn waitParked(net: *Net) !void {
    var tries: u32 = 0;
    while (!allParked(net) and tries < 2000) : (tries += 1) sleep(1_000_000);
    try std.testing.expect(allParked(net));
}

fn totalCounters(net: *Net) roci.Counters {
    var c: roci.Counters = .{};
    for (&net.nodes) |*nd| c.add(nd.t.counters());
    return c;
}

test "the idle park: proxies sleep between sends, every send after a sleep is posted at once (no wake waits for the safety timeout), the words land" {
    const net = try Net.open(.{}, .{ .nap = true, .idle_park_us = 200, .park_safety_ms = 3000 }, roci.identity);
    defer net.close();
    try net.start();
    const t0 = std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds;
    try runGaps(net, 40, 3 * std.time.ns_per_ms);
    const took_ms = @divTrunc(std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds - t0, std.time.ns_per_ms);
    const c = totalCounters(net);
    try std.testing.expect(c.parks > 0);
    try std.testing.expectEqual(@as(u64, 0), c.park_timeouts); // no sleep ended by the net: every wake came from the host
    try std.testing.expect(took_ms < 2500); // (a lost wake costs 3 s)
}

test "control: a host that never wakes the parked proxies leaves every send to the safety timeout" {
    roci.Transport.drop_wakes = true;
    defer roci.Transport.drop_wakes = false;
    const net = try Net.open(.{}, .{ .nap = true, .idle_park_us = 200, .park_safety_ms = 100 }, roci.identity);
    defer net.close();
    try net.start();
    try runGaps(net, 6, 3 * std.time.ns_per_ms);
    const c = totalCounters(net);
    try std.testing.expect(c.park_timeouts > 0);
}

test "a parked proxy ends on a failure: the abort wakes it before the safety timeout" {
    const net = try Net.open(.{}, .{ .nap = true, .idle_park_us = 200, .park_safety_ms = 5000 }, roci.identity);
    defer net.close();
    try net.start();
    try waitParked(net);
    net.nodes[0].t.fail(.peer, .own1, 1);
    var tries: u32 = 0;
    while (net.nodes[0].t.asleep[0].load(.acquire) + net.nodes[0].t.asleep[1].load(.acquire) != 0 and tries < 400) : (tries += 1) sleep(1_000_000);
    try std.testing.expectEqual(@as(u32, 0), net.nodes[0].t.asleep[0].load(.acquire) + net.nodes[0].t.asleep[1].load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), net.nodes[0].t.counters().park_timeouts);
    try std.testing.expectError(error.RociFailed, net.nodes[0].t.check());
}

test "a parked proxy ends on a stop: closing the transport does not wait for the safety timeout" {
    const net = try Net.open(.{}, .{ .nap = true, .idle_park_us = 200, .park_safety_ms = 5000 }, roci.identity);
    try net.start();
    try waitParked(net);
    const t0 = std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds;
    net.close();
    const ms = @divTrunc(std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds - t0, std.time.ns_per_ms);
    try std.testing.expect(ms < 1000);
}

test "a proxy that owes a peer's word does not park (the watchdog still has a thread to run on)" {
    const net = try Net.open(.{ .silent = 1 }, .{ .nap = true, .idle_park_us = 200, .dead_ms = 150 }, roci.identity);
    defer net.close();
    try net.start();
    const nd = &net.nodes[0];
    const s: FakeStream = .{};
    try nd.t.send(s, .own1, 0);
    try nd.t.wait(s, .own1); // released by the watchdog's abort
    try std.testing.expectEqual(roci.FailReason.watchdog, nd.t.failure().?.reason);
}

test "the wake causes: an idle nap ends at its deadline or early, never by a doorbell, and backstop_wakes is their sum" {
    const net = try Net.open(.{}, .{ .nap = true }, roci.identity);
    defer net.close();
    try net.start();
    sleep(30 * std.time.ns_per_ms);
    // The proxies are still running: a nap's cause is counted in steps (backstop_wakes, then deadline or early), so a reading
    // taken between the steps shows one more backstop wake than causes. Read once they have stopped (the counters are plain).
    joinProxies(net);
    const c = totalCounters(net);
    try std.testing.expect(c.wake_deadline + c.wake_early > 0);
    try std.testing.expectEqual(@as(u64, 0), c.wake_bell);
    try std.testing.expectEqual(c.backstop_wakes, c.wake_deadline + c.wake_early);
}

test "the wake causes: a nap that a doorbell ends counts as a bell wake" {
    const net = try Net.open(.{}, .{ .nap = true, .backstop_us = 100_000 }, roci.identity);
    defer net.close();
    try net.start();
    sleep(5 * std.time.ns_per_ms); // (every proxy is in its 100 ms nap)
    const s: FakeStream = .{};
    for (&net.nodes) |*nd| try nd.t.send(s, .own1, 0);
    for (&net.nodes) |*nd| try nd.t.wait(s, .own1);
    try std.testing.expect(totalCounters(net).wake_bell > 0);
}

/// Every node's proxies ended (their timers filed, an instrument build's), their transports kept.
fn joinProxies(net: *Net) void {
    for (&net.nodes) |*nd| nd.t.stop.store(true, .release);
    for (&net.nodes) |*nd| for (&nd.t.threads) |*th| if (th.*) |x| {
        x.join();
        th.* = null;
    };
}

test "the nap timer: a default build carries none; an instrument build's proxies time every nap they take" {
    if (!ti.enabled) return error.SkipZigTest;
    const net = try Net.open(.{}, .{ .nap = true }, roci.identity);
    defer net.close();
    try net.start();
    sleep(30 * std.time.ns_per_ms);
    joinProxies(net);
    for (&net.nodes) |*nd| {
        for (0..roci.rails) |r| {
            const t = ti.filedFor(nd.t, @intCast(r)).?;
            const c = nd.t.rail_counters[r];
            // the idle loop naps once a turn, and the timer saw every nap: its count is the loop's wakes by cause
            try std.testing.expect(t.naps > 0);
            try std.testing.expectEqual(c.wake_bell + c.backstop_wakes, t.naps);
            var in_hist: u64 = 0;
            for (t.hist) |h| in_hist += h;
            try std.testing.expectEqual(t.naps, in_hist);
            var early: u64 = 0;
            for (t.early_hist) |h| early += h;
            try std.testing.expectEqual(c.wake_early, early);
            // nothing rang the doorbell: every nap ran its 100 us deadline or an early wake ended it
            try std.testing.expectEqual(@as(u64, 0), c.wake_bell);
            try std.testing.expect(t.ticks > 0);
        }
    }
}

test "the nap timer: a nap a doorbell ends is timed as a bell wake, and the naps that waited for it lasted 100 us or more" {
    if (!ti.enabled) return error.SkipZigTest;
    const net = try Net.open(.{}, .{ .nap = true, .backstop_us = 100_000 }, roci.identity);
    defer net.close();
    try net.start();
    sleep(5 * std.time.ns_per_ms);
    const s: FakeStream = .{};
    for (&net.nodes) |*nd| try nd.t.send(s, .own1, 0);
    for (&net.nodes) |*nd| try nd.t.wait(s, .own1);
    joinProxies(net);
    var bells: u64 = 0;
    var long: u64 = 0;
    for (&net.nodes) |*nd| {
        for (0..roci.rails) |r| {
            bells += nd.t.rail_counters[r].wake_bell;
            long += ti.filedFor(nd.t, @intCast(r)).?.hist[ti.buckets - 1];
        }
    }
    try std.testing.expect(bells > 0);
    try std.testing.expect(long > 0);
}

test "the idle loop reads the clock once every housekeep_every turns, not every turn (control: every turn reads it every turn)" {
    var reads: [2]u64 = undefined;
    var wakes: [2]u64 = undefined;
    for ([_]u32{ 16, 1 }, 0..) |every, i| {
        const net = try Net.open(.{}, .{ .nap = true, .housekeep_every = every }, roci.identity);
        try net.start();
        const r0 = roci.clock_reads.load(.monotonic);
        sleep(60 * std.time.ns_per_ms);
        reads[i] = roci.clock_reads.load(.monotonic) - r0;
        wakes[i] = totalCounters(net).wakes;
        net.close();
    }
    try std.testing.expect(wakes[0] > 100 and wakes[1] > 100);
    try std.testing.expect(reads[0] * 4 < wakes[0]); // housekeeping only
    try std.testing.expect(reads[1] * 2 > wakes[1]); // the control reads at every wake
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
    /// The proxy's posts wait while this is set (a peer that is late: the other lane must not wait on it).
    hold: std.atomic.Value(bool) = .init(false),
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
    while (me.hold.load(.acquire)) sleep(100_000);
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
    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try roci.startLine(&w, 2, "mlx5_0,mlx5_2", 3, .{}, .write);
    const text = w.buffered();
    for ([_][]const u8{ "Rocinante transport", "639893504", "mlx5_0,mlx5_2", "GID 3", "layout hash", "idle park 0 us", "housekeeping every 16 turns" }) |k| try std.testing.expect(std.mem.indexOf(u8, text, k) != null);
}

test "the start line says what the transport carries: the owner sums, and the codes and the gathers where the arena holds their regions; the event counts follow" {
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("the owner sums of full passes (attention, MoE, Engram)", roci.carriesText(&buf, false, false));
    const g = roci.carriesText(&buf, false, true);
    try std.testing.expectEqualStrings("the owner sums of full passes (attention, MoE, Engram), the gathers (qr and y, x, the routing's codes and picks, the indexer's top-k, the boundary's cache rows)", g);
    const c = roci.carriesText(&buf, true, false);
    try std.testing.expectEqualStrings("the owner sums of full passes (attention, MoE, Engram), the codes families (wo_b's and the expert pack's)", c);
    const both = roci.carriesText(&buf, true, true);
    try std.testing.expect(std.mem.indexOf(u8, both, "NCCL") == null and std.mem.indexOf(u8, both, "the codes families") != null and std.mem.endsWith(u8, both, "the boundary's cache rows)"));
    try std.testing.expectEqual(@as(u64, 84), roci.eventsPerPass(false, false));
    try std.testing.expectEqual(@as(u64, 124), roci.eventsPerPass(true, false));
    try std.testing.expectEqual(@as(u64, 128), roci.eventsPerPass(false, true));
    try std.testing.expectEqual(@as(u64, 168), roci.eventsPerPass(true, true));
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

// -- two lanes: the second pass of a prompt in flight (TF_DS_ROCI_STAGGER) -------------------------------------------

/// Two nets of four nodes, node g of each the two lanes of one rank: each fails the other.
fn openLanes(m0: Mode, m1: Mode, set: roci.Settings, order: roci.Order) ![2]*Net {
    const a = try Net.open(m0, set, order);
    errdefer a.close();
    const b = try Net.open(m1, set, order);
    for (&a.nodes, &b.nodes) |*x, *y| roci.Transport.pair(x.t, y.t);
    return .{ a, b };
}

const LaneJob = struct {
    net: *Net,
    classes: []const C,
    gens: u64,
    order: roci.Order,
    gen_base: u64,
    timeout_ns: u64 = 10 * std.time.ns_per_s,
    run: ?Run = null,
    err: ?anyerror = null,
};

fn laneMain(j: *LaneJob) void {
    j.run = runProgramOn(j.net, j.classes, j.gens, j.order, false, j.gen_base, j.timeout_ns) catch |e| {
        j.err = e;
        return;
    };
}

test "two lanes in flight at once: each lane's bytes land in its own arena and its own words, every generation, both orders" {
    for ([_]roci.Order{ roci.identity, perm }) |order| {
        const nets = try openLanes(.{}, .{}, .{ .nap = false, .gathers = true, .codes = true }, order);
        defer nets[0].close();
        defer nets[1].close();
        try nets[0].start();
        try nets[1].start();
        const classes = [_]C{ .own1, .own2, .wob, .pack, .gm, .gf, .gb };
        var jobs = [2]LaneJob{
            .{ .net = nets[0], .classes = &classes, .gens = 4, .order = order, .gen_base = 0 },
            .{ .net = nets[1], .classes = &classes, .gens = 4, .order = order, .gen_base = 500 },
        };
        var th: [2]std.Thread = undefined;
        for (&th, &jobs) |*t, *j| t.* = try std.Thread.spawn(.{}, laneMain, .{j});
        for (th) |t| t.join();
        for (jobs) |j| {
            try std.testing.expectEqual(@as(?anyerror, null), j.err);
            for (j.run.?.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
            for (j.run.?.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
        }
        // each lane counted its own events: the lanes share no counter
        for (nets) |n| for (&n.nodes) |*nd| {
            try std.testing.expectEqual(@as(u64, classes.len * 4), nd.t.counters().events);
            try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
        };
    }
}

test "control: lane 1's chains and words wired into lane 0's nodes never reach lane 1's waiters (the isolation check can fail)" {
    const a = try Net.open(.{}, .{ .nap = false }, roci.identity);
    defer a.close();
    const b = try Net.open(.{ .cross_to = a }, .{ .nap = false }, roci.identity);
    defer b.close();
    try b.start();
    const classes = [_]C{ .own1, .wob };
    const run = try runProgramOn(b, &classes, 2, roci.identity, false, 500, 300 * std.time.ns_per_ms);
    // every rank's wait for lane 1's words timed out: they went to the other lane
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, error.WaitTimeout), e);
    // and they landed in lane 0's words: the wiring was live
    var seen: u64 = 0;
    for (&a.nodes) |*nd| {
        for (&nd.ctl.recv) |*c| for (c) |*sw| for (sw) |*w| {
            seen += w.v;
        };
    }
    try std.testing.expect(seen != 0);
}

test "lane 1 runs past the 8,192-entry ring while lane 0 idles, then lane 0 runs: neither lane's ticket count, words or bytes depend on the other" {
    const nets = try openLanes(.{}, .{}, .{ .nap = false }, roci.identity);
    defer nets[0].close();
    defer nets[1].close();
    try nets[0].start();
    try nets[1].start();
    const ns = try gpa.alloc(usize, 9000);
    defer gpa.free(ns);
    @memset(ns, 1);
    const classes = [_]C{.own2};
    const run1 = try runAny(nets[1], &classes, ns, roci.identity);
    for (run1.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run1.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    for (&nets[0].nodes) |*nd| try std.testing.expectEqual(@as(u64, 0), nd.t.counters().events);
    const run0 = try runProgramOn(nets[0], &.{ .own1, .gm }, 2, roci.identity, false, 0, 10 * std.time.ns_per_s);
    for (run0.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run0.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    for (&nets[1].nodes) |*nd| try std.testing.expectEqual(@as(u64, 9000), nd.t.counters().events);
}

test "a failure on lane 0 stops lane 1 of the same rank: abort values in its words, its next send and wait refuse, other ranks' lanes are untouched" {
    const nets = try openLanes(.{ .broken = 0 }, .{}, .{ .nap = false }, roci.identity);
    defer nets[0].close();
    defer nets[1].close();
    try nets[0].start();
    try nets[1].start();
    const s: FakeStream = .{};
    try nets[0].nodes[0].t.send(s, .wob, 0);
    var tries: u32 = 0;
    while (nets[1].nodes[0].t.failure() == null and tries < 2000) : (tries += 1) sleep(1_000_000);
    const f0 = nets[0].nodes[0].t.failure().?;
    try std.testing.expectEqual(roci.FailReason.post, f0.reason);
    const f1 = nets[1].nodes[0].t.failure().?;
    try std.testing.expectEqual(roci.FailReason.peer, f1.reason);
    // (the failure's flag is set first; the abort values are in the words when its reaction has finished)
    while (nets[1].nodes[0].t.failing.load(.acquire) != 0) sleep(100_000);
    try std.testing.expectError(error.RociFailed, nets[1].nodes[0].t.check());
    try std.testing.expectError(error.RociFailed, nets[1].nodes[0].t.send(s, .own1, 0));
    try std.testing.expectError(error.RociFailed, nets[1].nodes[0].t.wait(s, .own1));
    for (&nets[1].nodes[0].ctl.recv) |*c| for (c) |*sw| for (sw) |*w| try std.testing.expectEqual(roci.abort_value, w.v);
    for (nets) |n| for (n.nodes[1..]) |*nd| try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
}

test "control: without the pairing a failure on lane 0 leaves lane 1 serving" {
    const nets = [2]*Net{ try Net.open(.{ .broken = 0 }, .{ .nap = false }, roci.identity), try Net.open(.{}, .{ .nap = false }, roci.identity) };
    defer nets[0].close();
    defer nets[1].close();
    try nets[0].start();
    try nets[1].start();
    try nets[0].nodes[0].t.send(FakeStream{}, .wob, 0);
    var tries: u32 = 0;
    while (nets[0].nodes[0].t.failure() == null and tries < 2000) : (tries += 1) sleep(1_000_000);
    try std.testing.expect(nets[0].nodes[0].t.failure() != null);
    sleep(20_000_000);
    try std.testing.expectEqual(@as(?@TypeOf(nets[1].nodes[0].t.failure().?), null), nets[1].nodes[0].t.failure());
    try nets[1].nodes[0].t.check();
}

// -- three lanes: TF_DS_ROCI_STAGGER_LANES=3 -----------------------------------------------------------------------------

/// Three nets of four nodes, node g of each the three lanes of one rank: a ring, each fails the next, and so all.
fn openLanes3(m: [3]Mode, set: roci.Settings, order: roci.Order) ![3]*Net {
    const a = try Net.open(m[0], set, order);
    errdefer a.close();
    const b = try Net.open(m[1], set, order);
    errdefer b.close();
    const c = try Net.open(m[2], set, order);
    for (&a.nodes, &b.nodes, &c.nodes) |*x, *y, *z| roci.Transport.ring3(x.t, y.t, z.t);
    return .{ a, b, c };
}

test "three lanes in flight at once: each lane's bytes land in its own arena and its own words, every generation, both orders" {
    for ([_]roci.Order{ roci.identity, perm }) |order| {
        const nets = try openLanes3(.{ .{}, .{}, .{} }, .{ .nap = false, .gathers = true, .codes = true }, order);
        defer nets[0].close();
        defer nets[1].close();
        defer nets[2].close();
        for (nets) |n| try n.start();
        const classes = [_]C{ .own1, .own2, .wob, .pack, .gm, .gf, .gb };
        var jobs = [3]LaneJob{
            .{ .net = nets[0], .classes = &classes, .gens = 4, .order = order, .gen_base = 0 },
            .{ .net = nets[1], .classes = &classes, .gens = 4, .order = order, .gen_base = 500 },
            .{ .net = nets[2], .classes = &classes, .gens = 4, .order = order, .gen_base = 900 },
        };
        var th: [3]std.Thread = undefined;
        for (&th, &jobs) |*t, *j| t.* = try std.Thread.spawn(.{}, laneMain, .{j});
        for (th) |t| t.join();
        for (jobs) |j| {
            try std.testing.expectEqual(@as(?anyerror, null), j.err);
            for (j.run.?.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
            for (j.run.?.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
        }
        for (nets) |n| for (&n.nodes) |*nd| {
            try std.testing.expectEqual(@as(u64, classes.len * 4), nd.t.counters().events);
            try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
        };
    }
}

test "a failure on any one of three lanes stops the other two of the same rank, the ring in both directions; other ranks' lanes are untouched" {
    for ([_]usize{ 0, 1, 2 }) |broken_lane| {
        var modes = [3]Mode{ .{}, .{}, .{} };
        modes[broken_lane] = .{ .broken = 0 };
        const nets = try openLanes3(modes, .{ .nap = false }, roci.identity);
        defer nets[0].close();
        defer nets[1].close();
        defer nets[2].close();
        for (nets) |n| try n.start();
        const s: FakeStream = .{};
        try nets[broken_lane].nodes[0].t.send(s, .wob, 0);
        for (nets, 0..) |n, i| {
            var tries: u32 = 0;
            while (n.nodes[0].t.failure() == null and tries < 3000) : (tries += 1) sleep(1_000_000);
            const f = n.nodes[0].t.failure() orelse return error.LaneNotStopped;
            try std.testing.expectEqual(if (i == broken_lane) roci.FailReason.post else roci.FailReason.peer, f.reason);
            while (n.nodes[0].t.failing.load(.acquire) != 0) sleep(100_000);
            try std.testing.expectError(error.RociFailed, n.nodes[0].t.check());
        }
        for (nets) |n| for (n.nodes[1..]) |*nd| try std.testing.expectEqual(@as(?@TypeOf(nd.t.failure().?), null), nd.t.failure());
    }
}

test "control: a ring of two links (pair) leaves a third lane serving after a failure on lane 0" {
    const a = try Net.open(.{ .broken = 0 }, .{ .nap = false }, roci.identity);
    defer a.close();
    const b = try Net.open(.{}, .{ .nap = false }, roci.identity);
    defer b.close();
    const c = try Net.open(.{}, .{ .nap = false }, roci.identity);
    defer c.close();
    for (&a.nodes, &b.nodes) |*x, *y| roci.Transport.pair(x.t, y.t);
    for ([_]*Net{ a, b, c }) |n| try n.start();
    try a.nodes[0].t.send(FakeStream{}, .wob, 0);
    var tries: u32 = 0;
    while (b.nodes[0].t.failure() == null and tries < 3000) : (tries += 1) sleep(1_000_000);
    try std.testing.expect(b.nodes[0].t.failure() != null);
    sleep(20_000_000);
    try std.testing.expectEqual(@as(?@TypeOf(c.nodes[0].t.failure().?), null), c.nodes[0].t.failure());
}

const RaceJob = struct { t: *roci.Transport, closes: bool, go: *std.atomic.Value(bool) };

fn raceMain(j: *RaceJob) void {
    while (!j.go.load(.acquire)) std.atomic.spinLoopHint();
    if (j.closes) j.t.deinit() else j.t.fail(.peer, .wob, 0);
}

test "a failure on one lane while the other two close: the ring's links are read under the lock and a lane being failed is held by its count, so deinit never frees it under fail() and no closed lane is touched (many rounds)" {
    for (0..40) |round| {
        const nets = try openLanes3(.{ .{}, .{}, .{} }, .{ .nap = false }, roci.identity);
        defer nets[0].close();
        defer nets[1].close();
        defer nets[2].close();
        for (nets) |n| try n.start();
        var go = std.atomic.Value(bool).init(false);
        // lane 0 fails (reaching its siblings through the ring) while lane 1 and lane 2 close, in both orders
        var jobs = [3]RaceJob{
            .{ .t = nets[0].nodes[0].t, .closes = false, .go = &go },
            .{ .t = nets[1].nodes[0].t, .closes = true, .go = &go },
            .{ .t = nets[2].nodes[0].t, .closes = true, .go = &go },
        };
        if (round % 2 == 1) std.mem.swap(RaceJob, &jobs[1], &jobs[2]);
        nets[1].nodes[0].gone = true;
        nets[2].nodes[0].gone = true;
        var th: [3]std.Thread = undefined;
        for (&th, &jobs) |*t, *j| t.* = try std.Thread.spawn(.{}, raceMain, .{j});
        go.store(true, .release);
        for (th) |t| t.join();
        // lane 0 failed and nothing else was touched after its close: the survivors of the ring are lane 0 alone
        try std.testing.expect(nets[0].nodes[0].t.failure() != null);
        try std.testing.expectEqual(@as(?*roci.Transport, null), nets[0].nodes[0].t.sibling);
        // (check() reads the ring under the same lock)
        try std.testing.expectError(error.RociFailed, nets[0].nodes[0].t.check());
    }
}

test "a lane leaves the ring before it closes: the lane that pointed at it points past it, nothing keeps a pointer to a closed lane" {
    var a: roci.Transport = undefined;
    var b: roci.Transport = undefined;
    var c: roci.Transport = undefined;
    roci.Transport.ring3(&a, &b, &c);
    c.unlink(); // (lanes close last to first)
    try std.testing.expectEqual(@as(?*roci.Transport, &b), a.sibling);
    try std.testing.expectEqual(@as(?*roci.Transport, &a), b.sibling);
    try std.testing.expectEqual(@as(?*roci.Transport, null), c.sibling);
    b.unlink();
    try std.testing.expectEqual(@as(?*roci.Transport, null), a.sibling);
    try std.testing.expectEqual(@as(?*roci.Transport, null), b.sibling);
    a.unlink(); // (a lane with no sibling goes quietly)
    // any order of closing leaves no dangling pointer
    for ([_][3]usize{ .{ 0, 1, 2 }, .{ 0, 2, 1 }, .{ 1, 0, 2 }, .{ 1, 2, 0 }, .{ 2, 0, 1 }, .{ 2, 1, 0 } }) |order| {
        var ts: [3]roci.Transport = undefined;
        roci.Transport.ring3(&ts[0], &ts[1], &ts[2]);
        var gone = [_]bool{ false, false, false };
        for (order) |i| {
            ts[i].unlink();
            gone[i] = true;
            for (&ts, 0..) |*t, j| if (!gone[j]) if (t.sibling) |sb| {
                for (&ts, 0..) |*q, k| if (sb == q) try std.testing.expect(!gone[k]);
            };
        }
    }
}

// -- two lanes on a real driver: the streams of a pair (skipped without one) ----------------------------------------------

/// Two nodes of one lane: their arenas, control blocks and transports (the lane's own).
const GpuLane = struct {
    nodes: [2]GpuNode = undefined,

    fn open(d: *const cuda.Driver, arena_len: usize) !GpuLane {
        var l: GpuLane = .{};
        for (&l.nodes, 0..) |*n, i| {
            const a = try hostAlloc(d, arena_len, 0x01 | 0x02 | 0x04);
            const c = try hostAlloc(d, @sizeOf(roci.Control), 0x01 | 0x02);
            n.* = .{ .arena = a.p[0..arena_len], .arena_dev = a.dev, .ctl = @ptrCast(c.p), .ctl_dev = c.dev, .me = @intCast(i) };
            n.ctl.* = .{};
        }
        return l;
    }

    fn start(l: *GpuLane) !void {
        for (&l.nodes, 0..) |*n, i| {
            n.peer = &l.nodes[1 - i];
            n.t = try roci.Transport.init(gpa, @intCast(i), roci.identity, n.ctl, n.ctl_dev, .{ .ctx = n, .post_fn = gpuPost, .fail_fn = gpuFail, .reap_fn = gpuReap }, .{ .nap = false });
            try n.t.start(null);
        }
    }

    fn close(l: *GpuLane, d: *const cuda.Driver) void {
        for (&l.nodes) |*n| n.t.deinit();
        for (&l.nodes) |*n| {
            _ = d.api.cuMemFreeHost(n.arena.ptr);
            _ = d.api.cuMemFreeHost(@ptrCast(n.ctl));
        }
    }
};

/// The word each psum block holds: lane, node, block and generation.
fn laneWord(lane: u32, node: u32, block: u32, gen: u32) u32 {
    return (lane << 30) | (node << 28) | (block << 20) | gen;
}

fn laneWrite(d: *const cuda.Driver, nd: *GpuNode, s: cuda.Stream, lane: u32, gen: u32) !void {
    const l = roci.layout();
    const blk: usize = 1024 * 2560 * 4;
    for (0..4) |b| try d.check(d.api.cuMemsetD32Async(nd.arena_dev + l.at(.psum) + b * blk, laneWord(lane, nd.me, @intCast(b), gen), blk / 4, s.handle), "memset");
}

/// Differing words of node `nd`'s stage regions against what its peer wrote: lane `from`'s words.
fn laneStageBad(nd: *GpuNode, from: u32, gen: u32) u64 {
    const l = roci.layout();
    const blk: usize = 1024 * 2560 * 4;
    const peer: u32 = 1 - nd.me;
    const w = [2]u32{ laneWord(from, peer, nd.me, gen), laneWord(from, peer, nd.me ^ 2, gen) };
    var bad: u64 = 0;
    for (0..2) |k| {
        const got = std.mem.bytesAsSlice(u32, nd.arena[l.at(.stage) + k * blk ..][0..blk]);
        for (got) |x| bad += @intFromBool(x != w[k]);
    }
    return bad;
}

fn streamDone(s: cuda.Stream, ms: u64) bool {
    var waited: u64 = 0;
    while (waited < ms) : (waited += 1) {
        if (s.d.api.cuStreamQuery(s.handle) == 0) return true;
        sleep(1_000_000);
    }
    return false;
}

test "two lanes on one GPU: each lane's stream writes its arena, rings its own doorbell and waits its own words; every byte lands in its lane; the lanes' streams do not wait on each other" {
    var driver = cuda.Driver.open() catch return error.SkipZigTest;
    defer driver.close();
    var cx = cuda.Context.init(&driver, 0) catch return error.SkipZigTest;
    defer cx.deinit();
    var lanes: [2]GpuLane = undefined;
    lanes[0] = try GpuLane.open(&driver, 64 << 20);
    lanes[1] = try GpuLane.open(&driver, 64 << 20);
    try lanes[0].start();
    try lanes[1].start();
    defer for (&lanes) |*l| l.close(&driver);
    var st: [2][2]cuda.Stream = undefined; // [lane][node]
    for (&st) |*row| {
        for (row) |*x| x.* = try cuda.Stream.init(&driver, true);
    }
    defer for (&st) |*row| {
        for (row) |*x| x.deinit();
    };
    // generations in flight on both lanes at once: lane 1's sends are queued before lane 0's waits
    for (1..5) |gen| {
        for (0..2) |lane| for (0..2) |n| {
            try laneWrite(&driver, &lanes[lane].nodes[n], st[lane][n], @intCast(lane), @intCast(gen));
            try lanes[lane].nodes[n].t.send(st[lane][n], .own1, 0);
        };
        for (0..2) |lane| for (0..2) |n| try lanes[lane].nodes[n].t.wait(st[lane][n], .own1);
        for (0..2) |lane| for (0..2) |n| try st[lane][n].synchronize();
        for (0..2) |lane| for (0..2) |n| try std.testing.expectEqual(@as(u64, 0), laneStageBad(&lanes[lane].nodes[n], @intCast(lane), @intCast(gen)));
    }
    // each lane counted its own sends and waits
    for (&lanes) |*ln| for (&ln.nodes) |*nd| {
        try std.testing.expectEqual(@as(u64, 4), nd.t.counters().events);
        try std.testing.expectEqual(@as(u64, 4), nd.t.counters().wait_groups);
    };
    // control: a consumer that waits lane 0's words reads lane 1's stage before lane 1's bytes are there (lane 1's peer is late)
    lanes[1].nodes[1].hold.store(true, .release);
    for (0..2) |n| {
        try laneWrite(&driver, &lanes[1].nodes[n], st[1][n], 1, 9);
        try lanes[1].nodes[n].t.send(st[1][n], .own1, 0);
    }
    for (0..2) |n| {
        try laneWrite(&driver, &lanes[0].nodes[n], st[0][n], 0, 9);
        try lanes[0].nodes[n].t.send(st[0][n], .own1, 0);
    }
    for (0..2) |n| try lanes[0].nodes[n].t.wait(st[0][n], .own1);
    // liveness: lane 0's stream finishes while lane 1's node 0 waits on its late peer
    try lanes[1].nodes[0].t.wait(st[1][0], .own1);
    try std.testing.expect(streamDone(st[0][0], 3000));
    try std.testing.expect(streamDone(st[0][1], 3000));
    try std.testing.expect(!streamDone(st[1][0], 150)); // lane 1's wait is pending: the late peer has not posted
    try std.testing.expect(laneStageBad(&lanes[1].nodes[0], 1, 9) != 0); // the wrong-lane control: lane 1's bytes are not there yet
    lanes[1].nodes[1].hold.store(false, .release);
    try lanes[1].nodes[1].t.wait(st[1][1], .own1);
    try st[1][0].synchronize();
    try st[1][1].synchronize();
    try std.testing.expectEqual(@as(u64, 0), laneStageBad(&lanes[1].nodes[0], 1, 9));
    try std.testing.expectEqual(@as(u64, 0), laneStageBad(&lanes[0].nodes[0], 0, 9));
}

test "control: lane 1's posts aimed at lane 0's arena leave lane 1's stage empty and overwrite lane 0's" {
    var driver = cuda.Driver.open() catch return error.SkipZigTest;
    defer driver.close();
    var cx = cuda.Context.init(&driver, 0) catch return error.SkipZigTest;
    defer cx.deinit();
    var lanes: [2]GpuLane = undefined;
    lanes[0] = try GpuLane.open(&driver, 64 << 20);
    lanes[1] = try GpuLane.open(&driver, 64 << 20);
    try lanes[0].start();
    try lanes[1].start();
    defer for (&lanes) |*l| l.close(&driver);
    // the miswiring: lane 1's node 1 posts into lane 0's node 0 (arena and words)
    lanes[1].nodes[1].peer = &lanes[0].nodes[0];
    var st: [2]cuda.Stream = undefined;
    for (&st) |*x| x.* = try cuda.Stream.init(&driver, true);
    defer for (&st) |*x| x.deinit();
    try laneWrite(&driver, &lanes[1].nodes[1], st[1], 1, 3);
    try lanes[1].nodes[1].t.send(st[1], .own1, 0);
    try st[1].synchronize();
    var tries: u32 = 0;
    while (lanes[0].nodes[0].ctl.recv[@backingInt(C.own1)][1][0].v == 0 and tries < 2000) : (tries += 1) sleep(1_000_000);
    // lane 1's node 0 got nothing (its stage is zero: not the expected bytes) and lane 0's node 0 holds lane 1's bytes
    try std.testing.expect(laneStageBad(&lanes[1].nodes[0], 1, 3) != 0);
    try std.testing.expectEqual(@as(u64, 0), laneStageBad(&lanes[0].nodes[0], 1, 3));
}

test "the pair's cache gate: pass B's read of a layer's cache waits pass A's event; without the event it reads what A has not yet written" {
    var driver = cuda.Driver.open() catch return error.SkipZigTest;
    defer driver.close();
    var cx = cuda.Context.init(&driver, 0) catch return error.SkipZigTest;
    defer cx.deinit();
    const flag = try hostAlloc(&driver, 4096, 0x01 | 0x02);
    const flag_p: *u64 = @ptrCast(@alignCast(flag.p));
    defer _ = driver.api.cuMemFreeHost(flag.p);
    var cache = try cuda.DeviceBuffer.alloc(&driver, 4096);
    defer cache.free();
    var out = try cuda.DeviceBuffer.alloc(&driver, 8192);
    defer out.free();
    var sa = try cuda.Stream.init(&driver, true);
    defer sa.deinit();
    var sb = try cuda.Stream.init(&driver, true);
    defer sb.deinit();
    var ev = try cuda.Event.init(&driver, false);
    defer ev.deinit();
    try driver.check(driver.api.cuMemsetD32Async(cache.ptr, 0x11111111, 1024, sa.handle), "memset");
    try sa.synchronize();
    for ([_]bool{ true, false }) |gated| {
        flag_p.* = 0;
        const base: u64 = if (gated) 0 else 4096;
        // A's attention writes the cache only when the host releases it (the layer is still running)
        try sa.waitValues64(&.{flag.dev}, &.{1});
        try driver.check(driver.api.cuMemsetD32Async(cache.ptr, if (gated) 0x22222222 else 0x33333333, 1024, sa.handle), "memset");
        try ev.record(sa);
        if (gated) try sb.wait(ev);
        // B's attention reads the layer's cache
        try driver.check(driver.api.cuMemcpyDtoDAsync_v2(out.ptr + base, cache.ptr, 4096, sb.handle), "copy");
        sleep(100_000_000); // B has run as far as it may before A writes
        @atomicStore(u64, flag_p, 1, .release);
        try sa.synchronize();
        try sb.synchronize();
        var got: [1024]u32 = undefined;
        try driver.check(driver.api.cuMemcpyDtoH_v2(&got, out.ptr + base, 4096), "dtoh");
        const want: u32 = if (gated) 0x22222222 else 0x33333333;
        var bad: u32 = 0;
        for (got) |x| bad += @intFromBool(x != want);
        // gated: B read A's write; ungated: B read the old bytes (the control is live)
        if (gated) try std.testing.expectEqual(@as(u32, 0), bad) else try std.testing.expect(bad != 0);
    }
}
// -- the top-k at fewer keys, and the decoder rows' handoff ------------------------------------------------------------------------

test "the top-k gather at kk keys a row: every other node's rows once, contiguous at k q kk 8 bytes, kk = 512 the region's own plan" {
    const l = roci.layout();
    for ([_]usize{ 1, 7, 64, 255, 511, 512 }) |kk| {
        var n: usize = 1;
        while (n <= 4096) : (n += if (n < 70 or n > 4000) 1 else 61) {
            for ([_]roci.Order{ roci.identity, perm }) |order| for (0..roci.world) |g| {
                const e = roci.buildEventX(.gi, n, kk, g, order);
                try std.testing.expectEqual(@as(u8, 3), e.n_peers);
                const k = order[g];
                const q = slabp.q(n);
                const nk = slabp.slab(n, k).n;
                for (0..3) |j| {
                    var bytes: u64 = 0;
                    for (0..roci.rails) |r| for (e.chains[j][r].slice()) |p| {
                        try std.testing.expect(p.src == p.dst);
                        try std.testing.expect(p.src >= l.at(.g_cidx) + k * q * kk * 8 and p.src + p.len <= l.at(.g_cidx) + (k * q + nk) * kk * 8);
                        try std.testing.expect(inside(l, &.{.g_cidx}, p.src, p.len));
                        bytes += p.len;
                    };
                    try std.testing.expectEqual(nk * kk * 8, bytes);
                }
            };
        }
    }
    for (0..roci.world) |g| try sameEvent(roci.buildEventN(.gi, 1000, g, roci.identity), roci.buildEventX(.gi, 1000, 512, g, roci.identity));
}

test "control: the top-k plan of 100 keys is not the plan of 512" {
    const a = roci.buildEventX(.gi, 1000, 100, 0, roci.identity);
    const b = roci.buildEventX(.gi, 1000, 512, 0, roci.identity);
    try std.testing.expect(a.chains[0][0].bytes() != b.chains[0][0].bytes());
}

/// How many of the senders' windows (or this node's own, node 0 keeping its rows) cover each of the last 128 rows of a chunk of `n`
/// rows, for the handoff event of `d` rows: index i is row n - 128 + i (rows before the chunk's start count nothing).
fn handCover(n: usize, d: usize, order: roci.Order) ![128]u8 {
    var cover: [128]u8 = @splat(0);
    const first = n - d;
    for (0..roci.world) |s| {
        const sl = slabp.slab(n, order[s]);
        const lo = @max(first, sl.start);
        const hi = @min(n, sl.start + sl.n);
        if (s != 0) {
            const e = roci.buildEventX(.gh, n, d, s, order);
            try std.testing.expectEqual(@as(u8, 3), e.n_peers);
            const j = std.mem.indexOfScalar(u8, e.peerSlice(), 0).?;
            var hb: u64 = 0;
            for (0..roci.rails) |r| for (e.chains[j][r].slice()) |p| {
                try std.testing.expect(p.src == p.dst);
                if (inside(roci.layout(), &.{.hand_h}, p.src, p.len)) hb += p.len else try std.testing.expect(inside(roci.layout(), &.{.hand_pre}, p.src, p.len));
            };
            const m: u64 = if (hi > lo) hi - lo else 0;
            try std.testing.expectEqual(m * roci.hand_h_row, hb);
            if (m == 0) for (0..3) |jj| for (0..roci.rails) |r| try std.testing.expectEqual(@as(u8, 0), e.chains[jj][r].n);
        }
        if (hi > lo) for (lo..hi) |row| {
            if (row + 128 >= n) cover[row + 128 - n] += 1;
        };
    }
    return cover;
}

test "the handoff of the last d rows: the four nodes' windows cover [n - d, n) once at row (row - first), empty windows send only their word" {
    for ([_]usize{ 1, 2, 13, 16, 17, 48, 49, 64, 65, 127, 128, 129, 500, 1087, 2048, 3000, 4096 }) |n| {
        for ([_]usize{ 1, 2, 5, 17, 64, 100, 127, 128 }) |d| {
            if (d > n) continue;
            for ([_]roci.Order{ roci.identity, perm }) |order| {
                const cover = try handCover(n, d, order);
                for (0..128) |i| {
                    // index i is row n - 128 + i: a row of the window once, any other row (before it) not at all
                    const in_window = i + d >= 128;
                    try std.testing.expectEqual(@as(u8, if (in_window) 1 else 0), cover[i]);
                }
            }
        }
    }
}

test "control: a handoff event of d - 1 rows leaves the window's first row uncovered" {
    const cover = try handCover(500, 100, roci.identity);
    const short = blk: {
        // the senders' windows of 99 rows against a window of 100
        var c: [128]u8 = @splat(0);
        const first = 500 - 99;
        for (0..roci.world) |s| {
            const sl = slabp.slab(500, s);
            const lo = @max(first, sl.start);
            const hi = @min(500, sl.start + sl.n);
            if (hi > lo) for (lo..hi) |row| {
                c[row + 128 - 500] += 1;
            };
        }
        break :blk c;
    };
    try std.testing.expectEqual(@as(u8, 1), cover[128 - 100]);
    try std.testing.expectEqual(@as(u8, 0), short[128 - 100]);
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

// -- the host lane: the Engram records between column partners ------------------------------------------------------------

test "the host lane's layout: four regions on pages, nothing else, the prompt lanes' layout has none of them" {
    const h = roci.layoutOf(.host);
    const p = roci.layout();
    try std.testing.expectEqual(@as(u64, 0), h.bytes(.psum));
    try std.testing.expectEqual(@as(u64, 0), h.bytes(.g_b));
    for ([_]R{ .rec_send, .rec_land }) |r| try std.testing.expectEqual(@as(u64, roci.host_slots * roci.host_slot_bytes), h.bytes(r));
    for ([_]R{ .rec_hs, .rec_hl }) |r| try std.testing.expectEqual(@as(u64, roci.host_slots * roci.host_hdr_bytes), h.bytes(r));
    for (0..roci.n_regions) |i| try std.testing.expectEqual(@as(u64, 0), h.off[i] % roci.page);
    try std.testing.expectEqual(@as(u64, 0), roci.host_slot_bytes % roci.page);
    try std.testing.expect(h.hash() != p.hash());
    try std.testing.expect(h.total < 120 << 20);
}

test "a record event goes to the column partner from the send region to the same offset of the land region, in the two rails' halves" {
    for ([_]usize{ 0, 1, 2, 3 }) |g| for ([_]usize{ 0, 5, 15 }) |slot| for ([_]usize{ 0, 1, 777, roci.host_max_recs }) |count| {
        const e = roci.buildRecEvent(.rec, g, slot, count);
        try std.testing.expectEqual(@as(u8, 1), e.n_peers);
        try std.testing.expectEqual(@as(u8, @intCast(g ^ 2)), e.peers[0]);
        const L = roci.layoutOf(.host);
        var bytes: u64 = 0;
        for (0..roci.rails) |r| for (e.chains[0][r].slice()) |p| {
            try std.testing.expect(p.src >= L.at(.rec_send) + slot * roci.host_slot_bytes);
            try std.testing.expect(p.src + p.len <= L.at(.rec_send) + (slot + 1) * roci.host_slot_bytes);
            try std.testing.expectEqual(p.src - L.at(.rec_send), p.dst - L.at(.rec_land));
            bytes += p.len;
        };
        try std.testing.expectEqual(@as(u64, count * roci.host_rec), bytes);
        const h = roci.buildRecEvent(.rech, g, slot, 99);
        var hb: u64 = 0;
        for (0..roci.rails) |r| for (h.chains[0][r].slice()) |p| {
            hb += p.len;
        };
        try std.testing.expectEqual(@as(u64, roci.host_hdr_bytes), hb);
    };
}

const HostRun = struct {
    net: *Net,
    steps: usize,
    bad: [2]u64 = .{ 0, 0 },
    err: [2]?anyerror = .{ null, null },
};

fn hostMain(run: *HostRun, who: usize) void {
    hostBody(run, who) catch |e| {
        run.err[who] = e;
    };
}

/// Node 0 (who 0) and node 2 (who 1) send each other, every step, a header and a packed run of records whose bytes name (step, sender, index);
/// each waits for the other's words and checks the landed bytes. The ring wraps (steps > slots).
fn hostBody(run: *HostRun, who: usize) !void {
    const g: usize = who * 2;
    const nd = &run.net.nodes[g];
    const L = roci.layoutOf(.host);
    for (0..run.steps) |step| {
        const slot = step % roci.host_slots;
        const count: usize = 1 + (step * 977) % roci.host_max_recs;
        // the records: 264 bytes each, a value naming (step, sender, index)
        const send = nd.arena[L.at(.rec_send) + slot * roci.host_slot_bytes ..][0 .. count * roci.host_rec];
        for (0..count) |i| @memset(send[i * roci.host_rec ..][0..roci.host_rec], @truncate(step *% 31 +% g *% 7 +% i));
        const hdr = nd.arena[L.at(.rec_hs) + slot * roci.host_hdr_bytes ..][0..roci.host_hdr_bytes];
        std.mem.writeInt(u64, hdr[0..8], step, .little);
        std.mem.writeInt(u64, hdr[8..16], count, .little);
        try nd.t.sendRec(.rech, @intCast(slot), 1);
        try nd.t.sendRec(.rec, @intCast(slot), count);
        const want: u64 = step + 1;
        if (!try nd.t.waitHost(.rech, want, 5 * std.time.ns_per_s)) return error.WaitTimeout;
        const phdr = nd.arena[L.at(.rec_hl) + slot * roci.host_hdr_bytes ..][0..roci.host_hdr_bytes];
        const pstep = std.mem.readInt(u64, phdr[0..8], .little);
        const pcount: usize = @intCast(std.mem.readInt(u64, phdr[8..16], .little));
        if (pstep != step) run.bad[who] += 1;
        if (!try nd.t.waitHost(.rec, want, 5 * std.time.ns_per_s)) return error.WaitTimeout;
        const land = nd.arena[L.at(.rec_land) + slot * roci.host_slot_bytes ..][0 .. pcount * roci.host_rec];
        const peer: usize = g ^ 2;
        for (0..pcount) |i| {
            const want_b: u8 = @truncate(step *% 31 +% peer *% 7 +% i);
            for (land[i * roci.host_rec ..][0..roci.host_rec]) |b| run.bad[who] += @intFromBool(b != want_b);
        }
    }
}

test "the host lane between column partners: headers and packed records land byte for byte, the 16 slots wrap, the words keep their order" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false, .gathers = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: HostRun = .{ .net = net, .steps = 40 };
    var th: [2]std.Thread = undefined;
    for (&th, 0..) |*t, who| t.* = try std.Thread.spawn(.{}, hostMain, .{ &run, who });
    for (th) |t| t.join();
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    for ([_]usize{ 0, 2 }) |g| try std.testing.expectEqual(@as(?@TypeOf(net.nodes[g].t.failure().?), null), net.nodes[g].t.failure());
}

test "control: a NIC that writes the word before the records lets the partner read stale records" {
    const net = try Net.open(.{ .host = true, .word_first = true, .gap_ns = 5 * std.time.ns_per_ms }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: HostRun = .{ .net = net, .steps = 6 };
    var th: [2]std.Thread = undefined;
    for (&th, 0..) |*t, who| t.* = try std.Thread.spawn(.{}, hostMain, .{ &run, who });
    for (th) |t| t.join();
    // the checks read right after the word: records or headers not there yet
    try std.testing.expect(run.bad[0] + run.bad[1] > 0 or run.err[0] != null or run.err[1] != null);
}

test "waitHost: false at its deadline when the partner's word is not there, and a failed transport ends it with an error" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false, .dead_ms = 100_000 }, roci.identity);
    defer net.close();
    try net.start();
    const nd = &net.nodes[0];
    try std.testing.expect(!(try nd.t.waitHost(.rec, 1, 30 * std.time.ns_per_ms)));
    try std.testing.expect(!nd.t.hostWordReached(.rec, 1));
    nd.t.fail(.watchdog, .rec, 2);
    try std.testing.expectError(error.RociFailed, nd.t.waitHost(.rec, 1, 0));
}

test "waitHostStats: a word written while the host waits ends the wait, which counts its looks, naps and length" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    const nd = &net.nodes[0];
    const Poster = struct {
        fn run(t: *roci.Transport) void {
            sleep(8 * std.time.ns_per_ms);
            t.sendRec(.rech, 0, 1) catch {};
        }
    };
    // node 2 (the partner) posts its header to node 0
    const th = try std.Thread.spawn(.{}, Poster.run, .{net.nodes[2].t});
    var st: roci.Transport.HostWait = .{};
    try std.testing.expect(try nd.t.waitHostStats(.rech, 1, 5 * std.time.ns_per_s, &st));
    th.join();
    try std.testing.expect(st.ns >= 8 * std.time.ns_per_ms);
    try std.testing.expect(st.naps >= 1 and st.looks >= st.naps);
    // a wait that is already satisfied looks once and does not nap
    var st2: roci.Transport.HostWait = .{};
    try std.testing.expect(try nd.t.waitHostStats(.rech, 1, 5 * std.time.ns_per_ms, &st2));
    try std.testing.expectEqual(@as(u32, 0), st2.naps);
}

test "the host lane parks its proxies in the kernel: they sleep between looks (counted), and the records and headers still land byte for byte" {
    const net = try Net.open(.{ .host = true }, .{ .nap = true, .park = true, .gathers = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: HostRun = .{ .net = net, .steps = 20 };
    var th: [2]std.Thread = undefined;
    for (&th, 0..) |*t, who| t.* = try std.Thread.spawn(.{}, hostMain, .{ &run, who });
    for (th) |t| t.join();
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    sleep(20 * std.time.ns_per_ms);
    const c = net.nodes[0].t.counters();
    try std.testing.expect(c.sleeps > 0);
    try std.testing.expectEqual(@as(u64, 0), c.backstop_wakes); // (no nap on a doorbell)
}

test "the host lane's cores: unpinned by default, and a pin that is another lane's or a ring's refuses by the core" {
    const roci_engine = @import("roci_engine.zig");
    try std.testing.expectEqual(@as(?u16, null), roci_engine.hostPinClash(null, .{ 3, 4 }, .{ 1, 2 }, &.{ 0, 10, 11 }));
    try std.testing.expectEqual(@as(?u16, null), roci_engine.hostPinClash(.{ 15, 16 }, .{ 3, 4 }, .{ 1, 2 }, &.{ 0, 10, 11 }));
    try std.testing.expectEqual(@as(?u16, 4), roci_engine.hostPinClash(.{ 15, 4 }, .{ 3, 4 }, .{ 1, 2 }, &.{ 0, 10, 11 }));
    try std.testing.expectEqual(@as(?u16, 2), roci_engine.hostPinClash(.{ 2, 16 }, null, .{ 1, 2 }, &.{ 0, 10, 11 }));
    try std.testing.expectEqual(@as(?u16, 10), roci_engine.hostPinClash(.{ 16, 10 }, null, null, &.{ 0, 10, 11 }));
    // the two host proxies on one core also share it
    try std.testing.expectEqual(@as(?u16, 16), roci_engine.hostPinClash(.{ 16, 16 }, null, null, &.{}));
}

test "waitHost naps on the word it waits for (the Arm wait the proxies use), and sleeps a fixed time only where the CPU cannot nap" {
    const src = @embedFile("rocinante.zig");
    const at = std.mem.indexOf(u8, src, "pub fn waitHostStats(").?;
    const body = src[at..std.mem.indexOfPos(u8, src, at, "\n    }\n").?];
    try std.testing.expect(std.mem.indexOf(u8, body, "nap.napOnce(w, seen,") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "if (can_nap) {").? < std.mem.indexOf(u8, body, "std.c.nanosleep(").?);
    try std.testing.expect(std.mem.indexOf(u8, body, "20 * std.time.ns_per_us") == null);
}

// -- the partner split of the Engram records over the host lane ----------------------------------------------------------

const partner = @import("engram_partner.zig");

fn recordOf(id: i64, out: []u8) void {
    for (out, 0..) |*b, k| b.* = @truncate(@as(u64, @bitCast(id)) *% 131 +% k *% 7 +% 5);
}

const PReader = struct {
    reads: usize = 0, // records read
    calls: usize = 0,
    delay_ns: u64 = 0, // a slow drive
    entered_ns: u64 = 0, // when the step's splitRead was entered (set by the test)
    lag_max_ns: u64 = 0, // the longest time from entering a step to its first read

    fn read(ctx: *anyopaque, ids: []const i64, dsts: []const [*]u8) anyerror!void {
        const self: *PReader = @ptrCast(@alignCast(ctx));
        if (self.calls == 0 or self.entered_ns != 0) {
            self.lag_max_ns = @max(self.lag_max_ns, nowNsT() -| self.entered_ns);
            self.entered_ns = 0;
        }
        self.calls += 1;
        self.reads += ids.len;
        if (self.delay_ns != 0) sleep(self.delay_ns);
        for (ids, dsts) |id, d| recordOf(id, d[0..roci.host_rec]);
    }
};

fn nowNsT() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const PRun = struct {
    net: *Net,
    steps: usize,
    link: [2]partner.Link = undefined,
    rd: [2]PReader = .{ .{}, .{} },
    outcome: [2][64]partner.Outcome = undefined,
    bad: [2]u64 = .{ 0, 0 },
    err: [2]?anyerror = .{ null, null },
    /// the lists differ at these steps on the second node (a diverged cache)
    diverge_every: usize = 0,
    late_start_ns: u64 = 0, // the second node starts each step this late
    slack_ns: u64 = std.time.ns_per_s, // what a node gives its partner beyond its own read (long: no timing decides a test unless it sets this)
    delay0_ns: u64 = 0, // the first node's reads take this long
    skip1: u8 = 0, // the second node starts out backing off for this many steps
    step_ns_max: [2]u64 = .{ 0, 0 }, // the longest a node spent in one step
    slow_half_ns: u64 = 0, // the second node's reads take this long
    seed: u64 = 11,
};

fn partnerMain(run: *PRun, who: usize) void {
    partnerBody(run, who) catch |e| {
        run.err[who] = e;
    };
}

fn partnerBody(run: *PRun, who: usize) !void {
    const a = std.heap.page_allocator;
    const l = &run.link[who];
    if (who == 1) run.rd[1].delay_ns = run.slow_half_ns else run.rd[0].delay_ns = run.delay0_ns;
    var prng = std.Random.DefaultPrng.init(run.seed);
    const rng = prng.random();
    var stage: [3000 * roci.host_rec]u8 = undefined;
    for (0..run.steps) |step| {
        const m = 1 + rng.uintLessThan(usize, 2999);
        const ids = try a.alloc(i64, m);
        defer a.free(ids);
        for (ids) |*x| x.* = rng.intRangeLessThan(i64, 0, 5000);
        // a diverged cache on the second node: one id differs
        if (who == 1 and run.diverge_every != 0 and step % run.diverge_every == 0) ids[m / 2] = 9999;
        const dsts = try a.alloc([*]u8, m);
        defer a.free(dsts);
        for (dsts, 0..) |*d, i| d.* = stage[i * roci.host_rec ..].ptr;
        @memset(stage[0 .. m * roci.host_rec], 0);
        if (who == 1 and run.late_start_ns != 0) sleep(run.late_start_ns);
        const t_enter = nowNsT();
        run.rd[who].entered_ns = t_enter;
        const out = try partner.splitRead(l, ids, dsts, step, .{ .ctx = &run.rd[who], .read = PReader.read }, roci.host_rec);
        run.step_ns_max[who] = @max(run.step_ns_max[who], nowNsT() - t_enter);
        if (step < 64) run.outcome[who][step] = out;
        // the stage holds exactly what a whole local read gives
        var want: [roci.host_rec]u8 = undefined;
        for (ids, 0..) |id, i| {
            recordOf(id, &want);
            if (!std.mem.eql(u8, &want, stage[i * roci.host_rec ..][0..roci.host_rec])) run.bad[who] += 1;
        }
    }
}

fn runPartners(run: *PRun) !void {
    for (&run.link, 0..) |*l, who| {
        const g: usize = who * 2;
        l.* = .{ .t = run.net.nodes[g].t, .arena = run.net.nodes[g].arena, .role = @intCast(who), .gpa = std.heap.page_allocator };
        l.slack_ns = run.slack_ns;
        if (who == 1) l.skip = run.skip1;
    }
    defer for (&run.link) |*l| l.deinit();
    var th: [2]std.Thread = undefined;
    for (&th, 0..) |*t, who| t.* = try std.Thread.spawn(.{}, partnerMain, .{ run, who });
    for (th) |t| t.join();
}

test "partner split: each partner reads half of the misses, the stages equal a whole local read, the 16 slots wrap" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: PRun = .{ .net = net, .steps = 40 };
    try runPartners(&run);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    for (0..40) |s| for (0..2) |w| try std.testing.expectEqual(partner.Outcome.split, run.outcome[w][s]);
    // each read about half: together a whole list once (not twice)
    var total: usize = 0;
    var prng = std.Random.DefaultPrng.init(11);
    for (0..40) |_| {
        const m = 1 + prng.random().uintLessThan(usize, 2999);
        total += m;
        for (0..m) |_| _ = prng.random().intRangeLessThan(i64, 0, 5000);
    }
    try std.testing.expectEqual(total, run.rd[0].reads + run.rd[1].reads);
    try std.testing.expect(run.rd[0].reads * 10 < total * 6 and run.rd[1].reads * 10 < total * 6);
}

test "control: a NIC that writes the words before the data is seen: a stale header is a mismatch (both read everything) or stale records land in a stage" {
    const net = try Net.open(.{ .host = true, .word_first = true, .gap_ns = 3 * std.time.ns_per_ms }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: PRun = .{ .net = net, .steps = 6 };
    try runPartners(&run);
    // the ordering the hardware keeps matters: with it broken, either the header check caught the stale header, or stale records were read
    const caught = run.link[0].c.mismatch + run.link[1].c.mismatch + run.link[0].c.header_late + run.link[1].c.header_late;
    try std.testing.expect(caught > 0 or run.bad[0] + run.bad[1] > 0 or run.err[0] != null or run.err[1] != null);
}

test "partner split: a diverged list on one partner makes both read everything, and the stages still equal a whole read" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: PRun = .{ .net = net, .steps = 30, .diverge_every = 3 };
    try runPartners(&run);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    for (0..30) |s| {
        const want: partner.Outcome = if (s % 3 == 0) .local_mismatch else .split;
        for (0..2) |w| try std.testing.expectEqual(want, run.outcome[w][s]);
    }
    try std.testing.expect(run.link[0].c.mismatch == 10 and run.link[1].c.mismatch == 10);
}

test "partner split: a partner late with its header is read around, and the stages are still right on both" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: PRun = .{ .net = net, .steps = 6, .late_start_ns = 120 * std.time.ns_per_ms, .slack_ns = 20 * std.time.ns_per_ms };
    try runPartners(&run);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    try std.testing.expect(run.link[0].c.header_late >= 1);
}

test "partner split: a node reads its half before the partner's header is in (a partner that starts 40 ms late costs the other node's read nothing)" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: PRun = .{ .net = net, .steps = 4, .late_start_ns = 40 * std.time.ns_per_ms };
    try runPartners(&run);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    for (0..4) |s| for (0..2) |w| try std.testing.expectEqual(partner.Outcome.split, run.outcome[w][s]);
    // node 0 began each step 40 ms before its partner: its first read waited for nothing
    try std.testing.expect(run.rd[0].lag_max_ns < 15 * std.time.ns_per_ms);
}

test "partner split: a partner that is late costs a node at most its half read, a wait as long as that read, and the rest (not the old fixed deadlines)" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    // node 0 reads in 10 ms a call; node 1 takes 300 ms a call, so it is late with its records every step
    var run: PRun = .{ .net = net, .steps = 4, .slow_half_ns = 300 * std.time.ns_per_ms, .delay0_ns = 10 * std.time.ns_per_ms, .slack_ns = 2 * std.time.ns_per_ms };
    try runPartners(&run);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    try std.testing.expect(run.link[0].c.half_late + run.link[0].c.header_late >= 1);
    // without the early wake: half 10 + records deadline (3 x 10 + 20 = 50) + rest 10 >= 70 ms; with it: half 10 + wait <= 10 + 2 + rest 10 = about 32 ms
    try std.testing.expect(run.step_ns_max[0] < 55 * std.time.ns_per_ms);
}

test "partner split: a node that was let down backs off (whole lists for 2, 4, 8 steps), so a persistently late partner costs one probe now and then" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: PRun = .{ .net = net, .steps = 10, .slow_half_ns = 120 * std.time.ns_per_ms, .delay0_ns = 5 * std.time.ns_per_ms, .slack_ns = 2 * std.time.ns_per_ms };
    try runPartners(&run);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    const c = run.link[0].c;
    // probes at steps 0, 3 and 8 (after 2 and 4 steps of backing off): three were let down, seven steps read whole lists
    try std.testing.expectEqual(@as(u64, 3), c.half_late + c.header_late);
    try std.testing.expectEqual(@as(u64, 7), c.skipped);
    for (0..10) |s| {
        const o = run.outcome[0][s];
        if (s == 0 or s == 3 or s == 8) {
            try std.testing.expect(o == .partner_half_late or o == .local_header_late);
        } else try std.testing.expectEqual(partner.Outcome.skipped, o);
    }
}

test "partner split: a partner that reads its whole list says so in its header, and the other node reads the rest at once, without waiting" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: PRun = .{ .net = net, .steps = 6, .skip1 = 100, .delay0_ns = 10 * std.time.ns_per_ms, .slack_ns = std.time.ns_per_s };
    try runPartners(&run);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    for (0..6) |s| try std.testing.expectEqual(partner.Outcome.partner_declined, run.outcome[0][s]);
    try std.testing.expectEqual(@as(u64, 6), run.link[0].c.declined);
    try std.testing.expectEqual(@as(u64, 6), run.link[1].c.skipped);
    // it did not sit out its (one second) budget on the second node's records, which were not coming
    try std.testing.expect(run.step_ns_max[0] < 200 * std.time.ns_per_ms);
    // and it was not let down: no backing off of its own
    try std.testing.expectEqual(@as(u64, 0), run.link[0].c.skipped);
}

test "partner split: a partner late with its records is read around (the half read here), and the stages are still right" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: PRun = .{ .net = net, .steps = 5, .slow_half_ns = 200 * std.time.ns_per_ms, .slack_ns = 5 * std.time.ns_per_ms };
    try runPartners(&run);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    try std.testing.expect(run.link[0].c.half_late >= 1);
}

test "partner split: the counters name the records read, sent and received, the waits and the read of the half against the whole list" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false }, roci.identity);
    defer net.close();
    try net.start();
    var run: PRun = .{ .net = net, .steps = 30, .diverge_every = 3, .slow_half_ns = 2 * std.time.ns_per_ms };
    try runPartners(&run);
    for (run.err) |e| try std.testing.expectEqual(@as(?anyerror, null), e);
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
    // the lists' lengths, as the run drew them: a split step reads its half, sends it and gets the other; a mismatched step reads the whole list
    var prng = std.Random.DefaultPrng.init(11);
    var want_read: [2]u64 = .{ 0, 0 };
    var want_sent: [2]u64 = .{ 0, 0 };
    var want_got: [2]u64 = .{ 0, 0 };
    for (0..30) |s| {
        const m = 1 + prng.random().uintLessThan(usize, 2999);
        for (0..m) |_| _ = prng.random().intRangeLessThan(i64, 0, 5000);
        for (0..2) |w| {
            const mine = partner.mineCount(m, @intCast(w));
            want_sent[w] += mine; // (the records go out whatever the headers say)
            if (s % 3 == 0) {
                want_read[w] += m;
            } else {
                want_read[w] += mine;
                want_got[w] += m - mine;
            }
        }
    }
    for (0..2) |w| {
        const c = run.link[w].c;
        try std.testing.expectEqual(@as(u64, 30), c.steps);
        try std.testing.expectEqual(@as(u64, 10), c.mismatch);
        try std.testing.expectEqual(want_read[w], c.records_read);
        try std.testing.expectEqual(want_sent[w], c.records_sent);
        try std.testing.expectEqual(want_got[w], c.records_got);
        try std.testing.expectEqual(@as(u64, 50), c.waits); // a header wait each step and a records wait of each split step
        try std.testing.expect(c.wait_max_ns > 0 and c.wait_max_ns <= c.hdr_wait_ns + c.rec_wait_ns);
        try std.testing.expect(c.wait_looks >= c.wait_naps);
        // the slow node's half took 2 ms a step at least; the whole list at the same rate is more than the half
        try std.testing.expect(c.read_ns > 0 and c.full_ns >= c.read_ns);
        try std.testing.expect(c.copy_ns > 0);
    }
    try std.testing.expect(run.link[1].c.read_ns >= 30 * 2 * std.time.ns_per_ms);
    // the line's difference: counts since a copy, the longest wait kept
    var base = run.link[0].c;
    base.steps -= 10;
    base.wait_max_ns = 1;
    const d = run.link[0].c.since(base);
    try std.testing.expectEqual(@as(u64, 10), d.steps);
    try std.testing.expectEqual(run.link[0].c.wait_max_ns, d.wait_max_ns);
    try std.testing.expectEqual(@as(u64, 0), d.records_read);
}

test "a rank that refuses a primitive fails its lanes at once: the refused reason is named, the host place is logged, and the failed lane's check ends every wait with an error" {
    const net = try Net.open(.{ .host = true }, .{ .nap = false, .dead_ms = 100_000 }, roci.identity);
    defer net.close();
    try net.start();
    const nd = &net.nodes[0];
    try std.testing.expect(nd.t.failure() == null);
    nd.t.fail(.refused, .own1, 0);
    try std.testing.expectEqual(roci.FailReason.refused, nd.t.failure().?.reason);
    try std.testing.expectError(error.RociFailed, nd.t.check());
    try std.testing.expectError(error.RociFailed, nd.t.waitHost(.rec, 1, 0));
}

test "the arena and the plans follow the booted pass rows: 4,096 unless more (setCap), the plans of the whole pass and the resume unit" {
    const keep = roci.max_cap;
    const keep_caps = roci.Transport.shape_caps;
    defer {
        roci.max_cap = keep;
        roci.Transport.shape_caps = keep_caps;
    }
    const roci_engine = @import("roci_engine.zig");
    var totals: [3]u64 = undefined;
    for ([_]usize{ 2048, 4096, 8192 }, 0..) |p, i| {
        roci.setCap(p);
        totals[i] = roci.layout().total;
        // never below the baseline layout: a 2,048-row pass lays the arena out as before
        try std.testing.expectEqual(@max(@as(usize, 4096), p), roci.max_cap);
        try std.testing.expectEqual([2]usize{ @max(@as(usize, 4096), p), 2048 }, roci.Transport.shape_caps);
        // the whole pass's plan and the resume unit's exist; any other cap has none; any-size plans reach the pass
        _ = try roci_engine.Roci.shapeOf(p);
        _ = try roci_engine.Roci.shapeOf(2048);
        try std.testing.expectError(error.NoPlanForRows, roci_engine.Roci.shapeOf(roci.max_cap + 2048));
        _ = try roci_engine.Roci.shapeFor(p - 100, p, true, 0);
        try std.testing.expectError(error.NoPlanForRows, roci_engine.Roci.shapeFor(roci.max_cap + 1, p, true, 0));
        // the events of the whole pass are built at its rows, every class, every node
        for (0..roci.n_classes) |c| for (0..4) |g| {
            const ev = roci.buildEvent(@fromBackingInt(@intCast(c)), p, g, roci.identity);
            try std.testing.expect(ev.class == @as(roci.Class, @fromBackingInt(@intCast(c))));
        };
    }
    try std.testing.expectEqual(totals[0], totals[1]); // 2,048 and 4,096 lay out alike
    try std.testing.expect(totals[2] > totals[1] and totals[2] <= 2 * totals[1]); // 8,192 about twice
    roci.setCap(4096);
    try std.testing.expectEqual(keep_caps, roci.Transport.shape_caps);
    try std.testing.expectEqual(totals[1], roci.layout().total);
}
