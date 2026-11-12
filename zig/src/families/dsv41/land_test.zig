//! The landing's tests over a fake comm: one thread per rank, rank 0's frames on in-memory queues, votes and
//! broadcasts as barriers every rank must reach, "device memory" in host buffers.
const std = @import("std");
const land = @import("land.zig");
const Span = land.Span;
const Plane = land.Plane;
const Frame = land.Frame;
const io = std.testing.io;
const gpa = std.testing.allocator;

test "plan: the kept buffer, then each plane's rows of [base, base + at) in copyExtent's order" {
    const planes = [_]Plane{ .{ .plane = 1000, .row = 4, .ratio = 4 }, .{ .plane = 5000, .row = 2, .ratio = 4 }, .{ .plane = 9000, .row = 8, .ratio = 128 } };
    var out: [8]Span = undefined;
    const s = try land.plan(77, 50, &planes, 256, 512, &out);
    try std.testing.expectEqual(@as(usize, 4), s.len);
    try std.testing.expectEqual(Span{ .addr = 77, .len = 50 }, s[0]);
    try std.testing.expectEqual(Span{ .addr = 1000 + 64 * 4, .len = 128 * 4 }, s[1]); // rows 64..192 of ratio 4
    try std.testing.expectEqual(Span{ .addr = 5000 + 64 * 2, .len = 128 * 2 }, s[2]);
    try std.testing.expectEqual(Span{ .addr = 9000 + 2 * 8, .len = 4 * 8 }, s[3]); // rows 2..6 of ratio 128
    try std.testing.expectEqual(@as(usize, 50 + 512 + 256 + 32), land.total(s));
    try std.testing.expectError(error.BadView, land.plan(77, 50, &planes, 130, 512, &out)); // base off the ratio-4 rows
    try std.testing.expectError(error.BadView, land.plan(77, 50, &planes, 256, 500, &out));
    try std.testing.expectError(error.TooManySpans, land.plan(77, 50, &planes, 256, 512, out[0..2]));
}

test "cut: a piece's sub-spans cover exactly its bytes, split at span and piece boundaries" {
    const spans = [_]Span{ .{ .addr = 100, .len = 10 }, .{ .addr = 500, .len = 5 }, .{ .addr = 900, .len = 20 } };
    var out: [4]Span = undefined;
    var off: usize = 0;
    var seen: usize = 0;
    const total = land.total(&spans);
    for ([_]usize{ 1, 3, 7, 10, 35 }) |piece| {
        off = 0;
        seen = 0;
        while (off < total) : (off += piece) {
            const len = @min(piece, total - off);
            const c = try land.cut(&spans, off, len, &out);
            try std.testing.expectEqual(len, land.total(c));
            for (c) |x| {
                // every sub-span lies inside one span, at the right offset of the concatenation
                var base: usize = 0;
                const inside = for (spans) |sp| {
                    if (x.addr >= sp.addr and x.addr + x.len <= sp.addr + sp.len) {
                        try std.testing.expectEqual(off + (seen - off), base + (x.addr - sp.addr));
                        break true;
                    }
                    base += sp.len;
                } else false;
                try std.testing.expect(inside);
                seen += x.len;
            }
        }
        try std.testing.expectEqual(total, seen);
    }
    try std.testing.expectError(error.OutOfRange, land.cut(&spans, 30, 6, &out));
    try std.testing.expectError(error.TooManySpans, land.cut(&spans, 0, 35, out[0..2]));
}

// -- the fake comm ---------------------------------------------------------------------------------------------------

const world = 4;
const mem_bytes = 4096;

/// What rank 0 sends a follower: a landing frame, an unrelated round (a collective of its own), or the end.
const Msg = union(enum) { land: Frame, round: u32, done };

const Op = struct { kind: enum { vote, bcast, round }, bytes: usize };

/// Every rank's memory, rank 0's queues to the followers, and one barrier at a time.
const Net = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    mem: [world][mem_bytes]u8 = undefined,
    queues: [world]std.ArrayList(Msg) = @splat(.empty),
    log: [world]std.ArrayList(Op) = @splat(.empty),
    // the barrier: who arrived with what, and its generation
    arrived: usize = 0,
    gen: u64 = 0,
    kind: [world]Op = undefined,
    oks: [world]bool = undefined,
    spans: [world][]const Span = undefined,
    result: bool = false,
    mismatch: bool = false,

    fn deinit(n: *Net) void {
        for (&n.queues) |*q| q.deinit(gpa);
        for (&n.log) |*l| l.deinit(gpa);
    }

    fn addr(n: *Net, rank: usize, off: usize) u64 {
        return @intFromPtr(&n.mem[rank]) + off;
    }

    fn send(n: *Net, m: Msg) !void {
        n.mutex.lockUncancelable(io);
        defer n.mutex.unlock(io);
        for (n.queues[1..]) |*q| try q.append(gpa, m);
        n.cond.broadcast(io);
    }

    fn next(n: *Net, rank: usize) Msg {
        n.mutex.lockUncancelable(io);
        defer n.mutex.unlock(io);
        while (n.queues[rank].items.len == 0) n.cond.waitUncancelable(io, &n.mutex);
        return n.queues[rank].orderedRemove(0);
    }

    /// Every rank reaches the same collective (kind and bytes); the last one computes it for all.
    fn collective(n: *Net, rank: usize, op: Op, ok: bool, spans: []const Span) !bool {
        n.mutex.lockUncancelable(io);
        defer n.mutex.unlock(io);
        try n.log[rank].append(gpa, op);
        const my_gen = n.gen;
        n.kind[rank] = op;
        n.oks[rank] = ok;
        n.spans[rank] = spans;
        n.arrived += 1;
        if (n.arrived == world) {
            var all = true;
            for (n.kind[1..]) |k| if (k.kind != n.kind[0].kind or k.bytes != n.kind[0].bytes) {
                n.mismatch = true;
            };
            for (n.oks) |o| all = all and o;
            if (op.kind == .bcast and !n.mismatch) { // rank 0's bytes into every follower's spans, in order
                var src: std.ArrayList(u8) = .empty;
                defer src.deinit(gpa);
                for (n.spans[0]) |s| try src.appendSlice(gpa, @as([*]const u8, @ptrFromInt(s.addr))[0..s.len]);
                for (n.spans[1..]) |dst| {
                    var at: usize = 0;
                    for (dst) |s| {
                        @memcpy(@as([*]u8, @ptrFromInt(s.addr))[0..s.len], src.items[at .. at + s.len]);
                        at += s.len;
                    }
                }
            }
            n.result = all;
            n.arrived = 0;
            n.gen += 1;
            n.cond.broadcast(io);
        } else while (n.gen == my_gen) n.cond.waitUncancelable(io, &n.mutex);
        if (n.mismatch) return error.CollectiveMismatch;
        return n.result;
    }
};

/// A rank's view of the net, as land.Hooks.
const Rank = struct {
    net: *Net,
    rank: usize,
    idle: bool = true,
    in_flight: bool = false, // rank 0: a piece's event not yet complete
    complete_next: bool = false, // the fake event completes at the next `marked`
    sent_since_tick: usize = 0,
    most_in_flight: usize = 0,

    fn hooks(r: *Rank) land.Hooks {
        return .{ .ctx = r, .frame = frame, .vote = vote, .bcast = bcast, .idle = idleFn, .mark = mark, .marked = marked, .wait = wait };
    }
    fn of(ctx: *anyopaque) *Rank {
        return @ptrCast(@alignCast(ctx));
    }
    fn frame(ctx: *anyopaque, f: Frame) anyerror!void {
        try of(ctx).net.send(.{ .land = f });
    }
    fn vote(ctx: *anyopaque, ok: bool) anyerror!bool {
        const r = of(ctx);
        return r.net.collective(r.rank, .{ .kind = .vote, .bytes = 1 }, ok, &.{});
    }
    fn bcast(ctx: *anyopaque, spans: []const Span) anyerror!void {
        const r = of(ctx);
        _ = try r.net.collective(r.rank, .{ .kind = .bcast, .bytes = land.total(spans) }, true, spans);
        if (r.rank == 0) {
            r.sent_since_tick += 1;
            r.most_in_flight = @max(r.most_in_flight, r.sent_since_tick);
        }
    }
    fn idleFn(ctx: *anyopaque) bool {
        return of(ctx).idle;
    }
    fn mark(ctx: *anyopaque) anyerror!void {
        of(ctx).in_flight = true;
    }
    fn marked(ctx: *anyopaque) bool {
        const r = of(ctx);
        if (r.in_flight and r.complete_next) {
            r.in_flight = false;
            r.complete_next = false;
        }
        return !r.in_flight;
    }
    fn wait(ctx: *anyopaque) void {
        of(ctx).in_flight = false;
    }
    /// A service point passes: the piece in flight completes at the next look.
    fn tick(r: *Rank) void {
        r.complete_next = true;
        r.sent_since_tick = 0;
    }
};

/// A follower's buffers: offsets in its memory (address = mem + offset), counted when made and freed.
const Buf = struct { ptr: u64 };

const FollowerRig = struct {
    net: *Net,
    rank: usize,
    hooks_rank: Rank,
    f: land.Follower(Buf),
    next_off: usize = 1024, // pool buffers are carved from here on
    small_bytes: usize,
    adopted: std.ArrayList(struct { id: u64, buf: Buf }) = .empty,
    made: usize = 0,
    err: ?anyerror = null,

    fn alloc(ctx: *anyopaque) anyerror!Buf {
        const r: *FollowerRig = @ptrCast(@alignCast(ctx));
        if (r.next_off + r.small_bytes > mem_bytes) return error.OutOfMemory;
        const b: Buf = .{ .ptr = r.net.addr(r.rank, r.next_off) };
        r.next_off += r.small_bytes;
        r.made += 1;
        return b;
    }

    fn freeBuf(_: *anyopaque, _: Buf) void {}

    fn run(r: *FollowerRig) void {
        r.loop() catch |e| {
            r.err = e;
        };
    }

    fn loop(r: *FollowerRig) !void {
        while (true) switch (r.net.next(r.rank)) {
            .done => return,
            .round => |bytes| _ = try r.net.collective(r.rank, .{ .kind = .round, .bytes = bytes }, true, &.{}),
            .land => |fr| if (try r.f.onFrame(r.hooks_rank.hooks(), fr)) |a| {
                try r.adopted.append(gpa, .{ .id = a.id, .buf = a.buf });
                r.f.refill(); // outside any collective
            },
        };
    }
};

/// Rank 0's layout: its kept buffer at 0 (small bytes), three planes; each follower's planes sit elsewhere.
const small = 40;
fn planesOf(n: *Net, rank: usize) [3]Plane {
    const shift: usize = if (rank == 0) 0 else 64 * rank; // followers' planes at other addresses than rank 0's
    return .{
        .{ .plane = n.addr(rank, 100 + shift), .row = 2, .ratio = 4 },
        .{ .plane = n.addr(rank, 400 + shift), .row = 1, .ratio = 4 },
        .{ .plane = n.addr(rank, 700 + shift), .row = 4, .ratio = 128 },
    };
}

const Rig = struct {
    net: Net = .{},
    r0: Rank = undefined,
    lander: land.Lander = undefined,
    followers: [world - 1]FollowerRig = undefined,
    threads: [world - 1]std.Thread = undefined,
    planes0: [3]Plane = undefined,
    fplanes: [world - 1][3]Plane = undefined,

    fn start(rig: *Rig, slots: [world - 1]usize, smalls: [world - 1]usize, piece: usize) !void {
        for (&rig.net.mem, 0..) |*m, r| for (m, 0..) |*b, i| {
            b.* = @truncate(i *% 31 +% r *% 7 +% 1);
        };
        rig.r0 = .{ .net = &rig.net, .rank = 0 };
        rig.planes0 = planesOf(&rig.net, 0);
        rig.lander = .{ .hooks = rig.r0.hooks(), .piece = piece };
        for (&rig.followers, &rig.fplanes, 0..) |*f, *pl, k| {
            pl.* = planesOf(&rig.net, k + 1);
            f.* = .{ .net = &rig.net, .rank = k + 1, .hooks_rank = .{ .net = &rig.net, .rank = k + 1 }, .small_bytes = smalls[k], .f = undefined };
            f.f = try land.Follower(Buf).init(gpa, .{ .ctx = f, .alloc = FollowerRig.alloc, .free = FollowerRig.freeBuf }, smalls[k], pl, slots[k]);
        }
        for (&rig.followers, &rig.threads) |*f, *t| t.* = try std.Thread.spawn(.{}, FollowerRig.run, .{f});
    }

    fn stop(rig: *Rig) !void {
        try rig.net.send(.done);
        for (rig.threads) |t| t.join();
        for (&rig.followers) |*f| {
            f.f.deinit(gpa);
            f.adopted.deinit(gpa);
        }
        rig.net.deinit();
        for (rig.followers) |f| if (f.err) |e| return e;
    }

    /// Step until the landing ends, a service point (tick) before each step.
    fn finish(rig: *Rig) land.Step {
        var steps: usize = 0;
        while (steps < 10_000) : (steps += 1) {
            rig.r0.tick();
            const s = rig.lander.step();
            if (s != .pending) return s;
        }
        return .failed;
    }
};

fn bytesAt(n: *Net, a: u64, len: usize) []const u8 {
    _ = n;
    return @as([*]const u8, @ptrFromInt(a))[0..len];
}

test "a landing: every rank holds rank 0's kept buffer and extent rows, other rows untouched; one buffer adopted a rank" {
    var rig: Rig = .{};
    try rig.start(.{ 2, 2, 2 }, .{ small, small, small }, 7);
    const base = 256;
    const at = 256;
    try rig.lander.begin(42, rig.net.addr(0, 0), small, &rig.planes0, base, at);
    const st = rig.finish();
    try rig.net.send(.{ .round = 0 }); // a fence: every follower has handled the adopt before the checks
    _ = try rig.net.collective(0, .{ .kind = .round, .bytes = 0 }, true, &.{});
    try std.testing.expectEqual(land.Step.landed, st);
    var spans0: [8]Span = undefined;
    const s0 = try land.plan(rig.net.addr(0, 0), small, &rig.planes0, base, at, &spans0);
    for (&rig.followers, rig.fplanes) |*f, pl| {
        try std.testing.expectEqual(@as(usize, 1), f.adopted.items.len);
        try std.testing.expectEqual(@as(u64, 42), f.adopted.items[0].id);
        var spans: [8]Span = undefined;
        const s = try land.plan(f.adopted.items[0].buf.ptr, small, &pl, base, at, &spans);
        for (s, s0) |x, y| try std.testing.expectEqualSlices(u8, bytesAt(&rig.net, y.addr, y.len), bytesAt(&rig.net, x.addr, x.len));
        // the row just before and just after the extent in the first plane keep this rank's own bytes
        const p = pl[0];
        const before = p.plane + (base / p.ratio - 1) * p.row;
        const after = p.plane + ((base + at) / p.ratio) * p.row;
        const own = &rig.net.mem[f.rank];
        const off_b = before - rig.net.addr(f.rank, 0);
        const off_a = after - rig.net.addr(f.rank, 0);
        try std.testing.expectEqual(@as(u8, @truncate(off_b *% 31 +% f.rank *% 7 +% 1)), own[off_b]);
        try std.testing.expectEqual(@as(u8, @truncate(off_a *% 31 +% f.rank *% 7 +% 1)), own[off_a]);
        try std.testing.expectEqual(@as(usize, 2), f.f.free()); // refilled after the adopt
        try std.testing.expectEqual(@as(usize, 3), f.made);
    }
    try std.testing.expectEqual(@as(u64, 1), rig.lander.counts.landings);
    try std.testing.expectEqual(@as(u64, land.total(s0)), rig.lander.counts.bytes);
    try std.testing.expectEqual(@as(u64, (land.total(s0) + 6) / 7), rig.lander.counts.pieces);
    try rig.stop();
}

test "a follower with no free buffer: every rank skips after the vote, no broadcast, every buffer back in its pool" {
    var rig: Rig = .{};
    try rig.start(.{ 1, 0, 1 }, .{ small, small, small }, 64);
    try rig.lander.begin(7, rig.net.addr(0, 0), small, &rig.planes0, 256, 256);
    try std.testing.expectEqual(land.Step.refused, rig.finish());
    try rig.net.send(.{ .round = 0 });
    _ = try rig.net.collective(0, .{ .kind = .round, .bytes = 0 }, true, &.{});
    for (rig.net.log) |l| for (l.items) |op| try std.testing.expect(op.kind != .bcast);
    try std.testing.expectEqual(@as(usize, 1), rig.followers[0].f.free());
    try std.testing.expectEqual(@as(usize, 0), rig.followers[1].f.free());
    try std.testing.expectEqual(@as(usize, 1), rig.followers[2].f.free());
    try std.testing.expectEqual(@as(u64, 1), rig.lander.counts.skipped);
    try rig.stop();
}

test "one rank's state is another size: every rank skips" {
    var rig: Rig = .{};
    try rig.start(.{ 1, 1, 1 }, .{ small, small + 8, small }, 64);
    try rig.lander.begin(7, rig.net.addr(0, 0), small, &rig.planes0, 256, 256);
    try std.testing.expectEqual(land.Step.refused, rig.finish());
    try rig.net.send(.{ .round = 0 });
    _ = try rig.net.collective(0, .{ .kind = .round, .bytes = 0 }, true, &.{});
    for (rig.followers) |f| try std.testing.expectEqual(@as(usize, 1), f.f.free());
    try rig.stop();
}

test "an abort between pieces: the followers give their buffers back and adopt nothing" {
    var rig: Rig = .{};
    try rig.start(.{ 1, 1, 1 }, .{ small, small, small }, 16);
    try rig.lander.begin(9, rig.net.addr(0, 0), small, &rig.planes0, 256, 256);
    var pieces: usize = 0;
    while (pieces < 3) {
        rig.r0.tick();
        try std.testing.expectEqual(land.Step.pending, rig.lander.step());
        pieces = rig.lander.counts.pieces;
    }
    rig.lander.abort();
    try std.testing.expectEqual(@as(u64, 1), rig.lander.counts.aborted);
    try rig.net.send(.{ .round = 0 });
    _ = try rig.net.collective(0, .{ .kind = .round, .bytes = 0 }, true, &.{});
    for (rig.followers) |f| {
        try std.testing.expectEqual(@as(usize, 1), f.f.free());
        try std.testing.expectEqual(@as(usize, 0), f.adopted.items.len);
    }
    try std.testing.expectEqual(land.Step.failed, rig.lander.step()); // nothing in flight
    try rig.stop();
}

test "pieces between unrelated rounds keep every rank's collectives in the same order" {
    var rig: Rig = .{};
    try rig.start(.{ 1, 1, 1 }, .{ small, small, small }, 9);
    try rig.lander.begin(5, rig.net.addr(0, 0), small, &rig.planes0, 256, 256);
    var round: u32 = 0;
    var st: land.Step = .pending;
    while (st == .pending) : (round += 1) {
        try rig.net.send(.{ .round = round }); // a round of its own collective between every two steps
        _ = try rig.net.collective(0, .{ .kind = .round, .bytes = round }, true, &.{});
        rig.r0.tick();
        st = rig.lander.step();
    }
    try std.testing.expectEqual(land.Step.landed, st);
    try rig.net.send(.{ .round = 0 });
    _ = try rig.net.collective(0, .{ .kind = .round, .bytes = 0 }, true, &.{});
    for (rig.net.log[1..]) |l| {
        try std.testing.expectEqual(rig.net.log[0].items.len, l.items.len);
        for (rig.net.log[0].items, l.items) |a, b| try std.testing.expectEqual(a, b);
    }
    try rig.stop();
}

test "the fence: a step asked while rank 0's stream is busy sends nothing; pieces wait for the one in flight" {
    var rig: Rig = .{};
    try rig.start(.{ 1, 1, 1 }, .{ small, small, small }, 8);
    try rig.lander.begin(3, rig.net.addr(0, 0), small, &rig.planes0, 256, 256);
    rig.r0.idle = false;
    try std.testing.expectEqual(land.Step.pending, rig.lander.step()); // the open waits for an idle stream
    try std.testing.expectEqual(@as(usize, 0), rig.net.log[0].items.len);
    rig.r0.idle = true;
    try std.testing.expectEqual(land.Step.pending, rig.lander.step()); // open, vote, the first piece
    const after_first = rig.lander.counts.pieces;
    try std.testing.expectEqual(@as(u64, 1), after_first);
    try std.testing.expectEqual(land.Step.pending, rig.lander.step()); // its event has not completed: no piece
    try std.testing.expectEqual(after_first, rig.lander.counts.pieces);
    try std.testing.expectEqual(land.Step.landed, rig.finish());
    try std.testing.expectEqual(@as(usize, 1), rig.r0.most_in_flight);
    try rig.stop();
}

test "an open while another landing runs is refused: one landing at a time" {
    var rig: Rig = .{};
    try rig.start(.{ 2, 2, 2 }, .{ small, small, small }, 8);
    try rig.lander.begin(3, rig.net.addr(0, 0), small, &rig.planes0, 256, 256);
    try std.testing.expectError(error.LandingInFlight, rig.lander.begin(4, rig.net.addr(0, 0), small, &rig.planes0, 256, 256));
    try std.testing.expectEqual(land.Step.landed, rig.finish());
    try rig.lander.begin(4, rig.net.addr(0, 0), small, &rig.planes0, 0, 256); // the next one, once it ended
    try std.testing.expectEqual(land.Step.landed, rig.finish());
    try rig.stop();
}

test "a landing cancelled with a piece in flight (LOAD_WAIT) while another stream decodes: every rank gives its buffer back, every rank's collectives in one order" {
    var rig: Rig = .{};
    try rig.start(.{ 1, 1, 1 }, .{ small, small, small }, 8);
    try rig.lander.begin(11, rig.net.addr(0, 0), small, &rig.planes0, 256, 256);
    // the pieces go out between another stream's rounds (a collective of their own)
    var round: u32 = 0;
    while (rig.lander.counts.pieces < 3) : (round += 1) {
        try rig.net.send(.{ .round = round });
        _ = try rig.net.collective(0, .{ .kind = .round, .bytes = round }, true, &.{});
        rig.r0.tick();
        try std.testing.expectEqual(land.Step.pending, rig.lander.step());
    }
    try std.testing.expect(rig.r0.in_flight); // the third piece's event not complete: the cancel comes now
    rig.lander.abort(); // (the Leader's cancel when TF_SPILL_LOAD_WAIT_MS passes: Family.land's abort)
    try std.testing.expect(!rig.r0.in_flight); // it waited for the piece in flight before its frame
    try std.testing.expectEqual(@as(u64, 1), rig.lander.counts.aborted);
    // the other stream goes on decoding
    for (0..3) |k| {
        const b: u32 = 100 + @as(u32, @intCast(k));
        try rig.net.send(.{ .round = b });
        _ = try rig.net.collective(0, .{ .kind = .round, .bytes = b }, true, &.{});
    }
    for (rig.followers) |f| {
        try std.testing.expectEqual(@as(usize, 1), f.f.free());
        try std.testing.expectEqual(@as(usize, 0), f.adopted.items.len);
    }
    for (rig.net.log[1..]) |l| {
        try std.testing.expectEqual(rig.net.log[0].items.len, l.items.len);
        for (rig.net.log[0].items, l.items) |a, b| try std.testing.expectEqual(a, b);
    }
    // the next landing runs whole on the same buffers
    try rig.lander.begin(12, rig.net.addr(0, 0), small, &rig.planes0, 256, 256);
    try std.testing.expectEqual(land.Step.landed, rig.finish());
    try rig.stop();
}
