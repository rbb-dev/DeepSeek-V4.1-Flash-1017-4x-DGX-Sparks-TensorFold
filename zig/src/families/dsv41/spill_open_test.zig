//! Rank 0's store open on its own thread with a bounded wait (spill_open.zig): a store that opens in time turns the
//! tier on, a slow one leaves it off and is closed by its own thread when it finally opens, a failed one is named.
const std = @import("std");
const spill_open = @import("spill_open.zig");
const io = std.testing.io;
const gpa = std.testing.allocator;

/// A store standing in for the Side: how long its open takes, whether it fails, and whether it was closed.
const Fake = struct {
    open_ms: u64 = 0,
    fail: ?anyerror = null,
    /// How long close lingers after marking the store closed: it widens the window before the opener frees itself, so
    /// a test that returns too early fails every run.
    linger_ms: u64 = 0,
    closed: std.atomic.Value(bool) = .init(false),
    opened: std.atomic.Value(bool) = .init(false),

    fn open(ctx: *anyopaque) anyerror!*Fake {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.open_ms > 0) std.Io.sleep(io, .fromMilliseconds(@intCast(f.open_ms)), .awake) catch {};
        if (f.fail) |e| return e;
        f.opened.store(true, .release);
        return f;
    }

    fn close(_: *anyopaque, side: *Fake) void {
        side.closed.store(true, .release);
        if (side.linger_ms > 0) std.Io.sleep(io, .fromMilliseconds(@intCast(side.linger_ms)), .awake) catch {};
    }
};

const O = spill_open.Opener(Fake);

/// The testing allocator, counting what is live: a test waits for an abandoned opener (detached, never joined) to free
/// itself before the runner checks for leaks.
const Counted = struct {
    live: std.atomic.Value(usize) = .init(0),

    fn allocator(c: *Counted) std.mem.Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const c: *Counted = @ptrCast(@alignCast(ctx));
        const p = gpa.rawAlloc(len, a, ra) orelse return null;
        _ = c.live.fetchAdd(1, .acq_rel);
        return p;
    }
    fn resize(_: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        return gpa.rawResize(m, a, n, ra);
    }
    fn remap(_: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        return gpa.rawRemap(m, a, n, ra);
    }
    fn free(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        const c: *Counted = @ptrCast(@alignCast(ctx));
        gpa.rawFree(m, a, ra);
        _ = c.live.fetchSub(1, .acq_rel);
    }
};

test "a store that opens within the bound: the tier gets it" {
    var f: Fake = .{ .open_ms = 5 };
    const o = try O.start(gpa, io, &f, Fake.open, Fake.close);
    const r = o.wait(5 * std.time.ns_per_s);
    try std.testing.expect(r == .opened);
    try std.testing.expectEqual(&f, r.opened);
    try std.testing.expect(!f.closed.load(.acquire)); // the engine owns it now
}

test "a store slower than the bound: the start goes on without it, and its own thread closes it when it opens" {
    var f: Fake = .{ .open_ms = 300, .linger_ms = 100 };
    var counted: Counted = .{};
    const o = try O.start(counted.allocator(), io, &f, Fake.open, Fake.close);
    const r = o.wait(20 * std.time.ns_per_ms);
    try std.testing.expect(r == .timeout); // (the opener is abandoned, never joined: it frees itself)
    // it closes the store once it opens, then frees itself: wait for that, not for the close alone
    var k: usize = 0;
    while (k < 3000 and counted.live.load(.acquire) != 0) : (k += 1) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    try std.testing.expect(f.opened.load(.acquire));
    try std.testing.expect(f.closed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), counted.live.load(.acquire));
}

test "a store whose open fails: the tier is off, the error named" {
    var f: Fake = .{ .fail = error.StoreLocked };
    const o = try O.start(gpa, io, &f, Fake.open, Fake.close);
    const r = o.wait(5 * std.time.ns_per_s);
    try std.testing.expectEqual(error.StoreLocked, r.failed);
    try std.testing.expect(!f.closed.load(.acquire));
}

test "the off reason names the bound or the error" {
    var buf: [120]u8 = undefined;
    try std.testing.expectEqualStrings("the store did not open within TF_DS_SPILL_OPEN_MS=300000", spill_open.reason(.timeout, 300_000, &buf));
    try std.testing.expectEqualStrings("the store did not open: StoreLocked", spill_open.reason(.{ .failed = error.StoreLocked }, 300_000, &buf));
}
