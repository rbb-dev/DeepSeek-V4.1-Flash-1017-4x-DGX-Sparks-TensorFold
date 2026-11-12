//! TF_DS_ROCI_PAIR_LOG's GPU spans: every pair's span (its first queued work to its second pass's last, in cuda events) is printed once its events
//! are done, however far the host has queued ahead. A log that kept ONE pair of events and printed the previous pair's span only if it was
//! already done when the next pair began would almost never print when the host queues ahead (one line a fill). So each pair records into its own slot
//! of a ring and a sweep (at every pair's start and every join) prints the finished ones in order. Log only: nothing here runs without the log.
const std = @import("std");

pub const cap = 256;

/// The outstanding pairs in issue order: a slot index per pair (the model's event arrays are indexed by it).
pub const Ring = struct {
    pair: [cap]u64 = undefined,
    head: usize = 0, // the oldest outstanding slot
    len: usize = 0,
    lost: u64 = 0, // pairs whose events were not kept because `cap` were outstanding

    /// The slot of a new pair, or null (counted lost) when `cap` pairs are still outstanding.
    pub fn push(r: *Ring, pair: u64) ?usize {
        if (r.len == cap) {
            r.lost += 1;
            return null;
        }
        const slot = (r.head + r.len) % cap;
        r.pair[slot] = pair;
        r.len += 1;
        return slot;
    }

    /// Prints, oldest first, every outstanding pair whose events are done (`ctx.done(slot)`), with `ctx.elapsed(slot)` ms, by `ctx.emit(pair, ms)`.
    /// Events finish in the order the pairs were queued, so the sweep stops at the first pair not done. Returns how many it printed.
    pub fn sweep(r: *Ring, ctx: anytype) usize {
        var n: usize = 0;
        while (r.len > 0 and ctx.done(r.head)) {
            ctx.emit(r.pair[r.head], ctx.elapsed(r.head));
            r.head = (r.head + 1) % cap;
            r.len -= 1;
            n += 1;
        }
        return n;
    }
};

/// A fake event model: every pair's events finish `late` sweeps after the pair was queued.
const Fake = struct {
    queued_at: [cap]usize = @splat(0),
    now: usize = 0,
    late: usize,
    printed: std.ArrayList(struct { pair: u64, ms: f32 }) = .empty,

    pub fn done(f: *Fake, slot: usize) bool {
        return f.now >= f.queued_at[slot] + f.late;
    }
    pub fn elapsed(_: *Fake, slot: usize) f32 {
        return 1000.0 + @as(f32, @floatFromInt(slot));
    }
    pub fn emit(f: *Fake, pair: u64, ms: f32) void {
        f.printed.append(std.testing.allocator, .{ .pair = pair, .ms = ms }) catch unreachable;
    }
};

test "every pair's span is printed once its events are done, even when they finish many pairs late (the old single-event log printed one a fill)" {
    for ([_]usize{ 0, 1, 5, 40 }) |late| {
        var f: Fake = .{ .late = late };
        defer f.printed.deinit(std.testing.allocator);
        var r: Ring = .{};
        const n: u64 = 100;
        for (1..n + 1) |pair| {
            f.now += 1;
            // the sweep at the next pair's start
            _ = r.sweep(&f);
            const slot = r.push(pair).?;
            f.queued_at[slot] = f.now;
        }
        // the fill is over and the events are all done: the last sweeps (a join) print the rest
        f.now += late + 1;
        _ = r.sweep(&f);
        try std.testing.expectEqual(@as(usize, n), f.printed.items.len);
        for (f.printed.items, 1..) |p, want| try std.testing.expectEqual(@as(u64, want), p.pair); // in issue order, once each
        try std.testing.expectEqual(@as(usize, 0), r.len);
        try std.testing.expectEqual(@as(u64, 0), r.lost);
    }
}

test "the old rule (print the previous pair only if already done when the next begins) prints almost nothing when events finish late: the control that the fake model can fail" {
    var f: Fake = .{ .late = 5 };
    defer f.printed.deinit(std.testing.allocator);
    var prev: ?usize = null;
    var q: usize = 0;
    for (1..101) |pair| {
        f.now += 1;
        if (prev) |slot| if (f.done(slot)) {
            f.emit(pair - 1, f.elapsed(slot));
            prev = null;
        };
        f.queued_at[0] = f.now;
        prev = 0;
        q += 1;
    }
    try std.testing.expect(f.printed.items.len < 5);
}

test "a full ring counts the pairs it could not keep and keeps printing the others in order" {
    var f: Fake = .{ .late = 1000 };
    defer f.printed.deinit(std.testing.allocator);
    var r: Ring = .{};
    for (1..cap + 11) |pair| {
        if (r.push(pair)) |slot| f.queued_at[slot] = 0;
    }
    try std.testing.expectEqual(@as(usize, cap), r.len);
    try std.testing.expectEqual(@as(u64, 10), r.lost);
    f.now = 5000;
    try std.testing.expectEqual(@as(usize, cap), r.sweep(&f));
    try std.testing.expectEqual(@as(u64, 1), f.printed.items[0].pair);
    try std.testing.expectEqual(@as(u64, cap), f.printed.items[cap - 1].pair);
    // the ring wraps: new pairs go on after the swept ones
    try std.testing.expectEqual(@as(usize, 0), r.push(999).?);
}
