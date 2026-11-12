//! A wait for a GPU result that sleeps until shortly before the result is due and polls the last stretch.
//!
//! A blocking wait (CU_EVENT_BLOCKING_SYNC) costs the host its thread's wake-up after the GPU is done: MEASURED on rank 0 (deccap-r0)
//! cuEventSynchronize returns a median 69.7 us after its event, cuStreamSynchronize 68.6 us, and a decode round has two such waits.
//! A spinning wait has no such lag and burns a core for the whole round (what the blocking mode was made to stop). This one learns when
//! its wait ends (the time from the wait's start to the poll that saw the result, smoothed), sleeps to `margin` before that and polls
//! from there: the wake-up is paid inside the sleep, the poll catches the result within a poll's time, and the core is hot for about
//! `margin` of a 30-130 ms round. A wait that outlasts the poll window falls back to the blocking wait, so a slow round is never
//! polled for long. Off (`on` false) and while it has too few samples it is the blocking wait, and it learns from that one's ends.
const std = @import("std");

pub const Hybrid = struct {
    on: bool = false,
    ema_ns: f64 = 0,
    samples: u32 = 0,
    /// How long before the expected end the sleep stops and the polling starts.
    margin_ns: u64 = 250_000,
    /// The polling that gives up for the blocking wait.
    poll_max_ns: u64 = 2_000_000,
    /// Counters, for the instrument line: waits that slept then polled, those the result had already ended when the poll began, those that
    /// outlasted the poll window.
    polled: u64 = 0,
    early: u64 = 0,
    fell_back: u64 = 0,
    blocked: u64 = 0,

    pub const warm = 4;

    /// Waits for `ev` (`done() !bool`, `synchronize() !void`) with `clock` (`now() u64` ns, `sleep(ns)`).
    pub fn wait(h: *Hybrid, clock: anytype, ev: anytype) !void {
        const t0 = clock.now();
        if (!h.on or h.samples < warm) {
            try ev.synchronize();
            h.blocked += 1;
            h.learn(clock.now() - t0, false);
            return;
        }
        const due: u64 = @intFromFloat(@max(0.0, h.ema_ns));
        if (due > h.margin_ns) clock.sleep(due - h.margin_ns);
        var first = true;
        const poll_from = clock.now();
        while (!try ev.done()) {
            first = false;
            if (clock.now() - poll_from > h.poll_max_ns) {
                try ev.synchronize();
                h.fell_back += 1;
                h.learn(clock.now() - t0, false);
                return;
            }
            std.atomic.spinLoopHint();
        }
        h.polled += 1;
        const d = clock.now() - t0;
        if (first) {
            // already ended when the poll began: the true end is not seen, only that it came by now; step down a tenth (a wait that is
            // now too short polls a little longer, which the next sample corrects)
            h.early += 1;
            h.ema_ns *= 0.9;
        } else h.learn(d, true);
    }

    fn learn(h: *Hybrid, d: u64, exact: bool) void {
        const x: f64 = @floatFromInt(d);
        // a blocking wait's end is late by its wake-up: the estimate is for a result due, so the margin covers it
        if (h.samples == 0) h.ema_ns = x else h.ema_ns += 0.25 * (x - h.ema_ns);
        _ = exact;
        h.samples +%= 1;
    }
};

/// The host's clock for the wait.
pub const SystemClock = struct {
    pub fn now(_: SystemClock) u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    pub fn sleep(_: SystemClock, ns: u64) void {
        var ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
        _ = std.c.nanosleep(&ts, null);
    }
};

// -- tests: a virtual clock and an event that ends at a set time ----------------------------------------------------------------------

const Sim = struct {
    t: u64 = 0,
    ends: u64 = 0, // when the event ends
    wake_lag: u64 = 65_000, // a sleeping thread returns this late (a blocking wait's wake-up; a sleep's timer slack is small beside it)
    sleeps: u32 = 0,
    sync_calls: u32 = 0,
    polls: u32 = 0,
    poll_cost: u64 = 100, // a query's time

    fn now(s: *Sim) u64 {
        return s.t;
    }

    fn sleep(s: *Sim, ns: u64) void {
        s.sleeps += 1;
        s.t += ns + 20_000;
    }

    fn done(s: *Sim) !bool {
        s.polls += 1;
        s.t += s.poll_cost;
        return s.t >= s.ends;
    }

    fn synchronize(s: *Sim) !void {
        s.sync_calls += 1;
        if (s.t < s.ends) s.t = s.ends;
        s.t += s.wake_lag;
    }
};

const SimEv = struct {
    s: *Sim,
    fn done(e: SimEv) !bool {
        return e.s.done();
    }
    fn synchronize(e: SimEv) !void {
        return e.s.synchronize();
    }
};

const SimClock = struct {
    s: *Sim,
    fn now(c: SimClock) u64 {
        return c.s.now();
    }
    fn sleep(c: SimClock, ns: u64) void {
        c.s.sleep(ns);
    }
};

fn round(h: *Hybrid, s: *Sim, due_in: u64) !u64 {
    s.ends = s.t + due_in;
    const start = s.t;
    try h.wait(SimClock{ .s = s }, SimEv{ .s = s });
    return s.t - start;
}

test "off, it is the blocking wait: one synchronize, no sleep, no poll" {
    var h: Hybrid = .{};
    var s: Sim = .{};
    for (0..10) |_| _ = try round(&h, &s, 30_000_000);
    try std.testing.expectEqual(@as(u32, 10), s.sync_calls);
    try std.testing.expectEqual(@as(u32, 0), s.sleeps);
    try std.testing.expectEqual(@as(u32, 0), s.polls);
}

test "on: after its warm-up it sleeps once, polls the last stretch, and returns within a poll of the result instead of a wake-up after it" {
    var h: Hybrid = .{ .on = true };
    var s: Sim = .{};
    var late_blocking: u64 = 0;
    for (0..Hybrid.warm) |_| late_blocking = (try round(&h, &s, 30_000_000)) - 30_000_000;
    try std.testing.expectEqual(@as(u64, 65_000), late_blocking);
    s.sync_calls = 0;
    s.sleeps = 0;
    var worst: u64 = 0;
    for (0..60) |_| {
        const took = try round(&h, &s, 30_000_000);
        worst = @max(worst, took -| 30_000_000);
    }
    try std.testing.expectEqual(@as(u32, 0), s.sync_calls);
    try std.testing.expectEqual(@as(u32, 60), s.sleeps);
    // never later than a few polls after the result once settled; the blocking wait was 65,000 ns late
    try std.testing.expect(worst < 5_000);
    try std.testing.expect(h.polled == 60);
}

test "a round that runs long falls back to the blocking wait after the poll window: the core is not held for it" {
    var h: Hybrid = .{ .on = true };
    var s: Sim = .{};
    for (0..12) |_| _ = try round(&h, &s, 30_000_000);
    const polls0 = s.polls;
    s.sync_calls = 0;
    _ = try round(&h, &s, 40_000_000); // 10 ms past the learned end
    try std.testing.expectEqual(@as(u32, 1), s.sync_calls);
    try std.testing.expectEqual(@as(u64, 1), h.fell_back);
    // polled for about the window only: 2 ms of 100 ns queries
    try std.testing.expect(s.polls - polls0 <= 2_000_000 / 100 + 2);
}

test "a round that ends sooner than learned is caught on the first poll and the estimate steps down toward it" {
    var h: Hybrid = .{ .on = true };
    var s: Sim = .{};
    for (0..12) |_| _ = try round(&h, &s, 30_000_000);
    const before = h.ema_ns;
    _ = try round(&h, &s, 20_000_000);
    try std.testing.expectEqual(@as(u64, 1), h.early);
    try std.testing.expect(h.ema_ns < before);
    for (0..40) |_| _ = try round(&h, &s, 20_000_000);
    try std.testing.expect(h.ema_ns < 20_500_000 and h.ema_ns > 19_000_000);
}

test "the real clock sleeps at least what it was asked" {
    const c: SystemClock = .{};
    const t0 = c.now();
    c.sleep(2_000_000);
    try std.testing.expect(c.now() - t0 >= 2_000_000);
}
