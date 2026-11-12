//! A CPU-owned idle gate. GPU-written doorbells cannot wake a Linux futex:
//! the control thread parks only after a world-wide quiescence handshake,
//! and resumes the proxies before enqueueing the next distributed primitive.
const std = @import("std");
const linux = std.os.linux;

/// Host tests only: runs at the top of `wait`, before its first load of `parked` (null in production). A park set
/// here lands after a caller's own sample and before the wait's load, the gap a report keyed on the caller's sample
/// would sleep through. Written by the test before its proxy starts.
pub var wait_before: ?*const fn () void = null;

pub const Gate = struct {
    parked: std.atomic.Value(u32) = .init(0),

    pub fn park(g: *Gate) void {
        g.parked.store(1, .release);
    }

    pub fn wake(g: *Gate) void {
        if (g.parked.swap(0, .release) == 0) return;
        const rc = linux.futex_3arg(&g.parked.raw, .{ .cmd = .WAKE, .private = true }, 1);
        if (linux.errno(rc) != .SUCCESS) @panic("RDMA idle futex wake failed");
    }

    /// True when this call actually blocked in the futex (the gate was parked and this thread slept until a wake);
    /// false when it passed straight through. The proxy reports its interval and resets its window only on a true:
    /// a park that lands between the proxy's sample and this load would otherwise be slept through without a report,
    /// and the next park's interval would include the parked time.
    pub fn wait(g: *Gate) error{FutexWaitFailed}!bool {
        var blocked = false;
        if (wait_before) |f| f();
        while (g.parked.load(.acquire) != 0) {
            // Compare-and-sleep is atomic in the kernel, so resume-before-wait
            // returns AGAIN rather than losing a wake-up. No timeout/polling.
            blocked = true;
            const rc = linux.futex_4arg(&g.parked.raw, .{ .cmd = .WAIT, .private = true }, 1, null);
            switch (linux.errno(rc)) {
                .SUCCESS, .INTR, .AGAIN => {},
                else => return error.FutexWaitFailed,
            }
        }
        return blocked;
    }
};

test "a parked proxy blocks until resumed" {
    const Context = struct {
        gate: Gate = .{},
        entered: std.atomic.Value(bool) = .init(false),
        finished: std.atomic.Value(bool) = .init(false),
        fn run(c: *@This()) void {
            c.entered.store(true, .release);
            _ = c.gate.wait() catch @panic("wait failed");
            c.finished.store(true, .release);
        }
    };
    var ctx: Context = .{};
    ctx.gate.park();
    const thread = try std.Thread.spawn(.{}, Context.run, .{&ctx});
    defer {
        ctx.gate.wake();
        thread.join();
    }
    while (!ctx.entered.load(.acquire)) std.Thread.yield() catch {};
    try std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake);
    try std.testing.expect(!ctx.finished.load(.acquire));
    ctx.gate.wake();
}

test "resume before wait is not lost" {
    var gate: Gate = .{};
    for (0..1000) |_| {
        gate.park();
        gate.wake();
        _ = try gate.wait();
    }
}

test "repeated cross-thread park and resume preserve wake-ups" {
    const Context = struct {
        work: Gate = .{},
        done: Gate = .{},
        fn run(c: *@This()) void {
            for (0..1000) |_| {
                _ = c.work.wait() catch @panic("work wait failed");
                c.work.park(); // re-arm before acknowledging this round
                c.done.wake();
            }
        }
    };
    var ctx: Context = .{};
    ctx.work.park();
    const thread = try std.Thread.spawn(.{}, Context.run, .{&ctx});
    defer thread.join();
    for (0..1000) |_| {
        ctx.done.park();
        ctx.work.wake();
        _ = try ctx.done.wait();
    }
}
