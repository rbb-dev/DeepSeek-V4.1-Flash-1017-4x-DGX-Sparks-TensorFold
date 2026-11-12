//! A sleeping wait on a counter another thread advances: the waiter reads the bell's sequence, checks its condition, and
//! sleeps in the kernel only while the sequence still holds that value; the writer advances the counter, then rings.
//! Compare-and-sleep is atomic in the kernel, so a ring between the waiter's check and its sleep returns at once (no lost
//! wake-up); a ring with nobody asleep costs two atomics and no system call.
const std = @import("std");
const linux = std.os.linux;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @intCast(@as(i128, ts.sec) * 1_000_000_000 + ts.nsec);
}

pub const Bell = struct {
    seq: std.atomic.Value(u32) = .init(0),
    sleepers: std.atomic.Value(u32) = .init(0),
    // what the waiter did (read by the instrument line): sleeps begun, the ns spent asleep, sleeps ended by a ring
    sleeps: std.atomic.Value(u64) = .init(0),
    asleep_ns: std.atomic.Value(u64) = .init(0),
    rung: std.atomic.Value(u64) = .init(0),

    /// The sequence to read BEFORE the condition is checked, and to hand to `sleep` if it does not hold.
    pub fn seen(b: *const Bell) u32 {
        return b.seq.load(.seq_cst);
    }

    /// After the counter (or flag) the waiter tests has been advanced: wakes every sleeper.
    pub fn ring(b: *Bell) void {
        _ = b.seq.fetchAdd(1, .seq_cst);
        if (b.sleepers.load(.seq_cst) == 0) return;
        _ = linux.futex_3arg(&b.seq.raw, .{ .cmd = .WAKE, .private = true }, std.math.maxInt(i32));
    }

    /// Sleeps while the sequence is still `at` (the value `seen` returned before the failed check), for at most `timeout_ns`
    /// (null: until a ring). Returns when rung, on the timeout, or at once if a ring already happened.
    pub fn sleep(b: *Bell, at: u32, timeout_ns: ?u64) void {
        _ = b.sleepers.fetchAdd(1, .seq_cst);
        const t0 = nowNs();
        var ts: linux.timespec = undefined;
        if (timeout_ns) |n| ts = .{ .sec = @intCast(n / 1_000_000_000), .nsec = @intCast(n % 1_000_000_000) };
        const rc = linux.futex_4arg(&b.seq.raw, .{ .cmd = .WAIT, .private = true }, at, if (timeout_ns != null) &ts else null);
        _ = b.sleepers.fetchSub(1, .seq_cst);
        _ = b.sleeps.fetchAdd(1, .monotonic);
        _ = b.asleep_ns.fetchAdd(nowNs() -% t0, .monotonic);
        if (linux.errno(rc) == .SUCCESS) _ = b.rung.fetchAdd(1, .monotonic);
    }
};

fn threadCpuNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.THREAD_CPUTIME_ID, &ts);
    return @intCast(@as(i128, ts.sec) * 1_000_000_000 + ts.nsec);
}

const Waiter = struct {
    bell: Bell = .{},
    value: std.atomic.Value(usize) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    cpu_ns: std.atomic.Value(u64) = .init(0),
    woke_at_value: std.atomic.Value(usize) = .init(0),

    // waits for value > 0 or stop, the way the engine's loops do
    fn run(w: *Waiter, timeout_ns: ?u64) void {
        const c0 = threadCpuNs();
        while (w.value.load(.acquire) == 0) {
            if (w.stop.load(.acquire)) break;
            const at = w.bell.seen();
            if (w.value.load(.acquire) != 0 or w.stop.load(.acquire)) break;
            w.bell.sleep(at, timeout_ns);
        }
        w.woke_at_value.store(w.value.load(.acquire), .release);
        w.cpu_ns.store(threadCpuNs() -% c0, .release);
    }
};

test "a waiter sleeps without using the CPU and wakes on the increment" {
    var w: Waiter = .{};
    const th = try std.Thread.spawn(.{}, Waiter.run, .{ &w, @as(?u64, null) });
    try std.Io.sleep(std.testing.io, .fromMilliseconds(200), .awake);
    w.value.store(1, .release);
    w.bell.ring();
    th.join();
    try std.testing.expectEqual(@as(usize, 1), w.woke_at_value.load(.acquire));
    // 200 ms of waiting: a loop would have burned ~200 ms of CPU
    try std.testing.expect(w.cpu_ns.load(.acquire) < 20_000_000);
    try std.testing.expect(w.bell.sleeps.load(.monotonic) >= 1);
    try std.testing.expect(w.bell.asleep_ns.load(.monotonic) >= 150_000_000);
    try std.testing.expect(w.bell.rung.load(.monotonic) >= 1);
}

test "a stop ends the wait" {
    var w: Waiter = .{};
    const th = try std.Thread.spawn(.{}, Waiter.run, .{ &w, @as(?u64, null) });
    try std.Io.sleep(std.testing.io, .fromMilliseconds(50), .awake);
    w.stop.store(true, .release);
    w.bell.ring();
    th.join();
    try std.testing.expectEqual(@as(usize, 0), w.woke_at_value.load(.acquire));
}

test "a timed sleep returns by itself and still uses almost no CPU" {
    var w: Waiter = .{};
    const th = try std.Thread.spawn(.{}, Waiter.run, .{ &w, @as(?u64, 100_000) });
    try std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake);
    w.value.store(1, .release); // no ring: the timeout notices
    th.join();
    try std.testing.expectEqual(@as(usize, 1), w.woke_at_value.load(.acquire));
    try std.testing.expect(w.bell.sleeps.load(.monotonic) > 100); // 100 us timeouts over 100 ms
    try std.testing.expect(w.cpu_ns.load(.acquire) < 50_000_000);
}

test "a ring between the check and the sleep is not lost" {
    var b: Bell = .{};
    for (0..1000) |_| {
        const at = b.seen();
        b.ring(); // the writer lands in the gap
        b.sleep(at, null); // returns at once; a lost wake-up would hang the test
    }
    try std.testing.expectEqual(@as(u64, 1000), b.sleeps.load(.monotonic));
}

test "a ring with nobody asleep makes no system call (the sleeper count gates it)" {
    var b: Bell = .{};
    b.ring();
    try std.testing.expectEqual(@as(u32, 1), b.seq.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), b.sleepers.load(.monotonic));
}

test "many rings against one sleeper: it always ends" {
    const Both = struct {
        w: Waiter = .{},
        fn writer(s: *@This()) void {
            for (0..2000) |_| s.w.bell.ring();
            s.w.value.store(1, .release);
            s.w.bell.ring();
        }
    };
    for (0..50) |_| {
        var s: Both = .{};
        const a = try std.Thread.spawn(.{}, Waiter.run, .{ &s.w, @as(?u64, null) });
        const b = try std.Thread.spawn(.{}, Both.writer, .{&s});
        a.join();
        b.join();
        try std.testing.expectEqual(@as(usize, 1), s.w.woke_at_value.load(.acquire));
    }
}
