//! The decode rounds' instrument (rank 0): one log line every 64 rounds, so any two runs read alike. It uses host clocks and counters the proxies keep; it waits for no device.
//!
//! A round's wall is the lane's verify call, send to sampled tokens. A ring exchange's wait is the proxy's: from the
//! moment it posted this rank's part to the moment the last peer's part was published (zero when the peers were ahead).
//! A chunk's wall is the interval between two chunk issues of one fill, which in a steady pass is the chunk's time on the
//! GPU, rounds between them included.
const std = @import("std");

/// The proxy's clock for an exchange's wait: the virtual counter on aarch64 (a register read, no syscall, and not the
/// nap's counted clock), else the monotonic clock in ns. Its rate is rdma_nap.ticksPerUs.
pub fn ticks() u64 {
    if (@import("builtin").cpu.arch == .aarch64) return asm volatile ("mrs %[v], cntvct_el0"
        : [v] "=r" (-> u64),
    );
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A histogram of values in `width` units in `n` buckets (the last takes the rest), written from one thread and drained
/// from another: every cell is an atomic, a drain takes (and clears) what it reads.
pub fn Hist(comptime width: u64, comptime n: usize) type {
    return struct {
        const Self = @This();
        cells: [n + 1]std.atomic.Value(u64) = @splat(.init(0)),
        top: std.atomic.Value(u64) = .init(0),

        pub fn record(h: *Self, v: u64) void {
            const b: usize = @intCast(@min(v / width, n));
            _ = h.cells[b].fetchAdd(1, .monotonic);
            if (v > h.top.load(.monotonic)) _ = h.top.fetchMax(v, .monotonic);
        }

        /// The cells and the largest value drained into `acc`, which holds the sum of several histograms.
        pub fn drain(h: *Self, acc: *Acc) void {
            for (&h.cells, 0..) |*c, i| acc.cells[i] += c.swap(0, .monotonic);
            acc.top = @max(acc.top, h.top.swap(0, .monotonic));
        }

        pub const Acc = struct {
            cells: [n + 1]u64 = @splat(0),
            top: u64 = 0,

            pub fn count(a: *const Acc) u64 {
                var t: u64 = 0;
                for (a.cells) |c| t += c;
                return t;
            }

            /// The value below which a share `q` (0 .. 1) of the values lie: its bucket's midpoint, the largest value in the
            /// last bucket, 0 when empty.
            pub fn percentile(a: *const Acc, q: f64) u64 {
                const total = a.count();
                if (total == 0) return 0;
                const want: u64 = @max(1, @as(u64, @intFromFloat(@ceil(q * @as(f64, @floatFromInt(total))))));
                var seen: u64 = 0;
                for (a.cells, 0..) |c, i| {
                    seen += c;
                    if (seen >= want) return if (i == n) a.top else @min(i * width + width / 2, a.top);
                }
                return a.top;
            }
        };
    };
}

/// A ring exchange's wait, in microseconds.
pub const WaitHist = Hist(2, 256);
/// A round's wall, in microseconds (0.5 ms buckets to 128 ms).
pub const RoundHist = Hist(500, 256);

/// Rounds a report covers.
pub const span: u32 = 64;

pub const Instr = struct {
    rounds: RoundHist = .{},
    n: u32 = 0,
    chunks: u32 = 0, // chunk intervals measured in the span
    chunk_rows: u64 = 0,
    chunk_us: u64 = 0,
    prev_at: u64 = 0, // the previous chunk's issue (us; 0: none), and its rows
    prev_rows: u64 = 0,

    /// A chunk of `rows` issued at `now_us`: the interval since the previous one counts when it is short enough to be the
    /// same fill's.
    pub fn chunk(i: *Instr, now_us: u64, rows: u64) void {
        if (i.prev_at != 0 and now_us > i.prev_at and now_us - i.prev_at < 10 * std.time.us_per_s) {
            i.chunks += 1;
            i.chunk_rows += i.prev_rows;
            i.chunk_us += now_us - i.prev_at;
        }
        i.prev_at = now_us;
        i.prev_rows = rows;
    }

    /// A round of `wall_us`; true when the span is full and `line` is due.
    pub fn round(i: *Instr, wall_us: u64) bool {
        i.rounds.record(wall_us);
        i.n += 1;
        return i.n >= span;
    }

    /// A round done: the span's line (`line`, the rings' waits drained by `ctx.drain`) when it is the 64th, else null. What a
    /// rank-0 verify call logs.
    pub fn note(i: *Instr, buf: []u8, wall_us: u64, ctx: anytype) ?[]const u8 {
        if (!i.round(wall_us)) return null;
        var waits: WaitHist.Acc = .{};
        ctx.drain(&waits);
        return i.line(buf, &waits);
    }

    /// The span's line, with the rings' waits drained into `waits`; clears the span.
    pub fn line(i: *Instr, buf: []u8, waits: *const WaitHist.Acc) []const u8 {
        var r: RoundHist.Acc = .{};
        i.rounds.drain(&r);
        const ms = struct {
            fn of(us: u64) f64 {
                return @as(f64, @floatFromInt(us)) / 1000.0;
            }
        }.of;
        var tail: [96]u8 = undefined;
        const chunks: []const u8 = if (i.chunks == 0) "no chunk" else std.fmt.bufPrint(&tail, "{d} chunks, {d} rows in {d:.2} s ({d:.0} tok/s)", .{ i.chunks, i.chunk_rows, @as(f64, @floatFromInt(i.chunk_us)) / 1e6, @as(f64, @floatFromInt(i.chunk_rows)) * 1e6 / @as(f64, @floatFromInt(@max(i.chunk_us, 1))) }) catch "";
        const out = std.fmt.bufPrint(buf, "rounds {d}: wall p50 {d:.1} p90 {d:.1} max {d:.1} ms; ring exchange wait p50 {d} max {d} us over {d}; {s}", .{ i.n, ms(r.percentile(0.5)), ms(r.percentile(0.9)), ms(r.top), waits.percentile(0.5), waits.top, waits.count(), chunks }) catch buf[0..0];
        i.n = 0;
        i.chunks = 0;
        i.chunk_rows = 0;
        i.chunk_us = 0;
        return out;
    }
};

test "a histogram's percentiles are its buckets' midpoints, its last bucket answers with the largest value, and a drain clears it" {
    var h: WaitHist = .{};
    for (0..100) |i| h.record(10 + i % 5); // 10 .. 14 us
    h.record(900); // past the last bucket (512)
    var acc: WaitHist.Acc = .{};
    h.drain(&acc);
    try std.testing.expectEqual(@as(u64, 101), acc.count());
    try std.testing.expectEqual(@as(u64, 900), acc.top);
    const p50 = acc.percentile(0.5);
    try std.testing.expect(p50 >= 10 and p50 <= 15);
    try std.testing.expectEqual(@as(u64, 900), acc.percentile(1.0));
    var again: WaitHist.Acc = .{};
    h.drain(&again);
    try std.testing.expectEqual(@as(u64, 0), again.count());
    try std.testing.expectEqual(@as(u64, 0), again.percentile(0.5));
    // two histograms add
    var a: WaitHist = .{};
    var b: WaitHist = .{};
    a.record(4);
    b.record(100);
    var both: WaitHist.Acc = .{};
    a.drain(&both);
    b.drain(&both);
    try std.testing.expectEqual(@as(u64, 2), both.count());
    try std.testing.expectEqual(@as(u64, 100), both.top);
}

test "the line is due every 64 rounds, and the chunks' interval counts only inside one fill" {
    var i: Instr = .{};
    var due = false;
    for (0..64) |k| {
        try std.testing.expect(!due);
        due = i.round(30_000 + (k % 8) * 1000);
    }
    try std.testing.expect(due);
    i.chunk(1_000_000, 4096); // the first of a fill: no interval yet
    i.chunk(1_900_000, 4096); // 0.9 s for the first chunk's rows
    i.chunk(50_000_000, 2048); // 48 s later: another fill, not an interval
    var w: WaitHist.Acc = .{};
    w.cells[7] = 10;
    w.top = 40;
    var buf: [512]u8 = undefined;
    const l = i.line(&buf, &w);
    try std.testing.expect(std.mem.indexOf(u8, l, "rounds 64: wall p50 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, l, "ring exchange wait p50 15 max 40 us over 10") != null);
    try std.testing.expect(std.mem.indexOf(u8, l, "1 chunks, 4096 rows in 0.90 s (4551 tok/s)") != null);
    // cleared: the next span starts empty
    try std.testing.expectEqual(@as(u32, 0), i.n);
    try std.testing.expect(!i.round(1000));
    const l2 = i.line(&buf, &.{});
    try std.testing.expect(std.mem.indexOf(u8, l2, "rounds 1: wall") != null);
    try std.testing.expect(std.mem.indexOf(u8, l2, "no chunk") != null);
}

test "the 64th round emits the line, the 63 before it nothing, and the line starts with the text an operator greps for" {
    const Waits = struct {
        drains: u32 = 0,
        fn drain(w: *@This(), acc: *WaitHist.Acc) void {
            w.drains += 1;
            acc.cells[5] += 3;
            acc.top = 14;
        }
    };
    var w: Waits = .{};
    var i: Instr = .{};
    var buf: [512]u8 = undefined;
    for (0..63) |_| try std.testing.expectEqual(@as(?[]const u8, null), i.note(&buf, 38_000, &w));
    try std.testing.expectEqual(@as(u32, 0), w.drains); // (the rings are read once a span, not a round)
    const l = i.note(&buf, 38_000, &w).?;
    try std.testing.expect(std.mem.startsWith(u8, l, "rounds 64: wall p50 38"));
    try std.testing.expect(std.mem.indexOf(u8, l, "ring exchange wait p50 11 max 14 us over 3") != null);
    try std.testing.expectEqual(@as(u32, 1), w.drains);
    // the next span starts at once: 63 more rounds say nothing, the 64th speaks again
    for (0..63) |_| try std.testing.expectEqual(@as(?[]const u8, null), i.note(&buf, 38_000, &w));
    try std.testing.expect(std.mem.startsWith(u8, i.note(&buf, 38_000, &w).?, "rounds 64: wall"));
}
