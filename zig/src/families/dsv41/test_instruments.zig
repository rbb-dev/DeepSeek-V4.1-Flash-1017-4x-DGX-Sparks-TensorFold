//! Test instruments: measurements that sit in a hot path and so exist only in a binary built for them
//! (`zig build -Dtest_instruments=true`). A default build has no instrument: the types below are empty, their calls
//! are empty inline functions, and the proxy's loop compiles to what it was without them. An instrument build refuses
//! to open the transport unless TF_DS_TEST_INSTRUMENTS=1 is set (a test arm says so), and says on every start that it
//! is one, so a release cannot carry it unnoticed.
const std = @import("std");
const nap = @import("rdma_nap.zig");

/// Whether this binary was built with -Dtest_instruments=true (build.zig's options module "instruments").
pub const enabled: bool = @import("instruments").test_instruments;

pub const env_name = "TF_DS_TEST_INSTRUMENTS";

/// The words every message of an instrument build carries, and no byte of a default build does: the fork's image build
/// searches the engine binaries for them (zig/tests/test_instruments_release_paths.sh --binaries).
pub const marker = "TEST INSTRUMENTS BUILD";

/// What an instrument build says when it is refused.
pub const refusal = marker ++ ": this engine was built with test instruments (-Dtest_instruments): it must not serve; set TF_DS_TEST_INSTRUMENTS=1 only on a test arm";

/// The start line of an instrument build that was let through.
pub const warning = marker ++ ": Rocinante nap timer on";

pub const Gate = enum { absent, refused, allowed };

/// Whether a binary may open the transport: a default build always (there is nothing to refuse); an instrument build
/// only when the variable is exactly "1".
pub fn gateFor(built: bool, env: ?[]const u8) Gate {
    if (!built) return .absent;
    const v = env orelse return .refused;
    return if (std.mem.eql(u8, v, "1")) .allowed else .refused;
}

pub fn gate() Gate {
    const v = std.c.getenv(env_name);
    return gateFor(enabled, if (v) |p| std.mem.span(p) else null);
}

/// A nap's ends, as the proxy classifies them.
pub const Wake = enum { bell, deadline, early };

pub const buckets = 5;
/// The upper edges of the first four length buckets, in microseconds; the fifth is everything at or above the last.
pub const edges_us = [buckets - 1]u64{ 1, 10, 50, 100 };

/// The length bucket of a nap of `ticks`, at `per_us` ticks a microsecond.
pub fn bucketOf(ticks: u64, per_us: u64) usize {
    inline for (edges_us, 0..) |e, i| if (ticks < e * per_us) return i;
    return buckets - 1;
}

/// What a timer saw since its last summary. `share_permille` is the part of the span's wall time spent inside the nap
/// calls (a thousandth each); `mean_ns` the mean nap length; `hist` every nap by length, `early_hist` the naps that
/// ended with the doorbell and the deadline both untouched (an event, an interrupt, a set event register) by length.
pub const Summary = struct {
    span_us: u64 = 0,
    share_permille: u64 = 0,
    naps: u64 = 0,
    mean_ns: u64 = 0,
    skipped: u64 = 0,
    hist: [buckets]u64 = @splat(0),
    early_hist: [buckets]u64 = @splat(0),
};

/// The timer around one proxy's nap calls: the proxy thread alone writes it and alone reads it (it prints its own line).
/// Its stamps are reads of the virtual counter (no system call: one would set the event register and end the nap it
/// measures).
pub const NapTimer = struct {
    const Self = @This();
    per_us: u64 = 1,
    /// The stamp the current summary's span starts at.
    from: u64 = 0,
    naps: u64 = 0,
    skipped: u64 = 0,
    ticks: u64 = 0,
    hist: [buckets]u64 = @splat(0),
    early_hist: [buckets]u64 = @splat(0),
    last: struct { naps: u64 = 0, skipped: u64 = 0, ticks: u64 = 0, hist: [buckets]u64 = @splat(0), early_hist: [buckets]u64 = @splat(0) } = .{},

    /// The tick rate and the stamp the first span starts at.
    pub fn arm(self: *Self, per_us: u64, now: u64) void {
        self.per_us = @max(1, per_us);
        self.from = now;
    }

    /// A nap between stamps `t_in` and `t_out`: `ran` false when the word had moved before the instruction (no nap).
    pub fn record(self: *Self, t_in: u64, t_out: u64, ran: bool, how: Wake) void {
        const len = t_out -% t_in;
        const b = bucketOf(len, self.per_us);
        self.naps += 1;
        if (!ran) self.skipped += 1;
        self.ticks += len;
        self.hist[b] += 1;
        if (how == .early) self.early_hist[b] += 1;
    }

    /// Everything since the last summary, over the span to `now`; the next one starts there.
    pub fn summarize(self: *Self, now: u64) Summary {
        var s: Summary = .{};
        const span = now -% self.from;
        const ticks = self.ticks - self.last.ticks;
        s.naps = self.naps - self.last.naps;
        s.skipped = self.skipped - self.last.skipped;
        s.span_us = span / self.per_us;
        if (span != 0) s.share_permille = @intCast(@as(u128, ticks) * 1000 / span);
        if (s.naps != 0) s.mean_ns = @intCast(@as(u128, ticks) * 1000 / self.per_us / s.naps);
        for (0..buckets) |i| {
            s.hist[i] = self.hist[i] - self.last.hist[i];
            s.early_hist[i] = self.early_hist[i] - self.last.early_hist[i];
        }
        self.last = .{ .naps = self.naps, .skipped = self.skipped, .ticks = self.ticks, .hist = self.hist, .early_hist = self.early_hist };
        self.from = now;
        return s;
    }

    /// The fields a rail's line gains (each begins with a space), for the span to `now`.
    pub fn line(self: *Self, w: *std.Io.Writer, now: u64) !void {
        const s = self.summarize(now);
        try w.print(" nap_span_ms {d}.{d:0>3} nap_share {d}.{d}% naps {d} nap_mean_us {d}.{d:0>3} nap_hist_us(<1/1-10/10-50/50-100/>=100) {d}/{d}/{d}/{d}/{d} early_hist_us {d}/{d}/{d}/{d}/{d} early_lt1us {d} nap_skipped {d}", .{
            s.span_us / 1000, s.span_us % 1000, s.share_permille / 10, s.share_permille % 10, s.naps,
            s.mean_ns / 1000, s.mean_ns % 1000, s.hist[0],             s.hist[1],             s.hist[2],
            s.hist[3],        s.hist[4],        s.early_hist[0],       s.early_hist[1],       s.early_hist[2],
            s.early_hist[3],  s.early_hist[4],  s.early_hist[0],       s.skipped,
        });
    }
};

// -- the proxy's calls: each is an empty inline function in a default build (no stamp, no store, no branch) -------------------

/// The stamp a nap begins at: a counter read in an instrument build, nothing in a default one.
pub const NapStart = if (enabled) u64 else void;

/// What the calling proxy thread keeps: its timer and the nap that has ended and awaits its cause. Referenced only
/// under `enabled`, so a default build has no thread-local at all.
const Tls = struct {
    threadlocal var timer: NapTimer = .{};
    threadlocal var owner: usize = 0;
    threadlocal var rail: u8 = 0;
    threadlocal var t_in: u64 = 0;
    threadlocal var t_out: u64 = 0;
    threadlocal var ran: bool = true;
};

/// A proxy thread starts timing its naps (`owner` is its transport, `rail` its rail, `per_us` the counter's ticks a us).
pub inline fn attach(owner: *const anyopaque, rail: u8, per_us: u64) void {
    if (comptime enabled) {
        Tls.timer = .{};
        Tls.timer.arm(per_us, nap.now());
        Tls.owner = @intFromPtr(owner);
        Tls.rail = rail;
    }
}

pub inline fn napStart() NapStart {
    if (comptime enabled) return nap.now();
}

/// The nap that began at `start` has returned (`ran`: the instruction was reached); its cause follows in woke().
pub inline fn napEnded(start: NapStart, ran: bool) void {
    if (comptime enabled) {
        Tls.t_out = nap.now();
        Tls.t_in = start;
        Tls.ran = ran;
    }
}

/// The nap that has ended ended for `how`.
pub inline fn woke(how: Wake) void {
    if (comptime enabled) Tls.timer.record(Tls.t_in, Tls.t_out, Tls.ran, how);
}

/// The thread's timer fields for its rail line.
pub inline fn lineFor(w: *std.Io.Writer) !void {
    if (comptime enabled) try Tls.timer.line(w, nap.now());
}

/// The timers of proxies that have ended, for the host tests (an instrument build only): a proxy's last timer, filed at its
/// end; the last 64 are kept, the newest found first.
const filed_max = 64;
const Filed = struct { owner: usize, rail: u8, timer: NapTimer };
var filed: [filed_max]Filed = undefined;
var filed_n: usize = 0;
var filed_lock: std.atomic.Value(bool) = .init(false);

fn lockFiled() void {
    while (filed_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}

pub inline fn detach() void {
    if (comptime enabled) {
        lockFiled();
        defer filed_lock.store(false, .release);
        filed[filed_n % filed_max] = .{ .owner = Tls.owner, .rail = Tls.rail, .timer = Tls.timer };
        filed_n += 1;
    }
}

/// The last timer a proxy of `owner` on `rail` filed, if it has ended.
pub fn filedFor(owner: *const anyopaque, rail: u8) ?NapTimer {
    lockFiled();
    defer filed_lock.store(false, .release);
    var back: usize = 0;
    while (back < @min(filed_n, filed_max)) : (back += 1) {
        const f = &filed[(filed_n - 1 - back) % filed_max];
        if (f.owner == @intFromPtr(owner) and f.rail == rail) return f.timer;
    }
    return null;
}

test "a default build has no instrument: the proxy's calls take and return nothing (comptime)" {
    if (enabled) return error.SkipZigTest;
    comptime {
        std.debug.assert(NapStart == void);
        std.debug.assert(@TypeOf(napStart()) == void);
    }
}

test "an instrument build's calls time a nap between its stamp, its end and its cause, and file the timer at the thread's end" {
    if (!enabled) return error.SkipZigTest;
    var owner: u8 = 0;
    const per = nap.ticksPerUs();
    attach(&owner, 3, per);
    const start = napStart();
    var word: u64 = 0;
    const ran = nap.napOnce(&word, 0, nap.now() + per * 150);
    napEnded(start, ran);
    woke(.deadline);
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try lineFor(&w);
    try std.testing.expect(std.mem.indexOf(u8, w.buffered(), " naps 1 ") != null);
    detach();
    const f = filedFor(&owner, 3).?;
    try std.testing.expectEqual(@as(u64, 1), f.naps);
    try std.testing.expectEqual(@as(?NapTimer, null), filedFor(&owner, 2));
}

test "filed timers: the newest 64 are kept and found newest first (the host tests open hundreds of proxies)" {
    if (!enabled) return error.SkipZigTest;
    var owners: [filed_max + 6]u8 = undefined;
    for (&owners, 0..) |*o, i| {
        attach(o, 1, 1000);
        const start = napStart();
        napEnded(start, true);
        woke(.bell);
        if (i % 2 == 0) woke(.bell); // the even ones saw two naps
        detach();
    }
    try std.testing.expectEqual(@as(?NapTimer, null), filedFor(&owners[0], 1)); // pushed out by the newer ones
    try std.testing.expectEqual(@as(u64, 1), filedFor(&owners[owners.len - 1], 1).?.naps);
    try std.testing.expectEqual(@as(u64, 2), filedFor(&owners[owners.len - 2], 1).?.naps);
}

test "the gate: a default build has none; an instrument build serves only with TF_DS_TEST_INSTRUMENTS=1" {
    try std.testing.expectEqual(Gate.absent, gateFor(false, null));
    try std.testing.expectEqual(Gate.absent, gateFor(false, "1"));
    try std.testing.expectEqual(Gate.refused, gateFor(true, null));
    try std.testing.expectEqual(Gate.refused, gateFor(true, ""));
    try std.testing.expectEqual(Gate.refused, gateFor(true, "0"));
    try std.testing.expectEqual(Gate.refused, gateFor(true, "true"));
    try std.testing.expectEqual(Gate.refused, gateFor(true, "11"));
    try std.testing.expectEqual(Gate.allowed, gateFor(true, "1"));
}

test "the refusal names the build, the flag and the variable; the warning is the start line's words" {
    try std.testing.expect(std.mem.indexOf(u8, refusal, "-Dtest_instruments") != null);
    try std.testing.expect(std.mem.indexOf(u8, refusal, "must not serve") != null);
    try std.testing.expect(std.mem.indexOf(u8, refusal, "TF_DS_TEST_INSTRUMENTS=1") != null);
    try std.testing.expectEqualStrings("TEST INSTRUMENTS BUILD: Rocinante nap timer on", warning);
    try std.testing.expect(std.mem.startsWith(u8, refusal, "TEST INSTRUMENTS BUILD"));
    try std.testing.expectEqualStrings("TEST INSTRUMENTS BUILD", marker);
}

test "the length buckets: under 1 us, 1-10, 10-50, 50-100, 100 and over, at any tick rate" {
    for ([_]u64{ 1, 24, 1000 }) |per| {
        try std.testing.expectEqual(@as(usize, 0), bucketOf(0, per));
        try std.testing.expectEqual(@as(usize, 0), bucketOf(per - 1, per));
        try std.testing.expectEqual(@as(usize, 1), bucketOf(per, per));
        try std.testing.expectEqual(@as(usize, 1), bucketOf(10 * per - 1, per));
        try std.testing.expectEqual(@as(usize, 2), bucketOf(10 * per, per));
        try std.testing.expectEqual(@as(usize, 2), bucketOf(50 * per - 1, per));
        try std.testing.expectEqual(@as(usize, 3), bucketOf(50 * per, per));
        try std.testing.expectEqual(@as(usize, 3), bucketOf(100 * per - 1, per));
        try std.testing.expectEqual(@as(usize, 4), bucketOf(100 * per, per));
        try std.testing.expectEqual(@as(usize, 4), bucketOf(std.math.maxInt(u64), per));
    }
}

test "the timer counts naps, skipped naps, ticks, the length histogram and the early wakes by length (fake clock)" {
    var t: NapTimer = .{};
    t.arm(1000, 0); // 1000 ticks a microsecond, the first span starts at tick 0
    //            in   out      ran    how
    t.record(0, 500, true, .early); // 0.5 us: the instruction returned at once
    t.record(1000, 1000, false, .bell); // the word had moved: no nap, 0 us
    t.record(2_000, 7_000, true, .bell); // 5 us
    t.record(10_000, 40_000, true, .early); // 30 us
    t.record(50_000, 125_000, true, .deadline); // 75 us
    t.record(200_000, 400_000, true, .deadline); // 200 us
    t.record(500_000, 900_000, true, .early); // 400 us
    const s = t.summarize(1_000_000);
    try std.testing.expectEqual(@as(u64, 7), s.naps);
    try std.testing.expectEqual(@as(u64, 1), s.skipped);
    try std.testing.expectEqual([buckets]u64{ 2, 1, 1, 1, 2 }, s.hist);
    try std.testing.expectEqual([buckets]u64{ 1, 0, 1, 0, 1 }, s.early_hist);
    // 500 + 0 + 5,000 + 30,000 + 75,000 + 200,000 + 400,000 ticks of 1,000 a us = 710.5 us in 1,000 us
    try std.testing.expectEqual(@as(u64, 1000), s.span_us);
    try std.testing.expectEqual(@as(u64, 710), s.share_permille);
    try std.testing.expectEqual(@as(u64, 101_500), s.mean_ns); // 710,500 ns over 7 naps
}

test "a summary covers the span since the last one: counters and the span restart" {
    var t: NapTimer = .{};
    t.arm(1000, 0);
    t.record(0, 4_000, true, .bell);
    const a = t.summarize(10_000);
    try std.testing.expectEqual(@as(u64, 1), a.naps);
    try std.testing.expectEqual(@as(u64, 10), a.span_us);
    try std.testing.expectEqual(@as(u64, 400), a.share_permille);
    t.record(10_000, 12_000, true, .early);
    t.record(12_000, 13_000, true, .early);
    const b = t.summarize(30_000);
    try std.testing.expectEqual(@as(u64, 2), b.naps);
    try std.testing.expectEqual(@as(u64, 20), b.span_us);
    try std.testing.expectEqual(@as(u64, 150), b.share_permille); // 3,000 ticks of 20,000
    try std.testing.expectEqual([buckets]u64{ 0, 2, 0, 0, 0 }, b.early_hist);
    const c = t.summarize(40_000); // nothing happened
    try std.testing.expectEqual(@as(u64, 0), c.naps);
    try std.testing.expectEqual(@as(u64, 0), c.mean_ns);
    try std.testing.expectEqual(@as(u64, 0), c.share_permille);
}

test "a span of zero ticks divides nothing" {
    var t: NapTimer = .{};
    t.arm(1000, 0);
    t.record(0, 10, true, .bell);
    const s = t.summarize(0);
    try std.testing.expectEqual(@as(u64, 0), s.span_us);
    try std.testing.expectEqual(@as(u64, 0), s.share_permille);
}

test "the line names the span, the share, the naps, the mean, both histograms and the skipped naps" {
    var t: NapTimer = .{};
    t.arm(1000, 0);
    t.record(0, 500, true, .early);
    t.record(1000, 8_000, true, .bell);
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try t.line(&w, 10_000);
    try std.testing.expectEqualStrings(" nap_span_ms 0.010 nap_share 75.0% naps 2 nap_mean_us 3.750 nap_hist_us(<1/1-10/10-50/50-100/>=100) 1/1/0/0/0 early_hist_us 1/0/0/0/0 early_lt1us 1 nap_skipped 0", w.buffered());
}

test "the instrument times the real stand-in nap: a nap on a word nothing writes lasts its deadline" {
    var word: u64 = 0;
    const per = nap.ticksPerUs();
    var t: NapTimer = .{};
    t.arm(per, nap.now());
    const a = nap.now();
    const dl = a + per * 300; // the deadline from the stamp: the nap ends at it, so it lasts at least this
    // The stand-in's nap ends at the deadline. A real WFET nap (aarch64) can end early at the event stream (~10 kHz, napOnce's
    // doc), so nap again until the deadline, as the proxy does: the one interval the instrument times is then the whole wait.
    var ran = false;
    while (nap.now() < dl) ran = nap.napOnce(&word, 0, dl);
    const b = nap.now();
    t.record(a, b, ran, .deadline);
    const s = t.summarize(nap.now());
    try std.testing.expect(ran);
    try std.testing.expectEqual(@as(u64, 1), s.naps);
    try std.testing.expectEqual(@as(u64, 0), s.skipped);
    try std.testing.expectEqual(@as(u64, 1), s.hist[4]); // at least 100 us
    try std.testing.expect(s.mean_ns >= 300_000);
}

test "the instrument times a nap that returns at once: a word that already moved is a skipped nap under 1 us" {
    var word: u64 = 5;
    const per = nap.ticksPerUs();
    var t: NapTimer = .{};
    t.arm(per, nap.now());
    const a = nap.now();
    const ran = nap.napOnce(&word, 0, nap.now() + per * 1000);
    const b = nap.now();
    try std.testing.expect(!ran);
    t.record(a, b, ran, .bell);
    const s = t.summarize(nap.now());
    try std.testing.expectEqual(@as(u64, 1), s.skipped);
    try std.testing.expectEqual(@as(u64, 1), s.hist[0]);
}

test "openLane refuses an instrument build without the variable before it creates anything, and an instrument build that serves says so" {
    const src = @embedFile("roci_engine.zig");
    const at = std.mem.indexOf(u8, src, "pub fn openLane(").?;
    const body = src[at..];
    const gate_at = std.mem.indexOf(u8, body, "ti.gate() == .refused").?;
    const refuse_at = std.mem.indexOf(u8, body, "return error.TestInstrumentsRefused").?;
    const create_at = std.mem.indexOf(u8, body, "gpa.create(Roci)").?;
    try std.testing.expect(gate_at < refuse_at and refuse_at < create_at);
    // the gate and the warning are compiled only in an instrument build
    try std.testing.expect(std.mem.indexOf(u8, body[0..create_at], "if (comptime ti.enabled) if (ti.gate()") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "if (comptime ti.enabled) std.log.warn(\"{s}\", .{ti.warning})") != null);
}

test "the proxy's loop calls the timer only through the comptime-empty functions, one per step of a nap" {
    const src = @embedFile("rocinante.zig");
    const at = std.mem.indexOf(u8, src, "fn proxyMain(").?;
    const body = src[at..][0..std.mem.indexOf(u8, src[at..], "\n}\n").?];
    // the timer is no field of Transport and no local of the proxy (a field would move the transport's layout in a default
    // build, and its loop's code with it): the calls are attach, napStart, napEnded, woke (three causes) and detach
    try std.testing.expect(std.mem.indexOf(u8, src, "nap_timer") == null);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "ti.attach("));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "ti.napStart()"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "ti.napEnded("));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, body, "ti.woke("));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "defer ti.detach()"));
}
