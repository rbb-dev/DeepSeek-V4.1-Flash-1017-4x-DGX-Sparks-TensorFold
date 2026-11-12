//! Where this rank's host is, for the failure that names it: a transport's watchdog that fires says only which class and peer it
//! waited for, not what the host was doing meanwhile. A host that waits for something (the prompt passes' issue gate, the
//! transport's scheduling ring) writes its place on entering and puts the previous one back on leaving (three atomics, no lock);
//! whoever reports a failure reads them, from any thread.
const std = @import("std");

pub const Place = enum(u8) {
    main_loop, // outside any wait named here
    issue_gate, // asleep at the issue gate for the GPU to start a layer (a = layers started, b = layers issued)
    roci_reserve, // waiting for the transport's scheduling ring to drain before a send (a = the class)
    round, // rank 0 inside a decode round's verify call (a = windows)
    frame, // a follower running a frame from its main loop (a = frame kind, b = step)
    after_backend, // rank 0 back from a backend call (a round, a drafter pass) inside the lane core: committing tokens, delivering, choosing the next step
    draft, // rank 0 in the drafter pass (a = streams, b = steps)
    capture_vote, // inside a CUDA graph capture's agreement across the ranks (a = graph_fallback.Stage, b = the collectives recorded)
    engram_rows, // issuing a layer, waiting for the fill worker's rows (a = Engram layer slot, b = chunk)
    engram_flush, // waiting for the pinned Engram copies to finish before a worker starts
    engram_release, // waiting for the oldest pinned copy of a layer to finish before its slots go back (a = Engram layer)
    worker_join, // joining a thread the engine waits for: the Engram worker or a read-ahead
    settle_gather, // in the star's all-gather with every rank (the SSD tier's settle)
    idle_ack, // rank 0 waiting for every follower's ready, then committing the world-wide idle boundary
    link_send, // in a blocking send to a rank (a full socket under backpressure holds it)
    link_recv, // in a blocking receive from a rank
};

pub const Mark = struct { place: Place, a: u32, b: u32, since_ns: u64 };

var place_v: std.atomic.Value(u8) = .init(0);
var a_v: std.atomic.Value(u32) = .init(0);
var b_v: std.atomic.Value(u32) = .init(0);
var since_v: std.atomic.Value(u64) = .init(0);

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The place's time stamp: aarch64's virtual counter (one register read, 2-5 ns; clock_gettime is ~25 ns through the vDSO, and a place
/// changes ~14 times a round), else the monotonic clock in ns. Turned into the reader's nanoseconds in `get`.
fn stamp() u64 {
    if (@import("builtin").cpu.arch == .aarch64) return asm volatile ("mrs %[v], cntvct_el0"
        : [v] "=r" (-> u64),
    );
    return nowNs();
}

fn stampHz() u64 {
    if (@import("builtin").cpu.arch == .aarch64) return asm volatile ("mrs %[v], cntfrq_el0"
        : [v] "=r" (-> u64),
    );
    return std.time.ns_per_s;
}

/// The stamp clock against the monotonic clock, read together: a stamp is a fixed number of nanoseconds on the monotonic clock
/// (the same place read twice gives the same time). The monotonic clock is slewed and the counter is not, so a pairing drifts from
/// the day it was made: a reader that finds the pairing more than a second old pairs the clocks again. The pair sits behind a
/// generation (odd while it is rewritten; 0 before the first pairing), so a reader takes the stamp and the nanoseconds of one pairing.
var pair_gen: std.atomic.Value(u32) = .init(0);
var pair_ns: std.atomic.Value(u64) = .init(0);
var pair_stamp: std.atomic.Value(u64) = .init(0);
var pair_busy: std.atomic.Value(bool) = .init(false); // one reader pairs at a time

const Pair = struct { stamp: u64, ns: u64 };

/// The two clocks read as one instant: a stamp on each side of the clock read, kept only when the two stamps lie within 20 us
/// (a thread preempted between the reads would shift every place's time late by the preemption); the clock read is paired
/// with the stamps' midpoint.
fn measure() Pair {
    const tight = @max(1, stampHz() / 50_000); // 20 us in stamp ticks
    var tries: u32 = 0;
    while (true) : (tries += 1) {
        const s0 = stamp();
        const n = nowNs();
        const s1 = stamp();
        if (s1 -% s0 <= tight or tries >= 100) return .{ .stamp = s0 +% (s1 -% s0) / 2, .ns = n };
    }
}

/// Publishes a pairing (the caller holds `pair_busy`).
fn publish(p: Pair) void {
    const g = pair_gen.load(.monotonic);
    _ = pair_gen.swap(g | 1, .acq_rel); // odd: being rewritten
    // release: neither half of the pair becomes visible before the odd generation (a later plain store may, on aarch64), so a
    // reader that sees a new half also sees the generation move and reads again
    pair_stamp.store(p.stamp, .release);
    pair_ns.store(p.ns, .release);
    pair_gen.store((g | 1) +% 1, .release);
}

/// The current pairing, made on first use and made again once it is more than a second old.
fn pairing() Pair {
    while (true) {
        const g1 = pair_gen.load(.acquire);
        if (g1 == 0 or g1 & 1 == 1) {
            if (g1 == 0 and !pair_busy.swap(true, .acq_rel)) {
                publish(measure());
                pair_busy.store(false, .release);
            } else std.atomic.spinLoopHint();
            continue;
        }
        const p: Pair = .{ .stamp = pair_stamp.load(.acquire), .ns = pair_ns.load(.acquire) };
        if (pair_gen.load(.acquire) != g1) continue;
        const age: i64 = @bitCast(stamp() -% p.stamp);
        if (age <= @as(i64, @intCast(stampHz())) or pair_busy.swap(true, .acq_rel)) return p;
        const fresh = measure();
        publish(fresh);
        pair_busy.store(false, .release);
        return fresh;
    }
}

/// A stamp as nanoseconds on the monotonic clock.
fn stampNs(st: u64) u64 {
    if (st == 0) return 0;
    const p = pairing();
    const d: i64 = @bitCast(st -% p.stamp);
    const ns: i128 = @divTrunc(@as(i128, d) * std.time.ns_per_s, @as(i128, stampHz()));
    return @intCast(@max(0, @as(i128, p.ns) + ns));
}

pub fn set(p: Place, a: u32, b: u32) void {
    a_v.store(a, .monotonic);
    b_v.store(b, .monotonic);
    if (place_v.swap(@backingInt(p), .acq_rel) != @backingInt(p)) since_v.store(stamp(), .release);
}

pub fn get() Mark {
    return .{ .place = @fromBackingInt(@intCast(place_v.load(.acquire))), .a = a_v.load(.monotonic), .b = b_v.load(.monotonic), .since_ns = stampNs(since_v.load(.acquire)) };
}

/// The one thread whose waits are the host's place (0: none claimed, any thread writes). A failure's line names where THE ENGINE
/// THREAD was; a wait on another thread (one on the transport's scheduling ring) must not leave
/// its place, or take the engine thread's, in that line.
var owner: std.atomic.Value(u64) = .init(0);

/// The calling thread's waits are the host's place from now on.
pub fn claim() void {
    owner.store(@intCast(std.Thread.getCurrentId()), .release);
}

/// This thread's id, asked of the kernel once (a gettid call on every mark would be a syscall on the hot path).
threadlocal var my_id: u64 = 0;

fn thisThread() u64 {
    if (my_id == 0) my_id = @intCast(std.Thread.getCurrentId());
    return my_id;
}

fn mine() bool {
    const o = owner.load(.acquire);
    return o == 0 or o == thisThread();
}

/// What `enter` replaced: `leave` puts it back, and only if `enter` wrote.
pub const Held = struct { prev: Mark, wrote: bool };

/// The place `set` replaced is put back by `leave`, so a wait inside another place names itself and then hands the place back. Called
/// from a thread other than the owner it changes nothing.
pub fn enter(p: Place, a: u32, b: u32) Held {
    if (!mine()) return .{ .prev = get(), .wrote = false };
    const prev = get();
    set(p, a, b);
    return .{ .prev = prev, .wrote = true };
}

pub fn leave(h: Held) void {
    if (h.wrote) set(h.prev.place, h.prev.a, h.prev.b);
}

/// The mark in words, `now_ns` being the reader's clock.
pub fn describe(buf: []u8, m: Mark, now_ns: u64) []const u8 {
    const ms = if (now_ns > m.since_ns and m.since_ns != 0) (now_ns - m.since_ns) / std.time.ns_per_ms else 0;
    const text = switch (m.place) {
        .main_loop => std.fmt.bufPrint(buf, "outside any named wait, for {d} ms", .{ms}),
        .issue_gate => std.fmt.bufPrint(buf, "asleep at the issue gate for the GPU to start a layer (layers started {d}, issued {d}), for {d} ms", .{ m.a, m.b, ms }),
        .roci_reserve => std.fmt.bufPrint(buf, "waiting for the transport's scheduling ring before a send (class {d}), for {d} ms", .{ m.a, ms }),
        .round => std.fmt.bufPrint(buf, "inside a decode round's verify call ({d} windows), for {d} ms", .{ m.a, ms }),
        .frame => std.fmt.bufPrint(buf, "running a frame from its main loop (kind {d}, step {d}), for {d} ms", .{ m.a, m.b, ms }),
        .after_backend => std.fmt.bufPrint(buf, "in the lane core after a backend call (committing, delivering, choosing the next step), for {d} ms", .{ms}),
        .draft => std.fmt.bufPrint(buf, "in the drafter pass ({d} streams, {d} steps), for {d} ms", .{ m.a, m.b, ms }),
        .capture_vote => std.fmt.bufPrint(buf, "in a graph capture's vote across the ranks (stage {d}, {d} collectives recorded), for {d} ms", .{ m.a, m.b, ms }),
        .engram_rows => std.fmt.bufPrint(buf, "waiting for the fill worker's Engram rows (slot {d}, chunk {d}), for {d} ms", .{ m.a, m.b, ms }),
        .engram_flush => std.fmt.bufPrint(buf, "waiting for the pinned Engram copies before a worker starts, for {d} ms", .{ms}),
        .engram_release => std.fmt.bufPrint(buf, "waiting for a pinned Engram copy to finish before its slots go back (layer {d}), for {d} ms", .{ m.a, ms }),
        .worker_join => std.fmt.bufPrint(buf, "joining the Engram worker or a read-ahead, for {d} ms", .{ms}),
        .settle_gather => std.fmt.bufPrint(buf, "in the star's all-gather with every rank, for {d} ms", .{ms}),
        .idle_ack => std.fmt.bufPrint(buf, "waiting for every follower's idle ack, for {d} ms", .{ms}),
        .link_send => std.fmt.bufPrint(buf, "in a blocking send to a rank, for {d} ms", .{ms}),
        .link_recv => std.fmt.bufPrint(buf, "in a blocking receive from a rank, for {d} ms", .{ms}),
    } catch buf[0..0];
    return text;
}

// -- the frames this rank sent and took last ----------------------------------------------------------------------------

const journal_len = 16;
var jring: [journal_len]std.atomic.Value(u64) = @splat(.init(0));
var jhead: std.atomic.Value(u32) = .init(0);

/// A frame sent (`S`) or taken (`T`) by this rank: kind, step.
pub fn frame(dir: u8, kind: u8, step: u64) void {
    const i = jhead.fetchAdd(1, .monotonic);
    jring[i % journal_len].store((@as(u64, dir) << 56) | (@as(u64, kind) << 48) | (step & 0xffff_ffff_ffff), .release);
}

/// The last frames, oldest first: "S3#41 T3#41 ...".
pub fn journalText(buf: []u8) []const u8 {
    const head = jhead.load(.acquire);
    const n: u32 = @min(head, journal_len);
    var w: std.Io.Writer = .fixed(buf);
    var k: u32 = 0;
    while (k < n) : (k += 1) {
        const v = jring[(head - n + k) % journal_len].load(.acquire);
        if (k > 0) w.writeByte(' ') catch break;
        w.print("{c}{d}#{d}", .{ @as(u8, @intCast(v >> 56)), @as(u8, @intCast((v >> 48) & 0xff)), v & 0xffff_ffff_ffff }) catch break;
    }
    return w.buffered();
}

/// A follower's socket head: the kind of the next frame rank 0 sent that it has not taken (-1: none or no socket), set by the
/// follower's loop.
pub var head_probe: ?*const fn () i32 = null;

/// Everything a failure line says about this rank's host: the place, the last frames, the socket head.
pub fn report(buf: []u8) []const u8 {
    var pb: [220]u8 = undefined;
    var jb: [160]u8 = undefined;
    const head: i32 = if (head_probe) |f| f() else -1;
    return std.fmt.bufPrint(buf, "{s}; last frames {s}; socket head kind {d}", .{ now(&pb), journalText(&jb), head }) catch buf[0..0];
}

/// Joins `th` with the wait named: the thread's end is the engine's, so a hang in it names where the host sat.
pub fn joinThread(th: std.Thread) void {
    const held = enter(.worker_join, 0, 0);
    defer leave(held);
    th.join();
}

/// This rank's place in words now.
pub fn now(buf: []u8) []const u8 {
    return describe(buf, get(), nowNs());
}

test "the place is set from one thread and described from another, with how long the host has been there" {
    set(.issue_gate, 17, 27);
    const m = get();
    try std.testing.expectEqual(Place.issue_gate, m.place);
    var buf: [200]u8 = undefined;
    const l = describe(&buf, m, m.since_ns + 5_100 * std.time.ns_per_ms);
    try std.testing.expectEqualStrings("asleep at the issue gate for the GPU to start a layer (layers started 17, issued 27), for 5100 ms", l);
    // a second set in the same place keeps the time it began; another place restarts it
    const t0 = get().since_ns;
    set(.issue_gate, 18, 28);
    try std.testing.expectEqual(t0, get().since_ns);
    set(.roci_reserve, 3, 0);
    try std.testing.expect(get().since_ns >= t0);
    try std.testing.expectEqualStrings("waiting for the transport's scheduling ring before a send (class 3), for 0 ms", describe(&buf, .{ .place = .roci_reserve, .a = 3, .b = 0, .since_ns = 0 }, 99));
    set(.main_loop, 0, 0);
}

test "enter names a wait and leave hands the place back, nested waits included" {
    set(.main_loop, 0, 0);
    const outer = enter(.issue_gate, 4, 9);
    try std.testing.expectEqual(Place.issue_gate, get().place);
    const inner = enter(.roci_reserve, 2, 0);
    try std.testing.expectEqual(Place.roci_reserve, get().place);
    leave(inner);
    try std.testing.expectEqual(Place.issue_gate, get().place);
    try std.testing.expectEqual(@as(u32, 9), get().b);
    leave(outer);
    try std.testing.expectEqual(Place.main_loop, get().place);
}

test "only the owner thread's waits are the host's place: another thread enters and leaves without a trace" {
    set(.main_loop, 0, 0);
    claim();
    defer owner.store(0, .release);
    const outer = enter(.issue_gate, 4, 9);
    try std.testing.expect(outer.wrote);
    const T = struct {
        fn run() void {
            const h = enter(.roci_reserve, 7, 0); // (the Engram worker's lane in the scheduling ring)
            std.testing.expect(!h.wrote) catch @panic("a thread that is not the owner wrote the place");
            leave(h);
        }
    };
    const th = try std.Thread.spawn(.{}, T.run, .{});
    th.join();
    try std.testing.expectEqual(Place.issue_gate, get().place);
    try std.testing.expectEqual(@as(u32, 9), get().b);
    leave(outer);
    try std.testing.expectEqual(Place.main_loop, get().place);
}

test "the frame journal keeps the last 16 frames oldest first, and the report joins the place, the frames and the socket head" {
    for (0..20) |i| frame(if (i % 2 == 0) 'S' else 'T', @intCast(3 + i % 2), 100 + i);
    var buf: [200]u8 = undefined;
    const j = journalText(&buf);
    try std.testing.expect(std.mem.startsWith(u8, j, "S3#104 T4#105 ")); // frames 4 .. 19 of 20
    try std.testing.expect(std.mem.endsWith(u8, j, "T4#119"));
    try std.testing.expectEqual(@as(usize, 15), std.mem.count(u8, j, " "));
    const H = struct {
        fn five() i32 {
            return 5;
        }
    };
    head_probe = H.five;
    defer head_probe = null;
    set(.draft, 3, 2);
    var rb: [600]u8 = undefined;
    const r = report(&rb);
    try std.testing.expect(std.mem.indexOf(u8, r, "in the drafter pass (3 streams, 2 steps)") != null);
    try std.testing.expect(std.mem.indexOf(u8, r, "last frames S3#104") != null);
    try std.testing.expect(std.mem.endsWith(u8, r, "socket head kind 5"));
    // a wait names itself and gives the place back
    const held = enter(.engram_rows, 1, 7);
    try std.testing.expectEqual(Place.engram_rows, get().place);
    leave(held);
    try std.testing.expectEqual(Place.draft, get().place);
    set(.main_loop, 0, 0);
}

test "every new place has its words" {
    var buf: [200]u8 = undefined;
    for ([_]Place{ .round, .frame, .after_backend, .draft, .capture_vote, .engram_rows, .engram_flush, .engram_release, .worker_join, .settle_gather, .idle_ack, .link_send, .link_recv }) |p| {
        const t = describe(&buf, .{ .place = p, .a = 1, .b = 2, .since_ns = 0 }, 99);
        try std.testing.expect(t.len > 10);
    }
}

test "every thread join, blocking link call and unbounded wait on the engine thread sits inside a named place" {
    const files = [_][]const u8{ @embedFile("model.zig"), @embedFile("prompt.zig"), @embedFile("lanes.zig"), @embedFile("rocinante.zig") };
    // no bare join of a thread the engine waits for: where.joinThread names it
    for (files) |src| {
        const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"") orelse src.len];
        try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, code, ".join()"));
    }
    // each blocking function enters its place before its first wait and leaves it
    const Need = struct { src: []const u8, fn_head: []const u8, enter: []const u8 };
    const needs = [_]Need{
        .{ .src = @embedFile("star.zig"), .fn_head = "pub fn allGather(", .enter = "where.enter(.settle_gather, 0, 0)" },
        .{ .src = @embedFile("lanes.zig"), .fn_head = "pub fn idleRdma(", .enter = "where.enter(.idle_ack, 0, 0)" },
        .{ .src = @embedFile("prompt.zig"), .fn_head = "pub fn engramRelease(", .enter = "where.enter(.engram_release," },
        .{ .src = @embedFile("prompt.zig"), .fn_head = "pub fn engramFlushPins(", .enter = "where.enter(.engram_flush, 0, 0)" },
        .{ .src = @embedFile("prompt.zig"), .fn_head = "pub fn engramWorkerRows(", .enter = "where.enter(.engram_rows," },
        .{ .src = @embedFile("link.zig"), .fn_head = "pub fn send(", .enter = "where.enter(.link_send, 0, 0)" },
        .{ .src = @embedFile("link.zig"), .fn_head = "pub fn recv(", .enter = "where.enter(.link_recv, 0, 0)" },
    };
    for (needs) |n| {
        const at = std.mem.indexOf(u8, n.src, n.fn_head) orelse return error.TestUnexpectedResult;
        const body = n.src[at .. at + 420];
        const e = std.mem.indexOf(u8, body, n.enter) orelse return error.TestUnexpectedResult;
        try std.testing.expect(std.mem.indexOf(u8, body[e..], "defer where.leave(") != null);
    }
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("prompt.zig"), "where.joinThread(th);") != null);
}

test "a pairing made long ago is made again: a place set after the clocks drifted reads on the monotonic clock" {
    _ = stampNs(stamp()); // (a pairing exists)
    // a pairing from two seconds ago whose nanoseconds are 5 ms off, as if the monotonic clock had been slewed since
    const old = stamp() -% 2 * stampHz();
    const g = pair_gen.load(.acquire);
    pair_gen.store(g | 1, .release);
    pair_stamp.store(old, .release);
    pair_ns.store(nowNs() -| 2 * std.time.ns_per_s -| 5 * std.time.ns_per_ms, .release);
    pair_gen.store((g | 1) +% 1, .release);
    set(.settle_gather, 0, 0);
    const before = nowNs();
    set(.round, 9, 0);
    const after = nowNs();
    const m = get();
    try std.testing.expectEqual(Place.round, m.place);
    try std.testing.expect(m.since_ns + 1 * std.time.ns_per_ms >= before);
    try std.testing.expect(m.since_ns <= after + 1 * std.time.ns_per_ms);
}

test "a place's age reads as the time since it was set, on the reader's clock, whatever clock stamped it" {
    const t_before = nowNs();
    set(.round, 3, 0);
    const t_set = nowNs();
    var ts: std.c.timespec = .{ .sec = 0, .nsec = 20 * std.time.ns_per_ms };
    _ = std.c.nanosleep(&ts, null);
    const m = get();
    try std.testing.expectEqual(Place.round, m.place);
    const at = nowNs();
    try std.testing.expect(m.since_ns != 0);
    // stamped inside [t_before, t_set] (the anchor pairs the clocks within 20 us), whatever the scheduler did around the set
    try std.testing.expect(m.since_ns <= t_set + 1 * std.time.ns_per_ms);
    try std.testing.expect(m.since_ns + 1 * std.time.ns_per_ms >= t_before);
    var buf: [128]u8 = undefined;
    const text = describe(&buf, m, at);
    try std.testing.expect(std.mem.indexOf(u8, text, "for 2") != null or std.mem.indexOf(u8, text, "for 1") != null); // 20 ms +- jitter
    // the same place set again keeps its time; another place restarts it
    set(.round, 4, 0);
    try std.testing.expectEqual(m.since_ns, get().since_ns);
    set(.draft, 1, 1);
    try std.testing.expect(get().since_ns > m.since_ns + 10 * std.time.ns_per_ms);
    set(.main_loop, 0, 0);
}
