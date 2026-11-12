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

pub fn set(p: Place, a: u32, b: u32) void {
    a_v.store(a, .monotonic);
    b_v.store(b, .monotonic);
}

pub fn get() Mark {
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