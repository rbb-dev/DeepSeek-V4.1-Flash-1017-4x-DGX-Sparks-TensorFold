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
/// This thread's id, asked of the kernel once (a gettid call on every mark would be a syscall on the hot path).
threadlocal var my_id: u64 = 0;

fn thisThread() u64 {
    if (my_id == 0) my_id = @intCast(std.Thread.getCurrentId());
    return my_id;
}

    return o == 0 or o == thisThread();
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