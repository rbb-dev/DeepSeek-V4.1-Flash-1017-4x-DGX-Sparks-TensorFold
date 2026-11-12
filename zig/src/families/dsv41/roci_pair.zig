//! TF_DS_ROCI_STAGGER: the issue order and the cache gates of two consecutive whole passes in flight, one on each of a rank's two
//! Rocinante lanes (TF_DS_ROCI_STAGGER_LANES=3: three consecutive passes on three lanes, each gated on the one before it).
//!
//! A pass writes, at each layer, the compressed caches and the window ring that the next pass reads at the same layer; the boundary
//! layer (index `half`) writes its cache rows the same way. Nothing else crosses passes: the rest of a pass lives in its own
//! chunk and its lane's arena. So pass B's layer L may run once pass A's layer L has written its caches, and the previous pair's
//! B must have written its layer L before A's layer L reads it. The gate is one event a pass parity and layer, recorded on the
//! pass's stream after the writes and waited on the other pass's stream before the reads.
//!
//! The host issues A's layer L and then B's layer L - gap; every rank issues the same list, a pure function of (half, gap, first).
const std = @import("std");
const cuda = @import("cuda");

/// How many layers ahead pass A is issued when pass B's layer is enqueued.
pub const gap: usize = 2;

/// TF_DS_ROCI_STAGGER_GAP's text: 1, 2 or 3 (the issue order's lag), else the default; the bytes do not depend on it.
pub fn gapOf(text: ?[]const u8) usize {
    const t = text orelse return gap;
    const v = std.fmt.parseInt(usize, std.mem.trim(u8, t, " \t"), 10) catch return gap;
    return if (v >= 1 and v <= 3) v else gap;
}

/// TF_DS_ROCI_STAGGER_PRIO's text: 0 equal priorities (the default), 1 pass A's stream above B's, 2 B's above A's (the control); anything
/// else is 0. Which stream is urgent changes when a stream's ready CTAs are scheduled, not what any kernel computes.
pub const Prio = enum(u8) { equal = 0, a_first = 1, b_first = 2 };
pub fn prioOf(text: ?[]const u8) Prio {
    const t = std.mem.trim(u8, text orelse return .equal, " \t");
    if (std.mem.eql(u8, t, "1")) return .a_first;
    if (std.mem.eql(u8, t, "2")) return .b_first;
    return .equal;
}

/// The priorities of the two streams for a mode, from the context's range: (A, B). Equal: both the least (what a plain stream has).
pub fn prioritiesOf(p: Prio, least: c_int, greatest: c_int) [2]c_int {
    return switch (p) {
        .equal => .{ least, least },
        .a_first => .{ greatest, least },
        .b_first => .{ least, greatest },
    };
}

pub fn pairMajorOf(text: ?[]const u8) bool {
}

/// The most encoder layers a pair has (the engine's layers / 2) and so the most events a parity holds besides the boundary's.
pub const max_half: usize = 32;

/// The most passes in flight (TF_DS_ROCI_STAGGER_LANES).
pub const max_lanes: usize = 3;

/// Which pass of a pair (or of a group of three: `c`).
pub const PairPass = enum(u2) { a = 0, b = 1, c = 2 };

/// TF_DS_ROCI_STAGGER_LANES's text: unset is 2 (the pair as it always was), "2" and "3" are themselves, anything else is null (the start
/// refuses it by name: a value that is not a lane count is a mistake, not a default).
pub fn lanesOf(text: ?[]const u8) ?usize {
    const t = std.mem.trim(u8, text orelse return 2, " \t");
    if (std.mem.eql(u8, t, "2")) return 2;
    if (std.mem.eql(u8, t, "3")) return 3;
    return null;
}

/// The pass whose cache events pass `k` of a group waits on: the one before it, and for the first the last pass of the group queued before
/// it (`prev_lanes` passes: a group of two after a group of three still waits on pass index 2's events).
pub fn predecessor(k: usize, prev_lanes: usize) usize {
    return if (k == 0) prev_lanes - 1 else k - 1;
}

/// Engram worker holding their resume units (a ring of twice a pass's units), and nothing else in the engine at work (`pass_rows_now`
/// is the pass rows the scheduler's rule gives: the resume rows while a stream decodes, fills or waits, which at pass rows of 2,048
/// is the booted rows too, so the lanes' rule asks the host's facts as well: stream.wholePass).
pub fn eligible(on: bool, pass_rows: usize, pass_rows_now: usize, start_a: usize, n_a: usize, start_b: usize, n_b: usize, replay: usize, worker: bool) bool {
    return on and pass_rows_now == pass_rows and
        n_a == pass_rows and n_b == pass_rows and start_b == start_a + pass_rows and
        replay >= start_b + n_b and worker;
}

/// `eligible` for a group of `lanes` passes: `ns` the rows of each, the passes back to back from `start_a`, all whole passes of the booted
/// rows, the last with no decoder rows behind it (`replay` at its end or later). Two lanes are `eligible`'s rule exactly.
pub fn eligibleGroup(on: bool, lanes: usize, pass_rows: usize, pass_rows_now: usize, start_a: usize, ns: []const usize, replay: usize, worker: bool) bool {
    if (!on or lanes < 2 or lanes > max_lanes or ns.len != lanes or pass_rows_now != pass_rows or !worker) return false;
    for (ns) |n| if (n != pass_rows) return false;
    return replay >= start_a + lanes * pass_rows;
}

/// The resume units (`unit` rows) in a pair of passes of `pass_rows`: the chunks one pair-major group of the Engram worker's order
/// takes (prompt.WorkerOrder.group): both passes' layer 0, then both passes' layer 1.
pub fn groupChunks(pass_rows: usize, unit: usize) usize {
    return groupChunksN(2, pass_rows, unit);
}

/// The same for a group of `lanes` passes: every pass's layer 0, then every pass's layer 1.
pub fn groupChunksN(lanes: usize, pass_rows: usize, unit: usize) usize {
    return lanes * (pass_rows / unit);
}

/// The per-layer cache events: `[pass index][layer 0 .. half]`, the last the boundary layer's. One set a pass of the group, so a
/// recording never races a wait enqueued for the pass below (a wait binds to the record before it at its call).
pub const Events = struct {
    ev: [max_lanes][max_half + 1]cuda.Event,
    half: usize,
    lanes: usize,

    pub fn init(d: *const cuda.Driver, half: usize, lanes: usize) !Events {
        if (half == 0 or half > max_half) return error.BadHalf;
        if (lanes < 2 or lanes > max_lanes) return error.BadLanes;
        var e: Events = undefined;
        e.half = half;
        e.lanes = lanes;
        var made: usize = 0;
        errdefer for (0..made) |i| e.ev[i / (half + 1)][i % (half + 1)].deinit();
        for (0..lanes) |p| for (0..half + 1) |l| {
            e.ev[p][l] = try cuda.Event.init(d, false);
            made += 1;
        };
        return e;
    }

    pub fn deinit(e: *Events) void {
        for (0..e.lanes) |p| for (0..e.half + 1) |l| e.ev[p][l].deinit();
    }

    pub fn at(e: *Events, pass: usize, li: usize) *cuda.Event {
        return &e.ev[pass][li];
    }
};

/// One step of a pair's issue order. `wait` is the stream wait on the predecessor pass's cache event for `li` (`li == half` for the
/// boundary layer's); `layer` is the whole serial layer (whose attention records the event); `boundary` the pass's boundary layer.
pub const Step = union(enum) {
    wait: struct { pass: PairPass, li: usize },
    layer: struct { pass: PairPass, li: usize },
    boundary: PairPass,
};

/// The steps a pair uses at most.
pub fn maxSteps(half: usize) usize {
    return max_lanes * (2 * half + 4);
}

/// One pair's issue order for `half` encoder layers: A's layer L, then B's layer L - gap_layers, then B's remaining layers, then
/// A's boundary and B's, B's after a wait on A's boundary event. A's waits on the previous pair's B (`first` false) come before each of
/// its layers and its boundary. (The two-lane case of `groupSchedule`.)
pub fn pairSchedule(half: usize, gap_layers: usize, first: bool, out: []Step) []Step {
    return groupSchedule(2, half, gap_layers, first, out);
}

/// A group's issue order, `lanes` passes of `half` encoder layers: at step t every pass k with layer t - k * gap_layers in range issues
/// it, pass A first (so A's layer L, then B's layer L - gap, then C's layer L - 2 gap), then the boundaries in pass order. Each pass
/// waits its predecessor's cache event (`predecessor`: A's is the previous group's last pass, none when `first`) before each layer and
/// before its boundary. Two lanes give the pair's order exactly.
pub fn groupSchedule(lanes: usize, half: usize, gap_layers: usize, first: bool, out: []Step) []Step {
    var n: usize = 0;
    const push = struct {
        fn add(steps: []Step, i: *usize, st: Step) void {
            steps[i.*] = st;
            i.* += 1;
        }
    };
    var t: usize = 0;
    while (t < half + (lanes - 1) * gap_layers) : (t += 1) {
        for (0..lanes) |k| {
            if (t < k * gap_layers) continue;
            const li = t - k * gap_layers;
            if (li >= half) continue;
            const pass: PairPass = @fromBackingInt(@intCast(k));
            if (k != 0 or !first) push.add(out, &n, .{ .wait = .{ .pass = pass, .li = li } });
            push.add(out, &n, .{ .layer = .{ .pass = pass, .li = li } });
        }
    }
    for (0..lanes) |k| {
        const pass: PairPass = @fromBackingInt(@intCast(k));
        if (k != 0 or !first) push.add(out, &n, .{ .wait = .{ .pass = pass, .li = half } });
        push.add(out, &n, .{ .boundary = pass });
    }
    return out[0..n];
}

fn sameStep(a: Step, b: Step) bool {
    return switch (a) {
        .wait => |x| switch (b) {
            .wait => |y| x.pass == y.pass and x.li == y.li,
            else => false,
        },
        .layer => |x| switch (b) {
            .layer => |y| x.pass == y.pass and x.li == y.li,
            else => false,
        },
        .boundary => |x| switch (b) {
            .boundary => |y| x == y,
            else => false,
        },
    };
}

/// What `check` proves about a pair's steps.
pub const Verdict = struct { ok: bool = true, reason: []const u8 = "", at: usize = 0 };

/// Prove: each pass's own layers run in order 0 .. half - 1 and then its boundary; every layer and boundary is preceded by the wait for its
/// predecessor's event (B's on A's of the same pair, A's on the previous pair's B unless `first`); B's layer L (and its boundary) is
/// issued only after A's layer L (boundary); no step touches a layer twice; the list is complete. (The two-lane case of `checkGroup`.)
pub fn check(half: usize, first: bool, steps: []const Step) Verdict {
    return checkGroup(2, half, first, steps);
}

/// `check` for a group of `lanes` passes: pass k waits only a layer its predecessor k - 1 has issued; the list has every pass of the group
/// and no step of another.
pub fn checkGroup(lanes: usize, half: usize, first: bool, steps: []const Step) Verdict {
    var next: [max_lanes]usize = @splat(0); // the next layer each pass may run
    var boundary: [max_lanes]bool = @splat(false);
    var waited: [max_lanes]?usize = @splat(null); // the layer index each pass has waited for and not yet used
    for (steps, 0..) |st, i| switch (st) {
        .wait => |w| {
            const p = @backingInt(w.pass);
            if (p >= lanes) return .{ .ok = false, .reason = "a step of a pass the group does not have", .at = i };
            if (w.li > half) return .{ .ok = false, .reason = "a wait for a layer past the boundary", .at = i };
            if (boundary[p]) return .{ .ok = false, .reason = "a wait after the pass's boundary", .at = i };
            if (p != 0) {
                const done = if (w.li == half) boundary[p - 1] else next[p - 1] > w.li;
                if (!done) return .{ .ok = false, .reason = "a pass waits a layer its predecessor has not issued", .at = i };
            }
            waited[p] = w.li;
        },
        .layer => |l| {
            const p = @backingInt(l.pass);
            if (p >= lanes) return .{ .ok = false, .reason = "a step of a pass the group does not have", .at = i };
            if (l.li != next[p]) return .{ .ok = false, .reason = "a pass's layers are not in order", .at = i };
            if (p == 0 and first) {} else if (waited[p] == null or waited[p].? != l.li) return .{ .ok = false, .reason = "a layer runs before its cache wait", .at = i };
            waited[p] = null;
            next[p] += 1;
        },
        .boundary => |q| {
            const p = @backingInt(q);
            if (p >= lanes) return .{ .ok = false, .reason = "a step of a pass the group does not have", .at = i };
            if (next[p] != half) return .{ .ok = false, .reason = "a boundary before the pass's layers", .at = i };
            if (boundary[p]) return .{ .ok = false, .reason = "a pass's boundary twice", .at = i };
            if (p == 0 and first) {} else if (waited[p] == null or waited[p].? != half) return .{ .ok = false, .reason = "a boundary runs before its cache wait", .at = i };
            waited[p] = null;
            boundary[p] = true;
        },
    };
    for (0..lanes) |p| if (next[p] != half or !boundary[p]) return .{ .ok = false, .reason = "a pass is incomplete", .at = steps.len };
    return .{};
}

// ---------------------------------------------------------------------------------------------------------------
// Tests.

        try std.testing.expect(eligible(true, p, p, 0, p, p, p, 2 * p, true));
        try std.testing.expect(eligible(true, p, p, 3 * p, p, 4 * p, p, 5 * p + 77, true)); // anywhere in the prompt, a decoder window behind
        try std.testing.expect(!eligible(false, p, p, 0, p, p, p, 2 * p, true)); // switch off
        try std.testing.expect(!eligible(true, p, p / 2, 0, p, p, p, 2 * p, true)); // the rule cut the pass (below 2,048 the booted rows are the cut)
        try std.testing.expect(!eligible(true, p, 2 * p, 0, p, p, p, 2 * p, true)); // a pass above the booted rows
        try std.testing.expect(!eligible(true, p, p, 0, p - 1024, p, p, 2 * p, true)); // a short first pass
        try std.testing.expect(!eligible(true, p, p, 0, p, p, p - 1024, 2 * p, true)); // a short second pass
        try std.testing.expect(!eligible(true, p, p, 0, p, p + 1, p, 2 * p, true)); // not consecutive
        try std.testing.expect(!eligible(true, p, p, 0, p, p, p, 2 * p - 1, true)); // the second has decoder rows
        try std.testing.expect(!eligible(true, p, p, 0, p, p, p, 2 * p, false)); // no worker
    }
    // the rows are the booted ones: another setting's pass is no pair
    try std.testing.expect(!eligible(true, 4096, 4096, 0, 2048, 2048, 2048, 4096, true));
    try std.testing.expect(!eligible(true, 2048, 2048, 0, 4096, 4096, 4096, 8192, true));
    try std.testing.expect(!eligible(true, 8192, 8192, 0, 4096, 4096, 4096, 8192, true));
}

    try std.testing.expectEqual(@as(usize, 2), groupChunks(2048, 2048));
    try std.testing.expectEqual(@as(usize, 4), groupChunks(4096, 2048));
    try std.testing.expectEqual(@as(usize, 8), groupChunks(8192, 2048));
}

test "the pair's issue order: A's layer L, then B's layer L - gap; each pass's layers in order, every layer behind its wait, then the boundaries" {
    for ([_]bool{ true, false }) |first| {
        var buf: [maxSteps(20)]Step = undefined;
        const s = pairSchedule(20, gap, first, &buf);
        const v = check(20, first, s);
        if (!v.ok) {
            std.debug.print("{s} at {d}\n", .{ v.reason, v.at });
            return error.ScheduleRejected;
        }
        // B's layer 0 is issued after A's layer `gap`, not before; A's boundary precedes B's
        var a_gap: ?usize = null;
        var b0: ?usize = null;
        var ba: ?usize = null;
        var bb: ?usize = null;
        for (s, 0..) |st, i| switch (st) {
            .layer => |l| {
                if (l.pass == .a and l.li == gap) a_gap = i;
                if (l.pass == .b and l.li == 0) b0 = i;
            },
            .boundary => |q| if (q == .a) {
                ba = i;
            } else {
                bb = i;
            },
            else => {},
        };
        try std.testing.expect(a_gap.? < b0.?);
        try std.testing.expect(ba.? < bb.?);
        // the first pair of a prompt has no wait on a predecessor; a later pair waits every layer and its boundary
        var a_waits: usize = 0;
        for (s) |st| switch (st) {
            .wait => |w| a_waits += @intFromBool(w.pass == .a),
            else => {},
        };
        try std.testing.expectEqual(if (first) @as(usize, 0) else @as(usize, 21), a_waits);
    }
}

test "the issue gap: 1, 2 and 3 each give a schedule the checker accepts; anything else is the default" {
    for ([_]usize{ 1, 2, 3 }) |g| for ([_]bool{ true, false }) |first| {
        var buf: [maxSteps(20)]Step = undefined;
        const s = pairSchedule(20, g, first, &buf);
        try std.testing.expect(check(20, first, s).ok);
    };
    try std.testing.expectEqual(@as(usize, 3), gapOf("3"));
    try std.testing.expectEqual(@as(usize, 1), gapOf(" 1 "));
    for ([_]?[]const u8{ null, "", "0", "4", "x", "-1" }) |t| try std.testing.expectEqual(gap, gapOf(t));
}

test "the streams' priorities: 0 equal, 1 A above B, 2 the reverse; anything else is 0; a lower number is more urgent" {
    try std.testing.expectEqual(Prio.equal, prioOf(null));
    try std.testing.expectEqual(Prio.equal, prioOf("0"));
    try std.testing.expectEqual(Prio.a_first, prioOf("1"));
    try std.testing.expectEqual(Prio.b_first, prioOf(" 2 "));
    for ([_]?[]const u8{ "3", "", "x", "-1", "11" }) |t| try std.testing.expectEqual(Prio.equal, prioOf(t));
    try std.testing.expectEqual([2]c_int{ 0, 0 }, prioritiesOf(.equal, 0, -5));
    try std.testing.expectEqual([2]c_int{ -5, 0 }, prioritiesOf(.a_first, 0, -5));
    try std.testing.expectEqual([2]c_int{ 0, -5 }, prioritiesOf(.b_first, 0, -5));
    // a mode other than 0 puts one stream strictly more urgent than the other
    for ([_]Prio{ .a_first, .b_first }) |m| {
        const pr = prioritiesOf(m, 0, -5);
        try std.testing.expect(pr[0] != pr[1]);
    }
}

}

test "the list is a pure function of its inputs: two ranks build the same one" {
    var b1: [maxSteps(20)]Step = undefined;
    var b2: [maxSteps(20)]Step = undefined;
    const s1 = pairSchedule(20, gap, false, &b1);
    const s2 = pairSchedule(20, gap, false, &b2);
    try std.testing.expectEqual(s1.len, s2.len);
    for (s1, s2) |x, y| try std.testing.expect(sameStep(x, y));
}

test "the checker rejects a pair without the cache gate, a pass whose layers run out of order, and a boundary ahead of A's" {
    const half = 20;
    var steps: [maxSteps(half)]Step = undefined;
    // no waits at all: every layer would run before its cache is written
    var n: usize = 0;
    for (0..half) |li| {
        steps[n] = .{ .layer = .{ .pass = .a, .li = li } };
        n += 1;
        steps[n] = .{ .layer = .{ .pass = .b, .li = li } };
        n += 1;
    }
    steps[n] = .{ .boundary = .a };
    n += 1;
    steps[n] = .{ .boundary = .b };
    n += 1;
    try std.testing.expect(!check(half, true, steps[0..n]).ok);
    // B's layer 1 before its layer 0
    const bad = [_]Step{
        .{ .layer = .{ .pass = .a, .li = 0 } },
        .{ .layer = .{ .pass = .a, .li = 1 } },
        .{ .wait = .{ .pass = .b, .li = 1 } },
        .{ .layer = .{ .pass = .b, .li = 1 } },
    };
    try std.testing.expect(!check(2, true, &bad).ok);
    // B's boundary before A's: take a good list and swap the two boundaries (and B's wait on A's boundary)
    var buf: [maxSteps(half)]Step = undefined;
    const good = pairSchedule(half, gap, true, &buf);
    try std.testing.expect(check(half, true, good).ok);
    var sw: [maxSteps(half)]Step = undefined;
    @memcpy(sw[0..good.len], good);
    const k = good.len;
    sw[k - 3] = good[k - 2]; // B's wait on A's boundary
    sw[k - 2] = good[k - 1]; // B's boundary
    sw[k - 1] = good[k - 3]; // A's boundary, last
    try std.testing.expect(!check(half, true, sw[0..k]).ok);
    // a later pair whose A skips the wait on the previous pair's B
    const nowait = pairSchedule(half, gap, true, &buf); // the first-pair list has no A waits
    try std.testing.expect(!check(half, false, nowait).ok);
}

test "the cache events: one set a pass of the group and a layer including the boundary's" {
    const e: Events = undefined;
    try std.testing.expectEqual(@as(usize, max_lanes), e.ev.len);
    try std.testing.expectEqual(@as(usize, max_half + 1), e.ev[0].len);
}

test "streams made at the range's ends report them: the urgent one has the lower number" {
    var driver = cuda.Driver.open() catch return error.SkipZigTest;
    defer driver.close();
    var cx = cuda.Context.init(&driver, 0) catch return error.SkipZigTest;
    defer cx.deinit();
    const range = try cuda.Stream.priorityRange(&driver);
    try std.testing.expect(range.greatest <= range.least);
    const pr = prioritiesOf(.a_first, range.least, range.greatest);
    var a = try cuda.Stream.initPriority(&driver, pr[0]);
    defer a.deinit();
    var b = try cuda.Stream.initPriority(&driver, pr[1]);
    defer b.deinit();
    try std.testing.expectEqual(pr[0], try a.priority());
    try std.testing.expectEqual(pr[1], try b.priority());
    // (a context whose range is one value makes both equal: nothing to prefer, and no error)
    if (range.greatest < range.least) try std.testing.expect(try a.priority() < try b.priority());
}

// ---------------------------------------------------------------------------------------------------------------
// TF_DS_ROCI_STAGGER_LANES: three passes in flight.

/// The 2-lane schedule exactly as it was before the lane count existed (the reference `groupSchedule(2, ...)` must reproduce byte for byte).
fn referencePairSchedule(half: usize, gap_layers: usize, first: bool, out: []Step) []Step {
    var n: usize = 0;
    for (0..half) |li| {
        if (!first) {
            out[n] = .{ .wait = .{ .pass = .a, .li = li } };
            n += 1;
        }
        out[n] = .{ .layer = .{ .pass = .a, .li = li } };
        n += 1;
        if (li >= gap_layers) {
            out[n] = .{ .wait = .{ .pass = .b, .li = li - gap_layers } };
            n += 1;
            out[n] = .{ .layer = .{ .pass = .b, .li = li - gap_layers } };
            n += 1;
        }
    }
    for (half - gap_layers..half) |j| {
        out[n] = .{ .wait = .{ .pass = .b, .li = j } };
        n += 1;
        out[n] = .{ .layer = .{ .pass = .b, .li = j } };
        n += 1;
    }
    if (!first) {
        out[n] = .{ .wait = .{ .pass = .a, .li = half } };
        n += 1;
    }
    out[n] = .{ .boundary = .a };
    n += 1;
    out[n] = .{ .wait = .{ .pass = .b, .li = half } };
    n += 1;
    out[n] = .{ .boundary = .b };
    n += 1;
    return out[0..n];
}

test "two lanes: the generalised schedule is the pair's schedule byte for byte, at every half, gap and first" {
    for ([_]usize{ 1, 2, 5, 20, 32 }) |half| for ([_]usize{ 1, 2, 3 }) |g| for ([_]bool{ true, false }) |first| {
        if (g > half) continue;
        var b1: [maxSteps(32)]Step = undefined;
        var b2: [maxSteps(32)]Step = undefined;
        const want = referencePairSchedule(half, g, first, &b1);
        const got = groupSchedule(2, half, g, first, &b2);
        try std.testing.expectEqual(want.len, got.len);
        for (want, got) |x, y| try std.testing.expect(sameStep(x, y));
        const old = pairSchedule(half, g, first, &b1);
        try std.testing.expectEqual(want.len, old.len);
    };
}

test "three lanes: the checker accepts the schedule at every gap and first; C's layer L comes after B's, B's after A's" {
    for ([_]usize{ 1, 2, 3 }) |g| for ([_]bool{ true, false }) |first| {
        var buf: [maxSteps(20)]Step = undefined;
        const s = groupSchedule(3, 20, g, first, &buf);
        const v = checkGroup(3, 20, first, s);
        if (!v.ok) {
            std.debug.print("{s} at {d}\n", .{ v.reason, v.at });
            return error.ScheduleRejected;
        }
        var at: [3][21]usize = @splat(@splat(0));
        for (s, 0..) |st, i| switch (st) {
            .layer => |l| at[@backingInt(l.pass)][l.li] = i,
            else => {},
        };
        for (0..20) |li| {
            try std.testing.expect(at[0][li] < at[1][li]);
            try std.testing.expect(at[1][li] < at[2][li]);
        }
        // C's layer 0 is issued only after A's layer 2 gap layers on: the lag is gap a lane
        var c0: usize = 0;
        var a2g: usize = 0;
        for (s, 0..) |st, i| switch (st) {
            .layer => |l| {
                if (l.pass == .c and l.li == 0) c0 = i;
                if (l.pass == .a and l.li == 2 * g) a2g = i;
            },
            else => {},
        };
        try std.testing.expect(a2g < c0);
        // a later group's A waits the previous group's C (parity 2) at every layer and its boundary
        var a_waits: usize = 0;
        for (s) |st| switch (st) {
            .wait => |w| a_waits += @intFromBool(w.pass == .a),
            else => {},
        };
        try std.testing.expectEqual(if (first) @as(usize, 0) else @as(usize, 21), a_waits);
        try std.testing.expectEqual(@as(usize, 2), predecessor(0, 3));
        try std.testing.expectEqual(@as(usize, 0), predecessor(1, 3));
        try std.testing.expectEqual(@as(usize, 1), predecessor(2, 3));
        try std.testing.expectEqual(@as(usize, 1), predecessor(0, 2));
    };
}

test "three lanes: every rank builds the same list, and the checker rejects a lane ahead of its predecessor or a missing lane" {
    var b1: [maxSteps(20)]Step = undefined;
    var b2: [maxSteps(20)]Step = undefined;
    const s1 = groupSchedule(3, 20, gap, false, &b1);
    const s2 = groupSchedule(3, 20, gap, false, &b2);
    try std.testing.expectEqual(s1.len, s2.len);
    for (s1, s2) |x, y| try std.testing.expect(sameStep(x, y));
    // C's layer 0 before B has issued it
    const bad = [_]Step{
        .{ .layer = .{ .pass = .a, .li = 0 } },
        .{ .wait = .{ .pass = .c, .li = 0 } },
        .{ .layer = .{ .pass = .c, .li = 0 } },
    };
    try std.testing.expect(!checkGroup(3, 2, true, &bad).ok);
    // a schedule of two lanes is incomplete for three
    var buf: [maxSteps(20)]Step = undefined;
    const two = groupSchedule(2, 20, gap, true, &buf);
    try std.testing.expect(!checkGroup(3, 20, true, two).ok);
    // and a three-lane list is not a pair
    const three = groupSchedule(3, 20, gap, true, &buf);
    try std.testing.expect(!checkGroup(2, 20, true, three).ok);
}

test "TF_DS_ROCI_STAGGER_LANES: unset is 2, 2 and 3 are themselves, anything else is refused (null)" {
    try std.testing.expectEqual(@as(?usize, 2), lanesOf(null));
    try std.testing.expectEqual(@as(?usize, 2), lanesOf(" 2 "));
    try std.testing.expectEqual(@as(?usize, 3), lanesOf("3"));
    for ([_][]const u8{ "", "1", "4", "x", "-3", "03x" }) |t| try std.testing.expectEqual(@as(?usize, null), lanesOf(t));
}

test "a group of three is three whole encoder-only passes back to back with the worker; the pair's rule is the two-lane case" {
    var p: usize = 2048;
    while (p <= 16384) : (p += 2048) {
        const all = [_]usize{ p, p, p };
        try std.testing.expect(eligibleGroup(true, 3, p, p, 0, &all, 3 * p, true));
        try std.testing.expect(eligibleGroup(true, 3, p, p, 5 * p, &all, 8 * p + 9, true));
        try std.testing.expect(!eligibleGroup(true, 3, p, p, 0, &all, 3 * p - 1, true)); // the third has decoder rows
        try std.testing.expect(!eligibleGroup(true, 3, p, p, 0, &.{ p, p, p - 1024 }, 3 * p, true)); // a short third pass
        try std.testing.expect(!eligibleGroup(true, 3, p, p, 0, &.{ p, p }, 3 * p, true)); // two sizes for three lanes
        try std.testing.expect(!eligibleGroup(true, 3, p, p, 0, &all, 3 * p, false)); // no worker
        try std.testing.expect(!eligibleGroup(false, 3, p, p, 0, &all, 3 * p, true)); // switch off
        try std.testing.expect(eligibleGroup(true, 2, p, p, 0, all[0..2], 2 * p, true) == eligible(true, p, p, 0, p, p, p, 2 * p, true));
        try std.testing.expectEqual(3 * (p / 2048), groupChunksN(3, p, 2048));
        try std.testing.expectEqual(groupChunks(p, 2048), groupChunksN(2, p, 2048));
    }
}

test "the cache events: one set a lane, three lanes at most" {
    const e: Events = undefined;
    try std.testing.expectEqual(@as(usize, max_lanes), e.ev.len);
    try std.testing.expectEqual(@as(usize, 3), max_lanes);
    try std.testing.expectEqual(@as(u2, 2), @backingInt(PairPass.c));
}
