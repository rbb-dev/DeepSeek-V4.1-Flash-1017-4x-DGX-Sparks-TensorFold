//! CUDA graphs captured a key at a time by every rank together, and what happens when a capture fails: that work runs
//! eagerly on every rank instead, and its key runs eagerly for a while (nuk3s, TensorFold PR #45, carried over to the
//! ranks' vote the recipe's 0019 added: round.Graphs and draft.PassGraphs). Recording a graph runs nothing on the device, so
//! after a failed capture each rank puts its host state back and runs the work as it would without graphs; every rank
//! acts on the same vote, so every rank takes the same path and the exchanges stay matched.
const std = @import("std");
const cuda = @import("cuda");

/// A failed key's first hold in uses (each one eager), doubled at each further failure up to `max_hold`.
pub const first_hold: u32 = 1024;
pub const max_hold: u32 = first_hold << 6;

/// The uses a key runs eagerly after its `fails`-th failed capture (1 or more).
pub fn holdFor(fails: u32) u32 {
    const doublings: u5 = @intCast(@min(fails -| 1, std.math.log2_int(u32, max_hold / first_hold)));
    return first_hold << doublings;
}

/// The keys whose capture failed: how many more uses each runs eagerly, and how often its capture failed. Every rank
/// keeps the same (the same keys come up everywhere, and every change follows the ranks' vote).
pub fn Holds(comptime Key: type) type {
    return struct {
        const Self = @This();
        pub const Hold = struct { left: u32, fails: u32 };

        map: std.AutoHashMapUnmanaged(Key, Hold) = .empty,

        pub fn deinit(h: *Self, gpa: std.mem.Allocator) void {
            h.map.deinit(gpa);
        }

        /// Whether `key` runs eagerly this time: each such use counts its hold down; at zero it is captured again.
        pub fn eager(h: *Self, key: Key) bool {
            const x = h.map.getPtr(key) orelse return false;
            if (x.left == 0) return false;
            x.left -= 1;
            return true;
        }

        /// Room for one more failed key, made before the ranks vote: fail() after the vote cannot fail on one rank alone.
        pub fn reserve(h: *Self, gpa: std.mem.Allocator) !void {
            try h.map.ensureUnusedCapacity(gpa, 1);
        }

        /// Another failed capture of `key`, in the room reserve() made: its new hold (fails 1: its first failure).
        pub fn fail(h: *Self, key: Key) Hold {
            const got = h.map.getOrPutAssumeCapacity(key);
            if (!got.found_existing) got.value_ptr.* = .{ .left = 0, .fails = 0 };
            got.value_ptr.fails +|= 1;
            got.value_ptr.left = holdFor(got.value_ptr.fails);
            return got.value_ptr.*;
        }

        /// `key` was captured: it is held no longer.
        pub fn clear(h: *Self, key: Key) void {
            _ = h.map.remove(key);
        }
    };
}

/// How a rank's capture went: recorded and instantiated, failed once its work was recorded whole (at the capture's
/// end, its instantiate, or a test's injected failure), or failed before that (at its begin or in its work).
pub const Stage = enum { ok, after, during };

/// What every rank does after the vote: launch and keep its graph, run the work eagerly, or fail the round (G0.3's
/// answer to every failure before this).
pub const Outcome = enum { keep, eager, fail };

pub const ballot_len = 11;

/// rank failed each way, whether any captured, and whether every rank recorded as many calls: max(b) + max(255 - b)
/// is 255 only when every rank has the same byte b.
pub fn ballot(stage: Stage, calls: u32) [ballot_len]u8 {
    var b: [ballot_len]u8 = @splat(0);
    b[0] = @intFromBool(stage == .after);
    b[1] = @intFromBool(stage == .during);
    b[2] = @intFromBool(stage == .ok);
    const le = std.mem.toBytes(std.mem.nativeToLittle(u32, calls));
    for (le, 0..) |x, i| {
        b[3 + i] = x;
        b[7 + i] = 255 - x;
    }
    return b;
}

/// The ranks' choice from their ballots' byte-wise largest, the same on every rank:
///   none failed: keep;
///   every failure came once the work was recorded whole: every rank recorded the same exchanges, so eager;
///   without its variant): eager, which meets the same failure on every rank and fails the round cleanly;
///   rank may now differ from the others' (an RDMA ring's sequence lives on the device, untouched by a recording).
pub fn decide(m: [ballot_len]u8) Outcome {
    const after = m[0] != 0;
    const during = m[1] != 0;
    const captured = m[2] != 0;
    if (!after and !during) return .keep;
    if (!during) return .eager;
    var same = true;
    for (0..4) |i| same = same and @as(u16, m[3 + i]) + m[7 + i] == 255;
    return if (!captured and same) .eager else .fail;
}

/// The graph calls a capture makes, on CUDA (the host tests pass a fake with the same declarations).
pub const Cuda = struct {
    pub const Stream = cuda.Stream;
    pub const Graph = cuda.graph.Graph;
    pub const Exec = cuda.graph.Exec;

    pub fn begin(s: Stream) !void {
        return cuda.graph.beginCapture(s, .thread_local);
    }

    pub fn end(s: Stream) !Graph {
    }

    pub fn instantiate(g: Graph) !Exec {
    }

    pub fn launch(x: Exec, s: Stream) !void {
    }

    pub fn dropGraph(g: *Graph) void {
        g.deinit();
    }

    pub fn dropExec(x: *Exec) void {
        x.deinit();
    }
};

/// A kind of work's captures: the keys held eager, the captures attempted, the engine's count of eager fallbacks.
pub fn Lazy(comptime G: type, comptime Key: type) type {
    return struct {
        const Self = @This();

        holds: Holds(Key) = .{},
        captures: u64 = 0, // captures attempted (TF_DS_GRAPH_FAIL_TEST counts them)
        fallbacks: ?*std.atomic.Value(u64) = null, // the engine's work run eagerly after a failed capture (/metrics)

        pub fn deinit(l: *Self, gpa: std.mem.Allocator) void {
            l.holds.deinit(gpa);
            l.* = undefined;
        }

        /// `key`'s work when it has no graph. Held: eager. Else captured on every rank and, by the ranks' vote, launched
        /// and kept (store.keep), run eagerly with the key held (site.undo first), or failed (site.failed). `site`: the
        /// work (body), its host state put back (undo), its forks closed after a capture cut short (abort), the kept
        /// value (value), the log lines (fellBack, failed); `store`: room for the key (reserve) and the keeping (keep);
        /// (injected).
        pub fn run(l: *Self, gpa: std.mem.Allocator, s: G.Stream, key: Key, site: anytype, store: anytype, ranks: anytype) anyerror!void {
            if (l.holds.eager(key)) return site.body();
            // every rank to capture the same collectives (a rank that skipped the capture would leave the others'
            // recorded exchanges unmatched), and nothing after the vote below may fail on one rank alone. No room
            // anywhere: nothing is recorded anywhere, and every rank runs the work as it is.
            const room = if (store.reserve(gpa)) |_| l.holds.reserve(gpa) else |err| err;
            if (!try ranks.agreeAll(if (room) |_| true else |_| false)) return site.body();
            const before = ranks.calls();
            var stage: Stage = .ok;
            var failed: ?anyerror = null;
            var exec = record(s, site, &stage, &failed);
            l.captures += 1;
            if (stage == .ok and ranks.injected(l.captures)) {
                G.dropExec(&exec.?);
                exec = null;
                stage = .after;
                failed = error.GraphInjectedFailure;
            }
            // a capture cut short leaves its forks marked outstanding: their work was discarded with it, so the next
            // stretch must not wait on events recorded in it
            if (stage != .ok) site.abort();
            const outcome = ranks.agreeCapture(stage, ranks.calls() -% before) catch |err| {
                if (exec) |*x| G.dropExec(x);
                return err;
            };
            switch (outcome) {
                .keep => {
                    var x = exec.?;
                    errdefer G.dropExec(&x);
                    // launched before it is kept: a failed launch frees it here and leaves no handle in the store (a
                    // graph still running when destroyed is freed when it completes)
                    try G.launch(x, s);
                    store.keep(key, site.value(x));
                    l.holds.clear(key);
                },
                .eager => {
                    if (exec) |*x| G.dropExec(x);
                    site.undo();
                    const h = l.holds.fail(key);
                    if (l.fallbacks) |c| _ = c.fetchAdd(1, .monotonic);
                    if (h.fails == 1) site.fellBack(failed, h.left);
                    return site.body();
                },
                .fail => {
                    if (exec) |*x| G.dropExec(x);
                    site.failed(failed);
                    return failed orelse error.GraphCaptureFailed;
                },
            }
        }

        /// The site's work recorded and instantiated, else null with the stage and the error of its failure; a begun
        /// capture is always ended (its graph dropped).
        fn record(s: G.Stream, site: anytype, stage: *Stage, failed: *?anyerror) ?G.Exec {
            G.begin(s) catch |err| {
                stage.* = .during;
                failed.* = err;
                return null;
            };
            site.body() catch |err| {
                if (G.end(s)) |got| {
                    var g = got;
                    G.dropGraph(&g);
                } else |_| {}
                stage.* = .during;
                failed.* = err;
                return null;
            };
            var graph = G.end(s) catch |err| {
                stage.* = .after;
                failed.* = err;
                return null;
            };
            defer G.dropGraph(&graph);
            return G.instantiate(graph) catch |err| {
                stage.* = .after;
                failed.* = err;
                return null;
            };
        }
    };
}

// ---------------------------------------------------------------------------------------------------------- tests

/// A fake of the graph calls: each call's count, an error injected at a chosen call, the live graphs and execs.
const Fake = struct {
    pub const Stream = *Fake;
    pub const Graph = struct { f: *Fake };
    pub const Exec = struct { f: *Fake, id: u32 };
    pub const Call = enum { begin, end, instantiate, launch };

    fail_at: ?Call = null,
    fail_with: anyerror = error.CudaFailed,
    capturing: bool = false,
    calls: std.EnumArray(Call, u32) = .initFill(0),
    graphs: i32 = 0,
    execs: i32 = 0,

    fn step(f: *Fake, c: Call) !void {
        f.calls.getPtr(c).* += 1;
        if (f.fail_at == c) return f.fail_with;
    }

    pub fn begin(f: *Fake) !void {
        try f.step(.begin);
        f.capturing = true;
    }

    pub fn end(f: *Fake) !Graph {
        f.capturing = false; // (the driver ends a capture also when it reports it invalidated)
        try f.step(.end);
        f.graphs += 1;
        return .{ .f = f };
    }

    pub fn instantiate(g: Graph) !Exec {
        try g.f.step(.instantiate);
        g.f.execs += 1;
        return .{ .f = g.f, .id = g.f.calls.get(.instantiate) };
    }

    pub fn launch(x: Exec, f: *Fake) !void {
        std.debug.assert(x.f == f);
        try f.step(.launch);
    }

    pub fn dropGraph(g: *Graph) void {
        g.f.graphs -= 1;
    }

    pub fn dropExec(x: *Exec) void {
        x.f.execs -= 1;
    }
};

const TestKey = struct { rows: u32, bucket: u64 };
const Kept = struct { exec: Fake.Exec, state: u32 };

/// This rank's peers in the votes: their ballots (the same for every vote), whether they found room, and this rank's
const FakeRanks = struct {
    others: []const struct { Stage, u32 } = &.{ .{ .ok, 0 }, .{ .ok, 0 }, .{ .ok, 0 } },
    others_room: bool = true,
    inject_at: ?u64 = null,
    votes: u32 = 0, // agreeAll's and agreeCapture's
    vote_fails: ?anyerror = null,
    seen: ?Outcome = null,

    pub fn agreeAll(r: *FakeRanks, ok: bool) !bool {
        r.votes += 1;
        return ok and r.others_room;
    }

    pub fn agreeCapture(r: *FakeRanks, stage: Stage, recorded: u32) !Outcome {
        r.votes += 1;
        if (r.vote_fails) |e| return e;
        var m = ballot(stage, recorded);
        for (r.others) |o| {
            const b = ballot(o[0], o[1]);
            for (&m, b) |*x, y| x.* = @max(x.*, y);
        }
        r.seen = decide(m);
        return r.seen.?;
    }

    pub fn calls(r: *FakeRanks) u32 {
    }

    pub fn injected(r: *FakeRanks, nth: u64) bool {
        return r.inject_at == nth;
    }
};

/// runs captured and eager, its undos and aborts, its log lines, injected failures.
const FakeSite = struct {
    f: *Fake,
    r: *FakeRanks,
    state: u32 = 0,
    eager_runs: u32 = 0,
    undos: u32 = 0,
    aborts: u32 = 0,
    fell_back: u32 = 0,
    failed_lines: u32 = 0,
    body_fails: ?anyerror = null, // under capture
    eager_fails: ?anyerror = null,

    pub fn body(s: *FakeSite) !void {
        s.state += 1;
        if (s.f.capturing) {
            if (s.body_fails) |e| {
                return e;
            }
        } else {
            s.eager_runs += 1;
            if (s.eager_fails) |e| return e;
        }
    }

    pub fn undo(s: *FakeSite) void {
        s.undos += 1;
        s.state = 0;
    }

    pub fn abort(s: *FakeSite) void {
        s.aborts += 1;
    }

    pub fn value(s: *FakeSite, exec: Fake.Exec) Kept {
        return .{ .exec = exec, .state = s.state };
    }

    pub fn fellBack(s: *FakeSite, err: ?anyerror, left: u32) void {
        _ = err;
        _ = left;
        s.fell_back += 1;
    }

    pub fn failed(s: *FakeSite, err: ?anyerror) void {
        _ = err;
        s.failed_lines += 1;
    }
};

/// A fake store: the kept graphs by key, room refused when full.
const FakeStore = struct {
    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(TestKey, Kept) = .empty,
    full: bool = false,

    fn deinit(st: *FakeStore) void {
        var it = st.map.valueIterator();
        while (it.next()) |v| Fake.dropExec(&v.exec);
        st.map.deinit(st.gpa);
    }

    pub fn reserve(st: *FakeStore, gpa: std.mem.Allocator) !void {
        if (st.full) return error.OutOfMemory;
        try st.map.ensureUnusedCapacity(gpa, 1);
    }

    pub fn keep(st: *FakeStore, key: TestKey, v: Kept) void {
        st.map.putAssumeCapacityNoClobber(key, v);
    }
};

const TestLazy = Lazy(Fake, TestKey);

/// One rank's world for a test: the fakes and the lazy captures, torn down together.
const World = struct {
    f: Fake = .{},
    r: FakeRanks = .{},
    st: FakeStore = .{ .gpa = std.testing.allocator },
    l: TestLazy = .{},
    count: std.atomic.Value(u64) = .init(0),

    fn site(w: *World) FakeSite {
        return .{ .f = &w.f, .r = &w.r };
    }

    fn run(w: *World, key: TestKey, s: *FakeSite) anyerror!void {
        w.l.fallbacks = &w.count;
        return w.l.run(std.testing.allocator, &w.f, key, s, &w.st, &w.r);
    }

    fn deinit(w: *World) void {
        w.l.deinit(std.testing.allocator);
        w.st.deinit();
    }
};

/// The ranks' outcome when each holds one of these ballots.
fn outcomeOf(ballots: []const struct { Stage, u32 }) Outcome {
    var m: [ballot_len]u8 = @splat(0);
    for (ballots) |b| {
        const x = ballot(b[0], b[1]);
        for (&m, x) |*y, z| y.* = @max(y.*, z);
    }
    return decide(m);
}

test "a failed key's hold doubles with each further failure, up to the most" {
    try std.testing.expectEqual(first_hold, holdFor(1));
    try std.testing.expectEqual(first_hold * 2, holdFor(2));
    try std.testing.expectEqual(first_hold * 4, holdFor(3));
    try std.testing.expectEqual(max_hold, holdFor(7));
    try std.testing.expectEqual(max_hold, holdFor(8));
    try std.testing.expectEqual(max_hold, holdFor(std.math.maxInt(u32)));
}

test "a held key runs eagerly for its hold, then not; a failure in the room reserved before the vote counts again" {
    const gpa = std.testing.allocator;
    var h: Holds(TestKey) = .{};
    defer h.deinit(gpa);
    const k: TestKey = .{ .rows = 4, .bucket = 1024 };
    try std.testing.expect(!h.eager(k));
    try h.reserve(gpa);
    try std.testing.expectEqual(@as(u32, 1), h.fail(k).fails);
    for (0..first_hold) |_| try std.testing.expect(h.eager(k));
    try std.testing.expect(!h.eager(k));
    try h.reserve(gpa);
    const again = h.fail(k);
    try std.testing.expectEqual(@as(u32, 2), again.fails);
    try std.testing.expectEqual(first_hold * 2, again.left);
    h.clear(k);
    try std.testing.expect(!h.eager(k));
}

test "the ranks keep their graphs when none failed, and run eagerly when every failure came after the recording" {
    try std.testing.expectEqual(Outcome.keep, outcomeOf(&.{ .{ .ok, 40 }, .{ .ok, 40 }, .{ .ok, 40 }, .{ .ok, 40 } }));
    try std.testing.expectEqual(Outcome.eager, outcomeOf(&.{ .{ .ok, 40 }, .{ .after, 40 }, .{ .ok, 40 }, .{ .ok, 40 } }));
    try std.testing.expectEqual(Outcome.eager, outcomeOf(&.{ .{ .after, 40 }, .{ .after, 40 }, .{ .after, 40 }, .{ .after, 40 } }));
    // a world of one: its own ballot
    try std.testing.expectEqual(Outcome.keep, outcomeOf(&.{.{ .ok, 0 }}));
    try std.testing.expectEqual(Outcome.eager, outcomeOf(&.{.{ .after, 7 }}));
}

    // a launch no rank can make (a kit without its variant): every rank stops at it
    try std.testing.expectEqual(Outcome.eager, outcomeOf(&.{ .{ .during, 12 }, .{ .during, 12 }, .{ .during, 12 }, .{ .during, 12 } }));
    try std.testing.expectEqual(Outcome.eager, outcomeOf(&.{ .{ .during, 12 }, .{ .after, 12 }, .{ .during, 12 }, .{ .after, 12 } }));
    try std.testing.expectEqual(Outcome.fail, outcomeOf(&.{ .{ .ok, 40 }, .{ .during, 12 }, .{ .ok, 40 }, .{ .ok, 40 } }));
    try std.testing.expectEqual(Outcome.fail, outcomeOf(&.{ .{ .ok, 12 }, .{ .during, 12 }, .{ .ok, 12 }, .{ .ok, 12 } }));
    // every rank cut short, at different calls
    try std.testing.expectEqual(Outcome.fail, outcomeOf(&.{ .{ .during, 12 }, .{ .during, 13 }, .{ .during, 12 }, .{ .during, 12 } }));
}

    try std.testing.expectEqual(Outcome.fail, outcomeOf(&.{ .{ .during, 5 }, .{ .during, 5 + 256 } }));
    try std.testing.expectEqual(Outcome.fail, outcomeOf(&.{ .{ .during, 0 }, .{ .during, 1 << 24 } }));
    try std.testing.expectEqual(Outcome.fail, outcomeOf(&.{ .{ .during, 0x01020304 }, .{ .during, 0x04030201 } }));
    try std.testing.expectEqual(Outcome.eager, outcomeOf(&.{ .{ .during, std.math.maxInt(u32) }, .{ .during, std.math.maxInt(u32) } }));
    try std.testing.expectEqual(Outcome.eager, outcomeOf(&.{ .{ .during, 0 }, .{ .during, 0 }, .{ .during, 0 }, .{ .during, 0 } }));
}

test "every rank captures: the graph is launched once and kept, nothing runs eagerly" {
    var w: World = .{};
    defer w.deinit();
    var s = w.site();
    const k: TestKey = .{ .rows = 4, .bucket = 1024 };
    try w.run(k, &s);
    try std.testing.expectEqual(@as(usize, 1), w.st.map.count());
    try std.testing.expectEqual(@as(u32, 1), w.st.map.get(k).?.state); // the value made from the recorded state
    try std.testing.expectEqual(@as(u32, 1), w.f.calls.get(.launch));
    try std.testing.expectEqual(@as(u32, 0), s.eager_runs);
    try std.testing.expectEqual(@as(u32, 0), s.undos + s.aborts + s.fell_back + s.failed_lines);
    try std.testing.expectEqual(@as(i32, 0), w.f.graphs); // the graph is dropped once instantiated
    try std.testing.expectEqual(@as(i32, 1), w.f.execs);
    try std.testing.expectEqual(@as(u64, 0), w.count.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 2), w.r.votes); // the room's, then the capture's
}

test "another rank's failure after its recording: this rank drops its own graph, puts its state back, runs eagerly" {
    var w: World = .{};
    defer w.deinit();
    w.r.others = &.{ .{ .ok, 3 }, .{ .after, 3 }, .{ .ok, 3 } };
    var s = w.site();
    const k: TestKey = .{ .rows = 16, .bucket = 262144 };
    try w.run(k, &s);
    try std.testing.expectEqual(Outcome.eager, w.r.seen.?);
    try std.testing.expectEqual(@as(i32, 0), w.f.execs); // its own graph, captured fine, dropped
    try std.testing.expectEqual(@as(u32, 0), w.f.calls.get(.launch));
    try std.testing.expectEqual(@as(usize, 0), w.st.map.count());
    try std.testing.expectEqual(@as(u32, 1), s.undos);
    try std.testing.expectEqual(@as(u32, 1), s.eager_runs);
    try std.testing.expectEqual(@as(u32, 1), s.state); // the eager run started from the start state
    try std.testing.expectEqual(@as(u32, 1), s.fell_back);
    try std.testing.expectEqual(@as(u64, 1), w.count.load(.monotonic));
    try std.testing.expectEqual(first_hold, w.l.holds.map.get(k).?.left);
}

test "this rank's failure at begin, in its work, at the capture's end or its instantiate: the work runs from its start" {
    const Case = struct { at: ?Fake.Call, in_body: bool, others: Stage, aborts: u32 };
    const cases = [_]Case{
        // every rank fails the same way: a launch the kit has no variant for, a wait refused inside the capture
        .{ .at = null, .in_body = true, .others = .during, .aborts = 1 },
        .{ .at = .end, .in_body = false, .others = .ok, .aborts = 1 }, // CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (901)
        .{ .at = .instantiate, .in_body = false, .others = .ok, .aborts = 1 }, // CUDA_ERROR_NOT_PERMITTED (800)
        .{ .at = .begin, .in_body = false, .others = .during, .aborts = 1 },
    };
    for (cases) |c| {
        var w: World = .{};
        defer w.deinit();
        w.f.fail_at = c.at;
        const theirs: u32 = if (c.at == .begin) 0 else if (c.in_body) 1 else 3;
        const others = [_]struct { Stage, u32 }{ .{ c.others, theirs }, .{ c.others, theirs }, .{ c.others, theirs } };
        w.r.others = &others;
        var s = w.site();
        s.body_fails = if (c.in_body) error.MissingTritonVariant else null;
        const k: TestKey = .{ .rows = 16, .bucket = 262144 };
        try w.run(k, &s);
        try std.testing.expectEqual(Outcome.eager, w.r.seen.?);
        try std.testing.expectEqual(c.aborts, s.aborts);
        try std.testing.expectEqual(@as(u32, 1), s.undos);
        try std.testing.expectEqual(@as(u32, 1), s.eager_runs);
        try std.testing.expectEqual(@as(u32, 1), s.state);
        try std.testing.expect(!w.f.capturing); // a begun capture is always ended
        try std.testing.expectEqual(@as(u32, if (c.at == .begin) 0 else 1), w.f.calls.get(.end));
        try std.testing.expectEqual(@as(u32, 0), w.f.calls.get(.launch));
        try std.testing.expectEqual(@as(i32, 0), w.f.graphs);
        try std.testing.expectEqual(@as(i32, 0), w.f.execs);
        try std.testing.expectEqual(@as(usize, 0), w.st.map.count());
        try std.testing.expectEqual(@as(u64, 1), w.count.load(.monotonic));
        try std.testing.expectEqual(first_hold, w.l.holds.map.get(k).?.left);
    }
}

test "a rank cut short alone in its recording fails the round on every rank, as G0.3 did: nothing kept or held" {
    var w: World = .{};
    defer w.deinit();
    w.r.others = &.{ .{ .ok, 3 }, .{ .during, 1 }, .{ .ok, 3 } };
    var s = w.site();
    const k: TestKey = .{ .rows = 8, .bucket = 4096 };
    try std.testing.expectError(error.GraphCaptureFailed, w.run(k, &s));
    try std.testing.expectEqual(Outcome.fail, w.r.seen.?);
    try std.testing.expectEqual(@as(i32, 0), w.f.execs); // its own graph dropped
    try std.testing.expectEqual(@as(u32, 0), w.f.calls.get(.launch));
    try std.testing.expectEqual(@as(u32, 0), s.eager_runs);
    try std.testing.expectEqual(@as(u32, 1), s.failed_lines);
    try std.testing.expectEqual(@as(usize, 0), w.st.map.count());
    try std.testing.expect(w.l.holds.map.get(k) == null);
    try std.testing.expectEqual(@as(u64, 0), w.count.load(.monotonic));
    // this rank's own failure in the same vote is the error it returns
    var w2: World = .{};
    defer w2.deinit();
    w2.r.others = &.{ .{ .ok, 3 }, .{ .ok, 3 }, .{ .ok, 3 } };
    var s2 = w2.site();
    s2.body_fails = error.MissingTritonVariant;
    try std.testing.expectError(error.MissingTritonVariant, w2.run(k, &s2));
    try std.testing.expectEqual(@as(u32, 1), s2.aborts);
    try std.testing.expectEqual(@as(u32, 0), s2.eager_runs);
    // a capture this rank could not even begin, while the others recorded theirs: cut short too
    var w3: World = .{};
    defer w3.deinit();
    w3.f.fail_at = .begin;
    w3.r.others = &.{ .{ .ok, 3 }, .{ .ok, 3 }, .{ .ok, 3 } };
    var s3 = w3.site();
    try std.testing.expectError(error.CudaFailed, w3.run(k, &s3));
    try std.testing.expectEqual(Outcome.fail, w3.r.seen.?);
    try std.testing.expectEqual(@as(u32, 0), s3.eager_runs);
}

test "no room on some rank: every rank runs the work eagerly, with nothing recorded and nothing held" {
    var w: World = .{};
    defer w.deinit();
    w.r.others_room = false;
    var s = w.site();
    const k: TestKey = .{ .rows = 2, .bucket = 8192 };
    try w.run(k, &s);
    try std.testing.expectEqual(@as(u32, 0), w.f.calls.get(.begin));
    try std.testing.expectEqual(@as(u32, 1), s.eager_runs);
    try std.testing.expectEqual(@as(u32, 1), w.r.votes);
    try std.testing.expect(w.l.holds.map.get(k) == null);
    try std.testing.expectEqual(@as(u64, 0), w.count.load(.monotonic));
    // this rank's own store without room: the same
    var w2: World = .{};
    defer w2.deinit();
    w2.st.full = true;
    var s2 = w2.site();
    try w2.run(k, &s2);
    try std.testing.expectEqual(@as(u32, 0), w2.f.calls.get(.begin));
    try std.testing.expectEqual(@as(u32, 1), s2.eager_runs);
}

test "a held key runs eagerly with no vote for its hold, then is captured again; a success ends the hold" {
    var w: World = .{};
    defer w.deinit();
    w.f.fail_at = .instantiate;
    var s = w.site();
    const k: TestKey = .{ .rows = 2, .bucket = 8192 };
    try w.run(k, &s);
    try std.testing.expectEqual(@as(u32, 1), w.f.calls.get(.begin));
    const votes = w.r.votes;
    for (0..first_hold) |_| try w.run(k, &s);
    try std.testing.expectEqual(@as(u32, 1), w.f.calls.get(.begin)); // no capture while held
    try std.testing.expectEqual(votes, w.r.votes); // and no vote: every rank holds it alike
    try std.testing.expectEqual(first_hold + 1, s.eager_runs);
    try std.testing.expectEqual(@as(u32, 1), s.undos); // a held use has nothing to put back
    w.f.fail_at = null;
    try w.run(.{ .rows = 3, .bucket = 8192 }, &s); // another key is not held
    try std.testing.expectEqual(@as(u32, 2), w.f.calls.get(.begin));
    try w.run(k, &s); // its hold is over: captured again, and kept
    try std.testing.expectEqual(@as(u32, 3), w.f.calls.get(.begin));
    try std.testing.expect(w.st.map.get(k) != null);
    try std.testing.expect(w.l.holds.map.get(k) == null);
    try std.testing.expectEqual(@as(u32, 2), w.f.calls.get(.launch));
    try std.testing.expectEqual(@as(u64, 1), w.count.load(.monotonic));
}

test "each further failure of a key doubles its hold up to the most, and only its first is logged" {
    var w: World = .{};
    defer w.deinit();
    w.f.fail_at = .end;
    var s = w.site();
    const k: TestKey = .{ .rows = 1, .bucket = 1024 };
    var want: u32 = first_hold;
    for (0..3) |i| {
        try w.run(k, &s);
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), w.l.holds.map.get(k).?.fails);
        try std.testing.expectEqual(want, w.l.holds.map.get(k).?.left);
        for (0..want) |_| try w.run(k, &s);
        want *= 2;
    }
    try std.testing.expectEqual(@as(u32, 1), s.fell_back);
    try std.testing.expectEqual(@as(u64, 3), w.count.load(.monotonic));
    try w.run(.{ .rows = 1, .bucket = 2048 }, &s);
    try std.testing.expectEqual(@as(u32, 2), s.fell_back); // a new key logs its own first failure
}

test "an eager run after a failed capture returns its own error, and the key stays held" {
    var w: World = .{};
    defer w.deinit();
    w.f.fail_at = .end;
    var s = w.site();
    s.eager_fails = error.MissingTritonVariant;
    const k: TestKey = .{ .rows = 1, .bucket = 524288 };
    try std.testing.expectError(error.MissingTritonVariant, w.run(k, &s));
    try std.testing.expect(!w.f.capturing);
    try std.testing.expectEqual(first_hold, w.l.holds.map.get(k).?.left);
    try std.testing.expectError(error.MissingTritonVariant, w.run(k, &s)); // held: eager, no capture
    try std.testing.expectEqual(@as(u32, 1), w.f.calls.get(.begin));
}

test "a test's injected failure counts as one after the recording: every rank runs the work eagerly" {
    var w: World = .{};
    defer w.deinit();
    w.r.inject_at = 2;
    var s = w.site();
    try w.run(.{ .rows = 1, .bucket = 1024 }, &s); // the 1st capture: kept
    try w.run(.{ .rows = 2, .bucket = 1024 }, &s); // the 2nd: injected
    try std.testing.expectEqual(Outcome.eager, w.r.seen.?);
    try std.testing.expectEqual(@as(u32, 1), w.f.calls.get(.launch));
    try std.testing.expectEqual(@as(i32, 1), w.f.execs); // the 1st kept, the 2nd dropped
    try std.testing.expectEqual(@as(u32, 1), s.eager_runs);
    try std.testing.expectEqual(@as(u64, 2), w.l.captures);
}

test "a failed launch of a graph every rank kept is returned, and nothing is kept" {
    var w: World = .{};
    defer w.deinit();
    w.f.fail_at = .launch;
    var s = w.site();
    const k: TestKey = .{ .rows = 4, .bucket = 1024 };
    try std.testing.expectError(error.CudaFailed, w.run(k, &s));
    try std.testing.expectEqual(@as(u32, 0), s.eager_runs);
    try std.testing.expectEqual(@as(usize, 0), w.st.map.count());
    try std.testing.expectEqual(@as(i32, 0), w.f.execs);
}

test "a vote that fails is returned, this rank's graph dropped" {
    var w: World = .{};
    defer w.deinit();
    var s = w.site();
    try std.testing.expectEqual(@as(i32, 0), w.f.execs);
    try std.testing.expectEqual(@as(u32, 0), w.f.calls.get(.launch));
    try std.testing.expectEqual(@as(usize, 0), w.st.map.count());
}
