//! TF_DS_ENGRAM_KEEP_AHEAD=1: the Engram worker of a prompt's fill, and the rows it read ahead, stay across the decode round
//! that runs between two chunks (TF_DS_PREFILL_YIELD). The round reads Engram through buffers, a reader ring and records of
//! its own, so the worker has nothing to give up; this file holds the start decision, the check that the two sides share no
//! scratch, and the counters the pass line prints.

const std = @import("std");

pub const Decision = union(enum) {
    /// The variable is unset: a round between chunks joins the worker, as served.
    off,
    /// The worker stays.
    keep,
    /// Set, but there is no worker to keep (the start line says why).
    no_effect: []const u8,
    /// Set, and the round and the worker would share a writable reader: the start is refused.
    refuse: []const u8,
};

pub const sharing_pool = "TF_DS_ENGRAM_KEEP_AHEAD=1 needs TF_DS_ENGRAM_AIO=1: without it the round and the worker read through one reader pool, which takes one job at a time";

pub fn decide(on: bool, has_engram: bool, worker_on: bool, round_aio: bool, pass_aio: bool) Decision {
    if (!on) return .off;
    if (!has_engram) return .{ .no_effect = "the model has no Engram layers" };
    if (!worker_on) return .{ .no_effect = "the Engram worker is off (TF_DS_ENGRAM_WORKER=0)" };
    if (!round_aio or !pass_aio) return .{ .refuse = sharing_pool };
    return .keep;
}

/// A span of memory, by address.
pub const Range = struct { ptr: usize = 0, len: usize = 0 };

pub fn rangeOf(comptime T: type, s: []const T) Range {
    return .{ .ptr = @intFromPtr(s.ptr), .len = s.len * @sizeOf(T) };
}

/// The first range of `round` and the first of `worker` (by index in each) that overlap, empty ranges aside.
pub fn firstShared(round: []const Range, worker: []const Range) ?[2]usize {
    for (round, 0..) |a, i| {
        if (a.len == 0) continue;
        for (worker, 0..) |b, k| {
            if (b.len == 0) continue;
            if (a.ptr < b.ptr + b.len and b.ptr < a.ptr + a.len) return .{ i, k };
        }
    }
    return null;
}

pub const Taken = struct { wait_ns: u64, chunks: u64 };

/// What the pass line prints about the worker across rounds. Engine thread only.
pub const Stats = struct {
    starts: u64 = 0,
    rounds: u64 = 0,
    rounds_at_chunk: u64 = 0,
    after_round: bool = false,
    wait_ns: u64 = 0,
    chunks: u64 = 0,

    /// A prompt's fill begins: its worker starts are counted afresh.
    pub fn newPrompt(s: *Stats) void {
        s.starts = 0;
        s.rounds_at_chunk = s.rounds;
        s.after_round = false;
    }
    /// The fill's worker was (re)started: 1 for a prompt that never restarts it.
    pub fn workerStarted(s: *Stats) void {
        s.starts += 1;
    }
    /// A decode round ran.
    pub fn roundRan(s: *Stats) void {
        s.rounds += 1;
    }
    /// A chunk (or pair) begins: whether a round ran since the last one.
    pub fn chunkBegins(s: *Stats) void {
        s.after_round = s.rounds != s.rounds_at_chunk;
        s.rounds_at_chunk = s.rounds;
    }
    /// The engine thread waited `ns` for the rows of the chunk's Engram layer j = 0; kept for the first such wait of a
    /// chunk that came right after a round.
    pub fn firstLayerWait(s: *Stats, ns: u64) void {
        if (!s.after_round) return;
        s.after_round = false;
        s.wait_ns += ns;
        s.chunks += 1;
    }
    /// The wait and chunk counts since the last call.
    pub fn take(s: *Stats) Taken {
        const t: Taken = .{ .wait_ns = s.wait_ns, .chunks = s.chunks };
        s.wait_ns = 0;
        s.chunks = 0;
        return t;
    }
};

test "unset: a round between chunks joins the worker as served" {
    try std.testing.expect(decide(false, true, true, true, true) == .off);
}

test "set with a worker and both reader rings: the worker stays" {
    try std.testing.expect(decide(true, true, true, true, true) == .keep);
}

test "set without a round ring or a pass ring: the start is refused, naming the pool" {
    for ([_][2]bool{ .{ false, true }, .{ true, false }, .{ false, false } }) |a| {
        const d = decide(true, true, true, a[0], a[1]);
        try std.testing.expect(d == .refuse);
        try std.testing.expectEqualStrings(sharing_pool, d.refuse);
    }
}

test "set with no worker or no Engram: no effect, never a refusal" {
    try std.testing.expect(decide(true, true, false, false, false) == .no_effect);
    try std.testing.expect(decide(true, false, true, false, false) == .no_effect);
}

test "shared scratch: disjoint ranges pass, a shared byte is named, an empty range overlaps nothing" {
    var buf: [64]u8 = undefined;
    const a = rangeOf(u8, buf[0..16]);
    const b = rangeOf(u8, buf[16..32]);
    const c = rangeOf(u8, buf[31..40]);
    const e = rangeOf(u8, buf[20..20]);
    try std.testing.expect(firstShared(&.{ a, e }, &.{ b, e }) == null);
    const hit = firstShared(&.{ a, b }, &.{ e, c }).?;
    try std.testing.expectEqual([2]usize{ 1, 1 }, hit);
    // a range inside another overlaps it, and one that only touches does not
    try std.testing.expectEqual([2]usize{ 0, 0 }, firstShared(&.{rangeOf(u8, buf[0..32])}, &.{rangeOf(u8, buf[8..9])}).?);
    try std.testing.expect(firstShared(&.{a}, &.{b}) == null);
    var wide: [8]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 32), rangeOf(u32, &wide).len);
}

test "stats: starts count per prompt, and the first wait after a round is the only one kept" {
    var s: Stats = .{};
    s.newPrompt();
    s.workerStarted();
    s.chunkBegins(); // no round yet
    s.firstLayerWait(5_000);
    try std.testing.expectEqual(@as(u64, 0), s.take().chunks);
    s.roundRan();
    s.roundRan(); // two rounds between the chunks count once
    s.chunkBegins();
    s.firstLayerWait(7_000);
    s.firstLayerWait(9_000); // the second unit of a pass: not the first layer of the chunk
    const t = s.take();
    try std.testing.expectEqual(@as(u64, 1), t.chunks);
    try std.testing.expectEqual(@as(u64, 7_000), t.wait_ns);
    try std.testing.expectEqual(@as(u64, 0), s.take().chunks); // taken
    s.workerStarted();
    try std.testing.expectEqual(@as(u64, 2), s.starts);
    s.newPrompt();
    try std.testing.expectEqual(@as(u64, 0), s.starts);
}
