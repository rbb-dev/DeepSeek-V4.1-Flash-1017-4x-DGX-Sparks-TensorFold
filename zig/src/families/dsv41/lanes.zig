//! DeepSeek-V4.1-Flash behind TensorFold 1.0's lane core (core/lanes): the round loop picks each round's streams and
//! windows and keeps the accepted rows; this backend runs them on the port's model (model.zig) with the served lane's
//! arithmetic. Each stream has a pool slot (its window rings, positional stores and drafter rings) and an extent of
//! the shared compressed plane (first fit, chunk-aligned, as large as its prompt, its budget and a round's rows), and
//! its ids by position on both ranks (Engram's n-grams). verify is one round over every window (round.zig) with each
//! row drawn by the served sampler (sampling.zig) at its keyed position, and the drafting streams' window rows
//! absorbed into their drafter rings at once (the served eager absorb: the rows past the kept ones sit where the
//! drafter never reads, and the next window writes over them); draft is one batched drafter pass (draft.zig) for
//! every stream asked, the drafts held for the next round with their confidences, whose sigmoids are each draft's
//! chance of landing once the earlier ones did (probabilities: their running products, the chance that a draft and
//! every earlier one land; the round loop's allocation trims the windows by them). The caches are
//! positional, so keep only checks that a path is a prefix. Rank 0 runs the core; before each primitive it sends rank
//! 1 the primitive's inputs over the ranks' link, and rank 1 (follow) runs the same primitive: both ranks issue the
//! same kernels and collectives in the same order. The drafted reply is the serial one: every row is drawn with its
//! stream's key at its own position, whatever the round's composition. Structured output: a stream with a grammar
//! draws each row among its mask's tokens (the row with the others at -inf, drawn as any row), its first token from
//! the prompt's last row the prompt pass kept for it (first_masked), and the round loop reads its held drafts as host
//! tokens (held) to walk them through the grammar; rank 1 runs the same rounds, the masks are rank 0's alone.
const std = @import("std");
const lanes = @import("lanes");
const model = @import("model.zig");
const where = @import("where.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const sampling = @import("sampling.zig");
const link = @import("link.zig");
const cuda = @import("cuda");

const be = lanes.backend;
const Model = model.Model;
const max_streams = model.max_streams;
const pass_rows = 128; // a drafter pass's rows at most (prompt.grouped's decode window)
const max_rows = round.max_rows;

/// A window of the pending row and the drafter's block: the widest one the gates checked row for row.
pub const window_rows = 6;

/// The image rows a fill_images frame carries at most.
pub const image_piece = 32 << 10;

/// A gate's diagnostic (the tools' --logits-sha 1): each prompt's logits' sha256 on stderr.
pub var logits_sha = false;

/// The primitives rank 0 sends rank 1 (each frame: kind u8, step u64, its fields little-endian).
pub const Kind = enum(u8) { fill_begin = 1, fill_chunk = 2, verify = 3, pass = 4, snap_save = 5, snap_restore = 6, snap_drop = 7, fill_images = 8, done = 9, fill_switch = 10 };
        try m.eng.comm.gatherHost(d, p.dev.ptr, p.dev.ptr + 32, 32, m.eng.s);

/// A frame being written.
pub const Wire = struct {
    buf: std.ArrayList(u8) = .empty,

    pub fn begin(w: *Wire, gpa: std.mem.Allocator, kind: Kind, step: u64) !void {
        w.buf.clearRetainingCapacity();
        try w.int(gpa, u8, @backingInt(kind));
        try w.int(gpa, u64, step);
    }

    pub fn int(w: *Wire, gpa: std.mem.Allocator, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.buf.appendSlice(gpa, &b);
    }

    pub fn ids(w: *Wire, gpa: std.mem.Allocator, v: []const i32) !void {
        try w.int(gpa, u32, @intCast(v.len));
        try w.buf.ensureUnusedCapacity(gpa, v.len * 4);
        for (v) |x| try w.int(gpa, i32, x);
    }

    pub fn deinit(w: *Wire, gpa: std.mem.Allocator) void {
        w.buf.deinit(gpa);
    }
};

/// A frame being read.
pub const Reader = struct {
    b: []const u8,
    at: usize = 0,

    pub fn int(r: *Reader, comptime T: type) !T {
        const n = @sizeOf(T);
        if (r.at + n > r.b.len) return error.ShortFrame;
        const v = std.mem.readInt(T, r.b[r.at..][0..n], .little);
        r.at += n;
        return v;
    }

    /// The frame's bytes after those read.
    pub fn rest(r: *Reader) []const u8 {
        const b = r.b[r.at..];
        r.at = r.b.len;
        return b;
    }

    /// A list of ids written by Wire.ids, into `out` (replacing it).
    pub fn ids(r: *Reader, gpa: std.mem.Allocator, out: *std.ArrayList(i32)) !void {
        const n = try r.int(u32);
        try out.resize(gpa, n);
        for (out.items) |*x| x.* = try r.int(i32);
    }
};

/// A window of a round as both ranks build it: its stream's slot and extent, its first position, whether its rows'
/// taps go into the drafter, its rows' ids.
pub const Win = struct { slot: usize, base: usize, end: usize, start: usize, absorb: bool, ids: []const i32 };

/// The rows of a round (round.Rows' slices point here).
pub const Built = struct {
    ids: [max_rows]i64 = undefined,
    pos: [max_rows]i64 = undefined,
    slots: [max_rows]i64 = undefined,
    bases: [max_rows]i64 = undefined,
    ends: [max_rows]i64 = undefined,
    wins: [max_streams]round.Window = undefined,
    absorb: [max_streams]bool = undefined,
    r: usize = 0,
    n: usize = 0,

    /// Each window's ids written at its positions in its slot's sequence (the rows past them dropped: the rejected
    /// rows of the last round), then the round's rows in window order.
    pub fn build(b: *Built, gpa: std.mem.Allocator, seqs: *[max_streams]std.ArrayList(i32), wins: []const Win) !round.Rows {
        if (wins.len == 0 or wins.len > max_streams) return error.BadRound;
        b.r = 0;
        b.n = wins.len;
        for (wins, 0..) |w, i| {
            if (w.slot >= max_streams or w.ids.len == 0) return error.BadRound;
            if (b.r + w.ids.len > max_rows) return error.WindowTooWide;
            const seq = &seqs[w.slot];
            if (seq.items.len < w.start) return error.SequenceGap;
            try seq.resize(gpa, w.start + w.ids.len);
            @memcpy(seq.items[w.start..], w.ids);
            for (w.ids, 0..) |id, j| {
                b.ids[b.r + j] = id;
                b.pos[b.r + j] = @intCast(w.start + j);
                b.slots[b.r + j] = @intCast(w.slot);
                b.bases[b.r + j] = @intCast(w.base);
                b.ends[b.r + j] = @intCast(w.end);
            }
            b.wins[i] = .{ .row = b.r, .n = w.ids.len, .seq = seq.items[0 .. w.start + w.ids.len] };
            b.absorb[i] = w.absorb;
            b.r += w.ids.len;
        }
        const R = b.r;
        return .{ .ids = b.ids[0..R], .pos = b.pos[0..R], .slots = b.slots[0..R], .bases = b.bases[0..R], .ends = b.ends[0..R], .windows = b.wins[0..b.n] };
    }
};

/// The first chunk-aligned extent of `size` positions in [0, cap) that overlaps none of `used`.
/// Whether [lo, hi) meets any of `used`.
fn clashes(used: []const [2]usize, lo: usize, hi: usize) bool {
    for (used) |u| if (lo < u[1] and u[0] < hi) return true;
    return false;
}

/// Whether one of the prompt cache's marks has its resume point at `at` (each kept at the resume point at or before it).
fn markEnds(marks: []const u32, at: usize) bool {
    for (marks) |w| if (w / model.chunk_rows * model.chunk_rows == at) return true;
    return false;
}

/// Whether a pass from `start` may take the prompt's remainder (`rest`) as one final pass: the switch is on, the
/// remainder fits the cap, and the prompt's last grid remainder is not decode-sized. At or under
/// `prompt.decode_rows` the baseline tail pass takes a decode-class kernel (a ring window at 16 rows or fewer, the grouped
/// linears and the decode matmul at 128 or fewer); a merged pass runs those rows with the prompt kernels, a different
/// class. A remainder of 0 (or one past the decode rows) leaves the tail in the prompt kernels either way.
pub fn tailFinals(len: usize, start: usize, cap: usize, tail: bool) bool {
    if (!tail or len - start > cap) return false;
    const rem = len % model.chunk_rows;
    return rem == 0 or rem > prompt.decode_rows;
}

/// A prompt pass's rows from `start` of a `len`-row prompt under `cap` (stream.passRows'): a remainder under the resume
/// rows alone, else whole resume units up to the cap. With `tail`, a remainder that fits the cap is the final pass
/// itself, whole units or not, when tailFinals allows it; every pass before it stays whole units.
pub fn passLen(start: usize, len: usize, cap: usize, tail: bool) usize {
    const rest = len - start;
    if (tailFinals(len, start, cap, tail)) return rest;
    if (rest < model.chunk_rows) return rest;
    return @min(cap, rest / model.chunk_rows * model.chunk_rows);
}

/// A pass's rows from `start`: passLen's, cut at the first state it keeps. With `tail`, a final pass that tailFinals
/// allows ends at the prompt's end instead: the last resume point does not cut it (a mark still does).
pub fn passRowsAt(start: usize, len: usize, cap: usize, tail: bool, marks: []const u32, last_end: usize, replay: usize) usize {
    const n = passLen(start, len, cap, tail);
    const le = if (tailFinals(len, start, cap, tail)) 0 else last_end;
    return cutAtKeep(marks, le, replay, len, start, n);
}

/// The rows of the third pass of a group (TF_DS_ROCI_STAGGER_LANES=3) behind a pair whose second pass starts at `mid` and has `n2` rows, by
/// the pass rule (passRowsAt), or null where there is none: fewer than three lanes open, a stream decoding or the host owing rounds (the rounds
/// owed after a pair are counted for two passes), no prompt rows left behind the pair, or a stop at the pair's end (the prompt cache keeps a
/// state or a hook runs there, so the pair is the group). The pass's own eligibility (whole, encoder-only, the worker holding it) is the model's.
pub fn thirdPassRows(open_lanes: usize, decoding: bool, rounds: u32, mid: usize, n2: usize, len: usize, cap: usize, tail: bool, marks: []const u32, last_end: usize, replay: usize) ?usize {
    if (open_lanes < 3 or decoding or rounds != 0 or mid + n2 >= len) return null;
    const mid2 = mid + n2;
    if (mid2 <= replay and (mid2 == last_end or markEnds(marks, mid2))) return null;
    return passRowsAt(mid2, len, cap, tail, marks, last_end, replay);
}

/// A pass of a prompt's plan: its first row, its rows, and the cap the host allowed it.
pub const Pass = struct { start: usize, n: usize, cap: usize };

/// The most passes one prompt's plan holds.
pub const max_passes = 256;

/// A prompt's passes from `at` under the rule: their first rows and rows, and those over the resume rows.
pub const Passes = struct {
    list: [max_passes]Pass = undefined,
    len: usize = 0,
    long: usize = 0,

    pub fn slice(p: *const Passes) []const Pass {
        return p.list[0..p.len];
    }
};

/// The prompt's passes from `at` under the rule (the caps one after another from `caps`, the last repeated), each pass's
/// rows from passRowsAt.
pub fn passPlan(len: usize, at: usize, pass_rows: usize, pass_from: usize, caps: []const u32, marks: []const u32, tail: bool) !Passes {
    const r = model.chunk_rows;
    const replay = len -| 128;
    const last_end = @min(replay, len - 1) / r * r;
    var p: Passes = .{};
    var start = at;
    while (start < len) : (p.len += 1) {
        if (p.len == max_passes) return error.TooManyPasses;
        const quiet: lanes.stream.Facts = .{ .decoding = false, .filling = false, .waiting = false, .step_rows = std.math.maxInt(u32) };
        const asked: u32 = caps[@min(p.len, caps.len - 1)];
        const cap = if (asked > r) lanes.stream.passRows(quiet, start, pass_from, @intCast(pass_rows), r, prompt2d.pass_filling_asked) else r;
        const n = passRowsAt(start, len, cap, tail, marks, last_end, replay);
        if (n == 0) return error.EmptyPass;
        p.list[p.len] = .{ .start = start, .n = n, .cap = cap };
        p.long += @intFromBool(n > r);
        start += n;
    }
    if (start != len) return error.PassesDoNotTile;
    return p;
}

/// The prompt's passes from `at` under the rule, each checked against the 2,048-row grid's chunks: a pass holding a
/// decoder row (at or past the replay) or under the resume rows is exactly one of them (with `tail`, a final pass that
/// ends at the prompt may run up to its cap); a longer one runs encoder rows only and starts and ends on the grid. Every
/// mark's resume point (at or before the replay, past `at`, short of the end) is a pass end, and so is the last resume
/// point unless the final pass is merged.
pub fn checkPasses(len: usize, at: usize, pass_rows: usize, pass_from: usize, caps: []const u32, marks: []const u32, tail: bool) !Passes {
    const r = model.chunk_rows;
    const replay = len -| 128;
    const last_end = @min(replay, len - 1) / r * r;
    const p = try passPlan(len, at, pass_rows, pass_from, caps, marks, tail);
    var merged_final = false;
    for (p.slice()) |q| {
        const decoder = q.start + q.n > replay;
        const merged = tailFinals(len, q.start, q.cap, tail) and q.start % r == 0 and q.start + q.n == len and q.n <= q.cap and q.n > r;
        if (decoder or q.n < r) {
            if (q.start % r != 0 or (q.n != @min(r, len - q.start) and !merged)) return error.NotTheGridsChunk;
        } else if (q.start % r != 0 or (q.start + q.n) % r != 0 or q.start + q.n > replay) return error.LongPassNotEncoderOnly;
        if (q.n > pass_rows) return error.PassOverRows;
        merged_final = merged_final or merged;
    }
    if (last_end > at and !merged_final and !hasEnd(p.slice(), last_end)) return error.LastResumeNotAnEnd;
    for (marks) |w| {
        const k = w / r * r;
        if (k > at and k <= replay and k < len and !hasEnd(p.slice(), k)) return error.MarkNotAnEnd;
    }
    return p;
}

/// Whether a pass of `ps` ends at `x`.
fn hasEnd(ps: []const Pass, x: usize) bool {
    for (ps) |q| if (q.start + q.n == x) return true;
    return false;
}

/// A pass of `n` rows from `start` cut to end at the first state it keeps (the last resume point, or a mark's resume
/// point at or before the replay and short of the prompt's end), so that state stands at a pass end.
pub fn cutAtKeep(marks: []const u32, last_end: usize, replay: usize, len: usize, start: usize, n: usize) usize {
    var end = start + n;
    if (last_end > start and last_end < end) end = last_end;
    for (marks) |w| {
        const k = w / model.chunk_rows * model.chunk_rows;
        if (k > start and k < end and k <= replay and k < len) end = k;
    }
    return end - start;
}

test "cutAtKeep ends a long pass at the resume point it keeps" {
    const r = model.resume_rows;
    const c = model.chunk_rows;
    // no state to keep inside: the whole pass
    try std.testing.expectEqual(@as(usize, c), cutAtKeep(&.{}, 0, 10 * r, 10 * r, 0, c));
    // the last resume point inside the pass
    try std.testing.expectEqual(@as(usize, r), cutAtKeep(&.{}, 3 * r, 10 * r, 10 * r, 2 * r, c));
    // a mark floors to its resume point; the nearer of the two wins
    try std.testing.expectEqual(@as(usize, r), cutAtKeep(&.{@intCast(3 * r + 5)}, 9 * r, 10 * r, 10 * r, 2 * r, c));
    // a mark past the replay or at the prompt's end keeps nothing
    try std.testing.expectEqual(@as(usize, c), cutAtKeep(&.{@intCast(r)}, 0, r - 1, 10 * r, 0, c));
    // a resume point at the pass end needs no cut
    try std.testing.expectEqual(@as(usize, c), cutAtKeep(&.{@intCast(c)}, c, 10 * r, 10 * r, 0, c));
}

/// Room left past a kept state's extent while placements fit so, for its conversation's next turn to resume in place.
pub const growth_rows = 16 * model.chunk_rows;

/// The first fit of `size` clear of `live` and of kept extents plus growth_rows, else of their extents, else of their rows.
pub fn placeSoft(gpa: std.mem.Allocator, live: []const [2]usize, kept: []const [3]usize, size: usize, cap: usize) !?usize {
    var used: std.ArrayList([2]usize) = .empty;
    defer used.deinit(gpa);
    try used.appendSlice(gpa, live);
    for (kept) |k| try used.append(gpa, .{ k[0], @max(k[2], k[0] + k[1]) + growth_rows });
    if (place(used.items, size, cap)) |b| return b;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| try used.append(gpa, .{ k[0], @max(k[2], k[0] + k[1]) });
    if (place(used.items, size, cap)) |b| return b;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| try used.append(gpa, .{ k[0], k[0] + k[1] });
    return place(used.items, size, cap);
}

test "cutAtKeep ends a long pass at the resume point it keeps" {
    const r = model.chunk_rows;
    const c = 2 * r;
    // no state to keep inside: the whole pass
    try std.testing.expectEqual(@as(usize, c), cutAtKeep(&.{}, 0, 10 * r, 10 * r, 0, c));
    // the last resume point inside the pass
    try std.testing.expectEqual(@as(usize, r), cutAtKeep(&.{}, 3 * r, 10 * r, 10 * r, 2 * r, c));
    // a mark floors to its resume point; the nearer of the two wins
    try std.testing.expectEqual(@as(usize, r), cutAtKeep(&.{@intCast(3 * r + 5)}, 9 * r, 10 * r, 10 * r, 2 * r, c));
    // a mark past the replay or at the prompt's end keeps nothing
    try std.testing.expectEqual(@as(usize, c), cutAtKeep(&.{@intCast(r)}, 0, r - 1, 10 * r, 0, c));
    // a resume point at the pass end needs no cut
    try std.testing.expectEqual(@as(usize, c), cutAtKeep(&.{@intCast(c)}, c, 10 * r, 10 * r, 0, c));
}

test "a pass's rows: a remainder under the resume rows alone, else whole resume units up to the cap; a merged final pass takes the remainder" {
    const r = model.chunk_rows;
    try std.testing.expectEqual(@as(usize, 2 * r), passLen(0, 10 * r, 2 * r, false));
    try std.testing.expectEqual(@as(usize, r), passLen(0, 10 * r, r, false));
    // whole units only: 1.5 units left at a cap of two gives one
    try std.testing.expectEqual(@as(usize, r), passLen(8 * r, 9 * r + r / 2, 2 * r, false));
    try std.testing.expectEqual(@as(usize, r / 2), passLen(9 * r, 9 * r + r / 2, 2 * r, false));
    try std.testing.expectEqual(@as(usize, 5), passLen(0, 5, 2 * r, false));
    // the tail: a prompt-sized remainder up to the cap is the pass; a decode-sized one keeps whole units
    try std.testing.expectEqual(@as(usize, r), passLen(0, r + 1, 2 * r, true));
    try std.testing.expectEqual(@as(usize, r), passLen(0, r + 1, 2 * r, false));
    try std.testing.expectEqual(@as(usize, r + 129), passLen(0, r + 129, 2 * r, true));
    try std.testing.expectEqual(@as(usize, r), passLen(0, r + 129, 2 * r, false));
    try std.testing.expectEqual(@as(usize, 2 * r), passLen(0, 3 * r, 2 * r, true));
    // 19,685 rows from 16,384: the tail is 3,301 rows, not 2,048
    try std.testing.expectEqual(@as(usize, 3301), passLen(8 * r, 8 * r + 3301, 2 * r, true));
    try std.testing.expectEqual(@as(usize, r), passLen(8 * r, 8 * r + 3301, 2 * r, false));
}

test "the tail rule: a final pass that fits the cap is not cut at the last resume point, but a mark still cuts it" {
    const r = model.chunk_rows;
    // 12,288 rows from 8,192: two units, and the last resume point (10,240) does not cut them
    try std.testing.expectEqual(@as(usize, 2 * r), passRowsAt(4 * r, 6 * r, 2 * r, true, &.{}, 5 * r, 10 * r));
    try std.testing.expectEqual(@as(usize, r), passRowsAt(4 * r, 6 * r, 2 * r, false, &.{}, 5 * r, 10 * r));
    // a mark at 10,245 still cuts at 10,240, and the rest is the final pass
    try std.testing.expectEqual(@as(usize, r), passRowsAt(4 * r, 6 * r, 2 * r, true, &.{@intCast(5 * r + 5)}, 5 * r, 10 * r));
}

test "a prompt whose last grid remainder is decode-sized keeps the baseline passes: its tail runs the decode kernels" {
    const r = model.chunk_rows;
    // 4,097 rows: the baseline passes are 2,048, 2,048 and the 1-row tail. A merged 2,049-row pass would run that row in the
    // prompt kernels (non-ring window, one split, prefill linears) instead of the decode kernels (ring window, eight
    // splits, grouped linears), which changed bits.
    const ref = try checkPasses(2 * r + 1, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, false);
    const tail = try checkPasses(2 * r + 1, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, true);
    try std.testing.expectEqual(ref.len, tail.len);
    for (ref.slice(), tail.slice()) |a, b| {
        try std.testing.expectEqual(a.start, b.start);
        try std.testing.expectEqual(a.n, b.n);
    }
    // every remainder at or under the decode rows keeps the baseline split
    for ([_]usize{ 1, 16, 17, 127, 128 }) |rem| {
        const a = try checkPasses(2 * r + rem, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, false);
        const b = try checkPasses(2 * r + rem, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, true);
        try std.testing.expectEqual(a.len, b.len);
        for (a.slice(), b.slice()) |x, y| try std.testing.expectEqual(x.n, y.n);
    }
    // one past them, the remainder is a prompt-sized pass and merges: 2,177 rows are one pass (and 4,225 rows keep
    // the baseline 4,096 + 129 split, with the 129-row tail still a prompt-sized pass)
    try std.testing.expectEqual(@as(usize, 1), (try checkPasses(r + 129, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, true)).len);
    const two = try checkPasses(2 * r + 129, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, true);
    try std.testing.expectEqual(@as(usize, 2), two.len);
    try std.testing.expectEqual(@as(usize, 129), two.list[1].n);
}

test "the pass rule keeps every row in the grid's own chunk: a decoder row's pass and a short one are a 2,048-row chunk, a longer one encoder-only and whole; a merged final pass ends at the prompt" {
    const r = model.chunk_rows;
    var prng = std.Random.DefaultPrng.init(0x9a55);
    const rng = prng.random();
    var long_passes: usize = 0;
    var short_only: usize = 0;
    // every length to 8 resume units and 300 rows past, at pass rows 2,048 and 4,096, pass_from 0 and 8,192, without and
    // with a merged final pass, the cap 4,096 at every step, then 2,048 or 4,096 at random from step to step
    var len: usize = 1;
    while (len <= 8 * r + 300) : (len += 1) for ([_]usize{ r, 2 * r }) |pr| for ([_]usize{ 0, 4 * r }) |from| for ([_]bool{ false, true }) |tail| {
        const quiet = try checkPasses(len, 0, pr, from, &.{@intCast(2 * r)}, &.{}, tail);
        if (pr == r) short_only += quiet.long;
        long_passes += quiet.long;
        var caps: [16]u32 = undefined;
        for (&caps) |*c| c.* = if (rng.boolean()) @intCast(2 * r) else @intCast(r);
        _ = try checkPasses(len, 0, pr, from, &caps, &.{}, tail);
    };
    // resumed prompts with marks: from a resume point at or before the replay, two to five marks anywhere
    for (0..5000) |_| {
        const l = 1 + rng.uintLessThan(usize, 24 * r);
        const replay = l -| 128;
        const at = rng.uintLessThan(usize, replay / r + 1) * r;
        if (at >= l) continue;
        var marks: [5]u32 = undefined;
        const nm = 2 + rng.uintLessThan(usize, 4);
        for (marks[0..nm]) |*w| w.* = @intCast(rng.uintLessThan(usize, l + 1));
        _ = try checkPasses(l, at, 2 * r, rng.uintLessThan(usize, 5) * r, &.{@intCast(2 * r)}, marks[0..nm], rng.boolean());
    }
    // pass rows of 2,048 give no longer pass; 4,096 give them
    try std.testing.expectEqual(@as(usize, 0), short_only);
    try std.testing.expect(long_passes > 0);
    // a quiet 10-unit prompt (and its 128 decoder rows) from position 0 at pass rows 4,096: five passes of 4,096, then
    // the decoder rows' chunk
    const quiet = try checkPasses(10 * r + 128, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, false);
    try std.testing.expectEqual(@as(usize, 6), quiet.len);
    try std.testing.expectEqual(@as(usize, 5), quiet.long);
    // from pass_from 8,192: four passes of 2,048 first, then three of 4,096 and the decoder rows' chunk
    const from8 = try checkPasses(10 * r + 128, 0, 2 * r, 4 * r, &.{@intCast(2 * r)}, &.{}, false);
    try std.testing.expectEqual(@as(usize, 8), from8.len);
    try std.testing.expectEqual(@as(usize, 3), from8.long);
    // with the tail, 2,049 rows keep the baseline 2,048 + 1 split (the 1-row tail is decode-sized); 2,177 rows are one pass,
    // and 4,096 rows one, not two
    const small = try checkPasses(r + 1, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, true);
    try std.testing.expectEqual(@as(usize, 2), small.len);
    try std.testing.expectEqual(@as(usize, 1), small.list[1].n);
    try std.testing.expectEqual(@as(usize, 1), (try checkPasses(r + 129, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, true)).len);
    try std.testing.expectEqual(@as(usize, 2), (try checkPasses(r + 129, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, false)).len);
    try std.testing.expectEqual(@as(usize, 1), (try checkPasses(2 * r, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, true)).len);
    try std.testing.expectEqual(@as(usize, 2), (try checkPasses(2 * r, 0, 2 * r, 0, &.{@intCast(2 * r)}, &.{}, false)).len);
}

test "the pass counts of the served rule, D1 and the merged final pass at the design's lengths" {
    const r = model.chunk_rows;
    const Case = struct { len: usize, served: usize, d1: usize, d2a: usize };
    const cases = [_]Case{
        .{ .len = 546, .served = 1, .d1 = 1, .d2a = 1 },
        .{ .len = 1036, .served = 1, .d1 = 1, .d2a = 1 },
        .{ .len = 1133, .served = 1, .d1 = 1, .d2a = 1 },
        .{ .len = 2039, .served = 1, .d1 = 1, .d2a = 1 },
        .{ .len = 2225, .served = 2, .d1 = 2, .d2a = 1 },
        .{ .len = 3149, .served = 2, .d1 = 2, .d2a = 1 },
        .{ .len = 3229, .served = 2, .d1 = 2, .d2a = 1 },
        .{ .len = 4288, .served = 3, .d1 = 2, .d2a = 2 },
        .{ .len = 4703, .served = 3, .d1 = 2, .d2a = 2 },
        .{ .len = 6517, .served = 4, .d1 = 3, .d2a = 2 },
        .{ .len = 6767, .served = 4, .d1 = 3, .d2a = 2 },
        .{ .len = 8364, .served = 5, .d1 = 3, .d2a = 3 },
        .{ .len = 9180, .served = 5, .d1 = 3, .d2a = 3 },
        .{ .len = 12841, .served = 6, .d1 = 4, .d2a = 4 },
        .{ .len = 14524, .served = 7, .d1 = 5, .d2a = 4 },
        .{ .len = 17992, .served = 7, .d1 = 5, .d2a = 5 },
        .{ .len = 18797, .served = 8, .d1 = 6, .d2a = 5 },
        .{ .len = 37334, .served = 12, .d1 = 10, .d2a = 10 },
        .{ .len = 37588, .served = 12, .d1 = 10, .d2a = 10 },
        .{ .len = 156889, .served = 41, .d1 = 39, .d2a = 39 },
    };
    const cap: u32 = @intCast(2 * r);
    for (cases) |c| {
        try std.testing.expectEqual(c.served, (try checkPasses(c.len, 0, 2 * r, 4 * r, &.{cap}, &.{}, false)).len);
        try std.testing.expectEqual(c.d1, (try checkPasses(c.len, 0, 2 * r, 0, &.{cap}, &.{}, false)).len);
        const d2a = try checkPasses(c.len, 0, 2 * r, 0, &.{cap}, &.{}, true);
        try std.testing.expectEqual(c.d2a, d2a.len);
        // every pass starts on the grid, every pass but the last is whole units, and the last ends at the prompt within
        // the pass rows
        for (d2a.slice()) |q| try std.testing.expectEqual(@as(usize, 0), q.start % r);
        for (d2a.slice()[0 .. d2a.len - 1]) |q| try std.testing.expectEqual(@as(usize, 0), q.n % r);
        const last = d2a.list[d2a.len - 1];
        try std.testing.expectEqual(c.len, last.start + last.n);
        try std.testing.expect(last.n <= 2 * r);
    }
}

test "a placement leaves a kept state room to grow while one fits so" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    const kept = [_][3]usize{.{ 0, 48 * c, 50 * c }}; // rows [0, 48c), its stream's extent to 50c
    try std.testing.expectEqual(@as(?usize, 50 * c + growth_rows), try placeSoft(gpa, &.{}, &kept, 4 * c, 128 * c));
    // no room past the growth: flush against the extent, as before
    try std.testing.expectEqual(@as(?usize, 50 * c), try placeSoft(gpa, &.{}, &kept, 4 * c, 50 * c + growth_rows));
}

test "a kept state's tail stays free while a placement fits elsewhere" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    const kept = [_][3]usize{.{ 0, 4 * c, 5 * c }}; // rows [0, 4c), its stream's extent to 5c
    try std.testing.expectEqual(@as(?usize, 5 * c), try placeSoft(gpa, &.{}, &kept, 4 * c, 16 * c));
    // no room clear of the tail: the rows alone are held back
    const live = [_][2]usize{.{ 9 * c, 16 * c }};
    try std.testing.expectEqual(@as(?usize, 4 * c), try placeSoft(gpa, &live, &kept, 5 * c, 16 * c));
    try std.testing.expectEqual(@as(?usize, null), try placeSoft(gpa, &live, &kept, 6 * c, 16 * c));
}

/// Whether, with an extent of `size` at `b` taken besides `used`, another extent of `size` still fits in [0, cap).
pub fn roomAfter(gpa: std.mem.Allocator, used: []const [2]usize, b: usize, size: usize, cap: usize) !bool {
    const all = try gpa.alloc([2]usize, used.len + 1);
    defer gpa.free(all);
    @memcpy(all[0..used.len], used);
    all[used.len] = .{ b, b + size };
    return place(all, size, cap) != null;
}

test "a copy that fills the pool leaves no room for another request of its size" {
    const c = model.chunk_rows;
    // three live extents of 4 chunks in a 16-chunk pool: a fourth fits, and leaves no room for a fifth
    const used = [_][2]usize{ .{ 0, 4 * c }, .{ 4 * c, 8 * c }, .{ 8 * c, 12 * c } };
    const gpa = std.testing.allocator;
    const b = place(&used, 4 * c, 16 * c).?;
    try std.testing.expectEqual(@as(usize, 12 * c), b);
    try std.testing.expect(!try roomAfter(gpa, &used, b, 4 * c, 16 * c));
    // two live extents: the third leaves room for a fourth
    const b2 = place(used[0..2], 4 * c, 16 * c).?;
    try std.testing.expect(try roomAfter(gpa, used[0..2], b2, 4 * c, 16 * c));
}

pub fn place(used: []const [2]usize, size: usize, cap: usize) ?usize {
    var base: usize = 0;
    while (base + size <= cap) {
        var clash: ?usize = null;
        for (used) |u| {
            if (base < u[1] and u[0] < base + size) clash = @max(clash orelse 0, u[1]);
        }
        const past = clash orelse return base;
        base = std.mem.alignForward(usize, past, model.chunk_rows);
    }
    return null;
}

/// The token a row's logits draw at absolute position `position`: greedy without sampling (or at temperature 0),
/// else the served sampler with the stream's key.
/// draw from a row's first candidates (Track B): a greedy row's token is its first candidate (sampling.argmax's: the
/// first index of the largest value; the GPU's flags send a row with a NaN down draw's path instead); a sampled row's
/// is sampleRow's choice on its own candidateCount of them.
pub fn drawCandidates(vals: []const f32, ids: []const i64, n: usize, position: u64, s: ?lanes.Sampling) !u32 {
    const sm = s orelse return @intCast(ids[0]);
    if (sm.temperature <= 0) return @intCast(ids[0]);
    const smp: sampling.Sampling = .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p };
    const c = sampling.candidateCount(n, smp);
    if (c > vals.len) return error.TooManyCandidates;
    return @intCast(try sampling.sampleCandidates(vals[0..c], ids[0..c], position, smp));
}

/// TF_DS_SAMPLE_CAND=1: Track B's candidates path for sampled rounds. Unset or any other value: the whole logits.
fn candOn() bool {
    const S = struct {
        var v: ?bool = null;
    };
    if (S.v) |x| return x;
    const x = if (std.c.getenv("TF_DS_SAMPLE_CAND")) |e| std.mem.eql(u8, std.mem.span(e), "1") else false;
    S.v = x;
    return x;
}

pub fn draw(row: []const f32, position: u64, s: ?lanes.Sampling) !u32 {
    const sm = s orelse return @intCast(sampling.argmax(row));
    if (sm.temperature <= 0) return @intCast(sampling.argmax(row));
    const t = try sampling.sampleRow(row, position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p });
    return @intCast(t);
}

/// A row's draw among a grammar's allowed tokens (token t: bit t % 32 of word t / 32, lanes/grammar.zig): the row
/// with every other token at -inf (written into `buf`, the row's width), drawn as `draw` draws any row.
pub fn drawMasked(buf: []f32, row: []const f32, position: u64, s: ?lanes.Sampling, mask: []const u32) !u32 {
    var allowed = false;
    for (buf, row, 0..) |*x, v, id| {
        const on = lanes.grammar.allows(mask, id);
        x.* = if (on) v else -std.math.inf(f32);
        allowed = allowed or on;
    }
    if (!allowed) return error.NoAllowedToken;
    return draw(buf, position, s);
}

/// A row's mask width in words: one bit a token.
pub fn maskWords(vocab: usize) usize {
    return (vocab + 31) / 32;
}

/// A window's masks: none, or each row's bits (rows x words).
fn masksFit(w: be.Window, words: usize) bool {
    return w.masks.len == 0 or w.masks.len == w.rows() * words;
}

/// Where rank 0's backend calls spend their time (ns, summed; --profile): a round's frame to rank 1, its forward
/// and absorb (synchronized apart), the logits' copy to the host and the draws; the drafter passes; the prefills.
pub const Profile = struct {
    send: u64 = 0,
    enqueue: u64 = 0,
    forward: u64 = 0,
    absorb: u64 = 0,
    logits: u64 = 0,
    sample: u64 = 0,
    pass: u64 = 0,
    prefill: u64 = 0,
    rounds: u64 = 0,
    rows: u64 = 0,
    passes: u64 = 0,
    pass_streams: u64 = 0,
    prefills: u64 = 0,
};

/// A round-cost table "30.0,35.2,..." (TF_DS_ROUND_MS's form): ms by rows from 1.
pub fn parseMs(a: std.mem.Allocator, text: []const u8) ![]const f64 {
    var out: std.ArrayList(f64) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |x| try out.append(a, try std.fmt.parseFloat(f64, std.mem.trim(u8, x, " ")));
    if (out.items.len == 0) return error.BadRoundMs;
    return out.items;
}

/// A stream's place in the pool and the drafts it holds.
const Lane = struct {
    slot: usize,
    base: usize,
    end: usize,
    first: u64 = 0, // the prompt's draw (a handle)
    held: [draft.max_block]u32 = undefined,
    confs: [draft.max_block]f32 = undefined,
    nheld: usize = 0,
};

/// Host-owned prompt progress; the KV rows and rings already live in its slot between chunks.
const Fill = struct {
    start: usize,
    replay: usize,
    last_end: usize,
    spans: std.ArrayList(@import("prompt.zig").Span),
    rows: std.ArrayList(u8),

    fn deinit(f: *Fill, gpa: std.mem.Allocator) void {
        f.spans.deinit(gpa);
        f.rows.deinit(gpa);
    }
};

extern "c" fn exp(x: f64) f64;

/// A kept prompt state (the prompt cache's saved state, kept at a chunk end the prompt pass reached before its
/// replay): the slot's own part on the device (Model.snapCopy), and the extent its compressed rows stay in,
/// [base, base + at), held back from placement while it lives. Stale: a placement needed that room (its restore fails
/// and the prompt runs from 0).
pub const Snap = struct {
    id: u64,
    at: usize,
    base: usize,
    buf: cuda.DeviceBuffer,
    stale: bool = false,
    wanted: u64 = 0, // when a request first waited for the live stream that holds these rows (placeKept; 0: none)
    end: usize = 0, // the extent end of the stream that kept it (placement holds [base, end) back while room allows)
};

/// Kept states this long or longer say so in the log when they go stale (their prompts then run again from 0).
const note_rows = 32 * model.chunk_rows;

fn noteStale(sn: *const Snap, why: []const u8) void {
    if (sn.at >= note_rows) std.log.info("dsv41: the kept state of {d} tokens at {d} goes stale: {s}", .{ sn.at, sn.base, why });
}

/// Why a kept state does not restore for a prompt (Lanes.refusal).
const Refusal = enum {
    dropped, // the prompt cache freed it
    stale, // a stream wrote over its rows
    short, // the prompt ends at or before it
    off_chunk, // it is not at a chunk end
    near_end, // the prompt ends within the window past it: its replay starts before the state

    fn says(r: Refusal) []const u8 {
        return switch (r) {
            .dropped => "it was dropped",
            .stale => "a stream wrote over its rows",
            .short => "the prompt ends at or before it",
            .off_chunk => "it is not at a chunk end",
            .near_end => "the prompt ends within the window past it",
        };
    }
};

/// Every kept state a prompt does not resume says why in the log, whatever its length (a dropped one is not read).
fn noteRefusal(sn: *const Snap, len: usize, why: Refusal, then: []const u8) void {
    if (why == .dropped) return std.log.info("dsv41: a prompt of {d} tokens resumes no kept state: {s}; {s}", .{ len, why.says(), then });
    std.log.info("dsv41: a prompt of {d} tokens does not resume the kept state of {d} tokens at {d}: {s}; {s}", .{ len, sn.at, sn.base, why.says(), then });
}

/// A prompt pass between its pieces: the chunk it runs next and what its chunk loop reads.
const Fill = struct { s: *const lanes.Stream, slot: usize, len: usize, replay: usize, last_end: usize, start: usize, head: bool = false };

/// A parked pass's image spans and rows (TF_FILL_MANY): set again when the pass resumes after another one's pieces.
const Images = struct { spans: std.ArrayList(@import("prompt.zig").Span) = .empty, rows: std.ArrayList(u8) = .empty };

/// How long a request waits for the live stream that holds the kept state it resumes from to end (then it resumes
/// in place and every kept state stays) before other kept states give their room for a copy of it.
const held_wait_ns: u64 = 10 * std.time.ns_per_s;

/// Below this position a prompt pass runs model.resume_rows rows: a 4096-row pass scores its own later keys in the
/// indexer, which costs more than it saves near the prompt's start (slot pps: the 8K row 2,798 tok/s from 8192 vs
/// 2,650 from 0; 128K 2,947 vs 2,949).
pub const pass_from_default = 8192;

/// First tokens a handle names (each prompt's draw, read back right after its prefill).
const ring = 1024;

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    peer: ?link.Link, // rank 1 (null: one rank); every follower gets each frame (Model.followers)
    vision: ?*@import("vision.zig").Vision = null, // rank 0's tower (a vision checkpoint served with its kit's vision/)
    step: u64 = 0,
    rdma_idle: bool = false,
    wire: Wire = .{},
    streams: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    wc: [window_rows]lanes.config.Cost = undefined, // the window costs (init)
    sc: [max_rows]lanes.config.Cost = undefined, // the shared forwards' costs (init)
    used: [max_streams]bool = @splat(false),
    seqs: [max_streams]std.ArrayList(i32) = @splat(.empty),
    fills: [max_streams]?Fill = @splat(null),
    filling_slot: ?usize = null,
    built: Built = .{},
    prof: ?Profile = null,
    fix_k: ?usize = null, // a measurement (tf-dsv41-lanes --fix-k): K drafts a stream, chances 1 to K and 0 past it
    // TF_DS_SERVED_K: a stream's chances end past the served lane's own choice (draft.chooseK: its round table and
    // draft ms, the other live streams at their table depth), so a round takes at most those drafts: at any number of
    // streams (all), or with one stream live (solo: the served one-stream choice; more keep the lane core's allocation)
    served_k: enum { off, all, solo } = .off,
    solo_overhead: bool = false, // TF_DS_SOLO_OVERHEAD=1: a lone drafted stream's head windows cost the fixed draft allowance
    no_copies: bool = false, // TF_DS_NO_COPIES=1: a drafted stream proposes no suffix-lookup copy windows
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    snaps: std.ArrayList(*Snap) = .empty, // kept prompt states, oldest first
    next_snap: u64 = 1,
    fills: [max_streams]?Fill = @splat(null), // the prompt passes that run in pieces between rounds (by slot), each from its next chunk
    cur: ?usize = null, // the slot whose pass the model's fill state (fillBegin's view, images, read-ahead) holds
    // TF_FILL_MANY=1 (with TF_FILL_CHUNKS): a request is admitted while a pass fills, its pass's pieces between the
    // other's (the host gives the next piece to the fewest prompt tokens left); a switch sets the model's fill state
    // again (fill_switch). Unset: one pass at a time, as served
    fill_many: bool = false,
    imgs: [max_streams]Images = @splat(.{}),
    // TF_FILL_CHUNKS: a prompt pass runs this many chunks a step of the lane core's (prefill_step, one pass at a time),
    // the other streams' rounds between its steps; its first step TF_FILL_FIRST chunks (0: as many), so a prompt that
    // short runs whole. A step runs on past them while nothing else waits (the stream's share_check). 0: each pass
    // whole, as served
    fill_chunks: usize = 0,
    fill_first: usize = 0,
    // a prompt pass's rows at most: model.resume_rows below TF_DS_PASS_FROM positions (default pass_from_default),
    // then TF_DS_PASS_ROWS (a multiple of model.resume_rows up to model.chunk_rows; unset: model.chunk_rows). The same
    // bits at any: a pass ends on the resume grid either way
    pass_rows: usize = model.chunk_rows,
    pass_from: usize = pass_from_default,
    // structured output: the prompt's last row of the last stream with a grammar prefilled (its first token may be
    // drawn again among the grammar's tokens), and a row with the disallowed tokens at -inf
    first_row: []f32 = &.{},
    first_of: ?*const lanes.Stream = null,
    masked: []f32 = &.{},

    pub fn init(gpa: std.mem.Allocator, m: *Model) Lanes {
        var ln: Lanes = .{ .gpa = gpa, .m = m, .peer = if (m.world > 1) m.peer() else null };
        // the round costs by rows: ROUND_MS[min(rows, len) - 1] (the served table, or TF_DS_ROUND_MS)
        const t = m.round_ms;
        for (&ln.wc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        for (&ln.sc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        return ln;
    }

    /// Tells rank 1 to stop, then frees the host state (the model is the caller's).
        self.m.unwatch(); // (the followers end on done: their links close)
    pub fn deinit(self: *Lanes) void {
        for (self.snaps.items) |sn| {
            sn.buf.free();
            self.gpa.destroy(sn);
        }
        self.snaps.deinit(self.gpa);
        if (self.peer != null) {
            self.m.unwatch(); // (the followers end on done: their links close)
            self.send(.done) catch {};
        }
        for (&self.fills) |*f| if (f.*) |*x| x.deinit(self.gpa);
        self.streams.deinit(self.gpa);
        for (&self.seqs) |*q| q.deinit(self.gpa);
        self.wire.deinit(self.gpa);
        self.gpa.free(self.first_row);
        self.gpa.free(self.masked);
        for (&self.imgs) |*im| {
            im.spans.deinit(self.gpa);
            im.rows.deinit(self.gpa);
        }
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = if (self.fill_chunks == 0) &whole_vt else if (self.fill_many) &many_vt else &steps_vt };
    }

    const whole_vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .probabilities = probabilitiesFn,
        .first_masked = firstMaskedFn,
        .held = heldFn,
        .release = releaseFn,
    };
    // a pass in steps, one at a time: `fill` and the followers' read-ahead hold one pass
    const steps_vt: be.Backend.VTable = vt: {
        var v = whole_vt;
        v.prefill_step = prefillStepFn;
        v.prefill_one = true;
        break :vt v;
    };
    // TF_FILL_MANY: passes in steps, several at a time (`fills`, a switch between them sets the model's fill state)
    const many_vt: be.Backend.VTable = vt: {
        var v = whole_vt;
        v.prefill_step = prefillStepFn;
        break :vt v;
    };

    /// The pass `s` fills, if one runs.
    fn fillOf(self: *Lanes, s: *const lanes.Stream) ?*Fill {
        for (&self.fills) |*f| if (f.*) |*x| if (x.s == s) return x;
        return null;
    }

    fn anyFill(self: *const Lanes) bool {
        for (self.fills) |f| if (f != null) return true;
        return false;
    }

    /// Slot `slot`'s pass ends (done, failed or released): its parked images go.
    fn dropFill(self: *Lanes, slot: usize) void {
        self.fills[slot] = null;
        if (self.cur == slot) self.cur = null;
        self.imgs[slot].spans.clearRetainingCapacity();
        self.imgs[slot].rows.clearRetainingCapacity();
    }

    /// The model's fill state set to `f`'s pass again after another pass's pieces (TF_FILL_MANY), on every rank:
    /// fillBegin's view, window rings and slot (its Engram read-ahead dropped: rows read for another prompt are not
    /// this one's), then its images.
    fn fillSwitch(self: *Lanes, f: *const Fill) !void {
        const m = self.m;
        const base = (self.streams.get(f.s) orelse return error.UnknownStream).base;
        m.joinAhead();
        m.fill_seq = &.{};
        if (try self.begin(.fill_switch)) {
            try self.wire.int(self.gpa, u32, @intCast(f.slot));
            try self.wire.int(self.gpa, u64, base);
            try self.flush();
        }
        try m.fillBegin(f.slot, base);
        const im = &self.imgs[f.slot];
        if (im.spans.items.len > 0) {
            try self.sendImages(im.spans.items, im.rows.items);
            try m.setImages(im.spans.items, im.rows.items);
        }
        self.cur = f.slot;
    }

    /// A pass's image spans and their rows to rank 1, image_piece bytes of rows a frame (a follower's frame buffer
    /// holds 64 KiB and the ids of a full pool).
    fn sendImages(self: *Lanes, spans: []const @import("prompt.zig").Span, rows: []const u8) !void {
        var off: usize = 0;
        while (off < rows.len) {
            const n = @min(image_piece, rows.len - off);
            if (try self.begin(.fill_images)) {
                try self.wire.int(self.gpa, u32, @intCast(spans.len));
                for (spans) |sp| {
                    try self.wire.int(self.gpa, u64, sp.at);
                    try self.wire.int(self.gpa, u64, sp.len);
                }
                try self.wire.int(self.gpa, u64, off);
                try self.wire.buf.appendSlice(self.gpa, rows[off..][0..n]);
                try self.flush();
            }
            off += n;
        }
    }

    /// The facts the round loop reads: windows of the pending row and up to the drafter's block, several streams'
    /// windows in one forward of at most 16 rows (four streams), the drafter's chances, the served lane's round ms.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.m.drafting();
        return .{
            .exact_width = window_rows,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = if (self.m.dr) |dr| @intCast(dr.n) else 1,
            .window_costs = self.wc[0..],
            .mtp_step_ms = draft.draft_ms,
            .solo_head_overhead_ms = if (drafting and self.solo_overhead) draft.draft_ms else null, // opt-in: the Python engine's fixed draft allowance
            .copy_proposals = !(drafting and self.no_copies), // opt-in: the draft head proposes; the Python engine makes no copies either
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = @intCast(self.m.eng.round_rows),
            .max_streams = @intCast(self.m.streams),
            .shared_costs = self.sc[0..self.m.eng.round_rows],
            .draft_probabilities = drafting,
            .draft_streams = drafting,
        };
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The frame being built goes to rank 1 (after `begin` and the fields), or to ranks 1-3 on the four-node split.
    fn flush(self: *Lanes) !void {
        if (self.peer == null) return;
        for (self.m.followers()) |l| try l.?.send(self.wire.buf.items);
    }

    fn begin(self: *Lanes, kind: Kind) !bool {
        if (usesRdma(kind) and self.rdma_idle) {
            setRdmaIdle(self.m, false);
            self.rdma_idle = false;
        }
        where.frame('S', @backingInt(kind), self.step);
        try self.wire.begin(self.gpa, kind, self.step);
        self.step += 1;
        return true;
    }

    fn send(self: *Lanes, kind: Kind) !void {
        if (try self.begin(kind)) try self.flush();
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// The lowest free slot, the first extent that fits, the prompt in chunks of the served size (at multiples of it:
    /// the chunks the gates checked; the host's chunk plan is not followed), the head's draw of the first token.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        if (!try of(ptr).prefillPart(s, std.math.maxInt(usize))) return error.FillUnfinished;
    }

    /// Two-phase world-wide idle boundary. No rank parks while another may
    /// still need it as an RDMA relay. All streams/CUDA graphs finish first.
    pub fn idleRdma(self: *Lanes) !void {
        if (self.rdma_idle or self.streams.count() != 0 or !hasRdma(self.m)) return;
        self.m.joinAhead();
        const held = where.enter(.idle_ack, 0, 0);
        defer where.leave(held);
        try self.m.ctx.synchronize();
        try self.send(.rdma_idle);
        var ack: [64]u8 = undefined;
        for (self.m.followers()) |l| {
            if (!std.mem.eql(u8, try l.?.recv(&ack), idle_ready)) return error.BadIdleAck;
        }
        // Every GPU is quiescent now; only then permit any proxy to park.
        for (self.m.followers()) |l| try l.?.send(idle_commit);
        setRdmaIdle(self.m, true);
        self.rdma_idle = true;
    }

    /// A prompt pass's rows at most when it starts at `start`.
    pub fn passCap(self: *const Lanes, start: usize) usize {
        return if (start < self.pass_from) model.resume_rows else self.pass_rows;
        while (!try prefillStep(ptr, s)) {
        }
    }

    fn fillImages(self: *Lanes, spans: []const @import("prompt.zig").Span, rows: []const u8) !void {
        const gpa = self.gpa;
        const m = self.m;
        if (spans.len > 0) {
            // the spans and their rows, image_piece bytes of rows a frame (a follower's frame buffer holds 64 KiB and
            // the ids of a full pool)
            var off: usize = 0;
            while (off < rows.len) {
                const n = @min(image_piece, rows.len - off);
                if (try self.begin(.fill_images)) {
                    try self.wire.int(gpa, u32, @intCast(spans.len));
                    for (spans) |sp| {
                        try self.wire.int(gpa, u64, sp.at);
                        try self.wire.int(gpa, u64, sp.len);
                    }
                    try self.wire.int(gpa, u64, off);
                    try self.wire.buf.appendSlice(gpa, rows[off..][0..n]);
                    try self.flush();
                }
                off += n;
            }
            try m.setImages(spans, rows);
        }
        return true;
    }

    /// Placement, image encoding and a cached restore happen once, before this prompt's first chunk.
    fn startFill(self: *Lanes, s: *lanes.Stream) !void {
        const m = self.m;
        const gpa = self.gpa;
        m.joinAhead();
        m.fill_seq = &.{};
        self.filling_slot = null; // a failed setup must not leave another prompt's slot falsely selected
        const ids = s.prompt();
        const len = ids.len;
        if (len == 0) return error.EmptyPrompt;
        const size = std.mem.alignForward(usize, len + s.max_new + m.eng.round_rows + 2, model.chunk_rows);
        if (size > m.ctx_cap) return error.PromptTooLong;
        releaseFn(self, s);
        const slot = std.mem.indexOfScalar(bool, self.used[0..self.m.streams], false) orelse return error.NoFreeSlot;
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        // a kept state of this prompt's prefix: the pass resumes at its chunk end, in its extent, or in one of its own
        // with the kept rows copied there when a live stream holds that extent (a burst's second request on a prompt)
        s.cached = 0;
        var kept: ?*Snap = null;
        var moved = false;
        if (s.reuse.saved) |p| {
            const sn: *Snap = @ptrCast(@alignCast(p));
            // (a prompt with images resumes a kept state too: the cache keys the image positions by the picture's content, so the
            // state's rows are this prompt's own; the pictures before the cut need no rows again, below)
            if (self.usable(sn, len)) {
                kept = sn;
                // in place unless the extent runs past the pool, a live stream holds it, or another kept state has
                // rows past the kept ones (resuming there would stale it: the rows go to a free extent instead, as
                // the Python lane's _place resumes "here" only in a free extent)
                moved = sn.base + size > m.pool_cap or clashes(taken[0..nt], sn.base, sn.base + size) or self.keptOver(sn.base + sn.at, sn.base + size, sn);
            }
        }
        const base = if (kept != null and !moved) kept.?.base else try self.placeKept(taken[0..nt], size, kept);
        if (kept) |sn| {
            sn.wanted = 0;
            if (!moved) sn.end = @max(sn.end, base + size); // its extent now: the next resume in place finds it held
        }
        if (kept) |sn| if (moved and sn.stale) { // the placement needed the kept rows' room: the prompt runs from 0
            if (sn.at >= note_rows) std.log.info("dsv41: the kept state of {d} tokens at {d} found no room to move to ({d} positions); the prompt runs from 0", .{ sn.at, sn.base, size });
            kept = null;
            s.reuse_failed = true;
        };
        // the rows this stream writes are no other kept state's
        self.staleOver(if (kept != null and !moved) base + kept.?.at else base, base + size, kept);
        try self.streams.put(gpa, s, .{ .slot = slot, .base = base, .end = base + size });
        self.used[slot] = true;
        errdefer {
            _ = self.streams.remove(s);
            self.used[slot] = false;
            self.idleRdma() catch |err| std.debug.panic("RDMA idle after prefill error: {s}", .{@errorName(err)});
        }
        const seq = &self.seqs[slot];
        try seq.resize(gpa, len);
        for (seq.items, ids) |*q, id| q.* = @intCast(id);
        // images: each span's positions dead in the sequence (negative: no Engram n-gram reaches into or across one),
        // its rows ready (a gate's) or the tower's (rank 0), all of them sent to rank 1 after the fill's start; an image
        // the kept state holds whole needs no rows (the pass resumes past it)
        var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
        errdefer spans.deinit(gpa);
        var rows: std.ArrayList(u8) = .empty;
        errdefer rows.deinit(gpa);
        if (s.lane_images.len > 0) {
            const image_id = m.cfg.image_token orelse return error.NoVision;
            const row_bytes = m.cfg.hidden * 2;
            const resume_at: usize = if (kept) |sn| sn.at else 0;
            var at_row: usize = 0;
            // a resumed pass starts at the kept state's end: a picture wholly before it is already in the state (its positions stay
            // dead in the sequence for Engram's n-grams, but it needs no rows and no run of the tower)
            for (s.lane_images) |img| {
                if (img.tokens == 0 or @as(usize, img.at) + img.tokens > len) return error.BadImageSpan;
                for (seq.items[img.at..][0..img.tokens]) |*q| {
                    if (q.* != @as(i64, image_id)) return error.BadImageSpan;
                    q.* = -1;
                }
                // The keyed snapshot already contains these image rows; only a span crossing its cut needs encoding.
                if (kept) |sn| if (@as(usize, img.at) + img.tokens <= sn.at) continue;
                if (img.rows) {
                    if (img.bytes.len != img.tokens * row_bytes) return error.BadImageRows;
                    try rows.appendSlice(gpa, img.bytes);
                } else {
                    // a prepared picture (vision.zig): the tower's span rows, here on rank 0
                    const v = self.vision orelse return error.NoVisionTower;
                    if (try v.rows(img.bytes, gpa, &rows) != img.tokens) return error.BadImageSpan;
                }
                try spans.append(gpa, .{ .at = img.at, .len = img.tokens, .row = at_row });
                at_row += img.tokens;
            }
        }
        if (try self.begin(.fill_begin)) {
            try self.wire.int(gpa, u32, @intCast(slot));
            try self.wire.int(gpa, u64, base);
            try self.wire.ids(gpa, seq.items);
            try self.flush();
        }
        try m.fillBegin(slot, base);
        try self.fillImages(spans.items, rows.items);
        const replay = len -| m.cfg.window;
        var start: usize = 0;
        if (kept) |sn| {
            if (try self.begin(.snap_restore)) {
                try self.wire.int(gpa, u64, sn.id);
                try self.wire.int(gpa, u32, @intCast(slot));
                try self.wire.int(gpa, u64, sn.base);
                try self.wire.int(gpa, u64, base);
                try self.wire.int(gpa, u64, sn.at);
                try self.flush();
            }
            if (base != sn.base) try m.copyExtent(sn.base, base, sn.at);
            try m.snapCopy(slot, sn.buf.ptr, false);
            start = sn.at;
            s.cached = @intCast(sn.at);
        }
        // where this pass keeps states: the cache's marks and its last resume point before the replay (the served lane's
        // kept prompt), each at the resume point at or before it
        const last_end = @min(replay, len - 1) / model.resume_rows * model.resume_rows;
        self.fills[slot] = .{ .s = s, .slot = slot, .len = len, .replay = replay, .last_end = last_end, .start = start };
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        return l.first;
        if (s.progress) |p| {
            p.begin(start);
            // the rows count once the device has run them, not once the host has issued them (a pair returns ~0.9 s early)
            if (m.meterArm(slot)) |g| p.gated(g.cell, g.gen);
        }
        self.fills[slot] = .{
            .start = start,
            .replay = replay,
            .last_end = @min(replay, len - 1) / model.chunk_rows * model.chunk_rows,
            .spans = spans,
            .rows = rows,
        };
        self.filling_slot = slot;
    }

    /// Select a partially filled slot on every rank, without resending its full prompt or re-encoding its images.
    fn selectFill(self: *Lanes, slot: usize, base: usize) !void {
        const m = self.m;
        if (self.filling_slot != slot) {
            self.filling_slot = null;
            m.joinAhead();
            m.fill_seq = &.{};
            if (try self.begin(.fill_resume)) {
                try self.wire.int(self.gpa, u32, @intCast(slot));
                try self.wire.int(self.gpa, u64, base);
                try self.flush();
            }
            try m.fillBegin(slot, base);
            const f = &self.fills[slot].?;
            try self.fillImages(f.spans.items, f.rows.items);
            self.filling_slot = slot;
        }
        m.fill_seq = self.seqs[slot].items;
    }

    /// One unchanged 2048-row chunk. Other prompts and decode rounds can run before its next chunk.
    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        const t0 = m.now();
        defer if (self.prof) |*p| {
            p.prefill += m.now() - t0;
        };
        errdefer {
            m.joinAhead();
            m.fill_seq = &.{};
        }
        if (s.isCancelled()) return error.Cancelled;
        if (!self.streams.contains(s)) try self.startFill(s);
        const lane = self.streams.get(s).?;
        try self.selectFill(lane.slot, lane.base);
        const f = &self.fills[lane.slot].?;
        const seq = &self.seqs[lane.slot];
        const len = s.prompt_len;
        // the pass (rank 0's rule; the followers run the frame): the rows the host's facts allow, whole resume units up
        // to them (the final pass may take the remainder with TF_DS_PASS_TAIL), cut at the first state it keeps
        const hf = s.hostFacts();
        const decoding_now = if (hf) |x| x.decoding else false;
        const cap = lanes.stream.passRows(hf, f.start, prompt2d.pass_from_asked, @intCast(m.pass_rows), model.chunk_rows, prompt2d.pass_filling_asked);
        const n = passRowsAt(f.start, len, cap, prompt2d.pass_tail_asked, s.reuse.marks, f.last_end, f.replay);
        // TF_DS_ROCI_STAGGER: this pass and the next one as a pair (rank 0's rule; the followers run the frame). The pair does not end
        // its first pass where the prompt cache keeps a state or a hook runs, so every such stop is a stop of a pair.
        var n2: usize = 0;
        var n3: usize = 0; // the third pass of a group (TF_DS_ROCI_STAGGER_LANES=3), else 0
        // A whole pass is one the host's facts leave at the booted rows (stream.wholePass: at pass rows of 2,048 a cut pass has the same rows).
            const mid = f.start + n;
            const n_b = passRowsAt(mid, len, cap, prompt2d.pass_tail_asked, s.reuse.marks, f.last_end, f.replay);
            const stop_between = mid <= f.replay and (mid == f.last_end or markEnds(s.reuse.marks, mid));
            if (!stop_between and m.pairEligible(seq.items[0 .. mid + n_b], f.start, n, mid, n_b, f.replay, cap)) n2 = n_b;
            // TF_DS_ROCI_STAGGER_LANES=3: a third whole pass behind the pair, where nothing decodes (the host's rounds owed after a pair are counted for two
            // passes: beside generating users the pair stays a pair); the group falls back to the pair where the prompt, a stop or the rule leaves no third pass
            if (n2 != 0) {
                if (thirdPassRows(m.pair_lanes, decoding_now, rounds_now, mid, n2, len, cap, prompt2d.pass_tail_asked, s.reuse.marks, f.last_end, f.replay)) |n_c| {
                    if (m.groupEligible(seq.items[0 .. mid + n2 + n_c], f.start, &.{ n, n2, n_c }, f.replay, cap)) n3 = n_c;
                }
            }
        }
        if (n2 != 0) {
            if (try self.begin(.fill_pair)) {
                try self.wire.int(gpa, u64, f.start);
                try self.wire.int(gpa, u32, @intCast(n));
                try self.wire.int(gpa, u32, @intCast(n2));
                try self.wire.int(gpa, u32, @intCast(n3));
                try self.wire.int(gpa, u64, f.replay);
                try self.flush();
            }
        } else if (try self.begin(.fill_chunk)) {
            try self.wire.int(gpa, u64, f.start);
            try self.wire.int(gpa, u32, @intCast(n));
            try self.wire.int(gpa, u64, f.replay);
            try self.flush();
        }
        if (self.spill) |t| t.bump(lane.base + f.start, lane.base + f.start + n + n2 + n3);
        const head = if (n2 != 0) blk: {
            try m.fillGroup(seq.items[0 .. f.start + n + n2 + n3], f.start, if (n3 != 0) &.{ n, n2, n3 } else &.{ n, n2 }, f.replay);
            break :blk false;
        } else try m.fillChunk(seq.items[0 .. f.start + n], f.start, n, f.replay);
        f.start += n + n2 + n3;
        if (s.progress) |p| {
            if (n3 != 0) try m.meterGroup(lane.slot, &.{ f.start - n3 - n2, f.start - n3, f.start }) else if (n2 != 0) try m.meterPair(lane.slot, f.start - n2, f.start) else try m.meterChunk(lane.slot, f.start);
            p.reach(f.start);
        }
        // the host bills this step's time to decoders (TENSORFOLD_PREFILL_SHARE below 1, a stream decoding): fillChunk
        // never waits, so the last layers still run on the device here, and their tail would land in the next round's time
        if (s.settle_step) {
            try m.pairJoin();
            try m.stream.synchronize();
        }
        if (s.reuse.hook) |k| if (f.start < len and f.start <= f.replay and
            (f.start == f.last_end or markEnds(s.reuse.marks, f.start))) k.at(k.ptr, s, @intCast(f.start));
        if (f.start < len) {
                m.fill_seq = &.{};
            return false;
        }
        m.joinAhead();
        m.fill_seq = &.{};
        if (!head) return error.NoPromptLogits;
        // a stream with a grammar: the row stays for its first token's draw among the grammar's tokens
        if (s.grammar != null) {
            self.first_row = try self.gpa.realloc(self.first_row, logits.len);
            @memcpy(self.first_row, logits);
            self.first_of = s;
        }
        f.deinit(gpa);
        self.fills[lane.slot] = null;
        if (self.prof) |*p| p.prefills += 1;
        return true;
    }

    /// `first` among a grammar's tokens: the prompt's last row (the prompt pass kept it for the stream) drawn again
    /// with the other tokens at -inf, at the same keyed position.
    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const self = of(ptr);
        _ = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        if (self.first_of != s) return error.NoPromptRow;
        return self.take(try self.drawAllowed(self.first_row, position, s.sampling, mask));
    }

    /// The held drafts as host tokens in window order (a grammar walks them): draft holds them when its pass ends.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (out.len > l.nheld) return error.HeldMismatch;
        @memcpy(out, l.held[0..out.len]);
    }

    /// drawMasked into the backend's row.
    fn drawAllowed(self: *Lanes, row: []const f32, position: u64, s: ?lanes.Sampling, mask: []const u32) !u32 {
        self.masked = try self.gpa.realloc(self.masked, row.len);
        return drawMasked(self.masked, row, position, s, mask);
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = .{ ptr, s, feed, position };
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Every window as one round at its stream's cache length: the pending token, the held drafts (or the host's), each
    /// row drawn at its keyed position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (windows.len > max_streams) return error.WindowTooWide;
        var ids: [max_streams][max_rows]i32 = undefined;
        var wins: [max_streams]Win = undefined;
        const words = maskWords(m.vocab);
        var masked = false;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = self.streams.getPtr(w.stream) orelse return error.UnknownStream;
            const rows = w.rows();
            if (rows > window_rows) return error.WindowTooWide;
            const start: usize = @intCast(w.stream.cache_len);
            for (w.positions, 0..) |p, r| if (p != start + 1 + r) return error.PositionMismatch;
            if (w.held > l.nheld) return error.HeldMismatch;
            if (!masksFit(w, words)) return error.BadMasks;
            masked = masked or w.masks.len > 0;
            ids[k][0] = @intCast(w.pending);
            for (l.held[0..w.held], 0..) |t, j| ids[k][1 + j] = @intCast(t);
            for (w.tokens, 0..) |t, j| ids[k][1 + w.held + j] = @intCast(t);
            wins[k] = .{ .slot = l.slot, .base = l.base, .end = l.end, .start = start, .absorb = m.drafting() and w.stream.drafts, .ids = ids[k][0..rows] };
        }
        // every row greedy and none masked: the device's argmax (R tokens back); else the logits for the host's draws
        var greedy = !masked;
        for (windows) |w| {
            if (w.stream.sampling) |sm| if (sm.temperature > 0) {
                greedy = false;
            };
        }
        // Track B (TF_DS_SAMPLE_CAND=1): a sampled round's rows as their first candidates from the GPU (the host's
        // draws read only those; sampling.sampleCandidates), the whole logits only when a row needs them
        var cand_count: usize = 0;
        if (!greedy and !masked and candOn()) { // a masked row is drawn from its whole row
            cand_count = 1;
            for (windows) |w| if (w.stream.sampling) |sm| if (sm.temperature > 0) {
                if (sm.top_k == 0 or sampling.candidateCount(m.vocab, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p }) > sampling.max_candidates) {
                    cand_count = 0;
                    break;
                }
                cand_count = @max(cand_count, sampling.candidateCount(m.vocab, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p }));
            };
        }
        const t0 = m.now();
        where.set(.round, @intCast(windows.len), 0);
        defer where.set(.after_backend, 0, 0); // (the lane core's own work until the next backend call or the gate's next mark)
        defer m.flushDiag(false); // (TF_DS_ROCI_XLOG / TF_DS_NACT_LOG's files, every 256th call: nothing when both are off)
        if (try self.begin(.verify)) {
            try self.wire.int(gpa, u32, @intCast(windows.len));
            for (wins[0..windows.len]) |w| {
                try self.wire.int(gpa, u32, @intCast(w.slot));
                try self.wire.int(gpa, u64, w.base);
                try self.wire.int(gpa, u64, w.end);
                try self.wire.int(gpa, u64, w.start);
                try self.wire.int(gpa, u8, @intFromBool(w.absorb));
                try self.wire.ids(gpa, w.ids);
            }
            try self.flush();
        }
        const t1 = m.now();
        const rows = try self.built.build(gpa, &self.seqs, wins[0..windows.len]);
        try m.verify(rows, self.built.absorb[0..windows.len]);
        const t2 = m.now();
        // every row greedy and none masked: the device's argmax (R tokens back); else the logits for the host's draws
        var greedy = !masked;
        for (windows) |w| {
            if (w.stream.sampling) |sm| if (sm.temperature > 0) {
                greedy = false;
            };
        }
        const tokens: []const u32 = if (greedy) try m.roundArgmax(rows.ids.len) else &.{};
        const logits: []const f32 = if (greedy) &.{} else try m.roundLogits(rows.ids.len);
        const t3 = m.now();
        var whole = cand == null;
        if (cand) |cd| for (cd.flags) |fl| {
            whole = whole or fl != 0;
        };
        defer if (self.prof) |*p| {
            p.send += t1 - t0;
            p.enqueue += m.t_enqueue;
            p.forward += m.t_forward;
            p.absorb += m.t_absorb;
            p.logits += t3 - t2;
            p.sample += m.now() - t3;
            p.rounds += 1;
            p.rows += rows.ids.len;
        };
        const V = m.vocab;
        for (windows, out, 0..) |w, *o, k| {
            const row0 = self.built.wins[k].row;
            for (0..w.rows()) |r| o.sampled[r] = if (greedy)
                tokens[row0 + r]
            else if (w.masks.len > 0)
                try self.drawAllowed(logits[(row0 + r) * V ..][0..V], w.positions[r], w.stream.sampling, w.masks[r * words ..][0..words])
            else if (!whole)
                try drawCandidates(cand.?.vals[(row0 + r) * cand_count ..][0..cand_count], cand.?.ids[(row0 + r) * cand_count ..][0..cand_count], V, w.positions[r], w.stream.sampling)
            else
                try draw(logits[(row0 + r) * V ..][0..V], w.positions[r], w.stream.sampling);
            const l = self.streams.getPtr(w.stream).?;
            @memcpy(o.drafts[0..w.held], l.held[0..w.held]);
            @memcpy(o.drafts[w.held..][0..w.tokens.len], w.tokens);
            l.nheld = 0;
        }
    }

    /// The caches are positional: the next window starts at the stream's new length and writes over the rows past it.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = ptr;
        for (windows, paths) |_, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        }
    }

    /// One drafter pass for every stream asked: its pending token at its new length, `depth` drafts held.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (m.dr == null) return error.NoDraftHead;
        where.set(.draft, @intCast(requests.len), 0);
        defer where.set(.after_backend, 0, 0);
        var depths: [max_streams]usize = undefined;
        var asked: [max_streams]*Lane = undefined;
        var n: usize = 0;
        var steps: usize = 0;
        for (requests) |r| {
            if (r.lanes != null or r.early) return error.TreesNotBuilt;
            const l = self.streams.getPtr(r.stream) orelse return error.UnknownStream;
            l.nheld = 0;
            if (r.depth == 0) continue;
            if (n == max_streams) return error.TooManyStreams;
            if (r.position != r.stream.cache_len + 1) return error.PositionMismatch;
            const token: u32 = if (r.rows == null) self.value(r.first orelse return error.NoFirstToken) else (if (r.follow.len > 0) r.follow[r.follow.len - 1] else return error.NoFollowToken);
            tokens[n] = token;
            q0[n] = @intCast(r.position - 1);
            slots[n] = @intCast(l.slot);
            depths[n] = @min(r.depth, block);
            asked[n] = l;
            steps = @max(steps, depths[n]);
            n += 1;
        }
        if (n == 0) return;
        // a pass runs block rows a stream through the decode window's linears (prompt.grouped: 128 rows at most), so
        // more streams than 128 / block (25 at the 5-row block) draft in passes of that many, each its own wire message
        const per = @max(1, pass_rows / block);
        // a pass runs block rows a stream through the decode window's linears (prompt.grouped: 128 rows at most), so more
        // (TensorFold ba03026c; the followers are unchanged)
        var g0: usize = 0;
        while (g0 < n) : (g0 += per) {
            const g1 = @min(n, g0 + per);
            if (try self.begin(.pass)) {
                try self.wire.int(gpa, u32, @intCast(g1 - g0));
                try self.wire.int(gpa, u32, @intCast(steps));
                for (g0..g1) |i| {
                    try self.wire.int(gpa, i64, tokens[i]);
                    try self.wire.int(gpa, i64, q0[i]);
                    try self.wire.int(gpa, i64, slots[i]);
                }
                try self.flush();
            }
            const t0 = m.now();
            try m.pass(tokens[g0..g1], q0[g0..g1], slots[g0..g1], steps);
            if (self.prof) |*p| {
                p.pass += m.now() - t0;
                p.passes += 1;
                p.pass_streams += g1 - g0;
            }
            const d = &m.dr.?;
            for (asked[g0..g1], depths[g0..g1], 0..) |l, k, i| {
                for (0..k) |j| {
                    l.held[j] = @intCast(d.drafts[i][j]);
                    l.confs[j] = d.confs[i][j];
                }
                l.nheld = k;
            }
        }
    }

    /// The chance that each held draft lands with every earlier one, as the round loop's allocation reads it: the
    /// running product of the confidences' sigmoids (multi.py's conf rule: a sigmoid is the draft's survival once
    /// the earlier drafts survived, and the prefix survivals' sum is the expected kept drafts; draft.chooseK).
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return false;
        if (out.len > l.nheld) return false;
        if (self.fix_k) |k| {
            for (out, 0..) |*o, j| o.* = if (j < k) 1.0 else 0.0;
            return true;
        }
        survivals(l.confs[0..out.len], out);
        const live = @max(1, self.streams.count());
        if (self.served_k == .all or (self.served_k == .solo and live == 1)) {
            const drafts: usize = if (self.m.dr) |dr| dr.n else 1;
            const kmax = @min(out.len, draft.depth(live, drafts, self.m.eng.round_rows));
            const k = draft.chooseKWith(l.confs[0..out.len], kmax, live, drafts, self.m.eng.round_rows, self.m.round_ms, draft.cost_ms);
            for (out[k..]) |*o| o.* = 0.0;
        }
        return true;
    }

    /// Why a kept state does not restore for a prompt of `len` tokens (null: it does): it restores while live, not
    /// stale, at a chunk end before the prompt's replay (the rows before it ran the encoder layers alone, as a fresh
    /// pass of this prompt runs them).
    fn refusal(self: *const Lanes, sn: *const Snap, len: usize) ?Refusal {
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, @constCast(sn)) == null) return .dropped;
        if (sn.stale) return .stale;
        if (sn.at == 0 or sn.at >= len) return .short;
        if (sn.at % model.resume_rows != 0) return .off_chunk;
        if (sn.at > len -| self.m.cfg.window) return .near_end;
        return null;
    }

    /// The first extent that fits around the live streams and the kept states. Kept states give their room oldest
    /// first and `source` (the state this prompt resumes from, its rows copied here) last, as the Python lane's _place
    /// takes the oldest other kept prompt before the source; a state whose rows lie in a live stream's extent gives
    /// none (that stream holds the room) and stays. When a live stream holds `source`, the request first waits up to
    /// held_wait_ns for that stream to end (error.ContextFull: the host tries it again; it then resumes in place) if
    /// the copy needs another state's room or would leave none for another request of its size: a copy that fills the
    /// pool makes the next prompt evict a kept state (the 5 x 500K check with the duplicate admitted fourth). Kept
    /// states hold their stream's whole extent back while a placement still fits so (placeSoft): its tail stays free
    /// for the resume in place, which then stales nothing.
    fn placeKept(self: *Lanes, live: []const [2]usize, size: usize, source: ?*Snap) !usize {
        const gpa = self.gpa;
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(gpa);
        var rows: std.ArrayList([2]usize) = .empty;
        defer rows.deinit(gpa);
        const held = if (source) |sn| !sn.stale and clashes(live, sn.base, sn.base + sn.at) else false;
        while (true) {
            kept.clearRetainingCapacity();
            rows.clearRetainingCapacity();
            try rows.appendSlice(gpa, live);
            for (self.snaps.items) |sn| if (!sn.stale) {
                try kept.append(gpa, .{ sn.base, sn.at, sn.end });
                try rows.append(gpa, .{ sn.base, sn.base + sn.at });
            };
            const got = try placeSoft(gpa, live, kept.items, size, self.m.pool_cap);
            if (held and self.patient(source.?)) {
                const b = got orelse return error.ContextFull;
                return if (try roomAfter(gpa, rows.items, b, size, self.m.pool_cap)) b else error.ContextFull;
            }
            if (got) |b| return b;
            const victim = for (self.snaps.items) |sn| {
                if (!sn.stale and sn != source and !clashes(live, sn.base, sn.base + sn.at)) break sn;
            } else if (source) |sn| (if (!sn.stale and !clashes(live, sn.base, sn.base + sn.at)) sn else return error.ContextFull) else return error.ContextFull;
            noteStale(victim, "a placement needed its room (oldest first)");
            victim.stale = true;
        }
    }

    /// Whether a kept state other than `except` and its shorter siblings has rows in [lo, hi); a longer one may be another conversation's.
    fn keptOver(self: *const Lanes, lo: usize, hi: usize, except: *const Snap) bool {
        for (self.snaps.items) |sn| if (sn != except and !sn.stale and (sn.base != except.base or sn.at > except.at) and sn.base < hi and lo < sn.base + sn.at) return true;
        return false;
    }

    /// Whether a request still waits for the live stream that holds `sn` (held_wait_ns from its first wait).
    fn patient(self: *const Lanes, sn: *Snap) bool {
        const t = self.m.now();
        if (sn.wanted == 0) sn.wanted = t;
        return t - sn.wanted < held_wait_ns;
    }

    /// Kept states with rows in [lo, hi) go stale (a stream writes there), `except` aside.
    fn staleOver(self: *Lanes, lo: usize, hi: usize, except: ?*Snap) void {
        for (self.snaps.items) |sn| if (sn != except and sn.base < hi and lo < sn.base + sn.at) {
            if (!sn.stale) noteStale(sn, "a stream writes over its rows");
            sn.stale = true;
        };
    }

    /// The prompt cache's Snapshots functions (native.zig hands them to the CUDA host).
    pub fn snapBytesFn(ptr: *anyopaque, at: u32) u64 {
        _ = at;
        return of(ptr).m.snapBytes();
    }

    /// The live state at `at` (the pass stands at that chunk end): the slot's own part copied, the extent's rows kept.
    pub fn snapSaveFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = of(ptr);
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (at == 0 or at % model.resume_rows != 0 or at > s.prompt_len -| self.m.cfg.window) return error.NotAChunkEnd;
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = self.next_snap, .at = at, .base = l.base, .end = l.end, .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()) };
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        if (try self.begin(.snap_save)) {
            try self.wire.int(self.gpa, u64, sn.id);
            try self.wire.int(self.gpa, u32, @intCast(l.slot));
            try self.flush();
        }
        try self.m.snapCopy(l.slot, sn.buf.ptr, true);
        return sn;
    }

    pub fn snapRestoreFn(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    /// Whether a kept state restores for a prompt of `len` tokens: the prompt cache passes one that does not over for
    /// an earlier state (the prompt pass would refuse it, and the prompt would run from 0).
    pub fn snapResumesFn(ptr: *anyopaque, saved: *anyopaque, len: u32) bool {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        const why = of(ptr).refusal(sn, len) orelse return true;
        noteRefusal(sn, len, why, "the prompt cache looks for an earlier one");
        return false;
    }

    /// Whether a kept state still holds its rows: one a stream wrote over leaves the prompt cache.
    pub fn snapLiveFn(ptr: *anyopaque, saved: *anyopaque) bool {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return std.mem.indexOfScalar(*Snap, of(ptr).snaps.items, sn) != null and !sn.stale;
    }

    pub fn snapDropFn(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        const i = std.mem.indexOfScalar(*Snap, self.snaps.items, sn) orelse return;
        _ = self.snaps.orderedRemove(i);
        if (self.begin(.snap_drop) catch false) {
            self.wire.int(self.gpa, u64, sn.id) catch {};
            self.flush() catch |err| std.log.err("dsv41: rank 1 kept a dropped prompt state: {s}", .{@errorName(err)});
        }
        sn.buf.free();
        self.gpa.destroy(sn);
    }

        return .{ .ptr = self, .reserve = spillReserve, .install = spillInstall, .discard = spillDiscard, .info = spillInfo, .land = if (self.lander != null) spillLand else null, .probe = if (self.probe != null) spillProbe else null, .land_time = if (self.lander != null) &self.land_time else null, .land_counts = if (self.lander != null) &self.lander.?.counts else null, .sync_send = if (self.peer != null) syncSend else null, .sync_idle = landIdle };
        _ = piece_bytes;
    fn landBcast0(_: *anyopaque, _: []const land.Span) anyerror!void {
        return error.NoLanding; // (every rank restores from its own store: no span is broadcast)
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        if (self.first_of == s) self.first_of = null;
        if (self.streams.fetchRemove(s)) |kv| {
            const slot = kv.value.slot;
            if (self.fills[slot]) |*f| {
                if (self.filling_slot == slot) {
                    self.m.joinAhead();
                    self.m.fill_seq = &.{};
                }
                f.deinit(self.gpa);
                self.fills[slot] = null;
            }
            if (self.filling_slot == slot) self.filling_slot = null;
            self.used[slot] = false;
        }
        self.idleRdma() catch |err| std.debug.panic("RDMA idle transition failed: {s}", .{@errorName(err)});
    }
};

const idle_ready = "rdma-idle-ready-v1";
const idle_commit = "rdma-idle-commit-v1";

fn landBcast(_: *anyopaque, _: []const land.Span) anyerror!void {
    return error.NoLanding; // (every rank restores from its own store: no span is broadcast)
fn usesRdma(kind: Kind) bool {
    return switch (kind) {
        .fill_begin, .fill_chunk, .fill_pair, .verify, .pass, .head_whole => true,
        else => false,
    };
}

fn hasRdma(m: *const Model) bool {
    return m.ring != null or m.rdma_rings != null;
}

fn setRdmaIdle(m: *Model, idle: bool) void {
    if (m.ring) |r| {
        if (idle) r.park() else r.wake();
    }
    if (m.rdma_rings) |rs| {
        for ([_]?*@import("rdma.zig").Ring{ rs.all4, rs.col, rs.pair }) |y| if (y) |r| {
            if (idle) r.park() else r.wake();
        };
    }
}

/// Prefix survivals: out[j] = the product of sigmoid(conf[i]) for i <= j, in float64 (libm's exp, as chooseK).
pub fn survivals(conf: []const f32, out: []f64) void {
    var surv: f64 = 1.0;
    for (out, conf[0..out.len]) |*o, cf| {
        surv *= draft.landing(cf);
        o.* = surv;
    }
}

/// A primitive this rank refused: rank 0 runs the same one on the same inputs and refuses it at the same step (its
/// request ends with the error), so the follower reports it and takes the next frame instead of ending.
fn refused(err: anyerror) void {
    std.debug.print("{{\"follow_refused\": \"{s}\"}}\n", .{@errorName(err)});
}

/// Whether a refusal leaves every rank in step: the errors a primitive raises from its inputs alone, before it has posted or
/// awaited an exchange, which rank 0 raises at the same step. Any other error (a disk read, an allocation, a device call) is
/// this rank's alone.
fn refusalKeepsRanksInStep(err: anyerror) bool {
    return switch (err) {
        error.BadChunk, error.BadPair, error.BadSlot, error.BadRound, error.HeadCandOff, error.NoDrafter => true,
        else => false,
    };
}

/// A primitive that carries exchanges (a chunk, a pair, a round, a head, a drafter pass) refused: reported like any refusal,
/// and when the error is this rank's alone the transport fails at once, since the other ranks wait for outputs this rank
/// skipped.
fn refusedExchange(m: anytype, err: anyerror) error{GpuContextFailed}!void {
    refused(err);
    if (!refusalKeepsRanksInStep(err)) {
        m.failExchanges();
        // A GPU fault (an illegal address, a trap) leaves this rank's context running nothing more, and the lanes' failure
        // only reaches the ranks through the NIC: a follower whose context failed ends, so its link closes and rank 0's watch
        // (Model.watchPeers) stops rank 0 with a line that names this rank.
        if (err == error.CudaFailed and m.contextFailed()) return error.GpuContextFailed;
    }
}

/// A follower's verify times with Model.prof (a profile on ranks 1-3: rounds, issue and wall ns summed).
pub var follow_prof: struct { rounds: u64 = 0, enqueue: u64 = 0, forward: u64 = 0 } = .{};

/// The follower's link, for the failure line's socket head (where.head_probe).
var probe_peer: ?link.Link = null;

fn headProbe() i32 {
    const p = probe_peer orelse return -1;
    return if (p.peekKind() catch null) |k| k else -1;
}
                c.m.verify(rows, c.built.absorb[0..nw], head) catch |err| try refusedExchange(c.m, err);
                if (c.m.prof) {
                    follow_prof.rounds += 1;
                    follow_prof.enqueue += c.m.t_enqueue;
                    follow_prof.forward += c.m.t_forward;
                }
            },
            .head_whole => {
                // the round's candidates fell back on rank 0 (TF_DS_HEAD_CAND): its whole head, every rank alike
                const n: usize = try r.int(u32);
                c.m.headWhole(n) catch |err| try refusedExchange(c.m, err);
            },
            .pass => {
        }
        if (self.first_of == s) self.first_of = null;
        const kv = self.streams.fetchRemove(s) orelse return;
        self.used[kv.value.slot] = false;
    }
};

/// Prefix survivals: out[j] = the product of sigmoid(conf[i]) for i <= j, in float64 (libm's exp, as chooseK).
pub fn survivals(conf: []const f32, out: []f64) void {
    var surv: f64 = 1.0;
    for (out, conf[0..out.len]) |*o, cf| {
        surv *= 1.0 / (1.0 + exp(-@as(f64, cf)));
        o.* = surv;
    }
}

/// A primitive this rank refused: rank 0 runs the same one on the same inputs and refuses it at the same step (its
/// request ends with the error), so the follower reports it and takes the next frame instead of ending. A GPU fault
/// (an illegal address, a trap) is this rank's alone: rank 0 runs on into its next exchange and waits there for this
/// rank, whose context runs nothing after the fault (tp2 #25). So a CUDA failure that left the context failed ends
/// the follower: its link closes, and rank 0's watch stops rank 0 with a line that names this rank.
fn refused(m: anytype, err: anyerror) error{GpuContextFailed}!void {
    std.debug.print("{{\"follow_refused\": \"{s}\"}}\n", .{@errorName(err)});
    if (err == error.CudaFailed and m.contextFailed()) return error.GpuContextFailed;
}

/// A follower's verify times with Model.prof (a profile on ranks 1-3: rounds, issue and wall ns summed).
pub var follow_prof: struct { rounds: u64 = 0, enqueue: u64 = 0, forward: u64 = 0 } = .{};

/// Rank 1 (each of ranks 1-3 on the four-node split): the primitives rank 0 sends, run in order until it says done.
pub fn follow(gpa: std.mem.Allocator, m: *Model) !void {
    const peer = m.peer();
    probe_peer = peer;
    defer probe_peer = null;
    where.head_probe = headProbe;
    defer where.head_probe = null;
    defer gpa.free(buf);
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    var win_ids: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&win_ids) |*q| q.deinit(gpa);
    var built: Built = .{};
    var step: u64 = 0;
    var rdma_idle = false;
    var slot: usize = 0;
    var img_rows: std.ArrayList(u8) = .empty; // a prompt's image rows as their pieces come
    defer img_rows.deinit(gpa);
    var kept: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty; // rank 0's kept states, this rank's part
    defer {
        var ki = kept.valueIterator();
        while (ki.next()) |b| b.free();
        kept.deinit(gpa);
    }
    var lf = try land.Follower(cuda.DeviceBuffer).init(gpa, .{ .ctx = m, .alloc = landAlloc, .free = landFree }, m.snapBytes(), lplanes, 0);
    while (true) {
        where.set(.main_loop, 0, 0);
        // the next primitive, polled for 20 ms before the read blocks (rank 0 sends it a fraction of a millisecond
        // after this rank's GPU goes idle in a decode; a sleeping thread would add its core's wake-up to every frame)
        var r: Reader = .{ .b = try peer.recvSpin(buf, 20 * std.time.ns_per_ms) };
        const kind: Kind = switch (try r.int(u8)) {
            1 => .fill_begin,
            2 => .fill_chunk,
            3 => .verify,
            4 => .pass,
            5 => .snap_save,
            6 => .snap_restore,
            7 => .snap_drop,
            8 => .fill_images,
            9 => .done,
            10 => .fill_resume,
            11 => .rdma_idle,
            14 => .fill_pair,
            else => return error.BadFrame,
        };
        if (usesRdma(kind) and rdma_idle) {
            setRdmaIdle(m, false);
            rdma_idle = false;
        }
        if (try r.int(u64) != step) return error.OutOfStep;
        where.frame('T', @backingInt(kind), step);
        where.set(.frame, @backingInt(kind), @intCast(step));
        step += 1;
        defer m.flushDiag(false); // (TF_DS_ROCI_XLOG / TF_DS_NACT_LOG's files on a follower too: every 256th frame, nothing when both are off)
        // a chunk's read-ahead of the next is done before anything but that chunk (a prompt's ids may change)
        if (kind != .fill_chunk and kind != .snap_save and kind != .snap_restore and kind != .fill_images) {
            m.joinAhead();
            m.fill_seq = &.{};
        }
        switch (kind) {
            .fill_begin => {
                slot = try r.int(u32);
                const base = try r.int(u64);
                if (slot >= max_streams) return error.BadSlot;
                img_rows.clearRetainingCapacity();
                try r.ids(gpa, &seqs[slot]);
                m.fillBegin(slot, @intCast(base)) catch |err| try refused(m, err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_resume => {
                slot = try r.int(u32);
                const base = try r.int(u64);
                if (slot >= max_streams) return error.BadSlot;
                img_rows.clearRetainingCapacity();
                m.fillBegin(slot, @intCast(base)) catch |err| try refused(m, err);
                m.fillBegin(slot, @intCast(base)) catch |err| refused(err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_chunk => {
                const start: usize = @intCast(try r.int(u64));
                const n: usize = try r.int(u32);
                const replay: usize = @intCast(try r.int(u64));
                if (start + n > seqs[slot].items.len) return error.BadChunk;
                if (m.fill_seq.len == 0) m.fill_seq = seqs[slot].items; // a piece after a round: the read-ahead again
                _ = m.fillChunk(seqs[slot].items[0 .. start + n], start, n, replay) catch |err| try refused(m, err);
                _ = m.fillChunk(seqs[slot].items[0 .. start + n], start, n, replay) catch |err| try refusedExchange(m, err);
            },
            .fill_pair => {
                const start: usize = @intCast(try r.int(u64));
                const n_a: usize = try r.int(u32);
                const n_b: usize = try r.int(u32);
                const n_c: usize = try r.int(u32); // the third pass's rows (0: a pair)
                const replay: usize = @intCast(try r.int(u64));
                if (start + n_a + n_b + n_c > seqs[slot].items.len) return error.BadChunk;
                if (m.fill_seq.len == 0) m.fill_seq = seqs[slot].items;
                m.fillGroup(seqs[slot].items[0 .. start + n_a + n_b + n_c], start, if (n_c != 0) &.{ n_a, n_b, n_c } else &.{ n_a, n_b }, replay) catch |err| try refusedExchange(m, err);
            },
            .pass => {
                const n = try r.int(u32);
                const steps = try r.int(u32);
                if (n == 0 or n > max_streams) return error.BadFrame;
                var tokens: [max_streams]i64 = undefined;
                var q0: [max_streams]i64 = undefined;
                var slots: [max_streams]i64 = undefined;
                for (0..n) |i| {
                    tokens[i] = try r.int(i64);
                    q0[i] = try r.int(i64);
                    slots[i] = try r.int(i64);
                }
                m.pass(tokens[0..n], q0[0..n], slots[0..n], steps) catch |err| try refused(m, err);
            },
            .fill_images => {
                const count = try r.int(u32);
                var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
                defer spans.deinit(gpa);
                var row: usize = 0;
                for (0..count) |_| {
                    const at: usize = @intCast(try r.int(u64));
                    const n: usize = @intCast(try r.int(u64));
                    try spans.append(gpa, .{ .at = at, .len = n, .row = row });
                    row += n;
                }
                const off: usize = @intCast(try r.int(u64));
                const piece = r.rest();
                const total = row * m.cfg.hidden * 2;
                if (off != img_rows.items.len or off + piece.len > total or piece.len == 0) return error.BadFrame;
                try img_rows.appendSlice(gpa, piece);
                if (img_rows.items.len == total) {
                    m.setImages(spans.items, img_rows.items) catch |err| try refused(m, err);
                    img_rows.clearRetainingCapacity();
                }
            },
            .snap_save => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                if (s >= max_streams) return error.BadSlot;
                var b = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
                m.snapCopy(s, b.ptr, true) catch |err| {
                    b.free();
                    try refused(m, err);
                    continue;
                };
                try kept.put(gpa, id, b);
            },
            .snap_restore => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                if (s >= max_streams) return error.BadSlot;
                const from: usize = @intCast(try r.int(u64));
                const to: usize = @intCast(try r.int(u64));
                const at: usize = @intCast(try r.int(u64));
                const b = kept.get(id) orelse return error.UnknownSnapshot; // rank 0 restores only states both ranks keep
                if (from != to) m.copyExtent(from, to, at) catch |err| try refused(m, err);
                m.snapCopy(s, b.ptr, false) catch |err| try refused(m, err);
            },
            .snap_drop => {
                const id = try r.int(u64);
                if (kept.fetchRemove(id)) |kv| {
                    var b = kv.value;
                    b.free();
                }
            },
            .rdma_idle => {
                try m.ctx.synchronize();
                try peer.send(idle_ready);
                if (!std.mem.eql(u8, try peer.recv(buf), idle_commit)) return error.BadIdleCommit;
                setRdmaIdle(m, true);
                rdma_idle = true;
            },
            .done => {
                m.unwatch(); // (rank 0 closes its link after done)
                return;
            },
        }
    }
}

test "a CUDA failure that left the context failed ends the follower, any other refusal does not" {
    const Probe = struct {
        failed: bool,
        asked: usize = 0,
        pub fn contextFailed(p: *@This()) bool {
            p.asked += 1;
            return p.failed;
        }
    };
    var gone: Probe = .{ .failed = true };
    try std.testing.expectError(error.GpuContextFailed, refused(&gone, error.CudaFailed));
    var alive: Probe = .{ .failed = false }; // a failed call the context survived: refused as before
    try refused(&alive, error.CudaFailed);
    try std.testing.expectEqual(@as(usize, 1), alive.asked);
    var other: Probe = .{ .failed = true }; // rank 0 refuses these too: the context is not asked
    try refused(&other, error.MissingTritonVariant);
    try refused(&other, error.OutOfDeviceMemory);
    try std.testing.expectEqual(@as(usize, 0), other.asked);
}

test "draft chances are the prefix survivals chooseK sums" {
    var out: [3]f64 = undefined;
    survivals(&.{ 0.0, 0.0, 2.0 }, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), out[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.25 / (1.0 + @exp(@as(f64, -2.0))), out[2], 1e-12);
}

test "extents take the first chunk-aligned gap" {
    const used = [_][2]usize{ .{ 0, 4096 }, .{ 6144, 8192 } };
    try std.testing.expectEqual(@as(?usize, 4096), place(&used, 2048, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 8192), place(&used, 4096, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 0), place(&.{}, 2048, 2048));
    try std.testing.expectEqual(@as(?usize, null), place(&used, 4096, 10240));
}

test "a frame's fields read back as written" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    try w.begin(gpa, .verify, 7);
    try w.int(gpa, u64, 1 << 40);
    try w.ids(gpa, &.{ 1, -2, 129279 });
    var r: Reader = .{ .b = w.buf.items };
    try std.testing.expectEqual(@as(u8, @backingInt(Kind.verify)), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 7), try r.int(u64));
    try std.testing.expectEqual(@as(u64, 1 << 40), try r.int(u64));
    var got: std.ArrayList(i32) = .empty;
    defer got.deinit(gpa);
    try r.ids(gpa, &got);
    try std.testing.expectEqualSlices(i32, &.{ 1, -2, 129279 }, got.items);
    try std.testing.expectError(error.ShortFrame, r.int(u8));
}

test "a round's rows: each window at its positions, its ids written over the rejected rows" {
    const gpa = std.testing.allocator;
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    try seqs[0].appendSlice(gpa, &.{ 10, 11, 12, 13, 99, 98 }); // positions 4.. hold a rejected window
    try seqs[2].appendSlice(gpa, &.{ 20, 21 });
    var b: Built = .{};
    const rows = try b.build(gpa, &seqs, &.{
        .{ .slot = 0, .base = 0, .end = 4096, .start = 4, .absorb = true, .ids = &.{ 14, 15 } },
        .{ .slot = 2, .base = 4096, .end = 8192, .start = 2, .absorb = false, .ids = &.{22} },
    });
    try std.testing.expectEqualSlices(i64, &.{ 14, 15, 22 }, rows.ids);
    try std.testing.expectEqualSlices(i64, &.{ 4, 5, 2 }, rows.pos);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2 }, rows.slots.?);
    try std.testing.expectEqualSlices(i64, &.{ 4096, 4096, 8192 }, rows.ends.?);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 12, 13, 14, 15 }, rows.windows.?[0].seq);
    try std.testing.expectEqualSlices(i32, &.{ 20, 21, 22 }, rows.windows.?[1].seq);
    try std.testing.expectEqual(@as(usize, 2), rows.windows.?[1].row);
    try std.testing.expectError(error.SequenceGap, b.build(gpa, &seqs, &.{.{ .slot = 1, .base = 0, .end = 2048, .start = 3, .absorb = false, .ids = &.{1} }}));
}

test "greedy draws take the first largest logit" {
    const row = [_]f32{ 0.5, 2.0, -1.0, 2.0 };
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, null));
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, .{ .seed = 3, .temperature = 0 }));
}

test "a prefix two conversations share: the second one resuming it in place would stale the first one's later marks" {
    var m: Model = undefined;
    m.pool_cap = 1 << 20;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    // the first conversation's extent holds the shared prefix's mark (system prompt and tools) and its own later one
    var shared: Snap = .{ .id = 0, .at = 18432, .base = 0, .end = 430080, .buf = undefined };
    var later: Snap = .{ .id = 1, .at = 395264, .base = 0, .end = 430080, .buf = undefined };
    try ln.snaps.append(std.testing.allocator, &shared);
    try ln.snaps.append(std.testing.allocator, &later);
    try std.testing.expect(ln.keptOver(shared.base + shared.at, shared.base + 55296, &shared)); // the second one moves
    try std.testing.expect(!ln.keptOver(later.base + later.at, later.base + 432128, &later)); // the first one's next turn stays
}

test "TF_FILL_CHUNKS steps a prompt pass in the lane core, one pass at a time; unset, each pass runs whole" {
    var m: Model = undefined;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    const whole = ln.backend().vtable;
    try std.testing.expect(whole.prefill_step == null and !whole.prefill_one);
    ln.fill_chunks = 2;
    const steps = ln.backend().vtable;
    try std.testing.expect(steps.prefill_step != null and steps.prefill_one);
    // every other entry the same as served
    try std.testing.expect(steps.prefill == whole.prefill and steps.verify == whole.verify and steps.release == whole.release and steps.draft == whole.draft);
    // TF_FILL_MANY: the same steps, and the host admits while a pass fills
    ln.fill_many = true;
    const many = ln.backend().vtable;
    try std.testing.expect(many.prefill_step == steps.prefill_step and !many.prefill_one);
    try std.testing.expect(many.prefill == whole.prefill and many.verify == whole.verify and many.release == whole.release and many.draft == whole.draft);
    // without TF_FILL_CHUNKS it changes nothing: each pass whole
    ln.fill_chunks = 0;
    try std.testing.expect(ln.backend().vtable == &Lanes.whole_vt);
}

test "TF_FILL_MANY: each stream's pass is found by its stream, and one ending frees its slot and the model's fill state" {
    var m: Model = undefined;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    var a: lanes.Stream = undefined;
    var b: lanes.Stream = undefined;
    ln.fills[2] = .{ .s = &a, .slot = 2, .len = 10, .replay = 0, .last_end = 0, .start = 2048 };
    ln.cur = 2;
    try std.testing.expect(ln.fillOf(&a).? == &ln.fills[2].?);
    try std.testing.expect(ln.fillOf(&b) == null);
    ln.fills[5] = .{ .s = &b, .slot = 5, .len = 99000, .replay = 0, .last_end = 0, .start = 0 };
    try std.testing.expect(ln.fillOf(&b).? == &ln.fills[5].?);
    ln.dropFill(5);
    try std.testing.expect(ln.cur == 2 and ln.fillOf(&b) == null and ln.anyFill());
    ln.dropFill(2);
    try std.testing.expect(!ln.anyFill() and ln.cur == null);
}

/// A test row: token i's logit from splitmix64(i + 1) (exact in f32, a few ties).
fn testRow(row: []f32) void {
    for (row, 0..) |*x, i| {
        var z: u64 = @as(u64, i + 1) *% 0x9E3779B97F4A7C15;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        x.* = @as(f32, @floatFromInt((z ^ (z >> 31)) >> 52)) / 256.0 - 8.0;
    }
}

test "a masked draw is the row's draw with the other tokens at -inf, and never a disallowed token" {
    const V = 1000;
    var row: [V]f32 = undefined;
    testRow(&row);
    var buf: [V]f32 = undefined;
    var off: [V]f32 = undefined;
    var mask: [maskWords(V)]u32 = @splat(0);
    for (0..V) |i| if (i % 7 == 3 or i % 250 == 1) {
        mask[i / 32] |= @as(u32, 1) << @intCast(i % 32);
    };
    for (&off, row, 0..) |*o, v, i| o.* = if (lanes.grammar.allows(&mask, i)) v else -std.math.inf(f32);
    const all: [maskWords(V)]u32 = @splat(std.math.maxInt(u32));
    const settings = [_]?lanes.Sampling{
        null,
        .{ .seed = 5, .temperature = 0 },
        .{ .seed = 11, .temperature = 1.0, .top_k = 1 },
        .{ .seed = 12, .temperature = 1.0, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 13, .temperature = 0.7, .top_k = 56, .top_p = 0.9, .min_p = 0.05 },
        .{ .seed = 14, .temperature = 1.3, .top_k = 40, .top_p = 1.0 },
    };
    for (settings) |s| for ([_]u64{ 0, 17, 4096, 1 << 20 }) |position| {
        const got = try drawMasked(&buf, &row, position, s, &mask);
        try std.testing.expect(lanes.grammar.allows(&mask, got));
        try std.testing.expectEqual(try draw(&off, position, s), got);
        // every token allowed: the row's own draw
        try std.testing.expectEqual(try draw(&row, position, s), try drawMasked(&buf, &row, position, s, &all));
    };
    // greedy: the first largest allowed logit
    var best: ?usize = null;
    for (row, 0..) |v, i| if (lanes.grammar.allows(&mask, i) and (best == null or v > row[best.?])) {
        best = i;
    };
    try std.testing.expectEqual(@as(u32, @intCast(best.?)), try drawMasked(&buf, &row, 3, null, &mask));
    // one token allowed: that token at any sampling
    var one: [maskWords(V)]u32 = @splat(0);
    one[999 / 32] = @as(u32, 1) << (999 % 32);
    for (settings) |s| try std.testing.expectEqual(@as(u32, 999), try drawMasked(&buf, &row, 9, s, &one));
    // nothing allowed: no draw
    try std.testing.expectError(error.NoAllowedToken, drawMasked(&buf, &row, 9, null, &@as([maskWords(V)]u32, @splat(0))));
}

test "a stream with a grammar draws its first token from the kept prompt row, and its held drafts read back as host tokens" {
    const gpa = std.testing.allocator;
    // no model here: rdma_idle set, so release (0034) has no RDMA proxies to park
    var ln: Lanes = .{ .gpa = gpa, .m = undefined, .peer = null, .rdma_idle = true };
    defer ln.deinit();
    const b = ln.backend();
    try std.testing.expect(b.vtable.first_masked != null and b.vtable.held != null);
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2, 3 }, .max_new = 8, .sampling = .{ .seed = 7, .temperature = 0.8, .top_k = 20, .top_p = 0.95 } });
    defer s.deinit(gpa);
    try ln.streams.put(gpa, &s, .{ .slot = 0, .base = 0, .end = model.chunk_rows });
    ln.used[0] = true;
    var row: [1000]f32 = undefined;
    testRow(&row);
    var mask: [maskWords(1000)]u32 = @splat(0);
    for ([_]usize{ 4, 100, 640, 999 }) |t| mask[t / 32] |= @as(u32, 1) << @intCast(t % 32);
    // no kept row (the stream had no grammar at its prompt pass): no masked draw
    try std.testing.expectError(error.NoPromptRow, b.vtable.first_masked.?(b.ptr, &s, 3, &mask));
    ln.first_row = try gpa.dupe(f32, &row);
    ln.first_of = &s;
    try std.testing.expectError(error.PositionMismatch, b.vtable.first_masked.?(b.ptr, &s, 2, &mask));
    var buf: [1000]f32 = undefined;
    const want = try drawMasked(&buf, &row, 3, s.sampling, &mask);
    try std.testing.expectEqual(want, try b.read(try b.vtable.first_masked.?(b.ptr, &s, 3, &mask)));
    // the drafts the last pass held, in window order
    const l = ln.streams.getPtr(&s).?;
    l.held[0..3].* = .{ 50, 51, 52 };
    l.nheld = 3;
    var out: [2]u32 = undefined;
    try b.vtable.held.?(b.ptr, &s, &out);
    try std.testing.expectEqualSlices(u32, &.{ 50, 51 }, &out);
    var four: [4]u32 = undefined;
    try std.testing.expectError(error.HeldMismatch, b.vtable.held.?(b.ptr, &s, &four));
    // a released stream's row is no one's
    b.release(&s);
    try std.testing.expect(ln.first_of == null);
}

test "a window's masks are none or every row's bits" {
    const gpa = std.testing.allocator;
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{1}, .max_new = 4 });
    defer s.deinit(gpa);
    const words = maskWords(129280);
    try std.testing.expectEqual(@as(usize, 4040), words);
    const bits: [3 * 4040]u32 = @splat(1);
    const w: be.Window = .{ .stream = &s, .pending = 1, .held = 0, .tokens = &.{ 2, 3 }, .parents = null, .positions = &.{ 2, 3, 4 } };
    try std.testing.expect(masksFit(w, words));
    var with = w;
    with.masks = &bits;
    try std.testing.expect(masksFit(with, words));
    with.masks = bits[0 .. 2 * words];
    try std.testing.expect(!masksFit(with, words));
}

test "retained histories preserve their replay and output reservations (PR #10: eight 100K histories)" {
    var m: Model = undefined;
    m.watch = .{}; // (finish and deinit end the watch)
    m.pool_cap = 1 << 20;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var snapshots: [8]Snap = undefined;
    for (&snapshots, 0..) |*sn, i| {
        const base = try ln.placeKept(&.{}, 102400, null);
        try std.testing.expectEqual(i * (102400 + growth_rows), base); // each with room to grow while the pool allows

        sn.* = .{ .id = i, .at = 98304, .base = base, .end = base + 102400, .buf = undefined };
        try ln.snaps.append(std.testing.allocator, sn);
    }
    // All eight 100K histories can resume within their reservations without invalidating a neighbor.
    for (&snapshots) |*sn| {
        try std.testing.expect(!ln.keptOver(sn.base + sn.at, sn.end, sn));
        ln.staleOver(sn.base + sn.at, sn.end, sn);
    }
    for (snapshots) |sn| try std.testing.expect(!sn.stale);
    // a larger continuation resumes in place within its room to grow
    try std.testing.expect(!ln.keptOver(snapshots[0].at, 104448, &snapshots[0]));
    // past that room it would overwrite the next cached prefix, so it relocates
    try std.testing.expect(ln.keptOver(snapshots[0].at, 102400 + growth_rows + 2048, &snapshots[0]));
    // At genuine capacity pressure, kept states may still be evicted; a live range never is.
    const live = [_][2]usize{.{ 0, 102400 }};
    try std.testing.expectEqual(@as(usize, 102400), try ln.placeKept(&live, 409600, null));
    try std.testing.expect(ln.snaps.items[1].stale and ln.snaps.items[3].stale);
    try std.testing.expect(!ln.snaps.items[4].stale);
    try std.testing.expect(!ln.snaps.items[0].stale); // its rows lie in the live stream: that stream holds the room
}

test "a prefix two conversations share: the second one resuming it in place would stale the first one's later marks" {
    var m: Model = undefined;
    m.watch = .{}; // (finish and deinit end the watch)
    m.pool_cap = 1 << 20;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    // the first conversation's extent holds the shared prefix's mark (system prompt and tools) and its own later one
    var shared: Snap = .{ .id = 0, .at = 18432, .base = 0, .end = 430080, .buf = undefined };
    var later: Snap = .{ .id = 1, .at = 395264, .base = 0, .end = 430080, .buf = undefined };
    try ln.snaps.append(std.testing.allocator, &shared);
    try ln.snaps.append(std.testing.allocator, &later);
    try std.testing.expect(ln.keptOver(shared.base + shared.at, shared.base + 55296, &shared)); // the second one moves
    try std.testing.expect(!ln.keptOver(later.base + later.at, later.base + 432128, &later)); // the first one's next turn stays
}
    m.watch = .{}; // (finish and deinit end the watch)
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 },
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, // 9: the same kinds, the Rocinante-only Settings
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, // 11: the same kinds, fill_pair carries the third pass's rows
    m.watch = .{}; // (finish and deinit end the watch)
    m.watch = .{}; // (finish and deinit end the watch)
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "            c.m.verify(rows, c.built.absorb[0..nw], head) catch |err| try refusedExchange(c.m, err);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "                const n: usize = try r.int(u32);\n                c.m.headWhole(n) catch |err| try refusedExchange(c.m, err);"));

    const src = @embedFile("lanes.zig");
    const step = std.mem.indexOf(u8, src, "fn prefillStep(") orelse return error.TestUnexpectedResult;
    try std.testing.expect(rule < pair);
}

test "a prompt step the host bills to decoders ends on the device before prefillStep returns" {
    const src = @embedFile("lanes.zig");
    const step = std.mem.indexOf(u8, src, "fn prefillStep(") orelse return error.TestUnexpectedResult;
    const issued = std.mem.indexOfPos(u8, src, step, "} else try m.fillChunk(") orelse return error.TestUnexpectedResult;
    const settle = std.mem.indexOfPos(u8, src, step, "if (s.settle_step) {\n            try m.pairJoin();\n            try m.stream.synchronize();\n        }") orelse return error.TestUnexpectedResult;
    const leave = std.mem.indexOfPos(u8, src, step, "if (f.start < len) {") orelse return error.TestUnexpectedResult;
    // after the chunk or the pair was issued, before the step hands back to the host
    try std.testing.expect(issued < settle and settle < leave);
    // (a pair's second pass runs on its own stream: pairJoin makes the model's stream wait for its end before the host waits on the stream,
    // or the second pass's tail would land in the next round's time)
}
    try std.testing.expect(std.mem.indexOf(u8, src, "if (self.spill) |t| t.bump(lane.base + f.start, lane.base + f.start + n + n2 + n3);\n        self.instr.chunk(m.now() / std.time.ns_per_us, n);") != null);

test "every wait that can hold a host names itself, and each rank journals the frames it sends and takes" {
    const src = @embedFile("lanes.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"every wait that can hold a host names itself") orelse return error.TestUnexpectedResult];
    for ([_][]const u8{
        "where.frame('S', @backingInt(kind), self.step);",
        "where.frame('T', @backingInt(kind), step);",
        "where.set(.round, @intCast(windows.len), 0);\n        defer where.set(.after_backend, 0, 0);",
        "where.set(.draft, @intCast(requests.len), 0);\n        defer where.set(.after_backend, 0, 0);",
        "where.set(.frame, @backingInt(kind), @intCast(step));",
        "where.head_probe = headProbe;",
        "p.peekKind() catch null",
    }) |needle| try std.testing.expect(std.mem.indexOf(u8, code, needle) != null);
    const prompt_src = @embedFile("prompt.zig");
    const pcode = prompt_src[0 .. std.mem.indexOf(u8, prompt_src, "\ntest \"") orelse return error.TestUnexpectedResult];
    for ([_][]const u8{ "where.enter(.capture_vote, @backingInt(stage), recorded)", "where.enter(.engram_rows, @intCast(j), @intCast(ci))", "where.enter(.engram_flush, 0, 0)" }) |needle| try std.testing.expect(std.mem.indexOf(u8, pcode, needle) != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("rdma.zig"), "where.report(&pb)") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("model.zig"), "return where.report(buf);") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("rocinante.zig"), "var pb: [640]u8 = undefined;") != null);
}

test "a refusal that is this rank's alone fails the transport at once; a refusal every rank raises from the inputs does not" {
    // the errors rank 0 raises at the same step leave the ranks in step
    for ([_]anyerror{ error.BadChunk, error.BadPair, error.BadSlot, error.BadRound, error.HeadCandOff, error.NoDrafter }) |e| try std.testing.expect(refusalKeepsRanksInStep(e));
    // a disk read, an allocation, a device call: this rank skipped outputs the others wait for
    for ([_]anyerror{ error.OutOfMemory, error.InputOutput, error.CudaError, error.EngramRead }) |e| try std.testing.expect(!refusalKeepsRanksInStep(e));
    const src = @embedFile("lanes.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"a refusal that is this rank's alone") orelse return error.TestUnexpectedResult];
    // every primitive that carries exchanges reports through refusedExchange; the rest stay plain refusals
    for ([_][]const u8{ "m.fillChunk(seqs[slot]", "m.fillGroup(seqs[slot]" }) |call| {
        const at = std.mem.indexOfPos(u8, code, std.mem.indexOf(u8, code, "pub fn follow(") orelse return error.TestUnexpectedResult, call) orelse return error.TestUnexpectedResult;
        const line_end = std.mem.indexOfPos(u8, code, at, "\n") orelse return error.TestUnexpectedResult;
        try std.testing.expect(std.mem.indexOf(u8, code[at..line_end], "refusedExchange(m, err)") != null);
    }
    // the round frames (FollowCtx.roundFrame, run from the main loop and from inside a pass's layers) report the same way
    try std.testing.expect(std.mem.indexOf(u8, code, "m.failExchanges();") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("model.zig"), "rc.t.fail(why, .own1, gone);") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("model.zig"), "m.failLanes(.refused, 0);") != null);
}

test "TF_DS_ROCI_STAGGER: a pair forms only where no stream decodes, or where the host owes the streams a count of rounds (TF_DS_ROUNDS_BESIDE)" {
    // a prompt alone: the pair forms
    try std.testing.expect(pairMayForm(true, false, 0));
    // decoders running: no pair without a count of rounds

// The pair rules (R1-R7 of the frame-order panel, tip 7ce258cdb), one host test each. The rules are why a pair beside
// generating users is safe: frames cannot differ between ranks, and nothing waits on a frame that is not yet there.

test "R1: a pair is ONE frame with both sizes in it, and the follower reads the same fields in the same order" {
    const src = @embedFile("lanes.zig");
    const code = src;
    const step = std.mem.indexOf(u8, code, "fn prefillStep(") orelse return error.TestUnexpectedResult;
    const send = code[std.mem.indexOfPos(u8, code, step, "if (n2 != 0) {").?..std.mem.indexOfPos(u8, code, step, "} else if (try self.begin(.fill_chunk))").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, send, "self.begin(.fill_pair)"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, send, ".fill_chunk"));
    const sent = [_][]const u8{ "u64, f.start", "u32, @intCast(n))", "u32, @intCast(n2))", "u32, @intCast(n3))", "u64, f.replay" };
    var at: usize = 0;
    for (sent) |w| at = (std.mem.indexOfPos(u8, send, at, w) orelse return error.TestUnexpectedResult) + w.len;
    const h = code[std.mem.indexOf(u8, code, ".fill_pair => {").?..std.mem.indexOf(u8, code, ".verify, .head_whole, .pass => {").?]; // the follower's handler ends where the round frames start (D2: FollowCtx.roundFrame)
    at = 0;
    for ([_][]const u8{ "r.int(u64)", "r.int(u32)", "r.int(u32)", "r.int(u32)", "r.int(u64)" }) |w| at = (std.mem.indexOfPos(u8, h, at, w) orelse return error.TestUnexpectedResult) + w.len;
    // one primitive, one handler: the follower runs the pair (or the group of three: the frame's third size, 0 for a pair) in a single fillGroup call
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, h, "m.fillGroup("));
}

test "TF_DS_ROCI_STAGGER_LANES=3: the third pass forms only with three lanes open, nothing decoding and nothing owed, rows left behind the pair, and no stop at the pair's end; a pair is never cut short by it" {
    const no_marks = [_]u32{};
    // a 100K prompt, 4,096-row passes: the pair at 0 and 4096, the third at 8192 takes a whole pass
    try std.testing.expectEqual(@as(?usize, 4096), thirdPassRows(3, false, 0, 4096, 4096, 100_000, 4096, false, &no_marks, 0, 100_000));
    try std.testing.expectEqual(@as(?usize, 4096), thirdPassRows(3, false, 0, 4096, 4096, 100_000, 4096, true, &no_marks, 0, 100_000));
    // two lanes (the default): never
    try std.testing.expectEqual(@as(?usize, null), thirdPassRows(2, false, 0, 4096, 4096, 100_000, 4096, false, &no_marks, 0, 100_000));
    // a stream decoding, or the host owing rounds: the pair stays a pair
    try std.testing.expectEqual(@as(?usize, null), thirdPassRows(3, true, 0, 4096, 4096, 100_000, 4096, false, &no_marks, 0, 100_000));
    try std.testing.expectEqual(@as(?usize, null), thirdPassRows(3, false, 3, 4096, 4096, 100_000, 4096, false, &no_marks, 0, 100_000));
    // the pair ends the prompt: no rows left for a third
    try std.testing.expectEqual(@as(?usize, null), thirdPassRows(3, false, 0, 4096, 4096, 8192, 4096, false, &no_marks, 0, 8192));
    // a stop at the pair's end (the keep point of the prompt cache, or a mark inside the replay): the pair is the group
    try std.testing.expectEqual(@as(?usize, null), thirdPassRows(3, false, 0, 4096, 4096, 100_000, 4096, false, &no_marks, 8192, 100_000));
    try std.testing.expectEqual(@as(?usize, null), thirdPassRows(3, false, 0, 4096, 4096, 100_000, 4096, false, &.{8192}, 0, 100_000));
    // the same stop beyond the replay is no stop
    try std.testing.expectEqual(@as(?usize, 4096), thirdPassRows(3, false, 0, 4096, 4096, 100_000, 4096, false, &no_marks, 8192, 8000));
    // a short remainder is the pass rule's (the model refuses a pass that is not whole)
    try std.testing.expectEqual(@as(?usize, 1000), thirdPassRows(3, false, 0, 4096, 4096, 8192 + 1000, 4096, false, &no_marks, 0, 9192) orelse return error.TestUnexpectedResult);
    // the source: the third pass is asked only behind a pair, the frame carries it, and the pair's step stays a pair's for the host's rounds
    const src = @embedFile("lanes.zig");
    const step = std.mem.indexOf(u8, src, "fn prefillStep(") orelse return error.TestUnexpectedResult;
    const second = std.mem.indexOfPos(u8, src, step, "if (!stop_between and m.pairEligible(").?;
    const third = std.mem.indexOfPos(u8, src, step, "if (thirdPassRows(m.pair_lanes, decoding_now, rounds_now, mid, n2, len, cap").?;
    const frame = std.mem.indexOfPos(u8, src, step, "try self.begin(.fill_pair)").?;
    try std.testing.expect(second < third and third < frame);
    try std.testing.expect(std.mem.indexOfPos(u8, src, step, "if (m.groupEligible(seq.items[0 .. mid + n2 + n_c], f.start, &.{ n, n2, n_c }, f.replay, cap)) n3 = n_c;") != null);
    try std.testing.expect(std.mem.indexOfPos(u8, src, step, "t.bump(lane.base + f.start, lane.base + f.start + n + n2 + n3)") != null);
    try std.testing.expect(std.mem.indexOfPos(u8, src, step, "f.start += n + n2 + n3;") != null);
}

test "R2: no follower handler waits for a later frame: the frame loop reads one whole frame and runs it" {
    const src = @embedFile("lanes.zig");
    const code = src;
    // a follower that peeks at the link for the next frame's kind inside a pass is a deadlock
    // (the failure line's socket probe, headProbe, peeks and sleeps on nothing: it is outside the loop)
    const at = std.mem.indexOf(u8, code, "pub fn follow(gpa").?;
    const loop = code[at..std.mem.indexOfPos(u8, code, at, "\n}\n").?];
    try std.testing.expect(std.mem.indexOf(u8, loop, "peek" ++ "Kind") == null);
}

test "R4: the Engram worker is joined, or on its own reader, before a round follows a pair: rank 0 joins after the pair's issue, a follower's round frames join unless keep-ahead" {
    try std.testing.expect(leavesAhead(.fill_pair, false) and leavesAhead(.fill_chunk, false));
    for ([_]Kind{ .verify, .pass, .head_whole }) |k| {
        try std.testing.expect(!leavesAhead(k, false));
        try std.testing.expect(leavesAhead(k, true));
    }
    const src = @embedFile("lanes.zig");
    const code = src;
    const step = std.mem.indexOf(u8, code, "fn prefillStep(") orelse return error.TestUnexpectedResult;
    const issued = std.mem.indexOfPos(u8, code, step, "} else try m.fillChunk(").?;
    const join = std.mem.indexOfPos(u8, code, step, "if (yieldOn() and !m.keep_ahead) if (s.between)").?;
    try std.testing.expect(issued < join);
}

test "a CUDA failure that left the follower's context failed ends the follower, every other refusal does not" {
    const CtxProbe = struct {
        failed: bool,
        asked: usize = 0,
        failed_lanes: usize = 0,
        pub fn contextFailed(p: *@This()) bool {
            p.asked += 1;
            return p.failed;
        }
        pub fn failExchanges(p: *@This()) void {
            p.failed_lanes += 1;
        }
    };
    var gone: CtxProbe = .{ .failed = true };
    try std.testing.expectEqual(@as(usize, 1), gone.failed_lanes); // (the lanes fail first: the other ranks are told by the abort)
    var alive: CtxProbe = .{ .failed = false }; // a failed call the context survived: the lanes fail, the follower goes on
    try std.testing.expectEqual(@as(usize, 1), alive.asked);
    try std.testing.expectEqual(@as(usize, 1), alive.failed_lanes);
    var other: CtxProbe = .{ .failed = true }; // an error this rank alone raises, but no GPU fault: the context is not asked
    try std.testing.expectEqual(@as(usize, 0), other.asked);
    try std.testing.expectEqual(@as(usize, 1), other.failed_lanes);
    var step: CtxProbe = .{ .failed = true }; // rank 0 refuses these at the same step: nothing fails, nobody is asked
    try std.testing.expectEqual(@as(usize, 0), step.asked);
    try std.testing.expectEqual(@as(usize, 0), step.failed_lanes);
}

test "a peer that stops stops this rank: the watch starts before the follow loop and in the server, done and close end it" {
    const lanes_main = @embedFile("lanes_main.zig");
    const watch = std.mem.indexOf(u8, lanes_main, "m.watchPeers() catch").?;
    try std.testing.expect(watch < std.mem.indexOf(u8, lanes_main, "dsv41.lanes.follow(gpa, m, mirror) catch").?);
    try std.testing.expect(std.mem.indexOf(u8, lanes_main, "if (err == error.GpuContextFailed or m.contextFailed()) std.process.exit(1);") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("native.zig"), "st.m.watchPeers() catch") != null);
    const code = @embedFile("lanes.zig");
    const src = code[0 .. std.mem.indexOf(u8, code, "\ntest \"a CUDA failure that left the follower") orelse return error.TestUnexpectedResult];
    // the follower ends the watch on done, rank 0 before it sends done
    try std.testing.expect(std.mem.indexOf(u8, src, ".done => {\n                m.unwatch();") != null);
    const fin = src[std.mem.indexOf(u8, src, "pub fn finish(self: *Lanes) void {").?..];
    try std.testing.expect(std.mem.indexOf(u8, fin, "self.m.unwatch();").? < std.mem.indexOf(u8, fin, "self.begin(.done)").?);
    // the model's close ends it before the links go, and the watch's hook fails the lanes with the peer's reason
    const model_src = @embedFile("model.zig");
    try std.testing.expect(std.mem.indexOf(u8, model_src, "m.watch.stop(); // (still watching up to here").? < std.mem.indexOf(u8, model_src, "        m.fan.close();\n        m.w.deinit();").?);
    try std.testing.expect(std.mem.indexOf(u8, model_src, "m.failLanes(.peer, @intCast(rank));") != null);
    try std.testing.expect(std.mem.indexOf(u8, model_src, "m.failLanes(.peer, @intCast(rank));\n        m.abortRings(") != null);
}

test "draft chances are the prefix survivals chooseK sums" {
    var out: [3]f64 = undefined;
    survivals(&.{ 0.0, 0.0, 2.0 }, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), out[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.25 / (1.0 + @exp(@as(f64, -2.0))), out[2], 1e-12);
}

}
