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
const prompt = @import("prompt.zig");
const draft_vocab = @import("draft_vocab.zig");
const sampling = @import("sampling.zig");
const head_cand = @import("head_cand.zig");
const link = @import("link.zig");
const spill_tier = @import("spill_tier.zig");
const spill_mirror = @import("spill_mirror.zig");
const land = @import("land.zig");
const probe = @import("probe.zig");
const cuda = @import("cuda");
const live_config = model.live_config;

const be = lanes.backend;
const Model = model.Model;
const max_streams = model.max_streams;
const draft_pass_rows = 128; // a drafter pass's rows at most (prompt.grouped's decode window)
const max_rows = round.max_rows;

/// A window of the pending row and the drafter's block: the widest one the gates checked row for row.
pub const window_rows = 6;

/// The image rows a fill_images frame carries at most.
pub const image_piece = 32 << 10;

/// A gate's diagnostic (the tools' --logits-sha): 1 each prompt's logits' sha256 on stderr; 2 those and every verified
/// row's (rowShaLine), so a run's window rows compare byte for byte with the same rows of a run without drafts.
pub var logits_sha: u8 = 0;
/// The verifies under --logits-sha 2, counted: each row line's round.
var sha_round: u64 = 0;

/// --logits-sha's text: 0, 1 or 2; null: not a setting.
pub fn logitsSha(text: []const u8) ?u8 {
    for ([_][]const u8{ "0", "1", "2" }, 0..) |t, n| if (std.mem.eql(u8, text, t)) return @intCast(n);
    return null;
}

/// One verified row's logits line: its stream, the verify's count, the row in the window, the window's rows, the row's
/// position, its input token, and the sha256 of its logits' bytes.
pub fn rowShaLine(buf: []u8, id: []const u8, round_n: u64, row: usize, rows: usize, position: u64, input: i64, logits: []const f32) ![]const u8 {
    var dg: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(logits), &dg, .{});
    return std.fmt.bufPrint(buf, "{{\"row_logits\": \"{s}\", \"round\": {d}, \"row\": {d}, \"rows\": {d}, \"position\": {d}, \"input\": {d}, \"sha256\": \"{s}\"}}", .{ id, round_n, row, rows, position, input, std.fmt.bytesToHex(dg, .lower) });
}

/// Why a run that reads every round's whole rows on the host (`reads_rows`: --logits-log, --logits-sha 2) cannot, or null.
/// TF_DS_HEAD_KEY (`head_key`) and TF_DS_HEAD_CAND (`head_cand_on`) make a round's head each node's own bests or candidates,
/// which leave rd.logits unwritten (round.zig: headWhole runs only at head_k 0): the rows read would be stale memory, not
/// the round's (a quality gate scored such rows, NLL 11.75 = ln(vocabulary)). Turning both off changes no number
/// the model computes, only which node picks the token.
pub fn wholeRowsRefusal(reads_rows: bool, head_key: bool, head_cand_on: bool) ?[]const u8 {
    if (!reads_rows) return null;
    if (head_key) return "TF_DS_HEAD_KEY is on: a greedy round's head keeps each node's best and writes no whole rows; run the logits log with TF_DS_HEAD_KEY=0 on every rank";
    if (head_cand_on) return "TF_DS_HEAD_CAND is on: a sampled round's head keeps each node's candidates and writes no whole rows; run the logits log with TF_DS_HEAD_CAND=0 on every rank";
    return null;
}

/// TF_DS_NO_COPIES's text: "1" keeps a decode window off copied continuations (the lane core's copyProposal: the
/// drafter's drafts alone); unset or any other text: copies as served.
pub fn noCopies(text: ?[]const u8) bool {
    const v = text orelse return false;
    return std.mem.eql(u8, v, "1");
}

/// The matching tokens a copied continuation needs unless TF_DS_COPY_MATCH says otherwise (the lane core's enter_match).
pub const copy_match_default: i64 = 8;

/// TF_DS_COPY_MATCH's text: the matching tokens a copied continuation needs, 4 (the proposer's own least match, the lane
/// host's min_match) .. 64 (its longest); unset or empty: copy_match_default; null: not a setting (the start refuses).
pub fn copyMatch(text: ?[]const u8) ?i64 {
    const v = text orelse return copy_match_default;
    if (v.len == 0) return copy_match_default;
    const n = std.fmt.parseInt(i64, v, 10) catch return null;
    return if (n >= 4 and n <= 64) n else null;
}

/// A window's rows at most whose last row cannot overwrite a window key its first row reads: a row reads the 128 keys
/// to its own position, and the window rings (the target's and the drafter's) keep ring_extra slots past the window.
pub const copy_rows_most: u32 = 1 + prompt.ring_extra;

/// TF_DS_COPY_ROWS's text: a lone stream's copied window's rows at most (the lane core's copy_rows: its copies start at
/// 8 drafts and double, on rungs of 8, 12 and 16), past the drafter's window_rows .. copy_rows_most; unset, empty or
/// 0: 0, every window's window_rows; null: not a setting (the start refuses).
pub fn copyRows(text: ?[]const u8) ?u32 {
    const v = text orelse return 0;
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return 0;
    const n = std.fmt.parseInt(u32, v, 10) catch return null;
    return if (n > window_rows and n <= copy_rows_most) n else null;
}

/// The start's line for TF_DS_COPY_ROWS=`rows` (past window_rows): the rows, the drafts at most and the first.
pub fn copyRowsLine(buf: []u8, rows: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "TF_DS_COPY_ROWS={d}: a lone stream's copied continuation spans up to {d} rows ({d} drafts, from {d}; every other window {d})", .{ rows, rows, rows - 1, lanes.config.aloneFirst(rows - 1), window_rows });
}

/// The primitives rank 0 sends rank 1 (each frame: kind u8, step u64, its fields little-endian).
pub const Kind = enum(u8) { fill_begin = 1, fill_chunk = 2, verify = 3, pass = 4, snap_save = 5, snap_restore = 6, snap_drop = 7, fill_images = 8, done = 9, fill_resume = 10, rdma_idle = 11, snap_land = 12, head_whole = 13, fill_pair = 14, snap_sync = 15, cfg = 16 };

/// Whether a follower's frame leaves a standing Engram read-ahead alone: a prompt's own chunk and its snapshots always, a decode
/// round's frames (verify, pass, head_whole) with TF_DS_ENGRAM_KEEP_AHEAD; any other frame (a prompt about to change or go) joins it.
pub fn leavesAhead(kind: Kind, keep: bool) bool {
    return switch (kind) {
        .fill_chunk, .fill_pair, .snap_save, .snap_restore, .fill_images, .snap_land, .snap_sync => true,
        // a live change of a setting (cfg): the frame itself leaves the read-ahead alone; one that touches the worker says so (its flag)
        .cfg => true,
        .verify, .pass, .head_whole => keep,
        else => false,
    };
}

/// The landing's operations in a snap_land frame (land.Frame): an op byte, then its fields.
const LandOp = enum(u8) { open = 1, piece = 2, adopt = 3, abort = 4, probe = 5 };

pub fn writeLand(w: *Wire, gpa: std.mem.Allocator, f: land.Frame) !void {
    switch (f) {
        .open => |o| {
            try w.int(gpa, u8, @backingInt(LandOp.open));
            for ([_]u64{ o.id, o.base, o.at, o.total }) |v| try w.int(gpa, u64, v);
        },
        .piece => |p| {
            try w.int(gpa, u8, @backingInt(LandOp.piece));
            for ([_]u64{ p.id, p.off, p.len }) |v| try w.int(gpa, u64, v);
        },
        .adopt => |x| {
            try w.int(gpa, u8, @backingInt(LandOp.adopt));
            try w.int(gpa, u64, x.id);
        },
        .abort => |x| {
            try w.int(gpa, u8, @backingInt(LandOp.abort));
            try w.int(gpa, u64, x.id);
        },
        .probe => |p| {
            try w.int(gpa, u8, @backingInt(LandOp.probe));
            for ([_]u64{ p.id, p.base, p.at, p.seed }) |v| try w.int(gpa, u64, v);
            try w.int(gpa, u32, p.k);
        },
    }
}

pub fn readLand(r: *Reader) !land.Frame {
    return switch (std.enums.fromInt(LandOp, try r.int(u8)) orelse return error.BadFrame) {
        .open => .{ .open = .{ .id = try r.int(u64), .base = try r.int(u64), .at = try r.int(u64), .total = try r.int(u64) } },
        .piece => .{ .piece = .{ .id = try r.int(u64), .off = try r.int(u64), .len = try r.int(u64) } },
        .adopt => .{ .adopt = .{ .id = try r.int(u64) } },
        .abort => .{ .abort = .{ .id = try r.int(u64) } },
        .probe => .{ .probe = .{ .id = try r.int(u64), .base = try r.int(u64), .at = try r.int(u64), .seed = try r.int(u64), .k = try r.int(u32) } },
    };
}

/// The probe's device side on one rank: a host-synchronous copy for probe.digest, and the digests' all-gather.
pub const Probe = struct {
    m: *Model,
    dev: cuda.DeviceBuffer, // this rank's digest (32 bytes), then every rank's (32 bytes each)
    scratch: []u8, // host bytes for each copy (a piece of the kept buffer, one row)

    pub const scratch_bytes = 64 << 10;

    pub fn init(gpa: std.mem.Allocator, m: *Model) !Probe {
        var dev = try cuda.DeviceBuffer.alloc(m.ctx.d, 32 * (1 + @as(usize, m.world)));
        errdefer dev.free();
        return .{ .m = m, .dev = dev, .scratch = try gpa.alloc(u8, scratch_bytes) };
    }

    pub fn deinit(p: *Probe, gpa: std.mem.Allocator) void {
        p.dev.free();
        gpa.free(p.scratch);
    }

    pub fn d2h(p: *const Probe, dst: []u8, src: u64) !void {
        const d = p.m.ctx.d;
        try d.check(d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "cuMemcpyDtoH");
    }

    /// This rank's digest of kept state `buf` (0: this rank lacks it) and its extent rows, then every rank's into `out`
    /// by the all-gather (rank 0 first). A digest that cannot be made goes as zeros, so no rank skips the all-gather.
    pub fn gather(p: *Probe, buf: u64, planes: []const land.Plane, base: usize, at: usize, seed: u64, k: u32, out: [][32]u8) ![][32]u8 {
        const m = p.m;
        var mine: [32]u8 = @splat(0);
        if (buf != 0) {
            if (m.eng.s.synchronize()) |_| {
                mine = probe.digest(p, buf, m.snapBytes(), planes, base, at, seed, k, p.scratch) catch @splat(0);
            } else |_| {}
        }
        const d = m.ctx.d;
        const n: usize = m.world;
        if (out.len < n) return error.TooManyRanks;
        // on the model's stream, so the all-gather reads it after the copy (a synchronous copy runs on the legacy
        // stream, which does not order the model's non-blocking one)
        try m.eng.comm.gatherHost(d, p.dev.ptr, p.dev.ptr + 32, 32, m.eng.s);
        try m.eng.s.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(out.ptr), p.dev.ptr + 32, 32 * n), "cuMemcpyDtoH");
        return out[0..n];
    }
};

/// The SSD tier on rank 0: the prompt cache's Tier over this rank's planes (spill_tier.zig).
pub const SpillTier = spill_tier.Leader(Model.SpillDev);
/// A follower's store when every rank keeps its own (TF_DS_SSD_EVERY_RANK).
pub const FollowMirror = spill_mirror.Mirror(Model.SpillDev);

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
/// A drafter pass's frame after its header: the streams and steps, each stream's pending token, first position and
/// slot; with TF_DS_DRAFT_VOCAB's list in effect (`full` not null) one byte more, whether the pass takes the whole head.
pub const PassFrame = struct {
    n: usize,
    steps: usize,
    tokens: [max_streams]i64 = undefined,
    q0: [max_streams]i64 = undefined,
    slots: [max_streams]i64 = undefined,
    full: ?bool = null,

    pub fn write(p: *const PassFrame, w: *Wire, gpa: std.mem.Allocator) !void {
        try w.int(gpa, u32, @intCast(p.n));
        try w.int(gpa, u32, @intCast(p.steps));
        for (0..p.n) |i| {
            try w.int(gpa, i64, p.tokens[i]);
            try w.int(gpa, i64, p.q0[i]);
            try w.int(gpa, i64, p.slots[i]);
        }
        if (p.full) |f| try w.int(gpa, u8, @intFromBool(f));
    }

    /// The fields as write wrote them; `list`: the pruned list in effect, its byte last.
    pub fn read(r: *Reader, list: bool) !PassFrame {
        var p: PassFrame = .{ .n = try r.int(u32), .steps = try r.int(u32) };
        if (p.n == 0 or p.n > max_streams) return error.BadFrame;
        for (0..p.n) |i| {
            p.tokens[i] = try r.int(i64);
            p.q0[i] = try r.int(i64);
            p.slots[i] = try r.int(i64);
        }
        if (list) p.full = (try r.int(u8)) != 0;
        return p;
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

/// Whether one of the prompt cache's marks has its chunk end at `at` (each kept at the chunk end at or before it).
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
    // the third setting, 8,192 rows: a quiet 16-unit prompt (and its 128 decoder rows) from position 0 runs four passes
    // of 8,192, then the decoder rows' chunk; from pass_from 8,192 its four 2,048-row passes first, then three of 8,192
    const p8 = try checkPasses(16 * r + 128, 0, 4 * r, 0, &.{@intCast(4 * r)}, &.{}, false);
    try std.testing.expectEqual(@as(usize, 5), p8.len);
    try std.testing.expectEqual(@as(usize, 4), p8.long);
    const p8f = try checkPasses(16 * r + 128, 0, 4 * r, 4 * r, &.{@intCast(4 * r)}, &.{}, false);
    try std.testing.expectEqual(@as(usize, 8), p8f.len);
    try std.testing.expectEqual(@as(usize, 3), p8f.long);
}

test "the pass rule at every pass size from 2,048 to 16,384 rows keeps every row in the grid's own chunk, and a quiet prompt runs whole passes of the setting" {
    const r = model.chunk_rows;
    var prng = std.Random.DefaultPrng.init(0x16384);
    const rng = prng.random();
    var u: usize = 1;
    while (u <= 8) : (u += 1) {
        const pr = u * r;
        // every length to 20 resume units and 300 rows past, from position 0 and from 8,192, with and without the merged final pass,
        // the cap at the pass rows, then any whole number of units up to them at random from step to step
        var len: usize = 1;
        while (len <= 20 * r + 300) : (len += 37) for ([_]usize{ 0, 4 * r }) |from| for ([_]bool{ false, true }) |tail| {
            _ = try checkPasses(len, 0, pr, from, &.{@intCast(pr)}, &.{}, tail);
            var caps: [16]u32 = undefined;
            for (&caps) |*c| c.* = @intCast((1 + rng.uintLessThan(usize, u)) * r);
            _ = try checkPasses(len, 0, pr, from, &caps, &.{}, tail);
        };
        // a quiet 24-unit prompt and its 128 decoder rows from position 0: ceil(24 / u) passes of the setting, then the decoder rows' chunk
        const quiet = try checkPasses(24 * r + 128, 0, pr, 0, &.{@intCast(pr)}, &.{}, false);
        try std.testing.expectEqual((24 + u - 1) / u + 1, quiet.len);
        try std.testing.expectEqual(if (u == 1) @as(usize, 0) else (24 + u - 1) / u, quiet.long);
    }
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
    // a lone stream's rounds by their rows, and their time from the frame's send to the rows' draws (ns)
    lone_rounds: [max_rows + 1]u64 = @splat(0),
    lone_ns: [max_rows + 1]u64 = @splat(0),
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
    kept_restored: bool = false, // it resumed a state read back from disk (its drafter rings are checked at its end)

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
    slot: u32 = 0, // the slot it was kept from
    loading: bool = false, // a disk restore's destination: held back, not usable until every rank holds it
    restored: bool = false, // read back from the SSD tier
};

/// Kept states this long or longer say so in the log when they go stale (their prompts then run again from 0).
const note_rows = 32 * model.chunk_rows;

fn noteStale(sn: *const Snap, why: []const u8) void {
    if (sn.at >= note_rows) std.log.info("dsv41: the kept state of {d} tokens at {d} goes stale: {s}", .{ sn.at, sn.base, why });
}

/// How long a request waits for the live stream that holds the kept state it resumes from to end (then it resumes
/// in place and every kept state stays) before other kept states give their room for a copy of it.
const held_wait_ns: u64 = 10 * std.time.ns_per_s;

/// First tokens a handle names (each prompt's draw, read back right after its prefill).
const ring = 1024;

/// How a decode round's head logits reach rank 0's draws: the device's argmax (every row greedy, none masked), the
/// tokens merged from the nodes' bests (TF_DS_HEAD_KEY), the GPU's candidates, the candidates merged from the nodes'
/// parts (TF_DS_HEAD_CAND), candidates a flag (a NaN, a tie past the kernel's room) sent on to whole rows, or whole
/// rows to the host.
pub const HeadPath = enum(u3) { argmax, keys, candidates, merged, flagged, whole };

/// A round's HeadPath from its choices: `greedy` (its tokens), `keyed` (they are the nodes' bests merged: headTokens),
/// `cand` (candidates came back), `merged` (they are the nodes' parts merged: headCandidates), `whole` (roundLogits).
pub fn headPath(greedy: bool, keyed: bool, cand: bool, merged: bool, whole: bool) HeadPath {
    if (greedy) return if (keyed) .keys else .argmax;
    if (cand) return if (whole) .flagged else if (merged) .merged else .candidates;
    return .whole;
}

/// The verify frame's head byte (with TF_DS_HEAD_CAND or TF_DS_HEAD_KEY on every rank: Model.head_cand, head_key): 0
/// the whole rows, head_cand.key the nodes' bests (with `key`), the count each node selects (head_cand.kFor, with
/// `cand`); any other value refused.
pub fn readHead(r: *Reader, cand: bool, key: bool) !u8 {
    if (!cand and !key) return 0;
    const k = try r.int(u8);
    if (k == 0) return 0;
    if (key and k == head_cand.key) return k;
    if (cand and (k == head_cand.kFor(1) or k == head_cand.kFor(sampling.max_candidates))) return k;
    return error.BadFrame;
}

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
    // TF_DS_NO_COPIES / TF_DS_COPY_MATCH / TF_DS_COPY_ROWS (native.zig's start, rank 0's: it plans every window and the
    // followers run its frames): decode windows take copied continuations, of at least copy_match matching tokens, a
    // lone stream's up to copy_rows rows (0: window_rows) (facts)
    copies: bool = true,
    copy_match: i64 = copy_match_default,
    copy_rows: u32 = 0,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    snaps: std.ArrayList(*Snap) = .empty, // kept prompt states, oldest first
    next_snap: u64 = 1,
    // structured output: the prompt's last row of the last stream with a grammar prefilled (its first token may be
    // drawn again among the grammar's tokens), and a row with the disallowed tokens at -inf
    first_row: []f32 = &.{},
    first_of: ?*const lanes.Stream = null,
    masked: []f32 = &.{},
    spill: ?*SpillTier = null, // the SSD tier on rank 0 (native.zig opens it; null: off)
    lander: ?land.Lander = null, // rank 0 with followers and the tier on: the landing of a disk restore (startLanding)
    land_event: ?cuda.Event = null, // marks the landing's last piece (pacing: one piece in flight)
    land_planes: [4 * Model.spill.max_layers]land.Plane = undefined,
    land_n: usize = 0,
    probe: ?Probe = null, // rank 0 with followers and the tier on: the ranks' bytes compared (startLanding)
    probe_rows: u32 = 8, // TF_DS_SPILL_PROBE_ROWS
    land_time: spill_tier.prompt_cache.Timing = .{}, // the landing's steps that did work (the tier's land_piece time)
    lone_boundary: bool = false, // the tier part runs at a prompt chunk's boundary with no round (landIdle)
    probe_seed: u64 = 0x5eed_0f_d541,
    done_sent: bool = false, // the followers heard done (finish): no frame goes to them after it
    // rank 0's decode rounds by HeadPath (the server reads them at any time: /metrics' decode_rounds_by_head_path_total)
    head_rounds: [6]std.atomic.Value(u64) = @splat(.init(0)),
    // TF_DS_DRAFT_VOCAB's guard (rank 0's: the pass frame carries its call): the committed tokens outside the list
    guard: draft_vocab.Guard = .{},

    pub fn init(gpa: std.mem.Allocator, m: *Model) Lanes {
        var ln: Lanes = .{ .gpa = gpa, .m = m, .peer = if (m.world > 1) m.peer() else null };
        // the round costs by rows: ROUND_MS[min(rows, len) - 1] (the served table, or TF_DS_ROUND_MS)
        const t = m.round_ms;
        for (&ln.wc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        for (&ln.sc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        return ln;
    }

    /// The followers' last frame, done, sent once (by native.zig's finish after the lane host's engine thread has
    /// ended, or by deinit): each follower ends its follow loop and prints "followed": true.
    pub fn finish(self: *Lanes) void {
        if (self.done_sent or self.peer == null) return;
        self.m.unwatch(); // (the followers end on done: their links close)
        _ = self.begin(.done) catch |err| {
            std.log.warn("dsv41: orderly close: no done frame ({s})", .{@errorName(err)});
            self.done_sent = true;
            return;
        };
        self.done_sent = true;
        var told: usize = 0;
        for (self.m.followers(), 1..) |l, g| {
            if (l.?.send(self.wire.buf.items)) |_| told += 1 else |err| std.log.warn("dsv41: orderly close: rank {d} did not hear done ({s})", .{ g, @errorName(err) });
        }
        std.log.info("dsv41: orderly close: done to {d} follower{s}", .{ told, if (told == 1) "" else "s" });
    }

    /// Frees the host state and tells the followers done (unless finish did); the model is the caller's.
    pub fn deinit(self: *Lanes) void {
        for (self.snaps.items) |sn| {
            sn.buf.free();
            self.gpa.destroy(sn);
        }
        self.snaps.deinit(self.gpa);
        self.finish();
        for (&self.fills) |*f| if (f.*) |*x| x.deinit(self.gpa);
        self.streams.deinit(self.gpa);
        for (&self.seqs) |*q| q.deinit(self.gpa);
        self.wire.deinit(self.gpa);
        self.gpa.free(self.first_row);
        self.gpa.free(self.masked);
    }

    /// TF_DS_PREFILL_YIELD=1: the served host advances a prompt a chunk at a time (prefillStep), admitting the queued
    /// requests and running the live streams' rounds between its chunks. Else the whole prompt in one call, as served.
    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = if (yieldOn()) &vt_stepping else &vt_whole };
    }

    const vt_whole: be.Backend.VTable = .{
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
    const vt_stepping: be.Backend.VTable = blk: {
        var v = vt_whole;
        v.prefill_step = prefillStep;
        v.live_cfg = liveCfg;
        break :blk v;
    };

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
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = @intCast(self.m.eng.round_rows),
            .max_streams = @intCast(self.m.streams),
            .shared_costs = self.sc[0..self.m.eng.round_rows],
            .draft_probabilities = drafting,
            .draft_streams = drafting,
            .copies = self.copies,
            .enter_match = self.copy_match,
            .copy_rows = self.copy_rows,
        };
    }

    /// TF_DS_PREFILL_YIELD=1: Track D, a decode round of the live streams between a prompt's chunks, and in the served
    /// host the queued requests admitted between them too (0031's bounded prefill). Unset or any other value: the whole
    /// prompt first, as served.
    fn yieldOn() bool {
        const v = std.c.getenv("TF_DS_PREFILL_YIELD") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
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

    /// Begins a frame for the followers; false with one rank, or once they heard done (the kept states the prompt cache
    /// drops at the close then free rank 0's part alone).
    fn begin(self: *Lanes, kind: Kind) !bool {
        if (usesRdma(kind) and self.rdma_idle) {
            setRdmaIdle(self.m, false);
            self.rdma_idle = false;
        }
        if (self.peer == null or self.done_sent) return false;
        where.frame('S', @backingInt(kind), self.step);
        try self.wire.begin(self.gpa, kind, self.step);
        self.step += 1;
        return true;
    }

    fn send(self: *Lanes, kind: Kind) !void {
        if (try self.begin(kind)) try self.flush();
    }

    /// A sampled round's first `count` candidates a row: with the round's head by candidates (`head`, a node's count:
    /// TF_DS_HEAD_CAND) the nodes' parts merged (`merged` set); when the merge cannot rule out the baseline kernel flagging
    /// a row, or without it, the whole head (every rank: the head_whole frame) and the baseline kernel on it.
    fn candidates(self: *Lanes, R: usize, count: usize, head: u8, merged: *bool) !Model.Candidates {
        const m = self.m;
        if (head != 0) {
            if (try m.headCandidatesMerged(R, count, head)) |cd| {
                merged.* = true;
                return cd;
            }
            if (try self.begin(.head_whole)) {
                try self.wire.int(self.gpa, u32, @intCast(R));
                try self.flush();
            }
            try m.headWhole(R);
        }
        return m.roundCandidates(R, count);
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

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// The whole prompt (TF_DS_PREFILL_YIELD unset, and the standalone path); the served host with it set advances
    /// prefillStep one chunk at a time instead.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        while (!try prefillStep(ptr, s)) {
            const b = s.between orelse continue;
            const rounded = yieldOn() and b.ready(b.ptr);
            if (rounded) try b.run(b.ptr);
            // then the host's tier part, whenever it has work (pictures' passes too): its service runs no round and reads
            // no Engram. With no round here the stream is never idle at a boundary, so a landing may open anyway
            // (lone_boundary): its vote waits for this chunk's tail
            if (b.tier) |t| if (t.wanted(b.ptr)) {
                self.lone_boundary = !rounded;
                defer self.lone_boundary = false;
                t.run(b.ptr);
            };
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
        if (self.spill) |t| t.bumpSlot(slot);
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
            } else s.reuse_failed = true;
        }
        const base = if (kept != null and !moved) kept.?.base else try self.placeKept(taken[0..nt], size, kept);
        // read now: a keep or the tier part's adopt during the chunk loop may free the kept state
        const kept_restored = if (kept) |sn| sn.restored else false;
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
        // its rows ready (a gate's) or the tower's (rank 0), all of them sent to rank 1 after the fill's start
        var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
        errdefer spans.deinit(gpa);
        var rows: std.ArrayList(u8) = .empty;
        errdefer rows.deinit(gpa);
        if (s.lane_images.len > 0) {
            const image_id = m.cfg.image_token orelse return error.NoVision;
            const row_bytes = m.cfg.hidden * 2;
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
            if (base != sn.base) {
                if (self.spill) |t| t.bump(base, base + sn.at);
                try m.copyExtent(sn.base, base, sn.at);
            }
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
            .kept_restored = kept_restored,
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
        self.lone_boundary = false; // a chunk in flight: no boundary until it is launched
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
        s.step_pair = false;
        const rounds_now = if (hf) |x| x.rounds_beside else 0;
        // A whole pass is one the host's facts leave at the booted rows (stream.wholePass: at pass rows of 2,048 a cut pass has the same rows).
        // Beside generating users a pair forms only under TF_DS_ROUNDS_BESIDE (2N rounds follow it).
        if (pairMayForm(m.pair, decoding_now, rounds_now) and lanes.stream.wholePass(s.hostFacts(), f.start, prompt2d.pass_from_asked, prompt2d.pass_filling_asked, @intCast(m.pass_rows)) and n == m.pass_rows and f.start + n < len) {
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
        s.step_pair = n2 != 0; // (the host owes the streams twice the rounds after a pair)
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
            // Leave read-ahead running for a prompt alone; a slot switch joins it in startFill/selectFill. Without
            // TF_DS_PREFILL_YIELD no round runs between the chunks, so the read-ahead runs on, as served. A round joins it
            // unless TF_DS_ENGRAM_KEEP_AHEAD: the round's Engram buffers, reader ring and records are its own
            if (yieldOn() and !m.keep_ahead) if (s.between) |b| if (b.ready(b.ptr)) {
                m.joinAhead();
                m.fill_seq = &.{};
            };
            // a chunk boundary with no round yet: the host's tier part may open a landing here (landIdle), its vote
            // waiting for this chunk's tail; the next chunk or a round ends the boundary
            self.lone_boundary = true;
            return false;
        }
        m.joinAhead();
        m.fill_seq = &.{};
        if (!head) return error.NoPromptLogits;
        // a state read back from disk: the drafter's rings were rebuilt by the replay before the first draft
        if (self.spill != null and m.drafting() and f.kept_restored) try m.spillCheckResumed(lane.slot, len);
        const logits = try m.promptLogitsHost();
        if (logits_sha >= 1) {
            var dg: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(logits), &dg, .{});
            std.debug.print("{{\"prompt_logits\": \"{s}\", \"sha256\": \"{s}\"}}\n", .{ s.id, std.fmt.bytesToHex(dg, .lower) });
        }
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
        self.lone_boundary = false; // a round's launches: no prompt boundary
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
            // a lone stream's copied continuation (host tokens alone) may span copy_rows rows; every other window window_rows
            const most: usize = if (windows.len == 1 and w.held == 0 and self.copy_rows > window_rows) self.copy_rows else window_rows;
            if (rows > most) return error.WindowTooWide;
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
        // TF_DS_HEAD_KEY: a greedy round's head as each node's own bests; TF_DS_HEAD_CAND: a sampled round's as each
        // node's own first candidates (every rank told in the frame)
        const head: u8 = if (greedy and m.head_key) head_cand.key else if (m.head_cand and cand_count > 0) head_cand.kFor(cand_count) else 0;
        try m.checkTransport(); // (a failed engine sends no frame and runs no round)
        const t0 = m.now();
        where.set(.round, @intCast(windows.len), 0);
        defer where.set(.after_backend, 0, 0); // (the lane core's own work until the next backend call or the gate's next mark)
        defer m.flushDiag(false); // (TF_DS_ROCI_XLOG / TF_DS_NACT_LOG's files, every 256th call: nothing when both are off)
        if (try self.begin(.verify)) {
            try self.wire.int(gpa, u32, @intCast(windows.len));
            if (m.head_cand or m.head_key) try self.wire.int(gpa, u8, head);
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
        if (self.spill) |t| for (wins[0..windows.len]) |w| t.bump(w.base + w.start, w.base + w.start + w.ids.len);
        const rows = try self.built.build(gpa, &self.seqs, wins[0..windows.len]);
        // TF_DS_CAND_FIRST: the round's reads before the drafter's absorb (Model.verifyHead), read back after.
        // TF_DS_HEAD_CAND: with both on, the early reads are the device's merge, read back by headCandidatesMerged.
        if (m.cand_first) try m.verifyHead(rows, self.built.absorb[0..windows.len], if (greedy) .argmax else if (cand_count > 0) .{ .candidates = cand_count } else .none) else try m.verify(rows, self.built.absorb[0..windows.len], head);
        try m.checkTransport(); // (a round that ended on released waits delivers nothing)
        const keyed = head == head_cand.key;
        const tokens: []const u32 = if (!greedy) &.{} else if (keyed) try m.headTokens(rows.ids.len) else if (m.cand_first) try m.headArgmax(rows.ids.len) else try m.roundArgmax(rows.ids.len);
        var merged = false;
        const cand = if (cand_count > 0) try self.candidates(rows.ids.len, cand_count, head, &merged) else null;
        var whole = cand == null;
        if (cand) |cd| for (cd.flags) |fl| {
            whole = whole or fl != 0;
        };
        _ = self.head_rounds[@backingInt(headPath(greedy, keyed, cand != null, merged, whole))].fetchAdd(1, .monotonic);
        defer if (self.prof) |*p| {
            p.send += t1 - t0;
            p.enqueue += m.t_enqueue;
            p.forward += m.t_forward;
            p.absorb += m.t_absorb;
            p.logits += t3 - t2;
            p.sample += m.now() - t3;
            p.rounds += 1;
            p.rows += rows.ids.len;
            if (windows.len == 1) {
                p.lone_rounds[rows.ids.len] += 1;
                p.lone_ns[rows.ids.len] += m.now() - t0;
            }
        };
        const V = m.vocab;
        // the tools' --logits-sha 2: every row's logits, hashed (a greedy round reads them from the device for it)
        if (logits_sha >= 2) {
            const all = if (logits.len > 0) logits else try m.roundLogits(rows.ids.len);
            sha_round += 1;
            var line: [256]u8 = undefined;
            for (windows, 0..) |w, k| {
                const row0 = self.built.wins[k].row;
                for (0..w.rows()) |r| std.debug.print("{s}\n", .{try rowShaLine(&line, w.stream.id, sha_round, r, w.rows(), w.positions[r], ids[k][r], all[(row0 + r) * V ..][0..V])});
            }
        }
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
        const dr = &m.dr.?;
        const block = dr.n;
        // TF_DS_DRAFT_VOCAB: the guard counts every request's committed tokens, then calls this pass's head
        if (dr.vocab) |*v| {
            for (requests) |r| self.guard.note(&v.bits, r.follow);
            dr.vocab_full = self.guard.full;
            const st = &m.vocab_stats;
            st.tokens.store(self.guard.tokens, .monotonic);
            st.misses.store(self.guard.misses, .monotonic);
            st.full.store(self.guard.full, .monotonic);
        }
        var pf: PassFrame = .{ .n = 0, .steps = 0 };
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
            pf.tokens[n] = token;
            pf.q0[n] = @intCast(r.position - 1);
            pf.slots[n] = @intCast(l.slot);
            depths[n] = @min(r.depth, block);
            asked[n] = l;
            steps = @max(steps, depths[n]);
            n += 1;
        }
        if (n == 0) return;
        try m.checkTransport();
        pf.n = n;
        pf.steps = steps;
        if (dr.vocab != null) pf.full = dr.vocab_full;
        // a pass runs block rows a stream through the decode window's linears (prompt.grouped: 128 rows at most), so more
        // streams than 128 / block (25 at the 5-row block) draft in passes of that many, each its own wire message
        // (TensorFold ba03026c; the followers are unchanged)
        const per = @max(1, draft_pass_rows / block);
        var g0: usize = 0;
        while (g0 < n) : (g0 += per) {
            const g1 = @min(n, g0 + per);
            var sub: PassFrame = .{ .n = g1 - g0, .steps = steps, .full = pf.full };
            @memcpy(sub.tokens[0..sub.n], pf.tokens[g0..g1]);
            @memcpy(sub.q0[0..sub.n], pf.q0[g0..g1]);
            @memcpy(sub.slots[0..sub.n], pf.slots[g0..g1]);
            if (try self.begin(.pass)) {
                try sub.write(&self.wire, gpa);
                try self.flush();
            }
            const t0 = m.now();
            try m.pass(tokens[g0..g1], q0[g0..g1], slots[g0..g1], steps);
            try m.pass(sub.tokens[0..sub.n], sub.q0[0..sub.n], sub.slots[0..sub.n], steps);
            try m.checkTransport();
            if (dr.vocab != null) {
                const st = &m.vocab_stats;
                _ = (if (dr.vocab_full) &st.full_passes else &st.list_passes).fetchAdd(1, .monotonic);
            }
            if (self.prof) |*p| {
                p.pass += m.now() - t0;
                p.passes += 1;
                p.pass_streams += sub.n;
            }
            const d = dr;
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

    /// Whether a kept state still restores for a prompt of `len` tokens: live, not stale, a chunk end before the
    /// prompt's replay (the rows before it ran the encoder layers alone, as a fresh pass of this prompt runs them).
    fn usable(self: *const Lanes, sn: *const Snap, len: usize) bool {
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, @constCast(sn)) == null) return false;
        return !sn.stale and !sn.loading and sn.at > 0 and sn.at < len and sn.at % model.chunk_rows == 0 and sn.at <= len -| self.m.cfg.window;
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
                if (!sn.stale and !sn.loading and sn != source and !clashes(live, sn.base, sn.base + sn.at)) break sn;
            } else if (source) |sn| (if (!sn.stale and !clashes(live, sn.base, sn.base + sn.at)) sn else return error.ContextFull) else return error.ContextFull;
            noteStale(victim, "a placement needed its room (oldest first)");
            if (self.spill) |t| t.guardSnap(victim.id, 0); // its unpublished write stages or cancels before the rows go
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
            if (!sn.stale) {
                noteStale(sn, "a stream writes over its rows");
                if (self.spill) |t| t.guardSnap(sn.id, @intCast(@max(lo, sn.base) - sn.base));
            }
            sn.stale = true;
        };
    }

    /// The prompt cache's Snapshots.usable (with the SSD tier): a stale, loading or ineligible state is passed over
    /// before lengths are compared, so it never shadows a usable state or a durable point.
    pub fn usableFn(ptr: *anyopaque, saved: *anyopaque, len: u32) bool {
        return of(ptr).usable(@ptrCast(@alignCast(saved)), len);
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
        if (at == 0 or at % model.chunk_rows != 0 or at > s.prompt_len -| self.m.cfg.window) return error.NotAChunkEnd;
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = self.next_snap, .at = at, .base = l.base, .end = l.end, .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .slot = @intCast(l.slot) };
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

    pub fn snapDropFn(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        const i = std.mem.indexOfScalar(*Snap, self.snaps.items, sn) orelse return;
        _ = self.snaps.orderedRemove(i);
        if (self.spill) |t| t.dropSnap(sn.id); // waits first for a pending small-state capture of its buffer
        if (self.begin(.snap_drop) catch false) {
            self.wire.int(self.gpa, u64, sn.id) catch {};
            self.flush() catch |err| std.log.err("dsv41: rank 1 kept a dropped prompt state: {s}", .{@errorName(err)});
        }
        sn.buf.free();
        self.gpa.destroy(sn);
    }

    // -- the SSD tier's family (spill_tier.Family) ----------------------------------------------------------------

    pub fn spillFamily(self: *Lanes) spill_tier.Family {
        return .{ .ptr = self, .reserve = spillReserve, .install = spillInstall, .discard = spillDiscard, .info = spillInfo, .land = if (self.lander != null) spillLand else null, .probe = if (self.probe != null) spillProbe else null, .land_time = if (self.lander != null) &self.land_time else null, .land_counts = if (self.lander != null) &self.lander.?.counts else null, .sync_send = if (self.peer != null) syncSend else null, .sync_idle = landIdle };
    }

    /// With followers: the landing of disk restores in pieces of `piece_bytes` (TF_DS_SPILL_LAND_PIECE_MIB), and the
    /// probe; every follower made its pool from the start-up Settings' land_slots.
    pub fn startLanding(self: *Lanes, piece_bytes: usize, probe_rows: u32) !void {
        if (self.peer == null) return;
        self.land_n = (try self.m.landPlanes(&self.land_planes)).len;
        self.land_event = try cuda.Event.init(self.m.ctx.d, false);
        self.probe = try Probe.init(self.gpa, self.m);
        self.probe_rows = probe_rows;
        // each rank restores from its own store: rank 0 broadcasts no spans, so there is no lander
        _ = piece_bytes;
    }

    pub fn stopLanding(self: *Lanes) void {
        if (self.lander) |*l| l.abort();
        self.lander = null;
        if (self.land_event) |*e| e.deinit();
        self.land_event = null;
        if (self.probe) |*p| p.deinit(self.gpa);
        self.probe = null;
    }

    /// The start probe: `tokens` ids drawn from `seed` (a multiple of the prompt chunk) run through the encoder layers
    /// on every rank (no head, no reply) in slot 0 from the pool's start, kept under a probe id, the ranks' digests
    /// compared, then dropped. Busy while slot 0 or any stream is in use; never in the prompt cache. It checks the
    /// encoder half and the extent's rows; PROBE_EVERY re-checks real keeps.
    pub fn probeStart(self: *Lanes, tokens: usize, seed: u64) !spill_tier.ProbeVerdict {
        const m = self.m;
        const gpa = self.gpa;
        const p = &(self.probe orelse return .{ .failed = error.NoProbe });
        if (tokens == 0 or tokens % model.chunk_rows != 0 or tokens > m.pool_cap) return error.BadProbeTokens;
        const slot: usize = 0;
        if (self.used[slot] or self.streams.count() > 0) return .busy;
        const seq = &self.seqs[slot];
        try seq.resize(gpa, tokens);
        var prng: std.Random.DefaultPrng = .init(seed);
        for (seq.items) |*q| q.* = @intCast(prng.random().uintLessThan(usize, m.cfg.vocab));
        if (try self.begin(.fill_begin)) {
            try self.wire.int(gpa, u32, @intCast(slot));
            try self.wire.int(gpa, u64, 0);
            try self.wire.ids(gpa, seq.items);
            try self.flush();
        }
        try m.fillBegin(slot, 0);
        m.fill_seq = seq.items;
        defer {
            m.joinAhead();
            m.fill_seq = &.{};
        }
        var start: usize = 0;
        while (start < tokens) : (start += model.chunk_rows) {
            if (try self.begin(.fill_chunk)) {
                try self.wire.int(gpa, u64, start);
                try self.wire.int(gpa, u32, @intCast(model.chunk_rows));
                try self.wire.int(gpa, u64, tokens);
                try self.flush();
            }
            _ = try m.fillChunk(seq.items[0 .. start + model.chunk_rows], start, model.chunk_rows, tokens);
        }
        const id = self.next_snap;
        self.next_snap += 1;
        if (try self.begin(.snap_save)) {
            try self.wire.int(gpa, u64, id);
            try self.wire.int(gpa, u32, @intCast(slot));
            try self.flush();
        }
        var buf = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
        defer buf.free();
        defer if (self.begin(.snap_drop) catch false) {
            self.wire.int(gpa, u64, id) catch {};
            self.flush() catch {};
        };
        try m.snapCopy(slot, buf.ptr, true);
        try landFrame(self, .{ .probe = .{ .id = id, .base = 0, .at = tokens, .seed = seed, .k = self.probe_rows } });
        var out: [8][32]u8 = undefined;
        const got = try p.gather(buf.ptr, self.land_planes[0..self.land_n], 0, tokens, seed, self.probe_rows, &out);
        return if (probe.verdict(got)) |rank| .{ .differs = rank } else .same;
    }

    /// Family.probe: every rank's bytes of a kept state compared (the sampled re-check), only on an idle stream and
    /// with no landing in flight (busy otherwise). Its own seed each time: the rows it samples move.
    fn spillProbe(ptr: *anyopaque, saved: *anyopaque) spill_tier.ProbeVerdict {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        const p = &(self.probe orelse return .{ .failed = error.NoProbe });
        if (self.lander) |*l| if (l.inFlight()) return .busy;
        if (!(self.m.eng.s.done() catch false)) return .busy;
        if (sn.stale or sn.loading) return .{ .failed = error.ProbeStale };
        self.probe_seed +%= 0x9e37_79b9_7f4a_7c15;
        landFrame(self, .{ .probe = .{ .id = sn.id, .base = sn.base, .at = sn.at, .seed = self.probe_seed, .k = self.probe_rows } }) catch |e| return .{ .failed = e };
        var out: [8][32]u8 = undefined;
        const got = p.gather(sn.buf.ptr, self.land_planes[0..self.land_n], sn.base, sn.at, self.probe_seed, self.probe_rows, &out) catch |e| return .{ .failed = e };
        return if (probe.verdict(got)) |rank| .{ .differs = rank } else .same;
    }

    fn landHooks(self: *Lanes) land.Hooks {
        return .{ .ctx = self, .frame = landFrame, .vote = landVote0, .bcast = landBcast0, .idle = landIdle, .mark = landMark, .marked = landMarked, .wait = landWait };
    }

    fn landFrame(ctx: *anyopaque, f: land.Frame) anyerror!void {
        const self = of(ctx);
        if (!try self.begin(.snap_land)) return error.NoFollowers;
        try writeLand(&self.wire, self.gpa, f);
        try self.flush();
    }

    /// Backend.live_cfg: settings of class A changed live, at the host's loop top. Every rank applies them between the same two frames:
    /// rank 0 sends one cfg frame (kind 16: version, flags, then slot and value of each change, `unset` already the start's) and applies
    /// them itself. Nothing here touches the Engram worker: stopping it (joinAhead) ends it at the next (chunk, layer) step it reaches,
    /// and each step it has taken is one split read with the column partner (engram_partner.splitRead), so a stop at the moment each rank
    /// happens to apply a frame would leave the partners with different counts of exchanges (11-09 step 7 on the Sparks: ROCI failed, class
    /// rech). The flag only changes where the NEXT rounds join the worker, which every rank already does at the same frames (a verify,
    /// a pass or a head_whole frame, and rank 0's own round between chunks).
    fn liveCfg(ptr: *anyopaque, version: u64, items: []const lanes.backend.CfgItem) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        if (items.len > live_config.CfgFrame.max_items) return error.CfgTooLong;
        const held = self.poolHeld();
        m.kv_live_floor = held.end;
        m.kv_live_loading = held.loading; // (a live resize of the KV pool never goes under what a stream or a kept state holds)
        var resolved: [live_config.CfgFrame.max_items]live_config.CfgItem = undefined;
        const flags: u8 = 0; // (reserved)
        for (items, 0..) |it, i| resolved[i] = .{ .slot = it.slot, .value = if (it.value == live_config.cfg_unset) m.cfgStart(it.slot) else it.value };
        if (try self.begin(.cfg)) {
            var body: [live_config.CfgFrame.head_len + 5 * live_config.CfgFrame.max_items]u8 = undefined;
            try self.wire.buf.appendSlice(self.gpa, try live_config.CfgFrame.encode(&body, version, flags, resolved[0..items.len]));
            try self.flush();
        }
        try m.applyCfg(version, resolved[0..items.len]);
    }

    /// The end of the highest extent a stream or a kept state holds in the pool (a stale state's room is free).
    const PoolHeld = struct { end: usize, loading: u32 };
    fn poolHeld(self: *Lanes) PoolHeld {
        var h: PoolHeld = .{ .end = 0, .loading = 0 };
        var it = self.streams.valueIterator();
        while (it.next()) |l| h.end = @max(h.end, l.end);
        for (self.snaps.items) |sn| if (!sn.stale) {
            h.end = @max(h.end, sn.end);
            if (sn.loading) h.loading += 1; // a disk restore whose unpack has not been installed
        };
        return h;
    }

    /// One snap_sync frame to the other ranks: the bytes after the frame header are a sub-op and its fields (spill_wire).
    fn syncSend(ctx: *anyopaque, bytes: []const u8) anyerror!void {
        const self = of(ctx);
        if (!try self.begin(.snap_sync)) return error.NoFollowers;
        try self.wire.buf.appendSlice(self.gpa, bytes);
        try self.flush();
    }

    fn landVote0(ctx: *anyopaque, ok: bool) anyerror!bool {
        return of(ctx).m.eng.agreeAll(ok);
    }

    fn landBcast0(_: *anyopaque, _: []const land.Span) anyerror!void {
        return error.NoLanding; // (every rank restores from its own store: no span is broadcast)
    }

    /// The landing may open (its vote synchronizes): the model's stream idle, or a prompt pass's boundary with no
    /// round (lone_boundary: the vote then waits for the chunk in flight, which nothing else is waiting on).
    fn landIdle(ctx: *anyopaque) bool {
        const self = of(ctx);
        return self.lone_boundary or (self.m.eng.s.done() catch false);
    }

    fn landMark(ctx: *anyopaque) anyerror!void {
        const self = of(ctx);
        try self.land_event.?.record(self.m.eng.s);
    }

    fn landMarked(ctx: *anyopaque) bool {
        const e = of(ctx).land_event orelse return true;
        return e.done() catch true; // (a device error reads as done: the next call on the stream reports it)
    }

    fn landWait(ctx: *anyopaque) void {
        const e = of(ctx).land_event orelse return;
        e.synchronize() catch {};
    }

    /// Family.land: the restored Snap (rank 0 holds it, still loading) given to every follower, a step a poll; one
    /// landing at a time (another restore waits pending). Its spans: the Snap's buffer and its extent's rows.
    fn spillLand(ptr: *anyopaque, saved: *anyopaque, _: u64) spill_tier.LandStep {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        const l = &(self.lander orelse return .failed);
        if (!l.inFlight()) {
            l.begin(sn.id, sn.buf.ptr, self.m.snapBytes(), self.land_planes[0..self.land_n], sn.base, sn.at) catch return .failed;
        } else if (l.id != sn.id) return .pending;
        // timed when the step did work (an open and its vote, a piece, an end), not while it waits for a piece's event
        const before = l.counts;
        var t0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t0);
        const step = l.step();
        if (!std.meta.eql(before, l.counts)) {
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t1);
            const ns = (t1.sec - t0.sec) * std.time.ns_per_s + (t1.nsec - t0.nsec);
            self.land_time.observe(@intCast(@max(ns, 0)));
        }
        return step;
    }

    /// A restore's destination: an extent placed like a stream's (kept states give their room oldest first, guarded),
    /// a fresh Snap holding it back (loading: not usable until every rank holds it) and its buffer. `room`: the
    /// waiting prompt's length; when an extent of it and a growth margin fits without taking any kept state's room,
    /// the Snap's extent (`end`) holds that much, so the resume runs in place (no copyExtent of the restored rows).
    fn spillReserve(ptr: *anyopaque, at: usize, room: usize) anyerror!spill_tier.Dest {
        const self = of(ptr);
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        const want = @min(std.mem.alignForward(usize, @max(room, at) + growth_rows, model.chunk_rows), self.m.pool_cap);
        const roomy = if (want > at) try self.placeFree(taken[0..nt], want) else null;
        const base = roomy orelse try self.placeKept(taken[0..nt], at, null);
        self.staleOver(base, base + at, null);
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = self.next_snap, .at = at, .base = base, .end = base + (if (roomy != null) want else at), .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .loading = true };
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        return .{ .saved = sn, .id = sn.id, .base = base, .buf = sn.buf.ptr };
    }

    /// An extent of `size` around the live streams and the kept states' rows (placeSoft's rules: their held extents
    /// first), staling nobody: null when none.
    fn placeFree(self: *Lanes, live: []const [2]usize, size: usize) !?usize {
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(self.gpa);
        for (self.snaps.items) |sn| if (!sn.stale) try kept.append(self.gpa, .{ sn.base, sn.at, sn.end });
        return placeSoft(self.gpa, live, kept.items, size, self.m.pool_cap);
    }

    /// The restored state is held by every rank: usable from now.
    fn spillInstall(_: *anyopaque, saved: *anyopaque) void {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        sn.loading = false;
        sn.restored = true;
    }

    /// A restore that did not become ready: its Snap goes (a landing of it in flight is aborted first, its piece
    /// waited for, so no broadcast still reads the buffer; the followers then give theirs back).
    fn spillDiscard(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (self.lander) |*l| if (l.inFlight() and l.id == sn.id) l.abort();
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, sn)) |i| _ = self.snaps.orderedRemove(i);
        sn.buf.free();
        self.gpa.destroy(sn);
    }

    fn spillInfo(_: *anyopaque, saved: *anyopaque) spill_tier.SnapInfo {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return .{ .id = sn.id, .base = sn.base, .at = sn.at, .buf = sn.buf.ptr, .slot = sn.slot };
    }

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

/// A follower's landing hooks: the ranks' vote and rank 0's broadcast, on the engine's stream and communicator.
fn followerLand(m: *Model) land.Hooks {
    return .{ .ctx = m, .frame = noFrame, .vote = landVote, .bcast = landBcast, .idle = alwaysIdle, .mark = noMark, .marked = alwaysIdle, .wait = noWait };
}

fn landVote(ctx: *anyopaque, ok: bool) anyerror!bool {
    const m: *Model = @ptrCast(@alignCast(ctx));
    return m.eng.agreeAll(ok);
}

fn landBcast(_: *anyopaque, _: []const land.Span) anyerror!void {
    return error.NoLanding; // (every rank restores from its own store: no span is broadcast)
}

fn landAlloc(ctx: *anyopaque) anyerror!cuda.DeviceBuffer {
    const m: *Model = @ptrCast(@alignCast(ctx));
    return cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
}

fn landFree(_: *anyopaque, b: cuda.DeviceBuffer) void {
    var x = b;
    x.free();
}

fn noFrame(_: *anyopaque, _: land.Frame) anyerror!void {
    return error.FollowerSendsNoFrames;
}
fn alwaysIdle(_: *anyopaque) bool {
    return true;
}
fn noMark(_: *anyopaque) anyerror!void {}
fn noWait(_: *anyopaque) void {}

/// TF_DS_ROCI_STAGGER's pair forms only where no stream decodes, or, beside generating users, under TF_DS_ROUNDS_BESIDE: the host
/// owes the streams a count of rounds (2N follow a pair).
pub fn pairMayForm(pair_on: bool, decoding: bool, rounds_beside: u32) bool {
    if (!pair_on) return false;
    if (!decoding) return true;
    return rounds_beside != 0;
}

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

/// The follower's kept states as its store's Mirror reaches them: the `kept` map of device buffers by Snap id, and new
/// buffers for a restore's destination.
const FollowHost = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    kept: *std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer),
    /// Buffers made for restores in flight, by device address (the Mirror frees or adopts them by address).
    made: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty,

    fn me(c: *anyopaque) *FollowHost {
        return @ptrCast(@alignCast(c));
    }
    fn snapBuf(c: *anyopaque, id: u64) ?u64 {
        const b = me(c).kept.get(id) orelse return null;
        return b.ptr;
    }
    fn allocSnap(c: *anyopaque) anyerror!u64 {
        const h = me(c);
        try h.made.ensureUnusedCapacity(h.gpa, 1);
        const b = try cuda.DeviceBuffer.alloc(h.m.ctx.d, h.m.snapBytes());
        h.made.putAssumeCapacity(b.ptr, b);
        return b.ptr;
    }
    fn freeSnap(c: *anyopaque, buf: u64) void {
        const h = me(c);
        if (h.made.fetchRemove(buf)) |kv| {
            var b = kv.value;
            b.free();
        }
    }
    fn adoptSnap(c: *anyopaque, id: u64, buf: u64) anyerror!void {
        const h = me(c);
        const b = (h.made.fetchRemove(buf) orelse return error.UnknownBuffer).value;
        if (h.kept.fetchRemove(id)) |kv| {
            var old = kv.value;
            old.free();
        }
        try h.kept.put(h.gpa, id, b);
    }
    fn host(h: *FollowHost) spill_mirror.Host {
        return .{ .ctx = h, .snapBuf = snapBuf, .allocSnap = allocSnap, .freeSnap = freeSnap, .adoptSnap = adoptSnap };
    }
};

/// The follower's link, for the failure line's socket head (where.head_probe).
var probe_peer: ?link.Link = null;

fn headProbe() i32 {
    const p = probe_peer orelse return -1;
    return if (p.peekKind() catch null) |k| k else -1;
}

/// What a follower's frame handlers share: the rank, its buffers and where it is in rank 0's frame sequence.
const FollowCtx = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    peer: link.Link,
    buf: []u8,
    seqs: *[max_streams]std.ArrayList(i32),
    win_ids: *[max_streams]std.ArrayList(i32),
    built: *Built,
    step: *u64,
    rdma_idle: *bool,

    /// A decode round's frame (a verify, a whole head after candidates fell back, a drafter pass), run on this rank from the main loop.
    fn roundFrame(c: *FollowCtx, kind: Kind, r: *Reader) !void {
        switch (kind) {
            .verify => {
                const nw = try r.int(u32);
                if (nw == 0 or nw > max_streams) return error.BadFrame;
                const head = try readHead(r, c.m.head_cand, c.m.head_key);
                var wins: [max_streams]Win = undefined;
                for (0..nw) |k| {
                    const s: usize = try r.int(u32);
                    const base: usize = @intCast(try r.int(u64));
                    const end: usize = @intCast(try r.int(u64));
                    const start: usize = @intCast(try r.int(u64));
                    const absorb = try r.int(u8) != 0;
                    try r.ids(c.gpa, &c.win_ids[k]);
                    wins[k] = .{ .slot = s, .base = base, .end = end, .start = start, .absorb = absorb, .ids = c.win_ids[k].items };
                }
                const rows = try c.built.build(c.gpa, c.seqs, wins[0..nw]);
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
                const pf = try PassFrame.read(r, if (c.m.dr) |dr| dr.vocab != null else false);
                if (pf.full) |full| if (c.m.dr) |*dr| {
                    dr.vocab_full = full;
                };
                c.m.pass(pf.tokens[0..pf.n], pf.q0[0..pf.n], pf.slots[0..pf.n], pf.steps) catch |err| try refusedExchange(c.m, err);
            },
            else => return error.BadFrame,
        }
    }
};

/// Rank 1 (each of ranks 1-3 on the four-node split): the primitives rank 0 sends, run in order until it says done.
pub fn follow(gpa: std.mem.Allocator, m: *Model, mirror: ?*FollowMirror) !void {
    const peer = m.peer();
    probe_peer = peer;
    defer probe_peer = null;
    where.head_probe = headProbe;
    defer where.head_probe = null;
    // (with a store of its own: a keep's frame also carries the kept ids and, with pictures, the raw ids)
    const buf = try gpa.alloc(u8, (@max(m.pool_cap, m.pool_ceiling) + 1) * (if (mirror != null) @as(usize, 8) else 4) + (1 << 16)); // (a live resize may take the pool up to its ceiling)
    defer gpa.free(buf);
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    var win_ids: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&win_ids) |*q| q.deinit(gpa);
    var built: Built = .{};
    var step: u64 = 0;
    var rdma_idle = false;
    var slot: usize = 0;
    var bases: [max_streams]usize = @splat(0); // each slot's fill base (the generations the store mirrors are per extent)
    var ctx: FollowCtx = .{ .gpa = gpa, .m = m, .peer = peer, .buf = buf, .seqs = &seqs, .win_ids = &win_ids, .built = &built, .step = &step, .rdma_idle = &rdma_idle };
    var img_rows: std.ArrayList(u8) = .empty; // a prompt's image rows as their pieces come
    defer img_rows.deinit(gpa);
    var kept: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty; // rank 0's kept states, this rank's part
    defer {
        var ki = kept.valueIterator();
        while (ki.next()) |b| b.free();
        kept.deinit(gpa);
    }
    // the SSD tier's landings: rank 0's land_slots buffers made now (a landing past its vote never allocates), over
    // this rank's own planes in copyExtent's order; none when rank 0's tier is off (an open then votes not ready)
    var land_planes: [4 * Model.spill.max_layers]land.Plane = undefined;
    const lplanes: []const land.Plane = if (m.lead.land_slots == 0 or m.lead.land_slots > land.max_slots) &.{} else m.landPlanes(&land_planes) catch |err| blk: {
        std.log.err("dsv41: rank {d} cannot take landings ({s}): every disk restore is skipped", .{ m.rank, @errorName(err) });
        break :blk &.{};
    };
    // (with a store on every rank there is no landing: no landing buffers)
    var lf = try land.Follower(cuda.DeviceBuffer).init(gpa, .{ .ctx = m, .alloc = landAlloc, .free = landFree }, m.snapBytes(), lplanes, 0);
    defer lf.deinit(gpa);
    // the probe's buffers (rank 0's tier on): made now, so a probe frame never allocates before its all-gather
    var pr: ?Probe = if (m.lead.land_slots > 0) try Probe.init(gpa, m) else null;
    defer if (pr) |*x| x.deinit(gpa);
    var fh: FollowHost = .{ .gpa = gpa, .m = m, .kept = &kept };
    if (mirror) |mi| mi.host = fh.host();
    defer if (mirror) |mi| {
        // a clean end: the writes still queued staged and published within the stop's bound, the store closed
        _ = mi.side.flush(spill_tier.monotonicNs() +| mi.side.flush_ns);
        _ = mi.close();
    };
    while (true) {
        where.set(.main_loop, 0, 0);
        // the next primitive, polled for 20 ms before the read blocks (rank 0 sends it a fraction of a millisecond
        // after this rank's GPU goes idle in a decode; a sleeping thread would add its core's wake-up to every frame)
        // with the store busy (writes staging, a restore copying): the frame is waited for hot for a millisecond, then the store's
        // work is handed on, so a frame never waits behind it for more than that
        var frame: ?[]u8 = null;
        if (mirror) |mi| while (mi.busy()) {
            frame = try peer.recvSpinOrNone(buf, std.time.ns_per_ms);
            if (frame != null) break;
            mi.service();
        };
        var r: Reader = .{ .b = frame orelse if (rdma_idle) try peer.recv(buf) else try peer.recvSpin(buf, 20 * std.time.ns_per_ms) };
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
            12 => .snap_land,
            13 => .head_whole,
            14 => .fill_pair,
            15 => .snap_sync,
            16 => .cfg,
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
        if (!leavesAhead(kind, m.keep_ahead)) {
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
                bases[slot] = @intCast(base);
                if (mirror) |mi| mi.bumpSlot(slot);
                m.fillBegin(slot, @intCast(base)) catch |err| refused(err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_resume => {
                slot = try r.int(u32);
                const base = try r.int(u64);
                if (slot >= max_streams) return error.BadSlot;
                img_rows.clearRetainingCapacity();
                bases[slot] = @intCast(base);
                m.fillBegin(slot, @intCast(base)) catch |err| refused(err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_chunk => {
                const start: usize = @intCast(try r.int(u64));
                const n: usize = try r.int(u32);
                const replay: usize = @intCast(try r.int(u64));
                if (start + n > seqs[slot].items.len) return error.BadChunk;
                // a round between this prompt's chunks (Track D) joined the read-ahead and cleared its sequence: rank 0
                // puts its own back, and so does this rank (the same reads either way; the same bits)
                if (m.fill_seq.len == 0) m.fill_seq = seqs[slot].items;
                if (mirror) |mi| mi.bump(bases[slot] + start, bases[slot] + start + n);
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
            .verify, .head_whole, .pass => {
                where.set(.frame, @backingInt(kind), @intCast(step));
                try ctx.roundFrame(kind, &r);
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
                    m.setImages(spans.items, img_rows.items) catch |err| refused(err);
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
                    refused(err);
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
                if (from != to) {
                    if (mirror) |mi| mi.bump(to, to + at);
                    m.copyExtent(from, to, at) catch |err| refused(err);
                }
                m.snapCopy(s, b.ptr, false) catch |err| refused(err);
            },
            .snap_drop => {
                const id = try r.int(u64);
                if (mirror) |mi| mi.dropSnap(id); // (its write guarded, its small-state capture waited for)
                if (kept.fetchRemove(id)) |kv| {
                    var b = kv.value;
                    b.free();
                }
            },
            .snap_land => switch (try readLand(&r)) {
                .probe => |q| {
                    // every rank enters the all-gather, a rank without the state too (its zeros differ)
                    const p = &(pr orelse return error.NoProbe);
                    var out: [8][32]u8 = undefined;
                    const b: u64 = if (kept.get(q.id)) |x| x.ptr else 0;
                    _ = try p.gather(b, lplanes, @intCast(q.base), @intCast(q.at), q.seed, q.k, &out);
                },
                else => |fr| if (try lf.onFrame(followerLand(m), fr)) |a| {
                    // a landed state: kept under rank 0's id (snap_restore and snap_drop name it), the pool made whole
                    // again outside any collective
                    if (kept.fetchRemove(a.id)) |kv| {
                        var old = kv.value;
                        old.free();
                    }
                    try kept.put(gpa, a.id, a.buf);
                    lf.refill();
                },
            },
            .snap_sync => if (mirror) |mi| try mi.onSync(r.rest()),
            .cfg => {
                const d = try live_config.CfgFrame.decode(r.rest());
                var items: [live_config.CfgFrame.max_items]live_config.CfgItem = undefined;
                for (0..d.n) |i| items[i] = live_config.CfgFrame.item(d, i);
                try m.applyCfg(d.version, items[0..d.n]);
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

test "a pass frame: the streams, steps and each stream's token, position and slot; with the pruned list one byte more, its head" {
    const gpa = std.testing.allocator;
    var p: PassFrame = .{ .n = 2, .steps = 3 };
    p.tokens[0] = 11;
    p.tokens[1] = 129_279;
    p.q0[0] = 40;
    p.q0[1] = 7;
    p.slots[0] = 0;
    p.slots[1] = 15;
    // without the list: the baseline bytes (n, steps, then the streams' triples)
    var w: Wire = .{};
    defer w.deinit(gpa);
    try w.begin(gpa, .pass, 9);
    try p.write(&w, gpa);
    try std.testing.expectEqual(@as(usize, 1 + 8 + 4 + 4 + 2 * 24), w.buf.items.len);
    var r: Reader = .{ .b = w.buf.items[9..] };
    const got = try PassFrame.read(&r, false);
    try std.testing.expectEqual(@as(usize, 2), got.n);
    try std.testing.expectEqual(@as(usize, 3), got.steps);
    try std.testing.expectEqualSlices(i64, p.tokens[0..2], got.tokens[0..2]);
    try std.testing.expectEqualSlices(i64, p.q0[0..2], got.q0[0..2]);
    try std.testing.expectEqualSlices(i64, p.slots[0..2], got.slots[0..2]);
    try std.testing.expectEqual(@as(?bool, null), got.full);
    try std.testing.expectError(error.ShortFrame, r.int(u8));
    // with the list: its head last, both ways
    for ([_]bool{ false, true }) |full| {
        p.full = full;
        try w.begin(gpa, .pass, 9);
        try p.write(&w, gpa);
        try std.testing.expectEqual(@as(usize, 1 + 8 + 4 + 4 + 2 * 24 + 1), w.buf.items.len);
        var rl: Reader = .{ .b = w.buf.items[9..] };
        const gl = try PassFrame.read(&rl, true);
        try std.testing.expectEqual(@as(?bool, full), gl.full);
        try std.testing.expectEqualSlices(i64, p.tokens[0..2], gl.tokens[0..2]);
        try std.testing.expectError(error.ShortFrame, rl.int(u8));
    }
    // a frame of no streams or too many is refused
    var bad: Wire = .{};
    defer bad.deinit(gpa);
    try bad.int(gpa, u32, 0);
    try bad.int(gpa, u32, 1);
    var rb: Reader = .{ .b = bad.buf.items };
    try std.testing.expectError(error.BadFrame, PassFrame.read(&rb, false));
}

test "rank 0's drafter passes: TF_DS_DRAFT_VOCAB's guard counts every request's committed tokens first, then the pass and its frame take its head" {
    const src = @embedFile("lanes.zig");
    const at = std.mem.indexOf(u8, src, "    fn draftFn(ptr: *anyopaque") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    const note = std.mem.indexOf(u8, body, "for (requests) |r| self.guard.note(&v.bits, r.follow);") orelse return error.TestUnexpectedResult;
    const mode = std.mem.indexOf(u8, body, "dr.vocab_full = self.guard.full;") orelse return error.TestUnexpectedResult;
    const loop = std.mem.indexOf(u8, body, "for (requests) |r| {") orelse return error.TestUnexpectedResult;
    const frame = std.mem.indexOf(u8, body, "if (dr.vocab != null) pf.full = dr.vocab_full;") orelse return error.TestUnexpectedResult;
    const pass = std.mem.indexOf(u8, body, "try m.pass(sub.tokens[0..sub.n], sub.q0[0..sub.n], sub.slots[0..sub.n], steps);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(note < mode and mode < loop and loop < frame and frame < pass);
    // each pass counted by its head; the guard's counts copied out for /metrics
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "_ = (if (dr.vocab_full) &st.full_passes else &st.list_passes).fetchAdd(1, .monotonic);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "st.misses.store(self.guard.misses, .monotonic);"));
    // a follower takes its head from the frame
    const fol = std.mem.indexOf(u8, src, "            .pass => {") orelse return error.TestUnexpectedResult;
    const fb = src[fol .. fol + (std.mem.indexOf(u8, src[fol..], "\n            },\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, fb, "const pf = try PassFrame.read(r, if (c.m.dr) |dr| dr.vocab != null else false);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, fb, "if (pf.full) |full| if (c.m.dr) |*dr| {\n                    dr.vocab_full = full;"));
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
    m.pmem = .{}; // (and the memory-report thread: finish stops it)
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
    m.pmem = .{}; // (and the memory-report thread: finish stops it)
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
test "finish sends one done frame; nothing reaches a follower after it, and deinit sends no second done" {
    const gpa = std.testing.allocator;
    const c = std.c;
    var fds: [2]c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &fds)); // rank 0's end, the follower's
    var follower: link.Link = .{ .fd = fds[1] };
    defer follower.close();
    var m: Model = undefined;
    m.watch = .{}; // (finish and deinit end the watch)
    m.pmem = .{}; // (and the memory-report thread: finish stops it)
    m.world = 2;
    m.fan.links = .{ .{ .fd = fds[0] }, null, null };
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = m.peer() };
    const kept = struct {
        fn snap(ls: *Lanes, id: u64) !*Snap {
            const sn = try ls.gpa.create(Snap);
            sn.* = .{ .id = id, .at = 0, .base = 0, .buf = .{ .d = undefined, .ptr = 0, .len = 0 } };
            try ls.snaps.append(ls.gpa, sn);
            return sn;
        }
    };
    Lanes.snapDropFn(&ln, try kept.snap(&ln, 7)); // before the close: the follower hears the drop
    ln.finish();
    ln.finish();
    Lanes.snapDropFn(&ln, try kept.snap(&ln, 8)); // the prompt cache's drops at the close: rank 0's part only
    ln.deinit();
    m.fan.links[0].?.close(); // the follower then reads its end
    var buf: [64]u8 = undefined;
    var r: Reader = .{ .b = try follower.recv(&buf) };
    try std.testing.expectEqual(@as(u8, @backingInt(Kind.snap_drop)), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 0), try r.int(u64));
    try std.testing.expectEqual(@as(u64, 7), try r.int(u64));
    r = .{ .b = try follower.recv(&buf) };
    try std.testing.expectEqual(@as(u8, @backingInt(Kind.done)), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 1), try r.int(u64));
    try std.testing.expectError(error.PeerDown, follower.recv(&buf));
}

        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 },
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 },
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, // 9: the same kinds, the Rocinante-only Settings
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, // 10: and cfg, a live change of settings of class A
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, // 11: the same kinds, fill_pair carries the third pass's rows

test "a restore's destination is held back from placement, never a victim and never usable until installed" {
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.watch = .{}; // (finish and deinit end the watch)
    m.pmem = .{}; // (and the memory-report thread: finish stops it)
    m.pool_cap = 16 * c;
    m.cfg.window = 128;
    const gpa = std.testing.allocator;
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    defer ln.snaps.deinit(gpa);
    // a restore reading into [0, 4c), an ordinary kept state at [4c, 8c)
    var loading: Snap = .{ .id = 1, .at = 4 * c, .base = 0, .end = 4 * c, .buf = undefined, .loading = true };
    var kept: Snap = .{ .id = 2, .at = 4 * c, .base = 4 * c, .end = 8 * c, .buf = undefined };
    try ln.snaps.append(gpa, &loading);
    try ln.snaps.append(gpa, &kept);
    try std.testing.expect(!ln.usable(&loading, 8 * c)); // not until every rank installed it
    try std.testing.expect(ln.usable(&kept, 9 * c));
    // a placement that needs room past the free 8 chunks takes the ordinary kept state's, not the destination's
    const base = try ln.placeKept(&.{}, 12 * c, null);
    try std.testing.expect(kept.stale and !loading.stale);
    try std.testing.expectEqual(4 * c, base);
    // with nothing else to give, the destination still holds: the pool is full for this request
    kept.stale = false;
    kept.base = 4 * c;
    try std.testing.expectError(error.ContextFull, ln.placeKept(&.{.{ 4 * c, 16 * c }}, 4 * c, null));
    loading.loading = false;
    loading.restored = true;
    try std.testing.expect(ln.usable(&loading, 8 * c));
    try std.testing.expect(ln.spill == null);
}

test "a live KV resize sees a restore loading and the end of every extent held: a stale state's room is free" {
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.watch = .{};
    m.pmem = .{};
    m.pool_cap = 16 * c;
    const gpa = std.testing.allocator;
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    defer ln.snaps.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), ln.poolHeld().end);
    var loading: Snap = .{ .id = 1, .at = 4 * c, .base = 0, .end = 4 * c, .buf = undefined, .loading = true };
    var kept: Snap = .{ .id = 2, .at = 4 * c, .base = 4 * c, .end = 9 * c, .buf = undefined };
    try ln.snaps.append(gpa, &loading);
    try ln.snaps.append(gpa, &kept);
    var h = ln.poolHeld();
    try std.testing.expectEqual(@as(usize, 9 * c), h.end);
    try std.testing.expectEqual(@as(u32, 1), h.loading);
    kept.stale = true; // its room is free again
    h = ln.poolHeld();
    try std.testing.expectEqual(@as(usize, 4 * c), h.end);
    try std.testing.expectEqual(@as(u32, 1), h.loading);
    loading.loading = false; // installed on every rank
    try std.testing.expectEqual(@as(u32, 0), ln.poolHeld().loading);
}

test "a restore's destination holds the waiting prompt's room past its rows while that fits clear of the others" {
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.watch = .{}; // (finish and deinit end the watch)
    m.pmem = .{}; // (and the memory-report thread: finish stops it)
    m.pool_cap = 64 * c;
    m.cfg.window = 128;
    const gpa = std.testing.allocator;
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    defer ln.snaps.deinit(gpa);
    var kept: Snap = .{ .id = 1, .at = 4 * c, .base = 0, .end = 5 * c, .buf = undefined };
    try ln.snaps.append(gpa, &kept);
    // clear of the kept state's held extent and its growth: placed past it, nothing staled
    try std.testing.expectEqual(@as(?usize, 5 * c + growth_rows), try ln.placeFree(&.{}, 8 * c));
    try std.testing.expect(!kept.stale);
    // no room for the hint anywhere: null (spillReserve then places the rows alone, as placeKept does)
    try std.testing.expectEqual(@as(?usize, null), try ln.placeFree(&.{.{ 4 * c, 64 * c }}, 8 * c));
    try std.testing.expect(!kept.stale);
}

test "a snap_land frame's operations read back as written" {
    const gpa = std.testing.allocator;
    const frames = [_]land.Frame{
        .{ .open = .{ .id = 1 << 40, .base = 6144, .at = 131072, .total = 123_456_789 } },
        .{ .piece = .{ .id = 7, .off = 32 << 20, .len = 1 << 20 } },
        .{ .adopt = .{ .id = 9 } },
        .{ .abort = .{ .id = 11 } },
        .{ .probe = .{ .id = 13, .base = 2048, .at = 8192, .seed = 0xfeed_beef, .k = 8 } },
    };
    for (frames, 0..) |f, i| {
        var w: Wire = .{};
        defer w.deinit(gpa);
        try w.begin(gpa, .snap_land, i);
        try writeLand(&w, gpa, f);
        var r: Reader = .{ .b = w.buf.items };
        try std.testing.expectEqual(@as(u8, 12), try r.int(u8));
        try std.testing.expectEqual(@as(u64, i), try r.int(u64));
        try std.testing.expectEqual(f, try readLand(&r));
        try std.testing.expectError(error.ShortFrame, r.int(u8));
    }
    var bad: Wire = .{};
    defer bad.deinit(gpa);
    try bad.begin(gpa, .snap_land, 0);
    try bad.int(gpa, u8, 99);
    var r: Reader = .{ .b = bad.buf.items[9..] };
    try std.testing.expectError(error.BadFrame, readLand(&r));
}

test "TF_DS_NO_COPIES: \"1\" keeps copied continuations out of decode windows; unset, \"0\" or any other text: copies" {
    for ([_]?[]const u8{ null, "", "0", "yes", "11", " 1" }) |t| try std.testing.expect(!noCopies(t));
    try std.testing.expect(noCopies("1"));
}

test "TF_DS_COPY_MATCH: the matching tokens a copy needs, 4 .. 64; unset or empty 8; anything else no setting" {
    try std.testing.expectEqual(@as(?i64, 8), copyMatch(null));
    try std.testing.expectEqual(@as(?i64, 8), copyMatch(""));
    for ([_][]const u8{ "4", "8", "16", "64" }, [_]i64{ 4, 8, 16, 64 }) |t, n| try std.testing.expectEqual(@as(?i64, n), copyMatch(t));
    for ([_][]const u8{ "3", "65", "0", "-8", "8.5", "x", " 16" }) |t| try std.testing.expectEqual(@as(?i64, null), copyMatch(t));
}

test "the lane core takes rank 0's copy settings: facts carry Lanes.copies and copy_match, the start reads both" {
    const src = @embedFile("lanes.zig");
    const at = std.mem.indexOf(u8, src, "pub fn facts(self: *const Lanes) lanes.Model {").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, ".copies = self.copies,"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, ".enter_match = self.copy_match,"));
    const nat = @embedFile("native.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, nat, "const copies = !lanes_mod.noCopies(getenv(\"TF_DS_NO_COPIES\"));"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, nat, "st.lanes.copies = copies;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, nat, "const copy_match = lanes_mod.copyMatch(getenv(\"TF_DS_COPY_MATCH\")) orelse return error.BadCopyMatch;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, nat, "st.lanes.copy_match = copy_match;"));
    try std.testing.expect(std.mem.indexOf(u8, nat, "error.BadCopyMatch => \"TF_DS_COPY_MATCH: ") != null);
}

test "the leader decides a round's head reads before verify; with TF_DS_CAND_FIRST it verifies with them, then reads them back" {
    const src = @embedFile("lanes.zig");
    const at = std.mem.indexOf(u8, src, "    fn verifyFn(") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    var last: usize = 0;
    for ([_][]const u8{
        "var greedy = !masked;",
        "var cand_count: usize = 0;",
        "if (m.cand_first) try m.verifyHead(rows, self.built.absorb[0..windows.len], if (greedy) .argmax else if (cand_count > 0) .{ .candidates = cand_count } else .none) else try m.verify(rows, self.built.absorb[0..windows.len], head);",
        "const keyed = head == head_cand.key;\n        const tokens: []const u32 = if (!greedy) &.{} else if (keyed) try m.headTokens(rows.ids.len) else if (m.cand_first) try m.headArgmax(rows.ids.len) else try m.roundArgmax(rows.ids.len);",
        "const cand = if (cand_count > 0) try self.candidates(rows.ids.len, cand_count, head, &merged) else null;",
    }) |step| {
        const i = std.mem.indexOf(u8, body[last..], step) orelse return error.TestUnexpectedResult;
        last += i + step.len;
    }
}

test "TF_DS_COPY_ROWS: a lone stream's copied window's rows, past window_rows to copy_rows_most; unset, empty or 0 off; anything else no setting" {
    try std.testing.expectEqual(@as(?u32, 0), copyRows(null));
    try std.testing.expectEqual(@as(?u32, 0), copyRows(""));
    try std.testing.expectEqual(@as(?u32, 0), copyRows("0"));
    for ([_][]const u8{ "7", "9", "13", "17" }, [_]u32{ 7, 9, 13, 17 }) |t, n| try std.testing.expectEqual(@as(?u32, n), copyRows(t));
    for ([_][]const u8{ "6", "18", "00", "-17", "17.0", "x", " 17" }) |t| try std.testing.expectEqual(@as(?u32, null), copyRows(t));
    // a row reads the window's keys to its own position (the kit's _sparse_attn_part: p - (WIN - 1) .. p): the 17th row
    // of a window writes the slot of the key 128 before its first row, the one past that row's window; both rings keep
    // the same slots past the window
    try std.testing.expectEqual(@as(u32, 17), copy_rows_most);
    try std.testing.expectEqual(prompt.ring_extra, draft.ring_extra);
    try std.testing.expect(copy_rows_most <= max_rows);
}

test "TF_DS_COPY_ROWS's start line names the first width the lane core takes: min(8, rows - 1) drafts" {
    var buf: [256]u8 = undefined;
    for ([_]u32{ 7, 8, 9, 13, 17 }, [_][]const u8{ "(6 drafts, from 6;", "(7 drafts, from 7;", "(8 drafts, from 8;", "(12 drafts, from 8;", "(16 drafts, from 8;" }) |rows, want| {
        const line = try copyRowsLine(&buf, rows);
        try std.testing.expect(std.mem.indexOf(u8, line, want) != null);
        // the lane core's own first width for these rows (Config.init: wide_first up to copy_rows - 1)
        var c = try lanes.Config.init(std.testing.allocator, .{ .exact_width = window_rows, .drafts = window_rows - 1, .copy_rows = rows }, max_rows, max_rows - 1);
        defer c.deinit(std.testing.allocator);
        var first: [16]u8 = undefined;
        try std.testing.expect(std.mem.indexOf(u8, line, try std.fmt.bufPrint(&first, "from {d};", .{c.alone_first})) != null);
    }
}

test "the copy settings refuse a bad start before the weights load: TF_DS_COPY_ROWS and TF_DS_COPY_MATCH read before Model.open" {
    const nat = @embedFile("native.zig");
    const open_at = std.mem.indexOf(u8, nat, "st.m = try model.Model.open(gpa, io, ctx, .{").?;
    const rows_at = std.mem.indexOf(u8, nat, "lanes_mod.copyRows(getenv(\"TF_DS_COPY_ROWS\")) orelse return error.BadCopyRows;").?;
    const match_at = std.mem.indexOf(u8, nat, "lanes_mod.copyMatch(getenv(\"TF_DS_COPY_MATCH\")) orelse return error.BadCopyMatch;").?;
    try std.testing.expect(rows_at < open_at and match_at < open_at);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, nat, "lanes_mod.copyRows(getenv(\"TF_DS_COPY_ROWS\"))"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, nat, "lanes_mod.copyMatch(getenv(\"TF_DS_COPY_MATCH\"))"));
}

test "a lone wide window keeps every key its first row reads: copy_rows_most - 1 rows fit the slots each window ring keeps past WIN" {
    // a row reads WIN keys to its own position (the kit's _sparse_attn_part, launched with WIN = the window); a window's
    // row j writes the slot of the key ring - j before its first row: rows past the first fit ring - WIN of them
    var base = try cuda.aot.Manifest.parse(std.testing.allocator, @embedFile("fixtures/kit-drafter.json"));
    defer base.deinit();
    var extra = try cuda.aot.Manifest.parse(std.testing.allocator, @embedFile("fixtures/kit-drafter-extra.json"));
    defer extra.deinit();
    const window: u32 = 128; // the checkpoint's sliding window (config.zig pins it)
    var variants: usize = 0;
    for ([_]*const cuda.aot.Manifest{ &base, &extra }) |m| for (m.parsed.value.kernels) |k| {
        if (!std.mem.eql(u8, k.@"fn", "_sparse_attn_part") and !std.mem.eql(u8, k.@"fn", "_sparse_attn_part_u")) continue;
        try std.testing.expectEqual(@as(?i64, window), (k.consts.map.get("WIN") orelse return error.TestUnexpectedResult).int);
        variants += 1;
    };
    try std.testing.expect(variants >= 76);
    // the target's ring (round.zig) and the drafter's (draft.zig): the window and ring_extra slots past it
    const round_src = @embedFile("round.zig");
    try std.testing.expect(std.mem.indexOf(u8, round_src, "const ring_rows = c.window + ring_extra;") != null);
    try std.testing.expect(std.mem.indexOf(u8, round_src, "const ring_extra = prompt.ring_extra;") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("draft.zig"), ".ring = e.c.window + ring_extra,") != null);
    for ([_]u32{ window + prompt.ring_extra, window + draft.ring_extra }) |slots| try std.testing.expect(copy_rows_most - 1 <= slots - window);
    try std.testing.expectEqual(prompt.ring_extra, draft.ring_extra);
}

test "a lone stream's wide copies reach the lane core and the verify: facts carry copy_rows, the window's limit takes them, the start and the gates' tool read the setting" {
    const src = @embedFile("lanes.zig");
    const at = std.mem.indexOf(u8, src, "pub fn facts(self: *const Lanes) lanes.Model {").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, ".copy_rows = self.copy_rows,"));
    // a window past window_rows only alone, the backend holding none of its drafts (a copy: the host's tokens)
    const vat = std.mem.indexOf(u8, src, "    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {").?;
    const vbody = src[vat .. vat + std.mem.indexOf(u8, src[vat..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, vbody, "const most: usize = if (windows.len == 1 and w.held == 0 and self.copy_rows > window_rows) self.copy_rows else window_rows;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, vbody, "if (rows > most) return error.WindowTooWide;"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, vbody, "if (rows > window_rows)"));
    const nat = @embedFile("native.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, nat, "const copy_rows = lanes_mod.copyRows(getenv(\"TF_DS_COPY_ROWS\")) orelse return error.BadCopyRows;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, nat, "st.lanes.copy_rows = copy_rows;"));
    try std.testing.expect(std.mem.indexOf(u8, nat, "error.BadCopyRows => std.fmt.comptimePrint(\"TF_DS_COPY_ROWS: ") != null);
    const tool = @embedFile("lanes_main.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, tool, "ln.copy_rows = dsv41.lanes.copyRows(if (std.c.getenv(\"TF_DS_COPY_ROWS\")) |t| std.mem.span(t) else null) orelse return error.BadCopyRows;"));
    // the tool's other copy settings keep their defaults
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, tool, "TF_DS_COPY_MATCH"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, tool, "TF_DS_NO_COPIES"));
}

test "the tool's profile times a lone stream's rounds by their rows: verifyFn keeps them, tf-dsv41-lanes prints them" {
    const src = @embedFile("lanes.zig");
    const vat = std.mem.indexOf(u8, src, "    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {").?;
    const vbody = src[vat .. vat + std.mem.indexOf(u8, src[vat..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, vbody, "if (windows.len == 1) {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, vbody, "p.lone_rounds[rows.ids.len] += 1;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, vbody, "p.lone_ns[rows.ids.len] += m.now() - t0;"));
    const p: Profile = .{};
    try std.testing.expectEqual(@as(usize, max_rows + 1), p.lone_rounds.len);
    const tool = @embedFile("lanes_main.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, tool, "lone_round_ms_by_rows"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, tool, "for (p.lone_rounds, p.lone_ns, 0..) |n, ns, r| if (n > 0) {"));
}

test "--logits-sha: 0 off, 1 each prompt's logits, 2 each prompt's and every verified row's; anything else no setting" {
    for ([_][]const u8{ "0", "1", "2" }, [_]u8{ 0, 1, 2 }) |t, n| try std.testing.expectEqual(@as(?u8, n), logitsSha(t));
    for ([_][]const u8{ "", "3", "yes", " 1", "-1" }) |t| try std.testing.expectEqual(@as(?u8, null), logitsSha(t));
}

test "a verified row's logits line: its stream, round, row, window rows, position and input, and the sha256 of its bytes" {
    var row: [5]f32 = .{ 1.5, -2.0, 0.0, 3.25, -0.125 };
    var buf: [256]u8 = undefined;
    const line = try rowShaLine(&buf, "s7", 41, 3, 17, 4093, 1234, &row);
    const R = struct { row_logits: []const u8, round: u64, row: usize, rows: usize, position: u64, input: i64, sha256: []const u8 };
    const p = try std.json.parseFromSlice(R, std.testing.allocator, line, .{});
    defer p.deinit();
    try std.testing.expectEqualStrings("s7", p.value.row_logits);
    try std.testing.expectEqual(@as(u64, 41), p.value.round);
    try std.testing.expectEqual(@as(usize, 3), p.value.row);
    try std.testing.expectEqual(@as(usize, 17), p.value.rows);
    try std.testing.expectEqual(@as(u64, 4093), p.value.position);
    try std.testing.expectEqual(@as(i64, 1234), p.value.input);
    var dg: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(&row), &dg, .{});
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(dg, .lower), p.value.sha256);
    row[4] = -0.25; // one value's bits changed: another hash
    var buf2: [256]u8 = undefined;
    const other = try rowShaLine(&buf2, "s7", 41, 3, 17, 4093, 1234, &row);
    try std.testing.expect(std.mem.indexOf(u8, other, p.value.sha256) == null);
}

test "the verify hashes every row's logits under --logits-sha 2 alone; prompts from 1; the served start never sets it" {
    const src = @embedFile("lanes.zig");
    const vat = std.mem.indexOf(u8, src, "    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {").?;
    const vbody = src[vat .. vat + std.mem.indexOf(u8, src[vat..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, vbody, "if (logits_sha >= 2) {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, vbody, "rowShaLine("));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "if (logits_sha >= " ++ "1) {"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, @embedFile("native.zig"), "logits_sha"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, @embedFile("lanes_main.zig"), "dsv41.lanes.logits_sha = dsv41.lanes.logitsSha(val) orelse return error.BadLogitsSha;"));
}

test "a decode round's head path: argmax when greedy, the nodes' merged candidates or the GPU's unless a flag sent them on to whole rows, else whole" {
    try std.testing.expectEqual(HeadPath.argmax, headPath(true, false, false, false, false));
    try std.testing.expectEqual(HeadPath.keys, headPath(true, true, false, false, false));
    try std.testing.expectEqual(HeadPath.candidates, headPath(false, false, true, false, false));
    try std.testing.expectEqual(HeadPath.merged, headPath(false, false, true, true, false));
    try std.testing.expectEqual(HeadPath.flagged, headPath(false, false, true, false, true));
    try std.testing.expectEqual(HeadPath.whole, headPath(false, false, false, false, true));
    // the round counts its path once, right after its choices (the round's code comes before this test's text)
    const src = @embedFile("lanes.zig");
    const line = "_ = self.head_rounds[@backingInt(headPath(greedy, keyed, cand != null, merged, whole))].fetchAdd(1, .monotonic);";
    const at = std.mem.indexOf(u8, src, line) orelse return error.TestUnexpectedResult;
    try std.testing.expect(logits < at and at - logits < 200);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, src, line)); // the round's, and this test's text
    // merged: the nodes' parts came back merged (TF_DS_HEAD_CAND)
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "            if (try m.headCandidatesMerged(R, count, head)) |cd| {\n                merged.* = true;\n                return cd;\n            }"));
}

test "the verify frame's head byte: read only with TF_DS_HEAD_CAND, 0 or a node's count, anything else refused" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    try w.begin(gpa, .verify, 3);
    try w.int(gpa, u32, 2);
    try w.int(gpa, u8, head_cand.kFor(28));
    try w.int(gpa, u32, 5);
    var r: Reader = .{ .b = w.buf.items[9..] };
    try std.testing.expectEqual(@as(u32, 2), try r.int(u32));
    try std.testing.expectEqual(@as(u8, 32), try readHead(&r, true, false));
    try std.testing.expectEqual(@as(u32, 5), try r.int(u32)); // the windows after it
    // off: nothing read
    var off: Reader = .{ .b = &.{ 7, 0, 0, 0 } };
    try std.testing.expectEqual(@as(u8, 0), try readHead(&off, false, false));
    try std.testing.expectEqual(@as(u32, 7), try off.int(u32));
    // each switch's values with it alone, both with both; any other value refused
    const Case = struct { k: u8, cand: bool, key: bool, ok: bool };
    for ([_]Case{
        .{ .k = 0, .cand = true, .key = false, .ok = true },   .{ .k = 32, .cand = true, .key = false, .ok = true },
        .{ .k = 64, .cand = true, .key = false, .ok = true },  .{ .k = 1, .cand = true, .key = false, .ok = false },
        .{ .k = 0, .cand = false, .key = true, .ok = true },   .{ .k = 1, .cand = false, .key = true, .ok = true },
        .{ .k = 32, .cand = false, .key = true, .ok = false }, .{ .k = 64, .cand = false, .key = true, .ok = false },
        .{ .k = 1, .cand = true, .key = true, .ok = true },    .{ .k = 64, .cand = true, .key = true, .ok = true },
        .{ .k = 28, .cand = true, .key = true, .ok = false },  .{ .k = 33, .cand = true, .key = true, .ok = false },
        .{ .k = 65, .cand = true, .key = true, .ok = false },  .{ .k = 255, .cand = true, .key = true, .ok = false },
    }) |c| {
        var one: Reader = .{ .b = &.{c.k} };
        if (c.ok) try std.testing.expectEqual(c.k, try readHead(&one, c.cand, c.key)) else try std.testing.expectError(error.BadFrame, readHead(&one, c.cand, c.key));
    }
    var short: Reader = .{ .b = &.{} };
    try std.testing.expectError(error.ShortFrame, readHead(&short, true, false));
    // rank 0 writes it right after the windows' count, a follower reads it there; the fallback's frame carries the
    // round's rows, a follower's reads them and runs the whole head; it uses the rings (a parked ring wakes)
    const src = @embedFile("lanes.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"a decode round's head path") orelse return error.TestUnexpectedResult];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "            try self.wire.int(gpa, u32, @intCast(windows.len));\n            if (m.head_cand or m.head_key) try self.wire.int(gpa, u8, head);\n            for (wins[0..windows.len]) |w| {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "                if (nw == 0 or nw > max_streams) return error.BadFrame;\n                const head = try readHead(r, c.m.head_cand, c.m.head_key);\n"));
    // a greedy round's head by the nodes' bests (TF_DS_HEAD_KEY), its tokens from them
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "        const head: u8 = if (greedy and m.head_key) head_cand.key else if (m.head_cand and cand_count > 0) head_cand.kFor(cand_count) else 0;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "        const keyed = head == head_cand.key;\n        const tokens: []const u32 = if (!greedy) &.{} else if (keyed) try m.headTokens(rows.ids.len) else if (m.cand_first) try m.headArgmax(rows.ids.len) else try m.roundArgmax(rows.ids.len);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "            c.m.verify(rows, c.built.absorb[0..nw], head) catch |err| try refusedExchange(c.m, err);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "            if (try self.begin(.head_whole)) {\n                try self.wire.int(self.gpa, u32, @intCast(R));\n                try self.flush();\n            }\n            try m.headWhole(R);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "            13 => .head_whole,")); // (the main loop's kind table)
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "                const n: usize = try r.int(u32);\n                c.m.headWhole(n) catch |err| try refusedExchange(c.m, err);"));
    try std.testing.expect(usesRdma(.head_whole));
    try std.testing.expectEqual(@as(u8, 13), @backingInt(Kind.head_whole));
}

test "TF_DS_CAND_FIRST with TF_DS_HEAD_CAND or TF_DS_HEAD_KEY: the early read is the nodes' parts, and the whole-head launch is skipped" {
    const lanes_src = @embedFile("lanes.zig");
    // the verify frame carries the head byte with either switch, and a follower reads it there
    try std.testing.expect(std.mem.indexOf(u8, lanes_src, "if (m.head_cand or m.head_key) try self.wire.int(gpa, u8, head);") != null);
    try std.testing.expect(std.mem.indexOf(u8, lanes_src, "const head = try readHead(r, c.m.head_cand, c.m.head_key);") != null);
    // with CAND_FIRST on, the round's reads run early through verifyHead; the candidates come from self.candidates,
    // which merges the nodes' parts when the frame names a head (head != 0) and falls back to the baseline whole head
    const vf = std.mem.indexOf(u8, lanes_src, "    fn verifyFn(") orelse return error.TestUnexpectedResult;
    const vb = lanes_src[vf .. vf + (std.mem.indexOf(u8, lanes_src[vf..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expect(std.mem.indexOf(u8, vb, "if (m.cand_first) try m.verifyHead(rows, self.built.absorb[0..windows.len], if (greedy) .argmax else if (cand_count > 0) .{ .candidates = cand_count } else .none) else try m.verify(rows, self.built.absorb[0..windows.len], head);") != null);
    try std.testing.expect(std.mem.indexOf(u8, vb, "const cand = if (cand_count > 0) try self.candidates(rows.ids.len, cand_count, head, &merged) else null;") != null);
    // a greedy round with the key switch reads the nodes' bests; CAND_FIRST's whole-head argmax read is only the
    // fallback for a greedy round without it (never both: no second argmax launch)
    try std.testing.expect(std.mem.indexOf(u8, vb, "const tokens: []const u32 = if (!greedy) &.{} else if (keyed) try m.headTokens(rows.ids.len) else if (m.cand_first) try m.headArgmax(rows.ids.len) else try m.roundArgmax(rows.ids.len);") != null);
    const cf = std.mem.indexOf(u8, lanes_src, "    fn candidates(") orelse return error.TestUnexpectedResult;
    const cb = lanes_src[cf .. cf + (std.mem.indexOf(u8, lanes_src[cf..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expect(std.mem.indexOf(u8, cb, "if (try m.headCandidatesMerged(R, count, head)) |cd| {") != null);
    try std.testing.expect(std.mem.indexOf(u8, cb, "try m.headWhole(R);") != null);
    try std.testing.expect(std.mem.indexOf(u8, cb, "return m.roundCandidates(R, count);") != null);
    // Model.verifyHead: the head byte set for the forward's last stretch skips both whole-head launches; the event is
    // recorded after the forward (the head's kernel) and before the absorb, so the host's early read waits for the
    // head alone
    const model_src = @embedFile("model.zig");
    const vh = std.mem.indexOf(u8, model_src, "    pub fn verifyHead(") orelse return error.TestUnexpectedResult;
    const vhb = model_src[vh .. vh + (std.mem.indexOf(u8, model_src[vh..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, vhb, "if (hb == 0) try m.launchHead(head, R);"));
    try std.testing.expect(std.mem.indexOf(u8, vhb, "try m.forwardRows(rows);\n        if (hb == 0) try m.launchHead(head, R); // (hb != 0: round.forward's last stretch ran it)\n        try m.head_ready.?.record(m.stream);\n        try m.absorbRows(rows, absorb);") != null);
    // the head byte is the key for a greedy round (HEAD_KEY) or the nodes' count (HEAD_CAND); neither is 0
    try std.testing.expect(std.mem.indexOf(u8, vhb, ".argmax => if (m.head_key) head_cand.key else 0,") != null);
    try std.testing.expect(std.mem.indexOf(u8, vhb, ".candidates => |count| if (m.head_cand) head_cand.kFor(count) else 0,") != null);
}

test "TF_DS_ENGRAM_KEEP_AHEAD: a follower's decode-round frames leave the read-ahead alone with it and join it without; every other frame that is not the prompt's own still joins" {
    const own = [_]Kind{ .fill_chunk, .fill_pair, .snap_save, .snap_restore, .fill_images, .snap_land, .snap_sync, .cfg };
    for (own) |k| {
        try std.testing.expect(leavesAhead(k, false));
        try std.testing.expect(leavesAhead(k, true));
    }
    for ([_]Kind{ .verify, .pass, .head_whole }) |k| {
        try std.testing.expect(!leavesAhead(k, false));
        try std.testing.expect(leavesAhead(k, true));
    }
    // a prompt about to change or go, a slot switch, the end and the idle boundary: joined whatever the switch
    for ([_]Kind{ .fill_begin, .fill_resume, .snap_drop, .done, .rdma_idle }) |k| {
        try std.testing.expect(!leavesAhead(k, false));
        try std.testing.expect(!leavesAhead(k, true));
    }
}

test "TF_DS_ENGRAM_KEEP_AHEAD: rank 0 still joins the worker where a prompt changes or goes (startFill, selectFill's slot switch, the error path, release), and only the round's join is skipped" {
    const src = @embedFile("lanes.zig");
    const code = src[0 .. std.mem.indexOf(u8, src, "\ntest \"TF_DS_ENGRAM_KEEP_AHEAD: a follower's") orelse return error.TestUnexpectedResult];
    const sf = std.mem.indexOf(u8, code, "fn startFill(self: *Lanes, s: *lanes.Stream) !void {") orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, code[sf..][0..400], "m.joinAhead();") != null);
    const sel = std.mem.indexOf(u8, code, "fn selectFill(self: *Lanes, slot: usize, base: usize) !void {") orelse return error.TestUnexpectedResult;
    const sel_end = std.mem.indexOfPos(u8, code, sel, "m.fill_seq = self.seqs[slot].items;") orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, code[sel..sel_end], "if (self.filling_slot != slot) {\n            self.filling_slot = null;\n            m.joinAhead();") != null);
    const ps = std.mem.indexOf(u8, code, "fn prefillStep(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {") orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, code[ps..][0..700], "errdefer {\n            m.joinAhead();") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "if (self.filling_slot == slot) {\n                    self.m.joinAhead();") != null);
    // the round's join: skipped by the switch, in that one place
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "if (yieldOn() and !m.keep_ahead) if (s.between) |b|"));
    // the prompt's last chunk joins whatever the switch
    const last = std.mem.indexOf(u8, code, "        m.joinAhead();\n        m.fill_seq = &.{};\n        if (!head) return error.NoPromptLogits;") orelse return error.TestUnexpectedResult;
    try std.testing.expect(last > ps);
}

test "TF_DS_ENGRAM_KEEP_AHEAD: beside a round a pair forms only with TF_DS_ROUNDS_BESIDE and the host's step rows at the pass rows, and the kept worker's four slots hold both passes" {
    const resume_rows: u32 = model.chunk_rows;
    var decoding: lanes.stream.Facts = .{ .decoding = true, .filling = false, .waiting = false, .step_rows = resume_rows };
    for ([_]bool{ true, false }) |filling_pass| {
        // the default step rows (the resume rows): the pass is a single unit, never a pair's 2-unit pass
        const n = lanes.stream.passRows(decoding, 0, 0, 2 * resume_rows, resume_rows, filling_pass);
        try std.testing.expect(n != 2 * model.chunk_rows);
        // step rows at the pass rows with a count of rounds: the pass is whole and the pair may form
        decoding.step_rows = 2 * resume_rows;
        decoding.rounds_beside = 2;
        try std.testing.expectEqual(@as(u32, 2 * model.chunk_rows), lanes.stream.passRows(decoding, 0, 0, 2 * resume_rows, resume_rows, filling_pass));
        try std.testing.expect(pairMayForm(true, decoding.decoding, decoding.rounds_beside));
        try std.testing.expect(lanes.stream.wholePass(decoding, 0, 0, filling_pass, 2 * resume_rows));
        // a waiting request cuts the pass back to a unit: no pair
        decoding.waiting = true;
        try std.testing.expect(lanes.stream.passRows(decoding, 0, 0, 2 * resume_rows, resume_rows, filling_pass) != 2 * model.chunk_rows);
        decoding = .{ .decoding = true, .filling = false, .waiting = false, .step_rows = resume_rows };
    }
}

test "a pair of passes asks the pass rule's facts first: none forms while a stream decodes unless the host owes it a count of rounds" {
    const src = @embedFile("lanes.zig");
    const step = std.mem.indexOf(u8, src, "fn prefillStep(") orelse return error.TestUnexpectedResult;
    const pair = std.mem.indexOfPos(u8, src, step, "if (pairMayForm(m.pair, decoding_now, rounds_now) and lanes.stream.wholePass(s.hostFacts(), f.start, prompt2d.pass_from_asked, prompt2d.pass_filling_asked, @intCast(m.pass_rows)) and n == m.pass_rows") orelse return error.TestUnexpectedResult;
    const rule = std.mem.indexOfPos(u8, src, step, "lanes.stream.passRows(hf,") orelse return error.TestUnexpectedResult;
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
    for ([_][]const u8{ "c.m.verify(rows, c.built.absorb", "c.m.headWhole(n)", "c.m.pass(pf.tokens[0..pf.n], pf.q0" }) |call| {
        const at = std.mem.indexOf(u8, code, call) orelse return error.TestUnexpectedResult;
        const line_end = std.mem.indexOfPos(u8, code, at, "\n") orelse return error.TestUnexpectedResult;
        try std.testing.expect(std.mem.indexOf(u8, code[at..line_end], "refusedExchange(c.m, err)") != null);
    }
    try std.testing.expect(std.mem.indexOf(u8, code, "m.failExchanges();") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("model.zig"), "rc.t.fail(why, .own1, gone);") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("model.zig"), "m.failLanes(.refused, 0);") != null);
}

test "TF_DS_ROCI_STAGGER: a pair forms only where no stream decodes, or where the host owes the streams a count of rounds (TF_DS_ROUNDS_BESIDE)" {
    // a prompt alone: the pair forms
    try std.testing.expect(pairMayForm(true, false, 0));
    // decoders running: no pair without a count of rounds
    try std.testing.expect(!pairMayForm(true, true, 0));
    // no stagger configured: nothing ever pairs, with a count of rounds either
    try std.testing.expect(!pairMayForm(false, false, 0));
    try std.testing.expect(!pairMayForm(false, true, 3));
    // TF_DS_ROUNDS_BESIDE: beside generating users the pair forms
    try std.testing.expect(pairMayForm(true, true, 3));
    // the decision point is in the code, behind the pass rule's facts
    const code = @embedFile("lanes.zig");
    try std.testing.expect(std.mem.indexOf(u8, code, "if (pairMayForm(m.pair, decoding_now, rounds_now) and lanes.stream.wholePass(") != null);
}

test "a step reports whether it ran as a pair: reset on every step, set where the pair's frame is chosen" {
    const src = @embedFile("lanes.zig");
    const step = std.mem.indexOf(u8, src, "fn prefillStep(") orelse return error.TestUnexpectedResult;
    const reset = std.mem.indexOfPos(u8, src, step, "s.step_pair = false;") orelse return error.TestUnexpectedResult;
    const pair = std.mem.indexOfPos(u8, src, step, "if (pairMayForm(m.pair, decoding_now") orelse return error.TestUnexpectedResult;
    const set = std.mem.indexOfPos(u8, src, step, "s.step_pair = n2 != 0;") orelse return error.TestUnexpectedResult;
    const frame = std.mem.indexOfPos(u8, src, step, "try self.begin(.fill_pair)") orelse return error.TestUnexpectedResult;
    try std.testing.expect(reset < pair and pair < set and set < frame);
}

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
    // a draft's chance is draft.landing: the sigmoid of the calibrated confidence (conf_cal, default {1.0, 0.19}), so a
    // confidence of 0 lands 0.547, not 0.5; the expectation is built from the calibration the engine runs with
    const cal = draft.conf_cal;
    const at = struct {
        fn f(c: f64) f64 {
            return 1.0 / (1.0 + @exp(-(draft.conf_cal[0] * c + draft.conf_cal[1])));
        }
    }.f;
    var out: [3]f64 = undefined;
    survivals(&.{ 0.0, 0.0, 2.0 }, &out);
    try std.testing.expectApproxEqAbs(at(0.0), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(at(0.0) * at(0.0), out[1], 1e-12);
    try std.testing.expectApproxEqAbs(at(0.0) * at(0.0) * at(2.0), out[2], 1e-12);
    // with the identity calibration the chances are the plain sigmoids (0.5, 0.25, 0.25 / (1 + e^-2))
    draft.conf_cal = .{ 1.0, 0.0 };
    defer draft.conf_cal = cal;
    survivals(&.{ 0.0, 0.0, 2.0 }, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), out[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.25 / (1.0 + @exp(@as(f64, -2.0))), out[2], 1e-12);
}

test "a cfg frame is kind 16 behind the frame header, reads back whole, and its sender sends in one frame, then applies on rank 0" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    try w.begin(gpa, .cfg, 41);
    var body: [live_config.CfgFrame.head_len + 5 * live_config.CfgFrame.max_items]u8 = undefined;
    const items = [_]live_config.CfgItem{.{ .slot = @backingInt(live_config.Slot.keep_ahead), .value = 1 }};
    try w.buf.appendSlice(gpa, try live_config.CfgFrame.encode(&body, 9, 0, &items));
    var r: Reader = .{ .b = w.buf.items };
    try std.testing.expectEqual(@as(u8, 16), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 41), try r.int(u64));
    const d = try live_config.CfgFrame.decode(r.rest());
    try std.testing.expectEqual(@as(u64, 9), d.version);
    try std.testing.expectEqual(items[0], live_config.CfgFrame.item(d, 0));
    // the frame with its header is one send() of the link (frames up to 2,048 bytes are written whole)
    try std.testing.expect(w.buf.items.len <= 2048);
    // the sender's order: the frame is flushed before rank 0 applies; and neither side ever stops the Engram worker for a cfg frame (step 7's Spark failure)
    const code = @embedFile("lanes.zig");
    const at = std.mem.indexOf(u8, code, "fn liveCfg(ptr: *anyopaque").?;
    const end = std.mem.indexOfPos(u8, code, at, "/// One snap_sync frame").?;
    const f = code[at..end];
    const begin = std.mem.indexOf(u8, f, "if (try self.begin(.cfg)) {").?;
    const flush = std.mem.indexOf(u8, f, "try self.flush();").?;
    const apply = std.mem.indexOf(u8, f, "try m.applyCfg(version, resolved[0..items.len]);").?;
    try std.testing.expect(begin < flush and flush < apply);
    try std.testing.expect(std.mem.indexOf(u8, f, "joinAhead();") == null and std.mem.indexOf(u8, f, "fill_seq") == null);
    // the follower: the frame joins the worker only on the flag, then applies; and the backend offers it
    const fol = std.mem.indexOf(u8, code, ".cfg => {\n                const d = try live_config.CfgFrame.decode(r.rest());").?;
    const fend = std.mem.indexOfPos(u8, code, fol, ".rdma_idle => {").?;
    const fb = code[fol..fend];
    try std.testing.expect(std.mem.indexOf(u8, fb, "try m.applyCfg(d.version, items[0..d.n]);") != null);
    try std.testing.expect(std.mem.indexOf(u8, fb, "joinAhead") == null and std.mem.indexOf(u8, fb, "fill_seq") == null);
    try std.testing.expect(std.mem.indexOf(u8, code, "v.live_cfg = liveCfg;") != null);
    // a cfg frame is no round's: no RDMA wake is asked for it
    try std.testing.expect(!usesRdma(.cfg));
    try std.testing.expect(leavesAhead(.cfg, false) and leavesAhead(.cfg, true));
}

test "a run that reads whole rows on the host (--logits-log, --logits-sha 2) is refused while the head is sharded: rd.logits would be unwritten" {
    try std.testing.expect(wholeRowsRefusal(false, true, true) == null); // nothing read: any head
    try std.testing.expect(wholeRowsRefusal(true, false, false) == null); // the baseline whole head
    try std.testing.expect(std.mem.indexOf(u8, wholeRowsRefusal(true, true, false).?, "TF_DS_HEAD_KEY=0") != null);
    try std.testing.expect(std.mem.indexOf(u8, wholeRowsRefusal(true, false, true).?, "TF_DS_HEAD_CAND=0") != null);
    // tf-dsv41-lanes asks before it opens the log
    const src = @embedFile("lanes_main.zig");
    const ask = std.mem.indexOf(u8, src, "dsv41.lanes.wholeRowsRefusal(log_path != null or dsv41.lanes.logits_sha >= 2, m.head_key, m.head_cand)") orelse return error.NotAsked;
    try std.testing.expect(ask < std.mem.indexOf(u8, src, "sink = try dsv41.logits_log.Sink.init(").?);
}
