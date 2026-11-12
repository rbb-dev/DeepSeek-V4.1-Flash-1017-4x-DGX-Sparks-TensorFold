//! TF_DS_DRAFT_VOCAB: the drafter's head and Markov loop on a pruned vocabulary of 32,768 ids, 8,192 a node
//! (draft_vocab.bin: u32 little-endian, ascending; node r takes positions [8,192 r, 8,192 (r + 1)): every node the same
//! rows, the ids in vocabulary order, so a tie between nodes falls to the lower id as on the whole vocabulary). The head
//! is split by vocabulary quarters, so a node takes the rows of its ids from the nodes that hold them, once at the
//! start. A draft is only a proposal the target verifies: replies are the target's, and a draft is the whole
//! vocabulary's wherever that one's best token is in the list. The guard counts the committed tokens outside the list;
//! while more than 2% of the last 4,096 are, the passes run the whole head (kept loaded), back on the list under 1%.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const tri = @import("tri.zig");
const tri_markov = @import("tri_markov.zig");
const draft2d = @import("draft2d.zig");

/// The ids a node takes, the nodes, the list's ids.
pub const per_node = 8192;
pub const nodes = 4;
pub const count = per_node * nodes;

const bin = @embedFile("draft_vocab.bin");

/// The list's id at position i (i < count).
pub fn id(i: usize) u32 {
    return std.mem.readInt(u32, bin[4 * i ..][0..4], .little);
}

/// The special tokens the model writes, in the list always: every added token of the checkpoint's tokenizer but its
/// placeholders (the begin, end and pad tokens, the chat roles, the tool-call, thinking and DSML markers, the search
/// and image markers).
pub const forced = [_]u32{
    0,      1,      2,      128799, 128800, 128801, 128802, 128803, 128804, 128805, 128806, 128807, 128808, 128809, 128810,
    128811, 128812, 128813, 128814, 128815, 128816, 128817, 128818, 128819, 128820, 128821, 128822, 128825, 128826, 128827,
    128828, 128829, 128830, 128831, 128832, 128833, 128834, 128835, 128836, 128837, 128838, 128839, 128840, 128841, 128842,
    128843, 128844, 128845, 128846, 129262, 129263, 129264, 129271, 129272, 129273, 129274, 129275, 129276, 129277, 129278,
};

/// Why the list cannot serve a vocabulary of `vocab` ids (null: it can): count ids, ascending, each below vocab, every
/// forced token in it.
pub fn check(vocab: usize) ?[]const u8 {
    return listCheck(bin, vocab);
}

fn listCheck(b: []const u8, vocab: usize) ?[]const u8 {
    if (b.len != count * 4) return "the list is not 32,768 ids";
    var prev: ?u32 = null;
    for (0..count) |i| {
        const x = std.mem.readInt(u32, b[4 * i ..][0..4], .little);
        if (prev) |p| if (x <= p) return "the list's ids are not ascending";
        if (x >= vocab) return "the list names an id past the vocabulary";
        prev = x;
    }
    for (forced) |f| if (findIn(b, f) == null) return "a special token the model writes is not in the list";
    return null;
}

/// The list's position of `tok`; null: not in the list.
pub fn find(tok: u32) ?usize {
    return findIn(bin, tok);
}

fn findIn(b: []const u8, tok: u32) ?usize {
    const at = lowerIn(b, tok);
    if (at == count) return null;
    return if (std.mem.readInt(u32, b[4 * at ..][0..4], .little) == tok) at else null;
}

/// The first position whose id is `v` or more (count: none).
fn lowerIn(b: []const u8, v: usize) usize {
    var lo: usize = 0;
    var hi: usize = count;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (std.mem.readInt(u32, b[4 * mid ..][0..4], .little) < v) lo = mid + 1 else hi = mid;
    }
    return lo;
}

/// The positions [s, e) of the list's ids in [lo, hi) (the list ascending: one run).
pub fn run(lo: usize, hi: usize) [2]usize {
    return .{ lowerIn(bin, lo), lowerIn(bin, hi) };
}

/// Node r's positions.
pub fn nodeRange(r: usize) [2]usize {
    return .{ r * per_node, (r + 1) * per_node };
}

/// The positions both ranges hold (empty, at the later start, when they do not meet).
pub fn overlap(a: [2]usize, b: [2]usize) [2]usize {
    const lo = @max(a[0], b[0]);
    return .{ lo, @max(lo, @min(a[1], b[1])) };
}

/// The start's exchange of head rows for node `me`: per node j, the positions it sends j (its own quarter's ids in j's
/// range; j == me its own rows) and the positions j sends it (j's quarter's ids in me's range). `quarters`: each node's
/// head rows [lo, hi) of the vocabulary; they tile [0, vocab).
pub const Plan = struct {
    send: [nodes][2]usize,
    recv: [nodes][2]usize,

    /// The head rows this node takes from other nodes.
    pub fn fromOthers(p: Plan, me: usize) usize {
        var n: usize = 0;
        for (p.recv, 0..) |x, j| if (j != me) {
            n += x[1] - x[0];
        };
        return n;
    }
};

/// The plan for node `me` of these quarters; error.QuartersDoNotTile unless they cover [0, vocab) once, in some order.
pub fn plan(me: usize, quarters: [nodes][2]usize, vocab: usize) !Plan {
    var covered: usize = 0;
    for (quarters, 0..) |q, j| {
        if (q[0] >= q[1] or q[1] > vocab) return error.QuartersDoNotTile;
        for (quarters[j + 1 ..]) |o| if (q[0] < o[1] and o[0] < q[1]) return error.QuartersDoNotTile;
        covered += q[1] - q[0];
    }
    if (covered != vocab) return error.QuartersDoNotTile;
    var p: Plan = undefined;
    const mine = run(quarters[me][0], quarters[me][1]);
    for (0..nodes) |j| {
        p.send[j] = overlap(nodeRange(j), mine);
        p.recv[j] = overlap(nodeRange(me), run(quarters[j][0], quarters[j][1]));
    }
    return p;
}

/// The start's exchange rides the host links (comm.gatherHost: every rank's `chunk` bytes gathered to all four, a round at
/// a time; the engine has no other collective). Each node's export is one blob: for every other node m in rank order, the
/// rows m takes from this node's quarter, their codes (`cb` bytes a row) then their scales (`sb` a row). Round r gathers
/// every node's bytes [r chunk, (r + 1) chunk) of its blob (zeros past its end: the ranks go in step, `rounds` of them),
/// and each receiver copies what its sender addressed to it from the gathered chunk into its list's head.
pub const chunk: usize = 16 * 1024;

/// Where node m's part sits in sender j's blob: the codes' and the scales' offsets, and the rows.
pub const Part = struct { codes: usize, scales: usize, rows: usize };

fn vocabWidth(quarters: [nodes][2]usize) usize {
    var v: usize = 0;
    for (quarters) |q| v += q[1] - q[0];
    return v;
}

/// Sender j's part for node m (m != j; the rows it holds of m's range of the list).
pub fn blobPart(quarters: [nodes][2]usize, j: usize, m: usize, cb: usize, sb: usize) !Part {
    const pj = try plan(j, quarters, vocabWidth(quarters));
    var at: usize = 0;
    for (0..nodes) |x| {
        if (x == j) continue;
        const rows = pj.send[x][1] - pj.send[x][0];
        if (x == m) return .{ .codes = at, .scales = at + rows * cb, .rows = rows };
        at += rows * (cb + sb);
    }
    return error.NoSuchPart;
}

/// Sender j's blob in bytes.
pub fn blobBytes(quarters: [nodes][2]usize, j: usize, cb: usize, sb: usize) !usize {
    const pj = try plan(j, quarters, vocabWidth(quarters));
    var at: usize = 0;
    for (0..nodes) |x| {
        if (x != j) at += (pj.send[x][1] - pj.send[x][0]) * (cb + sb);
    }
    return at;
}

/// The exchange's rounds: the longest blob's chunks.
pub fn rounds(quarters: [nodes][2]usize, cb: usize, sb: usize) !usize {
    var most: usize = 0;
    for (0..nodes) |j| most = @max(most, try blobBytes(quarters, j, cb, sb));
    return (most + chunk - 1) / chunk;
}

/// One copy from a gathered chunk into the list's head.
pub const Copy = struct { src: usize, codes: bool, dst: usize, len: usize };

/// Round `round`'s copies on node `me` from sender j's chunk: its part's codes and scales that fall in the round's window
/// (src: the byte in j's chunk; dst: the byte in the head's codes or scales).
pub fn landings(quarters: [nodes][2]usize, me: usize, j: usize, round: usize, cb: usize, sb: usize) ![2]?Copy {
    if (j == me) return .{ null, null };
    const part = try blobPart(quarters, j, me, cb, sb);
    if (part.rows == 0) return .{ null, null };
    const pm = try plan(me, quarters, vocabWidth(quarters));
    const row0 = pm.recv[j][0] - nodeRange(me)[0];
    const lo = round * chunk;
    const hi = lo + chunk;
    var out: [2]?Copy = .{ null, null };
    const segs = [2][3]usize{ .{ part.codes, part.rows * cb, row0 * cb }, .{ part.scales, part.rows * sb, row0 * sb } };
    for (segs, 0..) |s, k| {
        const a = @max(lo, s[0]);
        const b = @min(hi, s[0] + s[1]);
        if (a < b) out[k] = .{ .src = a - lo, .codes = k == 0, .dst = s[2] + (a - s[0]), .len = b - a };
    }
    return out;
}

/// A membership table over the vocabulary, a bit an id: the guard's test of a committed token.
pub const Bits = struct {
    words: []u64,

    pub fn init(gpa: std.mem.Allocator, vocab: usize) !Bits {
        const w = try gpa.alloc(u64, (vocab + 63) / 64);
        @memset(w, 0);
        for (0..count) |i| {
            const x = id(i);
            if (x < vocab) w[x / 64] |= @as(u64, 1) << @intCast(x % 64);
        }
        return .{ .words = w };
    }

    pub fn deinit(b: *Bits, gpa: std.mem.Allocator) void {
        gpa.free(b.words);
        b.* = undefined;
    }

    pub fn has(b: *const Bits, tok: u32) bool {
        const w = tok / 64;
        return w < b.words.len and b.words[w] & (@as(u64, 1) << @intCast(tok % 64)) != 0;
    }
};

/// The miss guard: committed tokens in blocks of 1,024; at each block's end, over the last four blocks, the passes
/// take the whole head when more than 2% of the tokens are outside the list and come back to it under 1%. It starts on
/// the list.
pub const Guard = struct {
    pub const block = 1024;
    pub const blocks = 4;

    ring: [blocks][2]u64 = @splat(.{ 0, 0 }), // the finished blocks' tokens and misses
    filled: usize = 0,
    at: usize = 0,
    cur: [2]u64 = .{ 0, 0 },
    full: bool = false, // the passes take the whole head
    tokens: u64 = 0, // committed tokens counted
    misses: u64 = 0, // of them, outside the list
    flips: u64 = 0, // changes between the list and the whole head

    pub fn note(g: *Guard, bits: *const Bits, toks: []const u32) void {
        for (toks) |t| {
            const miss: u64 = @intFromBool(!bits.has(t));
            g.tokens += 1;
            g.misses += miss;
            g.cur[0] += 1;
            g.cur[1] += miss;
            if (g.cur[0] == block) g.close();
        }
    }

    fn close(g: *Guard) void {
        g.ring[g.at] = g.cur;
        g.at = (g.at + 1) % blocks;
        g.filled = @min(g.filled + 1, blocks);
        g.cur = .{ 0, 0 };
        var t: u64 = 0;
        var m: u64 = 0;
        for (g.ring[0..g.filled]) |b| {
            t += b[0];
            m += b[1];
        }
        const was = g.full;
        if (!g.full and m * 100 > t * 2) g.full = true else if (g.full and m * 100 < t) g.full = false;
        if (g.full != was) g.flips += 1;
    }
};

/// markov.Markov on node r's list positions: draft2d's 128-column segments, 64 a node, the first at 64 r; `head` the
/// list's rows of the Markov head [count, rank] fp16 in its order, the head's logits [rows, 8,192] fp32. A step's best
/// is a list position: steps maps it to its id.
pub fn markov(c: Config, r: usize, head: u64, emb: u64, none: u64) tri_markov.Markov {
    const segs = per_node / draft2d.seg_cols;
    const s_tiles: usize = tri.cdiv(draft2d.seg_cols, tri_markov.bc);
    return .{
        .head = head,
        .emb = emb,
        .none = none,
        .slot = none,
        .world = nodes,
        .rank_dim = c.markov_rank,
        .split = true,
        .n_cols = draft2d.seg_cols,
        .segs = segs,
        .seg0 = r * segs,
        .cols = per_node,
        .b_tiles = tri.cdiv(draft2d.seg_cols, tri_markov.bn * tri_markov.sub),
        .s_tiles = s_tiles,
        .n_part = s_tiles * segs,
        .bp = tri.pow2(s_tiles * segs),
        .lg_w = per_node,
    };
}

/// A step's picked list positions to their ids, in place: out[r * stride] for r < rows (ops.Ops.listIds).
pub const MapIds = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, out: u64, stride: usize, rows: usize) anyerror!void,
};

/// Markov.steps on the list: each step's bests gathered over the nodes and picked, then the picked positions mapped to
/// their ids, so that the next step reads the token's own embedding row and cached bias row, and out holds ids.
pub fn steps(m: tri_markov.Markov, t: tri.Tri, lg: u64, out: u64, ts: usize, n: usize, blk: usize, cnt: usize, sc: tri_markov.Scratch, gather: tri_markov.Gather, map: MapIds) !void {
    if (!m.split) return error.NotSplit;
    for (0..cnt) |i| {
        const send = (try m.localBest(t, lg, out, ts, n, blk, i, sc)) orelse return error.NotSplit;
        try m.pick(t, try gather.run(gather.ctx, send, n), m.world, out, ts, n, i);
        try map.run(map.ctx, out + 8 * (i + 1), ts, n);
    }
}

/// TF_DS_DRAFT_VOCAB's kit condition: null when `finds` (the kit's base set: has(name, args, consts)) holds every launch
/// the list's Markov loop takes (each step's bias, score, finish and pick, at one stream and at sixteen), else why not.
pub fn kitOff(gpa: std.mem.Allocator, finds: anytype, c: Config) !?[]const u8 {
    // the engine's layout: every buffer aligned, out int64 [streams, steps + 1], step i's tokens at out + 8 i
    const base: u64 = 0x7a00_0000_0000;
    const blk = c.dspark_block;
    for ([_]usize{ 1, tri_markov.np }) |n| {
        var log: tri.Log = .{ .gpa = gpa };
        defer log.deinit();
        var m = markov(c, 0, base, base + 0x100_0000, base + 0x200_0000);
        m.cache = base + 0x300_0000;
        const sc: tri_markov.Scratch = .{ .pv = base + 0x400_0000, .pi = base + 0x500_0000, .send = base + 0x600_0000, .stage = base + 0x700_0000 };
        const out = base + 0x800_0000;
        for (0..blk) |i| {
            _ = try m.localBest(.{ .log = &log }, base + 0x900_0000, out, blk + 1, n, blk, i, sc);
            try m.pick(.{ .log = &log }, base + 0xa00_0000, nodes, out, blk + 1, n, i);
        }
        for (log.items.items) |k| if (!finds.has(k.name, k.args, k.consts)) return "the kit has no variant of the list's Markov loop";
    }
    return null;
}

/// The list's sha256, its first four bytes in hex (the start line's).
pub fn digest(buf: *[8]u8) []const u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bin, &h, .{});
    return std.fmt.bufPrint(buf, "{x}", .{h[0..4]}) catch buf[0..0];
}

/// TF_DS_DRAFT_VOCAB's start line: on every rank with the head rows this node took from others, or off on every rank,
/// the ranks that cannot and this rank's reason (`why`, null: it can).
pub fn line(buf: []u8, cannot: u32, why: ?[]const u8, ranks: []const u8, from_others: usize, vocab: usize, dg: []const u8) []const u8 {
    if (cannot != 0) return std.fmt.bufPrint(buf, "TF_DS_DRAFT_VOCAB=1: off on every rank: {s} cannot run it (this rank: {s})", .{ ranks, why orelse "can" }) catch buf[0..0];
    return std.fmt.bufPrint(buf, "TF_DS_DRAFT_VOCAB=1: the drafter's head and Markov loop on {d} ids a node ({d} of {d}, list {s}), on every rank; {d} of this node's head rows from the other nodes; the whole head kept: a pass takes it while more than 2% of the last {d} committed tokens are outside the list (back under 1%)", .{ per_node, count, vocab, dg, from_others, Guard.block * Guard.blocks }) catch buf[0..0];
}

// -- tests ----------------------------------------------------------------------------------------------------------

test "the list: 32,768 ascending ids below the vocabulary, every special token the model writes in it" {
    try std.testing.expectEqual(@as(?[]const u8, null), check(129_280));
    try std.testing.expectEqual(@as(usize, count * 4), bin.len);
    try std.testing.expect(check(id(count - 1)) != null); // its largest id past a smaller vocabulary
    for (forced) |f| try std.testing.expect(find(f) != null);
    for (0..count) |i| try std.testing.expectEqual(@as(?usize, i), find(id(i)));
    // ids outside it: the placeholders, and a gap of the list
    try std.testing.expectEqual(@as(?usize, null), find(128_000));
    var gap: ?u32 = null;
    for (1..count) |i| if (id(i) > id(i - 1) + 1) {
        gap = id(i - 1) + 1;
        break;
    };
    try std.testing.expectEqual(@as(?usize, null), find(gap.?));
    // its check refuses a list out of order, short, or without a special token
    var b: [count * 4]u8 = bin[0 .. count * 4].*;
    std.mem.writeInt(u32, b[4..8], std.mem.readInt(u32, b[0..4], .little), .little);
    try std.testing.expect(listCheck(&b, 129_280) != null);
    try std.testing.expect(listCheck(bin[0 .. count * 4 - 4], 129_280) != null);
    // the last special token (the list's largest id) replaced by the placeholder after it: ascending, but without it
    var nf: [count * 4]u8 = bin[0 .. count * 4].*;
    try std.testing.expectEqual(@as(u32, 129_278), id(count - 1));
    std.mem.writeInt(u32, nf[4 * (count - 1) ..][0..4], 129_279, .little);
    try std.testing.expectEqualStrings("a special token the model writes is not in the list", listCheck(&nf, 129_280).?);
    try std.testing.expectEqualStrings("the list's ids are not ascending", listCheck(&b, 129_280).?);
}

test "the list's runs: a vocabulary range's ids are one run of positions, and the nodes' ranges split the list" {
    const q = [_]usize{ 0, 32_384, 64_640, 97_024, 129_280 };
    var total: usize = 0;
    for (0..4) |j| {
        const r = run(q[j], q[j + 1]);
        for (r[0]..r[1]) |i| try std.testing.expect(id(i) >= q[j] and id(i) < q[j + 1]);
        if (r[0] > 0) try std.testing.expect(id(r[0] - 1) < q[j]);
        if (r[1] < count) try std.testing.expect(id(r[1]) >= q[j + 1]);
        total += r[1] - r[0];
    }
    try std.testing.expectEqual(@as(usize, count), total);
    try std.testing.expectEqual([2]usize{ 0, 0 }, run(0, 0));
    try std.testing.expectEqual([2]usize{ count, count }, run(129_280, 200_000));
    try std.testing.expectEqual([2]usize{ 8192, 16384 }, nodeRange(1));
    try std.testing.expectEqual([2]usize{ 8192, 9000 }, overlap(.{ 0, 9000 }, nodeRange(1)));
    try std.testing.expectEqual([2]usize{ 16384, 16384 }, overlap(.{ 0, 9000 }, nodeRange(2)));
}

test "the start's exchange: every node's 8,192 rows taken once, from the node whose quarter holds the id" {
    // the served split's quarters, in an order of ranks other than vocabulary order (rank 1 holds the third quarter)
    const quarters = [nodes][2]usize{ .{ 0, 32_384 }, .{ 64_640, 97_024 }, .{ 32_384, 64_640 }, .{ 97_024, 129_280 } };
    var plans: [nodes]Plan = undefined;
    for (0..nodes) |me| plans[me] = try plan(me, quarters, 129_280);
    for (0..nodes) |me| {
        // the receives tile the node's range in vocabulary order, each from the right node
        var got: usize = 0;
        for (plans[me].recv, 0..) |x, j| {
            for (x[0]..x[1]) |i| try std.testing.expect(id(i) >= quarters[j][0] and id(i) < quarters[j][1]);
            got += x[1] - x[0];
            // what j sends me is what I receive from j
            try std.testing.expectEqual(plans[j].send[me], x);
        }
        try std.testing.expectEqual(@as(usize, per_node), got);
        try std.testing.expectEqual(plans[me].send[me], plans[me].recv[me]);
        try std.testing.expectEqual(per_node - (plans[me].recv[me][1] - plans[me].recv[me][0]), plans[me].fromOthers(me));
    }
    // the blobs: every sender's parts follow one another in rank order, and a receiver's landings of every round tile its
    // part of the list's head once, codes and scales (FP4: 2,560 and 160 bytes a row)
    const cb: usize = 2560;
    const sb: usize = 160;
    const nr = try rounds(quarters, cb, sb);
    for (0..nodes) |me| {
        var seen_c: [nodes]usize = @splat(0);
        var seen_s: [nodes]usize = @splat(0);
        for (0..nr) |r| for (0..nodes) |j| for (try landings(quarters, me, j, r, cb, sb)) |c| if (c) |x| {
            try std.testing.expect(x.src + x.len <= chunk);
            if (x.codes) seen_c[j] += x.len else seen_s[j] += x.len;
        };
        for (0..nodes) |j| {
            const rows = if (j == me) 0 else plans[me].recv[j][1] - plans[me].recv[j][0];
            try std.testing.expectEqual(rows * cb, seen_c[j]);
            try std.testing.expectEqual(rows * sb, seen_s[j]);
        }
    }
    var bytes: usize = 0;
    for (0..nodes) |j| bytes += try blobBytes(quarters, j, cb, sb);
    try std.testing.expectEqual(@as(usize, 7659 + 533 + 806 + 4637) * (cb + sb), bytes);
    // quarters that do not tile the vocabulary
    var gap = quarters;
    gap[3][1] = 129_279;
    try std.testing.expectError(error.QuartersDoNotTile, plan(0, gap, 129_280));
    var twice = quarters;
    twice[1] = .{ 0, 32_384 };
    try std.testing.expectError(error.QuartersDoNotTile, plan(0, twice, 129_280));
}

test "the guard: a block of 1,024 committed tokens at a time, the whole head past 2% of the last 4,096, the list under 1%" {
    const gpa = std.testing.allocator;
    var bits = try Bits.init(gpa, 129_280);
    defer bits.deinit(gpa);
    for (0..count) |i| try std.testing.expect(bits.has(id(i)));
    try std.testing.expect(!bits.has(128_000) and !bits.has(200_000));
    const in: u32 = id(100);
    const out: u32 = 128_000;
    var g: Guard = .{};
    // 3% of the first block's tokens outside: past 2% at the block's end, not before it
    var toks: [Guard.block]u32 = @splat(in);
    for (0..31) |i| toks[i * 33] = out;
    g.note(&bits, toks[0 .. Guard.block - 1]);
    try std.testing.expect(!g.full);
    g.note(&bits, toks[Guard.block - 1 ..]);
    try std.testing.expect(g.full);
    try std.testing.expectEqual(@as(u64, Guard.block), g.tokens);
    try std.testing.expectEqual(@as(u64, 31), g.misses);
    try std.testing.expectEqual(@as(u64, 1), g.flips);
    // clean blocks: over four blocks 31 / 4,096 is under 1% (0.76%): back on the list at the fourth block, not the third
    const clean: [Guard.block]u32 = @splat(in);
    g.note(&bits, &clean); // 31 / 2,048
    g.note(&bits, &clean); // 31 / 3,072 = 1.01%
    try std.testing.expect(g.full);
    g.note(&bits, &clean); // 31 / 4,096
    try std.testing.expect(!g.full);
    try std.testing.expectEqual(@as(u64, 2), g.flips);
    // 2% of a window or less stays on the list (81 of 4,096: 1.98%); more goes past it (82: 2.002%)
    var t20: [Guard.block]u32 = @splat(in);
    for (0..20) |i| t20[i * 12] = out;
    for ([_]usize{ 21, 22 }) |last| {
        var h: Guard = .{};
        h.note(&bits, &t20);
        h.note(&bits, &t20);
        h.note(&bits, &t20);
        try std.testing.expect(!h.full); // 60 / 3,072 = 1.95%
        var t4: [Guard.block]u32 = @splat(in);
        for (0..last) |i| t4[i * 12] = out;
        h.note(&bits, &t4);
        try std.testing.expectEqual(last == 22, h.full);
    }
    // the window is the last four blocks: an old block of misses leaves it
    var k: Guard = .{};
    var bad: [Guard.block]u32 = @splat(in);
    for (0..100) |i| bad[i * 10] = out;
    k.note(&bits, &bad);
    try std.testing.expect(k.full);
    for (0..4) |_| k.note(&bits, &clean);
    try std.testing.expect(!k.full);
    try std.testing.expectEqual(@as(u64, 100), k.misses);
}

test "the list's Markov geometry: 64 segments of 128 a node from 64 r, its logits 8,192 wide, its bests gathered over four nodes" {
    var cfg: Config = undefined;
    cfg.markov_rank = 256;
    for (0..nodes) |r| {
        const m = markov(cfg, r, 1, 2, 3);
        try std.testing.expectEqual(@as(usize, 64), m.segs);
        try std.testing.expectEqual(64 * r, m.seg0);
        try std.testing.expectEqual(@as(usize, 128), m.n_cols);
        try std.testing.expectEqual(@as(usize, per_node), m.cols);
        try std.testing.expectEqual(@as(usize, per_node), m.lg_w);
        try std.testing.expectEqual(@as(usize, 4), m.world);
        try std.testing.expect(m.split);
        try std.testing.expectEqual(@as(usize, 1), m.s_tiles);
        try std.testing.expectEqual(@as(usize, 64), m.n_part);
        try std.testing.expectEqual(@as(usize, 64), m.bp);
        try std.testing.expectEqual(@as(usize, 1), m.b_tiles);
        try std.testing.expectEqual(@as(u64, 1), m.head);
        try std.testing.expectEqual(@as(u64, 3), m.slot);
        // a step's best is (seg0 + s) * 128 + c: node r's positions
        try std.testing.expectEqual(nodeRange(r)[0], m.seg0 * m.n_cols);
        try std.testing.expectEqual(nodeRange(r)[1], (m.seg0 + m.segs) * m.n_cols);
    }
}

test "the list's steps: each step's bias, score and finish, the pick of the gathered bests, then the ids" {
    var log: tri.Log = .{ .gpa = std.testing.allocator };
    defer log.deinit();
    var cfg: Config = undefined;
    cfg.markov_rank = 256;
    const m = markov(cfg, 2, 0x7a00_0000_0000, 0x7a10_0000_0000, 0x7a20_0000_0000);
    const S = struct {
        gathered: u64 = 0,
        mapped: [8]u64 = @splat(0),
        n: usize = 0,
        fn gather(ctx: *anyopaque, send: u64, rows: usize) anyerror!u64 {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            try std.testing.expectEqual(@as(u64, 0x7c00_0000_0200), send);
            try std.testing.expectEqual(@as(usize, 3), rows);
            s.gathered += 1;
            return 0x7d00_0000_0000;
        }
        fn map(ctx: *anyopaque, out: u64, stride: usize, rows: usize) anyerror!void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            try std.testing.expectEqual(@as(usize, 4), stride);
            try std.testing.expectEqual(@as(usize, 3), rows);
            s.mapped[s.n] = out;
            s.n += 1;
        }
    };
    var s: S = .{};
    const out: u64 = 0x7b00_0000_0000;
    const sc: tri_markov.Scratch = .{ .pv = 0x7c00_0000_0000, .pi = 0x7c00_0000_0100, .send = 0x7c00_0000_0200, .stage = 0x7c00_0000_0300 };
    try steps(m, .{ .log = &log }, 0x7e00_0000_0000, out, 4, 3, 5, 3, sc, .{ .ctx = &s, .run = S.gather }, .{ .ctx = &s, .run = S.map });
    try std.testing.expectEqual(@as(u64, 3), s.gathered);
    try std.testing.expectEqual(@as(usize, 3), s.n);
    // each step's map follows its pick and writes the step's column: out[:, i + 1]
    for (0..3) |i| try std.testing.expectEqual(out + 8 * @as(u64, i + 1), s.mapped[i]);
    try std.testing.expectEqual(@as(usize, 12), log.items.items.len);
    const names = [_][]const u8{ "_bias", "_score", "_finish", "_pick" };
    for (log.items.items, 0..) |kk, j| try std.testing.expectEqualStrings(names[j % 4], kk.name);
    // a Markov that is not split has no bests to gather: refused
    var whole = m;
    whole.split = false;
    try std.testing.expectError(error.NotSplit, steps(whole, .{ .log = &log }, 0, out, 4, 3, 5, 1, sc, .{ .ctx = &s, .run = S.gather }, .{ .ctx = &s, .run = S.map }));
}

test "TF_DS_DRAFT_VOCAB's line: on every rank with the rows taken from other nodes, or off on every rank, which ranks and why" {
    var buf: [512]u8 = undefined;
    var db: [8]u8 = undefined;
    const dg = digest(&db);
    try std.testing.expectEqual(@as(usize, 8), dg.len);
    try std.testing.expectEqualStrings("82fa3034", dg);
    var want: [512]u8 = undefined;
    const w = try std.fmt.bufPrint(&want, "TF_DS_DRAFT_VOCAB=1: the drafter's head and Markov loop on 8192 ids a node (32768 of 129280, list 82fa3034), on every rank; 7659 of this node's head rows from the other nodes; the whole head kept: a pass takes it while more than 2% of the last 4096 committed tokens are outside the list (back under 1%)", .{});
    try std.testing.expectEqualStrings(w, line(&buf, 0, null, "", 7659, 129_280, dg));
    try std.testing.expectEqualStrings("TF_DS_DRAFT_VOCAB=1: off on every rank: rank 2 cannot run it (this rank: can)", line(&buf, 1 << 2, null, "rank 2", 0, 129_280, dg));
    try std.testing.expectEqualStrings("TF_DS_DRAFT_VOCAB=1: off on every rank: ranks 0, 1, 2 and 3 cannot run it (this rank: needs TF_DS_DRAFT_HEAD=fp4)", line(&buf, 0b1111, "needs TF_DS_DRAFT_HEAD=fp4", "ranks 0, 1, 2 and 3", 0, 129_280, dg));
}

test "the list's Markov loop resolves in the kit: every step's launch at one stream and at sixteen; a kit without its finish refuses" {
    // fixtures/kit-markov.json (tools/dsv41-zig/kit/kit_fixture.py --markov): the kit's entries of the four functions
    const gpa = std.testing.allocator;
    var cfg: Config = undefined;
    cfg.markov_rank = 256;
    cfg.dspark_block = 5;
    const Finds = struct {
        m: *const cuda.aot.Manifest,
        pub fn has(f: @This(), name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
            return f.m.find(name, args, consts) != null;
        }
    };
    const text = @embedFile("fixtures/kit-markov.json");
    var kit = try cuda.aot.Manifest.parse(gpa, text);
    defer kit.deinit();
    try std.testing.expectEqual(@as(?[]const u8, null), try kitOff(gpa, Finds{ .m = &kit }, cfg));
    // the same kit with no finish of 64 partials (the list's: 64 segments of one tile)
    const no64 = try std.mem.replaceOwned(u8, gpa, text, "\"BP\":{\"int\":64}", "\"BP\":{\"int\":63}");
    defer gpa.free(no64);
    var less = try cuda.aot.Manifest.parse(gpa, no64);
    defer less.deinit();
    try std.testing.expectEqualStrings("the kit has no variant of the list's Markov loop", (try kitOff(gpa, Finds{ .m = &less }, cfg)).?);
}
