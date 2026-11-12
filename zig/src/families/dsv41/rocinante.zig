//! Rocinante (Roci), the engine's own RDMA transport between the four nodes'
//! registered host memory. The tensors that cross the
//! wire live in one arena of registered pages, the producer kernels write them there and the consumer kernels read
//! them there. A part moves as two contiguous halves, one RDMA WRITE per rail straight to its final offset in the
//! peer's arena, each followed on the same queue pair by an 8-byte generation word. The sender needs one stream write
//! (the doorbell) after the producer, the receiver one stream wait on the words, and a proxy thread a rail posts
//! pre-built chains between them. Nothing is copied between a producer and its consumer.
//!
//! This file holds the pieces that need no device: the arena's layout, the events' chains, the control block, the
//! proxies, the watchdog and the abort. The NIC behind a post is an interface (`Nic`); the verbs implementation is in
//! rocinante_verbs.zig, a loopback one in the tests.
//! With TF_DS_ROCI_WAIT=1 (Settings.wait_qp) there is no proxy: the engine thread writes every chain into its queue pair when it reserves the
//! send, behind a WAIT on the doorbell word that the card runs, and one janitor thread watches and reaps (rocinante_cd.zig).

const std = @import("std");
const nap = @import("rdma_nap.zig");
const slabs = @import("slab_plan.zig");

pub const world: usize = 4;
pub const rails: usize = 2;
pub const line: usize = 64;
pub const page: usize = 4096;
/// The scheduling ring's entries: the host runs at least a pass ahead of the proxies, and a pass sends 168 times.
pub const sched_len: usize = 8192;
/// The arena is laid out for chunks of this many rows; a chunk uses lengths and offsets inside it. 4,096 unless the booted pass
/// rows are more (setCap, once at the start, before any layout is read: every rank alike, TF_DS_PASS_ROWS is compared).
pub var max_cap: usize = 4096;

/// The pass rows the arena's layout and the plans' shapes follow (Model.open, once): the pass rows booted, never below 4,096 (the
/// layout of a 2,048-row pass is the baseline).
pub fn setCap(pass_rows: usize) void {
    max_cap = @max(4096, pass_rows);
    Transport.shape_caps = .{ max_cap, 2048 };
}

// -- the arena -------------------------------------------------------------------------------------------------------

/// The tensors that cross the wire, in arena order.
pub const Region = enum(u8) {
    psum, // owner sums' fp32 partials [cap][2560]; serves attention and MoE
    stage, // part 1 landing: fp32 [2][cap/4][2560]
    send2, // part 2 source: bf16 [cap/4][2560]
    blk, // part 2 landing and the own block: bf16 [2][cap/4][2560]
    ek, // Engram's fp32 partials [cap][12800]
    eng_stage,
    eng_send2,
    eng_blk,
    wob_send, // wo_b's input as FP8 codes + scales: records of 2,304 bytes, one a row
    wob_land,
    pack_send, // the expert pack's codes + scales: records of 720 bytes, 7 slot rows a row
    pack_land,
    g_x, // gathered tensors: each rank's slab at its place
    g_qr,
    g_y,
    g_cidx,
    g_q,
    g_s,
    g_pick,
    g_wts,
    g_b, // the boundary layer's cache rows: each rank's slab holds the four arrays' rows, array after array
    hand_h, // the handoff of a chunk with decoder rows: the streams h (bf16 [hc][hidden] a row) of the rows from `first`, at row (row - first)
    hand_pre, // ... and their pre (fp32 [hc] a row)
    g_taps, // a decoder window's DSpark taps: bf16 [hand_rows][3][hidden], the three taps of a row side by side; each rank's slab at its rows
    head_x, // the head's input row (bf16 [hidden]): the collapse of the window's last row, from the rank that owns it
    // the host lane's (layoutOf(.host)): the partner exchange of the Engram worker's records; empty in the prompt lanes' layout
    rec_send, // [host_slots][host_max_recs] records, the ones this node read for its partner
    rec_land, // where the partner's land
    rec_hs, // [host_slots] 64-byte headers sent
    rec_hl, // and landed
};

/// The most rows a handoff carries: the decoder rows of a chunk (the window and the drafter's rows).
pub const hand_rows: usize = 128;
/// One row of the handoff's streams: hc = 4 streams of 5,120 bf16 values; and its pre: 4 fp32 values.
pub const hand_h_row: usize = 4 * 5120 * 2;
pub const hand_pre_row: usize = 16;
/// One row of the decoder window's taps: the three tap layers' means side by side (the drafter's row: taps * hidden bf16).
pub const taps_row: usize = 3 * 5120 * 2;
/// The head's input row: hidden bf16 values.
pub const head_row: usize = 5120 * 2;

pub const n_regions = std.enums.values(Region).len;

/// The partner exchange's geometry: a record is an Engram row's value bytes and scales (dim 256 + 8), a slot holds the records of half a
/// step's misses (2,048 rows × 12 columns / 2), and the ring has this many slots.
pub const host_rec: usize = 264;
pub const host_max_recs: usize = 12288;
pub const host_slots: usize = 16;
pub const host_slot_bytes: usize = host_max_recs * host_rec;
pub const host_hdr_bytes: usize = 64;
/// The longest nap of a host wait, in microseconds: it ends when the awaited word is written; the bound lets the wait look at the other rail's word.
pub const host_nap_us: u32 = 25;

/// Which arena a lane has: the prompt lanes' (the exchanged tensors of a pass), or the host lane's (the Engram records of the partner split).
pub const LayoutSet = enum { prompt, host };

pub const pair_w: usize = 2560; // a pair's columns of the hidden state
pub const eng_w: usize = 12800; // an Engram part's columns
/// One boundary-layer row of the four cache arrays: compressed codes (head_dim / 2) and scales (head_dim / 16), index codes
/// (index_head_dim / 2) and scales (index_head_dim / 32), for head_dim 512 and index_head_dim 128.
pub const gb_row: usize = 256 + 32 + 64 + 4;

/// A region's bytes at `cap` rows (cap a multiple of 4): the row-shaped sizes the maps give, before page rounding.
fn regionBytes(r: Region, cap: usize, set: LayoutSet) usize {
    const q = cap / world;
    if (set == .host) return switch (r) {
        .rec_send, .rec_land => host_slots * host_slot_bytes,
        .rec_hs, .rec_hl => host_slots * host_hdr_bytes,
        else => 0,
    };
    return switch (r) {
        .rec_send, .rec_land, .rec_hs, .rec_hl => 0,
        .psum => cap * pair_w * 4,
        .stage => 2 * q * pair_w * 4,
        .send2 => q * pair_w * 2,
        .blk => 2 * q * pair_w * 2,
        .ek => cap * eng_w * 4,
        .eng_stage => 2 * q * eng_w * 4,
        .eng_send2 => q * eng_w * 2,
        .eng_blk => 2 * q * eng_w * 2,
        .wob_send, .wob_land => cap * 2304,
        .pack_send, .pack_land => 7 * cap * 720,
        .g_x => cap * 5120 * 2,
        .g_qr => cap * 1280 * 2,
        .g_y => cap * 512 * 2,
        .g_cidx => cap * 512 * 8,
        .g_q => cap * 5120,
        .g_s => cap * 160 * 4,
        .g_pick, .g_wts => cap * 7 * 4,
        .g_b => cap * gb_row,
        .hand_h => hand_rows * hand_h_row,
        .hand_pre => hand_rows * hand_pre_row,
        .g_taps => hand_rows * taps_row,
        .head_x => head_row,
    };
}

/// The bytes of one row of a gathered region (the slab of rank k starts at k * q * row).
pub fn rowBytes(r: Region) usize {
    return switch (r) {
        .g_x => 5120 * 2,
        .g_qr => 1280 * 2,
        .g_y => 512 * 2,
        .g_cidx => 512 * 8,
        .g_q => 5120,
        .g_s => 160 * 4,
        .g_pick, .g_wts => 7 * 4,
        .g_b => gb_row,
        .hand_h => hand_h_row,
        .hand_pre => hand_pre_row,
        .g_taps => taps_row,
        .head_x => head_row,
        else => unreachable,
    };
}

pub const Layout = struct {
    off: [n_regions]u64,
    len: [n_regions]u64, // the region's bytes (page-rounded in `size`)
    total: u64,

    pub fn at(l: Layout, r: Region) u64 {
        return l.off[@backingInt(r)];
    }
    pub fn bytes(l: Layout, r: Region) u64 {
        return l.len[@backingInt(r)];
    }

    /// A word that two ranks compare at connect: equal layouts, equal hash.
    pub fn hash(l: Layout) u64 {
        var h: u64 = 0xcbf29ce484222325;
        for (l.off, l.len) |o, n| {
            h = (h ^ o) *% 0x100000001b3;
            h = (h ^ n) *% 0x100000001b3;
        }
        return h ^ l.total;
    }
};

/// The arena's regions, each starting on a page and sized for `max_cap` rows (a pure function: every rank computes the
/// same one).
pub fn layout() Layout {
    return layoutOf(.prompt);
}

/// The layout of a lane's arena (a pure function of the set: every rank computes the same one).
pub fn layoutOf(set: LayoutSet) Layout {
    var l: Layout = undefined;
    var at: u64 = 0;
    for (0..n_regions) |i| {
        const n = regionBytes(@fromBackingInt(@intCast(i)), max_cap, set);
        l.off[i] = at;
        l.len[i] = n;
        at += std.mem.alignForward(u64, n, page);
    }
    l.total = at;
    return l;
}

// -- events ----------------------------------------------------------------------------------------------------------

/// What a send is: one word class a kind. Pairwise events go to one partner; a gather goes to the other three.
pub const Class = enum(u8) { own1, own2, eng1, eng2, wob, pack, gf, ga, gm, gi, gb, gh, gt, hx, rech, rec };
pub const n_classes = std.enums.values(Class).len;

/// One RDMA WRITE: `len` bytes from arena offset `src` to the peer's arena offset `dst`.
pub const Piece = struct { src: u64, dst: u64, len: u64 };

pub const max_pieces = 4;

/// What one rail sends to one peer for one event: the pieces in order, then the generation word.
pub const Chain = struct {
    pieces: [max_pieces]Piece = undefined,
    n: u8 = 0,

    fn add(c: *Chain, p: Piece) void {
        if (p.len == 0) return;
        c.pieces[c.n] = p;
        c.n += 1;
    }

    pub fn slice(c: *const Chain) []const Piece {
        return c.pieces[0..c.n];
    }
    pub fn bytes(c: *const Chain) u64 {
        var t: u64 = 0;
        for (c.slice()) |p| t += p.len;
        return t;
    }
};

pub const max_peers = 3;

/// An event for one node: its peers and, for each, a chain a rail.
pub const Event = struct {
    class: Class,
    peers: [max_peers]u8 = undefined,
    n_peers: u8 = 0,
    chains: [max_peers][rails]Chain = @splat(.{ .{}, .{} }),

    pub fn peerSlice(e: *const Event) []const u8 {
        return e.peers[0..e.n_peers];
    }
};

/// The two rails' halves of `len` bytes: the first the larger when odd, on a 64-byte boundary, which is a page
/// boundary wherever the half is a page multiple (every region but the picks' and weights').
pub fn halves(len: u64) [2]u64 {
    const h = std.mem.alignForward(u64, len / 2, line);
    const a = @min(h, len);
    return .{ a, len - a };
}

/// A part of `len` bytes at `src` -> `dst` as the two rails' halves.
fn split(src: u64, dst: u64, len: u64) [rails]Piece {
    const h = halves(len);
    return .{ .{ .src = src, .dst = dst, .len = h[0] }, .{ .src = src + h[0], .dst = dst + h[0], .len = h[1] } };
}

/// The rows of the node `order[g]`-th block: `Two.rank_of[node]` in the engine, the identity by default.
pub const Order = [world]u8;
pub const identity: Order = .{ 0, 1, 2, 3 };

/// The event `class` of node `g` for a chunk of `cap` rows (cap a multiple of 4 up to max_cap).
pub fn buildEvent(class: Class, cap: usize, g: usize, order: Order) Event {
    const L = layout();
    const q: u64 = cap / world;
    var e: Event = .{ .class = class };
    switch (class) {
        .own1, .eng1 => {
            // psum (or ek) rows of the row partner's block and of the diagonal's block, each whole on one rail
            const w: u64 = if (class == .own1) pair_w else eng_w;
            const src_r: Region = if (class == .own1) .psum else .ek;
            const dst_r: Region = if (class == .own1) .stage else .eng_stage;
            const blk = q * w * 4;
            e.peers[0] = @intCast(g ^ 1);
            e.n_peers = 1;
            const rows = [rails]usize{ g ^ 1, g ^ 3 };
            for (0..rails) |rail| {
                e.chains[0][rail].add(.{ .src = L.at(src_r) + order[rows[rail]] * blk, .dst = L.at(dst_r) + rail * blk, .len = blk });
            }
        },
        .own2, .eng2 => {
            const w: u64 = if (class == .own2) pair_w else eng_w;
            const src_r: Region = if (class == .own2) .send2 else .eng_send2;
            const dst_r: Region = if (class == .own2) .blk else .eng_blk;
            const blk = q * w * 2;
            e.peers[0] = @intCast(g ^ 2);
            e.n_peers = 1;
            const pieces = split(L.at(src_r), L.at(dst_r) + (g >> 1) * blk, blk);
            for (0..rails) |rail| e.chains[0][rail].add(pieces[rail]);
        },
        .wob, .pack => {
            const s: Region = if (class == .wob) .wob_send else .pack_send;
            const d: Region = if (class == .wob) .wob_land else .pack_land;
            const n: u64 = if (class == .wob) cap * 2304 else 7 * cap * 720;
            e.peers[0] = @intCast(g ^ 2);
            e.n_peers = 1;
            const pieces = split(L.at(s), L.at(d), n);
            for (0..rails) |rail| e.chains[0][rail].add(pieces[rail]);
        },
        .gh, .gt, .hx => {}, // the handoff, the taps and the head's row have no full-cap plan: their rows are a decoder window's (buildEventX)
        .rech, .rec => {}, // the host lane's events have their own plan (buildRecEvent)
        .gf, .ga, .gm, .gi, .gb => {
            const regs: []const Region = switch (class) {
                .gf => &.{ .g_qr, .g_y },
                .ga => &.{.g_x},
                .gm => &.{ .g_q, .g_s, .g_pick, .g_wts },
                .gi => &.{.g_cidx},
                else => &.{.g_b},
            };
            var k: u8 = 0;
            for ([_]usize{ 1, 2, 3 }) |x| {
                const peer = g ^ x;
                e.peers[k] = @intCast(peer);
                for (regs) |r| {
                    const slab = order[g] * q * rowBytes(r);
                    const pieces = split(L.at(r) + slab, L.at(r) + slab, q * rowBytes(r));
                    for (0..rails) |rail| e.chains[k][rail].add(pieces[rail]);
                }
                k += 1;
            }
            e.n_peers = k;
        },
    }
    return e;
}

/// The event `class` of node `g` for a chunk of `n` rows of any count (1 to max_cap), the owned rows' plan of slab_plan.zig: rank k of
/// `order` owns the rows [k q, k q + n_k), q = slabs.q(n). The owner sums and the gathers move those slabs (a rank without rows
/// sends no WRITE, only its word), the codes move all `n` rows. For n = 4,096 and 2,048 it is buildEvent's event, piece for piece.
pub fn buildEventN(class: Class, n: usize, g: usize, order: Order) Event {
    return buildEventX(class, n, 0, g, order);
}

/// buildEventN with the event's one extra number `x`: the indexer's top-k width in keys for gi (0: the region's 512), the decoder rows for
/// gh (the handoff of the rows [n - x, n) of the chunk's streams, each owner's rows in that window at row (row - (n - x)) of the region).
/// For gt and hx, `n` is a decoder window's rows (1 to hand_rows): gt moves each rank's slab of the window's taps, its rows of the three
/// taps side by side, to the other three; hx the head's row from the rank whose slab holds the window's last row (headOwner), the others
/// sending their word alone.
pub fn buildEventX(class: Class, n: usize, x: usize, g: usize, order: Order) Event {
    const L = layout();
    const q: u64 = slabs.q(n);
    var e: Event = .{ .class = class };
    switch (class) {
        .own1, .eng1 => {
            const w: u64 = if (class == .own1) pair_w else eng_w;
            const src_r: Region = if (class == .own1) .psum else .ek;
            const dst_r: Region = if (class == .own1) .stage else .eng_stage;
            const row = w * 4;
            e.peers[0] = @intCast(g ^ 1);
            e.n_peers = 1;
            const rows = [rails]usize{ g ^ 1, g ^ 3 };
            for (0..rails) |rail| {
                const k = order[rows[rail]];
                e.chains[0][rail].add(.{ .src = L.at(src_r) + k * q * row, .dst = L.at(dst_r) + rail * q * row, .len = slabs.slab(n, k).n * row });
            }
        },
        .own2, .eng2 => {
            const w: u64 = if (class == .own2) pair_w else eng_w;
            const src_r: Region = if (class == .own2) .send2 else .eng_send2;
            const dst_r: Region = if (class == .own2) .blk else .eng_blk;
            const row = w * 2;
            e.peers[0] = @intCast(g ^ 2);
            e.n_peers = 1;
            const pieces = split(L.at(src_r), L.at(dst_r) + (g >> 1) * q * row, slabs.slab(n, order[g ^ 2]).n * row);
            for (0..rails) |rail| e.chains[0][rail].add(pieces[rail]);
        },
        .wob, .pack => {
            const s: Region = if (class == .wob) .wob_send else .pack_send;
            const d: Region = if (class == .wob) .wob_land else .pack_land;
            const len: u64 = if (class == .wob) n * 2304 else 7 * n * 720;
            e.peers[0] = @intCast(g ^ 2);
            e.n_peers = 1;
            const pieces = split(L.at(s), L.at(d), len);
            for (0..rails) |rail| e.chains[0][rail].add(pieces[rail]);
        },
        .gh => {
            const first: u64 = n - @min(x, n);
            const k = order[g];
            const sl = slabs.slab(n, k);
            const lo: u64 = @max(first, sl.start);
            const hi: u64 = @min(n, sl.start + sl.n);
            const m: u64 = if (hi > lo) hi - lo else 0;
            var j: u8 = 0;
            for ([_]usize{ 1, 2, 3 }) |xx| {
                e.peers[j] = @intCast(g ^ xx);
                for ([_]Region{ .hand_h, .hand_pre }) |r| {
                    const at = L.at(r) + (lo - @min(lo, first)) * rowBytes(r);
                    const pieces = split(at, at, m * rowBytes(r));
                    for (0..rails) |rail| e.chains[j][rail].add(pieces[rail]);
                }
                j += 1;
            }
            e.n_peers = j;
        },
        .gt, .hx => {
            std.debug.assert(n <= hand_rows);
            const sl = slabs.slab(n, order[g]);
            const at: u64 = if (class == .gt) L.at(.g_taps) + sl.start * taps_row else L.at(.head_x);
            const len: u64 = if (class == .gt) sl.n * taps_row else if (order[g] == headOwner(n)) head_row else 0;
            var j: u8 = 0;
            for ([_]usize{ 1, 2, 3 }) |px| {
                e.peers[j] = @intCast(g ^ px);
                const pieces = split(at, at, len);
                for (0..rails) |rail| e.chains[j][rail].add(pieces[rail]);
                j += 1;
            }
            e.n_peers = j;
        },
        .rech, .rec => {}, // the host lane's events have their own plan (buildRecEvent)
        .gf, .ga, .gm, .gi, .gb => {
            const regs: []const Region = switch (class) {
                .gf => &.{ .g_qr, .g_y },
                .ga => &.{.g_x},
                .gm => &.{ .g_q, .g_s, .g_pick, .g_wts },
                .gi => &.{.g_cidx},
                else => &.{.g_b},
            };
            const k = order[g];
            const nk: u64 = slabs.slab(n, k).n;
            var j: u8 = 0;
            for ([_]usize{ 1, 2, 3 }) |px| {
                e.peers[j] = @intCast(g ^ px);
                if (class == .gb and nk != q) {
                    // a short slab: the four arrays' rows, each array's q-row stride apart inside the slab
                    const rows_of = [4]u64{ 256, 32, 64, 4 };
                    var at: u64 = k * q * gb_row;
                    for (rows_of) |rb| {
                        const pieces = split(L.at(.g_b) + at, L.at(.g_b) + at, nk * rb);
                        for (0..rails) |rail| e.chains[j][rail].add(pieces[rail]);
                        at += q * rb;
                    }
                } else for (regs) |r| {
                    // the top-k's rows are `x` keys of 8 bytes, contiguous (a short compressed history), else the region's 512
                    const rb: u64 = if (class == .gi and x != 0) x * 8 else rowBytes(r);
                    const slab = k * q * rb;
                    const pieces = split(L.at(r) + slab, L.at(r) + slab, nk * rb);
                    for (0..rails) |rail| e.chains[j][rail].add(pieces[rail]);
                }
                j += 1;
            }
            e.n_peers = j;
        },
    }
    return e;
}

/// The rank (slab plan index, as `order` gives a node's) whose slab holds the last row of a window of `n` rows: the head's row is its.
pub fn headOwner(n: usize) usize {
    return (n - 1) / slabs.q(n);
}

/// The host lane's event `class` (rech: a 64-byte header, rec: `count` records) of node `g` for ring slot `slot`: to the partner g ^ 2, from this
/// node's send region to the same offset of the partner's land region, in the two rails' halves.
pub fn buildRecEvent(class: Class, g: usize, slot: usize, count: usize) Event {
    const L = layoutOf(.host);
    var e: Event = .{ .class = class };
    e.peers[0] = @intCast(g ^ 2);
    e.n_peers = 1;
    const src_r: Region = if (class == .rec) .rec_send else .rec_hs;
    const dst_r: Region = if (class == .rec) .rec_land else .rec_hl;
    const per: u64 = if (class == .rec) host_slot_bytes else host_hdr_bytes;
    const len: u64 = if (class == .rec) count * host_rec else host_hdr_bytes;
    const pieces = split(L.at(src_r) + slot * per, L.at(dst_r) + slot * per, len);
    for (0..rails) |rail| e.chains[0][rail].add(pieces[rail]);
    return e;
}

/// The word classes a rank receives on: from which sources an event of this class arrives.
pub fn sourcesOf(class: Class, g: usize, out: *[world]u8) []const u8 {
    var n: usize = 0;
    switch (class) {
        .own1, .eng1 => {
            out[0] = @intCast(g ^ 1);
            n = 1;
        },
        .own2, .eng2, .wob, .pack, .rech, .rec => {
            out[0] = @intCast(g ^ 2);
            n = 1;
        },
        else => for ([_]usize{ 1, 2, 3 }) |x| {
            out[n] = @intCast(g ^ x);
            n += 1;
        },
    }
    return out[0..n];
}

// -- the control block -----------------------------------------------------------------------------------------------

/// How a part's arrival is told to the receiver, on the data's own queue pair after the data WRITEs: `write`, a second RDMA WRITE
/// of the generation (the receiver's stream wait takes it as reached when it is at least the count); `atomic`, an RDMA ATOMIC
/// fetch-and-add of 1 on the same word. The InfiniBand ordering rules and PCIe's place an atomic (non-posted) after the writes
/// before it; two WRITEs are ordered only as the card happens to do them (the verbs promise order within one work request). In
/// both the word of a (class, source, rail) is at generation N after N events, so the wait is the same. The atomic is the NIC's:
/// a card that reports atomics only against its own atomics (IBV_ATOMIC_HCA) does not order a CPU or GPU store to the word
/// against one in flight, so with `atomic` nothing stores to a word the card updates but after the card has been fenced off the
/// control block (`Nic.fail_fn`).
pub const Signal = enum(u8) { write, atomic };

/// The signal the Settings structs of the NICs start from (the host tests' fake cards have no atomics).
pub const default_signal: Signal = .write;

/// The signal a start without TF_DS_ROCI_SIGNAL takes: the atomic, which the card executes only after the data WRITEs are placed (D79); a card without
/// atomics takes the WRITE with a warning (roci_engine.signalMode).
pub const env_default_signal: Signal = .atomic;

/// TF_DS_ROCI_SIGNAL's text: unset, empty or `atomic`: the fetch-and-add; `write`: the second WRITE; anything else null (refused).
pub fn signalFrom(text: ?[]const u8) ?Signal {
    const t = std.mem.trim(u8, text orelse return env_default_signal, " \t");
    if (t.len == 0) return env_default_signal;
    if (std.mem.eql(u8, t, "write")) return .write;
    if (std.mem.eql(u8, t, "atomic")) return .atomic;
    return null;
}

/// A word alone on its cache line: the monitor of a nap, a stream write and a NIC write never share one.
pub const Word = extern struct {
    v: u64 align(line) = 0,
    pad: [line - 8]u8 = @splat(0),
};

/// The abort value stored into every awaited word: above any generation, below 2^63 so the cyclic compare passes.
pub const abort_value: u64 = 1 << 62;

/// Ordinary mapped (not write-combined) memory: the CPU proxy loads it, the NICs and the GPU write it.
pub const Control = extern struct {
    doorbell: Word = .{},
    abort: Word = .{},
    done: [rails]Word = @splat(.{}),
    sched: [sched_len]u64 align(line) = @splat(0),
    /// An entry's extra number (buildEventX's x), stored before the entry itself is.
    sched_x: [sched_len]u32 = @splat(0),
    recv: [n_classes][world][rails]Word = @splat(@splat(@splat(.{}))),
};

/// A scheduling entry: the send's ticket (so a reader can tell the entry is there) and its event.
pub fn schedEntry(ticket: u64, class: Class, shape: u8) u64 {
    return (ticket & 0xffff_ffff) | (@as(u64, @backingInt(class)) << 32) | (@as(u64, shape) << 40);
}
/// The plan of an event of `n` rows of any count (shape any_shape, the rows in the entry's high 16 bits).
pub const any_shape: u8 = 255;
pub fn schedEntryN(ticket: u64, class: Class, n: usize) u64 {
    return schedEntry(ticket, class, any_shape) | (@as(u64, @intCast(n)) << 48);
}
pub fn schedRows(e: u64) usize {
    return @intCast(e >> 48);
}
pub fn schedTicket(e: u64) u32 {
    return @truncate(e);
}
pub fn schedClass(e: u64) Class {
    return @fromBackingInt(@intCast(@as(u8, @truncate(e >> 32))));
}
pub fn schedShape(e: u64) u8 {
    return @truncate(e >> 40);
}

// -- the NIC ---------------------------------------------------------------------------------------------------------

pub const PostError = error{ PostFailed, QpError };

/// What a proxy needs of the network: post one chain on a rail to a peer, ending in the generation word.
pub const Nic = struct {
    ctx: *anyopaque,
    post_fn: *const fn (ctx: *anyopaque, rail: u8, peer: u8, class: Class, chain: *const Chain, word: u64) PostError!void,
    /// Failure path: tell the peers (best effort), move every queue pair to error, and return only when the card can no longer
    /// write the control block (the abort values are stored into the words after it: a store racing a card's update of the same
    /// word could be overwritten by it, and a wait on that word would never end).
    fail_fn: *const fn (ctx: *anyopaque) void,
    /// The proxy's chance to reap completions: an error completion fails the transport.
    reap_fn: *const fn (ctx: *anyopaque, rail: u8) PostError!void,
    /// The card-waiting backend's post (Settings.wait_qp; null for the others): write the chain into the send queue of (rail, peer) now,
    /// behind a WAIT for the doorbell word to reach `wait_for` (0: no WAIT, the chain is on its way at once), and ring the queue.
    prepost_fn: ?*const fn (ctx: *anyopaque, rail: u8, peer: u8, class: Class, chain: *const Chain, word: u64, wait_for: u64) PostError!void = null,

    fn post(n: Nic, rail: u8, peer: u8, class: Class, chain: *const Chain, word: u64) PostError!void {
        return n.post_fn(n.ctx, rail, peer, class, chain, word);
    }
};

// -- counters --------------------------------------------------------------------------------------------------------

/// Plain counters, one writer each (the engine thread or one rail's proxy); read when the pass is quiet.
pub const Counters = struct {
    // engine thread
    events: u64 = 0,
    doorbells: u64 = 0,
    stream_writes: u64 = 0,
    wait_groups: u64 = 0,
    // a rail's proxy
    data_writes: u64 = 0,
    words: u64 = 0,
    posts: u64 = 0,
    bytes_out: u64 = 0,
    wakes: u64 = 0,
    backstop_wakes: u64 = 0,
    post_ns_max: u64 = 0,
    watchdog_max_age_ms: u64 = 0,

    pub fn add(a: *Counters, b: Counters) void {
        inline for (@typeInfo(Counters).@"struct".field_names) |name| {
            if (comptime std.mem.eql(u8, name, "post_ns_max") or std.mem.eql(u8, name, "watchdog_max_age_ms")) {
                @field(a, name) = @max(@field(a, name), @field(b, name));
            } else @field(a, name) += @field(b, name);
        }
    }
};

// -- the transport ---------------------------------------------------------------------------------------------------

/// The transport's name, as its start line prints it.
pub const transport_name = "Rocinante transport";

/// What the transport carries, as the start line says it: the owner sums of full passes always, the codes families and the gathers
/// when the arena holds their regions.
pub fn carriesText(buf: []u8, codes: bool, gathers: bool) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    w.writeAll("the owner sums of full passes (attention, MoE, Engram)") catch {};
    if (codes) w.writeAll(", the codes families (wo_b's and the expert pack's)") catch {};
    if (gathers) w.writeAll(", the gathers (qr and y, x, the routing's codes and picks, the indexer's top-k, the boundary's cache rows)") catch {};
    return w.buffered();
}

/// An owner pass (attention x20, MoE x20: two parts each; Engram x2: two parts each) sends this many events; the proxies print
/// their rail's line every time they have posted that many.
pub const events_per_pass: u64 = 84;

/// The events of a full pass with the codes families (wo_b's and the pack's, 20 each) and the gathers (qr and y 17, x 3, the
/// routing's 20, the top-k 3, the boundary 1) on the transport as well.
pub fn eventsPerPass(codes: bool, gathers: bool) u64 {
    return events_per_pass + (if (codes) @as(u64, 40) else 0) + (if (gathers) @as(u64, 44) else 0);
}

/// The time from a proxy's post of an event to the last of its peers' words on that rail seen by the same proxy (it looks every
/// wake: at most the backstop late). It holds the wire and the peers' lateness; per class.
pub const Wire = struct { n: u64 = 0, sum_ns: u64 = 0, max_ns: u64 = 0 };

const Pending = struct { class: u8, t_ns: u64, want: u64 };
const pending_len = 512;

/// What the start line says of a signal.
pub fn signalText(signal: Signal) []const u8 {
    return switch (signal) {
        .write => "an 8-byte RDMA WRITE after the data",
        .atomic => "an RDMA ATOMIC fetch-and-add after the data",
    };
}

/// The start line (once, on every rank): the name, the arena, the rails and the plan hash.
pub fn startLine(w: *std.Io.Writer, g: usize, rails_text: []const u8, gid_index: u8, s: Settings, signal: Signal) !void {
    const l = layout();
    var cb: [512]u8 = undefined;
    try w.print("{s} (Roci) on: node {d}, arena {d} bytes in {d} regions, control block {d} bytes, rails {s} (GID {d}), layout hash {x}, watchdog {d} ms, backstop {d} us, idle park {d} us, housekeeping every {d} turns, arrival signal {s}, carrying {s}{s}\n", .{ transport_name, g, l.total, n_regions, @sizeOf(Control), rails_text, gid_index, l.hash(), s.dead_ms, s.backstop_us, s.idle_park_us, s.housekeep_every, signalText(signal), carriesText(&cb, s.codes, s.gathers), if (s.wait_qp) ", waiting in the card (TF_DS_ROCI_WAIT, the default; =0 for the proxy threads: every chain pre-posted behind a WAIT on the doorbell, no proxy threads, one janitor)" else "" });
}

pub const Settings = struct {
    /// A nap's longest length: the backstop that catches a GPU write the nap missed.
    backstop_us: u32 = 100,
    /// No word from a peer for this long while one is owed: the pass stops, naming the peer and the class.
    dead_ms: u64 = 5_000,
    /// The two rails' proxies nap on the doorbell (TF_DS_RDMA_WFE's way) or poll.
    nap: bool = true,
    /// The codes families and the gathers of full passes ride the transport as well (TF_DS_ROCI_CODES, TF_DS_ROCI_GATHERS): the
    /// start line says so and the proxies' lines come once a pass of that many events.
    codes: bool = false,
    gathers: bool = false,
    /// After a failure stores the abort value into the words, the words are looked at again after this long (the fence is the NIC's word that
    /// the card is out of the control block; this looks at the result): a word that changed is stored again, and the look repeats until one
    /// whole settle passes with every word as stored, at most `abort_rounds` times (0: no look). The longest a failure waits: rounds x settle.
    abort_settle_us: u32 = 1_000,
    abort_rounds: u32 = 50,
    /// The longest a failure waits for the proxies to be out of their posts before the NIC takes the queue pairs away.
    proxy_out_ms: u32 = 100,
    /// The queue pairs wait in the card (TF_DS_ROCI_WAIT, rocinante_cd.zig): the engine thread writes every chain into its send queue behind a
    /// WAIT for the GPU's doorbell write when it reserves the send, so no proxy thread runs; one janitor thread, which sleeps in the kernel
    /// between 100 ms looks, watches the peers and reaps the completions. The proxy-only settings above are not read.
    wait_qp: bool = false,
    /// The janitor's period.
    janitor_ms: u32 = 100,
};

pub const FailReason = enum(u8) { none = 0, post, stamp, qp, watchdog, peer, host_wait, refused };

/// A spin lock over the sibling ring's links (one for the process: lanes close and fail rarely, and `check` holds it for two loads).
const RingLock = struct {
    held: std.atomic.Value(bool) = .init(false),
    fn lock(l: *RingLock) void {
        while (l.held.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
    fn unlock(l: *RingLock) void {
        l.held.store(false, .release);
    }
};
var ring_lock: RingLock = .{};

pub const Transport = struct {
    gpa: std.mem.Allocator,
    g: u8, // this node
    order: Order,
    ctl: *Control,
    /// The device address of `ctl` (cuMemHostGetDevicePointer); the host tests use the host address.
    ctl_dev: u64,
    nic: Nic,
    set: Settings,
    events: [2][n_classes]Event, // [shape][class]; shape 0 = cap 4,096, 1 = cap 2,048
    ticket: u64 = 0,
    want: [n_classes]u64 = @splat(0),
    sent: [rails][n_classes]u64 = @splat(@splat(0)),
    host: Counters = .{},
    rail_counters: [rails]Counters = @splat(.{}),
    wire: [rails][n_classes]Wire = @splat(@splat(.{})),
    pending: [rails][pending_len]Pending = undefined,
    pend_head: [rails]u64 = @splat(0),
    pend_tail: [rails]u64 = @splat(0),
    rail_prev: [rails]Counters = @splat(.{}),
    wire_prev: [rails][n_classes]Wire = @splat(@splat(.{})),
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(u8) = .init(0),
    fail_class: std.atomic.Value(u8) = .init(0),
    fail_peer: std.atomic.Value(u8) = .init(0),
    threads: [rails]?std.Thread = @splat(null),
    /// The other lane's transport when two passes of a prompt are in flight (each its own arena, control block, queue pairs and
    /// proxies): a failure on either stops both, and the host's check reads both.
    sibling: ?*Transport = null,
    /// Words the failure's look found changed after the abort value was stored (and stored again).
    abort_restored: u64 = 0,
    /// 1 while a failure's reaction is running on this transport (its look at the words takes milliseconds, on a proxy's thread or a
    /// sibling's): deinit waits for it, so the control block is not released under it.
    failing: std.atomic.Value(u32) = .init(0),
    /// A rail's proxy thread, once it runs, and set when it has left its loop (no post in progress, none to come): a failure raised by
    /// another thread waits for it before the NIC takes the queue pairs away (the proxy is the only poster on its rail).
    proxy_tid: [rails]std.atomic.Value(u32) = @splat(.init(0)),
    proxy_out: [rails]std.atomic.Value(u32) = @splat(.init(0)),
    /// Settings.wait_qp: the janitor, the words the engine thread has written into the send queues (the generation of each (rail, class)),
    /// whether it is inside a reservation (a failure waits for it to be out before the NIC takes the queues away) and which thread it is, and
    /// how far the janitor has walked the scheduling ring.
    janitor: ?std.Thread = null,
    wait_words: [rails][n_classes]u64 = @splat(@splat(0)),
    posting: std.atomic.Value(u32) = .init(0),
    poster_tid: std.atomic.Value(u32) = .init(0),
    cursor: std.atomic.Value(u64) = .init(0),

    /// The rows of the two fixed plans: a whole pass's (max_cap) and a resume unit's. Follows setCap.
    pub var shape_caps = [2]usize{ 4096, 2048 };

    pub fn init(gpa: std.mem.Allocator, g: u8, order: Order, ctl: *Control, ctl_dev: u64, nic: Nic, set: Settings) !*Transport {
        if (set.wait_qp and nic.prepost_fn == null) return error.NicCannotWait;
        const t = try gpa.create(Transport);
        t.* = .{ .gpa = gpa, .g = g, .order = order, .ctl = ctl, .ctl_dev = ctl_dev, .nic = nic, .set = set, .events = undefined };
        for (0..2) |s| for (0..n_classes) |c| {
            t.events[s][c] = buildEvent(@fromBackingInt(@intCast(c)), shape_caps[s], g, order);
        };
        return t;
    }

    /// Starts the two proxy threads; `cpus` are the cores to pin them to (null: no pinning).
    pub fn start(t: *Transport, cpus: ?[rails]u16) !void {
        if (t.set.wait_qp) {
            t.janitor = try std.Thread.spawn(.{}, janitorMain, .{t});
            return;
        }
    }

    pub fn deinit(t: *Transport) void {
        t.stop.store(true, .release);
        t.unlink();
        for (&t.threads) |*th| if (th.*) |x| {
            where.joinThread(x);
            th.* = null;
        };
        if (t.janitor) |x| {
            where.joinThread(x);
            t.janitor = null;
        }
        while (t.failing.load(.acquire) != 0) {
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 20 * std.time.ns_per_us };
            _ = std.c.nanosleep(&ts, null);
        }
        t.gpa.destroy(t);
    }

    fn word(t: *Transport, class: Class, src: u8, rail: usize) *Word {
        return &t.ctl.recv[@backingInt(class)][src][rail];
    }

    fn devAddr(t: *const Transport, p: *const anyopaque) u64 {
        return t.ctl_dev + (@intFromPtr(p) - @intFromPtr(t.ctl));
    }

    /// The sender's side of an event, queued behind the producer on `stream`: the scheduling entry, then one stream
    /// write of the ticket. `stream` has writeValue64(addr, value).
    pub fn send(t: *Transport, stream: anytype, class: Class, shape: u8) !void {
        const ticket = try t.reserve(class, shape);
        if (t.set.wait_qp) return t.sendBig(stream, ticket);
        try stream.writeValue64(t.devAddr(&t.ctl.doorbell.v), ticket);
        t.host.stream_writes += 1;
    }

    /// The stream write of a send's ticket in the card-waiting mode: the doorbell word that says it (see `go`).
    fn sendBig(t: *Transport, stream: anytype, ticket: u64) !void {
        try stream.writeValue64(t.devAddr(&t.ctl.doorbell.v), t.go(ticket));
        t.host.stream_writes += 1;
    }

    /// The doorbell word that says `ticket`: the ticket itself, or in the card-waiting mode its big-endian bytes (the card compares a WAIT's
    /// word as a big-endian number, measured, so the doorbell is stored that way by every writer: the host's stream write, a producer
    /// kernel's store of the value the host hands it (Roci.reserve), the host lane's store, the failure's abort value).
    pub fn go(t: *const Transport, ticket: u64) u64 {
        return if (t.set.wait_qp) std.mem.nativeToBig(u64, ticket) else ticket;
    }

    /// The ticket a doorbell word says (the inverse of `go`).
    pub fn ticketOf(t: *const Transport, w: u64) u64 {
        return if (t.set.wait_qp) std.mem.bigToNative(u64, w) else w;
    }

    /// The host's half of a send whose doorbell a kernel rings (a producer that stores `ticket` into the doorbell line itself,
    /// once its sent rows are written): the half-ring backpressure, the next ticket and its scheduling entry, in the same order
    /// `send` makes them. The proxies take tickets in order, so the kernel that rings this one must be launched before the
    /// next reserve and its ring must follow every earlier ticket's: one launch after the other on one stream does both. A
    /// ticket reserved and never rung ends in the watchdog's failure, like a stream write that never ran.
    pub fn reserve(t: *Transport, class: Class, shape: u8) !u64 {
        return t.reserveEntry(class, shape, 0, 0);
    }

    /// reserve for an event of `n` rows of any count (buildEventN's plan): the rows ride in the scheduling entry.
    pub fn reserveN(t: *Transport, class: Class, n: usize) !u64 {
        return t.reserveEntry(class, any_shape, n, 0);
    }

    /// reserveN for an event with buildEventX's extra number.
    pub fn reserveX(t: *Transport, class: Class, n: usize, x: usize) !u64 {
        return t.reserveEntry(class, any_shape, n, x);
    }

    /// send for an event of `n` rows of any count.
    pub fn sendN(t: *Transport, stream: anytype, class: Class, n: usize) !void {
        try t.sendX(stream, class, n, 0);
    }

    /// send for an event of `n` rows with buildEventX's extra number `x`.
    pub fn sendX(t: *Transport, stream: anytype, class: Class, n: usize, x: usize) !void {
        const ticket = try t.reserveX(class, n, x);
        if (t.set.wait_qp) return t.sendBig(stream, ticket);
        try stream.writeValue64(t.devAddr(&t.ctl.doorbell.v), ticket);
        t.host.stream_writes += 1;
    }

    /// A send posted by the host (the host lane: nothing else rings this doorbell): ring slot `slot`, `count` records for `.rec`. The
    /// ticket's entry is written, then the doorbell, which the proxy watches in host memory like a stream write's.
    pub fn sendRec(t: *Transport, class: Class, slot: u8, count: usize) !void {
        std.debug.assert(class == .rec or class == .rech);
        const ticket = try t.reserveEntry(class, slot, count, 0);
        if (t.set.wait_qp) {
            @atomicStore(u64, &t.ctl.doorbell.v, t.go(ticket), .release);
            t.host.stream_writes += 1;
            return;
        }
        @atomicStore(u64, &t.ctl.doorbell.v, ticket, .release);
        t.host.stream_writes += 1;
    }

    /// Whether the partner's words of `class` have reached `want` on both rails (the cyclic compare a stream wait makes).
    pub fn hostWordReached(t: *Transport, class: Class, want: u64) bool {
        var srcs: [world]u8 = undefined;
        for (sourcesOf(class, t.g, &srcs)) |s| for (0..rails) |r| {
            const have = @atomicLoad(u64, &t.word(class, s, r).v, .acquire);
            if (@as(i64, @bitCast(have -% want)) < 0) return false;
        };
        return true;
    }

    /// What one host wait did: the looks it took at the words, the naps among them, and its length.
    pub const HostWait = struct {
        looks: u32 = 0,
        naps: u32 = 0,
        ns: u64 = 0,
    };

    /// The first of the partner's words of `class` that has not reached `want` (the cyclic compare a stream wait makes), or null when
    /// every one has.
    fn firstWordBehind(t: *Transport, class: Class, want: u64) ?*u64 {
        var srcs: [world]u8 = undefined;
        for (sourcesOf(class, t.g, &srcs)) |s| for (0..rails) |r| {
            const w = &t.word(class, s, r).v;
            if (@as(i64, @bitCast(@atomicLoad(u64, w, .acquire) -% want)) < 0) return w;
        };
        return null;
    }

    /// The host waits for the partner's words of `class` to reach `want` (no stream): true when they have, false after `deadline_ns`
    /// (0: no deadline). Between looks the thread naps on the first word still behind, with the Arm wait the proxies use (the write that
    /// completes the word ends the nap), at most `host_nap_us` at a time so the other rail's word is looked at too; a CPU that cannot
    /// nap sleeps `host_nap_us` between looks. A failed transport ends the wait with an error. `st` receives the wait's looks, naps and length.
    pub fn waitHostStats(t: *Transport, class: Class, want: u64, deadline_ns: u64, st: *HostWait) !bool {
        const t0 = nowNs();
        defer st.ns +%= nowNs() - t0;
        const can_nap = nap.available() and nap.clock() == .fine;
        const ticks = nap.ticksPerUs();
        while (t.firstWordBehind(class, want)) |w| {
            st.looks += 1;
            try t.check();
            if (deadline_ns != 0 and nowNs() - t0 >= deadline_ns) return false;
            if (can_nap) {
                const seen = @atomicLoad(u64, w, .acquire);
                st.naps += 1;
                _ = nap.napOnce(w, seen, nap.now() + ticks * host_nap_us);
            } else {
                const ts: std.c.timespec = .{ .sec = 0, .nsec = host_nap_us * std.time.ns_per_us };
                _ = std.c.nanosleep(&ts, null);
            }
        }
        try t.check(); // (a failure stores the abort value into every word: they read as reached)
        return true;
    }

    /// waitHostStats without the statistics.
    pub fn waitHost(t: *Transport, class: Class, want: u64, deadline_ns: u64) !bool {
        var st: HostWait = .{};
        return t.waitHostStats(class, want, deadline_ns, &st);
    }

    fn reserveEntry(t: *Transport, class: Class, shape: u8, rows: usize, x: usize) !u64 {
        if (t.set.wait_qp) return t.reserveWait(class, shape, rows, x);
        try t.check();
        // the host may run ahead of the proxies by half the ring: past that it waits for them (they advance as the stream
        // reaches the doorbells), and a transport that does not advance in `dead_ms` has failed
        if (t.ticket -% t.doneMin() >= sched_len / 2) {
            const t0 = nowNs();
            while (t.ticket -% t.doneMin() >= sched_len / 2) {
                try t.check();
                if (nowNs() - t0 > t.set.dead_ms * std.time.ns_per_ms) {
                    t.fail(.host_wait, .own1, 0);
                    return error.RociFailed;
                }
                const ts: std.c.timespec = .{ .sec = 0, .nsec = 20 * std.time.ns_per_us };
                _ = std.c.nanosleep(&ts, null);
            }
        }
        t.ticket += 1;
        const slot = &t.ctl.sched[t.ticket % sched_len];
        @atomicStore(u32, &t.ctl.sched_x[t.ticket % sched_len], @intCast(x), .monotonic);
        @atomicStore(u64, slot, if (shape == any_shape) schedEntryN(t.ticket, class, rows) else schedEntry(t.ticket, class, shape) | (@as(u64, @intCast(rows)) << 48), .release);
        t.host.events += 1;
        t.host.doorbells += 1;
        return t.ticket;
    }

    /// The event a scheduling entry names, as a proxy builds it: the host lane's record events, the plan of any row count, or a fixed plan.
    /// (Settings.wait_qp only: the engine thread builds the chains the proxies would have.)
    fn eventOf(t: *Transport, e: u64, x: usize, built: *Event) *const Event {
        if (schedClass(e) == .rec or schedClass(e) == .rech) {
            built.* = buildRecEvent(schedClass(e), t.g, schedShape(e), schedRows(e));
            return built;
        }
        if (schedShape(e) == any_shape) {
            built.* = buildEventX(schedClass(e), schedRows(e), x, t.g, t.order);
            return built;
        }
        return &t.events[schedShape(e)][@backingInt(schedClass(e))];
    }

    /// reserveEntry when the queue pairs wait in the card: the ticket and its scheduling entry as always, then every chain of the event on
    /// every rail written into its send queue behind a WAIT for the doorbell word to reach this ticket (the host lane's record events,
    /// which nothing else rings, with no WAIT: they leave at once), one ring of the queue a chain. The GPU's write of the ticket into
    /// the doorbell, which `send` makes next, is what lets them go. `posting` says the engine thread is inside this: a failure waits for it
    /// before the NIC takes the queues away.
    fn reserveWait(t: *Transport, class: Class, shape: u8, rows: usize, x: usize) !u64 {
        t.poster_tid.store(@intCast(std.Thread.getCurrentId()), .monotonic);
        t.posting.store(1, .seq_cst);
        defer t.posting.store(0, .seq_cst);
        if (t.failed.load(.seq_cst) != 0) return error.RociFailed;
        try t.check();
        // the janitor reads the scheduling ring from its cursor: the host does not lap it
        if (t.ticket -% t.cursor.load(.acquire) >= sched_len / 2) {
            const t0 = nowNs();
            const prev_place = where.enter(.roci_reserve, @as(u32, @backingInt(class)), 0);
            defer where.leave(prev_place);
            while (t.ticket -% t.cursor.load(.acquire) >= sched_len / 2) {
                try t.check();
                if (nowNs() - t0 > t.set.dead_ms * std.time.ns_per_ms) {
                    t.fail(.host_wait, .own1, 0);
                    return error.RociFailed;
                }
                const ts: std.c.timespec = .{ .sec = 0, .nsec = 20 * std.time.ns_per_us };
                _ = std.c.nanosleep(&ts, null);
            }
        }
        t.ticket += 1;
        const ticket = t.ticket;
        const entry = if (shape == any_shape) schedEntryN(ticket, class, rows) else schedEntry(ticket, class, shape) | (@as(u64, @intCast(rows)) << 48);
        @atomicStore(u32, &t.ctl.sched_x[ticket % sched_len], @intCast(x), .monotonic);
        @atomicStore(u64, &t.ctl.sched[ticket % sched_len], entry, .release);
        t.host.events += 1;
        t.host.doorbells += 1;
        t.reserved.store(ticket, .seq_cst);
        var built: Event = undefined;
        const ev = t.eventOf(entry, x, &built);
        const cls = @backingInt(ev.class);
        const wait_for: u64 = if (class == .rec or class == .rech) 0 else ticket;
        for (0..rails) |r| t.wait_words[r][cls] += 1;
        for (ev.peerSlice(), 0..) |peer, k| for (0..rails) |r| {
            const ch = &ev.chains[k][r];
            t.nic.prepost_fn.?(t.nic.ctx, @intCast(r), peer, ev.class, ch, t.wait_words[r][cls], wait_for) catch {
                t.fail(.post, ev.class, peer);
                return error.RociFailed;
            };
            t.host.posts += 1;
            t.host.data_writes += ch.n;
            t.host.words += 1;
            t.host.bytes_out += ch.bytes();
        };
        return ticket;
    }

    /// Host tests only: a failure that stores the abort value into the doorbell BEFORE the card has taken the queue pairs away, the order the
    /// design forbids (every armed WAIT would release its chain), to see that the ordering test can fail.
    pub var abort_doorbell_first: bool = false;

    /// The device address of the doorbell line (where a stream write, or a kernel's ring, stores a ticket).
    pub fn doorbellAddr(t: *const Transport) u64 {
        return t.devAddr(&t.ctl.doorbell.v);
    }

    /// The receiver's side: the next consumer on `stream` waits for every source's words on both rails.
    pub fn wait(t: *Transport, stream: anytype, class: Class) !void {
        try t.check();
        t.want[@backingInt(class)] += 1;
        const v = t.want[@backingInt(class)];
        var srcs: [world]u8 = undefined;
        var addrs: [world * rails]u64 = undefined;
        var vals: [world * rails]u64 = undefined;
        var n: usize = 0;
        for (sourcesOf(class, t.g, &srcs)) |s| for (0..rails) |r| {
            addrs[n] = t.devAddr(&t.word(class, s, r).v);
            vals[n] = v;
            n += 1;
        };
        try stream.waitValues64(addrs[0..n], vals[0..n]);
        t.host.wait_groups += 1;
    }

    fn doneMin(t: *Transport) u64 {
        return @min(@atomicLoad(u64, &t.ctl.done[0].v, .acquire), @atomicLoad(u64, &t.ctl.done[1].v, .acquire));
    }

    /// Whether the transport has failed (the host reads it at every enqueue and at its own synchronisation points).
    pub fn check(t: *Transport) !void {
        if (t.failed.load(.acquire) != 0 or @atomicLoad(u64, &t.ctl.abort.v, .acquire) != 0) return error.RociFailed;
        // (the ring links change while lanes close: read the sibling under the lock, so it is not freed under the read)
        ring_lock.lock();
        defer ring_lock.unlock();
        if (t.sibling) |s| if (s.failed.load(.acquire) != 0 or @atomicLoad(u64, &s.ctl.abort.v, .acquire) != 0) return error.RociFailed;
    }

    /// Makes `a` and `b` the two lanes of one prompt: each fails the other.
    pub fn pair(a: *Transport, b: *Transport) void {
        ring_lock.lock();
        defer ring_lock.unlock();
        a.sibling = b;
        b.sibling = a;
    }

    /// Makes `a`, `b` and `c` the three lanes of one prompt (TF_DS_ROCI_STAGGER_LANES=3): a ring a -> b -> c -> a, so a failure on any
    /// one reaches the other two through `fail`'s sibling call (each returns at its first line once it has failed).
    pub fn ring3(a: *Transport, b: *Transport, c: *Transport) void {
        ring_lock.lock();
        defer ring_lock.unlock();
        a.sibling = b;
        b.sibling = c;
        c.sibling = a;
    }

    /// Takes this transport out of its sibling ring before it goes: the one that points at it points past it (null when none is left), so no
    /// lane keeps a pointer to a closed one.
    pub fn unlink(t: *Transport) void {
        ring_lock.lock();
        defer ring_lock.unlock();
        const next = t.sibling orelse return;
        var p = next;
        var hops: usize = 0;
        while (p.sibling != null and p.sibling.? != t) : (hops += 1) {
            if (hops > 4) break; // (a ring has at most three lanes)
            p = p.sibling.?;
        }
        if (p.sibling != null and p.sibling.? == t) p.sibling = if (next == p) null else next;
        t.sibling = null;
    }

    pub fn failure(t: *const Transport) ?struct { reason: FailReason, class: Class, peer: u8 } {
        const r = t.failed.load(.acquire);
        if (r == 0) return null;
        return .{ .reason = @fromBackingInt(@intCast(r)), .class = @fromBackingInt(@intCast(t.fail_class.load(.monotonic))), .peer = t.fail_peer.load(.monotonic) };
    }

    /// The failure reaction, in order: set the abort word; tell the peers and move the queue pairs to error and fence the card
    /// off the control block (so no late genuine word, WRITE or atomic, can overwrite the abort value); then store the abort
    /// value into every awaited word and the doorbell, which releases every queued wait. Whoever gets here first wins; the
    /// engine exits and restarts.
    pub fn fail(t: *Transport, reason: FailReason, class: Class, peer: u8) void {
        if (t.failed.cmpxchgStrong(0, @backingInt(reason), .acq_rel, .acquire) != null) return;
        _ = t.failing.fetchAdd(1, .acq_rel);
        defer _ = t.failing.fetchSub(1, .acq_rel);
            var pb: [640]u8 = undefined;
        t.fail_class.store(@backingInt(class), .monotonic);
        t.fail_peer.store(peer, .monotonic);
        @atomicStore(u64, &t.ctl.abort.v, 1, .release);
        t.wakeAll(); // (a parked proxy leaves its sleep and, seeing the failure, its loop)
        t.awaitProxiesOut();
        if (t.set.wait_qp) t.awaitPostingOut();
        if (builtin.is_test and abort_doorbell_first) @atomicStore(u64, &t.ctl.doorbell.v, t.go(abort_value), .release);
        t.nic.fail_fn(t.nic.ctx);
        t.storeAbort();
        @atomicStore(u64, &t.ctl.doorbell.v, abort_value, .release);
        // (the card-waiting mode stores it big-endian; the queue pairs are gone by now, so the native word just above releases nothing)
        if (t.set.wait_qp) @atomicStore(u64, &t.ctl.doorbell.v, t.go(abort_value), .release);
        // the other lane's streams may wait on words its peers will not send once this rank stops: stop it too (it fails
        // once; its own sibling call returns at the first line)
        // The sibling is taken under the ring lock and held by its `failing` count from there, so its deinit (which unlinks, then waits
        // for the count) cannot free it while this call runs; a lane already unlinked is not in the ring any more and is not reached.
        ring_lock.lock();
        const sib = t.sibling;
        if (sib) |s| _ = s.failing.fetchAdd(1, .acq_rel);
        ring_lock.unlock();
        if (sib) |s| {
            defer _ = s.failing.fetchSub(1, .acq_rel);
            s.fail(.peer, class, peer);
        }
    }

    /// Waits (short sleeps, at most `proxy_out_ms`) for each rail's proxy to have left its loop, so no post is running when the NIC takes
    /// its queue pairs away. A proxy that raised the failure itself is out of its post already and is not waited for; a proxy that
    /// is still in its post after the bound is named in the log and the failure goes on.
    fn awaitProxiesOut(t: *Transport) void {
        const me: u32 = @intCast(std.Thread.getCurrentId());
        const t0 = nowNs();
        for (0..rails) |r| {
            if (t.threads[r] == null) continue;
            while (t.proxy_out[r].load(.acquire) == 0 and t.proxy_tid[r].load(.acquire) != me) {
                if (nowNs() - t0 > t.set.proxy_out_ms * std.time.ns_per_ms) {
                    std.log.err("ROCI failed on node {d}: rail {c}'s proxy was still posting after {d} ms; the failure goes on", .{ t.g, @as(u8, 'A') + @as(u8, @intCast(r)), t.set.proxy_out_ms });
                    break;
                }
                const ts: std.c.timespec = .{ .sec = 0, .nsec = 50 * std.time.ns_per_us };
                _ = std.c.nanosleep(&ts, null);
            }
        }
    }

    /// awaitProxiesOut for the card-waiting mode: waits (short sleeps, at most `proxy_out_ms`) for the engine thread to be out of its
    /// reservation, so no chain is being written into a send queue the NIC takes away. A failure the engine thread raises itself, inside
    /// its reservation, does not wait for itself.
    fn awaitPostingOut(t: *Transport) void {
        const me: u32 = @intCast(std.Thread.getCurrentId());
        const t0 = nowNs();
        while (t.posting.load(.seq_cst) != 0 and t.poster_tid.load(.monotonic) != me) {
            if (nowNs() - t0 > t.set.proxy_out_ms * std.time.ns_per_ms) {
                std.log.err("ROCI failed on node {d}: the engine thread was still posting after {d} ms; the failure goes on", .{ t.g, t.set.proxy_out_ms });
                break;
            }
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 50 * std.time.ns_per_us };
            _ = std.c.nanosleep(&ts, null);
        }
    }

    /// Stores the abort value into every awaited word, then checks that it held: after each settle every word is read, a word that is not
    /// the abort value (a card's atomic that was in flight and landed after the store: old + 1) is stored again, until a whole settle passes
    /// with every word as stored. A word that never holds is named in the log.
    fn storeAbort(t: *Transport) void {
        for (&t.ctl.recv) |*c| for (c) |*s| for (s) |*w| @atomicStore(u64, &w.v, abort_value, .release);
        if (t.set.abort_rounds == 0) return;
        var round: u32 = 0;
        while (round < t.set.abort_rounds) : (round += 1) {
            const ts: std.c.timespec = .{ .sec = 0, .nsec = @as(isize, t.set.abort_settle_us) * std.time.ns_per_us };
            _ = std.c.nanosleep(&ts, null);
            var changed: u32 = 0;
            for (&t.ctl.recv) |*c| for (c) |*s| for (s) |*w| {
                if (@atomicLoad(u64, &w.v, .acquire) != abort_value) {
                    @atomicStore(u64, &w.v, abort_value, .release);
                    changed += 1;
                }
            };
            if (changed == 0) return;
            t.abort_restored += changed;
        }
        std.log.err("ROCI failed on node {d}: the awaited words did not hold the abort value through a {d} us settle in {d} rounds", .{ t.g, t.set.abort_settle_us, t.set.abort_rounds });
    }

    /// The per-peer progress watchdog of one rail: for any class where this rail has posted more than a peer's words
    /// received, if no word of that peer moved for `dead_ms`, the pass stops naming the peer and the class.
    fn watch(t: *Transport, rail: u8, last: *u64, sum: *u64, ctr: *Counters) void {
        var owed: ?struct { c: u8, p: u8 } = null;
        var s: u64 = 0;
        for (0..n_classes) |c| {
            var srcs: [world]u8 = undefined;
            for (sourcesOf(@fromBackingInt(@intCast(c)), t.g, &srcs)) |p| {
                const v = @atomicLoad(u64, &t.word(@fromBackingInt(@intCast(c)), p, rail).v, .acquire);
                s +%= v;
                if (v < t.sent[rail][c] and owed == null) owed = .{ .c = @intCast(c), .p = p };
            }
        }
        const now = nowNs();
        if (s != sum.* or owed == null) {
            sum.* = s;
            last.* = now;
            return;
        }
        const age_ms = (now - last.*) / std.time.ns_per_ms;
        ctr.watchdog_max_age_ms = @max(ctr.watchdog_max_age_ms, age_ms);
        if (age_ms >= t.set.dead_ms) t.fail(.watchdog, @fromBackingInt(@intCast(owed.?.c)), owed.?.p);
    }

    /// Events whose words from every source have all arrived on this rail: their post-to-last-word time into the class's Wire.
    fn reapWire(t: *Transport, rail: u8) void {
        while (t.pend_head[rail] < t.pend_tail[rail]) {
            const p = t.pending[rail][t.pend_head[rail] % pending_len];
            var srcs: [world]u8 = undefined;
            for (sourcesOf(@fromBackingInt(@intCast(p.class)), t.g, &srcs)) |src| {
                if (@atomicLoad(u64, &t.word(@fromBackingInt(@intCast(p.class)), src, rail).v, .acquire) < p.want) return;
            }
            const dt = nowNs() - p.t_ns;
            const w = &t.wire[rail][p.class];
            w.n += 1;
            w.sum_ns += dt;
            w.max_ns = @max(w.max_ns, dt);
            t.pend_head[rail] += 1;
        }
    }

    /// One line a pass and rail from its proxy: the posts, data WRITEs, words and bytes it made since its last line, and the
    /// post-to-last-word time of each class it saw complete (mean / max, microseconds).
    fn railLine(t: *Transport, rail: u8) void {
        const now = t.rail_counters[rail];
        const prev = t.rail_prev[rail];
        t.rail_prev[rail] = now;
        var w = std.Io.Writer.fixed(&buf);
        for (0..n_classes) |c| {
            const cur = t.wire[rail][c];
            const pw = t.wire_prev[rail][c];
            t.wire_prev[rail][c] = cur;
            if (cur.n == pw.n) continue;
            const n = cur.n - pw.n;
            w.print(" {s} {d}/{d} {d}", .{ @tagName(@as(Class, @fromBackingInt(@intCast(c)))), (cur.sum_ns - pw.sum_ns) / n / 1000, cur.max_ns / 1000, n }) catch return;
        }
        std.log.info("{s}", .{w.buffered()});
    }

    pub fn counters(t: *const Transport) Counters {
        var c = t.host;
        for (t.rail_counters) |r| c.add(r);
        return c;
    }

    /// The pass's counter line.
    pub fn counterLine(t: *const Transport, w: *std.Io.Writer, label: []const u8) !void {
        const c = t.counters();
            label,            c.events,      c.doorbells,           c.stream_writes,                              c.wait_groups,
            c.data_writes,    c.words,       c.posts,               c.bytes_out,                                  c.wakes,
        });
    }
};

fn pinCpu(c: u16) void {
    if (c >= @bitSizeOf(std.os.linux.cpu_set_t)) return;
    var set: std.os.linux.cpu_set_t = @splat(0);
    const bits = @bitSizeOf(usize);
    set[c / bits] |= @as(usize, 1) << @intCast(c % bits);
    std.os.linux.sched_setaffinity(0, &set) catch {};
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// One rail's proxy: waits on the doorbell (a nap on its line, a backstop for a write the nap missed), posts the
/// chains of every send the doorbell has passed, reaps its completions and watches its peers.
    t.proxy_tid[rail].store(@intCast(std.Thread.getCurrentId()), .release);
    defer t.proxy_out[rail].store(1, .release);
}

    if (cpu) |c| pinCpu(c);
    const ctr = &t.rail_counters[rail];
    var done: u64 = 0;
    var last_progress = nowNs();
    var progress_sum: u64 = 0;
    const ticks = nap.ticksPerUs();
    const can_nap = t.set.nap and nap.available() and nap.clock() == .fine;
    while (!t.stop.load(.acquire)) {
        t.reapWire(rail);
        const d = @atomicLoad(u64, &t.ctl.doorbell.v, .acquire);
        if (d == abort_value or t.failed.load(.acquire) != 0) break;
        if (d == done) {
                const dl = nap.now() + ticks * t.set.backstop_us;
            } else std.atomic.spinLoopHint();
            ctr.wakes += 1;
                t.watch(rail, &last_progress, &progress_sum, ctr);
                t.nic.reap_fn(t.nic.ctx, rail) catch {
                    t.fail(.qp, .own1, 0);
                    break;
                };
            continue;
        }
        const t0 = nowNs();
        while (done < d) {
            const i = done + 1;
            var e = @atomicLoad(u64, &t.ctl.sched[i % sched_len], .acquire);
            var spins: u32 = 0;
            while (schedTicket(e) != @as(u32, @truncate(i))) : (spins += 1) {
                if (spins > 20_000) {
                    t.fail(.stamp, .own1, 0);
                    return;
                }
                std.atomic.spinLoopHint();
                e = @atomicLoad(u64, &t.ctl.sched[i % sched_len], .acquire);
            }
            var built: Event = undefined;
            const ev: *const Event = if (schedClass(e) == .rec or schedClass(e) == .rech) blk: {
                built = buildRecEvent(schedClass(e), t.g, schedShape(e), schedRows(e));
                break :blk &built;
            } else if (schedShape(e) == any_shape) blk: {
                built = buildEventX(schedClass(e), schedRows(e), @atomicLoad(u32, &t.ctl.sched_x[i % sched_len], .monotonic), t.g, t.order);
                break :blk &built;
            } else &t.events[schedShape(e)][@backingInt(schedClass(e))];
            const cls = @backingInt(ev.class);
            t.sent[rail][cls] += 1;
            for (ev.peerSlice(), 0..) |peer, k| {
                const ch = &ev.chains[k][rail];
                t.nic.post(rail, peer, ev.class, ch, t.sent[rail][cls]) catch {
                    t.fail(.post, ev.class, peer);
                    return;
                };
                ctr.posts += 1;
                ctr.data_writes += ch.n;
                ctr.words += 1;
                ctr.bytes_out += ch.bytes();
            }
            done = i;
            @atomicStore(u64, &t.ctl.done[rail].v, done, .release);
            if (t.pend_tail[rail] - t.pend_head[rail] < pending_len) {
                t.pending[rail][t.pend_tail[rail] % pending_len] = .{ .class = @intCast(cls), .t_ns = nowNs(), .want = t.sent[rail][cls] };
                t.pend_tail[rail] += 1;
            }
            if (done % eventsPerPass(t.set.codes, t.set.gathers) == 0) t.railLine(rail);
        }
        ctr.post_ns_max = @max(ctr.post_ns_max, nowNs() - t0);
        last_progress = nowNs();
    }
}

/// The card-waiting mode's one thread a transport. It owns no posting: the engine thread wrote every chain when it reserved it. Each
/// period (`janitor_ms`, in a futex wait that a stop or a failure ends at once, so the thread is asleep in the kernel in between) it
/// walks the scheduling ring up to the doorbell (the releases the card has been given, which the watchdog's `sent` counts), runs the
/// peers' watchdog on both rails and reaps the completions of both rails, where an error completion fails the transport.
fn janitorMain(t: *Transport) void {
    var last_progress: [rails]u64 = @splat(nowNs());
    var progress_sum: [rails]u64 = @splat(0);
    var walked: u64 = 0;
    var line_at: u64 = 0;
    var line_prev: Counters = .{};
    const ctr = &t.rail_counters[0];
    const ts: std.os.linux.timespec = .{ .sec = t.set.janitor_ms / 1000, .nsec = (t.set.janitor_ms % 1000) * std.time.ns_per_ms };
    while (!t.stop.load(.acquire)) {
        const seq = t.wake_seq[0].load(.acquire);
        if (!t.stop.load(.acquire) and t.failed.load(.acquire) == 0) _ = std.os.linux.futex_4arg(&t.wake_seq[0].raw, .{ .cmd = .WAIT, .private = true }, seq, &ts);
        ctr.wakes += 1;
        if (t.stop.load(.acquire) or t.failed.load(.acquire) != 0) break;
        const d = t.ticketOf(@atomicLoad(u64, &t.ctl.doorbell.v, .acquire));
        if (d >= abort_value) break;
        const upto = @min(d, t.reserved.load(.acquire));
        while (walked < upto) {
            const i = walked + 1;
            const e = @atomicLoad(u64, &t.ctl.sched[i % sched_len], .acquire);
            if (schedTicket(e) != @as(u32, @truncate(i))) {
                t.fail(.stamp, .own1, 0);
                return;
            }
            const cls = @backingInt(schedClass(e));
            for (0..rails) |r| t.sent[r][cls] += 1;
            walked = i;
        }
        t.cursor.store(walked, .release);
        for (0..rails) |r| {
            t.watch(@intCast(r), &last_progress[r], &progress_sum[r], ctr);
            t.nic.reap_fn(t.nic.ctx, @intCast(r)) catch {
                t.fail(.qp, .own1, 0);
                return;
            };
        }
        if (walked / eventsPerPass(t.set.codes, t.set.gathers) != line_at) {
            line_at = walked / eventsPerPass(t.set.codes, t.set.gathers);
            const now = t.host;
            std.log.info("ROCI wait line, released {d}: posts {d} data_writes {d} words {d} bytes_out {d} janitor_wakes {d}", .{ walked, now.posts - line_prev.posts, now.data_writes - line_prev.data_writes, now.words - line_prev.words, now.bytes_out - line_prev.bytes_out, ctr.wakes - line_prev.wakes });
            line_prev = now;
            line_prev.wakes = ctr.wakes;
        }
    }
}

comptime {
    // the control block's lines: the doorbell and the abort never share one with a counter the NIC writes
    std.debug.assert(@sizeOf(Word) == line and @alignOf(Word) == line);
    std.debug.assert(@offsetOf(Control, "sched") % line == 0);
}