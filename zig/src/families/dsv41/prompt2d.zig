//! The exact 2D-TP4 split on the prompt path (DESIGN-zig-tp4.md sections 3-4; model.py Comm2D, wo_2d, moe_2d and
//! exl3/experts2d.py on four nodes): node g = 2 p + r is TP2 rank r of pair p. A node computes every output column it
//! holds whole, over its TP2 slice's K in TP2's order, and every exchange only moves bytes, so after an exchange a
//! node holds TP2 rank r's bytes: `quarters` assembles TP2's [2, n, N] gather of the two ranks' partials (which the
//! mixes then add in rank order, as on TP2), `catRank` the rank's whole wo_a output or expert intermediate from the
//! two pairs' parts. A prompt chunk's exchanges are Rocinante's (rocinante.zig, roci_engine.zig: owner sums, codes and gathers
//! written into the peers' arenas); decode windows' and the drafter's ride the RDMA rings (Two.rings, ring2d.zig), written straight
//! into place, and a ring that cannot take an exchange ends it by name: no other transport exists.
//!
//! prompt.zig takes it through thin hooks behind Engine.two; a TP2 engine has none and runs as before. The decode
//! rounds' part is round2d.zig's, on this node state.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const mx = @import("mx.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const ring2d = @import("ring2d.zig");
const rdma_mod = @import("rdma.zig");
const round2d = @import("round2d.zig");
const round_rows = @import("round.zig").max_rows;
const Link = @import("link.zig").Link;
const Ports = @import("link.zig").Ports;
const Comm = @import("comm.zig").Comm;
const roci_engine = @import("roci_engine.zig");
const rocinante = @import("rocinante.zig");

const nccl = cuda.nccl;

/// TF_DS_2D_YX=1 (model.zig sets it, every rank alike; off by default): prompt.gather exchanges the sublayer partials as
/// their bf16 sums (Two.ySums) instead of the four fp32 quarters.
pub var y_sums: bool = false;

/// The split of node `g` of `world` nodes: TP over 1 or 2, the 2D split over 4.
pub fn split(g: u32, world: u32) !plan.Split {
    return switch (world) {
        1, 2 => .{ .rank = g, .world = world },
        4 => .{ .rank = g % 2, .world = 2, .pair = g / 2 },
        else => error.UnsupportedWorld,
    };
}

/// How long a follower retries its connect to rank 0: TF_DS_CONNECT_S seconds, 300 by default. Rank 0's ports listen
/// from its start (Fan.listenAt, before its weights load), so this covers only rank 0's process starting late.
pub fn connectSeconds() u32 {
    const v = std.c.getenv("TF_DS_CONNECT_S") orelse return 300;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 300;
}

/// The links of a run: rank 0 listens on PORT + g - 1 for follower g, all the followers' ports at once, and takes them in
/// order (a follower that is ready early waits in its port's backlog).
pub const Fan = struct {
    links: [3]?Link = .{ null, null, null },
    ports: ?Ports = null, // rank 0's, from listenAt until join has taken every follower

    pub fn open(io: std.Io, ip: [4]u8, port: u16, rank: u32, world: u32) !Fan {
        return openAt(io, ip, null, port, rank, world);
    }

    /// open, rank 0 listening for follower g on `listen[g - 1]` when given (each follower's own link address of this
    /// node: on a ring without a switch a follower reaches the head at the address of the cable it comes in on), else
    /// on `ip` for all of them.
    pub fn openAt(io: std.Io, ip: [4]u8, listen: ?[3][4]u8, port: u16, rank: u32, world: u32) !Fan {
        var f = try listenAt(ip, listen, port, rank, world);
        errdefer f.close();
        try f.join(io, ip, port, rank, world, connectSeconds());
        return f;
    }

    /// openAt's first half: rank 0's ports for all its followers, listening from here on (Model.open calls it before
    /// the weights load, so a follower that loads faster than rank 0 is not left retrying); nothing on a follower.
    pub fn listenAt(ip: [4]u8, listen: ?[3][4]u8, port: u16, rank: u32, world: u32) !Fan {
        if (world > 4) return error.UnsupportedWorld;
        var f: Fan = .{};
        if (rank != 0 or world < 2) return f;
        var ips: [3][4]u8 = undefined;
        for (ips[0 .. world - 1], 0..) |*a, i| a.* = if (listen) |l| l[i] else ip;
        f.ports = try Ports.open(ips[0 .. world - 1], port);
        return f;
    }

    /// openAt's second half: rank 0 takes its followers in order; a follower connects to its port on rank 0, retrying
    /// for `seconds` while rank 0 starts.
    pub fn join(f: *Fan, io: std.Io, ip: [4]u8, port: u16, rank: u32, world: u32, seconds: u32) !void {
        if (world < 2) return;
        if (rank == 0) {
            for (1..world) |g| f.links[g - 1] = try f.ports.?.accept(g);
            f.ports = null;
            return;
        }
        const at = port + @as(u16, @intCast(rank - 1));
        f.links[0] = Link.connect(io, ip, at, seconds) catch |err| {
            if (err == error.ConnectFailed) std.log.err("rank {d}: rank 0 did not answer at {d}.{d}.{d}.{d}:{d} in {d} s (TF_DS_CONNECT_S)", .{ rank, ip[0], ip[1], ip[2], ip[3], at, seconds });
            return err;
        };
    }

    pub fn close(f: *Fan) void {
        if (f.ports) |*p| p.close();
        f.ports = null;
        for (&f.links) |*l| {
            if (l.*) |*x| x.close();
            l.* = null;
        }
    }

    /// The start-up exchange over every link, with this rank's Settings (Model.open's): rank 0 sends its own and waits for each
    /// follower's answer, a follower checks rank 0's against its own and answers. A difference ends every rank with
    /// SettingsDiffer, its words on stderr (a follower of fewer streams than rank 0 serves otherwise refuses the first step
    /// of a stream past its own and rank 0 waits for it in an exchange for good). Each answer carries the follower's weight
    /// digest (`digest`: this rank's); `heard` gets rank 0's Settings and the digests. The Comm it returns holds the rank and
    /// the world; its host links are the caller's to set.
    pub fn settingsChecked(f: *const Fan, rank: u32, world: u32, mine: ?Settings, digest: Digest, heard: ?*Heard) !Comm {
        var local: Heard = .{};
        const h = heard orelse &local;
        if (world > 1) {
            var why: [1024]u8 = undefined;
            const differ = try f.exchange(rank, mine, digest, &why, h);
            if (differ.len > 0) {
                std.debug.print("tf-dsv41: the ranks' settings differ: {s}\n", .{differ});
                return error.SettingsDiffer;
            }
        } else {
            h.* = .{ .lead = mine };
            h.weights[0] = digest;
        }
        return Comm.init(rank, world);
    }

    /// The links' part of settingsChecked: rank 0 sends `mine` down every link and hears every answer (lead), a follower reads
    /// it, checks and answers (follow). `heard` starts over with this rank's own digest at its rank. "" when every rank can
    /// follow, else the first follower's words (in `why`).
    fn exchange(f: *const Fan, rank: u32, mine: ?Settings, digest: Digest, why: []u8, heard: *Heard) ![]const u8 {
        return if (rank == 0) f.lead(mine, why, heard) else f.follow(rank, mine, digest, why, heard);
    /// Rank 0: `mine`, when given, down every link; then every follower's answer: its weight digest (the follower on link g - 1,
    /// rank g, into heard.weights[g]) and its words. "" when each one can follow, else the first one's words (in `why`).
    /// heard.lead = `mine`.
    fn lead(f: *const Fan, mine: ?Settings, why: []u8, heard: *Heard) ![]const u8 {
        var frame: [@sizeOf(Settings)]u8 = undefined;
        var n: usize = 0;
        if (mine) |s| {
            @memcpy(frame[0..@sizeOf(Settings)], std.mem.asBytes(&s));
            n = @sizeOf(Settings);
        }
        for (f.links) |l| if (l) |x| try x.send(frame[0..n]);
        if (mine == null) return "";
        var first: []const u8 = "";
        var rest: [1024]u8 = undefined;
        for (f.links) |l| if (l) |x| {
            const got = try x.recv(if (first.len == 0) why else &rest);
            if (first.len == 0 and got.len > 0) first = got;
        };
        return first;
    }

    /// A follower: rank 0's Settings (a rank 0 with nothing to check sends none) checked against `mine` and kept (heard.lead); the
    /// answer sent back: this rank's weight digest, then its words. "" when this rank can follow, else its words (in `why`).
    fn follow(f: *const Fan, rank: u32, mine: ?Settings, digest: Digest, why: []u8, heard: *Heard) ![]const u8 {
        const l = f.links[0].?;
        var frame: [@sizeOf(Settings)]u8 = undefined;
        const got = try l.recv(&frame);
        if (got.len == 0) return "";
        if (got.len != @sizeOf(Settings)) return error.BadAnswer;
        const theirs = std.mem.bytesToValue(Settings, got);
        if (theirs.magic != Settings.tag) return error.BadAnswer;
        const text = if (mine) |s| s.differ(theirs, rank, why) else "";
        try l.send(text);
        return text;
    }

    const answer_max = @sizeOf(Digest) + 1024; // a follower's answer: its weight digest, then its words (settingsChecked's why)
};

/// What the ranks of a run must agree on, checked at the start (Fan.commChecked) from Model.open's options.
/// What the start-up exchange gives a rank (Fan.settingsChecked).
    /// rank 0's Settings (rank 0: its own); null when rank 0 sent none
    /// the ranks' weight digests by rank: rank 0 holds every rank's, a follower its own (no_digest: none sent, or
    };

/// head_whole 13, TF_DS_HEAD_CAND's whole head after a round's candidates fell back); 7 = 1-14 (and fill_pair 14, TF_DS_ROCI_STAGGER's two passes); 8 = 1-15 (and snap_sync 15, the SSD tier's stores on every rank); 9 = the same kinds, a Settings of the Rocinante-only layout (no
/// NCCL cores, retired exchange and kernel bits) and a start-up exchange without a library's id. 10 = 1-16 (and cfg 16, a live change of settings of class A to every rank: live_config.CfgFrame). 11 = the same kinds, a fill_pair frame that carries the third pass's rows (TF_DS_ROCI_STAGGER_LANES; 0: a pair). A new set of kinds takes the next
pub const frame_protocol: u32 = 11;
/// What the ranks of a run must agree on, checked at the start (Fan.settingsChecked) from Model.open's options.
pub const Settings = extern struct {
    magic: u32 = tag,
    world: u32,
    streams: u32, // TF_DS_STREAMS (a follower's --streams): a follower holds every stream rank 0 serves
    round_rows: u32, // TF_DS_ROUND_ROWS (--round-rows): the same (it picks the decode kernels)
    rails: u32, // TF_RDMA_DEVICES (--rdma): as many devices
    drafts: u32, // the drafter (--drafts): the same
    graphs: u32, // TF_DS_GRAPHS (--graphs): the same (a rank with graphs votes on each capture, one without never does)
    graph_cap: u32, // TF_DS_GRAPH_CAP (--graph-cap), with graphs: the same (every rank keeps and drops the same graphs)
    pool: u64, // rank 0's window (--context, or TF_DS_WINDOW; a follower's --pool): a follower's holds rank 0's
    replay_floor: u32, // TF_DS_REPLAY_FLOOR: the same
    idx_split: u32 = 0, // TF_DS_IDX_SPLIT (prompt.split_from): the same (the keyed indexer layers all-gather inside)
    neighbor_rdma: u32 = 0, // TF_DS_NEIGHBOR_RDMA: the same exchange path on every rank
    reserved: u32 = 0,
    pool: u64, // rank 0's --context (a follower's --pool): a follower's holds rank 0's

    pub const tag: u32 = 0x31345344; // "DS41"

    /// What follower `rank` (its Settings `mine`) would have to change to follow rank 0's `lead`, in words ("": none).
    pub fn differ(mine: Settings, lead: Settings, rank: u32, buf: []u8) []const u8 {
        var w: std.Io.Writer = .fixed(buf);
        const sep = "; ";
        if (mine.world != lead.world)
            w.print("{s}rank {d} runs WORLD {d} and rank 0 {d}", .{ sepIf(&w, sep), rank, mine.world, lead.world }) catch {};
        if (mine.streams < lead.streams)
            w.print("{s}rank {d} holds {d} streams and rank 0 serves {d}: start rank {d} with --streams {d} (or TF_DS_STREAMS={d})", .{ sepIf(&w, sep), rank, mine.streams, lead.streams, rank, lead.streams, lead.streams }) catch {};
        if (mine.pool < lead.pool)
            w.print("{s}rank {d}'s pool holds {d} positions and rank 0's {d} (its --context, or TF_DS_WINDOW): start rank {d} with --pool {d}", .{ sepIf(&w, sep), rank, mine.pool, lead.pool, rank, lead.pool }) catch {};
        if (mine.round_rows != lead.round_rows)
            w.print("{s}rank {d} runs rounds of {d} rows and rank 0 {d}: start rank {d} with --round-rows {d} (or TF_DS_ROUND_ROWS={d})", .{ sepIf(&w, sep), rank, mine.round_rows, lead.round_rows, rank, lead.round_rows, lead.round_rows }) catch {};
        if (mine.rails != lead.rails)
            w.print("{s}rank {d} has {d} RDMA devices and rank 0 {d} (TF_RDMA_DEVICES): give rank {d} as many with --rdma", .{ sepIf(&w, sep), rank, mine.rails, lead.rails, rank }) catch {};
        if (mine.drafts != lead.drafts)
            w.print("{s}rank {d} {s} the drafter and rank 0 {s}: start rank {d} with --drafts {d}", .{ sepIf(&w, sep), rank, if (mine.drafts != 0) "loads" else "does not load", if (lead.drafts != 0) "does" else "does not", rank, lead.drafts }) catch {};
        if (mine.graphs != lead.graphs)
            w.print("{s}rank {d} runs {s} CUDA graphs and rank 0 {s}: start rank {d} with --graphs {d}", .{ sepIf(&w, sep), rank, if (mine.graphs != 0) "with" else "without", if (lead.graphs != 0) "with them" else "without them", rank, lead.graphs }) catch {}
        else if (mine.graphs != 0 and mine.graph_cap != lead.graph_cap)
            w.print("{s}rank {d} keeps {d} CUDA graphs at most and rank 0 {d} (0: all): start rank {d} with --graph-cap {d} (or TF_DS_GRAPH_CAP={d})", .{ sepIf(&w, sep), rank, mine.graph_cap, lead.graph_cap, rank, lead.graph_cap, lead.graph_cap }) catch {};
        if (mine.replay_floor != lead.replay_floor)
            w.print("{s}rank {d} has TF_DS_REPLAY_FLOOR={d} and rank 0 {d}: set the same on every rank", .{ sepIf(&w, sep), rank, mine.replay_floor, lead.replay_floor }) catch {};
        if (mine.idx_split != lead.idx_split)
            w.print("{s}rank {d} splits the keyed indexer rows from {d} columns and rank 0 from {d} (TF_DS_IDX_SPLIT, 0: unset): set the same on every rank", .{ sepIf(&w, sep), rank, mine.idx_split, lead.idx_split }) catch {};
        if (mine.neighbor_rdma != lead.neighbor_rdma)
            w.print("{s}rank {d} has TF_DS_NEIGHBOR_RDMA={d} and rank 0 {d}: set the same on every rank", .{ sepIf(&w, sep), rank, mine.neighbor_rdma, lead.neighbor_rdma }) catch {};
        return w.buffered();
    }

    fn sepIf(w: *const std.Io.Writer, sep: []const u8) []const u8 {
        return if (w.end > 0) sep else "";
    }
pub const exch_roci_p2_first: u32 = 1 << 8;
pub const exch_roci_engram_ahead: u32 = 1 << 9;
pub const exch_roci_sent_first: u32 = 1 << 11;
pub const exch_roci_combine_door: u32 = 1 << 18;
pub const exch_roci_stagger: u32 = 1 << 21;
pub const exch_ring_word: u32 = 1 << 25;
pub const exch_ring_word_atomic: u32 = 1 << 26; // the switch's signal is the atomic fetch-and-add (unset: the inline word)
pub const exch_ring_wait: u32 = 1 << 27; // TF_DS_RING_WAIT: the rings' queue pairs wait in the card (no proxy threads)
pub const exch_roci_stagger_lanes: u32 = 1 << 28; // TF_DS_ROCI_STAGGER_LANES=3 (unset or 2: clear): three passes in flight, a third lane on every rank
/// A switch that is on unless set otherwise (TF_DS_ROCI_WAIT, TF_DS_RING_WAIT): unset or "1" is on; any other text, "0" included, is off (a typo
/// takes the fallback path, never the new one by accident).
pub fn switchDefaultOn(text: ?[]const u8) bool {
    const v = text orelse return true;
    return std.mem.eql(u8, v, "1");
}

/// TF_DS_ROCI_P2_FIRST as Model.open read it (switchAsked), once (Settings.exch; Two.p2First): an owner sum's
/// second part is sent as soon as its add has made it, before the add of this node's own block.
pub var roci_p2_first_asked: bool = false;

/// TF_DS_ROCI_ENGRAM_AHEAD as Model.open read it (switchAsked), once (Settings.exch; Two.engramAhead): an
/// Engram layer's rows, projection and part-1 send run at the start of the layer before it.
pub var roci_engram_ahead_asked: bool = false;

/// TF_DS_ROCI_SENT_FIRST as Model.open read it (switchAsked), once (Settings.exch; Two.sentFirst): wo_b runs in
/// one launch with the row quarters an owner sum's first part sends first, and that launch rings the part's doorbell itself when those
/// quarters are stored.
pub var roci_sent_first_asked: bool = false;
/// The second set of links' port is the first's plus this: rank 0 listens on it for each follower, as it does on the first.
pub const host_links_port_offset: u16 = 32;

/// TF_DS_ROCI_COMBINE_DOOR as Model.open read it (switchAsked), once (Settings.exch; Two.combineDoor): the MoE's
/// routed sum runs in one launch with the quarters the first part sends first, and that launch rings the part's doorbell itself.
pub var roci_combine_door_asked: bool = false;

/// TF_DS_ROCI_STAGGER as Model.open read it (switchAsked), once (Settings.exch; effective with every Rocinante switch and TF_DS_ENGRAM_PINNED): two
/// consecutive whole passes of a prompt in flight, each on its own lane (a second Rocinante, chunk, stream and cuBLAS handle).
pub var roci_stagger_asked: bool = false;

/// TF_DS_ROCI_STAGGER_LANES as Model.open read it (roci_pair.lanesOf), once (Settings.exch; the ranks must agree: every rank opens the third
/// lane or none): the passes in flight, 2 (unset: a pair as it always was) or 3 (`roci_stagger3_asked`, the Settings bit), and whether the
/// operator set it (a box short of memory refuses a count that was set; one that was only the default keeps two lanes).
pub var roci_stagger_lanes: usize = 2;
pub var roci_stagger3_asked: bool = false;
pub var roci_stagger_lanes_set: bool = false;

    if (roci_p2_first_asked) x |= exch_roci_p2_first;
    if (roci_engram_ahead_asked) x |= exch_roci_engram_ahead;
    if (roci_sent_first_asked) x |= exch_roci_sent_first;
    if (roci_combine_door_asked) x |= exch_roci_combine_door;
    if (roci_stagger_asked) x |= exch_roci_stagger;
    if (roci_stagger3_asked) x |= exch_roci_stagger_lanes;
    if (ring_word_asked) x |= exch_ring_word;
    if (ring_word_atomic_asked) x |= exch_ring_word_atomic;
    if (ring_wait_asked) x |= exch_ring_wait;
/// text "1", switchAsked; TF_DS_QGATE_CANARY's "1", "2" or "3", qgateAsked); an empty name is a reserved bit (kern_reserved). Each is read once by Model.open, before the
};


/// TF_DS_RING_WORD as Model.open read it (switchDefaultOn: on unless set to something but 1), once (Settings.exch, rdma.Settings.roci): each lane's data is
/// a plain RDMA WRITE and its 64-bit generation word is written after it on the same queue pair, the collecting kernel
/// waiting on the words.
pub var ring_word_asked: bool = false;

/// TF_DS_RING_WORD was set (to 1): not the default's silence (a start without a ring says nothing of a switch that is on by default).
pub var ring_word_explicit: bool = false;

/// TF_DS_RING_WORD_SIGNAL=atomic as Model.open read it, once (Settings.exch bit 25, rdma.Settings.roci == .atomic): the
/// generation word is an RDMA ATOMIC fetch-and-add; unset or `word`: an inline 8-byte RDMA WRITE of the sequence.
pub var ring_word_atomic_asked: bool = false;

/// TF_DS_RING_WAIT as Model.open read it, once (Settings.exch bit 27, rdma.Settings.wait): the rings' queue pairs are Core Direct masters that wait
/// in the card on the doorbell word, their chains written when an exchange launches, so no proxy thread runs. Needs TF_DS_RING_WORD with the
/// word signal; every rank alike.
pub var ring_wait_asked: bool = false;

/// TF_DS_RING_WAIT was set (to 1): not the default's silence. A fallback to the proxy threads is for the default only; asked for outright, a ring
/// that cannot wait in the card refuses by name.
pub var ring_wait_explicit: bool = false;

/// TF_DS_QGATE_CANARY's canaries, a quality gate's known perturbations, never a serving setting: the head's rows one
/// more FP8 step (1: the ruler), or layer 0's MoE partials before their gather one more FP8 step (2) or one fp32 ulp
/// away from zero (3: a reduction order's size), the body's floors.
pub const Qgate = enum(u8) { off = 0, head = 1, body_fp8 = 2, body_ulp = 3 };

/// TF_DS_QGATE_CANARY's text: "1", "2" or "3" its canary; unset or any other text none.
pub fn qgateAsked(text: ?[]const u8) Qgate {
    const v = text orelse return .off;
    if (v.len != 1) return .off;
    return switch (v[0]) {
        '1' => .head,
        '2' => .body_fp8,
        '3' => .body_ulp,
        else => .off,
    };
}

/// TF_DS_QGATE_CANARY as Model.open read it (qgateAsked), once (Settings.kern: any canary; every rank's the same
/// canary, Model.open's agreement; prompt.qgate_canary, prompt.qgate_body).
pub var qgate_canary_asked: Qgate = .off;
    if (qgate_canary_asked != .off) x |= kern_qgate_canary;
/// TF_DS_2D_ORDER: the 2D node (role) each rank serves, "g0,g1,g2,g3" (rank 0 must serve node 0: it serves the
/// API); unset, rank g serves node g. On four GB10s cabled as a ring 0-1-2-3-0 without a switch, "0,1,3,2" makes every
/// node's row partner (g ^ 1) and column partner (g ^ 2) its cabled neighbours, and the transport's plans run over
/// the cables in rank order.
pub fn order() ![4]u32 {
    const v = std.c.getenv("TF_DS_2D_ORDER") orelse return .{ 0, 1, 2, 3 };
    return parseOrder(std.mem.span(v));
}

pub fn parseOrder(text: []const u8) ![4]u32 {
    var out: [4]u32 = undefined;
    var seen: [4]bool = @splat(false);
    var it = std.mem.splitScalar(u8, text, ',');
    for (&out) |*o| {
        const x = std.mem.trim(u8, it.next() orelse return error.BadOrder, " ");
        o.* = std.fmt.parseInt(u32, x, 10) catch return error.BadOrder;
        if (o.* > 3 or seen[o.*]) return error.BadOrder;
        seen[o.*] = true;
    }
    if (it.next() != null or out[0] != 0) return error.BadOrder;
    return out;
}

/// The rank of each node g under `ord` (the inverse permutation).
pub fn rankOf(ord: [4]u32) [4]u32 {
    var inv: [4]u32 = undefined;
    for (ord, 0..) |g, i| inv[g] = @intCast(i);
    return inv;
}

test "TF_DS_2D_ORDER: a ring's order and its inverse" {
    const o = try parseOrder("0,1,3,2");
    try std.testing.expectEqual([4]u32{ 0, 1, 3, 2 }, o);
    try std.testing.expectEqual([4]u32{ 0, 1, 3, 2 }, rankOf(o));
    try std.testing.expectEqual([4]u32{ 0, 2, 3, 1 }, rankOf(try parseOrder("0,3,1,2")));
    try std.testing.expectError(error.BadOrder, parseOrder("1,0,2,3"));
    try std.testing.expectError(error.BadOrder, parseOrder("0,1,1,2"));
    try std.testing.expectError(error.BadOrder, parseOrder("0,1,2"));
}

fn len(r: ?[2]usize) usize {
    const x = r orelse return 0;
    return x[1] - x[0];
}

/// A 2D node's part of the prompt path and the decode rounds: the pairs' widths of each split output and the exchange
/// buffers.
/// The widths and sizes of a node's decode exchanges, for decodeOff.
pub const DecodeShape = struct {
    ow: [2]usize, // wo_b's output columns a pair (fp32 partials)
    dw: [2]usize, // down's output columns a pair
    hw: [2]usize, // the head's columns a pair
    ew: [2]usize, // Engram's projection columns a pair (0, 0: none)
    uw: [2]usize, // wo_a's output a pair (bf16)
    gu: [2]usize, // gate / up's columns a pair (bf16)
    ir: usize, // the parity split's rest columns (0: none)
    hidden: usize, // the drafter's pair gather width (fp32)
    rows: usize, // a round's rows at most
    slots: usize, // expert slots a row
    slot: usize, // a ring's slot, in bytes
    cand_bytes: usize, // the head's candidate part of a round of `rows` rows
    pair_ring: bool, // the row pair's ring is open (the drafter's pair gather)
};

/// Why a node's decode exchanges cannot all ride the RDMA rings (null: every one can): each
/// quarters gather in row blocks of at most max_decode_blocks, each part a whole number of 16 bytes and in one slot.
pub fn decodeOff(x: DecodeShape) ?[]const u8 {
    const quarters = struct {
        fn fits(w: [2]usize, rows: usize, slot: usize) bool {
            if ((w[0] * 4) % 16 != 0 or (w[1] * 4) % 16 != 0) return false;
            var bb: [max_decode_blocks][2]usize = undefined;
            _ = decodeBlocks(rows, @max(w[0], w[1]) * 4, slot, &bb) catch return false;
            return true;
        }
    }.fits;
    const cat = struct {
        fn fits(w: [2]usize, rows: usize, slot: usize) bool {
            for (w) |v| {
                if ((v * 2) % 16 != 0 or rows * v * 2 > slot) return false;
            }
            return true;
        }
    }.fits;
    if (!quarters(x.ow, x.rows, x.slot)) return "the attention gather does not ride the ring (a part not whole 16 bytes, or wider than a slot in 8 row blocks)";
    if (!quarters(x.dw, x.rows, x.slot)) return "the MoE gather does not ride the ring";
    if (!quarters(x.hw, x.rows, x.slot)) return "the head's gather does not ride the ring";
    if (x.ew[0] + x.ew[1] != 0 and !quarters(x.ew, x.rows, x.slot)) return "Engram's gather does not ride the ring";
    if (x.cand_bytes > x.slot or x.cand_bytes % 16 != 0) return "the head's candidates do not fit a ring slot";
    if (!cat(x.uw, x.rows, x.slot)) return "wo_a's partner gather does not ride the column ring";
    if (x.ir == 0) {
        if (!cat(x.gu, x.rows * x.slots, x.slot)) return "the experts' partner gather does not ride the column ring";
    } else for (0..2) |p| {
        const width = x.gu[p] + x.ir;
        if ((width * 2) % 16 != 0 or x.rows * x.slots * width * 2 > x.slot) return "the parity experts' pack does not ride the column ring";
    }
    if (x.pair_ring and ((x.hidden * 4) % 16 != 0 or x.rows * x.hidden * 4 > x.slot)) return "the drafter's pair gather does not ride the row ring";
    if (x.rows * 16 > x.slot) return "the drafter's bests do not fit a ring slot";
    return null;
}

pub const Two = struct {
    r: usize,
    p: usize,
    heads: usize, // the heads here (its pair's half of the rank's)
    uw: [2]usize, // wo_a's output a pair (its groups x o_lora): pair 0's, pair 1's
    ow: [2]usize, // wo_b's output columns a pair
    gu: [2]usize, // gate / up's columns a pair
    dw: [2]usize, // down's output columns a pair
    ew: [2]usize, // Engram wkv's output columns a pair (0 without an Engram layer)
    hw: [2]usize, // the head's vocabulary columns a pair (253 / 252 blocks of a TP2 half)
    u_full: u64, // bf16 [cap, uw0 + uw1]: the rank's wo_a output, pair 0's groups then pair 1's
    xd_full: u64, // fp16 [cap * slots, gu0 + gu1]: the rank's expert intermediate, pair 0's blocks then pair 1's
    own: u64, // fp16 [round rows, uw[p]]: a round's half of wo_b's rotated input rows (wo_a's epilogue, round2d.woRot)
    rings: ?*const ring2d.Rings, // the RDMA rings (TF_DS_2D_INTO, TF_DS_2D_GROUPS), when the caller opened them
    // TF_DS_2D_GU=parity: the rest block's width (0: "first"), the rest launches' scratch for a decode window and for a
    // prompt chunk, and the column partner's pack (its main rows, then its rest rows) as the exchange delivers it
    ir: usize = 0,
    rest_rows: usize = 0, // the prompt chunks' rows the rest scratch holds (init's cap)
    rest_dec: ?exl3_experts2d.RestScratch = null,
    rest_chunk: ?exl3_experts2d.RestScratch = null,
    peer: u64 = 0,
    // the rank of each 2D node g (TF_DS_2D_ORDER, model.zig): the transport's slots and the partners' ranks (the identity by default)
    rank_of: [4]u32 = .{ 0, 1, 2, 3 },
    // WP6(b) (docs/prefill-plan.md, indexer row-sharding): agreed at startup (agreeIdxShard), independent of qhc:
    // the indexer's own scoring kernels, not the quarter-HC path's mix and gather.
    idx_shard: bool = false,
    rc: ?*roci_engine.Roci = null,

    /// The widths from the split (both pairs'), and the buffers for chunks of up to `cap` rows (and rounds) from `a`.
    pub fn init(c: *const Config, w: *const weights.Weights, a: *prompt.Arena, s: plan.Split, cap: usize) !Two {
        const pp = s.pair orelse return error.NotA2DSplit;
        const s0 = s.atPair(0);
        const s1 = s.atPair(1);
        const x0 = s0.expertParts(c.*);
        const x1 = s1.expertParts(c.*);
        var t: Two = undefined;
        t.neighbor_gather = null;
        t.rc = null;
        t.r = s.rank;
        t.p = pp;
        t.heads = s.headSpan(c.*)[1];
        t.uw = .{ s0.woAGroups(c.*)[1] * c.o_lora, s1.woAGroups(c.*)[1] * c.o_lora };
        t.ow = .{ len(s0.woBCols(c.*)), len(s1.woBCols(c.*)) };
        t.gu = .{ len(x0.gu), len(x1.gu) };
        t.dw = .{ len(x0.dcols), len(x1.dcols) };
        var engram = false;
        for (w.layers) |lay| engram = engram or lay.engram_wkv != null;
        t.ew = if (engram) .{ len(s0.engramCols(c.*)), len(s1.engramCols(c.*)) } else .{ 0, 0 };
        t.hw = .{ len(s0.headCols(c.*)), len(s1.headCols(c.*)) };
        // the weights loaded here are this split's
        const l0 = w.layers[0];
        if (l0.wq_b.n != t.heads * c.head_dim or l0.groups * c.o_lora != t.uw[pp] or l0.wo_b.n != t.ow[pp] or
            l0.experts.width != t.gu[pp] or l0.experts.down_n != t.dw[pp]) return error.WeightsNotThisSplit;
        // prompt.gather assembles either sublayer's partials: both splits must give the same widths
        if (t.ow[0] != t.dw[0] or t.ow[1] != t.dw[1]) return error.UnequalPartialSplits;
        const slots = c.top_k + 1;
        t.u_full = try a.take(cap * (t.uw[0] + t.uw[1]) * 2);
        // parity: the rest block after both pairs' main blocks in the assembled intermediate
        const ex0 = l0.experts;
        t.ir = if (ex0.rest) |r| r.width else 0;
        t.rest_rows = cap;
        t.rest_dec = null;
        t.rest_chunk = null;
        t.peer = 0;
        t.xd_full = try a.take(@max(cap, round_rows) * slots * (t.gu[0] + t.gu[1] + t.ir) * 2);
        if (t.ir != 0) {
            if (t.gu[0] != t.gu[1]) return error.UnequalParityParts;
            // (Format.native computes the rest in the main launch: no rest scratch)
            if (ex0.native == null) {
                t.rest_dec = try restScratch(a, ex0, exl3_experts.exact_rows, slots, false);
                t.rest_chunk = try restScratch(a, ex0, cap, slots, true);
            }
            t.peer = try a.take(@max(cap, round_rows) * slots * (t.gu[t.p] + t.ir) * 2);
        }
        t.own = try a.take(round_rows * t.uw[pp] * 2);
        t.rings = null;
        t.rank_of = .{ 0, 1, 2, 3 };
        return t;
    }

    /// The second lane's node: every setting the start agreed (whatever the first node's agreements and checks decided), with its own
    /// scratch from its own `init`. The decode rings and the transport are the first node's (the second lane gets its own transport after).
    pub fn adoptSettings(t: *Two, from: *const Two) void {
        const own = .{ t.u_full, t.xd_full, t.peer, t.own, t.rest_dec, t.rest_chunk, t.rest_rows };
        t.* = from.*;
        t.u_full, t.xd_full, t.peer, t.own, t.rest_dec, t.rest_chunk, t.rest_rows = own;
        t.rings = null;
        t.rc = null;
    }

    fn restScratch(a: *prompt.Arena, ex: weights.Experts, rows: usize, slots: usize, chunk: bool) !exl3_experts2d.RestScratch {
        const sz = try exl3_experts2d.RestScratch.sizes(rows, slots, ex, chunk);
        var rs: exl3_experts2d.RestScratch = undefined;
        inline for (.{ "ids", "count", "counts", "members", "cnt_gu", "z", "work_gu", "work_d", "rpick" }, 0..) |f, j| @field(rs, f) = try a.take(sz[j]);
        return rs;
    }

    /// The rest scratch's first use: members -1, the counters and partials zero (as the main scratch starts).
    pub fn clearRest(t: *const Two, e: *const prompt.Engine) !void {
        for ([_]?exl3_experts2d.RestScratch{ t.rest_dec, t.rest_chunk }, [_]bool{ false, true }) |o, chunk| {
            const rs = o orelse continue;
            const ex = e.w.layers[0].experts;
            const sz = try exl3_experts2d.RestScratch.sizes(if (chunk) t.rest_rows else exl3_experts.exact_rows, e.slots(), ex, chunk);
            try prompt.fill32(e, rs.members, 0xffffffff, sz[3] / 4);
            for ([_]u64{ rs.ids, rs.count, rs.counts, rs.cnt_gu, rs.z }, [_]usize{ sz[0], sz[1], sz[2], sz[4], sz[5] }) |ptr, n| try prompt.fill(e, ptr, 0, n);
        }
    }

    /// The column partner's pack (a parity split's main rows, then its rest rows: `bytes` each way) into t.peer, over the
    /// column ring.
    pub fn packExchange(t: *const Two, e: *const prompt.Engine, mine: u64, rows: usize, width: usize) !void {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        var dsts: [2]rdma_mod.Ring.Dst = .{ .{}, .{} };
        dsts[1 - t.p] = .{ .ptr = t.peer, .row4 = @intCast(width * 2 / 16), .stride4 = width * 2 / 16 };
        try rs.col.gatherInto(e.s, mine, @intCast(rows), @intCast(width * 2 / 16), &dsts);
    }

    /// moe_2d's parity experts after the main gate / up and the rest's: the packs exchanged with the column partner,
    /// the half's intermediate assembled into t.xd_full (pair 0's main, pair 1's main, each slot's rest from its
    /// owner), for down.
    pub fn assembleParity(t: *const Two, e: *const prompt.Engine, ex: weights.Experts, xd: u64, pick: u64, p: usize) !void {
        const rest = ex.rest orelse return error.NoRest;
        const ig = t.gu[t.p];
        try t.packExchange(e, xd, p, ig + t.ir);
        const part0 = if (t.p == 0) xd else t.peer;
        const part1 = if (t.p == 1) xd else t.peer;
        try e.ops.assembleRest(e.s, part0, part1, pick, rest.owner, t.xd_full, p, ig, t.ir);
    }

    /// Every node's part over NCCL into `parts` [4, n, wmax] (bytes), this node's `src` [n, w[p]] padded to wmax.
    fn allParts(t: *const Two, e: *const prompt.Engine, src: u64, n: usize, w: [2]usize, esize: usize) !usize {
        const wmax = @max(w[0], w[1]);
        if (n * wmax * esize > t.part_bytes) return error.PartTooBig;
        var send = src;
        if (w[t.p] != wmax) {
            try e.ops.copyRows(e.s, src, w[t.p] * esize, t.pad, wmax * esize, w[t.p] * esize, n);
            send = t.pad;
        }
        const bytes = n * wmax * esize;
        if (t.neighbor_gather) |ng| {
            if (neighbor.eligible(bytes)) {
                try ng.gather(e.s, send, t.parts, bytes);
                return wmax;
            }
        }
        try e.comm.allGather(send, t.parts, bytes, .u8, e.s);
        return wmax;
    }

    /// Whether a ring took the exchange: false when there is none or the part does not fit it (nothing sent then).
    fn ringed(r: anyerror!void) !bool {
        r catch |err| switch (err) {
            error.TooLarge, error.NotFloat4 => return false,
            else => return err,
        };
        return true;
    }

    pub fn quarters(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        if (!try ringed(rs.quarters(e.s, src, dst, n, w, esize))) return error.RingRefused;
    }

        const rs = t.rings orelse return error.DecodeNeedsRings;
        const blocks = try decodeBlocks(n, @max(w[0], w[1]) * esize, rs.slotBytes(), &bb);
            if (!sent) return if (i == 0) error.RingRefused else error.RingRefusedBlock;
        }
    }

    /// WP6(b) (docs/prefill-plan.md, indexer row-sharding): agreed at startup like qhc, but independent of
    /// it -- it only needs a 2D engine with every rank's cap evenly
    /// divisible into 16-row-aligned quarters (idxShardRange's own check; refused per-rank there, not here, so
    /// a mismatch across ranks still shows as every rank agreeing on the flag but only some actually sharding).
    pub fn agreeIdxShard(t: *Two, d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream) !void {
        const want = if (std.c.getenv("TF_DS_IDX_SHARD")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        const w = comm.world;
        var dev = try cuda.DeviceBuffer.alloc(d, (1 + w) * 4);
        defer dev.free();
        const mine: u32 = @intFromBool(want);
        try comm.gatherHost(d, dev.ptr, dev.ptr + 4, 4, stream);
        try stream.synchronize();
        var all: [8]u32 = undefined;
        if (w > all.len) return error.WorldTooLarge;
        try dev.download(4, std.mem.sliceAsBytes(all[0..w]));
        var on: usize = 0;
        for (all[0..w]) |x| on += x;
        if (on != 0 and on != w) {
            std.log.err("TF_DS_IDX_SHARD: set on {d} of {d} ranks (this one: {s}); start every rank with the same value", .{ on, w, if (want) "1" else "unset" });
            return error.IdxShardMismatch;
        }
        t.idx_shard = on == w;
        if (t.idx_shard) std.log.info("TF_DS_IDX_SHARD=1: the indexer's scoring and top-k, for layers before the candidate source on a full chunk, run only each rank's owned quarter, all-gathering the picks", .{});
    }

    /// Comm2D.gather (TP2 inside a pair: the DSpark drafter's sublayer partials): `src` [n, width] -> `dst` [2, n, width],
    /// this pair's two ranks in TP2 rank order: over the row pair's ring, else the four-rank all-gather's pair half
    /// (Comm2D.gather4(x)[2p:2p + 2]).
    pub fn pairGather(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, width: usize, esize: usize) !void {
        if (t.rings) |rs| if (try ringed(rs.pairGather(e.s, src, dst, n, width, esize))) return;
        if (t.neighbor_gather) |ng| {
            const bytes = try std.math.mul(usize, n, try std.math.mul(usize, width, esize));
            if (neighbor.eligible(bytes)) {
                try ng.pair(e.s, src, dst, bytes);
                return;
    /// TF_DS_ROCI in effect: `asked` (the start-up Settings made it the same on every rank) and rociOff's conditions on every
    /// rank (`why`: this rank's reason it cannot, null when it can), one world all-gather. A rank that cannot ends the start
    /// on every rank, naming which: Rocinante has no other transport to fall back to.
    /// TF_DS_ROCI_P2_FIRST in effect: asked (the start-up Settings made it the same on every rank).
    pub fn p2First(_: *const Two) bool {
        return roci_p2_first_asked;
    }

    /// TF_DS_ROCI_ENGRAM_AHEAD in effect: asked (the start-up Settings made it the same on every rank).
    pub fn engramAhead(_: *const Two) bool {
        return roci_engram_ahead_asked;
    }

    /// TF_DS_ROCI_SENT_FIRST in effect: asked (the start-up Settings made it the same on every rank).
    pub fn sentFirst(_: *const Two) bool {
        return roci_sent_first_asked;
    }

    /// TF_DS_ROCI_COMBINE_DOOR in effect: asked (the start-up Settings made it the same on every rank).
    pub fn combineDoor(_: *const Two) bool {
        return roci_combine_door_asked;
    }

    /// This node's decode exchanges (decodeOff's input) for rounds of `rows` rows, rings of `slot` bytes, and a head
    /// candidate part of `cand_bytes`.
    pub fn decodeShape(t: *const Two, hidden: usize, rows: usize, slots: usize, slot: usize, cand_bytes: usize) DecodeShape {
        return .{ .ow = t.ow, .dw = t.dw, .hw = t.hw, .ew = t.ew, .uw = t.uw, .gu = t.gu, .ir = t.ir, .hidden = hidden, .rows = rows, .slots = slots, .slot = slot, .cand_bytes = cand_bytes, .pair_ring = t.rings != null and t.rings.?.pair != null };
    }

    /// One start check of a family of Rocinante's exchanges, on every rank together: `why` is null where this rank can run it. A rank
    /// that cannot ends the start, naming itself and its reason (there is no other transport to fall back to). One gather over the
    /// host links.
    pub fn require(d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream, name: []const u8, why: ?[]const u8) !void {
        const cannot = try ranksThatCannot(d, comm, stream, why == null);
        if (cannot == 0) return;
        var buf: [64]u8 = undefined;
        std.log.err("{s}: {s} cannot run it (this rank: {s}); not started", .{ name, rankList(cannot, &buf), why orelse "can" });
        return error.RociCannot;
    }

    /// Every rank's node order (TF_DS_2D_ORDER, `rank_of`) and gather widths (`ow`) gathered over the host links and checked against
    /// this rank's own: a rank alone on a different permutation would address different partners under the same role numbers (the
    /// transport's plans, the rings and the node's weights all follow it), so the start ends by name. Before any prompt.
    pub fn agreeTopology(t: *const Two, d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream) !void {
        const w = comm.world;
        const Cfg = extern struct { rank_of: [4]u32, ow: [2]u32 };
        const mine: Cfg = .{ .rank_of = t.rank_of, .ow = .{ @intCast(t.ow[0]), @intCast(t.ow[1]) } };
        var dev = try cuda.DeviceBuffer.alloc(d, (1 + w) * @sizeOf(Cfg));
        defer dev.free();
        try comm.gatherHost(d, dev.ptr, dev.ptr + @sizeOf(Cfg), @sizeOf(Cfg), stream);
        try stream.synchronize();
        var all: [8]Cfg = undefined;
        if (w > all.len) return error.WorldTooLarge;
        try dev.download(@sizeOf(Cfg), std.mem.sliceAsBytes(all[0..w]));
        for (all[0..w], 0..) |x, i| {
            if (!std.mem.eql(u32, &x.rank_of, &mine.rank_of)) {
                std.log.err("rank {d}'s node order {any} differs from this rank's {any} (TF_DS_2D_ORDER mismatch): refusing to start", .{ i, x.rank_of, mine.rank_of });
                return error.TopologyMismatch;
            }
            if (x.ow[0] != mine.ow[0] or x.ow[1] != mine.ow[1]) {
                std.log.err("rank {d}'s gather widths {any} differ from this rank's {any}: refusing to start", .{ i, x.ow, mine.ow });
                return error.TopologyMismatch;
            }
        }
    }

    /// The drafter's per-step gather (Markov's bests): `rows` rows of four fp32 from every node into `dst` [4, rows, 4] in the
    /// communicator's rank order, over the four nodes' ring. The bytes comm.allGather of the same parts gives.
    pub fn nodesGather(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, rows: usize) !void {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        try rs.nodesGather(e.s, src, dst, rows, t.rank_of);
    pub fn node(t: *const Two) u32 {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        if (!try ringed(rs.catRank(e.s, src, dst, n, w, esize))) return error.RingRefused;
    }

    pub fn pairGather(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, width: usize, esize: usize) !void {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        if (!try ringed(rs.pairGather(e.s, src, dst, n, width, esize))) return error.RingRefused;
    }

    /// wo_2d after wo_a: the rank's four groups (this node's two in ch.u [n, uw[p]], the partner's two), then wo_b
    /// for this pair's output columns: ch.pa [n, ow[p]] fp32, the matching columns of TP2's partial.
    pub fn woB(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        _ = try t.woBOf(e, ch, lay, false);
    }

    /// woB with TF_DS_ROCI_SENT_FIRST's launch where it applies: the GEMM runs in one launch with the row quarters an owner
    /// sum's first part sends first and rings that part's doorbell itself (prompt.mmSentFirst), and the part is marked sent.
    /// Returns whether it did; otherwise woB's launches ran as woB's.
    pub fn woBSentFirst(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !bool {
        return t.woBOf(e, ch, lay, true);
    }

    fn woBOf(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer, sent_first: bool) !bool {
        return t.woBCodes(e, ch, lay, sent_first);
    ///
    /// TF_DS_ROCI_CODES, a whole pass: the part is quantized into the arena's send region, goes to the column partner as RDMA WRITEs
    /// into its land region, and the copies read the partner's part there (the exchange is the transport's).
    ///
    /// With TF_DS_ROCI_SENT_FIRST (`sent_first`, the caller's owner-sum conditions held) beside TF_DS_ROCI_CODES the GEMM on the landed
    /// codes is the one-launch kernel that rings the owner sum's first part itself (prompt.codesGemmSentFirst): true when it ran.
    pub fn woBCodes(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer, sent_first: bool) !bool {
        const shape = prompt.codesShape(e, ch) orelse return error.RociNoShape;
        const rc = t.rc.?;
        const send = rc.addr(.wob_send);
        const land = rc.addr(.wob_land);
        const pl = catCodesPlan(t.node(), t.rank_of, ch.n, t.uw, ch.u, send, land, ch.wob_q, ch.wob_s);
        try rc.send(e.s, .wob, shape);
        try rc.wait(e.s, .wob);
        if (sent_first) {
            try catCodesCopies(e.ops, e.s, pl);
            if (try prompt.codesGemmSentFirst(e, ch, t, lay.wo_b, pl.q, pl.s)) return true;
            try mx.gemm(k, e.s, lay.wo_b, pl.q, pl.k, pl.s, pl.k / 32, ch.n, ch.pa, lay.wo_b.n, .f32, false, ch.mxs.z);
            return false;
        }
        return false;
    }

    /// moe_2d's experts after the routing: this node's gate / up blocks, the rank's whole intermediate assembled, down
    /// for this node's output columns: ch.pm [n, dw[p]] fp32. A chunk of fewer than EXACT_ROWS rows takes the decode
    /// window's fused path and scratch (experts2d.fused_ok), as the rounds do.
    pub fn experts(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
    pub fn experts(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer, x_codes: bool, cf: ?*const mx.CombineFirst) !void {
        const c = e.c;
        const n = ch.n;
            // the pack as codes over the transport (the decode rounds' experts are expertsNative's, through the rings)
            return t.expertsCodes(e, ch, lay.experts, ch.x, c.hidden, ch.pick, ch.wts, ch.xs.xd, ch.pm, n, x_codes, cf);
        if (x_codes or cf != null) return error.NotNativeExperts;
        if (n < exl3_experts.exact_rows) return round2d.experts(t, e, lay.experts, ch.xsd, ch.x, c.hidden, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
        try exl3_experts2d.gateUp(e.ex, e.s, lay.experts, ch.xs, ch.x, c.hidden, ch.pick, n, c.swiglu_limit);
        if (t.ir != 0) {
            // parity: the rest columns of the experts this pair computes them for, the packs exchanged and assembled
            try exl3_experts2d.gateUpRest(e.ex, e.ops, e.s, lay.experts, ch.xs, t.rest_chunk.?, ch.pick, n, c.swiglu_limit);
            try t.assembleParity(e, lay.experts, ch.xs.xd, ch.pick, n * ch.xs.slots);
            return exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
        }
        try t.catRank(e, ch.xs.xd, t.xd_full, n * ch.xs.slots, t.gu, 2);
        try exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
    }

    /// Format.native's experts on a 2D node, chunk or round alike: this node's pack of every expert's gate / up (its main
    /// blocks, its rest block for the experts whose rest is this pair's) into xd [rows * slots, pack] bf16, the rank's
    /// whole intermediate assembled from the two pairs' packs (the same bytes moved as the EXL3 path's fp16 rows), down
    pub fn expertsNative(t: *const Two, e: *const prompt.Engine, ch: *const prompt.Chunk, ex: weights.Experts, x: u64, ldx: usize, pick: u64, wts: u64, xd: u64, out: u64, rows: usize, split_k: bool, x_codes: bool, cf: ?*const mx.CombineFirst) !void {
        const k = e.mx.?;
        if (!split_k) if (prompt.ptimer) |pt| pt.mark(e.s, .m_gu) catch {};
        }
        if (t.ir != 0) try t.assembleParity(e, ex, xd, pick, rows * ch.mxe.slots) else try t.catRank(e, xd, t.xd_full, rows * ch.mxe.slots, t.gu, 2);
        if (!split_k) if (prompt.ptimer) |pt| pt.mark(e.s, .m_x) catch {};
        if (cf) |f| {
            if (split_k) return error.CombineFirstOnDecode;
            return mx.downFirst(k, e.s, ex, ch.mxe, t.xd_full, ex.down_k, pick, rows, out, t.dw[t.p], false, f);
        }
        try mx.down(k, e.s, ex, ch.mxe, t.xd_full, ex.down_k, pick, rows, out, t.dw[t.p], false, split_k);
    }

    pub fn expertsCodes(t: *const Two, e: *const prompt.Engine, ch: *const prompt.Chunk, ex: weights.Experts, x: u64, ldx: usize, pick: u64, wts: u64, xd: u64, out: u64, rows: usize, x_codes: bool, cf: ?*const mx.CombineFirst) !void {
        const mine = t.rc.?.addr(.pack_send);
        const theirs = t.rc.?.addr(.pack_land);
        try t.rc.?.send(e.s, .pack, shape);
        try t.rc.?.wait(e.s, .pack);
        const part0 = if (t.p == 0) mine else theirs;
        const part1 = if (t.p == 1) mine else theirs;
        if (t.gu[0] == t.gu[1] and mx.threeTakes(k, ex, ch.mxe, t.gu[t.p], t.ir)) {
        if (cf) |f| return mx.downCodesFirst(k, e.s, ex, ch.mxe, pick, rows, out, t.dw[t.p], false, f);
    }

    /// Engram's projection summed over the ranks from its column quarters (model.py esum): the gathered width.
    pub fn engramWidth(t: *const Two) usize {
        return t.ew[0] + t.ew[1];
    }
};

/// The ranks of `comm` whose `can` is false, a bit each (bit k: rank k), from one gather of every rank's word over the
/// host links on `stream`: a start-up agreement, called by every rank together.
    try comm.gatherHost(d, dev.ptr, dev.ptr + 4, 4, stream);
/// The ranks whose `word` differs from rank 0's, a bit each (every rank's word gathered over the host links: every rank
/// the same mask); 0 when every rank gave rank 0's.
pub fn ranksDiffer(d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream, word: u32) !u32 {
    const w = comm.world;
    var all: [8]u32 = undefined;
    if (w > all.len) return error.WorldTooLarge;
    var dev = try cuda.DeviceBuffer.alloc(d, (1 + w) * 4);
    defer dev.free();
    try dev.uploadAsync(0, std.mem.asBytes(&word), stream.handle);
    try comm.gatherHost(d, dev.ptr, dev.ptr + 4, 4, stream);
    try stream.synchronize();
    try dev.download(4, std.mem.sliceAsBytes(all[0..w]));
    return differMask(all[0..w]);
}

/// The ranks whose word is not rank 0's (words[0]), a bit each.
pub fn differMask(words: []const u32) u32 {
    var m: u32 = 0;
    for (words, 0..) |x, k| {
        if (x != words[0]) m |= @as(u32, 1) << @intCast(k);
    }
    return m;
}

/// Why the owner sums over Rocinante cannot run on this rank (null: it can): attention, MoE and Engram all at owned rows,
pub fn rociOff(ow: [2]usize, pass_rows: usize, readers: bool) ?[]const u8 {
    if (ownerYxOff(ow)) |why| return why;
    if (!readers) return "the kit's extra set has no _hc_mix_pf_2b / _hc_post_2b / _engram_gate_2b variant";
    return null;
}

/// Why TF_DS_ROCI_DEC cannot run on this rank before its kernel check (null: it can): the decoder front on owned rows runs .once's kernels
/// (act_quant, wq_a and wkv on one quant), the baseline front over the decoder rows only with TF_DS_QUANT_ONCE (`quant_once`); the taps region
/// holds three taps of 5,120 values a row (`taps` the DSpark taps, none without a drafter; `hidden` the model's).
pub fn rociDecOff(quant_once: bool, taps: usize, hidden: usize) ?[]const u8 {
    if (!quant_once) return "needs TF_DS_QUANT_ONCE=1 (the decoder half's front on owned rows runs its kernels)";
    if (taps != 0 and taps * hidden * 2 != rocinante.taps_row) return "the arena's taps region holds three taps of 5,120 values a row";
    if (hidden * 2 != rocinante.head_row) return "the arena's head row holds 5,120 values";
    return null;
}

/// Why TF_DS_ROCI_CODES cannot run on this rank (null: it can): the codes exchanges of TF_DS_CAT_CODES and TF_DS_PACK_CODES are the
/// ones it carries, and the arena's two parts hold wo_b's part of 2,304 bytes a row (2,048 codes and their 64 scales) for both pairs
/// and the pack's 720 bytes a slot row (640 codes, 80 scales) for 7 slots a row.
pub fn rociCodesOff(uw: [2]usize, gu: [2]usize, ir: usize, slots: usize) ?[]const u8 {
    if (uw[0] != uw[1] or catCodesBytes(1, uw[0]) != 2304) return "the arena holds wo_b's part as 2,304 bytes a row (equal pairs of 2,048 columns)";
    if (slots != 7) return "the arena holds the pack for 7 slots a row";
    for (gu) |g| {
        const pc = packCodesPlan(1, g, ir) catch return "the pack is not whole 32-value groups";
        if (pc.bytes != 720) return "the arena holds the pack as 720 bytes a slot row";
    }
    return null;
}

/// Why TF_DS_ROCI_GATHERS cannot run on this rank (null: it can): the gathers it carries are the owned front's (qr, y), the owned
/// routing's, the indexer's top-k and the boundary's, so those switches must be in effect on the checkpoint's own weights; the
/// arena's regions hold rows of x 10,240 bytes, qr 2,560, y 1,024, top-k 4,096 (512 keys), the routing's codes 5,120, scales
/// 640, picks and weights 28, and the boundary's four cache arrays 356 bytes a row.
pub fn rociGathersOff(native: bool, hidden: usize, qw: usize, hd: usize, index_topk: usize, id: usize, dims: usize, slots: usize) ?[]const u8 {
    if (!native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    if (hidden * 2 != 10240 or qw * 2 != 2560 or hd * 2 != 1024) return "the arena holds x, qr and y rows of 10,240, 2,560 and 1,024 bytes";
    if (index_topk != 512) return "the arena holds the top-k as 512 keys a row";
    if (dims != 5120 or slots != 7) return "the arena holds the routing's codes as 5,120 bytes and 7 picks a row";
    if (hd / 2 + hd / 16 + id / 2 + id / 32 != 356) return "the arena holds the boundary's cache arrays as 356 bytes a row";
    return null;
}

pub fn catCodesOff(native: bool, layers: []const weights.Layer, uw: [2]usize) ?[]const u8 {
/// Both pairs' codes and scales into wo_b's input (pl.copies).
pub fn catCodesCopies(o: *const ops_mod.Ops, s: cuda.Stream, pl: CatCodes) !void {
    for (pl.copies) |cp| try o.copyRows(s, cp.src, cp.src_stride, cp.dst, cp.dst_stride, cp.width, cp.rows);
}

    try catCodesCopies(o, s, pl);
/// woBCodes of node g (2 p + r) under ranks `rank_of`, n rows: this node's u [n, uw[p]] bf16 quantized into
pub fn catCodesPlan(g: u32, rank_of: [4]u32, n: usize, uw: [2]usize, u: u64, codes: u64, stage: u64, wob_q: u64, wob_s: u64) CatCodes {
        .peer = rank_of[g ^ 2],
/// Why the owner sums cannot run (null: they can): widening the two pairs' sums into one row needs equal pair widths.
pub fn ownerYxOff(ow: [2]usize) ?[]const u8 {
/// Owner sums of node g under ranks `rank_of` (Two.rank_of), for owned blocks of q rows: rank k owns rows R(k) =
    row_peer: u32, // the row partner's rank (node g ^ 1: the other TP2 rank of this pair)
    col_peer: u32, // the column partner's rank (node g ^ 2: this TP2 rank of the other pair)
pub fn ownerPlan(g: u32, rank_of: [4]u32, q: usize) OwnerPlan {
        .row_peer = rank_of[g ^ 1],
        .col_peer = rank_of[g ^ 2],
        .send1 = .{ at(rank_of[g ^ 1], q), at(rank_of[g ^ 3], q) },
        .recv1 = .{ at(rank_of[g], q), at(rank_of[g ^ 2], q) },
        .send2 = at(rank_of[g ^ 2], q),
        .recv2 = at(rank_of[g], q),
const Span = @import("land.zig").Span;
test "a node's split: TP over two, the 2D split over four" {
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2 }, try split(1, 2));
    try std.testing.expectEqual(plan.Split{ .rank = 0, .world = 2, .pair = 1 }, try split(2, 4));
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2, .pair = 1 }, try split(3, 4));
    try std.testing.expectError(error.UnsupportedWorld, split(0, 3));
}

test "a follower holds rank 0's streams and pool and runs its round rows and graphs, or says what to change" {
    const served: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 0, .idx_rb = 1 };
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("", served.differ(served, 1, &buf));
    var more = served; // a follower may hold more streams and positions than rank 0 uses
    more.streams = 16;
    more.pool = 1 << 25;
    try std.testing.expectEqualStrings("", more.differ(served, 1, &buf));
    var four = served;
    four.streams = 4;
    try std.testing.expectEqualStrings("rank 1 holds 4 streams and rank 0 serves 16: start rank 1 with --streams 16 (or TF_DS_STREAMS=16)", four.differ(served, 1, &buf));
    var other = served;
    other.round_rows = 16;
    other.pool = 262144;
    other.rails = 0;
    other.drafts = 0;
    other.graph_cap = 64;
    const words = other.differ(served, 3, &buf);
    for ([_][]const u8{ "--pool 16777216", "--round-rows 48", "--rdma", "--drafts 1", "--graph-cap 128" }) |w|
        try std.testing.expect(std.mem.indexOf(u8, words, w) != null);
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, words, "; "));
    // graphs on one rank only: it would vote on captures the others never make
    var eager = served;
    eager.graphs = 0;
    eager.graph_cap = 0;
    try std.testing.expectEqualStrings("rank 2 runs without CUDA graphs and rank 0 with them: start rank 2 with --graphs 1", eager.differ(served, 2, &buf));
    // without graphs anywhere, the caps do not matter
    var off = served;
    off.graphs = 0;
    var off2 = off;
    off2.graph_cap = 7;
    try std.testing.expectEqualStrings("", off2.differ(off, 1, &buf));
}

test "a follower runs rank 0's replay floor, or says to set the same on every rank" {
    const on: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1 };
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("", on.differ(on, 2, &buf));
    var off = on;
    off.replay_floor = 0;
    try std.testing.expectEqualStrings("", off.differ(off, 2, &buf));
    try std.testing.expectEqualStrings("rank 2 has TF_DS_REPLAY_FLOOR=0 and rank 0 1: set the same on every rank", off.differ(on, 2, &buf));
    try std.testing.expectEqualStrings("rank 1 has TF_DS_REPLAY_FLOOR=1 and rank 0 0: set the same on every rank", on.differ(off, 1, &buf));
    // with the other words, one separator each
    var both = off;
    both.streams = 4;
    try std.testing.expectEqualStrings("rank 3 holds 4 streams and rank 0 serves 16: start rank 3 with --streams 16 (or TF_DS_STREAMS=16); rank 3 has TF_DS_REPLAY_FLOOR=0 and rank 0 1: set the same on every rank", both.differ(on, 3, &buf));
}

test "rank 0's settings go with NCCL's id: the follower answers, and an older rank 0's id alone is taken as before" {
    const io = std.testing.io;
    const served: Settings = .{ .world = 2, .streams = 8, .round_rows = 16, .rails = 0, .drafts = 1, .replay_floor = 1, .pool = 1 << 19 };
    var four = served;
    four.streams = 4;
    var neighbor_on = served;
    neighbor_on.neighbor_rdma = 1;
    const Case = struct { lead: ?Settings, follower: ?Settings, differs: bool };
    const cases = [_]Case{
        .{ .lead = served, .follower = served, .differs = false },
        .{ .lead = served, .follower = four, .differs = true },
        .{ .lead = served, .follower = neighbor_on, .differs = true },
        .{ .lead = neighbor_on, .follower = served, .differs = true },
        .{ .lead = neighbor_on, .follower = neighbor_on, .differs = false },
        .{ .lead = null, .follower = four, .differs = false }, // an older rank 0: the id alone, nothing to check
    };
    const base: u16 = 39500 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    for (cases, 0..) |cs, k| {
        const Peer = struct {
            id: [128]u8 = undefined,
            why: [1024]u8 = undefined,
            said: []const u8 = "",
            err: ?anyerror = null,
            fn run(p: *@This(), port: u16, mine: ?Settings) void {
                var f: Fan = .{};
                f.links[0] = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, port, 10) catch |e| {
                    p.err = e;
                    return;
                };
                defer f.close();
                p.said = f.follow(1, &p.id, mine, &p.why) catch |e| {
                    p.err = e;
                    return;
                };
            }
        };
        const port = base + @as(u16, @intCast(k));
        var peer: Peer = .{};
        const t = try std.Thread.spawn(.{}, Peer.run, .{ &peer, port, cs.follower });
        var f: Fan = .{};
        f.links[0] = Link.listen(.{ 127, 0, 0, 1 }, port) catch |e| {
            t.join();
            return e;
        };
        var id: [128]u8 = undefined;
        for (&id, 0..) |*b, i| b.* = @intCast(i);
        const pl = catCodesPlan(@intCast(g), rankOf(orders[0]), 2048, .{ 2048, 2048 }, u, codes, stage, wq, ws);
        const nc = rankOf(ord);
        // one exchange a pair of column partners: each names the other's rank, and what one sends the other takes
/// The node that rank `k` serves under `rank_of`.
fn nodeOfRank(rank_of: [4]u32, k: u32) usize {
    return std.mem.indexOfScalar(u32, &rank_of, k).?;
        const nc = rankOf(ord);
            // the partners: node g ^ 1 and node g ^ 2, by their ranks, and mutual
test "TF_DS_PASS_ROWS takes any whole number of resume units up to the engine's largest pass (unset: one); TF_DS_PASS_FROM any multiple of the unit, 0 included (unset: four)" {
    const top = prompt.max_pass_rows;
    try std.testing.expectEqual(@as(u32, 16384), top);
    try std.testing.expectEqual(@as(?u32, 2048), passRowsSetting(null, 2048, top));
    var units: u32 = 1;
    while (units <= 8) : (units += 1) {
        var buf: [8]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "{d}", .{units * 2048}) catch unreachable;
        try std.testing.expectEqual(@as(?u32, units * 2048), passRowsSetting(text, 2048, top));
    }
    // not a whole number of units, none at all, or above the largest pass: no setting, and the words name the setting, the unit and the largest
    for ([_][]const u8{ "", "0", "1024", "3000", "6143", "4096 ", "x", "-2048", "18432", "32768", "16385", "99999999999" }) |t| {
        try std.testing.expectEqual(@as(?u32, null), passRowsSetting(t, 2048, top));
        var wb: [320]u8 = undefined;
        const w = passRowsRefusal(&wb, t, 2048, top);
        try std.testing.expect(std.mem.indexOf(u8, w, "TF_DS_PASS_ROWS=") != null);
        try std.testing.expect(std.mem.indexOf(u8, w, "whole number of 2048-row resume units") != null);
        try std.testing.expect(std.mem.indexOf(u8, w, "16384") != null);
    // a smaller ceiling (a build for fewer rows) holds the same rule
    try std.testing.expectEqual(@as(?u32, 8192), passRowsSetting("8192", 2048, 8192));
    try std.testing.expectEqual(@as(?u32, null), passRowsSetting("10240", 2048, 8192));
    try std.testing.expectEqual(@as(?u32, 8192), passFromSetting(null, 2048));
    try std.testing.expectEqual(@as(?u32, 0), passFromSetting("0", 2048));
    try std.testing.expectEqual(@as(?u32, 2048), passFromSetting("2048", 2048));
    try std.testing.expectEqual(@as(?u32, 4096), passFromSetting("4096", 2048));
    try std.testing.expectEqual(@as(?u32, 6144), passFromSetting("6144", 2048));
    try std.testing.expectEqual(@as(?u32, 8192), passFromSetting("8192", 2048));
    try std.testing.expectEqual(@as(?u32, 16384), passFromSetting("16384", 2048));
    for ([_][]const u8{ "", "1", "3000", "9000", "-8192", "4096 ", "x" }) |t| try std.testing.expectEqual(@as(?u32, null), passFromSetting(t, 2048));
test "TF_DS_PASS_BESIDE: unset is the resume rows; any whole number of units up to the booted pass rows is a setting; anything else none" {
    try std.testing.expectEqual(@as(?u32, 2048), passBesideSetting(null, 2048, 2048));
    try std.testing.expectEqual(@as(?u32, 2048), passBesideSetting(null, 2048, 4096));
    try std.testing.expectEqual(@as(?u32, 2048), passBesideSetting("2048", 2048, 2048));
    try std.testing.expectEqual(@as(?u32, 2048), passBesideSetting("2048", 2048, 4096));
    try std.testing.expectEqual(@as(?u32, 4096), passBesideSetting("4096", 2048, 4096));
    try std.testing.expectEqual(@as(?u32, 4096), passBesideSetting("4096", 2048, 8192));
    try std.testing.expectEqual(@as(?u32, 6144), passBesideSetting("6144", 2048, 8192));
    try std.testing.expectEqual(@as(?u32, 8192), passBesideSetting("8192", 2048, 8192));
    // above the booted pass rows: refused
    try std.testing.expectEqual(@as(?u32, null), passBesideSetting("4096", 2048, 2048));
    try std.testing.expectEqual(@as(?u32, null), passBesideSetting("8192", 2048, 4096));
    for ([_][]const u8{ "", "0", "1024", "3000", "16384", "4096 ", "-4096", "x" }) |t| try std.testing.expectEqual(@as(?u32, null), passBesideSetting(t, 2048, 8192));
}
test "TF_DS_PASS_FILLING: the environment sets the switch and prints its start line when on" {
    const env = struct { // the process environment, as Model.open reads it
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    const name = "TF_DS_PASS_FILLING";
    if (std.c.getenv(name) != null) return error.SkipZigTest; // the caller's own setting: left alone
    defer pass_filling_asked = false;
    _ = env.unsetenv(name);
    readPassFillingEnv();
    try std.testing.expect(!pass_filling_asked);
    _ = env.setenv(name, "1", 1);
    readPassFillingEnv();
    try std.testing.expect(pass_filling_asked);
    try std.testing.expect(std.mem.startsWith(u8, pass_filling_line, "TF_DS_PASS_FILLING=1: "));
    _ = env.setenv(name, "0", 1); // any text but "1": off, as every switch asked this way
    readPassFillingEnv();
    try std.testing.expect(!pass_filling_asked);
    _ = env.unsetenv(name);
    readPassFillingEnv();
    try std.testing.expect(!pass_filling_asked);
}

test "TF_DS_PROMPT_ORDER, TF_DS_PROMPT_QOS and TF_DS_PROMPT_MAX_WAIT_S: the environment sets the prompt-step order" {
    const env = struct { // the process environment, as Model.open reads it
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    const order_name = "TF_DS_PROMPT_ORDER";
    const qos_name = "TF_DS_PROMPT_QOS";
    const wait_name = "TF_DS_PROMPT_MAX_WAIT_S";
    if (std.c.getenv(order_name) != null or std.c.getenv(qos_name) != null or std.c.getenv(wait_name) != null) return error.SkipZigTest; // the caller's own settings: left alone
    defer {
        _ = env.unsetenv(order_name);
        _ = env.unsetenv(qos_name);
        _ = env.unsetenv(wait_name);
        prompt_order = .{};
    }
    _ = env.unsetenv(order_name);
    _ = env.unsetenv(qos_name);
    try readPromptOrderEnv();
    try std.testing.expect(!prompt_order.on);
    _ = env.setenv(order_name, "turns", 1);
    try readPromptOrderEnv();
    try std.testing.expect(prompt_order.on and prompt_order.order == .turns and prompt_order.count_all);
    _ = env.setenv(qos_name, "4k:1:0,*:2:inf", 1);
    try readPromptOrderEnv();
    try std.testing.expect(prompt_order.on and !prompt_order.count_all and prompt_order.len == 2);
    _ = env.setenv(wait_name, "12", 1);
    try readPromptOrderEnv();
    try std.testing.expectEqual(@as(u32, 12), prompt_order.max_wait_s);
    _ = env.setenv(wait_name, "0", 1); // refused by name: the start stops, the policy before it stands
    try std.testing.expectError(error.BadPromptOrder, readPromptOrderEnv());
    try std.testing.expectEqual(@as(u32, 12), prompt_order.max_wait_s);
    _ = env.unsetenv(wait_name);
    _ = env.unsetenv(order_name);
    _ = env.setenv(qos_name, "garbage", 1); // refused by name, never the rotation unseen
    try std.testing.expectError(error.BadPromptOrder, readPromptOrderEnv());
    _ = env.unsetenv(qos_name);
    _ = env.setenv(order_name, "seriel", 1); // a typo of serial: refused
    try std.testing.expectError(error.BadPromptOrder, readPromptOrderEnv());
}

test "the coded parity assembly is assembleParity's with the assembly and down's quant in one launch" {
    const src = @embedFile("prompt2d.zig");
    const Body = struct {
        fn of(text: []const u8, name: []const u8) []const u8 {
            const at = std.mem.indexOf(u8, text, name).?;
            return text[at .. at + std.mem.indexOf(u8, text[at..], "\n    }\n").?];
        }
    };
    const plain = Body.of(src, "pub fn assembleParity(");
    const coded = Body.of(src, "pub fn assembleParityCodes(");
    // the exchange and the parts as assembleParity takes them, then the one launch in place of assembleRest
    for ([_][]const u8{ "const rest = ex.rest orelse return error.NoRest;", "const ig = t.gu[t.p];", "try t.packExchange(e, xd, p, ig + t.ir);", "const part0 = if (t.p == 0) xd else t.peer;", "const part1 = if (t.p == 1) xd else t.peer;" }) |line| {
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, plain, line));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, coded, line));
    }
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, plain, "try e.ops.assembleRest(e.s, part0, part1, pick, rest.owner, t.xd_full, p, ig, t.ir);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, coded, "try mx.assembleQuant(k, e.s, part0, part1, pick, rest.owner, t.xd_full, p, ig, t.ir, sc);"));
}

test "a decode gather's row blocks fit a slot" {
    _ = &Two.quartersDecode;
    var bb: [max_decode_blocks][2]usize = undefined;
    const head = 129536; // a pair-0 row of the head's quarters (32,384 fp32), 34 to a 4,456,448-byte slot
    const slot = 4456448;
    try std.testing.expectEqualSlices([2]usize, &.{.{ 0, 34 }}, try decodeBlocks(34, head, slot, &bb));
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 34 }, .{ 34, 1 } }, try decodeBlocks(35, head, slot, &bb));
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 34 }, .{ 34, 14 } }, try decodeBlocks(48, head, slot, &bb));
    // a drafter pass's partials: 80 rows of 10,240 bytes in one block
    try std.testing.expectEqualSlices([2]usize, &.{.{ 0, 80 }}, try decodeBlocks(80, 10240, slot, &bb));
    try std.testing.expectError(error.TooLarge, decodeBlocks(1, slot + 16, slot, &bb));
    // more blocks than a gather takes
    try std.testing.expectError(error.TooLarge, decodeBlocks(9 * 34, head, slot, &bb));
    try std.testing.expectEqual(@as(usize, 0), (try decodeBlocks(0, head, slot, &bb)).len);
}

test "a ring that cannot take an exchange says so: too large, not whole float4s, no row pair ring" {
    for ([_]anyerror{ error.TooLarge, error.NotFloat4, error.NoPairRing }) |err| try std.testing.expect(!try Two.ringed(err));
    try std.testing.expect(try Two.ringed({}));
    try std.testing.expectError(error.RingFailed, Two.ringed(error.RingFailed));
}

test "TF_DS_PF_RASTER: unset is off, 2 4 8 16 are the groups, any other text is no setting; a rank that differs is named" {
    try std.testing.expectEqual(@as(?u32, 0), rasterSetting(null));
    for ([_]u32{ 2, 4, 8, 16 }) |g| {
        var b: [4]u8 = undefined;
        try std.testing.expectEqual(@as(?u32, g), rasterSetting(try std.fmt.bufPrint(&b, "{d}", .{g})));
    }
    for ([_][]const u8{ "", "0", "1", "3", "32", "08", "8 ", "-8", "eight", "true" }) |t| try std.testing.expectEqual(@as(?u32, null), rasterSetting(t));
    const lead: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 0, .pool = 1 << 24, .replay_floor = 0, .idx_rb = 1, .pf_raster = 8 };
    var same = lead;
    var buf: [512]u8 = undefined;
    try std.testing.expectEqualStrings("", same.differ(lead, 2, &buf));
    same.pf_raster = 0;
    try std.testing.expectEqualStrings("rank 2 has TF_DS_PF_RASTER=0 and rank 0 8 (0: unset): set the same on every rank", same.differ(lead, 2, &buf));
    same.pf_raster = 4;
    try std.testing.expectEqualStrings("rank 1 has TF_DS_PF_RASTER=4 and rank 0 8 (0: unset): set the same on every rank", same.differ(lead, 1, &buf));
    try std.testing.expectEqual(@as(usize, 0), @sizeOf(Settings) % 8);
}

test "an answer too short for a weight digest is refused" {
    const Peer = struct {
        fn run(end: Link) void {
            var l = end;
            defer l.close();
            var frame: [@sizeOf(Settings)]u8 = undefined;
            _ = l.recv(&frame) catch return;
            l.send("rank 1 holds 4 streams") catch return; // words alone: no digest before them
        }
    };
    const ends = try testLinks();
    var f: Fan = .{};
    defer f.close();
    f.links[0] = ends[0];
    const t = try std.Thread.spawn(.{}, Peer.run, .{ends[1]});
    var why: [1024]u8 = undefined;
    var heard: Heard = .{};
    const served: Settings = .{ .world = 2, .streams = 8, .round_rows = 16, .rails = 0, .drafts = 1, .graphs = 0, .graph_cap = 0, .pool = 1 << 19, .replay_floor = 0, .idx_rb = 1 };
    const said = f.exchange(0, served, no_digest, &why, &heard);
    f.close();
    t.join();
    try std.testing.expectError(error.BadAnswer, said);
}

test "four ranks, digests not shared: rank 0 hears every follower's weight digest in link order, and the first follower's words" {
    const served: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1 };
            p.said = f.exchange(rank, mine, digest, &p.why, &p.heard) catch |e| {
    const said = f.exchange(0, served, @splat(0xd0), &why, &heard);
    const served: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1 };
            p.said = f.exchange(rank, mine, digest, &p.why, &p.heard) catch |e| {
        var why: [1024]u8 = undefined;
        const heard = f.lead(&id, cs.lead, &why);
        t.join();
        const said = f.exchange(0, served, @splat(0xd0), &why, &heard);
        f.close();
        try std.testing.expect(peer.err == null);
        try std.testing.expectEqualSlices(u8, &id, &peer.id);
        try std.testing.expectEqualStrings(peer.said, try heard);
        try std.testing.expectEqual(cs.differs, peer.said.len > 0);
        if (cs.differs) {
            const setting = if (cs.follower.?.streams != cs.lead.?.streams) "--streams 8" else "TF_DS_NEIGHBOR_RDMA";
            try std.testing.expect(std.mem.indexOf(u8, peer.said, setting) != null);
    const served: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1 };
            _ = f.exchange(rank, served, @splat(@intCast(0x10 * rank + rank)), &p.why, &p.heard) catch |e| {
        }
    }
    _ = try f.exchange(0, served, @splat(0xd0), &why, &heard);
test "a follower runs rank 0's TF_DS_CUDA_SCHED, or says what differs" {
    const lead: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1, .sched = cuda.context.sched_blocking };
}

test "a follower runs rank 0's indexer score rows, or says to set the same on every rank" {
    const lead: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 2 };
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("", lead.differ(lead, 1, &buf));
    var rb = lead;
    rb.idx_rb = 1;
    try std.testing.expectEqualStrings("", rb.differ(rb, 1, &buf));
    try std.testing.expectEqualStrings("rank 1 has TF_DS_IDX_RB=1 and rank 0 2: set the same on every rank", rb.differ(lead, 1, &buf));
    try std.testing.expectEqualStrings("rank 3 has TF_DS_IDX_RB=2 and rank 0 1: set the same on every rank", lead.differ(rb, 3, &buf));
    var all = rb;
    all.replay_floor = 0;
    try std.testing.expectEqualStrings("rank 3 has TF_DS_REPLAY_FLOOR=0 and rank 0 1: set the same on every rank; rank 3 has TF_DS_IDX_RB=1 and rank 0 2: set the same on every rank", all.differ(lead, 3, &buf));
    comptime var sum: usize = 0;
    inline for (@typeInfo(Settings).@"struct".field_names) |name| sum += @sizeOf(@FieldType(Settings, name));
    try std.testing.expectEqual(sum, @sizeOf(Settings));
    try std.testing.expectEqual(@as(u32, 11), frame_protocol);
test "a follower runs rank 0's TF_DS_PASS_ROWS, or says to set the same on every rank; rank 0's TF_DS_PASS_FROM travels and is never compared" {
    const lead: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1, .pass_rows = 4096, .pass_from = 8192 };
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("", lead.differ(lead, 1, &buf));
    var mine = lead;
    mine.pass_from = 0; // a follower's own: rank 0's alone is used
    try std.testing.expectEqualStrings("", mine.differ(lead, 1, &buf));
    mine.pass_rows = 2048;
    try std.testing.expectEqualStrings("rank 1 has TF_DS_PASS_ROWS=2048 and rank 0 4096: set the same on every rank", mine.differ(lead, 1, &buf));
    try std.testing.expectEqualStrings("rank 3 has TF_DS_PASS_ROWS=4096 and rank 0 2048: set the same on every rank", lead.differ(mine, 3, &buf));
    // every field whole 4- or 8-byte words: no padding travels
    const lead: Settings = .{ .world = 4, .streams = 8, .round_rows = 16, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 20, .replay_floor = 0, .idx_rb = 1, .land_slots = 2 };
    try std.testing.expectEqual(@as(u32, 11), frame_protocol); // kind 16 (cfg) after kind 15 (snap_sync), the Rocinante-only Settings; 11: fill_pair carries the third pass's rows
    older.protocol = 5; // a follower whose image lacks fill_pair
    try std.testing.expectEqualStrings("rank 2 decodes frame protocol 5 and rank 0 sends 11: run every rank from the same image", older.differ(lead, 2, &buf));
}
test "the expert pack as codes needs the checkpoint's own weights, the parity rest blocks and whole 128-value blocks" {
    try std.testing.expectEqual(@as(?[]const u8, null), packCodesOff(true, 512, 128));
    try std.testing.expectEqualStrings("needs the checkpoint's own FP8 / MXFP4 weights", packCodesOff(false, 512, 128).?);
    try std.testing.expectEqualStrings("needs TF_DS_2D_GU=parity (each slot's rest block in the pack)", packCodesOff(true, 512, 0).?);
    try std.testing.expectEqualStrings("needs a pack of whole 128-value blocks (its scales move in 16-byte units)", packCodesOff(true, 576, 128).?);
    try std.testing.expectEqualStrings("needs a pack of whole 128-value blocks (its scales move in 16-byte units)", packCodesOff(true, 512, 96).?);
test "wo_b's input as codes needs the checkpoint's own weights and an FP8 wo_b of both pairs' whole 32-value groups" {
    var lays: [40]weights.Layer = undefined;
    woBLayers(&lays);
    const uw = [2]usize{ 2048, 2048 };
    try std.testing.expectEqual(@as(?[]const u8, null), catCodesOff(true, &lays, uw));
    try std.testing.expectEqualStrings("needs the checkpoint's own FP8 / MXFP4 weights", catCodesOff(false, &lays, uw).?);
    const fp8 = "needs an FP8 wo_b on every layer (the checkpoint's own)";
    var odd = lays;
    odd[39].wo_b.fmt = .fp4;
    try std.testing.expectEqualStrings(fp8, catCodesOff(true, &odd, uw).?);
    odd = lays;
    odd[7].wo_b.fmt = .bf16;
    try std.testing.expectEqualStrings(fp8, catCodesOff(true, &odd, uw).?);
    const groups = "needs wo_b's input as both pairs' parts, each of whole 32-value groups";
    try std.testing.expectEqualStrings(groups, catCodesOff(true, &lays, .{ 2040, 2056 }).?);
    odd = lays;
    odd[20].wo_b.k = 4064;
    try std.testing.expectEqualStrings(groups, catCodesOff(true, &odd, uw).?);
    // unequal pair widths of whole groups run
    try std.testing.expectEqual(@as(?[]const u8, null), catCodesOff(true, &lays, .{ 2016, 2080 }));
test "quartersDecode sends each block's own rows of src at its place in the whole exchange, over the ring alone" {
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const blocks = try decodeBlocks(n, @max(w[0], w[1]) * esize, rs.slotBytes(), &bb);"));
    // no fallback: a refused block ends the exchange by name
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "error.RingRefusedBlock"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, qbody, "error.RingRefused"));
    try std.testing.expect(std.mem.indexOf(u8, qbody, "t.parts") == null);
}

test "adoptSettings keeps every device buffer of the second pass's Two as its own" {
    // A device pointer (or scratch set) added to `Two` that is missing from the keep list would be copied from the first pass and shared
    // by two passes in flight.
    const src = @embedFile("prompt2d.zig");
    const head = "pub fn adoptSettings(";
    const at = std.mem.indexOf(u8, src, head).?;
    const body = src[at..std.mem.indexOfPos(u8, src, at, "t.rc = null;").?];
    var seen: usize = 0;
    const info = @typeInfo(Two).@"struct";
    inline for (info.field_names, info.field_types) |name, ty| {
        if (ty == u64 or ty == ?exl3_experts2d.RestScratch) {
            seen += 1;
            try std.testing.expect(std.mem.indexOf(u8, body, "t." ++ name) != null);
        }
    }
    try std.testing.expect(seen >= 6);
test "the exchange switches that remain: a named bit each in Settings.exch, asked as \"1\" only; the retired bits are reserved and a rank that sets one is named as one this build has no name for" {
    const kept = [_]struct { name: []const u8, bit: u32, var_ptr: *bool }{
        .{ .name = "TF_DS_ROCI_P2_FIRST", .bit = exch_roci_p2_first, .var_ptr = &roci_p2_first_asked },
        .{ .name = "TF_DS_ROCI_ENGRAM_AHEAD", .bit = exch_roci_engram_ahead, .var_ptr = &roci_engram_ahead_asked },
        .{ .name = "TF_DS_ROCI_SENT_FIRST", .bit = exch_roci_sent_first, .var_ptr = &roci_sent_first_asked },
        .{ .name = "TF_DS_ROCI_COMBINE_DOOR", .bit = exch_roci_combine_door, .var_ptr = &roci_combine_door_asked },
        .{ .name = "TF_DS_ROCI_STAGGER", .bit = exch_roci_stagger, .var_ptr = &roci_stagger_asked },
        .{ .name = "TF_DS_ROCI_STAGGER_LANES", .bit = exch_roci_stagger_lanes, .var_ptr = &roci_stagger3_asked },
        .{ .name = "TF_DS_ENGRAM_PARTNER", .bit = exch_engram_partner, .var_ptr = &engram_partner_asked },
    };
    var buf: [1024]u8 = undefined;
    for (kept) |k| {
        try std.testing.expectEqualStrings(k.name, exch_names[@ctz(k.bit)]);
        const was = k.var_ptr.*;
        defer k.var_ptr.* = was;
        k.var_ptr.* = false;
        try std.testing.expectEqual(@as(u32, 0), exchAsked() & k.bit);
        k.var_ptr.* = true;
        try std.testing.expectEqual(k.bit, exchAsked() & k.bit);
        // a follower that differs is named
        const lead: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1, .exch = k.bit };
        var mine = lead;
        mine.exch = 0;
        var want: [160]u8 = undefined;
        const w = try std.fmt.bufPrint(&want, "rank 2 has {s}=0 and rank 0 1: set the same on every rank", .{k.name});
        try std.testing.expectEqualStrings(w, mine.differ(lead, 2, &buf));
    }
    // the retired switches' bits have no name: an image of the old layout is refused at the protocol before its bits are read, and a bit set by a
    // rank of this layout's number is named as one this build cannot name
    const lead: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1 };
    var odd = lead;
    odd.exch = 1 << 0; // the old TF_DS_OWNER_YX bit
    try std.testing.expect(std.mem.indexOf(u8, odd.differ(lead, 1, &buf), "switches this build has no name for") != null);
    try std.testing.expectEqualStrings("rank 1 decodes frame protocol 9 and rank 0 sends 11: run every rank from the same image", odd.differ(lead, 1, &buf));

test "a default-on switch (TF_DS_ROCI_WAIT, TF_DS_RING_WAIT): unset and 1 are on; 0, empty and any other text are off" {
    try std.testing.expect(switchDefaultOn(null));
    try std.testing.expect(switchDefaultOn("1"));
    for ([_][]const u8{ "0", "", "yes", "true", " 1", "11" }) |t| try std.testing.expect(!switchDefaultOn(t));
    // the other switches are unchanged: off unless 1
    try std.testing.expect(!switchAsked(null));
    try std.testing.expect(switchAsked("1"));
}

test "TF_DS_QGATE_CANARY's text: 1 the head, 2 and 3 the body; unset or other text none; any canary is kern bit 31" {
    try std.testing.expectEqual(Qgate.off, qgateAsked(null));
    try std.testing.expectEqual(Qgate.head, qgateAsked("1"));
    try std.testing.expectEqual(Qgate.body_fp8, qgateAsked("2"));
    try std.testing.expectEqual(Qgate.body_ulp, qgateAsked("3"));
    for ([_][]const u8{ "", "0", "4", "12", "2 ", " 3", "yes", "true" }) |t| try std.testing.expectEqual(Qgate.off, qgateAsked(t));
    const keep = qgate_canary_asked;
    defer qgate_canary_asked = keep;
    for ([_]Qgate{ .head, .body_fp8, .body_ulp }) |q| {
        qgate_canary_asked = q;
        try std.testing.expectEqual(kern_qgate_canary, kernAsked() & kern_qgate_canary);
    }
    qgate_canary_asked = .off;
    try std.testing.expectEqual(@as(u32, 0), kernAsked() & kern_qgate_canary);
}

test "the ranks whose word is not rank 0's, a bit each" {
    try std.testing.expectEqual(@as(u32, 0), differMask(&.{ 2, 2, 2, 2 }));
    try std.testing.expectEqual(@as(u32, 0b0010), differMask(&.{ 2, 3, 2, 2 }));
    try std.testing.expectEqual(@as(u32, 0b1110), differMask(&.{ 1, 2, 3, 3 }));
    try std.testing.expectEqual(@as(u32, 0b1000), differMask(&.{ 3, 3, 3, 1 }));
    try std.testing.expectEqual(@as(u32, 0), differMask(&.{3}));
}
