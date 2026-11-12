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

const ops_mod = @import("ops.zig");

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
    /// The ranks' weight digests are given back to every follower once rank 0 has them all and every settings check passed
    /// (TF_DS_SSD_EVERY_RANK: a rank's store is named from all four weight files). Every rank alike.
    share_digests: bool = false,

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
        heard.* = .{};
        if (rank < heard.weights.len) heard.weights[rank] = digest;
        return if (rank == 0) f.lead(mine, why, heard) else f.follow(rank, mine, digest, why, heard);
    }

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
        heard.lead = mine;
        var first: []const u8 = "";
        var answer: [answer_max]u8 = undefined;
        for (f.links, 1..) |l, g| if (l) |x| {
            const got = try x.recv(&answer);
            if (got.len < @sizeOf(Digest)) return error.BadAnswer;
            heard.weights[g] = got[0..@sizeOf(Digest)].*;
            const text = got[@sizeOf(Digest)..];
            if (first.len == 0 and text.len > 0) {
                const k = @min(text.len, why.len);
                @memcpy(why[0..k], text[0..k]);
                first = why[0..k];
            }
        };
        // every rank's digest back to each follower (all agreed: a follower that did not is leaving), in rank order
        if (f.share_digests and first.len == 0) {
            var table: [4 * @sizeOf(Digest)]u8 = undefined;
            for (heard.weights, 0..) |w, g| @memcpy(table[g * @sizeOf(Digest) ..][0..@sizeOf(Digest)], &w);
            for (f.links) |l| if (l) |x| try x.send(&table);
        }
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
        heard.lead = theirs;
        const text = if (mine) |s| s.differ(theirs, rank, why) else "";
        var answer: [answer_max]u8 = undefined;
        @memcpy(answer[0..@sizeOf(Digest)], &digest);
        const k = @min(text.len, answer.len - @sizeOf(Digest));
        @memcpy(answer[@sizeOf(Digest)..][0..k], text[0..k]);
        try l.send(answer[0 .. @sizeOf(Digest) + k]);
        if (f.share_digests and text.len == 0) {
            // rank 0 gives every rank's digest back once every rank agreed; this rank's own must be the one it heard
            var table: [4 * @sizeOf(Digest)]u8 = undefined;
            const t = try l.recv(&table);
            if (t.len != table.len or !std.mem.eql(u8, t[rank * @sizeOf(Digest) ..][0..@sizeOf(Digest)], &digest)) return error.BadAnswer;
            for (&heard.weights, 0..) |*w, g| w.* = t[g * @sizeOf(Digest) ..][0..@sizeOf(Digest)].*;
        }
        return text;
    }

    const answer_max = @sizeOf(Digest) + 1024; // a follower's answer: its weight digest, then its words (settingsChecked's why)
};

/// A rank's weight digest: the SHA-256 of its weight file (TF_DS_RANK_CACHE_SHA); no_digest: none.
pub const Digest = [32]u8;
pub const no_digest: Digest = @splat(0);

/// What the start-up exchange gives a rank (Fan.settingsChecked).
pub const Heard = struct {
    /// rank 0's Settings (rank 0: its own); null when rank 0 sent none
    lead: ?Settings = null,
    /// the ranks' weight digests by rank: rank 0 holds every rank's, a follower its own (no_digest: none sent, or
    /// another rank's) unless Fan.share_digests gave it all four
    weights: [4]Digest = @splat(no_digest),
};

/// TF_DS_RANK_CACHE_SHA's text as a digest: 64 lowercase hex digits, "sha256:" before them or not; anything else null.
pub fn weightDigest(text: []const u8) ?Digest {
    const prefix = "sha256:";
    const hex = if (std.mem.startsWith(u8, text, prefix)) text[prefix.len..] else text;
    if (hex.len != 2 * @sizeOf(Digest)) return null;
    var out: Digest = undefined;
    for (&out, 0..) |*b, i| b.* = (nibble(hex[2 * i]) orelse return null) << 4 | (nibble(hex[2 * i + 1]) orelse return null);
    return out;
}

fn nibble(c: u8) ?u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        else => null,
    };
}

/// 13; 4 = 1-11 (fill_resume 10, rdma_idle 11); 5 = 1-12 (and snap_land 12, the SSD tier's landing); 6 = 1-13 (and
/// head_whole 13, TF_DS_HEAD_CAND's whole head after a round's candidates fell back); 7 = 1-14 (and fill_pair 14, TF_DS_ROCI_STAGGER's two passes); 8 = 1-15 (and snap_sync 15, the SSD tier's stores on every rank); 9 = the same kinds, a Settings of the Rocinante-only layout (no
/// NCCL cores, retired exchange and kernel bits) and a start-up exchange without a library's id. 10 = 1-16 (and cfg 16, a live change of settings of class A to every rank: live_config.CfgFrame). 11 = the same kinds, a fill_pair frame that carries the third pass's rows (TF_DS_ROCI_STAGGER_LANES; 0: a pair). A new set of kinds takes the next
/// number, never an old one, so a follower image with other kinds refuses at the start (Settings) instead of ending
/// with BadFrame at a kind's first use.
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
    idx_rb: u32, // TF_DS_IDX_RB (tri_index.idx_rb, the indexer's prompt score rows a program): the same
    land_slots: u32 = 0, // rank 0's TF_DS_SPILL_LAND_SLOTS (0: its SSD tier off): every follower takes it, never compared
    exch: u32 = 0, // the exchange switches asked for, bit i exch_names[i] (exchAsked): the same
    kern: u32 = 0, // the kernel switches asked for, bit i kern_names[i] (kernAsked): the same
    kern2: u32 = 0, // the kernel switches beyond kern's 32 bits, bit i kern2_names[i] (kern2Asked): the same
    pf_split: u32 = 0, // TF_DS_PF_SPLIT asked for (mx.PfSplit): the same
    pass_rows: u32 = 2048, // TF_DS_PASS_ROWS (pass_rows_asked: a pass's rows at most, the chunk's allocation): the same
    pass_from: u32 = 0, // rank 0's TF_DS_PASS_FROM (its pass rule's): every follower takes it, never compared
    sched: u32 = 0, // TF_DS_CUDA_SCHED's request and what the primary context showed (cuda.context.schedCode): the same
    pf_raster: u32 = 0, // TF_DS_PF_RASTER's group of row tiles (pf_raster_asked; 0: off): the same (this word was the layout's padding, always 0)
    cfg: u64 = 0, // the digest of what engine.cfg set on this rank (settings.cfgDigest; 0: no file read): the same on every rank

    // "DS41" is another build's frame (its replay floor where this one carries graphs and the graph cap). A change of
    // the fields or of the follower's answer bumps the last letter, so a follower refuses a rank 0 of another layout.
    // A new exchange or kernel switch takes the next bit of exch or kern and keeps the letter: differ refuses every bit
    // that differs, one this build has no name for included
    pub const tag: u32 = 0x62355344; // "DS5b"

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
        if (mine.idx_rb != lead.idx_rb)
            w.print("{s}rank {d} has TF_DS_IDX_RB={d} and rank 0 {d}: set the same on every rank", .{ sepIf(&w, sep), rank, mine.idx_rb, lead.idx_rb }) catch {};
        if (mine.pass_rows != lead.pass_rows)
            w.print("{s}rank {d} has TF_DS_PASS_ROWS={d} and rank 0 {d}: set the same on every rank", .{ sepIf(&w, sep), rank, mine.pass_rows, lead.pass_rows }) catch {};
        if (mine.sched != lead.sched)
            w.print("{s}rank {d} has TF_DS_CUDA_SCHED {s} and rank 0 {s}: set the same on every rank", .{ sepIf(&w, sep), rank, cuda.context.schedWord(mine.sched), cuda.context.schedWord(lead.sched) }) catch {}
        else if (mine.sched == cuda.context.sched_invalid)
            w.print("{s}rank {d} and rank 0 have TF_DS_CUDA_SCHED text that is not one of auto, spin, yield, blocking: correct it on every rank", .{ sepIf(&w, sep), rank }) catch {}
        else if (mine.sched == cuda.context.sched_active_mismatch)
            w.print("{s}rank {d} and rank 0 asked for TF_DS_CUDA_SCHED and the primary context was already active with different scheduling flags: release it or set the same on every rank", .{ sepIf(&w, sep), rank }) catch {}
        else if (mine.sched == cuda.context.sched_after_mismatch)
            w.print("{s}rank {d} and rank 0 asked for TF_DS_CUDA_SCHED and the primary context did not take it: check the driver on every rank", .{ sepIf(&w, sep), rank }) catch {};
        if (mine.exch != lead.exch) bitsDiffer(&w, sep, &exch_names, "exchange", mine.exch, lead.exch, rank);
        if (mine.kern != lead.kern) bitsDiffer(&w, sep, &kern_names, "kernel", mine.kern, lead.kern, rank);
        if (mine.kern2 != lead.kern2) bitsDiffer(&w, sep, &kern2_names, "kernel (second word)", mine.kern2, lead.kern2, rank);
        if (mine.cfg != lead.cfg) {
            if (mine.cfg == 0)
                w.print("{s}rank {d} read no engine.cfg and rank 0 did: give every rank the same file (TF_ENGINE_CFG)", .{ sepIf(&w, sep), rank }) catch {}
            else if (lead.cfg == 0)
                w.print("{s}rank {d} read an engine.cfg and rank 0 read none: give every rank the same file (TF_ENGINE_CFG)", .{ sepIf(&w, sep), rank }) catch {}
            else
                w.print("{s}rank {d} applied another engine.cfg than rank 0 (the values it sets differ): give every rank the same file (TF_ENGINE_CFG)", .{ sepIf(&w, sep), rank }) catch {};
        }
        if (mine.pf_raster != lead.pf_raster)
            w.print("{s}rank {d} has TF_DS_PF_RASTER={d} and rank 0 {d} (0: unset): set the same on every rank", .{ sepIf(&w, sep), rank, mine.pf_raster, lead.pf_raster }) catch {};
        if (mine.pf_split != lead.pf_split) {
            if (mx.PfSplit.name(mine.pf_split)) |m| if (mx.PfSplit.name(lead.pf_split)) |l|
                w.print("{s}rank {d} has TF_DS_PF_SPLIT={s} and rank 0 {s}: set the same on every rank", .{ sepIf(&w, sep), rank, m, l }) catch {};
            if (mx.PfSplit.name(mine.pf_split) == null or mx.PfSplit.name(lead.pf_split) == null)
                w.print("{s}rank {d} and rank 0 differ in TF_DS_PF_SPLIT ({d} and {d}), a value this build has no name for: run every rank from the same image", .{ sepIf(&w, sep), rank, mine.pf_split, lead.pf_split }) catch {};
        }
        return w.buffered();
    }

    /// differ's words for a word of switches (`names[i]` bit i): each named one whose bit differs, then any other
    /// differing bit.
    fn bitsDiffer(w: *std.Io.Writer, sep: []const u8, names: []const []const u8, kind: []const u8, mine: u32, lead: u32, rank: u32) void {
        var named: u32 = 0; // (an empty name: a reserved bit, no switch of this build)
        for (names, 0..) |name, i| {
            const bit = @as(u32, 1) << @intCast(i);
            if (name.len == 0) continue;
            named |= bit;
            if ((mine ^ lead) & bit == 0) continue;
            w.print("{s}rank {d} has {s}={d} and rank 0 {d}: set the same on every rank", .{ sepIf(w, sep), rank, name, @intFromBool(mine & bit != 0), @intFromBool(lead & bit != 0) }) catch {};
        }
        const other = (mine ^ lead) & ~named;
        if (other != 0) w.print("{s}rank {d} and rank 0 differ in {s} switches this build has no name for (bits 0x{x}): run every rank from the same image", .{ sepIf(w, sep), rank, kind, other }) catch {};
    }

    fn sepIf(w: *const std.Io.Writer, sep: []const u8) []const u8 {
        return if (w.end > 0) sep else "";
    }

    fn setWord(x: u32) []const u8 {
        return if (x != 0) "set" else "unset";
    }
};

/// The exchange switches the start-up settings carry (Settings.exch): bit i is exch_names[i], set when asked for. Each
/// is read once by Model.open, before the settings and its agreement.
pub const exch_names = [_][]const u8{ "", "", "", "", "", "", "", "", "TF_DS_ROCI_P2_FIRST", "TF_DS_ROCI_ENGRAM_AHEAD", "", "TF_DS_ROCI_SENT_FIRST", "", "", "", "", "", "", "TF_DS_ROCI_COMBINE_DOOR", "", "", "TF_DS_ROCI_STAGGER", "TF_DS_ENGRAM_PARTNER", "TF_DS_RING_WC", "TF_DS_RING_INPLACE", "TF_DS_RING_WORD", "TF_DS_RING_WORD_SIGNAL", "TF_DS_RING_WAIT", "TF_DS_ROCI_STAGGER_LANES", "TF_DS_PACK_EPILOGUE", "", "" };
pub const exch_roci_p2_first: u32 = 1 << 8;
pub const exch_roci_engram_ahead: u32 = 1 << 9;
pub const exch_roci_sent_first: u32 = 1 << 11;
pub const exch_roci_combine_door: u32 = 1 << 18;
pub const exch_roci_stagger: u32 = 1 << 21;
pub const exch_engram_partner: u32 = 1 << 22;
pub const exch_ring_wc: u32 = 1 << 23;
pub const exch_ring_inplace: u32 = 1 << 24;
pub const exch_ring_word: u32 = 1 << 25;
pub const exch_ring_word_atomic: u32 = 1 << 26; // the switch's signal is the atomic fetch-and-add (unset: the inline word)
pub const exch_ring_wait: u32 = 1 << 27; // TF_DS_RING_WAIT: the rings' queue pairs wait in the card (no proxy threads)
pub const exch_pack_epilogue: u32 = 1 << 29; // TF_DS_PACK_EPILOGUE: gate/up's twin writes the expert pack's codes (every rank alike)
pub const exch_roci_stagger_lanes: u32 = 1 << 28; // TF_DS_ROCI_STAGGER_LANES=3 (unset or 2: clear): three passes in flight, a third lane on every rank
/// The last bit of the word (the others in between are free): every rank keeps its own SSD store (TF_DS_SSD_EVERY_RANK).
/// An exchange switch's text: "1" asks for it; unset or any other text does not.
/// A switch that is on unless set otherwise (TF_DS_ROCI_WAIT, TF_DS_RING_WAIT): unset or "1" is on; any other text, "0" included, is off (a typo
/// takes the fallback path, never the new one by accident).
pub fn switchDefaultOn(text: ?[]const u8) bool {
    const v = text orelse return true;
    return std.mem.eql(u8, v, "1");
}

pub fn switchAsked(text: ?[]const u8) bool {
    const v = text orelse return false;
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
/// TF_DS_SSD_FAULT as Model.open parsed it, once: the gate drill's faults of the four stores, each on one named rank (spill_sync.Fault).
/// Every rank reads the same text and acts only when it is the named one; it needs TF_DS_SSD_EVERY_RANK.
pub var ssd_fault: @import("spill_sync.zig").Fault = .{};

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

/// TF_DS_PACK_EPILOGUE as Model.open read it (switchAsked), once (Settings.exch): the expert pack's codes are written by gate/up's twin kernel
/// (tf_dsn_sf_gu_exact_q_kernel) and packQuant does not run. Bit-identical by contract (pack_epilogue.zig); default off.
pub var pack_epilogue_asked: bool = false;

/// TF_DS_ENGRAM_PARTNER as Model.open read it (switchAsked), once (Settings.exch; the ranks must agree: the start opens a lane on every rank or none):
/// the column partners each read half of a prompt step's Engram misses and swap them over a third Rocinante lane.
pub var engram_partner_asked: bool = false;

/// Settings.exch of the switches Model.open read.
pub fn exchAsked() u32 {
    var x: u32 = 0;
    if (roci_p2_first_asked) x |= exch_roci_p2_first;
    if (roci_engram_ahead_asked) x |= exch_roci_engram_ahead;
    if (roci_sent_first_asked) x |= exch_roci_sent_first;
    if (roci_combine_door_asked) x |= exch_roci_combine_door;
    if (roci_stagger_asked) x |= exch_roci_stagger;
    if (roci_stagger3_asked) x |= exch_roci_stagger_lanes;
    if (pack_epilogue_asked) x |= exch_pack_epilogue;
    if (ring_wc_asked) x |= exch_ring_wc;
    if (ring_inplace_asked) x |= exch_ring_inplace;
    if (ring_word_asked) x |= exch_ring_word;
    if (ring_word_atomic_asked) x |= exch_ring_word_atomic;
    if (ring_wait_asked) x |= exch_ring_wait;
    if (engram_partner_asked) x |= exch_engram_partner;
    return x;
}

/// The kernel switches the start-up settings carry (Settings.kern): bit i is kern_names[i], set when asked for (the
/// text "1", switchAsked; TF_DS_QGATE_CANARY's "1", "2" or "3", qgateAsked); an empty name is a reserved bit (kern_reserved). Each is read once by Model.open, before the
/// settings.
pub const kern_names = [_][]const u8{ "TF_DS_WOA_W8", "TF_DS_QUANT_ONCE", "TF_DS_PF_GEMM5", "TF_DS_GROUP_TILES", "TF_DS_TOPK_TILES", "TF_DS_IDX_PROJ_OWNED", "TF_DS_BF16_ROWS", "TF_DS_PF_RUN128", "TF_DS_DECODE_FUSE", "TF_DS_DRAFT_4WAY", "TF_DS_HEAD_PACK12", "TF_DS_HC_LATE_FORK", "TF_DS_ATTN_U", "TF_DS_IDX_U", "TF_DS_ATTN_U_DECODE", "TF_DS_ASSEMBLE_QUANT", "TF_DS_ATTN_EPILOGUE", "TF_DS_DRAFT_MAIN_4WAY", "TF_DS_MMA_SF", "TF_DS_IDX_V", "TF_DS_HC_STRIDE", "TF_DS_HC_FUSED", "TF_DS_MMA_SF_EXACT", "TF_DS_PASS_TAIL", "TF_DS_HEAD_CAND", "TF_DS_HEAD_KEY", "TF_DS_DECODE_KSPLIT", "TF_DS_ATTN_EPI_FUSED", "TF_DS_GEMM_W16", "TF_DS_GEMM_R2", "TF_DS_GEMM_HALVES", "TF_DS_QGATE_CANARY" };
pub const kern_woa_w8: u32 = 1 << 0;
pub const kern_quant_once: u32 = 1 << 1;
pub const kern_pf_gemm5: u32 = 1 << 2;
pub const kern_group_tiles: u32 = 1 << 3;
pub const kern_topk_tiles: u32 = 1 << 4;
pub const kern_idx_proj_owned: u32 = 1 << 5;
pub const kern_bf16_rows: u32 = 1 << 6;
pub const kern_pf_run128: u32 = 1 << 7;
pub const kern_decode_fuse: u32 = 1 << 8;
pub const kern_draft_4way: u32 = 1 << 9;
pub const kern_head_pack12: u32 = 1 << 10; // (its switch's code is the packed target head's)
pub const kern_hc_late_fork: u32 = 1 << 11;
pub const kern_attn_u: u32 = 1 << 12;
pub const kern_idx_u: u32 = 1 << 13;
pub const kern_idx_v: u32 = 1 << 19;
pub const kern_attn_u_decode: u32 = 1 << 14;
pub const kern_assemble_quant: u32 = 1 << 15;
pub const kern_hc_stride: u32 = 1 << 20;
pub const kern_draft_main_4way: u32 = 1 << 17;
pub const kern_mma_sf: u32 = 1 << 18;
pub const kern_attn_epilogue: u32 = 1 << 16;
pub const kern_mma_sf_exact: u32 = 1 << 22;
pub const kern_pass_tail: u32 = 1 << 23;
pub const kern_head_cand: u32 = 1 << 24;
pub const kern_head_key: u32 = 1 << 25;
pub const kern_decode_ksplit: u32 = 1 << 26;
pub const kern_gemm_w16: u32 = 1 << 28;
pub const kern_hc_fused: u32 = 1 << 21;
pub const kern_attn_epi_fused: u32 = 1 << 27;
pub const kern_gemm_r2: u32 = 1 << 29;
pub const kern_gemm_halves: u32 = 1 << 30;

/// The kernel switches beyond kern's 32 bits (Settings.kern2): bit i is kern2_names[i], set when asked for. Same rules as
/// kern_names: read once by Model.open, differ refuses every bit that differs, one this build has no name for included.
pub const kern2_names = [_][]const u8{ "TF_DS_HC_2S", "TF_DS_DRAFT_VOCAB" };
pub const kern2_hc_2s: u32 = 1 << 0;
pub const kern2_draft_vocab: u32 = 1 << 1;

/// The kernel word's reserved bits: an empty name in kern_names, a bit kept for a switch this build does not carry,
/// never asked for here.
pub const kern_reserved: u32 = blk: {
    var x: u32 = 0;
    for (kern_names, 0..) |name, i| {
        if (name.len == 0) x |= 1 << i;
    }
    break :blk x;
};

/// TF_DS_WOA_W8 as Model.open read it (switchAsked), once (Settings.kern, weights.woa_w8).
pub var woa_w8_asked: bool = false;

/// TF_DS_QUANT_ONCE as Model.open read it (switchAsked), once (Settings.kern, prompt.quant_once).
pub var quant_once_asked: bool = false;

/// TF_DS_PF_GEMM5 as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.useGemm5).
pub var pf_gemm5_asked: bool = false;

/// TF_DS_GROUP_TILES as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.useGroupTiles).
pub var group_tiles_asked: bool = false;

/// TF_DS_TOPK_TILES as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.useTopkTiles).
pub var topk_tiles_asked: bool = false;

/// TF_DS_IDX_PROJ_OWNED as Model.open read it (switchAsked), once (Settings.kern, Two.agreeIdxProjOwned).
pub var idx_proj_owned_asked: bool = false;

/// TF_DS_BF16_ROWS as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.bf16_rows).
pub var bf16_rows_asked: bool = false;

/// TF_DS_PF_RUN128 as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.useRun128).
pub var pf_run128_asked: bool = false;

/// TF_DS_DECODE_FUSE as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.useDecodeFuse).
pub var decode_fuse_asked: bool = false;

/// TF_DS_DRAFT_4WAY as Model.open read it (switchAsked), once (Settings.kern; in effect: weights.draft_4way, decided
/// before the weights load).
pub var draft_4way_asked: bool = false;

/// TF_DS_HC_LATE_FORK as Model.open read it (switchAsked), once (Settings.kern, round.Round.useLateFork).
pub var hc_late_fork_asked: bool = false;

/// TF_DS_ATTN_EPI_FUSED as Model.open read it (switchAsked), once (Settings.kern, tri_attn.attn_epi_fused): a prompt chunk's
/// attention epilogue (inverse RoPE and wo_a's quant) in the final store of the part kernel from the kit's extra set.
pub var attn_epi_fused_asked: bool = false;

/// TF_DS_HC_2S as Model.open read it (switchAsked), once (Settings.kern, tri_hc.hc_2s): the two-block mHC mix with the four streams of a
/// K block in one program (the kernel kit's extra set).
pub var hc_2s_asked: bool = false;

/// TF_DS_HC_FUSED as Model.open read it (switchAsked), once (Settings.kern, tri_hc.hc_fused): a prompt chunk's mHC
/// mixes and their finish in one launch from the kit's extra set.
pub var hc_fused_asked: bool = false;

/// TF_DS_ATTN_U as Model.open read it (switchAsked), once (Settings.kern, tri_attn.attn_u).
pub var attn_u_asked: bool = false;

/// TF_DS_IDX_U as Model.open read it (switchAsked), once (Settings.kern, tri_index.idx_u).
pub var idx_u_asked: bool = false;

/// TF_DS_IDX_V as Model.open read it (switchAsked), once (Settings.kern, tri_index.idx_v).
pub var idx_v_asked: bool = false;

/// TF_DS_ATTN_U_DECODE as Model.open read it (switchAsked), once (Settings.kern, tri_attn.attn_u_decode).
pub var attn_u_decode_asked: bool = false;
/// TF_DS_ATTN_TWIN as Model.open read it (u, w or z; null: unset = u), once (tri_attn.attn_twin). Not in Settings.kern:
/// the twins' bytes are _sparse_attn_part_u's (the kernels replay's attn_w / attn_z sections compare them).
pub var attn_twin_text: ?[]const u8 = null;
/// TF_DS_ROPE_W as Model.open read it (switchAsked), once (tri_norm.rope_w). Not in Settings.kern: the twin's bytes are
/// _rope_heads's (the kernels replay's rope_w section compares them).
pub var rope_w_asked: bool = false;
/// TF_DS_IDX_TPP as Model.open read it (the text of the env; null / unset = 4, the default), once (tri_index.idx_tpp). Not in Settings.kern: the twin's
/// bytes are _index_score_rows_u's (the kernels replay's idx_t section compares them).
pub var idx_tpp_text: ?[]const u8 = null;

/// TF_DS_ASSEMBLE_QUANT as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.assemble_quant).
pub var assemble_quant_asked: bool = false;

/// TF_DS_ATTN_EPILOGUE as Model.open read it (switchAsked), once (Settings.kern, prompt.attn_epilogue).
pub var attn_epilogue_asked: bool = false;

/// TF_DS_HC_STRIDE as Model.open read it (switchAsked), once (Settings.kern, prompt.hc_stride).
pub var hc_stride_asked: bool = false;

/// TF_DS_DRAFT_MAIN_4WAY as Model.open read it (switchAsked), once (Settings.kern; in effect: weights.draft_main_4way,
/// decided before the weights load).
pub var draft_main_4way_asked: bool = false;

/// TF_DS_MMA_SF as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.useMmaSf).
pub var mma_sf_asked: bool = false;

/// TF_DS_MMA_SF_EXACT as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.useMmaSf).
pub var mma_sf_exact_asked: bool = false;

/// TF_DS_HEAD_CAND as Model.open read it (switchAsked), once (Settings.kern, Model.head_cand).
pub var head_cand_asked: bool = false;

/// TF_DS_HEAD_KEY as Model.open read it (switchAsked), once (Settings.kern, Model.head_key).
pub var head_key_asked: bool = false;

/// TF_DS_GEMM_W16 as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.useGemmW16).
pub var gemm_w16_asked: bool = false;

/// TF_DS_SPLIT64N as Model.open read it (switchAsked), once (mx.Kernels.split64n; not in Settings: the 64-column tile's bytes are the 128-column
/// tile's, so ranks may differ): the exact arithmetic's split64 GEMMs of wkv and wq_a on a 64 x 64 tile.
pub var split64n_asked: bool = false;

/// TF_DS_PF_T64 as Model.open read it (switchAsked), once (mx.Kernels.usePfT64; not in Settings: the 64 x 64 tile's bytes are the 128 x 128
/// tile's, so ranks may differ): the router's bf16 GEMM of a prompt chunk on a 64 x 64 tile.
pub var pf_t64_asked: bool = false;

/// TF_DS_GEMM_R2 as Model.open read it (switchAsked), once (Settings.kern, mx.Kernels.useGemmR2).
pub var gemm_r2_asked: bool = false;

/// TF_DS_DECODE_KSPLIT as Model.open read it (switchAsked), once (Settings.kern, plan.Split.ksplit, round.zig's attention):
/// a decode round's wo_b takes this node's own wo_a groups' outputs alone (no column-ring exchange) over every output column,
/// and the four partials are gathered and summed. Not the same bits as the baseline (the fp32 additions' order), a decode-only switch.
pub var decode_ksplit_asked: bool = false;

/// TF_DS_RING_WC as Model.open read it (switchAsked), once (Settings.exch, rdma.Settings.wc): the RDMA rings' send and
/// landing slots from cudaHostAlloc(Mapped | WriteCombined).
pub var ring_wc_asked: bool = false;

/// TF_DS_RING_INPLACE as Model.open read it (switchAsked), once (Settings.exch, rdma.Settings.inplace): the producers of
/// the decode exchanges write their partial into a fixed send slot and publish it; needs TF_DS_RING_WC.
pub var ring_inplace_asked: bool = false;

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

/// TF_DS_GEMM_HALVES as Model.open read it (switchAsked), once (Settings.kern, mx.gemm_halves): a prompt pass of more
/// than 4,096 rows runs its dense FP8, bf16 and pf GEMMs as 4,096-row blocks at most.
pub var gemm_halves_asked: bool = false;

/// TF_DS_PF_RASTER as Model.open read it (rasterSetting; 0: unset), once (Settings.pf_raster, mx.pf_raster): the group of
/// row tiles the prompt tile GEMMs walk their grid in where a launch's activation slab reaches mx.raster_slab_bytes.
pub var pf_raster_asked: u32 = 0;

/// TF_DS_PF_RASTER's setting: unset is off (0), "2", "4", "8" or "16" the group of row tiles; any other text is no
/// setting (null).
pub fn rasterSetting(text: ?[]const u8) ?u32 {
    const t = text orelse return 0;
    for ([_]u32{ 2, 4, 8, 16 }) |g| {
        var buf: [4]u8 = undefined;
        if (std.mem.eql(u8, t, std.fmt.bufPrint(&buf, "{d}", .{g}) catch unreachable)) return g;
    }
    return null;
}

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
/// TF_DS_DRAFT_VOCAB as Model.open read it (switchAsked), once (Settings.kern, draft.Drafter.vocab).
pub var draft_vocab_asked: bool = false;

/// Settings.kern of the switches Model.open read.
pub fn kernAsked() u32 {
    var x: u32 = 0;
    if (woa_w8_asked) x |= kern_woa_w8;
    if (quant_once_asked) x |= kern_quant_once;
    if (pf_gemm5_asked) x |= kern_pf_gemm5;
    if (group_tiles_asked) x |= kern_group_tiles;
    if (topk_tiles_asked) x |= kern_topk_tiles;
    if (idx_proj_owned_asked) x |= kern_idx_proj_owned;
    if (bf16_rows_asked) x |= kern_bf16_rows;
    if (pf_run128_asked) x |= kern_pf_run128;
    if (decode_fuse_asked) x |= kern_decode_fuse;
    if (draft_4way_asked) x |= kern_draft_4way;
    if (hc_late_fork_asked) x |= kern_hc_late_fork;
    if (attn_u_asked) x |= kern_attn_u;
    if (idx_u_asked) x |= kern_idx_u;
    if (idx_v_asked) x |= kern_idx_v;
    if (attn_u_decode_asked) x |= kern_attn_u_decode;
    if (assemble_quant_asked) x |= kern_assemble_quant;
    if (attn_epilogue_asked) x |= kern_attn_epilogue;
    if (hc_stride_asked) x |= kern_hc_stride;
    if (draft_main_4way_asked) x |= kern_draft_main_4way;
    if (mma_sf_asked) x |= kern_mma_sf;
    if (mma_sf_exact_asked) x |= kern_mma_sf_exact;
    if (pass_tail_asked) x |= kern_pass_tail;
    if (head_cand_asked) x |= kern_head_cand;
    if (head_key_asked) x |= kern_head_key;
    if (gemm_w16_asked) x |= kern_gemm_w16;
    if (hc_fused_asked) x |= kern_hc_fused;
    if (attn_epi_fused_asked) x |= kern_attn_epi_fused;
    if (gemm_r2_asked) x |= kern_gemm_r2;
    if (decode_ksplit_asked) x |= kern_decode_ksplit;

    if (gemm_halves_asked) x |= kern_gemm_halves;
    if (qgate_canary_asked != .off) x |= kern_qgate_canary;
    return x;
}

/// Settings.kern2 from the switches Model.open read.
pub fn kern2Asked() u32 {
    var x: u32 = 0;
    if (hc_2s_asked) x |= kern2_hc_2s;
    if (draft_vocab_asked) x |= kern2_draft_vocab;
    return x;
}

/// TF_DS_PF_SPLIT as Model.open read it (mx.PfSplit.parse; text that is no setting: off), once (Settings.pf_split,
/// mx.Kernels.useSplit).
pub var pf_split_asked: mx.PfSplit = .off;

/// TF_DS_PASS_ROWS as Model.open read it (passRowsSetting), once (Settings.pass_rows, Model.pass_rows): every rank's
/// chunk is allocated for passes of this many rows.
pub var pass_rows_asked: u32 = 2048;

/// TF_DS_PASS_FROM as rank 0's Model.open read it (passFromSetting), once (Settings.pass_from, the pass rule's
/// pass_from): no pass longer than the resume rows starts below it (0: from the prompt's first row; unset: four units).
pub var pass_from_asked: u32 = 8192;

/// The resume grid (model.chunk_rows): every pass but a merged final one runs whole units of it, and every kept state
/// stands on one.
pub const chunk_rows: usize = 2048;

/// TF_DS_PASS_TAIL as Model.open read it (switchAsked), once (Settings.kern, the pass rule's tail): a prompt's final
/// pass may run up to the pass rows, the remainder after whole resume units merged into it; every other pass stays whole
/// units on the resume grid.
pub var pass_tail_asked: bool = false;

/// TF_DS_PASS_FILLING as rank 0's Model.open read it (switchAsked), once (the pass rule's filling passage): another
/// prompt filling does not cut a pass to the resume rows, so prompts filling together take turns at whole passes; a
/// request waiting for admission or a stream decoding still keeps the resume rows. Rank 0 decides every pass size and
/// the followers run its frames, so the setting is not compared at the start; every rank's allocations come from
/// TF_DS_PASS_ROWS, which is.
pub var pass_filling_asked: bool = false;

/// The start line TF_DS_PASS_FILLING=1 prints once (Model.open).
pub const pass_filling_line = "TF_DS_PASS_FILLING=1: while nothing decodes and no request waits, a prompt's pass stays up to the pass rows when another prompt fills, so prompts filling together take turns at whole passes";

/// Read TF_DS_PASS_FILLING once at start (Model.open), rank 0's: "1" asks for it, any other text or unset does not.
pub fn readPassFillingEnv() void {
    pass_filling_asked = switchAsked(if (std.c.getenv("TF_DS_PASS_FILLING")) |v| std.mem.span(v) else null);
    if (pass_filling_asked) std.log.info("{s}", .{pass_filling_line});
}

/// The prompt-step order as rank 0's Model.open read it, once (TF_DS_PROMPT_ORDER, TF_DS_PROMPT_QOS and TF_DS_PROMPT_MAX_WAIT_S; engine_api.fill_order):
/// unset, both: the rotation, one step each in turn. Rank 0's host chooses the order and the followers run its frames, so
/// the setting is not compared at the start.
pub var prompt_order: @import("engine_api").fill_order.Policy = .{};

/// Read TF_DS_PROMPT_ORDER (`serial` or `turns`), TF_DS_PROMPT_QOS (`size:class:overtakes[:max_wait_s],...`) and
/// TF_DS_PROMPT_MAX_WAIT_S (whole seconds, every class's longest wait for a step; 30 unset) once at start
/// (Model.open), rank 0's. A text that is no setting refuses the start by name (a typo never becomes the default unseen);
/// a setting that is on prints its line, which names TF_DS_PROMPT_ORDER.
pub fn readPromptOrderEnv() error{BadPromptOrder}!void {
    const fill_order = @import("engine_api").fill_order;
    const order_text: ?[]const u8 = if (std.c.getenv("TF_DS_PROMPT_ORDER")) |v| std.mem.span(v) else null;
    const qos_text: ?[]const u8 = if (std.c.getenv("TF_DS_PROMPT_QOS")) |v| std.mem.span(v) else null;
    const wait_text: ?[]const u8 = if (std.c.getenv("TF_DS_PROMPT_MAX_WAIT_S")) |v| std.mem.span(v) else null;
    const r = fill_order.parse(order_text, qos_text, wait_text);
    if (!@import("builtin").is_test and !r.wait_ok) std.log.err("TF_DS_PROMPT_MAX_WAIT_S={s} is not whole seconds from 1 to 86400 (unset: {d} s)", .{ wait_text.?, fill_order.default_max_wait_s });
    if (!@import("builtin").is_test and !r.order_ok) std.log.err("TF_DS_PROMPT_ORDER={s} is not serial or turns (unset: the rotation)", .{order_text.?});
    if (!@import("builtin").is_test and !r.qos_ok) std.log.err("TF_DS_PROMPT_QOS={s} is refused ({s}); a table is size:class:overtakes[:max_wait_s],... smallest first, sizes ascending, classes not falling", .{ qos_text.?, r.qos_why });
    if (!r.wait_ok or !r.order_ok or !r.qos_ok) return error.BadPromptOrder;
    prompt_order = r.policy;
    if (r.policy.on) {
        var buf: [512]u8 = undefined;
        std.log.info("TF_DS_PROMPT_ORDER={s}: {s}", .{ @tagName(r.policy.order), fill_order.describe(&r.policy, &buf) });
    }
}

/// TF_DS_PASS_ROWS's setting for a resume grid of `resume_rows`: one unit (unset), or any whole number of units by number up to
/// `max_rows` (the engine's largest pass, prompt.max_pass_rows); null: no setting (any other text).
pub fn passRowsSetting(text: ?[]const u8, resume_rows: u32, max_rows: u32) ?u32 {
    const t = text orelse return resume_rows;
    const v = std.fmt.parseInt(u32, t, 10) catch return null;
    return if (v >= resume_rows and v % resume_rows == 0 and v <= max_rows) v else null;
}

/// The words of a TF_DS_PASS_ROWS that is no setting (passRowsSetting null): the text, the unit and the range.
pub fn passRowsRefusal(buf: []u8, text: ?[]const u8, resume_rows: u32, max_rows: u32) []const u8 {
    return std.fmt.bufPrint(buf, "TF_DS_PASS_ROWS={s}: not a setting; a pass is a whole number of {d}-row resume units, from {d} to {d} rows (2048, 4096, 6144, 8192 ... {d})", .{ text orelse "", resume_rows, resume_rows, max_rows, max_rows }) catch "TF_DS_PASS_ROWS: not a whole number of resume units up to the largest pass";
}

/// TF_DS_PASS_BESIDE as Model.open read it, once: the rows a prompt's pass may run while a stream decodes (the host's step
/// rows, engine_api.Info.beside_rows); the resume rows when unset, which is the rule before the setting existed. Rank 0's
/// host decides every pass size and the followers run its frames, so the setting is not compared at the start.
pub var pass_beside_asked: u32 = 2048;

/// TF_DS_ROUNDS_BESIDE as Model.open read it, once (engine_api.Info.rounds_beside): the Token Generation rounds that follow
/// each long prompt step while a stream generates, 2N after a stagger pair; 0: unset (the share's and the pass rule's
/// own cadence). Rank 0's host runs the rounds and decides the pair, so the setting is not compared at the start.
pub var rounds_beside_asked: u32 = 0;

/// TF_DS_PASS_BESIDE's setting for a resume grid of `resume_rows` under booted pass rows of `pass_rows`: the resume rows
/// (unset), or any whole number of units by number, never above the pass rows (a step between two pass sizes runs whole
/// units: lanes.stream.besideRows); null: no setting.
pub fn passBesideSetting(text: ?[]const u8, resume_rows: u32, pass_rows: u32) ?u32 {
    const t = text orelse return resume_rows;
    const v = std.fmt.parseInt(u32, t, 10) catch return null;
    return if (v >= resume_rows and v % resume_rows == 0 and v <= pass_rows) v else null;
}

/// TF_DS_PASS_FROM's setting for a resume grid of `resume_rows`: four units (unset), or any multiple of the unit from 0
/// on (0: a pass of more units may start at the prompt's first row; a multiple from one unit on says where one may);
/// null: no setting.
pub fn passFromSetting(text: ?[]const u8, resume_rows: u32) ?u32 {
    const t = text orelse return 4 * resume_rows;
    const v = std.fmt.parseInt(u32, t, 10) catch return null;
    return if (v % resume_rows == 0) v else null;
}

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
/// The row blocks a decode exchange of n rows takes at most (decodeBlocks).
pub const max_decode_blocks = 8;

/// quartersDecode's row blocks of an exchange of n rows whose widest part is `row_bytes` a row: [start, rows] each, every
/// block as many rows as a slot of `slot` bytes holds (the last the rest); a row wider than a slot: error.TooLarge.
pub fn decodeBlocks(n: usize, row_bytes: usize, slot: usize, out: *[max_decode_blocks][2]usize) ![]const [2]usize {
    const per = try ring2d.rowsPerSlot(row_bytes, slot);
    var k: usize = 0;
    var r0: usize = 0;
    while (r0 < n) : (r0 += per) {
        if (k == max_decode_blocks) return error.TooLarge;
        out[k] = .{ r0, @min(per, n - r0) };
        k += 1;
    }
    return out[0..k];
}

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
    // TF_DS_IDX_PROJ_OWNED in effect (agreeIdxProjOwned): the indexer's projections of an idx_shard layer on the owned
    // rows the scorer reads (prompt.projRows)
    idx_proj_owned: bool = false,

    /// The widths from the split (both pairs'), and the buffers for chunks of up to `cap` rows (and rounds) from `a`;
    /// `links`: this rank's communicator has a two-rank link (Comm.links), whose exchanges pass through `parts`.
    pub fn init(c: *const Config, w: *const weights.Weights, a: *prompt.Arena, s: plan.Split, cap: usize) !Two {
        const pp = s.pair orelse return error.NotA2DSplit;
        const s0 = s.atPair(0);
        const s1 = s.atPair(1);
        const x0 = s0.expertParts(c.*);
        const x1 = s1.expertParts(c.*);
        var t: Two = undefined;
        t.idx_proj_owned = false; // agreeIdxProjOwned sets it on a model start; other callers project every row
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

    /// assembleParity with the assembly and down's quant of it in one launch (TF_DS_DECODE_FUSE, mx.assembleQuant):
    /// t.xd_full as assembleParity leaves it, its codes and scales in sc.hq / sc.hs for mx.downDecCodes.
    pub fn assembleParityCodes(t: *const Two, e: *const prompt.Engine, k: *const mx.Kernels, ex: weights.Experts, xd: u64, pick: u64, p: usize, sc: mx.ExpertScratch) !void {
        const rest = ex.rest orelse return error.NoRest;
        const ig = t.gu[t.p];
        try t.packExchange(e, xd, p, ig + t.ir);
        const part0 = if (t.p == 0) xd else t.peer;
        const part1 = if (t.p == 1) xd else t.peer;
        try mx.assembleQuant(k, e.s, part0, part1, pick, rest.owner, t.xd_full, p, ig, t.ir, sc);
    }

    /// Whether a ring took the exchange: false when there is none or the part does not fit it (nothing sent then).
    fn ringed(r: anyerror!void) !bool {
        r catch |err| switch (err) {
            error.TooLarge, error.NotFloat4, error.NoPairRing => return false,
            else => return err,
        };
        return true;
    }

    pub fn quarters(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        if (!try ringed(rs.quarters(e.s, src, dst, n, w, esize))) return error.RingRefused;
    }

    /// TF_DS_DECODE_KSPLIT: this node's full-width partial `src` [rows, row_bytes] to the four nodes' planes of `dst` [4, rows, row_bytes].
    pub fn planes4(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, rows: usize, row_bytes: usize) !void {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        if (!try ringed(rs.planes4(e.s, src, dst, rows, row_bytes))) return error.RingRefused;
    }

    pub fn quartersDecode(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        var bb: [max_decode_blocks][2]usize = undefined;
        const blocks = try decodeBlocks(n, @max(w[0], w[1]) * esize, rs.slotBytes(), &bb);
        for (blocks, 0..) |b, i| {
            const sent = try ringed(rs.quartersRows(e.s, src + b[0] * w[t.p] * esize, dst, n, b[0], b[1], w, esize));
            // (a refusal is by widths alone: the first block's, before anything is sent)
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
        try dev.uploadAsync(0, std.mem.asBytes(&mine), stream.handle);
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

    /// TF_DS_IDX_PROJ_OWNED in effect: `asked` (the start-up Settings made it the same on every rank) and every rank's
    /// start check (`why`: this rank's reason it cannot, null when it can; prompt.idxProjOwnedOff), one world all-gather
    /// when asked: off on every rank when any rank cannot. One line says which.
    pub fn agreeIdxProjOwned(t: *Two, d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream, asked: bool, why: ?[]const u8) !void {
        t.idx_proj_owned = false;
        if (!asked) return;
        const cannot = try ranksThatCannot(d, comm, stream, why == null);
        t.idx_proj_owned = agreedOn(asked, cannot);
        var buf: [64]u8 = undefined;
        if (t.idx_proj_owned) std.log.info("TF_DS_IDX_PROJ_OWNED=1: the indexer's queries and head weights of an idx_shard layer on the owned rows its scorer reads, on every rank", .{}) else std.log.info("TF_DS_IDX_PROJ_OWNED=1: off on every rank: {s} cannot run it (this rank: {s})", .{ rankList(cannot, &buf), why orelse "can" });
    }

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
        try dev.uploadAsync(0, std.mem.asBytes(&mine), stream.handle);
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
    }

    /// This node: 2 p + r.
    pub fn node(t: *const Two) u32 {
        return @intCast(t.r + 2 * t.p);
    }

    pub fn catRank(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        if (!try ringed(rs.catRank(e.s, src, dst, n, w, esize))) return error.RingRefused;
    }

    pub fn pairGather(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, width: usize, esize: usize) !void {
        const rs = t.rings orelse return error.DecodeNeedsRings;
        if (!try ringed(rs.pairGather(e.s, src, dst, n, width, esize))) return error.RingRefused;
    }

    /// wo_2d after wo_a: the rank's four groups (this node's two in ch.u [n, uw[p]], the partner's two), then wo_b
    /// for this pair's output columns: ch.pa [n, ow[p]] fp32, the matching columns of TP2's partial.
    /// TF_DS_CAT_PARTNER: the rank's four groups from the column partner's part alone (partnerCat), into the same
    /// bytes of u_full.
    /// TF_DS_CAT_CODES: as FP8 codes and scales when the column ring does not take the bf16 part (woBCodes).
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
    }

    /// TF_DS_CAT_CODES' wo_b of a prompt chunk (the column ring not taking the bf16 part): this node's part of wo_b's
    /// input quantized as wo_b's own quant quantizes the rank's whole input (act_quant: each 32-value group of a row
    /// lies in one pair's part), the codes and scales exchanged with the column partner in place of the bf16 part
    /// (Two.exchange), both pairs' codes and scales copied into ch.wob_q / ch.wob_s (catCodesPlan), then the GEMM
    /// mx.linear runs after its own quant, on them: ch.pa [n, ow[p]] fp32, the bytes woB gives. ch.mxs' codes and
    /// scales are not written (its split partials serve the GEMM as in mx.linear).
    ///
    /// TF_DS_ROCI_CODES, a whole pass: the part is quantized into the arena's send region, goes to the column partner as RDMA WRITEs
    /// into its land region, and the copies read the partner's part there (the exchange is the transport's).
    ///
    /// With TF_DS_ROCI_SENT_FIRST (`sent_first`, the caller's owner-sum conditions held) beside TF_DS_ROCI_CODES the GEMM on the landed
    /// codes is the one-launch kernel that rings the owner sum's first part itself (prompt.codesGemmSentFirst): true when it ran.
    pub fn woBCodes(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer, sent_first: bool) !bool {
        const k = e.mx orelse return error.NotAnMxLinear;
        const shape = prompt.codesShape(e, ch) orelse return error.RociNoShape;
        const rc = t.rc.?;
        const send = rc.addr(.wob_send);
        const land = rc.addr(.wob_land);
        const pl = catCodesPlan(t.node(), t.rank_of, ch.n, t.uw, ch.u, send, land, ch.wob_q, ch.wob_s);
        if (lay.wo_b.k != pl.k) return error.UnsupportedShape;
        try catCodesQuant(k, e.s, pl);
        try rc.send(e.s, .wob, shape);
        try rc.wait(e.s, .wob);
        if (sent_first) {
            try catCodesCopies(e.ops, e.s, pl);
            if (try prompt.codesGemmSentFirst(e, ch, t, lay.wo_b, pl.q, pl.s)) return true;
            try mx.gemm(k, e.s, lay.wo_b, pl.q, pl.k, pl.s, pl.k / 32, ch.n, ch.pa, lay.wo_b.n, .f32, false, ch.mxs.z);
            return false;
        }
        // (the GEMM reading the two pairs' parts where they lie, the two-base read, measured slower on the transport's host pages: 1.860 ms a launch
        // against the copies and the plain GEMM's 1.302, the A operand's re-reads by the 20 column tiles not served by L2: it stays off)
        try catCodesGemm(k, e.ops, e.s, lay.wo_b, pl, ch.n, ch.pa, ch.mxs.z);
        return false;
    }

    /// moe_2d's experts after the routing: this node's gate / up blocks, the rank's whole intermediate assembled, down
    /// for this node's output columns: ch.pm [n, dw[p]] fp32. A chunk of fewer than EXACT_ROWS rows takes the decode
    /// window's fused path and scratch (experts2d.fused_ok), as the rounds do. `x_codes`: x's codes and scales are in
    /// ch.mxe already (TF_DS_FFN_CODES's routing), gate / up reads them as they are.
    pub fn experts(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer, x_codes: bool, cf: ?*const mx.CombineFirst) !void {
        const c = e.c;
        const n = ch.n;
        if (e.mx != null) {
            // the pack as codes over the transport (the decode rounds' experts are expertsNative's, through the rings)
            return t.expertsCodes(e, ch, lay.experts, ch.x, c.hidden, ch.pick, ch.wts, ch.xs.xd, ch.pm, n, x_codes, cf);
        }
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
    /// for this node's output columns: out [rows, dw[p]] fp32. split_k: a decode round's (mx.gateUp). x_codes: x's
    /// codes and scales in ch.mxe already (mx.gateUpCodes).
    pub fn expertsNative(t: *const Two, e: *const prompt.Engine, ch: *const prompt.Chunk, ex: weights.Experts, x: u64, ldx: usize, pick: u64, wts: u64, xd: u64, out: u64, rows: usize, split_k: bool, x_codes: bool, cf: ?*const mx.CombineFirst) !void {
        const k = e.mx.?;
        if (x_codes) try mx.gateUpCodes(k, e.s, e.d, ex, ch.mxe, pick, wts, rows, e.c.swiglu_limit, xd, ex.width) else try mx.gateUp(k, e.s, e.d, ex, ch.mxe, x, ldx, pick, wts, rows, e.c.swiglu_limit, xd, ex.width, split_k);
        if (!split_k) if (prompt.ptimer) |pt| pt.mark(e.s, .m_gu) catch {};
        // TF_DS_ASSEMBLE_QUANT: the parity assembly and down's quant in one launch, down on the codes
        if (t.ir != 0 and k.assemble_quant) {
            try t.assembleParityCodes(e, k, ex, xd, pick, rows * ch.mxe.slots, ch.mxe);
            if (!split_k) if (prompt.ptimer) |pt| pt.mark(e.s, .m_x) catch {};
            if (cf) |f| {
                if (split_k) return error.CombineFirstOnDecode;
                return mx.downCodesFirst(k, e.s, ex, ch.mxe, pick, rows, out, t.dw[t.p], false, f);
            }
            return mx.downCodes(k, e.s, ex, ch.mxe, pick, rows, out, t.dw[t.p], false);
        }
        if (t.ir != 0) try t.assembleParity(e, ex, xd, pick, rows * ch.mxe.slots) else try t.catRank(e, xd, t.xd_full, rows * ch.mxe.slots, t.gu, 2);
        if (!split_k) if (prompt.ptimer) |pt| pt.mark(e.s, .m_x) catch {};
        if (cf) |f| {
            if (split_k) return error.CombineFirstOnDecode;
            return mx.downFirst(k, e.s, ex, ch.mxe, t.xd_full, ex.down_k, pick, rows, out, t.dw[t.p], false, f);
        }
        try mx.down(k, e.s, ex, ch.mxe, t.xd_full, ex.down_k, pick, rows, out, t.dw[t.p], false, split_k);
    }

    /// TF_DS_PACK_CODES' experts of a prompt chunk (native, parity; the column ring not taking the pack): gate / up as
    /// expertsNative into this node's pack xd, the pack quantized as down's quant quantizes it (packQuant: every
    /// 32-value group of a slot row's main and rest blocks, the groups of down's input), the codes and scales exchanged
    /// with the column partner in place of the bf16 pack (Two.exchange), both packs' codes and scales assembled as
    /// assembleParity assembles the values (packAssemble: each slot's rest from its owner pair) into down's input
    /// (ch.mxe.hq / hs), and down from them (mx.downCodes): out [rows, dw[p]] fp32, the bytes expertsNative gives.
    pub fn expertsCodes(t: *const Two, e: *const prompt.Engine, ch: *const prompt.Chunk, ex: weights.Experts, x: u64, ldx: usize, pick: u64, wts: u64, xd: u64, out: u64, rows: usize, x_codes: bool, cf: ?*const mx.CombineFirst) !void {
        const k = e.mx.?;
        const rest = ex.rest orelse return error.NoRest;
        const pc = try packCodesPlan(rows * ch.mxe.slots, t.gu[t.p], t.ir);
        const mine = t.rc.?.addr(.pack_send);
        const theirs = t.rc.?.addr(.pack_land);
        if (pack_epilogue_asked) {
            // TF_DS_PACK_EPILOGUE: gate/up's twin writes the pack's codes and scales into the send region itself (packSlot's places), so
            // packQuant does not run; where the twin cannot run this is an error, never the baseline two launches
            try mx.gateUpPack(k, e.s, e.d, ex, ch.mxe, x, ldx, pick, wts, rows, e.c.swiglu_limit, xd, ex.width, x_codes, .{ .codes = mine, .scales = mine + pc.scales, .ig = @intCast(pc.ig), .ir = @intCast(pc.ir), .P = @intCast(pc.p) });
        } else if (x_codes) try mx.gateUpCodes(k, e.s, e.d, ex, ch.mxe, pick, wts, rows, e.c.swiglu_limit, xd, ex.width) else try mx.gateUp(k, e.s, e.d, ex, ch.mxe, x, ldx, pick, wts, rows, e.c.swiglu_limit, xd, ex.width, false);
        if (prompt.ptimer) |pt| pt.mark(e.s, .m_gu) catch {};
        // the pack is quantized into the arena's send region and the column partner's lands in its land region, over the transport
        const shape = prompt.codesShape(e, ch) orelse return error.RociNoShape;
        if (!pack_epilogue_asked) try packQuant(k, e.s, xd, mine, pc);
        try t.rc.?.send(e.s, .pack, shape);
        try t.rc.?.wait(e.s, .pack);
        if (prompt.ptimer) |pt| pt.mark(e.s, .m_x) catch {};
        const part0 = if (t.p == 0) mine else theirs;
        const part1 = if (t.p == 1) mine else theirs;
        // TF_DS_ROCI_CODES: down reads the two parts where they lie and the owner's rest block (the three-base kernel), no assembly
        // and no hq / hs; where the kernels do not take the shape, the assembly and down on its output
        if (t.gu[0] == t.gu[1] and mx.threeTakes(k, ex, ch.mxe, t.gu[t.p], t.ir)) {
            var sc3 = ch.mxe;
            sc3.three = .{ .p0 = part0, .p1 = part1, .rows = pc.p, .ig = t.gu[t.p], .ir = t.ir };
            if (cf) |f| return mx.downCodesFirst(k, e.s, ex, sc3, pick, rows, out, t.dw[t.p], false, f);
            return mx.downCodes(k, e.s, ex, sc3, pick, rows, out, t.dw[t.p], false);
        }
        try packAssemble(e.ops, e.s, part0, part1, pick, rest.owner, ch.mxe, pc);
        if (cf) |f| return mx.downCodesFirst(k, e.s, ex, ch.mxe, pick, rows, out, t.dw[t.p], false, f);
        try mx.downCodes(k, e.s, ex, ch.mxe, pick, rows, out, t.dw[t.p], false);
    }

    /// Engram's projection summed over the ranks from its column quarters (model.py esum): the gathered width.
    pub fn engramWidth(t: *const Two) usize {
        return t.ew[0] + t.ew[1];
    }
};

/// The ranks of `comm` whose `can` is false, a bit each (bit k: rank k), from one gather of every rank's word over the
/// host links on `stream`: a start-up agreement, called by every rank together.
pub fn ranksThatCannot(d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream, can: bool) !u32 {
    const w = comm.world;
    var all: [8]u32 = undefined;
    if (w > all.len) return error.WorldTooLarge;
    var dev = try cuda.DeviceBuffer.alloc(d, (1 + w) * 4);
    defer dev.free();
    const mine: u32 = @intFromBool(can);
    try dev.uploadAsync(0, std.mem.asBytes(&mine), stream.handle);
    try comm.gatherHost(d, dev.ptr, dev.ptr + 4, 4, stream);
    try stream.synchronize();
    try dev.download(4, std.mem.sliceAsBytes(all[0..w]));
    return cannotMask(all[0..w]);
}

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

/// The ranks whose word is 0 (cannot), a bit each.
pub fn cannotMask(words: []const u32) u32 {
    var m: u32 = 0;
    for (words, 0..) |x, k| {
        if (x == 0) m |= @as(u32, 1) << @intCast(k);
    }
    return m;
}

/// A switch asked for on every rank is in effect when no rank cannot.
pub fn agreedOn(asked: bool, cannot: u32) bool {
    return asked and cannot == 0;
}

/// Why the owner sums over Rocinante cannot run on this rank (null: it can): attention, MoE and Engram all at owned rows,
/// passes of a whole number of resume units up to the largest pass (the transport's arenas and plans follow the booted rows:
/// rocinante.setCap), no second chunk stream, and the kit's two-block readers.
pub fn rociOff(ow: [2]usize, pass_rows: usize, readers: bool) ?[]const u8 {
    if (ownerYxOff(ow)) |why| return why;
    if (pass_rows < chunk_rows or pass_rows % chunk_rows != 0 or pass_rows > prompt.max_pass_rows) return "plans exist for passes of a whole number of 2048-row resume units, 2,048 to 16,384 rows";
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

/// The ranks of `mask` in words ("rank 2", "ranks 1 and 3", "ranks 0, 1 and 3").
pub fn rankList(mask: u32, buf: []u8) []const u8 {
    if (mask == 0) return "no rank";
    var w: std.Io.Writer = .fixed(buf);
    const n = @popCount(mask);
    w.writeAll(if (n == 1) "rank " else "ranks ") catch {};
    var seen: u32 = 0;
    var m = mask;
    while (m != 0) : (m &= m - 1) {
        const k = @ctz(m);
        seen += 1;
        const sep: []const u8 = if (seen == 1) "" else if (seen == n) " and " else ", ";
        w.print("{s}{d}", .{ sep, k }) catch {};
    }
    return w.buffered();
}

/// Why TF_DS_PACK_CODES, asked for, stays off (null: it runs): the codes are down's (the checkpoint's own weights), the
/// parity split's packs carry each slot's rest block (rest width `ir`), and a pack of `ig` main and `ir` rest values
/// splits into whole 32-value groups whose scales (four bytes a group) move in the assembly's 16-byte units.
pub fn packCodesOff(native: bool, ig: usize, ir: usize) ?[]const u8 {
    if (!native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    if (ir == 0) return "needs TF_DS_2D_GU=parity (each slot's rest block in the pack)";
    if (ig % 128 != 0 or ir % 128 != 0) return "needs a pack of whole 128-value blocks (its scales move in 16-byte units)";
    return null;
}

/// Whether a ring of `max_bytes` a part takes a bf16 pack of `rows` slot rows `width` values wide (rdma.Ring.gatherInto's
/// size rule for packExchange's slice).
pub fn ringTakes(max_bytes: usize, rows: usize, width: usize) bool {
    return rows * width * 2 <= max_bytes;
}

/// A pack as TF_DS_PACK_CODES moves it: `p` slot rows of a main block [p, ig] then a rest block [p, ir] (bf16 values,
/// gateUp's layout), quantized as one run of 32-value groups (the main block's p ig / 32, then the rest block's
/// p ir / 32): the codes (a byte a value, the same two blocks) at 0, the scales (fp32 a group, the same two blocks)
/// at `scales`, `bytes` in all.
pub const PackCodes = struct { p: usize, ig: usize, ir: usize, groups: usize, scales: usize, bytes: usize };

pub fn packCodesPlan(p: usize, ig: usize, ir: usize) !PackCodes {
    if (ig % 32 != 0 or ir % 32 != 0) return error.UnsupportedShape;
    const groups = p * (ig + ir) / 32;
    const scales = p * (ig + ir);
    return .{ .p = p, .ig = ig, .ir = ir, .groups = groups, .scales = scales, .bytes = scales + groups * 4 };
}

/// The pack's codes: act_quant of the bf16 pack `xd` as pc.groups rows of 32 values (one group a row: act_quant's
/// groups) into `codes` (pc's layout).
pub fn packQuant(k: *const mx.Kernels, s: cuda.Stream, xd: u64, codes: u64, pc: PackCodes) !void {
    try mx.quant(k, s, xd, .bf16, 32, pc.groups, 32, codes, 32, codes + pc.scales, 1);
}

/// The two pairs' pack codes `part0`, `part1` (pc's layout each) into down's input: sc.hq [p, 2 ig + ir] codes and
/// sc.hs [p, (2 ig + ir) / 32] scales, pair 0's main block, pair 1's, then each slot's rest from owner[pick]
/// (assembleRestBytes, once for the codes and once for the scales).
pub fn packAssemble(o: *const ops_mod.Ops, s: cuda.Stream, part0: u64, part1: u64, pick: u64, owner: u64, sc: mx.ExpertScratch, pc: PackCodes) !void {
    try o.assembleRestBytes(s, part0, part1, pick, owner, sc.hq, pc.p, pc.ig, pc.ir);
    try o.assembleRestBytes(s, part0 + pc.scales, part1 + pc.scales, pick, owner, sc.hs, pc.p, pc.ig / 32 * 4, pc.ir / 32 * 4);
}

/// Why TF_DS_CAT_CODES, asked for, stays off (null: it runs): the codes ride TF_DS_CAT_PARTNER's partner exchange (its
/// stage holds the partner's codes), they are wo_b's own input codes (the checkpoint's own weights, an FP8 wo_b on
/// every layer), and wo_b's K is both pairs' parts, each of whole 32-value groups.
pub fn catCodesOff(native: bool, layers: []const weights.Layer, uw: [2]usize) ?[]const u8 {
    if (!native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    for (layers) |lay| if (lay.wo_b.fmt != .fp8) return "needs an FP8 wo_b on every layer (the checkpoint's own)";
    const groups = "needs wo_b's input as both pairs' parts, each of whole 32-value groups";
    if (uw[0] % 32 != 0 or uw[1] % 32 != 0) return groups;
    for (layers) |lay| if (lay.wo_b.k != uw[0] + uw[1]) return groups;
    return null;
}

/// One act_quant launch: x [m, kdim] bf16 (row stride ldx) into the codes q (row stride ldq) and the scales s (row
/// stride lds).
pub const QuantCall = struct { x: u64, ldx: usize, m: usize, kdim: usize, q: u64, ldq: usize, s: u64, lds: usize };

/// woBCodes' launches and exchange: the quant of this node's part, one span of its codes then scales to its column
/// partner and one of the partner's into the stage, the four copies (pair 0's codes, pair 1's, pair 0's scales, pair
/// 1's) into wo_b's input, then the GEMM's operands: q [n, k] codes (row stride k), s [n, k / 32] fp32 scales (row
/// stride k / 32), k = uw0 + uw1: mx.linear's own layout after its quant.
pub const CatCodes = struct { quant: QuantCall, peer: u32, send: Span, recv: Span, copies: [4]RowCopy, q: u64, s: u64, k: usize };

/// woBCodes' first step: this node's part quantized into its codes and scales (pl.quant: act_quant, as wo_b's own).
pub fn catCodesQuant(k: *const mx.Kernels, s: cuda.Stream, pl: CatCodes) !void {
    const q = pl.quant;
    try mx.quant(k, s, q.x, .bf16, q.ldx, q.m, q.kdim, q.q, q.ldq, q.s, q.lds);
}

/// Both pairs' codes and scales into wo_b's input (pl.copies).
pub fn catCodesCopies(o: *const ops_mod.Ops, s: cuda.Stream, pl: CatCodes) !void {
    for (pl.copies) |cp| try o.copyRows(s, cp.src, cp.src_stride, cp.dst, cp.dst_stride, cp.width, cp.rows);
}

/// woBCodes' last step, after the exchange: both pairs' codes and scales into place (pl.copies), then wo_b's GEMM on
/// them, the call mx.linear makes after its quant: y [n, wo_b.n] fp32 (z: the split partials, mx.Scratch.z).
pub fn catCodesGemm(k: *const mx.Kernels, o: *const ops_mod.Ops, s: cuda.Stream, wo_b: weights.Linear, pl: CatCodes, n: usize, y: u64, z: u64) !void {
    try catCodesCopies(o, s, pl);
    try mx.gemm(k, s, wo_b, pl.q, pl.k, pl.s, pl.k / 32, n, y, wo_b.n, .f32, false, z);
}

/// Bytes of n rows of w values as codes and scales (a part of catCodesPlan's): n w codes, then n w / 32 fp32 scales.
pub fn catCodesBytes(n: usize, w: usize) usize {
    return n * w + n * (w / 32) * 4;
}

/// woBCodes of node g (2 p + r) under ranks `rank_of`, n rows: this node's u [n, uw[p]] bf16 quantized into
/// `codes` (its codes [n, uw[p]], then its scales [n, uw[p] / 32]: catCodesBytes), the column partner's in the same
/// layout into `stage`, then each pair's codes into wob_q [n, k] and its scales into wob_s [n, k / 32] at its columns
/// (pair 0's first, as partnerPlan places the bf16 parts).
pub fn catCodesPlan(g: u32, rank_of: [4]u32, n: usize, uw: [2]usize, u: u64, codes: u64, stage: u64, wob_q: u64, wob_s: u64) CatCodes {
    const p = g / 2;
    const k = uw[0] + uw[1];
    var pl: CatCodes = .{
        .quant = .{ .x = u, .ldx = uw[p], .m = n, .kdim = uw[p], .q = codes, .ldq = uw[p], .s = codes + n * uw[p], .lds = uw[p] / 32 },
        .peer = rank_of[g ^ 2],
        .send = .{ .addr = codes, .len = catCodesBytes(n, uw[p]) },
        .recv = .{ .addr = stage, .len = catCodesBytes(n, uw[1 - p]) },
        .copies = undefined,
        .q = wob_q,
        .s = wob_s,
        .k = k,
    };
    for (0..2) |pg| {
        const part = if (pg == p) codes else stage;
        pl.copies[pg] = .{ .src = part, .src_stride = uw[pg], .dst = wob_q + pg * uw[0], .dst_stride = k, .width = uw[pg], .rows = n };
        pl.copies[2 + pg] = .{ .src = part + n * uw[pg], .src_stride = uw[pg] / 32 * 4, .dst = wob_s + pg * (uw[0] / 32) * 4, .dst_stride = k / 32 * 4, .width = uw[pg] / 32 * 4, .rows = n };
    }
    return pl;
}

/// One copyRows: `rows` rows of `width` bytes from `src` (rows `src_stride` bytes apart) to `dst` (`dst_stride` apart).
pub const RowCopy = struct { src: u64, src_stride: usize, dst: u64, dst_stride: usize, width: usize, rows: usize };

/// Why the owner sums cannot run (null: they can): widening the two pairs' sums into one row needs equal pair widths.
pub fn ownerYxOff(ow: [2]usize) ?[]const u8 {
    if (ow[0] != ow[1]) return "needs equal pair widths";
    return null;
}

/// Owner sums of node g under ranks `rank_of` (Two.rank_of), for owned blocks of q rows: rank k owns rows R(k) =
/// [k q, k q + q). Every field is a first row; each part is q rows of a pair's columns.
pub const OwnerPlan = struct {
    row_peer: u32, // the row partner's rank (node g ^ 1: the other TP2 rank of this pair)
    col_peer: u32, // the column partner's rank (node g ^ 2: this TP2 rank of the other pair)
    send1: [2]usize, // to the row partner: its rows, then its column partner's
    recv1: [2]usize, // from the row partner, into the stage: this node's rows, then its column partner's (the add's rows)
    send2: usize, // to the column partner: the sums at its rows
    recv2: usize, // from the column partner: the other pair's sums at this node's rows
};

pub fn ownerPlan(g: u32, rank_of: [4]u32, q: usize) OwnerPlan {
    const at = struct {
        fn rows(k: u32, n: usize) usize {
            return @as(usize, k) * n;
        }
    }.rows;
    return .{
        .row_peer = rank_of[g ^ 1],
        .col_peer = rank_of[g ^ 2],
        .send1 = .{ at(rank_of[g ^ 1], q), at(rank_of[g ^ 3], q) },
        .recv1 = .{ at(rank_of[g], q), at(rank_of[g ^ 2], q) },
        .send2 = at(rank_of[g ^ 2], q),
        .recv2 = at(rank_of[g], q),
    };
}

/// The word plane 1 of the owner sums' tight buffer holds: fp32 -0.0, for which x + (-0.0) == x at every non-NaN x,
/// signed zeros included, so the readers' y = bf16(G0 + G1) keeps the baseline bits.
pub const plane1_word: u32 = 0x80000000;

/// The owner sums' stage (Chunk.own_stage) for q owned rows of a pair's width w: `stage` the row partner's fp32
/// [2, q, w], `sums` this pair's bf16 sums [2, q, w] (this node's rows, then its column partner's), `oth` the other
/// pair's bf16 sums at this node's rows [q, w].
pub const OwnStage = struct {
    stage: u64,
    sums: u64,
    oth: u64,

    pub fn bytes(q: usize, w: usize) usize {
        return q * w * (2 * 4 + 2 * 2 + 2);
    }

    pub fn at(base: u64, q: usize, w: usize) OwnStage {
        const sums = base + 2 * q * w * 4;
        return .{ .stage = base, .sums = sums, .oth = sums + 2 * q * w * 2 };
    }
};

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

test "the pack's codes: the main block's groups then the rest block's, codes then scales, 10,321,920 bytes at 2,048 rows" {
    const pc = try packCodesPlan(2048 * 7, 512, 128);
    try std.testing.expectEqual(@as(usize, 2048 * 7), pc.p);
    try std.testing.expectEqual(@as(usize, 2048 * 7 * 20), pc.groups);
    try std.testing.expectEqual(@as(usize, 2048 * 7 * 640), pc.scales);
    try std.testing.expectEqual(@as(usize, 10_321_920), pc.bytes);
    try std.testing.expectError(error.UnsupportedShape, packCodesPlan(7, 500, 128));
    try std.testing.expectError(error.UnsupportedShape, packCodesPlan(7, 512, 100));
}

/// tf_ds_assemble_rest_kernel on host bytes: out [p, 2 ig8 + ir8] 16-byte units from part0 / part1 (main rows [p, ig8]
/// then rest rows [p, ir8] units each), each row's rest from owner[pick[row]].
fn assembleModel(part0: []const u8, part1: []const u8, pick: []const i32, owner: []const i32, out: []u8, p: usize, ig8: usize, ir8: usize) void {
    const units = 2 * ig8 + ir8;
    for (0..p) |row| {
        const rest = if (owner[@intCast(pick[row])] == 0) part0 else part1;
        const o = out[row * units * 16 ..];
        @memcpy(o[0 .. ig8 * 16], part0[row * ig8 * 16 ..][0 .. ig8 * 16]);
        @memcpy(o[ig8 * 16 .. 2 * ig8 * 16], part1[row * ig8 * 16 ..][0 .. ig8 * 16]);
        @memcpy(o[2 * ig8 * 16 .. units * 16], rest[(p * ig8 + row * ir8) * 16 ..][0 .. ir8 * 16]);
    }
}

/// act_quant's indexing on host values with a stand-in for its arithmetic: x [m, kdim] bf16 (row stride ldx), each
/// 32-value group (row r, group g) to 32 codes at q[r ldq + 32 g ..] and one scale at s[r lds + g], every byte a
/// function of that group's values alone (as tf_dsn_act_quant_kernel's: its amax, its scale, each value's code).
fn quantModel(x: []const u16, ldx: usize, m: usize, kdim: usize, q: []u8, ldq: usize, s: []f32, lds: usize) void {
    for (0..m) |r| for (0..kdim / 32) |g| {
        const v = x[r * ldx + 32 * g ..][0..32];
        var sum: u32 = 0;
        for (v) |y| sum +%= y;
        for (v, 0..) |y, i| q[r * ldq + 32 * g + i] = @truncate(y ^ (sum >> 3));
        s[r * lds + g] = @floatFromInt(sum);
    };
}

test "the pack's codes assembled into down's input are down's quant of the assembled pack: every slot row, both pairs, each slot's rest from its owner" {
    const gpa = std.testing.allocator;
    const p = 9; // slot rows
    const ig = 256;
    const ir = 128;
    const i = 2 * ig + ir;
    const ne = 10;
    var prng = std.Random.DefaultPrng.init(0x7a);
    const rng = prng.random();
    var pick: [p]i32 = undefined;
    for (&pick) |*x| x.* = rng.intRangeLessThan(i32, 0, ne);
    var owner: [ne]i32 = undefined;
    for (&owner) |*x| x.* = rng.intRangeLessThan(i32, 0, 2);
    // the pairs' bf16 packs (gateUp's layout: main block [p, ig] then rest block [p, ir])
    var packs: [2][]u16 = undefined;
    for (&packs) |*pk| {
        pk.* = try gpa.alloc(u16, p * (ig + ir));
        for (pk.*) |*x| x.* = rng.int(u16);
    }
    defer for (packs) |pk| gpa.free(pk);
    // baseline: assembleParity's values (assembleRest: units of 8 fp16 values), then down's quant [p, i]
    const full = try gpa.alloc(u16, p * i);
    defer gpa.free(full);
    assembleModel(std.mem.sliceAsBytes(packs[0]), std.mem.sliceAsBytes(packs[1]), &pick, &owner, std.mem.sliceAsBytes(full), p, ig / 8, ir / 8);
    const hq = try gpa.alloc(u8, p * i);
    defer gpa.free(hq);
    const hs = try gpa.alloc(f32, p * i / 32);
    defer gpa.free(hs);
    quantModel(full, i, p, i, hq, i, hs, i / 32);
    // the codes path: each pair's pack quantized as packQuant does (pc.groups rows of 32), assembled as packAssemble
    // does (codes, then scales, in restUnits' units)
    const pc = try packCodesPlan(p, ig, ir);
    var codes: [2][]align(16) u8 = undefined;
    for (&codes, packs) |*c, pk| {
        c.* = try gpa.alignedAlloc(u8, .@"16", pc.bytes);
        quantModel(pk, 32, pc.groups, 32, c.*[0..pc.scales], 32, @alignCast(std.mem.bytesAsSlice(f32, c.*[pc.scales..pc.bytes])), 1);
    }
    defer for (codes) |c| gpa.free(c);
    const hq2 = try gpa.alloc(u8, p * i);
    defer gpa.free(hq2);
    const hs2 = try gpa.alloc(f32, p * i / 32);
    defer gpa.free(hs2);
    const uc = try ops_mod.restUnits(pc.ig, pc.ir);
    assembleModel(codes[0], codes[1], &pick, &owner, hq2, pc.p, uc[0], uc[1]);
    const us = try ops_mod.restUnits(pc.ig / 32 * 4, pc.ir / 32 * 4);
    assembleModel(codes[0][pc.scales..], codes[1][pc.scales..], &pick, &owner, std.mem.sliceAsBytes(hs2), pc.p, us[0], us[1]);
    try std.testing.expectEqualSlices(u8, hq, hq2);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(hs), std.mem.sliceAsBytes(hs2));
    // assembleRest's own units, in bytes, are restUnits' of twice the values
    try std.testing.expectEqual([2]usize{ ig / 8, ir / 8 }, try ops_mod.restUnits(2 * ig, 2 * ir));
    try std.testing.expectError(error.UnsupportedShape, ops_mod.restUnits(24, 16));
}

/// Every layer's wo_b as the checkpoint loads it on a 2D node: FP8, this TP2 rank's 4,096 input rows, this pair's
/// 2,560 output columns.
fn woBLayers(lays: []weights.Layer) void {
    for (lays, 0..) |*l, i| {
        l.* = .{ .idx = @intCast(i), .ratio = 0 };
        l.wo_b = .{ .words = 0, .k = 4096, .n = 2560, .fmt = .fp8 };
    }
}

test "TF_DS_CAT_CODES at a full chunk: 4,718,592 bytes each way in place of the bf16 part's 8,388,608, wo_b's input 9,437,184, the partner's codes inside cat_stage" {
    try std.testing.expectEqual(@as(usize, 4_718_592), catCodesBytes(2048, 2048));
    try std.testing.expectEqual(@as(usize, 9_437_184), catCodesBytes(2048, 4096));
    // cat_stage, sized by TF_DS_CAT_PARTNER for the bf16 part, holds the codes at every row count
    for ([_]usize{ 1, 63, 1000, 1089, 2048 }) |n| try std.testing.expect(catCodesBytes(n, 2048) <= n * 2048 * 2);
    const u: u64 = 0x1000;
    const codes: u64 = 0x10_0000;
    const stage: u64 = 0x100_0000;
    const wq: u64 = 0x1000_0000;
    const ws: u64 = 0x2000_0000;
    for (0..4) |g| {
        const pl = catCodesPlan(@intCast(g), rankOf(orders[0]), 2048, .{ 2048, 2048 }, u, codes, stage, wq, ws);
        try std.testing.expectEqual(@as(u32, @intCast(g ^ 2)), pl.peer);
        try std.testing.expectEqual(Span{ .addr = codes, .len = 4_718_592 }, pl.send);
        try std.testing.expectEqual(Span{ .addr = stage, .len = 4_718_592 }, pl.recv);
        // this node's u rows quantized as wo_b's own quant reads them: bf16 rows of uw[p] values, 64 groups a row
        try std.testing.expectEqual(QuantCall{ .x = u, .ldx = 2048, .m = 2048, .kdim = 2048, .q = codes, .ldq = 2048, .s = codes + 2048 * 2048, .lds = 64 }, pl.quant);
        // the GEMM's operands: mx.linear's after its quant (codes row stride K = 4,096, scales row stride K / 32)
        try std.testing.expectEqual(@as(usize, 4096), pl.k);
        try std.testing.expectEqual(wq, pl.q);
        try std.testing.expectEqual(ws, pl.s);
    }
    // the column ring takes the bf16 part (partnerCat's attempt) at 1,088 rows or fewer: the codes run past it
    try std.testing.expect(ringTakes(4_456_448, 1088, 2048));
    try std.testing.expect(!ringTakes(4_456_448, 1089, 2048));
}

test "wo_b's input from the two pairs' codes is wo_b's own quant of the assembled input: every node, every node order, equal and unequal pair widths" {
    const n = 5;
    const at_codes: u64 = 2048;
    const at_stage: u64 = 3072;
    const at_q: u64 = 4096;
    const at_s: u64 = 5120;
    for ([_][2]usize{ .{ 64, 64 }, .{ 32, 96 } }) |uw| for (orders) |ord| {
        const nc = rankOf(ord);
        const k = uw[0] + uw[1];
        var prng = std.Random.DefaultPrng.init(0x4b);
        const rng = prng.random();
        // node g's bytes: its part u [n, uw[p]] bf16 at 0, the rest a filler no step may leave behind
        var mem: [4][6144]u8 align(16) = undefined;
        for (&mem, 0..) |*m, g| {
            @memset(m, 0xee);
            for (std.mem.bytesAsSlice(u16, m[0 .. n * uw[g / 2] * 2])) |*x| x.* = rng.int(u16);
        }
        var plans: [4]CatCodes = undefined;
        for (&plans, 0..) |*pl, g| pl.* = catCodesPlan(@intCast(g), nc, n, uw, 0, at_codes, at_stage, at_q, at_s);
        // each node's quant of its part (the plan's launch)
        for (plans, &mem) |pl, *m| {
            const q = pl.quant;
            quantModel(@alignCast(std.mem.bytesAsSlice(u16, m[q.x..][0 .. q.m * q.ldx * 2])), q.ldx, q.m, q.kdim, m[q.q..][0 .. q.m * q.ldq], q.ldq, @alignCast(std.mem.bytesAsSlice(f32, m[q.s..][0 .. q.m * q.lds * 4])), q.lds);
        }
        // one exchange a pair of column partners: each names the other's rank, and what one sends the other takes
        var after = mem;
        for (plans, 0..) |pl, g| {
            const h = nodeOfRank(nc, pl.peer);
            try std.testing.expectEqual(g ^ 2, h);
            try std.testing.expectEqual(nc[g], plans[h].peer);
            try std.testing.expectEqual(pl.send.len, plans[h].recv.len);
            @memcpy(after[h][plans[h].recv.addr..][0..plans[h].recv.len], mem[g][pl.send.addr..][0..pl.send.len]);
        }
        for (plans, 0..) |pl, g| for (pl.copies) |cp| copyRowsHost(&after[g], cp);
        // wo_b's own quant (mx.linear's) of TP2 rank r's whole input: row i is pair 0's part (node r's) then pair 1's
        // (node r + 2's)
        for (0..4) |g| {
            var full: [n * 128]u16 = undefined;
            for (0..n) |i| for (0..2) |pg| @memcpy(full[i * k + pg * uw[0] ..][0..uw[pg]], std.mem.bytesAsSlice(u16, mem[g % 2 + 2 * pg][i * uw[pg] * 2 ..][0 .. uw[pg] * 2]));
            var wq: [n * 128]u8 = undefined;
            var ws: [n * 4]f32 = undefined;
            quantModel(full[0 .. n * k], k, n, k, wq[0 .. n * k], k, ws[0 .. n * k / 32], k / 32);
            try std.testing.expectEqualSlices(u8, wq[0 .. n * k], after[g][at_q..][0 .. n * k]);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ws[0 .. n * k / 32]), after[g][at_s..][0 .. n * k / 32 * 4]);
            // nothing past the chunk's rows
            try std.testing.expect(std.mem.allEqual(u8, after[g][at_q + n * k .. at_q + n * k + 64], 0xee));
            try std.testing.expect(std.mem.allEqual(u8, after[g][at_s + n * k / 32 * 4 .. at_s + n * k / 32 * 4 + 64], 0xee));
            // the GEMM reads them as mx.linear's: codes row stride K, scales row stride K / 32
            try std.testing.expectEqual(k, plans[g].k);
            try std.testing.expectEqual(at_q, plans[g].q);
            try std.testing.expectEqual(at_s, plans[g].s);
        }
    };
}

/// copyRows on host bytes `m` (device addresses as offsets into it).
fn copyRowsHost(m: []u8, cp: RowCopy) void {
    for (0..cp.rows) |i| @memcpy(m[cp.dst + i * cp.dst_stride ..][0..cp.width], m[cp.src + i * cp.src_stride ..][0..cp.width]);
}

/// The node that rank `k` serves under `rank_of`.
fn nodeOfRank(rank_of: [4]u32, k: u32) usize {
    return std.mem.indexOfScalar(u32, &rank_of, k).?;
}

/// The six node orders rank 0 can serve under (TF_DS_2D_ORDER: rank 0 serves node 0).
const orders = [_][4]u32{ .{ 0, 1, 2, 3 }, .{ 0, 1, 3, 2 }, .{ 0, 2, 1, 3 }, .{ 0, 2, 3, 1 }, .{ 0, 3, 1, 2 }, .{ 0, 3, 2, 1 } };

test "owner sums: each round is a perfect matching and both ends list the same segments, under every node order" {
    const q: usize = 512;
    for (orders) |ord| {
        const nc = rankOf(ord);
        var owned: [4]bool = @splat(false);
        for (0..4) |gi| {
            const g: u32 = @intCast(gi);
            const pl = ownerPlan(g, nc, q);
            // the partners: node g ^ 1 and node g ^ 2, by their ranks, and mutual
            try std.testing.expectEqual(nc[g ^ 1], pl.row_peer);
            try std.testing.expectEqual(nc[g ^ 2], pl.col_peer);
            try std.testing.expectEqual(nc[g], ownerPlan(g ^ 1, nc, q).row_peer);
            try std.testing.expectEqual(nc[g], ownerPlan(g ^ 2, nc, q).col_peer);
            // R(k) = [k q, k q + q): this node's rows, then its column partner's
            try std.testing.expectEqual([2]usize{ nc[g] * q, nc[g ^ 2] * q }, pl.recv1);
            try std.testing.expectEqual([2]usize{ nc[g ^ 1] * q, nc[g ^ 3] * q }, pl.send1);
            try std.testing.expectEqual(nc[g] * q, pl.recv2);
            try std.testing.expectEqual(nc[g ^ 2] * q, pl.send2);
            // the k-th send is the partner's k-th receive: the same rows (and q rows of the pair's columns each)
            try std.testing.expectEqual(pl.send1, ownerPlan(g ^ 1, nc, q).recv1);
            try std.testing.expectEqual(pl.send2, ownerPlan(g ^ 2, nc, q).recv2);
            owned[pl.recv1[0] / q] = true;
        }
        try std.testing.expectEqual([4]bool{ true, true, true, true }, owned); // the owners' rows cover the chunk
    }
}

test "own_stage: the row partner's fp32 [2, q, w], the sums [2, q, w] and the other pair's [q, w], back to back" {
    try std.testing.expectEqual(@as(usize, 18_350_080), OwnStage.bytes(512, 2560));
    try std.testing.expectEqual(@as(usize, 36_700_160), OwnStage.bytes(1024, 2560));
    const st = OwnStage.at(0x10000, 512, 2560);
    try std.testing.expectEqual(@as(u64, 0x10000), st.stage);
    try std.testing.expectEqual(@as(u64, 0x10000 + 2 * 512 * 2560 * 4), st.sums);
    try std.testing.expectEqual(@as(u64, 0x10000 + 2 * 512 * 2560 * 4 + 2 * 512 * 2560 * 2), st.oth);
}

test "a weight digest is 64 lowercase hex digits, with sha256: before them or not" {
    const hex = "00ff10a500ff10a500ff10a500ff10a500ff10a500ff10a500ff10a500ff10a5";
    var want: Digest = undefined;
    for (&want, 0..) |*b, i| b.* = ([_]u8{ 0x00, 0xff, 0x10, 0xa5 })[i % 4];
    try std.testing.expectEqual(@as(?Digest, want), weightDigest(hex));
    try std.testing.expectEqual(@as(?Digest, want), weightDigest("sha256:" ++ hex));
    const bad = [_][]const u8{
        "",
        "sha256:",
        hex[1..], // 63 digits
        hex ++ "0", // 65
        "00ff10A5" ++ hex[8..], // upper case
        "00ff10g5" ++ hex[8..], // not hex
        "SHA256:" ++ hex,
        "sha256: " ++ hex[1..],
        " " ++ hex[1..],
        "sha512:" ++ hex,
    };
    for (bad) |b| try std.testing.expectEqual(@as(?Digest, null), weightDigest(b));
}

/// The two ends of one link for a test: a UNIX socket pair (frames as over TCP; no port to take, no accept to wait in).
fn testLinks() ![2]Link {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    return .{ .{ .fd = pair[0] }, .{ .fd = pair[1] } };
}

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
    }
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
}

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
    // a rank that applied another engine.cfg than rank 0 is named, and one that read none
    var cfg = lead;
    cfg.cfg = 0x1234;
    var lead_cfg = lead;
    lead_cfg.cfg = 0x9999;
    try std.testing.expect(std.mem.indexOf(u8, cfg.differ(lead_cfg, 2, &buf), "rank 2 applied another engine.cfg") != null);
    try std.testing.expect(std.mem.indexOf(u8, cfg.differ(lead_cfg, 2, &buf), "TF_ENGINE_CFG") != null);
    cfg.cfg = 0;
    try std.testing.expect(std.mem.indexOf(u8, cfg.differ(lead_cfg, 3, &buf), "rank 3 read no engine.cfg") != null);
    try std.testing.expectEqualStrings("", lead_cfg.differ(lead_cfg, 1, &buf));
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
    var floorless = served; // rank 2 runs without the floor, rank 3 with fewer streams: rank 0 says rank 2's words
    floorless.replay_floor = 0;
    var eight = served;
    eight.streams = 8;
    const Peer = struct {
        why: [1024]u8 = undefined,
        said: []const u8 = "",
        heard: Heard = .{},
        err: ?anyerror = null,
        fn run(p: *@This(), end: Link, rank: u32, mine: Settings) void {
            var f: Fan = .{};
            f.links[0] = end;
            defer f.close();
            const digest: Digest = @splat(@intCast(0x10 * rank + rank)); // 0x11, 0x22, 0x33
            p.said = f.exchange(rank, mine, digest, &p.why, &p.heard) catch |e| {
                p.err = e;
                return;
            };
        }
    };
    var peers: [3]Peer = .{ .{}, .{}, .{} };
    var threads: [3]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |t| t.join(); // after rank 0's ends are closed (the defer below runs first)
    // link g - 1 is follower g's (Fan.openAt: rank 0 listens on port + g - 1 for it)
    var f: Fan = .{};
    defer f.close();
    const settings = [3]Settings{ served, floorless, eight };
    for (&peers, &threads, settings, 1..) |*p, *t, mine, g| {
        const ends = try testLinks();
        f.links[g - 1] = ends[0];
        t.* = try std.Thread.spawn(.{}, Peer.run, .{ p, ends[1], @as(u32, @intCast(g)), mine });
        started += 1;
    }
    var why: [1024]u8 = undefined;
    var heard: Heard = .{};
    const said = f.exchange(0, served, @splat(0xd0), &why, &heard);
    f.close(); // a follower still waiting on rank 0 ends (PeerDown) instead of holding the join
    for (threads) |t| t.join();
    started = 0;
    try std.testing.expectEqualStrings("rank 2 has TF_DS_REPLAY_FLOOR=0 and rank 0 1: set the same on every rank", try said);
    for (peers, 1..) |p, g| {
        try std.testing.expect(p.err == null);
        try std.testing.expectEqual(@as(?Settings, served), p.heard.lead);
        try std.testing.expectEqual(g >= 2, p.said.len > 0);
    }
    try std.testing.expectEqualStrings("rank 3 holds 8 streams and rank 0 serves 16: start rank 3 with --streams 16 (or TF_DS_STREAMS=16)", peers[2].said);
    const want = [4]Digest{ @splat(0xd0), @splat(0x11), @splat(0x22), @splat(0x33) };
    try std.testing.expectEqual(want, heard.weights);
    try std.testing.expectEqual(@as(?Settings, served), heard.lead);
}

test "with the digests shared, every follower ends the exchange holding all four ranks' weight digests, and none does when a follower's settings differ" {
    const served: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1 };
    var eight = served;
    eight.streams = 8;
    const Peer = struct {
        why: [1024]u8 = undefined,
        said: []const u8 = "",
        heard: Heard = .{},
        err: ?anyerror = null,
        fn run(p: *@This(), end: Link, rank: u32, mine: Settings) void {
            var f: Fan = .{ .share_digests = true };
            f.links[0] = end;
            defer f.close();
            const digest: Digest = @splat(@intCast(0x10 * rank + rank));
            p.said = f.exchange(rank, mine, digest, &p.why, &p.heard) catch |e| {
                p.err = e;
                return;
            };
        }
    };
    for ([_]bool{ false, true }) |one_differs| {
        var peers: [3]Peer = .{ .{}, .{}, .{} };
        var threads: [3]std.Thread = undefined;
        var started: usize = 0;
        defer for (threads[0..started]) |t| t.join();
        var f: Fan = .{ .share_digests = true };
        defer f.close();
        const settings = [3]Settings{ served, if (one_differs) eight else served, served };
        for (&peers, &threads, settings, 1..) |*p, *t, mine, g| {
            const ends = try testLinks();
            f.links[g - 1] = ends[0];
            t.* = try std.Thread.spawn(.{}, Peer.run, .{ p, ends[1], @as(u32, @intCast(g)), mine });
            started += 1;
        }
        var why: [1024]u8 = undefined;
        var heard: Heard = .{};
        const said = f.exchange(0, served, @splat(0xd0), &why, &heard);
        f.close();
        for (threads) |t| t.join();
        started = 0;
        const want = [4]Digest{ @splat(0xd0), @splat(0x11), @splat(0x22), @splat(0x33) };
        try std.testing.expectEqual(want, heard.weights);
        for (peers, 1..) |p, g| {
            if (one_differs) {
                // rank 0 says the first follower's words and gives nothing back: each follower keeps its own digest alone
                try std.testing.expectEqual(g == 2, p.said.len > 0);
                try std.testing.expectEqual(@as(Digest, @splat(@intCast(0x10 * g + g))), p.heard.weights[g]);
                try std.testing.expectEqual(no_digest, p.heard.weights[0]);
            } else {
                try std.testing.expect(p.err == null);
                try std.testing.expectEqual(want, p.heard.weights);
            }
        }
        _ = try said;
    }
}

test "four ranks with the digests shared: the store's name computed from each rank's own heard digests is the same on all four" {
    const identity = @import("spill_identity.zig");
    const served: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1 };
    const common: identity.Common = .{
        .engine = "native-12f3f2c3b4bf",
        .model_revision = "1017",
        .config = @splat(1),
        .ablation = "keys@abc123",
        .engram_revision = "e1",
        .engram_json = @splat(2),
        .engram_switches = "packed=1",
        .kit = "kit-7",
        .token_map = "map-3",
        .driver = 580178,
        .capability = 121,
        .math = "pair_sum=1",
        .layout = @splat(3),
        .namespace_version = 2,
        .state_version = 2,
    };
    const Peer = struct {
        why: [1024]u8 = undefined,
        heard: Heard = .{},
        err: ?anyerror = null,
        fn run(p: *@This(), end: Link, rank: u32, share: bool) void {
            var f: Fan = .{ .share_digests = share };
            f.links[0] = end;
            defer f.close();
            _ = f.exchange(rank, served, @splat(@intCast(0x10 * rank + rank)), &p.why, &p.heard) catch |e| {
                p.err = e;
            };
        }
    };
    var peers: [3]Peer = .{ .{}, .{}, .{} };
    var threads: [3]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |t| t.join();
    var f: Fan = .{ .share_digests = true };
    defer f.close();
    for (&peers, &threads, 1..) |*p, *t, g| {
        const ends = try testLinks();
        f.links[g - 1] = ends[0];
        t.* = try std.Thread.spawn(.{}, Peer.run, .{ p, ends[1], @as(u32, @intCast(g)), true });
        started += 1;
    }
    var why: [1024]u8 = undefined;
    var heard: Heard = .{};
    _ = try f.exchange(0, served, @splat(0xd0), &why, &heard);
    f.close();
    for (threads) |t| t.join();
    started = 0;
    var missing: identity.Missing = .{};
    const lead_name = try identity.name(common, &heard.weights, &missing);
    for (peers) |p| {
        try std.testing.expect(p.err == null);
        // every follower names its store from what it heard, and gets rank 0's name
        const mine = try identity.name(common, &p.heard.weights, &missing);
        try std.testing.expectEqual(lead_name, mine);
    }
}

test "a follower runs rank 0's TF_DS_CUDA_SCHED, or says what differs" {
    const lead: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 1, .sched = cuda.context.sched_blocking };
    var buf: [512]u8 = undefined;
    try std.testing.expectEqualStrings("", lead.differ(lead, 1, &buf));

    // TF_DS_CUDA_SCHED: a valid but different name on one rank
    var mine = lead;
    mine.sched = cuda.context.sched_spin;
    try std.testing.expectEqualStrings("rank 1 has TF_DS_CUDA_SCHED spin and rank 0 blocking: set the same on every rank", mine.differ(lead, 1, &buf));

    // TF_DS_CUDA_SCHED: a text this build has no name for on one rank
    mine = lead;
    mine.sched = cuda.context.sched_invalid;
    try std.testing.expectEqualStrings("rank 2 has TF_DS_CUDA_SCHED text that is not one of auto, spin, yield, blocking and rank 0 blocking: set the same on every rank", mine.differ(lead, 2, &buf));

    // TF_DS_CUDA_SCHED: both ranks invalid: refused even though they agree
    var both = mine;
    try std.testing.expectEqualStrings("rank 3 and rank 0 have TF_DS_CUDA_SCHED text that is not one of auto, spin, yield, blocking: correct it on every rank", both.differ(mine, 3, &buf));

    // TF_DS_CUDA_SCHED: the primary context was already active with other flags on one rank
    mine = lead;
    mine.sched = cuda.context.sched_active_mismatch;
    try std.testing.expectEqualStrings("rank 1 has TF_DS_CUDA_SCHED a request the already-active primary context did not match and rank 0 blocking: set the same on every rank", mine.differ(lead, 1, &buf));

    // TF_DS_CUDA_SCHED: both ranks active-mismatched: refused even though they agree
    var both_active = mine;
    try std.testing.expectEqualStrings("rank 2 and rank 0 asked for TF_DS_CUDA_SCHED and the primary context was already active with different scheduling flags: release it or set the same on every rank", both_active.differ(mine, 2, &buf));

    // TF_DS_CUDA_SCHED: the set did not take effect on one rank
    mine = lead;
    mine.sched = cuda.context.sched_after_mismatch;
    try std.testing.expectEqualStrings("rank 1 has TF_DS_CUDA_SCHED a request the primary context did not take and rank 0 blocking: set the same on every rank", mine.differ(lead, 1, &buf));
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
    // the layout carries no padding: every byte of the frame is a field's, and the tag names this layout ("DS5b")
    comptime var sum: usize = 0;
    inline for (@typeInfo(Settings).@"struct".field_names) |name| sum += @sizeOf(@FieldType(Settings, name));
    try std.testing.expectEqual(sum, @sizeOf(Settings));
    try std.testing.expectEqual(@as(u32, 0x62355344), Settings.tag);
    try std.testing.expectEqualSlices(u8, "DS5b", std.mem.asBytes(&Settings.tag));
    try std.testing.expectEqual(@as(u32, 11), frame_protocol);
}

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
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(Settings));
}

test "rank 0's landing slots travel to every follower and are never compared (a follower holds no tier settings)" {
    const lead: Settings = .{ .world = 4, .streams = 8, .round_rows = 16, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 20, .replay_floor = 0, .idx_rb = 1, .land_slots = 2 };
    var mine = lead;
    mine.land_slots = 0;
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("", mine.differ(lead, 1, &buf));
    try std.testing.expectEqual(@as(u32, 11), frame_protocol); // kind 16 (cfg) after kind 15 (snap_sync), the Rocinante-only Settings; 11: fill_pair carries the third pass's rows
    var older = mine;
    older.protocol = 5; // a follower whose image lacks fill_pair
    try std.testing.expectEqualStrings("rank 2 decodes frame protocol 5 and rank 0 sends 11: run every rank from the same image", older.differ(lead, 2, &buf));
}

test "the expert pack as codes needs the checkpoint's own weights, the parity rest blocks and whole 128-value blocks" {
    try std.testing.expectEqual(@as(?[]const u8, null), packCodesOff(true, 512, 128));
    try std.testing.expectEqualStrings("needs the checkpoint's own FP8 / MXFP4 weights", packCodesOff(false, 512, 128).?);
    try std.testing.expectEqualStrings("needs TF_DS_2D_GU=parity (each slot's rest block in the pack)", packCodesOff(true, 512, 0).?);
    try std.testing.expectEqualStrings("needs a pack of whole 128-value blocks (its scales move in 16-byte units)", packCodesOff(true, 576, 128).?);
    try std.testing.expectEqualStrings("needs a pack of whole 128-value blocks (its scales move in 16-byte units)", packCodesOff(true, 512, 96).?);
}

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
}

test "quartersDecode sends each block's own rows of src at its place in the whole exchange, over the ring alone" {
    const src = @embedFile("prompt2d.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn quartersDecode(").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const blocks = try decodeBlocks(n, @max(w[0], w[1]) * esize, rs.slotBytes(), &bb);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const sent = try ringed(rs.quartersRows(e.s, src + b[0] * w[t.p] * esize, dst, n, b[0], b[1], w, esize));"));
    // no fallback: a refused block ends the exchange by name
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "error.RingRefusedBlock"));
    const q = std.mem.indexOf(u8, src, "    pub fn quarters(").?;
    const qbody = src[q .. q + std.mem.indexOf(u8, src[q..], "\n    }\n").?];
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
}

test "the exchange switches that remain: a named bit each in Settings.exch, asked as \"1\" only; the retired bits are reserved and a rank that sets one is named as one this build has no name for" {
    const kept = [_]struct { name: []const u8, bit: u32, var_ptr: *bool }{
        .{ .name = "TF_DS_ROCI_P2_FIRST", .bit = exch_roci_p2_first, .var_ptr = &roci_p2_first_asked },
        .{ .name = "TF_DS_ROCI_ENGRAM_AHEAD", .bit = exch_roci_engram_ahead, .var_ptr = &roci_engram_ahead_asked },
        .{ .name = "TF_DS_ROCI_SENT_FIRST", .bit = exch_roci_sent_first, .var_ptr = &roci_sent_first_asked },
        .{ .name = "TF_DS_ROCI_COMBINE_DOOR", .bit = exch_roci_combine_door, .var_ptr = &roci_combine_door_asked },
        .{ .name = "TF_DS_ROCI_STAGGER", .bit = exch_roci_stagger, .var_ptr = &roci_stagger_asked },
        .{ .name = "TF_DS_ROCI_STAGGER_LANES", .bit = exch_roci_stagger_lanes, .var_ptr = &roci_stagger3_asked },
        .{ .name = "TF_DS_PACK_EPILOGUE", .bit = exch_pack_epilogue, .var_ptr = &pack_epilogue_asked },
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
    odd.protocol = 9; // the layout before this one
    try std.testing.expectEqualStrings("rank 1 decodes frame protocol 9 and rank 0 sends 11: run every rank from the same image", odd.differ(lead, 1, &buf));
}

test "a default-on switch (TF_DS_ROCI_WAIT, TF_DS_RING_WAIT): unset and 1 are on; 0, empty and any other text are off" {
    try std.testing.expect(switchDefaultOn(null));
    try std.testing.expect(switchDefaultOn("1"));
    for ([_][]const u8{ "0", "", "yes", "true", " 1", "11" }) |t| try std.testing.expect(!switchDefaultOn(t));
    // the other switches are unchanged: off unless 1
    try std.testing.expect(!switchAsked(null));
    try std.testing.expect(switchAsked("1"));
}
test "TF_DS_DRAFT_VOCAB: kern2 bit 1: a follower differing only in that bit is refused and the switch named; no bit of either word is two switches'" {
    const lead: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 2 };
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqual(@as(u32, 1 << 1), kern2_draft_vocab);
    try std.testing.expectEqualStrings("TF_DS_DRAFT_VOCAB", kern2_names[@ctz(kern2_draft_vocab)]);
    var dv = lead;
    dv.kern2 = kern2_draft_vocab;
    try std.testing.expectEqualStrings("", dv.differ(dv, 1, &buf));
    try std.testing.expectEqualStrings("rank 1 has TF_DS_DRAFT_VOCAB=1 and rank 0 0: set the same on every rank", dv.differ(lead, 1, &buf));
    try std.testing.expectEqualStrings("rank 3 has TF_DS_DRAFT_VOCAB=0 and rank 0 1: set the same on every rank", lead.differ(dv, 3, &buf));
    // alone in kern2: HC_2S's bit does not move and the first word is untouched
    try std.testing.expectEqual(@as(u32, 0), dv.kern);
    try std.testing.expectEqual(@as(u32, 0), kern2_draft_vocab & kern2_hc_2s);
    // Model.open's reading goes to kern2Asked, never to kernAsked
    const was = draft_vocab_asked;
    defer draft_vocab_asked = was;
    draft_vocab_asked = true;
    try std.testing.expectEqual(kern2_draft_vocab, kern2Asked() & kern2_draft_vocab);
    try std.testing.expectEqual(@as(u32, 0), kernAsked() & kern2_draft_vocab & 0);
    draft_vocab_asked = false;
    try std.testing.expectEqual(@as(u32, 0), kern2Asked() & kern2_draft_vocab);
    // every named bit of both words is one switch's, and no switch is in both
    for (kern_names) |a| if (a.len > 0) for (kern2_names) |b| try std.testing.expect(!std.mem.eql(u8, a, b));
    for (kern2_names, 0..) |a, i| for (kern2_names[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a, b));
}

test "the second kernel word: a follower differing only in kern2 is refused, and the switch named" {
    const lead: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .graphs = 1, .graph_cap = 128, .pool = 1 << 24, .replay_floor = 1, .idx_rb = 2 };
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("", lead.differ(lead, 1, &buf));
    var hc = lead;
    hc.kern2 = kern2_hc_2s;
    try std.testing.expectEqualStrings("rank 3 has TF_DS_HC_2S=1 and rank 0 0: set the same on every rank", hc.differ(lead, 3, &buf));
    try std.testing.expectEqualStrings("rank 2 has TF_DS_HC_2S=0 and rank 0 1: set the same on every rank", lead.differ(hc, 2, &buf));
    // a bit this build has no name for is refused too
    var other = lead;
    other.kern2 = 1 << 5;
    try std.testing.expectEqualStrings("rank 1 and rank 0 differ in kernel (second word) switches this build has no name for (bits 0x20): run every rank from the same image", other.differ(lead, 1, &buf));
    // kern2 is its own word: kern's bit 31 (the canary) and HC_2S never meet
    try std.testing.expectEqual(@as(u32, 1), kern2_hc_2s);
    try std.testing.expectEqualStrings("TF_DS_HC_2S", kern2_names[@ctz(kern2_hc_2s)]);
    try std.testing.expectEqual(@as(u32, 0), kernAsked() & kern2_hc_2s & 0); // (kernAsked has no HC_2S bit)
    const was = hc_2s_asked;
    defer hc_2s_asked = was;
    hc_2s_asked = true;
    try std.testing.expectEqual(kern2_hc_2s, kern2Asked());
    hc_2s_asked = false;
    try std.testing.expectEqual(@as(u32, 0), kern2Asked());
    // no two switches of kern2 share a bit, and every bit kern2_names names is one of them
    const all2 = [_]u32{ kern2_hc_2s, kern2_draft_vocab };
    var seen2: u32 = 0;
    for (all2) |v| {
        try std.testing.expect(@popCount(v) == 1 and seen2 & v == 0);
        seen2 |= v;
    }
    for (kern2_names, 0..) |n, i| if (n.len > 0) try std.testing.expect(seen2 & (@as(u32, 1) << @intCast(i)) != 0);
}

    const all = [_]u32{ kern_woa_w8, kern_quant_once, kern_pf_gemm5, kern_group_tiles, kern_topk_tiles, kern_idx_proj_owned, kern_bf16_rows, kern_pf_run128, kern_decode_fuse, kern_draft_4way, kern_head_pack12, kern_hc_late_fork, kern_attn_u, kern_idx_u, kern_idx_v, kern_attn_u_decode, kern_assemble_quant, kern_hc_stride, kern_draft_main_4way, kern_mma_sf, kern_attn_epilogue, kern_mma_sf_exact, kern_pass_tail, kern_head_cand, kern_head_key, kern_decode_ksplit, kern_attn_epi_fused, kern_gemm_w16, kern_hc_fused, kern_gemm_r2, kern_gemm_halves, kern_qgate_canary };

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
