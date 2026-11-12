//! The start check of the owned path at any row count (TF_DS_ROCI_ANY): every launch the encoder half makes at a slab of any chunk of
//! 1 to 4,096 rows must have a variant in this rank's kernel set, looked up the way a launch finds it. A missing one refuses the switch for
//! everyone, naming the launch and the slab. The decoder half's (decoderOff): every launch of every layer at a slab of a decoder window,
//! 1 row to the window's.

const std = @import("std");
const slab_plan = @import("slab_plan.zig");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const config_mod = @import("config.zig");
const tri = @import("tri.zig");
const tri_hc = @import("tri_hc.zig");
const tri_basic = @import("tri_basic.zig");

const QRows = prompt.QRows;

pub fn layers(lays: *[40]weights.Layer, cfg: config_mod.Config) void {
    const one: weights.Linear = .{ .words = 0, .k = 1, .n = 1 };
    for (lays, 0..) |*l, i| {
        l.* = .{ .idx = @intCast(i), .ratio = cfg.ratios[i] };
        l.comp_wkv = if (cfg.kv_sources.has(i)) one else null;
        l.idx_wq_b = if (cfg.index_sources.has(i)) one else null;
        l.engram_wkv = if (cfg.engram_layers.has(i)) one else null;
        l.engram_qk = 0x9000 + i * 0x1000;
        l.wq_a = .{ .words = 0, .k = 5120, .n = 1280, .fmt = .fp8 };
        l.wkv = .{ .words = 0, .k = 5120, .n = 512, .fmt = .fp8 };
        l.q_norm = 0x7000 + i * 0x1000;
        l.hc_attn = .{ 0x10000, 0x10100, 0x10200 };
        l.hc_ffn = .{ 0x11000, 0x11100, 0x11200 };
        l.attn_norm = 0x12000;
        l.ffn_norm = 0x13000;
        l.gate_b = 0x14000;
        l.gate_b_vl = 0x15000;
        l.gate_w = 0x16000;
    }
}

/// The mixes' launches on `r`'s rows (attnMixes / ffnMixes / the layers' hc_post), as the owned path makes them: the sums posted from
/// the fp32 planes, and from the transport's two blocks of a slab plan of stride `q` rows (the kit's extra set's two-block readers).
pub fn mixes(t: tri.Tri, cfg: config_mod.Config, lay: weights.Layer, r: QRows, q: usize) !void {
    const row_h: u64 = cfg.hc * cfg.hidden * 2;
    const row_p: u64 = cfg.hc * 4;
    const row_c: u64 = cfg.hc * cfg.hc * 4;
    const base_h: u64 = 1 << 32;
    const alt_h: u64 = 2 << 32;
    const blk: u64 = 10 << 32;
    const blk_rs = q * (cfg.hidden / 2);
    for ([_][3]u64{ lay.hc_attn, lay.hc_ffn }) |f| {
        _ = try tri_hc.hcPrePf(t, base_h + r.start * row_h, f[0], f[1], f[2], (3 << 32) + r.start * row_p, lay.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, (4 << 32) + r.start * cfg.hidden * 2, (5 << 32) + r.start * row_p, (6 << 32) + r.start * row_p, (7 << 32) + r.start * row_c, 8 << 32, .{ .gathered = 9 << 32, .world = 2, .h_out = alt_h + r.start * row_h }, r.n, cfg.hidden);
        _ = try tri_hc.hcPrePf(t, base_h + r.start * row_h, f[0], f[1], f[2], (3 << 32) + r.start * row_p, lay.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, (4 << 32) + r.start * cfg.hidden * 2, (5 << 32) + r.start * row_p, (6 << 32) + r.start * row_p, (7 << 32) + r.start * row_c, 8 << 32, .{ .gathered = blk, .world = 2, .h_out = alt_h + r.start * row_h, .blk_rs = blk_rs }, r.n, cfg.hidden);
        try tri_basic.hcPre(t, base_h + r.start * row_h, f[0], f[1], f[2], (3 << 32) + r.start * row_p, lay.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, (4 << 32) + r.start * cfg.hidden * 2, (5 << 32) + r.start * row_p, (6 << 32) + r.start * row_p, (7 << 32) + r.start * row_c, 8 << 32, r.n, cfg.hidden);
    }
    try tri_basic.hcPost(t, 9 << 32, base_h + r.start * row_h, (6 << 32) + r.start * row_p, (7 << 32) + r.start * row_c, base_h + r.start * row_h, 2, r.n, cfg.hidden);
    try tri_basic.hcPost2b(t, blk, blk_rs, base_h + r.start * row_h, (6 << 32) + r.start * row_p, (7 << 32) + r.start * row_c, base_h + r.start * row_h, r.n, cfg.hidden);
}

/// Whether `finds` has the variant a logged launch looks up: `finds.has` for the main set's launches, `finds.hasExtra` for the extra set's
/// (the two-block readers). Every lookup declares both, publicly (a lookup of the main set alone says what its hasExtra answers); there is
/// no dispatch on whether one exists (@hasDecl does not see a private method, and would skip it silently).
fn found(finds: anytype, l: tri.Log.Kept) bool {
}

/// Every launch of one slab `r` (of a plan of stride `q` rows) on layers `lays[0..half]`, as a reason (null: all found). `encoder`: the
/// encoder half's launches as well (the Engram sums and gate, the indexer's owned projections), which the decoder half has none of.
pub fn slabOff(gpa: std.mem.Allocator, finds: anytype, cfg: config_mod.Config, lays: []const weights.Layer, half: usize, encoder: bool, r: QRows, q: usize) !?[]const u8 {
    const one = [_]QRows{r};
    if (try prompt.frontOwnedOffAt(gpa, finds, true, lays, half, cfg.eps, &one)) |w| return w;
    if (try prompt.ffnCodesOffAt(gpa, finds, true, lays, half, cfg.experts, cfg.top_k, cfg.routed_scaling, cfg.top_k + 1, &one)) |w| return w;
    if (encoder) {
        if (try prompt.engramOwnerSumOffAt(gpa, finds, .{ 12800, 12800 }, lays, half, cfg.eps, cfg.hidden, cfg.hc, &one)) |w| return w;
        if (try prompt.idxProjOwnedOffAt(gpa, finds, true, true, lays, cfg.candidate_source, cfg.rope_dim, cfg.index_heads, cfg.index_head_dim, &one)) |w| return w;
    }
    var log: tri.Log = .{ .gpa = gpa };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log };
    for (lays[0..@min(half, lays.len)]) |lay| try mixes(t, cfg, lay, r, q);
    for (log.items.items) |l| if (!found(finds, l)) return l.name;
    return null;
}

/// The reason the owned path cannot run at any row count on this rank (null: every launch of every slab of chunks of 1 to `max_rows`
/// rows on layers `lays[0..half]` has a variant), allocated from `a`: which launch, at how many rows, in which chunk first met.
/// `finds.has(name, args, consts)` is the engine's lookup (`finds.hasExtra`, when it has one, the extra set's). `encoder`: slabOff's.
pub fn anyOwnedOff(gpa: std.mem.Allocator, a: std.mem.Allocator, finds: anytype, cfg: config_mod.Config, lays: []const weights.Layer, half: usize, encoder: bool, max_rows: usize) !?[]const u8 {
    const seen = try gpa.alloc(bool, max_rows + 1);
    defer gpa.free(seen);
    @memset(seen, false);
    var rows: usize = 1;
    while (rows <= max_rows) : (rows += 1) for (slab_plan.slabs(rows)) |s| {
        if (s.n == 0 or seen[s.n]) continue;
        seen[s.n] = true;
        if (try slabOff(gpa, finds, cfg, lays, half, encoder, .{ .start = s.start, .n = s.n }, slab_plan.q(rows))) |w| {
            return try std.fmt.allocPrint(a, "no kernel variant for {s} at a slab of {d} rows (the first {s} with such a slab has {d} rows)", .{ w, s.n, if (encoder) "chunk" else "decoder window", rows });
        }
    };
    return null;
}

/// The decoder half's start check: the reason layers 20 to 39 cannot run on owned rows (null: they can), anyOwnedOff over every layer
/// with the decoder half's launches at every slab of a decoder window of 1 row to the window's (`cfg.window`: the rows a chunk's
/// decoder half runs at most, prompt.decoderRows).
pub fn decoderOff(gpa: std.mem.Allocator, a: std.mem.Allocator, finds: anytype, cfg: config_mod.Config, lays: []const weights.Layer) !?[]const u8 {
    return anyOwnedOff(gpa, a, finds, cfg, lays, lays.len, false, cfg.window);
}

const AcceptAll = struct {
    reject: ?[]const u8 = null,
    pub fn has(f: AcceptAll, name: []const u8, _: anytype, _: anytype) bool {
        if (f.reject) |r| return !std.mem.eql(u8, r, name);
        return true;
    }
    pub fn hasExtra(f: AcceptAll, name: []const u8, args: anytype, consts: anytype) bool {
        return f.has(name, args, consts);
    }
};

test "the start check at any row count passes a kernel set that has every variant and names the launch and the slab when one is missing" {
    const gpa = std.testing.allocator;
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    layers(&lays, cfg);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    try std.testing.expectEqual(@as(?[]const u8, null), try anyOwnedOff(gpa, arena.allocator(), AcceptAll{}, cfg, &lays, 20, true, 130));
    for ([_][]const u8{ "_rmsnorm", "_route", "_hc_mix_pf", "_hc_post" }) |name| {
        const w = (try anyOwnedOff(gpa, arena.allocator(), AcceptAll{ .reject = name }, cfg, &lays, 20, true, 130)).?;
        try std.testing.expect(std.mem.indexOf(u8, w, "slab of 1 rows") != null or std.mem.indexOf(u8, w, "slab of") != null);
        try std.testing.expect(std.mem.indexOf(u8, w, "no kernel variant for") != null);
    }
}

/// A kernel set without one launch: by name in the main set, by name in the extra set (the two-block readers), by name at one
/// row count (its `rows` argument), or the front's q RMSNorm of the layers from one on (each layer's norm weight lies at
/// 0x7000 + layer * 0x1000 in `layers`).
const Reject = struct {
    name: ?[]const u8 = null,
    extra_name: ?[]const u8 = null,
    at_rows: ?usize = null,
    norm_from: ?usize = null,

    fn rowsOf(args: anytype) ?usize {
        for (args) |x| if (std.mem.eql(u8, x.name, "rows")) return @intCast(x.value.i32);
        return null;
    }

    pub fn has(f: Reject, name: []const u8, args: anytype, _: anytype) bool {
        if (f.name) |n| if (std.mem.eql(u8, n, name) and (f.at_rows == null or rowsOf(args) == f.at_rows)) return false;
        if (f.norm_from) |l| if (std.mem.eql(u8, name, "_rmsnorm")) for (args) |x| if (std.mem.eql(u8, x.name, "W") and x.value.ptr.addr >= 0x7000 + l * 0x1000) return false;
        return true;
    }

    pub fn hasExtra(f: Reject, name: []const u8, args: anytype, _: anytype) bool {
        if (f.extra_name) |n| if (std.mem.eql(u8, n, name) and (f.at_rows == null or rowsOf(args) == f.at_rows)) return false;
        return true;
    }
};

fn testSetup(gpa: std.mem.Allocator, lays: *[40]weights.Layer) !config_mod.Config {
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    layers(lays, cfg);
    return cfg;
}

test "the decoder half's start check reaches layers 20 to 39, which the encoder half's leaves out" {
    const gpa = std.testing.allocator;
    var lays: [40]weights.Layer = undefined;
    const cfg = try testSetup(gpa, &lays);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const late = Reject{ .norm_from = 20 };
    // the encoder half's check (layers 0-19, chunks of 1 to 4,096 rows) never launches a layer from 20 on
    try std.testing.expectEqual(@as(?[]const u8, null), try anyOwnedOff(gpa, a, late, cfg, &lays, 20, true, 130));
    // the decoder half's (every layer, every slab of a decoder window of 1 to the window's rows) does, and names the window
    const w = (try decoderOff(gpa, a, late, cfg, &lays)).?;
    try std.testing.expect(std.mem.indexOf(u8, w, "_rmsnorm") != null);
    try std.testing.expect(std.mem.indexOf(u8, w, "decoder window") != null);
    try std.testing.expectEqual(@as(?[]const u8, null), try decoderOff(gpa, a, AcceptAll{}, cfg, &lays));
}

test "the decoder half's start check finds the two-block readers in the extra set, at every slab's row count" {
    const gpa = std.testing.allocator;
    var lays: [40]weights.Layer = undefined;
    const cfg = try testSetup(gpa, &lays);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "_hc_mix_pf_2b", "_hc_post_2b" }) |n| {
        const w = (try decoderOff(gpa, a, Reject{ .extra_name = n }, cfg, &lays)).?;
        try std.testing.expect(std.mem.indexOf(u8, w, n) != null);
    }
    // one row count missing: a slab of 7 rows (the first window with one: 7 rows, one slab)
    const w7 = (try decoderOff(gpa, a, Reject{ .extra_name = "_hc_mix_pf_2b", .at_rows = 7 }, cfg, &lays)).?;
    try std.testing.expect(std.mem.indexOf(u8, w7, "at a slab of 7 rows (the first decoder window with such a slab has 7 rows)") != null);
    // the main set's mixes at one row count
    const m5 = (try decoderOff(gpa, a, Reject{ .name = "_hc_mix_pf", .at_rows = 5 }, cfg, &lays)).?;
    try std.testing.expect(std.mem.indexOf(u8, m5, "at a slab of 5 rows") != null);
    // a lookup without an extra set (the engine's main set alone) passes the readers it cannot see
    try std.testing.expectEqual(@as(?[]const u8, null), try decoderOff(gpa, a, AcceptAll{}, cfg, &lays));
}

test "a decoder window's slabs: q(d) is 16 or 32 rows and every slab of 1 to 32 rows occurs" {
    var seen: [33]bool = @splat(false);
    for (1..129) |d| {
        const q = slab_plan.q(d);
        try std.testing.expect(q == 16 or q == 32);
        for (slab_plan.slabs(d)) |s| {
            try std.testing.expect(s.n <= 32);
            seen[s.n] = true;
        }
    }
    for (seen[1..]) |s| try std.testing.expect(s);
}

test "the start check asks the lookup's extra set for every two-block reader launch, on every lookup alike" {
    const Count = struct {
        main: *usize,
        extra: *usize,
        pub fn has(f: @This(), _: []const u8, _: anytype, _: anytype) bool {
            f.main.* += 1;
            return true;
        }
        pub fn hasExtra(f: @This(), name: []const u8, _: anytype, _: anytype) bool {
            f.extra.* += 1;
            return std.mem.endsWith(u8, name, "_2b");
        }
    };
    const gpa = std.testing.allocator;
    var lays: [40]weights.Layer = undefined;
    const cfg = try testSetup(gpa, &lays);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var m: usize = 0;
    var x: usize = 0;
    try std.testing.expectEqual(@as(?[]const u8, null), try anyOwnedOff(gpa, arena.allocator(), Count{ .main = &m, .extra = &x }, cfg, &lays, 20, true, 130));
    // every slab size's mixes ask the extra set for _hc_mix_pf_2b (twice a layer) and _hc_post_2b (once): 3 a layer
    try std.testing.expect(x > 0 and x % (3 * 20) == 0);
    try std.testing.expect(m > x);
}

test "TF_DS_ROCI_ANY's start check refuses a kit whose extra set lacks a two-block reader the encoder half's slabs launch, naming the reader and the slab" {
    const gpa = std.testing.allocator;
    var lays: [40]weights.Layer = undefined;
    const cfg = try testSetup(gpa, &lays);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "_hc_mix_pf_2b", "_hc_post_2b" }) |n| {
        const w = (try anyOwnedOff(gpa, a, Reject{ .extra_name = n }, cfg, &lays, 20, true, 4096)).?;
        try std.testing.expect(std.mem.indexOf(u8, w, n) != null);
        try std.testing.expect(std.mem.indexOf(u8, w, "the first chunk with such a slab") != null);
    }
    // one row count's variant missing: a chunk of 1,000 rows has slabs of 256, 256, 256 and 232 rows
    const w = (try anyOwnedOff(gpa, a, Reject{ .extra_name = "_hc_mix_pf_2b", .at_rows = 232 }, cfg, &lays, 20, true, 4096)).?;
    try std.testing.expect(std.mem.indexOf(u8, w, "at a slab of 232 rows") != null);
    // control: every reader there, nothing refused
    try std.testing.expectEqual(@as(?[]const u8, null), try anyOwnedOff(gpa, a, Reject{}, cfg, &lays, 20, true, 4096));
}