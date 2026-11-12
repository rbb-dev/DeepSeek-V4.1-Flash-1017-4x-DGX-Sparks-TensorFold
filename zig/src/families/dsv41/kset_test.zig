//! The kit's Triton variants against the launches the any-row-count plan (slab_plan.zig) makes: for every chunk of 1 to
//! 4,096 rows the encoder half's slabs, and for every window of 1 to 128 decoder rows the decoder half's, each launch the
//! owned path makes at a slab's rows is looked up in the kit's aot.json the way a launch finds its variant. The kit
//! directory is TF_KSET_AOT_DIR (aot.json inside); without it the test is skipped.

const std = @import("std");
const aot = @import("cuda").aot;
const slab_plan = @import("slab_plan.zig");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const config_mod = @import("config.zig");
const tri = @import("tri.zig");
const tri_hc = @import("tri_hc.zig");
const tri_basic = @import("tri_basic.zig");
const any_check = @import("any_check.zig");

const QRows = prompt.QRows;

const ManifestFinds = struct {
    m: *const aot.Manifest,
    reject: ?[]const u8 = null,
    looked: ?*usize = null,

    pub fn has(f: ManifestFinds, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) bool {
        if (f.looked) |c| c.* += 1;
        if (f.reject) |r| if (std.mem.eql(u8, r, name)) return false;
        return f.m.find(name, args, consts) != null;
    }

    /// The base set's manifest alone: no extra set (a runTwin launch is then found as its base fallback, as Tri runs it).
    pub fn hasExtra(f: ManifestFinds, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) bool {
        _ = .{ f, name, args, consts };
        return false;
    }
};

/// The kit's base set and its extra set, looked up as a launch finds its variant (Set.find; the two-block readers Set.lookup of the
/// extra set, which Manifest.find mirrors).
const KitFinds = struct {
    base: ManifestFinds,
    extra: *const aot.Manifest,
    reject_extra: ?[]const u8 = null,

    pub fn has(f: KitFinds, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) bool {
        return f.base.has(name, args, consts);
    }

    pub fn hasExtra(f: KitFinds, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) bool {
        if (f.base.looked) |c| c.* += 1;
        if (f.reject_extra) |r| if (std.mem.eql(u8, r, name)) return false;
        return f.extra.find(name, args, consts) != null;
    }
};

fn kitManifest(gpa: std.mem.Allocator) !aot.Manifest {
    const dir = std.c.getenv("TF_KSET_AOT_DIR") orelse return error.SkipZigTest;
    return aot.Manifest.read(gpa, std.testing.io, std.mem.span(dir)) catch error.SkipZigTest;
}

fn layers(lays: *[40]weights.Layer, cfg: config_mod.Config) void {
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

/// The mixes' launches on `r`'s rows (attnMixes / ffnMixes / the layers' hc_post), as the owned path makes them.
fn mixes(t: tri.Tri, cfg: config_mod.Config, lay: weights.Layer, r: QRows) !void {
    const row_h: u64 = cfg.hc * cfg.hidden * 2;
    const row_p: u64 = cfg.hc * 4;
    const row_c: u64 = cfg.hc * cfg.hc * 4;
    const base_h: u64 = 1 << 32;
    const alt_h: u64 = 2 << 32;
    for ([_][3]u64{ lay.hc_attn, lay.hc_ffn }) |f| {
        _ = try tri_hc.hcPrePf(t, base_h + r.start * row_h, f[0], f[1], f[2], (3 << 32) + r.start * row_p, lay.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, (4 << 32) + r.start * cfg.hidden * 2, (5 << 32) + r.start * row_p, (6 << 32) + r.start * row_p, (7 << 32) + r.start * row_c, 8 << 32, .{ .gathered = 9 << 32, .world = 2, .h_out = alt_h + r.start * row_h }, r.n, cfg.hidden);
        try tri_basic.hcPre(t, base_h + r.start * row_h, f[0], f[1], f[2], (3 << 32) + r.start * row_p, lay.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, (4 << 32) + r.start * cfg.hidden * 2, (5 << 32) + r.start * row_p, (6 << 32) + r.start * row_p, (7 << 32) + r.start * row_c, 8 << 32, r.n, cfg.hidden);
    }
    try tri_basic.hcPost(t, 9 << 32, base_h + r.start * row_h, (6 << 32) + r.start * row_p, (7 << 32) + r.start * row_c, base_h + r.start * row_h, 2, r.n, cfg.hidden);
}

/// Every launch of one slab on layers `lays[0..half]`, as a reason (null: all found).
fn slabOff(gpa: std.mem.Allocator, finds: anytype, cfg: config_mod.Config, lays: []const weights.Layer, half: usize, encoder: bool, r: QRows) !?[]const u8 {
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
    for (lays[0..@min(half, lays.len)]) |lay| try mixes(t, cfg, lay, r);
    for (log.items.items) |l| if (!l.foundIn(finds)) return l.name;
    return null;
}

const Miss = struct { n_k: usize, why: []const u8 };

/// The distinct slab row counts of chunks of 1..max rows, each with its first start; the starts are all whole blocks.
fn distinctSlabs(gpa: std.mem.Allocator, max: usize) ![]QRows {
    var seen = try gpa.alloc(bool, max + 1);
    defer gpa.free(seen);
    @memset(seen, false);
    var out: std.ArrayList(QRows) = .empty;
    var rows: usize = 1;
    while (rows <= max) : (rows += 1) for (slab_plan.slabs(rows)) |s| {
        try std.testing.expectEqual(@as(usize, 0), s.start % slab_plan.block);
        if (s.n == 0 or seen[s.n]) continue;
        seen[s.n] = true;
        try out.append(gpa, .{ .start = s.start, .n = s.n });
    };
    return out.toOwnedSlice(gpa);
}

fn sweep(gpa: std.mem.Allocator, finds: anytype, cfg: config_mod.Config, lays: []const weights.Layer, half: usize, encoder: bool, slabs: []const QRows, miss: *std.ArrayList(Miss)) !void {
    for (slabs) |r| if (try slabOff(gpa, finds, cfg, lays, half, encoder, r)) |why| try miss.append(gpa, .{ .n_k = r.n, .why = why });
}

fn report(what: []const u8, miss: []const Miss) void {
    if (miss.len == 0) return;
    std.debug.print("T-KSET {s}: {d} slab row counts with no variant. first: ", .{ what, miss.len });
    for (miss[0..@min(miss.len, 24)]) |m| std.debug.print("{d} ({s}) ", .{ m.n_k, m.why });
    std.debug.print("\n", .{});
}

test "T-KSET: the kit's variants cover every launch of every slab of 1..4,096 encoder rows and 1..128 decoder rows" {
    const gpa = std.testing.allocator;
    var man = try kitManifest(gpa);
    defer man.deinit();
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    layers(&lays, cfg);
    var looked: usize = 0;
    const finds: ManifestFinds = .{ .m = &man, .looked = &looked };
    const enc = try distinctSlabs(gpa, 4096);
    defer gpa.free(enc);
    const dec = try distinctSlabs(gpa, 128);
    defer gpa.free(dec);
    var miss_e: std.ArrayList(Miss) = .empty;
    defer miss_e.deinit(gpa);
    var miss_d: std.ArrayList(Miss) = .empty;
    defer miss_d.deinit(gpa);
    try sweep(gpa, finds, cfg, &lays, 20, true, enc, &miss_e);
    try sweep(gpa, finds, cfg, lays[20..], 20, false, dec, &miss_d);
    std.debug.print("T-KSET: {d} distinct encoder slab row counts, {d} decoder, {d} launches looked up in the kit's {d} variants\n", .{ enc.len, dec.len, looked, man.parsed.value.kernels.len });
    try std.testing.expect(looked > 50_000);
    report("encoder half (layers 0-19)", miss_e.items);
    report("decoder half (layers 20-39)", miss_d.items);
    try std.testing.expectEqual(@as(usize, 0), miss_e.items.len + miss_d.items.len);
}

test "T-KSET control: a kit with a variant missing is refused, naming it" {
    const gpa = std.testing.allocator;
    var man = try kitManifest(gpa);
    defer man.deinit();
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    layers(&lays, cfg);
    for ([_][]const u8{ "_rmsnorm", "_route", "_engram_gate", "_rope_heads", "_hc_mix_pf", "_hc_post" }) |name| {
        const finds: ManifestFinds = .{ .m = &man, .reject = name };
        const w = try slabOff(gpa, finds, cfg, &lays, 20, true, .{ .start = 0, .n = 512 });
        try std.testing.expect(w != null);
    }
    const w = try slabOff(gpa, ManifestFinds{ .m = &man }, cfg, &lays, 20, true, .{ .start = 0, .n = 512 });
    try std.testing.expectEqual(@as(?[]const u8, null), w);
}

test "T-KSET control: the same kit without the _hc_mix_pf variant of the 5-row block's variant misses exactly the 5-row slab" {
    const gpa = std.testing.allocator;
    var man = try kitManifest(gpa);
    defer man.deinit();
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    layers(&lays, cfg);
    const all = man.parsed.value.kernels;
    var keep: std.ArrayList(@TypeOf(all[0])) = .empty;
    defer keep.deinit(gpa);
    var removed: usize = 0;
    for (all) |k| {
        if (std.mem.eql(u8, k.@"fn", "_hc_mix_pf")) if (k.consts.map.get("RB")) |c| if (c.int != null and c.int.? == 5) {
            removed += 1;
            continue;
        };
        try keep.append(gpa, k);
    }
    try std.testing.expect(removed > 0);
    man.parsed.value.kernels = keep.items;
    const enc = try distinctSlabs(gpa, 4096);
    defer gpa.free(enc);
    var miss: std.ArrayList(Miss) = .empty;
    defer miss.deinit(gpa);
    try sweep(gpa, ManifestFinds{ .m = &man }, cfg, &lays, 20, true, enc, &miss);
    try std.testing.expect(miss.items.len > 0);
    // RB is min(8, rows): only a slab of exactly 5 rows launches the RB = 5 variant
    for (miss.items) |m| try std.testing.expectEqual(@as(usize, 5), m.n_k);
    try std.testing.expectEqual(@as(usize, 1), miss.items.len);
}

test "T-KSET, the two-block readers: the kit's extra set has the reader variant of every launch of every slab of 1..1,024 rows" {
    const gpa = std.testing.allocator;
    const dir = std.c.getenv("TF_KSET_EXTRA_DIR") orelse return error.SkipZigTest;
    var man = aot.Manifest.read(gpa, std.testing.io, std.mem.span(dir)) catch return error.SkipZigTest;
    defer man.deinit();
    var rows: [1024]usize = undefined;
    for (&rows, 1..) |*r, i| r.* = i;
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    const reason = try prompt.rociReadersOff(gpa, ManifestFinds{ .m = &man }, &rows, cfg.hidden, 12800, cfg.eps, cfg.hc_eps, cfg.hc_iters);
    if (reason) |r| std.debug.print("T-KSET two-block readers: {s}\n", .{r});
    try std.testing.expectEqual(@as(?[]const u8, null), reason);
    // control: the extra set without _hc_post_2b is refused
    try std.testing.expect((try prompt.rociReadersOff(gpa, ManifestFinds{ .m = &man, .reject = "_hc_post_2b" }, &rows, cfg.hidden, 12800, cfg.eps, cfg.hc_eps, cfg.hc_iters)) != null);
}

test "T-KSET, the decoder half on owned rows: any_check.decoderOff finds every launch of layers 0-39 at every slab of a decoder window of 1..128 rows in the kit's base and extra sets" {
    const gpa = std.testing.allocator;
    var man = try kitManifest(gpa);
    defer man.deinit();
    const dir = std.c.getenv("TF_KSET_EXTRA_DIR") orelse return error.SkipZigTest;
    var extra = aot.Manifest.read(gpa, std.testing.io, std.mem.span(dir)) catch return error.SkipZigTest;
    defer extra.deinit();
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    any_check.layers(&lays, cfg);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var looked: usize = 0;
    const finds: KitFinds = .{ .base = .{ .m = &man, .looked = &looked }, .extra = &extra };
    const dec = try any_check.decoderOff(gpa, arena.allocator(), finds, cfg, &lays);
    // the encoder half's (TF_DS_ROCI_ANY's check) with the two-block readers looked up too
    const enc = try any_check.anyOwnedOff(gpa, arena.allocator(), finds, cfg, &lays, 20, true, 4096);
    std.debug.print("T-KSET decoder half: {d} launches looked up in the kit's {d} base and {d} extra variants; decoder: {s}; encoder with the readers: {s}\n", .{ looked, man.parsed.value.kernels.len, extra.parsed.value.kernels.len, dec orelse "every variant found", enc orelse "every variant found" });
    try std.testing.expectEqual(@as(?[]const u8, null), dec);
    try std.testing.expectEqual(@as(?[]const u8, null), enc);
    // control: the extra set without the two-block post refuses the decoder half, naming it
    const no_post: KitFinds = .{ .base = .{ .m = &man }, .extra = &extra, .reject_extra = "_hc_post_2b" };
    const w = (try any_check.decoderOff(gpa, arena.allocator(), no_post, cfg, &lays)).?;
    try std.testing.expect(std.mem.indexOf(u8, w, "_hc_post_2b") != null);
}

/// The kit's variants for every launch of a pass of `pass_rows` rows (a quarter of them a slab, `widest` rows): the base set's for the
/// encoder half, the extra set's two-block readers of the pass's one full slab, and the whole any-row-count check the owned path runs at start.
fn ksetAtPassRows(pass_rows: usize, widest_want: usize) !void {
    const gpa = std.testing.allocator;
    var man = try kitManifest(gpa);
    defer man.deinit();
    var why: config_mod.Why = .{};
    const cfg = try config_mod.parse(gpa, config_mod.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    layers(&lays, cfg);
    var looked: usize = 0;
    const finds: ManifestFinds = .{ .m = &man, .looked = &looked };
    const enc = try distinctSlabs(gpa, pass_rows);
    defer gpa.free(enc);
    var miss: std.ArrayList(Miss) = .empty;
    defer miss.deinit(gpa);
    try sweep(gpa, finds, cfg, &lays, 20, true, enc, &miss);
    var widest: usize = 0;
    for (enc) |r| widest = @max(widest, r.n);
    std.debug.print("T-KSET {d}: {d} distinct encoder slab row counts (the widest {d} rows), {d} launches looked up in the kit's {d} base variants\n", .{ pass_rows, enc.len, widest, looked, man.parsed.value.kernels.len });
    try std.testing.expectEqual(widest_want, widest);
    report("encoder half of a pass (layers 0-19)", miss.items);
    try std.testing.expectEqual(@as(usize, 0), miss.items.len);
    // the readers (TF_DS_ROCI's start check looks up the pass's one full slab: the pass rows / 4)
    const dir = std.c.getenv("TF_KSET_EXTRA_DIR") orelse return;
    var extra = aot.Manifest.read(gpa, std.testing.io, std.mem.span(dir)) catch return;
    defer extra.deinit();
    const reason = try prompt.rociReadersOff(gpa, ManifestFinds{ .m = &extra }, &[_]usize{pass_rows / 4}, cfg.hidden, 12800, cfg.eps, cfg.hc_eps, cfg.hc_iters);
    if (reason) |r| std.debug.print("T-KSET {d} two-block readers: {s}\n", .{ pass_rows, r });
    try std.testing.expectEqual(@as(?[]const u8, null), reason);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const kit: KitFinds = .{ .base = .{ .m = &man }, .extra = &extra };
    any_check.layers(&lays, cfg);
    const any = try any_check.anyOwnedOff(gpa, arena.allocator(), kit, cfg, &lays, 20, true, pass_rows);
    if (any) |r| std.debug.print("T-KSET {d} any_check: {s}\n", .{ pass_rows, r });
    try std.testing.expectEqual(@as(?[]const u8, null), any);
}

test "T-KSET at 8,192 rows: the kit's base set has every launch of every slab of 1..8,192 encoder rows (slabs to 2,048), and the extra set the two-block readers of the 2,048-row slab" {
    try ksetAtPassRows(8192, 2048);
}

test "T-KSET at 16,384 rows: the kit's base set has every launch of every slab of 1..16,384 encoder rows (slabs to 4,096), and the extra set the two-block readers of the 4,096-row slab" {
    try ksetAtPassRows(16384, 4096);
}
