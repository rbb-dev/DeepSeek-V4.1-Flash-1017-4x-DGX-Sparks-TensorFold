const any_check = @import("any_check.zig");
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

    if (try prompt.frontOwnedOffAt(gpa, finds, true, lays, half, cfg.eps, &one)) |w| return w;
    if (try prompt.ffnCodesOffAt(gpa, finds, true, lays, half, cfg.experts, cfg.top_k, cfg.routed_scaling, cfg.top_k + 1, &one)) |w| return w;
        if (try prompt.engramOwnerSumOffAt(gpa, finds, .{ 12800, 12800 }, lays, half, cfg.eps, cfg.hidden, cfg.hc, &one)) |w| return w;

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