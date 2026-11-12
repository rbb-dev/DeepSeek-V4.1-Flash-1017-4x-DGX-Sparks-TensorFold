
test "retired names: none is also a row, each says what replaced it, and the words name the switch" {
    var buf: [400]u8 = undefined;
    try std.testing.expect(settings.retired.len > 0);
    for (&settings.retired) |*r| {
        try std.testing.expect(settings.byEnv(r.env) == null);
        try std.testing.expect(r.why.len > 0);
        const v: []const u8 = "1";
        const why = settings.retiredRefusal(r, v, &buf).?;
        try std.testing.expect(std.mem.startsWith(u8, why, r.env));
        try std.testing.expect(std.mem.indexOf(u8, why, " no longer exists: ") != null);
    }
}

test "a retired name set in the environment refuses by name; every Rocinante selector is one" {
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    var buf: [400]u8 = undefined;
    try std.testing.expect(settings.firstRetired(&map, &buf) == null);
    const selectors = [_][]const u8{ "TF_DS_ROCI", "TF_DS_ROCI_CODES", "TF_DS_ROCI_GATHERS", "TF_DS_ROCI_ANY", "TF_DS_ROCI_DEC", "TF_DS_ROCI_DECODE", "TF_DS_ROCI_STARTUP", "TF_DS_ROCI_SSD", "TF_DS_SSD_EVERY_RANK" };
    for (selectors) |name| {
        map.clearRetainingCapacity();
        try map.put(name, "1");
        const why = settings.firstRetired(&map, &buf).?;
        var want: [200]u8 = undefined;
        const w = try std.fmt.bufPrint(&want, "{s} no longer exists: Rocinante is the only transport", .{name});
        try std.testing.expectEqualStrings(w, why);
        // off is not a state either: 0 selected NCCL
        try map.put(name, "0");
        try std.testing.expect(settings.firstRetired(&map, &buf) != null);
    }
}

test "TF_DS_ENGRAM_OWNED: 0 is the one state that remains (the recipe exports it), on is refused" {
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    var buf: [400]u8 = undefined;
    try map.put("TF_DS_ENGRAM_OWNED", "0");
    try std.testing.expect(settings.firstRetired(&map, &buf) == null);
    try map.put("TF_DS_ENGRAM_OWNED", "1");
    try std.testing.expect(std.mem.startsWith(u8, settings.firstRetired(&map, &buf).?, "TF_DS_ENGRAM_OWNED no longer exists: "));
}

test "a retired key in engine.cfg ends the start by name; unset on it is not a use" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    var problem: []const u8 = "";
    freshState();
    const p1 = try writeCfg(&tmp, a, "TF_DS_ROCI_DECODE = 1\n");
    try std.testing.expectError(error.Config, settings.bootstrap(a, std.testing.io, &map, p1, Cap, false, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "line 1: TF_DS_ROCI_DECODE no longer exists: Rocinante is the only transport") != null);
    freshState();
    const p2 = try writeCfg(&tmp, a, "TF_DS_ROCI_DECODE = unset\n");
    try settings.bootstrap(a, std.testing.io, &map, p2, Cap, false, &problem);
}