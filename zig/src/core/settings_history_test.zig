
test "a write row is one line; its persisted values are the replayed state, an unpersisted one is not, and its sha is the file's so the next start imports nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = try cfgIn(&tmp, a);
    const path = try history.pathFor(a, cfg);
    const readers = id("TF_DS_SPILL_READERS");
    const rounds = id("TF_DS_ROUNDS_BESIDE");
    history.path = path;
    history.version.store(0, .release);
    const sha: [32]u8 = @splat(7);
    const hex = std.fmt.bytesToHex(sha, .lower);
    const w = try history.appendWrite(a, std.testing.io, path, .{
        .at_ms = 5,
        .label = "admin",
        .scope = "admin",
        .peer = "10.0.0.1",
        .request_id = "r-1",
        .endpoint = "set",
        .apply = "live",
        .outcome = "applied",
        .reason = "a \"quoted\"\nreason",
        .release = "rel",
        .cfg_sha256 = &hex,
        .changes = &.{
            .{ .id = readers, .from = "8", .to = "6", .persisted = true, .state_after = "pending_restart" },
            .{ .id = rounds, .from = null, .to = "4", .persisted = false, .state_after = "active" },
        },
    });
    try std.testing.expectEqual(@as(u64, 1), w.version);
    try std.testing.expectEqual(@as(u64, 1), history.version.load(.acquire));
    const ls = try lines(a, path);
    try std.testing.expectEqual(@as(usize, 1), ls.len);
    try std.testing.expect(std.mem.indexOfScalar(u8, ls[0], '\n') == null);
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, ls[0], .{});
    try std.testing.expectEqualStrings("a \"quoted\"\nreason", v.object.get("reason").?.string);
    const st = try history.state(a, std.testing.io, path);
    try std.testing.expectEqualStrings("6", st.values.get(@intCast(readers)).?);
    try std.testing.expect(st.values.get(@intCast(rounds)) == null); // not persisted: not in the file
    try std.testing.expectEqualStrings(&hex, st.sha);
    // the next start, reading the file this write left (same sha, same values), records nothing
    var r: settings.Report = .{ .path = "x", .entries = &.{.{ .id = readers, .value = "6" }} };
    r.sha256 = sha;
    const res = try history.recordStart(a, std.testing.io, cfg, r, "rel", 9, Cap);
    try std.testing.expect(!res.written);
    try std.testing.expectEqual(@as(u64, 1), res.version);
}
