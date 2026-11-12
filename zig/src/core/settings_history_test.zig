//! engine.history.jsonl: the start's `imported` rows, the replay that finds the last recorded state, and the file's lines.
const std = @import("std");
const settings = @import("settings.zig");
const history = @import("settings_history.zig");

const Cap = struct {
    var buf: [4096]u8 = undefined;
    var len: usize = 0;
    pub fn line(comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(buf[len..], fmt ++ "\n", args) catch return;
        len += s.len;
    }
    fn text() []const u8 {
        return buf[0..len];
    }
    fn reset() void {
        len = 0;
    }
};

fn rep(sha_byte: u8, entries: []const settings.Entry) settings.Report {
    var r: settings.Report = .{ .path = "x", .entries = entries };
    r.sha256 = @splat(sha_byte);
    return r;
}

fn id(env: []const u8) u16 {
    return settings.byEnv(env).?.id;
}

fn cfgIn(tmp: *std.testing.TmpDir, a: std.mem.Allocator) ![]const u8 {
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/engine.cfg", .{tmp.sub_path});
}

fn lines(a: std.mem.Allocator, path: []const u8) ![]const []const u8 {
    const l = try history.load(a, std.testing.io, path);
    return l.lines;
}

test "the first start with an engine.cfg writes one imported row naming every value; the same file again writes none" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = try cfgIn(&tmp, a);
    const e = [_]settings.Entry{ .{ .id = id("TF_DS_SPILL_READERS"), .value = "8" }, .{ .id = id("TF_DS_SPILL_SMALLS"), .value = "5" } };
    Cap.reset();
    const r1 = try history.recordStart(a, std.testing.io, cfg, rep(1, &e), "rel-a", 1_794_000_000_000, Cap);
    try std.testing.expectEqual(@as(u64, 1), r1.version);
    try std.testing.expect(r1.written);
    const path = try history.pathFor(a, cfg);
    const ls = try lines(a, path);
    try std.testing.expectEqual(@as(usize, 1), ls.len);
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, ls[0], .{});
    try std.testing.expectEqualStrings("imported", v.object.get("outcome").?.string);
    try std.testing.expectEqualStrings("cli", v.object.get("label").?.string);
    try std.testing.expectEqualStrings("rel-a", v.object.get("release").?.string);
    try std.testing.expectEqual(@as(i64, 1), v.object.get("version").?.integer);
    try std.testing.expectEqual(@as(i64, 1_794_000_000_000), v.object.get("at_ms").?.integer);
    const ch = v.object.get("changes").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), ch.len);
    try std.testing.expect(ch[0].object.get("from").? == .null);
    try std.testing.expectEqualStrings("8", ch[0].object.get("to").?.string);
    try std.testing.expectEqualStrings("tf_ds_spill_readers", ch[0].object.get("name").?.string);
    try std.testing.expectEqualStrings("default", ch[0].object.get("source_before").?.string);
    // the same file again: nothing is written, the version stays
    const r2 = try history.recordStart(a, std.testing.io, cfg, rep(1, &e), "rel-a", 1_794_000_001_000, Cap);
    try std.testing.expect(!r2.written);
    try std.testing.expectEqual(@as(u64, 1), r2.version);
    try std.testing.expectEqual(@as(usize, 1), (try lines(a, path)).len);
}

test "an edited engine.cfg is imported as the difference from the last recorded state: changed, added, removed; a comment-only edit has none" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = try cfgIn(&tmp, a);
    const readers = id("TF_DS_SPILL_READERS");
    const smalls = id("TF_DS_SPILL_SMALLS");
    const quota = id("TF_DS_SPILL_QUOTA");
    _ = try history.recordStart(a, std.testing.io, cfg, rep(1, &.{ .{ .id = readers, .value = "8" }, .{ .id = smalls, .value = "5" } }), "r", 1, Cap);
    // readers 8 -> 6, smalls removed, quota added
    const r2 = try history.recordStart(a, std.testing.io, cfg, rep(2, &.{ .{ .id = readers, .value = "6" }, .{ .id = quota, .value = "100" } }), "r", 2, Cap);
    try std.testing.expectEqual(@as(u64, 2), r2.version);
    try std.testing.expectEqual(@as(usize, 3), r2.changes);
    // a new sha with the same values: a row, no changes
    const r3 = try history.recordStart(a, std.testing.io, cfg, rep(3, &.{ .{ .id = readers, .value = "6" }, .{ .id = quota, .value = "100" } }), "r", 3, Cap);
    try std.testing.expectEqual(@as(u64, 3), r3.version);
    try std.testing.expectEqual(@as(usize, 0), r3.changes);
    const ls = try lines(a, try history.pathFor(a, cfg));
    try std.testing.expectEqual(@as(usize, 3), ls.len);
    const v2 = try std.json.parseFromSliceLeaky(std.json.Value, a, ls[1], .{});
    for (v2.object.get("changes").?.array.items) |c| {
        const name = c.object.get("name").?.string;
        if (std.mem.eql(u8, name, "tf_ds_spill_readers")) {
            try std.testing.expectEqualStrings("8", c.object.get("from").?.string);
            try std.testing.expectEqualStrings("6", c.object.get("to").?.string);
            try std.testing.expectEqualStrings("cfg", c.object.get("source_before").?.string);
        } else if (std.mem.eql(u8, name, "tf_ds_spill_smalls")) {
            try std.testing.expectEqualStrings("5", c.object.get("from").?.string);
            try std.testing.expect(c.object.get("to").? == .null);
        } else try std.testing.expectEqualStrings("tf_ds_spill_quota", name);
    }
}

test "a stopped engine's edit after a long history is found by replaying every row, and a damaged line is skipped, counted and never written over" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = try cfgIn(&tmp, a);
    const readers = id("TF_DS_SPILL_READERS");
    _ = try history.recordStart(a, std.testing.io, cfg, rep(1, &.{.{ .id = readers, .value = "8" }}), "r", 1, Cap);
    const path = try history.pathFor(a, cfg);
    {
        const old = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 20));
        const damaged = try std.fmt.allocPrint(a, "{s}this is not json\n", .{old});
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = damaged });
    }
    Cap.reset();
    const r = try history.recordStart(a, std.testing.io, cfg, rep(2, &.{.{ .id = readers, .value = "9" }}), "r", 2, Cap);
    try std.testing.expectEqual(@as(u64, 2), r.version);
    try std.testing.expectEqual(@as(usize, 1), r.changes);
    try std.testing.expect(std.mem.indexOf(u8, Cap.text(), "1 line") != null);
    const l = try history.load(a, std.testing.io, path);
    try std.testing.expectEqual(@as(usize, 3), l.lines.len); // nothing was removed
}

test "every row is one valid JSON line, whatever the value holds" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = try cfgIn(&tmp, a);
    const dir = id("TF_DS_SPILL_DIR");
    const nasty = "a\"b\\c\td\xff\xfe/é\x01";
    _ = try history.recordStart(a, std.testing.io, cfg, rep(1, &.{.{ .id = dir, .value = nasty }}), "rel\"x", 1, Cap);
    const ls = try lines(a, try history.pathFor(a, cfg));
    try std.testing.expectEqual(@as(usize, 1), ls.len);
    try std.testing.expect(std.mem.indexOfScalar(u8, ls[0], '\n') == null);
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, ls[0], .{});
    try std.testing.expectEqualStrings("rel\"x", v.object.get("release").?.string);
    // a value that is valid text comes back as it went in (the replay compares by text)
    _ = try history.recordStart(a, std.testing.io, cfg, rep(2, &.{.{ .id = dir, .value = "a\"b\\c\t/é" }}), "r", 2, Cap);
    const again = try history.recordStart(a, std.testing.io, cfg, rep(2, &.{.{ .id = dir, .value = "a\"b\\c\t/é" }}), "r", 3, Cap);
    try std.testing.expect(!again.written);
}

test "a history that cannot be written does not end the start: it is told, once" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    Cap.reset();
    const r = try history.recordStart(a, std.testing.io, ".zig-cache/tmp/no-such-dir-here/engine.cfg", rep(1, &.{.{ .id = id("TF_DS_SPILL_READERS"), .value = "8" }}), "r", 1, Cap);
    try std.testing.expect(!r.written);
    try std.testing.expect(std.mem.indexOf(u8, Cap.text(), "engine.history.jsonl") != null);
}

test "the version is the newest recorded row's, and a missing file is version 0" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = try cfgIn(&tmp, a);
    try std.testing.expectEqual(@as(u64, 0), (try history.state(a, std.testing.io, try history.pathFor(a, cfg))).version);
    _ = try history.recordStart(a, std.testing.io, cfg, rep(1, &.{}), "r", 1, Cap);
    try std.testing.expectEqual(@as(u64, 1), (try history.state(a, std.testing.io, try history.pathFor(a, cfg))).version);
}

test "the history file is owned by the folder's owner and group, mode 0640, and the start line says so" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = try cfgIn(&tmp, a);
    Cap.reset();
    _ = try history.recordStart(a, std.testing.io, cfg, rep(1, &.{.{ .id = id("TF_DS_SPILL_READERS"), .value = "8" }}), "r", 1, Cap);
    const dirz = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}, 0);
    const filez = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dirz, history.file_name }, 0);
    const d = history.ownerOf(dirz).?;
    const f = history.ownerOf(filez).?;
    try std.testing.expectEqual(d.uid, f.uid);
    try std.testing.expectEqual(d.gid, f.gid);
    try std.testing.expectEqual(@as(u32, 0o640), f.mode & 0o777);
    try std.testing.expect(std.mem.indexOf(u8, Cap.text(), "owned by the folder's owner") != null);
}

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
