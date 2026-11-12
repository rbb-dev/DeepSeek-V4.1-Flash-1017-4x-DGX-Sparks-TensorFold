//! Rewriting an engine.cfg for the admin API's persisted changes: only the lines it names change.
const std = @import("std");
const settings = @import("settings.zig");
const cfg_write = @import("cfg_write.zig");

fn opt(env: []const u8) *const settings.Option {
    return settings.byEnv(env).?;
}

test "a named option's line is replaced in place; comments, other lines and their order stay" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const old = "# release b9\nschema 1\n\nTF_DS_SPILL_READERS = 8  \n# keep this\ntf_ds_spill_smalls = 5\n";
    const out = try cfg_write.rewrite(a, old, &.{.{ .o = opt("TF_DS_SPILL_READERS"), .value = "6" }}, 1);
    try std.testing.expectEqualStrings("# release b9\nschema 1\n\nTF_DS_SPILL_READERS = 6\n# keep this\ntf_ds_spill_smalls = 5\n", out);
}

test "a line spelled by id or by name is found; a repeated line collapses; an unset value takes the line out" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const id = opt("TF_DS_SPILL_SMALLS").id;
    const old = try std.fmt.allocPrint(a, "schema 1\n{d} = 5\nTF_DS_SPILL_SMALLS = 7\ntf_ds_spill_readers = 3\n", .{id});
    const out = try cfg_write.rewrite(a, old, &.{.{ .o = opt("TF_DS_SPILL_SMALLS"), .value = "unset" }}, 1);
    try std.testing.expectEqualStrings("schema 1\ntf_ds_spill_readers = 3\n", out);
    const kept = try cfg_write.rewrite(a, old, &.{.{ .o = opt("TF_DS_SPILL_SMALLS"), .value = "9" }}, 1);
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "schema 1\n{d} = 9\ntf_ds_spill_readers = 3\n", .{id}), kept);
}

test "an option the file lacks is added by name at the end; a file without a schema line gets one; no final newline is fine" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const added = try cfg_write.rewrite(a, "tf_ds_spill_readers = 3", &.{.{ .o = opt("TF_DS_ROUNDS_BESIDE"), .value = "5" }}, 4);
    try std.testing.expectEqualStrings("schema 4\ntf_ds_spill_readers = 3\ntf_ds_rounds_beside = 5\n", added);
    const empty = try cfg_write.rewrite(a, "", &.{.{ .o = opt("TF_DS_ROUNDS_BESIDE"), .value = "5" }}, 4);
    try std.testing.expectEqualStrings("schema 4\ntf_ds_rounds_beside = 5\n", empty);
    // unset of a line that is not there changes nothing
    const same = try cfg_write.rewrite(a, "schema 4\n", &.{.{ .o = opt("TF_DS_ROUNDS_BESIDE"), .value = "unset" }}, 4);
    try std.testing.expectEqualStrings("schema 4\n", same);
}

test "what is written reads back as the file the settings reader takes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const out = try cfg_write.rewrite(a, "schema 1\ntf_ds_spill_readers = 3\n", &.{ .{ .o = opt("TF_DS_ROUNDS_BESIDE"), .value = "5" }, .{ .o = opt("TF_DS_SPILL_READERS"), .value = "unset" } }, 1);
    const cfg = try settings.parseCfg(a, out);
    try std.testing.expectEqual(@as(usize, 0), cfg.bad_line);
    try std.testing.expectEqual(@as(u32, 1), cfg.declared);
    try std.testing.expectEqual(@as(usize, 1), cfg.lines.len);
    try std.testing.expectEqualStrings("tf_ds_rounds_beside", cfg.lines[0].key);
}

test "writeAtomic: the new text replaces the file whole, keeps its mode, leaves no temporary; a folder that is not there changes nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "engine.cfg", .data = "old\n" });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/engine.cfg", .{tmp.sub_path});
    const pz = try a.dupeSentinel(u8, path, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(pz, 0o640));
    try cfg_write.writeAtomic(a, path, "new text\n");
    const got = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 16));
    try std.testing.expectEqualStrings("new text\n", got);
    const owner = @import("settings_history.zig").ownerOf(pz).?;
    try std.testing.expectEqual(@as(u32, 0o640), owner.mode & 0o777);
    var n: usize = 0;
    var it = tmp.dir.iterate();
    while (try it.next(std.testing.io)) |_| n += 1;
    try std.testing.expectEqual(@as(usize, 1), n); // only engine.cfg
    try std.testing.expectError(error.FileNotFound, cfg_write.writeAtomic(a, ".zig-cache/tmp/not-here/engine.cfg", "x\n"));
}
