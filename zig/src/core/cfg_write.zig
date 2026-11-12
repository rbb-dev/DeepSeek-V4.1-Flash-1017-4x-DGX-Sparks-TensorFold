//! Rewriting an engine.cfg for a persisted change: the lines of the options a change names are replaced, added or removed, and every
//! other line (comments, the schema line, options and keys it does not touch) stays as it was.
const std = @import("std");
const settings = @import("settings.zig");

/// `value` `unset` takes the option's line out.
pub const Edit = struct { o: *const settings.Option, value: []const u8 };

/// `old` with `edits` made. An option's first line is replaced where it stands, its repeats go, and one the file lacks is added by name
/// at the end; a file with no `schema` line gets `schema_version`'s.
pub fn rewrite(a: std.mem.Allocator, old: []const u8, edits: []const Edit, schema_version: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    const done = try a.alloc(bool, edits.len);
    @memset(done, false);
    var has_schema = false;
    var it = std.mem.splitScalar(u8, old, '\n');
    var lines: std.ArrayList([]const u8) = .empty;
    while (it.next()) |l| try lines.append(a, l);
    if (lines.items.len > 0 and lines.items[lines.items.len - 1].len == 0) _ = lines.pop();
    for (lines.items) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (std.mem.startsWith(u8, line, "schema") and line.len > 6 and (line[6] == ' ' or line[6] == '\t')) has_schema = true;
    }
    if (!has_schema) try out.print(a, "schema {d}\n", .{schema_version});
    for (lines.items) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len > 0 and line[0] != '#') if (std.mem.indexOfScalar(u8, line, '=')) |eq| {
            const key = std.mem.trim(u8, line[0..eq], " \t");
            if (settings.byKey(key)) |o| {
                const hit = for (edits, 0..) |e, i| {
                    if (e.o == o) break i;
                } else null;
                if (hit) |i| {
                    if (!done[i]) {
                        done[i] = true;
                        if (!std.mem.eql(u8, edits[i].value, "unset")) try out.print(a, "{s} = {s}\n", .{ key, edits[i].value });
                    }
                    continue;
                }
            }
        };
        try out.appendSlice(a, raw);
        try out.append(a, '\n');
    }
    for (edits, done) |e, d| {
        if (d or std.mem.eql(u8, e.value, "unset")) continue;
        try out.print(a, "{s} = {s}\n", .{ e.o.name, e.value });
    }
    return out.items;
}

/// `text` becomes the file at `path` whole: written to a temporary beside it, synced, given the old file's mode and owner (0600 for a new
/// file), then renamed over it, so a reader sees the old file or the new one and a failure leaves the old.
pub fn writeAtomic(a: std.mem.Allocator, path: []const u8, text: []const u8) !void {
    const tmp = try std.fmt.allocPrintSentinel(a, "{s}.tmp{d}", .{ path, std.c.getpid() }, 0);
    const pz = try a.dupeSentinel(u8, path, 0);
    const old = @import("settings_history.zig").ownerOf(pz);
    const fd = try std.posix.openatZ(std.posix.AT.FDCWD, tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, 0o600);
    var closed = false;
    errdefer {
        if (!closed) _ = std.posix.system.close(fd);
        _ = std.c.unlink(tmp);
    }
    var done: usize = 0;
    while (done < text.len) {
        const w = std.posix.system.write(fd, text.ptr + done, text.len - done);
        if (w <= 0) return error.WriteFailed;
        done += @intCast(w);
    }
    if (std.posix.system.fsync(fd) != 0) return error.SyncFailed;
    if (old) |o| {
        _ = std.c.fchown(fd, o.uid, o.gid);
        _ = std.c.fchmod(fd, @intCast(o.mode & 0o7777));
    }
    _ = std.posix.system.close(fd);
    closed = true;
    if (std.c.rename(tmp, pz) != 0) return error.RenameFailed;
}
