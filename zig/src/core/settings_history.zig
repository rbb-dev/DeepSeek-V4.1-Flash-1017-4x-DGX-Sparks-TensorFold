//! engine.history.jsonl, beside engine.cfg: one line per change to the settings, appended and synced, never rewritten.
//!
//! A change is recorded when it is made (the admin API's writes) or, for a hand edit made while the engine was stopped,
//! when the next start finds engine.cfg's hash different from the one the last row carries: that start writes an
//! `imported` row (label `cli`) whose changes are the file's difference from the last recorded state, found by replaying
//! every row's `to` values in order. A start never depends on this file: when it cannot be read or written the start goes
//! on and says so. The version of a row is the engine's `config_version`: the newest row's number, plus one for the next.
const std = @import("std");
const settings = @import("settings.zig");

pub const file_name = "engine.history.jsonl";

/// The most history a start or a read will take in: a file larger than this is not read (it is rotated by size later).
pub const max_bytes = 32 << 20;

/// Where this process's history lives ("": no engine.cfg, so none) and the newest recorded version; set by `recordStart`.
pub var path: []const u8 = "";
pub var version: std.atomic.Value(u64) = .init(0);

/// `engine.history.jsonl` beside `cfg_path`.
pub fn pathFor(a: std.mem.Allocator, cfg_path: []const u8) std.mem.Allocator.Error![]const u8 {
    const dir = std.fs.path.dirname(cfg_path) orelse return a.dupe(u8, file_name);
    return std.fs.path.join(a, &.{ dir, file_name });
}

pub const Loaded = struct { lines: []const []const u8 };

/// The non-empty lines of the history at `p`; a missing file has none.
pub fn load(a: std.mem.Allocator, io: std.Io, p: []const u8) !Loaded {
    const text = std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(max_bytes)) catch |e| switch (e) {
        error.FileNotFound => return .{ .lines = &.{} },
        else => return e,
    };
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const l = std.mem.trim(u8, raw, " \t\r");
        if (l.len > 0) try out.append(a, l);
    }
    return .{ .lines = out.items };
}

/// What the rows recorded, replayed in order.
pub const State = struct {
    /// The newest row's version (0: none).
    version: u64 = 0,
    /// The newest `cfg_sha256` a row carried ("": none).
    sha: []const u8 = "",
    /// Each option id's last recorded value; an option a row removed is gone.
    values: std.AutoArrayHashMapUnmanaged(u16, []const u8) = .empty,
    /// Lines that are no row (kept in the file, counted here).
    bad: usize = 0,
};

const Replayed = struct {
    version: u64 = 0,
    cfg_sha256: []const u8 = "",
    changes: []const struct { id: u32 = 0, to: ?[]const u8 = null, persisted: bool = true } = &.{},
};

/// Replays the history at `p`.
pub fn state(a: std.mem.Allocator, io: std.Io, p: []const u8) !State {
    const l = try load(a, io, p);
    var s: State = .{};
    for (l.lines) |line| {
        const row = std.json.parseFromSliceLeaky(Replayed, a, line, .{ .ignore_unknown_fields = true }) catch {
            s.bad += 1;
            continue;
        };
        s.version = @max(s.version, row.version);
        if (row.cfg_sha256.len > 0) s.sha = row.cfg_sha256;
        for (row.changes) |c| {
            if (c.id == 0 or c.id > std.math.maxInt(u16) or !c.persisted) continue;
            if (c.to) |to| try s.values.put(a, @intCast(c.id), to) else _ = s.values.swapRemove(@intCast(c.id));
        }
    }
    return s;
}

/// A JSON string. Valid UTF-8 goes through as it is; a byte of text that is not valid is written as \u00XX, which the
/// reader will not give back as the same byte (the history is a record for people, not a copy of the file).
fn quote(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(a, '"');
    const valid = std.unicode.utf8ValidateSlice(s);
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(a, "\\\""),
        '\\' => try out.appendSlice(a, "\\\\"),
        '\n' => try out.appendSlice(a, "\\n"),
        '\t' => try out.appendSlice(a, "\\t"),
        '\r' => try out.appendSlice(a, "\\r"),
        0...8, 11, 12, 14...31, 127 => try out.print(a, "\\u{x:0>4}", .{c}),
        128...255 => if (valid) try out.append(a, c) else try out.print(a, "\\u{x:0>4}", .{c}),
        else => try out.append(a, c),
    };
    try out.append(a, '"');
}

fn optQuote(a: std.mem.Allocator, out: *std.ArrayList(u8), s: ?[]const u8) !void {
    if (s) |x| try quote(a, out, x) else try out.appendSlice(a, "null");
}

pub const Result = struct {
    /// The newest version after this call.
    version: u64,
    /// A row was appended.
    written: bool,
    /// How many changes it names.
    changes: usize = 0,
};

/// At a start with an engine.cfg: when its hash is not the one the history last recorded, appends the `imported` row for
/// the difference. `report` is what `settings.bootstrap` read. Sets `path` and `version`. A history that cannot be read or
/// written is told through `Log` and the start goes on.
pub fn recordStart(a: std.mem.Allocator, io: std.Io, cfg_path: []const u8, report: settings.Report, release: []const u8, now_ms: i64, comptime Log: type) std.mem.Allocator.Error!Result {
    const p = try pathFor(a, cfg_path);
    path = p;
    const st = state(a, io, p) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            Log.line("{s}: cannot read it ({s}): no row is written at this start", .{ p, @errorName(e) });
            return .{ .version = 0, .written = false };
        },
    };
    version.store(st.version, .release);
    if (st.bad > 0) Log.line("{s}: {d} line{s} is not a history row (skipped, kept in the file)", .{ p, st.bad, if (st.bad == 1) "" else "s" });
    const hex = std.fmt.bytesToHex(report.sha256, .lower);
    if (std.mem.eql(u8, st.sha, &hex)) return .{ .version = st.version, .written = false };

    var row: std.ArrayList(u8) = .empty;
    const v = st.version + 1;
    try row.print(a, "{{\"version\":{d},\"at_ms\":{d},\"label\":\"cli\",\"scope\":\"cli\",\"peer\":null,\"request_id\":null,\"endpoint\":\"start\",\"apply\":\"next_start\",\"outcome\":\"imported\",\"reason\":", .{ v, now_ms });
    try quote(a, &row, if (st.sha.len == 0) "the first start with this engine.cfg" else "engine.cfg changed while the engine was stopped");
    try row.print(a, ",\"schema_version\":{d},\"release\":", .{settings.schema_version});
    try quote(a, &row, release);
    try row.print(a, ",\"cfg_sha256\":\"{s}\",\"changes\":[", .{&hex});
    var n: usize = 0;
    var seen: std.AutoArrayHashMapUnmanaged(u16, void) = .empty;
    for (report.entries) |e| {
        try seen.put(a, e.id, {});
        const old = st.values.get(e.id);
        if (old != null and std.mem.eql(u8, old.?, e.value)) continue;
        try change(a, &row, &n, e.id, old, e.value);
    }
    for (st.values.keys(), st.values.values()) |id, old| {
        if (seen.contains(id)) continue;
        try change(a, &row, &n, id, old, null);
    }
    try row.appendSlice(a, "],\"lease\":null,\"active_after_ms\":0}\n");

    const handed = appendLine(a, p, row.items) catch |e| {
        Log.line("{s}: cannot write it ({s}): the change to engine.cfg is not recorded", .{ p, @errorName(e) });
        return .{ .version = st.version, .written = false };
    };
    version.store(v, .release);
    Log.line("{s}: version {d}: engine.cfg imported, {d} change{s}; the file is {s}, mode 0640", .{ p, v, n, if (n == 1) "" else "s", if (handed) "owned by the folder's owner and group" else "left with its creator as owner (this process may not hand it over)" });
    return .{ .version = v, .written = true, .changes = n };
}

/// The rows of an engine started with no engine.cfg: there is no folder to keep them in, so they live here, in memory, for the process.
var memory_rows: std.ArrayList([]u8) = .empty;
var memory_lock: std.atomic.Mutex = .unlocked;

fn lockMemory() void {
    while (!memory_lock.tryLock()) std.atomic.spinLoopHint();
}

/// The in-memory rows (copies, in order); empty when none was kept.
pub fn memoryLines(a: std.mem.Allocator) ![]const []const u8 {
    lockMemory();
    defer memory_lock.unlock();
    const out = try a.alloc([]const u8, memory_rows.items.len);
    for (memory_rows.items, out) |r, *o| o.* = try a.dupe(u8, std.mem.trimEnd(u8, r, "\n"));
    return out;
}

/// Forgets the in-memory rows (a test's own).
pub fn resetMemory() void {
    lockMemory();
    defer memory_lock.unlock();
    for (memory_rows.items) |r| std.heap.page_allocator.free(r);
    memory_rows.clearRetainingCapacity();
}

/// One change of a write row: what it was, what it is, whether engine.cfg holds it, and its state.
pub const WriteChange = struct { id: u16, from: ?[]const u8, from_file: ?[]const u8 = null, to: ?[]const u8, persisted: bool, state_after: []const u8 };

/// A row the admin API writes for a change it accepted.
pub const WriteRow = struct {
    at_ms: i64,
    label: []const u8,
    scope: []const u8,
    peer: []const u8,
    request_id: []const u8,
    endpoint: []const u8,
    apply: []const u8,
    outcome: []const u8,
    reason: []const u8,
    release: []const u8,
    /// The hash of the engine.cfg this write left (null: it did not write the file).
    cfg_sha256: ?[]const u8,
    changes: []const WriteChange,
    reverts_version: ?u64 = null,
};

/// Appends a write row at the next version and raises the version; the caller holds the writers' lock. The row reaches the file
/// (synced) before this returns; a failure leaves the version where it was.
pub fn appendWrite(a: std.mem.Allocator, io: std.Io, p: []const u8, r: WriteRow) !struct { version: u64, handed: bool } {
    _ = io;
    const v = version.load(.acquire) + 1;
    var row: std.ArrayList(u8) = .empty;
    try row.print(a, "{{\"version\":{d},\"at_ms\":{d},\"label\":", .{ v, r.at_ms });
    try quote(a, &row, r.label);
    try row.appendSlice(a, ",\"scope\":");
    try quote(a, &row, r.scope);
    try row.appendSlice(a, ",\"peer\":");
    try quote(a, &row, r.peer);
    try row.appendSlice(a, ",\"request_id\":");
    try quote(a, &row, r.request_id);
    try row.appendSlice(a, ",\"endpoint\":");
    try quote(a, &row, r.endpoint);
    try row.appendSlice(a, ",\"apply\":");
    try quote(a, &row, r.apply);
    try row.appendSlice(a, ",\"outcome\":");
    try quote(a, &row, r.outcome);
    try row.appendSlice(a, ",\"reason\":");
    try quote(a, &row, r.reason);
    try row.print(a, ",\"schema_version\":{d},\"release\":", .{settings.schema_version});
    try quote(a, &row, r.release);
    if (r.cfg_sha256) |h| try row.print(a, ",\"cfg_sha256\":\"{s}\"", .{h});
    if (r.reverts_version) |rv| try row.print(a, ",\"reverts_version\":{d}", .{rv});
    try row.appendSlice(a, ",\"changes\":[");
    var n: usize = 0;
    for (r.changes) |c| {
        if (n > 0) try row.append(a, ',');
        n += 1;
        const o = settings.byId(c.id);
        try row.print(a, "{{\"id\":{d},\"name\":", .{c.id});
        try quote(a, &row, if (o) |x| x.name else "");
        try row.appendSlice(a, ",\"from\":");
        try optQuote(a, &row, c.from);
        try row.appendSlice(a, ",\"from_file\":");
        try optQuote(a, &row, c.from_file);
        try row.appendSlice(a, ",\"to\":");
        try optQuote(a, &row, c.to);
        try row.print(a, ",\"persisted\":{s},\"source_before\":\"{s}\",\"state_after\":", .{ if (c.persisted) "true" else "false", if (c.from != null) "cfg" else "default" });
        try quote(a, &row, c.state_after);
        try row.print(a, ",\"out_changes\":{s}}}", .{if (o != null and o.?.out_changes) "true" else "false"});
    }
    try row.appendSlice(a, "],\"lease\":null,\"active_after_ms\":0}\n");
    // no history file (no engine.cfg): the row is kept in memory and the answer says so
    if (p.len == 0) {
        const copy = try std.heap.page_allocator.dupe(u8, row.items);
        lockMemory();
        defer memory_lock.unlock();
        try memory_rows.append(std.heap.page_allocator, copy);
        version.store(v, .release);
        return .{ .version = v, .handed = false };
    }
    const handed = try appendLine(a, p, row.items);
    version.store(v, .release);
    return .{ .version = v, .handed = handed };
}

fn change(a: std.mem.Allocator, row: *std.ArrayList(u8), n: *usize, id: u16, from: ?[]const u8, to: ?[]const u8) !void {
    if (n.* > 0) try row.append(a, ',');
    n.* += 1;
    const o = settings.byId(id);
    try row.print(a, "{{\"id\":{d},\"name\":", .{id});
    try quote(a, row, if (o) |x| x.name else "");
    try row.appendSlice(a, ",\"from\":");
    try optQuote(a, row, from);
    try row.appendSlice(a, ",\"to\":");
    try optQuote(a, row, to);
    try row.print(a, ",\"source_before\":\"{s}\",\"state_after\":\"active\",\"out_changes\":{s}}}", .{ if (from != null) "cfg" else "default", if (o != null and o.?.out_changes) "true" else "false" });
}

/// One line appended and synced; the row ends with a newline, so a partial write is a damaged last line, never a merged one.
pub const Owner = struct { uid: u32, gid: u32, mode: u32 };

/// The owner, group and permission bits of the file or folder at `pathz` (null when it cannot be read).
pub fn ownerOf(pathz: [*:0]const u8) ?Owner {
    var sx: std.os.linux.Statx = undefined;
    const rc = std.os.linux.statx(std.posix.AT.FDCWD, pathz, 0, .{ .MODE = true, .UID = true, .GID = true }, &sx);
    if (std.os.linux.errno(rc) != .SUCCESS) return null;
    return .{ .uid = sx.uid, .gid = sx.gid, .mode = sx.mode };
}

/// The file is handed to the folder's owner and group, mode 0640: the engine may run as root in a container while the operator who
/// owns the folder reads the history. Returns whether the hand-over was done (it is not when the process may not chown).
fn appendLine(a: std.mem.Allocator, p: []const u8, line: []const u8) !bool {
    const pz = try a.dupeSentinel(u8, p, 0);
    const fd = try std.posix.openatZ(std.posix.AT.FDCWD, pz, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true }, 0o640);
    defer _ = std.posix.system.close(fd);
    var handed = false;
    const dir = std.fs.path.dirname(p) orelse ".";
    const dz = try a.dupeSentinel(u8, dir, 0);
    if (ownerOf(dz)) |o| {
        handed = std.c.fchown(fd, o.uid, o.gid) == 0;
        _ = std.c.fchmod(fd, 0o640);
    }
    var done: usize = 0;
    while (done < line.len) {
        const w = std.posix.system.write(fd, line.ptr + done, line.len - done);
        if (w <= 0) return error.WriteFailed;
        done += @intCast(w);
    }
    if (std.posix.system.fsync(fd) != 0) return error.SyncFailed;
    return handed;
}
