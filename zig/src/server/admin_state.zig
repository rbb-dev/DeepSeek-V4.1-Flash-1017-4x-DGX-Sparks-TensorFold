//! What the admin API has changed since the engine started, per option: the value a live row was given (taken by the engine at its
//! next loop turn), the value engine.cfg now holds for the next start, who set it and at which version; and the writers' rate window.
//! Strings live for the process.
const std = @import("std");
const api = @import("engine_api");
const settings = api.settings;
const live_config = api.live_config;

pub const Row = struct {
    /// A live row's value as given (`unset`: the engine's own), the version it was published at; null: none was.
    live_text: ?[]const u8 = null,
    live_version: u64 = 0,
    /// What engine.cfg holds for this option since a write (`unset`: the line was taken out); null: this process has not written it.
    file_text: ?[]const u8 = null,
    set_by: []const u8 = "",
    since: u64 = 0,
};

/// The engine's release (the build's version), for the history rows; set once by the server's main.
pub var release: []const u8 = "";

pub const State = enum { active, pending_step, pending_restart };

var rows: [settings.options.len]Row = @splat(.{});
var write_mutex: std.atomic.Mutex = .unlocked;
var rows_mutex: std.atomic.Mutex = .unlocked;

fn lockRows() void {
    while (!rows_mutex.tryLock()) std.atomic.spinLoopHint();
}

/// One writer at a time: a change is checked, written to engine.cfg and the history, and applied under this lock.
pub fn lockWrites() void {
    while (!write_mutex.tryLock()) std.atomic.spinLoopHint();
}

pub fn unlockWrites() void {
    write_mutex.unlock();
}

pub fn get(o: *const settings.Option) Row {
    lockRows();
    defer rows_mutex.unlock();
    return rows[settings.indexOf(o)];
}

fn keep(s: []const u8) []const u8 {
    return std.heap.page_allocator.dupe(u8, s) catch "";
}

/// Records a change: `live` (a row's live value, published at `version`) and/or `file` (what engine.cfg now holds).
pub fn record(o: *const settings.Option, live: ?[]const u8, file: ?[]const u8, who: []const u8, version: u64) void {
    const text_live = if (live) |l| keep(l) else null;
    const text_file = if (file) |f| keep(f) else null;
    const label = keep(who);
    lockRows();
    defer rows_mutex.unlock();
    const r = &rows[settings.indexOf(o)];
    if (text_live) |t| {
        r.live_text = t;
        r.live_version = version;
    }
    if (text_file) |t| r.file_text = t;
    r.set_by = label;
    r.since = version;
}

/// Whether the engine has taken a live value: the host's active version has reached it (a row on a server thread is taken at once).
fn taken(o: *const settings.Option, r: Row) bool {
    if (r.live_text == null) return false;
}

/// The value the engine runs now, when this process changed it live and the engine took it; null: no live change (the start's value).
/// `unset` returns the option's own default.
pub fn liveValue(o: *const settings.Option) ?[]const u8 {
    const r = get(o);
    if (!taken(o, r)) return null;
    return r.live_text;
}

/// A row's state and the value waiting for it, if any.
pub fn stateOf(o: *const settings.Option, active: ?[]const u8) struct { state: State, requested: ?[]const u8 } {
    const r = get(o);
    if (r.live_text != null and !taken(o, r)) return .{ .state = .pending_step, .requested = r.live_text };
    if (r.file_text) |f| {
        const a = active orelse "unset";
        const f_cmp = f;
        if (!std.mem.eql(u8, f_cmp, a) and !(std.mem.eql(u8, f_cmp, "unset") and active == null)) return .{ .state = .pending_restart, .requested = f };
    }
    return .{ .state = .active, .requested = null };
}

/// The newest version every change of which is active: `version`, unless a live value is still waiting for the engine, then the
/// version before the oldest such wait.
pub fn activeVersion(version: u64) u64 {
    var v = version;
    for (live_config.wired) |w| {
        const o = settings.byId(w.id) orelse continue;
        const r = get(o);
        if (r.live_text != null and !taken(o, r) and r.live_version > 0) v = @min(v, r.live_version - 1);
    }
    return v;
}

/// The options whose live value the engine has not taken yet.
pub fn pendingNames(a: std.mem.Allocator) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (live_config.wired) |w| {
        const o = settings.byId(w.id) orelse continue;
        const r = get(o);
        if (r.live_text != null and !taken(o, r)) try out.append(a, o.name);
    }
    return out.items;
}

// ---- the writers' rate window: a count of writes a minute for each label

const slots = 16;
const Window = struct { label: [64]u8 = undefined, len: usize = 0, start_s: i64 = 0, count: u32 = 0 };
var windows: [slots]Window = @splat(.{});

/// Counts one write by `label` at `now_s` against `limit` a minute: null when it may go on, else the seconds until the window
/// ends. A label that does not fit the table shares the last slot.
pub fn rateAllow(label: []const u8, limit: u32, now_s: i64) ?u32 {
    lockRows();
    defer rows_mutex.unlock();
    var slot: usize = slots - 1;
    for (&windows, 0..) |*w, i| {
        if (w.len == label.len and std.mem.eql(u8, w.label[0..w.len], label)) {
            slot = i;
            break;
        }
        if (w.len == 0) {
            slot = i;
            break;
        }
    }
    const w = &windows[slot];
    if (w.len == 0 and label.len <= w.label.len) {
        @memcpy(w.label[0..label.len], label);
        w.len = label.len;
    }
    if (now_s - w.start_s >= 60) {
        w.start_s = now_s;
        w.count = 0;
    }
    if (w.count >= limit) return @intCast(@max(1, 60 - (now_s - w.start_s)));
    w.count += 1;
    return null;
}

/// Forgets every change and every window (a test's own).
pub fn reset() void {
    lockRows();
    defer rows_mutex.unlock();
    rows = @splat(.{});
    windows = @splat(.{});
}
