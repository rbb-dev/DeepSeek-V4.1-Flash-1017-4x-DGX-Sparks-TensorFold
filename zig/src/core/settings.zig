//! The engine's settings: one table of every option the engine reads (settings_table.zig, generated from
//! zig/config-schema.json), and the start-up reader that puts an engine.cfg's values where the engine's own parsers
//! look for them.
//!
//! Option ids are stable forever (never reused); the name and env name are how a person or an old script says the same
//! thing. With an engine.cfg, the file is the source of every option it names: its value replaces the environment's
//! (logged), except for a debug or bench option, where the environment stays the way to try something for one run. An
//! option the file leaves out keeps the environment's value, which is the bootstrap that lets a start with a few
//! variables become a start with a file. The parsers that read the environment today are untouched (the file's values
//! are put where they look), and a start without TF_ENGINE_CFG is byte-for-byte what it was. Every rank reads the same file
//! (the followers' binary calls the same bootstrap) and the ranks' digests of it are compared at the start.
const std = @import("std");

/// When an option may change (CONFIG design, section 1): per request, rank 0's host at its next turn, all four ranks at one
/// step, when idle, at once after a drain, or only at a restart.
pub const Apply = enum { P, H, A, I, N, R };
pub const Kind = enum { flag, int, float, text, choice };
/// Who may set an option through the admin surface.
pub const Who = enum { admin, agent };
/// Whether a test compares the option's default with the parser that reads it.
pub const Check = enum { unchecked, tested };

/// What an agent key may set a row to, and for how long.
pub const Agent = struct { min: f64, max: f64, max_ttl_s: u32 };
/// One thing a setting costs, with whether the number was measured or estimated and where it came from.
pub const Cost = struct { what: []const u8, value: []const u8, label: []const u8, source: []const u8 };
/// A screen of the page's settings, in order.
pub const Tab = struct { name: []const u8, title: []const u8, about: []const u8 };

pub const Option = struct {
    id: u16,
    name: []const u8,
    env: []const u8,
    group: []const u8,
    tab: []const u8,
    kind: Kind,
    unit: []const u8,
    min: ?f64,
    max: ?f64,
    /// The value when nothing sets it, as text; empty: the code decides (no setting).
    default: []const u8,
    /// When `default` is empty because the engine computes it or has none: the page's words for it ("Same as the context window").
    default_text: []const u8,
    check: Check,
    apply: Apply,
    /// A key: never shown, logged or set from a file.
    secret: bool,
    /// Never shown, listed or set through the admin surface: it answers a name that does not exist (a key file's path, the
    /// debug rows, the options that only a test binary reads).
    hidden: bool,
    who: []const Who,
    /// The option's name as the admin page prints it: sentence case, no underscores, no environment names.
    label: []const u8,
    doc: []const u8,
    values: []const []const u8,
    /// With an engine.cfg that names it, the environment's value still wins: a debug or bench option.
    env_override: bool,
    /// Documented to change what the engine replies: only an admin key may set it.
    out_changes: bool,
    /// Whether `now` may apply it: false for a change whose apply is long (a pool or arena resize), which waits for idle.
    now_ok: bool,
    /// What an agent key may set it to; null for a row no agent may set.
    agent: ?Agent,
    /// `name=value` conditions on other options under which this one means anything.
    requires: []const []const u8,
    cost: []const Cost,

    /// The kind's name as the admin surface says it.
    pub fn kindName(o: *const Option) []const u8 {
        return if (o.kind == .choice) "enum" else @tagName(o.kind);
    }
};

pub const table = @import("settings_table.zig");
pub const options = table.options;
pub const tabs = table.tabs;

/// A name that no longer exists. Naming it, in the environment or in engine.cfg, ends the start by name: an old
/// configuration is caught, never ignored.
pub const Retired = struct {
    env: []const u8,
    /// `.set`: any value refuses. `.on`: a value other than 0 refuses (0 is the one state that remains).
    when: When,
    /// What replaced it, in the words after "no longer exists: ".
    why: []const u8,

    pub const When = enum { set, on };
};
pub const retired = table.retired;
pub const schema_version = table.schema_version;

/// The retired name `env`, or null.
pub fn retiredByKey(key: []const u8) ?*const Retired {
    for (&retired) |*r| if (std.mem.eql(u8, r.env, key)) return r;
    return null;
}

/// The refusal for retired name `r` set to `value`, in `buf`; null when the value is the one that remains (`.on` and 0).
pub fn retiredRefusal(r: *const Retired, value: []const u8, buf: []u8) ?[]const u8 {
    if (r.when == .on and std.mem.eql(u8, value, "0")) return null;
    return say(buf, "{s} no longer exists: {s}", .{ r.env, r.why });
}

/// The refusal for the first retired name `lookup` (anything with `get(name) ?[]const u8`: the process environment in the
/// engine) finds set, in `buf`; null when none is.
pub fn firstRetired(lookup: anytype, buf: []u8) ?[]const u8 {
    for (&retired) |*r| if (lookup.get(r.env)) |v| if (retiredRefusal(r, v, buf)) |why| return why;
    return null;
}

/// The option with this id (the table is sorted by id).
pub fn byId(id: u32) ?*const Option {
    var lo: usize = 0;
    var hi: usize = options.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (options[mid].id == id) return &options[mid];
        if (options[mid].id < id) lo = mid + 1 else hi = mid;
    }
    return null;
}

/// The option whose environment name is `env`.
pub fn byEnv(env: []const u8) ?*const Option {
    for (&options) |*o| if (std.mem.eql(u8, o.env, env)) return o;
    return null;
}

/// The option a config key names: its decimal id, its environment name, or its lower-case name.
pub fn byKey(key: []const u8) ?*const Option {
    if (key.len > 0 and key[0] >= '0' and key[0] <= '9') {
        const id = std.fmt.parseInt(u32, key, 10) catch return null;
        return byId(id);
    }
    if (byEnv(key)) |o| return o;
    for (&options) |*o| if (std.mem.eql(u8, o.name, key)) return o;
    return null;
}

/// The option's place in the table.
pub fn indexOf(o: *const Option) usize {
    return (@intFromPtr(o) - @intFromPtr(&options[0])) / @sizeOf(Option);
}

/// The kind of refusal, for a caller that acts on it (the admin surface's `refused[].code`).
pub const Code = enum { empty, control_character, not_0_or_1, not_a_number, not_finite, below_minimum, above_maximum, not_a_choice };
pub const Refused = struct { code: Code, message: []const u8 };

/// Why `value` is not a setting of `o`, with the kind of refusal, or null. `buf` holds the message. Nothing is clamped: a
/// value outside the range is refused by name.
pub fn check(o: *const Option, value: []const u8, buf: []u8) ?Refused {
    if (value.len == 0) {
        if (o.kind == .choice) for (o.values) |v| if (v.len == 0) return null;
        return .{ .code = .empty, .message = say(buf, "{s}: empty (write `unset` to leave it to the engine)", .{o.env}) };
    }
    for (value) |c| if (c < 32 or c == 127) return .{ .code = .control_character, .message = say(buf, "{s}: a control character", .{o.env}) };
    switch (o.kind) {
        .flag => if (!std.mem.eql(u8, value, "0") and !std.mem.eql(u8, value, "1")) return .{ .code = .not_0_or_1, .message = say(buf, "{s}={s}: 0 or 1", .{ o.env, value }) },
        .int => {
            const v = std.fmt.parseInt(i64, value, 10) catch return .{ .code = .not_a_number, .message = say(buf, "{s}={s}: a whole number{s}", .{ o.env, value, rangeText(o, buf[buf.len / 2 ..]) }) };
            if (o.min) |lo| if (@as(f64, @floatFromInt(v)) < lo) return .{ .code = .below_minimum, .message = say(buf, "{s}={s}: below the minimum{s}", .{ o.env, value, rangeText(o, buf[buf.len / 2 ..]) }) };
            if (o.max) |hi| if (@as(f64, @floatFromInt(v)) > hi) return .{ .code = .above_maximum, .message = say(buf, "{s}={s}: above the maximum{s}", .{ o.env, value, rangeText(o, buf[buf.len / 2 ..]) }) };
        },
        .float => {
            const v = std.fmt.parseFloat(f64, value) catch return .{ .code = .not_a_number, .message = say(buf, "{s}={s}: a number{s}", .{ o.env, value, rangeText(o, buf[buf.len / 2 ..]) }) };
            if (!std.math.isFinite(v)) return .{ .code = .not_finite, .message = say(buf, "{s}={s}: a finite number", .{ o.env, value }) };
            if (o.min) |lo| if (v < lo) return .{ .code = .below_minimum, .message = say(buf, "{s}={s}: below the minimum{s}", .{ o.env, value, rangeText(o, buf[buf.len / 2 ..]) }) };
            if (o.max) |hi| if (v > hi) return .{ .code = .above_maximum, .message = say(buf, "{s}={s}: above the maximum{s}", .{ o.env, value, rangeText(o, buf[buf.len / 2 ..]) }) };
        },
        .choice => {
            for (o.values) |v| if (std.mem.eql(u8, v, value)) return null;
            return .{ .code = .not_a_choice, .message = say(buf, "{s}={s}: not one of the choices", .{ o.env, value }) };
        },
        .text => {},
    }
    return null;
}

/// Why `value` is not a setting of `o`, or null (`check`'s message).
pub fn refusal(o: *const Option, value: []const u8, buf: []u8) ?[]const u8 {
    return (check(o, value, buf) orelse return null).message;
}

fn say(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch buf[0..@min(buf.len, 64)];
}

/// " (from lo to hi unit)" for a reason's tail; empty when the option has no range.
fn rangeText(o: *const Option, buf: []u8) []const u8 {
    if (o.min == null and o.max == null) return "";
    if (o.min != null and o.max != null) return std.fmt.bufPrint(buf, " (from {d} to {d} {s})", .{ o.min.?, o.max.?, o.unit }) catch "";
    if (o.min) |lo| return std.fmt.bufPrint(buf, " (at least {d} {s})", .{ lo, o.unit }) catch "";
    return std.fmt.bufPrint(buf, " (at most {d} {s})", .{ o.max.?, o.unit }) catch "";
}

/// One `key = value` line of an engine.cfg.
pub const Line = struct { n: usize, key: []const u8, value: []const u8 };

pub const Cfg = struct {
    /// The `schema N` line's N (0: none).
    declared: u32 = 0,
    lines: []const Line = &.{},
    /// A line that is no setting and no comment (1-based), with why; 0: none.
    bad_line: usize = 0,
    bad_why: []const u8 = "",
};

/// Reads an engine.cfg's text: `#` lines and blank lines are skipped, `schema N` declares the table version, every
/// other line is `key = value`. Allocates the line list from `a`.
pub fn parseCfg(a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error!Cfg {
    var out: Cfg = .{};
    var lines: std.ArrayList(Line) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    var n: usize = 0;
    while (it.next()) |raw| {
        n += 1;
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.startsWith(u8, line, "schema") and line.len > 6 and (line[6] == ' ' or line[6] == '\t')) {
            out.declared = std.fmt.parseInt(u32, std.mem.trim(u8, line[6..], " \t"), 10) catch {
                out.bad_line = n;
                out.bad_why = "schema needs a whole number";
                break;
            };
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse {
            out.bad_line = n;
            out.bad_why = "not `key = value`, a comment or `schema N`";
            break;
        };
        const key = std.mem.trim(u8, line[0..eq], " \t");
        if (key.len == 0) {
            out.bad_line = n;
            out.bad_why = "no key before the =";
            break;
        }
        try lines.append(a, .{ .n = n, .key = key, .value = std.mem.trim(u8, line[eq + 1 ..], " \t") });
    }
    out.lines = lines.items;
    return out;
}

pub const Source = enum { default, env, cfg };

/// What start-up found for one option: the value that applies and where it came from, and, when the environment and
/// engine.cfg both named it, the value that lost and where it came from (`other_source` is `.default` when none did).
pub const Resolved = struct { source: Source = .default, value: []const u8 = "", other_source: Source = .default, other: []const u8 = "" };

/// One value engine.cfg sets (an `unset` line and an unknown key are not entries).
pub const Entry = struct { id: u16, value: []const u8 };

pub const Report = struct {
    path: []const u8 = "",
    sha256: [32]u8 = @splat(0),
    bytes: usize = 0,
    applied: usize = 0,
    /// Values of the file that replaced an environment value.
    cfg_won: usize = 0,
    /// Debug or bench options whose environment value kept its place over the file's.
    env_won: usize = 0,
    unknown: usize = 0,
    unset: usize = 0,
    declared: u32 = 0,
    entries: []const Entry = &.{},
};

/// What engine.cfg set, as one number the ranks compare at the start (prompt2d.Settings.cfg): 0 when no file was read, else
/// nonzero and the sum of a hash of each (id, value) the file set. Comments, order, unknown keys and `unset` lines do not
/// count; a rank that read another file, or none, differs.
pub fn cfgDigest() u64 {
    if (report.path.len == 0) return 0;
    var sum: u64 = 0;
    for (report.entries) |e| {
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.asBytes(&e.id));
        h.update(e.value);
        sum +%= h.final();
    }
    return sum | 1;
}

pub var resolved: [options.len]Resolved = @splat(.{});
pub var report: Report = .{};
pub var ready: std.atomic.Value(bool) = .init(false);

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// Reads `cfg_path` (null: none) over `map`, the process environment, and records every option's source and value.
/// With a file: each value it sets is put in `map` (and, `libc`, in the process's own environment, which the engine's
/// parsers read) in place of the environment's (logged), except for a debug or bench option, where the environment's
/// value stays (logged); an option the file leaves out keeps the environment's; a key this table lacks is tolerated
/// (logged), so an engine.cfg of a newer release still starts an older engine. Nothing is applied
/// from a file with a bad line or a value out of range: fails with `problem` naming it. Without a file nothing is
/// logged or changed.
pub fn bootstrap(gpa: std.mem.Allocator, io: std.Io, map: *std.process.Environ.Map, cfg_path: ?[]const u8, comptime Log: type, comptime libc: bool, problem: *[]const u8) error{Config}!void {
    for (&options, 0..) |*o, i| if (map.get(o.env)) |v| {
        resolved[i] = .{ .source = .env, .value = if (o.secret) "" else gpa.dupe(u8, v) catch "" };
    };
    defer ready.store(true, .release);
    const path = cfg_path orelse return;
    const fail = struct {
        fn at(g: std.mem.Allocator, p: *[]const u8, comptime fmt: []const u8, args: anytype) error{Config} {
            p.* = std.fmt.allocPrint(g, fmt, args) catch "engine.cfg: refused";
            return error.Config;
        }
    }.at;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch |e| return fail(gpa, problem, "TF_ENGINE_CFG={s}: cannot read it ({s})", .{ path, @errorName(e) });
    var rep: Report = .{ .path = gpa.dupe(u8, path) catch path, .bytes = text.len };
    std.crypto.hash.sha2.Sha256.hash(text, &rep.sha256, .{});
    const cfg = parseCfg(gpa, text) catch return fail(gpa, problem, "engine.cfg: out of memory", .{});
    if (cfg.bad_line != 0) return fail(gpa, problem, "{s}: line {d}: {s}", .{ path, cfg.bad_line, cfg.bad_why });
    rep.declared = cfg.declared;
    var buf: [400]u8 = undefined;
    var seen: [options.len]bool = @splat(false);
    for (cfg.lines) |l| {
        if (retiredByKey(l.key)) |r| if (!std.mem.eql(u8, l.value, "unset")) if (retiredRefusal(r, l.value, &buf)) |why| return fail(gpa, problem, "{s}: line {d}: {s}", .{ path, l.n, why });
        const o = byKey(l.key) orelse continue;
        const i = indexOf(o);
        if (seen[i]) return fail(gpa, problem, "{s}: line {d}: {s} is set twice", .{ path, l.n, o.env });
        seen[i] = true;
        if (o.secret) return fail(gpa, problem, "{s}: line {d}: {s} is a key; keys are not kept in engine.cfg", .{ path, l.n, o.env });
        if (std.mem.eql(u8, l.value, "unset")) continue;
        if (refusal(o, l.value, &buf)) |why| return fail(gpa, problem, "{s}: line {d}: {s}", .{ path, l.n, why });
    }
    var entries: std.ArrayList(Entry) = .empty;
    for (cfg.lines) |l| {
        const o = byKey(l.key) orelse {
            rep.unknown += 1;
            Log.line("engine.cfg line {d}: {s}: not in this engine's table (tolerated)", .{ l.n, l.key });
            continue;
        };
        if (std.mem.eql(u8, l.value, "unset")) {
            rep.unset += 1;
            continue;
        }
        const i = indexOf(o);
        entries.append(gpa, .{ .id = o.id, .value = l.value }) catch return fail(gpa, problem, "engine.cfg: out of memory", .{});
        if (map.get(o.env)) |have| {
            if (o.env_override) {
                rep.env_won += 1;
                resolved[i].other_source = .cfg;
                resolved[i].other = gpa.dupe(u8, l.value) catch "";
                Log.line("engine.cfg line {d}: {s} = {s} not applied: the environment has {s} (a debug or bench option: the environment wins)", .{ l.n, o.env, l.value, have });
                continue;
            }
            rep.cfg_won += 1;
            resolved[i].other_source = .env;
            resolved[i].other = if (o.secret) "" else gpa.dupe(u8, have) catch "";
            Log.line("engine.cfg line {d}: {s} = {s} replaces the environment's {s}", .{ l.n, o.env, l.value, have });
        }
        map.put(o.env, l.value) catch return fail(gpa, problem, "engine.cfg: out of memory", .{});
        if (libc) {
            const name = gpa.dupeSentinel(u8, o.env, 0) catch return fail(gpa, problem, "engine.cfg: out of memory", .{});
            const val = gpa.dupeSentinel(u8, l.value, 0) catch return fail(gpa, problem, "engine.cfg: out of memory", .{});
            _ = setenv(name.ptr, val.ptr, 1);
        }
        resolved[i] = .{ .source = .cfg, .value = gpa.dupe(u8, l.value) catch "", .other_source = resolved[i].other_source, .other = resolved[i].other };
        rep.applied += 1;
    }
    rep.entries = entries.items;
    report = rep;
    const hex = std.fmt.bytesToHex(rep.sha256[0..8].*, .lower);
    Log.line("engine.cfg {s}: sha256 {s}.., {d} bytes, schema {d}: applied {d} (replacing the environment's {d}), environment kept {d}, unset {d}, unknown {d}", .{ path, hex, rep.bytes, rep.declared, rep.applied, rep.cfg_won, rep.env_won, rep.unset, rep.unknown });
}
