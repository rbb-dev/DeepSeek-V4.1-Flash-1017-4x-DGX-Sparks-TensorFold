//! The speed report: which of the engine's essential speed features run, and why not where they do not.
const std = @import("std");
const speed_defaults = @import("speed_defaults.zig");

// The report is built from the environment the parsers read (speed_defaults has applied by then), the start lines the engine
// logged when a switch could not run (noteLine, fed by the log hook: a line "TF_NAME=1: off: why"), and the runtime facts of the
// prompt cache and the SSD tier (Runtime). It is printed once as one block at the start (emit) and served by /status.

pub const Feature = struct {
    name: []const u8,
    env: []const u8,
    on: bool,
    /// Off because the operator said so (a setting naming another value): not a warning. Off and not asked: the engine could not
    /// run it, or nothing gave it: a warning.
    asked: bool,
    reason: []const u8,
    cost: []const u8,
};

/// What the host knows at the end of its start.
pub const Runtime = struct {
    cache_bytes: u64 = 0,
    spill_asked: bool = false,
    spill_on: bool = false,
    spill_off: ?[]const u8 = null,
    spill_gib: f64 = 0,
};

/// One line a speed_defaults row, an extra and the two caches, with room to grow; a table that outgrows it fails the build
/// (once a 61st row pushed the report past 64 and put() dropped the last lines, the SSD tier among them).
pub const max_items = 128;

pub const Report = struct {
    items: [max_items]Feature = undefined,
    n: usize = 0,
    text: [max_items][260]u8 = undefined,

    pub fn counts(r: *const Report) struct { on: usize, off: usize, unasked: usize } {
        var c: usize = 0;
        var u: usize = 0;
        for (r.items[0..r.n]) |f| {
            if (f.on) c += 1 else if (!f.asked) u += 1;
        }
        return .{ .on = c, .off = r.n - c, .unasked = u };
    }

    /// The block, one line each with its level first (INFO / WARN), the header's level WARN when any off feature was not asked for.
    pub fn writeBlock(r: *const Report, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const c = r.counts();
        try w.print("{s} speed features: {d} on, {d} off{s}\n", .{ if (c.unasked > 0) "WARN" else "INFO", c.on, c.off, if (c.unasked > 0) ", some of them nobody asked for (set the setting named to turn each on)" else "" });
        for (r.items[0..r.n]) |f| {
            if (f.on) {
                try w.print("INFO   ON  {s}: {s}\n", .{ f.name, f.reason });
            } else {
                try w.print("{s}   OFF {s}: {s}; without it: {s}\n", .{ if (f.asked) "INFO" else "WARN", f.name, f.reason, f.cost });
            }
        }
    }
};

/// kind 0: the switch cannot run ("off"); kind 1: it runs, in another form than asked ("instead: ...", e.g. a narrower kit variant).
const Note = struct { env: [64]u8 = undefined, env_len: u8 = 0, why: [200]u8 = undefined, why_len: u8 = 0, kind: u8 = 0 };
var notes: [48]Note = undefined;
var notes_used: std.atomic.Value(usize) = .init(0);

pub fn reset() void {
    notes_used.store(0, .release);
}

/// True when a log format is a switch line this module reads (the hook formats only those).
pub fn watches(comptime format: []const u8) bool {
    return format.len > 3 and isSwitchPrefix(format);
}

// the engine's switch names start with these three characters (written apart so the schema scan does not read it as a name)
fn isSwitchPrefix(s: []const u8) bool {
    return s.len >= 3 and s[0] == 'T' and s[1] == 'F' and s[2] == '_';
}

/// A start line "TF_NAME[=v]: off ..." or "TF_NAME: ...: off ..." says the named switch cannot run: kept for the report.
pub fn noteLine(line: []const u8) void {
    var i: usize = 0;
    while (i < line.len and (std.ascii.isUpper(line[i]) or std.ascii.isDigit(line[i]) or line[i] == '_')) i += 1;
    if (i < 4 or !isSwitchPrefix(line) or i >= line.len) return;
    const env = line[0..i];
    var rest = line[i..];
    if (rest[0] == '=') { // skip the value
        const colon = std.mem.indexOfScalar(u8, rest, ':') orelse return;
        rest = rest[colon..];
    }
    if (rest.len < 3 or rest[0] != ':') return;
    rest = std.mem.trimStart(u8, rest[1..], " ");
    const instead = std.mem.startsWith(u8, rest, "instead: ");
    const head = rest[0..@min(rest.len, 90)];
    const off = instead or std.mem.startsWith(u8, rest, "off") or std.mem.indexOf(u8, head, ": off") != null;
    if (!off) return;
    if (std.mem.indexOf(u8, rest, "no effect") != null) return;
    const idx = notes_used.fetchAdd(1, .acq_rel);
    if (idx >= notes.len) return;
    const n = &notes[idx];
    n.env_len = @intCast(@min(env.len, n.env.len));
    @memcpy(n.env[0..n.env_len], env[0..n.env_len]);
    n.kind = if (instead) 1 else 0;
    // the reason without the leading "off: " / "instead: "
    var why = rest;
    if (std.mem.startsWith(u8, why, "off: ")) why = why[5..];
    if (instead) why = why[9..];
    n.why_len = @intCast(@min(why.len, n.why.len));
    @memcpy(n.why[0..n.why_len], why[0..n.why_len]);
}

/// The reason the last off line of `env` gave, or null.
pub fn noted(env: []const u8) ?[]const u8 {
    return notedKind(env, 0);
}

/// What the last "instead: " line of `env` said (it runs, in another form than asked), or null.
pub fn notedInstead(env: []const u8) ?[]const u8 {
    return notedKind(env, 1);
}

fn notedKind(env: []const u8, kind: u8) ?[]const u8 {
    var i = @min(notes_used.load(.acquire), notes.len);
    while (i > 0) {
        i -= 1;
        const n = &notes[i];
        if (n.kind == kind and std.mem.eql(u8, n.env[0..n.env_len], env)) return n.why[0..n.why_len];
    }
    return null;
}

fn getenvLookup(name: []const u8) ?[]const u8 {
    var buf: [96:0]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const v = std.c.getenv(&buf) orelse return null;
    return std.mem.span(v);
}

/// An essential speed feature the defaults do not give: on only when the operator set it.
const Extra = struct { name: []const u8, env: []const u8, on_value: []const u8, default_on: bool = false, cost: []const u8 };
const extras = [_]Extra{
    .{ .name = "Host threads sleep for the GPU", .env = "TF_DS_CUDA_SCHED", .on_value = "blocking", .cost = "a core spins at 100 % per thread waiting for the GPU: the Sparks run hotter (speed unchanged)" },
    .{ .name = "Event waits sleep", .env = "TF_DS_CUDA_EVENTS", .on_value = "blocking", .cost = "rank 0's engine thread spins at 100 % of a core (speed unchanged)" },
    .{ .name = "Decode rings wait in the network card", .env = "TF_DS_RING_WAIT", .on_value = "1", .default_on = true, .cost = "the decode rings run busy proxy threads: hotter Sparks (speed level)" },
    .{ .name = "Decode rings' generation word", .env = "TF_DS_RING_WORD", .on_value = "1", .default_on = true, .cost = "the rings' waiting in the card cannot run: hotter Sparks (speed level)" },
    .{ .name = "Rocinante waits in the network card", .env = "TF_DS_ROCI_WAIT", .on_value = "1", .default_on = true, .cost = "Rocinante runs busy proxy threads: hotter Sparks (speed level)" },
};

comptime {
    if (speed_defaults.table.len + extras.len + 2 > max_items) @compileError("speed_report: max_items is smaller than the lines it must hold");
}

fn put(r: *Report, name: []const u8, env: []const u8, on: bool, asked: bool, cost: []const u8, comptime fmt: []const u8, args: anytype) void {
    if (r.n >= max_items) return;
    const reason = std.fmt.bufPrint(&r.text[r.n], fmt, args) catch r.text[r.n][0..0];
    r.items[r.n] = .{ .name = name, .env = env, .on = on, .asked = asked, .reason = reason, .cost = cost };
    r.n += 1;
}

/// Builds the report. `lookup` reads an environment name (the process's own in the engine; a table in tests).
pub fn build(r: *Report, rt: Runtime, lookup: *const fn ([]const u8) ?[]const u8) void {
    r.n = 0;
    for (speed_defaults.table, 0..) |d, i| {
        const cur = lookup(d.env);
        if (noted(d.env)) |why| {
            put(r, d.what, d.env, false, false, d.cost, "{s} cannot run: {s}", .{ d.env, why });
        } else if (cur) |v| {
            var also = false;
            for (d.also_on) |o| also = also or std.mem.eql(u8, v, o);
            if (std.mem.eql(u8, v, d.value) or also) {
                const how = if (speed_defaults.applied[i]) "on by default" else "set by the operator";
                if (notedInstead(d.env)) |ins|
                    put(r, d.what, d.env, true, false, d.cost, "{s}={s}, {s}; {s}", .{ d.env, v, how, ins })
                else
                    put(r, d.what, d.env, true, false, d.cost, "{s}={s}, {s}", .{ d.env, v, how });
            } else {
                put(r, d.what, d.env, false, true, d.cost, "{s} is set to {s} (the default is {s})", .{ d.env, v, d.value });
            }
        } else {
            put(r, d.what, d.env, false, false, d.cost, "{s} is not set and no default was applied (set it to {s})", .{ d.env, d.value });
        }
    }
    for (extras) |x| {
        const cur = lookup(x.env);
        if (noted(x.env)) |why| {
            put(r, x.name, x.env, false, false, x.cost, "{s} cannot run: {s}", .{ x.env, why });
        } else if (cur) |v| {
            if (std.mem.eql(u8, v, x.on_value) or (x.default_on and !std.mem.eql(u8, v, "0")))
                put(r, x.name, x.env, true, false, x.cost, "{s}={s}", .{ x.env, v })
            else
                put(r, x.name, x.env, false, true, x.cost, "{s} is set to {s}", .{ x.env, v });
        } else if (x.default_on) {
            put(r, x.name, x.env, true, false, x.cost, "{s} is on by default", .{x.env});
        } else {
            put(r, x.name, x.env, false, false, x.cost, "{s} is not set (set it to {s})", .{ x.env, x.on_value });
        }
    }
    // the prompt cache in memory and its SSD tier
    const gib: f64 = @floatFromInt(@as(u64, 1) << 30);
    if (rt.cache_bytes > 0) {
        put(r, "Prompt cache in memory", "--prompt-cache-gib", true, false, "returning conversations recompute their whole prompt", "{d:.1} GiB kept", .{@as(f64, @floatFromInt(rt.cache_bytes)) / gib});
    } else {
        put(r, "Prompt cache in memory", "--prompt-cache-gib", false, false, "returning conversations recompute their whole prompt", "its size is 0 (set --prompt-cache-gib, the fork's scripts use PROMPT_CACHE_GIB)", .{});
    }
    const ssd_cost = "returning conversations recompute their whole prompt (about 23 s for 147K tokens at the measured 6,494 tokens/s) instead of restoring it from disk";
    if (rt.spill_on) {
        put(r, "SSD prompt cache", "TF_DS_SPILL_GIB", true, false, ssd_cost, "up to {d:.0} GiB of kept conversations on rank 0's disk", .{rt.spill_gib});
    } else if (rt.spill_off) |why| {
        put(r, "SSD prompt cache", "TF_DS_SPILL_GIB", false, false, ssd_cost, "{s}", .{why});
    } else if (rt.spill_asked) {
        put(r, "SSD prompt cache", "TF_DS_SPILL_GIB", false, false, ssd_cost, "asked for but not running", .{});
    } else {
        const v = lookup("TF_DS_SPILL_GIB");
        if (v != null and std.mem.eql(u8, v.?, "0"))
            put(r, "SSD prompt cache", "TF_DS_SPILL_GIB", false, true, ssd_cost, "TF_DS_SPILL_GIB is set to 0", .{})
        else
            put(r, "SSD prompt cache", "TF_DS_SPILL_GIB", false, false, ssd_cost, "no size/directory given (the fork's scripts set it from free disk; set SPILL_GIB / SPILL_DIR, or TF_DS_SPILL_GIB and TF_DS_SPILL_DIR)", .{});
    }
}

var runtime: Runtime = .{};

pub fn setRuntime(rt: Runtime) void {
    runtime = rt;
}

/// The report as the engine runs it now (the process environment, the notes, the runtime facts).
pub fn current(r: *Report) void {
    build(r, runtime, getenvLookup);
}

/// Logs the block, once, at the end of the host's start: INFO for ON and for OFF the operator chose, WARN for the rest.
pub fn emit(rt: Runtime) void {
    setRuntime(rt);
    var r: Report = undefined;
    current(&r);
    var buf: [64 * 1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    r.writeBlock(&w) catch return;
    var it = std.mem.splitScalar(u8, w.buffered(), '\n');
    while (it.next()) |line| {
        if (line.len < 6) continue;
        if (std.mem.startsWith(u8, line, "WARN ")) std.log.warn("{s}", .{line[5..]}) else std.log.info("{s}", .{line[5..]});
    }
}

fn fakeLookup(name: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_DS_PF_SPLIT")) return "0"; // the operator turned it off
    if (std.mem.eql(u8, name, "TF_DS_QUANT_ONCE")) return "1"; // the default's value
    if (std.mem.eql(u8, name, "TF_DS_ATTN_U")) return "1"; // asked, but the start will say it cannot run
    if (std.mem.eql(u8, name, "TF_DS_PASS_ROWS")) return "2048"; // the operator set another value
    if (std.mem.eql(u8, name, "TF_DS_ROCI_STAGGER")) return "1";
    return null;
}

fn find(r: *const Report, env: []const u8) ?Feature {
    for (r.items[0..r.n]) |f| if (std.mem.eql(u8, f.env, env)) return f;
    return null;
}

test "the report: a table switch is on at its value, off and asked when the operator changed it, off and unasked when the start said it cannot run" {
    reset();
    noteLine("TF_DS_ATTN_U=1: off: the kit's extra set (aot-extra/) did not load (no file)");
    var r: Report = undefined;
    build(&r, .{}, fakeLookup);
    const split = find(&r, "TF_DS_PF_SPLIT").?;
    try std.testing.expect(!split.on and split.asked);
    try std.testing.expect(std.mem.indexOf(u8, split.reason, "set to 0") != null);
    const q = find(&r, "TF_DS_QUANT_ONCE").?;
    try std.testing.expect(q.on and !q.asked);
    const au = find(&r, "TF_DS_ATTN_U").?;
    try std.testing.expect(!au.on and !au.asked);
    try std.testing.expect(std.mem.indexOf(u8, au.reason, "kit's extra set") != null);
    const pr = find(&r, "TF_DS_PASS_ROWS").?;
    try std.testing.expect(!pr.on and pr.asked);
    // a name nobody set and the defaults did not give: off, unasked
    const ic = find(&r, "TF_DS_ISSUE_LEAD").?;
    try std.testing.expect(!ic.on and !ic.asked);
    // the stagger is not a default: set, it is on; unset, off and unasked
    try std.testing.expect(find(&r, "TF_DS_ROCI_STAGGER").?.on);
    try std.testing.expect(!find(&r, "TF_DS_DRAFT_4WAY").?.on and !find(&r, "TF_DS_DRAFT_4WAY").?.asked);
}

test "the report: TF_DS_MERGE_D is listed as wanted and not running, with the kit's reason, when the loaded kit lacks the twins" {
    const keep_applied = pinApplied(false);
    defer unpin(keep_applied);
    reset();
    noteLine("TF_DS_MERGE_D: off: the loaded kit's extra set has no _sparse_attn_merge_d: kit a02466e or later");
    var r: Report = undefined;
    build(&r, .{}, fakeLookup);
    const m = find(&r, "TF_DS_MERGE_D").?;
    try std.testing.expect(!m.on and !m.asked); // a WARN: nobody turned it off
    try expectHas(m.reason, "has no _sparse_attn_merge_d: kit a02466e or later");
    try std.testing.expect(m.cost.len > 0);
    var buf: [64 * 1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try r.writeBlock(&w);
    try expectHas(w.buffered(), "WARN   OFF decode attention merge in column blocks");
    // the same switch with its kit: on at its default; set to 0 by the operator: off and asked
    reset();
    build(&r, .{}, struct {
        fn f(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DS_MERGE_D")) "1" else null;
        }
    }.f);
    try std.testing.expect(find(&r, "TF_DS_MERGE_D").?.on);
    build(&r, .{}, struct {
        fn f(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DS_MERGE_D")) "0" else null;
        }
    }.f);
    const off = find(&r, "TF_DS_MERGE_D").?;
    try std.testing.expect(!off.on and off.asked);
}

/// `hay` holds `needle`, else both are printed (the test's failure then says what the report wrote).
fn expectHas(hay: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, hay, needle) != null) return;
    std.debug.print("\nexpected to contain: {s}\nactual:              {s}\n", .{ needle, hay });
    return error.TestUnexpectedResult;
}

/// speed_defaults.applied is process-wide state other tests set: a test that reads the "on by default" / "set by the operator" words
/// pins the MERGE_D entry itself (and restores it with the `defer` its caller writes: `const keep = pinApplied(x); defer unpin(keep);`).
fn mergeIndex() usize {
    for (speed_defaults.table, 0..) |d, i| if (std.mem.eql(u8, d.env, "TF_DS_MERGE_D")) return i;
    unreachable;
}
fn pinApplied(v: bool) bool {
    const keep = speed_defaults.applied[mergeIndex()];
    speed_defaults.applied[mergeIndex()] = v;
    return keep;
}
fn unpin(keep: bool) void {
    speed_defaults.applied[mergeIndex()] = keep;
}

fn mergeLookup(comptime v: ?[]const u8) *const fn ([]const u8) ?[]const u8 {
    return struct {
        fn f(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DS_MERGE_D")) v else null;
        }
    }.f;
}

test "the report: TF_DS_MERGE_D is ON at every width (1, 2, 3), ON with the width it runs when the kit lacks the asked one, OFF with the kit's reason when the kit has no width" {
    var r: Report = undefined;
    var buf: [64 * 1024]u8 = undefined;
    const keep_applied = pinApplied(false);
    defer unpin(keep_applied);
    // every value that runs it is ON (3 is the default, 1 = 128 columns, 2 = 64); 0 is the operator's choice
    reset();
    inline for (.{ "1", "2", "3" }) |v| {
        build(&r, .{}, mergeLookup(v));
        const f = find(&r, "TF_DS_MERGE_D").?;
        try std.testing.expect(f.on and !f.asked);
        try std.testing.expect(std.mem.indexOf(u8, f.reason, "not ") == null); // no "instead" text without the kit's line
    }
    build(&r, .{}, mergeLookup("0"));
    try std.testing.expect(!find(&r, "TF_DS_MERGE_D").?.on and find(&r, "TF_DS_MERGE_D").?.asked);
    // a kit with 128 (and 64) but not 16: the start logs "instead": still ON, and the reason says which width runs instead of which
    reset();
    noteLine("TF_DS_MERGE_D: instead: runs 128 columns a program, not 16: the loaded kit's extra set has no _sparse_attn_merge_d at 16 columns (it needs the kit regenerated with that width)");
    try std.testing.expect(noted("TF_DS_MERGE_D") == null); // not an off line
    build(&r, .{}, mergeLookup("3"));
    const inst = find(&r, "TF_DS_MERGE_D").?;
    try std.testing.expect(inst.on and !inst.asked);
    try expectHas(inst.reason, "runs 128 columns a program, not 16");
    var w = std.Io.Writer.fixed(&buf);
    try r.writeBlock(&w);
    // (writeBlock: "INFO   ON  {name}: {reason}", and the name is the table's `what`, its parenthesis included)
    try expectHas(w.buffered(), "INFO   ON  decode attention merge in column blocks (the kernel kit's extra set): TF_DS_MERGE_D=3, set by the operator; runs 128 columns a program, not 16");
    // the same line when the engine applied the default itself (applied[] is what tells the two apart)
    _ = pinApplied(true);
    build(&r, .{}, mergeLookup("3"));
    var w2 = std.Io.Writer.fixed(&buf);
    try r.writeBlock(&w2);
    try expectHas(w2.buffered(), "INFO   ON  decode attention merge in column blocks (the kernel kit's extra set): TF_DS_MERGE_D=3, on by default; runs 128 columns a program, not 16");
    _ = pinApplied(false);
    // a kit with no merge_d at any width (or none at all): OFF, unasked, with the kit's reason, whatever the value
    reset();
    noteLine("TF_DS_MERGE_D: off: the loaded kit's extra set has no _sparse_attn_merge_d at 128, 64 or 16 columns a program");
    build(&r, .{}, mergeLookup("3"));
    const none = find(&r, "TF_DS_MERGE_D").?;
    try std.testing.expect(!none.on and !none.asked);
    try expectHas(none.reason, "at 128, 64 or 16 columns a program");
    try std.testing.expect(notedInstead("TF_DS_MERGE_D") == null);
    // an "instead" line of an earlier start does not outlive reset()
    reset();
    build(&r, .{}, mergeLookup("3"));
    try std.testing.expect(find(&r, "TF_DS_MERGE_D").?.on);
}

test "the report: the SSD tier and the prompt cache say what turns them off and what it costs" {
    reset();
    var r: Report = undefined;
    build(&r, .{ .cache_bytes = 0, .spill_asked = false }, fakeLookup);
    const ssd = find(&r, "TF_DS_SPILL_GIB").?;
    try std.testing.expect(!ssd.on and !ssd.asked);
    try std.testing.expect(std.mem.indexOf(u8, ssd.reason, "SPILL_DIR") != null);
    try std.testing.expect(std.mem.indexOf(u8, ssd.cost, "recompute") != null);
    const pc = find(&r, "--prompt-cache-gib").?;
    try std.testing.expect(!pc.on and !pc.asked); // a size of 0 is a warning, not a choice
    build(&r, .{ .cache_bytes = 2 << 30, .spill_asked = true, .spill_on = true, .spill_gib = 256 }, fakeLookup);
    try std.testing.expect(find(&r, "TF_DS_SPILL_GIB").?.on);
    try std.testing.expect(find(&r, "--prompt-cache-gib").?.on);
    build(&r, .{ .cache_bytes = 2 << 30, .spill_asked = true, .spill_off = "TF_DS_SPILL_DIR: not set" }, fakeLookup);
    const off = find(&r, "TF_DS_SPILL_GIB").?;
    try std.testing.expect(!off.on and !off.asked);
    try std.testing.expect(std.mem.indexOf(u8, off.reason, "TF_DS_SPILL_DIR: not set") != null);
}

test "the report: the default-on transports are on unless a start line said off or the operator set 0" {
    reset();
    noteLine("TF_DS_RING_WAIT: off (set to something but 1): the rings run their proxy threads");
    var r: Report = undefined;
    build(&r, .{}, fakeLookup);
    try std.testing.expect(!find(&r, "TF_DS_RING_WAIT").?.on);
    try std.testing.expect(find(&r, "TF_DS_ROCI_WAIT").?.on);
}

test "noteLine keeps the last off line of a name and ignores lines that are not an off line" {
    reset();
    noteLine("TF_DS_IDX_U=1: a prompt chunk's indexer scores from packed keys");
    noteLine("not a switch line: off");
    try std.testing.expect(noted("TF_DS_IDX_U") == null);
    noteLine("TF_DS_IDX_U=1: off: the kit's extra set (aot-extra/) did not load (none)");
    noteLine("TF_DS_IDX_U=1: off on every rank: rank 2 cannot run it");
    try std.testing.expect(std.mem.indexOf(u8, noted("TF_DS_IDX_U").?, "rank 2") != null);
}

test "the log block names every off feature at WARN only when nobody asked for it" {
    reset();
    var r: Report = undefined;
    build(&r, .{}, fakeLookup);
    var buf: [64 * 1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try r.writeBlock(&w);
    const text = w.buffered();
    try std.testing.expect(std.mem.startsWith(u8, text, "WARN speed features:"));
    try std.testing.expect(std.mem.indexOf(u8, text, "WARN   OFF SSD prompt cache") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "INFO   OFF 4,096-row prompt passes") != null); // the operator set 2048
}
