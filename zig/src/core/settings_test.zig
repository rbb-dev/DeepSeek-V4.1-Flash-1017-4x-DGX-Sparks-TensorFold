//! The settings table and its start-up reader.
const std = @import("std");
const settings = @import("settings.zig");
const engine_api = @import("engine_api.zig");
const share = @import("lane_host_share.zig");

const O = settings.Option;

test "the table: ids rise, names are unique, every default is a value its own option takes" {
    var prev: u16 = 0;
    var buf: [200]u8 = undefined;
    for (&settings.options, 0..) |*o, i| {
        try std.testing.expect(o.id > prev);
        prev = o.id;
        for (settings.options[0..i]) |*p| {
            try std.testing.expect(!std.mem.eql(u8, p.env, o.env));
            try std.testing.expect(!std.mem.eql(u8, p.name, o.name));
        }
        try std.testing.expect(o.env.len > 0 and o.name.len > 0);
        if (o.default.len > 0) {
            if (settings.refusal(o, o.default, &buf)) |why| {
                std.debug.print("{s}: default {s}: {s}\n", .{ o.env, o.default, why });
                return error.TestUnexpectedResult;
            }
        }
        if (o.kind == .choice) try std.testing.expect(o.values.len > 0);
        try std.testing.expect(settings.byId(o.id).? == o);
        try std.testing.expect(settings.byEnv(o.env).? == o);
        try std.testing.expectEqual(i, settings.indexOf(o));
    }
    try std.testing.expect(settings.byId(0) == null);
    try std.testing.expect(settings.byId(65535) == null);
}

test "byKey: an id, an environment name or a name" {
    const o = settings.byEnv("TF_DS_SPILL_READERS").?;
    try std.testing.expect(settings.byKey(o.name) == o);
    try std.testing.expect(settings.byKey("TF_DS_SPILL_READERS") == o);
    var id: [8]u8 = undefined;
    try std.testing.expect(settings.byKey(std.fmt.bufPrint(&id, "{d}", .{o.id}) catch unreachable) == o);
    try std.testing.expect(settings.byKey("TF_NO_SUCH_THING") == null);
    try std.testing.expect(settings.byKey("99999") == null);
}

test "refusal: a range is a refusal by name, never a clamp" {
    var buf: [200]u8 = undefined;
    const readers = settings.byEnv("TF_DS_SPILL_READERS").?;
    try std.testing.expect(settings.refusal(readers, "4", &buf) == null);
    try std.testing.expect(settings.refusal(readers, "0", &buf) != null);
    try std.testing.expect(settings.refusal(readers, "65", &buf) != null);
    try std.testing.expect(settings.refusal(readers, "four", &buf) != null);
    try std.testing.expect(settings.refusal(readers, "", &buf) != null);
    const quota = settings.byEnv("TF_DS_SPILL_QUOTA").?;
    try std.testing.expect(settings.refusal(quota, "0.25", &buf) == null);
    try std.testing.expect(settings.refusal(quota, "1.5", &buf) != null);
    const flag = settings.byEnv("TF_DS_ENGRAM_AHEAD_PASS").?;
    try std.testing.expect(settings.refusal(flag, "1", &buf) == null);
    try std.testing.expect(settings.refusal(flag, "0", &buf) == null);
    try std.testing.expect(settings.refusal(flag, "yes", &buf) != null);
    const choice = settings.byEnv("TF_DS_DRAFT_HEAD").?;
    try std.testing.expect(settings.refusal(choice, "fp4", &buf) == null);
    try std.testing.expect(settings.refusal(choice, "fp16", &buf) != null);
}

/// The schema's number for `env`, as the parser's own type.
fn row(env: []const u8) *const O {
    return settings.byEnv(env) orelse @panic(env);
}

test "defaults: the spill settings' table, field by field, is the schema's" {
    const d: engine_api.Spill = .{};
    inline for (engine_api.Spill.ints) |n| {
        const o = row(n.name);
        try std.testing.expectEqual(settings.Kind.int, o.kind);
        try std.testing.expectEqual(settings.Check.tested, o.check);
        try std.testing.expectEqual(@as(u32, @intCast(try std.fmt.parseInt(u64, o.default, 10))), @field(d, n.field));
        try std.testing.expectEqual(@as(f64, @floatFromInt(n.min)), o.min.?);
        try std.testing.expectEqual(@as(f64, @floatFromInt(n.max)), o.max.?);
        try std.testing.expectEqualStrings(n.unit, o.unit);
    }
    inline for (engine_api.Spill.floats) |n| {
        const o = row(n.name);
        try std.testing.expectEqual(settings.Kind.float, o.kind);
        try std.testing.expectEqual(settings.Check.tested, o.check);
        try std.testing.expectEqual(@field(d, n.field), try std.fmt.parseFloat(f64, o.default));
        try std.testing.expectEqual(n.min, o.min.?);
        try std.testing.expectEqual(n.max, o.max.?);
    }
    try std.testing.expectEqual(d.gib, try std.fmt.parseFloat(f64, row("TF_DS_SPILL_GIB").default));
    try std.testing.expectEqual(@as(f64, engine_api.Spill.max_gib), row("TF_DS_SPILL_GIB").max.?);
    try std.testing.expectEqual(d.image_cache, std.mem.eql(u8, row("TF_DS_IMAGE_CACHE").default, "1"));
    // the ioprio default, parsed by the tier's own reading
    const io = engine_api.Spill.ioPrio(row("TF_DS_SPILL_IOPRIO").default).?;
    try std.testing.expectEqual(d.ioprio.class, io.class);
    try std.testing.expectEqual(d.ioprio.level, io.level);
    try std.testing.expectEqualStrings("", row("TF_DS_SPILL_DIR").default);
}

test "defaults: every spill name the tier reads has a row" {
    inline for (engine_api.Spill.env_names) |name| _ = row(name);
}

test "defaults: the prefill share and the rounds beside" {
    const p = share.Share{};
    try std.testing.expectEqual(p.p, try std.fmt.parseFloat(f64, row("TENSORFOLD_PREFILL_SHARE").default));
    try std.testing.expectEqual(share.Share.min_share, row("TENSORFOLD_PREFILL_SHARE").min.?);
    try std.testing.expectEqual(@as(f64, 1), row("TENSORFOLD_PREFILL_SHARE").max.?);
    // unset is "none", 0; a setting is 1 or more
    try std.testing.expectEqual(@as(?u32, 0), share.roundsFromText(null));
    try std.testing.expectEqual(@as(u32, 0), (share.Share{}).rounds);
    try std.testing.expectEqualStrings("", row("TF_DS_ROUNDS_BESIDE").default); // unset is none; 0 is no setting
    try std.testing.expectEqual(@as(f64, 1), row("TF_DS_ROUNDS_BESIDE").min.?);
}

test "parseCfg: comments, the schema line, key = value, and the first line that is neither" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const good = try settings.parseCfg(a, "# a release's settings\nschema 3\n\n  428 = 512  \nTF_DS_SPILL_READERS=8\nspill_dir = /ssd/x y\n");
    try std.testing.expectEqual(@as(usize, 0), good.bad_line);
    try std.testing.expectEqual(@as(u32, 3), good.declared);
    try std.testing.expectEqual(@as(usize, 3), good.lines.len);
    try std.testing.expectEqualStrings("428", good.lines[0].key);
    try std.testing.expectEqualStrings("512", good.lines[0].value);
    try std.testing.expectEqual(@as(usize, 4), good.lines[0].n);
    try std.testing.expectEqualStrings("TF_DS_SPILL_READERS", good.lines[1].key);
    try std.testing.expectEqualStrings("8", good.lines[1].value);
    try std.testing.expectEqualStrings("/ssd/x y", good.lines[2].value);
    const bad = try settings.parseCfg(a, "# ok\nthis is not a setting\n");
    try std.testing.expectEqual(@as(usize, 2), bad.bad_line);
    const empty_key = try settings.parseCfg(a, " = 4\n");
    try std.testing.expectEqual(@as(usize, 1), empty_key.bad_line);
    const bad_schema = try settings.parseCfg(a, "schema x\n");
    try std.testing.expectEqual(@as(usize, 1), bad_schema.bad_line);
}

/// A log that keeps what it was told.
const Cap = struct {
    var buf: [8192]u8 = undefined;
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

fn writeCfg(tmp: *std.testing.TmpDir, a: std.mem.Allocator, data: []const u8) ![]const u8 {
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "engine.cfg", .data = data });
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/engine.cfg", .{tmp.sub_path});
}

fn freshState() void {
    settings.resolved = @splat(.{});
    settings.report = .{};
    settings.ready.store(false, .release);
}

test "bootstrap without a file: nothing logged, nothing set, sources read from the environment" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    try map.put("TF_DS_SPILL_READERS", "6");
    freshState();
    Cap.reset();
    var problem: []const u8 = "";
    try settings.bootstrap(arena.allocator(), std.testing.io, &map, null, Cap, false, &problem);
    try std.testing.expectEqualStrings("", Cap.text());
    try std.testing.expect(settings.ready.load(.acquire));
    const r = settings.resolved[settings.indexOf(row("TF_DS_SPILL_READERS"))];
    try std.testing.expectEqual(settings.Source.env, r.source);
    try std.testing.expectEqualStrings("6", r.value);
    const q = settings.resolved[settings.indexOf(row("TF_DS_SPILL_QUOTA"))];
    try std.testing.expectEqual(settings.Source.default, q.source);
    try std.testing.expectEqual(@as(usize, 1), map.count());
}

test "bootstrap with a file: engine.cfg replaces the environment's value; an option it leaves out keeps the environment's; unknown keys are tolerated" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try writeCfg(&tmp, a, "schema 1\nTF_DS_SPILL_READERS = 8\nTF_DS_SPILL_SMALLS = 5\n60000 = 1\nfuture_option = 3\nTF_DS_SPILL_QUOTA = unset\n");
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    try map.put("TF_DS_SPILL_SMALLS", "9");
    try map.put("TF_DS_SPILL_OPEN_MS", "3000"); // an option engine.cfg does not name: the environment is the bootstrap
    freshState();
    Cap.reset();
    var problem: []const u8 = "";
    try settings.bootstrap(a, std.testing.io, &map, path, Cap, false, &problem);
    try std.testing.expectEqualStrings("8", map.get("TF_DS_SPILL_READERS").?);
    try std.testing.expectEqualStrings("5", map.get("TF_DS_SPILL_SMALLS").?); // the file's value, not the environment's 9
    try std.testing.expectEqualStrings("3000", map.get("TF_DS_SPILL_OPEN_MS").?);
    try std.testing.expect(map.get("TF_DS_SPILL_QUOTA") == null);
    const smalls = settings.resolved[settings.indexOf(row("TF_DS_SPILL_SMALLS"))];
    try std.testing.expectEqual(settings.Source.cfg, settings.resolved[settings.indexOf(row("TF_DS_SPILL_READERS"))].source);
    try std.testing.expectEqual(settings.Source.cfg, smalls.source);
    try std.testing.expectEqualStrings("5", smalls.value);
    try std.testing.expectEqual(settings.Source.env, smalls.other_source); // what lost, and to whom
    try std.testing.expectEqualStrings("9", smalls.other);
    try std.testing.expectEqual(settings.Source.env, settings.resolved[settings.indexOf(row("TF_DS_SPILL_OPEN_MS"))].source);
    try std.testing.expectEqual(@as(usize, 2), settings.report.applied);
    try std.testing.expectEqual(@as(usize, 1), settings.report.cfg_won);
    try std.testing.expectEqual(@as(usize, 0), settings.report.env_won);
    try std.testing.expectEqual(@as(usize, 2), settings.report.unknown);
    try std.testing.expectEqual(@as(usize, 1), settings.report.unset);
    try std.testing.expect(std.mem.indexOf(u8, Cap.text(), "TF_DS_SPILL_SMALLS") != null); // the replacement is logged
    try std.testing.expect(std.mem.indexOf(u8, Cap.text(), "sha256") != null);
    try std.testing.expect(std.mem.indexOf(u8, Cap.text(), "future_option") != null);
}

test "bootstrap with a file: the environment wins only for a debug or bench option, and says so" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const o = settings.byId(917).?; // tf_round_log, group debug
    try std.testing.expect(o.env_override);
    const path = try writeCfg(&tmp, a, try std.fmt.allocPrint(a, "{s} = 1\n", .{o.env}));
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    try map.put(o.env, "0");
    freshState();
    Cap.reset();
    var problem: []const u8 = "";
    try settings.bootstrap(a, std.testing.io, &map, path, Cap, false, &problem);
    try std.testing.expectEqualStrings("0", map.get(o.env).?);
    const r = settings.resolved[settings.indexOf(o)];
    try std.testing.expectEqual(settings.Source.env, r.source);
    try std.testing.expectEqual(settings.Source.cfg, r.other_source);
    try std.testing.expectEqualStrings("1", r.other);
    try std.testing.expectEqual(@as(usize, 1), settings.report.env_won);
    try std.testing.expectEqual(@as(usize, 0), settings.report.applied);
    try std.testing.expect(std.mem.indexOf(u8, Cap.text(), "debug or bench") != null);
}

test "bootstrap records what the file set, by option id, for the history" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try writeCfg(&tmp, a, "TF_DS_SPILL_READERS = 8\ntf_ds_spill_smalls = 5\nTF_DS_SPILL_QUOTA = unset\nfuture = 1\n");
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    freshState();
    Cap.reset();
    var problem: []const u8 = "";
    try settings.bootstrap(a, std.testing.io, &map, path, Cap, false, &problem);
    const e = settings.report.entries; // a line that sets a value for a known option; not `unset`, not an unknown key
    try std.testing.expectEqual(@as(usize, 2), e.len);
    try std.testing.expectEqual(row("TF_DS_SPILL_READERS").id, e[0].id);
    try std.testing.expectEqualStrings("8", e[0].value);
    try std.testing.expectEqual(row("TF_DS_SPILL_SMALLS").id, e[1].id);
    try std.testing.expectEqualStrings("5", e[1].value);
}

test "check: every refusal carries a code, and its message is refusal's text" {
    var b1: [400]u8 = undefined;
    var b2: [400]u8 = undefined;
    const ints = row("TF_DS_SPILL_READERS"); // 1..64
    const Case = struct { o: *const O, v: []const u8, code: settings.Code };
    const cases = [_]Case{
        .{ .o = ints, .v = "", .code = .empty },
        .{ .o = ints, .v = "4\n", .code = .control_character },
        .{ .o = ints, .v = "four", .code = .not_a_number },
        .{ .o = ints, .v = "0", .code = .below_minimum },
        .{ .o = ints, .v = "65", .code = .above_maximum },
        .{ .o = row("TF_DS_SPILL_OFF_REASON"), .v = "", .code = .empty },
        .{ .o = settings.byId(917).?, .v = "2", .code = .not_0_or_1 },
    };
    for (cases) |c| {
        const r = settings.check(c.o, c.v, &b1).?;
        try std.testing.expectEqual(c.code, r.code);
        try std.testing.expectEqualStrings(settings.refusal(c.o, c.v, &b2).?, r.message);
    }
    try std.testing.expect(settings.check(ints, "8", &b1) == null);
    try std.testing.expect(settings.refusal(ints, "8", &b1) == null);
    // a float option: not finite and not a number are told apart
    var f: ?*const O = null;
    for (&settings.options) |*o| if (o.kind == .float) {
        f = o;
        break;
    };
    try std.testing.expectEqual(settings.Code.not_a_number, settings.check(f.?, "x", &b1).?.code);
    try std.testing.expectEqual(settings.Code.not_finite, settings.check(f.?, "inf", &b1).?.code);
    // an enum
    for (&settings.options) |*o| if (o.kind == .choice and o.values.len > 0 and o.values[0].len > 0) {
        try std.testing.expectEqual(settings.Code.not_a_choice, settings.check(o, "zz-not-a-choice", &b1).?.code);
        break;
    };
}

test "the schema's new fields: the environment overrides only the debug and bench rows; a row an agent may set has bounds; the pool and arena are never applied at once" {
    for (&settings.options) |*o| {
        const dbg = std.mem.eql(u8, o.group, "debug") or o.id == 454 or o.id == 920;
        try std.testing.expectEqual(dbg, o.env_override);
        if (o.who.len > 0) for (o.who) |w| if (w == .agent) {
            try std.testing.expect(o.agent != null and o.min != null and o.max != null);
        };
        if (o.agent) |ag| try std.testing.expect(ag.min <= ag.max and ag.max_ttl_s > 0);
    }
    try std.testing.expect(!settings.byId(504).?.now_ok); // the KV pool
    try std.testing.expect(!settings.byId(502).?.now_ok); // the arena
    try std.testing.expect(settings.byId(122).?.now_ok);
    try std.testing.expect(settings.tabs.len == 7);
    // every row's tab is a tab the schema lists
    for (&settings.options) |*o| {
        var found = false;
        for (&settings.tabs) |*t| if (std.mem.eql(u8, t.name, o.tab)) {
            found = true;
        };
        try std.testing.expect(found);
    }
}

test "bootstrap refuses by name: a missing file, a line that is no setting, a value out of range" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    var problem: []const u8 = "";
    freshState();
    try std.testing.expectError(error.Config, settings.bootstrap(a, std.testing.io, &map, ".zig-cache/tmp/none/engine.cfg", Cap, false, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "engine.cfg") != null);
    const p2 = try writeCfg(&tmp, a, "oops\n");
    try std.testing.expectError(error.Config, settings.bootstrap(a, std.testing.io, &map, p2, Cap, false, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "line 1") != null);
    const p3 = try writeCfg(&tmp, a, "TF_DS_SPILL_READERS = 65\n");
    try std.testing.expectError(error.Config, settings.bootstrap(a, std.testing.io, &map, p3, Cap, false, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "TF_DS_SPILL_READERS") != null);
    try std.testing.expect(map.get("TF_DS_SPILL_READERS") == null); // nothing applied from a refused file
}

test "bootstrap as the server runs it (libc): a value from engine.cfg reaches the process environment, so the real build compiles this path" {
    const c = struct {
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (std.c.getenv("TF_DS_SPILL_READERS") != null) return error.SkipZigTest; // the caller's own setting: left alone
    defer _ = c.unsetenv("TF_DS_SPILL_READERS");
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try writeCfg(&tmp, a, "schema 1\nTF_DS_SPILL_READERS = 7\n");
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    freshState();
    Cap.reset();
    var problem: []const u8 = "";
    try settings.bootstrap(a, std.testing.io, &map, path, Cap, true, &problem);
    try std.testing.expectEqualStrings("7", std.mem.span(std.c.getenv("TF_DS_SPILL_READERS").?));
}

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

test "cfgDigest: 0 without a file; with one it follows what the file set, not its comments or its order, and a rank that read another file differs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    var problem: []const u8 = "";
    freshState();
    try settings.bootstrap(a, std.testing.io, &map, null, Cap, false, &problem);
    try std.testing.expectEqual(@as(u64, 0), settings.cfgDigest());
    const digest = struct {
        fn of(t: *std.testing.TmpDir, al: std.mem.Allocator, text: []const u8) !u64 {
            const path = try writeCfg(t, al, text);
            var m = std.process.Environ.Map.init(std.testing.allocator);
            defer m.deinit();
            freshState();
            var pr: []const u8 = "";
            try settings.bootstrap(al, std.testing.io, &m, path, Cap, false, &pr);
            return settings.cfgDigest();
        }
    }.of;
    const base = try digest(&tmp, a, "TF_DS_SPILL_READERS = 8\ntf_ds_spill_smalls = 5\n");
    try std.testing.expect(base != 0);
    try std.testing.expectEqual(base, try digest(&tmp, a, "# c\nfuture_key = 1\nschema 1\nTF_DS_SPILL_READERS = 8\ntf_ds_spill_smalls = 5\n")); // an unknown key and the schema line do not count
    try std.testing.expectEqual(base, try digest(&tmp, a, "tf_ds_spill_smalls = 5\n# c\nTF_DS_SPILL_READERS = 8\n")); // order and comments do not count
    try std.testing.expect(base != try digest(&tmp, a, "TF_DS_SPILL_READERS = 9\ntf_ds_spill_smalls = 5\n"));
    try std.testing.expect(base != try digest(&tmp, a, "TF_DS_SPILL_READERS = 8\n"));
    try std.testing.expect(try digest(&tmp, a, "# a file that sets nothing\n") != 0); // a rank that read a file differs from one that read none
}

test "rows whose parser clamps or refuses carry the parser's own bounds (read from the parsers, 11-09)" {
    const Want = struct { env: []const u8, kind: settings.Kind, min: ?f64, max: ?f64, default: []const u8 };
    const want = [_]Want{
        .{ .env = "TF_DS_ROCI_BACKSTOP_US", .kind = .int, .min = 10, .max = 2500, .default = "100" }, // roci_engine.zig backstopUs
        .{ .env = "TF_DS_ROCI_HOUSEKEEP", .kind = .int, .min = 1, .max = 4294967295, .default = "16" }, // @max(1, envU32(..))
        .{ .env = "TF_DS_ROCI_IDLE_PARK_US", .kind = .int, .min = 0, .max = 4294967295, .default = "2000" }, // on by default (speed_defaults)
        .{ .env = "TENSORFOLD_MEMORY_RESERVE_GIB", .kind = .float, .min = 2, .max = null, .default = "" }, // cuda_memory.reserveBytes: g >= 2
        .{ .env = "TF_CUDA_DEVICE", .kind = .int, .min = 0, .max = 4294967295, .default = "0" }, // envNumber: u32
        .{ .env = "TF_DS_ARENA_GIB", .kind = .int, .min = 0, .max = null, .default = "0" }, // parseInt(usize)
        .{ .env = "TF_VIDEO_MAX_FRAMES", .kind = .int, .min = 1, .max = null, .default = "32" },
        .{ .env = "TF_VIDEO_MAX_SIDE", .kind = .int, .min = 32, .max = null, .default = "768" },
    };
    var buf: [300]u8 = undefined;
    for (want) |w| {
        const o = settings.byEnv(w.env) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(w.kind, o.kind);
        try std.testing.expectEqual(w.min, o.min);
        try std.testing.expectEqual(w.max, o.max);
        try std.testing.expectEqualStrings(w.default, o.default);
        // the bounds are the schema's refusal: one inside is taken, one outside is refused
        if (w.min) |lo| if (lo >= 1) {
            var t: [24]u8 = undefined;
            const below = std.fmt.bufPrint(&t, "{d}", .{lo - 1}) catch unreachable;
            try std.testing.expect(settings.refusal(o, below, &buf) != null);
        };
    }
}
