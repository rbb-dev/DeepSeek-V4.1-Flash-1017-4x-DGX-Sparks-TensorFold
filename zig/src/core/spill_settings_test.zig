//! The disk tier's settings, read once from the environment by `Spill.fromEnv`: a value that does not parse or is out
//! of range turns the tier off naming it (`off`), never clamped, and the engine starts either way.
const std = @import("std");
const api = @import("engine_api.zig");
const Spill = api.Spill;

/// An environment for tests: name=value pairs, the first match wins.
const Env = struct {
    pairs: []const [2][]const u8,
    pub fn get(e: Env, name: [:0]const u8) ?[]const u8 {
        for (e.pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

fn parse(buf: []u8, pairs: []const [2][]const u8) Spill {
    return Spill.fromEnv(Env{ .pairs = pairs }, buf);
}

const on = [_][2][]const u8{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "/spill" } };

test "nothing set, TF_DS_SPILL_GIB=0 or only scheduling names: the tier is not asked for, and the engine is unchanged" {
    var buf: [256]u8 = undefined;
    for ([_][]const [2][]const u8{
        &.{},
        &.{ .{ "TF_DS_SPILL_GIB", "0" }, .{ "TF_DS_SPILL_DIR", "/spill" } },
        &.{ .{ "TF_SPILL_TURNS", "9" }, .{ "TF_DS_SPILL_RESTORES", "3" } },
        &.{.{ "TF_DS_SPILL_GIB", "" }},
    }) |pairs| {
        const s = parse(&buf, pairs);
        try std.testing.expect(!s.asked);
        try std.testing.expect(s.off == null);
        try std.testing.expect(!s.enabled());
    }
}

test "TF_DS_SPILL_GIB and TF_DS_SPILL_DIR set: on, with every default" {
    var buf: [256]u8 = undefined;
    const s = parse(&buf, &on);
    try std.testing.expect(s.asked);
    try std.testing.expect(s.off == null);
    try std.testing.expect(s.enabled());
    try std.testing.expectEqualStrings("/spill", s.dir.?);
    try std.testing.expectEqual(@as(f64, 256), s.gib);
    try std.testing.expectEqual(@as(f64, 50), s.min_free_gib);
    try std.testing.expectEqual(@as(u32, 4), s.turns);
    try std.testing.expectEqual(@as(f64, 0), s.quota);
    try std.testing.expectEqual(@as(u32, 8192), s.min_tokens);
    try std.testing.expectEqual(@as(u32, 128), s.staging_mib);
    try std.testing.expectEqual(@as(u32, 15_000), s.flush_ms);
    try std.testing.expectEqual(@as(u32, 40), s.load_ms);
    try std.testing.expectEqual(@as(u32, 30_000), s.load_wait_ms);
    try std.testing.expectEqual(@as(u32, 5_000), s.defer_ms);
    try std.testing.expectEqual(@as(u32, 25), s.defer_retry_ms);
    try std.testing.expectEqual(@as(u32, 0), s.defer_min);
    try std.testing.expectEqual(@as(u32, 2), s.pump_ms);
    try std.testing.expectEqual(@as(u32, 2), s.poll_ms);
    try std.testing.expectEqual(@as(u32, 4), s.smalls);
    try std.testing.expectEqual(@as(u32, 2), s.restores);
    try std.testing.expectEqual(@as(u32, 4), s.readers);
    try std.testing.expectEqual(@as(f64, 16), s.backlog_gib);
    try std.testing.expectEqual(@as(u32, 1), s.keep_builds);
    try std.testing.expectEqual(@as(u32, 300_000), s.open_ms);
    try std.testing.expectEqual(Spill.IoPrio{ .class = .be, .level = 7 }, s.ioprio);
    try std.testing.expectEqual(@as(u32, 2), s.land_slots);
    try std.testing.expectEqual(@as(u32, 32), s.land_piece_mib);
    try std.testing.expectEqual(@as(u32, 8192), s.probe_tokens);
    try std.testing.expectEqual(@as(u32, 0), s.probe_every);
    try std.testing.expectEqual(@as(u32, 8), s.probe_rows);
    try std.testing.expect(s.image_cache);
}

test "every setting is read: a value other than its default reaches its field" {
    var buf: [256]u8 = undefined;
    const s = parse(&buf, &(on ++ [_][2][]const u8{
        .{ "TF_DS_SPILL_MIN_FREE_GIB", "60.5" }, .{ "TF_SPILL_TURNS", "0" },               .{ "TF_DS_SPILL_QUOTA", "0.25" },
        .{ "TF_DS_SPILL_MIN_TOKENS", "4096" },   .{ "TF_DS_SPILL_STAGING_MIB", "256" },    .{ "TF_SPILL_FLUSH_MS", "20000" },
        .{ "TF_SPILL_LOAD_MS", "30" },           .{ "TF_SPILL_LOAD_WAIT_MS", "1000" },     .{ "TF_SPILL_DEFER_MS", "0" },
        .{ "TF_SPILL_DEFER_RETRY_MS", "10" },    .{ "TF_SPILL_DEFER_MIN", "2048" },        .{ "TF_SPILL_PUMP_MS", "3" },
        .{ "TF_SPILL_POLL_MS", "5" },            .{ "TF_DS_SPILL_SMALLS", "6" },           .{ "TF_DS_SPILL_RESTORES", "3" },
        .{ "TF_DS_SPILL_READERS", "8" },         .{ "TF_DS_SPILL_BACKLOG_GIB", "8" },      .{ "TF_DS_SPILL_KEEP_BUILDS", "0" },
        .{ "TF_DS_SPILL_OPEN_MS", "120000" },    .{ "TF_DS_SPILL_IOPRIO", "idle" },        .{ "TF_DS_SPILL_LAND_SLOTS", "3" },
        .{ "TF_DS_SPILL_LAND_PIECE_MIB", "64" }, .{ "TF_DS_SPILL_PROBE_TOKENS", "16384" }, .{ "TF_DS_SPILL_PROBE_EVERY", "100" },
        .{ "TF_DS_IMAGE_CACHE", "0" },           .{ "TF_DS_SPILL_PROBE_ROWS", "3" },
    }));
    try std.testing.expect(s.off == null);
    try std.testing.expectEqual(@as(f64, 60.5), s.min_free_gib);
    try std.testing.expectEqual(@as(u32, 0), s.turns);
    try std.testing.expectEqual(@as(f64, 0.25), s.quota);
    try std.testing.expectEqual(@as(u32, 4096), s.min_tokens);
    try std.testing.expectEqual(@as(u32, 256), s.staging_mib);
    try std.testing.expectEqual(@as(u32, 20_000), s.flush_ms);
    try std.testing.expectEqual(@as(u32, 30), s.load_ms);
    try std.testing.expectEqual(@as(u32, 1000), s.load_wait_ms);
    try std.testing.expectEqual(@as(u32, 0), s.defer_ms);
    try std.testing.expectEqual(@as(u32, 10), s.defer_retry_ms);
    try std.testing.expectEqual(@as(u32, 2048), s.defer_min);
    try std.testing.expectEqual(@as(u32, 3), s.pump_ms);
    try std.testing.expectEqual(@as(u32, 5), s.poll_ms);
    try std.testing.expectEqual(@as(u32, 6), s.smalls);
    try std.testing.expectEqual(@as(u32, 3), s.restores);
    try std.testing.expectEqual(@as(u32, 8), s.readers);
    try std.testing.expectEqual(@as(f64, 8), s.backlog_gib);
    try std.testing.expectEqual(@as(u32, 0), s.keep_builds);
    try std.testing.expectEqual(@as(u32, 120_000), s.open_ms);
    try std.testing.expectEqual(Spill.IoPrio{ .class = .idle, .level = 0 }, s.ioprio);
    try std.testing.expectEqual(@as(u32, 3), s.land_slots);
    try std.testing.expectEqual(@as(u32, 64), s.land_piece_mib);
    try std.testing.expectEqual(@as(u32, 16384), s.probe_tokens);
    try std.testing.expectEqual(@as(u32, 100), s.probe_every);
    try std.testing.expectEqual(@as(u32, 3), s.probe_rows);
    try std.testing.expect(!s.image_cache);
}

test "a value that does not parse, a negative one or one out of range turns the tier off, naming the setting and its value" {
    var buf: [256]u8 = undefined;
    const Bad = struct { name: []const u8, value: []const u8 };
    var cases: std.ArrayList(Bad) = .empty;
    defer cases.deinit(std.testing.allocator);
    const a = std.testing.allocator;
    var texts: std.ArrayList([]u8) = .empty;
    defer {
        for (texts.items) |t| a.free(t);
        texts.deinit(a);
    }
    inline for (Spill.ints) |n| { // whole numbers: not a number, a fraction, negative, below the least, past the most
        for ([_][]const u8{ "x", "1.5", "-1", "12abc" }) |v| try cases.append(a, .{ .name = n.name, .value = v });
        const past = try std.fmt.allocPrint(a, "{d}", .{@as(u64, n.max) + 1});
        try texts.append(a, past);
        try cases.append(a, .{ .name = n.name, .value = past });
        if (n.min > 0) {
            const below = try std.fmt.allocPrint(a, "{d}", .{n.min - 1});
            try texts.append(a, below);
            try cases.append(a, .{ .name = n.name, .value = below });
        }
    }
    inline for (Spill.floats) |n| {
        for ([_][]const u8{ "x", "-1", "nan", "inf", "-0.5" }) |v| try cases.append(a, .{ .name = n.name, .value = v });
        const past = try std.fmt.allocPrint(a, "{d}", .{n.max * 2});
        try texts.append(a, past);
        try cases.append(a, .{ .name = n.name, .value = past });
    }
    for ([_][]const u8{ "x", "-1", "nan", "inf", "2000000" }) |v| try cases.append(a, .{ .name = "TF_DS_SPILL_GIB", .value = v });
    for ([_][]const u8{ "be/8", "rt/1", "be", "be/x", "x" }) |v| try cases.append(a, .{ .name = "TF_DS_SPILL_IOPRIO", .value = v });
    try std.testing.expect(cases.items.len > 100);
    for (cases.items) |c| {
        const pairs = [_][2][]const u8{ .{ c.name, c.value }, on[0], on[1] };
        const s = parse(&buf, &pairs);
        try std.testing.expect(s.asked);
        try std.testing.expect(!s.enabled());
        const want = try std.fmt.allocPrint(a, "{s}={s}: ", .{ c.name, c.value });
        defer a.free(want);
        std.testing.expect(std.mem.startsWith(u8, s.off.?, want)) catch |e| {
            std.debug.print("case {s}={s}: off is \"{s}\"\n", .{ c.name, c.value, s.off.? });
            return e;
        };
    }
}

test "staging's share of a service cannot pass the service: TF_SPILL_PUMP_MS above TF_SPILL_LOAD_MS is off, not clamped" {
    var buf: [256]u8 = undefined;
    const s = parse(&buf, &(on ++ [_][2][]const u8{ .{ "TF_SPILL_PUMP_MS", "5" }, .{ "TF_SPILL_LOAD_MS", "4" } }));
    try std.testing.expectEqualStrings("TF_SPILL_PUMP_MS=5: more than TF_SPILL_LOAD_MS=4 (staging's share of a service cannot pass the service)", s.off.?);
    const same = parse(&buf, &(on ++ [_][2][]const u8{ .{ "TF_SPILL_PUMP_MS", "4" }, .{ "TF_SPILL_LOAD_MS", "4" } }));
    try std.testing.expect(same.off == null);
}

test "a request that may never wait for its restore is no setting: TF_SPILL_LOAD_WAIT_MS=0 is off, not clamped (the lane host's rule)" {
    var buf: [256]u8 = undefined;
    const s = parse(&buf, &(on ++ [_][2][]const u8{.{ "TF_SPILL_LOAD_WAIT_MS", "0" }}));
    try std.testing.expectEqualStrings("TF_SPILL_LOAD_WAIT_MS=0: milliseconds, a whole number from 1 to 3600000", s.off orelse "(on)");
    const one = parse(&buf, &(on ++ [_][2][]const u8{.{ "TF_SPILL_LOAD_WAIT_MS", "1" }}));
    try std.testing.expect(one.off == null);
}

test "TF_DS_SPILL_GIB without an absolute TF_DS_SPILL_DIR is off, named" {
    var buf: [256]u8 = undefined;
    const none = parse(&buf, &.{.{ "TF_DS_SPILL_GIB", "256" }});
    try std.testing.expect(none.asked);
    try std.testing.expectEqualStrings("TF_DS_SPILL_DIR: not set (TF_DS_SPILL_GIB=256 asks for the tier)", none.off.?);
    const empty = parse(&buf, &.{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "" } });
    try std.testing.expectEqualStrings("TF_DS_SPILL_DIR: not set (TF_DS_SPILL_GIB=256 asks for the tier)", empty.off.?);
    const relative = parse(&buf, &.{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "spill" } });
    try std.testing.expectEqualStrings("TF_DS_SPILL_DIR=spill: an absolute path", relative.off.?);
}

test "the recipe's reason turns the tier off as it is, before any setting is read" {
    var buf: [256]u8 = undefined;
    const s = parse(&buf, &.{ .{ "TF_DS_SPILL_OFF_REASON", "the watchdog's tier-off flag (2 restarts within 3600 s)" }, .{ "TF_DS_SPILL_GIB", "x" } });
    try std.testing.expect(s.asked);
    try std.testing.expectEqualStrings("the watchdog's tier-off flag (2 restarts within 3600 s)", s.off.?);
    const empty = parse(&buf, &(on ++ [_][2][]const u8{.{ "TF_DS_SPILL_OFF_REASON", "" }}));
    try std.testing.expect(empty.off == null);
}

test "derived: room on the store's file system = (1 + keep_builds) x gib + the floor, and the start-up line shows every value" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqual(@as(f64, 562), parse(&buf, &on).roomGib());
    try std.testing.expectEqual(@as(f64, 306), parse(&buf, &(on ++ [_][2][]const u8{.{ "TF_DS_SPILL_KEEP_BUILDS", "0" }})).roomGib());
    const a = std.testing.allocator;
    var line: std.Io.Writer.Allocating = .init(a);
    defer line.deinit();
    try parse(&buf, &on).describe(&line.writer);
    for ([_][]const u8{ "spill: on: ", "dir=/spill", "gib=256", "min_free_gib=50", "room=(1 + keep_builds 1) x gib 256 + min_free_gib 50 = 562 GiB", "turns=4", "ioprio=be/7", "probe_every=0", "land_piece_mib=32", "image_cache=on" }) |part|
        std.testing.expect(std.mem.indexOf(u8, line.written(), part) != null) catch |e| {
            std.debug.print("missing {s} in: {s}\n", .{ part, line.written() });
            return e;
        };
    var off: std.Io.Writer.Allocating = .init(a);
    defer off.deinit();
    try parse(&buf, &.{.{ "TF_DS_SPILL_GIB", "x" }}).describe(&off.writer);
    try std.testing.expect(std.mem.startsWith(u8, off.written(), "spill: off: TF_DS_SPILL_GIB=x: "));
    var not: std.Io.Writer.Allocating = .init(a);
    defer not.deinit();
    try parse(&buf, &.{}).describe(&not.writer);
    try std.testing.expectEqualStrings("spill: not asked (TF_DS_SPILL_GIB unset or 0)", not.written());
}

test "every name the parser reads is listed once, for capabilities and the recipe's pass-through" {
    for (Spill.env_names, 0..) |n, i| {
        try std.testing.expect(std.mem.startsWith(u8, n, "TF_DS_SPILL_") or std.mem.startsWith(u8, n, "TF_SPILL_") or std.mem.eql(u8, n, "TF_DS_IMAGE_CACHE"));
        for (Spill.env_names[0..i]) |m| try std.testing.expect(!std.mem.eql(u8, n, m));
    }
    try std.testing.expectEqual(@as(usize, Spill.ints.len + Spill.floats.len + 5), Spill.env_names.len); // + GIB, DIR, IOPRIO, OFF_REASON, IMAGE_CACHE
    try std.testing.expect(!@hasDecl(Spill, "unbound")); // the tier never refuses the engine's start
}

test "TF_DS_IMAGE_CACHE is read one way: the tier takes it as the server's picture keys do (untrimmed; only exactly \"0\" is off)" {
    var buf: [256]u8 = undefined;
    const pc = @import("prompt_cache.zig");
    for ([_][]const u8{ "0", " 0", "0 ", "", "1", "off" }) |v| {
        const s = parse(&buf, &(on ++ [_][2][]const u8{.{ "TF_DS_IMAGE_CACHE", v }}));
        try std.testing.expectEqual(pc.imageCacheOn(v), s.image_cache);
    }
}
