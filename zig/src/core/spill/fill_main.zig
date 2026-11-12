//! tf-spill-fill: a synthetic store in a folder of its own, kept through the store's own write path (fill.zig) at
//! DeepSeek's geometry by default. Both modes time the store's open and print its counts.
//!
//!   fill   keep conversations until the store holds `--gib` GiB (the last ones make room in eviction order)
//!   probe  time a lookup of a kept conversation and one more conversation kept at the cap
//!
//! DIR must hold no other store's namespace (the tool refuses it): an engine's store is never pruned or filled.
const std = @import("std");
const core = @import("core");
const spill = core.spill;
const fill = spill.fill;

const usage =
    \\usage: tf-spill-fill fill|probe --dir DIR --gib N [--floor-gib G] [--turns T] [--sessions S] [--tokens N]
    \\       [--turn-tokens N] [--fork-every N] [--seed N] [--first N] [--staging-mib M] [--block-tokens N]
    \\       [--block-bytes N] [--small-bytes N] [--max-tokens N] [--lookup-session C]
    \\
;

const Args = struct {
    probe: bool = false,
    dir: ?[:0]const u8 = null,
    gib: u64 = 0,
    floor_gib: u64 = 50,
    turns: u32 = 4,
    sessions: ?u32 = null,
    plan: fill.Plan = .{ .sessions = 0, .tokens = 131072, .turn_tokens = 32768 },
    staging_mib: u64 = 128,
    geo: fill.Geometry = .{},
    lookup_session: u32 = 0,
};

fn parse(args: []const [:0]const u8) !Args {
    if (args.len < 2) return error.Usage;
    var a: Args = .{};
    if (std.mem.eql(u8, args[1], "probe")) a.probe = true else if (!std.mem.eql(u8, args[1], "fill")) return error.Usage;
    var i: usize = 2;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.Usage;
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--dir")) {
            a.dir = v;
            continue;
        }
        const n = std.fmt.parseInt(u64, v, 10) catch return error.Usage;
        const n32 = std.math.cast(u32, n) orelse return error.Usage;
        if (std.mem.eql(u8, k, "--gib")) a.gib = n else if (std.mem.eql(u8, k, "--floor-gib")) a.floor_gib = n else if (std.mem.eql(u8, k, "--turns")) a.turns = n32 else if (std.mem.eql(u8, k, "--sessions")) a.sessions = n32 else if (std.mem.eql(u8, k, "--tokens")) a.plan.tokens = n32 else if (std.mem.eql(u8, k, "--turn-tokens")) a.plan.turn_tokens = n32 else if (std.mem.eql(u8, k, "--fork-every")) a.plan.fork_every = n32 else if (std.mem.eql(u8, k, "--seed")) a.plan.seed = n else if (std.mem.eql(u8, k, "--first")) a.plan.first = n32 else if (std.mem.eql(u8, k, "--staging-mib")) a.staging_mib = n else if (std.mem.eql(u8, k, "--block-tokens")) a.geo.block_tokens = n32 else if (std.mem.eql(u8, k, "--block-bytes")) a.geo.block_bytes = n else if (std.mem.eql(u8, k, "--small-bytes")) a.geo.small_bytes = n else if (std.mem.eql(u8, k, "--max-tokens")) a.geo.max_tokens = n32 else if (std.mem.eql(u8, k, "--lookup-session")) a.lookup_session = n32 else return error.Usage;
    }
    const g = a.geo;
    if (a.dir == null or a.gib == 0 or g.block_tokens == 0 or g.block_bytes == 0 or g.small_bytes == 0) return error.Usage;
    if (a.plan.turn_tokens == 0 or a.plan.turn_tokens % g.block_tokens != 0 or a.plan.tokens < a.plan.turn_tokens or a.plan.tokens > g.max_tokens) return error.Usage;
    return a;
}

/// A conversation's bytes in the store, estimated: its blocks, and each turn's small state, ids and manifest.
fn sessionBytes(a: Args) u64 {
    const g = a.geo;
    const turns = a.plan.tokens / a.plan.turn_tokens;
    var n: u64 = @as(u64, a.plan.tokens / g.block_tokens) * g.block_bytes;
    var t: u64 = 1;
    while (t <= turns) : (t += 1) {
        const at = t * a.plan.turn_tokens;
        n += g.small_bytes + 4 * at + 256 + 80 * (at / g.block_tokens) + 64;
    }
    return n;
}

/// A line of /proc/self/status or /proc/meminfo ("VmRSS:", "Cached:") in KiB, or 0.
fn procKib(path: [:0]const u8, key: []const u8) u64 {
    var buf: [16384]u8 = undefined;
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return 0;
    var it = std.mem.splitScalar(u8, buf[0..@intCast(n)], '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        var f = std.mem.tokenizeAny(u8, line[key.len..], " \t");
        return std.fmt.parseInt(u64, f.next() orelse return 0, 10) catch 0;
    }
    return 0;
}

fn ms(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var a = parse(args) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    const dir = a.dir.?;
    if (try fill.foreignNamespace(gpa, dir, a.geo)) {
        std.debug.print("tf-spill-fill: {s} holds another store's namespace (native-spill-*): use a folder of its own\n", .{dir});
        return 2;
    }
    const cap = a.gib << 30;
    const per = sessionBytes(a);
    a.plan.sessions = a.sessions orelse @intCast(cap / per + 1);
    var staging: fill.Staging = .{};
    defer staging.deinit(gpa);
    const st = try staging.init(gpa, a.geo, a.staging_mib << 20);
    const cfg = fill.config(dir, a.geo, cap, a.floor_gib << 30, a.turns, st);
    const rss0 = procKib("/proc/self/status", "VmRSS:");
    const cached0 = procKib("/proc/meminfo", "Cached:");
    var t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const s = spill.Store.open(gpa, io, cfg) catch |e| {
        std.debug.print("tf-spill-fill: open of {s}: {t}\n", .{ dir, e });
        return 1;
    };
    defer s.close();
    const open_ms = ms(io, t0);
    var c = s.counts();
    std.debug.print("tf-spill-fill: open {d:.1} ms: points {d}, blocks {d}, stored {d:.2} GiB, recovered {d}, dropped {d}, quarantined {d}, deferred {d}\n", .{
        open_ms, c.points, c.blocks, @as(f64, @floatFromInt(c.stored_bytes)) / (1 << 30), c.recovered_points, c.recovered_dropped, c.quarantined, c.recover_deferred,
    });
    if (!a.probe) {
        std.debug.print("tf-spill-fill: filling {d} conversations of {d} tokens in turns of {d} (about {d:.1} MiB each, calculated) to a cap of {d} GiB\n", .{
            a.plan.sessions, a.plan.tokens, a.plan.turn_tokens, @as(f64, @floatFromInt(per)) / (1 << 20), a.gib,
        });
        t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        const got = try fill.fill(gpa, io, s, a.geo, a.plan);
        const fill_ms = ms(io, t0);
        c = s.counts();
        std.debug.print("tf-spill-fill: filled in {d:.1} s: queued {d}, already stored {d}, refused {d}; points {d}, blocks {d}, stored {d:.2} GiB, written {d:.2} GiB, pruned turns {d} forks {d} conversations {d}, direct_refused {d}\n", .{
            fill_ms / 1000,                                       got.queued,     got.stored,     got.refused,            c.points,         c.blocks, @as(f64, @floatFromInt(c.stored_bytes)) / (1 << 30),
            @as(f64, @floatFromInt(c.written_bytes)) / (1 << 30), c.pruned_turns, c.pruned_forks, c.pruned_conversations, c.direct_refused,
        });
    } else {
        const ids = try fill.sessionIds(gpa, a.plan, a.lookup_session);
        defer gpa.free(ids);
        t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        const hit = s.lookup(.{ .keyed = ids, .max_at = a.geo.max_tokens });
        const lookup_ms = ms(io, t0);
        std.debug.print("tf-spill-fill: lookup {d:.3} ms: {s} at {d} tokens\n", .{ lookup_ms, if (hit != null) "hit" else "miss", if (hit) |h| h.at else 0 });
        const before = s.counts();
        var plan = a.plan;
        plan.sessions = 1;
        plan.first = std.math.maxInt(u32) / 2; // a conversation no fill made
        t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        const got = try fill.fill(gpa, io, s, a.geo, plan);
        const keep_ms = ms(io, t0);
        c = s.counts();
        std.debug.print("tf-spill-fill: one conversation at the cap {d:.1} ms (its {d} turns kept and written, drained): queued {d}, refused {d}; pruned turns +{d} forks +{d} conversations +{d}; stored {d:.2} GiB\n", .{
            keep_ms,                                              got.queued + got.stored + got.refused,               got.queued, got.refused, c.pruned_turns - before.pruned_turns, c.pruned_forks - before.pruned_forks,
            c.pruned_conversations - before.pruned_conversations, @as(f64, @floatFromInt(c.stored_bytes)) / (1 << 30),
        });
    }
    std.debug.print("tf-spill-fill: store lock: wait max {d:.3} ms, hold max {d:.3} ms ({d} holds); VmRSS {d} -> {d} MiB; page cache (Cached) {d} -> {d} MiB\n", .{
        @as(f64, @floatFromInt(c.lock_wait.max_ns)) / 1e6, @as(f64, @floatFromInt(c.lock_hold.max_ns)) / 1e6, c.lock_hold.count,
        rss0 >> 10,                                        procKib("/proc/self/status", "VmRSS:") >> 10,      cached0 >> 10,
        procKib("/proc/meminfo", "Cached:") >> 10,
    });
    return 0;
}
