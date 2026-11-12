//! A server's life: keys, the listening socket, the startup lines, the live line, and a clean stop on SIGTERM or SIGINT.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const auth = @import("auth.zig");
const cli = @import("cli.zig");
const log = @import("log.zig");
const live = @import("live.zig");
const listener_mod = @import("listener.zig");
const model_text = @import("model_text.zig");
const server_mod = @import("server.zig");
const prepare_gate = @import("prepare_gate.zig");
const clock = @import("clock.zig");
const admin_series = @import("admin_series.zig");
const recall_routes = @import("recall_routes.zig");
const Allocator = std.mem.Allocator;
const posix = std.posix;

var stop_requested: std.atomic.Value(bool) = .init(false);

fn onStop(_: posix.SIG) callconv(.c) void {
    stop_requested.store(true, .release);
}

/// SIGTERM and SIGINT stop the server cleanly; SIGHUP rereads the key file; SIGPIPE is ignored, as Python ignores it.
pub fn installSignals(keys: bool) void {
    const stop: posix.Sigaction = .{ .handler = .{ .handler = onStop }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.TERM, &stop, null);
    posix.sigaction(.INT, &stop, null);
    const ignore: posix.Sigaction = .{ .handler = .{ .handler = posix.SIG.IGN }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.PIPE, &ignore, null);
    if (keys) auth.installHangup();
}

pub fn stopping() bool {
    return stop_requested.load(.acquire);
}

/// The hard limit macOS reads as unlimited.
const rlim_infinity: u64 = std.c.RLIM.INFINITY;

/// The targets to try, in order: the hard limit when finite, else the fallbacks 65536 then 10240, first that works; never an unlimited soft limit, since a loop over every descriptor would do absurd work.
fn fillTargets(max: u64, out: *[2]u64) []const u64 {
    if (max != rlim_infinity) {
        out[0] = max;
        return out[0..1];
    }
    out[0] = 65536;
    out[1] = 10240;
    return out;
}

/// Best-effort raise of the soft open-file limit so idle keep-alives cannot exhaust a default 1024; a failure keeps the current limits, and null (caller logs) means nothing moved.
fn raiseOpenFileLimit() ?[2]u64 {
    const limit = posix.getrlimit(.NOFILE) catch return null;
    const cur: u64 = @intCast(limit.cur);
    if (cur == rlim_infinity) return null; // already unlimited
    var buf: [2]u64 = undefined;
    for (fillTargets(@intCast(limit.max), &buf)) |target| {
        if (target <= cur) continue;
        posix.setrlimit(.NOFILE, .{ .cur = @intCast(target), .max = limit.max }) catch continue;
        return .{ cur, target };
    }
    return null;
}

/// What a server needs besides its flags.
pub const Setup = struct {
    engine: api.Engine,
    text: model_text.Text,
    served: []const u8,
    /// The checkpoint's folder (structured output reads its tokenizer.json and config.json, /slide shows its shards).
    model_dir: []const u8 = "",
    /// Sampling defaults from generation_config.json and the flags, as a JSON object; null: greedy.
    sampling: ?json.Value = null,
    environ: ?*const std.process.Environ.Map = null,
    started: i96 = 0,
    /// The release, backend and GPU the open /health names.
    identity: server_mod.Identity = .{},
    /// A test build's TF_SLIDE_FORGET_FAIL: the fact whose check a delete fails (0: none, as in every release).
    forget_fail: u32 = 0,
    /// Called once the socket listens, with its port (tests read it).
    on_listen: ?*const fn (port: u16) void = null,
    /// The engine's orderly close (api.Opened.quiesce), run at the stop before the connection wait; null: the engine is
    /// closed only after the server.
    quiesce: ?Quiesce = null,
    /// Recall's record, attached to the engine already (--recall); null: off.
    recall: ?*api.recall_store.Store = null,
};

/// TENSORFOLD_CLOSE_MS's default: above a disk tier's flush at the stop (TF_SPILL_FLUSH_MS, 15,000) and below the
/// recipe's 30 s stop timeout less the 2 s connection wait.
pub const default_close_ms: u32 = 20_000;

/// The engine's orderly close at a stop (api.Opened.quiesce): every request ends and the engine's own last work runs
/// (a family's followers hear done), with nothing freed that the connection threads still call.
pub const Quiesce = struct {
    call: *const fn (ctx: *anyopaque) void,
    ctx: *anyopaque,
};

/// Tests: Closer.run finds no thread to run the orderly close on (as a process at its thread limit).
pub var test_no_thread: bool = false;

/// The engine's orderly close on a thread of its own, so a hung engine cannot hold the stop past its bound.
pub const Closer = struct {
    /// Its thread, left running when the close outlasts its bound (the process then ends without joining it).
    thread: ?std.Thread = null,
    returned: std.Io.Event = .unset,

    fn body(c: *Closer, io: std.Io, q: Quiesce) void {
        q.call(q.ctx);
        c.returned.set(io);
    }

    /// Runs `q` and waits for it at most `ms`: true once it returned (its thread joined, or run here when no thread
    /// could be made); false past the bound (the thread left in `thread`, still running).
    pub fn run(c: *Closer, io: std.Io, q: Quiesce, ms: u32) bool {
        const began = clock.nowNs(io);
        const spawned = if (@import("builtin").is_test and test_no_thread) error.SystemResources else std.Thread.spawn(.{}, body, .{ c, io, q });
        c.thread = spawned catch |e| {
            // here then, unbounded by TENSORFOLD_CLOSE_MS (the tier's flush keeps its own bound; the recipe's stop
            // timeout bounds the rest): ending without it would lose the flush and leave the followers waiting
            log.line("stop: no thread for the engine's orderly close ({s}): running it on this one", .{@errorName(e)});
            q.call(q.ctx);
            log.line("stop: the engine closed in {d} ms", .{@divTrunc(clock.nowNs(io) - began, std.time.ns_per_ms)});
            return true;
        };
        const end = began + @as(i96, ms) * std.time.ns_per_ms;
        while (!c.returned.isSet()) {
            const left = end - clock.nowNs(io);
            if (left <= 0) break;
            c.returned.waitTimeout(io, .{ .duration = .{ .raw = .fromNanoseconds(left), .clock = .awake } }) catch {};
        }
        if (!c.returned.isSet()) {
            log.line("stop: the engine did not close within {d} ms (TENSORFOLD_CLOSE_MS): ending without its orderly close", .{ms});
            return false;
        }
        c.thread.?.join();
        c.thread = null;
        log.line("stop: the engine closed in {d} ms", .{@divTrunc(clock.nowNs(io) - began, std.time.ns_per_ms)});
        return true;
    }
};

/// How a stop ended. clean: the engine closed (or has no orderly close) and every connection ended, so the caller
/// may free the server and the engine. forced: the caller ends the process without freeing what is still read.
pub const Stopped = enum { clean, forced };

/// What a stop runs besides the server.
pub const Stop = struct {
    quiesce: ?Quiesce = null,
    /// TENSORFOLD_CLOSE_MS: the engine's orderly close is waited for at most this long (0: none runs at the stop; the
    /// engine is closed after the connections end).
    close_ms: u32 = default_close_ms,
    /// The live line, ended after the engine's close.
    ticker: ?*live.Ticker = null,
    closer: Closer = .{},
};

/// The stop, once the accept loop has ended: the listener closed, the idle keep-alive connections ended, the engine's
/// orderly close (bounded by close_ms, on its own thread), the live line ended, then up to 2 s for the replies still
/// open (each ends once the close has ended its request) to end.
pub fn stopServing(io: std.Io, srv: *server_mod.Server, lis: listener_mod.Listener, stop: *Stop) Stopped {
    lis.close();
    srv.endIdle();
    const closed = if (stop.quiesce) |q| (stop.close_ms == 0 or stop.closer.run(io, q, stop.close_ms)) else true;
    if (stop.ticker) |t| t.finish();
    var waited: u32 = 0;
    while (srv.open_connections.load(.acquire) > 0 and waited < 40) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    const open = srv.open_connections.load(.acquire);
    if (open > 0) log.line("stop: {d} connection{s} still open after 2 s: ending without freeing what they read", .{ open, if (open == 1) "" else "s" });
    return if (closed and open == 0) .clean else .forced;
}

/// TENSORFOLD_CLOSE_MS: unset or empty is the default; anything but a whole number of milliseconds is refused.
fn closeMs(text: ?[]const u8) error{Invalid}!u32 {
    const t = std.mem.trim(u8, text orelse return default_close_ms, " ");
    if (t.len == 0) return default_close_ms;
    return std.fmt.parseInt(u32, t, 10) catch error.Invalid;
}

fn env(s: Setup, name: []const u8) ?[]const u8 {
    const m = s.environ orelse return null;
    return m.get(name);
}

/// Serves until a stop signal; the process exit status.
pub fn run(gpa: Allocator, io: std.Io, args: cli.Args, s: Setup) u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var store = auth.Store.init(gpa, args.api_key, env(s, "TENSORFOLD_API_KEY") orelse "", args.api_key_file, args.metrics_open, &problem) catch {
        std.debug.print("tensorfold: {s}\n", .{if (problem.len > 0) problem else "API key file cannot be opened safely"});
        return 1;
    };
    defer store.deinit();
    // the admin surface's own keys (read-only API on the same port): none, and every /admin/ path is the unknown-path 404.
    // A file the rules or the filesystem refuse does not stop the start: the store is closed and the line says why.
    var admin_store: auth.Store = undefined;
    var admin_on = false;
    if (env(s, "TENSORFOLD_ADMIN_KEY_FILE")) |path| if (path.len > 0) {
        admin_store = auth.Store.initAdmin(gpa, path, if (store.enabled()) &store else null, &problem) catch {
            std.debug.print("tensorfold: TENSORFOLD_ADMIN_KEY_FILE: the admin key store could not be made\n", .{});
            return 1;
        };
        admin_on = true;
        if (admin_store.valid) {
            log.line("admin API: read-only at /admin/api/v1 (routes, schema, config, status, ledger, requests, series, logs, keys) and the page at /admin/ui, its own key file", .{});
        } else log.line("admin API: closed, every /admin/api path answers 404: {s}", .{admin_store.refusal});
    };
    if (!store.enabled() and !auth.loopback(args.host)) log.line("warning: non-loopback server has no API key; requests are open", .{});
    var ids: std.ArrayList([]const u8) = .empty;
    for ([_][]const []const u8{ &.{s.served}, args.alias }) |group| for (group) |raw| {
        const id = std.mem.trim(u8, raw, " \t\r\n");
        if (id.len == 0) continue;
        for (ids.items) |seen| {
            if (std.mem.eql(u8, seen, id)) break;
        } else ids.append(a, id) catch return 1;
    };
    const salt = if (env(s, "TENSORFOLD_SEED_SALT")) |t| std.fmt.parseInt(i64, std.mem.trim(u8, t, " "), 10) catch {
        std.debug.print("tensorfold: TENSORFOLD_SEED_SALT={s}: an integer\n", .{t});
        return 1;
    } else 0;
    const keepalive_s = keepaliveSeconds(env(s, "TF_SSE_KEEPALIVE_S")) orelse {
        std.debug.print("tensorfold: TF_SSE_KEEPALIVE_S: seconds, a number >= 0\n", .{});
        return 1;
    };
    const close_ms = closeMs(env(s, "TENSORFOLD_CLOSE_MS")) catch {
        std.debug.print("tensorfold: TENSORFOLD_CLOSE_MS={s}: a whole number of milliseconds the engine's orderly close may take at a stop (0: none before the connections end)\n", .{env(s, "TENSORFOLD_CLOSE_MS").?});
        return 1;
    };
    const keepalive_ms_s: u32 = if (env(s, "TENSORFOLD_SSE_KEEPALIVE")) |t| std.fmt.parseInt(u32, std.mem.trim(u8, t, " "), 10) catch {
        std.debug.print("tensorfold: TENSORFOLD_SSE_KEEPALIVE={s}: whole seconds (0: off)\n", .{t});
        return 1;
    } else 0;
    const stats_every: u32 = if (env(s, "TENSORFOLD_STATS_EVERY")) |t| std.fmt.parseInt(u32, std.mem.trim(u8, t, " "), 10) catch {
        std.debug.print("tensorfold: TENSORFOLD_STATS_EVERY={s}: whole seconds (0: off)\n", .{t});
        return 1;
    } else 0;
    const loop_windows: u32 = if (env(s, "TENSORFOLD_LOOP_WINDOWS")) |t| std.fmt.parseInt(u32, std.mem.trim(u8, t, " "), 10) catch {
        std.debug.print("tensorfold: TENSORFOLD_LOOP_WINDOWS={s}: a whole number of windows (0: exact cycles only)\n", .{t});
        return 1;
    } else 10;
    const plain_specials = if (env(s, "TENSORFOLD_PLAIN_SPECIALS")) |t| blk: {
        const v = std.mem.trim(u8, t, " ");
        if (std.mem.eql(u8, v, "1")) break :blk true;
        if (v.len == 0 or std.mem.eql(u8, v, "0")) break :blk false;
        std.debug.print("tensorfold: TENSORFOLD_PLAIN_SPECIALS={s}: 1 (on) or 0 (off)\n", .{t});
        return 1;
    } else false;
    const reply_ids = if (env(s, "TENSORFOLD_REPLY_IDS")) |t| blk: {
        const v = std.mem.trim(u8, t, " ");
        if (std.mem.eql(u8, v, "1")) break :blk true;
        if (v.len == 0 or std.mem.eql(u8, v, "0")) break :blk false;
        std.debug.print("tensorfold: TENSORFOLD_REPLY_IDS={s}: 1 (on) or 0 (off)\n", .{t});
        return 1;
    } else false;
    const config: server_mod.Config = .{
        .served_name = s.served,
        .model_dir = s.model_dir,
        .model_ids = ids.items,
        .default_max_tokens = args.max_tokens,
        .sse_keepalive_s = keepalive_s,
        .enable_thinking = args.thinking,
        .reasoning_effort = args.reasoning_effort,
        .thinking_budget = args.thinking_budget,
        .loop_guard = args.loop_guard,
        .loop_windows = loop_windows,
        .keep_warm_s = args.keep_warm,
        .sse_keepalive_ms = std.math.mul(u32, keepalive_ms_s, 1000) catch std.math.maxInt(u32),
        .stats_every_s = stats_every,
        .default_sampling = s.sampling,
        .use_drafts = !args.no_drafts,
        .vision_urls = args.vision_urls,
        .seed_salt = salt,
        .request_log = env(s, "TENSORFOLD_REQUEST_LOG"),
        .plain_specials = plain_specials,
        .reply_ids = reply_ids,
        .dashboard = args.dashboard,
        .compact_at = if (args.compact_auto) .{ .auto = {} } else if (args.compact_fraction) |f| .{ .fraction = f } else null,
        .compact_keep = args.compact_keep,
        .compact_memory = args.compact_memory,
        .prepare_floor_gib = prepare_gate.floorGib(env(s, "TENSORFOLD_PREPARE_FLOOR_GIB")) catch {
            std.debug.print("tensorfold: TENSORFOLD_PREPARE_FLOOR_GIB={s}: a number of GiB, 0 for no gate\n", .{env(s, "TENSORFOLD_PREPARE_FLOOR_GIB").?});
            return 1;
        },
        .slide_graph = args.slide_graph,
        .slide_forget_fail = s.forget_fail,
        .cortheon = args.cortheon,
        .cortheon_search_key = env(s, "BRAVE_SEARCH_API_KEY"),
        .saved_conversations = args.learn_conversations,
        .identity = s.identity,
    };
    const srv = server_mod.Server.init(gpa, io, s.engine, s.text, config, if (store.enabled()) &store else null) catch {
        std.debug.print("tensorfold: the server could not start\n", .{});
        return 1;
    };
    defer srv.deinit();
    if (admin_on) srv.admin_keys = &admin_store;
    var sampler: ?*admin_series.Sampler = null;
    if (admin_on) if (admin_series.Sampler.create(gpa)) |smp| {
        sampler = smp;
        srv.series = smp;
        smp.requests = &srv.requests; // (the seats' KV share, read once a second under the table's short lock: connection threads share it, the engine thread never does)
        smp.start(io, s.engine);
    } else |_| log.line("admin API: no series (out of memory)", .{});
    defer if (sampler) |smp| {
        srv.series = null;
        smp.destroy();
    };
    srv.recall = s.recall;
    if (args.loop_guard and !srv.info.loop_guard) {
        std.debug.print("tensorfold: --loop-guard is not supported by this native engine\n", .{});
        return 2;
    }
    if (raiseOpenFileLimit()) |r| log.line("open-file limit raised: soft {d} -> {d}", .{ r[0], r[1] });
    const address = resolve(io, args.host, args.port) orelse {
        std.debug.print("tensorfold: cannot resolve --host {s}\n", .{args.host});
        return 1;
    };
    const lis = listener_mod.Listener.open(address) catch |e| {
        std.debug.print("tensorfold: cannot listen on {s}:{d} ({s})\n", .{ args.host, args.port, @errorName(e) });
        return 1;
    };
    installSignals(store.enabled());
    const port = lis.port();
    if (s.on_listen) |f| f(port);
    const window = srv.info.context_window;
    if (srv.info.startup.len > 0) log.line("{s}", .{srv.info.startup});
    if (srv.info.spill) |si| if (si.settings.asked) {
        var spill_line: [2048]u8 = undefined;
        var sw: std.Io.Writer = .fixed(&spill_line);
        si.settings.describe(&sw) catch {};
        log.line("{s}", .{sw.buffered()});
    };
    const now = std.Io.Clock.awake.now(io).toNanoseconds();
    const loaded = if (s.started > 0) @as(f64, @floatFromInt(now - s.started)) / 1e9 else 0;
    var window_text: [24]u8 = undefined;
    log.line("serving {s} at http://{s}:{d}/v1 (sampling: {s}; drafts: {s}; context: {s}; loaded in {d:.1}s)", .{
        s.served,                                                   args.host,                                                                                   port,   shownSampling(a, s.sampling),
        if (args.no_drafts or srv.info.plain_only) "off" else "on", if (window > 0) std.fmt.bufPrint(&window_text, "{d}", .{window}) catch "?" else "unlimited", loaded,
    });
    if (stats_every > 0) {
        srv.metrics.startLines(io, now);
        log.line("stats line on (TENSORFOLD_STATS_EVERY): at most every {d}s as requests finish, what finished since the last: tokens a round, drafts kept, and the draft head's chains by depth (kept of drafted, kept when reached)", .{stats_every});
    }
    if (args.thinking and std.mem.indexOf(u8, s.text.templateSource(), "enable_thinking") != null)
        log.line("thinking on (the chat template's default): replies reason in reasoning_content before the answer in content, and max_tokens counts both. --no-thinking turns it off; a request can send chat_template_kwargs {{\"enable_thinking\": false}}", .{});
    if (s.recall) |r| recall_routes.line(r);
    if (srv.shield) |sh| log.line("plain specials on (TENSORFOLD_PLAIN_SPECIALS): the text of {d} added tokens is ordinary text where users, tools or search results wrote it and in tool calls{s}; the template's own markers keep their ids", .{ sh.added, if (sh.replies) ", and in assistant turns (reply ids on)" else "" });
    if (reply_ids) log.line("reply ids on (TENSORFOLD_REPLY_IDS): a reply's think block opens and closes, and its tool calls open, only at their {d} marker tokens; the same text spelled in ordinary tokens stays words", .{srv.reply_pieces.len});
    if (srv.cortheon) |c| log.line("cortheon on: chat requests get Cortheon's reasoning program, web research before building, and cortheon_search/cortheon_fetch run here ({s}); a request can send \"cortheon\": false", .{c.engine});
    live.columns_env = env(s, "COLUMNS");
    var ticker: live.Ticker = .{ .engine = s.engine };
    const drawing = live.wanted(io, if (env(s, "TENSORFOLD_NO_LIVE")) |v| std.mem.eql(u8, v, "1") else false);
    if (drawing) ticker.start();
    const accept = std.Thread.spawn(.{}, server_mod.Server.serve, .{ srv, lis, &stop_requested }) catch return 1;
    accept.join();
    var stop: Stop = .{ .quiesce = s.quiesce, .close_ms = close_ms, .ticker = if (drawing) &ticker else null };
    // the engine still closing, or replies still open: end without freeing what they read (exit 0 either way)
    if (stopServing(io, srv, lis, &stop) == .forced) std.process.exit(0);
    return 0;
}

/// TF_SSE_KEEPALIVE_S: unset or blank keeps 15 seconds; zero disables stream heartbeats.
fn keepaliveSeconds(raw: ?[]const u8) ?f64 {
    const t = std.mem.trim(u8, raw orelse "", " \t");
    if (t.len == 0) return 15;
    const x = std.fmt.parseFloat(f64, t) catch return null;
    const limit = @as(f64, @floatFromInt(std.math.maxInt(i64))) / std.time.ns_per_s;
    return if (std.math.isFinite(x) and x >= 0 and x < limit) x else null;
}

test "stream keepalive interval accepts zero and rejects non-finite or overflowing durations" {
    const t = std.testing;
    try t.expectEqual(@as(?f64, 15), keepaliveSeconds(null));
    try t.expectEqual(@as(?f64, 15), keepaliveSeconds(" "));
    try t.expectEqual(@as(?f64, 0), keepaliveSeconds("0"));
    try t.expectEqual(@as(?f64, 0.1), keepaliveSeconds("0.1"));
    for ([_][]const u8{ "-1", "nan", "inf", "1e40", "no" }) |bad| try t.expectEqual(null, keepaliveSeconds(bad));
}

/// ``--host`` as an address: an IP literal, else a name the resolver knows (localhost first).
fn resolve(io: std.Io, host: []const u8, port: u16) ?std.Io.net.IpAddress {
    if (std.Io.net.IpAddress.parse(host, port)) |ip| return ip else |_| {}
    if (std.ascii.eqlIgnoreCase(host, "localhost")) return .{ .ip4 = .loopback(port) };
    return std.Io.net.IpAddress.resolve(io, host, port) catch null;
}

/// The serving line's sampling: greedy, or each default as ``name value``.
fn shownSampling(a: Allocator, sampling: ?json.Value) []const u8 {
    const v = sampling orelse return "greedy";
    if (v != .object) return "greedy";
    const t = v.get("temperature") orelse return "greedy";
    const temp: f64 = switch (t) {
        .float => |f| f,
        .int => |i| std.fmt.parseFloat(f64, i) catch 0,
        else => 0,
    };
    if (temp <= 0) return "greedy";
    var parts: std.ArrayList([]const u8) = .empty;
    for (v.object.keys(), v.object.values()) |k, x| {
        var buf: [40]u8 = undefined;
        const text = switch (x) {
            .float => |f| json.floatRepr(&buf, f),
            .int => |i| i,
            else => "?",
        };
        parts.append(a, std.fmt.allocPrint(a, "{s} {s}", .{ k, text }) catch return "greedy") catch return "greedy";
    }
    return std.mem.join(a, ", ", parts.items) catch "greedy";
}

test "TENSORFOLD_CLOSE_MS: unset or empty is 20,000 ms, a whole number is taken as given, anything else is refused" {
    try std.testing.expectEqual(@as(u32, 20_000), try closeMs(null));
    try std.testing.expectEqual(@as(u32, 20_000), try closeMs(" "));
    try std.testing.expectEqual(@as(u32, 15_000), try closeMs("15000"));
    try std.testing.expectEqual(@as(u32, 0), try closeMs("0"));
    for ([_][]const u8{ "-1", "20s", "1e4", "4294967296" }) |bad| try std.testing.expectError(error.Invalid, closeMs(bad));
}

test "the targets are the hard limit, or the fallbacks under an unlimited one" {
    var buf: [2]u64 = undefined;
    try std.testing.expectEqualSlices(u64, &.{524288}, fillTargets(524288, &buf));
    try std.testing.expectEqualSlices(u64, &.{ 65536, 10240 }, fillTargets(rlim_infinity, &buf));
}

test "the raise lifts a low soft limit, and a second raise does nothing" {
    const before = try posix.getrlimit(.NOFILE);
    defer posix.setrlimit(.NOFILE, before) catch {};
    if (@as(u64, @intCast(before.max)) <= 256) return error.SkipZigTest;
    try posix.setrlimit(.NOFILE, .{ .cur = 256, .max = before.max });
    const raised = raiseOpenFileLimit() orelse return error.TestExpectedRaise;
    try std.testing.expectEqual(@as(u64, 256), raised[0]);
    try std.testing.expectEqual(raised[1], @as(u64, @intCast((try posix.getrlimit(.NOFILE)).cur)));
    try std.testing.expect(raiseOpenFileLimit() == null);
}
