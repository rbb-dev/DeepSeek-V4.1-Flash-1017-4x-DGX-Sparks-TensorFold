//! /health answers 503 while the engine cannot serve (a fault it cannot recover from, such as a lost GPU context),
//! with keys on or off, and /metrics says so as tensorfold:healthy.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const auth = @import("auth.zig");
const Conn = @import("http_conn.zig").Conn;
const log = @import("log.zig");

const NoText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine whose status reports `healthy`.
const Engine = struct {
    healthy: bool = true,
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        out.* = .{ .healthy = e.healthy };
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct { status: u16, body: []u8 };

/// GET `path` through the routes, as a connection runs it.
fn get(srv: *server.Server, path: []const u8) !Reply {
    return getWith(srv, path, "");
}

/// GET `path` with `headers` (each line ending CRLF), through the key gate and the routes, as a connection runs it.
fn getWith(srv: *server.Server, path: []const u8, headers: []const u8) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    var head_buf: [256]u8 = undefined;
    const head = try std.fmt.bufPrint(&head_buf, "GET {s} HTTP/1.1\r\nHost: t\r\n{s}\r\n", .{ path, headers });
    if (std.c.write(pair[0], head.ptr, head.len) != @as(isize, @intCast(head.len))) return error.Closed;
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    if (srv.authenticate(&conn, arena.allocator())) routes.dispatch(srv, &conn, arena.allocator());
    var raw: std.ArrayList(u8) = .empty;
    defer raw.deinit(std.testing.allocator);
    var tmp: [4096]u8 = undefined;
    while (true) {
        var pollfd = [_]std.posix.pollfd{.{ .fd = pair[0], .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&pollfd, 200) == 0) break;
        const n = try std.posix.read(pair[0], &tmp);
        if (n == 0) break;
        try raw.appendSlice(std.testing.allocator, tmp[0..n]);
    }
    const split = std.mem.indexOf(u8, raw.items, "\r\n\r\n") orelse return error.BadReply;
    const code = std.fmt.parseInt(u16, raw.items["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    return .{ .status = code, .body = try std.testing.allocator.dupe(u8, raw.items[split + 4 ..]) };
}

fn testServer(e: *Engine, text: *NoText, keys: ?*auth.Store) !*server.Server {
    return server.Server.init(std.testing.allocator, std.testing.io, e.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, keys);
}

test "/health answers 503 unavailable while the engine cannot serve, keys on or off; 200 while it can" {
    var problem: []const u8 = "";
    var keys_arena = std.heap.ArenaAllocator.init(std.testing.allocator); // (a store lives as long as its server's process)
    defer keys_arena.deinit();
    var keyed = try auth.Store.init(keys_arena.allocator(), &.{"k1"}, "", null, false, &problem);
    for ([_]?*auth.Store{ null, &keyed }) |keys| {
        var text: NoText = .{};
        var e: Engine = .{};
        const srv = try testServer(&e, &text, keys);
        defer srv.deinit();
        const ok = try get(srv, "/health");
        defer std.testing.allocator.free(ok.body);
        try std.testing.expectEqual(@as(u16, 200), ok.status);
        try std.testing.expect(std.mem.indexOf(u8, ok.body, "\"ok\"") != null);
        e.healthy = false;
        const down = try get(srv, "/health");
        defer std.testing.allocator.free(down.body);
        try std.testing.expectEqual(@as(u16, 503), down.status);
        try std.testing.expectEqualStrings("{\"status\": \"unavailable\"}", down.body);
    }
}

test "/metrics says whether the engine can serve: tensorfold:healthy 1, then 0" {
    var text: NoText = .{};
    var e: Engine = .{};
    const srv = try testServer(&e, &text, null);
    defer srv.deinit();
    for ([_]bool{ true, false }) |healthy| {
        e.healthy = healthy;
        const r = try get(srv, "/metrics");
        defer std.testing.allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 200), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "# TYPE tensorfold:healthy gauge\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, r.body, if (healthy) "tensorfold:healthy 1\n" else "tensorfold:healthy 0\n") != null);
    }
}

test "/health names the engine's start and its uptime once the server knows the start" {
    var text: NoText = .{};
    var e: Engine = .{};
    const srv = try testServer(&e, &text, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const status_routes = @import("status_routes.zig");
    try std.testing.expect((try status_routes.healthValue(srv, arena.allocator(), false)).get("started_at") == null);
    const now: f64 = @as(f64, @floatFromInt(std.Io.Clock.real.now(std.testing.io).toNanoseconds())) / 1e9;
    srv.metrics.started_unix = now - 120;
    const h = try status_routes.healthValue(srv, arena.allocator(), false);
    const at = h.get("started_at") orelse return error.NoStartedAt;
    const up = h.get("uptime_s") orelse return error.NoUptime;
    try std.testing.expect(at == .float and @abs(at.float - (now - 120)) < 0.001);
    try std.testing.expect(up == .float and up.float >= 119 and up.float < 180);
}

/// The access lines printed while `paths` are fetched in turn.
fn accessLines(srv: *server.Server, paths: []const []const u8, buf: []u8) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    log.capture = &w;
    defer log.capture = null;
    for (paths) |p| std.testing.allocator.free((try get(srv, p)).body);
    return w.buffered();
}

test "the polls answered 200 print no access line; other requests and an unhealthy /health print theirs" {
    var text: NoText = .{};
    var e: Engine = .{};
    const srv = try testServer(&e, &text, null);
    defer srv.deinit();
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("", try accessLines(srv, &.{ "/health", "/metrics", "/v1/metrics", "/v1/models", "/api/v0/models" }, &buf));
    try std.testing.expectEqualStrings("[tensorfold] test \"GET /nope HTTP/1.1\" 404 -\n", try accessLines(srv, &.{"/nope"}, &buf));
    e.healthy = false;
    try std.testing.expectEqualStrings("[tensorfold] test \"GET /health HTTP/1.1\" 503 -\n", try accessLines(srv, &.{"/health"}, &buf));
}

test "TENSORFOLD_LOG_POLLS=1: the polls print their access lines" {
    var text: NoText = .{};
    var e: Engine = .{};
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, e.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
        .log_polls = true,
    }, null);
    defer srv.deinit();
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("[tensorfold] test \"GET /health HTTP/1.1\" 200 -\n" ++
        "[tensorfold] test \"GET /metrics HTTP/1.1\" 200 -\n" ++
        "[tensorfold] test \"GET /v1/models HTTP/1.1\" 200 -\n", try accessLines(srv, &.{ "/health", "/metrics", "/v1/models" }, &buf));
}

test "a request starts with its access line on, whatever the request before it on the connection left" {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = "GET /health HTTP/1.1\r\nHost: t\r\n\r\n";
    if (std.c.write(pair[0], head.ptr, head.len) != @as(isize, head.len)) return error.Closed;
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    conn.quiet_log = true; // a reply that was never sent
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqual(.ready, try conn.readRequest(arena.allocator(), false));
    try std.testing.expect(!conn.quiet_log);
}

test "keys on: /health gives a key holder its whole answer; without a valid key it says ok and nothing more" {
    var problem: []const u8 = "";
    var keys_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer keys_arena.deinit();
    var keyed = try auth.Store.init(keys_arena.allocator(), &.{"k1"}, "", null, false, &problem);
    var text: NoText = .{};
    var e: Engine = .{};
    const srv = try testServer(&e, &text, &keyed);
    defer srv.deinit();
    const holder = try getWith(srv, "/health", "Authorization: Bearer k1\r\n");
    defer std.testing.allocator.free(holder.body);
    try std.testing.expectEqual(@as(u16, 200), holder.status);
    try std.testing.expect(std.mem.indexOf(u8, holder.body, "\"model\": \"m\"") != null);
    for ([_][]const u8{ "", "Authorization: Bearer nope\r\n", "x-api-key: nope\r\n" }) |h| {
        const r = try getWith(srv, "/health", h);
        defer std.testing.allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 200), r.status);
        try std.testing.expectEqualStrings("{\"status\": \"ok\"}", r.body);
    }
}

test "a GET's body is drained: the next request on a kept connection is read as itself" {
    var text: NoText = .{};
    var e: Engine = .{};
    const srv = try testServer(&e, &text, null);
    defer srv.deinit();
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    // the body is a request line of its own: read as the next request, it would be served as one
    const two = "GET /health HTTP/1.1\r\nHost: t\r\nContent-Length: 9\r\n\r\nGET /nope" ++
        "GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n";
    if (std.c.write(pair[0], two.ptr, two.len) != @as(isize, two.len)) return error.Closed;
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqual(.ready, try conn.readRequest(arena.allocator(), false));
    routes.dispatch(srv, &conn, arena.allocator());
    try std.testing.expectEqual(.ready, try conn.readRequest(arena.allocator(), false));
    try std.testing.expectEqualStrings("GET", conn.method);
    try std.testing.expectEqualStrings("/v1/models", conn.path);
}
