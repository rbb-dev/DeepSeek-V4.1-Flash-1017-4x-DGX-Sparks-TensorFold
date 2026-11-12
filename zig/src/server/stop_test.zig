//! Tests of the server's stop: at the stop signal idle keep-alive connections end, the engine's orderly close runs
//! before the connection wait, bounded by TENSORFOLD_CLOSE_MS, and a reply still streaming gets its end.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const serve = @import("serve.zig");
const listener_mod = @import("listener.zig");
const model_text = @import("model_text.zig");
const json = server.json;
const posix = std.posix;
const Threaded = std.Io.Threaded;
const gpa = std.testing.allocator;
const io = std.testing.io;

fn nowNs() i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn msSince(t: i96) i64 {
    return @intCast(@divTrunc(nowNs() - t, std.time.ns_per_ms));
}

/// Bytes as tokens; every prompt renders as "prompt" (6 tokens).
const TestText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
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

/// An engine that holds every request it takes (after one token "h", or none with `first_token` false: still in its
/// prompt pass) until a cancel or its quiesce ends it. Its quiesce counts itself, notes the server's open connections,
/// takes `delay_ms` (the pass's next chunk end) and, with `hang`, returns only once `released` is set.
const HoldEngine = struct {
    mutex: std.Io.Mutex = .init,
    held: [8]?Held = @splat(null),
    submitted: u32 = 0,
    quiesced: u32 = 0,
    open_at_quiesce: u32 = 0,
    srv: ?*server.Server = null,
    first_token: bool = true,
    delay_ms: u32 = 0,
    hang: bool = false,
    released: std.atomic.Value(bool) = .init(false),
    returned: std.atomic.Value(bool) = .init(false),

    const Held = struct { id: api.Id, sink: api.Sink };

    fn engine(e: *HoldEngine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn of(ctx: *anyopaque) *HoldEngine {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .lanes = 4, .context_window = 4096 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = of(ctx);
        e.mutex.lockUncancelable(io);
        if (e.quiesced > 0) {
            e.mutex.unlock(io);
            return error.Closed;
        }
        const slot = for (&e.held) |*h| {
            if (h.* == null) break h;
        } else {
            e.mutex.unlock(io);
            return error.Busy;
        };
        slot.* = .{ .id = id, .sink = sink };
        e.submitted += 1;
        e.mutex.unlock(io);
        if (e.first_token) {
            sink.event(sink.ctx, id, &.{ .prefilled = 0 });
            sink.event(sink.ctx, id, &.{ .tokens = &.{'h'} });
        }
    }

    /// Ends the held requests `id` names (every one: null), outside the lock.
    fn end(e: *HoldEngine, id: ?api.Id, reason: api.Reason, message: []const u8) void {
        var ending: [8]Held = undefined;
        var n: usize = 0;
        e.mutex.lockUncancelable(io);
        for (&e.held) |*h| if (h.*) |x| if (id == null or x.id == id.?) {
            ending[n] = x;
            n += 1;
            h.* = null;
        };
        e.mutex.unlock(io);
        for (ending[0..n]) |x| x.sink.event(x.sink.ctx, x.id, &.{ .finished = .{ .reason = reason, .message = message } });
    }

    fn cancel(ctx: *anyopaque, id: api.Id) void {
        of(ctx).end(id, .cancelled, "");
    }

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn quiesce(ctx: *anyopaque) void {
        const e = of(ctx);
        e.mutex.lockUncancelable(io);
        e.quiesced += 1;
        e.open_at_quiesce = if (e.srv) |s| s.open_connections.load(.acquire) else 0;
        e.mutex.unlock(io);
        if (e.delay_ms > 0) std.Io.sleep(io, .fromMilliseconds(e.delay_ms), .awake) catch {};
        while (e.hang and !e.released.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        e.end(null, .failed, api.LaneHost.stop_message); // as the lane host ends what the stop cuts
        e.returned.store(true, .release);
    }

    fn counts(e: *HoldEngine) struct { submitted: u32, quiesced: u32, open_at_quiesce: u32 } {
        e.mutex.lockUncancelable(io);
        defer e.mutex.unlock(io);
        return .{ .submitted = e.submitted, .quiesced = e.quiesced, .open_at_quiesce = e.open_at_quiesce };
    }
};

/// A client on its own socket, and what it has read.
const Client = struct {
    fd: posix.socket_t,
    got: std.ArrayList(u8) = .empty,

    fn connect(port: u16) !Client {
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const fd: posix.socket_t = @intCast(rc);
        errdefer _ = posix.system.close(fd);
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(port) };
        var storage: Threaded.PosixAddress = undefined;
        const len = Threaded.addressToPosix(&address, &storage);
        if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.ConnectFailed;
        return .{ .fd = fd };
    }

    fn deinit(c: *Client) void {
        _ = posix.system.close(c.fd);
        c.got.deinit(gpa);
    }

    fn send(c: *Client, bytes: []const u8) !void {
        var off: usize = 0;
        while (off < bytes.len) {
            const rc = posix.system.write(c.fd, bytes[off..].ptr, bytes.len - off);
            if (posix.errno(rc) != .SUCCESS or rc <= 0) return error.WriteFailed;
            off += @intCast(rc);
        }
    }

    /// One read into `got`, waiting at most `ms`: false when the server closed the connection or nothing came.
    fn readSome(c: *Client, ms: i64) !bool {
        var fds = [_]posix.pollfd{.{ .fd = c.fd, .events = posix.POLL.IN, .revents = 0 }};
        if (try posix.poll(&fds, @intCast(@max(ms, 0))) == 0) return false;
        var buf: [4096]u8 = undefined;
        const rc = posix.system.read(c.fd, &buf, buf.len);
        if (posix.errno(rc) != .SUCCESS) return error.ReadFailed;
        if (rc == 0) return false;
        try c.got.appendSlice(gpa, buf[0..@intCast(rc)]);
        return true;
    }

    /// Reads until `want` is in what it holds (true), or the connection closes or `ms` pass (false).
    fn readUntil(c: *Client, want: []const u8, ms: i64) !bool {
        const began = nowNs();
        while (std.mem.indexOf(u8, c.got.items, want) == null) {
            const left = ms - msSince(began);
            if (left <= 0 or !try c.readSome(left)) return std.mem.indexOf(u8, c.got.items, want) != null;
        }
        return true;
    }

    /// Reads until the server closes the connection (true), or `ms` pass first (false).
    fn readToEnd(c: *Client, ms: i64) !bool {
        const began = nowNs();
        while (true) {
            const left = ms - msSince(began);
            if (left <= 0) return false;
            var fds = [_]posix.pollfd{.{ .fd = c.fd, .events = posix.POLL.IN, .revents = 0 }};
            if (try posix.poll(&fds, @intCast(left)) == 0) return false;
            var buf: [4096]u8 = undefined;
            const rc = posix.system.read(c.fd, &buf, buf.len);
            if (posix.errno(rc) != .SUCCESS) return error.ReadFailed;
            if (rc == 0) return true;
            try c.got.appendSlice(gpa, buf[0..@intCast(rc)]);
        }
    }

    /// A whole response with a Content-Length body (a /health), read.
    fn readResponse(c: *Client, ms: i64) !bool {
        if (!try c.readUntil("\r\n\r\n", ms)) return false;
        const head_end = std.mem.indexOf(u8, c.got.items, "\r\n\r\n").? + 4;
        const at = std.ascii.findIgnoreCase(c.got.items[0..head_end], "content-length: ") orelse return error.NoLength;
        const value = c.got.items[at + "content-length: ".len ..];
        const n = try std.fmt.parseInt(usize, value[0..std.mem.indexOfScalar(u8, value, '\r').?], 10);
        const began = nowNs();
        while (c.got.items.len < head_end + n) {
            const left = ms - msSince(began);
            if (left <= 0 or !try c.readSome(left)) return false;
        }
        return true;
    }
};

/// The time (ns, the awake clock) a client's connection ended, read on its own thread.
const EndWatch = struct {
    at: std.atomic.Value(i64) = .init(0),
    fn run(w: *EndWatch, c: *Client) void {
        if (c.readToEnd(10_000) catch false) w.at.store(@intCast(nowNs()), .release);
    }
};

/// A server listening on a loopback port and its accept loop, as serve.run starts them.
const Running = struct {
    srv: *server.Server,
    lis: listener_mod.Listener,
    hold: *HoldEngine,
    stop: std.atomic.Value(bool) = .init(false),
    accept: ?std.Thread = null,
    port: u16 = 0,
    stopped: bool = false, // stopServing ran (it closed the listener)

    fn start(r: *Running) !void {
        r.port = r.lis.port();
        r.accept = try std.Thread.spawn(.{}, server.Server.serve, .{ r.srv, r.lis, &r.stop });
    }

    /// The stop signal, as SIGTERM sets it: the accept loop ends (within its 200 ms poll).
    fn signal(r: *Running) void {
        r.stop.store(true, .release);
        if (r.accept) |t| t.join();
        r.accept = null;
    }

    fn stopServing(r: *Running, stop: *serve.Stop) serve.Stopped {
        r.stopped = true;
        return serve.stopServing(io, r.srv, r.lis, stop);
    }

    /// The server freed once no connection thread reads it, whether or not the test reached its stop.
    fn deinit(r: *Running) void {
        r.signal();
        if (!r.stopped) {
            r.lis.close();
            r.srv.endIdle();
        }
        r.hold.end(null, .cancelled, ""); // the test's own teardown: whatever still holds
        const began = nowNs();
        while (r.srv.open_connections.load(.acquire) > 0) {
            if (msSince(began) > 5000) return; // left, not freed under a connection thread
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        r.srv.deinit();
    }
};

fn startServer(hold: *HoldEngine, text: *TestText) !Running {
    const srv = try server.Server.init(gpa, io, hold.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    errdefer srv.deinit();
    hold.srv = srv;
    return .{ .srv = srv, .lis = try listener_mod.Listener.open(.{ .ip4 = .loopback(0) }), .hold = hold };
}

const chat_body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100}";

fn sendChat(c: *Client) !void {
    var head: [160]u8 = undefined;
    try c.send(try std.fmt.bufPrint(&head, "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n", .{chat_body.len}));
    try c.send(chat_body);
}

fn waitSubmitted(hold: *HoldEngine, n: u32) !void {
    const began = nowNs();
    while (hold.counts().submitted < n) {
        if (msSince(began) > 5000) return error.NeverSubmitted;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
}

test "at the stop an idle keep-alive connection ends, and the engine closes before the connection wait" {
    var text: TestText = .{};
    var hold: HoldEngine = .{};
    var run = try startServer(&hold, &text);
    defer run.deinit();
    try run.start();
    // A: one /health, then idle on its kept-alive connection
    var a = try Client.connect(run.port);
    defer a.deinit();
    try a.send("GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
    try std.testing.expect(try a.readResponse(5000));
    try std.testing.expect(std.mem.startsWith(u8, a.got.items, "HTTP/1.1 200"));
    // B: a streamed chat the engine holds after its first token
    var b = try Client.connect(run.port);
    defer b.deinit();
    try sendChat(&b);
    try waitSubmitted(&hold, 1);
    try std.testing.expect(try b.readUntil("\"h\"", 5000));
    var a_end: EndWatch = .{};
    const watch = try std.Thread.spawn(.{}, EndWatch.run, .{ &a_end, &a });
    const signalled = nowNs();
    run.signal();
    var stop: serve.Stop = .{ .quiesce = .{ .call = HoldEngine.quiesce, .ctx = &hold }, .close_ms = 20_000 };
    const began = nowNs();
    const how = run.stopServing(&stop);
    const took_ms = msSince(began);
    watch.join();
    try std.testing.expectEqual(serve.Stopped.clean, how);
    try std.testing.expect(took_ms < 1500);
    const seen = hold.counts();
    try std.testing.expectEqual(@as(u32, 1), seen.quiesced);
    try std.testing.expect(seen.open_at_quiesce >= 1); // B still streamed: the close ran before the connection wait
    const a_at = a_end.at.load(.acquire);
    try std.testing.expect(a_at != 0); // A's connection ended
    try std.testing.expect(@divTrunc(a_at - signalled, std.time.ns_per_ms) < 1000);
    // B's reply ended with the stop's error and [DONE]
    try std.testing.expect(try b.readToEnd(5000));
    // cut by the stop: an error naming it (the client sends it again), never a reply that looks finished
    try std.testing.expect(std.mem.indexOf(u8, b.got.items, api.LaneHost.stop_message) != null);
    try std.testing.expect(std.mem.indexOf(u8, b.got.items, "\"finish_reason\": \"cancelled\"") == null);
    try std.testing.expect(std.mem.endsWith(u8, b.got.items, "data: [DONE]\n\n"));
}

test "an engine whose orderly close does not return within TENSORFOLD_CLOSE_MS: the stop ends forced at the bound" {
    var text: TestText = .{};
    var hold: HoldEngine = .{ .hang = true };
    var run = try startServer(&hold, &text);
    defer run.deinit();
    try run.start();
    run.signal();
    var stop: serve.Stop = .{ .quiesce = .{ .call = HoldEngine.quiesce, .ctx = &hold }, .close_ms = 200 };
    defer if (stop.closer.thread) |t| {
        hold.released.store(true, .release);
        t.join();
    };
    const began = nowNs();
    const how = run.stopServing(&stop);
    const took_ms = msSince(began);
    try std.testing.expectEqual(serve.Stopped.forced, how);
    try std.testing.expect(took_ms >= 200 and took_ms < 1500);
    try std.testing.expectEqual(@as(u32, 1), hold.counts().quiesced);
    try std.testing.expect(!hold.returned.load(.acquire));
    // the late close still runs on its thread (the process ends there: run calls exit(0)); here it is let go and joined
    hold.released.store(true, .release);
    stop.closer.thread.?.join();
    stop.closer.thread = null;
    try std.testing.expect(hold.returned.load(.acquire));
}

test "a stop while a stream is in its prompt pass: the close waits for the pass's next chunk, and the stream ends with its last chunk and [DONE]" {
    var text: TestText = .{};
    var hold: HoldEngine = .{ .first_token = false, .delay_ms = 300 };
    var run = try startServer(&hold, &text);
    defer run.deinit();
    try run.start();
    var b = try Client.connect(run.port);
    defer b.deinit();
    try sendChat(&b);
    try waitSubmitted(&hold, 1);
    try std.testing.expect(try b.readUntil("data: ", 5000)); // the role chunk: no token yet
    run.signal();
    var stop: serve.Stop = .{ .quiesce = .{ .call = HoldEngine.quiesce, .ctx = &hold }, .close_ms = 5000 };
    const began = nowNs();
    const how = run.stopServing(&stop);
    const took_ms = msSince(began);
    try std.testing.expectEqual(serve.Stopped.clean, how);
    try std.testing.expect(took_ms >= 300 and took_ms < 5000); // the close's own time, inside its bound
    try std.testing.expectEqual(@as(u32, 1), hold.counts().quiesced);
    try std.testing.expect(try b.readToEnd(5000));
    // cut by the stop: an error naming it (the client sends it again), never a reply that looks finished
    try std.testing.expect(std.mem.indexOf(u8, b.got.items, api.LaneHost.stop_message) != null);
    try std.testing.expect(std.mem.indexOf(u8, b.got.items, "\"finish_reason\": \"cancelled\"") == null);
    try std.testing.expect(std.mem.endsWith(u8, b.got.items, "data: [DONE]\n\n"));
}

test "with no thread for the orderly close it runs on the stopping thread: the flush and the followers' done still happen" {
    const Ran = struct {
        var called = std.atomic.Value(bool).init(false);
        fn call(_: *anyopaque) void {
            called.store(true, .release);
        }
    };
    serve.test_no_thread = true;
    defer serve.test_no_thread = false;
    var ctx: u8 = 0;
    var c: serve.Closer = .{};
    try std.testing.expect(c.run(io, .{ .call = Ran.call, .ctx = &ctx }, 1000));
    try std.testing.expect(Ran.called.load(.acquire));
    try std.testing.expect(c.thread == null);
}
