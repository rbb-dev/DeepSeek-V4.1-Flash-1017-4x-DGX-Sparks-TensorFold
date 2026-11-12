//! The admin surface: read-only, behind its own keys, and one 404 for every refusal.
const std = @import("std");
const posix = std.posix;
const api = @import("engine_api");
const settings = api.settings;
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const auth = @import("auth.zig");
const Conn = @import("http_conn.zig").Conn;
const admin = @import("admin.zig");
const log = @import("log.zig");
const admin_series = @import("admin_series.zig");
const admin_requests = @import("admin_requests.zig");

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

/// An engine that counts what reaches its queue: the admin surface must never submit or cancel anything. A test that
/// watches a server sees, inside `submit`, what the admin surface shows of the request being submitted.
const Engine = struct {
    submits: u32 = 0,
    cancels: u32 = 0,
    watch: ?*server.Server = null,
    seen_live: u32 = 0,
    seen: ?[]u8 = null,
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
        const e: *@This() = @ptrCast(@alignCast(ctx));
        e.submits += 1;
        if (e.watch) |srv| {
            e.seen_live = srv.requests.liveCount(std.testing.io);
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const v = admin.requestsValue(srv, arena.allocator()) catch return error.Closed;
            const text = json.stringify(arena.allocator(), v, .{}) catch return error.Closed;
            e.seen = std.testing.allocator.dupe(u8, text) catch return error.Closed;
        }
        return error.Closed;
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        e.cancels += 1;
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{ .healthy = true };
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct {
    status: u16,
    head: []u8,
    body: []u8,

    fn free(r: Reply) void {
        std.testing.allocator.free(r.head);
        std.testing.allocator.free(r.body);
    }

    fn hasHeader(r: Reply, line: []const u8) bool {
        var it = std.mem.splitSequence(u8, r.head, "\r\n");
        while (it.next()) |l| if (std.ascii.eqlIgnoreCase(l, line)) return true;
        return false;
    }
};

/// One request through the key gate and the routes, as a connection runs it. `headers` are whole lines ending in CRLF.
fn exchange(srv: *server.Server, method: []const u8, path: []const u8, headers: []const u8, body: []const u8) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(posix.AF.UNIX, posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    var head_buf: [1024]u8 = undefined;
    const head = try std.fmt.bufPrint(&head_buf, "{s} {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n{s}\r\n", .{ method, path, body.len, headers });
    if (std.c.write(pair[0], head.ptr, head.len) != @as(isize, @intCast(head.len))) return error.Closed;
    if (body.len > 0 and std.c.write(pair[0], body.ptr, body.len) != @as(isize, @intCast(body.len))) return error.Closed;
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
        var pollfd = [_]posix.pollfd{.{ .fd = pair[0], .events = posix.POLL.IN, .revents = 0 }};
        if (try posix.poll(&pollfd, 200) == 0) break;
        const n = try posix.read(pair[0], &tmp);
        if (n == 0) break;
        try raw.appendSlice(std.testing.allocator, tmp[0..n]);
    }
    const split = std.mem.indexOf(u8, raw.items, "\r\n\r\n") orelse return error.BadReply;
    const code = std.fmt.parseInt(u16, raw.items["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    return .{ .status = code, .head = try std.testing.allocator.dupe(u8, raw.items[0..split]), .body = try std.testing.allocator.dupe(u8, raw.items[split + 4 ..]) };
}

fn request(srv: *server.Server, method: []const u8, path: []const u8, headers: []const u8) !Reply {
    return exchange(srv, method, path, headers, "");
}

const Rig = struct {
    arena: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,
    engine: Engine = .{},
    text: NoText = .{},
    keys: auth.Store = undefined,
    admin: auth.Store = undefined,
    srv: *server.Server = undefined,

    /// `shared`: the /v1 key file's keys; `admin`: the admin key file's (null: no admin file at all).
    fn open(rig: *Rig, shared: ?[]const u8, admin_text: ?[]const u8) !void {
        rig.arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        rig.tmp = std.testing.tmpDir(.{});
        const a = rig.arena.allocator();
        var problem: []const u8 = "";
        if (shared) |k| rig.keys = try auth.Store.init(a, &.{k}, "", null, false, &problem);
        if (admin_text) |k| {
            try rig.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "admin-keys", .data = k });
            const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/admin-keys", .{rig.tmp.sub_path}, 0);
            try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o600, 0));
            rig.admin = try auth.Store.initAdmin(a, path, if (shared != null) &rig.keys else null, &problem);
        }
        rig.srv = try server.Server.init(std.testing.allocator, std.testing.io, rig.engine.engine(), rig.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, if (shared != null) &rig.keys else null);
        if (admin_text != null) rig.srv.admin_keys = &rig.admin;
    }

    fn close(rig: *Rig) void {
        rig.srv.deinit();
        rig.tmp.cleanup();
        rig.arena.deinit();
    }
};

/// `unit` repeated `n` times, at compile time.
fn repeat(comptime unit: []const u8, comptime n: usize) *const [unit.len * n]u8 {
    comptime {
        var out: [unit.len * n]u8 = undefined;
        for (0..n) |i| @memcpy(out[i * unit.len ..][0..unit.len], unit);
        const final = out;
        return &final;
    }
}

/// Admin keys are at least 32 bytes; the label is the scope.
const key_admin = "adm-key-1-0123456789abcdef01234567";
const key_read = "read-key-0123456789abcdef012345678";
const key_agent = "agent-key-0123456789abcdef01234567";
const key_other = "other-key-0123456789abcdef01234567";
const admin_keys_file = "admin: " ++ key_admin ++ "\nread-ops: " ++ key_read ++ "\nagent-night: " ++ key_agent ++ "\nops: " ++ key_other ++ "\n";
const bearer = "Authorization: Bearer " ++ key_admin ++ "\r\n";
const bearer_read = "Authorization: Bearer " ++ key_read ++ "\r\n";
const bearer_agent = "Authorization: Bearer " ++ key_agent ++ "\r\n";
const bearer_other = "Authorization: Bearer " ++ key_other ++ "\r\n";
const json_head = "Content-Type: application/json\r\n";


fn base(comptime name: []const u8) *const [admin.prefix.len + 1 + name.len]u8 {
    return admin.prefix ++ "/" ++ name;
}

test "with an admin key: schema, config, status and ledger answer 200 and the engine's queue is untouched" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open("shared-key-1", admin_keys_file);
    defer rig.close();
    const schema_reply = try request(rig.srv, "GET", "/admin/api/v1/schema", bearer);
    defer schema_reply.free();
    try std.testing.expectEqual(@as(u16, 200), schema_reply.status);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const doc = (try json.parse(arena.allocator(), schema_reply.body)).ok;
    var visible: usize = 0;
    for (&settings.options) |*o| visible += @intFromBool(!o.hidden);
    try std.testing.expectEqual(visible, doc.get("options").?.array.len);
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(arena.allocator(), "{d}", .{visible}), doc.get("count").?.int);
    try std.testing.expectEqual(settings.retired.len, doc.get("retired").?.array.len);
    var found = false;
    for (doc.get("options").?.array) |row| {
        if (std.mem.eql(u8, json.strOr(row.get("env")), "TF_DS_SPILL_READERS")) {
            found = true;
            try std.testing.expectEqualStrings("int", row.get("kind").?.string);
            try std.testing.expectEqualStrings("4", row.get("default").?.string);
            try std.testing.expectEqualStrings("R", row.get("apply").?.string);
        }
    }
    try std.testing.expect(found);

    const cfg = try request(rig.srv, "GET", "/admin/api/v1/config", bearer);
    defer cfg.free();
    try std.testing.expectEqual(@as(u16, 200), cfg.status);
    try std.testing.expect(std.mem.indexOf(u8, cfg.body, "\"engine_cfg\": null") != null);

    const st = try request(rig.srv, "GET", "/admin/api/v1/status", bearer);
    defer st.free();
    try std.testing.expectEqual(@as(u16, 200), st.status);
    try std.testing.expect(std.mem.indexOf(u8, st.body, "\"read_only\": false") != null);
    try std.testing.expect(std.mem.indexOf(u8, st.body, "\"status\": \"ok\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, st.body, "\"scope\": \"admin\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, st.body, "\"writable_classes\": [\"H\", \"A\", \"R\"]") != null);

    const led = try request(rig.srv, "GET", "/admin/api/v1/ledger", "x-api-key: " ++ key_admin ++ "\r\n");
    defer led.free();
    try std.testing.expectEqual(@as(u16, 200), led.status);
    try std.testing.expect(std.mem.indexOf(u8, led.body, "\"entries\": []") != null);

    try std.testing.expectEqual(@as(u32, 0), rig.engine.submits);
    try std.testing.expectEqual(@as(u32, 0), rig.engine.cancels);
}

test "/config never shows a key's value" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    const o = settings.byEnv("TENSORFOLD_API_KEY").?;
    const saved = settings.resolved[settings.indexOf(o)];
    defer settings.resolved[settings.indexOf(o)] = saved;
    settings.resolved[settings.indexOf(o)] = .{ .source = .env, .value = "sk-do-not-show-me" };
    const cfg = try request(rig.srv, "GET", "/admin/api/v1/config", bearer);
    defer cfg.free();
    try std.testing.expectEqual(@as(u16, 200), cfg.status);
    try std.testing.expect(std.mem.indexOf(u8, cfg.body, "sk-do-not-show-me") == null);
    try std.testing.expect(std.mem.indexOf(u8, cfg.body, "(set)") != null);
}

test "every refusal is the same unknown-path 404: no key, a wrong key, the /v1 key, a POST, an unknown admin path, no admin file" {
    var with: Rig = .{ .arena = undefined, .tmp = undefined };
    try with.open("shared-key-1", admin_keys_file);
    defer with.close();
    var without: Rig = .{ .arena = undefined, .tmp = undefined };
    try without.open("shared-key-1", null);
    defer without.close();
    const path = "/admin/api/v1/config";
    const reference = try request(without.srv, "GET", path, bearer); // no admin file at all
    defer reference.free();
    try std.testing.expectEqual(@as(u16, 404), reference.status);
    try std.testing.expect(std.mem.indexOf(u8, reference.body, "unknown path /admin/api/v1/config") != null);
    const cases = [_]struct { method: []const u8, headers: []const u8 }{
        .{ .method = "GET", .headers = "" },
        .{ .method = "GET", .headers = "Authorization: Bearer nope\r\n" },
        .{ .method = "GET", .headers = "Authorization: Bearer shared-key-1\r\n" }, // a /v1 key opens nothing here
        .{ .method = "POST", .headers = bearer },
        .{ .method = "DELETE", .headers = bearer },
        .{ .method = "GET", .headers = bearer ++ bearer }, // two Authorization headers
    };
    for (cases) |c| {
        const r = try request(with.srv, c.method, path, c.headers);
        defer r.free();
        try std.testing.expectEqual(@as(u16, 404), r.status);
        try std.testing.expectEqualStrings(reference.body, r.body);
    }
    // a path the surface lacks, even with the right key, is the same 404; a path ending in /models is not the key gate's 401
    // (/admin and /admin/ are not among them: they redirect to the page, the test of the trailing-slash redirect)
    for ([_][]const u8{ "/admin/api/v1/nothing", "/admin/x/models", "/admin/api/v1/set" }) |p| {
        const r = try request(with.srv, "GET", p, bearer);
        defer r.free();
        try std.testing.expectEqual(@as(u16, 404), r.status);
    }
    const bare = try request(with.srv, "GET", "/admin/x/models", "");
    defer bare.free();
    try std.testing.expectEqual(@as(u16, 404), bare.status);
}

test "every route answers the same 404 to a /v1 key, no key and the wrong method; an unbuilt POST route is the 404 to an admin key too" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open("shared-key-1", admin_keys_file);
    defer rig.close();
    var gone = try request(rig.srv, "GET", "/admin/api/v1/nothing", "");
    defer gone.free();
    inline for (get_routes) |name| {
        const unknown = try request(rig.srv, "GET", "/admin/api/v1/" ++ name, "Authorization: Bearer shared-key-1\r\n");
        defer unknown.free();
        try std.testing.expectEqual(@as(u16, 404), unknown.status);
        const none = try request(rig.srv, "GET", "/admin/api/v1/" ++ name, "");
        defer none.free();
        try std.testing.expectEqual(@as(u16, 404), none.status);
        try std.testing.expect(std.mem.indexOf(u8, none.body, "unknown path /admin/api/v1/" ++ name) != null);
        const post = try exchange(rig.srv, "POST", "/admin/api/v1/" ++ name, bearer ++ json_head, "{}"); // a GET route called with POST
        defer post.free();
        try std.testing.expectEqual(@as(u16, 404), post.status);
    }
        var buf: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "/admin/api/v1/{s}", .{name});
        const r = try exchange(rig.srv, "POST", p, bearer ++ json_head, "{}");
        defer r.free();
        try std.testing.expectEqual(@as(u16, 404), r.status);
        const g = try request(rig.srv, "GET", p, bearer);
        defer g.free();
        try std.testing.expectEqual(@as(u16, 404), g.status);
    }
    // keys/reload is a POST: a GET of it is the 404
    const g = try request(rig.srv, "GET", base("keys/reload"), bearer);
    defer g.free();
    try std.testing.expectEqual(@as(u16, 404), g.status);
    try std.testing.expectEqual(@as(u32, 0), rig.engine.submits);
}

test "an admin key opens nothing of /v1, and the /v1 gate is as it was" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open("shared-key-1", admin_keys_file);
    defer rig.close();
    const denied = try request(rig.srv, "GET", "/v1/models", bearer);
    defer denied.free();
    try std.testing.expectEqual(@as(u16, 401), denied.status);
    const ok = try request(rig.srv, "GET", "/v1/models", "Authorization: Bearer shared-key-1\r\n");
    defer ok.free();
    try std.testing.expectEqual(@as(u16, 200), ok.status);
}

test "the admin key file is reread like the API key file: a changed file changes who is let in" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    const ok = try request(rig.srv, "GET", "/admin/api/v1/ledger", bearer);
    defer ok.free();
    try std.testing.expectEqual(@as(u16, 200), ok.status);
    try rig.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "admin-keys", .data = "admin: adm-key-2-0123456789abcdef01234567\n" });
    const path = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/admin-keys", .{rig.tmp.sub_path}, 0);
    defer std.testing.allocator.free(path);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o600, 0));
    try std.Io.sleep(std.testing.io, .fromMilliseconds(1100), .awake); // the store looks at the file at most once a second
    const old = try request(rig.srv, "GET", "/admin/api/v1/ledger", bearer);
    defer old.free();
    try std.testing.expectEqual(@as(u16, 404), old.status);
    const new = try request(rig.srv, "GET", "/admin/api/v1/ledger", "Authorization: Bearer adm-key-2-0123456789abcdef01234567\r\n");
    defer new.free();
    try std.testing.expectEqual(@as(u16, 200), new.status);
}

test "/routes lists only what the key's scope may call, with each route's method" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const full = try request(rig.srv, "GET", base("routes"), bearer);
    defer full.free();
    try std.testing.expectEqual(@as(u16, 200), full.status);
    const all = (try json.parse(arena.allocator(), full.body)).ok.get("routes").?.array;
    try std.testing.expectEqual(admin.table.len, all.len);
    var reload_is_post = false;
    for (all) |r| if (std.mem.eql(u8, r.get("path").?.string, "/admin/api/v1/keys/reload")) {
        reload_is_post = std.mem.eql(u8, r.get("method").?.string, "POST");
    };
    try std.testing.expect(reload_is_post);
    for ([_][]const u8{ bearer_read, bearer_agent, bearer_other }) |h| {
        const r = try request(rig.srv, "GET", base("routes"), h);
        defer r.free();
        const rows = (try json.parse(arena.allocator(), r.body)).ok.get("routes").?.array;
        try std.testing.expect(rows.len > 0 and rows.len < all.len);
        for (rows) |row| {
            const path = row.get("path").?.string;
            try std.testing.expect(!std.mem.endsWith(u8, path, "/keys") and !std.mem.endsWith(u8, path, "/logs") and !std.mem.endsWith(u8, path, "/keys/reload"));
            if (std.mem.endsWith(u8, path, "/validate") or std.mem.endsWith(u8, path, "/set")) {
        }
    }
}

test "no admin route name ends the way the /v1 suffix router matches" {
    for (&admin.table) |*t| {
        for ([_][]const u8{ "/models", "/decisions", "/completions", "/chat/completions" }) |end| {
            var buf: [96]u8 = undefined;
            const full = try std.fmt.bufPrint(&buf, "/{s}", .{t.name});
            try std.testing.expect(!std.mem.endsWith(u8, full, end));
        }
        // and the gate never takes the route for one of /v1's
        try std.testing.expect(!auth.gated(admin.prefix ++ "/" ++ "x", false));
    }
}

test "scope: a read key, an agent key and a key with no prefix read, and are refused /keys, /logs and keys/reload with a 403" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    inline for ([_][]const u8{ bearer_read, bearer_agent, bearer_other }) |h| {
        for ([_][]const u8{ "schema", "config", "status", "ledger", "requests", "routes", "history", "leases", "policy" }) |name| {
            var buf: [64]u8 = undefined;
            const p = try std.fmt.bufPrint(&buf, "/admin/api/v1/{s}", .{name});
            const r = try request(rig.srv, "GET", p, h);
            defer r.free();
            try std.testing.expectEqual(@as(u16, 200), r.status);
        }
        for ([_][]const u8{ "keys", "logs" }) |name| {
            var buf: [64]u8 = undefined;
            const p = try std.fmt.bufPrint(&buf, "/admin/api/v1/{s}", .{name});
            const r = try request(rig.srv, "GET", p, h);
            defer r.free();
            try std.testing.expectEqual(@as(u16, 403), r.status);
            try std.testing.expect(std.mem.indexOf(u8, r.body, "scope_denied") != null);
        }
        const reload = try exchange(rig.srv, "POST", base("keys/reload"), h ++ json_head, "{}");
        defer reload.free();
        try std.testing.expectEqual(@as(u16, 403), reload.status);
    }
    const st = try request(rig.srv, "GET", base("status"), bearer_agent);
    defer st.free();
    try std.testing.expect(std.mem.indexOf(u8, st.body, "\"scope\": \"agent\"") != null);
    const ok = try request(rig.srv, "GET", base("keys"), bearer);
    defer ok.free();
    try std.testing.expectEqual(@as(u16, 200), ok.status);
    try std.testing.expect(std.mem.indexOf(u8, ok.body, "\"label\": \"agent-night\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, ok.body, "\"scope\": \"agent\"") != null);
}

test "a label that names no scope is read-only and is logged when its file loads" {
    var captured: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer captured.deinit();
    log.capture = &captured.writer;
    defer log.capture = null;
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    try std.testing.expect(std.mem.indexOf(u8, captured.written(), "admin key labelled ops names no scope") != null);
    try std.testing.expect(std.mem.indexOf(u8, captured.written(), "labelled admin ") == null);
    try std.testing.expect(std.mem.indexOf(u8, captured.written(), key_other) == null);
    try std.testing.expectEqual(auth.Scope.read, auth.scopeOf("ops").scope);
    try std.testing.expectEqual(auth.Scope.admin, auth.scopeOf("admin-night").scope);
    try std.testing.expectEqual(auth.Scope.read, auth.scopeOf("administrator").scope); // not a prefix of `admin-`
}

test "the admin store refuses a key shared with /v1 and a key under 32 bytes, closed, naming labels and never keys" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var problem: []const u8 = "";
    var shared = try auth.Store.init(a, &.{key_admin}, "", null, false, &problem);
    // a /v1 key that is also an admin key
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "same", .data = "admin: " ++ key_admin ++ "\n" });
    const same = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/same", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, same, 0o600, 0));
    var closed = try auth.Store.initAdmin(a, same, &shared, &problem);
    try std.testing.expect(!closed.valid);
    try std.testing.expect(std.mem.indexOf(u8, closed.refusal, "admin") != null and std.mem.indexOf(u8, closed.refusal, "cli-1") != null);
    try std.testing.expect(std.mem.indexOf(u8, closed.refusal, key_admin) == null);
    try std.testing.expect(closed.match(a, "Bearer " ++ key_admin, null) == null); // fail closed: even the key the file holds
    // 31 bytes refused, 32 loads
    const thirty_one = comptime repeat("k", 31);
    const thirty_two = comptime repeat("k", 32);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "short", .data = "admin: " ++ thirty_one ++ "\n" });
    const short = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/short", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, short, 0o600, 0));
    var too_short = try auth.Store.initAdmin(a, short, null, &problem);
    try std.testing.expect(!too_short.valid);
    try std.testing.expect(std.mem.indexOf(u8, too_short.refusal, "labelled admin") != null);
    try std.testing.expect(std.mem.indexOf(u8, too_short.refusal, thirty_one) == null);
    try std.testing.expect(too_short.match(a, "Bearer " ++ thirty_one, null) == null);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "long", .data = "admin: " ++ thirty_two ++ "\n" });
    const long = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/long", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, long, 0o600, 0));
    var fits = try auth.Store.initAdmin(a, long, &shared, &problem);
    try std.testing.expect(fits.valid);
    try std.testing.expectEqualStrings("admin", fits.match(a, "Bearer " ++ thirty_two, null).?);
    // a repaired file reopens a closed store at its next reread
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "short", .data = "admin: " ++ thirty_two ++ "\n" });
    try std.testing.expect(too_short.reloadNow().valid);
    try std.testing.expectEqualStrings("admin", too_short.match(a, "Bearer " ++ thirty_two, null).?);
}

test "a closed admin store answers 404 to every API path, and the page still answers" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open("shared-key-1", "admin: short-key\n");
    defer rig.close();
    try std.testing.expect(!rig.admin.valid);
    const r = try request(rig.srv, "GET", base("schema"), "Authorization: Bearer short-key\r\n");
    defer r.free();
    try std.testing.expectEqual(@as(u16, 404), r.status);
    const page = try request(rig.srv, "GET", "/admin/ui/", "");
    defer page.free();
    try std.testing.expectEqual(@as(u16, 200), page.status);
}

test "POST /keys/reload rereads both files at once, as SIGHUP does, and says only counts" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open("shared-key-1", admin_keys_file);
    defer rig.close();
    const before = try request(rig.srv, "GET", base("status"), bearer);
    defer before.free();
    try std.testing.expectEqual(@as(u16, 200), before.status);
    try rig.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "admin-keys", .data = "admin: adm-key-3-0123456789abcdef01234567\n" });
    const path = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/admin-keys", .{rig.tmp.sub_path}, 0);
    defer std.testing.allocator.free(path);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o600, 0));
    // the store would look again only a second after its last look: the reload does not wait
    const reload = try exchange(rig.srv, "POST", base("keys/reload"), "Authorization: Bearer " ++ key_admin ++ "\r\n" ++ json_head, "");
    defer reload.free();
    try std.testing.expectEqual(@as(u16, 200), reload.status);
    try std.testing.expect(std.mem.indexOf(u8, reload.body, "\"keys\": 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, reload.body, "adm-key") == null);
    const old = try request(rig.srv, "GET", base("status"), bearer);
    defer old.free();
    try std.testing.expectEqual(@as(u16, 404), old.status);
    const new = try request(rig.srv, "GET", base("status"), "Authorization: Bearer adm-key-3-0123456789abcdef01234567\r\n");
    defer new.free();
    try std.testing.expectEqual(@as(u16, 200), new.status);
}

test "a write needs JSON, this server's own Origin, a body of at most 65,536 bytes, and no field" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    const path = base("keys/reload");
    const plain = try exchange(rig.srv, "POST", path, bearer ++ "Content-Type: text/plain\r\n", "{}");
    defer plain.free();
    try std.testing.expectEqual(@as(u16, 415), plain.status);
    const nothing = try exchange(rig.srv, "POST", path, bearer, "{}");
    defer nothing.free();
    try std.testing.expectEqual(@as(u16, 415), nothing.status);
    const foreign = try exchange(rig.srv, "POST", path, bearer ++ json_head ++ "Origin: http://evil.example\r\n", "{}");
    defer foreign.free();
    try std.testing.expectEqual(@as(u16, 403), foreign.status);
    try std.testing.expect(std.mem.indexOf(u8, foreign.body, "origin_denied") != null);
    const own = try exchange(rig.srv, "POST", path, bearer ++ json_head ++ "Origin: http://t\r\n", "{}"); // Host: t
    defer own.free();
    try std.testing.expectEqual(@as(u16, 200), own.status);
    const big = try std.testing.allocator.alloc(u8, 65537);
    defer std.testing.allocator.free(big);
    @memset(big, ' ');
    const huge = try exchange(rig.srv, "POST", path, bearer ++ json_head, big);
    defer huge.free();
    try std.testing.expectEqual(@as(u16, 413), huge.status);
    try std.testing.expect(std.mem.indexOf(u8, huge.body, "body_too_large") != null);
    const field = try exchange(rig.srv, "POST", path, bearer ++ json_head, "{\"reason\": \"x\"}");
    defer field.free();
    try std.testing.expectEqual(@as(u16, 400), field.status);
    try std.testing.expect(std.mem.indexOf(u8, field.body, "unknown_field") != null);
    const bad = try exchange(rig.srv, "POST", path, bearer ++ json_head, "{not json");
    defer bad.free();
    try std.testing.expectEqual(@as(u16, 400), bad.status);
    try std.testing.expectEqual(@as(u32, 0), rig.engine.submits);
}

test "every admin answer is never cached, never sniffed, and carries the API's version" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    inline for (get_routes) |name| {
        const r = try request(rig.srv, "GET", "/admin/api/v1/" ++ name, bearer);
        defer r.free();
        try std.testing.expectEqual(@as(u16, 200), r.status);
        try std.testing.expect(r.hasHeader("Cache-Control: no-store"));
        try std.testing.expect(r.hasHeader("X-Content-Type-Options: nosniff"));
        try std.testing.expect(r.hasHeader("X-TF-Admin-API: 1.0"));
        try std.testing.expect(r.hasHeader("Content-Type: application/json"));
    }
    const denied = try request(rig.srv, "GET", base("keys"), bearer_read); // a 403 too
    defer denied.free();
    try std.testing.expect(denied.hasHeader("Cache-Control: no-store"));
}

test "an admin address without the page's trailing slash is redirected to /admin/ui/ by a relative Location, so the page's relative app.js and app.css resolve under /admin/ui/" {
    // /admin/ui once loaded the page, whose relative "app.js" then resolved to /admin/app.js (404): a dead page
    var with: Rig = .{ .arena = undefined, .tmp = undefined };
    try with.open("shared-key-1", admin_keys_file);
    defer with.close();
    const cases = [_]struct { path: []const u8, location: []const u8 }{
        .{ .path = "/admin", .location = "Location: admin/ui/" },
        .{ .path = "/admin/", .location = "Location: ui/" },
        .{ .path = "/admin/ui", .location = "Location: ui/" },
    };
    for (cases) |c| {
        const r = try request(with.srv, "GET", c.path, "");
        defer r.free();
        try std.testing.expectEqual(@as(u16, 301), r.status);
        try std.testing.expect(r.hasHeader(c.location));
    }
    // the page itself and its two files stay where the relative links point
    for ([_][]const u8{ "/admin/ui/", "/admin/ui/app.js", "/admin/ui/app.css" }) |path| {
        const r = try request(with.srv, "GET", path, "");
        defer r.free();
        try std.testing.expectEqual(@as(u16, 200), r.status);
    }
    // no admin key file: no redirect either, the 404 of any unknown path
    var without: Rig = .{ .arena = undefined, .tmp = undefined };
    try without.open("shared-key-1", null);
    defer without.close();
    for ([_][]const u8{ "/admin", "/admin/", "/admin/ui" }) |path| {
        const off = try request(without.srv, "GET", path, "");
        defer off.free();
        try std.testing.expectEqual(@as(u16, 404), off.status);
    }
}

test "the page's three paths answer without a key only while an admin key file is configured, to GET, and match exactly" {
    var with: Rig = .{ .arena = undefined, .tmp = undefined };
    try with.open("shared-key-1", admin_keys_file);
    defer with.close();
    var without: Rig = .{ .arena = undefined, .tmp = undefined };
    try without.open("shared-key-1", null);
    defer without.close();
    const files = [_]struct { path: []const u8, kind: []const u8, body: []const u8 }{
        .{ .path = "/admin/ui/", .kind = "Content-Type: text/html;charset=utf-8", .body = admin.page_html },
        .{ .path = "/admin/ui/app.js", .kind = "Content-Type: text/javascript;charset=utf-8", .body = admin.page_js },
        .{ .path = "/admin/ui/app.css", .kind = "Content-Type: text/css;charset=utf-8", .body = admin.page_css },
    };
    for (files) |f| {
        const r = try request(with.srv, "GET", f.path, "");
        defer r.free();
        try std.testing.expectEqual(@as(u16, 200), r.status);
        try std.testing.expectEqualStrings(f.body, r.body);
        try std.testing.expect(r.hasHeader(f.kind));
        try std.testing.expect(r.hasHeader("Content-Security-Policy: " ++ admin.page_csp));
        try std.testing.expect(r.hasHeader("X-Content-Type-Options: nosniff"));
        try std.testing.expect(r.hasHeader("Referrer-Policy: no-referrer"));
        try std.testing.expect(r.hasHeader("Cache-Control: no-cache"));
        try std.testing.expect(std.mem.indexOf(u8, r.head, "\r\nETag: \"") != null);
        // a key does not change the answer
        const keyed = try request(with.srv, "GET", f.path, bearer);
        defer keyed.free();
        try std.testing.expectEqualStrings(f.body, keyed.body);
        // no admin key file: the 404 of any unknown path
        const off = try request(without.srv, "GET", f.path, "");
        defer off.free();
        try std.testing.expectEqual(@as(u16, 404), off.status);
        // GET only
        const post = try exchange(with.srv, "POST", f.path, "", "{}");
        defer post.free();
        try std.testing.expectEqual(@as(u16, 404), post.status);
        const head = try request(with.srv, "HEAD", f.path, "");
        defer head.free();
        try std.testing.expectEqual(@as(u16, 404), head.status);
    }
    for ([_][]const u8{ "/admin/ui/app.js/", "/admin/ui/../x", "/admin/ui/app.js?x", "/admin/ui/app.js?", "/admin/ui//", "/admin/ui/x", "/admin/ui/index.html", "/admin/ui/app.JS", "/admin/ui/app.css/", "/admin/uix", "/admin/ui?x=1", "/admin/ui/%61pp.js" }) |p| {
        const r = try request(with.srv, "GET", p, "");
        defer r.free();
        if (r.status != 404) std.debug.print("path {s} answered {d}\n", .{ p, r.status });
        try std.testing.expectEqual(@as(u16, 404), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "unknown path") != null);
    }
}

test "the page holds no inline script or style and builds nothing from text as markup" {
    // the policy forbids inline script and style; the files must not need either
    try std.testing.expect(std.mem.indexOf(u8, admin.page_html, "<script>") == null);
    try std.testing.expect(std.mem.indexOf(u8, admin.page_html, "<style") == null);
    try std.testing.expect(std.mem.indexOf(u8, admin.page_html, " style=") == null);
    try std.testing.expect(std.mem.indexOf(u8, admin.page_html, " onclick") == null);
    for ([_][]const u8{ admin.page_html, admin.page_js, admin.page_css }) |text| {
        for ([_][]const u8{ "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "setAttribute('style'", "setAttribute(\"style\"", "https://", "local" ++ "Storage", "indexed" ++ "DB", "document.cookie", "pictures" }) |word| {
            if (std.mem.indexOf(u8, text, word)) |at| {
                std.debug.print("forbidden `{s}` at byte {d}\n", .{ word, at });
                return error.TestForbiddenWord;
            }
        }
    }
    // nothing is loaded from another origin: the one URL in the files is the SVG namespace
    for ([_][]const u8{ admin.page_html, admin.page_js, admin.page_css }) |text| {
        var from: usize = 0;
        while (std.mem.indexOfPos(u8, text, from, "http://")) |at| {
            try std.testing.expect(std.mem.startsWith(u8, text[at..], "http://www.w3.org/2000/svg"));
            from = at + 1;
        }
    }
    // the key is held in a variable and sent as a header; the one place it is kept is the tab's session storage (see the next test)
    try std.testing.expect(admin.page_html.len + admin.page_js.len + admin.page_css.len <= 128 * 1024); // 128 KiB (the health boxes, the graphs' hover and bucket drawing, the seats, charts, memory bars, a settings list with a details row each) and still loads nothing from outside
}

test "the admin key survives a refresh in the tab's session storage only, and a sign-out, a refusal, the idle time or a second failed read forgets it" {
    const js = admin.page_js;
    // one store, one slot, every access guarded (a browser that refuses storage must still show the sign-in)
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, js, "window.session" ++ "Storage."));
    try std.testing.expect(std.mem.indexOf(u8, js, "KEY_SLOT = 'tensorfold-admin-key'") != null);
    const kept = std.mem.indexOf(u8, js, "function keptKey()").?;
    try std.testing.expect(std.mem.indexOf(u8, js[kept..], "try {") != null and std.mem.indexOf(u8, js[kept..], "catch (e)") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "document." ++ "cookie") == null);
    try std.testing.expect(std.mem.indexOf(u8, js, "The key stays in this page only") == null);
    // a 401 anywhere after sign-in is a refusal like the 404 is
    try std.testing.expect(std.mem.indexOf(u8, js, "res.status === 401 && S.status") != null);
    // case 1, a refresh within the idle time signs straight back in: the stored value is {k, exp}, written after the engine accepted
    // the key, moved on by every touch, and read at start
    try std.testing.expect(std.mem.indexOf(u8, js, "JSON.stringify({ k: v, exp: Date.now() + IDLE_SIGN_OUT_MS })") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "keepKey(typed);") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "keepKey(key); }") != null); // (touch)
    try std.testing.expect(std.mem.indexOf(u8, js, "if (kept) signIn(kept, 1);") != null);
    // case 2, after the idle time the key is gone: an expired or malformed value is deleted and never returned
    try std.testing.expect(std.mem.indexOf(u8, js, "v.exp > Date.now()") != null);
    const expired = std.mem.indexOf(u8, js[kept..], "v.exp > Date.now()").?;
    const removed = std.mem.indexOf(u8, js[kept + expired ..], "removeItem(KEY_SLOT)").?;
    try std.testing.expect(removed < 120); // (the line after the check, before the function returns)
    // the sign-in screen never shows with a key stored, and it stops the idle timer
    const login = std.mem.indexOf(u8, js, "function loginScreen(message)").?;
    const login_end = std.mem.indexOf(u8, js[login..], "const input").?;
    try std.testing.expect(std.mem.indexOf(u8, js[login .. login + login_end], "keepKey('');") != null);
    try std.testing.expect(std.mem.indexOf(u8, js[login .. login + login_end], "clearTimeout(S.idleTimer);") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "loginScreen(message, keep)") == null);
    // case 3, a failed first read retries once, then wipes: only signIn retries, with a count that falls to 0, and the start passes 1
    const sign = std.mem.indexOf(u8, js, "async function signIn(value, retries)").?;
    try std.testing.expect(std.mem.indexOf(u8, js[sign..], "if (!retries) return loginScreen(why);") != null);
    try std.testing.expect(std.mem.indexOf(u8, js[sign..], "signIn(typed, retries - 1)") != null);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, js, "setTimeout(() => signIn("));
    try std.testing.expect(std.mem.indexOf(u8, js, "signIn(input.value);") != null); // (a typed key gets no retry)
    // the start wipes the stored value before showing the screen, so a failure after the one retry leaves nothing behind
    const start = std.mem.indexOf(u8, js, "const kept = keptKey();").?;
    try std.testing.expect(std.mem.indexOf(u8, js[start..], "loginScreen('');") != null);
}

test "the Overview's Node health is one card a Spark from /status node_health, grouped, temperatures on one colour scale, every row with a tip, and no way to change the GPU clock" {
    const js = admin.page_js;
    try std.testing.expect(std.mem.indexOf(u8, js, "function nodeCard(rank)") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "st.node_health") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "'NODE HEALTH'") != null);
    // the groups of a card: the old separate tables are gone
    for ([_][]const u8{ "'MEMORY'", "'TEMPERATURE'", "'GPU'", "'CPU'", "'NVME'", "'RAILS'", "'Rail A'", "'Rail B'" }) |g| try std.testing.expect(std.mem.indexOf(u8, js, g) != null);
    for ([_][]const u8{ "Memory on each Spark", "Health of each Spark", "function healthCard()" }) |gone| try std.testing.expect(std.mem.indexOf(u8, js, gone) == null);
    // every row is built by the one function that gives it a tip
    try std.testing.expect(std.mem.indexOf(u8, js, "title: tip") != null);
    // one fixed temperature scale in one place: green at 45, yellow 65, orange 80, bright red at 90, the zones' trip at 104
    try std.testing.expect(std.mem.indexOf(u8, js, "const TEMP_STOPS = [[45, [78, 201, 160]], [65, [230, 194, 122]], [80, [232, 135, 58]], [90, [255, 59, 48]]];") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "const TEMP_TRIP_C = 104;") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "function tempColor(c)") != null);
    // a card shows its age only when its Spark has been silent for over 6 s; there is no "Updated just now" label
    try std.testing.expect(std.mem.indexOf(u8, js, "r.age_ms > 6000") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "Updated just now") == null);
    // read-only: nothing on the page writes a clock, a power limit or a lock (the API has no such route, and the page names none)
    for ([_][]const u8{ "api('clock", "api(\"clock", "lock" ++ "ed-clocks", "-l" ++ "gc", "power" ++ "-limit", "nvml" ++ "DeviceSet" }) |word| try std.testing.expect(std.mem.indexOf(u8, js, word) == null);
    try std.testing.expect(std.mem.indexOf(u8, js, "this page cannot change the clock") != null);
}

test "the page's charts draw thin smooth lines through the engine's fixed buckets, with the peak on hover" {
    const js = admin.page_js;
    try std.testing.expect(std.mem.indexOf(u8, js, "function smoothPath(pts)") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "'stroke-width': '1.25'") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "body.peaks") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "body.last_ms") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "peak ${fmt(pk)}") != null);
    // the old straight-segment path and the 2 px stroke are gone
    try std.testing.expect(std.mem.indexOf(u8, js, "`${run.length ? 'L' : 'M'}${x},${y} `") == null);
    try std.testing.expect(std.mem.indexOf(u8, js, "stroke-width': '2', 'stroke-linejoin'") == null);
}

test "the lines, the KV pool and the SSD tier are drawn from the engine's own requests table, status and series, never from a gateway" {
    const js = admin.page_js;
    try std.testing.expect(std.mem.indexOf(u8, js, "function poolChart()") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "['decode_tps', 'prefill_tps', 'seat_kv']") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "x.kv_tokens") != null);
    // every seat is a line, always shown; its phase is one of three words
    try std.testing.expect(std.mem.indexOf(u8, js, "function linesBox()") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "'DECODE' : filling ? 'PREFILL' : 'QUEUED'") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "const n = Math.min(32, Math.max(none(max) ? 0 : max, used));") != null);
    // the board's boxes, each fed by status.health, status.kv_pool and status.health.prompt_cache.spill
    for ([_][]const u8{ "'LINES'", "'KV POOL'", "'SSD TIER'", "'RATE'", "'DECODE'", "'PREFILL'", "st.kv_pool", "pc.spill[0]", "pc.restore_wait_ns" }) |w| try std.testing.expect(std.mem.indexOf(u8, js, w) != null);
    // the board's gateway-only boxes do not exist here
    for ([_][]const u8{ "BY CLIENT", "OVERFLOW", "HEADROOM", "GATEWAY" }) |w| try std.testing.expect(std.mem.indexOf(u8, js, w) == null);
    var lower: [admin.page_js.len]u8 = undefined;
    for (js, &lower) |ch, *o| o.* = std.ascii.toLower(ch);
    try std.testing.expect(std.mem.indexOf(u8, &lower, "lite" ++ "llm") == null);
}

test "the poll feeds boxes in place: it never clears a view, and each box writes its own text nodes" {
    const js = admin.page_js;
    // the poll path: status, then the page's own pull, then onStatus, which calls the header and the open page's update; never selectTab or a view's builder
    const poll = std.mem.indexOf(u8, js, "async function poll()").?;
    const poll_end = std.mem.indexOf(u8, js[poll..], "function stopAll()").?;
    const body = js[poll .. poll + poll_end];
    for ([_][]const u8{ "selectTab(", "overview(", "shell(", "clear(", "replaceChildren", "settingsPage(" }) |w| try std.testing.expect(std.mem.indexOf(u8, body, w) == null);
    const on = std.mem.indexOf(u8, js, "function onStatus()").?;
    const on_end = std.mem.indexOf(u8, js[on..], "// The header strip").?;
    for ([_][]const u8{ "selectTab(", "overview(", "shell(", "clear(", "replaceChildren" }) |w| try std.testing.expect(std.mem.indexOf(u8, js[on .. on + on_end], w) == null);
    // the writers touch the DOM only when a value moved
    try std.testing.expect(std.mem.indexOf(u8, js, "function setText(el, text) { const t = String(text); if (el.textContent !== t) el.textContent = t; }") != null);
    // the one place a list is rebuilt is when its set of rows changes (a signature), never every update
    try std.testing.expect(std.mem.indexOf(u8, js, "if (next !== sig) {") != null);
    // the old timers that rebuilt boxes are gone
    for ([_][]const u8{ "setInterval(loadSlow", "setInterval(loadCharts", "paintSpeed" }) |w| try std.testing.expect(std.mem.indexOf(u8, js, w) == null);
}

test "the header strip holds Turbo and the prompt-vs-generation balance as compact tiles, and the Overview has no prose" {
    const js = admin.page_js;
    for ([_][]const u8{ "Turn Turbo on", "Turn Turbo off", "Prompt vs generation balance", "What's Goals", "Updated just now", "Speed features'", "Memory on each Spark", "nobody asked for" }) |w| try std.testing.expect(std.mem.indexOf(u8, js, w) == null);
    try std.testing.expect(std.mem.indexOf(u8, js, "function balanceTile(t)") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "function goalsRow()") != null);
    // the goals chart exists only while a goal is set
    try std.testing.expect(std.mem.indexOf(u8, js, "if (!goal || b.state === 'off'") != null);
    // speed features by their schema label, the env name on hover
    try std.testing.expect(std.mem.indexOf(u8, js, "S.byEnv.get(f.env)") != null);
}

test "the Turbo tile shows the state in force, confirms before it stages the other one through the ordinary tray, and takes every figure from status.turbo" {
    const js = admin.page_js;
    try std.testing.expect(std.mem.indexOf(u8, js, "function turboTile(t)") != null);
    // the state in force and whether it is the default, from the schema's own defaults (so it works whichever way the default is set)
    try std.testing.expect(std.mem.indexOf(u8, js, "btn.textContent = `Turbo ${g.shownOn ? 'on' : 'off'}`;") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "(the default)") != null);
    // a click asks first; only the confirm stages, option by option (the default is 'unset', anything else an explicit value)
    try std.testing.expect(std.mem.indexOf(u8, js, "confirming = true") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "yes.addEventListener('click'") != null and std.mem.indexOf(u8, js, "goTo(!g.shownOn)") != null and std.mem.indexOf(u8, js, "? 'unset' : want.pr") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "S.status.turbo") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "btn.disabled = !g.measured || !g.can || confirming;") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "Turbo is not measured on this build") != null);
    // every figure is read from the record; the fixed-pool era's memory figure is gone
    for ([_][]const u8{ "rec.long_speedup", "rec.short_speedup", "rec.beside_prompt_ratio", "rec.beside_streams_ratio", "rec.pool_cost_positions", "rec.pool_cost_pct" }) |f| try std.testing.expect(std.mem.indexOf(u8, js, f) != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "mem_cost_gib") == null);
    try std.testing.expect(std.mem.indexOf(u8, js, "Applies at the next restart") != null);
}

test "every option /schema serves is reachable in the Settings tab: it sits in a tab the schema lists, and the page gives an unlisted tab a group of its own" {
    var shown: usize = 0;
    var seen_split64n = false;
    var seen_vmm = false;
    var seen_free_target = false;
    for (&settings.options) |*o| {
        if (o.hidden) continue;
        shown += 1;
        var listed = false;
        for (&settings.tabs) |*t| listed = listed or std.mem.eql(u8, t.name, o.tab);
        if (!listed) std.debug.print("option {s} sits in tab `{s}`, which the schema does not list\n", .{ o.name, o.tab });
        try std.testing.expect(listed);
        seen_split64n = seen_split64n or std.mem.eql(u8, o.name, "tf_ds_split64n");
        seen_vmm = seen_vmm or std.mem.eql(u8, o.name, "tf_ds_kv_vmm");
        seen_free_target = seen_free_target or std.mem.eql(u8, o.name, "tf_ds_kv_free_target_gib");
    }
    try std.testing.expectEqual(@as(usize, 291), shown); // the 316 options less the 25 the API never serves
    try std.testing.expect(seen_split64n and seen_vmm and seen_free_target);
    // the page's side of it: the tab list is built from the schema's, then every option's own tab is added when missing; a row's doc is on hover
    const js = admin.page_js;
    try std.testing.expect(std.mem.indexOf(u8, js, "if (!listed.some((t) => t.name === n)) listed.push({ name: n, title: n, about: '' });") != null);
    try std.testing.expect(std.mem.indexOf(u8, js, "title: (o.doc || o.label || o.name) + '\\n' + o.env") != null);
}

test "no hidden row is in /schema or /config, and the hidden set is the key file's, the debug rows and the test-only options" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var expected: usize = 0;
    for (&settings.options) |*o| {
        try std.testing.expectEqual(should, o.hidden);
        expected += @intFromBool(should);
    }
    try std.testing.expectEqual(@as(usize, 25), expected);
    for ([_][]const u8{ "schema", "config", "status" }) |name| {
        var buf: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "/admin/api/v1/{s}", .{name});
        const r = try request(rig.srv, "GET", p, bearer);
        defer r.free();
        for (&settings.options) |*o| if (o.hidden) {
            try std.testing.expect(std.mem.indexOf(u8, r.body, o.name) == null);
            try std.testing.expect(std.mem.indexOf(u8, r.body, o.env) == null);
        };
    }
    const config = try request(rig.srv, "GET", base("config"), bearer);
    defer config.free();
    const rows = (try json.parse(arena.allocator(), config.body)).ok.get("values").?.array;
    for (rows) |row| {
        const id = try std.fmt.parseInt(u32, row.get("id").?.int, 10);
    }
}

test "row 504, tf_ds_window, is the KV pool: a token count of at least the context, restart class, written down" {
    const o = settings.byId(504).?;
    try std.testing.expectEqualStrings("tf_ds_window", o.name);
    try std.testing.expectEqual(settings.Kind.int, o.kind);
    try std.testing.expectEqualStrings("tokens", o.unit);
    try std.testing.expectEqual(@as(f64, 1048576), o.min.?);
    try std.testing.expectEqual(settings.Apply.R, o.apply);
    try std.testing.expect(o.doc.len > 100);
    try std.testing.expect(std.mem.indexOf(u8, o.doc, "context is never lowered") != null);
    var buf: [200]u8 = undefined;
    try std.testing.expect(settings.refusal(o, "1048575", &buf) != null);
    try std.testing.expect(settings.refusal(o, "1048576", &buf) == null);
    try std.testing.expect(settings.refusal(o, "16777216", &buf) == null);
}

test "/requests: an in-flight request shows its label and counts, never any text, and the entry is gone when its reply ends" {
    const canary = "CANARY-PROMPT-7f3a91";
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open("shared-key-1", admin_keys_file);
    defer rig.close();
    rig.engine.watch = rig.srv;
    defer if (rig.engine.seen) |s| std.testing.allocator.free(s);
    const chat = try exchange(rig.srv, "POST", "/v1/chat/completions", "Authorization: Bearer shared-key-1\r\n" ++ json_head, "{\"model\": \"m\", \"max_tokens\": 4, \"messages\": [{\"role\": \"user\", \"content\": \"" ++ canary ++ "\"}]}");
    defer chat.free();
    // the request's own slices do not outlive its dispatch: a later submit that no dispatch carried finds the default
    try std.testing.expectEqualStrings("", admin_requests.context.route);
    try std.testing.expectEqualStrings("unauthenticated", admin_requests.context.label);
    // the engine refused the submit (its queue is closed), and the request was tracked from before the submit to its end
    try std.testing.expectEqual(@as(u32, 1), rig.engine.submits);
    try std.testing.expectEqual(@as(u32, 1), rig.engine.seen_live);
    const seen = rig.engine.seen.?;
    try std.testing.expect(std.mem.indexOf(u8, seen, "\"key_label\": \"cli-1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, seen, "\"endpoint\": \"chat\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, seen, "\"prompt_tokens\": ") != null);
    try std.testing.expect(std.mem.indexOf(u8, seen, canary) == null);
    try std.testing.expectEqual(@as(u32, 0), rig.srv.requests.liveCount(std.testing.io));
    // no admin answer holds the canary, the key or the prompt
    inline for (get_routes) |name| {
        const r = try request(rig.srv, "GET", "/admin/api/v1/" ++ name, bearer);
        defer r.free();
        try std.testing.expect(std.mem.indexOf(u8, r.body, canary) == null);
    }
    const after = try request(rig.srv, "GET", base("requests"), bearer);
    defer after.free();
    try std.testing.expect(std.mem.indexOf(u8, after.body, "\"in_flight\": []") != null);
    try std.testing.expect(std.mem.indexOf(u8, after.body, "\"finish\": \"error\"") != null);
    // the replies counted by key label (a connection's hook counts them; here the counter is fed as it would be)
    rig.srv.metrics.httpRequest(std.testing.io, "cli-1", 200);
    rig.srv.metrics.httpRequest(std.testing.io, "cli-1", 429);
    rig.srv.metrics.httpRequest(std.testing.io, "other", 200);
    const counted = try request(rig.srv, "GET", base("requests"), bearer);
    defer counted.free();
    try std.testing.expect(std.mem.indexOf(u8, counted.body, "\"label\": \"cli-1\", \"requests\": 2, \"statuses\": {\"200\": 1, \"429\": 1}") != null);
    try std.testing.expect(std.mem.indexOf(u8, counted.body, "\"label\": \"other\", \"requests\": 1") != null);
}

test "/requests shows a request still in flight with the tokens its reply holds so far" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    const e = rig.srv.requests.begin(std.testing.allocator, std.testing.io, 0, 41, .{ .label = "gateway", .route = "/v1/messages" }, true, true, 900).?;
    e.generated.store(17, .monotonic);
    const r = try request(rig.srv, "GET", base("requests"), bearer_read);
    defer r.free();
    try std.testing.expectEqual(@as(u16, 200), r.status);
    for ([_][]const u8{ "\"key_label\": \"gateway\"", "\"endpoint\": \"messages\"", "\"priority\": \"background\"", "\"stream\": true", "\"prompt_tokens\": 900", "\"generated_tokens\": 17", "\"id\": 41" }) |want|
        try std.testing.expect(std.mem.indexOf(u8, r.body, want) != null);
    rig.srv.requests.end(std.testing.io, 1, e, .{});
    const gone = try request(rig.srv, "GET", base("requests"), bearer_read);
    defer gone.free();
    try std.testing.expect(std.mem.indexOf(u8, gone.body, "\"in_flight\": []") != null);
}

test "/status says whether the server is idle, from its own record" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    const quiet = try request(rig.srv, "GET", base("status"), bearer);
    defer quiet.free();
    try std.testing.expect(std.mem.indexOf(u8, quiet.body, "\"idle\": true") != null);
    try std.testing.expect(std.mem.indexOf(u8, quiet.body, "\"basis\": \"server\"") != null);
    const e = rig.srv.requests.begin(std.testing.allocator, std.testing.io, 0, 1, .{}, false, false, 1).?;
    const busy = try request(rig.srv, "GET", base("status"), bearer);
    defer busy.free();
    try std.testing.expect(std.mem.indexOf(u8, busy.body, "\"idle\": false") != null);
    rig.srv.requests.end(std.testing.io, 1, e, .{});
}

test "/series?name=seat_kv returns every seat's KV share (prompt plus reply so far) as points and peaks, and /requests names each request's slot and KV tokens" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const sampler = try admin_series.Sampler.create(std.testing.allocator);
    defer sampler.destroy();
    rig.srv.series = sampler;
    defer rig.srv.series = null;
    const e = rig.srv.requests.begin(std.testing.allocator, std.testing.io, 0, 77, .{ .label = "gateway", .route = "/v1/messages" }, false, true, 1000).?;
    defer rig.srv.requests.end(std.testing.io, 1, e, .{});
    e.generated.store(24, .monotonic);
    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        rig.srv.requests.seatKv(std.testing.io, &sampler.seat_kv);
        sampler.sample(std.testing.io, 5_000 + i * 1000, .{}, null);
    }
    const r = try request(rig.srv, "GET", base("series") ++ "?name=seat_kv&seconds=600&step=1", bearer_read);
    defer r.free();
    try std.testing.expectEqual(@as(u16, 200), r.status);
    const doc = (try json.parse(arena.allocator(), r.body)).ok;
    const seats = doc.get("seats").?.array;
    try std.testing.expectEqual(@as(usize, admin_series.seat_rows), seats.len);
    try std.testing.expectEqual(@as(usize, 3), seats[0].get("points").?.array.len);
    try std.testing.expectEqual(@as(f64, 1024), seats[0].get("points").?.array[2].float);
    try std.testing.expectEqual(@as(f64, 0), seats[1].get("points").?.array[2].float); // a free slot is zero, not missing
    try std.testing.expectEqualStrings("7000", doc.get("last_ms").?.int);
    const flying = try request(rig.srv, "GET", base("requests"), bearer_read);
    defer flying.free();
    const row = (try json.parse(arena.allocator(), flying.body)).ok.get("in_flight").?.array[0];
    try std.testing.expectEqualStrings("1024", row.get("kv_tokens").?.int);
    try std.testing.expectEqualStrings("0", row.get("slot").?.int);
}

test "/status carries the release (null when none was set) and each rank's memory now: rank 0 read live, a follower as it last reported with its age, never zeros" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    api.peer_mem.reset();
    defer api.peer_mem.reset();
    const quiet = try request(rig.srv, "GET", base("status"), bearer);
    defer quiet.free();
    const doc = (try json.parse(arena.allocator(), quiet.body)).ok;
    try std.testing.expect(doc.get("release") != null);
    const hm = doc.get("host_memory").?;
    try std.testing.expect(std.mem.indexOf(u8, hm.get("basis").?.string, "every 2 s") != null);
    // no start record in this rig: rank 0 only; /proc/meminfo exists where the tests run (Linux), else the figures are null
    const rows = hm.get("ranks").?.array;
    try std.testing.expectEqual(@as(usize, 1), rows.len);
    try std.testing.expectEqualStrings("0", rows[0].get("rank").?.int);
    if (rows[0].get("available_bytes").? != .null) try std.testing.expect((try std.fmt.parseInt(u64, rows[0].get("available_bytes").?.int, 10)) > 0);
    // a follower's report shows with its figures and an age; its silence before that is no row of zeros
    api.peer_mem.record(2, 11 << 30, 121 << 30);
    const heard = try request(rig.srv, "GET", base("status"), bearer);
    defer heard.free();
    const rows2 = (try json.parse(arena.allocator(), heard.body)).ok.get("host_memory").?.get("ranks").?.array;
    try std.testing.expectEqual(@as(usize, 2), rows2.len);
    try std.testing.expectEqualStrings("2", rows2[1].get("rank").?.int);
    try std.testing.expectEqual(@as(u64, 11 << 30), try std.fmt.parseInt(u64, rows2[1].get("available_bytes").?.int, 10));
    try std.testing.expectEqual(@as(u64, 121 << 30), try std.fmt.parseInt(u64, rows2[1].get("total_bytes").?.int, 10));
    try std.testing.expect(rows2[1].get("age_ms").? == .int);
}

test "/status carries each Spark's health block: figures the host gave, null for the rest, the GPU's clock events by name, the rails' counters whole" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    api.peer_mem.reset();
    defer api.peer_mem.reset();
    const quiet = try request(rig.srv, "GET", base("status"), bearer);
    defer quiet.free();
    const nh0 = (try json.parse(arena.allocator(), quiet.body)).ok.get("node_health").?;
    try std.testing.expectEqual(@as(usize, 8), nh0.get("zone_names").?.array.len);
    try std.testing.expectEqual(@as(usize, 1), nh0.get("ranks").?.array.len); // rank 0 only, no report yet
    try std.testing.expect(nh0.get("ranks").?.array[0].get("age_ms").? == .null);
    var h = api.peer_mem.emptyHealth();
    h.mem_available = 11 << 30;
    h.mem_total = 121 << 30;
    h.load1_x100 = 125;
    h.zone_mc[1] = 41_500;
    h.gpu_clock_mhz = 2190;
    h.gpu_clock_max_mhz = 3003;
    h.gpu_power_mw = 38_250;
    h.gpu_reasons = 0x1 | 0x20;
    h.rails[0] = .{ .rx_bytes = 1000, .tx_bytes = 2000, .errors = 3, .discards = 0 };
    api.peer_mem.recordHealth(2, &h);
    const heard = try request(rig.srv, "GET", base("status"), bearer);
    defer heard.free();
    const nh = (try json.parse(arena.allocator(), heard.body)).ok.get("node_health").?;
    const rows = nh.get("ranks").?.array;
    try std.testing.expectEqual(@as(usize, 2), rows.len);
    const r2 = rows[1];
    try std.testing.expectEqualStrings("2", r2.get("rank").?.int);
    try std.testing.expectEqual(@as(f64, 1.25), r2.get("load1").?.float);
    try std.testing.expect(r2.get("cpu_busy_pct").? == .null); // never a zero for a rate not yet known
    try std.testing.expectEqual(@as(f64, 41.5), r2.get("zones_c").?.array[1].float);
    try std.testing.expect(r2.get("zones_c").?.array[0] == .null);
    const gpu = r2.get("gpu").?;
    try std.testing.expectEqualStrings("2190", gpu.get("clock_mhz").?.int);
    try std.testing.expectEqualStrings("3003", gpu.get("clock_max_mhz").?.int);
    try std.testing.expectEqual(@as(f64, 38.25), gpu.get("power_w").?.float);
    try std.testing.expect(gpu.get("temp_c").? == .null);
    try std.testing.expect(gpu.get("idle").?.bool);
    const why = gpu.get("reasons").?.array;
    try std.testing.expectEqual(@as(usize, 1), why.len);
    try std.testing.expectEqualStrings("sw_thermal_slowdown", why[0].string);
    const rails = r2.get("rails").?.array;
    try std.testing.expectEqualStrings("2000", rails[0].get("tx_bytes").?.int);
    try std.testing.expectEqualStrings("3", rails[0].get("errors").?.int);
    try std.testing.expect(rails[1].get("rx_bytes").? == .null);
    try std.testing.expect(r2.get("nvme").?.get("wear_pct").? == .null);
}

test "/status carries the Turbo record: the measured one built in, and a record that lacks a number the tile uses reads not measured; only the known fields pass" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try request(rig.srv, "GET", base("status"), bearer);
    defer r.free();
    const built_in = (try json.parse(a, r.body)).ok.get("turbo").?;
    try std.testing.expect(built_in.get("measured").?.bool);
    try std.testing.expectEqualStrings("built in", built_in.get("source").?.string);
    try std.testing.expectEqual(@as(f64, 1.110), built_in.get("long_speedup").?.float);
    try std.testing.expectEqual(@as(f64, 43.2), built_in.get("pool_cost_pct").?.float);
    try std.testing.expectEqualStrings("7929856", built_in.get("pool_cost_positions").?.int);
    try std.testing.expect(built_in.get("mem_cost_gib") == null);
    // a complete record from a file
    const good = try admin.turboFrom(a, "{\"measured\": true, \"build\": \"T16\", \"long_speedup\": 1.07, \"short_speedup\": 1.02, \"beside_streams_ratio\": 1.1, \"pool_cost_positions\": 4000000, \"pool_cost_pct\": 30.5, \"replies_equal\": true, \"evil\": {\"x\": 1}}", "record file");
    try std.testing.expect(good.get("measured").?.bool);
    try std.testing.expectEqual(@as(f64, 1.07), good.get("long_speedup").?.float);
    try std.testing.expect(good.get("evil") == null);
    // each failure reads not measured: no flag, a missing number, a number that is text, text that is not JSON, JSON that is not an object
    for ([_][]const u8{
        "{\"measured\": false, \"long_speedup\": 1.07, \"short_speedup\": 1, \"pool_cost_positions\": 4, \"pool_cost_pct\": 3}",
        "{\"measured\": true, \"long_speedup\": 1.07, \"pool_cost_positions\": 4, \"pool_cost_pct\": 3}",
        "{\"measured\": true, \"long_speedup\": \"fast\", \"short_speedup\": 1, \"pool_cost_positions\": 4, \"pool_cost_pct\": 3}",
        "{\"measured\": true, \"long_speedup\": 1.07, \"short_speedup\": 1, \"pool_cost_pct\": 3}",
        "{\"measured\": true, \"long_speedup\": 1.07, \"short_speedup\": 1, \"pool_cost_positions\": \"many\", \"pool_cost_pct\": 3}",
        "{\"measured\": true, \"long_speedup\": 1.07, \"short_speedup\": 1, \"mem_cost_gib\": [4, 3]}",
        "not json",
        "[1, 2]",
        "",
    }) |text| {
        const v = try admin.turboFrom(a, text, "record file");
        try std.testing.expect(!v.get("measured").?.bool);
        try std.testing.expect(v.get("long_speedup") == null);
    }
}

test "/status names the checkpoint, not the served alias, and carries what the start decided about the transports (null where nothing was recorded)" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("DeepSeek V4.1 Flash 1017", (try admin.modelTitle(a, "/data/models/DeepSeek-V4.1-Flash-1017/")).?);
    try std.testing.expectEqualStrings("DeepSeek V4.1 Flash", (try admin.modelTitle(a, "/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/0123456789abcdef0123456789abcdef01234567")).?);
    try std.testing.expect((try admin.modelTitle(a, "")) == null);
    // the baked served folder is called "model": the name comes from the folder's own metadata
    try std.testing.expect((try admin.modelTitle(a, "/hf/tensorfold-dsv41/1bae0e10/model")) == null);
    try std.testing.expect((try admin.modelTitle(a, "/hf/x/0123456789abcdef0123456789abcdef01234567")) == null);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // 1. config.json names its source
    try tmp.dir.createDirPath(std.testing.io, "named/model");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "named/model/config.json", .data = "{\"model_type\": \"deepseek_v4\", \"_name_or_path\": \"deepseek-ai/DeepSeek-V4.1-Flash-1017\"}" });
    const named = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/named/model", .{tmp.sub_path});
    try std.testing.expectEqualStrings("DeepSeek V4.1 Flash 1017", (try admin.modelName(std.testing.io, a, named)).?);
    // 2. the bake's layout: model/config.json is a relative link into hub/models--org--name/snapshots/rev/
    try tmp.dir.createDirPath(std.testing.io, "baked/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/abc123def456");
    try tmp.dir.createDirPath(std.testing.io, "baked/data/model");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "baked/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/abc123def456/config.json", .data = "{\"architectures\": [\"DeepseekV4ForCausalLM\"]}" });
    const link = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/baked/data/model/config.json", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlink("../../hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/abc123def456/config.json", link));
    const baked = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/baked/data/model", .{tmp.sub_path});
    try std.testing.expectEqualStrings("DeepSeek V4.1 Flash 1017", (try admin.modelName(std.testing.io, a, baked)).?);
    // 3. nothing names it: a folder called model with a config that gives no name reads null; a folder with a real name reads that
    try tmp.dir.createDirPath(std.testing.io, "bare/model");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "bare/model/config.json", .data = "{\"architectures\": [\"X\"]}" });
    try std.testing.expect((try admin.modelName(std.testing.io, a, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/bare/model", .{tmp.sub_path}))) == null);
    try tmp.dir.createDirPath(std.testing.io, "Some-Model-7");
    try std.testing.expectEqualStrings("Some Model 7", (try admin.modelName(std.testing.io, a, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/Some-Model-7", .{tmp.sub_path}))).?);
    const saved = api.transport.facts;
    defer api.transport.facts = saved;
    api.transport.facts = .{ .rocinante_known = true, .rocinante_wait = true, .rocinante_signal = .atomic, .rings_known = true, .rings_on = true, .rings_wait = false, .rings_word = true, .rings_signal = .write, .watch_links = 3 };
    const r = try request(rig.srv, "GET", base("status"), bearer);
    defer r.free();
    const doc = (try json.parse(a, r.body)).ok;
    try std.testing.expect(doc.get("model_name").? == .null); // the rig has no model directory
    const t = doc.get("transport").?;
    const p = t.get("prompt_exchanges").?;
    try std.testing.expect(p.get("waits_in_card").?.bool);
    try std.testing.expectEqualStrings("atomic", p.get("arrival_signal").?.string);
    const g = t.get("generation_exchanges").?;
    try std.testing.expect(g.get("on").?.bool and !g.get("waits_in_card").?.bool);
    try std.testing.expectEqualStrings("write", g.get("arrival_signal").?.string);
    try std.testing.expectEqualStrings("3", t.get("link_watch").?.get("links").?.int);
    api.transport.facts = .{};
    const none = try request(rig.srv, "GET", base("status"), bearer);
    defer none.free();
    const t2 = (try json.parse(a, none.body)).ok.get("transport").?;
    try std.testing.expect(!t2.get("prompt_exchanges").?.get("known").?.bool);
    try std.testing.expect(t2.get("prompt_exchanges").?.get("arrival_signal").? == .null);
}

test "/logs: the ring's lines in order, cut at 512 bytes, with what was dropped; admin only" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    log.ring.reset();
    defer log.ring.reset();
    log.line("first line", .{});
    log.line("second line, {s}", .{"formatted"});
    const long = comptime repeat("x", 600);
    log.line("{s}", .{long});
    log.line("ROCI failed on rail 1", .{});
    log.line("refused cmpl-9: a request the engine would not take", .{});
    const r = try request(rig.srv, "GET", base("logs"), bearer);
    defer r.free();
    try std.testing.expectEqual(@as(u16, 200), r.status);
    const doc = (try json.parse(arena.allocator(), r.body)).ok;
    const lines = doc.get("lines").?.array;
    try std.testing.expectEqual(@as(usize, 5), lines.len);
    try std.testing.expectEqualStrings("first line", lines[0].get("text").?.string);
    try std.testing.expectEqualStrings("second line, formatted", lines[1].get("text").?.string);
    try std.testing.expectEqual(@as(usize, 512), lines[2].get("text").?.string.len);
    try std.testing.expectEqualStrings("1", lines[0].get("seq").?.int);
    try std.testing.expectEqualStrings("5", lines[4].get("seq").?.int);
    try std.testing.expectEqualStrings("6", doc.get("next_seq").?.int);
    try std.testing.expectEqualStrings("0", doc.get("dropped").?.int);
    try std.testing.expectEqualStrings("error", lines[3].get("level").?.string);
    try std.testing.expectEqualStrings("warn", lines[4].get("level").?.string);
    // since, level, contains and limit
    const tail = try request(rig.srv, "GET", base("logs") ++ "?since=3", bearer);
    defer tail.free();
    try std.testing.expect(std.mem.indexOf(u8, tail.body, "first line") == null and std.mem.indexOf(u8, tail.body, "ROCI failed") != null);
    const errors_only = try request(rig.srv, "GET", base("logs") ++ "?level=error", bearer);
    defer errors_only.free();
    try std.testing.expect(std.mem.indexOf(u8, errors_only.body, "ROCI failed") != null and std.mem.indexOf(u8, errors_only.body, "second line") == null);
    const grep = try request(rig.srv, "GET", base("logs") ++ "?contains=second+line", bearer);
    defer grep.free();
    try std.testing.expect(std.mem.indexOf(u8, grep.body, "second line") != null and std.mem.indexOf(u8, grep.body, "first line") == null);
    const two = try request(rig.srv, "GET", base("logs") ++ "?limit=2", bearer);
    defer two.free();
    const two_doc = (try json.parse(arena.allocator(), two.body)).ok;
    try std.testing.expectEqual(@as(usize, 2), two_doc.get("lines").?.array.len);
    try std.testing.expectEqualStrings("3", two_doc.get("next_seq").?.int);
    const bad = try request(rig.srv, "GET", base("logs") ++ "?level=loud", bearer);
    defer bad.free();
    try std.testing.expectEqual(@as(u16, 400), bad.status);
    // a read key may not read them
    const denied = try request(rig.srv, "GET", base("logs"), bearer_read);
    defer denied.free();
    try std.testing.expectEqual(@as(u16, 403), denied.status);
    try std.testing.expect(std.mem.indexOf(u8, denied.body, "first line") == null);
}

test "the log ring overwrites its oldest lines and says how many a reader missed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    log.ring.reset();
    defer log.ring.reset();
    var i: usize = 0;
    while (i < log.ring.capacity + 30) : (i += 1) log.line("line {d}", .{i});
    const page = try log.ring.read(arena.allocator(), 0, 10000, .info, "");
    try std.testing.expectEqual(@as(usize, log.ring.capacity), page.lines.len);
    try std.testing.expectEqualStrings("line 30", page.lines[0].text[0..page.lines[0].len]);
    try std.testing.expectEqual(@as(u64, 30), page.dropped);
    try std.testing.expectEqual(@as(u64, log.ring.capacity + 31), page.next_seq);
    const caught_up = try log.ring.read(arena.allocator(), page.next_seq - 1, 100, .info, "");
    try std.testing.expectEqual(@as(usize, 0), caught_up.lines.len);
    try std.testing.expectEqual(@as(u64, 0), caught_up.dropped);
}

test "a log line keeps a multi-byte character whole when the ring cuts it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    log.ring.reset();
    defer log.ring.reset();
    const accents = comptime repeat("\u{e9}", 300); // two bytes a character: byte 512 is the second half of one
    log.line("{s}", .{accents});
    const page = try log.ring.read(arena.allocator(), 0, 10, .info, "");
    const text = page.lines[0].text[0..page.lines[0].len];
    try std.testing.expect(std.unicode.utf8ValidateSlice(text));
    try std.testing.expectEqual(@as(usize, 512), text.len);
}

test "the access line in the ring carries no query string: a key or a prompt there stays out of /logs" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open("shared-key-1", admin_keys_file);
    defer rig.close();
    log.ring.reset();
    defer log.ring.reset();
    const r = try request(rig.srv, "GET", "/v1/nothing?api_key=CANARY-KEY-5521&q=CANARY-QUERY", "Authorization: Bearer shared-key-1\r\n");
    defer r.free();
    const logs = try request(rig.srv, "GET", base("logs"), bearer);
    defer logs.free();
    try std.testing.expectEqual(@as(u16, 200), logs.status);
    try std.testing.expect(std.mem.indexOf(u8, logs.body, "CANARY") == null);
    try std.testing.expect(std.mem.indexOf(u8, logs.body, "GET /v1/nothing HTTP/1.1") != null);
    try std.testing.expect(std.mem.indexOf(u8, logs.body, "shared-key-1") == null);
}

test "/series returns what the sampler took, by name, step and window" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const none = try request(rig.srv, "GET", base("series") ++ "?name=decode_tps", bearer_read);
    defer none.free();
    try std.testing.expectEqual(@as(u16, 200), none.status);
    try std.testing.expect(std.mem.indexOf(u8, none.body, "\"points\": []") != null);
    const sampler = try admin_series.Sampler.create(std.testing.allocator);
    defer sampler.destroy();
    rig.srv.series = sampler;
    defer rig.srv.series = null;
    var t: i64 = 5_000;
    var i: u32 = 0;
    while (i < 4) : (i += 1) {
        sampler.sample(std.testing.io, t, .{ .decode_tokens_per_second = @floatFromInt(100 + i), .waiting = i }, null);
        t += 1000;
    }
    const r = try request(rig.srv, "GET", base("series") ++ "?name=decode_tps&seconds=600&step=1", bearer_read);
    defer r.free();
    try std.testing.expectEqual(@as(u16, 200), r.status);
    const doc = (try json.parse(arena.allocator(), r.body)).ok;
    try std.testing.expectEqualStrings("decode_tps", doc.get("name").?.string);
    try std.testing.expectEqualStrings("5000", doc.get("since_ms").?.int);
    const points = doc.get("points").?.array;
    try std.testing.expectEqual(@as(usize, 4), points.len);
    try std.testing.expectEqual(@as(f64, 100), points[0].float);
    try std.testing.expectEqual(@as(f64, 103), points[3].float);
    // the highest sample of each bucket and the newest sample's time come with the means (the page's hover and its time axis)
    const peaks = doc.get("peaks").?.array;
    try std.testing.expectEqual(@as(usize, 4), peaks.len);
    try std.testing.expectEqual(@as(f64, 103), peaks[3].float);
    try std.testing.expectEqualStrings("8000", doc.get("last_ms").?.int);
    const waiting = try request(rig.srv, "GET", base("series") ++ "?name=waiting&seconds=2", bearer_read);
    defer waiting.free();
    try std.testing.expectEqual(@as(usize, 2), (try json.parse(arena.allocator(), waiting.body)).ok.get("points").?.array.len);
    const unnamed = try request(rig.srv, "GET", base("series"), bearer_read);
    defer unnamed.free();
    try std.testing.expectEqual(@as(u16, 400), unnamed.status);
    const wrong = try request(rig.srv, "GET", base("series") ++ "?name=bogus", bearer_read);
    defer wrong.free();
    try std.testing.expectEqual(@as(u16, 400), wrong.status);
    try std.testing.expectEqual(@as(u32, 0), rig.engine.submits);
}

test "/keys names labels and counts, never a key, a digest or a file's path" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    const r = try request(rig.srv, "GET", base("keys"), bearer);
    defer r.free();
    try std.testing.expectEqual(@as(u16, 200), r.status);
    for ([_][]const u8{ key_admin, key_read, key_agent, key_other }) |k| {
        try std.testing.expect(std.mem.indexOf(u8, r.body, k) == null);
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(k, &d, .{});
        try std.testing.expect(std.mem.indexOf(u8, r.body, &std.fmt.bytesToHex(d, .lower)) == null);
    }
    try std.testing.expect(std.mem.indexOf(u8, r.body, "admin-keys") == null);
    try std.testing.expect(std.mem.indexOf(u8, r.body, ".zig-cache") == null);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "\"label\": \"read-ops\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "\"label\": \"ops\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "\"valid\": true") != null);
}

test "history, leases and policy are empty views that say why" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    for ([_]struct { name: []const u8, list: []const u8 }{ .{ .name = "history", .list = "\"rows\": []" }, .{ .name = "leases", .list = "\"leases\": []" }, .{ .name = "policy", .list = "\"profiles\": []" } }) |c| {
        var buf: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "/admin/api/v1/{s}", .{c.name});
        const r = try request(rig.srv, "GET", p, bearer_read);
        defer r.free();
        try std.testing.expectEqual(@as(u16, 200), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, c.list) != null);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "\"note\"") != null);
    }
}

test "the access line names an admin request by its admin key's label, and an answered read poll prints none" {
    var captured: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer captured.deinit();
    log.capture = &captured.writer;
    defer log.capture = null;
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open("shared-key-1", admin_keys_file);
    defer rig.close();
    captured.clearRetainingCapacity();
    // a read poll answered 200 is quiet, as /health's is
    const poll = try request(rig.srv, "GET", base("status"), bearer_read);
    defer poll.free();
    try std.testing.expectEqual(@as(u16, 200), poll.status);
    try std.testing.expectEqualStrings("", captured.written());
    // a refusal and a write print, under the key's own label and never as unauthenticated
    const denied = try request(rig.srv, "GET", base("keys"), bearer_read);
    defer denied.free();
    try std.testing.expect(std.mem.indexOf(u8, captured.written(), "\"GET /admin/api/v1/keys HTTP/1.1\" 403 - key=read-ops") != null);
    const reload = try exchange(rig.srv, "POST", base("keys/reload"), bearer ++ json_head, "{}");
    defer reload.free();
    try std.testing.expect(std.mem.indexOf(u8, captured.written(), "\"POST /admin/api/v1/keys/reload HTTP/1.1\" 200 - key=admin\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, captured.written(), "unauthenticated") == null);
    // a stranger's line names no admin key
    const stranger = try request(rig.srv, "GET", base("status"), "");
    defer stranger.free();
    try std.testing.expect(std.mem.indexOf(u8, captured.written(), "\"GET /admin/api/v1/status HTTP/1.1\" 404 -") != null);
    try std.testing.expect(std.mem.indexOf(u8, captured.written(), "404 - key=") == null);
}

test "with polls logged, an answered admin read prints its line under the key's label" {
    var captured: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer captured.deinit();
    log.capture = &captured.writer;
    defer log.capture = null;
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open("shared-key-1", admin_keys_file);
    defer rig.close();
    rig.srv.config.log_polls = true;
    captured.clearRetainingCapacity();
    const r = try request(rig.srv, "GET", base("status"), bearer_agent);
    defer r.free();
    try std.testing.expect(std.mem.indexOf(u8, captured.written(), "\"GET /admin/api/v1/status HTTP/1.1\" 200 - key=agent-night") != null);
}
    // the pool (TF_DS_WINDOW) and the free target may not be named together, so the pool's row is asked in a group of its own
    var checked: usize = 0;
    var buf: [400]u8 = undefined;
    for ([_]u8{ 0, 1, 2 }) |group| {
            if ((@as(u8, if (o.id == 504) 1 else if (o.id == 122) 2 else 0)) != group) continue;
        try std.testing.expect(group != 0 or rows.items.len > 200);
                    // the pool below the context has its own name; every other code is settings.check's
                    const code: []const u8 = if (o.id == 504 and w.code == .below_minimum) "pool_below_window" else @tagName(w.code);
                    try std.testing.expectEqualStrings(code, got.?.get("code").?.string);
    }
        // a row with a live hook (the rounds beside) is taken at the next loop turn; every other row waits for the next start
        const hooked = std.mem.eql(u8, name, "tf_ds_rounds_beside");
        try std.testing.expectEqual(!hooked, x.get("needs_restart").?.bool);
        try std.testing.expectEqualStrings(if (hooked) "live" else "next_start", x.get("apply_mode").?.string);
        try std.testing.expectEqualStrings(if (hooked) "active" else "pending_restart", x.get("state_would_be").?.string);
    var idle_off = false;
    for (doc.get("modes").?.array) |m| if (std.mem.eql(u8, m.get("mode").?.string, "idle")) {
        idle_off = !m.get("available").?.bool and m.get("why").?.string.len > 0; // there is no class I apply on this engine
    try std.testing.expect(idle_off);
        .{ .body = "{\"set\": {\"tf_ds_rounds_beside\": 1}, \"apply\": \"idle\"}", .status = 400, .code = "apply_not_supported" },

// ---- step 4: the ledger, the pool and what a dry run says of it

const kv = api.kv_pool;
const live_config_for_test = api.live_config;
const Lin = struct {
    pub fn bytes(_: Lin, positions: usize) usize {
        return @intCast(gib_bytes + 890 * positions);
    }
    const gib_bytes: u64 = 1 << 30;
};

/// A start record of four Sparks: what each had free before the pool, the ledger each voted with, and the pool the vote chose.
fn fakeStart(free_gib: [4]u64, target_gib: u64) void {
    const lin: Lin = .{};
    const g: u64 = 1 << 30;
    var rep: kv.Report = .{};
    rep.world = 4;
    for (0..4) |r| {
        var p = kv.Part.init(@intCast(r));
        p.free_bytes = free_gib[r] * g;
        p.total_bytes = 121 * g;
        p.target_bytes = target_gib * g;
        p.growth_bytes = 6 * g;
        p.floor = 1_048_576;
        p.asked = 1_048_576;
        p.fit = if (target_gib != 0) kv.fit(p.free_bytes, p.target_bytes, p.growth_bytes, lin) else 0;
        p.floor_bytes = lin.bytes(1_048_576);
        p.b_lo = lin.bytes(kv.pool_step);
        p.b_hi = lin.bytes(17 * kv.pool_step);
        p.bytes_per_token = 890;
        p.n = 3;
        p.entries[0] = .{ .feature = @backingInt(kv.Feature.weights), .device = (80 - @as(u64, r)) * g, .host = 0 };
        p.entries[1] = .{ .feature = @backingInt(kv.Feature.transport), .device = 3 * g, .host = 1 << 20 };
        p.entries[2] = .{ .feature = @backingInt(kv.Feature.other), .device = 121 * g -| p.free_bytes -| (83 - @as(u64, r)) * g, .host = 0 };
        rep.table[r] = p;
        rep.final[r] = p;
        rep.final[r].free_bytes = p.free_bytes - 2 * g; // what the rest of the start took
        rep.final[r].n = 4;
        rep.final[r].entries[3] = .{ .feature = @backingInt(kv.Feature.kv_pool), .device = 2 * g, .host = 0 };
    }
    rep.have_final = true;
    rep.decision = kv.decide(rep.table[0..4]);
    rep.pool_active = if (rep.decision.verdict == .auto) rep.decision.pool else 1_048_576;
    rep.streams = 16;
    rep.pool_predicted = lin.bytes(rep.pool_active);
    rep.pool_taken = lin.bytes(rep.pool_active) + (100 << 20);
    rep.free_ready = rep.final[0].free_bytes;
    rep.mine.n = 4;
    rep.mine.sites[0] = .{ .file = "zig/src/families/dsv41/model.zig", .line = 2002 };
    rep.mine.entries[0] = rep.final[0].entries[0];
    rep.mine.sites[1] = .{ .file = "zig/src/families/dsv41/model.zig", .line = 2200 };
    rep.mine.entries[1] = rep.final[0].entries[1];
    rep.mine.sites[2] = .{ .file = "zig/src/core/kv_pool.zig", .line = 281 };
    rep.mine.entries[2] = rep.final[0].entries[2];
    rep.mine.sites[3] = .{ .file = "zig/src/families/dsv41/model.zig", .line = 2226 };
    rep.mine.entries[3] = rep.final[0].entries[3];
    kv.report = rep;
    kv.ready.store(true, .release);
}

fn noStart() void {
    kv.ready.store(false, .release);
    kv.report = .{};
}

test "/ledger: every rank's lines largest first with where each was taken, the ranks' sums, the pool and the vote; empty and saying so without a start record" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    noStart();
    const empty = try request(rig.srv, "GET", base("ledger"), bearer_read);
    defer empty.free();
    try std.testing.expect(std.mem.indexOf(u8, empty.body, "\"entries\": []") != null);
    try std.testing.expect(std.mem.indexOf(u8, empty.body, "\"note\"") != null);
    defer noStart();
    fakeStart(.{ 20, 18, 19, 17 }, 8);
    const r = try request(rig.srv, "GET", base("ledger"), bearer_read);
    defer r.free();
    try std.testing.expectEqual(@as(u16, 200), r.status);
    const doc = (try json.parse(a, r.body)).ok;
    const entries = doc.get("entries").?.array;
    try std.testing.expectEqual(@as(usize, 16), entries.len); // four lines on each of four ranks
    var prev: u64 = std.math.maxInt(u64);
    var sums: [4]u64 = @splat(0);
    for (entries) |e| {
        const bytes = try std.fmt.parseInt(u64, e.get("bytes").?.int, 10);
        try std.testing.expect(bytes <= prev); // largest first
        prev = bytes;
        const rank: usize = @intCast(try std.fmt.parseInt(u32, e.get("rank").?.int, 10));
        sums[rank] += try std.fmt.parseInt(u64, e.get("device_bytes").?.int, 10);
        try std.testing.expect(e.get("feature").?.string.len > 0);
        try std.testing.expect(e.get("site").?.string.len > 0);
        try std.testing.expectEqualStrings(if (std.mem.eql(u8, e.get("feature_id").?.string, "other")) "remainder" else "measured", e.get("label").?.string);
    }
    // the lines of a rank add up to what the device holds at the start's end: total less free
    const g: u64 = 1 << 30;
    for (doc.get("ranks").?.array, 0..) |rk, i| {
        const used = try std.fmt.parseInt(u64, rk.get("used_at_ready").?.int, 10);
        try std.testing.expectEqual(used, sums[i]);
        try std.testing.expectEqual(used, try std.fmt.parseInt(u64, rk.get("ledger_total").?.int, 10));
        try std.testing.expect(rk.get("pool_tokens").? != .null);
    }
    _ = g;
    const pool = doc.get("pool").?;
    try std.testing.expectEqualStrings("min over ranks", pool.get("vote").?.string);
    try std.testing.expectEqualStrings("8", pool.get("free_target_gib").?.string);
    try std.testing.expect(pool.get("predicted_bytes").? != .null and pool.get("taken_bytes").? != .null and pool.get("difference_gib").? != .null);
    try std.testing.expectEqualStrings("890", doc.get("bytes_per_token").?.int);
    // a site is where the start took the line, and the remainder's is where the ledger closed
    var saw_site = false;
    for (entries) |e| if (std.mem.eql(u8, e.get("site").?.string, "zig/src/families/dsv41/model.zig:2002")) {
        saw_site = true;
    };
    try std.testing.expect(saw_site);
    try std.testing.expectEqual(@as(u32, 0), rig.engine.submits);
}

test "/status carries the pool and each Spark's memory at the start once a start record exists, and omits both before" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    noStart();
    const none = try request(rig.srv, "GET", base("status"), bearer_read);
    defer none.free();
    const nd = (try json.parse(a, none.body)).ok;
    try std.testing.expect(nd.get("kv_pool") == null and nd.get("memory") == null); // never zero-filled
    defer noStart();
    fakeStart(.{ 20, 18, 19, 17 }, 8);
    const r = try request(rig.srv, "GET", base("status"), bearer_read);
    defer r.free();
    const doc = (try json.parse(a, r.body)).ok;
    const kvp = doc.get("kv_pool").?;
    try std.testing.expectEqualStrings("16777216", kvp.get("streams_x_context").?.int);
    try std.testing.expectEqualStrings("1048576", kvp.get("usable_window").?.int);
    try std.testing.expectEqualStrings("1048576", kvp.get("tokens_requested").?.int);
    try std.testing.expect(std.mem.eql(u8, kvp.get("mode").?.string, "free_target"));
    try std.testing.expectEqualStrings("8", kvp.get("free_target_gib").?.string);
    const active = try std.fmt.parseInt(u64, kvp.get("tokens_active").?.int, 10);
    try std.testing.expect(active % kv.pool_step == 0 and active >= 1_048_576);
    const extra = try std.fmt.parseInt(u64, kvp.get("extra_tokens").?.int, 10);
    try std.testing.expectEqual(if (active > 16_777_216) active - 16_777_216 else 0, extra); // the engine's own difference: the page never computes it
    const ranks = doc.get("memory").?.get("ranks").?.array;
    try std.testing.expectEqual(@as(usize, 4), ranks.len);
    try std.testing.expectEqualStrings("at the start", doc.get("memory").?.get("basis").?.string);
    try std.testing.expectEqualStrings("17", ranks[3].get("free_at_open_gib").?.string);
    try std.testing.expect(ranks[0].get("ledger_gib").? != .null and ranks[0].get("free_target_gib").? != .null);
    // a start with no target says the pool is the setting's
    fakeStart(.{ 20, 18, 19, 17 }, 0);
    const off = try request(rig.srv, "GET", base("status"), bearer_read);
    defer off.free();
    const od = (try json.parse(a, off.body)).ok.get("kv_pool").?;
    try std.testing.expectEqualStrings("set", od.get("mode").?.string);
    try std.testing.expect(od.get("free_target_gib").? == .null);
}

test "validate says what a free target would give the pool on each Spark, refuses one that cannot hold the context by name, and refuses a pool named twice" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    noStart();
    const cold = try postJson(&rig, base("validate"), bearer, "{\"set\": {\"tf_ds_kv_free_target_gib\": 12}}");
    defer cold.free();
    try std.testing.expect((try json.parse(a, cold.body)).ok.get("derived") == null); // no start record: nothing to derive from
    defer noStart();
    fakeStart(.{ 60, 50, 55, 58 }, 8);
    const same = try postJson(&rig, base("validate"), bearer, "{\"set\": {\"tf_ds_kv_free_target_gib\": 8}}");
    defer same.free();
    const sd = (try json.parse(a, same.body)).ok;
    try std.testing.expect(sd.get("ok").?.bool);
    const sk = sd.get("derived").?.get("kv_pool").?;
    try std.testing.expectEqual(@as(usize, 4), sk.get("ranks").?.array.len);
    const before = try std.fmt.parseInt(u64, sk.get("tokens_before").?.int, 10);
    try std.testing.expectEqual(before, try std.fmt.parseInt(u64, sk.get("tokens_after").?.int, 10)); // the target the start had: the same pool
    try std.testing.expectEqualStrings("estimated", sk.get("label").?.string);
    const more = try postJson(&rig, base("validate"), bearer, "{\"set\": {\"tf_ds_kv_free_target_gib\": 12}}");
    defer more.free();
    const mk = (try json.parse(a, more.body)).ok.get("derived").?.get("kv_pool").?;
    try std.testing.expect(try std.fmt.parseInt(u64, mk.get("tokens_after").?.int, 10) < before); // a larger target, a smaller pool
    for (mk.get("ranks").?.array) |rk| {
        try std.testing.expect(rk.get("tokens_after").? != .null and rk.get("free_after_gib").? != .null);
        try std.testing.expect(rk.get("tokens_before").? != .null);
    }
    // a target the Sparks cannot keep and still hold the context
    const huge = try postJson(&rig, base("validate"), bearer, "{\"set\": {\"tf_ds_kv_free_target_gib\": 100}}");
    defer huge.free();
    const hd = (try json.parse(a, huge.body)).ok;
    try std.testing.expect(!hd.get("ok").?.bool);
    const x = hd.get("refused").?.array[0];
    try std.testing.expectEqualStrings("memory_does_not_fit", x.get("code").?.string);
    try std.testing.expect(std.mem.indexOf(u8, x.get("message").?.string, "GiB short") != null);
    try std.testing.expect(std.mem.indexOf(u8, x.get("message").?.string, "the weights") != null); // the ledger's largest line
    // the pool itself: below the context is the pool's own code; the context is allowed; a larger pool too
    const low = try postJson(&rig, base("validate"), bearer, "{\"set\": {\"tf_ds_window\": 524288}}");
    defer low.free();
    try std.testing.expectEqualStrings("pool_below_window", (try firstRefused(a, low.body)).get("code").?.string);
    const edge = try postJson(&rig, base("validate"), bearer, "{\"set\": {\"tf_ds_window\": 1048576}}");
    defer edge.free();
    try std.testing.expect((try json.parse(a, edge.body)).ok.get("ok").?.bool);
    const big = try postJson(&rig, base("validate"), bearer, "{\"set\": {\"tf_ds_window\": 33554432}}");
    defer big.free();
    try std.testing.expect((try json.parse(a, big.body)).ok.get("ok").?.bool);
    // both named: the start refuses, so the dry run does
    const both = try postJson(&rig, base("validate"), bearer, "{\"set\": {\"tf_ds_window\": 33554432, \"tf_ds_kv_free_target_gib\": 8}}");
    defer both.free();
    const bd = (try json.parse(a, both.body)).ok;
    try std.testing.expect(!bd.get("ok").?.bool);
    try std.testing.expectEqualStrings("pool_named_twice", bd.get("refused").?.array[0].get("code").?.string);
    // `unset` on the free target is the off mode: the pool is the setting's
    const off = try postJson(&rig, base("validate"), bearer, "{\"set\": {\"tf_ds_kv_free_target_gib\": null}}");
    defer off.free();
    const od = (try json.parse(a, off.body)).ok;
    try std.testing.expect(od.get("ok").?.bool);
    try std.testing.expectEqualStrings("1048576", od.get("derived").?.get("kv_pool").?.get("tokens_after").?.int);
    try std.testing.expectEqual(@as(u32, 0), rig.engine.submits);
}

// ---- step 5: set, revert, persistence, the live block

const live = api.live_config;
const admin_state = @import("admin_state.zig");

/// A rig with an engine.cfg and its history in the rig's folder, the live block and the admin state fresh; `restore` puts the globals back.
const Writable = struct {
    saved_report: settings.Report,
    saved_path: []const u8,
    saved_version: u64,
    saved_polls: bool,
    cfg: []const u8,
    hist: []const u8,

    fn open(rig: *Rig, initial: []const u8) !Writable {
        const a = rig.arena.allocator();
        const w: Writable = .{
            .saved_report = settings.report,
            .saved_path = api.settings_history.path,
            .saved_version = api.settings_history.version.load(.acquire),
            .saved_polls = rig.srv.config.log_polls,
            .cfg = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/engine.cfg", .{rig.tmp.sub_path}),
            .hist = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/engine.history.jsonl", .{rig.tmp.sub_path}),
        };
        try rig.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "engine.cfg", .data = initial });
        settings.report = .{ .path = w.cfg };
        api.settings_history.path = w.hist;
        api.settings_history.version.store(0, .release);
        admin_state.reset();
        live.block = .{};
        admin_state.release = "test-release";
        return w;
    }

    fn restore(w: Writable) void {
        settings.report = w.saved_report;
        api.settings_history.path = w.saved_path;
        api.settings_history.version.store(w.saved_version, .release);
        admin_state.reset();
        live.block = .{};
    }

    fn cfgText(w: Writable, a: std.mem.Allocator) ![]const u8 {
        return std.Io.Dir.cwd().readFileAlloc(std.testing.io, w.cfg, a, .limited(1 << 20));
    }

    fn histLines(w: Writable, a: std.mem.Allocator) ![]const []const u8 {
        return (try api.settings_history.load(a, std.testing.io, w.hist)).lines;
    }
};

fn setBody(rig: *Rig, body: []const u8) !Reply {
    return exchange(rig.srv, "POST", base("set"), bearer ++ json_head, body);
}

test "set: a live row is published, the host takes it at its next turn, the file and the history hold it, and the engine thread is untouched" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "# a release's settings\nschema 1\ntf_ds_spill_readers = 8\n");
    defer w.restore();
    const r = try setBody(&rig, "{\"set\": {\"tf_ds_rounds_beside\": 5}, \"wait_ms\": 0, \"reason\": \"night digest\"}");
    defer r.free();
    try std.testing.expectEqual(@as(u16, 200), r.status);
    const doc = (try json.parse(a, r.body)).ok;
    try std.testing.expectEqualStrings("pending", doc.get("outcome").?.string); // the host has not turned yet
    try std.testing.expectEqualStrings("1", doc.get("config_version").?.int);
    try std.testing.expectEqualStrings("0", doc.get("active_version").?.int);
    const ch = doc.get("changes").?.array[0];
    try std.testing.expectEqualStrings("pending_step", ch.get("state").?.string);
    try std.testing.expectEqualStrings("live", ch.get("apply_mode").?.string);
    // the block holds the value; the engine's own turn takes it
    var seen: u64 = 0;
    var vals: [live.slot_count]u64 = undefined;
    try std.testing.expect(live.block.turn(&seen, &vals));
    try std.testing.expectEqual(@as(u64, 5), vals[@backingInt(live.Slot.rounds_beside)]);
    const cfg = try request(rig.srv, "GET", base("config"), bearer_read);
    defer cfg.free();
    const cd = (try json.parse(a, cfg.body)).ok;
    try std.testing.expectEqualStrings("1", cd.get("active_version").?.int);
    for (cd.get("values").?.array) |v| if (std.mem.eql(u8, v.get("name").?.string, "tf_ds_rounds_beside")) {
        try std.testing.expectEqualStrings("5", v.get("value").?.string);
        try std.testing.expectEqualStrings("active", v.get("state").?.string);
        try std.testing.expectEqualStrings("admin", v.get("source").?.string);
        try std.testing.expectEqualStrings("admin", v.get("set_by").?.string);
    };
    // engine.cfg: the line is added, the comment and the other line stay
    const text = try w.cfgText(a);
    try std.testing.expectEqualStrings("# a release's settings\nschema 1\ntf_ds_spill_readers = 8\ntf_ds_rounds_beside = 5\n", text);
    const ls = try w.histLines(a);
    try std.testing.expectEqual(@as(usize, 1), ls.len);
    const hv = try std.json.parseFromSliceLeaky(std.json.Value, a, ls[0], .{});
    try std.testing.expectEqualStrings("applied", hv.object.get("outcome").?.string);
    try std.testing.expectEqualStrings("night digest", hv.object.get("reason").?.string);
    try std.testing.expectEqualStrings("test-release", hv.object.get("release").?.string);
    try std.testing.expect(hv.object.get("cfg_sha256") != null);
    try std.testing.expectEqual(@as(u32, 0), rig.engine.submits);
    try std.testing.expectEqual(@as(u32, 0), rig.engine.cancels);
}

test "set: the same request again is no change, and a request that waits for the host answers applied once the host has turned" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    const Host = struct {
        fn run() void {
            // the host turns every few milliseconds, as its loop does
            var seen: u64 = 0;
            var vals: [live.slot_count]u64 = undefined;
            for (0..400) |_| {
                std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
                if (live.block.turn(&seen, &vals)) return;
            }
        }
    };
    const t = try std.Thread.spawn(.{}, Host.run, .{});
    const first = try setBody(&rig, "{\"set\": {\"tf_ds_rounds_beside\": 3}, \"wait_ms\": 3000}");
    defer first.free();
    t.join();
    const fd = (try json.parse(a, first.body)).ok;
    try std.testing.expectEqualStrings("applied", fd.get("outcome").?.string);
    try std.testing.expectEqualStrings("active", fd.get("changes").?.array[0].get("state").?.string);
    const again = try setBody(&rig, "{\"set\": {\"tf_ds_rounds_beside\": 3}, \"wait_ms\": 0}");
    defer again.free();
    const ad = (try json.parse(a, again.body)).ok;
    try std.testing.expect(ad.get("noop").?.bool);
    try std.testing.expectEqualStrings("1", ad.get("config_version").?.int); // no new version
    try std.testing.expectEqual(@as(usize, 1), (try w.histLines(a)).len); // and no new row
}

test "set: one refused row among five changes nothing; a stale if_version is a 409 that shows the rows; unset takes the line out" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\ntf_ds_spill_readers = 8\n");
    defer w.restore();
    const bad = try setBody(&rig, "{\"set\": {\"tf_ds_rounds_beside\": 5, \"tf_ds_spill_readers\": 6, \"tf_ds_spill_smalls\": 5, \"tf_ds_spill_quota\": 0.5, \"tf_ds_spill_open_ms\": 99999999999}, \"wait_ms\": 0}");
    defer bad.free();
    try std.testing.expectEqual(@as(u16, 422), bad.status);
    try std.testing.expect(std.mem.indexOf(u8, bad.body, "validation_failed") != null);
    try std.testing.expectEqualStrings("schema 1\ntf_ds_spill_readers = 8\n", try w.cfgText(a)); // untouched
    try std.testing.expectEqual(@as(u64, 0), api.settings_history.version.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), live.block.requested_version.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), (try w.histLines(a)).len);
    const ok = try setBody(&rig, "{\"set\": {\"tf_ds_spill_readers\": null}, \"wait_ms\": 0}");
    defer ok.free();
    try std.testing.expectEqual(@as(u16, 200), ok.status);
    try std.testing.expectEqualStrings("schema 1\n", try w.cfgText(a)); // the line left the file
    const stale = try setBody(&rig, "{\"set\": {\"tf_ds_spill_readers\": 9}, \"if_version\": 0, \"wait_ms\": 0}");
    defer stale.free();
    try std.testing.expectEqual(@as(u16, 409), stale.status);
    const sd = (try json.parse(a, stale.body)).ok.get("error").?;
    try std.testing.expectEqualStrings("version_conflict", sd.get("code").?.string);
    try std.testing.expectEqualStrings("1", sd.get("details").?.get("config_version").?.int);
    try std.testing.expectEqual(@as(usize, 1), sd.get("details").?.get("current").?.array.len);
}

test "set: a row without a live hook waits for the next start, the engine's block is untouched, and persist false is for live rows only" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    const r = try setBody(&rig, "{\"set\": {\"tf_ds_draft_ms\": 7}, \"wait_ms\": 0}");
    defer r.free();
    const d = (try json.parse(a, r.body)).ok;
    try std.testing.expectEqualStrings("applied", d.get("outcome").?.string);
    try std.testing.expectEqualStrings("pending_restart", d.get("changes").?.array[0].get("state").?.string);
    try std.testing.expectEqualStrings("next_start", d.get("effective_mode").?.string);
    try std.testing.expectEqual(@as(u64, 0), live.block.requested_version.load(.acquire));
    try std.testing.expectEqualStrings("schema 1\ntf_ds_draft_ms = 7\n", try w.cfgText(a));
    const nofile = try setBody(&rig, "{\"set\": {\"tf_ds_draft_ms\": 9}, \"persist\": false}");
    defer nofile.free();
    try std.testing.expectEqual(@as(u16, 400), nofile.status);
    try std.testing.expect(std.mem.indexOf(u8, nofile.body, "invalid_apply_mode") != null);
    // a live row with persist false: the engine takes it, the file and its replay do not hold it
    const only = try setBody(&rig, "{\"set\": {\"tf_ds_rounds_beside\": 4}, \"persist\": false, \"wait_ms\": 0}");
    defer only.free();
    try std.testing.expectEqual(@as(u16, 200), only.status);
    try std.testing.expectEqualStrings("schema 1\ntf_ds_draft_ms = 7\n", try w.cfgText(a));
    const st = try api.settings_history.state(a, std.testing.io, w.hist);
    try std.testing.expect(st.values.get(@intCast(settings.byEnv("TF_DS_ROUNDS_BESIDE").?.id)) == null);
    try std.testing.expect(st.values.get(@intCast(settings.byEnv("TF_DS_DRAFT_MS").?.id)) != null);
}

test "set: a file that cannot be written rejects the whole request and nothing is applied or recorded" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    settings.report.path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/not-here/engine.cfg", .{rig.tmp.sub_path});
    const r = try setBody(&rig, "{\"set\": {\"tf_ds_rounds_beside\": 5}, \"wait_ms\": 0}");
    defer r.free();
    try std.testing.expectEqual(@as(u16, 500), r.status);
    try std.testing.expectEqual(@as(u64, 0), live.block.requested_version.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), api.settings_history.version.load(.acquire));
    // no engine.cfg at all: persist is refused by name, a live row with persist false still goes
    settings.report.path = "";
    const none = try setBody(&rig, "{\"set\": {\"tf_ds_rounds_beside\": 5}}");
    defer none.free();
    try std.testing.expectEqual(@as(u16, 422), none.status);
    try std.testing.expect(std.mem.indexOf(u8, none.body, "no_engine_cfg") != null);
}

test "set: a reason is printable text of at most 200 bytes, quotes are kept whole in a history line, a key never reaches the history" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    for ([_][]const u8{ "{\"set\": {\"tf_ds_draft_ms\": 7}, \"reason\": \"a\\nb\"}", "{\"set\": {\"tf_ds_draft_ms\": 7}, \"reason\": \"a\\u0001b\"}" }) |body| {
        const r = try setBody(&rig, body);
        defer r.free();
        try std.testing.expectEqual(@as(u16, 400), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "invalid_reason") != null);
    }
    const pad = try a.alloc(u8, 10000);
    @memset(pad, 'x');
    const long = try std.fmt.allocPrint(a, "{{\"set\": {{\"tf_ds_draft_ms\": 7}}, \"reason\": \"{s}\"}}", .{pad});
    const big = try setBody(&rig, long);
    defer big.free();
    try std.testing.expectEqual(@as(u16, 400), big.status);
    const quoted = try setBody(&rig, "{\"set\": {\"tf_ds_draft_ms\": 7}, \"reason\": \"say \\\"hi\\\" \\\\ ok\", \"wait_ms\": 0}");
    defer quoted.free();
    try std.testing.expectEqual(@as(u16, 200), quoted.status);
    const ls = try w.histLines(a);
    try std.testing.expectEqual(@as(usize, 1), ls.len);
    const hv = try std.json.parseFromSliceLeaky(std.json.Value, a, ls[0], .{});
    try std.testing.expectEqualStrings("say \"hi\" \\ ok", hv.object.get("reason").?.string);
    const key = try setBody(&rig, "{\"set\": {\"tensorfold_api_key\": \"sk-canary-key\"}}");
    defer key.free();
    try std.testing.expectEqual(@as(u16, 422), key.status);
    try std.testing.expect(std.mem.indexOf(u8, key.body, "sk-canary-key") == null);
    for (try w.histLines(a)) |l| try std.testing.expect(std.mem.indexOf(u8, l, "sk-canary-key") == null);
    try std.testing.expect(std.mem.indexOf(u8, try w.cfgText(a), "sk-canary-key") == null);
}

test "set: a read key is refused 403, an agent key 403 for rows it may not set, no key 404; a GET of it is 404; admin writes are limited a minute with Retry-After" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    const body = "{\"set\": {\"tf_ds_draft_ms\": 7}, \"wait_ms\": 0}";
    const read = try exchange(rig.srv, "POST", base("set"), bearer_read ++ json_head, body);
    defer read.free();
    try std.testing.expectEqual(@as(u16, 403), read.status);
    const agent = try exchange(rig.srv, "POST", base("set"), bearer_agent ++ json_head, body);
    defer agent.free();
    try std.testing.expectEqual(@as(u16, 403), agent.status);
    try std.testing.expect(std.mem.indexOf(u8, agent.body, "not_permitted_for_agent") != null);
    const none = try exchange(rig.srv, "POST", base("set"), json_head, body);
    defer none.free();
    try std.testing.expectEqual(@as(u16, 404), none.status);
    const get = try request(rig.srv, "GET", base("set"), bearer);
    defer get.free();
    try std.testing.expectEqual(@as(u16, 404), get.status);
    try std.testing.expectEqual(@as(usize, 0), (try w.histLines(a)).len);
    admin_state.reset();
    var limited: ?Reply = null;
    defer if (limited) |l| l.free();
    for (0..31) |i| {
        const r = try setBody(&rig, if (i % 2 == 0) "{\"set\": {\"tf_ds_draft_ms\": 7}, \"wait_ms\": 0}" else "{\"set\": {\"tf_ds_draft_ms\": 8}, \"wait_ms\": 0}");
        if (i < 30) {
            defer r.free();
            try std.testing.expectEqual(@as(u16, 200), r.status);
        } else limited = r;
    }
    try std.testing.expectEqual(@as(u16, 429), limited.?.status);
    try std.testing.expect(limited.?.hasHeader("Retry-After: 60") or std.mem.indexOf(u8, limited.?.head, "Retry-After: ") != null);
    const ld = (try json.parse(a, limited.?.body)).ok.get("error").?;
    try std.testing.expectEqualStrings("rate_limited", ld.get("code").?.string);
    try std.testing.expect(ld.get("details").?.get("retry_after_s") != null);
}

test "set: a count of rounds beside a share below 1 is refused, and a log-of-polls row is taken at once on its own thread" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    const share = settings.indexOf(settings.byEnv("TENSORFOLD_PREFILL_SHARE").?);
    const saved = settings.resolved[share];
    defer settings.resolved[share] = saved;
    settings.resolved[share] = .{ .source = .cfg, .value = "0.5" };
    const r = try setBody(&rig, "{\"set\": {\"tf_ds_rounds_beside\": 5}, \"wait_ms\": 0}");
    defer r.free();
    try std.testing.expectEqual(@as(u16, 422), r.status);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "conflicting_options") != null);
    settings.resolved[share] = saved;
    try std.testing.expect(!rig.srv.config.log_polls);
    const p = try setBody(&rig, "{\"set\": {\"tensorfold_log_polls\": 1}, \"wait_ms\": 0}");
    defer p.free();
    const pd = (try json.parse(a, p.body)).ok;
    try std.testing.expectEqualStrings("applied", pd.get("outcome").?.string);
    try std.testing.expectEqualStrings("active", pd.get("changes").?.array[0].get("state").?.string);
    try std.testing.expect(rig.srv.config.log_polls);
    rig.srv.config.log_polls = w.saved_polls;
}

test "revert: the plan that undoes a history row or returns to a version applies nothing; set with reverts_version records it" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    for ([_][]const u8{ "{\"set\": {\"tf_ds_draft_ms\": 7}, \"wait_ms\": 0}", "{\"set\": {\"tf_ds_draft_ms\": 9, \"tf_ds_spill_readers\": 3}, \"wait_ms\": 0}" }) |b| {
        const r = try setBody(&rig, b);
        r.free();
    }
    const undo = try exchange(rig.srv, "POST", base("revert"), bearer ++ json_head, "{\"history_version\": 2}");
    defer undo.free();
    try std.testing.expectEqual(@as(u16, 200), undo.status);
    const ud = (try json.parse(a, undo.body)).ok;
    const set = ud.get("plan").?.get("set").?;
    try std.testing.expectEqualStrings("7", set.get("tf_ds_draft_ms").?.string); // back to what it was
    try std.testing.expect(set.get("tf_ds_spill_readers").? == .null); // it had no line: unset
    try std.testing.expectEqualStrings("2", ud.get("plan").?.get("reverts_version").?.int);
    const back = try exchange(rig.srv, "POST", base("revert"), bearer ++ json_head, "{\"to_version\": 1}");
    defer back.free();
    const bd = (try json.parse(a, back.body)).ok.get("plan").?.get("set").?;
    try std.testing.expectEqualStrings("7", bd.get("tf_ds_draft_ms").?.string);
    try std.testing.expect(bd.get("tf_ds_spill_readers").? == .null);
    // nothing was applied by asking
    try std.testing.expectEqualStrings("schema 1\ntf_ds_draft_ms = 9\ntf_ds_spill_readers = 3\n", try w.cfgText(a));
    try std.testing.expectEqual(@as(usize, 2), (try w.histLines(a)).len);
    const apply = try setBody(&rig, "{\"set\": {\"tf_ds_draft_ms\": 7, \"tf_ds_spill_readers\": null}, \"reverts_version\": 2, \"wait_ms\": 0}");
    defer apply.free();
    try std.testing.expectEqual(@as(u16, 200), apply.status);
    const ls = try w.histLines(a);
    const hv = try std.json.parseFromSliceLeaky(std.json.Value, a, ls[2], .{});
    try std.testing.expectEqualStrings("revert", hv.object.get("endpoint").?.string);
    try std.testing.expectEqual(@as(i64, 2), hv.object.get("reverts_version").?.integer);
    // a bad request
    const both = try exchange(rig.srv, "POST", base("revert"), bearer ++ json_head, "{\"to_version\": 1, \"history_version\": 2}");
    defer both.free();
    try std.testing.expectEqual(@as(u16, 400), both.status);
    const missing = try exchange(rig.srv, "POST", base("revert"), bearer ++ json_head, "{\"history_version\": 99}");
    defer missing.free();
    try std.testing.expectEqual(@as(u16, 404), missing.status);
    const agent = try exchange(rig.srv, "POST", base("revert"), bearer_agent ++ json_head, "{\"history_version\": 1}");
    defer agent.free();
    try std.testing.expectEqual(@as(u16, 403), agent.status);
}

test "set: keep-ahead (class A) is live only where the engine can take it, then it is published for the host to send to every rank" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    defer live.able.store(0, .release);
    // an engine that did not verify it at its start: the row is a setting for the next start; a live-only change is refused by name
    live.able.store(0, .release);
    const no = try setBody(&rig, "{\"set\": {\"tf_ds_engram_keep_ahead\": 0}, \"persist\": false, \"wait_ms\": 0}");
    defer no.free();
    try std.testing.expectEqual(@as(u16, 400), no.status);
    try std.testing.expect(std.mem.indexOf(u8, no.body, "invalid_apply_mode") != null);
    var seen: u64 = 0;
    var vals: [live.slot_count]u64 = undefined;
    try std.testing.expect(!live.block.turn(&seen, &vals)); // nothing was published
    // one that did: live, one publication in the class A slot, the host's turn takes it (and sends the frame: core's test)
    live.setAble(.keep_ahead, true);
    const yes = try setBody(&rig, "{\"set\": {\"tf_ds_engram_keep_ahead\": 0}, \"persist\": false, \"wait_ms\": 0}");
    defer yes.free();
    try std.testing.expectEqual(@as(u16, 200), yes.status);
    const doc = (try json.parse(a, yes.body)).ok;
    const ch = doc.get("changes").?.array[0];
    try std.testing.expectEqualStrings("live", ch.get("apply_mode").?.string);
    try std.testing.expectEqualStrings("A", ch.get("apply_class").?.string);
    try std.testing.expect(live.block.turn(&seen, &vals));
    try std.testing.expectEqual(@as(u64, 0), vals[@backingInt(live.Slot.keep_ahead)]); // 0: a change from the default-on 1 (speed_defaults, T6)
    // the engine's host has taken the version: the row is active (by construction: rank 0 handed the frame to the link)
    const cfg = try request(rig.srv, "GET", base("config"), bearer_read);
    defer cfg.free();
    const cd = (try json.parse(a, cfg.body)).ok;
    var found = false;
    for (cd.get("values").?.array) |v| if (std.mem.eql(u8, v.get("name").?.string, "tf_ds_engram_keep_ahead")) {
        found = true;
        try std.testing.expectEqualStrings("0", v.get("value").?.string); // the value set above (0, a change from the default-on 1)
        try std.testing.expectEqualStrings("active", v.get("state").?.string);
    };
    try std.testing.expect(found);
    // unset returns it to the start's: the slot carries the marker
    const un = try setBody(&rig, "{\"set\": {\"tf_ds_engram_keep_ahead\": null}, \"persist\": false, \"wait_ms\": 0}");
    defer un.free();
    try std.testing.expectEqual(@as(u16, 200), un.status);
    try std.testing.expect(live.block.turn(&seen, &vals));
    try std.testing.expectEqual(live.unset, vals[@backingInt(live.Slot.keep_ahead)]);
}
test "set: the KV free target (class A) is live only with a virtual-memory pool, travels as whole MiB, and null returns to the start's" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    defer live.able.store(0, .release);
    live.able.store(0, .release);
    const no = try setBody(&rig, "{\"set\": {\"tf_ds_kv_free_target_gib\": 10}, \"persist\": false, \"wait_ms\": 0}");
    defer no.free();
    try std.testing.expectEqual(@as(u16, 400), no.status); // no TF_DS_KV_VMM: a setting for the next start
    live.setAble(.kv_free_target, true);
    var seen: u64 = 0;
    var vals: [live.slot_count]u64 = undefined;
    try std.testing.expect(!live.block.turn(&seen, &vals));
    const ok = try setBody(&rig, "{\"set\": {\"tf_ds_kv_free_target_gib\": 10.5}, \"persist\": false, \"wait_ms\": 0}");
    defer ok.free();
    try std.testing.expectEqual(@as(u16, 200), ok.status);
    const ch = (try json.parse(a, ok.body)).ok.get("changes").?.array[0];
    try std.testing.expectEqualStrings("live", ch.get("apply_mode").?.string);
    try std.testing.expect(live.block.turn(&seen, &vals));
    try std.testing.expectEqual(@as(u64, 10752), vals[@backingInt(live.Slot.kv_free_target)]); // 10.5 GiB as MiB
    const un = try setBody(&rig, "{\"set\": {\"tf_ds_kv_free_target_gib\": null}, \"persist\": false, \"wait_ms\": 0}");
    defer un.free();
    try std.testing.expectEqual(@as(u16, 200), un.status);
    try std.testing.expect(live.block.turn(&seen, &vals));
    try std.testing.expectEqual(live.unset, vals[@backingInt(live.Slot.kv_free_target)]);
}
