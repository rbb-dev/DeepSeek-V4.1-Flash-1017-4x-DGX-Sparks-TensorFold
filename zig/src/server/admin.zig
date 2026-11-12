//! The admin surface: `/admin/api/v1/*` and the page at `/admin/ui`: the schema, the active values, the status, the ledger, the
//! requests, the series, the logs, the keys, the history; a dry run, `validate`; and the writes, `set` and `revert`. A write names
//! rows of class H or R: a row with a live hook (live_config.zig) is handed to the engine, which takes it at its next loop turn;
//! every row is kept in engine.cfg unless the caller says not, and the history row is on disk before the answer.
//!
//! It is the same port and the same kind of gate as /v1: a request needs a key. The keys are the admin keys' own file
//! (TENSORFOLD_ADMIN_KEY_FILE, read by `auth.Store` exactly as the API key file is, under two more rules: no key shorter
//! than 32 bytes and none that the /v1 file also holds); a key of the shared key file opens nothing here, and there is
//! no rule about source addresses (the network around the engine is the operator's). The key's label is its scope:
//! `admin` or `admin-*` may do everything, `agent-*` and `read-*` and any other label may read, and `/keys` and `/logs`
//! are the admin's alone. Every refusal of a caller that has no valid key is the 404 any unknown path gets: with no admin
//! key file, with no key, with a wrong key, with a method the route does not have, or with a path this surface does not
//! have, so nothing tells a caller that the API exists. A valid key that lacks the scope is told so (403). The page's
//! three static files are the one exception: with an admin key file configured they answer without a key, at exactly four
//! paths and only to GET, before the gate. The handlers run on the connection's own thread and read only what /health
//! reads already (the engine's status call, the metrics, the start-up record of `settings`) and the server's own records
//! of its requests, its log lines and its samples: none reaches the engine thread.
const std = @import("std");
const api = @import("engine_api");
const settings = api.settings;
const settings_history = api.settings_history;
const kv_pool = api.kv_pool;
const live_config = api.live_config;
const fill_order = api.fill_order;
const goals_mod = api.goals;
const cfg_write = api.cfg_write;
const admin_state = @import("admin_state.zig");
const json = @import("json");
const http_conn = @import("http_conn.zig");
const http_body = @import("http_body.zig");
const auth = @import("auth.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const routes = @import("routes.zig");
const status_routes = @import("status_routes.zig");
const prepare_gate = @import("prepare_gate.zig");
const admin_series = @import("admin_series.zig");
const admin_goals = @import("admin_goals.zig");
const Server = @import("server.zig").Server;
const Conn = http_conn.Conn;
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const prefix = "/admin/api/v1";

/// The API's own version, in every answer's `X-TF-Admin-API` and in /status.
pub const api_major = 1;
pub const api_minor = 0;

/// The page's three files, in the binary.
pub const page_html = @embedFile("admin_ui/index.html");
pub const page_js = @embedFile("admin_ui/app.js");
pub const page_css = @embedFile("admin_ui/app.css");

/// What every page file carries: no inline script or style may run, nothing may frame the page, and nothing leaves the
/// origin (the key lives in the page's memory only).
pub const page_csp = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

const Method = enum { get, post };
const Need = enum { read, agent, admin };
const Route = struct { name: []const u8, method: Method, need: Need, step: u8 };

/// Every route that answers. A route that is not here is the unknown-path 404 for everyone, whatever its method; a POST
/// arrives route by route as it is built.
pub const table = [_]Route{
    .{ .name = "routes", .method = .get, .need = .read, .step = 2 },
    .{ .name = "schema", .method = .get, .need = .read, .step = 2 },
    .{ .name = "config", .method = .get, .need = .read, .step = 2 },
    .{ .name = "status", .method = .get, .need = .read, .step = 2 },
    .{ .name = "ledger", .method = .get, .need = .read, .step = 4 },
    .{ .name = "requests", .method = .get, .need = .read, .step = 2 },
    .{ .name = "series", .method = .get, .need = .read, .step = 2 },
    .{ .name = "goals", .method = .get, .need = .read, .step = 6 },
    .{ .name = "logs", .method = .get, .need = .admin, .step = 2 },
    .{ .name = "keys", .method = .get, .need = .admin, .step = 2 },
    .{ .name = "history", .method = .get, .need = .read, .step = 3 },
    .{ .name = "leases", .method = .get, .need = .read, .step = 5 },
    .{ .name = "policy", .method = .get, .need = .read, .step = 5 },
    .{ .name = "keys/reload", .method = .post, .need = .admin, .step = 2 },
    .{ .name = "validate", .method = .post, .need = .agent, .step = 3 },
    .{ .name = "set", .method = .post, .need = .agent, .step = 5 },
    .{ .name = "revert", .method = .post, .need = .admin, .step = 5 },
};

/// Whether the request is for the admin surface (every path under /admin/), whatever its method.
pub fn owns(raw_path: []const u8) bool {
    const path = auth.routePath(raw_path);
    return std.mem.eql(u8, path, "/admin") or std.mem.startsWith(u8, path, "/admin/");
}

const Page = enum { html, js, css, to_index };

/// The page file a raw request path names, or null. The match is exact on the raw path: no query, no dot segments. The index
/// is /admin/ui/ only; /admin, /admin/ and /admin/ui redirect to it.
fn pageFile(raw_path: []const u8) ?Page {
    if (std.mem.eql(u8, raw_path, "/admin/ui/")) return .html;
    // an address without the page's trailing slash: the page's relative app.js / app.css would resolve under /admin/, so it is
    // redirected to /admin/ui/ (without it, /admin/ui loaded a page whose relative app.js and app.css did not resolve)
    if (std.mem.eql(u8, raw_path, "/admin") or std.mem.eql(u8, raw_path, "/admin/") or std.mem.eql(u8, raw_path, "/admin/ui")) return .to_index;
    if (std.mem.eql(u8, raw_path, "/admin/ui/app.js")) return .js;
    if (std.mem.eql(u8, raw_path, "/admin/ui/app.css")) return .css;
    return null;
}

/// Whether a key of `scope` may call a route that needs `need`: the admin may call all, an agent the agent's and the read
/// ones, a reader the read ones.
fn allows(need: Need, scope: auth.Scope) bool {
    return switch (need) {
        .read => true,
        .agent => scope == .admin or scope == .agent,
        .admin => scope == .admin,
    };
}

const Who = struct { label: []const u8, scope: auth.Scope };
const Ctx = struct { srv: *Server, conn: *Conn, a: Allocator, who: Who };

/// Answers an admin request, or the unknown-path 404. The caller has not read a body: a refused request's body is dropped.
pub fn handle(srv: *Server, conn: *Conn, a: Allocator, discard: *const fn (*Conn, Allocator) void) void {
    // the page's files, before the gate, only while an admin key file is configured
    if (std.mem.eql(u8, conn.method, "GET") and srv.admin_keys != null) if (pageFile(conn.path)) |f| {
        discard(conn, a);
        return sendPage(conn, a, f);
    };
    const who = identify(srv, conn, a) orelse {
        discard(conn, a);
        return routes.unknown(conn, a);
    };
    conn.log_label = who.label; // the access line says which admin key asked
    const route = auth.routePath(conn.path);
    const want: Method = if (std.mem.eql(u8, conn.method, "GET")) .get else if (std.mem.eql(u8, conn.method, "POST")) .post else {
        discard(conn, a);
        return routes.unknown(conn, a);
    };
    const r = if (std.mem.startsWith(u8, route, prefix ++ "/")) for (&table) |*t| {
        if (t.method == want and std.mem.eql(u8, t.name, route[prefix.len + 1 ..])) break t;
    } else null else null;
    const found = r orelse {
        discard(conn, a);
        return routes.unknown(conn, a);
    };
    if (!allows(found.need, who.scope)) {
        discard(conn, a);
        return fail(conn, a, 403, "scope_denied", "this key may not call this route");
    }
    const ctx: Ctx = .{ .srv = srv, .conn = conn, .a = a, .who = who };
    if (found.method == .get) discard(conn, a);
    const name = found.name;
    if (std.mem.eql(u8, name, "routes")) return reply(ctx, routesValue(a, who.scope));
    if (std.mem.eql(u8, name, "schema")) return reply(ctx, schema(a));
    if (std.mem.eql(u8, name, "config")) return reply(ctx, config(a));
    if (std.mem.eql(u8, name, "status")) return reply(ctx, statusValue(srv, a, who));
    if (std.mem.eql(u8, name, "ledger")) return reply(ctx, ledger(a));
    if (std.mem.eql(u8, name, "requests")) return reply(ctx, requestsValue(srv, a));
    if (std.mem.eql(u8, name, "series")) return seriesReply(ctx);
    if (std.mem.eql(u8, name, "goals")) return reply(ctx, goalsValue(srv, a));
    if (std.mem.eql(u8, name, "logs")) return logsReply(ctx);
    if (std.mem.eql(u8, name, "keys")) return reply(ctx, keysValue(srv, a));
    if (std.mem.eql(u8, name, "history")) return historyReply(ctx);
    if (std.mem.eql(u8, name, "leases")) return reply(ctx, emptyView(a, "leases", "no lease exists in this release"));
    if (std.mem.eql(u8, name, "policy")) return reply(ctx, policyValue(srv, a));
    if (std.mem.eql(u8, name, "keys/reload")) return keysReload(ctx);
    if (std.mem.eql(u8, name, "validate")) return validateReply(ctx);
    if (std.mem.eql(u8, name, "set")) return setReply(ctx);
    if (std.mem.eql(u8, name, "revert")) return revertReply(ctx);
    routes.unknown(conn, a);
}

/// The request carries an admin key. Exactly one Authorization header and one x-api-key, as /v1's gate reads them.
fn identify(srv: *Server, conn: *Conn, a: Allocator) ?Who {
    const keys = srv.admin_keys orelse return null;
    if (!keys.enabled()) return null;
    const label = keys.match(a, single(conn, "Authorization"), single(conn, "x-api-key")) orelse return null;
    const own = a.dupe(u8, label) catch return null; // the store may reread its file under another request
    return .{ .label = own, .scope = auth.scopeOf(own).scope };
}

fn single(conn: *const Conn, name: []const u8) ?[]const u8 {
    if (conn.headerCount(name) != 1) return null;
    return conn.header(name);
}

fn phrase(code: u16) []const u8 {
    return switch (code) {
        301 => "Moved Permanently",
        403 => "Forbidden",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        else => http_conn.reason(code),
    };
}

/// An answer under the admin headers: never cached, never sniffed. A page file also carries its policy headers and an
/// ETag (`page_tag`), and an API answer the API's version.
fn send(conn: *Conn, code: u16, content_type: []const u8, body: []const u8, page_tag: ?[]const u8) void {
    conn.startResponse(code, phrase(code)) catch return;
    conn.addHeader("Content-Type", content_type) catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    conn.addHeader("Cache-Control", if (page_tag != null) "no-cache" else "no-store") catch return;
    conn.addHeader("X-Content-Type-Options", "nosniff") catch return;
    if (page_tag) |tag| {
        conn.addHeader("ETag", tag) catch return;
        conn.addHeader("Content-Security-Policy", page_csp) catch return;
        conn.addHeader("Referrer-Policy", "no-referrer") catch return;
        conn.addHeader("X-Frame-Options", "DENY") catch return;
    } else conn.addHeader("X-TF-Admin-API", std.fmt.comptimePrint("{d}.{d}", .{ api_major, api_minor })) catch return;
    if (conn.close) conn.addHeader("Connection", "close") catch return;
    conn.finish(body) catch {};
}

fn sendPage(conn: *Conn, a: Allocator, f: Page) void {
    if (f == .to_index) {
        // relative, so a proxy's path prefix is kept: from /admin "admin/ui/", from /admin/ and /admin/ui "ui/"
        const location = if (std.mem.eql(u8, conn.path, "/admin")) "admin/ui/" else "ui/";
        conn.startResponse(301, phrase(301)) catch return;
        conn.addHeader("Location", location) catch return;
        conn.addHeader("Content-Length", "0") catch return;
        conn.addHeader("Cache-Control", "no-cache") catch return;
        if (conn.close) conn.addHeader("Connection", "close") catch return;
        conn.finish("") catch {};
        return;
    }
    const body = switch (f) {
        .html => page_html,
        .js => page_js,
        .css => page_css,
        .to_index => unreachable,
    };
    const kind = switch (f) {
        .html => "text/html;charset=utf-8",
        .js => "text/javascript;charset=utf-8",
        .css => "text/css;charset=utf-8",
        .to_index => unreachable,
    };
    // the tag changes with the bytes, so a new release's page is fetched at once
    const tag = std.fmt.allocPrint(a, "\"{x}\"", .{std.hash.Wyhash.hash(0, body)}) catch return;
    send(conn, 200, kind, body, tag);
}

fn sendValue(ctx: Ctx, code: u16, v: Value) void {
    // an answered read is a poll, as /health's is: its line prints only when polls are logged
    if (code == 200 and std.mem.eql(u8, ctx.conn.method, "GET")) ctx.conn.quiet_log = !ctx.srv.config.log_polls;
    const body = json.stringify(ctx.a, v, .{}) catch return fail(ctx.conn, ctx.a, 500, "internal", "admin: out of memory");
    send(ctx.conn, code, "application/json", body, null);
}

fn reply(ctx: Ctx, result: anyerror!Value) void {
    sendValue(ctx, 200, result catch return fail(ctx.conn, ctx.a, 500, "internal", "admin: out of memory"));
}

/// `{"error": {"code": ..., "message": ...}}` under the admin headers.
fn fail(conn: *Conn, a: Allocator, code: u16, name: []const u8, message: []const u8) void {
    const body = std.fmt.allocPrint(a, "{{\"error\": {{\"code\": {s}, \"message\": {s}}}}}", .{
        json.quote(a, name, .{}) catch return,
        json.quote(a, message, .{}) catch return,
    }) catch return;
    send(conn, code, "application/json", body, null);
}

fn str(s: []const u8) Value {
    return .{ .string = s };
}

fn optNum(a: Allocator, v: ?f64) !Value {
    const x = v orelse return .null;
    if (x == @floor(x) and @abs(x) < 1e15) return json.intValue(a, @as(i64, @intFromFloat(x)));
    return .{ .float = x };
}

/// GET /routes: what this key may call, so the page needs no list of its own.
pub fn routesValue(a: Allocator, scope: auth.Scope) !Value {
    const root = try json.newObject(a);
    const ver = try json.newObject(a);
    try ver.put(a, "major", try json.intValue(a, api_major));
    try ver.put(a, "minor", try json.intValue(a, api_minor));
    try root.put(a, "api", .{ .object = ver });
    var rows: std.ArrayList(Value) = .empty;
    for (&table) |*t| {
        if (!allows(t.need, scope)) continue;
        const o = try json.newObject(a);
        try o.put(a, "method", str(if (t.method == .get) "GET" else "POST"));
        try o.put(a, "path", str(try std.fmt.allocPrint(a, "{s}/{s}", .{ prefix, t.name })));
        try o.put(a, "scope", str(@tagName(t.need)));
        try o.put(a, "step", try json.intValue(a, t.step));
        try rows.append(a, .{ .object = o });
    }
    try root.put(a, "routes", .{ .array = rows.items });
    return .{ .object = root };
}

/// The options a caller may see: every row that is not hidden.
fn visibleCount() usize {
    var n: usize = 0;
    for (&settings.options) |*o| n += @intFromBool(!o.hidden);
    return n;
}

/// GET /schema: every option the engine reads that the surface shows (a hidden row is not here), and the retired names.
pub fn schema(a: Allocator) !Value {
    const root = try json.newObject(a);
    try root.put(a, "schema_version", try json.intValue(a, settings.schema_version));
    try root.put(a, "count", try json.intValue(a, visibleCount()));
    // each class says how long its apply takes: no probe has measured one, so the value is null and the label says so
    const apply = try json.newObject(a);
    inline for (.{
        .{ "P", "per request", false },
        .{ "H", "live: rank 0's host at its next loop turn", true },
        .{ "A", "live on all four ranks at one step", true },
        .{ "I", "when the engine is idle", true },
        .{ "N", "now, after closing admission and draining", true },
        .{ "R", "at a restart (engine.cfg)", false },
    }) |p| {
        const c = try json.newObject(a);
        try c.put(a, "text", str(p[1]));
        const t = try json.newObject(a);
        try t.put(a, "value_s", .null);
        try t.put(a, "label", str("unmeasured"));
        try t.put(a, "source", str("step-1 probe"));
        try c.put(a, "apply_time", .{ .object = t });
        try c.put(a, "now_ok", .{ .bool = p[2] });
        try apply.put(a, p[0], .{ .object = c });
    }
    try root.put(a, "apply_classes", .{ .object = apply });
    const tab_rows = try a.alloc(Value, settings.tabs.len);
    for (&settings.tabs, tab_rows) |*t, *slot| {
        const o = try json.newObject(a);
        try o.put(a, "name", str(t.name));
        try o.put(a, "title", str(t.title));
        try o.put(a, "about", str(t.about));
        slot.* = .{ .object = o };
    }
    try root.put(a, "tabs", .{ .array = tab_rows });
    var rows: std.ArrayList(Value) = .empty;
    for (&settings.options) |*o| {
        if (o.hidden) continue;
        const r = try json.newObject(a);
        try r.put(a, "id", try json.intValue(a, o.id));
        try r.put(a, "name", str(o.name));
        try r.put(a, "env", str(o.env));
        try r.put(a, "group", str(o.group));
        try r.put(a, "tab", str(o.tab));
        try r.put(a, "kind", str(o.kindName()));
        try r.put(a, "unit", str(o.unit));
        try r.put(a, "min", try optNum(a, o.min));
        try r.put(a, "max", try optNum(a, o.max));
        try r.put(a, "default", if (o.default.len == 0) .null else str(o.default));
        try r.put(a, "default_text", str(o.default_text));
        try r.put(a, "default_checked", .{ .bool = o.check == .tested });
        try r.put(a, "apply", str(@tagName(o.apply)));
        try r.put(a, "secret", .{ .bool = o.secret });
        const who = try a.alloc(Value, o.who.len);
        for (o.who, who) |w, *s| s.* = str(@tagName(w));
        try r.put(a, "who", .{ .array = who });
        const values = try a.alloc(Value, o.values.len);
        for (o.values, values) |v, *s| s.* = str(v);
        try r.put(a, "values", .{ .array = values });
        try r.put(a, "label", str(o.label));
        try r.put(a, "doc", str(o.doc));
        try r.put(a, "env_override", .{ .bool = o.env_override });
        try r.put(a, "out_changes", .{ .bool = o.out_changes });
        try r.put(a, "now_ok", .{ .bool = o.now_ok });
        // whether the engine takes a change to this row while it runs (a hook in rank 0's host or on a server thread)
        try r.put(a, "live", .{ .bool = live_config.isWired(o.id) });
        if (o.agent) |ag| {
            const g = try json.newObject(a);
            try g.put(a, "min", try optNum(a, ag.min));
            try g.put(a, "max", try optNum(a, ag.max));
            try g.put(a, "max_ttl_s", try json.intValue(a, ag.max_ttl_s));
            try r.put(a, "agent", .{ .object = g });
        } else try r.put(a, "agent", .null);
        const req = try a.alloc(Value, o.requires.len);
        for (o.requires, req) |c, *slot| slot.* = str(c);
        try r.put(a, "requires", .{ .array = req });
        const costs = try a.alloc(Value, o.cost.len);
        for (o.cost, costs) |c, *slot| {
            const k = try json.newObject(a);
            try k.put(a, "what", str(c.what));
            try k.put(a, "value", str(c.value));
            try k.put(a, "label", str(c.label));
            try k.put(a, "source", str(c.source));
            slot.* = .{ .object = k };
        }
        try r.put(a, "cost", .{ .array = costs });
        try rows.append(a, .{ .object = r });
    }
    try root.put(a, "options", .{ .array = rows.items });
    const gone = try a.alloc(Value, settings.retired.len);
    for (&settings.retired, gone) |*t, *slot| {
        const o = try json.newObject(a);
        try o.put(a, "env", str(t.env));
        try o.put(a, "when", str(@tagName(t.when)));
        try o.put(a, "why", str(t.why));
        slot.* = .{ .object = o };
    }
    try root.put(a, "retired", .{ .array = gone });
    return .{ .object = root };
}

/// GET /config: what start-up found for each option (its source and value), and the engine.cfg that was read.
/// "Active" is the start-up view: this release changes nothing after the start.
pub fn config(a: Allocator) !Value {
    const root = try json.newObject(a);
    try root.put(a, "schema_version", try json.intValue(a, settings.schema_version));
    try root.put(a, "active_means", str("the value the engine runs: what it read at its start (engine.cfg where it names the option, else the environment) unless a live change since has been taken: then the value it runs now"));
    // the newest row of engine.history.jsonl: the requested and the active state are one while nothing applies at run time
    const version_now = settings_history.version.load(.acquire);
    try root.put(a, "config_version", try json.intValue(a, version_now));
    try root.put(a, "active_version", try json.intValue(a, admin_state.activeVersion(version_now)));
    // false: no start-up record was taken in this process (a test server), so every source below is "default"
    try root.put(a, "bootstrapped", .{ .bool = settings.ready.load(.acquire) });
    const rep = settings.report;
    if (rep.path.len > 0) {
        const cfg = try json.newObject(a);
        try cfg.put(a, "path", str(rep.path));
        try cfg.put(a, "sha256", str(try std.fmt.allocPrint(a, "{x}", .{rep.sha256[0..]})));
        try cfg.put(a, "bytes", try json.intValue(a, rep.bytes));
        try cfg.put(a, "declared_schema", try json.intValue(a, rep.declared));
        try cfg.put(a, "applied", try json.intValue(a, rep.applied));
        try cfg.put(a, "cfg_won", try json.intValue(a, rep.cfg_won));
        try cfg.put(a, "environment_won", try json.intValue(a, rep.env_won));
        try cfg.put(a, "unset", try json.intValue(a, rep.unset));
        try cfg.put(a, "unknown", try json.intValue(a, rep.unknown));
        try root.put(a, "engine_cfg", .{ .object = cfg });
    } else try root.put(a, "engine_cfg", .null);
    var rows: std.ArrayList(Value) = .empty;
    for (&settings.options, 0..) |*o, i| {
        if (o.hidden) continue;
        const r = try json.newObject(a);
        const res = settings.resolved[i];
        try r.put(a, "id", try json.intValue(a, o.id));
        try r.put(a, "name", str(o.name));
        try r.put(a, "env", str(o.env));
        // a live change the engine has taken is the value it runs; otherwise the start's
        const live = admin_state.liveValue(o);
        try r.put(a, "source", str(if (live != null) "admin" else @tagName(res.source)));
        // a key is never shown: only that one is set
        if (live) |t| {
            try r.put(a, "value", if (std.mem.eql(u8, t, "unset")) .null else str(t));
        } else try r.put(a, "value", if (res.source == .default) .null else if (o.secret) str("(set)") else str(res.value));
        try r.put(a, "default", if (o.default.len == 0) .null else str(o.default));
        try r.put(a, "default_text", str(o.default_text));
        // what the environment and engine.cfg both named, and the one that did not apply
        if (res.other_source != .default) {
            const other = try json.newObject(a);
            try other.put(a, "source", str(@tagName(res.other_source)));
            try other.put(a, "value", if (o.secret) str("(set)") else str(res.other));
            try r.put(a, "other", .{ .object = other });
        } else try r.put(a, "other", .null);
        const st = admin_state.stateOf(o, currentText(o));
        const mine = admin_state.get(o);
        try r.put(a, "requested", if (st.requested) |t| str(t) else .null);
        try r.put(a, "state", str(@tagName(st.state)));
        try r.put(a, "set_by", if (mine.set_by.len > 0) str(mine.set_by) else .null);
        try r.put(a, "since_version", if (mine.since > 0) try json.intValue(a, mine.since) else .null);
        try rows.append(a, .{ .object = r });
    }
    try root.put(a, "values", .{ .array = rows.items });
    return .{ .object = root };
}

/// GET /status: the engine's health record (what /health answers a key holder), the settings' summary, the API's version,
/// the time, this caller's powers and the server's own view of whether it is idle.
pub fn statusValue(srv: *Server, a: Allocator, who: Who) !Value {
    const scope = who.scope;
    const root = try json.newObject(a);
    try root.put(a, "admin_api", str("v1"));
    const ver = try json.newObject(a);
    try ver.put(a, "major", try json.intValue(a, api_major));
    try ver.put(a, "minor", try json.intValue(a, api_minor));
    try root.put(a, "api", .{ .object = ver });
    const now_ms: i64 = @intCast(@divTrunc(std.Io.Clock.real.now(srv.io).toNanoseconds(), std.time.ns_per_ms));
    try root.put(a, "at_ms", try json.intValue(a, now_ms));
    try root.put(a, "at", str(try rfc3339(a, now_ms)));
    // what this key may do in this build: an admin key may write the rows of class H and A (live where the engine has a hook, A through a cfg frame) and R (engine.cfg)
    const writes = scope == .admin;
    try root.put(a, "read_only", .{ .bool = !writes });
    const caps = try json.newObject(a);
    try caps.put(a, "scope", str(@tagName(scope)));
    // the label of the key that asked (a label, never a key), so the page can say who is signed in
    try caps.put(a, "label", str(who.label));
    const classes = try a.alloc(Value, if (writes) 3 else 0);
    const modes = try a.alloc(Value, if (writes) 2 else 0);
    if (writes) {
        classes[0] = str("H");
        classes[1] = str("A");
        classes[2] = str("R");
        modes[0] = str("live");
        modes[1] = str("next_start");
    }
    try caps.put(a, "writable_classes", .{ .array = classes });
    try caps.put(a, "modes", .{ .array = modes });
    try root.put(a, "capabilities", .{ .object = caps });
    const ours = srv.requests.idle(srv.io, clock.nowNs(srv.io));
    const preparing = srv.preparing.load(.acquire) > 0;
    const idle = try json.newObject(a);
    try idle.put(a, "idle", .{ .bool = ours.idle and !preparing });
    try idle.put(a, "idle_for_s", if (ours.for_s) |x| (if (preparing) Value.null else Value{ .float = x }) else .null);
    // the server's own view: a hint, not the engine's idle condition
    try idle.put(a, "basis", str("server"));
    try root.put(a, "idle", .{ .object = idle });
    try root.put(a, "health", try status_routes.healthValue(srv, a, false));
    {
        const gcfg = admin_goals.config();
        const gr: api.goals.Report = if (srv.series) |sm| sm.goalsReport(srv.io, gcfg, admin_goals.nNow()) else .{ .state = if (gcfg.mode == .off) .off else .idle };
        try root.put(a, "goals", try admin_goals.statusBlock(a, gcfg, gr));
    }
    const version_now = settings_history.version.load(.acquire);
    const version = try json.intValue(a, version_now);
    const cv = try json.newObject(a);
    try cv.put(a, "config_version", version);
    try cv.put(a, "active_version", try json.intValue(a, admin_state.activeVersion(version_now)));
    try cv.put(a, "schema_version", try json.intValue(a, settings.schema_version));
    // rank 0 only: the other ranks report no version until a change can reach them
    const rank0 = try json.newObject(a);
    try rank0.put(a, "rank", try json.intValue(a, 0));
    try rank0.put(a, "version", version);
    const ranks = try a.alloc(Value, 1);
    ranks[0] = .{ .object = rank0 };
    try cv.put(a, "ranks", .{ .array = ranks });
    const waiting = try admin_state.pendingNames(a);
    const pend = try a.alloc(Value, if (waiting.len > 0) 1 else 0);
    if (waiting.len > 0) {
        const pv = try json.newObject(a);
        pv.put(a, "mode", str("live")) catch {};
        const names = try a.alloc(Value, waiting.len);
        for (waiting, names) |n, *slot| slot.* = str(n);
        try pv.put(a, "names", .{ .array = names });
        pend[0] = .{ .object = pv };
    }
    try cv.put(a, "pending", .{ .array = pend });
    try root.put(a, "config", .{ .object = cv });
    try putPoolStatus(a, root);
    try putSpeedFeatures(a, root);
    // the build's release (empty where the server's main set none) and each rank's memory NOW: rank 0's MemAvailable read on this
    // request, each follower's last report (they send one every 2 s over a link of their own: families/dsv41/peer_mem.zig) with its
    // age. A rank that has not reported is a row without figures, never zeros.
    try root.put(a, "release", if (admin_state.release.len > 0) str(admin_state.release) else .null);
    try root.put(a, "host_memory", try hostMemory(srv, a));
    try root.put(a, "node_health", try nodeHealth(srv, a));
    try root.put(a, "turbo", try turboRecord(srv, a));
    // the checkpoint's own name (the served name is an alias) and what the engine decided about its transports at the start
    const title = modelName(srv.io, a, srv.config.model_dir) catch null;
    try root.put(a, "model_name", if (title) |t| str(t) else .null);
    try root.put(a, "transport", try transportValue(a));
    const rep = settings.report;
    const cfg = try json.newObject(a);
    try cfg.put(a, "schema_version", try json.intValue(a, settings.schema_version));
    try cfg.put(a, "options", try json.intValue(a, visibleCount()));
    var from_env: usize = 0;
    var from_cfg: usize = 0;
    for (&settings.options, settings.resolved) |*o, r| {
        if (o.hidden) continue;
        switch (r.source) {
            .env => from_env += 1,
            .cfg => from_cfg += 1,
            .default => {},
        }
    }
    try cfg.put(a, "from_environment", try json.intValue(a, from_env));
    try cfg.put(a, "from_engine_cfg", try json.intValue(a, from_cfg));
    try cfg.put(a, "engine_cfg", if (rep.path.len > 0) str(rep.path) else .null);
    try root.put(a, "settings", .{ .object = cfg });
    // what the requests did with the per-request fields, per key label (counts only)
    const adm = try json.newObject(a);
    try adm.put(a, "policy_file", try policySummary(srv, a));
    try adm.put(a, "request_fields", try srv.policy.countsValue(srv.io, a));
    try adm.put(a, "request_fields_uncounted", try json.intValue(a, srv.policy.uncounted));
    try root.put(a, "admin", .{ .object = adm });
    return .{ .object = root };
}

/// status.host_memory: the memory each rank has free now. Rank 0's is read here; a follower's is its last report and the age of it.
fn hostMemory(srv: *Server, a: Allocator) !Value {
    const hm = try json.newObject(a);
    try hm.put(a, "basis", str("now: MemAvailable, rank 0 read at this request, each follower as it last reported (every 2 s)"));
    const world: usize = if (kv_pool.ready.load(.acquire)) @min(kv_pool.report.world, 4) else 1;
    var rows: std.ArrayList(Value) = .empty;
    var r: usize = 0;
    while (r < 4) : (r += 1) {
        const got = if (r == 0) null else api.peer_mem.read(r);
        if (r >= world and got == null) continue;
        const o = try json.newObject(a);
        try o.put(a, "rank", try json.intValue(a, r));
        if (r == 0) {
            if (prepare_gate.memAvailable(srv.io)) |avail| {
                try o.put(a, "available_bytes", try json.intValue(a, avail));
                try o.put(a, "age_ms", try json.intValue(a, 0));
            } else {
                try o.put(a, "available_bytes", .null);
                try o.put(a, "age_ms", .null);
            }
        } else if (got) |g| {
            try o.put(a, "available_bytes", try json.intValue(a, g.available));
            try o.put(a, "total_bytes", try json.intValue(a, g.total));
            try o.put(a, "age_ms", try json.intValue(a, g.age_ms));
        } else {
            try o.put(a, "available_bytes", .null);
            try o.put(a, "age_ms", .null);
        }
        try rows.append(a, .{ .object = o });
    }
    try hm.put(a, "ranks", .{ .array = rows.items });
    return .{ .object = hm };
}

/// The measured record of the Turbo switch (a prompt pass of 8,192 rows and split matrix multiplies: TF_DS_PASS_ROWS=8192 with
/// TF_DS_GEMM_HALVES=1), built into the binary as admin_ui/turbo.json; a file named by TENSORFOLD_TURBO_RECORD wins, so the
/// numbers can be filled in after a measurement without a new build. The page writes its tile from these numbers and never
/// from its own; a record that is missing, unreadable or incomplete reads `measured: false`. What Turbo costs is KV pool positions
/// (the pool is the memory it takes: free memory stays where the free target keeps it), so the record names the positions and their share.
const turbo_built_in = @embedFile("admin_ui/turbo.json");

pub fn turboFrom(a: Allocator, text: []const u8, source: []const u8) !Value {
    const out = try json.newObject(a);
    try out.put(a, "source", str(source));
    const parsed = json.parse(a, text) catch null;
    const doc: ?Value = if (parsed) |p| switch (p) {
        .ok => |v| if (v == .object) v else null,
        else => null,
    } else null;
    const d = doc orelse {
        try out.put(a, "measured", .{ .bool = false });
        return .{ .object = out };
    };
    // measured only when it says so and every number the page's tile uses is a number
    const numbers = [_][]const u8{ "long_speedup", "short_speedup", "pool_cost_positions", "pool_cost_pct" };
    var complete = if (d.get("measured")) |m| m == .bool and m.bool else false;
    for (numbers) |k| {
        const v = d.get(k) orelse {
            complete = false;
            continue;
        };
        if (v != .float and v != .int) complete = false;
    }
    try out.put(a, "measured", .{ .bool = complete });
    if (!complete) {
        if (d.get("note")) |n| if (n == .string) try out.put(a, "note", n);
        return .{ .object = out };
    }
    for ([_][]const u8{ "build", "date", "what", "source_run", "short_build", "beside_build", "long_speedup", "short_speedup", "beside_prompt_ratio", "beside_streams_ratio", "pool_normal_positions", "pool_turbo_positions", "pool_cost_positions", "pool_cost_pct", "replies_equal" }) |k| {
        if (d.get(k)) |v| switch (v) {
            .float, .int, .string, .bool => try out.put(a, k, v),
            else => {},
        };
    }
    return .{ .object = out };
}

fn turboRecord(srv: *Server, a: Allocator) !Value {
    if (std.c.getenv("TENSORFOLD_TURBO_RECORD")) |p| {
        const buf = try a.alloc(u8, 16 * 1024);
        if (std.Io.Dir.cwd().readFile(srv.io, std.mem.span(p), buf)) |text| {
            return turboFrom(a, text, "record file");
        } else |_| {}
    }
    return turboFrom(a, turbo_built_in, "built in");
}

/// A health figure the host gave, or null (`unknown` is never a number on the page).
fn knownInt(a: Allocator, v: i32) !Value {
    return if (v == api.peer_mem.unknown) .null else try json.intValue(a, v);
}
fn knownTenth(v: i32, per: f64) Value {
    return if (v == api.peer_mem.unknown) .null else .{ .float = @as(f64, @floatFromInt(v)) / per };
}
fn counterValue(a: Allocator, v: u64) !Value {
    return if (v == std.math.maxInt(u64)) .null else try json.intValue(a, v);
}

/// The NVML clock event reasons set in `bits`, by NVML's own names (the idle bit is the GPU resting, not a limit).
pub fn reasonNames(a: Allocator, bits: u64) ![]Value {
    const names = [_]struct { bit: u64, name: []const u8 }{
        .{ .bit = 0x2, .name = "applications_clocks_setting" }, .{ .bit = 0x4, .name = "sw_power_cap" },
        .{ .bit = 0x8, .name = "hw_slowdown" },                 .{ .bit = 0x10, .name = "sync_boost" },
        .{ .bit = 0x20, .name = "sw_thermal_slowdown" },        .{ .bit = 0x40, .name = "hw_thermal_slowdown" },
        .{ .bit = 0x80, .name = "hw_power_brake_slowdown" },    .{ .bit = 0x100, .name = "display_clock_setting" },
    };
    var out: std.ArrayList(Value) = .empty;
    for (names) |n| if (bits & n.bit != 0) try out.append(a, str(n.name));
    return out.items;
}

/// status.node_health: for each rank, the host's health as that Spark last reported it (rank 0 samples itself): memory, CPU,
/// the ACPI zones, the GPU (read-only: its clock and the limits NVML names, never a way to change them), the NVMe and the two rails'
/// cumulative counters (the page takes rates between two readings). A figure the host did not give is null, never zero.
fn nodeHealth(srv: *Server, a: Allocator) !Value {
    _ = srv;
    const hp = api.peer_mem;
    const nh = try json.newObject(a);
    try nh.put(a, "basis", str("as each Spark last reported it: a block when a value moved past its step, at least every 2 s; rank 0 samples itself every second"));
    try nh.put(a, "version", try json.intValue(a, hp.health_version));
    const names = try a.alloc(Value, hp.max_zones);
    for (names, 0..) |*slot, i| slot.* = str(hp.zoneName(i));
    try nh.put(a, "zone_names", .{ .array = names });
    const world: usize = if (kv_pool.ready.load(.acquire)) @min(kv_pool.report.world, 4) else 1;
    var rows: std.ArrayList(Value) = .empty;
    var r: usize = 0;
    while (r < 4) : (r += 1) {
        const got = hp.readHealth(r);
        if (r >= world and got == null) continue;
        const o = try json.newObject(a);
        try o.put(a, "rank", try json.intValue(a, r));
        if (got) |g| {
            const h = g.health;
            try o.put(a, "age_ms", try json.intValue(a, g.age_ms));
            try o.put(a, "mem_available_bytes", try json.intValue(a, h.mem_available));
            try o.put(a, "mem_total_bytes", try json.intValue(a, h.mem_total));
            try o.put(a, "load1", knownTenth(h.load1_x100, 100));
            try o.put(a, "cpu_busy_pct", knownTenth(h.cpu_busy_x10, 10));
            try o.put(a, "cpus", try knownInt(a, h.ncpu));
            const zs = try a.alloc(Value, hp.max_zones);
            for (zs, h.zone_mc) |*slot, mc| slot.* = knownTenth(mc, 1000);
            try o.put(a, "zones_c", .{ .array = zs });
            const gpu = try json.newObject(a);
            try gpu.put(a, "temp_c", try knownInt(a, h.gpu_temp_c));
            try gpu.put(a, "util_pct", try knownInt(a, h.gpu_util_pct));
            try gpu.put(a, "clock_mhz", try knownInt(a, h.gpu_clock_mhz));
            try gpu.put(a, "clock_max_mhz", try knownInt(a, h.gpu_clock_max_mhz));
            try gpu.put(a, "power_w", knownTenth(h.gpu_power_mw, 1000));
            const known = h.gpu_reasons != hp.reasons_unknown;
            try gpu.put(a, "reasons", if (known) .{ .array = try reasonNames(a, h.gpu_reasons) } else .null);
            try gpu.put(a, "idle", if (known) .{ .bool = h.gpu_reasons & 1 != 0 } else .null);
            try o.put(a, "gpu", .{ .object = gpu });
            const nv = try json.newObject(a);
            try nv.put(a, "temp_c", knownTenth(h.nvme_temp_mc, 1000));
            try nv.put(a, "wear_pct", try knownInt(a, h.nvme_wear_pct));
            try o.put(a, "nvme", .{ .object = nv });
            const rails = try a.alloc(Value, 2);
            for (rails, h.rails, [2][]const u8{ "A", "B" }) |*slot, rl, nm| {
                const ro = try json.newObject(a);
                try ro.put(a, "rail", str(nm));
                try ro.put(a, "rx_bytes", try counterValue(a, rl.rx_bytes));
                try ro.put(a, "tx_bytes", try counterValue(a, rl.tx_bytes));
                try ro.put(a, "errors", try counterValue(a, rl.errors));
                try ro.put(a, "discards", try counterValue(a, rl.discards));
                slot.* = .{ .object = ro };
            }
            try o.put(a, "rails", .{ .array = rails });
        } else try o.put(a, "age_ms", .null);
        try rows.append(a, .{ .object = o });
    }
    try nh.put(a, "ranks", .{ .array = rows.items });
    return .{ .object = nh };
}

extern "c" fn realpath(noalias path: [*:0]const u8, noalias resolved: [*]u8) ?[*:0]u8;

/// A directory name that names nothing: the folders a bake or a cache puts a checkpoint in.
fn genericName(name: []const u8) bool {
    if (name.len == 0) return true;
    for ([_][]const u8{ "model", "models", "weights", "checkpoint", "snapshot", "snapshots", "hub", "hf", "data", "current", "latest", "main" }) |g| {
        if (std.ascii.eqlIgnoreCase(name, g)) return true;
    }
    if (name.len >= 7) {
        for (name) |ch| if (!std.ascii.isHex(ch)) return false;
        return true; // a snapshot hash
    }
    return false;
}

/// "DeepSeek-V4.1-Flash-1017" reads "DeepSeek V4.1 Flash 1017".
fn tidyName(a: Allocator, name: []const u8) ![]const u8 {
    const out = try a.dupe(u8, name);
    for (out) |*ch| if (ch.* == '-' or ch.* == '_') {
        ch.* = ' ';
    };
    return out;
}

/// The repository name inside a Hugging Face cache path (.../models--org--name/snapshots/rev/...): "name". Null when the path has none.
fn hubName(path: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, path, '/');
    while (it.next()) |part| {
        if (!std.mem.startsWith(u8, part, "models--")) continue;
        const rest = part["models--".len..];
        const name = if (std.mem.indexOf(u8, rest, "--")) |i| rest[i + 2 ..] else rest;
        return if (genericName(name)) null else name;
    }
    return null;
}

/// The name a checkpoint's config.json gives itself (_name_or_path, name_or_path, name), without its organisation. Null when it names nothing
/// but a generic folder.
fn configName(a: Allocator, text: []const u8) ?[]const u8 {
    const doc = switch (json.parse(a, text) catch return null) {
        .ok => |v| v,
        .err => return null,
    };
    if (doc != .object) return null;
    for ([_][]const u8{ "_name_or_path", "name_or_path", "name" }) |k| {
        const v = doc.get(k) orelse continue;
        if (v != .string) continue;
        const base_name = std.fs.path.basename(std.mem.trimEnd(u8, v.string, "/"));
        if (!genericName(base_name)) return base_name;
    }
    return null;
}

/// A readable name for the directory's last component alone (a folder called "model" or a snapshot hash names nothing: null).
pub fn modelTitle(a: Allocator, dir: []const u8) !?[]const u8 {
    const path = std.mem.trimEnd(u8, dir, "/");
    if (hubName(path)) |n| return try tidyName(a, n);
    const last = std.fs.path.basename(path);
    if (genericName(last)) return null;
    return try tidyName(a, last);
}

/// The checkpoint's own name, from what the folder says about its source, in this order: where its config.json really lives (a baked
/// served folder links its files into the Hugging Face cache, models--org--name), the name inside config.json, then the folder's own name
/// unless it is a generic word. Null when none names it (the page then shows the served name).
pub fn modelName(io: std.Io, a: Allocator, dir: []const u8) !?[]const u8 {
    if (dir.len == 0) return null;
    const cfg = try std.fs.path.joinZ(a, &.{ dir, "config.json" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (realpath(cfg, &buf)) |real| if (hubName(std.mem.span(real))) |n| return try tidyName(a, n);
    const text = std.Io.Dir.cwd().readFileAlloc(io, cfg, a, .limited(16 << 20)) catch null;
    if (text) |t| if (configName(a, t)) |n| return try tidyName(a, n);
    return modelTitle(a, dir);
}

fn signalWord(sig: api.transport.Signal) Value {
    return switch (sig) {
        .write => str("write"),
        .atomic => str("atomic"),
        .unknown => .null,
    };
}

/// status.transport: what the start decided, in the engine's own switches (api.transport). A part the start did not record is null, never a guess.
fn transportValue(a: Allocator) !Value {
    const f = api.transport.facts;
    const t = try json.newObject(a);
    const roci = try json.newObject(a);
    try roci.put(a, "known", .{ .bool = f.rocinante_known });
    try roci.put(a, "waits_in_card", .{ .bool = f.rocinante_wait });
    try roci.put(a, "arrival_signal", signalWord(f.rocinante_signal));
    try t.put(a, "prompt_exchanges", .{ .object = roci });
    const rings = try json.newObject(a);
    try rings.put(a, "known", .{ .bool = f.rings_known });
    try rings.put(a, "on", .{ .bool = f.rings_on });
    try rings.put(a, "waits_in_card", .{ .bool = f.rings_wait });
    try rings.put(a, "arrival_signal", if (f.rings_word) signalWord(f.rings_signal) else .null);
    try t.put(a, "generation_exchanges", .{ .object = rings });
    const watch = try json.newObject(a);
    try watch.put(a, "links", try json.intValue(a, f.watch_links));
    try t.put(a, "link_watch", .{ .object = watch });
    // the failure counters (a lane failed, a ring aborted) are not kept anywhere a request can read without a lock on the exchange's path: not served
    return .{ .object = t };
}

const gib_f: f64 = 1 << 30;

/// A number of GiB as canonical text (the shortest decimal), the way every value in the admin answers is a string.
fn gibText(a: Allocator, bytes: u64) ![]const u8 {
    return std.fmt.allocPrint(a, "{d}", .{@as(f64, @floatFromInt(bytes)) / gib_f});
}

fn featureOf(id: u8) kv_pool.Feature {
    return if (id <= @backingInt(kv_pool.Feature.other)) @fromBackingInt(@intCast(id)) else .other;
}

/// GET /ledger: what the start took, line by line, on every rank, largest first, where each line was taken, each rank's sum
/// against what its device holds at the start's end, the pool and the vote that chose it. Empty, and saying so, in a process that
/// sized no pool.
pub fn ledger(a: Allocator) !Value {
    const root = try json.newObject(a);
    if (!kv_pool.ready.load(.acquire)) {
        try root.put(a, "entries", .{ .array = try a.alloc(Value, 0) });
        try root.put(a, "note", str("no start record: this process sized no KV pool, or the engine has not finished starting"));
        return .{ .object = root };
    }
    const rep = &kv_pool.report;
    const world: usize = @min(rep.world, 4);
    const parts: []const kv_pool.Part = if (rep.have_final) rep.final[0..world] else rep.table[0..world];
    const Row = struct { rank: u32, e: kv_pool.Entry, site: []const u8 };
    var rows: std.ArrayList(Row) = .empty;
    for (parts, 0..) |p, r| {
        for (p.entries[0..@min(p.n, kv_pool.max_entries)], 0..) |e, i| {
            // the sites are rank 0's own; another rank runs the same binary, so a feature's line is the same call
            var site: []const u8 = "";
            if (r == 0 and i < rep.mine.n) {
                site = try std.fmt.allocPrint(a, "{s}:{d}", .{ rep.mine.sites[i].file, rep.mine.sites[i].line });
            } else for (rep.mine.entries[0..rep.mine.n], 0..) |m, k| if (m.feature == e.feature) {
                site = try std.fmt.allocPrint(a, "{s}:{d}", .{ rep.mine.sites[k].file, rep.mine.sites[k].line });
                break;
            };
            try rows.append(a, .{ .rank = @intCast(r), .e = e, .site = site });
        }
    }
    std.mem.sort(Row, rows.items, {}, struct {
        fn big(_: void, x: Row, y: Row) bool {
            return (x.e.device + x.e.host) > (y.e.device + y.e.host);
        }
    }.big);
    const out = try a.alloc(Value, rows.items.len);
    for (rows.items, out) |r, *slot| {
        const f = featureOf(r.e.feature);
        const info = kv_pool.featureInfo(f);
        const o = try json.newObject(a);
        try o.put(a, "rank", try json.intValue(a, r.rank));
        try o.put(a, "feature_id", str(@tagName(f)));
        try o.put(a, "feature", str(info.label));
        try o.put(a, "option", if (info.option.len > 0) str(info.option) else .null);
        try o.put(a, "device_bytes", try json.intValue(a, r.e.device));
        try o.put(a, "host_bytes", try json.intValue(a, r.e.host));
        try o.put(a, "bytes", try json.intValue(a, r.e.device + r.e.host));
        try o.put(a, "runtime_cap_bytes", .null);
        try o.put(a, "site", str(r.site));
        try o.put(a, "label", str(if (f == .other) "remainder" else "measured"));
        slot.* = .{ .object = o };
    }
    try root.put(a, "entries", .{ .array = out });
    const ranks = try a.alloc(Value, world);
    for (parts, rep.table[0..world], ranks, 0..) |p, v, *slot, r| {
        var total: u64 = 0;
        for (p.entries[0..@min(p.n, kv_pool.max_entries)]) |e| total += e.device;
        const o = try json.newObject(a);
        try o.put(a, "rank", try json.intValue(a, r));
        try o.put(a, "device_bytes", try json.intValue(a, v.total_bytes));
        try o.put(a, "free_at_open", try json.intValue(a, v.free_bytes));
        try o.put(a, "free_at_ready", try json.intValue(a, p.free_bytes));
        try o.put(a, "used_at_ready", try json.intValue(a, p.total_bytes -| p.free_bytes));
        try o.put(a, "ledger_total", try json.intValue(a, total));
        try o.put(a, "reserve", try json.intValue(a, v.growth_bytes));
        try o.put(a, "pool_tokens", try json.intValue(a, rep.pool_active));
        try o.put(a, "fit_tokens", if (v.target_bytes != 0) try json.intValue(a, v.fit) else .null);
        slot.* = .{ .object = o };
    }
    try root.put(a, "ranks", .{ .array = ranks });
    const lead = rep.table[0];
    const pool = try json.newObject(a);
    try pool.put(a, "requested", try json.intValue(a, lead.asked));
    try pool.put(a, "active", try json.intValue(a, rep.pool_active));
    try pool.put(a, "predicted", try json.intValue(a, if (rep.decision.verdict == .auto) rep.decision.pool else lead.asked));
    try pool.put(a, "free_target_gib", if (lead.target_bytes != 0) str(try gibText(a, lead.target_bytes)) else .null);
    try pool.put(a, "vote", str("min over ranks"));
    try pool.put(a, "predicted_bytes", try json.intValue(a, rep.pool_predicted));
    try pool.put(a, "taken_bytes", try json.intValue(a, rep.pool_taken));
    // the engine's own difference: rank 0's pool as the driver counted it taken, less what the layout says (the page never computes it)
    const diff: f64 = (@as(f64, @floatFromInt(rep.pool_taken)) - @as(f64, @floatFromInt(rep.pool_predicted))) / gib_f;
    try pool.put(a, "difference_gib", .{ .float = diff });
    try root.put(a, "pool", .{ .object = pool });
    try root.put(a, "bytes_per_token", try json.intValue(a, lead.bytes_per_token));
    return .{ .object = root };
}

/// status.kv_pool and status.memory: the pool as the start sized it and each Spark's memory at the start. Both are left out
/// until a start has recorded them: a group with no source is never zero-filled.
/// `speed_features`: every essential speed feature with whether it runs, whether the operator chose it off (`asked`), the reason in
/// plain words and what it costs without it (core/speed_report.zig; the same list the start printed as one block).
fn putSpeedFeatures(a: Allocator, root: *json.Object) !void {
    const r = try a.create(api.speed_report.Report);
    api.speed_report.current(r);
    const rows = try a.alloc(Value, r.n);
    for (r.items[0..r.n], rows) |f, *slot| {
        const o = try json.newObject(a);
        try o.put(a, "name", str(f.name));
        try o.put(a, "env", str(f.env));
        try o.put(a, "on", .{ .bool = f.on });
        try o.put(a, "asked", .{ .bool = f.asked });
        try o.put(a, "reason", str(try a.dupe(u8, f.reason)));
        try o.put(a, "cost", str(f.cost));
        slot.* = .{ .object = o };
    }
    try root.put(a, "speed_features", .{ .array = rows });
}

fn putPoolStatus(a: Allocator, root: *json.Object) !void {
    if (!kv_pool.ready.load(.acquire)) return;
    const rep = &kv_pool.report;
    const world: usize = @min(rep.world, 4);
    const lead = rep.table[0];
    const kvo = try json.newObject(a);
    try kvo.put(a, "mode", str(if (lead.target_bytes != 0) "free_target" else "set"));
    try kvo.put(a, "tokens_active", try json.intValue(a, rep.pool_active));
    try kvo.put(a, "tokens_requested", try json.intValue(a, lead.asked));
    try kvo.put(a, "usable_window", try json.intValue(a, lead.floor));
    const sxc: u64 = @as(u64, rep.streams) * lead.floor;
    try kvo.put(a, "streams_x_context", try json.intValue(a, sxc));
    try kvo.put(a, "extra_tokens", try json.intValue(a, if (rep.pool_active > sxc) rep.pool_active - sxc else 0));
    try kvo.put(a, "free_target_gib", if (lead.target_bytes != 0) str(try gibText(a, lead.target_bytes)) else .null);
    try kvo.put(a, "growth_gib", str(try gibText(a, lead.growth_bytes)));
    try kvo.put(a, "bytes_per_token", try json.intValue(a, lead.bytes_per_token));
    if (rep.resize_version != 0) {
        const ro = try json.newObject(a);
        try ro.put(a, "version", try json.intValue(a, rep.resize_version));
        try ro.put(a, "verdict", str(@tagName(rep.resize_verdict)));
        try ro.put(a, "from_tokens", try json.intValue(a, rep.resize_from));
        try ro.put(a, "fit_tokens", try json.intValue(a, rep.resize_wanted));
        try ro.put(a, "rank", try json.intValue(a, rep.resize_rank));
        try kvo.put(a, "last_resize", .{ .object = ro });
    }
    try root.put(a, "kv_pool", .{ .object = kvo });
    const mem = try json.newObject(a);
    try mem.put(a, "basis", str("at the start"));
    const parts: []const kv_pool.Part = if (rep.have_final) rep.final[0..world] else rep.table[0..world];
    const ranks = try a.alloc(Value, world);
    for (rep.table[0..world], parts, ranks, 0..) |v, p, *slot, r| {
        var total: u64 = 0;
        for (p.entries[0..@min(p.n, kv_pool.max_entries)]) |e| total += e.device;
        const o = try json.newObject(a);
        try o.put(a, "rank", try json.intValue(a, r));
        try o.put(a, "device_bytes", try json.intValue(a, v.total_bytes));
        try o.put(a, "free_at_open_gib", str(try gibText(a, v.free_bytes)));
        try o.put(a, "free_target_gib", if (v.target_bytes != 0) str(try gibText(a, v.target_bytes)) else .null);
        try o.put(a, "ledger_gib", str(try gibText(a, total)));
        slot.* = .{ .object = o };
    }
    try mem.put(a, "ranks", .{ .array = ranks });
    try root.put(a, "memory", .{ .object = mem });
}

/// A read view of something no change has made yet: an empty list under `list` and the reason it is empty.
fn emptyView(a: Allocator, list: []const u8, note: []const u8) !Value {
    const root = try json.newObject(a);
    try root.put(a, list, .{ .array = try a.alloc(Value, 0) });
    try root.put(a, "note", str(note));
    return .{ .object = root };
}

/// GET /requests: the requests in flight and the last finished ones (labels, counts and times, never text), and the
/// replies counted so far by key label and status.
pub fn requestsValue(srv: *Server, a: Allocator) !Value {
    const v = try srv.requests.view(srv.io, a, clock.nowNs(srv.io));
    const root = try json.newObject(a);
    try root.put(a, "in_flight", v.in_flight);
    try root.put(a, "recent", v.recent);
    try root.put(a, "untracked", try json.intValue(a, v.untracked));
    try root.put(a, "by_label", try byLabel(srv, a));
    return .{ .object = root };
}

fn byLabel(srv: *Server, a: Allocator) !Value {
    const Row = struct { label: []const u8, total: u64, statuses: std.ArrayList(struct { status: u16, count: u64 }) };
    var rows: std.ArrayList(Row) = .empty;
    {
        srv.metrics.mutex.lockUncancelable(srv.io);
        defer srv.metrics.mutex.unlock(srv.io);
        for (srv.metrics.requests.items) |r| {
            const at = for (rows.items, 0..) |x, i| {
                if (std.mem.eql(u8, x.label, r.key)) break i;
            } else blk: {
                try rows.append(a, .{ .label = try a.dupe(u8, r.key), .total = 0, .statuses = .empty });
                break :blk rows.items.len - 1;
            };
            rows.items[at].total += r.count;
            try rows.items[at].statuses.append(a, .{ .status = r.status, .count = r.count });
        }
    }
    const out = try a.alloc(Value, rows.items.len);
    for (rows.items, out) |x, *slot| {
        const o = try json.newObject(a);
        try o.put(a, "label", str(x.label));
        try o.put(a, "requests", try json.intValue(a, x.total));
        const st = try json.newObject(a);
        for (x.statuses.items) |s| try st.put(a, try std.fmt.allocPrint(a, "{d}", .{s.status}), try json.intValue(a, s.count));
        try o.put(a, "statuses", .{ .object = st });
        slot.* = .{ .object = o };
    }
    return .{ .array = out };
}

/// The value of query parameter `name` in a raw request path, percent-decoded into `a`; null when it is absent.
fn queryValue(a: Allocator, raw_path: []const u8, name: []const u8) ?[]const u8 {
    const q = std.mem.indexOfScalar(u8, raw_path, '?') orelse return null;
    var it = std.mem.splitScalar(u8, raw_path[q + 1 ..], '&');
    while (it.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse pair.len;
        if (!std.mem.eql(u8, pair[0..eq], name)) continue;
        const raw = if (eq < pair.len) pair[eq + 1 ..] else "";
        var out = a.alloc(u8, raw.len) catch return null;
        var n: usize = 0;
        var i: usize = 0;
        while (i < raw.len) : (i += 1) {
            if (raw[i] == '+') {
                out[n] = ' ';
            } else if (raw[i] == '%' and i + 2 < raw.len + 0 and std.ascii.isHex(raw[i + 1]) and std.ascii.isHex(raw[i + 2])) {
                out[n] = std.fmt.parseInt(u8, raw[i + 1 .. i + 3], 16) catch return null;
                i += 2;
            } else out[n] = raw[i];
            n += 1;
        }
        return out[0..n];
    }
    return null;
}

fn queryNumber(ctx: Ctx, name: []const u8, default: u64, max: u64) ?u64 {
    const t = queryValue(ctx.a, ctx.conn.path, name) orelse return default;
    return @min(std.fmt.parseInt(u64, t, 10) catch return null, max);
}

/// GET /goals: the controller's state over its window (empty while the sampler is not running). Alert-only: nothing is changed.
fn goalsValue(srv: *Server, a: Allocator) !Value {
    const cfg = admin_goals.config();
    const r: api.goals.Report = if (srv.series) |s| s.goalsReport(srv.io, cfg, admin_goals.nNow()) else .{ .state = if (cfg.mode == .off) .off else .idle, .n_now = admin_goals.nNow() };
    return admin_goals.value(a, cfg, r);
}

/// GET /series?name=&seconds=&step=: the sampler's ring (empty while the sampler is not running).
fn seriesReply(ctx: Ctx) void {
    const name = queryValue(ctx.a, ctx.conn.path, "name") orelse return fail(ctx.conn, ctx.a, 400, "missing_field", "name: one of decode_tps, prefill_tps, running, waiting, generation_tokens, memory_active");
    const seconds = queryNumber(ctx, "seconds", 600, admin_series.capacity) orelse return fail(ctx.conn, ctx.a, 400, "bad_query", "seconds: a whole number");
    const step = queryNumber(ctx, "step", 1, admin_series.capacity) orelse return fail(ctx.conn, ctx.a, 400, "bad_query", "step: a whole number");
    const root = json.newObject(ctx.a) catch return;
    root.put(ctx.a, "name", str(name)) catch return;
    // "seat_kv": every seat's KV share in one reply (the pool chart's layers), each as a points / peaks pair
    if (std.mem.eql(u8, name, "seat_kv")) {
        const rows = ctx.a.alloc(Value, admin_series.seat_rows) catch return;
        var since: ?i64 = null;
        var last: ?i64 = null;
        var stepw: u32 = @intCast(@max(1, step));
        for (rows, 0..) |*slot, k| {
            const o = json.newObject(ctx.a) catch return;
            var nb: [24]u8 = undefined;
            const nm = std.fmt.bufPrint(&nb, "seat_kv_{d}", .{k}) catch return;
            const w = if (ctx.srv.series) |s| (s.window(ctx.srv.io, ctx.a, nm, @intCast(seconds), @intCast(@max(1, step))) catch return fail(ctx.conn, ctx.a, 500, "internal", "admin: out of memory")) else null;
            if (w) |x| {
                since = x.since_ms;
                last = x.last_ms;
                stepw = x.step_s;
                o.put(ctx.a, "points", .{ .array = x.points }) catch return;
                o.put(ctx.a, "peaks", .{ .array = x.peaks }) catch return;
            } else {
                o.put(ctx.a, "points", .{ .array = ctx.a.alloc(Value, 0) catch return }) catch return;
                o.put(ctx.a, "peaks", .{ .array = ctx.a.alloc(Value, 0) catch return }) catch return;
            }
            slot.* = .{ .object = o };
        }
        root.put(ctx.a, "step_s", json.intValue(ctx.a, stepw) catch return) catch return;
        root.put(ctx.a, "since_ms", if (since) |v| (json.intValue(ctx.a, v) catch return) else .null) catch return;
        root.put(ctx.a, "last_ms", if (last) |v| (json.intValue(ctx.a, v) catch return) else .null) catch return;
        root.put(ctx.a, "seats", .{ .array = rows }) catch return;
        return sendValue(ctx, 200, .{ .object = root });
    }
    if (ctx.srv.series) |s| {
        const w = (s.window(ctx.srv.io, ctx.a, name, @intCast(seconds), @intCast(@max(1, step))) catch return fail(ctx.conn, ctx.a, 500, "internal", "admin: out of memory")) orelse
            return fail(ctx.conn, ctx.a, 400, "unknown_series", "name: one of decode_tps, prefill_tps, running, waiting, generation_tokens, memory_active");
        root.put(ctx.a, "step_s", json.intValue(ctx.a, w.step_s) catch return) catch return;
        root.put(ctx.a, "since_ms", json.intValue(ctx.a, w.since_ms) catch return) catch return;
        root.put(ctx.a, "last_ms", json.intValue(ctx.a, w.last_ms) catch return) catch return;
        root.put(ctx.a, "points", .{ .array = w.points }) catch return;
        root.put(ctx.a, "peaks", .{ .array = w.peaks }) catch return;
    } else {
        root.put(ctx.a, "step_s", json.intValue(ctx.a, @max(1, step)) catch return) catch return;
        root.put(ctx.a, "since_ms", .null) catch return;
        root.put(ctx.a, "last_ms", .null) catch return;
        root.put(ctx.a, "points", .{ .array = ctx.a.alloc(Value, 0) catch return }) catch return;
        root.put(ctx.a, "peaks", .{ .array = ctx.a.alloc(Value, 0) catch return }) catch return;
    }
    sendValue(ctx, 200, .{ .object = root });
}

/// GET /logs?since=&limit=&level=&contains=: the newest server lines from the ring.
fn logsReply(ctx: Ctx) void {
    const since = queryNumber(ctx, "since", 0, std.math.maxInt(u64)) orelse return fail(ctx.conn, ctx.a, 400, "bad_query", "since: a whole number");
    const limit = queryNumber(ctx, "limit", 500, log.ring.capacity) orelse return fail(ctx.conn, ctx.a, 400, "bad_query", "limit: a whole number");
    const level_text = queryValue(ctx.a, ctx.conn.path, "level") orelse "info";
    const min: log.ring.Level = if (std.mem.eql(u8, level_text, "info")) .info else if (std.mem.eql(u8, level_text, "warn")) .warn else if (std.mem.eql(u8, level_text, "error")) .@"error" else return fail(ctx.conn, ctx.a, 400, "bad_query", "level: info, warn or error");
    const page = log.ring.read(ctx.a, since, @intCast(limit), min, queryValue(ctx.a, ctx.conn.path, "contains") orelse "") catch return fail(ctx.conn, ctx.a, 500, "internal", "admin: out of memory");
    const root = json.newObject(ctx.a) catch return;
    const rows = ctx.a.alloc(Value, page.lines.len) catch return;
    for (page.lines, rows) |x, *slot| {
        const o = json.newObject(ctx.a) catch return;
        o.put(ctx.a, "seq", json.intValue(ctx.a, x.seq) catch return) catch return;
        o.put(ctx.a, "at_ms", json.intValue(ctx.a, x.at_ms) catch return) catch return;
        o.put(ctx.a, "level", str(@tagName(x.level))) catch return;
        o.put(ctx.a, "text", str(ctx.a.dupe(u8, x.text[0..x.len]) catch return)) catch return;
        slot.* = .{ .object = o };
    }
    root.put(ctx.a, "lines", .{ .array = rows }) catch return;
    root.put(ctx.a, "next_seq", json.intValue(ctx.a, page.next_seq) catch return) catch return;
    root.put(ctx.a, "dropped", json.intValue(ctx.a, page.dropped) catch return) catch return;
    sendValue(ctx, 200, .{ .object = root });
}

/// GET /keys: the labels of both key files and what each store holds. Never a key, a digest or a file's path.
pub fn keysValue(srv: *Server, a: Allocator) !Value {
    const root = try json.newObject(a);
    var api_labels: []const []const u8 = &.{};
    var shared_summary: ?auth.Store.Summary = null;
    if (srv.keys) |k| shared_summary = try k.labels(a, &api_labels);
    const counts = try byLabel(srv, a);
    const rows = try a.alloc(Value, api_labels.len);
    for (api_labels, rows) |label, *slot| {
        const o = try json.newObject(a);
        try o.put(a, "label", str(label));
        var n: u64 = 0;
        for (counts.array) |c| if (std.mem.eql(u8, json.strOr(c.get("label")), label)) {
            n = std.fmt.parseInt(u64, c.get("requests").?.int, 10) catch 0;
        };
        try o.put(a, "requests", try json.intValue(a, n));
        slot.* = .{ .object = o };
    }
    try root.put(a, "api_keys", .{ .array = rows });
    var admin_labels: []const []const u8 = &.{};
    var admin_summary: ?auth.Store.Summary = null;
    if (srv.admin_keys) |k| admin_summary = try k.labels(a, &admin_labels);
    const arows = try a.alloc(Value, admin_labels.len);
    for (admin_labels, arows) |label, *slot| {
        const o = try json.newObject(a);
        const sc = auth.scopeOf(label);
        try o.put(a, "label", str(label));
        try o.put(a, "scope", str(@tagName(sc.scope)));
        if (sc.scope == .agent) try o.put(a, "profile", str(label));
        slot.* = .{ .object = o };
    }
    try root.put(a, "admin_keys", .{ .array = arows });
    const stores = try json.newObject(a);
    try stores.put(a, "api", try storeValue(a, shared_summary));
    try stores.put(a, "admin", try storeValue(a, admin_summary));
    try root.put(a, "stores", .{ .object = stores });
    return .{ .object = root };
}

fn storeValue(a: Allocator, summary: ?auth.Store.Summary) !Value {
    const o = try json.newObject(a);
    const s = summary orelse {
        try o.put(a, "keys", try json.intValue(a, 0));
        try o.put(a, "valid", .{ .bool = true });
        try o.put(a, "refused_reason", .null);
        return .{ .object = o };
    };
    try o.put(a, "keys", try json.intValue(a, s.keys));
    try o.put(a, "valid", .{ .bool = s.valid });
    try o.put(a, "refused_reason", if (s.refusal.len > 0) str(try a.dupe(u8, s.refusal)) else .null);
    return .{ .object = o };
}

/// What a write must pass before its body is looked at: JSON content, an Origin that is this server's own (a browser
/// form cannot set either), and a body of at most 64 KiB. Returns the body, or null after answering.
fn writeBody(ctx: Ctx) ?[]const u8 {
    const conn = ctx.conn;
    const a = ctx.a;
    const kind = conn.header("Content-Type") orelse "";
    const semi = std.mem.indexOfScalar(u8, kind, ';') orelse kind.len;
    if (!std.ascii.eqlIgnoreCase(std.mem.trim(u8, kind[0..semi], " \t"), "application/json")) {
        _ = http_body.read(conn, a, http_body.limit) catch {};
        fail(conn, a, 415, "unsupported_media_type", "send Content-Type: application/json");
        return null;
    }
    if (conn.header("Origin")) |origin| {
        const host = conn.header("Host") orelse "";
        const at = std.mem.indexOf(u8, origin, "://");
        const authority = if (at) |i| origin[i + 3 ..] else origin;
        if (host.len == 0 or !std.ascii.eqlIgnoreCase(authority, host)) {
            _ = http_body.read(conn, a, http_body.limit) catch {};
            fail(conn, a, 403, "origin_denied", "the Origin is not this server's");
            return null;
        }
    }
    const body = http_body.read(conn, a, 65536) catch {
        fail(conn, a, 500, "internal", "admin: out of memory");
        return null;
    };
    switch (body) {
        .ok => |b| return b,
        .refused => {
            conn.close = true;
            fail(conn, a, 413, "body_too_large", "an admin request body is at most 65,536 bytes");
            return null;
        },
    }
}

/// The policy file's state: whether one is configured, whether it loaded, why not, and how many labels have a row.
fn policySummary(srv: *Server, a: Allocator) !Value {
    const s = try srv.policy.summary(srv.io, a);
    const o = try json.newObject(a);
    try o.put(a, "configured", .{ .bool = s.configured });
    try o.put(a, "valid", .{ .bool = s.valid });
    try o.put(a, "labels", try json.intValue(a, s.labels));
    try o.put(a, "refused_reason", if (s.refusal.len > 0) str(s.refusal) else .null);
    return .{ .object = o };
}

/// GET /policy: what a caller may set about its own request, per key label, from the policy file; the agent profiles are not
/// built in this release (`profiles` is empty).
pub fn policyValue(srv: *Server, a: Allocator) !Value {
    const root = try json.newObject(a);
    try root.put(a, "profiles", .{ .array = try a.alloc(Value, 0) });
    try root.put(a, "note", str("no agent profile exists in this release; request_caps are the per-key limits on a request's own fields (pri, think_budget, wait_ms)"));
    try root.put(a, "policy_file", try policySummary(srv, a));
    try root.put(a, "request_caps", try srv.policy.capsValue(srv.io, a));
    try root.put(a, "request_fields", try srv.policy.countsValue(srv.io, a));
    return .{ .object = root };
}

/// POST /keys/reload: both key stores reread their files now (what SIGHUP does), and say how many keys each holds.
fn keysReload(ctx: Ctx) void {
    const body = writeBody(ctx) orelse return;
    const parsed = json.parse(ctx.a, if (body.len == 0) "{}" else body) catch return fail(ctx.conn, ctx.a, 500, "internal", "admin: out of memory");
    const doc = switch (parsed) {
        .ok => |v| v,
        .err => return fail(ctx.conn, ctx.a, 400, "bad_json", "the body is not JSON"),
    };
    if (doc != .object) return fail(ctx.conn, ctx.a, 400, "bad_json", "the body is a JSON object");
    if (doc.object.count() > 0) return fail(ctx.conn, ctx.a, 400, "unknown_field", doc.object.keys()[0]);
    const root = json.newObject(ctx.a) catch return;
    const api_summary: ?auth.Store.Summary = if (ctx.srv.keys) |k| k.reloadNow() else null;
    const admin_summary: ?auth.Store.Summary = if (ctx.srv.admin_keys) |k| k.reloadNow() else null;
    root.put(ctx.a, "api", storeValue(ctx.a, api_summary) catch return) catch return;
    root.put(ctx.a, "admin", storeValue(ctx.a, admin_summary) catch return) catch return;
    if (ctx.srv.policy.path != null) {
        ctx.srv.policy.reload(ctx.srv.io);
        root.put(ctx.a, "policy", policySummary(ctx.srv, ctx.a) catch return) catch return;
    }
    sendValue(ctx, 200, .{ .object = root });
}

/// GET /history?limit=&before_version=&label=&name=&outcome=: engine.history.jsonl's rows, newest first. A change to an option
/// the surface hides is left out of its row, so a filter by name cannot find one; a line that is no row is counted, never shown.
fn historyReply(ctx: Ctx) void {
    const a = ctx.a;
    const memory_only = settings_history.path.len == 0;
    if (memory_only) {
        const kept = settings_history.memoryLines(a) catch return fail(ctx.conn, a, 500, "internal", "admin: out of memory");
        if (kept.len == 0) return reply(ctx, emptyView(a, "rows", "no engine.cfg was read at the start, so no history file is kept"));
    }
    const limit = queryNumber(ctx, "limit", 100, 1000) orelse return fail(ctx.conn, a, 400, "bad_query", "limit: a whole number");
    const before = queryNumber(ctx, "before_version", std.math.maxInt(u64), std.math.maxInt(u64)) orelse return fail(ctx.conn, a, 400, "bad_query", "before_version: a whole number");
    const want_label = queryValue(a, ctx.conn.path, "label");
    const want_name = queryValue(a, ctx.conn.path, "name");
    const want_outcome = queryValue(a, ctx.conn.path, "outcome");
    const loaded: settings_history.Loaded = if (memory_only) .{ .lines = settings_history.memoryLines(a) catch return fail(ctx.conn, a, 500, "internal", "admin: out of memory") } else settings_history.load(a, ctx.srv.io, settings_history.path) catch |e| switch (e) {
        error.OutOfMemory => return fail(ctx.conn, a, 500, "internal", "admin: out of memory"),
        else => return reply(ctx, emptyView(a, "rows", std.fmt.allocPrint(a, "the history file cannot be read ({s})", .{@errorName(e)}) catch return)),
    };
    var kept: std.ArrayList(Value) = .empty;
    var damaged: usize = 0;
    var more = false;
    var i = loaded.lines.len;
    while (i > 0) {
        i -= 1;
        const parsed = json.parse(a, loaded.lines[i]) catch return fail(ctx.conn, a, 500, "internal", "admin: out of memory");
        const row = switch (parsed) {
            .ok => |v| v,
            .err => {
                damaged += 1;
                continue;
            },
        };
        if (row != .object) {
            damaged += 1;
            continue;
        }
        const vtext = if (row.get("version")) |v| (if (v == .int) v.int else "") else "";
        const version = std.fmt.parseInt(u64, vtext, 10) catch {
            damaged += 1;
            continue;
        };
        if (version >= before) continue;
        if (want_label) |l| if (!std.mem.eql(u8, json.strOr(row.get("label")), l)) continue;
        if (want_outcome) |o| if (!std.mem.eql(u8, json.strOr(row.get("outcome")), o)) continue;
        // the changes the surface may show: a hidden option's change is dropped
        var shown: std.ArrayList(Value) = .empty;
        if (row.get("changes")) |cs| if (cs == .array) for (cs.array) |c| {
            const idt = if (c.get("id")) |x| (if (x == .int) x.int else "") else "";
            const id = std.fmt.parseInt(u32, idt, 10) catch 0;
            if (settings.byId(id)) |o| if (o.hidden) continue;
            shown.append(a, c) catch return;
        };
        if (want_name) |n| {
            const found = for (shown.items) |c| {
                if (std.mem.eql(u8, json.strOr(c.get("name")), n)) break true;
            } else false;
            if (!found) continue;
        }
        if (kept.items.len >= limit) {
            more = true;
            break;
        }
        row.object.put(a, "changes", .{ .array = shown.items }) catch return;
        if (row.get("at_ms")) |ms| if (ms == .int) if (std.fmt.parseInt(i64, ms.int, 10)) |m| {
            row.object.put(a, "at", str(rfc3339(a, m) catch return)) catch return;
        } else |_| {};
        kept.append(a, row) catch return;
    }
    // the rows after this page are older: ask again for versions below the last one shown
    const root = json.newObject(a) catch return;
    root.put(a, "rows", .{ .array = kept.items }) catch return;
    const last_version: ?Value = if (more and kept.items.len > 0) kept.items[kept.items.len - 1].get("version") else null;
    root.put(a, "next_before_version", last_version orelse .null) catch return;
    root.put(a, "damaged", json.intValue(a, damaged) catch return) catch return;
    if (memory_only) {
        root.put(a, "memory_only", .{ .bool = true }) catch return;
        root.put(a, "note", str("no engine.cfg was read at the start: this history is kept in memory and is lost when the engine stops")) catch return;
    } else root.put(a, "memory_only", .{ .bool = false }) catch return;
    sendValue(ctx, 200, .{ .object = root });
}

/// An agent key's limits on a row, or null when it passes: the row must be one an agent may set (not an `out_changes` row), the
/// value inside the row's agent bounds, and a lease no longer than the row's longest. A value that is no number is left to
/// `settings.check`.
pub const Why = struct { code: []const u8, message: []const u8 };
pub fn agentCheck(o: *const settings.Option, value: []const u8, ttl_s: ?u64, buf: []u8) ?Why {
    var permitted = o.agent != null and !o.out_changes;
    if (permitted) permitted = for (o.who) |w| {
        if (w == .agent) break true;
    } else false;
    if (!permitted) return .{ .code = "not_permitted_for_agent", .message = std.fmt.bufPrint(buf, "{s}: an agent key may not set this option", .{o.name}) catch "an agent key may not set this option" };
    const ag = o.agent.?;
    if (std.fmt.parseFloat(f64, value)) |x| {
        if (x < ag.min or x > ag.max) return .{ .code = "outside_agent_range", .message = std.fmt.bufPrint(buf, "{s}={s}: an agent may set it from {d} to {d} {s}", .{ o.name, value, ag.min, ag.max, o.unit }) catch "outside the range an agent may set" };
    } else |_| {}
    if (ttl_s) |t| if (t > ag.max_ttl_s) return .{ .code = "ttl_too_long", .message = std.fmt.bufPrint(buf, "{s}: a lease on it lasts at most {d} s", .{ o.name, ag.max_ttl_s }) catch "the lease is too long" };
    return null;
}

/// The first condition of `requires` (`name=value`) that `look` (anything with `value(name) ?[]const u8`) says does not hold, or
/// null. An option `look` does not know holds no value, so a condition on it is not met.
pub fn requiresUnmet(requires: []const []const u8, look: anytype) ?[]const u8 {
    for (requires) |cond| {
        const eq = std.mem.indexOfScalar(u8, cond, '=') orelse cond.len;
        const have = look.value(cond[0..eq]) orelse return cond;
        if (!std.mem.eql(u8, have, if (eq < cond.len) cond[eq + 1 ..] else "")) return cond;
    }
    return null;
}

/// What a request key names: an option, a retired name, or nothing the caller may know of (a hidden option too).
const Named = union(enum) { option: *const settings.Option, retired: *const settings.Retired, nothing };

fn named(key: []const u8) Named {
    if (settings.retiredByKey(key)) |r| return .{ .retired = r };
    const o = settings.byKey(key) orelse return .nothing;
    if (o.hidden) return .nothing;
    return .{ .option = o };
}

/// The canonical text of a request's value: a string as it is, a number as its shortest decimal, true and false as 1 and 0
/// for a flag (as the words for any other kind, which `settings.check` then refuses), null as `unset`. Null: not a value.
fn valueText(a: Allocator, v: Value, flag: bool) ?[]const u8 {
    return switch (v) {
        .null => "unset",
        .string => |t| t,
        .int => |t| t,
        .float => |x| blk: {
            if (!std.math.isFinite(x)) break :blk "nan";
            if (x == @floor(x) and @abs(x) < 1e15) break :blk std.fmt.allocPrint(a, "{d}", .{@as(i64, @intFromFloat(x))}) catch null;
            break :blk std.fmt.allocPrint(a, "{d}", .{x}) catch null;
        },
        .bool => |b| if (flag) (if (b) "1" else "0") else (if (b) "true" else "false"),
        else => null,
    };
}

fn refusedRow(a: Allocator, name: []const u8, id: ?u16, value: ?[]const u8, code: []const u8, message: []const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "name", str(name));
    try o.put(a, "id", if (id) |x| try json.intValue(a, x) else .null);
    try o.put(a, "value", if (value) |x| str(x) else .null);
    try o.put(a, "code", str(code));
    try o.put(a, "message", str(message));
    return .{ .object = o };
}

/// One change a request asks for, once it passed every check: the option, the canonical text (`unset`: leave it to the engine),
/// what it is now and how it would apply.
pub const Change = struct { o: *const settings.Option, text: []const u8, from: ?[]const u8, file_from: ?[]const u8, live: bool };

/// What a request that names changes was found to be: the rows it may make, the ones refused, what to warn of, and its other fields.
const Analysis = struct {
    changes: []const Change,
    resolved: std.ArrayList(Value),
    refused: std.ArrayList(Value),
    warnings: std.ArrayList(Value),
    pool_decision: ?kv_pool.Decision,
    fit_out: [4]usize,
    start_world: usize,
    apply: []const u8 = "auto",
    if_version: ?u64 = null,
    reason: []const u8 = "",
    persist: ?bool = null,
    wait_ms: u64 = 2000,
    reverts_version: ?u64 = null,
    schema_version: ?u64 = null,
    ttl: ?u64 = null,
};

fn badField(ctx: Ctx, code: []const u8, msg: []const u8) ?Analysis {
    fail(ctx.conn, ctx.a, 400, code, msg);
    return null;
}

fn oom(ctx: Ctx) ?Analysis {
    fail(ctx.conn, ctx.a, 500, "internal", "admin: out of memory");
    return null;
}

/// A request's `reason`: printable text of at most 200 bytes (it is kept in the history and shown by the page as text).
fn reasonOk(t: []const u8) bool {
    if (t.len > 200) return false;
    for (t) |c| if (c < 32 or c == 127) return false;
    return std.unicode.utf8ValidateSlice(t);
}

/// Reads a request body's shape and checks every change it names against the table, the pool and the caller's scope, answering a
/// malformed request itself (null). `for_set`: the fields only a write takes are allowed.
fn analyze(ctx: Ctx, doc: Value, for_set: bool) ?Analysis {
    const a = ctx.a;
    const conn = ctx.conn;
    _ = conn;
    var out: Analysis = .{ .changes = &.{}, .resolved = .empty, .refused = .empty, .warnings = .empty, .pool_decision = null, .fit_out = undefined, .start_world = 0 };
    if (doc != .object) return badField(ctx, "bad_json", "the body is a JSON object");
    for (doc.object.keys(), doc.object.values()) |k, v| {
        if (std.mem.eql(u8, k, "set")) continue;
        if (std.mem.eql(u8, k, "apply")) {
            if (v != .string) return badField(ctx, "bad_json", "apply: a string");
            if (std.mem.eql(u8, v.string, "auto") or std.mem.eql(u8, v.string, "live") or std.mem.eql(u8, v.string, "next_start")) {
                out.apply = v.string;
                continue;
            }
            if (std.mem.eql(u8, v.string, "idle") or std.mem.eql(u8, v.string, "now"))
                return badField(ctx, "apply_not_supported", "this engine applies a change live (a row with a live hook) or at the next start: apply is auto, live or next_start");
            return badField(ctx, "invalid_apply_mode", "apply: auto, live, idle, now or next_start");
        }
        if (std.mem.eql(u8, k, "ttl_s")) {
            const t = if (v == .int) std.fmt.parseInt(u64, v.int, 10) catch 0 else 0;
            if (t == 0 or t > 31_536_000) return badField(ctx, "bad_json", "ttl_s: a whole number of seconds, at least 1");
            out.ttl = t;
            continue;
        }
        if (!for_set) return badField(ctx, "unknown_field", k);
        if (std.mem.eql(u8, k, "if_version") or std.mem.eql(u8, k, "reverts_version") or std.mem.eql(u8, k, "schema_version") or std.mem.eql(u8, k, "wait_ms")) {
            const n = if (v == .int) std.fmt.parseInt(u64, v.int, 10) catch null else null;
            if (v == .null and !std.mem.eql(u8, k, "wait_ms")) continue;
            const x = n orelse return badField(ctx, "bad_json", std.fmt.allocPrint(a, "{s}: a whole number", .{k}) catch k);
            if (std.mem.eql(u8, k, "if_version")) out.if_version = x else if (std.mem.eql(u8, k, "reverts_version")) out.reverts_version = x else if (std.mem.eql(u8, k, "schema_version")) out.schema_version = x else {
                if (x > 5000) return badField(ctx, "bad_json", "wait_ms: 0 to 5000");
                out.wait_ms = x;
            }
            continue;
        }
        if (std.mem.eql(u8, k, "reason")) {
            if (v != .string or !reasonOk(v.string)) return badField(ctx, "invalid_reason", "reason: printable text of at most 200 bytes");
            out.reason = v.string;
            continue;
        }
        if (std.mem.eql(u8, k, "persist")) {
            if (v != .bool) return badField(ctx, "bad_json", "persist: true or false");
            out.persist = v.bool;
            continue;
        }
        if (std.mem.eql(u8, k, "max_wait_s") or std.mem.eql(u8, k, "confirm"))
            return badField(ctx, "apply_not_supported", "leases, idle waits and now are not built: these fields have no use here");
        return badField(ctx, "unknown_field", k);
    }
    const set = doc.get("set") orelse return badField(ctx, "missing_field", "set");
    if (set != .object) return badField(ctx, "bad_json", "set: an object of option name or id to value");
    if (set.object.count() == 0) return badField(ctx, "missing_field", "set: no change is named");

    // every key's option first: one option named twice (by name and by id) is a malformed request, whatever it holds
    const keys = set.object.keys();
    const vals = set.object.values();
    const what = a.alloc(Named, keys.len) catch return oom(ctx);
    for (keys, what, 0..) |k, *w, i| {
        w.* = named(k);
        if (vals[i] == .array or vals[i] == .object) return badField(ctx, "bad_json", "a value is a string, a number, true, false or null");
        if (w.* == .option) for (what[0..i], 0..) |prev, j| if (prev == .option and prev.option == w.option)
            return badField(ctx, "duplicate_option", std.fmt.allocPrint(a, "{s} and {s} are the same option", .{ keys[j], k }) catch "the same option twice");
    }

    const Look = struct {
        keys: []const []const u8,
        what: []const Named,
        texts: []const ?[]const u8,
        pub fn value(l: @This(), name: []const u8) ?[]const u8 {
            for (l.what, l.texts) |w, t| if (w == .option and std.mem.eql(u8, w.option.name, name)) {
                const x = t orelse continue;
                return if (std.mem.eql(u8, x, "unset")) currentText(w.option) else x;
            };
            const o = settings.byKey(name) orelse return null;
            return currentText(o);
        }
    };
    const texts = a.alloc(?[]const u8, keys.len) catch return oom(ctx);
    for (what, vals, texts) |w, v, *t| t.* = valueText(a, v, w == .option and w.option.kind == .flag);
    const look: Look = .{ .keys = keys, .what = what, .texts = texts };
    var buf: [400]u8 = undefined;
    // the pool: a free target in the request is re-fitted from the numbers the start voted on; the pool named in the same request,
    // or in engine.cfg beside it, is a conflict the start would refuse
    const target_opt = settings.byEnv("TF_DS_KV_FREE_TARGET_GIB").?;
    const window_opt = settings.byEnv("TF_DS_WINDOW").?;
    const share_opt = settings.byEnv("TENSORFOLD_PREFILL_SHARE").?;
    var named_target: ?[]const u8 = null;
    var named_window: ?[]const u8 = null;
    var named_share: ?[]const u8 = null;
    for (what, texts) |w, t| if (w == .option) {
        if (w.option == target_opt) named_target = t;
        if (w.option == window_opt) named_window = t;
        if (w.option == share_opt) named_share = t;
    };
    const eff_target: ?[]const u8 = if (named_target) |t| (if (std.mem.eql(u8, t, "unset")) null else t) else currentText(target_opt);
    const eff_window: ?[]const u8 = if (named_window) |t| (if (std.mem.eql(u8, t, "unset")) null else t) else currentText(window_opt);
    const eff_share: ?[]const u8 = if (named_share) |t| (if (std.mem.eql(u8, t, "unset")) null else t) else currentText(share_opt);
    out.start_world = @min(kv_pool.report.world, 4);
    if (named_target != null and out.start_world > 0 and kv_pool.ready.load(.acquire) and kv_pool.report.table[0].floor != 0) {
        const t = named_target.?;
        const bytes: ?u64 = if (std.mem.eql(u8, t, "unset")) 0 else kv_pool.parseTargetGib(t) catch null;
        if (bytes) |nb| {
            out.pool_decision = kv_pool.refit(kv_pool.report.table[0..out.start_world], nb, out.fit_out[0..out.start_world]);
            for (kv_pool.report.table[0..out.start_world], 0..) |p, i| {
                shifted_parts[i] = p;
                shifted_parts[i].target_bytes = nb;
            }
        }
    }
    const rounds_opt = settings.byEnv("TF_DS_ROUNDS_BESIDE").?;
    var changes: std.ArrayList(Change) = .empty;
    for (keys, what, texts) |key, w, text_opt| {
        const text = text_opt orelse "";
        switch (w) {
            .nothing => out.refused.append(a, refusedRow(a, key, null, text, "unknown_option", std.fmt.allocPrint(a, "{s}: unknown option", .{key}) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx),
            .retired => |r| {
                if (settings.retiredRefusal(r, text, &buf)) |why| {
                    out.refused.append(a, refusedRow(a, r.env, null, text, "retired_option", a.dupe(u8, why) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
                } else out.warnings.append(a, warning(a, "retired_unset", r.env, null) catch return oom(ctx)) catch return oom(ctx);
            },
            .option => |o| {
                const id = o.id;
                if (o.secret) {
                    out.refused.append(a, refusedRow(a, o.name, id, null, "secret_not_settable", std.fmt.allocPrint(a, "{s} is a key: keys are not set through the admin surface", .{o.name}) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
                    continue;
                }
                if (ctx.who.scope == .agent) if (agentCheck(o, text, out.ttl, &buf)) |why| if (!std.mem.eql(u8, why.code, "outside_agent_range") or !std.mem.eql(u8, text, "unset")) {
                    out.refused.append(a, refusedRow(a, o.name, id, text, why.code, a.dupe(u8, why.message) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
                    continue;
                };
                const unset = std.mem.eql(u8, text, "unset");
                if (!unset) if (settings.check(o, text, &buf)) |bad| {
                    // the pool below the context has its own name: the context is never lowered
                    const code: []const u8 = if (o == window_opt and bad.code == .below_minimum) "pool_below_window" else @tagName(bad.code);
                    out.refused.append(a, refusedRow(a, o.name, id, text, code, a.dupe(u8, bad.message) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
                    continue;
                };
                // a pool named and a free target asked: the start refuses the pair, so the dry run does
                if (!unset and ((o == window_opt and eff_target != null) or (o == target_opt and named_window == null and eff_window != null))) {
                    out.refused.append(a, refusedRow(a, o.name, id, text, "pool_named_twice", "TF_DS_KV_FREE_TARGET_GIB sizes the pool from the free memory and TF_DS_WINDOW names it: set one") catch return oom(ctx)) catch return oom(ctx);
                    continue;
                }
                // a count of rounds beside a share below 1 is two controls of the same rounds: the host refuses the pair at its start
                if (!unset and (o == rounds_opt or o == share_opt)) {
                    const share_below_one = if (eff_share) |sh| (std.fmt.parseFloat(f64, sh) catch 1) < 1 else false;
                    const rounds_set = if (named_share != null and o == share_opt) (if (activeRounds(look)) |r| r != 0 else false) else (o == rounds_opt);
                    if (share_below_one and rounds_set) {
                        out.refused.append(a, refusedRow(a, o.name, id, text, "conflicting_options", "TF_DS_ROUNDS_BESIDE and TENSORFOLD_PREFILL_SHARE below 1 are two controls of the same rounds: set one") catch return oom(ctx)) catch return oom(ctx);
                        continue;
                    }
                }
                // the rows the host takes live have rules the host holds to: the order of prompt steps must parse, the step rows are whole
                // resume units within the booted pass rows, staging's share may not pass the service's budget
                if (!unset) if (liveRule(a, o, text, what, texts)) |why| {
                    out.refused.append(a, refusedRow(a, o.name, id, text, why.code, a.dupe(u8, why.message) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
                    continue;
                };
                // a free target the Sparks cannot keep and still hold the context
                if (o == target_opt and !unset) if (out.pool_decision) |dec| if (dec.verdict == .refused) {
                    var tb: [1600]u8 = undefined;
                    out.refused.append(a, refusedRow(a, o.name, id, text, "memory_does_not_fit", a.dupe(u8, kv_pool.refusalText(&tb, shifted_parts[0..out.start_world], dec)) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
                    continue;
                };
                const wired = live_config.isWired(id);
                const live = wired and !std.mem.eql(u8, out.apply, "next_start");
                const row = json.newObject(a) catch return oom(ctx);
                row.put(a, "name", str(o.name)) catch return oom(ctx);
                row.put(a, "id", json.intValue(a, id) catch return oom(ctx)) catch return oom(ctx);
                const from = if (live) currentText(o) else standingText(o); // what the engine runs (a live row), or what waits for the next start
                row.put(a, "from", if (from) |f| str(f) else .null) catch return oom(ctx);
                row.put(a, "to", if (unset) .null else str(text)) catch return oom(ctx);
                row.put(a, "apply_class", str(@tagName(o.apply))) catch return oom(ctx);
                row.put(a, "apply_mode", str(if (live) "live" else "next_start")) catch return oom(ctx);
                row.put(a, "state_would_be", str(if (live) "active" else "pending_restart")) catch return oom(ctx);
                row.put(a, "needs_restart", .{ .bool = !live }) catch return oom(ctx);
                row.put(a, "out_changes", .{ .bool = o.out_changes }) catch return oom(ctx);
                out.resolved.append(a, .{ .object = row }) catch return oom(ctx);
                changes.append(a, .{ .o = o, .text = text, .from = from, .file_from = fileText(o), .live = live }) catch return oom(ctx);
                if (!live) out.warnings.append(a, warning(a, if (o.apply == .H and !wired) "no_live_hook" else "restart_needed", o.name, null) catch return oom(ctx)) catch return oom(ctx);
                if (requiresUnmet(o.requires, look)) |cond| out.warnings.append(a, warning(a, "requires_unmet", o.name, cond) catch return oom(ctx)) catch return oom(ctx);
            },
        }
    }
    out.changes = changes.items;
    return out;
}

/// The parts of the start's vote with a request's free target in them (what `memory_does_not_fit` words against).
var shifted_parts: [4]kv_pool.Part = undefined;

fn activeRounds(look: anytype) ?u32 {
    const t = look.value("tf_ds_rounds_beside") orelse return null;
    return std.fmt.parseInt(u32, t, 10) catch null;
}

/// POST /validate: the dry run. It names what each change would do and whether it would be refused; it applies nothing, calls
/// nothing of the engine and writes nothing.
fn validateReply(ctx: Ctx) void {
    const a = ctx.a;
    const conn = ctx.conn;
    const body = writeBody(ctx) orelse return;
    const parsed = json.parse(a, if (body.len == 0) "{}" else body) catch return fail(conn, a, 500, "internal", "admin: out of memory");
    const doc = switch (parsed) {
        .ok => |v| v,
        .err => return fail(conn, a, 400, "bad_json", "the body is not JSON"),
    };
    const an = analyze(ctx, doc, false) orelse return;
    const root = json.newObject(a) catch return;
    root.put(a, "ok", .{ .bool = an.refused.items.len == 0 }) catch return;
    root.put(a, "resolved", .{ .array = an.resolved.items }) catch return;
    var all_live = an.changes.len > 0;
    for (an.changes) |c| if (!c.live) {
        all_live = false;
    };
    root.put(a, "effective_mode", str(if (all_live) "live" else "next_start")) catch return;
    const modes = a.alloc(Value, 4) catch return;
    inline for (.{
        .{ "live", true, "a row with a live hook is taken by the engine at its next loop turn; every other row is stored for the next start" },
        .{ "idle", false, "this engine has no class I apply" },
        .{ "now", false, "this engine has no class N apply" },
        .{ "next_start", true, "stored in engine.cfg (persist) and applied when the engine is next started with it" },
    }, 0..) |m, i| {
        const o = json.newObject(a) catch return;
        o.put(a, "mode", str(m[0])) catch return;
        o.put(a, "available", .{ .bool = m[1] }) catch return;
        o.put(a, if (m[1]) "cost" else "why", str(m[2])) catch return;
        modes[i] = .{ .object = o };
    }
    root.put(a, "modes", .{ .array = modes }) catch return;
    root.put(a, "warnings", .{ .array = an.warnings.items }) catch return;
    root.put(a, "refused", .{ .array = an.refused.items }) catch return;
    if (an.pool_decision) |dec| if (dec.verdict != .refused) {
        // what the change would leave of the pool and the memory, ESTIMATED from the start's vote (nothing is applied)
        const derived = json.newObject(a) catch return;
        derived.put(a, "kv_pool", derivedPool(a, dec, an.fit_out[0..an.start_world]) catch return) catch return;
        root.put(a, "derived", .{ .object = derived }) catch return;
    };
    root.put(a, "config_version", json.intValue(a, settings_history.version.load(.acquire)) catch return) catch return;
    sendValue(ctx, 200, .{ .object = root });
}

/// An error answer that names what a caller needs to go on: `details` is a JSON value.
fn failDetails(conn: *Conn, a: Allocator, code: u16, name: []const u8, message: []const u8, details: Value, retry_after: ?u32) void {
    const err = json.newObject(a) catch return;
    err.put(a, "code", str(name)) catch return;
    err.put(a, "message", str(message)) catch return;
    err.put(a, "details", details) catch return;
    const root = json.newObject(a) catch return;
    root.put(a, "error", .{ .object = err }) catch return;
    const body = json.stringify(a, .{ .object = root }, .{}) catch return;
    conn.startResponse(code, phrase(code)) catch return;
    conn.addHeader("Content-Type", "application/json") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    conn.addHeader("Cache-Control", "no-store") catch return;
    conn.addHeader("X-Content-Type-Options", "nosniff") catch return;
    conn.addHeader("X-TF-Admin-API", std.fmt.comptimePrint("{d}.{d}", .{ api_major, api_minor })) catch return;
    var ra: [16]u8 = undefined;
    if (retry_after) |r| conn.addHeader("Retry-After", std.fmt.bufPrint(&ra, "{d}", .{r}) catch return) catch return;
    if (conn.close) conn.addHeader("Connection", "close") catch return;
    conn.finish(body) catch {};
}

var request_counter: std.atomic.Value(u64) = .init(0);

fn nowMs(srv: *Server) i64 {
    return @intCast(@divTrunc(std.Io.Clock.real.now(srv.io).toNanoseconds(), std.time.ns_per_ms));
}

/// The row as a write answer shows it.
fn changeValue(a: Allocator, c: Change, state: []const u8, version: u64, persisted: bool) !Value {
    const o = try json.newObject(a);
    try o.put(a, "id", try json.intValue(a, c.o.id));
    try o.put(a, "name", str(c.o.name));
    try o.put(a, "from", if (c.from) |f| str(f) else .null);
    // what engine.cfg held, which differs from what the engine ran after a live change with persist false
    try o.put(a, "from_file", if (c.file_from) |f| str(f) else .null);
    try o.put(a, "to", if (std.mem.eql(u8, c.text, "unset")) .null else str(c.text));
    try o.put(a, "apply_class", str(@tagName(c.o.apply)));
    try o.put(a, "apply_mode", str(if (c.live) "live" else "next_start"));
    try o.put(a, "state", str(state));
    try o.put(a, "since_version", try json.intValue(a, version));
    try o.put(a, "persisted", .{ .bool = persisted });
    try o.put(a, "lease_id", .null);
    try o.put(a, "out_changes", .{ .bool = c.o.out_changes });
    return .{ .object = o };
}

/// An option's value in an engine.cfg's text (its first line), or null when the file has none.
fn fileValueIn(a: Allocator, text: []const u8, o: *const settings.Option) ?[]const u8 {
    const cfg = settings.parseCfg(a, text) catch return null;
    for (cfg.lines) |l| {
        const found = settings.byKey(l.key) orelse continue;
        if (found == o) return if (std.mem.eql(u8, l.value, "unset")) null else l.value;
    }
    return null;
}

/// Whether the engine's value `active` (null: the code decides) is what `wanted` asks (`unset`: the option's own default).
fn sameActive(o: *const settings.Option, active: ?[]const u8, wanted: []const u8) bool {
    if (std.mem.eql(u8, wanted, "unset")) {
        const def: ?[]const u8 = if (o.default.len == 0) null else o.default;
        return if (active) |x| (if (def) |d| std.mem.eql(u8, x, d) else false) else def == null;
    }
    return if (active) |x| std.mem.eql(u8, x, wanted) else false;
}

/// Whether engine.cfg's line for an option (null: none) is what `wanted` asks (`unset`: no line).
fn sameFile(file: ?[]const u8, wanted: []const u8) bool {
    if (std.mem.eql(u8, wanted, "unset")) return file == null;
    return if (file) |x| std.mem.eql(u8, x, wanted) else false;
}

/// The text a row would have to hold for a change to be no change: what the engine runs now, or what is waiting for the next start.
fn standingText(o: *const settings.Option) ?[]const u8 {
    const cur = currentText(o);
    const st = admin_state.stateOf(o, cur);
    if (st.requested) |r| return if (std.mem.eql(u8, r, "unset")) null else r;
    return cur;
}

/// What engine.cfg holds for an option: what this process last wrote, else the start's file value; null: no line.
fn fileText(o: *const settings.Option) ?[]const u8 {
    if (admin_state.get(o).file_text) |t| return if (std.mem.eql(u8, t, "unset")) null else t;
    const res = settings.resolved[settings.indexOf(o)];
    if (res.source == .cfg and !o.secret) return res.value;
    return null;
}

/// Hands a live value to the engine: the rounds beside go into the live block (the host takes them at its next turn), the log of polls
/// is a server flag set here.
fn publishLive(srv: *Server, c: Change, version: u64) void {
    const unset = std.mem.eql(u8, c.text, "unset");
    if (live_config.isFillRow(c.o.id)) return; // the three rows of the fill order are published together, after the loop (publishFillOrder)
    const slot = live_config.slotOf(c.o.id) orelse return;
    const n: u64 = if (unset) live_config.unset else switch (slot) {
        .log_polls => @intFromBool(std.mem.eql(u8, c.text, "1")),
        .goal_mode => @backingInt(goals_mod.modeOf(c.text)),
        .prefill_share => @as(u64, @bitCast(std.fmt.parseFloat(f64, c.text) catch return)),
        .kv_free_target => blk: { // GiB (a decimal) as whole MiB: what a cfg item's u32 carries
            const g = std.fmt.parseFloat(f64, c.text) catch return;
            if (!std.math.isFinite(g) or g <= 0) return;
            break :blk @intFromFloat(@round(g * 1024.0));
        },
        else => std.fmt.parseInt(u64, c.text, 10) catch return,
    };
    if (slot == .log_polls) @atomicStore(bool, &srv.config.log_polls, !unset and n == 1, .monotonic);
    live_config.block.publish(slot, n, version);
}

/// A request's own value for option `id`, if it names one.
fn requestText(id: u16, what: []const Named, texts: []const ?[]const u8) ?[]const u8 {
    for (what, texts) |w, t| if (w == .option and w.option.id == id) return t;
    return null;
}

/// What the host refuses of a live row's value in the company of the others (null: nothing). `texts` are the request's values.
fn liveRule(a: Allocator, o: *const settings.Option, text: []const u8, what: []const Named, texts: []const ?[]const u8) ?Why {
    switch (o.id) {
        119, 120, 121 => {
            const named_o: ??[]const u8 = requestText(120, what, texts);
            const named_q: ??[]const u8 = requestText(121, what, texts);
            const named_w: ??[]const u8 = requestText(119, what, texts);
            // a row named `unset` in the request leaves its text to the engine's own (none)
            const un = struct {
                fn f(n: ??[]const u8) ??[]const u8 {
                    const t = n orelse return null;
                    const v = t orelse return @as(??[]const u8, @as(?[]const u8, null));
                    return if (std.mem.eql(u8, v, "unset")) @as(??[]const u8, @as(?[]const u8, null)) else @as(??[]const u8, v);
                }
            }.f;
            const t = fillTexts(un(named_o), un(named_q), un(named_w));
            const r = fill_order.parse(t.order, t.qos, t.wait);
            if (o.id == 120 and !r.order_ok) return .{ .code = "invalid_value", .message = std.fmt.allocPrint(a, "{s}: not serial or turns", .{text}) catch "not serial or turns" };
            if (o.id == 121 and !r.qos_ok) return .{ .code = "invalid_value", .message = std.fmt.allocPrint(a, "{s}: refused ({s}); a table is size:class:overtakes[:max_wait_s],... smallest first, sizes ascending, classes not falling", .{ text, r.qos_why }) catch "not a table" };
            if (o.id == 119 and !r.wait_ok) return .{ .code = "invalid_value", .message = std.fmt.allocPrint(a, "{s}: not whole seconds from 1 to 86400", .{text}) catch "not whole seconds" };
        },
        // the graph cap travels as a u32 in a cfg frame, the top value being the marker for `unset`
        105 => {
            const v = std.fmt.parseInt(u64, text, 10) catch return null; // (`unset`, or text the kind's own check refuses)
            if (v >= std.math.maxInt(u32)) return .{ .code = "invalid_value", .message = std.fmt.allocPrint(a, "{s}: above 4,294,967,294 graphs", .{text}) catch "too many" };
        },
        106 => {
            const unit: u64 = if (o.min) |m| @intFromFloat(m) else 1;
            const v = std.fmt.parseInt(u64, text, 10) catch return null;
            if (unit > 0 and v % unit != 0) return .{ .code = "invalid_value", .message = std.fmt.allocPrint(a, "{s}: a whole number of {d}-row resume units", .{ text, unit }) catch "not whole resume units" };
            const pass_text = currentText(settings.byId(110).?) orelse "2048";
            const pass = std.fmt.parseInt(u64, pass_text, 10) catch 2048;
            if (v > pass) return .{ .code = "above_booted_pass_rows", .message = std.fmt.allocPrint(a, "{s}: above the booted pass rows ({d}, TF_DS_PASS_ROWS): the passes' buffers are sized by them", .{ text, pass }) catch "above the booted pass rows" };
        },
        450, 453 => {
            const pump_o = settings.byId(453).?;
            const load_o = settings.byId(450).?;
            const pump_t = requestText(453, what, texts) orelse currentText(pump_o);
            const load_t = requestText(450, what, texts) orelse currentText(load_o);
            const pump = std.fmt.parseInt(u64, if (pump_t) |t| (if (std.mem.eql(u8, t, "unset")) pump_o.default else t) else pump_o.default, 10) catch return null;
            const load = std.fmt.parseInt(u64, if (load_t) |t| (if (std.mem.eql(u8, t, "unset")) load_o.default else t) else load_o.default, 10) catch return null;
            if (pump > load) return .{ .code = "conflicting_options", .message = std.fmt.allocPrint(a, "TF_SPILL_PUMP_MS={d} is more than TF_SPILL_LOAD_MS={d}: staging's share of a service cannot pass the service", .{ pump, load }) catch "pump above load" };
        },
        else => {},
    }
    return null;
}

/// The three fill-order rows' effective texts, `named` rows replaced by a request's values, for fill_order.parse.
const FillTexts = struct { order: ?[]const u8, qos: ?[]const u8, wait: ?[]const u8 };

fn fillTexts(named_order: ??[]const u8, named_qos: ??[]const u8, named_wait: ??[]const u8) FillTexts {
    const order_o = settings.byId(120).?;
    const qos_o = settings.byId(121).?;
    const wait_o = settings.byId(119).?;
    return .{
        .order = if (named_order) |t| t else currentText(order_o),
        .qos = if (named_qos) |t| t else currentText(qos_o),
        .wait = if (named_wait) |t| t else currentText(wait_o),
    };
}

/// Publishes the prompt-step order a request's fill rows (and the standing ones) make, or the start's when none stands.
fn publishFillOrder(changes: []const Change, version: u64) void {
    var no: ??[]const u8 = null;
    var nq: ??[]const u8 = null;
    var nw: ??[]const u8 = null;
    var any = false;
    for (changes) |c| {
        if (!live_config.isFillRow(c.o.id)) continue;
        any = true;
        const t: ?[]const u8 = if (std.mem.eql(u8, c.text, "unset")) null else c.text;
        switch (c.o.id) {
            120 => no = t,
            121 => nq = t,
            else => nw = t,
        }
    }
    if (!any) return;
    const t = fillTexts(no, nq, nw);
    if (t.order == null and t.qos == null and t.wait == null) return live_config.block.publishFill(null, version);
    const r = fill_order.parse(t.order, t.qos, t.wait);
    if (r.order_ok and r.qos_ok and r.wait_ok) live_config.block.publishFill(r.policy, version);
}

fn readJsonBody(ctx: Ctx) ?Value {
    const body = writeBody(ctx) orelse return null;
    const parsed = json.parse(ctx.a, if (body.len == 0) "{}" else body) catch {
        fail(ctx.conn, ctx.a, 500, "internal", "admin: out of memory");
        return null;
    };
    return switch (parsed) {
        .ok => |v| v,
        .err => {
            fail(ctx.conn, ctx.a, 400, "bad_json", "the body is not JSON");
            return null;
        },
    };
}

/// POST /set: the changes, all or none. A row with a live hook is handed to the engine; every row is stored in engine.cfg unless
/// `persist` is false; the history row is on disk before the answer. Nothing here is the engine thread's: a live value goes into
/// the live block and the host takes it at its next loop turn.
fn setReply(ctx: Ctx) void {
    const a = ctx.a;
    const conn = ctx.conn;
    const doc = readJsonBody(ctx) orelse return;
    const an = analyze(ctx, doc, true) orelse return;
    const srv = ctx.srv;
    const now_ms = nowMs(srv);
    if (admin_state.rateAllow(ctx.who.label, if (ctx.who.scope == .admin) 30 else 6, @divTrunc(now_ms, 1000))) |retry| {
        const d = json.newObject(a) catch return;
        d.put(a, "retry_after_s", json.intValue(a, retry) catch return) catch return;
        return failDetails(conn, a, 429, "rate_limited", "too many writes a minute for this key", .{ .object = d }, retry);
    }
    if (an.refused.items.len > 0) {
        const d = json.newObject(a) catch return;
        d.put(a, "refused", .{ .array = an.refused.items }) catch return;
        const only_agent = for (an.refused.items) |x| {
            if (!std.mem.eql(u8, json.strOr(x.get("code")), "not_permitted_for_agent")) break false;
        } else true;
        return failDetails(conn, a, if (only_agent) 403 else 422, if (only_agent) "not_permitted_for_agent" else "validation_failed", "no change was made: the request has refused rows", .{ .object = d }, null);
    }
    if (an.schema_version) |sv| if (sv != settings.schema_version) return fail(conn, a, 409, "schema_changed", "the schema changed: read it again");
    const persist = an.persist orelse true;
    var any_file = persist;
    // engine.cfg as it is now: what a change is judged against, not what this process started with
    const cfg_now: ?[]const u8 = if (settings.report.path.len > 0) std.Io.Dir.cwd().readFileAlloc(srv.io, settings.report.path, a, .limited(1 << 20)) catch null else null;
    const changes = a.dupe(Change, an.changes) catch return fail(conn, a, 500, "internal", "admin: out of memory");
    if (cfg_now) |text| for (changes) |*c| {
        c.file_from = fileValueIn(a, text, c.o);
    };
    for (changes) |c| if (!c.live) {
        any_file = true;
    };
    if (!persist) for (changes) |c| if (!c.live) return fail(conn, a, 400, "invalid_apply_mode", "a change for the next start is kept by persisting it: persist is true for these rows");
    admin_state.lockWrites();
    defer admin_state.unlockWrites();
    const before = settings_history.version.load(.acquire);
    if (an.if_version) |iv| if (iv != before) {
        const d = json.newObject(a) catch return;
        d.put(a, "config_version", json.intValue(a, before) catch return) catch return;
        var cur: std.ArrayList(Value) = .empty;
        for (changes) |c| {
            const r = json.newObject(a) catch return;
            r.put(a, "name", str(c.o.name)) catch return;
            const st = admin_state.stateOf(c.o, currentText(c.o));
            const have = currentText(c.o);
            r.put(a, "value", if (have) |h| str(h) else .null) catch return;
            r.put(a, "requested", if (st.requested) |q| str(q) else .null) catch return;
            r.put(a, "state", str(@tagName(st.state))) catch return;
            r.put(a, "since_version", json.intValue(a, admin_state.get(c.o).since) catch return) catch return;
            cur.append(a, .{ .object = r }) catch return;
        }
        d.put(a, "current", .{ .array = cur.items }) catch return;
        return failDetails(conn, a, 409, "version_conflict", "the settings changed since this request was drawn up", .{ .object = d }, null);
    };
    // the same request twice is no change: a live row's change is judged against what the engine RUNS (and against engine.cfg only when this
    // request keeps it there), a row for the next start against what engine.cfg holds; the two differ after a live change with persist false
    var all_same = true;
    for (changes) |c| {
        const same = if (c.live) sameActive(c.o, currentText(c.o), c.text) and (!persist or sameFile(c.file_from, c.text)) else sameFile(c.file_from, c.text);
        if (!same) all_same = false;
    }
    if (all_same) {
        const root = json.newObject(a) catch return;
        root.put(a, "config_version", json.intValue(a, before) catch return) catch return;
        root.put(a, "previous_version", json.intValue(a, before) catch return) catch return;
        root.put(a, "active_version", json.intValue(a, admin_state.activeVersion(before)) catch return) catch return;
        root.put(a, "noop", .{ .bool = true }) catch return;
        root.put(a, "outcome", str("applied")) catch return;
        root.put(a, "changes", .{ .array = a.alloc(Value, 0) catch return }) catch return;
        root.put(a, "warnings", .{ .array = an.warnings.items }) catch return;
        return sendValue(ctx, 200, .{ .object = root });
    }
    const cfg_path = settings.report.path;
    if (any_file and cfg_path.len == 0) return fail(conn, a, 422, "no_engine_cfg", "no engine.cfg was read at the start: persist and next_start need one (TF_ENGINE_CFG); a live row may be set with persist false");
    const version = before + 1;
    // the file: the old text with these rows changed
    var new_sha: ?[32]u8 = null;
    var old_text: ?[]const u8 = null;
    var wrote = false;
    if (any_file) {
        const old = std.Io.Dir.cwd().readFileAlloc(srv.io, cfg_path, a, .limited(1 << 20)) catch |e| return fail(conn, a, 500, "internal", std.fmt.allocPrint(a, "engine.cfg: cannot read it ({s}): nothing was changed", .{@errorName(e)}) catch "engine.cfg: cannot read it");
        old_text = old;
        const edits = a.alloc(cfg_write.Edit, changes.len) catch return;
        for (changes, edits) |c, *e| e.* = .{ .o = c.o, .value = c.text };
        const text = cfg_write.rewrite(a, old, edits, settings.schema_version) catch return fail(conn, a, 500, "internal", "admin: out of memory");
        cfg_write.writeAtomic(a, cfg_path, text) catch |e| return fail(conn, a, 500, "internal", std.fmt.allocPrint(a, "engine.cfg: cannot write it ({s}): nothing was changed", .{@errorName(e)}) catch "engine.cfg: cannot write it");
        wrote = true;
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(text, &h, .{});
        new_sha = h;
    }
    // the history row, before anything is applied or answered
    const hist_changes = a.alloc(settings_history.WriteChange, changes.len) catch return;
    for (changes, hist_changes) |c, *hc| hc.* = .{ .id = c.o.id, .from = if (persist) c.file_from else c.from, .from_file = c.file_from, .to = if (std.mem.eql(u8, c.text, "unset")) null else c.text, .persisted = persist, .state_after = if (c.live) "active" else "pending_restart" };
    var hex_buf: [64]u8 = undefined;
    const hex: ?[]const u8 = if (new_sha) |h| blk: {
        hex_buf = std.fmt.bytesToHex(h, .lower);
        break :blk &hex_buf;
    } else null;
    const req_id = std.fmt.allocPrint(a, "r-{d}", .{request_counter.fetchAdd(1, .monotonic) + 1}) catch return;
    var mode_live = false;
    var mode_next = false;
    for (changes) |c| if (c.live) {
        mode_live = true;
    } else {
        mode_next = true;
    };
    const wrote_row = settings_history.appendWrite(a, srv.io, settings_history.path, .{
        .at_ms = now_ms,
        .label = ctx.who.label,
        .scope = @tagName(ctx.who.scope),
        .peer = conn.peer,
        .request_id = req_id,
        .endpoint = if (an.reverts_version != null) "revert" else "set",
        .apply = if (mode_live and mode_next) "mixed" else if (mode_live) "live" else "next_start",
        .outcome = "applied",
        .reason = an.reason,
        .release = admin_state.release,
        .cfg_sha256 = hex,
        .changes = hist_changes,
        .reverts_version = an.reverts_version,
    }) catch |e| {
        // the file is put back: a change that is not in the history is not made
        if (wrote) if (old_text) |ot| cfg_write.writeAtomic(a, cfg_path, ot) catch {};
        return fail(conn, a, 500, "internal", std.fmt.allocPrint(a, "the history cannot be written ({s}): nothing was changed", .{@errorName(e)}) catch "the history cannot be written");
    };
    _ = wrote_row;
    if (new_sha) |h| {
        settings.report.sha256 = h;
    }
    // apply: a live value goes to the engine, every row is remembered
    var any_rounds = false;
    for (changes) |c| {
        // a live value the engine already runs is not published again (it would wait for a turn to be taken for nothing)
        const runs = c.live and sameActive(c.o, currentText(c.o), c.text);
        if (c.live and !runs) {
            publishLive(srv, c, version);
            if (live_config.slotOf(c.o.id) != .log_polls) any_rounds = true; // a row the host takes: the answer waits for its turn
        }
        admin_state.record(c.o, if (c.live and !runs) c.text else null, if (persist) c.text else null, ctx.who.label, version);
    }
    publishFillOrder(changes, version);
    for (changes) |c| if (live_config.isFillRow(c.o.id) and c.live) {
        any_rounds = true;
    };
    if (any_rounds and an.wait_ms > 0) {
        const t0 = clock.nowNs(srv.io);
        while (live_config.block.active_version.load(.acquire) < version and clock.nowNs(srv.io) - t0 < @as(i96, @intCast(an.wait_ms)) * std.time.ns_per_ms) std.Io.sleep(srv.io, .fromMilliseconds(5), .awake) catch {};
    }
    const rows = a.alloc(Value, changes.len) catch return;
    var pending = false;
    for (changes, rows) |c, *slot| {
        const st = admin_state.stateOf(c.o, currentText(c.o));
        if (st.state == .pending_step) pending = true;
        slot.* = changeValue(a, c, @tagName(st.state), version, persist) catch return;
    }
    const root = json.newObject(a) catch return;
    root.put(a, "config_version", json.intValue(a, version) catch return) catch return;
    root.put(a, "previous_version", json.intValue(a, before) catch return) catch return;
    root.put(a, "active_version", json.intValue(a, admin_state.activeVersion(version)) catch return) catch return;
    root.put(a, "noop", .{ .bool = false }) catch return;
    root.put(a, "outcome", str(if (pending) "pending" else "applied")) catch return;
    root.put(a, "effective_mode", str(if (mode_live and !mode_next) "live" else "next_start")) catch return;
    root.put(a, "changes", .{ .array = rows }) catch return;
    var warns = an.warnings;
    if (settings_history.path.len == 0) warns.append(a, warningText(a, "history_in_memory", "history", "no engine.cfg was read at the start: the history of this change is kept in memory and is lost when the engine stops") catch return) catch return;
    if (persist) warns.append(a, warningText(a, "other_ranks_take_it_at_next_start", "engine.cfg", "the other ranks take it at the next start (start.sh copies the file)") catch return) catch return;
    root.put(a, "warnings", .{ .array = warns.items }) catch return;
    sendValue(ctx, 200, .{ .object = root });
}

/// POST /revert: the plan that undoes a history row, or returns to a version. It applies nothing: the page shows it, then sends it to
/// `set` with `reverts_version`.
fn revertReply(ctx: Ctx) void {
    const a = ctx.a;
    const conn = ctx.conn;
    const doc = readJsonBody(ctx) orelse return;
    if (doc != .object) return fail(conn, a, 400, "bad_json", "the body is a JSON object");
    var to_version: ?u64 = null;
    var undo: ?u64 = null;
    for (doc.object.keys(), doc.object.values()) |k, v| {
        const n = if (v == .int) std.fmt.parseInt(u64, v.int, 10) catch null else null;
        if (std.mem.eql(u8, k, "to_version")) to_version = n orelse return fail(conn, a, 400, "bad_json", "to_version: a whole number") else if (std.mem.eql(u8, k, "history_version")) undo = n orelse return fail(conn, a, 400, "bad_json", "history_version: a whole number") else return fail(conn, a, 400, "unknown_field", k);
    }
    if ((to_version == null) == (undo == null)) return fail(conn, a, 400, "missing_field", "name one of to_version or history_version");
    const loaded: settings_history.Loaded = if (settings_history.path.len == 0) .{ .lines = settings_history.memoryLines(a) catch return fail(conn, a, 500, "internal", "admin: out of memory") } else settings_history.load(a, ctx.srv.io, settings_history.path) catch return fail(conn, a, 500, "internal", "the history cannot be read");
    const Replay = struct { version: u64 = 0, changes: []const struct { id: u32 = 0, from: ?[]const u8 = null, to: ?[]const u8 = null, persisted: bool = true } = &.{} };
    var rows: std.ArrayList(Replay) = .empty;
    for (loaded.lines) |l| {
        const r = std.json.parseFromSliceLeaky(Replay, a, l, .{ .ignore_unknown_fields = true }) catch continue;
        rows.append(a, r) catch return;
    }
    // the plan: option id -> the text to set
    var plan: std.ArrayList(struct { id: u32, text: []const u8 }) = .empty;
    if (undo) |uv| {
        const row = for (rows.items) |r| {
            if (r.version == uv) break r;
        } else return fail(conn, a, 404, "unknown_version", "no history row has that version");
        for (row.changes) |c| plan.append(a, .{ .id = c.id, .text = c.from orelse "unset" }) catch return;
    } else {
        const tv = to_version.?;
        // the state after `tv`, against the state now: every option whose persisted value differs
        var then: std.AutoArrayHashMapUnmanaged(u32, []const u8) = .empty;
        var now: std.AutoArrayHashMapUnmanaged(u32, []const u8) = .empty;
        for (rows.items) |r| for (r.changes) |c| {
            if (!c.persisted) continue;
            if (r.version <= tv) {
                if (c.to) |t| then.put(a, c.id, t) catch return else _ = then.swapRemove(c.id);
            }
            if (c.to) |t| now.put(a, c.id, t) catch return else _ = now.swapRemove(c.id);
        };
        for (now.keys(), now.values()) |id, v| {
            const was = then.get(id);
            if (was == null) plan.append(a, .{ .id = id, .text = "unset" }) catch return else if (!std.mem.eql(u8, was.?, v)) plan.append(a, .{ .id = id, .text = was.? }) catch return;
        }
        for (then.keys(), then.values()) |id, v| if (!now.contains(id)) plan.append(a, .{ .id = id, .text = v }) catch return;
    }
    // the plan through the same checks `set` applies: a row that would be refused now is skipped with its reason
    var body: std.ArrayList(u8) = .empty;
    body.appendSlice(a, "{\"set\": {") catch return;
    var n: usize = 0;
    var skipped: std.ArrayList(Value) = .empty;
    var set_obj = json.newObject(a) catch return;
    for (plan.items) |p| {
        const o = settings.byId(p.id);
        if (o == null or o.?.hidden or o.?.secret) {
            const sk = json.newObject(a) catch return;
            sk.put(a, "name", str(if (o) |x| x.name else std.fmt.allocPrint(a, "{d}", .{p.id}) catch "")) catch return;
            sk.put(a, "reason", str("not_settable")) catch return;
            skipped.append(a, .{ .object = sk }) catch return;
            continue;
        }
        if (!std.mem.eql(u8, p.text, "unset")) if (settings.check(o.?, p.text, &revert_buf)) |_| {
            const sk = json.newObject(a) catch return;
            sk.put(a, "name", str(o.?.name)) catch return;
            sk.put(a, "reason", str("out_of_range")) catch return;
            skipped.append(a, .{ .object = sk }) catch return;
            continue;
        };
        set_obj.put(a, o.?.name, if (std.mem.eql(u8, p.text, "unset")) .null else str(p.text)) catch return;
        n += 1;
    }
    _ = &body;
    const planv = json.newObject(a) catch return;
    planv.put(a, "set", .{ .object = set_obj }) catch return;
    planv.put(a, "reverts_version", json.intValue(a, undo orelse to_version.?) catch return) catch return;
    const root = json.newObject(a) catch return;
    root.put(a, "plan", .{ .object = planv }) catch return;
    root.put(a, "skipped", .{ .array = skipped.items }) catch return;
    sendValue(ctx, 200, .{ .object = root });
}

var revert_buf: [400]u8 = undefined;

/// The pool in the start's numbers at a position count, from the line through its bytes at one step and at seventeen.
fn poolBytesAt(p: kv_pool.Part, positions: usize) f64 {
    if (p.b_hi <= p.b_lo) return 0;
    const slope = @as(f64, @floatFromInt(p.b_hi - p.b_lo)) / @as(f64, 16 * kv_pool.pool_step);
    return @as(f64, @floatFromInt(p.b_lo)) + slope * (@as(f64, @floatFromInt(positions)) - @as(f64, kv_pool.pool_step));
}

fn derivedPool(a: Allocator, dec: kv_pool.Decision, fits: []const usize) !Value {
    const rep = &kv_pool.report;
    const before = rep.pool_active;
    const after = dec.pool;
    const o = try json.newObject(a);
    const rows = try a.alloc(Value, fits.len);
    var delta_gib: f64 = 0;
    for (rep.table[0..fits.len], fits, rows, 0..) |p, f, *slot, r| {
        const x = try json.newObject(a);
        try x.put(a, "rank", try json.intValue(a, r));
        try x.put(a, "tokens_before", try json.intValue(a, before));
        try x.put(a, "tokens_after", try json.intValue(a, if (dec.verdict == .off) after else f));
        const left = @as(f64, @floatFromInt(p.free_bytes)) - poolBytesAt(p, after) - @as(f64, @floatFromInt(p.growth_bytes));
        try x.put(a, "free_after_gib", str(try std.fmt.allocPrint(a, "{d:.2}", .{left / gib_f})));
        slot.* = .{ .object = x };
        if (r == 0) delta_gib = (poolBytesAt(p, after) - poolBytesAt(p, before)) / gib_f;
    }
    try o.put(a, "ranks", .{ .array = rows });
    try o.put(a, "tokens_before", try json.intValue(a, before));
    try o.put(a, "tokens_after", try json.intValue(a, after));
    try o.put(a, "per_node_gib_delta", str(try std.fmt.allocPrint(a, "{d:.2}", .{delta_gib})));
    try o.put(a, "label", str("estimated"));
    return .{ .object = o };
}

fn warningText(a: Allocator, code: []const u8, name: []const u8, message: []const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "code", str(code));
    try o.put(a, "name", str(name));
    try o.put(a, "message", str(message));
    return .{ .object = o };
}

fn warning(a: Allocator, code: []const u8, name: []const u8, requires: ?[]const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "code", str(code));
    try o.put(a, "name", str(name));
    if (requires) |r| try o.put(a, "requires", str(r));
    return .{ .object = o };
}

/// An option's value as the engine runs it now: a live change the engine has taken, else the start's value; null when the code decides.
fn currentText(o: *const settings.Option) ?[]const u8 {
    if (admin_state.liveValue(o)) |t| return if (std.mem.eql(u8, t, "unset")) (if (o.default.len == 0) null else o.default) else t;
    return activeText(o);
}

/// An option's value as the engine started with it: the value start-up recorded, or its default; null when the code decides.
fn activeText(o: *const settings.Option) ?[]const u8 {
    const res = settings.resolved[settings.indexOf(o)];
    if (res.source != .default and !o.secret) return res.value;
    return if (o.default.len == 0) null else o.default;
}

/// RFC 3339 UTC time of a unix time in milliseconds.
fn rfc3339(a: Allocator, ms: i64) ![]const u8 {
    const secs: u64 = @intCast(@max(0, @divTrunc(ms, 1000)));
    const es: std.time.epoch.EpochSeconds = .{ .secs = secs };
    const day = es.getEpochDay().calculateYearDay();
    const md = day.calculateMonthDay();
    const t = es.getDaySeconds();
    return std.fmt.allocPrint(a, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}Z", .{
        day.year,                           @backingInt(md.month),  md.day_index + 1,
        t.getHoursIntoDay(),                t.getMinutesIntoHour(), t.getSecondsIntoMinute(),
        @as(u64, @intCast(@mod(ms, 1000))),
    });
}

test "a unix time reads as RFC 3339" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("2026-11-08T21:50:07.042Z", try rfc3339(arena.allocator(), 1_794_174_607_042));
    try std.testing.expectEqualStrings("1970-01-01T00:00:00.000Z", try rfc3339(arena.allocator(), 0));
}

test "a query value is percent-decoded and exact" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("a b&c", queryValue(a, "/x?name=a+b%26c&n=1", "name").?);
    try std.testing.expectEqualStrings("1", queryValue(a, "/x?name=a&n=1", "n").?);
    try std.testing.expect(queryValue(a, "/x?name=a", "nam") == null);
    try std.testing.expect(queryValue(a, "/x", "name") == null);
}
