//! The admin surface: `/admin/api/v1/*` and the page at `/admin/ui`: the schema, the active values, the status, the ledger, the
//! requests, the series, the logs, the keys, the history; a dry run, `validate`; and the writes, `set` and `revert`. A write names
//! rows of class H or R: a row with a live hook (live_config.zig) is handed to the engine, which takes it at its next loop turn;
//! every row is kept in engine.cfg unless the caller says not, and the history row is on disk before the answer.
const kv_pool = api.kv_pool;
const live_config = api.live_config;
const cfg_write = api.cfg_write;
const admin_state = @import("admin_state.zig");
const json = @import("json");
    .{ .name = "set", .method = .post, .need = .agent, .step = 5 },
    .{ .name = "revert", .method = .post, .need = .admin, .step = 5 },
    if (std.mem.eql(u8, name, "set")) return setReply(ctx);
    if (std.mem.eql(u8, name, "revert")) return revertReply(ctx);
        // whether the engine takes a change to this row while it runs (a hook in rank 0's host or on a server thread)
    const version_now = settings_history.version.load(.acquire);
    try root.put(a, "config_version", try json.intValue(a, version_now));
    try root.put(a, "active_version", try json.intValue(a, admin_state.activeVersion(version_now)));
        // a live change the engine has taken is the value it runs; otherwise the start's
        const live = admin_state.liveValue(o);
        try r.put(a, "source", str(if (live != null) "admin" else @tagName(res.source)));
        if (live) |t| {
            try r.put(a, "value", if (std.mem.eql(u8, t, "unset")) .null else str(t));
        } else try r.put(a, "value", if (res.source == .default) .null else if (o.secret) str("(set)") else str(res.value));
        const st = admin_state.stateOf(o, currentText(o));
        const mine = admin_state.get(o);
        try r.put(a, "requested", if (st.requested) |t| str(t) else .null);
        try r.put(a, "state", str(@tagName(st.state)));
        try r.put(a, "set_by", if (mine.set_by.len > 0) str(mine.set_by) else .null);
        try r.put(a, "since_version", if (mine.since > 0) try json.intValue(a, mine.since) else .null);
    // what this key may do in this build: an admin key may write the rows of class H and A (live where the engine has a hook, A through a cfg frame) and R (engine.cfg)
    const writes = scope == .admin;
    try root.put(a, "read_only", .{ .bool = !writes });
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
    const version_now = settings_history.version.load(.acquire);
    const version = try json.intValue(a, version_now);
    try cv.put(a, "active_version", try json.intValue(a, admin_state.activeVersion(version_now)));
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
    try putPoolStatus(a, root);
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
    if (!kv_pool.ready.load(.acquire)) {
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
/// status.kv_pool and status.memory: the pool as the start sized it and each Spark's memory at the start. Both are left out
/// until a start has recorded them: a group with no source is never zero-filled.
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
    _ = conn;
    var out: Analysis = .{ .changes = &.{}, .resolved = .empty, .refused = .empty, .warnings = .empty, .pool_decision = null, .fit_out = undefined, .start_world = 0 };
    if (doc != .object) return badField(ctx, "bad_json", "the body is a JSON object");
            if (v != .string) return badField(ctx, "bad_json", "apply: a string");
            if (std.mem.eql(u8, v.string, "auto") or std.mem.eql(u8, v.string, "live") or std.mem.eql(u8, v.string, "next_start")) {
                out.apply = v.string;
                continue;
            }
            if (std.mem.eql(u8, v.string, "idle") or std.mem.eql(u8, v.string, "now"))
                return badField(ctx, "apply_not_supported", "this engine applies a change live (a row with a live hook) or at the next start: apply is auto, live or next_start");
            return badField(ctx, "invalid_apply_mode", "apply: auto, live, idle, now or next_start");
            if (t == 0 or t > 31_536_000) return badField(ctx, "bad_json", "ttl_s: a whole number of seconds, at least 1");
            out.ttl = t;
        if (!for_set) return badField(ctx, "unknown_field", k);
        if (std.mem.eql(u8, k, "if_version") or std.mem.eql(u8, k, "reverts_version") or std.mem.eql(u8, k, "schema_version") or std.mem.eql(u8, k, "wait_ms")) {
            const n = if (v == .int) std.fmt.parseInt(u64, v.int, 10) catch null else null;
            if (v == .null and !std.mem.eql(u8, k, "wait_ms")) continue;
            const x = n orelse return badField(ctx, "bad_json", std.fmt.allocPrint(a, "{s}: a whole number", .{k}) catch k);
            if (std.mem.eql(u8, k, "if_version")) out.if_version = x else if (std.mem.eql(u8, k, "reverts_version")) out.reverts_version = x else if (std.mem.eql(u8, k, "schema_version")) out.schema_version = x else {
                if (x > 5000) return badField(ctx, "bad_json", "wait_ms: 0 to 5000");
                out.wait_ms = x;
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
    const what = a.alloc(Named, keys.len) catch return oom(ctx);
        if (vals[i] == .array or vals[i] == .object) return badField(ctx, "bad_json", "a value is a string, a number, true, false or null");
            return badField(ctx, "duplicate_option", std.fmt.allocPrint(a, "{s} and {s} are the same option", .{ keys[j], k }) catch "the same option twice");
                return if (std.mem.eql(u8, x, "unset")) currentText(w.option) else x;
            return currentText(o);
    const texts = a.alloc(?[]const u8, keys.len) catch return oom(ctx);
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
            .nothing => out.refused.append(a, refusedRow(a, key, null, text, "unknown_option", std.fmt.allocPrint(a, "{s}: unknown option", .{key}) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx),
                    out.refused.append(a, refusedRow(a, r.env, null, text, "retired_option", a.dupe(u8, why) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
                } else out.warnings.append(a, warning(a, "retired_unset", r.env, null) catch return oom(ctx)) catch return oom(ctx);
                    out.refused.append(a, refusedRow(a, o.name, id, null, "secret_not_settable", std.fmt.allocPrint(a, "{s} is a key: keys are not set through the admin surface", .{o.name}) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
                if (ctx.who.scope == .agent) if (agentCheck(o, text, out.ttl, &buf)) |why| if (!std.mem.eql(u8, why.code, "outside_agent_range") or !std.mem.eql(u8, text, "unset")) {
                    out.refused.append(a, refusedRow(a, o.name, id, text, why.code, a.dupe(u8, why.message) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
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
                // a free target the Sparks cannot keep and still hold the context
                if (o == target_opt and !unset) if (out.pool_decision) |dec| if (dec.verdict == .refused) {
                    var tb: [1600]u8 = undefined;
                    out.refused.append(a, refusedRow(a, o.name, id, text, "memory_does_not_fit", a.dupe(u8, kv_pool.refusalText(&tb, shifted_parts[0..out.start_world], dec)) catch return oom(ctx)) catch return oom(ctx)) catch return oom(ctx);
                const live = wired and !std.mem.eql(u8, out.apply, "next_start");
                const row = json.newObject(a) catch return oom(ctx);
                row.put(a, "name", str(o.name)) catch return oom(ctx);
                row.put(a, "id", json.intValue(a, id) catch return oom(ctx)) catch return oom(ctx);
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
    root.put(a, "ok", .{ .bool = an.refused.items.len == 0 }) catch return;
    root.put(a, "resolved", .{ .array = an.resolved.items }) catch return;
    var all_live = an.changes.len > 0;
    for (an.changes) |c| if (!c.live) {
        all_live = false;
    };
    root.put(a, "effective_mode", str(if (all_live) "live" else "next_start")) catch return;
        .{ "live", true, "a row with a live hook is taken by the engine at its next loop turn; every other row is stored for the next start" },
        .{ "idle", false, "this engine has no class I apply" },
        .{ "now", false, "this engine has no class N apply" },
        .{ "next_start", true, "stored in engine.cfg (persist) and applied when the engine is next started with it" },
    root.put(a, "warnings", .{ .array = an.warnings.items }) catch return;
    root.put(a, "refused", .{ .array = an.refused.items }) catch return;
    if (an.pool_decision) |dec| if (dec.verdict != .refused) {
        // what the change would leave of the pool and the memory, ESTIMATED from the start's vote (nothing is applied)
        const derived = json.newObject(a) catch return;
        derived.put(a, "kv_pool", derivedPool(a, dec, an.fit_out[0..an.start_world]) catch return) catch return;
        root.put(a, "derived", .{ .object = derived }) catch return;
    };
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
    const slot = live_config.slotOf(c.o.id) orelse return;
        .kv_free_target => blk: { // GiB (a decimal) as whole MiB: what a cfg item's u32 carries
            const g = std.fmt.parseFloat(f64, c.text) catch return;
            if (!std.math.isFinite(g) or g <= 0) return;
            break :blk @intFromFloat(@round(g * 1024.0));
        },
        },
        },
    }
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
        any_file = true;
    };
    admin_state.lockWrites();
    defer admin_state.unlockWrites();
    const before = settings_history.version.load(.acquire);
    if (an.if_version) |iv| if (iv != before) {
        const d = json.newObject(a) catch return;
        d.put(a, "config_version", json.intValue(a, before) catch return) catch return;
        var cur: std.ArrayList(Value) = .empty;
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
    var all_same = true;
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
        const text = cfg_write.rewrite(a, old, edits, settings.schema_version) catch return fail(conn, a, 500, "internal", "admin: out of memory");
        cfg_write.writeAtomic(a, cfg_path, text) catch |e| return fail(conn, a, 500, "internal", std.fmt.allocPrint(a, "engine.cfg: cannot write it ({s}): nothing was changed", .{@errorName(e)}) catch "engine.cfg: cannot write it");
        wrote = true;
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(text, &h, .{});
        new_sha = h;
    }
    // the history row, before anything is applied or answered
    var hex_buf: [64]u8 = undefined;
    const hex: ?[]const u8 = if (new_sha) |h| blk: {
        hex_buf = std.fmt.bytesToHex(h, .lower);
        break :blk &hex_buf;
    } else null;
    const req_id = std.fmt.allocPrint(a, "r-{d}", .{request_counter.fetchAdd(1, .monotonic) + 1}) catch return;
    var mode_live = false;
    var mode_next = false;
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
            publishLive(srv, c, version);
        }
    }
    if (any_rounds and an.wait_ms > 0) {
        const t0 = clock.nowNs(srv.io);
        while (live_config.block.active_version.load(.acquire) < version and clock.nowNs(srv.io) - t0 < @as(i96, @intCast(an.wait_ms)) * std.time.ns_per_ms) std.Io.sleep(srv.io, .fromMilliseconds(5), .awake) catch {};
    }
    var pending = false;
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

/// An option's value as the engine runs it now: a live change the engine has taken, else the start's value; null when the code decides.
fn currentText(o: *const settings.Option) ?[]const u8 {
    if (admin_state.liveValue(o)) |t| return if (std.mem.eql(u8, t, "unset")) (if (o.default.len == 0) null else o.default) else t;
    return activeText(o);
}
