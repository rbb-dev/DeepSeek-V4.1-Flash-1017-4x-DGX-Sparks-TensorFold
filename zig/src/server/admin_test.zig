    try std.testing.expect(std.mem.indexOf(u8, st.body, "\"read_only\": false") != null);
    try std.testing.expect(std.mem.indexOf(u8, st.body, "\"writable_classes\": [\"H\", \"A\", \"R\"]") != null);
            if (std.mem.endsWith(u8, path, "/validate") or std.mem.endsWith(u8, path, "/set")) {
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
