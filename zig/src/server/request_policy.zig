//! Per-request settings (class P): the few fields a request may set about its own stream, and the caps each key label puts on
//! them.
//!
//! A request carries them as a `tf` object in its body (`{"tf": {"pri": "batch", "think_budget": 4000, "wait_ms": 600000}}`) or
//! as headers (`X-TF-Priority`, `X-TF-Think-Budget`, `X-TF-Max-Wait-Ms`, or one `X-TF: pri=batch;think_budget=4000`). They are
//! the stream's own: nothing here touches a step the whole engine runs, so a request can only narrow what it gets or bound
//! what it waits for.
//!
//!   pri           `batch` runs the request as a background request (it yields to foreground prompts); `normal` leaves it be.
//!   think_budget  the most tokens the stream thinks (0: no budget).
//!
//! The caps come from `admin-policy.json` (TENSORFOLD_ADMIN_POLICY), under `request_caps`, one row per key label:
//!   {"request_caps": {"gateway": {"headers": true, "wait_ms_max": 1800000},
//!                     "agent-night": {"batch_only": true, "think_budget_max": 8000, "think_budget_default": 4000}}}
//! A label with no row has no per-request settings at all: whatever it sends is dropped (and counted). Nothing is rejected:
//! a field that is not allowed, malformed or over its cap is dropped or clamped, the reply still runs, and the counters say how
//! often (`/status`.admin.request_fields, per label). Headers count only from a label whose row says `headers: true` (the
//! gateway: it strips what a client sent and writes its own); from any other label they are dropped. A header beats the
//! body's field. The existing `priority` and `thinking_budget` body fields keep working as before; a row's
//! `think_budget_max` caps the thinking budget however it was asked.
//!
//! Cost: parsing runs on the connection's thread in `prepare`, once per request, and only a request that carries a field (or
//! whose label has a row with defaults) takes the counters' lock. The engine thread reads nothing from here.
const std = @import("std");
const posix = std.posix;
const json = @import("json");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const Field = enum(u2) { pri, think_budget, wait_ms };
const field_count = 3;

/// A set of fields.
pub const Mask = struct {
    bits: u8 = 0,
    pub fn add(m: *Mask, f: Field) void {
        m.bits |= @as(u8, 1) << @backingInt(f);
    }
    pub fn has(m: Mask, f: Field) bool {
        return m.bits & (@as(u8, 1) << @backingInt(f)) != 0;
    }
    pub fn count(m: Mask) u32 {
        return @popCount(m.bits);
    }
};

/// What one key label may do.
pub const Row = struct {
    /// Headers from this label are the policy's own (the gateway's).
    headers: bool = false,
    /// Every request of this label runs as batch, whatever it asks.
    batch_only: bool = false,
    /// 0: no cap.
    think_budget_max: u32 = 0,
    /// Used when the request names none; 0: the engine's own.
    think_budget_default: u32 = 0,
    wait_ms_max: u32 = 0,
    wait_ms_default: u32 = 0,
};

/// The fields a request asked for, as text, from one carrier.
pub const Raw = struct {
    pri: ?[]const u8 = null,
    think_budget: ?[]const u8 = null,
    wait_ms: ?[]const u8 = null,

    pub fn any(r: Raw) bool {
        return r.pri != null or r.think_budget != null or r.wait_ms != null;
    }
};

/// What a request gets.
pub const Applied = struct {
    /// Run as a background request (the request's own `priority: "background"` also does).
    batch: bool = false,
    /// The thinking budget the request gets; null: the request's own `thinking_budget`, else the engine's.
    think_budget: ?u32 = null,
    /// The cap on any thinking budget, however asked; 0: none.
    think_cap: u32 = 0,
    wait_ms: ?u32 = null,
    dropped: Mask = .{},
    clamped: Mask = .{},
    /// The fields that took effect.
    used: Mask = .{},
    /// Some field was asked or defaulted (the counters' trigger).
    touched: bool = false,

    /// A thinking budget as the request asked it (0: none), under the row's cap.
    pub fn capBudget(a: Applied, budget: i64) i64 {
        if (a.think_cap == 0) return budget;
        return if (budget <= 0 or budget > a.think_cap) a.think_cap else budget;
    }
};

fn parseBatch(text: []const u8) ?bool {
    if (std.ascii.eqlIgnoreCase(text, "batch") or std.ascii.eqlIgnoreCase(text, "background")) return true;
    if (std.ascii.eqlIgnoreCase(text, "normal") or std.ascii.eqlIgnoreCase(text, "foreground")) return false;
    return null;
}

fn parseCount(text: []const u8) ?u32 {
    const t = std.mem.trim(u8, text, " \t");
    if (t.len == 0) return null;
    return std.fmt.parseInt(u32, t, 10) catch null;
}

/// Decides what a request gets: `row` is its label's row (null: none), `body` and `header` what it carried.
pub fn resolve(row: ?Row, body: Raw, header: Raw) Applied {
    var out: Applied = .{};
    const asked = body.any() or header.any();
    const r = row orelse {
        if (asked) {
            out.touched = true;
            for ([_]struct { f: Field, text: ?[]const u8, other: ?[]const u8 }{
                .{ .f = .pri, .text = body.pri, .other = header.pri },
                .{ .f = .think_budget, .text = body.think_budget, .other = header.think_budget },
                .{ .f = .wait_ms, .text = body.wait_ms, .other = header.wait_ms },
            }) |c| if (c.text != null or c.other != null) out.dropped.add(c.f);
        }
        return out;
    };
    // a header counts only where the row trusts headers; elsewhere it is dropped and the body's field (if any) stands
    var pri = body.pri;
    var think = body.think_budget;
    var wait = body.wait_ms;
    if (header.any()) {
        out.touched = true;
        if (r.headers) {
            pri = header.pri orelse pri;
            think = header.think_budget orelse think;
            wait = header.wait_ms orelse wait;
        } else {
            if (header.pri != null) out.dropped.add(.pri);
            if (header.think_budget != null) out.dropped.add(.think_budget);
            if (header.wait_ms != null) out.dropped.add(.wait_ms);
        }
    }
    if (body.any()) out.touched = true;
    out.think_cap = r.think_budget_max;

    // pri
    var batch = r.batch_only;
    if (pri) |text| {
        if (parseBatch(text)) |b| {
            if (b) batch = true else if (r.batch_only) out.clamped.add(.pri);
        } else out.dropped.add(.pri);
    }
    out.batch = batch;
    if (batch) out.used.add(.pri);
    if (r.batch_only) out.touched = true;

    // think_budget
    var tb: ?u32 = null;
    if (think) |text| {
        if (parseCount(text)) |v| tb = v else out.dropped.add(.think_budget);
    }
    if (tb == null and r.think_budget_default > 0) {
        tb = r.think_budget_default;
        out.touched = true;
    }
    if (tb) |v| {
        if (r.think_budget_max > 0 and (v == 0 or v > r.think_budget_max)) {
            tb = r.think_budget_max;
            if (think != null) out.clamped.add(.think_budget);
        }
    }
    out.think_budget = tb;
    if (tb != null) out.used.add(.think_budget);

    // wait_ms
    var wm: ?u32 = null;
    if (wait) |text| {
        if (parseCount(text)) |v| wm = v else out.dropped.add(.wait_ms);
    }
    if (wm == null and r.wait_ms_default > 0) {
        wm = r.wait_ms_default;
        out.touched = true;
    }
    if (wm) |v| {
        if (r.wait_ms_max > 0 and (v == 0 or v > r.wait_ms_max)) {
            wm = r.wait_ms_max;
            if (wait != null) out.clamped.add(.wait_ms);
        }
        if (wm.? == 0) wm = null; // 0 asked, nothing capped: no bound
    }
    out.wait_ms = wm;
    if (wm != null) out.used.add(.wait_ms);
    return out;
}

/// The text of one field of a `tf` body object (a string, or a number as its decimal text); null when absent or of another type.
fn bodyText(a: Allocator, tf: Value, key: []const u8) ?[]const u8 {
    const v = tf.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        .int => |t| t,
        .float => |x| blk: {
            if (!std.math.isFinite(x) or x < 0 or x != @floor(x) or x > 4294967295.0) break :blk "bad";
            break :blk std.fmt.allocPrint(a, "{d}", .{@as(u64, @intFromFloat(x))}) catch null;
        },
        else => "bad", // present, of a type no field takes: counted as dropped
    };
}

/// The `tf` object of a request body (the raw fields); a `tf` that is not an object carries nothing.
pub fn fromBody(a: Allocator, fields: Value) Raw {
    const tf = fields.get("tf") orelse return .{};
    if (tf != .object) return .{};
    return .{ .pri = bodyText(a, tf, "pri"), .think_budget = bodyText(a, tf, "think_budget"), .wait_ms = bodyText(a, tf, "wait_ms") };
}

/// The fields of the request's headers: `get` answers one header's value by name (the first, case-insensitive).
pub fn fromHeaders(ctx: anytype, comptime get: fn (@TypeOf(ctx), []const u8) ?[]const u8) Raw {
    var out: Raw = .{
        .pri = get(ctx, "X-TF-Priority"),
        .think_budget = get(ctx, "X-TF-Think-Budget"),
        .wait_ms = get(ctx, "X-TF-Max-Wait-Ms"),
    };
    if (get(ctx, "X-TF")) |combined| {
        var it = std.mem.splitScalar(u8, combined, ';');
        while (it.next()) |part| {
            const eq = std.mem.indexOfScalar(u8, part, '=') orelse continue;
            const k = std.mem.trim(u8, part[0..eq], " \t");
            const v = std.mem.trim(u8, part[eq + 1 ..], " \t");
            if (std.mem.eql(u8, k, "pri") and out.pri == null) out.pri = v;
            if (std.mem.eql(u8, k, "think_budget") and out.think_budget == null) out.think_budget = v;
            if (std.mem.eql(u8, k, "wait_ms") and out.wait_ms == null) out.wait_ms = v;
        }
    }
    return out;
}

/// The headers of the request this thread is serving, set by the route before it calls `prepare` (the submit site is deep in
/// the reply code and has no connection). The slices belong to the connection's request.
pub threadlocal var carried: Raw = .{};

pub const max_labels = 32;
const label_cap = 64;

/// What each label's requests did with the fields: requests that carried or defaulted any, fields applied, clamped, dropped.
pub const Count = struct {
    label: [label_cap]u8 = undefined,
    label_len: u8 = 0,
    requests: u64 = 0,
    applied: u64 = 0,
    clamped: u64 = 0,
    dropped: u64 = 0,
};

/// The caps, read from the policy file, and the counters.
pub const Policy = struct {
    gpa: Allocator,
    path: ?[]const u8 = null,
    mutex: std.Io.Mutex = .init,
    arena: ?std.heap.ArenaAllocator = null,
    labels: []const []const u8 = &.{},
    rows: []const Row = &.{},
    valid: bool = true,
    /// Why the newest load was refused (no row stands then); empty while the file loads.
    refusal: []const u8 = "",
    counts: [max_labels]Count = @splat(.{}),
    counted: usize = 0,
    /// Requests of a label that found every counter slot taken.
    uncounted: u64 = 0,

    pub fn deinit(p: *Policy) void {
        if (p.arena) |*a| a.deinit();
        if (p.refusal.len > 0) p.gpa.free(p.refusal);
        p.* = .{ .gpa = p.gpa };
    }

    /// Reads `path` now: a missing or bad file leaves no row (every label's fields are dropped) and says why.
    pub fn load(p: *Policy, io: std.Io, path: []const u8) void {
        p.path = path;
        p.reload(io);
    }

    pub fn reload(p: *Policy, io: std.Io) void {
        const path = p.path orelse return;
        var problem: []const u8 = "";
        var arena: std.heap.ArenaAllocator = .init(p.gpa);
        const parsed = readRows(arena.allocator(), path, &problem);
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        if (p.arena) |*old| old.deinit();
        if (p.refusal.len > 0) p.gpa.free(p.refusal);
        p.refusal = "";
        if (parsed) |loaded| {
            p.arena = arena;
            p.labels = loaded.labels;
            p.rows = loaded.rows;
            p.valid = true;
        } else |_| {
            p.arena = null;
            p.labels = &.{};
            p.rows = &.{};
            p.valid = false;
            p.refusal = p.gpa.dupe(u8, if (problem.len > 0) problem else "the policy file cannot be read") catch ""; // before the arena that holds `problem` goes
            arena.deinit();
            log.line("admin policy: {s}; no key label has per-request settings until it is fixed", .{p.refusal});
        }
    }

    /// The row of `label`, or null.
    pub fn row(p: *Policy, io: std.Io, label: []const u8) ?Row {
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        for (p.labels, p.rows) |l, r| if (std.mem.eql(u8, l, label)) return r;
        return null;
    }

    /// Counts a request's outcome under its label.
    pub fn note(p: *Policy, io: std.Io, label: []const u8, applied: Applied) void {
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        const n = @min(label.len, label_cap);
        const slot: ?*Count = for (p.counts[0..p.counted]) |*c| {
            if (std.mem.eql(u8, c.label[0..c.label_len], label[0..n])) break c;
        } else if (p.counted < max_labels) blk: {
            const c = &p.counts[p.counted];
            p.counted += 1;
            c.* = .{ .label_len = @intCast(n) };
            @memcpy(c.label[0..n], label[0..n]);
            break :blk c;
        } else null;
        const c = slot orelse {
            p.uncounted += 1;
            return;
        };
        c.requests += 1;
        c.applied += applied.used.count();
        c.clamped += applied.clamped.count();
        c.dropped += applied.dropped.count();
    }

    /// The counters as a JSON array of per-label rows (copied into `a`), and the requests that found no slot.
    pub fn countsValue(p: *Policy, io: std.Io, a: Allocator) !Value {
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        var list: std.ArrayList(Value) = .empty;
        for (p.counts[0..p.counted]) |*c| {
            const o = try json.newObject(a);
            try o.put(a, "label", .{ .string = try a.dupe(u8, c.label[0..c.label_len]) });
            try o.put(a, "requests", try json.intValue(a, c.requests));
            try o.put(a, "applied", try json.intValue(a, c.applied));
            try o.put(a, "clamped", try json.intValue(a, c.clamped));
            try o.put(a, "dropped", try json.intValue(a, c.dropped));
            try list.append(a, .{ .object = o });
        }
        return .{ .array = list.items };
    }

    /// GET /policy's request caps: one object per label, as the file holds them.
    pub fn capsValue(p: *Policy, io: std.Io, a: Allocator) !Value {
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        var list: std.ArrayList(Value) = .empty;
        for (p.labels, p.rows) |l, r| {
            const o = try json.newObject(a);
            try o.put(a, "label", .{ .string = try a.dupe(u8, l) });
            try o.put(a, "headers", .{ .bool = r.headers });
            try o.put(a, "batch_only", .{ .bool = r.batch_only });
            try o.put(a, "think_budget_max", try json.intValue(a, r.think_budget_max));
            try o.put(a, "think_budget_default", try json.intValue(a, r.think_budget_default));
            try o.put(a, "wait_ms_max", try json.intValue(a, r.wait_ms_max));
            try o.put(a, "wait_ms_default", try json.intValue(a, r.wait_ms_default));
            try list.append(a, .{ .object = o });
        }
        return .{ .array = list.items };
    }

    pub const Summary = struct { labels: usize, valid: bool, refusal: []const u8, configured: bool };

    pub fn summary(p: *Policy, io: std.Io, a: Allocator) !Summary {
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        return .{ .labels = p.rows.len, .valid = p.valid, .refusal = try a.dupe(u8, p.refusal), .configured = p.path != null };
    }
};

const Loaded = struct { labels: []const []const u8, rows: []const Row };

const known_row_fields = [_][]const u8{ "headers", "batch_only", "think_budget_max", "think_budget_default", "wait_ms_max", "wait_ms_default" };

fn readRows(a: Allocator, path: []const u8, problem: *[]const u8) error{Refused}!Loaded {
    const pathz = a.dupeSentinel(u8, path, 0) catch return error.Refused;
    const fd = posix.openatZ(posix.AT.FDCWD, pathz, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true }, 0) catch {
        problem.* = "the policy file cannot be opened";
        return error.Refused;
    };
    defer _ = posix.system.close(fd);
    const cap = 1 << 20;
    const text = a.alloc(u8, cap + 1) catch return error.Refused;
    var got: usize = 0;
    while (got < text.len) {
        const n = posix.read(fd, text[got..]) catch {
            problem.* = "the policy file cannot be read";
            return error.Refused;
        };
        if (n == 0) break;
        got += n;
    }
    if (got > cap) {
        problem.* = "the policy file exceeds 1 MiB";
        return error.Refused;
    }
    return parseRows(a, text[0..got], problem);
}

/// The top-level key of the policy document that holds the per-request caps. The document's other top-level keys are the agent
/// profiles' labels (API-FULL 2.3), so a profile reader walks `profileLabels` and never meets this one; a key label named
/// `request_caps` can therefore have no profile.
pub const reserved_key = "request_caps";

/// The top-level keys of a policy document that name agent profiles: every key but `request_caps`. A document that is not
/// a JSON object has none.
pub fn profileLabels(a: Allocator, text: []const u8) Allocator.Error![]const []const u8 {
    const parsed = json.parse(a, text) catch |e| return e;
    const doc = switch (parsed) {
        .ok => |v| v,
        .err => return &.{},
    };
    if (doc != .object) return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    for (doc.object.keys()) |k| if (!std.mem.eql(u8, k, reserved_key)) try out.append(a, k);
    return out.items;
}

/// The rows of a policy document's `request_caps`; any other top-level key (the agent profiles) is not read here.
pub fn parseRows(a: Allocator, text: []const u8, problem: *[]const u8) error{Refused}!Loaded {
    const parsed = json.parse(a, text) catch {
        problem.* = "the policy file is not JSON";
        return error.Refused;
    };
    const doc = switch (parsed) {
        .ok => |v| v,
        .err => {
            problem.* = "the policy file is not JSON";
            return error.Refused;
        },
    };
    if (doc != .object) {
        problem.* = "the policy file is a JSON object";
        return error.Refused;
    }
    var labels: std.ArrayList([]const u8) = .empty;
    var rows: std.ArrayList(Row) = .empty;
    const caps = doc.get("request_caps") orelse return .{ .labels = &.{}, .rows = &.{} };
    if (caps != .object) {
        problem.* = "request_caps is an object of key labels";
        return error.Refused;
    }
    for (caps.object.keys(), caps.object.values()) |label, v| {
        if (v != .object) {
            problem.* = std.fmt.allocPrint(a, "request_caps.{s} is an object", .{label}) catch "a request_caps row is an object";
            return error.Refused;
        }
        var r: Row = .{};
        for (v.object.keys(), v.object.values()) |k, x| {
            var known = false;
            for (known_row_fields) |kf| known = known or std.mem.eql(u8, kf, k);
            if (!known) {
                problem.* = std.fmt.allocPrint(a, "request_caps.{s}: unknown field {s}", .{ label, k }) catch "a request_caps row has an unknown field";
                return error.Refused;
            }
            if (std.mem.eql(u8, k, "headers") or std.mem.eql(u8, k, "batch_only")) {
                if (x != .bool) {
                    problem.* = std.fmt.allocPrint(a, "request_caps.{s}.{s} is true or false", .{ label, k }) catch "a flag is true or false";
                    return error.Refused;
                }
                if (std.mem.eql(u8, k, "headers")) r.headers = x.bool else r.batch_only = x.bool;
                continue;
            }
            const n: u32 = if (x == .int) (std.fmt.parseInt(u32, x.int, 10) catch 0) else 0;
            if (x != .int or (n == 0 and !std.mem.eql(u8, x.int, "0"))) {
                problem.* = std.fmt.allocPrint(a, "request_caps.{s}.{s} is a whole number", .{ label, k }) catch "a cap is a whole number";
                return error.Refused;
            }
            if (std.mem.eql(u8, k, "think_budget_max")) r.think_budget_max = n;
            if (std.mem.eql(u8, k, "think_budget_default")) r.think_budget_default = n;
            if (std.mem.eql(u8, k, "wait_ms_max")) r.wait_ms_max = n;
            if (std.mem.eql(u8, k, "wait_ms_default")) r.wait_ms_default = n;
        }
        labels.append(a, label) catch return error.Refused;
        rows.append(a, r) catch return error.Refused;
    }
    return .{ .labels = labels.items, .rows = rows.items };
}

// ---------------------------------------------------------------------------------------------------------------------------
// tests

const testing = std.testing;

fn raw(pri: ?[]const u8, think: ?[]const u8, wait: ?[]const u8) Raw {
    return .{ .pri = pri, .think_budget = think, .wait_ms = wait };
}

test "a label with no row: everything it sends is dropped and nothing applies" {
    const o = resolve(null, raw("batch", "100", null), raw(null, null, "5"));
    try testing.expect(!o.batch);
    try testing.expectEqual(@as(?u32, null), o.think_budget);
    try testing.expectEqual(@as(?u32, null), o.wait_ms);
    try testing.expect(o.dropped.has(.pri) and o.dropped.has(.think_budget) and o.dropped.has(.wait_ms));
    try testing.expect(o.touched);
    const none = resolve(null, .{}, .{});
    try testing.expect(!none.touched);
}

test "priority: batch narrows, normal cannot lift a batch-only label, junk is dropped" {
    const open: Row = .{};
    try testing.expect(resolve(open, raw("batch", null, null), .{}).batch);
    try testing.expect(resolve(open, raw("background", null, null), .{}).batch);
    try testing.expect(!resolve(open, raw("normal", null, null), .{}).batch);
    const night: Row = .{ .batch_only = true };
    const lift = resolve(night, raw("normal", null, null), .{});
    try testing.expect(lift.batch and lift.clamped.has(.pri));
    try testing.expect(resolve(night, .{}, .{}).batch); // the key is the policy: it asks nothing and is batch
    const junk = resolve(open, raw("urgent", null, null), .{});
    try testing.expect(!junk.batch and junk.dropped.has(.pri));
}

test "think_budget and wait_ms: caps clamp, defaults fill, zero is no bound until a cap says otherwise" {
    const r: Row = .{ .think_budget_max = 8000, .think_budget_default = 4000, .wait_ms_max = 60000 };
    try testing.expectEqual(@as(?u32, 4000), resolve(r, .{}, .{}).think_budget); // the default
    const big = resolve(r, raw(null, "100000", "999999"), .{});
    try testing.expectEqual(@as(?u32, 8000), big.think_budget);
    try testing.expectEqual(@as(?u32, 60000), big.wait_ms);
    try testing.expect(big.clamped.has(.think_budget) and big.clamped.has(.wait_ms));
    const zero = resolve(r, raw(null, "0", "0"), .{});
    try testing.expectEqual(@as(?u32, 8000), zero.think_budget); // "no budget" is over a cap
    try testing.expectEqual(@as(?u32, 60000), zero.wait_ms);
    const small = resolve(r, raw(null, "500", "1000"), .{});
    try testing.expectEqual(@as(?u32, 500), small.think_budget);
    try testing.expectEqual(@as(?u32, 1000), small.wait_ms);
    try testing.expectEqual(@as(u32, 0), small.clamped.count());
    const free = resolve(.{}, raw(null, "0", "0"), .{});
    try testing.expectEqual(@as(?u32, 0), free.think_budget);
    try testing.expectEqual(@as(?u32, null), free.wait_ms);
    const bad = resolve(r, raw(null, "-5", "soon"), .{});
    try testing.expect(bad.dropped.has(.think_budget) and bad.dropped.has(.wait_ms));
    try testing.expectEqual(@as(?u32, 4000), bad.think_budget); // the default stands
}

test "a cap bounds the existing thinking_budget field too" {
    const a = resolve(.{ .think_budget_max = 1000 }, .{}, .{});
    try testing.expectEqual(@as(i64, 1000), a.capBudget(0));
    try testing.expectEqual(@as(i64, 1000), a.capBudget(5000));
    try testing.expectEqual(@as(i64, 300), a.capBudget(300));
    const none = resolve(.{}, .{}, .{});
    try testing.expectEqual(@as(i64, 0), none.capBudget(0));
}

test "headers count only from a label whose row trusts them, and beat the body" {
    const plain: Row = .{};
    const from_client = resolve(plain, raw("normal", null, null), raw("batch", "10", null));
    try testing.expect(!from_client.batch); // the header is dropped, the body's normal stands
    try testing.expect(from_client.dropped.has(.pri) and from_client.dropped.has(.think_budget));
    const gateway: Row = .{ .headers = true };
    const trusted = resolve(gateway, raw("normal", "50", null), raw("batch", null, null));
    try testing.expect(trusted.batch);
    try testing.expectEqual(@as(?u32, 50), trusted.think_budget); // the header named only pri
}

test "fromHeaders reads the separate headers and the combined one, the separate winning" {
    const H = struct {
        fn get(_: void, name: []const u8) ?[]const u8 {
            if (std.ascii.eqlIgnoreCase(name, "X-TF-Priority")) return "normal";
            if (std.ascii.eqlIgnoreCase(name, "X-TF")) return "pri=batch; think_budget = 700;wait_ms=9;junk;x=1";
            return null;
        }
    };
    const r = fromHeaders({}, H.get);
    try testing.expectEqualStrings("normal", r.pri.?);
    try testing.expectEqualStrings("700", r.think_budget.?);
    try testing.expectEqualStrings("9", r.wait_ms.?);
}

test "fromBody reads a tf object: strings, ints and whole floats; anything else is dropped by its parse" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parse(a, "{\"tf\": {\"pri\": \"batch\", \"think_budget\": 4000, \"wait_ms\": 2.5}, \"x\": 1}")).ok;
    const r = fromBody(a, doc);
    try testing.expectEqualStrings("batch", r.pri.?);
    try testing.expectEqualStrings("4000", r.think_budget.?);
    const o = resolve(.{}, r, .{});
    try testing.expect(o.batch and o.dropped.has(.wait_ms));
    const none = fromBody(a, (try json.parse(a, "{\"tf\": 3}")).ok);
    try testing.expect(!none.any());
    const odd = fromBody(a, (try json.parse(a, "{\"tf\": {\"pri\": true}}")).ok);
    try testing.expect(resolve(.{}, odd, .{}).dropped.has(.pri));
}

test "the policy file: rows per label, strict fields, other top-level keys left alone" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const ok = try parseRows(a, "{\"agent-night\": {\"names\": []}, \"request_caps\": {\"gateway\": {\"headers\": true, \"wait_ms_max\": 1800000}, \"agent-night\": {\"batch_only\": true, \"think_budget_max\": 8000}}}", &problem);
    try testing.expectEqual(@as(usize, 2), ok.rows.len);
    try testing.expectEqualStrings("gateway", ok.labels[0]);
    try testing.expect(ok.rows[0].headers and ok.rows[0].wait_ms_max == 1800000);
    try testing.expect(ok.rows[1].batch_only and ok.rows[1].think_budget_max == 8000);
    for ([_][]const u8{
        "{\"request_caps\": {\"g\": {\"speed\": 1}}}",
        "{\"request_caps\": {\"g\": {\"headers\": 1}}}",
        "{\"request_caps\": {\"g\": {\"wait_ms_max\": \"5\"}}}",
        "{\"request_caps\": {\"g\": {\"wait_ms_max\": -1}}}",
        "{\"request_caps\": {\"g\": 5}}",
        "{\"request_caps\": []}",
        "[]",
        "{not json",
    }) |bad| {
        problem = "";
        try testing.expectError(error.Refused, parseRows(a, bad, &problem));
        try testing.expect(problem.len > 0);
    }
    const empty = try parseRows(a, "{}", &problem);
    try testing.expectEqual(@as(usize, 0), empty.rows.len);
}

test "counters are kept per label, and a file that goes bad leaves no row" {
    var p: Policy = .{ .gpa = testing.allocator };
    defer p.deinit();
    const io = testing.io;
    var o = resolve(.{ .wait_ms_max = 10 }, raw(null, null, "99"), .{});
    p.note(io, "gateway", o);
    o = resolve(null, raw("batch", null, null), .{});
    p.note(io, "tools", o);
    p.note(io, "gateway", o);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v = try p.countsValue(io, arena.allocator());
    try testing.expectEqual(@as(usize, 2), v.array.len);
    try testing.expectEqualStrings("gateway", v.array[0].get("label").?.string);
    try testing.expectEqualStrings("2", v.array[0].get("requests").?.int);
    try testing.expectEqualStrings("1", v.array[0].get("clamped").?.int);
    try testing.expectEqualStrings("1", v.array[0].get("dropped").?.int);
    try testing.expectEqual(@as(?Row, null), p.row(io, "gateway"));
}

test "the profile reader's keys leave out request_caps, and the caps reader reads nothing else" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = "{\"agent-night\": {\"names\": []}, \"request_caps\": {\"gateway\": {\"headers\": true}}, \"agent-ops\": {}}";
    const keys = try profileLabels(a, text);
    try testing.expectEqual(@as(usize, 2), keys.len);
    try testing.expectEqualStrings("agent-night", keys[0]);
    try testing.expectEqualStrings("agent-ops", keys[1]);
    try testing.expectEqual(@as(usize, 0), (try profileLabels(a, "[1]")).len);
    try testing.expectEqual(@as(usize, 0), (try profileLabels(a, "{\"request_caps\": {}}")).len);
    var problem: []const u8 = "";
    const rows = try parseRows(a, text, &problem);
    try testing.expectEqual(@as(usize, 1), rows.rows.len); // the profiles are not caps rows
}
