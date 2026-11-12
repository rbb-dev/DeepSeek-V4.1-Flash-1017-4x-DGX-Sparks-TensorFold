//! The server's own record of the requests it has submitted: the ones in flight and the last finished ones.
//!
//! The engine reports counts and one token total per stream, never which request is which, so the server keeps what the
//! admin surface shows: an entry added where a request enters the engine (chat.zig's one submit) and removed where its
//! reply ends. An entry holds the key's label, the endpoint, counts and times and nothing of the request's text. Only
//! connection threads take the mutex; the engine thread never sees this record.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// Requests in flight that are tracked at once; one more is counted in `untracked` and runs as before.
pub const max_live = 256;
/// Finished requests kept, newest last.
pub const keep_recent = 200;
const label_cap = 64;

pub const Endpoint = enum { chat, completions, messages, responses, other };

/// The endpoint a route path (without its query) submits to.
pub fn endpointOf(route: []const u8) Endpoint {
    if (std.mem.endsWith(u8, route, "/chat/completions")) return .chat;
    if (std.mem.endsWith(u8, route, "/completions")) return .completions;
    if (std.mem.endsWith(u8, route, "/messages")) return .messages;
    if (std.mem.eql(u8, route, "/responses") or std.mem.endsWith(u8, route, "/v1/responses")) return .responses;
    return .other;
}

/// Who is asking and where, set by the connection's thread before it routes a request: the submit site is deep in the
/// reply code and has neither. Both slices point into the connection and live as long as its request.
pub const Context = struct {
    label: []const u8 = "unauthenticated",
    route: []const u8 = "",
};
pub threadlocal var context: Context = .{};

pub const Finish = enum { stop, length, cancelled, @"error" };

/// What a finished request leaves in the recent list.
pub const Done = struct {
    cached_tokens: u64 = 0,
    generated_tokens: u64 = 0,
    ttft_s: ?f64 = null,
    decode_tps: ?f64 = null,
    total_s: f64 = 0,
    end_unix_ms: i64 = 0,
    finish: Finish = .@"error",
};

pub const Live = struct {
    used: bool = false,
    id: u64 = 0,
    label: [label_cap]u8 = undefined,
    label_len: u8 = 0,
    endpoint: Endpoint = .other,
    background: bool = false,
    stream: bool = false,
    begun_ns: i96 = 0,
    prompt_tokens: u32 = 0,
    /// The tokens the reply holds so far, stored by the connection that owns the entry as its reply grows.
    generated: std.atomic.Value(u32) = .init(0),
};

const Recent = struct {
    id: u64 = 0,
    label: [label_cap]u8 = undefined,
    label_len: u8 = 0,
    endpoint: Endpoint = .other,
    prompt_tokens: u32 = 0,
    done: Done = .{},
};

pub const Registry = struct {
    mutex: std.Io.Mutex = .init,
    live: []Live = &.{},
    recent: []Recent = &.{},
    live_n: u32 = 0,
    /// Requests that found every entry taken.
    untracked: u64 = 0,
    ended: u64 = 0,
    last_end_ns: ?i96 = null,

    pub fn deinit(r: *Registry, gpa: Allocator) void {
        gpa.free(r.live);
        gpa.free(r.recent);
        r.* = .{};
    }

    /// Adds a request that is about to be submitted; null when every entry is taken (or memory is). The entry stays
    /// where it is until `end`, so its `generated` counter can be written without the mutex.
    pub fn begin(r: *Registry, gpa: Allocator, io: std.Io, now_ns: i96, id: u64, who: Context, background: bool, stream: bool, prompt_tokens: usize) ?*Live {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        if (r.live.len == 0) {
            r.live = gpa.alloc(Live, max_live) catch return null;
            for (r.live) |*e| e.* = .{};
            r.recent = gpa.alloc(Recent, keep_recent) catch {
                gpa.free(r.live);
                r.live = &.{};
                return null;
            };
        }
        for (r.live) |*e| if (!e.used) {
            const n = @min(who.label.len, label_cap);
            e.* = .{ .used = true, .id = id, .label_len = @intCast(n), .endpoint = endpointOf(who.route), .background = background, .stream = stream, .begun_ns = now_ns, .prompt_tokens = @intCast(@min(prompt_tokens, std.math.maxInt(u32))) };
            @memcpy(e.label[0..n], who.label[0..n]);
            r.live_n += 1;
            return e;
        };
        r.untracked += 1;
        return null;
    }

    /// Removes the entry and keeps what the request did in the recent list.
    pub fn end(r: *Registry, io: std.Io, now_ns: i96, entry: *Live, done: Done) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        const slot = &r.recent[@intCast(r.ended % keep_recent)];
        slot.* = .{ .id = entry.id, .label_len = entry.label_len, .endpoint = entry.endpoint, .prompt_tokens = entry.prompt_tokens, .done = done };
        @memcpy(slot.label[0..entry.label_len], entry.label[0..entry.label_len]);
        r.ended += 1;
        r.last_end_ns = now_ns;
        entry.used = false;
        r.live_n -= 1;
    }

    pub const Idle = struct { idle: bool, for_s: ?f64 };

    /// Whether nothing is in flight, and for how long (null: no request has ended since the start).
    pub fn idle(r: *Registry, io: std.Io, now_ns: i96) Idle {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        const secs: ?f64 = if (r.last_end_ns) |t| @max(0, @as(f64, @floatFromInt(now_ns - t)) / 1e9) else null;
        return .{ .idle = r.live_n == 0, .for_s = if (r.live_n == 0) secs else null };
    }

    /// The context tokens each of the first `out.len` slots holds now (prompt plus the reply so far), 0 for a free slot: the seats'
    /// share of the KV pool, one number a slot a second for the series the pool chart draws. One short lock of the table.
    pub fn seatKv(r: *Registry, io: std.Io, out: []f32) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        for (out, 0..) |*o, i| {
            o.* = if (i < r.live.len and r.live[i].used) @floatFromInt(@as(u64, r.live[i].prompt_tokens) + r.live[i].generated.load(.monotonic)) else 0;
        }
    }

    pub fn liveCount(r: *Registry, io: std.Io) u32 {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        return r.live_n;
    }

    /// `in_flight` and `recent` as the API shows them; labels, counts and times only.
    pub fn view(r: *Registry, io: std.Io, a: Allocator, now_ns: i96) !struct { in_flight: Value, recent: Value, untracked: u64 } {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        var flying: std.ArrayList(Value) = .empty;
        for (r.live, 0..) |*e, slot| if (e.used) {
            const o = try json.newObject(a);
            const generated = e.generated.load(.monotonic);
            try o.put(a, "id", try json.intValue(a, e.id));
            try o.put(a, "slot", try json.intValue(a, slot));
            // the context the request holds in the KV pool now: its prompt (cached or not) and the reply so far
            try o.put(a, "kv_tokens", try json.intValue(a, @as(u64, e.prompt_tokens) + generated));
            try o.put(a, "key_label", .{ .string = try a.dupe(u8, e.label[0..e.label_len]) });
            try o.put(a, "endpoint", .{ .string = @tagName(e.endpoint) });
            try o.put(a, "priority", .{ .string = if (e.background) "background" else "foreground" });
            try o.put(a, "stream", .{ .bool = e.stream });
            try o.put(a, "age_s", .{ .float = @max(0, @as(f64, @floatFromInt(now_ns - e.begun_ns)) / 1e9) });
            try o.put(a, "prompt_tokens", try json.intValue(a, e.prompt_tokens));
            try o.put(a, "generated_tokens", try json.intValue(a, generated));
            // submitted to the engine: its own queue and prefill are not told apart until the first token
            try o.put(a, "state", .{ .string = if (generated == 0) "submitted" else "decode" });
            try flying.append(a, .{ .object = o });
        };
        const kept: usize = @intCast(@min(r.ended, keep_recent));
        var done_rows: std.ArrayList(Value) = .empty;
        // newest first
        var i: usize = 0;
        while (i < kept) : (i += 1) {
            const at: usize = @intCast((r.ended - 1 - i) % keep_recent);
            const e = &r.recent[at];
            const o = try json.newObject(a);
            try o.put(a, "id", try json.intValue(a, e.id));
            try o.put(a, "key_label", .{ .string = try a.dupe(u8, e.label[0..e.label_len]) });
            try o.put(a, "endpoint", .{ .string = @tagName(e.endpoint) });
            try o.put(a, "prompt_tokens", try json.intValue(a, e.prompt_tokens));
            try o.put(a, "cached_tokens", try json.intValue(a, e.done.cached_tokens));
            try o.put(a, "generated_tokens", try json.intValue(a, e.done.generated_tokens));
            try o.put(a, "ttft_s", if (e.done.ttft_s) |x| Value{ .float = x } else .null);
            try o.put(a, "decode_tps", if (e.done.decode_tps) |x| Value{ .float = x } else .null);
            try o.put(a, "total_s", .{ .float = e.done.total_s });
            try o.put(a, "end_ms", try json.intValue(a, e.done.end_unix_ms));
            try o.put(a, "finish", .{ .string = @tagName(e.done.finish) });
            try done_rows.append(a, .{ .object = o });
        }
        return .{ .in_flight = .{ .array = flying.items }, .recent = .{ .array = done_rows.items }, .untracked = r.untracked };
    }
};

test "an entry lives from begin to end, then sits in the recent list, newest first" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r: Registry = .{};
    defer r.deinit(std.testing.allocator);
    try std.testing.expect(r.idle(io, 0).idle);
    const e = r.begin(std.testing.allocator, io, 1_000_000_000, 7, .{ .label = "gateway", .route = "/v1/chat/completions" }, false, true, 120).?;
    e.generated.store(5, .monotonic);
    try std.testing.expectEqual(@as(u32, 1), r.liveCount(io));
    try std.testing.expect(!r.idle(io, 2_000_000_000).idle);
    const v = try r.view(io, arena.allocator(), 3_000_000_000);
    const row = v.in_flight.array[0];
    try std.testing.expectEqualStrings("gateway", row.get("key_label").?.string);
    try std.testing.expectEqualStrings("chat", row.get("endpoint").?.string);
    try std.testing.expectEqualStrings("120", row.get("prompt_tokens").?.int);
    try std.testing.expectEqualStrings("5", row.get("generated_tokens").?.int);
    try std.testing.expectEqualStrings("125", row.get("kv_tokens").?.int);
    try std.testing.expectEqualStrings("0", row.get("slot").?.int);
    var seats: [3]f32 = undefined;
    r.seatKv(io, &seats);
    try std.testing.expectEqual(@as(f32, 125), seats[0]);
    try std.testing.expectEqual(@as(f32, 0), seats[1]); // (a free slot holds nothing)
    try std.testing.expectEqualStrings("decode", row.get("state").?.string);
    try std.testing.expectEqual(@as(f64, 2.0), row.get("age_s").?.float);
    r.end(io, 4_000_000_000, e, .{ .generated_tokens = 9, .finish = .stop, .total_s = 3 });
    try std.testing.expectEqual(@as(u32, 0), r.liveCount(io));
    const idle = r.idle(io, 5_000_000_000);
    try std.testing.expect(idle.idle and idle.for_s.? == 1.0);
    const after = try r.view(io, arena.allocator(), 5_000_000_000);
    try std.testing.expectEqual(@as(usize, 0), after.in_flight.array.len);
    try std.testing.expectEqualStrings("9", after.recent.array[0].get("generated_tokens").?.int);
    try std.testing.expectEqualStrings("stop", after.recent.array[0].get("finish").?.string);
}

test "the recent list keeps the newest 200, and a full table counts the request it cannot track" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r: Registry = .{};
    defer r.deinit(std.testing.allocator);
    var n: u64 = 0;
    while (n < keep_recent + 5) : (n += 1) {
        const e = r.begin(std.testing.allocator, io, 0, n, .{ .label = "k", .route = "/v1/completions" }, true, false, 1).?;
        r.end(io, 0, e, .{});
    }
    const v = try r.view(io, arena.allocator(), 0);
    try std.testing.expectEqual(@as(usize, keep_recent), v.recent.array.len);
    try std.testing.expectEqualStrings(std.fmt.comptimePrint("{d}", .{keep_recent + 4}), v.recent.array[0].get("id").?.int);
    var held: [max_live]*Live = undefined;
    for (&held, 0..) |*h, i| h.* = r.begin(std.testing.allocator, io, 0, i, .{}, false, false, 0).?;
    try std.testing.expect(r.begin(std.testing.allocator, io, 0, 999, .{}, false, false, 0) == null);
    try std.testing.expectEqual(@as(u64, 1), r.untracked);
    for (held) |h| r.end(io, 0, h, .{});
}

test "the endpoint is read from the route" {
    try std.testing.expectEqual(Endpoint.chat, endpointOf("/v1/chat/completions"));
    try std.testing.expectEqual(Endpoint.completions, endpointOf("/v1/completions"));
    try std.testing.expectEqual(Endpoint.messages, endpointOf("/v1/messages"));
    try std.testing.expectEqual(Endpoint.responses, endpointOf("/v1/responses"));
    try std.testing.expectEqual(Endpoint.other, endpointOf("/v1/tokenize"));
}
