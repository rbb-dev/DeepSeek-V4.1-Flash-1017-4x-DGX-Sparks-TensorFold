//! A reply max_tokens cuts inside a DeepSeek-V4.1 DSML call: the whole reply and the stream agree (the same content,
//! the same calls, finish_reason "length").
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;

/// Characters as tokens (a token is a character's code point, as a real tokenizer decodes whole characters); every
/// prompt renders as "prompt".
const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        return chars(a, input) catch return error.OutOfMemory;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        var buf: [4]u8 = undefined;
        for (ids) |id| {
            const n = std.unicode.utf8Encode(@intCast(id), &buf) catch 0;
            try out.appendSlice(a, buf[0..n]);
        }
        return out.toOwnedSlice(a);
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 0x110000;
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

/// Answers `reply` three tokens at a time and ends it with `reason`.
const Script = struct {
    reply: []const u32,
    reason: api.Reason = .length,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var start: usize = 0;
        while (start < e.reply.len) {
            const end = @min(start + 3, e.reply.len);
            sink.event(sink.ctx, id, &.{ .tokens = e.reply[start..end] });
            start = end;
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = e.reason } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// What the reply said: its content, its calls' names, and its finish reason (the stream's joined).
const Got = struct {
    a: std.mem.Allocator,
    content: std.ArrayList(u8) = .empty,
    calls: usize = 0,
    finish: []const u8 = "",

    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        const p = payload orelse return;
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const ch = choices.array[0];
        if (ch.get("finish_reason")) |f| if (f == .string) {
            g.finish = g.a.dupe(u8, f.string) catch return error.Closed;
        };
        const delta = ch.get("delta") orelse return;
        if (delta.get("content")) |c| if (c == .string) g.content.appendSlice(g.a, c.string) catch return error.Closed;
        if (delta.get("tool_calls")) |tc| if (tc == .array) for (tc.array) |call| {
            if (call.get("id") != null) g.calls += 1; // a call's first delta carries its id
        };
    }
    fn reply(ctx: *anyopaque, _: u16, payload: json.Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        const choices = payload.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const ch = choices.array[0];
        if (ch.get("finish_reason")) |f| if (f == .string) {
            g.finish = g.a.dupe(u8, f.string) catch "";
        };
        const m = ch.get("message") orelse return;
        if (m.get("content")) |c| if (c == .string) g.content.appendSlice(g.a, c.string) catch {};
        if (m.get("tool_calls")) |tc| if (tc == .array) {
            g.calls += tc.array.len;
        };
    }
};

fn chat(a: std.mem.Allocator, reply: []const u32, stream: bool) !Got {
    var text: Text = .{};
    var backend: Script = .{ .reply = reply };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    const body = if (stream)
        "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":4000," ++ tools ++ "}"
    else
        "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":4000," ++ tools ++ "}";
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{ .a = a };
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return got;
}

/// `s`'s characters as tokens.
fn chars(a: std.mem.Allocator, s: []const u8) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    var it = (try std.unicode.Utf8View.init(s)).iterator();
    while (it.nextCodepoint()) |cp| try ids.append(a, cp);
    return ids.toOwnedSlice(a);
}

const D = "\u{ff5c}DSML\u{ff5c}";
const lead = "Looking it up.";
const whole_call = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"lookup\">\n<" ++ D ++ " parameter name=\"q\" string=\"true\">abc</" ++
    D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";

test "a DSML call max_tokens cuts: the whole reply and the stream agree, finish_reason length" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // cut at every byte of the call, from just after its block opens to just before its block closes
    const call = try chars(a, whole_call);
    const lead_ids = try chars(a, lead);
    const open_end = (try chars(a, whole_call[0 .. std.mem.indexOf(u8, whole_call, " calls>").? + " calls>".len])).len;
    var cut = open_end;
    while (cut < call.len) : (cut += 5) {
        const reply = try std.mem.concat(a, u32, &.{ lead_ids, call[0..cut] });
        const whole = try chat(a, reply, false);
        const streamed = try chat(a, reply, true);
        errdefer std.debug.print("cut at {d}: whole {d} calls, finish {s}, content [{s}]; stream {d} calls, finish {s}, content [{s}]\n", .{ cut, whole.calls, whole.finish, whole.content.items, streamed.calls, streamed.finish, streamed.content.items });
        try std.testing.expectEqualStrings("length", whole.finish);
        try std.testing.expectEqualStrings("length", streamed.finish);
        try std.testing.expectEqual(whole.calls, streamed.calls);
        try std.testing.expectEqualStrings(whole.content.items, streamed.content.items);
    }
}

test "a whole DSML call and max_tokens right after it: one call, the same in both" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try chars(a, lead ++ whole_call);
    const whole = try chat(a, reply, false);
    const streamed = try chat(a, reply, true);
    try std.testing.expectEqual(@as(usize, 1), whole.calls);
    try std.testing.expectEqual(@as(usize, 1), streamed.calls);
    try std.testing.expectEqualStrings(whole.content.items, streamed.content.items);
}
