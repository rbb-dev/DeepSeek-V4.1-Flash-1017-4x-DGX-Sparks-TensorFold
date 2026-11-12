//! The reasoning split ends at the model's own close token: a `</think>` the reply quotes in pieces stays reasoning.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;

const think_end: u32 = 300; // the close marker as one token
const eos: u32 = 301;
const quote_open: u32 = 302; // the quoted tag, in the three pieces a reply writes it
const quote_mid: u32 = 303;
const quote_close: u32 = 304;
const all_bytes = blk: {
    var all: [256]u8 = undefined;
    for (&all, 0..) |*b, i| b.* = i;
    break :blk all;
};

/// Bytes as tokens, `</think>` as token 300 (unless `close_token` is false) and an end token 301.
const Text = struct {
    close_token: bool = true,

    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn piece(id: u32) []const u8 {
        if (id == think_end) return "</think>";
        if (id == eos) return "";
        if (id == quote_open) return "</";
        if (id == quote_mid) return "think";
        if (id == quote_close) return ">";
        return all_bytes[id..][0..1];
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| try out.appendSlice(a, piece(id));
        return out.toOwnedSlice(a);
    }

    fn tokenId(ctx: *anyopaque, p: []const u8) ?u32 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        if (!t.close_token) return null;
        return if (std.mem.eql(u8, p, "</think>")) think_end else null;
    }

    fn tokenString(ctx: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return decode(ctx, a, &.{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 305;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{eos};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Answers `reply` in chunks of `stride` tokens.
const Script = struct {
    reply: []const u32,
    stride: usize = 2,

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
            const end = @min(start + e.stride, e.reply.len);
            sink.event(sink.ctx, id, &.{ .tokens = e.reply[start..end] });
            start = end;
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// What the route wrote: a whole reply's message, or a stream's reasoning and content deltas joined.
const Got = struct {
    a: std.mem.Allocator,
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,

    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        const p = payload orelse return;
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const delta = choices.array[0].get("delta") orelse return;
        if (delta.get("reasoning_content")) |r| if (r == .string) g.reasoning.appendSlice(g.a, r.string) catch return error.Closed;
        if (delta.get("content")) |c| if (c == .string) g.content.appendSlice(g.a, c.string) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, _: u16, payload: json.Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        const choices = payload.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const m = choices.array[0].get("message") orelse return;
        if (m.get("reasoning_content")) |r| if (r == .string) g.reasoning.appendSlice(g.a, r.string) catch {};
        if (m.get("content")) |c| if (c == .string) g.content.appendSlice(g.a, c.string) catch {};
    }
};

/// The scripted reply through the chat route with thinking on, whole or streamed: its reasoning and content.
fn chat(a: std.mem.Allocator, text: *Text, reply: []const u32, stream: bool) !Got {
    var backend: Script = .{ .reply = reply };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = true,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const body = if (stream)
        "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":200}"
    else
        "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":200}";
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{ .a = a };
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok); // (the chat route)
    return got;
}

/// The text without the spaces at its ends (a whole reply strips its reasoning; a stream's pieces keep them).
fn trim(t: []const u8) []const u8 {
    return std.mem.trim(u8, t, " ");
}

/// "Plan: quote " + the tag in three pieces + " then more." + `close` + "Answer." + the end token.
fn script(a: std.mem.Allocator, close: []const u32) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    for ("Plan: quote ") |b| try ids.append(a, b);
    try ids.appendSlice(a, &.{ quote_open, quote_mid, quote_close });
    for (" then more.") |b| try ids.append(a, b);
    try ids.appendSlice(a, close);
    for ("Answer.") |b| try ids.append(a, b);
    try ids.append(a, eos);
    return ids.toOwnedSlice(a);
}

test "a </think> the reply quotes while it reasons stays reasoning: the model's close token ends it, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: Text = .{};
    const reply = try script(a, &.{think_end});
    for ([_]bool{ false, true }) |stream| {
        const got = try chat(a, &text, reply, stream);
        try std.testing.expectEqualStrings("Plan: quote </think> then more.", trim(got.reasoning.items));
        try std.testing.expectEqualStrings("Answer.", trim(got.content.items));
    }
}

test "with no close token in the tokenizer, the text's first spelling of the marker still ends the reasoning" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: Text = .{ .close_token = false };
    const reply = try script(a, &.{ quote_open, quote_mid, quote_close }); // the close spelled in pieces too
    for ([_]bool{ false, true }) |stream| {
        const got = try chat(a, &text, reply, stream);
        try std.testing.expectEqualStrings("Plan: quote", trim(got.reasoning.items));
        try std.testing.expectEqualStrings("then more.</think>Answer.", trim(got.content.items));
    }
}
