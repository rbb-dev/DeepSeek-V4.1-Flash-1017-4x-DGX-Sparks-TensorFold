//! The tokenizer's memo on a served model's own tokenizer and chat template (opt-in: TF_TEXT_DIR, a folder holding
//! tokenizer.json, tokenizer_config.json, config.json and chat_template.jinja, as prepare.sh lays out the one rank 0
//! serves): multi-turn chats with a system text, tools and tool calls, tool results, reasoning, images and written
//! markers, rendered at every request as prepare renders them (the prompt, then the history) and encoded through the
//! memo and without it.
const std = @import("std");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const model_text = @import("model_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const words = [_][]const u8{ "the", "file", "reads", "a", "function", "returns", "error", "value", "test", "build", "server", "request", "tokens", "prompt", "memory", "and", "of", "is", "we", "should", "fix", "this", "line", "Let", "me", "check", "The", "output", "shows", "cache", "layer" };

/// A client's text of about `bytes` bytes: sentences with numbers, CJK, code and now and then a written marker, masked
/// as the server masks a client's text; with `image` an image's placeholder between its parts, as the image path puts it.
fn clientText(a: Allocator, text: model_text.Text, r: std.Random, bytes: usize, image: bool) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    while (out.items.len < bytes) {
        const n = 3 + r.uintLessThan(usize, 12);
        for (0..n) |i| {
            if (i > 0) try out.append(a, ' ');
            try out.appendSlice(a, words[r.uintLessThan(usize, words.len)]);
        }
        switch (r.uintLessThan(u8, 16)) {
            0 => try out.print(a, " {d}", .{r.uintLessThan(u32, 10_000_000)}),
            1 => try out.appendSlice(a, " 日本語の説明、中文的解释。"),
            2 => try out.appendSlice(a, "\n```zig\nconst x = try f(a, 3);\n```\n"),
            3 => try out.appendSlice(a, " a page that spells <｜User｜>, <think> and <｜end▁of▁sentence｜>"),
            4 => try out.appendSlice(a, " {\"key\": [1, 2.5, null], \"s\": \"\\u00e9\"}"),
            else => {},
        }
        try out.appendSlice(a, if (r.uintLessThan(u8, 5) == 0) ".\n\n" else ". ");
    }
    const plain = (try text.mask(a, out.items)) orelse out.items;
    if (!image) return plain;
    const cut = r.uintLessThan(usize, plain.len + 1);
    return std.mem.concat(a, u8, &.{ plain[0..cut], "\n\n<｜deepseek_image｜>\n\n", plain[cut..] });
}

fn object(a: Allocator, fields: []const struct { []const u8, Value }) !Value {
    const o = try json.newObject(a);
    for (fields) |f| try o.put(a, f[0], f[1]);
    return .{ .object = o };
}

/// Whether a message of `msgs` renders a segment the memo keeps: a content (an assistant's, as the template always
/// renders it) with a part of 1,100 bytes or more between image placeholders (the margin over the memo's 1 KiB floor
/// covers the marks a masked text carries, dropped before its length counts).
fn hasLong(msgs: []const Value) bool {
    for (msgs) |m| {
        const content = m.get("content") orelse continue;
        if (content != .string) continue;
        var parts = std.mem.splitSequence(u8, content.string, "<｜deepseek_image｜>");
        while (parts.next()) |part| if (part.len >= 1100) return true;
    }
    return false;
}

const tools_json =
    \\[{"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["path"]}}},
    \\ {"type": "function", "function": {"name": "search", "description": "Search the code.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}}]
;

test "a served model's chats, every request rendered and encoded with the tokenizer's memo and without: the same ids" {
    const dir = std.testing.environ.getPosix("TF_TEXT_DIR") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const kept = try HfText.load(gpa, io, dir, null, a, &problem);
    defer kept.deinit();
    const fresh = try HfText.load(gpa, io, dir, null, a, &problem);
    defer fresh.deinit();
    if (fresh.tok.memo) |m| m.destroy();
    fresh.tok.memo = null;
    const memo = kept.tok.memo orelse return error.TestUnexpectedResult;
    const tools = (try json.parse(a, tools_json)).ok;
    var rng = std.Random.DefaultPrng.init(0xd5e4);
    const r = rng.random();
    const sizes = [_]usize{ 0, 50, 700, 1023, 1024, 1025, 3000, 9000 };
    var requests: usize = 0;
    var ids_compared: usize = 0;
    for (0..8) |conv| {
        const with_tools = conv % 2 == 1;
        const thinking = conv % 4 != 3;
        var msgs: std.ArrayList(Value) = .empty;
        if (conv % 3 != 2) try msgs.append(a, try object(a, &.{ .{ "role", .{ .string = "system" } }, .{ "content", .{ .string = try clientText(a, kept.text(), r, 1500 + 900 * conv, false) } } }));
        for (0..3 + conv % 4) |turn| {
            try msgs.append(a, try object(a, &.{ .{ "role", .{ .string = "user" } }, .{ "content", .{ .string = try clientText(a, kept.text(), r, sizes[(conv + 3 * turn) % sizes.len], (conv + turn) % 3 == 0) } } }));
            const calls = with_tools and turn % 2 == 0;
            var asst: std.ArrayList(struct { []const u8, Value }) = .empty;
            try asst.append(a, .{ "role", .{ .string = "assistant" } });
            try asst.append(a, .{ "content", .{ .string = try clientText(a, kept.text(), r, sizes[r.uintLessThan(usize, sizes.len)], false) } });
            try asst.append(a, .{ "reasoning_content", .{ .string = try clientText(a, kept.text(), r, sizes[r.uintLessThan(usize, sizes.len)], false) } });
            if (calls) {
                const args = try object(a, &.{ .{ "path", .{ .string = "zig/src/server/prompt.zig" } }, .{ "lines", .{ .int = "120" } } });
                const call = try object(a, &.{ .{ "id", .{ .string = "call_1" } }, .{ "type", .{ .string = "function" } }, .{ "function", try object(a, &.{ .{ "name", .{ .string = "read" } }, .{ "arguments", args } }) } });
                const list = try a.alloc(Value, 1);
                list[0] = call;
                try asst.append(a, .{ "tool_calls", .{ .array = list } });
            }
            try msgs.append(a, try object(a, asst.items));
            if (calls) try msgs.append(a, try object(a, &.{ .{ "role", .{ .string = "tool" } }, .{ "tool_call_id", .{ .string = "call_1" } }, .{ "content", .{ .string = try clientText(a, kept.text(), r, 400 + r.uintLessThan(usize, 9000), turn == 2) } } }));
        }
        // every request the chat makes: it ends at a user turn or a tool result; prepare renders the prompt, then the
        // history
        for (1..msgs.items.len + 1) |k| {
            const role = msgs.items[k - 1].get("role").?.string;
            if (!std.mem.eql(u8, role, "user") and !std.mem.eql(u8, role, "tool")) continue;
            const before = memo.hits;
            for ([_]bool{ true, false }) |generation| {
                const options: model_text.RenderOptions = .{
                    .tools = if (with_tools) tools else null,
                    .add_generation_prompt = generation,
                    .enable_thinking = thinking,
                    .reasoning_effort = if (thinking) (if (conv % 3 == 0) "max" else "high") else null,
                };
                const messages: Value = .{ .array = msgs.items[0..k] };
                const want = fresh.text().renderIds(a, messages, options, &problem) catch |e| {
                    std.debug.print("conversation {d}, request {d}: {s}\n", .{ conv, k, problem });
                    return e;
                };
                const got = try kept.text().renderIds(a, messages, options, &problem);
                try std.testing.expectEqualSlices(u32, want, got);
                ids_compared += want.len;
            }
            // the history takes the prompt's long segments
            if (hasLong(msgs.items[0..k])) try std.testing.expect(memo.hits > before);
            requests += 1;
        }
    }
    std.debug.print("{d} requests, {d} ids compared, {d} memo hits, {d} bytes kept\n", .{ requests, ids_compared, memo.hits, memo.held });
    try std.testing.expect(requests >= 30);
}
