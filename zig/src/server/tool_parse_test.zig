//! The tool_parse tests, as their own test binary: whole-reply parses and the close-call helpers.
const std = @import("std");
const json = @import("json");
const tool_parse = @import("tool_parse.zig");
const parse = tool_parse.parse;
const closeCall = tool_parse.closeCall;

test "families" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][2][]const u8{
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>", "{\"q\":\"x\"}" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>", "{\"n\":5}" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>", "{\"n\":7}" },
        .{ "<|tool_call>call:lookup{q:<|\"|>hi, there<|\"|>,n:3}<tool_call|>", "{\"q\":\"hi, there\",\"n\":3}" },
        .{ "{\"tool\":\"lookup\",\"query\":\"ping\"}", "{\"query\":\"ping\"}" },
    };
    for (cases) |c| {
        const r = try parse(a, c[0], tools, null);
        try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("arguments").?.string);
    }
    const kept = try parse(a, "hi <tool_call>{\"name\":\"other tool\"}</tool_call>", tools, null);
    try std.testing.expect(kept.calls == null);
    try std.testing.expectEqualStrings("hi <tool_call>{\"name\":\"other tool\"}</tool_call>", kept.content);
}

test "a call the end token left open closes when it parses whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][4][]const u8{ // the reply, what closes it, the call's arguments, the content left
        .{ "Looking.<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}\n", "</tool_call>", "{\"q\":\"x\"}", "Looking." },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":[\"x\"", "]}}</tool_call>", "{\"q\":[\"x\"]}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n", "</function></tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>\n<function=lookup>\n", "</function></tool_call>", "{}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n", "</tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value>", "</tool_call>", "{\"n\":7}", "" },
        .{ "<tool_call>{\"name\":\"lookup\"}</tool_call>\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":2}", "}</tool_call>", "{\"n\":2}", "" },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"<tool_call>\"}", "}</tool_call>", "{\"q\":\"<tool_call>\"}", "" },
    };
    for (cases) |c| {
        const close = try closeCall(a, c[0], tools);
        try std.testing.expectEqualStrings(c[1], close);
        const r = try parse(a, try std.mem.concat(a, u8, &.{ c[0], close }), tools, null);
        try std.testing.expectEqualStrings(c[2], r.calls.?[r.calls.?.len - 1].get("function").?.get("arguments").?.string);
        try std.testing.expectEqualStrings(c[3], r.content);
    }
}

test "a call left open that would not parse whole stays the reply's text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const replies = [_][]const u8{
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"fast ca", // inside a string
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":", // a key without its value
        "Trying.<tool_call>\n<function=lookup>\n<parameter=q>\nfast ca", // inside a parameter
        "Trying.<tool_call>\n<function=lookup>\n<parameter=n>\n5\n", // before its </parameter>
        "Trying.<tool_call>{\"name\":\"launch rocket!\",\"arguments\":{}}", // no function name
        "Trying.<tool_call>lookup<arg_key>n</arg_key><arg_value>7", // a GLM value left open
        "Trying.<tool_call>", // nothing written
        "Trying <tool_call> tags.<tool_call>{\"name\":\"lookup\"}", // an earlier opener would take the closer
    };
    for (replies) |text| {
        try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
        const r = try parse(a, text, tools, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(text, r.content);
    }
}

test "replies that end outside a call close nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const replies = [_][]const u8{
        "Done.",
        "<tool_call>{\"name\":\"lookup\"}</tool_call>",
        "<tool_call>\n<function=lookup>\n</function>\n</tool_call> Done.",
    };
    for (replies) |text| try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
}

test "a call to a tool the request did not declare stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const whole = [_][]const u8{
        "<tool_call><function=launch><parameter=when>now</parameter></function></tool_call>",
        "<tool_call>{\"name\": \"launch rocket!\", \"arguments\": {\"when\": \"now\"}}</tool_call>",
        "<tool_call>{\"name\": \"launch\", \"arguments\": {\"when\": NaN}}</tool_call>",
    };
    for (whole) |text| {
        const parallel = try parse(a, text, tools, null);
        try std.testing.expect(parallel.calls == null);
        try std.testing.expectEqualStrings(text, parallel.content);
        const single = try parse(a, text, tools, 1);
        try std.testing.expect(single.calls == null);
        try std.testing.expectEqualStrings("", single.content);
    }
    // an unclosed envelope is not a call in either mode, so the reply keeps it
    const open = "<tool_call>{\"name\":\"launch\",\"arguments\":{\"when\":\"now\"}";
    for ([_]?usize{ null, 1 }) |max_calls| {
        const k = try parse(a, open, tools, max_calls);
        try std.testing.expect(k.calls == null);
        try std.testing.expectEqualStrings(open, k.content);
    }
    const prose = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    try std.testing.expectEqualStrings("Launching.", (try parse(a, prose, tools, 1)).content);
    // an offered tool keeps its offered spelling; bare JSON naming one not offered stays content
    const offered = try parse(a, "<tool_call>{\"name\":\"LOOKUP\"}</tool_call>", tools, null);
    try std.testing.expectEqualStrings("lookup", offered.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expect((try parse(a, "{\"name\":\"launch\",\"arguments\":{}}", tools, null)).calls == null);
}

test "DeepSeek-V4.1's DSML calls, read as the served lane reads them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"weather\"}},{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const D = "\u{ff5c}DSML\u{ff5c}";
    // two invokes, string and JSON values, prose before the block
    const two = "Checking.\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"weather\">\n<" ++ D ++ " parameter name=\"city\" string=\"true\">Oslo</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"days\" string=\"false\">3</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n<" ++ D ++ " invoke name=\"LOOKUP\">\n<" ++ D ++ " parameter name=\"q\" string=\"false\">{\"a\": [1, 2.5, null]}</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const r = try parse(a, two, tools, null);
    try std.testing.expectEqualStrings("Checking.", r.content);
    try std.testing.expectEqual(@as(usize, 2), r.calls.?.len);
    try std.testing.expectEqualStrings("weather", r.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("{\"city\":\"Oslo\",\"days\":3}", r.calls.?[0].get("function").?.get("arguments").?.string);
    try std.testing.expectEqualStrings("lookup", r.calls.?[1].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("{\"q\":{\"a\":[1,2.5,null]}}", r.calls.?[1].get("function").?.get("arguments").?.string);
    // one call a reply: the second invoke drops out
    const single = try parse(a, two, tools, 1);
    try std.testing.expectEqual(@as(usize, 1), single.calls.?.len);
    // JSON that does not parse is the string; no string attribute is a string
    const loose = "<" ++ D ++ " calls><" ++ D ++ " invoke name=\"weather\"><" ++ D ++ " parameter name=\"x\" string=\"false\">not json</" ++ D ++ " parameter><" ++ D ++ " parameter name=\"y\">7</" ++ D ++ " parameter></" ++ D ++ " invoke></" ++ D ++ " calls>";
    try std.testing.expectEqualStrings("{\"x\":\"not json\",\"y\":\"7\"}", (try parse(a, loose, tools, null)).calls.?[0].get("function").?.get("arguments").?.string);
    // a tool the request did not offer is a call under the model's name, for the client to answer; one call a reply drops it
    const other = "<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"launch\">\n<" ++ D ++ " parameter name=\"when\" string=\"true\">now</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const kept = try parse(a, other, tools, null);
    try std.testing.expectEqualStrings("", kept.content);
    try std.testing.expectEqual(@as(usize, 1), kept.calls.?.len);
    try std.testing.expectEqualStrings("launch", kept.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("{\"when\":\"now\"}", kept.calls.?[0].get("function").?.get("arguments").?.string);
    const one = try parse(a, other, tools, 1);
    try std.testing.expectEqualStrings("", one.content);
    try std.testing.expect(one.calls == null);
    // a block with no invoke goes; an unclosed block stays text
    try std.testing.expectEqualStrings("Hi.", (try parse(a, "Hi.<" ++ D ++ " calls>\n</" ++ D ++ " calls>", tools, null)).content);
    const open = "Hi.<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"weather\">";
    try std.testing.expectEqualStrings(open, (try parse(a, open, tools, null)).content);
}

test "a DSML block is read from its own tags: <tool_call> text around it or in an argument moves nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"weather\"}},{\"type\":\"function\",\"function\":{\"name\":\"write\"}}]")).ok.array;
    const D = "\u{ff5c}DSML\u{ff5c}";
    const quoted = "a </tool_call> and <tool_call>{\"name\": \"weather\", \"arguments\": {}}</tool_call>";
    const block = "<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"write\">\n<" ++ D ++ " parameter name=\"content\" string=\"true\">" ++ quoted ++ "</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    // a <tool_call> in the prose before the block, and a </tool_call> and a whole call in its argument
    const r = try parse(a, "I will not use <tool_call> here.\n\n" ++ block, tools, null);
    try std.testing.expectEqualStrings("I will not use <tool_call> here.", r.content);
    try std.testing.expectEqual(@as(usize, 1), r.calls.?.len);
    try std.testing.expectEqualStrings("write", r.calls.?[0].get("function").?.get("name").?.string);
    const args = (try json.parse(a, r.calls.?[0].get("function").?.get("arguments").?.string)).ok;
    try std.testing.expectEqualStrings(quoted, args.get("content").?.string);
    // calls keep the reply's order: a <tool_call> call before the block comes first
    const both = try parse(a, "<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>\n" ++ block, tools, null);
    try std.testing.expectEqual(@as(usize, 2), both.calls.?.len);
    try std.testing.expectEqualStrings("weather", both.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("write", both.calls.?[1].get("function").?.get("name").?.string);
    // the call streamer reads only the text before the first block
    try std.testing.expectEqualStrings("Hi. ", tool_parse.beforeDsml("Hi. " ++ block ++ " more"));
    try std.testing.expectEqualStrings("no block", tool_parse.beforeDsml("no block"));
}

test "a <tool_call> whose payload is a DSML opener with no closer is no call and no crash, in either spelling" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const D = "\u{ff5c}DSML\u{ff5c}";
    inline for (.{ "<" ++ D ++ "tool_calls>", "<" ++ D ++ " calls>" }) |opener| {
        const text = "<tool_call>" ++ opener ++ "</tool_call>";
        for ([_]?usize{ null, 1 }) |quota| {
            const r = try parse(a, text, tools, quota);
            try std.testing.expect(r.calls == null or r.calls.?.len == 0);
            // (a malformed call stays text when calls are unlimited and drops out of a one-call reply)
            try std.testing.expectEqualStrings(if (quota == null) text else "", r.content);
        }
    }
}

test "a lone DSML opener inside a call's argument is the argument's text, in either spelling" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const D = "\u{ff5c}DSML\u{ff5c}";
    inline for (.{ "<" ++ D ++ "tool_calls>", "<" ++ D ++ " calls>" }) |opener| {
        const call = "{\"name\":\"lookup\",\"arguments\":{\"q\":\"" ++ opener ++ "\"}}";
        const inputs = [_][]const u8{ "<tool_call>" ++ call ++ "</tool_call>", call, "[" ++ call ++ "]", "```json\n" ++ call ++ "\n```" };
        for (inputs) |text| for ([_]?usize{ null, 1 }) |quota| {
            const r = try parse(a, text, tools, quota);
            try std.testing.expectEqual(@as(usize, 1), r.calls.?.len);
            const args = (try json.parse(a, r.calls.?[0].get("function").?.get("arguments").?.string)).ok;
            try std.testing.expectEqualStrings(opener, args.get("q").?.string);
            try std.testing.expectEqualStrings("", r.content);
        };
    }
}

/// One `write` call in a DSML block of either spelling (DeepSeek-V4.1's when `v41`), its `content` value as given.
fn dsmlWrite(a: std.mem.Allocator, v41: bool, value: []const u8) ![]const u8 {
    const D = "\u{ff5c}DSML\u{ff5c}";
    const s = if (v41) " " else "";
    const block = if (v41) "calls" else "tool_calls";
    return std.mem.concat(a, u8, &.{ "<", D, s, block, ">\n<", D, s, "invoke name=\"write\">\n<", D, s, "parameter name=\"content\" string=\"true\">", value, "</", D, s, "parameter>\n</", D, s, "invoke>\n</", D, s, block, ">" });
}

/// The streamed text never takes back what it showed, and it ends as the whole parse's content.
fn expectStreamAgrees(a: std.mem.Allocator, text: []const u8, content: []const u8) !void {
    const reply_text = @import("reply_text.zig");
    const final = try reply_text.hideToolCalls(a, text, true);
    try std.testing.expectEqualStrings(content, std.mem.trim(u8, final, " \t\r\n"));
    try expectNoRetraction(a, text);
}

/// Each streamed byte only adds to what was shown before it, up to the end of the reply.
fn expectNoRetraction(a: std.mem.Allocator, text: []const u8) !void {
    const reply_text = @import("reply_text.zig");
    var before: []const u8 = "";
    for (0..text.len + 2) |n| {
        const shown = if (n <= text.len) try reply_text.hideToolCalls(a, text[0..n], false) else try reply_text.hideToolCalls(a, text, true);
        if (!std.mem.startsWith(u8, shown, before)) {
            std.debug.print("after {d} bytes the stream showed {s}, taking back part of {s}\n", .{ n, shown, before });
            return error.TestUnexpectedResult;
        }
        before = try a.dupe(u8, shown);
    }
}

test "a DSML value that holds the block's own tags stays in the call, and none of them reaches the text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"write\"}}]")).ok.array;
    const D = "\u{ff5c}DSML\u{ff5c}";
    for ([_]bool{ true, false }) |v41| {
        const open = if (v41) "<" ++ D ++ " calls>" else "<" ++ D ++ "tool_calls>";
        const close = if (v41) "</" ++ D ++ " calls>" else "</" ++ D ++ "tool_calls>";
        // a file about DSML: the value holds the closer (#14), the opener (parsed before the change too), or both
        const values = [_][]const u8{
            try std.mem.concat(a, u8, &.{ "end: ", close }),
            try std.mem.concat(a, u8, &.{ "start: ", open }),
            try std.mem.concat(a, u8, &.{ open, " inner ", close }),
            try std.mem.concat(a, u8, &.{ close, " then ", open }),
        };
        for (values) |value| {
            const block = try dsmlWrite(a, v41, value);
            const want = try std.mem.concat(a, u8, &.{ "{\"content\":\"", value, "\"}" });
            const r = try parse(a, block, tools, null);
            try std.testing.expectEqual(@as(usize, 1), r.calls.?.len);
            try std.testing.expectEqualStrings(want, r.calls.?[0].get("function").?.get("arguments").?.string);
            try std.testing.expectEqualStrings("", r.content);
            try expectStreamAgrees(a, block, "");
            // prose around the block: before it streams at once, after it streams once the block has closed
            const around = try std.mem.concat(a, u8, &.{ "Writing.\n\n", block, "\nDone." });
            const ra = try parse(a, around, tools, null);
            try std.testing.expectEqual(@as(usize, 1), ra.calls.?.len);
            try std.testing.expectEqualStrings("Writing.\n\n\nDone.", ra.content);
            // this repository's stream leaves out the two newlines the V4.1 template writes before its calls block
            // (reply_text.calls' "\n\n" opener); the whole parse keeps them
            try expectStreamAgrees(a, around, if (v41) "Writing.\nDone." else "Writing.\n\n\nDone.");
        }
    }
}

test "around a DSML block: prose that quotes its closer, several blocks, both spellings, a reply cut inside one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"write\"}}]")).ok.array;
    const D = "\u{ff5c}DSML\u{ff5c}";
    for ([_]bool{ true, false }) |v41| {
        const close = if (v41) "</" ++ D ++ " calls>" else "</" ++ D ++ "tool_calls>";
        const plain = try dsmlWrite(a, v41, "plain");
        // prose after a complete block that quotes the closer: the call and the prose both stay, as before the change
        const quoted = try std.mem.concat(a, u8, &.{ plain, "\nA block ends with ", close, "." });
        const rq = try parse(a, quoted, tools, null);
        try std.testing.expectEqual(@as(usize, 1), rq.calls.?.len);
        try std.testing.expect(std.mem.indexOf(u8, rq.content, try std.mem.concat(a, u8, &.{ "ends with ", close })) != null);
        try expectStreamAgrees(a, quoted, rq.content);
        // two blocks with prose between them: two calls, the prose kept
        const twice = try std.mem.concat(a, u8, &.{ plain, "\nThen.\n", try dsmlWrite(a, v41, try std.mem.concat(a, u8, &.{ "end: ", close })) });
        const rt = try parse(a, twice, tools, null);
        try std.testing.expectEqual(@as(usize, 2), rt.calls.?.len);
        try std.testing.expectEqualStrings("Then.", rt.content);
        try expectStreamAgrees(a, twice, "Then.");
        // a reply cut inside a block whose value holds the closer: the first closer ends it, as it always did (the
        // parser drops the cut invoke), and the stream takes nothing back on the way there
        for ([_][]const u8{ try std.mem.concat(a, u8, &.{ "end: ", close }), "plain" }) |value| {
            const full = try dsmlWrite(a, v41, value);
            const cut = full[0 .. full.len - close.len];
            const rc = try parse(a, cut, tools, null);
            try std.testing.expect(rc.calls == null or rc.calls.?.len == 0);
            try expectNoRetraction(a, cut);
        }
        // a name that holds the closer reads as no structure: the first closer ends the block, as it always did, so the
        // complete invoke before it is still recovered, whether the reply ends in that name or goes on past it
        const named = try std.mem.concat(a, u8, &.{ if (v41) "<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"write\"></" ++ D ++ " invoke>\n<" ++ D ++ " invoke name=\"" else "<" ++ D ++ "tool_calls>\n<" ++ D ++ "invoke name=\"write\"></" ++ D ++ "invoke>\n<" ++ D ++ "invoke name=\"", close });
        const longer = try std.mem.concat(a, u8, &.{ named, if (v41) "\"></" ++ D ++ " invoke>" else "\"></" ++ D ++ "invoke>", close });
        for ([_][]const u8{ named, longer }) |text| {
            const rn = try parse(a, text, tools, null);
            if (v41) {
                try std.testing.expectEqual(@as(usize, 1), rn.calls.?.len);
                try std.testing.expectEqualStrings("write", rn.calls.?[0].get("function").?.get("name").?.string);
            }
            try expectNoRetraction(a, text);
        }
        // a block whose invoke never closes, then a whole block: the second block's call stays, as it always did (the
        // open invoke must not borrow the second block's closers)
        const inv = if (v41) "<" ++ D ++ " invoke name=\"" else "<" ++ D ++ "invoke name=\"";
        const invc = if (v41) "</" ++ D ++ " invoke>" else "</" ++ D ++ "invoke>";
        const opn = if (v41) "<" ++ D ++ " calls>" else "<" ++ D ++ "tool_calls>";
        const unclosed = try std.mem.concat(a, u8, &.{ opn, "\n", inv, "broken\">\n", close, "\n", opn, "\n", inv, "write\">", invc, "\n", close });
        const ru = try parse(a, unclosed, tools, null);
        if (v41) {
            try std.testing.expectEqual(@as(usize, 1), ru.calls.?.len);
            try std.testing.expectEqualStrings("write", ru.calls.?[0].get("function").?.get("name").?.string);
        }
        try expectNoRetraction(a, unclosed);
    }
    // a V4 block whose JSON value (string="false") holds a whole V4.1 block: the V4.1 call is kept, as it always was,
    // though the V4 block reads past its first closer (its value is not JSON, so reading it whole gives no call)
    const nested = "<" ++ D ++ "tool_calls>\n<" ++ D ++ "invoke name=\"write\">\n<" ++ D ++ "parameter name=\"content\" string=\"false\">\n</" ++ D ++ "tool_calls>\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"write\"></" ++ D ++ " invoke>\n</" ++ D ++ " calls>\n</" ++ D ++ "parameter>\n</" ++ D ++ "invoke>\n</" ++ D ++ "tool_calls>";
    const rn = try parse(a, nested, tools, null);
    try std.testing.expectEqual(@as(usize, 1), rn.calls.?.len);
    try std.testing.expectEqualStrings("write", rn.calls.?[0].get("function").?.get("name").?.string);
    try expectNoRetraction(a, nested);
    // both spellings in one reply: one call each
    const mixed = try std.mem.concat(a, u8, &.{ try dsmlWrite(a, true, "one"), "\nand\n", try dsmlWrite(a, false, "two") });
    const rm = try parse(a, mixed, tools, null);
    try std.testing.expectEqual(@as(usize, 2), rm.calls.?.len);
    try std.testing.expectEqualStrings("and", rm.content);
    try expectStreamAgrees(a, mixed, "and");
}
