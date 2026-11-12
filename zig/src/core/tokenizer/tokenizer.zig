//! TensorFold's tokenizer for Hugging Face tokenizer.json files: encode equals tokenizers' encode(text, add_special_tokens=False).
const std = @import("std");
const unicode = @import("unicode.zig");
const steps = @import("steps.zig");
const models = @import("model.zig");
const memo_mod = @import("memo.zig");
const Allocator = std.mem.Allocator;
pub const Memo = memo_mod.Memo;
/// A digest of what ids mean as loaded (identity.zig): what Recall keys its record by.
pub const identity = @import("identity.zig").identity;

pub const Kind = models.Kind;
pub const Error = models.Error || std.json.ParseError(std.json.Scanner) || error{UnsupportedAddedToken};

pub const AddedToken = struct {
    id: u32,
    content: []const u8,
    special: bool = false,
    lstrip: bool = false,
    rstrip: bool = false,
    normalized: bool = false,
};

/// Dense id-to-token table; entries put after loading stay owned by the caller.
pub const IdTable = struct {
    allocator: Allocator,
    tokens: std.ArrayList(?[]const u8) = .empty,

    pub fn put(t: *IdTable, id: u32, token: []const u8) Allocator.Error!void {
        if (id >= t.tokens.items.len) try t.tokens.appendNTimes(t.allocator, null, id + 1 - t.tokens.items.len);
        t.tokens.items[id] = token;
    }

    pub fn get(t: *const IdTable, id: u32) ?[]const u8 {
        return if (id < t.tokens.items.len) t.tokens.items[id] else null;
    }

    pub fn deinit(t: *IdTable) void {
        t.tokens.deinit(t.allocator);
    }
};

/// Byte trie over added-token contents for leftmost-longest matching, like tokenizers' Aho-Corasick split.
const Trie = struct {
    first: [256]u32 = @splat(0),
    edges: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    ends: std.ArrayList(?u32) = .empty,

    const Match = struct { start: usize, end: usize, token: u32 };

    fn insert(t: *Trie, a: Allocator, s: []const u8, token: u32) Allocator.Error!void {
        if (s.len == 0) return;
        if (t.ends.items.len == 0) try t.ends.append(a, null);
        var node: u32 = 0;
        for (s, 0..) |b, i| {
            const slot = if (i == 0) &t.first[b] else (try t.edges.getOrPutValue(a, @as(u64, node) << 8 | b, 0)).value_ptr;
            if (slot.* == 0) {
                try t.ends.append(a, null);
                slot.* = @intCast(t.ends.items.len - 1);
            }
            node = slot.*;
        }
        if (t.ends.items[node] == null) t.ends.items[node] = token;
    }

    fn next(t: *const Trie, s: []const u8, from: usize) ?Match {
        if (t.ends.items.len == 0) return null;
        for (from..s.len) |i| {
            var node = t.first[s[i]];
            if (node == 0) continue;
            var best: ?Match = null;
            var j = i + 1;
            while (true) {
                if (t.ends.items[node]) |token| best = .{ .start = i, .end = j, .token = token };
                if (j == s.len) break;
                node = t.edges.get(@as(u64, node) << 8 | s[j]) orelse break;
                j += 1;
            }
            if (best) |b| return b;
        }
        return null;
    }
};

const Segment = struct { start: usize, end: usize, id: ?u32 };

pub const Tokenizer = struct {
    allocator: Allocator,
    arena: ?*std.heap.ArenaAllocator = null,
    /// Token to id for the model vocabulary and every added token.
    vocab: std.StringHashMap(u32),
    id_to_token: IdTable,
    unk_id: ?u32 = null,
    added: []const AddedToken = &.{},
    flagged_specials: []const AddedToken = &.{},
    model: models.Model,
    normalizer: ?steps.Normalizer = null,
    pre_tokenizer: ?steps.PreTokenizer = null,
    decoder: ?steps.Decoder = null,
    raw: Trie = .{},
    normalized: Trie = .{},
    specials: std.StringHashMapUnmanaged(void) = .empty,
    /// Long segments' ids kept between encodes (encodeText; null: each encode tokenizes everything).
    memo: ?*Memo = null,

    pub fn deinit(t: *Tokenizer) void {
        t.vocab.deinit();
        t.id_to_token.deinit();
        if (t.memo) |m| m.destroy();
        if (t.arena) |arena| {
            arena.deinit();
            t.allocator.destroy(arena);
        }
        t.* = undefined;
    }

    /// Id of a token written out whole, such as a chat template's special token.
    pub fn specialTokenId(t: *const Tokenizer, text: []const u8) ?u32 {
        return t.vocab.get(text);
    }

    /// Token ids for `text`, without post-processor tokens; invalid UTF-8 reads as U+FFFD. Caller frees with `a`.
    pub fn encode(t: *const Tokenizer, a: Allocator, text: []const u8) Error![]u32 {
        // the scratch comes from the tokenizer's allocator, not `a`, so it goes back when the ids exist even when `a` is
        // an arena that lives on (a server request's); only the returned ids are `a`'s
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var word = std.heap.ArenaAllocator.init(t.allocator);
        defer word.deinit();
        var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() }, .word = &word };
        var valid = text;
        if (!std.unicode.utf8ValidateSlice(text)) {
            var fixed: std.ArrayList(u8) = .empty;
            try unicode.appendLossy(&fixed, ctx.a, text);
            valid = fixed.items;
        }
        var ids: std.ArrayList(u32) = .empty;
        for (try t.splitAdded(&ctx, &t.raw, valid)) |seg| {
            if (seg.id) |id| try ids.append(ctx.a, id) else if (seg.end > seg.start) try t.encodeText(&ctx, valid[seg.start..seg.end], seg.start == 0, &ids);
        }
        return a.dupe(u32, ids.items);
    }

    /// `encode` with no added token matched: text that spells one (a quoted marker) is ordinary text.
    pub fn encodePlain(t: *const Tokenizer, a: Allocator, text: []const u8) Error![]u32 {
        var arena = std.heap.ArenaAllocator.init(t.allocator); // as encode's
        defer arena.deinit();
        var word = std.heap.ArenaAllocator.init(t.allocator);
        defer word.deinit();
        var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() }, .word = &word };
        var valid = text;
        if (!std.unicode.utf8ValidateSlice(text)) {
            var fixed: std.ArrayList(u8) = .empty;
            try unicode.appendLossy(&fixed, ctx.a, text);
            valid = fixed.items;
        }
        var ids: std.ArrayList(u32) = .empty;
        if (valid.len > 0) try t.encodeTextWith(&ctx, valid, true, &ids, false);
        return a.dupe(u32, ids.items);
    }

    fn encodeText(t: *const Tokenizer, ctx: *steps.Context, text: []const u8, first: bool, ids: *std.ArrayList(u32)) Error!void {
        return t.encodeTextWith(ctx, text, first, ids, true);
    }

    fn encodeTextWith(t: *const Tokenizer, ctx: *steps.Context, text: []const u8, first: bool, ids: *std.ArrayList(u32), added: bool) Error!void {
        var zero: usize = if (first) unicode.decodeAt(text, 0).len else 0;
        const normalized = if (t.normalizer) |*n| try n.apply(ctx, text, &zero) else text;
        for (try t.splitAdded(ctx, &t.normalized, normalized)) |seg| {
            if (seg.id) |id| {
                try ids.append(ctx.a, id);
                continue;
            }
            if (seg.end == seg.start) continue;
            const part = normalized[seg.start..seg.end];
            const lead = seg.start < zero;
            // a long segment goes through the memo here, between every added token: a chat's markers can be ones
            // matched after normalization (DeepSeek-V4's are), and only here does a message's text stand alone, the
            // same in the prompt's render, the history's and the next turn's. Its ids come from its text and `first`
            // alone, the memo's key.
            const kept = if (plain.len >= memo_mod.min_bytes) t.memo else null;
            if (kept) |k| if (try k.append(piece.text, piece.first, ctx.a, ids)) continue;
            const from = ids.items.len;
            var pieces: std.ArrayList(steps.Piece) = .empty;
            try pieces.append(ctx.a, .{ .text = part, .first = lead });
            if (t.pre_tokenizer) |*p| try p.apply(ctx, &pieces);
            for (pieces.items) |piece| try t.model.tokenize(ctx, &t.vocab, piece.text, ids);
            if (kept) |k| k.put(part, lead, ids.items[from..]);
            } else try walk.run(t, 0, piece, ids, ctx.a);
            if (kept) |k| k.put(piece.text, piece.first, ids.items[from..]);
        }
    }

    /// Added-token split with tokenizers' lstrip and rstrip rules.
    fn splitAdded(t: *const Tokenizer, ctx: *steps.Context, trie: *const Trie, s: []const u8) Error![]const Segment {
        var out: std.ArrayList(Segment) = .empty;
        var done: usize = 0;
        var at: usize = 0;
        while (trie.next(s, at)) |hit| {
            at = hit.end;
            const token = t.added[hit.token];
            var start = hit.start;
            var stop = hit.end;
            if (token.lstrip) {
                var k = start;
                while (k > done) {
                    var c = k - 1;
                    while (s[c] & 0xC0 == 0x80) c -= 1;
                    if (!unicode.isSpace(unicode.decodeAt(s, c).cp)) break;
                    k = c;
                }
                start = k;
            }
            if (token.rstrip) while (stop < s.len) {
                const d = unicode.decodeAt(s, stop);
                if (!unicode.isSpace(d.cp)) break;
                stop += d.len;
            };
            if (start > done) try out.append(ctx.a, .{ .start = done, .end = start, .id = null });
            try out.append(ctx.a, .{ .start = start, .end = stop, .id = token.id });
            done = stop;
        }
        if (done < s.len) try out.append(ctx.a, .{ .start = done, .end = s.len, .id = null });
        return out.items;
    }

    /// Text for `ids` through the decoder chain; unknown ids are skipped. Caller frees with `a`.
    pub fn decode(t: *const Tokenizer, a: Allocator, ids: []const u32, skip_special_tokens: bool) Error![]u8 {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
        var tokens = try std.ArrayList([]const u8).initCapacity(ctx.a, ids.len);
        for (ids) |id| {
            const token = t.id_to_token.get(id) orelse continue;
            if (skip_special_tokens and t.specials.contains(token)) continue;
            tokens.appendAssumeCapacity(token);
        }
        if (t.decoder) |*d| return d.decode(a, &ctx, tokens.items);
        return std.mem.join(a, " ", tokens.items);
    }

        const masked = try t.maskAdded(a, text);
        defer if (masked) |m| a.free(m);
    /// Exact, possibly incomplete UTF-8 bytes; null for context-dependent decoders. Caller frees with `a`.
    pub fn tokenBytes(t: *const Tokenizer, a: Allocator, id: u32) Error!?[]u8 {
        const d = if (t.decoder) |*decoder| decoder else return null;
        if (!d.tokenLocal()) return null;
        const piece = t.id_to_token.get(id) orelse return try a.dupe(u8, "");
        if (t.specials.contains(piece)) return try a.dupe(u8, piece);
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() }, .raw_bytes = true };
        return try d.decode(a, &ctx, &.{piece});
    }
};

/// Loads `path/tokenizer.json`, or `path` itself when it names a .json file.
pub fn loadTokenizer(io: std.Io, a: Allocator, path: []const u8) !Tokenizer {
    const file = if (std.mem.endsWith(u8, path, ".json")) try a.dupe(u8, path) else try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{path});
    defer a.free(file);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(1 << 30));
    defer a.free(bytes);
    return parse(a, bytes);
}

pub fn parse(a: Allocator, json: []const u8) Error!Tokenizer {
    const arena = try a.create(std.heap.ArenaAllocator);
    arena.* = .init(a);
    var t = Tokenizer{ .allocator = a, .arena = arena, .vocab = .init(a), .id_to_token = .{ .allocator = a }, .model = .empty(.bpe) };
    errdefer t.deinit();
    const keep = arena.allocator();
    var scratch_arena = std.heap.ArenaAllocator.init(a);
    defer scratch_arena.deinit();
    const scratch = scratch_arena.allocator();
    var scanner = std.json.Scanner.initCompleteInput(scratch, json);
    const options = std.json.ParseOptions{ .max_value_len = json.len, .allocate = .alloc_always };
    var added: std.json.Value = .null;
    var have_model = false;
    if (try scanner.next() != .object_begin) return error.InvalidTokenizer;
    while (true) {
        const key = switch (try scanner.nextAlloc(scratch, .alloc_if_needed)) {
            .object_end => break,
            .string => |s| s,
            .allocated_string => |s| s,
            else => return error.InvalidTokenizer,
        };
        if (std.mem.eql(u8, key, "model")) {
            t.model = try models.load(keep, scratch, &scanner, &t.vocab, &t.id_to_token);
            have_model = true;
        } else if (std.mem.eql(u8, key, "added_tokens")) {
            added = try std.json.innerParse(std.json.Value, scratch, &scanner, options);
        } else if (std.mem.eql(u8, key, "normalizer")) {
            const v = try std.json.innerParse(std.json.Value, keep, &scanner, options);
            if (v != .null) t.normalizer = try steps.Normalizer.parse(keep, v);
        } else if (std.mem.eql(u8, key, "pre_tokenizer")) {
            const v = try std.json.innerParse(std.json.Value, keep, &scanner, options);
            if (v != .null) t.pre_tokenizer = try steps.PreTokenizer.parse(keep, v);
        } else if (std.mem.eql(u8, key, "decoder")) {
            const v = try std.json.innerParse(std.json.Value, keep, &scanner, options);
            if (v != .null) t.decoder = try steps.Decoder.parse(keep, v);
        } else try scanner.skipValue();
    }
    if (!have_model) return error.InvalidTokenizer;
    // Metaspace's "first" scheme needs original offsets, tracked here through Prepend, Replace and Strip only.
    if (t.pre_tokenizer) |*p| if (p.firstScheme() == .nested or (p.firstScheme() == .leading and t.normalizer != null and t.normalizer.?.hasForms()))
        return error.UnsupportedPreTokenizer;
    try addTokens(&t, keep, scratch, added);
    t.unk_id = if (t.model.unk_token) |u| t.model.id(&t.vocab, u) else null;
    return t;
}

/// Registers added tokens the way tokenizers' AddedVocabulary does: vocabulary ids first, then new ids past the model.
fn addTokens(t: *Tokenizer, keep: Allocator, scratch: Allocator, list: std.json.Value) Error!void {
    if (list == .null) return;
    if (list != .array) return error.InvalidTokenizer;
    const items = list.array.items;
    var ids = std.StringHashMap(u32).init(scratch);
    var max: ?u32 = null;
    for (items) |item| {
        const content = steps.string(item, "content") orelse return error.InvalidTokenizer;
        // tokenizers' single_word uses Unicode \w (Alphabetic, from PropList), which these tables do not carry.
        if (steps.flag(item, "single_word", false)) return error.UnsupportedAddedToken;
        if (content.len == 0) continue;
        if (steps.flag(item, "special", false)) try t.specials.put(keep, try keep.dupe(u8, content), {});
        if (ids.contains(content)) continue;
        const id = t.model.id(&t.vocab, content) orelse if (max) |m| (if (m >= t.model.size or t.model.size == 0) m + 1 else t.model.size) else t.model.size;
        max = @max(max orelse id, id);
        try ids.put(content, id);
    }
    var tokens: std.ArrayList(AddedToken) = .empty;
    var seen = std.StringHashMap(void).init(scratch);
    // tokenizers lists special tokens before the others; a content's first special entry sets its flags.
    for ([_]bool{ true, false }) |special_pass| for (items) |item| {
        const content = steps.string(item, "content").?;
        const special = steps.flag(item, "special", false);
        if (content.len == 0 or special != special_pass or seen.contains(content)) continue;
        if (!special and t.specials.contains(content)) continue;
        try seen.put(content, {});
        try tokens.append(keep, .{
            .id = ids.get(content).?,
            .content = try keep.dupe(u8, content),
            .special = special,
            .lstrip = steps.flag(item, "lstrip", false),
            .rstrip = steps.flag(item, "rstrip", false),
            .normalized = steps.flag(item, "normalized", !special),
        });
    };
    t.added = tokens.items;
    var specials: usize = 0;
    while (specials < tokens.items.len and tokens.items[specials].special) specials += 1;
    t.flagged_specials = tokens.items[0..specials];
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var ctx = steps.Context{ .a = arena.allocator(), .matcher = .{ .a = arena.allocator() } };
    for (tokens.items, 0..) |token, index| {
        if (t.model.id(&t.vocab, token.content) == null) {
            try t.vocab.put(token.content, token.id);
            try t.model.added_only.put(keep, token.content, {});
        }
        var zero: usize = 0;
        const text = if (token.normalized and t.normalizer != null) try keep.dupe(u8, try t.normalizer.?.apply(&ctx, token.content, &zero)) else token.content;
        // tokenizers matches and decodes a normalized added token by its normalized content.
        try t.id_to_token.put(token.id, text);
        try (if (token.normalized) &t.normalized else &t.raw).insert(keep, text, @intCast(index));
    }
}

test "byte-level BPE with merges, added tokens and decode" {
    const json =
        \\{"added_tokens":[{"id":7,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        \\ "normalizer":{"type":"NFC"},
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    const ids = try t.encode(a, "ab ab<|end|>c");
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 3, 5, 7, 6 }, ids);
    const text = try t.decode(a, ids, false);
    defer a.free(text);
    try std.testing.expectEqualStrings("ab ab<|end|>c", text);
    const skipped = try t.decode(a, ids, true);
    defer a.free(skipped);
    try std.testing.expectEqualStrings("ab abc", skipped);
    try std.testing.expectEqual(@as(?u32, 7), t.specialTokenId("<|end|>"));
    try std.testing.expectEqual(@as(usize, 1), t.flagged_specials.len);
    const marker = (try t.tokenBytes(a, 7)).?;
    defer a.free(marker);
    try std.testing.expectEqualStrings("<|end|>", marker);
    const unknown = (try t.tokenBytes(a, 99)).?;
    defer a.free(unknown);
    try std.testing.expectEqualStrings("", unknown);
    return deepseekWith(a,
        \\{"added_tokens":[{"id":0,"content":"<｜User｜>","special":true,"lstrip":false,"rstrip":false,"normalized":false,"single_word":false}],
        \\ "normalizer":null,
    , words);
}

/// deepseekLike with `head`'s added tokens and normalizer (the JSON before the pre-tokenizer); the vocabulary's ids from 1.
fn deepseekWith(a: Allocator, head: []const u8, words: []const []const u8) !Tokenizer {
    try w.writeAll(head);
}

test "with the memo, long segments encode as without it, and the second encode takes their kept ids" {
    const json =
        \\{"added_tokens":[{"id":7,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    for (0..300) |_| try text.appendSlice(a, "ab abc ");
    try text.appendSlice(a, "<|end|>");
    for (0..300) |_| try text.appendSlice(a, " ab cab");
    const plain = try t.encode(a, text.items);
    defer a.free(plain);
    t.memo = try Memo.create(a, 1 << 20);
    for (0..2) |_| {
        const kept = try t.encode(a, text.items);
        defer a.free(kept);
        try std.testing.expectEqualSlices(u32, plain, kept);
    }
    try std.testing.expectEqual(@as(u64, 2), t.memo.?.hits); // both segments, the second time
}

test "a kept segment's ids equal a fresh encode's wherever the segment sits" {
    const added =
        \\{"added_tokens":[{"id":20,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        \\ {"id":21,"content":"<|l|>","single_word":false,"lstrip":true,"rstrip":false,"normalized":false,"special":true},
        \\ {"id":22,"content":"<|r|>","single_word":false,"lstrip":false,"rstrip":true,"normalized":false,"special":true}],
    ;
    const kinds = [_][]const u8{
        added ++
            \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
            \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
            \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6,"x":7,"ï":8,"¿":9,"½":10},"merges":[["Ġ","a"],"a b","Ġa b"]}}
        ,
        added ++
            \\ "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true},
            \\ "model":{"type":"BPE","unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,
            \\  "vocab":{"<unk>":0,"a":1,"b":2,"▁":3,"ab":4,"▁a":5,"▁ab":6,"c":7,"x":8,"<0xEF>":9,"<0xBF>":10,"<0xBD>":11},"merges":["a b","▁ a","▁a b"]}}
        ,
    };
    const a = std.testing.allocator;
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(a);
    for (0..200) |i| try body.appendSlice(a, if (i % 3 == 0) "ab abc " else "cab ab ");
    const b = body.items;
    // first or not, after lstrip and rstrip tokens, with a leading space, around a lossy byte, repeated in one prompt
    const texts = [_][]const []const u8{
        &.{b},               &.{ "<|end|>", b }, &.{ b, "<|end|>", b }, &.{ "x  <|l|>", b },
        &.{ "<|r|>   ", b }, &.{ " ", b },       &.{ b, "\xff", b },    &.{ "<|end|> ", b, "<|r|>", b },
    };
    for (kinds, 0..) |json, kind| {
        var t = try parse(a, json);
        defer t.deinit();
        var fresh: [texts.len][]u32 = undefined;
        for (texts, &fresh) |parts, *ids| {
            const s = try std.mem.concat(a, u8, parts);
            defer a.free(s);
            ids.* = try t.encode(a, s);
        }
        defer for (fresh) |ids| a.free(ids);
        // the Metaspace tokenizer prepends to the first segment alone, so a memo blind to `first` fails below
        if (kind == 1) try std.testing.expect(!std.mem.eql(u32, fresh[0], fresh[1][1..]));
        t.memo = try Memo.create(a, 1 << 20);
        for (0..2) |round| for (0..texts.len) |k| {
            const i = if (round == 0) k else texts.len - 1 - k;
            const s = try std.mem.concat(a, u8, texts[i]);
            defer a.free(s);
            const kept = try t.encode(a, s);
            defer a.free(kept);
            try std.testing.expectEqualSlices(u32, fresh[i], kept);
        };
        try std.testing.expect(t.memo.?.hits >= 10); // every long segment of the second round, at least
    }
}

test "the memo keeps a message's text between markers matched after normalization, as a chat's are" {
    // DeepSeek-V4's chat markers are added tokens with "normalized": true: the prompt's render and the history's
    // differ past the message, and the message's text is the same segment in both only after the split on them
    const json =
        \\{"added_tokens":[{"id":7,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        \\ {"id":8,"content":"<|user|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},
        \\ {"id":9,"content":"<|bot|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(a);
    for (0..300) |_| try body.appendSlice(a, "ab abc ");
    const history = try std.mem.concat(a, u8, &.{ "<|end|><|user|>", body.items });
    defer a.free(history);
    const prompt = try std.mem.concat(a, u8, &.{ history, "<|bot|>" });
    defer a.free(prompt);
    const fresh_prompt = try t.encode(a, prompt);
    defer a.free(fresh_prompt);
    const fresh_history = try t.encode(a, history);
    defer a.free(fresh_history);
    try std.testing.expectEqual(@as(u32, 9), fresh_prompt[fresh_prompt.len - 1]);
    t.memo = try Memo.create(a, 1 << 20);
    const kept_prompt = try t.encode(a, prompt);
    defer a.free(kept_prompt);
    const kept_history = try t.encode(a, history);
    defer a.free(kept_history);
    try std.testing.expectEqualSlices(u32, fresh_prompt, kept_prompt);
    try std.testing.expectEqualSlices(u32, fresh_history, kept_history);
    try std.testing.expectEqual(@as(u64, 1), t.memo.?.hits); // the message's text, in the history's encode
}

/// DeepSeek-V4's added tokens as its tokenizer.json flags them (the sentence ends and the image token matched before
/// the normalizer, the chat markers after it) and its empty normalizer; their ids come from addTokens (past the model).
const v4_head =
    \\{"added_tokens":[
    \\ {"id":5000,"content":"<｜begin▁of▁sentence｜>","special":true,"lstrip":false,"rstrip":false,"normalized":false,"single_word":false},
    \\ {"id":5001,"content":"<｜end▁of▁sentence｜>","special":true,"lstrip":false,"rstrip":false,"normalized":false,"single_word":false},
    \\ {"id":5002,"content":"<｜deepseek_image｜>","special":false,"lstrip":false,"rstrip":false,"normalized":false,"single_word":false},
    \\ {"id":5003,"content":"<｜System｜>","special":false,"lstrip":false,"rstrip":false,"normalized":true,"single_word":false},
    \\ {"id":5004,"content":"<｜User｜>","special":false,"lstrip":false,"rstrip":false,"normalized":true,"single_word":false},
    \\ {"id":5005,"content":"<｜Assistant｜>","special":false,"lstrip":false,"rstrip":false,"normalized":true,"single_word":false},
    \\ {"id":5006,"content":"<think>","special":false,"lstrip":false,"rstrip":false,"normalized":true,"single_word":false},
    \\ {"id":5007,"content":"</think>","special":false,"lstrip":false,"rstrip":false,"normalized":true,"single_word":false},
    \\ {"id":5008,"content":"｜DSML｜","special":false,"lstrip":false,"rstrip":false,"normalized":true,"single_word":false}],
    \\ "normalizer":{"type":"Sequence","normalizers":[]},
;

const ChatMsg = struct { role: enum { system, user, tool, assistant }, text: []const u8, reasoning: []const u8 = "", call: []const u8 = "" };

/// `msgs` as DeepSeek-V4.1's chat template lays them out (the recipe's model/chat_template.jinja): the sentence start,
/// the effort line, the system text and its tools, each user turn with the tool results merged into it, each assistant
/// turn (its reasoning, text and DSML call, the sentence end), and the assistant header after a user turn the assistant
/// answers, or after the last one when `generation`.
fn chatV4(a: Allocator, msgs: []const ChatMsg, thinking: bool, generation: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "<｜begin▁of▁sentence｜>");
    if (thinking or msgs[0].role == .system) try out.appendSlice(a, "<｜System｜>");
    if (thinking) try out.appendSlice(a, "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n");
    for (msgs, 0..) |m, i| {
        switch (m.role) {
            .system => {
                try out.appendSlice(a, m.text);
                try out.appendSlice(a, "\n\n## Tools\n\nYou can invoke tools by writing a \"<｜DSML｜ calls>\" block:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n");
            },
            .user => {
                try out.appendSlice(a, "<｜User｜>");
                try out.appendSlice(a, m.text);
            },
            .tool => {
                const merged = i > 0 and (msgs[i - 1].role == .user or msgs[i - 1].role == .tool);
                try out.appendSlice(a, if (merged) "\n\n" else "<｜User｜>");
                try out.print(a, "<tool_result>{s}</tool_result>", .{m.text});
            },
            .assistant => {
                if (thinking) {
                    try out.appendSlice(a, m.reasoning);
                    try out.appendSlice(a, "</think>");
                }
                try out.appendSlice(a, m.text);
                if (m.call.len > 0) try out.print(a, "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">{s}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>", .{m.call});
                try out.appendSlice(a, "<｜end▁of▁sentence｜>");
            },
        }
        const user_turn = m.role == .user or m.role == .tool;
        const last = i + 1 == msgs.len;
        if (user_turn and ((last and generation) or (!last and msgs[i + 1].role == .assistant))) {
            try out.appendSlice(a, "<｜Assistant｜>");
            try out.appendSlice(a, if (thinking) "<think>" else "</think>");
        }
    }
    return out.toOwnedSlice(a);
}

const chat_words = [_][]const u8{ "the", "file", "reads", "a", "function", "returns", "error", "value", "test", "build", "server", "request", "tokens", "prompt", "memory", "and", "of", "is", "we", "should", "fix", "this", "line", "Let", "me", "check", "The", "output", "shows" };

/// A client's text of about `bytes` bytes as the server hands it to the template: sentences of chat_words with numbers,
/// CJK, code and now and then a written marker, masked as the server masks a client's text (maskAdded), then with
/// `images` an image's placeholder between its parts (the server's image path).
fn chatText(a: Allocator, t: *const Tokenizer, r: std.Random, bytes: usize, images: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    while (out.items.len < bytes) {
        const n = 3 + r.uintLessThan(usize, 12);
        for (0..n) |i| {
            if (i > 0) try out.append(a, ' ');
            try out.appendSlice(a, chat_words[r.uintLessThan(usize, chat_words.len)]);
        }
        switch (r.uintLessThan(u8, 14)) {
            0 => try out.print(a, " {d}", .{r.uintLessThan(u32, 10_000_000)}),
            1 => try out.appendSlice(a, " 日本語の説明、"),
            2 => try out.appendSlice(a, "\n```zig\nconst x = try f(a, 3);\n```\n"),
            3 => try out.appendSlice(a, " a page that spells <｜User｜> and </think>"),
            4 => try out.appendSlice(a, " <｜end▁of▁sentence｜> in a quote"),
            else => {},
        }
        try out.appendSlice(a, if (r.uintLessThan(u8, 5) == 0) ".\n\n" else ". ");
    }
    const masked = try t.maskAdded(a, out.items);
    defer if (masked) |m| a.free(m);
    const text = masked orelse out.items;
    if (!images) return a.dupe(u8, text);
    const cut = r.uintLessThan(usize, text.len + 1);
    return std.mem.concat(a, u8, &.{ text[0..cut], "\n\n<｜deepseek_image｜>\n\n", text[cut..] });
}

/// Six conversations of system, user, assistant and tool turns (sizes around the memo's 1 KiB floor, images, tool
/// calls, reasoning), each rendered at every request it makes: the prompt with the assistant header and the history
/// without, as server/prompt.zig's prepare renders them. Thinking on in every other conversation.
const ChatCorpus = struct {
    arena: std.heap.ArenaAllocator,
    texts: std.ArrayList([]const u8) = .empty, // prompt, history, prompt, history, ...

    fn init(gpa: Allocator, t: *const Tokenizer) !ChatCorpus {
        var c: ChatCorpus = .{ .arena = .init(gpa) };
        errdefer c.arena.deinit();
        const a = c.arena.allocator();
        var rng = std.Random.DefaultPrng.init(0x5eed);
        const r = rng.random();
        const sizes = [_]usize{ 0, 40, 700, 1023, 1024, 1025, 2500, 6000 };
        for (0..6) |conv| {
            var msgs: std.ArrayList(ChatMsg) = .empty;
            try msgs.append(a, .{ .role = .system, .text = try chatText(a, t, r, 2000 + 700 * conv, false) });
            for (0..4 + conv % 3) |turn| {
                try msgs.append(a, .{ .role = .user, .text = try chatText(a, t, r, sizes[(conv + turn) % sizes.len], (conv + turn) % 4 == 1) });
                for (0..2) |_| {
                    const calls = r.boolean();
                    try msgs.append(a, .{ .role = .assistant, .text = try chatText(a, t, r, sizes[r.uintLessThan(usize, sizes.len)], false), .reasoning = try chatText(a, t, r, sizes[r.uintLessThan(usize, sizes.len)], false), .call = if (calls) "src/server/prompt.zig" else "" });
                    if (!calls) break;
                    try msgs.append(a, .{ .role = .tool, .text = try chatText(a, t, r, 500 + r.uintLessThan(usize, 7000), turn == 1) });
                }
            }
            const thinking = conv % 2 == 0;
            for (1..msgs.items.len + 1) |k| {
                const end = msgs.items[k - 1].role;
                if (end != .user and end != .tool) continue;
                try c.texts.append(a, try chatV4(a, msgs.items[0..k], thinking, true));
                try c.texts.append(a, try chatV4(a, msgs.items[0..k], thinking, false));
            }
        }
        return c;
    }

    fn deinit(c: *ChatCorpus) void {
        c.arena.deinit();
    }
};

test "DeepSeek-V4 chats through the memo: every request's prompt and history ids are a fresh encode's" {
    const gpa = std.testing.allocator;
    var fresh = try deepseekWith(gpa, v4_head, &chat_words);
    defer fresh.deinit();
    const user = fresh.specialTokenId("<｜User｜>").?;
    var kept = try deepseekWith(gpa, v4_head, &chat_words);
    defer kept.deinit();
    kept.memo = try Memo.create(gpa, 64 << 20);
    var corpus = try ChatCorpus.init(gpa, &fresh);
    defer corpus.deinit();
    try std.testing.expect(corpus.texts.items.len >= 40);
    var requests: usize = 0;
    var i: usize = 0;
    while (i < corpus.texts.items.len) : (i += 2) {
        const before = kept.memo.?.hits;
        for (corpus.texts.items[i .. i + 2], 0..) |text, which| {
            const want = try fresh.encode(gpa, text);
            defer gpa.free(want);
            const got = try kept.encode(gpa, text);
            defer gpa.free(got);
            try std.testing.expectEqualSlices(u32, want, got);
            // the markers the template wrote are their tokens
            try std.testing.expect(std.mem.indexOfScalar(u32, want, user) != null);
            if (which == 1) try std.testing.expect(kept.memo.?.hits > before); // the history takes the prompt's segments
        }
        requests += 1;
    }
    try std.testing.expect(kept.memo.?.hits >= 2 * requests);
    try std.testing.expect(kept.memo.?.held <= 64 << 20);
}

const Encoder = struct {
    t: *const Tokenizer,
    texts: []const []const u8,
    want: []const []const u32,
    start: usize,
    bad: *std.atomic.Value(u32),

    fn run(e: Encoder) void {
        for (0..e.texts.len) |k| {
            const i = (e.start + 7 * k) % e.texts.len;
            const got = e.t.encode(std.heap.page_allocator, e.texts[i]) catch {
                _ = e.bad.fetchAdd(1, .monotonic);
                continue;
            };
            defer std.heap.page_allocator.free(got);
            if (!std.mem.eql(u32, e.want[i], got)) _ = e.bad.fetchAdd(1, .monotonic);
        }
    }
};

test "a memo too small for a chat (every segment dropped in turn), four threads at once: still a fresh encode's ids" {
    const gpa = std.testing.allocator;
    var fresh = try deepseekWith(gpa, v4_head, &chat_words);
    defer fresh.deinit();
    var kept = try deepseekWith(gpa, v4_head, &chat_words);
    defer kept.deinit();
    const cap = 24 << 10;
    kept.memo = try Memo.create(std.heap.page_allocator, cap);
    var corpus = try ChatCorpus.init(gpa, &fresh);
    defer corpus.deinit();
    const texts = corpus.texts.items;
    const want = try gpa.alloc([]const u32, texts.len);
    defer gpa.free(want);
    for (texts, want) |text, *w| w.* = try fresh.encode(gpa, text);
    defer for (want) |w| gpa.free(w);
    var bad: std.atomic.Value(u32) = .init(0);
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*th, k| th.* = try std.Thread.spawn(.{}, Encoder.run, .{Encoder{ .t = &kept, .texts = texts, .want = want, .start = k * 11, .bad = &bad }});
    for (threads) |th| th.join();
    try std.testing.expectEqual(@as(u32, 0), bad.load(.monotonic));
    try std.testing.expect(kept.memo.?.hits > 0);
    try std.testing.expect(kept.memo.?.held <= cap);
}

test {
    _ = @import("memo.zig");
}

test {
    _ = @import("identity.zig");
}
