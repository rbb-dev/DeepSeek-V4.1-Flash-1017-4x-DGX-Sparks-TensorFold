//! DeepSeek-V4.1-Flash dimensions from config.json (its text_config), checked against the shapes our kernels serve.
const std = @import("std");

/// The 40 target layers and the 3 DSpark blocks after them.
pub const max_layers = 48;
/// Layer ids a config lists (KV sources, index sources, Engram, DSpark taps).
pub const max_ids = 8;

/// Why a check refused the checkpoint, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    /// Keep the reason, cut short if it is long.
    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

/// A short list of layer ids.
pub const Ids = struct {
    items: [max_ids]u16 = @splat(0),
    len: usize = 0,

    pub fn slice(self: *const Ids) []const u16 {
        return self.items[0..self.len];
    }

    pub fn has(self: Ids, id: usize) bool {
        for (self.items[0..self.len]) |x| if (x == id) return true;
        return false;
    }
};

/// The checkpoint's weight format: exl3, an EXL3 quant (mul1); native, DeepSeek's own release (FP8 E4M3 linears with
/// UE8M0 scales on 32 x 32 blocks, MXFP4 routed experts, bf16 elsewhere), served by native.zig.
pub const Format = enum { exl3, native };

pub const Config = struct {
    format: Format = .exl3,
    layers: usize,
    draft_layers: usize,
    hidden: usize,
    vocab: usize,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    q_lora: usize,
    o_lora: usize,
    o_groups: usize,
    window: usize,
    experts: usize,
    top_k: usize,
    expert_width: usize,
    shared_experts: usize,
    routed_scaling: f32,
    swiglu_limit: f32,
    eps: f32,
    /// Compression ratio a layer (0: window only, 1: every token, 2: every second), targets then DSpark blocks.
    ratios: [max_layers]u8 = @splat(0),
    kv_sources: Ids = .{},
    index_sources: Ids = .{},
    index_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    candidate_source: usize,
    candidate_blocks: usize,
    candidate_block: usize,
    hc: usize,
    hc_iters: usize,
    hc_eps: f32,
    engram_layers: Ids = .{},
    engram_rows: [2]u64 = .{ 0, 0 },
    engram_heads: usize,
    engram_head_dim: usize,
    engram_ngram: usize,
    engram_vocab: usize,
    engram_pad: u32,
    engram_compressed_vocab: usize,
    dspark_block: usize,
    dspark_noise: u32,
    dspark_taps: Ids = .{},
    markov_rank: usize,
    draft_experts: usize,
    draft_top_k: usize,
    max_position: usize,
    rope_theta: f64,
    rope_factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    original_max_position: usize,
    compress_rope_theta: f64,
    bos: u32,
    eos: u32,
    image_token: ?u32 = null,
    vision: bool = false,

    /// `dir`/config.json.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        var why: Why = .{};
        const c = parse(gpa, text, &why) catch |e| {
            if (why.len > 0) std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
        checkShapes(c, &why) catch |e| {
            std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
        return c;
    }
};

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse {
        std.log.err("config.json has no {s}", .{key});
        return error.BadConfig;
    };
}

fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try field(o, key);
    if (v != .integer or v.integer < 0) return error.BadConfig;
    return @intCast(v.integer);
}

fn number(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (try field(o, key)) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn ids(o: std.json.ObjectMap, key: []const u8) !Ids {
    const v = try field(o, key);
    if (v != .array or v.array.items.len > max_ids) return error.BadConfig;
    var out: Ids = .{};
    for (v.array.items) |x| {
        if (x != .integer or x.integer < 0) return error.BadConfig;
        out.items[out.len] = @intCast(x.integer);
        out.len += 1;
    }
    return out;
}

/// config.json's text; a refusal's reason goes to `why`.
pub fn parse(gpa: std.mem.Allocator, text: []const u8, why: *Why) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const top = parsed.value.object;
    const mt = try field(top, "model_type");
    if (mt != .string or !std.mem.eql(u8, mt.string, "deepseek_v41")) {
        why.set("config.json's model_type is {f}, not deepseek_v41", .{std.json.fmt(mt, .{})});
        return error.NotDeepSeekV41;
    }
    const format = try quantization(top, why);
    const o = if (top.get("text_config")) |t| (if (t == .object) t.object else top) else top;
    const rope_v = try field(o, "rope_scaling");
    if (rope_v != .object) return error.BadConfig;
    const rope = rope_v.object;
    var c = Config{
        .format = format,
        .layers = try int(o, "num_hidden_layers"),
        .draft_layers = try int(o, "num_nextn_predict_layers"),
        .hidden = try int(o, "hidden_size"),
        .vocab = try int(o, "vocab_size"),
        .heads = try int(o, "num_attention_heads"),
        .head_dim = try int(o, "head_dim"),
        .rope_dim = try int(o, "qk_rope_head_dim"),
        .q_lora = try int(o, "q_lora_rank"),
        .o_lora = try int(o, "o_lora_rank"),
        .o_groups = try int(o, "o_groups"),
        .window = try int(o, "sliding_window"),
        .experts = try int(o, "n_routed_experts"),
        .top_k = try int(o, "num_experts_per_tok"),
        .expert_width = try int(o, "moe_intermediate_size"),
        .shared_experts = try int(o, "n_shared_experts"),
        .routed_scaling = @floatCast(try number(o, "routed_scaling_factor")),
        .swiglu_limit = @floatCast(try number(o, "swiglu_limit")),
        .eps = @floatCast(try number(o, "rms_norm_eps")),
        .kv_sources = try ids(o, "kv_source_layer_ids"),
        .index_sources = try ids(o, "index_source_layer_ids"),
        .index_heads = try int(o, "index_n_heads"),
        .index_head_dim = try int(o, "index_head_dim"),
        .index_topk = try int(o, "index_topk"),
        .candidate_source = try int(o, "candidate_source_layer_id"),
        .candidate_blocks = try int(o, "candidate_topk_blocks"),
        .candidate_block = try int(o, "candidate_block_size"),
        .hc = try int(o, "hc_mult"),
        .hc_iters = try int(o, "hc_sinkhorn_iters"),
        .hc_eps = @floatCast(try number(o, "hc_eps")),
        .engram_layers = try ids(o, "engram_layer_ids"),
        .engram_heads = try int(o, "engram_n_heads"),
        .engram_head_dim = try int(o, "engram_head_dim"),
        .engram_ngram = try int(o, "engram_max_ngram_size"),
        .engram_vocab = try int(o, "engram_vocab_size"),
        .engram_pad = @intCast(try int(o, "engram_pad_token_id")),
        .engram_compressed_vocab = try int(o, "engram_compressed_vocab_size"),
        .dspark_block = try int(o, "dspark_block_size"),
        .dspark_noise = @intCast(try int(o, "dspark_noise_token_id")),
        .dspark_taps = try ids(o, "dspark_target_layer_ids"),
        .markov_rank = try int(o, "dspark_markov_rank"),
        .draft_experts = try int(o, "dspark_n_routed_experts"),
        .draft_top_k = try int(o, "dspark_num_experts_per_tok"),
        .max_position = try int(o, "max_position_embeddings"),
        .rope_theta = try number(o, "rope_theta"),
        .rope_factor = try number(rope, "factor"),
        .beta_fast = try number(rope, "beta_fast"),
        .beta_slow = try number(rope, "beta_slow"),
        .original_max_position = try int(rope, "original_max_position_embeddings"),
        .compress_rope_theta = try number(o, "compress_rope_theta"),
        .bos = @intCast(try int(top, "bos_token_id")),
        .eos = @intCast(try int(top, "eos_token_id")),
    };
    const ratios_v = try field(o, "compress_ratios");
    if (ratios_v != .array) return error.BadConfig;
    const ratios = ratios_v.array.items;
    if (c.layers + c.draft_layers > max_layers or ratios.len != c.layers + c.draft_layers) {
        why.set("config.json lists {d} compress_ratios for {d} layers and {d} DSpark blocks", .{ ratios.len, c.layers, c.draft_layers });
        return error.BadConfig;
    }
    for (ratios, 0..) |r, i| {
        if (r != .integer or r.integer < 0 or r.integer > 2) return error.BadConfig;
        c.ratios[i] = @intCast(r.integer);
    }
    const rows_v = try field(o, "engram_num_embeddings");
    if (rows_v != .array or rows_v.array.items.len != c.engram_layers.len or rows_v.array.items.len > 2) return error.BadConfig;
    for (rows_v.array.items, 0..) |r, i| {
        if (r != .integer or r.integer <= 0) return error.BadConfig;
        c.engram_rows[i] = @intCast(r.integer);
    }
    if (top.get("image_token_id")) |v| if (v == .integer) {
        c.image_token = @intCast(v.integer);
    };
    c.vision = top.get("vision_config") != null;
    return c;
}

/// The checkpoints our kernels read: an EXL3 quant (quant_method exl3, the mul1 codebook), or DeepSeek's own release
/// (quant_method fp8, 32 x 32 weight blocks, ue8m0 scales, fp4 experts).
fn quantization(top: std.json.ObjectMap, why: *Why) !Format {
    const q = top.get("quantization_config") orelse {
        why.set("config.json declares no quantization_config; the DeepSeek-V4.1 kernels read EXL3 or DeepSeek's FP8 / FP4 checkpoints", .{});
        return error.UnsupportedQuantization;
    };
    const m = if (q == .object) q.object.get("quant_method") else null;
    if (m != null and m.? == .string and std.mem.eql(u8, m.?.string, "fp8")) {
        const o = q.object;
        const blk = o.get("weight_block_size");
        const blk_ok = blk != null and blk.? == .array and blk.?.array.items.len == 2 and
            blk.?.array.items[0] == .integer and blk.?.array.items[0].integer == 32 and
            blk.?.array.items[1] == .integer and blk.?.array.items[1].integer == 32;
        const fmt = o.get("scale_fmt");
        const fmt_ok = fmt != null and fmt.? == .string and std.mem.eql(u8, fmt.?.string, "ue8m0");
        const ed = o.get("expert_dtype");
        const ed_ok = ed != null and ed.? == .string and std.mem.eql(u8, ed.?.string, "fp4");
        if (!blk_ok or !fmt_ok or !ed_ok) {
            why.set("config.json's fp8 quantization is not DeepSeek-V4.1's (32 x 32 blocks, ue8m0 scales, fp4 experts)", .{});
            return error.UnsupportedQuantization;
        }
        return .native;
    }
    if (m == null or m.? != .string or !std.mem.eql(u8, m.?.string, "exl3")) {
        why.set("config.json's quant_method is neither exl3 nor fp8; the DeepSeek-V4.1 kernels read EXL3 or DeepSeek's FP8 / FP4 checkpoints", .{});
        return error.UnsupportedQuantization;
    }
    const cb = q.object.get("codebook");
    if (cb == null or cb.? != .string or !std.mem.eql(u8, cb.?.string, "mul1")) {
        why.set("config.json's EXL3 codebook is not mul1; the served kernels were built for mul1", .{});
        return error.UnsupportedQuantization;
    }
    return .exl3;
}

/// The shapes the served kernels and our TP2 split (DESIGN.md section 2) are built for.
pub fn checkShapes(c: Config, why: *Why) !void {
    const ok = c.layers == 40 and c.draft_layers == 3 and c.hidden == 5120 and c.vocab == 129280 and c.heads == 64 and
        c.head_dim == 512 and c.rope_dim == 64 and c.q_lora == 1280 and c.o_lora == 1024 and c.o_groups == 8 and
        c.window == 128 and c.experts == 384 and c.top_k == 6 and c.expert_width == 2304 and c.shared_experts == 1 and
        c.index_heads == 32 and c.index_head_dim == 128 and c.index_topk == 512 and c.candidate_blocks == 2048 and
        c.candidate_block == 8 and c.hc == 4 and c.engram_layers.len == 2 and c.engram_heads == 8 and
        c.engram_head_dim == 256 and c.dspark_block == 5 and c.markov_rank == 256 and c.draft_experts == 128 and
        c.draft_top_k == 3;
    if (!ok) {
        why.set("these DeepSeek-V4.1 shapes differ from the ones the served kernels and the TP2 split were built for", .{});
        return error.UnsupportedShapes;
    }
}

/// What the RoPE tables are cut for: the YaRN factor and the largest position it reaches (original_max_position x factor).
pub const Rope = struct { factor: f64, max_position: usize };

/// TF_DS_YARN_FACTOR: unset or empty is the checkpoint's own factor; a number 1..1024 asks for another one (the kit's
/// RoPE tables must have been cut for it: checkRope).
pub fn yarnFactorAsked(text: ?[]const u8) error{BadYarnFactor}!?f64 {
    const t = text orelse return null;
    if (t.len == 0) return null;
    const f = std.fmt.parseFloat(f64, t) catch return error.BadYarnFactor;
    if (!(f >= 1.0 and f <= 1024.0)) return error.BadYarnFactor;
    return f;
}

/// The "factor" of a kit's rope.json, null when it names none (a kit cut for the checkpoint's own factor).
pub fn kitRopeFactor(gpa: std.mem.Allocator, text: []const u8) !?f64 {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const v = parsed.value.object.get("factor") orelse return null;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

/// The factor the engine runs and the positions it reaches: the checkpoint's own (rope_scaling.factor, max_position_embeddings)
/// unless TF_DS_YARN_FACTOR asks for another, which reaches original_max_position x that factor.
pub fn ropeRange(c: *const Config, asked: ?f64) Rope {
    const f = asked orelse return .{ .factor = c.rope_factor, .max_position = c.max_position };
    return .{ .factor = f, .max_position = @intFromFloat(@round(f * @as(f64, @floatFromInt(c.original_max_position)))) };
}

/// The start's two length checks: the kit's RoPE tables were cut for the factor the engine runs (a kit that names none was cut
/// for the checkpoint's), and a stream's context does not pass the positions that factor reaches.
pub fn checkRope(c: *const Config, asked: ?f64, kit_factor: ?f64, context: usize, why: *Why) !Rope {
    const r = ropeRange(c, asked);
    const kf = kit_factor orelse c.rope_factor;
    if (kf != r.factor) {
        why.set("the kit's RoPE tables are cut for YaRN factor {d}, the engine runs factor {d} (TF_DS_YARN_FACTOR; unset is the checkpoint's {d}): use the kit and the setting together", .{ kf, r.factor, c.rope_factor });
        return error.RopeFactorMismatch;
    }
    if (context > r.max_position) {
        why.set("the context {d} is past the {d} positions YaRN factor {d} reaches over {d} (the checkpoint's {d}): TF_DS_YARN_FACTOR and a kit cut for it", .{ context, r.max_position, r.factor, c.original_max_position, c.rope_factor });
        return error.ContextPastRope;
    }
    return r;
}

/// CUDA's grid: at most 65,535 programs along y (and z).
pub const grid_yz_max = 65_535;

/// The longest context the decode indexer reaches: `_index_score` runs a program a `tile`-key tile along CUDA's grid y over
/// an index layer's keys (a round's bucket, which stops at the context, over the layer's compress ratio), so each index
/// layer holds at most 65,535 tiles. Our checkpoint's ratio-1 layers (every token a key): 65,535 x 64 = 4,194,240 tokens.
pub fn indexWindowMax(c: *const Config, tile: usize) usize {
    var most: usize = std.math.maxInt(usize);
    for (c.index_sources.items[0..c.index_sources.len]) |l| {
        const r: usize = c.ratios[l];
        if (r != 0) most = @min(most, grid_yz_max * tile * r);
    }
    return most;
}

/// A context that is no power of two is a round's top bucket (round.zig bucketFor stops there), and the sizes a decode round
/// derives from it (an index layer's keys, their 64-key tiles, the candidate pool's stride) are integer arguments of Triton
/// variants the kit captured at power-of-two buckets, each such integer divisible by 16 (cuda/aot.zig matches: a div16
/// variant takes only a value divisible by 16). A multiple of 65,536 keeps them so down to bucket / 4,096: 4,128,768
/// (63 x 65,536) answered 10 / 10; 4,194,240 read its whole prompt, then its tile count 65,535 (odd) found no
/// variant at the first round.
pub const bucket_step = 65_536;

/// Whether a context keeps the kit's indexer variants (see bucket_step).
pub fn contextKeepsVariants(context: usize) bool {
    return context == 0 or std.math.isPowerOfTwo(context) or context % bucket_step == 0;
}

/// The longest context at or under `context` that keeps them: a multiple of 65,536, or below 65,536 a power of two.
pub fn variantContextBelow(context: usize) usize {
    if (context >= bucket_step) return context / bucket_step * bucket_step;
    return if (context == 0) 0 else std.math.floorPowerOfTwo(usize, context);
}

/// The start's checks of the context against the decode indexer: within indexWindowMax (past it a round's first indexer launch
/// is refused by the driver), and a power of two or a multiple of 65,536 (else the first round finds no kit variant).
pub fn checkIndexWindow(c: *const Config, context: usize, tile: usize, why: *Why) !void {
    const most = indexWindowMax(c, tile);
    if (context > most) {
        const works = variantContextBelow(most);
        why.set("the context {d} is past {d}: the decode indexer scores an index layer's keys in {d}-key tiles along CUDA's grid y, at most {d} of them (compress ratio 1: every token a key): CONTEXT {d} works", .{ context, most, tile, grid_yz_max, works });
        return error.ContextPastIndexGrid;
    }
    if (!contextKeepsVariants(context)) {
        why.set("the context {d} is neither a power of two nor a multiple of {d}: it is a round's top bucket, and the kit's indexer variants take its tile counts divisible by 16 (else the first reply fails after the whole prompt): CONTEXT {d} works", .{ context, bucket_step, variantContextBelow(context) });
        return error.ContextOffBucketGrid;
    }
}

/// The fields our checkpoint's config.json has (DeepSeek-V4.1-Flash, EXL3 2.9 bpw mul1), trimmed to what parse reads.
pub const test_config =
    \\{"model_type": "deepseek_v41", "bos_token_id": 0, "eos_token_id": 1, "image_token_id": 129264,
    \\ "quantization_config": {"quant_method": "exl3", "codebook": "mul1"},
    \\ "text_config": {"vocab_size": 129280, "hidden_size": 5120, "moe_intermediate_size": 2304,
    \\  "num_hidden_layers": 40, "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64,
    \\  "q_lora_rank": 1280, "o_lora_rank": 1024, "o_groups": 8, "swiglu_limit": 10.0, "rms_norm_eps": 1e-20,
    \\  "max_position_embeddings": 1048576, "rope_theta": 10000,
    \\  "rope_scaling": {"factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536},
    \\  "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6, "routed_scaling_factor": 1.5,
    \\  "sliding_window": 128,
    \\  "compress_ratios": [0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    \\  "compress_rope_theta": 160000, "kv_source_layer_ids": [2, 8, 14, 20],
    \\  "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36], "index_n_heads": 32, "index_head_dim": 128,
    \\  "index_topk": 512, "candidate_source_layer_id": 20, "candidate_topk_blocks": 2048, "candidate_block_size": 8,
    \\  "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "engram_layer_ids": [1, 14],
    \\  "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4, "engram_vocab_size": 16000000,
    \\  "engram_n_heads": 8, "engram_head_dim": 256, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99092,
    \\  "num_nextn_predict_layers": 3, "dspark_block_size": 5, "dspark_noise_token_id": 128799,
    \\  "dspark_target_layer_ids": [37, 38, 39], "dspark_markov_rank": 256, "dspark_n_routed_experts": 128,
    \\  "dspark_num_experts_per_tok": 3},
    \\ "vision_config": {"num_hidden_layers": 32}}
;

test "parse our checkpoint's config and accept its shapes" {
    var why: Why = .{};
    const c = try parse(std.testing.allocator, test_config, &why);
    try checkShapes(c, &why);
    try std.testing.expectEqual(@as(u8, 2), c.ratios[2]);
    try std.testing.expectEqual(@as(u8, 1), c.ratios[39]);
    try std.testing.expectEqual(@as(u8, 0), c.ratios[42]);
    try std.testing.expect(c.index_sources.has(36) and !c.index_sources.has(37));
    try std.testing.expectEqual(@as(u64, 384016682), c.engram_rows[1]);
    try std.testing.expectEqual(@as(?u32, 129264), c.image_token);
    try std.testing.expect(c.vision);
    try std.testing.expectEqual(@as(f32, 1e-20), c.eps);
}

test "the context stops where an index layer's 64-key tiles fill CUDA's grid y" {
    var why: Why = .{};
    const c = try parse(std.testing.allocator, test_config, &why);
    // ratio 1 on index layers 20, 24, 28, 32, 36: 65,535 tiles of 64 keys
    try std.testing.expectEqual(@as(usize, 4_194_240), indexWindowMax(&c, 64));
    try checkIndexWindow(&c, 1_048_576, 64, &why);
    try checkIndexWindow(&c, 2_097_152, 64, &why);
    // 4,128,768 = 63 x 65,536: 64,512 tiles, a 4M run that answered 10 / 10
    try checkIndexWindow(&c, 4_128_768, 64, &why);
    // 4,194,304 = 65,536 tiles: a 4M start served its prompt, then the driver refused the first round's launch
    try std.testing.expectError(error.ContextPastIndexGrid, checkIndexWindow(&c, 4_194_304, 64, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "CONTEXT 4128768 works") != null);
    // 4,194,240 = 65,535 tiles, odd: within the grid, but the first round found no kit variant
    try std.testing.expectError(error.ContextOffBucketGrid, checkIndexWindow(&c, 4_194_240, 64, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "CONTEXT 4128768 works") != null);
    try std.testing.expectError(error.ContextOffBucketGrid, checkIndexWindow(&c, 1_500_000, 64, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "CONTEXT 1441792 works") != null);
    try std.testing.expect(contextKeepsVariants(0) and contextKeepsVariants(4096) and contextKeepsVariants(3 * 65_536));
    try std.testing.expect(!contextKeepsVariants(65_536 + 4096) and !contextKeepsVariants(5000));
    try std.testing.expectEqual(@as(usize, 4096), variantContextBelow(5000));
    // index layers of ratio 2 only: twice as far
    var c2 = c;
    for (c2.index_sources.items[0..c2.index_sources.len]) |l| c2.ratios[l] = 2;
    try std.testing.expectEqual(@as(usize, 8_388_480), indexWindowMax(&c2, 64));
}

test "refuse other model types, other quantization and other shapes" {
    const a = std.testing.allocator;
    var why: Why = .{};
    const other = try std.mem.replaceOwned(u8, a, test_config, "\"deepseek_v41\"", "\"deepseek_v4\"");
    defer a.free(other);
    try std.testing.expectError(error.NotDeepSeekV41, parse(a, other, &why));
    const fp8 = try std.mem.replaceOwned(u8, a, test_config, "\"quant_method\": \"exl3\"", "\"quant_method\": \"fp8\"");
    defer a.free(fp8);
    try std.testing.expectError(error.UnsupportedQuantization, parse(a, fp8, &why)); // fp8 without DeepSeek's blocks
    const int8 = try std.mem.replaceOwned(u8, a, test_config, "\"quant_method\": \"exl3\"", "\"quant_method\": \"int8\"");
    defer a.free(int8);
    try std.testing.expectError(error.UnsupportedQuantization, parse(a, int8, &why));
    const mcg = try std.mem.replaceOwned(u8, a, test_config, "\"mul1\"", "\"mcg\"");
    defer a.free(mcg);
    try std.testing.expectError(error.UnsupportedQuantization, parse(a, mcg, &why));
    const wide = try std.mem.replaceOwned(u8, a, test_config, "\"num_experts_per_tok\": 6", "\"num_experts_per_tok\": 8");
    defer a.free(wide);
    try std.testing.expectError(error.UnsupportedShapes, checkShapes(try parse(a, wide, &why), &why));
}

test "DeepSeek's own release (FP8 linears, FP4 experts) parses as the native format" {
    const a = std.testing.allocator;
    var why: Why = .{};
    const native = try std.mem.replaceOwned(u8, a, test_config, "{\"quant_method\": \"exl3\", \"codebook\": \"mul1\"}", "{\"quant_method\": \"fp8\", \"activation_scheme\": \"dynamic\", \"weight_block_size\": [32, 32], \"scale_fmt\": \"ue8m0\", \"expert_dtype\": \"fp4\"}");
    defer a.free(native);
    const c = try parse(a, native, &why);
    try checkShapes(c, &why);
    try std.testing.expectEqual(Format.native, c.format);
    try std.testing.expectEqual(Format.exl3, (try parse(a, test_config, &why)).format);
    const blk128 = try std.mem.replaceOwned(u8, a, native, "[32, 32]", "[128, 128]");
    defer a.free(blk128);
    try std.testing.expectError(error.UnsupportedQuantization, parse(a, blk128, &why));
}

test "a refusal's reason is cut at the buffer, never past it" {
    var why: Why = .{};
    const long: [400]u8 = @splat('x');
    why.set("{s}", .{&long});
    try std.testing.expectEqual(@as(usize, 320), why.text().len);
}

test "the YaRN factor: unset is the checkpoint's, 32 reaches 2,097,152, a kit and a setting must agree" {
    const a = std.testing.allocator;
    var why: Why = .{};
    const c = try parse(a, test_config, &why);
    // baseline: nothing set, a kit that names no factor, the 1M context
    const r0 = try checkRope(&c, try yarnFactorAsked(null), null, 1_048_576, &why);
    try std.testing.expectEqual(@as(f64, 16), r0.factor);
    try std.testing.expectEqual(@as(usize, 1_048_576), r0.max_position);
    try std.testing.expectEqual(@as(?f64, null), try yarnFactorAsked(""));
    // asking for the checkpoint's own factor changes nothing
    const r16 = try checkRope(&c, try yarnFactorAsked("16"), null, 1_048_576, &why);
    try std.testing.expectEqual(@as(usize, 1_048_576), r16.max_position);
    // 1M refuses past 1,048,576 with the checkpoint's factor; 2M passes with 32 and a kit that says 32
    try std.testing.expectError(error.ContextPastRope, checkRope(&c, null, null, 2_097_152, &why));
    const r32 = try checkRope(&c, try yarnFactorAsked("32"), 32.0, 2_097_152, &why);
    try std.testing.expectEqual(@as(usize, 2_097_152), r32.max_position);
    try std.testing.expectError(error.ContextPastRope, checkRope(&c, try yarnFactorAsked("32"), 32.0, 2_097_153, &why));
    // a kit and a setting that disagree never start (either way)
    try std.testing.expectError(error.RopeFactorMismatch, checkRope(&c, try yarnFactorAsked("32"), null, 1_048_576, &why));
    try std.testing.expectError(error.RopeFactorMismatch, checkRope(&c, null, 32.0, 1_048_576, &why));
    try std.testing.expect(why.len != 0);
    try std.testing.expectError(error.BadYarnFactor, yarnFactorAsked("abc"));
    try std.testing.expectError(error.BadYarnFactor, yarnFactorAsked("0.5"));
    // rope.json: a factor field, integer or float; none
    try std.testing.expectEqual(@as(?f64, 32.0), try kitRopeFactor(a, "{\"rows\": 2162688, \"half\": 32, \"factor\": 32}"));
    try std.testing.expectEqual(@as(?f64, 32.5), try kitRopeFactor(a, "{\"factor\": 32.5}"));
    try std.testing.expectEqual(@as(?f64, null), try kitRopeFactor(a, "{\"rows\": 1114112, \"half\": 32}"));
}
