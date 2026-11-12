//! Rank 0's SSD tier at the start (spill_start.zig): what the settings decide before the model loads, and the
//! store name's inputs gathered after it; every problem a named off reason, never a refused start.
const std = @import("std");
const api = @import("engine_api");
const start = @import("spill_start.zig");
const identity = @import("spill_identity.zig");
const io = std.testing.io;

const Env = struct {
    pairs: []const [2][]const u8,
    pub fn get(e: Env, name: [:0]const u8) ?[]const u8 {
        for (e.pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

fn spill(pairs: []const [2][]const u8, buf: []u8) api.Spill {
    return api.Spill.fromEnv(Env{ .pairs = pairs }, buf);
}

test "before the model loads: not asked is nothing; off is named with no staging and no landing slots; on gives both" {
    var buf: [256]u8 = undefined;
    const none = start.before(spill(&.{}, &buf), 4);
    try std.testing.expect(!none.asked and none.settings == null and none.off == null and none.land_slots == 0);
    const bad = start.before(spill(&.{.{ "TF_DS_SPILL_GIB", "x" }}, &buf), 4);
    try std.testing.expect(bad.asked and bad.settings == null and bad.land_slots == 0);
    try std.testing.expect(std.mem.startsWith(u8, bad.off.?, "TF_DS_SPILL_GIB=x: "));
    const recipe = start.before(spill(&.{.{ "TF_DS_SPILL_OFF_REASON", "the watchdog's tier-off flag" }}, &buf), 4);
    try std.testing.expectEqualStrings("the watchdog's tier-off flag", recipe.off.?);
    try std.testing.expectEqual(@as(u32, 0), recipe.land_slots);
    const on = start.before(spill(&.{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "/spill" }, .{ "TF_DS_SPILL_LAND_SLOTS", "3" } }, &buf), 4);
    try std.testing.expect(on.asked and on.off == null);
    try std.testing.expectEqualStrings("/spill", on.settings.?.dir);
    try std.testing.expectEqual(@as(u32, 3), on.land_slots);
    try std.testing.expectEqual(@as(u32, 0), start.before(spill(&.{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "/spill" } }, &buf), 1).land_slots); // one rank: nothing to land
}

const recipe_env = [_][2][]const u8{
    .{ "TF_DS_ENGINE_ID", "native-12f3f2c3b4bf" },
    .{ "TF_DS_MODEL_REVISION", "deepseek-ai/DeepSeek-V4.1-Flash-1017@abc" },
    .{ "TF_DS_ABLATION_ID", "keys@def" },
    .{ "TF_DS_ENGRAM_REVISION", "deepseek-ai/DeepSeek-V4.1-Flash-1017@abc" },
    .{ "TF_DS_KIT_ID", "kit@ghi" },
    .{ "TF_DS_ENGRAM_PACKED", "/engram/packed" },
    .{ "TF_DS_ENGRAM_OWNED", "1" },
    .{ "TF_DS_ENGRAM_AIO", "1" },
};

const math: start.Math = .{ .pair_sum = true, .qhc = true, .owner_gather = true, .idx_shard = true, .idx_rb = 2, .negzero = false, .replay_floor = true, .owner_yx = true, .cat_partner = true, .front_owned = true, .ffn_codes = true, .pack_codes = true, .engram_owner_sum = true, .boundary_owned = true, .cat_codes = true, .pf_split = "C7", .woa_w8 = true, .quant_once = true, .pf_gemm5 = true, .group_tiles = true, .topk_tiles = true, .idx_proj_owned = true, .bf16_rows = true, .pf_run128 = true, .pass_rows = 4096, .pass_from = 8192, .pass_tail = true, .decode_fuse = true, .attn_u = true, .idx_u = true, .mma_sf = true, .mma_sf_exact = true, .roci = true, .roci_collect = true, .idx_v = true, .gemm_w16 = true, .assemble_quant = true, .attn_epilogue = true, .hc_stride = true, .hc_fused = true, .attn_epi_fused = true, .gemm_r2 = true, .gemm_halves = true, .qgate_canary = 1, .hc_2s = true, .pf_raster = 8 };

fn facts(tmp: *std.testing.TmpDir) start.Facts {
    return .{ .model = tmp.dir, .kit = tmp.dir, .token_map = "tokens.bin", .driver = 13000, .capability = 121, .layout = @splat(7), .namespace_version = 2, .state_version = 2 };
}

test "the math text holds every switch at the widest TF_DS_IDX_RB" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    var widest = math;
    widest.idx_rb = std.math.maxInt(usize);
    var texts: start.Texts = .{};
    const c = try start.common(io, Env{ .pairs = &recipe_env }, facts(&tmp), widest, &texts);
    try std.testing.expect(std.mem.indexOf(u8, c.math, " engram_owner_sum=1 boundary_owned=1 cat_codes=1 ") != null);
    try std.testing.expect(std.mem.indexOf(u8, c.math, "idx_rb=18446744073709551615 ") != null);
}

test "the store name's inputs: the recipe's names, the model's and the kit's files, the CUDA facts, the math in effect" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\": \"deepseek_v41\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.json", .data = "{\"layers\": [2, 15]}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokens.bin", .data = "token map bytes" });
    var texts: start.Texts = .{};
    const c = try start.common(io, Env{ .pairs = &recipe_env }, facts(&tmp), math, &texts);
    try std.testing.expectEqualStrings("native-12f3f2c3b4bf", c.engine);
    try std.testing.expectEqualStrings("keys@def", c.ablation);
    try std.testing.expectEqualStrings("kit@ghi", c.kit);
    try std.testing.expectEqualStrings("packed=1 owned=0 instream=0 aio=1", c.engram_switches);
    // the engine's own readings: an empty switch is on, "0" alone is off
    var texts2: start.Texts = .{};
    const odd = try start.common(io, Env{ .pairs = &.{ .{ "TF_DS_ENGRAM_OWNED", "" }, .{ "TF_DS_ENGRAM_AIO", "0" }, .{ "TF_DS_ENGRAM_PACKED", "" } } }, facts(&tmp), math, &texts2);
    try std.testing.expectEqualStrings("packed=1 owned=0 instream=0 aio=0", odd.engram_switches);
    try std.testing.expectEqualStrings("pair_sum=1 qhc=1 owner_gather=1 idx_shard=1 idx_rb=2 negzero=0 replay_floor=1 owner_yx=1 cat_partner=1 front_owned=1 ffn_codes=1 pack_codes=1 engram_owner_sum=1 boundary_owned=1 cat_codes=1 pf_split=C7 woa_w8=1 quant_once=1 pf_gemm5=1 group_tiles=1 topk_tiles=1 idx_proj_owned=1 bf16_rows=1 pf_run128=1 pass_rows=4096 pass_from=8192 pass_tail=1 decode_fuse=1 attn_u=1 idx_u=1 mma_sf=1 mma_sf_exact=1 roci=1 roci_collect=1 idx_v=1 gemm_w16=1 assemble_quant=1 attn_epilogue=1 hc_stride=1 hc_fused=1 attn_epi_fused=1 hc_2s=1 gemm_r2=1 gemm_halves=1 qgate_canary=1 pf_raster=8", c.math);
    // every field of the math in its text
    inline for (@typeInfo(start.Math).@"struct".field_names) |f| try std.testing.expect(std.mem.indexOf(u8, c.math, f ++ "=") != null);
    // every switch at its longest value fits the text
    var longest = math;
    longest.idx_rb = std.math.maxInt(usize);
    var texts3: start.Texts = .{};
    _ = try start.common(io, Env{ .pairs = &recipe_env }, facts(&tmp), longest, &texts3);
    try std.testing.expectEqual(@as(u32, 121), c.capability);
    var want: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("{\"model_type\": \"deepseek_v41\"}", &want, .{});
    try std.testing.expectEqualSlices(u8, &want, &c.config);
    std.crypto.hash.sha2.Sha256.hash("token map bytes", &want, .{});
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(want, .lower), c.token_map);
    // every rank's weights, and the name; a rank without its digest is named
    const four = [_]identity.Digest{ @splat(1), @splat(2), @splat(3), @splat(4) };
    var missing: identity.Missing = .{};
    _ = try identity.name(c, &four, &missing);
    var w = four;
    w[3] = @splat(0);
    try std.testing.expectError(error.MissingIdentity, identity.name(c, &w, &missing));
    var buf: [120]u8 = undefined;
    try std.testing.expectEqualStrings("rank 3: no weight digest (TF_DS_RANK_CACHE_SHA)", missing.text(&buf));
}

test "a missing file or name goes in empty and is named by the store's name, never a failed start" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    var texts: start.Texts = .{};
    // no engram.json and no token map file: zero digests and "none" (the name says which is missing)
    const c = try start.common(io, Env{ .pairs = recipe_env[0..4] }, facts(&tmp), math, &texts);
    try std.testing.expect(std.mem.allEqual(u8, &c.engram_json, 0));
    try std.testing.expectEqualStrings("", c.kit); // TF_DS_KIT_ID unset
    const four = [_]identity.Digest{ @splat(1), @splat(2), @splat(3), @splat(4) };
    var missing: identity.Missing = .{};
    try std.testing.expectError(error.MissingIdentity, identity.name(c, &four, &missing));
    try std.testing.expectEqualStrings("TF_DS_KIT_ID", missing.what);
    var with_kit = c;
    with_kit.kit = "kit@ghi";
    try std.testing.expectError(error.MissingIdentity, identity.name(with_kit, &four, &missing));
    try std.testing.expectEqualStrings("TF_DS_TOKEN_MAP", missing.what);
    with_kit.token_map = "none";
    try std.testing.expectError(error.MissingIdentity, identity.name(with_kit, &four, &missing));
    try std.testing.expectEqualStrings("engram.json's digest", missing.what);
}

test "the staging minimum: the store's slot plan at these smalls and restores, with its formula" {
    const spill_state = @import("spill_state.zig");
    const shapes = [_]spill_state.LayerShape{ .{ .ratio = 2, .kv = true, .idx = true, .raw = true }, .{ .ratio = 1, .kv = true, .idx = true } };
    const l = try spill_state.StateLayout.init(&shapes, .{ .block_tokens = 16, .window = 8, .ring_extra = 4, .raw_rows = 4, .head_dim = 64, .index_head_dim = 64 }, .replicated);
    var buf: [200]u8 = undefined;
    const m = start.stagingMin(&l, 4, 2, &buf);
    const core = @import("core");
    const small = std.mem.alignForward(u64, l.small_bytes, core.spill.slots.alignment);
    const block = std.mem.alignForward(u64, l.block_bytes, core.spill.slots.alignment);
    try std.testing.expectEqual(small * 6 + 4 * block, m.bytes);
    // the store's own plan takes exactly that much, and refuses a byte less
    _ = try core.spill.slots.plan(m.bytes, l.block_bytes, l.small_bytes, 4, 2);
    try std.testing.expectError(error.StagingTooSmall, core.spill.slots.plan(m.bytes - 1, l.block_bytes, l.small_bytes, 4, 2));
    try std.testing.expect(std.mem.startsWith(u8, m.formula, "small slot "));
    try std.testing.expect(std.mem.indexOf(u8, m.formula, " x (smalls 4 + restores 2) + 4 block slots x ") != null);
}

test "the tier's settings set on a follower by hand are named in one line and ignored there" {
    var buf: [256]u8 = undefined;
    const names = [_][]const u8{ "PATH", "TF_DS_SPILL_GIB", "TF_DS_RANK_CACHE_SHA", "TF_SPILL_TURNS", "TF_DS_REPLAY_FLOOR", "TF_DS_SPILLX" };
    try std.testing.expectEqualStrings("TF_DS_SPILL_GIB, TF_SPILL_TURNS set on rank 2: ignored there, the SSD tier runs on rank 0 alone (set them on rank 0)", start.followerIgnored(&names, 2, &buf).?);
    // a follower's own names (its weight digest, the floor every rank shares) are not the tier's
    try std.testing.expect(start.followerIgnored(&.{ "TF_DS_RANK_CACHE_SHA", "TF_DS_REPLAY_FLOOR", "HOME" }, 1, &buf) == null);
    try std.testing.expect(start.followerIgnored(&.{}, 3, &buf) == null);
    // too many to fit: the line is cut, still one line, never a failure
    var small: [40]u8 = undefined;
    const cut = start.followerIgnored(&.{ "TF_DS_SPILL_GIB", "TF_DS_SPILL_DIR", "TF_DS_SPILL_MIN_FREE_GIB" }, 1, &small).?;
    try std.testing.expect(cut.len <= small.len and std.mem.startsWith(u8, cut, "TF_DS_SPILL_GIB, "));
}

test "TF_DS_SPILL_PROBE_TOKENS: a multiple of the prompt chunk, at most the usable window, else the tier off, named" {
    var buf: [160]u8 = undefined;
    try std.testing.expect(start.probeTokens(8192, 2048, 1048526, &buf) == null);
    try std.testing.expect(std.mem.indexOf(u8, start.probeTokens(8000, 2048, 1048526, &buf).?, "multiple of the prompt chunk (2048)") != null);
    const past = start.probeTokens(2097152, 2048, 1048526, &buf).?; // passes the parser and the pool, not the window
    try std.testing.expect(std.mem.startsWith(u8, past, "TF_DS_SPILL_PROBE_TOKENS=2097152: "));
    try std.testing.expect(std.mem.indexOf(u8, past, "1048526") != null);
}

test "the math text names TF_DS_ROCI, TF_DS_ROCI_COLLECT, TF_DS_IDX_V, TF_DS_GEMM_W16, TF_DS_ASSEMBLE_QUANT, TF_DS_ATTN_EPILOGUE, TF_DS_HC_STRIDE, TF_DS_HC_FUSED, TF_DS_ATTN_EPI_FUSED, TF_DS_HC_2S, TF_DS_GEMM_R2 and TF_DS_GEMM_HALVES and TF_DS_GEMM_HALVES only when they are on: the names of states stored without it do not change" {
    var buf: [1024]u8 = undefined;
    var off = math;
    off.roci = false;
    off.roci_collect = false;
    off.idx_v = false;
    off.gemm_w16 = false;
    off.assemble_quant = false;
    off.attn_epilogue = false;
    off.hc_stride = false;
    off.hc_fused = false;
    off.attn_epi_fused = false;
    off.hc_2s = false;
    off.gemm_r2 = false;
    off.qgate_canary = 0;
    off.pf_raster = 0;
    const text = try start.mathText(&buf, off);
    try std.testing.expect(std.mem.indexOf(u8, text, "roci") == null);
    try std.testing.expect(std.mem.endsWith(u8, text, "mma_sf_exact=1"));
    var buf2: [1024]u8 = undefined;
    const on = try start.mathText(&buf2, math);
    try std.testing.expectEqualStrings(text, on[0..text.len]);
    try std.testing.expectEqualStrings(" roci=1 roci_collect=1 idx_v=1 gemm_w16=1 assemble_quant=1 attn_epilogue=1 hc_stride=1 hc_fused=1 attn_epi_fused=1 hc_2s=1 gemm_r2=1 gemm_halves=1 qgate_canary=1 pf_raster=8", on[text.len..]);
}

test "the math text names TF_DS_PF_RASTER with its group only when it is on, after the other optional switches" {
    var off_buf: [1024]u8 = undefined;
    var on_buf: [1024]u8 = undefined;
    var m = math;
    m.pf_raster = 0;
    const off = try start.mathText(&off_buf, m);
    try std.testing.expect(std.mem.indexOf(u8, off, "pf_raster") == null);
    for ([_]u32{ 2, 4, 8, 16 }) |g| {
        m.pf_raster = g;
        const on = try start.mathText(&on_buf, m);
        try std.testing.expectEqualStrings(off, on[0..off.len]);
        var want: [16]u8 = undefined;
        try std.testing.expectEqualStrings(try std.fmt.bufPrint(&want, " pf_raster={d}", .{g}), on[off.len..]);
    }
}
    on.qgate_canary = 1;

test "the canary in the tier's math is a number: the head's 1 keeps its old name, the body's 2 and 3 name themselves, and none is the same text as before" {
    var base_buf: [1024]u8 = undefined;
    var m = math;
    m.pf_raster = 0; // (its group comes last: the canary joins before it)
    m.qgate_canary = 0;
    const none = try start.mathText(&base_buf, m);
    try std.testing.expect(std.mem.indexOf(u8, none, "qgate_canary") == null);
    for ([_]u8{ 1, 2, 3 }) |c| {
        var buf: [1024]u8 = undefined;
        m.qgate_canary = c;
        const t = try start.mathText(&buf, m);
        var want: [32]u8 = undefined;
        // after the other optional switches, before pf_raster
        try std.testing.expect(std.mem.indexOf(u8, t, try std.fmt.bufPrint(&want, " qgate_canary={d}", .{c})) != null);
        try std.testing.expectEqualStrings(none, t[0..none.len]);
    }
}
