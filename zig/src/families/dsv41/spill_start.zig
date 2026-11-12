//! Rank 0's SSD tier at the engine's start: what the parsed settings decide before the model loads (the pinned staging
//! comes out of memory before the pool is sized, and every follower makes its landing buffers from the start-up
//! Settings), and the store name's common inputs gathered once the model is up. The tier never refuses the start: a
//! problem is an off reason, named on /health.
const std = @import("std");
const api = @import("engine_api");
const spill_tier = @import("spill_tier.zig");
const identity = @import("spill_identity.zig");

/// What the start does from the settings alone, before the model loads.
pub const Before = struct {
    /// The tier asked for (Spill.asked): /health shows it, on or off.
    asked: bool = false,
    /// The tier's settings while it may still open (nothing turned it off yet).
    settings: ?spill_tier.Settings = null,
    /// Why it is off already (a bad setting, or TF_DS_SPILL_OFF_REASON).
    off: ?[]const u8 = null,
    /// The landing buffers each follower makes (Settings.land_slots; 0: none).
    land_slots: u32 = 0,
};

pub fn before(sp: api.Spill, world: u32) Before {
    if (!sp.asked) return .{};
    if (sp.off) |why| return .{ .asked = true, .off = why };
    if (!sp.enabled()) return .{ .asked = true, .off = "TF_DS_SPILL_GIB and TF_DS_SPILL_DIR: not both set" };
    return .{ .asked = true, .settings = spill_tier.Settings.fromSpill(sp), .land_slots = if (world > 1) sp.land_slots else 0 };
}

/// The math switches in effect after the start-up agreement (Model.two's agreed ones, tri_index.idx_rb,
/// prompt.replay_floor, the kernel switches: pf_split as TF_DS_PF_SPLIT's text). pair_sum, owner_gather and negzero
/// are no longer switches: they keep the values the shipped settings gave them, so a store keeps its name. Every kernel
/// switch a prompt pass computes with is one, same-bytes ones too (native.zig's test ties prompt2d.kern_names to it).
pub const Math = struct { pair_sum: bool, qhc: bool, owner_gather: bool, idx_shard: bool, idx_rb: usize, negzero: bool, replay_floor: bool, owner_yx: bool, cat_partner: bool, front_owned: bool, ffn_codes: bool, pack_codes: bool, engram_owner_sum: bool, boundary_owned: bool, cat_codes: bool, pf_split: []const u8, woa_w8: bool, quant_once: bool, pf_gemm5: bool, group_tiles: bool, topk_tiles: bool, idx_proj_owned: bool, bf16_rows: bool, pf_run128: bool, pass_rows: usize, pass_from: usize, pass_tail: bool, decode_fuse: bool, attn_u: bool, idx_u: bool, mma_sf: bool, mma_sf_exact: bool, roci: bool = false, roci_collect: bool = false, idx_v: bool = false, gemm_w16: bool = false, assemble_quant: bool = false, attn_epilogue: bool = false, hc_stride: bool = false, hc_fused: bool = false, gemm_r2: bool = false, gemm_halves: bool = false, pf_raster: u32 = 0, attn_epi_fused: bool = false, qgate_canary: u8 = 0, hc_2s: bool = false };

/// Where the name's files are and the facts the start already holds.
pub const Facts = struct {
    /// The checkpoint (config.json) and the kernel kit (engram.json).
    model: std.Io.Dir,
    kit: std.Io.Dir,
    /// TF_DS_TOKEN_MAP's file (null: none set).
    token_map: ?[]const u8,
    /// The CUDA driver's version and the GPU's compute capability.
    driver: u32,
    capability: u32,
    /// spill_state's StateLayout.digest(), core/spill's namespace version and spill_state's format version.
    layout: identity.Digest,
    namespace_version: u32,
    state_version: u32,
};

/// The text `common` formats (it lives as long as the Common it returns).
pub const Texts = struct {
    engram: [64]u8 = undefined,
    math: [768]u8 = undefined,
    token_map: [64]u8 = undefined,
};

/// The math switches in effect as the store name's text (`m`'s every field, its name=value in Math's order).
pub fn mathText(buf: []u8, m: Math) ![]const u8 {
    const base = try mathBase(buf, m);
    // the switches below name themselves only when they are on: the names of states stored without them stay as they were
    var n = base.len;
    inline for (optional) |o| {
        if (@field(m, o[1])) {
            const x = try std.fmt.bufPrint(buf[n..], " {s}=1", .{o[0]});
            n += x.len;
        }
    }
    // the canary in effect (TF_DS_QGATE_CANARY: the head's 1, the body's 2 or 3), after the other optional switches, as the
    // head's 1 always was: a state kept under one canary is never restored under another
    if (m.qgate_canary != 0) {
        const x = try std.fmt.bufPrint(buf[n..], " qgate_canary={d}", .{m.qgate_canary});
        n += x.len;
    }
    if (m.pf_raster != 0) {
        const x = try std.fmt.bufPrint(buf[n..], " pf_raster={d}", .{m.pf_raster});
        n += x.len;
    }
    return buf[0..n];
}

/// The switches whose text joins the math's only when they are on: (the text's name, Math's field), in the order they join.
const optional = [_][2][]const u8{
    .{ "roci", "roci" },
    .{ "roci_collect", "roci_collect" },
    .{ "idx_v", "idx_v" },
    .{ "gemm_w16", "gemm_w16" },
    .{ "assemble_quant", "assemble_quant" },
    .{ "attn_epilogue", "attn_epilogue" },
    .{ "hc_stride", "hc_stride" },
    .{ "hc_fused", "hc_fused" },
    .{ "attn_epi_fused", "attn_epi_fused" },
    .{ "hc_2s", "hc_2s" },
    .{ "gemm_r2", "gemm_r2" },
    .{ "gemm_halves", "gemm_halves" },
};

fn mathBase(buf: []u8, m: Math) ![]const u8 {
    return std.fmt.bufPrint(buf, "pair_sum={d} qhc={d} owner_gather={d} idx_shard={d} idx_rb={d} negzero={d} replay_floor={d} owner_yx={d} cat_partner={d} front_owned={d} ffn_codes={d} pack_codes={d} engram_owner_sum={d} boundary_owned={d} cat_codes={d} pf_split={s} woa_w8={d} quant_once={d} pf_gemm5={d} group_tiles={d} topk_tiles={d} idx_proj_owned={d} bf16_rows={d} pf_run128={d} pass_rows={d} pass_from={d} pass_tail={d} decode_fuse={d} attn_u={d} idx_u={d} mma_sf={d} mma_sf_exact={d}", .{
        @intFromBool(m.pair_sum),    @intFromBool(m.qhc),         @intFromBool(m.owner_gather), @intFromBool(m.idx_shard),   m.idx_rb,                         @intFromBool(m.negzero),        @intFromBool(m.replay_floor), @intFromBool(m.owner_yx),
        @intFromBool(m.cat_partner), @intFromBool(m.front_owned), @intFromBool(m.ffn_codes),    @intFromBool(m.pack_codes),  @intFromBool(m.engram_owner_sum), @intFromBool(m.boundary_owned), @intFromBool(m.cat_codes),    m.pf_split,
        @intFromBool(m.woa_w8),      @intFromBool(m.quant_once),  @intFromBool(m.pf_gemm5),     @intFromBool(m.group_tiles), @intFromBool(m.topk_tiles),       @intFromBool(m.idx_proj_owned), @intFromBool(m.bf16_rows),    @intFromBool(m.pf_run128),
        m.pass_rows,                 m.pass_from,                 @intFromBool(m.pass_tail),    @intFromBool(m.decode_fuse), @intFromBool(m.attn_u),           @intFromBool(m.idx_u),          @intFromBool(m.mma_sf),       @intFromBool(m.mma_sf_exact),
    });
}

/// The store name's common inputs: the TF_DS_* names (`env.get`), config.json's and engram.json's content, the token
/// map's digest ("none" without one; empty when its file does not read), the CUDA facts, the math switches in effect,
/// the layout and the formats. A missing name or file goes in empty or as zeros, and identity.name names it.
pub fn common(io: std.Io, env: anytype, f: Facts, m: Math, t: *Texts) !identity.Common {
    const config = identity.fileDigest(io, f.model, "config.json") catch @as(identity.Digest, @splat(0));
    const engram_json = identity.fileDigest(io, f.kit, "engram.json") catch @as(identity.Digest, @splat(0));
    const token_map: []const u8 = if (f.token_map) |path| blk: {
        const d = identity.fileDigest(io, std.Io.Dir.cwd(), path) catch identity.fileDigest(io, f.model, path) catch break :blk "";
        t.token_map = std.fmt.bytesToHex(d, .lower);
        break :blk &t.token_map;
    } else "none";
    const engram = try std.fmt.bufPrint(&t.engram, "packed={d} owned={d} instream={d} aio={d}", .{
        @intFromBool(set(env, "TF_DS_ENGRAM_PACKED")),  @intFromBool(false), // (the owned records do not exist: the name keeps its text)
        @intFromBool(on(env, "TF_DS_ENGRAM_INSTREAM")), @intFromBool(on(env, "TF_DS_ENGRAM_AIO")),
    });
    const math = try mathText(&t.math, m);
    return .{
        .engine = text(env, "TF_DS_ENGINE_ID"),
        .model_revision = text(env, "TF_DS_MODEL_REVISION"),
        .config = config,
        .ablation = text(env, "TF_DS_ABLATION_ID"),
        .engram_revision = text(env, "TF_DS_ENGRAM_REVISION"),
        .engram_json = engram_json,
        .engram_switches = engram,
        .kit = text(env, "TF_DS_KIT_ID"),
        .token_map = token_map,
        .driver = f.driver,
        .capability = f.capability,
        .math = math,
        .layout = f.layout,
        .namespace_version = f.namespace_version,
        .state_version = f.state_version,
    };
}

fn text(env: anytype, comptime name: [:0]const u8) []const u8 {
    return std.mem.trim(u8, env.get(name) orelse "", " ");
}

/// Set at all, as the engine reads TF_DS_ENGRAM_PACKED (model.zig: getenv non-null).
fn set(env: anytype, comptime name: [:0]const u8) bool {
    return env.get(name) != null;
}

/// A switch on as the engine reads it (engram_rows.envOn, native.zig's TF_DS_ENGRAM_AIO): set and not exactly "0"
/// (an empty value is on).
fn on(env: anytype, comptime name: [:0]const u8) bool {
    const v = env.get(name) orelse return false;
    return !std.mem.eql(u8, v, "0");
}

/// The pinned staging the store's slot plan needs at least (core/spill slots.plan: a small slot a queued write and a
/// restore, then four block slots, two a direction), and how it is made, for /health and the off reason.
pub fn stagingMin(l: *const @import("spill_state.zig").StateLayout, smalls: u32, restores: u32, buf: []u8) struct { bytes: u64, formula: []const u8 } {
    const a = @import("core").spill.slots.alignment;
    const small = std.mem.alignForward(u64, @max(l.small_bytes, 1), a);
    const block = std.mem.alignForward(u64, @max(l.block_bytes, 1), a);
    const bytes = small * (smalls + restores) + 4 * block;
    const formula = std.fmt.bufPrint(buf, "small slot {d} x (smalls {d} + restores {d}) + 4 block slots x {d} = {d} bytes", .{ small, smalls, restores, block, bytes }) catch "";
    return .{ .bytes = bytes, .formula = formula };
}

/// The one line for a follower's log naming the tier's settings set there (TF_DS_SPILL_*, TF_SPILL_*), which it
/// ignores: the tier runs on rank 0 alone, and a follower's landing buffers come from rank 0's start-up Settings. Cut
/// to `buf` when long; null when none is set. `names`: the environment's names (std.process.Environ.Map.keys()).
pub fn followerIgnored(names: []const []const u8, rank: u32, buf: []u8) ?[]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    var n: usize = 0;
    for (names) |name| {
        if (!std.mem.startsWith(u8, name, "TF_DS_SPILL_") and !std.mem.startsWith(u8, name, "TF_SPILL_")) continue;
        w.print("{s}{s}", .{ if (n == 0) "" else ", ", name }) catch return w.buffered();
        n += 1;
    }
    if (n == 0) return null;
    w.print(" set on rank {d}: ignored there, the SSD tier runs on rank 0 alone (set them on rank 0)", .{rank}) catch {};
    return w.buffered();
}

/// TF_DS_SPILL_PROBE_TOKENS checked against this engine: the start probe runs that many positions through the encoder
/// before the API comes up, so they must be a multiple of the prompt chunk and at most the usable window (the parser
/// and the pool allow more; past the window, positions run past the RoPE tables). The off reason, or null if it fits.
pub fn probeTokens(tokens: u32, chunk: usize, usable: usize, buf: []u8) ?[]const u8 {
    if (tokens % chunk != 0) return std.fmt.bufPrint(buf, "TF_DS_SPILL_PROBE_TOKENS={d}: a multiple of the prompt chunk ({d})", .{ tokens, chunk }) catch "TF_DS_SPILL_PROBE_TOKENS: a multiple of the prompt chunk";
    if (tokens > usable) return std.fmt.bufPrint(buf, "TF_DS_SPILL_PROBE_TOKENS={d}: past the usable window ({d} tokens)", .{ tokens, usable }) catch "TF_DS_SPILL_PROBE_TOKENS: past the usable window";
    return null;
}
