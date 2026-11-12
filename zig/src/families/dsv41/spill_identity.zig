//! The store's name: rank 0 names its one store from everything the stored bytes depend on, the inputs in Common and
//! every rank's weight digest in rank order, so a change to any of them gives another namespace and a store never
//! restores into other weights. One SHA-256 over a domain tag, a version and each part's length and bytes, pinned by
//! a test. A missing input is named and keeps the tier off for that start; it never refuses the engine's start.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Digest = [32]u8;

/// Bump on any change to how the name is built: every stored point then belongs to another namespace.
pub const version: u32 = 1;

/// The most ranks a name covers.
pub const max_world = 8;

/// What every rank's bytes depend on, as rank 0 holds it (the TF_DS_* names come from rank 0's environment).
pub const Common = struct {
    /// TF_DS_ENGINE_ID: the engine image's identity (its layers and config), the same on every rank.
    engine: []const u8,
    /// TF_DS_MODEL_REVISION, and config.json's content digest.
    model_revision: []const u8,
    config: Digest,
    /// TF_DS_ABLATION_ID: the weights' ablation and its revision, or "none".
    ablation: []const u8,
    /// TF_DS_ENGRAM_REVISION, engram.json's content digest, and the Engram switches as text.
    engram_revision: []const u8,
    engram_json: Digest,
    engram_switches: []const u8,
    /// TF_DS_KIT_ID: the kernel kit, which lives outside the image.
    kit: []const u8,
    /// TF_DS_TOKEN_MAP: the Engram token map.
    token_map: []const u8,
    /// The CUDA driver's version and the GPU's compute capability.
    driver: u32,
    capability: u32,
    /// The math switches in effect after the start-up agreement, as text (pair_sum, qhc, owner_gather, idx_shard,
    /// idx_rb, negzero, replay_floor).
    math: []const u8,
    /// spill_state.StateLayout.digest(), the store's namespace version and the state's format version.
    layout: Digest,
    namespace_version: u32,
    state_version: u32,
};

/// The input a name lacks (`rank` for a rank's weight digest).
pub const Missing = struct {
    what: []const u8 = "",
    rank: ?u32 = null,

    /// The tier's off reason for it.
    pub fn text(m: Missing, buf: []u8) []const u8 {
        if (m.rank) |r| return std.fmt.bufPrint(buf, "rank {d}: no weight digest ({s})", .{ r, m.what }) catch m.what;
        return std.fmt.bufPrint(buf, "no {s}", .{m.what}) catch m.what;
    }
};

/// The store's name over `c` and `weights` (one SHA-256 a rank, rank 0 first: world = weights.len). An empty text or a
/// zero digest is error.MissingIdentity, named in `missing`.
pub fn name(c: Common, weights: []const Digest, missing: *Missing) error{MissingIdentity}!Digest {
    const texts = .{
        .{ "engine", "TF_DS_ENGINE_ID" },                .{ "model_revision", "TF_DS_MODEL_REVISION" }, .{ "ablation", "TF_DS_ABLATION_ID" },
        .{ "engram_revision", "TF_DS_ENGRAM_REVISION" }, .{ "engram_switches", "the Engram switches" }, .{ "kit", "TF_DS_KIT_ID" },
        .{ "token_map", "TF_DS_TOKEN_MAP" },             .{ "math", "the math switches" },
    };
    inline for (texts) |t| if (@field(c, t[0]).len == 0) return miss(missing, t[1], null);
    const digests = .{ .{ "config", "config.json's digest" }, .{ "engram_json", "engram.json's digest" }, .{ "layout", "the state layout's digest" } };
    inline for (digests) |t| if (zero(&@field(c, t[0]))) return miss(missing, t[1], null);
    if (c.driver == 0) return miss(missing, "the CUDA driver's version", null);
    if (c.capability == 0) return miss(missing, "the GPU's compute capability", null);
    if (weights.len == 0 or weights.len > max_world) return miss(missing, "the ranks' weight digests", null);
    for (weights, 0..) |w, r| if (zero(&w)) return miss(missing, "TF_DS_RANK_CACHE_SHA", @intCast(r));
    var s = Sha256.init(.{});
    s.update("tf-dsv41-spill-store\x00");
    int(&s, version);
    part(&s, c.engine);
    part(&s, c.model_revision);
    part(&s, &c.config);
    part(&s, c.ablation);
    part(&s, c.engram_revision);
    part(&s, &c.engram_json);
    part(&s, c.engram_switches);
    part(&s, c.kit);
    part(&s, c.token_map);
    int(&s, c.driver);
    int(&s, c.capability);
    part(&s, c.math);
    part(&s, &c.layout);
    int(&s, c.namespace_version);
    int(&s, c.state_version);
    int(&s, @intCast(weights.len));
    for (weights) |w| part(&s, &w);
    var out: Digest = undefined;
    s.final(&out);
    return out;
}

/// A file's SHA-256 over its content (config.json, engram.json; its name and place are not part of it).
pub fn fileDigest(io: std.Io, dir: std.Io.Dir, sub_path: []const u8) !Digest {
    var file = try dir.openFile(io, sub_path, .{});
    defer file.close(io);
    var s = Sha256.init(.{});
    var buf: [64 << 10]u8 = undefined;
    var at: u64 = 0;
    while (true) {
        const n = try file.readPositional(io, &.{&buf}, at);
        if (n == 0) break;
        s.update(buf[0..n]);
        at += n;
    }
    var out: Digest = undefined;
    s.final(&out);
    return out;
}

fn miss(m: *Missing, what: []const u8, rank: ?u32) error{MissingIdentity} {
    m.* = .{ .what = what, .rank = rank };
    return error.MissingIdentity;
}

fn zero(x: *const Digest) bool {
    return std.mem.allEqual(u8, x, 0);
}

/// One part: its length (u64, little-endian), then its bytes.
fn part(s: *Sha256, bytes: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, bytes.len, .little);
    s.update(&len);
    s.update(bytes);
}

fn int(s: *Sha256, v: u32) void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    part(s, &b);
}
