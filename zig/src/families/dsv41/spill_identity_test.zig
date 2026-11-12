//! The one store's name (spill_identity.zig): every input moves it, every rank's weights included, and a missing
//! input is named rather than hashed.
const std = @import("std");
const id = @import("spill_identity.zig");
const Digest = id.Digest;

fn d(byte: u8) Digest {
    return @splat(byte);
}

const base: id.Common = .{
    .engine = "native-12f3f2c3b4bf",
    .model_revision = "1017",
    .config = d(1),
    .ablation = "keys@abc123",
    .engram_revision = "e1",
    .engram_json = d(2),
    .engram_switches = "packed=1 owned=1 instream=0 aio=1",
    .kit = "kit-7",
    .token_map = "map-3",
    .driver = 580178,
    .capability = 121,
    .math = "pair_sum=1 qhc=1 owner_gather=1 idx_shard=1 idx_rb=0 negzero=0 replay_floor=0",
    .layout = d(3),
    .namespace_version = 2,
    .state_version = 2,
};
const four = [_]Digest{ d(10), d(11), d(12), d(13) };

fn named(c: id.Common, w: []const Digest) !Digest {
    var missing: id.Missing = .{};
    return id.name(c, w, &missing);
}

test "every common input moves the store's name" {
    const want = try named(base, &four);
    const info = @typeInfo(id.Common).@"struct";
    inline for (info.field_names, info.field_types) |fname, T| {
        var c = base;
        switch (T) {
            []const u8 => @field(c, fname) = "other",
            Digest => @field(c, fname)[7] ^= 0x40,
            u32 => @field(c, fname) += 1,
            else => @compileError("a Common field of a type the test does not move: " ++ fname),
        }
        const got = try named(c, &four);
        std.testing.expect(!std.mem.eql(u8, &want, &got)) catch |e| {
            std.debug.print("field {s} does not move the name\n", .{fname});
            return e;
        };
    }
}

test "rank 0's store depends on every rank's weights: one rank's digest, two swapped, or another world move it" {
    const want = try named(base, &four);
    for (0..4) |r| {
        var w = four;
        w[r][31] ^= 1;
        try std.testing.expect(!std.mem.eql(u8, &want, &try named(base, &w)));
    }
    var swapped = four;
    std.mem.swap(Digest, &swapped[1], &swapped[3]);
    try std.testing.expect(!std.mem.eql(u8, &want, &try named(base, &swapped)));
    try std.testing.expect(!std.mem.eql(u8, &want, &try named(base, four[0..2])));
    try std.testing.expectEqualSlices(u8, &want, &try named(base, &four)); // and the same inputs, the same name
}

test "a missing input is named, not hashed: an empty name, a zero digest, a rank without its weight digest" {
    var missing: id.Missing = .{};
    var c = base;
    c.engine = "";
    try std.testing.expectError(error.MissingIdentity, id.name(c, &four, &missing));
    try std.testing.expectEqualStrings("TF_DS_ENGINE_ID", missing.what);
    try std.testing.expect(missing.rank == null);
    c = base;
    c.config = d(0);
    try std.testing.expectError(error.MissingIdentity, id.name(c, &four, &missing));
    try std.testing.expectEqualStrings("config.json's digest", missing.what);
    c = base;
    c.capability = 0;
    try std.testing.expectError(error.MissingIdentity, id.name(c, &four, &missing));
    try std.testing.expectEqualStrings("the GPU's compute capability", missing.what);
    var w = four;
    w[3] = d(0);
    try std.testing.expectError(error.MissingIdentity, id.name(base, &w, &missing));
    try std.testing.expectEqualStrings("TF_DS_RANK_CACHE_SHA", missing.what);
    try std.testing.expectEqual(@as(?u32, 3), missing.rank);
    var buf: [120]u8 = undefined;
    try std.testing.expectEqualStrings("rank 3: no weight digest (TF_DS_RANK_CACHE_SHA)", missing.text(&buf));
    missing = .{ .what = "TF_DS_KIT_ID" };
    try std.testing.expectEqualStrings("no TF_DS_KIT_ID", missing.text(&buf));
    try std.testing.expectError(error.MissingIdentity, id.name(base, &.{}, &missing));
    try std.testing.expectEqualStrings("the ranks' weight digests", missing.what);
}

test "the name's construction is pinned" {
    // a change to how the name is built must bump `version` (every stored point then belongs to another namespace)
    var hex: [64]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x}", .{&try named(base, &four)}) catch unreachable;
    try std.testing.expectEqualStrings(pinned, &hex);
}

const pinned = "6f07a4c5941f2ddbe24f54cab83234867137bf24d50cb10602261eb69aeacbae";
