//! The probe's digests (probe.zig): what each rank hashes of a kept state, and which rank differs.
const std = @import("std");
const probe = @import("probe.zig");
const land = @import("land.zig");

/// Host memory standing in for the device: addresses are host pointers, copies are memcpy.
const Host = struct {
    pub fn d2h(_: Host, dst: []u8, src: u64) !void {
        @memcpy(dst, @as([*]const u8, @ptrFromInt(src))[0..dst.len]);
    }
};

test "rows: the first and last row and k more from the seed, sorted, the same for the same seed" {
    var a: [16]usize = undefined;
    var b: [16]usize = undefined;
    const x = probe.rows(7, 1000, 6, &a);
    const y = probe.rows(7, 1000, 6, &b);
    try std.testing.expectEqualSlices(usize, x, y);
    try std.testing.expectEqual(@as(usize, 0), x[0]);
    try std.testing.expectEqual(@as(usize, 999), x[x.len - 1]);
    try std.testing.expect(x.len >= 3 and x.len <= 8);
    for (x[1..], x[0 .. x.len - 1]) |hi, lo| try std.testing.expect(hi > lo);
    const z = probe.rows(8, 1000, 6, &b);
    try std.testing.expect(!std.mem.eql(usize, x, z)); // another seed, other rows
    try std.testing.expectEqual(@as(usize, 1), probe.rows(7, 1, 6, &a).len); // one row: just it
    try std.testing.expectEqual(@as(usize, 0), probe.rows(7, 0, 6, &a).len);
    try std.testing.expectEqual(@as(usize, 4), probe.rows(7, 4, 50, &a).len); // more asked than there are: each once
}

const Rig = struct {
    mem: [4096]u8 = undefined,
    planes: [2]land.Plane = undefined,

    fn init(r: *Rig) void {
        for (&r.mem, 0..) |*b, i| b.* = @truncate(i *% 13 +% 5);
        const a = @intFromPtr(&r.mem);
        r.planes = .{ .{ .plane = a + 512, .row = 8, .ratio = 4 }, .{ .plane = a + 2048, .row = 4, .ratio = 128 } };
    }
    fn digest(r: *Rig, seed: u64) ![32]u8 {
        var scratch: [64]u8 = undefined;
        return probe.digest(Host{}, @intFromPtr(&r.mem), 100, &r.planes, 256, 1024, seed, 3, &scratch);
    }
};

test "the digest covers the kept buffer and the sampled rows: a flip there moves it, one outside the sample does not" {
    var r: Rig = .{};
    r.init();
    const want = try r.digest(42);
    try std.testing.expectEqualSlices(u8, &want, &try r.digest(42));
    r.mem[37] ^= 1; // the kept buffer (100 bytes, read in 64-byte pieces)
    try std.testing.expect(!std.mem.eql(u8, &want, &try r.digest(42)));
    r.mem[37] ^= 1;
    var sample: [16]usize = undefined;
    const rs = probe.rows(probe.planeSeed(42, 0), 1024 / 4, 3, &sample); // the first plane's sampled rows
    const row0 = 512 + (256 / 4) * 8; // its extent's first row, in mem
    r.mem[row0 + rs[1] * 8 + 3] ^= 1;
    try std.testing.expect(!std.mem.eql(u8, &want, &try r.digest(42)));
    r.mem[row0 + rs[1] * 8 + 3] ^= 1;
    var outside: usize = 1;
    while (std.mem.indexOfScalar(usize, rs, outside) != null) outside += 1;
    r.mem[row0 + outside * 8] ^= 1; // a row the sample skips goes unseen
    try std.testing.expectEqualSlices(u8, &want, &try r.digest(42));
    r.mem[row0 + outside * 8] ^= 1;
    r.mem[512 + 10] ^= 1; // a row before the extent: not part of the state
    try std.testing.expectEqualSlices(u8, &want, &try r.digest(42));
}

test "verdict: the first rank whose digest is not rank 0's, or none" {
    const a: [32]u8 = @splat(1);
    const b: [32]u8 = @splat(2);
    try std.testing.expectEqual(@as(?u16, null), probe.verdict(&.{ a, a, a, a }));
    try std.testing.expectEqual(@as(?u16, 2), probe.verdict(&.{ a, a, b, b }));
    try std.testing.expectEqual(@as(?u16, 1), probe.verdict(&.{ a, b, a, a }));
    try std.testing.expectEqual(@as(?u16, null), probe.verdict(&.{a}));
}
