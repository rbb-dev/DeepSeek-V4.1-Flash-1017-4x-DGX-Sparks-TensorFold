//! CRC-32 (ISO-HDLC, the polynomial 0x04C11DB7 reflected: zlib's), with std.hash.Crc32's values and interface.
const std = @import("std");
const builtin = @import("builtin");

/// Whether this build's CPU has the Arm CRC32 instructions (a GB10's cores do: an image built for its own CPU uses them).
pub const arm = builtin.cpu.arch == .aarch64 and std.Target.aarch64.featureSetHas(builtin.cpu.features, .crc);

/// The Arm instructions where the CPU has them (eight bytes an instruction), else std's table (one byte a lookup).
pub const Crc32 = if (arm) ArmCrc32 else std.hash.Crc32;

const ArmCrc32 = struct {
    crc: u32,

    pub fn init() ArmCrc32 {
        return .{ .crc = 0xffffffff };
    }

    pub fn update(c: *ArmCrc32, bytes: []const u8) void {
        var crc = c.crc;
        var i: usize = 0;
        while (i + 8 <= bytes.len) : (i += 8) {
            const v = std.mem.readInt(u64, bytes[i..][0..8], .little);
            crc = asm ("crc32x %[out:w], %[crc:w], %[v:x]"
                : [out] "=r" (-> u32),
                : [crc] "r" (crc),
                  [v] "r" (v),
            );
        }
        while (i < bytes.len) : (i += 1) {
            crc = asm ("crc32b %[out:w], %[crc:w], %[v:w]"
                : [out] "=r" (-> u32),
                : [crc] "r" (crc),
                  [v] "r" (@as(u32, bytes[i])),
            );
        }
        c.crc = crc;
    }

    pub fn final(c: ArmCrc32) u32 {
        return ~c.crc;
    }

    pub fn hash(bytes: []const u8) u32 {
        var c = init();
        c.update(bytes);
        return c.final();
    }
};

test "the same values as std.hash.Crc32: every length to 300, random buffers, any split of the updates" {
    var buf: [4096 + 7]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    prng.random().bytes(&buf);
    for (0..301) |n| try std.testing.expectEqual(std.hash.Crc32.hash(buf[0..n]), Crc32.hash(buf[0..n]));
    for (0..64) |k| {
        const start = k % 7; // unaligned starts too
        const len = 1 + prng.random().uintLessThan(usize, buf.len - start);
        const data = buf[start..][0..len];
        const cut = prng.random().uintLessThan(usize, len + 1);
        var c = Crc32.init();
        c.update(data[0..cut]);
        c.update(data[cut..]);
        try std.testing.expectEqual(std.hash.Crc32.hash(data), c.final());
        try std.testing.expectEqual(std.hash.Crc32.hash(data), Crc32.hash(data));
    }
    try std.testing.expectEqual(@as(u32, 0xcbf43926), Crc32.hash("123456789")); // the check value
}

test "the Arm instructions where the CPU has them (else this build uses the table, already compared)" {
    if (!arm) return error.SkipZigTest;
    try std.testing.expect(Crc32 == ArmCrc32);
}
