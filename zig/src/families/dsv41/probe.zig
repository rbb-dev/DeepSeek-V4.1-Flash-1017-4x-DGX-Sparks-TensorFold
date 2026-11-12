//! The probe: every rank's bytes of one kept state compared, at the start and every TF_DS_SPILL_PROBE_EVERY keeps.
//! One store on rank 0 serves every rank only while the ranks hold the same bits (every reduction is an all-gather
//! each rank adds in rank order, comm.zig). Each rank hashes the kept buffer and a seeded sample of the extent's rows;
//! a difference turns the tier off for the process, named. A row the sample skips goes unseen; each probe reseeds.
const std = @import("std");
const land = @import("land.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Digest = [32]u8;

/// The most rows a plane's sample holds (the first, the last, and the seeded ones).
pub const max_rows = 64;

/// The rows a plane's extent of `n` rows is sampled at: its first and last, and `k` more drawn from `seed`; sorted,
/// each once (fewer when the extent is short). The same seed gives every rank the same rows.
pub fn rows(seed: u64, n: usize, k: usize, out: []usize) []usize {
    if (n == 0) return out[0..0];
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    var m: usize = 0;
    const want = @min(out.len, k + 2);
    out[m] = 0;
    m += 1;
    if (n > 1 and m < want) {
        out[m] = n - 1;
        m += 1;
    }
    var tries: usize = 0;
    while (m < want and m < n and tries < 4 * want) : (tries += 1) {
        const x = r.uintLessThan(usize, n);
        if (std.mem.indexOfScalar(usize, out[0..m], x) == null) {
            out[m] = x;
            m += 1;
        }
    }
    std.mem.sort(usize, out[0..m], {}, std.sort.asc(usize));
    return out[0..m];
}

/// The seed of plane `i`'s sample (each plane samples other rows).
pub fn planeSeed(seed: u64, i: usize) u64 {
    return seed +% (@as(u64, i) *% 0x9e37_79b9_7f4a_7c15);
}

/// This rank's digest of a kept state: its kept buffer (`small_bytes` at `buf`), then each plane's sampled rows of the
/// extent [base, base + at) in copyExtent's order. `dev.d2h(dst, src)` copies device bytes to the host and returns when
/// they are there (the caller synchronizes its stream first); `scratch` is at least one row and holds each copy.
pub fn digest(dev: anytype, buf: u64, small_bytes: usize, planes: []const land.Plane, base: usize, at: usize, seed: u64, k: usize, scratch: []u8) !Digest {
    var s = Sha256.init(.{});
    var off: usize = 0;
    while (off < small_bytes) {
        const n = @min(scratch.len, small_bytes - off);
        try dev.d2h(scratch[0..n], buf + off);
        s.update(scratch[0..n]);
        off += n;
    }
    var sample: [max_rows]usize = undefined;
    for (planes, 0..) |p, i| {
        if (p.ratio == 0 or base % p.ratio != 0 or at % p.ratio != 0) return error.BadView;
        if (p.row > scratch.len) return error.ScratchTooSmall;
        const first = base / p.ratio;
        for (rows(planeSeed(seed, i), at / p.ratio, k, &sample)) |r| {
            try dev.d2h(scratch[0..p.row], p.plane + (first + r) * p.row);
            s.update(scratch[0..p.row]);
        }
    }
    var out: Digest = undefined;
    s.final(&out);
    return out;
}

/// The first rank whose digest is not rank 0's (null: every rank agrees).
pub fn verdict(digests: []const Digest) ?u16 {
    if (digests.len < 2) return null;
    for (digests[1..], 1..) |x, r| if (!std.mem.eql(u8, &x, &digests[0])) return @intCast(r);
    return null;
}
