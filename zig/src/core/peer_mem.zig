//! The memory and the health each rank last reported to rank 0, kept for the admin surface (/status host_memory and node_health).
//! The engine's follower ranks send their health block over a link of their own (families/dsv41/peer_mem.zig); rank 0's reader
//! stores each report here with the time it came (rank 0's own sampler stores its node's the same way), and the admin page reads
//! the last one and its age. Nothing here is on a round's path: the writers are threads that sleep in poll between reports, the
//! reader is a request for /status.
const std = @import("std");

/// Ranks a report can come from (the four-Spark split is the largest world).
pub const max_ranks = 4;

/// A reading the host could not give (a missing sensor, a library that is not there): never a zero.
pub const unknown: i32 = std.math.minInt(i32);

/// One ACPI thermal zone per entry; a Spark has seven.
pub const max_zones = 8;

/// One ConnectX-7 port's counters (an InfiniBand port counter counts 4-byte words; these are bytes), cumulative since the driver loaded.
pub const Rail = extern struct { rx_bytes: u64, tx_bytes: u64, errors: u64, discards: u64 };

/// What one Spark reports about itself, the wire block of the health link and the stored reading. Versioned: a reader takes a block
/// of its own version only. Every field is 8 bytes or two of 4, so the block is whole 8-byte words (the store keeps it as atomics).
pub const Health = extern struct {
    magic: u32,
    version: u32,
    mem_available: u64,
    mem_total: u64,
    /// The 1-minute load average times 100, and the share of CPU time not idle since the last sample in tenths of a percent.
    load1_x100: i32,
    cpu_busy_x10: i32,
    ncpu: i32,
    gpu_temp_c: i32,
    /// Milli-degrees Celsius of each ACPI zone, `unknown` past the zones the host has.
    zone_mc: [max_zones]i32,
    gpu_util_pct: i32,
    gpu_clock_mhz: i32,
    gpu_clock_max_mhz: i32,
    gpu_power_mw: i32,
    nvme_temp_mc: i32,
    nvme_wear_pct: i32,
    /// NVML's clock event reasons (a bit set; 0 = none), `reasons_unknown` when NVML gave none.
    gpu_reasons: u64,
    rails: [2]Rail,
};
pub const health_magic: u32 = 0x31544c48; // "HLT1"
pub const health_version: u32 = 1;
pub const reasons_unknown: u64 = std.math.maxInt(u64);
const health_words = @sizeOf(Health) / 8;
comptime {
    std.debug.assert(@sizeOf(Health) % 8 == 0 and @alignOf(Health) == 8);
}

/// A block with nothing known: the sampler fills in what it can read.
pub fn emptyHealth() Health {
    var h = std.mem.zeroes(Health);
    h.magic = health_magic;
    h.version = health_version;
    h.load1_x100 = unknown;
    h.cpu_busy_x10 = unknown;
    h.ncpu = unknown;
    h.gpu_temp_c = unknown;
    h.zone_mc = @splat(unknown);
    h.gpu_util_pct = unknown;
    h.gpu_clock_mhz = unknown;
    h.gpu_clock_max_mhz = unknown;
    h.gpu_power_mw = unknown;
    h.nvme_temp_mc = unknown;
    h.nvme_wear_pct = unknown;
    h.gpu_reasons = reasons_unknown;
    for (&h.rails) |*r| r.* = .{ .rx_bytes = std.math.maxInt(u64), .tx_bytes = std.math.maxInt(u64), .errors = std.math.maxInt(u64), .discards = std.math.maxInt(u64) };
    return h;
}

/// Whether `now` differs from `sent` by more than a reader of the page would see: the step a value must move before a block is
/// sent ahead of the heartbeat. Counters that only grow (the rails') move by a mebibyte or any error.
pub fn moved(sent: *const Health, now: *const Health) bool {
    const far = struct {
        fn i(a: i32, b: i32, step: i32) bool {
            if ((a == unknown) != (b == unknown)) return true;
            if (a == unknown) return false;
            return @abs(@as(i64, a) - @as(i64, b)) >= @as(u64, @intCast(step));
        }
    };
    if (far.i(sent.load1_x100, now.load1_x100, 10) or far.i(sent.cpu_busy_x10, now.cpu_busy_x10, 20)) return true;
    if (far.i(sent.gpu_temp_c, now.gpu_temp_c, 1) or far.i(sent.gpu_util_pct, now.gpu_util_pct, 5)) return true;
    if (far.i(sent.gpu_clock_mhz, now.gpu_clock_mhz, 15) or far.i(sent.gpu_clock_max_mhz, now.gpu_clock_max_mhz, 1)) return true;
    if (far.i(sent.gpu_power_mw, now.gpu_power_mw, 1000) or far.i(sent.nvme_temp_mc, now.nvme_temp_mc, 1000) or far.i(sent.nvme_wear_pct, now.nvme_wear_pct, 1)) return true;
    for (sent.zone_mc, now.zone_mc) |a, b| if (far.i(a, b, 1000)) return true;
    if (sent.gpu_reasons != now.gpu_reasons) return true;
    if (@abs(@as(i64, @intCast(sent.mem_available)) - @as(i64, @intCast(now.mem_available))) >= (64 << 20)) return true;
    for (sent.rails, now.rails) |a, b| {
        if (a.errors != b.errors or a.discards != b.discards) return true;
        if ((a.rx_bytes == std.math.maxInt(u64)) != (b.rx_bytes == std.math.maxInt(u64))) return true;
        if (a.rx_bytes != std.math.maxInt(u64)) {
            if (@abs(@as(i128, a.rx_bytes) - @as(i128, b.rx_bytes)) >= (1 << 20) or @abs(@as(i128, a.tx_bytes) - @as(i128, b.tx_bytes)) >= (1 << 20)) return true;
        }
    }
    return false;
}

/// The ACPI thermal zones' own names ("acpitz", "soc_thermal", ...), read from the host once; "" for a zone the host does not have.
/// The same hardware on every Spark, so rank 0's names label every node's zones.
pub fn zoneName(i: usize) []const u8 {
    if (i >= max_zones) return "";
    if (zone_state.load(.acquire) != 2) {
        if (zone_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return "";
        for (0..max_zones) |z| {
            zone_len[z] = 0;
            var pb: [96]u8 = undefined;
            const p = std.fmt.bufPrintSentinel(&pb, "/sys/class/thermal/thermal_zone{d}/type", .{z}, 0) catch continue;
            const fd = std.c.open(p, .{ .ACCMODE = .RDONLY });
            if (fd < 0) continue;
            defer _ = std.c.close(fd);
            const n = std.c.read(fd, &zone_text[z], zone_text[z].len);
            if (n <= 0) continue;
            zone_len[z] = @intCast(std.mem.trim(u8, zone_text[z][0..@intCast(n)], "\n\r \t").len);
        }
        zone_state.store(2, .release);
    }
    return zone_text[i][0..zone_len[i]];
}
var zone_state: std.atomic.Value(u8) = .init(0);
var zone_text: [max_zones][24]u8 = undefined;
var zone_len: [max_zones]u8 = @splat(0);

const Slot = struct {
    /// The health block as words, written under `seq` (odd while a write is in progress); `has_health` once one came.
    seq: std.atomic.Value(u32) = .init(0),
    words: [health_words]std.atomic.Value(u64) = @splat(.init(0)),
    has_health: std.atomic.Value(bool) = .init(false),
    available: std.atomic.Value(u64) = .init(0),
    total: std.atomic.Value(u64) = .init(0),
    /// The monotonic time of the report (nowNs); 0 until the first.
    at_ns: std.atomic.Value(u64) = .init(0),
};

var slots: [max_ranks]Slot = @splat(.{});

/// The monotonic clock's nanoseconds: the one clock both ends of the table use.
pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Rank `rank`'s report: the memory its host could give a new allocation now, and the host's total, in bytes.
pub fn record(rank: usize, available: u64, total: u64) void {
    if (rank >= max_ranks) return;
    const s = &slots[rank];
    s.available.store(available, .release);
    s.total.store(total, .release);
    s.at_ns.store(@max(nowNs(), 1), .release);
}

/// Rank `rank`'s health block (one writer a rank): its memory figures too, so `read` and `readHealth` agree.
pub fn recordHealth(rank: usize, h: *const Health) void {
    if (rank >= max_ranks) return;
    const s = &slots[rank];
    const src: *const [health_words]u64 = @ptrCast(h);
    _ = s.seq.fetchAdd(1, .acq_rel);
    for (&s.words, src) |*w, v| w.store(v, .release);
    s.has_health.store(true, .release);
    _ = s.seq.fetchAdd(1, .release);
    record(rank, h.mem_available, h.mem_total);
}

pub const HealthReading = struct { health: Health, age_ms: u64 };

/// Rank `rank`'s last health block with its age; null before the first.
pub fn readHealth(rank: usize) ?HealthReading {
    if (rank >= max_ranks) return null;
    const s = &slots[rank];
    var out: [health_words]u64 = undefined;
    var tries: u32 = 0;
    while (tries < 100) : (tries += 1) {
        const before = s.seq.load(.acquire);
        if (before & 1 != 0) continue;
        if (!s.has_health.load(.acquire)) return null;
        for (&out, &s.words) |*o, *w| o.* = w.load(.acquire);
        if (s.seq.load(.acquire) != before) continue;
        const at = s.at_ns.load(.acquire);
        return .{ .health = std.mem.bytesToValue(Health, std.mem.asBytes(&out)), .age_ms = (nowNs() -| at) / std.time.ns_per_ms };
    }
    return null;
}

pub const Reading = struct { available: u64, total: u64, age_ms: u64 };

/// Rank `rank`'s last report and how long ago it came; null before its first (a rank that never reported is not zero).
pub fn read(rank: usize) ?Reading {
    if (rank >= max_ranks) return null;
    const s = &slots[rank];
    const at = s.at_ns.load(.acquire);
    if (at == 0) return null;
    return .{ .available = s.available.load(.acquire), .total = s.total.load(.acquire), .age_ms = (nowNs() -| at) / std.time.ns_per_ms };
}

/// Forgets every report (a new start; the tests).
pub fn reset() void {
    for (&slots) |*s| {
        s.at_ns.store(0, .release);
        s.available.store(0, .release);
        s.total.store(0, .release);
        s.has_health.store(false, .release);
    }
}

test "a report is read back with an age, and a rank that never reported reads null" {
    reset();
    defer reset();
    try std.testing.expect(read(1) == null);
    record(1, 15 << 30, 121 << 30);
    const r = read(1).?;
    try std.testing.expectEqual(@as(u64, 15 << 30), r.available);
    try std.testing.expectEqual(@as(u64, 121 << 30), r.total);
    try std.testing.expect(r.age_ms < 1000);
    try std.testing.expect(read(2) == null);
    try std.testing.expect(read(max_ranks) == null);
    reset();
    try std.testing.expect(read(1) == null);
}

test "a health block is read back whole with its age, and a rank that reported only memory has none" {
    reset();
    defer reset();
    try std.testing.expect(readHealth(1) == null);
    record(2, 1 << 30, 2 << 30);
    try std.testing.expect(readHealth(2) == null);
    var h = emptyHealth();
    h.mem_available = 7 << 30;
    h.mem_total = 121 << 30;
    h.zone_mc[3] = 61_500;
    h.gpu_clock_mhz = 2190;
    h.rails[1].rx_bytes = 12345;
    recordHealth(1, &h);
    const got = readHealth(1).?;
    try std.testing.expectEqual(@as(u64, 7 << 30), got.health.mem_available);
    try std.testing.expectEqual(@as(i32, 61_500), got.health.zone_mc[3]);
    try std.testing.expectEqual(unknown, got.health.zone_mc[4]);
    try std.testing.expectEqual(@as(i32, 2190), got.health.gpu_clock_mhz);
    try std.testing.expectEqual(@as(u64, 12345), got.health.rails[1].rx_bytes);
    try std.testing.expect(got.age_ms < 1000);
    // the memory view agrees with the health view
    try std.testing.expectEqual(@as(u64, 7 << 30), read(1).?.available);
    try std.testing.expect(readHealth(max_ranks) == null);
    reset();
    try std.testing.expect(readHealth(1) == null);
}

test "a block moves only past a step: a degree, a fifth of a load, 64 MiB, a mebibyte of traffic, any error" {
    var a = emptyHealth();
    var b = a;
    try std.testing.expect(!moved(&a, &b));
    b.gpu_temp_c = 50;
    try std.testing.expect(moved(&a, &b)); // unknown -> known
    a = b;
    try std.testing.expect(!moved(&a, &b));
    b.zone_mc[2] = 40_500;
    try std.testing.expect(moved(&a, &b));
    a = b;
    b.zone_mc[2] = 40_900;
    try std.testing.expect(!moved(&a, &b)); // under a degree
    b.gpu_power_mw = 20_000;
    a.gpu_power_mw = 19_500;
    try std.testing.expect(!moved(&a, &b));
    b.mem_available = 70 << 20;
    try std.testing.expect(moved(&a, &b));
    a = b;
    b.rails[0] = .{ .rx_bytes = 100, .tx_bytes = 100, .errors = 0, .discards = 0 };
    try std.testing.expect(moved(&a, &b)); // a rail appeared
    a = b;
    b.rails[0].rx_bytes += 4096;
    try std.testing.expect(!moved(&a, &b));
    b.rails[0].rx_bytes += 2 << 20;
    try std.testing.expect(moved(&a, &b));
    a = b;
    b.rails[0].errors = 1;
    try std.testing.expect(moved(&a, &b));
}

test "zone names are read once and a missing zone is empty" {
    try std.testing.expectEqualStrings("", zoneName(max_zones));
    const n = zoneName(0);
    try std.testing.expect(n.len <= 24);
    try std.testing.expectEqualStrings(n, zoneName(0));
}
