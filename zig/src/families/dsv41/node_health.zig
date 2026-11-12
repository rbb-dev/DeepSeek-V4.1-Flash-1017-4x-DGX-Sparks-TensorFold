//! One Spark's health, read from the host for the admin page: memory, CPU load, the ACPI temperature zones, the GPU through NVML,
//! the NVMe's temperature and wear, and the two ConnectX-7 rails' counters.
//!
//! Where it runs and what it costs: one thread per rank (families/dsv41/peer_mem.zig's), asleep in poll between samples, awake once
//! a second for about forty small reads (sysfs and procfs files, a handful of NVML calls, the NVMe log page once a minute). It
//! never touches the engine thread, a round, or a CUDA call: NVML is the driver's monitoring library, which allocates no GPU memory
//! and launches nothing. Every reading is optional: a file that is not there, a library that cannot be opened or a call the GPU
//! refuses leaves that field `unknown`, and the page says "not reported".
//!
//! Read-only: nothing here sets a clock, a power limit or any other state of the host.
const std = @import("std");
const api = @import("engine_api");
const c = std.c;
const store = api.peer_mem;
pub const Health = store.Health;
const unknown = store.unknown;

/// The two rails of a Spark: the InfiniBand devices of its ConnectX-7 port's two PCIe functions.
pub const rail_devices = [2][:0]const u8{ "mlx5_0", "mlx5_2" };

fn slurp(path: [*:0]const u8, buf: []u8) ?[]const u8 {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = c.close(fd);
    const n = c.read(fd, buf.ptr, buf.len);
    if (n <= 0) return null;
    return buf[0..@intCast(n)];
}

fn slurpFmt(comptime fmt: []const u8, args: anytype, buf: []u8) ?[]const u8 {
    var pb: [192]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&pb, fmt, args, 0) catch return null;
    return slurp(p, buf);
}

fn number(comptime T: type, text: []const u8) ?T {
    return std.fmt.parseInt(T, std.mem.trim(u8, text, " \t\r\n"), 10) catch null;
}

pub const Stat = struct { busy: u64, total: u64, cpus: u32 };

/// /proc/stat's "cpu" line (jiffies: user nice system idle iowait irq softirq steal) and the count of its "cpuN" lines.
pub fn parseStat(text: []const u8) ?Stat {
    var lines = std.mem.splitScalar(u8, text, '\n');
    var out: ?Stat = null;
    var cpus: u32 = 0;
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "cpu")) continue;
        if (line.len > 3 and std.ascii.isDigit(line[3])) {
            cpus += 1;
            continue;
        }
        var it = std.mem.tokenizeAny(u8, line[3..], " \t");
        var v: [8]u64 = undefined;
        for (&v) |*x| x.* = std.fmt.parseInt(u64, it.next() orelse return null, 10) catch return null;
        var total: u64 = 0;
        for (v) |x| total += x;
        out = .{ .busy = total - v[3] - v[4], .total = total, .cpus = 0 };
    }
    var o = out orelse return null;
    o.cpus = cpus;
    return o;
}

/// /proc/loadavg's first figure times 100 (0.52 reads 52).
pub fn parseLoad(text: []const u8) ?i32 {
    var it = std.mem.tokenizeAny(u8, text, " \t\n");
    const f = std.fmt.parseFloat(f64, it.next() orelse return null) catch return null;
    if (!(f >= 0 and f < 1e6)) return null;
    const x100: i32 = @intFromFloat(@round(f * 100));
    return x100;
}

/// An NVMe SMART log page (log id 2): the percentage used (byte 5) and the composite temperature in kelvin (bytes 1-2).
pub const Smart = struct { wear_pct: i32, temp_k: i32 };
pub fn parseSmart(page: []const u8) ?Smart {
    if (page.len < 6) return null;
    const pct: i32 = page[5];
    return .{ .wear_pct = pct, .temp_k = @as(i32, page[1]) | (@as(i32, page[2]) << 8) };
}

const NvmeAdminCmd = extern struct {
    opcode: u8,
    flags: u8,
    rsvd1: u16,
    nsid: u32,
    cdw2: u32,
    cdw3: u32,
    metadata: u64,
    addr: u64,
    metadata_len: u32,
    data_len: u32,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
    timeout_ms: u32,
    result: u32,
};
comptime {
    std.debug.assert(@sizeOf(NvmeAdminCmd) == 72);
}
/// _IOWR('N', 0x41, struct nvme_admin_cmd)
const nvme_ioctl_admin_cmd: c_ulong = 0xC0484E41;
extern "c" fn ioctl(fd: c_int, request: c_ulong, ...) c_int;

/// The drive's SMART page by the admin ioctl (a container without the device node or the capability gets null).
fn readSmart() ?Smart {
    const fd = c.open("/dev/nvme0", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = c.close(fd);
    var page: [512]u8 align(8) = undefined;
    var cmd = std.mem.zeroes(NvmeAdminCmd);
    cmd.opcode = 0x02; // get log page
    cmd.nsid = 0xffffffff;
    cmd.addr = @intFromPtr(&page);
    cmd.data_len = page.len;
    cmd.cdw10 = (@as(u32, page.len / 4 - 1) << 16) | 2;
    if (ioctl(fd, nvme_ioctl_admin_cmd, &cmd) != 0) return null;
    return parseSmart(&page);
}

const Util = extern struct { gpu: u32, memory: u32 };
const Nvml = struct {
    tried: bool = false,
    lib: ?std.DynLib = null,
    handle: ?*anyopaque = null,
    temp: ?*const fn (?*anyopaque, c_int, *u32) callconv(.c) c_int = null,
    util: ?*const fn (?*anyopaque, *Util) callconv(.c) c_int = null,
    clock: ?*const fn (?*anyopaque, c_int, *u32) callconv(.c) c_int = null,
    clock_max: ?*const fn (?*anyopaque, c_int, *u32) callconv(.c) c_int = null,
    power: ?*const fn (?*anyopaque, *u32) callconv(.c) c_int = null,
    reasons: ?*const fn (?*anyopaque, *u64) callconv(.c) c_int = null,

    /// Opens libnvidia-ml once; a library or a GPU that is not there leaves every call null.
    fn open(n: *Nvml) void {
        if (n.tried) return;
        n.tried = true;
        var lib = std.DynLib.open("libnvidia-ml.so.1") catch return;
        const init = lib.lookup(*const fn () callconv(.c) c_int, "nvmlInit_v2") orelse {
            lib.close();
            return;
        };
        const by_index = lib.lookup(*const fn (u32, *?*anyopaque) callconv(.c) c_int, "nvmlDeviceGetHandleByIndex_v2") orelse {
            lib.close();
            return;
        };
        var h: ?*anyopaque = null;
        if (init() != 0 or by_index(0, &h) != 0 or h == null) {
            lib.close();
            return;
        }
        n.handle = h;
        n.temp = lib.lookup(@TypeOf(n.temp.?), "nvmlDeviceGetTemperature");
        n.util = lib.lookup(@TypeOf(n.util.?), "nvmlDeviceGetUtilizationRates");
        n.clock = lib.lookup(@TypeOf(n.clock.?), "nvmlDeviceGetClockInfo");
        n.clock_max = lib.lookup(@TypeOf(n.clock_max.?), "nvmlDeviceGetMaxClockInfo");
        n.power = lib.lookup(@TypeOf(n.power.?), "nvmlDeviceGetPowerUsage");
        n.reasons = lib.lookup(@TypeOf(n.reasons.?), "nvmlDeviceGetCurrentClocksEventReasons") orelse
            lib.lookup(@TypeOf(n.reasons.?), "nvmlDeviceGetCurrentClocksThrottleReasons");
        n.lib = lib;
    }

    fn fill(n: *Nvml, h: *Health) void {
        n.open();
        if (n.handle == null) return;
        var v: u32 = 0;
        if (n.temp) |f| if (f(n.handle, 0, &v) == 0) {
            h.gpu_temp_c = @intCast(@min(v, 1000));
        };
        if (n.util) |f| {
            var u: Util = .{ .gpu = 0, .memory = 0 };
            if (f(n.handle, &u) == 0) h.gpu_util_pct = @intCast(@min(u.gpu, 100));
        }
        if (n.clock) |f| if (f(n.handle, 1, &v) == 0) {
            h.gpu_clock_mhz = @intCast(@min(v, 100_000));
        };
        if (n.clock_max) |f| if (f(n.handle, 1, &v) == 0) {
            h.gpu_clock_max_mhz = @intCast(@min(v, 100_000));
        };
        if (n.power) |f| if (f(n.handle, &v) == 0) {
            h.gpu_power_mw = @intCast(@min(v, 10_000_000));
        };
        if (n.reasons) |f| {
            var r: u64 = 0;
            if (f(n.handle, &r) == 0) h.gpu_reasons = r;
        }
    }
};

pub const Sampler = struct {
    prev_busy: u64 = 0,
    prev_total: u64 = 0,
    have_prev: bool = false,
    nvml: Nvml = .{},
    nvme_hwmon: ?u8 = null,
    wear: i32 = unknown,
    smart_at_ns: u64 = 0,

    /// One reading of this host. Costs about forty small reads; see the file's head.
    pub fn sample(sp: *Sampler) Health {
        var h = store.emptyHealth();
        var buf: [8192]u8 = undefined;
        if (slurp("/proc/meminfo", buf[0..4096])) |t| {
            if (@import("peer_mem.zig").parse(t)) |u| {
                h.mem_available = u.available;
                h.mem_total = u.total;
            }
        }
        if (slurp("/proc/loadavg", buf[0..256])) |t| if (parseLoad(t)) |l| {
            h.load1_x100 = l;
        };
        if (slurp("/proc/stat", &buf)) |t| if (parseStat(t)) |st| {
            h.ncpu = @intCast(st.cpus);
            if (sp.have_prev and st.total > sp.prev_total) {
                const dt = st.total - sp.prev_total;
                const db = st.busy -| sp.prev_busy;
                h.cpu_busy_x10 = @intCast(@min(1000, db * 1000 / dt));
            }
            sp.prev_busy = st.busy;
            sp.prev_total = st.total;
            sp.have_prev = true;
        };
        for (&h.zone_mc, 0..) |*z, i| {
            if (slurpFmt("/sys/class/thermal/thermal_zone{d}/temp", .{i}, buf[0..32])) |t| if (number(i32, t)) |v| {
                z.* = v;
            };
        }
        sp.nvml.fill(&h);
        sp.nvme(&h, &buf);
        for (&h.rails, rail_devices) |*r, dev| {
            const rx = counter(dev, "port_rcv_data", &buf);
            const tx = counter(dev, "port_xmit_data", &buf);
            if (rx == null or tx == null) continue;
            r.rx_bytes = rx.? *% 4; // (an InfiniBand data counter counts 4-byte words)
            r.tx_bytes = tx.? *% 4;
            r.errors = counter(dev, "port_rcv_errors", &buf) orelse 0;
            r.discards = counter(dev, "port_xmit_discards", &buf) orelse 0;
        }
        return h;
    }

    fn nvme(sp: *Sampler, h: *Health, buf: []u8) void {
        // the composite temperature from hwmon (the number of the hwmon directory is looked for once, and again if it stops answering)
        var tries: u32 = 0;
        while (tries < 2) : (tries += 1) {
            if (sp.nvme_hwmon == null and tries == 0) {
                for (0..32) |n| {
                    if (slurpFmt("/sys/class/nvme/nvme0/hwmon{d}/temp1_input", .{n}, buf[0..32]) != null) {
                        sp.nvme_hwmon = @intCast(n);
                        break;
                    }
                }
            }
            const i = sp.nvme_hwmon orelse return;
            if (slurpFmt("/sys/class/nvme/nvme0/hwmon{d}/temp1_input", .{i}, buf[0..32])) |t| {
                if (number(i32, t)) |v| h.nvme_temp_mc = v;
                break;
            }
            sp.nvme_hwmon = null;
        }
        // the wear changes over months: asked once a minute (and then the page keeps the last answer)
        const now = store.nowNs();
        if (sp.smart_at_ns == 0 or now -% sp.smart_at_ns >= 60 * std.time.ns_per_s) {
            sp.smart_at_ns = now;
            if (readSmart()) |sm| {
                sp.wear = sm.wear_pct;
                if (h.nvme_temp_mc == unknown) h.nvme_temp_mc = (sm.temp_k - 273) * 1000;
            }
        }
        h.nvme_wear_pct = sp.wear;
    }
};

fn counter(dev: []const u8, name: []const u8, buf: []u8) ?u64 {
    return if (slurpFmt("/sys/class/infiniband/{s}/ports/1/counters/{s}", .{ dev, name }, buf[0..64])) |t| number(u64, t) else null;
}

test "the CPU line gives busy and total jiffies and the count of cpuN lines" {
    const t = "cpu  100 0 50 800 20 5 5 10 0 0\ncpu0 50 0 25 400 10 2 3 5 0 0\ncpu1 50 0 25 400 10 3 2 5 0 0\nintr 1\n";
    const s = parseStat(t).?;
    try std.testing.expectEqual(@as(u64, 990), s.total);
    try std.testing.expectEqual(@as(u64, 990 - 800 - 20), s.busy);
    try std.testing.expectEqual(@as(u32, 2), s.cpus);
    try std.testing.expect(parseStat("intr 1\n") == null);
    try std.testing.expect(parseStat("cpu  1 2 3\n") == null);
}

test "the load average reads times 100 and a bad figure reads null" {
    try std.testing.expectEqual(@as(i32, 52), parseLoad("0.52 0.40 0.30 1/200 999\n").?);
    try std.testing.expectEqual(@as(i32, 1250), parseLoad("12.5 1 1 1/2 3").?);
    try std.testing.expect(parseLoad("x") == null);
    try std.testing.expect(parseLoad("") == null);
}

test "an NVMe SMART page gives the wear and the temperature in kelvin" {
    var page: [512]u8 = @splat(0);
    page[1] = 0x28; // 0x0128 = 296 K
    page[2] = 0x01;
    page[5] = 3;
    const s = parseSmart(&page).?;
    try std.testing.expectEqual(@as(i32, 3), s.wear_pct);
    try std.testing.expectEqual(@as(i32, 296), s.temp_k);
    try std.testing.expect(parseSmart(page[0..4]) == null);
}

test "one sample of this host fills what the host has, leaves the rest unknown, and costs well under a tenth of a second" {
    if (@import("builtin").os.tag != .linux) return;
    var sp: Sampler = .{};
    const t0 = store.nowNs();
    const first = sp.sample();
    const mid = store.nowNs();
    const second = sp.sample();
    const t1 = store.nowNs();
    std.debug.print("node health: one sample {d} us (first, NVML opened), {d} us (second)\n", .{ (mid - t0) / 1000, (t1 - mid) / 1000 });
    // what this host answered (the report of which fields were filled: NVML in a container may have no library or no device)
    std.debug.print("node health: filled gpu_temp={} gpu_util={} gpu_clock={} gpu_clock_max={} gpu_power={} reasons={} zones={d} nvme_temp={} nvme_wear={} rails={} {}\n", .{
        second.gpu_temp_c != unknown,   second.gpu_util_pct != unknown,              second.gpu_clock_mhz != unknown,                  second.gpu_clock_max_mhz != unknown,
        second.gpu_power_mw != unknown, second.gpu_reasons != store.reasons_unknown,
        blk: {
            var n: u32 = 0;
            for (second.zone_mc) |z| n += @intFromBool(z != unknown);
            break :blk n;
        },
        second.nvme_temp_mc != unknown, second.nvme_wear_pct != unknown,             second.rails[0].rx_bytes != std.math.maxInt(u64), second.rails[1].rx_bytes != std.math.maxInt(u64),
    });
    try std.testing.expectEqual(store.health_magic, first.magic);
    try std.testing.expectEqual(store.health_version, first.version);
    try std.testing.expect(first.mem_total > 0 and first.mem_available <= first.mem_total);
    try std.testing.expect(first.cpu_busy_x10 == unknown); // (a rate needs two samples)
    try std.testing.expect(second.cpu_busy_x10 == unknown or (second.cpu_busy_x10 >= 0 and second.cpu_busy_x10 <= 1000));
    try std.testing.expect(second.ncpu > 0);
    try std.testing.expect((t1 - mid) < 100 * std.time.ns_per_ms);
}
