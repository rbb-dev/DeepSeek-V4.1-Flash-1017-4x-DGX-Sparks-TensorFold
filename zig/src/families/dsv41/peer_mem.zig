//! Each follower's health to rank 0, while the engine runs, for the admin page's live memory and node health per Spark.
//!
//! The follower ranks send one versioned health block (core/peer_mem.zig Health: memory, CPU load, the temperature zones, the
//! GPU through NVML, the NVMe, both rails' counters; families/dsv41/node_health.zig reads them) over a link of their own: when a
//! value has moved past its step, and at least every `period_ms`, never more often than once a second. Rank 0 keeps the last
//! report of each rank with its age (read by /status) and samples its own node the same way on a thread of its own. The link is separate on
//! purpose: the step links carry a frame per round, and the host-collective links (`vfan`) are a lockstep protocol whose
//! parts are all one length, so an unasked frame on either would be read as something else. Nothing here touches a round:
//! the follower's thread sleeps in poll between reports, rank 0's thread sleeps in poll until a report arrives.
//!
//! Upstream: ashhart/TensorFold zig/src/fabric/tp/fate.zig (`kind_mem`: a {magic, kind, value} frame of the rank's MemAvailable
//! on a period, the followers' watch thread sending it, rank 0's watch thread keeping the value and its time per rank, setting
//! TF_TP_PEER_MEM_S) is the design this follows. Upstream sends it over the sockets of its fail-fast channel, which this engine
//! does not have (the dead-peer watch, link.Watch, polls the step links); the period here is the constant `period_ms`.
//!
//! Never fatal: a rank that cannot open or reach the link reports nothing, and the page says "not reported".
const std = @import("std");
const api = @import("engine_api");
const link = @import("link.zig");
const node_health = @import("node_health.zig");
const c = std.c;
const store = api.peer_mem;

/// The port of follower g's link is the base port plus this plus g - 1, as the first set (and `host_links_port_offset`, 32).
pub const port_offset: u16 = 64;

/// The longest a follower is silent: a heartbeat when no value has moved past its step.
pub const period_ms: c_int = 2000;

/// How often a rank looks at its host (the shortest time between two blocks sent).
pub const tick_ms: c_int = 1000;

/// The wire block is the stored Health, whole; a block with another magic or version is dropped.
const Frame = store.Health;

pub const Usage = struct { available: u64, total: u64 };

/// MemAvailable and MemTotal from /proc/meminfo in bytes (null when unreadable).
pub fn meminfo(_: u32) ?Usage {
    const fd = c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = c.close(fd);
    var buf: [4096]u8 = undefined;
    const n = c.read(fd, &buf, buf.len);
    if (n <= 0) return null;
    return parse(buf[0..@intCast(n)]);
}

pub fn parse(text: []const u8) ?Usage {
    var avail: ?u64 = null;
    var total: ?u64 = null;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const slot: *?u64 = if (std.mem.startsWith(u8, line, "MemAvailable:")) &avail else if (std.mem.startsWith(u8, line, "MemTotal:")) &total else continue;
        var it = std.mem.tokenizeAny(u8, line[std.mem.indexOfScalar(u8, line, ':').? + 1 ..], " \t");
        const kb = std.fmt.parseInt(u64, it.next() orelse return null, 10) catch return null;
        if (!std.mem.eql(u8, it.next() orelse return null, "kB")) return null;
        slot.* = std.math.mul(u64, kb, 1024) catch return null;
    }
    const a = avail orelse return null;
    const t = total orelse return null;
    return if (t > 0 and a <= t) .{ .available = a, .total = t } else null;
}

/// The IPv4 address on the other end of a connected socket.
fn peerIp(fd: c.fd_t) ?[4]u8 {
    var a: c.sockaddr.in = undefined;
    var len: c.socklen_t = @sizeOf(c.sockaddr.in);
    if (c.getpeername(fd, @ptrCast(&a), &len) != 0 or a.family != c.AF.INET) return null;
    return @bitCast(a.addr);
}

/// Whether a connection still stands: a peek that finds the peer closed or the socket failed says no.
fn alive(fd: c.fd_t) bool {
    var b: [1]u8 = undefined;
    const n = c.recv(fd, &b, 1, c.MSG.PEEK | c.MSG.DONTWAIT);
    if (n > 0) return true;
    return n < 0 and (c.errno(n) == .AGAIN or c.errno(n) == .INTR);
}

const Timeval = extern struct { sec: c_long, usec: c_long };

pub const Channel = struct {
    rank: u32 = 0,
    world: u32 = 1,
    /// Rank 0: the listening sockets, one per follower.
    ports: link.Ports = .{},
    /// Rank 0: the address each follower's report may come from (null: not known, so none is taken). The start knows them from the
    /// step links (`expectFrom`).
    expect: [3]?[4]u8 = .{ null, null, null },
    /// A follower: rank 0's address for its link.
    ip: [4]u8 = .{ 0, 0, 0, 0 },
    port: u16 = 0,
    period: c_int = period_ms,
    /// What a follower reports as memory alone (tests replace the whole host reading with it: null reads the host).
    usable: ?*const fn (rank: u32) ?Usage = null,
    /// Rank 0 samples its own node too (the engine sets it; a test channel does not).
    own: bool = false,
    wake: [2]c.fd_t = .{ -1, -1 },
    stopping: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    own_thread: ?std.Thread = null,

    /// Rank 0 starts listening for its followers' links; a follower notes where rank 0 is. Never fails: a rank without its link
    /// reports nothing.
    pub fn prepare(ch: *Channel, master: [4]u8, listen: ?[3][4]u8, base: u16, rank: u32, world: u32) void {
        ch.* = .{ .rank = rank, .world = world, .ip = master, .port = base + @as(u16, @intCast(rank -| 1)), .own = rank == 0 };
        if (world < 2 or world > 4 or rank != 0) return;
        var ips: [3][4]u8 = undefined;
        for (ips[0 .. world - 1], 0..) |*a, i| a.* = if (listen) |l| l[i] else master;
        ch.ports = link.Ports.open(ips[0 .. world - 1], base) catch |err| {
            std.log.warn("dsv41: the memory-report link cannot listen on port {d}: {s}; the admin page will not show the followers' memory", .{ base, @errorName(err) });
            return;
        };
    }

    /// Rank 0: the addresses the followers' step links come from; a report is taken only from its rank's address.
    pub fn expectFrom(ch: *Channel, steps: []const ?link.Link) void {
        for (steps, 0..) |l, g| if (l) |x| if (g < 3) {
            ch.expect[g] = peerIp(x.fd);
        };
    }

    /// Starts the thread (rank 0's reader, or a follower's sender). Never fails.
    pub fn start(ch: *Channel) void {
        if (ch.thread != null or ch.world < 2) return;
        if (ch.rank == 0 and ch.ports.fds[0] < 0) return;
        ch.stopping.store(false, .release);
        var p: [2]c.fd_t = undefined;
        if (c.pipe2(&p, .{ .CLOEXEC = true }) != 0) return;
        ch.wake = p;
        if (ch.own and ch.rank == 0) ch.own_thread = std.Thread.spawn(.{}, sampleOwn, .{ch}) catch null; // (its node's figures, never fatal)
        const spawned = if (ch.rank == 0) std.Thread.spawn(.{}, serve, .{ch}) else std.Thread.spawn(.{}, report, .{ch});
        ch.thread = spawned catch |err| {
            std.log.warn("dsv41: rank {d}: the memory-report thread did not start: {s}", .{ ch.rank, @errorName(err) });
            ch.stopping.store(true, .release);
            if (ch.own_thread) |t| t.join();
            ch.own_thread = null;
            ch.closeWake();
            return;
        };
    }

    fn closeWake(ch: *Channel) void {
        for (&ch.wake) |*fd| if (fd.* >= 0) {
            _ = c.close(fd.*);
            fd.* = -1;
        };
    }

    /// Ends the thread (nothing when it never started); the listening sockets stay until `close`.
    pub fn stop(ch: *Channel) void {
        ch.stopping.store(true, .release);
        if (ch.thread) |t| {
            if (ch.wake[1] >= 0) _ = c.write(ch.wake[1], "x", 1);
            t.join();
        }
        ch.thread = null;
        if (ch.own_thread) |t| t.join();
        ch.own_thread = null;
        ch.closeWake();
    }

    pub fn close(ch: *Channel) void {
        ch.stop();
        ch.ports.close();
    }

    // ---- rank 0: take each follower's link as it comes, read its reports
    fn serve(ch: *Channel) void {
        var conn: [3]c.fd_t = .{ -1, -1, -1 };
        var have: [3]usize = .{ 0, 0, 0 };
        var buf: [3][@sizeOf(Frame)]u8 align(8) = undefined;
        const n_follow: usize = ch.world - 1;
        defer for (conn) |fd| if (fd >= 0) {
            _ = c.close(fd);
        };
        while (!ch.stopping.load(.acquire)) {
            var p: [7]c.pollfd = undefined;
            var kind: [7]struct { g: usize, listen: bool } = undefined;
            var n: usize = 0;
            p[n] = .{ .fd = ch.wake[0], .events = c.POLL.IN, .revents = 0 };
            n += 1;
            for (0..n_follow) |g| {
                if (ch.ports.fds[g] >= 0) {
                    p[n] = .{ .fd = ch.ports.fds[g], .events = c.POLL.IN, .revents = 0 };
                    kind[n] = .{ .g = g, .listen = true };
                    n += 1;
                }
                if (conn[g] >= 0) {
                    p[n] = .{ .fd = conn[g], .events = c.POLL.IN, .revents = 0 };
                    kind[n] = .{ .g = g, .listen = false };
                    n += 1;
                }
            }
            if (c.poll(&p, @intCast(n), -1) <= 0) continue;
            if (p[0].revents != 0) return;
            for (p[1..n], kind[1..n]) |q, k| {
                if (q.revents == 0) continue;
                if (k.listen) {
                    const fd = c.accept(q.fd, null, null);
                    if (fd < 0) continue;
                    // only the follower's own address may report for it, and a second connection for a rank that is connected is
                    // refused (a follower that comes again finds its old connection dead, and is taken)
                    const from = peerIp(fd);
                    if (from == null or ch.expect[k.g] == null or !std.mem.eql(u8, &from.?, &ch.expect[k.g].?)) {
                        _ = c.close(fd);
                        continue;
                    }
                    if (conn[k.g] >= 0) {
                        if (alive(conn[k.g])) {
                            _ = c.close(fd);
                            continue;
                        }
                        _ = c.close(conn[k.g]);
                    }
                    conn[k.g] = fd;
                    have[k.g] = 0;
                    continue;
                }
                const got = c.recv(q.fd, &buf[k.g][have[k.g]], @sizeOf(Frame) - have[k.g], c.MSG.DONTWAIT);
                if (got <= 0) {
                    if (got < 0 and (c.errno(got) == .AGAIN or c.errno(got) == .INTR)) continue;
                    _ = c.close(conn[k.g]); // its reports stop; the age on the page grows
                    conn[k.g] = -1;
                    have[k.g] = 0;
                    continue;
                }
                have[k.g] += @intCast(got);
                if (have[k.g] < @sizeOf(Frame)) continue;
                have[k.g] = 0;
                const f = std.mem.bytesToValue(Frame, &buf[k.g]);
                if (f.magic == store.health_magic and f.version == store.health_version) store.recordHealth(k.g + 1, &f);
            }
        }
    }

    // ---- a follower: connect when it can, look at the host each tick, send when a value moved or the heartbeat is due, never block
    fn report(ch: *Channel) void {
        var fd: c.fd_t = -1;
        defer if (fd >= 0) {
            _ = c.close(fd);
        };
        var sampler: node_health.Sampler = .{};
        var sent: Frame = store.emptyHealth();
        var sent_at: u64 = 0;
        while (!ch.stopping.load(.acquire)) {
            if (fd < 0) fd = connectOnce(ch.ip, ch.port);
            if (fd >= 0) if (ch.reading(&sampler)) |f| {
                const now = store.nowNs();
                const due = sent_at == 0 or now -% sent_at >= @as(u64, @intCast(ch.period)) * std.time.ns_per_ms or store.moved(&sent, &f);
                if (due) {
                    const n = c.send(fd, std.mem.asBytes(&f), @sizeOf(Frame), c.MSG.DONTWAIT | c.MSG.NOSIGNAL);
                    if (n != @sizeOf(Frame)) { // (a partial or failed send: the link is dropped and made again next tick)
                        _ = c.close(fd);
                        fd = -1;
                        sent_at = 0;
                    } else {
                        sent = f;
                        sent_at = now;
                    }
                }
            };
            var w = [1]c.pollfd{.{ .fd = ch.wake[0], .events = c.POLL.IN, .revents = 0 }};
            if (c.poll(&w, 1, @min(ch.period, tick_ms)) > 0) return;
        }
    }

    /// Rank 0: its own node, on a thread of its own, each tick.
    fn sampleOwn(ch: *Channel) void {
        var sampler: node_health.Sampler = .{};
        while (!ch.stopping.load(.acquire)) {
            const f = sampler.sample();
            store.recordHealth(0, &f);
            var w = [1]c.pollfd{.{ .fd = ch.wake[0], .events = c.POLL.IN, .revents = 0 }};
            if (c.poll(&w, 1, tick_ms) > 0) return;
        }
    }

    /// What this tick reports: the host's block, or (a test's hook) the figures the hook gives and nothing else.
    fn reading(ch: *Channel, sampler: *node_health.Sampler) ?Frame {
        if (ch.usable) |u| {
            const m = u(ch.rank) orelse return null;
            var f = store.emptyHealth();
            f.mem_available = m.available;
            f.mem_total = m.total;
            return f;
        }
        return sampler.sample();
    }

    /// One connect attempt that gives up after a second (rank 0 may not be listening yet, or the cable is out).
    fn connectOnce(ip: [4]u8, port: u16) c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return -1;
        const tv: Timeval = .{ .sec = 1, .usec = 0 };
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.SNDTIMEO, std.mem.asBytes(&tv), @sizeOf(Timeval));
        var a: c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
        if (c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) {
            _ = c.close(fd);
            return -1;
        }
        return fd;
    }
};

fn nap(ms: i64) void {
    std.Io.sleep(std.testing.io, .fromMilliseconds(ms), .awake) catch {};
}

fn testUsage(rank: u32) ?Usage {
    return .{ .available = (10 + @as(u64, rank)) << 30, .total = 121 << 30 };
}

test "meminfo is read in bytes, and a text without both figures reads null" {
    const u = parse("MemTotal:       131072 kB\nMemFree:          1024 kB\nMemAvailable:    65536 kB\n").?;
    try std.testing.expectEqual(@as(u64, 131072 * 1024), u.total);
    try std.testing.expectEqual(@as(u64, 65536 * 1024), u.available);
    try std.testing.expect(parse("MemTotal: 1024 kB\n") == null);
    try std.testing.expect(parse("MemTotal: 1024 kB\nMemAvailable: 2048 kB\n") == null);
    try std.testing.expect(parse("MemTotal: 1024 MB\nMemAvailable: 1 kB\n") == null);
    try std.testing.expect(meminfo(0) != null or @import("builtin").os.tag != .linux);
}

test "four hosts: each follower's report reaches rank 0 with its own figures, a follower that stops goes stale and one that comes again is taken" {
    store.reset();
    defer store.reset();
    const lo = [4]u8{ 127, 0, 0, 1 };
    var head: Channel = .{ .rank = 0, .world = 4, .period = 50 };
    head.ports = try link.Ports.open(&.{ lo, lo, lo }, 0);
    head.expect = .{ lo, lo, lo };
    defer head.close();
    head.start();
    try std.testing.expect(head.thread != null);
    var fol: [3]Channel = undefined;
    for (&fol, 1..) |*f, r| {
        f.* = .{ .rank = @intCast(r), .world = 4, .ip = lo, .port = try head.ports.portOf(r), .period = 50, .usable = testUsage };
        f.start();
    }
    defer for (&fol) |*f| f.close();
    var waited: u32 = 0;
    while (waited < 100 and !(store.read(1) != null and store.read(2) != null and store.read(3) != null)) : (waited += 1) nap(50);
    for (1..4) |r| {
        const got = store.read(r) orelse return error.NoReport;
        try std.testing.expectEqual((10 + @as(u64, r)) << 30, got.available);
        try std.testing.expectEqual(@as(u64, 121 << 30), got.total);
        try std.testing.expect(got.age_ms < 2000);
    }
    try std.testing.expect(store.read(0) == null); // rank 0 reports nothing about itself here
    // rank 2 stops: its age grows while the others stay fresh
    fol[1].close();
    nap(400);
    try std.testing.expect(store.read(2).?.age_ms >= 300);
    try std.testing.expect(store.read(1).?.age_ms < 300);
    // rank 2's link comes again (the follower process restarted): rank 0 takes it
    fol[1].port = try head.ports.portOf(2);
    fol[1].start();
    waited = 0;
    while (waited < 100 and store.read(2).?.age_ms >= 200) : (waited += 1) nap(20);
    try std.testing.expect(store.read(2).?.age_ms < 200);
}

test "a link that cannot be opened or reached never fails the rank: it reports nothing" {
    store.reset();
    defer store.reset();
    // rank 0's port already taken: prepare leaves no listening sockets and start starts no thread
    var taken = try link.Ports.open(&.{.{ 127, 0, 0, 1 }}, 0);
    defer taken.close();
    const busy = try taken.portOf(1);
    var head: Channel = .{};
    head.prepare(.{ 127, 0, 0, 1 }, null, busy, 0, 2);
    defer head.close();
    try std.testing.expect(head.ports.fds[0] < 0);
    head.start();
    try std.testing.expect(head.thread == null);
    // a follower whose rank 0 does not listen keeps trying quietly and stops when told
    var f: Channel = undefined;
    f.prepare(.{ 127, 0, 0, 1 }, null, busy +% 1, 1, 2);
    f.period = 30;
    f.usable = testUsage;
    f.start();
    nap(150);
    f.close();
    try std.testing.expect(store.read(1) == null);
}

test "stop and close are safe on a channel that never started, and when called twice; a started one stops and closes twice too" {
    var never: Channel = .{};
    never.stop();
    never.close();
    never.stop();
    never.close();
    try std.testing.expect(never.thread == null);
    // rank 0 with its ports open and its thread running, then ended twice
    var head: Channel = .{ .rank = 0, .world = 2, .period = 50 };
    head.ports = try link.Ports.open(&.{.{ 127, 0, 0, 1 }}, 0);
    head.start();
    try std.testing.expect(head.thread != null);
    head.stop();
    head.stop();
    try std.testing.expect(head.thread == null);
    head.close();
    head.close();
    try std.testing.expect(head.ports.fds[0] < 0);
}

fn dial(port: u16, from: ?[4]u8) !c.fd_t {
    const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    if (from) |f| {
        var b: c.sockaddr.in = .{ .port = 0, .addr = @bitCast(f) };
        if (c.bind(fd, @ptrCast(&b), @sizeOf(c.sockaddr.in)) != 0) return error.BindFailed;
    }
    var a: c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    if (c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.ConnectFailed;
    return fd;
}

fn sendFrame(fd: c.fd_t, avail: u64) void {
    var f: Frame = store.emptyHealth();
    f.mem_available = avail;
    f.mem_total = 100 << 30;
    _ = c.send(fd, std.mem.asBytes(&f), @sizeOf(Frame), c.MSG.NOSIGNAL);
}

test "a report is taken only from its rank's address, and a second connection for a connected rank is refused" {
    store.reset();
    defer store.reset();
    const lo = [4]u8{ 127, 0, 0, 1 };
    var head: Channel = .{ .rank = 0, .world = 2, .period = 50 };
    head.ports = try link.Ports.open(&.{lo}, 0);
    defer head.close();
    // the start knows rank 1 at 127.0.0.2: a report from 127.0.0.1 is refused (RED before the check: it was stored)
    // (written before the thread starts; the test changes it once afterwards, between connections)
    head.expect = .{ .{ 127, 0, 0, 2 }, null, null };
    head.start();
    const port = try head.ports.portOf(1);
    const stranger = try dial(port, lo);
    sendFrame(stranger, 1 << 30);
    nap(300);
    try std.testing.expect(store.read(1) == null);
    _ = c.close(stranger);
    // the start's address for rank 1 is this one (a WSL loopback shows every source as 127.0.0.1, so the address moves, not the dialler)
    head.expect[0] = lo;
    const good = try dial(port, null);
    defer _ = c.close(good);
    sendFrame(good, 7 << 30);
    nap(300);
    try std.testing.expectEqual(@as(u64, 7 << 30), store.read(1).?.available);
    // a duplicate from the right address while the first lives is refused: its figures never replace the first's
    const dup = try dial(port, null);
    defer _ = c.close(dup);
    sendFrame(dup, 9 << 30);
    nap(300);
    try std.testing.expectEqual(@as(u64, 7 << 30), store.read(1).?.available);
    sendFrame(good, 8 << 30);
    nap(300);
    try std.testing.expectEqual(@as(u64, 8 << 30), store.read(1).?.available);
}

test "a follower's whole health block reaches rank 0 and is read back, a block of another version is dropped, and rank 0 samples its own node" {
    store.reset();
    defer store.reset();
    const lo = [4]u8{ 127, 0, 0, 1 };
    var head: Channel = .{ .rank = 0, .world = 2, .period = 50, .own = true };
    head.ports = try link.Ports.open(&.{lo}, 0);
    head.expect = .{ lo, null, null };
    defer head.close();
    head.start();
    var fol: Channel = .{ .rank = 1, .world = 2, .ip = lo, .port = try head.ports.portOf(1), .period = 50 };
    fol.start(); // (no hook: the follower reads this host)
    defer fol.close();
    var waited: u32 = 0;
    while (waited < 100 and !(store.readHealth(1) != null and store.readHealth(0) != null)) : (waited += 1) nap(50);
    const f = (store.readHealth(1) orelse return error.NoReport).health;
    try std.testing.expectEqual(store.health_magic, f.magic);
    try std.testing.expect(f.mem_total > 0);
    const own = (store.readHealth(0) orelse return error.NoOwnReport).health;
    try std.testing.expect(own.mem_total > 0);
    // a block of a version the reader does not know is dropped (RED before the version check: it was stored)
    fol.close();
    nap(100);
    store.reset();
    const stranger = try dial(try head.ports.portOf(1), null);
    defer _ = c.close(stranger);
    var bad: Frame = store.emptyHealth();
    bad.version = store.health_version + 1;
    bad.mem_total = 5;
    _ = c.send(stranger, std.mem.asBytes(&bad), @sizeOf(Frame), c.MSG.NOSIGNAL);
    nap(300);
    try std.testing.expect(store.readHealth(1) == null);
}
