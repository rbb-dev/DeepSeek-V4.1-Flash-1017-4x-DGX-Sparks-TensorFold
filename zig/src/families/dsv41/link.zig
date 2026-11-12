//! Rank 0 to rank 1 over TCP (the fabric addresses): length-prefixed frames for each round's step and the start-up settings, as
//! multi.py's Link sends them. One connection, TCP_NODELAY, SO_BUSY_POLL, blocking reads (recvSpin polls first).
const std = @import("std");
const where = @import("where.zig");
const c = std.c;

pub const Link = struct {
    fd: c.fd_t,

    fn socket() !c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        tune(fd);
        return fd;
    }

    /// TCP_NODELAY, and SO_BUSY_POLL: a read polls the NIC's receive queue itself (up to 50 us) instead of waiting for
    /// its interrupt, whose core may be leaving a deep idle state (node to node, 64-byte frames: round trip 1.0 ms
    /// blocking, 0.72 ms polling the socket, 11 us with SO_BUSY_POLL).
    fn tune(fd: c.fd_t) void {
        const one: c_int = 1;
        const busy_us: c_int = 50;
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.BUSY_POLL, std.mem.asBytes(&busy_us), @sizeOf(c_int));
    }

    fn addr(ip: [4]u8, port: u16) c.sockaddr.in {
        return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
    }

    /// Rank 0: listen on `ip`:`port` and take the one peer's connection.
    pub fn listen(ip: [4]u8, port: u16) !Link {
        var p = try Ports.open(&.{ip}, port);
        defer p.close();
        return p.accept(1);
    }

    /// A socket bound to `ip`:`port` and listening (a backlog of one: the port is one peer's).
    fn bound(ip: [4]u8, port: u16) !c.fd_t {
        const fd = try socket();
        errdefer _ = c.close(fd);
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
        var a = addr(ip, port);
        if (c.bind(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.BindFailed;
        if (c.listen(fd, 1) != 0) return error.ListenFailed;
        return fd;
    }

    /// Rank 1: connect to rank 0, retrying for up to `seconds` while it starts.
    pub fn connect(io: std.Io, ip: [4]u8, port: u16, seconds: u32) !Link {
        var tries: u32 = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            var a = addr(ip, port);
            if (c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) == 0) return .{ .fd = fd };
            _ = c.close(fd);
            if (tries >= seconds * 4) return error.ConnectFailed;
            std.Io.sleep(io, .fromMilliseconds(250), .awake) catch {};
        }
    }

    pub fn close(l: *Link) void {
        _ = c.close(l.fd);
        l.* = undefined;
    }

    fn writeAll(l: Link, bytes: []const u8, flags: u32) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = c.send(l.fd, bytes.ptr + done, bytes.len - done, flags);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    fn readAll(l: Link, out: []u8) !void {
        var done: usize = 0;
        while (done < out.len) {
            const n = c.recv(l.fd, out.ptr + done, out.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    /// Whether a frame has started to arrive within `us` microseconds (rounded up to a millisecond): the caller does other work
    /// while it waits and reads the frame with recv when this says true. A closed peer reads as ready (recv then fails).
    pub fn wait(l: Link, us: u32) bool {
        var p = [1]c.pollfd{.{ .fd = l.fd, .events = c.POLL.IN, .revents = 0 }};
        const ms: c_int = @intCast((us + 999) / 1000);
        return c.poll(&p, 1, ms) != 0;
    }

    /// Frames up to this many bytes go out in one send() call, length and payload together.
    const one_call = 2048;

    /// One frame: its length (u32, little-endian), then its bytes. A small frame leaves in one send() call, so its first bytes
    /// (the length) never reach a follower alone: a follower that peeks at the socket (peekHead) sees the length and the kind
    /// byte together, and rank 0's engine thread makes one syscall a follower a frame. A larger one sends its length with
    /// MSG_MORE, which holds it for the payload.
    pub fn send(l: Link, bytes: []const u8) !void {
        const held = where.enter(.link_send, 0, 0);
        defer where.leave(held);
        var one: [4 + one_call]u8 = undefined;
        const len = one[0..4];
        std.mem.writeInt(u32, len, @intCast(bytes.len), .little);
        if (bytes.len <= one_call) {
            @memcpy(one[4..][0..bytes.len], bytes);
            return l.writeAll(one[0 .. 4 + bytes.len], 0);
        }
        try l.writeAll(len, c.MSG.MORE);
        try l.writeAll(bytes, 0);
    }

    /// recv, polling the socket without blocking for up to `spin_ns` first: a frame that comes within it is read
    /// without the receiving thread's sleep and wake-up (a core's deepest idle state takes 433 us to leave on GB10; the
    /// follower's frames come a fraction of a millisecond after its GPU goes idle), and after it the read blocks.
    pub fn recvSpin(l: Link, buf: []u8, spin_ns: u64) ![]u8 {
        var len: [4]u8 = undefined;
        var got: usize = 0;
        const t0 = monotonicNs();
        var polls: u32 = 0;
        while (got < len.len) {
            const n = c.recv(l.fd, @as([*]u8, &len) + got, len.len - got, c.MSG.DONTWAIT);
            if (n > 0) {
                got += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN => {},
                .INTR => continue,
                else => return error.PeerDown,
            }
            polls +%= 1;
            if (polls % 64 == 0 and monotonicNs() -% t0 > spin_ns) break;
            std.atomic.spinLoopHint();
        }
        if (got < len.len) try l.readAll(len[got..]);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }

    /// recvSpin that gives up when nothing has begun to arrive within `spin_ns`: null then, the link untouched. A frame that has
    /// begun is read whole.
    pub fn recvSpinOrNone(l: Link, buf: []u8, spin_ns: u64) !?[]u8 {
        var len: [4]u8 = undefined;
        var got: usize = 0;
        const t0 = monotonicNs();
        var polls: u32 = 0;
        while (got < len.len) {
            const n = c.recv(l.fd, @as([*]u8, &len) + got, len.len - got, c.MSG.DONTWAIT);
            if (n > 0) {
                got += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN => {},
                .INTR => continue,
                else => return error.PeerDown,
            }
            polls +%= 1;
            if (got == 0 and polls % 64 == 0 and monotonicNs() -% t0 > spin_ns) return null;
            std.atomic.spinLoopHint();
        }
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }

    fn monotonicNs() u64 {
        var ts: c.timespec = undefined;
        _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    /// The kind byte of the next frame (the first byte of its payload) when its length and that byte have arrived, without
    /// reading anything; null when they have not. A frame's other bytes may still be on their way.
    pub fn peekKind(l: Link) !?u8 {
        return switch (try l.peekHead()) {
            .kind => |k| k,
            .empty, .partial => null,
        };
    }

    /// What the socket holds of the next frame's first five bytes (its length and its kind byte), without reading anything.
    pub const Head = union(enum) {
        empty, // nothing there
        partial, // some of the five bytes: the sender wrote the length and its payload has not yet come
        kind: u8,
    };

    pub fn peekHead(l: Link) !Head {
        var head: [5]u8 = undefined;
        const n = c.recv(l.fd, &head, head.len, c.MSG.PEEK | c.MSG.DONTWAIT);
        if (n == 0) return error.PeerDown;
        if (n < 0) switch (c.errno(n)) {
            .AGAIN => return .empty,
            .INTR => return .empty,
            else => return error.PeerDown,
        };
        return if (n >= head.len) .{ .kind = head[4] } else .partial;
    }

    /// The next frame into `buf` (a frame longer than `buf` is refused); its bytes.
    pub fn recv(l: Link, buf: []u8) ![]u8 {
        const held = where.enter(.link_recv, 0, 0);
        defer where.leave(held);
        var len: [4]u8 = undefined;
        try l.readAll(&len);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }
};

/// Rank 0's ports for its followers: follower g's (g = 1 to n) is `port + g - 1` on `ips[g - 1]`. All of them listen
/// from `open` on, so a follower that is ready early connects into its port's backlog (the kernel completes the
/// handshake) and waits there while rank 0 loads its weights or takes the followers before it.
pub const Ports = struct {
    fds: [3]c.fd_t = .{ -1, -1, -1 },

    pub fn open(ips: []const [4]u8, port: u16) !Ports {
        if (ips.len > 3) return error.TooManyFollowers;
        var p: Ports = .{};
        errdefer p.close();
        for (ips, 0..) |ip, i| p.fds[i] = try Link.bound(ip, if (port == 0) 0 else port + @as(u16, @intCast(i)));
        return p;
    }

    /// The port follower g's socket listens on (g from 1): what the kernel gave it when `open` was asked for port 0.
    pub fn portOf(p: Ports, g: usize) !u16 {
        const fd = p.fds[g - 1];
        if (fd < 0) return error.NoSuchFollower;
        var a: c.sockaddr.in = undefined;
        var len: c.socklen_t = @sizeOf(c.sockaddr.in);
        if (c.getsockname(fd, @ptrCast(&a), &len) != 0) return error.NoSockName;
        return std.mem.bigToNative(u16, a.port);
    }

    /// Follower g's link (g from 1): its connection, waited for if it has not come yet; its port then closes.
    pub fn accept(p: *Ports, g: usize) !Link {
        const fd = p.fds[g - 1];
        if (fd < 0) return error.NoSuchFollower;
        const peer = c.accept(fd, null, null);
        if (peer < 0) return error.AcceptFailed;
        _ = c.close(fd);
        p.fds[g - 1] = -1;
        Link.tune(peer);
        return .{ .fd = peer };
    }

    pub fn close(p: *Ports) void {
        for (&p.fds) |*fd| if (fd.* >= 0) {
            _ = c.close(fd.*);
            fd.* = -1;
        };
    }
};

/// Linux poll's POLLRDHUP (std's POLL leaves it out): the peer shut its end. Frames waiting to be read do not set it.
const POLLRDHUP: i16 = 0x2000;

/// A thread that ends this process when a peer rank goes. The ranks run every step together, so a rank that stops (an error,
/// a crash, a kill) leaves the others waiting inside their next exchange for good, their GPUs spinning on a word it will
/// never write; nothing else looks at the dead rank's link while a wait on the GPU is all the engine thread does. When a
/// watched link's peer shuts its end and nothing stops the watch within `grace_ms` (a clean end of the peer comes with this
/// process's own end of the watch first), the watch says which rank went, runs `stop_hook` (the engine fails its transports
/// there: the abort reaches the ranks still up and releases every queued wait) and exits 1. The watch only sleeps in poll:
/// it touches no state of the engine thread's. Stop it before this process ends its links itself.
pub const Watch = struct {
    fds: [max_links]c.fd_t = undefined,
    ranks: [max_links]u32 = undefined,
    n: usize = 0,
    rank: u32 = 0, // this process's rank
    grace_ms: u32 = 5000,
    stopped: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// What runs once a peer has been gone for the grace, after the line naming it and before the exit: told `ctx` and the rank
    /// that went. It runs on the watch's thread.
    stop_hook: ?*const fn (ctx: ?*anyopaque, rank: u32) void = null,
    ctx: ?*anyopaque = null,
    /// A test's stand-in for the line and the exit: told the rank that went.
    on_gone: ?*const fn (w: *Watch, rank: u32) void = null,

    pub const max_links = 4;
    /// How long a look at the links waits before the stop flag is read again (the longest a stop takes).
    const look_ms = 100;

    /// Watches `l`, the link to rank `peer_rank`.
    pub fn add(w: *Watch, l: Link, peer_rank: u32) !void {
        if (w.n == max_links) return error.TooManyLinks;
        w.fds[w.n] = l.fd;
        w.ranks[w.n] = peer_rank;
        w.n += 1;
    }

    pub fn start(w: *Watch) !void {
        if (w.n == 0 or w.thread != null) return;
        w.stopped.store(false, .release);
        w.thread = try std.Thread.spawn(.{}, run, .{w});
    }

    /// Ends the watch (nothing when it never started).
    pub fn stop(w: *Watch) void {
        w.stopped.store(true, .release);
        if (w.thread) |t| t.join();
        w.thread = null;
    }

    fn run(w: *Watch) void {
        var p: [max_links]c.pollfd = undefined;
        for (p[0..w.n], w.fds[0..w.n]) |*q, fd| q.* = .{ .fd = fd, .events = POLLRDHUP, .revents = 0 };
        while (!w.stopped.load(.acquire)) {
            const r = c.poll(&p, @intCast(w.n), look_ms);
            if (r == 0) continue;
            if (r < 0) { // (interrupted: look again; else wait a while rather than spin on the error)
                _ = c.poll(&p, 0, look_ms);
                continue;
            }
            const went = for (p[0..w.n], 0..) |q, i| {
                if (q.revents & c.POLL.NVAL != 0) return; // (a link closed under the watch: this process is ending it)
                if (q.revents & (POLLRDHUP | c.POLL.HUP | c.POLL.ERR) != 0) break i;
            } else continue;
            var waited: u32 = 0;
            while (waited < w.grace_ms) : (waited += look_ms) {
                if (w.stopped.load(.acquire)) return;
                _ = c.poll(&p, 0, look_ms); // (no descriptors: a sleep)
            }
            if (w.stopped.load(.acquire)) return;
            const gone = w.ranks[went];
            if (w.on_gone == null) std.debug.print("dsv41: rank {d} stopped (its link to rank {d} closed), so rank {d} stops too: the ranks run every step together, and it would wait for rank {d} for good (rank {d}'s own output says why it stopped)\n", .{ gone, w.rank, w.rank, gone, gone });
            if (w.stop_hook) |f| f(w.ctx, gone);
            if (w.on_gone) |f| return f(w, gone);
            c._exit(1);
        }
    }
};

/// "192.0.2.7" as four bytes.
pub fn parseIp(text: []const u8) ![4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    for (&out) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    if (it.next() != null) return error.BadAddress;
    return out;
}

test "addresses parse as four bytes" {
    try std.testing.expectEqual([4]u8{ 192, 0, 2, 7 }, try parseIp("192.0.2.7"));
    try std.testing.expectError(error.BadAddress, parseIp("10.200.30"));
}

test "frames cross a loopback link in order" {
    const lo: [4]u8 = .{ 127, 0, 0, 1 };
    var ports = try Ports.open(&.{lo}, 0); // the kernel's port: another copy of this test on the host cannot share it
    defer ports.close();
    const port = try ports.portOf(1);
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            defer l.close();
            l.send("step 1") catch return;
            l.send(&.{}) catch return;
            l.send("polled") catch return;
            l.send("late") catch return;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try ports.accept(1);
    defer l.close();
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("step 1", try l.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try l.recv(&buf)).len);
    try std.testing.expectEqualStrings("polled", try l.recvSpin(&buf, 1000 * std.time.ns_per_ms));
    try std.testing.expectEqualStrings("late", try l.recvSpin(&buf, 0));
    t.join();
}

test "peekKind sees the next frame's kind without reading it, and nothing before the length and the kind byte have come" {
    var fds: [2]c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &fds));
    const a: Link = .{ .fd = fds[0] };
    const b: Link = .{ .fd = fds[1] };
    defer {
        _ = c.close(fds[0]);
        _ = c.close(fds[1]);
    }
    try std.testing.expectEqual(@as(?u8, null), try b.peekKind());
    try std.testing.expect(!b.wait(1000));
    try a.writeAll(&[_]u8{ 7, 0, 0, 0 }, 0); // a length of 7 and no payload yet
    try std.testing.expect(b.wait(50000));
    try std.testing.expectEqual(@as(?u8, null), try b.peekKind());
    try std.testing.expectEqual(Link.Head.partial, try b.peekHead());
    try a.writeAll(&[_]u8{ 3, 1, 2, 3, 4, 5, 6 }, 0); // kind 3 (verify), its step and more
    try std.testing.expectEqual(@as(?u8, 3), try b.peekKind());
    try std.testing.expectEqual(@as(?u8, 3), try b.peekKind()); // (nothing was read)
    var buf: [64]u8 = undefined;
    const f = try b.recv(&buf);
    try std.testing.expectEqual(@as(usize, 7), f.len);
    try std.testing.expectEqual(@as(u8, 3), f[0]);
    try std.testing.expectEqual(@as(?u8, null), try b.peekKind());
}

test "a frame's length and payload reach the receiver together: the first bytes are never seen alone" {
    var fds: [2]c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &fds));
    defer {
        _ = c.close(fds[0]);
        _ = c.close(fds[1]);
    }
    const frames = 200_000;
    const Sender = struct {
        fn run(l: Link) void {
            const body = [_]u8{ 3, 0, 0, 0, 0, 0, 0, 0, 0 };
            for (0..frames) |_| l.send(&body) catch return;
        }
    };
    const rx: Link = .{ .fd = fds[1] };
    const t = try std.Thread.spawn(.{}, Sender.run, .{Link{ .fd = fds[0] }});
    var partial: u32 = 0;
    var buf: [64]u8 = undefined;
    for (0..frames) |_| {
        // as a follower's gate does: the head is peeked as soon as the socket holds anything
        var h = try rx.peekHead();
        while (h == .empty) h = try rx.peekHead();
        if (h == .partial) partial += 1;
        _ = try rx.recv(&buf);
    }
    t.join();
    std.debug.print("\nframes seen with only part of their first five bytes: {d} of {d}\n", .{ partial, frames });
    try std.testing.expectEqual(@as(u32, 0), partial);
}

test "a frame above the one-call size crosses intact, length first" {
    var fds: [2]c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &fds));
    defer {
        _ = c.close(fds[0]);
        _ = c.close(fds[1]);
    }
    var big: [3 * Link.one_call + 5]u8 = undefined;
    for (&big, 0..) |*b, i| b.* = @truncate(i *% 7 +% 3);
    const Tx = struct {
        fn run(l: Link, b: []const u8) void {
            l.send(b) catch {};
            l.send(&.{9}) catch {};
        }
    };
    const t = try std.Thread.spawn(.{}, Tx.run, .{ Link{ .fd = fds[0] }, @as([]const u8, &big) });
    var buf: [4 * Link.one_call]u8 = undefined;
    const rx: Link = .{ .fd = fds[1] };
    try std.testing.expectEqualSlices(u8, &big, try rx.recv(&buf));
    try std.testing.expectEqualSlices(u8, &.{9}, try rx.recv(&buf));
    t.join();
}

test "a frame of exactly the one-call size and one byte more both cross intact" {
    var fds: [2]c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &fds));
    defer {
        _ = c.close(fds[0]);
        _ = c.close(fds[1]);
    }
    var a: [Link.one_call]u8 = undefined;
    var b: [Link.one_call + 1]u8 = undefined;
    for (&a, 0..) |*x, i| x.* = @truncate(i *% 5 +% 1);
    for (&b, 0..) |*x, i| x.* = @truncate(i *% 11 +% 2);
    const tx: Link = .{ .fd = fds[0] };
    const rx: Link = .{ .fd = fds[1] };
    try tx.send(&a);
    try tx.send(&b);
    try tx.send(&.{});
    var buf: [4 * Link.one_call]u8 = undefined;
    try std.testing.expectEqualSlices(u8, &a, try rx.recv(&buf));
    try std.testing.expectEqualSlices(u8, &b, try rx.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try rx.recv(&buf)).len);
}

/// What a test's watch saw: the rank that went (0: none yet; the tests' peers are ranks 1 and up), the order of its two steps
/// (the hook, then the stand-in for the exit) and when the stand-in ran.
var seen_gone: std.atomic.Value(u32) = .init(0);
var hook_seen: std.atomic.Value(u32) = .init(0);
var hook_before_gone: std.atomic.Value(bool) = .init(false);
var gone_at_ns: std.atomic.Value(u64) = .init(0);

fn sawHook(_: ?*anyopaque, rank: u32) void {
    hook_seen.store(rank, .release);
}

fn sawGone(_: *Watch, rank: u32) void {
    hook_before_gone.store(hook_seen.load(.acquire) == rank, .release);
    gone_at_ns.store(Link.monotonicNs(), .release);
    seen_gone.store(rank, .release);
}

fn resetSeen() void {
    seen_gone.store(0, .release);
    hook_seen.store(0, .release);
    hook_before_gone.store(false, .release);
    gone_at_ns.store(0, .release);
}

/// Waits up to `ms` for a test's watch to see a rank go.
fn awaitGone(ms: u32) u32 {
    var p: [1]c.pollfd = undefined;
    var waited: u32 = 0;
    while (waited < ms) : (waited += 20) {
        const r = seen_gone.load(.acquire);
        if (r != 0) return r;
        _ = c.poll(&p, 0, 20);
    }
    return seen_gone.load(.acquire);
}

/// A loopback pair: the listening side's link and the peer thread's, which `Peer` runs.
fn pair(comptime Peer: type) !struct { l: Link, t: std.Thread } {
    var ports = try Ports.open(&.{.{ 127, 0, 0, 1 }}, 0);
    defer ports.close();
    const port = try ports.portOf(1);
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    return .{ .l = try ports.accept(1), .t = t };
}

test "a killed peer's link is named within the grace: the line's rank, the stop hook first, frames on the link are no close" {
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            l.send("a frame") catch {};
            var buf: [8]u8 = undefined;
            _ = l.recv(&buf) catch {}; // the test's go: then this rank is gone
            l.close();
        }
    };
    resetSeen();
    var pr = try pair(Peer);
    defer pr.l.close();
    var w: Watch = .{ .rank = 0, .grace_ms = 300, .stop_hook = sawHook, .on_gone = sawGone };
    try w.add(pr.l, 3);
    try w.start();
    defer w.stop();
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("a frame", try pr.l.recv(&buf));
    try std.testing.expectEqual(@as(u32, 0), awaitGone(500)); // a frame, read or not, is no close
    try pr.l.send("go");
    pr.t.join(); // the peer's link is closed from here
    const closed = Link.monotonicNs();
    try std.testing.expectEqual(@as(u32, 3), awaitGone(5000));
    try std.testing.expectEqual(@as(u32, 3), hook_seen.load(.acquire));
    try std.testing.expect(hook_before_gone.load(.acquire)); // (the engine's transports fail before the process ends)
    // nothing before the grace, and the grace plus a look after it at the most
    const took_ms = (gone_at_ns.load(.acquire) -| closed) / std.time.ns_per_ms;
    try std.testing.expect(took_ms >= 250);
    try std.testing.expect(took_ms <= 300 + 2 * Watch.look_ms + 400);
}

test "the watch names the link that closed, of several" {
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            var buf: [8]u8 = undefined;
            _ = l.recv(&buf) catch {};
            l.close();
        }
    };
    resetSeen();
    var a = try pair(Peer);
    defer a.l.close();
    var b = try pair(Peer);
    defer b.l.close();
    var w: Watch = .{ .rank = 0, .grace_ms = 100, .on_gone = sawGone };
    try w.add(a.l, 1);
    try w.add(b.l, 2);
    try w.start();
    defer w.stop();
    try b.l.send("go"); // rank 2 goes; rank 1 stays
    b.t.join();
    try std.testing.expectEqual(@as(u32, 2), awaitGone(5000));
    try a.l.send("go");
    a.t.join();
}

test "a stopped watch lets its links close quietly" {
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            var buf: [8]u8 = undefined;
            _ = l.recv(&buf) catch {};
            l.close();
        }
    };
    resetSeen();
    var pr = try pair(Peer);
    defer pr.l.close();
    var w: Watch = .{ .rank = 1, .grace_ms = 100, .stop_hook = sawHook, .on_gone = sawGone };
    try w.add(pr.l, 2);
    try w.start();
    w.stop(); // (rank 0's done, or a follower reading it)
    try pr.l.send("go");
    pr.t.join();
    try std.testing.expectEqual(@as(u32, 0), awaitGone(400));
    try std.testing.expectEqual(@as(u32, 0), hook_seen.load(.acquire));
}

test "a watch's grace lets this process end first" {
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            l.close();
        }
    };
    resetSeen();
    var pr = try pair(Peer);
    defer pr.l.close();
    var w: Watch = .{ .rank = 2, .grace_ms = 3000, .stop_hook = sawHook, .on_gone = sawGone };
    try w.add(pr.l, 1);
    try w.start();
    pr.t.join();
    try std.testing.expectEqual(@as(u32, 0), awaitGone(300)); // closed, but within the grace
    w.stop(); // this process ends its links itself in time
    try std.testing.expectEqual(@as(u32, 0), awaitGone(100));
    try std.testing.expectEqual(@as(u32, 0), hook_seen.load(.acquire));
}

test "a watch names the rank whose link closed, and frames on the link are no close" {
    const io = std.testing.io;
    const port: u16 = 29500 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            l.send("a frame") catch {};
            var buf: [8]u8 = undefined;
            _ = l.recv(&buf) catch {}; // the test's go: then this rank stops
            l.close();
        }
    };
    seen_gone.store(0, .release);
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var w: Watch = .{ .rank = 0, .grace_ms = 100, .on_gone = sawGone };
    try w.add(l, 3);
    try w.start();
    defer w.stop();
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("a frame", try l.recv(&buf));
    try std.testing.expectEqual(@as(u32, 0), awaitGone(400)); // a frame, read or not, is no close
    try l.send("go");
    t.join();
    try std.testing.expectEqual(@as(u32, 3), awaitGone(5000));
}
