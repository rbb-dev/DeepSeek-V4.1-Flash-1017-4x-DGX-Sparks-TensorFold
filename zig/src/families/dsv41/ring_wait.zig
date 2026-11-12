//! The decode rings without a proxy thread (TF_DS_RING_WAIT=1): each queue pair of a ring is a Core Direct master made through DEVX
//! (mlx5_devx.zig), its send queue and doorbell record in host memory. The engine thread writes every exchange's chain
//! `[WAIT doorbell >= q][data WRITE][inline generation-word WRITE]` into the queue when the exchange is LAUNCHED (an eager gather's own
//! call, or a captured stretch's replay), and rings the queue; the card runs the WAIT on the staging kernel's doorbell store, then
//! the WRITEs. Nothing polls and nothing wakes: a janitor thread looks every few milliseconds for a failure.
//!
//! The send credit (what `sent[slot]` is on the proxy path) is the card's completion of the word WRITE of exchange q, read by the
//! staging kernel itself: one completion queue a queue pair, the word WRITE the only signalled entry, so exchange q's completion is entry
//! q - 1 of every queue pair's completion queue (out/CREDIT.md: why a completion and not the peers' words). `cqeState` is the one
//! statement of the entry's test; rdma_gather.cu's `cqe_state` is its transliteration, and the two are checked against each other.
//!
//! The doorbell word is stored BIG-ENDIAN (the card compares the 64-bit word as a big-endian number: ENGINE-FACTS, M19g) and the WAIT is
//! the INVERT|BIGGER condition (0x12), which fires iff word >= value.
//!
//! The order of the failure is this file's contract (`Backend.fence`): the queue pairs go to ERR (an armed WAIT is flushed, not released)
//! and are destroyed before anything the card reads (the registrations, the doorbell's memory) is unpinned.

const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const mx = @import("mlx5_devx.zig");
const seqs = @import("rdma_seq.zig");

pub const max_ranks = seqs.max_ranks;
pub const max_devices = seqs.max_devices;

// -- the compare ----------------------------------------------------------------------------------------------------------------

/// The WAIT's compare of the doorbell word with a sequence (measured on our ConnectX-7s: the card reads the word as a
/// big-endian number; BIGGER with the invert flag fires iff word >= value).
pub const wait_cond: mx.WaitCond = .bigger;
pub const wait_invert = true;
pub const wait_mask: u64 = ~@as(u64, 0);

/// The WAIT entry's value for sequence `q`: the bytes of the doorbell word that say `q`, raw in the entry (mlx5_devx.waitWqe stores it
/// as given).
pub fn waitValue(q: u64) u64 {
    return std.mem.nativeToBig(u64, q);
}

/// What the staging kernel stores as the doorbell for sequence `q` (`__builtin_bswap64(q)`): the same bytes as `waitValue`.
pub fn bellWord(q: u64) u64 {
    return std.mem.nativeToBig(u64, q);
}

// -- the completion entry --------------------------------------------------------------------------------------------------------

pub const CqeState = enum { empty, ready, failed };

/// An entry's state from its `op_own` byte, for the entry number `idx` of a ring of `cnt` (a power of two) entries: empty (never written
/// by the card, or the last lap's), ready (a successful completion of this lap), failed (an error completion of this lap). The owner bit
/// of the first lap is 0; libmlx5 starts every entry as opcode 0xf with owner 1.
pub fn cqeState(op_own: u8, idx: u64, cnt: u32) CqeState {
    const opc = op_own >> 4;
    if (opc == mx.cqe_invalid) return .empty;
    const lap: u8 = @intFromBool(idx & cnt != 0); // bit log2(cnt) of the entry number: the lap (cnt is a power of two)
    if ((op_own & 1) != lap) return .empty;
    if (opc == mx.cqe_req_err or opc == mx.cqe_resp_err) return .failed;
    return .ready;
}

pub const Credit = enum { wait, granted, failed };

/// The staging kernel's test over the entry `idx` of every queue pair's completion queue: granted when every one is ready, failed when
/// any is, else wait.
pub fn creditOf(states: []const CqeState) Credit {
    var all = true;
    for (states) |s| switch (s) {
        .failed => return .failed,
        .empty => all = false,
        .ready => {},
    };
    return if (all) .granted else .wait;
}

/// The completion entry exchange `q` (1 on) leaves on every queue pair: its word WRITE is the one signalled entry of the chain.
pub fn cqeIndexOf(q: u64) u64 {
    std.debug.assert(q >= 1);
    return q - 1;
}

/// The device's view of one queue pair's completion queue, as rdma_gather.cu's `CqRef` reads it (the staging kernel's credit table).
pub const CqRef = extern struct { base: u64, cnt: u32, size: u32 };

comptime {
    std.debug.assert(@sizeOf(CqRef) == 16);
}

/// The most lanes a ring has (rdma_gather.cu's MAX_CQ).
pub const max_lanes = 16;

/// The staging kernels' credit table, passed by value in their parameters (no load of it from host memory): the lanes' completion queues.
pub const CqTab = extern struct {
    n: u32 = 0,
    pad: u32 = 0,
    e: [max_lanes]CqRef = @splat(.{ .base = 0, .cnt = 0, .size = 0 }),
};

comptime {
    std.debug.assert(@sizeOf(CqTab) == 264 and @alignOf(CqTab) == 8);
}

// -- one exchange's chain ----------------------------------------------------------------------------------------------------------

pub const Part = struct { dst: u64, rkey: u32, src: u64, lkey: u32, len: u32 };

/// One exchange on one lane: the WAIT's sequence and doorbell, the data part (none when the lane's part is empty) and the generation word.
pub const Chain = struct {
    wait_for: u64,
    bell_va: u64,
    bell_lkey: u32,
    data: ?Part,
    word_dst: u64,
    word_rkey: u32,
    word: u64,
};

/// The send-queue entries a chain takes (a WAIT, the data WRITE if any, the word).
pub fn chainEntries(has_data: bool) u32 {
    return if (has_data) 3 else 2;
}

pub const page = 4096;

/// A queue pair of the ring, its send queue, doorbell record and completion queue.
pub const Qp = struct {
    obj: ?*mx.DevxObj = null,
    qpn: u32 = 0,
    umem: ?*mx.DevxUmem = null,
    /// The send queue, then the doorbell record: one page-aligned block the card knows as a umem.
    mem: []align(page) u8 = &.{},
    depth: u32 = 0,
    dbrec: [*]volatile u32 = undefined,
    /// Entries written (the producer index), the producer index last rung, and the producer index after each chain (by chain number).
    pi: u32 = 0,
    rung: u32 = 0,
    ends: []u32 = &.{},
    chains: u64 = 0,
    cq: Cq = .{},
    /// The completion queue's consumer index (entries released to the card) and the completions seen in order.
    ci: u64 = 0,
    done: u64 = 0,
    rail: u8 = 0,
    peer: u8 = 0,
    uar: ?*mx.DevxUar = null,

    fn slot(q: *Qp, i: u32) *[mx.wqebb]u8 {
        return @ptrCast(q.mem[(i & (q.depth - 1)) * mx.wqebb ..][0..mx.wqebb]);
    }

    /// The producer index the card has retired: the end of the last chain whose completion was seen.
    fn retired(q: *const Qp) u32 {
        if (q.done == 0) return 0;
        return q.ends[@intCast((q.done - 1) & (q.ends.len - 1))];
    }
};

/// A queue pair's completion queue as the card writes it: libmlx5's verbs queue (its memory, doorbell record, entry count and size)
/// and the device address the staging kernel reads it at.
pub const Cq = struct {
    buf: [*]u8 = undefined,
    dbrec: [*]volatile u32 = undefined,
    cnt: u32 = 0,
    size: u32 = 0,
    cqn: u32 = 0,
    dev: u64 = 0,
    handle: ?*anyopaque = null, // the verbs queue (the caller destroys it)
    reg: ?*anyopaque = null, // the address the caller registered for the GPU (the caller unregisters it)
};

fn cqeBase(c: *const Cq, idx: u64) [*]u8 {
    return c.buf + @as(usize, @intCast(idx & (c.cnt - 1))) * c.size + (if (c.size == 128) @as(usize, 64) else 0);
}

fn cqeOwn(c: *const Cq, idx: u64) u8 {
    return @atomicLoad(u8, &cqeBase(c, idx)[63], .acquire);
}

/// What an error completion says, for the line the failure writes.
pub const CqeError = struct { opcode: u8, syndrome: u8, vendor: u8, qpn: u32, wqe: u16 };

fn cqeError(c: *const Cq, idx: u64) CqeError {
    const e = cqeBase(c, idx);
    return .{ .opcode = e[63] >> 4, .syndrome = e[55], .vendor = e[54], .qpn = std.mem.readInt(u32, e[56..60], .big) & 0xff_ffff, .wqe = std.mem.readInt(u16, e[60..62], .big) };
}

// -- what posting costs: counters the log line reports ---------------------------------------------------------------------------------

/// The engine thread's posting, summed over every ring of the process (relaxed counters; the engine thread writes them, the log reads).
pub const Stats = struct {
    launches: std.atomic.Value(u64) = .init(0), // graph launches that had a plan, and the time `replay` took posting them
    launch_ns: std.atomic.Value(u64) = .init(0),
    launch_max_ns: std.atomic.Value(u64) = .init(0),
    launch_exchanges: std.atomic.Value(u64) = .init(0),
    posts: std.atomic.Value(u64) = .init(0), // exchanges written (a replay's and an eager launch's), the time of each call
    post_ns: std.atomic.Value(u64) = .init(0),
    entries: std.atomic.Value(u64) = .init(0), // send-queue entries written
    rings: std.atomic.Value(u64) = .init(0), // doorbell rings (a record store, two barriers and a register store each)
    waits: std.atomic.Value(u64) = .init(0), // posts that had to wait for room
};

pub var stats: Stats = .{};

    _ = c.fetchAdd(n, .monotonic);
}

/// The line the Sparks read: per graph launch and per exchange, the posting's mean and the longest launch, the queue entries and rings it made.
pub fn statsLine(buf: []u8) []const u8 {
    const l = stats.launches.load(.monotonic);
    const ln = stats.launch_ns.load(.monotonic);
    const p = stats.posts.load(.monotonic);
    const pn = stats.post_ns.load(.monotonic);
    const lx = stats.launch_exchanges.load(.monotonic);
    return std.fmt.bufPrint(buf, "RING_WAIT posting (engine thread, all rings): {d} planned graph launches took {d} us in all (mean {d} ns, longest {d} us, {d} exchanges, {d} ns each); {d} exchanges written in {d} us ({d} ns each); {d} queue entries, {d} doorbell rings, {d} waits for room", .{ l, ln / 1000, if (l > 0) ln / l else 0, stats.launch_max_ns.load(.monotonic) / 1000, lx, if (lx > 0) ln / lx else 0, p, pn / 1000, if (p > 0) pn / p else 0, stats.entries.load(.monotonic), stats.rings.load(.monotonic), stats.waits.load(.monotonic) }) catch "RING_WAIT posting";
}

/// A planned launch's line goes out every this many launches (a power of two).
pub const stats_every: u64 = 1 << 14;

// -- the backend --------------------------------------------------------------------------------------------------------------------

pub const Params = struct {
    /// The send queue's size in entries as a power of two (m19c ran 11; a create the card refuses is retried one smaller, down to 9).
    log_sq_size: u8 = 12,
    /// The completion queue entries kept unconsumed at the front: the staging kernel asks for entry q - slots - 1 only.
    lag: u32,
    /// The wait for send-queue room (a stalled peer, or a GPU that has not run), then the post fails.
    sq_wait_ms: u64 = 10_000,
    /// The janitor's look interval.
    sweep_ms: u32 = 20,
    /// The wait for the card to flush a queue pair taken to ERR before it is destroyed.
    settle_us: u32 = 2_000,
};

/// A lane's fixed addressing: everything of a chain but its sequence and size.
pub const Lane = struct {
    qp: *Qp,
    peer: u8,
    lane: u8,
    lanes: u32, // the peer's lane count (the parts' split)
    peer_recv: u64, // the peer's receive ring
    recv_rkey: u32,
    lkey_send: u32,
    bell_va: u64,
    bell_lkey: u32,
    gen_va: u64, // the peer's generation word of this rank's lane
    gen_rkey: u32,
};

/// The ring's geometry as the chains need it.
pub const Geometry = struct {
    rank: u32,
    world: u32,
    slots: u32,
    max_bytes: u64,
    send_va: u64, // the host address of the send region (slot 0)
    atomic_depth: u8 = 0, // the ATOMICs in flight its queue pairs are set up for (the ring's slots, capped by the card's limits)
};

/// The chain of exchange `q` of `n` bytes on `l`, its data read from send slot `send_slot`.
pub fn laneChain(g: Geometry, l: Lane, q: u64, n: u64, send_slot: usize) Chain {
    const pt = seqs.part(n, l.lanes, l.lane);
    const at: u64 = (slot * g.world + g.rank) * g.max_bytes + pt.off;
    return .{
        .wait_for = q,
        .bell_va = l.bell_va,
        .bell_lkey = l.bell_lkey,
        .data = if (pt.len > 0) .{ .dst = l.peer_recv + at, .rkey = l.recv_rkey, .src = g.send_va + send_slot * g.max_bytes + pt.off, .lkey = l.lkey_send, .len = @intCast(pt.len) } else null,
        .word_dst = l.gen_va,
        .word_rkey = l.gen_rkey,
        .word = q,
    };
}

pub const PostError = error{ Fenced, SqFull, Aborted };

/// Whether a refused create is logged as an error (the host test that provokes one turns it off: a logged error fails a test run).
pub var log_errors: bool = true;

pub const Rail = struct {
    ctx: ?mx.Ctx = null,
    pdn: u32 = 0,
    uar: ?*mx.DevxUar = null,
};

pub const Verdict = union(enum) {
    ok,
    /// An error completion: which queue pair and what it said.
    cqe: struct { rail: u8, peer: u8, err: CqeError },
    /// The doorbell holds a sequence nobody posted a chain for.
    bell_ahead: struct { bell: u64, posted: u64 },
};

pub const Backend = struct {
    gpa: std.mem.Allocator,
    mlx: *const mx.Mlx5,
    p: Params,
    geo: Geometry,
    rails: [max_devices]Rail = @splat(.{}),
    qps: [max_devices][max_ranks]?Qp = @splat(@splat(null)),
    lanes: std.ArrayList(Lane) = .empty,
    /// Held by a post (its whole write of one lane's chains, the room wait included), a sweep and the fence.
    lock: std.atomic.Value(u32) = .init(0),
    fencing: std.atomic.Value(bool) = .init(false),
    fenced: bool = false,
    /// The exchanges whose chains are written (the engine thread's counter, and the same number for the janitor's doorbell check).
    q: u64 = 0,
    posted: std.atomic.Value(u64) = .init(0),
    abort: *const u64,
    bell: *const u64,
    owner: *anyopaque,
    fail_fn: *const fn (owner: *anyopaque, why: []const u8) void,
    janitor: ?std.Thread = null,
    stop_flag: std.atomic.Value(bool) = .init(false),

    pub fn init(gpa: std.mem.Allocator, mlx: *const mx.Mlx5, p: Params, geo: Geometry, abort: *const u64, bell: *const u64, owner: *anyopaque, fail_fn: *const fn (*anyopaque, []const u8) void) !*Backend {
        const b = try gpa.create(Backend);
        b.* = .{ .gpa = gpa, .mlx = mlx, .p = p, .geo = geo, .abort = abort, .bell = bell, .owner = owner, .fail_fn = fail_fn };
        return b;
    }

    // -- the lock

    fn acquire(b: *Backend) void {
        while (b.lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn tryAcquire(b: *Backend) bool {
        return b.lock.cmpxchgStrong(0, 1, .acquire, .monotonic) == null;
    }

    fn release(b: *Backend) void {
        b.lock.store(0, .release);
    }

    // -- the queue pairs

    /// One Core Direct master queue pair of rail `r` toward `peer`, on its own completion queue `cq` (made by the caller on the rail's
    /// context: libmlx5's verbs queue, whose memory the caller has registered for the GPU). A create the card refuses is retried with a
    /// send queue half the size.
    pub fn createQp(b: *Backend, r: usize, peer: usize, cq: Cq) !void {
        const rl = &b.rails[r];
        var log = b.p.log_sq_size;
        while (true) : (log -= 1) {
            const depth: usize = @as(usize, 1) << @intCast(log);
            const dbr_off = depth * mx.wqebb;
            const total = std.mem.alignForward(usize, dbr_off + 64, page);
            const mem = try b.gpa.alignedAlloc(u8, .fromByteUnits(page), total);
            @memset(mem, 0);
            const umem = b.mlx.api.mlx5dv_devx_umem_reg(rl.ctx.?, mem.ptr, total, 1) orelse {
                b.gpa.free(mem);
                return error.QpCreateFailed;
            };
            var in: [mx.qp_msg_bytes]u8 = undefined;
            var out: [mx.small_msg_bytes]u8 = @splat(0);
            mx.createQpIn(&in, .{ .pdn = rl.pdn, .uar_page = rl.uar.?.page_id, .cqn = cq.cqn, .log_sq_size = log, .umem_id = umem.umem_id, .dbr_off = dbr_off });
            if (b.mlx.api.mlx5dv_devx_obj_create(rl.ctx.?, &in, in.len, &out, out.len)) |obj| {
                const ends = try b.gpa.alloc(u32, depth);
                @memset(ends, 0);
                b.qps[r][peer] = .{ .obj = obj, .qpn = @intCast(mx.get(&out, mx.msg.create_out_qpn)), .umem = umem, .mem = mem, .depth = @intCast(depth), .dbrec = @ptrCast(@alignCast(mem[dbr_off..].ptr)), .ends = ends, .cq = cq, .rail = @intCast(r), .peer = @intCast(peer), .uar = rl.uar };
                return;
            }
            const err = std.c._errno().*;
            _ = b.mlx.api.mlx5dv_devx_umem_dereg(umem);
            b.gpa.free(mem);
            if (log <= 9) {
                if (log_errors) std.log.err("RING_WAIT: CREATE_QP (cd_master) refused at log_sq_size {d}: errno {d} syndrome 0x{x}", .{ log, err, mx.get(&out, mx.msg.out_syndrome) });
                return error.QpCreateFailed;
            }
            std.log.warn("RING_WAIT: CREATE_QP refused at log_sq_size {d} (errno {d}, syndrome 0x{x}): trying {d}", .{ log, err, mx.get(&out, mx.msg.out_syndrome), log - 1 });
        }
    }

    /// RESET to INIT to RTR to RTS against the peer's queue pair.
    pub fn connectQp(b: *Backend, r: usize, peer: usize, remote: mx.Remote, gid_index: u8, udp_sport: u16, mtu: u8) !void {
        const q = &b.qps[r][peer].?;
        const depth: u8 = if (b.geo.atomic) b.geo.atomic_depth else 0;
        var in: [mx.qp_msg_bytes]u8 = undefined;
        var out: [mx.small_msg_bytes]u8 = undefined;
        mx.rst2initIn(&in, q.qpn);
        try b.command(q, &in, &out, "RST2INIT");
        mx.init2rtrIn(&in, q.qpn, remote, gid_index, udp_sport, mtu);
        try b.command(q, &in, &out, "INIT2RTR");
        mx.rtr2rtsIn(&in, q.qpn);
        try b.command(q, &in, &out, "RTR2RTS");
    }

    fn command(b: *Backend, q: *Qp, in: []const u8, out: []u8, what: []const u8) !void {
        @memset(out, 0);
        const rc = b.mlx.api.mlx5dv_devx_obj_modify(q.obj.?, in.ptr, in.len, out.ptr, out.len);
        if (rc != 0 or mx.get(out, mx.msg.out_status) != 0) {
            std.log.err("RING_WAIT: {s} of queue pair {d} (rail {d}, peer {d}) failed: return {d}, status {d}, syndrome 0x{x}", .{ what, q.qpn, q.rail, q.peer, rc, mx.get(out, mx.msg.out_status), mx.get(out, mx.msg.out_syndrome) });
            return error.CommandFailed;
        }
    }

    // -- the completions

    /// Advances `q.done` over the completion entries the card has written, in order: false at an error completion (not advanced past it).
    fn refreshDone(q: *Qp) bool {
        while (true) {
            switch (cqeState(cqeOwn(&q.cq, q.done), q.done, q.cq.cnt)) {
                .ready => q.done += 1,
                .failed => return false,
                .empty => return true,
            }
        }
    }

    /// Releases to the card the entries it may reuse: all but the `lag` newest completed ones (the staging kernel still reads those).
    fn consume(b: *const Backend, q: *Qp) void {
        if (q.done <= b.p.lag) return;
        const upto = q.done - b.p.lag;
        if (upto <= q.ci) return;
        q.ci = upto;
        q.cq.dbrec[0] = std.mem.nativeToBig(u32, @intCast(q.ci & 0xff_ffff));
    }

    // -- the post

    /// Room for `entries` more send-queue entries on `q`: pending entries are rung first (a stretch longer than the queue must let the
    /// card run), then the completions are read until a retired chain frees the room, bounded by `sq_wait_ms`, the abort word and a fence.
    fn reserve(b: *Backend, q: *Qp, entries: u32) PostError!void {
        if (q.pi -% q.retired() +% entries <= q.depth) return;
        bump(&stats.waits, 1);
        if (q.pi != q.rung) ringQp(q);
        const t0 = nowNs();
        var spins: u32 = 0;
        while (true) : (spins +%= 1) {
            if (b.fencing.load(.acquire)) return error.Fenced;
            if (@atomicLoad(u64, b.abort, .acquire) != 0) return error.Aborted;
            _ = refreshDone(q);
            if (q.pi -% q.retired() +% entries <= q.depth) return;
            if (spins % 1024 == 1023 and nowNs() - t0 > b.p.sq_wait_ms * std.time.ns_per_ms) return error.SqFull;
            std.atomic.spinLoopHint();
        }
    }

    fn writeChain(q: *Qp, c: Chain) void {
        var pi = q.pi;
        mx.waitWqe(q.slot(pi), @truncate(pi), q.qpn, wait_cond, wait_invert, c.bell_lkey, c.bell_va, waitValue(c.wait_for), wait_mask);
        pi +%= 1;
        if (c.data) |d| {
            mx.writeWqe(q.slot(pi), @truncate(pi), q.qpn, false, d.dst, d.rkey, d.lkey, d.src, d.len);
            pi +%= 1;
        }
        pi +%= 1;
        q.ends[@intCast(q.chains & (q.ends.len - 1))] = pi;
        q.chains += 1;
        bump(&stats.entries, pi -% q.pi);
        q.pi = pi;
    }

    /// The doorbell: the entries are in memory before the record names them, the record before the register (m19c_wait.cu:229-235).
    fn ringQp(q: *Qp) void {
        mx.wmbDev();
        q.dbrec[1] = std.mem.nativeToBig(u32, q.pi & 0xffff);
        mx.wmbDev();
        const last = q.slot(q.pi -% 1);
        const reg: *volatile u64 = @ptrCast(@alignCast(q.uar.?.reg_addr.?));
        reg.* = std.mem.bytesToValue(u64, last[0..8]);
        mx.dsbSt();
        bump(&stats.rings, 1);
        q.rung = q.pi;
    }

    /// Writes the next exchange (of `n` bytes, its data from send slot `send_slot`, or the staged slot of its sequence when null) into every lane's queue, one chain a lane. With `defer_ring` the
    /// rings wait for `flush` (a stretch's exchanges go out together); the queue is rung at once when a lane runs short of room.
    pub fn postExchange(b: *Backend, n: u64, send_slot: ?usize, defer_ring: bool) PostError!void {
        if (b.fencing.load(.acquire)) return error.Fenced;
        b.acquire();
        defer b.release();
        if (b.fenced) return error.Fenced;
            bump(&stats.posts, 1);
            bump(&stats.post_ns, nowNs() - t0);
        const q = b.q + 1;
        for (b.lanes.items) |l| {
            try b.reserve(l.qp, chainEntries(c.data != null));
            writeChain(l.qp, c);
            if (!defer_ring) ringQp(l.qp);
        }
        b.q = q;
        b.posted.store(q, .release);
    }

    /// Rings every queue with entries not yet rung.
    pub fn flush(b: *Backend) PostError!void {
        if (b.fencing.load(.acquire)) return error.Fenced;
        b.acquire();
        defer b.release();
        if (b.fenced) return error.Fenced;
        for (b.lanes.items) |l| if (l.qp.pi != l.qp.rung) ringQp(l.qp);
    }

    // -- the janitor

    /// One look: the completion entries of every queue pair (an error completion is the verdict), the doorbell against the posts, and the
    /// consumer indexes. The caller holds the lock.
    fn sweepLocked(b: *Backend) Verdict {
        for (&b.qps) |*row| for (row) |*qo| if (qo.*) |*q| {
            if (q.obj == null) continue;
            if (!refreshDone(q)) return .{ .cqe = .{ .rail = q.rail, .peer = q.peer, .err = cqeError(&q.cq, q.done) } };
            b.consume(q);
        };
        const bell = std.mem.bigToNative(u64, @atomicLoad(u64, b.bell, .acquire));
        const posted = b.posted.load(.acquire);
        if (bell > posted) return .{ .bell_ahead = .{ .bell = bell, .posted = posted } };
        return .ok;
    }

    /// A look from outside the janitor thread (a test, or the ring's check): the verdict, or `.ok` when the engine thread holds the lock.
    pub fn sweep(b: *Backend) Verdict {
        if (!b.tryAcquire()) return .ok;
        defer b.release();
        if (b.fenced) return .ok;
        return b.sweepLocked();
    }

    pub fn verdictLine(v: Verdict, buf: []u8) []const u8 {
        return switch (v) {
            .ok => "ok",
            .cqe => |c| std.fmt.bufPrint(buf, "RING_WAIT: an error completion on the queue pair toward rank {d} (rail {d}): opcode 0x{x}, syndrome 0x{x}, vendor 0x{x}, queue pair {d}, entry {d}", .{ c.peer, c.rail, c.err.opcode, c.err.syndrome, c.err.vendor, c.err.qpn, c.err.wqe }) catch "RING_WAIT: an error completion",
            .bell_ahead => |x| std.fmt.bufPrint(buf, "RING_WAIT: the doorbell holds sequence {d} and only {d} were posted: an exchange was launched that the ring did not see (a graph launched without its plan)", .{ x.bell, x.posted }) catch "RING_WAIT: the doorbell is ahead of the posts",
        };
    }

    fn janitorMain(b: *Backend) void {
        var buf: [320]u8 = undefined;
        while (!b.stop_flag.load(.acquire)) {
            sleepNs(@as(u64, b.p.sweep_ms) * std.time.ns_per_ms);
            if (b.stop_flag.load(.acquire) or b.fencing.load(.acquire)) break;
            if (!b.tryAcquire()) continue;
            const v = if (b.fenced) Verdict.ok else b.sweepLocked();
            b.release();
            if (v != .ok) {
                b.fail_fn(b.owner, verdictLine(v, &buf));
                break;
            }
        }
    }

    pub fn startJanitor(b: *Backend) !void {
        if (b.janitor != null) return;
        b.stop_flag.store(false, .release);
        b.janitor = try std.Thread.spawn(.{}, janitorMain, .{b});
    }

    pub fn stopJanitor(b: *Backend) void {
        b.stop_flag.store(true, .release);
        if (b.janitor) |t| t.join();
        b.janitor = null;
    }

    // -- the failure and the close

    /// Takes every queue pair from the card (idempotent; any thread but one holding the lock): to ERR (an armed WAIT is flushed, not
    /// released), a short settle, then destroyed. After it no post goes in and the janitor sees nothing to do.
    pub fn fence(b: *Backend) void {
        b.fencing.store(true, .release);
        b.acquire();
        defer b.release();
        if (b.fenced) return;
        b.fenced = true;
        var any = false;
        for (&b.qps) |*row| for (row) |*qo| if (qo.*) |*q| if (q.obj) |obj| {
            var in: [mx.small_msg_bytes]u8 = undefined;
            var out: [mx.small_msg_bytes]u8 = @splat(0);
            mx.qp2errIn(&in, q.qpn);
            _ = b.mlx.api.mlx5dv_devx_obj_modify(obj, &in, in.len, &out, out.len);
            any = true;
        };
        if (any) sleepNs(@as(u64, b.p.settle_us) * std.time.ns_per_us);
        for (&b.qps) |*row| for (row) |*qo| if (qo.*) |*q| if (q.obj) |obj| {
            _ = b.mlx.api.mlx5dv_devx_obj_destroy(obj);
            q.obj = null;
            if (q.umem) |u| _ = b.mlx.api.mlx5dv_devx_umem_dereg(u);
            q.umem = null;
            b.gpa.free(q.mem);
            q.mem = &.{};
            b.gpa.free(q.ends);
            q.ends = &.{};
        };
    }

    /// Everything of the backend released but the completion queues (the caller's: they outlive the queue pairs): the janitor stopped, the
    /// queue pairs taken from the card, the user access regions freed.
    pub fn close(b: *Backend) void {
        b.stopJanitor();
        b.fence();
        for (&b.rails) |*rl| {
            if (rl.uar) |u| b.mlx.api.mlx5dv_devx_free_uar(u);
            rl.* = .{};
        }
        b.lanes.deinit(b.gpa);
        const gpa = b.gpa;
        gpa.destroy(b);
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn sleepNs(ns: u64) void {
    const ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}

// -- the plan: what a captured stretch posts when it launches ----------------------------------------------------------------------------

/// What a ring offers the plan: post one exchange (n bytes, user = the in-place user whose fixed send slot holds it, or -1 for the staged
/// slot), flush the deferred rings, fail by name.
pub const Ops = struct {
    post: *const fn (ring: *anyopaque, n: u64, user: i32, defer_ring: bool) PostError!void,
    flush: *const fn (ring: *anyopaque) PostError!void,
    fail: *const fn (ring: *anyopaque, why: []const u8) void,
};

/// One exchange a captured stretch makes, in the order the stretch's graph runs them.
pub const Note = struct { ops: *const Ops, ring: *anyopaque, n: u64, user: i32 };

/// Whether any ring runs in this mode: every other path reads this one flag and nothing else (the default build's graph launches are
/// what they were).
pub var enabled: bool = false;

var spin: std.atomic.Value(u32) = .init(0);

fn lockPlans() void {
    while (spin.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}

fn unlockPlans() void {
    spin.store(0, .release);
}

const alloc = std.heap.page_allocator;

threadlocal var rec_items: std.ArrayList(Note) = .empty;
threadlocal var rec_pending: std.ArrayList(Note) = .empty;

var plans: std.AutoHashMapUnmanaged(usize, []Note) = .empty;

/// A capture begins on this thread: whatever an earlier one left is dropped.
pub fn recordBegin() void {
    if (!enabled) return;
    rec_items.clearRetainingCapacity();
    rec_pending.clearRetainingCapacity();
}

/// The capture ended: its notes wait for the instantiate that names the graph.
pub fn recordEnd() void {
    if (!enabled) return;
    std.mem.swap(std.ArrayList(Note), &rec_items, &rec_pending);
    rec_items.clearRetainingCapacity();
}

/// A ring's exchange launched while a stream is capturing: noted, not posted (nothing runs until the graph does).
pub fn note(n: Note) error{OutOfMemory}!void {
    try rec_items.append(alloc, n);
}

/// The graph `handle` was instantiated from the last capture: the plan its launches post. A graph with no exchange gets an empty plan, so a
/// handle reused after a destroyed graph never inherits the old one's.
pub fn recordInstantiate(handle: usize) void {
    if (!enabled) return;
    const copy = alloc.dupe(Note, rec_pending.items) catch return;
    lockPlans();
    defer unlockPlans();
    const got = plans.getOrPut(alloc, handle) catch {
        alloc.free(copy);
        return;
    };
    if (got.found_existing) alloc.free(got.value_ptr.*);
    got.value_ptr.* = copy;
}

/// The graph `handle` is destroyed: its plan is dropped.
pub fn forget(handle: usize) void {
    if (!enabled) return;
    lockPlans();
    defer unlockPlans();
    if (plans.fetchRemove(handle)) |kv| alloc.free(kv.value);
}

/// What `replay` returns when a ring refused: the ring has failed by name.
pub const ReplayError = error{Failed};

fn postPlan(list: []const Note) ReplayError!void {
    var seen: [8]*anyopaque = undefined;
    var ops_of: [8]*const Ops = undefined;
    var n_seen: usize = 0;
    for (list) |nt| {
        nt.ops.post(nt.ring, nt.n, nt.user, true) catch |e| {
            nt.ops.fail(nt.ring, switch (e) {
                error.Fenced => "RING_WAIT: a graph launched on a ring that was taken from the card",
                error.SqFull => "RING_WAIT: a stretch's send queue never freed room (the GPU is not running its exchanges)",
                error.Aborted => "RING_WAIT: a graph launched on an aborted ring",
            });
            return error.Failed;
        };
        var known = false;
        for (seen[0..n_seen]) |s| known = known or s == nt.ring;
        if (!known and n_seen < seen.len) {
            seen[n_seen] = nt.ring;
            ops_of[n_seen] = nt.ops;
            n_seen += 1;
        }
    }
    for (seen[0..n_seen], ops_of[0..n_seen]) |s, ops| ops.flush(s) catch {
        ops.fail(s, "RING_WAIT: a ring could not ring its queues");
        return error.Failed;
    };
}

/// Posts the exchanges of the graph `handle` (nothing for a graph with none) and rings each ring's queues once: before the graph launches.
pub fn replay(handle: usize) ReplayError!void {
    if (!enabled) return;
    lockPlans();
    defer unlockPlans();
    const list = plans.get(handle) orelse return;
    if (list.len == 0) return;
        const dt = nowNs() - t0;
        bump(&stats.launch_ns, dt);
        bump(&stats.launch_exchanges, list.len);
        if (dt > stats.launch_max_ns.load(.monotonic)) stats.launch_max_ns.store(dt, .monotonic);
        const n = stats.launches.fetchAdd(1, .monotonic) + 1;
        if (n % stats_every == 0) {
            var lb: [512]u8 = undefined;
            std.log.info("{s}", .{statsLine(&lb)});
        }
    try postPlan(list);
}

/// The graph launch every site makes: the plan's chains first, then the graph.
pub fn launchGraph(x: cuda.graph.Exec, s: cuda.Stream) !void {
    if (enabled) try replay(@intFromPtr(x.handle));
    return x.launchOn(s);
}

// -- tests: a card that decodes the bytes written -------------------------------------------------------------------------------------

const testing = std.testing;

/// A model of the card for one queue pair: it runs the send queue from its consumer index up to the rung producer index, in order, stops at a
/// WAIT whose condition does not hold, and writes each WRITE into `mem` (a flat model of every address) and a signalled entry's completion into
/// the queue pair's completion queue with the owner bit of its lap.
const Card = struct {
    /// The memory every address of the model points into: [va - base].
    mem: []u8,
    base: u64,
    /// The word the WAIT reads: its address is inside `mem`.
    cons: u32 = 0,
    cqe_n: u64 = 0,
    reads_big_endian: bool = true,
    fired: u32 = 0,
    wrote: u32 = 0,

    fn rd64(c: *const Card, va: u64) u64 {
        return std.mem.readInt(u64, c.mem[va - c.base ..][0..8], if (c.reads_big_endian) .big else .little);
    }

    fn cond(c: *const Card, w: *const [mx.wqebb]u8) bool {
        const op = std.mem.readInt(u32, w[16..20], .big);
        const va = (@as(u64, std.mem.readInt(u32, w[24..28], .big)) << 32) | (std.mem.readInt(u32, w[28..32], .big) & ~@as(u32, 7));
        const raw_v = w[32..40];
        const word = c.rd64(va);
        const value = std.mem.readInt(u64, raw_v, if (c.reads_big_endian) .big else .little);
        const code = op & 0xf;
        const invert = op & 0x10 != 0;
        const r = switch (code) {
            1 => word == value,
            2 => value > word,
            else => false,
        };
        return if (invert) !r else r;
    }

    /// Runs the queue pair `q` as far as the card would; the WAIT's word is read now.
    fn run(c: *Card, q: *Qp) void {
        while (c.cons != q.rung) {
            const w = q.slot(c.cons);
            const opc = w[3];
            if (opc == mx.opcode_wait) {
                if (!c.cond(w)) return;
                c.fired += 1;
            } else {
                // a WRITE: remote address, then the data (segment) or the inline word
                const raddr = std.mem.readInt(u64, w[16..24], .big);
                const bytes = std.mem.readInt(u32, w[32..36], .big);
                if (bytes & mx.inline_seg != 0) {
                    @memcpy(c.mem[raddr - c.base ..][0..8], w[36..44]);
                } else {
                    const laddr = std.mem.readInt(u64, w[40..48], .big);
                    @memcpy(c.mem[raddr - c.base ..][0..bytes], c.mem[laddr - c.base ..][0..bytes]);
                }
                c.wrote += 1;
                if (w[11] & mx.ctrl_cq_update != 0) c.complete(q, 0x0);
            }
            c.cons +%= 1;
        }
    }

    fn complete(c: *Card, q: *Qp, opcode: u8) void {
        const idx = c.cqe_n;
        const e = cqeBase(&q.cq, idx);
        @memset(e[0..64], 0);
        e[63] = (opcode << 4) | @intFromBool(idx & q.cq.cnt != 0);
        c.cqe_n += 1;
    }
};

fn testCq(buf: []u8, cnt: u32) Cq {
    for (0..cnt) |i| buf[i * 64 + 63] = 0xf1; // libmlx5's start: opcode 0xf, owner 1
    return .{ .buf = buf.ptr, .dbrec = @ptrCast(@alignCast(buf[cnt * 64 ..].ptr)), .cnt = cnt, .size = 64 };
}

fn testQp(depth: u32, cq: Cq) !Qp {
    const mem = try testing.allocator.alignedAlloc(u8, .fromByteUnits(page), depth * mx.wqebb + 64);
    @memset(mem, 0);
    const ends = try testing.allocator.alloc(u32, depth);
    @memset(ends, 0);
    return .{ .qpn = 0x42, .mem = mem, .depth = depth, .dbrec = @ptrCast(@alignCast(mem[depth * mx.wqebb ..].ptr)), .ends = ends, .cq = cq };
}

fn freeQp(q: *Qp) void {
    testing.allocator.free(q.mem);
    testing.allocator.free(q.ends);
}

fn testChain(q: u64, with_data: bool) Chain {
    return .{ .wait_for = q, .bell_va = 0x1000, .bell_lkey = 7, .data = if (with_data) .{ .dst = 0x2000, .rkey = 9, .src = 0x3000, .lkey = 8, .len = 16 } else null, .word_dst = 0x2100, .word_rkey = 10, .word = q };
}

test "the doorbell is stored big-endian and the WAIT fires iff word >= value: the stage kernel's word releases its own chain and not the next; a little-endian store would misfire" {
    var mem: [0x4000]u8 = @splat(0);
    var cqbuf: [64 * 8 + 64]u8 align(64) = undefined;
    var q = try testQp(64, testCq(&cqbuf, 8));
    defer freeQp(&q);
    var card: Card = .{ .mem = &mem, .base = 0 };
    // chains 1 and 2 posted; the data's source holds recognisable bytes
    @memcpy(mem[0x3000..][0..16], "0123456789abcdef");
    Backend.writeChain(&q, testChain(1, true));
    Backend.writeChain(&q, testChain(2, true));
    q.rung = q.pi;
    card.run(&q);
    try testing.expectEqual(@as(u32, 0), card.wrote); // the doorbell is 0: nothing fires
    std.mem.writeInt(u64, mem[0x1000..][0..8], 1, .big); // bellWord(1) as the kernel stores it
    try testing.expectEqual(std.mem.nativeToBig(u64, 1), bellWord(1));
    try testing.expectEqual(@as(u64, 1), std.mem.readInt(u64, mem[0x1000..][0..8], .big));
    card.run(&q);
    try testing.expectEqual(@as(u32, 1), card.fired); // chain 1's WAIT passed, chain 2's did not
    try testing.expectEqual(@as(u32, 2), card.wrote); // its data WRITE and its word WRITE
    try testing.expectEqualSlices(u8, "0123456789abcdef", mem[0x2000..][0..16]);
    try testing.expectEqual(@as(u64, 1), std.mem.readInt(u64, mem[0x2100..][0..8], builtin.cpu.arch.endian()));
    std.mem.writeInt(u64, mem[0x1000..][0..8], 2, .big);
    card.run(&q);
    try testing.expectEqual(@as(u32, 2), card.fired);
    try testing.expectEqual(@as(u64, 2), std.mem.readInt(u64, mem[0x2100..][0..8], builtin.cpu.arch.endian()));
    // the control: the GPU store of the native (little-endian) number 1 reads as 2^56 and would release every chain at once
    var card2: Card = .{ .mem = &mem, .base = 0 };
    var q2 = try testQp(64, testCq(&cqbuf, 8));
    defer freeQp(&q2);
    std.mem.writeInt(u64, mem[0x1000..][0..8], 1, .little);
    Backend.writeChain(&q2, testChain(5, false));
    q2.rung = q2.pi;
    card2.run(&q2);
    try testing.expectEqual(@as(u32, 1), card2.fired); // chain 5 fired on a word that says 1: the wrong-endian store is detected by this test
}

test "a chain is a WAIT, the data WRITE if any and a signalled word WRITE: the entries, the producer index, the end of each chain, and a wrap of the queue" {
    var cqbuf: [64 * 8 + 64]u8 align(64) = undefined;
    var q = try testQp(8, testCq(&cqbuf, 8));
    defer freeQp(&q);
    Backend.writeChain(&q, testChain(1, true));
    try testing.expectEqual(@as(u32, 3), q.pi);
    Backend.writeChain(&q, testChain(2, false));
    try testing.expectEqual(@as(u32, 5), q.pi);
    try testing.expectEqual(@as(u32, 3), q.ends[0]);
    try testing.expectEqual(@as(u32, 5), q.ends[1]);
    // only the word's entry asks for a completion
    try testing.expectEqual(@as(u8, 0), q.slot(0)[11]); // the WAIT
    try testing.expectEqual(@as(u8, 0), q.slot(1)[11]); // the data
    try testing.expectEqual(@as(u8, mx.ctrl_cq_update), q.slot(2)[11]); // the word
    try testing.expectEqual(@as(u8, mx.ctrl_cq_update), q.slot(4)[11]);
    // the WAIT names the doorbell, its key and the value as raw big-endian bytes
    const w = q.slot(0);
    try testing.expectEqual(@as(u32, 7), std.mem.readInt(u32, w[20..24], .big));
    try testing.expectEqual(@as(u32, 0x12), std.mem.readInt(u32, w[16..20], .big));
    try testing.expectEqual(@as(u64, 1), std.mem.readInt(u64, w[32..40], .big));
    try testing.expectEqual(~@as(u64, 0), std.mem.readInt(u64, w[40..48], builtin.cpu.arch.endian()));
    // the entry index in each control segment is the producer index, which wraps with the queue
    q.done = 2;
    Backend.writeChain(&q, testChain(3, true));
    Backend.writeChain(&q, testChain(4, true)); // entries 5..7 and 8..10: the queue's depth is 8
    try testing.expectEqual(@as(u32, 11), q.pi);
    try testing.expectEqual(@as(u32, 8 | 0), (std.mem.readInt(u32, q.slot(8)[0..4], .big) >> 8) & 0xffff);
    try testing.expectEqual(@as(u8, mx.opcode_wait), q.slot(8)[3]); // entry 8 sits in slot 0
}

test "the completion test: empty, a lap's ready, the last lap's stale, an error" {
    // first lap (owner 0), a ring of 4
    try testing.expectEqual(CqeState.empty, cqeState(0xf1, 0, 4)); // libmlx5's start
    try testing.expectEqual(CqeState.ready, cqeState(0x00, 0, 4));
    try testing.expectEqual(CqeState.ready, cqeState(0x00, 3, 4));
    // the second lap wants owner 1: the first lap's entry (owner 0) at index 4 is stale
    try testing.expectEqual(CqeState.empty, cqeState(0x00, 4, 4));
    try testing.expectEqual(CqeState.ready, cqeState(0x01, 4, 4));
    // the third lap wants owner 0 again: the second lap's entry is stale
    try testing.expectEqual(CqeState.empty, cqeState(0x01, 8, 4));
    try testing.expectEqual(CqeState.ready, cqeState(0x00, 8, 4));
    // an error of this lap fails; an error of the last lap is stale
    try testing.expectEqual(CqeState.failed, cqeState((mx.cqe_req_err << 4) | 0, 1, 4));
    try testing.expectEqual(CqeState.failed, cqeState((mx.cqe_resp_err << 4) | 1, 5, 4));
    try testing.expectEqual(CqeState.empty, cqeState((mx.cqe_req_err << 4) | 0, 5, 4));
    // the entry's other opcodes (a send's 0x0, and any success) are ready
    try testing.expectEqual(CqeState.ready, cqeState(0x10, 0, 4));
}

test "cqeState agrees with rdma_gather.cu's cqe_state on every (op_own, index) of rings of 4 and 4096 (the vector hash)" {
    // The CUDA transliteration (rdma_gather.cu cqe_state) was run on the same inputs on the workstation with g++ (out/CQE-VECTORS.txt); the
    // hash below is the Zig function's, the C++ gave the same one.
    var h = std.hash.Fnv1a_64.init();
    for ([_]u32{ 4, 4096 }) |cnt| {
        var own: u32 = 0;
        while (own < 256) : (own += 1) {
            var idx: u64 = 0;
            while (idx < 3 * @as(u64, cnt) + 5) : (idx += if (cnt == 4096) 97 else 1) {
                const s = cqeState(@intCast(own), idx, cnt);
                h.update(&.{@backingInt(s)});
            }
        }
    }
    try testing.expectEqual(vector_hash, h.final());
}

const vector_hash: u64 = 12574449981182225893;

test "the credit is every queue pair's completion of the exchange: waits for the slowest lane, fails on an error, is not given by the peers' words" {
    var s = [_]CqeState{ .ready, .ready, .empty };
    try testing.expectEqual(Credit.wait, creditOf(&s));
    s[2] = .ready;
    try testing.expectEqual(Credit.granted, creditOf(&s));
    s[1] = .failed;
    try testing.expectEqual(Credit.failed, creditOf(&s));
    s[0] = .empty;
    try testing.expectEqual(Credit.failed, creditOf(&s)); // a failure outranks a wait
    // the entry an exchange leaves is q - 1
    try testing.expectEqual(@as(u64, 0), cqeIndexOf(1));
    try testing.expectEqual(@as(u64, 41), cqeIndexOf(42));
}

test "NEGATIVE CONTROL for the credit: the peers' generation words do not free a slot the card has not finished reading; only the completion does" {
    // Two lanes. The model plays the rejected design: the peer's word for the next exchange has landed (the data arrived), but the lane's
    // acknowledgement has not come back, so the word WRITE's completion is not written. The stage kernel's test must refuse.
    var mem: [0x4000]u8 = @splat(0);
    var cq0: [64 * 8 + 64]u8 align(64) = undefined;
    var cq1: [64 * 8 + 64]u8 align(64) = undefined;
    var qs = [_]Qp{ try testQp(64, testCq(&cq0, 8)), try testQp(64, testCq(&cq1, 8)) };
    defer for (&qs) |*q| freeQp(q);
    var cards = [_]Card{ .{ .mem = &mem, .base = 0 }, .{ .mem = &mem, .base = 0 } };
    @memcpy(mem[0x3000..][0..16], "0123456789abcdef");
    for (&qs) |*q| Backend.writeChain(q, testChain(1, true));
    for (&qs) |*q| q.rung = q.pi;
    std.mem.writeInt(u64, mem[0x1000..][0..8], 1, .big);
    // lane 0's chain runs completely (data, word, completion); lane 1's data and word are WRITTEN (the peer's word is there) but its ack is lost
    cards[0].run(&qs[0]);
    cards[1].run(&qs[1]);
    // lane 1: erase the completion the model wrote, as if the ack had not come
    cqeBase(&qs[1].cq, 0)[63] = 0xf1;
    cards[1].cqe_n = 0;
    try testing.expectEqual(@as(u64, 1), std.mem.readInt(u64, mem[0x2100..][0..8], builtin.cpu.arch.endian())); // the peer's word says 1 on both lanes
    var st: [2]CqeState = undefined;
    for (&qs, 0..) |*q, i| st[i] = cqeState(cqeOwn(&q.cq, cqeIndexOf(1)), cqeIndexOf(1), q.cq.cnt);
    try testing.expectEqual(CqeState.ready, st[0]);
    try testing.expectEqual(CqeState.empty, st[1]);
    try testing.expectEqual(Credit.wait, creditOf(&st)); // the word alone does not grant it
    // the ack comes: now it does
    cards[1].complete(&qs[1], 0);
    for (&qs, 0..) |*q, i| st[i] = cqeState(cqeOwn(&q.cq, cqeIndexOf(1)), cqeIndexOf(1), q.cq.cnt);
    try testing.expectEqual(Credit.granted, creditOf(&st));
}

test "aliasing control: a completion of the last lap at the same index is not this exchange's; the test waits until the card has written this lap" {
    var cqbuf: [64 * 4 + 64]u8 align(64) = undefined;
    const cq = testCq(&cqbuf, 4);
    var card: Card = .{ .mem = &.{}, .base = 0 };
    var qq: Qp = .{ .cq = cq };
    // laps 0 and 1 written for entries 0..7; exchange 9 (entry 8) is the third lap's: the entry slot holds lap 1's (owner 1)
    for (0..8) |_| card.complete(&qq, 0);
    try testing.expectEqual(CqeState.empty, cqeState(cqeOwn(&cq, 8), 8, 4));
    card.complete(&qq, 0); // entry 8 written, owner 0
    try testing.expectEqual(CqeState.ready, cqeState(cqeOwn(&cq, 8), 8, 4));
}

test "the janitor releases all but `lag` completed entries to the card, advances the record, and reports an error completion" {
    var cqbuf: [64 * 16 + 64]u8 align(64) = undefined;
    var q = try testQp(64, testCq(&cqbuf, 16));
    defer freeQp(&q);
    var card: Card = .{ .mem = &.{}, .base = 0 };
    var zero: u64 = 0;
    var b: Backend = .{ .gpa = testing.allocator, .mlx = undefined, .p = .{ .lag = 4 }, .geo = undefined, .abort = &zero, .bell = &zero, .owner = undefined, .fail_fn = undefined };
    for (0..10) |_| card.complete(&q, 0);
    try testing.expect(Backend.refreshDone(&q));
    try testing.expectEqual(@as(u64, 10), q.done);
    b.consume(&q);
    try testing.expectEqual(@as(u64, 6), q.ci);
    try testing.expectEqual(std.mem.nativeToBig(u32, 6), q.cq.dbrec[0]);
    // a few more, then an error entry: done stops before it
    for (0..2) |_| card.complete(&q, 0);
    card.complete(&q, mx.cqe_req_err);
    try testing.expect(!Backend.refreshDone(&q));
    try testing.expectEqual(@as(u64, 12), q.done);
    const e = cqeError(&q.cq, q.done);
    try testing.expectEqual(@as(u8, mx.cqe_req_err), e.opcode);
}

test "room: a queue that would overflow waits for a completion, rings what is pending first, and gives up by name; never overwrites an entry the card has not retired" {
    var cqbuf: [64 * 8 + 64]u8 align(64) = undefined;
    var q = try testQp(8, testCq(&cqbuf, 8));
    defer freeQp(&q);
    var zero: u64 = 0;
    var b: Backend = .{ .gpa = testing.allocator, .mlx = undefined, .p = .{ .lag = 2, .sq_wait_ms = 20 }, .geo = undefined, .abort = &zero, .bell = &zero, .owner = undefined, .fail_fn = undefined };
    // two chains of 3 entries fill 6 of 8; a third (3) does not fit
    try b.reserve(&q, 3);
    Backend.writeChain(&q, testChain(1, true));
    try b.reserve(&q, 3);
    Backend.writeChain(&q, testChain(2, true));
    try testing.expectEqual(@as(u32, 6), q.pi);
    // ringQp is the card's door: a stand-in UAR page
    var reg: u64 = 0;
    var uar: mx.DevxUar = .{ .reg_addr = &reg, .base_addr = null, .page_id = 0, .mmap_off = 0, .comp_mask = 0 };
    q.uar = &uar;
    try testing.expectError(error.SqFull, b.reserve(&q, 3)); // nothing completed: the wait ends by name
    try testing.expectEqual(q.pi, q.rung); // the pending entries were rung before it waited
    // chain 1's completion arrives: 3 entries free
    var card: Card = .{ .mem = &.{}, .base = 0 };
    card.complete(&q, 0);
    try b.reserve(&q, 3);
    try testing.expectEqual(@as(u32, 3), q.retired());
    // an aborted ring and a fencing one refuse at once
    var one: u64 = 1;
    b.abort = &one;
    try testing.expectError(error.Aborted, b.reserve(&q, 8));
    b.abort = &zero;
    b.fencing.store(true, .release);
    try testing.expectError(error.Fenced, b.reserve(&q, 8));
}

test "the lane's chain: the part by lane, the landing slot by sequence and rank, the source slot (staged: by sequence; in place: the user's), the word's address" {
    const g: Geometry = .{ .rank = 1, .world = 4, .slots = 4, .max_bytes = 4096, .send_va = 0x10_0000 };
    var q: Qp = .{};
    const l: Lane = .{ .qp = &q, .peer = 3, .lane = 1, .lanes = 2, .peer_recv = 0x90_0000, .recv_rkey = 55, .lkey_send = 66, .bell_va = 0x2000, .bell_lkey = 77, .gen_va = 0xa0_0000 + (1 * max_devices + 1) * 8, .gen_rkey = 88 };
    // 6 exchanges so far: q = 7 lands in slot 3; 96 bytes over 2 lanes: 48 each, lane 1 at offset 48
    const c = laneChain(g, l, 7, 96, 3);
    try testing.expectEqual(@as(u64, 7), c.wait_for);
    const d = c.data.?;
    try testing.expectEqual(@as(u64, 0x90_0000 + (3 * 4 + 1) * 4096 + 48), d.dst);
    try testing.expectEqual(@as(u64, 0x10_0000 + 3 * 4096 + 48), d.src);
    try testing.expectEqual(@as(u32, 48), d.len);
    try testing.expectEqual(@as(u32, 55), d.rkey);
    try testing.expectEqual(@as(u32, 66), d.lkey);
    try testing.expectEqual(@as(u64, 7), c.word);
    try testing.expectEqual(@as(u32, 88), c.word_rkey);
    // an in-place send names its own slot (slot 5 = slots + user 1)
    const c2 = laneChain(g, l, 7, 96, 5);
    try testing.expectEqual(@as(u64, 0x10_0000 + 5 * 4096 + 48), c2.data.?.src);
    // a part of no bytes is a chain of two entries
    const c3 = laneChain(g, .{ .qp = &q, .peer = 3, .lane = 0, .lanes = 2, .peer_recv = 0x90_0000, .recv_rkey = 55, .lkey_send = 66, .bell_va = 0x2000, .bell_lkey = 77, .gen_va = 0xa0_0000, .gen_rkey = 88 }, 8, 16, 0); // 16 bytes over 2 lanes: the last lane takes the 16, lane 0 none
    try testing.expect(c3.data == null);
    try testing.expectEqual(@as(u32, 16), laneChain(g, l, 8, 16, 0).data.?.len);
    try testing.expectEqual(@as(u32, 2), chainEntries(c3.data != null));
}

const FakeMlx = struct {
    var log: [16]u16 = undefined;
    var n: usize = 0;
    var destroyed: usize = 0;
    var dereg: usize = 0;

    fn modify(_: *mx.DevxObj, in: *const anyopaque, _: usize, _: *anyopaque, _: usize) callconv(.c) c_int {
        const b: *const [2]u8 = @ptrCast(in);
        log[n] = std.mem.readInt(u16, b, .big);
        n += 1;
        return 0;
    }
    fn destroy(_: *mx.DevxObj) callconv(.c) c_int {
        log[n] = 0xdddd;
        n += 1;
        destroyed += 1;
        return 0;
    }
    fn dereg_(_: *mx.DevxUmem) callconv(.c) c_int {
        log[n] = 0xeeee;
        n += 1;
        dereg += 1;
        return 0;
    }
    fn free_uar(_: *mx.DevxUar) callconv(.c) void {}
    fn nope() noreturn {
        unreachable;
    }
    fn api() mx.Api {
        var a: mx.Api = undefined;
        a.mlx5dv_devx_obj_modify = modify;
        a.mlx5dv_devx_obj_destroy = destroy;
        a.mlx5dv_devx_umem_dereg = dereg_;
        a.mlx5dv_devx_free_uar = free_uar;
        return a;
    }
};

test "the fence: every queue pair to ERR first, then each destroyed with its umem, memory freed last; a post after it is refused; a second fence does nothing" {
    var cqbuf: [64 * 8 + 64]u8 align(64) = undefined;
    var zero: u64 = 0;
    const mlx = mx.Mlx5.fake(FakeMlx.api());
    FakeMlx.n = 0;
    var b = try Backend.init(testing.allocator, &mlx, .{ .lag = 2, .settle_us = 10 }, .{ .rank = 0, .world = 3, .slots = 4, .max_bytes = 4096, .send_va = 0x1000 }, &zero, &zero, undefined, undefined);
    var cqs: [2][64 * 8 + 64]u8 align(64) = undefined;
    _ = &cqbuf;
    for (0..2) |i| {
        var q = try testQp(8, testCq(&cqs[i], 8));
        q.obj = @ptrFromInt(0x1000 + i * 16);
        q.umem = @ptrFromInt(0x2000 + i * 16);
        q.peer = @intCast(i + 1);
        b.qps[0][i + 1] = q;
    }
    b.fence();
    // 2ERR (0x507) on both, then destroy + umem dereg for each: no destroy before the last 2ERR
    try testing.expectEqual(@as(usize, 6), FakeMlx.n);
    try testing.expectEqual(@as(u16, 0x507), FakeMlx.log[0]);
    try testing.expectEqual(@as(u16, 0x507), FakeMlx.log[1]);
    try testing.expectEqual(@as(u16, 0xdddd), FakeMlx.log[2]);
    try testing.expectEqual(@as(u16, 0xeeee), FakeMlx.log[3]);
    try testing.expectEqual(@as(u16, 0xdddd), FakeMlx.log[4]);
    try testing.expectEqual(@as(u16, 0xeeee), FakeMlx.log[5]);
    try testing.expectError(error.Fenced, b.postExchange(16, 0, false));
    try testing.expectError(error.Fenced, b.flush());
    b.fence();
    try testing.expectEqual(@as(usize, 6), FakeMlx.n);
    b.lanes.deinit(testing.allocator);
    testing.allocator.destroy(b);
}

const PlanRing = struct {
    q: u64 = 0,
    flushed: u32 = 0,
    failed: ?[]const u8 = null,
    fail_at: u64 = 0, // post fails with Fenced at this exchange (0: never)
    posted: [16]struct { n: u64, user: i32, defer_ring: bool } = undefined,

    fn post(ring: *anyopaque, n: u64, user: i32, defer_ring: bool) PostError!void {
        const r: *PlanRing = @ptrCast(@alignCast(ring));
        if (r.fail_at != 0 and r.q + 1 == r.fail_at) return error.Fenced;
        r.posted[r.q] = .{ .n = n, .user = user, .defer_ring = defer_ring };
        r.q += 1;
    }
    fn flush(ring: *anyopaque) PostError!void {
        const r: *PlanRing = @ptrCast(@alignCast(ring));
        r.flushed += 1;
    }
    fn fail(ring: *anyopaque, why: []const u8) void {
        const r: *PlanRing = @ptrCast(@alignCast(ring));
        r.failed = why;
    }
    const ops: Ops = .{ .post = post, .flush = flush, .fail = fail };
};

test "a captured stretch's exchanges are posted at each launch in the graph's order, advancing the ring's sequence by the plan's length; the capture itself posts nothing" {
    enabled = true;
    defer enabled = false;
    var r: PlanRing = .{};
    recordBegin();
    try note(.{ .ops = &PlanRing.ops, .ring = &r, .n = 64, .user = -1 });
    try note(.{ .ops = &PlanRing.ops, .ring = &r, .n = 32, .user = 1 });
    try note(.{ .ops = &PlanRing.ops, .ring = &r, .n = 16, .user = -1 });
    recordEnd();
    try testing.expectEqual(@as(u64, 0), r.q); // recording posts nothing
    recordInstantiate(0xabc);
    try replay(0xabc);
    try testing.expectEqual(@as(u64, 3), r.q);
    try testing.expectEqual(@as(u64, 32), r.posted[1].n);
    try testing.expectEqual(@as(i32, 1), r.posted[1].user);
    try testing.expect(r.posted[0].defer_ring);
    try testing.expectEqual(@as(u32, 1), r.flushed); // one ring of the queues for the whole stretch
    try replay(0xabc); // the next launch of the same graph: the next three sequences
    try testing.expectEqual(@as(u64, 6), r.q);
    try replay(0xdef); // a graph that never recorded: nothing
    try testing.expectEqual(@as(u64, 6), r.q);
    // a handle reused by a graph with no exchanges does not inherit the old plan
    recordBegin();
    recordEnd();
    recordInstantiate(0xabc);
    try replay(0xabc);
    try testing.expectEqual(@as(u64, 6), r.q);
    forget(0xabc);
}

test "a ring that refuses mid-stretch fails by name and the replay stops: nothing more is posted, nothing is flushed" {
    enabled = true;
    defer enabled = false;
    var r: PlanRing = .{ .fail_at = 2 };
    recordBegin();
    for (0..3) |_| try note(.{ .ops = &PlanRing.ops, .ring = &r, .n = 16, .user = -1 });
    recordEnd();
    recordInstantiate(0x777);
    try testing.expectError(error.Failed, replay(0x777));
    try testing.expectEqual(@as(u64, 1), r.q);
    try testing.expectEqual(@as(u32, 0), r.flushed);
    try testing.expect(r.failed != null);
    forget(0x777);
}

test "with the mode off, recording and replay do nothing at all" {
    enabled = false;
    const r: PlanRing = .{};
    recordBegin();
    try testing.expectEqual(@as(usize, 0), rec_items.items.len);
    recordInstantiate(1);
    try replay(1);
    try testing.expectEqual(@as(u64, 0), r.q);
}

test "the exchange's post: one chain a lane, in each lane's queue, the sequence advancing by one, rung once a lane unless deferred" {
    var mem: [0x8000]u8 = @splat(0);
    var cq0: [64 * 16 + 64]u8 align(64) = undefined;
    var cq1: [64 * 16 + 64]u8 align(64) = undefined;
    var zero: u64 = 0;
    var mlx = mx.Mlx5.fake(FakeMlx.api());
    var b = try Backend.init(testing.allocator, &mlx, .{ .lag = 2 }, .{ .rank = 0, .world = 2, .slots = 4, .max_bytes = 4096, .send_va = 0x4000 }, &zero, &zero, undefined, undefined);
    defer {
        b.lanes.deinit(testing.allocator);
        testing.allocator.destroy(b);
    }
    var reg: u64 = 0;
    var uar: mx.DevxUar = .{ .reg_addr = &reg, .base_addr = null, .page_id = 0, .mmap_off = 0, .comp_mask = 0 };
    var qs = [_]Qp{ try testQp(64, testCq(&cq0, 16)), try testQp(64, testCq(&cq1, 16)) };
    defer for (&qs) |*q| freeQp(q);
    for (&qs) |*q| q.uar = &uar;
    for (0..2) |k| try b.lanes.append(testing.allocator, .{ .qp = &qs[k], .peer = 1, .lane = @intCast(k), .lanes = 2, .peer_recv = 0x6000, .recv_rkey = 1, .lkey_send = 2, .bell_va = 0x100, .bell_lkey = 3, .gen_va = 0x200 + k * 8, .gen_rkey = 4 });
    try b.postExchange(64, 0, false);
    try testing.expectEqual(@as(u64, 1), b.q);
    try testing.expectEqual(@as(u64, 1), b.posted.load(.acquire));
    for (&qs) |*q| {
        try testing.expectEqual(@as(u32, 3), q.pi);
        try testing.expectEqual(q.pi, q.rung);
        try testing.expectEqual(@as(u64, 1), q.chains);
    }
    try b.postExchange(64, 1, true);
    for (&qs) |*q| {
        try testing.expectEqual(@as(u32, 6), q.pi);
        try testing.expectEqual(@as(u32, 3), q.rung); // deferred
    }
    try b.flush();
    for (&qs) |*q| try testing.expectEqual(q.pi, q.rung);
    // the doorbell check: a sequence rung that nobody posted is the janitor's verdict
    var bell: u64 = std.mem.nativeToBig(u64, 5);
    b.bell = &bell;
    switch (b.sweep()) {
        .bell_ahead => |x| {
            try testing.expectEqual(@as(u64, 5), x.bell);
            try testing.expectEqual(@as(u64, 2), x.posted);
        },
        else => return error.TestUnexpectedResult,
    }
    bell = std.mem.nativeToBig(u64, 2);
    try testing.expect(b.sweep() == .ok);
    _ = &mem;
}

const CreateMlx = struct {
    var tries: [8]u8 = undefined;
    var n: usize = 0;
    var refuse_at_or_above: u8 = 12;
    var umem: mx.DevxUmem = .{ .umem_id = 77 };
    var obj_byte: u8 = 0;
    var dereg: usize = 0;
    var modified: [8]u16 = undefined;
    var nm: usize = 0;

    fn umem_reg(_: mx.Ctx, _: ?*anyopaque, _: usize, _: u32) callconv(.c) ?*mx.DevxUmem {
        return &umem;
    }
    fn umem_dereg(_: *mx.DevxUmem) callconv(.c) c_int {
        dereg += 1;
        return 0;
    }
    fn obj_create(_: mx.Ctx, in: *const anyopaque, inlen: usize, out: *anyopaque, _: usize) callconv(.c) ?*mx.DevxObj {
        const buf: [*]const u8 = @ptrCast(in);
        const log: u8 = @intCast(mx.get(buf[0..inlen], .{ .off = (mx.msg.qpc_at + 0) * 8 + mx.qpc.log_sq_size.off, .bits = 4 }));
        tries[n] = log;
        n += 1;
        if (log >= refuse_at_or_above) return null;
        const o: [*]u8 = @ptrCast(out);
        mx.put(o[0..16], mx.msg.create_out_qpn, 0x1234);
        return @ptrCast(&obj_byte);
    }
        const b: *const [2]u8 = @ptrCast(in);
        modified[nm] = std.mem.readInt(u16, b, .big);
        nm += 1;
        _ = out;
        return 0;
    }
};

test "a queue pair create the card refuses is retried with half the send queue; the one it takes keeps its size, its queue number and its completion queue; the three transitions go in order" {
    var a: mx.Api = undefined;
    a.mlx5dv_devx_umem_reg = CreateMlx.umem_reg;
    a.mlx5dv_devx_umem_dereg = CreateMlx.umem_dereg;
    a.mlx5dv_devx_obj_create = CreateMlx.obj_create;
    a.mlx5dv_devx_obj_modify = CreateMlx.modify;
    const mlx = mx.Mlx5.fake(a);
    var zero: u64 = 0;
    var b = try Backend.init(testing.allocator, &mlx, .{ .lag = 2, .log_sq_size = 12 }, .{ .rank = 0, .world = 2, .slots = 4, .max_bytes = 4096, .send_va = 0 }, &zero, &zero, undefined, undefined);
    var uar: mx.DevxUar = .{ .reg_addr = null, .base_addr = null, .page_id = 9, .mmap_off = 0, .comp_mask = 0 };
    b.rails[0] = .{ .ctx = @ptrFromInt(8), .pdn = 3, .uar = &uar };
    CreateMlx.n = 0;
    CreateMlx.refuse_at_or_above = 12;
    var cqbuf: [64 * 8 + 64]u8 align(64) = undefined;
    const cq = testCq(&cqbuf, 8);
    try b.createQp(0, 1, cq);
    try testing.expectEqual(@as(usize, 2), CreateMlx.n);
    try testing.expectEqual(@as(u8, 12), CreateMlx.tries[0]);
    try testing.expectEqual(@as(u8, 11), CreateMlx.tries[1]);
    const q = b.qps[0][1].?;
    try testing.expectEqual(@as(u32, 2048), q.depth);
    try testing.expectEqual(@as(u32, 0x1234), q.qpn);
    try testing.expectEqual(@as(usize, 2048), q.ends.len);
    try testing.expectEqual(@as(u32, 8), q.cq.cnt);
    try testing.expectEqual(@as(usize, 1), CreateMlx.dereg); // the refused try's umem was released
    CreateMlx.nm = 0;
    try b.connectQp(0, 1, .{ .qpn = 0x55, .gid = @splat(1), .mac = .{ 0, 1, 2, 3, 4, 5 } }, 3, 50000, 5);
    try testing.expectEqualSlices(u16, &.{ mx.op.rst2init_qp, mx.op.init2rtr_qp, mx.op.rtr2rts_qp }, CreateMlx.modified[0..3]);
    // a card that refuses every size ends at 512 entries, by name
    var b2 = try Backend.init(testing.allocator, &mlx, .{ .lag = 2, .log_sq_size = 12 }, .{ .rank = 0, .world = 2, .slots = 4, .max_bytes = 4096, .send_va = 0 }, &zero, &zero, undefined, undefined);
    b2.rails[0] = .{ .ctx = @ptrFromInt(8), .pdn = 3, .uar = &uar };
    CreateMlx.n = 0;
    CreateMlx.refuse_at_or_above = 0;
    log_errors = false;
    defer log_errors = true;
    try testing.expectError(error.QpCreateFailed, b2.createQp(0, 1, cq));
    try testing.expectEqual(@as(u8, 9), CreateMlx.tries[CreateMlx.n - 1]);
    b.qps[0][1].?.obj = null;
    var qq = b.qps[0][1].?;
    testing.allocator.free(qq.mem);
    testing.allocator.free(qq.ends);
    qq.mem = &.{};
    b.lanes.deinit(testing.allocator);
    testing.allocator.destroy(b);
    b2.lanes.deinit(testing.allocator);
    testing.allocator.destroy(b2);
}

test "the posting counters: entries and rings counted as written, the exchange's time and the planned launch's, the line names them" {
    stats = .{};
    enabled = true;
    defer enabled = false;
    var cqbuf: [64 * 8 + 64]u8 align(64) = undefined;
    var q = try testQp(64, testCq(&cqbuf, 8));
    defer freeQp(&q);
    var reg: u64 = 0;
    var uar: mx.DevxUar = .{ .reg_addr = &reg, .base_addr = null, .page_id = 0, .mmap_off = 0, .comp_mask = 0 };
    q.uar = &uar;
    Backend.writeChain(&q, testChain(1, true));
    Backend.writeChain(&q, testChain(2, false));
    ringQpTest(&q);
        try testing.expectEqual(@as(u64, 5), stats.entries.load(.monotonic));
        try testing.expectEqual(@as(u64, 1), stats.rings.load(.monotonic));
    var r: PlanRing = .{};
    recordBegin();
    try note(.{ .ops = &PlanRing.ops, .ring = &r, .n = 16, .user = -1 });
    try note(.{ .ops = &PlanRing.ops, .ring = &r, .n = 16, .user = -1 });
    recordEnd();
    recordInstantiate(0x99);
    try replay(0x99);
    try replay(0x99);
    var lb: [512]u8 = undefined;
    const line = statsLine(&lb);
        try testing.expect(std.mem.indexOf(u8, line, "2 planned graph launches") != null);
        try testing.expect(std.mem.indexOf(u8, line, "5 queue entries, 1 doorbell rings") != null);
    forget(0x99);
    stats = .{};
}

fn ringQpTest(q: *Qp) void {
    Backend.ringQp(q);
}
    var b = try Backend.init(testing.allocator, &mlx, .{ .lag = 2 }, .{ .rank = 0, .world = 2, .slots = 4, .max_bytes = 4096, .send_va = 0x4000, .atomic = true, .atomic_depth = 4 }, &zero, &zero, undefined, undefined);
        var b = try Backend.init(testing.allocator, &mlx, .{ .lag = 2 }, .{ .rank = 0, .world = 2, .slots = 4, .max_bytes = 4096, .send_va = 0, .atomic = atomic, .atomic_depth = 4 }, &zero, &zero, undefined, undefined);