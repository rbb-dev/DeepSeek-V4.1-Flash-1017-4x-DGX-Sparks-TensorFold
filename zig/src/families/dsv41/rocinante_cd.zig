//! Rocinante's NIC that waits in the card: one reliable-connected queue pair a peer on each of the two rails, made through DEVX as a
//! Core Direct master (mlx5_devx.zig), its send queue and doorbell record in host memory. The engine thread writes every chain into the
//! queue when it reserves the send, behind a WAIT-on-data entry for the doorbell word the GPU's stream write sets, and rings the queue; the
//!
//! The arena and the control block are registered again under the DEVX context (every object of a queue pair belongs to one), so this
//! backend and the verbs one are never open together. A queue pair that holds a WAIT cannot carry anything else, so the failure's abort
//! notice to a peer goes over a plain verbs queue pair of its own (never given a WAIT), per rail and peer. Completions are read by hand from
//! the completion queue's memory: the verbs poll does not know a DEVX queue pair.
//!
//! The order of the failure is the contract of this file (Nic.fail_fn, close): the queue pairs are taken from the card before the doorbell
//! can hold the abort value, because that value satisfies every armed WAIT and would send unproduced data and arrival words into the
//! peers' arenas.

const std = @import("std");
const builtin = @import("builtin");
const fabric = @import("fabric");
const vabi = fabric.verbs_abi;
const Verbs = fabric.verbs.Verbs;
const roci = @import("rocinante.zig");
const rv = @import("rocinante_verbs.zig");
const mx = @import("mlx5_devx.zig");

pub const Info = rv.Info;

/// The WAIT's compare of the doorbell word with a ticket: the one place the choice is made. measured on our ConnectX-7s (between two Sparks,
/// both store orders): the card compares the 64-bit word as a BIG-ENDIAN number, so the doorbell is stored big-endian
/// (Transport.go) in this mode; the condition BIGGER with the invert flag (0x12) fires iff word >= value, with value = the ticket and the
/// mask all ones. (BIGGER alone fires iff value > word; CYCLIC_SMALLER iff word > value; EQUAL only on the exact value.) The abort value
/// 2^62, stored the same way, satisfies every armed WAIT.
pub const wait_cond: mx.WaitCond = .bigger;
pub const wait_invert = true;
pub const wait_mask: u64 = ~@as(u64, 0);

/// The value of the WAIT entry of `ticket`: the bytes of the doorbell word that says `ticket` (Transport.go), raw in the entry as m19c_wait.cu
/// puts them, no further swap.
pub fn waitValue(ticket: u64) u64 {
    return std.mem.nativeToBig(u64, ticket);
}

pub const Settings = struct {
    /// The two rails' devices (rail A, rail B).
    devices: [roci.rails][]const u8 = .{ "mlx5_0", "mlx5_2" },
    gid_index: u8 = 3,
    /// The send queue's size in entries, as a power of two. The card's largest is not known until a Spark answers (m19c ran 11): a
    /// create the card refuses is retried one smaller.
    log_sq_size: u8 = 12,
    /// One chain in this many asks for a completion, which retires the entries before it.
    signal_every: u32 = 16,
    /// The wait for send-queue room (a stalled peer or a GPU that has not rung), then the post fails.
    sq_wait_ms: u64 = 10_000,
    /// The abort notices' wait for their completions.
    abort_wait_ms: u32 = 10,
    /// The wait for the card to settle after a control block registration that is refused.
    settle_ms: u32 = 100,
    /// The MAC of the network device behind an RDMA device (the host tests give their own).
    mac_of: *const fn (device: []const u8) error{NoMac}![6]u8 = netdevMac,
};

/// Whether an error completion is logged (the host tests that provoke one turn it off: a logged error fails a test).
pub var log_errors: bool = true;


const max_chain_entries = roci.max_pieces + 2;

const Qp = struct {
    obj: ?*mx.DevxObj,
    qpn: u32,
    umem: ?*mx.DevxUmem,
    /// The send queue, then the doorbell record, one page-aligned block the card knows as a umem.
    mem: []align(roci.page) u8,
    depth: u32,
    dbrec: [*]volatile u32,
    /// Entries written (the producer index), and entries the card has completed (a signalled entry's completion retires everything before it).
    pi: u32 = 0,
    retired: u32 = 0,
    chains: u32 = 0,

    fn slot(q: *Qp, i: u32) *[mx.wqebb]u8 {
        return @ptrCast(q.mem[(i & (q.depth - 1)) * mx.wqebb ..][0..mx.wqebb]);
    }
};

const Rail = struct {
    ctx: ?mx.Ctx = null,
    pd: ?*vabi.Pd = null,
    pdn: u32 = 0,
    cq: ?*vabi.Cq = null,
    cqn: u32 = 0,
    view: mx.CqView = undefined,
    uar: ?*mx.DevxUar = null,
    mr_arena: ?*vabi.Mr = null,
    mr_ctl: ?*vabi.Mr = null, // null once fenced off: the card can no longer touch the control block
    mtu: u8 = 5,
    mac: [6]u8 = @splat(0),
    qps: [roci.world]?Qp = @splat(null),
    abort_cq: ?*vabi.Cq = null,
    abort_qps: [roci.world]?*vabi.Qp = @splat(null),
    /// Set by the fence: a post waiting for room gives up, and no entry is written into a queue pair taken away.
    fenced: bool = false,
};

pub const CdNic = struct {
    gpa: std.mem.Allocator,
    v: *const Verbs,
    mlx: *const mx.Mlx5,
    set: Settings,
    me: u8,
    arena: []u8,
    ctl: []u8,
    rail: [roci.rails]Rail = @splat(.{}),
    peers: [roci.world]Info = @splat(.{}),
    layout_hash: u64 = 0,
    /// Held by whoever reads a completion queue (the engine thread for room, the janitor for errors) and by the fence while it takes the
    /// queue pairs away.
    lock: std.atomic.Value(u32) = .init(0),
    opened: bool = false,

    /// Open both rails under DEVX, register `arena` and `ctl`, and create every queue pair (RESET). `info` then describes this node; `connect`
    /// takes every node's.
    pub fn open(gpa: std.mem.Allocator, v: *const Verbs, mlx: *const mx.Mlx5, set: Settings, me: u8, arena: []u8, ctl: []u8, layout_hash: u64) !*CdNic {
        const n = try gpa.create(CdNic);
        n.* = .{ .gpa = gpa, .v = v, .mlx = mlx, .set = set, .me = me, .arena = arena, .ctl = ctl, .layout_hash = layout_hash };
        errdefer {
            n.teardown();
            gpa.destroy(n);
        }
        if (@intFromPtr(ctl.ptr) % roci.line != 0) return error.Misaligned;
        for (0..roci.rails) |r| try n.openRail(r);
        n.opened = true;
        return n;
    }

    fn openRail(n: *CdNic, r: usize) !void {
        const api = n.v.api;
        const rl = &n.rail[r];
        const ctx = try openDevx(n.v, n.mlx, n.set.devices[r]);
        rl.ctx = ctx;
        const port = try n.v.port(ctx, 1);
        if (port.state != vabi.port_active) return error.PortDown;
        rl.mtu = @intCast(@min(port.active_mtu, @as(c_int, @intCast(mx.mtu_4k))));
        rl.pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        const depth: u32 = @as(u32, 1) << @intCast(n.set.log_sq_size);
        rl.cq = api.ibv_create_cq(ctx, @intCast(depth * roci.world), null, null, 0) orelse return error.VerbsFailed;
        var dpd: mx.DvPd = .{};
        var dcq: mx.DvCq = .{};
        var obj: mx.DvObj = .{};
        obj.pairs[6] = .{ rl.pd, &dpd };
        if (n.mlx.api.mlx5dv_init_obj(&obj, mx.obj_pd) != 0) return error.DevxUnavailable;
        rl.pdn = dpd.pdn;
        obj = .{};
        obj.pairs[1] = .{ rl.cq, &dcq };
        if (n.mlx.api.mlx5dv_init_obj(&obj, mx.obj_cq) != 0) return error.DevxUnavailable;
        rl.cqn = dcq.cqn;
        rl.view = .{ .buf = @ptrCast(dcq.buf.?), .dbrec = dcq.dbrec.?, .cqe_cnt = dcq.cqe_cnt, .cqe_size = dcq.cqe_size };
        rl.uar = n.mlx.api.mlx5dv_devx_alloc_uar(ctx, mx.uar_type_bf) orelse n.mlx.api.mlx5dv_devx_alloc_uar(ctx, mx.uar_type_nc) orelse return error.DevxUnavailable;
        rl.mac = try n.set.mac_of(n.set.devices[r]);
        rl.mr_arena = api.ibv_reg_mr(rl.pd.?, n.arena.ptr, n.arena.len, vabi.access_local_write | vabi.access_remote_write) orelse return error.VerbsFailed;
        rl.abort_cq = api.ibv_create_cq(ctx, @intCast(roci.world * 4), null, null, 0) orelse return error.VerbsFailed;
        for (0..roci.world) |p| {
            if (p == n.me) continue;
            rl.qps[p] = try n.createQp(r);
            var init: vabi.QpInitAttr = .{ .send_cq = rl.abort_cq.?, .recv_cq = rl.abort_cq.?, .cap = .{ .max_send_wr = 4, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 16 }, .qp_type = vabi.qpt_rc };
            rl.abort_qps[p] = api.ibv_create_qp(rl.pd.?, &init) orelse return error.VerbsFailed;
        }
    }

    /// One Core Direct master queue pair of rail `r`: its send queue and doorbell record in a umem of its own. A create the card refuses
    /// (a send queue past what it supports) is retried with a send queue half the size, down to 512 entries.
    fn createQp(n: *CdNic, r: usize) !Qp {
        const rl = &n.rail[r];
        var log = n.set.log_sq_size;
        while (true) : (log -= 1) {
            const depth: usize = @as(usize, 1) << @intCast(log);
            const dbr_off = depth * mx.wqebb;
            const total = std.mem.alignForward(usize, dbr_off + 64, roci.page);
            const mem = try n.gpa.alignedAlloc(u8, .fromByteUnits(roci.page), total);
            @memset(mem, 0);
            const umem = n.mlx.api.mlx5dv_devx_umem_reg(rl.ctx.?, mem.ptr, total, vabi.access_local_write) orelse {
                n.gpa.free(mem);
                return error.QpCreateFailed;
            };
            var in: [mx.qp_msg_bytes]u8 = undefined;
            var out: [mx.small_msg_bytes]u8 = @splat(0);
            mx.createQpIn(&in, .{ .pdn = rl.pdn, .uar_page = rl.uar.?.page_id, .cqn = rl.cqn, .log_sq_size = log, .umem_id = umem.umem_id, .dbr_off = dbr_off });
            if (n.mlx.api.mlx5dv_devx_obj_create(rl.ctx.?, &in, in.len, &out, out.len)) |obj| {
                return .{ .obj = obj, .qpn = @intCast(mx.get(&out, mx.msg.create_out_qpn)), .umem = umem, .mem = mem, .depth = @intCast(depth), .dbrec = @ptrCast(@alignCast(mem[dbr_off..].ptr)) };
            }
            const err = std.c._errno().*;
            _ = n.mlx.api.mlx5dv_devx_umem_dereg(umem);
            n.gpa.free(mem);
            if (log <= 9) {
                std.log.err("ROCI wait: CREATE_QP (cd_master) on {s} refused at log_sq_size {d}: errno {d} syndrome 0x{x}", .{ n.set.devices[r], log, err, mx.get(&out, mx.msg.out_syndrome) });
                return error.QpCreateFailed;
            }
            std.log.warn("ROCI wait: CREATE_QP on {s} refused at log_sq_size {d} (errno {d}, syndrome 0x{x}): trying {d}", .{ n.set.devices[r], log, err, mx.get(&out, mx.msg.out_syndrome), log - 1 });
        }
    }

    pub fn info(n: *CdNic) !Info {
        for (0..roci.rails) |r| {
            const rl = &n.rail[r];
            i.gid[r] = (try n.v.gid(rl.ctx.?, 1, n.set.gid_index)).raw;
            i.rkey_arena[r] = rl.mr_arena.?.rkey;
            i.rkey_ctl[r] = rl.mr_ctl.?.rkey;
            @memcpy(i.mac[r][0..6], &rl.mac);
            for (rl.qps, 0..) |q, p| if (q) |x| {
                i.qpn[r][p] = x.qpn;
            };
            for (rl.abort_qps, 0..) |q, p| if (q) |x| {
                i.abort_qpn[r][p] = x.qp_num;
                i.psn[r][p] = (x.qp_num *% 2654435761) & 0xffffff;
            };
        }
        return i;
    }

    /// RESET to INIT to RTR to RTS on every queue pair against the peer's queue pair for this node: the data pairs by DEVX command, the
    /// abort pairs by verbs.
    pub fn connect(n: *CdNic, all: *const [roci.world]Info) !void {
        const api = n.v.api;
        const m = vabi.mask;
        const mine = try n.info();
        try rv.checkPeers(all, mine);
        n.peers = all.*;
        for (0..roci.rails) |r| for (0..roci.world) |p| {
            const rl = &n.rail[r];
            const q = rl.qps[p] orelse continue;
            const theirs = all[p];
            var in: [mx.qp_msg_bytes]u8 = undefined;
            var out: [mx.small_msg_bytes]u8 = undefined;
            mx.rst2initIn(&in, q.qpn);
            try n.command(r, q.obj.?, &in, &out, "RST2INIT");
            const remote: mx.Remote = .{ .qpn = theirs.qpn[r][n.me], .gid = theirs.gid[r], .mac = theirs.mac[r][0..6].* };
            mx.init2rtrIn(&in, q.qpn, remote, n.set.gid_index, udpSport(q.qpn, r, p), rl.mtu);
            try n.command(r, q.obj.?, &in, &out, "INIT2RTR");
            mx.rtr2rtsIn(&in, q.qpn);
            try n.command(r, q.obj.?, &in, &out, "RTR2RTS");
            // the abort pair: the verbs path's transitions
            const aq = rl.abort_qps[p].?;
            var a: vabi.QpAttr = .{ .qp_state = vabi.qps_init, .port_num = 1, .qp_access_flags = @intCast(vabi.access_remote_write) };
            if (api.ibv_modify_qp(aq, &a, m.state | m.pkey_index | m.port | m.access_flags) != 0) return error.VerbsFailed;
            a = .{ .qp_state = vabi.qps_rtr, .path_mtu = rl.mtu, .dest_qp_num = theirs.abort_qpn[r][n.me], .rq_psn = theirs.psn[r][n.me], .max_dest_rd_atomic = 0 };
            a.ah_attr = .{ .grh = .{ .dgid = .{ .raw = theirs.gid[r] }, .sgid_index = n.set.gid_index, .hop_limit = 64 }, .is_global = 1, .port_num = 1 };
            a.min_rnr_timer = 12;
            if (api.ibv_modify_qp(aq, &a, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
            a = .{ .qp_state = vabi.qps_rts, .sq_psn = mine.psn[r][p], .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .max_rd_atomic = 0 };
            if (api.ibv_modify_qp(aq, &a, m.state | m.sq_psn | m.timeout | m.retry_cnt | m.rnr_retry | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
        };
    }

    fn command(n: *CdNic, r: usize, obj: *mx.DevxObj, in: []const u8, out: []u8, what: []const u8) !void {
        @memset(out, 0);
        const rc = n.mlx.api.mlx5dv_devx_obj_modify(obj, in.ptr, in.len, out.ptr, out.len);
        if (rc != 0 or mx.get(out, mx.msg.out_status) != 0) {
            std.log.err("ROCI wait: {s} on {s} failed: return {d}, status {d}, syndrome 0x{x}", .{ what, n.set.devices[r], rc, mx.get(out, mx.msg.out_status), mx.get(out, mx.msg.out_syndrome) });
            return error.CommandFailed;
        }
    }

    pub fn nic(n: *CdNic) roci.Nic {
        return .{ .ctx = n, .post_fn = postFn, .fail_fn = failFn, .reap_fn = reapFn, .prepost_fn = prepostFn };
    }

    /// The control block's abort word: a failure's first store, which ends a wait for room.
    fn abortWord(n: *const CdNic) u64 {
        return @atomicLoad(u64, @as(*const u64, @ptrCast(@alignCast(n.ctl[@offsetOf(roci.Control, "abort")..].ptr))), .acquire);
    }

    // -- the lock ------------------------------------------------------------------------------------------------------

    fn acquire(n: *CdNic) void {
        while (n.lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn tryAcquire(n: *CdNic) bool {
        return n.lock.cmpxchgStrong(0, 1, .acquire, .monotonic) == null;
    }

    fn release(n: *CdNic) void {
        n.lock.store(0, .release);
    }

    // -- the post ------------------------------------------------------------------------------------------------------

    /// This transport does not post from a proxy: a Nic used that way is a mistake in the caller.
    fn postFn(_: *anyopaque, _: u8, _: u8, _: roci.Class, _: *const roci.Chain, _: u64) roci.PostError!void {
        return error.PostFailed;
    }

    /// signalled completion frees some, as the verbs post waits.
    fn prepostFn(ctx: *anyopaque, rail: u8, peer: u8, class: roci.Class, chain: *const roci.Chain, word: u64, wait_for: u64) roci.PostError!void {
        const n: *CdNic = @ptrCast(@alignCast(ctx));
        const rl = &n.rail[rail];
        if (@atomicLoad(bool, &rl.fenced, .acquire)) return error.PostFailed;
        const q: *Qp = if (rl.qps[peer]) |*x| x else return error.PostFailed;
        const mr_ctl = rl.mr_ctl orelse return error.PostFailed;
        const mr_arena = rl.mr_arena.?;
        const theirs = &n.peers[peer];
        if (q.pi -% @atomicLoad(u32, &q.retired, .acquire) +% need +% margin > q.depth) {
            const t0 = nowNs();
            var spins: u32 = 0;
            while (q.pi -% @atomicLoad(u32, &q.retired, .acquire) +% need +% margin > q.depth) : (spins +%= 1) {
                if (@atomicLoad(bool, &rl.fenced, .acquire) or n.abortWord() != 0) return error.PostFailed;
                n.acquire();
                const bad = n.pollRail(rail);
                n.release();
                try bad;
                if (spins % 1024 == 1023 and nowNs() - t0 > n.set.sq_wait_ms * std.time.ns_per_ms) return error.PostFailed;
                std.atomic.spinLoopHint();
            }
        }
        var pi = q.pi;
        const arena_va = @intFromPtr(n.arena.ptr);
        if (wait_for != 0) {
            mx.waitWqe(q.slot(pi), @truncate(pi), q.qpn, wait_cond, wait_invert, mr_ctl.lkey, @intFromPtr(n.ctl.ptr) + @offsetOf(roci.Control, "doorbell"), waitValue(wait_for), wait_mask);
            pi +%= 1;
        }
        for (chain.slice()) |p| {
            mx.writeWqe(q.slot(pi), @truncate(pi), q.qpn, false, theirs.arena_base + p.dst, theirs.rkey_arena[rail], mr_arena.lkey, arena_va + p.src, @intCast(p.len));
            pi +%= 1;
        }
        q.chains +%= 1;
        const signal = q.chains % n.set.signal_every == 0;
        const cell = @offsetOf(roci.Control, "recv") + ((@as(usize, @backingInt(class)) * roci.world + n.me) * roci.rails + rail) * @sizeOf(roci.Word);
        q.pi = pi;
        ring(rl, q);
    }

    /// The doorbell: the entries are in memory before the record names them, the record before the register (m19c_wait.cu:229-235): the
    /// producer index into the record (MLX5_SND_DBR is dword 1), then the first 8 bytes of the last entry to the UAR's register.
    fn ring(rl: *Rail, q: *Qp) void {
        mx.wmbDev();
        q.dbrec[1] = std.mem.nativeToBig(u32, q.pi & 0xffff);
        mx.wmbDev();
        const last = q.slot(q.pi -% 1);
        const reg: *volatile u64 = @ptrCast(@alignCast(rl.uar.?.reg_addr.?));
        reg.* = std.mem.bytesToValue(u64, last[0..8]);
        mx.dsbSt();
    }

    // -- the completions -----------------------------------------------------------------------------------------------

    /// Reads rail `r`'s completion queue (the lock held): a signalled completion retires the entries up to and including its own; an error
    /// completion fails the transport.
    fn pollRail(n: *CdNic, r: usize) roci.PostError!void {
        const rl = &n.rail[r];
        while (rl.view.pollOne()) |c| {
            if (c.isError()) {
                if (log_errors) std.log.err("ROCI wait: error completion on {s}: opcode 0x{x}, syndrome 0x{x}, vendor 0x{x}, queue pair {d}, entry {d}", .{ n.set.devices[r], c.opcode, c.syndrome, c.vendor_err, c.qpn, c.wqe_counter });
                return error.QpError;
            }
            for (&rl.qps) |*qo| if (qo.*) |*q| if (q.qpn == c.qpn) {
                const step: u32 = (@as(u32, c.wqe_counter) +% 1 -% (q.retired & 0xffff)) & 0xffff;
                if (step <= q.pi -% q.retired) @atomicStore(u32, &q.retired, q.retired +% step, .release);
            };
        }
    }

    /// The janitor's look: the lock is the engine thread's while it makes room, so a look that finds it taken is skipped.
    fn reapFn(ctx: *anyopaque, rail: u8) roci.PostError!void {
        const n: *CdNic = @ptrCast(@alignCast(ctx));
        if (@atomicLoad(bool, &n.rail[rail].fenced, .acquire)) return;
        if (!n.tryAcquire()) return;
        defer n.release();
        try n.pollRail(rail);
    }

    // -- the failure and the close -------------------------------------------------------------------------------------

    /// The failure path, in the order that matters: the notices go to the peers over the abort pairs (a data pair behind an armed WAIT would
    /// never send one), then every data pair goes to ERR (an armed WAIT is flushed, not released) and is destroyed, then the control block's
    /// registration is released; only then does the transport store the abort value, which would satisfy every armed WAIT. The engine thread is
    /// out of its post (Transport.awaitPostingOut) and the janitor skips its looks while the fence holds the lock.
    fn failFn(ctx: *anyopaque) void {
        const n: *CdNic = @ptrCast(@alignCast(ctx));
        for (&n.rail) |*rl| @atomicStore(bool, &rl.fenced, true, .release);
        n.acquire();
        defer n.release();
        n.notify();
        n.fence();
    }

    /// The abort word to every peer on every rail, best effort, completions read for a bounded time.
    fn notify(n: *CdNic) void {
        var one: u64 = 1;
        var sent: [roci.rails]usize = @splat(0);
        for (0..roci.rails) |r| for (0..roci.world) |p| {
            const rl = &n.rail[r];
            const aq = rl.abort_qps[p] orelse continue;
            var sge: vabi.Sge = .{ .addr = @intFromPtr(&one), .length = 8, .lkey = 0 };
            var wr: vabi.SendWr = .{ .sg_list = @ptrCast(&sge), .num_sge = 1, .opcode = vabi.wr_rdma_write, .send_flags = vabi.send_inline | vabi.send_signaled, .remote_addr = n.peers[p].ctl_base + @offsetOf(roci.Control, "abort"), .rkey = n.peers[p].rkey_ctl[r] };
            var bad: ?*vabi.SendWr = null;
            if (rl.ctx.?.ops.post_send(aq, &wr, &bad) == 0) sent[r] += 1;
        };
        const t0 = nowNs();
        for (0..roci.rails) |r| {
            const rl = &n.rail[r];
            var got: usize = 0;
            var wc: [8]vabi.Wc = undefined;
            while (got < sent[r] and nowNs() - t0 < n.set.abort_wait_ms * std.time.ns_per_ms) {
                const k = rl.ctx.?.ops.poll_cq(rl.abort_cq.?, 8, &wc);
                if (k > 0) got += @intCast(k) else sleepNs(20_000);
            }
        }
    }

    /// Takes the queue pairs and then the control block away from the card (the lock held): every data pair to ERR, its flush read, the
    /// pair destroyed; the abort pairs destroyed; then the control block's registration released (a registration under a live queue pair is
    /// the card's to read). A deregistration that is refused leaves the card a settle time.
    fn fence(n: *CdNic) void {
        var held = false;
        for (0..roci.rails) |r| {
            const rl = &n.rail[r];
            @atomicStore(bool, &rl.fenced, true, .release);
            for (&rl.qps) |*qo| if (qo.*) |*q| if (q.obj) |obj| {
                var in: [mx.small_msg_bytes]u8 = undefined;
                var out: [mx.small_msg_bytes]u8 = @splat(0);
                mx.qp2errIn(&in, q.qpn);
                _ = n.mlx.api.mlx5dv_devx_obj_modify(obj, &in, in.len, &out, out.len);
            };
            for (&rl.qps) |*qo| if (qo.*) |*q| if (q.obj) |obj| {
                _ = n.mlx.api.mlx5dv_devx_obj_destroy(obj);
                q.obj = null;
                if (q.umem) |u| _ = n.mlx.api.mlx5dv_devx_umem_dereg(u);
                q.umem = null;
            };
            for (&rl.abort_qps) |*aq| if (aq.*) |q| {
                _ = n.v.api.ibv_destroy_qp(q);
                aq.* = null;
            };
            if (rl.mr_ctl) |m| {
                if (n.v.api.ibv_dereg_mr(m) == 0) rl.mr_ctl = null else held = true;
            }
        }
        if (held) {
            std.log.err("ROCI wait: the control block's registration could not be released at a failure: the card is given {d} ms to finish what it has begun", .{n.set.settle_ms});
            sleepNs(@as(u64, n.set.settle_ms) * std.time.ns_per_ms);
        }
    }

    /// The flush completions of a queue pair taken to ERR, read and dropped until the queue is quiet (at most 5 ms): they are not news.
    fn drain(n: *CdNic, r: usize) void {
        const rl = &n.rail[r];
        const t0 = nowNs();
        var quiet: u32 = 0;
        while (quiet < 3 and nowNs() - t0 < 5 * std.time.ns_per_ms) {
            if (rl.view.pollOne() != null) {
                quiet = 0;
            } else {
                quiet += 1;
                sleepNs(100_000);
            }
        }
    }

    /// Everything released, the queue pairs first (taking them from the card if a failure did not): the control block outlives them.
    fn teardown(n: *CdNic) void {
        n.acquire();
        n.fence();
        n.release();
        const api = n.v.api;
        for (&n.rail) |*rl| {
            for (&rl.qps) |*qo| if (qo.*) |*q| {
                n.gpa.free(q.mem);
                qo.* = null;
            };
            if (rl.abort_cq) |c| _ = api.ibv_destroy_cq(c);
            if (rl.mr_arena) |m| _ = api.ibv_dereg_mr(m);
            if (rl.cq) |c| _ = api.ibv_destroy_cq(c);
            if (rl.uar) |u| n.mlx.api.mlx5dv_devx_free_uar(u);
            if (rl.pd) |p| _ = api.ibv_dealloc_pd(p);
            if (rl.ctx) |c| _ = api.ibv_close_device(c);
            rl.* = .{};
        }
    }

    pub fn close(n: *CdNic) void {
        n.teardown();
        n.gpa.destroy(n);
    }
};

/// Opens `name` with DEVX enabled: the context every object of a queue pair made through DEVX belongs to.
fn openDevx(v: *const Verbs, m: *const mx.Mlx5, name: []const u8) !mx.Ctx {
    var count: c_int = 0;
    const list = v.api.ibv_get_device_list(&count) orelse return error.NoDeviceList;
    defer v.api.ibv_free_device_list(list);
    var i: usize = 0;
    while (list[i]) |dev| : (i += 1) {
        const got = v.api.ibv_get_device_name(dev) orelse continue;
        if (!std.mem.eql(u8, std.mem.span(got), name)) continue;
        var attr: mx.ContextAttr = .{ .flags = mx.context_flag_devx };
        return m.api.mlx5dv_open_device(dev, &attr) orelse error.DevxOpenFailed;
    }
    return error.NoSuchDevice;
}

/// The UDP source port of RoCE v2's flow entropy for a queue pair: from its number, the rail and the peer, in the dynamic port range.
fn udpSport(qpn: u32, rail: usize, peer: usize) u16 {
    const h = (qpn *% 2654435761) ^ @as(u32, @intCast(rail * 7 + peer));
    return @intCast(49152 + h % 16000);
}

/// A MAC from sysfs's text ("00:11:22:33:44:55\n").
pub fn parseMac(text: []const u8) ?[6]u8 {
    var out: [6]u8 = undefined;
    var it = std.mem.splitScalar(u8, std.mem.trim(u8, text, " \t\r\n"), ':');
    for (&out) |*b| b.* = std.fmt.parseInt(u8, it.next() orelse return null, 16) catch return null;
    if (it.next() != null) return null;
    return out;
}

/// The MAC of the network device behind an RDMA device: sysfs's `/sys/class/infiniband/<dev>/device/net/<netdev>/address`, the way
/// m19c_wait.cu's netdev_mac finds it.
pub fn netdevMac(device: []const u8) error{NoMac}![6]u8 {
    var path: [256]u8 = undefined;
    const dir = std.fmt.bufPrintSentinel(&path, "/sys/class/infiniband/{s}/device/net", .{device}, 0) catch return error.NoMac;
    const d = std.c.opendir(dir) orelse return error.NoMac;
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name[0] == '.') continue;
        var fp: [512]u8 = undefined;
        const file = std.fmt.bufPrintSentinel(&fp, "{s}/{s}/address", .{ dir, name }, 0) catch continue;
        const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) continue;
        defer _ = std.c.close(fd);
        var buf: [64]u8 = undefined;
        const got = std.c.read(fd, &buf, buf.len);
        if (got <= 0) continue;
        return parseMac(buf[0..@intCast(got)]) orelse continue;
    }
    return error.NoMac;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn sleepNs(ns: u64) void {
    const ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}
