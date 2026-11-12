//! Host tests of Rocinante on queue pairs that wait in the card (rocinante_cd.zig, TF_DS_ROCI_WAIT=1): four nodes in one process over a
//! model of the card. The model reads what the backend really writes: the DEVX commands (decoded with the field table), the send queue's
//! entries as bytes (control segment, WAIT, WRITEs, inline word), the doorbell record, and it writes completion entries the backend parses by
//! hand. It runs the entries of each queue pair in order, a WAIT holding its queue pair's entries until the doorbell word satisfies it, and
//! it checks every field the real card would (index, queue pair number, units, keys and the memory they cover). The release rule of a WAIT
//! condition is the model's reading of NVIDIA's DPDK driver, not a measurement: the Spark's gate decides it.

const std = @import("std");
const builtin = @import("builtin");
const fabric = @import("fabric");
const vabi = fabric.verbs_abi;
const Verbs = fabric.verbs.Verbs;
const roci = @import("rocinante.zig");
const rv = @import("rocinante_verbs.zig");
const cd = @import("rocinante_cd.zig");
const mx = @import("mlx5_devx.zig");

const gpa = std.testing.allocator;
const world = roci.world;
const rails = roci.rails;

fn sleepNs(ns: u64) void {
    if (ns == 0) return;
    const ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

// -- the model of the card -------------------------------------------------------------------------------------------

/// What a WAIT does with its operation word, value, mask and the memory word, as measured (M19g): both are read as BIG-ENDIAN numbers (the
/// bytes as stored), BIGGER fires iff the value is above the word, SMALLER iff below, the cyclic ones compare the difference as signed, 0x10 inverts.
fn modelReleased(op: u32, value: u64, mask: u64, mem: u64) bool {
    const inv = op & 0x10 != 0;
    const v = std.mem.bigToNative(u64, value & mask);
    const m = std.mem.bigToNative(u64, mem & mask);
    const base = switch (op & 0xf) {
        0 => true,
        1 => v == m,
        2 => v > m,
        3 => v < m,
        4 => @as(i64, @bitCast(v -% m)) > 0,
        5 => @as(i64, @bitCast(v -% m)) < 0,
        else => false,
    };
    return base != inv;
}

const FMr = struct { addr: usize, len: usize, access: c_int, node: u8, rail: u8, key: u32, live: bool, ctl: bool, mr: vabi.Mr };
const FUm = struct { addr: usize, len: usize, id: u32, um: mx.DevxUmem };
const FCq = struct { ibv: vabi.Cq, buf: []align(64) u8, cnt: u32, pi: u32, dbrec: [2]u32, cqn: u32, node: u8, rail: u8, wcs: [16]vabi.Wc, n_wc: usize, live: bool };
const FCtx = struct { ctx: vabi.Context, pd: vabi.Pd, node: u8, rail: u8, pdn: u32 };
const FQp = struct {
    obj: u8, // (the handle's address is this struct's)
    used: bool,
    dead: bool,
    state: u8,
    node: u8,
    rail: u8,
    qpn: u32,
    depth: u32,
    sq: usize,
    dbr: usize,
    cq: *FCq,
    cd_master: bool,
    rq_type: u32,
    remote_qpn: u32,
    ci: u16,
    flushed: bool,
    /// What the transitions said about atomics (the qpc's rae, atomic_mode, log_rra_max at RTR, log_sra_max at RTS) and the opt-param masks.
    rae: bool = false,
    atomic_mode: u8 = 0,
    log_rra: u8 = 0,
    log_sra: u8 = 0,
    opt_masks: [3]u32 = @splat(0),
};
const FVqp = struct { qp: vabi.Qp, used: bool, node: u8, rail: u8, remote: u32, cq: *FCq, state: c_int };

const Fx = struct {
    lock: std.atomic.Value(u32) = .init(0),
    cur_node: u8 = 0,
    stop: std.atomic.Value(bool) = .init(false),
    card: ?std.Thread = null,
    devs: [rails]vabi.Device = undefined,
    dev_list: [rails:null]?*vabi.Device = undefined,
    ctxs: [world * rails]FCtx = undefined,
    mrs: [256]FMr = undefined,
    n_mr: usize = 0,
    ums: [256]FUm = undefined,
    n_um: usize = 0,
    qps: [256]FQp = undefined,
    n_qp: usize = 0,
    vqps: [256]FVqp = undefined,
    n_vqp: usize = 0,
    cqs: [64]FCq = undefined,
    n_cq: usize = 0,
    ctl_addr: [world]usize = @splat(0),
    uar_cell: [rails]u64 = @splat(0),
    // knobs
    err_delay_ns: u64 = 0,
    max_log_sq: u8 = 15,
    bf_uar_fails: bool = false,
    drop_cqes: bool = false,
    atomic_cap: c_int = vabi.atomic_hca,
    rd_atom: c_int = 16,
    // what happened
    atomics: u64 = 0,
    writes: u64 = 0,
    words: u64 = 0,
    waits_released: u64 = 0,
    violations: u32 = 0,
    err_cqes: u32 = 0,
    created_cd_master: u32 = 0,
    created_other: u32 = 0,
    last_uar_type: u32 = 99,
    events: [512]u8 = undefined,
    n_ev: usize = 0,
    transitions: [256]u16 = undefined,
    n_tr: usize = 0,
    next_qpn: u32 = 0x100,
    next_umem: u32 = 1,
    next_vqpn: u32 = 0x9000,
    n_open_ctx: usize = 0,
    n_registered_under_devx: usize = 0,
    unregistered_use: u32 = 0,

    fn lockIt(f: *Fx) void {
        while (f.lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
    fn unlockIt(f: *Fx) void {
        f.lock.store(0, .release);
    }
    fn ev(f: *Fx, c: u8) void {
        if (f.n_ev < f.events.len) {
            f.events[f.n_ev] = c;
            f.n_ev += 1;
        }
    }
};

var fx: Fx = .{};

fn fxReset() void {
    fx = .{};
    fx.dev_list = .{ &fx.devs[0], &fx.devs[1] };
    for (&fx.devs, 0..) |*d, i| {
        d.* = std.mem.zeroes(vabi.Device);
        const nm = if (i == 0) "mlx5_0" else "mlx5_2";
        @memcpy(d.name[0..nm.len], nm);
    }
    for (&fx.ctxs, 0..) |*c, i| {
        c.* = std.mem.zeroes(FCtx);
        c.node = @intCast(i / rails);
        c.rail = @intCast(i % rails);
        c.pdn = @intCast(0x40 + i);
        c.ctx.device = &fx.devs[i % rails];
        c.ctx.ops.post_send = vPostSend;
        c.ctx.ops.poll_cq = vPollCq;
        c.pd.context = &c.ctx;
    }
}

fn be32(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .big);
}
fn be64(b: []const u8) u64 {
    return std.mem.readInt(u64, b[0..8], .big);
}

fn ctxOf(c: *vabi.Context) *FCtx {
    return @fieldParentPtr("ctx", c);
}

fn mrOfKey(key: u32) ?*FMr {
    for (fx.mrs[0..fx.n_mr]) |*m| if (m.live and m.key == key) return m;
    return null;
}

fn covers(m: *const FMr, addr: usize, len: usize) bool {
    return addr >= m.addr and addr + len <= m.addr + m.len;
}

fn qpByQpn(qpn: u32) ?*FQp {
    for (fx.qps[0..fx.n_qp]) |*q| if (q.used and !q.dead and q.qpn == qpn) return q;
    return null;
}

/// A completion entry into `cq`: the body, then the opcode and owner byte last, as the card does.
fn pushCqe(cq: *FCq, qpn: u32, idx: u16, opcode: u8, syndrome: u8) void {
    const consumed = std.mem.bigToNative(u32, @atomicLoad(u32, &cq.dbrec[0], .acquire)) & 0xff_ffff;
    if (cq.pi -% consumed >= cq.cnt) {
        fx.violations += 1; // an overflowed completion queue
        return;
    }
    const e = cq.buf[(cq.pi & (cq.cnt - 1)) * 64 ..][0..64];
    @memset(e[0..63], 0);
    std.mem.writeInt(u32, e[56..60], qpn, .big);
    std.mem.writeInt(u16, e[60..62], idx, .big);
    e[55] = syndrome;
    const owner: u8 = @intCast((cq.pi / cq.cnt) & 1);
    @atomicStore(u8, &e[63], (opcode << 4) | owner, .release);
    cq.pi +%= 1;
    if (opcode == mx.cqe_req_err) fx.err_cqes += 1;
}

fn failQp(q: *FQp, idx: u16, syndrome: u8) void {
    q.state = 6;
    pushCqe(q.cq, q.qpn, idx, mx.cqe_req_err, syndrome);
}

fn flushQp(q: *FQp) void {
    if (q.flushed) return;
    q.flushed = true;
    const pi_vis: u16 = @truncate(std.mem.bigToNative(u32, @atomicLoad(u32, @as(*u32, @ptrFromInt(q.dbr + 4)), .acquire)));
    var n: u32 = 0;
    while (q.ci != pi_vis and n < 32) : (n += 1) {
        pushCqe(q.cq, q.qpn, q.ci, mx.cqe_req_err, 5);
        q.ci +%= 1;
    }
    q.ci = pi_vis;
}

fn doWrite(q: *FQp, w: *const [64]u8) bool {
    const raddr: usize = @intCast(be64(w[16..24]));
    const rkey = be32(w[24..28]);
    const bc = be32(w[32..36]);
    var len: usize = undefined;
    var src: usize = undefined;
    if (bc & mx.inline_seg != 0) {
        len = bc & 0x7fff_ffff;
        if (len != 8) {
            fx.violations += 1;
            failQp(q, q.ci, 1);
            return false;
        }
        src = @intFromPtr(&w[36]);
    } else {
        len = bc;
        const lkey = be32(w[36..40]);
        src = @intCast(be64(w[40..48]));
        const lm = mrOfKey(lkey) orelse {
            fx.violations += 1;
            failQp(q, q.ci, 4); // local protection
            return false;
        };
        if (lm.node != q.node or lm.rail != q.rail or !covers(lm, src, len)) {
            fx.violations += 1;
            failQp(q, q.ci, 4);
            return false;
        }
    }
    const dest = qpByQpn(q.remote_qpn);
    const live = if (dest) |d| (d.state == 2 or d.state == 3) and d.rail == q.rail else false;
    if (!live) {
        failQp(q, q.ci, 0x15); // retry exceeded: the peer's queue pair is gone
        return false;
    }
    const rm = mrOfKey(rkey) orelse {
        failQp(q, q.ci, 0x13); // remote access: no such registration (a peer whose control block is released)
        return false;
    };
    if (rm.node != dest.?.node or rm.rail != q.rail or rm.access & vabi.access_remote_write == 0 or !covers(rm, raddr, len)) {
        fx.violations += 1;
        failQp(q, q.ci, 0x13);
        return false;
    }
    if (rm.ctl and len == 8) {
        // an arrival word: the cell of (class, source node, rail) is written by that source's queue pair of that rail
        const cell = (raddr - fx.ctl_addr[rm.node] - @offsetOf(roci.Control, "recv")) / @sizeOf(roci.Word);
        if (cell % rails != q.rail or (cell / rails) % world != q.node) fx.violations += 1;
    }
    const dst: [*]u8 = @ptrFromInt(raddr);
    const from: [*]const u8 = @ptrFromInt(src);
    if (len == 8 and raddr % 8 == 0) {
        const v = std.mem.readInt(u64, from[0..8], builtin.cpu.arch.endian());
        @atomicStore(u64, @as(*u64, @ptrCast(@alignCast(dst))), v, .release);
        fx.words += 1;
    } else @memcpy(dst[0..len], from[0..len]);
    fx.writes += 1;
    return true;
}

/// The ATOMIC fetch-and-add entry as the card runs it: the requester must have been given depth (log_sra_max is 4 bits of depth in the model:
/// any value), the responder's queue pair must have rae set with atomic_mode IB_SPEC (1) and its registration ATOMIC access, the remote address
/// 8-byte aligned, the data segment an 8-byte local-write landing. The add is the WQE's big-endian addend onto the host-endian word (the
/// card's host-endianness atomics mode), the old value goes to the landing.
fn doAtomic(q: *FQp, w: *const [64]u8) bool {
    const raddr: usize = @intCast(be64(w[16..24]));
    const rkey = be32(w[24..28]);
    const add = be64(w[32..40]);
    const bc = be32(w[48..52]);
    const lkey = be32(w[52..56]);
    const laddr: usize = @intCast(be64(w[56..64]));
    if (bc != 8 or raddr % 8 != 0 or laddr % 8 != 0 or be32(w[28..32]) != 0 or be64(w[40..48]) != 0) {
        fx.violations += 1;
        failQp(q, q.ci, 2);
        return false;
    }
    const lm = mrOfKey(lkey) orelse {
        fx.violations += 1;
        failQp(q, q.ci, 4);
        return false;
    };
    if (lm.node != q.node or lm.rail != q.rail or lm.access & vabi.access_local_write == 0 or !covers(lm, laddr, 8)) {
        fx.violations += 1;
        failQp(q, q.ci, 4);
        return false;
    }
    const dest = qpByQpn(q.remote_qpn);
    const live = if (dest) |d| (d.state == 2 or d.state == 3) and d.rail == q.rail else false;
    if (!live) {
        failQp(q, q.ci, 0x15);
        return false;
    }
    const rm = mrOfKey(rkey) orelse {
        failQp(q, q.ci, 0x13);
        return false;
    };
    if (!dest.?.rae or dest.?.atomic_mode != 1 or rm.node != dest.?.node or rm.rail != q.rail or rm.access & vabi.access_remote_atomic == 0 or !covers(rm, raddr, 8)) {
        fx.violations += 1;
        failQp(q, q.ci, 0x13); // remote access: the responder's queue pair or registration does not grant atomics
        return false;
    }
    if (rm.ctl) {
        const cell = (raddr - fx.ctl_addr[rm.node] - @offsetOf(roci.Control, "recv")) / @sizeOf(roci.Word);
        if (cell % rails != q.rail or (cell / rails) % world != q.node) fx.violations += 1;
    }
    const old = @atomicRmw(u64, @as(*u64, @ptrFromInt(raddr)), .Add, add, .acq_rel);
    @atomicStore(u64, @as(*u64, @ptrFromInt(laddr)), old, .release);
    fx.atomics += 1;
    return true;
}

fn stepQp(q: *FQp) void {
    if (!q.used or q.dead) return;
    if (q.state == 6) {
        flushQp(q);
        return;
    }
    if (q.state != 3) return;
    const pi_vis: u16 = @truncate(std.mem.bigToNative(u32, @atomicLoad(u32, @as(*u32, @ptrFromInt(q.dbr + 4)), .acquire)));
    while (q.ci != pi_vis) {
        const w: *const [64]u8 = @ptrFromInt(q.sq + (@as(usize, q.ci) & (q.depth - 1)) * 64);
        const w0 = be32(w[0..4]);
        const w1 = be32(w[4..8]);
        const opcode: u8 = @truncate(w0);
        const want_ds: u32 = if (opcode == mx.opcode_atomic_fa) mx.atomic_ds_units else mx.ds_units;
        if (((w0 >> 8) & 0xffff) != q.ci or (w1 >> 8) != q.qpn or (w1 & 0x3f) != want_ds) {
            fx.violations += 1;
            failQp(q, q.ci, 2);
            return;
        }
        switch (opcode) {
            mx.opcode_wait => {
                const va: usize = @intCast((@as(u64, be32(w[24..28])) << 32) | (be32(w[28..32]) & ~@as(u32, 7)));
                const lm = mrOfKey(be32(w[20..24]));
                if ((w0 >> 24) != mx.opmod_wait_data or lm == null or lm.?.node != q.node or lm.?.rail != q.rail or !covers(lm.?, va, 8)) {
                    fx.violations += 1;
                    failQp(q, q.ci, 4);
                    return;
                }
                const mem = @atomicLoad(u64, @as(*const u64, @ptrFromInt(va)), .acquire);
                const value = std.mem.readInt(u64, w[32..40], builtin.cpu.arch.endian());
                const mask = std.mem.readInt(u64, w[40..48], builtin.cpu.arch.endian());
                if (!modelReleased(be32(w[16..20]), value, mask, mem)) return;
                fx.waits_released += 1;
            },
            mx.opcode_rdma_write => if (!doWrite(q, w)) return,
            mx.opcode_atomic_fa => if (!doAtomic(q, w)) return,
            else => {
                fx.violations += 1;
                failQp(q, q.ci, 2);
                return;
            },
        }
        if (w[11] & mx.ctrl_cq_update != 0 and !fx.drop_cqes) pushCqe(q.cq, q.qpn, q.ci, 0, 0);
        q.ci +%= 1;
    }
}

fn cardMain() void {
    while (!fx.stop.load(.acquire)) {
        fx.lockIt();
        for (fx.qps[0..fx.n_qp]) |*q| stepQp(q);
        fx.unlockIt();
        sleepNs(1_000);
    }
}

// -- the fake verbs and the fake libmlx5 -----------------------------------------------------------------------------

fn vGetList(num: *c_int) callconv(.c) ?[*:null]?*vabi.Device {
    num.* = rails;
    return &fx.dev_list;
}
fn vFreeList(_: [*:null]?*vabi.Device) callconv(.c) void {}
fn vDevName(d: *vabi.Device) callconv(.c) ?[*:0]const u8 {
    return @ptrCast(&d.name);
}
fn vCloseDev(_: *vabi.Context) callconv(.c) c_int {
    return 0;
}
fn vQueryPort(_: *vabi.Context, _: u8, a: *vabi.PortAttr) callconv(.c) c_int {
    a.state = vabi.port_active;
    a.active_mtu = 5;
    a.gid_tbl_len = 4;
    return 0;
}
fn vQueryGid(c: *vabi.Context, _: u8, _: c_int, g: *vabi.Gid) callconv(.c) c_int {
    const x = ctxOf(c);
    g.raw = @splat(0);
    g.raw[10] = 0xff;
    g.raw[11] = 0xff;
    g.raw[12] = 10;
    g.raw[13] = 255;
    g.raw[14] = x.rail;
    g.raw[15] = 33 + x.node;
    return 0;
}
fn vQueryDevice(_: *vabi.Context, a: *vabi.DeviceAttr) callconv(.c) c_int {
    a.atomic_cap = fx.atomic_cap;
    a.max_qp_rd_atom = fx.rd_atom;
    a.max_qp_init_rd_atom = fx.rd_atom;
    return 0;
}
fn vAllocPd(c: *vabi.Context) callconv(.c) ?*vabi.Pd {
    return &ctxOf(c).pd;
}
fn vDeallocPd(_: *vabi.Pd) callconv(.c) c_int {
    return 0;
}
fn vRegMr(pd: *vabi.Pd, addr: ?*anyopaque, len: usize, access: c_int) callconv(.c) ?*vabi.Mr {
    const x = ctxOf(pd.context);
    fx.lockIt();
    defer fx.unlockIt();
    const m = &fx.mrs[fx.n_mr];
    const key: u32 = @intCast(1000 + fx.n_mr);
    m.* = .{ .addr = @intFromPtr(addr), .len = len, .access = access, .node = x.node, .rail = x.rail, .key = key, .live = true, .ctl = @intFromPtr(addr) == fx.ctl_addr[x.node], .mr = undefined };
    m.mr.addr = addr;
    m.mr.length = len;
    m.mr.lkey = key;
    m.mr.rkey = key;
    fx.n_mr += 1;
    fx.n_registered_under_devx += 1;
    return &m.mr;
}
fn vDeregMr(mr: *vabi.Mr) callconv(.c) c_int {
    fx.lockIt();
    defer fx.unlockIt();
    const m: *FMr = @fieldParentPtr("mr", mr);
    m.live = false;
    if (m.ctl) fx.ev('M');
    return 0;
}
fn newCq(c: *vabi.Context, cqe: c_int) *FCq {
    const x = ctxOf(c);
    var cnt: u32 = 16;
    while (cnt < cqe) cnt *= 2;
    const q = &fx.cqs[fx.n_cq];
    q.* = .{ .ibv = std.mem.zeroes(vabi.Cq), .buf = gpa.alignedAlloc(u8, .@"64", cnt * 64) catch @panic("cq"), .cnt = cnt, .pi = 0, .dbrec = .{ 0, 0 }, .cqn = @intCast(0x200 + fx.n_cq), .node = x.node, .rail = x.rail, .wcs = undefined, .n_wc = 0, .live = true };
    q.ibv.context = c;
    for (0..cnt) |i| q.buf[i * 64 + 63] = 0xf0;
    fx.n_cq += 1;
    return q;
}
fn vCreateCq(c: *vabi.Context, cqe: c_int, _: ?*anyopaque, _: ?*anyopaque, _: c_int) callconv(.c) ?*vabi.Cq {
    fx.lockIt();
    defer fx.unlockIt();
    return &newCq(c, cqe).ibv;
}
fn vDestroyCq(cq: *vabi.Cq) callconv(.c) c_int {
    fx.lockIt();
    defer fx.unlockIt();
    const q: *FCq = @fieldParentPtr("ibv", cq);
    if (q.live) gpa.free(q.buf);
    q.live = false;
    return 0;
}
fn vCreateQp(pd: *vabi.Pd, init: *vabi.QpInitAttr) callconv(.c) ?*vabi.Qp {
    const x = ctxOf(pd.context);
    fx.lockIt();
    defer fx.unlockIt();
    const q = &fx.vqps[fx.n_vqp];
    fx.n_vqp += 1;
    q.* = .{ .qp = std.mem.zeroes(vabi.Qp), .used = true, .node = x.node, .rail = x.rail, .remote = 0, .cq = @fieldParentPtr("ibv", init.send_cq), .state = 0 };
    q.qp.qp_num = fx.next_vqpn;
    fx.next_vqpn += 1;
    return &q.qp;
}
fn vModifyQp(qp: *vabi.Qp, a: *vabi.QpAttr, _: c_int) callconv(.c) c_int {
    fx.lockIt();
    defer fx.unlockIt();
    const q: *FVqp = @fieldParentPtr("qp", qp);
    q.state = a.qp_state;
    if (a.qp_state == vabi.qps_rtr) q.remote = a.dest_qp_num;
    return 0;
}
fn vDestroyQp(qp: *vabi.Qp) callconv(.c) c_int {
    fx.lockIt();
    defer fx.unlockIt();
    const q: *FVqp = @fieldParentPtr("qp", qp);
    q.used = false;
    fx.ev('A');
    return 0;
}
/// The abort pair's post: an inline WRITE delivered at once to the peer's abort pair's node, if that pair is up and the control block is registered.
fn vPostSend(qp: *vabi.Qp, wr: *vabi.SendWr, _: *?*vabi.SendWr) callconv(.c) c_int {
    fx.lockIt();
    defer fx.unlockIt();
    const q: *FVqp = @fieldParentPtr("qp", qp);
    if (q.state != vabi.qps_rts) return 1;
    var dest: ?*FVqp = null;
    for (fx.vqps[0..fx.n_vqp]) |*o| if (o.used and o.qp.qp_num == q.remote) {
        dest = o;
    };
    const d = dest orelse return 0; // (best effort: a peer that is gone takes nothing)
    const rm = mrOfKey(wr.rkey) orelse return 0;
    if (rm.node != d.node or rm.rail != q.rail or rm.access & vabi.access_remote_write == 0 or !covers(rm, @intCast(wr.remote_addr), 8)) return 0;
    const v = std.mem.readInt(u64, @as(*align(1) const [8]u8, @ptrFromInt(wr.sg_list[0].addr)), builtin.cpu.arch.endian());
    @atomicStore(u64, @as(*u64, @ptrFromInt(wr.remote_addr)), v, .release);
    if (wr.send_flags & vabi.send_signaled != 0 and q.cq.n_wc < q.cq.wcs.len) {
        q.cq.wcs[q.cq.n_wc] = std.mem.zeroes(vabi.Wc);
        q.cq.n_wc += 1;
    }
    return 0;
}
fn vPollCq(cq: *vabi.Cq, n: c_int, wc: [*]vabi.Wc) callconv(.c) c_int {
    fx.lockIt();
    defer fx.unlockIt();
    const q: *FCq = @fieldParentPtr("ibv", cq);
    const k: usize = @min(q.n_wc, @as(usize, @intCast(n)));
    for (0..k) |i| wc[i] = q.wcs[i];
    q.n_wc -= k;
    return @intCast(k);
}

fn mOpen(dev: *vabi.Device, attr: *mx.ContextAttr) callconv(.c) ?mx.Ctx {
    if (attr.flags & mx.context_flag_devx == 0) return null;
    const i: usize = if (dev == &fx.devs[0]) 0 else 1;
    fx.n_open_ctx += 1;
    return &fx.ctxs[@as(usize, fx.cur_node) * rails + i].ctx;
}
fn mInitObj(obj: *mx.DvObj, t: u64) callconv(.c) c_int {
    if (t == mx.obj_pd) {
        const pd: *vabi.Pd = @ptrCast(@alignCast(obj.pairs[6][0].?));
        const out: *mx.DvPd = @ptrCast(@alignCast(obj.pairs[6][1].?));
        out.pdn = ctxOf(pd.context).pdn;
        return 0;
    }
    if (t == mx.obj_cq) {
        const cq: *vabi.Cq = @ptrCast(@alignCast(obj.pairs[1][0].?));
        const out: *mx.DvCq = @ptrCast(@alignCast(obj.pairs[1][1].?));
        const q: *FCq = @fieldParentPtr("ibv", cq);
        out.* = .{ .buf = q.buf.ptr, .dbrec = @ptrCast(&q.dbrec), .cqe_cnt = q.cnt, .cqe_size = 64, .cqn = q.cqn };
        return 0;
    }
    return 1;
}
fn mObjCreate(c: mx.Ctx, in: *const anyopaque, inlen: usize, out: *anyopaque, outlen: usize) callconv(.c) ?*mx.DevxObj {
    const x = ctxOf(c);
    const b: [*]const u8 = @ptrCast(in);
    const o: [*]u8 = @ptrCast(out);
    std.debug.assert(inlen == mx.qp_msg_bytes and outlen == mx.small_msg_bytes);
    fx.lockIt();
    defer fx.unlockIt();
    const q_ = b[mx.msg.qpc_at..inlen];
    const log: u8 = @intCast(mx.get(q_, mx.qpc.log_sq_size));
    if (mx.get(b[0..inlen], mx.msg.opcode) != mx.op.create_qp or log > fx.max_log_sq) {
        mx.put(o[0..outlen], mx.msg.out_syndrome, 0x7);
        std.c._errno().* = 22;
        return null;
    }
    const wq = mx.get(b[0..inlen], mx.msg.wq_umem_id);
    const dbru = mx.get(q_, mx.qpc.dbr_umem_id);
    var sq: usize = 0;
    var dbr: usize = 0;
    for (fx.ums[0..fx.n_um]) |u| {
        if (u.id == wq) sq = u.addr;
        if (u.id == dbru) dbr = u.addr + @as(usize, @intCast(mx.get(q_, mx.qpc.dbr_addr)));
    }
    var cqp: ?*FCq = null;
    for (fx.cqs[0..fx.n_cq]) |*cq| if (cq.live and cq.cqn == mx.get(q_, mx.qpc.cqn_snd)) {
        cqp = cq;
    };
    // a queue pair as m19c makes it: reliable-connected, no receive queue, a send queue and a doorbell record in umems, a pd of this context
    if (mx.get(q_, mx.qpc.st) != mx.st_rc or mx.get(q_, mx.qpc.rq_type) != mx.rq_type_zero_size or sq == 0 or dbr == 0 or cqp == null or mx.get(q_, mx.qpc.pd) != x.pdn or mx.get(b[0..inlen], mx.msg.wq_umem_valid) != 1 or mx.get(q_, mx.qpc.dbr_umem_valid) != 1) {
        fx.violations += 1;
        std.c._errno().* = 22;
        return null;
    }
    const qp = &fx.qps[fx.n_qp];
    fx.n_qp += 1;
    const cdm = mx.get(q_, mx.qpc.cd_master) == 1;
    qp.* = .{ .obj = 0, .used = true, .dead = false, .state = 0, .node = x.node, .rail = x.rail, .qpn = fx.next_qpn, .depth = @as(u32, 1) << @intCast(log), .sq = sq, .dbr = dbr, .cq = cqp.?, .cd_master = cdm, .rq_type = @intCast(mx.get(q_, mx.qpc.rq_type)), .remote_qpn = 0, .ci = 0, .flushed = false };
    fx.next_qpn += 1;
    if (cdm) fx.created_cd_master += 1 else fx.created_other += 1;
    mx.put(o[0..outlen], mx.msg.create_out_qpn, qp.qpn);
    return @ptrCast(qp);
}
fn mObjModify(obj: *mx.DevxObj, in: *const anyopaque, inlen: usize, out: *anyopaque, outlen: usize) callconv(.c) c_int {
    const q: *FQp = @ptrCast(@alignCast(obj));
    const b = @as([*]const u8, @ptrCast(in))[0..inlen];
    const o = @as([*]u8, @ptrCast(out))[0..outlen];
    const opc: u16 = @intCast(mx.get(b, mx.msg.opcode));
    if (opc == mx.op.qp_2err and fx.err_delay_ns != 0) sleepNs(fx.err_delay_ns);
    fx.lockIt();
    defer fx.unlockIt();
    if (fx.n_tr < fx.transitions.len) {
        fx.transitions[fx.n_tr] = opc;
        fx.n_tr += 1;
    }
    const qq = if (inlen > mx.msg.qpc_at) b[mx.msg.qpc_at..] else b[0..0];
    const step: usize = switch (opc) {
        mx.op.rst2init_qp => 0,
        mx.op.init2rtr_qp => 1,
        mx.op.rtr2rts_qp => 2,
        else => 3,
    };
    if (step < 3 and inlen > mx.msg.qpc_at) {
        q.opt_masks[step] = @intCast(mx.get(b, mx.msg.opt_param_mask));
        if (mx.get(qq, mx.qpc.rae) == 1) {
            q.rae = true;
            q.atomic_mode = @intCast(mx.get(qq, mx.qpc.atomic_mode));
        }
    }
    switch (opc) {
        mx.op.rst2init_qp => {
            if (q.state != 0 or mx.get(b, mx.msg.qpn) != q.qpn) return bad(o);
            q.state = 1;
        },
        mx.op.init2rtr_qp => {
            if (q.state != 1 or mx.get(b, mx.msg.qpn) != q.qpn) return bad(o);
            // the peer's address: a GID and a MAC that are not empty, a path MTU of 4 KiB
            var any: u8 = 0;
            for (qq[mx.qpc.pap_rgid_rip_byte..][0..16]) |c| any |= c;
            if (any == 0 or mx.get(qq, mx.qpc.pap_rmac_31_0) == 0 or mx.get(qq, mx.qpc.mtu) != 5) return bad(o);
            q.remote_qpn = @intCast(mx.get(qq, mx.qpc.remote_qpn));
            q.log_rra = @intCast(mx.get(qq, mx.qpc.log_rra_max));
            q.state = 2;
        },
        mx.op.rtr2rts_qp => {
            if (q.state != 2) return bad(o);
            q.log_sra = @intCast(mx.get(qq, mx.qpc.log_sra_max));
            q.state = 3;
        },
        mx.op.qp_2err => {
            if (q.dead) return bad(o);
            q.state = 6;
            fx.ev('E');
        },
        else => return bad(o),
    }
    return 0;
}
fn bad(o: []u8) c_int {
    mx.put(o, mx.msg.out_status, 3);
    return 22;
}
fn mObjQuery(obj: *mx.DevxObj, _: *const anyopaque, _: usize, out: *anyopaque, outlen: usize) callconv(.c) c_int {
    const q: *FQp = @ptrCast(@alignCast(obj));
    const o = @as([*]u8, @ptrCast(out))[0..outlen];
    mx.put(o[mx.msg.qpc_at..], mx.qpc.state, q.state);
    return 0;
}
fn mObjDestroy(obj: *mx.DevxObj) callconv(.c) c_int {
    const q: *FQp = @ptrCast(@alignCast(obj));
    fx.lockIt();
    defer fx.unlockIt();
    q.dead = true;
    fx.ev('Q');
    return 0;
}
fn mUmemReg(_: mx.Ctx, addr: ?*anyopaque, size: usize, _: u32) callconv(.c) ?*mx.DevxUmem {
    fx.lockIt();
    defer fx.unlockIt();
    const u = &fx.ums[fx.n_um];
    fx.n_um += 1;
    u.* = .{ .addr = @intFromPtr(addr), .len = size, .id = fx.next_umem, .um = .{ .umem_id = fx.next_umem } };
    fx.next_umem += 1;
    return &u.um;
}
fn mUmemDereg(_: *mx.DevxUmem) callconv(.c) c_int {
    return 0;
}
fn mAllocUar(c: mx.Ctx, flags: u32) callconv(.c) ?*mx.DevxUar {
    const x = ctxOf(c);
    fx.last_uar_type = flags;
    if (flags == mx.uar_type_bf and fx.bf_uar_fails) return null;
    const S = struct {
        var uars: [world * rails]mx.DevxUar = undefined;
    };
    const u = &S.uars[@as(usize, x.node) * rails + x.rail];
    u.* = .{ .reg_addr = &fx.uar_cell[x.rail], .base_addr = null, .page_id = 7, .mmap_off = 0, .comp_mask = 0 };
    return u;
}
fn mFreeUar(_: *mx.DevxUar) callconv(.c) void {}

fn fakeVerbs() Verbs {
    var v: Verbs = undefined;
    v.api.ibv_get_device_list = vGetList;
    v.api.ibv_free_device_list = vFreeList;
    v.api.ibv_get_device_name = vDevName;
    v.api.ibv_close_device = vCloseDev;
    v.api.ibv_query_port = vQueryPort;
    v.api.ibv_query_gid = vQueryGid;
    v.api.ibv_query_device = vQueryDevice;
    v.api.ibv_alloc_pd = vAllocPd;
    v.api.ibv_dealloc_pd = vDeallocPd;
    v.api.ibv_reg_mr = vRegMr;
    v.api.ibv_dereg_mr = vDeregMr;
    v.api.ibv_create_cq = vCreateCq;
    v.api.ibv_destroy_cq = vDestroyCq;
    v.api.ibv_create_qp = vCreateQp;
    v.api.ibv_modify_qp = vModifyQp;
    v.api.ibv_destroy_qp = vDestroyQp;
    return v;
}

fn fakeMlx() mx.Mlx5 {
    return mx.Mlx5.fake(.{
        .mlx5dv_open_device = mOpen,
        .mlx5dv_init_obj = mInitObj,
        .mlx5dv_devx_obj_create = mObjCreate,
        .mlx5dv_devx_obj_modify = mObjModify,
        .mlx5dv_devx_obj_query = mObjQuery,
        .mlx5dv_devx_obj_destroy = mObjDestroy,
        .mlx5dv_devx_umem_reg = mUmemReg,
        .mlx5dv_devx_umem_dereg = mUmemDereg,
        .mlx5dv_devx_alloc_uar = mAllocUar,
        .mlx5dv_devx_free_uar = mFreeUar,
    });
}

fn fakeMac(device: []const u8) error{NoMac}![6]u8 {
    return .{ 0x02, 0x00, 0x00, 0x00, if (std.mem.eql(u8, device, "mlx5_0")) 0 else 1, @intCast(fx.cur_node + 1) };
}

// -- four nodes ------------------------------------------------------------------------------------------------------

const Opts = struct {
    log_sq: u8 = 11,
    signal_every: u32 = 16,
    janitor_ms: u32 = 20,
    sq_wait_ms: u64 = 2000,
    dead_ms: u64 = 5000,
    settle_ms: u32 = 5,
    host: bool = false,
    start: bool = true,
    max_log_sq: u8 = 15,
    bf_uar_fails: bool = false,
    signal: roci.Signal = .write,
    atomic_cap: c_int = vabi.atomic_hca,
    rd_atom: c_int = 16,
};

const Nd = struct {
    arena: []align(std.heap.page_size_min) u8,
    ctl: *roci.Control,
    nic: *cd.CdNic = undefined,
    t: *roci.Transport = undefined,
};

const Fab = struct {
    nd: [world]Nd = undefined,
    verbs: Verbs,
    mlx: mx.Mlx5,
    opts: Opts,

    fn open(o: Opts) !*Fab {
        fxReset();
        fx.max_log_sq = o.max_log_sq;
        fx.bf_uar_fails = o.bf_uar_fails;
        fx.atomic_cap = o.atomic_cap;
        fx.rd_atom = o.rd_atom;
        const f = try gpa.create(Fab);
        f.* = .{ .nd = undefined, .verbs = fakeVerbs(), .mlx = fakeMlx(), .opts = o };
        const total = roci.layoutOf(if (o.host) .host else .prompt).total;
        for (&f.nd, 0..) |*nd, g| {
            nd.arena = try std.posix.mmap(null, total, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
            nd.ctl = try gpa.create(roci.Control);
            nd.ctl.* = .{};
            fx.ctl_addr[g] = @intFromPtr(nd.ctl);
        }
        fx.card = try std.Thread.spawn(.{}, cardMain, .{});
        var all: [world]rv.Info = undefined;
        for (&f.nd, 0..) |*nd, g| {
            fx.cur_node = @intCast(g);
            const ctl_bytes = std.mem.asBytes(nd.ctl);
            nd.nic = try cd.CdNic.open(gpa, &f.verbs, &f.mlx, .{ .log_sq_size = o.log_sq, .signal_every = o.signal_every, .sq_wait_ms = o.sq_wait_ms, .settle_ms = o.settle_ms, .mac_of = fakeMac, .signal = o.signal }, @intCast(g), nd.arena, ctl_bytes, 77);
            all[g] = try nd.nic.info();
        }
        for (&f.nd) |*nd| try nd.nic.connect(&all);
        const set: roci.Settings = .{ .wait_qp = true, .nap = false, .janitor_ms = o.janitor_ms, .dead_ms = o.dead_ms, .abort_rounds = 1, .abort_settle_us = 100, .codes = true, .gathers = true };
        for (&f.nd, 0..) |*nd, g| {
            nd.t = try roci.Transport.init(gpa, @intCast(g), roci.identity, nd.ctl, @intFromPtr(nd.ctl), nd.nic.nic(), set);
            if (o.start) try nd.t.start(null);
        }
        return f;
    }

    fn close(f: *Fab) void {
        for (&f.nd) |*nd| nd.t.deinit();
        for (&f.nd) |*nd| nd.nic.close();
        fx.stop.store(true, .release);
        fx.card.?.join();
        for (&f.nd) |*nd| {
            std.posix.munmap(nd.arena);
            gpa.destroy(nd.ctl);
        }
        gpa.destroy(f);
    }
};

/// The stream of a node: an op runs when it is called; a wait blocks the caller until its words are there (cyclic compare), as the GPU's
/// stream wait does.
const FakeStream = struct {
    timeout_ns: u64 = 10 * std.time.ns_per_s,

    pub fn writeValue64(_: FakeStream, addr: u64, v: u64) !void {
        @atomicStore(u64, @as(*u64, @ptrFromInt(addr)), v, .release);
    }

    pub fn waitValues64(s: FakeStream, words: []const u64, values: []const u64) !void {
        const t0 = nowNs();
        for (words, values) |w, v| while (true) {
            const have = @atomicLoad(u64, @as(*const u64, @ptrFromInt(w)), .acquire);
            if (@as(i64, @bitCast(have -% v)) >= 0) break;
            if (nowNs() - t0 > s.timeout_ns) return error.WaitTimeout;
            std.atomic.spinLoopHint();
        };
    }
};

fn pattern(buf: []u8, tag: u64) void {
    const ws = std.mem.bytesAsSlice(u64, buf[0 .. buf.len / 8 * 8]);
    for (ws, 0..) |*w, i| w.* = tag *% 0x9e3779b97f4a7c15 +% i;
}

fn fillSources(nd: *Nd, me: usize, class: roci.Class, gen: u64, cap: usize) void {
    const e = roci.buildEvent(class, cap, me, roci.identity);
    for (0..e.n_peers) |k| for (0..rails) |r| for (e.chains[k][r].slice()) |p| {
        pattern(nd.arena[p.src..][0..p.len], gen *% 1000003 +% @as(u64, me) *% 7919 +% p.src);
    };
}

fn badBytes(f: *Fab, me: usize, class: roci.Class, gen: u64, cap: usize) !u64 {
    var bad_: u64 = 0;
    var srcs: [world]u8 = undefined;
    for (roci.sourcesOf(class, @intCast(me), &srcs)) |s| {
        const e = roci.buildEvent(class, cap, s, roci.identity);
        const k = std.mem.indexOfScalar(u8, e.peerSlice(), @intCast(me)).?;
        for (0..rails) |r| for (e.chains[k][r].slice()) |p| {
            const want = try gpa.alloc(u8, p.len);
            defer gpa.free(want);
            pattern(want, gen *% 1000003 +% @as(u64, s) *% 7919 +% p.src);
            const got = f.nd[me].arena[p.dst..][0..p.len];
            const n = p.len / 8 * 8;
            if (!std.mem.eql(u8, want[0..n], got[0..n])) for (want[0..n], got[0..n]) |a, b| {
                bad_ += @intFromBool(a != b);
            };
        };
    }
    return bad_;
}

const Barrier = struct {
    n: std.atomic.Value(u32) = .init(0),
    gen: std.atomic.Value(u32) = .init(0),
    fn wait(b: *Barrier, parties: u32) void {
        const g = b.gen.load(.acquire);
        if (b.n.fetchAdd(1, .acq_rel) + 1 == parties) {
            b.n.store(0, .release);
            _ = b.gen.fetchAdd(1, .acq_rel);
        } else while (b.gen.load(.acquire) == g) std.atomic.spinLoopHint();
    }
};

const Run = struct {
    f: *Fab,
    bar: *Barrier,
    classes: []const roci.Class,
    gens: u64,
    bad: [world]u64 = @splat(0),
    err: [world]?anyerror = @splat(null),
};

fn rankMain(run: *Run, g: usize) void {
    rankBody(run, g) catch |e| {
        run.err[g] = e;
    };
}

fn rankBody(run: *Run, g: usize) !void {
    const nd = &run.f.nd[g];
    const s: FakeStream = .{};
    for (0..run.gens) |gen_| {
        for (run.classes) |class| {
            const cap: usize = if (gen_ % 2 == 0) 4096 else 2048;
            const shape: u8 = if (gen_ % 2 == 0) 0 else 1;
            fillSources(nd, g, class, gen_ + 1, cap);
            run.bar.wait(world); // (the loopback has no GPU: the test keeps the dataflow's fence)
            try nd.t.send(s, class, shape);
            try nd.t.wait(s, class);
            run.bad[g] += try badBytes(run.f, g, class, gen_ + 1, cap);
            run.bar.wait(world);
        }
    }
}

fn runProgram(f: *Fab, classes: []const roci.Class, gens: u64) !Run {
    var bar: Barrier = .{};
    var run: Run = .{ .f = f, .bar = &bar, .classes = classes, .gens = gens };
    var th: [world]std.Thread = undefined;
    for (&th, 0..) |*t, g| t.* = try std.Thread.spawn(.{}, rankMain, .{ &run, g });
    for (th) |t| t.join();
    return run;
}

fn expectClean(run: Run) !void {
    for (run.err) |e| if (e) |x| {
        std.debug.print("rank error {s}\n", .{@errorName(x)});
        return error.TestUnexpectedResult;
    };
    for (run.bad) |b| try std.testing.expectEqual(@as(u64, 0), b);
}

// -- tests -----------------------------------------------------------------------------------------------------------

test "wait mode: the WAIT's code is INVERT|BIGGER (0x12) with the ticket's big-endian bytes as value and a full mask, and releases a ticket's chain iff the big-endian doorbell word has reached it, across byte boundaries too; the model can tell a little-endian word and the other codes from it" {
    // the pinned encoding: code 0x12, value bytes big-endian, mask all ones
    try std.testing.expectEqual(@as(u32, 0x12), @as(u32, @backingInt(cd.wait_cond)) | (if (cd.wait_invert) @as(u32, 0x10) else 0));
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0, 0, 0, 0x01, 0x02 }, std.mem.asBytes(&cd.waitValue(0x0102)));
    try std.testing.expectEqual(~@as(u64, 0), cd.wait_mask);
    const op: u32 = 0x12;
    const tickets = [_]u64{ 1, 2, 3, 254, 255, 256, 257, 511, 512, 65535, 65536, 70000, 1 << 24, (1 << 24) + 1 };
    for (tickets) |t| {
        for ([_]u64{ 0, 1, t -| 1, t, t + 1, t + 255, t * 2, 1 << 30 }) |word| {
            const mem = std.mem.nativeToBig(u64, word);
            try std.testing.expectEqual(word >= t, modelReleased(op, cd.waitValue(t), cd.wait_mask, mem));
        }
        // the abort value (2^62), stored big-endian, satisfies every ticket's wait
        try std.testing.expect(modelReleased(op, cd.waitValue(t), cd.wait_mask, std.mem.nativeToBig(u64, roci.abort_value)));
    }
    // control 1: a little-endian word: ticket 256 against word 255 (bytes ff 00 ..) reads as a huge number and releases early
    try std.testing.expect(modelReleased(op, cd.waitValue(256), cd.wait_mask, 255));
    // control 2: BIGGER alone fires while the word is below the ticket (value above word), CYCLIC_SMALLER with value ticket - 1 needs the same order
    try std.testing.expect(modelReleased(@backingInt(mx.WaitCond.bigger), cd.waitValue(5), cd.wait_mask, std.mem.nativeToBig(u64, 3)));
    // control 3: EQUAL misses a word that went past the ticket
    try std.testing.expect(!modelReleased(@backingInt(mx.WaitCond.equal), cd.waitValue(5), cd.wait_mask, std.mem.nativeToBig(u64, 7)));
}

test "wait mode: the doorbell word is the ticket's big-endian bytes in this mode and the ticket itself in the other, and ticketOf is its inverse" {
    const ctl = try gpa.create(roci.Control);
    defer gpa.destroy(ctl);
    ctl.* = .{};
    const S = struct {
        fn post(_: *anyopaque, _: u8, _: u8, _: roci.Class, _: *const roci.Chain, _: u64) roci.PostError!void {}
        fn pre(_: *anyopaque, _: u8, _: u8, _: roci.Class, _: *const roci.Chain, _: u64, _: u64) roci.PostError!void {}
        fn fail(_: *anyopaque) void {}
        fn reap(_: *anyopaque, _: u8) roci.PostError!void {}
    };
    var dummy: u8 = 0;
    const nic: roci.Nic = .{ .ctx = &dummy, .post_fn = S.post, .fail_fn = S.fail, .reap_fn = S.reap, .prepost_fn = S.pre };
    const w = try roci.Transport.init(gpa, 0, roci.identity, ctl, @intFromPtr(ctl), nic, .{ .wait_qp = true });
    const d = try roci.Transport.init(gpa, 0, roci.identity, ctl, @intFromPtr(ctl), nic, .{});
    defer d.deinit();
    defer w.deinit();
    for ([_]u64{ 1, 255, 256, 0x0102030405060708, roci.abort_value }) |t| {
        try std.testing.expectEqual(t, d.go(t));
        try std.testing.expectEqual(t, d.ticketOf(t));
        try std.testing.expectEqual(t, w.ticketOf(w.go(t)));
        var be: [8]u8 = undefined;
        std.mem.writeInt(u64, &be, t, .big);
        try std.testing.expectEqualSlices(u8, &be, std.mem.asBytes(&w.go(t)));
        // the WAIT entry's value is the doorbell word's bytes
        try std.testing.expectEqual(w.go(t), cd.waitValue(t));
    }
    // a send's stream write stores the swapped word in this mode, the ticket as it always did in the other
    const F = struct {
        var last: u64 = 0;
        pub fn writeValue64(_: @This(), _: u64, v: u64) !void {
            last = v;
        }
    };
    const f: F = .{};
    try w.send(f, .own1, 0);
    try std.testing.expectEqual(w.go(1), F.last);
    try std.testing.expect(F.last != 1);
    try d.send(f, .own1, 0);
    try std.testing.expectEqual(@as(u64, 1), F.last);
}

test "wait mode: open makes the queue pairs as m19c does (Core Direct master, no receive queue, RST2INIT, INIT2RTR, RTR2RTS), under a DEVX context, with an abort pair beside each, and starts no proxy" {
    const f = try Fab.open(.{});
    defer f.close();
    // 4 nodes x 2 rails x 3 peers
    try std.testing.expectEqual(@as(u32, 24), fx.created_cd_master);
    try std.testing.expectEqual(@as(u32, 0), fx.created_other);
    try std.testing.expectEqual(@as(usize, 8), fx.n_open_ctx);
    try std.testing.expectEqual(@as(u32, 0), fx.violations);
    // every queue pair walked RST2INIT, INIT2RTR, RTR2RTS in that order
    try std.testing.expectEqual(@as(usize, 72), fx.n_tr);
    for (0..24) |i| {
        try std.testing.expectEqual(@as(u16, mx.op.rst2init_qp), fx.transitions[i * 3]);
        try std.testing.expectEqual(@as(u16, mx.op.init2rtr_qp), fx.transitions[i * 3 + 1]);
        try std.testing.expectEqual(@as(u16, mx.op.rtr2rts_qp), fx.transitions[i * 3 + 2]);
    }
    for (fx.qps[0..24]) |q| try std.testing.expectEqual(@as(u8, 3), q.state);
    // the abort pairs: 24 more, plain verbs
    try std.testing.expectEqual(@as(usize, 24), fx.n_vqp);
    // each data pair is connected to the peer's pair made for it: its remote number is a pair of the same rail on the other node
    for (fx.qps[0..24]) |q| {
        const r = qpByQpn(q.remote_qpn).?;
        try std.testing.expectEqual(q.rail, r.rail);
        try std.testing.expect(r.node != q.node);
        try std.testing.expectEqual(q.qpn, r.remote_qpn);
    }
    // the proxies' threads are not there: the janitor is
    for (&f.nd) |*nd| {
        for (nd.t.threads) |th| try std.testing.expect(th == null);
        try std.testing.expect(nd.t.janitor != null);
    }
    // each node registered the arena and the control block on both rails under its DEVX contexts
    try std.testing.expectEqual(@as(usize, 16), fx.n_registered_under_devx);
    // the Info says wait, the MACs and the abort pairs
    const info = try f.nd[2].nic.info();
    try std.testing.expectEqual(@as(u64, 1), info.wait);
    try std.testing.expect(info.mac[1][5] == 3 and info.mac[1][4] == 1);
    try std.testing.expect(info.abort_qpn[0][0] != 0 and info.abort_qpn[0][2] == 0);
}

test "wait mode: a queue pair create the card refuses is retried one size down; a BlueFlame UAR it refuses falls back to the non-cached one" {
    const f = try Fab.open(.{ .log_sq = 12, .start = false, .max_log_sq = 11, .bf_uar_fails = true });
    defer f.close();
    for (fx.qps[0..24]) |q| try std.testing.expectEqual(@as(u32, 2048), q.depth);
    try std.testing.expectEqual(mx.uar_type_nc, fx.last_uar_type);
}

test "wait mode: nodes that disagree about waiting, about the arrival signal or about the ATOMICs in flight refuse to connect, by name" {
    var all: [world]rv.Info = @splat(.{ .layout_hash = 7, .signal = 0, .wait = 1 });
    const mine = all[0];
    try rv.checkPeers(&all, mine);
    all[3].wait = 0; // a node on the verbs path
    try std.testing.expectError(error.WaitMismatch, rv.checkPeers(&all, mine));
    all[3].wait = 1;
    all[2].signal = @backingInt(roci.Signal.atomic); // a node that signals with ATOMICs into a block the others write
    try std.testing.expectError(error.SignalMismatch, rv.checkPeers(&all, mine));
}

/// The words a part's arrival signal writes as WRITEs: one for the write signal, none for the ATOMIC one (its word is added to by the card).
fn wordWrites(sig: roci.Signal) u64 {
    return if (sig == .write) 1 else 0;
}

fn chainBehindWait(sig: roci.Signal) !void {
    const f = try Fab.open(.{ .signal = sig });
    defer f.close();
    const a = &f.nd[0];
    const s: FakeStream = .{};
    const lo = roci.layout();
    pattern(a.arena[lo.at(.psum)..][0..4096], 11);
    // three events reserved (their chains pre-posted) before any doorbell write
    const t1 = try a.t.reserve(.own1, 0);
    const t2 = try a.t.reserve(.own1, 0);
    const t3 = try a.t.reserve(.own1, 0);
    try std.testing.expectEqual(@as(u64, 3), t3);
    try std.testing.expect(t1 == 1 and t2 == 2);
    sleepNs(30 * std.time.ns_per_ms);
    fx.lockIt();
    try std.testing.expectEqual(@as(u64, 0), fx.writes);
    try std.testing.expectEqual(@as(u64, 0), fx.waits_released);
    fx.unlockIt();
    // the peer's words have not moved either
    const peer = &f.nd[1];
    for (&peer.ctl.recv[@backingInt(roci.Class.own1)][0]) |*w| try std.testing.expectEqual(@as(u64, 0), w.v);
    // the doorbell at 1 lets the first event go and no more: 2 rails x 1 peer = 2 chains of (1 data + 1 word)
    try s.writeValue64(a.t.doorbellAddr(), a.t.go(1));
    const t0 = nowNs();
    while (@atomicLoad(u64, &peer.ctl.recv[@backingInt(roci.Class.own1)][0][0].v, .acquire) < 1 or @atomicLoad(u64, &peer.ctl.recv[@backingInt(roci.Class.own1)][0][1].v, .acquire) < 1) {
        if (nowNs() - t0 > 3 * std.time.ns_per_s) return error.TestUnexpectedResult;
        sleepNs(100_000);
    }
    sleepNs(30 * std.time.ns_per_ms);
    fx.lockIt();
    try std.testing.expectEqual(@as(u64, 2), fx.waits_released);
    try std.testing.expectEqual(2 * (1 + wordWrites(sig)), fx.writes);
    try std.testing.expectEqual(2 * (1 - wordWrites(sig)), fx.atomics);
    fx.unlockIt();
    try std.testing.expectEqual(@as(u64, 1), peer.ctl.recv[@backingInt(roci.Class.own1)][0][0].v);
    // the doorbell at 3 (a stream write that skipped 2) releases the other two at once: the compare is "reached", not "equal"
    try s.writeValue64(a.t.doorbellAddr(), a.t.go(3));
    const t1_ = nowNs();
    while (@atomicLoad(u64, &peer.ctl.recv[@backingInt(roci.Class.own1)][0][0].v, .acquire) < 3 or @atomicLoad(u64, &peer.ctl.recv[@backingInt(roci.Class.own1)][0][1].v, .acquire) < 3) {
        if (nowNs() - t1_ > 3 * std.time.ns_per_s) return error.TestUnexpectedResult;
        sleepNs(100_000);
    }
    try std.testing.expectEqual(@as(u32, 0), fx.violations);
    // the words are the generations, 1, 2, 3, each on its rail
    try std.testing.expectEqual(@as(u64, 3), peer.ctl.recv[@backingInt(roci.Class.own1)][0][1].v);
}

test "wait mode: a chain is in the queue behind its WAIT and the card sends nothing until the doorbell reaches the ticket; chains released in order, a later one waiting for its own ticket (the write signal)" {
    try chainBehindWait(.write);
}

test "wait mode, ATOMIC signal: a chain is in the queue behind its WAIT and the card sends nothing until the doorbell reaches the ticket; the receiver reads the same words, 1, 2, 3, as with the write signal" {
    try chainBehindWait(.atomic);
}

fn fourNodes(sig: roci.Signal) !u64 {
    const f = try Fab.open(.{ .signal = sig });
    defer f.close();
    const classes = [_]roci.Class{ .own1, .own2, .eng1, .eng2, .wob, .pack, .ga, .gf };
    const run = try runProgram(f, &classes, 3);
    try expectClean(run);
    fx.lockIt();
    defer fx.unlockIt();
    try std.testing.expectEqual(@as(u32, 0), fx.violations);
    try std.testing.expect(fx.waits_released > 0 and fx.writes + fx.atomics > fx.waits_released);
    for (&f.nd) |*nd| {
        const c = nd.t.counters();
        try std.testing.expect(c.events == 3 * classes.len and c.stream_writes == 3 * classes.len);
    }
    // the arrival signals the card ran: the word WRITEs, or the ATOMICs
    return if (sig == .write) fx.words else fx.atomics;
}

test "wait mode: four nodes, owner sums, codes and gathers: every byte lands, generation after generation, both caps, with no proxy thread" {
    const signals = try fourNodes(.write);
    try std.testing.expect(signals > 0);
}

test "wait mode, ATOMIC signal: four nodes, owner sums, codes and gathers: every byte lands as with the write signal, the same number of arrival signals are run, none of them a word WRITE" {
    const want = try fourNodes(.write);
    const got = try fourNodes(.atomic);
    try std.testing.expectEqual(want, got);
    try std.testing.expectEqual(@as(u64, 0), fx.words);
}

fn hostRecords(sig: roci.Signal) !void {
    const f = try Fab.open(.{ .host = true, .signal = sig });
    defer f.close();
    const a = &f.nd[0];
    const lo = roci.layoutOf(.host);
    pattern(a.arena[lo.at(.rec_send) + 2 * roci.host_slot_bytes ..][0 .. 3 * roci.host_rec], 5);
    try a.t.sendRec(.rec, 2, 3);
    const partner = &f.nd[2]; // g ^ 2
    const t0 = nowNs();
    while (@atomicLoad(u64, &partner.ctl.recv[@backingInt(roci.Class.rec)][0][0].v, .acquire) < 1 or @atomicLoad(u64, &partner.ctl.recv[@backingInt(roci.Class.rec)][0][1].v, .acquire) < 1) {
        if (nowNs() - t0 > 3 * std.time.ns_per_s) return error.TestUnexpectedResult;
        sleepNs(100_000);
    }
    fx.lockIt();
    defer fx.unlockIt();
    try std.testing.expectEqual(@as(u64, 0), fx.waits_released); // (no WAIT was ever in the queue)
    try std.testing.expect(fx.writes >= 2 + 2 * wordWrites(sig));
    try std.testing.expectEqual(2 * (1 - wordWrites(sig)), fx.atomics);
    var got: [3 * roci.host_rec]u8 = undefined;
    const slot_at = lo.at(.rec_land) + 2 * roci.host_slot_bytes;
    @memcpy(&got, partner.arena[slot_at..][0..got.len]);
    var want: [3 * roci.host_rec]u8 = undefined;
    pattern(&want, 5);
    try std.testing.expectEqualSlices(u8, want[0 .. want.len / 8 * 8], got[0 .. got.len / 8 * 8]);
}

test "wait mode: the host lane's record events are posted at once with no WAIT, and land with no stream write" {
    try hostRecords(.write);
}

test "wait mode, ATOMIC signal: the host lane's record events are posted at once with no WAIT (the chain ends in the ATOMIC), and land with no stream write" {
    try hostRecords(.atomic);
}

/// Node 0 with two events reserved and never rung, so each of its queue pairs to the peers holds an armed WAIT.
fn armed(f: *Fab) !void {
    const a = &f.nd[0];
    const lo = roci.layout();
    pattern(a.arena[lo.at(.psum)..][0..65536], 3);
    _ = try a.t.reserve(.own1, 0);
    _ = try a.t.reserve(.ga, 0);
    sleepNs(20 * std.time.ns_per_ms);
    fx.lockIt();
    defer fx.unlockIt();
    try std.testing.expectEqual(@as(u64, 0), fx.writes);
}

fn peersUntouched(f: *Fab) !void {
    for (f.nd[1..]) |*nd| {
        for (&nd.ctl.recv) |*c| for (c) |*s| for (s) |*w| try std.testing.expect(w.v == 0 or w.v == roci.abort_value);
        for (nd.arena[roci.layout().at(.stage)..][0..4096]) |b| try std.testing.expectEqual(@as(u8, 0), b);
    }
}

fn failureOrder(sig: roci.Signal) !void {
    const f = try Fab.open(.{ .signal = sig });
    defer f.close();
    fx.err_delay_ns = 5 * std.time.ns_per_ms; // (the ERR command takes real time: the card runs meanwhile)
    try armed(f);
    const a = &f.nd[0];
    a.t.fail(.peer, .own1, 1);
    fx.lockIt();
    const writes = fx.writes;
    const released = fx.waits_released;
    fx.unlockIt();
    try std.testing.expectEqual(@as(u64, 0), writes);
    try std.testing.expectEqual(@as(u64, 0), released);
    try std.testing.expectEqual(a.t.go(roci.abort_value), @atomicLoad(u64, &a.ctl.doorbell.v, .acquire));
    try peersUntouched(f);
    // every peer was told: the abort word, over the abort pairs
    for (f.nd[1..]) |*nd| try std.testing.expectEqual(@as(u64, 1), @atomicLoad(u64, &nd.ctl.abort.v, .acquire));
    // the order of what happened to node 0's card, rail by rail: its three queue pairs to ERR (E), destroyed (Q), the abort pairs destroyed (A),
    // then the rail's control block released (M); the arena's registration stays until close
    try std.testing.expectEqualStrings("EEEQQQAAAM" ++ "EEEQQQAAAM", fx.events[0..fx.n_ev]);
    try std.testing.expectEqual(@as(u32, 0), fx.violations);
    try std.testing.expectEqual(@as(u64, 0), fx.atomics);
}

test "wait mode: a failure takes the queue pairs from the card before the doorbell can hold the abort value: no armed chain is released, the peers hear of it over the abort pairs, the registration goes after the queue pairs" {
    try failureOrder(.write);
}

test "wait mode, ATOMIC signal: the same failure order: no armed chain (no ATOMIC either) is released, the peers hear of it over the abort pairs, the control block is released after the queue pairs" {
    try failureOrder(.atomic);
}

test "control: with the abort value stored into the doorbell first, the armed chains are released and unproduced data and arrival words land in the peers (the ordering test can fail)" {
    const f = try Fab.open(.{});
    defer f.close();
    fx.err_delay_ns = 20 * std.time.ns_per_ms;
    try armed(f);
    roci.Transport.abort_doorbell_first = true;
    defer roci.Transport.abort_doorbell_first = false;
    f.nd[0].t.fail(.peer, .own1, 1);
    fx.lockIt();
    const writes = fx.writes;
    const released = fx.waits_released;
    fx.unlockIt();
    try std.testing.expect(released > 0);
    try std.testing.expect(writes > 0);
    // an "arrived" word in a peer: a count of 1 where no sender ever produced anything
    var arrived: u32 = 0;
    for (f.nd[1..]) |*nd| for (&nd.ctl.recv) |*c| for (c) |*s| for (s) |*w| {
        if (w.v == 1) arrived += 1;
    };
    try std.testing.expect(arrived > 0);
}

fn failFrom(t: *roci.Transport) void {
    t.fail(.peer, .own1, 1);
}

test "wait mode: a failure raised by another thread waits for the engine thread to be out of its reservation before the card is touched; one the engine thread raises itself does not wait for itself" {
    {
        const f = try Fab.open(.{});
        defer f.close();
        const a = &f.nd[0];
        // the engine thread is inside a reservation (some other thread than the failing one)
        a.t.poster_tid.store(1, .monotonic);
        a.t.posting.store(1, .seq_cst);
        const th = try std.Thread.spawn(.{}, failFrom, .{a.t});
        sleepNs(40 * std.time.ns_per_ms);
        fx.lockIt();
        const touched = fx.n_ev;
        fx.unlockIt();
        try std.testing.expectEqual(@as(usize, 0), touched); // (no queue pair has been taken from the card)
        try std.testing.expect(a.t.failure() != null); // (the failure is raised: new enqueues refuse)
        a.t.posting.store(0, .seq_cst);
        th.join();
        try std.testing.expect(fx.n_ev > 0);
    }
    {
        const f = try Fab.open(.{});
        defer f.close();
        const a = &f.nd[0];
        a.t.poster_tid.store(@intCast(std.Thread.getCurrentId()), .monotonic);
        a.t.posting.store(1, .seq_cst);
        const t0 = nowNs();
        a.t.fail(.post, .own1, 1);
        try std.testing.expect(nowNs() - t0 < 60 * std.time.ns_per_ms);
        a.t.posting.store(0, .seq_cst);
        try std.testing.expect(fx.n_ev > 0);
    }
}

test "wait mode: stop with chains armed: the queue pairs are taken from the card and destroyed first, then the registrations; nothing is sent" {
    const f = try Fab.open(.{});
    try armed(f);
    // the transports stop (the janitor ends), then each node's NIC closes
    for (&f.nd) |*nd| nd.t.deinit();
    for (&f.nd) |*nd| nd.nic.close();
    fx.lockIt();
    try std.testing.expectEqual(@as(u64, 0), fx.writes);
    try std.testing.expectEqual(@as(u32, 0), fx.violations);
    fx.unlockIt();
    // every node, rail by rail: queue pairs to ERR, destroyed, abort pairs destroyed, then the control block's registration released
    try std.testing.expectEqualStrings("EEEQQQAAAMEEEQQQAAAM" ++ "EEEQQQAAAMEEEQQQAAAM" ++ "EEEQQQAAAMEEEQQQAAAM" ++ "EEEQQQAAAMEEEQQQAAAM", fx.events[0..fx.n_ev]);
    for (fx.qps[0..fx.n_qp]) |q| try std.testing.expect(q.dead);
    // (the transports and nics are gone: finish what close would)
    fx.stop.store(true, .release);
    fx.card.?.join();
    for (&f.nd) |*nd| {
        std.posix.munmap(nd.arena);
        gpa.destroy(nd.ctl);
    }
    gpa.destroy(f);
}

fn roomThrough(sig: roci.Signal) !void {
    const f = try Fab.open(.{ .log_sq = 6, .signal_every = 4, .signal = sig });
    defer f.close();
    const classes = [_]roci.Class{ .own1, .own2 };
    const run = try runProgram(f, &classes, 20);
    try expectClean(run);
    for (&f.nd) |*nd| try std.testing.expect(nd.t.failure() == null);
    // each queue pair's backlog stayed inside its queue
    fx.lockIt();
    defer fx.unlockIt();
    try std.testing.expectEqual(@as(u32, 0), fx.violations);
}

test "wait mode: the send queue's room is counted in entries and a signalled completion retires them: a hundred events through a 64-entry queue all arrive" {
    try roomThrough(.write);
}

test "wait mode, ATOMIC signal: the send queue's room counts the ATOMIC entry's blocks (one) and a signalled ATOMIC's completion retires the entries before it: a hundred events through a 64-entry queue all arrive" {
    try roomThrough(.atomic);
}

test "wait mode: a queue pair whose completions never come stops the transport after the room wait, naming the post" {
    const f = try Fab.open(.{ .log_sq = 6, .signal_every = 4, .sq_wait_ms = 100 });
    defer f.close();
    fx.drop_cqes = true;
    const a = &f.nd[0];
    const s: FakeStream = .{};
    var failed = false;
    var n: u32 = 0;
    while (n < 400) : (n += 1) {
        a.t.send(s, .own1, 0) catch {
            failed = true;
            break;
        };
    }
    try std.testing.expect(failed);
    try std.testing.expect(n >= 5 and n < 400);
    const fl = a.t.failure().?;
    try std.testing.expectEqual(roci.FailReason.post, fl.reason);
}

test "wait mode: a dead peer's queue pair gives an error completion, the janitor reads it and fails the transport (reason qp), every wait released" {
    const f = try Fab.open(.{ .janitor_ms = 5 });
    defer f.close();
    cd.log_errors = false;
    defer cd.log_errors = true;
    const a = &f.nd[0];
    const s: FakeStream = .{};
    // node 1's queue pairs are gone (its process was killed)
    fx.lockIt();
    for (fx.qps[0..fx.n_qp]) |*q| if (q.node == 1) {
        q.dead = true;
    };
    fx.unlockIt();
    try a.t.send(s, .own1, 0); // (a part for node 1)
    const t0 = nowNs();
    while (a.t.failure() == null) {
        if (nowNs() - t0 > 3 * std.time.ns_per_s) return error.TestUnexpectedResult;
        sleepNs(1_000_000);
    }
    try std.testing.expectEqual(roci.FailReason.qp, a.t.failure().?.reason);
    // (the failure is raised first; the doorbell holds the abort value when its reaction is done)
    const t1 = nowNs();
    while (@atomicLoad(u64, &a.ctl.doorbell.v, .acquire) != a.t.go(roci.abort_value)) {
        if (nowNs() - t1 > 3 * std.time.ns_per_s) return error.TestUnexpectedResult;
        sleepNs(1_000_000);
    }
    try std.testing.expectError(error.RociFailed, a.t.send(s, .own1, 0));
}

test "wait mode: the janitor sleeps in the kernel: about ten looks a second at 100 ms, and a stop ends it at once, not a period later" {
    const f = try Fab.open(.{ .janitor_ms = 100 });
    const t0 = nowNs();
    sleepNs(350 * std.time.ns_per_ms);
    const wakes = f.nd[0].t.rail_counters[0].wakes;
    try std.testing.expect(wakes >= 2 and wakes <= 6);
    const t1 = nowNs();
    f.close();
    try std.testing.expect(nowNs() - t1 < 80 * std.time.ns_per_ms);
    _ = t0;
}

test "wait mode: a queue pair's index, number and units are what the card model checks, and an entry with a wrong lkey is refused (the model's checks can fail)" {
    // a WAIT entry whose lkey is the arena's, not the control block's: the doorbell word is not in that registration
    const f = try Fab.open(.{});
    defer f.close();
    cd.log_errors = false;
    defer cd.log_errors = true;
    const a = &f.nd[0];
    const q = &fx.qps[0];
    // hand-write a bad entry into node 0's first queue pair: a WAIT with a key that does not cover the doorbell
    const sq: [*]u8 = @ptrFromInt(q.sq);
    var w: [64]u8 = undefined;
    mx.waitWqe(&w, 0, q.qpn, .cyclic_smaller, false, 1000 + 0, @intFromPtr(&a.ctl.doorbell.v), 0, ~@as(u64, 0));
    // key 1000 is the first registration of node 0's rail A: its arena
    @memcpy(sq[0..64], &w);
    fx.lockIt();
    @atomicStore(u32, @as(*u32, @ptrFromInt(q.dbr + 4)), std.mem.nativeToBig(u32, 1), .release);
    fx.unlockIt();
    sleepNs(20 * std.time.ns_per_ms);
    fx.lockIt();
    defer fx.unlockIt();
    try std.testing.expect(fx.violations >= 1);
    try std.testing.expectEqual(@as(u8, 6), q.state);
}

test "wait mode: parseMac reads sysfs's text and refuses the rest" {
    try std.testing.expectEqualSlices(u8, &.{ 0x00, 0x11, 0x22, 0x33, 0x44, 0x55 }, &cd.parseMac("00:11:22:33:44:55\n").?);
    try std.testing.expect(cd.parseMac("00:11:22:33:44") == null);
    try std.testing.expect(cd.parseMac("00:11:22:33:44:55:66") == null);
    try std.testing.expect(cd.parseMac("zz:11:22:33:44:55") == null);
}

test "wait mode: the Settings struct's own default is the proxy path (the engine's default is roci_engine.waitMode's, the card's WAIT): the transport starts two proxies and no janitor, and the verbs Info says it does not wait" {
    const s: roci.Settings = .{};
    try std.testing.expect(!s.wait_qp);
    const i: rv.Info = .{};
    try std.testing.expectEqual(@as(u64, 0), i.wait);
    for (i.abort_qpn) |r| for (r) |x| try std.testing.expectEqual(@as(u32, 0), x);
    // the start line of the default path has no word of waiting
    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try roci.startLine(&w, 0, "mlx5_0,mlx5_2", 3, .{}, .write);
    try std.testing.expect(std.mem.indexOf(u8, w.buffered(), "waiting in the card") == null);
    var w2 = std.Io.Writer.fixed(&buf);
    try roci.startLine(&w2, 0, "mlx5_0,mlx5_2", 3, .{ .wait_qp = true }, .write);
    try std.testing.expect(std.mem.indexOf(u8, w2.buffered(), "waiting in the card (TF_DS_ROCI_WAIT, the default; =0 for the proxy threads") != null);
}

test "wait mode: a transport asked to wait in the card refuses a NIC that cannot (no pre-post), by name; the verbs NIC's Nic has none" {
    const ctl = try gpa.create(roci.Control);
    defer gpa.destroy(ctl);
    ctl.* = .{};
    const S = struct {
        fn post(_: *anyopaque, _: u8, _: u8, _: roci.Class, _: *const roci.Chain, _: u64) roci.PostError!void {}
        fn fail(_: *anyopaque) void {}
        fn reap(_: *anyopaque, _: u8) roci.PostError!void {}
    };
    var dummy: u8 = 0;
    const nic: roci.Nic = .{ .ctx = &dummy, .post_fn = S.post, .fail_fn = S.fail, .reap_fn = S.reap };
    try std.testing.expectError(error.NicCannotWait, roci.Transport.init(gpa, 0, roci.identity, ctl, @intFromPtr(ctl), nic, .{ .wait_qp = true }));
    const t = try roci.Transport.init(gpa, 0, roci.identity, ctl, @intFromPtr(ctl), nic, .{});
    t.deinit();
}

// -- the ATOMIC arrival signal ---------------------------------------------------------------------------------------

test "wait mode, ATOMIC signal: the queue pairs are set up for atomics (rae, atomic_mode IB_SPEC, depth 16 both ends, the RAE optional-parameter bit), the control block and a landing line are registered for them, and the write signal's queue pairs and registrations are what they were" {
    for ([_]roci.Signal{ .write, .atomic }) |sig| {
        const f = try Fab.open(.{ .signal = sig });
        defer f.close();
        const atomic = sig == .atomic;
        try std.testing.expectEqual(@as(u32, 24), fx.created_cd_master);
        for (fx.qps[0..24]) |q| {
            try std.testing.expectEqual(atomic, q.rae);
            try std.testing.expectEqual(@as(u8, if (atomic) 1 else 0), q.atomic_mode);
            try std.testing.expectEqual(@as(u8, if (atomic) 4 else 0), q.log_rra);
            try std.testing.expectEqual(@as(u8, if (atomic) 4 else 0), q.log_sra);
            try std.testing.expectEqual(@as(u32, 0), q.opt_masks[0]);
            try std.testing.expectEqual(@as(u32, if (atomic) @intCast(mx.opt_rae) else 0), q.opt_masks[1]);
            try std.testing.expectEqual(@as(u32, @intCast(mx.opt_min_rnr_nak | (if (atomic) mx.opt_rae else 0))), q.opt_masks[2]);
        }
        // registrations: arena + control block per rail and node (16), and with atomics one landing line each (24)
        try std.testing.expectEqual(@as(usize, if (atomic) 24 else 16), fx.n_registered_under_devx);
        var ctl_n: usize = 0;
        var sink_n: usize = 0;
        for (fx.mrs[0..fx.n_mr]) |m| {
            if (m.ctl) {
                ctl_n += 1;
                try std.testing.expectEqual(atomic, m.access & vabi.access_remote_atomic != 0);
                try std.testing.expect(m.access & vabi.access_remote_write != 0);
            } else {
                // the arena and the landing line grant no ATOMIC to anyone
                try std.testing.expect(m.access & vabi.access_remote_atomic == 0);
                if (m.len == roci.line * roci.rails * roci.world) {
                    sink_n += 1;
                    try std.testing.expectEqual(@as(c_int, vabi.access_local_write), m.access);
                }
            }
        }
        try std.testing.expectEqual(@as(usize, 8), ctl_n);
        try std.testing.expectEqual(@as(usize, if (atomic) 8 else 0), sink_n);
        // the Info a node gives its peers
        const info = try f.nd[1].nic.info();
        try std.testing.expectEqual(@as(u64, @backingInt(sig)), info.signal);
        try std.testing.expectEqual(@as(u32, if (atomic) 16 else 0), info.rd_atomic[0]);
        try std.testing.expectEqual(@as(u32, if (atomic) 16 else 0), info.rd_atomic[1]);
        try std.testing.expectEqual(@as(u64, 1), info.wait);
        try std.testing.expectEqual(@as(u32, 0), fx.violations);
    }
}

test "wait mode, ATOMIC signal: a chain's entries are [WAIT][data WRITE][ATOMIC fetch-and-add] in consecutive blocks, the ATOMIC at the peer's word of (class, source, rail) with the control block's key, landing in this node's own line; the queue's producer index counts blocks" {
    const f = try Fab.open(.{ .signal = .atomic });
    defer f.close();
    const a = &f.nd[0];
    const lo = roci.layout();
    pattern(a.arena[lo.at(.psum)..][0..4096], 11);
    _ = try a.t.reserve(.own1, 0);
    sleepNs(30 * std.time.ns_per_ms);
    // node 0's queue pair on rail 0 to node 1
    const peer: usize = 1;
    var q: ?*FQp = null;
    for (fx.qps[0..fx.n_qp]) |*c| {
        if (c.node != 0 or c.rail != 0) continue;
        if (qpByQpn(c.remote_qpn).?.node == peer) q = c;
    }
    const qp = q.?;
    const be = std.mem.readInt;
    const blk = struct {
        fn at(x: *FQp, i: usize) *const [64]u8 {
            return @ptrFromInt(x.sq + i * 64);
        }
    }.at;
    // [WAIT] opcode 0x0f, opmod 1, 3 units
    try std.testing.expectEqual(@as(u32, 0x0f), be(u32, blk(qp, 0)[0..4], .big) & 0xff);
    try std.testing.expectEqual(@as(u32, 3), be(u32, blk(qp, 0)[4..8], .big) & 0x3f);
    // [WRITE] opcode 0x08, 3 units
    try std.testing.expectEqual(@as(u32, 0x08), be(u32, blk(qp, 1)[0..4], .big) & 0xff);
    try std.testing.expectEqual(@as(u32, 3), be(u32, blk(qp, 1)[4..8], .big) & 0x3f);
    // [ATOMIC] opcode 0x12, 4 units, index 2, this queue pair's number
    const at = blk(qp, 2);
    try std.testing.expectEqual(@as(u32, 0x12), be(u32, at[0..4], .big) & 0xff);
    try std.testing.expectEqual(@as(u32, 2), (be(u32, at[0..4], .big) >> 8) & 0xffff);
    try std.testing.expectEqual(@as(u32, qp.qpn << 8 | 4), be(u32, at[4..8], .big));
    const cell = @offsetOf(roci.Control, "recv") + ((@as(usize, @backingInt(roci.Class.own1)) * roci.world + 0) * roci.rails + 0) * @sizeOf(roci.Word);
    try std.testing.expectEqual(@as(u64, fx.ctl_addr[peer] + cell), be(u64, at[16..24], .big));
    const rm = mrOfKey(be(u32, at[24..28], .big)).?;
    try std.testing.expect(rm.ctl and rm.node == peer and rm.rail == 0);
    try std.testing.expectEqual(@as(u64, 1), be(u64, at[32..40], .big));
    try std.testing.expectEqual(@as(u32, 8), be(u32, at[48..52], .big));
    const lm = mrOfKey(be(u32, at[52..56], .big)).?;
    try std.testing.expect(lm.node == 0 and lm.rail == 0 and lm.access == vabi.access_local_write);
    // the landing is node 0's line for (rail 0, peer 1): inside the sink, 8-byte aligned, a line of its own
    const land: usize = @intCast(be(u64, at[56..64], .big));
    try std.testing.expect(covers(lm, land, 8) and land % 8 == 0);
    try std.testing.expectEqual(@as(usize, lm.addr + (0 * roci.world + peer) * roci.line), land);
    // the producer index the card was told: three blocks (not 2 + the atomic's four units)
    try std.testing.expectEqual(@as(u32, 3), std.mem.bigToNative(u32, @as(*u32, @ptrFromInt(qp.dbr + 4)).*));
    try std.testing.expectEqual(@as(u32, 3), a.nic.rail[0].qps[peer].?.pi);
    // nothing ran: the WAIT is armed
    fx.lockIt();
    defer fx.unlockIt();
    try std.testing.expectEqual(@as(u64, 0), fx.atomics + fx.writes);
    try std.testing.expectEqual(@as(u32, 0), fx.violations);
}

test "wait mode, ATOMIC signal: the ATOMIC adds 1 to the peer's word after the data is placed, and the old value lands in the sender's line for (rail, peer)" {
    const f = try Fab.open(.{ .signal = .atomic });
    defer f.close();
    const a = &f.nd[0];
    const peer = &f.nd[1];
    const s: FakeStream = .{};
    const lo = roci.layout();
    pattern(a.arena[lo.at(.psum)..][0..4096], 21);
    _ = try a.t.reserve(.own1, 0);
    _ = try a.t.reserve(.own1, 0);
    try s.writeValue64(a.t.doorbellAddr(), a.t.go(2));
    const t0 = nowNs();
    const w0 = &peer.ctl.recv[@backingInt(roci.Class.own1)][0][0];
    while (@atomicLoad(u64, &w0.v, .acquire) < 2 or @atomicLoad(u64, &peer.ctl.recv[@backingInt(roci.Class.own1)][0][1].v, .acquire) < 2) {
        if (nowNs() - t0 > 3 * std.time.ns_per_s) return error.TestUnexpectedResult;
        sleepNs(100_000);
    }
    try std.testing.expectEqual(@as(u64, 2), @atomicLoad(u64, &w0.v, .acquire));
    fx.lockIt();
    defer fx.unlockIt();
    try std.testing.expectEqual(@as(u64, 4), fx.atomics); // 2 events x 2 rails
    try std.testing.expectEqual(@as(u64, 0), fx.words);
    try std.testing.expectEqual(@as(u32, 0), fx.violations);
    // the sender's lines hold the old values the card returned: the sources of the two events, 0 then 1 (the last one to land wins: it is 1)
    for (0..roci.rails) |r| {
        const line = f.nd[0].nic.sink[(r * roci.world + 1) * roci.line ..][0..8];
        try std.testing.expectEqual(@as(u64, 1), std.mem.readInt(u64, line, builtin.cpu.arch.endian()));
    }
}

test "wait mode, ATOMIC signal: the start refuses by name where the card does no ATOMICs or grants none in flight, and the write signal does not ask" {
    rv.log_refusals = false;
    defer rv.log_refusals = true;
    const cases = [_]struct { cap: c_int, rd: c_int }{
        .{ .cap = vabi.atomic_none, .rd = 16 }, // no ATOMIC operations
        .{ .cap = vabi.atomic_hca, .rd = 0 }, // none in flight
    };
    for (cases) |c| {
        fxReset();
        fx.atomic_cap = c.cap;
        fx.rd_atom = c.rd;
        var verbs = fakeVerbs();
        var mlx = fakeMlx();
        const arena = try std.posix.mmap(null, roci.layoutOf(.prompt).total, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        defer std.posix.munmap(arena);
        const ctl = try gpa.create(roci.Control);
        defer gpa.destroy(ctl);
        ctl.* = .{};
        fx.ctl_addr[0] = @intFromPtr(ctl);
        fx.cur_node = 0;
        try std.testing.expectError(error.AtomicUnsupported, cd.CdNic.open(gpa, &verbs, &mlx, .{ .mac_of = fakeMac, .signal = .atomic }, 0, arena, std.mem.asBytes(ctl), 77));
        // the write signal never asks the card about atomics
        const n = try cd.CdNic.open(gpa, &verbs, &mlx, .{ .mac_of = fakeMac, .signal = .write }, 0, arena, std.mem.asBytes(ctl), 77);
        n.close();
    }
}

test "wait mode, ATOMIC signal, control: a responder queue pair without rae (or a control block registered without ATOMIC access) gives an error completion on the first ATOMIC, and the transport fails (reason qp): the model's checks can fail" {
    const Case = enum { no_rae, no_mr_access };
    for ([_]Case{ .no_rae, .no_mr_access }) |c| {
        const f = try Fab.open(.{ .janitor_ms = 5, .signal = .atomic });
        defer f.close();
        cd.log_errors = false;
        defer cd.log_errors = true;
        const a = &f.nd[0];
        const s: FakeStream = .{};
        fx.lockIt();
        switch (c) {
            .no_rae => for (fx.qps[0..fx.n_qp]) |*q| {
                q.rae = false;
            },
            .no_mr_access => for (fx.mrs[0..fx.n_mr]) |*m| if (m.ctl) {
                m.access &= ~@as(c_int, vabi.access_remote_atomic);
            },
        }
        fx.unlockIt();
        try a.t.send(s, .own1, 0);
        const t0 = nowNs();
        while (a.t.failure() == null) {
            if (nowNs() - t0 > 3 * std.time.ns_per_s) return error.TestUnexpectedResult;
            sleepNs(1_000_000);
        }
        try std.testing.expectEqual(roci.FailReason.qp, a.t.failure().?.reason);
        fx.lockIt();
        defer fx.unlockIt();
        try std.testing.expect(fx.violations > 0 and fx.err_cqes > 0);
        try std.testing.expectEqual(@as(u64, 0), fx.atomics);
    }
}
