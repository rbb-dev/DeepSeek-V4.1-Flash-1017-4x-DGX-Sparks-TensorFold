//! Rocinante's NIC on ibverbs: one RC queue pair a peer on each of the two rails (RoCE v2), the arena and the control

const std = @import("std");
const fabric = @import("fabric");
const vabi = fabric.verbs_abi;
const Verbs = fabric.verbs.Verbs;
const roci = @import("rocinante.zig");

pub const Settings = struct {
    /// The two rails' devices (rail A, rail B).
    devices: [roci.rails][]const u8 = .{ "mlx5_0", "mlx5_2" },
    gid_index: u8 = 3,
    sq_depth: u32 = 512,
    signal_every: u32 = 16,
    /// The wait for the send queue to drain when it is full (a stalled peer), then the post fails.
    sq_wait_ms: u64 = 10_000,
};

/// What a node tells the others to connect: for each peer and rail its queue pair and PSN, per rail its GID, the
/// registered keys and bases of the arena and the control block, and the layout's hash.
pub const Info = extern struct {
    qpn: [roci.rails][roci.world]u32 = @splat(@splat(0)),
    psn: [roci.rails][roci.world]u32 = @splat(@splat(0)),
    gid: [roci.rails][16]u8 = @splat(@splat(0)),
    rkey_arena: [roci.rails]u32 = @splat(0),
    rkey_ctl: [roci.rails]u32 = @splat(0),
    arena_base: u64 = 0,
    ctl_base: u64 = 0,
    layout_hash: u64 = 0,
    /// Whether the node's queue pairs wait in the card (TF_DS_ROCI_WAIT, rocinante_cd.zig): 1, else 0 (this file's queue pairs). The nodes compare it at connect.
    wait: u64 = 0,
    /// The card-waiting backend's connection data, zero here: each rail's port MAC (RoCE v2's destination MAC for a queue pair made
    /// through DEVX) and the number of the queue pair that carries the abort notice to each peer.
    mac: [roci.rails][8]u8 = @splat(@splat(0)),
    abort_qpn: [roci.rails][roci.world]u32 = @splat(@splat(0)),
};

pub const Error = error{ VerbsFailed, PortDown, LayoutMismatch, SignalMismatch, DepthMismatch, NoGid, AtomicUnsupported, Misaligned, WaitMismatch };
    for (all) |a| if (a.wait != mine.wait) return error.WaitMismatch;

const Rail = struct {
    ctx: *vabi.Context,
    pd: *vabi.Pd,
    cq: *vabi.Cq,
    mr_arena: *vabi.Mr,
    mtu: c_int,
    qps: [roci.world]?*vabi.Qp = @splat(null),
    /// WRs posted a queue pair, and those a signaled completion has retired.
    posted: [roci.world]u64 = @splat(0),
    retired: [roci.world]u64 = @splat(0),
    chains: [roci.world]u32 = @splat(0),
    signalled: u64 = 0,
};

pub const VerbsNic = struct {
    gpa: std.mem.Allocator,
    v: *const Verbs,
    set: Settings,
    me: u8,
    arena: []u8,
    ctl: []u8,
    rail: [roci.rails]Rail = undefined,
    peers: [roci.world]Info = @splat(.{}),
    opened: bool = false,

    /// Open both rails, register `arena` and `ctl` on each, and create the queue pairs (RESET). `info` then describes
    /// this node; `connect` takes every node's.
        const n = try gpa.create(VerbsNic);
        errdefer gpa.destroy(n);
        const api = v.api;
        for (0..roci.rails) |r| {
            const ctx = try v.openDevice(set.devices[r]);
            const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
            const port = try v.port(ctx, 1);
            if (port.state != vabi.port_active) return error.PortDown;
            const cq = api.ibv_create_cq(ctx, @intCast(set.sq_depth * roci.world), null, null, 0) orelse return error.VerbsFailed;
            const access = vabi.access_local_write | vabi.access_remote_write;
            const mr_a = api.ibv_reg_mr(pd, arena.ptr, arena.len, access) orelse return error.VerbsFailed;
            for (0..roci.world) |p| {
                if (p == me) continue;
                var init: vabi.QpInitAttr = .{ .send_cq = cq, .recv_cq = cq, .cap = .{ .max_send_wr = set.sq_depth, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 16 }, .qp_type = vabi.qpt_rc };
                const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
                n.rail[r].qps[p] = qp;
            }
        }
        n.opened = true;
        return n;
    }

    pub fn info(n: *VerbsNic) !Info {
        for (0..roci.rails) |r| {
            const g = try n.v.gid(n.rail[r].ctx, 1, n.set.gid_index);
            i.gid[r] = g.raw;
            i.rkey_arena[r] = n.rail[r].mr_arena.rkey;
            for (n.rail[r].qps, 0..) |q, p| if (q) |qp| {
                i.qpn[r][p] = qp.qp_num;
                i.psn[r][p] = (qp.qp_num *% 2654435761) & 0xffffff;
            };
        }
        return i;
    }

    /// RESET to INIT to RTR to RTS on every queue pair against the peer's queue pair for this node.
    pub fn connect(n: *VerbsNic, all: *const [roci.world]Info) !void {
        const api = n.v.api;
        const m = vabi.mask;
        const mine = try n.info();
        n.peers = all.*;
        for (0..roci.rails) |r| for (0..roci.world) |p| {
            const qp = n.rail[r].qps[p] orelse continue;
            const theirs = all[p];
            if (api.ibv_modify_qp(qp, &a, m.state | m.pkey_index | m.port | m.access_flags) != 0) return error.VerbsFailed;
            a.ah_attr = .{ .grh = .{ .dgid = .{ .raw = theirs.gid[r] }, .sgid_index = n.set.gid_index, .hop_limit = 64 }, .is_global = 1, .port_num = 1 };
            a.min_rnr_timer = 12;
            if (api.ibv_modify_qp(qp, &a, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
            if (api.ibv_modify_qp(qp, &a, m.state | m.sq_psn | m.timeout | m.retry_cnt | m.rnr_retry | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
        };
    }

    pub fn close(n: *VerbsNic) void {
        const api = n.v.api;
        for (&n.rail) |*rl| {
            for (rl.qps) |q| if (q) |qp| {
                _ = api.ibv_destroy_qp(qp);
            };
            _ = api.ibv_dereg_mr(rl.mr_arena);
            _ = api.ibv_destroy_cq(rl.cq);
            _ = api.ibv_dealloc_pd(rl.pd);
            _ = api.ibv_close_device(rl.ctx);
        }
        n.gpa.destroy(n);
    }

    pub fn nic(n: *VerbsNic) roci.Nic {
        return .{ .ctx = n, .post_fn = postFn, .fail_fn = failFn, .reap_fn = reapFn };
    }

    /// Completions of rail `r`: a signaled one retires the WRs up to its id; an error one fails the transport.
    fn poll(n: *VerbsNic, r: usize) roci.PostError!void {
        const rl = &n.rail[r];
        var wc: [16]vabi.Wc = undefined;
        const got = rl.ctx.ops.poll_cq(rl.cq, 16, &wc);
        if (got < 0) return error.QpError;
        for (wc[0..@intCast(got)]) |w| {
            if (w.status != vabi.wc_success) return error.QpError;
            const peer: usize = @intCast(w.wr_id >> 56);
            const upto = w.wr_id & 0x00ff_ffff_ffff_ffff;
            if (peer < roci.world and upto > rl.retired[peer]) rl.retired[peer] = upto;
            rl.signalled += 1;
        }
    }

    fn postFn(ctx: *anyopaque, rail: u8, peer: u8, class: roci.Class, chain: *const roci.Chain, word: u64) roci.PostError!void {
        const n: *VerbsNic = @ptrCast(@alignCast(ctx));
        const rl = &n.rail[rail];
        const wrs = chain.n + 1;
        var spins: u64 = 0;
        while (rl.posted[peer] - rl.retired[peer] + wrs + n.set.signal_every * 8 >= n.set.sq_depth) : (spins += 1) {
            try n.poll(rail);
            if (spins > n.set.sq_wait_ms * 100_000) return error.PostFailed;
        }
        var sges: [roci.max_pieces + 1]vabi.Sge = undefined;
        var list: [roci.max_pieces + 1]vabi.SendWr = undefined;
        rl.chains[peer] += 1;
        const signal = rl.chains[peer] % n.set.signal_every == 0;
        const posted_after = rl.posted[peer] + wrs;
        var bad: ?*vabi.SendWr = null;
        if (rl.ctx.ops.post_send(qp, &list[0], &bad) != 0) return error.PostFailed;
        rl.posted[peer] = posted_after;
        if (!signal and rl.chains[peer] % 4 == 0) try n.poll(rail);
    }

    /// The failure path: an inline WRITE of the abort word to every peer on every rail (best effort, completions polled
    /// for a bounded time), then every queue pair to ERR so no late genuine word can overwrite the abort value.
    fn failFn(ctx: *anyopaque) void {
        const n: *VerbsNic = @ptrCast(@alignCast(ctx));
        var one: u64 = 1;
        for (0..roci.rails) |r| for (0..roci.world) |p| {
            const qp = n.rail[r].qps[p] orelse continue;
            var sge: vabi.Sge = .{ .addr = @intFromPtr(&one), .length = 8, .lkey = 0 };
            var wr: vabi.SendWr = .{ .sg_list = @ptrCast(&sge), .num_sge = 1, .opcode = vabi.wr_rdma_write, .send_flags = vabi.send_inline, .remote_addr = n.peers[p].ctl_base + @offsetOf(roci.Control, "abort"), .rkey = n.peers[p].rkey_ctl[r] };
            var bad: ?*vabi.SendWr = null;
            _ = n.rail[r].ctx.ops.post_send(qp, &wr, &bad);
        };
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 10 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
        for (0..roci.rails) |r| for (n.rail[r].qps) |q| if (q) |qp| {
            var a: vabi.QpAttr = .{ .qp_state = vabi.qps_err };
            _ = n.v.api.ibv_modify_qp(qp, &a, vabi.mask.state);
        };
    }

    fn reapFn(ctx: *anyopaque, rail: u8) roci.PostError!void {
        const n: *VerbsNic = @ptrCast(@alignCast(ctx));
        try n.poll(rail);
    }
};

comptime {
    // the Info a node sends is a plain extern struct the start-up exchange can carry as bytes
    std.debug.assert(@sizeOf(Info) % 8 == 0);
}