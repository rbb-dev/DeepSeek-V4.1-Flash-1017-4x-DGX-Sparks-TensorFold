//! Rocinante's NIC on ibverbs: one RC queue pair a peer on each of the two rails (RoCE v2), the arena and the control
//! block registered once per rail over the same pages, a chain posted as one linked list of RDMA WRITEs ending in the
//! arrival signal: an inline 8-byte generation word (a WRITE), or an RDMA ATOMIC fetch-and-add of 1 on the word
//! (`Settings.signal`). Data WRs and signals are unsignaled but for one chain in `signal_every` a queue pair, which
//! reclaims the send queue; an error completion fails the transport.

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
    /// The arrival signal (TF_DS_ROCI_SIGNAL). Every node must run the same one (compared at connect).
    signal: roci.Signal = roci.default_signal,
    /// The most ATOMICs a queue pair keeps in flight (its requester's and responder's depth), capped by what the card grants.
    atomic_depth: u8 = 16,
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
    /// The node's arrival signal (a roci.Signal): the nodes compare it at connect.
    signal: u64 = 0,
    /// Whether the node's queue pairs wait in the card (TF_DS_ROCI_WAIT, rocinante_cd.zig): 1, else 0 (this file's queue pairs). The nodes compare it at connect.
    wait: u64 = 0,
    /// The card-waiting backend's connection data, zero here: each rail's port MAC (RoCE v2's destination MAC for a queue pair made
    /// through DEVX) and the number of the queue pair that carries the abort notice to each peer.
    mac: [roci.rails][8]u8 = @splat(@splat(0)),
    abort_qpn: [roci.rails][roci.world]u32 = @splat(@splat(0)),
};

pub const Error = error{ VerbsFailed, PortDown, LayoutMismatch, SignalMismatch, DepthMismatch, NoGid, AtomicUnsupported, Misaligned, WaitMismatch };

/// Every node's layout and arrival signal equal this node's: a node that signals with a WRITE into a control block registered for ATOMICs
/// (or the other way round) would fail at its first post, so the start refuses by name.
pub fn checkPeers(all: *const [roci.world]Info, mine: Info) Error!void {
    for (all) |a| if (a.layout_hash != mine.layout_hash) return error.LayoutMismatch;
    for (all) |a| if (a.signal != mine.signal) return error.SignalMismatch;
    for (all) |a| if (a.wait != mine.wait) return error.WaitMismatch;
}

/// What a rail's registrations and queue pairs grant a peer's NIC: WRITE always, and with the atomic signal ATOMIC on the control block
/// (the words the atomics add to) and on the queue pair's responder.
pub fn remoteAccess(signal: roci.Signal) c_int {
    return vabi.access_local_write | vabi.access_remote_write | (if (signal == .atomic) @as(c_int, vabi.access_remote_atomic) else 0);
}

/// The atomic signal's start-up check, per rail: why the card cannot run it, or null. It must do ATOMICs at all (atomic_cap above none) and grant
/// a queue pair at least one in flight. Atomic_cap `hca` (the CX-7's) is enough: every writer of a word is a card's atomic until the card is
/// fenced off (Nic.fail_fn). The WRITE signal asks nothing. The engine refuses to start, naming the reason: a setting its card cannot run.
pub fn signalRefusal(buf: []u8, signal: roci.Signal, device: []const u8, atomic_cap: c_int, max_qp_rd_atom: c_int, max_qp_init_rd_atom: c_int) ?[]const u8 {
    if (signal != .atomic) return null;
    if (atomic_cap < vabi.atomic_hca)
        return std.fmt.bufPrint(buf, "TF_DS_ROCI_SIGNAL=atomic: device {s} reports atomic_cap {d} (no ATOMIC operations): the arrival signal stays a WRITE (TF_DS_ROCI_SIGNAL=write)", .{ device, atomic_cap }) catch "TF_DS_ROCI_SIGNAL=atomic: the device does no ATOMIC operations";
    if (max_qp_rd_atom < 1 or max_qp_init_rd_atom < 1)
        return std.fmt.bufPrint(buf, "TF_DS_ROCI_SIGNAL=atomic: device {s} grants a queue pair {d} responder and {d} requester ATOMICs in flight: it needs at least one each", .{ device, max_qp_rd_atom, max_qp_init_rd_atom }) catch "TF_DS_ROCI_SIGNAL=atomic: the device grants no ATOMICs in flight";
    return null;
}

/// The ATOMICs in flight a queue pair is set up for (max_dest_rd_atomic at RTR, max_rd_atomic at RTS): the setting, capped by both limits of the
/// card; 0 for the WRITE signal, which has none.
pub fn atomicDepth(signal: roci.Signal, want: u8, max_qp_rd_atom: c_int, max_qp_init_rd_atom: c_int) u8 {
    if (signal != .atomic) return 0;
    const cap: c_int = @min(max_qp_rd_atom, max_qp_init_rd_atom, @as(c_int, want));
    return @intCast(@max(cap, 1));
}

const Rail = struct {
    ctx: *vabi.Context,
    pd: *vabi.Pd,
    cq: *vabi.Cq,
    mr_arena: *vabi.Mr,
    mr_ctl: ?*vabi.Mr, // null once fenced off (failFn): the card can no longer touch the control block
    /// Where the card puts the old value an ATOMIC returns (the atomic signal only): one word a peer, local write.
    mr_sink: ?*vabi.Mr = null,
    mtu: c_int,
    /// The ATOMICs in flight a queue pair is set up for (0: the WRITE signal).
    rd_atomic: u8 = 0,
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
    /// The returned old values of the ATOMICs: one 64-byte line a (rail, peer), 64-byte aligned, registered per rail (atomic signal only).
    sink: []align(64) u8 = &.{},
    rail: [roci.rails]Rail = undefined,
    peers: [roci.world]Info = @splat(.{}),
    opened: bool = false,

    /// Open both rails, register `arena` and `ctl` on each, and create the queue pairs (RESET). `info` then describes
    /// this node; `connect` takes every node's.
        const n = try gpa.create(VerbsNic);
        errdefer gpa.destroy(n);
        // a word the card updates with an ATOMIC is 8-byte aligned: the control block starts on a line, its words are lines (comptime, below)
        if (@intFromPtr(ctl.ptr) % roci.line != 0) return error.Misaligned;
        if (set.signal == .atomic) {
            n.sink = try gpa.alignedAlloc(u8, .@"64", roci.line * roci.rails * roci.world);
            @memset(n.sink, 0);
        }
        errdefer if (n.sink.len != 0) gpa.free(n.sink);
        const api = v.api;
        for (0..roci.rails) |r| {
            const ctx = try v.openDevice(set.devices[r]);
            const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
            const port = try v.port(ctx, 1);
            if (port.state != vabi.port_active) return error.PortDown;
            const cq = api.ibv_create_cq(ctx, @intCast(set.sq_depth * roci.world), null, null, 0) orelse return error.VerbsFailed;
            const dev = try v.device(ctx);
            var why: [256]u8 = undefined;
            if (signalRefusal(&why, set.signal, set.devices[r], dev.atomic_cap, dev.max_qp_rd_atom, dev.max_qp_init_rd_atom)) |text| {
                return error.AtomicUnsupported;
            }
            // the arena takes WRITEs only; the control block's words are what the signal writes or adds to
            const access = vabi.access_local_write | vabi.access_remote_write;
            const mr_a = api.ibv_reg_mr(pd, arena.ptr, arena.len, access) orelse return error.VerbsFailed;
            const mr_c = api.ibv_reg_mr(pd, ctl.ptr, ctl.len, remoteAccess(set.signal)) orelse return error.VerbsFailed;
            const mr_s: ?*vabi.Mr = if (set.signal == .atomic) (api.ibv_reg_mr(pd, n.sink.ptr, n.sink.len, vabi.access_local_write) orelse return error.VerbsFailed) else null;
            n.rail[r] = .{ .ctx = ctx, .pd = pd, .cq = cq, .mr_arena = mr_a, .mr_ctl = mr_c, .mr_sink = mr_s, .mtu = @min(port.active_mtu, 5), .rd_atomic = atomicDepth(set.signal, set.atomic_depth, dev.max_qp_rd_atom, dev.max_qp_init_rd_atom) };
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
        var i: Info = .{ .arena_base = @intFromPtr(n.arena.ptr), .ctl_base = @intFromPtr(n.ctl.ptr), .layout_hash = n.layout_hash, .signal = @backingInt(n.set.signal) };
        for (0..roci.rails) |r| {
            const g = try n.v.gid(n.rail[r].ctx, 1, n.set.gid_index);
            i.gid[r] = g.raw;
            i.rkey_arena[r] = n.rail[r].mr_arena.rkey;
            i.rkey_ctl[r] = n.rail[r].mr_ctl.?.rkey;
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
        try checkPeers(all, mine);
        n.peers = all.*;
        for (0..roci.rails) |r| for (0..roci.world) |p| {
            const qp = n.rail[r].qps[p] orelse continue;
            const theirs = all[p];
            var a: vabi.QpAttr = .{ .qp_state = vabi.qps_init, .port_num = 1, .qp_access_flags = @intCast(remoteAccess(n.set.signal)) };
            if (api.ibv_modify_qp(qp, &a, m.state | m.pkey_index | m.port | m.access_flags) != 0) return error.VerbsFailed;
            a = .{ .qp_state = vabi.qps_rtr, .path_mtu = n.rail[r].mtu, .dest_qp_num = theirs.qpn[r][n.me], .rq_psn = theirs.psn[r][n.me], .max_dest_rd_atomic = n.rail[r].rd_atomic };
            a.ah_attr = .{ .grh = .{ .dgid = .{ .raw = theirs.gid[r] }, .sgid_index = n.set.gid_index, .hop_limit = 64 }, .is_global = 1, .port_num = 1 };
            a.min_rnr_timer = 12;
            if (api.ibv_modify_qp(qp, &a, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
            a = .{ .qp_state = vabi.qps_rts, .sq_psn = mine.psn[r][p], .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .max_rd_atomic = n.rail[r].rd_atomic };
            if (api.ibv_modify_qp(qp, &a, m.state | m.sq_psn | m.timeout | m.retry_cnt | m.rnr_retry | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
        };
    }

    pub fn close(n: *VerbsNic) void {
        const api = n.v.api;
        for (&n.rail) |*rl| {
            for (rl.qps) |q| if (q) |qp| {
                _ = api.ibv_destroy_qp(qp);
            };
            if (rl.mr_sink) |m| _ = api.ibv_dereg_mr(m);
            if (rl.mr_ctl) |m| _ = api.ibv_dereg_mr(m);
            _ = api.ibv_dereg_mr(rl.mr_arena);
            _ = api.ibv_destroy_cq(rl.cq);
            _ = api.ibv_dealloc_pd(rl.pd);
            _ = api.ibv_close_device(rl.ctx);
        }
        if (n.sink.len != 0) n.gpa.free(n.sink);
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

    /// The work requests of one chain on `rail` to `peer`: its data WRITEs, then the arrival signal, linked in order on the one queue pair.
    /// The signal is an inline WRITE of `word` (the generation), or an ATOMIC fetch-and-add of 1 on the same word of the peer's control
    /// block, its returned old value going to this node's sink line for the (rail, peer): either rides behind the data on the same queue pair.
    /// `word` must outlive the post (an inline WR copies it when it is posted). Returns the number of WRs.
    fn buildWrs(n: *const VerbsNic, rail: u8, peer: u8, class: roci.Class, chain: *const roci.Chain, word: *const u64, signaled: bool, wr_id: u64, sges: *[roci.max_pieces + 1]vabi.Sge, list: *[roci.max_pieces + 1]vabi.SendWr) usize {
        const rl = &n.rail[rail];
        const theirs = &n.peers[peer];
        for (chain.slice(), 0..) |p, i| {
            sges[i] = .{ .addr = @intFromPtr(n.arena.ptr) + p.src, .length = @intCast(p.len), .lkey = rl.mr_arena.lkey };
            list[i] = .{ .sg_list = @ptrCast(&sges[i]), .num_sge = 1, .opcode = vabi.wr_rdma_write, .send_flags = 0, .remote_addr = theirs.arena_base + p.dst, .rkey = theirs.rkey_arena[rail] };
        }
        const last = chain.n;
        const word_cell = @offsetOf(roci.Control, "recv") + ((@as(usize, @backingInt(class)) * roci.world + n.me) * roci.rails + rail) * @sizeOf(roci.Word);
        const flags: c_uint = if (signaled) vabi.send_signaled else 0;
        switch (n.set.signal) {
            .write => {
                sges[last] = .{ .addr = @intFromPtr(word), .length = 8, .lkey = 0 };
                list[last] = .{ .wr_id = wr_id, .sg_list = @ptrCast(&sges[last]), .num_sge = 1, .opcode = vabi.wr_rdma_write, .send_flags = vabi.send_inline | flags, .remote_addr = theirs.ctl_base + word_cell, .rkey = theirs.rkey_ctl[rail] };
            },
            .atomic => {
                sges[last] = .{ .addr = @intFromPtr(n.sink.ptr) + (@as(usize, rail) * roci.world + peer) * roci.line, .length = 8, .lkey = rl.mr_sink.?.lkey };
                list[last] = vabi.fetchAddWr(@ptrCast(&sges[last]), theirs.ctl_base + word_cell, 1, theirs.rkey_ctl[rail], wr_id);
                list[last].send_flags = flags;
            },
        }
        for (0..last) |i| list[i].next = &list[i + 1];
        return last + 1;
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
        const w = word;
        rl.chains[peer] += 1;
        const signal = rl.chains[peer] % n.set.signal_every == 0;
        const posted_after = rl.posted[peer] + wrs;
        _ = n.buildWrs(rail, peer, class, chain, &w, signal, (@as(u64, peer) << 56) | posted_after, &sges, &list);
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
        n.fence();
    }

    fn fence(n: *VerbsNic) void {
        var held = false;
            };
        if (held) {
            std.log.err("ROCI: the control block's registration could not be released at a failure: the card is given 100 ms to finish what it has begun", .{});
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
            _ = std.c.nanosleep(&ts, null);
        }
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

/// Whether a NIC accepts a block of memory as a memory region, with the access the arena is registered with: the device opened once,
/// one protection domain, `try_` registers and at once deregisters.
pub const RegProbe = struct {
    v: Verbs,
    ctx: *vabi.Context,
    pd: *vabi.Pd,

    pub fn open(device: []const u8) !RegProbe {
        var v = try Verbs.open(null);
        errdefer v.close();
        const ctx = try v.openDevice(device);
        errdefer _ = v.api.ibv_close_device(ctx);
        const pd = v.api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        return .{ .v = v, .ctx = ctx, .pd = pd };
    }

    pub fn close(p: *RegProbe) void {
        _ = p.v.api.ibv_dealloc_pd(p.pd);
        _ = p.v.api.ibv_close_device(p.ctx);
        p.v.close();
    }

    /// Null when ibv_reg_mr accepted [ptr, ptr + len) (and it is deregistered again), else the errno it left.
    pub fn refused(p: *const RegProbe, ptr: *anyopaque, len: usize) ?c_int {
        std.c._errno().* = 0;
        const mr = p.v.api.ibv_reg_mr(p.pd, ptr, len, vabi.access_local_write | vabi.access_remote_write) orelse return std.c._errno().*;
        _ = p.v.api.ibv_dereg_mr(mr);
        return null;
    }
};

test "the atomic signal's start-up check refuses by name a card without ATOMICs or without room for one in flight; the WRITE signal asks nothing" {
    var buf: [256]u8 = undefined;
    // the CX-7's answers: atomic_cap 1 (the card's own atomics), 16 responder / 16 requester in flight
    try std.testing.expectEqual(@as(?[]const u8, null), signalRefusal(&buf, .atomic, "mlx5_0", vabi.atomic_hca, 16, 16));
    try std.testing.expectEqual(@as(?[]const u8, null), signalRefusal(&buf, .atomic, "mlx5_2", vabi.atomic_glob, 16, 16));
    const none = signalRefusal(&buf, .atomic, "mlx5_2", vabi.atomic_none, 16, 16).?;
    try std.testing.expect(std.mem.indexOf(u8, none, "TF_DS_ROCI_SIGNAL=atomic") != null and std.mem.indexOf(u8, none, "mlx5_2") != null and std.mem.indexOf(u8, none, "atomic_cap 0") != null);
    const resp = signalRefusal(&buf, .atomic, "mlx5_0", vabi.atomic_hca, 0, 16).?;
    try std.testing.expect(std.mem.indexOf(u8, resp, "mlx5_0") != null and std.mem.indexOf(u8, resp, "0 responder and 16 requester") != null);
    const req = signalRefusal(&buf, .atomic, "mlx5_0", vabi.atomic_hca, 16, 0).?;
    try std.testing.expect(std.mem.indexOf(u8, req, "16 responder and 0 requester") != null);
    // the WRITE signal runs on a card that reports none
    try std.testing.expectEqual(@as(?[]const u8, null), signalRefusal(&buf, .write, "mlx5_0", vabi.atomic_none, 0, 0));
}

test "queue pairs are set up for the ATOMICs in flight the card grants, none for the WRITE signal" {
    try std.testing.expectEqual(@as(u8, 0), atomicDepth(.write, 16, 16, 16));
    try std.testing.expectEqual(@as(u8, 16), atomicDepth(.atomic, 16, 16, 16));
    try std.testing.expectEqual(@as(u8, 16), atomicDepth(.atomic, 16, 128, 128));
    try std.testing.expectEqual(@as(u8, 4), atomicDepth(.atomic, 16, 4, 128));
    try std.testing.expectEqual(@as(u8, 2), atomicDepth(.atomic, 16, 128, 2));
    try std.testing.expectEqual(@as(u8, 8), atomicDepth(.atomic, 8, 128, 128));
    try std.testing.expectEqual(@as(u8, 1), atomicDepth(.atomic, 0, 128, 128));
}

test "the control block grants a peer's NIC remote ATOMIC access only for the atomic signal; the arena never" {
    try std.testing.expectEqual(@as(c_int, vabi.access_local_write | vabi.access_remote_write), remoteAccess(.write));
    try std.testing.expectEqual(@as(c_int, vabi.access_local_write | vabi.access_remote_write | vabi.access_remote_atomic), remoteAccess(.atomic));
    try std.testing.expect(remoteAccess(.write) & vabi.access_remote_atomic == 0);
}

test "every word an ATOMIC adds to is 8-byte aligned: the control block starts on a line and each word is a line of its own" {
    comptime {
        std.debug.assert(@alignOf(roci.Control) >= roci.line and @offsetOf(roci.Control, "recv") % roci.line == 0 and @sizeOf(roci.Word) == roci.line);
    }
    var seen: usize = 0;
    for (0..roci.n_classes) |c| for (0..roci.world) |src| for (0..roci.rails) |r| {
        const off = @offsetOf(roci.Control, "recv") + ((c * roci.world + src) * roci.rails + r) * @sizeOf(roci.Word);
        try std.testing.expectEqual(@as(usize, 0), off % 8);
        seen += 1;
    };
    try std.testing.expectEqual(roci.n_classes * roci.world * roci.rails, seen);
}

test "nodes with different arrival signals or layouts refuse to connect, by name" {
    var all: [roci.world]Info = @splat(.{ .layout_hash = 7, .signal = 1 });
    const mine = all[0];
    try checkPeers(&all, mine);
    all[2].signal = 0;
    try std.testing.expectError(error.SignalMismatch, checkPeers(&all, mine));
    all[2].signal = 1;
    all[3].layout_hash = 8;
    try std.testing.expectError(error.LayoutMismatch, checkPeers(&all, mine));
}

/// A VerbsNic with no card behind it: what `buildWrs` reads (the arena, the sink, the keys, the peers' bases and keys).
fn fakeNic(signal: roci.Signal, arena: []u8, sink: []align(64) u8, mr_arena: *vabi.Mr, mr_sink: *vabi.Mr) VerbsNic {
    var n: VerbsNic = .{ .gpa = std.testing.allocator, .v = undefined, .set = .{ .signal = signal }, .me = 2, .arena = arena, .ctl = &.{}, .sink = sink };
    for (&n.rail) |*rl| rl.* = .{ .ctx = undefined, .pd = undefined, .cq = undefined, .mr_arena = mr_arena, .mr_ctl = null, .mr_sink = if (signal == .atomic) mr_sink else null, .mtu = 5 };
    for (&n.peers, 0..) |*p, i| p.* = .{ .arena_base = 0x1000_0000 * (i + 1), .ctl_base = 0x2000_0000 * (i + 1), .rkey_arena = .{ 100 + @as(u32, @intCast(i)), 200 + @as(u32, @intCast(i)) }, .rkey_ctl = .{ 300 + @as(u32, @intCast(i)), 400 + @as(u32, @intCast(i)) } };
    return n;
}

test "a chain's work requests: the data WRITEs, then the signal on the same queue pair (an inline WRITE of the word, or an ATOMIC fetch-and-add of 1 on the word's cell with its old value to the sink)" {
    var arena: [4096]u8 align(64) = undefined;
    var sink: [roci.line * roci.rails * roci.world]u8 align(64) = undefined;
    var mr_a: vabi.Mr = undefined;
    mr_a.lkey = 11;
    var mr_s: vabi.Mr = undefined;
    mr_s.lkey = 22;
    var chain: roci.Chain = .{};
    chain.pieces[0] = .{ .src = 64, .dst = 4096, .len = 1024 };
    chain.pieces[1] = .{ .src = 2048, .dst = 8192, .len = 512 };
    chain.n = 2;
    const class = roci.Class.wob;
    const rail: u8 = 1;
    const peer: u8 = 3;
    const cell = @offsetOf(roci.Control, "recv") + ((@as(usize, @backingInt(class)) * roci.world + 2) * roci.rails + rail) * @sizeOf(roci.Word);
    var sges: [roci.max_pieces + 1]vabi.Sge = undefined;
    var list: [roci.max_pieces + 1]vabi.SendWr = undefined;
    const word: u64 = 41;
    for ([_]roci.Signal{ .write, .atomic }) |signal| {
        const n = fakeNic(signal, &arena, &sink, &mr_a, &mr_s);
        const count = n.buildWrs(rail, peer, class, &chain, &word, true, 0x77, &sges, &list);
        try std.testing.expectEqual(@as(usize, 3), count);
        // the data: plain WRITEs from the arena at the piece's source to the peer's arena at its destination, unsignaled, linked in order
        for (chain.slice(), 0..) |p, i| {
            try std.testing.expectEqual(@as(c_int, vabi.wr_rdma_write), list[i].opcode);
            try std.testing.expectEqual(@as(c_uint, 0), list[i].send_flags);
            try std.testing.expectEqual(n.peers[peer].arena_base + p.dst, list[i].remote_addr);
            try std.testing.expectEqual(n.peers[peer].rkey_arena[rail], list[i].rkey);
            try std.testing.expectEqual(@intFromPtr(&arena) + p.src, sges[i].addr);
            try std.testing.expectEqual(@as(u32, 11), sges[i].lkey);
            try std.testing.expectEqual(@intFromPtr(&list[i + 1]), @intFromPtr(list[i].next.?));
        }
        // the signal is the last WR of the same list, signaled (the caller asked), and the end of it
        const last = &list[2];
        try std.testing.expect(last.next == null);
        try std.testing.expectEqual(@as(u64, 0x77), last.wr_id);
        try std.testing.expect(last.send_flags & vabi.send_signaled != 0);
        try std.testing.expectEqual(n.peers[peer].ctl_base + cell, last.remote_addr);
        switch (signal) {
            .write => {
                try std.testing.expectEqual(@as(c_int, vabi.wr_rdma_write), last.opcode);
                try std.testing.expect(last.send_flags & vabi.send_inline != 0);
                try std.testing.expectEqual(n.peers[peer].rkey_ctl[rail], last.rkey);
                try std.testing.expectEqual(@intFromPtr(&word), sges[2].addr);
                try std.testing.expectEqual(@as(u32, 8), sges[2].length);
            },
            .atomic => {
                try std.testing.expectEqual(@as(c_int, vabi.wr_atomic_fetch_add), last.opcode);
                try std.testing.expect(last.send_flags & vabi.send_inline == 0); // an atomic is never inline
                const raw = std.mem.asBytes(last);
                try std.testing.expectEqual(@as(u64, 1), std.mem.readInt(u64, raw[48..56], .little)); // the addend
                try std.testing.expectEqual(n.peers[peer].rkey_ctl[rail], std.mem.readInt(u32, raw[64..68], .little));
                // the old value goes to this node's own line for the (rail, peer), local write, 8 bytes, 8-byte aligned
                try std.testing.expectEqual(@intFromPtr(&sink) + (@as(usize, rail) * roci.world + peer) * roci.line, sges[2].addr);
                try std.testing.expectEqual(@as(u32, 8), sges[2].length);
                try std.testing.expectEqual(@as(u32, 22), sges[2].lkey);
                try std.testing.expectEqual(@as(u64, 0), sges[2].addr % 8);
            },
        }
    }
    // an unsignaled chain's signal carries no completion request
    const n = fakeNic(.atomic, &arena, &sink, &mr_a, &mr_s);
    _ = n.buildWrs(0, 1, class, &chain, &word, false, 1, &sges, &list);
    try std.testing.expectEqual(@as(c_uint, 0), list[2].send_flags);
}