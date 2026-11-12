//! The ConnectX's DEVX interface (libmlx5's `mlx5dv_*`), as far as Rocinante's Core Direct queue pairs need it: the entry
//! points by dlopen, the layout of the device commands that create and connect a queue pair, the work queue entries a
//! host writes by hand (WAIT-on-data, RDMA WRITE, inline RDMA WRITE) and the completion entries it reads by hand.
//!
//! Third-party layouts, copied as facts of the device interface and kept with their notices:
//!  * the command layouts (`qpc` and the create/modify/query messages): NVIDIA DOCA GPUNetIO, `include/host/mlx5_ifc.h`
//!    (BSD-3-Clause, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES), the line of each field beside it, and the
//!    transitions' field order from its `src/doca_verbs_qp.cpp` (create_qp_obj, rst2init, init2rtr, rtr2rts);
//!  * the WAIT segment (`struct mlx5_wqe_wseg`: operation, lkey, va_high, va_low, value, mask) and its condition codes:
//!    DPDK `drivers/common/mlx5/mlx5_prm.h` (SPDX BSD-3-Clause, Copyright 2016 6WIND S.A., Copyright 2016 Mellanox Technologies,
//!    Ltd) and `drivers/net/mlx5/mlx5_tx.h` (the same licence, Copyright 2021 6WIND S.A., Copyright 2021 Mellanox Technologies, Ltd);
//!  * the control, remote-address and data segments, the CQE and the doorbell record index: rdma-core
//!    `providers/mlx5/mlx5dv.h` (GPL-2.0 or the OpenIB.org BSD licence, Copyright (c) 2017 Mellanox Technologies, Inc.) and `qp.c`'s inline segment.
//! The licence texts and the copyright lines are in LICENSES/ and THIRD_PARTY_NOTICES.md. NVIDIA's DOCA GPUNetIO is not linked
//! or called: only the field layout is used.
//!
//! Every function here is pure or takes the table of entry points as an argument, so the host tests drive it with a
//! table of their own and decode the bytes it writes with a model of the card.

const std = @import("std");
const builtin = @import("builtin");
const fabric = @import("fabric");
const vabi = fabric.verbs_abi;

// -- the device command's bit fields ---------------------------------------------------------------------------------

/// A field of a device command: its bit offset from the start of the message (or of the `qpc`) and its width, counted
/// in the big-endian bit order the interface uses (bit 0 is the top bit of the first byte).
pub const Field = struct { off: u16, bits: u8 };

/// Stores `v` in the field: a 64-bit field is a whole big-endian quad word at a multiple of 64 bits (DEVX_SET64), any other
/// lies inside one big-endian double word (DEVX_SET, `_devx_set` in mlx5dv.h:1971-1982).
pub fn put(buf: []u8, f: Field, v: u64) void {
    if (f.bits == 64) {
        std.debug.assert(f.off % 64 == 0);
        std.mem.writeInt(u64, buf[f.off / 8 ..][0..8], v, .big);
        return;
    }
    const dw = f.off / 32;
    std.debug.assert(f.bits >= 1 and f.bits <= 32 and (f.off & 31) + f.bits <= 32);
    const shift: u6 = @intCast(32 - f.bits - (f.off & 31));
    const mask: u64 = (@as(u64, 1) << @intCast(f.bits)) - 1;
    var w: u64 = std.mem.readInt(u32, buf[dw * 4 ..][0..4], .big);
    w = (w & ~(mask << shift)) | ((v & mask) << shift);
    std.mem.writeInt(u32, buf[dw * 4 ..][0..4], @intCast(w), .big);
}

/// The value of the field (DEVX_GET, DEVX_GET64).
pub fn get(buf: []const u8, f: Field) u64 {
    if (f.bits == 64) return std.mem.readInt(u64, buf[f.off / 8 ..][0..8], .big);
    const dw = f.off / 32;
    const shift: u6 = @intCast(32 - f.bits - (f.off & 31));
    const mask: u64 = (@as(u64, 1) << @intCast(f.bits)) - 1;
    return (@as(u64, std.mem.readInt(u32, buf[dw * 4 ..][0..4], .big)) >> shift) & mask;
}

/// The command opcodes (mlx5_ifc.h:77-85).
pub const op = struct {
    pub const create_qp: u16 = 0x500;
    pub const rst2init_qp: u16 = 0x502;
    pub const init2rtr_qp: u16 = 0x503;
    pub const rtr2rts_qp: u16 = 0x504;
    pub const qp_2err: u16 = 0x507;
    pub const query_qp: u16 = 0x50b;
};

/// mlx5_prm.h: MLX5_QPC_RQ_TYPE_ZERO_SIZE_RQ; mlx5_ifc.h:3386 MLX5_QPC_ST_RC; :3390 MLX5_QPC_PM_STATE_MIGRATED; mlx5_prm.h MLX5_QPC_MTU_4K_BYTES.
pub const rq_type_zero_size: u64 = 0x3;
pub const st_rc: u64 = 0x0;
pub const pm_state_migrated: u64 = 0x3;
pub const mtu_4k: u64 = 0x5;
/// mlx5_prm.h MLX5_QPC_STATE_*: the states a QUERY_QP reports.
pub const qp_state = struct {
    pub const rtr: u64 = 2;
    pub const rts: u64 = 3;
    pub const err: u64 = 6;
};

/// The `qpc` (queue pair context) of mlx5_ifc.h:3457 (1,856 bits): each field's offset from the start of the `qpc`.
pub const qpc = struct {
    pub const state: Field = .{ .off = 0, .bits = 4 }; // mlx5_ifc.h:3458
    pub const st: Field = .{ .off = 8, .bits = 8 }; // :3460
    pub const pm_state: Field = .{ .off = 19, .bits = 2 }; // :3463
    pub const pd: Field = .{ .off = 40, .bits = 24 }; // :3478
    pub const uar_page: Field = .{ .off = 104, .bits = 24 }; // :3494
    pub const cqn_snd: Field = .{ .off = 1000, .bits = 24 }; // :3526
    pub const cqn_rcv: Field = .{ .off = 1256, .bits = 24 }; // :3562
    pub const log_sq_size: Field = .{ .off = 81, .bits = 4 }; // :3486
    pub const rq_type: Field = .{ .off = 1381, .bits = 3 }; // :3569
    pub const send_dbr_mode: Field = .{ .off = 128, .bits = 2 }; // :3496
    pub const dbr_umem_valid: Field = .{ .off = 1667, .bits = 1 }; // :3592
    pub const dbr_umem_id: Field = .{ .off = 1824, .bits = 32 }; // :3596
    pub const dbr_addr: Field = .{ .off = 1280, .bits = 64 }; // :3564
    pub const cd_master: Field = .{ .off = 1183, .bits = 1 }; // :3552
    pub const log_page_size: Field = .{ .off = 163, .bits = 5 }; // :3501
    pub const rwe: Field = .{ .off = 1169, .bits = 1 }; // :3544
    pub const next_rcv_psn: Field = .{ .off = 1192, .bits = 24 }; // :3556
    pub const remote_qpn: Field = .{ .off = 168, .bits = 24 }; // :3502
    pub const log_msg_max: Field = .{ .off = 67, .bits = 5 }; // :3481
    pub const mtu: Field = .{ .off = 64, .bits = 3 }; // :3480
    pub const log_rra_max: Field = .{ .off = 1160, .bits = 3 }; // :3540
    pub const next_send_psn: Field = .{ .off = 968, .bits = 24 }; // :3523
    pub const retry_count: Field = .{ .off = 909, .bits = 3 }; // :3512
    pub const rnr_retry: Field = .{ .off = 912, .bits = 3 }; // :3513
    pub const log_sra_max: Field = .{ .off = 904, .bits = 3 }; // :3510
    pub const log_ack_req_freq: Field = .{ .off = 896, .bits = 4 }; // :3508
    pub const min_rnr_nak: Field = .{ .off = 1187, .bits = 5 }; // :3555
    // the primary address path (mlx5_ifc_ads_bits, mlx5_ifc.h:3408-3451, at 192 bits in the qpc)
    pub const pap_vhca_port_num: Field = .{ .off = 488, .bits = 8 }; // :3445
    pub const pap_pkey_index: Field = .{ .off = 208, .bits = 16 }; // :3413
    pub const pap_hop_limit: Field = .{ .off = 280, .bits = 8 }; // :3425
    pub const pap_src_addr_index: Field = .{ .off = 264, .bits = 8 }; // :3422
    pub const pap_rmac_47_32: Field = .{ .off = 496, .bits = 16 }; // :3446
    pub const pap_rmac_31_0: Field = .{ .off = 512, .bits = 32 }; // :3448
    pub const pap_udp_sport: Field = .{ .off = 464, .bits = 16 }; // :3440
    pub const pap_dscp: Field = .{ .off = 458, .bits = 6 }; // :3439
    pub const pap_ack_timeout: Field = .{ .off = 256, .bits = 5 }; // :3420
    /// The address path's 16-byte remote GID (or IP): a byte array at 320 bits (:3431).
    pub const pap_rgid_rip_byte: usize = 40;
};

/// The messages: where the `qpc` starts (192 bits) and the other fields' offsets (mlx5_ifc.h:3636-3934, :3879-3890, :3907-3921).
pub const msg = struct {
    pub const qpc_at: usize = 24;
    pub const opcode: Field = .{ .off = 0, .bits = 16 };
    pub const qpn: Field = .{ .off = 72, .bits = 24 }; // the modify, 2err and query messages (:3809, :3887, :3934)
    pub const opt_param_mask: Field = .{ .off = 128, .bits = 32 }; // :3813
    pub const wq_umem_offset: Field = .{ .off = 2048, .bits = 64 }; // create_qp_in (:3663)
    pub const wq_umem_id: Field = .{ .off = 2112, .bits = 32 }; // :3665
    pub const wq_umem_valid: Field = .{ .off = 2144, .bits = 1 }; // :3667
    pub const out_status: Field = .{ .off = 0, .bits = 8 }; // every out message (:3637, :5836)
    pub const out_syndrome: Field = .{ .off = 32, .bits = 32 }; // :3640, :5839
    pub const create_out_qpn: Field = .{ .off = 72, .bits = 24 }; // create_qp_out (:3643)
};

/// Message sizes in bytes: the create and the three modifies and the query's reply are 2,176 bits; the 2err and query requests, and every other reply, 128.
pub const qp_msg_bytes: usize = 272;
pub const small_msg_bytes: usize = 16;

/// The `opt_param_mask` bit that lets RTR2RTS set `min_rnr_nak`: PRIV_DOCA_MLX5_QP_OPT_PARAM_MIN_RNR_NAK = bit 6
/// (GPUNetIO doca_verbs_qp.cpp, as m19c_wait.cu:143 uses it).
pub const opt_min_rnr_nak: u64 = 1 << 6;

/// What creating a Core Direct master queue pair needs: the objects it is made on.
pub const QpCreate = struct {
    pdn: u32,
    uar_page: u32,
    cqn: u32,
    log_sq_size: u8,
    umem_id: u32,
    /// Where the doorbell record is in the umem (the send queue is at 0).
    dbr_off: u64,
    cd_master: bool = true,
};

/// CREATE_QP, field for field as m19c_wait.cu:90-108: a reliable-connected queue pair with no receive queue, its send queue and
/// doorbell record in one umem of host memory, one completion queue for both directions, `cd_master` as asked.
pub fn createQpIn(buf: *[qp_msg_bytes]u8, c: QpCreate) void {
    @memset(buf, 0);
    const q = buf[msg.qpc_at..];
    put(buf, msg.opcode, op.create_qp);
    put(q, qpc.st, st_rc);
    put(q, qpc.pd, c.pdn);
    put(q, qpc.uar_page, c.uar_page);
    put(q, qpc.cqn_snd, c.cqn);
    put(q, qpc.cqn_rcv, c.cqn);
    put(q, qpc.log_sq_size, c.log_sq_size);
    put(q, qpc.rq_type, rq_type_zero_size);
    put(q, qpc.send_dbr_mode, 0);
    put(q, qpc.dbr_umem_valid, 1);
    put(q, qpc.dbr_umem_id, c.umem_id);
    put(q, qpc.dbr_addr, c.dbr_off);
    put(q, qpc.cd_master, @intFromBool(c.cd_master));
    put(buf, msg.wq_umem_offset, 0);
    put(buf, msg.wq_umem_id, c.umem_id);
    put(buf, msg.wq_umem_valid, 1);
    put(q, qpc.log_page_size, 0);
}

/// RST2INIT (m19c_wait.cu:116-120).
pub fn rst2initIn(buf: *[qp_msg_bytes]u8, qpn: u32) void {
    @memset(buf, 0);
    const q = buf[msg.qpc_at..];
    put(buf, msg.opcode, op.rst2init_qp);
    put(buf, msg.qpn, qpn);
    put(q, qpc.pap_vhca_port_num, 1);
    put(q, qpc.pm_state, pm_state_migrated);
    put(q, qpc.pap_pkey_index, 0);
    put(q, qpc.rwe, 1);
}

/// The peer a queue pair is connected to.
pub const Remote = struct {
    qpn: u32,
    gid: [16]u8,
    mac: [6]u8,
};

/// INIT2RTR (m19c_wait.cu:122-135): the peer's queue pair, GID and MAC, the path MTU (the port's active one, at most 4 KiB: the
/// verbs MTU enum's numbers are the interface's), the GID index of this side, a UDP source port the caller picks (RoCE v2's flow
/// entropy), no optional parameter.
pub fn init2rtrIn(buf: *[qp_msg_bytes]u8, qpn: u32, r: Remote, gid_index: u8, udp_sport: u16, mtu: u8) void {
    @memset(buf, 0);
    const q = buf[msg.qpc_at..];
    put(buf, msg.opcode, op.init2rtr_qp);
    put(buf, msg.qpn, qpn);
    put(q, qpc.next_rcv_psn, 0);
    put(q, qpc.remote_qpn, r.qpn);
    put(q, qpc.log_msg_max, 30);
    put(q, qpc.mtu, mtu);
    @memcpy(q[qpc.pap_rgid_rip_byte..][0..16], &r.gid);
    put(q, qpc.pap_hop_limit, 64);
    put(q, qpc.pap_src_addr_index, gid_index);
    put(q, qpc.pap_rmac_47_32, std.mem.readInt(u16, r.mac[0..2], .big));
    put(q, qpc.pap_rmac_31_0, std.mem.readInt(u32, r.mac[2..6], .big));
    put(q, qpc.pap_udp_sport, udp_sport);
    put(q, qpc.pap_dscp, 0);
    put(q, qpc.pap_pkey_index, 0);
    put(q, qpc.rwe, 1);
    put(q, qpc.log_rra_max, 0);
}

/// RTR2RTS (m19c_wait.cu:137-144): the verbs path's timeout 14 and seven retries (rocinante_verbs.zig connect), min_rnr_nak 12.
pub fn rtr2rtsIn(buf: *[qp_msg_bytes]u8, qpn: u32) void {
    @memset(buf, 0);
    const q = buf[msg.qpc_at..];
    put(buf, msg.opcode, op.rtr2rts_qp);
    put(buf, msg.qpn, qpn);
    put(q, qpc.next_send_psn, 0);
    put(q, qpc.pap_ack_timeout, 14);
    put(q, qpc.retry_count, 7);
    put(q, qpc.rnr_retry, 7);
    put(q, qpc.log_sra_max, 0);
    put(q, qpc.rwe, 1);
    put(q, qpc.log_ack_req_freq, 0);
    put(q, qpc.min_rnr_nak, 12);
    put(buf, msg.opt_param_mask, opt_min_rnr_nak);
}

/// 2ERR_QP: the queue pair to the error state (every outstanding work request flushes).
pub fn qp2errIn(buf: *[small_msg_bytes]u8, qpn: u32) void {
    @memset(buf, 0);
    put(buf, msg.opcode, op.qp_2err);
    put(buf, msg.qpn, qpn);
}

/// QUERY_QP's request.
pub fn queryQpIn(buf: *[small_msg_bytes]u8, qpn: u32) void {
    @memset(buf, 0);
    put(buf, msg.opcode, op.query_qp);
    put(buf, msg.qpn, qpn);
}

// -- the work queue entries ------------------------------------------------------------------------------------------

/// One work queue entry basic block (a send queue's slot).
pub const wqebb: usize = 64;

/// mlx5dv.h: MLX5_OPCODE_RDMA_WRITE (:1077); DPDK mlx5_prm.h: MLX5_OPCODE_WAIT 0x0f and MLX5_OPC_MOD_WAIT_DATA 1.
pub const opcode_rdma_write: u8 = 0x08;
pub const opcode_wait: u8 = 0x0f;
pub const opmod_wait_data: u8 = 1;
/// mlx5dv.h:34689 MLX5_WQE_CTRL_CQ_UPDATE (= 2 << 2): a completion for this entry.
pub const ctrl_cq_update: u8 = 2 << 2;
/// mlx5dv.h:34909 MLX5_INLINE_SEG: the top bit of an inline segment's byte count.
pub const inline_seg: u32 = 0x8000_0000;
/// Every entry this transport writes is a control segment and two more: 3 sixteen-byte units, 48 bytes, one block.
pub const ds_units: u8 = 3;

/// A WAIT's compare of the memory word with its value (DPDK mlx5_prm.h, `enum mlx5_wait_cond`): ALWAYS_TRUE 0, EQUAL 1, BIGGER 2,
/// SMALLER 3, CYCLIC_BIGGER 4, CYCLIC_SMALLER 5, INVERT 0x10 as a flag.
pub const WaitCond = enum(u8) { always_true = 0, equal = 1, bigger = 2, smaller = 3, cyclic_bigger = 4, cyclic_smaller = 5 };

/// mlx5dv_set_ctrl_seg (mlx5dv.h:39493-39509): opmod, index and opcode in the first word, the queue pair number and the count of
/// sixteen-byte units in the second, the flags in byte 11. The entry is zeroed first.
pub fn ctrlSeg(w: *[wqebb]u8, pi: u16, opcode: u8, opmod: u8, qpn: u32, fm_ce_se: u8, ds: u8) void {
    @memset(w, 0);
    std.mem.writeInt(u32, w[0..4], (@as(u32, opmod) << 24) | (@as(u32, pi) << 8) | opcode, .big);
    std.mem.writeInt(u32, w[4..8], (qpn << 8) | ds, .big);
    w[11] = fm_ce_se;
}

/// WAIT-on-data (m19c_wait.cu:220-228, DPDK `struct mlx5_wqe_wseg`): the segment after the control segment is {operation: the
/// condition in the low bits and 0x10 for invert, lkey, address high word, address low word with its low 3 bits the fail action
/// (0), value, mask}; value and mask are the bytes of the word as they sit in memory, no byte swap.
pub fn waitWqe(w: *[wqebb]u8, pi: u16, qpn: u32, cond: WaitCond, invert: bool, lkey: u32, va: u64, value: u64, mask: u64) void {
    ctrlSeg(w, pi, opcode_wait, opmod_wait_data, qpn, 0, ds_units);
    std.debug.assert(va % 8 == 0);
    std.mem.writeInt(u32, w[16..20], @as(u32, @backingInt(cond)) | (if (invert) @as(u32, 0x10) else 0), .big);
    std.mem.writeInt(u32, w[20..24], lkey, .big);
    std.mem.writeInt(u32, w[24..28], @intCast(va >> 32), .big);
    std.mem.writeInt(u32, w[28..32], @as(u32, @truncate(va)) & ~@as(u32, 7), .big);
    std.mem.writeInt(u64, w[32..40], value, builtin.cpu.arch.endian());
    std.mem.writeInt(u64, w[40..48], mask, builtin.cpu.arch.endian());
}

/// An RDMA WRITE of `len` bytes from the local address (m19c_wait.cu:212-218 post_write, mlx5dv.h:35942 raddr segment, :35148 data segment).
pub fn writeWqe(w: *[wqebb]u8, pi: u16, qpn: u32, signal: bool, raddr: u64, rkey: u32, lkey: u32, laddr: u64, len: u32) void {
    ctrlSeg(w, pi, opcode_rdma_write, 0, qpn, if (signal) ctrl_cq_update else 0, ds_units);
    std.mem.writeInt(u64, w[16..24], raddr, .big);
    std.mem.writeInt(u32, w[24..28], rkey, .big);
    std.mem.writeInt(u32, w[32..36], len, .big);
    std.mem.writeInt(u32, w[36..40], lkey, .big);
    std.mem.writeInt(u64, w[40..48], laddr, .big);
}

/// An RDMA WRITE of the 8 bytes `word` as they sit in memory, carried in the entry: an inline segment (the byte count with its top
/// bit, then the bytes), as rdma-core's set_data_inl_seg writes it (qp.c:327-360): control 16 + remote address 16 + 4 + 8 bytes,
/// `align(8 + 4, 16) / 16 + 2` = 3 units, one block.
pub fn inlineWordWqe(w: *[wqebb]u8, pi: u16, qpn: u32, signal: bool, raddr: u64, rkey: u32, word: u64) void {
    ctrlSeg(w, pi, opcode_rdma_write, 0, qpn, if (signal) ctrl_cq_update else 0, ds_units);
    std.mem.writeInt(u64, w[16..24], raddr, .big);
    std.mem.writeInt(u32, w[24..28], rkey, .big);
    std.mem.writeInt(u32, w[32..36], 8 | inline_seg, .big);
    std.mem.writeInt(u64, w[36..44], word, builtin.cpu.arch.endian());
}

// -- the completion queue read by hand -------------------------------------------------------------------------------

/// The completion opcodes an entry's `op_own` carries in its top four bits (m19c_wait.cu:241, :250): 0xf an empty slot, 0xd a
/// requester's error, 0xe a responder's error.
pub const cqe_invalid: u8 = 0xf;
pub const cqe_req_err: u8 = 0xd;
pub const cqe_resp_err: u8 = 0xe;

/// One completion, decoded: mlx5_cqe64 (mlx5dv.h: sop_drop_qpn at byte 56, wqe_counter at 60, op_own at 63) and mlx5_err_cqe
/// (vendor_err_synd at 54, syndrome at 55).
pub const Completion = struct {
    opcode: u8,
    qpn: u32,
    wqe_counter: u16,
    syndrome: u8,
    vendor_err: u8,

    pub fn isError(c: Completion) bool {
        return c.opcode == cqe_req_err or c.opcode == cqe_resp_err;
    }
};

/// The view of one completion queue (mlx5dv_cq: buffer, doorbell record, entries, entry size) and the consumer index.
pub const CqView = struct {
    buf: [*]u8,
    dbrec: [*]volatile u32,
    cqe_cnt: u32,
    cqe_size: u32,
    ci: u32 = 0,

    /// The next completion, or null when the slot's owner bit says the card has not written it (m19c_wait.cu:238-245): the consumer
    /// index and the doorbell record advance past it.
    pub fn pollOne(c: *CqView) ?Completion {
        const n = c.ci;
        const base = c.buf + @as(usize, n & (c.cqe_cnt - 1)) * c.cqe_size + (if (c.cqe_size == 128) @as(usize, 64) else 0);
        const own = @atomicLoad(u8, &base[63], .acquire);
        const opc = own >> 4;
        const wrap: u8 = @intFromBool(n & c.cqe_cnt != 0);
        if (opc == cqe_invalid or ((own & 1) ^ wrap) != 0) return null;
        const e = base[0..64];
        const out: Completion = .{
            .opcode = opc,
            .qpn = std.mem.readInt(u32, e[56..60], .big) & 0xff_ffff,
            .wqe_counter = std.mem.readInt(u16, e[60..62], .big),
            .syndrome = e[55],
            .vendor_err = e[54],
        };
        c.ci +%= 1;
        c.dbrec[0] = std.mem.nativeToBig(u32, c.ci & 0xff_ffff);
        return out;
    }
};

// -- the entry points ------------------------------------------------------------------------------------------------

pub const Ctx = *vabi.Context;

pub const ContextAttr = extern struct { flags: u32 = 0, comp_mask: u64 = 0, fds: ?*anyopaque = null };
/// mlx5dv_context_attr_flags: MLX5DV_CONTEXT_FLAGS_DEVX (mlx5dv.h:46794).
pub const context_flag_devx: u32 = 1;

pub const DevxObj = opaque {};
pub const DevxUmem = extern struct { umem_id: u32 };
pub const DevxUar = extern struct { reg_addr: ?*anyopaque, base_addr: ?*anyopaque, page_id: u32, mmap_off: i64, comp_mask: u64 };
/// mlx5_ib_uapi_uar_alloc_type (rdma-core kernel-headers/rdma/mlx5_user_ioctl_verbs.h:89-91): BlueFlame 0, non-cached 1.
pub const uar_type_bf: u32 = 0;
pub const uar_type_nc: u32 = 1;

pub const DvCq = extern struct { buf: ?*anyopaque = null, dbrec: ?[*]volatile u32 = null, cqe_cnt: u32 = 0, cqe_size: u32 = 0, cq_uar: ?*anyopaque = null, cqn: u32 = 0, comp_mask: u64 = 0 };
pub const DvPd = extern struct { pdn: u32 = 0, comp_mask: u64 = 0 };
/// mlx5dv_obj: eight (in, out) pairs: qp, cq, srq, rwq, dm, ah, pd, devx.
pub const DvObj = extern struct { pairs: [8][2]?*anyopaque = @splat(.{ null, null }) };
pub const obj_cq: u64 = 1 << 1;
pub const obj_pd: u64 = 1 << 6;

/// The libmlx5 entry points, each resolved by its exact name (a missing one refuses the library).
pub const Api = struct {
    mlx5dv_open_device: *const fn (dev: *vabi.Device, attr: *ContextAttr) callconv(.c) ?Ctx,
    mlx5dv_init_obj: *const fn (obj: *DvObj, obj_type: u64) callconv(.c) c_int,
    mlx5dv_devx_obj_create: *const fn (ctx: Ctx, in: *const anyopaque, inlen: usize, out: *anyopaque, outlen: usize) callconv(.c) ?*DevxObj,
    mlx5dv_devx_obj_modify: *const fn (obj: *DevxObj, in: *const anyopaque, inlen: usize, out: *anyopaque, outlen: usize) callconv(.c) c_int,
    mlx5dv_devx_obj_query: *const fn (obj: *DevxObj, in: *const anyopaque, inlen: usize, out: *anyopaque, outlen: usize) callconv(.c) c_int,
    mlx5dv_devx_obj_destroy: *const fn (obj: *DevxObj) callconv(.c) c_int,
    mlx5dv_devx_umem_reg: *const fn (ctx: Ctx, addr: ?*anyopaque, size: usize, access: u32) callconv(.c) ?*DevxUmem,
    mlx5dv_devx_umem_dereg: *const fn (umem: *DevxUmem) callconv(.c) c_int,
    mlx5dv_devx_alloc_uar: *const fn (ctx: Ctx, flags: u32) callconv(.c) ?*DevxUar,
    mlx5dv_devx_free_uar: *const fn (uar: *DevxUar) callconv(.c) void,
};

pub const Error = error{ LibraryUnavailable, MissingSymbol };

pub const library: [:0]const u8 = "libmlx5.so.1";

/// libmlx5 by dlopen, as the engine loads libibverbs (fabric/verbs.zig).
pub const Mlx5 = struct {
    lib: ?std.DynLib,
    api: Api,

    pub fn open() Error!Mlx5 {
        var lib = std.DynLib.openZ(library.ptr) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        return .{ .lib = lib, .api = api };
    }

    /// A table the host tests fill in themselves: no library behind it.
    pub fn fake(api: Api) Mlx5 {
        return .{ .lib = null, .api = api };
    }

    pub fn close(m: *Mlx5) void {
        if (m.lib) |*l| l.close();
        m.lib = null;
    }
};

// -- barriers --------------------------------------------------------------------------------------------------------

/// The stores before it reach the card's view of memory before the stores after it (`dmb oshst` on arm64, as m19c_wait.cu:60
/// wmb_dev). On the host tests' x86 a release fence's worth: a compiler barrier is the strongest thing there is to do.
pub inline fn wmbDev() void {
    if (builtin.cpu.arch == .aarch64) asm volatile ("dmb oshst" ::: .{ .memory = true }) else asm volatile ("" ::: .{ .memory = true });
}

/// `dsb st`: the doorbell store has completed (m19c_wait.cu:234).
pub inline fn dsbSt() void {
    if (builtin.cpu.arch == .aarch64) asm volatile ("dsb st" ::: .{ .memory = true }) else asm volatile ("" ::: .{ .memory = true });
}

// -- tests -----------------------------------------------------------------------------------------------------------

test "the field setter is DEVX_SET: a field in the middle of a double word, a whole double word, a quad word, all big endian" {
    var b: [32]u8 = @splat(0);
    put(&b, .{ .off = 12, .bits = 4 }, 0xa);
    try std.testing.expectEqual(@as(u8, 0x0a), b[1]);
    put(&b, .{ .off = 32, .bits = 32 }, 0x11223344);
    try std.testing.expectEqualSlices(u8, &.{ 0x11, 0x22, 0x33, 0x44 }, b[4..8]);
    put(&b, .{ .off = 64, .bits = 64 }, 0x0102030405060708);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, b[8..16]);
    // a field set twice keeps its neighbours: bits 0..3 and 4..7 of one byte
    put(&b, .{ .off = 128, .bits = 4 }, 0x5);
    put(&b, .{ .off = 132, .bits = 4 }, 0x9);
    try std.testing.expectEqual(@as(u8, 0x59), b[16]);
    put(&b, .{ .off = 128, .bits = 4 }, 0x0);
    try std.testing.expectEqual(@as(u8, 0x09), b[16]);
    try std.testing.expectEqual(@as(u64, 0x1122_3344), get(&b, .{ .off = 32, .bits = 32 }));
    try std.testing.expectEqual(@as(u64, 0x0102030405060708), get(&b, .{ .off = 64, .bits = 64 }));
    // a value wider than the field is cut to it, as DEVX_SET's mask does
    put(&b, .{ .off = 12, .bits = 4 }, 0x1f5);
    try std.testing.expectEqual(@as(u64, 5), get(&b, .{ .off = 12, .bits = 4 }));
}

test "create QP: the bytes m19c_wait.cu's DEVX_SET calls make (qpc at byte 24; each field where mlx5_ifc.h puts it)" {
    var b: [qp_msg_bytes]u8 = undefined;
    createQpIn(&b, .{ .pdn = 0x12, .uar_page = 0x345, .cqn = 0x67, .log_sq_size = 12, .umem_id = 0x89, .dbr_off = 0x40000 });
    try std.testing.expectEqual(@as(u64, op.create_qp), get(&b, msg.opcode));
    const q = b[msg.qpc_at..];
    try std.testing.expectEqual(@as(u64, 0), get(q, qpc.st));
    try std.testing.expectEqual(@as(u64, 0x12), get(q, qpc.pd));
    try std.testing.expectEqual(@as(u64, 0x345), get(q, qpc.uar_page));
    try std.testing.expectEqual(@as(u64, 0x67), get(q, qpc.cqn_snd));
    try std.testing.expectEqual(@as(u64, 0x67), get(q, qpc.cqn_rcv));
    try std.testing.expectEqual(@as(u64, 12), get(q, qpc.log_sq_size));
    try std.testing.expectEqual(@as(u64, 3), get(q, qpc.rq_type));
    try std.testing.expectEqual(@as(u64, 1), get(q, qpc.dbr_umem_valid));
    try std.testing.expectEqual(@as(u64, 0x89), get(q, qpc.dbr_umem_id));
    try std.testing.expectEqual(@as(u64, 0x40000), get(q, qpc.dbr_addr));
    try std.testing.expectEqual(@as(u64, 1), get(q, qpc.cd_master));
    try std.testing.expectEqual(@as(u64, 0x89), get(&b, msg.wq_umem_id));
    try std.testing.expectEqual(@as(u64, 1), get(&b, msg.wq_umem_valid));
    // the message ends where the structure does: nothing set past the last field's double word
    for (b[(msg.wq_umem_valid.off / 8) + 4 ..]) |x| try std.testing.expectEqual(@as(u8, 0), x);
    // cd_master is one bit: off, the queue pair is an ordinary one
    createQpIn(&b, .{ .pdn = 1, .uar_page = 1, .cqn = 1, .log_sq_size = 11, .umem_id = 1, .dbr_off = 64, .cd_master = false });
    try std.testing.expectEqual(@as(u64, 0), get(b[msg.qpc_at..], qpc.cd_master));
}

test "the field offsets of the qpc are the ones the header's own bit counts give (spot checks against independent sums)" {
    // the qpc's first dwords: state 4 + lag 4 + st 8 + ... then pd at 40, log_sq_size at 81; the address path begins at 192 bits and holds ack_timeout at 64 bits into it
    try std.testing.expectEqual(@as(u16, 192 + 64), qpc.pap_ack_timeout.off);
    try std.testing.expectEqual(@as(u16, 192 + 128), @as(u16, qpc.pap_rgid_rip_byte * 8));
    // the address path is 352 bits (rgid_rip 128 of them): the field after it, in the qpc, is past 192 + 352
    try std.testing.expect(qpc.pap_rmac_31_0.off + 32 <= 192 + 352);
    try std.testing.expectEqual(@as(u16, 192 + 320), qpc.pap_rmac_31_0.off);
    // 64-bit fields on a 64-bit boundary
    try std.testing.expectEqual(@as(u16, 0), qpc.dbr_addr.off % 64);
    try std.testing.expectEqual(@as(u16, 0), msg.wq_umem_offset.off % 64);
}

test "the three transitions set what m19c_wait.cu sets" {
    var b: [qp_msg_bytes]u8 = undefined;
    rst2initIn(&b, 0xabcdef);
    try std.testing.expectEqual(@as(u64, op.rst2init_qp), get(&b, msg.opcode));
    try std.testing.expectEqual(@as(u64, 0xabcdef), get(&b, msg.qpn));
    var q = b[msg.qpc_at..];
    try std.testing.expectEqual(@as(u64, 1), get(q, qpc.pap_vhca_port_num));
    try std.testing.expectEqual(@as(u64, 3), get(q, qpc.pm_state));
    try std.testing.expectEqual(@as(u64, 1), get(q, qpc.rwe));
    const gid: [16]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 10, 255, 0, 34 };
    init2rtrIn(&b, 0xabcdef, .{ .qpn = 0x123456, .gid = gid, .mac = .{ 0x00, 0x11, 0x22, 0x33, 0x44, 0x55 } }, 3, 50000, 5);
    try std.testing.expectEqual(@as(u64, op.init2rtr_qp), get(&b, msg.opcode));
    q = b[msg.qpc_at..];
    try std.testing.expectEqual(@as(u64, 0x123456), get(q, qpc.remote_qpn));
    try std.testing.expectEqual(@as(u64, 30), get(q, qpc.log_msg_max));
    try std.testing.expectEqual(@as(u64, 5), get(q, qpc.mtu));
    try std.testing.expectEqualSlices(u8, &gid, q[qpc.pap_rgid_rip_byte..][0..16]);
    try std.testing.expectEqual(@as(u64, 64), get(q, qpc.pap_hop_limit));
    try std.testing.expectEqual(@as(u64, 3), get(q, qpc.pap_src_addr_index));
    try std.testing.expectEqual(@as(u64, 0x0011), get(q, qpc.pap_rmac_47_32));
    try std.testing.expectEqual(@as(u64, 0x22334455), get(q, qpc.pap_rmac_31_0));
    try std.testing.expectEqual(@as(u64, 50000), get(q, qpc.pap_udp_sport));
    try std.testing.expectEqual(@as(u64, 0), get(&b, msg.opt_param_mask));
    rtr2rtsIn(&b, 0xabcdef);
    try std.testing.expectEqual(@as(u64, op.rtr2rts_qp), get(&b, msg.opcode));
    q = b[msg.qpc_at..];
    try std.testing.expectEqual(@as(u64, 14), get(q, qpc.pap_ack_timeout));
    try std.testing.expectEqual(@as(u64, 7), get(q, qpc.retry_count));
    try std.testing.expectEqual(@as(u64, 7), get(q, qpc.rnr_retry));
    try std.testing.expectEqual(@as(u64, 12), get(q, qpc.min_rnr_nak));
    try std.testing.expectEqual(@as(u64, 1 << 6), get(&b, msg.opt_param_mask));
    var s: [small_msg_bytes]u8 = undefined;
    qp2errIn(&s, 0x777);
    try std.testing.expectEqual(@as(u64, 0x507), get(&s, msg.opcode));
    try std.testing.expectEqual(@as(u64, 0x777), get(&s, msg.qpn));
}

test "a WAIT entry: opcode 0x0f, opmod 1 in bits 31:24, 3 units, then {condition, lkey, va high, va low, value, mask} in DPDK's order, value and mask as stored" {
    var w: [wqebb]u8 = undefined;
    waitWqe(&w, 0x1234, 0xabcdef, .cyclic_smaller, false, 0xdeadbeef, 0x0000_7fff_1234_5640, 0x1122334455667788, ~@as(u64, 0));
    try std.testing.expectEqual(@as(u8, 1), w[0]); // opmod 1 in the top byte
    try std.testing.expectEqual(@as(u8, 0x0f), w[3]);
    try std.testing.expectEqual(@as(u16, 0x1234), @as(u16, @intCast((std.mem.readInt(u32, w[0..4], .big) >> 8) & 0xffff)));
    try std.testing.expectEqual(@as(u32, 0xabcdef03), std.mem.readInt(u32, w[4..8], .big));
    try std.testing.expectEqual(@as(u32, 5), std.mem.readInt(u32, w[16..20], .big));
    try std.testing.expectEqual(@as(u32, 0xdeadbeef), std.mem.readInt(u32, w[20..24], .big));
    try std.testing.expectEqual(@as(u32, 0x7fff), std.mem.readInt(u32, w[24..28], .big));
    try std.testing.expectEqual(@as(u32, 0x1234_5640), std.mem.readInt(u32, w[28..32], .big));
    try std.testing.expectEqual(@as(u64, 0x1122334455667788), std.mem.readInt(u64, w[32..40], builtin.cpu.arch.endian()));
    try std.testing.expectEqual(~@as(u64, 0), std.mem.readInt(u64, w[40..48], builtin.cpu.arch.endian()));
    for (w[48..]) |x| try std.testing.expectEqual(@as(u8, 0), x);
    // the invert flag is bit 4 of the operation word, and the address's low three bits are never set (the fail action)
    waitWqe(&w, 0, 1, .bigger, true, 1, 0x1000, 0, 0);
    try std.testing.expectEqual(@as(u32, 0x12), std.mem.readInt(u32, w[16..20], .big));
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, w[28..32], .big) & 7);
}

test "a data WRITE and an inline word WRITE: one block each, 3 units, the signal bit only where asked" {
    var w: [wqebb]u8 = undefined;
    writeWqe(&w, 7, 0x55, false, 0x1_0000_2000, 0x77, 0x88, 0x3000, 4096);
    try std.testing.expectEqual(@as(u32, 0x0000_0708), std.mem.readInt(u32, w[0..4], .big));
    try std.testing.expectEqual(@as(u32, 0x55 << 8 | 3), std.mem.readInt(u32, w[4..8], .big));
    try std.testing.expectEqual(@as(u8, 0), w[11]);
    try std.testing.expectEqual(@as(u64, 0x1_0000_2000), std.mem.readInt(u64, w[16..24], .big));
    try std.testing.expectEqual(@as(u32, 0x77), std.mem.readInt(u32, w[24..28], .big));
    try std.testing.expectEqual(@as(u32, 4096), std.mem.readInt(u32, w[32..36], .big));
    try std.testing.expectEqual(@as(u32, 0x88), std.mem.readInt(u32, w[36..40], .big));
    try std.testing.expectEqual(@as(u64, 0x3000), std.mem.readInt(u64, w[40..48], .big));
    writeWqe(&w, 7, 0x55, true, 0, 0, 0, 0, 0);
    try std.testing.expectEqual(@as(u8, 8), w[11]);
    inlineWordWqe(&w, 9, 0x55, true, 0x4000, 0x99, 0x0102030405060708);
    try std.testing.expectEqual(@as(u32, 0x0000_0908), std.mem.readInt(u32, w[0..4], .big));
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, w[4..8], .big) & 0xff);
    try std.testing.expectEqual(@as(u8, 8), w[11]);
    try std.testing.expectEqual(@as(u32, 0x8000_0008), std.mem.readInt(u32, w[32..36], .big));
    try std.testing.expectEqual(@as(u64, 0x0102030405060708), std.mem.readInt(u64, w[36..44], builtin.cpu.arch.endian()));
    for (w[44..]) |x| try std.testing.expectEqual(@as(u8, 0), x);
}

test "the completion reader: an empty slot, an owner bit that flips with each lap of the ring, an error entry, the doorbell record" {
    var ring: [4 * 64]u8 align(64) = @splat(0);
    for (0..4) |i| ring[i * 64 + 63] = 0xf0; // never written: opcode 0xf
    var db: [2]u32 = .{ 0, 0 };
    var cq: CqView = .{ .buf = &ring, .dbrec = &db, .cqe_cnt = 4, .cqe_size = 64 };
    try std.testing.expect(cq.pollOne() == null);
    // the card writes lap 0 with owner bit 0: entry 0 a send completion of WQE 0x21, entry 1 a requester error
    std.mem.writeInt(u32, ring[56..60], 0x0000_0042, .big);
    std.mem.writeInt(u16, ring[60..62], 0x21, .big);
    ring[63] = 0x00;
    std.mem.writeInt(u32, ring[64 + 56 ..][0..4], 0x0000_0042, .big);
    ring[64 + 54] = 0x7;
    ring[64 + 55] = 0x12;
    ring[64 + 63] = (cqe_req_err << 4) | 0;
    const a = cq.pollOne().?;
    try std.testing.expectEqual(@as(u32, 0x42), a.qpn);
    try std.testing.expectEqual(@as(u16, 0x21), a.wqe_counter);
    try std.testing.expect(!a.isError());
    const e = cq.pollOne().?;
    try std.testing.expect(e.isError());
    try std.testing.expectEqual(@as(u8, 0x12), e.syndrome);
    try std.testing.expectEqual(@as(u8, 0x7), e.vendor_err);
    try std.testing.expectEqual(std.mem.nativeToBig(u32, 2), db[0]);
    try std.testing.expect(cq.pollOne() == null);
    // the second lap: owner bit 1; the slot still holding lap 0's entry (owner 0) is not new
    cq.ci = 4;
    try std.testing.expect(cq.pollOne() == null);
    ring[63] = 0x01;
    try std.testing.expect(cq.pollOne() != null);
}