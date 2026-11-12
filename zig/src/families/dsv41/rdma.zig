//! The family's one-hop RDMA all-gather of small fp32 slices (TP decode on GB10), host half: the Python build's
//! cuda/rdma_gather.cu v7 in Zig. The staging kernel copies this rank's slice into a slot of pinned host memory and
//! rings a doorbell; the proxy thread splits the slot over the node's RDMA devices and writes the parts, with an
//! immediate, into every peer's receive slot for this rank; a peer's proxy sees each part's receive completion and,
//! once all of a sequence's parts are in, publishes it in that peer's flag; the collecting kernel waits for the flags
//! and copies the slots out in rank order (or straight into the caller's views). The kernel pair is rdma_gather.cu's
//! and runs in graphs like any other kernel; a failed write or `abort` sets an abort word every wait also reads.
//!
//! Routes (a ring of nodes without a switch): every listed device is opened, each with its own RoCE v2 IPv4 GID (the
//! first in its table unless Settings.gid_index says), and toward each peer the devices on a subnet the peer's share
//! (TF_DS_RDMA_PREFIX bits, default 24; a port and its PCIe twin on one subnet both, in list order on either side)
//! carry the stripes. A peer no device reaches is relayed: the ranks routed to both it and this one (on four nodes in a
//! ring, the two neighbours) each write half of this rank's slot on from their own receive slot into its, as the slot's
//! owner would (the immediate names the owner), once all of it is in; a relaying rank posts its own sequence q only
//! after every relay of sequences up to q - (slots - 1) has completed, so no owner overwrites a slot the NIC is still
//! reading (the owner writes q + 1 into that slot only after collecting this rank's q). The
//! GPU half and the bytes are the same either way.
const std = @import("std");
const cuda = @import("cuda");
const fabric = @import("fabric");
const seqs = @import("rdma_seq.zig");
const ring_watch = @import("ring_watch.zig");
const where = @import("where.zig");
const wait = @import("ring_wait.zig");
const mx = @import("mlx5_devx.zig");

const vabi = fabric.verbs_abi;
const Verbs = fabric.verbs.Verbs;
const DevicePtr = cuda.abi.DevicePtr;

pub const max_ranks = seqs.max_ranks;
pub const max_devices = seqs.max_devices;
const recv_depth = 256; // receive requests kept posted on every QP (a write with immediate consumes one)
/// The streams that may gather in place on one ring: each gets a fixed send slot after the rotating ones, so a
/// captured graph's producer pointer is stable and a staged send's rotating slot can never overwrite it.
pub const max_users = 2;
/// The device scalars: seq (u64, offset 0), staged (u32, 8), probe (8 x u64, 16), then send_last [max_users] (80): the sequence each in-place user
/// last published, written and read by the GPU alone, so it lives in device memory (the await kernel reads it from L2, not from pinned host memory).
const scalars_bytes = 80 + max_users * 8;
/// Whether an in-place ring is possible: it needs the write-combined data slots; null: it is, else why not.
pub fn inplaceOff(wc: bool, inplace: bool) ?[]const u8 {
    if (inplace and !wc) return "needs TF_DS_RING_WC=1 (the send slot is a write-combined data slot its producer writes)";
    return null;
}

/// The arrival signal TF_DS_RING_WORD may use: the sending proxy writes each lane's data with a plain RDMA WRITE
/// and then, on the same queue pair, the lane's 64-bit generation word. `.word` writes the sequence itself as an
/// inline 8-byte RDMA WRITE; `.atomic` adds one to the word with an RDMA ATOMIC fetch-and-add (the signal the IB
/// spec covers for a write that follows data). The collecting kernel waits on the word instead of a proxy-published
/// flag.
pub const Roci = enum {
    off,
    word,
    atomic,

    pub fn on(x: Roci) bool {
        return x != .off;
    }

    /// The signal's name for the bench's line (TF_DS_RING_WORD_SIGNAL's own words).
    pub fn signalName(x: Roci) []const u8 {
        return switch (x) {
            .off => "off",
            .word => "word",
            .atomic => "atomic",
        };
    }

    /// The words a peer's lane holds after `sent` signals from zero: the sequence for `.word`, the count for `.atomic`.
    pub fn reached(w: u64, expected: u64) bool {
        return w >= expected;
    }
};

/// The generation words a peer's lane writes into this rank's region: one per lane, at `peer * max_devices + lane`;
/// the last word is the proxy's local source for the inline write. The region is ordinary pinned host memory (the
/// words are never write-combined: the NIC and the CPU read-modify-write them).
pub fn rociWords() usize {
    return max_ranks * max_devices + 1;
}

/// The signals a `.word` ring's lane has sent when it reaches sequence `q`: the word is the sequence. An `.atomic`
/// lane's word counts the fetch-and-adds; the first sequence is 1, so its word is its sequence too.
pub fn ringWordSignalWord(q: u64) u64 {
    return q;
}
const send_depth = 512;
const cq_depth = 4096;
const stage_threads = 512;
const collect_threads = 256;
comptime {
    // rdma_gather.cu's COLLECT_THREADS pins the same 256: the host's unit4 = 4 * collect_threads and the kernel's
    // c0 + k * COLLECT_THREADS must agree.
    if (collect_threads != 256) @compileError("collect_threads must equal rdma_gather.cu's COLLECT_THREADS (256)");
}
const mtu_4096 = 5;
const port_num = 1;
const relay_wr: u64 = 1 << 63; // a relayed write's wr_id: its sequence with this bit
const fault_magic: u64 = 0x524f_4349_0000_0001; // a generation-word wait's record: "ROCI", written once
const wc_recv_rdma_with_imm = 129;
const host_register_portable = 0x01;
const host_register_devicemap = 0x02;

pub const Error = error{ VerbsFailed, PostFailed, BadInfo, TooLarge, Failed, NoHostRegister, NoGid } || cuda.Error || fabric.verbs.Error || std.posix.MMapError || std.Thread.SpawnError;

/// TF_RDMA_GID: the RDMA rings' source GID index (the port's RoCE v2 IPv4 entry); 5 as served on the recipe authors' nodes.
pub fn rdmaGid() u8 {
    const v = std.c.getenv("TF_RDMA_GID") orelse return 5;
    return std.fmt.parseInt(u8, std.mem.span(v), 10) catch 5;
}

/// The Python build's switches, with their defaults (TF_RDMA_SLOTS, TF_RDMA_STAGE_BLOCKS, ..., TF_RDMA_PDL as served).
pub const Settings = struct {
    slots: u32 = 4,
    max_bytes: usize,
    gid_index: ?u8 = null, // null: each device's first RoCE v2 IPv4 GID
    prefix: ?u8 = null, // the subnets' prefix length that pairs devices with a peer's (null: TF_DS_RDMA_PREFIX, else 24)
    stage_blocks: u32 = 16,
    collect_blocks: u32 = 48,
    collect_grain: u32 = 4,
    pdl: bool = true,
    own_early: bool = true,
    unroll: bool = true,
    fence: bool = true,
    probing: bool = false,
    wc: bool = false, // TF_DS_RING_WC: the send and landing data slots from cudaHostAlloc(Mapped | WriteCombined)
    inplace: bool = false, // TF_DS_RING_INPLACE (needs wc): a producer writes a fixed send slot; no staging copy
    roci: Roci = .off, // TF_DS_RING_WORD: each lane's plain data write then the generation word it waits on in-kernel
    roci_deadline_ns: u64 = 120 * std.time.ns_per_s, // a lane's word not reaching its sequence: fail stop after this
    wait_explicit: bool = true, // TF_DS_RING_WAIT was set (to 1), not the default: only names the switch in a refusal (Model.open has already fallen back where the default cannot run)
    wait: bool = false, // TF_DS_RING_WAIT (needs roci == .word): Core Direct queue pairs wait in the card on the doorbell; no proxy thread
    cpu: ?u16 = null, // the proxy thread's CPU (TF_DS_RDMA_CPUS), else wherever the scheduler puts it
    collect_peers: bool = false, // TF_DS_RDMA_COLLECT_PEERS: gather_into's copy-out by dsv41_rdma_collect_peers
};

/// What the other side's queue pairs toward this rank need (a device each), and where this rank's receive ring is.
pub const Info = extern struct {
    devices: u32,
    flags: u32, // the rings' switches (switchBits): every rank must run the same ones, or the ring's slots lay out another way
    qpn: [max_devices]u32,
    psn: [max_devices]u32,
    rkey: [max_devices]u32,
    gid: [max_devices][16]u8,
    recv: u64,
    gen: u64, // TF_DS_RING_WORD: this rank's generation-word region (0: none)
    gen_rkey: [max_devices]u32, // its registration on each device
    mac: [max_devices][6]u8, // TF_DS_RING_WAIT: the MAC of each device's network port (a DEVX queue pair is given its peer's, not resolved)
};

/// The kernel pair, from the family's rdma_gather fatbin (collect_peers, publish and await_send: null in a fatbin
/// without them).
pub const Kernels = struct {
    stage: cuda.Function,
    collect: cuda.Function,
    collect_into: cuda.Function,
    collect_peers: ?cuda.Function = null,
    collect_roci: ?cuda.Function = null,
    collect_into_roci: ?cuda.Function = null,
    collect_peers_roci: ?cuda.Function = null,
    publish: ?cuda.Function = null,
    await_send: ?cuda.Function = null,
    stage_wait: ?cuda.Function = null,
    publish_wait: ?cuda.Function = null,
    await_send_wait: ?cuda.Function = null,
    // TF_DS_ROCI_XLOG's variants of the three kernels that log (rdma_gather.cu, after the originals; null in an older fatbin). The host launches them
    // ONLY with the log on: the originals above carry no logging code, so with the log off the exchange runs the instructions it ran before the log.
    stage_wait_x: ?cuda.Function = null,
    publish_wait_x: ?cuda.Function = null,
    collect_peers_roci_x: ?cuda.Function = null,

    pub fn load(m: cuda.Module) cuda.Error!Kernels {
        return .{ .stage = try m.function("dsv41_rdma_stage"), .collect = try m.function("dsv41_rdma_collect"), .collect_into = try m.function("dsv41_rdma_collect_into"), .collect_peers = m.function("dsv41_rdma_collect_peers") catch null, .collect_roci = m.function("dsv41_rdma_collect_roci") catch null, .collect_into_roci = m.function("dsv41_rdma_collect_into_roci") catch null, .collect_peers_roci = m.function("dsv41_rdma_collect_peers_roci") catch null, .publish = m.function("dsv41_rdma_publish") catch null, .await_send = m.function("dsv41_rdma_await_send") catch null, .stage_wait = m.function("dsv41_rdma_stage_wait") catch null, .publish_wait = m.function("dsv41_rdma_publish_wait") catch null, .await_send_wait = m.function("dsv41_rdma_await_send_wait") catch null, .stage_wait_x = m.function("dsv41_rdma_stage_wait_x") catch null, .publish_wait_x = m.function("dsv41_rdma_publish_wait_x") catch null, .collect_peers_roci_x = m.function("dsv41_rdma_collect_peers_roci_x") catch null };
    }

    /// Whether TF_DS_RING_WAIT can run: its three kernels (the staging, publish and await_send kernels reading the completion queues) are in the fatbin.
    pub fn hasWait(k: Kernels) bool {
        return k.stage_wait != null and k.publish_wait != null and k.await_send_wait != null;
    }

    /// The kernel to launch for a plain one and its logging variant: the variant when TF_DS_ROCI_XLOG is on and the fatbin has it, else the plain one.
    fn pick(plain: ?cuda.Function, x: ?cuda.Function) struct { f: ?cuda.Function, logging: bool } {
        if (xlog) if (x) |xf| return .{ .f = xf, .logging = true };
        return .{ .f = plain, .logging = false };
    }

    /// Whether the in-place path can run: both of its kernels are in the fatbin.
    pub fn hasInplace(k: Kernels) bool {
        return k.publish != null and k.await_send != null;
    }

    /// Whether the generation-word path can run: the collecting kernels it waits in are in the fatbin.
    pub fn hasRoci(k: Kernels) bool {
        return k.collect_roci != null and k.collect_into_roci != null and k.collect_peers_roci != null;
    }
};

/// gather_into's destinations as the collecting kernels take them (rdma_gather.cu Dsts): rank p's slice, rows of row4[p]
/// float4 packed in its ring slot, copied to rows stride4[p] float4 apart from ptr[p] (0: not copied out; row4 0: zero
/// width, not copied out either).
pub const Dsts = extern struct { ptr: [8]u64 = @splat(0), stride4: [8]i64 = @splat(0), row4: [8]c_int = @splat(0) };

/// What a collecting kernel reads besides the destinations: the receive ring (its device address), the flags [slots]
/// [world], the sequence word, the abort word and the probe (device addresses; probe 0: none), this rank and the
/// world; with TF_DS_RING_WORD the generation words [peer][lane], the lanes toward each peer, the first expired
/// wait's record and the wait's deadline instead of the flags.
pub const CollectRing = struct {
    recv: DevicePtr,
    flags: DevicePtr,
    seq: DevicePtr,
    abort: DevicePtr,
    probe: DevicePtr = 0,
    rank: u32,
    world: u32,
    gen: DevicePtr = 0,
    lanes: DevicePtr = 0,
    fault: DevicePtr = 0,
    deadline_ns: u64 = 0,
    roci: Roci = .off,
    /// This rank's own slice, copied by dsv41_rdma_collect_peers_roci instead of the publishing kernel (own_dst 0: none): where it is read from,
    /// where it lands (rows of own_row4 float4, own_stride4 apart), and its float4.
    own_src: DevicePtr = 0,
    own_dst: DevicePtr = 0,
    own_row4: u32 = 0,
    own_stride4: u64 = 0,
    own_n4: u32 = 0,
};

/// Whether a gather's copy-out waits on the lanes' generation words: the ring's own signal (CollectRing.roci), which is off when
/// connect refused the setting, so no kernel is handed words that do not exist.
fn collectsOnWords(s: Settings, ring: CollectRing) bool {
    _ = s;
    return ring.roci.on();
}

/// A diagnostic log's file (TF_DS_ROCI_XLOG, TF_DS_NACT_LOG): `text` to <dir named by the environment variable `dir_env`, else the folder's log
/// directory $HOME/.cache/dsv41-tensorfold/logs>/<name>, the directory and its parents made if missing. True when written; a failure is a log
/// line, never an error: these are measurement aids.
pub fn diagWriteFile(dir_env: [:0]const u8, name: []const u8, text: []const u8) bool {
    var dirbuf: [512]u8 = undefined;
    const dir: []const u8 = if (std.c.getenv(dir_env.ptr)) |v| std.mem.span(v) else blk: {
        const home = std.c.getenv("HOME") orelse "/tmp";
        break :blk std.fmt.bufPrint(&dirbuf, "{s}/.cache/dsv41-tensorfold/logs", .{std.mem.span(home)}) catch "/tmp";
    };
    var pathbuf: [640:0]u8 = undefined;
    var i: usize = 1;
    while (i <= dir.len) : (i += 1) {
        if (i == dir.len or dir[i] == '/') {
            const part = std.fmt.bufPrintSentinel(&pathbuf, "{s}", .{dir[0..i]}, 0) catch return false;
            _ = std.c.mkdir(part.ptr, 0o755);
        }
    }
    const path = std.fmt.bufPrintSentinel(&pathbuf, "{s}/{s}", .{ dir, name }, 0) catch return false;
    const f = std.c.fopen(path.ptr, "w") orelse {
        std.log.warn("{s}: cannot write {s}", .{ dir_env, path });
        return false;
    };
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(text.ptr, 1, text.len, f);
    return true;
}

/// The exchange log's file name for a ring: `roci-xlog-rank<R>.txt` for the four nodes' ring (the decode exchanges' and the report's default), a suffix for
/// the column and row rings (their sequences are their own, and three rings writing one name overwrote one another).
pub fn xlogFileName(buf: []u8, rank: u32, ring_name: []const u8) ?[]const u8 {
    const sfx: []const u8 = if (std.mem.indexOf(u8, ring_name, "four") != null) "" else if (std.mem.indexOf(u8, ring_name, "column") != null) "-col" else if (std.mem.indexOf(u8, ring_name, "row") != null) "-row" else "-other";
    return std.fmt.bufPrint(buf, "roci-xlog-rank{d}{s}.txt", .{ rank, sfx }) catch null;
}

/// TF_DS_ROCI_XLOG=1 (read at Model.open; log only, off by default): the peers kernel logs, in a ring of `xlog_records`, the sequence and the
/// time (its GPU's globaltimer) at which each peer's words were all in. A capture then matches every exchange across ranks by sequence
/// instead of by time; read it with Ring.xlogRead after the stream is synchronized.
pub var xlog: bool = false;
pub const xlog_records = 8192; // the doorbell log's records: one a sequence, 8,192 sequences (~48 decode rounds)
/// The receive log's records, four words each: indexed (sequence * 8 + peer) & (xlog_recv_records - 1), so it holds the last xlog_recv_records / 8 = 8,192 SEQUENCES (~48 decode
/// rounds at 169 exchanges), as many as the doorbell log. 1,024 (the first size: 8,192 records) left the ranks' files without a common sequence: each rank's file is written at
/// its own time (every 256th call or 3 s), rank 0's ~1,330 exchanges before the followers', so the receive windows of 1,024 sequences did not overlap and the report matched none (#398).
pub const xlog_recv_records = 65536; // 2 MiB of the fault buffer
/// This rank's own doorbells (the sender side, one per exchange): the sequence and this GPU's clock just after the doorbell store, a ring of
/// `xlog_records` in the fault buffer behind the receive records (xlog_recv_records); written by the card-WAIT staging / publishing kernels (their `probe` argument,
/// tagged with bit 0, points at it). A peer's words-in time minus this, once the two GPUs' clocks are fitted, is the wire.
pub const XlogPub = struct { seq: u64, at_ns: u64 };

pub fn xlogPubDecode(words: []const u64, out: []XlogPub) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i + 1 < words.len and n < out.len) : (i += 2) {
        if (words[i + 1] == 0) continue;
        out[n] = .{ .seq = words[i], .at_ns = words[i + 1] };
        n += 1;
    }
    return n;
}

pub const XlogRec = struct { seq: u64, peer: u32, at_ns: u64, start_ns: u64 = 0, end_ns: u64 = 0 };

/// The log's records from the device words (four a record: sequence << 3 | peer, the clock when the peer's words were all in, the kernel's
/// start, and the end of the last block's copy-out; a slot never written has clock 0), in ring
/// order, up to out.len. The kernel's layout is the only reader of this: rdma_gather.cu (collect_peers_roci).
pub fn xlogDecode(words: []const u64, out: []XlogRec) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i + 3 < words.len and n < out.len) : (i += 4) {
        if (words[i + 1] == 0) continue;
        out[n] = .{ .seq = words[i] >> 3, .peer = @truncate(words[i] & 7), .at_ns = words[i + 1], .start_ns = words[i + 2], .end_ns = words[i + 3] };
        n += 1;
    }
    return n;
}

/// One rank's log file: a header line naming the rank, the world, the ring and its slot count (so four files pair on the sequence with no other
/// context), then one line a record sorted by sequence and peer: `seq peer words_in_ns start_ns end_ns` (this rank's GPU globaltimer, a clock of its own; end_ns 0: not written).
pub fn xlogRender(gpa: std.mem.Allocator, rank: u32, world: u32, ring: []const u8, slots: u32, recs: []XlogRec) ![]u8 {
    return xlogRenderPub(gpa, rank, world, ring, slots, recs, &.{});
}

/// xlogRender with this rank's doorbell records after the receive records, one `P seq at_ns` line each, sorted by sequence (the header says how many).
pub fn xlogRenderPub(gpa: std.mem.Allocator, rank: u32, world: u32, ring: []const u8, slots: u32, recs: []XlogRec, pubs: []XlogPub) ![]u8 {
    std.mem.sort(XlogPub, pubs, {}, struct {
        fn less(_: void, a: XlogPub, b: XlogPub) bool {
            return a.seq < b.seq;
        }
    }.less);
    std.mem.sort(XlogRec, recs, {}, struct {
        fn less(_: void, a: XlogRec, b: XlogRec) bool {
            return if (a.seq != b.seq) a.seq < b.seq else a.peer < b.peer;
        }
    }.less);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    if (pubs.len == 0) {
        try out.print(gpa, "# TF_DS_ROCI_XLOG v1 rank={d} world={d} ring=\"{s}\" slots={d} records={d} clock=globaltimer_ns\n", .{ rank, world, ring, slots, recs.len });
    } else try out.print(gpa, "# TF_DS_ROCI_XLOG v1 rank={d} world={d} ring=\"{s}\" slots={d} records={d} pubs={d} clock=globaltimer_ns\n", .{ rank, world, ring, slots, recs.len, pubs.len });
    for (recs) |q| try out.print(gpa, "{d} {d} {d} {d} {d}\n", .{ q.seq, q.peer, q.at_ns, q.start_ns, q.end_ns });
    for (pubs) |q| try out.print(gpa, "P {d} {d}\n", .{ q.seq, q.at_ns });
    return out.toOwnedSlice(gpa);
}

/// gather_into's copy-out on `stream`, after the staging kernel: with `s.roci` the generation-word kernels (whose
/// wait is the lanes' words, bounded); else dsv41_rdma_collect_peers when `s.collect_peers` and the fatbin has it (on
/// peersGrid's grid; no probe), else dsv41_rdma_collect_into on the baseline grid (the float4 copied out, `copy4`, a block
/// for each collect_grain x 256 of them, up to collect_blocks). Every path copies the same bytes.
pub fn launchCollect(k: Kernels, s: Settings, stream: cuda.Stream, d: Dsts, rows: u32, ring: CollectRing, copy4: u64) Error!void {
    if (collectsOnWords(s, ring)) {
        const w = ring;
        const pr = Kernels.pick(k.collect_peers_roci, k.collect_peers_roci_x);
        if (s.collect_peers) if (pr.f) |f| {
            var row4s: [max_ranks]u32 = @splat(0);
            for (0..w.world) |p| row4s[p] = if (peerCopied(d, @intCast(p), w.rank)) @intCast(d.row4[p]) else 0;
            const g = peersGrid(rows, row4s[0..w.world], w.rank, s.collect_blocks);
            var b: cuda.Args = .{};
            b.add(d);
            b.add(@as(c_int, @intCast(rows)));
            b.add(w.recv);
            b.add(w.gen);
            b.add(w.lanes);
            b.add(w.seq);
            b.add(@as(c_int, @intCast(w.rank)));
            b.add(@as(c_int, @intCast(w.world)));
            b.add(@as(c_int, @intCast(s.slots)));
            b.add(@as(u64, s.max_bytes));
            b.add(w.abort);
            b.add(w.deadline_ns);
            b.add(w.fault | @intFromBool(pr.logging)); // (the logging variant takes the address tagged with bit 0 and strips it)
            b.add(@as(c_int, @intFromBool(s.pdl)));
            b.add(w.own_src);
            b.add(w.own_dst);
            b.add(@as(c_int, @intCast(w.own_row4)));
            b.add(@as(c_longlong, @intCast(w.own_stride4)));
            b.add(@as(c_int, @intCast(w.own_n4)));
            if (b.count != 19) return error.BadInfo; // dsv41_rdma_collect_peers_roci's 19 parameters
            return cuda.launch.launch(f, .{ .grid = .{ .x = g.x, .y = g.y, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = s.pdl }, stream, &b);
        };
        const f = k.collect_into_roci orelse return error.BadInfo;
        var a: cuda.Args = .{};
        a.add(d);
        a.add(@as(c_int, @intCast(rows)));
        a.add(w.recv);
        a.add(w.gen);
        a.add(w.lanes);
        a.add(w.seq);
        a.add(@as(c_int, @intCast(w.rank)));
        a.add(@as(c_int, @intCast(w.world)));
        a.add(@as(c_int, @intCast(s.slots)));
        a.add(@as(u64, s.max_bytes));
        a.add(w.probe);
        a.add(w.abort);
        a.add(w.deadline_ns);
        a.add(w.fault);
        a.add(@as(c_int, @intFromBool(s.pdl)));
        if (a.count != 15) return error.BadInfo; // dsv41_rdma_collect_into_roci's 15 parameters
        const per = collect_threads * s.collect_grain;
        const blocks = std.math.clamp((copy4 + per - 1) / per, 1, s.collect_blocks);
        return cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = s.pdl }, stream, &a);
    }
    if (s.collect_peers) if (k.collect_peers) |f| {
        // every peer's slot at once: a row of blocks a peer copied out
        var row4s: [max_ranks]u32 = @splat(0);
        for (0..ring.world) |p| row4s[p] = if (peerCopied(d, @intCast(p), ring.rank)) @intCast(d.row4[p]) else 0;
        const g = peersGrid(rows, row4s[0..ring.world], ring.rank, s.collect_blocks);
        var b: cuda.Args = .{};
        b.add(d);
        b.add(@as(c_int, @intCast(rows)));
        b.add(ring.recv);
        b.add(ring.flags);
        b.add(ring.seq);
        b.add(@as(c_int, @intCast(ring.rank)));
        b.add(@as(c_int, @intCast(ring.world)));
        b.add(@as(c_int, @intCast(s.slots)));
        b.add(@as(u64, s.max_bytes));
        b.add(ring.abort);
        b.add(@as(c_int, @intFromBool(s.pdl)));
        return cuda.launch.launch(f, .{ .grid = .{ .x = g.x, .y = g.y, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = s.pdl }, stream, &b);
    };
    var a: cuda.Args = .{};
    a.add(d);
    a.add(@as(c_int, @intCast(rows)));
    a.add(ring.recv);
    a.add(ring.flags);
    a.add(ring.seq);
    a.add(@as(c_int, @intCast(ring.rank)));
    a.add(@as(c_int, @intCast(ring.world)));
    a.add(@as(c_int, @intCast(s.slots)));
    a.add(@as(u64, s.max_bytes));
    a.add(ring.probe);
    a.add(ring.abort);
    a.add(@as(c_int, @intFromBool(s.pdl)));
    const per = collect_threads * s.collect_grain;
    const blocks = std.math.clamp((copy4 + per - 1) / per, 1, s.collect_blocks);
    try cuda.launch.launch(k.collect_into, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = s.pdl }, stream, &a);
}

/// dsv41_rdma_collect_peers' grid: y a row of blocks for each peer copied out, x blocks a row.
pub const PeersGrid = struct { x: u32, y: u32 };

/// TF_DS_RDMA_COLLECT_PEERS' grid for a gather_into of `rows` rows: a row of blocks (y) for each peer copied out
/// (row4[p] > 0 and p not `rank`), and as many blocks a row (x) as the widest of them has units (a row's 1,024 float4 a
/// unit: the kernel's PEER_UNIT4), up to `collect_blocks` shared out among those peers, at least one each; a grid of one
/// when no peer is copied out.
pub fn peersGrid(rows: u32, row4: []const u32, rank: u32, collect_blocks: u32) PeersGrid {
    const unit4: u32 = 4 * collect_threads;
    var peers: u32 = 0;
    var units: u32 = 0;
    for (row4, 0..) |w, p| {
        if (p == rank or w == 0) continue;
        peers += 1;
        units = @max(units, rows * ((w + unit4 - 1) / unit4));
    }
    if (peers == 0) return .{ .x = 1, .y = 1 };
    return .{ .x = std.math.clamp(units, 1, @max(1, collect_blocks / peers)), .y = peers };
}

/// Whether the collecting kernels copy rank `p`'s slot out: a destination pointer and a nonzero row width, and not
/// this rank's own slice. The host's grid (peersGrid) and dsv41_rdma_collect_peers' block row selection both use this
/// rule; the kernel's condition in rdma_gather.cu must match it.
pub fn peerCopied(d: Dsts, p: u32, rank: u32) bool {
    return p != rank and d.ptr[p] != 0 and d.row4[p] != 0;
}

/// The peer dsv41_rdma_collect_peers' block row `y` copies: the y-th rank p in order with peerCopied(d, p, rank); null
/// when the grid's rows are exhausted. Mirrors the kernel's `p` selection (rdma_gather.cu).
pub fn peerAtRow(d: Dsts, world: u32, rank: u32, y: u32) ?u32 {
    var left = y;
    var p: u32 = 0;
    while (p < world) : (p += 1) {
        if (!peerCopied(d, p, rank)) continue;
        if (left == 0) return p;
        left -= 1;
    }
    return null;
}

/// cuMemHostRegister / cuMemHostUnregister, which the runtime's driver table does not hold.
pub const HostReg = struct {
    lib: std.DynLib,
    register: *const fn (?*anyopaque, usize, c_uint) callconv(.c) cuda.abi.Result,
    unregister: *const fn (?*anyopaque) callconv(.c) cuda.abi.Result,

    pub fn open() Error!HostReg {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.NoHostRegister;
        errdefer lib.close();
        return .{
            .lib = lib,
            .register = lib.lookup(*const fn (?*anyopaque, usize, c_uint) callconv(.c) cuda.abi.Result, "cuMemHostRegister_v2") orelse return error.NoHostRegister,
            .unregister = lib.lookup(*const fn (?*anyopaque) callconv(.c) cuda.abi.Result, "cuMemHostUnregister") orelse return error.NoHostRegister,
        };
    }
};

extern "c" fn mlock(addr: *const anyopaque, len: usize) c_int;

/// Zeroed pages CUDA maps for the device. `alloc` takes anonymous pages, mlock keeps them resident (best effort) and
/// cuMemHostRegister makes them mapped and portable. `allocHost` takes them from cudaHostAlloc instead: with `wc` it
/// asks for write-combined pages, which a kernel writes and the NIC reads (the GPU reads them too; a CPU read of them
/// is uncached and slow, so only the rings' data slots live here, never a flag, size, sent word or the doorbell).
pub const Pinned = struct {
    mem: []align(std.heap.page_size_min) u8,
    dev: DevicePtr,
    host: bool = false, // cudaHostAlloc memory: free it with cuMemFreeHost, not munmap + cuMemHostUnregister
    d: ?*const cuda.Driver = null,

    pub fn alloc(d: *const cuda.Driver, hr: *const HostReg, len: usize) Error!Pinned {
        const n = std.mem.alignForward(usize, len, 1 << 16);
        const mem = try std.posix.mmap(null, n, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        errdefer std.posix.munmap(mem);
        @memset(mem, 0);
        _ = mlock(mem.ptr, mem.len);
        try d.check(hr.register(mem.ptr, mem.len, host_register_portable | host_register_devicemap), "cuMemHostRegister");
        var dev: DevicePtr = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, mem.ptr, 0), "cuMemHostGetDevicePointer");
        return .{ .mem = mem, .dev = dev };
    }

    /// Mapped, pinned pages from cudaHostAlloc; `wc` adds WriteCombined. Zeroed like `alloc` (the allocation's bytes are
    /// not defined), by a CPU store: write-combined pages take it slowly, once, before anything reads them.
    pub fn allocHost(d: *const cuda.Driver, len: usize, wc: bool) Error!Pinned {
        const n = std.mem.alignForward(usize, len, 1 << 16);
        var p: ?*anyopaque = null;
        try d.check(d.api.cuMemHostAlloc(&p, n, cuda.abi.host_alloc_devicemap | (if (wc) cuda.abi.host_alloc_writecombined else 0)), "cuMemHostAlloc");
        // a page-aligned allocation is what every data slot's type carries (an unaligned one is refused, not trapped)
        if (@intFromPtr(p.?) % std.heap.page_size_min != 0) {
            _ = d.api.cuMemFreeHost(p.?);
            return error.BadInfo;
        }
        const base: [*]align(std.heap.page_size_min) u8 = @ptrCast(@alignCast(p.?));
        const mem = base[0..n];
        @memset(mem, 0);
        var dev: DevicePtr = 0;
        d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, mem.ptr, 0), "cuMemHostGetDevicePointer") catch |e| {
            _ = d.api.cuMemFreeHost(mem.ptr);
            return e;
        };
        return .{ .mem = mem, .dev = dev, .host = true, .d = d };
    }

    pub fn free(p: *Pinned, hr: *const HostReg) void {
        if (p.host) {
            _ = p.d.?.api.cuMemFreeHost(p.mem.ptr);
            p.* = undefined;
            return;
        }
        _ = hr.unregister(p.mem.ptr);
        std.posix.munmap(p.mem);
    }
};

/// One RDMA device: its context, protection domain, one completion queue for every QP, the rings' registrations and an
/// RC queue pair per peer (only those on a route leave INIT).
const Nic = struct {
    ctx: *vabi.Context,
    pd: *vabi.Pd,
    cq: *vabi.Cq,
    mr_send: *vabi.Mr,
    mr_recv: *vabi.Mr,
    mr_gen: ?*vabi.Mr = null, // TF_DS_RING_WORD: the generation words, written by the peers' NICs
    mr_meta: ?*vabi.Mr = null, // TF_DS_RING_WAIT: the doorbell's memory (the card's WAIT reads it)
    mr_sink: ?*vabi.Mr = null, // TF_DS_RING_WAIT with the atomic signal: where the atomics' old values land
    mac: [6]u8 = @splat(0), // TF_DS_RING_WAIT: this port's MAC
    gid: [16]u8,
    gid_index: u8,
    mtu: c_int = mtu_4096, // the port's active MTU (ibv_query_port), capped at 4096: the QPs' path MTU
    atomic_cap: c_int = vabi.atomic_glob, // ibv_query_device's ibv_atomic_cap
    max_rd_atom: c_int = 16, // ibv_query_device's max_qp_rd_atom (ATOMICs in flight a responder grants a queue pair) and max_qp_init_rd_atom (a requester)
    max_init_rd_atom: c_int = 16,
    inline_ok: bool = true, // the queue pairs granted room for the 8-byte generation word
    name: []const u8 = "",
    used: bool = false, // some route runs on it (the proxy polls its queue)
    qps: [max_ranks]?*vabi.Qp = @splat(null),
    psns: [max_ranks]u32 = @splat(0),
};

/// A peer: the lanes toward it (this rank's device and the peer's device each lane pairs, the peer's key and queue
/// pair there) and its receive ring; or, with no lane, the ranks that relay to it and from it.
const Peer = struct {
    lanes: u32 = 0,
    nic: [max_devices]u8 = @splat(0), // this rank's device of each lane
    qpn: [max_devices]u32 = @splat(0),
    psn: [max_devices]u32 = @splat(0),
    rkey: [max_devices]u32 = @splat(0),
    gid: [max_devices][16]u8 = @splat(@splat(0)),
    recv: u64 = 0,
    gen: u64 = 0, // TF_DS_RING_WORD: the peer's generation-word region
    gen_rkey: [max_devices]u32 = @splat(0), // its registration on the device of each lane
    mac: [max_devices][6]u8 = @splat(@splat(0)), // TF_DS_RING_WAIT: the MAC of its port on the device of each lane
    relayers: [max_ranks]u8 = @splat(0), // relayed: the relaying ranks, in rank order (relayer k carries half k)
    n_relayers: u32 = 0,
};

/// One relay this rank runs each sequence: half `half` of `of` halves of rank `src`'s slot on to rank `dst`, on lane
/// `lane` toward it.
const Job = struct { src: u8, dst: u8, half: u8, of: u8, lane: u8 };

/// The IPv4 address in a RoCE v2 IPv4 GID (::ffff:a.b.c.d), else null.
fn gidIp(g: [16]u8) ?u32 {
    for (g[0..10]) |b| if (b != 0) return null;
    if (g[10] != 0xff or g[11] != 0xff) return null;
    return std.mem.readInt(u32, g[12..16], .big);
}

/// Whether sysfs calls device `name`'s GID `index` RoCE v2 (/sys/class/infiniband/NAME/ports/1/gid_attrs/types/INDEX).
fn sysfsV2(name: []const u8, index: u32) bool {
    var path_buf: [256]u8 = undefined;
    const path = std.mem.printSentinel(&path_buf, "/sys/class/infiniband/{s}/ports/{d}/gid_attrs/types/{d}", .{ name, port_num, index }, 0) catch return false;
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0) catch return false;
    defer _ = std.posix.system.close(fd);
    var buf: [64]u8 = undefined;
    const n = std.posix.read(fd, &buf) catch return false;
    return std.mem.indexOf(u8, buf[0..n], "v2") != null;
}

fn sameNet(a: u32, b: u32, prefix: u8) bool {
    if (prefix == 0) return true;
    if (prefix >= 32) return a == b;
    const sh: u5 = @intCast(32 - @as(u32, prefix));
    return (a >> sh) == (b >> sh);
}

/// The lanes between two ranks' devices: in each subnet, the k-th device of one with the k-th of the other (in list
/// order), so both ends pair the same devices. `out[k]` = (index in `mine`, index in `theirs`); null: no IPv4 GIDs, so
/// device d pairs with device d (the switch's layout).
fn pairLanes(mine: []const [16]u8, theirs: []const [16]u8, prefix: u8, out: *[max_devices][2]u8) ?u32 {
    for (mine) |g| if (gidIp(g) == null) return null;
    for (theirs) |g| if (gidIp(g) == null) return null;
    var n: u32 = 0;
    var taken: [max_devices]bool = @splat(false);
    for (mine, 0..) |g, i| {
        const ip = gidIp(g).?;
        // this device's place among this rank's devices on its subnet
        var k: usize = 0;
        for (mine[0..i]) |h| if (sameNet(gidIp(h).?, ip, prefix)) {
            k += 1;
        };
        var seen: usize = 0;
        for (theirs, 0..) |t, j| {
            if (!sameNet(gidIp(t).?, ip, prefix)) continue;
            if (seen == k and !taken[j]) {
                taken[j] = true;
                out[n] = .{ @intCast(i), @intCast(j) };
                n += 1;
                break;
            }
            seen += 1;
        }
    }
    return n;
}

fn linked(a: Info, b: Info, prefix: u8) bool {
    var out: [max_devices][2]u8 = undefined;
    const n = pairLanes(a.gid[0..a.devices], b.gid[0..b.devices], prefix, &out) orelse return a.devices == b.devices;
    return n > 0;
}

/// The decode ring's progress watch: a posted sequence whose peers' parts do not all come within this many milliseconds
/// fails the ring by name (TF_DS_RING_DEAD_MS).
pub const default_dead_ms: u64 = 10_000;

/// TF_DS_RING_DEAD_MS's value: the default when unset, a whole number of milliseconds from 1 when set, else null (the
/// start refuses by name: a typo must never become the default unseen).
pub fn deadMsFrom(text: ?[]const u8) ?u64 {
    const t = text orelse return default_dead_ms;
    const v = std.fmt.parseInt(u64, t, 10) catch return null;
    return if (v == 0) null else v;
}

/// The UDP source port of RoCE v2's flow entropy for a queue pair: from its number, the device and the peer, in the dynamic port range.
fn udpSport(qpn: u32, nic: usize, peer: usize) u16 {
    const h = (qpn *% 2654435761) ^ @as(u32, @intCast(nic * 7 + peer));
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

/// The MAC of the network device behind an RDMA device (`/sys/class/infiniband/<dev>/device/net/<netdev>/address`, as m19c_wait.cu's netdev_mac finds it).
fn netdevMac(device: []const u8) error{NoMac}![6]u8 {
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

/// The atomic arrival signal's start-up check for one device: why it cannot run, or null. The card must do ATOMICs at all (atomic_cap above none: HCA,
/// the CX-7's, or global) and grant a queue pair at least one in flight each way (rocinante_verbs.signalRefusal's check).
pub fn atomicRefusal(buf: []u8, device: []const u8, atomic_cap: c_int, max_qp_rd_atom: c_int, max_qp_init_rd_atom: c_int) ?[]const u8 {
    if (atomic_cap < vabi.atomic_hca)
        return std.fmt.bufPrint(buf, "device {s} reports atomic_cap {d}: it does no ATOMIC operations", .{ device, atomic_cap }) catch "the device does no ATOMIC operations";
    if (max_qp_rd_atom < 1 or max_qp_init_rd_atom < 1)
        return std.fmt.bufPrint(buf, "device {s} grants a queue pair {d} responder and {d} requester ATOMICs in flight: it needs at least one each", .{ device, max_qp_rd_atom, max_qp_init_rd_atom }) catch "the device grants no ATOMICs in flight";
    return null;
}

/// The ATOMICs in flight a ring's queue pair is set up for: its slots, capped by both limits of the card, at least one.
pub fn atomicDepth(slots: u32, max_qp_rd_atom: c_int, max_qp_init_rd_atom: c_int) u8 {
    const cap: c_int = @min(max_qp_rd_atom, max_qp_init_rd_atom, @as(c_int, @intCast(@min(slots, 255))));
    return @intCast(@max(cap, 1));
}

/// A refusal's first words: the switch as it was set.
pub fn waitRefusalHead(buf: []u8, explicit: bool) []const u8 {
    return std.fmt.bufPrint(buf, "{s} refused", .{if (explicit) "TF_DS_RING_WAIT=1" else "TF_DS_RING_WAIT (unset: the default)"}) catch "TF_DS_RING_WAIT refused";
}

/// What a device answered about atomics.
pub const AtomicCaps = struct { atomic_cap: c_int, max_qp_rd_atom: c_int, max_qp_init_rd_atom: c_int };

/// Why the atomic signal cannot run on these devices (the first that cannot), for the start's fallback to the WRITE / inline word; null only when every
/// device was opened, asked, and can. A device that cannot be opened or asked, or a verbs library that does not load, is a card that cannot: the
/// reason names the device and the error. `src` answers `caps(name) !AtomicCaps` (the real verbs, or a card model in the tests).
pub fn probeAtomicWith(buf: []u8, names: []const []const u8, src: anytype) ?[]const u8 {
    for (names) |name| {
        const c = src.caps(name) catch |e|
            return std.fmt.bufPrint(buf, "device {s} cannot be asked about atomics ({s})", .{ name, @errorName(e) }) catch "a device cannot be asked about atomics";
        if (atomicRefusal(buf, name, c.atomic_cap, c.max_qp_rd_atom, c.max_qp_init_rd_atom)) |why| return why;
    }
    return null;
}

const VerbsCaps = struct {
    v: *const Verbs,
    fn caps(self: VerbsCaps, name: []const u8) !AtomicCaps {
        const ctx = try self.v.openDevice(name);
        defer _ = self.v.api.ibv_close_device(ctx);
        var a: vabi.DeviceAttr = undefined;
        if (self.v.api.ibv_query_device(ctx, &a) != 0) return error.QueryFailed;
        return .{ .atomic_cap = a.atomic_cap, .max_qp_rd_atom = a.max_qp_rd_atom, .max_qp_init_rd_atom = a.max_qp_init_rd_atom };
    }
};

pub fn probeAtomic(buf: []u8, names: []const []const u8) ?[]const u8 {
    var v = Verbs.open(null) catch |e|
        return std.fmt.bufPrint(buf, "the verbs library does not load ({s})", .{@errorName(e)}) catch "the verbs library does not load";
    defer v.close();
    return probeAtomicWith(buf, names, VerbsCaps{ .v = &v });
}

pub const Ring = struct {
    d: *const cuda.Driver,
    verbs: Verbs,
    v: *const Verbs = undefined,
    k: Kernels,
    s: Settings,
    rank: u32,
    world: u32,
    hr: HostReg,
    send: Pinned,
    recv: Pinned,
    meta: Pinned,
    dead_ms: u64 = default_dead_ms, // TF_DS_RING_DEAD_MS: the progress watch's limit, read once at create
    gen: ?Pinned = null, // TF_DS_RING_WORD: the generation words [peer][lane] and the proxy's local word
    lanes_dev: ?cuda.DeviceBuffer = null, // the lanes toward each peer, the waiting kernel's [world] u32
    fault: ?cuda.DeviceBuffer = null, // the first expired generation-word wait: magic, seq, rank, peer/lane, seen, waited, when, why
    roci_live: bool = false, // the switch is on and this ring can run it (a lane to every peer, the kernels, the atomics)
    roci_why: [96]u8 = @splat(0), // why it cannot, when it was asked for
    roci_order: c_int = -1, // ibv_query_qp_data_in_order(RDMA_WRITE) on the lanes, the most conservative answer (-1: not asked)
    mlx: ?mx.Mlx5 = null, // TF_DS_RING_WAIT: libmlx5, loaded by dlopen
    cd: ?*wait.Backend = null, // TF_DS_RING_WAIT: the queue pairs that wait in the card
    cd_tab: wait.CqTab = .{}, // TF_DS_RING_WAIT: the staging kernels' credit table (one wait.CqRef a lane), passed to them by value
    cd_n: u32 = 0, // its entries
    cd_sink: ?Pinned = null, // the atomic signal's landing words [peer][lane] (host memory, local write): the old value of each lane's last atomic
    scalars: cuda.DeviceBuffer, // seq (u64), staged (u32), probe (8 x u64)
    nics: [max_devices]Nic = undefined,
    nd: u32 = 0,
    peers: [max_ranks]Peer = @splat(.{}),
    jobs: [2 * max_ranks]Job = undefined, // the relays this rank runs (on a ring of four: two)
    n_jobs: u32 = 0,
    proxy: ?std.Thread = null,
    idle_gate: IdleGate = .{},
    failure_peer: ?*Ring = null, // the ring whose failure this one follows (neighbor pairs): fail() fails both
    running: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    why: [128]u8 = @splat(0),
    streams: [max_users]?cuda.abi.Stream = @splat(null), // the streams an in-place send slot was given to
    n_users: u32 = 0,

    // the metadata block: flags [slots][world], doorbell, sizes [slots], sent [slots], abort word, then the in-place
    // words: send_q [slots] (the sequence a slot's send_slot names), send_slot [slots], send_last [max_users] (the
    // sequence a producer last published) and send_done [max_users] (the sequence the proxy last sent)
    fn flags(r: *const Ring) [*]u64 {
        return @ptrCast(@alignCast(r.meta.mem.ptr));
    }
    fn doorbell(r: *const Ring) *u64 {
        return &r.flags()[r.s.slots * r.world];
    }
    fn sizes(r: *const Ring) [*]u64 {
        return r.flags() + r.s.slots * r.world + 1;
    }
    fn sent(r: *const Ring) [*]u64 {
        return r.sizes() + r.s.slots;
    }
    fn abortWord(r: *const Ring) *u64 {
        return &r.sent()[r.s.slots];
    }
    fn sendQ(r: *const Ring) [*]u64 {
        return r.flags() + r.s.slots * r.world + 1 + 2 * r.s.slots + 1;
    }
    fn sendSlotWord(r: *const Ring) [*]u64 {
        return r.sendQ() + r.s.slots;
    }
    fn sendLast(r: *const Ring) [*]u64 {
        return r.sendSlotWord() + r.s.slots;
    }
    fn sendDone(r: *const Ring) [*]u64 {
        return r.sendLast() + max_users;
    }
    /// The metadata block's words: the ring's own, then the in-place words (the allocation always has room for them,
    /// so their offsets never move and an older ring image's words keep their places).
    pub fn metaWords(r: *const Ring) usize {
        return r.s.slots * r.world + 1 + 2 * r.s.slots + 1 + 2 * r.s.slots + 2 * max_users;
    }
    /// The rings' switches as the peers compare them (Info.flags).
    pub fn switchBits(r: *const Ring) u32 {
        return @intFromBool(r.s.wc) | (@as(u32, @intFromBool(r.s.inplace)) << 1) |
            (@as(u32, @intFromBool(r.s.roci.on())) << 2) | (@as(u32, @intFromBool(r.s.roci == .atomic)) << 3) | (@as(u32, @intFromBool(r.s.wait)) << 4) | (@as(u32, r.cdDepth()) << 8);
    }

    /// The generation words: `peer * max_devices + lane` for each sender's lane, then the proxy's local word (the
    /// source of an inline write; the NICs of the peers never write it).
    fn genWords(r: *const Ring) [*]u64 {
        return @ptrCast(@alignCast(r.gen.?.mem.ptr));
    }

    /// The word the proxy writes inline: the sequence, written before the sequence's word writes are posted (the
    /// provider copies inline data when the work request is posted).
    fn rociWord(r: *const Ring) *u64 {
        return &r.genWords()[max_ranks * max_devices];
    }

    fn rociWordBytes(r: *const Ring) []const u8 {
        const p: [*]const u8 = @ptrCast(r.rociWord());
        return p[0..8];
    }
    /// The write region a staged send reads: slots 0 .. slots - 1. The in-place users' slots follow.
    fn sendBytes(r: *const Ring) usize {
        return (r.s.slots + (if (r.s.inplace) @as(usize, max_users) else 0)) * r.s.max_bytes;
    }
    /// The send slot of an in-place user (after the rotating slots).
    fn userSendSlot(r: *const Ring, u: u32) usize {
        return r.s.slots + u;
    }
    fn userOfHandle(r: *const Ring, h: cuda.abi.Stream) ?u32 {
        for (r.streams[0..r.n_users], 0..) |s, i| if (s == h) return @intCast(i);
        return null;
    }
    /// The in-place send slot `stream` owns, assigned on first use: a stream's producer always writes the same slot,
    /// so a captured graph's pointer stays right whatever sequence a replay reaches. Null when the switch is off.
    fn userSlot(r: *Ring, stream: cuda.Stream) Error!?u32 {
        if (!r.s.inplace) return null;
        if (r.userOfHandle(stream.handle)) |u| return u;
        if (r.n_users == max_users) return error.BadInfo;
        const u = r.n_users;
        r.streams[u] = stream.handle;
        r.n_users += 1;
        return u;
    }
    /// The device address of `stream`'s in-place send slot: a producer writes its partial there. Null when the
    /// switch is off or the exchange is larger than a slot (then the staged path must run).
    pub fn sendSlotDev(r: *Ring, stream: cuda.Stream, bytes: usize) Error!?DevicePtr {
        const u = try r.userSlot(stream) orelse return null;
        if (bytes > r.s.max_bytes) return null;
        return r.send.dev + @as(DevicePtr, @intCast(r.userSendSlot(u))) * r.s.max_bytes;
    }
    /// The in-place send slot already given to `stream` (no assignment): what a gather tests its source against.
    fn sendSlotHeld(r: *const Ring, stream: cuda.Stream, send: DevicePtr) ?u32 {
        if (!r.s.inplace) return null;
        const u = r.userOfHandle(stream.handle) orelse return null;
        if (send != r.send.dev + @as(DevicePtr, @intCast(r.userSendSlot(u))) * r.s.max_bytes) return null;
        return u;
    }
    fn metaDev(r: *const Ring, p: anytype) DevicePtr {
        return r.meta.dev + (@intFromPtr(p) - @intFromPtr(r.meta.mem.ptr));
    }

    /// Rings, metadata and queue pairs (in INIT, receives posted) on `devices`; `connect` with every rank's `info` next.
    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, k: Kernels, devices: []const []const u8, rank: u32, world: u32, s: Settings) Error!*Ring {
        if (world < 2 or world > max_ranks or rank >= world or devices.len == 0 or devices.len > max_devices) return error.BadInfo;
        if (s.slots < 2 or s.slots > seqs.max_slots or s.max_bytes == 0 or s.max_bytes % 64 != 0) return error.BadInfo;
        if (inplaceOff(s.wc, s.inplace) != null) return error.BadInfo; // the send slot must be write-combined
        if (s.roci.on() and s.roci_deadline_ns == 0) return error.BadInfo; // a lost word must end, never wait forever
        if (s.wait and (!s.roci.on() or !k.hasWait())) {
            // the chains end in the generation word's signal (the inline WRITE or the atomic) and the kernels read the completion queues: the start's
            // checks (Model.open) refuse first
            std.log.err("TF_DS_RING_WAIT=1: needs TF_DS_RING_WORD=1 and the wait kernels in the fatbin", .{});
            return error.BadInfo;
        }
        const dead_ms = deadMsFrom(if (std.c.getenv("TF_DS_RING_DEAD_MS")) |v| std.mem.span(v) else null) orelse {
            std.log.err("TF_DS_RING_DEAD_MS must be a whole number of milliseconds from 1 (the decode ring's progress watch; unset: {d})", .{default_dead_ms});
            return error.BadInfo;
        };
        const r = try gpa.create(Ring);
        errdefer gpa.destroy(r);
        var verbs = try Verbs.open(null);
        errdefer verbs.close();
        var hr = try HostReg.open();
        errdefer hr.lib.close();
        var mlx: ?mx.Mlx5 = null;
        if (s.wait) {
            mlx = mx.Mlx5.open() catch |e| {
                std.log.err("TF_DS_RING_WAIT=1: libmlx5 ({s}) is not usable: {s}", .{ mx.library, @errorName(e) });
                return error.BadInfo;
            };
        }
        errdefer if (mlx) |*m| m.close();
        var cd_sink: ?Pinned = null;
        if (s.wait and s.roci == .atomic) cd_sink = try Pinned.alloc(d, &hr, max_ranks * max_devices * 8);
        errdefer if (cd_sink) |*t| t.free(&hr);
        const send_bytes = (s.slots + (if (s.inplace) @as(usize, max_users) else 0)) * s.max_bytes;
        var send = if (s.wc) try Pinned.allocHost(d, send_bytes, true) else try Pinned.alloc(d, &hr, send_bytes);
        errdefer send.free(&hr);
        var recv = if (s.wc) try Pinned.allocHost(d, s.slots * world * s.max_bytes, true) else try Pinned.alloc(d, &hr, s.slots * world * s.max_bytes);
        errdefer recv.free(&hr);
        var meta = try Pinned.alloc(d, &hr, (s.slots * world + 1 + 2 * s.slots + 1 + 2 * s.slots + 2 * max_users) * 8);
        errdefer meta.free(&hr);
        var scalars = try cuda.DeviceBuffer.alloc(d, scalars_bytes);
        errdefer scalars.free();
        try scalars.fill8(0, null);
        var gen: ?Pinned = null;
        if (s.roci.on()) {
            gen = try Pinned.alloc(d, &hr, rociWords() * 8);
            errdefer gen.?.free(&hr);
        }
        var lanes_dev: ?cuda.DeviceBuffer = null;
        if (s.roci.on()) {
            lanes_dev = try cuda.DeviceBuffer.alloc(d, world * 4);
            errdefer lanes_dev.?.free();
            try lanes_dev.?.fill8(0, null);
        }
        var fault: ?cuda.DeviceBuffer = null;
        if (s.roci.on()) {
            fault = try cuda.DeviceBuffer.alloc(d, 64 + xlog_recv_records * 32 + xlog_records * 16);
            errdefer fault.?.free();
            try fault.?.fill8(0, null);
        }
        r.* = .{ .d = d, .verbs = verbs, .k = k, .s = s, .rank = rank, .world = world, .hr = hr, .send = send, .recv = recv, .meta = meta, .gen = gen, .lanes_dev = lanes_dev, .fault = fault, .scalars = scalars, .dead_ms = dead_ms, .mlx = mlx, .cd_sink = cd_sink };
        r.v = &r.verbs;
        errdefer for (r.nics[0..r.nd]) |*n| r.closeNic(n);
        for (devices) |name| {
            r.nics[r.nd] = try r.openNic(name);
            r.nd += 1;
        }
        if (s.wait) {
            wait.enabled = true;
            errdefer r.waitClose();
            try r.waitOpen(gpa);
        }
        return r;
    }

    fn openNic(r: *Ring, name: []const u8) Error!Nic {
        const api = r.v.api;
        const ctx = if (r.s.wait) try r.openDevx(name) else try r.v.openDevice(name);
        errdefer _ = api.ibv_close_device(ctx);
        const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dealloc_pd(pd);
        const cq = api.ibv_create_cq(ctx, cq_depth, null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq);
        const access = vabi.access_local_write | vabi.access_remote_write;
        const mr_send = api.ibv_reg_mr(pd, r.send.mem.ptr, r.send.mem.len, access) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dereg_mr(mr_send);
        const mr_recv = api.ibv_reg_mr(pd, r.recv.mem.ptr, r.recv.mem.len, access) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dereg_mr(mr_recv);
        var mr_gen: ?*vabi.Mr = null;
        if (r.s.roci.on()) {
            // the peers' NICs write these words: remote write, and with the atomic signal remote atomic
            const ga = access | (if (r.s.roci == .atomic) @as(c_int, vabi.access_remote_atomic) else 0);
            mr_gen = api.ibv_reg_mr(pd, r.gen.?.mem.ptr, r.gen.?.mem.len, ga) orelse return error.VerbsFailed;
            errdefer _ = api.ibv_dereg_mr(mr_gen.?);
        }
        var mr_sink: ?*vabi.Mr = null;
        if (r.s.wait and r.s.roci == .atomic) {
            mr_sink = api.ibv_reg_mr(pd, r.cd_sink.?.mem.ptr, r.cd_sink.?.mem.len, vabi.access_local_write) orelse return error.VerbsFailed;
            errdefer _ = api.ibv_dereg_mr(mr_sink.?);
        }
        var mr_meta: ?*vabi.Mr = null;
        if (r.s.wait) {
            // the card's WAIT reads the doorbell word: its memory is registered on this device
            mr_meta = api.ibv_reg_mr(pd, r.meta.mem.ptr, r.meta.mem.len, access) orelse return error.VerbsFailed;
            errdefer _ = api.ibv_dereg_mr(mr_meta.?);
        }
        var dev_attr: vabi.DeviceAttr = undefined;
        if (api.ibv_query_device(ctx, &dev_attr) != 0) return error.VerbsFailed;
        const gi = try r.pickGid(ctx, name);
        const gid = try r.v.gid(ctx, port_num, gi);
        // Path MTU from the port's active MTU, capped at 4096 (enum 5), instead of a fixed 4096: on a fabric whose
        // Ethernet MTU is 1500 the RoCE active MTU is 1024 and a 4096 path MTU loses every larger packet (the idea of
        // bertholomus/TensorFold#12, local endpoint only; #12 also exchanges and takes the smaller of both ends).
        const pa = try r.v.port(ctx, port_num);
        const mtu: c_int = if (pa.active_mtu >= 1 and pa.active_mtu < mtu_4096) pa.active_mtu else mtu_4096;
        var nic: Nic = .{ .ctx = ctx, .pd = pd, .cq = cq, .mr_send = mr_send, .mr_recv = mr_recv, .mr_gen = mr_gen, .mr_meta = mr_meta, .mr_sink = mr_sink, .gid = gid.raw, .gid_index = gi, .mtu = mtu, .atomic_cap = dev_attr.atomic_cap, .max_rd_atom = dev_attr.max_qp_rd_atom, .max_init_rd_atom = dev_attr.max_qp_init_rd_atom, .name = name };
        errdefer for (nic.qps) |q| if (q) |qp| {
            _ = api.ibv_destroy_qp(qp);
        };
        if (r.s.wait) {
            // the queue pairs are DEVX objects made by waitOpen (every object of one belongs to this context)
            nic.mac = netdevMac(name) catch {
                std.log.err("TF_DS_RING_WAIT=1: no MAC for the network device behind {s}", .{name});
                return error.NoGid;
            };
            return nic;
        }
        for (0..r.world) |p| {
            if (p == r.rank) continue;
            // the generation word is 8 bytes and rides inline: ask the queue pair for room for it
            const inline_data: u32 = if (r.s.roci.on()) 16 else 0;
            var init: vabi.QpInitAttr = .{ .send_cq = cq, .recv_cq = cq, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = recv_depth, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = inline_data }, .qp_type = vabi.qpt_rc };
            const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
            nic.qps[p] = qp;
            if (r.s.roci.on()) {
                // the granted inline capacity, not the requested one: an 8-byte word must fit whole
                var qa: vabi.QpAttr = undefined;
                var qia: vabi.QpInitAttr = undefined;
                if (api.ibv_query_qp(qp, &qa, 0, &qia) != 0) return error.VerbsFailed;
                if (qia.cap.max_inline_data < 8) nic.inline_ok = false;
            }
            const qp_access: c_uint = vabi.access_remote_write | (if (r.s.roci == .atomic) @as(c_uint, vabi.access_remote_atomic) else 0);
            var a: vabi.QpAttr = .{ .qp_state = vabi.qps_init, .port_num = port_num, .qp_access_flags = qp_access };
            const m = vabi.mask;
            if (api.ibv_modify_qp(qp, &a, m.state | m.pkey_index | m.port | m.access_flags) != 0) return error.VerbsFailed;
            for (0..recv_depth) |_| try postRecv(ctx, qp);
            nic.psns[p] = (qp.qp_num *% 2654435761) & 0xFF_FFFF;
        }
        return nic;
    }

    /// Settings.gid_index, else the device's first RoCE v2 GID that holds an IPv4 address (its type from
    /// ibv_query_gid_ex, else from sysfs; neither: error.NoGid, and Settings.gid_index / TF_DS_RDMA_GID must say).
    fn pickGid(r: *Ring, ctx: *vabi.Context, name: []const u8) Error!u8 {
        if (r.s.gid_index) |g| return g;
        const p = try r.v.port(ctx, port_num);
        var i: u32 = 0;
        while (i < @as(u32, @intCast(@max(0, @min(p.gid_tbl_len, 256))))) : (i += 1) {
            const g = r.v.gid(ctx, port_num, i) catch continue;
            if (gidIp(g.raw) == null) continue;
            const v2 = if (r.v.gidType(ctx, port_num, i)) |t| t == 2 else sysfsV2(name, i);
            if (v2) return @intCast(i);
        }
        return error.NoGid;
    }

    fn closeNic(r: *Ring, n: *Nic) void {
        const api = r.v.api;
        for (n.qps) |q| if (q) |qp| {
            _ = api.ibv_destroy_qp(qp);
        };
        if (n.mr_sink) |m| _ = api.ibv_dereg_mr(m);
        if (n.mr_meta) |m| _ = api.ibv_dereg_mr(m);
        if (n.mr_gen) |m| _ = api.ibv_dereg_mr(m);
        _ = api.ibv_dereg_mr(n.mr_recv);
        _ = api.ibv_dereg_mr(n.mr_send);
        _ = api.ibv_destroy_cq(n.cq);
        _ = api.ibv_dealloc_pd(n.pd);
        _ = api.ibv_close_device(n.ctx);
    }

    // -- TF_DS_RING_WAIT: queue pairs that wait in the card (ring_wait.zig) ---------------------------------------------------------

    /// `name` opened with DEVX enabled: the context every object of a queue pair made through DEVX belongs to.
    fn openDevx(r: *Ring, name: []const u8) Error!*vabi.Context {
        var count: c_int = 0;
        const list = r.v.api.ibv_get_device_list(&count) orelse return error.VerbsFailed;
        defer r.v.api.ibv_free_device_list(list);
        var i: usize = 0;
        while (list[i]) |dev| : (i += 1) {
            const got = r.v.api.ibv_get_device_name(dev) orelse continue;
            if (!std.mem.eql(u8, std.mem.span(got), name)) continue;
            var attr: mx.ContextAttr = .{ .flags = mx.context_flag_devx };
            return r.mlx.?.api.mlx5dv_open_device(dev, &attr) orelse error.VerbsFailed;
        }
        return error.VerbsFailed;
    }

    /// The backend, each rail's DEVX objects, and a queue pair toward every peer on every rail, each on a completion queue of its own whose
    /// memory the GPU can read (the staging kernel's credit).
    fn waitOpen(r: *Ring, gpa: std.mem.Allocator) Error!void {
        const mlx = &r.mlx.?;
        const b = wait.Backend.init(gpa, mlx, .{ .lag = 2 * r.s.slots + 8 }, .{ .rank = r.rank, .world = r.world, .slots = r.s.slots, .max_bytes = r.s.max_bytes, .send_va = @intFromPtr(r.send.mem.ptr), .atomic = r.s.roci == .atomic, .atomic_depth = r.ringAtomicDepth() }, r.abortWord(), r.doorbell(), r, failCb) catch return error.BadInfo;
        r.cd = b;
        const api = r.v.api;
        for (r.nics[0..r.nd], 0..) |*n, d| {
            var dpd: mx.DvPd = .{};
            var obj: mx.DvObj = .{};
            obj.pairs[6] = .{ n.pd, &dpd };
            if (mlx.api.mlx5dv_init_obj(&obj, mx.obj_pd) != 0) return error.VerbsFailed;
            const uar = mlx.api.mlx5dv_devx_alloc_uar(n.ctx, mx.uar_type_bf) orelse mlx.api.mlx5dv_devx_alloc_uar(n.ctx, mx.uar_type_nc) orelse return error.VerbsFailed;
            b.rails[d] = .{ .ctx = n.ctx, .pdn = dpd.pdn, .uar = uar };
            for (0..r.world) |p| {
                if (p == r.rank) continue;
                const cq = api.ibv_create_cq(n.ctx, cq_depth, null, null, 0) orelse return error.VerbsFailed;
                var dcq: mx.DvCq = .{};
                var o2: mx.DvObj = .{};
                o2.pairs[1] = .{ cq, &dcq };
                if (mlx.api.mlx5dv_init_obj(&o2, mx.obj_cq) != 0 or dcq.buf == null or dcq.dbrec == null) {
                    _ = api.ibv_destroy_cq(cq);
                    return error.VerbsFailed;
                }
                // the card writes the entries and the GPU reads them: the queue's pages are registered for the device like the rings' own
                const addr = @intFromPtr(dcq.buf.?);
                const base = std.mem.alignBackward(usize, addr, std.heap.page_size_min);
                const end = std.mem.alignForward(usize, addr + @as(usize, dcq.cqe_cnt) * dcq.cqe_size, std.heap.page_size_min);
                r.d.check(r.hr.register(@ptrFromInt(base), end - base, host_register_portable | host_register_devicemap), "cuMemHostRegister (a completion queue)") catch |e| {
                    std.log.err("TF_DS_RING_WAIT=1: the GPU cannot map a completion queue ({s}): the staging kernel reads the card's completions from it", .{@errorName(e)});
                    _ = api.ibv_destroy_cq(cq);
                    return e;
                };
                var dev: DevicePtr = 0;
                r.d.check(r.d.api.cuMemHostGetDevicePointer_v2(&dev, @ptrFromInt(base), 0), "cuMemHostGetDevicePointer (a completion queue)") catch |e| {
                    _ = r.hr.unregister(@ptrFromInt(base));
                    _ = api.ibv_destroy_cq(cq);
                    return e;
                };
                b.createQp(d, p, .{ .buf = @ptrCast(dcq.buf.?), .dbrec = dcq.dbrec.?, .cnt = dcq.cqe_cnt, .size = dcq.cqe_size, .cqn = dcq.cqn, .dev = dev + (addr - base), .handle = cq, .reg = @ptrFromInt(base) }) catch {
                    _ = r.hr.unregister(@ptrFromInt(base));
                    _ = api.ibv_destroy_cq(cq);
                    return error.VerbsFailed;
                };
            }
        }
    }

    /// The queue pairs and their completion queues released: the queue pairs taken from the card first (ring_wait.Backend.close), then
    /// the queues the GPU could read, then the user access regions are the backend's.
    fn waitClose(r: *Ring) void {
        const b = r.cd orelse return;
        var cqs: [max_devices * max_ranks]wait.Cq = undefined;
        var n: usize = 0;
        for (&b.qps) |*row| for (row) |*qo| if (qo.*) |q| {
            if (q.cq.handle != null) {
                cqs[n] = q.cq;
                n += 1;
            }
        };
        b.close();
        r.cd = null;
        for (cqs[0..n]) |c| {
            if (c.reg) |g| _ = r.hr.unregister(g);
            _ = r.v.api.ibv_destroy_cq(@ptrCast(@alignCast(c.handle.?)));
        }
    }

    /// The lane table (what a chain needs of each lane) and the staging kernel's credit table, once connect has paired the lanes.
    fn waitLanes(r: *Ring) Error!void {
        const b = r.cd.?;
        b.lanes.clearRetainingCapacity();
        for (0..r.world) |p| {
            if (p == r.rank) continue;
            const peer = &r.peers[p];
            for (0..peer.lanes) |k| {
                const di: usize = peer.nic[k];
                const nic = &r.nics[di];
                b.lanes.append(b.gpa, .{
                    .qp = &b.qps[di][p].?,
                    .peer = @intCast(p),
                    .lane = @intCast(k),
                    .lanes = peer.lanes,
                    .peer_recv = peer.recv,
                    .recv_rkey = peer.rkey[k],
                    .lkey_send = nic.mr_send.lkey,
                    .bell_va = @intFromPtr(r.doorbell()),
                    .bell_lkey = nic.mr_meta.?.lkey,
                    .gen_va = peer.gen + (@as(u64, r.rank) * max_devices + @as(u64, @intCast(k))) * 8,
                    .gen_rkey = peer.gen_rkey[k],
                    .sink_va = if (r.cd_sink) |sk| @intFromPtr(sk.mem.ptr) + (p * max_devices + k) * 8 else 0,
                    .sink_lkey = if (nic.mr_sink) |m| m.lkey else 0,
                }) catch return error.BadInfo;
            }
        }
        if (b.lanes.items.len > wait.max_lanes) return error.BadInfo; // rdma_gather.cu's MAX_CQ
        var t: wait.CqTab = .{};
        for (b.lanes.items, 0..) |l, i| t.e[i] = .{ .base = l.qp.cq.dev, .cnt = l.qp.cq.cnt, .size = l.qp.cq.size };
        t.n = @intCast(b.lanes.items.len);
        r.cd_tab = t;
        r.cd_n = t.n;
    }

    /// The ATOMICs in flight the queue pairs are set up for: the slots, capped by the smallest limits of this ring's devices.
    fn ringAtomicDepth(r: *const Ring) u8 {
        var rd: c_int = std.math.maxInt(c_int);
        var init: c_int = std.math.maxInt(c_int);
        for (r.nics[0..r.nd]) |n| {
            rd = @min(rd, n.max_rd_atom);
            init = @min(init, n.max_init_rd_atom);
        }
        return atomicDepth(r.s.slots, rd, init);
    }

    /// The ATOMICs in flight the card-waiting queue pairs were set up for (0: not the atomic signal or not waiting): part of the switch word the peers
    /// compare, so two ranks that chose differently refuse each other at connect.
    fn cdDepth(r: *const Ring) u8 {
        return if (r.s.wait and r.s.roci == .atomic and r.cd != null) r.ringAtomicDepth() else 0;
    }

    fn failCb(owner: *anyopaque, why: []const u8) void {
        const r: *Ring = @ptrCast(@alignCast(owner));
        r.fail(why);
    }

    /// One exchange's chains, written into every lane's queue now (an eager launch), or noted for the stretch's replay when `stream` is
    /// being captured (nothing runs until its graph does). Before the kernel that rings the doorbell is launched.
    fn cdLaunch(r: *Ring, stream: cuda.Stream, n: u64, user: i32) Error!void {
        const st = try cuda.graph.captureStatus(stream);
        if (st != .none) {
            wait.note(.{ .ops = &cd_ops, .ring = r, .n = n, .user = user }) catch return error.PostFailed;
            return;
        }
        r.cdPost(n, user, false) catch |e| {
            r.fail(cdWhy(e));
            return error.Failed;
        };
    }

    fn cdPost(r: *Ring, n: u64, user: i32, defer_ring: bool) wait.PostError!void {
        const slot: ?usize = if (user >= 0) r.userSendSlot(@intCast(user)) else null;
        return r.cd.?.postExchange(n, slot, defer_ring);
    }

    fn cdWhy(e: wait.PostError) []const u8 {
        return switch (e) {
            error.Fenced => "RING_WAIT: an exchange launched on a ring that was taken from the card",
            error.SqFull => "RING_WAIT: a send queue never freed room (the GPU is not running its exchanges)",
            error.Aborted => "RING_WAIT: an exchange launched on an aborted ring",
        };
    }

    const cd_ops: wait.Ops = .{ .post = opsPost, .flush = opsFlush, .fail = opsFail };

    fn opsPost(ring: *anyopaque, n: u64, user: i32, defer_ring: bool) wait.PostError!void {
        const r: *Ring = @ptrCast(@alignCast(ring));
        return r.cdPost(n, user, defer_ring);
    }

    fn opsFlush(ring: *anyopaque) wait.PostError!void {
        const r: *Ring = @ptrCast(@alignCast(ring));
        return r.cd.?.flush();
    }

    fn opsFail(ring: *anyopaque, why: []const u8) void {
        const r: *Ring = @ptrCast(@alignCast(ring));
        r.fail(why);
    }

    /// TF_DS_RING_WAIT's start line.
    pub fn waitLine(r: *const Ring, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "TF_DS_RING_WAIT: on: {s}: no proxy thread; {d} queue pair(s) wait in the card on the doorbell (big-endian word, WAIT >= the sequence), their chains written when an exchange launches, each ending in {s}; the staging kernel's credit is the card's completion of that last entry (one completion queue a queue pair); one janitor thread looks every {d} ms for a failure", .{ r.s.name, r.cd_n, if (r.s.roci == .atomic) "the RDMA atomic fetch-and-add of the peer's generation word (the guaranteed arrival signal)" else "the inline WRITE of the generation word", r.cd.?.p.sweep_ms }) catch "TF_DS_RING_WAIT=1: on";
    }

    pub fn destroy(r: *Ring, gpa: std.mem.Allocator) void {
        r.xlogWriteFile();
        r.stop();
        r.waitClose();
        for (r.nics[0..r.nd]) |*n| r.closeNic(n);
        if (r.cd_sink) |*t| t.free(&r.hr);
        if (r.mlx) |*m| m.close();
        if (r.fault) |*b| b.free();
        if (r.lanes_dev) |*b| b.free();
        if (r.gen) |*g| g.free(&r.hr);
        r.scalars.free();
        r.meta.free(&r.hr);
        r.recv.free(&r.hr);
        r.send.free(&r.hr);
        r.hr.lib.close();
        r.verbs.close();
        gpa.destroy(r);
    }

    /// What peer `p`'s queue pairs toward this rank need.
    pub fn info(r: *const Ring, p: u32) Info {
        var out: Info = std.mem.zeroes(Info);
        out.devices = r.nd;
        out.flags = r.switchBits();
        out.recv = @intFromPtr(r.recv.mem.ptr);
        out.gen = if (r.gen) |g| @intFromPtr(g.mem.ptr) else 0;
        for (r.nics[0..r.nd], 0..) |n, d| {
            out.qpn[d] = if (r.cd) |c| (if (c.qps[d][p]) |q| q.qpn else 0) else if (n.qps[p]) |qp| qp.qp_num else 0;
            out.psn[d] = n.psns[p];
            out.mac[d] = n.mac;
            out.rkey[d] = n.mr_recv.rkey;
            out.gid[d] = n.gid;
            out.gen_rkey[d] = if (n.mr_gen) |m| m.rkey else 0;
        }
        return out;
    }

    /// `remote[p]`: what rank p's `info(rank)` gave. Routes each peer (pairLanes; a peer with no lane relayed through the
    /// ranks routed to both), finds this rank's relays, and moves every routed QP to RTR and RTS (the Python build's
    /// attributes).
    pub fn connect(r: *Ring, remote: []const Info) Error!void {
        if (remote.len != r.world) return error.BadInfo;
        const api = r.v.api;
        const m = vabi.mask;
        const prefix: u8 = r.s.prefix orelse blk: {
            const e = std.c.getenv("TF_DS_RDMA_PREFIX") orelse break :blk 24;
            break :blk std.fmt.parseInt(u8, std.mem.span(e), 10) catch 24;
        };
        // every rank's devices (this rank's own from its info toward any peer): who reaches whom
        var all: [max_ranks]Info = undefined;
        for (0..r.world) |p| all[p] = if (p == r.rank) r.info(@intCast((r.rank + 1) % r.world)) else remote[p];
        for (remote, 0..) |ri, p| if (p != r.rank and (ri.devices == 0 or ri.devices > max_devices)) return error.BadInfo;
        // the switches change where and how a part is laid out: a peer that runs others would read another race
        for (remote, 0..) |ri, p| if (p != r.rank and ri.flags != r.switchBits()) {
            std.log.err("{s}: rank {d}'s ring switches 0x{x} differ from this rank's 0x{x} (bit 2 generation word, 3 atomic signal, 4 card WAIT, bits 8-15 the ATOMICs in flight): every rank must choose alike", .{ r.s.name, p, ri.flags, r.switchBits() });
            return error.BadInfo;
        };
        var mine_g: [max_devices][16]u8 = undefined;
        for (r.nics[0..r.nd], 0..) |n, d| mine_g[d] = n.gid;
        for (remote, 0..) |ri, p| {
            if (p == r.rank) continue;
            const peer = &r.peers[p];
            peer.* = .{ .recv = ri.recv };
            var pairs: [max_devices][2]u8 = undefined;
            const n = pairLanes(mine_g[0..r.nd], ri.gid[0..ri.devices], prefix, &pairs) orelse blk: {
                // no IPv4 GIDs: device d with device d, every device
                if (ri.devices != r.nd) return error.BadInfo;
                for (0..r.nd) |d| pairs[d] = .{ @intCast(d), @intCast(d) };
                break :blk r.nd;
            };
            peer.lanes = n;
            peer.gen = ri.gen;
            for (pairs[0..n], 0..) |pr, k| {
                peer.nic[k] = pr[0];
                peer.qpn[k] = ri.qpn[pr[1]];
                peer.psn[k] = ri.psn[pr[1]];
                peer.rkey[k] = ri.rkey[pr[1]];
                peer.gid[k] = ri.gid[pr[1]];
                peer.gen_rkey[k] = ri.gen_rkey[pr[1]];
                peer.mac[k] = ri.mac[pr[1]];
            }
            if (n == 0) {
                // relayed: every rank linked to both, in rank order
                for (0..r.world) |q| {
                    if (q == p or q == r.rank) continue;
                    if (linked(all[q], all[p], prefix) and linked(all[q], all[r.rank], prefix)) {
                        peer.relayers[peer.n_relayers] = @intCast(q);
                        peer.n_relayers += 1;
                    }
                }
                if (peer.n_relayers == 0 or peer.n_relayers > max_devices) return error.BadInfo;
            }
        }
        // TF_DS_RING_WORD: every peer's lane writes its generation word, so a relayed peer (a rank no device here
        // reaches) cannot run it: its writes would come from a relaying rank that would have to wait on its own words
        // first. The atomic signal also needs global atomics on every device it runs on.
        r.roci_live = false;
        r.roci_order = -1;
        if (r.s.roci.on()) {
            var ok = r.k.hasRoci();
            if (!ok) r.rociNote("the rings' kernels have no dsv41_rdma_collect_roci / _into_roci / _peers_roci", .{});
            if (ok) for (0..r.world) |p| {
                if (p == r.rank) continue;
                if (r.peers[p].lanes == 0) {
                    ok = false;
                    r.rociNote("the ring relays to node {d}, a rank no device here reaches: the generation word needs a lane to every peer", .{p});
                    break;
                }
                if (r.peers[p].gen == 0 or r.peers[p].gen_rkey[0] == 0) {
                    ok = false;
                    r.rociNote("node {d}'s generation words are not registered (its build or start has none)", .{p});
                    break;
                }
            };
            // the devices the lanes run on (connect marks `used` only after this check)
            if (ok and r.s.roci == .atomic) {
                for (0..r.world) |p| {
                    if (p == r.rank) continue;
                    for (0..r.peers[p].lanes) |k| {
                        const n = r.nics[r.peers[p].nic[k]];
                        // a generation word has one remote writer (the peer's queue pair for the lane) and the GPU only reads it: the atomic is there
                        // for its place after the earlier WRITEs, so the card's own atomics (HCA scope, the CX-7's) are enough, as for Rocinante's words
                        var rb: [300]u8 = undefined;
                        if (atomicRefusal(&rb, n.name, n.atomic_cap, n.max_rd_atom, n.max_init_rd_atom)) |why| {
                            ok = false;
                            r.rociNote("{s}", .{why});
                            break;
                        }
                    }
                    if (!ok) break;
                }
            }
            if (ok and r.s.roci == .word) {
                for (0..r.world) |p| {
                    if (p == r.rank) continue;
                    for (0..r.peers[p].lanes) |k| {
                        const n = r.nics[r.peers[p].nic[k]];
                        if (!n.inline_ok) {
                            ok = false;
                            r.rociNote("device {s} granted less than 8 bytes of inline data: the generation word needs 8", .{n.name});
                            break;
                        }
                    }
                    if (!ok) break;
                }
            }
            r.roci_live = ok;
        }
        if (r.s.wait and !r.roci_live) {
            var hb: [200]u8 = undefined;
            std.log.err("{s}: the generation word cannot run on this ring: {s}", .{ waitRefusalHead(&hb, r.s.wait_explicit), std.mem.sliceTo(&r.roci_why, 0) });
            return error.BadInfo;
        }
        // this rank's relays: from a peer to a peer it routes to both of, when those two have no lane between them
        r.n_jobs = 0;
        for (0..r.world) |src| for (0..r.world) |dst| {
            if (src == dst or src == r.rank or dst == r.rank) continue;
            if (r.peers[src].lanes == 0 or r.peers[dst].lanes == 0 or linked(all[src], all[dst], prefix)) continue;
            var k: u8 = 0;
            var of: u8 = 0;
            for (0..r.world) |q| {
                if (q == src or q == dst) continue;
                const both = if (q == r.rank) true else linked(all[q], all[src], prefix) and linked(all[q], all[dst], prefix);
                if (!both) continue;
                if (q == r.rank) k = of;
                of += 1;
            }
            if (r.n_jobs == r.jobs.len) return error.BadInfo;
            r.jobs[r.n_jobs] = .{ .src = @intCast(src), .dst = @intCast(dst), .half = k, .of = of, .lane = @intCast(k % r.peers[dst].lanes) };
            r.n_jobs += 1;
        };
        for (0..r.world) |p| {
            if (p == r.rank) continue;
            const peer = &r.peers[p];
            for (0..peer.lanes) |k| {
                const n = &r.nics[peer.nic[k]];
                n.used = true;
                if (r.cd) |c| {
                    c.connectQp(peer.nic[k], p, .{ .qpn = peer.qpn[k], .gid = peer.gid[k], .mac = peer.mac[k] }, n.gid_index, udpSport(c.qps[peer.nic[k]][p].?.qpn, peer.nic[k], p), @intCast(n.mtu)) catch return error.VerbsFailed;
                    continue;
                }
                const qp = n.qps[p].?;
                // the atomic signal may have a ring's slots of fetch-and-adds outstanding on one queue pair
                const rd_atomic: u8 = if (r.s.roci == .atomic) @intCast(r.s.slots) else 1;
                var a: vabi.QpAttr = .{ .qp_state = vabi.qps_rtr, .path_mtu = n.mtu, .dest_qp_num = peer.qpn[k], .rq_psn = peer.psn[k], .max_dest_rd_atomic = rd_atomic, .min_rnr_timer = 12 };
                a.ah_attr = .{ .grh = .{ .dgid = .{ .raw = peer.gid[k] }, .sgid_index = n.gid_index, .hop_limit = 64 }, .is_global = 1, .port_num = port_num };
                if (api.ibv_modify_qp(qp, &a, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
                a = .{ .qp_state = vabi.qps_rts, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .sq_psn = n.psns[p], .max_rd_atomic = rd_atomic };
                if (api.ibv_modify_qp(qp, &a, m.state | m.timeout | m.retry_cnt | m.rnr_retry | m.sq_psn | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
            }
        }
        // TF_DS_RING_WORD's ordering record: what the provider says about a WRITE's data, the most conservative
        // answer over the lanes. The wait itself relies on the observed order, as the other arrival signals do.
        if (r.roci_live) if (r.v.in_order) |f| {
            var best: c_int = std.math.maxInt(c_int);
            for (0..r.world) |p| {
                if (p == r.rank) continue;
                for (0..r.peers[p].lanes) |k| {
                    const qp = r.nics[r.peers[p].nic[k]].qps[p] orelse continue;
                    best = @min(best, f(qp, vabi.wr_rdma_write, 0));
                }
            }
            if (best != std.math.maxInt(c_int)) r.roci_order = best;
        };
        if (r.cd != null) {
            try r.waitLanes();
            r.roci_order = 1; // measured on both rails through DEVX: qpc.data_in_order = 1 (whole message), ENGINE-FACTS
        }
    }

    /// The routes connect found, for the startup line: per peer its lanes (0: relayed) and this rank's relay count.
    pub fn routes(r: *const Ring) struct { lanes: [max_ranks]u32, relays: u32 } {
        var out: [max_ranks]u32 = @splat(0);
        for (0..r.world) |p| out[p] = r.peers[p].lanes;
        return .{ .lanes = out, .relays = r.n_jobs };
    }

    }

    pub fn start(r: *Ring) Error!void {
        if (r.running.swap(true, .acq_rel)) return;
        if (r.s.roci.on()) {
            if (r.roci_live) {
                // the lanes toward each peer, as the waiting kernel reads them
                var lanes: [max_ranks]u32 = @splat(0);
                for (0..r.world) |p| lanes[p] = r.peers[p].lanes;
                if (r.lanes_dev) |*b| try b.upload(0, std.mem.sliceAsBytes(lanes[0..r.world]));
            }
            var buf: [384]u8 = undefined;
            r.logLine(r.rociLine(&buf));
        }
        if (r.cd) |c| {
            // no proxy: the card runs the chains; one janitor thread looks for a failure
            r.napping = false;
            r.nap_how = null;
            r.late_window = .{};
            r.nap_ticks = 0;
            r.nap_stats = .{};
            var wbuf: [640]u8 = undefined;
            r.logLine(r.waitLine(&wbuf));
            c.startJanitor() catch |e| {
                r.running.store(false, .release);
                return e;
            };
            return;
        }
    }

    pub fn stop(r: *Ring) void {
        if (r.running.swap(false, .acq_rel)) if (r.proxy) |t| t.join();
        r.proxy = null;
    }

    /// Why the generation-word switch cannot run on this ring, kept for its start line.
    fn rociNote(r: *Ring, comptime fmt: []const u8, args: anytype) void {
        const n = std.fmt.bufPrint(&r.roci_why, fmt, args) catch return;
        if (n.len < r.roci_why.len) r.roci_why[n.len] = 0;
    }

    /// TF_DS_RING_WORD's start line: on with its signal and what the in-order query said, or off and why.
    pub fn rociLine(r: *const Ring, buf: []u8) []const u8 {
        if (!r.s.roci.on()) return "";
        if (!r.roci_live) return std.fmt.bufPrint(buf, "TF_DS_RING_WORD=1: off: {s}", .{std.mem.sliceTo(&r.roci_why, 0)}) catch "TF_DS_RING_WORD=1: off";
        const order: []const u8 = switch (r.roci_order) {
            1 => "whole message",
            2 => "128-byte blocks",
            else => "not promised",
        };
        const signal: []const u8 = if (r.s.roci == .word)
            "an inline 8-byte RDMA WRITE after it on the same queue pair"
        else
            "an RDMA ATOMIC fetch-and-add after it on the same queue pair";
        return std.fmt.bufPrint(buf, "TF_DS_RING_WORD=1: each lane's data is a plain RDMA WRITE and its 64-bit generation word {s}; the collecting kernel waits on every lane's word (>= the sequence, bounded by a deadline and the abort word; ibv_query_qp_data_in_order over the lanes: {s}): the same bytes", .{ signal, order }) catch "TF_DS_RING_WORD=1: on";
    }

    /// The first expired generation-word wait, if one wrote its record; call after the stream is synchronized. The
    /// record: magic (tags a written record), sequence, this rank, peer and lane, the word seen, the wait's ns, when,
    /// and why (1: the abort word, 2: the deadline).
    pub const Fault = struct { seq: u64 = 0, rank: u32 = 0, peer: u32 = 0, lane: u32 = 0, seen: u64 = 0, waited_ns: u64 = 0, at_ns: u64 = 0, why: u64 = 0 };

    /// The exchange log (TF_DS_ROCI_XLOG): the records written so far, in ring order, empty slots left out; call after the stream is
    /// synchronized. Returns the count.
    pub fn xlogRead(r: *Ring, out: []XlogRec) usize {
        const b = r.fault orelse return 0;
        if (!xlog) return 0;
        const words = std.heap.page_allocator.alloc(u64, xlog_recv_records * 4) catch return 0;
        defer std.heap.page_allocator.free(words);
        b.download(64, std.mem.sliceAsBytes(words)) catch return 0;
        return xlogDecode(words, out);
    }

    /// This rank's doorbell records (TF_DS_ROCI_XLOG), as xlogRead.
    pub fn xlogPubRead(r: *Ring, out: []XlogPub) usize {
        const b = r.fault orelse return 0;
        if (!xlog) return 0;
        const words = std.heap.page_allocator.alloc(u64, xlog_records * 2) catch return 0;
        defer std.heap.page_allocator.free(words);
        b.download(64 + xlog_recv_records * 32, std.mem.sliceAsBytes(words)) catch return 0;
        return xlogPubDecode(words, out);
    }

    /// TF_DS_ROCI_XLOG: this rank's log to `<dir>/roci-xlog-rank<R>.txt` (dir: TF_DS_ROCI_XLOG_DIR, else the folder's log dir
    /// $HOME/.cache/dsv41-tensorfold/logs; created if missing). Called once, from destroy, when the GPU side is quiescent: never inside a round.
    /// A failure is a log line, never an error: this is a measurement aid.
    pub fn xlogWriteFile(r: *Ring) void {
        if (!xlog or r.fault == null) return;
        const gpa = std.heap.page_allocator;
        const recs = gpa.alloc(XlogRec, xlog_recv_records) catch return;
        defer gpa.free(recs);
        const n = r.xlogRead(recs);
        const pubs = gpa.alloc(XlogPub, xlog_records) catch return;
        defer gpa.free(pubs);
        const np = r.xlogPubRead(pubs);
        const text = xlogRenderPub(gpa, @intCast(r.rank), @intCast(r.world), r.s.name, @intCast(r.s.slots), recs[0..n], pubs[0..np]) catch return;
        defer gpa.free(text);
        var namebuf: [64]u8 = undefined;
        const name = xlogFileName(&namebuf, @intCast(r.rank), r.s.name) orelse return;
        if (diagWriteFile("TF_DS_ROCI_XLOG_DIR", name, text)) std.log.info("TF_DS_ROCI_XLOG: {d} exchange records written ({s})", .{ n, name });
    }

    pub fn faultRecord(r: *Ring, out: *Fault) bool {
        const b = r.fault orelse return false;
        var words: [8]u64 = undefined;
        b.download(0, std.mem.sliceAsBytes(&words)) catch return false;
        if (words[0] != fault_magic) return false;
        out.* = .{ .seq = words[1], .rank = @truncate(words[2]), .peer = @truncate(words[3] >> 32), .lane = @truncate(words[3]), .seen = words[4], .waited_ns = words[5], .at_ns = words[6], .why = words[7] };
        return true;
    }

        const was_running = r.running.swap(false, .acq_rel);
        if (was_running and r.cd != null) {
            r.cd.?.stopJanitor(); // no proxy: the queue pairs stay armed until the ring is destroyed or fails
                var sb: [512]u8 = undefined;
                r.logLine(wait.statsLine(&sb));
        } else if (was_running) {
            r.idle_gate.wake(); // a parked proxy must wake before join
            if (r.proxy) |t| t.join();
        }
        r.proxy = null;
    /// Only the CPU control protocol calls these at globally quiescent boundaries.
    pub fn park(r: *Ring) void {
        r.idle_gate.park();
    }
    pub fn wake(r: *Ring) void {
        r.idle_gate.wake();
    }

    /// Every wait on this rank gives up from now on (a watchdog lost a peer); `failure` says why.
    pub fn abort(r: *Ring, why: []const u8) void {
        r.fail(why);
    }

    pub fn failure(r: *const Ring) ?[]const u8 {
        if (!r.failed.load(.acquire)) {
            // TF_DS_RING_WORD: a waiting kernel's expiry sets the abort word (a system store into this memory); the
            // next gather must refuse rather than post into a ring whose sequences are no longer trusted
            if (@atomicLoad(u64, r.abortWord(), .acquire) != 0) return "a bounded generation-word wait ended: the ring is poisoned";
            return null;
        }
        return std.mem.sliceTo(&r.why, 0);
    }

    fn fail(r: *Ring, why: []const u8) void {
        r.failLocal(why);
        if (r.failure_peer) |peer| peer.failLocal(why);
    }

    fn failLocal(r: *Ring, why: []const u8) void {
        if (!r.failed.swap(true, .acq_rel)) {
            const n = @min(why.len, r.why.len - 1);
            @memcpy(r.why[0..n], why[0..n]);
        }
        // TF_DS_RING_WAIT: the queue pairs leave the card first (an armed WAIT is flushed, not released), then the abort word
        if (r.cd) |c| c.fence();
        @atomicStore(u64, r.abortWord(), 1, .release);
    }

    fn proxyLoop(r: *Ring) void {
        if (r.s.cpu) |c| {
                var set: std.os.linux.cpu_set_t = @splat(0);
                const bits = @bitSizeOf(usize);
                set[c / bits] |= @as(usize, 1) << @intCast(c % bits);
                std.os.linux.sched_setaffinity(0, &set) catch {};
            }
        var need: u32 = 0;
        var arrivals: seqs.Arrivals = .{};
        for (0..r.world) |p| {
            if (p == r.rank) continue;
            need += r.peers[p].lanes;
            arrivals.lanes[p] = if (r.peers[p].lanes > 0) r.peers[p].lanes else r.peers[p].n_relayers;
        }
        var sends: seqs.Sends = .{ .need = need, .slots = r.s.slots };
        var relays = seqs.Relays.init(r.n_jobs, r.s.slots);
        // the bytes of each peer's sequence in (by slot), for the halves a relay sends on
        var got_bytes: [max_ranks][seqs.max_slots]u64 = @splat(@splat(0));
        var posted: u64 = 0;
        // the progress watchdog: a sequence posted whose peers' parts do not all come within TF_DS_RING_DEAD_MS (default 10 s) fails
        // the ring by name
        const dead_ms: u64 = r.dead_ms;
        var watch: ring_watch.Watch = .{ .dead = dead_ms * 1000 * tpu };
        var wc: [64]vabi.Wc = undefined;
        while (r.running.load(.monotonic)) {
                r.fail("RDMA idle futex wait failed");
                return;
            };
            if (!r.running.load(.acquire)) break;
            if (watch.open > 0 and watch.expired(instr.ticks())) {
                var pb: [640]u8 = undefined;
                var mb: [900]u8 = undefined;
                const msg = std.fmt.bufPrint(&mb, "{s}: a sequence posted by this rank has had no peer part for {d} ms (posted {d}, {d} open); this rank's host was {s}", .{ r.s.name, dead_ms, posted, watch.open, where.report(&pb) }) catch "a posted sequence got no peer part";
                std.log.err("{s}", .{msg});
                r.fail(msg);
                watch.open = 0;
            }
            const ring = @atomicLoad(u64, r.doorbell(), .acquire);
            // TF_DS_RING_WORD: the doorbell is a counter whose every sequence must be posted; the GPU cannot ring
            // more than the ring's slots ahead (the staging kernel waits on `sent`), so more pending than slots is a
            // skipped sequence, and the peers would wait on a word that never comes. Fail stop here.
            if (r.roci_live and ring -| posted > r.s.slots) {
                r.fail("the doorbell holds more sequences than the ring's slots: a sequence was skipped");
                return;
            }
            // a relaying rank's own sequence q goes out once its relays of every sequence up to q - (slots - 1) are
            // done: an owner sends q + 1 (into the slot of q + 1 - slots) only after collecting this rank's q
            while (posted < ring and (r.n_jobs == 0 or relays.upto + r.s.slots >= posted + 2)) {
                const q = posted + 1;
                const slot: usize = @intCast(q % r.s.slots);
                const n = @atomicLoad(u64, &r.sizes()[slot], .acquire);
                // an in-place publish names the fixed slot its producer wrote (send_q holds q, so a staged send's
                // rotation is never mistaken for it); a staged send reads the rotating slot itself
                const send_slot: usize = if (r.s.inplace and @atomicLoad(u64, &r.sendQ()[slot], .acquire) == q)
                    r.userSendSlot(@intCast(@atomicLoad(u64, &r.sendSlotWord()[slot], .acquire)))
                else
                    slot;
                // the inline word's source: one value for every lane of this sequence, copied when each WQE is posted
                if (r.roci_live and r.s.roci == .word) @atomicStore(u64, r.rociWord(), ringWordSignalWord(q), .release);
                for (0..r.world) |p| {
                    if (p == r.rank) continue;
                    const peer = &r.peers[p];
                    for (0..peer.lanes) |k| {
                        const nic = &r.nics[peer.nic[k]];
                        const pt = seqs.part(n, peer.lanes, @intCast(k));
                        const off: usize = @intCast(pt.off);
                        const len: usize = @intCast(pt.len);
                        const local = r.send.mem[send_slot * r.s.max_bytes + off ..][0..len];
                        const at: u64 = @intCast((slot * r.world + r.rank) * r.s.max_bytes + off);
                        if (r.roci_live) {
                            const mr_gen = nic.mr_gen orelse {
                                r.fail("a generation-word region without a registration");
                                return;
                            };
                            // the word of the sender's lane k, in the receiver's [peer][lane] region (its pairing
                            // gives both ends the same lane index)
                            const gen_at = peer.gen + (@as(u64, r.rank) * max_devices + @as(u64, @intCast(k))) * 8;
                            if (len > 0) postData(nic.ctx, nic.qps[p].?, local, nic.mr_send.lkey, peer.recv + at, peer.rkey[k], q) catch {
                                r.fail("ibv_post_send failed");
                                return;
                            };
                            if (r.s.roci == .word) {
                                postWord(nic.ctx, nic.qps[p].?, r.rociWordBytes(), mr_gen.lkey, gen_at, peer.gen_rkey[k], q) catch {
                                    r.fail("ibv_post_send (the generation word) failed");
                                    return;
                                };
                            } else postFetchAdd(nic.ctx, nic.qps[p].?, gen_at, peer.gen_rkey[k], q) catch {
                                r.fail("ibv_post_send (the generation word) failed");
                                return;
                            };
                        } else postWrite(nic.ctx, nic.qps[p].?, local, nic.mr_send.lkey, peer.recv + at, peer.rkey[k], q, seqs.immediate(q, null)) catch {
                            r.fail("ibv_post_send failed");
                            return;
                        };
                    }
                }
                // (under TF_DS_RING_WORD no arrival reaches this proxy to close the sequence: the collecting kernel's deadline is the bound)
                if (ring_watch.armed(!r.roci_live)) watch.posted(slot, post_t[slot]);
                posted = q;
            }
            for (r.nics[0..nd], 0..) |nic, d| {
                // with the generation word the receive side is passive: a sequence is quiet once its own word writes
                // have completed (earlier data writes on the same queue pairs complete before them)
                if (if (r.roci_live) inflight == 0 else nap.quiet(posted, ring, inflight, arrivals.published[0..r.world], r.rank)) {
            for (r.nics[0..r.nd], 0..) |nic, d| {
                if (!nic.used) continue;
                const k = nic.ctx.ops.poll_cq(nic.cq, wc.len, &wc);
                if (k < 0) {
                    r.fail("ibv_poll_cq failed");
                    return;
                }
                for (wc[0..@intCast(k)]) |w| {
                    if (w.status != vabi.wc_success) {
                        r.fail(r.v.statusText(w.status));
                        return;
                    }
                    if (w.opcode == wc_recv_rdma_with_imm) {
                        if (r.roci_live) {
                            // this path posts no receives and every peer runs it (the rings' switch word refuses a
                            // peer that does not): an arrival here belongs to another path
                            r.fail("a receive completion on a ring that waits on generation words");
                            return;
                        }
                        const from = r.peerOf(nic, w.qp_num) orelse {
                            r.fail("receive completion on an unknown QP");
                            return;
                        };
                        const im = seqs.decode(w.imm_data);
                        // a relayed part: its owner's, on the lane of the relaying rank `from`
                        const p: usize = if (im.relayed_from) |s| s else from;
                        if (p >= r.world or p == r.rank) {
                            r.fail("receive completion for a bad rank");
                            return;
                        }
                        const lane: usize = if (im.relayed_from != null) r.relayerLane(p, from) orelse {
                            r.fail("relayed part from a rank that does not relay it");
                            return;
                        } else r.laneOf(from, d) orelse {
                            r.fail("receive completion on a device off the peer's route");
                            return;
                        };
                        const got = arrivals.arrive(p, lane, im.low);
                        if (im.relayed_from == null) got_bytes[p][@intCast(got.seq % r.s.slots)] += w.byte_len;
                        var q = got.from;
                        while (q <= got.to) : (q += 1) {
                            if (im.relayed_from == null and r.n_jobs > 0) {
                                const bytes = &got_bytes[p][@intCast(q % r.s.slots)];
                                r.relay(p, q, bytes.*) catch {
                                    r.fail("ibv_post_send (relay) failed");
                                    return;
                                };
                                bytes.* = 0;
                            } else if (im.relayed_from == null) got_bytes[p][@intCast(q % r.s.slots)] = 0;
                            @atomicStore(u64, r.flagAt(q, p), q, .release);
                                watch.completed(sl, instr.ticks(), posted >= q);
                        }
                        postRecv(nic.ctx, nic.qps[from].?) catch {
                            r.fail("ibv_post_recv failed");
                            return;
                        };
                    } else if (w.wr_id & relay_wr != 0) {
                        relays.complete(w.wr_id & ~relay_wr);
                        if (sends.complete(w.wr_id)) |q| {
                            const qslot: usize = @intCast(q % r.s.slots);
                            // read the slot's name before publishing `sent`: a producer of q + slots may overwrite the
                            // name only after sent[qslot] reaches q
                            const in_place = r.s.inplace and @atomicLoad(u64, &r.sendQ()[qslot], .acquire) == q;
                            const u: usize = if (in_place) @intCast(@atomicLoad(u64, &r.sendSlotWord()[qslot], .acquire)) else 0;
                            @atomicStore(u64, &r.sent()[qslot], q, .release);
                            // the fixed send slot's producer may write again once this send has completed
                            if (in_place) @atomicStore(u64, &r.sendDone()[u], q, .release);
                        }
                    }
                }
            }
                if (if (r.roci_live) inflight == 0 else nap.quiet(posted, ring, inflight, arrivals.published[0..r.world], r.rank)) r.napOn(posted, &naps_since_cpu);
            }
        }
    }

        }
    /// Rank `src`'s sequence q (`n` bytes, all in this rank's receive slot) on to the ranks this one relays it to.
    fn relay(r: *Ring, src: usize, q: u64, n: u64) Error!void {
        const slot: usize = @intCast(q % r.s.slots);
        for (r.jobs[0..r.n_jobs]) |j| {
            if (j.src != src) continue;
            const peer = &r.peers[j.dst];
            const nic = &r.nics[peer.nic[j.lane]];
            const pt = seqs.part(n, j.of, j.half);
            const at: usize = (slot * r.world + src) * r.s.max_bytes + @as(usize, @intCast(pt.off));
            const local = r.recv.mem[at..][0..@intCast(pt.len)];
            try postWrite(nic.ctx, nic.qps[j.dst].?, local, nic.mr_recv.lkey, peer.recv + at, peer.rkey[j.lane], q | relay_wr, seqs.immediate(q, @intCast(src)));
        }
    }

    /// The lane of peer p that runs on this rank's device d.
    fn laneOf(r: *const Ring, p: usize, d: usize) ?usize {
        const peer = &r.peers[p];
        for (0..peer.lanes) |k| if (peer.nic[k] == d) return k;
        return null;
    }

    /// The lane of relayed peer p that relaying rank `via` brings.
    fn relayerLane(r: *const Ring, p: usize, via: usize) ?usize {
        const peer = &r.peers[p];
        for (0..peer.n_relayers) |k| if (peer.relayers[k] == via) return k;
        return null;
    }

    fn flagAt(r: *const Ring, q: u64, p: usize) *u64 {
        const slot: usize = @intCast(q % r.s.slots);
        return &r.flags()[slot * r.world + p];
    }

    fn peerOf(r: *const Ring, nic: Nic, qpn: u32) ?usize {
        for (0..r.world) |p| if (nic.qps[p]) |qp| if (qp.qp_num == qpn) return p;
        return null;
    }

    fn scalar(r: *const Ring, offset: usize) DevicePtr {
        return r.scalars.ptr + offset;
    }

    /// send_last [max_users] in device memory (see scalars_bytes); the metadata block's host words of the same name stay where they were and are not given to a kernel.
    fn sendLastDev(r: *const Ring) DevicePtr {
        return r.scalar(80);
    }

    /// Wait, before a producer writes its in-place send slot, for the previous send from that slot to complete. The
    /// producer kernel itself must follow this on the stream (after it, its stores are the only writer); false when the
    /// ring is not in place. `sendSlotDev` assigns the slot first (Round.init), so a gather finds it here.
    pub fn prepareSend(r: *Ring, stream: cuda.Stream) Error!bool {
        if (!r.s.inplace) return false;
        const u = r.userOfHandle(stream.handle) orelse return false;
        if (r.cd != null) {
            const fw = r.k.await_send_wait orelse return error.BadInfo;
            var w: cuda.Args = .{};
            w.add(r.cd_tab);
            w.add(r.sendLastDev());
            w.add(@as(c_int, @intCast(u)));
            w.add(r.metaDev(r.abortWord()));
            w.add(@as(c_int, @intFromBool(r.s.pdl)));
            try cuda.launch.launch(fw, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &w);
            return true;
        }
        const f = r.k.await_send orelse return error.BadInfo;
        var a: cuda.Args = .{};
        a.add(r.metaDev(r.sendDone()));
        a.add(r.sendLastDev());
        a.add(@as(c_int, @intCast(u)));
        a.add(r.metaDev(r.abortWord()));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
        return true;
    }

    /// Publish `stream`'s in-place send slot as sequence `*seq + 1`: the slot's size, the slot's name and the sequence,
    /// then the doorbell (a system release, as the staging kernel's). The producer wrote the partial into the slot
    /// before this launch; `own` (0: none) is this rank's slice of the caller's layout, copied here as stage copied it.
    fn publish(r: *Ring, stream: cuda.Stream, n4: u32, own: DevicePtr, own_row4: u32, own_stride4: u64) Error!void {
        const u = (try r.userSlot(stream)) orelse return error.BadInfo;
        const waiting = r.cd != null;
        if (waiting) try r.cdLaunch(stream, @as(u64, n4) * 16, @intCast(u));
        const pw = Kernels.pick(r.k.publish_wait, r.k.publish_wait_x);
        const f = (if (waiting) pw.f else r.k.publish) orelse return error.BadInfo;
        var a: cuda.Args = .{};
        a.add(r.metaDev(r.sizes()));
        a.add(r.metaDev(r.doorbell()));
        a.add(r.scalar(0));
        a.add(r.metaDev(r.sendQ()));
        a.add(r.metaDev(r.sendSlotWord()));
        a.add(r.sendLastDev());
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(c_int, @intCast(u)));
        if (waiting) a.add(r.cd_tab) else a.add(r.metaDev(r.sent())); // the card's completions (by value), where the proxy path's `sent` words are
        a.add(r.metaDev(r.abortWord()));
        // the own slice's source is the fixed send slot the producer wrote; its destination is the caller's layout
        const slot: DevicePtr = r.send.dev + @as(DevicePtr, @intCast(r.userSendSlot(u))) * r.s.max_bytes;
        a.add(if (own != 0) slot else 0);
        a.add(own);
        a.add(@as(c_int, @intCast(own_row4)));
        a.add(@as(c_longlong, @intCast(own_stride4)));
        a.add(@as(u64, @as(u64, n4) * 16));
        a.add(if (waiting) r.probeOrPubLog(pw.logging) else r.probe());
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        // dsv41_rdma_publish's 17 parameters (dsv41_rdma_publish_wait's 17 too: the completion table by value for `sent`); the driver reads
        // one argument address per parameter, so a short pack would read past the set ones
        if (a.count != 17) return error.BadInfo;
        const blocks = std.math.clamp((@as(u64, n4) + 1023) / 1024, 1, r.s.stage_blocks);
        const g: u32 = if (own != 0) @intCast(blocks) else 1;
        try cuda.launch.launch(f, .{ .grid = .{ .x = g, .y = 1, .z = 1 }, .block = .{ .x = stage_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }

    /// The collecting half of `gather`: every rank's slot (and this rank's own, from `own_src` unless `skip_own`) into
    /// `recv`, rank after rank.
    fn launchGatherCollect(r: *Ring, stream: cuda.Stream, own_src: DevicePtr, skip_own: bool, n4: u32, recv: DevicePtr) Error!void {
        if (r.roci_live) {
            const f = r.k.collect_roci orelse return error.BadInfo;
            var b: cuda.Args = .{};
            b.add(own_src);
            b.add(recv);
            b.add(@as(c_int, @intCast(n4)));
            b.add(r.recv.dev);
            b.add(r.gen.?.dev);
            b.add(r.lanes_dev.?.ptr);
            b.add(r.scalar(0));
            b.add(@as(c_int, @intCast(r.rank)));
            b.add(@as(c_int, @intCast(r.world)));
            b.add(@as(c_int, @intCast(r.s.slots)));
            b.add(@as(u64, r.s.max_bytes));
            b.add(r.probe());
            b.add(r.metaDev(r.abortWord()));
            b.add(@as(c_int, @intFromBool(skip_own)));
            b.add(@as(c_int, @intFromBool(r.s.unroll)));
            b.add(r.s.roci_deadline_ns);
            b.add(r.fault.?.ptr);
            b.add(@as(c_int, @intFromBool(r.s.pdl)));
            if (b.count != 18) return error.BadInfo; // dsv41_rdma_collect_roci's 18 parameters
            const per = collect_threads * r.s.collect_grain;
            const blocks = std.math.clamp((@as(u64, n4) * r.world + per - 1) / per, 1, r.s.collect_blocks);
            return cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &b);
        }
        var a: cuda.Args = .{};
        a.add(own_src);
        a.add(recv);
        a.add(@as(c_int, @intCast(n4)));
        a.add(r.recv.dev);
        a.add(r.metaDev(r.flags()));
        a.add(r.scalar(0));
        a.add(@as(c_int, @intCast(r.rank)));
        a.add(@as(c_int, @intCast(r.world)));
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(u64, r.s.max_bytes));
        a.add(r.probe());
        a.add(r.metaDev(r.abortWord()));
        a.add(@as(c_int, @intFromBool(skip_own)));
        a.add(@as(c_int, @intFromBool(r.s.unroll)));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        const per = collect_threads * r.s.collect_grain;
        const blocks = std.math.clamp((@as(u64, n4) * r.world + per - 1) / per, 1, r.s.collect_blocks);
        try cuda.launch.launch(r.k.collect, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }

    /// recv [world x n4 float4] <- every rank's send [n4 float4] (fp32, 16-byte aligned), on `stream`. With the in-place
    /// switch on and `send` this stream's send slot (Round.init's pointer), the producer's bytes are published as they
    /// are: no staging copy; the collecting copy-out is unchanged.
    pub fn gather(r: *Ring, stream: cuda.Stream, send: DevicePtr, n4: u32, recv: DevicePtr) Error!void {
        if (r.failure() != null) return error.Failed;
        if (@as(usize, n4) * 16 > r.s.max_bytes) return error.TooLarge;
        if (r.sendSlotHeld(stream, send) != null) {
            try r.publish(stream, n4, 0, 0, 0);
            return r.launchGatherCollect(stream, send, false, n4, recv);
        }
        const own: DevicePtr = if (r.s.own_early) recv + @as(u64, r.rank) * n4 * 16 else 0;
        try r.stage(stream, send, n4, own, n4, n4);
        try r.launchGatherCollect(stream, send, r.s.own_early, n4, recv);
    }

    /// A rank's slice in `dsts` (its rows `row4` float4 wide, `stride4` apart; ptr 0: not copied out).
    pub const Dst = struct { ptr: DevicePtr = 0, row4: u32 = 0, stride4: u64 = 0 };

    /// Every rank's [rows, w_p] fp32 slice straight into dsts[p]; `send` is this rank's slice, contiguous. Every rank
    /// names the same widths, so each peer's ring slot holds rows x w_p floats.
    pub fn gatherInto(r: *Ring, stream: cuda.Stream, send: DevicePtr, rows: u32, own_row4: u32, dsts: []const Dst) Error!void {
        if (r.failure() != null) return error.Failed;
        if (dsts.len != r.world) return error.BadInfo;
        const n4 = rows * own_row4;
        if (@as(usize, n4) * 16 > r.s.max_bytes) return error.TooLarge;
        var dd: Dsts = .{};
        var copy4: u64 = 0;
        for (dsts, 0..) |t, p| {
            if (t.ptr == 0) continue;
            if (@as(usize, rows) * t.row4 * 16 > r.s.max_bytes) return error.TooLarge;
            dd.ptr[p] = t.ptr;
            dd.row4[p] = @intCast(t.row4);
            dd.stride4[p] = @intCast(t.stride4);
            if (p != r.rank) copy4 += @as(u64, rows) * t.row4;
        }
        const mine = dsts[r.rank];
        const own_row4n = if (mine.ptr != 0) mine.row4 else n4;
        const own_stride = if (mine.ptr != 0) mine.stride4 else n4;
        // With the generation-word peers kernel the copy-out copies this rank's own slice too (the publishing / staging kernel does not): its
        // doorbell is not behind the copy
        const fold_own = mine.ptr != 0 and r.roci_live and r.s.roci.on() and r.s.collect_peers and r.k.collect_peers_roci != null;
        const post_own: DevicePtr = if (fold_own) 0 else mine.ptr;
        if (r.sendSlotHeld(stream, send) != null) {
            try r.publish(stream, n4, post_own, own_row4n, own_stride);
        } else try r.stage(stream, send, n4, post_own, own_row4n, own_stride);
        try launchCollect(r.k, r.s, stream, dd, rows, .{
            .recv = r.recv.dev,
            .flags = r.metaDev(r.flags()),
            .seq = r.scalar(0),
            .abort = r.metaDev(r.abortWord()),
            .probe = r.probe(),
            .rank = r.rank,
            .world = r.world,
            .gen = if (r.roci_live) r.gen.?.dev else 0,
            .lanes = if (r.roci_live) r.lanes_dev.?.ptr else 0,
            .fault = if (r.roci_live) r.fault.?.ptr else 0,
            .deadline_ns = r.s.roci_deadline_ns,
            .roci = if (r.roci_live) r.s.roci else .off,
            .own_src = if (fold_own) send else 0,
            .own_dst = if (fold_own) mine.ptr else 0,
            .own_row4 = own_row4n,
            .own_stride4 = own_stride,
            .own_n4 = n4,
        }, copy4);
    }

    /// `probe` for the card-WAIT staging / publishing kernels: with TF_DS_ROCI_XLOG on and no probe running, the doorbell log's address tagged with
    /// bit 0 (the kernels strip it and log each doorbell); else the probe as it is.
    fn probeOrPubLog(r: *const Ring, logging: bool) DevicePtr {
        if (logging and !r.s.probing and r.fault != null) return (r.fault.?.ptr + 64 + xlog_recv_records * 32) | 1;
        return r.probe();
    }

    fn probe(r: *const Ring) DevicePtr {
        return if (r.s.probing) r.scalar(16) else 0;
    }

    fn stage(r: *Ring, stream: cuda.Stream, send: DevicePtr, n4: u32, own: DevicePtr, own_row4: u32, own_stride4: u64) Error!void {
        if (r.cd != null) return r.stageWait(stream, send, n4, own, own_row4, own_stride4);
        var a: cuda.Args = .{};
        a.add(send);
        a.add(@as(c_int, @intCast(n4)));
        a.add(r.send.dev);
        a.add(r.metaDev(r.sizes()));
        a.add(r.metaDev(r.doorbell()));
        a.add(r.metaDev(r.sent()));
        a.add(r.scalar(0));
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(u64, r.s.max_bytes));
        a.add(r.probe());
        a.add(r.metaDev(r.abortWord()));
        a.add(r.scalar(8));
        a.add(@as(c_int, @intFromBool(r.s.fence)));
        a.add(own);
        a.add(@as(c_int, @intCast(own_row4)));
        a.add(@as(c_longlong, @intCast(own_stride4)));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        const blocks = std.math.clamp((@as(u64, n4) + 1023) / 1024, 1, r.s.stage_blocks);
        try cuda.launch.launch(r.k.stage, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = stage_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }

    /// TF_DS_RING_WAIT's staging: the exchange's chains first (written or noted), then the kernel that copies the slice in and rings the doorbell
    /// (stored big-endian) once the card's completions of the exchange `slots` before have freed the slot.
    fn stageWait(r: *Ring, stream: cuda.Stream, send: DevicePtr, n4: u32, own: DevicePtr, own_row4: u32, own_stride4: u64) Error!void {
        try r.cdLaunch(stream, @as(u64, n4) * 16, -1);
        const sw = Kernels.pick(r.k.stage_wait, r.k.stage_wait_x);
        var a: cuda.Args = .{};
        a.add(send);
        a.add(@as(c_int, @intCast(n4)));
        a.add(r.send.dev);
        a.add(r.metaDev(r.sizes()));
        a.add(r.metaDev(r.doorbell()));
        a.add(r.cd_tab);
        a.add(r.scalar(0));
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(u64, r.s.max_bytes));
        a.add(r.probeOrPubLog(sw.logging));
        a.add(r.metaDev(r.abortWord()));
        a.add(r.scalar(8));
        a.add(@as(c_int, @intFromBool(r.s.fence)));
        a.add(own);
        a.add(@as(c_int, @intCast(own_row4)));
        a.add(@as(c_longlong, @intCast(own_stride4)));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        if (a.count != 17) return error.BadInfo; // dsv41_rdma_stage_wait's 17 parameters
        const blocks = std.math.clamp((@as(u64, n4) + 1023) / 1024, 1, r.s.stage_blocks);
        try cuda.launch.launch(sw.f.?, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = stage_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }

    /// (stage, doorbell to every flag, copy-out) mean ns over the probed gathers and their count; resets the sums.
    pub fn probes(r: *Ring) Error![4]f64 {
        var host: [8]u64 = undefined;
        try r.scalars.download(16, std.mem.sliceAsBytes(&host));
        try r.d.check(r.d.api.cuMemsetD8_v2(r.scalar(16), 0, 64), "cuMemsetD8");
        const n: f64 = if (host[3] > 0) @floatFromInt(host[3]) else 1;
        return .{ @as(f64, @floatFromInt(host[0])) / n, @as(f64, @floatFromInt(host[1])) / n, @as(f64, @floatFromInt(host[2])) / n, @floatFromInt(host[3]) };
    }
};

fn postRecv(ctx: *vabi.Context, qp: *vabi.Qp) Error!void {
    var wr: vabi.RecvWr = .{};
    var bad: ?*vabi.RecvWr = null;
    if (ctx.ops.post_recv(qp, &wr, &bad) != 0) return error.PostFailed;
}

fn postWrite(ctx: *vabi.Context, qp: *vabi.Qp, local: []const u8, lkey: u32, remote: u64, rkey: u32, wr_id: u64, imm: u32) Error!void {
    var sge: vabi.Sge = .{ .addr = @intFromPtr(local.ptr), .length = @intCast(local.len), .lkey = lkey };
    var wr: vabi.SendWr = .{ .wr_id = wr_id, .sg_list = @ptrCast(&sge), .num_sge = @intFromBool(local.len > 0), .opcode = vabi.wr_rdma_write_imm, .send_flags = vabi.send_signaled, .imm_data = imm, .remote_addr = remote, .rkey = rkey };
    var bad: ?*vabi.SendWr = null;
    if (ctx.ops.post_send(qp, &wr, &bad) != 0) return error.PostFailed;
}

/// TF_DS_RING_WORD's data: a plain RDMA WRITE, unsignalled (the lane's word write, posted after it on the same
/// queue pair, carries the completion; reliable-connection sends complete in post order, so the word's completion
/// also means the data has been read).
fn postData(ctx: *vabi.Context, qp: *vabi.Qp, local: []const u8, lkey: u32, remote: u64, rkey: u32, wr_id: u64) Error!void {
    var sge: vabi.Sge = .{ .addr = @intFromPtr(local.ptr), .length = @intCast(local.len), .lkey = lkey };
    var wr: vabi.SendWr = .{ .wr_id = wr_id, .sg_list = @ptrCast(&sge), .num_sge = 1, .opcode = vabi.wr_rdma_write, .send_flags = 0, .remote_addr = remote, .rkey = rkey };
    var bad: ?*vabi.SendWr = null;
    if (ctx.ops.post_send(qp, &wr, &bad) != 0) return error.PostFailed;
}

/// TF_DS_RING_WORD's signal, `.word`: an 8-byte inline RDMA WRITE of `local` (the sequence) to the peer's word.
fn postWord(ctx: *vabi.Context, qp: *vabi.Qp, local: []const u8, lkey: u32, remote: u64, rkey: u32, wr_id: u64) Error!void {
    var sge: vabi.Sge = .{ .addr = @intFromPtr(local.ptr), .length = 8, .lkey = lkey };
    var wr: vabi.SendWr = .{ .wr_id = wr_id, .sg_list = @ptrCast(&sge), .num_sge = 1, .opcode = vabi.wr_rdma_write, .send_flags = vabi.send_signaled | vabi.send_inline, .remote_addr = remote, .rkey = rkey };
    var bad: ?*vabi.SendWr = null;
    if (ctx.ops.post_send(qp, &wr, &bad) != 0) return error.PostFailed;
}

/// An RDMA ATOMIC fetch-and-add's work request: ibv_send_wr's atomic arm (the union begins where the RDMA arm's
/// remote address and key do), `compare_add` the addend and `swap` unused.
const AtomicWr = extern struct {
    wr_id: u64 = 0,
    next: ?*AtomicWr = null,
    sg_list: ?[*]vabi.Sge = null,
    num_sge: c_int = 0,
    opcode: c_int = 0,
    send_flags: c_uint = 0,
    imm_data: u32 = 0,
    remote_addr: u64 = 0,
    compare_add: u64 = 0,
    swap: u64 = 0,
    rkey: u32 = 0,
    _pad: [60]u8 = @splat(0),
};
comptime {
    if (@sizeOf(AtomicWr) != 128 or @offsetOf(AtomicWr, "remote_addr") != 40 or @offsetOf(AtomicWr, "compare_add") != 48 or @offsetOf(AtomicWr, "swap") != 56 or @offsetOf(AtomicWr, "rkey") != 64) @compileError("AtomicWr must be ibv_send_wr's atomic arm");
}

/// TF_DS_RING_WORD's signal, `.atomic`: one fetch-and-add of 1 on the peer's 64-bit word (the signal the IB spec
/// covers after a WRITE: a later ATOMIC is ordered after it and its memory element is visible).
fn postFetchAdd(ctx: *vabi.Context, qp: *vabi.Qp, remote: u64, rkey: u32, wr_id: u64) Error!void {
    var wr: AtomicWr = .{ .wr_id = wr_id, .opcode = vabi.wr_atomic_fetch_add, .send_flags = vabi.send_signaled, .remote_addr = remote, .compare_add = 1, .rkey = rkey };
    var bad: ?*vabi.SendWr = null;
    if (ctx.ops.post_send(qp, @ptrCast(&wr), &bad) != 0) return error.PostFailed;
}

test "every declaration compiles (the GPU and verbs paths run only on the nodes)" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Ring);
    std.testing.refAllDecls(Kernels);
}

    words: [64]u64 = undefined, // an inline 8-byte write's value, in post order
    n_words: usize = 0,
    adds: [64]u64 = undefined, // an ATOMIC fetch-and-add's addend, in post order
    n_adds: usize = 0,
        _ = d.posts.fetchAdd(1, .acq_rel);
        if (wr.opcode == vabi.wr_atomic_fetch_add) {
            const aw: *const AtomicWr = @ptrCast(@alignCast(wr));
            d.adds[d.n_adds] = aw.compare_add;
            d.n_adds += 1;
        } else if (wr.num_sge == 1 and wr.sg_list[0].length == 8) {
            const p: *const u64 = @ptrFromInt(wr.sg_list[0].addr);
            d.words[d.n_words] = p.*;
            d.n_words += 1;
        }
        // an unsignalled work request completes without a completion queue entry (a plain data write)
        if (wr.send_flags & vabi.send_signaled == 0) return 0;
    gen_mem: ?[]align(std.heap.page_size_min) u8 = null,
        t.gen_mem = null;
    /// A ring with TF_DS_RING_WORD set and runnable: a generation-word region (the fakery writes and reads it as
    /// the peers' NICs would) and every device's fake registration of it.
    fn initRoci(wfe: ?bool, nap_us: u32, sig: Roci) !TestRing {
        var t = try TestRing.init(wfe, nap_us);
        const gpa = std.testing.allocator;
        const page = comptime std.mem.Alignment.fromByteUnits(std.heap.page_size_min);
        const mem = try gpa.alignedAlloc(u8, page, rociWords() * 8);
        @memset(mem, 0);
        t.gen_mem = mem;
        t.r.gen = .{ .mem = mem, .dev = 0 };
        t.r.s.roci = sig;
        t.r.roci_live = true;
        for (t.devs, 0..) |d, i| t.r.nics[i].mr_gen = &d.mr;
        return t;
    }

        if (t.gen_mem) |g| gpa.free(g);
test "a watch's hook aborts a ring: its waits give up and the failure says why" {
    const link = @import("link.zig");
    var t = try TestRing.init(false, 0);
    defer t.deinit();
    try std.testing.expect(t.r.failure() == null);
    var fds: [2]std.c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &fds));
    defer _ = std.c.close(fds[0]);
    const Hook = struct {
        fn gone(ctx: ?*anyopaque, _: u32) void {
            const r: *Ring = @ptrCast(@alignCast(ctx.?));
            r.abort("a peer rank's link closed: the rank stopped");
        }
        fn exit(_: *link.Watch, _: u32) void {}
    };
    var w: link.Watch = .{ .rank = 0, .grace_ms = 100, .stop_hook = Hook.gone, .ctx = t.r, .on_gone = Hook.exit };
    try w.add(.{ .fd = fds[0] }, 1);
    try w.start();
    defer w.stop();
    _ = std.c.close(fds[1]);
    var waited: u32 = 0;
    while (t.r.failure() == null and waited < 4000) : (waited += 10) {
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 10 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
    try std.testing.expectEqualStrings("a peer rank's link closed: the rank stopped", t.r.failure() orelse return error.TestUnexpectedResult);
    try std.testing.expectEqual(@as(u64, 1), @atomicLoad(u64, t.r.abortWord(), .acquire));
}

test "TF_DS_RING_WORD's signals: the word is the sequence, the atomic counts, and a wait ends at or past it" {
    try std.testing.expect(!Roci.off.on());
    try std.testing.expect(Roci.word.on() and Roci.atomic.on());
    try std.testing.expectEqual(@as(u64, 7), ringWordSignalWord(7));
    try std.testing.expect(Roci.reached(7, 7));
    try std.testing.expect(Roci.reached(8, 7));
    try std.testing.expect(!Roci.reached(6, 7));
    // the switch word a peer compares: the switch in bit 2, the atomic signal in bit 3
    var r: Ring = undefined;
    r.s = .{ .max_bytes = 64 };
    r.s.roci = .word;
    try std.testing.expectEqual(@as(u32, 0b0100), r.switchBits());
    r.s.roci = .atomic;
    try std.testing.expectEqual(@as(u32, 0b1100), r.switchBits());
    r.s.roci = .off;
    try std.testing.expectEqual(@as(u32, 0), r.switchBits());
}

test "TF_DS_RING_WORD refused at connect: the copy-out takes the flag kernels, not the generation-word kernels with null words" {
    // a ring whose setting asks for the word but whose connect refused it (the card's atomic_cap, a peer without words) passes no
    // gen / lanes / fault: the word kernels would read address 0 (CUDA_ERROR_ILLEGAL_ADDRESS)
    const refused: CollectRing = .{ .recv = 1, .flags = 2, .seq = 3, .abort = 4, .rank = 0, .world = 4, .roci = .off };
    for ([_]Roci{ .word, .atomic }) |signal| {
        const s: Settings = .{ .max_bytes = 64, .roci = signal };
        try std.testing.expect(!collectsOnWords(s, refused));
        var live = refused;
        live.roci = signal;
        live.gen = 5;
        live.lanes = 6;
        live.fault = 7;
        try std.testing.expect(collectsOnWords(s, live));
    }
    try std.testing.expect(!collectsOnWords(.{ .max_bytes = 64 }, refused));
}

test "TF_DS_RING_WORD: the proxy posts each lane's data, then its generation word, catching up every pending sequence" {
    var t = try TestRing.initRoci(false, 2_000_000, .word); // spin: the test drives the doorbell
    defer t.deinit();
    try t.r.start();
    defer t.r.stop();
    t.ring(1);
    t.ring(2);
    t.ring(3);
    try t.until(struct {
        fn done(x: *TestRing) bool {
            return x.sent(3) == 3;
        }
    }.done);
    // three sequences, two lanes each, a data write and a word write a lane
    try std.testing.expectEqual(@as(u32, 3 * 2 * 2), t.posts());
    // the proxy catches up every sequence the doorbell names (a counter, never a level), and the word is the sequence
    try std.testing.expectEqual(@as(u64, 3), t.sent(3));
    for (t.devs) |d| {
        try std.testing.expectEqual(@as(usize, 3), d.n_words);
        try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3 }, d.words[0..3]);
    }
}

test "TF_DS_RING_WORD: a doorbell further ahead than the ring's slots fails the ring instead of skipping a sequence" {
    var t = try TestRing.initRoci(false, 2_000_000, .word);
    defer t.deinit();
    // the staging kernel cannot ring more than slots ahead; a doorbell that does means a sequence was skipped
    t.ring(t.r.s.slots + 1);
    try t.r.start();
    defer t.r.stop();
    try t.until(struct {
        fn failed(x: *TestRing) bool {
            return x.r.failure() != null;
        }
    }.failed);
    const why = t.r.failure().?;
    try std.testing.expect(std.mem.indexOf(u8, why, "more sequences than the ring's slots") != null);
}

test "TF_DS_RING_WORD's atomic signal posts one fetch-and-add on each lane's word" {
    var t = try TestRing.initRoci(false, 2_000_000, .atomic);
    defer t.deinit();
    try t.r.start();
    defer t.r.stop();
    t.ring(1);
    t.ring(2);
    try t.until(struct {
        fn done(x: *TestRing) bool {
            return x.sent(2) == 2;
        }
    }.done);
    try std.testing.expectEqual(@as(u32, 2 * 2 * 2), t.posts());
    for (t.devs) |d| {
        try std.testing.expectEqual(@as(usize, 2), d.n_adds);
        try std.testing.expectEqualSlices(u64, &.{ 1, 1 }, d.adds[0..2]);
        // no inline word: the atomic carries the signal
        try std.testing.expectEqual(@as(usize, 0), d.n_words);
    }
}

test "the info a peer reads is a plain extern struct (it crosses the link as bytes)" {
    // TF_DS_RING_WAIT appended each device's port MAC (24 bytes, zero unless the switch is on): the other fields did not move
    try std.testing.expectEqual(@as(usize, 176), @sizeOf(Info));
    try std.testing.expectEqual(@as(usize, 152), @offsetOf(Info, "mac"));
    try std.testing.expectEqual(@as(usize, 120), @offsetOf(Info, "recv"));
    try std.testing.expectEqual(@as(usize, 128), @offsetOf(Info, "gen"));
    try std.testing.expectEqual(@as(usize, 136), @offsetOf(Info, "gen_rkey"));
}

test "TF_DS_RDMA_COLLECT_PEERS' grid: a row of blocks a peer copied out, as many as the widest peer's units up to its share of the collecting blocks" {
    // the modal all4 exchange: 2 rows of 640 float4 (10,240 bytes) from each of 3 peers, a block a row a peer
    try std.testing.expectEqual(PeersGrid{ .x = 2, .y = 3 }, peersGrid(2, &.{ 640, 640, 640, 640 }, 1, 48));
    // the head's quarters (8,096 / 8,064 float4 a row, 8 units each): 16 a peer, a third of the 48
    try std.testing.expectEqual(PeersGrid{ .x = 16, .y = 3 }, peersGrid(2, &.{ 8096, 8064, 8096, 8064 }, 0, 48));
    // 26 rows (16 streams) of 640: 26 units a peer, capped at 16
    try std.testing.expectEqual(PeersGrid{ .x = 16, .y = 3 }, peersGrid(26, &.{ 640, 640, 640, 640 }, 2, 48));
    // the column ring's one peer: its units up to all 48
    try std.testing.expectEqual(PeersGrid{ .x = 2, .y = 1 }, peersGrid(2, &.{ 256, 256 }, 0, 48));
    try std.testing.expectEqual(PeersGrid{ .x = 48, .y = 1 }, peersGrid(34, &.{ 8096, 8096 }, 1, 48));
    // a null destination (row4 0) is not copied out, nor this rank's own slice
    try std.testing.expectEqual(PeersGrid{ .x = 3, .y = 2 }, peersGrid(3, &.{ 640, 0, 640, 640 }, 3, 48));
    // none copied out: a grid of one
    try std.testing.expectEqual(PeersGrid{ .x = 1, .y = 1 }, peersGrid(2, &.{ 0, 640 }, 1, 48));
    // a budget of one block still gives each peer one
    try std.testing.expectEqual(PeersGrid{ .x = 1, .y = 3 }, peersGrid(2, &.{ 640, 640, 640, 640 }, 0, 1));
    // a row of 1,025 float4: two units
    try std.testing.expectEqual(PeersGrid{ .x = 2, .y = 1 }, peersGrid(1, &.{ 1025, 1025 }, 0, 48));
}

test "collect_peers' block rows skip a zero-width destination and keep every copied peer's row" {
    // rank 1's destination has a pointer but row4 = 0: the host's grid excludes it. Ranks 2 and 3 are copied out;
    // rank 0 is this rank. The kernel's y-th row must select 2 and 3, not 1 and 2.
    const d: Dsts = .{
        .ptr = .{ 0x1000, 0x2000, 0x3000, 0x4000, 0, 0, 0, 0 },
        .row4 = .{ 64, 0, 64, 64, 0, 0, 0, 0 },
        .stride4 = .{ 64, 64, 64, 64, 0, 0, 0, 0 },
    };
    try std.testing.expectEqual(PeersGrid{ .x = 2, .y = 2 }, peersGrid(2, &.{ 64, 0, 64, 64 }, 0, 48));
    try std.testing.expect(!peerCopied(d, 0, 0)); // this rank's own slice
    try std.testing.expect(!peerCopied(d, 1, 0)); // a pointer but row4 0: not copied out
    try std.testing.expect(peerCopied(d, 2, 0));
    try std.testing.expect(peerCopied(d, 3, 0));
    try std.testing.expectEqual(@as(?u32, 2), peerAtRow(d, 4, 0, 0));
    try std.testing.expectEqual(@as(?u32, 3), peerAtRow(d, 4, 0, 1));
    try std.testing.expectEqual(@as(?u32, null), peerAtRow(d, 4, 0, 2));
}

test "TF_DS_RDMA_COLLECT_PEERS is off by default; gatherInto takes dsv41_rdma_collect_peers only when it is set and the fatbin has the kernel" {
    const s: Settings = .{ .max_bytes = 64 };
    try std.testing.expect(!s.collect_peers);
    const src = @embedFile("rdma.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, ".collect_peers = m.function(\"dsv41_rdma_collect_peers\") catch null") != null);
    const at = std.mem.indexOf(u8, src, "pub fn launchCollect(") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n}\n") orelse return error.TestUnexpectedResult)];
    const peers = std.mem.indexOf(u8, body, "if (s.collect_peers) if (k.collect_peers) |f| {") orelse return error.TestUnexpectedResult;
    const grid = std.mem.indexOf(u8, body, "const g = peersGrid(rows, row4s[0..ring.world], ring.rank, s.collect_blocks);") orelse return error.TestUnexpectedResult;
    const today = std.mem.indexOf(u8, body, "try cuda.launch.launch(k.collect_into,") orelse return error.TestUnexpectedResult;
    try std.testing.expect(peers < grid and grid < today);
    // gather_into stages, then launches its copy-out through launchCollect alone
    const gi = std.mem.indexOf(u8, src, "pub fn gatherInto(") orelse return error.TestUnexpectedResult;
    const gbody = src[gi .. gi + (std.mem.indexOf(u8, src[gi..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, gbody, "try launchCollect(r.k, r.s, stream, dd, rows, "));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, gbody, "cuda.launch.launch("));
}

test "lanes pair each subnet's devices in list order; a peer across the ring has none" {
    const g = struct {
        fn of(a: u8, b: u8, c: u8, d: u8) [16]u8 {
            return .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, a, b, c, d };
        }
    }.of;
    // spark1: f0 toward rank 3 (10.0.41.x), f1 toward rank 1 (10.0.22.x), and their PCIe twins (.10x)
    const s1 = [_][16]u8{ g(10, 0, 41, 1), g(10, 0, 22, 1), g(10, 0, 41, 101), g(10, 0, 22, 101) };
    const s2 = [_][16]u8{ g(10, 0, 33, 2), g(10, 0, 22, 2), g(10, 0, 33, 102), g(10, 0, 22, 102) };
    const s3 = [_][16]u8{ g(10, 0, 33, 3), g(10, 0, 34, 3), g(10, 0, 33, 103), g(10, 0, 34, 103) };
    var out: [max_devices][2]u8 = undefined;
    try std.testing.expectEqual(@as(?u32, 2), pairLanes(&s1, &s2, 24, &out));
    try std.testing.expectEqual([2]u8{ 1, 1 }, out[0]);
    try std.testing.expectEqual([2]u8{ 3, 3 }, out[1]);
    try std.testing.expectEqual(@as(?u32, 2), pairLanes(&s2, &s1, 24, &out));
    try std.testing.expectEqual([2]u8{ 1, 1 }, out[0]);
    try std.testing.expectEqual(@as(?u32, 0), pairLanes(&s1, &s3, 24, &out));
    // /26 tells the twins apart: .1 with .2, .101 with .102
    try std.testing.expectEqual(@as(?u32, 2), pairLanes(&s1, &s2, 26, &out));
    try std.testing.expect(sameNet(0x0a001601, 0x0a001602, 26));
    try std.testing.expect(!sameNet(0x0a001601, 0x0a001666, 26));
    // a link-local GID: no routes by subnet
    const ll = [_][16]u8{.{ 0xfe, 0x80, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8 }};
    try std.testing.expectEqual(@as(?u32, null), pairLanes(&ll, &s2, 24, &out));
}

test "TF_DS_RING_INPLACE needs TF_DS_RING_WC: the setting is refused in words and no ring is made without it" {
    try std.testing.expectEqual(@as(?[]const u8, null), inplaceOff(false, false));
    try std.testing.expectEqual(@as(?[]const u8, null), inplaceOff(true, false));
    try std.testing.expectEqual(@as(?[]const u8, null), inplaceOff(true, true));
    try std.testing.expectEqualStrings("needs TF_DS_RING_WC=1 (the send slot is a write-combined data slot its producer writes)", inplaceOff(false, true).?);
    const src = @embedFile("rdma.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "if (inplaceOff(s.wc, s.inplace) != null) return error.BadInfo;") != null);
    const s: Settings = .{ .max_bytes = 64 };
    try std.testing.expect(!s.wc and !s.inplace); // both off by default
}

test "an in-place ring puts each stream's fixed send slot after the rotating ones, and only while the switch is on" {
    var t = try TestRing.init(false, 1000);
    defer t.deinit();
    const a: cuda.abi.Stream = @ptrFromInt(0x1000);
    const b: cuda.abi.Stream = @ptrFromInt(0x2000);
    const c: cuda.abi.Stream = @ptrFromInt(0x3000);
    // off: no slot, nothing assigned
    try std.testing.expectEqual(@as(?u32, null), try t.r.userSlot(.{ .d = undefined, .handle = a }));
    try std.testing.expectEqual(@as(u32, 0), t.r.n_users);
    try std.testing.expectEqual(@as(usize, 4 * 64), t.r.sendBytes());
    // on: a stream keeps its index; the slots start after the rotating ones; the region grows by max_users
    t.r.s.wc = true;
    t.r.s.inplace = true;
    const sa: cuda.Stream = .{ .d = undefined, .handle = a };
    const sb: cuda.Stream = .{ .d = undefined, .handle = b };
    const sc: cuda.Stream = .{ .d = undefined, .handle = c };
    try std.testing.expectEqual(@as(?u32, 0), try t.r.userSlot(sa));
    try std.testing.expectEqual(@as(?u32, 0), try t.r.userSlot(sa));
    try std.testing.expectEqual(@as(?u32, 1), try t.r.userSlot(sb));
    try std.testing.expectError(error.BadInfo, t.r.userSlot(sc)); // max_users reached
    try std.testing.expectEqual(@as(usize, 4), t.r.userSendSlot(0));
    try std.testing.expectEqual(@as(usize, 5), t.r.userSendSlot(1));
    try std.testing.expectEqual(@as(usize, (4 + max_users) * 64), t.r.sendBytes());
    try std.testing.expectEqual(@as(?u32, 0), t.r.sendSlotHeld(sa, 4 * 64));
    try std.testing.expectEqual(@as(?u32, null), t.r.sendSlotHeld(sa, 0)); // not this stream's slot
    try std.testing.expectEqual(@as(?u32, null), t.r.sendSlotHeld(sc, (4 + max_users - 1) * 64)); // unassigned stream
    try std.testing.expectEqual(@as(u32, 3), t.r.switchBits());
}

test "the in-place words follow the ring's own metadata and keep the doorbell, sizes, sent and abort where they were" {
    var t = try TestRing.init(false, 1000);
    defer t.deinit();
    const base = @intFromPtr(t.r.flags());
    const words = t.r.s.slots * t.r.world + 1 + 2 * t.r.s.slots + 1;
    try std.testing.expectEqual(@as(usize, words + 2 * t.r.s.slots + 2 * max_users), t.r.metaWords());
    try std.testing.expectEqual(base + (t.r.s.slots * t.r.world) * 8, @intFromPtr(t.r.doorbell()));
    try std.testing.expectEqual(base + (t.r.s.slots * t.r.world + 1) * 8, @intFromPtr(t.r.sizes()));
    try std.testing.expectEqual(base + (t.r.s.slots * t.r.world + 1 + t.r.s.slots) * 8, @intFromPtr(t.r.sent()));
    try std.testing.expectEqual(base + (words - 1) * 8, @intFromPtr(t.r.abortWord()));
    try std.testing.expectEqual(base + words * 8, @intFromPtr(t.r.sendQ()));
    try std.testing.expectEqual(base + (words + t.r.s.slots) * 8, @intFromPtr(t.r.sendSlotWord()));
    try std.testing.expectEqual(base + (words + 2 * t.r.s.slots) * 8, @intFromPtr(t.r.sendLast()));
    try std.testing.expectEqual(base + (words + 2 * t.r.s.slots + max_users) * 8, @intFromPtr(t.r.sendDone()));
    // a set of slots that differs is refused: the peers lay slots out the same way or not at all
    var o = t.r.*;
    o.s.wc = true;
    try std.testing.expect(t.r.switchBits() != o.switchBits());
}

test "gather and gatherInto take the in-place publish only for a stream's own send slot, and stage otherwise" {
    const src = @embedFile("rdma.zig");
    const g = std.mem.indexOf(u8, src, "pub fn gather(r: *Ring,") orelse return error.TestUnexpectedResult;
    const gend = g + (std.mem.indexOf(u8, src[g..], "\n    }\n") orelse return error.TestUnexpectedResult);
    const gbody = src[g..gend];
    const held = std.mem.indexOf(u8, gbody, "if (r.sendSlotHeld(stream, send) != null) {") orelse return error.TestUnexpectedResult;
    const publish_at = std.mem.indexOf(u8, gbody, "try r.publish(stream, n4, 0, 0, 0);") orelse return error.TestUnexpectedResult;
    const stage = std.mem.indexOf(u8, gbody, "try r.stage(stream, send, n4, own, n4, n4);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(held < publish_at and publish_at < stage);
    const gi = std.mem.indexOf(u8, src, "pub fn gatherInto(") orelse return error.TestUnexpectedResult;
    const giend = gi + (std.mem.indexOf(u8, src[gi..], "\n    }\n") orelse return error.TestUnexpectedResult);
    const gbody2 = src[gi..giend];
    const held2 = std.mem.indexOf(u8, gbody2, "if (r.sendSlotHeld(stream, send) != null) {") orelse return error.TestUnexpectedResult;
    const publish_at2 = std.mem.indexOf(u8, gbody2, "try r.publish(stream, n4, post_own, own_row4n, own_stride);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(held2 < publish_at2);
    // the awaiting producer is the caller's: the ring offers it, the gather does not wait after the producer
    try std.testing.expect(std.mem.indexOf(u8, src, "pub fn prepareSend(r: *Ring, stream: cuda.Stream) Error!bool {") != null);
}

test "TF_DS_RING_WC is off by default; a ring's switch word carries it, and the peers must agree" {
    const s: Settings = .{ .max_bytes = 64 };
    try std.testing.expect(!s.wc);
    var t = try TestRing.init(false, 1000);
    defer t.deinit();
    try std.testing.expectEqual(@as(u32, 0), t.r.switchBits());
    t.r.s.wc = true;
    try std.testing.expectEqual(@as(u32, 1), t.r.switchBits());
    // the switch changes pages, not the ring's fields: a Pinned made by alloc (not allocHost) frees by unregister
    const p: Pinned = .{ .mem = t.r.send.mem, .dev = 0 };
    try std.testing.expect(!p.host and p.d == null);
}

test "TF_DS_RING_DEAD_MS: unset is the default; a whole number from 1 is taken; anything else refuses (null), never the default" {
    try std.testing.expectEqual(@as(?u64, default_dead_ms), deadMsFrom(null));
    try std.testing.expectEqual(@as(?u64, 2500), deadMsFrom("2500"));
    try std.testing.expectEqual(@as(?u64, 1), deadMsFrom("1"));
    for ([_][]const u8{ "", "0", "-5", "10s", "1e4", " 100", "x" }) |t| try std.testing.expectEqual(@as(?u64, null), deadMsFrom(t));
}

test "TF_DS_RING_WAIT: the Settings struct's own default is the proxy path (the engine's default is Model.open's, the card's WAIT); a ring's switch word carries it only when set, its info carries no MAC then, the proxy path's kernels and arguments are the ones they were" {
    const s: Settings = .{ .max_bytes = 64 };
    try std.testing.expect(!s.wait);
    var t = try TestRing.init(false, 1000);
    defer t.deinit();
    try std.testing.expectEqual(@as(u32, 0), t.r.switchBits() & 16);
    try std.testing.expect(t.r.cd == null and t.r.cd_n == 0 and t.r.mlx == null);
    const inf = t.r.info(1);
    for (inf.mac) |m| try std.testing.expectEqualSlices(u8, &(@as([6]u8, @splat(0))), &m);
    t.r.s.wait = true;
    try std.testing.expectEqual(@as(u32, 16), t.r.switchBits() & 16);
    // the kernels' optional entry points are all there or the mode cannot run
    var k: Kernels = undefined;
    k.stage_wait = null;
    k.publish_wait = null;
    k.await_send_wait = null;
    try std.testing.expect(!k.hasWait());
}

test "TF_DS_RING_WAIT's order of events is in the source: the chains before the doorbell kernel, the queue pairs off the card before the abort word and before anything the card reads is released, the proxy path untouched" {
    const src = @embedFile("rdma.zig");
    // a staged exchange: the chains are written (or noted) before the staging kernel that rings the doorbell is launched
    const sw = std.mem.indexOf(u8, src, "fn stageWait(") orelse return error.TestUnexpectedResult;
    const swbody = src[sw .. sw + (std.mem.indexOf(u8, src[sw..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    const post_at = std.mem.indexOf(u8, swbody, "try r.cdLaunch(stream, @as(u64, n4) * 16, -1);") orelse return error.TestUnexpectedResult;
    const kern_at = std.mem.indexOf(u8, swbody, "cuda.launch.launch(sw.f.?") orelse return error.TestUnexpectedResult;
    try std.testing.expect(post_at < kern_at);
    // an in-place publish: the same
    const pb = std.mem.indexOf(u8, src, "fn publish(r: *Ring") orelse return error.TestUnexpectedResult;
    const pbbody = src[pb .. pb + (std.mem.indexOf(u8, src[pb..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expect(std.mem.indexOf(u8, pbbody, "if (waiting) try r.cdLaunch(stream, @as(u64, n4) * 16, @intCast(u));").? < std.mem.indexOf(u8, pbbody, "cuda.launch.launch(f,").?);
    // the proxy path's stage() is what it was, after the one branch
    const st = std.mem.indexOf(u8, src, "    fn stage(r: *Ring, stream") orelse return error.TestUnexpectedResult;
    const stbody = src[st .. st + (std.mem.indexOf(u8, src[st..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expect(std.mem.indexOf(u8, stbody, "if (r.cd != null) return r.stageWait(").? < std.mem.indexOf(u8, stbody, "a.add(r.metaDev(r.sent()));").?);
    try std.testing.expect(std.mem.indexOf(u8, stbody, "cuda.launch.launch(r.k.stage,") != null);
    // a failure takes the queue pairs from the card, then sets the abort word
    const fl = std.mem.indexOf(u8, src, "    fn failLocal(r: *Ring, why: []const u8) void {") orelse return error.TestUnexpectedResult;
    const flbody = src[fl .. fl + (std.mem.indexOf(u8, src[fl..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expect(std.mem.indexOf(u8, flbody, "if (r.cd) |c| c.fence();").? < std.mem.indexOf(u8, flbody, "@atomicStore(u64, r.abortWord(), 1, .release);").?);
    // teardown: the queue pairs before the registrations and before the pinned memory
    const ds = std.mem.indexOf(u8, src, "    pub fn destroy(r: *Ring, gpa: std.mem.Allocator) void {") orelse return error.TestUnexpectedResult;
    const dsbody = src[ds .. ds + (std.mem.indexOf(u8, src[ds..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    const wc = std.mem.indexOf(u8, dsbody, "r.waitClose();") orelse return error.TestUnexpectedResult;
    try std.testing.expect(wc < std.mem.indexOf(u8, dsbody, "r.closeNic(n)").?);
    try std.testing.expect(wc < std.mem.indexOf(u8, dsbody, "r.meta.free(&r.hr);").?);
    try std.testing.expect(std.mem.indexOf(u8, dsbody, "r.stop();").? < wc);
    // the proxy path's stop() still wakes the gate and joins the proxy
    const sp = std.mem.indexOf(u8, src, "    pub fn stop(r: *Ring) void {") orelse return error.TestUnexpectedResult;
    const spbody = src[sp .. sp + (std.mem.indexOf(u8, src[sp..], "\n    }\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expect(std.mem.indexOf(u8, spbody, "r.idle_gate.wake();") != null and std.mem.indexOf(u8, spbody, "if (r.proxy) |t| t.join();") != null);
}

test "TF_DS_RING_WAIT's conditions: the word signal only, and a ring whose lanes are not all direct refuses by name" {
    const src = @embedFile("rdma.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "if (s.wait and (s.roci != .word or !k.hasWait()))") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "if (r.s.wait and !r.roci_live) {") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "TF_DS_RING_WAIT=1 refused: the generation word cannot run on this ring") != null);
    // a MAC from sysfs's text
    try std.testing.expectEqualSlices(u8, &.{ 0x00, 0x11, 0x22, 0x33, 0x44, 0xaf }, &parseMac("00:11:22:33:44:af\n").?);
    try std.testing.expect(parseMac("00:11:22:33:44") == null);
    try std.testing.expect(parseMac("00:11:22:33:44:55:66") == null);
}

test "the ring word's atomic check: no atomics or no atomics in flight refuse, the card's own atomics (HCA, the CX-7's) and global ones are enough" {
    var buf: [300]u8 = undefined;
    // the word has one remote writer (the peer's queue pair) and the GPU only reads it: the atomic is for the order after the WRITEs
    try std.testing.expect(atomicRefusal(&buf, "mlx5_0", vabi.atomic_hca, 16, 16) == null);
    try std.testing.expect(atomicRefusal(&buf, "mlx5_2", vabi.atomic_glob, 16, 16) == null);
    const none = atomicRefusal(&buf, "mlx5_0", 0, 16, 16).?;
    try std.testing.expect(std.mem.indexOf(u8, none, "mlx5_0") != null and std.mem.indexOf(u8, none, "atomic_cap 0") != null);
    const resp = atomicRefusal(&buf, "mlx5_0", vabi.atomic_hca, 0, 16).?;
    try std.testing.expect(std.mem.indexOf(u8, resp, "in flight") != null);
    try std.testing.expect(atomicRefusal(&buf, "mlx5_0", vabi.atomic_hca, 16, 0) != null);
    // the depth a queue pair is set up for: the slots, capped by both limits of the card, at least one
    try std.testing.expectEqual(@as(u8, 4), atomicDepth(4, 16, 16));
    try std.testing.expectEqual(@as(u8, 2), atomicDepth(4, 2, 16));
    try std.testing.expectEqual(@as(u8, 3), atomicDepth(4, 16, 3));
    try std.testing.expectEqual(@as(u8, 1), atomicDepth(4, 0, 0));
    // the connect check is this one (not a global-only demand)
    const src = @embedFile("rdma.zig");
    const needle = "n.atomic_cap < vabi." ++ "atomic_glob";
    try std.testing.expect(std.mem.indexOf(u8, src, needle) == null);
    try std.testing.expect(std.mem.indexOf(u8, src, "atomicRefusal(&rb, n.name, n.atomic_cap") != null);
}

test "a refused TF_DS_RING_WAIT names itself as it was set: 1 says 1, the default says unset" {
    var buf: [200]u8 = undefined;
    try std.testing.expect(std.mem.startsWith(u8, waitRefusalHead(&buf, true), "TF_DS_RING_WAIT=1 refused"));
    const d = waitRefusalHead(&buf, false);
    try std.testing.expect(std.mem.indexOf(u8, d, "unset") != null and std.mem.indexOf(u8, d, "TF_DS_RING_WAIT=1") == null);
}

const ModelCard = struct {
    outcome: enum { ok, no_atomics, open_fails, query_fails },
    fn caps(self: ModelCard, _: []const u8) !AtomicCaps {
        return switch (self.outcome) {
            .ok => .{ .atomic_cap = vabi.atomic_hca, .max_qp_rd_atom = 16, .max_qp_init_rd_atom = 16 },
            .no_atomics => .{ .atomic_cap = 0, .max_qp_rd_atom = 16, .max_qp_init_rd_atom = 16 },
            .open_fails => error.NoSuchDevice,
            .query_fails => error.QueryFailed,
        };
    }
};

test "the atomic probe's outcomes on a card model: can is null; a card without atomics, a device that will not open and one that will not answer are each a reason naming the device (never 'can')" {
    var buf: [300]u8 = undefined;
    const names = [_][]const u8{ "mlx5_0", "mlx5_2" };
    try std.testing.expect(probeAtomicWith(&buf, &names, ModelCard{ .outcome = .ok }) == null);
    const na = probeAtomicWith(&buf, &names, ModelCard{ .outcome = .no_atomics }).?;
    try std.testing.expect(std.mem.indexOf(u8, na, "mlx5_0") != null and std.mem.indexOf(u8, na, "atomic_cap 0") != null);
    const of = probeAtomicWith(&buf, &names, ModelCard{ .outcome = .open_fails }).?;
    try std.testing.expect(std.mem.indexOf(u8, of, "mlx5_0") != null and std.mem.indexOf(u8, of, "NoSuchDevice") != null);
    const qf = probeAtomicWith(&buf, &names, ModelCard{ .outcome = .query_fails }).?;
    try std.testing.expect(std.mem.indexOf(u8, qf, "QueryFailed") != null);
}

test "two ranks that set the card-waiting queue pairs up for different atomic depths (or one atomic and one word) have different switch words, so connect refuses them" {
    var a = try TestRing.init(false, 1000);
    defer a.deinit();
    var b = try TestRing.init(false, 1000);
    defer b.deinit();
    for ([_]*TestRing{ &a, &b }) |t| {
        t.r.s.wait = true;
        t.r.s.roci = .atomic;
    }
    try std.testing.expectEqual(a.r.switchBits(), b.r.switchBits()); // (no queue pairs in the test ring: depth 0 on both)
    try std.testing.expectEqual(@as(u32, 0), a.r.switchBits() >> 8);
    // with queue pairs (a stand-in backend pointer; no devices, so the depth is the slots) the depth is in the word
    const fake: *wait.Backend = @ptrFromInt(0x1000);
    a.r.cd = fake;
    b.r.cd = fake;
    defer {
        a.r.cd = null;
        b.r.cd = null;
    }
    a.r.s.slots = 4;
    b.r.s.slots = 2;
    try std.testing.expectEqual(@as(u32, 4), a.r.switchBits() >> 8);
    try std.testing.expect(a.r.switchBits() != b.r.switchBits());
    b.r.s.slots = 4;
    try std.testing.expectEqual(a.r.switchBits(), b.r.switchBits());
    // one rank on the word signal against one on the atomic differ too
    b.r.s.roci = .word;
    try std.testing.expect(a.r.switchBits() != b.r.switchBits());
    // the depth is in bits 8-15 whenever queue pairs exist
    const src = @embedFile("rdma.zig");
    try std.testing.expect(std.mem.indexOf(u8, src, "(@as(u32, r.cdDepth()) << 8)") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "every rank must choose alike") != null);
}

test "the exchange log: device words decode to records in ring order, empty slots left out, and out.len bounds them" {
    // a fake log buffer as the kernel writes it: slot ((q * 8 + peer) & 32767), words {q << 3 | peer, clock}
    var words: [32]u64 = @splat(0); // 8 slots of 4 words
    words[4 * 1] = (5 << 3) | 1;
    words[4 * 1 + 1] = 1000;
    words[4 * 1 + 2] = 400;
    words[4 * 1 + 3] = 1900;
    words[4 * 4] = (6 << 3) | 2;
    words[4 * 4 + 1] = 2500;
    words[4 * 6] = (7 << 3) | 3;
    words[4 * 6 + 1] = 0; // a slot whose clock is 0 was never written
    var out: [4]XlogRec = undefined;
    try std.testing.expectEqual(@as(usize, 2), xlogDecode(&words, &out));
    try std.testing.expectEqual(XlogRec{ .seq = 5, .peer = 1, .at_ns = 1000, .start_ns = 400, .end_ns = 1900 }, out[0]);
    try std.testing.expectEqual(XlogRec{ .seq = 6, .peer = 2, .at_ns = 2500 }, out[1]);
    var one: [1]XlogRec = undefined;
    try std.testing.expectEqual(@as(usize, 1), xlogDecode(&words, &one));
}

test "the doorbell log: device words {sequence, clock} decode in ring order, empty slots out" {
    var words: [16]u64 = @splat(0); // 8 slots
    words[2 * 3] = 12;
    words[2 * 3 + 1] = 5000;
    words[2 * 5] = 13; // a slot with a sequence and no clock: never written
    var out: [4]XlogPub = undefined;
    try std.testing.expectEqual(@as(usize, 1), xlogPubDecode(&words, &out));
    try std.testing.expectEqual(XlogPub{ .seq = 12, .at_ns = 5000 }, out[0]);
}

test "the exchange log file: a header naming rank, world and ring, then the records sorted by sequence and peer" {
    const gpa = std.testing.allocator;
    var recs = [_]XlogRec{ .{ .seq = 9, .peer = 2, .at_ns = 30, .start_ns = 5, .end_ns = 40 }, .{ .seq = 8, .peer = 3, .at_ns = 20, .start_ns = 4, .end_ns = 33 }, .{ .seq = 8, .peer = 1, .at_ns = 10, .start_ns = 4, .end_ns = 29 } };
    const text = try xlogRender(gpa, 1, 4, "the decode ring", 8, &recs);
    defer gpa.free(text);
    try std.testing.expectEqualStrings("# TF_DS_ROCI_XLOG v1 rank=1 world=4 ring=\"the decode ring\" slots=8 records=3 clock=globaltimer_ns\n8 1 10 4 29\n8 3 20 4 33\n9 2 30 5 40\n", text);
    // with doorbell records (this rank's own publishes): a second block, sorted by sequence, after a line naming it
    var pubs = [_]XlogPub{ .{ .seq = 9, .at_ns = 77 }, .{ .seq = 8, .at_ns = 66 } };
    const text2 = try xlogRenderPub(gpa, 1, 4, "the decode ring", 8, &recs, &pubs);
    defer gpa.free(text2);
    try std.testing.expectEqualStrings("# TF_DS_ROCI_XLOG v1 rank=1 world=4 ring=\"the decode ring\" slots=8 records=3 pubs=2 clock=globaltimer_ns\n8 1 10 4 29\n8 3 20 4 33\n9 2 30 5 40\nP 8 66\nP 9 77\n", text2);
}

test "the exchange log's file for a ring: the four nodes' ring keeps roci-xlog-rank<R>.txt, the column and row rings' files are told apart (one name for all three would overwrite the first)" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("roci-xlog-rank2.txt", xlogFileName(&b, 2, "the four nodes' ring").?);
    try std.testing.expectEqualStrings("roci-xlog-rank0-col.txt", xlogFileName(&b, 0, "the column pair's ring").?);
    try std.testing.expectEqualStrings("roci-xlog-rank3-row.txt", xlogFileName(&b, 3, "the row pair's ring").?);
    try std.testing.expectEqualStrings("roci-xlog-rank1-other.txt", xlogFileName(&b, 1, "a ring nobody named").?);
}
