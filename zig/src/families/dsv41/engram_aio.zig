//! Decode rounds' Engram reads without a thread hand-off (the served lane's engram_io.cpp aio_init / aio_start /
//! aio_wait / aio_touch): Linux AIO on the tables' O_DIRECT descriptors. Each read takes its row's 4 KiB-aligned span
//! (one or two blocks) into a bounce slot of a ring; a batch's rows are copied out to their places as its reads
//! complete, reaped by the thread that waits (polling, no sleep): on GB10 a woken reader thread costs ~0.6 ms after an
//! idle round, an AIO submit and reap by the round's own thread ~0.4-0.8 ms for a 6-row window. A touch (a round's
//! first rows read only to wake the drive) goes through a thread of its own and copies nothing. The bytes are the
//! preads' (engram_io.Pool.gather's).
const std = @import("std");
const linux = std.os.linux;
const engram_io = @import("engram_io.zig");

pub const block = 4096;
/// A read's slot: a row's span is at most two blocks (rows are far smaller than a block).
pub const slot = 2 * block;

const aio_context_t = usize;
const cmd_pread: u16 = 0; // IOCB_CMD_PREAD

/// struct iocb (linux/aio_abi.h, little-endian).
const Iocb = extern struct {
    data: u64 = 0,
    key: u32 = 0,
    rw_flags: u32 = 0,
    lio_opcode: u16 = 0,
    reqprio: i16 = 0,
    fildes: u32 = 0,
    buf: u64 = 0,
    nbytes: u64 = 0,
    offset: i64 = 0,
    reserved2: u64 = 0,
    flags: u32 = 0,
    resfd: u32 = 0,
};

/// struct io_event.
const IoEvent = extern struct { data: u64, obj: u64, res: i64, res2: i64 };

fn ioSetup(n: u32, ctx: *aio_context_t) usize {
    return linux.syscall2(.io_setup, n, @intFromPtr(ctx));
}

fn ioDestroy(ctx: aio_context_t) usize {
    return linux.syscall1(.io_destroy, ctx);
}

fn ioSubmit(ctx: aio_context_t, n: usize, cbs: [*]*Iocb) isize {
    return @bitCast(linux.syscall3(.io_submit, ctx, n, @intFromPtr(cbs)));
}

fn ioGetevents(ctx: aio_context_t, min: usize, n: usize, ev: [*]IoEvent, ts: ?*linux.timespec) isize {
    return @bitCast(linux.syscall5(.io_getevents, ctx, min, n, @intFromPtr(ev), @intFromPtr(ts)));
}

/// A batch's reads are ring slots [first, first + n) (mod slots), and the per-read facts (where the row goes, the block skew, the row's bytes) live in
/// the Aio's arrays at the same slot: a slot belongs to one live batch at a time (the ring-free wait in the submit guarantees it), so the arrays are
/// made once with the ring and no batch allocates.
const Batch = struct {
    first: u64, // reads [first, first + n) of the ring
    n: usize,
    left: usize, // still in flight
    failed: bool = false,
    touch: bool = false,
};

fn aioNowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub const Aio = struct {
    // what the ring's users spent (a pair line reads the differences): time in io_submit, time in the wait for completions (the
    // reaping and the copy of each row out of the ring are done in that wait, and in the submit's wait for a free ring slot), reads
    // in flight now and the most since it was last read
    st_submit_ns: std.atomic.Value(u64) = .init(0),
    st_wait_ns: std.atomic.Value(u64) = .init(0),
    // TF_DS_ENGRAM_AIO_SLEEP: a waiter sleeps in the kernel (io_getevents asking for `wake_min` completions or `wake_ns`) instead of polling
    sleep: bool = false,
    // TF_DS_ENGRAM_AIO_DEPTH: a cap on reads in flight (0: none), submitted `depth_slice` at a time; the drive's throughput falls when it is over-queued
    depth_cap: usize = 0,
    depth_slice: usize = 1,
    wake_min: usize = 32,
    spin: usize = 0, // empty polls a sleeping waiter makes before it sleeps (after a poll that found completions it polls again at once)
    wake_ns: i64 = 1_000_000,
    st_wakes: std.atomic.Value(u64) = .init(0), // blocking io_getevents calls and the completions they brought
    st_wake_events: std.atomic.Value(u64) = .init(0),
    st_calls: std.atomic.Value(u64) = .init(0), // io_submit calls, the reads they took, and the longest call
    st_reads: std.atomic.Value(u64) = .init(0),
    st_call_max_ns: std.atomic.Value(u64) = .init(0),
    st_inflight: std.atomic.Value(u32) = .init(0),
    st_inflight_max: std.atomic.Value(u32) = .init(0),

    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: aio_context_t = 0,
    ring: []align(block) u8,
    slots: usize,
    // per ring slot, made once in init: where the slot's row goes (null: a touch, nothing copied), the block skew, the row's bytes; and the
    // submit's scratch (the iocbs and their pointers), used under the mutex
    dst_s: []?[*]u8 = &.{},
    skew_s: []usize = &.{},
    bytes_s: []usize = &.{},
    cb_s: []Iocb = &.{},
    ptr_s: []*Iocb = &.{},
    head: u64 = 0, // reads issued so far (slot = index % slots)
    next_id: u64 = 1,
    batches: std.AutoHashMapUnmanaged(u64, Batch) = .empty,
    live: std.ArrayList([2]u64) = .empty, // (batch id, first read) in issue order
    mutex: std.Io.Mutex = .init,

    /// The AIO context and a ring of `slots` bounce slots (8 KiB each); error.AioUnavailable when the kernel refuses.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, slots: usize) !*Aio {
        const a = try gpa.create(Aio);
        errdefer gpa.destroy(a);
        a.* = .{ .gpa = gpa, .io = io, .ring = try gpa.alignedAlloc(u8, .fromByteUnits(block), slots * slot), .slots = slots };
        errdefer gpa.free(a.ring);
        a.dst_s = try gpa.alloc(?[*]u8, slots);
        errdefer gpa.free(a.dst_s);
        a.skew_s = try gpa.alloc(usize, slots);
        errdefer gpa.free(a.skew_s);
        a.bytes_s = try gpa.alloc(usize, slots);
        errdefer gpa.free(a.bytes_s);
        a.cb_s = try gpa.alloc(Iocb, slots);
        errdefer gpa.free(a.cb_s);
        a.ptr_s = try gpa.alloc(*Iocb, slots);
        errdefer gpa.free(a.ptr_s);
        if (linux.errno(ioSetup(@intCast(@min(slots, 65536)), &a.ctx)) != .SUCCESS) return error.AioUnavailable;
        return a;
    }

    pub fn deinit(a: *Aio) void {
        // every read in flight lands before the ring goes
        a.mutex.lockUncancelable(a.io);
        while (a.inFlight()) _ = a.reap();
        a.mutex.unlock(a.io);
        _ = ioDestroy(a.ctx);
        a.batches.deinit(a.gpa);
        a.live.deinit(a.gpa);
        a.gpa.free(a.ptr_s);
        a.gpa.free(a.cb_s);
        a.gpa.free(a.bytes_s);
        a.gpa.free(a.skew_s);
        a.gpa.free(a.dst_s);
        a.gpa.free(a.ring);
        a.gpa.destroy(a);
    }

    /// io_submit of `n` reads, timed: its wall time, the reads it took and the reads then in flight.
    fn submitTimed(a: *Aio, n: usize, cbs: [*]*Iocb) isize {
        const t0 = aioNowNs();
        const r = ioSubmit(a.ctx, n, cbs);
        const dt = aioNowNs() - t0;
        _ = a.st_submit_ns.fetchAdd(dt, .monotonic);
        _ = a.st_call_max_ns.fetchMax(dt, .monotonic);
        _ = a.st_calls.fetchAdd(1, .monotonic);
        if (r > 0) {
            _ = a.st_reads.fetchAdd(@intCast(r), .monotonic);
            const now_in = a.st_inflight.fetchAdd(@intCast(r), .monotonic) + @as(u32, @intCast(r));
            _ = a.st_inflight_max.fetchMax(now_in, .monotonic);
        }
        return r;
    }

    fn inFlight(a: *Aio) bool {
        var it = a.batches.valueIterator();
        while (it.next()) |b| if (b.left > 0) return true;
        return false;
    }

    /// Completions already taken from the kernel: rows copied out, batches counted down. Holds `mutex`.
    fn handle(a: *Aio, evs: []const IoEvent) void {
        for (evs) |e| {
            const id = e.data >> 20;
            const j: usize = @intCast(e.data & 0xFFFFF);
            const b = a.batches.getPtr(id) orelse continue;
            const sl: usize = @intCast((b.first + j) % a.slots);
            if (e.res < @as(i64, @intCast(a.skew_s[sl] + a.bytes_s[sl]))) {
                b.failed = true;
            } else if (a.dst_s[sl]) |d| {
                const at = sl * slot + a.skew_s[sl];
                @memcpy(d[0..a.bytes_s[sl]], a.ring[at..][0..a.bytes_s[sl]]);
            }
            b.left -= 1;
            _ = a.st_inflight.fetchSub(1, .monotonic);
        }
    }

    /// Whatever has completed (no wait): rows copied out, batches counted down. Holds `mutex`.
    fn reap(a: *Aio) usize {
        var ev: [256]IoEvent = undefined;
        var zero: linux.timespec = .{ .sec = 0, .nsec = 0 };
        var got: usize = 0;
        while (true) {
            const r = ioGetevents(a.ctx, 0, ev.len, &ev, &zero);
            if (r <= 0) return got;
            a.handle(ev[0..@intCast(r)]);
            got += @intCast(r);
        }
    }

    /// Sleeps in the kernel until `want` completions are in or `wake_ns` has passed, and takes them (they are handled by whoever holds
    /// `mutex` next): the batched wake-up, a wake per `want` completions and not one per read. Needs no lock.
    fn sleepFor(a: *Aio, want: usize, ev: []IoEvent) usize {
        var ts: linux.timespec = .{ .sec = 0, .nsec = a.wake_ns };
        const r = ioGetevents(a.ctx, @max(@as(usize, 1), want), ev.len, ev.ptr, &ts);
        _ = a.st_wakes.fetchAdd(1, .monotonic);
        if (r <= 0) return 0;
        _ = a.st_wake_events.fetchAdd(@intCast(r), .monotonic);
        return @intCast(r);
    }

    /// The ring-full wait of a submit (`mutex` held): polls, or sleeps for a completion and handles it.
    fn waitSlot(a: *Aio) void {
        if (!a.sleep) {
            _ = a.reap();
            return;
        }
        var ev: [256]IoEvent = undefined;
        const n = a.sleepFor(1, &ev);
        a.handle(ev[0..n]);
        _ = a.reap();
    }

    /// TF_DS_ENGRAM_AIO_DEPTH: at most `cap` reads in flight (0: no cap), submitted in slices of `slice` (at most a quarter of the cap).
    pub fn configureDepth(a: *Aio, cap: usize, slice: usize) void {
        a.depth_cap = cap;
        a.depth_slice = if (cap == 0) 1 else @max(@as(usize, 1), @min(slice, @max(@as(usize, 1), cap / 4)));
    }

    /// The ring's way of waiting for completions: TF_DS_ENGRAM_AIO_SLEEP's settings (the decode rounds' ring keeps polling: a woken
    /// thread costs it more than a poll).
    pub fn configureSleep(a: *Aio, on: bool, min_events: usize, wake_us: u32, spin: usize) void {
        a.sleep = on;
        a.spin = spin;
        a.wake_min = @max(@as(usize, 1), min_events);
        a.wake_ns = @as(i64, wake_us) * 1000;
    }

    /// The submission (`mutex` held): rows `ids` of table `t`, a weight then a scale read each, copied into out_w /
    /// out_s as they complete when given (null: a touch). Returns the batch id.
    fn submitLocked(a: *Aio, t: engram_io.Table, ids: []const i64, out_w: ?[]u8, out_s: ?[]u8) !u64 {
        const reads = 2 * ids.len;
        if (reads > a.slots or reads >= 1 << 20) return error.TooManyRows;
        if (t.row_w + block > slot or t.row_s + block > slot) return error.RowTooWide;
        for (ids) |x| if (x < 0 or @as(u64, @intCast(x)) >= t.rows) return error.RowOutOfRange; // before the batch allocates
        const fd_w = if (t.fd_wd >= 0) t.fd_wd else t.fd_w;
        const fd_s = if (t.fd_sd >= 0) t.fd_sd else t.fd_s;
        // the ring's slots [head, head + reads) must be free: reap until no live batch of the last lap holds them
        while (true) {
            var busy = false;
            for (a.live.items) |l| {
                const b = a.batches.get(l[0]) orelse continue;
                const lap_lo = a.head -| a.slots;
                const lap_hi = (a.head + reads) -| a.slots;
                if (b.left > 0 and l[1] < lap_hi and l[1] + b.n > lap_lo) {
                    busy = true;
                    break;
                }
            }
            if (!busy) break;
            const tw = aioNowNs();
            a.waitSlot();
            _ = a.st_wait_ns.fetchAdd(aioNowNs() - tw, .monotonic);
        }
        const id = a.next_id;
        a.next_id += 1;
        const b: Batch = .{
            .first = a.head,
            .n = reads,
            .left = reads,
            .touch = out_w == null,
        };
        const cbs = a.cb_s[0..reads];
        const ptrs = a.ptr_s[0..reads];
        for (0..reads) |j| {
            const i = j >> 1;
            const sc = j & 1 == 1;
            const rb = if (sc) t.row_s else t.row_w;
            const row: u64 = @intCast(ids[i]); // checked above
            const off = (if (sc) t.base_s else t.base_w) + row * rb;
            const lo = std.mem.alignBackward(u64, off, block);
            const sl: usize = @intCast((a.head + j) % a.slots);
            a.skew_s[sl] = @intCast(off - lo);
            a.bytes_s[sl] = rb;
            a.dst_s[sl] = if (out_w) |w| (if (sc) out_s.?[i * t.row_s ..].ptr else w[i * t.row_w ..].ptr) else null;
            cbs[j] = .{
                .data = (id << 20) | j,
                .lio_opcode = cmd_pread,
                .fildes = @intCast(if (sc) fd_s else fd_w),
                .buf = @intFromPtr(a.ring[((a.head + j) % a.slots) * slot ..].ptr),
                .nbytes = std.mem.alignForward(u64, off + rb - lo, block),
                .offset = @intCast(lo),
            };
            ptrs[j] = &cbs[j];
        }
        a.head += reads;
        try a.batches.put(a.gpa, id, b);
        try a.live.append(a.gpa, .{ id, b.first });
        var sent: usize = 0;
        while (sent < reads) {
            // TF_DS_ENGRAM_AIO_DEPTH: at most `depth_cap` reads in flight, submitted a slice at a time as completions make room
            var n = reads - sent;
            if (a.depth_cap != 0) {
                n = @min(n, a.depth_slice);
                while (a.st_inflight.load(.monotonic) + n > a.depth_cap) {
                    const tw = aioNowNs();
                    a.waitSlot();
                    _ = a.st_wait_ns.fetchAdd(aioNowNs() - tw, .monotonic);
                }
            }
            const r = a.submitTimed(n, ptrs[sent..].ptr);
            if (r <= 0) {
                const bp = a.batches.getPtr(id).?;
                bp.left -= reads - sent; // never submitted
                bp.failed = true;
                break;
            }
            sent += @intCast(r);
        }
        // finished touches leave the bookkeeping (a waited-for batch leaves in wait)
        var k: usize = 0;
        while (k < a.live.items.len) {
            const lid = a.live.items[k][0];
            const lb = a.batches.get(lid);
            if (lb == null or (lb.?.touch and lb.?.left == 0)) {
                _ = a.batches.fetchRemove(lid);
                _ = a.live.orderedRemove(k);
            } else k += 1;
        }
        return id;
    }

    /// The submission of a packed shard's records (`mutex` held): one read a row, ids[i]'s record copied into dsts[i]
    /// as it completes. Returns the batch id.
    fn submitRecordsLocked(a: *Aio, pk: engram_io.Packed, ids: []const i64, dsts: []const [*]u8) !u64 {
        const reads = ids.len;
        if (reads > a.slots or reads >= 1 << 20) return error.TooManyRows;
        if (pk.rec + block > slot) return error.RowTooWide;
        for (ids) |x| if (x < 0 or @as(u64, @intCast(x)) < pk.lo or @as(u64, @intCast(x)) >= pk.hi) return error.RowOutOfRange;
        const fd = if (pk.fd_d >= 0) pk.fd_d else pk.fd;
        // the ring's slots [head, head + reads) must be free: reap until no live batch of the last lap holds them
        while (true) {
            var busy = false;
            for (a.live.items) |l| {
                const b = a.batches.get(l[0]) orelse continue;
                const lap_lo = a.head -| a.slots;
                const lap_hi = (a.head + reads) -| a.slots;
                if (b.left > 0 and l[1] < lap_hi and l[1] + b.n > lap_lo) {
                    busy = true;
                    break;
                }
            }
            if (!busy) break;
            const tw = aioNowNs();
            a.waitSlot();
            _ = a.st_wait_ns.fetchAdd(aioNowNs() - tw, .monotonic);
        }
        const id = a.next_id;
        a.next_id += 1;
        const b: Batch = .{
            .first = a.head,
            .n = reads,
            .left = reads,
        };
        const cbs = a.cb_s[0..reads];
        const ptrs = a.ptr_s[0..reads];
        for (0..reads) |j| {
            const row: u64 = @intCast(ids[j]);
            const off = pk.base + (row - pk.lo) * pk.rec;
            const lo = std.mem.alignBackward(u64, off, block);
            const sl: usize = @intCast((a.head + j) % a.slots);
            a.skew_s[sl] = @intCast(off - lo);
            a.bytes_s[sl] = pk.rec;
            a.dst_s[sl] = dsts[j];
            cbs[j] = .{
                .data = (id << 20) | j,
                .lio_opcode = cmd_pread,
                .fildes = @intCast(fd),
                .buf = @intFromPtr(a.ring[((a.head + j) % a.slots) * slot ..].ptr),
                .nbytes = std.mem.alignForward(u64, off + pk.rec - lo, block),
                .offset = @intCast(lo),
            };
            ptrs[j] = &cbs[j];
        }
        a.head += reads;
        try a.batches.put(a.gpa, id, b);
        try a.live.append(a.gpa, .{ id, b.first });
        var sent: usize = 0;
        while (sent < reads) {
            // TF_DS_ENGRAM_AIO_DEPTH: at most `depth_cap` reads in flight, submitted a slice at a time as completions make room
            var n = reads - sent;
            if (a.depth_cap != 0) {
                n = @min(n, a.depth_slice);
                while (a.st_inflight.load(.monotonic) + n > a.depth_cap) {
                    const tw = aioNowNs();
                    a.waitSlot();
                    _ = a.st_wait_ns.fetchAdd(aioNowNs() - tw, .monotonic);
                }
            }
            const r = a.submitTimed(n, ptrs[sent..].ptr);
            if (r <= 0) {
                const bp = a.batches.getPtr(id).?;
                bp.left -= reads - sent; // never submitted
                bp.failed = true;
                break;
            }
            sent += @intCast(r);
        }
        return id;
    }

    /// Submit the records of `ids` from packed shard `pk`, ids[i]'s landing in dsts[i] (pk.rec bytes) as it completes
    /// (keep them alive until `wait` returns). Returns the batch id; an empty `ids` returns 0, which `wait` passes.
    pub fn startRecords(a: *Aio, pk: engram_io.Packed, ids: []const i64, dsts: []const [*]u8) !u64 {
        if (dsts.len != ids.len) return error.BadSize;
        if (ids.len == 0) return 0;
        a.mutex.lockUncancelable(a.io);
        defer a.mutex.unlock(a.io);
        return a.submitRecordsLocked(pk, ids, dsts);
    }

    /// Submit rows `ids` of table `t` (weight and scale rows), landing in out_w ([ids.len, row_w]) and out_s as they
    /// complete (keep both alive until `wait` returns). Returns the batch id.
    pub fn start(a: *Aio, t: engram_io.Table, ids: []const i64, out_w: []u8, out_s: []u8) !u64 {
        if (out_w.len < ids.len * t.row_w or out_s.len < ids.len * t.row_s) return error.BadSize;
        a.mutex.lockUncancelable(a.io);
        defer a.mutex.unlock(a.io);
        return a.submitLocked(t, ids, out_w, out_s);
    }

    /// A touch of rows `ids`: their reads submitted, nothing copied, nobody waits (the lane hands these to a thread of
    /// their own so a slow submit never holds the round's thread; callers that want that run it on one).
    pub fn touch(a: *Aio, t: engram_io.Table, ids: []const i64) !void {
        a.mutex.lockUncancelable(a.io);
        defer a.mutex.unlock(a.io);
        _ = try a.submitLocked(t, ids, null, null);
    }

    /// Reap (polling) until batch `id`'s rows are all in; error.ReadFailed when one failed. A batch is waited for once.
    pub fn wait(a: *Aio, id: u64) !void {
        const t0 = aioNowNs();
        defer _ = a.st_wait_ns.fetchAdd(aioNowNs() - t0, .monotonic);
        var ev: [256]IoEvent = undefined;
        var taken: usize = 0; // completions the last sleep brought, to be handled under the lock
        var empty: usize = 0;
        while (true) {
            a.mutex.lockUncancelable(a.io);
            if (taken > 0) {
                a.handle(ev[0..taken]);
                taken = 0;
            }
            const b = a.batches.getPtr(id) orelse {
                a.mutex.unlock(a.io);
                return;
            };
            const got = a.reap();
            if (b.left == 0) {
                const failed = b.failed;
                _ = a.batches.fetchRemove(id);
                for (a.live.items, 0..) |l, k| if (l[0] == id) {
                    _ = a.live.orderedRemove(k);
                    break;
                };
                a.mutex.unlock(a.io);
                if (failed) return error.ReadFailed;
                return;
            }
            const want = @min(b.left, a.wake_min);
            a.mutex.unlock(a.io);
            // polling: round again; sleeping: wait here, off the lock, for a batch of completions
            if (a.sleep) {
                // a poll that found completions is followed by another at once; only polls that find none count towards the sleep
                if (got > 0 and a.spin > 0) {
                    empty = 0;
                } else {
                    empty += 1;
                    if (empty > a.spin) {
                        taken = a.sleepFor(want, &ev);
                        empty = 0;
                    }
                }
            }
        }
    }
};

fn aioRowsRun(sleep: bool, min_events: usize, wake_us: u32, depth: usize, spin: usize) !u32 {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a [rows, 100] weight table and its [rows, 3] scales, rows straddling block boundaries
    const rows = 300;
    var json_buf: [256]u8 = undefined;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"U8\",\"shape\":[{d},100],\"data_offsets\":[0,{d}]}},\"layers.1.engram.embed.scale\":{{\"dtype\":\"U8\",\"shape\":[{d},3],\"data_offsets\":[{d},{d}]}}}}", .{ rows, rows * 100, rows, rows * 100, rows * 103 });
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    for (0..rows * 103) |i| try file.append(a, @intCast((i * 7 + 3) % 251));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var t = try engram_io.Tables.open(a, io, dir);
    defer t.close();
    const tab = t.layers.get(1).?;
    const pool = try engram_io.Pool.init(a, io, 2);
    defer pool.deinit(a);
    const aio = Aio.init(a, io, 16) catch |e| switch (e) {
        error.AioUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer aio.deinit();
    aio.configureSleep(sleep, min_events, wake_us, spin);
    aio.configureDepth(depth, 2);
    var prng = std.Random.DefaultPrng.init(7);
    for (0..12) |round| {
        var ids: [6]i64 = undefined;
        for (&ids) |*x| x.* = prng.random().intRangeLessThan(i64, 0, rows);
        var want_w: [6 * 100]u8 = undefined;
        var want_s: [6 * 3]u8 = undefined;
        try pool.gather(tab, &ids, &want_w, &want_s);
        var got_w: [6 * 100]u8 = @splat(0);
        var got_s: [6 * 3]u8 = @splat(0);
        if (round % 3 == 0) try aio.touch(tab, ids[0..2]);
        const id = try aio.start(tab, &ids, &got_w, &got_s);
        try aio.wait(id);
        try std.testing.expectEqualSlices(u8, &want_w, &got_w);
        try std.testing.expectEqualSlices(u8, &want_s, &got_s);
    }
    // the ring's counters: every submission counted with its reads, the longest call timed, reads in flight seen
    try std.testing.expect(aio.st_calls.load(.monotonic) >= 12);
    try std.testing.expect(aio.st_reads.load(.monotonic) >= 12 * 6);
    try std.testing.expect(aio.st_reads.load(.monotonic) >= aio.st_calls.load(.monotonic));
    try std.testing.expect(aio.st_call_max_ns.load(.monotonic) > 0);
    try std.testing.expect(aio.st_inflight_max.load(.monotonic) >= 1);
    try std.testing.expect(aio.st_wait_ns.load(.monotonic) > 0);
    // sleeping waiters sleep (and take completions on the way); polling ones never call the blocking wait
    if (sleep) try std.testing.expect(aio.st_wakes.load(.monotonic) > 0) else try std.testing.expectEqual(@as(u64, 0), aio.st_wakes.load(.monotonic));
    var big_w: [9 * 100]u8 = undefined;
    var big_s: [9 * 3]u8 = undefined;
    const nine: [9]i64 = @splat(0);
    try std.testing.expectError(error.TooManyRows, aio.start(tab, &nine, &big_w, &big_s));
    return aio.st_inflight_max.load(.monotonic);
}

test "AIO rows equal the pool's, across a ring that wraps" {
    _ = try aioRowsRun(false, 4, 500, 0, 0);
}

test "AIO rows equal the pool's, across a ring that wraps, with the waiters sleeping for batches of completions" {
    _ = try aioRowsRun(true, 4, 500, 0, 0);
}

test "AIO rows equal the pool's with sleeping waiters asking for more completions than there are, ending on the time bound, and for one" {
    _ = try aioRowsRun(true, 1000, 100, 0, 0);
    _ = try aioRowsRun(true, 1, 50, 0, 0);
}

test "AIO rows equal the pool's with sleeping waiters that poll on after a poll that found completions and sleep after a few that found none" {
    _ = try aioRowsRun(true, 4, 500, 0, 3);
    _ = try aioRowsRun(true, 1000, 100, 4, 1);
}

test "AIO rows equal the pool's with a cap on reads in flight, polling and sleeping: the cap holds, and without it the batch goes in whole" {
    const open_max = try aioRowsRun(false, 4, 500, 0, 0);
    try std.testing.expect(open_max > 4); // (the control: a batch of 12 reads goes in at once, so the cap below is live)
    for ([_]bool{ false, true }) |sleep| {
        const m = try aioRowsRun(sleep, 4, 500, 4, 0);
        try std.testing.expect(m >= 1 and m <= 4);
    }
    // a cap of one: every read waits for the one before it
    try std.testing.expect((try aioRowsRun(true, 1, 50, 1, 0)) == 1);
}

test "a refused row id leaves the ring and its allocations as they were" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const rows = 40;
    var json_buf: [256]u8 = undefined;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"U8\",\"shape\":[{d},100],\"data_offsets\":[0,{d}]}},\"layers.1.engram.embed.scale\":{{\"dtype\":\"U8\",\"shape\":[{d},3],\"data_offsets\":[{d},{d}]}}}}", .{ rows, rows * 100, rows, rows * 100, rows * 103 });
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    for (0..rows * 103) |i| try file.append(a, @intCast((i * 7 + 3) % 251));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var t = try engram_io.Tables.open(a, io, dir);
    defer t.close();
    const tab = t.layers.get(1).?;
    const aio = Aio.init(a, io, 16) catch |e| switch (e) {
        error.AioUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer aio.deinit();
    var w: [2 * 100]u8 = undefined;
    var sc: [2 * 3]u8 = undefined;
    // an id past the table after a good one, and a negative id: refused before anything is allocated
    try std.testing.expectError(error.RowOutOfRange, aio.start(tab, &.{ 3, rows }, &w, &sc));
    try std.testing.expectError(error.RowOutOfRange, aio.start(tab, &.{-1}, w[0..100], sc[0..3]));
    // the ring still reads: the same bytes as the pool
    const pool = try engram_io.Pool.init(a, io, 2);
    defer pool.deinit(a);
    var want_w: [2 * 100]u8 = undefined;
    var want_s: [2 * 3]u8 = undefined;
    try pool.gather(tab, &.{ 3, rows - 1 }, &want_w, &want_s);
    const id = try aio.start(tab, &.{ 3, rows - 1 }, &w, &sc);
    try aio.wait(id);
    try std.testing.expectEqualSlices(u8, &want_w, &w);
    try std.testing.expectEqualSlices(u8, &want_s, &sc);
}

test "a batch allocates nothing once the ring is warm: the per-read arrays and the submit scratch are made with the ring, whole laps of the ring give the same bytes as the pool" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const rows = 40;
    var json_buf: [256]u8 = undefined;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"U8\",\"shape\":[{d},100],\"data_offsets\":[0,{d}]}},\"layers.1.engram.embed.scale\":{{\"dtype\":\"U8\",\"shape\":[{d},3],\"data_offsets\":[{d},{d}]}}}}", .{ rows, rows * 100, rows, rows * 100, rows * 103 });
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    for (0..rows * 103) |i| try file.append(a, @intCast((i * 7 + 3) % 251));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var t = try engram_io.Tables.open(a, io, dir);
    defer t.close();
    const tab = t.layers.get(1).?;
    // the Aio on a counting allocator: every allocation after the warm-up batches is a failure of the claim
    var fa = std.testing.FailingAllocator.init(a, .{});
    const aio = Aio.init(fa.allocator(), io, 16) catch |e| switch (e) {
        error.AioUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer aio.deinit();
    const pool = try engram_io.Pool.init(a, io, 2);
    defer pool.deinit(a);
    var prng = std.Random.DefaultPrng.init(11);
    var warm_allocs: usize = 0;
    // 40 batches of 5 rows (10 reads) wrap a 16-slot ring many times; touches and reads alternate
    for (0..40) |round| {
        var ids: [5]i64 = undefined;
        for (&ids) |*x| x.* = prng.random().intRangeLessThan(i64, 0, rows);
        var want_w: [5 * 100]u8 = undefined;
        var want_s: [5 * 3]u8 = undefined;
        try pool.gather(tab, &ids, &want_w, &want_s);
        var got_w: [5 * 100]u8 = @splat(0);
        var got_s: [5 * 3]u8 = @splat(0);
        if (round % 4 == 1) try aio.touch(tab, ids[0..3]);
        const id = try aio.start(tab, &ids, &got_w, &got_s);
        try aio.wait(id);
        try std.testing.expectEqualSlices(u8, &want_w, &got_w);
        try std.testing.expectEqualSlices(u8, &want_s, &got_s);
        if (round == 7) warm_allocs = fa.allocations;
    }
    try std.testing.expect(warm_allocs > 0);
    try std.testing.expectEqual(warm_allocs, fa.allocations);
}
