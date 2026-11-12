//! The SSD tier with a store on every rank, kept in step. Rank 0 alone decides (keep, evict, restore); each decision
//! travels to the other ranks as an ordered frame and every rank carries it out on its own disk. Nothing is believed
//! until all four ranks say so over the start-up links (star.zig):
//!   - a keep is visible to lookups only after every rank reports it written, flushed and checksummed, with the same
//!     digest of the bytes it holds;
//!   - a restore adopts the state only after every rank reports a clean read whose digest is the one the keep committed;
//!   - at start-up the four indexes are compared by digest and an entry survives only if all four hold it, equal.
//! Any other outcome (a refused write, a failed or damaged read, a digest that differs, a state that never settles)
//! drops the entry on all four ranks and the request recomputes. A collective that ends (a link closed) puts the peer
//! down: its pending operations are dropped and every later operation is refused. Nothing here stops the engine.
//!
//! Host logic only: a `Disk` stands for one rank's store (the engine binds its Side to it), a `Sender` for rank 0's
//! ordered frame stream to the others. The ranks call `reconcile` together before the first frame; rank 0 then drives a
//! `Leader` (keep, evict, restore at the tier's decision points, settle at service points while anything is in flight) and
//! every other rank feeds the frames it receives, in order, to `Peer.onFrame`. The decisions every rank reaches are the
//! same list, in the same order.
const std = @import("std");

pub const Star = @import("star.zig").Star;
pub const Key = [32]u8;
pub const Digest = [32]u8;

/// Operations in flight at once (a settle reports on all of them: 33 bytes each in a star part).
pub const max_batch = 256;
/// Settles an operation may stay pending through before every rank counts it failed.
pub const max_pending_settles = 4096;
/// Index entries one reconcile page carries.
pub const page_entries = 128;

/// A rank's report on one operation. A larger value is the worse outcome.
pub const Status = enum(u8) { ok = 0, pending = 1, fail = 2 };

pub const Report = struct { status: Status, digest: Digest = @splat(0) };

/// One entry of a rank's store as the start-up scan names it: its key and the digest of the bytes it holds (all zero
/// when the entry is damaged: it then differs from every other rank's).
pub const Listed = struct { key: Key, digest: Digest };

/// One rank's store, as the protocol drives it. Calls return at once; a write or a read that takes time reports
/// `pending` until it ends.
pub const Disk = struct {
    ptr: *anyopaque,
    /// Begin writing the entry `key` from `body` (the keep's content, opaque here). False: refused now.
    keep: *const fn (ptr: *anyopaque, key: Key, body: []const u8) bool,
    keepReport: *const fn (ptr: *anyopaque, key: Key) Report,
    /// Begin reading entry `key` into the destination of restore `id` (`body`: rank 0's description of the destination,
    /// opaque here). False: cannot (absent, no destination).
    read: *const fn (ptr: *anyopaque, key: Key, id: u64, body: []const u8) bool,
    readReport: *const fn (ptr: *anyopaque, id: u64) Report,
    /// The decision on restore `id`: keep the destination (adopt) or clear and give it back.
    finishRead: *const fn (ptr: *anyopaque, id: u64, adopt: bool) void,
    /// The entry removed; a write of it still running is cancelled. Removing what is absent is fine.
    remove: *const fn (ptr: *anyopaque, key: Key) void,
    /// Every entry the store holds. An error: this rank's store cannot be read.
    list: *const fn (ptr: *anyopaque, gpa: std.mem.Allocator, out: *std.ArrayList(Listed)) anyerror!void,
    /// Damage the committed entry `key` on disk (a byte of each of its blocks) so a later read fails its check. Only the fault
    /// drill's corrupt fault calls it; a Disk that cannot leaves it null.
    damage: ?*const fn (ptr: *anyopaque, key: Key) void = null,
};

/// TF_DS_SSD_FAULT: the gate drill's faults, each on one named rank and consumed by its first use. `write:R`: rank R's next
/// keep fails to write (its report is a fail); `corrupt:S`: after the next keep every rank committed, rank S damages its own
/// copy's blocks so the restore's read fails the check. Either way the protocol must drop the entry on all four ranks and
/// the request recompute. Every rank parses the same text and acts only when it is the named one.
pub const Fault = struct {
    write: ?u32 = null,
    corrupt: ?u32 = null,

    pub const Error = error{BadFault};

    /// `asked` as the environment gave it (null or empty: no fault): `write:R`, `corrupt:S` or both joined by a comma, each at most
    /// once, each rank below `world`. Anything else is BadFault (the start is refused: a drill that does not do what it says
    /// would pass for a protocol that works).
    pub fn parse(asked: ?[]const u8, world: u32) Error!Fault {
        const t = asked orelse return .{};
        if (t.len == 0) return .{};
        var f: Fault = .{};
        var it = std.mem.splitScalar(u8, t, ',');
        while (it.next()) |part| {
            const colon = std.mem.indexOfScalar(u8, part, ':') orelse return error.BadFault;
            const rank = std.fmt.parseInt(u32, part[colon + 1 ..], 10) catch return error.BadFault;
            if (rank >= world) return error.BadFault;
            const kind = part[0..colon];
            if (std.mem.eql(u8, kind, "write")) {
                if (f.write != null) return error.BadFault;
                f.write = rank;
            } else if (std.mem.eql(u8, kind, "corrupt")) {
                if (f.corrupt != null) return error.BadFault;
                f.corrupt = rank;
            } else return error.BadFault;
        }
        return f;
    }

    /// What `rank` does of it.
    pub fn armed(f: Fault, rank: u32) Armed {
        return .{ .write = f.write == rank, .corrupt = f.corrupt == rank };
    }

    pub fn any(f: Fault) bool {
        return f.write != null or f.corrupt != null;
    }

    /// The start line's words: "off", or the faults as they were asked.
    pub fn text(f: Fault, buf: []u8) []const u8 {
        if (!f.any()) return "off";
        var w = std.Io.Writer.fixed(buf);
        if (f.write) |r| w.print("write:{d}", .{r}) catch {};
        if (f.corrupt) |r| w.print("{s}corrupt:{d}", .{ if (f.write != null) "," else "", r }) catch {};
        return w.buffered();
    }
};

/// The faults this rank still has to carry out.
pub const Armed = struct { write: bool = false, corrupt: bool = false };

/// A rank with no store: every call fails, nothing is listed (the disk of a rank that abstains or has no store to offer).
pub fn nullDisk() Disk {
    const N = struct {
        var dummy: u8 = 0;
        fn keep(_: *anyopaque, _: Key, _: []const u8) bool {
            return false;
        }
        fn report(_: *anyopaque, _: Key) Report {
            return .{ .status = .fail };
        }
        fn read(_: *anyopaque, _: Key, _: u64, _: []const u8) bool {
            return false;
        }
        fn readReport(_: *anyopaque, _: u64) Report {
            return .{ .status = .fail };
        }
        fn finish(_: *anyopaque, _: u64, _: bool) void {}
        fn remove(_: *anyopaque, _: Key) void {}
        fn list(_: *anyopaque, _: std.mem.Allocator, _: *std.ArrayList(Listed)) anyerror!void {
            return error.NoStore;
        }
    };
    return .{ .ptr = &N.dummy, .keep = N.keep, .keepReport = N.report, .read = N.read, .readReport = N.readReport, .finishRead = N.finish, .remove = N.remove, .list = N.list };
}

// -- frames -------------------------------------------------------------------------------------------------------

pub const Op = enum(u8) { keep = 1, evict = 2, restore = 3, settle = 4, end = 5 };

/// One decision of rank 0, in the order made. `body` is the keep's content for the disk; `id` names a restore.
pub const Frame = struct {
    op: Op,
    key: Key = @splat(0),
    id: u64 = 0,
    body: []const u8 = &.{},

    const head = 1 + 32 + 8 + 4;

    pub fn encode(f: Frame, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        out.clearRetainingCapacity();
        try out.ensureTotalCapacity(gpa, head + f.body.len);
        out.appendAssumeCapacity(@backingInt(f.op));
        out.appendSliceAssumeCapacity(&f.key);
        out.appendSliceAssumeCapacity(&std.mem.toBytes(std.mem.nativeToLittle(u64, f.id)));
        out.appendSliceAssumeCapacity(&std.mem.toBytes(std.mem.nativeToLittle(u32, @intCast(f.body.len))));
        out.appendSliceAssumeCapacity(f.body);
    }

    /// The frame in `bytes` (its body borrowed from them). Anything short, long or unnamed is BadFrame.
    pub fn decode(bytes: []const u8) error{BadFrame}!Frame {
        if (bytes.len < head) return error.BadFrame;
        const op = std.enums.fromInt(Op, bytes[0]) orelse return error.BadFrame;
        const n = std.mem.readInt(u32, bytes[41..45], .little);
        if (bytes.len != head + n) return error.BadFrame;
        return .{ .op = op, .key = bytes[1..33].*, .id = std.mem.readInt(u64, bytes[33..41], .little), .body = bytes[head..] };
    }
};

/// Rank 0's ordered stream to the other ranks (the engine's command stream; a test's sockets).
pub const Sender = struct {
    ptr: *anyopaque,
    send: *const fn (ptr: *anyopaque, frame: []const u8) anyerror!void,
};

// -- what a settle decided ----------------------------------------------------------------------------------------

pub const Decided = struct {
    pub const Kind = enum {
        /// Every rank committed the keep: the entry is visible.
        kept,
        /// The keep failed on some rank, or differed, or never settled: gone from all four.
        keep_dropped,
        /// Every rank read the entry clean: adopt the restored state.
        adopt,
        /// A rank could not read it clean: the entry gone from all four, the request recomputes.
        restore_dropped,
    };
    kind: Kind,
    key: Key,
    id: u64 = 0,
};

const Item = struct {
    kind: enum { keep, restore },
    key: Key,
    id: u64,
    /// This rank's disk refused to begin it.
    refused: bool,
    /// A restore's committed digest every rank's read must equal.
    expect: Digest = @splat(0),
    settles: u32 = 0,
};

/// The first 8 bytes of a key in hex: how a log line names an entry.
const Short = struct {
    buf: [16]u8,
    fn slice(self: *const Short) []const u8 {
        return &self.buf;
    }
};

fn hexShort(key: *const Key) Short {
    var s: Short = undefined;
    _ = std.fmt.bufPrint(&s.buf, "{x}", .{key[0..8]}) catch unreachable;
    return s;
}

/// The ranks whose bits are set, as "0,2".
fn ranksOf(mask: u32, buf: []u8) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    var first = true;
    for (0..4) |r| if (mask & (@as(u32, 1) << @intCast(r)) != 0) {
        w.print("{s}{d}", .{ if (first) "" else ",", r }) catch {};
        first = false;
    };
    return w.buffered();
}

// -- one rank ------------------------------------------------------------------------------------------------------

/// One rank's side of the protocol: the same code runs on every rank, so the ledger, the visible set and every
/// decision are the same function of the same frames and the same gathered reports.
pub const Peer = struct {
    gpa: std.mem.Allocator,
    star: Star,
    rank: u32,
    disk: Disk,
    items: [max_batch]Item = undefined,
    n: usize = 0,
    /// Entries every rank committed, with the digest they agreed on. Only a settle adds to it.
    visible: std.AutoHashMapUnmanaged(Key, Digest) = .empty,
    /// A collective ended: nothing is trusted from now on.
    down: bool = false,
    /// A digest of the settings the stores must share (reconcile refuses ranks that differ).
    cfg: Digest = @splat(0),
    /// The drill's faults this rank carries out (TF_DS_SSD_FAULT); each is cleared as it fires.
    fault: Armed = .{},

    pub const Error = error{ OutOfStep, Down, BadFrame };

    pub fn init(gpa: std.mem.Allocator, star: Star, rank: u32, disk: Disk) Peer {
        return .{ .gpa = gpa, .star = star, .rank = rank, .disk = disk };
    }

    pub fn deinit(p: *Peer) void {
        p.visible.deinit(p.gpa);
    }

    pub fn isVisible(p: *const Peer, key: Key) bool {
        return !p.down and p.visible.contains(key);
    }

    pub fn pending(p: *const Peer) usize {
        return p.n;
    }

    /// Carry out one of rank 0's frames. Decisions it makes (a settle's, an evict's cancelled keeps) are appended to `out`.
    pub fn onFrame(p: *Peer, f: Frame, out: *std.ArrayList(Decided)) !void {
        switch (f.op) {
            .end => {},
            .keep => {
                if (p.n == max_batch) return error.OutOfStep;
                const faulted = p.fault.write;
                if (faulted) {
                    p.fault.write = false;
                    std.log.warn("dsv41 spill: rank {d}: the drill's write fault: this keep's write fails ({s})", .{ p.rank, hexShort(&f.key).slice() });
                }
                const ok = !p.down and !faulted and p.disk.keep(p.disk.ptr, f.key, f.body);
                p.items[p.n] = .{ .kind = .keep, .key = f.key, .id = 0, .refused = !ok };
                p.n += 1;
            },
            .evict => {
                p.disk.remove(p.disk.ptr, f.key);
                _ = p.visible.remove(f.key);
                // a keep of it still settling is cancelled: it must not become visible after the evict
                var i: usize = 0;
                while (i < p.n) {
                    if (p.items[i].kind == .keep and std.mem.eql(u8, &p.items[i].key, &f.key)) {
                        try out.append(p.gpa, .{ .kind = .keep_dropped, .key = f.key });
                        p.logDrop(.keep_dropped, f.key, "evicted while it settled");
                        p.drop(i);
                    } else i += 1;
                }
            },
            .restore => {
                if (p.n == max_batch) return error.OutOfStep;
                const expect = if (p.down) null else p.visible.get(f.key);
                const ok = expect != null and p.disk.read(p.disk.ptr, f.key, f.id, f.body);
                p.items[p.n] = .{ .kind = .restore, .key = f.key, .id = f.id, .refused = !ok, .expect = expect orelse @splat(0) };
                p.n += 1;
            },
            .settle => try p.settle(out),
        }
    }

    /// One line on every rank for an entry the protocol dropped from all four stores: what was dropped, which entry, and why.
    fn logDrop(p: *const Peer, kind: Decided.Kind, key: Key, why: []const u8) void {
        const what = if (kind == .restore_dropped) "restore" else "keep";
        std.log.warn("dsv41 spill: rank {d}: {s} dropped on all four stores: entry {s}: {s}", .{ p.rank, what, hexShort(&key).slice(), why });
    }

    fn drop(p: *Peer, i: usize) void {
        std.mem.copyForwards(Item, p.items[i .. p.n - 1], p.items[i + 1 .. p.n]);
        p.n -= 1;
    }

    fn report(p: *Peer, it: Item) Report {
        if (it.refused or p.down) return .{ .status = .fail };
        return switch (it.kind) {
            .keep => p.disk.keepReport(p.disk.ptr, it.key),
            .restore => p.disk.readReport(p.disk.ptr, it.id),
        };
    }

    /// Every rank's report on each operation in flight gathered over the links, and each one decided the same way on
    /// every rank: a fail anywhere, digests that differ, or a state that never settles drops it; any pending leaves it
    /// for the next settle; all four ok and equal commits it (a restore: all four equal to the keep's digest).
    /// A collective that ends puts this peer down (`fail`) and returns Down, its operations already in `out`.
    pub fn settle(p: *Peer, out: *std.ArrayList(Decided)) !void {
        if (p.n == 0) return;
        if (p.down) return error.Down;
        const n = p.n;
        var mine: [max_batch * 33]u8 = undefined;
        for (p.items[0..n], 0..) |it, i| {
            const r = p.report(it);
            mine[i * 33] = @backingInt(r.status);
            @memcpy(mine[i * 33 + 1 ..][0..32], &r.digest);
        }
        var table: [4 * max_batch * 33]u8 = undefined;
        const part = n * 33;
        p.star.allGather(mine[0..part], table[0 .. 4 * part]) catch {
            try p.fail(out);
            return error.Down;
        };
        var keep_going: [max_batch]bool = undefined;
        for (p.items[0..n], 0..) |*it, i| {
            var worst: Status = .ok;
            var same = true;
            var failed: u32 = 0; // the ranks (bit per rank) whose report was a fail
            const first = table[i * 33 + 1 ..][0..32];
            for (0..4) |r| {
                const cell = table[r * part + i * 33 ..][0..33];
                const s = std.enums.fromInt(Status, cell[0]) orelse Status.fail;
                worst = @fromBackingInt(@intCast(@max(@backingInt(worst), @backingInt(s))));
                if (s == .fail) failed |= @as(u32, 1) << @intCast(r);
                if (!std.mem.eql(u8, cell[1..33], first)) same = false;
                if (it.kind == .restore and !std.mem.eql(u8, cell[1..33], &it.expect)) same = false;
            }
            it.settles += 1;
            if (worst == .pending and it.settles > max_pending_settles) worst = .fail;
            keep_going[i] = worst == .pending;
            if (worst == .pending) continue;
            const good = worst == .ok and same;
            var why_buf: [96]u8 = undefined;
            var ranks_buf: [16]u8 = undefined;
            const why: []const u8 = if (good) "" else if (failed != 0)
                std.fmt.bufPrint(&why_buf, "{s} failed on rank(s) {s}", .{ if (it.kind == .keep) "the write" else "the read", ranksOf(failed, &ranks_buf) }) catch "a rank failed"
            else if (it.settles > max_pending_settles) "never settled" else "the four stores' digests differ";
            switch (it.kind) {
                .keep => if (good) {
                    try p.visible.put(p.gpa, it.key, first.*);
                    try out.append(p.gpa, .{ .kind = .kept, .key = it.key });
                    if (p.fault.corrupt) if (p.disk.damage) |dmg| {
                        p.fault.corrupt = false;
                        dmg(p.disk.ptr, it.key);
                        std.log.warn("dsv41 spill: rank {d}: the drill's corrupt fault: this rank's copy of entry {s} is damaged on disk", .{ p.rank, hexShort(&it.key).slice() });
                    };
                } else {
                    p.disk.remove(p.disk.ptr, it.key);
                    try out.append(p.gpa, .{ .kind = .keep_dropped, .key = it.key });
                    p.logDrop(.keep_dropped, it.key, why);
                },
                .restore => if (good) {
                    p.disk.finishRead(p.disk.ptr, it.id, true);
                    try out.append(p.gpa, .{ .kind = .adopt, .key = it.key, .id = it.id });
                } else {
                    p.disk.finishRead(p.disk.ptr, it.id, false);
                    p.disk.remove(p.disk.ptr, it.key);
                    _ = p.visible.remove(it.key);
                    try out.append(p.gpa, .{ .kind = .restore_dropped, .key = it.key, .id = it.id });
                    p.logDrop(.restore_dropped, it.key, why);
                },
            }
        }
        var w: usize = 0;
        for (0..n) |i| if (keep_going[i]) {
            p.items[w] = p.items[i];
            w += 1;
        };
        p.n = w;
    }

    /// This peer goes down: every operation in flight is dropped (a restore's destination given back), no entry is
    /// visible, and every later operation is refused. The disks keep what they hold; the next start reconciles them.
    pub fn fail(p: *Peer, out: *std.ArrayList(Decided)) !void {
        p.down = true;
        for (p.items[0..p.n]) |it| switch (it.kind) {
            .keep => {
                p.disk.remove(p.disk.ptr, it.key);
                try out.append(p.gpa, .{ .kind = .keep_dropped, .key = it.key });
                p.logDrop(.keep_dropped, it.key, "a link ended");
            },
            .restore => {
                p.disk.finishRead(p.disk.ptr, it.id, false);
                try out.append(p.gpa, .{ .kind = .restore_dropped, .key = it.key, .id = it.id });
                p.logDrop(.restore_dropped, it.key, "a link ended");
            },
        };
        p.n = 0;
        p.visible.clearRetainingCapacity();
    }
};

// -- rank 0 ---------------------------------------------------------------------------------------------------------

/// Rank 0's decisions: each is sent to the other ranks as a frame and then carried out here, in that order.
pub const Leader = struct {
    peer: *Peer,
    tx: Sender,
    scratch: std.ArrayList(u8) = .empty,

    pub const Keep = enum { queued, already, refused };

    pub fn deinit(l: *Leader) void {
        l.scratch.deinit(l.peer.gpa);
    }

    fn send(l: *Leader, f: Frame, out: *std.ArrayList(Decided)) bool {
        const p = l.peer;
        if (p.down) return false;
        f.encode(p.gpa, &l.scratch) catch return false;
        l.tx.send(l.tx.ptr, l.scratch.items) catch {
            p.fail(out) catch {};
            return false;
        };
        return true;
    }

    /// Keep entry `key` (content `body`): sent, written here, and invisible until a settle commits it.
    pub fn keep(l: *Leader, key: Key, body: []const u8, out: *std.ArrayList(Decided)) Keep {
        const p = l.peer;
        if (p.isVisible(key)) return .already;
        for (p.items[0..p.n]) |it| if (it.kind == .keep and std.mem.eql(u8, &it.key, &key)) return .queued;
        if (p.down or p.n == max_batch) return .refused;
        const f: Frame = .{ .op = .keep, .key = key, .body = body };
        if (!l.send(f, out)) return .refused;
        p.onFrame(f, out) catch return .refused;
        return .queued;
    }

    /// Entry `key` removed on every rank, at once.
    pub fn evict(l: *Leader, key: Key, out: *std.ArrayList(Decided)) void {
        const f: Frame = .{ .op = .evict, .key = key };
        if (!l.send(f, out)) return;
        l.peer.onFrame(f, out) catch {};
    }

    /// Begin restore `id` of `key`. False: the entry is not visible (never committed by all four, or gone) or the
    /// peer is down; the request recomputes.
    pub fn restore(l: *Leader, key: Key, id: u64, body: []const u8, out: *std.ArrayList(Decided)) bool {
        const p = l.peer;
        if (!p.isVisible(key) or p.n == max_batch) return false;
        const f: Frame = .{ .op = .restore, .key = key, .id = id, .body = body };
        if (!l.send(f, out)) return false;
        p.onFrame(f, out) catch return false;
        return true;
    }

    /// The operations in flight settled, when there are any (a frame to the ranks, then the gather).
    pub fn settle(l: *Leader, out: *std.ArrayList(Decided)) !void {
        const p = l.peer;
        if (p.n == 0) return;
        if (!l.send(.{ .op = .settle }, out)) return error.Down;
        try p.settle(out);
    }

    /// The other ranks told the stream ends (sent also when this peer is down: they are waiting for frames).
    pub fn end(l: *Leader) void {
        const f: Frame = .{ .op = .end };
        f.encode(l.peer.gpa, &l.scratch) catch return;
        l.tx.send(l.tx.ptr, l.scratch.items) catch {};
    }
};

// -- start-up -------------------------------------------------------------------------------------------------------

pub const Reconciled = struct { kept: u32 = 0, deleted: u32 = 0 };

fn keyLess(_: void, a: Listed, b: Listed) bool {
    return std.mem.lessThan(u8, &a.key, &b.key);
}

fn indexDigest(entries: []const Listed) Digest {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (entries) |e| {
        h.update(&e.key);
        h.update(&e.digest);
    }
    var d: Digest = undefined;
    h.final(&d);
    return d;
}

const Tally = struct { seen: u8 = 0, digest: Digest = undefined, equal: bool = true };

/// The four stores made one: each rank scans its own, the indexes are compared by digest over the links, and an entry
/// survives only if all four hold it with the same digest; every other entry (present on fewer ranks, or different) is
/// deleted on the rank that holds it. A rank whose store cannot be read counts as holding nothing and makes every entry
/// go. All four ranks call it together, before any frame. `peer.visible` becomes the survivors. A collective that ends
/// returns Down and deletes nothing.
pub fn reconcile(p: *Peer) !Reconciled {
    return reconcileAs(p, false);
}

/// A rank that cannot take part (its store did not open, its tier is off) calls this in place of `reconcile`: every rank's
/// call then returns Abstained, and nothing is deleted anywhere.
pub fn abstain(p: *Peer) !Reconciled {
    return reconcileAs(p, true);
}

fn reconcileAs(p: *Peer, abstaining: bool) !Reconciled {
    const gpa = p.gpa;
    var mine: std.ArrayList(Listed) = .empty;
    defer mine.deinit(gpa);
    var readable = true;
    if (!abstaining) p.disk.list(p.disk.ptr, gpa, &mine) catch {
        readable = false;
        mine.clearRetainingCapacity();
    };
    std.mem.sort(Listed, mine.items, {}, keyLess);
    // header: count, state (0 unreadable, 1 readable, 2 abstaining), the index's digest, the settings' digest
    var head: [4 + 1 + 32 + 32]u8 = undefined;
    std.mem.writeInt(u32, head[0..4], @intCast(mine.items.len), .little);
    head[4] = if (abstaining) 2 else @intFromBool(readable);
    @memcpy(head[5..37], &indexDigest(mine.items));
    @memcpy(head[37..69], &p.cfg);
    var heads: [4 * head.len]u8 = undefined;
    p.star.allGather(&head, &heads) catch return error.Down;
    var all_ok = true;
    var all_same = true;
    var most: u32 = 0;
    var any_abstain = false;
    var same_cfg = true;
    for (0..4) |r| {
        const h = heads[r * head.len ..][0..head.len];
        if (h[4] == 2) any_abstain = true;
        if (!std.mem.eql(u8, h[37..69], heads[37..69])) same_cfg = false;
    }
    if (any_abstain) return error.Abstained;
    if (!same_cfg) return error.ConfigDiffers;
    for (0..4) |r| {
        const h = heads[r * head.len ..][0..head.len];
        if (h[4] == 0) all_ok = false;
        if (!std.mem.eql(u8, h[5..37], heads[5..37])) all_same = false;
        most = @max(most, std.mem.readInt(u32, h[0..4], .little));
    }
    var res: Reconciled = .{};
    var keep_set: std.AutoHashMapUnmanaged(Key, Digest) = .empty;
    defer keep_set.deinit(gpa);
    if (all_ok and all_same) {
        for (mine.items) |e| try keep_set.put(gpa, e.key, e.digest);
    } else if (all_ok) {
        var tally: std.AutoHashMapUnmanaged(Key, Tally) = .empty;
        defer tally.deinit(gpa);
        const page_bytes = 2 + page_entries * 64;
        var round: u32 = 0;
        const rounds = (most + page_entries - 1) / page_entries;
        while (round < rounds) : (round += 1) {
            var page: [page_bytes]u8 = @splat(0);
            const from = @as(usize, round) * page_entries;
            const take = if (from < mine.items.len) @min(page_entries, mine.items.len - from) else 0;
            std.mem.writeInt(u16, page[0..2], @intCast(take), .little);
            for (mine.items[@min(from, mine.items.len)..][0..take], 0..) |e, i| {
                @memcpy(page[2 + i * 64 ..][0..32], &e.key);
                @memcpy(page[2 + i * 64 + 32 ..][0..32], &e.digest);
            }
            var pages: [4 * page_bytes]u8 = undefined;
            p.star.allGather(&page, &pages) catch return error.Down;
            for (0..4) |r| {
                const pg = pages[r * page_bytes ..][0..page_bytes];
                const cnt = std.mem.readInt(u16, pg[0..2], .little);
                if (cnt > page_entries) return error.Down;
                for (0..cnt) |i| {
                    const key: Key = pg[2 + i * 64 ..][0..32].*;
                    const dig: Digest = pg[2 + i * 64 + 32 ..][0..32].*;
                    const gop = try tally.getOrPut(gpa, key);
                    if (!gop.found_existing) gop.value_ptr.* = .{ .digest = dig };
                    gop.value_ptr.seen += 1;
                    if (!std.mem.eql(u8, &gop.value_ptr.digest, &dig)) gop.value_ptr.equal = false;
                }
            }
        }
        for (mine.items) |e| {
            const t = tally.get(e.key).?;
            if (t.seen == 4 and t.equal) try keep_set.put(gpa, e.key, e.digest);
        }
    }
    // (a rank whose store could not be read: nothing survives anywhere)
    if (!all_ok) keep_set.clearRetainingCapacity();
    p.visible.clearRetainingCapacity();
    for (mine.items) |e| {
        if (keep_set.get(e.key)) |d| {
            try p.visible.put(gpa, e.key, d);
            res.kept += 1;
        } else {
            p.disk.remove(p.disk.ptr, e.key);
            res.deleted += 1;
        }
    }
    return res;
}
