//! A follower's store, kept in step with rank 0's: the other ranks' end of spill_sync. It opens a `Side` (the same store
//! rank 0 has, for this rank's own disk) and carries out rank 0's keep, evict and restore frames on it, in order, without
//! ever waiting for the disk: writes and reads run on the store's threads and the Mirror's `service` hands their device
//! copies on between frames. Its reports (a write published, a read whole and checked) are what the four ranks vote on.
const std = @import("std");
const core = @import("core");
const tier = @import("spill_tier.zig");
const sync = @import("spill_sync.zig");
const wire = @import("spill_wire.zig");
const Star = @import("star.zig").Star;

const format = core.spill.format;

/// What the Mirror needs of its rank's kept states (lanes.zig's follower loop binds them to its `kept` map): a kept state's
/// device buffer by Snap id, a new buffer for a restore's destination and its release, and a restored state given the id
/// rank 0 chose.
pub const Host = struct {
    ctx: *anyopaque,
    snapBuf: *const fn (ctx: *anyopaque, id: u64) ?u64,
    allocSnap: *const fn (ctx: *anyopaque) anyerror!u64,
    freeSnap: *const fn (ctx: *anyopaque, buf: u64) void,
    adoptSnap: *const fn (ctx: *anyopaque, id: u64, buf: u64) anyerror!void,
};

/// A digest of what the four stores must share for their entries to mean the same: the store's name (the checkpoint, the
/// kit, the math switches) and what the tier keeps (the cap, the quota, the shortest point, the turns, the image cache).
pub fn cfgDigest(compat: sync.Digest, s: tier.Settings) sync.Digest {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("tf-spill/four-stores");
    h.update(&compat);
    inline for (.{ s.gib, s.quota }) |x| h.update(std.mem.asBytes(&@as(u64, @bitCast(x))));
    inline for (.{ s.min_tokens, s.turns, @as(u32, @intFromBool(s.image_cache)) }) |x| h.update(std.mem.asBytes(&std.mem.nativeToLittle(u32, x)));
    var d: sync.Digest = undefined;
    h.final(&d);
    return d;
}

/// Time one service pass may spend staging writes.
pub const service_budget_ns: u64 = 2 * std.time.ns_per_ms;

pub fn Mirror(comptime Dev: type) type {
    return struct {
        const Self = @This();
        pub const S = tier.Side(Dev);

        const Rest = struct { rx: *tier.Rx, snap: u64, buf: u64 };

        gpa: std.mem.Allocator,
        side: *S,
        peer: sync.Peer,
        host: Host,
        rests: std.AutoHashMapUnmanaged(u64, Rest) = .empty,
        out: std.ArrayList(sync.Decided) = .empty,

        /// `side` is moved in. `cfg`: the digest of the settings the four stores must share.
        pub fn create(gpa: std.mem.Allocator, side: *S, star: Star, rank: u32, host: Host, cfg: sync.Digest) !*Self {
            const m = try gpa.create(Self);
            m.* = .{ .gpa = gpa, .side = side, .peer = undefined, .host = host };
            m.peer = sync.Peer.init(gpa, star, rank, m.disk());
            m.peer.cfg = cfg;
            return m;
        }

        /// The store closed (bounded by the stop's flush deadline) and everything freed. False: a store thread was in a disk
        /// call; the store is left to it.
        pub fn close(self: *Self) bool {
            var it = self.rests.valueIterator();
            while (it.next()) |r| {
                self.side.rxEnd(r.rx, .cancel, false);
                self.host.freeSnap(self.host.ctx, r.buf);
            }
            self.rests.deinit(self.gpa);
            self.out.deinit(self.gpa);
            self.peer.deinit();
            const clean = self.side.close();
            self.gpa.destroy(self);
            return clean;
        }

        /// The Disk the protocol drives. The pointer is to this Mirror: it must not move once the Peer holds it.
        pub fn disk(self: *Self) sync.Disk {
            return .{ .ptr = self, .keep = keepFn, .keepReport = keepReportFn, .read = readFn, .readReport = readReportFn, .finishRead = finishFn, .remove = removeFn, .list = listFn, .damage = damageFn };
        }

        /// The ranks' start-up reconcile (every rank calls it, rank 0 too, before any frame).
        pub fn reconcile(self: *Self) !sync.Reconciled {
            return sync.reconcile(&self.peer);
        }

        fn me(p: *anyopaque) *Self {
            return @ptrCast(@alignCast(p));
        }

        // -- the frames ---------------------------------------------------------------------------------------------

        /// One snap_sync frame's bytes: a sub-op byte, then its fields.
        pub fn onSync(self: *Self, bytes: []const u8) !void {
            if (bytes.len == 0) return error.BadFrame;
            const op = std.enums.fromInt(wire.SubOp, bytes[0]) orelse return error.BadFrame;
            switch (op) {
                .proto => {
                    const f = try sync.Frame.decode(bytes[1..]);
                    self.out.clearRetainingCapacity();
                    try self.peer.onFrame(f, &self.out);
                },
                .guard => {
                    const g = try wire.Guard.decode(bytes[1..]);
                    self.side.guard(g.snap, g.from);
                },
            }
        }

        /// Whatever is in flight: writes to stage, restores to copy, operations to settle.
        pub fn busy(self: *const Self) bool {
            return self.side.busy() or self.rests.count() > 0 or self.peer.pending() > 0;
        }

        /// Between frames, never waiting: finished copies handed on, a few blocks staged, every restore's ready blocks copied.
        pub fn service(self: *Self) void {
            _ = self.side.service(service_budget_ns, tier.pump_blocks);
            var it = self.rests.valueIterator();
            while (it.next()) |r| self.step(r.rx);
        }

        fn step(self: *Self, rx: *tier.Rx) void {
            _ = self.side.retire();
            self.side.rxStep(rx, null, service_budget_ns);
            self.side.rxSmall(rx);
        }

        // the generations and kept states this rank mirrors from rank 0's own calls (lanes.zig's follower arms)
        pub fn bump(self: *Self, lo: usize, hi: usize) void {
            self.side.bump(lo, hi);
        }
        pub fn bumpSlot(self: *Self, slot: usize) void {
            self.side.bumpSlot(slot);
        }
        pub fn dropSnap(self: *Self, snap: u64) void {
            self.side.dropSnap(snap);
        }

        // -- the Disk -----------------------------------------------------------------------------------------------

        fn keepFn(p: *anyopaque, key: sync.Key, body: []const u8) bool {
            const self = me(p);
            var arena_state = std.heap.ArenaAllocator.init(self.gpa);
            defer arena_state.deinit();
            const arena = arena_state.allocator();
            const b = wire.KeepBody.decode(arena, body) catch return false;
            if (b.at == 0 or b.keyed.len < b.at) return false;
            const buf = self.host.snapBuf(self.host.ctx, b.snap) orelse return false;
            const keyed = b.keyed[0..b.at];
            var chain: std.ArrayList(format.Digest) = .empty;
            defer chain.deinit(self.gpa);
            format.chain(self.gpa, keyed, self.side.bt, &chain) catch return false;
            const media = b.media;
            const k = format.pointKeyChained(chain.items, keyed, self.side.bt, media);
            // the same ids must name the same point on every rank
            if (!std.mem.eql(u8, &k, &key)) return false;
            const lineage: format.Lineage = .{ .root = b.lineage_root, .part = b.lineage_part, .forks = b.forks };
            const r = self.side.keepPoint(.{ .keyed = keyed, .key = k, .chain = chain.items, .at = b.at, .base = @intCast(b.base), .snap = b.snap, .buf = buf, .lineage = lineage, .origin = b.origin, .shared = b.shared, .group = b.group, .raw = if (media.len > 0) b.raw else null, .media = media });
            return r != .refused;
        }

        fn keepReportFn(p: *anyopaque, key: sync.Key) sync.Report {
            const self = me(p);
            if (self.side.writing(key)) return .{ .status = .pending };
            const d = self.side.store.pointDigest(key) orelse return .{ .status = .fail };
            return .{ .status = .ok, .digest = d };
        }

        fn readFn(p: *anyopaque, key: sync.Key, id: u64, body: []const u8) bool {
            const self = me(p);
            const b = wire.RestoreBody.decode(body) catch return false;
            const side = self.side;
            if (b.at == 0 or b.at % side.bt != 0 or !side.store.presence(key)) return false;
            self.rests.ensureUnusedCapacity(self.gpa, 1) catch return false;
            const rx = side.newRx(id, key, b.at, b.at / side.bt) catch return false;
            rx.rid = side.store.beginRead(key) catch {
                side.rxDrop(rx);
                return false;
            };
            const buf = self.host.allocSnap(self.host.ctx) catch {
                side.rxEnd(rx, .cancel, false);
                return false;
            };
            rx.snap = .{ .id = b.snap, .buf = buf };
            rx.dst = side.gens.reserve(@intCast(b.base), @intCast(b.base + b.at)) catch {
                side.rxEnd(rx, .cancel, false);
                self.host.freeSnap(self.host.ctx, buf);
                return false;
            };
            self.rests.putAssumeCapacity(id, .{ .rx = rx, .snap = b.snap, .buf = buf });
            return true;
        }

        fn readReportFn(p: *anyopaque, id: u64) sync.Report {
            const self = me(p);
            const r = self.rests.get(id) orelse return .{ .status = .fail };
            self.step(r.rx);
            if (r.rx.why != .none) return .{ .status = .fail };
            if (!S.rxReady(r.rx)) return .{ .status = .pending };
            if (!self.side.rxValid(r.rx)) return .{ .status = .fail };
            const d = self.side.store.pointDigest(r.rx.key) orelse return .{ .status = .fail };
            return .{ .status = .ok, .digest = d };
        }

        fn finishFn(p: *anyopaque, id: u64, adopt: bool) void {
            const self = me(p);
            const r = (self.rests.fetchRemove(id) orelse return).value;
            const side = self.side;
            if (adopt) {
                const key = r.rx.key;
                if (side.rxCommit(r.rx)) |done| {
                    var d = done;
                    d.lineage.deinit(self.gpa);
                    side.rxDrop(r.rx);
                    // (the state is rank 0's to use or drop; this rank's marker is not kept past the adoption)
                    side.store.restored(key);
                    self.host.adoptSnap(self.host.ctx, r.snap, r.buf) catch self.host.freeSnap(self.host.ctx, r.buf);
                    return;
                } else |_| {}
                side.rxEnd(r.rx, .abort, false);
                self.host.freeSnap(self.host.ctx, r.buf);
                return;
            }
            side.rxEnd(r.rx, .abort, true);
            self.host.freeSnap(self.host.ctx, r.buf);
        }

        fn removeFn(p: *anyopaque, key: sync.Key) void {
            const self = me(p);
            self.side.cancelKey(key);
            self.side.store.forget(key, &.{}, &.{});
        }

        fn damageFn(p: *anyopaque, key: sync.Key) void {
            _ = me(p).side.store.damage(key);
        }

        /// The drill's faults this rank carries out (TF_DS_SSD_FAULT).
        pub fn arm(self: *Self, f: sync.Armed) void {
            self.peer.fault = f;
        }

        fn listFn(p: *anyopaque, g: std.mem.Allocator, out: *std.ArrayList(sync.Listed)) anyerror!void {
            const self = me(p);
            var pts: std.ArrayList(core.spill.store.PointEntry) = .empty;
            defer pts.deinit(g);
            try self.side.store.pointList(g, &pts);
            for (pts.items) |e| try out.append(g, .{ .key = e.key, .digest = e.digest });
        }
    };
}
