//! The live block: the settings of class H that rank 0's host takes at its next loop turn.
//!
//! The admin API's threads publish a value into the block (a mutex among the writers, a seqlock for the reader); the host calls
//! `turn` once a loop turn, which is one atomic load of the version word while nothing has changed. A change copies the
//! published values under the seqlock, so a turn never reads half of a publication, and then raises `active_version`, which is
//! what a writer waiting for its change polls. Nothing here is taken by the engine thread except that one load.
//!
//! Only the rows listed in `wired` are live: each is a row of class H whose consumer reads the block (the host's rounds beside a
//! prompt step) or is applied by the API itself on a thread that is not the engine's (the log of polls), or a row of class A, which
//! the host does not apply by itself: it sends the change to every rank in one cfg frame (`CfgFrame`, kind 16) and each rank, rank 0
//! included, applies it between two frames. Every other row stays a setting that applies at the next start, and the API says so.
//! A row of class A is live only where the engine says it can take it (`able`: what the start verified), else it is a setting for the
//! next start.
const std = @import("std");
const lanes = @import("lanes");

    keep_ahead,
    kv_free_target,
pub const slot_count = std.enums.values(Slot).len;
/// A slot holding this has no value: the engine's own (the start's).
pub const unset: u64 = std.math.maxInt(u64);

pub const Wired = struct { id: u16, slot: Slot };
    // class A, sent to every rank in a cfg frame (412 tf_ds_engram_keep_ahead: whether a round between a prompt's chunks leaves the Engram worker running)
    .{ .id = 412, .slot = .keep_ahead },
    // 505 tf_ds_kv_free_target_gib (class A, with TF_DS_KV_VMM=1): MiB the pool leaves free; every rank resizes it at the frame (0: the start's pool)
    .{ .id = 505, .slot = .kv_free_target },

pub fn slotOf(id: u32) ?Slot {
    for (wired) |w| if (w.id == id) return w.slot;
    return null;
}

    return takes(id) or isFillRow(id);
/// Whether the slot's change goes to every rank in a cfg frame (class A) instead of being taken by rank 0's host alone.
pub fn isA(slot: Slot) bool {
    return slot == .keep_ahead or slot == .graph_cap or slot == .kv_free_target;
}

/// The slots of class A the engine can take now, one bit each: set by the family at the start, once it has verified the slot's
/// preconditions on this rank (keep-ahead: an Engram worker, its own reader rings, no shared scratch).
pub var able: std.atomic.Value(u32) = .init(0);

pub fn setAble(slot: Slot, on: bool) void {
    const bit = @as(u32, 1) << @intCast(@backingInt(slot));
    if (on) _ = able.fetchOr(bit, .release) else _ = able.fetchAnd(~bit, .release);
}

/// Whether a change of the row is live now: it has a hook, and a row of class A is one the engine can take.
pub fn takes(id: u32) bool {
    const slot = slotOf(id) orelse return false;
    if (!isA(slot)) return true;
    return able.load(.acquire) & (@as(u32, 1) << @intCast(@backingInt(slot))) != 0;
}

/// A change of class A as it travels: the slot (a dense index, not the option id: E5) and its value, `cfg_unset` for "the start's".
pub const CfgItem = lanes.backend.CfgItem;
pub const cfg_unset = lanes.backend.cfg_unset;

/// The cfg frame's bytes after the frame header (kind u8, step u64): version u64, flags u8, count u8, then count x (slot u8, value
/// u32), all little-endian. At most 255 items: 10 + 5 x 255 = 1,285 bytes, so the frame with its header is one send() of the link
/// (at most 2,048 bytes).
pub const CfgFrame = struct {
    pub const max_items = 255;
    pub const head_len = 10;

    pub fn encode(buf: []u8, version: u64, flags: u8, items: []const CfgItem) ![]u8 {
        if (items.len > max_items or buf.len < head_len + 5 * items.len) return error.CfgTooLong;
        std.mem.writeInt(u64, buf[0..8], version, .little);
        buf[8] = flags;
        buf[9] = @intCast(items.len);
        for (items, 0..) |it, i| {
            buf[head_len + 5 * i] = it.slot;
            std.mem.writeInt(u32, buf[head_len + 5 * i + 1 ..][0..4], it.value, .little);
        }
        return buf[0 .. head_len + 5 * items.len];
    }

    pub const Decoded = struct { version: u64, flags: u8, n: u8, bytes: []const u8 };

    pub fn decode(b: []const u8) !Decoded {
        if (b.len < head_len) return error.BadFrame;
        const n = b[9];
        if (b.len != head_len + 5 * @as(usize, n)) return error.BadFrame;
        return .{ .version = std.mem.readInt(u64, b[0..8], .little), .flags = b[8], .n = n, .bytes = b[head_len..] };
    }

    pub fn item(d: Decoded, i: usize) CfgItem {
        return .{ .slot = d.bytes[5 * i], .value = std.mem.readInt(u32, d.bytes[5 * i + 1 ..][0..4], .little) };
    }
};

pub const Block = struct {
    /// The seqlock's word: odd while a publication is being written.
    seq: std.atomic.Value(u64) = .init(0),
    values: [slot_count]std.atomic.Value(u64) = @splat(.init(unset)),
    /// The version of the newest publication, and the newest the host has taken.
    requested_version: std.atomic.Value(u64) = .init(0),
    active_version: std.atomic.Value(u64) = .init(0),
    mutex: std.atomic.Mutex = .unlocked,

    fn lock(b: *Block) void {
        while (!b.mutex.tryLock()) std.atomic.spinLoopHint();
    }

    /// Publishes every slot at once as one version (the writers' side; any thread).
    pub fn publishAll(b: *Block, vals: [slot_count]u64, version: u64) void {
        b.lock();
        defer b.mutex.unlock();
        const s = b.seq.load(.monotonic);
        b.seq.store(s + 1, .release);
        for (&b.values, vals) |*slot, v| slot.store(v, .monotonic);
        b.seq.store(s + 2, .release);
        b.requested_version.store(version, .release);
    }

    /// Publishes one slot, the others as they are.
    pub fn publish(b: *Block, slot: Slot, value: u64, version: u64) void {
        b.lock();
        defer b.mutex.unlock();
        const s = b.seq.load(.monotonic);
        b.seq.store(s + 1, .release);
        b.values[@backingInt(slot)].store(value, .monotonic);
        b.seq.store(s + 2, .release);
        b.requested_version.store(version, .release);
    }

        const want = b.requested_version.load(.acquire);
        if (want == seen.*) return false;
        while (true) {
            const s1 = b.seq.load(.acquire);
            if (s1 & 1 != 0) {
                std.atomic.spinLoopHint();
                continue;
            }
        }
        // the version a reader pairs with the values is the newest one published by now (values may be newer than `want`)
        const got = b.requested_version.load(.acquire);
        seen.* = got;
        b.active_version.store(got, .release);
        return true;
    }
};

/// The process's block: the API publishes into it, rank 0's host takes from it.
pub var block: Block = .{};
