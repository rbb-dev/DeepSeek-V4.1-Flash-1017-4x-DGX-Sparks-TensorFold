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
const builtin = @import("builtin");
const lanes = @import("lanes");
const fill_order = @import("fill_order.zig");

pub const Slot = enum(u8) {
    rounds_beside,
    log_polls,
    prefill_share,
    pass_beside,
    spill_load_ms,
    spill_pump_ms,
    spill_load_wait_ms,
    spill_flush_ms,
    spill_defer_ms,
    spill_defer_retry_ms,
    spill_defer_min,
    spill_poll_ms,
    keep_ahead,
    graph_cap,
    kv_free_target,
};
pub const slot_count = std.enums.values(Slot).len;
/// A slot holding this has no value: the engine's own (the start's).
pub const unset: u64 = std.math.maxInt(u64);

pub const Wired = struct { id: u16, slot: Slot };
/// The rows with a hook. 122 tf_ds_rounds_beside, 100 tensorfold_prefill_share and 106 tf_ds_pass_beside are the host's balance of
/// prompt steps and decode rounds; 446 to 453 are its tier timings (TF_SPILL_*_MS and TF_SPILL_DEFER_MIN, which the host reads at
/// each service); 4 tensorfold_log_polls is a flag on the server's own threads. The three fill-order rows (119, 120, 121) are one
/// policy, published whole (`publishFill`).
pub const wired = [_]Wired{
    .{ .id = 122, .slot = .rounds_beside },
    .{ .id = 4, .slot = .log_polls },
    .{ .id = 100, .slot = .prefill_share },
    .{ .id = 106, .slot = .pass_beside },
    .{ .id = 450, .slot = .spill_load_ms },
    .{ .id = 453, .slot = .spill_pump_ms },
    .{ .id = 451, .slot = .spill_load_wait_ms },
    .{ .id = 449, .slot = .spill_flush_ms },
    .{ .id = 447, .slot = .spill_defer_ms },
    .{ .id = 448, .slot = .spill_defer_retry_ms },
    .{ .id = 446, .slot = .spill_defer_min },
    .{ .id = 452, .slot = .spill_poll_ms },
    // class A, sent to every rank in a cfg frame (412 tf_ds_engram_keep_ahead: whether a round between a prompt's chunks leaves the Engram worker running)
    .{ .id = 412, .slot = .keep_ahead },
    // 105 tf_ds_graph_cap (class A): the CUDA graphs kept at most, 0 all; lowering destroys the excess on every rank at the frame
    .{ .id = 105, .slot = .graph_cap },
    // 505 tf_ds_kv_free_target_gib (class A, with TF_DS_KV_VMM=1): MiB the pool leaves free; every rank resizes it at the frame (0: the start's pool)
    .{ .id = 505, .slot = .kv_free_target },
};

pub fn slotOf(id: u32) ?Slot {
    for (wired) |w| if (w.id == id) return w.slot;
    return null;
}

/// 119 tf_ds_prompt_max_wait_s, 120 tf_ds_prompt_order and 121 tf_ds_prompt_qos: the host's prompt-step order, one policy.
pub fn isFillRow(id: u32) bool {
    return id == 119 or id == 120 or id == 121;
}

/// Whether the engine takes a change to this row while it runs.
pub fn isWired(id: u32) bool {
    return takes(id) or isFillRow(id);
}

/// What the host takes at a turn: every slot, and the fill policy when one was published (null: the start's own).
pub const Snapshot = struct { vals: [slot_count]u64, fill: ?fill_order.Policy };

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

/// The fallback fence of an architecture the two fences of Block do not name: a sequentially consistent no-op RMW.
var barrier_word: std.atomic.Value(u32) = .init(0);

pub const Block = struct {
    /// The seqlock's word: odd while a publication is being written.
    seq: std.atomic.Value(u64) = .init(0),
    values: [slot_count]std.atomic.Value(u64) = @splat(.init(unset)),
    /// The published fill policy (`has_fill`: false means the start's own); read under the seqlock like the values.
    fill: fill_order.Policy = .{},
    has_fill: bool = false,
    /// The version of the newest publication, and the newest the host has taken.
    requested_version: std.atomic.Value(u64) = .init(0),
    active_version: std.atomic.Value(u64) = .init(0),
    mutex: std.atomic.Mutex = .unlocked,

    /// The seqlock's writer side, after the odd store: the data stores that follow must not become visible before it. A release
    /// store only orders what comes before it, and Arm lets a later store pass an earlier one, so without this a reader could see
    /// the new data under a still-even word and take half a publication. (x86 keeps stores in order: a compiler barrier.)
    fn storesAfterOdd() void {
        switch (builtin.cpu.arch) {
            .aarch64 => asm volatile ("dmb ishst" ::: .{ .memory = true }),
            .x86_64 => asm volatile ("" ::: .{ .memory = true }),
            else => _ = barrier_word.fetchAdd(0, .seq_cst),
        }
    }

    /// The seqlock's reader side, between the data loads and the re-read of the word: an acquire load orders what follows it, not
    /// the loads before it, so on Arm the data loads could be satisfied after the re-read and a torn pair pass the check.
    fn loadsBeforeRecheck() void {
        switch (builtin.cpu.arch) {
            .aarch64 => asm volatile ("dmb ishld" ::: .{ .memory = true }),
            .x86_64 => asm volatile ("" ::: .{ .memory = true }),
            else => _ = barrier_word.fetchAdd(0, .seq_cst),
        }
    }

    fn lock(b: *Block) void {
        while (!b.mutex.tryLock()) std.atomic.spinLoopHint();
    }

    /// Publishes every slot at once as one version (the writers' side; any thread).
    pub fn publishAll(b: *Block, vals: [slot_count]u64, version: u64) void {
        b.lock();
        defer b.mutex.unlock();
        const s = b.seq.load(.monotonic);
        b.seq.store(s + 1, .release);
        storesAfterOdd();
        for (&b.values, vals) |*slot, v| slot.store(v, .monotonic);
        b.seq.store(s + 2, .release);
        b.requested_version.store(version, .release);
    }

    /// Publishes the fill policy (null: back to the start's), the slots as they are.
    pub fn publishFill(b: *Block, policy: ?fill_order.Policy, version: u64) void {
        b.lock();
        defer b.mutex.unlock();
        const s = b.seq.load(.monotonic);
        b.seq.store(s + 1, .release);
        storesAfterOdd();
        b.has_fill = policy != null;
        if (policy) |p| b.fill = p;
        b.seq.store(s + 2, .release);
        b.requested_version.store(version, .release);
    }

    /// Publishes one slot, the others as they are.
    pub fn publish(b: *Block, slot: Slot, value: u64, version: u64) void {
        b.lock();
        defer b.mutex.unlock();
        const s = b.seq.load(.monotonic);
        b.seq.store(s + 1, .release);
        storesAfterOdd();
        b.values[@backingInt(slot)].store(value, .monotonic);
        b.seq.store(s + 2, .release);
        b.requested_version.store(version, .release);
    }

    /// The host's call, once a loop turn: false (one load) while the version is the one it has; else everything published into `out`
    /// (never a torn publication), `seen` and `active_version` at their version, true.
    pub fn turnAll(b: *Block, seen: *u64, out: *Snapshot) bool {
        const want = b.requested_version.load(.acquire);
        if (want == seen.*) return false;
        while (true) {
            const s1 = b.seq.load(.acquire);
            if (s1 & 1 != 0) {
                std.atomic.spinLoopHint();
                continue;
            }
            for (&b.values, &out.vals) |*slot, *o| o.* = slot.load(.monotonic);
            const has = b.has_fill;
            const policy = b.fill;
            loadsBeforeRecheck();
            if (b.seq.load(.acquire) == s1) {
                out.fill = if (has) policy else null;
                break;
            }
        }
        // the version a reader pairs with the values is the newest one published by now (values may be newer than `want`)
        const got = b.requested_version.load(.acquire);
        seen.* = got;
        b.active_version.store(got, .release);
        return true;
    }

    /// The slots alone, for a caller that has no use for the policy.
    pub fn turn(b: *Block, seen: *u64, out: *[slot_count]u64) bool {
        var snap: Snapshot = undefined;
        if (!b.turnAll(seen, &snap)) return false;
        out.* = snap.vals;
        return true;
    }
};

/// The process's block: the API publishes into it, rank 0's host takes from it.
pub var block: Block = .{};
