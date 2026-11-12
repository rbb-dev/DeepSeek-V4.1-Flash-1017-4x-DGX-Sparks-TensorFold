//! prompt step) or is applied by the API itself on a thread that is not the engine's (the log of polls), or a row of class A, which
//! the host does not apply by itself: it sends the change to every rank in one cfg frame (`CfgFrame`, kind 16) and each rank, rank 0
//! included, applies it between two frames. Every other row stays a setting that applies at the next start, and the API says so.
//! A row of class A is live only where the engine says it can take it (`able`: what the start verified), else it is a setting for the
//! next start.
const lanes = @import("lanes");
    keep_ahead,
    // class A, sent to every rank in a cfg frame (412 tf_ds_engram_keep_ahead: whether a round between a prompt's chunks leaves the Engram worker running)
    .{ .id = 412, .slot = .keep_ahead },
    return takes(id) or isFillRow(id);
/// Whether the slot's change goes to every rank in a cfg frame (class A) instead of being taken by rank 0's host alone.
pub fn isA(slot: Slot) bool {
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
