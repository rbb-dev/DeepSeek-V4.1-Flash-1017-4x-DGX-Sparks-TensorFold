//! The prompt passes' finished rows, for /health's live prefill speed: a pass is issued by the host well before the device
//! finishes it (a pair's first pass returns after ~0.3-0.7 s of issue against ~1.2 s of kernels), so the rows count when the
//! pass's last kernel is done. A slot's cell holds `generation << 32 | rows`; a host function queued behind the pass's last
//! kernel raises it. Cells and marks live as long as the model: a callback of a fill that has since been re-armed (another
//! generation) changes nothing.
const std = @import("std");
const cuda = @import("cuda");

pub const max_slots = 32;
const max_marks = 16;

const Mark = struct { cell: *std.atomic.Value(u64), gen: u32, rows: u32 };

pub const Armed = struct { cell: *const std.atomic.Value(u64), gen: u32 };

pub const Meter = struct {
    cells: [max_slots]std.atomic.Value(u64) = @splat(.init(0)),
    marks: [max_marks]Mark = undefined,
    next: usize = 0,

    /// The cell of a slot a new fill is about to use (generation one up, no rows finished), null for a slot beyond the table.
    pub fn arm(m: *Meter, slot: usize) ?Armed {
        if (slot >= max_slots) return null;
        const gen: u32 = @truncate((m.cells[slot].load(.monotonic) >> 32) +% 1);
        m.cells[slot].store(@as(u64, gen) << 32, .release);
        return .{ .cell = &m.cells[slot], .gen = gen };
    }

    /// `rows` are finished once everything queued on `stream` so far is. Engine thread only.
    pub fn mark(m: *Meter, stream: cuda.Stream, slot: usize, rows: usize) !void {
        if (slot >= max_slots) return;
        const k = &m.marks[m.next % max_marks];
        m.next +%= 1;
        k.* = .{ .cell = &m.cells[slot], .gen = @truncate(m.cells[slot].load(.monotonic) >> 32), .rows = @intCast(rows) };
        try stream.hostFunc(done, k);
    }

    /// The host function: on a driver thread, no CUDA call. The rows only rise, and only within the generation they were queued for.
    fn done(data: ?*anyopaque) callconv(.c) void {
        const k: *const Mark = @ptrCast(@alignCast(data.?));
        finish(k.cell, k.gen, k.rows);
    }
};

/// The cell raised to `rows` of generation `gen` (never lowered, never across generations).
pub fn finish(cell: *std.atomic.Value(u64), gen: u32, rows: u32) void {
    var v = cell.load(.acquire);
    while (@as(u32, @truncate(v >> 32)) == gen and @as(u32, @truncate(v)) < rows) {
        v = cell.cmpxchgWeak(v, (@as(u64, gen) << 32) | rows, .acq_rel, .acquire) orelse return;
    }
}

test "a cell takes the rows of its own generation, only upward" {
    var m: Meter = .{};
    const a = m.arm(3).?;
    try std.testing.expectEqual(@as(u32, 1), a.gen);
    finish(&m.cells[3], a.gen, 4096);
    try std.testing.expectEqual((@as(u64, 1) << 32) | 4096, m.cells[3].load(.monotonic));
    finish(&m.cells[3], a.gen, 2048); // B finishing before A's mark lands: not lowered
    try std.testing.expectEqual((@as(u64, 1) << 32) | 4096, m.cells[3].load(.monotonic));
    const b = m.arm(3).?; // the slot's next fill
    try std.testing.expectEqual(@as(u32, 2), b.gen);
    finish(&m.cells[3], a.gen, 8192); // the first fill's late callback
    try std.testing.expectEqual(@as(u64, 2) << 32, m.cells[3].load(.monotonic));
    try std.testing.expect(m.arm(max_slots) == null);
}

test "a pair's two marks raise a gated pass in two steps" {
    const lanes = @import("lanes");
    var m: Meter = .{};
    var p: lanes.stream.Progress = .{};
    p.begin(0);
    const a = m.arm(0).?;
    p.gated(a.cell, a.gen);
    p.reach(8192); // the host returned from issuing the pair
    try std.testing.expectEqual(@as(u32, 0), p.computed());
    finish(&m.cells[0], a.gen, 4096); // pass A's mark
    try std.testing.expectEqual(@as(u32, 4096), p.computed());
    finish(&m.cells[0], a.gen, 8192); // pass B's
    try std.testing.expectEqual(@as(u32, 8192), p.computed());
}

test "the prompt step gates its progress and marks every pass it issues" {
    const src = @embedFile("lanes.zig");
    const arm = std.mem.indexOf(u8, src, "p.gated(g.cell, g.gen)").?;
    try std.testing.expect(std.mem.lastIndexOf(u8, src[0..arm], "p.begin(start)") != null);
    const issue = std.mem.indexOf(u8, src, "m.meterGroup(lane.slot, &.{ f.start - n3 - n2, f.start - n3, f.start })").?;
    const chunk = std.mem.indexOf(u8, src, "m.meterChunk(lane.slot, f.start)").?;
    const reach = std.mem.indexOf(u8, src, "p.reach(f.start)").?;
    try std.testing.expect(issue < reach and chunk < reach and std.mem.indexOf(u8, src[issue..reach], "fillGroup") == null);
    const fp = std.mem.indexOf(u8, src, "try m.fillGroup(").?;
    try std.testing.expect(fp < issue);
}
