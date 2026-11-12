//! F5 (TF_DS_PACK_EPILOGUE): the contract between gate/up's epilogue and the expert pack's transport, host side. The epilogue (native.cu's
//! gu_store) will emit the pack's codes and scales itself; these functions say where each value goes and when the switch may run, so the kernel
//! twin and packQuant agree byte for byte.
const std = @import("std");

const prompt2d = @import("prompt2d.zig");
const PackCodes = prompt2d.PackCodes;
const packCodesPlan = prompt2d.packCodesPlan;

/// Where one value of the expert pack lands as packQuant leaves it: the code byte at `code` (the value's flat index in gateUp's layout:
/// the main block [p, ig] first, then the rest block [p, ir]) and the 32-value group's fp32 scale at `scale` (pc.scales + 4 a group).
pub const PackSlot = struct { code: usize, scale: usize };

/// F5's contract for gate/up's epilogue (TF_DS_PACK_EPILOGUE): the pack's slot row `entry`, output column `col` of the layer's gate/up
/// columns (the first `pc.ig` are the main block, the next `pc.ir` the rest block: native.cu's gu_store target `col < main ? entry * ldh + col :
/// P * ldh + entry * rest_w + (col - main)` with ldh = ig, main = ig, rest_w = ir) goes to this place, so that a warp's 32 aligned columns
/// of one row are one group (one amax, one scale word) and the bytes equal packQuant's act_quant of the bf16 pack (which reads the pack as
/// pc.groups rows of 32 values: row r's codes at 32 r, its scale at pc.scales + 4 r).
pub fn packSlot(pc: PackCodes, entry: usize, col: usize) PackSlot {
    const flat = if (col < pc.ig) entry * pc.ig + col else pc.p * pc.ig + entry * pc.ir + (col - pc.ig);
    return .{ .code = flat, .scale = pc.scales + flat / 32 * 4 };
}

/// Why TF_DS_PACK_EPILOGUE, asked for, stays off (null: it runs): the expert pack's codes are the transport's (TF_DS_PACK_CODES in effect,
/// which has the rest block and whole 32-value groups), and gate/up's own store geometry is the pack's: the main block's row stride (`ldh`) and
/// its columns (`main`) are the pack's `ig`, the rest block's width (`rest_w`) is its `ir`, each a whole number of 32-value groups, so no group
/// straddles the blocks or a row.
pub fn packEpilogueOff(codes_on: bool, ig: usize, ir: usize, ldh: usize, main: usize, rest_w: usize) ?[]const u8 {
    if (!codes_on) return "needs TF_DS_PACK_CODES (the pack's codes are the transport's)";
    if (ig == 0 or ig % 32 != 0 or ir % 32 != 0) return "needs a pack of whole 32-value groups in both blocks";
    if (ldh != ig or main != ig or rest_w != ir) return "needs gate/up's store geometry to be the pack's (main stride and columns = the pair's ig, rest width = ir)";
    return null;
}

test "F5 contract: the gate and up epilogue places every value of the expert pack where packQuant's act_quant leaves its code and its group's scale" {
    for ([_][3]usize{ .{ 6, 128, 128 }, .{ 14, 256, 128 }, .{ 7, 128, 256 }, .{ 33, 384, 128 } }) |sh| {
        const pc = try packCodesPlan(sh[0], sh[1], sh[2]);
        const total = pc.p * (pc.ig + pc.ir);
        const seen = try std.testing.allocator.alloc(u8, total);
        defer std.testing.allocator.free(seen);
        @memset(seen, 0);
        const scale_users = try std.testing.allocator.alloc(u32, pc.groups);
        defer std.testing.allocator.free(scale_users);
        @memset(scale_users, 0);
        for (0..pc.p) |entry| for (0..pc.ig + pc.ir) |col| {
            const sl = packSlot(pc, entry, col);
            // the code is the flat index of gateUp's layout (main block, then rest block) and lands once
            try std.testing.expect(sl.code < total);
            seen[sl.code] += 1;
            // the scale is the group's: act_quant's row sl.code / 32, its word after the codes
            try std.testing.expectEqual(pc.scales + (sl.code / 32) * 4, sl.scale);
            try std.testing.expect(sl.scale >= pc.scales and sl.scale + 4 <= pc.bytes);
            // the 32 aligned columns of a block's row are one group: the same scale word for the warp's lanes
            const first = packSlot(pc, entry, col / 32 * 32);
            try std.testing.expectEqual(first.scale, sl.scale);
            if (col % 32 == 0) scale_users[(sl.scale - pc.scales) / 4] += 1;
        };
        for (seen) |c| try std.testing.expectEqual(@as(u8, 1), c);
        for (scale_users) |c| try std.testing.expectEqual(@as(u32, 1), c);
        try std.testing.expectEqual(pc.bytes, pc.scales + pc.groups * 4);
        // no group straddles the blocks or a row: the first rest value starts a group, and so does every row of either block
        try std.testing.expectEqual(@as(usize, 0), packSlot(pc, 0, pc.ig).code % 32);
        try std.testing.expectEqual(@as(usize, 0), packSlot(pc, pc.p - 1, 0).code % 32);
        try std.testing.expectEqual(@as(usize, 0), packSlot(pc, pc.p - 1, pc.ig).code % 32);
    }
}

test "F5 contract: the epilogue's switch stays off unless the pack's codes are the transport's and the gate and up store geometry is the pack's" {
    try std.testing.expectEqual(@as(?[]const u8, null), packEpilogueOff(true, 640, 320, 640, 640, 320));
    try std.testing.expect(packEpilogueOff(false, 640, 320, 640, 640, 320) != null); // no TF_DS_PACK_CODES
    try std.testing.expect(packEpilogueOff(true, 0, 320, 0, 0, 320) != null); // no main block
    try std.testing.expect(packEpilogueOff(true, 650, 320, 650, 650, 320) != null); // a group would straddle
    try std.testing.expect(packEpilogueOff(true, 640, 330, 640, 640, 330) != null);
    try std.testing.expect(packEpilogueOff(true, 640, 320, 704, 640, 320) != null); // gate/up's row stride is not the pack's
    try std.testing.expect(packEpilogueOff(true, 640, 320, 640, 576, 320) != null); // its main columns are not
    try std.testing.expect(packEpilogueOff(true, 640, 320, 640, 640, 256) != null); // its rest width is not
}

test "TF_DS_PACK_EPILOGUE is wired and refuses by name: off by default, the twin loaded at the start or the start ends, the twin launched where exact gate/up is and an error elsewhere, packQuant skipped only when the twin ran" {
    const mx_src = @embedFile("mx.zig");
    const mx_code = mx_src[0 .. std.mem.indexOf(u8, mx_src, "\ntest \"") orelse return error.TestUnexpectedResult];
    const m_src = @embedFile("model.zig");
    const m_code = m_src[0 .. std.mem.indexOf(u8, m_src, "\ntest \"") orelse return error.TestUnexpectedResult];
    const p_src = @embedFile("prompt2d.zig");
    const p_code = p_src;
    // off unless asked ("1"), a bit every rank agrees on
    try std.testing.expect(!prompt2d.pack_epilogue_asked);
    try std.testing.expectEqualStrings("TF_DS_PACK_EPILOGUE", prompt2d.exch_names[@ctz(prompt2d.exch_pack_epilogue)]);
    // the start: the twin by its name in the block-scaled kernels' module, a reason naming the setting and the kernel when the build lacks it, said on
    // every rank (on everywhere or off everywhere with the reason: the same bytes either way, so no refused start)
    try std.testing.expect(std.mem.indexOf(u8, mx_code, "Kernels.pfFunction(sk.module, \"tf_dsn_sf_gu_exact_q_kernel\", pf_gu_r128_smem) catch return error.NoPackEpilogueKernel") != null);
    try std.testing.expect(std.mem.indexOf(u8, m_code, "TF_DS_PACK_EPILOGUE=1 needs tf_dsn_sf_gu_exact_q_kernel, which this build lacks") != null);
    try std.testing.expect(std.mem.indexOf(u8, m_code, "TF_DS_PACK_EPILOGUE=1: off: {s} cannot run it (this rank: {s}); gate/up and packQuant run as two launches") != null);
    try std.testing.expect(std.mem.indexOf(u8, m_code, "which have no quantising epilogue") != null);
    const req = std.mem.indexOf(u8, m_code, "const cannot = try prompt2d.ranksThatCannot(d, &m.comm, m.stream, why == null);\n                if (cannot != 0) {\n                    var rb: [64]u8 = undefined;\n                    prompt2d.pack_epilogue_asked = false;").?;
    const codes = std.mem.indexOf(u8, m_code, "\"the expert pack as codes\"").?;
    try std.testing.expect(codes < req);
    // the launch: the twin only on the exact block-scaled path (else an error, not the baseline launch), with the QOut after the tile list
    const gu = mx_code[std.mem.indexOf(u8, mx_code, "fn gateUpOf(").?..];
    try std.testing.expect(std.mem.indexOf(u8, gu, "if (pq != null and (split or k.gu_q == null)) return error.PackEpilogueNotApplicable;") != null);
    const twin = std.mem.indexOf(u8, gu, "args.add(q);\n            return launchSharedT(k.gu_q.?").?;
    const plain = std.mem.indexOf(u8, gu, "if (sfExperts(k, nx.s1, false)) |fsf| return launchSharedT(fsf.f").?;
    try std.testing.expect(twin < plain);
    // not beside R2's or the 16-warp kernels (their tiles and blocks are other ones): an error, and the start line says so
    try std.testing.expect(std.mem.indexOf(u8, gu[0..twin], "if (k.sf != .exact or k.gemm_r2 or k.gemm_w16) return error.PackEpilogueNotApplicable;") != null);
    // QOut is the twin's struct: two pointers, three ints
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(@import("mx.zig").PackQ));
    // the call site: the twin writes into the send region and the scales after the codes, and packQuant runs only without it
    try std.testing.expect(std.mem.indexOf(u8, p_code, ".codes = mine, .scales = mine + pc.scales, .ig = @intCast(pc.ig), .ir = @intCast(pc.ir), .P = @intCast(pc.p)") != null);
    try std.testing.expect(std.mem.indexOf(u8, p_code, "if (!pack_epilogue_asked) try packQuant(k, e.s, xd, mine, pc);") != null);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, p_code, "packQuant(k, e.s, xd, mine, pc)"));
}
