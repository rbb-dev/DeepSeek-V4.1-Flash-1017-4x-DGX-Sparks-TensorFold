//! spill_state over the native checkpoint format (DeepSeek's own FP8 linears and FP4 experts): its state layout is the
//! exl3 fixture's, because the KV pool's planes do not depend on how the weights are quantized.
const std = @import("std");
const config = @import("config.zig");
const spill = @import("spill_state.zig");

test "the native-format checkpoint has the exl3 fixture's state layout: the KV pool does not depend on the weight format" {
    const a = std.testing.allocator;
    var why: config.Why = .{};
    const native = try std.mem.replaceOwned(u8, a, config.test_config, "{\"quant_method\": \"exl3\", \"codebook\": \"mul1\"}", "{\"quant_method\": \"fp8\", \"activation_scheme\": \"dynamic\", \"weight_block_size\": [32, 32], \"scale_fmt\": \"ue8m0\", \"expert_dtype\": \"fp4\"}");
    defer a.free(native);
    const cn = try config.parse(a, native, &why);
    const ce = try config.parse(a, config.test_config, &why);
    try std.testing.expectEqual(config.Format.native, cn.format);
    const ln = try spill.StateLayout.fromConfig(&cn, spill.Geometry.of(&cn, 2048, 16, 64), .replicated);
    const le = try spill.StateLayout.fromConfig(&ce, spill.Geometry.of(&ce, 2048, 16, 64), .replicated);
    try std.testing.expectEqual(le.block_bytes, ln.block_bytes);
    try std.testing.expectEqual(le.small_bytes, ln.small_bytes);
    try std.testing.expectEqualSlices(u8, &le.digest(), &ln.digest());
}
