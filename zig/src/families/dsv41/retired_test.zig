//! A retired switch set in the environment ends Model.open at its first line, by name (core/settings.zig's retired table).
const std = @import("std");
const cuda = @import("cuda");
const model = @import("model.zig");

test "a retired switch set in the environment ends Model.open before anything is made, by name; a clean environment passes" {
    const env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    var buf: [512]u8 = undefined;
    // a retired name already set by the caller would make the control meaningless
    inline for (.{ "TF_DS_ROCI_DECODE", "TF_DS_ROCI", "TF_DS_SSD_EVERY_RANK", "TF_DS_ENGRAM_OWNED" }) |n| if (std.c.getenv(n) != null) return error.SkipZigTest;
    try std.testing.expect(model.retiredSwitch(&buf) == null);
    defer _ = env.unsetenv("TF_DS_ROCI_DECODE");
    _ = env.setenv("TF_DS_ROCI_DECODE", "0", 1);
    try std.testing.expectEqualStrings("TF_DS_ROCI_DECODE no longer exists: Rocinante is the only transport", model.retiredSwitch(&buf).?);
    _ = env.unsetenv("TF_DS_ROCI_DECODE");
    // the recipe exports TF_DS_ENGRAM_OWNED=0: the one state that remains passes; on is refused
    defer _ = env.unsetenv("TF_DS_ENGRAM_OWNED");
    _ = env.setenv("TF_DS_ENGRAM_OWNED", "0", 1);
    try std.testing.expect(model.retiredSwitch(&buf) == null);
    _ = env.setenv("TF_DS_ENGRAM_OWNED", "1", 1);
    try std.testing.expect(std.mem.startsWith(u8, model.retiredSwitch(&buf).?, "TF_DS_ENGRAM_OWNED no longer exists: "));
    // Model.open asks first: before the model's allocation, the links and the CUDA context's first use
    const src = @embedFile("model.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: Options) !*Model {").?;
    const open = src[at..];
    try std.testing.expect(std.mem.indexOf(u8, open, "retiredSwitch(&why)").? < std.mem.indexOf(u8, open, "const m = try gpa.create(Model);").?);
    try std.testing.expect(std.mem.indexOf(u8, open, "if (o.world != 4)").? < std.mem.indexOf(u8, open, "const m = try gpa.create(Model);").?);
}
