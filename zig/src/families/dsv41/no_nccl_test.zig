//! The engine has no NCCL: nothing in its sources names the library or an entry point of it, the cuda module has no binding, and the
//! host collectives are the star's alone. The scan runs from the engine's root, as the other source checks do.
const std = @import("std");
const cuda = @import("cuda");
const comm = @import("comm.zig");

comptime {
    if (@hasDecl(cuda, "nccl")) @compileError("the cuda module still exposes NCCL");
    if (@hasField(comm.Comm, "lib") or @hasField(comm.Comm, "comm") or @hasField(comm.Comm, "links")) @compileError("Comm still holds an NCCL communicator");
    for ([_][]const u8{ "allGather", "allGatherMany", "maxBytes", "exchange", "exchangeMany", "broadcastSpans", "linkGather", "linkSum", "initLinks" }) |name| {
        if (@hasDecl(comm.Comm, name)) @compileError("Comm still has an NCCL call: " ++ name);
    }
}

const lib = "lib" ++ "nccl";
/// What names NCCL's library or its entry points (assembled so this file does not name them itself).
const tokens = [_][]const u8{ lib, "nccl" ++ "CommInitRank", "nccl" ++ "AllGather", "nccl" ++ "AllReduce", "nccl" ++ "Send", "nccl" ++ "Recv", "nccl" ++ "Broadcast", "nccl" ++ "GetUniqueId", "nccl" ++ "GroupStart", "@import(\"nccl" ++ ".zig\")", "dlopen(\"" ++ lib };

fn scan(gpa: std.mem.Allocator, io: std.Io, path: []const u8, found: *std.ArrayList([]u8)) !void {
    var d = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch return;
    defer d.close(io);
    var it = d.iterate();
    while (try it.next(io)) |e| {
        const sub = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ path, e.name });
        defer gpa.free(sub);
        switch (e.kind) {
            .directory => try scan(gpa, io, sub, found),
            .file => {
                if (std.mem.startsWith(u8, e.name, "nccl")) try found.append(gpa, try std.fmt.allocPrint(gpa, "{s}: a file named for NCCL", .{sub}));
                if (!(std.mem.endsWith(u8, e.name, ".zig") or std.mem.endsWith(u8, e.name, ".sh") or std.mem.endsWith(u8, e.name, ".py"))) continue;
                const bytes = std.Io.Dir.cwd().readFileAlloc(io, sub, gpa, .limited(32 << 20)) catch continue;
                defer gpa.free(bytes);
                for (tokens) |t| if (std.mem.indexOf(u8, bytes, t) != null) try found.append(gpa, try std.fmt.allocPrint(gpa, "{s}: {s}", .{ sub, t }));
            },
            else => {},
        }
    }
}

test "no source of the engine, its tests, its build or its tools names NCCL's library or an entry point of it" {
    const gpa = std.testing.allocator;
    var found: std.ArrayList([]u8) = .empty;
    defer {
        for (found.items) |x| gpa.free(x);
        found.deinit(gpa);
    }
    var seen = false;
    for ([_][]const u8{ "zig/src/families/dsv41", "zig/src/cuda", "zig/src/native/cuda.zig", "zig/build", "zig/kernels/cuda", "tools/dsv41-zig" }) |root| { // (the DeepSeek engine and the cuda module; BertholomusAI's other families keep their own NCCL users)
        var probe = std.Io.Dir.cwd().openDir(std.testing.io, root, .{}) catch continue;
        probe.close(std.testing.io);
        seen = true;
        try scan(gpa, std.testing.io, root, &found);
    }
    if (!seen) return error.SkipZigTest; // (not run from the engine's root)
    for (found.items) |x| std.debug.print("NCCL named: {s}\n", .{x});
    try std.testing.expectEqual(@as(usize, 0), found.items.len);
}

test "control: the scan finds a name it is told of" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "x.zig", .data = "const l = \"" ++ lib ++ ".so.2\";\n" });
    const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(path);
    var found: std.ArrayList([]u8) = .empty;
    defer {
        for (found.items) |x| gpa.free(x);
        found.deinit(gpa);
    }
    try scan(gpa, std.testing.io, path, &found);
    try std.testing.expectEqual(@as(usize, 1), found.items.len);
}

test "the host collectives are the star's: without its links a gather or a vote fails by name, never reaches another transport" {
    var c = comm.Comm.init(1, 4);
    try std.testing.expect(c.host.star == null);
    const stream: cuda.Stream = .{ .d = undefined, .handle = null };
    const d: *const cuda.Driver = undefined;
    try std.testing.expectError(error.NoHostLinks, c.gatherHost(d, 0x1000, 0x2000, 8, stream));
    try std.testing.expectError(error.NoHostLinks, c.maxHost(d, 0x3000, 1, stream));
    c.deinit();
}
