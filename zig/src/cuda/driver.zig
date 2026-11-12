//! The CUDA driver opened at run time: libcuda.so.1's versioned entry points in one table, and checked calls.

const std = @import("std");
const abi = @import("abi.zig");

pub const Error = error{ DriverUnavailable, MissingSymbol, CudaFailed, CudaContextLost, OutOfDeviceMemory, OutOfMemory, NotReady, NotFound, Invalid };

pub const Driver = struct {
    lib: std.DynLib,
    api: abi.Api,

    pub fn open() Error!Driver {
        return openPath("libcuda.so.1");
    }

    /// Resolves every field of `abi.Api` by its exact name; a missing symbol refuses the whole driver. cuInit runs
    /// first, so a profiler's injection (Nsight Systems' CUPTI, loaded by cuInit) is in place when the entry points are
    /// resolved. TF_CUDA_PROC_ADDRESS=1: the launches (cuLaunchKernel, cuLaunchKernelEx, cuGraphLaunch) through
    /// cuGetProcAddress at CUDA 12.0's versions, the table such an injection hooks.
    pub fn openPath(path: []const u8) Error!Driver {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        var api: abi.Api = undefined;
        api.cuInit = lib.lookup(@TypeOf(api.cuInit), "cuInit") orelse return error.MissingSymbol;
        api.cuGetErrorName = lib.lookup(@TypeOf(api.cuGetErrorName), "cuGetErrorName") orelse return error.MissingSymbol;
        api.cuGetErrorString = lib.lookup(@TypeOf(api.cuGetErrorString), "cuGetErrorString") orelse return error.MissingSymbol;
        try (Driver{ .lib = lib, .api = api }).check(api.cuInit(0), "cuInit");
        const info = @typeInfo(abi.Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        if (std.c.getenv("TF_CUDA_PROC_ADDRESS")) |v| if (v[0] == '1') {
            const GetProc = *const fn ([*:0]const u8, *?*anyopaque, c_int, u64, ?*c_int) callconv(.c) abi.Result;
            const get = lib.lookup(GetProc, "cuGetProcAddress_v2") orelse return error.MissingSymbol;
            inline for (.{ "cuLaunchKernel", "cuLaunchKernelEx", "cuGraphLaunch" }) |name| {
                var p: ?*anyopaque = null;
                var status: c_int = 0;
                if (get(name, &p, 12000, 0, &status) == abi.success and status == 0 and p != null) @field(api, name) = @ptrCast(@alignCast(p.?));
            }
        };
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Driver) void {
        self.lib.close();
    }

    /// Logs a failed call with the driver's own name and text for the code; NOT_READY is a state, not a failure.
    pub fn check(self: *const Driver, res: abi.Result, what: []const u8) Error!void {
        if (res == abi.success) return;
        if (res == abi.error_not_ready) return error.NotReady;
        @import("context_health.zig").observe(self, res); // (before the log line: readers see a lost context at once)
        std.log.err("{s}: {s} ({d}) {s}", .{ what, self.errorName(res), res, self.errorText(res) });
        return errorOf(res);
    }

    /// The one place a synchronous host-to-device copy is issued (a source-scan test holds every call site of the engine to it). `loc` is the caller's @src(), comptime: nothing of it
    /// costs anything when the copy succeeds; a failure logs the call site, both ends and the length (copyFailed).
    pub inline fn copyHtoD(self: *const Driver, comptime loc: std.builtin.SourceLocation, dst: abi.DevicePtr, src: ?*const anyopaque, bytes: usize) Error!void {
        const res = self.api.cuMemcpyHtoD_v2(dst, src, bytes);
        if (res == abi.success) return;
        return self.copyFailed("cuMemcpyHtoD", loc, dst, src, bytes, null, res);
    }

    /// `copyHtoD` on `stream`: the asynchronous copy (a pageable source is staged by the driver before the call returns).
    pub inline fn copyHtoDAsync(self: *const Driver, comptime loc: std.builtin.SourceLocation, dst: abi.DevicePtr, src: ?*const anyopaque, bytes: usize, stream: abi.Stream) Error!void {
        const res = self.api.cuMemcpyHtoDAsync_v2(dst, src, bytes, stream);
        if (res == abi.success) return;
        return self.copyFailed("cuMemcpyHtoDAsync", loc, dst, src, bytes, stream, res);
    }

    /// A failed host-to-device copy: the line `check` would write (the driver's name and text for the code, first, as every log reader has it), then where it was issued
    /// (file, line, function), the destination (and the device allocation around it: its base, its size, and whether the copy leaves it), the source (its memory kind to the driver:
    /// host / device / unified, or "unknown to CUDA": pageable, or not an address at all), the length and the stream. Only this branch asks the driver anything.
    noinline fn copyFailed(self: *const Driver, what: []const u8, comptime loc: std.builtin.SourceLocation, dst: abi.DevicePtr, src: ?*const anyopaque, bytes: usize, stream: ?abi.Stream, res: abi.Result) Error {
        if (res == abi.error_not_ready) return error.NotReady;
        @import("context_health.zig").observe(self, res);
        var db: [160]u8 = undefined;
        var sb: [96]u8 = undefined;
        std.log.err("{s}: {s} ({d}) {s}; from {s}:{d} in {s}: dst 0x{x} ({s}), src 0x{x} ({s}), {d} bytes{s}", .{
            what,                                                  self.errorName(res), res,                     self.errorText(res),
            loc.file,                                              loc.line,            loc.fn_name,             dst,
            self.deviceEnd(dst, bytes, &db),                       @intFromPtr(src),    self.hostKind(src, &sb), bytes,
            if (stream) |_| ", asynchronous" else ", synchronous",
        });
        return errorOf(res);
    }

    /// What the driver says the source of a host-to-device copy is (the failure branch only).
    fn hostKind(self: *const Driver, src: ?*const anyopaque, buf: []u8) []const u8 {
        var kind: c_uint = 0;
        const r = self.api.cuPointerGetAttribute(&kind, 2, @intFromPtr(src)); // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
        if (r != abi.success) return std.fmt.bufPrint(buf, "unknown to CUDA: pageable host memory, or not an address ({s})", .{self.errorName(r)}) catch "unknown to CUDA";
        return switch (kind) {
            1 => "page-locked host memory",
            2 => "device memory",
            3 => "array",
            4 => "unified memory",
            else => std.fmt.bufPrint(buf, "memory type {d}", .{kind}) catch "memory type",
        };
    }

    /// The device allocation around a copy's destination (the failure branch only): its base and size, and whether dst + bytes stays inside it.
    fn deviceEnd(self: *const Driver, dst: abi.DevicePtr, bytes: usize, buf: []u8) []const u8 {
        var base: abi.DevicePtr = 0;
        var size: usize = 0;
        const r = self.api.cuMemGetAddressRange_v2(&base, &size, dst);
        if (r != abi.success) return std.fmt.bufPrint(buf, "in no device allocation ({s})", .{self.errorName(r)}) catch "in no device allocation";
        const off = dst - base;
        return std.fmt.bufPrint(buf, "allocation 0x{x} + {d} bytes, offset {d}, {s}", .{ base, size, off, if (off + bytes > size) "the copy LEAVES it" else "the copy stays inside it" }) catch "allocation";
    }

    /// `check` without its log line: a sticky fault is kept for the context's health all the same.
    pub fn result(self: *const Driver, res: abi.Result) Error!void {
        if (res == abi.success) return;
        if (res == abi.error_not_ready) return error.NotReady;
        @import("context_health.zig").observe(self, res);
        return errorOf(res);
    }

    /// The error a failed call returns.
    pub fn errorOf(res: abi.Result) Error {
        if (@import("context_health.zig").sticky(res)) return error.CudaContextLost;
        return switch (res) {
            2 => error.OutOfDeviceMemory,
            abi.error_not_found => error.NotFound,
            else => error.CudaFailed,
        };
    }

    pub fn errorName(self: *const Driver, res: abi.Result) []const u8 {
        var s: ?[*:0]const u8 = null;
        if (self.api.cuGetErrorName(res, &s) != abi.success) return "CUDA_ERROR_UNKNOWN_CODE";
        return if (s) |p| std.mem.span(p) else "CUDA_ERROR_UNKNOWN_CODE";
    }

    pub fn errorText(self: *const Driver, res: abi.Result) []const u8 {
        var s: ?[*:0]const u8 = null;
        if (self.api.cuGetErrorString(res, &s) != abi.success) return "";
        return if (s) |p| std.mem.span(p) else "";
    }

    /// The driver's CUDA version as 1000 * major + 10 * minor.
    pub fn version(self: *const Driver) Error!c_int {
        var v: c_int = 0;
        try self.check(self.api.cuDriverGetVersion(&v), "cuDriverGetVersion");
        return v;
    }

    pub fn deviceCount(self: *const Driver) Error!c_int {
        var n: c_int = 0;
        try self.check(self.api.cuDeviceGetCount(&n), "cuDeviceGetCount");
        return n;
    }

    /// Device `ordinal`'s compute capability as 10 * major + minor, read without making a context.
    pub fn capability(self: *const Driver, ordinal: c_int) Error!u32 {
        var dev: abi.Device = 0;
        try self.check(self.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var major: c_int = 0;
        var minor: c_int = 0;
        try self.check(self.api.cuDeviceGetAttribute(&major, .compute_capability_major, dev), "cuDeviceGetAttribute");
        try self.check(self.api.cuDeviceGetAttribute(&minor, .compute_capability_minor, dev), "cuDeviceGetAttribute");
        return @intCast(10 * major + minor);
    }

    /// Device `ordinal`'s name ("NVIDIA GB10"), read without making a context.
    pub fn deviceName(self: *const Driver, ordinal: c_int, buf: []u8) Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        var dev: abi.Device = 0;
        try self.check(self.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        try self.check(self.api.cuDeviceGetName(buf.ptr, @intCast(buf.len), dev), "cuDeviceGetName");
        return std.mem.sliceTo(buf, 0);
    }
};

test "every host-to-device copy of the cuda module goes through Driver.copyHtoD / copyHtoDAsync, the one place that names the driver's entry points and says where a failure came from" {
    const d = @embedFile("driver.zig");
    const m = @embedFile("memory.zig");
    // the driver's own two calls, once each; no other file of this module calls either entry point
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, d, "self.api.cuMemcpy" ++ "HtoD_v2(dst, src, bytes)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, d, "self.api.cuMemcpy" ++ "HtoDAsync_v2(dst, src, bytes, stream)"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, m, "api.cuMemcpy" ++ "HtoD"));
    // the failure branch alone asks the driver about the two ends
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, d, "self.api.cuPointer" ++ "GetAttribute("));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, d, "self.api.cuMemGet" ++ "AddressRange_v2("));
    // the success path is the call and one comparison: nothing is formatted before it
    const at = std.mem.indexOf(u8, d, "pub inline fn copyHtoDAsync(").?;
    const body = d[at .. at + std.mem.indexOf(u8, d[at..], "\n    }\n").?];
    try std.testing.expect(std.mem.indexOf(u8, body, "if (res == abi.success) return;").? < std.mem.indexOf(u8, body, "copyFailed(").?);
    try std.testing.expect(std.mem.indexOf(u8, body, "bufPrint") == null and std.mem.indexOf(u8, body, "std.log") == null);
}
