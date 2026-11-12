//! A tensor-parallel group of ranks for a family that splits a layer over processes (Nemotron's latent MoE,
//! TF_NEMOTRON_RANK of TF_NEMOTRON_WORLD). This engine has no NCCL (the DeepSeek engine's exchanges are its RDMA rings and the
//! star's host collectives), so a group cannot be opened: `open` says so and every other call refuses. The type keeps the
//! family's call sites compiling; BertholomusAI's group over NCCL is not in this build.
const std = @import("std");
const driver = @import("driver.zig");
const stream_ = @import("stream.zig");

const Stream = stream_.Stream;

pub const Group = struct {
    rank: usize,
    world: usize,

    /// Always refused: no collective library is linked here.
    pub fn open(rank: usize, world: usize) !Group {
        _ = rank;
        _ = world;
        return error.NoTensorParallel;
    }

    pub fn close(g: *Group) void {
        g.* = undefined;
    }

    pub fn allGather(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
        _ = .{ g, s, mine, all, bytes };
        return error.NoTensorParallel;
    }

    pub fn send(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        _ = .{ g, s, ptr, bytes, peer };
        return error.NoTensorParallel;
    }

    pub fn recv(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        _ = .{ g, s, ptr, bytes, peer };
        return error.NoTensorParallel;
    }

    pub fn connect(g: *const Group, s: Stream, ptr: u64, peer: usize) !void {
        _ = .{ g, s, ptr, peer };
        return error.NoTensorParallel;
    }

    pub fn gather(g: *const Group, comptime T: type, d: *const driver.Driver, s: Stream, mine: T, out: []T) !void {
        _ = .{ g, d, s, mine, out };
        return error.NoTensorParallel;
    }
};
