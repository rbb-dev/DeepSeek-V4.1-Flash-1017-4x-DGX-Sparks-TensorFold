//! A request's refusal: Python's RequestError family, or another exception's bare message.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Kind = enum {
    /// RequestError: 400, ``{"message", "type": "invalid_request_error"}``.
    request,
    /// ContextLengthError: as request, plus ``param`` and ``code: context_length_exceeded``.
    context_length,
    /// CapacityError: the engine is overloaded: 503 `overloaded_error` with Retry-After (529 on the Messages route), retry shortly.
    capacity,
    /// Any other exception (ValueError, KeyError...): 400 with ``{"message"}`` only.
    other,
    /// An exception while the reply runs: 500 with ``{"message"}``, or a stream's server_error event.
    server,
};

pub const context_limit = "This server's maximum context length is"; // OpenAI's wording, which clients match to compact

pub const Refused = error{ Refused, OutOfMemory };

/// Retry-After for a capacity refusal, in seconds: a constant. The engine does not know when a lane frees (its status holds
/// counts of running and waiting streams, not their remaining time), so an estimate would be a guess dressed as a number.
pub const retry_after_default_s: u32 = 5;

/// The Retry-After the reply being built on this thread carries (0: none). A capacity refusal sets it; the JSON sender adds the
/// header to a 503 or 529 and the route clears it when the request ends.
pub threadlocal var retry_after_s: u32 = 0;

/// The request's allocator and the refusal it ended with.
pub const Cx = struct {
    a: Allocator,
    kind: Kind = .request,
    message: []const u8 = "",

    pub fn fail(cx: *Cx, kind: Kind, comptime fmt: []const u8, args: anytype) Refused {
        cx.kind = kind;
        if (kind == .capacity) retry_after_s = retry_after_default_s;
        cx.message = try std.fmt.allocPrint(cx.a, fmt, args);
        return error.Refused;
    }

    pub fn refuse(cx: *Cx, message: []const u8) Refused {
        cx.kind = .request;
        cx.message = message;
        return error.Refused;
    }

    pub fn other(cx: *Cx, message: []const u8) Refused {
        cx.kind = .other;
        cx.message = message;
        return error.Refused;
    }

    pub fn status(cx: *const Cx) u16 {
        return switch (cx.kind) {
            .capacity => 503,
            .server => 500,
            else => 400,
        };
    }
};
