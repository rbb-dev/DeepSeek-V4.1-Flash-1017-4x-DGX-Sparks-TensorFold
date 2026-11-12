//! What the engine decided about its transports at the start, for the admin surface (/status transport): written once by the
//! family while it opens (Rocinante's start line, the rings' switches, the link watch), read by a request for /status. Plain words
//! the page prints; nothing here is read by an exchange.
pub const Signal = enum { unknown, write, atomic };

pub const Facts = struct {
    /// Rocinante (the prompt exchanges): recorded at its start line.
    rocinante_known: bool = false,
    /// The queue pairs wait in the card (no proxy threads), else the proxy threads run.
    rocinante_wait: bool = false,
    rocinante_signal: Signal = .unknown,
    /// The decode rings: recorded when the rings are opened, or found not to exist.
    rings_known: bool = false,
    rings_on: bool = false,
    rings_wait: bool = false,
    /// The rings' arrival signal: a generation word (write or atomic), else none (the rings' completion events).
    rings_word: bool = false,
    rings_signal: Signal = .unknown,
    /// The link watch: how many peer links it watches (0: none).
    watch_links: u32 = 0,
};

pub var facts: Facts = .{};
