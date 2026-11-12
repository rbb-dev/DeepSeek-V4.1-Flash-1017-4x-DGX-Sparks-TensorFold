# fabric-xfer: the setup's copy over both rails

`./setup.sh` uses it to put each worker's share on the other three Sparks: RDMA over both fabric rails, written in C
against libibverbs, with nothing to install on a DGX Spark (gcc, make and libibverbs-dev come with DGX OS 7.6). Where it
cannot be built, or its `selftest` finds no rails, the setup copies with `scripts/fabric-copy.sh` (TCP) instead.

## What it does

- **Reads each file once.** The sender reads 16 MiB chunks with `O_DIRECT` into a ring of memory registered with the network card, and RDMA-writes each
  chunk to every receiver that needs it, alternating the two rails. Files that go to different receivers (each worker's
  own weight file) are read interleaved, chunk 0 of every file, then chunk 1, so every receiver's disk writes at once.
- **Flow control by credits.** A receiver writes each ring slot to disk on its own threads and returns the slot (a
  one-byte send with the slot number as immediate) only once it is on disk; the sender never writes into a slot it holds
  no credit for, so a slow disk paces its own receiver only.
- **Checks every chunk.** The sender computes a CRC-32C checksum of each chunk from the bytes it read. Each receiver
  computes one from the bytes that landed in its ring (ARMv8 CRC32C instructions, checked against a table implementation
  before they are trusted).
  A file is renamed from `name.partial` into place, with its mode and modification time, only when every chunk agrees.
- **Sends only what is missing.** A file a receiver already holds at the sender's size is skipped; `--verify-existing`
  also compares the chunk checksums of copies in place.
- **One run for every worker's share.** A line of the path list is `PATH` (to every target) or `N[,N...]<TAB>PATH`
  (only to those targets, 1 being the first). One run means one sender and one ring; two runs at once would each lock a
  ring in memory and compete for it.
- **Progress as written.** Once a second (`--progress-s`), the sender prints how many bytes are on the receivers' disks out of
  the total, the per cent done, the rate and the time left, each receiver's share, and the rate on each rail.

## Safety

- A receiver accepts one connection, only from the sender's addresses (`--allow`), and only after a challenge: the
  wrapper makes a one-time token, passes it on ssh's stdin and the sender's fd 3 (never on a command line), and the
  sender answers the receiver's random nonce with HMAC-SHA256 over it.
- Every path is relative, without `.` or `..` segments or control characters; the receiver opens everything beneath its
  destination folder (`openat2` with `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`) and masks mode bits to `0777` (no setuid, setgid or sticky bits).
- The slot, chunk number and length of every write are checked against the receiver's ring and the plan.
- The data itself is not encrypted on the fabric.

## Memory

The ring is at most 8 GiB and at most half of `MemAvailable`; it is a quarter of `MemAvailable` when a container named like
an engine runs (`dsv41`, `tensorfold` or `glm53`; `--engine-pattern` sets the pattern) or less than 16 GiB is available.
`--ring-gib N` sets the ring directly, up to 80 % of `MemAvailable`. A few GiB already keep a disk busy for seconds;
a larger ring takes longer to pin and only lengthens the wait for the disks at the end.

## Command line

```
fabric-xfer.sh [-n THREADS] [-p PORT] [-l LABEL] [-s] SSH_TARGETS ADDRESSES SRC_DIR DEST_DIR < paths
```

- `SSH_TARGETS`: the workers, comma separated, as `ssh` reaches them (their LAN addresses: `user@192.0.2.2,user@192.0.2.3`).
- `ADDRESSES`: each worker's fabric addresses, rail A first, workers separated by `;`
  (fabric addresses: `10.255.0.2,10.255.1.2;10.255.0.3,10.255.1.3`).
- `paths`: one a line, relative to `SRC_DIR`, optionally routed as above; each lands at the same path under `DEST_DIR`
  (an absolute path, or one under a literal `$HOME/`).
- `-n`: disk threads on each side (8; 1 to 99). `-p`: the control port on each worker (5400; 1024 to 65535). `-s`: print the bytes that would
  be sent. `FX_OPTS` passes options to the sender and the receivers (`--ring-gib`, `--chunk-mib`, `--no-direct`,
  `--readback`, `--verify-existing`, `--gid-index`, `--engine-pattern`, `--progress-s`), `FX_SEND_OPTS` to the sender only (`--manifest PATH`).

The wrapper pipes the binary to each worker over ssh, runs it there and removes it; nothing is installed on the workers.

## Build and test

```
make            # the binary, on a Spark (libibverbs)
make test       # the core, engine and wrapper tests; no RDMA device needed
./fabric-xfer selftest --addr 10.255.0.1 --addr 10.255.1.1    # this Spark's rail A and rail B addresses: device, port,
                                                               # GID, CRC speed and memory
```

The engine tests run the real sender and receivers over an in-process wire: every tail size, more chunks than slots,
three receivers in different states (each needed chunk read once), routed files, corruption in flight, a bad disk write,
hostile paths, symlinks in the destination, wrong tokens and hostile immediate values.

## Measured

On four DGX Sparks (one ConnectX-7 port each, two rails, a RoCE switch with priority flow control), engine stopped:

| What | Data | Time |
|---|---|---|
| each worker's own 80 GB weight file, three at once from one run | 241 GB | 44.5 s (5.41 GB/s) |
| Engram's packed shards to all three workers | 405.5 GB read once, 1,216.5 GB written | 207 s (1.96 GB/s read, 5.88 GB/s written in all) |

Every chunk was checked by CRC-32C on both ends. In these runs every file was also checked by SHA-256 on both ends,
outside the tool (`sha256sum`).
