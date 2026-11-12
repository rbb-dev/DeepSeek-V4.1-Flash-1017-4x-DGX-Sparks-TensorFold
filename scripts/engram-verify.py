#!/usr/bin/env python3
"""Engram's packed shards (engram_pack.zig) against the tables they are cut from, standard library
only. A shard is one file a layer and node: a 4 KiB header (magic TFENGRM1, version 1, layer, the table's rows, the
node's rows [lo, hi), the row widths), then one record a row of [lo, hi): the row's FP8 values, then its scales. Node g
of n holds rows [rows * g / n, rows * (g + 1) / n); n = 1 is the whole table.

Run as a file here, or on a Spark that holds no copy of this recipe from stdin: python3 -I - ARGS < engram-verify.py.
A path that starts with $HOME/ is that Spark's home.

  plan NODE NODES TABLE...
      each shard node NODE of NODES reads, from the tables' own headers (every *.safetensors given; the checkpoint's
      Engram files also hold other tensors): "<file> <bytes> <layer> <rows> <lo> <hi> <row_w> <row_s>", by layer
  verify NODE NODES DIR TABLE... [--mbps M] [--manifest FILE] [--ok FILE] [--host-dir D] [--run-rows N]
      every shard of the node in DIR, against the tables: its size, every header field, then every record, in order,
      equal to its table row's values and scales; each file's sha256 in the same read. Prints one JSON line a shard
      (file, its path on its Spark, its stat key, sha256, bytes). --manifest merges "sha256  file" lines into FILE,
      --ok merges the verified lines into FILE (what `check` trusts); --host-dir names DIR's path on its Spark when
      this runs in a container. Reads are dropped from the page cache behind them and paced at M MB/s (0: unpaced).
  check DIR OKFILE MODEL NODES [--hash] [--source FILE=BLOB ...] SPEC...
      whether a Spark's shards (SPEC: plan's fields joined by ':') may be read as they are: each file at its exact size
      and with its header, MANIFEST.sha256 and PACKED.json beside them (PACKED.json from MODEL, for NODES nodes, from
      these source blobs when given), and each file's line in OKFILE (path, stat key, sha256) equal to the file and to
      the manifest. --hash: a file without such a line is read once and its line written when its sha256 is the
      manifest's.
  record DIR MODEL NODES IMAGE [--source FILE=BLOB ...] SPEC...
      PACKED.json in DIR: the checkpoint, its Engram files' blobs, and for NODES nodes the image, the time and each
      shard's size, merged with the sets already there (a folder holds one checkpoint's shards)
  merge [--manifest FILE] [--ok FILE] < JSON lines
      verify's lines into a manifest and an OKFILE

Exit: 0 done, 1 a defect (named: the file, the row, the byte), 2 a usage error, 3 (check) a shard missing or of
another size.
"""
import datetime
import hashlib
import json
import os
import struct
import sys
import time

MAGIC = b"TFENGRM1"
VERSION = 1
HEADER = 4096
RUN_ROWS = 65536
MANIFEST = "MANIFEST.sha256"
PACKED = "PACKED.json"
SIZES = {"F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U8": 1, "I8": 1, "BF16": 2, "F16": 2, "F32": 4}
HEADER_MAX = 1 << 30


class Defect(Exception):
    """A shard, a table or a record that is not what the release reads (exit 1)."""


class Missing(Exception):
    """A shard missing or of another size (check: exit 3)."""


class Usage(Exception):
    """A usage error (exit 2)."""


def home(p):
    """A path's leading $HOME/ as this Spark's home."""
    if p.startswith("$HOME/"):
        return os.environ.get("HOME", "") + p[5:]
    return p


def say(msg):
    print("engram-verify: " + msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- the tables

class Table:
    """One layer's table: [rows, row_w] values and [rows, row_s] scales, by file and offset."""

    def __init__(self, layer, rows, w, s):
        self.layer, self.rows = layer, rows
        self.w_path, self.w_base, self.row_w = w
        self.s_path, self.s_base, self.row_s = s


def safetensors_header(path):
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise Defect(f"{path}: no safetensors header (shorter than 8 bytes)")
        (n,) = struct.unpack("<Q", head)
        if n > HEADER_MAX:
            raise Defect(f"{path}: a header of {n} bytes is no safetensors header")
        text = f.read(n)
    if len(text) != n:
        raise Defect(f"{path}: its header ends early ({len(text)} of {n} bytes)")
    try:
        h = json.loads(text)
    except ValueError as e:
        raise Defect(f"{path}: its header is no JSON: {e}") from None
    if not isinstance(h, dict):
        raise Defect(f"{path}: its header is no JSON object")
    return n, h


def layer_of(name):
    """layers.<id>.engram.embed.* (a model. prefix allowed), as the engine reads it."""
    at = name.find("layers.")
    rest = name[at + 7:] if at >= 0 else ""
    head = rest.split(".", 1)[0]
    if at < 0 or not head.isdigit():
        raise Defect(f"{name}: no layer number after 'layers.'")
    return int(head)


def tables(paths):
    """Every Engram layer's table in the given files: each tensor whose name ends engram.embed.weight (the values) or
    engram.embed.scale (the scales), as the engine's Tables.open takes them."""
    halves = {}
    for p in paths:
        p = home(p)
        n, h = safetensors_header(p)
        for name, e in h.items():
            if name.endswith("engram.embed.weight"):
                which = 0
            elif name.endswith("engram.embed.scale"):
                which = 1
            else:
                continue
            layer = layer_of(name)
            if not isinstance(e, dict):
                raise Defect(f"{p}: {name}: not a tensor entry")
            dtype, shape, offs = e.get("dtype"), e.get("shape"), e.get("data_offsets")
            size = SIZES.get(dtype)
            if size is None:
                raise Defect(f"{p}: {name}: dtype {dtype} is not one the engine reads")
            if (not isinstance(shape, list) or len(shape) != 2 or not isinstance(offs, list) or len(offs) != 2
                    or not all(isinstance(x, int) and x >= 0 for x in shape + offs)):
                raise Defect(f"{p}: {name}: not a 2-d tensor with data offsets")
            rows, cols = shape
            if offs[1] - offs[0] != rows * cols * size:
                raise Defect(f"{p}: {name}: {offs[1] - offs[0]} data bytes for [{rows}, {cols}] {dtype}")
            slot = halves.setdefault(layer, [None, None])
            if slot[which] is not None:
                raise Defect(f"{name}: layer {layer}'s {('values', 'scales')[which]} twice ({slot[which][0]}, {p})")
            slot[which] = (p, 8 + n + offs[0], cols * size, rows)
    out = {}
    for layer, (w, s) in sorted(halves.items()):
        if w is None or s is None:
            raise Defect(f"layer {layer}: its table's {'values' if w is None else 'scales'} are in none of {list(paths)}")
        if w[3] != s[3]:
            raise Defect(f"layer {layer}: {w[3]} rows of values, {s[3]} rows of scales")
        out[layer] = Table(layer, w[3], w[:3], s[:3])
    if not out:
        raise Defect(f"no Engram table in {list(paths)}")
    return out


class Shard:
    """One shard a node reads: its file name, size, and header fields."""

    FIELDS = ("name", "bytes", "layer", "rows", "lo", "hi", "row_w", "row_s")

    def __init__(self, name, nbytes, layer, rows, lo, hi, row_w, row_s):
        self.name, self.bytes, self.layer, self.rows = name, nbytes, layer, rows
        self.lo, self.hi, self.row_w, self.row_s = lo, hi, row_w, row_s

    def spec(self):
        return ":".join(str(getattr(self, f)) for f in self.FIELDS)

    @classmethod
    def parse(cls, text):
        f = text.split(":")
        if len(f) != 8 or not f[0] or not all(x.isdigit() for x in f[1:]):
            raise Usage(f"{text}: a shard is file:bytes:layer:rows:lo:hi:row_w:row_s")
        return cls(f[0], *(int(x) for x in f[1:]))


def plan(node, nodes, tabs):
    if nodes < 1 or not 0 <= node < nodes:
        raise Usage(f"node {node} of {nodes}: NODE is below NODES")
    out = []
    for layer, t in sorted(tabs.items()):
        lo, hi = t.rows * node // nodes, t.rows * (node + 1) // nodes
        out.append(Shard(f"engram-l{layer}-n{node}of{nodes}.tfe", HEADER + (hi - lo) * (t.row_w + t.row_s), layer,
                         t.rows, lo, hi, t.row_w, t.row_s))
    return out


# ---------------------------------------------------------------- reading

class Pacer:
    """At most `mbps` MB/s (10^6 bytes) over every byte taken, from the first; 0: unpaced."""

    def __init__(self, mbps, clock=time.monotonic, sleep=time.sleep):
        self.rate = float(mbps) * 1e6
        self.clock, self.sleep = clock, sleep
        self.t0 = None
        self.done = 0

    def take(self, n):
        if self.rate <= 0:
            return
        if self.t0 is None:
            self.t0 = self.clock()
        self.done += n
        wait = self.t0 + self.done / self.rate - self.clock()
        if wait > 0:
            self.sleep(wait)


def drop(fd, off, n):
    advise = getattr(os, "posix_fadvise", None)
    if advise is not None:
        try:
            advise(fd, off, n, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def pread_exact(fd, n, off, pacer, path):
    parts, got = [], 0
    while got < n:
        b = os.pread(fd, min(n - got, 1 << 30), off + got)
        if not b:
            raise Defect(f"{path}: ends at byte {off + got}, before the {n} bytes at {off}")
        parts.append(b)
        got += len(b)
    drop(fd, off, n)
    pacer.take(n)
    return parts[0] if len(parts) == 1 else b"".join(parts)


def stat_key(path):
    s = os.stat(path)
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev]


def header_fields(b):
    """(magic, version, layer, rows, lo, hi, row_w, row_s, the rest all zero)"""
    version, layer, rows, lo, hi, row_w, row_s = struct.unpack_from("<IIQQQII", b, 8)
    return b[:8], version, layer, rows, lo, hi, row_w, row_s, not any(b[48:HEADER])


def check_header(path, b, sh):
    magic, version, layer, rows, lo, hi, row_w, row_s, clear = header_fields(b)
    if magic != MAGIC:
        raise Defect(f"{path}: no packed shard (magic {magic!r}, not {MAGIC!r})")
    if version != VERSION:
        raise Defect(f"{path}: header version {version}; this release reads version {VERSION}")
    want = (sh.layer, sh.rows, sh.lo, sh.hi, sh.row_w, sh.row_s)
    have = (layer, rows, lo, hi, row_w, row_s)
    if have != want:
        raise Defect(f"{path}: holds layer {layer} rows [{lo}, {hi}) of {rows} ({row_w} + {row_s} B a row); this node reads "
                     f"layer {sh.layer} rows [{sh.lo}, {sh.hi}) of {sh.rows} ({sh.row_w} + {sh.row_s} B a row)")
    if not clear:
        raise Defect(f"{path}: its header's bytes 48-4095 are not all zero")


def word(row_w, row_s):
    """The widest unsigned word both row widths are whole numbers of: (bytes, memoryview format)."""
    for u, f in ((8, "Q"), (4, "I"), (2, "H")):
        if row_w % u == 0 and row_s % u == 0:
            return u, f
    return 1, "B"


def first_difference(r, w, s, row_w, row_s):
    """The first (row in the run, byte in its record) where the records differ from the table's rows, or None."""
    u, f = word(row_w, row_s)
    rec_u, w_u, s_u = (row_w + row_s) // u, row_w // u, row_s // u
    rv, wv, sv = memoryview(r).cast(f), memoryview(w).cast(f), memoryview(s).cast(f)
    same = all(rv[j::rec_u] == wv[j::w_u] for j in range(w_u)) and \
        all(rv[w_u + k::rec_u] == sv[k::s_u] for k in range(s_u))
    if same:
        return None
    rec = row_w + row_s
    for i in range(len(r) // rec):
        want = w[i * row_w:(i + 1) * row_w] + s[i * row_s:(i + 1) * row_s]
        got = r[i * rec:(i + 1) * rec]
        if got != want:
            j = next(k for k in range(rec) if got[k] != want[k])
            return i, j, got[j], want[j]
    raise AssertionError("the strided compare found a difference the record compare did not")


def verify_one(path, sh, t, pacer, run_rows):
    """One shard against its table: (stat key, sha256) or a Defect."""
    try:
        key = stat_key(path)
    except FileNotFoundError:
        raise Defect(f"{path}: missing") from None
    if key[0] != sh.bytes:
        raise Defect(f"{path}: {key[0]} bytes; layer {sh.layer} rows [{sh.lo}, {sh.hi}) take {sh.bytes}")
    h = hashlib.sha256()
    rec = sh.row_w + sh.row_s
    fds = []
    try:
        fd = os.open(path, os.O_RDONLY)
        fds.append(fd)
        fw = os.open(t.w_path, os.O_RDONLY)
        fds.append(fw)
        fs = os.open(t.s_path, os.O_RDONLY)
        fds.append(fs)
        hb = pread_exact(fd, HEADER, 0, pacer, path)
        check_header(path, hb, sh)
        h.update(hb)
        at = sh.lo
        while at < sh.hi:
            n = min(run_rows, sh.hi - at)
            w = pread_exact(fw, n * sh.row_w, t.w_base + at * sh.row_w, pacer, t.w_path)
            s = pread_exact(fs, n * sh.row_s, t.s_base + at * sh.row_s, pacer, t.s_path)
            r = pread_exact(fd, n * rec, HEADER + (at - sh.lo) * rec, pacer, path)
            h.update(r)
            bad = first_difference(r, w, s, sh.row_w, sh.row_s)
            if bad:
                i, j, got, want = bad
                part = f"value byte {j}" if j < sh.row_w else f"scale byte {j - sh.row_w}"
                raise Defect(f"{path}: row {at + i} (record {at + i - sh.lo}) differs from layer {sh.layer}'s table at its "
                             f"{part} (file offset {HEADER + (at + i - sh.lo) * rec + j}): {got:#04x}, the table "
                             f"{want:#04x}")
            at += n
    finally:
        for x in fds:
            os.close(x)
    if stat_key(path) != key:
        raise Defect(f"{path}: changed while it was read")
    return key, h.hexdigest()


# ---------------------------------------------------------------- the manifest, the verified lines, PACKED.json

def write_atomic(path, text):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_manifest(path):
    """{file: sha256} of a manifest ("sha256  file" lines), or None when there is none."""
    try:
        text = open(path).read()
    except FileNotFoundError:
        return None
    out = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdef" for c in parts[0]):
            raise Defect(f"{path}: line {n} is not 'sha256  file'")
        out[parts[1].strip().lstrip("*")] = parts[0]
    return out


def merge_manifest(path, lines):
    have = read_manifest(path) or {}
    have.update({x["file"]: x["sha256"] for x in lines})
    write_atomic(path, "".join(f"{have[k]}  {k}\n" for k in sorted(have)))


def read_ok(path):
    """{path: line} of an OKFILE (one JSON object a line: path, key, sha256); {} when there is none."""
    try:
        text = open(path).read()
    except FileNotFoundError:
        return {}
    out = {}
    for line in text.splitlines():
        try:
            x = json.loads(line)
            out[x["path"]] = x
        except (ValueError, KeyError, TypeError):
            continue
    return out


def merge_ok(path, lines):
    have = read_ok(path)
    for x in lines:
        have[x["path"]] = {"path": x["path"], "key": x["key"], "sha256": x["sha256"]}
    write_atomic(path, "".join(json.dumps(have[k]) + "\n" for k in sorted(have)))


def read_packed(path):
    try:
        with open(path) as f:
            p = json.load(f)
    except FileNotFoundError:
        return None
    except ValueError as e:
        raise Defect(f"{path}: not JSON: {e}") from None
    if not isinstance(p, dict):
        raise Defect(f"{path}: not a JSON object")
    return p


def sources_of(items):
    out = {}
    for s in items:
        f, eq, blob = s.partition("=")
        if not eq or not f or not blob:
            raise Usage(f"--source {s}: FILE=BLOB")
        out[f] = blob
    return out


# ---------------------------------------------------------------- the commands

def options(args, flags, values):
    """(positional, {option: value}, [repeated --source values]): flags take no value."""
    pos, opt, src = [], {}, []
    i = 0
    while i < len(args):
        a = args[i]
        if a in flags:
            opt[a] = True
        elif a in values or a == "--source":
            if i + 1 >= len(args):
                raise Usage(f"{a} needs a value")
            i += 1
            if a == "--source":
                src.append(args[i])
            else:
                opt[a] = args[i]
        elif a.startswith("--"):
            raise Usage(f"no option {a}")
        else:
            pos.append(a)
        i += 1
    return pos, opt, src


def number(text, what):
    try:
        v = float(text)
    except ValueError:
        raise Usage(f"{what} is a number, not {text}") from None
    if v < 0:
        raise Usage(f"{what} is at least 0, not {text}")
    return v


def count(text, what):
    if not text.isdigit():
        raise Usage(f"{what} is a whole number, not {text}")
    return int(text)


def cmd_plan(args):
    pos, _, _ = options(args, (), ())
    if len(pos) < 3:
        raise Usage("plan NODE NODES TABLE...")
    for sh in plan(count(pos[0], "NODE"), count(pos[1], "NODES"), tables(pos[2:])):
        print(" ".join(str(getattr(sh, f)) for f in Shard.FIELDS))
    return 0


def cmd_verify(args, clock=time.monotonic, sleep=time.sleep):
    pos, opt, _ = options(args, (), ("--mbps", "--manifest", "--ok", "--host-dir", "--run-rows"))
    if len(pos) < 4:
        raise Usage("verify NODE NODES DIR TABLE... [--mbps M] [--manifest FILE] [--ok FILE] [--host-dir D]")
    node, nodes, d = count(pos[0], "NODE"), count(pos[1], "NODES"), home(pos[2])
    tabs = tables(pos[3:])
    pacer = Pacer(number(opt.get("--mbps", "0"), "--mbps"), clock, sleep)
    run_rows = count(opt.get("--run-rows", str(RUN_ROWS)), "--run-rows") or RUN_ROWS
    host_dir = home(opt.get("--host-dir", d))
    lines = []
    for sh in plan(node, nodes, tabs):
        path = os.path.join(d, sh.name)
        t0 = time.monotonic()
        key, sha = verify_one(path, sh, tabs[sh.layer], pacer, run_rows)
        x = {"file": sh.name, "path": os.path.join(host_dir, sh.name), "key": key, "sha256": sha, "bytes": sh.bytes}
        lines.append(x)
        print(json.dumps(x), flush=True)
        say(f"{path}: {sh.hi - sh.lo} records equal to layer {sh.layer}'s table rows [{sh.lo}, {sh.hi}), sha256 "
            f"{sha[:16]}... ({time.monotonic() - t0:.0f} s)")
    if opt.get("--manifest"):
        merge_manifest(home(opt["--manifest"]), lines)
    if opt.get("--ok"):
        merge_ok(home(opt["--ok"]), lines)
    return 0


def sha256_file(path):
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        off = 0
        while True:
            b = os.pread(fd, 16 << 20, off)
            if not b:
                break
            h.update(b)
            drop(fd, off, len(b))
            off += len(b)
    finally:
        os.close(fd)
    return h.hexdigest()


def cmd_check(args):
    pos, opt, src = options(args, ("--hash",), ())
    if len(pos) < 5:
        raise Usage("check DIR OKFILE MODEL NODES [--hash] [--source FILE=BLOB ...] SPEC...")
    d, okfile, model, nodes = home(pos[0]), home(pos[1]), pos[2], count(pos[3], "NODES")
    specs = [Shard.parse(x) for x in pos[4:]]
    sources = sources_of(src)
    gone = []
    for sh in specs:
        p = os.path.join(d, sh.name)
        try:
            size = os.stat(p).st_size
        except FileNotFoundError:
            gone.append(f"{p}: missing")
            continue
        if size != sh.bytes:
            gone.append(f"{p}: {size} bytes, not {sh.bytes} (layer {sh.layer} rows [{sh.lo}, {sh.hi}) of {sh.rows})")
    if gone:
        raise Missing("; ".join(gone))
    manifest = read_manifest(os.path.join(d, MANIFEST))
    if manifest is None:
        raise Defect(f"packed shards without a manifest: {os.path.join(d, MANIFEST)} is missing (run "
                     "scripts/engram-pack.sh on rank 0; a worker gets its files, MANIFEST.sha256 and PACKED.json)")
    packed = read_packed(os.path.join(d, PACKED))
    if packed is None:
        raise Defect(f"packed shards without {os.path.join(d, PACKED)}: whose checkpoint they are cut from is not "
                     "known (run scripts/engram-pack.sh on rank 0)")
    if packed.get("format") != 1:
        raise Defect(f"{os.path.join(d, PACKED)}: format {packed.get('format')}, not 1")
    if packed.get("model") != model:
        raise Defect(f"{os.path.join(d, PACKED)}: cut from {packed.get('model')}; this release reads {model}")
    for f, blob in sorted(sources.items()):
        if (packed.get("sources") or {}).get(f) != blob:
            raise Defect(f"{os.path.join(d, PACKED)}: cut from {f} blob {(packed.get('sources') or {}).get(f)}; this "
                         f"release's checkpoint has blob {blob}")
    files = ((packed.get("sets") or {}).get(str(nodes)) or {}).get("files") or {}
    for sh in specs:
        if files.get(sh.name) != sh.bytes:
            raise Defect(f"{os.path.join(d, PACKED)}: lists {sh.name} at {files.get(sh.name)} bytes for {nodes} node(s), "
                         f"not {sh.bytes}")
    for sh in specs:
        p = os.path.join(d, sh.name)
        with open(p, "rb") as f:
            check_header(p, f.read(HEADER), sh)
    oks = read_ok(okfile)
    new, said, bad = [], [], []
    for sh in specs:
        p = os.path.join(d, sh.name)
        want = manifest.get(sh.name)
        if want is None:
            bad.append(f"{p}: no line in {MANIFEST}")
            continue
        line, key = oks.get(p), stat_key(p)
        if line and line.get("key") == key and line.get("sha256") == want:
            said.append(f"ok {sh.name}: verified ({okfile})")
            continue
        if not opt.get("--hash"):
            bad.append(f"{p}: not verified on this Spark as it is now (no line of it in {okfile}): prepare.sh would "
                       f"read it whole against {MANIFEST} ({sh.bytes / 1e9:.1f} GB)")
            continue
        t0 = time.monotonic()
        got = sha256_file(p)
        if got != want:
            bad.append(f"{p}: sha256 {got}, {MANIFEST} {want}: it differs from the shard that was verified")
            continue
        if stat_key(p) != key:
            bad.append(f"{p}: changed while it was hashed")
            continue
        new.append({"path": p, "key": key, "sha256": got})
        said.append(f"hashed {sh.name}: equal to {MANIFEST} ({time.monotonic() - t0:.0f} s)")
    if new:
        merge_ok(okfile, new)
    if bad:
        raise Defect("; ".join(bad))
    for s in said:
        print(s)
    return 0


def cmd_record(args):
    pos, _, src = options(args, (), ())
    if len(pos) < 5:
        raise Usage("record DIR MODEL NODES IMAGE [--source FILE=BLOB ...] SPEC...")
    d, model, nodes, image = home(pos[0]), pos[1], count(pos[2], "NODES"), pos[3]
    specs = [Shard.parse(x) for x in pos[4:]]
    sources = sources_of(src)
    path = os.path.join(d, PACKED)
    p = read_packed(path) or {"format": 1, "model": model, "sources": sources, "sets": {}}
    if p.get("model") != model:
        raise Defect(f"{path}: this folder holds shards cut from {p.get('model')}, not {model}: give "
                     "ENGRAM_PACKED_DIR a folder of its own")
    if sources and p.get("sources") and p["sources"] != sources:
        raise Defect(f"{path}: cut from the blobs {p['sources']}, not {sources}")
    p["format"], p["sources"] = 1, sources or p.get("sources") or {}
    sets = p.setdefault("sets", {})
    one = sets.setdefault(str(nodes), {"files": {}})
    one["image"] = image
    one["at"] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")
    one.setdefault("files", {}).update({sh.name: sh.bytes for sh in specs})
    write_atomic(path, json.dumps(p, indent=1, sort_keys=True) + "\n")
    return 0


def cmd_merge(args):
    _, opt, _ = options(args, (), ("--manifest", "--ok"))
    lines = [json.loads(x) for x in sys.stdin.read().splitlines() if x.strip()]
    if opt.get("--manifest"):
        merge_manifest(home(opt["--manifest"]), lines)
    if opt.get("--ok"):
        merge_ok(home(opt["--ok"]), lines)
    return 0


COMMANDS = {"plan": cmd_plan, "verify": cmd_verify, "check": cmd_check, "record": cmd_record, "merge": cmd_merge}


def main(argv):
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        return COMMANDS[argv[0]](argv[1:])
    except Usage as e:
        say(f"usage: {e}")
        return 2
    except Missing as e:
        print(f"missing: {e}")
        return 3
    except Defect as e:
        print(f"defect: {e}")
        return 1
    except OSError as e:
        print(f"defect: {e.filename or ''}: {e.strerror}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
