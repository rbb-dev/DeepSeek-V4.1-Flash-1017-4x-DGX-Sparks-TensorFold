#!/usr/bin/env python3
"""draft_dump_cmp.py TP2_DIR FOUR_DIR: one drafter pass's dump (TF_DS_DRAFT_DUMP, written by node 0) of the TP2
drafter against the same pass's dump of the drafter split four ways (TF_DS_DRAFT_4WAY=1), file by file.

Four ways, node 0 (TP2 rank 0, pair 0) holds the first half of its TP2 rank's heads, its pair's two wo_a groups (the
rank's first two) and the first half of wo_b's and down's output columns. So its per-head files (q, o_rope, o: bf16
[R, heads, head_dim]) are compared with the TP2 node's first heads, u (bf16 [R, groups x o_lora]) with its first
columns, and the partials attn_out / moe_out (fp32, written as [R, hidden] either way; four ways the node's columns
fill the first R x hidden / 2 floats, row stride hidden / 2) with its first half of each row. Every other file is
compared whole. The dumps must hold one pass of one composition: meta.txt (streams, steps, the first stream's slot,
q0 and token) and the absorbed rows (taps, tpos) equal, else nothing else is compared.

One line a file ("equal", or how many bytes differ), then the verdict. Exit 0 when every file compared is equal, 1
otherwise, 2 for a usage problem. Standard library only."""
import os
import sys

PER_HEAD = ("q", "o_rope", "o")
PER_COLUMN = ("u",)
PARTIALS = ("attn_out", "moe_out")


def meta(d):
    m = {}
    for line in open(os.path.join(d, "meta.txt")):
        k, v = line.split()
        m[k] = int(v)
    return m


def first_part(two, four, rows):
    """The first len(four) / rows bytes of each of `rows` rows of `two` (its row twice as long), joined."""
    if rows == 0 or len(two) % rows or len(four) % rows:
        return None
    w2, w4 = len(two) // rows, len(four) // rows
    if w2 != 2 * w4:
        return None
    return b"".join(two[r * w2:r * w2 + w4] for r in range(rows))


def differ(a, b):
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tp2, four = sys.argv[1], sys.argv[2]
    ok = True

    def say(name, n, total, how=""):
        nonlocal ok
        if n:
            ok = False
        print(f"{name}{how}: {'equal' if n == 0 else f'{n} of {total} bytes differ'}")

    m2, m4 = meta(tp2), meta(four)
    if m2 != m4:
        print(f"meta.txt: the passes differ ({m2} against {m4}): not one pass of one composition, nothing compared")
        sys.exit(1)
    print("meta.txt: equal (" + ", ".join(f"{k} {v}" for k, v in m2.items()) + ")")
    for name in ("taps-0.bin", "tpos-0.bin"):
        a, b = open(os.path.join(tp2, name), "rb").read(), open(os.path.join(four, name), "rb").read()
        if a != b:
            print(f"{name}: {differ(a, b)} of {len(a)} bytes differ: the passes absorbed other rows, nothing else compared")
            sys.exit(1)
        print(f"{name}: equal")
    rows = m2["N"] * m2["n"]
    hidden = None
    names = sorted(set(os.listdir(tp2)) | set(os.listdir(four)))
    for name in names:
        if name in ("meta.txt", "taps-0.bin", "tpos-0.bin") or not name.endswith(".bin"):
            continue
        p2, p4 = os.path.join(tp2, name), os.path.join(four, name)
        if not (os.path.exists(p2) and os.path.exists(p4)):
            ok = False
            print(f"{name}: only in {'the TP2 dump' if os.path.exists(p2) else 'the four-way dump'}")
            continue
        a, b = open(p2, "rb").read(), open(p4, "rb").read()
        kind = name.rsplit("-", 1)[0]
        if kind in PER_HEAD or kind in PER_COLUMN:
            part = first_part(a, b, rows)
            if part is None:
                ok = False
                print(f"{name}: {len(b)} bytes four ways against {len(a)}: not half a row of {rows} rows")
                continue
            say(name, differ(part, b), len(b), " (the first heads)" if kind in PER_HEAD else " (the first columns)")
        elif kind in PARTIALS:
            if hidden is None:
                hidden = len(a) // (rows * 4)
            half = rows * (hidden // 2) * 4
            two = b"".join(a[r * hidden * 4:r * hidden * 4 + hidden * 2] for r in range(rows))
            say(name, differ(two, b[:half]), half, " (the first half of each row's columns)")
        else:
            say(name, differ(a, b), len(a))
    if ok:
        print("draft_dump_cmp: every file of the pass equal (the node's heads and columns where four ways holds a part)")
        sys.exit(0)
    print("draft_dump_cmp: FAILED (the lines above)")
    sys.exit(1)


if __name__ == "__main__":
    main()
