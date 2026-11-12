#!/usr/bin/env python3
"""kit_fixture.py KIT_DIR OUT_DIR: the kit's variants of the drafter stage's attention launches and the twins, for
draft.zig's test that a four-way stage's launches resolve in the kit, and tri_attn.zig's and tri_index.zig's that every
_sparse_attn_part and _index_score_rows variant has its twin.

For each of _rope_heads, _kv_norm_rope, _sparse_attn_part, _sparse_attn_merge, _sparse_attn_part_u, _index_score_rows,
_index_score_rows_u and _index_score_rows_v: every entry of
KIT_DIR/aot/aot.json into OUT_DIR/kit-drafter.json and every entry of KIT_DIR/aot-extra/aot.json (none when the kit has
no extra set) into OUT_DIR/kit-drafter-extra.json, in the set's order, with the fields aot.zig matches (fn, hash, name,
num_warps, params, consts): each file an aot.json of its own. Standard library only."""
import json
import os
import sys

FUNCTIONS = ("_rope_heads", "_kv_norm_rope", "_sparse_attn_part", "_sparse_attn_merge", "_sparse_attn_part_u",
             "_index_score_rows", "_index_score_rows_u", "_index_score_rows_v")
KEEP = ("fn", "hash", "name", "num_warps", "params", "consts")


    if not os.path.exists(path):
        return []
    out = []
    for k in json.load(open(path))["kernels"]:
            continue
        if k.get("global_scratch", 0) or k.get("profile_scratch", 0):
            sys.exit(f"{path}: {k['fn']} {k['hash']} needs scratch, which the engine refuses")
        out.append({f: k[f] for f in KEEP})
    return out


def write(path, items):
    # one entry a line, so a kit change reads as the lines it changes
    with open(path, "w") as f:
        f.write('{"kernels": [\n' + ",\n".join(json.dumps(k, sort_keys=True, separators=(",", ":")) for k in items) + "\n]}\n")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    kit, out = sys.argv[1], sys.argv[2]
    base = entries(os.path.join(kit, "aot", "aot.json"))
    extra = entries(os.path.join(kit, "aot-extra", "aot.json"))
    if not base:
        sys.exit(f"{kit}/aot/aot.json: none of {', '.join(FUNCTIONS)}")
    write(os.path.join(out, "kit-drafter.json"), base)
    write(os.path.join(out, "kit-drafter-extra.json"), extra)
    print(f"{out}: {len(base)} base entries, {len(extra)} extra")


if __name__ == "__main__":
    main()
