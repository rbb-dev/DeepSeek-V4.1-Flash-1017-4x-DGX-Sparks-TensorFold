#!/usr/bin/env python3
"""The drafter's pruned vocabulary (TF_DS_DRAFT_VOCAB): 32,768 token ids, 8,192 a node, frequency-balanced.

The list is the checkpoint's own special tokens the model writes (every added token but the placeholders) plus the ids
most frequent in a ranking text (ties to the lower id; if the text has seen fewer ids than there are slots, the lowest
unused ids fill the rest), sorted ascending. Node g takes positions [8,192 g, 8,192 (g + 1)): the same number of rows
on every node, the ids in vocabulary order, so a tie between nodes falls to the lower id as on the full vocabulary.

Inputs are pinned by sha256 (the tokenizer the engine's checkpoint ships, the frozen corpus); the output is u32
little-endian. Usage:
  draft_vocab.py TOKENIZER_JSON CORPUS_DIR OUT_BIN
Run with the tokenizers package (0.22.2) and `python3 -I`.
"""
import hashlib
import re
import struct
import sys
from collections import Counter

V = 129_280
NODES = 4
PER_NODE = 8_192
TOTAL = NODES * PER_NODE
TOKENIZER_SHA = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
CORPUS = {  # the frozen corpus: the project's own documents and the engine's Zig source
    "realtext-prose.txt": "2c9acb9a775c09bb1f3cc64397dbf963bce8e517e50c5aa84ee7c08db2757aff",
    "realtext-code.txt": "32336b9b52c6886ffb7b645ab137618c0b985f8db009cbc9af50c3334365b970",
}
PLACEHOLDER = re.compile(r"place.?holder", re.IGNORECASE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def forced_ids(tokenizer_json):
    """The added tokens the model writes: every one whose name is not a placeholder."""
    import json
    t = json.load(open(tokenizer_json, encoding="utf-8"))
    return sorted(a["id"] for a in t["added_tokens"] if not PLACEHOLDER.search(a["content"]) and a["id"] < V)


def build(counts, forced):
    """The list from id counts (a Counter) and the forced ids: sorted ascending, TOTAL ids."""
    keep = set(forced)
    ranked = sorted((i for i in counts if counts[i] > 0 and i < V), key=lambda i: (-counts[i], i))
    for i in ranked:
        if len(keep) == TOTAL:
            break
        keep.add(i)
    i = 0
    while len(keep) < TOTAL:  # the filler: the lowest unused ids
        if i not in keep:
            keep.add(i)
        i += 1
    return sorted(keep)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    tok_path, corpus_dir, out = sys.argv[1:]
    if sha256(tok_path) != TOKENIZER_SHA:
        sys.exit("draft_vocab: the tokenizer is not the checkpoint's (sha256 differs)")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    counts = Counter()
    for name, want in CORPUS.items():
        path = f"{corpus_dir}/{name}"
        if sha256(path) != want:
            sys.exit(f"draft_vocab: {name} is not the frozen corpus (sha256 differs)")
        text = open(path, encoding="utf-8", errors="replace").read()
        # the documents one at a time (the corpus's ===== name ===== headers), as the coverage measurements cut them
        for doc in re.split(r"^===== .* =====$", text, flags=re.M):
            if doc.strip():
                counts.update(tok.encode(doc, add_special_tokens=False).ids)
    forced = forced_ids(tok_path)
    ids = build(counts, forced)
    assert len(ids) == TOTAL and len(set(ids)) == TOTAL and ids == sorted(ids) and ids[-1] < V
    assert all(f in set(ids) for f in forced)
    blob = b"".join(struct.pack("<I", i) for i in ids)
    open(out, "wb").write(blob)
    n = sum(counts.values())
    kept = sum(counts[i] for i in ids)
    quarters = [0, 32_384, 64_640, 97_024, V]
    per_q = [sum(1 for i in ids if quarters[q] <= i < quarters[q + 1]) for q in range(4)]
    print(f"draft_vocab: {TOTAL} ids ({len(forced)} forced), sha256 {hashlib.sha256(blob).hexdigest()}")
    print(f"draft_vocab: ranking text {n} tokens, {sum(1 for c in counts.values() if c > 0)} distinct ids; "
          f"in sample the list covers {100 * kept / n:.3f}%")
    print(f"draft_vocab: ids by the vocabulary's quarters {per_q}; node g's ids {[ (ids[g * PER_NODE], ids[(g + 1) * PER_NODE - 1]) for g in range(NODES)]}")


if __name__ == "__main__":
    main()
