#!/usr/bin/env python3
"""Fetch and pre-cut the test data. Standard library only. Run it with `python3 -I prepare_data.py ...` (it reads
downloaded files; -I keeps the current directory off the import path).

  python3 -I prepare_data.py mrcr --lengths 1m,2m,4m [--rows 96]
  python3 -I prepare_data.py babilong
  data/venv-pa/bin/python -I prepare_data.py openai      (openai/mrcr; pyarrow needed for the parquet)

MRCR: Google DeepMind's MRCR v2 files (https://storage.googleapis.com/mrcr_v2/, 8 needles, the "dynamic_fewshot
text_style fast" CSVs; code and README: https://github.com/google-deepmind/eval_hub/tree/master/eval_hub/mrcr_v2).
Each CSV row is a whole prompt (queries) = ONE shared transcript + that row's final question, plus the expected answer.
The files hold many rows per transcript, and each is 5-20 MB, so this streams the CSV over HTTPS, keeps the transcript
ONCE (the first row's, checking that every later row begins with exactly those characters), keeps each row's final
question and answer, and stops after --rows rows. It never saves the 3-12 GB files. Output: data/mrcr/<len>.json.

BABILong: RMT-team/babilong's 0k files for qa2 and qa3 (the bAbI stories alone, with their questions and targets: plain
JSON) and the PG19 TEST books from DeepMind's own bucket (gs://deepmind-gutenberg/test, https, 100 books), which is the
background text BABILong is built from. Output: data/babilong/{qa2,qa3}_0k.json and data/babilong/pg19_test/*.txt.
"""
import argparse
import csv
import io
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

MRCR_BASE = "https://storage.googleapis.com/mrcr_v2"
# the published bins: (lo, hi] in Gemini tokens; the first transcript of a bin is built to the bin's upper end
MRCR_BINS = {"32k": "(16384,32768)", "256k": "(131072,262144)",
             "1m": "(524288,1048576)", "2m": "(1048576,2097152)", "4m": "(2097152,4194304)"}
MRCR_FILE = "mrcr_v2p1_{n}needle_in_{bin}_dynamic_fewshot_text_style_fast.csv"
# a length label is a bin name, with "-2n" / "-4n" for the 2- / 4-needle files ("1m-2n"); no suffix = 8 needles

BABI_BASE = "https://huggingface.co/datasets/RMT-team/babilong/resolve/main/data"
PG19_LIST = "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?prefix=test/&maxResults=1000"
PG19_FILE = "https://storage.googleapis.com/deepmind-gutenberg/test/{name}"

UA = {"User-Agent": "reason4m-data/1"}


def fetch(url, timeout=600):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def prepare_mrcr(length, rows_wanted):
    base, _, nn = length.partition("-")
    needles = int(nn[:-1]) if nn else 8
    if base not in MRCR_BINS or needles not in (2, 4, 8):
        sys.exit(f"MRCR lengths: {', '.join(MRCR_BINS)}, each optionally with -2n or -4n")
    name = MRCR_FILE.format(n=needles, bin=MRCR_BINS[base])
    url = MRCR_BASE + "/" + urllib.parse.quote(name)
    csv.field_size_limit(1 << 30)
    out_dir = os.path.join(DATA, "mrcr")
    os.makedirs(out_dir, exist_ok=True)
    transcript = None
    rows = []
    with fetch(url) as resp:
        reader = csv.reader(io.TextIOWrapper(resp, encoding="utf-8", newline=""))
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        for need in ("queries", "answer", "view_ops", "num_relevant", "context_len"):
            if need not in col:
                sys.exit(f"{name}: no column {need!r}; header {header}")
        for row in reader:
            if len(row) != len(header):
                break   # the stream ended inside a row
            q, final = row[col["queries"]], row[col["view_ops"]]
            if not q.endswith(final):
                sys.exit("a row's queries does not end with its view_ops: the file format changed")
            tr = q[: len(q) - len(final)]
            if transcript is None:
                transcript = tr
            elif tr != transcript:
                print(f"row {len(rows)} starts a different transcript: stopping", flush=True)
                break
            rows.append({
                "final_query": final,
                "answer": row[col["answer"]],
                "random_string": row[col["answer"]][:12],
                "num_relevant": int(row[col["num_relevant"]]),
                "gemini_context_len": int(row[col["context_len"]]),
                "answer_context_position": row[col["answer_context_position"]] if "answer_context_position" in col else "",
            })
            if len(rows) % 8 == 0:
                print(f"{length}: {len(rows)} rows", flush=True)
            if len(rows) >= rows_wanted:
                break
    if transcript is None or not rows:
        sys.exit("no rows read")
    doc = {"source": url, "bin": MRCR_BINS[base], "n_needles": rows[0]["num_relevant"],
           "transcript": transcript, "rows": rows}
    path = os.path.join(out_dir, f"{length}.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    print(f"wrote {path}: transcript {len(transcript):,} chars, {len(rows)} rows", flush=True)


def prepare_openai(needles, rows_wanted):
    """OpenAI's openai/mrcr, the set DeepSeek's report cites. Needs pyarrow to read the parquet (not in the system
    python): run this one step with the venv python, `data/venv-pa/bin/python -I prepare_data.py openai`
    (python3 -m venv data/venv-pa && data/venv-pa/bin/pip install pyarrow). Keeps the largest bin (the 100 longest
    samples of the second file: 524,288-1,048,576 tokens by o200k_base, prompt + answer) as JSON lines."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is needed for this step: python3 -m venv data/venv-pa && data/venv-pa/bin/pip install pyarrow, "
                 "then data/venv-pa/bin/python -I prepare_data.py openai")
    import random
    out_dir = os.path.join(DATA, "openai")
    os.makedirs(out_dir, exist_ok=True)
    for n in needles:
        pq_path = os.path.join(out_dir, f"{n}needle_1.parquet")
        if not os.path.exists(pq_path):
            url = f"https://huggingface.co/datasets/openai/mrcr/resolve/main/{n}needle/{n}needle_1.parquet"
            with fetch(url) as r, open(pq_path, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        rows = pq.ParquetFile(pq_path).read().to_pylist()
        rows.sort(key=lambda r: -r["n_chars"])
        top = rows[:100]       # the largest bin: 100 samples per bin
        random.Random(1).shuffle(top)
        top = [r for r in top if r["n_chars"] <= 4_950_000][:rows_wanted]
        path = os.path.join(out_dir, f"{n}needle-1m.jsonl")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for i, r in enumerate(top):
                msgs = json.loads(r["prompt"])
                f.write(json.dumps({"id": f"o{n}-{i}", "messages": msgs, "answer": r["answer"],
                                    "random_string": r["random_string_to_prepend"], "n_needles": r["n_needles"],
                                    "n_chars": r["n_chars"], "desired_msg_index": r["desired_msg_index"],
                                    "total_messages": r["total_messages"], "date_added": str(r["date_added"])},
                                   ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
        print(f"wrote {path}: {len(top)} samples, {min(r['n_chars'] for r in top):,}-{max(r['n_chars'] for r in top):,} chars",
              flush=True)


def prepare_babilong():
    out = os.path.join(DATA, "babilong")
    os.makedirs(os.path.join(out, "pg19_test"), exist_ok=True)
    for task in ("qa2", "qa3"):
        path = os.path.join(out, f"{task}_0k.json")
        if not os.path.exists(path):
            with fetch(f"{BABI_BASE}/{task}/0k.json") as r, open(path, "wb") as f:
                f.write(r.read())
        print("have", path, flush=True)
    with fetch(PG19_LIST) as r:
        listing = json.load(r)
    names = [it["name"].split("/", 1)[1] for it in listing.get("items", []) if it["name"].endswith(".txt")]
    if listing.get("nextPageToken"):
        print("note: more than 1000 objects listed; using the first 1000", flush=True)
    for n in sorted(names):
        path = os.path.join(out, "pg19_test", n)
        if os.path.exists(path):
            continue
        with fetch(PG19_FILE.format(name=n)) as r, open(path, "wb") as f:
            f.write(r.read())
    print(f"have {len(names)} PG19 test books", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    m = sub.add_parser("mrcr")
    m.add_argument("--lengths", default="32k,256k,1m,2m,4m,1m-2n")
    m.add_argument("--rows", type=int, default=96, help="rows to keep per length (8 per needle group)")
    sub.add_parser("babilong")
    o = sub.add_parser("openai", help="openai/mrcr largest bin (needs pyarrow: run with data/venv-pa/bin/python)")
    o.add_argument("--needles", default="8,2")
    o.add_argument("--rows", type=int, default=40)
    a = p.parse_args()
    if a.what == "mrcr":
        for ln in a.lengths.split(","):
            prepare_mrcr(ln.strip().lower(), a.rows)
    elif a.what == "openai":
        prepare_openai([int(x) for x in a.needles.split(",")], a.rows)
    else:
        prepare_babilong()


if __name__ == "__main__":
    main()
