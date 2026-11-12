#!/usr/bin/env python3
"""Seeds zig/config-schema.json from the engine's source and a parameters file (--params PATH, params-ds.json).

It is the one-time migration reader: every environment name the engine reads gets a row, with an id that is then
frozen in the JSON (a later run keeps every existing row as it is and appends only the names that have none, so ids
are never renumbered or reused). What the source cannot tell stays marked: a row's "seed" says whether its type came
from params-ds.json ("params") or from its name ("name"), and "default_check" says whether a test compares its default
with the parser's ("test") or not yet ("none").

usage: seed.py [--params PATH]   (run from the engine's root)
"""
import glob
import json
import os
import re
import sys

SCHEMA = "zig/config-schema.json"
SRC = "zig/src"

# group -> (first id, tab). Ids partitioned for readability, never for meaning (API.md section 3).
GROUPS = {
    "serving": (1, "Requests & keys"),
    "prompt": (100, "Passes & memory"),
    "decode": (200, "Generation"),
    "transport": (300, "Transport"),
    "cache": (400, "Cache & SSD"),
    "memory": (500, "Passes & memory"),
    "other": (600, "Advanced"),
    "glm": (700, "Advanced"),
    "debug": (900, "Advanced"),
}

# env literals that are not settings (error-code names, a name prefix used to build others), and names this engine never
# reads although its code holds them: a test's check that a name is NOT read (TF_YS), a hook only a -Dslide-test-hooks
# build reads (TF_SLIDE_FORGET_FAIL), and the base engine's GID reader rdma.rdmaGid, which nothing calls (TF_RDMA_GID;
# this engine's rings take TF_DS_RDMA_GID)
NOT_SETTINGS = {"CUDA_ERROR_STAND_IN", "CUDA_ERROR_UNKNOWN_CODE", "TF_SPILL_", "TF_DS_SPILL_", "TF_DS_",
                "TF_YS", "TF_SLIDE_FORGET_FAIL", "TF_RDMA_GID"}

# code the DeepSeek engine does not run, which the base engine carries: a name read only there needs no row (a name also
# read in this engine's code is found there). Paths under zig/src; a trailing "/" is a folder, anything else a prefix.
NOT_THIS_ENGINE = (
    "xpu/", "hip/",                                                    # the Intel and AMD GPU backends
    "families/glm/", "families/nemotron/", "families/qwen27/", "families/flashnext/",   # other models
    "native/flashnext_host.zig", "native/qwen27_", "native/xpu",       # other models' hosts
    "fabric/mcdma.zig",                                                # the Macs' Thunderbolt fabric
    "core/heat.zig",                                                   # GLM's expert heat (GLM is its only importer)
    "core/bf16_topk_gpu.zig",                                          # Qwen's GPU top-k (Qwen is its only importer)
    "core/rows_",                                                      # the rows store (nothing outside it imports it)
    "cli/cuda_main.zig", "cuda/tp_watch.zig",                          # the `tensorfold` run-and-oracle tool, its NCCL watch
    "server/benchmark_",                                               # the benchmark host and server
    "families/dsv41/split_bench_main.zig",                             # a microbenchmark program
)

NAME_RE = r'"((?:TF_|TENSORFOLD_|GLM_|FZ_|MCDMA_|NCCL_|CUDA_)[A-Z0-9_]*)"'
CALL_RE = r'(?:getenv|environ_map\.get|env\(s,|envOn|envInt|envFloat)\(\s*"([A-Z][A-Z0-9_]*)"'


def is_source(path):
    # the generated table names every option: it is the schema, not a reader of it
    return not (path == "core/settings_table.zig" or path.endswith("_test.zig") or "/tests/" in path or path.startswith("core/template/tests")
                or path.startswith(NOT_THIS_ENGINE))


def scan():
    """name -> (file, line) of its first use."""
    found = {}
    for p in sorted(glob.glob("**/*.zig", root_dir=SRC, recursive=True)):
        if not is_source(p):
            continue
        for ln, line in enumerate(open(os.path.join(SRC, p)), 1):
            if line.strip().startswith("//"):
                continue
            code = line
            for m in list(re.finditer(NAME_RE, code)) + list(re.finditer(CALL_RE, code)):
                n = m.group(1)
                if n not in NOT_SETTINGS and n not in found:
                    found[n] = (p, ln)
    return found


def doc_at(path, ln):
    """The doc comment above the first use, or the trailing comment on it."""
    lines = open(os.path.join(SRC, path)).read().split("\n")
    here = lines[ln - 1]
    out = []
    i = ln - 2
    while i >= 0 and lines[i].strip().startswith("//"):
        out.insert(0, lines[i].strip().lstrip("/! ").strip())
        i -= 1
    if out:
        return " ".join(out)[:300]
    m = re.search(r'//\s*(.*)$', here)
    return m.group(1)[:300] if m else ""


def group_of(n):
    if n.startswith("GLM_") or n.startswith("FZ_"):
        return "glm"
    if n.startswith(("TF_SPILL_", "TF_DS_SPILL_", "TF_DS_IMAGE_CACHE", "TF_DS_ENGRAM", "TF_DS_TOKEN_MAP", "TF_DS_RANK_CACHE")):
        return "cache"
    if n.startswith(("TENSORFOLD_API", "TENSORFOLD_CLOSE", "TENSORFOLD_LOG", "TENSORFOLD_NO_", "TENSORFOLD_REQUEST", "TENSORFOLD_SEED", "TF_SSE", "TF_VIDEO", "TF_GRAMMAR", "TENSORFOLD_GRAMMAR", "TENSORFOLD_PREPARE")):
        return "serving"
    if n.startswith(("TENSORFOLD_MEMORY", "TENSORFOLD_CUDA_MEMORY", "TF_DS_ARENA", "TF_DS_WINDOW", "TF_DS_POOL", "TF_DS_DIRECT_WEIGHTS")):
        return "memory"
    if re.search(r'(RDMA|ROCI|NCCL|RING|TP_|FABRIC|MCDMA|WFE|CUDA_SCHED|CUDA_EVENTS|CUDA_DEVICE|CUDA_PROC|LINK_|KIT)', n):
        return "transport"
    if re.search(r'(DUMP|TRACE|_LOG|PROF|_FAULT|FAIL_TEST|REPLAY|CLOCK_|OVERLAP|KSET|STAND_IN|PROMPT_TRACE|TIMING)', n):
        return "debug"
    if re.search(r'(PASS|PROMPT|PREFILL|ROUNDS_BESIDE|BESIDE|PF_|CHUNK|STREAMS|ROUND_ROWS|GRAPH)', n):
        return "prompt"
    if re.search(r'(DRAFT|SAMPLE|HEAD|COPY|CAND|ROUND|CONF|DEC_|DECODE|SERVED_K|FIX_K)', n):
        return "decode"
    return "other"


# apply class (ENGINE.md section 1, REVIEW-ENGINE.md): P per request, H rank-0 host, A agreed live, I on idle, N now, R restart
H_NAMES = r'^(TF_DS_ROUNDS_BESIDE|TENSORFOLD_PREFILL_SHARE|TF_DS_PASS_BESIDE|TF_DS_PASS_FROM|TF_DS_PASS_FILLING|TF_DS_PROMPT_ORDER|TF_DS_PROMPT_QOS|TF_DS_PROMPT_MAX_WAIT_S|TF_DS_PREFILL_YIELD|TF_DS_DRAFT_MS|TF_DS_CONF_CAL|TF_DS_ROUND_MS|TF_DS_SAMPLE_CAND|TF_DS_SPILL_(MIN_TOKENS|MIN_FREE_GIB|QUOTA|IOPRIO|BACKLOG_GIB|KEEP_BUILDS|PROBE_[A-Z]+)|TF_SPILL_[A-Z_]+|TENSORFOLD_API_KEY|TENSORFOLD_CLOSE_MS|TENSORFOLD_LOG_POLLS|TENSORFOLD_NO_LIVE|TENSORFOLD_REQUEST_LOG|TF_SSE_KEEPALIVE_S)$'
A_NAMES = r'^(TF_DS_GRAPH_CAP|TF_DS_ENGRAM_KEEP_AHEAD|TF_DS_ENGRAM_AHEAD_PASS|TF_DS_ISSUE_LEAD_[A-Z_]+|TF_DS_RING_DEAD_MS|TF_DS_RING_WORD_DEADLINE_MS|TF_DS_ENGRAM_AIO_(SLEEP|WAKE|WAKE_US|SPIN)|TF_DS_RDMA_NAP_US|TF_DS_ROCI_STAGGER_(GAP|PRIO))$'
I_NAMES = r'^(TF_DS_IMAGE_CACHE|TF_DS_ARENA_GIB|TF_DS_WINDOW|TF_DS_ENGRAM_CACHE_MIB)$'


def apply_of(n):
    if re.match(H_NAMES, n):
        return "H"
    if re.match(A_NAMES, n):
        return "A"
    if re.match(I_NAMES, n):
        return "I"
    return "R"


FLOAT_RE = r'(_GIB|_MIB|SHARE|RATIO|RATE_GBPS|_GB$|FPS|QUOTA|TEMP)'
INT_RE = r'(_MS$|_US$|_NS$|_S$|ROWS|_CAP$|DEPTH|_PORT$|WORLD|COUNT|_MAX$|_MIN$|TURNS|STREAMS|SLOTS|READERS|RESTORES|SMALLS|FRAMES|SIDE$|_DEVICE$|SEGMENTS|_TC$|_SL$|LEAD$|_BEFORE|AFTER|WAIT|SLICE|SPIN|WAKE|SLEEP|CONNECT_S|PIECE|TOKENS|EVERY|BUILDS|HOLD|WARN)'
TEXT_RE = r'(CPUS|DEVICES|_DIR$|_DIR_|PATH|_FILE$|_LIB$|LIBRARY|_ID$|_SHA$|REVISION|MASTER|LISTEN|KIT|PREFIX|GID|SOCKET|TOKENIZER|VOCAB$|_KEY$|HOME|ABLATION|_LOCAL$|ORDER|QOS|KERNELS$|IOPRIO|REASON|SETS|HOT|GEOMETRY)'


def kind_of(n):
    if re.search(TEXT_RE, n):
        return "text"
    if re.search(FLOAT_RE, n):
        return "float"
    if re.search(INT_RE, n):
        return "int"
    return "flag"


def unit_of(n):
    for suffix, unit in (("_MS", "ms"), ("_US", "us"), ("_NS", "ns"), ("_GIB", "GiB"), ("_MIB", "MiB"), ("_GBPS", "GB/s"), ("_S", "s"), ("ROWS", "rows")):
        if n.endswith(suffix):
            return unit
    return ""


def main():
    params_path = "params-ds.json"
    if "--params" in sys.argv:
        params_path = sys.argv[sys.argv.index("--params") + 1]
    params = {}
    if os.path.exists(params_path):
        params = json.load(open(params_path))["keys"]
    old = json.load(open(SCHEMA)) if os.path.exists(SCHEMA) else {"schema_version": 1, "options": []}
    rows = old["options"]
    have = {r["env"] for r in rows}
    next_id = {g: max([r["id"] for r in rows if r["group"] == g] + [GROUPS[g][0] - 1]) + 1 for g in GROUPS}
    added = 0
    found = scan()
    for n in sorted(found):
        if n in have:
            continue
        g = group_of(n)
        path, ln = found[n]
        row = {
            "id": next_id[g],
            "name": n.lower(),
            "env": n,
            "group": g,
            "tab": GROUPS[g][1],
            "kind": kind_of(n),
            "unit": unit_of(n),
            "min": None,
            "max": None,
            "default": "",
            "default_check": "none",
            "apply": apply_of(n),
            "who": ["admin"],
            "doc": doc_at(path, ln),
            "values": [],
            "seed": "name",
            "first_use": "%s:%d" % (path, ln),
        }
        p = params.get(n)
        if p and p.get("class") in ("editable", "expert"):
            row["seed"] = "params"
            t = p.get("type")
            row["kind"] = {"bool01": "flag", "int": "int", "number": "float", "text": "text", "cpus3": "text", "preset": "enum"}.get(t, row["kind"])
            if "min" in p:
                row["min"] = p["min"]
            if "max" in p:
                row["max"] = p["max"]
            if p.get("values"):
                row["values"] = p["values"]
            if p.get("meaning"):
                row["doc"] = p["meaning"]
        next_id[g] += 1
        rows.append(row)
        added += 1
    rows.sort(key=lambda r: r["id"])
    old["options"] = rows
    os.makedirs(os.path.dirname(SCHEMA), exist_ok=True)
    with open(SCHEMA, "w") as f:
        json.dump(old, f, indent=1)
        f.write("\n")
    print("schema: %d rows, %d added" % (len(rows), added))


if __name__ == "__main__":
    main()
