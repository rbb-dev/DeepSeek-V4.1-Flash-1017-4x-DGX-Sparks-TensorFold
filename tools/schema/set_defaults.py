#!/usr/bin/env python3
"""Writes the defaults, ranges and units that a Zig test compares with the parsers' own (default_check "test").

Spill's whole-number and fractional settings are read from core/engine_api.zig (the table `ints`/`floats` and the
field defaults), so the schema cannot drift from them unseen: core/settings_test.zig fails if it does.
Run from the engine's root, after seed.py.
"""
import json
import re

SCHEMA = "zig/config-schema.json"
api = open("zig/src/core/engine_api.zig").read()

defaults = {}
for m in re.finditer(r'^\s+(\w+): (?:u32|f64|bool) = ([0-9_.]+|true|false),', api[api.index("pub const Spill = struct"):], re.M):
    defaults[m.group(1)] = m.group(2).replace("_", "")

rows = {}
for m in re.finditer(r'\.\{ \.name = "(\w+)", \.field = "(\w+)", \.min = ([^,]+), \.max = ([^,]+), \.unit = "([^"]+)" \}', api):
    name, field, lo, hi, unit = m.groups()
    ev = lambda s: float(eval(s.replace("<<", "<<").replace("1.0 / 1024.0", "(1.0/1024.0)")))
    rows[name] = dict(field=field, min=ev(lo), max=ev(hi), unit=unit, default=defaults[field])

fixed = {
    "TF_DS_SPILL_GIB": dict(default="0", min=0, max=1048576, unit="GiB", kind="float"),
    "TF_DS_SPILL_DIR": dict(default="", kind="text"),
    "TF_DS_SPILL_IOPRIO": dict(default="be/7", kind="text"),
    "TF_DS_IMAGE_CACHE": dict(default="1", kind="flag"),
    "TENSORFOLD_PREFILL_SHARE": dict(default="1", min=0.01, max=1, kind="float"),
    "TF_DS_ROUNDS_BESIDE": dict(default="", min=1, kind="int", unit="rounds"),
    "TENSORFOLD_CLOSE_MS": dict(default="20000", min=0, kind="int", unit="ms"),
    "TF_SSE_KEEPALIVE_S": dict(default="15", min=0, kind="float", unit="s"),
    "TENSORFOLD_PREPARE_FLOOR_GIB": dict(default="3", min=0, max=1048576, kind="float", unit="GiB"),
    "TENSORFOLD_SEED_SALT": dict(default="0", kind="int"),
    "TF_VIDEO_FPS": dict(default="2", kind="float"),
    "TF_VIDEO_MAX_FRAMES": dict(default="32", min=1, kind="int", unit="frames"),
    "TF_VIDEO_MAX_SIDE": dict(default="768", min=32, kind="int", unit="px"),
}

secrets = ["TENSORFOLD_API_KEY"]

docs = {
    "TF_ENGINE_CFG": "Path of this release's engine.cfg: `key = value` lines (a key is an option id, its name or its environment name) put where the engine's parsers read them: a value in the file replaces the environment's, except for a debug or bench option, and an option the file leaves out keeps the environment's. Unset: no file is read. A change made to the file while the engine is stopped is recorded in engine.history.jsonl beside it at the next start.",
    "TENSORFOLD_ADMIN_KEY_FILE": "The admin API's own key file (label: key lines, mode 0600, the format of --api-key-file). Unset: every /admin/ path is the unknown-path 404.",
}

d = json.load(open(SCHEMA))
by = {r["env"]: r for r in d["options"]}
for name, v in rows.items():
    r = by[name]
    r.update(kind="float" if v["unit"] in ("GiB", "of the cap") else "int", unit=v["unit"], min=v["min"], max=v["max"], default=v["default"], default_check="test")
    if r["kind"] == "int":
        r["min"], r["max"] = int(v["min"]), int(v["max"])
        r["default"] = str(int(float(v["default"])))
for name, v in fixed.items():
    by[name].update(v)
    # TF_VIDEO_* and the seed salt have their defaults inline in their reader: read in source, not compared by a test
    by[name]["default_check"] = "none" if name.startswith("TF_VIDEO_") or name == "TENSORFOLD_SEED_SALT" else "test"
for k, v in docs.items():
    by[k]["doc"] = v
for r in d["options"]:
    r["secret"] = r["env"] in secrets
    if r["secret"]:
        r["who"] = []
with open(SCHEMA, "w") as f:
    json.dump(d, f, indent=1)
    f.write("\n")
print("defaults checked by test:", sum(1 for r in d["options"] if r["default_check"] == "test"))
