#!/usr/bin/env python3
"""Fails when the engine reads an environment name that has no schema row, or the generated table is out of date.

usage: check_env_schema.py        (run from the engine's root; exit 1 on a finding)
The scan is seed.py's: every string literal that names an environment variable (TF_*, TENSORFOLD_*, GLM_*, FZ_*, MCDMA_*,
NCCL_*, CUDA_*, or the literal first argument of getenv / environ_map.get / envOn / envInt / envFloat) in zig/src,
tests, comments and the code this engine does not run (seed.NOT_THIS_ENGINE: other models, backends and tools) left out.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import seed  # noqa: E402

rows = {r["env"] for r in json.load(open(seed.SCHEMA))["options"]}
missing = {n: where for n, where in seed.scan().items() if n not in rows}
bad = False
for n, (p, ln) in sorted(missing.items()):
    print("no schema row: %s (%s:%d)" % (n, p, ln))
    bad = True
gen = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "gen_schema.py"), "--check"])
if gen.returncode:
    bad = True
if not bad:
    print("every environment name this engine reads has a schema row (%d names; the base engine's other models and tools left out: seed.NOT_THIS_ENGINE)" % len(rows))
sys.exit(1 if bad else 0)
