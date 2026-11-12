#!/usr/bin/env python3
"""Every host-to-device copy of the engine goes through cuda/driver.zig's copyHtoD / copyHtoDAsync (the one place that says where a failed copy came from).
usage: check_htod_sites.py   (run from the engine's root; exit 1 names every call of the driver's entry points elsewhere, and every wrapper call that does not pass @src())"""
import os, re, sys
BAD = re.compile(r"api\.cuMemcpyHtoD(Async)?_v2")
bad = []
n = 0
for root, _, names in os.walk("zig/src"):
    for f in names:
        if not f.endswith(".zig"): continue
        p = os.path.join(root, f)
        if p in ("zig/src/cuda/driver.zig", "zig/src/cuda/abi.zig"): continue
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            if BAD.search(line): bad.append("%s:%d calls the driver's copy entry point itself: %s" % (p, i, line.strip()[:100]))
            for m in re.finditer(r"\.copyHtoD(Async)?\((?!@src\(\), )", line):
                if '" ++ "' not in line: bad.append("%s:%d copyHtoD without @src(): %s" % (p, i, line.strip()[:100]))
            n += len(re.findall(r"\.copyHtoD(?:Async)?\(@src\(\), ", line))
print("\n".join(bad) if bad else "every host-to-device copy goes through Driver.copyHtoD / copyHtoDAsync (%d calls)" % n)
sys.exit(1 if bad else 0)
