#!/usr/bin/env bash
# Long-context retrieval checks: the sets this repository's README reports, run against your own server.
#
#   tools/needle_test.sh 1m   one needle at 0/25/50/75/100 % of a 128K and a 1M prompt (tools/needle_bench.py),
#                             then ten needles in one 128K and one 1M prompt (tools/multi_needle.py)
#   tools/needle_test.sh 2m   one needle at 128K, 1M, 1.5M and 2M, ten needles at 128K, 1M and 2M; start the
#                             engine with the 2M window first:
#                             CONTEXT=2097152 TF_DS_YARN_FACTOR=32 ./start.sh restart
#
# URL (default http://localhost:8000) and MODEL (default deepseek-v4.1-flash-1017) from the environment; with keys on, API_KEY or
# API_KEY_FILE as tools/recipe_bench.py. For cold numbers start with PROMPT_CACHE_GIB=0 SPILL_GIB=0. Every prompt is one
# request at a time. Time on four Sparks, cold: 1m ~20 min, 2m ~1 h 30 min. Results: needle-<set>-<date>/*.jsonl.
# Exit 0 when every needle came back right.
set -u
here=$(cd "$(dirname "$0")" && pwd)
set_=${1:-}
case $set_ in
  1m) one="131072,1000000"; ten="131072,1000000" ;;
  2m) one="131072,1000000,1500000,2000000"; ten="131072,1000000,2000000" ;;
  *) echo "usage: $0 1m|2m" >&2; exit 2 ;;
esac
URL=${URL:-http://localhost:8000}; MODEL=${MODEL:-deepseek-v4.1-flash-1017}
out=needle-$set_-$(date +%Y%m%d-%H%M%S); mkdir -p "$out"
echo "== $set_: one needle at $one tokens, depths 0/25/50/75/100 ($URL, model $MODEL; results in $out/)"
python3 "$here/needle_bench.py" --url "$URL" --model "$MODEL" --lens "$one" --depths 0,25,50,75,100 --json "$out/one-needle.jsonl"; a=$?
echo "== $set_: ten needles in one prompt at $ten tokens, depths 5..95 %"
python3 "$here/multi_needle.py" --url "$URL" --model "$MODEL" --lens "$ten" --json "$out/ten-needles.jsonl"; b=$?
(( a == 0 && b == 0 )) && echo "== $set_: every needle found" || echo "== $set_: NOT every needle found (one needle exit $a, ten needles exit $b)"
(( a == 0 && b == 0 ))
