"""Many needles in one haystack: six-digit codes, each tied to its own city, hidden at evenly spread depths of ONE filler
prompt (ten needles: 5, 15, 25 ... 95 %; twenty: 2.5, 7.5 ... 97.5 %); the question asks for them at once; the score is
how many come back right. Harder than tools/needle_bench.py's one needle: the model must find many facts spread over the
whole window in a single reply.

    python3 tools/multi_needle.py [--url http://localhost:8000] [--model deepseek-v4.1-flash-1017] [--lens 131072,1000000]
        [--reps 1] [--needles 10] [--ask N] [--seed S] [--json OUT.jsonl]

--needles 10 or 20 plants that many; --ask N asks for N of them, spread evenly (every second one of twenty for 10), the
rest stay planted. --seed S draws the same codes every run: two runs with the same seed, lengths and needles send the
same haystack, and only the question at its end differs, so a server's prompt cache can serve the second from the first
(plant 20, ask 10 cold; then ask all 20 from the cache).

Needs only the Python standard library. The filler, the calibration (one short request turns "tokens" into "sentence
repeats", so prompt_tokens is approximate) and the request (greedy, thinking off) are tools/needle_bench.py's. With
keys on (start.sh with API_KEY=1) it sends API_KEY, else the first key in API_KEY_FILE (default
~/.local/state/dsv41-tensorfold/api-key). Exit 0 when every code of every length came back right.
"""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

args = sys.argv[1:]


def arg(name, default):
    if name not in args:
        return default
    i = args.index(name) + 1
    if i >= len(args) or args[i].startswith("--"):
        sys.exit(f"{name} needs a value")
    return args[i]


URL = arg("--url", "http://localhost:8000").rstrip("/")
MODEL = arg("--model", "deepseek-v4.1-flash-1017")
try:
    LENS = [int(x) for x in arg("--lens", "131072,1000000").split(",")]
    REPS = int(arg("--reps", "1"))
except ValueError as e:
    sys.exit(f"--lens and --reps take whole numbers: {e}")
if not all(x > 0 for x in LENS) or REPS < 1:
    sys.exit("--lens takes positive token counts, --reps at least 1")
JSON_OUT = arg("--json", None)

CITIES = ["Lisbon", "Kyoto", "Nairobi", "Quito", "Oslo", "Hanoi", "Dakar", "Lima", "Tbilisi", "Perth",
          "Accra", "Bergen", "Cusco", "Durban", "Fukuoka", "Gdansk", "Havana", "Izmir", "Jaipur", "Kazan"]
try:
    NEEDLES = int(arg("--needles", "10"))
    ASK = int(arg("--ask", str(NEEDLES)))
except ValueError as e:
    sys.exit(f"--needles and --ask take whole numbers: {e}")
if NEEDLES not in (10, 20) or not 1 <= ASK <= NEEDLES:
    sys.exit("--needles takes 10 or 20, --ask 1 to --needles")
SEED = arg("--seed", None)
if SEED is not None:
    random.seed(SEED)
# the planted cities in depth order (ten: the first ten, at 5 ... 95 %), and the ASK of them the question names, evenly spread
NAMES = CITIES[:NEEDLES]
DEPTHS = [(i + 0.5) * 100 / NEEDLES for i in range(NEEDLES)]
ASKED = [NAMES[i * NEEDLES // ASK] for i in range(ASK)]
FILLER = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "


def api_key():
    """The server's API key: API_KEY, else the first key in API_KEY_FILE, else none (as tools/recipe_bench.py)."""
    key = os.environ.get("API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("API_KEY_FILE", os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key"))
    try:
        with open(path) as f:
            line = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    except OSError:
        return ""
    return line.split(":", 1)[1].strip() if ":" in line else line


HEADERS = {"content-type": "application/json"}
if api_key():
    HEADERS["authorization"] = "Bearer " + api_key()


def post(body, timeout=3600):   # a cold 2M-token prompt reads in ~7 min on four Sparks
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")}


def chat(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    return post(body)


def text(r):
    """The reply's text when r is a completion with non-empty assistant content, else None (an error is never read as
    the model's reply)."""
    try:
        m = r["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    if "error" in r or not isinstance(m, dict) or m.get("role", "assistant") != "assistant":
        return None
    c = m.get("content")
    return c if isinstance(c, str) and c else None


def prompt_tokens(r):
    usage = r.get("usage") if isinstance(r, dict) else None
    n = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None


def found_codes(reply):
    """City -> the six-digit code the reply gives it ("City: 123456", a few separators allowed), or None."""
    out = {}
    for n in ASKED:
        m = re.search(n + r"\W{0,6}(\d{6})", reply or "")
        out[n] = m.group(1) if m else None
    return out


calib_repeats = 200
calib = chat(FILLER * calib_repeats + "\nSay OK.", max_tokens=5)
calib_tok = prompt_tokens(calib)
if text(calib) is None or calib_tok is None or calib_tok < calib_repeats:
    sys.exit("Calibration request failed: " + json.dumps(calib)[:300])
tokens_per_repeat = calib_tok / calib_repeats
print(f"Calibration: {calib_tok} tokens for {calib_repeats} filler repeats ({tokens_per_repeat:.2f} tokens/repeat)\n", flush=True)

results = []
for length in LENS:
    for rep in range(REPS):
        repeats = max(1, int(length / tokens_per_repeat))
        codes = {n: str(random.randint(100000, 999999)) for n in NAMES}
        pieces = [FILLER] * repeats
        # deepest first, so inserting one needle does not move the positions of the ones still to come (every planted
        # city's code is drawn, asked or not: the same seed plants the same haystack whatever --ask is)
        for n, d in sorted(zip(NAMES, DEPTHS), key=lambda x: -x[1]):
            pieces.insert(min(int(repeats * d / 100), len(pieces)), f"The vault code of {n} is {codes[n]}. ")
        # the question comes after the whole haystack: requests that plant the same needles share every token before it
        ask = ("\nList the vault code of each city named above, one line each, in the form City: code. Cities: "
               + ", ".join(ASKED) + ".")
        t0 = time.time()
        r = chat("".join(pieces) + ask, max_tokens=max(200, 14 * ASK))
        dt = time.time() - t0
        reply = text(r)
        found = found_codes(reply)
        right = [n for n in ASKED if found[n] == codes[n]]
        depth = dict(zip(NAMES, DEPTHS))
        row = {"length": length, "rep": rep, "needles": NEEDLES, "prompt_tokens": prompt_tokens(r), "right": len(right),
               "of": len(ASKED), "seconds": round(dt, 1),
               "misses": {f"{n} ({depth[n]:g} %)": [codes[n], found[n]] for n in ASKED if found[n] != codes[n]},
               "reply": (reply or "")[:400]}
        if reply is None:
            row["error"] = json.dumps(r)[:300]
        results.append(row)
        print(f"len={length:>8} tok={row['prompt_tokens']} right={len(right)}/{len(ASKED)} of {NEEDLES} planted {dt:>6.1f}s"
              + (f" misses (expected, got)={row['misses']}" if row["misses"] else "")
              + (f" {row['error']}" if reply is None else ""), flush=True)
        if JSON_OUT:
            with open(JSON_OUT, "a") as f:
                f.write(json.dumps(row) + "\n")

print("\n" + "  ".join(f"{r['length']}: {r['right']}/{r['of']}" for r in results))
sys.exit(0 if all(r["right"] == r["of"] for r in results) else 1)
