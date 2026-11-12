#!/usr/bin/env python3
"""Decode and prefill speed of an OpenAI-compatible server, from a client that cannot be the bottleneck (one process
per concurrent request; standard library only).

  decode:  N requests at once, streamed. A request's rate = (completion tokens - 1) / (last token - first token),
           completion tokens from the server's usage (reasoning included); TTFT = first token - send. "agg" = the
           requests' rates summed (the aggregate); "wall" = all completion tokens / (last token - first send).
           prose: thinking on, temperature 0.7; code: greedy, thinking off.
  prefill: one request at a time, a prompt of about SIZE tokens that no earlier request shares (a fresh nonce first,
           so no prompt cache can serve any of it), 1 token out, thinking off; tok/s = prompt tokens / TTFT.

  python3 tools/bench.py URL decode --kind prose --conc 1,2,4,8,16 --reps 3 --max-tokens 1024 --out FILE.jsonl
  python3 tools/bench.py URL prefill --sizes 4096,8192,16384,32768,65536,131072,262144 --reps 1 --out FILE.jsonl
With keys on: --key-file FILE (its first key: "label: key" lines or a bare key, # comments), or ENGINE_API_KEY in the
environment; sent as "Authorization: Bearer", never printed. Every result line carries what was measured, where, and
the settings; the summary prints medians per setting.
"""
import argparse, json, multiprocessing as mp, os, random, statistics, sys, time, urllib.request

# The key reaches the request processes through ENGINE_API_KEY in the environment (they inherit it); never printed.


def first_key(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line.split(":", 1)[1].strip() if ":" in line else line
    sys.exit(f"{path} holds no key")


def headers():
    key = os.environ.get("ENGINE_API_KEY")
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}

PROSE = ["the history of the printing press", "how bridges carry load", "the water cycle", "the economics of tea",
         "why the sky is blue", "the life of honeybees", "how vaccines train immunity", "the Roman road network",
         "the invention of the telescope", "ocean currents and climate", "the chemistry of bread",
         "how jet engines work", "the migration of monarch butterflies", "the story of the transistor",
         "volcanoes and plate tectonics", "the design of medieval castles"]
CODE = ["an LRU cache class", "a JSON tokenizer", "a thread-safe bounded queue", "Dijkstra's shortest paths",
        "a trie with prefix search", "a rate limiter (token bucket)", "an interval tree", "a CSV parser with quotes",
        "a min-heap from scratch", "a topological sort with cycle detection", "a simple regex matcher (., *)",
        "a Bloom filter", "a binary search tree with deletion", "a union-find with path compression",
        "an event emitter with once()", "a matrix multiplication with tiling"]


def stream(url, model, messages, max_tokens, temperature, thinking, extra=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    body.update(extra or {})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), headers())
    sent = time.time()
    first = last = None
    usage = None
    chars = 0
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning") or "") + (delta.get("reasoning_content") or "")
                if text:
                    now = time.time()
                    first = first or now
                    last = now
                    chars += len(text)
    return {"sent": sent, "first": first, "last": last, "usage": usage, "chars": chars}


def one_decode(a):
    url, model, kind, i, max_tokens, barrier_at = a
    while time.time() < barrier_at:
        time.sleep(0.001)
    if kind == "prose":
        msg = [{"role": "user", "content": f"Write a detailed, well-structured essay about {PROSE[i % len(PROSE)]}."}]
        r = stream(url, model, msg, max_tokens, 0.7, True)
    else:
        msg = [{"role": "user", "content": f"Write a complete, well-commented Python implementation of "
                                           f"{CODE[i % len(CODE)]}, with a short usage example. Code only."}]
        r = stream(url, model, msg, max_tokens, 0.0, False)
    n = (r["usage"] or {}).get("completion_tokens", 0)
    span = (r["last"] - r["first"]) if r["first"] and r["last"] else 0
    r.update(tokens=n, rate=(n - 1) / span if span > 0 and n > 1 else 0.0, ttft=(r["first"] or r["sent"]) - r["sent"])
    return r


COMMON = ("time year people way day man thing woman life child world school state family student group country "
          "problem hand part place case week company system program question work government number night point "
          "home water room mother area money story fact month lot right study book eye job word business issue "
          "side kind head house service friend father power hour game line end member law car city community name "
          "president team minute idea kid body information back parent face others level office door health person "
          "art war history party result change morning reason research girl guy moment air teacher force education "
          "river garden window letter music market paper table field picture bridge stone light road train").split()


def filler(n_words, seed):
    """n_words common English words in a random order: random letter strings tokenize at ~3.5 tokens a word
    (measured on GLM), common words near one; main() scales n_words by the server's own count (calibrate)."""
    rnd = random.Random(seed)
    return " ".join(rnd.choice(COMMON) for _ in range(n_words))


def calibrate(url, model):
    """Tokens a filler word, from the server's usage on a 4,000-word prompt."""
    nonce = f"{time.time_ns()}"
    r = stream(url, model, [{"role": "user", "content": f"[{nonce}] " + filler(4000, nonce)}], 1, 0.0, False)
    return max(0.5, ((r["usage"] or {}).get("prompt_tokens", 4000) - 20) / 4000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url"); p.add_argument("mode", choices=["decode", "prefill"])
    p.add_argument("--model", default="deepseek-v4.1-flash-1017"); p.add_argument("--kind", default="prose", choices=["prose", "code"])
    p.add_argument("--conc", default="1,2,4,8"); p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--sizes", default="4096,8192,16384,32768,65536,131072,262144")
    p.add_argument("--label", default=""); p.add_argument("--out")
    p.add_argument("--key-file")
    a = p.parse_args()
    if a.key_file:
        os.environ["ENGINE_API_KEY"] = first_key(a.key_file)
    out = open(a.out, "a") if a.out else None
    summary = []

    def emit(rec):
        rec.update(label=a.label, url=a.url, at=time.strftime("%F %T"))
        line = json.dumps(rec)
        print(line, flush=True)
        if out:
            out.write(line + "\n"); out.flush()

    if a.mode == "decode":
        for c in [int(x) for x in a.conc.split(",")]:
            runs = []
            for rep in range(a.reps):
                at = time.time() + 1.0
                with mp.get_context("fork").Pool(c) as pool:
                    rs = pool.map(one_decode, [(a.url, a.model, a.kind, rep * 7 + i, a.max_tokens, at) for i in range(c)])
                agg = sum(r["rate"] for r in rs)
                wall = sum(r["tokens"] for r in rs) / (max(r["last"] or 0 for r in rs) - min(r["sent"] for r in rs))
                rec = {"mode": "decode", "kind": a.kind, "conc": c, "rep": rep, "max_tokens": a.max_tokens,
                       "agg_tok_s": round(agg, 1), "per_req_tok_s": round(statistics.mean(r["rate"] for r in rs), 1),
                       "wall_tok_s": round(wall, 1), "ttft_s": round(statistics.mean(r["ttft"] for r in rs), 3),
                       "tokens": [r["tokens"] for r in rs]}
                emit(rec); runs.append(rec)
            summary.append((f"decode {a.kind} x{c}", statistics.median(r["agg_tok_s"] for r in runs),
                            statistics.median(r["per_req_tok_s"] for r in runs), statistics.median(r["ttft_s"] for r in runs)))
        print("\n| setting | agg tok/s | per request tok/s | TTFT s |\n|---|---:|---:|---:|")
        for s in summary:
            print(f"| {s[0]} | {s[1]:.1f} | {s[2]:.1f} | {s[3]:.2f} |")
    else:
        print_rows = []
        per_word = calibrate(a.url, a.model)
        print(f"calibration: {per_word:.3f} tokens a filler word", flush=True)
        for size in [int(x) for x in a.sizes.split(",")]:
            runs = []
            for rep in range(a.reps):
                nonce = f"{time.time_ns()}-{random.random()}"
                text = f"[{nonce}] " + filler(int(size / per_word), nonce) + "\n\nReply with the single word OK."
                r = stream(a.url, a.model, [{"role": "user", "content": text}], 1, 0.0, False)
                pt = (r["usage"] or {}).get("prompt_tokens", 0)
                ttft = (r["first"] or r["last"] or time.time()) - r["sent"]
                rec = {"mode": "prefill", "size": size, "rep": rep, "prompt_tokens": pt, "ttft_s": round(ttft, 3),
                       "tok_s": round(pt / ttft, 1) if ttft > 0 else 0}
                emit(rec); runs.append(rec)
            print_rows.append((size, statistics.median(r["prompt_tokens"] for r in runs),
                               statistics.median(r["ttft_s"] for r in runs), statistics.median(r["tok_s"] for r in runs)))
        print("\n| size | prompt tokens | TTFT s | tok/s |\n|---:|---:|---:|---:|")
        for s in print_rows:
            print(f"| {s[0]} | {s[1]:,} | {s[2]:.2f} | {s[3]:,.1f} |")


if __name__ == "__main__":
    main()
