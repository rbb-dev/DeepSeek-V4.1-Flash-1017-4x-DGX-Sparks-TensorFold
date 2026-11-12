#!/usr/bin/env python3
"""LongBench v2 against the engine, the way its authors' pred.py runs it (THUDM/LongBench, MIT; the data: Hugging Face
zai-org/LongBench-v2, data.json): their 0-shot template, temperature 0.1, 128 answer tokens, a prompt longer than the
limit cut in the middle by the model's own tokens (the first and last halves kept), the answer read from "The correct
answer is (X)". Thinking is off unless --thinking names an effort; then the reasoning gets --max-tokens (16,384).

One JSON line an item goes to --out as it finishes; an item already there is not asked again, so a stopped run goes on
where it stopped. At the end: the accuracy, overall and by difficulty and domain, with its standard error.

  python3 tools/longbench2.py --url http://localhost:8000 --key-file ~/.config/dsv41/api-keys \\
      --data data.json --tokenizer <the checkpoint's tokenizer.json> --length long --out lb2-long.jsonl

Cutting a prompt needs the `tokenizers` package and the checkpoint's tokenizer.json; --max-prompt-tokens 0 sends every
prompt whole (the engine refuses one longer than its window).
"""
import argparse, collections, json, math, os, re, sys, time, urllib.error, urllib.request

MODEL = "deepseek-v4.1-flash-1017"
# the authors' prompts/0shot.txt (THUDM/LongBench @2e00731f8d0b), word for word
TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


def middle_cut(ids, limit):
    """The authors' cut: a prompt longer than limit tokens keeps its first and last limit/2 tokens (0: no limit)."""
    if limit <= 0 or len(ids) <= limit:
        return ids
    return ids[:limit // 2] + ids[-(limit // 2):]


def extract_answer(text):
    """pred.py's extract_answer: the letter of "The correct answer is (X)" or "... is X", '*' removed; else None."""
    text = (text or "").replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", text) or re.search(r"The correct answer is ([A-D])", text)
    return m.group(1) if m else None


def prompt_of(item):
    return (TEMPLATE.replace("$DOC$", item["context"].strip()).replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip()).replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip()).replace("$C_D$", item["choice_D"].strip()))


def read_key(path):
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split(":", 1)[1].strip() if ":" in line else line
    sys.exit(f"{path}: no key in it")


def ask(url, key, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **({"authorization": f"Bearer {key}"} if key else {})})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def summary(rows):
    def line(name, rs):
        n = len(rs); k = sum(1 for r in rs if r["judge"])
        p = k / n if n else 0.0
        se = math.sqrt(p * (1 - p) / n) if n else 0.0
        return f"{name:36s} accuracy {k} of {n} = {100 * p:.1f} % (se {100 * se:.1f})"
    out = [line("all", rows)]
    for key in ("difficulty", "domain"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        out += ["  " + line(f"{key} {g}", rs) for g, rs in sorted(groups.items())]
    unread = sum(1 for r in rows if r["pred"] is None)
    out.append(f"answers not in the authors' form (counted wrong): {unread}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--key-file", help="a file of 'label: key' lines; its first key is sent")
    ap.add_argument("--data", required=True, help="LongBench v2's data.json")
    ap.add_argument("--tokenizer", help="the checkpoint's tokenizer.json (needed to cut a prompt over the limit)")
    ap.add_argument("--max-prompt-tokens", type=int, default=1024000, help="the cut's limit in tokens (0: none)")
    ap.add_argument("--length", choices=("short", "medium", "long", "all"), default="all")
    ap.add_argument("--ids", help="a file of item ids, one a line: only those")
    ap.add_argument("--thinking", choices=("off", "low", "medium", "high", "max"), default="off")
    ap.add_argument("--max-tokens", type=int, help="answer tokens (default 128, or 16,384 with thinking)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds one request may take")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = json.load(open(a.data))
    if a.length != "all":
        items = [d for d in items if d["length"] == a.length]
    if a.ids:
        want = [l.strip() for l in open(a.ids) if l.strip()]
        by = {d["_id"]: d for d in items}
        items = [by[i] for i in want if i in by]
    done = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                r = json.loads(line); done[r["_id"]] = r
    key = read_key(a.key_file) if a.key_file else None
    tok = None
    max_tokens = a.max_tokens or (128 if a.thinking == "off" else 16384)
    todo = [d for d in items if d["_id"] not in done]
    print(f"longbench2: {len(items)} items ({a.length}), {len(done)} already in {a.out}, {len(todo)} to ask; thinking "
          f"{a.thinking}, {max_tokens} answer tokens, prompts cut at {a.max_prompt_tokens or 'no limit'}", flush=True)
    with open(a.out, "a") as fout:
        for i, d in enumerate(todo, 1):
            prompt = prompt_of(d)
            cut = False
            if a.max_prompt_tokens > 0 and len(prompt) > a.max_prompt_tokens:   # fewer characters than the limit: under it
                if tok is None:
                    if not a.tokenizer:
                        sys.exit("a prompt may be over --max-prompt-tokens: give --tokenizer <tokenizer.json> (or --max-prompt-tokens 0)")
                    from tokenizers import Tokenizer
                    tok = Tokenizer.from_file(a.tokenizer)
                ids = tok.encode(prompt, add_special_tokens=False).ids
                kept = middle_cut(ids, a.max_prompt_tokens)
                if len(kept) < len(ids):
                    prompt = tok.decode(kept, skip_special_tokens=True); cut = True
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1,
                    "max_tokens": max_tokens}
            if a.thinking == "off":
                body["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                body["reasoning_effort"] = a.thinking
            try:
                out, secs = ask(a.url, key, body, a.timeout)
                msg = out["choices"][0]["message"]
                text, finish = msg.get("content") or "", out["choices"][0].get("finish_reason")
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                usage = out.get("usage", {})
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                print(f"  {d['_id']}: request failed ({e}); not written, a later run asks it again", flush=True)
                continue
            pred = extract_answer(text)
            r = {"_id": d["_id"], "domain": d["domain"], "sub_domain": d["sub_domain"], "difficulty": d["difficulty"],
                 "length": d["length"], "answer": d["answer"], "pred": pred, "judge": pred == d["answer"],
                 "response": text, "reasoning_chars": len(reasoning), "finish": finish, "cut": cut, "seconds": round(secs, 2),
                 "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                 "thinking": a.thinking, "model": a.model}
            fout.write(json.dumps(r, ensure_ascii=False) + "\n"); fout.flush()
            done[d["_id"]] = r
            print(f"  {i}/{len(todo)} {d['_id']} {d['difficulty']:4s} {r['prompt_tokens'] or '?':>9} tokens{' (cut)' if cut else ''} "
                  f"{secs:7.1f} s: {pred or '-'} vs {d['answer']} {'right' if r['judge'] else 'wrong'}", flush=True)
    rows = [done[d["_id"]] for d in items if d["_id"] in done]
    print(summary(rows))


if __name__ == "__main__":
    main()
