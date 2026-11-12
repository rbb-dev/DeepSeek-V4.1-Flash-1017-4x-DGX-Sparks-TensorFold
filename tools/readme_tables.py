#!/usr/bin/env python3
"""The README's performance tables, rebuilt from the benchmark runs' own files (the evidence folder holds them):

  python3 tools/readme_tables.py --main recipe-main.json --turbo recipe-turbo.json --code code.jsonl

--main and --turbo are tools/recipe_bench.py's --out files (the prefill, decode and multi suites; the Turbo run's prefill);
--code is tools/bench.py's --out file (decode --kind code). Each cell is the median of the runs' repetitions, as the
README says; a first token is the mean of the requests' own first tokens in one repetition, then that median.
Prints the three tables in the README's own form, so a table can be checked against the files that made it.
"""
import argparse, collections, json, statistics


def label(n):
    return f"{n // 1048576}M" if n % 1048576 == 0 else f"{n // 1000000}M" if n % 1000000 == 0 else f"{n // 1024}K"


def secs(s):
    """A time the way the README writes it: 710 ms, 1.70 s, 16.4 s."""
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.2f} s" if s < 10 else f"{s:.1f} s"


def rate(x, digits=0):
    return f"{x:,.{digits}f}"


def prompt_table(main, turbo):
    sizes = sorted(int(k.split(".")[1]) for k in main["summary"] if k.startswith("prefill.") and k.endswith(".tok_s"))
    best = max(main["summary"][f"prefill.{n}.tok_s"] for n in sizes)
    out = ["| Prompt | Tokens a second | First token | With Turbo | First token | Turbo speed change |",
           "|---:|---:|---:|---:|---:|---:|"]
    for n in sizes:
        s, t = main["summary"][f"prefill.{n}.tok_s"], main["summary"][f"prefill.{n}.ttft_s"]
        cell = f"**{rate(s)}**" if s == best else rate(s)
        if turbo and f"prefill.{n}.tok_s" in turbo["summary"]:
            ts, tt = turbo["summary"][f"prefill.{n}.tok_s"], turbo["summary"][f"prefill.{n}.ttft_s"]
            out.append(f"| {label(n)} | {cell} | {secs(t)} | {rate(ts)} | {secs(tt)} | {100 * (ts / s - 1):+.1f} % |")
        else:
            out.append(f"| {label(n)} | {cell} | {secs(t)} | | | |")
    return "\n".join(out)


def replies_table(main):
    first = collections.defaultdict(list)
    for r in main["runs"]:
        if r.get("suite") == "multi" and r.get("ttft_s"):
            first[r["streams"]].append(statistics.mean(r["ttft_s"]))
    streams = sorted(int(k.split(".")[1]) for k in main["summary"] if k.startswith("multi.") and k.endswith(".aggregate_tok_s"))
    out = ["| Requests at once | Total | Each | First token |", "|---:|---:|---:|---:|"]
    for n in streams:
        tot, each = main["summary"][f"multi.{n}.aggregate_tok_s"], main["summary"][f"multi.{n}.per_stream_tok_s"]
        cell = f"**{tot:.1f}**" if n in (streams[0], streams[-1]) else f"{tot:.1f}"
        out.append(f"| {n} | {cell} | {each:.1f} | {secs(statistics.median(first[n]))} |")
    return "\n".join(out)


def code_table(rows):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("mode") == "decode":
            by[r["conc"]].append(r)
    out = ["| Requests at once | Total | Each | First token |", "|---:|---:|---:|---:|"]
    cs = sorted(by)
    for c in cs:
        rs = by[c]
        tot = statistics.median(r["wall_tok_s"] for r in rs)
        cell = f"**{tot:.1f}**" if c in (cs[0], cs[-1]) else f"{tot:.1f}"
        out.append(f"| {c} | {cell} | {statistics.median(r['per_req_tok_s'] for r in rs):.1f} | "
                   f"{secs(statistics.median(r['ttft_s'] for r in rs))} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--main", required=True, help="recipe_bench.py --out of the prefill, decode and multi suites")
    ap.add_argument("--turbo", help="recipe_bench.py --out of the Turbo run's prefill suite")
    ap.add_argument("--code", help="bench.py --out of decode --kind code")
    a = ap.parse_args()
    m = json.load(open(a.main))
    t = json.load(open(a.turbo)) if a.turbo else None
    print("Reading a prompt\n\n" + prompt_table(m, t) + "\n")
    print("Writing replies\n\n" + replies_table(m) + "\n")
    if a.code:
        print("Writing code\n\n" + code_table([json.loads(l) for l in open(a.code) if l.strip()]))


if __name__ == "__main__":
    main()
