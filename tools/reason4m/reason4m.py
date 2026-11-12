#!/usr/bin/env python3
"""Does the model REASON over a very long prompt (1M / 2M / 4M tokens), not just retrieve? Two suites, one request at
a time, against an OpenAI-compatible server (streamed). Standard library only.

  python3 reason4m.py run --suite mrcr --lengths 1m --modes off,high,max --samples 20 --url URL --out mrcr.jsonl
  python3 reason4m.py run --suite babilong --lengths 128k,1m --modes off,high,max --samples 20 --out babi.jsonl
  python3 reason4m.py run ... --dry-run          # the plan, sizes and time estimate; sends nothing
  python3 reason4m.py run ... --resume           # skips (length, task, mode, sample) rows already in --out
  python3 reason4m.py summary mrcr.jsonl babi.jsonl
  python3 reason4m.py estimate --suite mrcr --lengths 1m,2m,4m --modes off,high,max --samples 20
  python3 reason4m.py verify-data                # checks the downloaded data against the format this code assumes

README.md in this directory says what each suite measures, how its samples are built and what differs from the
published sets. Key: --key-file FILE ("label: key" lines or a bare key), else ENGINE_API_KEY, API_KEY, API_KEY_FILE
(as tools/bench.py and tools/needle_bench.py); sent as "Authorization: Bearer", never printed.
"""
import argparse
import difflib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# lengths are DECIMAL token targets of the prompt as OUR server counts them (usage.prompt_tokens)
LENGTHS = {"32k": 32_000, "128k": 128_000, "256k": 256_000, "1m": 1_000_000, "2m": 2_000_000, "4m": 4_000_000}


def split_label(label):
    """'1m-2n' -> ('1m', 2): the length name and, for MRCR, the needle count of the file (no suffix = 8)."""
    base, _, nn = label.partition("-")
    if base not in LENGTHS or (nn and nn not in ("2n", "4n", "8n")):
        raise ValueError(f"unknown length {label!r}; lengths: {', '.join(LENGTHS)}, MRCR also with -2n / -4n")
    return base, (int(nn[:-1]) if nn else 8)
MODES = {
    "off": {"thinking": False, "effort": None},
    "high": {"thinking": True, "effort": "high"},
    "max": {"thinking": True, "effort": "max"},
}
# max_tokens per mode (completion budget, reasoning included). MRCR must reproduce a whole ~450-token text; thinking
# modes are given room to think but bounded, because a token costs 1/15.5 s at 4M depth (32,768 tokens = 35 min).
MAX_TOKENS = {
    "mrcr": {"off": 2048, "high": 16384, "max": 32768},
    "omrcr": {"off": 2048, "high": 16384, "max": 32768},
    "babilong": {"off": 256, "high": 16384, "max": 32768},
}
DEFAULT_SEED = 20261112
SLICE_CHARS = 60_000      # a calibration request carries this much text (about 14K tokens)
N_SLICES = 4
ASSUMED_CPT = 4.3         # characters per token assumed ONLY where no server is asked (dry-run, estimate)

# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------


def mrcr_score(prediction, target):
    """Google DeepMind MRCR v2's metric (eval_hub/mrcr_v2/run_evaluation.py, mrcr_v2_metric), as the code does it:
    0 unless the target's 12-character random string occurs in the prediction; then SequenceMatcher(target_ref,
    text after the LAST occurrence of the string).ratio(), both stripped. (The docstring says the string must be the
    first 12 characters; the code takes the last occurrence anywhere. See score_strict for the docstring's reading.)"""
    if not isinstance(prediction, str) or not prediction:
        return 0.0
    target = target.strip()
    if len(target) < 12:
        return 0.0
    random_hash, target_ref = target[:12], target[12:].strip()
    prediction = prediction.strip()
    start = prediction.rfind(random_hash)
    if start == -1:
        return 0.0
    content = prediction[start + 12:].strip()
    return difflib.SequenceMatcher(a=target_ref, b=content).ratio()


def mrcr_score_strict(prediction, target):
    """The same, but the reply must BEGIN with the random string (the docstring's rule, and OpenAI's)."""
    if not isinstance(prediction, str) or not prediction.strip() or len(target.strip()) < 12:
        return 0.0
    return mrcr_score(prediction, target) if prediction.strip().startswith(target.strip()[:12]) else 0.0


def openai_mrcr_score(response, answer, random_string_to_prepend):
    """OpenAI's openai/mrcr grade() as its dataset card gives it: 0 unless the response starts with the string, then
    the SequenceMatcher ratio of the two texts with the string removed from the front."""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return difflib.SequenceMatcher(None, response, answer).ratio()


TASK_LABELS = {
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
}


def _babilong_preprocess(output):
    output = output.lower()
    output = output.split(".")[0]          # only the first sentence
    output = output.split("<context>")[0]  # not an invented continuation
    output = output.split("<example>")[0]
    output = output.split("Question")[0]
    return output


def babilong_correct(target, output, question, task_labels):
    """booydar/babilong babilong/metrics.py compare_answers, verbatim in logic: the labels named in the first sentence
    of the output, minus those named in the question, must be exactly the target."""
    output = _babilong_preprocess(output)
    target = target.lower()
    task_labels = {label.lower() for label in task_labels}
    labels_in_question = {label for label in task_labels if label in question.lower()}
    labels_in_output = {label for label in task_labels if label in output}
    labels_in_output = labels_in_output - labels_in_question
    if "," in target and len(target) > 3:
        subtargets = target.split(",")
        return all(t in labels_in_output for t in subtargets) and len(labels_in_output) == len(subtargets)
    return target in labels_in_output and len(labels_in_output) == 1


# ---------------------------------------------------------------------------------------------------------------
# MRCR plan: the published transcript, fitted to our token count by dropping filler turns only
# ---------------------------------------------------------------------------------------------------------------

FILLER_TOPICS = {"black holes", "dreams", "Cambrian Explosion", "The Renaissance", "sign language", "dystopia"}
INDEX_NAMES = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
EXAMPLE3 = "======== EXAMPLE 3 ========\n\n"
_BLOCK = re.compile(r"User: Write an? (.+?) about (.+?) in (.+?) style\.\n\nAssistant: ", re.S)
_FINAL = re.compile(r"User: Prepend (\w{12}) to the (\w+) (.+?) about (.+?) in an? (.+?) style\.", re.S)


def split_transcript(transcript):
    """(head, blocks): the few-shot prefix through '======== EXAMPLE 3 ========' and the transcript's turn pairs, each
    'User: ...\\n\\nAssistant: ...\\n\\n' (head + ''.join(blocks) == transcript)."""
    i = transcript.rfind(EXAMPLE3)
    if i < 0:
        raise ValueError("no '======== EXAMPLE 3 ========' marker: not an MRCR v2 text_style transcript")
    cut = i + len(EXAMPLE3)
    head, body = transcript[:cut], transcript[cut:]
    blocks = [b for b in re.split(r"(?m)^(?=User: )", body) if b]
    return head, blocks


def block_topic(block):
    m = _BLOCK.match(block)
    if not m:
        raise ValueError("a transcript turn does not look like 'User: Write a F about T in S style.': " + block[:80])
    return m.group(1), m.group(2), m.group(3)


def drop_order(n):
    """Block positions 0..n-1 in an order whose every prefix is spread evenly (golden-ratio sequence)."""
    seen, order, j = set(), [], 0
    phi = 0.6180339887498949
    while len(order) < n and j < 20 * n + 100:
        k = int(((j * phi) % 1.0) * n)
        j += 1
        if k not in seen:
            seen.add(k)
            order.append(k)
    for k in range(n):          # any position the sequence missed
        if k not in seen:
            order.append(k)
    return order


def fit_transcript(transcript, drop_chars):
    """Drop whole FILLER turn pairs (topics in FILLER_TOPICS), spread evenly, until at least drop_chars characters are
    gone. Relevant turns are never touched, so the order of each (format, topic, style) group's instances, and so the
    meaning of 'the third riddle about ...', is unchanged. Returns (new transcript, number of turns dropped)."""
    if drop_chars <= 0:
        return transcript, 0
    head, blocks = split_transcript(transcript)
    filler = [i for i, b in enumerate(blocks) if block_topic(b)[1] in FILLER_TOPICS]
    gone, dropped = 0, set()
    for k in drop_order(len(filler)):
        if gone >= drop_chars:
            break
        dropped.add(filler[k])
        gone += len(blocks[filler[k]])
    if gone < drop_chars:
        have = sum(len(blocks[i]) for i in filler)
        raise ValueError(f"cannot drop {drop_chars:,} characters: all {len(filler)} filler turns hold {have:,}")
    return head + "".join(b for i, b in enumerate(blocks) if i not in dropped), len(dropped)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mrcr_samples(rows, n, seed):
    """n row indexes spread over the needle groups: groups in a seeded order, one random row per group per pass."""
    rng = random.Random(seed)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["random_string"], []).append(i)
    keys = sorted(groups)
    rng.shuffle(keys)
    pools = {k: rng.sample(v, len(v)) for k, v in groups.items()}
    out = []
    while len(out) < n and any(pools.values()):
        for k in keys:
            if pools[k] and len(out) < n:
                out.append(pools[k].pop())
    return out


def verify_mrcr_samples(prefix, samples):
    """Check on the transcript AS SENT that every question still has its answer: the group the question names has all its
    instances (the file's needle count) and the n-th one, in order of appearance, is the expected text. Returns the list
    of failing sample ids (empty = all good)."""
    _, blocks = split_transcript(prefix)
    groups = {}
    for b in blocks:
        groups.setdefault(block_topic(b), []).append(b)
    bad = []
    for s in samples:
        m = _FINAL.match(s["tail"])
        if not m:
            bad.append(s["id"])
            continue
        h, ordn, fmt, topic, style = m.groups()
        inst = groups.get((fmt, topic, style), [])
        i = INDEX_NAMES.index(ordn) if ordn in INDEX_NAMES else 99
        text = inst[i].split("Assistant: ", 1)[1].rstrip("\n") if i < len(inst) else None
        if (text is None or s["expected"][:12] != h or s["expected"][12:].strip() != text.strip()
                or len(inst) != s["needles"]):
            bad.append(s["id"])
    return bad


def to_chat_messages(prompt):
    """The same MRCR prompt as real alternating chat turns (--render chat): the few-shot preface joins the first user
    turn, every 'User: ...' / 'Assistant: ...' pair becomes a user and an assistant message, and the final question
    ('Prepend ... in your response.') is the last user message with nothing after it."""
    i = prompt.rfind("User: Prepend")
    head, blocks = split_transcript(prompt[:i])
    final = prompt[i + len("User: "):].rsplit("\n\nAssistant:", 1)[0]
    msgs = []
    for k, b in enumerate(blocks):
        u, a = b.split("\n\nAssistant: ", 1)
        u = u[len("User: "):]
        msgs.append({"role": "user", "content": head + u if k == 0 else u})
        msgs.append({"role": "assistant", "content": a.rstrip("\n")})
    msgs.append({"role": "user", "content": final})
    return msgs


class Plan:
    """One (suite, task, length): a prefix shared byte for byte by every sample, and a short per-sample tail."""

    def __init__(self, suite, task, length, target_tokens, prefix, samples, cpt, notes):
        self.suite, self.task, self.length, self.target_tokens = suite, task, length, target_tokens
        self.prefix, self.samples, self.cpt, self.notes = prefix, samples, cpt, notes

    def prompt(self, sample):
        if "messages" in sample:         # openai/mrcr: every sample is its own conversation (no shared prefix)
            return sample["messages"]
        return self.prefix + sample["tail"]

    @property
    def size_chars(self):
        if self.prefix:
            return len(self.prefix)
        return int(sum(s["n_chars"] for s in self.samples) / max(1, len(self.samples)))

    @property
    def est_prompt_tokens(self):
        return int(self.size_chars / self.cpt)


def mrcr_calibration_texts(length, data_dir):
    doc = load_json(os.path.join(data_dir, "mrcr", f"{length}.json"))
    return spread_slices(doc["transcript"])


def spread_slices(text, n=N_SLICES, size=SLICE_CHARS):
    if len(text) <= size:
        return [text]
    return [text[int(i * (len(text) - size) / max(1, n - 1)):][:size] for i in range(n)]


def make_mrcr_plan(length, samples, seed, cpt, data_dir, target_tokens=None, drop_chars=None):
    doc = load_json(os.path.join(data_dir, "mrcr", f"{length}.json"))
    target = target_tokens or LENGTHS[split_label(length)[0]]
    transcript = doc["transcript"]
    if drop_chars is None:
        drop_chars = max(0, len(transcript) - int(target * cpt))
    try:
        fitted, ndrop = fit_transcript(transcript, drop_chars)
    except ValueError:
        if target >= 500_000:
            raise
        # a short transcript may hold too little filler to reach the target (32k has none): send it as published,
        # up to ~3 % over the target, far inside any window; the row's prompt_tokens says by how much
        head_, blocks_ = split_transcript(transcript)
        allf = sum(len(x) for x in blocks_ if block_topic(x)[1] in FILLER_TOPICS)
        fitted, ndrop = fit_transcript(transcript, min(drop_chars, allf))
    idx = mrcr_samples(doc["rows"], samples, seed)
    out = []
    for i in idx:
        r = doc["rows"][i]
        m = _FINAL.match(r["final_query"])
        out.append({"id": f"{length}-r{i}", "question": r["final_query"].strip(), "expected": r["answer"],
                    "tail": r["final_query"], "group": r["random_string"],
                    "ordinal": m.group(2) if m else "", "needles": r["num_relevant"]})
    bad = verify_mrcr_samples(fitted, out)
    if bad:
        raise ValueError(f"{len(bad)} of {len(out)} questions no longer match their n-th instance in the transcript to send: {bad[:5]}")
    notes = {"source": doc["source"], "published_chars": len(transcript), "answers_verified_on_sent_transcript": len(out), "dropped_chars": len(transcript) - len(fitted),
             "dropped_turns": ndrop, "drop_chars_requested": drop_chars}
    return Plan("mrcr", f"{doc['n_needles']}needle", length, target, fitted, out, cpt, notes)


def omrcr_path(length, data_dir):
    base, needles = split_label(length)
    if base != "1m":
        raise ValueError("openai/mrcr data is kept for its largest bin only (length 1m, optionally 1m-2n)")
    return os.path.join(data_dir, "openai", f"{needles}needle-1m.jsonl")


def omrcr_rows(length, data_dir):
    with open(omrcr_path(length, data_dir), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def omrcr_calibration_texts(length, data_dir):
    rows = omrcr_rows(length, data_dir)
    return spread_slices("\n\n".join(m["content"] for m in rows[0]["messages"]))


def make_omrcr_plan(length, samples, seed, cpt, data_dir, target_tokens=None):
    """openai/mrcr, largest bin, as OpenAI's own snippet runs it: the stored messages sent as they are, and a sample
    whose prompt is over the limit skipped (OpenAI's snippet skips prompts over its window; here the limit is OUR token
    count: target tokens x the calibrated chars per token). Samples are drawn in a seeded order from those that fit."""
    target = target_tokens or LENGTHS[split_label(length)[0]]
    rows = omrcr_rows(length, data_dir)
    limit = int(target * cpt)
    fit = [r for r in rows if r["n_chars"] <= limit]
    random.Random(seed).shuffle(fit)
    out = []
    for r in fit[:samples]:
        msgs = r["messages"]
        out.append({"id": f"{length}-{r['id']}", "question": msgs[-1]["content"], "expected": r["answer"],
                    "messages": msgs, "n_chars": r["n_chars"], "random_string": r["random_string"],
                    "needles": r["n_needles"], "group": r["random_string"], "tail": ""})
    notes = {"source": "openai/mrcr (largest bin; 4/12/2025 data with the 12/5/2025 bugfix rows)",
             "candidates": len(rows), "fitting_the_target": len(fit), "samples": len(out),
             "note": "every sample is its own conversation: one cold read per sample"}
    needles = rows[0]["n_needles"]
    return Plan("omrcr", f"{needles}needle", length, target, "", out, cpt, notes)


# ---------------------------------------------------------------------------------------------------------------
# BABILong plan: the facts of many bAbI stories (renamed, so they do not collide) hidden in ONE PG19 haystack
# ---------------------------------------------------------------------------------------------------------------

TEMPLATE = "{instruction}\n\n{examples}\n\n{post_prompt}\n\n<context>\n{context}\n</context>\n\nQuestion: "
_SYL1 = ["Bar", "Cor", "Dal", "Fen", "Gor", "Hal", "Jor", "Kel", "Lor", "Mar", "Nor", "Pel", "Quin", "Ros", "Sar", "Tor",
         "Vel", "Wen", "Yar", "Zan"]
_SYL2 = ["bek", "dric", "fold", "gard", "lin", "mund", "nek", "ric", "sten", "thos", "vik", "wald", "zim", "bram"]
_OBJECTS = ("lantern kettle compass saddle anvil trumpet ladder basket mirror hammer candle bucket violin barrel "
            "umbrella telescope helmet scarf blanket pillow towel clock locket shovel teapot wallet whistle yoyo "
            "backpack banjo bicycle camera crayon drum feather flute globe hatchet journal kayak lamp mallet "
            "necklace oar paddle quilt rake sandal tambourine trowel vase wrench anchor bell cushion dagger easel "
            "flask goblet harp inkwell jar kite lute magnet needle ocarina parasol quiver ribbon spoon thimble urn "
            "vest wagon xylophone zither mug plate crown boot glove brush comb fork stool ladle sled spade tray "
            "pitcher rope drumstick trophy sword shield bottle notebook radio ticket suitcase").split()
_PEOPLE = ["Mary", "John", "Sandra", "Daniel"]
_THINGS = ["football", "apple", "milk"]


def sentences_of(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip()) if s]


def load_noise(data_dir, seed, min_chars):
    """PG19 test-book sentences, books in a seeded order, until min_chars characters (cycling if the books run out)."""
    d = os.path.join(data_dir, "babilong", "pg19_test")
    books = sorted(f for f in os.listdir(d) if f.endswith(".txt"))
    if not books:
        raise ValueError(f"no PG19 books in {d}: run prepare_data.py babilong")
    random.Random(seed).shuffle(books)
    out, total, i = [], 0, 0
    while total < min_chars:
        with open(os.path.join(d, books[i % len(books)]), encoding="utf-8", errors="replace") as f:
            sents = sentences_of(f.read())
        out.extend(sents)
        total += sum(len(s) + 1 for s in sents)
        i += 1
        if i > 10 * len(books) + 10:
            raise ValueError("the PG19 books are too short for the requested length")
    return out


def babilong_calibration_texts(data_dir, seed):
    sents = load_noise(data_dir, seed, N_SLICES * SLICE_CHARS * 8)
    return spread_slices(" ".join(sents))


def story_facts(inp):
    return [s.strip() for s in re.findall(r"[^.]+\.", inp) if s.strip()]


def unique_names(n, rng, forbidden_lower):
    pool = [a + b for a in _SYL1 for b in _SYL2]
    rng.shuffle(pool)
    pool = [x for x in pool if x.lower() not in forbidden_lower]
    if len(pool) < n:
        raise ValueError(f"need {n} unique names, have {len(pool)}")
    return pool[:n]


def make_babilong_plan(task, length, samples, seed, cpt, data_dir, target_tokens=None):
    target = target_tokens or LENGTHS[length]
    prompts = load_json(os.path.join(data_dir, "babilong", "prompts_qa2_qa3.json"))[task]
    stories = load_json(os.path.join(data_dir, "babilong", f"{task}_0k.json"))
    rng = random.Random(f"{seed}/{task}/{length}")
    chosen = rng.sample(range(len(stories)), min(samples, len(stories)))
    target_chars = int(target * cpt)
    sents = load_noise(data_dir, seed, target_chars)
    forbidden = {w for w in re.findall(r"[a-z']+", " ".join(sents).lower())}
    names = unique_names(4 * len(chosen), rng, forbidden | {p.lower() for p in _PEOPLE})
    things = list(_OBJECTS)
    rng.shuffle(things)
    if 3 * len(chosen) > len(things):
        raise ValueError(f"need {3 * len(chosen)} unique objects, have {len(things)}")
    out = []
    for si, k in enumerate(chosen):
        st = stories[k]
        mp = {**dict(zip(_PEOPLE, names[4 * si:4 * si + 4])), **dict(zip(_THINGS, things[3 * si:3 * si + 3]))}

        def ren(t, mp=mp):
            return re.sub(r"\b(" + "|".join(mp) + r")\b", lambda m: mp[m.group(1)], t)
        facts = [ren(f) for f in story_facts(st["input"])]
        q = ren(st["question"]).strip()
        out.append({"id": f"{task}-{length}-s{k}", "question": q, "expected": st["target"], "tail": q,
                    "story": k, "n_facts": len(facts), "facts": facts})
    # the haystack is cut so that haystack + facts + the instructions come to the target, not the haystack alone
    fixed = len(TEMPLATE.format(context="", **prompts)) + sum(len(f) + 1 for o in out for f in o["facts"]) + 100
    kept, total = [], 0
    for s_ in sents:
        if total + len(s_) + 1 + fixed > target_chars:
            break
        kept.append(s_)
        total += len(s_) + 1
    inserts = {}     # sentence index -> facts (kept in story order)
    for o in out:
        pos = sorted(rng.randrange(len(kept) + 1) for _ in o["facts"])
        for p, f in zip(pos, o["facts"]):
            inserts.setdefault(p, []).append(f)
    parts = []
    for i in range(len(kept) + 1):
        parts.extend(inserts.get(i, []))
        if i < len(kept):
            parts.append(kept[i])
    context = " ".join(parts)
    prefix = TEMPLATE.format(context=context.strip(), **prompts)
    notes = {"noise_chars": total, "stories": len(chosen), "facts": sum(o["n_facts"] for o in out)}
    return Plan("babilong", task, length, target, prefix, out, cpt, notes)


# ---------------------------------------------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------------------------------------------


def first_key(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line.split(":", 1)[1].strip() if ":" in line else line
    sys.exit(f"{path} holds no key")


def find_key(key_file=None):
    """--key-file, else ENGINE_API_KEY (tools/bench.py), else API_KEY, else the first key of API_KEY_FILE or the
    default key file (tools/needle_bench.py). Returns "" when there is none."""
    if key_file:
        return first_key(key_file)
    for name in ("ENGINE_API_KEY", "API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    path = os.environ.get("API_KEY_FILE", os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key"))
    try:
        return first_key(path)
    except (OSError, SystemExit):
        return ""


def request_body(model, prompt, mode, max_tokens, temperature, top_p):
    m = MODES[mode]
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": m["thinking"]}}
    if m["effort"]:
        body["reasoning_effort"] = m["effort"]
    return body


def stream_chat(url, key, body, timeout):
    """One streamed completion. Returns a dict: content, reasoning_chars, usage, finish_reason, ttft, total, error.
    `timeout` is the socket's: the longest silence tolerated (a cold 4M-token read is 21+ minutes before token one)."""
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.time()
    res = {"content": "", "reasoning_chars": 0, "reasoning": "", "usage": None, "finish_reason": None, "ttft": None, "error": None}
    content = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if isinstance(ev, dict) and ev.get("error"):
                    res["error"] = "stream error: " + json.dumps(ev["error"])[:300]
                    break
                if ev.get("usage"):
                    res["usage"] = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    r = d.get("reasoning_content") or d.get("reasoning") or ""
                    c = d.get("content") or ""
                    if (r or c) and res["ttft"] is None:
                        res["ttft"] = time.time() - t0
                    res["reasoning_chars"] += len(r)
                    res["reasoning"] += r
                    if c:
                        content.append(c)
                    if ch.get("finish_reason"):
                        res["finish_reason"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        res["error"] = f"HTTP {e.code}: " + e.read().decode("utf-8", "replace")[:300]
    except (urllib.error.URLError, OSError, ValueError) as e:
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    res["content"] = "".join(content)
    res["total"] = time.time() - t0
    return res


def calibrate(url, key, model, texts, timeout):
    """Characters per token of THIS text on THIS server: each slice as a one-token request, tokens from usage (the
    sizing method of tools/needle_bench.py: ask the live server, no local tokenizer). Returns (cpt, detail)."""
    chars = toks = 0
    for t in texts:
        body = request_body(model, t, "off", 1, 0.0, 1.0)
        r = stream_chat(url, key, body, timeout)
        n = (r["usage"] or {}).get("prompt_tokens")
        if r["error"] or not isinstance(n, int) or n <= 0:
            raise RuntimeError("calibration request failed: " + str(r["error"] or r["usage"]))
        chars += len(t)
        toks += n
    return chars / toks, {"slices": len(texts), "chars": chars, "tokens": toks}


# ---------------------------------------------------------------------------------------------------------------
# time estimates (ESTIMATES from this repository's measured prompt and decode speeds; see README)
# ---------------------------------------------------------------------------------------------------------------

COLD_READ_S = {"32k": 10, "128k": 40, "256k": 75, "1m": 156, "2m": 420, "4m": 1260}   # assumed cold reads (1M and 4M as measured)
WARM_TAIL_S = 7.2                                              # a 4M prompt re-asked with a new last question
ASSUMED_MEAN_COMPLETION = {"mrcr": {"off": 500, "high": 4000, "max": 12000},
                           "omrcr": {"off": 500, "high": 4000, "max": 12000},
                           "babilong": {"off": 25, "high": 3000, "max": 8000}}


def decode_speed(length):
    """tok/s: 35 short, 15.5 at 4M, log-linear in between (an interpolation, not a measurement)."""
    t = LENGTHS[split_label(length)[0]]
    if t <= 128_000:
        return 35.0
    return 35.0 - (35.0 - 15.5) * math.log2(t / 128_000) / math.log2(4_000_000 / 128_000)


def estimate(suite, length, modes, samples, tasks=1, shared_cache=True, cold_each=False):
    """Seconds: (assumed-mean completion, every request at its max_tokens budget)."""
    sp = decode_speed(length)
    n_req = samples * len(modes) * tasks
    cold = COLD_READ_S[split_label(length)[0]] * tasks * (1 if shared_cache else len(modes))
    if cold_each:          # every sample is a different conversation: it is read cold, once, then the other modes re-ask it
        cold = COLD_READ_S[split_label(length)[0]] * samples
    mean = cold + n_req * WARM_TAIL_S + sum(samples * tasks * ASSUMED_MEAN_COMPLETION[suite][m] / sp for m in modes)
    worst = cold + n_req * WARM_TAIL_S + sum(samples * tasks * MAX_TOKENS[suite][m] / sp for m in modes)
    return mean, worst


def hms(s):
    s = int(round(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------


def read_rows(path):
    plans, rows = {}, []
    if not path or not os.path.exists(path):
        return plans, rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("kind") == "plan":
                plans[(d["suite"], d["task"], d["length"])] = d
            elif d.get("kind") == "result":
                rows.append(d)
    return plans, rows


def append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def tasks_for(suite, a, length="1m"):
    return [f"{split_label(length)[1]}needle"] if suite in ("mrcr", "omrcr") else [t.strip() for t in a.tasks.split(",")]


def get_plan(a, suite, task, length, key, old_plan):
    """The plan of this (suite, task, length). On --resume the recorded cpt and drop are reused, so a resumed run
    sends byte-identical prompts (and the server's kept prefix still serves them)."""
    target = a.target_tokens.get(length) if a.target_tokens else None
    if old_plan:
        cpt = old_plan["chars_per_token"]
        calib = {"reused": True}
    elif a.dry_run:
        cpt, calib = a.assume_cpt, {"assumed": a.assume_cpt}
    else:
        texts = (mrcr_calibration_texts(length, a.data) if suite == "mrcr" else
                 omrcr_calibration_texts(length, a.data) if suite == "omrcr" else babilong_calibration_texts(a.data, a.seed))
        cpt, calib = calibrate(a.url, key, a.model, texts, a.timeout)
    if suite == "mrcr":
        plan = make_mrcr_plan(length, a.samples, a.seed, cpt, a.data, target,
                              old_plan["notes"]["drop_chars_requested"] if old_plan else None)
    elif suite == "omrcr":
        plan = make_omrcr_plan(length, a.samples, a.seed, cpt, a.data, target)
    else:
        plan = make_babilong_plan(task, length, a.samples, a.seed, cpt, a.data, target)
    return plan, calib


def score_row(suite, task, sample, reply):
    if suite == "omrcr":       # OpenAI's grade() on the reply exactly as it came (no strip): the string must lead
        sc = round(openai_mrcr_score(reply or "", sample["expected"], sample["random_string"]), 6)
        return sc, sc
    if suite == "mrcr":
        return round(mrcr_score(reply, sample["expected"]), 6), round(mrcr_score_strict(reply, sample["expected"]), 6)
    ok = bool(reply) and babilong_correct(sample["expected"], reply, sample["question"], TASK_LABELS[task])
    return (1.0 if ok else 0.0), (1.0 if ok else 0.0)


def cmd_run(a):
    suites = [a.suite]
    modes = [m.strip() for m in a.modes.split(",")]
    lengths = [x.strip().lower() for x in a.lengths.split(",")]
    for m in modes:
        if m not in MODES:
            sys.exit(f"--modes takes off,high,max; got {m!r}")
    for x in lengths:
        try:
            split_label(x)
        except ValueError as e:
            sys.exit(str(e))
    a.target_tokens = dict(kv.split("=") for kv in a.target.split(",")) if a.target else {}
    a.target_tokens = {k: int(v) for k, v in a.target_tokens.items()}
    key = "" if a.dry_run else find_key(a.key_file)
    old_plans, old_rows = read_rows(a.out) if a.resume else ({}, [])
    done = {(r["suite"], r["task"], r["length"], r["mode"], r["sample_id"]) for r in old_rows if not r.get("error")}
    budget = {m: a.max_tokens.get(m, MAX_TOKENS[a.suite][m]) for m in modes}
    total = 0.0
    for suite in suites:
        for length in lengths:
            for task in tasks_for(suite, a, length):
                plan, calib = get_plan(a, suite, task, length, key, old_plans.get((suite, task, length)))
                pending = [(s, m) for s in plan.samples for m in modes
                           if (suite, task, length, m, s["id"]) not in done]
                mean, worst = estimate(suite, length, modes, len(plan.samples), shared_cache=True,
                                       cold_each=(suite == "omrcr"))
                print(f"== {suite}/{task} {length}: target {plan.target_tokens:,} tokens, {'prefix' if plan.prefix else 'mean prompt'} {plan.size_chars:,} chars "
                      f"= ~{plan.est_prompt_tokens:,} tokens at {plan.cpt:.3f} chars/token "
                      f"({'assumed' if a.dry_run else 'calibrated' if 'slices' in calib else 'reused from --out'}); "
                      f"{len(plan.samples)} samples x {len(modes)} modes = {len(plan.samples) * len(modes)} requests, "
                      f"{len(pending)} to send", flush=True)
                print(f"   max_tokens {budget}; notes {plan.notes}", flush=True)
                print(f"   time estimate {hms(mean)} at assumed mean completions, {hms(worst)} if every reply hits its "
                      f"max_tokens (ESTIMATES)", flush=True)
                total += mean
                if a.dry_run:
                    for s in plan.samples[:3]:
                        print(f"   e.g. {s['id']}: question {s['question'][:110]!r} expected {s['expected'][:40]!r}")
                    continue
                if not a.resume or (suite, task, length) not in old_plans:
                    append(a.out, {"kind": "plan", "suite": suite, "task": task, "length": length,
                                   "chars_per_token": plan.cpt, "target_tokens": plan.target_tokens, "notes": plan.notes,
                                   "calibration": calib, "seed": a.seed, "samples": [s["id"] for s in plan.samples],
                                   "at": time.strftime("%F %T")})
                for n, (s, m) in enumerate(pending, 1):
                    prompt = plan.prompt(s)
                    if a.render == "chat" and suite == "mrcr":
                        prompt = to_chat_messages(prompt)
                    body = request_body(a.model, prompt, m, budget[m], a.temperature, a.top_p)
                    r = stream_chat(a.url, key, body, a.timeout)
                    usage = r["usage"] or {}
                    reply = r["content"]
                    sc, strict = score_row(suite, task, s, reply) if not r["error"] else (None, None)
                    row = {"kind": "result", "suite": suite, "task": task, "length": length,
                           "target_tokens": plan.target_tokens, "prompt_tokens": usage.get("prompt_tokens"),
                           "mode": m, "sample_id": s["id"], "question": s["question"], "expected": s["expected"],
                           "answer": reply, "reasoning_chars": r["reasoning_chars"], "reasoning": r.get("reasoning", ""),
                           "completion_tokens": usage.get("completion_tokens"), "score": sc, "score_strict": strict,
                           "first_token_s": None if r["ttft"] is None else round(r["ttft"], 2),
                           "total_s": round(r["total"], 2), "finish_reason": r["finish_reason"], "error": r["error"],
                           "max_tokens": budget[m], "render": a.render if suite == "mrcr" else "text", "temperature": a.temperature, "top_p": a.top_p,
                           "model": a.model, "at": time.strftime("%F %T")}
                    append(a.out, row)
                    print(f"[{n}/{len(pending)}] {s['id']} {m}: score {sc} prompt_tokens {row['prompt_tokens']} "
                          f"completion {row['completion_tokens']} ttft {row['first_token_s']}s total {row['total_s']}s "
                          f"finish {row['finish_reason']}{' ERROR ' + r['error'] if r['error'] else ''}", flush=True)
                del plan
    if a.dry_run:
        print(f"\ndry run: nothing sent. Total estimate {hms(total)} (assumed-mean completions).")


# ---------------------------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------------------------

MRCR_NOISE = {"2needle": "51", "4needle": "27", "8needle": "15"}   # eval_hub/mrcr_v2/README.md, "NOISE RATES"


def chance(suite, task):
    if suite in ("mrcr", "omrcr"):
        return f"{MRCR_NOISE[task]} (random relevant needle; ~1 any reply)"
    return f"{100 / len(TASK_LABELS[task]):.1f} (1/{len(TASK_LABELS[task])} locations)"


def cmd_summary(a):
    rows = []
    for p in a.files:
        rows += read_rows(p)[1]
    groups = {}
    for r in rows:
        groups.setdefault((r["suite"], r["task"], r["length"], r["mode"]), []).append(r)
    order = {"128k": 0, "1m": 1, "2m": 2, "4m": 3}
    mo = {"off": 0, "high": 1, "max": 2}
    print(f"{'suite':9} {'task':8} {'len':4} {'mode':5} {'n':>3} {'score%':>7} {'+-se':>5} {'strict%':>7} {'err':>3} {'cut':>3} "
          f"{'think-tok':>9} {'ttft-med':>8} {'chance%'}")
    for k in sorted(groups, key=lambda k: (k[0], k[1], order.get(k[2], 9), mo.get(k[3], 9))):
        g = groups[k]
        ok = [r for r in g if not r.get("error") and r.get("score") is not None]
        sc = [r["score"] for r in ok]
        n = len(sc)
        mean = 100 * sum(sc) / n if n else float("nan")
        se = 100 * (math.sqrt(sum((x - sum(sc) / n) ** 2 for x in sc) / (n - 1)) / math.sqrt(n)) if n > 1 else float("nan")
        strict = 100 * sum(r["score_strict"] for r in ok) / n if n else float("nan")
        cut = sum(1 for r in ok if r.get("finish_reason") == "length")
        toks = [r["completion_tokens"] for r in ok if isinstance(r.get("completion_tokens"), int)]
        ttft = sorted(r["first_token_s"] for r in ok if r.get("first_token_s") is not None)
        print(f"{k[0]:9} {k[1]:8} {k[2]:4} {k[3]:5} {n:>3} {mean:>7.1f} {se:>5.1f} {strict:>7.1f} {len(g) - n:>3} {cut:>3} "
              f"{(sum(toks) / len(toks) if toks else float('nan')):>9.0f} {(ttft[len(ttft) // 2] if ttft else float('nan')):>8.1f} "
              f"{chance(k[0], k[1])}")
    print("\nscore% = mean of the suite's per-sample score (MRCR: match ratio, the published MMR; BABILong: exact label); "
          "strict% = MRCR needing the random string first; err = requests that failed; cut = replies stopped by "
          "max_tokens; think-tok = mean completion tokens (reasoning included).")


def cmd_estimate(a):
    modes = [m.strip() for m in a.modes.split(",")]
    print(f"{'suite':9} {'len':4} {'modes':14} {'samples':>7} {'cold read':>9} {'decode tok/s':>12} {'assumed-mean':>13} {'all at budget':>13}")
    for length in [x.strip().lower() for x in a.lengths.split(",")]:
        tasks = 1 if a.suite == "mrcr" else len(a.tasks.split(","))
        mean, worst = estimate(a.suite, length, modes, a.samples, tasks)
        print(f"{a.suite:9} {length:4} {','.join(modes):14} {a.samples:>7} {hms(COLD_READ_S[split_label(length)[0]] * tasks):>9} "
              f"{decode_speed(length):>12.1f} {hms(mean):>13} {hms(worst):>13}")


# ---------------------------------------------------------------------------------------------------------------
# verify-data
# ---------------------------------------------------------------------------------------------------------------


def verify_mrcr(path, length):
    doc = load_json(path)
    head, blocks = split_transcript(doc["transcript"])
    assert head + "".join(blocks) == doc["transcript"], "split_transcript lost characters"
    groups = {}
    for b in blocks:
        fmt, topic, style = block_topic(b)
        groups.setdefault((fmt, topic, style), []).append(b)
    bad = 0
    for r in doc["rows"]:
        m = _FINAL.match(r["final_query"])
        if not m:
            bad += 1
            continue
        h, ordn, fmt, topic, style = m.groups()
        inst = groups.get((fmt, topic, style), [])
        want = r["answer"][12:]
        got = inst[INDEX_NAMES.index(ordn)] if len(inst) > INDEX_NAMES.index(ordn) else ""
        text = got.split("Assistant: ", 1)[-1].rstrip("\n")
        if r["answer"][:12] != h or text.strip() != want.strip() or len(inst) != r["num_relevant"]:
            bad += 1
    nfill = sum(1 for b in blocks if block_topic(b)[1] in FILLER_TOPICS)
    print(f"mrcr {length}: transcript {len(doc['transcript']):,} chars, {len(blocks)} turns ({nfill} filler), "
          f"{len(doc['rows'])} rows, answers matching the n-th instance of their group: {len(doc['rows']) - bad}/{len(doc['rows'])}")
    return bad == 0


def verify_omrcr_row(r):
    """True when the stored answer is the k-th instance of the asked request in the stored conversation, k from the
    final question ('the 3rd (1 indexed) ...'), the conversation holds exactly n_needles instances, and the last message
    is that question, last."""
    m = r["messages"]
    q = re.match(r"Prepend (\w+) to the (\d+)(?:st|nd|rd|th) \(1 indexed\) ", m[-1]["content"])
    if not q or m[-1]["role"] != "user" or q.group(1) != r["random_string"]:
        return False
    k = int(q.group(2))
    req = m[r["desired_msg_index"]]
    inst = [j for j, x in enumerate(m[:-1]) if x["role"] == "user" and x["content"] == req["content"]]
    if len(inst) != r["n_needles"] or k > len(inst) or inst[k - 1] != r["desired_msg_index"]:
        return False
    return (m[inst[k - 1] + 1]["role"] == "assistant"
            and m[inst[k - 1] + 1]["content"].strip() == r["answer"][len(r["random_string"]):].strip()
            and r["answer"].startswith(r["random_string"]))


def verify_omrcr(path, n):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    good = sum(verify_omrcr_row(r) for r in rows)
    print(f"openai/mrcr {n}-needle largest bin: {len(rows)} samples, {min(r['n_chars'] for r in rows):,}-"
          f"{max(r['n_chars'] for r in rows):,} chars, answer = k-th instance of the asked request: {good}/{len(rows)}")
    return good == len(rows)


def cmd_verify(a):
    ok = True
    d = os.path.join(a.data, "mrcr")
    files = sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []
    for length in files:
        p = os.path.join(d, f"{length}.json")
        ok &= verify_mrcr(p, length)
        n = len(load_json(p)["rows"])
        try:       # the same check on the transcript AS CUT for the target at --cpt, every stored row
            plan = R_plan = make_mrcr_plan(length, n, DEFAULT_SEED, a.cpt, a.data)
            print(f"   cut at {a.cpt} chars/token for {plan.target_tokens:,} tokens: dropped {plan.notes['dropped_turns']} filler "
                  f"turns / {plan.notes['dropped_chars']:,} chars; all {plan.notes['answers_verified_on_sent_transcript']} "
                  f"answers are still the n-th instance IN THE CUT TRANSCRIPT")
            del plan, R_plan
        except ValueError as e:
            print(f"   cut at {a.cpt} chars/token: FAILED: {e}")
            ok = False
    if not files:
        print("no MRCR data (prepare_data.py mrcr)")
    for n in (2, 4, 8):
        p = os.path.join(a.data, "openai", f"{n}needle-1m.jsonl")
        if os.path.exists(p):
            ok &= verify_omrcr(p, n)
    for task in ("qa2", "qa3"):
        p = os.path.join(a.data, "babilong", f"{task}_0k.json")
        if os.path.exists(p):
            st = load_json(p)
            targets = {s["target"] for s in st}
            print(f"babilong {task}: {len(st)} stories, targets {sorted(targets)}, all labelled: "
                  f"{targets <= set(TASK_LABELS[task])}")
            ok &= targets <= set(TASK_LABELS[task])
        else:
            print(f"babilong {task}: {p} missing (prepare_data.py babilong)")
    d = os.path.join(a.data, "babilong", "pg19_test")
    print("pg19 books:", len([f for f in os.listdir(d) if f.endswith('.txt')]) if os.path.isdir(d) else "missing")
    sys.exit(0 if ok else 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="send the requests (or --dry-run)")
    r.add_argument("--url", default="http://localhost:8000")
    r.add_argument("--model", default="deepseek-v4.1-flash-1017")
    r.add_argument("--suite", choices=["mrcr", "omrcr", "babilong"], required=True)
    r.add_argument("--lengths", default="1m")
    r.add_argument("--modes", default="off,high,max")
    r.add_argument("--samples", type=int, default=20)
    r.add_argument("--tasks", default="qa2,qa3", help="babilong tasks")
    r.add_argument("--out", required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--timeout", type=float, default=7200, help="socket silence tolerated, seconds (cold 4M read: 1260+)")
    r.add_argument("--key-file")
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r.add_argument("--target", help="override target prompt tokens, e.g. 1m=900000,4m=3900000")
    r.add_argument("--max-tokens", type=json.loads, default={}, help='JSON per mode, e.g. {"max": 20000}')
    r.add_argument("--assume-cpt", type=float, default=ASSUMED_CPT, help="chars/token for --dry-run only")
    r.add_argument("--render", choices=["text", "chat"], default="text",
                   help="MRCR: text = the published single user message (default); chat = real alternating user/assistant turns")
    r.add_argument("--data", default=DATA)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    e = sub.add_parser("estimate")
    e.add_argument("--suite", choices=["mrcr", "omrcr", "babilong"], default="mrcr")
    e.add_argument("--lengths", default="1m,2m,4m")
    e.add_argument("--modes", default="off,high,max")
    e.add_argument("--samples", type=int, default=20)
    e.add_argument("--tasks", default="qa2,qa3")
    v = sub.add_parser("verify-data")
    v.add_argument("--data", default=DATA)
    v.add_argument("--cpt", type=float, default=4.7, help="chars/token assumed for the cut check")
    a = p.parse_args(argv)
    {"run": cmd_run, "summary": cmd_summary, "estimate": cmd_estimate, "verify-data": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
