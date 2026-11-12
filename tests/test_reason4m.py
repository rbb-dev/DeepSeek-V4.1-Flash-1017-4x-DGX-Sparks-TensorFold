"""Offline tests: python3 -I -m unittest discover -s tests -v   (from the workspace). No real server is contacted."""
import contextlib
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "reason4m")
sys.path.insert(0, ROOT)
import reason4m as R  # noqa: E402
from fake_server import FakeServer  # noqa: E402

REAL = os.path.join(ROOT, "data")
BABI = os.path.exists(os.path.join(REAL, "babilong", "qa2_0k.json"))   # tools/reason4m/prepare_data.py babilong
NO_BABI = "the bAbI stories not downloaded (tools/reason4m/prepare_data.py babilong)"

# ------------------------------------------------------------------------------------------------ synthetic MRCR data

WORDS = "river stone amber lantern quiet harvest signal meadow copper winter orchard thunder velvet marble".split()


def mini_mrcr_doc(seed=1, n_filler=60):
    rng = random.Random(seed)
    head = ("Here are some examples of conversations succeeded by a follow-up question answered correctly:\n"
            "======== EXAMPLE 1 ========\n\nUser: Write a poem about lions in regal style.\n\nAssistant: Hark.\n\n"
            "======== END EXAMPLE 1 ========\n\n======== EXAMPLE 3 ========\n\n")
    combos = [("poem", "ducks", "formal"), ("tweet", "penguins", "archaic"), ("blog post", "ancient myths", "technical")]
    blocks = []
    for fmt, topic, style in combos:
        for k in range(8):
            text = f"{fmt}/{topic}/{style} #{k} " + " ".join(rng.choice(WORDS) for _ in range(120))
            blocks.append((("rel", fmt, topic, style), f"User: Write a {fmt} about {topic} in {style} style.\n\n"
                                                      f"Assistant: {text}\n\n", text))
    for k in range(n_filler):
        topic = rng.choice(["dreams", "black holes", "The Renaissance"])
        text = f"filler {k} " + " ".join(rng.choice(WORDS) for _ in range(120))
        blocks.append((("fill",), f"User: Write a poem about {topic} in formal style.\n\nAssistant: {text}\n\n", text))
    rng.shuffle(blocks)
    transcript = head + "".join(b[1] for b in blocks)
    rows = []
    for fmt, topic, style in combos:
        inst = [b[2] for b in blocks if b[0] == ("rel", fmt, topic, style)]
        h = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(12))
        for i, name in enumerate(R.INDEX_NAMES):
            final = (f"User: Prepend {h} to the {name} {fmt} about {topic} in a {style} style. Do not include any other "
                     f"text in your response.\n\nAssistant:")
            rows.append({"final_query": final, "answer": h + inst[i], "random_string": h, "num_relevant": 8,
                         "gemini_context_len": 1, "answer_context_position": ""})
    return {"source": "mini", "bin": "mini", "n_needles": 8, "transcript": transcript, "rows": rows}


def solve_mrcr_prompt(prompt):
    """What a perfect reader answers: the n-th instance of the group the last question asks about."""
    i = prompt.rfind("User: Prepend")
    if i < 0:
        return "ok"           # a calibration slice
    head, blocks = R.split_transcript(prompt[:i])
    m = R._FINAL.match(prompt[i:])
    h, ordn, fmt, topic, style = m.groups()
    inst = [b for b in blocks if R.block_topic(b) == (fmt, topic, style)]
    return h + inst[R.INDEX_NAMES.index(ordn)].split("Assistant: ", 1)[1].rstrip("\n")


# ------------------------------------------------------------------------------------------------ bAbI oracle

MOVE = re.compile(r"^(\w+) (?:went back to|went to|journeyed to|moved to|travelled to) the (\w+)\.$")
TAKE = re.compile(r"^(\w+) (?:got|took|grabbed|picked up) the (\w+)(?: there)?\.$")
DROP = re.compile(r"^(\w+) (?:dropped|discarded|put down|left) the (\w+)(?: there)?\.$")


def history(facts, obj):
    loc, held, at, hist = {}, {}, {}, []
    for f in facts:
        m = MOVE.match(f)
        if m:
            loc[m.group(1)] = m.group(2)
        elif TAKE.match(f):
            p, o = TAKE.match(f).groups()
            held[o] = p
        elif DROP.match(f):
            p, o = DROP.match(f).groups()
            held.pop(o, None)
            at[o] = loc.get(p)
        else:
            raise ValueError(f)
        cur = loc.get(held[obj]) if obj in held else at.get(obj)
        if cur is not None and (not hist or hist[-1] != cur):
            hist.append(cur)
    return hist


def oracle(task, facts, question):
    if task == "qa2":
        return history(facts, re.match(r"Where is the (\w+)\?", question).group(1))[-1]
    obj, loc = re.match(r"Where was the (\w+) before the (\w+)\?", question).groups()
    h = history(facts, obj)
    return h[max(i for i, x in enumerate(h) if x == loc) - 1]


def mini_data_dir():
    d = tempfile.mkdtemp(prefix="r4m-test-")
    os.makedirs(os.path.join(d, "mrcr"))
    with open(os.path.join(d, "mrcr", "1m.json"), "w") as f:
        json.dump(mini_mrcr_doc(), f)
    bd = os.path.join(d, "babilong")
    os.makedirs(os.path.join(bd, "pg19_test"))
    for f in ("qa2_0k.json", "qa3_0k.json", "prompts_qa2_qa3.json"):
        shutil.copy(os.path.join(REAL, "babilong", f), bd)
    rng = random.Random(5)
    vocab = [w + str(i) for i, w in enumerate(WORDS * 20)]
    for b in range(3):
        sents = [" ".join(rng.choice(vocab) for _ in range(rng.randint(6, 18))).capitalize() + "." for _ in range(2500)]
        with open(os.path.join(bd, "pg19_test", f"{b}.txt"), "w") as f:
            f.write("\n".join(sents))
    return d


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        R.main(list(argv))
    return out.getvalue()


def results(path):
    return R.read_rows(path)[1]


# ------------------------------------------------------------------------------------------------ tests


class Scoring(unittest.TestCase):
    H = "aB3dE6gH9jK2"

    def test_mrcr_exact_and_partial(self):
        target = self.H + "Dear friends, the harvest is in."
        self.assertEqual(R.mrcr_score(target, target), 1.0)
        self.assertEqual(R.mrcr_score("Dear friends, the harvest is in.", target), 0.0)      # no random string
        self.assertEqual(R.mrcr_score("", target), 0.0)
        self.assertEqual(R.mrcr_score(None, target), 0.0)
        s = R.mrcr_score(self.H + " Dear friends, the harvest is out.", target)
        self.assertTrue(0.9 < s < 1.0)
        # same ratio as difflib on the two texts
        import difflib
        self.assertAlmostEqual(s, difflib.SequenceMatcher(a="Dear friends, the harvest is in.",
                                                          b="Dear friends, the harvest is out.").ratio())

    def test_mrcr_uses_last_occurrence_of_the_string(self):
        target = self.H + "Alpha beta gamma."
        reply = f"Sure: {self.H} wrong text here\n{self.H}Alpha beta gamma."
        self.assertEqual(R.mrcr_score(reply, target), 1.0)
        self.assertEqual(R.mrcr_score_strict(reply, target), 0.0)        # does not BEGIN with it
        self.assertEqual(R.mrcr_score_strict(target + "\n", target), 1.0)

    def test_openai_rule(self):
        ans = self.H[:10] + "Hello there world"
        self.assertEqual(R.openai_mrcr_score(ans, ans, self.H[:10]), 1.0)
        self.assertEqual(R.openai_mrcr_score("x" + ans, ans, self.H[:10]), 0.0)
        self.assertEqual(R.openai_mrcr_score(self.H[:10] + "Hello there", ans, self.H[:10]),
                         R.difflib.SequenceMatcher(None, "Hello there", "Hello there world").ratio())

    def test_babilong_compare(self):
        L = R.TASK_LABELS["qa2"]
        c = lambda t, o, q: R.babilong_correct(t, o, q, L)
        self.assertTrue(c("kitchen", "The football is in the kitchen.", "Where is the football?"))
        self.assertFalse(c("kitchen", "The football is in the garden.", "Where is the football?"))
        self.assertFalse(c("kitchen", "It is in the kitchen or the garden.", "Where is the football?"))
        self.assertFalse(c("kitchen", "I cannot tell.", "Where is the football?"))
        # a label named in the question never counts as the answer (qa3)
        L3 = R.TASK_LABELS["qa3"]
        q3 = "Where was the apple before the garden?"
        self.assertTrue(R.babilong_correct("kitchen", "Before the garden the apple was in the kitchen.", q3, L3))
        self.assertFalse(R.babilong_correct("kitchen", "Before the garden the apple was in the office.", q3, L3))
        # only the first sentence counts
        self.assertTrue(c("office", "In the office. It might also be the garden.", "Where is the milk?"))

    @unittest.skipUnless(BABI, NO_BABI)
    def test_babilong_oracle_matches_the_published_targets(self):
        for task in ("qa2", "qa3"):
            with open(os.path.join(REAL, "babilong", f"{task}_0k.json")) as f:
                stories = json.load(f)
            for s in stories:
                facts = [x.strip() for x in re.findall(r"[^.]+\.", s["input"])]
                self.assertEqual(oracle(task, facts, s["question"].strip()), s["target"])


class MrcrFit(unittest.TestCase):
    def test_fit_drops_only_filler_and_keeps_every_answer(self):
        doc = mini_mrcr_doc()
        T = doc["transcript"]
        fitted, n = R.fit_transcript(T, 12_000)
        self.assertGreater(n, 0)
        self.assertLessEqual(len(fitted), len(T) - 12_000)
        _, blocks = R.split_transcript(fitted)
        self.assertEqual(sum(1 for b in blocks if R.block_topic(b)[1] not in R.FILLER_TOPICS), 24)
        for r in doc["rows"]:                       # every question still has the same n-th instance
            self.assertEqual(solve_mrcr_prompt(fitted + r["final_query"]), r["answer"])
        self.assertEqual(R.fit_transcript(T, 0), (T, 0))
        with self.assertRaises(ValueError):
            R.fit_transcript(T, len(T))

    def test_drop_is_spread_evenly(self):
        order = R.drop_order(1000)[:50]
        self.assertEqual(len(set(order)), 50)
        buckets = {k // 100 for k in order}
        self.assertEqual(len(buckets), 10)          # every tenth of the transcript is thinned

    def test_samples_spread_over_groups(self):
        doc = mini_mrcr_doc()
        idx = R.mrcr_samples(doc["rows"], 6, 7)
        self.assertEqual(len(set(idx)), 6)
        self.assertEqual(len({doc["rows"][i]["random_string"] for i in idx}), 3)
        self.assertEqual(idx, R.mrcr_samples(doc["rows"], 6, 7))      # deterministic

    @unittest.skipUnless(os.path.exists(os.path.join(REAL, "mrcr", "4m.json")), "published data not downloaded")
    def test_real_published_data(self):
        for ln in ("1m", "2m", "4m"):
            doc = R.load_json(os.path.join(REAL, "mrcr", f"{ln}.json"))
            plan = R.make_mrcr_plan(ln, 20, 1, 4.3, REAL)
            self.assertLessEqual(len(plan.prefix), int(R.LENGTHS[ln] * 4.3) + 1)
            self.assertEqual(len(plan.samples), 20)
            self.assertGreaterEqual(len({s["group"] for s in plan.samples}), 12)
            for s in plan.samples[:6]:
                self.assertEqual(solve_mrcr_prompt(plan.prompt(s)), s["expected"].strip())
            self.assertEqual(plan.notes["published_chars"] - plan.notes["dropped_chars"], len(plan.prefix))
            del doc, plan


@unittest.skipUnless(BABI, NO_BABI)
class Babilong(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = mini_data_dir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_haystack_holds_each_story_once_in_order_and_is_solvable(self):
        for task in ("qa2", "qa3"):
            plan = R.make_babilong_plan(task, "128k", 12, 3, 4.0, self.d, target_tokens=40_000)
            self.assertLessEqual(len(plan.prefix), 40_000 * 4.0 + 40_000)
            per_story = []
            for s in plan.samples:
                pos = -1
                for f in s["facts"]:
                    i = plan.prefix.find(f, pos + 1)
                    self.assertGreater(i, pos, f"fact out of order or missing: {f}")
                    pos = i
                self.assertEqual(oracle(task, s["facts"], s["question"]), s["expected"])
                per_story.append({f.split(" ")[0] for f in s["facts"]})
            # no two stories share a character (they would contradict each other in one haystack)
            self.assertEqual(sum(len(x) for x in per_story), len(set().union(*per_story)))
            # no original bAbI character or object left, so stories cannot contradict each other
            for w in ("Mary", "John", "Sandra", "Daniel"):
                self.assertNotIn(f" {w} ", plan.prefix.split("<context>", 1)[1])   # (the official examples name them)
            for s in plan.samples:
                for w in ("football", "apple", "milk"):
                    self.assertNotIn(w, s["question"])

    def test_prefix_is_shared_and_question_is_last(self):
        plan = R.make_babilong_plan("qa2", "128k", 5, 3, 4.0, self.d, target_tokens=20_000)
        self.assertTrue(plan.prefix.endswith("</context>\n\nQuestion: "))
        ps = [plan.prompt(s) for s in plan.samples]
        for p, s in zip(ps, plan.samples):
            self.assertTrue(p.startswith(plan.prefix))
            self.assertEqual(p[len(plan.prefix):], s["question"])
            self.assertLess(len(s["question"]), 100)
        self.assertEqual(len({s["question"] for s in plan.samples}), 5)

    def test_official_prompt_text_is_used(self):
        plan = R.make_babilong_plan("qa3", "128k", 2, 3, 4.0, self.d, target_tokens=20_000)
        self.assertTrue(plan.prefix.startswith("I give you context with the facts about locations and actions"))
        self.assertIn("Before the $location_1$ the $item$ was in the $location_2$.", plan.prefix)
        self.assertIn("<example>\nJohn journeyed to the bedroom.", plan.prefix)


@unittest.skipUnless(BABI, NO_BABI)
class RunWithFakeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = mini_data_dir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def setUp(self):
        self.out = os.path.join(self.d, f"out-{self._testMethodName}.jsonl")

    def mrcr_script(self, body):
        prompt = body["messages"][-1]["content"]
        ans = solve_mrcr_prompt(prompt)
        think = "hmm " * 30 if body["chat_template_kwargs"]["enable_thinking"] else ""
        return think, ans

    def args(self, url, *extra, suite="mrcr"):
        return ["run", "--url", url, "--suite", suite, "--lengths", "1m", "--out", self.out, "--data", self.d,
                "--target", "1m=15000", *extra]

    def test_request_shape_per_mode_and_rows(self):
        with FakeServer(self.mrcr_script) as srv:
            run_cli(*self.args(srv.url, "--samples", "2", "--modes", "off,high,max"))
        reqs = [b for b in srv.requests if b["max_tokens"] > 1]
        self.assertEqual(len(reqs), 6)
        calib = [b for b in srv.requests if b["max_tokens"] == 1]
        self.assertEqual(len(calib), R.N_SLICES)
        by = {}
        for b in reqs:
            by.setdefault(b["max_tokens"], []).append(b)
        self.assertEqual(sorted(by), [2048, 16384, 32768])
        for b in by[2048]:
            self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})
            self.assertNotIn("reasoning_effort", b)
        for b in by[16384]:
            self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(b["reasoning_effort"], "high")
        for b in by[32768]:
            self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(b["reasoning_effort"], "max")
        for b in reqs:
            self.assertIs(b["stream"], True)
            self.assertEqual(b["stream_options"], {"include_usage": True})
            self.assertEqual(b["model"], "deepseek-v4.1-flash-1017")
            self.assertEqual(b["temperature"], 1.0)
            self.assertEqual(len(b["messages"]), 1)
            self.assertEqual(b["messages"][0]["role"], "user")
        rows = results(self.out)
        self.assertEqual(len(rows), 6)
        for r in rows:
            for k in ("suite", "task", "length", "target_tokens", "prompt_tokens", "mode", "sample_id", "question",
                      "expected", "answer", "reasoning_chars", "score", "first_token_s", "total_s", "finish_reason"):
                self.assertIn(k, r)
            self.assertEqual(r["score"], 1.0)
            self.assertEqual(r["finish_reason"], "stop")
            self.assertIsNotNone(r["prompt_tokens"])
            self.assertGreaterEqual(r["total_s"], 0)
            self.assertEqual(r["reasoning_chars"] > 0, r["mode"] != "off")

    def test_shared_prefix_byte_for_byte(self):
        with FakeServer(self.mrcr_script) as srv:
            run_cli(*self.args(srv.url, "--samples", "6", "--modes", "off"))
        prompts = [b["messages"][0]["content"] for b in srv.requests if b["max_tokens"] > 1]
        self.assertEqual(len(prompts), 6)
        with open(self.out) as f:
            plan_line = [json.loads(x) for x in f if '"kind": "plan"' in x][0]
        # rebuild the plan the way the run did and compare byte for byte
        plan = R.make_mrcr_plan("1m", 6, R.DEFAULT_SEED, plan_line["chars_per_token"], self.d, 15000)
        for p in prompts:
            self.assertTrue(p.startswith(plan.prefix))
            self.assertLess(len(p) - len(plan.prefix), 400)
        # everything before "User: Prepend" is identical across all prompts
        cut = [p.rfind("User: Prepend") for p in prompts]
        self.assertEqual(len({p[:c] for p, c in zip(prompts, cut)}), 1)
        self.assertEqual(len({p[c:] for p, c in zip(prompts, cut)}), 6)

    def test_calibrated_size_hits_the_target(self):
        with FakeServer(self.mrcr_script) as srv:
            run_cli(*self.args(srv.url, "--samples", "1", "--modes", "off"))
        row = results(self.out)[0]
        self.assertLessEqual(row["prompt_tokens"], 15000 + 60)       # fake: 4.0 chars per token exactly

    def test_resume_skips_done_rows_and_retries_errors(self):
        with FakeServer(self.mrcr_script) as srv:
            run_cli(*self.args(srv.url, "--samples", "2", "--modes", "off"))
        self.assertEqual(len([b for b in srv.requests if b["max_tokens"] > 1]), 2)
        with FakeServer(self.mrcr_script) as srv2:
            out = run_cli(*self.args(srv2.url, "--samples", "3", "--modes", "off,high", "--resume"))
        sent = [b for b in srv2.requests if b["max_tokens"] > 1]
        self.assertEqual(len(sent), 3 * 2 - 2)          # 6 wanted, 2 already done (off, first two samples)
        self.assertEqual([b for b in srv2.requests if b["max_tokens"] == 1], [])    # plan reused: no calibration
        self.assertEqual(len(results(self.out)), 6)
        # an errored row is not "done"
        rows = results(self.out)
        with open(self.out, "a") as f:
            f.write(json.dumps({**rows[0], "error": "HTTP 500", "score": None, "mode": "max"}) + "\n")
        with FakeServer(self.mrcr_script) as srv3:
            run_cli(*self.args(srv3.url, "--samples", "3", "--modes", "off,high,max", "--resume"))
        sent3 = [b for b in srv3.requests if b["max_tokens"] > 1]
        self.assertEqual(len(sent3), 3)                 # the three max rows (one had errored, two never ran)
        # resumed prompts are byte-identical to the first run's
        first = [b["messages"][0]["content"] for b in srv.requests if b["max_tokens"] > 1]
        again = [b["messages"][0]["content"] for b in srv3.requests if b["max_tokens"] > 1]
        for p in first:
            self.assertTrue(any(p[:p.rfind("User: Prepend")] == q[:q.rfind("User: Prepend")] for q in again))

    def test_error_reply_is_recorded_not_scored(self):
        with FakeServer(self.mrcr_script, fail_requests={R.N_SLICES + 2}) as srv:
            run_cli(*self.args(srv.url, "--samples", "1", "--modes", "off,high"))
        rows = results(self.out)
        self.assertEqual(len(rows), 2)
        bad = [r for r in rows if r["error"]]
        self.assertEqual(len(bad), 1)
        self.assertIsNone(bad[0]["score"])
        self.assertIn("500", bad[0]["error"])

    def test_truncated_thinking_scores_zero_and_is_flagged(self):
        def script(body):
            if body["chat_template_kwargs"]["enable_thinking"]:
                return "word " * 100000, "never reached"
            return "", solve_mrcr_prompt(body["messages"][0]["content"])
        with FakeServer(script) as srv:
            run_cli(*self.args(srv.url, "--samples", "1", "--modes", "high", "--max-tokens", '{"high": 50}'))
        row = results(self.out)[0]
        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(row["answer"], "")
        self.assertEqual(row["score"], 0.0)
        self.assertEqual(row["max_tokens"], 50)

    def test_key_is_sent_and_never_printed(self):
        keyfile = os.path.join(self.d, "keys")
        with open(keyfile, "w") as f:
            f.write("# comment\ntools: sk-test-SECRET-123\nother: nope\n")
        with FakeServer(self.mrcr_script, require_key="sk-test-SECRET-123") as srv:
            out = run_cli(*self.args(srv.url, "--samples", "1", "--modes", "off", "--key-file", keyfile))
        self.assertNotIn("SECRET", out)
        with open(self.out) as f:
            self.assertNotIn("SECRET", f.read())
        self.assertEqual({k.lower(): v for k, v in srv.headers[-1].items()}.get("authorization"), "Bearer sk-test-SECRET-123")
        self.assertEqual(results(self.out)[0]["score"], 1.0)

    def test_dry_run_sends_nothing_and_prints_sizes(self):
        with FakeServer(self.mrcr_script) as srv:
            out = run_cli(*self.args(srv.url, "--samples", "4", "--modes", "off,high,max", "--dry-run"))
        self.assertEqual(srv.requests, [])
        self.assertFalse(os.path.exists(self.out))
        self.assertIn("1m", out)
        self.assertIn("12 requests", out)
        self.assertIn("max_tokens", out)
        self.assertIn("ESTIMATES", out)
        self.assertIn("dry run: nothing sent", out)

    @unittest.skipUnless(os.path.exists(os.path.join(REAL, "mrcr", "4m.json")), "published data not downloaded")
    def test_dry_run_on_the_real_sizes(self):
        out = run_cli("run", "--suite", "mrcr", "--lengths", "1m,2m,4m", "--samples", "20", "--out", "/nonexistent/x",
                      "--dry-run", "--assume-cpt", "4.3")
        for ln, tgt in (("1m", 1_000_000), ("2m", 2_000_000), ("4m", 4_000_000)):
            m = re.search(rf"== mrcr/8needle {ln}: target {tgt:,} tokens, prefix ([\d,]+) chars = ~([\d,]+) tokens", out)
            self.assertIsNotNone(m, out)
            toks = int(m.group(2).replace(",", ""))
            self.assertLessEqual(toks, tgt)
            self.assertGreater(toks, tgt * 0.99)
        self.assertEqual(out.count("20 samples x 3 modes = 60 requests"), 3)

    def test_babilong_run(self):
        def script(body):
            p = body["messages"][0]["content"]
            q = p[p.rfind("Question: ") + 10:]
            # find the story whose question this is via the plan the run recorded
            return "", ANS.get(q, "ok")
        plan_args = ("qa2", "128k", 3, R.DEFAULT_SEED, 4.0, self.d, 6000)
        plan = R.make_babilong_plan(*plan_args)
        ANS = {s["question"]: f"The thing is in the {s['expected']}." for s in plan.samples}
        with FakeServer(script) as srv:
            run_cli("run", "--url", srv.url, "--suite", "babilong", "--tasks", "qa2", "--lengths", "128k",
                    "--samples", "3", "--modes", "off", "--target", "128k=6000", "--out", self.out, "--data", self.d)
        rows = results(self.out)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["score"] for r in rows], [1.0, 1.0, 1.0])
        self.assertEqual({r["task"] for r in rows}, {"qa2"})
        prompts = [b["messages"][0]["content"] for b in srv.requests if b["max_tokens"] > 1]
        cut = prompts[0].rfind("Question: ")
        self.assertEqual(len({p[:cut] for p in prompts}), 1)        # one haystack, byte for byte
        self.assertEqual(len({p[cut:] for p in prompts}), 3)

    def test_summary_table(self):
        with FakeServer(self.mrcr_script) as srv:
            run_cli(*self.args(srv.url, "--samples", "3", "--modes", "off,high"))
        out = run_cli("summary", self.out)
        self.assertRegex(out, r"mrcr\s+8needle\s+1m\s+off\s+3\s+100\.0")
        for other in ("37.5", "76.9", "V4-Flash"):   # another checkpoint's published scores are never printed as a reference
            self.assertNotIn(other, out)
        self.assertIn("15 (random relevant needle", out)

    def test_estimate_table(self):
        out = run_cli("estimate", "--suite", "mrcr", "--lengths", "1m,2m,4m", "--samples", "20")
        self.assertIn("4m", out)
        self.assertAlmostEqual(R.decode_speed("4m"), 15.5, places=1)
        self.assertAlmostEqual(R.decode_speed("128k"), 35.0, places=1)
        mean, worst = R.estimate("mrcr", "4m", ["off"], 20)
        self.assertAlmostEqual(mean, 1260 + 20 * 7.2 + 20 * 500 / 15.5, delta=1)
        self.assertGreater(worst, mean)


@unittest.skipUnless(BABI, NO_BABI)
class ControlsAndRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = mini_data_dir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_verify_on_sent_transcript_catches_a_removed_needle(self):
        doc = mini_mrcr_doc()
        plan = R.make_mrcr_plan("1m", 6, 1, 4.0, self.d, 15000)
        self.assertEqual(R.verify_mrcr_samples(plan.prefix, plan.samples), [])
        self.assertEqual(plan.notes["answers_verified_on_sent_transcript"], 6)
        _, blocks = R.split_transcript(doc["transcript"])
        s = plan.samples[0]
        m = R._FINAL.match(s["tail"])
        victim = [b for b in blocks if R.block_topic(b) == (m.group(3), m.group(4), m.group(5))][0]
        broken = doc["transcript"].replace(victim, "")
        self.assertEqual(R.verify_mrcr_samples(broken, [s]), [s["id"]])       # the n-th instance shifted

    def test_chat_render_alternates_roles_and_ends_with_the_question(self):
        doc = mini_mrcr_doc()
        r = doc["rows"][3]
        msgs = R.to_chat_messages(doc["transcript"] + r["final_query"])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"] * ((len(msgs) - 1) // 2) + ["user"])
        self.assertTrue(msgs[0]["content"].startswith("Here are some examples"))
        self.assertTrue(msgs[-1]["content"].startswith("Prepend " + r["random_string"]))
        self.assertFalse(msgs[-1]["content"].rstrip().endswith("Assistant:"))
        self.assertIn(r["answer"][12:].strip(), [m["content"].strip() for m in msgs])
        # every sample shares all messages but the last
        a = R.to_chat_messages(doc["transcript"] + doc["rows"][0]["final_query"])
        b = R.to_chat_messages(doc["transcript"] + doc["rows"][9]["final_query"])
        self.assertEqual(a[:-1], b[:-1])
        self.assertNotEqual(a[-1], b[-1])

    def test_render_chat_run(self):
        def script(body):
            msgs = body["messages"]
            if len(msgs) < 3:
                return "", "ok"
            q = msgs[-1]["content"]
            m = re.match(r"Prepend (\w{12}) to the (\w+) (.+?) about (.+?) in an? (.+?) style", q)
            h, ordn, fmt, topic, style = m.groups()
            inst = [msgs[i + 1]["content"] for i in range(0, len(msgs) - 1, 2)
                    if msgs[i]["content"].endswith(f"Write a {fmt} about {topic} in {style} style.")
                    or msgs[i]["content"].endswith(f"Write an {fmt} about {topic} in {style} style.")]
            return "", h + inst[R.INDEX_NAMES.index(ordn)]
        out = os.path.join(self.d, "chat.jsonl")
        with FakeServer(script) as srv:
            run_cli("run", "--url", srv.url, "--suite", "mrcr", "--lengths", "1m", "--samples", "3", "--modes", "off",
                    "--render", "chat", "--target", "1m=15000", "--out", out, "--data", self.d)
        rows = results(out)
        self.assertEqual([r["score"] for r in rows], [1.0] * 3)
        self.assertEqual({r["render"] for r in rows}, {"chat"})
        real = [b for b in srv.requests if b["max_tokens"] > 1]
        self.assertTrue(all(len(b["messages"]) > 10 for b in real))

    def test_default_render_is_one_user_message(self):
        out = os.path.join(self.d, "text.jsonl")
        with FakeServer() as srv:
            run_cli("run", "--url", srv.url, "--suite", "mrcr", "--lengths", "1m", "--samples", "1", "--modes", "off",
                    "--target", "1m=15000", "--out", out, "--data", self.d)
        real = [b for b in srv.requests if b["max_tokens"] > 1]
        self.assertEqual(len(real[0]["messages"]), 1)
        self.assertTrue(real[0]["messages"][0]["content"].endswith("\n\nAssistant:"))

    @unittest.skipUnless(os.path.exists(os.path.join(REAL, "mrcr", "1m-2n.json")), "control data not downloaded")
    def test_short_and_two_needle_controls_share_one_prefix(self):
        for label, tgt, needles in (("32k", 32_000, 8), ("256k", 256_000, 8), ("1m-2n", 1_000_000, 2)):
            plan = R.make_mrcr_plan(label, 20, R.DEFAULT_SEED, 4.7, REAL)
            self.assertEqual(plan.task, f"{needles}needle")
            self.assertEqual(len(plan.samples), 20)
            self.assertLessEqual(len(plan.prefix), tgt * 4.7 * 1.04)
            self.assertGreater(len(plan.prefix), tgt * 4.7 * 0.9)
            self.assertEqual(plan.notes["answers_verified_on_sent_transcript"], 20)
            ps = [plan.prompt(s) for s in plan.samples]
            cut = len(plan.prefix)
            self.assertEqual(len({p[:cut] for p in ps}), 1)
            self.assertTrue(all(len(p) - cut < 400 for p in ps))
            self.assertTrue(all(s["needles"] == needles for s in plan.samples))
            self.assertGreaterEqual(len({s["group"] for s in plan.samples}), 9)
        self.assertEqual(R.split_label("1m-2n"), ("1m", 2))
        with self.assertRaises(ValueError):
            R.split_label("7m")

    def test_chance_by_needle_count(self):
        self.assertIn("51", R.chance("mrcr", "2needle"))
        self.assertIn("15", R.chance("mrcr", "8needle"))


def mini_omrcr(path, n_rows=6, needles=2, seed=3):
    rng = random.Random(seed)
    rows = []
    topics = ["poem about tapirs", "blog post about rocks", "essay about tea", "riddle about owls"]
    ordn = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    for r in range(n_rows):
        msgs = [{"role": "user", "content": "Here are some examples of conversations succeeded by a follow-up question..."}]
        pool = []
        for t in topics[:2]:
            pool += [t] * needles
        pool += [rng.choice(topics[2:]) for _ in range(30)]
        rng.shuffle(pool)
        seen = {}
        texts = {}
        for t in pool:
            msgs.append({"role": "user", "content": "write a " + t})
            txt = f"{t} #{len(texts)} " + " ".join(rng.choice(WORDS) for _ in range(60))
            texts[len(msgs)] = txt
            msgs.append({"role": "assistant", "content": txt})
        target = topics[r % 2]
        k = rng.randint(1, needles)
        idxs = [j for j, x in enumerate(msgs) if x["role"] == "user" and x["content"] == "write a " + target]
        h = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(10))
        msgs.append({"role": "user", "content": f"Prepend {h} to the {ordn[k]} (1 indexed) {target}. Do not include any other text in your response."})
        ans = h + msgs[idxs[k - 1] + 1]["content"]
        rows.append({"id": f"o{needles}-{r}", "messages": msgs, "answer": ans, "random_string": h, "n_needles": needles,
                     "n_chars": sum(len(m["content"]) for m in msgs), "desired_msg_index": idxs[k - 1],
                     "total_messages": len(msgs), "date_added": "x"})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def solve_omrcr(msgs):
    m = re.match(r"Prepend (\w+) to the (\d+)\w\w \(1 indexed\) (.+?)\. Do not", msgs[-1]["content"])
    h, k, topic = m.group(1), int(m.group(2)), m.group(3)
    idxs = [j for j, x in enumerate(msgs[:-1]) if x["role"] == "user" and x["content"] == "write a " + topic]
    return h + msgs[idxs[k - 1] + 1]["content"]


class OpenAiMrcr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix="r4m-omrcr-")
        cls.rows = mini_omrcr(os.path.join(cls.d, "openai", "2needle-1m.jsonl"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_stored_rows_are_consistent(self):
        self.assertTrue(all(R.verify_omrcr_row(r) for r in self.rows))
        bad = dict(self.rows[0], desired_msg_index=self.rows[0]["desired_msg_index"] + 2)
        self.assertFalse(R.verify_omrcr_row(bad))

    def test_openai_rule_needs_the_string_first_without_strip(self):
        r = self.rows[0]
        self.assertEqual(R.score_row("omrcr", "2needle", {"expected": r["answer"], "random_string": r["random_string"]}, r["answer"])[0], 1.0)
        self.assertEqual(R.score_row("omrcr", "2needle", {"expected": r["answer"], "random_string": r["random_string"]}, "\n\n" + r["answer"])[0], 0.0)
        self.assertEqual(R.score_row("omrcr", "2needle", {"expected": r["answer"], "random_string": r["random_string"]}, "")[0], 0.0)

    def test_run_sends_the_stored_messages_unchanged(self):
        out = os.path.join(self.d, "o.jsonl")
        with FakeServer(lambda b: ("", solve_omrcr(b["messages"]) if len(b["messages"]) > 5 else "ok")) as srv:
            run_cli("run", "--url", srv.url, "--suite", "omrcr", "--lengths", "1m-2n", "--samples", "4", "--modes", "off,high",
                    "--target", "1m=15000", "--out", out, "--data", self.d)
        real = [b for b in srv.requests if b["max_tokens"] > 1]
        self.assertEqual(len(real), 8)
        stored = {json.dumps(r["messages"]) for r in self.rows}
        self.assertTrue(all(json.dumps(b["messages"]) in stored for b in real))
        rows = results(out)
        self.assertEqual([r["score"] for r in rows], [1.0] * 8)
        self.assertEqual({r["suite"] for r in rows}, {"omrcr"})
        self.assertEqual({r["task"] for r in rows}, {"2needle"})
        self.assertEqual(len({r["sample_id"] for r in rows}), 4)
        self.assertIn("omrcr", run_cli("summary", out))

    def test_too_long_samples_are_skipped(self):
        plan = R.make_omrcr_plan("1m-2n", 6, 1, 4.0, self.d, target_tokens=(min(r["n_chars"] for r in self.rows) + 1) // 4 + 1)
        self.assertEqual(len(plan.samples), 1)


if __name__ == "__main__":
    unittest.main()
