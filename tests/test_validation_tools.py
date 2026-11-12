"""Offline regression tests for tools/validate.py and tools/needle_bench.py: each test starts a mock OpenAI-compatible
server on 127.0.0.1 and runs the real tool against it as a subprocess, checking its output and exit code. No GPU,
weights, Docker or Sparks needed.

    python3 -m unittest discover -s tests -v
    TOOLS_DIR=/path/to/other/tools python3 -m unittest discover -s tests -v   # run the same tests against other tools
"""
import json
import os
import shutil
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = os.environ.get("TOOLS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
PROMPTS = ["Write a haiku about the sea.", "List the first ten prime numbers.", "Explain TCP in two sentences.",
           "Write a Python function that reverses a string.", "Name five rivers in Europe.",
           "What causes the seasons on Earth? Two sentences.", "Translate 'good morning' into French, Spanish and German.",
           "Give three tips for writing clear documentation."]
DEBUG_MARK = "DEBUG-PROMPT-MARK"


def reply_to(p):
    return f"Reply to: {p}"


def completion(content, **message):
    return 200, {"choices": [{"message": {"role": "assistant", "content": content, **message}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class Mock:
    """A mock server whose replies come from respond(body) -> (status, json object or str)."""

    def __init__(self, respond, needs_key=None):
        mock = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                status, out = mock.respond(body)
                if needs_key and self.headers.get("authorization") != f"Bearer {needs_key}":
                    status, out = 401, {"error": {"message": "A valid API key is required", "code": "invalid_api_key"}}
                data = (out if isinstance(out, str) else json.dumps(out)).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.respond = respond
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def run_tool(name, args, env=None):
    e = {k: v for k, v in os.environ.items() if k != "TF_DEBUG_PROMPT_FILE"}
    e.update(env or {})
    p = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, name), *args], capture_output=True, text=True,
                       timeout=120, env=e)
    return p.returncode, p.stdout + p.stderr


def check_line(out, name):
    return next((l for l in out.splitlines() if name in l and l[:4] in ("PASS", "FAIL")), "")


class ToolTest(unittest.TestCase):
    def assertPassed(self, code, out, name):
        self.assertTrue(check_line(out, name).startswith("PASS"), out)
        self.assertEqual(code, 0, out)

    def assertFailedCleanly(self, code, out, name):
        """A deliberate FAIL of check <name>: its FAIL line, exit code 1, and no traceback (not a crash)."""
        self.assertTrue(check_line(out, name).startswith("FAIL"), out)
        self.assertEqual(code, 1, out)
        self.assertNotIn("Traceback", out)


# ---------------------------------------------------------------- validate.py

def validate_server(overrides=None, debug=None, features=None):
    """A server that answers every validate.py check correctly; overrides maps a test prompt to its reply text or a
    whole (status, body) answer, debug is the (status, body) answer to the TF_DEBUG_PROMPT_FILE prompt, and
    features(status, body) -> (status, body) rewrites the answers to the feature checks (text to long prompt)."""
    overrides = overrides or {}

    def respond(body):
        msgs = body["messages"]
        c = msgs[-1].get("content")
        answer = feature_answer(body)
        if answer is not None:
            return features(*answer) if features else answer
        if isinstance(c, str) and DEBUG_MARK in c:
            return debug or completion("Debug reply.")
        if c in PROMPTS:
            o = overrides.get(c, reply_to(c))
            return o if isinstance(o, tuple) else completion(o)
        return 400, {"error": {"message": "unexpected prompt"}}

    return respond


def feature_answer(body):
    """The correct answer to one of validate.py's feature checks, or None for any other prompt."""
    msgs = body["messages"]
    last = msgs[-1]
    c = last.get("content")
    if isinstance(c, list):
        kinds = {part.get("type") for part in c}
        return completion("Red at first, then blue." if "video_url" in kinds else "A yellow circle.")
    if last["role"] == "tool":
        return completion("It is 21 degrees and clear in Tokyo.")
    if DEBUG_MARK in c:
        return None
    if "capital of France" in c:
        return completion("Paris")
    if "17 * 23" in c:
        return completion("391", reasoning_content="17 * 23 = 391.")
    if "weather in Tokyo" in c:
        # a valid tool call: empty content, the call in tool_calls
        return completion("", tool_calls=[{"id": "call_0", "type": "function",
                                           "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}])
    if "item 1234" in c:
        return completion("8638")
    return None


class ValidateKey(ToolTest):
    """--key-file: every request carries "Authorization: Bearer <key>", the file's first key ("label: key" lines or bare
    keys, # comments), as the engine reads its own key file; a server with API keys refuses requests without one."""

    def setUp(self):
        self.mock = Mock(validate_server(), needs_key="k-one")
        self.addCleanup(self.mock.close)
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        self.kf = os.path.join(d, "keys")
        with open(self.kf, "w") as f:
            f.write("# the engine's keys\ntools: k-one\ngateway: k-two\n")

    def test_with_the_key_file_the_checks_pass(self):
        code, out = run_tool("validate.py", [self.mock.url, "--key-file", self.kf])
        self.assertEqual(code, 0, out)

    def test_without_it_the_server_refuses_them(self):
        code, out = run_tool("validate.py", [self.mock.url])
        self.assertNotEqual(code, 0, out)


class ValidateCompare(ToolTest):
    NAME = "equal to the saved replies"

    def run_compare(self, ref, overrides=None, raw=None):
        m = Mock(validate_server(overrides))
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                f.write(raw if raw is not None else json.dumps(ref))
            return run_tool("validate.py", [m.url, "--compare", f.name])
        finally:
            m.close()
            os.unlink(f.name)

    def assertCompare(self, ok, ref=None, overrides=None, raw=None, why=None):
        code, out = self.run_compare(ref, overrides, raw)
        if ok:
            self.assertPassed(code, out, self.NAME)
        else:
            self.assertFailedCleanly(code, out, self.NAME)
        if why:
            self.assertIn(why, check_line(out, self.NAME))

    def test_full_correct_reference_passes(self):
        self.assertCompare(True, [reply_to(p) for p in PROMPTS])

    def test_empty_reference_fails(self):
        self.assertCompare(False, [], why="holds 0 replies, not 8")

    def test_short_reference_fails(self):
        self.assertCompare(False, [reply_to(PROMPTS[0])], why="holds 1 replies, not 8")

    def test_long_reference_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS] + ["one more"])

    def test_reference_not_a_list_fails(self):
        self.assertCompare(False, {reply_to(p): 1 for p in PROMPTS})

    def test_reference_null_fails(self):
        self.assertCompare(False, raw="null")

    def test_reference_with_non_string_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [None], why="empty or non-text reply")

    def test_reference_not_json_fails(self):
        self.assertCompare(False, raw="not json")

    def test_missing_reference_file_fails(self):
        m = Mock(validate_server())
        try:
            code, out = run_tool("validate.py", [m.url, "--compare", os.path.join(tempfile.gettempdir(), "no-such-ref.json")])
        finally:
            m.close()
        self.assertFailedCleanly(code, out, self.NAME)
        self.assertIn("cannot read", check_line(out, self.NAME))

    def test_reordered_reference_fails(self):
        # the same replies in another order: each reply is compared with its own prompt's saved one
        self.assertCompare(False, [reply_to(p) for p in reversed(PROMPTS)], why="0 of 8 equal")

    def test_wrong_reply_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + ["something else"])

    def test_whitespace_difference_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [reply_to(PROMPTS[-1]) + " "])

    def test_line_ending_difference_fails(self):
        ref = [reply_to(p) for p in PROMPTS]
        ref[0] += "\n"
        self.assertCompare(False, ref, overrides={PROMPTS[0]: reply_to(PROMPTS[0]) + "\r\n"})

    def test_case_difference_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [reply_to(PROMPTS[-1]).upper()])

    def test_empty_reference_reply_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [""], why="empty or non-text reply")

    def test_empty_replies_on_both_sides_fail(self):
        # the server's reply and the saved one are both empty: equal, but not a reply
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [""], overrides={PROMPTS[-1]: ""})

    def test_error_marked_replies_fail(self):
        # every reply carries correct-looking choices and an error: an error is never a reply
        bad = {p: (200, {**completion(reply_to(p))[1], "error": {"message": "engine failed"}}) for p in PROMPTS}
        self.assertCompare(False, [reply_to(p) for p in PROMPTS], overrides=bad)


class ValidateConcurrent(ToolTest):
    NAME = "concurrent == solo"

    def run_validate(self, overrides):
        m = Mock(validate_server(overrides))
        try:
            return run_tool("validate.py", [m.url])
        finally:
            m.close()

    def test_all_replies_pass(self):
        self.assertPassed(*self.run_validate({}), self.NAME)

    def test_empty_reply_fails_without_compare(self):
        self.assertFailedCleanly(*self.run_validate({PROMPTS[3]: ""}), self.NAME)

    def test_error_marked_reply_fails_without_compare(self):
        bad = {PROMPTS[3]: (200, {**completion(reply_to(PROMPTS[3]))[1], "error": {"message": "engine failed"}})}
        self.assertFailedCleanly(*self.run_validate(bad), self.NAME)


class ValidateFeatures(ToolTest):
    CHECKS = ["text", "thinking", "tool call", "image", "long prompt"]

    def run_validate(self, features):
        m = Mock(validate_server(features=features))
        try:
            return run_tool("validate.py", [m.url])
        finally:
            m.close()

    def assertFeaturesFail(self, features, checks=None):
        code, out = self.run_validate(features)
        for name in checks or self.CHECKS:
            self.assertTrue(check_line(out, name).startswith("FAIL"), f"{name}:\n{out}")
        self.assertEqual(code, 1, out)
        self.assertNotIn("Traceback", out)

    def test_correct_features_pass(self):
        code, out = self.run_validate(None)
        for name in self.CHECKS + ["tool result"]:
            self.assertTrue(check_line(out, name).startswith("PASS"), f"{name}:\n{out}")
        self.assertEqual(code, 0, out)

    def test_error_marked_feature_replies_fail(self):
        # correct-looking replies that also carry an error: an error is never a reply
        self.assertFeaturesFail(lambda st, b: (st, {**b, "error": {"message": "engine failed"}}))

    def test_wrong_role_feature_replies_fail(self):
        def user_role(st, b):
            msg = {**b["choices"][0]["message"], "role": "user"}
            return st, {**b, "choices": [{"message": msg}]}
        self.assertFeaturesFail(user_role)

    def test_malformed_tool_call_fails_cleanly(self):
        def broken_call(st, b):
            msg = b["choices"][0]["message"]
            if msg.get("tool_calls"):
                return st, {**b, "choices": [{"message": {**msg, "tool_calls": [{"function": None}]}}]}
            return st, b
        self.assertFeaturesFail(broken_call, ["tool call"])

    def test_tool_call_arguments_not_json_fail_cleanly(self):
        def bad_args(st, b):
            msg = b["choices"][0]["message"]
            if msg.get("tool_calls"):
                call = {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{city: Tokyo"}}
                return st, {**b, "choices": [{"message": {**msg, "tool_calls": [call]}}]}
            return st, b
        self.assertFeaturesFail(bad_args, ["tool call"])

    def test_malformed_tool_calls_fail(self):
        good = {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}
        cases = {
            "type not function": [{**good, "type": "not-a-function"}],
            "no id": [{k: v for k, v in good.items() if k != "id"}],
            "empty id": [{**good, "id": ""}],
            "no type": [{k: v for k, v in good.items() if k != "type"}],
            "a null call after a good one": [good, None],
            "arguments not an object": [{**good, "function": {"name": "get_weather", "arguments": "[\"Tokyo\"]"}}],
            "city not a string": [{**good, "function": {"name": "get_weather", "arguments": "{\"city\": [\"Tokyo\"]}"}}],
            "a nameless call after a good one": [good, {**good, "id": "call_1", "function": {"name": "", "arguments": "{}"}}],
            "a name that is not a string": [good, {**good, "id": "call_1", "function": {"name": 7, "arguments": "{}"}}],
            "a call to a tool not offered after a good one": [good, {**good, "id": "call_1", "function": {"name": "nonexistent_tool", "arguments": "{\"city\": \"Oslo\"}"}}],
            "a second call without the required city": [good, {**good, "id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}],
            "NaN in the arguments": [{**good, "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\", \"x\": NaN}"}}],
        }
        for case, calls in cases.items():
            with self.subTest(case=case):
                def bad(st, b, calls=calls):
                    msg = b["choices"][0]["message"]
                    if msg.get("tool_calls"):
                        return st, {**b, "choices": [{"message": {**msg, "tool_calls": calls}}]}
                    return st, b
                self.assertFeaturesFail(bad, ["tool call"])

    def test_tool_call_with_null_content_passes(self):
        def null_content(st, b):
            msg = b["choices"][0]["message"]
            if msg.get("tool_calls"):
                return st, {**b, "choices": [{"message": {**msg, "content": None}}]}
            return st, b
        code, out = self.run_validate(null_content)
        self.assertTrue(check_line(out, "tool call").startswith("PASS"), out)
        self.assertEqual(code, 0, out)


class ValidateArguments(unittest.TestCase):
    def test_option_without_a_value_stops(self):
        # every case names a dead URL first: a tool that misreads its options must not reach a live server
        dead = "http://127.0.0.1:9"
        for argv in ([dead, "--model"], [dead, "--compare"], [dead, "--save"],
                     [dead, "--model", "--compare", "f.json"]):
            with self.subTest(args=argv):
                code, out = run_tool("validate.py", argv)
                self.assertNotEqual(code, 0, out)
                self.assertIn("needs a value", out)
                self.assertNotIn("Traceback", out)
                self.assertNotIn("PASS", out)


class ValidateDebugPrompt(ToolTest):
    NAME = "debug prompt"

    def run_debug(self, debug):
        m = Mock(validate_server(debug=debug))
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write(f"{DEBUG_MARK} a long debug prompt")
            return run_tool("validate.py", [m.url], {"TF_DEBUG_PROMPT_FILE": f.name})
        finally:
            m.close()
            os.unlink(f.name)

    def test_valid_reply_passes(self):
        self.assertPassed(*self.run_debug(completion("Debug reply.")), self.NAME)

    def test_http_error_fails(self):
        self.assertFailedCleanly(*self.run_debug((500, {"error": {"message": "internal error"}})), self.NAME)

    def test_empty_content_fails(self):
        self.assertFailedCleanly(*self.run_debug(completion("")), self.NAME)

    def test_non_text_content_fails(self):
        self.assertFailedCleanly(*self.run_debug(completion(42)), self.NAME)

    def test_null_choice_fails(self):
        self.assertFailedCleanly(*self.run_debug((200, {"choices": [None]})), self.NAME)

    def test_non_object_body_fails(self):
        self.assertFailedCleanly(*self.run_debug((200, [])), self.NAME)

    def test_wrong_role_fails(self):
        self.assertFailedCleanly(*self.run_debug(completion("Debug reply.", role="user")), self.NAME)


# ---------------------------------------------------------------- needle_bench.py

def needle_server(mode, calib=(200, None)):
    """Calibration answers normally (unless calib says otherwise); every needle prompt is answered by mode."""

    def respond(body):
        prompt = body["messages"][-1]["content"]
        if prompt.endswith("Say OK."):
            status, out = calib
            return (status, out) if out is not None else (200, {
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 3200, "completion_tokens": 1}})
        code = re.search(r"The secret code is: (\d+)\.", prompt).group(1)
        usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": 3}
        if mode == "correct":
            return 200, {"choices": [{"message": {"role": "assistant", "content": code}}], "usage": usage}
        if mode == "wrong":
            return 200, {"choices": [{"message": {"role": "assistant", "content": "000000"}}], "usage": usage}
        if mode == "http400_echo":
            return 400, {"error": {"message": "invalid request", "prompt": prompt}}
        if mode == "error_object":
            return 200, {"error": {"message": f"engine failed on code {code}"}}
        if mode == "no_choices":
            return 200, {"usage": usage, "detail": f"no choices, code {code}"}
        if mode == "empty_choices":
            return 200, {"choices": [], "usage": usage, "detail": code}
        if mode == "null_content":
            return 200, {"choices": [{"message": {"role": "assistant", "content": None}}], "usage": usage, "detail": code}
        if mode == "error_with_choices":
            return 200, {"choices": [{"message": {"role": "assistant", "content": code}}], "usage": usage,
                         "error": {"message": "engine failed"}}
        if mode in ("number_content", "list_content", "object_content"):
            content = {"number_content": int(code), "list_content": [code], "object_content": {"text": code}}[mode]
            return 200, {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": usage}
        if mode == "wrong_role":
            return 200, {"choices": [{"message": {"role": "user", "content": code}}], "usage": usage}
        if mode == "list_body":
            return 200, [code]
        raise ValueError(mode)

    return respond


class MultiNeedle(unittest.TestCase):
    """tools/multi_needle.py: codes planted at spread depths of one haystack; --needles 10 or 20 planted, --ask N of them
    asked (spread evenly), --seed S the same codes every run, so two runs with one seed send the same haystack and differ
    only in the question after it (docs/long-context.md's 4M test: plant 20, ask 10 cold, then ask all 20 again)."""

    def run_multi(self, *extra):
        prompts = []

        def respond(body):
            prompt = body["messages"][-1]["content"]
            if prompt.endswith("Say OK."):
                return 200, {"choices": [{"message": {"role": "assistant", "content": "OK"}}],
                             "usage": {"prompt_tokens": 3200, "completion_tokens": 1}}
            prompts.append(prompt)
            planted = dict(re.findall(r"The vault code of (\w+) is (\d{6})\. ", prompt))
            asked = re.search(r"Cities: (.*)\.$", prompt).group(1).split(", ")
            reply = "\n".join(f"{c}: {planted.get(c, '000000')}" for c in asked)
            return 200, {"choices": [{"message": {"role": "assistant", "content": reply}}],
                         "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 9}}

        m = Mock(respond)
        try:
            code, out = run_tool("multi_needle.py", ["--url", m.url, *extra, "--lens", "4000"])
        finally:
            m.close()
        return code, out, prompts

    @staticmethod
    def haystack(prompt):
        return prompt[:prompt.index("\nList the vault code")]

    def test_ten_planted_and_asked_by_default(self):
        code, out, prompts = self.run_multi()
        self.assertEqual(code, 0, out)
        self.assertIn("right=10/10", out)
        self.assertEqual(len(re.findall(r"The vault code of \w+ is \d{6}\. ", prompts[0])), 10)

    def test_twenty_planted_ten_asked(self):
        code, out, prompts = self.run_multi("--needles", "20", "--ask", "10")
        self.assertEqual(code, 0, out)
        self.assertIn("right=10/10 of 20 planted", out)
        self.assertEqual(len(re.findall(r"The vault code of \w+ is \d{6}\. ", prompts[0])), 20)
        self.assertEqual(len(re.search(r"Cities: (.*)\.$", prompts[0]).group(1).split(", ")), 10)

    def test_one_seed_sends_the_same_haystack_whatever_is_asked(self):
        _, out1, p1 = self.run_multi("--needles", "20", "--ask", "10", "--seed", "1")
        _, out2, p2 = self.run_multi("--needles", "20", "--ask", "20", "--seed", "1")
        self.assertEqual(self.haystack(p1[0]), self.haystack(p2[0]))
        self.assertNotEqual(p1[0], p2[0])
        self.assertIn("right=20/20 of 20 planted", out2)

    def test_values_it_cannot_plant_are_refused(self):
        for extra in (("--needles", "15"), ("--needles", "10", "--ask", "11"), ("--ask", "0")):
            code, out, _ = self.run_multi(*extra)
            self.assertNotEqual(code, 0, (extra, out))
            self.assertIn("--needles takes 10 or 20", out)


class Needle(unittest.TestCase):
    def run_needle(self, mode, extra=(), **kw):
        m = Mock(needle_server(mode, **kw))
        try:
            # extra first: needle_bench.py reads an option's first occurrence
            code, out = run_tool("needle_bench.py", ["--url", m.url, *extra, "--lens", "64,128", "--depths", "0,50,100"])
        finally:
            m.close()
        return code, out

    def assertAllFail(self, mode):
        """Every cell fails, reported as such: exit code 1, 0/6, no cell ok, no traceback."""
        code, out = self.run_needle(mode)
        self.assertEqual(code, 1, out)
        self.assertIn("0/6 passed", out)
        self.assertNotIn("ok=True", out)
        self.assertNotIn("Traceback", out)

    def test_correct_code_passes(self):
        code, out = self.run_needle("correct")
        self.assertEqual(code, 0, out)
        self.assertIn("6/6 passed", out)

    def test_wrong_code_fails(self):
        self.assertAllFail("wrong")

    def test_http_400_echoing_the_code_fails(self):
        self.assertAllFail("http400_echo")

    def test_error_object_fails(self):
        self.assertAllFail("error_object")

    def test_error_with_choices_fails(self):
        self.assertAllFail("error_with_choices")

    def test_missing_choices_fails(self):
        self.assertAllFail("no_choices")

    def test_empty_choices_fails(self):
        self.assertAllFail("empty_choices")

    def test_null_content_fails(self):
        self.assertAllFail("null_content")

    def test_number_content_fails(self):
        self.assertAllFail("number_content")

    def test_list_content_fails(self):
        self.assertAllFail("list_content")

    def test_object_content_fails(self):
        self.assertAllFail("object_content")

    def test_wrong_role_fails(self):
        self.assertAllFail("wrong_role")

    def test_non_object_body_fails(self):
        self.assertAllFail("list_body")

    def test_error_diagnostics_kept(self):
        # a failed cell says why, on the console and in the --json row, with the server's own error
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            code, out = self.run_needle("http400_echo", extra=("--json", path))
            with open(path) as f:
                rows = [json.loads(l) for l in f]
        self.assertEqual(code, 1, out)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertFalse(row["ok"])
            self.assertEqual(row["reply"], "")
            self.assertTrue(row["error"].startswith("error: "), row)
            self.assertIn("invalid request", row["error"])
        self.assertEqual(out.count("error: "), 6, out)

    def test_no_reply_diagnostics_kept(self):
        # a reply without choices: the cell says so, with what the server sent
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            code, out = self.run_needle("no_choices", extra=("--json", path))
            with open(path) as f:
                rows = [json.loads(l) for l in f]
        self.assertEqual(code, 1, out)
        for row in rows:
            self.assertTrue(row["error"].startswith("no reply text: "), row)
            self.assertIn("no choices, code", row["error"])
        self.assertEqual(out.count("no reply text: "), 6, out)

    def assertCalibrationStops(self, calib):
        code, out = self.run_needle("correct", calib=calib)
        self.assertEqual(code, 1, out)
        self.assertIn("Calibration request failed", out)
        self.assertNotIn("passed", out)
        self.assertNotIn("Traceback", out)

    def test_failed_calibration_stops(self):
        self.assertCalibrationStops((400, {"error": {"message": "bad"}}))

    def test_error_marked_calibration_stops(self):
        self.assertCalibrationStops((200, {"error": {"message": "engine failed"}, "usage": {"prompt_tokens": 1}}))

    def test_implausible_calibration_counts_stop(self):
        for n in (1, -3200, 0, True, "3200", 3200.5):
            with self.subTest(prompt_tokens=n):
                self.assertCalibrationStops((200, {"choices": [{"message": {"role": "assistant", "content": "OK"}}],
                                                   "usage": {"prompt_tokens": n}}))

    def test_bad_grid_arguments_stop(self):
        for extra in (("--reps", "0"), ("--reps", "-1"), ("--depths", "150"), ("--depths", "-5"), ("--lens", "0"),
                      ("--lens", "abc"), ("--reps", "x")):
            with self.subTest(args=extra):
                code, out = self.run_needle("correct", extra=extra)
                self.assertNotEqual(code, 0, out)
                self.assertNotIn("passed", out)
                self.assertNotIn("Traceback", out)

    def test_option_without_a_value_stops(self):
        # each case but --url's own names a dead URL first: a tool that misreads its options must not reach a live
        # server (the default grid runs to 1M tokens)
        dead = ["--url", "http://127.0.0.1:9"]
        for argv in (dead + ["--lens"], dead + ["--depths"], dead + ["--reps"], dead + ["--model"], ["--url"],
                     dead + ["--reps", "--lens", "64"]):
            with self.subTest(args=argv):
                code, out = run_tool("needle_bench.py", argv)
                self.assertNotEqual(code, 0, out)
                self.assertIn("needs a value", out)
                self.assertNotIn("Traceback", out)
                self.assertNotIn("passed", out)


if __name__ == "__main__":
    unittest.main()


class EnvSchemaScope(unittest.TestCase):
    """tools/schema/check_env_schema.py holds every environment name THIS engine reads to a schema row: the scan leaves out
    the code the DeepSeek engine does not run (the base engine's other models and GPU backends, its run, benchmark and
    oracle tools, readers nothing here calls), so a name read only there needs no row, while a name read both there and in
    this engine's code is still found, at its use here. On the repository it passes."""
    def scan_tree(self, files):
        tmp = Path(tempfile.mkdtemp(prefix="tf-schema-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        for rel, text in files.items():
            p = tmp / "zig" / "src" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        spec = importlib.util.spec_from_file_location("seed_under_test", ROOT / "tools" / "schema" / "seed.py")
        seed = importlib.util.module_from_spec(spec); spec.loader.exec_module(seed)
        old = os.getcwd(); os.chdir(tmp)
        try:
            return seed.scan()
        finally:
            os.chdir(old)

    def test_a_name_this_engine_reads_is_found(self):
        found = self.scan_tree({"families/dsv41/x.zig": 'const v = std.c.getenv("TF_DS_PLANTED");\n',
                                "server/serve.zig": 'const w = env(s, "TENSORFOLD_PLANTED");\n'})
        self.assertEqual(found.get("TF_DS_PLANTED"), ("families/dsv41/x.zig", 1))
        self.assertIn("TENSORFOLD_PLANTED", found)

    def test_a_name_only_other_models_or_tools_read_needs_no_row(self):
        found = self.scan_tree({"families/glm/a.zig": 'const v = std.c.getenv("TF_GLM_PLANTED");\n',
                                "xpu/b.zig": 'const v = std.c.getenv("TF_XPU_PLANTED");\n',
                                "server/benchmark_host.zig": 'const v = env.get("TF_BENCH_PLANTED");\n',
                                "families/dsv41/split_bench_main.zig": 'const v = std.c.getenv("TF_SPLIT_PLANTED");\n'})
        for n in ("TF_GLM_PLANTED", "TF_XPU_PLANTED", "TF_BENCH_PLANTED", "TF_SPLIT_PLANTED"):
            self.assertNotIn(n, found)

    def test_a_name_read_there_and_here_is_found_here(self):
        found = self.scan_tree({"families/glm/a.zig": 'const v = std.c.getenv("TF_BOTH_PLANTED");\n',
                                "families/dsv41/z.zig": '\nconst v = std.c.getenv("TF_BOTH_PLANTED");\n'})
        self.assertEqual(found.get("TF_BOTH_PLANTED"), ("families/dsv41/z.zig", 2))

    def test_the_repository_passes(self):
        r = subprocess.run([sys.executable, "-I", "tools/schema/check_env_schema.py"], cwd=ROOT, capture_output=True,
                           text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


# ---------------------------------------------------------------- longbench2.py

def lb2_module():
    spec = importlib.util.spec_from_file_location("longbench2_under_test", os.path.join(TOOLS, "longbench2.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


LB2_ITEMS = [
    {"_id": "a1", "domain": "Single-Document QA", "sub_domain": "x", "difficulty": "easy", "length": "long",
     "question": "Which colour is the door?", "choice_A": "red", "choice_B": "blue", "choice_C": "green",
     "choice_D": "grey", "answer": "B", "context": "The door is blue. " * 20},
    {"_id": "a2", "domain": "Multi-Document QA", "sub_domain": "y", "difficulty": "hard", "length": "long",
     "question": "Who wrote it?", "choice_A": "Ann", "choice_B": "Bo", "choice_C": "Cy", "choice_D": "Di",
     "answer": "C", "context": "Cy wrote it. " * 20},
    {"_id": "a3", "domain": "Code Repository Understanding", "sub_domain": "z", "difficulty": "hard", "length": "long",
     "question": "Which file?", "choice_A": "a.py", "choice_B": "b.py", "choice_C": "c.py", "choice_D": "d.py",
     "answer": "D", "context": "def f(): pass\n" * 20},
    {"_id": "s1", "domain": "Single-Document QA", "sub_domain": "x", "difficulty": "easy", "length": "short",
     "question": "Short one?", "choice_A": "1", "choice_B": "2", "choice_C": "3", "choice_D": "4",
     "answer": "A", "context": "One."},
]


class LongBench2(ToolTest):
    """tools/longbench2.py runs LongBench v2 (THUDM) the way its authors' pred.py does: their 0-shot template, temperature
    0.1, 128 answer tokens, a context longer than the window cut in the middle by the model's own tokens (the first and
    last halves kept), the answer read from "The correct answer is (X)". Thinking off by default; --thinking EFFORT
    gives the reasoning room. It writes one line an item as it goes, skips items already written, and sums up."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tf-lb2-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = os.path.join(self.tmp, "data.json")
        with open(self.data, "w") as f:
            json.dump(LB2_ITEMS, f)
        self.out = os.path.join(self.tmp, "out.jsonl")
        self.bodies = []

    def server(self, answers, key=None):
        def respond(body):
            self.bodies.append(body)
            q = body["messages"][0]["content"]
            text = next((a for k, a in answers.items() if k in q), "I am not sure.")
            return completion(text)
        s = Mock(respond, needs_key=key)
        self.addCleanup(s.close)
        return s

    def results(self):
        with open(self.out) as f:
            return {json.loads(l)["_id"]: json.loads(l) for l in f if l.strip()}

    def test_the_middle_cut_keeps_the_first_and_last_halves(self):
        m = lb2_module()
        self.assertEqual(m.middle_cut(list(range(10)), 4), [0, 1, 8, 9])
        self.assertEqual(m.middle_cut(list(range(10)), 10), list(range(10)))
        self.assertEqual(m.middle_cut(list(range(3)), 0), [0, 1, 2])          # 0: no limit

    def test_the_answer_is_read_the_authors_way(self):
        m = lb2_module()
        self.assertEqual(m.extract_answer("The correct answer is (B)"), "B")
        self.assertEqual(m.extract_answer("**The correct answer is C**"), "C")
        self.assertIsNone(m.extract_answer("Probably the second one."))

    def test_a_run_asks_each_item_with_the_authors_template_and_scores_it(self):
        s = self.server({"Which colour": "The correct answer is (B)", "Who wrote": "The correct answer is (A)",
                         "Which file": "It is d.py."})
        code, out = run_tool("longbench2.py", ["--url", s.url, "--data", self.data, "--length", "long", "--out", self.out,
                                               "--max-prompt-tokens", "0"])
        self.assertEqual(code, 0, out)
        r = self.results()
        self.assertEqual(sorted(r), ["a1", "a2", "a3"])                         # the long bin only
        self.assertEqual((r["a1"]["judge"], r["a2"]["judge"], r["a3"]["judge"]), (True, False, False))
        self.assertIsNone(r["a3"]["pred"])
        b = self.bodies[0]
        self.assertEqual((b["temperature"], b["max_tokens"]), (0.1, 128))
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})
        q = b["messages"][0]["content"]
        self.assertTrue(q.startswith("Please read the following text and answer the question below."), q[:80])
        self.assertIn("What is the correct answer to this question: Which colour is the door?", q)
        self.assertIn("(B) blue", q)
        self.assertIn('Format your response as follows: "The correct answer is (insert answer here)".', q)
        self.assertIn("accuracy 1 of 3", out)

    def test_items_already_written_are_not_asked_again(self):
        s = self.server({"Which colour": "The correct answer is (B)"})
        run_tool("longbench2.py", ["--url", s.url, "--data", self.data, "--length", "long", "--out", self.out, "--max-prompt-tokens", "0"])
        n = len(self.bodies)
        code, out = run_tool("longbench2.py", ["--url", s.url, "--data", self.data, "--length", "long", "--out", self.out,
                                               "--max-prompt-tokens", "0"])
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.bodies), n)
        self.assertEqual(len(self.results()), 3)

    def test_thinking_gives_the_reasoning_room(self):
        s = self.server({"Which colour": "The correct answer is (B)"})
        code, out = run_tool("longbench2.py", ["--url", s.url, "--data", self.data, "--length", "short", "--out", self.out,
                                               "--max-prompt-tokens", "0", "--thinking", "high"])
        self.assertEqual(code, 0, out)
        b = self.bodies[0]
        self.assertEqual((b["reasoning_effort"], b["max_tokens"]), ("high", 16384))
        self.assertNotIn("chat_template_kwargs", b)

    def test_the_key_file_is_sent(self):
        key = os.path.join(self.tmp, "keys")
        with open(key, "w") as f:
            f.write("tools: sk-lb2-test\n")
        os.chmod(key, 0o600)
        s = self.server({"Short one": "The correct answer is (A)"}, key="sk-lb2-test")
        code, out = run_tool("longbench2.py", ["--url", s.url, "--data", self.data, "--length", "short", "--out", self.out,
                                               "--max-prompt-tokens", "0", "--key-file", key])
        self.assertEqual(code, 0, out)
        self.assertTrue(self.results()["s1"]["judge"], out)


# ---------------------------------------------------------------- readme_tables.py

class ReadmeTables(unittest.TestCase):
    """tools/readme_tables.py rebuilds the README's performance tables from the runs' own files: each cell the median of
    the repetitions, a first token the mean of one repetition's requests then that median, times written as the README
    writes them (710 ms, 1.70 s, 16.4 s), the column's best prompt speed in bold, Turbo's change as a signed percentage."""

    def write(self, name, obj):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as f:
            if isinstance(obj, list):
                f.write("".join(json.dumps(r) + "\n" for r in obj))
            else:
                json.dump(obj, f)
        return p

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tf-tables-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_the_three_tables_from_their_files(self):
        main = self.write("main.json", {
            "summary": {"prefill.4096.tok_s": 5559.4, "prefill.4096.ttft_s": 0.7098, "prefill.1000000.tok_s": 6205.5,
                        "prefill.1000000.ttft_s": 154.38, "multi.1.aggregate_tok_s": 69.87, "multi.1.per_stream_tok_s": 69.87,
                        "multi.16.aggregate_tok_s": 227.11, "multi.16.per_stream_tok_s": 16.17},
            "runs": [{"suite": "multi", "streams": 1, "ttft_s": [0.13]}, {"suite": "multi", "streams": 1, "ttft_s": [0.135]},
                     {"suite": "multi", "streams": 1, "ttft_s": [0.2]},
                     {"suite": "multi", "streams": 16, "ttft_s": [1.5, 1.6]}, {"suite": "multi", "streams": 16, "ttft_s": [1.54]},
                     {"suite": "multi", "streams": 16, "ttft_s": [1.6]}]})
        turbo = self.write("turbo.json", {"summary": {"prefill.4096.tok_s": 5292.0, "prefill.4096.ttft_s": 0.746,
                                                       "prefill.1000000.tok_s": 6501.0, "prefill.1000000.ttft_s": 147.4}})
        code = self.write("code.jsonl", [
            {"mode": "decode", "conc": 1, "wall_tok_s": w, "per_req_tok_s": e, "ttft_s": t}
            for w, e, t in ((120.2, 123.1, 0.13), (103.0, 105.9, 0.132), (114.9, 117.4, 0.133))])
        code_, out = run_tool("readme_tables.py", ["--main", main, "--turbo", turbo, "--code", code])
        self.assertEqual(code_, 0, out)
        for row in ("| 4K | 5,559 | 710 ms | 5,292 | 746 ms | -4.8 % |",
                    "| 1M | **6,206** | 154.4 s | 6,501 | 147.4 s | +4.8 % |",
                    "| 1 | **69.9** | 69.9 | 135 ms |",
                    "| 16 | **227.1** | 16.2 | 1.55 s |",
                    "| 1 | **114.9** | 117.4 | 132 ms |"):
            self.assertIn(row, out)
