"""A fake OpenAI-compatible chat server for the offline tests: stdlib http.server, SSE streaming, usage in the last chunk.
prompt_tokens = len(user text) // 4 (so calibrating it gives exactly 4.0 characters per token). Every request body is
kept in .requests. `script(body) -> (reasoning_text, content_text)` decides the reply; finish_reason is "length" when the
reply would not fit max_tokens (words counted as tokens)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeServer:
    def __init__(self, script=None, require_key=None, fail_requests=()):
        self.script = script or (lambda body: ("", "ok"))
        self.requests, self.headers = [], []
        self.require_key = require_key
        self.fail_requests = set(fail_requests)     # 1-based request numbers that answer HTTP 500
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n))
                outer.requests.append(body)
                outer.headers.append(dict(self.headers))
                if outer.require_key and self.headers.get("authorization") != "Bearer " + outer.require_key:
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b'{"error":"no key"}')
                    return
                if len(outer.requests) in outer.fail_requests:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error":"boom"}')
                    return
                reasoning, content = outer.script(body)
                prompt = body["messages"][-1]["content"]
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()

                def send(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b": keepalive\n\n")
                send({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
                budget = body.get("max_tokens", 10 ** 9)
                words = (reasoning.split(" ") if reasoning else [])
                cut = False
                if len(words) > budget:
                    words, content, cut = words[:budget], "", True
                for i in range(0, len(words), 5):
                    send({"choices": [{"index": 0, "delta": {"reasoning_content": " ".join(words[i:i + 5]) + " "}}]})
                if content:
                    for i in range(0, len(content), 40):
                        send({"choices": [{"index": 0, "delta": {"content": content[i:i + 40]}}]})
                send({"choices": [{"index": 0, "delta": {}, "finish_reason": "length" if cut else "stop"}]})
                send({"choices": [], "usage": {"prompt_tokens": len(prompt) // 4,
                                              "completion_tokens": len(words) + len(content.split()),
                                              "total_tokens": 0}})
                self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()
