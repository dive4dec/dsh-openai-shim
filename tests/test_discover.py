"""Tests for dsh_openai_shim.discover + the runtime self-heal retry.

Two hermes-style mechanisms keep the completion-token cap correct without the
student knowing it:
  1. endpoint probe — GET /v1/models → max_model_len / max_completion_tokens
  2. error self-heal — parse the ceiling out of a 400 "too large" error and retry
"""
import json
import threading

import pytest

from dsh_openai_shim.discover import (
    discover_completion_cap,
    parse_completion_cap_from_error,
    is_output_cap_error,
)


# ─────────────────────────────────────────────────────────────
# parse_completion_cap_from_error
# ─────────────────────────────────────────────────────────────

def test_parse_sglang_at_most_completion_tokens():
    msg = ('{"object":"error","message":"max_completion_tokens is too large: '
           '100000.This model supports at most 64000 completion tokens.",'
           '"code":400}')
    assert parse_completion_cap_from_error(msg) == 64000


def test_parse_does_not_return_sent_value():
    # "too large: 100000" is what we SENT, not the limit — must not return 100000.
    msg = "max_completion_tokens is too large: 100000. at most 64000 completion tokens"
    assert parse_completion_cap_from_error(msg) == 64000


def test_parse_dashscope_range_form():
    assert parse_completion_cap_from_error("Range of max_tokens should be [1, 65536]") == 65536


def test_parse_not_output_cap_error_returns_none():
    # prompt-too-long: not an output-cap error → no cap (don't touch the cap)
    assert parse_completion_cap_from_error("prompt is too long: 200000 tokens, maximum is 131072") is None


def test_parse_no_number_returns_none():
    assert parse_completion_cap_from_error("max_tokens is too large for this model") is None


def test_is_output_cap_error_distinct_from_prompt_too_long():
    assert is_output_cap_error("max_completion_tokens is too large: 100000")
    assert is_output_cap_error("max_tokens is too large for this endpoint")
    assert not is_output_cap_error("prompt is too long: 300000 > 131072")
    assert not is_output_cap_error("context_length_exceeded: input too large")


# ─────────────────────────────────────────────────────────────
# discover_completion_cap — endpoint probe
# ─────────────────────────────────────────────────────────────

class ModelsUpstream:
    """Serves GET /v1/models with a given payload; nothing else."""

    def __init__(self, port, models_payload, require_auth=False):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                if require_auth and not (self.headers.get("Authorization") or "").startswith("Bearer "):
                    body = json.dumps({"error": "Unauthorized"}).encode()
                    self.send_response(401)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = json.dumps(models_payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.httpd.server_close()


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_discover_reads_max_model_len_from_models():
    port = _free_port()
    up = ModelsUpstream(port, {"object": "list", "data": [
        {"id": "Socrates", "object": "model", "owned_by": "sglang", "max_model_len": 64000}]})
    try:
        cap = discover_completion_cap(f"http://127.0.0.1:{port}/v1")
        assert cap == 64000
    finally:
        up.stop()


def test_discover_prefers_explicit_max_completion_tokens():
    port = _free_port()
    up = ModelsUpstream(port, {"object": "list", "data": [
        {"id": "m", "max_model_len": 131072, "max_completion_tokens": 16384}]})
    try:
        assert discover_completion_cap(f"http://127.0.0.1:{port}") == 16384
    finally:
        up.stop()


def test_discover_requires_key_when_server_asks():
    port = _free_port()
    up = ModelsUpstream(port, {"data": [{"id": "m", "max_model_len": 32000}]},
                        require_auth=True)
    try:
        # no key → 401 → None (falls back to default + self-heal later)
        assert discover_completion_cap(f"http://127.0.0.1:{port}/v1", api_key="") is None
        # with key → discovered
        assert discover_completion_cap(f"http://127.0.0.1:{port}/v1", api_key="k") == 32000
    finally:
        up.stop()


def test_discover_unreachable_returns_none():
    # nothing listening on this port → clean None, no exception
    assert discover_completion_cap("http://127.0.0.1:1/v1", timeout=2) is None


# ─────────────────────────────────────────────────────────────
# runtime self-heal retry (end-to-end through the shim)
# ─────────────────────────────────────────────────────────────

class CappedUpstream:
    """Upstream that rejects max_tokens > cap with the sglang 400 message."""

    def __init__(self, port, cap):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _read(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                outer.last_body = json.loads(self.rfile.read(n)) if n else {}

            def do_POST(self):
                self._read()
                mt = outer.last_body.get("max_tokens", 0)
                if mt > cap:
                    body = json.dumps({"object": "error",
                                       "message": f"max_completion_tokens is too large: {mt}."
                                                  f"This model supports at most {cap} completion tokens.",
                                       "type": "BadRequestError", "code": 400}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = json.dumps({"id": "x", "object": "chat.completion",
                                   "choices": [{"index": 0,
                                                "message": {"role": "assistant", "content": "pong"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.last_body = {}

    def stop(self):
        self.httpd.server_close()


def test_self_heal_retries_with_lowered_cap(tmp_path, monkeypatch, capsys):
    """A 400 'too large ... at most 64000' is corrected and retried once."""
    import urllib.request
    from dsh_openai_shim import make_handler
    from http.server import ThreadingHTTPServer

    # point DSH_HOME at a temp config with a provider that has NO cap yet,
    # so the self-heal has something to persist to.
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    (tmp_path / "proxy.conf").write_text(json.dumps({
        "provider": "spark",
        "providers": {"spark": {"base_url": "http://127.0.0.1:9/v1", "api_key": "k"}},
    }))

    up_port = _free_port()
    shim_port = _free_port()
    upstream = CappedUpstream(up_port, cap=64000)
    try:
        cfg = __import__("dsh_openai_shim").ShimConfig(
            upstream=f"http://127.0.0.1:{up_port}/v1",
            listen_port=shim_port,
            effort_mode="off",
            token_cap=100000,          # too high → upstream will 400 first
            provider_name="spark",
        )
        Handler, _ = make_handler(cfg)
        httpd = ThreadingHTTPServer(("127.0.0.1", shim_port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        body = json.dumps({"model": "Socrates", "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{shim_port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            assert r.status == 200
            assert json.loads(r.read())["choices"][0]["message"]["content"] == "pong"

        # the retried body carried the lowered cap
        assert upstream.last_body["max_tokens"] == 64000
        # in-memory cap was corrected
        assert cfg.token_cap == 64000
        # and it was persisted to proxy.conf
        saved = json.loads((tmp_path / "proxy.conf").read_text())
        assert saved["providers"]["spark"]["token_cap"] == 64000
        assert "self-heal" in capsys.readouterr().out
    finally:
        httpd.server_close()
        upstream.stop()


def test_no_self_heal_on_non_cap_error(tmp_path, monkeypatch):
    """A 400 that is NOT an output-cap error is forwarded, not retried."""
    import urllib.request
    import urllib.error
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from dsh_openai_shim import make_handler

    monkeypatch.setenv("DSH_HOME", str(tmp_path))

    up_port = _free_port()
    shim_port = _free_port()
    count = {"n": 0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n:
                self.rfile.read(n)
            count["n"] += 1
            body = json.dumps({"error": {"message": "invalid api key"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd_up = ThreadingHTTPServer(("127.0.0.1", up_port), H)
    threading.Thread(target=httpd_up.serve_forever, daemon=True).start()
    cfg = __import__("dsh_openai_shim").ShimConfig(
        upstream=f"http://127.0.0.1:{up_port}/v1", listen_port=shim_port,
        effort_mode="off", token_cap=100000, provider_name="x")
    Handler, _ = make_handler(cfg)
    httpd = ThreadingHTTPServer(("127.0.0.1", shim_port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        body = json.dumps({"messages": []}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{shim_port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"},
                                     method="POST")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=20)
        assert ei.value.code == 400
        # exactly one attempt — no retry for a non-cap error
        assert count["n"] == 1
    finally:
        httpd.server_close()
        httpd_up.server_close()


# ─────────────────────────────────────────────────────────────
# context-window self-heal (the sglang spark bug)
# ─────────────────────────────────────────────────────────────

def test_parse_context_window_error_real_message():
    """The exact sglang spark error parses to (window=64000, input=7897, total)."""
    from dsh_openai_shim.discover import parse_context_window_error
    msg = ("Requested token count exceeds the model's maximum context length of "
           "64000 tokens. You requested a total of 71897 tokens: 7897 tokens "
           "from the input messages and 64000 tokens for the completion.")
    window, input_tokens, total = parse_context_window_error(msg)
    assert window == 64000      # the TOTAL window, not the completion figure
    assert input_tokens == 7897  # the actual prompt count
    assert total == 71897


def test_parse_context_window_error_not_output_cap_form():
    """An output-cap error ('at most N completion tokens') is NOT a context error."""
    from dsh_openai_shim.discover import parse_context_window_error
    window, input_tokens, total = parse_context_window_error(
        "max_completion_tokens is too large: 100000. "
        "This model supports at most 64000 completion tokens.")
    assert window is None and input_tokens is None and total is None


def test_clamp_uses_window_minus_input():
    """apply_rewrites clamps max_tokens to window − est_input − reserve."""
    from dsh_openai_shim import ShimConfig, apply_rewrites
    # window=64000, cap=64000; prompt ~ 40000 chars → est ~10000 → max ~53872.
    cfg = ShimConfig(upstream="http://u/v1", listen_port=1,
                     effort_mode="off", token_cap=64000, context_window=64000)
    long_prompt = "word " * 8000   # 40000 chars
    data = json.dumps({"model": "m", "max_tokens": 60000,
                       "messages": [{"role": "user", "content": long_prompt}]}).encode()
    out, changes = apply_rewrites(data, cfg)
    got = json.loads(out)["max_tokens"]
    # must drop well below 60000 to fit the window
    assert got < 60000
    assert got <= 64000 - 10000  # window minus the estimate
    # a SHORT prompt should NOT be clamped below the cap
    short = json.dumps({"model": "m", "max_tokens": 50000,
                        "messages": [{"role": "user", "content": "hi"}]}).encode()
    out2, _ = apply_rewrites(short, cfg)
    assert json.loads(out2)["max_tokens"] == 50000  # 50000 < cap, small input → kept


def test_self_heal_context_window_retries_and_fits(tmp_path, monkeypatch, capsys):
    """End-to-end: sglang's context-length 400 → shim reads the EXACT input
    count, retries once with max_tokens = window − input − reserve → 200."""
    import urllib.request
    from dsh_openai_shim import make_handler, _estimate_input_tokens
    from http.server import ThreadingHTTPServer

    WINDOW = 64000
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    (tmp_path / "proxy.conf").write_text(json.dumps({
        "provider": "spark",
        "providers": {"spark": {"base_url": "http://127.0.0.1:9/v1", "api_key": "k"}},
    }))

    up_port = _free_port()
    shim_port = _free_port()
    seen = []   # every max_tokens the mock received, in order

    # A mock sglang: input is a fixed 7897 (the prompt's real token count);
    # 400s with the real message when input + max_tokens > WINDOW, else 200.
    class SglangContextUpstream(ThreadingHTTPServer):
        pass

    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        INPUT = 7897

        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n)) if n else {}
            mt = body.get("max_tokens", 0)
            seen.append(mt)
            total = H.INPUT + mt
            if total > WINDOW:
                err = json.dumps({
                    "error": {
                        "message": (f"Requested token count exceeds the model's "
                                    f"maximum context length of {WINDOW} tokens. "
                                    f"You requested a total of {total} tokens: "
                                    f"{H.INPUT} tokens from the input messages and "
                                    f"{mt} tokens for the completion."),
                        "code": 400,
                    }}).encode()
                self.send_response(400)
            else:
                err = json.dumps({"id": "x", "object": "chat.completion",
                                  "choices": [{"index": 0,
                                               "message": {"role": "assistant",
                                                          "content": "pong"}}]}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    httpd_up = ThreadingHTTPServer(("127.0.0.1", up_port), H)
    threading.Thread(target=httpd_up.serve_forever, daemon=True).start()
    try:
        # cap=64000 (the window) and NO context_window yet → first request sets
        # max_tokens=64000 (cap), overflows (7897+64000>64000), self-heals.
        cfg = __import__("dsh_openai_shim").ShimConfig(
            upstream=f"http://127.0.0.1:{up_port}/v1", listen_port=shim_port,
            effort_mode="off", token_cap=64000, context_window=0,
            provider_name="spark")
        Handler, _ = make_handler(cfg)
        httpd = ThreadingHTTPServer(("127.0.0.1", shim_port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        body = json.dumps({"model": "Socrates", "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{shim_port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            assert r.status == 200
            assert json.loads(r.read())["choices"][0]["message"]["content"] == "pong"

        # two attempts: first overflowed, second fit exactly.
        assert len(seen) == 2
        assert seen[0] == 64000                      # first: cap = full window (overflows)
        # second: window − EXACT reported input − reserve (64000 − 7897 − 128)
        assert seen[1] == 64000 - 7897 - 128
        assert seen[1] == 55975
        assert H.INPUT + seen[1] <= WINDOW           # the retry actually fits
        # the window was learned in-memory and persisted to proxy.conf
        assert cfg.context_window == 64000
        saved = json.loads((tmp_path / "proxy.conf").read_text())
        assert saved["providers"]["spark"]["context_window"] == 64000
        assert "self-heal" in capsys.readouterr().out
    finally:
        httpd.server_close()
        httpd_up.server_close()
