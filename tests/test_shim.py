import json
import threading

import pytest

from dsh_openai_shim import ShimConfig, apply_rewrites, make_handler


# ─────────────────────────────────────────────────────────────
# apply_rewrites: pure function, no network
# ─────────────────────────────────────────────────────────────

def _cfg(**kw):
    base = dict(upstream="http://up", token_cap=100000)
    base.update(kw)
    return ShimConfig(**base)


def test_map_high_to_medium():
    body = json.dumps({"model": "Socrates", "reasoning_effort": "high", "max_tokens": 50}).encode()
    new, changes = apply_rewrites(body, _cfg(effort_mode="map"))
    d = json.loads(new)
    assert d["reasoning_effort"] == "medium"
    assert any("high" in c and "medium" in c for c in changes)


def test_map_preserves_supported_value():
    body = json.dumps({"reasoning_effort": "xhigh"}).encode()
    new, changes = apply_rewrites(body, _cfg(effort_mode="map", effort_map={"high": "medium", "xhigh": "xhigh"}))
    d = json.loads(new)
    assert d["reasoning_effort"] == "xhigh"  # unchanged
    assert not any("reasoning_effort" in c for c in changes)


def test_drop_removes_effort():
    body = json.dumps({"reasoning_effort": "high"}).encode()
    new, changes = apply_rewrites(body, _cfg(effort_mode="drop"))
    d = json.loads(new)
    assert "reasoning_effort" not in d
    assert any("dropped" in c for c in changes)


def test_off_leaves_effort():
    body = json.dumps({"reasoning_effort": "high"}).encode()
    new, changes = apply_rewrites(body, _cfg(effort_mode="off"))
    d = json.loads(new)
    assert d["reasoning_effort"] == "high"
    assert "reasoning_effort" not in " ".join(changes)


def test_max_tokens_clamped_down():
    body = json.dumps({"max_tokens": 256000, "reasoning_effort": "high"}).encode()
    new, changes = apply_rewrites(body, _cfg(token_cap=100000, effort_mode="map"))
    d = json.loads(new)
    assert d["max_tokens"] == 100000
    assert any("max_tokens" in c for c in changes)


def test_max_tokens_below_cap_untouched():
    body = json.dumps({"max_tokens": 50}).encode()
    new, changes = apply_rewrites(body, _cfg(token_cap=100000, effort_mode="off"))
    d = json.loads(new)
    assert d["max_tokens"] == 50


def test_max_tokens_added_when_absent():
    body = json.dumps({"messages": []}).encode()
    new, changes = apply_rewrites(body, _cfg(token_cap=100000, effort_mode="off"))
    d = json.loads(new)
    assert d["max_tokens"] == 100000


def test_non_json_passthrough():
    body = b"not json at all"
    new, changes = apply_rewrites(body, _cfg())
    assert new == body
    assert changes == []


def test_zero_token_cap_disables_max():
    body = json.dumps({}).encode()
    new, changes = apply_rewrites(body, _cfg(token_cap=0, effort_mode="off"))
    d = json.loads(new)
    assert "max_tokens" not in d


# ─────────────────────────────────────────────────────────────
# end-to-end against a local fake upstream (no external network)
# ─────────────────────────────────────────────────────────────

class FakeUpstream:
    """Minimal OpenAI-style upstream that echoes what it received and records it."""

    def __init__(self, port):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.port = port
        self.last_body = {}
        self.last_auth = None
        self.requests = 0

        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _read(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                outer.requests += 1
                outer.last_auth = self.headers.get("Authorization")
                if n:
                    try:
                        outer.last_body = json.loads(self.rfile.read(n))
                    except Exception:
                        outer.last_body = {"raw": self.rfile.read(n).decode(errors="replace")}

            def do_POST(self):
                self._read()
                resp = json.dumps({
                    "id": "x", "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}}],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def do_GET(self):
                self._read()
                resp = json.dumps({"object": "list", "data": [{"id": "Socrates"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

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


def test_end_to_end_rewrites_and_forwards():
    import urllib.request

    up_port = _free_port()
    shim_port = _free_port()
    upstream = FakeUpstream(up_port)
    try:
        cfg = ShimConfig(
            upstream=f"http://127.0.0.1:{up_port}/v1",
            listen_port=shim_port,
            effort_mode="map",
            token_cap=100000,
        )
        Handler, ThreadingHTTPServer = make_handler(cfg)
        httpd = ThreadingHTTPServer(("127.0.0.1", shim_port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # client POSTs through the shim exactly like dsh does
        body = json.dumps({
            "model": "Socrates",
            "reasoning_effort": "high",
            "max_tokens": 256000,
            "messages": [{"role": "user", "content": "hi"}],
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{shim_port}/v1/chat/completions",
            data=body,
            headers={"Authorization": "Bearer client-key", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            assert r.status == 200
            out = json.loads(r.read())

        # upstream received the rewritten body
        assert upstream.last_body["reasoning_effort"] == "medium"
        assert upstream.last_body["max_tokens"] == 100000
        assert upstream.last_body["model"] == "Socrates"
        # client auth forwarded through
        assert upstream.last_auth == "Bearer client-key"
        # response passed through to client
        assert out["choices"][0]["message"]["content"] == "pong"
    finally:
        httpd.server_close()
        upstream.stop()


def test_upstream_key_override():
    import urllib.request

    up_port = _free_port()
    shim_port = _free_port()
    upstream = FakeUpstream(up_port)
    try:
        cfg = ShimConfig(
            upstream=f"http://127.0.0.1:{up_port}/v1",
            listen_port=shim_port,
            effort_mode="map",
            upstream_key="server-side-key",
        )
        Handler, ThreadingHTTPServer = make_handler(cfg)
        httpd = ThreadingHTTPServer(("127.0.0.1", shim_port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        body = json.dumps({"reasoning_effort": "high"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{shim_port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()
        assert upstream.last_auth == "Bearer server-side-key"
    finally:
        httpd.server_close()
        upstream.stop()


def test_invalid_effort_mode_raises():
    with pytest.raises(ValueError):
        _cfg(effort_mode="bogus")


# ─────────────────────────────────────────────────────────────
# ensure_shim: the lazy user-side sidecar starter (persona path)
# ─────────────────────────────────────────────────────────────

def test_ensure_shim_starts_and_serves():
    import urllib.request

    up_port = _free_port()
    shim_port = _free_port()
    upstream = FakeUpstream(up_port)
    try:
        from dsh_openai_shim import ensure_shim, is_shim_running

        assert not is_shim_running(shim_port)
        url = ensure_shim(
            port=shim_port,
            upstream=f"http://127.0.0.1:{up_port}",   # no trailing /v1 (client appends it)
            upstream_key="server-key",
        )
        assert url == f"http://127.0.0.1:{shim_port}/v1"
        assert is_shim_running(shim_port)

        # idempotent: second call returns immediately (same url, no error)
        assert ensure_shim(port=shim_port, upstream=f"http://127.0.0.1:{up_port}") == url

        # it actually rewrites + forwards
        body = json.dumps({"model": "Socrates", "reasoning_effort": "high",
                           "max_tokens": 256000, "messages": []}).encode()
        req = urllib.request.Request(f"{url}/chat/completions", data=body,
                                     headers={"Authorization": "Bearer dummy",
                                              "Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            assert json.loads(r.read())["choices"][0]["message"]["content"] == "pong"
        # upstream got rewritten body + the real key (not the dummy)
        assert upstream.last_body["reasoning_effort"] == "medium"
        assert upstream.last_body["max_tokens"] == 100000
        assert upstream.last_auth == "Bearer server-key"
    finally:
        # clean up the detached shim deterministically (frees the port for other tests)
        from dsh_openai_shim import stop_shim
        stop_shim(shim_port)
        upstream.stop()


def test_ensure_shim_requires_upstream(monkeypatch):
    from dsh_openai_shim import ensure_shim
    monkeypatch.delenv("DSH_SHIM_UPSTREAM", raising=False)
    with pytest.raises(RuntimeError, match="no upstream"):
        ensure_shim(port=_free_port(), upstream=None)
