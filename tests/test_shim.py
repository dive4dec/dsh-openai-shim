import json
import socket
import threading
import time
import urllib.error
from types import SimpleNamespace

import pytest

from dsh_openai_shim import ShimConfig, apply_rewrites, make_handler
from dsh_openai_shim import _SSEMessageStartDedup
from dsh_openai_shim import _is_thinking_error, _error_message


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


# ─────────────────────────────────────────────────────────────
# streaming (SSE) pass-through + transport hardening
#
# The historical "TRANSPORT" bug: a mid-stream cut raised IncompleteRead from
# `raw = resp.read()` OUTSIDE any try, killing the handler thread and the whole
# shim. These tests pin the new behaviour:
#   * SSE is forwarded chunk-by-chunk (the client sees tokens live, not after
#     the whole generation is buffered)
#   * a mid-stream cut is a clean terminal SSE event, NOT a raw close
#   * a pre-response transport failure retries once, then a clean 502
#   * the shim SURVIVES a mid-stream cut and serves the next request
# ─────────────────────────────────────────────────────────────

def _sse_lines(n):
    """n SSE `data:` deltas + the [DONE] marker (list of bytes)."""
    out = [f'data: {{"id":"x","choices":[{{"delta":{{"content":"tok{i} "}}}}]}}\n\n'.encode()
           for i in range(1, n + 1)]
    out.append(b"data: [DONE]\n\n")
    return out


def _make_fake_sse_upstream(port):
    """Raw-socket fake upstream (sendall only — raw sockets have no .flush()).

    Three behaviours by request path:
      * /sse — 200, no framing, drips SSE events with a 50ms gap (liveness is
        measurable) then a clean close (EOF).
      * /sse-chunked — 200, Transfer-Encoding: chunked, the SSE events delivered
        as HTTP chunks (this is how LiteLLM actually serves SSE), then a clean
        close. Used to prove the shim de-chunks and does NOT leak hex chunk-size
        markers into the forwarded stream.
      * /cut — 200, no framing, one SSE event, then SO_LINGER(1,0) → a TCP RST
        mid-stream. This is the REAL production cut signature: the shim's
        read1 raises ConnectionResetError.
      * other — a normal non-stream JSON 200 (post-cut liveness check).
    """
    import struct

    def _serve_conn(c):
        try:
            buf = b""
            while b"\r\n\r\n" not in buf:
                d = c.recv(65536)
                if not d:
                    return
                buf += d
            # Split head / already-received body. The body (or part of it) may
            # have arrived in the SAME recv as the head on localhost — we must
            # NOT re-read those bytes or we hang waiting for data already sent.
            head, _, body = buf.partition(b"\r\n\r\n")
            path = head.split(b" ", 2)[1].split(b"?", 1)[0].decode()
            n = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    n = int(line.split(b":", 1)[1])
            while len(body) < n:
                d = c.recv(65536)
                if not d:
                    break
                body += d
            if path == "/cut":
                c.sendall(b"HTTP/1.1 200 OK\r\n"
                          b"Content-Type: text/event-stream\r\n\r\n")
                c.sendall(b'data: {"partial":true}\n\n')
                time.sleep(0.2)  # let the shim read the event, then
                c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                             struct.pack("ii", 1, 0))  # RST on close
                c.close()
                return
            if path == "/sse":
                c.sendall(b"HTTP/1.1 200 OK\r\n"
                          b"Content-Type: text/event-stream\r\n\r\n")
                for ev in _sse_lines(8):
                    c.sendall(ev)
                    time.sleep(0.05)
                c.close()
                return
            if path == "/sse-chunked":
                # 200 + Transfer-Encoding: chunked, SSE delivered as HTTP chunks
                # (the real LiteLLM shape). urllib de-chunks resp.read1(); the
                # OLD resp.fp.read1(4096) forwarded these hex size lines into
                # the stream -> dsh "SSE contains invalid JSON".
                c.sendall(b"HTTP/1.1 200 OK\r\n"
                          b"Content-Type: text/event-stream\r\n"
                          b"Transfer-Encoding: chunked\r\n\r\n")
                for ev in _sse_lines(8):
                    c.sendall(("%x\r\n" % len(ev)).encode() + ev + b"\r\n")
                    time.sleep(0.05)
                c.sendall(b"0\r\n\r\n")
                c.close()
                return
            resp = json.dumps({"id": "x", "object": "chat.completion",
                               "choices": [{"index": 0, "message": {
                                   "role": "assistant", "content": "pong"}}]}).encode()
            c.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(resp)).encode() +
                b"\r\n\r\n" + resp)
            c.close()
        except OSError:
            pass

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(16)

    def _accept_loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=_serve_conn, args=(c,), daemon=True).start()

    threading.Thread(target=_accept_loop, daemon=True).start()
    return SimpleNamespace(stop=srv.close)


def _inproc_shim(upstream_url, timeout=300):
    """Start an in-process shim (the code under test, in this interpreter) on a
    free port pointed at `upstream_url`. The shim forwards to
    `upstream_url + client_path`, so pass the upstream WITHOUT a path suffix to
    control exactly what the fake upstream sees. Returns (httpd, base_url, port)."""
    from http.server import ThreadingHTTPServer
    cfg = ShimConfig(upstream=upstream_url, listen_port=_free_port(),
                     effort_mode="off", timeout=timeout)
    Handler, ThreadingHTTPServer = make_handler(cfg)
    httpd = ThreadingHTTPServer(("127.0.0.1", cfg.listen_port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{cfg.listen_port}", cfg.listen_port


def _stream_request(base, path, body, read_seconds):
    """POST a streaming request, return (raw_bytes, first_byte_after_sec, status).

    Uses ``resp.read1(1024)`` (one socket read) rather than ``resp.read(n)`` so
    the first-byte timestamp reflects when the shim actually sent data — a
    buffered ``read(n)`` would coalesce the whole stream and defeat the
    liveness assertion.
    """
    import http.client as _hc
    from urllib.parse import urlsplit
    _p = urlsplit(base)
    t0 = time.monotonic()
    first = None
    data = b""
    conn = _hc.HTTPConnection(_p.hostname, _p.port, timeout=read_seconds + 10)
    conn.request("POST", _p.path.rstrip("/") + path, body=body,
                 headers={"Content-Type": "application/json",
                          "Authorization": "Bearer x"})
    resp = conn.getresponse()
    status = resp.status
    while True:
        try:
            chunk = resp.read1(1024)
        except Exception:
            break  # a raw client-side cut: stop, we have what we got
        if not chunk:
            break
        if first is None:
            first = time.monotonic() - t0
        data += chunk
    conn.close()
    return data, first, status


def test_sse_passthrough_streams_live():
    """The client must see the first delta well BEFORE the upstream finishes —
    proof the shim forwards as it arrives (read1) rather than buffering the
    whole generation and delivering it at the end (read)."""
    up_port = _free_port()
    up = _make_fake_sse_upstream(up_port)
    httpd, base, _port = _inproc_shim(f"http://127.0.0.1:{up_port}")
    try:
        body = json.dumps({"model": "Socrates", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        data, first, status = _stream_request(base, "/sse", body, 10)
        assert status == 200
        assert b"tok1" in data            # saw the first token
        assert b"[DONE]" in data          # and the terminal marker
        # 8 events @ 50ms = ~0.4s upstream; a live first byte must land well
        # before the stream finishes (buffered would land at ~0.4s+).
        assert first is not None and first < 0.25
    finally:
        httpd.server_close()
        up.stop()


def test_sse_chunked_upstream_dechunks_no_markers():
    """Upstream serves SSE as Transfer-Encoding: chunked (the real LiteLLM
    shape). The shim MUST de-chunk: the bytes the client receives contain
    every SSE event verbatim and NO hex chunk-size marker lines. The old
    resp.fp.read1(4096) leaked those markers into the stream, and when a
    data: JSON payload straddled a chunk boundary dsh aborted with
    "DeepSeek Messages SSE contains invalid JSON".
    """
    up_port = _free_port()
    up = _make_fake_sse_upstream(up_port)
    httpd, base, _port = _inproc_shim(f"http://127.0.0.1:{up_port}")
    try:
        body = json.dumps({"model": "Socrates", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        data, _first, status = _stream_request(base, "/sse-chunked", body, 10)
        assert status == 200
        assert b"tok1" in data            # first event present, intact
        assert b"tok8" in data            # last event present
        assert b"[DONE]" in data          # terminal marker
        # every data: JSON event must parse ([DONE] is the terminal marker,
        # not JSON — skip it)
        for line in data.split(b"\n"):
            if line.startswith(b"data:") and line.strip() != b"data: [DONE]":
                json.loads(line[5:])
        # THE POINT: no leaked chunk-size marker (a line of pure hex).
        for line in data.replace(b"\r", b"").split(b"\n"):
            s = line.strip()
            assert not (s and all(c in b"0123456789abcdef" for c in s)), \
                f"leaked chunk marker: {line!r}"
    finally:
        httpd.server_close()
        up.stop()


def test_sse_midstream_cut_is_clean_and_shim_survives():
    """A genuine mid-stream TCP RST must:
      1. forward whatever was received,
      2. emit a clean terminal SSE event (not a raw client-side reset), and
      3. leave the shim ALIVE to serve the next request.
    """
    up_port = _free_port()
    up = _make_fake_sse_upstream(up_port)
    httpd, base, _port = _inproc_shim(f"http://127.0.0.1:{up_port}")
    try:
        body = json.dumps({"model": "Socrates", "stream": True, "messages": []}).encode()
        data, _first, status = _stream_request(base, "/cut", body, 10)
        assert status == 200                        # we committed to the stream
        assert b"partial" in data                   # forwarded the bytes before the cut
        assert b"upstream stream interrupted" in data  # clean terminal event, not a raw close
        # THE POINT: the shim survived the RST and still serves the next request.
        assert _stream_request_ok(base)
    finally:
        httpd.server_close()
        up.stop()


def _stream_request_ok(base):
    """A plain non-stream request after a prior cut — proves the shim is alive."""
    import urllib.request
    body = json.dumps({"model": "Socrates", "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = urllib.request.Request(base + "/ok", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer x"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def test_prebyte_transport_error_retries_once_then_502():
    """A connection-level failure BEFORE any response is a safe full-request
    retry (the shim is stateless); after it fails again, a clean 502 (not a raw
    close) reaches the client."""
    dead = _free_port()  # nothing listening here: connect fails pre-response
    httpd, base, _port = _inproc_shim(f"http://127.0.0.1:{dead}")
    try:
        import urllib.request
        body = json.dumps({"model": "Socrates", "stream": False,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        code = None
        try:
            urllib.request.urlopen(req, timeout=15)
        except urllib.error.HTTPError as e:
            code = e.code
        # dead port -> clean 502 after the single retry, not a bare close
        assert code == 502
    finally:
        httpd.server_close()


# ─────────────────────────────────────────────────────────────
# message_start dedup — the dsh 0.2.0-rc.1 / LiteLLM fix
# ─────────────────────────────────────────────────────────────

# A LiteLLM-shaped Anthropic stream: TWO message_start (same id), a thinking
# block, text, message_delta, message_stop. This is the exact upstream shape
# that dsh 0.2.0-rc.1 rejects with "duplicate message_start / MALFORMED_RESPONSE".
def _messages_stream(events, dup_start=True):
    start = (b'event: message_start\n'
             b'data: {"type":"message_start","message":{"id":"msg_abc",'
             b'"role":"assistant","model":"Socrates"}}\n\n')
    evs = [start]
    if dup_start:
        evs.append(start)  # the bug: LiteLLM emits it twice
    evs += [
        b'event: content_block_start\n'
        b'data: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"thinking","thinking":""}}\n\n',
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"thinking_delta","thinking":"hello"}}\n\n',
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","stop_reason":"end_turn"}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    if events is not None:
        evs = events
    return b"".join(evs)


def test_dedup_unit_drops_same_id_start():
    d = _SSEMessageStartDedup()
    out = b"".join(d.feed(_messages_stream(None))) + d.flush()
    assert out.count(b"event: message_start") == 1   # 2 upstream -> 1 forwarded
    # every other event survives, byte-identical, in order
    for needle in (b"content_block_start", b"thinking_delta",
                   b"content_block_stop", b"message_delta", b"message_stop"):
        assert needle in out
    # the original (first) message_start payload is preserved
    assert b'"id":"msg_abc"' in out


def test_dedup_unit_keeps_distinct_id_start():
    # Two DIFFERENT message ids must BOTH be forwarded (a legit multi-message
    # stream, or the first id unparseable -> we must not drop real content).
    a = (b'event: message_start\n'
         b'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n')
    b = (b'event: message_start\n'
         b'data: {"type":"message_start","message":{"id":"msg_2"}}\n\n')
    d = _SSEMessageStartDedup()
    out = b"".join(d.feed(a + b)) + d.flush()
    assert out.count(b"event: message_start") == 2
    assert b"msg_1" in out and b"msg_2" in out


def test_dedup_unit_openai_stream_is_noop():
    # A normal OpenAI stream has no "event: message_start" lines -> unchanged.
    evs = (b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n'
           b'data: [DONE]\n\n')
    d = _SSEMessageStartDedup()
    out = b"".join(d.feed(evs)) + d.flush()
    assert out == evs


def test_dedup_unit_split_across_chunks():
    # The duplicate start's "\n\n" terminator may land in a different TCP chunk
    # than its body (or the event itself is split mid-line). The filter must
    # still recognize + drop it and not corrupt the following event.
    stream = _messages_stream(None)
    d = _SSEMessageStartDedup()
    out = b""
    for i in range(0, len(stream), 3):        # 3-byte dribble = worst-case splits
        out += b"".join(d.feed(stream[i:i + 3]))
    out += d.flush()
    assert out.count(b"event: message_start") == 1
    assert b"message_stop" in out


def _make_fake_messages_upstream(port):
    """Raw-socket upstream that, for /messages, emits the LiteLLM-shaped
    DUPLICATE message_start stream (dripped), then EOF."""
    def _serve_conn(c):
        try:
            buf = b""
            while b"\r\n\r\n" not in buf:
                d = c.recv(65536)
                if not d:
                    return
                buf += d
            head, _, body = buf.partition(b"\r\n\r\n")
            n = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    n = int(line.split(b":", 1)[1])
            while len(body) < n:
                d = c.recv(65536)
                if not d:
                    break
                body += d
            c.sendall(b"HTTP/1.1 200 OK\r\n"
                      b"Content-Type: text/event-stream\r\n\r\n")
            for ev in _messages_stream(None).split(b"\n\n"):
                if ev:
                    c.sendall(ev + b"\n\n")
                    time.sleep(0.03)
            c.close()
        except OSError:
            pass

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(16)

    def _accept_loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=_serve_conn, args=(c,), daemon=True).start()

    threading.Thread(target=_accept_loop, daemon=True).start()
    return SimpleNamespace(stop=srv.close)


def test_sse_duplicate_message_start_is_dropped_end_to_end():
    """The production failure, end to end: an upstream that emits two
    message_start (same id) must reach the client as ONE, with the rest of the
    stream intact and still streaming live."""
    up_port = _free_port()
    up = _make_fake_messages_upstream(up_port)
    httpd, base, _port = _inproc_shim(f"http://127.0.0.1:{up_port}")
    try:
        body = json.dumps({"model": "Socrates", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        data, first, status = _stream_request(base, "/messages", body, 10)
        assert status == 200
        text = data.decode("utf-8", "replace")
        # THE POINT: exactly one message_start event survives the shim
        assert text.count("event: message_start") == 1
        # the rest of the stream is intact and in order
        assert "content_block_delta" in text and "thinking_delta" in text
        assert "message_delta" in text and "message_stop" in text
        # and it still streamed live (first byte well before the end)
        assert first is not None and first < 0.3
    finally:
        httpd.server_close()
        up.stop()


def test_dedup_unit_chunked_wire_prefix_first_event():
    # THE REAL LITELLM SHAPE: the stream is Transfer-Encoding: chunked and the
    # shim reads the raw wire, so the FIRST message_start event arrives with a
    # hex chunk-size line (b"2be\r\n" = 666) glued to its front, while the
    # DUPLICATE second start is clean. The filter must still drop the duplicate
    # even though the first event does not begin with "event:".
    chunked_first = (b"2be\r\n"
                     b"event: message_start\n"
                     b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_chunked\"}}\n\n")
    clean_second = (b"event: message_start\n"
                    b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_chunked\"}}\n\n")
    rest = (b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\"}\n\n"
            b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
    d = _SSEMessageStartDedup()
    out = b"".join(d.feed(chunked_first + clean_second + rest)) + d.flush()
    assert out.count(b"event: message_start") == 1   # 2 upstream -> 1 forwarded
    assert b"content_block_stop" in out and b"message_stop" in out
    assert b"msg_chunked" in out


# ─────────────────────────────────────────────────────────────
# Extended-thinking self-heal (dsh sends thinking:{type:enabled}; strict
# backends 400 on it → shim strips the field and retries once)
# ─────────────────────────────────────────────────────────────

def _thinking_err(message: str) -> bytes:
    return json.dumps({"type": "error", "error": {
        "type": "invalid_request_error", "message": message}}).encode()


def test_is_thinking_error_budget_required():
    # The socratic/ai-test form, captured verbatim from the live 400.
    assert _is_thinking_error(_thinking_err(
        "thinking: Value error, thinking.budget_tokens is required "
        "when thinking.type is 'enabled'"))


def test_is_thinking_error_no_reasoning_parser():
    # The spark form, captured verbatim from the live 400.
    assert _is_thinking_error(_thinking_err(
        "Anthropic thinking is not supported for models without a reasoning parser"))


def test_is_thinking_error_ignores_unrelated():
    assert not _is_thinking_error(_thinking_err("max_tokens is too large"))
    assert not _is_thinking_error(_thinking_err("not a json body"))
    assert not _is_thinking_error(b"")
    # an unrelated error that merely MENTIONS thinking in prose is not matched
    assert not _is_thinking_error(_thinking_err(
        "provider did not accept the request"))


def test_apply_rewrites_drop_thinking():
    body = json.dumps({"model": "Socrates", "thinking": {"type": "enabled"},
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    # not dropped by default
    new, changes = apply_rewrites(body, _cfg(effort_mode="off"), drop_thinking=False)
    assert "thinking" in json.loads(new)
    # dropped when asked
    new, changes = apply_rewrites(body, _cfg(effort_mode="off"), drop_thinking=True)
    out = json.loads(new)
    assert "thinking" not in out
    assert any("thinking" in c for c in changes)
    # a body with NO thinking field is untouched (no spurious change) —
    # token_cap=0 disables the max_tokens clamp so the only possible change
    # would be a spurious one from the thinking drop itself.
    plain = json.dumps({"model": "Socrates",
                        "messages": [{"role": "user", "content": "hi"}]}).encode()
    new2, changes2 = apply_rewrites(plain, _cfg(effort_mode="off", token_cap=0),
                                    drop_thinking=True)
    assert new2 == plain and changes2 == []


def test_end_to_end_thinking_self_heal_retries_and_drops():
    """End to end: upstream rejects thinking:{type:enabled} with the budget
    400, the shim retries WITHOUT the thinking field, and the client gets a
    clean 200. Mirrors the real socratic/ai-test failure."""
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        seen = []
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n))
            H.seen.append(body)
            if "thinking" in body:
                raw = _thinking_err(
                    "thinking: Value error, thinking.budget_tokens is required "
                    "when thinking.type is 'enabled'")
                self.send_response(400)
            else:
                raw = json.dumps({"id": "x", "type": "message", "content":
                    [{"type": "text", "text": "pong"}]}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    up_port = _free_port(); shim_port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", up_port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        cfg = ShimConfig(upstream=f"http://127.0.0.1:{up_port}/v1",
                         listen_port=shim_port, effort_mode="off", token_cap=100000)
        Handler, ThreadingHTTPServer = make_handler(cfg)
        shttpd = ThreadingHTTPServer(("127.0.0.1", shim_port), Handler)
        threading.Thread(target=shttpd.serve_forever, daemon=True).start()
        try:
            body = json.dumps({"model": "Socrates", "thinking": {"type": "enabled"},
                               "max_tokens": 1024,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{shim_port}/v1/messages",
                data=body, headers={"Authorization": "Bearer k",
                                    "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=20) as r:
                assert r.status == 200
                out = json.loads(r.read())
            assert out["content"][0]["text"] == "pong"
            # two upstream hits: first WITH thinking (400), second WITHOUT (200)
            assert len(H.seen) == 2
            assert "thinking" in H.seen[0]
            assert "thinking" not in H.seen[1]
        finally:
            shttpd.server_close()
    finally:
        httpd.server_close()


def test_apply_rewrites_drop_output_config():
    body = json.dumps({"model": "m", "output_config": {"effort": "high"},
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    # not dropped by default
    new, _ = apply_rewrites(body, _cfg(effort_mode="off"), drop_output_config=False)
    assert "output_config" in json.loads(new)
    # dropped when asked
    new, changes = apply_rewrites(body, _cfg(effort_mode="off"), drop_output_config=True)
    assert "output_config" not in json.loads(new)
    assert any("output_config" in c for c in changes)
    # no field -> untouched (token_cap=0 disables the clamp)
    plain = json.dumps({"model": "m",
                        "messages": [{"role": "user", "content": "hi"}]}).encode()
    new2, changes2 = apply_rewrites(plain, _cfg(effort_mode="off", token_cap=0),
                                    drop_output_config=True)
    assert new2 == plain and changes2 == []


def test_end_to_end_stacked_self_heal_thinking_then_output_config():
    """End to end: a strict backend 400s on ``thinking`` AND 500s on
    ``output_config``. The single request must stack BOTH self-heals (drop
    thinking, then drop output_config) and reach a clean 200. Mirrors the real
    socratic/ai-test failure, where the first (thinking) fix alone was not
    enough — the retry still 500'd on output_config."""
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    _oc = json.dumps({"type": "error", "error": {"type": "api_error",
                  "message": "Internal server error"}}).encode()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        seen = []
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n))
            H.seen.append(body)
            if "thinking" in body:
                raw = _thinking_err(
                    "thinking: Value error, thinking.budget_tokens is required "
                    "when thinking.type is 'enabled'")
                self.send_response(400)
            elif "output_config" in body:
                raw = _oc
                self.send_response(500)
            else:
                raw = json.dumps({"id": "x", "type": "message", "content":
                    [{"type": "text", "text": "pong"}]}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    up_port = _free_port(); shim_port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", up_port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        cfg = ShimConfig(upstream=f"http://127.0.0.1:{up_port}/v1",
                         listen_port=shim_port, effort_mode="off", token_cap=100000)
        Handler, ThreadingHTTPServer = make_handler(cfg)
        shttpd = ThreadingHTTPServer(("127.0.0.1", shim_port), Handler)
        threading.Thread(target=shttpd.serve_forever, daemon=True).start()
        try:
            body = json.dumps({"model": "Socrates", "thinking": {"type": "enabled"},
                               "output_config": {"effort": "high"}, "max_tokens": 1024,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{shim_port}/v1/messages",
                data=body, headers={"Authorization": "Bearer k",
                                    "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=20) as r:
                assert r.status == 200
                out = json.loads(r.read())
            assert out["content"][0]["text"] == "pong"
            # THREE upstream hits: thinking+oc (400) -> oc (500) -> neither (200)
            assert len(H.seen) == 3, f"expected 3 attempts, got {len(H.seen)}"
            assert "thinking" in H.seen[0] and "output_config" in H.seen[0]
            assert "thinking" not in H.seen[1] and "output_config" in H.seen[1]
            assert "thinking" not in H.seen[2] and "output_config" not in H.seen[2]
        finally:
            shttpd.server_close()
    finally:
        httpd.server_close()
