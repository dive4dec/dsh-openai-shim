"""Core of dsh-openai-shim: a tiny stdlib-only OpenAI-compatible proxy.

It sits between an OpenAI-compatible client (dsh / DeepSeek Harness) and an
upstream OpenAI endpoint (Socrates / LiteLLM / any sglang/vLLM endpoint),
forwarding requests while applying two safe rewrites:

  * ``reasoning_effort``  — remap to a supported value or drop it
  * ``max_tokens``        — clamp so input+output stays under the context cap

No third-party dependencies. Importable for unit testing; runnable standalone.
"""
from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from typing import Any, Optional

__version__ = "0.2.1"

__all__ = ["__version__", "ShimConfig", "apply_rewrites", "make_handler", "serve",
           "ensure_shim", "shim_base_url", "is_shim_running", "DEFAULT_SHIM_PORT"]

from .runtime import (  # noqa: F401  (re-export)
    DEFAULT_SHIM_PORT,
    ensure_shim,
    is_shim_running,
    shim_base_url,
    shim_pidfile,
    stop_shim,
)


class ShimConfig:
    """Configuration for a single shim instance."""

    DEFAULT_EFFORT_MAP: dict[str, str] = {"high": "medium", "xhigh": "xhigh"}

    def __init__(
        self,
        upstream: str,
        listen_host: str = "127.0.0.1",
        listen_port: int = 8090,
        effort_mode: str = "map",          # "map" | "drop" | "off"
        effort_map: Optional[dict[str, str]] = None,
        token_cap: int = 100000,          # clamp completion max_tokens to this
        context_window: int = 0,         # upstream max_model_len (input+output); 0 = unknown
        upstream_key: Optional[str] = None,  # force a specific upstream key
        provider_name: Optional[str] = None, # proxy.conf provider this serves (for self-heal)
        timeout: int = 300,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        if effort_mode not in ("map", "drop", "off"):
            raise ValueError(f"effort_mode must be map|drop|off, got {effort_mode!r}")
        self.effort_mode = effort_mode
        self.effort_map = dict(effort_map or self.DEFAULT_EFFORT_MAP)
        self.token_cap = int(token_cap)
        self.context_window = int(context_window)
        self.upstream_key = upstream_key
        self.provider_name = provider_name
        self.timeout = int(timeout)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"ShimConfig(upstream={self.upstream!r}, port={self.listen_port}, "
            f"effort={self.effort_mode}, token_cap={self.token_cap})"
        )


def apply_rewrites(body: Optional[bytes], cfg: ShimConfig,
                   force_max_tokens: Optional[int] = None) -> tuple[bytes, list[str]]:
    """Apply the shim's rewrites to a request body.

    ``force_max_tokens`` (set only by the self-heal retry) pins ``max_tokens``
    to an exact value — the window minus the upstream's reported input count —
    overriding the usual cap/window clamp for that one attempt.

    Returns ``(new_body, changes)``. Bodies that are not JSON objects pass
    through untouched.
    """
    changes: list[str] = []
    if not body:
        return (body or b""), changes
    try:
        data: Any = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body, changes
    if not isinstance(data, dict):
        return body, changes

    # --- reasoning_effort ---------------------------------------------------
    if cfg.effort_mode != "off" and "reasoning_effort" in data:
        val = data["reasoning_effort"]
        if cfg.effort_mode == "drop":
            del data["reasoning_effort"]
            changes.append(f"dropped reasoning_effort={val!r}")
        else:
            new = cfg.effort_map.get(str(val))
            if new is not None and new != val:
                data["reasoning_effort"] = new
                changes.append(f"reasoning_effort {val!r}->{new!r}")

    # --- max_tokens clamp ---------------------------------------------------
    # Keep the client's explicit max_tokens when it is already small (a short
    # answer is fine for a small cap); only OVERRIDE it when it would overflow
    # the model's context window (see _clamp_max_tokens), so a long context
    # never 400s — this is what the self-heal relies on to make the retry fit.
    data.pop("force_max_tokens", None)
    clamp_desc = _clamp_max_tokens(data, cfg, force=force_max_tokens)
    if clamp_desc:
        changes.append(clamp_desc)

    if not changes:
        return body, changes
    return (json.dumps(data).encode("utf-8"), changes)


def _estimate_input_tokens(content) -> int:
    """Rough input-token estimate from a chat request's ``messages`` field.

    No tokenizer is available in a stdlib-only shim, so this counts characters
    across all message content and divides by ~4 (an English token is ~3.5-4
    chars). The estimate only needs to be *close* — it drives the proactive
    clamp, and the self-heal (which reads the server's EXACT input count from
    its error) corrects any misestimate on the retry. Slightly-overestimating
    is safe (it only makes the request more conservative, never less).
    """
    chars = 0

    def _count(c) -> None:
        nonlocal chars
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, str):
                    chars += len(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    chars += len(part["text"])

    for msg in content if isinstance(content, list) else []:
        if isinstance(msg, dict):
            _count(msg.get("content"))
    return max(1, chars // 4)


def _clamp_max_tokens(data: dict, cfg: "ShimConfig",
                      force: Optional[int] = None) -> Optional[str]:
    """Set ``data["max_tokens"]`` so ``input + output`` fits the model.

    The binding constraint is the *minimum* of the provider's output cap
    (``cfg.token_cap``) and the context window minus the estimated input and a
    small reserve (``cfg.context_window``) — because sglang/vLLM's
    ``max_model_len`` is the TOTAL window (input + output), not an output cap.
    ``force`` (the self-heal path) pins the value from the upstream's exact
    input count, overriding both. Returns a change description or ``None``.
    """
    if force is not None:
        forced = max(1, int(force))
        if data.get("max_tokens") != forced:
            prev = data.get("max_tokens")
            data["max_tokens"] = forced
            return (f"max_tokens {prev}->{forced} (context-fit, "
                    f"window={cfg.context_window})")
        return None
    cap = cfg.token_cap
    if not isinstance(cap, int) or cap <= 0:
        return None
    est = _estimate_input_tokens(data.get("messages"))
    eff = cap
    if isinstance(cfg.context_window, int) and cfg.context_window > 0:
        headroom = cfg.context_window - est - 128
        if headroom < cap:
            eff = max(1, headroom)
    prev = data.get("max_tokens")
    if isinstance(prev, int) and prev > eff:
        data["max_tokens"] = eff
        return f"max_tokens {prev}->{eff} (cap={cap}, window={cfg.context_window})"
    if "max_tokens" not in data and eff:
        data["max_tokens"] = eff
        return f"set max_tokens={eff} (cap={cap}, window={cfg.context_window})"
    return None


def _error_message(raw: bytes) -> str:
    """The human-readable ``message`` from an OpenAI-style error body, or ''."""
    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    err = payload.get("error")
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return err["message"]
    if isinstance(payload.get("message"), str):
        return payload["message"]
    return ""


def _parse_error_cap(raw: bytes) -> Optional[int]:
    """Completion-token ceiling reported in an upstream error body, or None."""
    message = _error_message(raw)
    if not message:
        return None
    from .discover import parse_completion_cap_from_error
    return parse_completion_cap_from_error(message)


def _parse_context_error(raw: bytes) -> tuple:
    """(window, input_tokens, total) from a context-window error body, or Nones."""
    message = _error_message(raw)
    if not message:
        return (None, None, None)
    from .discover import parse_context_window_error
    return parse_context_window_error(message)


def _persist_provider_cap(name: Optional[str], cap, context_window=None) -> None:
    """Persist a self-healed cap / context window back to proxy.conf (best-effort).

    The in-memory ``cfg.token_cap`` / ``cfg.context_window`` are what fix the
    in-flight retry; this write makes them survive the next restart. Failure to
    write (no name, read-only home, …) must never break a request — the
    in-memory fix stands on its own, so swallow errors.
    """
    if not name:
        return
    try:
        from .config import persist_provider_cap
        if persist_provider_cap(name, cap, context_window):
            print(f"[shim] persisted token_cap={cap} context_window="
                  f"{context_window} for provider {name!r}", flush=True)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[shim] could not persist cap: {e}", flush=True)


# Hop-by-hop / re-settable headers that must not be forwarded verbatim.
_STRIP_REQ_HEADERS = {
    "host", "content-length", "accept-encoding", "connection",
    "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailer", "transfer-encoding", "upgrade",
}
_STRIP_RES_HEADERS = {
    "content-length", "connection", "keep-alive", "transfer-encoding",
    "upgrade", "proxy-authenticate", "proxy-authorization",
}


def _filter_headers(headers: Any, strip: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        items = list(headers.items())
    except AttributeError:
        items = [(k, v) for k, v in (headers or [])]  # http.client-style list
    for k, v in items:
        if k.lower() not in strip:
            out[k] = v
    return out


def make_handler(cfg: ShimConfig):
    """Build ``(HandlerClass, ThreadingHTTPServer)`` bound to ``cfg``."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            try:
                print(f"[shim {cfg.listen_port}] {self.command} {self.path}", flush=True)
            except Exception:
                pass

        def _send_success(self, resp, raw: bytes) -> None:
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in _STRIP_RES_HEADERS:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_error(self, code: int, raw: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _do_forward_once(self, method: str, url: str, headers: dict,
                             data: Optional[bytes]):
            """Send one upstream attempt.

            Returns ``None`` once a response has been written to the client
            (a success, or a network/502 failure — both terminal). Returns
            ``(code, raw)`` when the upstream answered with an HTTP error that
            has NOT yet been sent to the client — so the caller can either
            self-heal (lower the cap and retry) or forward the error.
            """
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                resp = urllib.request.urlopen(req, timeout=cfg.timeout)
            except urllib.error.HTTPError as e:
                raw = e.read()
                print(f"[shim {cfg.listen_port}] upstream HTTP {e.code}: {raw[:200]!r}",
                      flush=True)
                return e.code, raw
            except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
                msg = json.dumps(
                    {"error": {"message": f"shim upstream error: {e}", "code": 502}}
                ).encode()
                self._send_error(502, msg)
                return None
            raw = resp.read()
            self._send_success(resp, raw)
            return None

        def _forward(self, method: str, body: Optional[bytes]) -> None:
            url = cfg.upstream + self.path
            headers = _filter_headers(self.headers, _STRIP_REQ_HEADERS)
            if cfg.upstream_key:
                headers["Authorization"] = f"Bearer {cfg.upstream_key}"
            data: Optional[bytes] = None
            if method == "POST" and body is not None:
                data = body
            elif method != "POST":
                # GET/HEAD/OPTIONS — no self-heal, one shot.
                outcome = self._do_forward_once(method, url, headers, None)
                if outcome is None:
                    return
                code, raw = outcome
                self._send_error(code, raw)
                return

            orig_body = data
            # When set (by the self-heal on the first attempt), the retry pins
            # max_tokens to this exact value instead of the usual clamp.
            retry_force: Optional[int] = None
            for attempt in (1, 2):
                rewritten, changes = apply_rewrites(
                    orig_body, cfg, force_max_tokens=retry_force)
                for c in changes:
                    print(f"[shim {cfg.listen_port}] rewrite: {c}", flush=True)
                # log the model field (helps debug which model reached upstream)
                try:
                    _m = json.loads(rewritten).get("model")
                    if _m:
                        print(f"[shim {cfg.listen_port}] model: {_m}", flush=True)
                except Exception:
                    pass
                h = dict(headers)
                h["Content-Length"] = str(len(rewritten))
                outcome = self._do_forward_once("POST", url, h, rewritten)
                if outcome is None:
                    return  # a client response was written (success / 502)
                code, raw = outcome
                if attempt == 1:
                    # --- Self-heal (hermes-style), two distinct error forms ---
                    # 1) CONTEXT-WINDOW: "…exceeds the model's maximum context
                    #    length of W tokens. You requested a total of T tokens:
                    #    I tokens from the input messages…" → the prompt plus
                    #    our max_tokens overflowed the TOTAL window. Retry with
                    #    max_tokens = W - I - reserve (the server's EXACT input
                    #    count), and persist the window so future turns clamp
                    #    proactively. This is the sglang/vLLM case that broke
                    #    spark (max_model_len is input+output, not the cap).
                    window, in_tok, _total = _parse_context_error(raw)
                    if window is not None:
                        if in_tok is None:
                            in_tok = _estimate_input_tokens(
                                json.loads(rewritten).get("messages"))
                        retry_force = max(1, window - in_tok - 128)
                        if not cfg.context_window or cfg.context_window != window:
                            cfg.context_window = window
                        print(f"[shim {cfg.listen_port}] self-heal: context "
                              f"window={window}, input={in_tok} tokens -> retry "
                              f"max_tokens={retry_force}", flush=True)
                        _persist_provider_cap(cfg.provider_name, cfg.token_cap,
                                              window)
                        continue
                    # 2) OUTPUT-CAP: "…at most N completion tokens" / "max_tokens
                    #    is too large" → lower the output cap and retry.
                    cap = _parse_error_cap(raw)
                    if cap is not None and cap < cfg.token_cap:
                        old = cfg.token_cap
                        cfg.token_cap = max(cap, 256)
                        print(f"[shim {cfg.listen_port}] self-heal: output cap "
                              f"{old}->{cfg.token_cap} (upstream reports max {cap})",
                              flush=True)
                        _persist_provider_cap(cfg.provider_name, cfg.token_cap)
                        continue
                self._send_error(code, raw)
                return

        def do_GET(self) -> None:
            self._forward("GET", None)

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(n) if n else b""
            self._forward("POST", body)

        def do_HEAD(self) -> None:
            self._forward("GET", None)

        def do_OPTIONS(self) -> None:
            self._forward("GET", None)

    return Handler, ThreadingHTTPServer


def serve(cfg: ShimConfig) -> None:
    """Run the shim as a threaded HTTP server (blocks until interrupted)."""
    Handler, ThreadingHTTPServer = make_handler(cfg)
    httpd = ThreadingHTTPServer((cfg.listen_host, cfg.listen_port), Handler)
    httpd.daemon_threads = True
    print(
        f"[shim] listening on http://{cfg.listen_host}:{cfg.listen_port} "
        f"-> {cfg.upstream} (effort={cfg.effort_mode}, token_cap={cfg.token_cap})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
