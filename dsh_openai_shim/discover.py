"""Auto-discover an upstream model's token limits (hermes-style).

The dsh shim clamps every request's ``max_tokens`` to a cap before forwarding.
A single global cap (the old 100000 default) breaks any custom provider whose
model has a smaller ceiling (e.g. a local sglang/vLLM server whose
``max_model_len`` is 64000 — the request is rejected with an HTTP 400 before a
single token is produced).

There are TWO distinct limits, and confusing them is exactly what broke spark:

* **output cap** — the most *completion* tokens the model may emit
  (``max_completion_tokens`` / ``max_output_tokens``).
* **context window** — the total ``input + output`` tokens the model accepts
  (``max_model_len`` / ``context_length`` / …). sglang/vLLM report
  ``max_model_len`` as this *total* window. Setting ``max_tokens`` to the full
  window therefore overflows a non-trivial prompt: ``input + max_tokens``
  exceeds the window and the upstream 400s with "…exceeds the model's maximum
  context length of W tokens. You requested a total of T tokens: I tokens from
  the input messages and O tokens [for the completion]".

Rather than make the student know their model's limits, we discover them the
same two ways Hermes does (``agent/model_metadata.py``):

1. **Endpoint probe** — ``GET {base}/v1/models`` and read what the server
   reports itself. Used when a provider is first added, so the right values are
   set *before* the first request.

2. **Error self-heal** — if the upstream still rejects a request, parse the
   real numbers out of the error message (output-cap and context-window forms
   alike) and retry once. This is the authoritative signal and covers endpoints
   that don't report the limit from ``/v1/models``.

Both are stdlib-only (``urllib``/``json``/``re``) — no third-party HTTP lib —
so they run inside the shim, which must stay dependency-free.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional

__all__ = [
    "discover_completion_cap",
    "discover_model_limits",
    "parse_completion_cap_from_error",
    "parse_context_window_error",
    "is_output_cap_error",
]

# Explicit completion/output ceilings, preferred over the context window.
_COMPLETION_KEYS = ("max_completion_tokens", "max_output_tokens", "max_tokens")
# Context-window fields (input+output). Safe upper bound for output: a model
# can never emit more completion tokens than its context window.
_CONTEXT_KEYS = (
    "max_model_len", "context_length", "context_window", "context_size",
    "max_context_length", "max_position_embeddings", "max_input_tokens",
    "max_sequence_length", "max_seq_len", "n_ctx_train", "n_ctx", "ctx_size",
)

# A discovered cap is only trusted if it is a sane token count.
_MIN_CAP = 256
_MAX_CAP = 10_000_000


def _extract_first_int(payload, keys):
    """First positive int among ``keys`` in ``payload`` (one level deep)."""
    if not isinstance(payload, dict):
        return None
    for k in keys:
        if k in payload:
            v = payload[k]
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    # vLLM/sglang sometimes nest settings one level down.
    for v in payload.values():
        if isinstance(v, dict):
            got = _extract_first_int(v, keys)
            if got is not None:
                return got
    return None


def _cap(cand: Optional[int]) -> Optional[int]:
    if cand is not None and _MIN_CAP <= cand <= _MAX_CAP:
        return cand
    return None


def discover_model_limits(
    base_url: str,
    api_key: str = "",
    timeout: int = 8,
) -> tuple[Optional[int], Optional[int]]:
    """Probe ``GET {base}/v1/models`` and return ``(output_cap, context_window)``.

    Each element is the model's reported value, or ``None`` when the endpoint
    can't be reached or doesn't report that limit (the caller then falls back
    to its defaults and lets the error self-heal learn the real one).

    An explicit completion/output ceiling (``max_completion_tokens`` /
    ``max_output_tokens``) wins for ``output_cap``. Otherwise — as with
    sglang/vLLM, which only report the *total* context window (``max_model_len``)
    — both are set from the context window, which is a safe upper bound: a model
    can never emit more completion tokens than its whole window. The *tightest*
    (min) window across any model on the endpoint is the universally-safe value.

    ``base_url`` may carry a trailing ``/v1`` or not; both forms are tried.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return None, None
    candidates = []
    if base.endswith("/v1"):
        candidates += [base, base[:-3]]
    else:
        candidates += [base + "/v1", base]

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for c in candidates:
        url = c.rstrip("/") + "/models"
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            payload = json.loads(raw.decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            continue
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list) or not models:
            continue
        # 1) an explicit completion/output cap wins outright.
        for m in models:
            completion = _cap(_extract_first_int(m, _COMPLETION_KEYS))
            if completion is not None:
                ctx = _cap(_extract_first_int(m, _CONTEXT_KEYS))
                if ctx is None:
                    ctx = completion
                return completion, ctx
        # 2) else the context window IS the safe bound for both.
        contexts: list[int] = []
        for m in models:
            ctx = _cap(_extract_first_int(m, _CONTEXT_KEYS))
            if ctx is not None:
                contexts.append(ctx)
        if contexts:
            win = min(contexts)
            return win, win
    return None, None


def discover_completion_cap(
    base_url: str,
    api_key: str = "",
    timeout: int = 8,
) -> Optional[int]:
    """Output-cap ceiling from ``/v1/models`` (or ``None``). See
    :func:`discover_model_limits` — returns just the output cap. Kept for
    backward compatibility with existing callers/tests."""
    cap, _ = discover_model_limits(base_url, api_key=api_key, timeout=timeout)
    return cap


def is_output_cap_error(message: str) -> bool:
    """True when an upstream error is a *max-tokens-too-large* (output-cap)
    error, as opposed to a *prompt-too-long* (input-overflow) error.

    Mirrors Hermes' distinction: an output-cap error means "lower the output
    cap", never "compress the prompt".
    """
    if not message:
        return False
    low = message.lower()
    return (
        "max_completion_tokens is too large" in low
        or "max_tokens is too large" in low
        or ("completion tokens" in low and
            any(w in low for w in ("at most", "too large", "maximum", "exceed")))
        or ("max_tokens" in low and
            any(w in low for w in ("too large", "maximum", "exceed", "greater than")))
        or "range of max_tokens should be" in low
    )


def parse_completion_cap_from_error(message: str) -> Optional[int]:
    """Extract the completion-token CEILING an upstream reported in an error.

    Returns ``None`` unless ``message`` is an output-cap error AND a cap can be
    read from it. Deliberately ignores the "too large: <sent-value>" figure —
    that number is the value we sent, not the limit — and returns only the
    server-stated maximum.
    """
    if not is_output_cap_error(message):
        return None
    low = message.lower()
    patterns = [
        # sglang: "…supports at most 64000 completion tokens"
        r'at most\s+(\d{2,})\s*(?:completion|output)\s+tokens?',
        r'supports\s+(?:at most\s+)?(\d{2,})\s*(?:completion|output)\s+tokens?',
        # "maximum 64000 completion tokens" / "max completion tokens: 64000"
        r'max(?:imum)?\s*(?:completion|output)\s+tokens?\s*(?:is|of|:)?\s*(\d{2,})',
        r'(\d{2,})\s*(?:completion|output)\s+tokens?\s*(?:is|:)?\s*the\s+(?:max(?:imum)?|limit)',
        # DashScope/Alibaba: "Range of max_tokens should be [1, 65536]"
        r'range of max_tokens should be\s*\[\s*\d+\s*,\s*(\d+)\s*\]',
        # generic: "maximum completion length is 64000"
        r'max(?:imum)?\s+(?:completion|output)\s+length\s*(?:is|:)?\s*(\d{2,})',
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            return _cap(int(m.group(1)))
    return None


def parse_context_window_error(
    message: str,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse a context-WINDOW error into ``(window, input_tokens, total)``.

    Distinct from :func:`parse_completion_cap_from_error` (which handles the
    output-cap form "at most N completion tokens"). This handles the form sglang
    emits when ``input + max_tokens`` overflows the total window:

        "Requested token count exceeds the model's maximum context length of
         64000 tokens. You requested a total of 71897 tokens: 7897 tokens from
         the input messages and 64000 tokens [for the completion]."

    ``window`` is the model's total context length (``W``); ``input_tokens`` the
    actual prompt token count (``I``); ``total`` the sum it reported (``T``).
    Any field the message doesn't state is ``None``. Returns all-``None`` when
    the message is not a context-window error.
    """
    if not message:
        return None, None, None
    low = message.lower()
    # Not a context-window error (no window stated, or it's the output-cap form
    # the other parser handles) -> bail without misfiring.
    if "context" not in low and "max_model_len" not in low:
        return None, None, None
    if not any(
        w in low
        for w in (
            "exceeds", "exceed", "greater than", "too large", "too long",
            "maximum", "maximum context", "maximum length", "limit",
            "overflow",
        )
    ):
        return None, None, None

    # W: the model's maximum context length. Anchored to the specific phrasing
    # (not a bare "N tokens", which would grab the wrong figure), with a guard
    # that rejects a match immediately followed by "…for the completion".
    window = None
    for w_pat in (
        r'max(?:imum)?\s+context\s+length\s+of\s+(\d{2,})\s+tokens?',
        r'max(?:imum)?\s+context\s+window\s+(?:of\s+)?(\d{2,})\s+tokens?',
        r'context\s+(?:length|window|size)\s+(?:of\s+)?(\d{2,})\s+tokens?',
        r'context\s+length\s+of\s+(\d{2,})',
        r'max(?:imum)?\s+context\s+(?:length|window)\s+is\s+(\d{2,})',
    ):
        for w_match in re.finditer(w_pat, low):
            num = int(w_match.group(1))
            after = low[w_match.end():w_match.end() + 18]
            if "completion" in after or "output" in after:
                continue
            window = num
            break
        if window is not None:
            break

    # I: the input (prompt) tokens the server actually counted.
    i_match = re.search(
        r'(\d+)\s+tokens?\s+from\s+the\s+input',
        low,
    ) or re.search(
        r'input\s+(?:tokens?\s+[:=]?\s*)(\d+)',
        low,
    ) or re.search(
        r'prompt\s+(?:tokens?\s+[:=]?\s*)(\d+)',
        low,
    )
    input_tokens = int(i_match.group(1)) if i_match else None

    # T: the total it requested.
    t_match = re.search(
        r'(?:total|sum) of\s*(\d{2,})\s+tokens?',
        low,
    ) or re.search(
        r'requested a total of\s*(\d{2,})',
        low,
    )
    total = int(t_match.group(1)) if t_match else None

    if window is None and input_tokens is None and total is None:
        return None, None, None
    return (
        _cap(window) if window is not None else None,
        input_tokens,
        total,
    )
