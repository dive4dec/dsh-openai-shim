# dsh-openai-shim Changelog

All notable changes to the `dsh-openai-shim` package are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.7] — 2026-10-01

### Added
- **Proactive context-window detection via `GET /v1/models` enrichment
  (hermes-style).** SGLang/vLLM report a model's context window as
  `max_model_len` — a field dsh's own deduce does **not** read (it only looks
  at `contextWindow` / `context_window` / `context_length` / `max_input_tokens`
  / `limit.context`). So against such an endpoint dsh couldn't read the real
  limit and silently fell back to a hardcoded `65536` default, then the first
  long turn 400'd with "…exceeds the model's maximum context length of 64000
  tokens". The shim is the one hop every provider goes through, so it now
  intercepts `GET /v1/models` and, for each model that reports a window under
  any hermes-recognised name but not `context_window`, copies that value into
  `context_window` — a field dsh DOES read. The original upstream field is
  preserved (never stripped), and a body that reports no window is forwarded
  byte-for-byte unchanged. This makes dsh resolve the real window **before the
  first request**, complementing (not replacing) the existing self-heal that
  learns the same number from a first context-length 400.
- Tests: pure `enrich_models_body` coverage (SGLang `max_model_len`, LiteLLM
  `max_input_tokens`, no-clobber, no-window passthrough, non-JSON, multi-model)
  plus two full end-to-end tests through a real handler + fake upstream.

## [0.2.6] — 2026-10-01

### Fixed
- **SSE chunk-size markers leaking into the streamed body →
  `DeepSeek Messages SSE contains invalid JSON`.** The stream-forwarder read
  the upstream body with `resp.fp.read1(4096)` — the *raw socket* — which
  bypasses Python's chunked-encoding de-chunking. The upstream (LiteLLM)
  serves `Transfer-Encoding: chunked`, so the hex chunk-size markers
  (`2be\r\n`, `85\r\n`, …) were forwarded into the SSE byte stream. dsh's SSE
  parser only tolerates a marker that lands on a standalone line; when a
  `data:` JSON payload straddles a chunk boundary the marker lands mid-JSON
  and the stream aborts with "SSE contains invalid JSON". Fixed by reading
  `resp.read1(4096)` (HTTPResponse) instead: it de-chunks but still returns a
  buffer as soon as any data is available, so live token-by-token streaming
  is preserved.

## [0.2.5] — 2026-09-29

### Fixed
- **Completed the non-litellm dsh fix that 0.2.4 only half-finished.** 0.2.4's
  single self-heal dropped the `thinking` field, but the *retry* still failed on
  two backends, because the shim's retry loop applied only **one** self-heal per
  request and two further problems remained:
  - **socratic (`ai-test`)** still 500'd (opaque "Internal server error") on
    dsh's nonstandard `output_config: {effort: "high"}` field (added by dsh's
    `pi-ai` layer). Bisected in-pod: dropping *only* `output_config` → 200.
  - **spark (64k window)** overflowed its context: dsh set `max_tokens` from its
    own (under-) input estimate, and the shim's proactive clamp over-saw the
    input because its estimator counted **only `messages`** (~145 tok) instead of
    the full prompt (`messages` + `system` + **24 tool definitions** ≈ 5068 tok).
    `input + max_tokens` then exceeded the 64000 window.
  
  Three changes:
  1. **`output_config` self-heal** — on a 500 *with* the field in the request,
     drop it and retry (a genuine 500 without the field is NOT masked).
  2. **Stacking retry loop** — up to 3 attempts; each self-heal (context-window,
     output-cap, thinking, output_config) is idempotent and can fire, so a single
     request can accumulate the multiple adaptations it needs (e.g. drop thinking
     *and* re-clamp max_tokens *and* drop output_config).
  3. **Full-input estimator** — counts `messages` + `system` + `tools` (not just
     `messages`), so the proactive `max_tokens` clamp sets a safe value on the
     *first* request and small-window backends don't overflow. Overestimating
     input is safe (it only lowers `max_tokens`).
  
  Verified in-pod against all three real backends (a real `dsh headless` turn
  each): spark (short + long prompt) and socratic now answer cleanly; litellm
  unchanged (200). 99 tests pass (3 new).

## [0.2.4] — 2026-09-29

### Fixed
- **dsh fails on non-litellm providers with `thinking: Value error,
  thinking.budget_tokens is required when thinking.type is 'enabled'`.**
  Not a provider bug in the shim — the shim never reads or writes the
  Anthropic `thinking` field (the only `litellm` references in the package
  are a doc comment and a provider *label* used when seeding `proxy.conf`).
  The cause: dsh 0.2.0-rc.1 sends `thinking: {type: "enabled"}` (no budget)
  for every provider, and the backends disagree — litellm's Anthropic adapter
  accepts it, `ai-test` (socratic) requires `budget_tokens`, and spark's model
  has no reasoning parser and rejects `thinking` entirely. The shim now
  **self-heals** the way it already does for `max_tokens`: on a 400 whose
  error is about the `thinking` field, it strips the field and retries once.
  This makes socratic and spark work while leaving litellm untouched, and it
  is generic (any future backend that rejects thinking) rather than a
  per-provider special case. Verified in-pod against the real spark (400 →
  retry without `thinking` → 200, self-heal logged) and socratic (200).

## [0.2.3] — 2026-09-29

### Fixed
- **Duplicate `message_start` in the Anthropic `/v1/messages` stream → dsh
  `MALFORMED_RESPONSE: duplicate message_start`.** LiteLLM emits two
  `message_start` SSE events (same id) per message. The shim's forward path
  now de-dupes them, dropping a repeated start that shares its id. The
  detector matches the `event: message_start` *line* anywhere in the event —
  not at position 0 — because the stream is `Transfer-Encoding: chunked` and
  the first event arrives with a hex chunk-size line glued to its front.

## [0.2.2] — 2026-09-23

### Fixed
- **Mid-stream cut no longer kills the shim (the dsh "TRANSPORT" bug).** The
  forward handler buffered the entire upstream response with `raw = resp.read()`
  *outside* any try/except. When the upstream (e.g. an ingress severing a long
  SSE stream) cut the connection mid-body, the `IncompleteRead` /
  `ConnectionResetError` propagated out of the handler thread — killing the
  thread and the whole shim — so the dsh client saw a raw connection close and
  surfaced a bare `TRANSPORT` failure. A mid-stream cut now emits a clean
  terminal SSE event (`data: [ERROR] upstream stream interrupted`) and the shim
  stays up to serve the next request.

### Changed
- **SSE responses stream live instead of being buffered.** Stream requests are
  now forwarded chunk-by-chunk with `resp.fp.read1(4096)` (one socket read,
  returns as soon as any byte is available) rather than `resp.read()` (waits for
  the whole body). The dsh UI sees tokens as they generate; a stalling client
  can no longer wedge the upstream socket buffer. Stream responses are framed
  with `Connection: close` so the client gets a definite end-of-stream.
- **Pre-response transport failure retries once, then a clean 502.** A
  connection-level failure before any upstream byte (dead upstream, connect
  error, buffered read that died pre-response) is a safe full-request resend
  (the shim is stateless). After one retry it returns a clean `502` instead of
  a raw close. This composes with the existing context-window / output-cap
  self-heal retry.

Upstream routing is unchanged — this release only makes the shim resilient to a
cut and streams live; it does not alter which backend the requests reach.

## [0.2.1] — 2026-09-22

### Added
- **`dsh-proxy ensure`** — the single in-pod startup orchestrator. It syncs the
  deployment's `DIVEAI_*`/`LITELLM_*` entries into `~/.dsh/proxy.conf`, then
  starts the daemon if it is down or restarts it if the sync changed anything
  (unchanged + already running → no-op). The image's `40-dsh-proxy` boot hook now
  calls this ONE command instead of a hand-rolled bash sync+start, and the
  duplicated copy that lived in the hub values files' `jupyter_server_config`
  has been removed.
- **`dsh-proxy seed-settings`** — seeds `~/.dsh/settings.yaml` on first boot by
  DISCOVERING the model name and its `contextWindow` from the deployment
  provider's `/v1/models` (the same way hermes does) instead of the image
  baking a fixed model name + `contextWindow`. The model is deployment policy
  (`DSH_DEFAULT_MODEL` env; unset → first advertised model); the window is
  omitted rather than fabricated when undiscoverable. Seeded only when the file
  is absent (student edits survive). Also adds `discover_model_ids()` and
  `discover_model_context_window()` to the discovery module (shared
  `_fetch_models()` fetch, no duplicated HTTP).

### Changed
- **First-run default provider is now deployment policy, not a hardcoded value.**
  `ensure`/`sync` read it from the `DSH_PROXY_DEFAULT_PROVIDER` env var
  (unset → nothing forced; the student keeps/picks their own). The old
  `--default-provider litellm` that was baked into the image is gone, so the
  same image can serve clusters whose default dsh provider is not litellm.
  (A `--default-provider` flag remains on both commands for tests/overrides.)

### Tests
- Added coverage for `ensure`: env-driven first-run default, the
  no-hardcoded-default regression (provider stays `None` when the env is unset),
  idempotent re-run (no stop/start), and student-choice preservation.
- Added coverage for `seed-settings`: discovers model name + `contextWindow`
  from the endpoint, honors `DSH_DEFAULT_MODEL`, never clobbers an existing
  `settings.yaml`, skips (no fabricated file) when no model can be determined,
  and a regression asserting the package contains no baked model name or window.

## [0.2.0] — 2026-09-21

### Added
- **Context-window-aware `max_tokens` (fixes the sglang/vLLM "context length" 400).**
  Providers such as sglang/vLLM report `max_model_len` as the *total* context
  window (input + output), not the output cap. A non-trivial prompt plus a large
  `max_tokens` overflowed the window and 400'd with
  *"exceeds the model's maximum context length of N tokens. You requested a
  total of X tokens: I tokens from the input messages and N tokens for
  completion."* This version handles that in two layers:
  - **Proactive clamp** — `proxy.conf` now stores both `token_cap` (output) and
    `context_window` (total); the shim clamps
    `max_tokens = min(token_cap, context_window − est_input − 128)` so a long
    prompt never overflows. `dsh-proxy use --base` auto-discovers **both**
    limits from `/v1/models`.
  - **Self-heal backstop** — if a context-window 400 still occurs, the shim
    parses the server's *exact* input-token count from the error, retries once
    with `context_window − input − 128`, and persists the discovered
    `context_window` so the next request clamps proactively (no further 400).

### Changed
- `resolve_upstream` / `ShimConfig` now carry the context window alongside the
  output cap (4-tuple resolution). The build-time deployment-agnostic gate is
  updated for the 4-tuple.
- Provider discovery reports both `token_cap` and `context_window`
  (`discover_model_limits`); `discover_completion_cap` remains as a
  back-compatible wrapper.

### Fixed
- Requests with a large prompt + a large `max_tokens` no longer fail against
  sglang/vLLM endpoints (the second, context-window 400 that the 0.1.x
  output-cap-only self-heal did not catch).

## [0.1.1] — 2026-09

### Added
- **Per-provider completion-token cap, auto-discovered.** `dsh-proxy use --base`
  probes `/v1/models` to learn the upstream's output limit and stores it as the
  provider's `token_cap`; the shim clamps `max_tokens` to it and self-heals on a
  *completion-cap* 400 (parses the server-reported cap, retries, persists).
- **`dsh-proxy use` restarts the shim** (was stop-only, leaving a dead endpoint).

### Fixed
- `dsh-proxy status` exit code and the `serve` pidfile handling (two
  boot-breaking bugs).
- Default provider is now `litellm` (DiveAI 404s on `/chat/completions` and is
  not a valid default).

## [0.1.0] — 2026-09

### Added
- Initial release: a tiny, **zero-dependency (stdlib-only)** OpenAI-compatible
  proxy that adapts DeepSeek Harness (dsh) to any OpenAI-compatible endpoint
  (Socrates / LiteLLM / any sglang/vLLM endpoint).
- Safe request rewrites: `reasoning_effort` (remap to a supported value or drop)
  and `max_tokens` (clamp).
- `dsh-openai-shim` (serve) and `dsh-proxy` (use/list/show/status/restart/sync)
  command-line entry points.

[0.2.0]: https://github.com/dive4dec/dsh-openai-shim/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/dive4dec/dsh-openai-shim/compare/v0.1.0...v0.1.1
