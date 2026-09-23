# dsh-openai-shim Changelog

All notable changes to the `dsh-openai-shim` package are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
