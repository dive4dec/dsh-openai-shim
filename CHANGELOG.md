# dsh-openai-shim Changelog

All notable changes to the `dsh-openai-shim` package are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
