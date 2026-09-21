# dsh-openai-shim

A **zero-dependency, stdlib-only** OpenAI-compatible proxy that adapts
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) to
**any** OpenAI-compatible endpoint — Socrates, LiteLLM, sglang, vLLM, etc.

## Why

`dsh`'s bundled `deepseek-official` provider is written for the DeepSeek API and
does two things that break against general OpenAI endpoints like Socrates:

1. It **hard-injects `reasoning_effort: "high"`** into every request. Socrates
   (sglang) rejects `"high"` — it only accepts `xhigh` / `medium` / `low`.
   → `HTTP 400: Unexpected reasoning effort high.`
2. It requests a **default `max_tokens` of 256000**, which — combined with a
   large system prompt — can exceed the model's context window.
   → `HTTP 400: Requested token count exceeds the model's maximum context length.`

Both are *request-parameter* mismatches, not protocol mismatches. `dsh` is not
DeepSeek-locked; it connects, authenticates, routes, and runs its full agent
tool loop against any OpenAI endpoint. The only friction is these two params.

Rather than fight dsh's `cordis.patch.yml` / model-catalog (dev-preview,
compiled into a native binary, version-churny), this shim is the **single
compatibility chokepoint**: a tiny proxy dsh points at, which forwards to your
endpoint and rewrites the two params. It is:

- **agent-agnostic** — works for every dsh profile (`sdk`, `sdk-minimal`,
  `acp`, `web`) and any OpenAI client.
- **version-immune** — if dsh changes its defaults, the shim still fixes them.
- **transparent** — passes through auth headers, streaming, `/models`,
  tool-calling, everything else unchanged.

## What it rewrites

| Param | Default behavior | Options |
|-------|------------------|---------|
| `reasoning_effort` | `high → medium` (map to a supported value) | `--effort-mode map` (default) / `drop` (remove it) / `off` |
| `max_tokens` | clamp to the provider's `token_cap` (env default `DSH_SHIM_TOKEN_CAP`, `100000`) | per-provider cap (auto-discovered); `0` disables clamping |

The `max_tokens` cap is **per-provider**, not global: different models have
different completion ceilings (a local sglang/vLLM server may allow 64000, a
big hosted endpoint 100000+). The cap is discovered automatically when a
provider is added, corrected automatically if the upstream rejects it, and can
always be overridden with `--token-cap` — see [Per-user providers](#per-user-providers-dsh-proxy).

## Install

```sh
pip install dsh-openai-shim
# or from source
pip install -e .
```

No third-party runtime dependencies — pure Python stdlib.

## Use

Run the shim, then point dsh at it:

```sh
# 1. start the shim (maps high→medium, caps max_tokens at 100000)
dsh-openai-shim serve \
    --upstream https://socratic.cs.cityu.edu.hk/ai-test \
    --port 8090

# 2. point dsh at the shim (not the upstream directly)
DSH_HOME=/tmp/dsh python - <<'PY'
from deepseek_harness import DeepSeekHarness
with DeepSeekHarness(
    dsh_home="/tmp/dsh",
    cwd="/workspace",
    provider="deepseek-official",
    model="Socrates",                 # your endpoint's model id
    base_url="http://127.0.0.1:8090/v1",   # NOTE: points at the shim
    api_key="...",
    max_tokens=100000,
) as h:
    print(h.run("Say hi.", session_id="s").final_response)
PY
```

> **Path note:** if the client appends `/v1/...` to `base_url`, set
> `--upstream` to the path **without** a trailing `/v1`.

### Useful flags

```
--effort-mode {map,drop,off}   how to handle reasoning_effort (default: map)
--token-cap N                  clamp completion max_tokens to N (default: 100000, 0 = off)
--upstream-key Bearer          force a specific Authorization key upstream
--host / --port                listen address (default 127.0.0.1:8090)
```

### One-shot rewrite (no server) — for tests / debugging

```sh
echo '{"reasoning_effort":"high","max_tokens":256000}' | \
    dsh-openai-shim rewrite --effort-mode map
# {"reasoning_effort": "medium", "max_tokens": 100000}
```

## Per-user providers (`dsh-proxy`)

Beyond the one-shot `dsh-openai-shim serve --upstream …`, the package ships a
**per-user provider manager** (`dsh-proxy`) for deployments that run one pod
per student (e.g. JupyterHub). The student's choice of upstream is persisted
in `~/.dsh/proxy.conf` (NFS home) so it survives pod restarts — the same
file-based model as Hermes' `~/.hermes/config.yaml`:

```json
{
  "provider": "litellm",
  "providers": {
    "diveai":   {"base_url": "https://…/ai/v1",   "api_key": "…"},
    "litellm":  {"base_url": "https://…/litellm/v1", "api_key": "…"},
    "myprov":   {"base_url": "http://10.0.0.5:30000/v1", "api_key": "…",
                 "token_cap": 64000}
  }
}
```

- **The deployment's providers** (e.g. `diveai`, `litellm`) are upserted by
  `dsh-proxy sync` on every notebook start — it reads the endpoint + key from
  the pod environment (`DIVEAI_API_BASE/_KEY`, `LITELLM_API_BASE/_KEY`) and
  refreshes those entries (key rotation, endpoint moves), without touching
  student-added providers. The first-run default provider is set once and
  never clobbered afterwards. A student-set `token_cap` survives a re-sync.
- **Custom providers** are the student's own: `dsh-proxy use <name> --base
  <url> [--key <key>]` adds ANY name (as many as you like) and switches to
  it. Omit `--key` to forward the caller's `Authorization` header as-is.
- **Completion-token cap is auto-discovered** (the student never has to know
  the model's limit). On `use <name> --base <url>`, the shim probes
  `GET /v1/models` on the new endpoint and reads the model's completion ceiling
  (`max_completion_tokens` / `max_model_len`) into `token_cap`. This is the same
  technique Hermes uses when you add a custom endpoint and skip the context
  length. If the probe can't reach the endpoint (or it doesn't advertise a
  limit), the cap is left at the env default — and the shim **self-heals at
  request time**: if the upstream rejects a request with an "N too large … at
  most M tokens" 400, the shim parses `M`, lowers the cap, retries the request
  once, and persists `M` to proxy.conf. So a wrong/missing cap corrects itself
  on the first failed turn. Override either step with `--token-cap N`
  (or `--no-discover` to skip the probe entirely).
- **Resolution is file-based and env-inert** — the daemon reads only
  proxy.conf. A trailing `/v1` on a base is stripped automatically (the shim
  re-appends it when forwarding).

```sh
dsh-proxy list                      # all providers, current flagged (*), caps shown
dsh-proxy use litellm               # switch to an existing provider
dsh-proxy use myprov --base https://host/v1 --key sk-...   # add + switch (cap auto-detected)
dsh-proxy use spark --base http://10.0.0.5:30000/v1 --key sk-...   # cap auto-detected
dsh-proxy use myprov --base https://host/v1 --key sk-... --token-cap 40000  # force a cap
dsh-proxy show                      # current provider + upstream + effective cap (key masked)
dsh-proxy sync                      # (in-pod) refresh deployment providers
dsh-proxy status                    # is the shim daemon running?
dsh-proxy restart                   # restart the shim daemon from proxy.conf
```

`use` writes the file and then **restarts** the running shim so the new
provider takes effect immediately (there is no lazy auto-start in the request
path — dsh makes plain HTTP calls to the loopback port). `sync` never
restarts anything; the caller (the pod startup hook) restarts the daemon
only when `sync` prints `changed=yes`.

The package itself is **deployment-agnostic**: no provider catalog, no
endpoints, no env-var names beyond the two raw `DIVEAI_*` / `LITELLM_*` pair
the sync reads. A build-time gate (`dsh_shim_deployment_agnostic_gate.py` in
the Jupyter image) fails the build if a deployment host ever leaks into the
installed package or if resolution starts reading the environment.

## Library use

```python
from dsh_openai_shim import ShimConfig, apply_rewrites, make_handler, serve

cfg = ShimConfig(upstream="http://127.0.0.1:9999/v1",
                 effort_mode="map", token_cap=100000)
new_body, changes = apply_rewrites(body, cfg)   # pure, testable
serve(cfg)                                       # run the server
```

## Security

- By default the shim **forwards the client's `Authorization` header** to the
  upstream. Use `--upstream-key` to replace it with a server-side key so the
  client never holds the real endpoint credential.
- Bind to `127.0.0.1` (default). Do not expose the shim to untrusted networks
  without auth.

## Tests

```sh
python -m pytest tests/ -q
```

The end-to-end tests run a local fake OpenAI upstream (no external network, no
dsh required) and assert the shim rewrites `reasoning_effort` and `max_tokens`
while passing through auth and the response.

## Status

v0.1.0 — spike-proven against Socrates (dsh → shim → Socrates → `pong`).
Zero deps, MIT.
