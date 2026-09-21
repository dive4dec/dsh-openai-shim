"""``dsh-proxy`` — manage the dsh OpenAI proxy: providers + daemon.

``~/.dsh/proxy.conf`` (NFS home) is the single source of truth for ALL
providers — the deployment's (synced in from the environment on every
notebook start, hermes-style) and the student's own custom ones:

    dsh-proxy list                        # all providers, current one flagged
    dsh-proxy use litellm                 # switch to an existing provider
    dsh-proxy use myprovider --base https://host/v1 --key sk-...
                                          # add ANY custom provider + switch
    dsh-proxy use custom --base https://host/v1           # pass-through key
    dsh-proxy show                      # current provider + upstream (key masked)
    dsh-proxy sync                      # (in-pod) upsert deployment providers
    dsh-proxy ensure                    # (in-pod) sync + start/restart daemon — the
                                        #   single startup orchestrator (boot hook)
    dsh-proxy seed-settings             # (in-pod) seed settings.yaml on first boot —
                                        #   discover model + contextWindow (no hardcode)
    dsh-proxy status                    # is the shim daemon running?
    dsh-proxy restart                   # restart the shim daemon from proxy.conf
    dsh-proxy serve                     # run the shim from proxy.conf (daemon process)
    dsh-proxy init                      # write an empty config if absent

The selection is persisted to proxy.conf and survives pod restarts. ``use``
writes the file, then RESTARTS a running shim so the new provider takes
effect immediately (there is no lazy auto-start in the request path — the
dsh client makes plain HTTP calls to the loopback port, so merely stopping
the shim would leave the endpoint dead).

``ensure`` is the SINGLE in-pod startup orchestrator: it syncs the
deployment's providers from the environment, then starts the daemon if it is
down or restarts it if the sync changed anything. It is the one thing the
image's before-notebook.d boot hook calls — the values files no longer carry
a copy of this logic (that was the duplication that let them drift). The
provider selected on a student's first run is the ``DSH_PROXY_DEFAULT_PROVIDER``
env var (deployment policy) — the package and image hardcode NO default.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import ShimConfig, apply_rewrites, serve
from .config import (
    ProxyConfig,
    config_path,
    ensure_default,
    load_config,
    resolve_upstream,
    save_config,
    sync_deployment,
)
from .discover import (discover_model_context_window, discover_model_ids,
                       discover_model_limits)
from .runtime import is_shim_running, stop_shim, start_shim_detached, shim_base_url

__all__ = ["main", "build_parser"]


def _mask(key):
    if not key:
        return "(pass-through)"
    if len(key) <= 8:
        return "***"
    return key[:4] + "…" + key[-4:]


def _upstream_for(cfg: ProxyConfig):
    """``(base, key, cap, window, note)`` for display / serving; raises on unknown.

    ``cap`` is the provider's stored ``token_cap`` (or ``None`` → env
    default); ``window`` its stored ``context_window`` (or ``None`` → no window
    clamp).
    """
    try:
        base, key, cap, window = resolve_upstream(cfg)
    except ValueError as e:
        raise SystemExit(f"dsh-proxy: {e}")
    note = None
    if not key:
        note = f"note: no api_key stored for {cfg.provider!r}; the caller's " \
               "Authorization header is forwarded as-is"
    return base, key, cap, window, note


def _build_shim_cfg(cfg: ProxyConfig) -> ShimConfig:
    base, key, cap, window, _ = _upstream_for(cfg)
    # Per-provider cap wins; env default (DSH_SHIM_TOKEN_CAP) is the fallback.
    env_cap = int(os.environ.get("DSH_SHIM_TOKEN_CAP", 100000))
    return ShimConfig(
        upstream=base,
        listen_host=os.environ.get("DSH_SHIM_HOST", "127.0.0.1"),
        listen_port=int(os.environ.get("DSH_SHIM_PORT", 8090)),
        effort_mode=os.environ.get("DSH_SHIM_EFFORT_MODE", "map"),
        token_cap=cap if cap is not None else env_cap,
        context_window=window or 0,
        upstream_key=key,
        provider_name=cfg.provider,
    )


def cmd_list(_args) -> int:
    cfg = load_config()
    if not cfg.providers:
        print("no providers configured yet — add one with "
              "`dsh-proxy use <name> --base <url> [--key <key>]`")
        return 0
    for name in sorted(cfg.providers):
        entry = cfg.providers[name]
        cur = "  *" if cfg.provider == name else "   "
        cap = entry.get("token_cap")
        cap_s = f"  cap={cap}" if cap else ""
        print(f"{cur} {name:<16} {entry.get('base_url', '')} "
              f"(key {_mask(entry.get('api_key'))}{cap_s})")
    if not cfg.provider:
        print("\n(no provider selected yet — run `dsh-proxy use <name>`)")
    return 0


def cmd_show(_args) -> int:
    cfg = load_config()
    try:
        base, key, cap, window, note = _upstream_for(cfg)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"provider : {cfg.provider}")
    print(f"upstream : {base}")
    print(f"api key  : {_mask(key)}")
    env_cap = int(os.environ.get("DSH_SHIM_TOKEN_CAP", 100000))
    print(f"token cap: {cap if cap is not None else f'{env_cap} (env default)'}")
    print(f"ctx win  : {window if window is not None else '(no window clamp)'}")
    if note:
        print(note, file=sys.stderr)
    print(f"config   : {config_path()}")
    return 0


def cmd_use(args) -> int:
    name = args.provider
    base, key = args.base, args.key
    if base is None and key is None:
        # Switching to an existing provider.
        if name not in load_config().providers:
            print(f"dsh-proxy: provider {name!r} is not configured; provide "
                  f"`--base <url> [--key <key>]` to add it. "
                  "See `dsh-proxy list`.", file=sys.stderr)
            return 2
    elif base is None:
        print("dsh-proxy: --key without --base is not allowed", file=sys.stderr)
        return 2
    elif key is None:
        # base without key = pass-through of the caller's Authorization.
        key = None

    # Completion cap + context window for the (new) provider. An explicit
    # --token-cap wins for the cap; otherwise, when adding a provider we PROBE
    # the endpoint and discover BOTH the output cap and the total context
    # window automatically (hermes-style) so the student needn't know them.
    # Probe is best-effort: a failed/unreachable endpoint just leaves them
    # unset (env default / no window clamp) and the runtime self-heal learns
    # the window later from the first context-length 400.
    cap = args.token_cap
    window = None
    adding = base is not None
    if adding and not args.no_discover:
        dcap, dwindow = discover_model_limits(base, api_key=key or "")
        if cap is None and dcap is not None:
            cap = dcap
            print(f"dsh-proxy: detected max completion tokens = {dcap} "
                  f"for {name} (override with --token-cap N)", file=sys.stderr)
        if dwindow is not None:
            window = dwindow
            print(f"dsh-proxy: detected context window = {dwindow} tokens "
                  f"for {name}", file=sys.stderr)

    cfg = load_config()
    prev = (cfg.provider, dict(cfg.providers))
    cfg.provider = name
    if base is not None:
        entry = {"base_url": base, "api_key": key}
        if cap is not None:
            entry["token_cap"] = int(cap)
        if window is not None:
            entry["context_window"] = int(window)
        cfg.providers[name] = entry

    # Validate the new selection resolves cleanly BEFORE writing the file —
    # a failed `use` must not clobber the student's currently-working config.
    try:
        rb, rk, rcap, rwin, note = _upstream_for(cfg)
    except SystemExit as e:
        cfg.provider, cfg.providers = prev
        print(str(e), file=sys.stderr)
        return 2

    save_config(cfg)
    print(f"dsh-proxy: provider set to {name} -> {rb} (key {_mask(rk)})")
    if rcap is not None:
        print(f"dsh-proxy: completion-token cap = {rcap}")
    if rwin is not None:
        print(f"dsh-proxy: context window = {rwin}")
    if note:
        print(note, file=sys.stderr)

    # RESTART the shim so it picks up the new provider immediately. (There is
    # no lazy auto-start in the request path — the dsh client just hits the
    # loopback port — so merely *stopping* it left the endpoint dead.)
    if is_shim_running():
        if not stop_shim():
            print("dsh-proxy: shim was running but could not be stopped; "
                  "it will pick up the change on its next start", file=sys.stderr)
            return 0
    if not start_shim_detached():
        print("dsh-proxy: WARNING: could not restart the shim; run "
              "`dsh-proxy restart` (or restart your server) to recover",
              file=sys.stderr)
        return 1
    print(f"dsh-proxy: shim restarted with provider {name}")
    return 0


def _deployment_entries() -> dict:
    """The deployment's provider entries, from the spawner's environment.

    The names are fixed (diveai / litellm) — they are the labels the hub's
    single-source DIVEAI_*/LITELLM_* variables stand for. Endpoints/keys come
    ONLY from the environment, never from this package.
    """
    env = os.environ
    entries = {}
    if env.get("DIVEAI_API_BASE"):
        entries["diveai"] = {"base_url": env["DIVEAI_API_BASE"],
                             "api_key": env.get("DIVEAI_API_KEY")}
    if env.get("LITELLM_API_BASE"):
        entries["litellm"] = {"base_url": env["LITELLM_API_BASE"],
                              "api_key": env.get("LITELLM_API_KEY")}
    return entries


def cmd_sync(args) -> int:
    """Upsert the deployment's providers into proxy.conf (in-pod startup).

    Mirrors the hermes config sync: on every notebook start the deployment's
    entries are refreshed from the environment (key rotation, endpoint moves)
    without touching student-added providers. Prints ``changed=yes|no`` on
    the last line so the caller can decide whether the running shim needs a
    restart.
    """
    cfg, _created = ensure_default()
    entries = _deployment_entries()
    changed = sync_deployment(cfg, entries,
                              default_provider=args.default_provider)
    if changed:
        save_config(cfg)
    if not cfg.provider and cfg.providers:
        print(f"dsh-proxy: no provider selected yet; available: "
              f"{', '.join(sorted(cfg.providers))}", file=sys.stderr)
    print("changed=yes" if changed else "changed=no")
    return 0


def cmd_status(_args) -> int:
    port = int(os.environ.get("DSH_SHIM_PORT", 8090))
    running = is_shim_running(port)
    print(f"shim : {'running' if running else 'not running'} (127.0.0.1:{port})")
    # Exit code mirrors the state (0 = running, 1 = not) so callers can branch
    # on it without parsing the text. NB: the text intentionally says "not
    # running" — a naive `grep running` would match that too; use the exit code.
    return 0 if running else 1


def cmd_init(_args) -> int:
    cfg, created = ensure_default()
    if created:
        print(f"dsh-proxy: wrote empty config to {config_path()}")
    else:
        print(f"dsh-proxy: config already present at {config_path()} "
              f"(provider={cfg.provider}); left unchanged")
    return 0


def cmd_serve(args) -> int:
    """Run the shim daemon using the provider from proxy.conf.

    This IS the daemon (the boot hook launches it with ``nohup setsid``), so it
    records its own PID in the pidfile at startup — otherwise ``stop_shim()``
    (used by ``dsh-proxy use`` to restart with a new provider) can't find it.
    The pidfile is removed on clean shutdown.
    """
    # First-run file; the deployment sync (in-pod startup) populates providers.
    cfg, _created = ensure_default()
    scfg = _build_shim_cfg(cfg)

    # Record our own PID so stop_shim() can reap us (ephemeral /tmp, not home).
    pidfile = os.path.join("/tmp", f"dsh-shim-{scfg.listen_port}.pid")
    try:
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass  # non-fatal: stop_shim falls back to "not running"

    try:
        serve(scfg)
    finally:
        try:
            os.remove(pidfile)
        except OSError:
            pass
    return 0


def cmd_rewrite(args) -> int:
    # Kept for parity with the dsh-openai-shim entry point (one-shot rewrites).
    cfg = ShimConfig(upstream="http://upstream", effort_mode=args.effort_mode,
                     token_cap=args.token_cap)
    body = sys.stdin.buffer.read()
    new, changes = apply_rewrites(body, cfg)
    for c in changes:
        print(f"rewrite: {c}", file=sys.stderr)
    sys.stdout.buffer.write(new)
    sys.stdout.buffer.flush()
    return 0


def cmd_restart(_args) -> int:
    """Restart the shim from proxy.conf (manual recovery after a failed stop)."""
    cfg, _ = ensure_default()
    if is_shim_running():
        stop_shim()
    if start_shim_detached():
        base, key, _cap, _win, _note = _upstream_for(cfg)
        print(f"dsh-proxy: shim running on {shim_base_url()} -> {base} (key {_mask(key)})")
        return 0
    print("dsh-proxy: could not start the shim; check /tmp/dsh-shim.log", file=sys.stderr)
    return 1


def _default_provider_from_env() -> str | None:
    """Deployment-policy default provider, from the pod env — or None.

    The image and the shim package hardcode NO provider. Which provider a
    student gets on FIRST run is deployment policy: whoever deploys sets
    ``DSH_PROXY_DEFAULT_PROVIDER`` in the pod env (a value that exists in
    proxy.conf, e.g. ``litellm``). Unset → no default is forced; the student
    picks (or the daemon stays on whatever proxy.conf already holds).
    """
    return os.environ.get("DSH_PROXY_DEFAULT_PROVIDER") or None


def _default_model_from_env() -> str | None:
    """Deployment-policy default dsh model, from the pod env — or None.

    The image hardcodes NO model. Which model the seeded ``settings.yaml`` uses
    is deployment policy (it belongs in the hub values file, like hermes'
    agent-default-model), set as ``DSH_DEFAULT_MODEL``. Unset → we fall back to
    the first model the endpoint advertises (discovered), never a baked name.
    """
    return os.environ.get("DSH_DEFAULT_MODEL") or None


def _reasoning_effort_from_env() -> str:
    """Per-model reasoning effort for the seeded default (deployment policy).

    A display/preference value, not a model property, so it's a dsh default
    (``high``) overridable via ``DSH_REASONING_EFFORT`` — not baked logic.
    """
    return os.environ.get("DSH_REASONING_EFFORT") or "high"


def _yaml_scalar(value: str) -> str:
    """Emit a single-quoted YAML scalar, bare when it needs no quoting.

    The shim is stdlib-only (no PyYAML), so we hand-emit the seed file. Bare
    when the value is a safe token; otherwise single-quoted with ``''`` escapes.
    """
    import re
    if re.fullmatch(r"[A-Za-z0-9._/\-]+", value):
        return value
    return "'" + value.replace("'", "''") + "'"


def cmd_seed_settings(args) -> int:
    """Seed ``~/.dsh/settings.yaml`` on first boot — discover, don't hardcode.

    The old boot hook baked a specific default model name AND its
    ``contextWindow`` into the image. Both are discoverable the same way
    hermes discovers them: query the endpoint's ``/v1/models``. This is the
    single place the seed logic lives; the boot hook just calls it.

    Contract (mirrors hermes + the proxy.conf block — seed once, never clobber):
      * Write ONLY when ``settings.yaml`` is absent. A student who later edits
        it (switch model, add to the catalog) keeps their choice forever.
      * Model name: ``DSH_DEFAULT_MODEL`` env (deployment policy) if set, else
        the first model id the endpoint advertises. The image bakes no name.
      * ``contextWindow``: discovered from the endpoint (``max_model_len`` /
        ``context_length`` / …). Omitted when it can't be discovered, so we
        never write a fabricated number — dsh/shim self-heal learns it later.
      * ``provider`` is the dsh→shim routing constant (``deepseek-official``,
        pointed at ``DEEPSEEK_BASE_URL`` by the spawner) — a dsh protocol name,
        not a deployment-specific upstream.

    Must run AFTER ``ensure`` so proxy.conf holds the deployment provider's
    base+key to query. Best-effort: a failure here never breaks notebook boot.
    """
    settings_path = os.path.join(
        os.environ.get("DSH_HOME") or os.path.expanduser("~/.dsh"), "settings.yaml")
    if os.path.exists(settings_path):
        print(f"dsh-proxy: settings seed skipped (already present: {settings_path})")
        return 0

    cfg = load_config()
    provider = args.provider or cfg.provider
    if not provider:
        print("dsh-proxy: settings seed skipped (no provider selected; set "
              "DSH_PROXY_DEFAULT_PROVIDER or `dsh-proxy use <name>`)", file=sys.stderr)
        return 0

    entry = cfg.providers.get(provider) or {}
    base = entry.get("base_url") or ""
    key = entry.get("api_key") or ""
    if not base:
        print(f"dsh-proxy: settings seed skipped (provider {provider!r} has no "
              "base_url to discover from)", file=sys.stderr)
        return 0

    # Discover model ids + the context window from the endpoint (best-effort).
    model_ids = discover_model_ids(base, api_key=key)
    _cap, window = discover_model_limits(base, api_key=key)

    model = _default_model_from_env()
    if model:
        if model_ids and model not in model_ids:
            print(f"dsh-proxy: note: DSH_DEFAULT_MODEL={model!r} not in the "
                  f"endpoint's advertised models {model_ids}; using it anyway "
                  "(deployment policy).", file=sys.stderr)
    else:
        model = model_ids[0] if model_ids else None
    if not model:
        print("dsh-proxy: settings seed skipped (no model name could be "
              "determined; set DSH_DEFAULT_MODEL or expose /v1/models)",
              file=sys.stderr)
        return 0

    # The window that belongs in THIS model's catalog entry: the endpoint's
    # per-model max_model_len (not the min-across-all used for clamping). Fall
    # back to the endpoint-level bound when the model reports none.
    per_model_window = discover_model_context_window(base, model, api_key=key)
    window = per_model_window if per_model_window is not None else window

    # Hand-emit settings.yaml (stdlib-only; dsh reads this on every start).
    lines = [
        "# dsh default model — seeded by `dsh-proxy seed-settings` on first boot;",
        "# edit freely (persists on NFS). Both the in-editor @dsh chat and `dsh`",
        "# run in a terminal read this file.",
        "#",
        "# agent-default-model = the model used when nothing else is chosen.",
        "#   provider \"deepseek-official\" is routed at the in-pod shim",
        "#   (DEEPSEEK_BASE_URL) regardless of the name.",
        "#   To use a different model, change `model:` below, or pick one in chat.",
        "#",
        "# llm-deepseek.models = the model picker's list (discovered from the",
        "#   endpoint at seed time; add more here as the provider exposes them).",
        "agent-default-model:",
        "  provider: deepseek-official",
        f"  model: {_yaml_scalar(model)}",
        f"  reasoningEffort: {_yaml_scalar(_reasoning_effort_from_env())}",
        "llm-deepseek:",
        "  models:",
        f"    - id: {_yaml_scalar(model)}",
        f"      name: {_yaml_scalar(model)}",
    ]
    if window is not None:
        lines.append(f"      contextWindow: {int(window)}")
    lines.append("      inputModalities: [text]")
    body = "\n".join(lines) + "\n"

    try:
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, "w") as f:
            f.write(body)
        try:
            os.chmod(settings_path, 0o600)
        except OSError:
            pass
    except OSError as e:
        print(f"dsh-proxy: could not write settings seed to {settings_path}: {e}",
              file=sys.stderr)
        return 1

    win_note = f", contextWindow={window} (discovered)" if window is not None \
        else ", contextWindow not discovered (omitted; self-heal will learn it)"
    src = "DSH_DEFAULT_MODEL" if _default_model_from_env() else "first advertised model"
    print(f"dsh-proxy: seeded dsh settings.yaml (model={model} via {src}{win_note}) "
          f"at {settings_path}")
    return 0


def cmd_ensure(args) -> int:
    """Single in-pod startup orchestrator: sync deployment providers, then
    start the daemon if down, or restart it if the sync changed anything.

    This replaces the two duplicated copies that existed — the image's
    40-dsh-proxy boot hook doing a manual sync+start in bash, and the
    values-file jupyter_server_config Python block doing a second
    sync+restart at server-config time. Both are gone; the boot hook now
    calls ``dsh-proxy ensure`` and that is the only place the logic lives.

    The first-run default provider comes from the ``DSH_PROXY_DEFAULT_PROVIDER``
    env (deployment policy); the flag exists for tests/overrides but is unset
    in normal operation, so the image hardcodes no provider.
    """
    cfg, _created = ensure_default()
    entries = _deployment_entries()
    default_provider = args.default_provider or _default_provider_from_env()
    changed = sync_deployment(cfg, entries, default_provider=default_provider)
    if changed:
        save_config(cfg)

    if not cfg.provider and cfg.providers:
        print(f"dsh-proxy: no provider selected yet; available: "
              f"{', '.join(sorted(cfg.providers))}", file=sys.stderr)

    # Start-or-restart: restart (stop+start) only when the sync changed
    # something OR the daemon is not running; otherwise leave a healthy
    # running daemon untouched (reSTARTABLE re-runs are a no-op).
    if not changed and is_shim_running():
        print("dsh-proxy: ensure ok (no change; shim already running)")
        return 0

    if is_shim_running():
        stop_shim()
    if start_shim_detached():
        base, key, _cap, _win, _note = _upstream_for(cfg)
        print(f"dsh-proxy: ensure ok (sync changed={changed}) — shim running "
              f"on {shim_base_url()} -> {base} (key {_mask(key)})")
        return 0
    print("dsh-proxy: ensure: sync ok but could not start the shim; "
          "check /tmp/dsh-shim.log", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dsh-proxy",
                                description="Manage the dsh OpenAI proxy (providers + daemon).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all configured providers (current flagged)")
    sub.add_parser("show", help="show the current provider/upstream (key masked)")

    u = sub.add_parser(
        "use",
        help="switch to a provider, or add one with --base/--key",
    )
    u.add_argument("provider", help="provider name (any name; add a new one "
                                    "with --base)")
    u.add_argument("--base", help="base URL for a new provider (trailing /v1 ok)")
    u.add_argument("--key", help="API key (omit to pass the caller's Authorization through)")
    u.add_argument("--token-cap", type=int, default=None, metavar="N",
                   help="completion-token ceiling for this provider. Omit to "
                        "auto-detect from the endpoint at add time; a later "
                        "mismatch is corrected automatically by the shim.")
    u.add_argument("--no-discover", action="store_true",
                   help="skip auto-detection of the token cap when adding a "
                        "provider (use the env default)")

    s = sub.add_parser("sync",
                       help="(in-pod) upsert the deployment's providers from the environment")
    s.add_argument("--default-provider", default=None,
                   help="provider to select on first run only (never clobbers)")

    e = sub.add_parser("ensure",
                       help="(in-pod) sync deployment providers + start/restart "
                            "the shim — the single startup orchestrator (boot hook)")
    e.add_argument("--default-provider", default=None,
                   help="override the DSH_PROXY_DEFAULT_PROVIDER env (first-run only)")

    ss = sub.add_parser("seed-settings",
                        help="(in-pod) seed ~/.dsh/settings.yaml on first boot — "
                             "discover model name + contextWindow from the endpoint")
    ss.add_argument("--provider", default=None,
                    help="provider to discover from (default: the selected one)")

    sub.add_parser("status", help="is the shim daemon running?")
    sub.add_parser("restart", help="restart the shim daemon from proxy.conf")
    sub.add_parser("init", help="write an empty config if absent (never clobbers)")

    sub.add_parser("serve", help="run the shim daemon from proxy.conf (used by the boot hook)")

    r = sub.add_parser("rewrite", help="apply rewrites to a JSON body (stdin -> stdout)")
    r.add_argument("--effort-mode", choices=["map", "drop", "off"], default="map")
    r.add_argument("--token-cap", type=int, default=100000)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return {
        "list": cmd_list,
        "show": cmd_show,
        "use": cmd_use,
        "sync": cmd_sync,
        "ensure": cmd_ensure,
        "seed-settings": cmd_seed_settings,
        "status": cmd_status,
        "restart": cmd_restart,
        "init": cmd_init,
        "serve": cmd_serve,
        "rewrite": cmd_rewrite,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
