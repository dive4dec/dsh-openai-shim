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
    dsh-proxy status                    # is the shim daemon running?
    dsh-proxy restart                   # restart the shim daemon from proxy.conf
    dsh-proxy serve                     # run the shim from proxy.conf (boot hook)
    dsh-proxy init                      # write an empty config if absent

The selection is persisted to proxy.conf and survives pod restarts. ``use``
writes the file, then RESTARTS a running shim so the new provider takes
effect immediately (there is no lazy auto-start in the request path — the
dsh client makes plain HTTP calls to the loopback port, so merely stopping
the shim would leave the endpoint dead). ``sync`` never restarts anything:
it is called by the in-pod startup script, which restarts the daemon itself
only when ``sync`` reports a change.
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
from .discover import discover_model_limits
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
        "status": cmd_status,
        "restart": cmd_restart,
        "init": cmd_init,
        "serve": cmd_serve,
        "rewrite": cmd_rewrite,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
