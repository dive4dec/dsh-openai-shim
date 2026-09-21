"""Process/runtime helpers: lazily start the shim sidecar (no init system).

Design
------
The dsh persona (in the Jupyter server) and the ``%%dsh`` magic (in the kernel)
need an OpenAI-compatible base_url that speaks to the endpoint with the right
``reasoning_effort`` / ``max_tokens`` rewrites. Rather than require an
``start-notebook.d`` hook (which ``start.sh`` sources as the *container start
user* — ambiguous between root and the notebook user under different runtimes),
the shim is started **lazily, as the current user**, the first time it's needed.

This mirrors the pattern this image already uses: ``%%hermes`` has no
container-sidecar — its ``AcpConnection.get()`` lazily spawns ``hermes acp`` as
a user-side subprocess singleton. ``ensure_shim()`` is the shim equivalent.

Layering guarantee (the whole point):
  * the shim *process* runs from the conda env (ephemeral image layer);
  * it logs to ``/tmp`` (ephemeral);
  * it never writes to the user home. The dsh *runtime* writes only its own
    ``DSH_HOME`` (NFS home), and only dsh does that — never the shim.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

__all__ = ["DEFAULT_SHIM_PORT", "ensure_shim", "shim_base_url", "is_shim_running",
           "stop_shim", "start_shim_detached", "shim_pidfile"]

DEFAULT_SHIM_PORT = 8090


def shim_pidfile(port: int | None = None) -> str:
    """Pidfile for the shim on ``port`` (ephemeral /tmp — never the user home)."""
    port = int(port or os.environ.get("DSH_SHIM_PORT") or DEFAULT_SHIM_PORT)
    return f"/tmp/dsh-shim-{port}.pid"


def _port_open(port: int, timeout: float = 0.3) -> bool:
    """True if something is accepting TCP connections on 127.0.0.1:port."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def is_shim_running(port: int | None = None) -> bool:
    port = int(port or os.environ.get("DSH_SHIM_PORT") or DEFAULT_SHIM_PORT)
    return _port_open(port)


def shim_base_url(port: int | None = None) -> str:
    """The base_url dsh should point at for a shim listening on ``port``."""
    port = int(port or os.environ.get("DSH_SHIM_PORT") or DEFAULT_SHIM_PORT)
    return f"http://127.0.0.1:{port}/v1"


def stop_shim(port: int | None = None, timeout: float = 5.0) -> bool:
    """Stop a shim started by :func:`ensure_shim` (via its pidfile). Returns True if it was running."""
    port = int(port or os.environ.get("DSH_SHIM_PORT") or DEFAULT_SHIM_PORT)
    pidfile = shim_pidfile(port)
    try:
        with open(pidfile) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_open(port):
            try:
                os.remove(pidfile)
            except OSError:
                pass
            return True
        time.sleep(0.05)
    return False


def start_shim_detached(port: int | None = None, wait_seconds: float = 15.0) -> bool:
    """Start the shim detached from proxy.conf (the same launch the boot hook uses).

    ``dsh-proxy serve`` is the daemon; we spawn it in its own session (setsid)
    so it survives the caller (e.g. ``dsh-proxy use``) exiting, logging to
    ``/tmp/dsh-shim.log`` (ephemeral). Returns True once it is listening.

    This is the missing piece that ``use`` relied on: there is no lazy
    auto-start in the request path (the dsh client makes plain HTTP calls to
    the loopback port), so a shim that is stopped must be started explicitly.
    """
    port = int(port or os.environ.get("DSH_SHIM_PORT") or DEFAULT_SHIM_PORT)
    if _port_open(port):
        return True
    log = open("/tmp/dsh-shim.log", "ab", buffering=0)
    try:
        subprocess.Popen(
            [sys.executable, "-m", "dsh_openai_shim.proxy_cli", "serve"],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=dict(os.environ, DSH_SHIM_PORT=str(port)),
        )
    except Exception:
        return False
    finally:
        log.close()
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.05)
    return _port_open(port)


def ensure_shim(
    port: int | None = None,
    upstream: str | None = None,
    upstream_key: str | None = None,
    wait_seconds: float = 20.0,
) -> str:
    """Ensure the dsh-openai-shim is running; return the dsh ``base_url``.

    Idempotent: if the shim is already listening on ``port``, returns
    immediately without spawning. Otherwise spawns
    ``python -m dsh_openai_shim.cli serve`` as the current user, detached into
    its own process session, logging to ``/tmp/dsh-shim.log`` (ephemeral), and
    waits until it accepts connections. The PID is recorded in
    ``/tmp/dsh-shim-<port>.pid`` for :func:`stop_shim`.

    Configuration is read from the environment (set by the JupyterHub spawner):
      * ``DSH_SHIM_UPSTREAM``       — upstream base URL, NO trailing ``/v1``
        (required; raises ``RuntimeError`` if unset)
      * ``DSH_SHIM_UPSTREAM_KEY``   — the real endpoint key (stays in the shim;
        dsh connects with a dummy key)
      * ``DSH_SHIM_PORT``           — listen port (default 8090)
      * ``DSH_SHIM_EFFORT_MODE`` / ``DSH_SHIM_TOKEN_CAP`` — rewrite behavior

    Returns a base_url like ``http://127.0.0.1:8090/v1``.
    """
    port = int(port or os.environ.get("DSH_SHIM_PORT") or DEFAULT_SHIM_PORT)
    upstream = upstream or os.environ.get("DSH_SHIM_UPSTREAM")
    upstream_key = upstream_key if upstream_key is not None else os.environ.get("DSH_SHIM_UPSTREAM_KEY")
    if not upstream:
        raise RuntimeError(
            "dsh-openai-shim: no upstream configured. Set DSH_SHIM_UPSTREAM "
            "(and DSH_SHIM_UPSTREAM_KEY) in the spawner environment."
        )

    base_url = f"http://127.0.0.1:{port}/v1"
    if _port_open(port):
        return base_url

    log = open("/tmp/dsh-shim.log", "ab", buffering=0)
    env = dict(os.environ)
    env["DSH_SHIM_PORT"] = str(port)
    env["DSH_SHIM_UPSTREAM"] = upstream
    if upstream_key:
        env["DSH_SHIM_UPSTREAM_KEY"] = upstream_key

    proc = subprocess.Popen(
        [sys.executable, "-m", "dsh_openai_shim.cli", "serve"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach: survive the parent's terminal hangup
        cwd="/tmp",
    )
    # Record the PID so stop_shim() can reap it (ephemeral /tmp, not the home).
    try:
        with open(shim_pidfile(port), "w") as f:
            f.write(str(proc.pid))
    except OSError:
        pass
    deadline = time.time() + wait_seconds
    try:
        while time.time() < deadline:
            if _port_open(port):
                return base_url
            if proc.poll() is not None:
                raise RuntimeError(
                    f"dsh-openai-shim failed to start (exit {proc.returncode}); see /tmp/dsh-shim.log"
                )
            time.sleep(0.1)
    finally:
        log.close()
    raise RuntimeError(
        f"dsh-openai-shim did not come up within {wait_seconds:.0f}s; see /tmp/dsh-shim.log"
    )
