"""proxy.conf — per-user provider config for ``dsh-proxy`` / the dsh shim.

``~/.dsh/proxy.conf`` (NFS home) is the SINGLE source of truth for every
provider the student can switch to — the deployment's providers (e.g.
``diveai``, ``litellm``) AND the student's own custom providers — plus which
one is active:

    {
      "provider": "litellm",
      "providers": {
        "diveai":   {"base_url": "https://…/v1",  "api_key": "…"},
        "litellm":  {"base_url": "https://…/v1",  "api_key": "…"},
        "spark":    {"base_url": "http://10.37.1.72:30000/v1",
                      "api_key": "…", "token_cap": 64000}
      }
    }

``token_cap`` is OPTIONAL per provider: the completion-token ceiling the shim
should clamp ``max_tokens`` to for THAT upstream. Different models have
different ceilings (a local sglang/vLLM server may allow 64000 while the
deployment's endpoint allows 100000+), so the cap is per-provider, not global.
It is set automatically when a provider is added (``dsh-proxy use … --base``
probes the endpoint, hermes-style) and can be corrected by the shim's runtime
self-heal if the upstream reports a tighter ceiling; see
:mod:`dsh_openai_shim.discover`. When absent, the shim falls back to the
``DSH_SHIM_TOKEN_CAP`` environment default.

Deployment-agnostic by design: this package knows NO provider names, NO
endpoints, and NO env vars. The deployment pushes its own entries into
``providers`` on every notebook start (the in-pod sync, mirroring how the
hub keeps ``~/.hermes/config.yaml``'s provider entries current) — see
:func:`sync_deployment`. It owns ONLY the entries the deployment itself
passes in; student-added providers are never touched, so a student can keep
any number of custom endpoints alongside the deployment's.

``api_key`` may be ``null``: the shim then forwards the caller's
``Authorization`` header as-is (pass-through), which is what ``dsh`` sends
for its deepseek-official provider.

Legacy format (pre-provider-dict) — ``{"provider": "custom", "custom":
{"base_url": …, "api_key": …}}`` — is migrated to ``providers["custom"]``
on load, so existing students' files keep working unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = [
    "ProxyConfig", "load_config", "save_config", "ensure_default",
    "resolve_upstream", "sync_deployment", "config_path",
    "persist_token_cap", "persist_provider_cap",
]


def config_path() -> Path:
    """Path to proxy.conf: ``$DSH_HOME/proxy.conf`` else ``~/.dsh/proxy.conf``.

    Honors ``DSH_HOME`` (dsh's own home) so the config sits next to the dsh
    profiles; falls back to ``~/.dsh`` when it is not set — dsh's default.
    """
    base = os.environ.get("DSH_HOME")
    p = Path(base) if base else Path.home() / ".dsh"
    return p / "proxy.conf"


@dataclass
class ProxyConfig:
    """A persisted provider selection.

    ``provider`` is the ACTIVE provider name. ``None`` means "unset" — it
    resolves to the single configured provider when there is exactly one,
    otherwise resolution fails with the list of available names.

    ``providers`` maps a provider name to ``{"base_url": str, "api_key":
    str|None, "token_cap": int|None, "context_window": int|None}``. ``base_url``
    may carry a trailing ``/v1`` (it is stripped at resolve time; the shim
    re-appends it when forwarding). ``token_cap`` is an optional completion-
    token ceiling and ``context_window`` the model's TOTAL context length
    (input+output, e.g. sglang's ``max_model_len``); both optional — absent/
    None means "use the env default / no window clamp". See module docstring.
    """

    provider: Optional[str] = None
    providers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"provider": self.provider, "providers": self.providers}


def load_config(path: Optional[Path] = None) -> ProxyConfig:
    """Load proxy.conf. Missing / unreadable / malformed files yield empty.

    Never raises on a bad file — a corrupt config must not prevent the pod
    from starting. Migrates the legacy single-``custom`` format into the
    ``providers`` dict (see module docstring).
    """
    path = path or config_path()
    if not path.exists():
        return ProxyConfig()
    try:
        raw = json.loads(path.read_text())
    except (ValueError, OSError):
        return ProxyConfig()
    if not isinstance(raw, dict):
        return ProxyConfig()

    provider = raw.get("provider")
    if provider is not None and not isinstance(provider, str):
        provider = None
    providers: dict = {}
    prov = raw.get("providers")
    if isinstance(prov, dict):
        for name, entry in prov.items():
            if isinstance(entry, dict) and entry.get("base_url"):
                providers[str(name)] = _entry_dict(
                    str(entry["base_url"]), entry.get("api_key"),
                    _coerce_cap(entry.get("token_cap")),
                    _coerce_cap(entry.get("context_window")),
                )
    # Legacy format: {"custom": {"base_url": …, "api_key": …}}.
    if "custom" not in providers:
        legacy = raw.get("custom")
        if isinstance(legacy, dict) and legacy.get("base_url"):
            providers["custom"] = _entry_dict(
                str(legacy["base_url"]), legacy.get("api_key"),
                _coerce_cap(legacy.get("token_cap")),
                _coerce_cap(legacy.get("context_window")),
            )
    return ProxyConfig(provider=provider, providers=providers)


def _entry_dict(base: str, key, cap, window=None) -> dict:
    """A provider entry in its minimal on-disk shape.

    The ``token_cap`` / ``context_window`` keys are present only when set — so
    an in-memory entry is byte-identical to what :func:`save_config` writes,
    which keeps :func:`sync_deployment`'s change-detection idempotent (no
    spurious ``changed=yes`` from a None-vs-absent key mismatch).
    """
    d = {"base_url": base, "api_key": key}
    if cap is not None:
        d["token_cap"] = cap
    if window is not None:
        d["context_window"] = window
    return d


def _coerce_cap(val) -> Optional[int]:
    """Normalize a stored ``token_cap`` to a positive int or ``None``."""
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)) and val > 0:
        return int(val)
    return None


def save_config(cfg: ProxyConfig, path: Optional[Path] = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop a token_cap of None so the stored file stays minimal and the
    # "absent = use env default" meaning is preserved on disk.
    out: dict = {"provider": cfg.provider, "providers": {}}
    for name, entry in cfg.providers.items():
        clean = {"base_url": entry.get("base_url"), "api_key": entry.get("api_key")}
        cap = _coerce_cap(entry.get("token_cap"))
        if cap is not None:
            clean["token_cap"] = cap
        window = _coerce_cap(entry.get("context_window"))
        if window is not None:
            clean["context_window"] = window
        out["providers"][name] = clean
    path.write_text(json.dumps(out, indent=2) + "\n")
    return path


def persist_token_cap(
    name: str,
    cap: int,
    path: Optional[Path] = None,
) -> bool:
    """Set ``token_cap`` on one provider (runtime self-heal / explicit set).

    Loads the config, updates the cap, saves, and returns True if the value
    actually changed. No-op (False) if the provider is unknown or the cap is
    already stored — so an idempotent self-heal never churns the file.
    """
    coerced: Optional[int] = _coerce_cap(cap)
    if coerced is None:
        return False
    cfg = load_config(path)
    entry = cfg.providers.get(name)
    if not entry or _coerce_cap(entry.get("token_cap")) == coerced:
        return False
    entry["token_cap"] = coerced
    save_config(cfg, path)
    return True


def persist_provider_cap(
    name: str,
    cap: Optional[int] = None,
    context_window: Optional[int] = None,
    path: Optional[Path] = None,
) -> bool:
    """Persist ``token_cap`` and/or ``context_window`` on one provider.

    The runtime self-heal uses this to make a corrected cap / discovered
    context window survive the next restart. Only the non-None arguments are
    written. Returns True if ANY value actually changed, so an idempotent
    self-heal never churns the file. No-op (False) for an unknown provider or
    when nothing changes.
    """
    cap_c = _coerce_cap(cap)
    win_c = _coerce_cap(context_window)
    if cap_c is None and win_c is None:
        return False
    cfg = load_config(path)
    entry = cfg.providers.get(name)
    if not entry:
        return False
    changed = False
    if cap_c is not None and _coerce_cap(entry.get("token_cap")) != cap_c:
        entry["token_cap"] = cap_c
        changed = True
    if win_c is not None and _coerce_cap(entry.get("context_window")) != win_c:
        entry["context_window"] = win_c
        changed = True
    if changed:
        save_config(cfg, path)
    return changed


def ensure_default(path: Optional[Path] = None) -> tuple[ProxyConfig, bool]:
    """Return ``(config, created)``.

    If proxy.conf does not exist yet, write an empty config (no providers,
    no active provider) and return it with ``created=True`` — the deployment
    sync populates the providers shortly after, at notebook start. If the
    file already exists, load and return it untouched with ``created=False``:
    the student's choice is never clobbered.
    """
    path = path or config_path()
    if not path.exists():
        cfg = ProxyConfig()
        save_config(cfg, path)
        return cfg, True
    return load_config(path), False


def sync_deployment(
    cfg: ProxyConfig,
    entries: dict,
    default_provider: Optional[str] = None,
) -> bool:
    """Upsert the deployment's provider entries into ``cfg`` (in place).

    ``entries`` maps provider name -> ``{"base_url": str, "api_key":
    str|None}``. This is the hermes ``_upsert_provider`` pattern: on EVERY
    sync the deployment's entries are refreshed to the current endpoints/
    keys (rotation), while any entries the student added are left alone.

    ``default_provider`` (a name from ``entries``) is written as the active
    provider ONLY while ``cfg.provider`` is unset — first-run behavior; a
    student who has picked a provider is never clobbered.

    Entries without a ``base_url`` are ignored (a half-set deployment var
    must not wipe a working entry). The deployment's ``entries`` carry only
    ``base_url``/``api_key`` — a student's ``token_cap`` on a deployment entry
    (if they ever set one) is preserved across refreshes, never clobbered by a
    key/endpoint rotation. Returns True if anything changed.
    """
    changed = False
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        base = entry.get("base_url")
        if not base:
            continue
        new = {"base_url": str(base), "api_key": entry.get("api_key")}
        existing = cfg.providers.get(name)
        if isinstance(existing, dict) and existing.get("token_cap"):
            new["token_cap"] = existing["token_cap"]
        if isinstance(existing, dict) and existing.get("context_window"):
            new["context_window"] = existing["context_window"]
        if existing != new:
            cfg.providers[str(name)] = new
            changed = True
    if not cfg.provider and default_provider and default_provider in cfg.providers:
        cfg.provider = default_provider
        changed = True
    return changed


def _strip_v1(base: str) -> str:
    """Strip a trailing ``/v1`` (and trailing slashes) so the shim won't double it."""
    b = base.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b


def _available(cfg: ProxyConfig) -> str:
    names = sorted(cfg.providers)
    return ", ".join(names) if names else "(none yet)"


def resolve_upstream(
    cfg: ProxyConfig,
) -> tuple[str, Optional[str], Optional[int], Optional[int]]:
    """Return ``(upstream_base, api_key, token_cap, context_window)`` for ``cfg``.

    ``upstream_base`` has NO trailing ``/v1`` (the shim appends it when
    forwarding). ``token_cap`` is the provider's stored completion-token
    ceiling and ``context_window`` its TOTAL context length (input+output);
    each is ``None`` when unset (the shim then uses its env default / no
    window clamp). No environment is consulted — proxy.conf is authoritative,
    which is what lets a student define any number of custom providers and lets
    the deployment refresh its own entries on every start.

    Raises ``ValueError`` with an actionable message when the active
    provider is unknown or nothing is configured yet.
    """
    name = cfg.provider
    if name is None:
        if len(cfg.providers) == 1:
            name = next(iter(cfg.providers))
        else:
            raise ValueError(
                "no provider selected and no unique provider configured "
                f"(available: {_available(cfg)}). Add one with "
                "`dsh-proxy use <name> --base <url> [--key <key>]` or pick "
                "one with `dsh-proxy use <name>`."
            )
    entry = cfg.providers.get(name)
    if entry is None:
        raise ValueError(
            f"provider {name!r} is not in proxy.conf. Available: "
            f"{_available(cfg)}. Add it with `dsh-proxy use {name} "
            "--base <url> [--key <key>]`."
        )
    if not entry.get("base_url"):
        raise ValueError(
            f"provider {name!r} has no base_url — re-add it with "
            f"`dsh-proxy use {name} --base <url> [--key <key>]`."
        )
    return (
        _strip_v1(entry["base_url"]),
        entry.get("api_key"),
        _coerce_cap(entry.get("token_cap")),
        _coerce_cap(entry.get("context_window")),
    )
