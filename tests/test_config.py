"""Tests for dsh_openai_shim.config — the file-based provider model.

proxy.conf is the single source of truth for ALL providers (deployment +
custom). The package is deployment-agnostic: NO test may rely on any
hardcoded endpoint — every base comes from proxy.conf (the deployment's
entries get there via sync_deployment, mirroring the in-pod startup).
"""
import json

import pytest

from dsh_openai_shim.config import (
    ProxyConfig,
    config_path,
    ensure_default,
    load_config,
    resolve_upstream,
    save_config,
    sync_deployment,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate each test from ambient deployment env."""
    for var in ("DIVEAI_API_BASE", "DIVEAI_API_KEY", "LITELLM_API_BASE",
                "LITELLM_API_KEY", "DSH_PROXY_DEFAULT_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


# ─────────────────────────────────────────────────────────────
# config_path honors DSH_HOME
# ─────────────────────────────────────────────────────────────

def test_config_path_uses_dsh_home(monkeypatch, tmp_path):
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    assert config_path() == tmp_path / "dsh" / "proxy.conf"


def test_config_path_fallback_home(monkeypatch, tmp_path):
    monkeypatch.delenv("DSH_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_path() == tmp_path / ".dsh" / "proxy.conf"


# ─────────────────────────────────────────────────────────────
# ensure_default: first-run writes empty file, never clobbers
# ─────────────────────────────────────────────────────────────

def test_ensure_default_creates_then_preserves(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    path = config_path()
    assert not path.exists()

    cfg, created = ensure_default()
    assert created is True
    assert cfg.provider is None
    assert cfg.providers == {}
    assert path.exists()
    assert json.loads(path.read_text()) == {"provider": None, "providers": {}}

    # Student picks a provider; a second boot must NOT clobber it.
    save_config(ProxyConfig(provider="diveai",
                            providers={"diveai": {"base_url": "https://d.x/v1",
                                                  "api_key": "k"}}), path)
    cfg2, created2 = ensure_default()
    assert created2 is False
    assert cfg2.provider == "diveai"
    assert cfg2.providers["diveai"]["api_key"] == "k"


# ─────────────────────────────────────────────────────────────
# load_config: missing / malformed / roundtrip / legacy migration
# ─────────────────────────────────────────────────────────────

def test_load_config_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    cfg = load_config()
    assert cfg.provider is None
    assert cfg.providers == {}


def test_load_config_malformed_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ this is not json ]")
    cfg = load_config()
    assert cfg.provider is None
    assert cfg.providers == {}


def test_load_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    cfg = ProxyConfig(
        provider="myprov",
        providers={"myprov": {"base_url": "https://mine.example/v1",
                              "api_key": "sk-abc"},
                   "custom": {"base_url": "https://legacy.example",
                               "api_key": None}},
    )
    save_config(cfg)
    got = load_config()
    assert got.provider == "myprov"
    assert got.providers["myprov"] == {"base_url": "https://mine.example/v1",
                                       "api_key": "sk-abc"}
    assert got.providers["custom"] == {"base_url": "https://legacy.example",
                                       "api_key": None}


def test_load_config_legacy_custom_format_migrated(tmp_path, monkeypatch):
    """Pre-dict format {"provider","custom":{...}} still loads (as 'custom')."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"provider": "custom",
                             "custom": {"base_url": "https://old.example/v1",
                                        "api_key": "sk-legacy"}}))
    cfg = load_config()
    assert cfg.provider == "custom"
    assert cfg.providers == {"custom": {"base_url": "https://old.example/v1",
                                        "api_key": "sk-legacy"}}


def test_load_config_legacy_custom_coexists_with_new_dict(tmp_path, monkeypatch):
    """A file with BOTH forms keeps the providers-dict entry (it wins)."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "provider": "custom",
        "providers": {"custom": {"base_url": "https://new.example",
                                 "api_key": "sk-new"}},
        "custom": {"base_url": "https://old.example", "api_key": "sk-old"},
    }))
    cfg = load_config()
    assert cfg.providers["custom"]["base_url"] == "https://new.example"


def test_load_config_ignores_bad_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "provider": "good",
        "providers": {"good": {"base_url": "https://g.example", "api_key": "k"},
                      "nourl": {"api_key": "k"},          # no base_url
                      "badtype": "not-a-dict"},
    }))
    cfg = load_config()
    assert set(cfg.providers) == {"good"}


# ─────────────────────────────────────────────────────────────
# sync_deployment: hermes-style upsert, never touches student entries
# ─────────────────────────────────────────────────────────────

def _deploy():
    return {"diveai": {"base_url": "https://dive.cs.example/ai/v1",
                       "api_key": "dkey"},
            "litellm": {"base_url": "https://socratic.cs.example/litellm/v1",
                        "api_key": "lkey"}}


def test_sync_adds_deployment_entries_and_sets_first_run_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    cfg, _ = ensure_default()
    changed = sync_deployment(cfg, _deploy(), default_provider="litellm")
    assert changed is True
    assert cfg.provider == "litellm"  # first run only
    assert set(cfg.providers) == {"diveai", "litellm"}


def test_sync_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    cfg, _ = ensure_default()
    assert sync_deployment(cfg, _deploy(), default_provider="litellm") is True
    assert sync_deployment(cfg, _deploy(), default_provider="litellm") is False


def test_sync_never_touches_student_custom_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    cfg = load_config()
    cfg.providers["myprov"] = {"base_url": "https://mine.example/v1",
                               "api_key": "mine"}
    cfg.provider = "myprov"
    changed = sync_deployment(cfg, _deploy(), default_provider="litellm")
    assert changed is True  # deployment entries were added
    assert cfg.provider == "myprov"  # student choice preserved
    assert cfg.providers["myprov"] == {"base_url": "https://mine.example/v1",
                                       "api_key": "mine"}


def test_sync_updates_rotated_key_without_clobbering_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    cfg = ProxyConfig(provider="diveai",
                      providers={"diveai": {"base_url": "https://old.example/v1",
                                            "api_key": "old"}})
    changed = sync_deployment(cfg, {"diveai": {"base_url": "https://new.example/v1",
                                               "api_key": "new"}},
                              default_provider="litellm")
    assert changed is True
    assert cfg.provider == "diveai"  # still their choice
    assert cfg.providers["diveai"] == {"base_url": "https://new.example/v1",
                                       "api_key": "new"}


def test_sync_ignores_entries_without_base(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    cfg = ProxyConfig(providers={"diveai": {"base_url": "https://d.example/v1",
                                            "api_key": "k"}})
    changed = sync_deployment(cfg, {"diveai": {"base_url": None, "api_key": "x"},
                                    "litellm": {"base_url": "https://l.example/v1",
                                                "api_key": "lk"}})
    assert changed is True  # only litellm added
    assert cfg.providers["diveai"]["base_url"] == "https://d.example/v1"  # not wiped
    assert "litellm" in cfg.providers


def test_sync_default_not_applied_when_student_already_chose(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    cfg = ProxyConfig(provider="custom",
                      providers={"custom": {"base_url": "https://c.example",
                                            "api_key": None}})
    changed = sync_deployment(cfg, _deploy(), default_provider="litellm")
    assert cfg.provider == "custom"
    assert changed is True  # entries still added


def test_sync_default_not_applied_when_name_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    cfg = ProxyConfig()
    sync_deployment(cfg, _deploy(), default_provider="nosuch")
    assert cfg.provider is None


# ─────────────────────────────────────────────────────────────
# resolve_upstream: purely from proxy.conf
# ─────────────────────────────────────────────────────────────

def test_resolve_from_file_strips_v1():
    cfg = ProxyConfig(provider="diveai",
                      providers={"diveai": {"base_url": "https://d.example/ai/v1",
                                            "api_key": "k"}})
    base, key, cap, window = resolve_upstream(cfg)
    assert base == "https://d.example/ai"
    assert key == "k"
    assert cap is None  # unset → env default
    assert window is None  # unset → no window clamp


def test_resolve_returns_stored_token_cap():
    cfg = ProxyConfig(provider="spark",
                      providers={"spark": {"base_url": "http://1.2.3.4:30000/v1",
                                            "api_key": "k", "token_cap": 64000}})
    base, key, cap, window = resolve_upstream(cfg)
    assert (base, key, cap, window) == ("http://1.2.3.4:30000", "k", 64000, None)


def test_resolve_returns_stored_context_window():
    cfg = ProxyConfig(provider="spark",
                      providers={"spark": {"base_url": "http://1.2.3.4:30000/v1",
                                            "api_key": "k", "token_cap": 64000,
                                            "context_window": 64000}})
    base, key, cap, window = resolve_upstream(cfg)
    assert (base, key, cap, window) == ("http://1.2.3.4:30000", "k", 64000, 64000)


def test_resolve_no_key_yields_none_passthrough():
    cfg = ProxyConfig(provider="custom",
                      providers={"custom": {"base_url": "https://c.example/v1",
                                            "api_key": None}})
    base, key, cap, window = resolve_upstream(cfg)
    assert base == "https://c.example"
    assert key is None
    assert cap is None
    assert window is None


def test_resolve_unset_provider_uses_unique_entry():
    cfg = ProxyConfig(provider=None,
                      providers={"only": {"base_url": "https://o.example/v1",
                                          "api_key": "k"}})
    base, key, cap, window = resolve_upstream(cfg)
    assert (base, key) == ("https://o.example", "k")


def test_resolve_unset_provider_multiple_fails_with_list():
    cfg = ProxyConfig(provider=None,
                      providers={"a": {"base_url": "https://a.example", "api_key": None},
                                 "b": {"base_url": "https://b.example", "api_key": None}})
    with pytest.raises(ValueError, match=r"b, a|a, b"):
        resolve_upstream(cfg)


def test_resolve_unknown_provider_fails_with_available_list():
    cfg = ProxyConfig(provider="ghost",
                      providers={"real": {"base_url": "https://r.example",
                                          "api_key": "k"}})
    with pytest.raises(ValueError, match="ghost") as ei:
        resolve_upstream(cfg)
    assert "real" in str(ei.value)


def test_resolve_no_providers_fails_actionably():
    with pytest.raises(ValueError, match="dsh-proxy use"):
        resolve_upstream(ProxyConfig())


def test_resolve_ignores_environment_completely(monkeypatch):
    """No env vars exist for providers anymore — even a hostile env is inert."""
    monkeypatch.setenv("DIVEAI_API_BASE", "https://evil.example/v1")
    monkeypatch.setenv("LITELLM_API_BASE", "https://evil2.example/v1")
    cfg = ProxyConfig(provider="custom",
                      providers={"custom": {"base_url": "https://c.example/v1",
                                            "api_key": "k"}})
    base, _key, _cap, _window = resolve_upstream(cfg)
    assert base == "https://c.example"
