"""Tests for dsh_openai_shim.proxy_cli — provider + daemon management.

Covers the file-based contract: `use` with ANY name (add or switch), `list`,
`sync` (deployment upsert from env, changed=yes|no), show/status messaging —
with no network and no real daemon (the runtime helpers are stubbed).
"""
import json

import pytest

from dsh_openai_shim.config import load_config, ProxyConfig, save_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("DIVEAI_API_BASE", "DIVEAI_API_KEY", "LITELLM_API_BASE",
                "LITELLM_API_KEY", "DSH_PROXY_DEFAULT_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_daemon(monkeypatch):
    """CLI tests must never start/stop the real shim daemon (port 8090)."""
    from dsh_openai_shim import proxy_cli

    monkeypatch.setattr(proxy_cli, "is_shim_running", lambda port=None: False)
    monkeypatch.setattr(proxy_cli, "start_shim_detached",
                        lambda port=None, wait_seconds=15.0: True)
    monkeypatch.setattr(proxy_cli, "stop_shim", lambda port=None, timeout=5.0: False)


def main(*argv):
    from dsh_openai_shim.proxy_cli import main as _m
    return _m(list(argv))


# ─────────────────────────────────────────────────────────────
# use: add ANY custom provider + switch
# ─────────────────────────────────────────────────────────────

def test_use_adds_arbitrary_named_provider(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert main("use", "myprovider",
                "--base", "https://socratic.cs.example/litellm/v1",
                "--key", "sk-0iDQ-DG6W65OX8YlxEMySw") == 0
    cfg = load_config()
    assert cfg.provider == "myprovider"
    assert cfg.providers["myprovider"] == {
        "base_url": "https://socratic.cs.example/litellm/v1",
        "api_key": "sk-0iDQ-DG6W65OX8YlxEMySw"}
    out = capsys.readouterr().out
    assert "provider set to myprovider" in out
    assert "sk-0iDQ" not in out  # key masked


def test_use_pass_through_key_when_omitted(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert main("use", "custom", "--base", "https://mine.example/v1") == 0
    cfg = load_config()
    assert cfg.provider == "custom"
    assert cfg.providers["custom"]["api_key"] is None


def test_use_key_without_base_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert main("use", "x", "--key", "sk-x") == 2
    assert "--base" in capsys.readouterr().err


def test_use_switch_to_existing_provider(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    main("use", "a", "--base", "https://a.example/v1", "--key", "ka")
    main("use", "b", "--base", "https://b.example/v1", "--key", "kb")
    assert main("use", "a") == 0  # bare switch, no --base needed
    assert load_config().provider == "a"
    assert "provider set to a" in capsys.readouterr().out


def test_use_unknown_bare_name_rejected_without_clobbering(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    main("use", "a", "--base", "https://a.example/v1", "--key", "ka")
    assert main("use", "nosuch") == 2
    err = capsys.readouterr().err
    assert "nosuch" in err and "--base" in err
    cfg = load_config()
    assert cfg.provider == "a"  # previous choice intact
    assert set(cfg.providers) == {"a"}


def test_use_repoints_deployment_entry_locally(tmp_path, monkeypatch):
    """A student may re-point a deployment-owned entry (hub re-syncs at boot)."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert main("use", "litellm", "--base", "https://own.example/v1",
                "--key", "sk-own") == 0
    cfg = load_config()
    assert cfg.provider == "litellm"
    assert cfg.providers["litellm"]["base_url"] == "https://own.example/v1"


def test_use_overwrites_same_name_preserves_others(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    main("use", "a", "--base", "https://a1.example", "--key", "ka1")
    main("use", "b", "--base", "https://b1.example", "--key", "kb1")
    main("use", "a", "--base", "https://a2.example", "--key", "ka2")
    cfg = load_config()
    assert cfg.providers["a"]["base_url"] == "https://a2.example"
    assert cfg.providers["b"]["base_url"] == "https://b1.example"
    assert cfg.provider == "a"


# ─────────────────────────────────────────────────────────────
# list / show
# ─────────────────────────────────────────────────────────────

def test_list_flags_current_and_masks_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    main("use", "a", "--base", "https://a.example/v1", "--key", "averylongkey12345")
    main("use", "b", "--base", "https://b.example", "--key", "kb")
    main("use", "a")
    assert main("list") == 0
    out = capsys.readouterr().out
    assert "* a" in out or "*  a" in out
    assert "b" in out
    assert "averylongkey12345" not in out  # masked
    assert "(pass-through)" not in out


def test_list_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert main("list") == 0
    assert "no providers configured" in capsys.readouterr().out


def test_show_masks_key_and_shows_upstream(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    main("use", "a", "--base", "https://a.example/v1", "--key", "averylongkey12345")
    assert main("show") == 0
    out = capsys.readouterr().out
    assert "provider : a" in out
    assert "upstream : https://a.example" in out
    assert "averylongkey12345" not in out
    assert "…" in out


def test_show_unset_provider_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert main("show") == 2
    assert "no provider selected" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────
# sync: deployment upsert from the environment (in-pod startup)
# ─────────────────────────────────────────────────────────────

def test_sync_first_run_adds_entries_and_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    monkeypatch.setenv("DIVEAI_API_BASE", "https://dive.cs.example/ai/v1")
    monkeypatch.setenv("DIVEAI_API_KEY", "dkey")
    monkeypatch.setenv("LITELLM_API_BASE", "https://socratic.cs.example/litellm/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "lkey")
    assert main("sync", "--default-provider", "litellm") == 0
    assert "changed=yes" in capsys.readouterr().out
    cfg = load_config()
    assert cfg.provider == "litellm"
    assert set(cfg.providers) == {"diveai", "litellm"}
    assert cfg.providers["diveai"]["api_key"] == "dkey"


def test_sync_second_run_idempotent(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    monkeypatch.setenv("DIVEAI_API_BASE", "https://dive.cs.example/ai/v1")
    monkeypatch.setenv("DIVEAI_API_KEY", "dkey")
    main("sync", "--default-provider", "litellm")
    assert main("sync", "--default-provider", "litellm") == 0
    assert "changed=no" in capsys.readouterr().out


def test_sync_keeps_student_choice_and_custom_entries(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    main("use", "myprov", "--base", "https://mine.example/v1", "--key", "mine")
    monkeypatch.setenv("LITELLM_API_BASE", "https://socratic.cs.example/litellm/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "lkey")
    assert main("sync", "--default-provider", "litellm") == 0
    assert "changed=yes" in capsys.readouterr().out
    cfg = load_config()
    assert cfg.provider == "myprov"  # student's choice survives
    assert "myprov" in cfg.providers
    assert "litellm" in cfg.providers


def test_sync_detects_key_rotation(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    monkeypatch.setenv("LITELLM_API_BASE", "https://socratic.cs.example/litellm/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "oldkey")
    main("sync", "--default-provider", "litellm")
    monkeypatch.setenv("LITELLM_API_KEY", "newkey")
    assert main("sync", "--default-provider", "litellm") == 0
    assert "changed=yes" in capsys.readouterr().out
    assert load_config().providers["litellm"]["api_key"] == "newkey"


def test_sync_no_deployment_vars_is_noop(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert main("sync", "--default-provider", "litellm") == 0
    assert "changed=no" in capsys.readouterr().out
    assert load_config().providers == {}


# ─────────────────────────────────────────────────────────────
# status: exit code is the branchable signal (regression)
#
# The boot hook's "already running?" guard MUST branch on this exit code.
# The text says "not running", which contains "running" — so `grep running`
# is wrong and would suppress the daemon start on a fresh pod.
# ─────────────────────────────────────────────────────────────

def test_status_exit_code_not_running(monkeypatch, capsys):
    from dsh_openai_shim import proxy_cli
    monkeypatch.setattr(proxy_cli, "is_shim_running", lambda port=None: False)
    assert proxy_cli.cmd_status(None) == 1
    assert "not running" in capsys.readouterr().out


def test_status_exit_code_running(monkeypatch, capsys):
    from dsh_openai_shim import proxy_cli
    monkeypatch.setattr(proxy_cli, "is_shim_running", lambda port=None: True)
    assert proxy_cli.cmd_status(None) == 0
    assert "shim : running" in capsys.readouterr().out
