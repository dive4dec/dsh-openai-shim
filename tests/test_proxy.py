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
                "LITELLM_API_KEY", "DSH_PROXY_DEFAULT_PROVIDER",
                "DSH_DEFAULT_MODEL", "DSH_REASONING_EFFORT"):
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
# ensure: the single in-pod startup orchestrator (sync + start/restart)
#
# This is the ONLY place the sync+daemon logic lives — the image's 40-dsh-proxy
# boot hook calls `dsh-proxy ensure`, and the values files no longer carry a
# copy. The first-run default provider comes from the DSH_PROXY_DEFAULT_PROVIDER
# env (deployment policy); the package hardcodes NO provider.
# ─────────────────────────────────────────────────────────────

def _deployment_env(monkeypatch):
    monkeypatch.setenv("DIVEAI_API_BASE", "https://dive.cs.example/ai/v1")
    monkeypatch.setenv("DIVEAI_API_KEY", "dkey")
    monkeypatch.setenv("LITELLM_API_BASE", "https://socratic.cs.example/litellm/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "lkey")


def test_ensure_starts_daemon_and_uses_env_default(tmp_path, monkeypatch, capsys):
    """First run: sync entries + apply the DSH_PROXY_DEFAULT_PROVIDER env + start."""
    from dsh_openai_shim import proxy_cli
    cap = {"start": 0, "stop": 0, "running": False}
    monkeypatch.setattr(proxy_cli, "is_shim_running", lambda port=None: cap["running"])
    monkeypatch.setattr(proxy_cli, "start_shim_detached",
                        lambda port=None, wait_seconds=15.0: cap.update(start=cap["start"] + 1) or True)
    monkeypatch.setattr(proxy_cli, "stop_shim", lambda port=None, timeout=5.0: cap.update(stop=cap["stop"] + 1) or True)
    monkeypatch.setattr(proxy_cli, "_upstream_for",
                        lambda cfg: ("http://u", "k", None, None, ""))

    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    _deployment_env(monkeypatch)
    monkeypatch.setenv("DSH_PROXY_DEFAULT_PROVIDER", "litellm")

    assert main("ensure") == 0
    assert "ensure ok" in capsys.readouterr().out
    assert cap["start"] == 1 and cap["stop"] == 0  # started, no prior daemon to stop
    cfg = load_config()
    assert cfg.provider == "litellm"  # from the ENV, not any hardcoded value
    assert set(cfg.providers) == {"diveai", "litellm"}


def test_ensure_no_hardcoded_default(tmp_path, monkeypatch, capsys):
    """Regression: with DSH_PROXY_DEFAULT_PROVIDER unset, ensure must force
    NO provider. A baked-in default (e.g. the old `--default-provider litellm`)
    would select one here and fail this test."""
    from dsh_openai_shim import proxy_cli
    monkeypatch.setattr(proxy_cli, "is_shim_running", lambda port=None: False)
    monkeypatch.setattr(proxy_cli, "start_shim_detached",
                        lambda port=None, wait_seconds=15.0: True)
    monkeypatch.setattr(proxy_cli, "stop_shim", lambda port=None, timeout=5.0: True)
    monkeypatch.setattr(proxy_cli, "_upstream_for",
                        lambda cfg: ("http://u", "k", None, None, ""))

    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    _deployment_env(monkeypatch)
    # DSH_PROXY_DEFAULT_PROVIDER is already unset (autouse _clean_env)

    assert main("ensure") == 0
    err = capsys.readouterr().err
    assert load_config().provider is None  # NOTHING forced — deployment-agnostic
    assert "no provider selected yet" in err


def test_ensure_idempotent_leaves_running_daemon(tmp_path, monkeypatch):
    """After the daemon is up, an unchanged re-run is a no-op (no stop, no start)."""
    from dsh_openai_shim import proxy_cli
    cap = {"start": 0, "stop": 0, "running": False}
    monkeypatch.setattr(proxy_cli, "is_shim_running", lambda port=None: cap["running"])
    monkeypatch.setattr(proxy_cli, "start_shim_detached",
                        lambda port=None, wait_seconds=15.0:
                        cap.update(start=cap["start"] + 1, running=True) or True)
    monkeypatch.setattr(proxy_cli, "stop_shim",
                        lambda port=None, timeout=5.0:
                        cap.update(stop=cap["stop"] + 1, running=False) or True)
    monkeypatch.setattr(proxy_cli, "_upstream_for",
                        lambda cfg: ("http://u", "k", None, None, ""))

    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    _deployment_env(monkeypatch)
    main("ensure", "--default-provider", "litellm")  # run 1: changed -> start
    assert cap["start"] == 1 and cap["running"] is True
    main("ensure", "--default-provider", "litellm")  # run 2: unchanged + running -> no-op
    assert cap["start"] == 1 and cap["stop"] == 0  # left running, untouched


def test_ensure_preserves_student_choice(tmp_path, monkeypatch):
    """A student's selected provider survives ensure even with an env default."""
    from dsh_openai_shim import proxy_cli
    monkeypatch.setattr(proxy_cli, "is_shim_running", lambda port=None: False)
    monkeypatch.setattr(proxy_cli, "start_shim_detached",
                        lambda port=None, wait_seconds=15.0: True)
    monkeypatch.setattr(proxy_cli, "stop_shim", lambda port=None, timeout=5.0: True)
    monkeypatch.setattr(proxy_cli, "_upstream_for",
                        lambda cfg: ("http://u", "k", None, None, ""))

    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    save_config(ProxyConfig(provider="myprov",
                            providers={"myprov": {"base_url": "https://mine.example/v1",
                                                  "api_key": "mk"}}))
    _deployment_env(monkeypatch)
    monkeypatch.setenv("DSH_PROXY_DEFAULT_PROVIDER", "litellm")
    assert main("ensure") == 0
    cfg = load_config()
    assert cfg.provider == "myprov"  # never clobbered
    assert "litellm" in cfg.providers  # deployment entry still synced in


# ─────────────────────────────────────────────────────────────
# seed-settings: first-boot dsh settings.yaml — DISCOVER, don't hardcode
#
# The old boot hook baked model name ("Socrates") AND contextWindow (262144)
# into the image. Now `dsh-proxy seed-settings` discovers both from the
# endpoint's /v1/models (the same way hermes does). The deployment policy
# model is DSH_DEFAULT_MODEL (env); unset → first advertised model. No model
# name or window is baked into the package — this section is the regression
# guard for that.
# ─────────────────────────────────────────────────────────────

def _seed_models_server():
    """Local upstream serving GET /v1/models; returns (up, base_url, port)."""
    from test_discover import ModelsUpstream, _free_port
    port = _free_port()
    payload = {"object": "list", "data": [
        {"id": "Socrates", "object": "model", "owned_by": "litellm",
         "max_model_len": 262144},
        {"id": "OtherModel", "object": "model", "max_model_len": 64000},
    ]}
    up = ModelsUpstream(port, payload)
    return up, f"http://127.0.0.1:{port}/v1", port


def test_seed_discovers_model_and_context_window(tmp_path, monkeypatch, capsys):
    """No DSH_DEFAULT_MODEL: first advertised model + discovered contextWindow."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    up, base, _port = _seed_models_server()
    try:
        # add the deployment provider pointing at the local models server
        assert main("use", "litellm", "--base", base, "--key", "lkey") == 0
        assert main("seed-settings") == 0
        p = tmp_path / "settings.yaml"
        assert p.exists()
        text = p.read_text()
        # discovered the first advertised model (Socrates is listed first)
        assert "model: Socrates" in text
        # discovered contextWindow from max_model_len (NOT a baked constant)
        assert "contextWindow: 262144" in text
        # provider routing constant is present (dsh→shim protocol name)
        assert "provider: deepseek-official" in text
        assert "reasoningEffort: high" in text
    finally:
        up.stop()


def test_seed_uses_dsh_default_model_env(tmp_path, monkeypatch, capsys):
    """DSH_DEFAULT_MODEL (deployment policy) wins over the first advertised."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    up, base, _port = _seed_models_server()
    monkeypatch.setenv("DSH_DEFAULT_MODEL", "OtherModel")
    try:
        assert main("use", "litellm", "--base", base, "--key", "lkey") == 0
        assert main("seed-settings") == 0
        text = (tmp_path / "settings.yaml").read_text()
        assert "model: OtherModel" in text
        # still discovered the window for that model
        assert "contextWindow: 64000" in text
    finally:
        up.stop()


def test_seed_does_not_clobber_existing_settings(tmp_path, monkeypatch, capsys):
    """A student's edited settings.yaml is never overwritten by the seed."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    p = tmp_path / "settings.yaml"
    p.write_text("agent-default-model:\n  provider: deepseek-official\n"
                 "  model: StudentChose\n  reasoningEffort: low\n")
    up, base, _port = _seed_models_server()
    try:
        assert main("use", "litellm", "--base", base, "--key", "lkey") == 0
        assert main("seed-settings") == 0
        text = p.read_text()
        assert "model: StudentChose" in text          # untouched
        assert "model: Socrates" not in text           # not re-seeded
    finally:
        up.stop()


def test_seed_no_model_determined_skips(tmp_path, monkeypatch, capsys):
    """Unreachable endpoint + no DSH_DEFAULT_MODEL → skip (no fabricated file).

    A skip is BENIGN (best-effort; the student just picks a model), so the
    exit code is 0 — not a warning that would alarm the boot hook.
    """
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    # provider points at a dead port so discovery finds no models
    assert main("use", "litellm", "--base", "http://127.0.0.1:1/v1",
                "--key", "lkey", "--no-discover") == 0
    assert main("seed-settings") == 0          # benign skip, not a failure
    assert not (tmp_path / "settings.yaml").exists()
    assert "no model name could be determined" in capsys.readouterr().err


def test_seed_settings_no_hardcoded_name_regression():
    """The package contains NO baked model name or window literal.

    Guards the whole point of moving the seed out of the image: the shim must
    derive model + contextWindow from the endpoint (or the env), never from a
    hardcoded default.
    """
    import dsh_openai_shim.proxy_cli as pc
    import inspect
    src = inspect.getsource(pc)
    # "Socrates" and the old baked window must not appear as literals in the CLI
    assert '"Socrates"' not in src and "'Socrates'" not in src
    assert "262144" not in src


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
