import asyncio
import json

import pytest

from gdpval_timing.sandbox import Sandbox, native_docker_arch, scrubbed_host_env


def test_native_docker_arch_aliases():
    assert native_docker_arch("arm64") == "arm64"
    assert native_docker_arch("aarch64") == "arm64"
    assert native_docker_arch("x86_64") == "amd64"


def test_docker_command_is_limited_native_and_hardened(tmp_path):
    sandbox = Sandbox(
        {
            "backend": "docker", "image": "gdpval:test", "cpus": 2,
            "memory": "6g", "memory_swap": "6g", "pids_limit": 300,
            "require_native_arch": True,
        },
        tmp_path,
        60,
    )
    cmd = sandbox.docker_run_command()
    joined = " ".join(cmd)
    assert "--cpus 2" in joined
    assert "--memory 6g --memory-swap 6g" in joined
    assert "--pids-limit 300" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--read-only" in cmd
    assert f"--platform linux/{native_docker_arch()}" in joined
    assert f"type=bind,src={tmp_path.resolve()},dst=/workspace" in joined
    assert f"--user {__import__('os').getuid()}:{__import__('os').getgid()}" in joined
    assert not any("API_KEY" in part for part in cmd)


def test_scrubbed_env_removes_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-leak")
    monkeypatch.setenv("CUSTOM_API_KEY", "do-not-leak")
    monkeypatch.setenv("SESSION_TOKEN", "do-not-leak")
    monkeypatch.setenv("NORMAL_SETTING", "kept")
    env = scrubbed_host_env()
    assert env["NORMAL_SETTING"] == "kept"
    assert "OPENAI_API_KEY" not in env
    assert "CUSTOM_API_KEY" not in env
    assert "SESSION_TOKEN" not in env


def test_preflight_does_not_fall_back_without_docker(monkeypatch):
    monkeypatch.setattr("gdpval_timing.sandbox.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="will not fall back"):
        asyncio.run(Sandbox.preflight({"backend": "docker", "image": "x"}))


def test_local_requires_explicit_opt_in(tmp_path):
    with pytest.raises(RuntimeError, match="allow_local"):
        asyncio.run(Sandbox.preflight({"backend": "local"}))


def test_preflight_rejects_emulated_image(monkeypatch):
    monkeypatch.setattr("gdpval_timing.sandbox.shutil.which", lambda _: "/usr/bin/docker")

    async def capture(cmd, *_):
        if cmd[1] == "info":
            return json.dumps({"Architecture": native_docker_arch(), "OperatingSystem": "test"})
        return json.dumps({"Architecture": "amd64" if native_docker_arch() == "arm64" else "arm64", "Id": "sha256:x"})

    monkeypatch.setattr(Sandbox, "_capture", capture)
    with pytest.raises(RuntimeError, match="emulation distorts tool timing"):
        asyncio.run(Sandbox.preflight({"backend": "docker", "image": "x"}))


def test_preflight_runs_offline_healthcheck(monkeypatch):
    monkeypatch.setattr("gdpval_timing.sandbox.shutil.which", lambda _: "/usr/bin/docker")
    commands = []

    async def capture(cmd, *_):
        commands.append(cmd)
        if cmd[1] == "info":
            return json.dumps({"Architecture": native_docker_arch(), "OperatingSystem": "test"})
        if cmd[1:3] == ["image", "inspect"]:
            return json.dumps({"Architecture": native_docker_arch(), "Id": "sha256:x"})
        return '{"ok": true, "unavailable_optional_capabilities": ["cadquery"]}'

    monkeypatch.setattr(Sandbox, "_capture", capture)
    result = asyncio.run(Sandbox.preflight({"backend": "docker", "image": "x"}))
    assert result["healthcheck"] == "passed"
    assert result["capabilities"]["unavailable_optional_capabilities"] == ["cadquery"]
    health = commands[-1]
    assert "none" in health
    assert "--read-only" in health
    assert health[-1] == "gdpval-image-healthcheck"
