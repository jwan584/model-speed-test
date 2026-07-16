from __future__ import annotations

import asyncio
import json
import os
import platform
import shlex
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


_SECRET_NAMES = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "BRAVE_API_KEY"}


def scrubbed_host_env() -> dict[str, str]:
    """Environment for Docker/local subprocesses; never forward API credentials."""
    return {
        key: value
        for key, value in os.environ.items()
        if key not in _SECRET_NAMES
        and not key.endswith("API_KEY")
        and not key.endswith("_TOKEN")
        and not key.endswith("_SECRET")
    }


def native_docker_arch(machine: str | None = None) -> str:
    machine = (machine or platform.machine()).lower()
    return {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "amd64": "amd64"}.get(machine, machine)


class Sandbox:
    def __init__(self, config: dict[str, Any], workdir: Path, timeout: float):
        self.config, self.workdir, self.timeout = config, workdir.resolve(), timeout
        self.backend = config.get("backend", "docker")
        self.container = f"gdpval-{uuid.uuid4().hex[:12]}"

    @classmethod
    async def preflight(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Fail closed before a paid run if Docker or the image is unsuitable."""
        backend = config.get("backend", "docker")
        if backend == "local":
            if not config.get("allow_local", False):
                raise RuntimeError("Local sandbox requires sandbox.allow_local=true")
            return {"backend": "local", "warning": "unisolated development backend"}
        if backend != "docker":
            raise ValueError(f"Unknown sandbox backend: {backend}")
        if shutil.which("docker") is None:
            raise RuntimeError(
                "Docker CLI is unavailable. Install Docker Desktop or Colima, then run "
                "`gdpval-time --config CONFIG --preflight`. The harness will not fall back to local execution."
            )
        env = scrubbed_host_env()
        server = await cls._capture(
            ["docker", "info", "--format", "{{json .}}"], Path.cwd(), 30, env
        )
        try:
            server_info = json.loads(server)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Docker daemon returned invalid metadata: {server[:500]}") from exc
        image = config.get("image")
        if not image:
            raise ValueError("sandbox.image is required for Docker execution")
        image_raw = await cls._capture(
            ["docker", "image", "inspect", image, "--format", "{{json .}}"],
            Path.cwd(), 30, env,
        )
        image_info = json.loads(image_raw)
        host_arch = native_docker_arch()
        image_arch = image_info.get("Architecture")
        require_native = config.get("require_native_arch", True)
        if require_native and image_arch != host_arch:
            raise RuntimeError(
                f"Image architecture linux/{image_arch} does not match host linux/{host_arch}; "
                "emulation distorts tool timing. Rebuild with docker/build-native.sh."
            )
        health_command = config.get("healthcheck_command", "gdpval-image-healthcheck")
        health_raw = await cls._capture(
            [
                "docker", "run", "--rm", "--network", "none", "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", image, health_command,
            ],
            Path.cwd(), float(config.get("preflight_timeout_seconds", 120)), env,
        )
        try:
            health = json.loads(health_raw.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            health = {"ok": True, "raw": health_raw[-1000:]}
        return {
            "backend": "docker",
            "server_arch": server_info.get("Architecture"),
            "server_os": server_info.get("OperatingSystem"),
            "image": image,
            "image_id": image_info.get("Id"),
            "image_arch": image_arch,
            "native_arch": image_arch == host_arch,
            "healthcheck": "passed",
            "capabilities": health,
        }

    def docker_run_command(self) -> list[str]:
        image = self.config["image"]
        cmd = [
            "docker", "run", "-d", "--rm", "--name", self.container,
            "--network", self.config.get("network", "bridge"),
            "--memory", str(self.config.get("memory", "8g")),
            "--memory-swap", str(self.config.get("memory_swap", self.config.get("memory", "8g"))),
            "--cpus", str(self.config.get("cpus", 4)),
            "--pids-limit", str(self.config.get("pids_limit", 512)),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=2g",
            "--mount", f"type=bind,src={self.workdir},dst=/workspace",
            "-w", "/workspace",
        ]
        container_user = self.config.get("user", "host")
        if container_user == "host" and hasattr(os, "getuid"):
            container_user = f"{os.getuid()}:{os.getgid()}"
        if container_user:
            cmd.extend(["--user", str(container_user)])
        if self.config.get("require_native_arch", True):
            cmd.extend(["--platform", f"linux/{native_docker_arch()}"])
        cmd.extend([image, "sleep", "infinity"])
        return cmd

    async def __aenter__(self):
        if self.backend == "local":
            if not self.config.get("allow_local", False):
                raise RuntimeError("Local sandbox requires sandbox.allow_local=true")
            return self
        if self.backend != "docker":
            raise ValueError(f"Unknown sandbox backend: {self.backend}")
        if shutil.which("docker") is None:
            raise RuntimeError("Docker CLI is unavailable; benchmark mode will not fall back to local execution")
        await self._process(self.docker_run_command(), self.workdir, 120)
        return self

    async def __aexit__(self, *_):
        if self.backend == "docker":
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", self.container,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                env=scrubbed_host_env(),
            )
            await proc.wait()

    async def run(self, command: str) -> str:
        if self.backend == "docker":
            cmd = ["docker", "exec", self.container, "bash", "-lc", command]
        else:
            command = command.replace("/workspace", shlex.quote(str(self.workdir)))
            cmd = ["bash", "-lc", command]
        return await self._process(cmd, self.workdir, self.timeout)

    async def _process(self, cmd: list[str], cwd: Path, timeout: float) -> str:
        env = scrubbed_host_env()
        if self.backend == "local":
            env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
        return await self._capture(cmd, cwd, timeout, env)

    @staticmethod
    async def _capture(cmd: list[str], cwd: Path, timeout: float, env: dict[str, str]) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=env,
        )
        try:
            output, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"Command timed out after {timeout}s")
        text = output.decode(errors="replace")
        if proc.returncode:
            raise RuntimeError(f"Command exited {proc.returncode}:\n{text[-12000:]}")
        return text[-12000:]
