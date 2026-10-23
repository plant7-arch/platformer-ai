"""Shared pytest setup.

Header: the project version, platform, Python, the JAX backend and devices, the settings that change what the tests do
(JAX_PLATFORMS, XLA memory, PLATFORMER_LEVEL) and the level and env sizes they run on.
Short test summary: skipped tests (-ra in pyproject.toml), plus notes on how this run differs from the usual one.
"""
import os
import platform
import sys
import tomllib
from pathlib import Path

import jax
import pytest

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "PLATFORMER_LEVEL",
            "JAX_COMPILATION_CACHE_DIR")


def project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def notes() -> list[str]:
    out = []
    if jax.default_backend() == "cpu":
        out.append("INFO backend cpu: no GPU in use (JAX_PLATFORMS=cpu or no CUDA device); the suite takes longer, "
                   "the numbers are the same")
    if os.environ.get("PLATFORMER_LEVEL"):
        out.append(f"WARN PLATFORMER_LEVEL={os.environ['PLATFORMER_LEVEL']}: tests that use the default level expect "
                   "the built-in one and may fail")
    return out


def pytest_report_header(config):
    from platformer_ai import env as E
    from platformer_ai import level as L
    from platformer_ai.ppo import Config

    p = project()
    deps = ", ".join(p.get("dependencies", []))
    settings = "  ".join(f"{k}={os.environ.get(k, '-')}" for k in SETTINGS)
    c = Config()
    return [
        f"project: {p['name']} {p['version']} (pyproject.toml, requires-python {p.get('requires-python', '?')})",
        f"platform: {platform.platform()}  python {sys.version.split()[0]} ({sys.executable})",
        f"jax {jax.__version__}: backend {jax.default_backend()}, devices {jax.devices()}",
        f"dependencies: {deps}",
        f"settings: {settings}",
        f"level: {L.LEVEL_SOURCE}  {L.W}x{L.H}, {count(L.N_ENEMIES, 'enemy', 'enemies')}, "
        f"{count(L.N_PLATS, 'moving platform', 'moving platforms')}, {count(L.N_COINS, 'coin', 'coins')}"
        f"{'  (built-in map)' if L.IS_BUILTIN else ''}",
        f"env: obs {E.OBS_DIM}, actions {E.N_ACTIONS}, max steps {E.MAX_STEPS};  ppo defaults: {c.num_envs} envs x "
        f"{c.num_steps} steps, {c.total_timesteps:,} total, hidden {c.hidden}",
    ]


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """The notes go under 'short test summary info' (pytest prints that header itself when something was skipped or
    failed; otherwise it is printed here)."""
    lines = notes()
    if not lines:
        return
    shown = any(terminalreporter.stats.get(k) for k in ("failed", "error", "skipped", "xfailed", "xpassed"))
    if not shown:
        terminalreporter.write_sep("=", "short test summary info")
    for line in lines:
        terminalreporter.write_line(line)
