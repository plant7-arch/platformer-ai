"""The terminal player: runs on the CPU by default, shows how an episode ended, follows the policy.  Needs jax; each
case is its own process (play reads the level when it is imported)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(code=None, args=(), extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS", "PLATFORMER_LEVEL")}
    env.update(extra_env or {})
    cmd = [sys.executable, "-c", code] if code else [sys.executable, "-m", "platformer_ai.play", *args]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)


def test_play_defaults_to_the_cpu():
    """Bug (0.1.0): play took GPU memory next to a training run (the viewer already stayed on the CPU).  A value you
    set yourself (JAX_PLATFORMS=cuda) is kept: os.environ.setdefault, as in the viewer."""
    code = "import platformer_ai.play, os, jax; print(os.environ['JAX_PLATFORMS'], jax.default_backend())"
    r = run(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["cpu", "cpu"]


def test_play_finishes_episodes_and_names_the_policy():
    r = run(args=["--params", "pretrained", "--level", "docs/levels/01_walk.txt", "--episodes", "2", "--fps", "1000"])
    assert r.returncode == 0, r.stderr
    lines = [x for x in r.stdout.splitlines() if x.startswith("episode")]
    assert len(lines) == 2 and all("goal=True" in x and "(pretrained)" in x for x in lines), r.stdout


def test_the_last_frame_shows_the_player_on_the_flag():
    """Bug (0.1.0): play drew the state after the auto-reset, so the last frame was always the spawn."""
    code = """
import jax
from platformer_ai import env as E, play, render
s = E.reset()
done = False
while not done:
    s, _, _, done, info = E.step(s, 2)
frame = render.frame(info["final_state"]).splitlines()
print(play.outcome(info))
print(frame[1].index("@"), frame[1].index("G"))
"""
    r = run(code, extra_env={"PLATFORMER_LEVEL": "docs/levels/01_walk.txt"})
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    assert out[0] == "goal reached!"
    player, goal = map(int, out[1].split())
    assert player >= goal - 1                                             # at the flag, not back at column 0


def test_play_refuses_bad_numbers():
    r = run(args=["--episodes", "-1"])
    assert r.returncode == 2 and "--episodes must be 0 or more" in r.stderr
