# Changelog

## 0.2.0 (2026-10-23)

### New
- **Watch a run learn.** `viewer` and `play` follow the policy: a newer `params.msgpack` (the run that is training
  saved again, or a newer latest run) is loaded at the start of the next episode. The viewer's HUD and `play` name the
  policy (`RUN  update 75/190  eval goal 0.62`); the viewer's `L` key loads the newest one at once; `--no-follow` keeps
  the first one. If there is no policy yet, the agent takes over as soon as a run saves one.
- `train --save-every N` (default 25): how often the policy and the checkpoint are saved; a small N gives the viewer
  more snapshots to show. It can also be given with `--resume`.
- `play --episodes 0` plays until Ctrl-C.
- `env.step` returns the state an episode ended in, before the auto-reset: `info["final_state"]` (training and
  evaluation do not use it, so it costs nothing there).
- pytest: the header names the project version, platform, Python, JAX backend and devices, the settings and the level;
  the short test summary lists skipped tests and notes (e.g. running on the CPU).

### Fixed
- **Random starts inside an enemy.** A random start riding a moving platform with an enemy on it could begin inside
  that enemy when the platform was narrow (the enemy cannot be pushed far enough along its short patrol): on
  `docs/levels/03_many.txt` 12.6% of all training starts, 6.4% dead within 3 steps whatever the agent did. Such a
  start now becomes a spawn start. Every other start is unchanged (checked bit for bit on the built-in level and 30
  generated tier-3 levels), so training on them gives the same numbers as before.
- **`--resume` from another folder.** `run.json` kept `--levels` / `--test-levels` / `PLATFORMER_LEVEL` as typed, so
  resuming from a subfolder could not find the levels. They are now kept relative to the project folder (absolute
  outside it). Older runs still resume: their paths are tried as they are, then inside the project folder.
- **Half-written policy files.** `params.msgpack` was written in place, so a viewer opening a run that was saving (or
  Ctrl-C during the save) could see a broken file, reported as "does not fit a network of width 128". Every run file is
  now written beside and renamed over the old one; the policy goes last, after `run.json`.
- **`play` used the GPU.** It now runs on the CPU like the viewer (`JAX_PLATFORMS=cpu` unless you set it), so it does
  not take GPU memory from a training run.
- **`play` never showed how an episode ended**: the last frame was already the next episode's spawn. It now shows the
  final frame with `goal reached!` / `game over` / `time is up`. The viewer's end-of-episode picture is the real final
  state too (it was one step before it).
- **The built-in level's signs on every level.** "Falling hurts..." and "Great Job!" are only drawn on the built-in
  map (`level.py`, or a copy such as `docs/builtin.txt`).
- `levelplan` on a folder without level files printed a traceback; it is a message now (exit code 2).
- A missing level folder was reported as "cannot read the level file ...: No such file or directory"; now "no such
  level file or folder".
- `train -v 0` help: with `--eval-every 0` there is no final evaluation either.

## 0.1.0

First version: the JAX platformer and PPO, viewer, levels as input, level generator and planner, evaluation,
curriculum, run folders with resume.
