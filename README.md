# platformer-ai

A small platformer written as pure JAX functions, plus PPO. Environment stepping, the policy, GAE and the
optimiser updates all run inside **one jitted function on the GPU**: no game engine, no CPU round trips
(the CPU only dispatches the update and reads back a few logging numbers). A pygame window lets you watch
the trained agent, or take over and play the same level yourself.

![The viewer](docs/preview.png)

On an RTX 3050 laptop GPU (plugged in), 25M environment steps take about a minute (~600k steps/s). The
included policy reaches the flag in 99% of fixed evaluation episodes with random platform/enemy phases
(return 13.9 of a possible 14).

## Quick start

Needs [uv](https://docs.astral.sh/uv/), which installs the Python version pinned in `.python-version`. The lockfile targets an NVIDIA GPU with CUDA 13
(`jax[cuda13]`); for CUDA 12 use `jax[cuda12]`, for CPU-only plain `jax` (it works, just slower).

```bash
git clone https://github.com/plant7-arch/platformer-ai.git
cd platformer-ai
uv sync

uv run python -m platformer_ai.viewer                 # watch the included trained agent (Tab: play yourself)
uv run python -m platformer_ai.train                  # train your own: 25M steps, saves runs/latest/params.msgpack
uv run pytest                                         # tests
```

Optional JAX settings: put these in a `.env` file (it is git-ignored) and run `uv run --env-file .env ...`,
or `export` them in your shell:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false      # do not grab all GPU memory up front
JAX_COMPILATION_CACHE_DIR=.jax_cache     # faster start from the second run on
```

If you run out of GPU memory, lower `--num-envs` (default 1024). On a laptop, plug it in: the GPU runs in a
much lower power state on battery and training can be several times slower.

## Project layout

```
src/platformer_ai/
  level.py      the level as ASCII (edit it! also holds the S start points)
  env.py        reset / step / observe: pure functions on a small pytree; vmap for many envs
  ppo.py        PureJaxRL-style PPO (MLP actor-critic)
  evaluate.py   fixed GPU evaluation: 512 full episodes from the spawn with random phases
  train.py      training loop, prints metrics + periodic evaluation, saves params
  play.py       watch a trained policy in the terminal (ASCII)
  render.py     ASCII rendering
  terminal.py   alternate-screen drawing used by play.py
  viewer.py     pygame window: watch the agent or play yourself (reads the env, never trains)
  viewlogic.py  viewer logic without pygame/jax: coordinates, animation, camera, controls
  scene.py      numpy renderer for one 480x270 frame (tilemaps + sprites)
  world_tiles.npz  tile layout of the art, decoded once from the original Godot scene
assets/         sprites/ and fonts/ from the Brackeys platformer bundle (CC0)
runs/latest/    the trained policy (params.msgpack)
tests/          env behaviour tests, viewer-logic tests, a PPO smoke test
```

## Training: what the options do and how to read the log

The level has two hard parts (the ferry and the enemy) that a random policy practically never gets
through, so plain PPO from the spawn stalls after the first coins. The switches below fix that:

* **Random phases** (`--no-random-phases` turns them off). Every episode starts with the platform and the
  enemy at random phases, and the spawn position is jittered. With fixed phases the agent can learn one
  timed route by heart and never look at the platform or the enemy; with random phases it has to react.
* **Random starts** (`--no-random-starts` turns them off). With probability `1 - spawn-prob` (default
  0.5) an episode begins at one of the `S` points in `level.py` or riding the ferry, so the hard parts are
  practised directly. `play` always starts at the spawn.
* **Progress bonus** (`--progress-coef 0.1`, `0` = off): +0.1 per tile of *new* furthest-right progress.
  Walking back and forth pays nothing, waiting costs nothing (the ferry needs waiting).
* Longer horizon: `gamma` 0.995 and 128-step rollouts.

```
upd   60/190  steps ...  SPAWN eps  310 return  6.20 coins  5.1 goal 0.30 max_x  52.3 len  612 | ALL return ... 
    EVAL greedy   512 fresh episodes, random phases:  return 13.80  coins  8.9  goal 0.98  max_x  68.5  len  291
    EVAL sampled  512 fresh episodes, random phases:  return 13.10  coins  8.7  goal 0.93  max_x  67.9  len  300
```

* **`EVAL` lines are the number to trust** (every 20 updates, `--eval-every`). 512 fresh episodes from the
  real spawn with random phases, always the same 512 start conditions, each run to its very end
  (900 steps). `greedy` = most likely action, `sampled` = like the training policy. Max return is 14;
  `goal` is the share that reaches the flag. If `goal` is high here, the agent really reacts to the
  platform and enemy.
* `SPAWN ...` (every update) only averages episodes that have *finished*, so while many episodes are still
  running it looks better than reality (stuck episodes only count when they time out after 900 steps).
  Use it for trends, not for absolute values.
* `ALL ...` includes the random starts (easier on average). Everything is averaged over 10 updates.
* Logged returns are the task return (coins + goal + death); the progress bonus is only used for training.

If the spawn is learned too slowly, raise `--spawn-prob` (e.g. 0.75, or 1.0 with `--no-random-starts`);
if the agent only works from the spawn but not from the other starts, lower it.

`play --random-phases` shows an agent with random phases and a jittered spawn.

`play` draws on the terminal's alternate screen (like `vim` or `htop`): your scrollback, including the
end of the training log, stays intact, and the per-episode results are printed after the game view closes.
Ctrl-C leaves cleanly. When stdout is not a terminal (piped), it only prints the results.

## The viewer (pygame)

```bash
uv run python -m platformer_ai.viewer --params runs/latest/params.msgpack
```

A 480x270 picture scaled up in whole-number steps (the window is resizable), camera following the knight,
tilemaps, knight / slime / coin / platform animations, coin counter. Run it from the repository root so it
finds `assets/` (or pass `--assets DIR`).

| key | action |
| --- | --- |
| `Tab` | switch between the **agent** and **you** (same level, same physics) |
| arrows / `A` `D`, `Space` / `Up` / `W` | move, jump |
| `P`, `.` | pause, advance one step while paused |
| `R` | restart the episode |
| `Esc` / `Q` | quit |

Options: `--human` (start with the keyboard), `--sample`, `--random-phases`, `--scale N` (initial window size),
`--assets DIR`, `--seed`. Without a params file it starts in keyboard mode.

The viewer runs one env on the **CPU** (`JAX_PLATFORMS=cpu` is set unless you set it yourself), so it never
touches the GPU, even while a training run is going. The art is only pictures: the physics are `env.py`, and
the sprites are placed from the env state (feet position, one tile = 16 px).

## The game (ported from the original Godot level)

* **Level**: the 70 x 10 tile collision map of the Godot project, as ASCII in `level.py`
  (the 3 nearly flat slope tiles became flat solid tiles).
* **Player**: a 0.6 x 0.8 box. Run left/right, jump about 2.7 tiles high (only from the ground or a platform).
  Units are tiles; one env step = 1/30 s.
* **Actions (6)**: horizontal `[none, left, right][a % 3]`, jump when `a >= 3`.
* **One-way platforms**: a static one (spawn drops onto it; it is out of reach from the ground) and a
  moving one that ferries the player across the 8-tile pit while carrying whoever stands on it.
* **Enemy**: patrols between two walls, turns around when its sensor touches a wall, touch = death.
* **Coins**: 9 (8 on the map, 1 riding the moving platform), circles with the original 5 px radius.
* **Goal**: a flag at the far right. The original level had no way to win; this is the one addition.
* **Speeds**: our player is slower than the Godot one (6 vs 11.25 tiles/s), so the platform and the
  enemy were scaled by the same factor (3.2 and 2.0 tiles/s); relative timing is preserved.
* **Reward**: +1 coin, +5 goal (ends the episode), -1 death by pit or enemy (ends the episode).
  Max return is 9 coins + 5 = 14. A scripted bot that waits for the ferry and hops the enemy scores 12.
* Episode is cut at 900 steps and treated as truncation (PPO bootstraps from the last state's value).
* **Observation (778 floats)**: 9 x 17 tile window around the player with 5 channels (solid, one-way /
  moving platform, coin, goal, enemy) plus 13 scalars: velocity, on-floor, sub-tile position, progress,
  time used, and relative position + direction of the enemy and of the moving platform.

## Where to add features later

* **More level / different layout**: edit `LEVEL` in `level.py` (all sizes are derived from it).
* **More enemies / platforms**: make `enemy_x`, `plat_x` arrays, `vmap` their update inside `step`, and add
  their channels to `observe`.
* **Random levels or start positions**: `reset(key)` already takes a key; thread a key through the
  state if `step` needs randomness after reset.

## Credits and license

* Code: [MIT](LICENSE).
* Art and fonts: [Brackeys Platformer Bundle](https://brackeysgames.itch.io/brackeys-platformer-bundle), CC0.
* The level layout and mechanics were ported from the author's earlier Godot version of this project.
