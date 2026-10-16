"""PPO written PureJaxRL-style: environment stepping, GAE and the optimiser updates all live
inside one jitted function, so nothing leaves the GPU during an update.

    init, update, network = make_train(Config())
    runner = init(jax.random.PRNGKey(0))
    runner, metrics = update(runner)        # one rollout (num_envs * num_steps) + PPO epochs

runner = (train_state, env_states, obs, rng, curriculum).  The adaptive curriculum (Config.curriculum, several levels
only) is a small dict of per-level numbers that lives on the GPU with the rest:

    n, g     faded counts of the episodes that started at the real spawn and ended (n), and that reached the goal (g),
             from the training rollouts themselves; every update multiplies them by curriculum_decay first, so they
             follow the agent as it improves (decay 0.95: roughly the last 20 updates count)
    rate     (g + 1) / (n + 2): the estimated goal rate (a level with no episodes yet counts as 0.5)
    p        mix / L + (1 - mix) * (1 - rate) / sum(1 - rate): the chance each level is picked at a reset.  A share
             `curriculum_mix` stays uniform (nothing is forgotten), the rest goes where the agent fails
    logits   log p, handed to env.step for the resets of the next rollout
Without the curriculum the level choice is uniform and every number is exactly as before.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from platformer_ai import env as E


@dataclass(frozen=True)
class Config:
    total_timesteps: int = 25_000_000
    num_envs: int = 1024
    num_steps: int = 128           # rollout length per env per update
    update_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    seed: int = 0
    random_phases: bool = True     # platform / enemy start at random phases, spawn position jittered
    random_starts: bool = True     # some episodes start at random points of the level ('S' markers)
    spawn_prob: float = 0.5        # with random_starts: share of episodes starting at the spawn
    progress_coef: float = 0.1     # bonus per tile of new furthest-right progress (0 = off)
    curriculum: bool = False       # adaptive curriculum: pick levels the agent fails on more often (several levels)
    curriculum_mix: float = 0.3    # ... share of resets that stay uniform over all levels
    curriculum_decay: float = 0.95  # ... per update fading of the per-level episode counts

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def n_updates(self) -> int:
        return self.total_timesteps // self.batch_size

    def problems(self) -> list[str]:
        """What makes this configuration unusable (empty list = fine)."""
        out = []
        if self.num_envs < 1 or self.num_steps < 1:
            out.append(f"--num-envs and --num-steps must be at least 1 (got {self.num_envs} and {self.num_steps})")
        elif self.batch_size % self.num_minibatches:
            out.append(f"one update is --num-envs x --num-steps = {self.num_envs} x {self.num_steps} = "
                       f"{self.batch_size:,} steps, which does not split into {self.num_minibatches} equal minibatches: "
                       f"make it a multiple of {self.num_minibatches}")
        elif self.n_updates < 1:
            out.append(f"--total-steps {self.total_timesteps:,} is less than one update ({self.num_envs} envs x "
                       f"{self.num_steps} steps = {self.batch_size:,}): nothing would be trained")
        return out


def restore_params(template, data: bytes):
    """Policy bytes (a params.msgpack) -> params with the structure of `template`.  Raises ValueError if they do not
    fit (another network width, another observation size, or not a policy file at all): flax's from_bytes alone
    accepts arrays of any shape and the mismatch would only show up later as a traceback deep inside the network."""
    try:
        params = serialization.from_bytes(template, data)
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"not a policy file ({e.__class__.__name__})") from None
    have, want = jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(template)
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(template):
        raise ValueError("the network layout differs")
    for a, b in zip(have, want, strict=True):
        if np.shape(a) != np.shape(b):
            raise ValueError(f"a weight has shape {tuple(np.shape(a))}, this network needs {tuple(np.shape(b))}")
    return params


class ActorCritic(nn.Module):
    n_actions: int
    hidden: int = 128

    @nn.compact
    def __call__(self, x):
        def mlp(h, out, scale):
            h = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                                 bias_init=constant(0.0))(h))
            h = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                                 bias_init=constant(0.0))(h))
            return nn.Dense(out, kernel_init=orthogonal(scale), bias_init=constant(0.0))(h)

        logits = mlp(x, self.n_actions, 0.01)
        value = mlp(x, 1, 1.0)
        return logits, jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array


def _log_prob(logits, action):
    logp = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(logp, action[..., None], axis=-1)[..., 0], logp


def curriculum_init(n_levels: int) -> dict:
    """No episodes seen yet: every level estimated at 0.5, picked uniformly."""
    z = jnp.zeros(n_levels, jnp.float32)
    p = jnp.full(n_levels, 1.0 / n_levels, jnp.float32)
    return dict(n=z, g=z, rate=z + 0.5, p=p, logits=jnp.log(p))


def curriculum_update(curr: dict, level, ended, reached, mix: float, decay: float) -> dict:
    """Fold one rollout into the curriculum.  level / ended / reached: per transition, the level index, 1 if an
    episode from the spawn ended there, 1 if it also reached the goal."""
    n_lv = curr["n"].shape[0]
    n = decay * curr["n"] + jax.ops.segment_sum(ended, level, num_segments=n_lv)
    g = decay * curr["g"] + jax.ops.segment_sum(reached, level, num_segments=n_lv)
    rate = (g + 1.0) / (n + 2.0)
    fail = 1.0 - rate
    p = mix / n_lv + (1.0 - mix) * fail / fail.sum()
    return dict(n=n, g=g, rate=rate, p=p, logits=jnp.log(p))


def make_train(cfg: Config, levels=None):
    """levels: env.make_levels(...) to train on (default: the level of level.py).  Every environment plays a random
    one of them and gets a new random one at every reset, so each batch mixes all levels."""
    if cfg.num_envs < 1 or cfg.num_steps < 1 or cfg.batch_size % cfg.num_minibatches:
        raise ValueError(cfg.problems()[0])
    network = ActorCritic(E.N_ACTIONS, cfg.hidden)
    n_lv = E.n_levels(E.DEFAULT_LEVELS if levels is None else levels)
    use_curriculum = cfg.curriculum and n_lv > 1
    step_fn = functools.partial(
        E.step, random_starts=cfg.random_starts, random_phases=cfg.random_phases,
        spawn_prob=cfg.spawn_prob, progress_coef=cfg.progress_coef, levels=levels)
    if use_curriculum:
        v_step = jax.vmap(lambda s, a, lg: step_fn(s, a, level_logits=lg), in_axes=(0, 0, None))
    else:
        v_step = jax.vmap(lambda s, a, lg: step_fn(s, a), in_axes=(0, 0, None))
    v_reset = jax.vmap(functools.partial(
        E.reset, random_start=cfg.random_starts, random_phases=cfg.random_phases,
        spawn_prob=cfg.spawn_prob, levels=levels))
    v_observe = jax.vmap(functools.partial(E.observe, levels=levels))

    def lr_schedule(count):
        # `count` counts gradient steps; decay linearly to 0 over all updates
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.n_updates
        return cfg.lr * frac

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=lr_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
    )

    def init(rng):
        rng, k_net, k_env = jax.random.split(rng, 3)
        params = network.init(k_net, jnp.zeros((E.OBS_DIM,), jnp.float32))
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        env_states = v_reset(jax.random.split(k_env, cfg.num_envs))
        obs = v_observe(env_states)
        return train_state, env_states, obs, rng, curriculum_init(n_lv)

    def update(runner):
        train_state, env_states, obs, rng, curr = runner

        # ---------------- collect a rollout ----------------
        def env_step(carry, _):
            env_states, obs, rng = carry
            rng, key = jax.random.split(rng)
            logits, value = network.apply(train_state.params, obs)
            action = jax.random.categorical(key, logits)
            log_prob, _ = _log_prob(logits, action)
            env_states, next_obs, reward, done, info = v_step(env_states, action, curr["logits"])
            # time-limit truncation is not a real terminal: bootstrap from the final state's value
            _, v_final = network.apply(train_state.params, info["final_obs"])
            reward = reward + cfg.gamma * v_final * info["truncated"].astype(jnp.float32)
            tr = Transition(done.astype(jnp.float32), action, value, reward, log_prob, obs)
            ep = dict(done=done, ep_return=info["ep_return"],
                      ep_length=info["ep_length"].astype(jnp.float32),
                      ep_coins=info["ep_coins"], goal=info["goal"].astype(jnp.float32),
                      max_x=info["max_x"], from_spawn=info["from_spawn"], level=info["level"])
            return (env_states, next_obs, rng), (tr, ep)

        (env_states, obs, rng), (traj, ep) = jax.lax.scan(
            env_step, (env_states, obs, rng), None, length=cfg.num_steps)

        # ---------------- GAE ----------------
        _, last_value = network.apply(train_state.params, obs)

        def gae_step(carry, tr):
            gae, next_value = carry
            delta = tr.reward + cfg.gamma * next_value * (1.0 - tr.done) - tr.value
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - tr.done) * gae
            return (gae, tr.value), gae

        _, advantages = jax.lax.scan(
            gae_step, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        targets = advantages + traj.value

        # ---------------- PPO epochs ----------------
        def minibatch_step(train_state, mb):
            mb_obs, mb_action, mb_logp, mb_value, mb_adv, mb_target = mb

            def loss_fn(params):
                logits, value = network.apply(params, mb_obs)
                log_prob, logp_all = _log_prob(logits, mb_action)
                entropy = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
                value_clipped = mb_value + jnp.clip(value - mb_value, -cfg.clip_eps, cfg.clip_eps)
                v_loss = 0.5 * jnp.maximum((value - mb_target) ** 2,
                                           (value_clipped - mb_target) ** 2).mean()
                adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                ratio = jnp.exp(log_prob - mb_logp)
                pg_loss = jnp.maximum(-adv * ratio,
                                      -adv * jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)).mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                return loss, (pg_loss, v_loss, entropy)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            return train_state.apply_gradients(grads=grads), (loss, *aux)

        def epoch(carry, _):
            train_state, rng = carry
            rng, key = jax.random.split(rng)
            batch = (traj.obs, traj.action, traj.log_prob, traj.value, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), batch)
            perm = jax.random.permutation(key, cfg.batch_size)
            batch = jax.tree_util.tree_map(lambda x: x[perm], batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((cfg.num_minibatches, -1) + x.shape[1:]), batch)
            train_state, losses = jax.lax.scan(minibatch_step, train_state, minibatches)
            return (train_state, rng), losses

        (train_state, rng), losses = jax.lax.scan(
            epoch, (train_state, rng), None, length=cfg.update_epochs)

        # ---------------- metrics ----------------
        # Sums and counts of the episodes that finished during this rollout (train.py averages
        # them over a window of updates).  "sp_*" = episodes that started at the real spawn:
        # these are the numbers that tell you whether the whole level is being solved.
        f32 = jnp.float32
        all_eps = ep["done"].astype(f32)
        sp_eps = (ep["done"] & ep["from_spawn"]).astype(f32)

        def total(v, m):
            return (v * m).sum()

        loss, pg_loss, v_loss, entropy = (x.mean() for x in losses)
        metrics = dict(
            n_all=all_eps.sum(), all_return=total(ep["ep_return"], all_eps),
            all_goal=total(ep["goal"], all_eps),
            n_spawn=sp_eps.sum(), sp_return=total(ep["ep_return"], sp_eps),
            sp_coins=total(ep["ep_coins"], sp_eps), sp_goal=total(ep["goal"], sp_eps),
            sp_maxx=total(ep["max_x"], sp_eps), sp_len=total(ep["ep_length"], sp_eps),
            loss=loss, pg_loss=pg_loss, v_loss=v_loss, entropy=entropy,
        )
        if use_curriculum:
            curr = curriculum_update(curr, ep["level"].ravel(), sp_eps.ravel(), (ep["goal"] * sp_eps).ravel(),
                                     cfg.curriculum_mix, cfg.curriculum_decay)
            metrics.update(level_p=curr["p"], level_rate=curr["rate"], level_n=curr["n"])
        return (train_state, env_states, obs, rng, curr), metrics

    return init, jax.jit(update), network
