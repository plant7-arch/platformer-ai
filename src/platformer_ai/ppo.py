"""PPO written PureJaxRL-style: environment stepping, GAE and the optimiser updates all live
inside one jitted function, so nothing leaves the GPU during an update.

    init, update = make_train(Config())
    runner = init(jax.random.PRNGKey(0))
    runner, metrics = update(runner)        # one rollout (num_envs * num_steps) + PPO epochs
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

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def n_updates(self) -> int:
        return self.total_timesteps // self.batch_size


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


def make_train(cfg: Config, levels=None):
    """levels: env.make_levels(...) to train on (default: the level of level.py).  Every environment plays a random
    one of them and gets a new random one at every reset, so each batch mixes all levels."""
    network = ActorCritic(E.N_ACTIONS, cfg.hidden)
    v_step = jax.vmap(functools.partial(
        E.step, random_starts=cfg.random_starts, random_phases=cfg.random_phases,
        spawn_prob=cfg.spawn_prob, progress_coef=cfg.progress_coef, levels=levels))
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
        return train_state, env_states, obs, rng

    def update(runner):
        train_state, env_states, obs, rng = runner

        # ---------------- collect a rollout ----------------
        def env_step(carry, _):
            env_states, obs, rng = carry
            rng, key = jax.random.split(rng)
            logits, value = network.apply(train_state.params, obs)
            action = jax.random.categorical(key, logits)
            log_prob, _ = _log_prob(logits, action)
            env_states, next_obs, reward, done, info = v_step(env_states, action)
            # time-limit truncation is not a real terminal: bootstrap from the final state's value
            _, v_final = network.apply(train_state.params, info["final_obs"])
            reward = reward + cfg.gamma * v_final * info["truncated"].astype(jnp.float32)
            tr = Transition(done.astype(jnp.float32), action, value, reward, log_prob, obs)
            ep = dict(done=done, ep_return=info["ep_return"],
                      ep_length=info["ep_length"].astype(jnp.float32),
                      ep_coins=info["ep_coins"], goal=info["goal"].astype(jnp.float32),
                      max_x=info["max_x"], from_spawn=info["from_spawn"])
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
        return (train_state, env_states, obs, rng), metrics

    return init, jax.jit(update), network
