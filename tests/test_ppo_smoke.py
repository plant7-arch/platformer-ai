"""Tiny end-to-end run of the PPO update: shapes, finiteness, parameters actually change."""
import jax
import numpy as np
import pytest

from platformer_ai.ppo import Config, make_train


@pytest.mark.parametrize("random_starts,random_phases", [(True, True), (False, True), (False, False)])
def test_two_updates_run_and_change_params(random_starts, random_phases):
    cfg = Config(total_timesteps=2 * 16 * 8, num_envs=16, num_steps=8, num_minibatches=2,
                 update_epochs=2, hidden=16, random_starts=random_starts, random_phases=random_phases)
    init, update, _ = make_train(cfg)
    runner = init(jax.random.PRNGKey(0))
    before = jax.tree_util.tree_leaves(runner[0].params)[0]
    for _ in range(2):
        runner, metrics = update(runner)
    metrics = jax.device_get(metrics)
    assert all(np.isfinite(v) for v in metrics.values()), metrics
    after = jax.tree_util.tree_leaves(runner[0].params)[0]
    assert not np.allclose(before, after)


def test_training_on_two_levels_of_different_sizes():
    from platformer_ai import env as E
    from platformer_ai import level as L
    small = L.build("............\nP....o.o...G\n############\n", "small.txt")
    levels = E.make_levels([L.CURRENT, small])
    cfg = Config(total_timesteps=2 * 16 * 8, num_envs=16, num_steps=8, num_minibatches=2, update_epochs=2, hidden=16)
    init, update, _ = make_train(cfg, levels)
    runner = init(jax.random.PRNGKey(0))
    assert set(np.asarray(runner[1].level).tolist()) == {0, 1}          # the batch plays both levels
    for _ in range(2):
        runner, metrics = update(runner)
    assert all(np.isfinite(v) for v in jax.device_get(metrics).values())


def test_config_problems_are_found_before_training():
    from platformer_ai.ppo import Config
    assert Config().problems() == []
    assert "does not split into 8" in Config(num_envs=10, num_steps=3).problems()[0]
    assert "less than one update" in Config(total_timesteps=1000).problems()[0]
    assert "at least 1" in Config(num_envs=0).problems()[0]
    with pytest.raises(ValueError, match="does not split"):
        make_train(Config(num_envs=10, num_steps=3))


def test_restore_params_refuses_another_network():
    import jax.numpy as jnp
    from flax import serialization

    from platformer_ai import env as E
    from platformer_ai.ppo import ActorCritic, restore_params
    x = jnp.zeros((E.OBS_DIM,), jnp.float32)
    small, big = (ActorCritic(E.N_ACTIONS, h).init(jax.random.PRNGKey(0), x) for h in (16, 32))
    data = serialization.to_bytes(small)
    back = restore_params(small, data)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(back), jax.tree_util.tree_leaves(small), strict=True))
    with pytest.raises(ValueError, match="shape"):
        restore_params(big, data)
    with pytest.raises(ValueError):
        restore_params(small, b"not a policy")
