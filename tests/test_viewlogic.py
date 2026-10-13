"""Viewer logic that needs neither a window nor jax."""
import numpy as np
import pytest

from platformer_ai import level as L
from platformer_ai import viewlogic as V


def test_coordinates_match_the_original_scene():
    # the original player node sat at pixel (430, 140); our spawn must be within a few pixels of it
    px, py = V.to_px(L.SPAWN_X, L.SPAWN_Y)
    assert abs(px - 430) < 16 and abs(py - 140) < 8
    assert V.to_px(0, 0) == (160, 80)


def test_tile_art_matches_the_level():
    """Every solid cell of the physics level has a ground tile in the decoded art (and nothing is shifted)."""
    from pathlib import Path
    import platformer_ai
    t = np.load(Path(platformer_ai.__file__).with_name("world_tiles.npz"))
    mid = {(int(x), int(y)) for x, y, *_ in t["mid"]}
    solid = np.argwhere(L.SOLID)
    assert len(solid) > 50
    for r, c in solid:
        assert (c + V.ORIGIN_TX, r + V.ORIGIN_TY) in mid, (r, c)
    assert len(t["background"]) > 1000


@pytest.mark.parametrize("left,right,jump,expect", [
    (0, 0, 0, 0), (1, 0, 0, 1), (0, 1, 0, 2), (1, 1, 0, 0),
    (0, 0, 1, 3), (1, 0, 1, 4), (0, 1, 1, 5), (1, 1, 1, 3)])
def test_keys_to_action(left, right, jump, expect):
    assert V.keys_to_action(bool(left), bool(right), bool(jump)) == expect


def test_keys_match_env_coding():
    for a in range(6):                                   # the env decodes a % 3 -> (none, left, right), a >= 3 -> jump
        h, j = a % 3, a >= 3
        assert V.keys_to_action(h == 1, h == 2, j) == a


def test_player_animation_choice_and_facing():
    assert V.player_anim(0.0, 0.0) == "idle"
    assert V.player_anim(6.0, 0.0) == "run" and V.player_anim(-6.0, 0.0) == "run"
    assert V.player_anim(0.0, -5.0) == "jump" and V.player_anim(6.0, 2.0) == "jump"
    assert V.facing_left(-6.0, False) is True
    assert V.facing_left(6.0, True) is False
    assert V.facing_left(0.0, True) is True and V.facing_left(0.0, False) is False


def test_animation_tables_fit_the_sheets():
    for sheet, (w, h) in dict(player=(256, 256), enemy=(96, 72), coin=(192, 16)).items():
        for a in V.ANIMS[sheet].values():
            for x, y, fw, fh in a["frames"]:
                assert x + fw <= w and y + fh <= h
    assert len(V.ANIMS["player"]["run"]["frames"]) == 16 and len(V.ANIMS["coin"]["spin"]["frames"]) == 12


def test_frame_index_loops():
    assert [V.frame_index(t, 10.0, 4) for t in (0.0, 0.09, 0.1, 0.39, 0.4, 0.55)] == [0, 0, 1, 3, 0, 1]


def test_animator_restarts_when_the_animation_changes():
    an = V.Animator(V.ANIMS["player"])
    idle = V.ANIMS["player"]["idle"]["frames"]
    assert an.frame("idle", 5.0) == idle[0]
    assert an.frame("idle", 5.1) == idle[1]
    an.frame("run", 5.15)                                # switching starts the new animation at its first frame
    assert an.frame("run", 5.15) == V.ANIMS["player"]["run"]["frames"][0]
    assert an.frame("idle", 5.2) == idle[0]


def test_camera_follows_with_offset_limit_and_smoothing():
    cam = V.Camera()
    cam.snap(430, 100)
    assert (cam.x, cam.y) == (430, 96)                   # offset (0, -4)
    cam.snap(430, 140)
    assert cam.y == 135                                  # 136 would show below y = 270: the bottom limit wins
    cam.snap(430, 600)                                   # far below: bottom limit keeps the view above y = 270
    assert cam.y == V.CAM_LIMIT_BOTTOM - V.SCREEN_H / 2
    assert cam.top_left()[1] + V.SCREEN_H == V.CAM_LIMIT_BOTTOM
    cam.snap(100, 100)
    cam.update(300, 100, 1 / 60)
    assert 100 < cam.x < 300                             # moves towards the target, does not jump
    for _ in range(600):
        cam.update(300, 100, 1 / 60)
    assert cam.x == pytest.approx(300, abs=0.01)


def test_platform_runs_and_coins():
    runs = V.platform_runs()
    n_oneway = int(L.ONEWAY.sum())
    assert len(runs) == (n_oneway + 1) // 2 and n_oneway == 2   # the level has one 2-tile floating platform
    (cx, top), = runs
    assert top == float(np.argwhere(L.ONEWAY)[0][0]) and cx == np.argwhere(L.ONEWAY)[:, 1].mean() + 0.5
    coins = V.coin_positions(30.0)
    assert len(coins) == L.N_COINS
    assert coins[-1] == (30.0 + L.RIDER_COIN_OFFSET[0], L.PLAT_TOP + L.RIDER_COIN_OFFSET[1])
    assert coins[0] == (float(L.COIN_X[0]), float(L.COIN_Y[0]))
    assert V.coin_positions(31.0)[-1][0] == coins[-1][0] + 1.0 and V.coin_positions(31.0)[0] == coins[0]


def test_text_helpers():
    assert V.score_text(0) == "You collected 0 coins." and V.score_text(1) == "You collected 1 coin."
    lines = V.wrap_words("You collected 9 coins.", 102, lambda s: 6 * len(s))
    assert lines == ["You collected 9", "coins."]
    assert all(6 * len(s) <= 102 for s in lines)
    assert V.wrap_words("", 50, len) == []
    assert V.wrap_words("averyveryverylongword", 5, len) == ["averyveryverylongword"]


def test_lerp_pos_smooths_but_not_across_teleports():
    assert V.lerp_pos(1.0, 2.0, 0.25) == 1.25
    assert V.lerp_pos(1.0, 2.0, 5.0) == 2.0 and V.lerp_pos(1.0, 2.0, -1.0) == 1.0
    assert V.lerp_pos(60.0, 17.5, 0.3) == 17.5           # reset: jump straight to the new position
