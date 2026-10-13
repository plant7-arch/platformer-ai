"""The numpy renderer.  Needs Pillow; the picture tests also need the Brackeys assets in ./assets."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from platformer_ai import level as L
from platformer_ai import scene
from platformer_ai import viewlogic as V

ASSETS = Path(__file__).resolve().parents[1] / "assets"
needs_assets = pytest.mark.skipif(not (ASSETS / "sprites" / "knight.png").is_file(), reason="assets/ not present")


def test_blit_alpha_and_clipping():
    dst = np.full((4, 4, 3), 100, np.uint8)
    src = np.zeros((2, 2, 4), np.uint8)
    src[0, 0] = (200, 0, 0, 255)                        # opaque
    src[0, 1] = (200, 0, 0, 0)                          # transparent: leaves dst alone
    src[1, 0] = (200, 0, 0, 128)                        # half: mixes
    scene.blit(dst, src, 1, 1)
    assert tuple(dst[1, 1]) == (200, 0, 0) and tuple(dst[1, 2]) == (100, 100, 100)
    assert 148 <= dst[2, 1, 0] <= 150
    scene.blit(dst, src, -1, -1)                        # partly outside: only the corner pixel lands
    assert tuple(dst[0, 0]) == (100, 100, 100)          # src[1, 1] is transparent (zeros)
    scene.blit(dst, src, 50, 50)                        # fully outside: no error
    flipped = np.full((2, 2, 3), 0, np.uint8)
    scene.blit(flipped, src, 0, 0, flip=True)           # opaque pixel moves to the right column
    assert tuple(flipped[0, 1]) == (200, 0, 0) and tuple(flipped[0, 0]) == (0, 0, 0)


def test_crop_outside_the_world_is_clear_colour():
    world = np.zeros((10, 10, 3), np.uint8)
    out = scene.crop(world, -3, -3, 8, 8)
    assert out.shape == (8, 8, 3)
    assert tuple(out[0, 0]) == scene.CLEAR and tuple(out[3, 3]) == (0, 0, 0)
    assert tuple(scene.crop(world, 100, 100, 4, 4)[0, 0]) == scene.CLEAR


@needs_assets
def test_frame_shows_what_it_should():
    a = scene.load_assets(ASSETS)
    cam = V.Camera()
    cam.snap(*V.to_px(L.SPAWN_X, L.SPAWN_Y))

    def frame(**kw):
        args = dict(player_xy=(L.SPAWN_X, L.SPAWN_Y), player_frame=V.ANIMS["player"]["idle"]["frames"][0],
                    player_flip=False, enemy_x=L.ENEMY_X, enemy_flip=False, enemy_frame=V.ANIMS["enemy"]["walk"]["frames"][0],
                    coin_frame=V.ANIMS["coin"]["spin"]["frames"][0], coins_alive=[True] * L.N_COINS, plat_x=L.PLAT_X_MIN)
        args.update(kw)
        return scene.compose(a, cam.top_left(), **args)

    base = frame()
    assert base.shape == (V.SCREEN_H, V.SCREEN_W, 3) and base.dtype == np.uint8
    assert not np.array_equal(base, frame(coins_alive=[False] * L.N_COINS))            # coins are drawn
    assert not np.array_equal(base, frame(player_flip=True))                           # knight flips
    assert not np.array_equal(base, frame(player_xy=(L.SPAWN_X + 3, L.SPAWN_Y)))       # knight moves
    # the slime is far off-screen at spawn; put it next to the knight and the picture changes
    assert np.array_equal(base, frame(enemy_x=L.ENEMY_X + 1))
    assert not np.array_equal(base, frame(enemy_x=L.SPAWN_X + 2))
    assert not np.array_equal(base, frame(plat_x=L.SPAWN_X - 2))                       # moving platform drawn at plat_x


@needs_assets
def test_world_covers_the_whole_level():
    a = scene.load_assets(ASSETS)
    x0, y0 = V.to_px(0, 0)
    x1, y1 = V.to_px(L.W, L.H)
    ox, oy = a.world_origin
    assert ox <= x0 and oy <= y0 and ox + a.world.shape[1] >= x1 and oy + a.world.shape[0] >= y1
