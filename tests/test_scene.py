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


@needs_assets
def test_picture_follows_an_edited_level():
    """A different ASCII grid gives a different world picture, sized to the grid (no stored art)."""
    tileset = scene.load_rgba(ASSETS / "sprites" / "world_tileset.png")

    def world(rows, goal_rows=()):
        solid = np.array([[c == "#" for c in r] for r in rows])
        goal = np.array([[c == "G" for c in r] for r in (goal_rows or rows)])
        return scene.make_world(tileset, solid, goal, decorate=False)

    flat, org = world(["........", "........", "########"])
    wide, _ = world(["................", "................", "################"])
    tower, _ = world(["........", "...#....", "########"])
    assert wide.shape[1] - flat.shape[1] == 8 * V.TILE                  # 8 more columns -> 8 tiles wider
    assert not np.array_equal(flat, tower)                              # moving a block moves the art
    with_flag, _ = world(["........", "........", "########"], ["....G...", "....G...", "........"])
    assert not np.array_equal(flat, with_flag)                          # a goal draws a flag


# ---- tiny / empty / odd grids: the world must cover everything the camera can show ---------------------------------
def fake_tileset():
    """A tileset without assets: every pixel opaque, never the 'nothing here' colour, so a clear-coloured pixel in a
    cropped view can only mean the world did not cover that part."""
    rng = np.random.default_rng(0)
    ts = rng.integers(0, 256, (256, 256, 4)).astype(np.uint8)
    ts[..., 3] = 255
    ts[..., 0] &= 0xFE                                    # red is even, the clear colour's red (77) is odd
    return ts


GRIDS = [(1, 1), (1, 12), (2, 8), (3, 3), (10, 70), (30, 14), (1, 200), (12, 1)]


@pytest.mark.parametrize("h,w", GRIDS)
def test_world_covers_every_camera_position_for_any_grid(h, w):
    rng = np.random.default_rng(h * 1000 + w)
    for solid in (np.zeros((h, w), bool), rng.random((h, w)) < 0.4, np.ones((h, w), bool)):
        goal = np.zeros((h, w), bool)
        goal[0, -1] = True
        world, (ox, oy) = scene.make_world(fake_tileset(), solid, goal, decorate=True)
        assert world.shape[1] == (w + 40) * V.TILE                     # 20 tiles of margin on both sides
        # camera centres the viewer can reach: x over the whole width, y from above the map to the bottom limit
        xs = [V.to_px(0.0, 0.0)[0], V.to_px(w / 2, 0.0)[0], V.to_px(float(w), 0.0)[0]]
        limit = V.cam_limit_bottom(h) - V.SCREEN_H / 2                 # the camera never goes lower than this
        top = min(V.to_px(0.0, -3.0)[1] + V.CAM_OFFSET_Y, limit)
        bottom = min(V.to_px(0.0, h + 1.0)[1] + V.CAM_OFFSET_Y, limit)
        for cx in xs:
            for cy in (top, (top + bottom) / 2, bottom):
                x0, y0 = int(round(cx - V.SCREEN_W / 2)), int(round(cy - V.SCREEN_H / 2))
                assert ox <= x0 and x0 + V.SCREEN_W <= ox + world.shape[1], (h, w, cx)
                assert oy <= y0 and y0 + V.SCREEN_H <= oy + world.shape[0], (h, w, cy)
                view = scene.crop(world, x0 - ox, y0 - oy, V.SCREEN_W, V.SCREEN_H)
                assert not (view == np.array(scene.CLEAR)).all(axis=2).any(), (h, w, cx, cy)


def test_world_height_grows_with_the_grid():
    ts = fake_tileset()
    small, _ = scene.make_world(ts, np.zeros((2, 14), bool), np.zeros((2, 14), bool))
    tall, _ = scene.make_world(ts, np.zeros((30, 14), bool), np.zeros((30, 14), bool))
    assert tall.shape[0] - small.shape[0] == 28 * V.TILE
    assert V.cam_limit_bottom(10) == 270 and V.cam_limit_bottom(30) == 590
