"""Terrain art generated from the ASCII level: pure numpy, no assets needed."""
import numpy as np
import pytest

from platformer_ai import level as L
from platformer_ai import terrain as T


def grid(*rows):
    return np.array([[ch == "#" for ch in r] for r in rows])


def test_one_tile_per_solid_cell_at_the_right_place():
    solid = grid("..#.", "####", "####")
    t = T.terrain_tiles(solid, origin=(10, 5))
    assert len(t) == int(solid.sum())
    cells = {(int(x) - 10, int(y) - 5) for x, y, *_ in t}
    assert cells == {(c, r) for r, c in np.argwhere(solid)}


def test_top_and_body_tiles():
    solid = grid("..#.", "####", "####")
    top = set(T.THEMES["grass"]["top"])
    body = set(T.THEMES["grass"]["body"])
    for x, y, ax, ay in T.terrain_tiles(solid).tolist():
        above_solid = y > 0 and solid[y - 1, x]
        assert (ax, ay) in (body if above_solid else top), (x, y)


def test_the_art_is_deterministic_and_uses_variants():
    t1, t2 = T.terrain_tiles(L.SOLID, (10, 5)), T.terrain_tiles(L.SOLID, (10, 5))
    assert np.array_equal(t1, t2)
    bodies = {(ax, ay) for x, y, ax, ay in t1.tolist() if (ax, ay) in set(T.THEMES["grass"]["body"])}
    assert len(bodies) > 1


def test_every_theme_uses_tiles_inside_the_tileset():
    for name, th in T.THEMES.items():
        for ax, ay in th["top"] + th["body"] + T.DECOR:
            assert 0 <= ax < 16 and 0 <= ay < 16, name
    for name in T.THEMES:
        assert len(T.terrain_tiles(L.SOLID, theme=name)) == int(L.SOLID.sum())


def test_decorations_stand_on_ground_and_respect_clear_cells():
    solid = np.zeros((6, 200), bool)
    solid[4:] = True                                   # flat ground: row 4 is the surface
    d = T.decor_tiles(solid)
    assert 0 < len(d) < 200
    assert all(y == 3 for _, y, *_ in d.tolist())      # in the row above the surface
    clear = np.zeros_like(solid)
    clear[3, :100] = True
    d2 = T.decor_tiles(solid, clear)
    assert len(d2) and all(x >= 100 for x, *_ in d2.tolist())
    assert len(T.decor_tiles(solid[:1] & False)) == 0   # no ground, no decorations


def test_backdrop_rows_match_the_original_stripes():
    assert [T.backdrop_row(y) for y in (-20, -9, 2, 3, 4, 8, 9, 10, 15, 16, 17, 27, 99)] == \
        [9, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 15]
    b = T.background_tiles(-3, 4, -2, 5)
    assert b.shape == (7 * 7, 4) and set(b[:, 2].tolist()) == {0}
    assert len({tuple(r[:2]) for r in b.tolist()}) == 49            # every cell exactly once


def test_goal_runs_and_flag():
    g = np.zeros((6, 8), bool)
    g[1:3, 5] = True
    g[4, 2] = True
    assert T.goal_runs(g) == [(2, 4, 1), (5, 1, 2)]
    assert T.goal_runs(np.zeros((3, 3), bool)) == []
    for n in (1, 2, 3):
        f = T.flag_sprite(n)
        assert f.shape == (16 * n, 16, 4) and f.dtype == np.uint8
        assert f[..., 3].max() == 255 and f[..., 3].min() == 0      # drawn, with transparent surroundings
        assert f[-1, :, 3].any() and not f[0, :, 3].any()           # stands on the bottom edge, nothing above the pole
    assert T.goal_runs(L.GOAL) == [(69, 2, 2)]


@pytest.mark.parametrize("h,w", [(1, 1), (1, 9), (3, 1), (2, 2)])
def test_terrain_functions_accept_tiny_and_empty_grids(h, w):
    empty = np.zeros((h, w), bool)
    assert T.terrain_tiles(empty, (10, 5)).shape == (0, 4)
    assert T.decor_tiles(empty, None, (10, 5)).shape == (0, 4)
    assert T.top_mask(empty).shape == (h, w) and not T.top_mask(empty).any()
    assert T.goal_runs(empty) == []
    full = np.ones((h, w), bool)
    tiles = T.terrain_tiles(full, (10, 5))
    assert len(tiles) == h * w
    assert len(T.decor_tiles(full)) == 0                 # nothing can stand on the top row of a full map
    assert T.background_tiles(0, 1, 0, 1).shape == (1, 4)
