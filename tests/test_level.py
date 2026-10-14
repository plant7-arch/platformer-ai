"""Level validation: what is an error, what is only a warning, and that the messages say where."""
import logging
import os
import random
import tempfile

import numpy as np
import pytest

from platformer_ai import level as L
from platformer_ai import levelcheck as K

BASE = L.LEVEL
ROWS = BASE.strip("\n").split("\n")


def edit(*changes, base=ROWS):
    """changes: (row, col, char), 0-based."""
    r = [list(x) for x in base]
    for row, col, ch in changes:
        r[row][col] = ch
    return "\n".join("".join(x) for x in r) + "\n"


def codes(problems):
    return {p.code for p in problems}


def test_the_built_in_level_is_clean():
    rep = K.check(BASE, L.PLATFORM_TRAVEL, L.ENEMY_RAY_LEN, L.ENEMY_RAY_H, L.ENEMY_HALF_W)
    assert rep.ok and rep.errors == [] and rep.warnings == []


def test_refactor_kept_the_level_numbers():
    assert (L.H, L.W, L.N_COINS, L.N_STARTS) == (10, 70, 9, 8)
    assert (L.ENEMY_MIN, L.ENEMY_MAX, L.PLAT_X_MIN, L.PLAT_X_MAX) == (44.5625, 52.4375, 28.0, 34.0)
    assert (L.SPAWN_X, L.SPAWN_Y, L.PLAT_TOP) == (17.5, 4.0, 7.0)


ERRORS = [
    ("unknown symbol", edit((0, 5, "x")), {"unknown-symbol"}),
    ("lowercase p", edit((1, 9, "p")), {"unknown-symbol"}),
    ("tab / space", edit((0, 0, " ")), {"unknown-symbol"}),
    ("two spawns", edit((0, 3, "P")), {"spawn-count"}),
    ("no spawn", BASE.replace("P", "."), {"spawn-count"}),
    ("two enemies", edit((0, 3, "E")), {"enemy-count"}),
    ("no enemy", BASE.replace("E", "."), {"enemy-count"}),
    ("no platform", BASE.replace("M", "#"), {"platform-missing"}),
    ("platform in two rows", edit((8, 30, "M")), {"platform-rows"}),
    ("platform with a gap", edit((7, 27, "M"), (7, 28, "."), (7, 29, "M")), {"platform-gap"}),
    ("platform sweeps through rock", edit((7, 31, "#")), {"platform-sweep-solid"}),
    ("platform sweeps out of the map", "P...E..G\n........\n....M...\n########\n", {"platform-sweep-map"}),
    ("ragged rows", BASE.replace("\n", ".\n", 1), {"ragged-rows"}),
    ("empty", "", {"empty-level"}),
    ("only newlines", "\n\n", {"empty-level"}),
]


@pytest.mark.parametrize("name,text,expected", ERRORS)
def test_errors(name, text, expected):
    rep = K.check(text)
    assert not rep.ok
    assert expected <= codes(rep.errors), (name, [str(e) for e in rep.errors])


@pytest.mark.parametrize("name,text,expected", ERRORS)
def test_parse_raises_a_level_error_for_every_error_case(name, text, expected):
    with pytest.raises(K.LevelError):
        L.parse(text)


WARNINGS = [
    ("no goal", BASE.replace("G", "."), {"no-goal"}),
    ("spawn in the air", edit((3, 17, "."), (0, 40, "P")), {"no-ground-below"}),
    ("start over a pit", edit((4, 31, "S")), {"no-ground-below"}),
    ("enemy floating", edit((7, 48, ".")), {"enemy-floating", "enemy-patrol-ground"}),
    ("enemy boxed in", edit((6, 47, "#"), (6, 49, "#")), {"enemy-boxed-in"}),
    ("walled-in coin", edit((3, 30, "o"), *[(2, c, "#") for c in (29, 30, 31)], *[(4, c, "#") for c in (29, 30, 31)],
                            (3, 29, "#"), (3, 31, "#")), {"walled-in"}),
]


@pytest.mark.parametrize("name,text,expected", WARNINGS)
def test_warnings_do_not_stop_the_level(name, text, expected):
    rep = K.check(text, L.PLATFORM_TRAVEL, L.ENEMY_RAY_LEN, L.ENEMY_RAY_H, L.ENEMY_HALF_W)
    assert rep.ok, [str(e) for e in rep.errors]
    assert expected <= codes(rep.warnings), (name, [str(w) for w in rep.warnings])
    assert L.parse(text)                                  # and it loads


def test_all_errors_are_reported_at_once_with_row_and_column():
    text = edit((0, 5, "x"), (0, 3, "P"))                 # a typo and a second spawn
    with pytest.raises(K.LevelError) as e:
        K.validate(text)
    msg = str(e.value)
    assert isinstance(e.value, ValueError)
    assert "2 problems" in msg and "unknown-symbol" in msg and "spawn-count" in msg
    assert "row 1, col 6" in msg                           # 1-based, like an editor
    assert "np.str_" not in msg


def test_hint_for_a_lowercase_symbol():
    rep = K.check(edit((1, 9, "p")))
    assert "did you mean 'P'" in rep.errors[0].message and "row 2, col 10" in rep.errors[0].message


def test_warnings_are_logged_not_raised():
    seen = []

    class H(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    h = H()
    logging.getLogger("platformer_ai.level").addHandler(h)
    try:
        L.parse(BASE.replace("G", "."))                    # no goal: loads, but says so
        L.parse(BASE)                                      # the built-in level: silent
    finally:
        logging.getLogger("platformer_ai.level").removeHandler(h)
    assert len(seen) == 1 and "no-goal" in seen[0]


def test_walled_in_means_enclosed_not_just_hard_to_reach():
    solid = np.zeros((6, 8), bool)
    solid[4:, :] = True                                    # floor; the start is on it at row 3
    solid[1:4, 5] = True                                   # a wall that stops below the top of the map...
    assert K._reachable(solid, (3, 1))[3, 7]               # ...can be jumped over: reachable
    solid[0, 5] = True                                     # a wall from the top of the map to the floor
    assert K._reachable(solid, (3, 1))[3, 7]               # still counted as reachable (the sky above the map is open)
    pocket = np.zeros((6, 8), bool)
    pocket[2:5, 5:8] = True
    pocket[3, 6] = False                                   # a one-cell room inside solid rock
    reach = K._reachable(pocket, (0, 0))
    assert reach[0, 0] and not reach[3, 6]


def test_check_never_raises_on_garbage():
    rnd = random.Random(0)
    alphabet = ".#-oGPESM xp\t0=_q"
    for _ in range(300):
        h, w = rnd.randint(0, 6), rnd.randint(0, 9)
        rows = ["".join(rnd.choice(alphabet) for _ in range(w if rnd.random() < 0.9 else w + 1)) for _ in range(h)]
        rep = K.check("\n".join(rows))
        assert isinstance(rep, K.Report)
        for p in rep.errors + rep.warnings:
            assert str(p)


def test_command_line_checker():
    bad = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    good = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        bad.write(edit((0, 5, "x")))
        good.write(BASE)
        bad.close()
        good.close()
        assert K.main([bad.name]) == 1
        assert K.main([good.name]) == 0
        assert K.main([]) == 0                             # the level in level.py
    finally:
        os.unlink(bad.name)
        os.unlink(good.name)
