"""Level validation: what is an error, what is only a warning, and that the messages say where."""
import logging
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from platformer_ai import level as L
from platformer_ai import levelcheck as K

BASE = L.LEVEL
ROWS, _ = K.split_text(BASE)                                  # the 10 map rows
PATHS = "platform 8 28 to 34\nenemy 7 49 from 44.5625 to 52.4375\n"      # the path lines of the built-in level


def edit(*changes, base=ROWS, paths=PATHS):
    """The built-in level with changes: (row, col, char), 0-based; `paths` = the lines under the map."""
    r = [list(x) for x in base]
    for row, col, ch in changes:
        r[row][col] = ch
    return "\n".join("".join(x) for x in r) + "\n" + ("\n" + paths if paths else "")


def mini(*rows, paths=""):
    """A level from map rows (shorter rows are padded: ground rows with '#', others with '.') and path lines."""
    w = max(len(r) for r in rows)
    rows = [r + ("#" if set(r) == {"#"} else ".") * (w - len(r)) for r in rows]
    return "\n".join(rows) + "\n" + ("\n" + paths + "\n" if paths else "")


def check(text):
    return K.check(text, L.DEFAULT_PLATFORM_TRAVEL, L.DEFAULT_ENEMY_REACH, L.ENEMY_HALF_W, L.ENEMY_H)


def codes(problems):
    return {p.code for p in problems}


def test_the_built_in_level_is_clean():
    rep = check(BASE)
    assert rep.ok and rep.errors == [] and rep.warnings == []


def test_refactor_kept_the_level_numbers():
    assert (L.H, L.W, L.N_COINS, L.N_STARTS) == (10, 70, 9, 8)
    assert (L.N_ENEMIES, L.N_PLATS) == (1, 1)
    assert (L.ENEMY_MIN[0], L.ENEMY_MAX[0], L.PLAT_X_MIN[0], L.PLAT_X_MAX[0]) == (44.5625, 52.4375, 28.0, 34.0)
    assert (L.SPAWN_X, L.SPAWN_Y, L.PLAT_TOP[0]) == (17.5, 4.0, 7.0)


ERRORS = [
    ("unknown symbol", edit((0, 5, "x")), {"unknown-symbol"}),
    ("lowercase p", edit((1, 9, "p")), {"unknown-symbol"}),
    ("tab / space", edit((0, 0, " ")), {"unknown-symbol"}),
    ("two spawns", edit((0, 3, "P")), {"spawn-count"}),
    ("no spawn", BASE.replace("P", "."), {"spawn-count"}),
    ("platform sweeps through rock", edit((7, 31, "#")), {"platform-sweep-solid"}),
    ("second platform sweeps through rock", edit((3, 3, "M"), (3, 8, "#")), {"platform-sweep-solid"}),
    # --- platform paths
    ("platform stops beyond the right edge", edit(paths="platform 8 28 to 80\nenemy 7 49 from 44.5625 to 52.4375"),
     {"platform-sweep-map"}),
    ("platform stops beyond the left edge", edit(paths="platform 8 28 to 0.5\nenemy 7 49 from 44.5625 to 52.4375"),
     {"platform-sweep-map"}),
    ("platform stop inside rock", edit(paths="platform 8 28 to 40\nenemy 7 49 from 44.5625 to 52.4375"),
     {"platform-sweep-solid"}),
    ("platform that does not move", edit(paths="platform 8 28 to 28\nenemy 7 49 from 44.5625 to 52.4375"),
     {"platform-no-travel"}),
    ("platform sweeps a static tile", edit((7, 31, "-")), {"platform-sweep-static"}),
    ("platform sweeps a coin", edit((7, 31, "o")), {"platform-sweep-coin"}),
    ("platform sweeps the goal", edit((7, 31, "G")), {"platform-sweep-goal"}),
    ("platform sweeps the spawn", edit((3, 17, "."), (7, 31, "P")), {"platform-sweep-spawn"}),
    ("platform sweeps a start point", edit((7, 31, "S")), {"platform-sweep-start"}),
    # --- enemy paths
    ("enemy path into the wall", edit(paths="platform 8 28 to 34\nenemy 7 49 from 40 to 52.4375"), {"enemy-path-solid"}),
    ("enemy path off the right edge", edit(paths="platform 8 28 to 34\nenemy 7 49 from 44.5625 to 70"),
     {"enemy-path-outside"}),
    ("enemy path off the left edge", edit(paths="platform 8 28 to 34\nenemy 7 49 from -1 to 52"), {"enemy-path-outside"}),
    ("enemy starts outside its path", edit(paths="platform 8 28 to 34\nenemy 7 49 from 49 to 52"),
     {"enemy-start-outside"}),
    ("enemy with no room to walk", edit(paths="platform 8 28 to 34\nenemy 7 49 from 48.5 to 48.55"),
     {"enemy-no-travel"}),
    ("enemy path with a hole in the ground", edit((7, 46, ".")), {"enemy-path-ground"}),
    ("enemy walks into a coin", edit((6, 46, "o")), {"enemy-path-coin"}),
    ("enemy walks into the goal", edit((6, 46, "G")), {"enemy-path-goal"}),
    ("enemy walks into a static tile", edit((6, 46, "-")), {"enemy-path-static"}),
    ("enemy walks into a block", edit((6, 46, "#")), {"enemy-path-solid"}),
    ("enemy walks over the spawn", edit((3, 17, "."), (6, 50, "P")), {"enemy-hits-spawn"}),
    # --- path lines themselves
    ("path line for a missing platform", edit(paths=PATHS + "platform 1 1 to 5"), {"path-target"}),
    ("path line for a missing enemy", edit(paths=PATHS + "enemy 1 1 from 1 to 5"), {"path-target"}),
    ("two lines for one platform", edit(paths=PATHS + "platform 8 29 to 33"), {"path-duplicate"}),
    ("from larger than to", edit(paths="platform 8 28 to 34\nenemy 7 49 from 52 to 44"), {"path-order"}),
    ("garbage path line", edit(paths=PATHS + "platform 8 28 until 34"), {"path-syntax"}),
    ("map row after a blank line", edit(paths=PATHS + "....P....."), {"path-syntax"}),
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
    ("walled-in coin", edit((3, 30, "o"), *[(2, c, "#") for c in (29, 30, 31)], *[(4, c, "#") for c in (29, 30, 31)],
                            (3, 29, "#"), (3, 31, "#")), {"walled-in"}),
]


@pytest.mark.parametrize("name,text,expected", WARNINGS)
def test_warnings_do_not_stop_the_level(name, text, expected):
    rep = check(text)
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


def _python(code, level_text):
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        f.write(level_text)
        f.close()
        env = {**os.environ, "PLATFORMER_LEVEL": f.name, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    finally:
        os.unlink(f.name)


def test_level_file_override():
    r = _python("from platformer_ai import level as L; print(L.H, L.W, L.N_COINS)", "P....E.....MM.......\n####################\n")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["2", "20", "1"]            # the file replaced the built-in level


def test_level_file_override_is_validated_too():
    r = _python("from platformer_ai import level", "PE.MM.......x\n")
    assert r.returncode != 0 and "unknown-symbol" in r.stderr and "row 1, col 13" in r.stderr


# ---- several enemies and platforms (or none) ------------------------------------------------------------------------
FLOOR = "#" * 24


def test_any_number_of_enemies_and_platforms_is_fine():
    for name, text, n_e, n_m in [
        ("no enemy", edit((6, 48, "."), paths="platform 8 28 to 34"), 0, 1),
        ("no platform", edit((7, 27, "#"), (7, 28, "#"), paths="enemy 7 49 from 44.5625 to 52.4375"), 1, 0),
        ("neither", edit((6, 48, "."), (7, 27, "#"), (7, 28, "#"), paths=""), 0, 0),
        ("two enemies", edit((6, 41, "E"), (6, 42, "."), (7, 40, "#"), paths=PATHS + "enemy 7 42 from 41.5 to 42.5"), 2, 1),
        ("a platform in a second row", edit((4, 8, "M"), (4, 9, "M"), paths=PATHS + "platform 5 9 to 16"), 1, 2),
    ]:
        rep = check(text)
        assert rep.ok, (name, [str(e) for e in rep.errors])
        p = L.parse(text)
        assert (len(p["enemies"]), len(p["platforms"])) == (n_e, n_m), name


def test_parse_numbers_enemies_and_platforms_row_by_row():
    text = mini("P.......................", ".MM.....MMM.......M.....", "......E.....E...........", FLOOR,
                paths="enemy 3 13 from 11 to 14\nenemy 3 7 from 5 to 9\n# a comment\nplatform 2 2 to 5\n"
                      "platform 2 9 to 12\nplatform 2 19 to 22.5")
    p = L.parse(text)
    assert p["enemies"] == [(6.5, 3.0, 5.0, 9.0), (12.5, 3.0, 11.0, 14.0)]    # (x, feet y, patrol min x, max x)
    # "MMM" and "M" are separate platforms.  (top y, start x, stop x, width)
    assert p["platforms"] == [(1.0, 2.0, 5.0, 2.0), (1.0, 9.5, 12.0, 3.0), (1.0, 18.5, 22.5, 1.0)]


def test_paths_without_path_lines_use_the_defaults():
    text = mini("P........................", "........E....MM...........", "#" * 25)
    p = L.parse(text)
    assert p["platforms"] == [(1.0, 14.0, 14.0 + L.DEFAULT_PLATFORM_TRAVEL, 2.0)]
    assert p["enemies"] == [(8.5, 2.0, 8.5 - L.DEFAULT_ENEMY_REACH, 8.5 + L.DEFAULT_ENEMY_REACH)]


def test_a_platform_may_start_at_its_right_end():
    text = mini("P.......................", "..................MM....", FLOOR, paths="platform 2 19 to 10")
    p = L.parse(text)
    assert p["platforms"] == [(1.0, 19.0, 10.0, 2.0)]


def test_platform_errors_name_the_platform():
    rep = check(mini("P.......", "........", ".M..MM..", "########", paths="platform 3 2 to 3\nplatform 3 5 to 9"))
    assert not rep.ok
    msgs = " ".join(e.message for e in rep.errors)
    assert "platform-sweep-map" in codes(rep.errors) and "cols 5..6" in msgs and "cols 2..2" not in msgs


def test_platforms_on_the_same_stretch_are_an_error_only_in_the_same_row():
    two_rows = mini("P.......................", ".MM.....................", "........................",
                    ".MM.....................", FLOOR, paths="platform 2 2 to 8\nplatform 4 2 to 8")
    one_row = mini("P.......................", ".MM.MM..................", FLOOR, paths="platform 2 2 to 8\nplatform 2 5 to 8")
    assert check(two_rows).ok
    assert "platform-overlap" in codes(check(one_row).errors)
    next_to = mini("P.......................", ".MM.......MM...........", FLOOR, paths="platform 2 2 to 6\nplatform 2 11 to 15")
    assert check(next_to).ok                               # separate stretches in one row are fine


def test_platform_and_enemy_in_one_row_may_not_share_a_stretch():
    row = ["P.........E.MM......", "####################"]
    clash = mini(*row, paths="enemy 1 11 from 8 to 12\nplatform 1 13 to 17")
    apart = mini(*row, paths="enemy 1 11 from 8 to 10.5\nplatform 1 13 to 17")
    assert "platform-sweep-enemy" in codes(check(clash).errors)
    assert "platform-sweep-enemy" not in codes(check(apart).errors)


def test_enemies_on_the_same_stretch_clash():
    rows = ("P...................", "......E.....E.......", "####################")
    assert "enemy-overlap" in codes(check(mini(*rows, paths="enemy 2 7 from 5 to 10\nenemy 2 13 from 9 to 14")).errors)
    assert check(mini(*rows, paths="enemy 2 7 from 5 to 9\nenemy 2 13 from 11 to 14")).ok


def test_enemy_path_ignores_walls_and_is_not_set_by_them():
    # a wall does not shorten the path: the path is what the line says (and a wall inside it is an error)
    text = mini("P.........................", "...........E#..............", "#" * 27, paths="enemy 2 12 from 8 to 12.5")
    assert "enemy-path-solid" in codes(check(text).errors)
    ok = mini("P.........................", "...........E.#............", "#" * 27, paths="enemy 2 12 from 9 to 12.5")
    assert check(ok).ok
    assert L.parse(ok)["enemies"] == [(11.5, 2.0, 9.0, 12.5)]


def test_an_enemy_close_to_the_real_spawn():
    ground = "#" * 27
    far = mini("P.........................", "..........E...............", ground, paths="enemy 2 11 from 8 to 14")
    assert "enemy-near-spawn" not in codes(check(far).warnings) and check(far).ok
    near = mini("P.........................", "....E.....................", ground,
                paths="enemy 2 5 from 3.5 to 5.5")                       # its body comes within 2.6 tiles of the spawn
    assert "enemy-near-spawn" in codes(check(near).warnings) and check(near).ok
    close = mini("P....E....................", ground, paths="enemy 1 6 from 0.5 to 8")   # walks over the spawn
    assert "enemy-hits-spawn" in codes(check(close).errors)
    # the same enemy one row above the spawn cannot touch the player at the start
    above = mini(".....E.....................", "P.........................", ground, paths="enemy 1 6 from 0.5 to 8")
    assert "enemy-hits-spawn" not in codes(check(above).errors)


def test_path_line_messages_say_what_is_wrong():
    rep = check(edit(paths=PATHS + "platform 8 28 until 34\n....P....."))
    msgs = [e.message for e in rep.errors]
    assert any("line 14" in m and "platform ROW COL to STOP_X" in m for m in msgs)           # the allowed forms are listed
    assert any("looks like a map row" in m for m in msgs)
    rep = check(edit(paths="platform 8 28 to 34\nenemy 7 49 from 52 to 44"))
    assert "line 13" in rep.errors[0].message


def test_path_lines_may_have_comments_and_any_case():
    text = edit(paths="# the ferry\nPlatform  8   28  TO 34\n\nENEMY 7 49 FROM 44.5625 TO 52.4375\n")
    assert check(text).ok and L.parse(text)["platforms"][0][2] == 34.0


# ---- short messages for a level that is not valid ------------------------------------------------------------------------
def test_an_invalid_level_file_prints_a_short_message_not_a_traceback():
    bad = mini("P....E.....", "#####x#####", paths="enemy 1 6 from 4 to 9")
    r = _python("from platformer_ai import level", bad)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr and "File \"" not in r.stderr
    assert "unknown-symbol" in r.stderr and "enemy-path-ground" in r.stderr          # every problem, listed
    assert ".txt" in r.stderr and "levelcheck" in r.stderr                            # which file, and how to check it


def test_a_missing_level_file_is_a_short_message_too():
    env = {**os.environ, "PLATFORMER_LEVEL": "/nonexistent/level.txt", "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    r = subprocess.run([sys.executable, "-c", "from platformer_ai import level"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "Traceback" not in r.stderr and "/nonexistent/level.txt" in r.stderr


def test_warnings_name_the_level_file():
    r = _python("from platformer_ai import level", mini("P....", "#####"))            # no goal: a warning
    assert r.returncode == 0 and "no-goal" in r.stderr and ".txt" in r.stderr


def test_the_checker_works_on_a_file_without_a_valid_level_py():
    bad = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        bad.write(mini("P....E.....", "#####x#####"))
        bad.close()
        assert K.main([bad.name]) == 1
    finally:
        os.unlink(bad.name)
