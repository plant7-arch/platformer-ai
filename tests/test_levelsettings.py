"""Level generator settings: file and command line say the same things, defaults are the built-in tiers, the settings
change the levels the way they say, and impossible settings are refused with a clear message."""
import random
import tomllib
from pathlib import Path

import pytest

from platformer_ai import levelcheck, levelgen
from platformer_ai import levelsettings as LS

ROOT = Path(__file__).resolve().parents[1]
DRILL = ROOT / "docs" / "levelgen" / "tier3_drill.toml"


def cli(*args):
    return levelgen.settings_from_args(levelgen.build_parser().parse_args(list(args)))


def gen(cfg, n, seed=0):
    rng = random.Random(seed)
    return [levelgen.generate(cfg, rng) for _ in range(n)]


def enemies_and_plats(text):
    _, plats, enemies, _ = levelcheck.layout(text, 6.0, 3.0)
    return enemies, plats


def test_no_settings_is_the_built_in_generator():
    for t in levelgen.TIERS:
        assert LS.resolve({}, "train", t) == LS.resolve({}, "test", t)
        assert gen(t, 3, seed=t) == gen(LS.resolve({}, "train", t), 3, seed=t)
    assert LS.tier_plan(6, {1: 1, 2: 1, 3: 1}) == [1, 2, 3, 1, 2, 3]


def test_settings_file_and_command_line_are_the_same():
    from_file = cli("--config", str(DRILL))
    from_cli = cli("--out", "gen_t3", "--seed", "1", "--train", "600", "--test", "60", "--test-seed", "1000003",
                   "--tiers", "1=1,2=1,3=3", "--set", "tier.3.riders=2", "--set", "tier.3.narrow_stones=0.6",
                   "--set", "tier.3.weight={ferry=1.5,upper=1.5,stones=1.5}", "--set", "test_set.tiers=1,2,3",
                   "--set", "test_set.tier.3.riders=1", "--set", "test_set.tier.3.narrow_stones=0.3",
                   "--set", "test_set.tier.3.weight.ferry=1", "--set", "test_set.tier.3.weight.upper=1",
                   "--set", "test_set.tier.3.weight.stones=1")
    assert LS.dump(from_file) == LS.dump(from_cli)


def test_named_options_equal_set():
    a = cli("--gap", "2-3", "--weight", "enemy=2,gap=0.5", "--without", "stones", "--max-enemies", "4", "--no-verify")
    b = cli("--set", "gap=[2, 3]", "--set", "weight.enemy=2", "--set", "weight.gap=0.5",
            "--set", 'without=["stones"]', "--set", "max_enemies=4", "--set", "verify=false")
    assert a == b


def test_dump_reads_back_to_the_same_settings(tmp_path):
    s = cli("--config", str(DRILL), "--gap", "3", "--min", "ferry=1")
    f = tmp_path / "s.toml"
    f.write_text(LS.dump(s))
    tomllib.loads(f.read_text())                       # valid TOML
    assert LS.dump(LS.load_file(str(f))) == LS.dump(s)


def test_command_line_wins_over_file(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text('gap = [1, 2]\nweight = { enemy = 3 }\n[tier.3]\nriders = 2\n')
    s = cli("--config", str(f), "--gap", "3", "--weight", "gap=2", "--set", "tier.3.riders=0")
    c = LS.resolve(s, "train", 3)
    assert c.gap == (3, 3) and c.riders == 0
    assert c.weights["enemy"] == 3 * 20 and c.weights["gap"] == 2 * 20      # weight tables merge kind by kind


def test_precedence_top_tier_test_set():
    s = cli("--set", "coins=2", "--set", "tier.3.coins=3", "--set", "test_set.coins=4",
            "--set", "test_set.tier.3.coins=5")
    assert LS.resolve(s, "train", 1).coins == 2 and LS.resolve(s, "train", 3).coins == 3
    assert LS.resolve(s, "test", 1).coins == 4 and LS.resolve(s, "test", 3).coins == 5


def test_tier_mix():
    assert LS.tier_plan(5, {3: 1}) == [3] * 5
    plan = LS.tier_plan(10, {1: 1, 3: 4})
    assert plan.count(1) == 2 and plan.count(3) == 8
    assert LS.run_settings(cli("--tiers", "2,3"))["tiers"] == {2: 1, 3: 1}
    assert LS.run_settings(cli("--tiers", "3", "--set", "test_set.tiers=[1]"))["test_tiers"] == {1: 1}


@pytest.mark.parametrize("tier", levelgen.TIERS)
def test_only_without_and_limits_shape_the_levels(tier):
    s = cli("--only", "gap,enemy", "--max-enemies", "2", "--gap", "3")
    cfg = LS.resolve(s, "train", tier)
    for text in gen(cfg, 15, seed=tier):
        enemies, plats = enemies_and_plats(text)
        g = levelcheck.layout(text, 6.0, 3.0)[0]
        assert not plats and not any("-" in r for r in g)            # no ferries, upper platforms, ledges, stones
        assert len(enemies) <= 2
        assert levelcheck.check(text).ok


def test_min_and_riders():
    cfg = LS.resolve(cli("--min", "ferry=2", "--riders", "3", "--tiers", "3"), "train", 3)
    for text in gen(cfg, 10, seed=4):
        enemies, plats = enemies_and_plats(text)
        assert len(plats) >= 2
    rides = sum(e.ride >= 0 for t in gen(cfg, 10, seed=4) for e in enemies_and_plats(t)[0])
    assert rides >= 10
    off = LS.resolve(cli("--riders", "0"), "train", 3)
    assert not any(e.ride >= 0 or e.run for t in gen(off, 15) for e in enemies_and_plats(t)[0])


def test_no_coins():
    cfg = LS.resolve(cli("--coins", "0"), "train", 2)
    assert all("o" not in t.split("\n\n")[0] for t in gen(cfg, 10))


def test_kind_a_tier_lacks_gets_a_base_weight():
    assert LS.resolve(cli("--weight", "stones=1"), "train", 1).weights["stones"] == 10
    assert LS.resolve(cli("--only", "upper,flat"), "train", 1).weights == dict(
        flat=25, step=0, gap=0, enemy=0, ledge=0, ferry=0, stones=0, upper=10)


@pytest.mark.parametrize("args, says", [
    (["--gap", "5"], "outside 1-4"),
    (["--max-step", "3"], "rises about 2.2"),
    (["--weight", "enemie=2"], "did you mean 'enemy'"),
    (["--set", "tier.3.ridres=2"], "did you mean 'riders'"),
    (["--set", "tier.4.gap=2"], "no tier 4"),
    (["--only", "gap", "--min", "ferry=1"], "ferry is switched off"),
    (["--without", "flat,step,gap,enemy,ledge,ferry,stones,upper"], "every segment kind is switched off"),
    (["--width", "60-40"], "goes backwards"),
    (["--pairs", "2"], "at most 1"),
    (["--tiers", "3=0"], "all shares are 0"),
])
def test_bad_settings_are_refused(args, says):
    with pytest.raises(LS.SettingsError, match=says.replace("(", r"\(").replace("'", "'")):
        s = cli(*args)
        for t in levelgen.TIERS:
            LS.resolve(s, "train", t)


def test_main_reports_errors_without_traceback(capsys):
    assert levelgen.main(["--gap", "9", "--print-settings"]) == 2
    assert "gap 9-9 is outside 1-4" in capsys.readouterr().err


def test_settings_saved_with_the_set(tmp_path):
    out = tmp_path / "g"
    assert levelgen.main(["--out", str(out), "--train", "3", "--test", "0", "--no-verify", "--tiers", "3",
                          "--riders", "2"]) == 0
    saved = LS.load_file(str(out / "settings.toml"))
    assert saved["riders"] == 2 and LS.run_settings(saved)["tiers"] == {3: 1}
    assert sorted(p.name for p in (out / "train").iterdir()) == ["t3_0000.txt", "t3_0001.txt", "t3_0002.txt"]
