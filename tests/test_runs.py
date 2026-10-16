"""Run folders under runs/ and how --params finds the latest one (plain Python, no jax)."""
import datetime as dt
import os

from platformer_ai import runs


def make_run(root, name, mtime, info=None):
    d = root / name
    d.mkdir(parents=True)
    (d / runs.PARAMS).write_bytes(b"x")
    os.utime(d / runs.PARAMS, (mtime, mtime))
    if info is not None:
        runs.write_info(d, info)
    return d


def test_new_run_dirs_are_named_by_time_and_never_reused(tmp_path):
    t = dt.datetime(2026, 10, 14, 20, 54, 7)
    a = runs.new_run_dir(root=tmp_path, now=t)
    b = runs.new_run_dir(root=tmp_path, now=t)                       # same second: gets a suffix
    c = runs.new_run_dir("three_levels", root=tmp_path, now=t)
    assert [p.name for p in (a, b, c)] == ["2026-10-14_205407", "2026-10-14_205407-2", "2026-10-14_205407_three_levels"]
    assert all(p.is_dir() for p in (a, b, c))


def test_latest_is_the_most_recently_saved_policy(tmp_path):
    assert runs.latest(tmp_path) is None and runs.latest(tmp_path / "missing") is None
    make_run(tmp_path, "2026-10-14_100000", 1000)
    newest = make_run(tmp_path, "older_name_but_saved_last", 3000)
    make_run(tmp_path, "2026-10-14_200000", 2000)
    (tmp_path / "empty_run").mkdir()                                 # no params yet: not a candidate
    assert runs.latest(tmp_path) == newest
    assert [p.name for p in runs.all_runs(tmp_path)] == ["2026-10-14_100000", "2026-10-14_200000",
                                                           "older_name_but_saved_last"]


def test_pretrained_is_only_the_fallback(tmp_path):
    shipped = make_run(tmp_path, runs.PRETRAINED, 1000)
    assert runs.latest(tmp_path) == shipped                          # nothing else: the shipped policy
    mine = make_run(tmp_path, "2026-10-14_212223", 500)              # saved earlier than the pretrained file
    assert runs.latest(tmp_path) == mine                             # (e.g. a git pull touched pretrained)
    assert runs.resolve_params("pretrained", tmp_path) == shipped / runs.PARAMS
    lines = runs.table(tmp_path)
    assert any(line.startswith("  pretrained") and "shipped with the repository" in line for line in lines)
    assert any(line.startswith("* 2026-10-14_212223") for line in lines)


def test_resolve_params(tmp_path):
    run = make_run(tmp_path, "2026-10-14_205407_mine", 1000)
    make_run(tmp_path, "old", 500)
    assert runs.resolve_params("latest", tmp_path) == run / runs.PARAMS
    assert runs.resolve_params(None, tmp_path) == run / runs.PARAMS
    assert runs.resolve_params(str(tmp_path / "old"), tmp_path) == tmp_path / "old" / runs.PARAMS       # a folder
    assert runs.resolve_params("old", tmp_path) == tmp_path / "old" / runs.PARAMS                       # a run name
    f = tmp_path / "somewhere.msgpack"
    assert runs.resolve_params(str(f), tmp_path) == f                                                   # a file
    assert runs.resolve_params("latest", tmp_path / "none") == tmp_path / "none" / runs.PRETRAINED / runs.PARAMS


def test_table_lists_runs_with_their_info(tmp_path):
    assert "no runs" in runs.table(tmp_path)[0]
    make_run(tmp_path, "plain", 1000)                                 # e.g. a run from before run.json existed
    make_run(tmp_path, "2026-10-14_205407_three", 2000,
             dict(levels=["a.txt", "b.txt", "c.txt"], steps=24903680, finished=True,
                  last_eval={"greedy": dict(goal=0.9634, ret=7.2213)}))
    lines = runs.table(tmp_path)
    assert lines[1].startswith("  plain") and lines[1].rstrip().endswith("-")
    assert lines[2].startswith("* 2026-10-14_205407_three")
    for part in ("24,903,680", " 3 ", "0.96", "7.22", "done"):
        assert part in lines[2], (part, lines[2])
    assert runs.read_info(tmp_path / "plain") == {}


def test_resolve_run(tmp_path):
    a = make_run(tmp_path, "2026-10-14_100000", 1000)
    b = make_run(tmp_path, "2026-10-14_200000_x", 2000)
    assert runs.resolve_run("latest", tmp_path) == b and runs.resolve_run(None, tmp_path) == b
    assert runs.resolve_run("2026-10-14_100000", tmp_path) == a
    assert runs.resolve_run(str(a), tmp_path) == a
    assert runs.resolve_run("nope", tmp_path) is None
    assert runs.resolve_run("latest", tmp_path / "empty") is None


def test_train_stops_and_resumes(tmp_path):
    """Real (tiny) training: 4 updates, then --resume to 8; the optimiser state carries on.  Needs jax (~1 min)."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    from flax import serialization

    level = Path(__file__).resolve().parents[1] / "docs" / "levels" / "01_walk.txt"
    run = tmp_path / "run"
    small = ["--num-envs", "16", "--num-steps", "8", "--eval-every", "0"]

    env = {k: v for k, v in os.environ.items() if k != "PLATFORMER_LEVEL"}

    def train(*extra):
        return subprocess.run([sys.executable, "-m", "platformer_ai.train", *extra], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=600)

    r = train("--total-steps", "512", "--levels", str(level), "--out", str(run), *small)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads((run / runs.INFO).read_text())
    assert (info["updates"], info["finished"]) == (4, True)
    params_before = (run / runs.PARAMS).read_bytes()

    r = train("--resume", str(run))                                    # finished: nothing left to do
    assert r.returncode != 0 and "bigger --total-steps" in r.stdout + r.stderr
    assert "resumed" not in json.loads((run / runs.INFO).read_text())  # a refused resume leaves the run alone

    r = train("--resume", str(run), "--total-steps", "1024", "--eval-every", "0")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "resuming after update 4" in r.stdout and "upd    5/8" in r.stdout and "upd    1/" not in r.stdout
    info = json.loads((run / runs.INFO).read_text())
    assert (info["updates"], info["steps"], info["finished"]) == (8, 1024, True)
    assert len(info["resumed"]) == 1 and info["resumed"][0]["from_update"] == 4
    state = serialization.msgpack_restore((run / runs.STATE).read_bytes())
    assert int(state["train_state"]["step"]) == 8 * 8 * 4             # 8 updates x 8 minibatches x 4 epochs
    assert (run / runs.PARAMS).read_bytes() != params_before

    r = train("--resume", str(run), "--levels", str(level))
    assert r.returncode != 0 and "leave out --levels" in r.stdout + r.stderr


def test_init_from_starts_a_new_run_from_another_policy(tmp_path):
    """--init-from: a new run from the weights of a run that cannot be resumed (like runs/pretrained/).  Needs jax."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    level = Path(__file__).resolve().parents[1] / "docs" / "levels" / "01_walk.txt"
    small = ["--num-envs", "16", "--num-steps", "8", "--eval-every", "0", "--total-steps", "256", "--levels", str(level)]
    env = {k: v for k, v in os.environ.items() if k != "PLATFORMER_LEVEL"}

    def train(*extra):
        return subprocess.run([sys.executable, "-m", "platformer_ai.train", *extra], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=600)

    shipped = tmp_path / "runs" / runs.PRETRAINED
    assert train("--out", str(shipped), *small).returncode == 0
    for f in (runs.STATE, runs.INFO, "levels.txt"):                  # keep only the policy, like the git checkout
        (shipped / f).unlink()

    r = train("--resume", "pretrained")
    assert r.returncode != 0 and "--init-from pretrained" in r.stdout + r.stderr

    r = train("--init-from", "pretrained", "--name", "more", *small)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "network starts from runs/pretrained/params.msgpack" in r.stdout
    new = [d for d in (tmp_path / "runs").iterdir() if d.name.endswith("_more")]
    assert len(new) == 1
    info = json.loads((new[0] / runs.INFO).read_text())
    assert info["init_from"] == "runs/pretrained/params.msgpack" and info["finished"]
    assert (new[0] / runs.STATE).is_file()                           # the new run itself can be resumed later

    r = train("--init-from", "pretrained", "--resume")
    assert r.returncode != 0 and "use one of them" in r.stdout + r.stderr
    r = train("--init-from", "nothing_here", *small)
    assert r.returncode != 0 and "no policy at" in r.stdout + r.stderr


def test_train_refuses_unusable_settings_without_making_a_run_folder(tmp_path, monkeypatch):
    """Too few steps for one update, a batch that does not split into minibatches, and settings given with --resume
    (they used to be ignored without a word) all stop before anything is written."""
    import pytest

    from platformer_ai import train
    monkeypatch.chdir(tmp_path)
    for argv, msg in ((["--total-steps", "1000"], "less than one update"),
                      (["--num-envs", "10", "--num-steps", "3"], "does not split into 8"),
                      (["--resume", "--lr", "1", "--seed", "5"], "leave out --lr, --seed"),
                      (["--resume", "x", "--curriculum", "--no-random-starts"], "leave out --curriculum, --no-random-starts"),
                      (["--resume", "x", "--num-envs", "1024"], "leave out --num-envs")):    # even the default value
        with pytest.raises(SystemExit) as e:
            train.main(argv)
        assert msg in str(e.value), (argv, str(e.value))
    with pytest.raises(SystemExit) as e:                  # the allowed options get as far as looking for the run
        train.main(["--resume", "x", "--total-steps", "5000", "-v", "2", "--log-every", "3", "--eval-every", "0",
                    "--eval-per-level", "2", "--test-eval-per-level", "9"])
    assert "no run 'x'" in str(e.value)
    assert not (tmp_path / "runs").exists()
