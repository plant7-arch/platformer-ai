"""Live reload of the policy (viewer / play follow a run while it trains).  Plain Python, no jax: `load` is faked."""
import json
import os

import pytest

from platformer_ai import runs
from platformer_ai.policywatch import PolicyWatch, describe


def load(data: bytes):
    """Stands in for restore_params: a 'policy' is any bytes except b'broken'."""
    if data == b"broken":
        raise ValueError("not a policy file")
    return data


def save(run, data: bytes, mtime=None):
    run.mkdir(parents=True, exist_ok=True)
    runs.write_atomic(run / runs.PARAMS, data)                      # what train.py does
    if mtime is not None:
        os.utime(run / runs.PARAMS, (mtime, mtime))


def test_reloads_when_the_run_saves_again(tmp_path):
    run = tmp_path / "2026-10-23_120000"
    save(run, b"v1", 1000)
    w = PolicyWatch("latest", load, root=tmp_path)
    assert (w.params, w.loads) == (b"v1", 1)
    assert w.poll() is None and w.loads == 1                        # nothing new: nothing loaded
    save(run, b"v2", 2000)
    assert "policy loaded: 2026-10-23_120000" in w.poll()
    assert (w.params, w.loads) == (b"v2", 2)


def test_latest_moves_to_a_newer_run(tmp_path):
    save(tmp_path / "old", b"old", 1000)
    w = PolicyWatch(None, load, root=tmp_path)
    save(tmp_path / "new", b"new", 2000)                             # a run started after the viewer opened
    w.poll()
    assert w.params == b"new" and w.path == tmp_path / "new" / runs.PARAMS


def test_a_named_run_is_followed_but_latest_is_not(tmp_path):
    save(tmp_path / "mine", b"a", 1000)
    w = PolicyWatch("mine", load, root=tmp_path)
    save(tmp_path / "other", b"b", 2000)                             # newer, but not the run that was asked for
    assert w.poll() is None and w.params == b"a"


def test_a_broken_policy_keeps_the_current_one_and_is_not_retried(tmp_path):
    run = tmp_path / "r"
    save(run, b"good", 1000)
    w = PolicyWatch("latest", load, root=tmp_path)
    save(run, b"broken", 2000)
    msg = w.poll()
    assert "cannot load" in msg and "keeping the current policy" in msg and w.params == b"good"
    assert w.poll() is None                                          # the same broken file: not every episode again
    save(run, b"fixed", 3000)
    w.poll()
    assert w.params == b"fixed"


def test_the_first_load_raises_for_the_caller():
    """The viewer / play turn this into 'does not fit a network of width ... (try --hidden)'."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        save(Path(d) / "r", b"broken")
        with pytest.raises(ValueError):
            PolicyWatch("latest", load, root=Path(d))


def test_no_policy_yet_then_one_appears(tmp_path):
    w = PolicyWatch("latest", load, root=tmp_path)
    assert w.params is None and w.label == ""
    assert w.poll() is None
    save(tmp_path / "first", b"p", 1000)                             # training saved its first policy
    w.poll()
    assert w.params == b"p"


def test_follow_off_and_force(tmp_path):
    run = tmp_path / "r"
    save(run, b"v1", 1000)
    w = PolicyWatch("latest", load, root=tmp_path, follow=False)
    save(run, b"v2", 2000)
    assert w.poll() is None and w.params == b"v1"                   # --no-follow
    assert w.poll(force=True) and w.params == b"v2"                 # the viewer's L key
    assert w.poll(force=True) and w.loads == 3                      # force loads even an unchanged file


def test_describe_names_run_update_and_eval(tmp_path):
    run = tmp_path / "2026-10-23_131500_t3"
    save(run, b"p")
    assert describe(run / runs.PARAMS) == "2026-10-23_131500_t3"     # no run.json (like runs/pretrained/)
    (run / runs.INFO).write_text(json.dumps(dict(
        updates=75, finished=False, config=dict(num_envs=1024, num_steps=128, total_timesteps=25_000_000),
        last_eval={"greedy": dict(goal=0.6234)})))
    assert describe(run / runs.PARAMS) == "2026-10-23_131500_t3  update 75/190  eval goal 0.62"
    other = tmp_path / "sent_to_me.msgpack"
    other.write_bytes(b"p")
    assert describe(other) == "sent_to_me.msgpack"


def test_saves_never_leave_a_tmp_file(tmp_path):
    runs.write_atomic(tmp_path / "f.bin", b"abc")
    runs.write_atomic(tmp_path / "f.bin", b"abcd")
    runs.write_atomic(tmp_path / "t.json", "{}")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.bin", "t.json"]
    assert (tmp_path / "f.bin").read_bytes() == b"abcd"
