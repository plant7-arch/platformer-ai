"""Follow a policy while it trains: the viewer and play reload a newer params.msgpack between episodes.

    watch = PolicyWatch("latest", load=lambda data: restore_params(template, data))
    watch.params                  # the policy now (None if there is no policy file yet)
    msg = watch.poll()            # at every episode start: loads a newer policy, returns a message (None: nothing new)

'latest' (and no --params) is looked up again at every poll, so a run you start after opening the viewer is picked up
as well.  A run or file given by name is followed too: its file is reloaded when training saves it again (train.py
writes it atomically, every --save-every updates).  A policy that cannot be loaded keeps the current one (the
message says why).  No jax, no pygame: `load` turns the file's bytes into params.
"""
from __future__ import annotations

from pathlib import Path

from platformer_ai import runs


def describe(path: Path) -> str:
    """'RUN  update 75/190  goal 0.62' from the run.json beside a policy file ('RUN' / the file name without one)."""
    path = Path(path)
    info = runs.read_info(path.parent)
    name = path.parent.name if path.name == runs.PARAMS else path.name
    if not info:
        return name
    out = name
    cfg = info.get("config") or {}
    batch = int(cfg.get("num_envs", 0)) * int(cfg.get("num_steps", 0))
    if "updates" in info and batch:
        out += f"  update {info['updates']}/{int(cfg.get('total_timesteps', 0)) // batch}"
        if info.get("finished"):
            out += " done"
    goal = ((info.get("last_eval") or {}).get("greedy") or {}).get("goal")
    if goal is not None:
        out += f"  eval goal {goal:.2f}"
    return out


class PolicyWatch:
    def __init__(self, arg: str | None, load, root: Path | None = None, follow: bool = True):
        """arg: what --params says ('latest', a run name or folder, a .msgpack file).  load(bytes) -> params, raising
        ValueError if the bytes are not a policy for this network.  The first load raises that ValueError (the
        caller explains it); later ones keep the current policy.  follow=False: never reload by itself."""
        self.arg, self.load, self.follow = arg, load, follow
        self.root = runs.ROOT if root is None else Path(root)
        self.path = self.resolve()
        self.stamp = None
        self.params = None
        self.label = ""
        self.loads = 0                              # how many policies were loaded (1 = only the first)
        stamp = self._stamp(self.path)
        if stamp is not None:
            self._take(self.path, stamp, self.load(self.path.read_bytes()))

    def resolve(self) -> Path:
        return runs.resolve_params(self.arg, self.root)

    @staticmethod
    def _stamp(path: Path):
        """What says that a file changed: a new file (train.py renames a new one over it), a new time or size."""
        try:
            st = Path(path).stat()
        except OSError:
            return None
        return st.st_ino, st.st_mtime_ns, st.st_size

    def _take(self, path: Path, stamp, params) -> None:
        self.path, self.stamp, self.params = path, stamp, params
        self.label = describe(path)
        self.loads += 1

    def poll(self, force: bool = False) -> str | None:
        """Load the policy if it is new: another file ('latest' moved to a new run) or the same file saved again.
        force: look even with follow off, and load the file again even if it did not change (the viewer's L key).
        -> a message for the log, or None if nothing changed."""
        if not (self.follow or force):
            return None
        path = self.resolve()
        stamp = self._stamp(path)
        if stamp is None:                            # no policy there (yet): keep what we have
            return f"no policy at {path}: keeping the current one" if force else None
        if not force and path == self.path and stamp == self.stamp:
            return None
        try:
            params = self.load(path.read_bytes())
        except (OSError, ValueError) as e:
            self.path, self.stamp = path, stamp      # do not try the same broken file again at every episode
            return f"cannot load {path} ({e}): keeping the current policy"
        self._take(path, stamp, params)
        return f"policy loaded: {self.label}"
