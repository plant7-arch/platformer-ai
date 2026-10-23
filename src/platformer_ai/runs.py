"""Training runs: one folder per run under runs/, and which one is the latest.

    runs/2026-10-14_2054/                 a run (train.py makes the name from the start time; --name adds a label)
        params.msgpack                    the policy (saved every --save-every updates, default 25, and at the end)
        checkpoint.msgpack                network + optimiser state + random key (for train --resume)
        run.json                          what it was trained on, the settings, progress and the last evaluation
        levels.txt                        the level files (train / test)

    runs/pretrained/                      the policy that comes with the repository (only params.msgpack is in git)

Level paths in run.json are relative to the project folder (or absolute outside it), so --resume works from any
subfolder.  runs/ is the one in the project folder (the folder with this project's pyproject.toml, found from the current folder
upwards), so the commands find the runs from any subfolder of the repository; outside of it, ./runs is used.

"latest" = your run whose params.msgpack was saved most recently (a run still training counts too), so the viewer and
play pick up a new run without being told (and follow it while it trains: they reload a newer policy between
episodes, see policywatch.py).  runs/pretrained/ is only the fallback when you have no runs of your own
(a git pull that updates it does not make it "latest"); use it on purpose with --params pretrained.
Train on from any policy (also pretrained, which has no optimiser checkpoint):  train --init-from pretrained
List them all:

    uv run python -m platformer_ai.runs
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path


def project_root(start: Path | None = None) -> Path | None:
    """The platformer-ai checkout that contains `start` (default: the current folder): the nearest folder upwards with
    this project's pyproject.toml.  None outside of it."""
    here = Path.cwd() if start is None else Path(start)
    here = here.resolve()
    for d in (here, *here.parents):
        try:
            if 'name = "platformer-ai"' in (d / "pyproject.toml").read_text():
                return d
        except (OSError, UnicodeDecodeError):
            continue
    return None


def project_path(name: str, start: Path | None = None) -> Path:
    """`name` inside the project folder (runs, assets), as short a path as possible: plain 'runs' when you are in the
    project folder, '../runs' from a subfolder; outside of any checkout just `name` (the current folder)."""
    root = project_root(start)
    if root is None:
        return Path(name)
    try:
        return Path(os.path.relpath(root / name, Path.cwd() if start is None else start))
    except ValueError:                          # another drive (Windows): no relative path exists
        return root / name


def to_project(path: str, start: Path | None = None) -> str:
    """A path typed on the command line (relative to the current folder) as run.json keeps it: relative to the project
    folder when it is inside it (so a resume works from any subfolder, and in another clone), else absolute."""
    here = Path.cwd() if start is None else Path(start)
    full = (here / path).resolve()
    root = project_root(here)
    if root is not None:
        try:
            return full.relative_to(root).as_posix()
        except ValueError:
            pass
    return str(full)


def from_project(stored: str, start: Path | None = None) -> str:
    """A path from run.json (see to_project) as a path to open from the current folder."""
    if os.path.isabs(stored):
        return stored
    return str(project_path(stored, start))


def stored_paths(info: dict, key: str, start: Path | None = None) -> list[str] | None:
    """run.json's level paths (key: 'level_args' / 'test_level_args') as paths to open now.  Runs made before 0.2.0
    kept the paths as typed (relative to wherever training was started): those are tried as they are, then inside the
    project folder."""
    paths = info.get(key)
    if not paths:
        return paths
    if info.get("paths") == "project":
        return [from_project(p, start) for p in paths]
    here = Path.cwd() if start is None else Path(start)
    out = []
    for p in paths:
        if os.path.isabs(p) or (here / p).exists():
            out.append(p)
        else:
            alt = project_path(p, start)
            out.append(str(alt) if (here / alt).exists() else p)
    return out


def write_atomic(path: Path, data: bytes | str) -> None:
    """Write a file so that a reader (the viewer following a run) never sees half of it: write a .tmp beside it,
    then rename it over the old one (one step for the file system)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, str):
        tmp.write_text(data)
    else:
        tmp.write_bytes(data)
    tmp.replace(path)


ROOT = project_path("runs")
PARAMS = "params.msgpack"
STATE = "checkpoint.msgpack"     # network + optimiser state + random key: what --resume needs
INFO = "run.json"
PRETRAINED = "pretrained"        # the shipped policy: runs/pretrained/


def new_run_dir(name: str | None = None, root: Path = ROOT, now: _dt.datetime | None = None) -> Path:
    """A new, not yet existing folder runs/YYYY-MM-DD_HHMMSS[_name] (created)."""
    stamp = (now or _dt.datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    base = f"{stamp}_{name}" if name else stamp
    path, i = Path(root) / base, 2
    while path.exists():                        # two runs started in the same second
        path, i = Path(root) / f"{base}-{i}", i + 1
    path.mkdir(parents=True)
    return path


def all_runs(root: Path = ROOT) -> list[Path]:
    """Every folder under root that has a policy, oldest save first."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = [d for d in root.iterdir() if d.is_dir() and (d / PARAMS).is_file()]
    return sorted(found, key=lambda d: ((d / PARAMS).stat().st_mtime, d.name))


def latest(root: Path = ROOT) -> Path | None:
    """Your run with the most recently saved params.msgpack; runs/pretrained/ only if there is no other; else None."""
    found = all_runs(root)
    own = [r for r in found if r.name != PRETRAINED]
    return (own or found or [None])[-1]


def resolve_params(arg: str | None, root: Path = ROOT) -> Path:
    """What --params means: 'latest' (default), a run folder, a run name under runs/, or a params file."""
    if arg in (None, "", "latest"):
        run = latest(root)
        return (run / PARAMS) if run else Path(root) / PRETRAINED / PARAMS    # (not there: the caller says so)
    p = Path(arg)
    if p.is_dir():
        return p / PARAMS
    if not p.exists() and (Path(root) / arg).is_dir():
        return Path(root) / arg / PARAMS
    return p


def resolve_run(arg: str | None, root: Path = ROOT) -> Path | None:
    """A run folder from 'latest', a folder path or a run name under runs/ (None if there is no such run)."""
    if arg in (None, "", "latest"):
        return latest(root)
    for p in (Path(arg), Path(root) / arg):
        if p.is_dir():
            return p
    return None


def read_info(run: Path) -> dict:
    try:
        return json.loads((Path(run) / INFO).read_text())
    except (OSError, ValueError):
        return {}


def write_info(run: Path, info: dict) -> None:
    write_atomic(Path(run) / INFO, json.dumps(info, indent=2))         # never a half-written file


def table(root: Path = ROOT) -> list[str]:
    """One line per run (oldest first), the latest marked with '*'."""
    runs = all_runs(root)
    if not runs:
        return [f"no runs in {root}/ yet (train one: uv run python -m platformer_ai.train)"]
    newest = latest(root)
    lines = [f"  {'run':<32} {'saved':<16} {'steps':>12}  {'levels':>6}  {'goal':>5} {'return':>7}  status"]
    for run in runs:
        info = read_info(run)
        saved = _dt.datetime.fromtimestamp((run / PARAMS).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        steps = info.get("steps")
        ev = (info.get("last_eval") or {}).get("greedy") or {}
        n_lv = len(info.get("levels") or []) or "-"
        status = ("shipped with the repository" if run.name == PRETRAINED and not info else
                  "-" if not info else "done" if info.get("finished") else "unfinished")
        goal, ret = (f"{ev['goal']:.2f}", f"{ev['ret']:.2f}") if ev else ("-", "-")
        lines.append(f"{'*' if run == newest else ' '} {run.name:<32} {saved:<16} "
                     f"{f'{steps:,}' if steps else '-':>12}  {n_lv:>6}  {goal:>5} {ret:>7}  {status}")
    lines.append("* = latest (what viewer / play use by default);  goal / return: last greedy evaluation")
    return lines


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else ROOT
    print("\n".join(table(root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
