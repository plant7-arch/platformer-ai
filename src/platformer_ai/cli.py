"""`--level FILE` for the viewer and play.

The level module reads PLATFORMER_LEVEL once, when it is first imported, and the viewer / play draw that level.  So
the argument is turned into that variable before anything imports the level (call level_from_argv() first thing)."""
from __future__ import annotations

import os
import sys


def level_from_argv(argv=None) -> str | None:
    """If argv has `--level FILE` (or --level=FILE): set PLATFORMER_LEVEL=FILE and return FILE."""
    argv = sys.argv[1:] if argv is None else argv
    path = None
    for i, a in enumerate(argv):
        if a == "--level" and i + 1 < len(argv):
            path = argv[i + 1]
        elif a.startswith("--level="):
            path = a.split("=", 1)[1]
    if path:
        os.environ["PLATFORMER_LEVEL"] = path
    return path


def add_level_argument(ap) -> None:
    """The matching argparse option (so it shows up in --help); level_from_argv() has already applied it."""
    ap.add_argument("--level", metavar="FILE",
                    help="level text file to play (default: the level in level.py, or PLATFORMER_LEVEL); "
                         "any trained policy can play any level")
