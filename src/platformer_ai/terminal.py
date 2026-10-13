"""Full-screen terminal drawing that leaves your scrollback alone.

The old play.py cleared the whole screen every frame (ESC[2J), which wipes whatever was on the
terminal, including the end of the training log.  Screen draws on the *alternate screen buffer*
instead (what vim / less / htop do): the normal screen is restored untouched when we exit, even
after Ctrl-C or an error.  No dependencies; does nothing when stdout is not a terminal.
"""
import sys

ALT_ON, ALT_OFF = "\033[?1049h", "\033[?1049l"      # enter / leave the alternate screen buffer
HIDE_CURSOR, SHOW_CURSOR = "\033[?25l", "\033[?25h"
HOME, CLEAR_EOL, CLEAR_BELOW = "\033[H", "\033[K", "\033[J"


class Screen:
    def __init__(self, out=None, enabled=None):
        self.out = sys.stdout if out is None else out
        self.enabled = self.out.isatty() if enabled is None else enabled

    def __enter__(self):
        if self.enabled:
            self.out.write(ALT_ON + HIDE_CURSOR)
            self.out.flush()
        return self

    def draw(self, text: str) -> None:
        """Replace the screen contents with `text` (cursor home + overwrite, no full clear)."""
        if self.enabled:
            self.out.write(HOME + (CLEAR_EOL + "\n").join(text.split("\n")) + CLEAR_EOL + CLEAR_BELOW)
            self.out.flush()

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self.out.write(SHOW_CURSOR + ALT_OFF)
            self.out.flush()
        return False                                     # never swallow exceptions
