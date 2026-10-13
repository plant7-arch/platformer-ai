"""play.py must not wipe the terminal: it draws on the alternate screen and restores everything."""
import io

import pytest

from platformer_ai.terminal import ALT_OFF, ALT_ON, HIDE_CURSOR, HOME, SHOW_CURSOR, Screen

ERASE_WHOLE_SCREEN = "\033[2J"      # the code that used to destroy the training log


def test_enters_and_restores_the_normal_screen():
    out = io.StringIO()
    with Screen(out, enabled=True) as screen:
        screen.draw("frame 1\nline two")
        screen.draw("frame 2\nline two")
    text = out.getvalue()
    assert text.startswith(ALT_ON + HIDE_CURSOR)           # game view lives on the alternate screen
    assert text.endswith(SHOW_CURSOR + ALT_OFF)            # normal screen and cursor come back
    assert text.count(ALT_ON) == 1 and text.count(ALT_OFF) == 1


def test_never_erases_the_whole_screen():
    out = io.StringIO()
    with Screen(out, enabled=True) as screen:
        for i in range(5):
            screen.draw(f"frame {i}\nsecond line")
    assert ERASE_WHOLE_SCREEN not in out.getvalue()
    assert out.getvalue().count(HOME) == 5                 # each frame redraws from the top-left


def test_restores_even_when_the_loop_crashes():
    out = io.StringIO()
    with pytest.raises(RuntimeError):
        with Screen(out, enabled=True) as screen:
            screen.draw("frame")
            raise RuntimeError("boom")
    assert out.getvalue().endswith(SHOW_CURSOR + ALT_OFF)   # and the error is not swallowed


def test_draw_text_is_complete_and_leftovers_are_cleared():
    out = io.StringIO()
    with Screen(out, enabled=True) as screen:
        screen.draw("aaa\nbbb")
    body = out.getvalue().split(HOME, 1)[1]
    assert "aaa" in body and "bbb" in body
    assert "\033[K" in body and "\033[J" in body            # shorter lines / old rows do not leave junk


def test_silent_when_not_a_terminal():
    out = io.StringIO()                                     # StringIO.isatty() is False
    with Screen(out) as screen:
        screen.draw("frame")
    assert out.getvalue() == ""
