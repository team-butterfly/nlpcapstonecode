"""
Various utilities for pretty console output
Ported nigh-verbatim from a similar file I use for node
"""
import time as python_time
import os
from sys import stdout

class colors():
    END = "\033[0m"
    BRIGHT = "\033[1m"
    DIM = "\033[2m"
    UNDERSCORE = "\033[4m"
    BLINK = "\033[5m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    DK_RED = "\033[41m"
    DK_GREEN = "\033[42m"
    DK_YELLOW = "\033[43m"
    DK_BLUE = "\033[44m"
    DK_MAGENTA = "\033[45m"
    DK_CYAN = "\033[46m"
    DK_WHITE = "\033[47m"

timers = {}

def fmt(iterable):
    return " ".join(str(i) for i in iterable)
def h1(*args):
    print(colors.BRIGHT + fmt(args) + colors.END)
def wait(*args):
    input(colors.BLUE + fmt(args) + colors.END)
def log(*args):
    print(colors.YELLOW + fmt(args) + colors.END)
def info(*args):
    print(colors.DIM + fmt(args) + colors.END)
def debug(*args):
    print(colors.DK_BLUE + "\t" + fmt(args) + colors.END)
def warn(*args):
    print(colors.DK_CYAN + "WARN:\t" + colors.END + colors.CYAN + fmt(args) + colors.END)
def error(*args):
    print(colors.DK_RED + colors.BLINK + "ERROR:\t" + colors.END + colors.RED + fmt(args) + colors.END)
def time(key):
    timers[key] = python_time.time()
def time_end(key):
    if key in timers:
        t = python_time.time() - timers[key]
        print("\t" + str(t) + colors.DIM  + " s \t" + key + colors.END)
        del timers[key]
def notify(*args):
    # Play bell
    print('\a')
    # Attempt to send a notification (will fail, but not crash, if not on macOS)
    os.system("""
          osascript -e 'display notification "{}" with title "{}"'
          """.format(args[0], fmt(args[1:])))

def progress_bar(title, pct, length, style="[=]"):
    ticks = min(length, round(pct*length))
    bar = style[1:-1] * ticks + " " * (length-ticks)
    stdout.write("\r" + style[0] + bar + style[-1] + colors.DIM + " " + title + colors.END)
    stdout.flush()
