"""List every overfull hbox of both builds with the source file that was open when TeX reported it.
Run from work\manuscript\scripts:  python overfull_where.py"""
import os, re

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source")
OPEN = re.compile(r"\((\./(?:sections|tables|figures)/[^\s()]+)")
OVER = re.compile(r"Overfull \\hbox \(([\d.]+)pt too wide\) (?:in paragraph at lines (\d+)--(\d+)|detected at line (\d+))")

for lay in ("two_columns", "one_column"):
    log = open(os.path.join(SRC, f"{lay}.log"), encoding="latin-1").read()
    ev = [(m.start(), "open", m.group(1)) for m in OPEN.finditer(log)]
    ev += [(m.start(), "over", m) for m in OVER.finditer(log)]
    ev.sort(key=lambda e: e[0])
    cur = "?"
    print(lay)
    for pos, kind, x in ev:
        if kind == "open":
            cur = x
        else:
            lines = f"{x.group(2)}-{x.group(3)}" if x.group(2) else x.group(4)
            print(f"   {float(x.group(1)):6.1f} pt  {cur}  lines {lines}")
