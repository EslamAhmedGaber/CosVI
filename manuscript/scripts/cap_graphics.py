r"""Cap every figure of the section files at the line width (adjustbox 'max width' scales down only, never up).
Idempotent: an inclusion that already has options is left alone.  Run from work\manuscript\scripts."""
import os, re, glob

SEC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source", "sections")
for f in glob.glob(os.path.join(SEC, "*.tex")):
    s = open(f, encoding="utf-8").read()
    new = re.sub(r"\\includegraphics\{", r"\\includegraphics[max width=\\linewidth]{", s)
    if new != s:
        open(f, "w", encoding="utf-8").write(new)
        print("capped", os.path.basename(f), s.count("\\includegraphics{"))
