"""Rebuild the manuscript: numbers -> figures (optional) -> both PDFs -> page renders, with a log report.
Run: python build.py [--figures]   (pdflatex with elsarticle; Python with matplotlib and PyMuPDF for page renders)."""
import os, subprocess, sys, shutil, re
HERE = os.path.dirname(os.path.abspath(__file__)); MAN = os.path.dirname(HERE); SRC = os.path.join(MAN, "source")


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(f"failed: {cmd}")
    return r.stdout


run([sys.executable, "-X", "utf8", "make_macros.py"], HERE); print("ok make_macros.py")
if "--figures" in sys.argv:
    for s in sorted(f for f in os.listdir(HERE) if f.startswith("fig") and f.endswith(".py")):
        run([sys.executable, "-X", "utf8", s], HERE); print("ok", s)
for lay in ("one_column", "two_columns"):
    for _ in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{lay}.tex"], cwd=SRC, capture_output=True)
    log = open(os.path.join(SRC, f"{lay}.log"), encoding="latin-1").read()
    errs = re.findall(r"^! .*", log, re.M)
    undef = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' on page \d+ undefined", log)))
    over = len(re.findall(r"Overfull \\hbox", log))
    pages = re.findall(r"Output written on .*?\((\d+) pages", log)
    print(f"{lay}: pages {pages[-1] if pages else '?'}, errors {len(errs)}, undefined {len(undef)}, overfull {over}")
    for e in errs[:5]:
        print("   ", e)
    if undef:
        print("    undefined:", ", ".join(undef[:20]))
    shutil.copy(os.path.join(SRC, f"{lay}.pdf"), os.path.join(MAN, f"CosVI_{lay}.pdf"))
try:
    import fitz
    rev = os.path.join(MAN, "review", "pages"); os.makedirs(rev, exist_ok=True)
    for lay in ("one_column", "two_columns"):
        d = fitz.open(os.path.join(MAN, f"CosVI_{lay}.pdf"))
        for i, pg in enumerate(d):
            pg.get_pixmap(dpi=80).save(os.path.join(rev, f"{lay}_p{i + 1:02d}.png"))
        print(lay, "page renders", d.page_count)
except ImportError:
    print("PyMuPDF not available: page renders skipped")
