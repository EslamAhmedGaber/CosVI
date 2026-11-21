"""Regenerate the numeric material and, when supplied, compile the article sources."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent.parent
SOURCE = SCRIPTS.parent / "source"


def run_script(name):
    subprocess.run([sys.executable, "-X", "utf8", str(SCRIPTS / name)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="render all figures before compiling")
    args = parser.parse_args()
    run_script("make_macros.py")
    run_script("make_tables.py")
    if args.figures:
        for path in sorted(SCRIPTS.glob("fig*.py")):
            run_script(path.name)
    if not (SOURCE / "CosVI_manuscript_src.tex").is_file():
        print("Numerical material regenerated. Article sources are supplied in the journal archive.")
        return
    run_script("flatten_submission.py")
    release = ROOT.parent if ROOT.name == "CosVI_code_and_results" else ROOT / "release_build"
    flat = release / "CosVI_LaTeX_source"
    for name in ("CosVI_manuscript", "CosVI_supplementary_material"):
        with tempfile.TemporaryDirectory() as work:
            command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                       f"-output-directory={work}", f"{name}.tex"]
            for _ in range(3):
                subprocess.run(command, cwd=flat, check=True, stdout=subprocess.DEVNULL)
            pdf = Path(work, name + ".pdf")
            (release / pdf.name).write_bytes(pdf.read_bytes())


if __name__ == "__main__":
    main()
