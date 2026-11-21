"""Flatten modular manuscript and Supplementary Material sources into self-contained flat LaTeX files.

Inlines all \\input{...} files (tables, bibliography) and expands all generated numeric macros from
source/tables/numbers.tex so that pdflatex compiles each file in 3 passes with zero subfolders
and zero BibTeX runs. Also checks abstract word count.

Run: python -X utf8 flatten_submission.py
"""
import os, re, shutil
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.dirname(HERE)
SRC = os.path.join(MAN, "source")
ROOT = Path(MAN).parent
PKG = str(ROOT.parent if ROOT.name == "CosVI_code_and_results" else ROOT / "release_build")
LATEX_DIR = os.path.join(PKG, "CosVI_LaTeX_source")
os.makedirs(LATEX_DIR, exist_ok=True)


def parse_macros(path):
    macros = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}(?:\s*%.*)?$", line.strip())
        if m:
            # Handle balanced braces on line if comment had braces
            key = m.group(1)
            raw = line.split(f"\\newcommand{{\\{key}}}{{", 1)[1]
            depth = 1
            val = []
            for ch in raw:
                if ch == "{":
                    depth += 1
                    val.append(ch)
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                    val.append(ch)
                else:
                    val.append(ch)
            macros[key] = "".join(val)
    return macros


def expand_inputs_and_macros(src_path, macros):
    txt = open(src_path, encoding="utf-8").read()
    # Remove \input{tables/numbers.tex}
    txt = re.sub(r"\\input\{tables/numbers\.tex\}\s*\n?", "% (all numerical macros inlined from stored results)\n", txt)

    # Recursively inline other \input{...}
    def repl_input(m):
        rel = m.group(1)
        fp = os.path.join(SRC, rel)
        return open(fp, encoding="utf-8").read()

    txt = re.sub(r"\\input\{([^}]+)\}", repl_input, txt)

    # Expand macros (longest name first to avoid prefix collision, only when followed by non-letter)
    keys = sorted(macros.keys(), key=len, reverse=True)
    # Two passes in case a macro contains another macro
    for _ in range(2):
        for k in keys:
            v = macros[k]
            txt = re.sub(rf"\\{k}(?![A-Za-z])(?:\{{\}})?", lambda _m, _v=v: _v, txt)
    return txt


def count_abstract_words(tex):
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        return 0
    body = m.group(1)
    # Rough academic word count (strip simple formatting macros while keeping words/numbers)
    body = re.sub(r"\\CosVI(?:J\[[^\]]*\])?", "CosVI", body)
    body = re.sub(r"\\MAfBm", "MA-fBm", body)
    body = re.sub(r"\\ensuremath\{([^}]*)\}", r"\1", body)
    body = re.sub(r"\\[A-Za-z]+\*?", " ", body)
    body = re.sub(r"[{}~$^_\\,]", " ", body)
    words = [w for w in body.split() if re.search(r"[A-Za-z0-9]", w)]
    return len(words)


def main():
    macros = parse_macros(os.path.join(SRC, "tables", "numbers.tex"))
    print(f"Loaded {len(macros)} macros from numbers.tex")

    main_flat = expand_inputs_and_macros(os.path.join(SRC, "CosVI_manuscript_src.tex"), macros)
    out_main = os.path.join(LATEX_DIR, "CosVI_manuscript.tex")
    open(out_main, "w", encoding="utf-8").write(main_flat)
    n_abs = count_abstract_words(main_flat)
    print(f"Wrote {out_main} ({len(main_flat)} chars); abstract word count ~= {n_abs} (CSDA limit <= 250)")

    sm_flat = expand_inputs_and_macros(os.path.join(SRC, "CosVI_supplementary_src.tex"), macros)
    out_sm = os.path.join(LATEX_DIR, "CosVI_supplementary_material.tex")
    open(out_sm, "w", encoding="utf-8").write(sm_flat)
    print(f"Wrote {out_sm} ({len(sm_flat)} chars)")

    # Copy freshly rendered Figures 1..6 (and any available 7..11) from source/figures/one_column
    FIG_MAP = {
        "Figure_1": "fig01_hero",
        "Figure_2": "fig02_procedure",
        "Figure_3": "fig03_fidelity",
        "Figure_4": "fig04_linear",
        "Figure_5": "fig05_benchmark",
        "Figure_6": "fig06_doublewell",
        "Figure_7": "fig11_solver",
        "Figure_8": "fig09_calib_ablation",
        "Figure_9": "fig07_twodim",
        "Figure_10": "fig08_real",
        "Figure_11": "fig10_cost",
    }
    fig_dir = os.path.join(PKG, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    one_col = os.path.join(SRC, "figures", "one_column")
    copied = []
    for target, stem in FIG_MAP.items():
        src_pdf = os.path.join(one_col, f"{stem}.pdf")
        src_png = os.path.join(one_col, f"{stem}.png")
        if os.path.exists(src_pdf):
            shutil.copy2(src_pdf, os.path.join(LATEX_DIR, f"{target}.pdf"))
            shutil.copy2(src_pdf, os.path.join(fig_dir, f"{target}.pdf"))
            copied.append(target)
        if os.path.exists(src_png):
            shutil.copy2(src_png, os.path.join(fig_dir, f"{target}.png"))
    print(f"Copied {len(copied)} vector PDFs from one_column: {', '.join(copied)}")


if __name__ == "__main__":
    main()
