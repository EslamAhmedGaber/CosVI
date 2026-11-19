"""List overfull boxes of both builds with the source file they come from (tracks the input-file stack in the log)."""
import os, re
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source")
for lay in ("one_column", "two_columns"):
    log = open(os.path.join(SRC, f"{lay}.log"), encoding="latin-1").read()
    stack, out = [], []
    for tok in re.finditer(r"\((\./[^\s()]+\.tex)|\)|Overfull \\hbox \(([\d.]+)pt too wide\) (?:in paragraph|in alignment|detected) at lines? (\d+)", log):
        if tok.group(1):
            stack.append(tok.group(1))
        elif tok.group(0) == ")":
            if stack:
                stack.pop()
        else:
            out.append((stack[-1] if stack else "?", tok.group(3), float(tok.group(2))))
    print(f"{lay}: {len(out)} overfull")
    for f, line, pt in out:
        if pt > 1.0:
            print(f"   {f}:{line}  {pt:.1f}pt")
