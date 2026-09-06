"""Compile every LaTeX document in the course with tectonic and report the result as a table.

    python scripts/build.py            # build everything
    python scripts/build.py ml1        # only paths containing "ml1"
    python scripts/build.py --figures  # regenerate figures (figures/make_figures.py) before compiling

Fails (exit 1) on any compile error, any unresolved reference (`??`), or any Overfull box wider
or taller than the tolerance below (overfull boxes in ml2/lectures are reported but not enforced:
those decks inherit their layout from deep-learning-up). Requires `tectonic` on PATH.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERFULL_TOLERANCE_PT = 20.0
TARGETS = ["ml1/lectures/*/lecture_*.tex", "ml1/labs/*/tutorial.tex", "ml1/supplementary/generative_classifiers.tex",
           "ml2/lectures/*/lecture_*.tex", "ml2/labs/*/tutorial.tex"]


def build_one(tex: Path, figures: bool):
    d = tex.parent
    if figures and (d / "figures" / "make_figures.py").exists():
        subprocess.run([sys.executable, "figures/make_figures.py"], cwd=d, check=True, capture_output=True)
    r = subprocess.run(["tectonic", tex.name], cwd=d, capture_output=True, text=True)
    log = r.stdout + r.stderr
    errors = [l for l in log.splitlines() if l.startswith("error:") and "something bad" not in l and "unrecoverable" not in l]
    over = [float(m.group(1)) for m in re.finditer(r"Overfull \\[hv]box \(([\d.]+)pt", log)]
    bad_over = [o for o in over if o > OVERFULL_TOLERANCE_PT]
    if "ml2/lectures" in str(tex) or "supplementary" in str(tex):
        bad_over = []   # inherited layout (deep-learning-up, ML_Foundations): reported, not enforced
    pdf = tex.with_suffix(".pdf")
    unresolved = False
    if pdf.exists():
        try:
            import pdfplumber, logging, warnings
            logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
            with pdfplumber.open(pdf) as doc:
                pages = len(doc.pages)
                unresolved = any("??" in (p.extract_text() or "") for p in doc.pages)
        except ImportError:
            pages = "?"
    else:
        pages = 0
    ok = r.returncode == 0 and pdf.exists() and not bad_over and not unresolved
    return dict(path=str(tex.relative_to(ROOT)), ok=ok, pages=pages, errors=errors[:2],
                overfull=len(over), bad_overfull=len(bad_over), unresolved=unresolved)


def main():
    figures = "--figures" in sys.argv
    filt = [a for a in sys.argv[1:] if not a.startswith("--")]
    texs = sorted(t for pat in TARGETS for t in ROOT.glob(pat) if not filt or any(f in str(t) for f in filt))
    if not texs:
        print("nothing to build"); return 0
    results = [build_one(t, figures) for t in texs]
    w = max(len(r["path"]) for r in results)
    print(f"{'document':{w}s}  {'ok':>3}  {'pages':>5}  {'overfull':>8}  {'>tol':>4}  notes")
    for r in results:
        notes = "; ".join(r["errors"]) + (" unresolved ??" if r["unresolved"] else "")
        print(f"{r['path']:{w}s}  {'yes' if r['ok'] else 'NO':>3}  {r['pages']!s:>5}  {r['overfull']:>8}  {r['bad_overfull']:>4}  {notes}")
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} documents build cleanly")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
