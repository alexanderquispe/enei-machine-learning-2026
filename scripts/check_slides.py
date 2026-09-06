"""Static checks on the compiled decks: the defects a reader would see.

    python scripts/check_slides.py

Fails (exit 1) on:
  * placeholder titles ("Frame Title", "Untitled", "Lorem")
  * frames with a title and no body
  * leftover TODO / FIXME / XXX markers in the .tex
  * slide count outside the budget for a 3-hour session (25-80)
  * a figure referenced from a .tex that does not exist on disk
  * `reference/` tracked in git while the GitHub repository is public
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = re.compile(r"^(frame title|untitled|lorem ipsum|title here)$", re.I)
MARKERS = re.compile(r"\b(TODO|FIXME|XXX)\b")   # bare markers; \texttt{TODO} mentions (lab instructions) are excluded below
SLIDE_BUDGET = (25, 80)   # derivation-dense ML I decks run ~30 slides in 3 h; the two merged ML II decks run ~80
PROBLEMS = []


def problem(msg):
    PROBLEMS.append(msg); print("  ✗", msg)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("\\&", "&"))


def check_pdf(pdf: Path, section_titles=()):
    import pdfplumber, logging, warnings
    logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
    sections = {_norm(s) for s in section_titles}
    with pdfplumber.open(pdf) as doc:
        n = len(doc.pages)
        for i, page in enumerate(doc.pages, 1):
            lines = [l.strip() for l in (page.extract_text() or "").splitlines() if l.strip()]
            if not lines:
                problem(f"{pdf.name} p{i}: empty page"); continue
            title = lines[0]
            if PLACEHOLDERS.match(title):
                problem(f"{pdf.name} p{i}: placeholder title '{title}'")
            # body = everything but the title and the trailing page number
            body = [l for l in lines[1:] if not re.fullmatch(r"\d+ ?/ ?\d+|\d+", l)]
            # a section divider is a single centred line with no page number; anything else with a
            # title and no body is a frame someone forgot to fill in
            if not body and len(lines) > 1 and _norm(title) not in sections:
                problem(f"{pdf.name} p{i}: frame '{title}' has no body")
    return n


def check_tex(tex: Path):
    src = tex.read_text()
    scrubbed = re.sub(r"\\texttt\{[^}]*\}", "", src)   # `\texttt{\# TODO}` in lab instructions is not a leftover
    for m in MARKERS.finditer(scrubbed):
        line = scrubbed[: m.start()].count("\n") + 1
        problem(f"{tex.relative_to(ROOT)}:{line}: leftover marker '{m.group(0)}'")
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", src):
        f = m.group(1)
        cands = [tex.parent / f, tex.parent / "figures" / f]
        if not any(c.exists() for c in cands):
            problem(f"{tex.relative_to(ROOT)}: figure '{f}' not found")


def check_visibility():
    tracked = subprocess.run(["git", "ls-files", "reference"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not tracked:
        return
    r = subprocess.run(["gh", "repo", "view", "--json", "visibility", "-q", ".visibility"], cwd=ROOT, capture_output=True, text=True)
    vis = r.stdout.strip().upper()
    if vis and vis != "PRIVATE":
        problem(f"repository is {vis} but reference/ (third-party material) is tracked")
    else:
        print(f"  visibility: {vis or 'unknown (gh not available)'}; reference/ tracked: yes")


def main():
    decks = sorted(list(ROOT.glob("ml1/lectures/*/lecture_*.tex")) + list(ROOT.glob("ml2/lectures/*/lecture_*.tex")))
    for tex in decks:
        print(tex.relative_to(ROOT))
        check_tex(tex)
        pdf = tex.with_suffix(".pdf")
        if not pdf.exists():
            problem(f"{tex.relative_to(ROOT)}: no PDF (run scripts/build.py)"); continue
        src = tex.read_text()
        sections = (re.findall(r"\\section\{([^}]*)\}", src) + re.findall(r"\\sectionframe\{([^}]*)\}", src)
                    + re.findall(r"\\begin\{transitionframe\}.*?\\Huge\s*\\textbf\{([^}]*)\}", src, flags=re.S))
        sections = [re.sub(r"\\texorpdfstring\{[^}]*\}\{([^}]*)\}", r"\1", s.replace("$", "")) for s in sections]
        n = check_pdf(pdf, sections)
        lo, hi = SLIDE_BUDGET
        if not lo <= n <= hi:
            problem(f"{pdf.name}: {n} slides, outside the {lo}-{hi} budget for a 3-hour session")
        else:
            print(f"  {n} slides")
    for tex in sorted(ROOT.glob("ml*/labs/*/tutorial.tex")):
        check_tex(tex)
    print("repository")
    check_visibility()
    print()
    if PROBLEMS:
        print(f"{len(PROBLEMS)} problem(s)"); return 1
    print("no problems found"); return 0


if __name__ == "__main__":
    sys.exit(main())
