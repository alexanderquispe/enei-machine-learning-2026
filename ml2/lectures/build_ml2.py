"""Assemble the six ML II decks from the deep-learning-up sources, without touching the sources.

    python ml2/lectures/build_ml2.py           # write lecture_0N.tex for the six sessions
    python ml2/lectures/build_ml2.py --compile # also run tectonic on each

The sources live in reference/deep_learning_up (a git submodule) and are read only. Each ENEI
session is a list of source lectures with cut rules — whole sections dropped, individual frames
dropped, and a few frames inserted — applied by title, so the mapping is auditable in
ADAPTATION.md, which this script also writes.

Why the cuts: ML I already covered supervised learning, linear models, gradient descent,
regularization and PCA/SVD in full; the ENEI course is Caltech-style, so the applied-economics
motivation frames go; and two lectures share one three-hour slot in sessions 1 and 2.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reference" / "deep_learning_up"
OUT = Path(__file__).resolve().parent

INSTITUTE = r"\institute[ENEI]{\small{Alexander Quispe \\ ENEI -- INEI \,\textperiodcentered\, PEU-CD 2026}}"

# --------------------------------------------------------------------------- inserted frames
VANISHING_GRADIENT = r"""
\begin{frame}{Vanishing Gradients: The Product of Jacobians}\footnotesize
For $\mathbf{h}_t=\sigma(\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b})$, write $\mathbf{a}_t=\mathbf{W}\mathbf{h}_{t-1}+\mathbf{U}\mathbf{x}_t+\mathbf{b}$.
\begin{block}{One step of the chain rule}
\[
\frac{\partial\mathbf{h}_t}{\partial\mathbf{h}_{t-1}}=\operatorname{diag}\!\bigl(\sigma'(\mathbf{a}_t)\bigr)\,\mathbf{W},
\qquad\text{so}\qquad
\frac{\partial\mathbf{h}_T}{\partial\mathbf{h}_t}=\prod_{k=t+1}^{T}\operatorname{diag}\!\bigl(\sigma'(\mathbf{a}_k)\bigr)\,\mathbf{W}.
\]
\end{block}
\begin{block}{Bounding the product}
Let $\gamma=\max_z\sigma'(z)$ ($\gamma=1$ for $\tanh$, $\gamma=\tfrac14$ for the sigmoid) and $\|\mathbf{W}\|_2$ the largest singular value. Sub-multiplicativity of the spectral norm gives
\[
\Bigl\|\frac{\partial\mathbf{h}_T}{\partial\mathbf{h}_t}\Bigr\|_2\ \le\ \prod_{k=t+1}^{T}\bigl\|\operatorname{diag}(\sigma'(\mathbf{a}_k))\bigr\|_2\,\|\mathbf{W}\|_2\ \le\ \bigl(\gamma\,\|\mathbf{W}\|_2\bigr)^{\,T-t}.
\]
\end{block}
\begin{itemize}
  \item If $\gamma\|\mathbf{W}\|_2<1$ the bound decays \textbf{exponentially} in the time gap $T-t$: the gradient from a loss at time $T$ cannot reach a parameter that mattered at time $t$. This is the vanishing-gradient problem, and it is unavoidable for a sigmoid ($\gamma=\tfrac14$) unless $\|\mathbf{W}\|_2>4$.
  \item If $\gamma\|\mathbf{W}\|_2>1$ nothing prevents \textbf{explosion}; gradient clipping is the standard patch.
  \item The LSTM's cell state has Jacobian $\operatorname{diag}(\mathbf{f}_t)$ --- the forget gate --- with no $\mathbf{W}$ in the product. That is the whole design.
\end{itemize}
\end{frame}
"""

ML1_BRIDGE = r"""
\begin{frame}{What We Carry Over From Machine Learning I}\small
Lecture 6 of ML I proved, in full, the linear-algebra facts this lecture builds on. They are used, not re-derived:
\begin{itemize}
  \item \textbf{Projections.} $\mathbf{P}=\mathbf{V}\mathbf{V}^\top$ with $\mathbf{V}^\top\mathbf{V}=\mathbf{I}_k$ is symmetric and idempotent; $\|\mathbf{x}\|^2=\|\mathbf{P}\mathbf{x}\|^2+\|(\mathbf{I}-\mathbf{P})\mathbf{x}\|^2$.
  \item \textbf{PCA.} The top eigenvector of $\mathbf{S}=\tfrac1N\mathbf{X}^\top\mathbf{X}$ maximizes projected variance (Lagrangian); the top $k$ maximize $\operatorname{tr}(\mathbf{V}^\top\mathbf{S}\mathbf{V})$ (Ky Fan); retained variance $=\lambda_1+\dots+\lambda_k$; maximum variance $\Leftrightarrow$ minimum reconstruction error.
  \item \textbf{SVD.} For centred $\mathbf{X}=\mathbf{U}\mathbf{D}\mathbf{V}^\top$, the right singular vectors are the principal components and $\lambda_j=d_j^2/N$.
  \item \textbf{Eckart--Young.} Among all matrices of rank $\le k$, the truncated SVD $\mathbf{U}_k\mathbf{D}_k\mathbf{V}_k^\top$ is closest to $\mathbf{X}$ in Frobenius norm, with error $\sum_{j>k}d_j^2$.
\end{itemize}
\begin{alertblock}{Today's question}
Eckart--Young needs \emph{every} entry of $\mathbf{X}$. What if most entries are missing --- a user has rated 20 of 10{,}000 films, a firm is observed only before a policy? That is the latent-factor problem, and the trace norm is how the rank constraint survives it.
\end{alertblock}
\end{frame}
"""

# --------------------------------------------------------------------------- session specs
# Each entry: (source lecture number, dict of rules). Rules:
#   drop_sections: section titles to remove entirely (all frames until the next \section)
#   keep_frames:   frame titles to keep even inside a dropped section
#   drop_frames:   frame titles to remove wherever they are
#   insert_after:  {frame title: latex} to insert after that frame
#   insert_before_section: {section title: latex}
SESSIONS = [
    dict(n=1, title="From the Perceptron to Backpropagation", date="October 2, 2026", sources=[
        (1, dict(drop_sections=["The Supervised Learning Problem", "Linear Models as Starting Point"],
                 keep_frames=["The Limitation of Linear Models"],
                 drop_frames=["Deep Learning in Economics -- Why Should You Care?", "Course Roadmap",
                              "Next Lecture: Backpropagation \\& Gradient Descent", "References"])),
        (2, dict(drop_frames=["Recap from Lecture 1"])),
    ]),
    dict(n=2, title="Training and Convolutional Networks", date="October 9, 2026", sources=[
        (3, dict(drop_frames=["L2 Regularization (Weight Decay)", "L1 Regularization (Sparsity)",
                              "Why Depth Matters -- Economic Analogy", "Next Lecture: Convolutional Neural Networks", "References"])),
        (4, dict(drop_frames=["Recap from Lecture 3", "Why CNNs Matter for Economists"])),
    ]),
    dict(n=3, title="Sequences: RNNs, LSTMs and GRUs", date="October 12, 2026", sources=[
        (5, dict(insert_after={"Why Long-Range Dependencies Are Hard": VANISHING_GRADIENT})),
    ]),
    dict(n=4, title="Matrix Factorization and Latent Factors", date="October 16, 2026", sources=[
        (6, dict(drop_sections=["Linear Algebra Recap"], insert_before_section={"Matrix Norms": ML1_BRIDGE})),
    ]),
    dict(n=5, title="Embeddings and Text as Data", date="October 19, 2026", sources=[(7, dict())]),
    dict(n=6, title="Attention and Self-Attention", date="October 23, 2026", sources=[(8, dict())]),
]

FRAME_TITLE = re.compile(r"\\begin\{frame\}(?:<[^>]*>)?(?:\[[^\]]*\])?\{([^}]*)\}|\\frametitle(?:<[^>]*>)?\{([^}]*)\}")


def split_blocks(body: str):
    """Yield ('section', title, text) and ('frame', title, (pre, text)) blocks in order.

    Both \\begin{frame} and deep-learning-up's own \\begin{transitionframe} (the yellow section
    dividers) count as frames, so dropping a section also drops its divider."""
    lines = body.splitlines(keepends=True)
    blocks, buf, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*\\section\{(.*)\}", line)
        if m and not line.lstrip().startswith("%"):
            blocks.append(("section", m.group(1), "".join(buf) + line)); buf = []; i += 1; continue
        env = "frame" if "\\begin{frame}" in line else ("transitionframe" if "\\begin{transitionframe}" in line else None)
        if env and not line.lstrip().startswith("%"):
            j = i
            while f"\\end{{{env}}}" not in lines[j]:
                j += 1
            text = "".join(lines[i:j + 1])
            t = FRAME_TITLE.search(text) if env == "frame" else re.search(r"\\Huge\s*\\textbf\{([^}]*)\}", text)
            title = ((t.group(1) or (t.group(2) if t.lastindex and t.lastindex >= 2 else "")) if t else "").strip()
            # keep the text preceding the frame separate: it may hold \tikzset / \section-level setup
            # that must survive even when the frame itself is dropped
            blocks.append(("frame", title, ("".join(buf), text))); buf = []; i = j + 1; continue
        buf.append(line); i += 1
    if buf:
        blocks.append(("tail", "", "".join(buf)))
    return blocks


def adapt(src_n: int, rules: dict, log: list):
    text = (SRC / f"lecture_{src_n}" / f"lecture_{src_n}_slides.tex").read_text()
    pre, rest = text.split("\\begin{document}", 1)
    body = rest.rsplit("\\end{document}", 1)[0]
    out, section, dropping = [], None, False
    for kind, title, blk in split_blocks(body):
        if kind == "section":
            section = title; dropping = title in rules.get("drop_sections", [])
            if title in rules.get("insert_before_section", {}):
                out.append(rules["insert_before_section"][title]); log.append(f"    + inserted frame before section '{title}'")
            if dropping:
                log.append(f"    - dropped section '{title}'")
            else:
                out.append(blk)
            continue
        if kind == "tail":
            out.append(blk); continue
        pre_text, blk = blk
        out.append(pre_text)  # setup between frames is always kept
        if "\\titlepage" in blk or "\\maketitle" in blk or "\\tableofcontents" in blk:
            continue  # one title and one outline are added per session
        if dropping and title not in rules.get("keep_frames", []):
            continue
        if title in rules.get("drop_frames", []):
            log.append(f"    - dropped frame '{title}'"); continue
        blk = re.sub(r"(Recap from Lecture \d+)", "Recap", blk)
        blk = re.sub(r"Lecture \d+ Summary", "Summary", blk)
        out.append(blk)
        if title in rules.get("insert_after", {}):
            out.append(rules["insert_after"][title]); log.append(f"    + inserted frame after '{title}'")
    return pre, "".join(out)


def patch_preamble(pre: str, n: int, title: str, date: str) -> str:
    pre = re.sub(r"^\\usepackage\{bbm\}", "%\\\\usepackage{bbm}   % PK-only fonts; not renderable by tectonic (indicator uses \\\\mathbf{1})", pre, flags=re.M)
    pre = re.sub(r"^\\usepackage\{(lipsum|multimedia)\}", r"%\\usepackage{\1}", pre, flags=re.M)
    pre = re.sub(r"\\title\[[^\]]*\]\{.*\}", r"\\title[]{\\textcolor{blue}{Machine Learning II}\\\\ \\textcolor{black}{\\normalsize Lecture %d: %s}}" % (n, title), pre)
    pre = re.sub(r"\\institute\[[^\]]*\]\{.*\}", INSTITUTE.replace("\\", "\\\\"), pre)
    pre = re.sub(r"\\date\{[^}]*\}", r"\\date{%s}" % date, pre)
    return pre



def adapt_pre(src_n: int) -> str:
    return (SRC / f"lecture_{src_n}" / f"lecture_{src_n}_slides.tex").read_text().split("\\begin{document}", 1)[0]


MACRO_LINE = re.compile(r"^\\(newcommand|renewcommand|DeclareMathOperator\*?|newcolumntype|tikzset|tikzstyle|definecolor|newenvironment|NewEnviron)\s*\{?\\?([A-Za-z@]+)", re.M)


def merge_macros(pre: str, other_pres: list) -> str:
    """Append to `pre` every macro/colour/tikz definition that a later source's preamble has and `pre` lacks.

    A merged session keeps the first source's preamble; the second source's body may use macros only
    its own preamble defined. Definitions are copied line by line, by name, so nothing is redefined."""
    have = set(m.group(2) for m in MACRO_LINE.finditer(pre))
    extra = []
    for op in other_pres:
        for line in op.splitlines():
            m = MACRO_LINE.match(line)
            if m and m.group(2) not in have:
                extra.append(line); have.add(m.group(2))
    if extra:
        pre += "\n% ---- definitions merged from the second source lecture ----\n" + "\n".join(extra) + "\n"
    return pre


def build():
    log_lines = ["# ML II decks — how each was assembled from deep-learning-up", "",
                 "Generated by `build_ml2.py`. Sources are read from the submodule and never modified.", ""]
    written = []
    for s in SESSIONS:
        n = s["n"]; log_lines.append(f"## Lecture {n} — {s['title']} ({s['date']})")
        parts, pre = [], None
        for src_n, rules in s["sources"]:
            log_lines.append(f"  source: deep-learning-up lecture {src_n}")
            p, body = adapt(src_n, rules, log_lines)
            body = body.replace("\\mathbbm{1}", "\\mathbf{1}")
            if pre is None:
                pre = p
            parts.append(body)
        pre = patch_preamble(pre, n, s["title"], s["date"])
        pre = merge_macros(pre, [adapt_pre(src_n) for src_n, _ in s["sources"][1:]])
        doc = (pre + "\\begin{document}\n\n\\begin{frame}[plain]\\titlepage\\end{frame}\n\n"
               "\\begin{frame}{Outline}\\tableofcontents\\end{frame}\n" + "\n".join(parts) + "\n\\end{document}\n")
        folder = sorted(OUT.glob(f"lecture_0{n}_*"))[0]
        target = folder / f"lecture_0{n}.tex"
        target.write_text(doc); written.append(target)
        nfr = doc.count("\\begin{frame}")
        log_lines.append(f"  -> {target.relative_to(ROOT)}: {nfr} frames"); log_lines.append("")
        print(f"lecture_0{n}.tex  {nfr} frames  <- DL-UP {[x[0] for x in s['sources']]}")
    (OUT / "ADAPTATION.md").write_text("\n".join(log_lines))
    return written


if __name__ == "__main__":
    files = build()
    if "--compile" in sys.argv:
        for f in files:
            r = subprocess.run(["tectonic", f.name], cwd=f.parent, capture_output=True, text=True)
            errs = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("error:") and "something bad" not in l and "unrecoverable" not in l]
            over = len(re.findall(r"Overfull", r.stdout + r.stderr))
            print(f"  {f.name}: {'ok' if r.returncode == 0 else 'FAILED'}  overfull={over}  {errs[:2]}")
