# Machine Learning I & II — ENEI (INEI), PEU-CD 2026

Course materials for **Machine Learning I** and **Machine Learning II**, Module II of the
*Programa de Especialización Universitaria en Ciencia de Datos* (PEU-CD 2026) at the
**Escuela Nacional de Estadística e Informática (ENEI)** of Peru's National Institute of
Statistics and Informatics (INEI).

| | |
|---|---|
| **Instructor** | Alexander Quispe |
| **Teaching assistant** | Rodrigo Grijalba |
| **Machine Learning I** | September 7 – September 30, 2026 — 6 lectures + 4 labs |
| **Machine Learning II** | October 2 – October 26, 2026 — 6 lectures + 4 labs |
| **Session** | 3 hours, 19:00–22:00 Lima time, online |
| **Hours per course** | 30: 18 instructor (lectures) + 12 teaching assistant (labs) |

The course is taught in the style of Caltech's CS 155: every result is derived on the slide, every
worked example is recomputed in code, and applications illustrate a method rather than motivate it.
The full design rationale is in [`PLAN.md`](PLAN.md).

---

## Schedule

Lectures are the instructor's; labs are the teaching assistant's and fall on the Wednesdays.
The calendar is derived from the official ENEI and PUCP calendars by the pipeline in
[`schedule/`](schedule/); the Excel sent to the ENEI is
[`schedule/output/horario_peu_cd_2026_ml1_ml2.xlsx`](schedule/output/horario_peu_cd_2026_ml1_ml2.xlsx).

### Machine Learning I

| Date | Session | Slides / material |
|---|---|---|
| Mon Sep 7 | **Lecture 1** — Supervised learning and linear models | [`ml1/lectures/lecture_01_supervised_learning/lecture_01.pdf`](ml1/lectures/lecture_01_supervised_learning/lecture_01.pdf) |
| Wed Sep 9 | Lab 1 — Working environment and a first end-to-end pipeline | [`ml1/labs/lab_01_pipeline/`](ml1/labs/lab_01_pipeline/) |
| Fri Sep 11 | **Lecture 2** — Optimization and linear classifiers | [`ml1/lectures/lecture_02_optimization/lecture_02.pdf`](ml1/lectures/lecture_02_optimization/lecture_02.pdf) |
| Mon Sep 14 | **Lecture 3** — Regularization and evaluation | [`ml1/lectures/lecture_03_regularization/lecture_03.pdf`](ml1/lectures/lecture_03_regularization/lecture_03.pdf) |
| Wed Sep 16 | Lab 2 — Ridge, lasso and cross-validation | [`ml1/labs/lab_02_regularization/`](ml1/labs/lab_02_regularization/) |
| Fri Sep 18 | **Lecture 4** — Decision trees and bagging | [`ml1/lectures/lecture_04_trees/lecture_04.pdf`](ml1/lectures/lecture_04_trees/lecture_04.pdf) |
| Mon Sep 21 | **Lecture 5** — Boosting | [`ml1/lectures/lecture_05_boosting/lecture_05.pdf`](ml1/lectures/lecture_05_boosting/lecture_05.pdf) |
| Wed Sep 23 | Lab 3 — Trees, random forests and gradient boosting | [`ml1/labs/lab_03_ensembles/`](ml1/labs/lab_03_ensembles/) |
| Mon Sep 28 | **Lecture 6** — Unsupervised learning: clustering and PCA | [`ml1/lectures/lecture_06_unsupervised/lecture_06.pdf`](ml1/lectures/lecture_06_unsupervised/lecture_06.pdf) |
| Wed Sep 30 | Lab 4 — Clustering, PCA and the ML I project | [`ml1/labs/lab_04_unsupervised/`](ml1/labs/lab_04_unsupervised/) |

Supplementary reading (not examined): [`ml1/supplementary/generative_classifiers.pdf`](ml1/supplementary/generative_classifiers.pdf)
— multinomial logistic regression, LDA, GDA and Naive Bayes, 101 slides.

### Machine Learning II

| Date | Session | Slides / material |
|---|---|---|
| Fri Oct 2 | **Lecture 1** — From the perceptron to backpropagation | [`ml2/lectures/lecture_01_perceptron_backprop/lecture_01.pdf`](ml2/lectures/lecture_01_perceptron_backprop/lecture_01.pdf) |
| Wed Oct 7 | Lab 1 — PyTorch: tensors, autograd and a first MLP | [`ml2/labs/lab_01_pytorch/`](ml2/labs/lab_01_pytorch/) |
| Fri Oct 9 | **Lecture 2** — Training and convolutional networks | [`ml2/lectures/lecture_02_training_cnn/lecture_02.pdf`](ml2/lectures/lecture_02_training_cnn/lecture_02.pdf) |
| Mon Oct 12 | **Lecture 3** — Sequences: RNNs, LSTMs and GRUs | [`ml2/lectures/lecture_03_rnn/lecture_03.pdf`](ml2/lectures/lecture_03_rnn/lecture_03.pdf) |
| Wed Oct 14 | Lab 2 — Backpropagation by hand, and CNNs | [`ml2/labs/lab_02_backprop_cnn/`](ml2/labs/lab_02_backprop_cnn/) |
| Fri Oct 16 | **Lecture 4** — Matrix factorization and latent factors | [`ml2/lectures/lecture_04_matrix_factorization/lecture_04.pdf`](ml2/lectures/lecture_04_matrix_factorization/lecture_04.pdf) |
| Mon Oct 19 | **Lecture 5** — Embeddings and text as data | [`ml2/lectures/lecture_05_embeddings/lecture_05.pdf`](ml2/lectures/lecture_05_embeddings/lecture_05.pdf) |
| Wed Oct 21 | Lab 3 — Factorization and embeddings in practice | [`ml2/labs/lab_03_factorization_embeddings/`](ml2/labs/lab_03_factorization_embeddings/) |
| Fri Oct 23 | **Lecture 6** — Attention and self-attention | [`ml2/lectures/lecture_06_attention/lecture_06.pdf`](ml2/lectures/lecture_06_attention/lecture_06.pdf) |
| Mon Oct 26 | Lab 4 — Project: matrix completion on a panel | [`ml2/labs/lab_04_matrix_completion/`](ml2/labs/lab_04_matrix_completion/) |

---

## What is in each folder

```
style/          Shared LaTeX identity: Beamer preamble, article preamble, one notation file (macros.tex)
ml1/lectures/   Six Beamer decks, each with figures/make_figures.py that regenerates every figure
ml1/labs/       Four labs: tutorial.pdf (the session plan and the math), student and solution notebooks,
                and build_notebook.py, the single source both notebooks are generated from
ml1/supplementary/  Generative classifiers, assembled from the instructor's ML_Foundations deck
ml2/lectures/   Six decks assembled from the deep-learning-up sources by build_ml2.py;
                ADAPTATION.md records exactly which frames were dropped or added and why
ml2/labs/       Four labs: tutorial.pdf plus the deep-learning-up homework notebooks and data,
                placed by copy_notebooks.py
schedule/       The calendar pipeline (see schedule/README.md)
reference/      Source material — Caltech CS 155, ML_Foundations, deep-learning-up (submodule). Private.
scripts/        build.py, check_slides.py, verify_math.py — see below
```

Every lab notebook pair (`*_student.ipynb`, `*_solution.ipynb`) in `ml1/` is generated by the
`build_notebook.py` next to it; edit the script, not the notebooks. `python build_notebook.py --check`
executes every solution cell in order.

---

## Building

There is no `pdflatex` here; everything builds with [`tectonic`](https://tectonic-typesetting.github.io),
which fetches packages on demand.

```bash
pip install -r requirements.txt          # numpy, pandas, scikit-learn, matplotlib, pdfplumber, torch
python scripts/build.py                  # compile every deck and tutorial; table of results
python scripts/build.py --figures ml1    # regenerate figures first, ML I only
python ml2/lectures/build_ml2.py --compile   # re-assemble the ML II decks from the submodule
```

### Checks

Three scripts, run before every push:

| Script | What it enforces |
|---|---|
| `scripts/build.py` | Every `.tex` compiles; no unresolved `??`; no Overfull box beyond 20 pt |
| `scripts/check_slides.py` | No placeholder titles, no empty frames, no TODO markers, slide count within 35–75 per 3-hour session, every figure present, and `reference/` is not tracked in a public repository |
| `scripts/verify_math.py` | Every numerical example printed in the ML I slides is recomputed in numpy and compared to the printed value; each is tagged `% verify: <name>` in the source |

```bash
python scripts/build.py && python scripts/check_slides.py && python scripts/verify_math.py
```

---

## Sources and licensing

This repository is **private** and must stay so: `reference/caltech_cs155/` holds Prof. Yisong Yue's
CS 155 lecture notes and problem sets, which are not ours to redistribute. They are tracked with
Git LFS — run `git lfs install` once before cloning. Provenance and terms for every source are in
[`reference/README.md`](reference/README.md).

The ML I decks are original, written for this course on the structure of CS 155 lectures 1–6 and 9
and reusing text from the instructor's ML_Foundations deck. The ML II decks are the instructor's
deep-learning-up lectures, adapted by `ml2/lectures/build_ml2.py` (the mapping is in
`ml2/lectures/ADAPTATION.md`). The ML II labs use the deep-learning-up homeworks as written.

## Bibliography

- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed. — the reference for ML I; section pointers close every lecture.
- Goodfellow, Bengio & Courville (2016), *Deep Learning*; Zhang et al. (2023), *Dive into Deep Learning* — ML II.
- Novikoff (1962); Breiman (1996, 2001); Freund & Schapire (1997); Schapire et al. (1998); Tibshirani (1996); Arthur & Vassilvitskii (2007); Chen & Guestrin (2016) — the papers behind the proofs.
