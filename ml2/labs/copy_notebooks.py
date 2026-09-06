"""Populate the ML II lab folders with the notebooks and data from deep-learning-up.

    python ml2/labs/copy_notebooks.py

The homework notebooks in the submodule are the instructor's own material and are used as-is; each
lab folder gets the student and solution notebooks it needs plus the data files they read. The
mapping is the one in PLAN.md §4.4:

    lab_01_pytorch                 <- homework_1/code  (Exercises 1-3: OLS vs GD, a neuron, manual backprop)
    lab_02_backprop_cnn            <- homework_1/code  (Exercises 4-6) + homework_2/code
    lab_03_factorization_embeddings<- homework_3/code  (Problems 17 and 19)
    lab_04_matrix_completion       <- homework_3/code  (Problem 17 in full; 18 and 20 as extensions)

Problem statements (the LaTeX PDFs) are copied alongside so Rodrigo has the math in the same folder.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reference" / "deep_learning_up"
LABS = ROOT / "ml2" / "labs"

COPIES = {
    "lab_01_pytorch": [
        ("homework_1/code/Homework1_code.ipynb", "homework_1_code.ipynb"),
        ("homework_1/data/synthetic_2d.csv", "data/synthetic_2d.csv"),
        ("homework_1/data/diabetes.csv", "data/diabetes.csv"),
        ("homework_1/PDF/Homework1_math.pdf", "homework_1_theory.pdf"),
    ],
    "lab_02_backprop_cnn": [
        ("homework_1/PDF/Homework1_math.pdf", "homework_1_theory.pdf"),
        ("homework_2/homework_2_student.pdf", "homework_2_student.pdf"),
        ("homework_2/homework_2_solutions.pdf", "homework_2_solutions.pdf"),
        ("homework_2/code/homework_2_student.ipynb", "homework_2_student.ipynb"),
        ("homework_2/code/homework_2_solution.ipynb", "homework_2_solution.ipynb"),
    ],
    "lab_03_factorization_embeddings": [
        ("homework_3/homework_3_student.pdf", "homework_3_student.pdf"),
        ("homework_3/homework_3_solutions.pdf", "homework_3_solutions.pdf"),
        ("homework_3/code/homework_3_student.ipynb", "homework_3_student.ipynb"),
        ("homework_3/code/homework_3_solution.ipynb", "homework_3_solution.ipynb"),
        ("homework_3/code/data/dr_seuss.txt", "data/dr_seuss.txt"),
    ],
    "lab_04_matrix_completion": [
        ("homework_3/homework_3_student.pdf", "homework_3_student.pdf"),
        ("homework_3/code/homework_3_student.ipynb", "homework_3_student.ipynb"),
        ("homework_3/code/homework_3_solution.ipynb", "homework_3_solution.ipynb"),
    ],
}


def main():
    for lab, files in COPIES.items():
        for src, dst in files:
            s, d = SRC / src, LABS / lab / dst
            if not s.exists():
                print(f"  missing in submodule: {src}"); continue
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
        print(f"{lab}: {len(files)} files")


if __name__ == "__main__":
    main()
