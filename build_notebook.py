"""Assemble the scripts in scripts/ into an executable, labeled Jupyter notebook.

Each generated script becomes one code cell, preceded by a markdown cell that
names the script it came from. Cells appear in execution order (01 -> 09).

Usage:
    python build_notebook.py                       # write the notebook
    jupyter nbconvert --to notebook --execute --inplace fashion_mnist_training_accuracy.ipynb
    jupyter nbconvert --to html fashion_mnist_training_accuracy.ipynb
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

SCRIPTS_DIR = Path("scripts")
NOTEBOOK_PATH = Path("fashion_mnist_training_accuracy.ipynb")

TITLES = {
    "01_setup.py": "Setup",
    "02_load_data.py": "Load Fashion MNIST",
    "03_explore_data.py": "Explore the data",
    "04_build_model.py": "Build the MLP classifier",
    "05_train_utils.py": "Training and evaluation helpers",
    "06_train_model.py": "Train the model",
    "07_plot_training_accuracy.py": "Plot the training accuracy",
    "08_evaluate_and_predict.py": "Evaluate on the test set and predict",
    "09_save_model.py": "Save and reload the model",
}

def split_docstring(source):
    """Return (docstring, code) for a script starting with a docstring."""
    if source.startswith('"""'):
        end = source.index('"""', 3)
        return source[3:end].strip(), source[end + 3:].lstrip("\n")
    return "", source

cells = [new_markdown_cell(
    "# Problem 1 — Building a Neural Network with PyTorch "
    "and Plotting the Training Accuracy\n\n"
    "Template: *Chapter 10 – Building Neural Networks with PyTorch* "
    "(`10_neural_nets_with_pytorch.ipynb`).\n\n"
    "The whole process (setup → data → model → training → **training "
    "accuracy plot** → evaluation → saving) was built incrementally with "
    "Claude Code as separate scripts in `scripts/`. Each script is placed "
    "below as one code cell, in executable order, and every cell is "
    "labeled with the script it came from.\n\n"
    "| Cell | Generated script | Purpose |\n|---|---|---|\n" +
    "\n".join(f"| {i} | `scripts/{name}` | {title} |"
              for i, (name, title) in enumerate(TITLES.items(), start=1))
)]

for i, (name, title) in enumerate(TITLES.items(), start=1):
    doc, code = split_docstring((SCRIPTS_DIR / name).read_text())
    body = "\n".join(doc.splitlines()[1:]).strip()  # drop the title line
    cells.append(new_markdown_cell(
        f"## Cell {i} — Script `scripts/{name}`: {title}\n\n{body}"))
    cells.append(new_code_cell(f"# ===== Script: scripts/{name} =====\n{code}"))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3",
                   "language": "python"},
    "language_info": {"name": "python"},
})
nbformat.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH} with {len(cells)} cells")
