# CSCI E-89 — Problem 1: Neural Network with PyTorch + Training Accuracy Plot

Fashion MNIST MLP classifier based on the Chapter 10 template
(`template/10_neural_nets_with_pytorch.ipynb`), built incrementally as scripts
with Claude Code.

| File | Description |
|---|---|
| `fashion_mnist_training_accuracy.ipynb` | Executed notebook, one labeled cell per script |
| `fashion_mnist_training_accuracy.html` | HTML version of the notebook |
| `SUMMARY.md` | Summary of the Claude Code dialog |
| `scripts/01_…09_*.py` | Generated scripts, in execution order |
| `build_notebook.py` | Rebuilds the notebook from the scripts |

## Reproduce locally
```bash
pip install torch torchvision torchmetrics matplotlib scikit-learn jupyter nbconvert
python scripts/run_all.py                    # run the scripts directly, or:
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace fashion_mnist_training_accuracy.ipynb
jupyter nbconvert --to html fashion_mnist_training_accuracy.ipynb
```

## Problem 2 — the whole process in one notebook (`problem2/`)
| File | Description |
|---|---|
| `problem2/problem2_fashion_mnist_notebook.ipynb` | Self-contained notebook written by Claude Code: config, data, model, training, training-accuracy plot, evaluation, save/reload, and a final self-test cell |
| `problem2/problem2_fashion_mnist_notebook.html` | HTML version (full 20-epoch run) |

To test it locally, open the notebook and choose **Kernel → Restart & Run All**. The last cell should print
"All checks passed." Set `QUICK_RUN = True` in the first code cell for a run that takes about 20 seconds.
