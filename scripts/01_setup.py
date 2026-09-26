"""Script 01 - Setup: imports, version checks, device selection, plot style.

Based on the "Setup" section of the Chapter 10 template notebook
(10_neural_nets_with_pytorch.ipynb).
"""
import sys
from pathlib import Path

assert sys.version_info >= (3, 10)

from packaging.version import Version
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchmetrics

assert Version(torch.__version__) >= Version("2.6.0")

# Pick the fastest available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Default font sizes to make the figures prettier (as in the template)
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Where to save figures
IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=150):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

n_epochs = 20

print(f"PyTorch {torch.__version__} | device: {device} | epochs: {n_epochs}")
