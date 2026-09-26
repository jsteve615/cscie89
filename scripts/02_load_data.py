"""Script 02 - Load Fashion MNIST with TorchVision and build the DataLoaders.

Based on "Using TorchVision to Load the Dataset" in the template notebook.
Requires: 01_setup.py
"""
import contextlib
import io
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# Silence the download progress bar so it doesn't flood the notebook
with contextlib.redirect_stdout(io.StringIO()), \
     contextlib.redirect_stderr(io.StringIO()):
    train_and_valid_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=True, download=True, transform=toTensor)
    test_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

class_names = train_and_valid_data.classes
print(f"train: {len(train_data)}, valid: {len(valid_data)}, "
      f"test: {len(test_data)}")
