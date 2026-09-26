"""Script 09 - Save the trained weights and reload them to check they work.

Based on "Saving and Loading a PyTorch Model" in the template notebook.
Requires: 06_train_model.py
"""
MODEL_PATH = Path() / "models" / "fashion_mnist_mlp.pt"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

loaded_model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300,
                               n_hidden2=100, n_classes=10).to(device)
loaded_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
loaded_acc = evaluate_tm(loaded_model, test_loader, accuracy).item()
print(f"Saved to {MODEL_PATH}; reloaded model test accuracy: {loaded_acc:.4f}")
