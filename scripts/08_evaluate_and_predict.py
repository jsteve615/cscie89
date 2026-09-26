"""Script 08 - Evaluate on the test set and make predictions.

Based on the prediction cells of "Building the Classifier" in the template.
Requires: 06_train_model.py
"""
import torch.nn.functional as F

test_acc = evaluate_tm(model, test_loader, accuracy).item()
print(f"Test accuracy: {test_acc:.4f}")

model.eval()
X_new, y_new = next(iter(valid_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1)
y_proba = F.softmax(y_pred_logits, dim=1).cpu()

print("predicted:", [class_names[index] for index in y_pred])
print("actual:   ", [class_names[index] for index in y_new[:3]])
print("probabilities:\n", y_proba.round(decimals=3))
