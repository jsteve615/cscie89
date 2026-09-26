"""Script 07 - Plot the training accuracy.

Left: training accuracy per epoch (with validation accuracy for reference).
The training accuracy is averaged *during* each epoch, so it is plotted half
an epoch earlier, as in the template notebook's learning curves.
Right: training accuracy per batch (200-batch moving average) with the
training loss per epoch on a secondary axis.
Requires: 06_train_model.py
"""
epochs = np.arange(1, n_epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs - 0.5, history["train_metrics"], "b.--",
         label="Training accuracy")
ax1.plot(epochs, history["valid_metrics"], "r.-",
         label="Validation accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_title("Training accuracy per epoch")
ax1.set_xlim(0, n_epochs + 0.5)
ax1.grid()
ax1.legend(loc="lower right")

batch_acc = np.array(history["batch_train_metrics"])
batches_per_epoch = len(batch_acc) / n_epochs
x_batches = np.arange(1, len(batch_acc) + 1) / batches_per_epoch
window = 200
moving_avg = np.convolve(batch_acc, np.ones(window) / window, mode="valid")
ax2.plot(x_batches[window - 1:], moving_avg, "b-",
         label=f"Moving avg ({window} batches)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Training accuracy")
ax2.set_title("Training accuracy per batch")
ax2.set_xlim(0, n_epochs + 0.5)
ax2.grid()
ax3 = ax2.twinx()
ax3.plot(epochs, history["train_losses"], "g.-", label="Training loss")
ax3.set_ylabel("Training loss", color="g")
lines = ax2.get_legend_handles_labels()
lines3 = ax3.get_legend_handles_labels()
ax2.legend(lines[0] + lines3[0], lines[1] + lines3[1], loc="center right",
           fontsize=12)

save_fig("training_accuracy_plot")
plt.show()

print(f"Final training accuracy:   {history['train_metrics'][-1]:.4f}")
print(f"Final validation accuracy: {history['valid_metrics'][-1]:.4f}")
