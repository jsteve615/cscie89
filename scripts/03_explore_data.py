"""Script 03 - Inspect a sample and plot a grid of training images.

Requires: 01_setup.py, 02_load_data.py
"""
X_sample, y_sample = train_data[0]
print("shape:", X_sample.shape, "| dtype:", X_sample.dtype,
      "| class:", class_names[y_sample])

n_rows, n_cols = 4, 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.4))
for index in range(n_rows * n_cols):
    image, label = train_data[index]
    plt.subplot(n_rows, n_cols, index + 1)
    plt.imshow(image.squeeze(0), cmap="binary")
    plt.axis("off")
    plt.title(class_names[label], fontsize=9)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
save_fig("fashion_mnist_samples", tight_layout=False)
plt.show()
