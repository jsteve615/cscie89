"""Script 06 - Train the classifier on Fashion MNIST.

Requires: 01-05
"""
history = train(model, optimizer, xentropy, accuracy, train_loader,
                valid_loader, n_epochs)
