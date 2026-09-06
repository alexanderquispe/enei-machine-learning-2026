# ==============================================================================
# Problem C: Modeling Part 1 [8 Points]
# ==============================================================================
#
# Using PyTorch's "Sequential" model class, build a deep network to classify
# the handwritten digits. You may only use the following layers:
#
#   - Linear: A fully-connected layer
#   - ReLU (activation): Sets negative inputs to 0
#   - Softmax (activation): Rescales input so that it can be interpreted as
#     a (discrete) probability distribution.
#   - Dropout: Takes some probability and at every iteration sets weights to
#     zero at random with that probability (effectively regularization)
#
# A sample network with 20 hidden units is in the sample code file.
# (Note: activations, Dropout, and your last Linear layer do not count toward
# your hidden unit count, because the final layer is "observed" and not hidden.)
#
# Use categorical cross entropy as your loss function. There are also a number
# of optimizers you can use (an optimizer is just a fancier version of SGD),
# and feel free to play around with them, but RMSprop and Adam are the most
# popular and will probably work best. You also should find the batch size and
# number of epochs that give you the best results (default is batch_size=32,
# epochs=10).
#
# Look at the sample code to see how to train your model. PyTorch should make
# it very easy to tinker with your network architecture.
#
# YOUR TASK: Using at most 100 hidden units, build a network using only the
# allowed layers that achieves test accuracy of at least 0.975. Turn in the
# code of your model as well as the best test accuracy that it achieved.
#
# HINT: For best results on this problem and the two following problems,
# normalize the input vectors by dividing the values by 255 (as the pixel
# values range from 0 to 255).
# ==============================================================================



import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.ToTensor())

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1000,  
                                          shuffle=False)

# model definition
# Using 100 hidden units total: 64 + 36 = 100
# Deeper network with lower dropout for better accuracy
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),    # First hidden layer: 64 units
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 36),     # Second hidden layer: 36 units (total: 64+36=100)
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(36, 10)      # Output layer (not counted as hidden)
)
print(model)
print(f"Total hidden units: 64 + 36 = 100")

# loss function
loss_fn = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# number of epochs to train
num_epochs = 20
# training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # forward pass (ToTensor() already normalizes to [0,1])
        output = model(data)
        loss = loss_fn(output, target)
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_accuracy = correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy:.4f}')
# Final test accuracy
print(f'Final Test Accuracy: {test_accuracy:.4f}')

