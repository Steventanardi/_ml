# 🧠 Homework 10: Calling an AI Model – Basic Example (PyTorch)

This project demonstrates how to define and call a simple AI model using PyTorch. It walks through building a minimal neural network, generating input, and performing a forward pass.

---

## 📘 Overview

This is a basic example of how to:
- Define a model using `torch.nn.Module`
- Use `torch.nn.functional` for activation functions
- Pass input through the model
- Print the model’s output

---

## 🎯 Objective

- ✅ Create a simple feedforward neural network in PyTorch
- ✅ Perform a forward pass using randomly generated input
- ✅ Understand how data flows through each layer

---

## 🏗️ Model Architecture

| Layer         | Description                                |
|---------------|--------------------------------------------|
| **Input**     | 10 input features                          |
| **Hidden**    | 5 neurons with ReLU activation             |
| **Output**    | 1 neuron (suitable for regression/binary)  |

---

## 🚀 How It Works

1. Define a neural network with two layers:
    - `fc1`: Linear layer (10 → 5)
    - `fc2`: Linear layer (5 → 1)
2. Apply **ReLU activation** between layers.
3. Generate a random input tensor of shape `[1, 10]`.
4. Call the model and print the output.

---

## 🧾 Example Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate model
model = SimpleModel()

# Create random input
input_data = torch.randn(1, 10)

# Forward pass
output = model(input_data)

print("Model output:", output)
