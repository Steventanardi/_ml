import torch
import torch.nn.functional as F

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[5.0], [7.0], [9.0], [11.0]])

W = torch.randn((1, 1), requires_grad=True)
b = torch.randn((1,), requires_grad=True)

lr = 0.01
epochs = 1000

for epoch in range(epochs):
    y_pred = X @ W + b

    loss = F.mse_loss(y_pred, Y)

    loss.backward()

    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad

        W.grad.zero_()
        b.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print("\nLearned parameters:")
print(f"W = {W.item():.4f}, b = {b.item():.4f}")

x_test = torch.tensor([[5.0]])
y_test = x_test @ W + b
print(f"Prediction for x = 5: y = {y_test.item():.4f}")
