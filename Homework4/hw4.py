import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

inputs = np.array([
    [1,1,1,1,1,1,0],  
    [0,1,1,0,0,0,0],  
    [1,1,0,1,1,0,1],  
    [1,1,1,1,0,0,1],  
    [0,1,1,0,0,1,1],  
    [1,0,1,1,0,1,1],  
    [1,0,1,1,1,1,1],  
    [1,1,1,0,0,0,0],  
    [1,1,1,1,1,1,1],  
    [1,1,1,1,0,1,1],  
])

labels = np.eye(10)

n_input = 7
n_hidden = 5
n_output = 10

np.random.seed(42)
W1 = np.random.randn(n_input, n_hidden) * 0.1
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output) * 0.1
b2 = np.zeros((1, n_output))

learning_rate = 0.1
epochs = 5000
epsilon = 1e-5

def numerical_gradient(param, forward_func):
    grad = np.zeros_like(param)
    for i in range(param.shape[0]):
        for j in range(param.shape[1]):
            original_value = param[i, j]

            param[i, j] = original_value + epsilon
            loss_plus = forward_func()

            param[i, j] = original_value - epsilon
            loss_minus = forward_func()

            grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            param[i, j] = original_value 
    return grad

for epoch in range(epochs):
    z1 = inputs @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    output = sigmoid(z2)

    loss = mean_squared_error(labels, output)

    loss_fn_w2 = lambda: mean_squared_error(labels, sigmoid(a1 @ (W2 + 0) + b2))
    loss_fn_b2 = lambda: mean_squared_error(labels, sigmoid(a1 @ W2 + (b2 + 0)))
    loss_fn_w1 = lambda: mean_squared_error(labels, sigmoid((sigmoid(inputs @ (W1 + 0) + b1)) @ W2 + b2))
    loss_fn_b1 = lambda: mean_squared_error(labels, sigmoid((sigmoid(inputs @ W1 + (b1 + 0))) @ W2 + b2))

    dW2 = numerical_gradient(W2, loss_fn_w2)
    db2 = numerical_gradient(b2, loss_fn_b2)
    dW1 = numerical_gradient(W1, loss_fn_w1)
    db1 = numerical_gradient(b1, loss_fn_b1)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

final_output = sigmoid(sigmoid(inputs @ W1 + b1) @ W2 + b2)
predicted_labels = np.argmax(final_output, axis=1)
print("\nPredicted labels:", predicted_labels)
