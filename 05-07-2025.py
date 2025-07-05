import numpy as np
import matplotlib.pyplot as plt

# Perceptron Class
class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # insert bias
        z = np.dot(self.weights, x)
        return self.activation(z)

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                xi_bias = np.insert(xi, 0, 1)  # insert bias
                z = np.dot(self.weights, xi_bias)
                pred = self.activation(z)
                error = target - pred
                self.weights += self.lr * error * xi_bias

# AND Gate Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])

# Train Perceptron
p = Perceptron(input_size=2, lr=0.1, epochs=10)
p.train(X, y)

# Test Predictions
print("Predictions:")
for xi in X:
    print(f"{xi} -> {p.predict(xi)}")

# Plotting
def plot_decision_boundary(X, y, model):
    x1 = np.linspace(-0.5, 1.5, 10)
    x2 = -(model.weights[1] * x1 + model.weights[0]) / model.weights[2]
    plt.plot(x1, x2, 'k--', label='Decision Boundary')
    for i in range(len(y)):
        if y[i] == 0:
            plt.plot(X[i][0], X[i][1], 'ro')
        else:
            plt.plot(X[i][0], X[i][1], 'go')
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Perceptron - AND Gate Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.savefig("perceptron_05-07-2025.png")
    plt.show()

plot_decision_boundary(X, y, p)
