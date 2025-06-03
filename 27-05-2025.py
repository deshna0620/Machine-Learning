import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv("new_student-mat.csv")

# Encode categorical variables
categorical = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical, drop_first=True)

# Feature Engineering
data['avg_grade'] = (data['G1'] + data['G2']) / 2
data['engagement_score'] = data['studytime'] - data['goout']

# Select features
features = ['G1', 'G2', 'studytime', 'failures', 'absences', 'avg_grade', 'engagement_score']
X = data[features]
y = data['G3']

# Handle outliers (optional)
X = X[(X['absences'] < 50) & (X['failures'] < 4)]
y = y[X.index]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 3: Linear Regression from Scratch
class LinearRegressionScratch:
    def __init__(self, lr=0.0001, iterations=10000):
        self.lr = lr
        self.iterations = iterations
        self.losses = []

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.zeros(X.shape[1])
        m = len(y)

        for _ in range(self.iterations):
            predictions = X.dot(self.theta)
            errors = predictions - y
            gradient = (1 / m) * X.T.dot(errors)
            self.theta -= self.lr * gradient
            loss = np.mean(errors ** 2)
            self.losses.append(loss)

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.theta)

scratch_model = LinearRegressionScratch()
scratch_model.fit(X_train.values, y_train.values)
y_pred_scratch = scratch_model.predict(X_test.values)

# Save training loss plot
plt.plot(range(len(scratch_model.losses)), scratch_model.losses)
plt.title("Loss vs Iterations (Scratch)")
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.savefig("training_loss_curve.png", dpi=300)
plt.show()

# Task 4: Linear Regression using scikit-learn
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_sklearn = lr_model.predict(X_test)

print("Coefficients:", lr_model.coef_)
print("Intercept:", lr_model.intercept_)

# Task 5: Visualization
plt.scatter(y_test, y_pred_scratch, color='red', label='Scratch Model')
plt.scatter(y_test, y_pred_sklearn, color='blue', label='Sklearn Model')
plt.plot([0, 20], [0, 20], color='black', linestyle='--')
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.legend()
plt.title("Actual vs Predicted Grades")
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

plt.scatter(y_test, y_test - y_pred_sklearn, label='Residuals (Sklearn)')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Actual G3")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.legend()
plt.savefig("residual_plot.png", dpi=300)
plt.show()

# Task 6: Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Task 7: Evaluation
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}:\nMSE = {mse:.2f}, RMSE = {rmse:.2f}, R² = {r2:.2f}\n")

evaluate(y_test, y_pred_scratch, "Linear Regression (Scratch)")
evaluate(y_test, y_pred_sklearn, "Linear Regression (Sklearn)")
evaluate(y_test, y_pred_poly, "Polynomial Regression (Degree 2)")
