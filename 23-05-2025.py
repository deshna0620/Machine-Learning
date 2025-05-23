import csv
import numpy as np

# Load CSV manually
with open('Titanic-Dataset.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    rows = list(reader)

# Extract column indices
pclass_idx = header.index('Pclass')
age_idx = header.index('Age')
sibsp_idx = header.index('SibSp')
parch_idx = header.index('Parch')
fare_idx = header.index('Fare')

# Prepare feature and target lists
X_list = []
y_list = []

for row in rows:
    try:
        pclass = float(row[pclass_idx])
        age = float(row[age_idx])
        sibsp = float(row[sibsp_idx])
        parch = float(row[parch_idx])
        fare = float(row[fare_idx])
        X_list.append([pclass, age, sibsp, parch])
        y_list.append(fare)
    except:
        continue  # skip rows with missing/invalid data

# Convert to numpy arrays
X = np.array(X_list)
y = np.array(y_list).reshape(-1, 1)

# Add bias term
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Compute coefficients (Normal Equation)
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Predict
y_pred = X_b @ theta

# Evaluation metrics
mse = np.mean((y - y_pred) ** 2)
mae = np.mean(np.abs(y - y_pred))
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2_score = 1 - (ss_res / ss_total)

# Output
print("Linear Regression using NumPy")
print("-----------------------------")
print("Model Coefficients:")
print(f"Intercept: {theta[0][0]:.4f}")
print(f"Pclass:    {theta[1][0]:.4f}")
print(f"Age:       {theta[2][0]:.4f}")
print(f"SibSp:     {theta[3][0]:.4f}")
print(f"Parch:     {theta[4][0]:.4f}")
print()
print(f"Mean Squared Error (MSE):      {mse:.4f}")
print(f"Mean Absolute Error (MAE):     {mae:.4f}")
print(f"R² Score (Coefficient of Determination): {r2_score:.4f}")
