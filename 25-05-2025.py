import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("auto-mpg.csv")

# Handle missing values
df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], inplace=True)
df['horsepower'] = df['horsepower'].astype(float)

# Select feature and target
X = df[['horsepower']].values
y = df['mpg'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and evaluate polynomial models
degrees = [1, 2, 3]
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Degree {d}:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R^2 Score: {r2:.3f}")
    print()

# Plot the fitted curves
plt.figure(figsize=(10, 6))
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    x_poly = poly.fit_transform(x_range)

    model = LinearRegression()
    model.fit(poly.fit_transform(X), y)
    y_poly_pred = model.predict(x_poly)

    plt.plot(x_range, y_poly_pred, label=f'Degree {d}')

plt.scatter(X, y, color='gray', alpha=0.5, label='Data')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Polynomial Regression Fit: Horsepower vs MPG')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("polynomial_fit.png")
plt.show()
