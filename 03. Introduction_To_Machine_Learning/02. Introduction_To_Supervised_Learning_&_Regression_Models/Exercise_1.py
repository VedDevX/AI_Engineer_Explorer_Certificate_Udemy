# Implement a simple linear regression model

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate Synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 100
Y = 3 * X + np.random.randn(100, 1) * 2

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit Linear regressin
model = LinearRegression()
model.fit(X_train, Y_train)

# make predictions
Y_pred = model.predict(X_test)

# Print Coefficients
print("Slope: ", model.coef_[0][0])
print("Intercept: ", model.intercept_[0])

# Visualize the regression line
plt.scatter(X_test, Y_test, color="blue", label="Actual")
plt.plot(X_test, Y_pred, color="red", label="Predicted")
plt.title("Linear Regression Model")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Evaluate the performance
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("MSE: ", mse)
print("R-Squared: ", r2)