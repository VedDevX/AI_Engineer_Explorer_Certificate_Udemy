# Task 1: Perform EDA and Preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Datsets
data = fetch_california_housing(as_frame=True)
df = data.frame

X = df[["MedInc", "HouseAge", "AveRooms"]]
Y = df["MedHouseVal"]

# Inspect the data
print(df.info())
print(df.describe())

# Visualize relationships
sns.pairplot(df, vars=['MedInc', 'AveRooms', 'HouseAge', 'MedHouseVal'])
plt.show() 

# Check for Missing values
print("Missing values: \n", df.isnull().sum())

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# train the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# make prediction
Y_pred = model.predict(X_test)

# Evaluate Performance 
mse = mean_squared_error(Y_test, Y_pred)
print("Linear Regression MSE: ", mse)