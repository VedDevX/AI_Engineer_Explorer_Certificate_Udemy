# Task 1: Perform EDA and Preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load Telco Customer Churn Dataset
df_telco = pd.read_csv('Telco-Customer-Churn.csv')

# Enccode Categorical Variables 
le = LabelEncoder()
df_telco['churn'] = le.fit_transform(df_telco['churn'])

# Define features and target
X = df_telco.drop(columns=['churn'])
Y = df_telco['churn']

# scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the logistic regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, Y_train)

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

# Evaluate Model
log_pred = log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

print("\n Logistic Regression Classification Report: ")
print(classification_report(Y_test, log_pred))

print("\n KNN Classification Report: ")
print(classification_report(Y_test, knn_pred))

# Confusion matrix for logistic regression
print("Confusion Matrix: \n", confusion_matrix(Y_test, log_pred))

# Inspect the data
print(df_telco.info())
print(df_telco.describe())

# Visualize the churn distribution
sns.countplot(x='churn', data=df_telco)
plt.title("Churn Distribution")
plt.show()

# Handling Missing Values
df_telco.fillna(df_telco.mean(), inplace=True)