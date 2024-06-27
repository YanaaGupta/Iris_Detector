import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Loading, displaying, and exploring the dataset 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=column_names)
print(iris.head())

# Preprocessing the data and encoding the species column to numeric values
iris['species'] = iris['species'].astype('category').cat.codes

# Separate features and target variable
X = iris.drop('species', axis=1)
y = iris['species']

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Making predictions
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
print(f"Predicted class: {prediction[0]}")

# Saving the model
joblib.dump(model, 'iris_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Loading the model
loaded_model = joblib.load('iris_classifier.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Making a prediction with the loaded model
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
new_data = loaded_scaler.transform(new_data)
prediction = loaded_model.predict(new_data)
print(f"Predicted class with loaded model: {prediction[0]}")
