# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Print the input features (independent variables)
print("Input Features (X):")
print(X)

# Encoding categorical data (for the State column)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# The OneHotEncoder will transform the categorical 'State' column into numerical values
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Print the transformed input features after encoding categorical data
print("Encoded Input Features (X):")
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# Using 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression

# Create the regressor and fit it to the training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Set the number of decimal places to 2 for printing
np.set_printoptions(precision=2)

# Concatenate the predicted and actual values for comparison
# Print the predicted and actual values side by side
print("Predicted vs Actual Values:")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))
