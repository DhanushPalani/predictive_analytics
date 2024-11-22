## Ex.No.3 Predictive Analytics

## Aim:
```
To build and evaluate predictive models using Linear Regression, Logistic Regression, and Decision Trees.
To interpret regression coefficients and assess classification model accuracy using confusion matrices
```
## Tools Required
```
Python: Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
```
## Part A: Regression Analysis
## Procedure:

## 1.Build a Linear Regression Model

Predict a target variable based on an independent variable.
Train and test the model, and compute regression coefficients.
## 2.Evaluate the Model

Use Mean Squared Error (MSE) and R-squared metrics for evaluation.
## 3.Visualize Regression Line

Plot data points and the regression line.
## code
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Example dataset (predicting 'y' based on 'x')
data = {
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'y': [2, 4, 5, 4, 5, 7, 8, 9, 10, 12]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the predictor and target variables
X = df[['x']]  # Predictor variable (independent)
y = df['y']    # Target variable (dependent)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target variable using the test set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output regression coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Model evaluation results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plotting the regression line
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/6df4e82f-2744-4826-90d1-b51f87dab77c)
## Explaination:
```
1.Linear Regression Model: The model assumes a linear relationship between the predictor variable (x) and the target variable (y). The equation of the regression line is y = b₀ + b₁ * x.
2.Intercept and Coefficient: The intercept is where the line crosses the y-axis, and the coefficient indicates how much y increases for each unit increase in x.
3.Model Evaluation: MSE helps to evaluate the error between the actual and predicted values, and R-squared helps to determine the proportion of variance explained by the model.
```
## Part B: Classification Models
## Procedure:

## 1.mplement Logistic Regression and Decision Trees

Train and test models using the Iris dataset.
## 2.Evaluate Models

Use confusion matrices and accuracy scores.
## 3.Visualize Results

Display confusion matrices as heatmaps.
## code:
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Example dataset (Iris dataset for classification)
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Decision Tree Model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

# Confusion Matrix for Logistic Regression
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

# Confusion Matrix for Decision Tree
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Output confusion matrices and accuracies
print("Logistic Regression Accuracy:", accuracy_log_reg)
print("Decision Tree Accuracy:", accuracy_dt)

print("\nConfusion Matrix for Logistic Regression:")
print(conf_matrix_log_reg)

print("\nConfusion Matrix for Decision Tree:")
print(conf_matrix_dt)

# Plot Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0])
ax[0].set_title('Logistic Regression Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1])
ax[1].set_title('Decision Tree Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
```
## output:
![image](https://github.com/user-attachments/assets/f9f42a44-54ef-4917-b1e1-27635e5fdb10)
## Confusion Matrix Visualization
![image](https://github.com/user-attachments/assets/67fd4f26-24e4-411e-b60b-6c9a673923c8)

## Result:
```
A Linear Regression model was built, coefficients interpreted, and model evaluation performed using MSE and R-squared.
Logistic Regression and Decision Trees were implemented for classification, with performance assessed via accuracy and confusion matrices.
Visualization tools like scatter plots and heatmaps effectively communicated results.
```



