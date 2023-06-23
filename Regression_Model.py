import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Datasets/NLSY97_subset.csv')  # Replace 'your_dataset.csv' with the actual filename

# Select the relevant columns for the regression
columns = ['EXP', 'FEMALE', 'MALE', 'AGE', 'EDUCPROF', 'EDUCPHD', 'EDUCMAST', 'EDUCBA']
X = data[columns]
y = data['EARNINGS']

# Fill missing values with column means
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit the multivariable regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())

# Get the predicted values from the model
predicted_salaries = model.predict(X)

# Create a scatter plot of actual vs. predicted salaries
plt.scatter(y, predicted_salaries)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Add a reference line for perfect predictions
plt.xlabel('Actual Earnings')
plt.ylabel('Predicted Earnings')
plt.title('Actual vs. Predicted Earnings')
plt.savefig('Images/Regression Results/Final Results.png')
