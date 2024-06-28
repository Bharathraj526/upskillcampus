import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Collection and Preparation
crop_data = pd.read_csv('/path/to/your/agricultural_data.csv')  # Replace with actual path

# Example: Data preprocessing and feature engineering
crop_data['Is_Rainy_Season'] = crop_data['Rainfall'].apply(lambda x: 1 if x > threshold else 0)  # Example feature engineering

# Step 2: Data Preprocessing
X = crop_data[['Rainfall', 'Temperature', 'Fertilizer_Usage']]
y = crop_data['Crop_Yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 4: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

# Step 5: Visualization and Reporting
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Crop Yield')
plt.ylabel('Predicted Crop Yield')
plt.title('Actual vs Predicted Crop Yield')
plt.show()

print("Key Findings:")
print(f"1. Mean squared error of the crop yield prediction model: {mse:.2f}")
print("2. Coefficients of the linear regression model indicating the impact of each feature on crop yield:")
print(coefficients)
