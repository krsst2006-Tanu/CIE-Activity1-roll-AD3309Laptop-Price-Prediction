# Laptop Price Prediction using Multiple Linear Regression
# Educational project: the included dataset is synthetic, not live market data.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load the dataset
data = pd.read_csv("laptop_data.csv")
print("First five rows:")
print(data.head())

# 2. Select input features (X) and target (y)
features = ["RAM_GB", "Storage_GB", "Processor_Score",
            "Screen_Size_Inches", "Weight_KG"]
X = data[features]
y = data["Price_INR"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict prices
y_pred = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\\nModel Evaluation")
print(f"MAE: ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R² Score: {r2:.3f}")

# 7. Plot actual vs predicted prices
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")
plt.title("Actual vs Predicted Laptop Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Predict the price of a new laptop
new_laptop = pd.DataFrame([{
    "RAM_GB": 16,
    "Storage_GB": 512,
    "Processor_Score": 4,
    "Screen_Size_Inches": 15.6,
    "Weight_KG": 1.8
}])
predicted_price = model.predict(new_laptop)[0]
print(f"\\nEstimated price of the new laptop: ₹{predicted_price:,.2f}")
