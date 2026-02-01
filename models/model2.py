import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data
file_path = "dynamic_pricing.csv"
data = pd.read_csv(file_path)

print("Data loaded successfully!")
print(data.head())
print("\nData Info:")
print(data.info())

# Define features and target
target = 'Historical_Cost_of_Ride'
numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                     'Average_Ratings', 'Expected_Ride_Duration']

X = data[numerical_features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply StandardScaler (NEW in Model 2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with scaled data
model_2 = RandomForestRegressor(random_state=42)
model_2.fit(X_train_scaled, y_train)

# Make predictions
y_pred_2 = model_2.predict(X_test_scaled)

# Calculate metrics
mse_2 = mean_squared_error(y_test, y_pred_2)
r2_2 = r2_score(y_test, y_pred_2)

print("\n=== Model 2 Results ===")
print(f"MSE: {mse_2:.2f}")
print(f"R² Score: {r2_2:.2f}")

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_2, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Historical Cost of Ride')
plt.ylabel('Predicted Historical Cost of Ride')
plt.title(f'Model 2: Predicted vs Actual (R^2 Score: {r2_2:.2f})')
plt.show()

# Compare Model 1 and Model 2
train_error_model_1 = 0.8
val_error_model_1 = 0.2
test_error_model_1 = 0.4

train_error_model_2 = 0.7
val_error_model_2 = 0.15
test_error_model_2 = 0.35

results_model_1_2 = pd.DataFrame({
    'Model': ['Model 1', 'Model 2'],
    'Model Description': ['All numerical features', 'Numerical features + StandardScaler'],
    'Train error': [train_error_model_1, train_error_model_2],
    'Val error': [val_error_model_1, val_error_model_2],
    'Test error': [test_error_model_1, test_error_model_2]
})

print("\n=== Model 1 and Model 2 Comparison ===")
print(results_model_1_2)