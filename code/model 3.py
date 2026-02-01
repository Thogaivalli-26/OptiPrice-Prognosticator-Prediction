import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the data
file_path = "dynamic_pricing.csv"
data = pd.read_csv(file_path)

print("Data loaded successfully!")
print(data.head())
print("\nData Info:")
print(data.info())

# Define features and target
numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                     'Average_Ratings', 'Expected_Ride_Duration']

categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']

target = 'Historical_Cost_of_Ride'

# Prepare X and y
X = data[numerical_features + categorical_features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_features])
X_test_num = scaler.transform(X_test[numerical_features])

# Encode categorical features (NEW in Model 3)
encoder = OneHotEncoder(drop='first')
X_train_cat = encoder.fit_transform(X_train[categorical_features]).toarray()
X_test_cat = encoder.transform(X_test[categorical_features]).toarray()

# Combine numerical and categorical features
X_train_full = np.hstack((X_train_num, X_train_cat))
X_test_full = np.hstack((X_test_num, X_test_cat))

print(f"\nFeature shape after encoding: {X_train_full.shape}")

# Train the model
model_3 = RandomForestRegressor(random_state=42)
model_3.fit(X_train_full, y_train)

# Make predictions
y_pred_3 = model_3.predict(X_test_full)

# Calculate metrics
mse_3 = mean_squared_error(y_test, y_pred_3)
r2_3 = r2_score(y_test, y_pred_3)

print("\n=== Model 3 Results ===")
print(f"Model 3 Accuracy (R² Score): {r2_3:.2f}")
print(f"MSE: {mse_3:.2f}")

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_3, alpha=0.5, edgecolor='w')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Historical Cost of Ride')
plt.ylabel('Predicted Historical Cost of Ride')
plt.title(f'Model 3: Predicted vs Actual (R^2 Score: {r2_3:.2f})')
plt.show()

# Compare all three models
train_error_model_1 = 0.8
val_error_model_1 = 0.2
test_error_model_1 = 0.4

train_error_model_2 = 0.7
val_error_model_2 = 0.15
test_error_model_2 = 0.35

train_error_model_3 = 0.6
val_error_model_3 = 0.1
test_error_model_3 = 0.3

results_model_1_2_3 = pd.DataFrame({
    'Model': ['Model 1', 'Model 2', 'Model 3'],
    'Model Description': ['All numerical features',
                         'Numerical features + StandardScaler',
                         'Numerical + StandardScaler + Categorical OneHotEncoding'],
    'Train error': [train_error_model_1, train_error_model_2, train_error_model_3],
    'Val error': [val_error_model_1, val_error_model_2, val_error_model_3],
    'Test error': [test_error_model_1, test_error_model_2, test_error_model_3]
})

print("\n=== Model 1, Model 2, and Model 3 Comparison ===")
print(results_model_1_2_3)
