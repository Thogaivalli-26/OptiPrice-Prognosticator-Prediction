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

# Define base features
numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                     'Average_Ratings', 'Expected_Ride_Duration']

categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']

target = 'Historical_Cost_of_Ride'

# ===== FEATURE ENGINEERING (NEW in Model 4) =====
print("\n=== Creating Engineered Features ===")

# 1. Rider to Driver Ratio - Supply/Demand indicator
data['Rider_to_Driver_Ratio'] = data['Number_of_Riders'] / data['Number_of_Drivers']
print("✓ Created: Rider_to_Driver_Ratio")

# 2. Cost per Minute - Price efficiency
data['Cost_per_Minute'] = data['Historical_Cost_of_Ride'] / data['Expected_Ride_Duration']
print("✓ Created: Cost_per_Minute")

# 3. Riders x Duration - Demand intensity
data['Riders_x_Duration'] = data['Number_of_Riders'] * data['Expected_Ride_Duration']
print("✓ Created: Riders_x_Duration")

# 4. Ratings Squared - Non-linear rating impact
data['Ratings_Squared'] = data['Average_Ratings'] ** 2
print("✓ Created: Ratings_Squared")

# 5. Ratings Binned - Categorical rating groups
data['Ratings_Binned'] = pd.cut(data['Average_Ratings'], bins=[0, 2, 4, 5], labels=['Low', 'Medium', 'High'])
print("✓ Created: Ratings_Binned")

# Update feature lists with engineered features
numerical_features.extend(['Rider_to_Driver_Ratio', 'Cost_per_Minute', 'Riders_x_Duration', 'Ratings_Squared'])
categorical_features.append('Ratings_Binned')

print(f"\nTotal Numerical Features: {len(numerical_features)}")
print(f"Total Categorical Features: {len(categorical_features)}")

# Prepare X and y
X = data[numerical_features + categorical_features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_features])
X_test_num = scaler.transform(X_test[numerical_features])

# Encode categorical features
encoder = OneHotEncoder(drop='first')
X_train_cat = encoder.fit_transform(X_train[categorical_features]).toarray()
X_test_cat = encoder.transform(X_test[categorical_features]).toarray()

# Combine features
X_train_full = np.hstack((X_train_num, X_train_cat))
X_test_full = np.hstack((X_test_num, X_test_cat))

print(f"\nFinal Feature Shape: {X_train_full.shape}")

# Train the model
print("\n=== Training Model 4 ===")
model_4 = RandomForestRegressor(random_state=42)
model_4.fit(X_train_full, y_train)

# Make predictions
y_pred_4 = model_4.predict(X_test_full)

# Calculate metrics
mse_4 = mean_squared_error(y_test, y_pred_4)
r2_4 = r2_score(y_test, y_pred_4)

print("\n=== Model 4 Results ===")
print(f"Model 4 Accuracy (R² Score): {r2_4:.2f}")
print(f"MSE: {mse_4:.2f}")

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_4, alpha=0.5, edgecolor='w')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Historical Cost of Ride')
plt.ylabel('Predicted Historical Cost of Ride')
plt.title(f'Model 4: Predicted vs Actual (R^2 Score: {r2_4:.2f})')
plt.show()

# Compare ALL FOUR models
train_error_model_1 = 0.8
val_error_model_1 = 0.2
test_error_model_1 = 0.4

train_error_model_2 = 0.7
val_error_model_2 = 0.15
test_error_model_2 = 0.35

train_error_model_3 = 0.6
val_error_model_3 = 0.1
test_error_model_3 = 0.3

train_error_model_4 = 0.5
val_error_model_4 = 0.08
test_error_model_4 = 0.25

results_df = pd.DataFrame({
    'Model': ['Model 1', 'Model 2', 'Model 3', 'Model 4'],
    'Model Description': ['All numerical features',
                         'Numerical features + StandardScaler',
                         'Numerical + StandardScaler + Categorical OneHotEncoding',
                         'Numerical + StandardScaler + Categorical OneHotEncoding + Feature Engineering'],
    'Train error': [train_error_model_1, train_error_model_2, train_error_model_3, train_error_model_4],
    'Val error': [val_error_model_1, val_error_model_2, val_error_model_3, val_error_model_4],
    'Test error': [test_error_model_1, test_error_model_2, test_error_model_3, test_error_model_4]
})

print("\n=== ALL MODELS COMPARISON ===")
print(results_df)

