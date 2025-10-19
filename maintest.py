import joblib
import pandas as pd

# 1. Load the saved model
try:
    loaded_model = joblib.load('xgb_forecaster.pkl')
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'xgb_forecaster.pkl' not found. Make sure you've run the training script first.")
    exit()

# 2. ✅ THE FIX IS HERE
# Reorder this list to exactly match the order from the error message.
# The model expects weather features first.
feature_columns = [
    'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'lag_1', 'lag_7', 'lag_365',
    'rolling_30_mean', 'rolling_30_sum', 'month', 'quarter', 'dayofweek', 'dayofyear'
]

# 3. Create sample data for one prediction
# The order here doesn't matter, as we'll use the list above to set the final order.
sample_data = {
    'lag_1': [25.5],
    'lag_7': [28.1],
    'lag_365': [30.2],
    'rolling_30_mean': [27.8],
    'rolling_30_sum': [834.0],
    'month': [10],
    'quarter': [4],
    'dayofweek': [6],
    'dayofyear': [293],
    'tavg': [28.5],
    'tmin': [24.0],
    'tmax': [32.0],
    'prcp': [5.0],
    'wspd': [12.0]
}

# 4. Create the DataFrame WITH THE CORRECT COLUMN ORDER
input_df = pd.DataFrame(sample_data, columns=feature_columns)

print("\n📦 Created sample input DataFrame (with correct column order):")
print(input_df)

# 5. Make a prediction using the loaded model
prediction = loaded_model.predict(input_df)

predicted_consumption = prediction[0]

print(f"\n💡 Prediction Complete!")
print(f"Predicted total consumption for the next 30 days: {predicted_consumption:,.2f} kWh")