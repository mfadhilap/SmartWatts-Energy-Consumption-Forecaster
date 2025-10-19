import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1. Create Sample High-Frequency Data
# Imagine this is your normal minute-by-minute data (e.g., between 1-3 kW)
normal_usage = np.random.normal(loc=2, scale=0.5, size=300)

# Create a clear anomaly (e.g., a huge spike to 15 kW)
anomaly = np.array([15.0])

# Combine them into a single dataset
usage_data = np.concatenate([normal_usage[:150], anomaly, normal_usage[150:]])
df = pd.DataFrame(usage_data, columns=['power_kw'])

# 2. Train the Isolation Forest Model
# The 'contamination' parameter tells the model what proportion of the data is expected to be anomalous.
# 'auto' is a good starting point.
# Tell the model that anomalies are rare (e.g., 1% of the data)
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(df[['power_kw']])

# 3. Predict Anomalies
# The model returns 1 for normal points and -1 for anomalies.
df['anomaly'] = model.predict(df[['power_kw']])

# 4. Show the Results
anomalies = df[df['anomaly'] == -1]
print("Anomalies detected at the following points:")
print(anomalies)

# Plot the data to visualize the detected anomaly
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['power_kw'], label='Energy Usage (kW)')
plt.scatter(anomalies.index, anomalies['power_kw'], color='red', s=100, label='Anomaly Detected', zorder=5)
plt.title('Anomaly Detection with Isolation Forest')
plt.xlabel('Time (Minutes)')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid(True)
plt.show()


# import pandas as pd
# from sklearn.ensemble import IsolationForest
# import joblib

# def train_and_save_anomaly_model(filepath):
#     """
#     Loads high-frequency energy data, trains an Isolation Forest model,
#     and saves it to a file.
#     """
#     print("1. Loading high-frequency energy data...")
    
#     # Load the original dataset
#     df = pd.read_csv(filepath, delimiter=';', low_memory=False)
    
#     # --- Data Preparation ---
#     # Convert 'Global_active_power' to a numeric type, forcing errors to NaN
#     df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    
#     # Drop any rows where the power reading is missing
#     df.dropna(subset=['Global_active_power'], inplace=True)
    
#     # Select only the feature we need for training
#     X_train = df[['Global_active_power']]
    
#     print(f"   -> Data loaded successfully. Training on {len(X_train)} data points.")

#     # --- Model Training ---
#     print("2. Training Isolation Forest model...")
#     # We use contamination=0.01 because we've determined it works well.
#     # n_jobs=-1 uses all available CPU cores to speed up training.
#     model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    
#     model.fit(X_train)
#     print("   -> Model training complete.")

#     # --- Save the Model ---
#     print("3. Saving model to file...")
#     joblib.dump(model, 'isolation_forest_model.pkl')
#     print("   -> ✅ Model saved successfully as 'isolation_forest_model.pkl'")


# if __name__ == '__main__':
#     # Make sure to use the correct path to your dataset
#     train_and_save_anomaly_model('EnergyDataset.txt')