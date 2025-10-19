import joblib
import pandas as pd

def check_anomaly(power_reading, model):
    """
    Checks if a single power reading is an anomaly using the loaded model.
    Returns True if it's an anomaly, False otherwise.
    """
    # The model expects a 2D array as input, so we create a DataFrame
    input_df = pd.DataFrame({'Global_active_power': [power_reading]})
    
    # Predict returns -1 for an anomaly, 1 for normal
    prediction = model.predict(input_df)
    
    if prediction[0] == -1:
        return True
    else:
        return False

# --- Main part of the script ---
try:
    # 1. Load the saved Isolation Forest model
    anomaly_model = joblib.load('isolation_forest_model.pkl')
    print("✅ Anomaly detection model loaded successfully.")

    # 2. Test with some example power readings
    normal_reading = 0.9   # A typical reading in kW
    anomaly_reading = 15.0 # A huge, unusual spike in kW

    # Check the normal reading
    is_anomaly_1 = check_anomaly(normal_reading, anomaly_model)
    print(f"\nChecking a power reading of {normal_reading} kW...")
    print(f" -> Is it an anomaly? {is_anomaly_1}")

    # Check the anomaly reading
    is_anomaly_2 = check_anomaly(anomaly_reading, anomaly_model)
    print(f"\nChecking a power reading of {anomaly_reading} kW...")
    print(f" -> Is it an anomaly? {is_anomaly_2}")

except FileNotFoundError:
    print("❌ Error: 'isolation_forest_model.pkl' not found.")
    print("Please run the 'train_anomaly_model.py' script first.")