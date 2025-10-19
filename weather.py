# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import matplotlib.pyplot as plt

# def load_and_preprocess_data(filepath):
#     """Loads and preprocesses the energy consumption data in kWh."""
#     print("1. Loading and preprocessing energy data...")
#     df = pd.read_csv(filepath, delimiter=';', low_memory=False)
#     df['date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
#     df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
#     df.dropna(subset=['date_time', 'Global_active_power'], inplace=True)
#     df['Global_active_power'] = df['Global_active_power'] / 60
#     df.set_index('date_time', inplace=True)
#     df_daily = df['Global_active_power'].resample('D').sum().to_frame()
#     return df_daily

# def load_weather_data(filepath):
#     """Loads, cleans, and prepares the NEW weather data."""
#     print("1b. Loading and preprocessing NEW weather data...")
#     weather_df = pd.read_csv(filepath)
    
#     # --- ✅ ADAPTED FOR NEW FILE ---
#     # Rename columns to match what the rest of the script expects
#     weather_df.rename(columns={
#         'date': 'time',
#         'temp_mean_C': 'tavg',
#         'temp_min_C': 'tmin',
#         'temp_max_C': 'tmax',
#         'precipitation_mm': 'prcp',
#         'windspeed_10m_max_kmh': 'wspd'
#     }, inplace=True)
    
#     weather_df['time'] = pd.to_datetime(weather_df['time'])
#     weather_df.set_index('time', inplace=True)
    
#     # Select only the features we need
#     relevant_features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd']
#     weather_df = weather_df[relevant_features]
    
#     # In case there are any gaps, interpolate them
#     weather_df.interpolate(method='time', inplace=True)
#     return weather_df

# def create_features(df):
#     """Creates time-series features from the daily data."""
#     print("2. Engineering features...")
#     df_featured = df.copy()
    
#     df_featured['lag_1'] = df_featured['Global_active_power'].shift(1)
#     df_featured['lag_7'] = df_featured['Global_active_power'].shift(7)
    
#     # --- ✅ RESTORED POWERFUL FEATURE ---
#     # With a complete dataset, we can use the yearly lag again.
#     df_featured['lag_365'] = df_featured['Global_active_power'].shift(365)
    
#     # Using min_periods=1 to handle any gaps within the energy data itself
#     df_featured['rolling_30_mean'] = df_featured['Global_active_power'].rolling(window=30, min_periods=1).mean()
#     df_featured['rolling_30_sum'] = df_featured['Global_active_power'].rolling(window=30, min_periods=1).sum()
    
#     df_featured['month'] = df_featured.index.month
#     df_featured['quarter'] = df_featured.index.quarter
#     df_featured['dayofweek'] = df_featured.index.dayofweek
#     df_featured['dayofyear'] = df_featured.index.dayofyear
    
#     # Using modern syntax to fill NaNs created by lags/rolling
#     df_featured.bfill(inplace=True)
    
#     return df_featured

# def create_target_variable(df):
#     """Creates the target variable robustly."""
#     print("3. Creating target variable...")
#     df_final = df.copy()
    
#     # Using min_periods=1 to handle potential gaps in the energy data
#     df_final['30_day_future_consumption'] = df_final['Global_active_power'].rolling(window=30, min_periods=1).sum().shift(-30)
    
#     # Drop only the final rows that have no future
#     df_final.dropna(inplace=True)
#     return df_final

# def plot_results(y_true, y_pred, title):
#     """Plots actual vs. predicted values."""
#     plt.figure(figsize=(15, 6))
#     results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred}, index=y_true.index)
#     plt.plot(results.index, results['Actual'], label='Actual Future Consumption', marker='o', linestyle='-', alpha=0.7)
#     plt.plot(results.index, results['Predicted'], label='Predicted Future Consumption', marker='x', linestyle='--', alpha=0.7)
#     plt.title(title)
#     plt.xlabel('Date')
#     plt.ylabel('Total 30-Day Consumption (kWh)')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# def main():
#     """Main function to run the forecasting pipeline."""
#     df_daily = load_and_preprocess_data('EnergyDataset.txt')
#     # Make sure to use the name of your NEW weather file here
#     df_weather = load_weather_data('new_weather_data.csv') 

#     df_merged = df_daily.join(df_weather, how='inner')
    
#     df_featured = create_features(df_merged)
#     df_with_target = create_target_variable(df_featured)

#     if df_with_target.empty:
#         print("\nCRITICAL ERROR: Dataframe is empty. Check data source files for gaps.")
#         return

#     X = df_with_target.drop(['30_day_future_consumption', 'Global_active_power'], axis=1)
#     y = df_with_target['30_day_future_consumption']

#     split_date = y.index.max() - pd.DateOffset(days=180)
    
#     X_train = X.loc[X.index <= split_date]
#     y_train = y.loc[y.index <= split_date]
#     X_test = X.loc[X.index > split_date]
#     y_test = y.loc[y.index > split_date]
    
#     if X_train.empty or X_test.empty:
#         print("\nCRITICAL ERROR: Training or test set is empty after splitting.")
#         return

#     print(f"\nTraining data shape: {X_train.shape}")
#     print(f"Test data shape: {X_test.shape}")

#     print("\nTraining XGBoost model...")
#     reg = xgb.XGBRegressor(
#         objective='reg:squarederror',
#         n_estimators=1000,
#         learning_rate=0.05,
#         max_depth=5,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         early_stopping_rounds=10
#     )
#     reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
#     print("Training complete.")

#     print("\nEvaluating model on the test set...")
#     predictions = reg.predict(X_test)
#     mae = mean_absolute_error(y_test, predictions)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
#     r2 = r2_score(y_test, predictions)

#     print("\n--- Model Evaluation Metrics ---")
#     print(f"Mean Absolute Error (MAE): {mae:,.2f} kWh")
#     print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} kWh")
#     print(f"R-squared (R²) Score: {r2:.2f}")

#     print("\nPlotting results...")
#     plot_results(y_test, predictions, 'Hold-Out Test Set: Actual vs. Predicted (with Weather Data)')
#     plt.figure(figsize=(10, 8))
#     xgb.plot_importance(reg, height=0.9)
#     plt.title('Feature Importance')
#     plt.tight_layout()
#     plt.show()

#     print("\nScript finished. Press Enter to exit.")
#     input()

# if __name__ == '__main__':
#     main()


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the energy consumption data in kWh."""
    print("1. Loading and preprocessing energy data...")
    df = pd.read_csv(filepath, delimiter=';', low_memory=False)
    df['date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df.dropna(subset=['date_time', 'Global_active_power'], inplace=True)
    df['Global_active_power'] = df['Global_active_power'] / 60
    df.set_index('date_time', inplace=True)
    df_daily = df['Global_active_power'].resample('D').sum().to_frame()
    return df_daily

def load_weather_data(filepath):
    """Loads, cleans, and prepares the NEW weather data."""
    print("1b. Loading and preprocessing NEW weather data...")
    weather_df = pd.read_csv(filepath)
    
    weather_df.rename(columns={
        'date': 'time',
        'temp_mean_C': 'tavg',
        'temp_min_C': 'tmin',
        'temp_max_C': 'tmax',
        'precipitation_mm': 'prcp',
        'windspeed_10m_max_kmh': 'wspd'
    }, inplace=True)
    
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df.set_index('time', inplace=True)
    
    relevant_features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd']
    weather_df = weather_df[relevant_features]
    
    weather_df.interpolate(method='time', inplace=True)
    return weather_df

def create_features(df):
    """Creates time-series features from the daily data."""
    print("2. Engineering features...")
    df_featured = df.copy()
    
    df_featured['lag_1'] = df_featured['Global_active_power'].shift(1)
    df_featured['lag_7'] = df_featured['Global_active_power'].shift(7)
    df_featured['lag_365'] = df_featured['Global_active_power'].shift(365)
    
    df_featured['rolling_30_mean'] = df_featured['Global_active_power'].rolling(window=30, min_periods=1).mean()
    df_featured['rolling_30_sum'] = df_featured['Global_active_power'].rolling(window=30, min_periods=1).sum()
    
    df_featured['month'] = df_featured.index.month
    df_featured['quarter'] = df_featured.index.quarter
    df_featured['dayofweek'] = df_featured.index.dayofweek
    df_featured['dayofyear'] = df_featured.index.dayofyear
    
    df_featured.bfill(inplace=True)
    return df_featured

def create_target_variable(df):
    """Creates the target variable robustly."""
    print("3. Creating target variable...")
    df_final = df.copy()
    
    df_final['30_day_future_consumption'] = df_final['Global_active_power'].rolling(window=30, min_periods=1).sum().shift(-30)
    
    df_final.dropna(inplace=True)
    return df_final

def plot_results(y_true, y_pred, title):
    """Plots actual vs. predicted values."""
    plt.figure(figsize=(15, 6))
    results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred}, index=y_true.index)
    plt.plot(results.index, results['Actual'], label='Actual Future Consumption', marker='o', linestyle='-', alpha=0.7)
    plt.plot(results.index, results['Predicted'], label='Predicted Future Consumption', marker='x', linestyle='--', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Total 30-Day Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the forecasting pipeline."""
    df_daily = load_and_preprocess_data('EnergyDataset.txt')
    df_weather = load_weather_data('new_weather_data.csv') # Use your new weather file name

    df_merged = df_daily.join(df_weather, how='inner')
    
    df_featured = create_features(df_merged)
    df_with_target = create_target_variable(df_featured)

    if df_with_target.empty:
        print("\nCRITICAL ERROR: Dataframe is empty. Check data source files for gaps.")
        return

    X = df_with_target.drop(['30_day_future_consumption', 'Global_active_power'], axis=1)
    y = df_with_target['30_day_future_consumption']

    split_date = y.index.max() - pd.DateOffset(days=180)
    
    X_train = X.loc[X.index <= split_date]
    y_train = y.loc[y.index <= split_date]
    X_test = X.loc[X.index > split_date]
    y_test = y.loc[y.index > split_date]
    
    if X_train.empty or X_test.empty:
        print("\nCRITICAL ERROR: Training or test set is empty after splitting.")
        return

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    print("\nTraining XGBoost model...")
    reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=10
    )
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
    print("Training complete.")
    import joblib

    # ... your existing code ...

    # After this line in your main function:
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
    print("Training complete.")

    # --- Save the trained model to a file ---
    joblib.dump(reg, 'xgb_forecaster.pkl')
    print("Model saved successfully as xgb_forecaster.pkl")

    print("\nEvaluating model on the test set...")
    predictions = reg.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:,.2f} kWh")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} kWh")
    print(f"R-squared (R²) Score: {r2:.2f}")

    # --- ✅ NEW CODE BLOCK STARTS HERE ---
    # Calculate total consumption over the test period
    actual_total_test_period = df_daily.loc[y_test.index.min():y_test.index.max()]['Global_active_power'].sum()
    
    # This is a flawed approximation, but included as requested
    daily_predicted = predictions / 30
    predicted_total_test_period = daily_predicted.sum()
    
    # print(f"\nActual Total Consumption (Test Period): {actual_total_test_period:,.2f} kWh")
    # print(f"Predicted Total Consumption (Test Period, approx.): {predicted_total_test_period:,.2f} kWh")

    # Calculate consumption for the last 30 days of the test period
    last_30_days_actual = df_daily.loc[y_test.index[-30]:y_test.index[-1]]['Global_active_power'].sum()
    last_30_days_predicted = daily_predicted[-30:].sum()

    print("\n--- Last 30-Day Consumption Comparison ---")
    print(f"Actual Last 30-Day Consumption: {last_30_days_actual:.2f} kWh")
    print(f"Predicted Last 30-Day Consumption: {last_30_days_predicted:.2f} kWh")
    # --- ✅ NEW CODE BLOCK ENDS HERE ---
    
    print("\nPlotting results...")
    plot_results(y_test, predictions, 'Hold-Out Test Set: Actual vs. Predicted (with Weather Data)')
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(reg, height=0.9)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    print("\nScript finished. Press Enter to exit.")
    input()

if __name__ == '__main__':
    main()