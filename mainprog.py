import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the energy consumption data in kWh."""
    print("1. Loading and preprocessing data...")
    
    # FIX: The file must be read before the 'df' variable can be used.
    df = pd.read_csv(filepath, delimiter=';', low_memory=False)
    
    # Combine Date and Time into a single datetime column
    df['date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    
    # Convert power column to numeric, coercing errors to NaN
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    
    # Drop rows where datetime or power is missing
    df.dropna(subset=['date_time', 'Global_active_power'], inplace=True)

    # Convert from kW per minute to kWh (kW * (1/60) hours)
    df['Global_active_power'] = df['Global_active_power'] / 60

    df.set_index('date_time', inplace=True)
    
    # Resample to daily frequency and sum the consumption (kWh)
    df_daily = df['Global_active_power'].resample('D').sum().to_frame()
    print("Data loading complete.")
    return df_daily

def create_features(df):
    """Creates time-series features from the daily data."""
    print("2. Engineering features...")
    df_featured = df.copy()
    
    # Lag features
    df_featured['lag_1'] = df_featured['Global_active_power'].shift(1)
    df_featured['lag_7'] = df_featured['Global_active_power'].shift(7)
    df_featured['lag_365'] = df_featured['Global_active_power'].shift(365)
    
    # Rolling statistics
    df_featured['rolling_30_mean'] = df_featured['Global_active_power'].rolling(window=30).mean()
    df_featured['rolling_30_sum'] = df_featured['Global_active_power'].rolling(window=30).sum()
    
    # Calendar features
    df_featured['month'] = df_featured.index.month
    df_featured['quarter'] = df_featured.index.quarter
    df_featured['dayofweek'] = df_featured.index.dayofweek
    df_featured['dayofyear'] = df_featured.index.dayofyear
    print("Feature engineering complete.")
    return df_featured

def create_target_variable(df):
    """Creates the target variable for future 30-day consumption."""
    print("3. Creating target variable...")
    df_final = df.copy()
    
    # Target is the sum of consumption over the next 30 days
    df_final['30_day_future_consumption'] = df_final['Global_active_power'].rolling(window=30).sum().shift(-30)
    
    # Drop NaNs created by lagging features and shifting the target
    df_final = df_final.dropna()
    print("Target variable created.")
    return df_final

def plot_results(y_true, y_pred, title):
    """Plots actual vs. predicted values."""
    plt.figure(figsize=(15, 6))
    results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
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
    df_featured = create_features(df_daily)
    df_final = create_target_variable(df_featured)

    # Time-based split: last 365 days for testing
    split_date = df_final.index.max() - pd.DateOffset(days=365)
    train_df = df_final.loc[df_final.index <= split_date]
    test_df = df_final.loc[df_final.index > split_date]

    # Define features (X) and target (y)
    X_train = train_df.drop(['30_day_future_consumption', 'Global_active_power'], axis=1)
    y_train = train_df['30_day_future_consumption']
    X_test = test_df.drop(['30_day_future_consumption', 'Global_active_power'], axis=1)
    y_test = test_df['30_day_future_consumption']

    # XGBoost requires feature names to be strings
    X_train.columns = [str(col) for col in X_train.columns]
    X_test.columns = [str(col) for col in X_test.columns]

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Model training
    print("\n4. Training XGBoost model...")
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

    # Evaluation
    print("\n5. Evaluating model on the test set...")
    predictions = reg.predict(X_test)

    # Calculate standard regression metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Total Consumption Comparison
    actual_total_daily = df_daily.loc[test_df.index[0]:test_df.index[-1]]['Global_active_power'].sum()
    
    daily_predicted = predictions / 30 
    predicted_total_daily = daily_predicted.sum()

    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:,.2f} kWh")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} kWh")
    print(f"R-squared (R²) Score: {r2:.2f}")
    print(f"Actual Total Consumption (Test Period): {actual_total_daily:,.2f} kWh")
    print(f"Predicted Total Consumption (Test Period, approx.): {predicted_total_daily:,.2f} kWh")

    # Last 30-Day Consumption Comparison
    last_30_days_actual = df_daily.loc[test_df.index[-30]:test_df.index[-1]]['Global_active_power'].sum()
    last_30_days_predicted = daily_predicted[-30:].sum()

    print("\n--- Last 30-Day Consumption Comparison ---")
    print(f"Actual Last 30-Day Consumption: {last_30_days_actual:.2f} kWh")
    print(f"Predicted Last 30-Day Consumption: {last_30_days_predicted:.2f} kWh")

    # Visualization
    print("\n6. Plotting results...")
    plot_results(y_test, predictions, 'Hold-Out Test Set: Actual vs. Predicted 30-Day Consumption')

    plt.figure(figsize=(10, 8))
    xgb.plot_importance(reg, height=0.9)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()