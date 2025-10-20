# import streamlit as st
# import pandas as pd
# import numpy as np
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# import time
# import collections

# # --- PAGE CONFIGURATION ---
# st.set_page_config(
#     page_title="Real-Time LSTM Forecaster Demo",
#     page_icon="🧠",
#     layout="wide"
# )

# # --- MODEL AND DATA LOADING (CACHED) ---
# @st.cache_resource
# def load_keras_model():
#     """Load the trained Keras LSTM model."""
#     try:
#         model = load_model('lstm_model.keras')
#         return model
#     except (IOError, FileNotFoundError):
#         return None

# @st.cache_data
# def load_and_prepare_data():
#     """
#     Loads the high-frequency data and fits the scaler, replicating the training process.
#     """
#     try:
#         # Load the full dataset to fit the scaler correctly
#         df = pd.read_csv('EnergyDataset.txt', delimiter=';', low_memory=False)
#         df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
#         df.dropna(subset=['Global_active_power'], inplace=True)

#         # Fit the scaler on the entire dataset's power values, just like in training
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         dataset = df.Global_active_power.values.astype('float32').reshape(-1, 1)
#         scaler.fit(dataset)

#         # Now, load the data again but with a datetime index for simulation
#         df['date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
#         df.dropna(subset=['date_time'], inplace=True)
#         df.set_index('date_time', inplace=True)
        
#         return df, scaler
#     except FileNotFoundError:
#         return None, None

# # --- APP LAYOUT ---
# st.title("🧠 Real-Time LSTM Power Forecaster")
# st.markdown("A demonstration by **Fadhil**. This app simulates a live data stream to showcase the LSTM model's ability to predict the next minute's energy consumption.")

# # Load model and data
# lstm_model = load_keras_model()
# df, scaler = load_and_prepare_data()

# if lstm_model is None or df is None:
#     st.error("Error: `lstm_model.keras` or `EnergyDataset.txt` not found. Please ensure all files are in the directory.")
# else:
#     # --- SIMULATION CONTROLS ---
#     st.sidebar.header("Simulation Controls")
    
#     # Dynamically find the last two months available in the data
#     available_months = df.index.to_period('M').unique()
#     available_months_str = sorted([month.strftime('%Y-%m') for month in available_months])
    
#     selected_month = st.sidebar.selectbox(
#         "Choose a month to simulate:",
#         options=available_months_str,
#         index=len(available_months_str) - 2 # Default to the second to last month
#     )
    
#     simulation_speed = st.sidebar.slider("Simulation Speed (delay between minutes)", 0.0, 0.5, 0.05, 0.01)

#     if st.sidebar.button("Start/Reset Simulation", type="primary"):
#         st.session_state.run_simulation = True
#         st.session_state.simulation_month = selected_month

#     # --- SIMULATION LOGIC ---
#     if 'run_simulation' in st.session_state and st.session_state.run_simulation:
        
#         # Filter data for the selected month
#         minute_data_to_simulate = df.loc[st.session_state.simulation_month]['Global_active_power']
        
#         # --- LAYOUT PLACEHOLDERS ---
#         st.header(f"Live Simulation for: {pd.to_datetime(st.session_state.simulation_month).strftime('%B %Y')}")
        
#         col1, col2, col3 = st.columns(3)
#         placeholder_actual = col1.empty()
#         placeholder_predicted = col2.empty()
#         placeholder_error = col3.empty()
        
#         chart_placeholder = st.empty()
#         progress_bar = st.progress(0, text="Initializing simulation...")

#         # Initialize data structures for the simulation
#         history = collections.deque(maxlen=30) # A queue that automatically keeps the last 30 items
#         chart_data = pd.DataFrame(columns=["Actual", "Predicted"])

#         # "Warm up" the history buffer with the 30 minutes prior to the simulation start
#         warm_up_start = minute_data_to_simulate.index[0] - pd.DateOffset(minutes=30)
#         warm_up_data = df.loc[warm_up_start:minute_data_to_simulate.index[0]-pd.DateOffset(minutes=1), 'Global_active_power']
#         for val in warm_up_data:
#             history.append(val)

#         # Main simulation loop
#         for i, (timestamp, actual_value) in enumerate(minute_data_to_simulate.items()):
            
#             # --- PREDICTION STEP ---
#             # 1. Scale the input history
#             input_sequence = np.array(list(history)).reshape(-1, 1)
#             scaled_input = scaler.transform(input_sequence)
            
#             # 2. Reshape for LSTM: [samples, timesteps, features]
#             reshaped_input = np.reshape(scaled_input, (1, 1, 30))
            
#             # 3. Predict
#             prediction_scaled = lstm_model.predict(reshaped_input, verbose=0)
            
#             # 4. Inverse scale the prediction to get real kW value
#             prediction_kw = scaler.inverse_transform(prediction_scaled)[0][0]

#             # --- UPDATE UI ---
#             placeholder_actual.metric("Actual Power", f"{actual_value:.2f} kW")
#             placeholder_predicted.metric("Predicted Power", f"{prediction_kw:.2f} kW")
#             error = actual_value - prediction_kw
#             placeholder_error.metric("Difference", f"{error:.2f} kW")

#             # Update chart data
#             new_row = pd.DataFrame({"Actual": [actual_value], "Predicted": [prediction_kw]}, index=[timestamp])
#             chart_data = pd.concat([chart_data, new_row])
#             chart_placeholder.line_chart(chart_data)

#             # Update progress bar
#             progress_bar.progress((i + 1) / len(minute_data_to_simulate), text=f"Simulating: {timestamp.strftime('%Y-%m-%d %H:%M')}")
            
#             # --- UPDATE HISTORY for next iteration ---
#             history.append(actual_value)

#             time.sleep(simulation_speed)
        
#         st.success("Simulation Complete!")
#         st.session_state.run_simulation = False


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
import collections

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Energy Intelligence Demo",
    page_icon="⚡",
    layout="wide"
)

# --- MODEL AND DATA LOADING (CACHED) ---
@st.cache_resource
def load_all_models():
    """Load all trained models."""
    try:
        xgb_model = joblib.load('xgb_forecaster.pkl')
        isoforest_model = joblib.load('isolation_forest_model.pkl')
        lstm_model = load_model('lstm_model.keras')
        return xgb_model, isoforest_model, lstm_model
    except (IOError, FileNotFoundError) as e:
        st.error(f"Error loading models: {e}. Please ensure all .pkl and .keras files are present.")
        return None, None, None

@st.cache_data
def load_all_data():
    """Load and preprocess all necessary datasets."""
    try:
        # High-frequency data
        energy_minute_df = pd.read_csv('EnergyDataset.txt', delimiter=';', low_memory=False)
        energy_minute_df['date_time'] = pd.to_datetime(energy_minute_df['Date'] + ' ' + energy_minute_df['Time'], dayfirst=True, errors='coerce')
        energy_minute_df['Global_active_power'] = pd.to_numeric(energy_minute_df['Global_active_power'], errors='coerce')
        energy_minute_df.dropna(subset=['date_time', 'Global_active_power'], inplace=True)
        energy_minute_df.set_index('date_time', inplace=True)

        # Scaler for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = energy_minute_df.Global_active_power.values.astype('float32').reshape(-1, 1)
        scaler.fit(dataset)

        # Daily data
        energy_daily_df = energy_minute_df['Global_active_power'].resample('D').sum().to_frame() / 60
        weather_df = pd.read_csv('new_weather_data.csv')
        weather_df.rename(columns={'date': 'time', 'temp_mean_C': 'tavg', 'temp_min_C': 'tmin', 'temp_max_C': 'tmax'}, inplace=True)
        weather_df['time'] = pd.to_datetime(weather_df['time'])
        weather_df.set_index('time', inplace=True)
        full_daily_df = energy_daily_df.join(weather_df, how='inner')
        
        return energy_minute_df, full_daily_df, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}. Please ensure EnergyDataset.txt and new_weather_data.csv are present.")
        return None, None, None

def create_forecast_features(target_date, historical_data):
    """Create the feature vector for the XGBoost model."""
    features = {}
    features['lag_1'] = historical_data.loc[target_date - pd.DateOffset(days=1), 'Global_active_power']
    features['lag_7'] = historical_data.loc[target_date - pd.DateOffset(days=7), 'Global_active_power']
    features['lag_365'] = historical_data.loc[target_date - pd.DateOffset(days=365), 'Global_active_power']
    rolling_window = historical_data.loc[target_date - pd.DateOffset(days=29):target_date, 'Global_active_power']
    features['rolling_30_mean'] = rolling_window.mean()
    features['rolling_30_sum'] = rolling_window.sum()
    features['month'] = target_date.month
    features['quarter'] = target_date.quarter
    features['dayofweek'] = target_date.dayofweek
    features['dayofyear'] = target_date.dayofyear
    weather_today = historical_data.loc[target_date]
    features['tavg'] = weather_today['tavg']
    features['tmin'] = weather_today['tmin']
    features['tmax'] = weather_today['tmax']
    features['prcp'] = weather_today.get('prcp', 0)
    features['wspd'] = weather_today.get('wspd', 0)
    feature_columns = [
        'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'lag_1', 'lag_7', 'lag_365',
        'rolling_30_mean', 'rolling_30_sum', 'month', 'quarter', 'dayofweek', 'dayofyear'
    ]
    return pd.DataFrame([features], columns=feature_columns)

# --- MAIN APP ---
st.title("⚡ Comprehensive Energy Intelligence Demo")
st.markdown("A unified dashboard by **Fadhil** showcasing the project's forecasting and detection models.")

# Load all assets
xgb_model, isoforest_model, lstm_model = load_all_models()
energy_minute_df, full_daily_df, scaler = load_all_data()

if all(v is not None for v in [xgb_model, isoforest_model, lstm_model, energy_minute_df, full_daily_df, scaler]):
    
    # --- CALCULATE AVAILABLE TEST MONTHS ---
    test_start_date = full_daily_df.index.max() - pd.DateOffset(days=180)
    available_months = full_daily_df[full_daily_df.index >= test_start_date].index.to_period('M').unique()
    available_months_str = sorted([month.strftime('%Y-%m') for month in available_months])

    # --- TABS FOR DIFFERENT DEMOS ---
    tab1, tab2, tab3 = st.tabs(["🗓️ 30-Day Forecast", "🧠 Real-Time Forecast", "🚨 Anomaly Detection"])

    # --- TAB 1: XGBOOST 30-DAY FORECAST ---
    with tab1:
        st.header("Monthly Consumption Forecast (XGBoost)")
        st.write("This tool predicts the total energy consumption for a selected month.")
        
        selected_month_xgb = st.selectbox("Choose a month to predict:", options=available_months_str, index=len(available_months_str) - 2)

        if st.button("Generate 30-Day Forecast", type="primary"):
            forecast_date = pd.to_datetime(selected_month_xgb + '-01') - pd.DateOffset(days=1)
            with st.spinner(f"Predicting for {pd.to_datetime(selected_month_xgb).strftime('%B %Y')}..."):
                features_df = create_forecast_features(forecast_date, full_daily_df)
                prediction = xgb_model.predict(features_df)[0]
                actual = full_daily_df.loc[selected_month_xgb, 'Global_active_power'].sum()

            col1, col2 = st.columns(2)
            col1.metric("Predicted Consumption", f"{prediction:,.2f} kWh")
            col2.metric("Actual Consumption", f"{actual:,.2f} kWh", delta=f"{prediction - actual:,.2f} kWh")
            st.success("Forecast complete!")

    # --- TAB 2: LSTM REAL-TIME FORECAST ---
    with tab2:
        st.header("Next-Minute Power Prediction (LSTM)")
        st.write("This simulates a live data stream, using the last 30 minutes of data to predict the power consumption for the very next minute.")
        st.sidebar.header("Real-Time Simulation Controls")
        selected_month_lstm = st.sidebar.selectbox("Choose a month to simulate for LSTM:", options=available_months_str, index=len(available_months_str) - 2)
        sim_speed_lstm = st.sidebar.slider("Simulation Speed (LSTM)", 0.0, 0.5, 0.05, 0.01)

        if st.sidebar.button("Start LSTM Simulation", type="primary"):
            minute_data = energy_minute_df.loc[selected_month_lstm]['Global_active_power']
            
            ph_actual = st.empty()
            ph_pred = st.empty()
            ph_error = st.empty()
            chart_ph = st.empty()
            progress_bar = st.progress(0)

            history = collections.deque(maxlen=30)
            warm_up_start = minute_data.index[0] - pd.DateOffset(minutes=30)
            warm_up_data = energy_minute_df.loc[warm_up_start:minute_data.index[0]-pd.DateOffset(minutes=1), 'Global_active_power']
            for val in warm_up_data: history.append(val)
            
            chart_data = pd.DataFrame(columns=["Actual", "Predicted"])

            for i, (ts, actual) in enumerate(minute_data.items()):
                input_seq = np.array(list(history)).reshape(-1, 1)
                scaled_input = scaler.transform(input_seq)
                reshaped_input = np.reshape(scaled_input, (1, 1, 30))
                
                pred_scaled = lstm_model.predict(reshaped_input, verbose=0)
                pred_kw = scaler.inverse_transform(pred_scaled)[0][0]

                ph_actual.metric("Actual Power", f"{actual:.2f} kW")
                ph_pred.metric("Predicted Power", f"{pred_kw:.2f} kW")
                ph_error.metric("Difference", f"{actual - pred_kw:.2f} kW")
                
                new_row = pd.DataFrame({"Actual": [actual], "Predicted": [pred_kw]}, index=[ts])
                chart_data = pd.concat([chart_data, new_row])
                chart_ph.line_chart(chart_data)

                progress_bar.progress((i + 1) / len(minute_data), text=f"Simulating: {ts.strftime('%Y-%m-%d %H:%M')}")
                history.append(actual)
                time.sleep(sim_speed_lstm)
            st.success("LSTM Simulation Complete!")

    # --- TAB 3: ISOLATION FOREST ANOMALY DETECTION ---
    with tab3:
        st.header("Live Anomaly Detection (Isolation Forest)")
        st.write("This simulates a live data stream and uses the Isolation Forest model to flag unusual power readings in real time.")
        st.sidebar.header("Anomaly Detection Controls")
        selected_month_iso = st.sidebar.selectbox("Choose a month to simulate for Anomalies:", options=available_months_str, index=len(available_months_str) - 2)
        sim_speed_iso = st.sidebar.slider("Simulation Speed (Anomalies)", 0.0, 0.1, 0.01, 0.001, format="%.3f s")

        if st.sidebar.button("Start Anomaly Simulation", type="primary"):
            st.session_state.run_anomaly_sim = True
            st.session_state.anomaly_alerts = []
            st.session_state.sim_month_iso = selected_month_iso

        if st.session_state.get('run_anomaly_sim', False):
            minute_data = energy_minute_df.loc[st.session_state.sim_month_iso]
            st.info(f"Simulating {len(minute_data)} minutes...")
            
            progress_bar_iso = st.progress(0)
            st.subheader("Anomaly Log")
            alert_placeholder = st.empty()

            for i, (ts, row) in enumerate(minute_data.iterrows()):
                power = row['Global_active_power']
                input_df = pd.DataFrame({'Global_active_power': [power]})
                prediction = isoforest_model.predict(input_df)

                if prediction[0] == -1:
                    alert_msg = f"🚨 Anomaly Detected at **{ts.strftime('%Y-%m-%d %H:%M')}** | Power: **{power:.2f} kW**"
                    st.session_state.anomaly_alerts.append(alert_msg)
                    with alert_placeholder.container():
                        for alert in reversed(st.session_state.anomaly_alerts):
                            st.warning(alert)
                
                progress_bar_iso.progress((i + 1) / len(minute_data), text=f"Simulating: {ts.strftime('%Y-%m-%d %H:%M')}")
                time.sleep(sim_speed_iso)
            
            st.success("Anomaly Simulation Complete!")
            st.session_state.run_anomaly_sim = False



