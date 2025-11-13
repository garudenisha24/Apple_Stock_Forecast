import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Hybrid SARIMAX+LSTM Stock Forecast", layout="wide")
st.title("Hybrid SARIMAX + LSTM Stock Price Forecast")

# --- Function to load data and models (cached for performance) ---
@st.cache_resource
def load_models_and_data():
    """Loads all necessary files."""
    try:
        sarimax_fit = joblib.load("sarimax_model.pkl")
        lstm_model = load_model("lstm_model.h5")
        scaler = joblib.load("residual_scaler.pkl")
        df = pd.read_csv("preloaded_data.csv")
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        return df, sarimax_fit, lstm_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please make sure all model and data files are in the correct directory.")
        return None, None, None, None

# --- Load Data and Models ---
df, sarimax_fit, lstm_model, scaler = load_models_and_data()

if df is not None:
    # --- Display Dataset Preview ---
    st.subheader("Historical Data Preview")
    st.dataframe(df.tail())

    # --- User Input for Forecast Period ---
    st.sidebar.header("Forecasting Parameters")
    future_days = st.sidebar.slider(
        "Select number of days to forecast",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )
    
    # --- Allow user to choose date frequency ---
    # This is the key change to address the date discrepancy
    freq_option = st.sidebar.selectbox(
        "Select Forecast Frequency",
        ('B', 'D'),
        format_func=lambda x: 'Business Days (B)' if x == 'B' else 'Calendar Days (D)'
    )

    # --- Button to generate the forecast ---
    if st.sidebar.button("Generate Forecast"):
        with st.spinner('Generating forecast... Please wait.'):
        
            # ** FOR REPRODUCIBILITY **
            # To get the exact same forecast every time, set a random seed.
            # Remove the comment from the line below to make results deterministic.
            # np.random.seed(42)

            # --- Define constants and features ---
            features = ['nasdaq_index', 'sp500_index', 'inflation_rate',
                        'unemployment_rate', 'interest_rate', 'market_sentiment']
            time_step = 60

            # 1. Compute and scale residuals
            residuals = df['log_return'].values - sarimax_fit.predict(start=0, end=len(df)-1, exog=df[features])
            residuals_scaled = scaler.transform(np.array(residuals).reshape(-1, 1))
            last_residuals = residuals_scaled[-time_step:].reshape(1, time_step, 1)
            resid_std = residuals_scaled[-time_step:].std()

            # 2. Prepare exogenous variables
            future_exog = df[features].iloc[-1:].values
            future_exog = np.repeat(future_exog, future_days, axis=0)
            future_sarimax = sarimax_fit.forecast(steps=future_days, exog=future_exog)

            # 3. Combine SARIMAX forecast with LSTM-predicted residuals
            future_pred = []
            temp_resid_input = last_residuals.copy()

            for i in range(future_days):
                lstm_resid = lstm_model.predict(temp_resid_input, verbose=0)[0, 0]
                lstm_resid += np.random.randn() * resid_std
                pred = future_sarimax.iloc[i] + scaler.inverse_transform([[lstm_resid]])[0, 0]
                future_pred.append(pred)
                temp_resid_input = np.append(temp_resid_input[:, 1:, :], [[[lstm_resid]]], axis=1)

            # 4. Convert log returns to prices
            last_price = df['stock_price'].iloc[-1]
            price_forecast = [last_price]
            for r in future_pred:
                price_forecast.append(price_forecast[-1] * np.exp(r))
            price_forecast = price_forecast[1:]

            # 5. Create forecast dates using the selected frequency
            last_date = df['Date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=future_days, 
                freq=freq_option # Use the user-selected frequency
            )

            # 6. Create final DataFrame
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecasted_Price': price_forecast
            })

            st.subheader(f"Forecasted Prices for the Next {future_days} Days")
            st.dataframe(forecast_df.head(5))
            #forecast_df.head(5)

            # 7. Plotting
            st.subheader("Forecast Visualization")
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(df['Date'].tail(90), df['stock_price'].tail(90), label='Historical Price', color='blue')
            ax.plot(forecast_dates, price_forecast, label='Forecasted Price', color='purple', marker='o', linestyle='--')
            ax.set_title('Hybrid SARIMAX + LSTM Stock Price Forecast', fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
else:
    st.warning("Application could not start because required files were not found.")