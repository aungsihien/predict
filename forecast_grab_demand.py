import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import os
from datetime import datetime

# Ignore warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

print("Starting Grab voucher demand forecasting...")

# Create output directory for plots
output_dir = "forecast_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the processed data
data_file = "grab_voucher_timeseries.csv"
print(f"Reading data from {data_file}...")
df = pd.read_csv(data_file)

# Convert year and month to datetime for proper time series analysis
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# Function to calculate MAPE
def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# Function to train Prophet model and forecast
def prophet_forecast(data, price, periods=6):
    """Train a Prophet model and forecast future demand"""
    # Prepare data for Prophet
    model_data = data[data['Price'] == price].copy()
    model_data = model_data[['date', 'sold_count']].rename(columns={'date': 'ds', 'sold_count': 'y'})
    
    # Train model
    model = Prophet(yearly_seasonality=True, 
                   weekly_seasonality=False, 
                   daily_seasonality=False,
                   seasonality_mode='multiplicative',
                   interval_width=0.95)
    model.fit(model_data)
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    
    # Calculate MAPE on training data
    train_predictions = forecast[forecast['ds'].isin(model_data['ds'])]['yhat'].values
    train_actual = model_data['y'].values
    mape = calculate_mape(train_actual, train_predictions)
    
    # Extract only the forecasted months
    forecast_result = forecast[~forecast['ds'].isin(model_data['ds'])].copy()
    forecast_result['year'] = forecast_result['ds'].dt.year
    forecast_result['month'] = forecast_result['ds'].dt.month
    forecast_result['Price'] = price
    forecast_result['predicted_sold_count'] = forecast_result['yhat'].round().astype(int)
    forecast_result['lower_bound'] = forecast_result['yhat_lower'].round().astype(int)
    forecast_result['upper_bound'] = forecast_result['yhat_upper'].round().astype(int)
    forecast_result['mape'] = mape
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(model_data['ds'], model_data['y'], label='Historical', marker='o')
    
    # Plot forecasted data
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
    
    # Plot confidence interval
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='red', alpha=0.2, label='95% Confidence Interval')
    
    # Add annotations
    plt.title(f'Grab Voucher Demand Forecast - Price {price:,} MMK (MAPE: {mape:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Monthly Sold Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add MAPE annotation
    plt.annotate(f'MAPE: {mape:.2f}%', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save the plot
    plt.savefig(f"{output_dir}/prophet_forecast_price_{price}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return forecast_result[['year', 'month', 'Price', 'predicted_sold_count', 'lower_bound', 'upper_bound', 'mape']]

# Function to train SARIMA model and forecast
def sarima_forecast(data, price, periods=6):
    """Train a SARIMA model and forecast future demand"""
    # Prepare data for SARIMA
    model_data = data[data['Price'] == price].copy()
    model_data = model_data.sort_values('date')
    y = model_data['sold_count'].values
    
    # Determine best SARIMA parameters (simplified approach)
    # In a production environment, we would do a more thorough grid search
    try:
        # Try SARIMA with seasonal component (12 months)
        model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
    except:
        # Fallback to simpler ARIMA model if SARIMA fails
        model = ARIMA(y, order=(1, 1, 1))
        model_fit = model.fit()
    
    # Forecast
    forecast_values = model_fit.forecast(steps=periods)
    forecast_values = np.maximum(forecast_values, 0)  # Ensure non-negative forecasts
    
    # Calculate confidence intervals (approximate)
    std_err = np.sqrt(model_fit.cov_params().diagonal())
    lower_bound = forecast_values - 1.96 * std_err[0] * np.sqrt(np.arange(1, periods+1))
    upper_bound = forecast_values + 1.96 * std_err[0] * np.sqrt(np.arange(1, periods+1))
    
    # Ensure non-negative bounds
    lower_bound = np.maximum(lower_bound, 0)
    
    # Calculate MAPE on training data
    train_predictions = model_fit.fittedvalues
    mape = calculate_mape(y[1:], train_predictions[1:])  # Skip first value due to differencing
    
    # Create forecast dataframe
    last_date = model_data['date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
    
    forecast_result = pd.DataFrame({
        'date': forecast_dates,
        'year': forecast_dates.year,
        'month': forecast_dates.month,
        'Price': price,
        'predicted_sold_count': forecast_values.round().astype(int),
        'lower_bound': lower_bound.round().astype(int),
        'upper_bound': upper_bound.round().astype(int),
        'mape': mape
    })
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(model_data['date'], model_data['sold_count'], label='Historical', marker='o')
    
    # Plot forecasted data
    plt.plot(forecast_dates, forecast_values, label='Forecast', color='green')
    
    # Plot confidence interval
    plt.fill_between(forecast_dates, lower_bound, upper_bound, 
                    color='green', alpha=0.2, label='95% Confidence Interval')
    
    # Add annotations
    plt.title(f'Grab Voucher Demand Forecast - Price {price:,} MMK (MAPE: {mape:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Monthly Sold Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add MAPE annotation
    plt.annotate(f'MAPE: {mape:.2f}%', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save the plot
    plt.savefig(f"{output_dir}/sarima_forecast_price_{price}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return forecast_result[['year', 'month', 'Price', 'predicted_sold_count', 'lower_bound', 'upper_bound', 'mape']]

# Get unique price points
price_points = sorted(df['Price'].unique())
print(f"Found {len(price_points)} price points: {price_points}")

# Store forecasts from both models
prophet_forecasts = []
sarima_forecasts = []

# Generate forecasts for each price point
for price in price_points:
    print(f"\nForecasting demand for Price {price:,} MMK...")
    
    try:
        # Prophet forecast
        print(f"  Training Prophet model...")
        prophet_result = prophet_forecast(df, price)
        prophet_forecasts.append(prophet_result)
        prophet_mape = prophet_result['mape'].iloc[0]
        print(f"  Prophet MAPE: {prophet_mape:.2f}%")
        
        # SARIMA forecast
        print(f"  Training SARIMA model...")
        sarima_result = sarima_forecast(df, price)
        sarima_forecasts.append(sarima_result)
        sarima_mape = sarima_result['mape'].iloc[0]
        print(f"  SARIMA MAPE: {sarima_mape:.2f}%")
        
    except Exception as e:
        print(f"  Error forecasting for Price {price}: {str(e)}")

# Combine forecasts from both models, selecting the one with better MAPE for each price
best_forecasts = []

for price in price_points:
    prophet_forecast_for_price = next((f for f in prophet_forecasts if f['Price'].iloc[0] == price), None)
    sarima_forecast_for_price = next((f for f in sarima_forecasts if f['Price'].iloc[0] == price), None)
    
    if prophet_forecast_for_price is not None and sarima_forecast_for_price is not None:
        prophet_mape = prophet_forecast_for_price['mape'].iloc[0]
        sarima_mape = sarima_forecast_for_price['mape'].iloc[0]
        
        if prophet_mape <= sarima_mape:
            best_forecasts.append(prophet_forecast_for_price)
            print(f"Price {price:,} MMK: Selected Prophet model (MAPE: {prophet_mape:.2f}%)")
        else:
            best_forecasts.append(sarima_forecast_for_price)
            print(f"Price {price:,} MMK: Selected SARIMA model (MAPE: {sarima_mape:.2f}%)")
    elif prophet_forecast_for_price is not None:
        best_forecasts.append(prophet_forecast_for_price)
        print(f"Price {price:,} MMK: Selected Prophet model (only option)")
    elif sarima_forecast_for_price is not None:
        best_forecasts.append(sarima_forecast_for_price)
        print(f"Price {price:,} MMK: Selected SARIMA model (only option)")

# Combine all best forecasts
if best_forecasts:
    final_forecast = pd.concat(best_forecasts)
    
    # Save final forecast to CSV
    forecast_file = "grab_voucher_forecast.csv"
    final_forecast.to_csv(forecast_file, index=False)
    print(f"\nFinal forecast saved to {forecast_file}")
    
    # Save final forecast to Excel
    excel_file = "grab_voucher_forecast.xlsx"
    final_forecast.to_excel(excel_file, index=False)
    print(f"Final forecast also saved to {excel_file}")
    
    # Create a combined visualization of all price points
    plt.figure(figsize=(15, 10))
    
    # Plot historical and forecasted data for each price
    for price in price_points:
        # Historical data
        historical = df[df['Price'] == price]
        
        # Forecast data
        forecast = final_forecast[final_forecast['Price'] == price]
        
        # Get the last historical date and first forecast date
        if not historical.empty and not forecast.empty:
            last_hist_date = historical['date'].max()
            forecast_dates = pd.date_range(start=last_hist_date + pd.DateOffset(months=1), periods=len(forecast), freq='MS')
            
            # Plot
            plt.plot(historical['date'], historical['sold_count'], marker='o', label=f'Historical {price:,} MMK')
            plt.plot(forecast_dates, forecast['predicted_sold_count'], marker='s', linestyle='--', label=f'Forecast {price:,} MMK')
    
    plt.title('Grab Voucher Demand Forecast - All Price Points')
    plt.xlabel('Date')
    plt.ylabel('Monthly Sold Count')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_forecast.png", dpi=300, bbox_inches='tight')
    
    # Print summary of forecast accuracy
    print("\nForecast Accuracy Summary:")
    accuracy_summary = final_forecast.groupby('Price')['mape'].first().reset_index()
    for _, row in accuracy_summary.iterrows():
        status = "✅ MEETS TARGET" if row['mape'] <= 10 else "❌ BELOW TARGET"
        print(f"Price {row['Price']:,} MMK: MAPE = {row['mape']:.2f}% - {status}")
    
    # Overall accuracy
    overall_mape = accuracy_summary['mape'].mean()
    overall_status = "✅ MEETS TARGET" if overall_mape <= 10 else "❌ BELOW TARGET"
    print(f"\nOverall MAPE: {overall_mape:.2f}% - {overall_status}")
else:
    print("No forecasts were generated. Please check the data and models.")

print("\nForecasting completed!")
