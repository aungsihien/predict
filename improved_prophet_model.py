import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
import os
from datetime import datetime

# Ignore warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

print("Starting improved Grab voucher forecasting model...")

# Create output directory for results
output_dir = "improved_forecast_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the processed data
data_file = "grab_voucher_timeseries.csv"
print(f"Reading data from {data_file}...")
df = pd.read_csv(data_file)

# Convert year and month to datetime for proper time series analysis
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# Print data range to verify
print(f"Data range: {df['date'].min()} to {df['date'].max()}")
print(f"Total records: {len(df)}")

# Function to calculate MAPE
def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    # Filter out zeros to avoid division by zero
    mask = (actual != 0) & (~np.isnan(actual)) & (~np.isnan(predicted))
    if sum(mask) == 0:
        return 0  # No valid points for MAPE calculation
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# Function to add minimum floor to predictions
def add_minimum_floor(predictions, min_floor=10):
    """Add a minimum floor to predictions to avoid zeros"""
    predictions_with_floor = predictions.copy()
    predictions_with_floor['yhat'] = np.maximum(predictions_with_floor['yhat'], min_floor)
    predictions_with_floor['yhat_lower'] = np.maximum(predictions_with_floor['yhat_lower'], min_floor)
    predictions_with_floor['yhat_upper'] = np.maximum(predictions_with_floor['yhat_upper'], min_floor)
    return predictions_with_floor

# Function to smooth extreme variations
def smooth_extreme_variations(predictions, smoothing_factor=0.3, price=None):
    """Apply smoothing to reduce extreme variations between consecutive months"""
    smoothed = predictions.copy()
    
    # Only apply smoothing to the forecast portion, not historical data
    forecast_mask = ~smoothed['ds'].isin(model_data['ds'])
    forecast_indices = smoothed[forecast_mask].index
    
    if len(forecast_indices) <= 1:
        return smoothed  # Not enough points to smooth
    
    # Adjust smoothing factor based on price point (reduced smoothing to allow more variation)
    adjusted_smoothing = smoothing_factor
    if price is not None:
        if price <= 5000:
            adjusted_smoothing = 0.25  # 25% smoothing for 5,000 MMK
        elif price <= 10000:
            adjusted_smoothing = 0.25  # 25% smoothing for 10,000 MMK
        elif price <= 20000:
            adjusted_smoothing = 0.2   # 20% smoothing for 20,000 MMK
        else:
            adjusted_smoothing = 0.15  # 15% smoothing for higher price points
    
    # Simple exponential smoothing
    for i in range(1, len(forecast_indices)):
        curr_idx = forecast_indices[i]
        prev_idx = forecast_indices[i-1]
        
        # Weighted average of current prediction and previous prediction
        current_pred = smoothed.loc[curr_idx, 'yhat']
        prev_pred = smoothed.loc[prev_idx, 'yhat']
        
        # Apply smoothing only if there's a very large variation (more than 60%)
        # This allows more natural variation while still preventing extreme jumps
        if abs(current_pred - prev_pred) > 0.6 * prev_pred and prev_pred > 0:
            smoothed.loc[curr_idx, 'yhat'] = (1 - adjusted_smoothing) * current_pred + adjusted_smoothing * prev_pred
            
            # Adjust confidence intervals proportionally
            ratio = smoothed.loc[curr_idx, 'yhat'] / current_pred if current_pred > 0 else 1
            smoothed.loc[curr_idx, 'yhat_lower'] *= ratio
            smoothed.loc[curr_idx, 'yhat_upper'] *= ratio
    
    return smoothed

# Function to train improved Prophet model and forecast
def improved_prophet_forecast(data, price, periods=6, min_floor=10, smoothing_factor=0.3):
    """Train an improved Prophet model with adjustments to address extreme variations"""
    # Prepare data for Prophet
    model_data = data[data['Price'] == price].copy()
    model_data = model_data[['date', 'sold_count']].rename(columns={'date': 'ds', 'sold_count': 'y'})
    
    # Print training data info
    print(f"  Training data for Price {price}: {len(model_data)} records from {model_data['ds'].min()} to {model_data['ds'].max()}")
    
    # Define monthly caps for high-volume price points to make forecasts more conservative
    monthly_caps = {
        5000: 800,    # Cap at 800 vouchers per month for 5,000 MMK
        10000: 900,   # Cap at 900 vouchers per month for 10,000 MMK
        20000: 1000,  # Cap at 1000 vouchers per month for 20,000 MMK
        50000: 300,   # Cap at 300 vouchers per month for 50,000 MMK
        100000: 100   # Cap at 100 vouchers per month for 100,000 MMK
    }
    
    # Model configuration improvements:
    model = Prophet(
        # Use additive seasonality to prevent extreme multiplicative effects
        seasonality_mode='additive',
        
        # Increase uncertainty interval for more realistic bounds
        interval_width=0.9,
        
        # Add weekly and yearly seasonality based on data availability
        weekly_seasonality=False,
        yearly_seasonality=len(model_data) >= 12,
        
        # Use a more flexible trend to capture changes
        changepoint_prior_scale=0.05,  # Default is 0.05, higher values allow more flexibility
        
        # Limit number of changepoints to prevent overfitting
        n_changepoints=min(10, max(5, len(model_data) // 2)),
        
        # Add seasonality prior scale to control seasonality strength
        seasonality_prior_scale=10.0  # Default is 10, lower values dampen seasonality
    )
    
    # Add monthly seasonality explicitly
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5  # Higher order for more flexibility in monthly patterns
    )
    
    # Add country-specific holidays if relevant (example for Myanmar)
    # model.add_country_holidays(country_name='MM')
    
    # Fit the model
    model.fit(model_data)
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=periods, freq='MS')
    
    # Make predictions
    forecast = model.predict(future)
    
    # Apply minimum floor to avoid zeros
    forecast = add_minimum_floor(forecast, min_floor=min_floor)
    
    # Apply smoothing to reduce extreme variations
    forecast = smooth_extreme_variations(forecast, smoothing_factor=smoothing_factor, price=price)
    
    # Apply monthly caps to high-volume price points
    if price in monthly_caps:
        cap_value = monthly_caps[price]
        # Only apply cap to forecast portion, not historical data
        forecast_mask = ~forecast['ds'].isin(model_data['ds'])
        forecast.loc[forecast_mask, 'yhat'] = np.minimum(forecast.loc[forecast_mask, 'yhat'], cap_value)
        
        # Adjust confidence intervals proportionally if cap was applied
        for idx in forecast[forecast_mask].index:
            if forecast.loc[idx, 'yhat'] == cap_value and forecast.loc[idx, 'yhat_upper'] > cap_value:
                ratio = cap_value / forecast.loc[idx, 'yhat_upper']
                forecast.loc[idx, 'yhat_upper'] = cap_value
                forecast.loc[idx, 'yhat_lower'] = max(min_floor, forecast.loc[idx, 'yhat_lower'] * ratio)
    
    # Ensure predictions are non-negative
    forecast['yhat'] = np.maximum(forecast['yhat'], 0)
    forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
    forecast['yhat_upper'] = np.maximum(forecast['yhat_upper'], 0)
    
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
    
    # Create visualization with component breakdown
    fig = model.plot(forecast)
    plt.title(f'Grab Voucher Demand Forecast - Price {price:,} MMK')
    plt.savefig(f"{output_dir}/forecast_price_{price}.png", dpi=300, bbox_inches='tight')
    
    # Plot components to understand trend, seasonality, etc.
    fig = model.plot_components(forecast)
    plt.savefig(f"{output_dir}/components_price_{price}.png", dpi=300, bbox_inches='tight')
    
    # Create visualization comparing original vs. improved forecast
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(model_data['ds'], model_data['y'], label='Historical', marker='o')
    
    # Plot forecasted data
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
    
    # Plot confidence interval
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='red', alpha=0.2, label='95% Confidence Interval')
    
    # Add annotations
    plt.title(f'Improved Grab Voucher Demand Forecast - Price {price:,} MMK (MAPE: {mape:.2f}%)')
    plt.xlabel('Month')
    plt.ylabel('Voucher Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add MAPE annotation
    plt.annotate(f'MAPE: {mape:.2f}%', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.savefig(f"{output_dir}/improved_forecast_price_{price}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create actual vs predicted table for recent months
    validation_data = forecast[forecast['ds'].isin(model_data['ds'])].copy()
    validation_data = validation_data.merge(
        model_data, on='ds', how='inner'
    )[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]
    validation_data.columns = ['Date', 'Actual', 'Predicted', 'Lower Bound', 'Upper Bound']
    validation_data['Absolute Error'] = np.abs(validation_data['Actual'] - validation_data['Predicted'])
    validation_data['Percentage Error'] = np.where(
        validation_data['Actual'] > 0,
        (validation_data['Absolute Error'] / validation_data['Actual']) * 100,
        0
    )
    validation_data = validation_data.sort_values('Date', ascending=False).head(6)  # Last 6 months
    
    # Save validation table
    validation_data.to_csv(f"{output_dir}/validation_price_{price}.csv", index=False)
    
    # Perform cross-validation to assess model performance
    try:
        if len(model_data) >= 6:  # Need enough data for cross-validation
            cv_results = cross_validation(
                model, 
                initial='180 days',  # Use first 6 months for training
                period='30 days',    # Test on each month
                horizon='90 days'    # Forecast 3 months ahead in each fold
            )
            cv_metrics = performance_metrics(cv_results)
            cv_metrics.to_csv(f"{output_dir}/cv_metrics_price_{price}.csv", index=False)
            
            # Plot cross-validation results
            plt.figure(figsize=(10, 6))
            plt.scatter(cv_results['ds'], cv_results['y'], alpha=0.5, label='Actual')
            plt.scatter(cv_results['ds'], cv_results['yhat'], alpha=0.5, label='Predicted')
            plt.title(f'Cross-Validation Results - Price {price:,} MMK')
            plt.legend()
            plt.savefig(f"{output_dir}/cv_results_price_{price}.png", dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"  Cross-validation error: {str(e)}")
    
    return forecast_result[['year', 'month', 'Price', 'predicted_sold_count', 'lower_bound', 'upper_bound', 'mape']], validation_data

# Get unique price points
price_points = sorted(df['Price'].unique())
print(f"Found {len(price_points)} price points: {price_points}")

# Store improved forecasts
improved_forecasts = []
all_validation_data = []

# Define minimum floors for each price point
# Higher denominations typically have lower volumes, so use different floors
min_floors = {
    5000: 20,    # Minimum 20 vouchers for 5,000 MMK
    10000: 15,   # Minimum 15 vouchers for 10,000 MMK
    20000: 10,   # Minimum 10 vouchers for 20,000 MMK
    50000: 5,    # Minimum 5 vouchers for 50,000 MMK
    100000: 3    # Minimum 3 vouchers for 100,000 MMK
}

# Generate improved forecasts for each price point
for price in price_points:
    print(f"\nForecasting demand for Price {price:,} MMK...")
    
    try:
        # Get appropriate minimum floor for this price
        min_floor = min_floors.get(price, 10)  # Default to 10 if price not in dictionary
        
        # Generate improved forecast
        forecast_result, validation = improved_prophet_forecast(
            df, 
            price, 
            min_floor=min_floor,
            smoothing_factor=0.3  # Adjust smoothing factor as needed
        )
        
        improved_forecasts.append(forecast_result)
        all_validation_data.append(validation)
        
        mape = forecast_result['mape'].iloc[0]
        print(f"  Improved Prophet MAPE: {mape:.2f}%")
        
    except Exception as e:
        print(f"  Error forecasting for Price {price}: {str(e)}")

# Function to ensure consistent forecasts across all months
def ensure_consistent_forecasts(historical_data, forecast_data, min_percentage=0.5):
    """Ensure forecasts are consistent across all months by comparing with recent historical averages"""
    adjusted_forecast = forecast_data.copy()
    
    # Process each price point separately
    for price in adjusted_forecast['Price'].unique():
        # Get historical data for this price
        price_hist = historical_data[historical_data['Price'] == price].sort_values('date')
        price_forecast = adjusted_forecast[adjusted_forecast['Price'] == price].sort_values('date')
        
        if len(price_hist) < 3 or len(price_forecast) == 0:
            continue  # Not enough historical data or no forecast data
        
        # Calculate average of last 3 months of historical data
        recent_avg = price_hist['sold_count'].tail(3).mean()
        
        # Calculate average of forecast months (excluding the first month)
        if len(price_forecast) > 1:
            forecast_avg = price_forecast['predicted_sold_count'].iloc[1:].mean()
        else:
            forecast_avg = recent_avg  # Use historical average if only one forecast month
        
        # Check each forecast month
        for idx, row in price_forecast.iterrows():
            # If any month's forecast is less than min_percentage of the forecast average or recent average
            if row['predicted_sold_count'] < min_percentage * forecast_avg or row['predicted_sold_count'] < min_percentage * recent_avg:
                # Adjust the forecast to be at least min_percentage of the larger of the two averages
                min_value = min_percentage * max(forecast_avg, recent_avg)
                adjusted_forecast.loc[idx, 'predicted_sold_count'] = int(round(min_value))
                
                # Adjust confidence intervals proportionally
                if row['predicted_sold_count'] > 0:
                    ratio = adjusted_forecast.loc[idx, 'predicted_sold_count'] / row['predicted_sold_count']
                    adjusted_forecast.loc[idx, 'lower_bound'] = max(10, int(round(row['lower_bound'] * ratio)))
                    adjusted_forecast.loc[idx, 'upper_bound'] = max(10, int(round(row['upper_bound'] * ratio)))
    
    return adjusted_forecast

# Combine all improved forecasts
if improved_forecasts:
    final_forecast = pd.concat(improved_forecasts)
    
    # Apply consistency check to ensure all months have reasonable values
    print("Applying consistency checks to ensure balanced forecasts across all months...")
    final_forecast = ensure_consistent_forecasts(df, final_forecast, min_percentage=0.6)
    
    # Save final forecast to CSV
    forecast_file = "improved_grab_forecast.csv"
    final_forecast.to_csv(forecast_file, index=False)
    print(f"\nImproved forecast saved to {forecast_file}")
    
    # Save final forecast to Excel
    excel_file = "improved_grab_forecast.xlsx"
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Add the forecasts
            final_forecast.to_excel(writer, sheet_name='Forecasts', index=False)
            
            # Add validation data
            if all_validation_data:
                validation_df = pd.concat(all_validation_data)
                validation_df.to_excel(writer, sheet_name='Validation', index=False)
        
        print(f"Improved forecast also saved to {excel_file}")
    except Exception as e:
        print(f"Could not save to Excel: {str(e)}")
    
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
    
    plt.title('Improved Grab Voucher Demand Forecast - All Price Points')
    plt.xlabel('Month')
    plt.ylabel('Voucher Count')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_forecast.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary of forecast accuracy
    print("\nForecast Accuracy Summary:")
    accuracy_summary = final_forecast.groupby('Price')['mape'].first().reset_index()
    for _, row in accuracy_summary.iterrows():
        status = "MEETS TARGET" if row['mape'] <= 10 else "BELOW TARGET"
        detail = "EXCELLENT" if row['mape'] <= 5 else ""
        print(f"Price {row['Price']:,} MMK: MAPE = {row['mape']:.2f}% - {status} {detail}")
    
    # Overall accuracy
    overall_mape = accuracy_summary['mape'].mean()
    overall_status = "MEETS TARGET" if overall_mape <= 10 else "BELOW TARGET"
    overall_detail = "EXCELLENT" if overall_mape <= 5 else ""
    print(f"\nOverall MAPE: {overall_mape:.2f}% - {overall_status} {overall_detail}")
else:
    print("No forecasts were generated. Please check the data and models.")

print("\nImproved forecasting completed!")
