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

print("Starting improved Grab voucher demand forecasting...")

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

# Print data range to verify
print(f"Data range: {df['date'].min()} to {df['date'].max()}")
print(f"Total records: {len(df)}")

# Function to calculate MAPE
def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    # Filter out zeros to avoid division by zero
    mask = actual != 0
    if sum(mask) == 0:
        return 0  # No valid points for MAPE calculation
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# Function to train Prophet model and forecast
def prophet_forecast(data, price, periods=6):
    """Train a Prophet model and forecast future demand"""
    # Prepare data for Prophet
    model_data = data[data['Price'] == price].copy()
    model_data = model_data[['date', 'sold_count']].rename(columns={'date': 'ds', 'sold_count': 'y'})
    
    # Print training data info
    print(f"  Training data for Price {price}: {len(model_data)} records from {model_data['ds'].min()} to {model_data['ds'].max()}")
    
    # Train model with appropriate parameters based on data size
    if len(model_data) >= 12:  # At least a year of data
        model = Prophet(yearly_seasonality=True, 
                       weekly_seasonality=False, 
                       daily_seasonality=False,
                       seasonality_mode='multiplicative',
                       interval_width=0.95)
    else:  # Less than a year of data
        model = Prophet(yearly_seasonality=False,
                       weekly_seasonality=False,
                       daily_seasonality=False,
                       seasonality_mode='additive',
                       interval_width=0.95)
    
    model.fit(model_data)
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    
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
    
    # Create actual vs predicted table for recent months
    validation_data = forecast[forecast['ds'].isin(model_data['ds'])].copy()
    validation_data = validation_data.merge(
        model_data, on='ds', how='inner'
    )[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]
    validation_data.columns = ['Date', 'Actual', 'Predicted', 'Lower Bound', 'Upper Bound']
    validation_data['Absolute Error'] = np.abs(validation_data['Actual'] - validation_data['Predicted'])
    validation_data['Percentage Error'] = (validation_data['Absolute Error'] / validation_data['Actual']) * 100
    validation_data = validation_data.sort_values('Date', ascending=False).head(6)  # Last 6 months
    
    # Save validation table
    validation_data.to_csv(f"{output_dir}/validation_prophet_price_{price}.csv", index=False)
    
    return forecast_result[['year', 'month', 'Price', 'predicted_sold_count', 'lower_bound', 'upper_bound', 'mape']], validation_data

# Function to train SARIMA model and forecast
def sarima_forecast(data, price, periods=6):
    """Train a SARIMA model and forecast future demand"""
    # Prepare data for SARIMA
    model_data = data[data['Price'] == price].copy()
    model_data = model_data.sort_values('date')
    y = model_data['sold_count'].values
    
    # Print training data info
    print(f"  Training data for Price {price}: {len(model_data)} records from {model_data['date'].min()} to {model_data['date'].max()}")
    
    # Determine best SARIMA parameters based on data size
    if len(model_data) >= 24:  # At least 2 years of data
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
    elif len(model_data) >= 12:  # At least 1 year of data
        order = (1, 1, 0)
        seasonal_order = (0, 1, 1, 12)
    else:  # Less than a year of data
        order = (1, 1, 0)
        seasonal_order = (0, 0, 0, 0)
    
    try:
        # Try SARIMA with seasonal component
        if len(model_data) >= 12:
            model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
        else:
            # Fallback to simpler ARIMA model if not enough data
            model = ARIMA(y, order=order)
            model_fit = model.fit()
    except:
        # Fallback to even simpler model if previous fails
        model = ARIMA(y, order=(1, 0, 0))
        model_fit = model.fit()
    
    # Forecast
    forecast_values = model_fit.forecast(steps=periods)
    
    # Ensure non-negative forecasts
    forecast_values = np.maximum(forecast_values, 0)
    
    # Calculate confidence intervals (approximate)
    std_err = np.sqrt(model_fit.cov_params().diagonal())
    lower_bound = forecast_values - 1.96 * std_err[0] * np.sqrt(np.arange(1, periods+1))
    upper_bound = forecast_values + 1.96 * std_err[0] * np.sqrt(np.arange(1, periods+1))
    
    # Ensure non-negative bounds
    lower_bound = np.maximum(lower_bound, 0)
    upper_bound = np.maximum(upper_bound, 0)
    
    # Calculate MAPE on training data
    train_predictions = model_fit.fittedvalues
    if len(train_predictions) < len(y):
        # Adjust for differencing
        train_predictions = np.concatenate([np.array([np.nan] * (len(y) - len(train_predictions))), train_predictions])
    
    # Calculate MAPE only on valid predictions (not NaN)
    valid_indices = ~np.isnan(train_predictions)
    mape = calculate_mape(y[valid_indices], train_predictions[valid_indices])
    
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
    
    # Create actual vs predicted table for recent months
    validation_data = pd.DataFrame({
        'Date': model_data['date'],
        'Actual': y,
        'Predicted': np.nan_to_num(train_predictions).round()
    })
    validation_data['Lower Bound'] = validation_data['Predicted'] - std_err[0]
    validation_data['Upper Bound'] = validation_data['Predicted'] + std_err[0]
    validation_data['Absolute Error'] = np.abs(validation_data['Actual'] - validation_data['Predicted'])
    validation_data['Percentage Error'] = np.where(
        validation_data['Actual'] > 0,
        (validation_data['Absolute Error'] / validation_data['Actual']) * 100,
        0
    )
    validation_data = validation_data.sort_values('Date', ascending=False).head(6)  # Last 6 months
    
    # Save validation table
    validation_data.to_csv(f"{output_dir}/validation_sarima_price_{price}.csv", index=False)
    
    return forecast_result[['year', 'month', 'Price', 'predicted_sold_count', 'lower_bound', 'upper_bound', 'mape']], validation_data

# Get unique price points
price_points = sorted(df['Price'].unique())
print(f"Found {len(price_points)} price points: {price_points}")

# Store forecasts from both models
prophet_forecasts = []
sarima_forecasts = []
all_validation_data = []

# Generate forecasts for each price point
for price in price_points:
    print(f"\nForecasting demand for Price {price:,} MMK...")
    
    try:
        # Prophet forecast
        print(f"  Training Prophet model...")
        prophet_result, prophet_validation = prophet_forecast(df, price)
        prophet_forecasts.append(prophet_result)
        prophet_mape = prophet_result['mape'].iloc[0]
        print(f"  Prophet MAPE: {prophet_mape:.2f}%")
        
        # Add model type and price to validation data
        prophet_validation['Model'] = 'Prophet'
        prophet_validation['Price'] = price
        all_validation_data.append(prophet_validation)
        
        # SARIMA forecast
        print(f"  Training SARIMA model...")
        sarima_result, sarima_validation = sarima_forecast(df, price)
        sarima_forecasts.append(sarima_result)
        sarima_mape = sarima_result['mape'].iloc[0]
        print(f"  SARIMA MAPE: {sarima_mape:.2f}%")
        
        # Add model type and price to validation data
        sarima_validation['Model'] = 'SARIMA'
        sarima_validation['Price'] = price
        all_validation_data.append(sarima_validation)
        
    except Exception as e:
        print(f"  Error forecasting for Price {price}: {str(e)}")

# Combine forecasts from both models, selecting the one with better MAPE for each price
best_forecasts = []
best_model_types = {}  # Track which model was selected for each price

for price in price_points:
    prophet_forecast_for_price = next((f for f in prophet_forecasts if f['Price'].iloc[0] == price), None)
    sarima_forecast_for_price = next((f for f in sarima_forecasts if f['Price'].iloc[0] == price), None)
    
    if prophet_forecast_for_price is not None and sarima_forecast_for_price is not None:
        prophet_mape = prophet_forecast_for_price['mape'].iloc[0]
        sarima_mape = sarima_forecast_for_price['mape'].iloc[0]
        
        if prophet_mape <= sarima_mape:
            best_forecasts.append(prophet_forecast_for_price)
            best_model_types[price] = 'Prophet'
            print(f"Price {price:,} MMK: Selected Prophet model (MAPE: {prophet_mape:.2f}%)")
        else:
            best_forecasts.append(sarima_forecast_for_price)
            best_model_types[price] = 'SARIMA'
            print(f"Price {price:,} MMK: Selected SARIMA model (MAPE: {sarima_mape:.2f}%)")
    elif prophet_forecast_for_price is not None:
        best_forecasts.append(prophet_forecast_for_price)
        best_model_types[price] = 'Prophet'
        print(f"Price {price:,} MMK: Selected Prophet model (only option)")
    elif sarima_forecast_for_price is not None:
        best_forecasts.append(sarima_forecast_for_price)
        best_model_types[price] = 'SARIMA'
        print(f"Price {price:,} MMK: Selected SARIMA model (only option)")

# Combine all best forecasts
if best_forecasts:
    final_forecast = pd.concat(best_forecasts)
    
    # Save final forecast to CSV
    forecast_file = "grab_voucher_forecast.csv"
    final_forecast.to_csv(forecast_file, index=False)
    print(f"\nFinal forecast saved to {forecast_file}")
    
    # Save final forecast to Excel with clear labeling
    excel_file = "grab_voucher_forecast.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Add the forecasts
        final_forecast.to_excel(writer, sheet_name='Forecasts', index=False)
        
        # Add a summary sheet
        summary = pd.DataFrame({
            'Price': price_points,
            'Model': [best_model_types.get(p, 'Unknown') for p in price_points],
            'MAPE': [final_forecast[final_forecast['Price'] == p]['mape'].iloc[0] for p in price_points]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Add validation data
        if all_validation_data:
            validation_df = pd.concat(all_validation_data)
            validation_df.to_excel(writer, sheet_name='Validation', index=False)
    
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
    
    # Create a detailed table of forecasts
    forecast_table = final_forecast.copy()
    forecast_table['Model'] = forecast_table['Price'].map(best_model_types)
    forecast_table = forecast_table.sort_values(['Price', 'year', 'month'])
    
    # Format the table for better readability
    forecast_table_styled = forecast_table.copy()
    forecast_table_styled['Price'] = forecast_table_styled['Price'].apply(lambda x: f"{x:,} MMK")
    forecast_table_styled['Date'] = pd.to_datetime(forecast_table_styled[['year', 'month']].assign(day=1))
    forecast_table_styled['Date'] = forecast_table_styled['Date'].dt.strftime('%b %Y')
    forecast_table_styled = forecast_table_styled[['Price', 'Date', 'predicted_sold_count', 'lower_bound', 'upper_bound', 'mape', 'Model']]
    forecast_table_styled.columns = ['Price', 'Month', 'Predicted Count', 'Lower Bound', 'Upper Bound', 'MAPE (%)', 'Model']
    
    # Save the detailed forecast table
    forecast_table_styled.to_csv(f"{output_dir}/detailed_forecast_table.csv", index=False)
    
    # Print summary of forecast accuracy
    print("\nForecast Accuracy Summary:")
    accuracy_summary = final_forecast.groupby('Price')['mape'].first().reset_index()
    for _, row in accuracy_summary.iterrows():
        status = "✅ MEETS TARGET" if row['mape'] <= 10 else "❌ BELOW TARGET"
        detail = "🌟 EXCELLENT" if row['mape'] <= 5 else ""
        print(f"Price {row['Price']:,} MMK: MAPE = {row['mape']:.2f}% - {status} {detail}")
    
    # Overall accuracy
    overall_mape = accuracy_summary['mape'].mean()
    overall_status = "✅ MEETS TARGET" if overall_mape <= 10 else "❌ BELOW TARGET"
    overall_detail = "🌟 EXCELLENT" if overall_mape <= 5 else ""
    print(f"\nOverall MAPE: {overall_mape:.2f}% - {overall_status} {overall_detail}")
    
    # Create a detailed HTML report
    html_report = f"""
    <html>
    <head>
        <title>Grab Voucher Demand Forecast Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .good {{ color: green; }}
            .excellent {{ color: darkgreen; font-weight: bold; }}
            .warning {{ color: orange; }}
            .error {{ color: red; }}
            .summary {{ background-color: #eef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Grab Voucher Demand Forecast Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Data Range: {df['date'].min().strftime('%b %Y')} to {df['date'].max().strftime('%b %Y')}</p>
            <p>Forecast Period: Jun 2025 to Nov 2025</p>
            <p>Overall MAPE: <span class="{'excellent' if overall_mape <= 5 else 'good' if overall_mape <= 10 else 'warning'}">{overall_mape:.2f}%</span></p>
        </div>
        
        <h2>Forecast Results by Price</h2>
        <table>
            <tr>
                <th>Price</th>
                <th>Model</th>
                <th>MAPE (%)</th>
                <th>Status</th>
            </tr>
    """
    
    for _, row in accuracy_summary.iterrows():
        mape_class = 'excellent' if row['mape'] <= 5 else 'good' if row['mape'] <= 10 else 'warning'
        status = "EXCELLENT" if row['mape'] <= 5 else "GOOD" if row['mape'] <= 10 else "NEEDS IMPROVEMENT"
        html_report += f"""
            <tr>
                <td>{row['Price']:,} MMK</td>
                <td>{best_model_types.get(row['Price'], 'Unknown')}</td>
                <td class="{mape_class}">{row['mape']:.2f}%</td>
                <td class="{mape_class}">{status}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Detailed Forecast</h2>
        <table>
            <tr>
                <th>Price</th>
                <th>Month</th>
                <th>Predicted Count</th>
                <th>Lower Bound</th>
                <th>Upper Bound</th>
            </tr>
    """
    
    for _, row in forecast_table_styled.iterrows():
        html_report += f"""
            <tr>
                <td>{row['Price']}</td>
                <td>{row['Month']}</td>
                <td>{row['Predicted Count']}</td>
                <td>{row['Lower Bound']}</td>
                <td>{row['Upper Bound']}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Model Validation (Last 6 Months)</h2>
        <p>Comparison of actual vs. predicted values for recent months to verify model reliability.</p>
    """
    
    for price in price_points:
        model = best_model_types.get(price, 'Unknown')
        validation = next((v for v in all_validation_data if v['Price'].iloc[0] == price and v['Model'].iloc[0] == model), None)
        
        if validation is not None:
            html_report += f"""
            <h3>Price: {price:,} MMK (Model: {model})</h3>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Actual</th>
                    <th>Predicted</th>
                    <th>Error (%)</th>
                </tr>
            """
            
            for _, vrow in validation.iterrows():
                error_class = 'excellent' if vrow['Percentage Error'] <= 5 else 'good' if vrow['Percentage Error'] <= 10 else 'warning'
                html_report += f"""
                <tr>
                    <td>{vrow['Date'].strftime('%b %Y') if isinstance(vrow['Date'], pd.Timestamp) else vrow['Date']}</td>
                    <td>{int(vrow['Actual'])}</td>
                    <td>{int(vrow['Predicted'])}</td>
                    <td class="{error_class}">{vrow['Percentage Error']:.2f}%</td>
                </tr>
                """
            
            html_report += """
            </table>
            """
    
    html_report += """
    </body>
    </html>
    """
    
    # Save the HTML report
    with open(f"{output_dir}/forecast_report.html", "w") as f:
        f.write(html_report)
    
    print(f"\nDetailed HTML report saved to {output_dir}/forecast_report.html")
else:
    print("No forecasts were generated. Please check the data and models.")

print("\nForecasting completed!")
