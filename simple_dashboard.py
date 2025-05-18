import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for plots
output_dir = "dashboard_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Creating AYA Pay Grab Voucher Dashboard...")

# Load the data
historical_data = pd.read_csv("grab_voucher_timeseries.csv")
forecast_data = pd.read_csv("grab_voucher_forecast.csv")

# Convert year and month to datetime
historical_data['date'] = pd.to_datetime(historical_data[['year', 'month']].assign(day=1))
forecast_data['date'] = pd.to_datetime(forecast_data[['year', 'month']].assign(day=1))

# Calculate restock amounts (with 20% buffer)
forecast_data['restock_amount'] = (forecast_data['predicted_sold_count'] * 1.2).round().astype(int)

# Get unique price points
price_points = sorted(historical_data['Price'].unique())

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

# Create visualizations for each price point
for price in price_points:
    price_historical = historical_data[historical_data['Price'] == price]
    price_forecast = forecast_data[forecast_data['Price'] == price]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(price_historical['date'], price_historical['sold_count'], 
             marker='o', label='Historical', linewidth=2)
    
    # Plot forecasted data
    plt.plot(price_forecast['date'], price_forecast['predicted_sold_count'], 
             marker='s', linestyle='--', label='Forecast', linewidth=2)
    
    # Plot confidence interval
    plt.fill_between(price_forecast['date'], 
                    price_forecast['lower_bound'], 
                    price_forecast['upper_bound'], 
                    alpha=0.2, label='95% Confidence Interval')
    
    # Add restock amount as bar chart
    plt.bar(price_forecast['date'], price_forecast['restock_amount'], 
            alpha=0.3, width=15, label='Restock Amount')
    
    # Add annotations
    plt.title(f'Grab Voucher Demand Forecast - Price {price:,} MMK (MAPE: {price_forecast["mape"].iloc[0]:.2f}%)')
    plt.xlabel('Month')
    plt.ylabel('Voucher Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig(f"{output_dir}/dashboard_price_{price}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Create a combined visualization of all price points
plt.figure(figsize=(15, 8))

# Plot historical and forecasted data for each price
for price in price_points:
    # Historical data
    price_historical = historical_data[historical_data['Price'] == price]
    
    # Forecast data
    price_forecast = forecast_data[forecast_data['Price'] == price]
    
    # Plot
    plt.plot(price_historical['date'], price_historical['sold_count'], 
             marker='o', linewidth=2, label=f'Historical {price:,} MMK')
    plt.plot(price_forecast['date'], price_forecast['predicted_sold_count'], 
             marker='s', linestyle='--', linewidth=2, label=f'Forecast {price:,} MMK')

plt.title('Grab Voucher Demand Forecast - All Price Points')
plt.xlabel('Month')
plt.ylabel('Voucher Count')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/dashboard_all_prices.png", dpi=300, bbox_inches='tight')
plt.close()

# Create detailed forecast table with restock recommendations
table_data = forecast_data.copy()
table_data['Month-Year'] = table_data['date'].dt.strftime('%b %Y')
table_data['Price_MMK'] = table_data['Price'].apply(lambda x: f"{int(x):,} MMK")
table_data = table_data[['Month-Year', 'Price_MMK', 'predicted_sold_count', 'lower_bound', 'upper_bound', 'restock_amount']]
table_data.columns = ['Month', 'Price', 'Predicted Demand', 'Lower Bound', 'Upper Bound', 'Restock Amount']

# Save the table to CSV
table_data.to_csv(f"{output_dir}/forecast_table.csv", index=False)

# Create a summary table of total restock amounts by price
summary_data = forecast_data.groupby('Price')['restock_amount'].sum().reset_index()
summary_data['Price_MMK'] = summary_data['Price'].apply(lambda x: f"{int(x):,} MMK")
summary_data = summary_data[['Price_MMK', 'restock_amount']]
summary_data.columns = ['Price', 'Total 6-Month Restock']

# Save the summary to CSV
summary_data.to_csv(f"{output_dir}/restock_summary.csv", index=False)

print(f"Dashboard created successfully! Output files saved to {output_dir}/")
print("\nSummary of 6-Month Restock Requirements:")
for _, row in summary_data.iterrows():
    print(f"  {row['Price']}: {row['Total 6-Month Restock']:,} vouchers")

print("\nTo view the dashboard:")
print(f"1. Open the PNG files in {output_dir}/ to see the visualizations")
print(f"2. Open forecast_table.csv for detailed monthly forecasts")
print(f"3. Open restock_summary.csv for total restock requirements")
