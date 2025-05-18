import pandas as pd
import os
from datetime import datetime

print("Starting Grab voucher time-series data processing...")

# Read the Excel file
file_path = "Grab analysis.xlsx"
print(f"Reading file: {file_path}")
df = pd.read_excel(file_path)
print(f"Data loaded successfully. Total records: {len(df)}")

# Parse buyTimed into proper datetime format
print("Converting buyTimed to datetime format...")
df['buyTimed'] = pd.to_datetime(df['buyTimed'])

# Extract year and month from buyTimed
print("Extracting year and month...")
df['year'] = df['buyTimed'].dt.year
df['month'] = df['buyTimed'].dt.month

# Group by year, month, and Price, and count the number of vouchers sold
print("Grouping data by year, month, and Price...")
grouped_df = df.groupby(['year', 'month', 'Price']).size().reset_index(name='sold_count')

# Sort chronologically for each price
print("Sorting data chronologically...")
grouped_df = grouped_df.sort_values(['Price', 'year', 'month'])

# Verify all price points are included
expected_prices = [5000, 10000, 20000, 50000, 100000]
actual_prices = sorted(grouped_df['Price'].unique())
print(f"Expected price points: {expected_prices}")
print(f"Actual price points found: {actual_prices}")

# Final dataset should have these columns: ['year', 'month', 'Price', 'sold_count']
final_columns = ['year', 'month', 'Price', 'sold_count']
final_df = grouped_df[final_columns]

# Save as CSV
csv_output = "grab_voucher_timeseries.csv"
final_df.to_csv(csv_output, index=False)
print(f"Time-series dataset saved to {csv_output}")

# Save as Excel
excel_output = "grab_voucher_timeseries.xlsx"
final_df.to_excel(excel_output, index=False)
print(f"Time-series dataset also saved to {excel_output}")

# Display sample of the final dataset
print("\nSample of the final time-series dataset:")
print(final_df.head(15))

# Show summary statistics
print("\nSummary statistics by Price:")
summary = final_df.groupby('Price')['sold_count'].agg(['count', 'sum', 'mean', 'min', 'max'])
print(summary)

# Show the total number of records in the final dataset
print(f"\nTotal records in final dataset: {len(final_df)}")

# Show the date range in the dataset
min_date = df['buyTimed'].min().strftime('%Y-%m-%d')
max_date = df['buyTimed'].max().strftime('%Y-%m-%d')
print(f"Date range in original data: {min_date} to {max_date}")
