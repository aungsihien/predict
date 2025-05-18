import pandas as pd
import os

print("Starting data processing...")
print(f"Current working directory: {os.getcwd()}")

# Check if file exists
file_path = "Grab analysis.xlsx"
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found")
    exit(1)

print(f"Reading file: {file_path}")
try:
    # Read the Excel file
    df = pd.read_excel(file_path)
    print(f"Successfully read file. Shape: {df.shape}")
    
    # Display first few rows and columns
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    
    # Parse buyTimed into proper datetime format
    print("\nConverting buyTimed to datetime...")
    df['buyTimed'] = pd.to_datetime(df['buyTimed'])
    
    # Extract year and month
    df['year'] = df['buyTimed'].dt.year
    df['month'] = df['buyTimed'].dt.month
    
    # Group by year, month, and Price, and count vouchers
    print("\nGrouping data...")
    grouped_df = df.groupby(['year', 'month', 'Price']).size().reset_index(name='sold_count')
    
    # Sort chronologically for each price
    grouped_df = grouped_df.sort_values(['Price', 'year', 'month'])
    
    # Save the processed data
    output_file = "grab_timeseries_data.csv"
    grouped_df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")
    
    # Display sample of processed data
    print("\nSample of processed data:")
    print(grouped_df.head(10))
    
    # Show unique price points
    print("\nUnique Price values:")
    print(sorted(df['Price'].unique()))
    
except Exception as e:
    print(f"Error processing data: {str(e)}")
