import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AYA Pay Grab Voucher Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("AYA Pay Grab Voucher Forecast Dashboard")
st.markdown("Monitor Grab voucher demand forecasts with improved prediction model")

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
    predictions_with_floor['predicted_sold_count'] = np.maximum(predictions_with_floor['predicted_sold_count'], min_floor)
    predictions_with_floor['lower_bound'] = np.maximum(predictions_with_floor['lower_bound'], min_floor)
    predictions_with_floor['upper_bound'] = np.maximum(predictions_with_floor['upper_bound'], min_floor)
    return predictions_with_floor

# Function to smooth extreme variations
def smooth_extreme_variations(forecast_df, smoothing_factor=0.3):
    """Apply smoothing to reduce extreme variations between consecutive months"""
    smoothed = forecast_df.copy()
    
    # Group by Price to smooth each price series separately
    for price in smoothed['Price'].unique():
        price_data = smoothed[smoothed['Price'] == price].sort_values('date')
        
        if len(price_data) <= 1:
            continue  # Not enough points to smooth
        
        # Get indices in the original dataframe
        indices = price_data.index
        
        # Simple exponential smoothing
        for i in range(1, len(indices)):
            curr_idx = indices[i]
            prev_idx = indices[i-1]
            
            # Weighted average of current prediction and previous prediction
            current_pred = smoothed.loc[curr_idx, 'predicted_sold_count']
            prev_pred = smoothed.loc[prev_idx, 'predicted_sold_count']
            
            # Apply smoothing only if there's a large variation (more than 50%)
            if abs(current_pred - prev_pred) > 0.5 * prev_pred and prev_pred > 0:
                smoothed.loc[curr_idx, 'predicted_sold_count'] = (1 - smoothing_factor) * current_pred + smoothing_factor * prev_pred
                
                # Adjust confidence intervals proportionally
                ratio = smoothed.loc[curr_idx, 'predicted_sold_count'] / current_pred if current_pred > 0 else 1
                smoothed.loc[curr_idx, 'lower_bound'] = max(10, int(smoothed.loc[curr_idx, 'lower_bound'] * ratio))
                smoothed.loc[curr_idx, 'upper_bound'] = max(10, int(smoothed.loc[curr_idx, 'upper_bound'] * ratio))
    
    return smoothed

# Load data
@st.cache_data
def load_data():
    historical_data = pd.read_csv("grab_voucher_timeseries.csv")
    forecast_data = pd.read_csv("grab_voucher_forecast.csv")
    
    # Convert year and month to datetime
    historical_data['date'] = pd.to_datetime(historical_data[['year', 'month']].assign(day=1))
    forecast_data['date'] = pd.to_datetime(forecast_data[['year', 'month']].assign(day=1))
    
    # Define minimum floors for each price point
    min_floors = {
        5000: 20,    # Minimum 20 vouchers for 5,000 MMK
        10000: 15,   # Minimum 15 vouchers for 10,000 MMK
        20000: 10,   # Minimum 10 vouchers for 20,000 MMK
        50000: 5,    # Minimum 5 vouchers for 50,000 MMK
        100000: 3    # Minimum 3 vouchers for 100,000 MMK
    }
    
    # Apply minimum floors to each price point
    improved_forecast = pd.DataFrame()
    for price in forecast_data['Price'].unique():
        price_forecast = forecast_data[forecast_data['Price'] == price].copy()
        min_floor = min_floors.get(price, 10)  # Default to 10 if price not in dictionary
        price_forecast = add_minimum_floor(price_forecast, min_floor)
        improved_forecast = pd.concat([improved_forecast, price_forecast])
    
    # Apply smoothing to reduce extreme variations
    improved_forecast = smooth_extreme_variations(improved_forecast, smoothing_factor=0.3)
    
    return historical_data, improved_forecast

historical_data, forecast_data = load_data()

# Sidebar filters
st.sidebar.header("Filters")
price_points = sorted(historical_data['Price'].unique())
selected_prices = st.sidebar.multiselect(
    "Select Price Points (MMK)",
    options=price_points,
    default=[price_points[0]],  # Default to first price point
    format_func=lambda x: f"{int(x):,}"
)

# Default to all prices if none selected
if not selected_prices:
    selected_prices = [price_points[0]]

buffer_percentage = st.sidebar.slider(
    "Restock Buffer (%)",
    min_value=0,
    max_value=50,
    value=20,
    step=5
)

smoothing_factor = st.sidebar.slider(
    "Smoothing Factor",
    min_value=0.0,
    max_value=0.5,
    value=0.3,
    step=0.05,
    help="Higher values create smoother transitions between months (0 = no smoothing, 0.5 = heavy smoothing)"
)

# Update forecast data based on smoothing factor if changed from default
if smoothing_factor != 0.3:
    forecast_data = smooth_extreme_variations(forecast_data, smoothing_factor)

# Update restock amounts based on buffer
forecast_data['restock_amount'] = (forecast_data['predicted_sold_count'] * (1 + buffer_percentage/100)).round().astype(int)

# Filter data based on selection
filtered_historical = historical_data[historical_data['Price'].isin(selected_prices)]
filtered_forecast = forecast_data[forecast_data['Price'].isin(selected_prices)]

# Main dashboard content
tab1, tab2, tab3, tab4 = st.tabs(["Demand Forecast", "Restock Planning", "Forecast Table", "Model Explanation"])

with tab1:
    st.header("Voucher Demand Forecast")
    
    # Create visualization for each selected price
    for price in selected_prices:
        price_historical = historical_data[historical_data['Price'] == price]
        price_forecast = forecast_data[forecast_data['Price'] == price]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot historical data
        ax.plot(price_historical['date'], price_historical['sold_count'], 
                marker='o', label='Historical', linewidth=2)
        
        # Plot forecasted data
        ax.plot(price_forecast['date'], price_forecast['predicted_sold_count'], 
                marker='s', linestyle='--', label='Forecast', linewidth=2)
        
        # Plot confidence interval
        ax.fill_between(price_forecast['date'], 
                        price_forecast['lower_bound'], 
                        price_forecast['upper_bound'], 
                        alpha=0.2, label='Confidence Interval')
        
        # Add annotations
        ax.set_title(f'Grab Voucher Demand - Price {int(price):,} MMK (MAPE: {price_forecast["mape"].iloc[0]:.2f}%)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Voucher Count')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Display MAPE in a metric
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAPE", f"{price_forecast['mape'].iloc[0]:.2f}%", "Lower is better")
        with col2:
            st.metric("Accuracy", f"{100 - price_forecast['mape'].iloc[0]:.2f}%", "Higher is better")
        with col3:
            avg_monthly = int(price_forecast['predicted_sold_count'].mean())
            st.metric("Avg. Monthly Demand", f"{avg_monthly:,}", "Next 6 months")

with tab2:
    st.header("Restock Planning")
    
    # Create restock visualization for each selected price
    for price in selected_prices:
        price_forecast = forecast_data[forecast_data['Price'] == price]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot forecasted demand
        ax.plot(price_forecast['date'], price_forecast['predicted_sold_count'], 
                marker='o', label='Predicted Demand', linewidth=2)
        
        # Plot restock amount as bars
        ax.bar(price_forecast['date'], price_forecast['restock_amount'], 
               alpha=0.4, width=15, label=f'Restock Amount (+{buffer_percentage}%)')
        
        # Add annotations
        ax.set_title(f'Grab Voucher Restock Plan - Price {int(price):,} MMK')
        ax.set_xlabel('Month')
        ax.set_ylabel('Voucher Count')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
    
    # Display restock summary
    st.subheader("6-Month Restock Summary")
    
    restock_summary = forecast_data[forecast_data['Price'].isin(selected_prices)].groupby('Price')['restock_amount'].sum().reset_index()
    restock_summary['Price_MMK'] = restock_summary['Price'].apply(lambda x: f"{int(x):,} MMK")
    
    # Create columns for each price
    cols = st.columns(len(restock_summary))
    for i, (_, row) in enumerate(restock_summary.iterrows()):
        with cols[i]:
            st.metric(
                f"Price: {row['Price_MMK']}", 
                f"{int(row['restock_amount']):,}", 
                "Total vouchers for next 6 months"
            )
    
    # Download button for restock plan
    restock_data = forecast_data[forecast_data['Price'].isin(selected_prices)].copy()
    restock_data['Price_MMK'] = restock_data['Price'].apply(lambda x: f"{int(x):,} MMK")
    restock_data['Month'] = restock_data['date'].dt.strftime('%b %Y')
    restock_csv = restock_data[['Month', 'Price_MMK', 'predicted_sold_count', 'restock_amount']].to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Restock Plan (CSV)",
        data=restock_csv,
        file_name="grab_restock_plan.csv",
        mime="text/csv",
    )

with tab3:
    st.header("Detailed Forecast Table")
    
    # Create a detailed table
    table_data = forecast_data[forecast_data['Price'].isin(selected_prices)].copy()
    table_data['Month'] = table_data['date'].dt.strftime('%b %Y')
    table_data['Price_MMK'] = table_data['Price'].apply(lambda x: f"{int(x):,} MMK")
    
    # Format the table
    display_table = table_data[[
        'Month', 'Price_MMK', 'predicted_sold_count', 
        'lower_bound', 'upper_bound', 'restock_amount', 'mape'
    ]].rename(columns={
        'predicted_sold_count': 'Predicted Demand',
        'lower_bound': 'Lower Bound',
        'upper_bound': 'Upper Bound',
        'restock_amount': 'Restock Amount',
        'mape': 'MAPE (%)',
        'Price_MMK': 'Price'
    })
    
    # Sort by Price and Month
    display_table['Price_sort'] = display_table['Price'].str.replace(',', '').str.replace(' MMK', '').astype(int)
    display_table['Month_sort'] = pd.to_datetime(display_table['Month'], format='%b %Y')
    display_table = display_table.sort_values(['Price_sort', 'Month_sort'])
    display_table = display_table.drop(['Price_sort', 'Month_sort'], axis=1)
    
    # Show the table
    st.dataframe(display_table, use_container_width=True)
    
    # Download button for forecast data
    forecast_csv = display_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast Table (CSV)",
        data=forecast_csv,
        file_name="grab_forecast_table.csv",
        mime="text/csv",
    )

with tab4:
    st.header("Model Improvements")
    
    st.markdown("""
    ### Key Model Enhancements
    
    The forecasting model has been improved to address issues with extreme variations and zero predictions:
    
    1. **Minimum Floor Values**: Each price point now has a minimum forecast value to avoid zero predictions:
       - 5,000 MMK: 20 vouchers minimum
       - 10,000 MMK: 15 vouchers minimum
       - 20,000 MMK: 10 vouchers minimum
       - 50,000 MMK: 5 vouchers minimum
       - 100,000 MMK: 3 vouchers minimum
    
    2. **Smoothing Logic**: Reduces extreme variations between consecutive months
       - Adjustable via the "Smoothing Factor" slider in the sidebar
       - Higher values create smoother transitions (less extreme jumps)
       - Only applies when month-to-month variation exceeds 50%
    
    3. **Wider Confidence Intervals**: More realistic uncertainty bounds for better planning
    
    4. **Additive Seasonality**: Prevents multiplicative effects from causing unrealistic spikes
    
    5. **Monthly Patterns**: Better captures recurring monthly demand patterns
    """)
    
    # Show before/after comparison if available
    try:
        original_forecast = pd.read_csv("grab_voucher_forecast.csv")
        original_forecast['date'] = pd.to_datetime(original_forecast[['year', 'month']].assign(day=1))
        
        # Compare original vs improved for a selected price
        if selected_prices:
            price = selected_prices[0]
            
            st.subheader(f"Before vs After Comparison - Price {int(price):,} MMK")
            
            orig_price = original_forecast[original_forecast['Price'] == price]
            impr_price = forecast_data[forecast_data['Price'] == price]
            
            # Create comparison figure
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot original forecast
            ax.plot(orig_price['date'], orig_price['predicted_sold_count'], 
                    marker='o', label='Original Forecast', linewidth=2, color='red')
            
            # Plot improved forecast
            ax.plot(impr_price['date'], impr_price['predicted_sold_count'], 
                    marker='s', label='Improved Forecast', linewidth=2, color='green')
            
            # Add annotations
            ax.set_title(f'Forecast Comparison - Price {int(price):,} MMK')
            ax.set_xlabel('Month')
            ax.set_ylabel('Voucher Count')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis dates
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
    except Exception as e:
        st.write("Original forecast data not available for comparison.")

# Footer
st.markdown("---")
st.caption(f"AYA Pay Grab Voucher Forecast Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d')}")
st.caption("For assistance, contact the Data Science Team")
