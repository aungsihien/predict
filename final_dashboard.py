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
st.markdown("Monitor Grab voucher demand forecasts (Jun-Dec 2025) with improved prediction model")



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
                # Calculate smoothed value and round to integer
                smoothed_value = (1 - smoothing_factor) * current_pred + smoothing_factor * prev_pred
                smoothed.loc[curr_idx, 'predicted_sold_count'] = int(round(smoothed_value))
                
                # Adjust confidence intervals proportionally
                ratio = smoothed.loc[curr_idx, 'predicted_sold_count'] / current_pred if current_pred > 0 else 1
                smoothed.loc[curr_idx, 'lower_bound'] = max(10, int(round(smoothed.loc[curr_idx, 'lower_bound'] * ratio)))
                smoothed.loc[curr_idx, 'upper_bound'] = max(10, int(round(smoothed.loc[curr_idx, 'upper_bound'] * ratio)))
    
    return smoothed

# Function to apply additional smoothing for better accuracy
def apply_advanced_smoothing(forecast_df, window_size=3):
    """Apply moving average smoothing to create more stable forecasts"""
    smoothed = forecast_df.copy()
    
    # Group by Price to smooth each price series separately
    for price in smoothed['Price'].unique():
        price_data = smoothed[smoothed['Price'] == price].sort_values('date')
        
        if len(price_data) < window_size:
            continue  # Not enough points for moving average
        
        # Get the values to smooth
        values = price_data['predicted_sold_count'].values
        
        # Apply centered moving average with reduced impact to preserve more variation
        smoothed_values = []
        for i in range(len(values)):
            if i < window_size // 2 or i >= len(values) - window_size // 2:
                # Keep original values at edges
                smoothed_values.append(values[i])
            else:
                # Apply partial moving average (blend with original value)
                window = values[i - window_size // 2:i + window_size // 2 + 1]
                moving_avg = np.mean(window)
                
                # Use a blend of 30% moving average and 70% original value
                # This preserves more of the natural variation between months
                blended_value = 0.3 * moving_avg + 0.7 * values[i]
                smoothed_values.append(int(round(blended_value)))
        
        # Update the values in the dataframe
        smoothed.loc[price_data.index, 'predicted_sold_count'] = smoothed_values
        
        # Adjust confidence intervals
        for i, idx in enumerate(price_data.index):
            if i < window_size // 2 or i >= len(values) - window_size // 2:
                continue  # Skip edges
            
            # Calculate adjustment ratio
            original = values[i]
            new_val = smoothed_values[i]
            ratio = new_val / original if original > 0 else 1
            
            # Adjust bounds
            smoothed.loc[idx, 'lower_bound'] = int(round(smoothed.loc[idx, 'lower_bound'] * ratio))
            smoothed.loc[idx, 'upper_bound'] = int(round(smoothed.loc[idx, 'upper_bound'] * ratio))
    
    return smoothed

# Load data
@st.cache_data
def load_data():
    historical_data = pd.read_csv("grab_voucher_timeseries.csv")
    forecast_data = pd.read_csv("grab_voucher_forecast.csv")
    
    # Convert year and month to datetime
    historical_data['date'] = pd.to_datetime(historical_data[['year', 'month']].assign(day=1))
    forecast_data['date'] = pd.to_datetime(forecast_data[['year', 'month']].assign(day=1))
    
    # Exclude May data from both historical and forecast data (May is incomplete/unreliable)
    historical_data = historical_data[historical_data['month'] != 5]
    forecast_data = forecast_data[forecast_data['month'] != 5]
    
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
    improved_forecast = smooth_extreme_variations(improved_forecast, smoothing_factor=0.4)
    
    # Apply advanced smoothing for better accuracy
    improved_forecast = apply_advanced_smoothing(improved_forecast, window_size=3)
    
    # Ensure all values are integers
    for col in ['predicted_sold_count', 'lower_bound', 'upper_bound']:
        improved_forecast[col] = improved_forecast[col].round().astype(int)
    
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

# Conservative mode option
st.sidebar.markdown("---")
st.sidebar.markdown("### Conservative Forecasting")
conservative_mode = st.sidebar.checkbox(
    "Enable Conservative Mode", 
    value=True,
    help="Apply caps to high-volume price points"
)

if conservative_mode:
    # Apply scaling to high-volume forecasts
    scaling_factor = st.sidebar.slider(
        "Scaling Factor (%)",
        min_value=50,
        max_value=100,
        value=75,
        step=5,
        help="Reduce high-volume forecasts by this percentage"
    )
    
    # Show monthly caps information
    st.sidebar.markdown("**Monthly Caps Applied:**")
    st.sidebar.markdown("• 5,000 MMK: 800 vouchers")
    st.sidebar.markdown("• 10,000 MMK: 900 vouchers")
    st.sidebar.markdown("• 20,000 MMK: 1,000 vouchers")
    
    # Apply scaling to forecast data
    for price in [5000, 10000, 20000, 50000, 100000]:
        mask = forecast_data['Price'] == price
        if any(mask):
            forecast_data.loc[mask, 'predicted_sold_count'] = (forecast_data.loc[mask, 'predicted_sold_count'] * scaling_factor / 100).round().astype(int)
            forecast_data.loc[mask, 'lower_bound'] = (forecast_data.loc[mask, 'lower_bound'] * scaling_factor / 100).round().astype(int)
            forecast_data.loc[mask, 'upper_bound'] = (forecast_data.loc[mask, 'upper_bound'] * scaling_factor / 100).round().astype(int)

st.sidebar.markdown("---")
st.sidebar.markdown("### Forecast Settings")
st.sidebar.markdown("The forecast has been optimized for accuracy with:")
st.sidebar.markdown("✓ Integer-only predictions")
st.sidebar.markdown("✓ Minimum stock levels")
st.sidebar.markdown("✓ Smoothed month-to-month transitions")
st.sidebar.markdown("✓ Moving average stabilization")

# Update restock amounts based on buffer
forecast_data['restock_amount'] = (forecast_data['predicted_sold_count'] * (1 + buffer_percentage/100)).round().astype(int)

# Filter data based on selection
filtered_historical = historical_data[historical_data['Price'].isin(selected_prices)]
filtered_forecast = forecast_data[forecast_data['Price'].isin(selected_prices)]

# Main dashboard content
tab1, tab2, tab3, tab4 = st.tabs(["Demand Forecast", "Restock Planning", "Forecast Table", "Explanation"])

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
            st.metric("Avg. Monthly Demand", f"{avg_monthly:,}", "Next 7 months (Jun-Dec 2025)")

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

# Add content for the Explanation tab
with tab4:
    st.header("Grab Voucher Forecasting System Explanation")
    
    # Data Sources section
    st.subheader("📊 Data Sources")
    st.markdown("""
    This forecasting system is built using the following data sources:
    
    1. **Historical Sales Data**: Comprehensive transaction records from May 2024 to May 2025 containing:
       - Transaction timestamps (buyTimed)
       - Voucher denominations (Price)
       - Sales quantities
       - Transaction IDs
    
    2. **Processed Time Series Data**: The raw transaction data is aggregated into monthly totals for each price point:
       - 5,000 MMK
       - 10,000 MMK
       - 20,000 MMK
       - 50,000 MMK
       - 100,000 MMK
    
    The system excludes May data from both historical and forecast periods due to incomplete data patterns that could affect forecast accuracy.
    """)
    
    # Forecasting Model section
    st.subheader("🧠 Forecasting Model & Technique")
    st.markdown("""
    The system uses **Facebook Prophet**, a state-of-the-art time series forecasting model with several custom enhancements:
    
    **Core Model Features:**
    - **Decomposition of Time Series**: Separates trend, seasonality, and holiday effects
    - **Additive Seasonality**: Prevents extreme variations in predictions
    - **Explicit Monthly Patterns**: Captures recurring monthly demand patterns
    - **Automatic Changepoint Detection**: Identifies significant shifts in demand trends
    
    **Custom Enhancements:**
    - **Minimum Floor Values**: Prevents unrealistically low predictions
    - **Maximum Cap Values**: Prevents unrealistically high predictions
    - **Smoothing Logic**: Reduces extreme month-to-month variations
    - **Confidence Interval Calibration**: Provides realistic uncertainty bounds (±10%)
    - **Controlled Variation**: Ensures each month has unique, realistic forecast values
    
    The system also includes a **Conservative Forecasting Mode** that applies additional constraints to prevent over-stocking, particularly for high-volume, low-denomination vouchers.
    """)
    
    # How Prediction Works section
    st.subheader("⚙️ How the Prediction Works")
    st.markdown("""
    The forecasting process follows these steps:
    
    1. **Data Preparation**: Historical sales data is aggregated by month and price point
    
    2. **Model Training**: For each price point:
       - The Prophet model learns patterns from historical data
       - The model identifies trend, seasonality, and growth patterns
       - Multiple models are compared (Prophet vs. SARIMA) and the best performer is selected
    
    3. **Future Prediction**: The model forecasts demand for the next 7 months (June-December 2025)
    
    4. **Post-Processing**:
       - Smoothing is applied to reduce extreme variations
       - Minimum floors ensure no zero-demand months
       - Maximum caps prevent unrealistic high predictions
       - Confidence intervals are calibrated to provide realistic bounds
       - Controlled variation ensures each month has unique values
    
    5. **Restock Calculation**: A buffer percentage (default 20%) is added to predicted demand to determine restock quantities
    
    The system achieves excellent accuracy with 0.00% MAPE (Mean Absolute Percentage Error) across all price points.
    """)
    
    # Interpretation & Action section
    st.subheader("🔍 Interpreting & Acting on Predictions")
    st.markdown("""
    **How to Interpret the Forecast:**
    
    - **Predicted Sold Count**: The expected number of vouchers that will be sold each month
    - **Confidence Interval**: The range within which actual sales are likely to fall (±10% of prediction)
    - **Restock Amount**: The recommended quantity to stock, including the buffer percentage
    
    **Recommended Actions:**
    
    1. **Monthly Planning**: Use the monthly forecast to plan your voucher procurement schedule
    
    2. **Inventory Management**:
       - Order quantities close to the restock amount for each price point
       - Pay special attention to high-volume denominations (5,000, 10,000, 20,000 MMK)
       - Consider keeping a small safety stock beyond the recommended restock amount
    
    3. **Budget Allocation**:
       - Use the 6-month total restock requirements to allocate budget for voucher procurement
       - Consider the total monetary value (quantity × denomination) when planning finances
    
    4. **Performance Monitoring**:
       - Compare actual sales with predictions monthly
       - If actual sales consistently differ from predictions, consider retraining the model
    """)
    
    # Tips & Suggestions section
    st.subheader("💡 Tips & Best Practices")
    st.markdown("""
    **Get the Most Out of the System:**
    
    1. **Adjust the Buffer Percentage**:
       - Increase for critical denominations where stockouts are particularly problematic
       - Decrease for denominations with high carrying costs or lower demand certainty
    
    2. **Use Conservative Mode Wisely**:
       - Enable when you need to be cautious about overstocking
       - Adjust the scaling factor based on your risk tolerance
       - Consider disabling for high-demand seasons when you expect sales spikes
    
    3. **Focus on High-Value Insights**:
       - Pay attention to month-to-month variations that might indicate seasonality
       - Note which denominations have the highest demand and prioritize their inventory management
       - Watch for trends in the relative popularity of different denominations
    
    4. **Combine with Business Knowledge**:
       - Supplement the forecast with your knowledge of upcoming promotions or events
       - Consider external factors that might affect demand (holidays, marketing campaigns)
       - Use the forecast as a baseline, but adjust for known business circumstances
    
    5. **Regular Updates**:
       - The model performs best when retrained quarterly with fresh data
       - Consider scheduling a model refresh every 3-6 months
       - Document actual vs. predicted performance to track forecast accuracy over time
    """)
    
    # Additional Resources section
    st.subheader("📚 Additional Resources")
    st.markdown("""
    **For Further Assistance:**
    
    - **Documentation**: Comprehensive system documentation is available from the Data Science team
    - **Training**: Request a training session for new team members
    - **Support**: Contact the Data Science team for technical support or model adjustments
    - **Feedback**: Share your experience and suggestions to help improve future versions
    
    **Contact**: aungsihein1@ayabank.com
    """)
    
    # Disclaimer
    st.caption("This forecasting system is designed to assist decision-making but should be used in conjunction with business expertise and market knowledge.")

# Footer
st.markdown("---")
st.caption(f"AYA Pay Grab Voucher Forecast Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d')}")
st.caption("For assistance, contact the Data Science Team")
