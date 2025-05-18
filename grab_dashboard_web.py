import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
st.markdown("Monitor Grab voucher demand forecasts and decide when to restock")

# Load data
@st.cache_data
def load_data():
    historical_data = pd.read_csv("grab_voucher_timeseries.csv")
    forecast_data = pd.read_csv("grab_voucher_forecast.csv")
    
    # Convert year and month to datetime
    historical_data['date'] = pd.to_datetime(historical_data[['year', 'month']].assign(day=1))
    forecast_data['date'] = pd.to_datetime(forecast_data[['year', 'month']].assign(day=1))
    
    # Calculate restock amounts (with 20% buffer)
    forecast_data['restock_amount'] = (forecast_data['predicted_sold_count'] * 1.2).round().astype(int)
    
    return historical_data, forecast_data

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

# Update restock amounts based on buffer
forecast_data['restock_amount'] = (forecast_data['predicted_sold_count'] * (1 + buffer_percentage/100)).round().astype(int)

# Filter data based on selection
filtered_historical = historical_data[historical_data['Price'].isin(selected_prices)]
filtered_forecast = forecast_data[forecast_data['Price'].isin(selected_prices)]

# Main dashboard content
tab1, tab2, tab3 = st.tabs(["Demand Forecast", "Restock Planning", "Forecast Table"])

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
                        alpha=0.2, label='95% Confidence Interval')
        
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

# Footer
st.markdown("---")
st.caption(f"AYA Pay Grab Voucher Forecast Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d')}")
st.caption("For assistance, contact the Data Science Team")
