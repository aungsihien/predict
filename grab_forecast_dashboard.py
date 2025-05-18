import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="AYA Pay Grab Voucher Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .highlight-good {
        color: #059669;
        font-weight: bold;
    }
    .highlight-warning {
        color: #D97706;
        font-weight: bold;
    }
    .highlight-danger {
        color: #DC2626;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">AYA Pay Grab Voucher Forecast Dashboard</div>', unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    # Load the historical and forecast data
    historical_data = pd.read_csv("grab_voucher_timeseries.csv")
    forecast_data = pd.read_csv("grab_voucher_forecast.csv")
    
    # Convert year and month to datetime
    historical_data['date'] = pd.to_datetime(historical_data[['year', 'month']].assign(day=1))
    forecast_data['date'] = pd.to_datetime(forecast_data[['year', 'month']].assign(day=1))
    
    # Add a type column to distinguish historical from forecast
    historical_data['type'] = 'Historical'
    forecast_data['type'] = 'Forecast'
    
    # Rename columns for consistency
    historical_data = historical_data.rename(columns={'sold_count': 'actual_sold_count'})
    
    # Combine datasets for visualization
    combined_data = pd.concat([
        historical_data[['date', 'year', 'month', 'Price', 'actual_sold_count', 'type']],
        forecast_data[['date', 'year', 'month', 'Price', 'predicted_sold_count', 'lower_bound', 'upper_bound', 'mape', 'type']]
    ], axis=0)
    
    return historical_data, forecast_data, combined_data

# Function to calculate restock amounts
def calculate_restock_amount(forecast_df, buffer_percentage=20):
    """Calculate recommended restock amounts based on forecast with buffer"""
    restock_df = forecast_df.copy()
    restock_df['restock_amount'] = (restock_df['predicted_sold_count'] * (1 + buffer_percentage/100)).round().astype(int)
    return restock_df

# Load data
historical_data, forecast_data, combined_data = load_data()

# Get unique price points
price_points = sorted(combined_data['Price'].unique())

# Sidebar filters
st.sidebar.markdown('## Filters')
selected_prices = st.sidebar.multiselect(
    'Select Price Points (MMK)',
    options=price_points,
    default=price_points,
    format_func=lambda x: f"{int(x):,}"
)

# Default to all prices if none selected
if not selected_prices:
    selected_prices = price_points

buffer_percentage = st.sidebar.slider(
    'Restock Buffer (%)',
    min_value=0,
    max_value=50,
    value=20,
    step=5,
    help="Additional percentage above forecast to order as safety stock"
)

restock_threshold = st.sidebar.slider(
    'Restock Alert Threshold (%)',
    min_value=50,
    max_value=100,
    value=80,
    step=5,
    help="Percentage of forecasted demand that triggers restock alert"
)

# Calculate restock amounts
restock_data = calculate_restock_amount(forecast_data, buffer_percentage)

# Filter data based on selection
filtered_historical = historical_data[historical_data['Price'].isin(selected_prices)]
filtered_forecast = forecast_data[forecast_data['Price'].isin(selected_prices)]
filtered_combined = combined_data[combined_data['Price'].isin(selected_prices)]
filtered_restock = restock_data[restock_data['Price'].isin(selected_prices)]

# Main dashboard content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="sub-header">Voucher Demand Forecast</div>', unsafe_allow_html=True)
    
    # Create interactive line chart with Plotly
    fig = go.Figure()
    
    for price in selected_prices:
        price_historical = filtered_historical[filtered_historical['Price'] == price]
        price_forecast = filtered_forecast[filtered_forecast['Price'] == price]
        
        # Add historical line
        fig.add_trace(go.Scatter(
            x=price_historical['date'],
            y=price_historical['actual_sold_count'],
            mode='lines+markers',
            name=f'Actual {int(price):,} MMK',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=price_forecast['date'],
            y=price_forecast['predicted_sold_count'],
            mode='lines+markers',
            name=f'Forecast {int(price):,} MMK',
            line=dict(width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([price_forecast['date'], price_forecast['date'].iloc[::-1]]),
            y=pd.concat([price_forecast['upper_bound'], price_forecast['lower_bound'].iloc[::-1]]),
            fill='toself',
            fillcolor=f'rgba(0, 100, 80, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='Monthly Voucher Demand: Historical vs. Forecast',
        xaxis_title='Month',
        yaxis_title='Voucher Count',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model accuracy metrics
    st.markdown('<div class="sub-header">Model Accuracy</div>', unsafe_allow_html=True)
    
    accuracy_df = forecast_data.drop_duplicates('Price')[['Price', 'mape']]
    accuracy_df = accuracy_df[accuracy_df['Price'].isin(selected_prices)]
    
    # Create accuracy metrics
    accuracy_cols = st.columns(len(accuracy_df))
    
    for i, (_, row) in enumerate(accuracy_df.iterrows()):
        with accuracy_cols[i]:
            mape = row['mape']
            mape_class = 'highlight-good' if mape <= 5 else 'highlight-warning' if mape <= 10 else 'highlight-danger'
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Price: {int(row['Price']):,} MMK</div>
                <div class="metric-value {mape_class}">{mape:.2f}%</div>
                <div class="metric-label">MAPE</div>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sub-header">Recommended Restock Amounts</div>', unsafe_allow_html=True)
    
    # Group by price and calculate total restock amount
    restock_summary = filtered_restock.groupby('Price')['restock_amount'].sum().reset_index()
    
    # Create bar chart for restock amounts
    fig_restock = px.bar(
        filtered_restock,
        x='date',
        y='restock_amount',
        color='Price',
        barmode='group',
        labels={'restock_amount': 'Restock Amount', 'date': 'Month'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        title='Monthly Restock Amounts by Price'
    )
    
    fig_restock.update_layout(
        xaxis_title='Month',
        yaxis_title='Restock Amount',
        legend_title='Price (MMK)',
        margin=dict(l=20, r=20, t=60, b=20),
        height=300
    )
    
    st.plotly_chart(fig_restock, use_container_width=True)
    
    # Show restock summary
    st.markdown('<div class="sub-header">Total 6-Month Restock</div>', unsafe_allow_html=True)
    
    for _, row in restock_summary.iterrows():
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Price: {int(row['Price']):,} MMK</div>
            <div class="metric-value">{int(row['restock_amount']):,}</div>
            <div class="metric-label">Total vouchers for next 6 months</div>
        </div>
        """, unsafe_allow_html=True)

# Detailed forecast table
st.markdown('<div class="sub-header">Detailed Forecast Table</div>', unsafe_allow_html=True)

# Format the table for display
display_forecast = filtered_forecast.copy()
display_forecast['Price'] = display_forecast['Price'].apply(lambda x: f"{int(x):,} MMK")
display_forecast['Month-Year'] = display_forecast['date'].dt.strftime('%b %Y')
display_forecast = display_forecast[['Month-Year', 'Price', 'predicted_sold_count', 'lower_bound', 'upper_bound']]
display_forecast.columns = ['Month', 'Price', 'Predicted Demand', 'Lower Bound', 'Upper Bound']

# Add restock threshold indicators
display_restock = filtered_restock.copy()
display_restock['Price'] = display_restock['Price'].apply(lambda x: f"{int(x):,} MMK")
display_restock['Month-Year'] = display_restock['date'].dt.strftime('%b %Y')
display_restock = display_restock[['Month-Year', 'Price', 'restock_amount']]
display_restock.columns = ['Month', 'Price', 'Restock Amount']

# Merge forecast and restock data
display_combined = pd.merge(
    display_forecast, 
    display_restock[['Month', 'Price', 'Restock Amount']], 
    on=['Month', 'Price'], 
    how='left'
)

# Sort by Price and Month
display_combined['Price_sort'] = display_combined['Price'].str.replace(',', '').str.replace(' MMK', '').astype(int)
display_combined['Month_sort'] = pd.to_datetime(display_combined['Month'], format='%b %Y')
display_combined = display_combined.sort_values(['Price_sort', 'Month_sort'])
display_combined = display_combined.drop(['Price_sort', 'Month_sort'], axis=1)

# Show the table
st.dataframe(display_combined, use_container_width=True)

# Download buttons
st.markdown('<div class="sub-header">Download Data</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Convert forecast data to CSV
    forecast_csv = filtered_forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast Data (CSV)",
        data=forecast_csv,
        file_name="grab_voucher_forecast.csv",
        mime="text/csv",
    )

with col2:
    # Convert restock data to CSV
    restock_csv = filtered_restock.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Restock Recommendations (CSV)",
        data=restock_csv,
        file_name="grab_voucher_restock.csv",
        mime="text/csv",
    )

# Footer
st.markdown("""
<div class="footer">
    <p>AYA Pay Grab Voucher Forecast Dashboard | Last updated: May 17, 2025</p>
    <p>For assistance, contact the Data Science Team</p>
</div>
""", unsafe_allow_html=True)
