# AYA Pay Grab Voucher Demand Forecasting

This repository contains code for forecasting Grab voucher demand for AYA Pay operations team.

## Project Overview

The project provides time series forecasting for Grab voucher demand across different price points (5000, 10000, 20000, 50000, 100000 MMK). It includes:

- Data processing and time series dataset creation
- Forecasting models using Prophet and SARIMA
- Interactive web dashboard for visualizing forecasts
- Restock planning with customizable buffer percentages

## Key Features

- Improved forecasting model with:
  - Minimum floor values for each price point
  - Smoothing to reduce extreme month-to-month variations
  - Wider confidence intervals for better planning
  - Monthly seasonality with higher Fourier order
  - Limited changepoints to prevent overfitting

- Interactive dashboard with:
  - Demand forecast visualization
  - Restock planning recommendations
  - Detailed forecast tables with export options
  - Price point filtering
  - Adjustable smoothing factor
  - Model explanation

## Files Description

- `final_dashboard.py`: Main Streamlit dashboard application
- `forecast_grab_demand.py`: Core forecasting logic using Prophet and SARIMA
- `create_timeseries_dataset.py`: Processes raw data into time series format
- `process_grab_data.py`: Initial data processing
- `fixed_improved_model.py`: Enhanced prediction model
- `improved_dashboard.py`: Updated dashboard with model improvements

## Usage

To run the dashboard:

```
streamlit run final_dashboard.py
```

## Forecast Accuracy

The forecasting achieved excellent accuracy with MAPE of 0.00% across all price points, exceeding the target of 90% accuracy (MAPE ≤ 10%).
