# Global Weather & Environmental Impact Analysis

**Author:** Ishan Kaundal

## Project Overview
This repository contains a comprehensive data science project analyzing global weather patterns, detecting environmental anomalies (specifically regarding PM2.5 air quality), and building predictive time-series models. 

## Repository Contents
* `Global_Weather_Analysis_Report.pdf`: The complete, compiled project report containing all visual insights, data methodology, and ML evaluation metrics.
* `GlobalWeatherRepository.csv`: The core dataset utilized for the analysis (Not included in zip due to size, but assumed present in the environment).
* `/images`: Visualizations generated during EDA (Spatial distribution, Correlation heatmaps, Model forecasts).

## Methodology & Models
1. **Data Cleaning & Engineering:** Addressed missing parameters and created continuous time-lag features (`temp_lag_1`, `temp_lag_2`, etc.) for autoregressive forecasting.
2. **Anomaly Detection:** Deployed an `IsolationForest` (contamination=0.01) to isolate critical intersections of extreme temperatures and severe air pollution.
3. **Machine Learning:** * Trained a `RandomForestRegressor` to establish feature importance for PM2.5 accumulation.
    * Built a predictive ensemble combining `RandomForestRegressor` and `GradientBoostingRegressor` to minimize RMSE in temperature forecasting.

## Key Insights
* **Air Quality Drivers:** Visibility and atmospheric pressure are the most reliable non-linear predictors for severe PM2.5 events.
* **Forecasting Accuracy:** While baseline statistical regression performs well on short-term horizons, ensembling multiple tree-based models provides a more robust framework for handling non-linear weather fluctuations.
