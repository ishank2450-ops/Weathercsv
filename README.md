# Global Weather & Environmental Impact Analysis

**Author:** Ishan Kaundal

## Project Overview
This project is an end-to-end data science and environmental analysis pipeline built using Python. It explores a high-frequency global weather dataset to unearth macro-climate patterns, isolate severe environmental anomalies (specifically regarding PM2.5 air pollution), and develop a robust machine learning ensemble for time-series temperature forecasting.

## Repository Contents
* 'Global_Weather_Analysis_Report.pdf': The compiled project report containing all visual insights, data methodology, and ML evaluation metrics.
* 'images': Visualizations generated during EDA (Spatial distribution, Correlation heatmaps, Model forecasts, Anomaly scatter plots).
* 'anomaly.py' , 'forcasting.py' , 'uniqueanalysis.py': The core script containing the data cleaning, model training, and visualization code.

## Dataset Overview
The dataset comprises **142,288 globally distributed weather records** across 257 unique locations. Key features include continuous meteorological data such as:
* Temperature (°C / °F)
* Atmospheric Pressure & Humidity
* Wind Speed & Direction
* Visibility & Precipitation
* Air Quality Indices (PM2.5, PM10, Carbon Monoxide, Ozone)

## Methodology

### 1.Exploratory Data Analysis (EDA) & Spatial Mapping
* **Geographical Trends:** Mapped global latitude and longitude coordinates against temperature to visualize Earth's thermal gradients and equatorial clustering.
* **Environmental Correlation:** Computed linear correlation matrices to evaluate the relationship between core weather parameters and fine particulate matter (PM2.5) accumulation.

### 2.Anomaly Detection
* **Algorithm:** `IsolationForest` (Scikit-learn)
* **Configuration:** Contamination threshold set to 1% (`contamination=0.01`).
* **Objective:** Isolate critical intersections of extreme temperatures and severe air pollution to identify highly anomalous weather/smog events.

### 3.Feature Importance Modeling
* **Algorithm:** `RandomForestRegressor`
* **Objective:** Moved beyond linear constraints to establish non-linear feature importance for predicting PM2.5 severity using Gini impurity metrics.

### 4.Time-Series Temperature Forecasting
Isolated high-frequency historical data for specific locations (e.g., Kabul) to build an autoregressive predictive system.
* **Feature Engineering:** Created continuous time-lag features (`temp_lag_1`, `temp_lag_2`, `temp_lag_3`).
* **Models Trained:** * Linear Regression (Baseline)
  * Random Forest Regressor (Non-linear Bagging)
  * Gradient Boosting Regressor (Sequential Boosting)
* **Ensemble Strategy:** Averaged the outputs of the Tree-based models (RF + GB) to minimize residual error and increase forecasting stability.

## Key Insights & Results

1. **Anomaly Detection Outcomes:** * Out of 142k records, 1,423 critical anomalies were flagged.
   * The average PM2.5 for anomalous records was remarkably high at **249.5**, maxing out at **1,614.1**—indicating severe, localized pollution events rather than global trends.
2. **Air Quality Drivers:** * **Visibility** and **Atmospheric Pressure** emerged as the strongest non-linear predictors for severe PM2.5 accumulation. Hotter, stagnant high-pressure air masses contribute significantly to trapping particulate matter.
3. **Forecasting Accuracy:**
   * The **Ensemble Model (RF + GB)** outperformed individual tree models in time-series forecasting, achieving the lowest error rates on the holdout test set:
     * **RMSE:** 2.90
     * **MAE:** 2.10

## Technologies & Libraries Used
* **Python 3.x**
* **Pandas & NumPy** (Data manipulation and feature engineering)
* **Scikit-Learn** (Machine learning, Anomaly Detection, Ensembling)
* **Matplotlib & Seaborn** (Data visualization and spatial plotting)

## How to Run the Code
1. Clone this repository: `git clone https://github.com/yourusername/weather-analysis.git`
2. Ensure you have the required libraries installed:
