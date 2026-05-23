import pandas as pd
df = pd.read_csv('GlobalWeatherRepository.csv')
print("Total rows:", len(df))
print("Unique locations:", df['location_name'].nunique())
location_counts = df['location_name'].value_counts()
print("Top 5 locations by count:")
print(location_counts.head())
print("Time range:")
print(df['last_updated'].min(), "to", df['last_updated'].max())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv('GlobalWeatherRepository.csv')

# Filter for a specific location to build a time-series forecast
city = 'Kabul'
df_city = df[df['location_name'] == city].copy()

# Convert to datetime and sort
df_city['last_updated'] = pd.to_datetime(df_city['last_updated'])
df_city = df_city.sort_values('last_updated').reset_index(drop=True)
df_city.set_index('last_updated', inplace=True)

# Create lag features (using previous 3 time steps to predict the current one)
for i in range(1, 4):
    df_city[f'temp_lag_{i}'] = df_city['temperature_celsius'].shift(i)

# Drop missing values created by shifting
df_city.dropna(subset=['temp_lag_1', 'temp_lag_2', 'temp_lag_3'], inplace=True)

# Define X (features) and y (target)
features = ['temp_lag_1', 'temp_lag_2', 'temp_lag_3']
X = df_city[features]
y = df_city['temperature_celsius']

# Train/Test Split (80% Train, 20% Test)
split_idx = int(len(df_city) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Initialize Models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train Models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Generate Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

# Create Ensemble (Average of Random Forest and Gradient Boosting)
ens_pred = (rf_pred + gb_pred) / 2

# Calculate Evaluation Metrics
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Ensemble (RF + GB)'],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred)),
        np.sqrt(mean_squared_error(y_test, gb_pred)),
        np.sqrt(mean_squared_error(y_test, ens_pred))
    ],
    'MAE': [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, gb_pred),
        mean_absolute_error(y_test, ens_pred)
    ]
})

print(results.to_string(index=False))

# Plotting the forecast
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Temperature', color='black', linewidth=2)
plt.plot(y_test.index, lr_pred, label='Linear Regression', alpha=0.6, linestyle=':')
plt.plot(y_test.index, rf_pred, label='Random Forest', alpha=0.6, linestyle='-.')
plt.plot(y_test.index, gb_pred, label='Gradient Boosting', alpha=0.6, linestyle='--')
plt.plot(y_test.index, ens_pred, label='Ensemble (RF+GB)', color='red', linewidth=2)

plt.title(f'Temperature Forecasting with Multiple Models ({city})')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.savefig('forecast_ensemble.png')
plt.close()