import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load data
df = pd.read_csv('GlobalWeatherRepository.csv')

# 1. Spatial Analysis & Geographical Patterns
plt.figure(figsize=(12, 6))
scatter = plt.scatter(df['longitude'], df['latitude'], c=df['temperature_celsius'], cmap='coolwarm', alpha=0.5, s=10)
plt.colorbar(scatter, label='Temperature (°C)')
plt.title('Global Temperature Distribution (Spatial Analysis)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('spatial_temperature.png')
plt.close()

# 2. Environmental Impact (Air Quality Correlation)
weather_vars = ['air_quality_PM2.5', 'temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'visibility_km']
corr_matrix = df[weather_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation between Air Quality and Weather Parameters')
plt.tight_layout()
plt.savefig('air_quality_correlation.png')
plt.close()

# 3. Feature Importance (Predicting PM2.5)
features = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'visibility_km', 'cloud', 'uv_index']
target = 'air_quality_PM2.5'

# Drop NaNs for safety
df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(features)[indices], palette='viridis', hue=np.array(features)[indices], legend=False)
plt.title('Feature Importance for Predicting PM2.5 (Random Forest)')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 4. Climate Analysis Summary (Aggregations)
country_summary = df.groupby('country').agg({
    'temperature_celsius': 'mean',
    'air_quality_PM2.5': 'mean',
    'precip_mm': 'mean'
}).sort_values('temperature_celsius', ascending=False)

print("Top 5 Hottest Countries (Avg Temp):")
print(country_summary['temperature_celsius'].head(5))
print("\nTop 5 Coldest Countries (Avg Temp):")
print(country_summary['temperature_celsius'].tail(5))
print("\nTop 5 Countries with Highest PM2.5:")
print(country_summary.sort_values('air_quality_PM2.5', ascending=False)['air_quality_PM2.5'].head(5))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load data
df = pd.read_csv('GlobalWeatherRepository.csv')

features = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'visibility_km', 'cloud', 'uv_index']
target = 'air_quality_PM2.5'

df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]

rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(features)[indices], palette='viridis')
plt.title('Feature Importance for Predicting PM2.5 (Random Forest)')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()