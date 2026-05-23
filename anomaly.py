import pandas as pd
df = pd.read_csv('GlobalWeatherRepository.csv')
print(df.info())
print(df.head())

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv('GlobalWeatherRepository.csv')

features = ['temperature_celsius', 'air_quality_PM2.5']
df_sample = df.dropna(subset=features).copy()

iso_forest = IsolationForest(contamination=0.01, random_state=42)
df_sample['anomaly'] = iso_forest.fit_predict(df_sample[features])

anomalies = df_sample[df_sample['anomaly'] == -1]
normal = df_sample[df_sample['anomaly'] == 1]

plt.figure(figsize=(10, 6))
plt.scatter(normal['temperature_celsius'], normal['air_quality_PM2.5'], color='blue', label='Normal', alpha=0.5, s=10)
plt.scatter(anomalies['temperature_celsius'], anomalies['air_quality_PM2.5'], color='red', label='Anomaly', alpha=0.5, s=10)
plt.title('Anomaly Detection: Temperature vs PM2.5')
plt.xlabel('Temperature (°C)')
plt.ylabel('PM2.5')
plt.legend()
plt.tight_layout()
plt.savefig('anomaly_temp_pm25.png')
plt.close()

print(f"Total records: {len(df_sample)}")
print(f"Anomalies detected: {len(anomalies)}")
print("\nAnomaly Description:")
print(anomalies[features].describe())