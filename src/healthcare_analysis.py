import pandas as pd
import matplotlib.pyplot as plt


file_path = r"C:\Users\Priya\Downloads\INDIA_AQI_CLEANED.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)


def health_zone(aqi):
    if aqi <= 50:
        return "Green (Healthy)"
    elif aqi <= 100:
        return "Yellow (Sensitive Groups)"
    elif aqi <= 200:
        return "Orange (Unhealthy)"
    elif aqi <= 300:
        return "Red (Very Unhealthy)"
    else:
        return "Severe (Hazardous)"

df['Health_Zone'] = df['US_AQI'].apply(health_zone)

print("\nHealth Zone Distribution:")
print(df['Health_Zone'].value_counts())


city_health = df.groupby('City')['US_AQI'].mean().sort_values(ascending=False)

print("\nTop 10 Most Health Risk Cities:")
print(city_health.head(10))


plt.figure(figsize=(10,6))
city_health.head(10).plot(kind='bar')
plt.title("Top 10 Health Risk Cities (Based on Average AQI)")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

updated_path = r"C:\Users\Priya\Downloads\INDIA_AQI_WITH_HEALTH_ZONE.csv"
df.to_csv(updated_path, index=False)