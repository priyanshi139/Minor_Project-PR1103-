import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\Priya\Downloads\INDIA_AQI_WITH_HEALTH_ZONE.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)


zone_counts = df['Health_Zone'].value_counts()

print("\nHealth Zone Distribution:")
print(zone_counts)

plt.figure(figsize=(8,6))
zone_counts.plot(kind='bar')
plt.title("Health Zone Distribution in India")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


city_health = df.groupby('City')['US_AQI'].mean().sort_values(ascending=False)

print("\nTop 10 Most Polluted Cities:")
print(city_health.head(10))

plt.figure(figsize=(10,6))
city_health.head(10).plot(kind='bar')
plt.title("Top 10 Most Health Risk Cities")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    monthly_trend = df.groupby(df['Date'].dt.to_period('M'))['US_AQI'].mean()

    plt.figure(figsize=(12,6))
    monthly_trend.plot()
    plt.title("Monthly AQI Trend Over Time")
    plt.ylabel("Average AQI")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()