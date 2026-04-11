import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load AQI dataset
df = pd.read_csv(r"C:\Users\Priya\Downloads\INDIA_AQI_WITH_HEALTH_ZONE.csv")

# Calculate overall India average AQI
india_avg_aqi = df['US_AQI'].mean()

# Load world map
world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")

# Select India
india = world[world['ADMIN'] == 'India']

# Add AQI value
india['AQI'] = india_avg_aqi

# Plot
fig, ax = plt.subplots(figsize=(6,8))
india.plot(
    column='AQI',
    cmap='YlOrRd',
    edgecolor='black',
    legend=True,
    ax=ax
)

plt.title("India Average AQI (Healthcare Focus)")
plt.axis('off')
plt.show()