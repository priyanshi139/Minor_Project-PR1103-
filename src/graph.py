import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load AQI dataset
df = pd.read_csv(r"C:\Users\Priya\Downloads\INDIA_AQI_WITH_HEALTH_ZONE.csv")

# City-wise average AQI
city_data = df.groupby(['City','Latitude','Longitude'])['US_AQI'].mean().reset_index()

# Load India country boundary
world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
india = world[world['ADMIN'] == 'India']

# Plot
fig, ax = plt.subplots(figsize=(8,10))

# Draw India boundary
india.boundary.plot(ax=ax, linewidth=1)

# Plot city dots
scatter = ax.scatter(
    city_data['Longitude'],
    city_data['Latitude'],
    c=city_data['US_AQI'],
    cmap='RdYlGn_r',
    s=70
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Average AQI")

plt.title("City-wise AQI Health Risk Map (India)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()