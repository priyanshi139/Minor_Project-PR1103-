import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = r"C:\Users\Priya\Downloads\INDIA_AQI_COMPLETE_20251126.csv"

df = pd.read_csv(file_path)

print("Shape of dataset:", df.shape)
df.head()
df.info()
df.describe()
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)
df = df.drop(columns=[
    'Temp_80m_C',
    'Temp_120m_C',
    'Temp_180m_C',
    'Wind_Speed_80m_kmh',
    'Wind_Speed_120m_kmh',
    'UV_Index',
    'NH3_ugm3',
    'Inversion_Strength_C'
])

print("After dropping useless columns:", df.shape)
df = df.dropna(subset=['US_AQI'])
print("After removing missing AQI:", df.shape)
clean_path_csv = r"C:\Users\Priya\Downloads\INDIA_AQI_CLEANED.csv"
df.to_csv(clean_path_csv, index=False)

print("CSV file saved successfully!")