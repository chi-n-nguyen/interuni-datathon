import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

print("WEATHER-VISITATION CORRELATION ANALYSIS")
print("=" * 60)

# Load datasets
visitation_data = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Visitation Data")
climate_data = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Climate Data")

# Weather station mapping
weather_station_mapping = {
 71032: 'Thredbo', 71075: 'Perisher', 72161: 'Charlotte Pass',
 83024: 'Mt. Buller', 83084: 'Falls Creek', 83085: 'Mt. Hotham', 85291: 'Mt. Baw Baw'
}

climate_data['Resort'] = climate_data['Bureau of Meteorology station number'].map(weather_station_mapping)
climate_data['Date'] = pd.to_datetime(climate_data[['Year', 'Month', 'Day']])

# Create ski season week mapping function
def get_ski_week_dates(year, week):
 """Convert year/week to actual dates (approximate)"""
 # Week 1 starts around June 8th (based on README)
 base_date = datetime(year, 6, 8) 
 week_start = base_date + timedelta(weeks=int(week-1))
 week_end = week_start + timedelta(days=6)
 return week_start, week_end

print("CREATING WEATHER-VISITATION DATASET...")

# Create combined dataset
combined_data = []

for _, vis_row in visitation_data.iterrows():
 year = vis_row['Year']
 week = vis_row['Week']
 
 week_start, week_end = get_ski_week_dates(year, week)
 
 # Get weather data for each resort during this week
 week_climate = climate_data[
 (climate_data['Date'] >= week_start) & 
 (climate_data['Date'] <= week_end) &
 (climate_data['Year'] == year)
 ]
 
 # Calculate weekly weather averages for each resort
 for resort in ['Mt. Baw Baw', 'Mt. Stirling', 'Mt. Hotham', 'Falls Creek', 
 'Mt. Buller', 'Thredbo', 'Perisher', 'Charlotte Pass']:
 
 # Get weather data for this resort (handle special cases)
 if resort == 'Mt. Stirling':
 resort_weather = week_climate[week_climate['Resort'] == 'Mt. Buller']
 elif resort == 'Selwyn':
 # Skip Selwyn for now - no weather data
 continue
 else:
 resort_weather = week_climate[week_climate['Resort'] == resort]
 
 if len(resort_weather) > 0:
 # Calculate weekly averages
 avg_max_temp = resort_weather['Maximum temperature (Degree C)'].mean()
 avg_min_temp = resort_weather['Minimum temperature (Degree C)'].mean()
 total_rainfall = resort_weather['Rainfall amount (millimetres)'].sum()
 
 # Count potential snow days (assuming temp < 2°C + rain = snow)
 snow_days = len(resort_weather[
 (resort_weather['Maximum temperature (Degree C)'] < 2) & 
 (resort_weather['Rainfall amount (millimetres)'] > 0)
 ])
 
 combined_data.append({
 'Year': year,
 'Week': week,
 'Resort': resort,
 'Visitors': vis_row[resort] if resort in vis_row else 0,
 'Avg_Max_Temp': avg_max_temp,
 'Avg_Min_Temp': avg_min_temp,
 'Total_Rainfall': total_rainfall,
 'Snow_Days': snow_days,
 'Avg_Temp': (avg_max_temp + avg_min_temp) / 2
 })

# Create DataFrame
df = pd.DataFrame(combined_data)
df = df.dropna() # Remove rows with missing weather data

print(f" Combined dataset created: {len(df)} records")
print(f" Covering {df['Resort'].nunique()} resorts over {df['Year'].nunique()} years")

print("\n CORRELATION ANALYSIS:")
print("-" * 40)

# Calculate correlations for each resort
correlations = {}
for resort in df['Resort'].unique():
 resort_data = df[df['Resort'] == resort]
 
 if len(resort_data) > 20: # Need sufficient data points
 temp_corr = resort_data['Visitors'].corr(resort_data['Avg_Temp'])
 rain_corr = resort_data['Visitors'].corr(resort_data['Total_Rainfall'])
 snow_corr = resort_data['Visitors'].corr(resort_data['Snow_Days'])
 
 correlations[resort] = {
 'Temperature': temp_corr,
 'Rainfall': rain_corr, 
 'Snow_Days': snow_corr
 }
 
 print(f"\n{resort}:")
 print(f" Temperature vs Visitors: {temp_corr:.3f}")
 print(f" Rainfall vs Visitors: {rain_corr:.3f}")
 print(f" Snow Days vs Visitors: {snow_corr:.3f}")

print("\n OPTIMAL WEATHER CONDITIONS:")
print("-" * 40)

# Find optimal temperature ranges
high_visitor_weeks = df[df['Visitors'] > df['Visitors'].quantile(0.75)]
low_visitor_weeks = df[df['Visitors'] < df['Visitors'].quantile(0.25)]

print(f"High Visitor Weeks (top 25%):")
print(f" Avg Temperature: {high_visitor_weeks['Avg_Temp'].mean():.1f}°C")
print(f" Avg Rainfall: {high_visitor_weeks['Total_Rainfall'].mean():.1f}mm")
print(f" Avg Snow Days: {high_visitor_weeks['Snow_Days'].mean():.1f}")

print(f"\nLow Visitor Weeks (bottom 25%):")
print(f" Avg Temperature: {low_visitor_weeks['Avg_Temp'].mean():.1f}°C")
print(f" Avg Rainfall: {low_visitor_weeks['Total_Rainfall'].mean():.1f}mm")
print(f" Avg Snow Days: {low_visitor_weeks['Snow_Days'].mean():.1f}")

print("\n KEY INSIGHTS FOR 2026 PLANNING:")
print("-" * 40)
# Identify sweet spot conditions
optimal_temp = high_visitor_weeks['Avg_Temp'].mean()
optimal_rainfall = high_visitor_weeks['Total_Rainfall'].mean()

print(f" Optimal temperature range: {optimal_temp-1:.1f}°C to {optimal_temp+1:.1f}°C")
print(f" Optimal snow conditions: {optimal_rainfall:.1f}mm weekly rainfall")
print(f" Temperature correlation varies by resort - some prefer slightly warmer conditions")

# Save the combined dataset for further analysis
df.to_csv('weather_visitation_combined.csv', index=False)
print(f"\n Analysis complete! Data saved to weather_visitation_combined.csv")