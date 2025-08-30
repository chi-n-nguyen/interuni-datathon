import pandas as pd
import numpy as np

print("🌡️ WEATHER STATION TO RESORT MAPPING")
print("=" * 50)

# Load climate data
climate_data = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Climate Data")

# Create weather station to resort mapping based on README
weather_station_mapping = {
    71032: 'Thredbo',          # Thredbo AWS
    71075: 'Perisher',         # Perisher AWS - also covers Charlotte Pass
    72161: 'Charlotte Pass',   # Cabramurra SMHEA AWS - close to Charlotte Pass
    83024: 'Mt. Buller',       # Mount Buller - also covers Mt. Stirling
    83084: 'Falls Creek',      # Falls Creek
    83085: 'Mt. Hotham',       # Mount Hotham
    85291: 'Mt. Baw Baw'       # Mount Baw Baw
}

# Add resort names to climate data
climate_data['Resort'] = climate_data['Bureau of Meteorology station number'].map(weather_station_mapping)

print("📍 WEATHER STATION COVERAGE:")
for station, resort in weather_station_mapping.items():
    count = climate_data[climate_data['Bureau of Meteorology station number'] == station].shape[0]
    date_range = climate_data[climate_data['Bureau of Meteorology station number'] == station]['Year']
    if len(date_range) > 0:
        print(f"  Station {station} → {resort}: {count:,} records ({date_range.min()}-{date_range.max()})")

print("\n🎿 RESORT COVERAGE ANALYSIS:")
print("-" * 30)
# Note: Selwyn doesn't have direct weather station coverage - this is a data gap we need to address
resort_coverage = {
    'Mt. Baw Baw': '✅ Direct (Station 85291)',
    'Mt. Stirling': '⚠️  Uses Mt. Buller data (nearby)',
    'Mt. Hotham': '✅ Direct (Station 83085)',
    'Falls Creek': '✅ Direct (Station 83084)',
    'Mt. Buller': '✅ Direct (Station 83024)',
    'Selwyn': '❌ No weather station - DATA GAP',
    'Thredbo': '✅ Direct (Station 71032)',
    'Perisher': '✅ Direct (Station 71075)',
    'Charlotte Pass': '✅ Two options (71075 & 72161)'
}

for resort, coverage in resort_coverage.items():
    print(f"  {resort}: {coverage}")

print("\n🌡️ TEMPERATURE ANALYSIS BY RESORT:")
print("-" * 40)
# Create date column for filtering
climate_data['Date'] = pd.to_datetime(climate_data[['Year', 'Month', 'Day']])

# Filter for ski season months (June-September)
ski_season = climate_data[climate_data['Month'].isin([6, 7, 8, 9])]

for resort in weather_station_mapping.values():
    resort_data = ski_season[ski_season['Resort'] == resort]
    if len(resort_data) > 0:
        avg_max = resort_data['Maximum temperature (Degree C)'].mean()
        avg_min = resort_data['Minimum temperature (Degree C)'].mean()
        print(f"  {resort}:")
        print(f"    Avg Max Temp: {avg_max:.1f}°C")
        print(f"    Avg Min Temp: {avg_min:.1f}°C")
        print(f"    Temp Range: {avg_max-avg_min:.1f}°C")

print(f"\n✅ Weather mapping complete! Data ready for correlation analysis.")
print(f"⚠️  Note: Selwyn resort has no direct weather data - will need interpolation or use nearest station.")