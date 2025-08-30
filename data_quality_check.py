import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("DATATHON - DATA QUALITY ASSESSMENT")
print("=" * 50)

# Load the datasets
print("\nLOADING DATASETS...")
visitation_data = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Visitation Data")
climate_data = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Climate Data")

print(f"Visitation data loaded: {visitation_data.shape}")
print(f"Climate data loaded: {climate_data.shape}")

print("\nVISITATION DATA OVERVIEW:")
print("-" * 30)
print("Columns:", list(visitation_data.columns))
print(f"Years covered: {visitation_data['Year'].min()}-{visitation_data['Year'].max()}")
print(f"Weeks per year: {visitation_data['Week'].min()}-{visitation_data['Week'].max()}")

# Get resort names
resort_columns = [col for col in visitation_data.columns if col not in ['Year', 'Week']]
print(f"Resorts tracked: {len(resort_columns)}")
for i, resort in enumerate(resort_columns, 1):
    print(f"  {i}. {resort}")

print("\nCLIMATE DATA OVERVIEW:")
print("-" * 30)
print("Columns:", list(climate_data.columns))
print(f"Date range: {climate_data['Year'].min()}-{climate_data['Year'].max()}")
print(f"Weather stations: {sorted(climate_data['Bureau of Meteorology station number'].unique())}")

print("\nMISSING VALUES ANALYSIS:")
print("-" * 30)
print("Visitation Data Missing Values:")
missing_vis = visitation_data.isnull().sum()
for col, missing in missing_vis.items():
    if missing > 0:
        print(f"  {col}: {missing} missing ({missing/len(visitation_data)*100:.1f}%)")
    else:
        print(f"  {col}: No missing values")

print("\nClimate Data Missing Values:")
missing_clim = climate_data.isnull().sum()
for col, missing in missing_clim.items():
    if missing > 0:
        print(f"  {col}: {missing} missing ({missing/len(climate_data)*100:.1f}%)")

print("\nBASIC STATISTICS:")
print("-" * 30)
print("Visitation Data Summary:")
print(visitation_data.describe())

print("\nClimate Data Summary:")
print(climate_data[['Maximum temperature (Degree C)', 'Minimum temperature (Degree C)', 'Rainfall amount (millimetres)']].describe())