import pandas as pd

print("CHECKING ACTUAL DATASET DATE RANGES")
print("=" * 50)

# Load datasets
visitation_data = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Visitation Data")
climate_data = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Climate Data")

print("VISITATION DATA:")
print(f"Years: {visitation_data['Year'].min()} to {visitation_data['Year'].max()}")
print(f"Unique years: {sorted(visitation_data['Year'].unique())}")
print(f"Records per year: {visitation_data.groupby('Year').size()}")

print(f"\nCLIMATE DATA:")
print(f"Years: {climate_data['Year'].min()} to {climate_data['Year'].max()}")
print(f"Date range detailed:")
print(f"  Earliest: {climate_data['Year'].min()}-{climate_data['Month'].min():02d}-{climate_data['Day'].min():02d}")

# Get latest date
latest_year = climate_data['Year'].max()
latest_data = climate_data[climate_data['Year'] == latest_year]
latest_month = latest_data['Month'].max()
latest_day_data = latest_data[latest_data['Month'] == latest_month]
latest_day = latest_day_data['Day'].max()

print(f"  Latest: {latest_year}-{latest_month:02d}-{latest_day:02d}")

print(f"\nRECORDS BY YEAR:")
yearly_counts = climate_data.groupby('Year').size()
for year, count in yearly_counts.items():
    print(f"  {year}: {count:,} records")

print(f"\n2025 DATA BREAKDOWN:")
data_2025 = climate_data[climate_data['Year'] == 2025]
monthly_2025 = data_2025.groupby('Month').size()
print("Monthly counts in 2025:")
for month, count in monthly_2025.items():
    print(f"  Month {month:02d}: {count} records")

# Check if we have 2025 visitation data
print(f"\n2025 VISITATION DATA CHECK:")
if 2025 in visitation_data['Year'].values:
    data_2025_vis = visitation_data[visitation_data['Year'] == 2025]
    print(f"2025 visitation records: {len(data_2025_vis)}")
    print(f"Weeks covered: {sorted(data_2025_vis['Week'].unique())}")
else:
    print("No 2025 visitation data found")
    
print(f"\nCOVID PERIOD ANALYSIS UPDATE:")
if 2025 in visitation_data['Year'].values:
    print("POST-COVID period should be: 2023-2025 (3 years)")
    post_covid = visitation_data[visitation_data['Year'] >= 2023]
    print(f"Post-COVID data points: {len(post_covid)} records")
else:
    print("POST-COVID period: 2023-2024 (2 years)")
    post_covid = visitation_data[visitation_data['Year'] >= 2023]
    print(f"Post-COVID data points: {len(post_covid)} records")