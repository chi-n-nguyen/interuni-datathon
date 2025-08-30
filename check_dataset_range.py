import pandas as pd

# Load from our comprehensive analysis CSV
df = pd.read_csv('comprehensive_ski_analysis.csv')

print("ACTUAL DATASET DATE RANGES FROM COMPREHENSIVE ANALYSIS")
print("=" * 60)

print("VISITATION DATA RANGE:")
print(f"Years: {df['Year'].min()} to {df['Year'].max()}")
print(f"Unique years: {sorted(df['Year'].unique())}")

# Count records per year
yearly_counts = df.groupby('Year').size()
print(f"\nRecords per year:")
for year, count in yearly_counts.items():
    print(f"  {year}: {count} records")


print(f"\nUPDATED COVID ANALYSIS PERIODS:")
print("-" * 40)

periods = {
    'Pre-COVID': (2014, 2019),
    'COVID Era': (2020, 2022),
    'Post-COVID': (2023, 2024),
    'Full Dataset': (2014, 2024)
}

for period_name, (start, end) in periods.items():
    period_data = df[(df['Year'] >= start) & (df['Year'] <= end)]
    years = end - start + 1
    print(f"{period_name}: {start}-{end} ({years} years, {len(period_data)} records)")

print(f"\nSTATISTICAL SIGNIFICANCE RE-ASSESSMENT:")
print("-" * 40)

post_covid_data = df[df['Year'] >= 2023]
post_covid_points = len(post_covid_data)

analyses_needed = {
    'Correlation Analysis': 30,
    'Seasonal Pattern Detection': 45, 
    'Predictive Modeling': 50,
    'Weather-Visitor Relationships': 40
}

print(f"Post-COVID data points: {post_covid_points}")
print(f"Statistical adequacy:")

for analysis, min_req in analyses_needed.items():
    status = "ADEQUATE" if post_covid_points >= min_req else "INSUFFICIENT"
    print(f"  {analysis:30}: {status:12} ({post_covid_points}/{min_req})")

print(f"\nRECOMMENDation UPDATE:")
print("-" * 40)

if post_covid_points >= 45:
    print("✓ POST-COVID ANALYSIS VIABLE!")
    print("  - Sufficient data points for robust analysis")
    print("  - Strong statistical foundation")
else:
    print("→ HYBRID APPROACH RECOMMENDED")
    print("  - Post-COVID adequate but not ideal") 
    print("  - COVID-weighted full dataset remains best")