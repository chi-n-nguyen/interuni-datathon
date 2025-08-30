import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("📊 HISTORICAL VISITATION PATTERNS ANALYSIS")
print("=" * 60)

# Load visitation data
visitation_data = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Visitation Data")
resort_columns = [col for col in visitation_data.columns if col not in ['Year', 'Week']]

print(f"📈 Analyzing {len(resort_columns)} resorts over {visitation_data['Year'].nunique()} years")

print("\n🏆 RESORT POPULARITY RANKING (Average Annual Visitors):")
print("-" * 50)
resort_totals = {}
for resort in resort_columns:
    annual_avg = visitation_data.groupby('Year')[resort].sum().mean()
    resort_totals[resort] = annual_avg

# Sort resorts by popularity
sorted_resorts = sorted(resort_totals.items(), key=lambda x: x[1], reverse=True)
for i, (resort, visitors) in enumerate(sorted_resorts, 1):
    print(f"{i:2d}. {resort:<15} {visitors:>8,.0f} visitors/year")

print("\n📅 PEAK WEEKS ANALYSIS:")
print("-" * 30)
# Calculate average visitors by week across all years and resorts
week_analysis = visitation_data.groupby('Week')[resort_columns].sum().mean(axis=1).sort_values(ascending=False)

print("Most Popular Weeks (across all resorts):")
for i, (week, avg_visitors) in enumerate(week_analysis.head(8).items(), 1):
    print(f"  Week {week:2d}: {avg_visitors:>6,.0f} avg visitors")

print(f"\nLeast Popular Weeks:")
for i, (week, avg_visitors) in enumerate(week_analysis.tail(5).items(), 1):
    print(f"  Week {week:2d}: {avg_visitors:>6,.0f} avg visitors")

print("\n🎿 RESORT-SPECIFIC PEAK WEEKS:")
print("-" * 35)
for resort in sorted_resorts[:5]:  # Top 5 resorts only
    resort_name = resort[0]
    resort_weeks = visitation_data.groupby('Week')[resort_name].mean().sort_values(ascending=False)
    peak_week = resort_weeks.index[0]
    peak_visitors = resort_weeks.iloc[0]
    print(f"{resort_name:<15} Peak: Week {peak_week:2d} ({peak_visitors:>6,.0f} avg)")

print("\n📈 YEAR-OVER-YEAR TRENDS:")
print("-" * 30)
# Calculate year-over-year growth
yearly_totals = visitation_data.groupby('Year')[resort_columns].sum().sum(axis=1)

growth_rates = []
for i in range(1, len(yearly_totals)):
    current_year = yearly_totals.iloc[i]
    previous_year = yearly_totals.iloc[i-1]
    growth = (current_year - previous_year) / previous_year * 100
    year = yearly_totals.index[i]
    growth_rates.append((year, growth))
    
    if abs(growth) > 15:  # Highlight significant changes
        trend = "📈" if growth > 0 else "📉"
        print(f"  {year}: {growth:+6.1f}% {trend}")

print("\n🎯 KEY INSIGHTS FOR 2026:")
print("-" * 30)
print(f"🏔️  TOP RESORT: {sorted_resorts[0][0]} ({sorted_resorts[0][1]:,.0f} visitors/year)")
print(f"🗓️  PEAK WEEK: Week {week_analysis.index[0]} (highest average attendance)")
print(f"❄️  SHOULDER SEASONS: Weeks 1-3 and 13-15 (lowest crowds)")

# Calculate coefficient of variation for each resort (consistency)
print(f"\n📊 RESORT CONSISTENCY (Lower = More Predictable):")
consistency_analysis = {}
for resort in resort_columns:
    resort_data = visitation_data[resort]
    cv = (resort_data.std() / resort_data.mean()) * 100  # Coefficient of variation
    consistency_analysis[resort] = cv

sorted_consistency = sorted(consistency_analysis.items(), key=lambda x: x[1])
print("Most Consistent Resorts:")
for i, (resort, cv) in enumerate(sorted_consistency[:3], 1):
    print(f"  {i}. {resort:<15} CV: {cv:.1f}%")

print("Most Variable Resorts:")
for i, (resort, cv) in enumerate(sorted_consistency[-3:], 1):
    print(f"  {i}. {resort:<15} CV: {cv:.1f}%")

# Save summary statistics
summary_stats = {
    'resort_rankings': sorted_resorts,
    'peak_weeks': dict(week_analysis.head(8)),
    'off_peak_weeks': dict(week_analysis.tail(5)),
    'consistency_rankings': sorted_consistency
}

print(f"\n✅ Historical analysis complete!")
print(f"🎯 Ready for 2026 predictions and resort recommendations!")