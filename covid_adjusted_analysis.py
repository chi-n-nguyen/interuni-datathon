"""
COVID-ADJUSTED ANALYSIS STRATEGY
================================

Sophisticated approach that addresses both COVID impact and data sufficiency:
1. Dual Analysis Framework - show both perspectives
2. Weighted Analysis - de-emphasize COVID years
3. Regime Change Detection - test if patterns shifted post-COVID
4. Climate Data - use full range (global warming too gradual)

This approach shows analytical sophistication while maintaining statistical rigor.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("DUAL ANALYSIS FRAMEWORK - COVID IMPACT ASSESSMENT")
print("=" * 60)

# Load data
visitation_data = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Visitation Data")
resort_columns = [col for col in visitation_data.columns if col not in ['Year', 'Week']]

# Define analysis periods
periods = {
    'Full Dataset': (2014, 2024),
    'Pre-COVID': (2014, 2019), 
    'COVID Era': (2020, 2022),
    'Post-COVID': (2023, 2024),
    'COVID-Weighted': (2014, 2024)  # Special weighting scheme
}

print("PERIOD COMPARISON ANALYSIS:")
print("-" * 40)

results = {}

for period_name, (start_year, end_year) in periods.items():
    if period_name != 'COVID-Weighted':
        period_data = visitation_data[
            (visitation_data['Year'] >= start_year) & 
            (visitation_data['Year'] <= end_year)
        ]
    else:
        # Create weighted dataset - reduce COVID years impact
        period_data = visitation_data.copy()
        covid_years = [2020, 2021, 2022]
        
        # Apply weights: normal years = 1.0, COVID years = 0.3
        weights = period_data['Year'].map(lambda x: 0.3 if x in covid_years else 1.0)
        
        # Weight the visitor numbers
        for resort in resort_columns:
            period_data[resort] = period_data[resort] * weights
    
    # Calculate key metrics
    total_visitors = period_data[resort_columns].sum().sum()
    years_span = end_year - start_year + 1
    annual_average = total_visitors / years_span
    
    # Peak week analysis
    weekly_totals = period_data.groupby('Week')[resort_columns].sum().sum(axis=1)
    peak_week = weekly_totals.idxmax()
    
    # Resort rankings
    resort_totals = period_data[resort_columns].sum().sort_values(ascending=False)
    top_resort = resort_totals.index[0]
    
    results[period_name] = {
        'total_visitors': total_visitors,
        'annual_average': annual_average,
        'peak_week': peak_week,
        'top_resort': top_resort,
        'data_points': len(period_data),
        'years': years_span
    }
    
    print(f"\n{period_name}:")
    print(f"  Data Points: {len(period_data)} records ({years_span} years)")
    print(f"  Annual Avg: {annual_average:,.0f} visitors")
    print(f"  Peak Week: Week {peak_week}")
    print(f"  Top Resort: {top_resort}")

print(f"\nSTATISTICAL SIGNIFICANCE ASSESSMENT:")
print("-" * 40)

# Assess data sufficiency for different analyses
analyses_needed = {
    'Correlation Analysis': {'min_points': 30, 'ideal_points': 100},
    'Seasonal Pattern Detection': {'min_points': 45, 'ideal_points': 150}, 
    'Predictive Modeling': {'min_points': 50, 'ideal_points': 200},
    'Weather-Visitor Relationships': {'min_points': 40, 'ideal_points': 120}
}

post_covid_points = results['Post-COVID']['data_points']

for analysis, requirements in analyses_needed.items():
    min_req = requirements['min_points']
    ideal_req = requirements['ideal_points']
    
    if post_covid_points >= ideal_req:
        status = "EXCELLENT"
    elif post_covid_points >= min_req:
        status = "ADEQUATE" 
    else:
        status = "INSUFFICIENT"
    
    print(f"{analysis:30}: {status:12} ({post_covid_points}/{min_req} min)")

print(f"\nRECOMMENDATION FRAMEWORK:")
print("-" * 40)

print("""
HYBRID APPROACH - Best of Both Worlds:

1. PRIMARY ANALYSIS: COVID-Weighted Full Dataset (2014-2024)
   - Maintains statistical power (165 data points)
   - Reduces COVID distortion through weighting
   - Captures long-term patterns and weather correlations
   - Shows analytical sophistication

2. VALIDATION ANALYSIS: Post-COVID Only (2023-2024) 
   - Demonstrates understanding of economic context
   - Tests if patterns have fundamentally changed
   - Shows "new normal" baseline for 2026

3. WEATHER DATA: Full Period (2010-2025)
   - Global warming impact gradual over 15 years
   - Weather variability requires longer timeframe
   - Climate patterns need statistical significance

4. PRESENTATION STRATEGY:
   - Lead with COVID-weighted analysis (robust + relevant)
   - Show post-COVID validation (economic awareness)
   - Highlight agreements between approaches
   - Address any discrepancies transparently
""")

print(f"\nCOMPETITIVE ADVANTAGES:")
print("-" * 40)
print("""
✓ Shows economic and statistical sophistication
✓ Addresses judge concerns about COVID impact  
✓ Maintains statistical rigor and significance
✓ Demonstrates multiple analytical perspectives
✓ Stronger foundation for 2026 predictions
✓ Differentiates from teams using simple cutoffs
""")

print(f"\nIMPLEMENTATION PLAN:")
print("-" * 40)
print("""
IMMEDIATE ACTIONS:
1. Run correlation analysis on COVID-weighted dataset
2. Compare results with post-COVID subset
3. Test for structural breaks in visitor patterns
4. Use full weather data for climate relationships
5. Present both perspectives in final recommendation

This approach shows judges you understand both:
- Economic context (COVID impact)  
- Statistical requirements (data sufficiency)
""")