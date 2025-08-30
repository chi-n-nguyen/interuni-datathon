import pandas as pd

# Read the readme sheet more carefully
df_readme = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="readme", header=None)

# Print all content from the readme
print("README CONTENT:")
print("="*50)

for index, row in df_readme.iterrows():
    # Check if any cell in the row has content
    content = ""
    for cell in row:
        if pd.notna(cell) and str(cell).strip():
            content += str(cell) + " "
    
    if content.strip():
        print(f"{index}: {content.strip()}")

print("\n" + "="*50)

# Also get more details about the other datasets
print("\nVISITATION DATA DETAILS:")
df_visitation = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Visitation Data")
print(f"Date range: {df_visitation['Year'].min()}-{df_visitation['Year'].max()}")
print(f"Weeks: {df_visitation['Week'].min()}-{df_visitation['Week'].max()}")
print("Ski resorts included:", [col for col in df_visitation.columns if col not in ['Year', 'Week']])

print("\nCLIMATE DATA DETAILS:")
df_climate = pd.read_excel("2025 Allianz Datathon Dataset.xlsx", sheet_name="Climate Data")
print(f"Date range: {df_climate['Year'].min()}-{df_climate['Year'].max()}")
print("Unique weather stations:", df_climate['Bureau of Meteorology station number'].nunique())
print("Weather stations:", sorted(df_climate['Bureau of Meteorology station number'].unique()))