import pandas as pd

# Read the Excel file
file_path = "2025 Allianz Datathon Dataset.xlsx"

# First, let's see what sheets are available
xl_file = pd.ExcelFile(file_path)
print("Available sheets:")
for sheet in xl_file.sheet_names:
    print(f"- {sheet}")

print("\n" + "="*50)

# Read each sheet and display basic information
for sheet_name in xl_file.sheet_names:
    print(f"\nSheet: {sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\n" + "-"*30)