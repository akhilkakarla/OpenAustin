import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths for bail data from 2022-2025
file_path1 = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/Bail_Data_2022.csv"
file_path2 = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/Bail_Data_2023.csv"
file_path3 = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/Bail_Data_2024.csv"
file_path4 = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/Bail_Data_2025.csv"

# Read all datasets
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)
df4 = pd.read_csv(file_path4)

# Add year column to each dataframe
df1['Year'] = 2022
df2['Year'] = 2023
df3['Year'] = 2024
df4['Year'] = 2025

# Combine all datasets
all_data = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Filter for Travis County and Felony degree
# Filter for Travis County (case-insensitive)
travis_data = all_data[all_data['County of Arrest'].str.contains('Travis', case=False, na=False)]

# Filter for Felony degree (case-insensitive, includes all felony types)
misdemeanor_data = travis_data[travis_data['Level/Degree'].str.contains('Misdemeanor', case=False, na=False)]

# Count felonies per year
misdemeanor_counts = misdemeanor_data['Year'].value_counts().sort_index()

# Create the years range for 2022-2025
years = list(range(2022, 2026))
misdemeanor_counts = misdemeanor_counts.reindex(years, fill_value=0)

# Create the visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Create bar chart
bars = ax.bar(misdemeanor_counts.index, misdemeanor_counts.values, color='#0000FF', alpha=0.7, edgecolor='black', linewidth=1)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max(misdemeanor_counts.values)*0.01,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Customize the plot
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Misdemeanors', fontsize=12, fontweight='bold')
ax.set_title('Travis County Misdemeanors by Year (2022-2025)', fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks and limits
ax.set_xticks(years)
ax.set_xlim(2021.5, 2025.5)

# Add grid for better readability
ax.grid(True, alpha=0.3, axis='y')

# Add some styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Add total count annotation
total_misdemeanors = misdemeanor_counts.sum()
ax.text(0.02, 0.98, f'Total Misdemeanors: {total_misdemeanors:,}', 
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
        verticalalignment='top')

plt.tight_layout()

# Save the plot
output_path = "/Users/akhilkakarla/Desktop/OpenAustin/travis_county_misdemeanors_2022_2025.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Misdemeanor counts by year:")
for year, count in misdemeanor_counts.items():
    print(f"{year}: {count:,} misdemeanors")

print(f"\nTotal misdemeanors in Travis County (2022-2025): {total_misdemeanors:,}")
print(f"Plot saved to: {output_path}")

