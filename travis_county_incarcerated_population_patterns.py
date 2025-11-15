import matplotlib.pyplot as plt
import pandas as pd

path_to_initial_dataset = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/initial_dataset_travis_county.csv"

df = pd.read_csv(path_to_initial_dataset)

# Ensure there is a Year column; try to derive it if not present
def ensure_year_column(frame: pd.DataFrame) -> pd.DataFrame:
    if 'Year' in frame.columns:
        return frame

    # Prefer 'offensedate' (case-insensitive) if available
    offense_date_candidates = [
        'offensedate', 'OffenseDate', 'OFFENSEDATE',
        'offense_date', 'Offense_Date', 'OFFENSE_DATE'
    ]
    for col in offense_date_candidates:
        if col in frame.columns:
            frame['Year'] = pd.to_datetime(frame[col], errors='coerce').dt.year
            return frame

    # Fallbacks if offensedate is not present
    possible_date_cols = [
        'date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'TIMESTAMP',
        'year', 'Year', 'YEAR'
    ]
    for col in possible_date_cols:
        if col in frame.columns:
            try:
                if col.lower() == 'year':
                    frame['Year'] = pd.to_numeric(frame[col], errors='coerce')
                else:
                    frame['Year'] = pd.to_datetime(frame[col], errors='coerce').dt.year
                return frame
            except Exception:
                continue

    # If no suitable column found, raise a clear error
    raise ValueError("Could not infer 'Year'. Provide 'offensedate' or a parsable date/year column.")


df = ensure_year_column(df)

# Filter to the requested year range
YEAR_START, YEAR_END = 2016, 2023
mask = (df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)
df_range = df.loc[mask].copy()

# Build counts per year, ensuring all years 2016-2023 appear
years_index = list(range(YEAR_START, YEAR_END + 1))
year_counts = (
    df_range['Year']
    .value_counts()
    .reindex(years_index, fill_value=0)
    .sort_index()
)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(year_counts.index, year_counts.values, color="green")
ax.set_xlabel("Year")
ax.set_ylabel("Count")
ax.set_title("Counts of Incarcerated Individuals in Travis County per year (2016-2023)")

# Set x-axis strictly to 2016–2023
ax.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
ax.set_xticks(years_index)

plt.tight_layout()

# Save figure next to this script
output_path = "/Users/akhilkakarla/Desktop/OpenAustin/travis_county_incarcerated_population_by_year_2016_2023.png"
plt.savefig(output_path, dpi=150)
plt.close(fig)

print(f"Saved plot to {output_path}")
