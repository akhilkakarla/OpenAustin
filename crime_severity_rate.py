import matplotlib.pyplot as plt
import pandas as pd

file_path = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/initial_dataset_travis_county.csv"

df = pd.read_csv(file_path)

def ensure_year_column(frame: pd.DataFrame) -> pd.DataFrame:
    if 'Year' in frame.columns:
        return frame

    offense_date_candidates = [
        'offensedate', 'OffenseDate', 'OFFENSEDATE',
        'offense_date', 'Offense_Date', 'OFFENSE_DATE'
    ]
    for col in offense_date_candidates:
        if col in frame.columns:
            frame['Year'] = pd.to_datetime(frame[col], errors='coerce').dt.year
            return frame

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

    raise ValueError("Could not infer 'Year'. Provide 'offensedate' or a parsable date/year column.")


def find_severity_column(frame: pd.DataFrame) -> str:
    candidates = [
        'crimeseverityrating', 'CrimeSeverityRating', 'CRIMESEVERITYRATING',
        'severity', 'Severity', 'SEVERITY'
    ]
    lower_cols = {c.lower(): c for c in frame.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    for c in frame.columns:
        if 'severity' in c.lower():
            return c
    raise ValueError("Could not find 'crimeseverityrating' column.")


df = ensure_year_column(df)

YEAR_START, YEAR_END = 2016, 2023
mask = (df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)
df_range = df.loc[mask].copy()

severity_col = find_severity_column(df_range)

# Convert to numeric; coerce invalid to NaN
df_range['severity_value'] = pd.to_numeric(df_range[severity_col], errors='coerce')

years_index = list(range(YEAR_START, YEAR_END + 1))
avg_severity_by_year = (
    df_range.groupby('Year')['severity_value']
    .mean()
    .reindex(years_index, fill_value=float('nan'))
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(avg_severity_by_year.index, avg_severity_by_year.values, color="blue")
ax.set_xlabel("Year")
ax.set_ylabel("Average crime severity rating")
ax.set_title("Average crime severity rating per year in Travis County (2016–2023)")
ax.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
ax.set_xticks(years_index)
plt.tight_layout()

output_path = "/Users/akhilkakarla/Desktop/OpenAustin/travis_county_average_crime_severity_2016_2023.png"
plt.savefig(output_path, dpi=150)
plt.close(fig)
print(f"Saved plot to {output_path}")
