import matplotlib.pyplot as plt
import pandas as pd

dataset_path = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/initial_dataset_travis_county.csv"

df = pd.read_csv(dataset_path)

# Ensure Year column exists, prefer deriving from offensedate
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

    # Fallbacks if offensedate not present
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


def find_sentence_column(frame: pd.DataFrame) -> str:
    candidates = [
        'sentence', 'Sentence', 'SENTENCE',
        'sentence_years', 'Sentence_Years', 'SENTENCE_YEARS',
        'prison_sentence', 'Prison_Sentence', 'PRISON_SENTENCE'
    ]
    lower_cols = {c.lower(): c for c in frame.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    for c in frame.columns:
        if 'sentence' in c.lower():
            return c
    raise ValueError("Could not find a 'sentence' column. Provide a column named like 'sentence'.")


def parse_sentence_to_years(val) -> float:
    try:
        return float(val)
    except Exception:
        pass
    if isinstance(val, str):
        s = val.strip().lower()
        import re
        m = re.search(r"([0-9]+\.?[0-9]*)", s)
        if not m:
            return float('nan')
        num = float(m.group(1))
        if 'year' in s:
            return num
        if 'month' in s:
            return num / 12.0
        if 'day' in s:
            return num / 365.0
    return float('nan')


df = ensure_year_column(df)

YEAR_START, YEAR_END = 2016, 2023
mask = (df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)
df_range = df.loc[mask].copy()

sentence_col = find_sentence_column(df_range)
df_range['sentence_years'] = df_range[sentence_col].apply(parse_sentence_to_years)

years_index = list(range(YEAR_START, YEAR_END + 1))
avg_by_year = (
    df_range.groupby('Year')['sentence_years']
    .mean()
    .reindex(years_index, fill_value=float('nan'))
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(avg_by_year.index, avg_by_year.values, color="blue")
ax.set_xlabel("Year")
ax.set_ylabel("Average sentence (years)")
ax.set_title("Average prison sentence per year in Travis County (2016–2023)")
ax.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
ax.set_xticks(years_index)
plt.tight_layout()

output_path = "/Users/akhilkakarla/Desktop/OpenAustin/travis_county_average_sentence_years_2016_2023.png"
plt.savefig(output_path, dpi=150)
plt.close(fig)
print(f"Saved plot to {output_path}")
