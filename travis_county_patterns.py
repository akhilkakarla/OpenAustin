import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/initial_dataset_travis_county.csv"
file_path2 = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/Incident_Reports_Texas.csv"

df = pd.read_csv(file_path)
df2 = pd.read_csv(file_path2)

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


# Process data
df = ensure_year_column(df)

YEAR_START, YEAR_END = 2016, 2024
mask = (df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)
df_range = df.loc[mask].copy()

# Get incarcerated people count per year
years_index = list(range(YEAR_START, YEAR_END + 1))
incarcerated_counts = (
    df_range['Year']
    .value_counts()
    .reindex(years_index, fill_value=0)
    .sort_index()
)

# Get average sentence duration per year
sentence_col = find_sentence_column(df_range)
df_range['sentence_years'] = df_range[sentence_col].apply(parse_sentence_to_years)

avg_sentence_by_year = (
    df_range.groupby('Year')['sentence_years']
    .mean()
    .reindex(years_index, fill_value=float('nan'))
)

# Process incident reports data
def ensure_year_column_incidents(frame: pd.DataFrame) -> pd.DataFrame:
    if 'Year' in frame.columns:
        return frame

    # Prioritize dateofincident column
    dateofincident_candidates = [
        'dateofincident', 'DateOfIncident', 'DATEOFINCIDENT',
        'date_of_incident', 'Date_Of_Incident', 'DATE_OF_INCIDENT'
    ]
    for col in dateofincident_candidates:
        if col in frame.columns:
            frame['Year'] = pd.to_datetime(frame[col], errors='coerce').dt.year
            return frame

    # Fallback to other common incident date columns
    incident_date_candidates = [
        'incident_date', 'Incident_Date', 'INCIDENT_DATE',
        'date', 'Date', 'DATE', 'occurred_date', 'Occurred_Date',
        'timestamp', 'Timestamp', 'TIMESTAMP'
    ]
    for col in incident_date_candidates:
        if col in frame.columns:
            frame['Year'] = pd.to_datetime(frame[col], errors='coerce').dt.year
            return frame

    # Final fallback to year column
    if 'year' in frame.columns:
        frame['Year'] = pd.to_numeric(frame['year'], errors='coerce')
        return frame

    raise ValueError("Could not find 'dateofincident' column or other date columns in incident reports dataset.")

df2 = ensure_year_column_incidents(df2)
mask2 = (df2['Year'] >= YEAR_START) & (df2['Year'] <= YEAR_END)
df2_range = df2.loc[mask2].copy()

# Get incident reports count per year
incident_counts = (
    df2_range['Year']
    .value_counts()
    .reindex(years_index, fill_value=0)
    .sort_index()
)

# Create triple y-axis line plot
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot incarcerated people count (left y-axis)
color1 = 'blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Incarcerated People', color=color1)
line1 = ax1.plot(incarcerated_counts.index, incarcerated_counts.values, 
                 color=color1, marker='o', linewidth=2, label='Incarcerated People')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
ax1.set_xticks(years_index)

# Create second y-axis for average sentence duration
ax2 = ax1.twinx()
color2 = 'red'
ax2.set_ylabel('Average Prison Sentence Duration (Years)', color=color2)
line2 = ax2.plot(avg_sentence_by_year.index, avg_sentence_by_year.values, 
                 color=color2, marker='s', linewidth=2, label='Average Sentence Duration')
ax2.tick_params(axis='y', labelcolor=color2)

# Create third y-axis for incident reports
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
color3 = 'green'
ax3.set_ylabel('Number of Reported Incidents', color=color3)
line3 = ax3.plot(incident_counts.index, incident_counts.values, 
                 color=color3, marker='^', linewidth=2, label='Reported Incidents')
ax3.tick_params(axis='y', labelcolor=color3)

# Add title and legend
plt.title('Travis County Crime & Incarceration Trends (2016-2024)\nIncarcerated People, Sentence Duration, and Reported Incidents', 
          fontsize=14, pad=20)

# Combine legends from all three axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

plt.tight_layout()

# Save the plot
output_path = "/Users/akhilkakarla/Desktop/OpenAustin/travis_county_incarceration_trends_2016_2024.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved dual line graph to {output_path}")
