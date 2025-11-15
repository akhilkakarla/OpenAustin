import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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


# Process data
df = ensure_year_column(df)

YEAR_START, YEAR_END = 2016, 2023
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

# Process crime severity rating data from the first dataset
# Find the crime severity rating column
severity_col = find_severity_column(df_range)

# Convert to numeric; coerce invalid to NaN
df_range['severity_value'] = pd.to_numeric(df_range[severity_col], errors='coerce')

# Calculate average crime severity rating per year
avg_severity_by_year = (
    df_range.groupby('Year')['severity_value']
    .mean()
    .reindex(years_index, fill_value=float('nan'))
)

print(f"Calculated average crime severity ratings for {len(avg_severity_by_year.dropna())} years")

# Create triple y-axis line plot
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot incarcerated people count (left y-axis)
color1 = 'green'
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Incarcerated People', color=color1)
line1 = ax1.plot(incarcerated_counts.index, incarcerated_counts.values, 
                 color=color1, marker='o', linewidth=5, label='Incarcerated People')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
ax1.set_xticks(years_index)

# Create second y-axis for average sentence duration
ax2 = ax1.twinx()
color2 = 'red'
ax2.set_ylabel('Average Prison Sentence Duration (Years)', color=color2)
line2 = ax2.plot(avg_sentence_by_year.index, avg_sentence_by_year.values, 
                 color=color2, marker='s', linewidth=5, label='Average Sentence Duration')
ax2.tick_params(axis='y', labelcolor=color2)

# Create third y-axis for average crime severity rating
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
color3 = 'blue'
ax3.set_ylabel('Average Crime Severity Rating', color=color3)
# Filter out NaN values for plotting
severity_data_for_plot = avg_severity_by_year.dropna()
if len(severity_data_for_plot) > 0:
    line3 = ax3.plot(severity_data_for_plot.index, severity_data_for_plot.values, 
                     color=color3, marker='^', linewidth=5, label='Average Crime Severity Rating')
    
    # Set custom y-axis tick spacing for crime severity rating
    # Get current y-axis limits to calculate appropriate ticks
    ax3.relim()  # Recalculate limits
    ax3.autoscale()  # Auto-scale to fit data
    ymin, ymax = ax3.get_ylim()
    
    # Set tick interval (adjust this value to change spacing)
    # 0.2 = ticks every 0.2 units, 0.5 = ticks every 0.5 units, etc.
    tick_interval = 0.2
    
    # Calculate tick positions
    # Round down min and round up max to nice intervals
    start_tick = np.floor(ymin / tick_interval) * tick_interval
    end_tick = np.ceil(ymax / tick_interval) * tick_interval
    custom_ticks = np.arange(start_tick, end_tick + tick_interval, tick_interval)
    
    # Apply custom ticks
    ax3.set_yticks(custom_ticks)
    ax3.set_ylim(start_tick, end_tick)
    ax3.tick_params(axis='y', labelcolor=color3)
else:
    print("Warning: No valid crime severity data to plot")
    line3 = None
    ax3.tick_params(axis='y', labelcolor=color3)

# Add title and legend
plt.title('Travis County Crime & Incarceration Trends (2016-2023)\nIncarcerated People, Sentence Duration, and Crime Severity Rating', 
          fontsize=14, pad=20)

# Combine legends from all three axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
if len(severity_data_for_plot) > 0:
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center')
else:
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

plt.tight_layout()

# Save the plot
output_path = "/Users/akhilkakarla/Desktop/OpenAustin/travis_county_incarceration_trends_with_crime_severity_2016_2023.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nSaved triple line graph to {output_path}")
print(f"\nAverage crime severity ratings by year:")
for year, severity in avg_severity_by_year.items():
    if pd.notna(severity):
        print(f"{year}: {severity:.4f}")
    else:
        print(f"{year}: No data available")
