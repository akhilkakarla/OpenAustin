import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/initial_dataset_travis_county.csv"
file_path2 = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/Property_Tax_Data.csv"


# Read datasets: incarceration/incident dataset and property tax dataset
df_incarceration = pd.read_csv(file_path)
df_prop = pd.read_csv(file_path2)

# Print available columns in property tax dataset to help mapping (useful when column names vary)
print("Property tax dataset columns:", list(df_prop.columns))

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
df_incarceration = ensure_year_column(df_incarceration)

YEAR_START, YEAR_END = 2016, 2024
mask = (df_incarceration['Year'] >= YEAR_START) & (df_incarceration['Year'] <= YEAR_END)
df_range = df_incarceration.loc[mask].copy()

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

# Attempt to extract incident reports from the incarceration dataset (if it contains incident/date columns)
incident_counts = pd.Series(0, index=years_index)
try:
    df_incident_candidate = ensure_year_column_incidents(df_incarceration.copy())
    mask2 = (df_incident_candidate['Year'] >= YEAR_START) & (df_incident_candidate['Year'] <= YEAR_END)
    df2_range = df_incident_candidate.loc[mask2].copy()
    incident_counts = (
        df2_range['Year']
        .value_counts()
        .reindex(years_index, fill_value=0)
        .sort_index()
    )
except ValueError:
    # If no incident/date column found in incarceration dataset, leave incident_counts as zeros
    print("No incident date column found in incarceration dataset; skipping incident counts.")

# --- Property tax dataset processing ---
def compute_property_tax_rate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure a Year column and compute a per-record property tax rate (as percentage).

    Strategies (in order):
    - If a column with 'rate' and 'tax' in its name exists, use it.
    - Else if a tax amount and an assessed/appraised/value column exist, compute rate = tax_amount / assessed_value.
    - Else if a column clearly named like 'effective_tax_rate' exists, use it.
    - Resulting rate is returned as percent (0-100).
    """
    # Prefer explicit tax-year column names for property tax dataset
    # (this function expects the property dataset's year column to be one of these)
    year_candidates = ['taxyear', 'tax_year', 'year', 'Year', 'YearTax', 'TaxYear', 'Tax_Year']
    for yc in year_candidates:
        if yc in frame.columns:
            frame['Year'] = pd.to_numeric(frame[yc], errors='coerce')
            break
    # Fallback to general date/year inference
    if 'Year' not in frame.columns or frame['Year'].isnull().all():
        frame = ensure_year_column(frame)

    cols_lower = {c.lower(): c for c in frame.columns}

    # Candidate lists (ordered preference)
    rate_col_candidates = [
        'taxrate', 'tax_rate', 'effective_tax_rate', 'effectivetaxrate', 'total_tax_rate',
        'property_tax_rate', 'propertytaxrate', 'rate', 'percent', 'tax_rate_percent'
    ]
    tax_amount_candidates = [
        'taxamount', 'tax_amount', 'tax_paid', 'taxpaid', 'total_tax', 'total_tax_amount', 'taxlevy', 'tax_levy'
    ]
    value_candidates = [
        'assessedvalue', 'assessed_value', 'appraisedvalue', 'appraised_value', 'taxablevalue',
        'taxable_value', 'marketvalue', 'market_value', 'value', 'total_assessed_value'
    ]

    # 1) direct rate column (look for percentage strings too)
    rate_col = None
    for cand in rate_col_candidates:
        if cand in cols_lower:
            rate_col = cols_lower[cand]
            break

    if rate_col is not None:
        raw = frame[rate_col].astype(str).str.strip()
        # remove percent sign and commas
        raw_clean = raw.str.replace('%', '', regex=False).str.replace(',', '', regex=False)
        frame['property_tax_rate'] = pd.to_numeric(raw_clean, errors='coerce')
        # If values look like proportions (max <= 1.0), convert to percent
        if frame['property_tax_rate'].abs().max(skipna=True) <= 1.0:
            frame['property_tax_rate'] = frame['property_tax_rate'] * 100.0
        print(f"Using direct rate column '{rate_col}' for property tax rate.")
        return frame

    # 2) compute from tax amount and assessed/appraised value
    tax_amount_col = None
    value_col = None
    for cand in tax_amount_candidates:
        if cand in cols_lower:
            tax_amount_col = cols_lower[cand]
            break
    for cand in value_candidates:
        if cand in cols_lower:
            value_col = cols_lower[cand]
            break

    if tax_amount_col and value_col:
        tax_raw = frame[tax_amount_col].astype(str).str.replace(',', '', regex=False).str.replace('$', '', regex=False)
        val_raw = frame[value_col].astype(str).str.replace(',', '', regex=False).str.replace('$', '', regex=False)
        tax_num = pd.to_numeric(tax_raw, errors='coerce')
        val_num = pd.to_numeric(val_raw, errors='coerce')
        frame['property_tax_rate'] = (tax_num / val_num) * 100.0
        print(f"Computed property tax rate from '{tax_amount_col}' / '{value_col}'.")
        return frame

    # 3) last resort: try any string percent column or numeric column that looks like a rate
    for c in frame.columns:
        if frame[c].dtype == object:
            sample = frame[c].dropna().astype(str).head(10).tolist()
            if any('%' in s for s in sample):
                raw = frame[c].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
                frame['property_tax_rate'] = pd.to_numeric(raw, errors='coerce')
                if frame['property_tax_rate'].abs().max(skipna=True) <= 1.0:
                    frame['property_tax_rate'] = frame['property_tax_rate'] * 100.0
                print(f"Inferred property tax rate from string-percent column '{c}'.")
                return frame

    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        low = c.lower()
        if 'rate' in low or 'percent' in low or ('tax' in low and 'amount' not in low and 'amt' not in low):
            frame['property_tax_rate'] = pd.to_numeric(frame[c], errors='coerce')
            if frame['property_tax_rate'].abs().max(skipna=True) <= 1.0:
                frame['property_tax_rate'] = frame['property_tax_rate'] * 100.0
            print(f"Inferred property tax rate from numeric column '{c}'.")
            return frame

    raise ValueError('Could not determine property tax rate column from property tax dataset.\n'
                     f'Available columns: {list(frame.columns)}')


try:
    df_prop = compute_property_tax_rate_dataframe(df_prop)
    mask_prop = (df_prop['Year'] >= YEAR_START) & (df_prop['Year'] <= YEAR_END)
    df_prop_range = df_prop.loc[mask_prop].copy()
    avg_prop_tax_by_year = (
        df_prop_range.groupby('Year')['property_tax_rate']
        .mean()
        .reindex(years_index, fill_value=float('nan'))
    )
except Exception as e:
    print(f"Property tax processing failed: {e}")
    avg_prop_tax_by_year = pd.Series([float('nan')] * len(years_index), index=years_index)

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

# Create fourth y-axis for average property tax rate
ax4 = ax1.twinx()
ax4.spines['right'].set_position(('outward', 120))
color4 = 'purple'
ax4.set_ylabel('Average Property Tax Rate (%)', color=color4)
line4 = ax4.plot(avg_prop_tax_by_year.index, avg_prop_tax_by_year.values,
                 color=color4, marker='d', linewidth=2, label='Average Property Tax Rate (%)')
ax4.tick_params(axis='y', labelcolor=color4)

# Add title and legend
plt.title('Travis County Crime, Incarceration & Property Tax Trends (2016-2024)\nIncarcerated People, Sentence Duration, Reported Incidents, and Avg Property Tax Rate', 
          fontsize=14, pad=20)

# Combine legends from all three axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3 + lines4, labels1 + labels2 + labels3 + labels4, loc='upper left')

plt.tight_layout()

# Save the plot
output_path = "/Users/akhilkakarla/Desktop/OpenAustin/travis_county_incarceration_trends_with_property_tax_2016_2024.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved combined trends graph to {output_path}")
