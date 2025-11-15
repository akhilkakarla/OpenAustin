import pandas as pd

file_path = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/Austin_Crime_Data.csv"
output_path = "/Users/akhilkakarla/Desktop/OpenAustinDatasets/Austin_Crime_Data_cleaned.csv"

def ensure_year_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract year from date columns if Year column doesn't exist."""
    if 'Year' in frame.columns:
        return frame

    # Common date column names for crime data
    offense_date_candidates = [
        'offensedate', 'OffenseDate', 'OFFENSEDATE',
        'offense_date', 'Offense_Date', 'OFFENSE_DATE',
        'occurred_date', 'Occurred_Date', 'OCCURRED_DATE',
        'occurreddate', 'OccurredDate', 'OCCURREDDATE', 'Occurred Date'
    ]
    for col in offense_date_candidates:
        if col in frame.columns:
            frame['Year'] = pd.to_datetime(frame[col], errors='coerce').dt.year
            return frame

    # Fallback to other common date columns
    possible_date_cols = [
        'date', 'Date', 'DATE', 
        'timestamp', 'Timestamp', 'TIMESTAMP',
        'year', 'Year', 'YEAR',
        'incident_date', 'Incident_Date', 'INCIDENT_DATE', 'Occurred Year'
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

    raise ValueError("Could not infer 'Year'. Available columns: " + str(frame.columns.tolist()))


def find_offense_column(frame: pd.DataFrame) -> str:
    """Find the column containing offense descriptions or codes."""
    offense_column_candidates = [
        'offense', 'Offense', 'OFFENSE',
        'offense_description', 'Offense_Description', 'OFFENSE_DESCRIPTION',
        'description', 'Description', 'DESCRIPTION',
        'crime', 'Crime', 'CRIME',
        'offense_code', 'Offense_Code', 'OFFENSE_CODE',
        'offense_type', 'Offense_Type', 'OFFENSE_TYPE',
        'charge', 'Charge', 'CHARGE',
        'level_degree', 'Level/Degree', 'LEVEL_DEGREE'
        'NIBRS Offense Code and Extension Description', 'NIBRS Group', 'NIBRS Category'
    ]
    
    lower_cols = {c.lower(): c for c in frame.columns}
    for cand in offense_column_candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    
    # Look for columns containing "offense" or "crime" in the name
    for col in frame.columns:
        if 'offense' in col.lower() or 'crime' in col.lower() or 'charge' in col.lower():
            return col
    
    raise ValueError("Could not find offense column. Available columns: " + str(frame.columns.tolist()))


def classify_offense_degree(offense_text: str) -> str:
    """
    Classify offense based on Texas Penal Code.
    Returns the degree classification as it would be charged in Austin, TX.
    """
    if pd.isna(offense_text) or offense_text == '':
        return 'Unknown'
    
    offense_upper = str(offense_text).upper()
    
    # First, check if already classified in the data (most reliable)
    if 'FELONY' in offense_upper:
        if 'CAPITAL' in offense_upper:
            return 'Capital Felony'
        elif 'FIRST' in offense_upper or '1ST' in offense_upper or '1 DEGREE' in offense_upper:
            return 'First Degree Felony'
        elif 'SECOND' in offense_upper or '2ND' in offense_upper or '2 DEGREE' in offense_upper:
            return 'Second Degree Felony'
        elif 'THIRD' in offense_upper or '3RD' in offense_upper or '3 DEGREE' in offense_upper:
            return 'Third Degree Felony'
        elif 'STATE JAIL' in offense_upper:
            return 'State Jail Felony'
        else:
            # Generic felony - try to classify further
            pass
    
    if 'MISDEMEANOR' in offense_upper:
        if 'CLASS A' in offense_upper:
            return 'Class A Misdemeanor'
        elif 'CLASS B' in offense_upper:
            return 'Class B Misdemeanor'
        elif 'CLASS C' in offense_upper:
            return 'Class C Misdemeanor'
        else:
            return 'Misdemeanor'
    
    # Check for DWI offenses first (specific logic)
    is_dwi = 'DWI' in offense_upper or 'DRIVING WHILE INTOXICATED' in offense_upper
    if is_dwi:
        if '3' in offense_upper or 'THIRD' in offense_upper:
            return 'Third Degree Felony'
        elif '2' in offense_upper or 'SECOND' in offense_upper:
            return 'Class A Misdemeanor'
        else:
            # First offense or unspecified
            return 'Class B Misdemeanor'
    
    # Capital Felonies
    if 'CAPITAL MURDER' in offense_upper:
        return 'Capital Felony'
    
    # First Degree Felonies
    first_degree_keywords = [
        'AGGRAVATED ROBBERY', 'MURDER', 'AGGRAVATED KIDNAPPING',
        'AGGRAVATED SEXUAL ASSAULT', 'CONTINUOUS SEXUAL ABUSE',
        'TRAFFICKING OF PERSONS'
    ]
    if any(keyword in offense_upper for keyword in first_degree_keywords):
        return 'First Degree Felony'
    
    # Second Degree Felonies
    second_degree_keywords = [
        'AGGRAVATED ASSAULT', 'ROBBERY', 'KIDNAPPING', 'ARSON',
        'BURGLARY OF HABITATION', 'INDECENCY WITH A CHILD',
        'SEXUAL ASSAULT', 'AGGRAVATED SEXUAL ASSAULT OF A CHILD'
    ]
    if any(keyword in offense_upper for keyword in second_degree_keywords):
        return 'Second Degree Felony'
    
    # Third Degree Felonies (check before state jail felonies)
    third_degree_keywords = [
        'INJURY TO A CHILD', 'INJURY TO ELDERLY', 'INJURY TO DISABLED',
        'BURGLARY OF BUILDING', 'EVADING ARREST', 'ASSAULT OF PUBLIC SERVANT',
        'FELONY ASSAULT', 'AGGRAVATED ASSAULT WITH DEADLY WEAPON'
    ]
    if any(keyword in offense_upper for keyword in third_degree_keywords):
        return 'Third Degree Felony'
    
    # State Jail Felonies
    state_jail_keywords = [
        'CREDIT CARD ABUSE', 'CREDIT CARD OR DEBIT CARD ABUSE',
        'UNAUTHORIZED USE OF VEHICLE', 'POSSESSION OF CONTROLLED SUBSTANCE',
        'POSS CS', 'POSSESSION CS', 'POSS CONTROLLED SUBSTANCE'
    ]
    if any(keyword in offense_upper for keyword in state_jail_keywords):
        if 'AGGRAVATED' in offense_upper:
            return 'Third Degree Felony'
        return 'State Jail Felony'
    
    # Theft classification (amount-dependent, but default to state jail felony)
    if 'THEFT' in offense_upper:
        if 'AGGRAVATED' in offense_upper:
            return 'First Degree Felony'
        # Most thefts in this range are state jail felonies
        return 'State Jail Felony'
    
    # Class A Misdemeanors
    class_a_keywords = [
        'ASSAULT CAUSES BODILY INJURY', 'ASSAULT BODILY INJURY',
        'ASSAULT FAMILY MEMBER', 'TERRORISTIC THREAT',
        'CRIMINAL MISCHIEF', 'RESISTING ARREST', 'ASSAULT'
    ]
    if any(keyword in offense_upper for keyword in class_a_keywords):
        # Make sure it's not a felony assault
        if 'AGGRAVATED' in offense_upper or 'FELONY' in offense_upper:
            pass  # Skip, will be caught by felony keywords
        else:
            return 'Class A Misdemeanor'
    
    # Class B Misdemeanors
    class_b_keywords = [
        'INDECENT EXPOSURE', 'CRIMINAL TRESPASS',
        'POSSESSION OF MARIJUANA', 'POSS MARIJUANA'
    ]
    if any(keyword in offense_upper for keyword in class_b_keywords):
        return 'Class B Misdemeanor'
    
    # Class C Misdemeanors
    class_c_keywords = [
        'PUBLIC INTOXICATION', 'DISORDERLY CONDUCT', 'TRAFFIC VIOLATION',
        'NOISE VIOLATION', 'TRESPASS'
    ]
    if any(keyword in offense_upper for keyword in class_c_keywords):
        return 'Class C Misdemeanor'
    
    # Texas Penal Code section numbers (common sections)
    # Section 19.03 - Capital Murder
    if '19.03' in offense_upper or '1903' in offense_upper.replace('.', ''):
        return 'Capital Felony'
    # Section 19.02 - Murder
    if '19.02' in offense_upper or '1902' in offense_upper.replace('.', ''):
        return 'First Degree Felony'
    # Section 22.02 - Aggravated Assault
    if '22.02' in offense_upper or '2202' in offense_upper.replace('.', ''):
        return 'Second Degree Felony'
    # Section 22.01 - Assault
    if '22.01' in offense_upper or '2201' in offense_upper.replace('.', ''):
        return 'Class A Misdemeanor'
    # Section 49.04 - DWI
    if '49.04' in offense_upper or '4904' in offense_upper.replace('.', ''):
        return 'Class B Misdemeanor'
    
    # Default to Unknown if no match found
    return 'Unknown'


# Read the dataset
print("Reading dataset...")
df = pd.read_csv(file_path)
print(f"Original dataset size: {len(df)} records")
print(f"Original columns: {df.columns.tolist()}")

# Extract year column
print("\nExtracting year from date columns...")
df = ensure_year_column(df)

# Check year distribution before cleaning
print("\nYear distribution before cleaning:")
year_counts = df['Year'].value_counts().sort_index()
print(year_counts)

# Count records with year 2024 or 2025
records_2024_2025 = df[df['Year'].isin([2024, 2025])].shape[0]
print(f"\nRecords with year 2024 or 2025: {records_2024_2025}")

# Clean data: Remove records with year 2024 or 2025
print("\nCleaning data...")
df_cleaned = df[~df['Year'].isin([2024, 2025])].copy()
print(f"Cleaned dataset size: {len(df_cleaned)} records")
print(f"Records removed: {len(df) - len(df_cleaned)}")

# Add degree classification column
print("\nClassifying offenses by degree...")
try:
    offense_col = find_offense_column(df_cleaned)
    print(f"Using offense column: {offense_col}")
    
    # Apply classification function
    df_cleaned['degree'] = df_cleaned[offense_col].apply(classify_offense_degree)
    
    # Show classification distribution
    print("\nDegree classification distribution:")
    degree_counts = df_cleaned['degree'].value_counts()
    print(degree_counts)
    print(f"\nTotal classified: {len(df_cleaned[df_cleaned['degree'] != 'Unknown'])}")
    print(f"Unknown/Unclassified: {len(df_cleaned[df_cleaned['degree'] == 'Unknown'])}")
    
except Exception as e:
    print(f"Warning: Could not classify offenses: {e}")
    print("Available columns:", df_cleaned.columns.tolist())
    # Try to add a placeholder column
    df_cleaned['degree'] = 'Unknown'

# Check year distribution after cleaning
print("\nYear distribution after cleaning:")
year_counts_cleaned = df_cleaned['Year'].value_counts().sort_index()
print(year_counts_cleaned)

# Save cleaned dataset
print(f"\nSaving cleaned dataset to: {output_path}")
df_cleaned.to_csv(output_path, index=False)
print("Cleaning complete!")

# Summary statistics
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Original records: {len(df):,}")
print(f"Records with year 2024 or 2025: {records_2024_2025:,}")
print(f"Cleaned records: {len(df_cleaned):,}")
print(f"Records removed: {len(df) - len(df_cleaned):,}")
if 'degree' in df_cleaned.columns:
    print(f"\nDegree classifications added:")
    for degree, count in df_cleaned['degree'].value_counts().items():
        print(f"  {degree}: {count:,}")
print(f"\nOutput file: {output_path}")

