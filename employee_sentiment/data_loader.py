# Handles loading and cleaning the raw email data from the CSV file.
import pandas as pd
from typing import Optional

def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    """Loads and cleans the Enron email dataset from a CSV file."""
    try:
        # This dataset had some weird encoding issues, 'cp1252' was the one that worked.
        raw_df = pd.read_csv(file_path, encoding='cp1252')
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")
        return None

    # We only need who sent it, when, and what it said.
    df = raw_df[['from', 'date', 'message']].copy()
    df.rename(columns={'from': 'employee', 'date': 'date', 'message': 'text'}, inplace=True)

    # Can't analyze sentiment without an author or any text.
    df.dropna(subset=['employee', 'text'], inplace=True)

    # Dates are messy. Coercing errors will just turn un-parseable dates into nulls.
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # TODO: The current date parsing drops rows if the format is weird.
    # A better approach might be to try a few different common date formats before giving up.

    df['date'] = df['date'].dt.date

    return df
