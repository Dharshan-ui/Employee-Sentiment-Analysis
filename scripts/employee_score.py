import pandas as pd
from textblob import TextBlob

def get_sentiment(text):
    if pd.isnull(text):
        return "Neutral"
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def run_employee_score(filepath='test.csv'):
    """
    Computes monthly sentiment scores for each employee.
    Positive Message: +1
    Negative Message: -1
    Neutral Message: 0

    Returns a DataFrame with employee_id, year_month, and monthly_score.
    """
    # Load data
    df = pd.read_csv(filepath)
    # Robust date parsing: coerce errors to NaT, then drop
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Use 'from' as employee_id if 'employee_id' column not present
    if 'employee_id' not in df.columns:
        df['employee_id'] = df['from']
    # Generate sentiment if missing
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['message'].apply(get_sentiment)

    # Group messages by month
    df['year_month'] = df['date'].dt.to_period('M')
    # Assign score to each message based on sentiment
    sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    df['score'] = df['sentiment'].map(sentiment_map)

    # Aggregate scores monthly for each employee
    monthly_scores = (
        df.groupby(['employee_id', 'year_month'])['score']
        .sum()
        .reset_index()
        .rename(columns={'score': 'monthly_score'})
    )

    # Sort results for readability
    monthly_scores = monthly_scores.sort_values(['employee_id', 'year_month'])

    return monthly_scores