import pandas as pd
import matplotlib.pyplot as plt
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

def run_employee_ranking(input_path='test.csv'):
    df = pd.read_csv(input_path)

    # Robust date parsing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Use 'from' as employee_id if 'employee_id' missing
    if 'employee_id' not in df.columns:
        df['employee_id'] = df['from']

    # Generate sentiment if missing
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['message'].apply(get_sentiment)

    # Generate sentiment_score if missing
    sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)

    df['year_month'] = df['date'].dt.to_period('M')

    # Calculate monthly sentiment scores
    monthly_scores = df.groupby(['employee_id', 'year_month'])['sentiment_score'].sum().reset_index()
    monthly_scores.rename(columns={'sentiment_score': 'monthly_sentiment_score'}, inplace=True)

    def get_monthly_rankings(monthly_scores):
        rankings = []
        months = monthly_scores['year_month'].unique()
        for month in months:
            df_month = monthly_scores[monthly_scores['year_month'] == month]
            top_pos = df_month.sort_values(
                by=['monthly_sentiment_score', 'employee_id'],
                ascending=[False, True]
            ).head(3)
            top_neg = df_month.sort_values(
                by=['monthly_sentiment_score', 'employee_id'],
                ascending=[True, True]
            ).head(3)
            rankings.append((month, top_pos, top_neg))
        return rankings

    rankings = get_monthly_rankings(monthly_scores)

    # Save a single visualization for the latest month for submission
    if len(rankings) > 0:
        month, top_pos, top_neg = rankings[-1]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        top_pos.plot(
            kind='bar', x='employee_id', y='monthly_sentiment_score',
            ax=axes[0], color='green', legend=False
        )
        axes[0].set_title(f"Top 3 Positive: {month}")
        axes[0].set_ylabel("Monthly Sentiment Score")
        axes[0].set_xlabel("Employee ID")

        top_neg.plot(
            kind='bar', x='employee_id', y='monthly_sentiment_score',
            ax=axes[1], color='red', legend=False
        )
        axes[1].set_title(f"Top 3 Negative: {month}")
        axes[1].set_ylabel("Monthly Sentiment Score")
        axes[1].set_xlabel("Employee ID")

        plt.tight_layout()
        plt.savefig('visualization/employee_ranking.png')
        plt.show()

    # Print all rankings to console
    for month, top_pos, top_neg in rankings:
        print(f"\nMonth: {month}")
        print("Top 3 Positive Employees:")
        print(top_pos[['employee_id', 'monthly_sentiment_score']].to_string(index=False))
        print("Top 3 Negative Employees:")
        print(top_neg[['employee_id', 'monthly_sentiment_score']].to_string(index=False))