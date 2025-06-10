import pandas as pd
import matplotlib.pyplot as plt

def run_flight_risk(input_path='test.csv'):
    # Load the labeled dataset
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Robust date parsing
    df = df.dropna(subset=['date'])  # Drop rows with bad dates

    # Use 'from' as employee_id if 'employee_id' column not present
    if 'employee_id' not in df.columns:
        df['employee_id'] = df['from']

    # Use 'sentiment' column if present, else generate it (optional)
    if 'sentiment' not in df.columns:
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
        df['sentiment'] = df['message'].apply(get_sentiment)

    df_sorted = df.sort_values(['employee_id', 'date']).copy()

    # Mark negative messages as 1, else 0
    df_sorted['is_negative'] = (df_sorted['sentiment'] == 'Negative').astype(int)

    # Set index for rolling operation
    df_sorted.set_index('date', inplace=True)

    # Calculate rolling sum using groupby and transform (returns a Series aligned with original DataFrame)
    df_sorted['neg_count_30d'] = (
        df_sorted.groupby('employee_id')['is_negative']
        .transform(lambda s: s.rolling('30D').sum())
    )

    # Reset index if you want 'date' back as a column
    df_sorted = df_sorted.reset_index()

    # Find employees with any 30-day window with 4+ negative messages
    flight_risk_flags = df_sorted[df_sorted['neg_count_30d'] >= 4]
    flight_risk_emps = flight_risk_flags['employee_id'].unique().tolist()
    print("Flight risk employees:", flight_risk_emps)

    # Visualization: bar plot of number of times each employee is flagged as flight risk
    flag_counts = flight_risk_flags['employee_id'].value_counts()
    if not flag_counts.empty:
        plt.figure(figsize=(7,4))
        flag_counts.plot(kind='bar', color='orange')
        plt.title('Employees Flagged as Flight Risk (Count of 30-day Windows)')
        plt.xlabel('Employee ID')
        plt.ylabel('Times Flagged')
        plt.tight_layout()
        plt.savefig('visualization/flight_risk_flagged.png')
        plt.show()
    else:
        print("No employees flagged as flight risk. No plot saved.")

    return flight_risk_emps