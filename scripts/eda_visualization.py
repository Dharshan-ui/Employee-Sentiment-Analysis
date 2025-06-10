import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def run_eda_visualization(filepath='test.csv'):
    df = pd.read_csv(filepath)
    # Robust date conversion with error handling
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  
    print("Number of bad dates:", df['date'].isna().sum())
    df = df.dropna(subset=['date'])

    # 3. Basic statistics
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.info())
    print(df.head())

    # 4. Missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # 5. Sentiment distribution 
    df['sentiment'] = df['message'].apply(get_sentiment)
    print("\nSentiment value counts:")
    print(df['sentiment'].value_counts())
    plt.figure(figsize=(6,4))
    sns.countplot(x='sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
    plt.title('Sentiment Label Distribution')
    plt.tight_layout()
    plt.show(block=True)   

    # 6. Message count per sender
    print("\nTop 10 senders by message count:")
    print(df['from'].value_counts().head(10))
    top_senders = df['from'].value_counts().index[:10]
    plt.figure(figsize=(8,5))
    sns.countplot(x='from', data=df[df['from'].isin(top_senders)], order=top_senders)
    plt.title('Message Count per Top 10 Senders')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=True)

    # 7. Sentiment over time (now works)
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_sentiment = df.groupby(['year_month', 'sentiment']).size().unstack().fillna(0)
    print("\nMonthly sentiment table:")
    print(monthly_sentiment.tail())
    monthly_sentiment.plot(kind='bar', stacked=True, figsize=(10,6))
    plt.title('Monthly Sentiment Counts')
    plt.tight_layout()
    plt.show()

    # 8. Message length distribution by sender
    df['message_length'] = df['message'].str.len()
    plt.figure(figsize=(8,5))
    sns.boxplot(x='from', y='message_length', data=df[df['from'].isin(top_senders)], order=top_senders)
    plt.title('Message Length by Top 10 Sender')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 9. Distribution of message lengths
    plt.figure(figsize=(8,4))
    sns.histplot(df['message_length'], bins=10, kde=True)
    plt.title('Distribution of Message Lengths')
    plt.tight_layout()
    plt.show()

    # 10. Outlier detection
    long_threshold = df['message_length'].quantile(0.95)
    short_threshold = df['message_length'].quantile(0.05)
    long_messages = df[df['message_length'] > long_threshold]
    short_messages = df[df['message_length'] < short_threshold]
    print("\nUnusually long messages (above 95th percentile):")
    print(long_messages[['from', 'date', 'message', 'message_length']])
    print("\nUnusually short messages (below 5th percentile):")
    print(short_messages[['from', 'date', 'message', 'message_length']])

    # 11. Sender message count breakdown
    sender_counts = df['from'].value_counts()
    print("\nSender-level message counts:")
    print(sender_counts)

    sender_counts.plot(kind='bar', figsize=(8,5))
    plt.title('Message Count per Sender')
    plt.tight_layout()
    plt.show()

    # 12. Temporal gaps in messaging
    df_sorted = df.sort_values(['from', 'date'])
    df_sorted['prev_date'] = df_sorted.groupby('from')['date'].shift(1)
    df_sorted['days_since_last'] = (df_sorted['date'] - df_sorted['prev_date']).dt.days
    print("\nLargest gaps between messages by sender (days):")
    print(df_sorted.groupby('from')['days_since_last'].max())

    # 13. Senders with only one message
    msg_counts = df['from'].value_counts()
    single_msgs = msg_counts[msg_counts == 1]
    print("\nSenders with only one message:", single_msgs.index.tolist())

    # 14. Senders with only one subject (bonus: replace sentiment logic)
    subject_counts = df.groupby('from')['subject'].nunique()
    only_one_subject = subject_counts[subject_counts == 1].index.tolist()
    print("\nSenders with only one subject:", only_one_subject)