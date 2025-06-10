import pandas as pd
from textblob import TextBlob

def run_sentiment_labeling(input_path='test.csv'):
    # Load the dataset
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"File not found: {input_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Check if 'message' column exists
    if 'message' not in df.columns:
        print("Error: 'message' column not found in the CSV.")
        return

    # Sentiment labeling function
    def get_sentiment_label(text):
        if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
            return "Neutral"
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"

    # Apply the sentiment function
    df['sentiment'] = df['message'].apply(get_sentiment_label)

    # Map sentiment to score
    score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['sentiment_score'] = df['sentiment'].map(score_map)

    # Print first few labeled rows
    print("Labeled Data Preview:")
    print(df.head())

