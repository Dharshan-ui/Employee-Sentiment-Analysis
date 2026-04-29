# Performs sentiment analysis on the email text using TextBlob.
import pandas as pd
from textblob import TextBlob

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes the sentiment of each email and adds polarity and sentiment label."""
    
    # I tried VADER first, but it was flagging a lot of normal office-speak as negative.
    # TextBlob seems a bit more conservative and fits this dataset better.
    df['polarity'] = df['text'].apply(lambda text: TextBlob(text).sentiment.polarity)

    def assign_sentiment_label(polarity: float) -> str:
        """Assigns a sentiment label based on the polarity score."""
        # The 0.1 threshold felt right after spot-checking ~30 messages manually.
        # It's a bit of an art, but it catches clear positive/negative tones without overreacting.
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df['polarity'].apply(assign_sentiment_label)
    return df
