import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

def run_predictive_model(input_path='test.csv'):
    df = pd.read_csv(input_path)

    # Feature engineering: message length, word count, message count per month
    df['message_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Robust parsing
    df = df.dropna(subset=['date'])  # Drop invalid date rows
    df['year_month'] = df['date'].dt.to_period('M')

    # Use 'from' as employee_id if missing
    if 'employee_id' not in df.columns:
        df['employee_id'] = df['from']

    # If sentiment_score is missing, generate it
    sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    if 'sentiment_score' not in df.columns:
        if 'sentiment' not in df.columns:
            df['sentiment'] = df['message'].apply(get_sentiment)
        df['sentiment_score'] = df['sentiment'].map(sentiment_map)

    # Aggregate features by employee and month
    features = df.groupby(['employee_id', 'year_month']).agg({
        'message_length': 'mean',
        'word_count': 'mean',
        'message': 'count',
        'sentiment_score': 'sum'
    }).rename(columns={'message': 'message_count', 'sentiment_score': 'monthly_sentiment_score'}).reset_index()

    # Drop rows with NaNs before modeling
    features = features.dropna(subset=['message_length', 'word_count', 'message_count', 'monthly_sentiment_score'])

    # Prepare data
    X = features[['message_length', 'word_count', 'message_count']]
    y = features['monthly_sentiment_score']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Plot predicted vs actual
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Monthly Sentiment Score")
    plt.ylabel("Predicted Score")
    plt.title("Linear Regression: Actual vs. Predicted")
    plt.savefig('visualization/linear_regression_actual_vs_predicted.png')
    plt.show()

    # --- Model Interpretation ---
    print("\nLinear Regression Model Interpretation")
    print("-" * 40)
    print("Intercept (baseline sentiment score):", model.intercept_)
    for name, coef in zip(X.columns, model.coef_):
        direction = "positive" if coef > 0 else "negative"
        print(f"Coefficient for {name}: {coef:.4f} ({direction} influence)")
    print("\nInterpretation:")
    print("The intercept represents the baseline monthly sentiment score when all features are zero.")
    print("Each coefficient represents the expected change in the predicted sentiment score for a one-unit increase in the corresponding feature, holding other features constant.")
    print("Positive coefficients indicate a positive influence on sentiment score, while negative coefficients indicate a negative influence.")
    print("The magnitude of the coefficient indicates the strength of the association between the feature and the sentiment score.")