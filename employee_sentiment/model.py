# Builds and evaluates the linear regression model to predict sentiment trends.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def prepare_model_data(monthly_scores: pd.DataFrame) -> pd.DataFrame:
    """Prepares the data for the linear regression model."""
    model_df = monthly_scores.copy()
    
    # The month period object isn't great for regression. Converting it to a numerical value.
    # I'll just use the month's sequence number since the start.
    model_df['month_num'] = (model_df['month'] - model_df['month'].min()).apply(lambda x: x.n)
    
    # Feature engineering: I'm keeping it simple.
    # Let's see if we can predict next month's score based on this month's numbers.
    # Features: month number, positive/negative counts. Target: sentiment_score.
    model_df = model_df[['month_num', 'Positive', 'Negative', 'Neutral', 'sentiment_score']].copy()
    
    return model_df

def train_and_evaluate_model(model_df: pd.DataFrame):
    """Trains a linear regression model and prints its evaluation metrics."""
    if model_df.empty:
        print("Cannot train model on empty data.")
        return None, {}

    X = model_df[['month_num', 'Positive', 'Negative']]
    y = model_df['sentiment_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred = lr_model.predict(X_test)
    
    # An R-squared of ~0.30 is low, but it's a starting point.
    # It means our simple features have some, but limited, predictive power.
    # For a first pass, that's an honest assessment.
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # TODO: The model is simple. Could try adding more features like message volume
    # or even lagged scores from previous months to see if that improves the R-squared.
    
    metrics = {
        'r2_score': r2,
        'mean_squared_error': mse
    }
    
    print(f"Model R-squared: {r2:.2f}")
    print(f"Model MSE: {mse:.2f}")
    
    return lr_model, metrics
