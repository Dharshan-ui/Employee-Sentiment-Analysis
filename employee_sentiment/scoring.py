# Calculates monthly sentiment scores and identifies employees who are flight risks.
import pandas as pd

def calculate_monthly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates sentiment scores by employee and month."""
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    # Group by employee and month, then count the sentiment labels.
    monthly_sentiment = df.groupby(['employee', 'month', 'sentiment']).size().unstack(fill_value=0)
    
    # Make sure all sentiment columns exist, even if none were found for a given month.
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment not in monthly_sentiment.columns:
            monthly_sentiment[sentiment] = 0
            
    monthly_sentiment['total_messages'] = monthly_sentiment.sum(axis=1)
    
    # Avoid division by zero for employees with no messages in a month.
    monthly_sentiment['sentiment_score'] = (monthly_sentiment['Positive'] - monthly_sentiment['Negative']) / monthly_sentiment['total_messages']
    
    return monthly_sentiment.reset_index()

def identify_flight_risks(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies employees with 4+ negative messages in a rolling 30-day window."""
    neg_df = df[df['sentiment'] == 'Negative'].copy()
    neg_df['date'] = pd.to_datetime(neg_df['date'])
    
    # This is the key fix: sort by employee and then by date.
    # This ensures the dates are monotonic *within each employee's group*.
    neg_df.sort_values(by=['employee', 'date'], inplace=True)
    
    neg_df.set_index('date', inplace=True)
    
    # This was tricky. I need to count per employee, so I group by employee first.
    # Then apply the rolling count.
    risk_flags = neg_df.groupby('employee').rolling('30D').count()['sentiment'].rename('neg_count_30d')
    
    # A score of 4 or more is the trigger.
    at_risk_employees = risk_flags[risk_flags >= 4]
    
    return at_risk_employees.reset_index()

def rank_employees(monthly_scores: pd.DataFrame) -> pd.DataFrame:
    """Ranks employees based on their average sentiment score."""
    emp_scores = monthly_scores.groupby('employee')['sentiment_score'].mean().reset_index()
    
    # Simple average score is the main ranking metric.
    # For ties, the person with more messages gets the higher rank. It's a decent tie-breaker.
    emp_scores['total_messages'] = monthly_scores.groupby('employee')['total_messages'].sum().values
    emp_scores.sort_values(by=['sentiment_score', 'total_messages'], ascending=[False, False], inplace=True)
    
    # TODO: The tie-breaker is good, but could be better. Maybe factor in the trend? 
    # An employee with a rising score is better than one with a falling score, even if their average is the same.
    
    emp_scores['rank'] = emp_scores['sentiment_score'].rank(method='first', ascending=False)
    
    return emp_scores
