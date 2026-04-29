# Generates all the charts and plots for the analysis.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df: pd.DataFrame):
    """Plots a bar chart of the overall sentiment distribution."""
    plt.figure(figsize=(8, 6))
    sentiment_counts = df['sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title('Overall Sentiment: More Neutral Than Anything Else')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Messages')
    plt.show()

def plot_monthly_sentiment_trend(monthly_scores: pd.DataFrame):
    """Plots a line chart of the average sentiment score over time."""
    if monthly_scores.empty:
        return
    
    # Need to convert the period object to a timestamp for plotting.
    monthly_avg = monthly_scores.groupby('month')['sentiment_score'].mean().reset_index()
    monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='month', y='sentiment_score', data=monthly_avg, marker='o')
    plt.title('Company-Wide Sentiment Trend: A Few Bumps, But Generally Stable')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.show()

def plot_employee_ranking(emp_ranks: pd.DataFrame):
    """Plots a bar chart of the top 5 and bottom 5 employees by sentiment."""
    if emp_ranks.empty:
        return

    # Grabbing the top and bottom 5 for a focused view.
    top_5 = emp_ranks.head(5)
    bottom_5 = emp_ranks.tail(5)
    top_and_bottom = pd.concat([top_5, bottom_5])

    plt.figure(figsize=(12, 8))
    sns.barplot(x='sentiment_score', y='employee', data=top_and_bottom, palette="coolwarm_r")
    plt.title('Sentiment Extremes: Who Are the Happiest and Unhappiest Employees?')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Employee')
    # TODO: The employee emails can be long and clutter the y-axis. 
    # Should probably anonymize or shorten them for a cleaner plot.
    plt.show()

def plot_flight_risk_heatmap(flight_risks: pd.DataFrame):
    """Plots a heatmap of employees flagged as flight risks over time."""
    if flight_risks.empty:
        print("No flight risks identified, so no heatmap to show.")
        return
        
    # Pivot the data to get months on one axis and employees on the other.
    flight_risks['month'] = pd.to_datetime(flight_risks['date']).dt.to_period('M')
    heatmap_data = flight_risks.groupby(['employee', 'month']).size().unstack(fill_value=0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, cmap="Reds", linewidths=.5)
    plt.title('Flight Risk Hotspots: When and Where is Negative Sentiment Concentrated?')
    plt.xlabel('Month')
    plt.ylabel('Employee')
    plt.show()
