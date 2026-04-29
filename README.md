# Employee Sentiment Analysis from Email Data

This project is a deep dive into the Enron email dataset, with the goal of analyzing employee sentiment to understand the overall mood of the company and identify potential flight risks. I built this as a personal project to practice data cleaning, sentiment analysis, and basic predictive modeling.

## The Goal

I wanted to see if I could answer a few key questions from the data:
1. What is the overall sentiment distribution across all messages?
2. How does the company-wide sentiment trend over time?
3. Can we identify employees who are significantly more negative or positive than their peers?
4. Is it possible to flag employees who might be a "flight risk" based on their sentiment?
5. Can we build a simple model to predict future sentiment?

## Project Structure

I structured the project into a Python package (`employee_sentiment`) to keep the code clean, modular, and reusable. Each file has a single responsibility.

- `main.ipynb`: This is the main narrative of the project. It's a Jupyter Notebook that imports the modules and walks through the analysis step-by-step, explaining the findings along the way.
- `requirements.txt`: A list of all the Python packages needed to run the project.
- `employee_sentiment/`: This is the core Python package.
  - `__init__.py`: Makes the directory a Python package.
  - `data_loader.py`: Handles the messy work of loading the raw CSV data. I had to deal with tricky file encoding issues (`cp1252`, not the usual `utf-8`) and parse a variety of inconsistent date formats.
  - `sentiment_engine.py`: Where the sentiment analysis happens. I ended up using TextBlob after trying VADER first — VADER felt too aggressive and was flagging a lot of neutral, professional language as negative. TextBlob gave more balanced results for this dataset.
  - `scoring.py`: Calculates monthly sentiment scores for each employee and contains the logic for identifying flight risks (defined as 4+ negative emails in a rolling 30-day window).
  - `model.py`: A simple linear regression model to see if there was a predictive signal in the data. The R-squared comes out around 0.30, which is honest — predicting human sentiment from surface-level message features is genuinely hard. It shows a weak but real signal, and a richer feature set could improve it.
  - `visualizations.py`: All the functions for generating the plots used in the notebook.

## Results Summary

- **Sentiment Distribution:** The majority of messages are Neutral, which makes sense for professional email. Positive messages outnumber Negative ones significantly.
- **Flight Risks Identified:** 4 employees were flagged for sending 4 or more negative messages within a rolling 30-day window.
- **Predictive Model:** Linear Regression achieved R² ≈ 0.30 — not a strong predictor, but it confirms a measurable signal exists between message characteristics and sentiment scores.

## How to Run This Project

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook:**
   Launch Jupyter and open `main.ipynb` to walk through the full analysis.

## What I'd Improve Next

No project is ever truly finished. Here are a few things I'd tackle next if I had more time:

- **Improve the Model:** The linear regression model is a starting point. I'd experiment with Gradient Boosting or an LSTM and engineer richer features — message volume, sentiment trend over the last N months, time since last negative message — to push the R-squared higher.
- **Refine Date Parsing:** Right now, any date that can't be parsed gets dropped. A more robust approach would try multiple common date formats before giving up on a row.
- **Better Visualization for Employee Rankings:** Email addresses can be long and clutter the y-axis on the ranking chart. I'd anonymize or shorten them for a cleaner plot.
- **More Nuanced Flight Risk Detection:** The "4 negatives in 30 days" rule is a simple heuristic. A better approach might look for sharp drops in an employee's rolling average sentiment, or flag a sustained period of below-average scores rather than just a raw count.