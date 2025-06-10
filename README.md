# Employee Sentiment Analysis

## Project Overview

This project analyzes an unlabeled dataset of employee emails/messages (`test.csv`) to assess sentiment, engagement, and potential flight risk within a company using Natural Language Processing (NLP) and statistical techniques. The project is structured into distinct tasks, each building towards actionable insights and predictive modeling for HR and management.

**Note:**  
This submission is for internal evaluation only and should not be shared publicly or with others.

---

## Project Objective

The main objectives are to:
- **Label** each message as Positive, Negative, or Neutral.
- **Analyze** and visualize the data to understand sentiment and engagement trends.
- **Score employees** monthly based on their message sentiment.
- **Rank** employees by sentiment score.
- **Flag potential flight risks** (employees sending 4+ negative messages in any 30-day window).
- **Build a linear regression model** to predict sentiment scores based on message characteristics.

---

## Project Structure

```
AI-project-submission/
├── main.ipynb                          # Main notebook with code, analysis, and commentary
├── scripts/                            # (Optional) Supporting .py scripts if any
│   └── ...
├── visualization/                      # All EDA and model visualizations
│   └── ...
├── README.md                           # This file
├── Final_Report.docx                   # Detailed report
├── test.csv                            # Input dataset (if included)
└── .env.example                        # (If applicable)
```

---

## Setup & Usage

### Prerequisites

- Python 3.7+
- Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
- Place `test.csv` in the project root directory.

### Running the Analysis

1. Open `Employee_Sentiment_Analysis.ipynb` in Jupyter Notebook/Lab.
2. Run all cells in sequence:
    - Sentiment labeling 
    - EDA & visualizations
    - Monthly sentiment scoring & ranking
    - Flight risk identification
    - Predictive modeling
3. Outputs (tables, plots) will be saved in the `visualization/` folder.

---

## Methodology Summary

### Task 1: Sentiment Labeling

- Each message is labeled as Positive, Negative, or Neutral using the TextBlob sentiment polarity.
- Thresholds: Polarity > 0.1 = Positive, < -0.1 = Negative, else Neutral.

### Task 2: Exploratory Data Analysis (EDA)

- Analyzed message counts, sentiment distribution, employee activity, and time trends.
- Visualizations include sentiment pie/bar charts, activity timelines, and more.

### Task 3: Employee Score Calculation

- Each message scored: +1 (Positive), -1 (Negative), 0 (Neutral).
- Monthly sentiment scores aggregated for each employee.

### Task 4: Employee Ranking

- For each month, employees ranked by sentiment score.
- **Top 3 Positive** and **Top 3 Negative** employees identified monthly.

### Task 5: Flight Risk Identification

- Employees flagged if they sent 4+ negative messages in any rolling 30-day window.

### Task 6: Predictive Modeling

- Linear regression model predicts monthly sentiment scores using:
    - Message length
    - Word count
    - Message count
- Model evaluated using MSE and R² score.

---

## Key Results Summary

- **Top 3 Positive Employees (Overall):**
    - don.baughman@enron.com
    - sally.beck@enron.com
    - john.arnold@enron.com

- **Top 3 Negative Employees (Overall):**
    - bobette.riner@enron.com
    - johnny.palmer@enron.com
    - rhonda.denton@enron.com

- **Flagged Flight Risk Employees:**
    - bobette.riner@enron.com
    - johnny.palmer@enron.com
    - lydia.delgado@enron.com
    - sally.beck@enron.com

- **Predictive Model Performance:**
    - R² ≈ 0.50 (moderate predictive power)
    - Message count and word count were the strongest positive predictors of sentiment score.

---

## Recommendations

- Regularly monitor employee sentiment and engagement.
- Reach out to employees with repeated negative sentiment.
- Use predictive modeling as an early warning system for HR interventions.

---

## Reproducibility & Documentation

- The code is fully commented for clarity.
- Each section in the notebook is titled and includes observations.
- The entire process is reproducible from raw data to final outputs.

---

## Contact

**Author:** Dharshan R  
**Submission for:** AI-project-submission (internal evaluation)  
**Dataset:** test.csv

---
