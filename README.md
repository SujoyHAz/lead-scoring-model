# Lead Scoring Model

Most sales teams treat every lead the same. They work through a list top 
to bottom, waste time on people who were never going to buy, and miss the 
ones who were ready to convert. This project fixes that.

## What I built

A machine learning model that takes a raw CRM export of 9,240 leads and 
scores each one by how likely they are to convert. Every lead gets a score 
from 0 to 100 and a priority label: Hot, Warm, or Cold. The sales team 
works the Hot leads first.

The model ended up at 82% accuracy, tested on 1,848 leads it had never 
seen before.

## The data

Real dataset from X Education, an online course provider. 9,240 leads, 
37 columns, and a confirmed outcome for each lead (converted or didn't). 
That last part is what made it useful: I could actually check whether the 
model was right.

Source: [Kaggle - Lead Scoring Dataset](https://www.kaggle.com/datasets/ashydv/leads-dataset)

## How it works

1. Load and clean the raw data (missing values, useless columns, text 
   converted to numbers)
2. Train a Logistic Regression model on 80% of the leads
3. Test it on the remaining 20% it hasn't seen
4. Score all 9,240 leads by predicted conversion probability
5. Output a ranked CSV and charts

## Results

| Metric | Value |
|---|---|
| Model accuracy | 82.0% |
| Total leads scored | 9,240 |
| Hot leads | 2,288 |
| Warm leads | 1,395 |
| Cold leads | 5,557 |

## What the model looks at

- Time spent on the website
- Number of visits
- Page views per visit
- Where the lead came from (origin and source)
- Most recent activity
- Lead quality rating

## How to run it yourself

```bash
pip install pandas scikit-learn matplotlib seaborn
python lead_scorer.py
```

Results will appear in the `output/` folder: a scored CSV and three charts.

## Tools used

Python, pandas, scikit-learn, matplotlib, seaborn

