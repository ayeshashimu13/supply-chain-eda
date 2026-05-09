# Predicting Late Deliveries in a Global Supply Chain

## Overview
Analyzed 180,000+ orders from a global supply chain company to uncover why 55% of deliveries arrive late, and built a machine learning model to predict late deliveries before they happen.

## Business Questions Answered
- Why are more than half of all orders delivered late?
- Which shipping modes are most problematic?
- Is the issue location-specific or company-wide?
- Can we predict which orders will be late before they ship?

## Key Findings

### The Root Cause: Unrealistic Delivery Promises
The company is not shipping slowly — it is **overpromising delivery times**:

| Shipping Mode | Promised Days | Actual Days | Late Rate |
|--------------|--------------|-------------|-----------|
| First Class | 1 day | 2 days | 95% |
| Second Class | 2 days | 4 days | 77% |
| Standard Class | 4 days | 4 days | 38% |
| Same Day | 0 days | 0.5 days | 46% |

Standard Class is the only mode where the promise matches reality — and it has the lowest late rate.

### It's a System-Wide Problem
- Late delivery rates are nearly identical across all 23 regions (49%-58%)
- All 50 product categories show similar late rates (52%-58%)
- Customer segment (Consumer, Corporate, Home Office) has no impact
- This is not a regional or product issue — it's a **policy issue**

### Other Insights
- Most late deliveries are only **1 day behind schedule** — adding 1 day to promises could fix thousands of "late" deliveries
- **Transfer payments** have a 16% cancellation rate vs 0% for other types — suggesting a payment processing issue
- Orders flagged as **SUSPECTED_FRAUD** are 100% canceled (never shipped) — fraud detection is working

## Predictive Model

### Model Evolution
| Version | Model | Features | F1 Score |
|---------|-------|----------|----------|
| V1 | Random Forest | 6 basic features | 0.68 |
| V2 | Random Forest | + Shipping Mode encoded | 0.70 |
| V3 | XGBoost | + Time features | 0.71 |
| V4 | XGBoost | + Threshold optimization | 0.75 |
| V5 (Final) | XGBoost | + Target-encoded location/category | **0.80** |

### Final Model Performance
- **F1 Score: 0.80**
- **Recall: 86%** — catches 86 out of 100 late deliveries
- **Precision: 75%** — when it predicts late, it's correct 75% of the time
- **Accuracy: 75.7%**

### Data Leakage Awareness
Including "Days for shipping (real)" gave 100% accuracy — but this is data leakage since actual shipping time is only known after delivery. The final model uses only features available **before** an order ships.

## Recommendations
1. **Fix delivery promises** — adjust First Class from 1→2 days and Second Class from 2→4 days. This alone could drop the late rate from 55% to under 20%
2. **Audit First Class shipping** — 95% late rate is unacceptable for a premium service
3. **Investigate transfer payment cancellations** — 16% cancellation rate needs attention
4. **Deploy the prediction model** — flag high-risk orders before shipping for proactive intervention
5. **Track delivery partners** — data on which couriers handle shipments would enable deeper root cause analysis

## Tech Stack
- **Python** — pandas, NumPy, matplotlib, seaborn
- **Machine Learning** — scikit-learn (Random Forest, Decision Tree, GridSearchCV), XGBoost
- **Techniques** — feature engineering, one-hot encoding, target encoding, threshold optimization, cross-validation, precision-recall analysis

## Dataset
DataCo Global Supply Chain Dataset (180,519 orders, 53 features) from Kaggle.

## How to Run
1. Download the dataset from Kaggle
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`
3. Open `Supply_Chain_Late_Delivery_Analysis.ipynb` in Jupyter Notebook
