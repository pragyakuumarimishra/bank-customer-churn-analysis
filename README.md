## Bank Customer Churn Analysis

### Problem Statement
Banks face customer attrition which impacts revenue. This project analyzes customer
demographics and account behavior to identify key factors driving churn.

### Dataset
A synthetic dataset modeled on publicly available bank churn datasets
(e.g., Kaggle Churn Modelling) to ensure privacy and reproducibility.

### Tools Used
- Python (pandas, numpy, matplotlib)
- Excel (for quick validation)
- SQL (conceptual analysis)

### Key Insights
- Inactive customers show significantly higher churn
- Customers aged 40–60 churn more frequently
- Customers with 3 or more products have higher churn
- Mid-balance customers are more stable

### Business Recommendations
- Re-engagement campaigns for inactive users
- Simplify product bundles
- Targeted loyalty programs for mid-aged customers

### How to Run
```bash
pip install -r requirements.txt
python churn_analysis.py
