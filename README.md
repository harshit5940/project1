# Customer Churn Analysis (Python + Streamlit)

An end-to-end churn analysis project using the IBM Telco Customer Churn dataset. It includes data cleaning, EDA, feature engineering, modeling (Logistic Regression and Random Forest), and an interactive Streamlit dashboard for business insights.

## Quickstart

1. Create a Python 3.10+ environment.
2. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit app:
   
   ```bash
   streamlit run app/app.py
   ```

4. Open the Jupyter notebook for EDA & modeling:
   
   - `notebooks/churn_analysis.ipynb`

## Data

- Default: IBM Telco Customer Churn dataset
- The app/notebook will auto-download the CSV if not found into `data/`.

## Project Structure

- `app/app.py` — Streamlit dashboard
- `src/churn/utils.py` — Data loading, cleaning, feature engineering, modeling
- `notebooks/churn_analysis.ipynb` — EDA and model development
- `data/` — Dataset directory (auto-populated)
- `executive_summary.md` — Business insights and recommendations template

## KPIs in Dashboard

- Churn Rate (%), Retention Rate (%)
- Average Tenure (months)
- High-Risk Customer Share (%)
- CLV estimate (simple heuristic)

## Notes

- No API keys required.
- You can replace the dataset by placing your CSV in `data/` and updating the path in the app or notebook.
