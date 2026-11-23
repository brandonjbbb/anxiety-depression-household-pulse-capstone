# Anxiety and Depression Symptoms in U.S. Adults - Predictive Models
Capstone project, Northwest Missouri State University.
Data: CDC/NCHS Household Pulse Survey indicators (data.cdc.gov 8pt5-q6wp), cleaned to 16,087 rows.
- predictive_models.py: trains Mean baseline, Linear Regression, Random Forest, Gradient Boosting (scikit-learn) and ARIMA / exponential smoothing (statsmodels); writes results.csv, ts.csv, impA.csv, impB.csv
- make_figures.py: builds the pipeline diagram and results figure
Run: python predictive_models.py (expects the cleaned CSV, update the path at the top)
