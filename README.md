# Retail Sales & Demand Forecasting

## Future Interns – Machine Learning Internship
**Task 1: Retail Sales & Demand Forecasting**

### 1. Project Overview
This project analyzes historical retail sales data and uses Machine Learning algorithms to predict monthly sales.

The main objective is to understand sales trends, identify patterns and build a forecasting model that can support better business decisions.

### 2. Dataset
- Dataset: Superstore Sales Dataset
- Total records: 9,994
- Total columns: 21
- Data period: January 2014 – December 2017
- Missing values: 0
- Duplicate records: 0

### 3. Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- VS Code

### 4. Project Workflow

1. Loaded and explored the retail sales dataset.
2. Checked missing values and duplicate records.
3. Converted order dates into datetime format.
4. Extracted Year and Month features.
5. Aggregated daily transaction records into monthly sales.
6. Visualized monthly sales trends.
7. Prepared training and testing datasets.
8. Trained Machine Learning models.
9. Evaluated model performance.
10. Compared actual and predicted sales.

### 5. Machine Learning Models

Three models were tested:

- Linear Regression
- Seasonal Linear Regression
- Random Forest Regressor

### 6. Model Evaluation Results

| Metric | Linear Regression | Seasonal Linear Regression | Random Forest |
|---|---:|---:|---:|
| MAE | 11,599.75 | 18,100.53 | 14,702.23 |
| RMSE | 17,017.19 | 23,284.25 | 17,982.80 |
| R² Score | 0.4900 | 0.0451 | 0.4304 |

**Best-performing model:** Linear Regression

The original Linear Regression model achieved the lowest MAE and RMSE and the highest R² score among the three tested models.

### 7. Visualizations

The project includes:

- Monthly Sales Trend – Shows changes in monthly sales from 2014 to 2017.
- Actual vs Predicted Sales – Compares actual monthly sales with model predictions.

Both graphs are saved in the `visualizations` folder.

### 8. Key Business Insights

- Monthly sales fluctuate considerably.
- Sales generally increased towards 2017.
- November 2017 recorded the highest monthly sales of approximately 118,447.
- February 2014 recorded the lowest monthly sales of approximately 4,520.
- The original Linear Regression model performed better than the other tested models.

### 9. Project Structure

```text
Future_Interns_ML_Task1/
│
├── dataset/
│   ├── superstore.csv
│   └── monthly_sales.csv
│
├── visualizations/
│   ├── monthly_sales_trend.png
│   └── actual_vs_predicted.png
│
├── notebooks/
├── venv/
├── main.py
└── README.md
```

### 10. How to Run the Project

**Step 1:** Clone or download the project.

**Step 2:** Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Step 3:** Activate the virtual environment (if using one).

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

**Step 4:** Run the Python program.

```bash
python main.py
```

### 11. Conclusion

This project demonstrates how Python and Machine Learning can be used to analyze retail sales data and predict monthly sales.

By comparing different algorithms and evaluating their performance, the project identifies Linear Regression as the best-performing model among the three tested approaches.

The analysis provides useful insights into historical sales trends and demonstrates the practical application of Machine Learning in retail demand forecasting.

---

**Internship:** Future Interns  
**Project:** Machine Learning – Task 1  
**Project Title:** Retail Sales & Demand Forecasting