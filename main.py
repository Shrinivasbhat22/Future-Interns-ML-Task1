
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression

# Create visualizations folder if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

# --------------------------------
# STEP 1: LOAD DATASET
# --------------------------------

print("\n--- LOADING DATASET ---")

df = pd.read_csv("dataset/superstore.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------
# STEP 2: DATASET INFORMATION
# --------------------------------

print("\n--- DATASET INFORMATION ---")

df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())


# --------------------------------
# STEP 3: DATE CONVERSION
# --------------------------------

print("\n--- DATE CONVERSION ---")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%m/%d/%Y"
)

print("Order dates converted successfully!")

print("First order date:", df["Order Date"].min())
print("Last order date:", df["Order Date"].max())


# --------------------------------
# STEP 4: TIME-BASED FEATURES
# --------------------------------

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

print("\nTime-based features created!")


# --------------------------------
# STEP 5: MONTHLY SALES
# --------------------------------

print("\n--- MONTHLY SALES ---")

monthly_sales = (
    df.groupby(
        pd.Grouper(key="Order Date", freq="MS")
    )["Sales"]
    .sum()
    .reset_index()
)

print(monthly_sales)

# Save monthly sales data
monthly_sales.to_csv(
    "dataset/monthly_sales.csv",
    index=False
)

print("Monthly sales saved successfully!")


# --------------------------------
# STEP 6: VISUALIZE MONTHLY SALES
# --------------------------------

print("\n--- CREATING SALES GRAPH ---")

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()

plt.savefig("visualizations/monthly_sales_trend.png")

plt.show()

print("Sales graph saved successfully!")


# --------------------------------
# STEP 7: PREPARE DATA FOR ML
# --------------------------------

print("\n--- PREPARING MACHINE LEARNING DATA ---")

monthly_sales["Year"] = monthly_sales["Order Date"].dt.year
monthly_sales["Month"] = monthly_sales["Order Date"].dt.month

X = monthly_sales[["Year", "Month"]]
y = monthly_sales["Sales"]


# --------------------------------
# STEP 8: TRAIN-TEST SPLIT
# --------------------------------

split_index = int(len(monthly_sales) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------
# STEP 9: TRAIN LINEAR REGRESSION
# --------------------------------

print("\n--- MODEL TRAINING ---")

model = LinearRegression()

model.fit(X_train, y_train)

print("Linear Regression model trained successfully!")


# --------------------------------
# STEP 10: SALES PREDICTION
# --------------------------------

print("\n--- SALES PREDICTIONS ---")

y_pred = model.predict(X_test)

for actual, predicted in zip(y_test, y_pred):
    print(
        "Actual Sales:", round(actual, 2),
        "| Predicted Sales:", round(predicted, 2)
    )


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# --------------------------------
# STEP 11: IMPROVED MODEL
# --------------------------------

print("\n--- IMPROVED MODEL TRAINING ---")

monthly_sales["Month_Sin"] = np.sin(
    2 * np.pi * monthly_sales["Month"] / 12
)

monthly_sales["Month_Cos"] = np.cos(
    2 * np.pi * monthly_sales["Month"] / 12
)

X_new = monthly_sales[
    ["Year", "Month_Sin", "Month_Cos"]
]

y_new = monthly_sales["Sales"]

split_index = int(len(monthly_sales) * 0.8)

X_train_new = X_new.iloc[:split_index]
X_test_new = X_new.iloc[split_index:]

y_train_new = y_new.iloc[:split_index]
y_test_new = y_new.iloc[split_index:]

improved_model = LinearRegression()

improved_model.fit(X_train_new, y_train_new)

y_pred_new = improved_model.predict(X_test_new)

mae_new = mean_absolute_error(y_test_new, y_pred_new)
rmse_new = np.sqrt(
    mean_squared_error(y_test_new, y_pred_new)
)
r2_new = r2_score(y_test_new, y_pred_new)

print("\n--- IMPROVED MODEL RESULTS ---")
print("MAE:", round(mae_new, 2))
print("RMSE:", round(rmse_new, 2))
print("R2 Score:", round(r2_new, 4))


# --------------------------------
# STEP 12: RANDOM FOREST MODEL
# --------------------------------

from sklearn.ensemble import RandomForestRegressor

print("\n--- RANDOM FOREST MODEL ---")

X_rf = monthly_sales[["Year", "Month"]]
y_rf = monthly_sales["Sales"]

split_index = int(len(monthly_sales) * 0.8)

X_train_rf = X_rf.iloc[:split_index]
X_test_rf = X_rf.iloc[split_index:]

y_train_rf = y_rf.iloc[:split_index]
y_test_rf = y_rf.iloc[split_index:]

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train_rf, y_train_rf)

y_pred_rf = rf_model.predict(X_test_rf)

rf_mae = mean_absolute_error(y_test_rf, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
rf_r2 = r2_score(y_test_rf, y_pred_rf)

print("\n--- RANDOM FOREST RESULTS ---")
print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2 Score:", round(rf_r2, 4))


# --------------------------------
# STEP 13: ACTUAL VS PREDICTED GRAPH
# --------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["Order Date"].iloc[split_index:],
    y_test,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    monthly_sales["Order Date"].iloc[split_index:],
    y_pred,
    marker="o",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig("visualizations/actual_vs_predicted.png")

plt.show()

print("Actual vs Predicted graph saved successfully!")


print("\n--- PROJECT EXECUTION COMPLETED ---")
