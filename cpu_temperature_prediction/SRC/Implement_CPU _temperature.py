#Google Cloud AI aims to improve the reliability of large-scale computing infrastructure by
# predicting server CPU temperature under varying workloads. Using parameters such as CPU utilization,
# memory usage, clock speed, ambient temperature, voltage, and current load, develop a supervised machine
# learning regression model to estimate server temperature. The model should help in proactive cooling management and failure prevention
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
# ===============================
# 1. Load Dataset
# ===============================
server_data = pd.read_csv('/Users/devanshbansal/Downloads/server_cpu_dataset.csv')
# ===============================
# 2. Data Cleaning & Preprocessing through EDA
# ===============================
# first we will check for null values in our dataset
print(server_data.isnull().sum())
server_data = server_data.drop_duplicates()
server_data = server_data.dropna()
print(server_data.isnull().sum())# no missing data now so no EDA needed for missing data


# Convert numeric columns safely 
# creating a list named numeric_cols which contains the names of the columns as a string which are 
# supposed to be numeric in excel 
numeric_cols = [
    'CPU Usage (%)',
    'Memory Usage (%)',
    'Clock Speed (GHz)',
    'Ambient Temperature (°C)',
    'Voltage (V)',
    'Current Load (A)',
    'Cache Miss Rate (%)',
    'Power Consumption (W)',
    'CPU Temperature (°C)'
]

# converting all our numeric columns to numeric using pd.to_numeric and error is set to 'coerce' so that 
# if there are any invalid values they are converted to NaN
for col in numeric_cols:
    server_data[col] = pd.to_numeric(server_data[col], errors='coerce')
# Remove rows that had invalid values
server_data = server_data.dropna()


# ===============================
# 3. Exploratory Data Analysis (EDA)
# ===============================
# -------- EDA 1: Distribution of CPU Temperature  --------
plt.figure()
plt.scatter(
    server_data['CPU Temperature (°C)'],
    server_data['CPU Usage (%)'],
    alpha=0.05# to change the transparency of the points so that we can see the density of points in areas where they overlap
)

# Trend line
z = np.polyfit( #fits the curve to the data points
    server_data['CPU Temperature (°C)'], #x-value independent variable
    server_data['CPU Usage (%)'], #y-value dependent variable
    1 # straight line degree of the polynomial
)
p = np.poly1d(z) # to convert curve into a mathematical function taht we used to plot the trend line
plt.plot(
    server_data['CPU Temperature (°C)'],
    p(server_data['CPU Temperature (°C)']),
    color='red'
)

plt.xlabel('CPU Temperature (°C)')
plt.ylabel('CPU Usage (%)')
plt.title('CPU Usage vs CPU Temperature')
plt.show()

# -------- EDA 2: Distribution of CPU Temperature  --------
plt.figure()
plt.hist(
    server_data['CPU Temperature (°C)'],
    bins=25, #number of intervals in which the data is divided
    edgecolor='black',
    alpha=0.7,
    color='skyblue'
)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('CPU Temperature (°C)')
plt.ylabel('Frequency')
plt.title('Distribution of CPU Temperature')
plt.show()

# -------- EDA 3: Power Consumption vs CPU Temperature  --------
plt.figure()
plt.scatter(
    server_data['Power Consumption (W)'],
    server_data['CPU Temperature (°C)'],
    alpha=0.05
)
plt.xlabel('Power Consumption (W)')
plt.ylabel('CPU Temperature (°C)')
plt.title('Power Consumption vs CPU Temperature')
plt.show()



# -------- EDA 4: Voltage vs CPU Temperature  --------
plt.figure(figsize=(8, 6))

# Scatter plot
plt.scatter(
    server_data['Voltage (V)'],
    server_data['CPU Temperature (°C)'],
    alpha=0.05
)
# Trend line
z = np.polyfit(
    server_data['Voltage (V)'],
    server_data['CPU Temperature (°C)'],
    1
)
p = np.poly1d(z)

plt.plot(
    server_data['Voltage (V)'],
    p(server_data['Voltage (V)']),
    color='red'
)

plt.xlabel('Voltage (V)')
plt.ylabel('CPU Temperature (°C)')
plt.title('Voltage vs CPU Temperature')
plt.show()

# -------- EDA 5: Current Load vs CPU Temperature  --------
plt.figure(figsize=(8, 6))
# Scatter plot
plt.scatter(
    server_data['Current Load (A)'],
    server_data['CPU Temperature (°C)'],
    alpha=0.05
)
# Trend line
z = np.polyfit(
    server_data['Current Load (A)'],
    server_data['CPU Temperature (°C)'],
    1
)
p = np.poly1d(z)
plt.plot(
    server_data['Current Load (A)'],
    p(server_data['Current Load (A)']),
    color='red'
)

plt.xlabel('Current Load (A)')
plt.ylabel('CPU Temperature (°C)')
plt.title('Current Load vs CPU Temperature')
plt.show()
# ===============================
# 4. Random Forest Regression
# ===============================

# -------- Feature Selection --------
X = server_data[
    [
        'CPU Usage (%)',
        'Memory Usage (%)',
        'Clock Speed (GHz)',
        'Ambient Temperature (°C)',
        'Voltage (V)',
        'Current Load (A)',
        'Cache Miss Rate (%)',
        'Power Consumption (W)',
    ]
]
y = server_data['CPU Temperature (°C)']


def train_test_rf(X, y, test_size=0.2, n_estimators=500, max_depth=12, 
                  min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=42):
    """
    Train and evaluate Random Forest Regressor.
    
    Returns:
        model: trained RandomForestRegressor
        metrics: dict containing MAE, RMSE, R2
        y_test, y_pred: test labels and predictions
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state
    )
    
    # Train
    rf_model.fit(X_train, y_train)
    
    # Predict
    y_pred = rf_model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    
    return rf_model, metrics, y_test, y_pred
# tell us the varience of the dependent/element to be found/predicted based on the independent variables/
# elements we are using to predict it. R2 score of 1 means perfect prediction and 0 means the model is
# not able to predict anything better than the mean of the target variable. Negative R2 score means the
# model is performing worse than a horizontal line (mean of target variable) which is a very bad sign for
# our model.

rf_20, metrics_20, y20_test, y20_pred = train_test_rf(X, y, test_size=0.2)
rf_10, metrics_10, y10_test, y10_pred = train_test_rf(X, y, test_size=0.1)
rf_30, metrics_30, y30_test, y30_pred = train_test_rf(X, y, test_size=0.3)

rf_models = [rf_10, rf_20, rf_30]
rf_metrics = [metrics_10["R2"], metrics_20["R2"], metrics_30["R2"]]

best_rf_model = rf_models[np.argmax(rf_metrics)]
best_index = np.argmax(rf_metrics)
print(f"Best Random Forest selected: rf_{[10,20,30][best_index]} with R2={rf_metrics[best_index]:.4f}")

# Global train-test split for DT & LR
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------- Decision Tree --------
dt_model = DecisionTreeRegressor(
    max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42
)
dt_model.fit(X_train, y_train)
y_dt_pred = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, y_dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, y_dt_pred))
dt_r2 = r2_score(y_test, y_dt_pred)

print("\nDecision Tree MAE:", dt_mae)
print("Decision Tree RMSE:", dt_rmse)
print("Decision Tree R2 Score:", dt_r2)


# -------- Linear Regression --------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, y_lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_lr_pred))
lr_r2 = r2_score(y_test, y_lr_pred)

print("\nLinear Regression MAE:", lr_mae)
print("Linear Regression RMSE:", lr_rmse)
print("Linear Regression R2 Score:", lr_r2)




models = ["Random Forest", "Decision Tree", "Linear Regression"]
mae_values = [metrics_20["MAE"], dt_mae, lr_mae]
rmse_values = [metrics_20["RMSE"], dt_rmse, lr_rmse]
r2_values = [metrics_20["R2"], dt_r2, lr_r2]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, mae_values, width, label="MAE")
rects2 = ax.bar(x, rmse_values, width, label="RMSE")
rects3 = ax.bar(x + width, r2_values, width, label="R2")

ax.set_ylabel("Metric Value")
ax.set_title("Model Comparison: Random Forest, Decision Tree, Linear Regression")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.show()


# picking the best model based on R2 score which is Random Forest in this case and we will use it for user input prediction in the next step
# Pick best model based on R2
# Create lists of models and their R2 scores
models_list = [best_rf_model, dt_model, lr_model]
r2_scores = [max(metrics_20), dt_r2, lr_r2]

# Pick the best model based on R2 score
best_model = models_list[np.argmax(r2_scores)]
# Print the best model based on R2 score after the graph
best_model_name = models[np.argmax(r2_values)]
print(f"\nBased on R2 score, the best performing model is: {best_model_name} with R2 = {max(r2_values):.4f}")

# ===============================
# 5. User Input Prediction
# ===============================

print("\nEnter system metrics to predict CPU temperature")

usage = float(input("CPU Usage (%): "))
memory = float(input("Memory Usage (%): "))
clock = float(input("Clock Speed (GHz): "))
ambient = float(input("Ambient Temperature (°C): "))
voltage = float(input("Voltage (V): "))
current = float(input("Current Load (A): "))
cache = float(input("Cache Miss Rate (%): "))
power = float(input("Power Consumption (W): "))

# Prepare data for model
user_input = pd.DataFrame(
    [[usage, memory, clock, ambient, voltage, current, cache, power]],
    columns=[
        'CPU Usage (%)',
        'Memory Usage (%)',
        'Clock Speed (GHz)',
        'Ambient Temperature (°C)',
        'Voltage (V)',
        'Current Load (A)',
        'Cache Miss Rate (%)',
        'Power Consumption (W)',
    ]
)

# Predict
predicted_temp = best_model.predict(user_input)

# Show result
print("\nPredicted CPU Temperature:", round(predicted_temp[0], 2), "°C")
