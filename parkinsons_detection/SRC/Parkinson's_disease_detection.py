"""
Dataset: Parkinson's Disease Detection
Samples: 195 patients
Target:
- status = 1 → Parkinson’s
- status = 0 → Healthy
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load the Parkinson's disease dataset
parkinson_data = pd.read_csv('/Users/devanshbansal/Downloads/parkinsons data.csv')

#displaying the total number of null values in each column and checking where we have null values and cleaning is required
print(parkinson_data.isnull().sum())
parkinson_data= parkinson_data.drop(columns=["name"], errors="ignore")

# Summary statistics for all numerical features
print(parkinson_data.describe())


# Count number of ill (1) and not ill (0)
status_counts = parkinson_data["status"].value_counts()


# Labels and values
labels = ["Not Ill", "Ill"]
values = [status_counts[0], status_counts[1]]


# Create bar chart
plt.figure()
plt.bar(labels, values)


# Axis labels
plt.xlabel("Health Status")
plt.ylabel("Number of Persons")

# Chart title
plt.title("Distribution of Parkinson’s Disease Cases")

# Show values on top of bars
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha="center", va="bottom")
plt.show()


# Separate features and target
X = parkinson_data.drop(columns=["status"])
y = parkinson_data["status"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, #control randomness
    stratify=y# to keep classification balanced in train and test sets
)
#decision tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)


#improving the decision tree model by tuning hyperparameters
dt_model = DecisionTreeClassifier(
    max_depth=5,            # control tree depth
    min_samples_split=5,    # prevent very small splits
    min_samples_leaf=2,     # smoother leaf nodes
    random_state=42
)

#training the improved decision tree model
dt_model.fit(X_train, y_train)
#predictions
y_pred_dt = dt_model.predict(X_test)
#accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Improved Decision Tree Accuracy:", accuracy_dt)

# Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
# Train the model
log_model.fit(X_train, y_train)
# Predictions
y_pred = log_model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)

#improving logistic regression by tuning hyperparameters
# Feature scaling (VERY IMPORTANT for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Tuned Logistic Regression model
log_model = LogisticRegression(
    C=1.0,              # regularization strength of how much they are allowed to fit
    solver="lbfgs",     # stable solver
    max_iter=1000
)
# Train model
log_model.fit(X_train_scaled, y_train)
# Predictions
y_pred = log_model.predict(X_test_scaled)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Tuned Logistic Regression Accuracy:", accuracy)


#random forest model
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf)


#improving the random forest model by tuning hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=300,        # more trees = more stability
    max_depth=10,            # prevents overfitting
    min_samples_split=5,     # avoid very small splits
    min_samples_leaf=2,      # smoother decision boundaries
    random_state=42
)


#training the improved random forest model
rf_model.fit(X_train, y_train)
#predictions
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
#accuracy
print("Improved Random Forest Accuracy:", accuracy_rf)

# -----------------------------
# Model Comparison Graph
# -----------------------------

models = ["Decision Tree", "Logistic Regression", "Random Forest"]
accuracies = [accuracy_dt, accuracy, accuracy_rf]

plt.figure()

bars = plt.bar(models, accuracies)

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Comparison of Model Performance for Parkinson's Detection")

# show accuracy values on bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center')

plt.ylim(0, 1)
plt.show()

print("\nBest Model: Tuned Logistic Regression with Accuracy:", accuracy)

# FINAL MODEL (after training)
final_model = log_model   # tuned logistic regression
final_scaler = scaler     # fitted scaler
def predict_parkinsons(input_data):
    """
    input_data: list or array of feature values in the SAME ORDER as X columns
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    # Scale input
    input_scaled = final_scaler.transform(input_df)
    
    # Predict
    prediction = final_model.predict(input_scaled)[0]
    
    # Interpret result
    if prediction == 1:
        return "Parkinson’s Disease Detected"
    else:
        return "Healthy (No Parkinson’s Detected)"
print("\nEnter patient voice measurements:")

user_input = []
for feature in X.columns:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

result = predict_parkinsons(user_input)

print("\n==============================")
print("Prediction Result:", result)
print("==============================")
