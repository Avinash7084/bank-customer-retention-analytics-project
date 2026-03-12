# 1. Import Required Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import joblib



# 2. Load Dataset

df = pd.read_csv("Dataset/European_Bank.csv")

print("Dataset Loaded Successfully")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])


# 3. KPI Engineering (Business Metrics)

# Engagement Retention Ratio (ERR)

# Churn rate for active customers
churn_active = df[df["IsActiveMember"] == 1]["Exited"].mean()

# Churn rate for inactive customers
churn_inactive = df[df["IsActiveMember"] == 0]["Exited"].mean()

# ERR formula
ERR = churn_inactive / churn_active


#  Product Depth Index (PDI)
# Average number of products among retained customers
PDI = df[df["Exited"] == 0]["NumOfProducts"].mean()


#  Relationship Strength Index (RSI)
# Combines engagement + product usage + credit card ownership

df["RSI"] = (
    0.4 * df["IsActiveMember"] +
    0.4 * (df["NumOfProducts"] / 4) +
    0.2 * df["HasCrCard"]
)


# 4. Behavioral Customer Segmentation

def get_behavior_segment(row):

    # Active / Inactive
    if row["IsActiveMember"] == 1:
        activity = "Active"
    else:
        activity = "Inactive"

    # High value / Low value
    if row["Balance"] > 100000:
        value = "High Value"
    else:
        value = "Low Value"

    return activity + " + " + value


# Create new segmentation column
df["Behavior_Profile"] = df.apply(get_behavior_segment, axis=1)


# 5. Prepare Data for Machine Learning

# Remove unnecessary columns
df_ml = df.drop(["Year", "CustomerId", "Surname", "Behavior_Profile"], axis=1)


# Convert Gender into numeric
encoder = LabelEncoder()
df_ml["Gender"] = encoder.fit_transform(df_ml["Gender"])


# Convert Geography into dummy variables
df_ml = pd.get_dummies(df_ml, columns=["Geography"], drop_first=True)


# 6. Train Test Split

X = df_ml.drop("Exited", axis=1)
y = df_ml["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 7. Train Machine Learning Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# 8. Model Evaluation

y_prob = model.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_prob)


# 9. Save Model & Processed Dataset

joblib.dump(model, "churn_model.pkl")

df.to_csv("Final_Bank_Data_With_KPIs.csv", index=False)


# 10. Final Output

print("\nProject Completed Successfully")
print(f"Engagement Retention Ratio (ERR): {ERR:.2f}")
print(f"Product Depth Index (PDI): {PDI:.2f}")
print(f"Model AUC Score: {auc_score:.2f}")
