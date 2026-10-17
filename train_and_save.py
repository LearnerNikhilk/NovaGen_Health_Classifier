"""Train the final Random Forest and save it with metadata for the app."""
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

df = pd.read_csv("novagen_dataset.csv")
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)
df = df.drop_duplicates()

bounds = {"Age": (0, 110), "BMI": (10, 70), "Blood_Pressure": (60, 260),
          "Heart_Rate": (30, 220), "Glucose_Level": (30, 600), "Cholesterol": (50, 600)}
mask = pd.Series(True, index=df.index)
for col, (lo, hi) in bounds.items():
    mask &= df[col].between(lo, hi)
df = df[mask].reset_index(drop=True)

X, y = df.drop(columns="Target"), df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Best parameters from the tuning step
model = RandomForestClassifier(n_estimators=300, max_features="log2", max_depth=None,
                               min_samples_split=2, min_samples_leaf=1,
                               random_state=42, n_jobs=1).fit(X_train, y_train)

pred, proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
print("Test accuracy:", round(accuracy_score(y_test, pred), 3),
      "| F1:", round(f1_score(y_test, pred), 3),
      "| ROC AUC:", round(roc_auc_score(y_test, proba), 3))

meta = {
    "features": list(X.columns),
    "stats": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                  "median": float(X[c].median())} for c in X.columns},
    "test_metrics": {"accuracy": round(float(accuracy_score(y_test, pred)), 3),
                     "f1": round(float(f1_score(y_test, pred)), 3),
                     "roc_auc": round(float(roc_auc_score(y_test, proba)), 3)},
}
joblib.dump(model, "model.joblib", compress=3)
json.dump(meta, open("model_meta.json", "w"), indent=1)
print("Saved model.joblib and model_meta.json")
