import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NovaGen Health Risk Classifier", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    meta = json.load(open("model_meta.json"))
    return model, meta


model, meta = load_artifacts()
stats = meta["stats"]

st.title("🩺 NovaGen Health Risk Classifier")
st.write("Enter a participant's measurements to classify them as **Healthy** or **Unhealthy** "
         "(higher health risk). Built for research stratification, not for medical diagnosis.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Flag as Unhealthy if probability is at least",
        min_value=0.20, max_value=0.80, value=0.50, step=0.05,
        help="Lower values catch more at-risk participants but raise false alarms. "
             "0.50 is the standard setting the model was evaluated at.")
    st.header("About the model")
    st.write("Random Forest (300 trees) trained on 7,564 participants.")
    tm = meta["test_metrics"]
    st.write(f"Held-out test results: accuracy **{tm['accuracy']}**, "
             f"F1 **{tm['f1']}**, ROC-AUC **{tm['roc_auc']}**.")

# ---------------- Input form ----------------
# name: (label, min, max, step)  (ints for whole-number fields, floats otherwise)
CONTINUOUS = {
    "Age": ("Age (years)", 0, 110, 1),
    "BMI": ("BMI", 10.0, 70.0, 0.1),
    "Blood_Pressure": ("Systolic blood pressure (mmHg)", 60.0, 260.0, 1.0),
    "Cholesterol": ("Cholesterol (mg/dL)", 50.0, 600.0, 1.0),
    "Glucose_Level": ("Glucose level (mg/dL)", 30.0, 600.0, 1.0),
    "Heart_Rate": ("Resting heart rate (bpm)", 30.0, 220.0, 1.0),
    "Sleep_Hours": ("Sleep (hours per day)", 0.0, 24.0, 0.5),
    "Exercise_Hours": ("Exercise (hours per day)", 0.0, 24.0, 0.5),
    "Water_Intake": ("Water intake (litres per day)", 0.0, 20.0, 0.5),
    "Stress_Level": ("Stress level", 0.0, 15.0, 1.0),
}
LEVELS = {
    "Smoking": "Smoking (level)",
    "Alcohol": "Alcohol (level)",
    "Diet": "Diet (level)",
    "MentalHealth": "Mental health (level)",
    "PhysicalActivity": "Physical activity (level)",
    "MedicalHistory": "Medical history (level)",
    "Allergies": "Allergies (level)",
}

values = {}
with st.form("participant"):
    st.subheader("Measurements")
    cols = st.columns(2)
    for i, (name, (label, lo, hi, step)) in enumerate(CONTINUOUS.items()):
        default = stats[name]["median"]
        if isinstance(step, int):
            default = int(round(default))
        else:
            default = float(round(default, 1))
        values[name] = cols[i % 2].number_input(label, min_value=lo, max_value=hi,
                                                value=default, step=step)

    st.subheader("Lifestyle and history")
    st.caption("These fields are coded 0, 1 or 2 in the dataset; the dataset does not document "
               "what each level means.")
    cols = st.columns(2)
    for i, (name, label) in enumerate(LEVELS.items()):
        default = int(round(stats[name]["median"]))
        values[name] = cols[i % 2].selectbox(label, [0, 1, 2], index=default)

    diet_type = st.radio("Diet type", ["Neither", "Vegan", "Vegetarian"], horizontal=True)
    blood_group = st.radio("Blood group", ["A", "AB", "B", "O"], horizontal=True)

    submitted = st.form_submit_button("Classify participant", type="primary")

# ---------------- Prediction ----------------
if submitted:
    row = dict(values)
    row["Diet_Type__Vegan"] = int(diet_type == "Vegan")
    row["Diet_Type__Vegetarian"] = int(diet_type == "Vegetarian")
    row["Blood_Group_AB"] = int(blood_group == "AB")
    row["Blood_Group_B"] = int(blood_group == "B")
    row["Blood_Group_O"] = int(blood_group == "O")
    X = pd.DataFrame([row])[meta["features"]]

    # Warn about values outside the range the model was trained on
    outside = [name for name in CONTINUOUS
               if not (stats[name]["min"] <= row[name] <= stats[name]["max"])]
    if outside:
        st.warning("These values are outside the range seen in training, so the prediction may be "
                   "unreliable: " + ", ".join(outside))

    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    p_unhealthy = float(proba[classes.index(0)])
    p_healthy = float(proba[classes.index(1)])

    if p_unhealthy >= threshold:
        st.error(f"### Unhealthy (higher health risk)\nProbability of unhealthy: **{p_unhealthy:.1%}**")
    else:
        st.success(f"### Healthy\nProbability of healthy: **{p_healthy:.1%}**")

    c1, c2 = st.columns(2)
    c1.metric("P(Unhealthy)", f"{p_unhealthy:.1%}")
    c2.metric("P(Healthy)", f"{p_healthy:.1%}")
    st.progress(p_unhealthy, text="Estimated risk")

    with st.expander("Input sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}))

st.caption("Research tool. Predictions are based on statistical patterns in the training data and "
           "are not a medical diagnosis.")
