# NovaGen Health Risk Classifier

Supervised machine learning project (Assignment 5): classify individuals as **healthy** or **unhealthy** (higher health risk) from physiological, lifestyle and medical-history data, to support participant selection and risk-based stratification at NovaGen Research Labs.

**Result:** a Random Forest reaches **93.7% accuracy, F1 0.940 and ROC-AUC 0.982** on a held-out test set. It is served through a Streamlit web app.

## Dataset
- `novagen_dataset.csv`: 9,549 participants, 22 features plus `Target`.
- No missing values or duplicates. Target is balanced (52% class 1, 48% class 0).
- **Label meaning (inferred):** `0 = Unhealthy`, `1 = Healthy`. Class 0 is older with much higher systolic blood pressure. The dataset does not state this, so confirm it with the data owner.

| Group | Features |
|---|---|
| Continuous | Age, BMI, Blood_Pressure, Cholesterol, Glucose_Level, Heart_Rate, Sleep_Hours, Exercise_Hours, Water_Intake, Stress_Level |
| Coded 0/1/2 | Smoking, Alcohol, Diet, MentalHealth, PhysicalActivity, MedicalHistory, Allergies |
| One-hot | Diet_Type__Vegan, Diet_Type__Vegetarian, Blood_Group_AB, Blood_Group_B, Blood_Group_O |

### Data issues
- 9,549 rows supplied vs 9,800 stated in the brief.
- 93 rows with systolic blood pressure below 60 mmHg were removed as implausible (9,456 rows kept).
- `Stress_Level` goes up to 12 on a 1-10 scale, and `Age` is 0 for 96 rows. These were kept and flagged.
- The 0/1/2 fields are undocumented, and several variables have very narrow ranges, which suggests synthetic or heavily processed data.

## Method
1. Inspect the data, check the label meaning and plot distributions and correlations.
2. Clean implausible values and split 80/20 (stratified, `random_state=42`).
3. Scale continuous features inside a pipeline, so nothing leaks from the test set.
4. Compare five models with 5-fold cross-validation.
5. Tune the top two (Random Forest, SVM) with randomized search, scored on F1.
6. Evaluate the chosen model once on the test set, then check feature importance.

## Results
**Cross-validation (training set)**

| Model | F1 | ROC-AUC |
|---|---|---|
| Random Forest | 0.940 | 0.980 |
| SVM | 0.938 | 0.979 |
| KNN | 0.930 | 0.976 |
| Gradient Boosting | 0.922 | 0.971 |
| Logistic Regression | 0.821 | 0.888 |

**Test set (selected model: Random Forest)**

| Accuracy | F1 (Unhealthy) | F1 (Healthy) | Recall (Unhealthy) | ROC-AUC |
|---|---|---|---|---|
| 0.937 | 0.934 | 0.940 | 0.918 | 0.982 |

Tuning gave almost no gain over the defaults, and the SVM is effectively tied with the Random Forest. The top features under both impurity and permutation importance are **BMI, Blood_Pressure and Cholesterol**. About 8% of truly unhealthy participants are classified as healthy, which matters for trial selection; the app's threshold slider lets you trade precision for recall.

## Project structure
```
├── README.md                       This file
├── novagen_dataset.csv             Dataset
├── NovaGen_Health_Classifier.ipynb Full pipeline, step by step
├── NovaGen_Health_Risk_Report.md   Written report with results and limitations
├── outputs/                        EDA, evaluation and importance figures
└── deploy/
    ├── app.py                      Streamlit web app
    ├── model.joblib                Trained Random Forest
    ├── model_meta.json             Feature order, ranges and test metrics
    ├── train_and_save.py           Rebuilds the model from the CSV
    ├── requirements.txt            Pinned package versions
    └── README.md                   Deployment guide
```

## Quick start
```bash
pip install -r deploy/requirements.txt jupyter matplotlib seaborn

# Run the analysis
jupyter notebook NovaGen_Health_Classifier.ipynb

# Run the web app
cd deploy
streamlit run app.py
```
The app opens at http://localhost:8501.

## Deployment
The app can be hosted free on Streamlit Community Cloud: upload the files in `deploy/` (without the CSV, if you like) to a GitHub repository, then create an app at https://share.streamlit.io pointing at `app.py`. See `deploy/README.md` for the steps.

**Keep `scikit-learn` pinned to 1.8.0.** A model saved with one version can fail when loaded with another. If you change the version, rerun `deploy/train_and_save.py`.

## Limitations
- The label meaning is inferred, not documented.
- The data appear synthetic or processed, so results may not transfer to real clinical populations.
- The data are observational: feature importance shows association, not causation.
- The 93 low blood pressure participants were excluded, so predictions for very low blood pressure are unreliable.
- The meaning of the 0/1/2 levels is unknown, which limits interpretation of those features.
- This is a research stratification tool, **not a medical diagnostic device**.

## Requirements
Python 3.10+ with `pandas`, `numpy`, `scikit-learn==1.8.0`, `matplotlib`, `seaborn`, `joblib` and `streamlit`.
