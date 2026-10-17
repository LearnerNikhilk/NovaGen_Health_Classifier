# NovaGen Health Risk Classifier: deployment

Streamlit web app that serves the tuned Random Forest (test accuracy 0.937, ROC-AUC 0.982).

## Files
| File | Purpose |
|---|---|
| `app.py` | The web app |
| `model.joblib`, `model_meta.json` | Trained model and feature metadata (loaded by the app) |
| `requirements.txt` | Pinned package versions |
| `train_and_save.py` | Rebuilds the model from `novagen_dataset.csv` |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501.

## Deploy free on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `model.joblib`, `model_meta.json` and `requirements.txt`
   (the CSV and `train_and_save.py` are optional).
2. Go to https://share.streamlit.io, sign in with GitHub and choose **Create app**.
3. Select the repository, branch `main` and main file `app.py`, then click **Deploy**.
4. You get a public URL such as `https://your-app-name.streamlit.app`.

## Important
- **Keep `scikit-learn` pinned.** A model saved with one version can fail or behave differently when
  loaded with another. If you change the version, rerun `python train_and_save.py` to rebuild the model.
- The model was trained on the 80% training split (7,564 rows), so the reported test metrics apply to it.
- The app is a research tool, not a diagnostic device.
