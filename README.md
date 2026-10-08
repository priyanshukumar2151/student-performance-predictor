# Student Performance Predictor

Machine Learning project built for the **AICTE | IBM SkillsBuild Machine Learning & Applied AI Internship 2026**.

Predicts whether a student will **pass or fail** from study habits, attendance and family/social factors, and gives simple improvement suggestions through a Streamlit web app.

## Dataset
UCI Student Performance dataset (`student-mat.csv`): https://archive.ics.uci.edu/dataset/320/student+performance
Target: pass if final grade G3 >= 10. Previous grades (G1, G2) are excluded so the model learns from habits.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

## How to run
```bash
pip install -r requirements.txt
python train.py          # trains models, saves charts + model.joblib
streamlit run app.py     # launches the web app
```

## Models compared
Logistic Regression, Decision Tree, Random Forest (tuned with GridSearchCV, 5-fold CV).
Metrics: Accuracy, Precision, Recall, F1 (see `model_comparison.csv`).

## Files
- `train.py` - data cleaning, EDA, training, evaluation
- `app.py` - Streamlit prediction app
- `model_comparison.csv`, `*.png` - generated results

## Future scope
Add the Portuguese-course dataset, try XGBoost, add explainability (SHAP), deploy on Streamlit Cloud.
