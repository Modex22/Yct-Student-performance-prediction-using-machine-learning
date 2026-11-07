"""Train a first-semester to second-semester score model from paired records.

Expected columns: student_id, course, first_semester_score, second_semester_score.
This model is deliberately separate from the demonstration assessment model.
"""
from __future__ import annotations
import json, os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from src.config import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'semester_transition_model.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'semester_transition_metrics.json')
REQUIRED = ['student_id', 'course', 'first_semester_score', 'second_semester_score']

def validate_training_frame(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    df = df[REQUIRED]
    df['student_id'] = df.student_id.astype(str).str.strip()
    df['course'] = df.course.astype(str).str.strip().str.upper()
    if (df.student_id == '').any() or (df.course == '').any():
        raise ValueError('Student ID and course cannot be blank.')
    for col in ['first_semester_score', 'second_semester_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            raise ValueError(f'{col} contains blank or non-numeric values.')
        if not df[col].between(0, 100).all():
            raise ValueError(f'{col} must contain scores from 0 to 100.')
    if df.duplicated(['student_id', 'course']).any():
        raise ValueError('Each student_id + course combination must appear only once.')
    if df.student_id.nunique() < 10:
        raise ValueError('At least 10 distinct students with paired results are required before training.')
    return df

def train_transition_model(df):
    df = validate_training_frame(df)
    X = df[['course', 'first_semester_score']]
    y = df['second_semester_score']
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df.student_id))
    pre = ColumnTransformer([
        ('course', OneHotEncoder(handle_unknown='ignore'), ['course']),
        ('score', StandardScaler(), ['first_semester_score']),
    ])
    candidates = {
        'RandomForest': RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
    }
    results, best_name, best_mae = {}, None, float('inf')
    for name, estimator in candidates.items():
        pipe = Pipeline([('preprocessor', pre), ('model', estimator)])
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        mae = float(mean_absolute_error(y.iloc[test_idx], pred))
        results[name] = {'mae': round(mae, 3), 'rmse': round(float(mean_squared_error(y.iloc[test_idx], pred) ** .5), 3), 'r2': round(float(r2_score(y.iloc[test_idx], pred)), 3) if len(test_idx) > 1 else None}
        if mae < best_mae: best_name, best_mae = name, mae
    final = Pipeline([('preprocessor', pre), ('model', candidates[best_name])])
    final.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    metrics = {'best_model': best_name, 'models': results, 'rows': int(len(df)), 'students': int(df.student_id.nunique()), 'courses': int(df.course.nunique()), 'holdout_students': int(df.iloc[test_idx].student_id.nunique()), 'target': 'second_semester_score', 'features': ['course', 'first_semester_score']}
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=2)
    return metrics

def template_csv():
    return 'student_id,course,first_semester_score,second_semester_score\nSTU001,SWD 321,75,81\nSTU002,SWD 321,60,66\n'

def predict_second_semester(course, first_semester_score):
    if not os.path.exists(MODEL_PATH):
        raise ValueError('A paired first- and second-semester training dataset has not been uploaded by the administrator yet.')
    model=joblib.load(MODEL_PATH)
    value=float(model.predict(pd.DataFrame([{'course':str(course).strip().upper(),'first_semester_score':float(first_semester_score)}]))[0])
    return round(max(0.0,min(100.0,value)),1)
