"""Train exam-score estimator from lecturer-uploaded historical assessment rows."""
import os, json
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from src.config import BASE_DIR

MODEL_PATH=os.path.join(BASE_DIR,'models','current_exam_model.joblib')
METRICS_PATH=os.path.join(BASE_DIR,'models','current_exam_metrics.json')
REQUIRED=['student_id','course','study_time_hours','test_score','practical_score','exam_score']

def validate_exam_frame(frame):
    df=frame.copy(); df.columns=[str(c).strip().lower().replace(' ','_') for c in df.columns]
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    df=df[REQUIRED]; df['student_id']=df.student_id.astype(str).str.strip(); df['course']=df.course.astype(str).str.strip().str.upper()
    if (df.student_id=='').any() or (df.course=='').any(): raise ValueError('Student ID and course cannot be blank.')
    for c in ['study_time_hours','test_score','practical_score','exam_score']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
        if df[c].isna().any(): raise ValueError(f'{c} contains blank or non-numeric values.')
    if not df.study_time_hours.between(0,80).all(): raise ValueError('Study time must be between 0 and 80 hours per week.')
    for c in ['test_score','practical_score','exam_score']:
        if not df[c].between(0,100).all(): raise ValueError(f'{c} must be between 0 and 100.')
    if df.duplicated(['student_id','course']).any(): raise ValueError('Each student and course combination must appear only once.')
    if df.student_id.nunique()<10: raise ValueError('At least 10 distinct students are required to train the exam estimator.')
    return df

def train_current_exam_model(frame):
    df=validate_exam_frame(frame); X=df[['study_time_hours','test_score','practical_score']]; y=df.exam_score
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    train_idx,test_idx=next(splitter.split(X,y,groups=df.student_id))
    pre=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),['study_time_hours','test_score','practical_score'])])
    models={'ExtraTrees':ExtraTreesRegressor(n_estimators=250,min_samples_leaf=2,random_state=42),'RandomForest':RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42),'GradientBoosting':GradientBoostingRegressor(random_state=42)}
    scores={}; best=None; best_mae=float('inf')
    for name,est in models.items():
        pipe=Pipeline([('preprocessor',pre),('model',est)]); pipe.fit(X.iloc[train_idx],y.iloc[train_idx]); pred=pipe.predict(X.iloc[test_idx])
        mae=float(mean_absolute_error(y.iloc[test_idx],pred)); scores[name]={'mae':round(mae,3),'rmse':round(float(mean_squared_error(y.iloc[test_idx],pred)**.5),3),'r2':round(float(r2_score(y.iloc[test_idx],pred)),3) if len(test_idx)>1 else None}
        if mae<best_mae: best,best_mae=name,mae
    final=Pipeline([('preprocessor',pre),('model',models[best])]); final.fit(X,y)
    os.makedirs(os.path.dirname(MODEL_PATH),exist_ok=True); joblib.dump(final,MODEL_PATH)
    metrics={'best_model':best,'models':scores,'rows':len(df),'students':int(df.student_id.nunique()),'courses':int(df.course.nunique()),'holdout_students':int(df.iloc[test_idx].student_id.nunique()),'features':['study_time_hours','test_score','practical_score'],'target':'exam_score'}
    with open(METRICS_PATH,'w') as f: json.dump(metrics,f,indent=2)
    return metrics

def predict_current_exam(study_time_hours,test_score,practical_score):
    if not os.path.exists(MODEL_PATH): raise ValueError('The lecturer has not uploaded training records and trained the exam prediction model yet.')
    model=joblib.load(MODEL_PATH)
    pred=float(model.predict(pd.DataFrame([{'study_time_hours':study_time_hours,'test_score':test_score,'practical_score':practical_score}]))[0])
    return round(max(0,min(100,pred)),1)

def exam_template_csv():
    return 'student_id,course,study_time_hours,test_score,practical_score,exam_score\nSTU001,SWD 411,8,75,82,78\nSTU002,SWD 411,5,65,70,68\n'
