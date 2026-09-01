"""
Model training, evaluation, persistence, and test prediction module.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import FEATURE_COLS, MODELS_DIR, TRAIN_UNTIL, VAL_UNTIL


def train_and_save(df_model: pd.DataFrame, class_names):
    """Train RF + XGBoost with GridSearch, calibrate, save best model."""
    mask_train = df_model['Fecha'] < TRAIN_UNTIL
    mask_val = (df_model['Fecha'] >= TRAIN_UNTIL) & (df_model['Fecha'] < VAL_UNTIL)
    mask_test = df_model['Fecha'] >= VAL_UNTIL
    
    X_train = df_model.loc[mask_train, FEATURE_COLS].values
    y_train = df_model.loc[mask_train, 'Target'].values
    X_val = df_model.loc[mask_val, FEATURE_COLS].values
    y_val = df_model.loc[mask_val, 'Target'].values
    X_test = df_model.loc[mask_test, FEATURE_COLS].values
    y_test = df_model.loc[mask_test, 'Target'].values
    
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test) if len(X_test) > 0 else X_test
    X_trainval_sc = scaler.fit_transform(X_trainval)
    
    cv = StratifiedKFold(n_splits=3, shuffle=False)
    
    # Random Forest
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            'n_estimators': [200, 400],
            'max_depth': [None, 8, 15],
            'min_samples_leaf': [1, 3],
            'class_weight': ['balanced'],
        },
        cv=cv,
        scoring='neg_log_loss',
        n_jobs=-1,
        refit=True,
    )
    rf_grid.fit(X_trainval_sc, y_trainval)
    
    # XGBoost
    xgb_grid = GridSearchCV(
        XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        {
            'n_estimators': [200, 400],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        },
        cv=cv,
        scoring='neg_log_loss',
        n_jobs=-1,
        refit=True,
    )
    xgb_grid.fit(X_trainval_sc, y_trainval)
    
    # Pick winner by val log_loss
    rf_best = rf_grid.best_estimator_
    xgb_best = xgb_grid.best_estimator_
    ll_rf = log_loss(y_val, rf_best.predict_proba(X_val_sc), labels=[0, 1, 2])
    ll_xgb = log_loss(y_val, xgb_best.predict_proba(X_val_sc), labels=[0, 1, 2])
    
    if ll_xgb <= ll_rf:
        best_model, best_name = xgb_best, 'XGBoost'
    else:
        best_model, best_name = rf_best, 'Random Forest'
    
    # Calibrate
    best_model_base = clone(best_model)
    best_model_base.fit(X_train_sc, y_train)
    try:
        calibrated = CalibratedClassifierCV(estimator=best_model_base, method='isotonic', cv='prefit')
        calibrated.fit(X_val_sc, y_val)
    except ValueError:
        calibrated = CalibratedClassifierCV(estimator=best_model_base, method='isotonic', cv=3)
        calibrated.fit(X_val_sc, y_val)
    
    if len(X_test_sc) > 0:
        ll_uncal = log_loss(y_test, best_model.predict_proba(X_test_sc))
        ll_cal = log_loss(y_test, calibrated.predict_proba(X_test_sc))
        final_model = calibrated if ll_cal <= ll_uncal else best_model
    else:
        final_model = calibrated
    
    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / 'model.joblib')
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')
    joblib.dump(class_names, MODELS_DIR / 'class_names.joblib')
    joblib.dump(best_name, MODELS_DIR / 'model_name.joblib')
    
    return final_model, scaler, class_names, best_name


def load_model():
    """Load pre-trained model, scaler, and class names from disk."""
    model = joblib.load(MODELS_DIR / 'model.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
    class_names = joblib.load(MODELS_DIR / 'class_names.joblib')
    model_name = joblib.load(MODELS_DIR / 'model_name.joblib')
    return model, scaler, class_names, model_name


def model_exists() -> bool:
    """Check if pre-trained model artifacts exist on disk."""
    return (MODELS_DIR / 'model.joblib').exists()


def get_test_predictions(df_model: pd.DataFrame, model, scaler, class_names, n: int = 6):
    """Get the last N predictions from the test set with actual results."""
    mask_test = df_model['Fecha'] >= VAL_UNTIL
    df_test = df_model[mask_test].copy()
    if df_test.empty:
        return []
    X_test = scaler.transform(df_test[FEATURE_COLS].values)
    probas = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    results = []
    class_list = list(class_names)
    for i, (_, row) in enumerate(df_test.tail(n).iterrows()):
        idx = len(df_test) - n + i
        if idx < 0:
            continue
        p = probas[idx]
        pred_class = class_list[preds[idx]]  # 'A', 'D', or 'H'
        actual = row['Resultado']
        
        if pred_class == 'H':
            pred_label = 'HOME WIN'
        elif pred_class == 'A':
            pred_label = 'AWAY WIN'
        else:
            pred_label = 'DRAW'
        
        prob_pct = p[preds[idx]] * 100
        correct = (pred_class == actual)
        
        results.append({
            'home': row['EquipoLocal'],
            'away': row['EquipoVisitante'],
            'score': f"{int(row['GL'])}-{int(row['GV'])}",
            'prediction': pred_label,
            'confidence': prob_pct,
            'correct': correct,
            'fecha': row['Fecha'],
        })
    return results
