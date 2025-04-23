from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, 
                           average_precision_score)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def train_models(X_train, y_train):
    """Train multiple models with resampling"""
    
    # Define models
    models = {
        'RandomForest': RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            eval_metric='aucpr',
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            is_unbalance=True,
            metric='average_precision',
            random_state=42
        )
    }
    
    # Resampling strategy
    resampling = Pipeline([
        ('oversample', SMOTE(sampling_strategy=0.1, random_state=42)),
        ('undersample', RandomUnderSampler(sampling_strategy=0.5, random_state=42))
    ])
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with resampling and model
        pipeline = Pipeline([
            ('resampling', resampling),
            ('classifier', model)
        ])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        
        # Store results
        results[name] = {
            'auprc': average_precision_score(y_train, y_pred),
            'roc_auc': roc_auc_score(y_train, y_pred)
        }
        
        # Fit on full training data
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f'../models/{name.lower()}_model.pkl')
    
    return results

def evaluate_models(models, X_test, y_test):
    """Evaluate models on test set"""
    
    results = []
    for name in models:
        model = joblib.load(f'../models/{name.lower()}_model.pkl')
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auprc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Save results
        results.append({
            'Model': name,
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1-Score': report['1']['f1-score'],
            'AUPRC': auprc,
            'ROC AUC': roc_auc
        })
        
        # Save confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'../results/confusion_matrices/{name.lower()}_cm.png')
        plt.close()
    
    # Save metrics
    pd.DataFrame(results).to_csv('../results/metrics.csv', index=False)
    return results