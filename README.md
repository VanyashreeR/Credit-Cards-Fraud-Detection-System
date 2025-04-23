# Credit Card Fraud Detection System

A machine learning system to detect fraudulent credit card transactions using the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Features
- Comprehensive data preprocessing
- Advanced feature engineering
- Multiple ML models with hyperparameter tuning
- Detailed evaluation metrics
- Class imbalance handling techniques

## Installation
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Download dataset from Kaggle and place in `data/` folder

## Usage
Run the notebooks in order:
1. Data Exploration
2. Feature Engineering
3. Model Training

Or run the Python scripts:
```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/model_training.py
## Result
Best model achieved:

AUPRC: 0.85

ROC AUC: 0.98

Recall: 0.82

Precision: 0.93
