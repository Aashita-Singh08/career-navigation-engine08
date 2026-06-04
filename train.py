import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

def preprocess_data(df):
    data = df.copy()
    
    binary_cols = [
        'self-learning capability?', 
        'Extra-courses did', 
        'Taken inputs from seniors or elders', 
        'worked in teams ever?', 
        'Introvert'
    ]
    for col in binary_cols:
        data[col] = data[col].str.lower().map({'yes': 1, 'no': 0})
        
    ordinal_cols = ['reading and writing skills', 'memory capability score']
    skill_map = {'poor': 0, 'medium': 1, 'excellent': 2}
    for col in ordinal_cols:
        data[col] = data[col].str.lower().map(skill_map)
        
    data['Management or Technical'] = data['Management or Technical'].str.lower().map({'technical': 1, 'management': 0})
    data['hard/smart worker'] = data['hard/smart worker'].str.lower().map({'smart worker': 1, 'hard worker': 0})
    
    categorical_cols = [
        'certifications',
        'workshops',
        'Interested subjects',
        'interested career area ',
        'Type of company want to settle in?',
        'Interested Type of Books'
    ]
    
    mappings = {}
    for col in categorical_cols:
        unique_vals = sorted(list(data[col].unique()))
        val_map = {val: idx for idx, val in enumerate(unique_vals)}
        data[col] = data[col].map(val_map)
        mappings[col] = val_map
        
    target_col = 'Suggested Job Role'
    unique_targets = sorted(list(data[target_col].unique()))
    target_map = {val: idx for idx, val in enumerate(unique_targets)}
    reverse_target_map = {idx: val for idx, val in enumerate(unique_targets)}
    data[target_col] = data[target_col].map(target_map)
    
    mappings['target'] = target_map
    mappings['target_reverse'] = reverse_target_map
    
    return data, mappings

def train():
    if not os.path.exists('PS2_Dataset.csv'):
        print("Dataset not found!")
        return
        
    df = pd.read_csv('PS2_Dataset.csv')
    processed_df, mappings = preprocess_data(df)
    
    X = processed_df.drop(columns=['Suggested Job Role'])
    y = processed_df['Suggested Job Role']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=101, stratify=y
    )
    
    # Train Random Forest (Ensemble Method)
    rf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=12, 
        min_samples_split=4,
        random_state=101
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    # Train MLP Classifier (Neural Network for Deep Learning concept)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        max_iter=300, 
        random_state=101,
        early_stopping=True,
        n_iter_no_change=15
    )
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    print(f"Neural Network (MLP) Accuracy: {mlp_acc:.4f}")
    
    best_model = rf if rf_acc >= mlp_acc else mlp
    best_name = "Random Forest" if rf_acc >= mlp_acc else "Neural Network (MLP)"
    print(f"Selecting {best_name} as the production model.")
    
    assets = {
        'model': best_model,
        'mappings': mappings,
        'features': list(X.columns),
        'rf_accuracy': rf_acc,
        'mlp_accuracy': mlp_acc,
        'best_model_name': best_name
    }
    
    with open('model_assets.pkl', 'wb') as f:
        pickle.dump(assets, f)
    print("Model assets saved successfully to model_assets.pkl")

if __name__ == '__main__':
    train()
