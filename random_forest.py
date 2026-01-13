from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train_and_evaluate_random_forest_regressor(X, y, cv_indices, X_prediction=None, n_estimators=500, random_state=0):
    """
    Train and evaluate Random Forest Regressor, returning the model with the lowest MAE.
    """
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST REGRESSOR (Selection Metric: MAE)")
    print("="*70)
    
    cv_mae_scores = []
    cv_mse_scores = []
    cv_r2_scores = []
    cv_accuracy_scores = []
    
    best_mae = float('inf')
    best_model = None
    best_fold_idx = -1
    
    print("\nTraining on folds...")
    for fold_idx, (train_idx, val_idx) in enumerate(cv_indices):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        rf.fit(X_train_fold, y_train_fold)
        y_pred = rf.predict(X_val_fold)
        
        mae = mean_absolute_error(y_val_fold, y_pred)
        mse = mean_squared_error(y_val_fold, y_pred)
        r2 = r2_score(y_val_fold, y_pred)
        
        y_val_class = (y_val_fold >= -6).astype(int)  # True values: 1=permeable, 0=not
        y_pred_class = (y_pred >= -6).astype(int)     # Predicted values: 1=permeable, 0=not
        accuracy = accuracy_score(y_val_class, y_pred_class)
        
        cv_mae_scores.append(mae)
        cv_mse_scores.append(mse)
        cv_r2_scores.append(r2)
        cv_accuracy_scores.append(accuracy)
        
        print(f"Fold {fold_idx+1}/{len(cv_indices)}: MAE = {mae:.4f}, MSE = {mse:.4f}, R2 = {r2:.4f}, Accuracy = {accuracy:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = rf
            best_fold_idx = fold_idx
    
    # Summary Statistics
    mean_mae = np.mean(cv_mae_scores)
    std_mae = np.std(cv_mae_scores)
    mean_accuracy = np.mean(cv_accuracy_scores)  # NEW
    std_accuracy = np.std(cv_accuracy_scores)    # NEW
    
    print(f"\n{'='*70}")
    print(f"Cross-Validation Results:")
    print(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"Mean R2: {np.mean(cv_r2_scores):.4f}")
    print(f"Mean Accuracy (threshold=-6): {mean_accuracy:.4f} ± {std_accuracy:.4f}")  # NEW
    print(f"Best Model found in Fold {best_fold_idx+1} (MAE: {best_mae:.4f})")
    print(f"{'-'*70}")
    print("Top 20 Features by Importance (Best Model):")
    
    if best_model is not None:
        importances = best_model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print(feature_importance_df.head(20).to_string(index=False))
    
    print(f"{'='*70}\n")
    
    predictions = None
    if X_prediction is not None and best_model is not None:
        print("="*70)
        print(f"PREDICTING ON {len(X_prediction)} UNLABELED MOLECULES (No PAMPA)")
        print("="*70)
        
        predictions = best_model.predict(X_prediction)
        
        pred_classes = (predictions >= -6).astype(int)
        n_permeable = np.sum(pred_classes)
        n_not_permeable = len(pred_classes) - n_permeable
        
        print(f"\nClassification Results (threshold=-6):")
        print(f"  Permeable (>=6): {n_permeable} molecules")
        print(f"  Not Permeable (<-6): {n_not_permeable} molecules")
        
        # Display first few predictions as a sanity check
        print("\nFirst 5 predictions:")
        for i, pred in enumerate(predictions[:5]):
            class_label = "Permeable" if pred >= -6 else "Not Permeable"
            print(f"Molecule {X_prediction.index[i]}: {pred:.4f} ({class_label})")
        print(f"\nPredictions generated successfully.\n")
    
    if X_prediction is not None:
        return best_model, predictions
        
    return best_model

def encode_categorical_features(features_df, encoders=None, fit=True):
    encoded_df = features_df.copy()
    non_numeric_cols = encoded_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(non_numeric_cols) == 0:
        print("All columns are already numeric.")
        return encoded_df, {}
    
    print(f"\nEncoding {len(non_numeric_cols)} non-numeric columns...")
    
    if encoders is None:
        encoders = {}
    
    for col in non_numeric_cols:
        if fit:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            encoders[col] = le
            print(f"  '{col}': {len(le.classes_)} unique values -> [0, {len(le.classes_)-1}]")
        else:
            if col not in encoders:
                raise ValueError(f"Encoder for column '{col}' not found in provided encoders")
            
            le = encoders[col]
            encoded_values = []
            for val in encoded_df[col].astype(str):
                if val in le.classes_:
                    encoded_values.append(le.transform([val])[0])
                else:
                    print(f"  Warning: Unseen value '{val}' in column '{col}', mapping to class 0")
                    encoded_values.append(0)
            encoded_df[col] = encoded_values
    
    return encoded_df, encoders
