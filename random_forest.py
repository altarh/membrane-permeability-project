from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def train_and_evaluate_random_forest(X, y, cv_indices, n_estimators=500, 
                                      class_weight='balanced', random_state=0):
    """
    Train and evaluate Random Forest with k-fold cross-validation.
    
    Parameters:
    -----------
    X : Feature matrix
    y : Target labels
    cv_indices : List of (train_idx, val_idx) from create_tanimoto_kfold_partition
    n_estimators : Number of trees in Random Forest
    class_weight : Class weights for handling imbalance
    random_state : Random seed
        
    Returns:
    --------
    results : Dictionary containing:
        - 'cv_scores': List of AUCPR scores for each fold
        - 'mean_auprc': Mean AUCPR across folds
        - 'std_auprc': Standard deviation of AUCPR
        - 'cv_predictions': Combined predictions from all folds
        - 'cv_true_labels': Combined true labels from all folds
        - 'trained_model': Final Random Forest model trained on all data
    """
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST WITH 5-FOLD CROSS-VALIDATION")
    print("="*70)
    
    # Convert to numpy if needed
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    # Initialize Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Train and evaluate on each fold
    cv_scores = []
    cv_predictions = []
    cv_true_labels = []
    
    print("\nTraining on folds...")
    for fold_idx, (train_idx, val_idx) in enumerate(cv_indices):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        rf.fit(X_train, y_train)
        y_val_pred_proba = rf.predict_proba(X_val)[:, 1]
        
        auprc = average_precision_score(y_val, y_val_pred_proba)
        cv_scores.append(auprc)
        
        cv_predictions.extend(y_val_pred_proba)
        cv_true_labels.extend(y_val)
        
        print(f"Fold {fold_idx+1}/{len(cv_indices)}: AUCPR = {auprc:.4f}")
    
    mean_auprc = np.mean(cv_scores)
    std_auprc = np.std(cv_scores)
    
    print(f"\n{'='*70}")
    print(f"Cross-Validation Results:")
    print(f"Mean AUCPR: {mean_auprc:.4f} ± {std_auprc:.4f}")
    print(f"Individual fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"{'='*70}\n")
    
    # Train final model on all data
    print("Training final model on all data...")
    rf_final = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    rf_final.fit(X, y)
    print("Done.\n")
    
    return {
        'cv_scores': cv_scores,
        'mean_auprc': mean_auprc,
        'std_auprc': std_auprc,
        'cv_predictions': np.array(cv_predictions),
        'cv_true_labels': np.array(cv_true_labels),
        'trained_model': rf_final
    }


def plot_precision_recall_curve(cv_true_labels, cv_predictions, title='Precision-Recall Curve'):
    """
    Plot Precision-Recall curve from cross-validation predictions.
    
    Parameters:
    -----------
    cv_true_labels : True labels from all CV folds
    cv_predictions : Predicted probabilities from all CV folds
    title : Plot title
    """
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(cv_true_labels, cv_predictions)
    auprc = average_precision_score(cv_true_labels, cv_predictions)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2, 
             label=f'Random Forest (AUCPR={auprc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return auprc


def analyze_feature_importance(trained_model, feature_names, top_n=15):
    """
    Analyze and visualize feature importance from trained Random Forest.
    
    Parameters:
    -----------
    trained_model : Trained Random Forest model
    feature_names : Names of features
    top_n : Number of top features to display
        
    Returns:
    --------
    importance_df : pd.DataFrame
        DataFrame with features sorted by importance
    """
    feature_importances = trained_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Print top features
    print("\n" + "="*70)
    print(f"Top {min(10, len(importance_df))} Most Important Features:")
    print("="*70)
    print(importance_df.head(10).to_string(index=False))
    print()
    
    # Plot top N features
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['Importance'].values, 
             color='steelblue', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['Feature'].values, fontsize=10)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features - Random Forest', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df
