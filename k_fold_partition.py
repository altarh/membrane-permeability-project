from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np

def create_tanimoto_kfold_partition(X, y, groups, n_splits=5, random_state=0):
    n_samples = len(y)
    
    # Calculate safe number of bins
    # Each bin needs at least n_splits samples to split into n_splits folds
    min_samples_per_bin = n_splits * 2  # Safety factor
    n_bins = max(3, min(10, n_samples // min_samples_per_bin))
    
    print(f"\nPAMPA range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Creating {n_bins} bins for {n_samples} samples")
    
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    
    # Report bin distribution
    unique, counts = np.unique(y_binned, return_counts=True)
    # print(f"Bin distribution: {dict(zip(unique, counts))}")
    print(f"Min bin size: {counts.min()} (needs ≥ {n_splits})")
    
    # Create stratified folds
    cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_folds = list(cv_splitter.split(X, y_binned, groups))
    
    # Print fold statistics
    print(f"\nCreated {n_splits} folds:")
    for i, (train_idx, val_idx) in enumerate(cv_folds):
        val_y = y.iloc[val_idx]
        print(f"  Fold {i+1}: {len(val_idx)} val samples, "
              f"PAMPA range: [{val_y.min():.2f}, {val_y.max():.2f}]")
    
    return cv_folds
