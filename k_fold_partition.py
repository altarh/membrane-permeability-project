import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def create_tanimoto_kfold_partition(X, y, groups, n_splits=5, random_state=0):
    """
    1. Splits data 80/20 (Train/Test) using StratifiedGroupKFold (preserving groups/class balance).
    2. Performs 'n_splits' Cross-Validation on the 80% Train set.
    """
    # use .astype(int) as requested to create discrete bins from continuous targets
    y_binned = y.astype(int)
    
    # Outer Split (80% Train, 20% Test)
    # We use n_splits=5 and take the first fold as the split.
    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(outer_splitter.split(X, y_binned, groups))
    
    # Isolate the training data (subset) to feed into the inner CV
    X_train = X.iloc[train_idx]
    y_train_binned = y_binned.iloc[train_idx]
    
    groups_train_subset = groups[train_idx]

    # Generate the CV folds for the training subset
    inner_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_cv_folds = list(inner_cv.split(
        X_train, 
        y_train_binned, 
        groups_train_subset
    ))
    
    return train_idx, test_idx, train_cv_folds
