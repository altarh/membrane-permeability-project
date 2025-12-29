from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

def create_tanimoto_kfold_partition(X, y, groups, n_splits=5, random_state=0):
    """
    Create k-fold cross-validation indices with Tanimoto-based groups.
    
    Parameters:
    -----------
    X : Feature matrix
    y : Target labels
    groups : Group assignments from create_tanimoto_groups
    n_splits : Number of folds
    random_state : Random seed
        
    Returns:
    --------
    cv_indices : List of (train_indices, val_indices) for each fold
    """
    cv_split = StratifiedGroupKFold(
        n_splits=n_splits, 
        shuffle=True, 
        random_state=random_state
    )
    
    return list(cv_split.split(X, y, groups))
