import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def create_tanimoto_kfold_partition(X, y, groups, n_splits=5, random_state=0):

    y_binned = y.astype(int)    
    cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_folds = list(cv_splitter.split(
        X, 
        y_binned, 
        groups
    ))
    
    return cv_folds
