"""
compare_models.py

Model comparison between Random Forest and GNN.
Expects pre-computed features and datasets from project.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, f1_score, confusion_matrix,
)
from torch_geometric.loader import DataLoader


DEFAULT_THRESHOLD = -6.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_targets(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def _to_array(y):
    return y.values if isinstance(y, pd.Series) else np.asarray(y, dtype=float)


def _predict_both(rf_model, gnn_model, X, dataset, device='cpu'):
    rf_pred = rf_model.predict(X)

    gnn_model.eval()
    gnn_model = gnn_model.to(device)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    gnn_parts = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = gnn_model(data.x, data.edge_attr, data.edge_index, data.batch)
            gnn_parts.append(out.squeeze().cpu().numpy())
    gnn_pred = np.concatenate(gnn_parts)

    return {'Random Forest': rf_pred, 'GNN': gnn_pred}


def _results_table(metrics_by_model, columns):
    rows = [{'Model': name, **m} for name, m in metrics_by_model.items()]
    return pd.DataFrame(rows)[['Model'] + columns]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, threshold=DEFAULT_THRESHOLD):
    y_true, y_pred = _clean_targets(y_true, y_pred)
    if y_true.size == 0:
        return dict(r2=np.nan, mae=np.nan, mse=np.nan, rmse=np.nan,
                    accuracy=np.nan, n_samples=0)

    mse = mean_squared_error(y_true, y_pred)
    true_bin = (y_true >= threshold).astype(int)
    pred_bin = (y_pred >= threshold).astype(int)

    return dict(
        r2=r2_score(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        mse=mse,
        rmse=np.sqrt(mse),
        accuracy=accuracy_score(true_bin, pred_bin),
        n_samples=len(y_true),
    )


def calculate_binary_metrics(y_true, y_pred, threshold=DEFAULT_THRESHOLD):
    y_true, y_pred = _clean_targets(y_true, y_pred)
    if y_true.size == 0:
        return dict(accuracy=np.nan, precision=np.nan, recall=np.nan,
                    f1=np.nan, n_samples=0, confusion_matrix=None)

    true_bin = (y_true >= threshold).astype(int)
    pred_bin = (y_pred >= threshold).astype(int)

    return dict(
        accuracy=accuracy_score(true_bin, pred_bin),
        precision=precision_score(true_bin, pred_bin, zero_division=0),
        recall=recall_score(true_bin, pred_bin, zero_division=0),
        f1=f1_score(true_bin, pred_bin, zero_division=0),
        n_samples=len(y_true),
        confusion_matrix=confusion_matrix(true_bin, pred_bin),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_true_vs_pred(y_true, predictions, label_name, path, show):
    y_true = np.asarray(y_true, dtype=float)
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]

    plt.figure(figsize=(6, 6))
    all_vals = [y_true]
    for name, preds in predictions.items():
        p = np.asarray(preds, dtype=float)[mask]
        plt.scatter(y_true, p, alpha=0.6, s=18, label=name)
        all_vals.append(p)

    lo, hi = min(v.min() for v in all_vals), max(v.max() for v in all_vals)
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x')
    plt.xlabel(f'True {label_name}')
    plt.ylabel(f'Predicted {label_name}')
    plt.title(f'True vs Predicted ({label_name})')
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Comparison functions (ordered by call sequence in full_comparison_pipeline)
# ---------------------------------------------------------------------------

REGRESSION_COLS = ['r2', 'mae', 'mse', 'rmse', 'accuracy', 'n_samples']
REGRESSION_RENAME = dict(r2='R2', mae='MAE', mse='MSE', rmse='RMSE',
                         accuracy='Accuracy', n_samples='N_Samples')


def compare_models_on_pampa(rf_model, gnn_model, X_test, dataset_test, y_test,
                            threshold=DEFAULT_THRESHOLD, device='cpu',
                            plot=True, plot_path='true_vs_pred_pampa.png',
                            show_plot=True):
    print("=" * 80)
    print("COMPARISON 1: Models on samples with PAMPA labels")
    print("=" * 80)

    predictions = _predict_both(rf_model, gnn_model, X_test, dataset_test, device)
    y = _to_array(y_test)

    metrics = {name: calculate_metrics(y, p, threshold) for name, p in predictions.items()}
    results = _results_table(metrics, REGRESSION_COLS).rename(columns=REGRESSION_RENAME)

    print(results.to_string(index=False))
    print("-" * 80)

    if plot:
        _plot_true_vs_pred(y, predictions, 'PAMPA', plot_path, show_plot)

    return results


def compare_binary_classification(rf_model, gnn_model, X_test, dataset_test, y_test,
                                  threshold=DEFAULT_THRESHOLD, device='cpu'):
    print("\n" + "=" * 80)
    print(f"COMPARISON 2: Binary Classification (threshold = {threshold})")
    print("=" * 80)

    predictions = _predict_both(rf_model, gnn_model, X_test, dataset_test, device)
    y = _to_array(y_test)

    metrics = {name: calculate_binary_metrics(y, p, threshold) for name, p in predictions.items()}
    results = _results_table(
        metrics, ['accuracy', 'precision', 'recall', 'f1', 'n_samples']
    ).rename(columns=dict(accuracy='Accuracy', precision='Precision',
                          recall='Recall', f1='F1-Score', n_samples='N_Samples'))

    print(results.to_string(index=False))
    for name in predictions:
        print(f"\n{name} Confusion Matrix:")
        print(metrics[name]['confusion_matrix'])
    print("\n" + "-" * 80)

    return results


def compare_models_on_caco(rf_model, gnn_model, X_caco, dataset_caco, y_caco,
                           threshold=DEFAULT_THRESHOLD, device='cpu'):
    print("\n" + "=" * 80)
    print("COMPARISON 3: Models on samples with Caco-2 labels (trained on PAMPA)")
    print("=" * 80)

    predictions = _predict_both(rf_model, gnn_model, X_caco, dataset_caco, device)
    y = _to_array(y_caco)

    metrics = {name: calculate_metrics(y, p, threshold) for name, p in predictions.items()}
    results = _results_table(metrics, REGRESSION_COLS).rename(columns=REGRESSION_RENAME)

    print(results.to_string(index=False))
    print("\n" + "-" * 80)

    return results


# ---------------------------------------------------------------------------
# Pipeline (uses pre-computed data from project.py)
# ---------------------------------------------------------------------------

def full_comparison_pipeline(df, features_encoded, dataset,
                             rf_model, gnn_model, test_indices,
                             threshold=DEFAULT_THRESHOLD, device='cpu'):
    print("\n" + "=" * 80)
    print("FULL MODEL COMPARISON PIPELINE")
    print("=" * 80)

    results_dict = {}

    y_test = df['Class_Label'].iloc[test_indices]
    valid_mask = y_test.notna()
    valid_indices = test_indices[valid_mask.values]

    X_valid = features_encoded.iloc[valid_indices]
    ds_valid = dataset[valid_indices]
    y_valid = df['Class_Label'].iloc[valid_indices]

    results_dict['pampa'] = compare_models_on_pampa(
        rf_model, gnn_model, X_valid, ds_valid, y_valid, threshold, device
    )

    results_dict['binary'] = compare_binary_classification(
        rf_model, gnn_model, X_valid, ds_valid, y_valid, threshold, device
    )

    if 'Caco2' in df.columns:
        caco_indices = np.where(df['Caco2'].notna())[0]
        if len(caco_indices) > 0:
            results_dict['caco'] = compare_models_on_caco(
                rf_model, gnn_model,
                features_encoded.iloc[caco_indices],
                dataset[caco_indices],
                df['Caco2'].iloc[caco_indices],
                threshold, device,
            )
        else:
            print("\nNo samples with Caco-2 labels found")
    else:
        print("\nSkipping Caco-2 comparison (column not found)")

    return results_dict


def save_comparison_results(results_dict, output_path='model_comparison_results.csv'):
    with open(output_path, 'w') as f:
        for name, result_df in results_dict.items():
            f.write(f"\n# {name.upper()} COMPARISON\n")
            result_df.to_csv(f, index=False)
            f.write("\n")
    print(f"\nResults saved to: {output_path}")
