"""
compare_models.py

Comprehensive model comparison between Random Forest and GNN models.
Compares performance on:
1. Samples with PAMPA labels
2. Samples with Caco labels when trained on PAMPA  
3. Binary classification with threshold -6
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.loader import DataLoader

# Import project modules
from data_loading import read_file_and_add_Class_Label
from mol_properties import get_features_and_morgan_fingerprints, create_tanimoto_groups
from mol_to_GNN import molecule_to_graph
from CNN import CustomGraphDataset, GCN
from feature_columns import FEATURE_COLUMNS
from random_forest import encode_categorical_features


def calculate_metrics(y_true, y_pred, threshold=-6.0):
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            'r2': np.nan,
            'mae': np.nan,
            'mse': np.nan,
            'rmse': np.nan,
            'accuracy': np.nan,
            'n_samples': 0
        }

    # Regression metrics
    r2 = r2_score(y_true_clean, y_pred_clean)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)

    # Classification metrics (binary with threshold)
    y_true_binary = (y_true_clean >= threshold).astype(int)
    y_pred_binary = (y_pred_clean >= threshold).astype(int)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)

    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'accuracy': accuracy,
        'n_samples': len(y_true_clean)
    }


def predict_rf(model, X):
    return model.predict(X)


def predict_gnn(model, dataset, device='cpu'):
    model.eval()
    model = model.to(device)

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    predictions = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_attr, data.edge_index, data.batch)
            predictions.append(out.squeeze().cpu().numpy())

    return np.concatenate(predictions)


def compare_models_on_pampa(rf_model, gnn_model, X_test, dataset_test, y_test, 
                            threshold=-6.0, device='cpu'):
    print("\n" + "="*80)
    print("COMPARISON 1: Models on samples with PAMPA labels")
    print("="*80)

    # Get predictions
    rf_pred = predict_rf(rf_model, X_test)
    gnn_pred = predict_gnn(gnn_model, dataset_test, device)

    # Calculate metrics
    rf_metrics = calculate_metrics(y_test.values if isinstance(y_test, pd.Series) else y_test, 
                                   rf_pred, threshold)
    gnn_metrics = calculate_metrics(y_test.values if isinstance(y_test, pd.Series) else y_test, 
                                    gnn_pred, threshold)

    # Create comparison dataframe
    results = pd.DataFrame({
        'Model': ['Random Forest', 'GNN'],
        'R2': [rf_metrics['r2'], gnn_metrics['r2']],
        'MAE': [rf_metrics['mae'], gnn_metrics['mae']],
        'MSE': [rf_metrics['mse'], gnn_metrics['mse']],
        'RMSE': [rf_metrics['rmse'], gnn_metrics['rmse']],
        'Accuracy': [rf_metrics['accuracy'], gnn_metrics['accuracy']],
        'N_Samples': [rf_metrics['n_samples'], gnn_metrics['n_samples']]
    })

    print("\nResults:")
    print(results.to_string(index=False))
    print("\n" + "-"*80)

    return results


def compare_models_on_caco(rf_model, gnn_model, X_caco, dataset_caco, y_caco, threshold=-6.0, device='cpu'):
    print("\n" + "="*80)
    print("COMPARISON 2: Models on samples with Caco labels (trained on PAMPA)")
    print("="*80)

    # Get predictions
    rf_pred = predict_rf(rf_model, X_caco)
    gnn_pred = predict_gnn(gnn_model, dataset_caco, device)

    # Calculate metrics
    rf_metrics = calculate_metrics(y_caco.values if isinstance(y_caco, pd.Series) else y_caco,
                                   rf_pred, threshold)
    gnn_metrics = calculate_metrics(y_caco.values if isinstance(y_caco, pd.Series) else y_caco,
                                    gnn_pred, threshold)

    # Create comparison dataframe
    results = pd.DataFrame({
        'Model': ['Random Forest', 'GNN'],
        'R2': [rf_metrics['r2'], gnn_metrics['r2']],
        'MAE': [rf_metrics['mae'], gnn_metrics['mae']],
        'MSE': [rf_metrics['mse'], gnn_metrics['mse']],
        'RMSE': [rf_metrics['rmse'], gnn_metrics['rmse']],
        'Accuracy': [rf_metrics['accuracy'], gnn_metrics['accuracy']],
        'N_Samples': [rf_metrics['n_samples'], gnn_metrics['n_samples']]
    })

    print("\nResults:")
    print(results.to_string(index=False))
    print("\n" + "-"*80)

    return results


def compare_binary_classification(rf_model, gnn_model, X_test, dataset_test, y_test,
                                  threshold=-6.0, device='cpu'):
    print("\n" + "="*80)
    print(f"COMPARISON 3: Binary Classification (threshold = {threshold})")
    print("="*80)

    # Get predictions
    rf_pred = predict_rf(rf_model, X_test)
    gnn_pred = predict_gnn(gnn_model, dataset_test, device)

    # Convert to binary
    y_true = y_test.values if isinstance(y_test, pd.Series) else y_test
    mask = ~np.isnan(y_true)
    y_true_clean = y_true[mask]
    rf_pred_clean = rf_pred[mask]
    gnn_pred_clean = gnn_pred[mask]

    y_true_binary = (y_true_clean >= threshold).astype(int)
    rf_pred_binary = (rf_pred_clean >= threshold).astype(int)
    gnn_pred_binary = (gnn_pred_clean >= threshold).astype(int)

    # Calculate detailed metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    rf_accuracy = accuracy_score(y_true_binary, rf_pred_binary)
    rf_precision = precision_score(y_true_binary, rf_pred_binary, zero_division=0)
    rf_recall = recall_score(y_true_binary, rf_pred_binary, zero_division=0)
    rf_f1 = f1_score(y_true_binary, rf_pred_binary, zero_division=0)

    gnn_accuracy = accuracy_score(y_true_binary, gnn_pred_binary)
    gnn_precision = precision_score(y_true_binary, gnn_pred_binary, zero_division=0)
    gnn_recall = recall_score(y_true_binary, gnn_pred_binary, zero_division=0)
    gnn_f1 = f1_score(y_true_binary, gnn_pred_binary, zero_division=0)

    # Create comparison dataframe
    results = pd.DataFrame({
        'Model': ['Random Forest', 'GNN'],
        'Accuracy': [rf_accuracy, gnn_accuracy],
        'Precision': [rf_precision, gnn_precision],
        'Recall': [rf_recall, gnn_recall],
        'F1-Score': [rf_f1, gnn_f1],
        'N_Samples': [len(y_true_clean), len(y_true_clean)]
    })

    print("\nBinary Classification Metrics:")
    print(results.to_string(index=False))

    # Print confusion matrices
    print(f"\nRandom Forest Confusion Matrix:")
    print(confusion_matrix(y_true_binary, rf_pred_binary))
    print(f"\nGNN Confusion Matrix:")
    print(confusion_matrix(y_true_binary, gnn_pred_binary))
    print("\n" + "-"*80)

    return results


def full_comparison_pipeline(csv_path='CycPeptMPDB_Peptide_All.csv',
                             rf_model=None, gnn_model=None,
                             train_indices=None, test_indices=None,
                             threshold=-6.0, device='cpu'):
    print("\n" + "="*80)
    print("FULL MODEL COMPARISON PIPELINE")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = read_file_and_add_Class_Label(csv_path)

    # Check for Caco column
    has_caco = 'Caco' in df.columns

    if has_caco:
        print(f"Found Caco column: {df['Caco'].notna().sum()} non-null values")
    else:
        print("Warning: No Caco column found in dataset")

    # Prepare features
    molecules_rdkit, molecules_morgan_fp = get_features_and_morgan_fingerprints(df)
    features_from_table = df[FEATURE_COLUMNS]
    features_encoded, _ = encode_categorical_features(features_from_table, fit=True)

    # Apply log transform to Ipc if present
    if 'Ipc' in features_encoded.columns:
        features_encoded['Ipc'] = np.log1p(features_encoded['Ipc'])

    # Prepare graph data
    molecules_graph = [molecule_to_graph(mol) for mol in molecules_rdkit]
    dataset = CustomGraphDataset(molecules_graph, df['Class_Label'])

    results_dict = {}

    if test_indices is not None and rf_model is not None and gnn_model is not None:
        # 1. Compare on PAMPA labels
        X_test = features_encoded.iloc[test_indices]
        dataset_test = dataset[test_indices]
        y_test = df['Class_Label'].iloc[test_indices]

        # Filter out NaN labels for evaluation
        valid_label_mask = y_test.notna()
        valid_indices = test_indices[valid_label_mask.values]

        X_test_valid = features_encoded.iloc[valid_indices]
        dataset_test_valid = dataset[valid_indices]
        y_test_valid = df['Class_Label'].iloc[valid_indices]

        results_dict['pampa'] = compare_models_on_pampa(
            rf_model, gnn_model, X_test_valid, dataset_test_valid, 
            y_test_valid, threshold, device
        )

        # 3. Binary classification comparison
        results_dict['binary'] = compare_binary_classification(
            rf_model, gnn_model, X_test_valid, dataset_test_valid,
            y_test_valid, threshold, device
        )

        # 2. Compare on Caco labels if available
        if has_caco:
            caco_mask = df['Caco'].notna()
            caco_indices = np.where(caco_mask)[0]

            if len(caco_indices) > 0:
                X_caco = features_encoded.iloc[caco_indices]
                dataset_caco = dataset[caco_indices]
                y_caco = df['Caco'].iloc[caco_indices]

                results_dict['caco'] = compare_models_on_caco(
                    rf_model, gnn_model, X_caco, dataset_caco,
                    y_caco, threshold, device
                )
            else:
                print("\nWarning: No samples with Caco labels found")
        else:
            print("\nSkipping Caco comparison (column not found)")

    return results_dict


def save_comparison_results(results_dict, output_path='model_comparison_results.csv'):
    with open(output_path, 'w') as f:
        for comparison_name, df in results_dict.items():
            f.write(f"\n# {comparison_name.upper()} COMPARISON\n")
            df.to_csv(f, index=False)
            f.write("\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    print("compare_models.py - Model Comparison Module")
    print("Import this module and use the comparison functions with your trained models.")
