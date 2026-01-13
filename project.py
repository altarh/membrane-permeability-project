# -*- coding: utf-8 -*-
import torch

# NOT NEEDED - original homework data download
#!wget https://github.com/jertubiana/jertubiana.github.io/raw/master/misc/MLCB_2024_HW2_Data.zip
#!unzip /content/MLCB_2024_HW2_Data.zip
#!pip install matplotlib seaborn shap torch_geometric rdkit

"""# Package & data loading"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from data_loading import read_file_and_add_Class_Label
from mol_properties import get_features_and_morgan_fingerprints
from mol_to_GNN import molecule_to_graph
from mol_properties import create_tanimoto_groups
from k_fold_partition import create_tanimoto_kfold_partition
from random_forest import train_and_evaluate_random_forest_regressor, encode_categorical_features

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)  # Fallback to print

# table_first_round_molecules   =  pd.read_excel( '/content/MLCB_2024_HW2_Data/training_table.xlsx',skiprows=1,sheet_name='S1B')
table_first_round_molecules = read_file_and_add_Class_Label('CycPeptMPDB_Peptide_All.csv')

"""# Part 0: Parsing the data into Tabular Machine Learning format using the RDKIT package

In this section we:
- Convert each SMILE into a RDKIT Molecule class instance, which has many useful methods.
- Calculate various physio-chemical features for each molecule using RDKIT pre-provided functions. Example of features include number of atoms and bonds of each type, number of hydrogen bond donor and acceptors, etc.

"""

"""# Part I: Exploratory Data Analysis

1.	Display, as a scatter plot, the distribution of the number of atoms and covalent bonds per molecule. Color by the measured  class. Is there a relationship between molecule size and activity?
"""



"""2.	For each of the following five features:
*   Octanol-water partition coefficient
*   Molecular weight
*   Number of hydrogen bond donors
*   Number of hydrogen bond acceptors
*   Number of aromatic rings


Display the histogram of the feature value, for each of the two classes. Which feature(s) have the strongest discriminative power?

"""



"""# Part II: Data Partition

Our training data is NOT independently distributed because many known antimicrobial molecules were obtained by small modifications of previously known molecules. Indeed, similar molecules often have similar chemical activity. We thus need to partition the data with grouped splits.


Specifically, our goal is to build a model that can generalize well to unseen molecules. To this end, we want to make sure that our molecules from the train, validation and test are “dissimilar enough”.


We will use the Tanimoto similarity, a custom metric for calculating similarity between molecules (from 0 = dissimilar to 1 = maximally similar).


"""

# We are assuming all molecules are successfully parsed into RDKIT Molecule objects
first_round_molecules_rdkit, first_round_molecules_morgan_fingerprints = get_features_and_morgan_fingerprints(table_first_round_molecules)

# Create Tanimoto groups (uses median cutoff adaptively)
print(f"Calculating tanimoto similarities...")
n_groups, groups = create_tanimoto_groups(first_round_molecules_morgan_fingerprints)
print(f"Created {n_groups} groups using tanimoto partition")

print("Splitting data")

# Define the feature columns we want to use from table_first_round_molecules
FEATURE_COLUMNS = [
    'Monomer_Length', 'Monomer_Length_in_Main_Chain', 'Molecule_Shape',
    'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
    'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt',
    'NumValenceElectrons', 'NumRadicalElectrons',
    'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
    'BalabanJ', 'BertzCT',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
    'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
    'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
    'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
    'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
    'PEOE_VSA8', 'PEOE_VSA9',
    'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
    'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',
    'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
    'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',
    'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA',
    'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
    'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
    'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
    'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9',
    'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
    'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
    'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
    'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount',
    'MolLogP', 'MolMR',
    'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N',
    'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S',
    'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O',
    'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
    'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
    'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo',
    'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
    'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
    'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
    'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
    'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
    'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
    'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',
    'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
    'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
    'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
    'fr_unbrch_alkane', 'fr_urea',
    'PC1', 'PC2', 'CHCl3_3DPSA', 'H2O_3DPSA'
]

# Extract features from table_first_round_molecules
features_from_table = table_first_round_molecules[FEATURE_COLUMNS]

print(f"Extracted {len(FEATURE_COLUMNS)} features from table_first_round_molecules")
print(f"Original feature matrix shape: {features_from_table.shape}")

# Encode categorical features
features_from_table, categorical_encoders = encode_categorical_features(
    features_from_table, 
    fit=True
)

print(f"\nFinal encoded feature matrix shape: {features_from_table.shape}")
print(f"All columns numeric: {features_from_table.select_dtypes(include=[np.number]).shape[1] == features_from_table.shape[1]}")

# print("\n=== FEATURE DIAGNOSTICS ===")
# for col in features_from_table.columns:
#     col_data = features_from_table[col]
#     has_inf = np.isinf(col_data).any()
#     has_nan = col_data.isna().any()
#     col_max = col_data.max()
#     col_min = col_data.min()
    
#     if has_inf or has_nan or col_max > 1e15 or col_min < -1e15:
#         print(f"{col}: inf={has_inf}, nan={has_nan}, range=[{col_min:.2e}, {col_max:.2e}]")


print("\nApplying log transformation to heavily skewed features...")

if 'Ipc' in features_from_table.columns:
    # Check the range
    ipc_min = features_from_table['Ipc'].min()
    ipc_max = features_from_table['Ipc'].max()
    print(f"Ipc range: [{ipc_min:.2e}, {ipc_max:.2e}]")
    
    # Log transform (add small constant to avoid log(0) log(1+x))
    features_from_table['Ipc'] = np.log1p(features_from_table['Ipc'])
    
    new_min = features_from_table['Ipc'].min()
    new_max = features_from_table['Ipc'].max()
    print(f"After log transform: [{new_min:.2f}, {new_max:.2f}]")

# Create train/test split (80/20) respecting Tanimoto groups
train_test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
[(train_and_val_index, test_index)] = train_test_split.split(
    features_from_table,
    table_first_round_molecules['Class_Label'],
    groups
)

# 1. Identify which rows in the WHOLE table have missing PAMPA (label)
missing_pampa_mask = table_first_round_molecules['Class_Label'].isna()
indices_missing_pampa = np.where(missing_pampa_mask)[0]

# 2. Find which of these ended up in the train_and_val_index
rows_to_move = np.intersect1d(train_and_val_index, indices_missing_pampa)

if len(rows_to_move) > 0:
    print(f"\nDeleting {len(rows_to_move)} rows with missing PAMPA from Train and eval.")
    
    # 3. Remove them from train_and_val_index
    train_and_val_index = np.setdiff1d(train_and_val_index, rows_to_move)
    
    # 4. Sort indices to keep things tidy
    train_and_val_index.sort()
    test_index.sort()  # allowing rows without label for the test dataset

# Use this train/val split for training the GNN.
train_val_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
[(train_index, val_index)] = train_val_split.split(
    features_from_table.iloc[train_and_val_index],
    table_first_round_molecules['Class_Label'].iloc[train_and_val_index],
    groups[train_and_val_index]
)

train_index = train_and_val_index[train_index]
val_index = train_and_val_index[val_index]

# Create 5-fold cross-validation partition for Random Forest
X_train_subset = features_from_table.iloc[train_and_val_index]
y_train_subset = table_first_round_molecules['Class_Label'].iloc[train_and_val_index]
groups_train_subset = groups[train_and_val_index]

train_cv_folds = create_tanimoto_kfold_partition(
    X=X_train_subset,
    y=y_train_subset,
    groups=groups_train_subset,
    n_splits=5,
    random_state=0
)

# Train and evaluate Random Forest with 5-fold CV
print("\n" + "="*70)
print("RANDOM FOREST REGRESSION")
print("="*70)

missing_pampa_mask = table_first_round_molecules['Class_Label'].isna()
X_no_pampa = features_from_table[missing_pampa_mask]

best_random_forest_model, pampa_predictions = train_and_evaluate_random_forest_regressor(
    X=X_train_subset,
    y=y_train_subset,
    cv_indices=train_cv_folds,
    X_prediction=X_no_pampa
)


"""# Part IV: Post-hoc explanations of the tree ensemble model

7.	Using sklearn, calculate the feature importance using the Permutation Feature Importance metric over the train and test set. Which features are the most informative?
"""



"""8.	Using sklearn, display the Partial Dependence Plots (PDP) for the 10 most important features, ordered by importance. Comment on the findings."""



"""9.	Predict the activity of Halicin (SMILE representation: Nc1nnc(Sc2ncc([N+](=O)[O-])s2)s1),  Amoxicilin (SMILE representation: CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O) and calculate the Shapley values associated to these prediction using shap. Are they predicted to be active and why?"""



"""# Part V: Training a Graph Neural Network

To train a GNN, we first create a graph-based representation of each molecule suitable for processing it with GNNs. Here, each node correspond to one heavy atom (excluding hydrogen atoms), and each edge to a covalent bond.

We have per-node features (atom type, etc.) and per-edge features (covalent bond type).  These can be summarized as:

- A node feature matrix $F$, of size Number of atoms by Number of feature nodes.
- An edge feature matrix $F_E$, of size Number of edges by Number of feature nodes.
- An edge indices matrix $E$, of size Number of edges by 2 ($E_{k1},E_{k2}$ are the indices of the incoming and outgoing node for edge k).
"""
"""# Inmplementing a Graph Convolution Network using PyTorch Geometric

This implementation is adapted from https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=0gZ-l0npPIca

I recommend to go over this notebook first.
"""


"""
Step 1: Convert data into PyTorch Geometric Dataset
"""

from CNN import CustomGraphDataset
from torch_geometric.loader import DataLoader

print("\n" + "="*70)
print("GRAPH CONVOLUTION NETWORK")
print("="*70)

print(f"\nBuilding GNN\n")

first_round_molecules_graph = [molecule_to_graph(mol) for mol in first_round_molecules_rdkit]

# Create the dataset
dataset = CustomGraphDataset(first_round_molecules_graph, table_first_round_molecules['Class_Label'])

train_dataset = dataset[train_index]
validation_dataset = dataset[val_index]
test_dataset = dataset[test_index]
test_only_with_labels_indices = [
    i for i, d in enumerate(test_dataset)
    if not torch.isnan(d.y)
]
evaluation_dataset = test_dataset[test_only_with_labels_indices]

# Report proportion of positive labels (with threshold -6.0)
full_positive_prop = (table_first_round_molecules['Class_Label'] >= -6).mean()
train_positive_prop = (table_first_round_molecules['Class_Label'].iloc[train_index] >= -6).mean()
val_positive_prop = (table_first_round_molecules['Class_Label'].iloc[val_index] >= -6).mean()
test_positive_prop = (table_first_round_molecules['Class_Label'].iloc[test_index] >= -6).mean()

print(f"Full dataset positive proportion: {full_positive_prop:.4f}")
print(f"Train positive proportion: {train_positive_prop:.4f}")
print(f"Validation positive proportion: {val_positive_prop:.4f}")
print(f"Test positive proportion: {test_positive_prop:.4f}")
print()

# Create a DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
evaluation_loader = DataLoader(evaluation_dataset, batch_size=64, shuffle=False)

"""
Step 2: Define a simple Graph Convolution Network
Note that here, we don't use the edge features at all
"""
from CNN import GCN

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GCN(dataset.num_node_features, hidden_channels=64)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = torch.nn.L1Loss()

"""# Step 3: Train the model!

*!!! Make sure that you selected the GPU runtype !!* (Runtime -> Change runtime type)
"""

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_attr, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = loss_function(out.squeeze(), data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()
    total_mae = 0
    total_mse = 0
    correct_preds = 0
    total_samples = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_attr, data.edge_index, data.batch)
        mae = torch.nn.functional.l1_loss(out.squeeze(), data.y.float(), reduction='sum')  # computes the sum of absolute errors
        mse = torch.nn.functional.mse_loss(out.squeeze(), data.y.float(), reduction='sum')  # computes the sum of squared errors
        total_mae += mae.item()
        total_mse += mse.item()
        total_samples += data.y.size(0)  # Count the number of samples.

        true_binary = (data.y >= -6.0)
        pred_binary = (out.squeeze() >= -6.0)
        correct_preds += (true_binary == pred_binary).sum().item()

    return (total_mae / total_samples, total_mse / total_samples, correct_preds / total_samples)  # calcualte MAE, MSE and accuracy


import copy  # Needed for deepcopy

# Configuration
patience = 5
best_val_mae = np.inf
patience_counter = 0
best_model_weights = None

for epoch in range(1, 100):  # Increased range to allow early stopping to work
    train()
    train_results = test(train_loader)
    val_results = test(validation_loader)
    test_results = test(evaluation_loader)
    print(f'Epoch: {epoch:03d}:')
    print(f'\t\tTrain MAE: {train_results[0]:.4f}, Val MAE: {val_results[0]:.4f}, Test MAE: {test_results[0]:.4f}')
    print(f'\t\tTrain MSE: {train_results[1]:.4f}, Val MSE: {val_results[1]:.4f}, Test MSE: {test_results[1]:.4f}')
    print(f'\t\tTrain Acc: {train_results[2]:.4f}, Val Acc: {val_results[2]:.4f}, Test Acc: {test_results[2]:.4f}')

    train_mae = train_results[0]
    val_mae = val_results[0]
    test_mae = test_results[0]

    # --- Early Stopping Logic ---
    if val_mae < best_val_mae:
        # 1. Improvement found: Update best score
        best_val_mae = val_mae
        # 2. Reset patience counter
        patience_counter = 0
        # 3. Save a DEEP COPY of the weights (not a pointer/reference)
        best_model_weights = copy.deepcopy(model.state_dict())
    else:
        # No improvement
        patience_counter += 1
        print(f"--> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("--> Early stopping triggered!")
            # 4. Restore the best weights found
            model.load_state_dict(best_model_weights)
            break

# Ensure the model has the best weights if the loop finishes without triggering early stop
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    print(f"Loaded best model weights with Val MAE: {best_val_mae:.4f}")

def plot_confusion_matrix(loader):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(next(model.parameters()).device)
            out = model(data.x, data.edge_attr, data.edge_index, data.batch)  # Perform a single forward pass.
            true_binary = (data.y >= -6.0)
            pred_binary = (out.squeeze() >= -6.0)

            all_preds.append(pred_binary.cpu())
            all_labels.append(true_binary.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()

print("\nFinal evaluation on evaluation set:")
final_test_results = test(evaluation_loader)
print(f'\t\tTest MAE: {final_test_results[0]:.4f}, Test MSE: {final_test_results[1]:.4f}, Test Acc: {final_test_results[2]:.4f}')
plot_confusion_matrix(evaluation_loader)
