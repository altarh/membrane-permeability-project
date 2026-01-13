from rdkit import Chem
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_array
from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf, DataStructs
from rdkit.Chem import rdFingerprintGenerator

import numpy as np
import pandas as pd

def extract_chemical_features(list_molecules):
  def num_atoms(mol):
    atom_counts = {
        'Total': 0,
        'Carbon': 0,
        'Nitrogen': 0,
        'Oxygen': 0,
        'Sulfur': 0,
        'Fluorine': 0,
        'Phosphate':0,
        'Chlorine':0,
        'Other':0,
    }
    for atom in mol.GetAtoms():
      atom_counts['Total'] +=1
      atom_type = atom.GetSymbol()
      if atom_type == 'C':
          atom_counts['Carbon'] += 1
      elif atom_type == 'N':
          atom_counts['Nitrogen'] += 1
      elif atom_type == 'O':
          atom_counts['Oxygen'] += 1
      elif atom_type == 'S':
          atom_counts['Sulfur'] += 1
      elif atom_type == 'F':
          atom_counts['Fluorine'] += 1
      elif atom_type == 'P':
          atom_counts['Phosphate'] += 1
      elif atom_type == 'Cl':
          atom_counts['Chlorine'] += 1
      else:
          atom_counts['Other'] += 1
    return atom_counts


  def num_bonds(mol):
    bond_counts = {
        'Total': 0,
        'Single': 0,
        'Double': 0,
        'Triple': 0,
        'Aromatic': 0,
    }
    # Iterate through all bonds in the molecule
    for bond in mol.GetBonds():
      bond_counts['Total'] +=1
      bond_type = bond.GetBondType()
      if bond_type == Chem.BondType.SINGLE:
          bond_counts['Single'] += 1
      elif bond_type == Chem.BondType.DOUBLE:
          bond_counts['Double'] += 1
      elif bond_type == Chem.BondType.TRIPLE:
          bond_counts['Triple'] += 1
      elif bond_type == Chem.BondType.AROMATIC:
          bond_counts['Aromatic'] += 1
    return bond_counts


  """
  Extract chemical features for a list of molecules.

  Parameters:
  list_molecules (list): List of small molecules (RDKIT class instances).
  Returns:
  pandas.DataFrame: Table of descriptors.
  """

  # List to store results
  results = []

  # Define descriptors to compute (MOE-like)
  descriptors = {
      'MolWt': Descriptors.MolWt,  # Molecular weight
      'LogP': Crippen.MolLogP,     # Octanol-water partition coefficient
      'TPSA': MolSurf.TPSA,        # Topological polar surface area
      'HBD': Lipinski.NumHDonors,  # Number of hydrogen bond donors
      'HBA': Lipinski.NumHAcceptors,  # Number of hydrogen bond acceptors
      'RotBonds': Lipinski.NumRotatableBonds,  # Number of rotatable bonds
      'NumAromRings': Descriptors.NumAromaticRings,  # Number of aromatic rings
      'NumHeteroatoms': Descriptors.NumHeteroatoms,  # Number of heteroatoms
      'FractionSP3': Descriptors.FractionCSP3,  # Fraction of sp3 carbons
      'MolarRefractivity': Crippen.MolMR,  # Molar refractivity (MOE's mr)
      'NumAtoms': num_atoms,
      'NumBonds': num_bonds,
  }

  all_mol_features = []
  for molecule in list_molecules:
    mol_features = {}
    for desc_name, desc_func in descriptors.items():
      results = desc_func(molecule)
      if isinstance(results,dict):
        for key, value in results.items():
          mol_features[f'{desc_name}{key}'] = value
      else:
        mol_features[desc_name] = results
    all_mol_features.append(mol_features)
  return pd.DataFrame(all_mol_features)


def calculate_morgan_fingerprints(list_molecules, radius=2, fpSize=1024):
  '''
  A count-based representation of small molecules (e.g., how many occurence of O with double bond to C, etc.)
  '''
  mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=fpSize)
  fingerprints = [mfpgen.GetFingerprint(mol) for mol in list_molecules]
  return fingerprints


def calculate_tanimoto_similarity(fp1,fp2):
    """
    Calculate the Tanimoto similarity between two molecules given their Morgan fingerprints.

    Parameters:
    fp1 (str): Morgan Fingerprints of the first molecule.
    fp2 (str): Morgan Fingerprints of the second molecule.

    Returns:
    float: Tanimoto similarity score (0 to 1), or None if invalid SMILES.
    """
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_features_and_morgan_fingerprints(molecules_table):
    # SMILES is a string-based representation of molecules.
    molecules_smiles = molecules_table['SMILES']

    # We first turn each molecule into an instance of the RDKIT molecule.
    molecules_rdkit = [Chem.MolFromSmiles(smiles) for smiles in molecules_smiles]

    # Discard examples for which conversion failed.
    molecules_success = [i for i in range(len(molecules_rdkit)) if molecules_rdkit[i] is not None]

    print(f'Molecule construction suceeded for {len(molecules_success)}/{len(molecules_rdkit)} examples in the first round dataset')
    if len(molecules_success) < len(molecules_rdkit):
        print(f'Molecule construction failed for { len(molecules_rdkit)-len(molecules_success) }/{len(molecules_rdkit)} examples in the first round dataset')

    molecules_table = molecules_table.iloc[molecules_success].reset_index()
    molecules_rdkit = [mol for mol in molecules_rdkit if mol is not None]

    # print('Extracting chemical features/descriptors for each molecule...')
    # molecules_features = extract_chemical_features(molecules_rdkit)
    # print('Done.')

    print('Calculating Morgan fingerprints...')
    molecules_morgan_fingerprints = calculate_morgan_fingerprints(molecules_rdkit)
    print('Done.')

    return molecules_rdkit, molecules_morgan_fingerprints

def calculate_tanimoto_similarities(morgan_fingerprints):
    """
    Calculate Tanimoto similarity between all pairs of molecules given their Morgan fingerprints.
    """
    # Calculate pairwise Tanimoto similarities
    n_mols = len(morgan_fingerprints)
    tanimoto_matrix = np.zeros((n_mols, n_mols))
    
    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            sim = calculate_tanimoto_similarity(morgan_fingerprints[i], morgan_fingerprints[j])
            tanimoto_matrix[i, j] = sim
            tanimoto_matrix[j, i] = sim

    return tanimoto_matrix

def create_tanimoto_groups(morgan_fingerprints):
    """
    Create Tanimoto-based groups from pre-calculated Morgan fingerprints.
    cutoff: The similarity threshold used to determine group membership. Using the median of the maximum similarities for each molecule.
    
    Parameters:
    -----------
    morgan_fingerprints : List of Morgan fingerprints (from get_features_and_morgan_fingerprints)
        
    Returns:
    --------
    groups : Group assignments for each molecule
    """
    tanimoto_matrix = calculate_tanimoto_similarities(morgan_fingerprints)

    # Create binary similarity graph
    np.fill_diagonal(tanimoto_matrix, 0)
    max_similarities = tanimoto_matrix.max(axis=1)
    cutoff = np.median(max_similarities)
    binary_similarity_graph = tanimoto_matrix >= cutoff
    
    # Find connected components (molecules connected by similarity >= cutoff)
    n_groups, groups = connected_components(
        csgraph=csr_array(binary_similarity_graph), 
        directed=False, 
        return_labels=True
    )
    
    return n_groups, groups
