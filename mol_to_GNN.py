from rdkit import Chem
import torch


def molecule_to_graph(mol):
    """
    Convert a RDKIT Molecule object into a graph-based representation.

    Parameters:
    smiles (str): SMILES string of the molecule.

    Returns:
    - node_features: List of feature vectors for each atom.
    - edge_index: List of [source, target] pairs representing bonds.
    - edge_features: List of feature vectors for each bond.
    """

    atom_types = ['C','O','N','S','F','Cl','Br','P']
    hybridization_types = list(Chem.rdchem.HybridizationType.values.values())


    def ont_hot_encoding(symbol,symbols):
      nsymbols = len(symbols)
      idx = symbols.index(symbol) if symbol in symbols else nsymbols
      one_hot = torch.zeros(nsymbols+1)
      one_hot[idx] = 1
      return one_hot


    # Define node feature extraction
    def get_atom_features(atom):
        """
        Extract features for a single atom.
        """
        all_features =  torch.concatenate([
            ont_hot_encoding(atom.GetSymbol() , atom_types  ), # Heavy atom type
            ont_hot_encoding(atom.GetHybridization() , hybridization_types  ), # Electronic orbital type (Sp3, Sp2...)
            torch.tensor([atom.GetDegree()]),                # Number of bonds
            torch.tensor([atom.GetFormalCharge()]),          # Formal charge
            torch.tensor([atom.GetTotalNumHs()]),            # Number of hydrogen atoms attached to it.
            torch.tensor([atom.IsInRing()])      # Is the atom in a ring? (1 or 0)

        ]).to(torch.float32)
        return all_features


    # Define edge feature extraction
    def get_bond_features(bond):
        """
        Extract features for a single bond.
        Returns a list of features: [is_single, is_double, is_triple, is_aromatic, is_conjugated]
        """
        bond_type = bond.GetBondType()
        return [
            1 if bond_type == Chem.BondType.SINGLE else 0,
            1 if bond_type == Chem.BondType.DOUBLE else 0,
            1 if bond_type == Chem.BondType.TRIPLE else 0,
            1 if bond_type == Chem.BondType.AROMATIC else 0,
            1 if bond.GetIsConjugated() else 0
        ]


    # Initialize lists for graph components
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]  # Extract node features for each atom

    edge_features = []
    edge_index = []

    # Extract edge features and connectivity
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        # Add edges in both directions (undirected graph)
        edge_index.append([start_idx, end_idx])
        edge_index.append([end_idx, start_idx])
        # Add bond features for both directions
        bond_feats = get_bond_features(bond)
        edge_features.append(bond_feats)
        edge_features.append(bond_feats)

    # Convert to torch tensors
    node_features = torch.stack(node_features,axis=0)
    if len(edge_features)==0:
      print(mol,'This molecule has no edges!')
      edge_features = torch.zeros([0, 5],dtype=torch.float32)
      edge_index = torch.zeros([0, 2],dtype=torch.int64)
    else:
      edge_features = torch.tensor(edge_features,dtype=torch.float32)
      edge_index = torch.tensor( edge_index, dtype=torch.int64)  # Shape: [num_edges,2]
    return node_features,edge_features,edge_index
