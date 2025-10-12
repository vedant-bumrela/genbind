import os
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import selfies as sf

def mol_to_smiles(mol):
    return Chem.MolToSmiles(mol)

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def is_valid_smiles(smiles):
    return smiles_to_mol(smiles) is not None

def canonicalize(smiles):
    mol = smiles_to_mol(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None

def calc_qed(smiles):
    mol = smiles_to_mol(smiles)
    if mol:
        return QED.qed(mol)
    return 0

def calc_sa(smiles):
    try:
        from rdkit.Chem import rdMolDescriptors
        mol = smiles_to_mol(smiles)
        if mol:
            return rdMolDescriptors.CalcSyntheticAccessibilityScore(mol)
    except:
        return None

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
import os
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import selfies as sf

def mol_to_smiles(mol):
    return Chem.MolToSmiles(mol)

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def is_valid_smiles(smiles):
    return smiles_to_mol(smiles) is not None

def canonicalize(smiles):
    mol = smiles_to_mol(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None

def calc_qed(smiles):
    mol = smiles_to_mol(smiles)
    if mol:
        return QED.qed(mol)
    return 0

def calc_sa(smiles):
    try:
        from rdkit.Chem import rdMolDescriptors
        mol = smiles_to_mol(smiles)
        if mol:
            return rdMolDescriptors.CalcSyntheticAccessibilityScore(mol)
    except:
        return None

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
