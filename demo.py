from rdkit import Chem
from rdkit.Chem import Draw

# Define the SMILES string
smiles = "N12NN1O2"

# Convert SMILES to molecule
mol = Chem.MolFromSmiles(smiles)

# Generate the image
img = Draw.MolToImage(mol, size=(300, 300))
img.show()