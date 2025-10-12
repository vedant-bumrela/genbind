import os
import selfies as sf
from src.utils import canonicalize, is_valid_smiles, ensure_dir
from pathlib import Path
import yaml

# Load config
with open("src/config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = cfg['data']['raw_dir']
PROCESSED_DIR = cfg['data']['processed_dir']
TRAIN_FILE = cfg['data']['train_file']
VOCAB_FILE = cfg['data']['vocab_file']

ensure_dir(PROCESSED_DIR)

def load_raw_smiles():
    all_smiles = []
    for file in Path(RAW_DIR).glob("*.smi"):
        with open(file) as f:
            all_smiles.extend([line.strip() for line in f])
    return all_smiles

def preprocess_smiles(smiles_list):
    canonical_smiles = [canonicalize(s) for s in smiles_list]
    canonical_smiles = [s for s in canonical_smiles if s and is_valid_smiles(s)]
    return canonical_smiles

def smiles_to_selfies(smiles_list):
    return [sf.encoder(s) for s in smiles_list]

def build_vocab(selfies_list):
    tokens = set()
    for s in selfies_list:
        tokens.update(sf.split_selfies(s))
    tokens = sorted(list(tokens))
    with open(os.path.join(PROCESSED_DIR, VOCAB_FILE), 'w') as f:
        f.write("\n".join(tokens))
    return tokens

def save_processed(selfies_list):
    with open(os.path.join(PROCESSED_DIR, TRAIN_FILE), 'w') as f:
        f.write("\n".join(selfies_list))

if __name__ == "__main__":
    raw_smiles = load_raw_smiles()
    print(f"Loaded {len(raw_smiles)} SMILES")
    processed_smiles = preprocess_smiles(raw_smiles)
    selfies_list = smiles_to_selfies(processed_smiles)
    build_vocab(selfies_list)
    save_processed(selfies_list)
    print(f"Saved processed {len(selfies_list)} molecules")
