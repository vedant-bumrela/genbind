import torch
from torch.utils.data import Dataset
import yaml
from src.data_prep import PROCESSED_DIR, TRAIN_FILE, VOCAB_FILE
import selfies as sf

with open("src/config.yaml") as f:
    cfg = yaml.safe_load(f)

class MoleculeDataset(Dataset):
    def __init__(self):
        self.data_file = f"{PROCESSED_DIR}/{TRAIN_FILE}"
        self.vocab_file = f"{PROCESSED_DIR}/{VOCAB_FILE}"
        self.load_data()
        self.build_token_mappings()

    def load_data(self):
        with open(self.data_file) as f:
            self.smiles = [line.strip() for line in f]

    def build_token_mappings(self):
        with open(self.vocab_file) as f:
            tokens = [line.strip() for line in f]
        self.token2idx = {t:i+1 for i,t in enumerate(tokens)}
        self.token2idx['<pad>'] = 0
        self.idx2token = {i:t for t,i in self.token2idx.items()}

    def __len__(self):
        return len(self.smiles)

    def encode_selfies(self, s):
        return [self.token2idx[t] for t in sf.split_selfies(s)]

    def __getitem__(self, idx):
        tokens = self.encode_selfies(self.smiles[idx])
        return torch.tensor(tokens, dtype=torch.long)
