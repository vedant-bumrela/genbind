import torch
from src.models import SeqVAE
from src.dataset import MoleculeDataset
import yaml
import selfies as sf
from rdkit import Chem
import os
import torch.nn.functional as F

# ----------------------------
# Load config
# ----------------------------
with open("src/config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg['training']['device'])
num_samples = cfg['sampling']['num_samples']
max_len = cfg['sampling']['max_len']
checkpoint_dir = cfg['training']['checkpoint_dir']
temperature = cfg['sampling'].get('temperature', 0.8)  # default 0.8

# ----------------------------
# Dataset & vocab
# ----------------------------
dataset = MoleculeDataset()
vocab_size = len(dataset.token2idx)
idx2token = dataset.idx2token

# ----------------------------
# Load model checkpoint
# ----------------------------
model = SeqVAE(vocab_size).to(device)
checkpoint_path = "checkpoints/vae_epoch50.pt"  # old checkpoint

# --- Partial loading: only load matching layers ---
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = model.state_dict()
for k in checkpoint:
    if k in state_dict and checkpoint[k].shape == state_dict[k].shape:
        state_dict[k] = checkpoint[k]

model.load_state_dict(state_dict)
model.eval()

# ----------------------------
# Generate molecules
# ----------------------------
samples = []
with torch.no_grad():
    for _ in range(num_samples):
        z = torch.randn(1, cfg['training']['latent_dim']).to(device)
        logits = model.decoder(z, max_len)

        # --- temperature sampling ---
        probs = F.softmax(logits / temperature, dim=-1)
        token_ids = torch.multinomial(probs.squeeze(0), 1).squeeze().tolist()

        if isinstance(token_ids, int):  # single-token case
            token_ids = [token_ids]
        tokens = [idx2token[i] for i in token_ids if i != 0]
        selfies_str = "".join(tokens)
        try:
            smiles = sf.decoder(selfies_str)
            # Keep only valid SMILES
            if Chem.MolFromSmiles(smiles) is not None:
                samples.append(smiles)
        except Exception:
            continue

# ----------------------------
# Print results
# ----------------------------
print("Generated Molecules:")
for i, s in enumerate(samples, 1):
    print(f"{i}: {s}")

if len(samples) < num_samples:
    print(f"\n{num_samples - len(samples)} invalid molecules filtered out.")
