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
num_samples = cfg['sampling'].get('num_samples', 500)  # generate more molecules
max_len = cfg['sampling']['max_len']
checkpoint_dir = cfg['training']['checkpoint_dir']
temperature = cfg['sampling'].get('temperature', 0.8)  # temperature sampling

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
checkpoint_path = f"{checkpoint_dir}/vae_epoch50.pt"  # adjust if needed
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ----------------------------
# Generate molecules
# ----------------------------
samples = []
valid_smiles = []

with torch.no_grad():
    for i in range(num_samples):
        z = torch.randn(1, cfg['training']['latent_dim']).to(device)
        logits = model.decoder(z, max_len)

        # temperature-controlled sampling
        probs = F.softmax(logits / temperature, dim=-1)
        token_ids = torch.multinomial(probs.squeeze(0), 1).squeeze().tolist()

        if isinstance(token_ids, int):
            token_ids = [token_ids]
        tokens = [idx2token[i] for i in token_ids if i != 0]
        selfies_str = "".join(tokens)

        try:
            smiles = sf.decoder(selfies_str)
            if Chem.MolFromSmiles(smiles) is not None:
                samples.append(smiles)
                valid_smiles.append(smiles)
        except Exception:
            continue

# ----------------------------
# Save to file
# ----------------------------
output_file = "generated_molecules.txt"
with open(output_file, "w") as f:
    for smi in valid_smiles:
        f.write(f"{smi}\n")

print(f"Generated {len(samples)} molecules, {len(valid_smiles)} valid.")
print(f"Valid molecules saved to {output_file}")
