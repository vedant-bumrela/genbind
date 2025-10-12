import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from src.dataset import MoleculeDataset
from src.models import SeqVAE
from src.utils import ensure_dir
import yaml
import os

with open("src/config.yaml") as f:
    cfg = yaml.safe_load(f)

batch_size = cfg['training']['batch_size']
lr = cfg['training']['lr']
epochs = cfg['training']['epochs']
checkpoint_dir = cfg['training']['checkpoint_dir']
device = cfg['training']['device']

ensure_dir(checkpoint_dir)

dataset = MoleculeDataset()
vocab_size = len(dataset.token2idx)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True))

model = SeqVAE(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)

def loss_fn(recon_x, x, mu, logvar):
    recon_x = recon_x.view(-1, recon_x.size(-1))
    x = x.view(-1)
    CE = criterion(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return CE + KLD

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x, x.size(1))
        loss = loss_fn(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"vae_epoch{epochs}.pt"))

