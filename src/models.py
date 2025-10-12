import torch
import torch.nn as nn
import yaml

with open("src/config.yaml") as f:
    cfg = yaml.safe_load(f)

latent_dim = cfg['training']['latent_dim']
hidden_dim = cfg['training']['hidden_dim']
device = cfg['training']['device']

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, h = self.rnn(embedded)
        h = h.squeeze(0)
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.latent_fc = nn.Linear(latent_dim, hidden_dim)  # map z to hidden_dim
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, seq_len):
        # Map latent vector to RNN input size
        z = self.latent_fc(z)  # shape: [batch, hidden_dim]
        # Repeat along sequence length
        z = z.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden_dim]
        out, _ = self.rnn(z)
        return self.fc(out)

class SeqVAE(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, seq_len):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, seq_len), mu, logvar
