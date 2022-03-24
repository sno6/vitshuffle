from torch import nn
from torch.functional import F

import torch
import math


class Config:
    """
    Just decided to chuck everything in together cus' why not.
    """

    # Model configuration.
    embed_dim = 512
    n_blocks = 8
    n_heads = 8

    image_size = 28
    patch_size = 7

    # Trainer configuration.
    epochs = 100
    batch_size = 1
    learning_rate = 1e-3

    print_loss_every_iter = 10
    test_every_n_epochs = 10
    save_chkpt_every_n_epochs = 10000

    # Logging to wandb.
    logging = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.n_heads = config.n_heads

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiheadSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.embed_dim, config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim, config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VitShuffle(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_patches = int((config.image_size / config.patch_size)**2)

        self.embedding = nn.Linear(config.patch_size**2, config.embed_dim)
        # self.positional = nn.Parameter(torch.zeros(1, n_patches, config.embed_dim))

        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_blocks)])
        self.mlp = nn.Linear(config.embed_dim, n_patches)

    def forward(self, x, y=None):
        b, t, c = x.size()

        x = self.embedding(x)
        # x = x + self.positional[:, :t, :]
        y_hat = self.blocks(x)
        y_hat = self.mlp(y_hat)

        loss = None
        if y is not None:
            loss = F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), y.view(-1), reduction="mean")

        return y_hat, loss
