import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, emb_size=384, img_size=64):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = self.proj(x)  # [B, emb_size, H', W']
        x = x.flatten(2)  # [B, emb_size, N]
        x = x.transpose(1, 2)  # [B, N, emb_size]
        x = self.norm(x)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, num_classes=7, emb_size=384, depth=8, n_heads=12, mlp_dim=768):
        super(ViT, self).__init__()

        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))
        self.dropout = nn.Dropout(0.2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            activation='gelu',
            batch_first=True,
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, emb_size]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_size]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, emb_size]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 0]  # CLS token
        x = self.mlp_head(x)
        return x
