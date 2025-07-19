import torch
import torch.nn as nn 
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_2tuple(x):
    return (x, x) if not isinstance(x, (tuple, list)) else tuple(x[:2])


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = self.proj(x)                  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)                  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)             # (B, num_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, dropout):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q = qkv[0].view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = qkv[1].view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = qkv[2].view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, n_embd, n_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads, dropout)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, n_heads=12, dropout=0.1, num_classes=1000,proj_dim = 512)):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, n_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, proj_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                              # (B, 196, 768)
        cls_tokens = self.cls_token.expand(B, 1, -1)         # (B, 1, 768)
        x = torch.cat((cls_tokens, x), dim=1)                # (B, 197, 768)
        x = x + self.pos_embed                               # Add positional embedding
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x[:, 0])                               # Use [CLS] token only
        return x
