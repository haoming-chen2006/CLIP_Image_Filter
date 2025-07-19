import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# ---- Normalization Block ----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=1, keepdim=True) / (x.shape[1] ** 0.5)
        return self.scale * x / (norm + self.eps)


# ---- Basic Conv Block ----
class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm = RMSNorm(out_dim)
        self.nonlin = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return self.dropout(x)


# ---- Residual Block ----
class ResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block1 = Block(in_dim, out_dim)
        self.block2 = Block(out_dim, out_dim)
        self.skip_proj = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip_proj(x)


# ---- Downsampling ----
class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)
        self.conv = nn.Conv2d(dim * 4, dim_out, kernel_size=1)

    def forward(self, x):
        x = self.rearrange(x)
        return self.conv(x)


# ---- CLIP-style Image Tokenizer ----
class CLIPImageTokenizer(nn.Module):
    def __init__(self, in_channels=3, init_dim=64, depth=4, mults=(1, 2, 4, 8)):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, init_dim, kernel_size=7, stride=2, padding=3)
        dims = [init_dim * m for m in mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Initial res block
        self.down_blocks.append(ResNetBlock(init_dim, dims[0]))

        # Progressive downsampling
        for in_dim, out_dim in in_out:
            self.down_blocks.append(ResNetBlock(in_dim, in_dim))
            self.downsamples.append(Downsample(in_dim, out_dim))

        # Final output projection to embedding space
        self.final_norm = nn.LayerNorm(dims[-1])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # global embedding

    def forward(self, x):
        x = self.init_conv(x)
        for res_block, down in zip(self.down_blocks, self.downsamples + [None]):
            x = res_block(x)
            if down:
                x = down(x)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        return self.final_norm(x)


