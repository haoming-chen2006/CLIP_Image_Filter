import torch

b = torch.randn(2,2,2,2)
from einops.layers.torch import Rearrange
print(b)
rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)
c = rearrange(b)
print(c)