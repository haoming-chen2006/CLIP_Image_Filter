import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
from config import CFG


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F 

class CasualSelfAttention(nn.Module):
    def __init__(self,n_embed,block_size,num_heads,dropout = 0.1):
        self.qkv = nn.Linear(n_embed,3*n_embed)
        self.dropout = nn.dropout(dropout)
        self.output_projection = nn.Linear(n_embed,n_embed)
        self.num_heads = num_heads
        self.block_size = block_size
       self.register_buffer("bias",torch.trill(torch.ones(self.block_size,self.block_size)).view(1,1,self.block_size,self.block_size))

    def forward(self,x):
        B,T,C = x.shape()
        q,k,v = self.qkv(x).split(dim = -1, n_embed)
        q = q.view(B,T,num_heads,n_embed/num_heads).transpose(1,2)
        k = k.view(B,T,num_heads,n_embed/num_heads).transpose(1,2)
        v = v.view(B,T,num_heads,n_embed/num_heads).transpose(1,2)
        score = q@k.transpose(-2,-1)/(C)**0.5
        score.masked_fill(self.bias[:,:,"T","T"] == 0, float("-inf"))
        score = F.softmax(score,dim = -1)
        y = score@v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        output = self.output_projection(y)
        return output

class MLP(nn.Module):
    def __init__(self,n_embed,hidden_dim,dropout = 0.1):
        self.fn1 = nn.Linear(n_embed,hidden_dim)
        self.fn2 = nn.Linear(hidden_dim,n_embed)
        self.dropout = nn.dropout(dropout)
    def forward(self,x):
        x = self.fn1(x)
        x = nn.GeLu(x)
        x = self.fn2(x)
        x = self.dropout(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self,n_embed,block_size,num_heads,hidden_dim,dropout = 0.1):
        self.attention = CasualSelfAttention(n_embed,block_size,num_heads,dropout)
        self.mlp = MLP(n_embed,hidden_dim,dropout)
        self.dropout = nn.dropout(dropout)
        self.layernorm = nn.LayerNorm()
    def forward(self,x):
        x = x + self.attention(self.layernorm(x))
        x = x + self.mlp(self.layernorm(x))
        return x


class GPT2(nn.Module):
    def __init__(self,n_blocks,vocab_size,n_embed,block_size,num_heads,hidden_dim,dropout = 0.1):
        self.dropout = nn.dropout(0.1)
        self.lm_head = nn.Linear(n_embed,vocab_size)
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embeddings(vocab_size,n_embed)
                wpe = nn.Embeddings(block_size,n_embed)
                blocks = nn.ModuleList(AttentionBlock(n_embed,block_size,num_heads,hidden_dim,dropout) for _ in range(n_blocks))
                ln_f = nn.LayerNorm()
            )
        )
    def forward(self,idx):
        B,T = idx.size()
        l = torch.arange(0,T)
        token = self.transformer.wte(idx)
        pos = self.transformer.wpe(idx)
        x = self.dropout(token + pos)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(:,[-1],:)
        return logits