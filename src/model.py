import torch
import torch.nn as nn

import math
import torch.nn.functional as F


# GPT block architecture
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init()
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(self.dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BangaGPT(nn.Module):
    def __init__(self, max_len, num_blocks, embed_dim, num_head, dropout):
        super().__init__()
        self.max_len = max_len
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        
        
        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_head, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        




if __name__ == '__main__':
    
    
    
    
    pass






