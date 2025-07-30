import math
import tiktoken

import torch
import torch.nn as nn
import torch.nn.functional as F

config = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "n_heads": 12,
        "num_layers": 12,
        "context_length": 1024,
        "drop_rate": 0.1,
        "qkv_bias": False,    
    }

class GPT2(nn.Module): 
    def __init__(self, config):
        super().__init__()

        self.tok_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = nn.Dropout(config['drop_rate'])
        self.trf_layers = nn.Sequential(*[TransformerBlock(config) for _ in range(config['num_layers'])])
        self.final_norm = LayerNorm(config['emb_dim'])
        self.output_head = nn.Linear(config['emb_dim'], config['vocab_size'])
        # self.output_head.weight = self.tok_emb.weight

    def forward(self, x):
        batch_size, seq_len = x.shape

        tok_embd = self.tok_emb(x)
        pos_embd = self.pos_emb(torch.arange(seq_len, device=x.device))

        x = tok_embd + pos_embd
        x = self.drop_emb(x)
        x = self.trf_layers(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.att = MultiHeadAttention(d_in = config['emb_dim'],
                                      d_out = config['emb_dim'],
                                      n_heads = config['n_heads'],
                                      context_length= config['context_length'],
                                      dropout = config['drop_rate'],
                                      qkv_bias= config['qkv_bias']
                                      )
        
        self.norm1 = LayerNorm(config['emb_dim'])
        self.norm2 = LayerNorm(config['emb_dim'])
        self.ff = Feedforward(config)
        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        return x
        
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, context_length=1024, dropout=0.1, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out has to be a multiple of n_heads"
        self.dk = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.Wq = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_out, d_out)

        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape
        mask = self.mask.bool()[:num_tokens, :num_tokens]

        # batch_size, num_tokens, d_in @ d_in, d_out -> batch_size, num_tokens, d_out
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # batch_size, num_tokens, d_out -> batch_size, num_tokens, n_heads, head_dim
        Q = Q.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        K = K.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        V = V.view(batch_size, num_tokens, self.n_heads, self.head_dim)

        # batch_size, num_tokens, n_heads, head_dim -> batch_size, n_heads, num_tokens, head_dim
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        #batch_size, n_heads, num_tokens, num_tokens
        raw_attention = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        masked_attention = torch.masked_fill(raw_attention, mask, -math.inf)
        scaled_attention = self.dropout((F.softmax(masked_attention, dim=-1)))

        #batch_size, n_heads, num_tokens, head_dim
        context_vector = scaled_attention @ V
        #batch_size, num_tokens, n_heads, head_dim
        context_vector = context_vector.transpose(1, 2)
        #batch_size, num_tokens, dk
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.dk)
        output = self.output(context_vector)
        return output

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = torch.mean(x, dim =-1, keepdim=True)
        var = torch.var(x, dim = -1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)
    
class Feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(config['emb_dim'], config['emb_dim'] * 4),
            GELU(),
            nn.Dropout(config['drop_rate']),
            nn.Linear(config['emb_dim'] * 4, config['emb_dim'])
        )

    def forward(self, x):
        return self.layers(x)
    
