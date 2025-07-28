import math
import tiktoken

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.shift = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, context_length, dropout, qkv_bias):
        super().__init__()

        assert d_out % n_heads == 0, "d_out should be a multiple of n_heads"

        self.dk = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_out, d_out, bias=qkv_bias)

        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape

        # Q, K, V -> batch_size, num_tokens, dk
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Q, K, V -> batch_size, num_tokens, n_heads, head_dim
        Q = Q.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        K = K.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        V = V.view(batch_size, num_tokens, self.n_heads, self.head_dim)

        # Q, K, V -> batch_size, n_heads, num_tokens, head_dim
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        mask = self.mask.bool()[:num_tokens, :num_tokens].to(x.device)

        raw_attention = Q @ K.transpose(2, 3) / math.sqrt(self.head_dim)
        masked_attention = torch.masked_fill(raw_attention, mask, -math.inf)
        scaled_attention = self.dropout(F.softmax(masked_attention, dim=-1))

        context_vectors = scaled_attention @ V
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.dk)
        output = self.output(context_vectors)
        
        return output
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi))) * (x + 0.04775 * (x ** 3))
        return F.gelu(x, approximate='tanh')
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )

    def forward(self, x):
        return self.layers(x)

    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            cfg['emb_dim'],
            cfg['emb_dim'],
            cfg['n_heads'],
            cfg['context_length'],
            cfg['drop_rate'],
            cfg['qkv_bias']
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['num_layers'])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias = cfg['qkv_bias'])
        self.out_head.weight = self.tok_emb.weight


    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embd = self.tok_emb(in_idx)
        pos_embd = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embd + pos_embd
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


if __name__ == '__main__':
    cfg = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "n_heads": 12,
        "num_layers": 12,
        "context_length": 1024,
        "drop_rate": 0.1,
        "qkv_bias": False,    
    }


    model = GPT2Model(cfg)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dummy_input = torch.randint(0, cfg['vocab_size'], (2, cfg['context_length']))  # (batch_size=2, seq_len=1024)
    dummy_input.to(device)
    logits = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    print("Model initialized.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Total Parameters: {total_params:,}")
