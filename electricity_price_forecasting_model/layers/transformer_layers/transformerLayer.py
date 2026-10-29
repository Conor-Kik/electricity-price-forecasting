import math
import torch
import torch.nn as nn

class MaskAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super(MaskAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

    def forward(self, x, mask=None):
        # x shape: (seq_len, batch_size, d_model)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        return attn_output

class FeedForward(nn.Module): # two layer fully connected feed foward network
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Main feed forward
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super(DecoderBlock, self).__init__()
        self.self_attn = MaskAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Main decoder
        attn_output = self.self_attn(x, mask)
        x = self.attention_norm(x + self.dropout(attn_output))
        feed_forward_out = self.feed_forward(x)
        x = self.feed_forward_norm(x + self.dropout(feed_forward_out))
        return x

class CrossAttentionDecoderBlock(nn.Module):
    """Transformer decoder block that attends to separate encoder memory.

    ``target`` and ``memory`` use batch-first shapes:
    ``(batch, target_length, d_model)`` and
    ``(batch, memory_length, d_model)``.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super(CrossAttentionDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.self_attention_norm = nn.LayerNorm(d_model)
        self.cross_attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target,
        memory,
        target_mask=None,
        memory_mask=None,
        target_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        self_attn_output, _ = self.self_attn(
            target,
            target,
            target,
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
            need_weights=False,
        )
        target = self.self_attention_norm(target + self.dropout(self_attn_output))

        cross_attn_output, _ = self.cross_attn(
            target,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        target = self.cross_attention_norm(target + self.dropout(cross_attn_output))

        feed_forward_output = self.feed_forward(target)
        return self.feed_forward_norm(target + self.dropout(feed_forward_output))

class DecoderTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.0):
        super(DecoderTransformer, self).__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    '''
    Positional encoding according to the original transformer paper.
    '''
    def positional_encoding(self, x):
        seq_len = x.size(1)
        d_model = x.size(2)

        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(seq_len, d_model, device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[:encoding[:, 1::2].size(1)])

        return encoding.to(dtype=x.dtype).unsqueeze(0)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        if mask is None:
            mask = causal_mask
        elif mask.dtype == torch.bool:
            mask = mask | causal_mask
        elif torch.is_floating_point(mask):
            mask = mask.masked_fill(causal_mask, float("-inf"))
        else:
            raise TypeError("Attention mask must have boolean or floating-point dtype")

        x = x + self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)