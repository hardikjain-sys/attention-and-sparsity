import torch
import torch.nn as nn
from torch.nn import functional as F
import math


MAX = 4096

class gateConvFFN(nn.Module):
    def __init__(self, dModel, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(dModel,2 * dModel,kernel_size,padding=0)
        self.conv2 = nn.Conv1d(dModel, dModel, kernel_size, padding=0)

    def forward(self, x):

        x = x.transpose(1, 2)

        pad1 = self.conv1.kernel_size[0] - 1

        x = F.pad(x, (pad1, 0))

        z = self.conv1(x)

        A, B = z.chunk(2, dim=1)

        gate = A * torch.sigmoid(B)

        pad2 = self.conv2.kernel_size[0] - 1

        gate = F.pad(gate, (pad2, 0))

        out = self.conv2(gate)

        out = out.transpose(1, 2)
        return out


class Conv1DBlock(nn.Module):
    def __init__(self, dModel, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(
            dModel, dModel,
            kernel_size,
            padding=0
        )
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        pad = self.conv.kernel_size[0] - 1
        x = F.pad(x, (pad, 0))
        x = self.conv(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        return x

class SingleBlockMQAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.num_heads = config.n_heads
        self.head_dim = config.dModel // self.num_heads
        self.config = config

        if config.ConvType == 1:
            self.ln0 = nn.LayerNorm(config.dModel)
            self.conv = Conv1DBlock(config.dModel,config.kernel_size)

        self.ln1 = nn.LayerNorm(config.dModel)
        self.query = nn.Linear(config.dModel, config.dModel, bias=False)
        self.key = nn.Linear(config.dModel, self.head_dim, bias=False)
        self.value = nn.Linear(config.dModel, self.head_dim, bias=False)

        self.ln2 = nn.LayerNorm(config.dModel)
        if config.ConvType == 2:
            self.perceptron = gateConvFFN(config.dModel, config.kernel_size)
        else:
            self.perceptron = nn.Sequential(
                nn.Linear(config.dModel, 4 * config.dModel, bias=True),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * config.dModel, config.dModel, bias=True),
                nn.Dropout(config.dropout)
            )
        self.proj = nn.Linear(config.dModel, config.dModel, bias=False)

        if config.embedding_type == 2:
            invF = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            t = torch.arange(MAX).float()
            freqs = torch.einsum("i,j->ij", t, invF)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("sin", emb.sin()[None, None, :, :])
            self.register_buffer("cos", emb.cos()[None, None, :, :])

        if config.embedding_type == 3:
            slopes = torch.linspace(1.0, 0.1, steps=self.num_heads)
            self.register_buffer("alibi_slopes", slopes)

        if config.embedding_type == 4:
            self.relative_bias = nn.Parameter(
                torch.zeros(self.num_heads, config.seq_len, config.seq_len)
            )

    def rope(self, q, k, sin, cos):

        half = q.shape[-1] // 2

        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]

        q_rot = torch.cat([-q2, q1], dim=-1)
        k_rot = torch.cat([-k2, k1], dim=-1)

        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin

        return q, k

    def forward(self, x):
        B, T, C = x.shape
        if self.config.ConvType == 1:
            x = x + self.conv(self.ln0(x))
        norm_x = self.ln1(x)

        Q = self.query(norm_x)
        K = self.key(norm_x)
        V = self.value(norm_x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, 1, self.head_dim).transpose(1, 2)
        V = V.view(B, T, 1, self.head_dim).transpose(1, 2)

        K = K.expand(-1, self.num_heads, -1, -1)
        V = V.expand(-1, self.num_heads, -1, -1)

        if self.config.embedding_type == 2:
            sin = self.sin[:, :, :T, :]
            cos = self.cos[:, :, :T, :]
            Q, K = self.rope(Q, K, sin, cos)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        if self.config.embedding_type == 3:
            pos = torch.arange(T, device=x.device)
            bias = pos[None, :] - pos[:, None]
            bias = bias.unsqueeze(0).unsqueeze(0)
            slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1)
            scores = scores + slopes * bias

        if self.config.embedding_type == 4:
            scores = scores + self.relative_bias[:, :T, :T].unsqueeze(0)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)
        out = self.resid_dropout(out)
        x = x + out
        xNormalF = self.ln2(x)
        xNormalF = self.perceptron(xNormalF)
        x = x + xNormalF

        return x


class SingleBlockLinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if config.ConvType == 1:
            self.ln0 = nn.LayerNorm(config.dModel)
            self.conv = Conv1DBlock(config.dModel,config.kernel_size)
        self.num_heads = config.n_heads
        self.config = config
        self.head_dim = config.dModel // self.num_heads

        self.ln1 = nn.LayerNorm(config.dModel)
        self.query = nn.Linear(config.dModel, config.dModel, bias=False)
        self.key = nn.Linear(config.dModel, config.dModel, bias=False)
        self.value = nn.Linear(config.dModel, config.dModel, bias=False)

        self.ln2 = nn.LayerNorm(config.dModel)
        if config.ConvType == 2:
            self.perceptron = gateConvFFN(config.dModel, config.kernel_size)
        else:
            self.perceptron = nn.Sequential(
                nn.Linear(config.dModel, 4 * config.dModel, bias=True),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * config.dModel, config.dModel, bias=True),
                nn.Dropout(config.dropout)
            )
        self.proj = nn.Linear(config.dModel, config.dModel, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        if self.config.ConvType == 1:
            x = x + self.conv(self.ln0(x))
        norm_x = self.ln1(x)


        Q = self.query(norm_x)
        K = self.key(norm_x)
        V = self.value(norm_x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        phi_Q = torch.nn.functional.elu(Q) + 1
        phi_K = torch.nn.functional.elu(K) + 1

        Q_t = phi_Q.transpose(1, 2)
        K_t = phi_K.transpose(1, 2)
        V_t = V.transpose(1, 2)

        KV = torch.einsum('bthd,bthv->bthdv', K_t, V_t)

        KV_cum = KV.cumsum(dim=1)
        K_cum = K_t.cumsum(dim=1)

        num = torch.einsum('bthd,bthdv->bthv', Q_t, KV_cum)
        den = torch.einsum('bthd,bthd->bth', Q_t, K_cum)
        den = den.clamp(min=1e-6).unsqueeze(-1)

        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)
        out = self.resid_dropout(out)
        x = x + out
        xNormalF = self.ln2(x)
        xNormalF = self.perceptron(xNormalF)
        x = x + xNormalF

        return x


class SingleBlockSliding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if config.ConvType == 1:
            self.ln0 = nn.LayerNorm(config.dModel)
            self.conv = Conv1DBlock(config.dModel,config.kernel_size)
        self.num_heads = config.n_heads
        self.head_dim = config.dModel // self.num_heads
        self.config = config

        self.ln1 = nn.LayerNorm(config.dModel)
        self.query = nn.Linear(config.dModel, config.dModel, bias=False)
        self.key = nn.Linear(config.dModel, config.dModel, bias=False)
        self.value = nn.Linear(config.dModel, config.dModel, bias=False)

        self.ln2 = nn.LayerNorm(config.dModel)
        if config.ConvType == 2:
            self.perceptron = gateConvFFN(config.dModel, config.kernel_size)
        else:
            self.perceptron = nn.Sequential(
                nn.Linear(config.dModel, 4 * config.dModel, bias=True),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * config.dModel, config.dModel, bias=True),
                nn.Dropout(config.dropout)
            )
        self.proj = nn.Linear(config.dModel, config.dModel, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        w = self.config.window_size

        if self.config.ConvType == 1:
            x = x + self.conv(self.ln0(x))

        norm_x = self.ln1(x)
        Q = self.query(norm_x)
        K = self.key(norm_x)
        V = self.value(norm_x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        K_pad = F.pad(K, (0, 0, w, 0))
        V_pad = F.pad(V, (0, 0, w, 0))

        K_win = K_pad.unfold(2, w + 1, 1).permute(0, 1, 2, 4, 3)
        V_win = V_pad.unfold(2, w + 1, 1).permute(0, 1, 2, 4, 3)

        Q = Q.unsqueeze(3)

        idx = torch.arange(T, device=x.device)
        valid_len = torch.clamp(idx + 1, max=w + 1)
        block = torch.arange(w + 1, device=x.device).unsqueeze(0) < (w + 1 - valid_len).unsqueeze(1)
        float_mask = block.float().masked_fill(block, float('-inf'))
        float_mask = float_mask.unsqueeze(0).unsqueeze(0).unsqueeze(3)

        Q = Q.contiguous()
        K_win = K_win.contiguous()
        V_win = V_win.contiguous()

        out = F.scaled_dot_product_attention(Q, K_win, V_win, attn_mask=float_mask)



        out = out.squeeze(3).transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.resid_dropout(out)
        x = x + out
        x = x + self.perceptron(self.ln2(x))

        return x



class SingleBlockSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_heads

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if config.ConvType == 1:
            self.ln0 = nn.LayerNorm(config.dModel)
            self.conv = Conv1DBlock(config.dModel,config.kernel_size)



        self.config = config
        self.head_dim = config.dModel // self.num_heads

        self.ln1 = nn.LayerNorm(config.dModel)
        self.query = nn.Linear(config.dModel, config.dModel, bias=False)
        self.key = nn.Linear(config.dModel, config.dModel, bias=False)
        self.value = nn.Linear(config.dModel, config.dModel, bias=False)

        self.ln2 = nn.LayerNorm(config.dModel)
        if config.ConvType == 2:
            self.perceptron = gateConvFFN(config.dModel, config.kernel_size)
        else:
            self.perceptron = nn.Sequential(
                nn.Linear(config.dModel, 4 * config.dModel, bias=True),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * config.dModel, config.dModel, bias=True),
                nn.Dropout(config.dropout)
            )




        self.proj = nn.Linear(config.dModel, config.dModel, bias=False)

        if config.embedding_type == 2:
            invF = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            t = torch.arange(MAX).float()
            freqs = torch.einsum("i,j->ij", t, invF)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("sin", emb.sin()[None, None, :, :])
            self.register_buffer("cos", emb.cos()[None, None, :, :])

        if config.embedding_type == 3:
            slopes = torch.linspace(1.0, 0.1, steps=self.num_heads)
            self.register_buffer("alibi_slopes", slopes)

        if config.embedding_type == 4:
            self.relative_bias = nn.Parameter(
                torch.zeros(self.num_heads, config.seq_len, config.seq_len)
            )

    def rope(self, q, k, sin, cos):

        half = q.shape[-1] // 2

        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]

        q_rot = torch.cat([-q2, q1], dim=-1)
        k_rot = torch.cat([-k2, k1], dim=-1)

        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin

        return q, k

    def forward(self, x):
        B, T, C = x.shape

        if self.config.ConvType == 1:
            x = x + self.conv(self.ln0(x))
        norm_x = self.ln1(x)

        Q = self.query(norm_x)
        K = self.key(norm_x)
        V = self.value(norm_x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.config.embedding_type == 2:
            sin = self.sin[:, :, :T, :]
            cos = self.cos[:, :, :T, :]
            Q, K = self.rope(Q, K, sin, cos)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        if self.config.embedding_type == 3:
            pos = torch.arange(T, device=x.device)
            bias = pos[None, :] - pos[:, None]
            bias = bias.unsqueeze(0).unsqueeze(0)
            slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1)
            scores = scores + slopes * bias

        if self.config.embedding_type == 4:
            scores = scores + self.relative_bias[:, :T, :T].unsqueeze(0)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)
        out = self.resid_dropout(out)
        x = x + out
        xNormalF = self.ln2(x)
        xNormalF = self.perceptron(xNormalF)
        x = x + xNormalF

        return x


def get_attention_block(config):
    if config.attention_type == 0:
        return SingleBlockSelfAttention(config)
    elif config.attention_type == 1:
        return SingleBlockSliding(config)
    elif config.attention_type == 2:
        return SingleBlockLinearAttention(config)
    elif config.attention_type == 3:
        return SingleBlockMQAAttention(config)
    else:
        return SingleBlockSelfAttention(config)