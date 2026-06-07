import torch
import torch.nn as nn
import math
from model.attention import get_attention_block




MAX = 4096

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.dModel)

        if config.embedding_type == 0:
            self.pos_emb = nn.Embedding(config.seq_len, config.dModel)

        elif config.embedding_type == 1:
            pe = torch.zeros(MAX, config.dModel)

            position = torch.arange(0,MAX).unsqueeze(1).float()

            div_term = torch.exp(
                torch.arange(0, config.dModel, 2).float() * (-math.log(10000.0) / config.dModel))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        self.blocks = nn.ModuleList([
            get_attention_block(config)
            for _ in range(config.num_layers)
        ])

        self.lnF = nn.LayerNorm(config.dModel)
        self.linear = nn.Linear(config.dModel, config.vocab_size, bias=False)


    def forward(self, x):
        B, T = x.shape
        tok = self.embedding(x)

        if self.config.embedding_type == 0:
            pos = torch.arange(T, device=x.device)
            pos = self.pos_emb(pos)
            x = tok + pos

        elif self.config.embedding_type == 1:
            pos = self.pe[:, :T, :]
            x = tok + pos

        elif self.config.embedding_type == 2:
            x = tok

        elif self.config.embedding_type == 3:
            x = tok

        elif self.config.embedding_type == 4:
            x = tok
        else:
            pos = torch.arange(T, device=x.device)
            pos = self.pos_emb(pos)
            x = tok + pos



        for block in self.blocks:
            x = block(x)

        x = self.lnF(x)
        final = self.linear(x)

        return final
