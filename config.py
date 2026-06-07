class Config:
    def __init__(self):
        self.window_size = 64
        self.seq_len = 1024
        self.dModel = 256
        self.n_heads = 2

        ## 0-normal, 1 - sliding, 2 - Linear ,3 - MAQ ##
        self.attention_type = 3
        ## 0-normal,1 - sin , 2 - RoPE, 3 - ALiBi ,4 - Positional Based ##
        self.embedding_type = 3

        self.vocab_size = 50257
        self.num_layers = 4
        self.batchSize = 2
        self.kernel_size = 3
        self.ConvType = 2
        self.epochs = 10
        self.lr = 3e-4
        self.dropout = 0.1