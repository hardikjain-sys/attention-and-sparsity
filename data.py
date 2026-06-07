import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class CustomDataset(Dataset):

  def __init__(self, tokens, seq_len):

    self.tokens = torch.tensor(tokens, dtype=torch.long)
    self.seq_len = seq_len

  def __len__(self):
    return (len(self.tokens) -1 ) // self.seq_len

  def __getitem__(self, index):
        start = index * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y



def DataLoaders(config):
    with open('train.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(content)

    split = int(0.9 * len(tokens))
    trainTokens = tokens[:split]
    testTokens = tokens[split:]

    trainDataset = CustomDataset(trainTokens, config.seq_len)
    testDataset = CustomDataset(testTokens, config.seq_len)

    trainLoader = DataLoader(trainDataset, batch_size=config.batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=config.batchSize, shuffle=False)

    return trainLoader, testLoader
