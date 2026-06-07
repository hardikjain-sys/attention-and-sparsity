import torch
from config import Config
from data import DataLoaders
from model.model import Model
from experiments import run_experiments

def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    config = Config()
    train_loader, test_loader = DataLoaders(config)
    model = Model(config).to(device)

    x, y = next(iter(train_loader))

    x = x.to(device)

    logits = model(x)
    print(logits.shape)

if __name__ == "__main__":
    main()