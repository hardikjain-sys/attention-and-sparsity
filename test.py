import time
import math
import torch
import torch.nn as nn

from config import Config
from data import DataLoaders
from model.model import Model


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def trainOneEpoch(model, trainLoader, optimizer, criterion):

    model.train()

    totalLoss = 0
    losses = []

    startTime = time.time()

    for x, y in trainLoader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        B, T, C = logits.shape

        loss = criterion(
            logits.view(B * T, C),
            y.view(B * T)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        lossItem = loss.item()

        totalLoss += lossItem
        losses.append(lossItem)

    epochTime = time.time() - startTime

    avgLoss = totalLoss / len(trainLoader)

    variance = sum(
        (l - avgLoss) ** 2 for l in losses
    ) / len(losses)

    return avgLoss, variance, epochTime


@torch.no_grad()
def evaluateModel(model, valLoader, criterion):

    model.eval()

    totalLoss = 0

    for x, y in valLoader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        B, T, C = logits.shape

        loss = criterion(
            logits.view(B * T, C),
            y.view(B * T)
        )

        totalLoss += loss.item()

    avgLoss = totalLoss / len(valLoader)

    perplexity = math.exp(avgLoss)

    return avgLoss, perplexity


@torch.no_grad()
def benchmarkModel(model, seqLen, batchSize=1, runs=20):

    model.eval()

    dummyInput = torch.randint(
        0,
        config.vocab_size,
        (batchSize, seqLen)
    ).to(device)

    for _ in range(5):
        _ = model(dummyInput)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(runs):
        _ = model(dummyInput)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    totalTime = end - start

    avgLatency = (totalTime / runs) * 1000

    totalTokens = batchSize * seqLen * runs

    throughput = totalTokens / totalTime

    return avgLatency, throughput


def getPeakMemory():

    if device == "cuda":

        peakMemory = (
            torch.cuda.max_memory_allocated() / 1024**2
        )

    elif device == "mps":

        peakMemory = (
            torch.mps.current_allocated_memory() / 1024**2
        )

    else:
        peakMemory = 0

    return peakMemory


@torch.no_grad()
def extrapolationTest(model, criterion, config):

    testLengths = [512, 1024, 2048]

    originalLen = config.seq_len

    for L in testLengths:

        config.seq_len = L

        _, valLoader = DataLoaders(config)

        valLoss, ppl = evaluateModel(
            model,
            valLoader,
            criterion
        )

        print(L)
        print(valLoss)
        print(ppl)

    config.seq_len = originalLen


if __name__ == "__main__":

    config = Config()

    trainLoader, valLoader = DataLoaders(config)

    model = Model(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs
    )

    criterion = nn.CrossEntropyLoss()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(device)
    print(config.attention_type)
    print(config.embedding_type)
    print(config.ConvType)
    print(config.seq_len)

    for epoch in range(config.epochs):

        print(epoch + 1)

        trainLoss, variance, epochTime = trainOneEpoch(
            model,
            trainLoader,
            optimizer,
            criterion
        )

        scheduler.step()

        print(trainLoss)

    valLoss, perplexity = evaluateModel(
        model,
        valLoader,
        criterion
    )

    latency, throughput = benchmarkModel(
        model,
        config.seq_len
    )

    peakMemory = getPeakMemory()

    print(trainLoss)
    print(valLoss)
    print(perplexity)

    print(variance)

    print(epochTime)

    print(latency)
    print(throughput)

    print(peakMemory)