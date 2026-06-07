import numpy as np

lr = 0.1

lam = 0.05


def forward(g):

    taskLoss = 0.5 * np.sum(g ** 2)

    l1Loss = lam * np.sum(np.abs(g))

    totalLoss = taskLoss + l1Loss

    return totalLoss


def backward(g):

    gradTask = g

    gradL1 = lam * np.sign(g)

    gradTotal = gradTask + gradL1

    return gradTotal


def sgdStep(g):

    grad = backward(g)

    gNew = g - lr * grad

    return gNew


g = np.array(
    [
        0.10,
        -0.20,
        0.67,
        0.00,
    ],
    dtype=np.float32,
)

print(g)

lossBefore = forward(g)

print(lossBefore)

grad = backward(g)

print(grad)

g = sgdStep(g)

lossAfter = forward(g)

print(g)

print(lossAfter)