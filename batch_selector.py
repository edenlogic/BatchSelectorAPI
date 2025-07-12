import math
from typing import Callable

def selectBatchSize(N: int,
                    model_fn: Callable,
                    train_fn: Callable,
                    widen: int = 2) -> int:
    """
    Select optimal batch size via actual training efficiency.
    Compares powers-of-two around sqrt(N) and selects the one
    with highest (accuracy / training time).

    Parameters:
        N (int): Number of training samples
        model_fn (Callable): Function that returns a new model
        train_fn (Callable): Function(model, batch_size) -> (accuracy, time)
        widen (int): Number of powers-of-two to test on each side

    Returns:
        int: Best-performing batch size
    """
    assert N > 0, "N must be a positive integer"

    center_exp = round(math.log2(math.sqrt(N)))
    candidates = [2 ** i for i in range(max(4, center_exp - widen), center_exp + widen + 1)]

    results = []
    for B in candidates:
        acc, t = train_fn(model_fn(), B)
        efficiency = acc / t
        results.append((efficiency, B))

    _, best_B = max(results, key=lambda x: x[0])
    return best_B


def getBatchSizeMNIST(N: int, widen: int = 2) -> int:
    """
    Returns best batch size for MNIST using selectBatchSize with fixed model and train function.
    """
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    import time

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        def forward(self, x):
            return self.fc(x)

    def model_fn():
        return SimpleNet()

    def train_fn(model, batch_size):
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="./data", train=False, transform=transform)

        subset = Subset(train_data, list(range(min(N, 60000))))
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=256)

        model.train()
        opt = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        start = time.time()
        for xb, yb in train_loader:
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        elapsed = time.time() - start

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        accuracy = correct / total
        return accuracy, elapsed

    return selectBatchSize(N, model_fn, train_fn, widen)