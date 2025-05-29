import math
import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import stateless

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 28 * 28
NUM_CLASSES_TASK = 5
BATCH_SIZE = 64
N_TRAIN_PER_TASK = 3000
N_TEST_PER_TASK = 1000
EPOCHS_A = 5
EPOCHS_B = 5
LR = 1e-2
K_DOM = 5
PWR_ITERS = 50
PWR_TOL = 1e-4

# Experiment knobs
HIDDEN_SIZES = [30, 100]
PROJECT_METHODS = ["vanilla", "bulk", "dominant", "random"]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def flat_grad(model: nn.Module) -> torch.Tensor:
    g = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    return torch.cat(g) if g else torch.zeros_like(flat_params(model))


def hvp(loss_fn, params_flat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(loss_fn(params_flat), params_flat, create_graph=True)[0]
    dot = torch.dot(grads, vec)
    hv, = torch.autograd.grad(dot, params_flat, retain_graph=False)
    return hv.detach()


def topk_eigvecs(loss_fn, params_flat: torch.Tensor, k: int, iters=PWR_ITERS, tol=PWR_TOL) -> torch.Tensor:
    """Power‑iteration with simple deflation to extract top‑k eigenvectors."""
    n = params_flat.numel()
    vecs: List[torch.Tensor] = []
    for _ in range(k):
        v = torch.randn(n, device=params_flat.device)
        v /= v.norm()
        for _ in range(iters):
            Hv = hvp(loss_fn, params_flat, v)
            if vecs:
                proj = torch.stack(vecs, 1) @ (torch.stack(vecs) @ Hv)
                Hv = Hv - proj
            v_next = Hv / (Hv.norm() + 1e-12)
            if torch.norm(v_next - v) < tol:
                break
            v = v_next
        vecs.append(v / v.norm())
    return torch.stack(vecs, 1)  # [n, k]


def split_mnist_loaders(digits: List[int], n_train: int, n_test: int) -> Tuple[DataLoader, DataLoader]:
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_all = torchvision.datasets.MNIST("data", train=True, download=True, transform=tf)
    test_all = torchvision.datasets.MNIST("data", train=False, download=True, transform=tf)

    def select(ds, n_samples):
        idx = [i for i, (_, y) in enumerate(ds) if y in digits]
        np.random.shuffle(idx)
        return Subset(ds, idx[:n_samples])

    class Remap(torch.utils.data.Dataset):
        def __init__(self, sub):
            self.sub = sub
        def __len__(self):
            return len(self.sub)
        def __getitem__(self, i):
            x, y = self.sub[i]
            return x, digits.index(int(y))  # map digits to 0‑4

    train_loader = DataLoader(Remap(select(train_all, n_train)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Remap(select(test_all, n_test)), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))


def train(model: nn.Module, loader: DataLoader, epochs: int, *, proj_mode: str = None, V: torch.Tensor | None = None, R: torch.Tensor | None = None):
    """Train for a few epochs; optionally project gradients."""
    opt = optim.SGD(model.parameters(), lr=LR)

    # Map from flat gradient slice back to per‑parameter view
    slices, start = [], 0
    for p in model.parameters():
        end = start + p.numel()
        slices.append((p, slice(start, end)))
        start = end

    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()

            if proj_mode in {"bulk", "dominant", "random"}:
                g = flat_grad(model).to(DEVICE)
                if V is not None:
                    V = V.to(DEVICE)
                if R is not None:
                    R = R.to(DEVICE)

                if proj_mode == "bulk":
                    g_proj = g - V @ (V.T @ g)
                elif proj_mode == "dominant":
                    g_proj = V @ (V.T @ g)
                else:  # random
                    g_proj = g - R @ (R.T @ g)

                # copy the projected gradient back into parameter gradients
                for p, sl in slices:
                    p.grad.copy_(g_proj[sl].view_as(p))

            opt.step()


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
    return 100.0 * correct / len(loader.dataset)


def run_experiment(hidden_size: int, proj_mode: str):
    set_seed(SEED)

    trA, teA = split_mnist_loaders(list(range(0, 5)), N_TRAIN_PER_TASK, N_TEST_PER_TASK)
    trB, teB = split_mnist_loaders(list(range(5, 10)), N_TRAIN_PER_TASK, N_TEST_PER_TASK)

    # --- Task A ---
    netA = TinyMLP(INPUT_SIZE, hidden_size, NUM_CLASSES_TASK).to(DEVICE)
    train(netA, trA, EPOCHS_A)
    acc_A0 = evaluate(netA, teA)
    stateA = copy.deepcopy(netA.state_dict())

    # Build auxiliary structures for Hessian power‑iteration
    names_shapes = [(n, p.shape) for n, p in netA.named_parameters()]
    total_params = sum(math.prod(s) for _, s in names_shapes)

    param0 = flat_params(netA).clone().detach().to(DEVICE).requires_grad_(True)

    def flat_to_paramdict(vec: torch.Tensor):
        d, offset = {}, 0
        for (name, shape) in names_shapes:
            n = math.prod(shape)
            d[name] = vec[offset:offset + n].view(shape)
            offset += n
        return d

    xA, yA = next(iter(trA))
    xA, yA = xA.to(DEVICE), yA.to(DEVICE)

    def loss_fn(vec: torch.Tensor):
        logits = stateless.functional_call(netA, flat_to_paramdict(vec), (xA,))
        return nn.CrossEntropyLoss()(logits, yA)

    V = topk_eigvecs(loss_fn, param0, K_DOM)  # stays on DEVICE

    R = None
    if proj_mode == "random":
        R = torch.randn(total_params, K_DOM, device=DEVICE)
        R = R / R.norm(dim=0, keepdim=True)

    # --- Task B ---
    netB = TinyMLP(INPUT_SIZE, hidden_size, NUM_CLASSES_TASK).to(DEVICE)
    netB.load_state_dict(stateA)
    train(netB, trB, EPOCHS_B, proj_mode=proj_mode, V=V, R=R)

    acc_A = evaluate(netB, teA)
    acc_B = evaluate(netB, teB)
    return acc_A0, acc_A, acc_B


def main():
    for h in HIDDEN_SIZES:
        for mode in PROJECT_METHODS:
            print(f"\n--- Hidden {h}, Mode {mode} ---")
            acc_A0, acc_A, acc_B = run_experiment(h, mode)
            print(f"Initial Task A acc: {acc_A0:.2f}%")
            print(f"Post‑B Task A acc: {acc_A:.2f}%")
            print(f"Task B acc       : {acc_B:.2f}%")
            print(f"Forgetting A      : {acc_A0 - acc_A:.2f} pp")
            print(f"Task A + Task B acc      : {acc_A + acc_B:.2f} pp")


if __name__ == "__main__":
    main()
