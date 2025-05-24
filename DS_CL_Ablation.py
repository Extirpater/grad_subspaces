
"""
Continual-learning toy experiment (Split-MNIST, 2 tasks)

Task A (digits 0-4) – train from scratch, extract top-k Hessian directions.
Task B (digits 5-9) – three fine-tuning modes
    1. vanilla SGD
    2. bulk-space projection   (I − P_A) g
    3. DS-CL projection        P_B (I − P_A) g
and we compare forgetting.
"""

import math, copy, time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import stateless

# ─────────────────────────── Hyper-parameters ────────────────────────────
SEED                 = 0
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE           = 28 * 28
HIDDEN_DIM           = 30
NUM_CLASSES_TASK     = 5
BATCH_SIZE           = 64
N_TRAIN_PER_TASK     = 3000
N_TEST_PER_TASK      = 1000
EPOCHS_A             = 5
EPOCHS_B             = 5
LR                   = 1e-2
K_DOM                = 5          # dominant subspace rank
PWR_ITERS            = 50
PWR_TOL              = 1e-4
# ──────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────── Model & Data ────────────────────────────────
class TinyMLP(nn.Module):
    def __init__(self, in_dim=INPUT_SIZE, hid=HIDDEN_DIM, out_dim=NUM_CLASSES_TASK):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid),    nn.Tanh(),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))

def split_mnist_loaders(digits: List[int],
                        n_train: int,
                        n_test: int) -> Tuple[DataLoader, DataLoader]:
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.1307,), (0.3081,))])
    train_all = torchvision.datasets.MNIST('data', train=True,  download=True, transform=tf)
    test_all  = torchvision.datasets.MNIST('data', train=False, download=True, transform=tf)

    def select(ds, n):
        idx = [i for i, (_, y) in enumerate(ds) if y in digits]
        np.random.shuffle(idx)
        return Subset(ds, idx[:n])

    def remap(ds):
        class _R(torch.utils.data.Dataset):
            def __init__(self, sub): self.sub = sub
            def __len__(self):       return len(self.sub)
            def __getitem__(self, i):
                x, y = self.sub[i]
                return x, digits.index(int(y))   # map to 0-4
        return _R(ds)

    train_loader = DataLoader(remap(select(train_all, n_train)),
                              batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=DEVICE.type == 'cuda')
    test_loader  = DataLoader(remap(select(test_all,  n_test)),
                              batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=DEVICE.type == 'cuda')
    return train_loader, test_loader
# ─────────────────────────── Helpers ─────────────────────────────────────
def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def flat_grad(model):
    g = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    return torch.cat(g) if g else torch.zeros_like(flat_params(model))

def hvp(loss_fn, params_flat, vec):
    """ Hessian-vector product H·v via autograd. """
    if hasattr(torch.autograd.functional, "hvp"):
        _, hv = torch.autograd.functional.hvp(
            loss_fn, params_flat, vec, create_graph=False
        )
        return hv.detach()
    grads = torch.autograd.grad(loss_fn(params_flat), params_flat, create_graph=True)[0]
    dot = torch.dot(grads, vec)
    hv, = torch.autograd.grad(dot, params_flat, retain_graph=False)
    return hv.detach()

def topk_eigvecs(loss_fn, params_flat, k, iters=PWR_ITERS, tol=PWR_TOL):
    """Power-iteration + deflation to extract top-k eigenvectors."""
    n = params_flat.numel()
    vecs = []
    for _ in range(k):
        v = torch.randn(n, device=DEVICE); v /= v.norm()
        for _ in range(iters):
            Hv = hvp(loss_fn, params_flat, v)
            if vecs:                          # deflate old comps
                proj = torch.stack(vecs, 1) @ (torch.stack(vecs) @ Hv)
                Hv = Hv - proj
            v_next = Hv / (Hv.norm() + 1e-12)
            if torch.norm(v_next - v) < tol:
                break
            v = v_next
        vecs.append(v / v.norm())
    return torch.stack(vecs, 1)               # shape [P, k]

criterion = nn.CrossEntropyLoss()

def train(model, loader, epochs, proj_mat=None):
    """SGD training loop with optional gradient projection."""
    opt = optim.SGD(model.parameters(), lr=LR)
    # slices for copying projected flat-grad back into each tensor
    slices, start = [], 0
    for p in model.parameters():
        end = start + p.numel()
        slices.append((p, slice(start, end)))
        start = end

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()

            if proj_mat is not None:
                g = flat_grad(model)
                g_proj = proj_mat @ g
                for p, sl in slices:
                    p.grad.copy_(g_proj[sl].view_as(p))

            opt.step()
            running += loss.item()
        print(f"  Ep {ep}/{epochs} | loss {running/len(loader):.4f}")

def evaluate(model, loader, name=''):
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
    acc = 100.0 * correct / len(loader.dataset)
    print(f"  {name} acc: {acc:.2f}%")
    return acc
# ─────────────────────────── Main Experiment ────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    trA, teA = split_mnist_loaders(list(range(0, 5)), N_TRAIN_PER_TASK, N_TEST_PER_TASK)
    trB, teB = split_mnist_loaders(list(range(5, 10)), N_TRAIN_PER_TASK, N_TEST_PER_TASK)

    # ---------- Task A training ----------
    net_A = TinyMLP().to(DEVICE)
    print("\n=== Train Task A ===")
    train(net_A, trA, EPOCHS_A)
    acc_A0 = evaluate(net_A, teA, 'Task A')

    # ---------- utilities for flat θ ↔ param-dict ----------
    names_shapes = [(n, p.shape) for n, p in net_A.named_parameters()]
    total_params = sum(math.prod(s) for _, s in names_shapes)

    def flat_to_paramdict(vec):
        out, off = {}, 0
        for (name, shape) in names_shapes:
            n = math.prod(shape)
            out[name] = vec[off:off + n].view(shape)
            off += n
        return out

    # sample batches for Hessian sketches
    sample_xA, sample_yA = next(iter(trA)); sample_xA, sample_yA = sample_xA.to(DEVICE), sample_yA.to(DEVICE)
    sample_xB, sample_yB = next(iter(trB)); sample_xB, sample_yB = sample_xB.to(DEVICE), sample_yB.to(DEVICE)

    # ---------- dominant subspace of Task A ----------
    paramsA = flat_params(net_A).clone().requires_grad_(True)
    def lossA(p): return criterion(stateless.functional_call(net_A, flat_to_paramdict(p), (sample_xA,)), sample_yA)
    print(f"\nEstimating Task-A dominant subspace (k={K_DOM}) …")
    V_A = topk_eigvecs(lossA, paramsA, K_DOM)                       # [P,k]
    P_preserve = torch.eye(total_params, device=DEVICE) - V_A @ V_A.T
    print("  P_preserve (freeze A sharp dirs) ready.")

    state_A = copy.deepcopy(net_A.state_dict())                     # θ_A

    # ---------- dominant subspace of Task B ----------
    net_tmp = TinyMLP().to(DEVICE); net_tmp.load_state_dict(state_A)
    paramsB0 = flat_params(net_tmp).clone().requires_grad_(True)
    def lossB(p): return criterion(stateless.functional_call(net_tmp, flat_to_paramdict(p), (sample_xB,)), sample_yB)
    print(f"\nEstimating Task-B dominant subspace (k={K_DOM}) …")
    V_B = topk_eigvecs(lossB, paramsB0, K_DOM)
    P_B = V_B @ V_B.T
    P_DSCL = P_B @ P_preserve                                      # DS-CL projector
    print("  DS-CL projection matrix ready.")

    # ---------- Task B training: DS-CL ----------
    print("\n=== Train Task B (DS-CL) ===")
    net_B_dscl = TinyMLP().to(DEVICE); net_B_dscl.load_state_dict(state_A)
    train(net_B_dscl, trB, EPOCHS_B, proj_mat=P_DSCL)
    acc_A_dscl = evaluate(net_B_dscl, teA, 'Task A after DS-CL-B')
    acc_B_dscl = evaluate(net_B_dscl, teB, 'Task B DS-CL')

    # ---------- Task B training: bulk-only ----------
    print("\n=== Train Task B (bulk-proj) ===")
    net_B_bulk = TinyMLP().to(DEVICE); net_B_bulk.load_state_dict(state_A)
    train(net_B_bulk, trB, EPOCHS_B, proj_mat=P_preserve)
    acc_A_bulk = evaluate(net_B_bulk, teA, 'Task A after bulk-B')
    acc_B_bulk = evaluate(net_B_bulk, teB, 'Task B bulk')

    # ---------- Task B training: vanilla ----------
    print("\n=== Train Task B (vanilla) ===")
    net_B_van = TinyMLP().to(DEVICE); net_B_van.load_state_dict(state_A)
    train(net_B_van, trB, EPOCHS_B)
    acc_A_van = evaluate(net_B_van, teA, 'Task A after van-B')
    acc_B_van = evaluate(net_B_van, teB, 'Task B van')

    # ---------- Summary ----------
    print("\n──────── Summary ────────")
    print(f"Task A acc after A-only      : {acc_A0:5.2f}%")
    print("Fine-tune B (vanilla)        : "
          f"A {acc_A_van:5.2f}% | B {acc_B_van:5.2f}% | forgetting {acc_A0 - acc_A_van:5.2f} pp")
    print("Fine-tune B (proj bulk-A)    : "
          f"A {acc_A_bulk:5.2f}% | B {acc_B_bulk:5.2f}% | forgetting {acc_A0 - acc_A_bulk:5.2f} pp")
    print("Fine-tune B (DS-CL)          : "
          f"A {acc_A_dscl:5.2f}% | B {acc_B_dscl:5.2f}% | forgetting {acc_A0 - acc_A_dscl:5.2f} pp")
    print("────────────────────────────")

if __name__ == '__main__':
    main()
