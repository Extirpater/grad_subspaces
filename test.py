import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import hessian
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MLP_HIDDEN_SIZE = 10 #paper says 200
N_SAMPLES = 1000   #paper has 5000
BATCH_SIZE = 32    #paper has more i tink
TOTAL_EPOCHS = 3    
K_DOMINANT = 10     

LEARNING_RATE = 0.01
EPOCHS_SGD_ONLY_FOR_CHIK_TRACKING = 1 # In 'sgd' mode, start computing Hessian for chi_k after this many epochs
EMA_FACTOR = 0.9
SWITCH_THRESHOLD = 0.95 # For observation during SGD run



class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=MLP_HIDDEN_SIZE, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        # print(f"Model params: fc1({input_size*hidden_size}), fc2({hidden_size*hidden_size}), fc3({hidden_size*num_classes})")


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.tanh1(self.fc1(x))
        x = self.tanh2(self.fc2(x))
        x = self.fc3(x)
        return x

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    offset = 0
    for param in model.parameters():
        param.data.copy_(flat_params[offset:offset + param.numel()].view_as(param.data))
        offset += param.numel()

def get_flat_grad(model):
    valid_grads = [p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]
    if not valid_grads:
        num_params = sum(p.numel() for p in model.parameters())
        print("no valid grads returning 0")
        return torch.zeros(num_params, device=DEVICE)
    return torch.cat(valid_grads)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
indices = torch.randperm(len(train_dataset_full))[:N_SAMPLES]
train_dataset = Subset(train_dataset_full, indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)

def mse_loss_for_classification(outputs, targets):
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[1]).float()
    return torch.nn.functional.mse_loss(outputs, targets_one_hot)

criterion = mse_loss_for_classification

def train_model(mode='sgd', initial_model_state_dict=None):
    print(f"\n--- Training in {mode} mode ---")
    model = MLP().to(DEVICE)
    if initial_model_state_dict:
        model.load_state_dict(initial_model_state_dict)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    history = {'loss': [], 'chi_k': [], 'step': [], 'ema_chi_k': []}
    current_step = 0
    ema_chi_k_val = 0.0 # Track EMA of chi_k for this run

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model for this run has {num_params} parameters.")

    for epoch in range(TOTAL_EPOCHS):
        epoch_loss_sum = 0
        num_batches_in_epoch = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs_batch, labels_batch = inputs.to(DEVICE), labels.to(DEVICE)

            model.train()
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()

            original_flat_grad = get_flat_grad(model).clone()
            
            if torch.norm(original_flat_grad) < 1e-9: # Effectively zero gradient
                optimizer.step() # Proceed with SGD update (which will be tiny)
                history['loss'].append(loss.item())
                history['chi_k'].append(0.0) # Can't compute chi_k
                history['ema_chi_k'].append(ema_chi_k_val)
                history['step'].append(current_step)
                current_step += 1
                epoch_loss_sum += loss.item()
                num_batches_in_epoch += 1
                if i % (max(1, len(train_loader)//2)) == 0:
                    print(f"  Epoch [{epoch+1}/{TOTAL_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f} (Skipped Hessian)")
                continue

            Pk, P_bulk = None, None
            hessian_computed_this_step = False
            
            # Determine if Hessian computation is needed
            # For 'dom' or 'bulk' mode, always try.
            # For 'sgd' mode, only after EPOCHS_SGD_ONLY_FOR_CHIK_TRACKING for chi_k calculation.
            compute_hessian_trigger = (mode != 'sgd') or \
                                     (mode == 'sgd' and epoch >= EPOCHS_SGD_ONLY_FOR_CHIK_TRACKING)

            if compute_hessian_trigger:
                model.eval()
                original_model_params_backup_for_hessian = get_flat_params(model).clone()

                def hessian_loss_fn(params_to_diff_flat):
                    set_flat_params(model, params_to_diff_flat)
                    # Use the same inputs_batch, labels_batch from the current training step
                    out_hess = model(inputs_batch)
                    loss_hess_val = criterion(out_hess, labels_batch)
                    return loss_hess_val

                current_model_params_flat_for_hessian = get_flat_params(model).clone().requires_grad_(True)
                
                hessian_calc_success = False
                try:
                    # print(f"  Step {current_step}: Computing Hessian...")
                    H_matrix = hessian(hessian_loss_fn, current_model_params_flat_for_hessian, create_graph=False)
                    hessian_calc_success = True
                except Exception as e:
                    print(f"  Error computing Hessian at step {current_step}: {e}. Falling back to SGD.")
                finally:
                    set_flat_params(model, original_model_params_backup_for_hessian) # Restore params
                    model.train() # Switch back to train mode

                if hessian_calc_success:
                    eig_decomp_success = False
                    try:
                        eigenvalues, eigenvectors = torch.linalg.eigh(H_matrix.cpu().double())       # for stability with eigh, then move back
                        eigenvectors = eigenvectors.float().to(DEVICE)
                        eig_decomp_success = True
                    except Exception as e:
                        print(f"  Error in eigendecomposition at step {current_step}: {e}. Falling back to SGD.")
                    
                    if eig_decomp_success:
                        hessian_computed_this_step = True
                        top_k_eigenvectors = eigenvectors[:, -K_DOMINANT:]
                        Pk = top_k_eigenvectors @ top_k_eigenvectors.T
                        P_bulk = torch.eye(Pk.shape[0], device=DEVICE) - Pk
            
            # Calculate chi_k if Hessian was computed
            current_chi_k_val = 0.0
            if hessian_computed_this_step and Pk is not None:
                dominant_grad_component = Pk @ original_flat_grad
                norm_orig_grad = torch.norm(original_flat_grad)
                if norm_orig_grad > 1e-9:
                    current_chi_k_val = (torch.norm(dominant_grad_component) / norm_orig_grad).item()
                
                if ema_chi_k_val == 0.0 and current_chi_k_val > 0.0 : # Initialize EMA
                    ema_chi_k_val = current_chi_k_val
                elif current_chi_k_val > 0.0: # Update EMA
                    ema_chi_k_val = EMA_FACTOR * ema_chi_k_val + (1 - EMA_FACTOR) * current_chi_k_val
            
            history['chi_k'].append(current_chi_k_val)
            history['ema_chi_k'].append(ema_chi_k_val)

            # Apply SGD, Dom-SGD, or Bulk-SGD update
            if mode == 'dom' and hessian_computed_this_step and Pk is not None:
                projected_grad = Pk @ original_flat_grad
            elif mode == 'bulk' and hessian_computed_this_step and P_bulk is not None:
                projected_grad = P_bulk @ original_flat_grad
            else: # SGD mode, or Hessian/projection failed
                projected_grad = original_flat_grad # Use original gradient

            # Manually set gradients for the optimizer from the (projected) flat gradient
            offset = 0
            for param in model.parameters():
                if param.grad is not None: # Ensure grad tensor exists
                    numel = param.numel()
                    param.grad.data = projected_grad[offset : offset + numel].view_as(param.grad.data)
                    offset += numel
            
            optimizer.step()
            
            history['loss'].append(loss.item())
            history['step'].append(current_step)
            
            epoch_loss_sum += loss.item()
            num_batches_in_epoch += 1
            current_step += 1

            if i % (max(1,len(train_loader)//5)) == 0: # Print more frequently for small datasets
                 print(f"  Epoch [{epoch+1}/{TOTAL_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, EMA chi_k: {ema_chi_k_val:.3f}, Inst chi_k: {current_chi_k_val:.3f}")
        
        avg_epoch_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        print(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}] completed. Avg Loss: {avg_epoch_loss:.4f}, Final EMA chi_k: {ema_chi_k_val:.3f}")

    return model.state_dict(), history

prng_state = torch.get_rng_state() # Save PRNG state for consistent initialization
initial_model_for_state = MLP().to(DEVICE)
initial_model_state_dict_global = initial_model_for_state.state_dict()
del initial_model_for_state


torch.set_rng_state(prng_state) # Restore PRNG state
sgd_final_state, sgd_history = train_model(mode='sgd', initial_model_state_dict=initial_model_state_dict_global)

torch.set_rng_state(prng_state) # Restore PRNG state
dom_sgd_final_state, dom_sgd_history = train_model(mode='dom', initial_model_state_dict=initial_model_state_dict_global)

torch.set_rng_state(prng_state) # Restore PRNG state
bulk_sgd_final_state, bulk_sgd_history = train_model(mode='bulk', initial_model_state_dict=initial_model_state_dict_global)


# --- Plotting ---
plt.figure(figsize=(18, 6))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(sgd_history['step'], sgd_history['loss'], label=f'SGD (H={MLP_HIDDEN_SIZE})', alpha=0.8, linewidth=2)
plt.plot(dom_sgd_history['step'], dom_sgd_history['loss'], label=f'Dom-SGD (H={MLP_HIDDEN_SIZE})', alpha=0.8, linewidth=2)
plt.plot(bulk_sgd_history['step'], bulk_sgd_history['loss'], label=f'Bulk-SGD (H={MLP_HIDDEN_SIZE})', alpha=0.8, linewidth=2)
plt.xlabel('Training Steps')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.yscale('log') # Log scale for loss as in paper

# Plot instantaneous chi_k
plt.subplot(1, 3, 2)
plt.plot(sgd_history['step'], sgd_history['chi_k'], label='SGD chi_k', alpha=0.6)
plt.plot(dom_sgd_history['step'], dom_sgd_history['chi_k'], label='Dom-SGD chi_k (orig grad)', alpha=0.6)
plt.plot(bulk_sgd_history['step'], bulk_sgd_history['chi_k'], label='Bulk-SGD chi_k (orig grad)', alpha=0.6)
plt.xlabel('Training Steps')
plt.ylabel('chi_k (Alignment)')
plt.title('Instantaneous chi_k')
plt.legend()
plt.grid(True)
plt.ylim(-0.05, 1.1)

# Plot EMA of chi_k
plt.subplot(1, 3, 3)
plt.plot(sgd_history['step'], sgd_history['ema_chi_k'], label='SGD EMA chi_k', alpha=0.8, linewidth=2)
plt.plot(dom_sgd_history['step'], dom_sgd_history['ema_chi_k'], label='Dom-SGD EMA chi_k', alpha=0.8, linewidth=2)
plt.plot(bulk_sgd_history['step'], bulk_sgd_history['ema_chi_k'], label='Bulk-SGD EMA chi_k', alpha=0.8, linewidth=2)
if EPOCHS_SGD_ONLY_FOR_CHIK_TRACKING > 0 and len(sgd_history['step']) > 0: # only plot if relevant
    first_hess_step_sgd = next((s for e, s in zip(np.cumsum([len(train_loader)] * TOTAL_EPOCHS), sgd_history['step']) if e // len(train_loader) >= EPOCHS_SGD_ONLY_FOR_CHIK_TRACKING), None)
    if first_hess_step_sgd is not None:
       pass # Could plot a vertical line here
plt.axhline(y=SWITCH_THRESHOLD, color='r', linestyle='--', label=f'Observed Switch Thr. ({SWITCH_THRESHOLD})')
plt.xlabel('Training Steps')
plt.ylabel('EMA chi_k')
plt.title('EMA of chi_k')
plt.legend()
plt.grid(True)
plt.ylim(-0.05, 1.1)


plt.tight_layout()
plt.suptitle(f"MLP H={MLP_HIDDEN_SIZE}, N_samples={N_SAMPLES}, BS={BATCH_SIZE}, LR={LEARNING_RATE}, K_dom={K_DOMINANT}", y=1.02)
plt.show()

