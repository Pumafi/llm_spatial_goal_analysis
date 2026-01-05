import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from grid_functions import generate_simple_example

# Metrics and vizualization functions

def reconstruction_accuracy(model, linear_model, test_loader, device):
    model.eval()
    linear_model.eval()

    n_layers = model.cfg.n_layers
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"]
            goal_grid = batch["goal_grid"].to(device)

            _, cache = model.run_with_cache(text)

            all_acts = [
                cache["resid_pre", layer][:, -1]
                for layer in range(n_layers)
            ]  # list: [batch, hidden]

            all_acts = torch.stack(all_acts, dim=1)

            all_acts = all_acts.reshape(all_acts.shape[0], -1)

            logits = linear_model(all_acts)  # [batch, 25]
            preds = torch.argmax(logits, dim=-1) # [batch]

            # Convert back to grid
            preds = F.one_hot(preds, num_classes=25).float()

            exact_match = torch.all(preds == goal_grid, dim=1)

            correct += exact_match.sum().item()
            total += goal_grid.size(0)

    return correct / total



def cross_entropy_vs_layer(model, linear_model, test_loader, device):
    model.eval()
    linear_model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    loss_sum = torch.zeros(n_layers, device=device)
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"]
            goal_grid = batch["goal_grid"].to(device)

            _, cache = model.run_with_cache(text)

            
            layer_acts = torch.stack([cache["resid_pre", l][:, -1] for l in range(n_layers)], dim=1)
            # [batch, n_layers, d_model]

            goal_grid = goal_grid.view(goal_grid.size(0), -1)

            targets = goal_grid.argmax(dim=-1)

            for l in range(n_layers):
                masked_acts = torch.zeros_like(layer_acts)
                masked_acts[:, l] = layer_acts[:, l]

                masked_acts = masked_acts.view(masked_acts.size(0), -1)

                logits = linear_model(masked_acts)

                loss = F.cross_entropy(logits, goal_grid, reduction="sum")
                loss_sum[l] += loss.item()

            total += targets.size(0)

    return (loss_sum / total).cpu()

def accuracy_vs_layer(model, linear_model, test_loader, device):
    model.eval()
    linear_model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    correct = torch.zeros(n_layers, device=device)
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"]
            goal_grid = batch["goal_grid"].to(device)

            _, cache = model.run_with_cache(text)

            layer_acts = torch.stack(
                [cache["resid_pre", l][:, -1] for l in range(n_layers)],
                dim=1
            )

            goal_grid = goal_grid.view(goal_grid.size(0), -1)

            for l in range(n_layers):
                masked_acts = torch.zeros_like(layer_acts)
                masked_acts[:, l] = layer_acts[:, l]

                masked_acts = masked_acts.view(masked_acts.size(0), -1)

                logits = linear_model(masked_acts)
                preds = torch.argmax(logits, dim=-1)
                preds = F.one_hot(preds, num_classes=25)

                exact = torch.all(preds == goal_grid, dim=1)
                correct[l] += exact.sum().item()

            total += goal_grid.size(0)

    return (correct / total).cpu()

def reconstruction_accuracy_with_agent_replace(model, linear_model, test_loader, new_position, device):
    model.eval()
    linear_model.eval()

    n_layers = model.cfg.n_layers
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"]
            texts = []
            for t in text:
                t_new = t.replace('Agent position: (0, 0)', f'Agent position: {new_position}')
                texts.append(t_new)
            text = texts

            goal_grid = batch["goal_grid"].to(device)

            # Run model and cache
            _, cache = model.run_with_cache(text)

            # Collect last-token resid_pre from all layers
            all_acts = [
                cache["resid_pre", layer][:, -1]
                for layer in range(n_layers)
            ]  # list of [batch, hidden]

            # [batch, layers, hidden]
            all_acts = torch.stack(all_acts, dim=1)

            # [batch, layers * hidden]
            all_acts = all_acts.reshape(all_acts.shape[0], -1)

            # Linear probe
            logits = linear_model(all_acts)          # [batch, 25]
            preds = torch.argmax(logits, dim=-1)     # [batch]

            # Convert to binary grid
            preds = F.one_hot(preds, num_classes=25).float()

            # Exact match per example
            exact_match = torch.all(preds == goal_grid, dim=1)

            correct += exact_match.sum().item()
            total += goal_grid.size(0)

    return correct / total

def r2_vs_layer(model, linear_model, test_loader, device):
    model.eval()
    linear_model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    loss_sum = torch.zeros(n_layers, device=device)
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"]
            goal_grid = batch["goal_grid"].to(device)

            _, cache = model.run_with_cache(text)

            layer_acts = torch.stack(
                [cache["resid_pre", l][:, -1] for l in range(n_layers)], dim=1)

            goal_grid = goal_grid.view(goal_grid.size(0), -1)

            #targets = goal_grid.argmax(dim=-1)

            for l in range(n_layers):
                masked_acts = torch.zeros_like(layer_acts)
                masked_acts[:, l] = layer_acts[:, l]

                masked_acts = masked_acts.view(masked_acts.size(0), -1)

                logits = linear_model(masked_acts)
                preds = torch.argmax(logits, dim=-1)
                preds = F.one_hot(preds, num_classes=25)
                
                loss = r2_score(preds.cpu(), goal_grid.cpu())
                loss_sum[l] += loss

    return loss_sum.cpu().numpy()


def compute_layer_cell_contributions(linear_model, model):
    W = linear_model.probe.weight.detach()  # [25, n_layers * d_model]

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    W = W.view(25, n_layers, d_model)

    contrib = torch.norm(W, dim=-1)

    return contrib.cpu().numpy()


def analyze_individual_neurons(model, linear_model, device):
    model.eval()
    linear_model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    activations = np.array([
        [[ [ 0. for _ in range(5) ] for _ in range(5) ]
         for _ in range(d_model)]
        for _ in range(n_layers)
    ])

    with torch.no_grad():
        for x in range(5):
            for y in range(5):
                if x == 0 and y == 0:
                    continue
                    
                data = generate_simple_example((0, 0), (x, y))
                text = data["inputs"]["simple"]

                _, cache = model.run_with_cache(text)

                for l in range(n_layers):
                    acts = cache["resid_pre", l][:, -1].squeeze(0)
                    for n in range(d_model):
                        activations[l, n, x, y] = acts[n].item()
    means = np.zeros((n_layers, d_model, 5, 5))
    means_for_std = np.zeros((n_layers, d_model))

    for l in range(n_layers):
        for n in range(d_model):
            for x in range(5):
                for y in range(5):
                    if x == 0 and y == 0:
                        continue
                        
                    # 23 because we also remove (0, 0) which is never a goal in our setup
                    means[l, n, x, y] = (np.sum(activations[l][n]) - activations[l][n][x][y]) / 23.
                        
    specificity = activations - means
    
    max_indices = np.empty((5, 5, 2), dtype=int)
    max_values = np.empty((5, 5))

    for x in range(5):
        for y in range(5):
            if x == 0 and y == 0:
                # don't need those
                max_indices[x, y] = [-1, -1]
                max_values[x, y] = np.nan
                continue

            slice_xy = specificity[:, :, x, y] # shape (L, N)

            flat_index = np.argmax(slice_xy)

            l, n = np.unravel_index(flat_index, slice_xy.shape)

            max_indices[x, y] = [l, n]
            max_values[x, y] = slice_xy[l, n]

    return max_indices, max_values, specificity