import torch
import torch.nn.functional as F
import numpy as np
from grid_functions import generate_simple_example

def collect_activations_table(model, agent_position):
    model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    activations = np.zeros((5, 5, d_model * n_layers))

    # Fixed agent
    with torch.no_grad():
        for x in range(5):
            for y in range(5):
                if x == 0 and y == 0:
                    continue
                    
                data = generate_simple_example(agent_position, (x, y))
                text = data["inputs"]["simple"]

                _, cache = model.run_with_cache(text)
                all_acts = [
                    cache["resid_pre", l][:, -1]
                    for l in range(n_layers)
                ]  # list of [1, hidden]
                all_acts = torch.stack(all_acts, dim=1)

                all_acts = all_acts.reshape(all_acts.shape[0], -1)

                activations[x, y] = all_acts.detach().cpu().numpy()

    return activations

def collect_all_activations(model, test_loader, device):
    # For PCA
    model.eval()

    n_layers = model.cfg.n_layers

    x = []
    y_goal = []
    y_agent = []

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
            x.append(all_acts.cpu())
            y.append(goal_grid.cpu())

    return torch.cat(x, dim=0), torch.cat(y, dim=0)