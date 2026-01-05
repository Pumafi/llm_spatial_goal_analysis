import torch
import torch.nn as nn
import tqdm

class LinearProbe(nn.Module):
    def __init__(self, activation_dim, num_classes=25):
        super().__init__()
        self.probe = nn.Linear(activation_dim, num_classes)

    def forward(self, activation):
        y = self.probe(activation)
        return y



def switch_tqdm(loader, desc, use_tqdm):
    return tqdm.tqdm(loader, desc=desc) if use_tqdm else loader

def train_linear_probe_all_layers(
    linear_model,
    llm_model,
    train_loader,
    device,
    num_epochs,
    history={"loss": []},
    use_tqdm: bool = True,
):
    linear_model.to(device)
    optimizer = torch.optim.Adam(
        linear_model.parameters(), lr=1e-3)#, weight_decay=1e-6
    #)

    n_layers = llm_model.cfg.n_layers
    linear_model.train()

    for epoch in range(num_epochs):
        loader = switch_tqdm(
            train_loader, f"Epoch {epoch + 1}", use_tqdm
        )
        avg_loss = 0.0

        for batch in loader:
            texts = batch["text"]
            goal_grids = batch["goal_grid"].to(device)

            with torch.no_grad():
                _, cache = llm_model.run_with_cache(texts)

                # Collect activations from all layers
                all_acts = [
                    cache["resid_pre", layer][:, -1]
                    for layer in range(n_layers)
                ]  # list of [batch, hidden]

                
                # [batch, layers, hidden]
                all_acts = torch.stack(all_acts, dim=1)

                # [batch, layers * hidden]
                # # Then authors collectd hidden states from all layers of the last token in the template.
                all_acts = all_acts.reshape(all_acts.shape[0], -1).to(device)

            optimizer.zero_grad()
            preds = linear_model(all_acts)
            loss = nn.CrossEntropyLoss()(preds, goal_grids)
            loss.backward()
            optimizer.step()

            if use_tqdm:
                loader.set_postfix(loss=loss.item())

            avg_loss += loss.item()

        avg_loss /= len(train_loader)
        history["loss"].append(avg_loss)

    return history
