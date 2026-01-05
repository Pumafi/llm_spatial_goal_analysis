import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader


class GridTextDataset(Dataset):
    def __init__(
        self,
        full_dataset,
        mode="train",
        train_ratio=1.0,
        valid_ratio=0.0
    ):
        """
        full_dataset: list of dicts (your original data)
        mode: 'train' | 'valid' | 'test'
        """

        self.samples = []

        type_of_text = [
            'cardinal',
            'action',
            'simple', 
        ]

        # Build flat sample list
        for datapoint in full_dataset:
            inputs = datapoint['inputs']
            goal_grid = torch.tensor(
                datapoint['output']['goal_grid'], dtype=torch.float32
            )
            agent_grid = torch.tensor(
                datapoint['output']['agent_grid'], dtype=torch.float32
            )

            for t_type in type_of_text:
                text = inputs.get(t_type)
                if text is not None:
                    self.samples.append({
                        "text_type": t_type,
                        "text": text.replace('"', '').replace('\\n', '\n'),
                        "goal_grid": goal_grid,
                        "agent_grid": agent_grid
                    })

        # ---------- split logic (like use_index) ----------
        n = len(self.samples)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))

        if mode == "train":
            self.use_index = np.arange(0, train_end)
        elif mode == "valid":
            self.use_index = np.arange(train_end, valid_end)
        elif mode == "test":
            self.use_index = np.arange(valid_end, n)
        else:
            raise ValueError("mode must be train, valid, or test")

    def __len__(self):
        return len(self.use_index)

    def __getitem__(self, idx):
        sample = self.samples[self.use_index[idx]]

        return {
            "text_type": sample["text_type"],
            "text": sample["text"],
            "goal_grid": sample["goal_grid"],
            "agent_grid": sample["agent_grid"]
        }

def gridtext_collate_fn(batch):
    return {
        "text_type": [b["text_type"] for b in batch],
        "text": [b["text"] for b in batch],
        "goal_grid": torch.stack([b["goal_grid"] for b in batch]),
        "agent_grid": torch.stack([b["agent_grid"] for b in batch]),
    }


def get_dataloader(
    full_dataset,
    batch_size=8,
    num_workers=4
):
    train_dataset = GridTextDataset(full_dataset, mode="train")
    valid_dataset = GridTextDataset(full_dataset, mode="valid")
    test_dataset  = GridTextDataset(full_dataset, mode="test")

    print("Training dataset len:", len(train_dataset))
    print("Validation dataset len:", len(valid_dataset))
    print("Test dataset len:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=gridtext_collate_fn,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=gridtext_collate_fn,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=gridtext_collate_fn,
        num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader
