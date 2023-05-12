"""
The trainer module contains the Trainer class, which is responsible for training the model contrastive learning
"""

from typing import Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader


class ContrastiveTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.device = device

    def train(
        self,
        max_steps: Optional[int] = None,
        log_interval: int = 100,
    ) -> None:
        """
        Train the model for the specified number of steps

        Args:
            steps: Number of steps to train the model for
            log_interval: Number of steps after which to log the training loss
        """
        self.model.train()
        for step, batch in enumerate(self.train_dataloader):
            if max_steps is not None and step >= max_steps:
                break

            self.optimizer.zero_grad()
            # we could consider using a label matrix instead
            # e.g. see:
            # https://www.kaggle.com/code/debarshichanda/pytorch-supervised-contrastive-learning
            # but let us start with the simple approach first
            y_1 = self.model(batch)
            y_2 = self.model(batch)
            loss = self.criterion(y_1, y_2)
            loss.backward()
            self.optimizer.step()

            if step % log_interval == 0:
                wandb.log({"loss": loss.item()})


