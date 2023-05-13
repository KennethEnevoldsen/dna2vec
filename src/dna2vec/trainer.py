"""
The trainer module contains the Trainer class, which is responsible for training the model contrastive learning
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb


class ContrastiveTrainer:
    def __init__(
        self,
        encoder: nn.Module,
        pooling: nn.Module,
        similarity: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        device: torch.device,
    ):
        self.encoder = encoder
        self.loss = loss
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.device = device
        self.similarity = similarity
        self.pooling = pooling

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
        self.encoder.train()
        for step, (x_1, x_2) in enumerate(self.train_dataloader):
            if max_steps is not None and step >= max_steps:
                break

            self.optimizer.zero_grad()

            last_hidden_x_1 = self.encoder(**x_1)
            last_hidden_x_2 = self.encoder(**x_2)
            y_1 = self.pooling(last_hidden_x_1, attention_mask=x_1["attention_mask"])
            y_2 = self.pooling(last_hidden_x_2, attention_mask=x_2["attention_mask"])

            # Calculate similarity
            y_1 = y_1.rename(None)  # names: [batch, embedding]
            y_2 = y_2.rename(None)  # names: [batch, embedding]
            sim = self.similarity(y_1.unsqueeze(1), y_2.unsqueeze(0))

            labels = torch.arange(sim.size(0)).long().to(self.device)

            loss = self.loss(sim, labels)
            loss.backward()
            self.optimizer.step()

            if step % log_interval == 0:
                wandb.log({"loss": loss.item()})
