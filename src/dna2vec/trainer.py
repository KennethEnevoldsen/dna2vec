"""
The trainer module contains the Trainer class, which is responsible for training the model contrastive learning
"""

from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import wandb
from dna2vec.config_schema import ConfigSchema
from dna2vec.dataset import collate_fn
from dna2vec.model import model_from_config

from dna2vec.tokenizer import BPTokenizer


class ContrastiveTrainer:
    def __init__(
        self,
        encoder: nn.Module,
        pooling: nn.Module,
        similarity: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        scheduler: LRScheduler,
        device: torch.device,
        config: ConfigSchema,
        tokenizer: BPTokenizer,
        regularizer: float = 0,  # Regularizer weight, set to 0 to disable
    ):
        self.encoder = encoder
        self.loss = loss
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.device = device
        self.similarity = similarity
        self.pooling = pooling
        self.scheduler = scheduler
        self.config = config
        self.tokenizer = tokenizer
        self.best_loss = float("inf")
        self.regularizer = regularizer

        self.training_config = config.training_config

    def model_to_device(self, device: Optional[torch.device] = None) -> None:
        """
        Move the model to the specified device

        Args:
            device: Device to move the model to
        """
        if device is not None:
            self.device = device
        self.encoder.to(self.device)

    def pooling_to_device(self, device: Optional[torch.device] = None) -> None:
        """
        Move the pooling layer to the specified device

        Args:
            device: Device to move the pooling layer to
        """
        if device is not None:
            self.device = device
        self.pooling.to(self.device)

    def dict_to_device(self, d: dict):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(self.device)

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

        self.model_to_device()
        self.encoder.train()
        for step, sub_ex in enumerate(self.train_dataloader):

            if max_steps is not None and step >= max_steps:
                break

            fragment = sub_ex.fragment
            self.dict_to_device(fragment)

            last_hidden_fragment = self.encoder(**fragment)  # long sequence

            # THE cls token is the first token

            if self.training_config.pool_type == "cls":
                fragment_embedding = last_hidden_fragment[:, 0, :]
                self.pooling_to_device()
                fragment_embedding = self.pooling(fragment_embedding)

            elif self.training_config.pool_type == "mean":
                fragment_embedding = self.pooling(
                    last_hidden_fragment, attention_mask=fragment["attention_mask"]
                )

            else:
                raise ValueError(
                    f"Pooling type {self.training_config.pool_type} not implemented"
                )

            if sub_ex.read_regularization:
                read1, read2 = sub_ex.read
                self.dict_to_device(read1)
                self.dict_to_device(read2)
                last_hidden_read2 = self.encoder(**read2)  # subsequence
                # read2_embedding_cls = last_hidden_read2[:, 0, :]

                if self.training_config.pool_type == "cls":
                    read2_embedding = last_hidden_read2[:, 0, :]
                    read2_embedding = self.pooling(read2_embedding)

                elif self.training_config.pool_type == "mean":

                    read2_embedding = self.pooling(
                        last_hidden_read2, attention_mask=read2["attention_mask"]
                    )

            else:
                read1 = sub_ex.read
                self.dict_to_device(read1)

            last_hidden_read1 = self.encoder(**read1)  # subsequence

            if self.training_config.pool_type == "cls":
                read1_embedding = last_hidden_read1[:, 0, :]
                read1_embedding = self.pooling(read1_embedding)

            elif self.training_config.pool_type == "mean":
                read1_embedding = self.pooling(
                    last_hidden_read1, attention_mask=read1["attention_mask"]
                )

            # batch x 1 x embedding y1 , 1 x batch x embedding y2 - > batch x batch
            sim_fragment_read1 = self.similarity(
                fragment_embedding.unsqueeze(1), read1_embedding.unsqueeze(0)
            )  # outer-product

            sim_fragment_read2 = self.similarity(
                fragment_embedding.unsqueeze(1), read2_embedding.unsqueeze(0)
            )

            labels1 = torch.arange(sim_fragment_read1.size(0)).long().to(self.device)
            labels2 = torch.arange(sim_fragment_read2.size(0)).long().to(self.device)

            loss_read1 = self.loss(sim_fragment_read1, labels1)
            # loss_read2 = self.loss(sim_fragment_read2, labels2)

            # loss = loss_read1 + loss_read2
            loss = loss_read1

            # Compute similarity score between y2 and y3 and add to loss
            # if sub_ex.read_regularization:
            #     sim_read_read = self.similarity(
            #         read1_embedding.unsqueeze(1), read2_embedding.unsqueeze(0)
            #     )
            #     # Take the diagonal of the similarity matrix and sum the terms to add to the loss
            #     diag_sim_read_read = torch.diag(sim_read_read)
            #     normalized_read_read = torch.sum(
            #         diag_sim_read_read
            #     ) / diag_sim_read_read.size(0)
            #     max_sim = 0.8  # Hyperparameter that determines the maximum similarity score between the two reads
            #     reg_loss = torch.abs(max_sim - normalized_read_read)
            #     loss += reg_loss * self.regularizer

            loss = loss / self.training_config.accumulation_steps
            loss.backward()
            if (step + 1) % self.training_config.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.training_config.max_grad_norm)  # type: ignore

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            if step % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "total_loss": loss,
                        "step": step,
                        "lr": current_lr,
                        "loss_read1": loss_read1,
                    }
                )

            # save the model
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_to_disk()

            # trying to resolve the CUDA out of memory error
            if step % 1000 == 0:
                torch.cuda.empty_cache()
            # delete the tensors: https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3?u=nagabhushansn95
            del (
                fragment_embedding,
                read1_embedding,
                sim_fragment_read1,
                sim_fragment_read2,
                labels1,
                labels2,
                loss,
                loss_read1,
            )

            if sub_ex.read_regularization:
                del (
                    read2_embedding,
                    last_hidden_read2,
                )
                for k, v in read2.items():
                    del v

                del read2

            for k, v in fragment.items():
                del v
            for k, v in read1.items():
                del v

            del last_hidden_fragment, last_hidden_read1, read1, fragment

    def save_to_disk(self, path: Optional[str] = None):
        if path is None:
            save_path = self.config.training_config.save_path
        else:
            save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # record model state train/eval
        state = self.encoder.training

        save_dict = {
            "model": self.encoder.state_dict(),
            "pooling": self.pooling.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }

        torch.save(
            save_dict,
            save_path
            / (
                "chromosome_2"
                + str(self.training_config.regularizer)
                + "_pool_1020_ART_READS"
                + self.training_config.pool_type
                + "_checkpoint.pt"
            ),
        )

    @staticmethod
    def load_from_disk(path: str) -> "ContrastiveTrainer":

        checkpoint = torch.load(path)
        config = checkpoint["config"]
        encoder, pooling, tokenizer = model_from_config(config.model_config)
        encoder.load_state_dict(checkpoint["model"])
        optimizer_state = checkpoint["optimizer"]
        scheduler_state = checkpoint["scheduler"]

        dataset_kwargs = config.dataset_config.dict()
        dataset_fn = dataset_kwargs.pop("dataset")
        dataset = dataset_fn(**dataset_kwargs)

        _collate_fn = partial(collate_fn, tokenizer=tokenizer)

        train_cfg = config.training_config
        dataloader = DataLoader(
            dataset, batch_size=train_cfg.batch_size, collate_fn=_collate_fn
        )

        sim = train_cfg.similarity(temperature=train_cfg.temperature)

        # recreate optimizer and scheduler states
        opt_kwargs = train_cfg.optimizer_config.dict()
        optimizer_ = train_cfg.optimizer(encoder.parameters(), **opt_kwargs)
        optimizer_.load_state_dict(optimizer_state)
        scheduler_ = train_cfg.scheduler(
            optimizer_, **train_cfg.scheduler_config.dict()
        )
        scheduler_.load_state_dict(scheduler_state)

        trainer = ContrastiveTrainer(
            encoder=encoder,
            pooling=pooling,
            loss=train_cfg.loss,
            optimizer=optimizer_,
            scheduler=scheduler_,
            device=train_cfg.device,
            train_dataloader=dataloader,
            similarity=sim,
            config=config,
            tokenizer=tokenizer,
            regularizer=train_cfg.regularizer,
        )

        return trainer
