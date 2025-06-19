"""
SFT Trainer for supervised fine-tuning with contrastive loss
"""

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import wandb

from dna2vec_sft.sft_config_schema import SFTConfigSchema
from dna2vec_sft.sft_model import SFTModel
from dna2vec.tokenizer import BPTokenizer


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer using contrastive loss
    """
    
    def __init__(
        self,
        sft_model: SFTModel,
        similarity: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        scheduler: LRScheduler,
        device: torch.device,
        config: SFTConfigSchema,
        tokenizer: BPTokenizer,
        regularizer: float = 0,
        val_dataloader: Optional[DataLoader] = None,
        best_model_save_path: Optional[Path] = None,
    ):
        self.sft_model = sft_model
        self.loss = loss
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.similarity = similarity
        self.scheduler = scheduler
        self.config = config
        self.tokenizer = tokenizer
        self.regularizer = regularizer
        self.best_model_save_path = best_model_save_path or Path("./best_model_checkpoints")
        
        self.training_config = config.training_config
        
        # Early stopping variables
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None
        
        # # Trainable loss weights (initialized with reasonable values) # TODO: Maybe make them trainable?
        # self.alpha = nn.Parameter(torch.tensor(0.5))  # Fragment losses weight
        # self.beta = nn.Parameter(torch.tensor(0.25))  # Triplet loss weight  
        # self.gamma = nn.Parameter(torch.tensor(0.25)) # Directional loss weight
        
        # loss_weight_params = [self.alpha, self.beta, self.gamma]
        # self.optimizer.add_param_group({'params': loss_weight_params})
        
        # Move model to device
        self.sft_model.to(device)
        
    def dict_to_device(self, d: dict):
        """Move dictionary tensors to device"""
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(self.device)
    
    def forward_pass(self, batch_data: dict, pool_type: str = "mean") -> torch.Tensor:
        """
        Forward pass through SFT model
        
        Args:
            batch_data: Dictionary with input_ids and attention_mask
            pool_type: Pooling type for the model
            
        Returns:
            Model output embeddings
        """
        return self.sft_model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            pool_type=pool_type
        )
    
    def compute_contrastive_loss(self, read_emb_1: torch.Tensor, read_emb_2: torch.Tensor) -> torch.Tensor:
        """
        Computes a triplet margin loss to enforce local positional geometry within a fragment.

        Justification:
        While the contrastive loss separates different fragments, this triplet loss teaches
        the model about the internal structure of a single fragment. Given three reads sampled
        in order from a fragment (r1, r2, r3), this loss enforces that the distance in the
        embedding space reflects the physical distance on the genome. Specifically, it ensures
        d(h(r1), h(r2)) < d(h(r1), h(r3)).

        This moves beyond simple similarity clustering and teaches the model about the relative
        "betweenness" of reads. The symmetrical implementation (calculating loss from both r1's
        and r3's perspectives) makes this understanding more robust. This explicit geometric
        constraint is a crucial step towards creating the "order-preserving 1D manifold"
        observed as an emergent property in our previous work. A model that
        understands this local geometry is better prepared for more complex tasks that rely on
        read order, such as de novo genome assembly.
        """
        # Compute similarity matrix
        sim_matrix = self.similarity(
            read_emb_1.unsqueeze(1), read_emb_2.unsqueeze(0)
        )
        
        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(sim_matrix.size(0)).long().to(self.device)
        
        return self.loss(sim_matrix, labels)
    
    def compute_triplet_margin_loss(self, read_emb_1: torch.Tensor, read_emb_2: torch.Tensor, read_emb_3: torch.Tensor) -> torch.Tensor:
        """
        Computes a triplet margin loss to enforce local positional geometry within a fragment.

        Justification:
        While the contrastive loss separates different fragments, this triplet loss teaches
        the model about the internal structure of a single fragment. Given three reads sampled
        in order from a fragment (r1, r2, r3), this loss enforces that the distance in the
        embedding space reflects the physical distance on the genome. Specifically, it ensures
        d(h(r1), h(r2)) < d(h(r1), h(r3)).

        This moves beyond simple similarity clustering and teaches the model about the relative
        "betweenness" of reads. The symmetrical implementation (calculating loss from both r1's
        and r3's perspectives) makes this understanding more robust. This explicit geometric
        constraint is a crucial step towards creating the "order-preserving 1D manifold"
        observed as an emergent property in our previous work. A model that
        understands this local geometry is better prepared for more complex tasks that rely on
        read order, such as de novo genome assembly.
        """
        # define the triplet margin loss
        triplet_margin_loss = nn.TripletMarginLoss(margin=0.25, p=2.0, eps=1e-6)
        loss_1 = triplet_margin_loss(read_emb_1, read_emb_2, read_emb_3)
        loss_2 = triplet_margin_loss(read_emb_3, read_emb_2, read_emb_1)
        
        return (loss_1 + loss_2) / 2
    
    def compute_directional_loss(self, read_emb_1: torch.Tensor, read_emb_2: torch.Tensor, read_emb_3: torch.Tensor) -> torch.Tensor:
        """
        Computes a directional loss to create a consistent and navigable vector space.

        Justification:
        This is the most sophisticated of the three losses, designed to explicitly train the
        "emergent geometry" noted in Figure 4 of our DNA-ESA paper. The goal is to
        make the vector difference between embeddings meaningful. The vector `h(r2) - h(r1)`
        should represent a consistent "step" along the genome's 1D manifold.

        By penalizing `1 - cosine_similarity` between consecutive step vectors (v1 and v2),
        we teach the model that moving from r1 to r2 is directionally similar to moving
        from r2 to r3. This creates a predictable structure in the embedding space, making it
        less like a simple map of clusters and more like a true coordinate system. Such a
        navigable space is essential for the "walk along the ID manifold" concept proposed
        for de novo assembly. It provides a powerful signal for ordering reads even
        in the absence of direct sequence overlap, moving the model's capability closer
        to this challenging future goal.
        """
        # define the directional loss
        v1 = read_emb_2 - read_emb_1 # shape: (batch_size, embedding_dim)
        v2 = read_emb_3 - read_emb_2 # shape: (batch_size, embedding_dim)
        
        sim = torch.cosine_similarity(v1, v2, dim=1) # shape: (batch_size, 1)
        loss = (1 - sim).mean() # shape: (1,)
        return loss # shape: (1,)
    
    def train_step(self, batch) -> dict:
        """
        Single training step with multiple contrastive objectives
        
        Returns:
            Dictionary with loss information
        """
        fragment = batch.fragment
        read1, read2 = batch.read
        
        # Move to device
        self.dict_to_device(fragment)
        self.dict_to_device(read1)
        self.dict_to_device(read2)
        
        # Forward pass for all inputs
        fragment_embedding = self.forward_pass(fragment, self.training_config.pool_type)
        read1_embedding = self.forward_pass(read1, self.training_config.pool_type)
        read2_embedding = self.forward_pass(read2, self.training_config.pool_type)
        
        # Loss 1: Fragment-Read1 contrastive loss (original objective)
        loss_fragment_read1 = self.compute_contrastive_loss(fragment_embedding, read1_embedding)
        
        # Loss 2: Fragment-Read2 contrastive loss (original objective) 
        loss_fragment_read2 = self.compute_contrastive_loss(fragment_embedding, read2_embedding)
        
        # Loss 3: Read1-Read2 contrastive loss (make reads from same fragment close)
        loss_read1_read2 = self.compute_contrastive_loss(read1_embedding, read2_embedding)
        
        # TODO: We want to enforce distance between reads based on position in the fragment -> Check that
        # TODO: Maybe we can add Read3 since we can define triangle
        # if read 1 precedes read 2, r2-r1 direction 
        
        # Combine losses with weights
        fragment_loss_weight = 0.4  # Weight for fragment-read losses
        read_read_loss_weight = 0.6  # Weight for read-read loss (can tune this)
        
        total_loss = (
            fragment_loss_weight * (loss_fragment_read1 + loss_fragment_read2) +
            read_read_loss_weight * loss_read1_read2
        )
        
        return {
            "total_loss": total_loss,
            "loss_fragment_read1": loss_fragment_read1,
            "loss_fragment_read2": loss_fragment_read2,
            "loss_read1_read2": loss_read1_read2,
        }

    def train_step_triplet(self, batch) -> dict:
        """
        Single training step with multiple contrastive objectives
        
        Returns:
            Dictionary with loss information
        """
        fragment = batch.fragment
        read1, read2, read3 = batch.read
        
        # Move to device
        self.dict_to_device(fragment)
        self.dict_to_device(read1)
        self.dict_to_device(read2)
        self.dict_to_device(read3)
        
        # Forward pass for all inputs
        # fragment_embedding = self.forward_pass(fragment, self.training_config.pool_type)
        read1_embedding = self.forward_pass(read1, self.training_config.pool_type)
        read2_embedding = self.forward_pass(read2, self.training_config.pool_type)
        read3_embedding = self.forward_pass(read3, self.training_config.pool_type)
        
        # Loss 1: Fragment-Read1 contrastive loss (original objective)
        # loss_fragment_read1 = self.compute_contrastive_loss(fragment_embedding, read1_embedding)
        
        # Loss 2: Fragment-Read2 contrastive loss (original objective) 
        # loss_fragment_read2 = self.compute_contrastive_loss(fragment_embedding, read2_embedding)
        
        # Loss 3: Fragment-Read3 contrastive loss (original objective)
        # loss_fragment_read3 = self.compute_contrastive_loss(fragment_embedding, read3_embedding)
        
        # Loss 4: Triplet margin loss
        loss_triplet = self.compute_triplet_margin_loss(read1_embedding, read2_embedding, read3_embedding)
        
        # Loss 5: Directional loss
        # loss_directional = self.compute_directional_loss(read1_embedding, read2_embedding, read3_embedding)
        
        # Combine losses with weights
        total_loss = (
            # self.training_config.alpha * (loss_fragment_read1 + loss_fragment_read2 + loss_fragment_read3) / 3 +
            # self.training_config.beta * loss_triplet
            # self.training_config.gamma * loss_directional
            loss_triplet
        )
        
        return {
            "total_loss": total_loss,
            # "loss_fragment_read1": loss_fragment_read1,
            # "loss_fragment_read2": loss_fragment_read2,
            # "loss_fragment_read3": loss_fragment_read3,
            "loss_triplet": loss_triplet,
            # "loss_directional": loss_directional,
        }

    
    def validate(self) -> float:
        """
        Validation step
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return float("inf")
            
        self.sft_model.eval()
        total_val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                loss_info = self.train_step_triplet(batch)
                # loss_info = self.train_step(batch)
                total_val_loss += loss_info["total_loss"].item()
                num_batches += 1
                
                # Limit validation to avoid taking too long
                if num_batches >= 50:
                    break
        
        avg_val_loss = total_val_loss / max(num_batches, 1)
        self.sft_model.train()
        
        return avg_val_loss
    
    def should_stop_early(self, val_loss: float) -> bool:
        """
        Check if we should stop training early.
        Saves model to disk when it achieves the best validation loss.
        """
        if val_loss < self.best_val_loss - self.training_config.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            # Save best model state in memory
            self.best_model_state = {
                'sft_model_state_dict': self.sft_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss
            }
            
            # Save best model to disk
            try:
                self.save_model(self.best_model_save_path)
                print(f"Saved best model with val_loss: {val_loss:.4f} to {self.best_model_save_path}")
            except Exception as e:
                print(f"Warning: Failed to save best model: {e}")
            
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.training_config.patience
    
    def train(
        self,
        max_steps: Optional[int] = None,
        log_interval: int = 100,
        val_interval: int = 1000,
    ) -> None:
        """
        Main training loop
        
        Args:
            max_steps: Maximum number of training steps
            log_interval: Steps between logging
            val_interval: Steps between validation
        """
        if max_steps is None:
            max_steps = self.training_config.max_steps
            
        self.sft_model.train()
        
        for step, batch in enumerate(self.train_dataloader):
            if step >= max_steps:
                break
                
            # Training step
            loss_info = self.train_step_triplet(batch) #TODO: comment this out
            # loss_info = self.train_step(batch)
            total_loss = loss_info["total_loss"]
            
            # Backward pass
            total_loss = total_loss / self.training_config.accumulation_steps
            total_loss.backward()
            
            # Optimizer step
            if (step + 1) % self.training_config.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.sft_model.parameters(), 
                    self.training_config.max_grad_norm
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            # Logging
            if step % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                
                log_dict = {
                    "step": step,
                    "lr": current_lr,
                    "total_loss": total_loss.item(),
                }
                
                # Add individual losses if available
                for key, value in loss_info.items():
                    if key != "total_loss":
                        log_dict[key] = value.item()
                
                wandb.log(log_dict)
                
                print(f"Step {step}, Loss: {total_loss.item():.4f}, LR: {current_lr:.2e}")
            
            # Validation
            if step % val_interval == 0 and step > 0:
                val_loss = self.validate()
                wandb.log({"val_loss": val_loss, "step": step})
                
                print(f"Step {step}, Val Loss: {val_loss:.4f}")
                
                # Early stopping check
                if self.should_stop_early(val_loss):
                    print(f"Early stopping at step {step}")
                    break
        
        # Load best model if we have one
        if self.best_model_state is not None:
            print(f"Loading best model with val_loss: {self.best_model_state['val_loss']:.4f}")
            self.sft_model.load_state_dict(self.best_model_state['sft_model_state_dict'])
    
    def save_model(self, save_path: Path) -> None:
        """
        Save the trained model
        
        Args:
            save_path: Path to save the model
        """
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'sft_model_state_dict': self.sft_model.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        torch.save(checkpoint, save_path / "sft_model.pt")
        print(f"Model saved to {save_path / 'sft_model.pt'}")
    
    @staticmethod
    def load_model(load_path: Path, device: torch.device) -> 'SFTTrainer':
        """
        Load a trained SFT model
        
        Args:
            load_path: Path to load the model from
            device: Device to load the model on
            
        Returns:
            Loaded SFTTrainer instance
        """
        checkpoint = torch.load(load_path / "sft_model.pt", map_location=device)
        
        config = checkpoint['config']
        
        # Recreate model (this would need to be implemented properly)
        # For now, returning None as this would require more complex logic
        raise NotImplementedError("Model loading not fully implemented") 