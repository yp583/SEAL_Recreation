"""
ReSTEM Optimizer Implementation

Implements the ReSTEM (Rejection Sampling + Supervised Fine-Tuning) approach
for optimizing self-edit generation in SEAL.

ReSTEM can be viewed as an EM procedure:
- E-step: Sample candidate outputs from current model policy  
- M-step: Reinforce only positive-reward samples through SFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import logging
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


class ReSTEMDataset(Dataset):
    """Dataset for ReSTEM training containing positive self-edits"""
    
    def __init__(
        self,
        self_edits: List[str],
        contexts: List[Any],
        rewards: List[float],
        tokenizer: Any,
        reward_threshold: float = 0.0,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Filter for positive rewards only (rejection sampling)
        self.positive_samples = []
        for edit, context, reward in zip(self_edits, contexts, rewards):
            if reward > reward_threshold:
                self.positive_samples.append({
                    "self_edit": edit,
                    "context": context,
                    "reward": reward
                })
                
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ReSTEM dataset: {len(self.positive_samples)} positive samples "
                        f"out of {len(self_edits)} total samples")
    
    def __len__(self):
        return len(self.positive_samples)
    
    def __getitem__(self, idx):
        sample = self.positive_samples[idx]
        
        # Create training text combining context and self-edit
        text = f"Context: {sample['context']}\nSelf-Edit: {sample['self_edit']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),  # For causal LM loss
            "reward": torch.tensor(sample["reward"], dtype=torch.float32)
        }


class ReSTEMOptimizer:
    """
    ReSTEM optimizer implementing rejection sampling + supervised fine-tuning.
    
    This implements the simplified RL approach used in SEAL:
    1. Sample self-edits from current policy
    2. Filter for positive rewards (rejection sampling)  
    3. Fine-tune on positive samples only (supervised learning)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        num_epochs: int = 1,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.logger = logger or logging.getLogger(__name__)
        
        # Training statistics
        self.update_count = 0
        self.total_positive_samples = 0
        self.update_history = []
        
    def update(
        self,
        self_edits: List[str],
        rewards: List[float],
        contexts: List[Any],
        reward_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Update model using ReSTEM approach.
        
        Args:
            self_edits: Generated self-edits
            rewards: Corresponding rewards
            contexts: Original contexts
            reward_threshold: Minimum reward for positive samples
            
        Returns:
            Update statistics
        """
        self.update_count += 1
        
        # Create dataset with positive samples only
        dataset = ReSTEMDataset(
            self_edits, contexts, rewards, self.tokenizer, reward_threshold
        )
        
        if len(dataset) == 0:
            self.logger.warning("No positive samples for ReSTEM update")
            return {
                "positive_samples": 0,
                "loss": 0.0,
                "skipped": True
            }
        
        self.total_positive_samples += len(dataset)
        
        # Create data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        num_training_steps = len(dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=min(self.warmup_steps, num_training_steps // 4),
            num_training_steps=num_training_steps
        )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            for batch in tqdm(dataloader, desc=f"ReSTEM Update {self.update_count}, Epoch {epoch+1}"):
                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            total_loss += epoch_loss
            
        avg_loss = total_loss / max(num_batches, 1)
        
        # Update statistics
        update_stats = {
            "positive_samples": len(dataset),
            "total_samples": len(self_edits),
            "positive_ratio": len(dataset) / len(self_edits),
            "loss": avg_loss,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "skipped": False
        }
        
        self.update_history.append(update_stats)
        
        self.logger.info(f"ReSTEM update {self.update_count}: "
                        f"{update_stats['positive_samples']}/{update_stats['total_samples']} "
                        f"positive samples, loss={avg_loss:.4f}")
        
        return update_stats
    
    def _collate_fn(self, batch):
        """Custom collate function for ReSTEM dataset"""
        # Stack tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        rewards = torch.stack([item["reward"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rewards": rewards
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        if not self.update_history:
            return {"num_updates": 0}
            
        recent_updates = self.update_history[-10:]  # Last 10 updates
        
        stats = {
            "num_updates": self.update_count,
            "total_positive_samples": self.total_positive_samples,
            "recent_avg_loss": sum(u["loss"] for u in recent_updates) / len(recent_updates),
            "recent_avg_positive_ratio": sum(u["positive_ratio"] for u in recent_updates) / len(recent_updates),
            "update_history": self.update_history
        }
        
        return stats
    
    def save_state(self, path: str):
        """Save optimizer state"""
        state = {
            "update_count": self.update_count,
            "total_positive_samples": self.total_positive_samples,
            "update_history": self.update_history,
            "config": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "warmup_steps": self.warmup_steps,
                "max_grad_norm": self.max_grad_norm
            }
        }
        torch.save(state, path)
        self.logger.info(f"ReSTEM state saved to {path}")
    
    def load_state(self, path: str):
        """Load optimizer state"""
        state = torch.load(path, map_location=self.model.device)
        self.update_count = state["update_count"]
        self.total_positive_samples = state["total_positive_samples"]
        self.update_history = state["update_history"]
        self.logger.info(f"ReSTEM state loaded from {path}")


class ReSTEMAnalyzer:
    """Utility class for analyzing ReSTEM training dynamics"""
    
    @staticmethod
    def analyze_reward_distribution(rewards: List[float]) -> Dict[str, float]:
        """Analyze distribution of rewards"""
        if not rewards:
            return {}
            
        rewards_tensor = torch.tensor(rewards)
        
        return {
            "mean": rewards_tensor.mean().item(),
            "std": rewards_tensor.std().item(),
            "min": rewards_tensor.min().item(),
            "max": rewards_tensor.max().item(),
            "median": rewards_tensor.median().item(),
            "positive_fraction": (rewards_tensor > 0).float().mean().item()
        }
    
    @staticmethod
    def compute_rejection_sampling_efficiency(
        rewards: List[float], 
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """Compute efficiency metrics for rejection sampling"""
        total_samples = len(rewards)
        positive_samples = sum(1 for r in rewards if r > threshold)
        
        return {
            "total_samples": total_samples,
            "positive_samples": positive_samples,
            "acceptance_rate": positive_samples / max(total_samples, 1),
            "rejection_rate": 1 - (positive_samples / max(total_samples, 1)),
            "efficiency": positive_samples / max(total_samples, 1)  # Same as acceptance rate
        }
    
    @staticmethod
    def plot_training_dynamics(update_history: List[Dict[str, Any]]):
        """Plot ReSTEM training dynamics (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            if not update_history:
                print("No update history to plot")
                return
                
            updates = range(1, len(update_history) + 1)
            losses = [u["loss"] for u in update_history]
            positive_ratios = [u["positive_ratio"] for u in update_history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot loss
            ax1.plot(updates, losses, 'b-', label='Loss')
            ax1.set_xlabel('Update')
            ax1.set_ylabel('Loss')
            ax1.set_title('ReSTEM Training Loss')
            ax1.grid(True)
            
            # Plot positive ratio
            ax2.plot(updates, positive_ratios, 'r-', label='Positive Ratio')
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Positive Sample Ratio')
            ax2.set_title('Rejection Sampling Efficiency')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")