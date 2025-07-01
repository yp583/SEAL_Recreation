"""
Core SEAL Framework Implementation

Implements Algorithm 1: Self-Adapting LLMs (SEAL) with nested RL and inner update loops.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import logging
from tqdm import tqdm
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from .self_edit import SelfEditGenerator
from .restem import ReSTEMOptimizer
from ..models.lora_wrapper import LoRAWrapper


@dataclass
class SEALConfig:
    """Configuration for SEAL framework"""
    model_name: str = "microsoft/DialoGPT-medium"
    max_outer_iterations: int = 1000
    samples_per_iteration: int = 4
    reward_threshold: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 1e-4
    device: str = "auto"
    seed: int = 42


class SEALFramework:
    """
    Main SEAL Framework implementing the nested RL optimization.
    
    This class implements Algorithm 1 from the paper:
    1. Outer RL loop: optimizes self-edit generation 
    2. Inner update loop: applies self-edits via supervised fine-tuning
    """
    
    def __init__(
        self,
        config: SEALConfig,
        domain: Any,  # Domain-specific implementation
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.domain = domain
        self.logger = logger or logging.getLogger(__name__)
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
            
        # Set random seed
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        
        # Initialize model and tokenizer
        self._load_model()
        
        # Initialize components
        self.self_edit_generator = SelfEditGenerator(
            self.model, self.tokenizer, self.domain
        )
        self.lora_wrapper = LoRAWrapper(
            self.model, 
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )
        self.optimizer = ReSTEMOptimizer(
            self.model,
            self.tokenizer,
            learning_rate=config.learning_rate
        )
        
        # Training state
        self.iteration = 0
        self.training_history = []
        
    def _load_model(self):
        """Load the base language model and tokenizer"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        
        self.logger.info(f"Model loaded on device: {self.device}")
        
    def train(self, dataset: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """
        Main training loop implementing Algorithm 1
        
        Args:
            dataset: List of (context, task) pairs
            
        Returns:
            Training statistics and final model state
        """
        self.logger.info("Starting SEAL training...")
        self.logger.info(f"Dataset size: {len(dataset)}")
        self.logger.info(f"Max iterations: {self.config.max_outer_iterations}")
        
        progress_bar = tqdm(
            range(self.config.max_outer_iterations),
            desc="SEAL Training"
        )
        
        for t in progress_bar:
            self.iteration = t + 1
            
            # Sample (C, τ) from dataset - Line 3 in Algorithm 1
            context, task = random.choice(dataset)
            
            # Generate self-edits - Line 4 in Algorithm 1
            self_edits, edit_info = self._generate_self_edits(context)
            
            # Evaluate self-edits and compute rewards
            rewards, eval_results = self._evaluate_self_edits(
                self_edits, context, task
            )
            
            # Update model using ReSTEM - Line 8 in Algorithm 1
            update_info = self.optimizer.update(
                self_edits, rewards, context, self.config.reward_threshold
            )
            
            # Log iteration results
            iteration_stats = {
                "iteration": self.iteration,
                "num_self_edits": len(self_edits),
                "positive_rewards": sum(1 for r in rewards if r > self.config.reward_threshold),
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "max_reward": max(rewards) if rewards else 0.0,
                **edit_info,
                **eval_results,
                **update_info
            }
            
            self.training_history.append(iteration_stats)
            
            # Update progress bar
            progress_bar.set_postfix({
                "mean_reward": f"{iteration_stats['mean_reward']:.3f}",
                "pos_rewards": f"{iteration_stats['positive_rewards']}/{len(self_edits)}"
            })
            
            # Log detailed stats every 10 iterations
            if self.iteration % 10 == 0:
                self._log_training_stats(iteration_stats)
                
        self.logger.info("SEAL training completed!")
        
        return {
            "final_iteration": self.iteration,
            "training_history": self.training_history,
            "model_state": self.model.state_dict(),
        }
    
    def _generate_self_edits(self, context: Any) -> Tuple[List[str], Dict[str, Any]]:
        """Generate multiple self-edit candidates for given context"""
        self_edits = []
        generation_times = []
        
        for _ in range(self.config.samples_per_iteration):
            import time
            start_time = time.time()
            
            # Generate self-edit using current model policy
            self_edit = self.self_edit_generator.generate(context)
            self_edits.append(self_edit)
            
            generation_times.append(time.time() - start_time)
        
        edit_info = {
            "avg_generation_time": sum(generation_times) / len(generation_times),
            "self_edit_lengths": [len(edit.split()) for edit in self_edits]
        }
        
        return self_edits, edit_info
    
    def _evaluate_self_edits(
        self, 
        self_edits: List[str], 
        context: Any, 
        task: Any
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Evaluate self-edits by applying them and measuring performance
        
        This implements the inner loop (Lines 5-7 in Algorithm 1):
        1. Apply self-edit via SFT to get θ'_t
        2. Evaluate adapted model on task τ  
        3. Compute reward based on performance
        """
        rewards = []
        evaluation_times = []
        
        for self_edit in self_edits:
            import time
            start_time = time.time()
            
            # Inner Loop Update: θ'_t ← SFT(θ_t, SE) - Line 5
            adapted_model = self.lora_wrapper.apply_self_edit(self_edit, context)
            
            # Evaluate: Ans ∼ LM_θ'_t(·|τ) - Line 6  
            performance = self.domain.evaluate(adapted_model, self.tokenizer, task)
            
            # Compute reward: r ← r(Ans, τ) - Line 7
            reward = self.domain.compute_reward(performance, task)
            rewards.append(reward)
            
            evaluation_times.append(time.time() - start_time)
            
        eval_results = {
            "avg_evaluation_time": sum(evaluation_times) / len(evaluation_times),
            "reward_std": torch.tensor(rewards).std().item() if len(rewards) > 1 else 0.0
        }
        
        return rewards, eval_results
    
    def _log_training_stats(self, stats: Dict[str, Any]):
        """Log detailed training statistics"""
        self.logger.info(f"Iteration {stats['iteration']}:")
        self.logger.info(f"  Generated {stats['num_self_edits']} self-edits")
        self.logger.info(f"  Positive rewards: {stats['positive_rewards']}/{stats['num_self_edits']}")
        self.logger.info(f"  Mean reward: {stats['mean_reward']:.4f}")
        self.logger.info(f"  Max reward: {stats['max_reward']:.4f}")
        if 'reward_std' in stats:
            self.logger.info(f"  Reward std: {stats['reward_std']:.4f}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint and training state"""
        checkpoint = {
            "iteration": self.iteration,
            "model_state_dict": self.model.state_dict(),
            "training_history": self.training_history,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.iteration = checkpoint["iteration"]
        self.training_history = checkpoint["training_history"]
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def evaluate_model(self, test_dataset: List[Tuple[Any, Any]]) -> Dict[str, float]:
        """Evaluate the current model on a test dataset"""
        self.logger.info("Evaluating model...")
        
        total_performance = 0.0
        num_samples = len(test_dataset)
        
        for context, task in tqdm(test_dataset, desc="Evaluation"):
            performance = self.domain.evaluate(self.model, self.tokenizer, task)
            total_performance += self.domain.compute_reward(performance, task)
        
        avg_performance = total_performance / num_samples
        
        results = {
            "average_performance": avg_performance,
            "num_samples": num_samples
        }
        
        self.logger.info(f"Evaluation results: {results}")
        return results