"""
Reward Model Implementation

Implements reward computation for SEAL framework across different domains.
The reward model evaluates the effectiveness of self-edits and provides
the reward signal for the ReSTEM optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod


class BaseRewardModel(ABC):
    """
    Abstract base class for reward models.
    
    Different domains may require different reward computation strategies.
    """
    
    @abstractmethod
    def compute_reward(
        self,
        performance_before: Dict[str, float],
        performance_after: Dict[str, float],
        task: Any
    ) -> float:
        """
        Compute reward based on performance improvement.
        
        Args:
            performance_before: Performance metrics before applying self-edit
            performance_after: Performance metrics after applying self-edit
            task: Domain-specific task
            
        Returns:
            Reward value (higher is better)
        """
        pass


class BinaryRewardModel(BaseRewardModel):
    """
    Binary reward model that gives 1.0 for improvement, 0.0 otherwise.
    
    This is the reward model used in the SEAL paper:
    r(SE, τ, θ_t) = 1 if adaptation using SE improves LM_θ_t's performance
                     0 otherwise
    """
    
    def __init__(
        self,
        threshold: float = 0.0,
        metric_key: str = "accuracy",
        logger: Optional[logging.Logger] = None
    ):
        self.threshold = threshold
        self.metric_key = metric_key
        self.logger = logger or logging.getLogger(__name__)
        
    def compute_reward(
        self,
        performance_before: Dict[str, float],
        performance_after: Dict[str, float],
        task: Any
    ) -> float:
        """
        Compute binary reward based on performance improvement.
        
        Args:
            performance_before: Performance before self-edit
            performance_after: Performance after self-edit
            task: Task instance
            
        Returns:
            1.0 if improvement, 0.0 otherwise
        """
        before_score = performance_before.get(self.metric_key, 0.0)  
        after_score = performance_after.get(self.metric_key, 0.0)
        
        improvement = after_score - before_score
        reward = 1.0 if improvement > self.threshold else 0.0
        
        self.logger.debug(f"Reward computation: {before_score:.3f} -> {after_score:.3f}, "
                         f"improvement: {improvement:.3f}, reward: {reward}")
        
        return reward


class ContinuousRewardModel(BaseRewardModel):
    """
    Continuous reward model that returns the actual improvement value.
    
    This provides more nuanced reward signals but may be less stable
    for training than binary rewards.
    """
    
    def __init__(
        self,
        metric_key: str = "accuracy",
        scaling_factor: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        self.metric_key = metric_key
        self.scaling_factor = scaling_factor
        self.logger = logger or logging.getLogger(__name__)
        
    def compute_reward(
        self,
        performance_before: Dict[str, float],
        performance_after: Dict[str, float],
        task: Any
    ) -> float:
        """
        Compute continuous reward based on performance improvement.
        
        Args:
            performance_before: Performance before self-edit
            performance_after: Performance after self-edit  
            task: Task instance
            
        Returns:
            Improvement value scaled by scaling_factor
        """
        before_score = performance_before.get(self.metric_key, 0.0)
        after_score = performance_after.get(self.metric_key, 0.0)
        
        improvement = after_score - before_score
        reward = improvement * self.scaling_factor
        
        self.logger.debug(f"Reward computation: {before_score:.3f} -> {after_score:.3f}, "
                         f"improvement: {improvement:.3f}, reward: {reward:.3f}")
        
        return reward


class ThresholdedRewardModel(BaseRewardModel):
    """
    Reward model that gives different rewards based on improvement thresholds.
    
    Provides more granular reward signals than binary while maintaining stability.
    """
    
    def __init__(
        self,
        thresholds: List[float] = [0.0, 0.1, 0.2, 0.3],
        rewards: List[float] = [0.0, 0.25, 0.5, 1.0],
        metric_key: str = "accuracy",
        logger: Optional[logging.Logger] = None
    ):
        assert len(thresholds) == len(rewards), "Thresholds and rewards must have same length"
        
        self.thresholds = sorted(thresholds)
        self.rewards = rewards
        self.metric_key = metric_key
        self.logger = logger or logging.getLogger(__name__)
        
    def compute_reward(
        self,
        performance_before: Dict[str, float],
        performance_after: Dict[str, float],
        task: Any
    ) -> float:
        """
        Compute thresholded reward based on improvement level.
        
        Args:
            performance_before: Performance before self-edit
            performance_after: Performance after self-edit
            task: Task instance
            
        Returns:
            Reward based on improvement threshold
        """
        before_score = performance_before.get(self.metric_key, 0.0)
        after_score = performance_after.get(self.metric_key, 0.0)
        
        improvement = after_score - before_score
        
        # Find appropriate reward based on thresholds
        reward = self.rewards[0]  # Default to lowest reward
        for i, threshold in enumerate(self.thresholds):
            if improvement >= threshold:
                reward = self.rewards[i]
                
        self.logger.debug(f"Reward computation: {before_score:.3f} -> {after_score:.3f}, "
                         f"improvement: {improvement:.3f}, reward: {reward}")
        
        return reward


class MultiMetricRewardModel(BaseRewardModel):
    """
    Reward model that considers multiple performance metrics.
    
    Combines multiple metrics with weights to compute final reward.
    """
    
    def __init__(
        self,
        metric_weights: Dict[str, float] = {"accuracy": 1.0},
        reward_type: str = "binary",  # "binary", "continuous", "thresholded"
        threshold: float = 0.0,
        logger: Optional[logging.Logger] = None
    ):
        self.metric_weights = metric_weights
        self.reward_type = reward_type
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Normalize weights
        total_weight = sum(self.metric_weights.values())
        if total_weight > 0:
            self.metric_weights = {k: v / total_weight for k, v in self.metric_weights.items()}
    
    def compute_reward(
        self,
        performance_before: Dict[str, float],
        performance_after: Dict[str, float],
        task: Any
    ) -> float:
        """
        Compute reward based on weighted combination of multiple metrics.
        
        Args:
            performance_before: Performance before self-edit
            performance_after: Performance after self-edit
            task: Task instance
            
        Returns:
            Combined reward value
        """
        total_improvement = 0.0
        
        for metric, weight in self.metric_weights.items():
            before_score = performance_before.get(metric, 0.0)
            after_score = performance_after.get(metric, 0.0)
            improvement = after_score - before_score
            total_improvement += improvement * weight
            
        # Apply reward type
        if self.reward_type == "binary":
            reward = 1.0 if total_improvement > self.threshold else 0.0
        elif self.reward_type == "continuous":
            reward = total_improvement
        else:  # thresholded
            reward = max(0.0, total_improvement)
            
        self.logger.debug(f"Multi-metric reward: improvement={total_improvement:.3f}, "
                         f"reward={reward:.3f}")
        
        return reward


class RewardModel:
    """
    Main reward model class that orchestrates reward computation for SEAL.
    
    This class handles:
    - Baseline performance computation
    - Self-edit application and evaluation  
    - Reward calculation using specified reward model
    """
    
    def __init__(
        self,
        reward_model: BaseRewardModel,
        domain: Any,
        logger: Optional[logging.Logger] = None
    ):
        self.reward_model = reward_model
        self.domain = domain
        self.logger = logger or logging.getLogger(__name__)
        
        # Statistics tracking
        self.reward_history = []
        self.performance_history = []
        
    def compute_rewards(
        self,
        base_model: torch.nn.Module,
        tokenizer: Any,
        self_edits: List[str],
        contexts: List[Any],
        tasks: List[Any],
        lora_wrapper: Any
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Compute rewards for a batch of self-edits.
        
        Args:
            base_model: Base language model
            tokenizer: Model tokenizer
            self_edits: List of generated self-edits
            contexts: List of contexts used to generate self-edits
            tasks: List of evaluation tasks
            lora_wrapper: LoRA wrapper for model adaptation
            
        Returns:
            Tuple of (rewards, performance_details)
        """
        rewards = []
        performance_details = []
        
        for self_edit, context, task in zip(self_edits, contexts, tasks):
            # Compute baseline performance (without self-edit)
            baseline_performance = self.domain.evaluate(base_model, tokenizer, task)
            
            # Apply self-edit to create adapted model
            adapted_model = lora_wrapper.apply_self_edit(self_edit, context, tokenizer)
            
            # Compute performance after adaptation
            adapted_performance = self.domain.evaluate(adapted_model, tokenizer, task)
            
            # Compute reward
            reward = self.reward_model.compute_reward(
                baseline_performance, adapted_performance, task
            )
            
            rewards.append(reward)
            
            # Store performance details
            performance_detail = {
                "baseline_performance": baseline_performance,
                "adapted_performance": adapted_performance,
                "reward": reward,
                "self_edit": self_edit[:100] + "..." if len(self_edit) > 100 else self_edit
            }
            performance_details.append(performance_detail)
            
        # Update statistics
        self.reward_history.extend(rewards)
        self.performance_history.extend(performance_details)
        
        self.logger.info(f"Computed rewards for {len(self_edits)} self-edits: "
                        f"mean={torch.tensor(rewards).mean():.3f}, "
                        f"positive={sum(1 for r in rewards if r > 0)}/{len(rewards)}")
        
        return rewards, performance_details
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about reward distribution"""
        if not self.reward_history:
            return {}
            
        rewards = torch.tensor(self.reward_history)
        
        return {
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "min_reward": rewards.min().item(),
            "max_reward": rewards.max().item(),
            "positive_reward_rate": (rewards > 0).float().mean().item(),
            "total_evaluations": len(self.reward_history)
        }
    
    def get_performance_trends(self, window_size: int = 100) -> Dict[str, List[float]]:
        """Get performance trends over recent evaluations"""
        if len(self.performance_history) < window_size:
            recent_history = self.performance_history
        else:
            recent_history = self.performance_history[-window_size:]
            
        trends = {
            "baseline_accuracy": [],
            "adapted_accuracy": [],
            "improvement": []
        }
        
        for perf in recent_history:
            baseline_acc = perf["baseline_performance"].get("accuracy", 0.0)
            adapted_acc = perf["adapted_performance"].get("accuracy", 0.0)
            improvement = adapted_acc - baseline_acc
            
            trends["baseline_accuracy"].append(baseline_acc)
            trends["adapted_accuracy"].append(adapted_acc)
            trends["improvement"].append(improvement)
            
        return trends
    
    def reset_statistics(self):
        """Reset reward and performance statistics"""
        self.reward_history = []
        self.performance_history = []
        self.logger.info("Reward statistics reset")


class RewardAnalyzer:
    """Utility class for analyzing reward patterns and trends"""
    
    @staticmethod
    def analyze_reward_distribution(rewards: List[float]) -> Dict[str, Any]:
        """Analyze the distribution of rewards"""
        if not rewards:
            return {}
            
        rewards_tensor = torch.tensor(rewards)
        
        analysis = {
            "count": len(rewards),
            "mean": rewards_tensor.mean().item(),
            "std": rewards_tensor.std().item(),
            "min": rewards_tensor.min().item(),
            "max": rewards_tensor.max().item(),
            "median": rewards_tensor.median().item(),
            "positive_count": (rewards_tensor > 0).sum().item(),
            "zero_count": (rewards_tensor == 0).sum().item(),
            "negative_count": (rewards_tensor < 0).sum().item(),
            "positive_rate": (rewards_tensor > 0).float().mean().item()
        }
        
        return analysis
    
    @staticmethod
    def detect_reward_stagnation(
        rewards: List[float], 
        window_size: int = 50,
        stagnation_threshold: float = 0.01
    ) -> bool:
        """Detect if rewards have stagnated (not improving)"""
        if len(rewards) < window_size * 2:
            return False
            
        # Compare recent window to previous window
        recent_rewards = rewards[-window_size:]
        previous_rewards = rewards[-window_size*2:-window_size]
        
        recent_mean = torch.tensor(recent_rewards).mean().item()
        previous_mean = torch.tensor(previous_rewards).mean().item()
        
        improvement = recent_mean - previous_mean
        
        return improvement < stagnation_threshold
    
    @staticmethod
    def suggest_reward_model_adjustments(
        rewards: List[float],
        performance_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Suggest adjustments to reward model based on training dynamics"""
        analysis = RewardAnalyzer.analyze_reward_distribution(rewards)
        
        suggestions = {
            "analysis": analysis,
            "recommendations": []
        }
        
        # If too many zero rewards, suggest lowering threshold
        if analysis.get("positive_rate", 0) < 0.1:
            suggestions["recommendations"].append(
                "Consider lowering reward threshold - too few positive rewards"
            )
            
        # If all rewards are positive, suggest raising threshold
        elif analysis.get("positive_rate", 0) > 0.9:
            suggestions["recommendations"].append(
                "Consider raising reward threshold - too many positive rewards"
            )
            
        # If rewards have low variance, suggest different reward model
        if analysis.get("std", 0) < 0.1:
            suggestions["recommendations"].append(
                "Consider using continuous or thresholded reward model for more variance"
            )
            
        return suggestions