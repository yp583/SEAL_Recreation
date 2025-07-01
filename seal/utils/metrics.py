"""
Metrics and Evaluation Utilities for SEAL

Implements metrics computation, tracking, and visualization for SEAL training.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from collections import defaultdict, deque
import time
import json
from pathlib import Path


class MetricsTracker:
    """
    Tracks and computes metrics during SEAL training.
    
    This class maintains running statistics and provides utilities
    for metric computation and analysis.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        save_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.window_size = window_size
        self.save_path = save_path
        self.logger = logger or logging.getLogger(__name__)
        
        # Metric storage
        self.metrics = defaultdict(list)
        self.recent_metrics = defaultdict(lambda: deque(maxlen=window_size))
        
        # Timing information
        self.start_time = time.time()
        self.iteration_times = deque(maxlen=window_size)
        
        # Best metrics tracking
        self.best_metrics = {}
        
    def add_metric(self, name: str, value: float, iteration: Optional[int] = None):
        """Add a metric value"""
        timestamp = time.time()
        
        metric_entry = {
            "value": value,
            "timestamp": timestamp,
            "iteration": iteration
        }
        
        self.metrics[name].append(metric_entry)
        self.recent_metrics[name].append(value)
        
        # Update best metrics
        if name not in self.best_metrics or value > self.best_metrics[name]["value"]:
            self.best_metrics[name] = metric_entry.copy()
            
    def add_metrics(self, metrics: Dict[str, float], iteration: Optional[int] = None):
        """Add multiple metrics at once"""
        for name, value in metrics.items():
            self.add_metric(name, value, iteration)
            
    def get_metric(self, name: str) -> List[Dict[str, Any]]:
        """Get all values for a metric"""
        return self.metrics[name]
    
    def get_recent_metric(self, name: str) -> List[float]:
        """Get recent values for a metric"""
        return list(self.recent_metrics[name])
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.recent_metrics or not self.recent_metrics[name]:
            return {}
            
        values = list(self.recent_metrics[name])
        values_tensor = torch.tensor(values)
        
        return {
            "mean": values_tensor.mean().item(),
            "std": values_tensor.std().item(),
            "min": values_tensor.min().item(),
            "max": values_tensor.max().item(),
            "latest": values[-1],
            "count": len(values)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        stats = {}
        for name in self.metrics:
            stats[name] = self.get_metric_stats(name)
        return stats
    
    def get_best_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get best values for all metrics"""
        return self.best_metrics.copy()
    
    def add_iteration_time(self, duration: float):
        """Add iteration timing information"""
        self.iteration_times.append(duration)
        self.add_metric("iteration_time", duration)
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics"""
        if not self.iteration_times:
            return {}
            
        times = list(self.iteration_times)
        times_tensor = torch.tensor(times)
        
        total_time = time.time() - self.start_time
        estimated_total_iterations = len(self.metrics.get("iteration_time", []))
        
        stats = {
            "mean_iteration_time": times_tensor.mean().item(),
            "std_iteration_time": times_tensor.std().item(),
            "total_time": total_time,
            "total_iterations": estimated_total_iterations
        }
        
        if estimated_total_iterations > 0:
            stats["iterations_per_second"] = estimated_total_iterations / total_time
            
        return stats
    
    def save_metrics(self, path: Optional[str] = None):
        """Save metrics to file"""
        save_path = path or self.save_path
        if not save_path:
            self.logger.warning("No save path specified for metrics")
            return
            
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            "metrics": dict(self.metrics),
            "best_metrics": self.best_metrics,
            "timing_stats": self.get_timing_stats(),
            "summary_stats": self.get_all_stats()
        }
        
        # Save as JSON
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
            
        self.logger.info(f"Metrics saved to {save_path}")
    
    def load_metrics(self, path: str):
        """Load metrics from file"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.metrics = defaultdict(list, data.get("metrics", {}))
        self.best_metrics = data.get("best_metrics", {})
        
        # Rebuild recent metrics
        for name, entries in self.metrics.items():
            recent_values = [entry["value"] for entry in entries[-self.window_size:]]
            self.recent_metrics[name] = deque(recent_values, maxlen=self.window_size)
            
        self.logger.info(f"Metrics loaded from {path}")
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.recent_metrics.clear()
        self.best_metrics.clear()
        self.iteration_times.clear()
        self.start_time = time.time()
        
    def summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics and statistics"""
        summary = {
            "metrics_summary": self.get_all_stats(),
            "best_metrics": self.get_best_metrics(),
            "timing_stats": self.get_timing_stats(),
            "total_metrics_tracked": len(self.metrics)
        }
        
        return summary


class SEALMetrics:
    """
    SEAL-specific metrics computation and tracking.
    
    This class computes domain-specific metrics and provides
    analysis tools for SEAL training dynamics.
    """
    
    def __init__(self, domain: str, logger: Optional[logging.Logger] = None):
        self.domain = domain
        self.logger = logger or logging.getLogger(__name__)
        self.tracker = MetricsTracker()
        
    def compute_training_metrics(
        self,
        rewards: List[float],
        performance_details: List[Dict[str, Any]],
        iteration: int
    ) -> Dict[str, float]:
        """
        Compute training metrics for a SEAL iteration.
        
        Args:
            rewards: List of rewards for self-edits
            performance_details: Detailed performance information
            iteration: Current iteration number
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        if rewards:
            rewards_tensor = torch.tensor(rewards)
            
            # Basic reward metrics
            metrics["mean_reward"] = rewards_tensor.mean().item()
            metrics["std_reward"] = rewards_tensor.std().item()
            metrics["max_reward"] = rewards_tensor.max().item()
            metrics["min_reward"] = rewards_tensor.min().item()
            
            # Reward distribution metrics
            metrics["positive_reward_rate"] = (rewards_tensor > 0).float().mean().item()
            metrics["zero_reward_rate"] = (rewards_tensor == 0).float().mean().item()
            metrics["negative_reward_rate"] = (rewards_tensor < 0).float().mean().item()
            
            # Success metrics
            metrics["num_positive_rewards"] = (rewards_tensor > 0).sum().item()
            metrics["num_samples"] = len(rewards)
            metrics["success_rate"] = metrics["num_positive_rewards"] / metrics["num_samples"]
            
        # Performance improvement metrics
        if performance_details:
            baseline_scores = []
            adapted_scores = []
            improvements = []
            
            for detail in performance_details:
                baseline_perf = detail["baseline_performance"]
                adapted_perf = detail["adapted_performance"]
                
                # Extract accuracy or main metric
                baseline_score = baseline_perf.get("accuracy", baseline_perf.get("average_score", 0.0))
                adapted_score = adapted_perf.get("accuracy", adapted_perf.get("average_score", 0.0))
                
                baseline_scores.append(baseline_score)
                adapted_scores.append(adapted_score)
                improvements.append(adapted_score - baseline_score)
            
            if baseline_scores:
                metrics["mean_baseline_performance"] = torch.tensor(baseline_scores).mean().item()
                metrics["mean_adapted_performance"] = torch.tensor(adapted_scores).mean().item()
                metrics["mean_improvement"] = torch.tensor(improvements).mean().item()
                metrics["std_improvement"] = torch.tensor(improvements).std().item()
                metrics["positive_improvement_rate"] = (torch.tensor(improvements) > 0).float().mean().item()
                
        # Add to tracker
        self.tracker.add_metrics(metrics, iteration)
        
        return metrics
    
    def compute_domain_specific_metrics(
        self,
        performance_details: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute domain-specific metrics"""
        if self.domain == "knowledge_incorporation":
            return self._compute_knowledge_incorporation_metrics(performance_details)
        elif self.domain == "few_shot_learning":
            return self._compute_few_shot_learning_metrics(performance_details)
        else:
            return {}
    
    def _compute_knowledge_incorporation_metrics(
        self,
        performance_details: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute knowledge incorporation specific metrics"""
        metrics = {}
        
        all_baseline_accuracies = []
        all_adapted_accuracies = []
        question_counts = []
        
        for detail in performance_details:
            baseline_perf = detail["baseline_performance"]
            adapted_perf = detail["adapted_performance"]
            
            all_baseline_accuracies.append(baseline_perf.get("accuracy", 0.0))
            all_adapted_accuracies.append(adapted_perf.get("accuracy", 0.0))
            question_counts.append(baseline_perf.get("total_questions", 0))
        
        if all_baseline_accuracies:
            metrics["knowledge_baseline_accuracy"] = torch.tensor(all_baseline_accuracies).mean().item()
            metrics["knowledge_adapted_accuracy"] = torch.tensor(all_adapted_accuracies).mean().item()
            metrics["knowledge_accuracy_improvement"] = (
                metrics["knowledge_adapted_accuracy"] - metrics["knowledge_baseline_accuracy"]
            )
            
        if question_counts:
            metrics["avg_questions_per_task"] = torch.tensor(question_counts, dtype=torch.float32).mean().item()
            
        return metrics
    
    def _compute_few_shot_learning_metrics(
        self,
        performance_details: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute few-shot learning specific metrics"""
        metrics = {}
        
        all_baseline_accuracies = []
        all_adapted_accuracies = []
        
        for detail in performance_details:
            baseline_perf = detail["baseline_performance"]
            adapted_perf = detail["adapted_performance"]
            
            all_baseline_accuracies.append(baseline_perf.get("accuracy", 0.0))
            all_adapted_accuracies.append(adapted_perf.get("accuracy", 0.0))
        
        if all_baseline_accuracies:
            metrics["arc_baseline_accuracy"] = torch.tensor(all_baseline_accuracies).mean().item()
            metrics["arc_adapted_accuracy"] = torch.tensor(all_adapted_accuracies).mean().item()
            metrics["arc_accuracy_improvement"] = (
                metrics["arc_adapted_accuracy"] - metrics["arc_baseline_accuracy"]
            )
            
        return metrics
    
    def analyze_training_dynamics(
        self,
        window_size: int = 50
    ) -> Dict[str, Any]:
        """Analyze training dynamics and trends"""
        analysis = {}
        
        # Get recent reward trends
        recent_rewards = self.tracker.get_recent_metric("mean_reward")
        if len(recent_rewards) >= window_size:
            early_rewards = recent_rewards[:window_size//2]
            late_rewards = recent_rewards[-window_size//2:]
            
            analysis["reward_trend"] = {
                "early_mean": torch.tensor(early_rewards).mean().item(),
                "late_mean": torch.tensor(late_rewards).mean().item(),
                "improvement": torch.tensor(late_rewards).mean().item() - torch.tensor(early_rewards).mean().item()
            }
        
        # Get recent success rate trends
        recent_success_rates = self.tracker.get_recent_metric("success_rate")
        if len(recent_success_rates) >= window_size:
            early_success = recent_success_rates[:window_size//2]
            late_success = recent_success_rates[-window_size//2:]
            
            analysis["success_rate_trend"] = {
                "early_mean": torch.tensor(early_success).mean().item(),
                "late_mean": torch.tensor(late_success).mean().item(),
                "improvement": torch.tensor(late_success).mean().item() - torch.tensor(early_success).mean().item()
            }
        
        # Analyze variance/stability
        if recent_rewards:
            analysis["training_stability"] = {
                "reward_variance": torch.tensor(recent_rewards).var().item(),
                "reward_cv": torch.tensor(recent_rewards).std().item() / max(torch.tensor(recent_rewards).mean().item(), 1e-8)
            }
        
        # Detect potential issues
        analysis["potential_issues"] = self._detect_training_issues()
        
        return analysis
    
    def _detect_training_issues(self) -> List[str]:
        """Detect potential training issues"""
        issues = []
        
        # Check for stagnant rewards
        recent_rewards = self.tracker.get_recent_metric("mean_reward")
        if len(recent_rewards) >= 20:
            recent_var = torch.tensor(recent_rewards[-20:]).var().item()
            if recent_var < 0.001:
                issues.append("Rewards appear stagnant (low variance)")
        
        # Check for very low success rates
        recent_success_rates = self.tracker.get_recent_metric("success_rate")
        if recent_success_rates:
            latest_success_rate = recent_success_rates[-1]
            if latest_success_rate < 0.1:
                issues.append("Very low success rate (<10%)")
                
        # Check for declining performance
        if len(recent_rewards) >= 20:
            early_rewards = recent_rewards[:10]
            late_rewards = recent_rewards[-10:]
            if torch.tensor(late_rewards).mean() < torch.tensor(early_rewards).mean() - 0.1:
                issues.append("Performance appears to be declining")
        
        return issues
    
    def get_summary_report(self) -> str:
        """Generate a summary report of training progress"""
        summary = self.tracker.summary()
        analysis = self.analyze_training_dynamics()
        
        report = []
        report.append("=== SEAL Training Summary ===")
        report.append("")
        
        # Basic stats
        if "mean_reward" in summary["metrics_summary"]:
            reward_stats = summary["metrics_summary"]["mean_reward"]
            report.append(f"Mean Reward: {reward_stats['latest']:.4f} (±{reward_stats['std']:.4f})")
            
        if "success_rate" in summary["metrics_summary"]:
            success_stats = summary["metrics_summary"]["success_rate"]
            report.append(f"Success Rate: {success_stats['latest']:.2%}")
            
        # Best metrics
        if summary["best_metrics"]:
            report.append("")
            report.append("Best Metrics:")
            for name, best in summary["best_metrics"].items():
                report.append(f"  {name}: {best['value']:.4f} (iteration {best.get('iteration', 'N/A')})")
        
        # Training dynamics
        if "reward_trend" in analysis:
            trend = analysis["reward_trend"]
            report.append("")
            report.append(f"Reward Trend: {trend['improvement']:+.4f}")
            
        # Issues
        if analysis.get("potential_issues"):
            report.append("")
            report.append("Potential Issues:")
            for issue in analysis["potential_issues"]:
                report.append(f"  - {issue}")
        
        # Timing
        timing = summary["timing_stats"]
        if timing:
            report.append("")
            report.append(f"Training Time: {timing.get('total_time', 0):.1f}s")
            report.append(f"Iterations/sec: {timing.get('iterations_per_second', 0):.2f}")
            
        return "\n".join(report)


class Metrics:
    """Main metrics interface for SEAL"""
    
    def __init__(self, domain: str, save_path: Optional[str] = None):
        self.domain = domain
        self.seal_metrics = SEALMetrics(domain)
        self.save_path = save_path
        
    def compute_iteration_metrics(
        self,
        rewards: List[float],
        performance_details: List[Dict[str, Any]],
        iteration: int,
        iteration_time: float
    ) -> Dict[str, float]:
        """Compute metrics for a single iteration"""
        # Training metrics
        training_metrics = self.seal_metrics.compute_training_metrics(
            rewards, performance_details, iteration
        )
        
        # Domain-specific metrics
        domain_metrics = self.seal_metrics.compute_domain_specific_metrics(
            performance_details
        )
        
        # Combine metrics
        all_metrics = {**training_metrics, **domain_metrics}
        
        # Add timing
        self.seal_metrics.tracker.add_iteration_time(iteration_time)
        
        return all_metrics
    
    def get_tracker(self) -> MetricsTracker:
        """Get the underlying metrics tracker"""
        return self.seal_metrics.tracker
    
    def analyze_training(self) -> Dict[str, Any]:
        """Analyze training dynamics"""
        return self.seal_metrics.analyze_training_dynamics()
    
    def generate_report(self) -> str:
        """Generate summary report"""
        return self.seal_metrics.get_summary_report()
    
    def save(self, path: Optional[str] = None):
        """Save metrics"""
        self.seal_metrics.tracker.save_metrics(path or self.save_path)
    
    def load(self, path: str):
        """Load metrics"""
        self.seal_metrics.tracker.load_metrics(path)