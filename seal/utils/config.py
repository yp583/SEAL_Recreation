"""
Configuration Management for SEAL

Provides configuration classes and utilities for managing hyperparameters
and settings across different components of the SEAL framework.
"""

import yaml
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging


@dataclass
class SEALConfig:
    """Main configuration for SEAL framework"""
    
    # Model configuration
    model_name: str = "microsoft/DialoGPT-medium"
    model_cache_dir: Optional[str] = None
    device: str = "auto"
    torch_dtype: str = "auto"  # "float16", "float32", "auto"
    
    # Training configuration
    max_outer_iterations: int = 1000
    samples_per_iteration: int = 4
    reward_threshold: float = 0.0
    learning_rate: float = 1e-4
    seed: int = 42
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # ReSTEM configuration
    restem_batch_size: int = 4
    restem_num_epochs: int = 1
    restem_warmup_steps: int = 100
    restem_max_grad_norm: float = 1.0
    
    # Self-edit generation configuration
    generation_temperature: float = 1.0
    generation_top_k: int = 50
    generation_top_p: float = 0.95
    generation_max_length: int = 512
    
    # Domain configuration
    domain: str = "knowledge_incorporation"  # or "few_shot_learning"
    
    # Data configuration
    data_config: Dict[str, Any] = field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    eval_frequency: int = 10
    eval_samples: int = 100
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 100
    max_checkpoints: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SEALConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to file"""
        path = Path(path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, path: str) -> 'SEALConfig':
        """Load configuration from file"""
        path = Path(path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        return cls.from_dict(config_dict)


@dataclass 
class KnowledgeIncorporationConfig:
    """Configuration specific to Knowledge Incorporation domain"""
    
    # Data configuration
    dataset_name: str = "squad"
    split: str = "train"
    max_samples: Optional[int] = None
    min_questions_per_context: int = 1
    
    # Preprocessing configuration
    max_passage_length: int = 1000
    max_question_length: int = 200
    max_answer_length: int = 100
    min_passage_length: int = 50
    filter_duplicates: bool = True
    
    # Domain-specific configuration
    max_implications: int = 10
    max_implication_length: int = 200
    prompt_type: str = "implications"  # "implications", "questions_answers", "rewrite", "facts"
    
    # Evaluation configuration
    evaluation_temperature: float = 0.8


@dataclass
class FewShotLearningConfig:
    """Configuration specific to Few-Shot Learning domain"""
    
    # Data configuration
    dataset_path: Optional[str] = None
    split: str = "train"
    max_samples: Optional[int] = None
    min_examples_per_task: int = 1
    
    # Preprocessing configuration
    max_grid_size: int = 30
    min_grid_size: int = 1
    max_examples_per_task: int = 10
    normalize_colors: bool = True
    
    # Domain-specific configuration
    max_augmentations: int = 10
    available_augmentations: List[str] = field(default_factory=lambda: [
        "rotation_90", "rotation_180", "rotation_270",
        "flip_horizontal", "flip_vertical", 
        "transpose", "reflection_diagonal",
        "size_augmentation", "color_permutation",
        "chain_augmentations", "repeat_augmentations"
    ])
    
    # Hyperparameter options
    available_learning_rates: List[float] = field(default_factory=lambda: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4])
    available_epochs: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    available_loss_strategies: List[str] = field(default_factory=lambda: ["all_tokens", "output_only"])
    available_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])


class Config:
    """Main configuration manager for SEAL"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        if config_path:
            self.seal_config = SEALConfig.load(config_path)
        else:
            self.seal_config = SEALConfig()
            
        # Initialize domain-specific config
        self._init_domain_config()
        
    def _init_domain_config(self):
        """Initialize domain-specific configuration"""
        if self.seal_config.domain == "knowledge_incorporation":
            self.domain_config = KnowledgeIncorporationConfig(**self.seal_config.data_config)
        elif self.seal_config.domain == "few_shot_learning":
            self.domain_config = FewShotLearningConfig(**self.seal_config.data_config)
        else:
            raise ValueError(f"Unknown domain: {self.seal_config.domain}")
    
    def get_seal_config(self) -> SEALConfig:
        """Get main SEAL configuration"""
        return self.seal_config
    
    def get_domain_config(self) -> Union[KnowledgeIncorporationConfig, FewShotLearningConfig]:
        """Get domain-specific configuration"""
        return self.domain_config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self.seal_config, key):
                setattr(self.seal_config, key, value)
                self.logger.info(f"Updated {key} = {value}")
            else:
                self.logger.warning(f"Unknown config key: {key}")
                
        # Reinitialize domain config if needed
        if "domain" in updates or "data_config" in updates:
            self._init_domain_config()
    
    def save_config(self, path: str):
        """Save current configuration"""
        self.seal_config.save(path)
        self.logger.info(f"Configuration saved to {path}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for model initialization"""
        return {
            "model_name": self.seal_config.model_name,
            "cache_dir": self.seal_config.model_cache_dir,
            "device": self.seal_config.device,
            "torch_dtype": self.seal_config.torch_dtype,
        }
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration"""
        return {
            "r": self.seal_config.lora_r,
            "alpha": self.seal_config.lora_alpha,
            "dropout": self.seal_config.lora_dropout,
            "target_modules": self.seal_config.lora_target_modules,
        }
    
    def get_restem_config(self) -> Dict[str, Any]:
        """Get ReSTEM optimizer configuration"""
        return {
            "learning_rate": self.seal_config.learning_rate,
            "batch_size": self.seal_config.restem_batch_size,
            "num_epochs": self.seal_config.restem_num_epochs,
            "warmup_steps": self.seal_config.restem_warmup_steps,
            "max_grad_norm": self.seal_config.restem_max_grad_norm,
        }
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get self-edit generation configuration"""
        return {
            "temperature": self.seal_config.generation_temperature,
            "top_k": self.seal_config.generation_top_k,
            "top_p": self.seal_config.generation_top_p,
            "max_length": self.seal_config.generation_max_length,
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data loading configuration"""
        config = asdict(self.domain_config)
        config.update(self.seal_config.preprocessing_config)
        return config
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate basic parameters
        if self.seal_config.max_outer_iterations <= 0:
            issues.append("max_outer_iterations must be positive")
            
        if self.seal_config.samples_per_iteration <= 0:
            issues.append("samples_per_iteration must be positive")
            
        if self.seal_config.learning_rate <= 0:
            issues.append("learning_rate must be positive")
            
        # Validate LoRA parameters
        if self.seal_config.lora_r <= 0:
            issues.append("lora_r must be positive")
            
        if self.seal_config.lora_alpha <= 0:
            issues.append("lora_alpha must be positive")
            
        if not 0 <= self.seal_config.lora_dropout <= 1:
            issues.append("lora_dropout must be between 0 and 1")
            
        # Validate generation parameters
        if not 0 < self.seal_config.generation_temperature <= 2:
            issues.append("generation_temperature must be between 0 and 2")
            
        if self.seal_config.generation_top_k <= 0:
            issues.append("generation_top_k must be positive")
            
        if not 0 < self.seal_config.generation_top_p <= 1:
            issues.append("generation_top_p must be between 0 and 1")
            
        # Validate domain
        if self.seal_config.domain not in ["knowledge_incorporation", "few_shot_learning"]:
            issues.append("domain must be 'knowledge_incorporation' or 'few_shot_learning'")
            
        return issues


class ConfigBuilder:
    """Builder class for creating configurations"""
    
    def __init__(self):
        self.config_dict = {}
        
    def set_model(self, model_name: str, device: str = "auto") -> 'ConfigBuilder':
        """Set model configuration"""
        self.config_dict.update({
            "model_name": model_name,
            "device": device
        })
        return self
    
    def set_training(
        self, 
        max_iterations: int = 1000,
        samples_per_iteration: int = 4,
        learning_rate: float = 1e-4
    ) -> 'ConfigBuilder':
        """Set training configuration"""
        self.config_dict.update({
            "max_outer_iterations": max_iterations,
            "samples_per_iteration": samples_per_iteration,
            "learning_rate": learning_rate
        })
        return self
    
    def set_lora(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1
    ) -> 'ConfigBuilder':
        """Set LoRA configuration"""
        self.config_dict.update({
            "lora_r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout
        })
        return self
    
    def set_domain(self, domain: str, data_config: Dict[str, Any] = None) -> 'ConfigBuilder':
        """Set domain configuration"""
        self.config_dict["domain"] = domain
        if data_config:
            self.config_dict["data_config"] = data_config
        return self
    
    def set_logging(
        self,
        log_level: str = "INFO",
        wandb_project: Optional[str] = None
    ) -> 'ConfigBuilder':
        """Set logging configuration"""
        self.config_dict.update({
            "log_level": log_level,
            "wandb_project": wandb_project
        })
        return self
    
    def build(self) -> SEALConfig:
        """Build the configuration"""
        return SEALConfig.from_dict(self.config_dict)


def create_default_configs() -> Dict[str, SEALConfig]:
    """Create default configurations for different use cases"""
    
    configs = {}
    
    # Knowledge Incorporation with SQuAD
    configs["knowledge_incorporation_squad"] = (
        ConfigBuilder()
        .set_model("microsoft/DialoGPT-medium")
        .set_training(max_iterations=500, samples_per_iteration=4)
        .set_lora(r=16, alpha=32)
        .set_domain("knowledge_incorporation", {
            "dataset_name": "squad",
            "max_samples": 1000,
            "prompt_type": "implications"
        })
        .set_logging("INFO", "seal-knowledge-incorporation")
        .build()
    )
    
    # Few-Shot Learning with ARC
    configs["few_shot_learning_arc"] = (
        ConfigBuilder()
        .set_model("microsoft/DialoGPT-medium")
        .set_training(max_iterations=1000, samples_per_iteration=2)
        .set_lora(r=8, alpha=16)
        .set_domain("few_shot_learning", {
            "max_samples": 500,
            "max_examples_per_task": 5
        })
        .set_logging("INFO", "seal-few-shot-learning")
        .build()
    )
    
    # Quick test configuration
    configs["test"] = (
        ConfigBuilder()
        .set_model("microsoft/DialoGPT-small")
        .set_training(max_iterations=10, samples_per_iteration=2)
        .set_lora(r=4, alpha=8)
        .set_domain("knowledge_incorporation", {
            "max_samples": 10
        })
        .set_logging("DEBUG")
        .build()
    )
    
    return configs


def load_config_from_env() -> SEALConfig:
    """Load configuration from environment variables"""
    import os
    
    config = SEALConfig()
    
    # Override with environment variables if present
    if "SEAL_MODEL_NAME" in os.environ:
        config.model_name = os.environ["SEAL_MODEL_NAME"]
    if "SEAL_DEVICE" in os.environ:
        config.device = os.environ["SEAL_DEVICE"]
    if "SEAL_MAX_ITERATIONS" in os.environ:
        config.max_outer_iterations = int(os.environ["SEAL_MAX_ITERATIONS"])
    if "SEAL_LEARNING_RATE" in os.environ:
        config.learning_rate = float(os.environ["SEAL_LEARNING_RATE"])
    if "SEAL_DOMAIN" in os.environ:
        config.domain = os.environ["SEAL_DOMAIN"]
    if "WANDB_PROJECT" in os.environ:
        config.wandb_project = os.environ["WANDB_PROJECT"]
        
    return config