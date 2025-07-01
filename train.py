#!/usr/bin/env python3
"""
Main training script for SEAL (Self-Adapting LLMs)

This script provides a command-line interface for training SEAL models
on different domains with various configurations.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch

# Add SEAL to path
sys.path.append(str(Path(__file__).parent))

from seal import SEALFramework
from seal.utils.config import SEALConfig
from seal.domains.knowledge_incorporation import KnowledgeIncorporationDomain
from seal.domains.few_shot_learning import FewShotLearningDomain
from seal.models.lora_wrapper import LoRAWrapper
from seal.models.reward_model import BinaryRewardModel, RewardModel
from seal.data.loaders import DataLoader
from seal.data.preprocessors import DataPreprocessor
from seal.utils.config import Config, create_default_configs
from seal.utils.logging import create_logger
from seal.utils.metrics import Metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train SEAL model")
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["knowledge_incorporation_squad", "few_shot_learning_arc", "test"],
        help="Use a preset configuration"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    # Training arguments
    parser.add_argument(
        "--domain",
        type=str,
        choices=["knowledge_incorporation", "few_shot_learning"],
        default="knowledge_incorporation",
        help="Training domain"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of training iterations"
    )
    parser.add_argument(
        "--samples-per-iteration",
        type=int,
        default=4,
        help="Number of self-edit samples per iteration"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="squad",
        help="Dataset name (for knowledge incorporation)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to use"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project name"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=50,
        help="Save checkpoint every N iterations"
    )
    
    return parser.parse_args()


def load_config(args) -> Config:
    """Load configuration from arguments"""
    if args.config:
        # Load from file
        config = Config(args.config)
    elif args.preset:
        # Load preset configuration
        presets = create_default_configs()
        seal_config = presets[args.preset]
        config = Config()
        config.seal_config = seal_config
        config._init_domain_config()
    else:
        # Create default configuration
        config = Config()
    
    # Override with command line arguments
    overrides = {}
    if args.model_name:
        overrides["model_name"] = args.model_name
    if args.device != "auto":
        overrides["device"] = args.device
    if args.domain:
        overrides["domain"] = args.domain
    if args.max_iterations:
        overrides["max_outer_iterations"] = args.max_iterations
    if args.samples_per_iteration:
        overrides["samples_per_iteration"] = args.samples_per_iteration
    if args.learning_rate:
        overrides["learning_rate"] = args.learning_rate
    if args.log_level:
        overrides["log_level"] = args.log_level
    if args.log_file:
        overrides["log_file"] = args.log_file
    if args.wandb_project:
        overrides["wandb_project"] = args.wandb_project
    if args.output_dir:
        overrides["checkpoint_dir"] = args.output_dir
    if args.save_frequency:
        overrides["save_frequency"] = args.save_frequency
    
    # Data config overrides
    data_overrides = {}
    if args.dataset_name:
        data_overrides["dataset_name"] = args.dataset_name
    if args.max_samples:
        data_overrides["max_samples"] = args.max_samples
    
    if data_overrides:
        current_data_config = config.seal_config.data_config.copy()
        current_data_config.update(data_overrides)
        overrides["data_config"] = current_data_config
    
    if overrides:
        config.update_config(overrides)
    
    return config


def setup_domain(config: Config, logger):
    """Setup domain-specific implementation"""
    if config.seal_config.domain == "knowledge_incorporation":
        domain_config = config.get_domain_config()
        domain = KnowledgeIncorporationDomain(
            max_implications=domain_config.max_implications,
            max_implication_length=domain_config.max_implication_length,
            temperature=domain_config.evaluation_temperature,
            logger=logger.logger
        )
    elif config.seal_config.domain == "few_shot_learning":
        domain_config = config.get_domain_config()
        domain = FewShotLearningDomain(
            max_augmentations=domain_config.max_augmentations,
            available_augmentations=domain_config.available_augmentations,
            logger=logger.logger
        )
    else:
        raise ValueError(f"Unknown domain: {config.seal_config.domain}")
    
    return domain


def load_data(config: Config, logger):
    """Load and preprocess data"""
    logger.info("Loading data...")
    
    # Load data
    data_loader = DataLoader(
        domain=config.seal_config.domain,
        config=config.get_data_config(),
        logger=logger.logger
    )
    
    dataset = data_loader.load_data()
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Preprocess data
    preprocessor = DataPreprocessor(
        domain=config.seal_config.domain,
        config=config.get_data_config(),
        logger=logger.logger
    )
    
    processed_data = preprocessor.preprocess_data(dataset.data)
    logger.info(f"After preprocessing: {len(processed_data)} examples")
    
    # Get preprocessing statistics
    stats = preprocessor.get_preprocessing_stats(processed_data)
    logger.info("Data statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"  {key}: {value}")
    
    return processed_data


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Validate configuration
    issues = config.validate_config()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(config.seal_config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = None
    if config.seal_config.log_file:
        log_file = config.seal_config.log_file
    elif args.log_file:
        log_file = args.log_file
    else:
        log_file = str(output_dir / "train.log")
    
    wandb_config = None
    if config.seal_config.wandb_project:
        wandb_config = config.seal_config.to_dict()
    
    logger = create_logger(
        name="seal_training",
        level=config.seal_config.log_level,
        log_file=log_file,
        wandb_project=config.seal_config.wandb_project,
        wandb_config=wandb_config
    )
    
    try:
        # Setup domain
        domain = setup_domain(config, logger)
        
        # Load data
        data = load_data(config, logger)
        
        # Setup reward model
        reward_model = BinaryRewardModel(
            threshold=config.seal_config.reward_threshold,
            logger=logger.logger
        )
        
        # Initialize SEAL framework
        logger.info("Initializing SEAL framework...")
        seal_config = SEALConfig(
            model_name=config.seal_config.model_name,
            max_outer_iterations=config.seal_config.max_outer_iterations,
            samples_per_iteration=config.seal_config.samples_per_iteration,
            reward_threshold=config.seal_config.reward_threshold,
            lora_r=config.seal_config.lora_r,
            lora_alpha=config.seal_config.lora_alpha,
            lora_dropout=config.seal_config.lora_dropout,
            learning_rate=config.seal_config.learning_rate,
            device=config.seal_config.device,
            seed=config.seal_config.seed
        )
        
        seal = SEALFramework(
            config=seal_config,
            domain=domain,
            logger=logger.logger
        )
        
        # Setup metrics tracking
        metrics = Metrics(
            domain=config.seal_config.domain,
            save_path=str(output_dir / "metrics.json")
        )
        
        # Log training start
        logger.log_training_start(
            config=config.seal_config.to_dict(),
            dataset_size=len(data)
        )
        
        # Start training
        start_time = time.time()
        training_results = seal.train(data)
        total_time = time.time() - start_time
        
        # Save final checkpoint
        final_checkpoint_path = output_dir / "final_checkpoint.pt"
        seal.save_checkpoint(str(final_checkpoint_path))
        logger.log_checkpoint(str(final_checkpoint_path), seal.iteration)
        
        # Save metrics
        metrics.save()
        
        # Generate final report
        final_metrics = {
            "final_iteration": training_results["final_iteration"],
            "total_training_time": total_time,
            **metrics.get_tracker().get_all_stats()
        }
        
        # Log training completion
        logger.log_training_end(final_metrics, total_time)
        
        # Print summary report
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        print(metrics.generate_report())
        
        # Save final configuration
        config.save_config(str(output_dir / "final_config.yaml"))
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.log_error(e, "training")
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    main()