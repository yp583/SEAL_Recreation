#!/usr/bin/env python3
"""
Example usage of the SEAL framework

This script demonstrates how to use SEAL for both knowledge incorporation
and few-shot learning domains with various configurations.
"""

import sys
from pathlib import Path

# Add SEAL to path
sys.path.append(str(Path(__file__).parent))

from seal import SEALFramework
from seal.core.framework import SEALConfig
from seal.domains.knowledge_incorporation import KnowledgeIncorporationDomain
from seal.domains.few_shot_learning import FewShotLearningDomain
from seal.models.reward_model import BinaryRewardModel
from seal.data.loaders import DataLoader
from seal.data.preprocessors import DataPreprocessor
from seal.utils.config import ConfigBuilder, create_default_configs
from seal.utils.logging import create_logger
from seal.utils.metrics import Metrics


def example_knowledge_incorporation():
    """Example: Knowledge Incorporation with SQuAD data"""
    print("="*60)
    print("EXAMPLE 1: Knowledge Incorporation")
    print("="*60)
    
    # Create configuration
    config = (
        ConfigBuilder()
        .set_model("microsoft/DialoGPT-small")  # Small model for quick testing
        .set_training(max_iterations=10, samples_per_iteration=2)
        .set_lora(r=8, alpha=16)
        .set_domain("knowledge_incorporation", {
            "dataset_name": "squad",
            "max_samples": 5,  # Very small for testing
            "prompt_type": "implications"
        })
        .set_logging("INFO")
        .build()
    )
    
    # Setup logging
    logger = create_logger("seal_example_ki", level="INFO")
    
    # Setup domain
    domain = KnowledgeIncorporationDomain(
        max_implications=5,
        max_implication_length=100,
        logger=logger.logger
    )
    
    # Load and preprocess data
    logger.info("Loading SQuAD data...")
    data_loader = DataLoader(
        domain="knowledge_incorporation",
        config={
            "dataset_name": "squad",
            "max_samples": 5,
            "split": "train"
        },
        logger=logger.logger
    )
    
    try:
        dataset = data_loader.load_data()
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Take first few examples for demonstration
        demo_data = dataset.data[:3]
        
        # Initialize SEAL framework
        seal = SEALFramework(
            config=config,
            domain=domain,
            logger=logger.logger
        )
        
        # Setup metrics
        metrics = Metrics(domain="knowledge_incorporation")
        
        logger.info("Starting knowledge incorporation training...")
        
        # Train for a few iterations
        results = seal.train(demo_data)
        
        logger.info("Knowledge incorporation example completed!")
        print(metrics.generate_report())
        
    except Exception as e:
        logger.error(f"Knowledge incorporation example failed: {e}")
    
    finally:
        logger.finish()


def example_few_shot_learning():
    """Example: Few-Shot Learning with synthetic ARC data"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Few-Shot Learning")
    print("="*60)
    
    # Create configuration
    config = (
        ConfigBuilder()
        .set_model("microsoft/DialoGPT-small")
        .set_training(max_iterations=5, samples_per_iteration=2)
        .set_lora(r=4, alpha=8)
        .set_domain("few_shot_learning", {
            "max_samples": 3,
            "max_examples_per_task": 3
        })
        .set_logging("INFO")
        .build()
    )
    
    # Setup logging
    logger = create_logger("seal_example_fsl", level="INFO")
    
    # Setup domain
    domain = FewShotLearningDomain(
        max_augmentations=5,
        logger=logger.logger
    )
    
    # Load synthetic data (since ARC dataset may not be available)
    logger.info("Loading synthetic ARC data...")
    data_loader = DataLoader(
        domain="few_shot_learning",
        config={
            "max_samples": 3,
            "split": "train"
        },
        logger=logger.logger
    )
    
    try:
        dataset = data_loader.load_data()
        logger.info(f"Loaded {len(dataset)} examples")
        
        demo_data = dataset.data
        
        # Initialize SEAL framework
        seal = SEALFramework(
            config=config,
            domain=domain,
            logger=logger.logger
        )
        
        # Setup metrics
        metrics = Metrics(domain="few_shot_learning")
        
        logger.info("Starting few-shot learning training...")
        
        # Train for a few iterations
        results = seal.train(demo_data)
        
        logger.info("Few-shot learning example completed!")
        print(metrics.generate_report())
        
    except Exception as e:
        logger.error(f"Few-shot learning example failed: {e}")
    
    finally:
        logger.finish()


def example_custom_configuration():
    """Example: Custom configuration and domain usage"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Configuration")
    print("="*60)
    
    # Load a preset configuration and modify it
    presets = create_default_configs()
    config = presets["test"]  # Start with test preset
    
    # Modify configuration
    config.model_name = "microsoft/DialoGPT-small"
    config.max_outer_iterations = 5
    config.samples_per_iteration = 2
    config.data_config = {
        "dataset_name": "squad",
        "max_samples": 3
    }
    
    print("Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Domain: {config.domain}")
    print(f"  Max iterations: {config.max_outer_iterations}")
    print(f"  Samples per iteration: {config.samples_per_iteration}")
    print(f"  LoRA rank: {config.lora_r}")
    
    # Save configuration
    config.save("example_config.yaml")
    print("  Saved configuration to: example_config.yaml")
    
    # Load configuration back
    loaded_config = SEALConfig.load("example_config.yaml")
    print(f"  Loaded configuration domain: {loaded_config.domain}")


def example_self_edit_generation():
    """Example: Manual self-edit generation and evaluation"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Self-Edit Generation")
    print("="*60)
    
    # Setup domain
    domain = KnowledgeIncorporationDomain()
    
    # Example context
    from seal.domains.knowledge_incorporation import KnowledgeIncorporationContext
    
    context = KnowledgeIncorporationContext(
        passage="The Apollo program was the third United States human spaceflight program. "
                "It accomplished landing humans on the Moon and bringing them safely back to Earth. "
                "The program was carried out from 1961 to 1975.",
        title="Apollo Program"
    )
    
    # Generate self-edit prompt
    prompt = domain.create_self_edit_prompt(context)
    print("Generated prompt:")
    print(prompt)
    print("\n" + "-"*40)
    
    # Example self-edit (normally generated by model)
    example_self_edit = """1. The Apollo program was a human spaceflight initiative by the United States.
2. The primary goal was to land humans on the Moon and return them safely.
3. The program spanned from 1961 to 1975, lasting 14 years.
4. It was the third major spaceflight program in the US.
5. The Apollo program successfully achieved lunar landings."""
    
    # Post-process the self-edit
    processed_edit = domain.post_process_self_edit(example_self_edit, context)
    print("Processed self-edit:")
    print(processed_edit)


def example_metrics_and_analysis():
    """Example: Metrics tracking and analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Metrics and Analysis")
    print("="*60)
    
    # Create metrics tracker
    metrics = Metrics(domain="knowledge_incorporation")
    
    # Simulate some training iterations
    import random
    
    print("Simulating training iterations...")
    for iteration in range(10):
        # Simulate metrics
        rewards = [random.random() for _ in range(4)]
        performance_details = []
        
        for reward in rewards:
            performance_details.append({
                "baseline_performance": {"accuracy": random.uniform(0.2, 0.4)},
                "adapted_performance": {"accuracy": random.uniform(0.3, 0.6)},
                "reward": reward
            })
        
        # Compute iteration metrics
        iteration_metrics = metrics.compute_iteration_metrics(
            rewards, performance_details, iteration, random.uniform(10, 30)
        )
        
        if iteration % 3 == 0:
            print(f"Iteration {iteration}: mean_reward={iteration_metrics['mean_reward']:.3f}, "
                  f"success_rate={iteration_metrics['success_rate']:.3f}")
    
    # Generate analysis
    analysis = metrics.analyze_training()
    print("\nTraining Analysis:")
    if "reward_trend" in analysis:
        trend = analysis["reward_trend"]
        print(f"  Reward improvement: {trend['improvement']:+.3f}")
    
    # Generate report
    print("\n" + "-"*40)
    print("TRAINING REPORT")
    print("-"*40)
    report = metrics.generate_report()
    print(report)


def main():
    """Run all examples"""
    print("SEAL Framework Examples")
    print("This script demonstrates various features of the SEAL framework.")
    print("Note: Some examples use small models and limited data for quick execution.")
    
    try:
        # Example 1: Knowledge Incorporation
        example_knowledge_incorporation()
        
        # Example 2: Few-Shot Learning  
        example_few_shot_learning()
        
        # Example 3: Custom Configuration
        example_custom_configuration()
        
        # Example 4: Self-Edit Generation
        example_self_edit_generation()
        
        # Example 5: Metrics and Analysis
        example_metrics_and_analysis()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED")
        print("="*60)
        print("Check the output above for results from each example.")
        print("For full training, use the train.py script with appropriate configurations.")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()