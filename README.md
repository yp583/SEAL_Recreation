# SEAL: Self-Adapting LLMs

A PyTorch implementation of **Self-Adapting LLMs (SEAL)**, a framework that enables language models to improve themselves by generating their own synthetic data and optimization parameters ("self-edits") in response to new data.

## Overview

SEAL implements a novel meta-learning approach with two nested loops:
- **Outer RL loop**: Optimizes self-edit generation using reinforcement learning
- **Inner update loop**: Applies generated self-edits via supervised fine-tuning with LoRA

The framework supports two domains:
1. **Knowledge Incorporation**: Efficiently incorporating passage information into model weights
2. **Few-Shot Learning**: Adapting to novel tasks using the ARC benchmark with test-time training

## Key Features

- 🔄 **Nested Loop Architecture**: Outer RL optimization + inner SFT updates
- 🎯 **ReSTEM Optimizer**: Rejection sampling + supervised fine-tuning for stable training
- ⚡ **LoRA Integration**: Efficient parameter updates for frequent adaptations
- 📊 **Comprehensive Metrics**: Detailed tracking of training dynamics and performance
- 🛠️ **Modular Design**: Easy to extend with new domains and reward models
- 📝 **Experiment Tracking**: Integration with Weights & Biases
- ⚙️ **Flexible Configuration**: YAML/JSON configuration system

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SEAL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

### Knowledge Incorporation with SQuAD

Train SEAL to incorporate passage information into model weights:

```bash
python train.py \
    --preset knowledge_incorporation_squad \
    --max-iterations 500 \
    --output-dir ./outputs/knowledge_incorporation
```

### Few-Shot Learning with ARC

Train SEAL for few-shot learning on ARC tasks:

```bash
python train.py \
    --preset few_shot_learning_arc \
    --max-iterations 1000 \
    --output-dir ./outputs/few_shot_learning
```

### Custom Configuration

Create a custom configuration file:

```yaml
# config.yaml
model_name: "microsoft/DialoGPT-medium"
domain: "knowledge_incorporation"
max_outer_iterations: 500
samples_per_iteration: 4
learning_rate: 1e-4

lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

data_config:
  dataset_name: "squad"
  max_samples: 1000
  prompt_type: "implications"

wandb_project: "seal-experiments"
```

Then run:
```bash
python train.py --config config.yaml
```

## Framework Architecture

```
SEAL Framework
├── Core Components
│   ├── SEALFramework     # Main orchestrator (Algorithm 1)
│   ├── SelfEditGenerator # Generates domain-specific self-edits
│   └── ReSTEMOptimizer   # Rejection sampling + SFT optimizer
├── Domains
│   ├── KnowledgeIncorporation # Passage → implications generation
│   └── FewShotLearning        # ARC tasks with TTT protocol
├── Models
│   ├── LoRAWrapper       # Efficient parameter updates
│   └── RewardModel       # Binary/continuous reward computation
├── Data
│   ├── DataLoader        # SQuAD, ARC, custom datasets
│   └── DataPreprocessor  # Domain-specific preprocessing
└── Utils
    ├── Config            # Configuration management
    ├── Metrics           # Training dynamics tracking
    └── Logging           # Standard + WandB logging
```

## Domains

### Knowledge Incorporation

**Goal**: Incorporate passage information into model weights for no-context question answering.

**Process**:
1. Generate "implications" from passages
2. Fine-tune model on these implications using LoRA
3. Evaluate on questions WITHOUT providing passage context
4. Reward self-edits that improve no-context accuracy

**Supported Datasets**: SQuAD, SQuAD v2, custom JSON format

**Self-Edit Types**:
- `implications`: Inferences and logical consequences
- `questions_answers`: Q&A pairs about the passage
- `rewrite`: Rewritten versions for memorization
- `facts`: Structured key facts extraction

### Few-Shot Learning

**Goal**: Adapt to novel ARC tasks using optimized test-time training.

**Process**:
1. Generate configuration specifying data augmentations and hyperparameters
2. Apply augmentations to training examples
3. Fine-tune model with specified parameters
4. Evaluate on held-out test input
5. Reward configurations that improve test performance

**Supported Datasets**: ARC (local JSON or HuggingFace), synthetic tasks

**Augmentation Types**:
- Geometric: rotations, flips, transpositions
- Size: grid resizing operations
- Chained: combinations of transformations
- Repeated: multiple applications with variations

## Configuration

### Main Configuration (`SEALConfig`)

```python
@dataclass
class SEALConfig:
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    device: str = "auto"
    
    # Training settings
    max_outer_iterations: int = 1000
    samples_per_iteration: int = 4
    learning_rate: float = 1e-4
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Domain and data
    domain: str = "knowledge_incorporation"
    data_config: Dict[str, Any] = field(default_factory=dict)
```

### Preset Configurations

The framework includes several preset configurations:

- `knowledge_incorporation_squad`: SQuAD-based knowledge incorporation
- `few_shot_learning_arc`: ARC-based few-shot learning  
- `test`: Quick testing with small model and limited data

## Training Process

The SEAL training loop implements Algorithm 1 from the paper:

```python
for iteration in range(max_iterations):
    # Sample (context, task) from dataset
    context, task = sample_from_dataset()
    
    # Generate self-edits using current policy
    self_edits = generate_self_edits(context)
    
    # Inner loop: apply each self-edit and evaluate
    for self_edit in self_edits:
        adapted_model = apply_self_edit(model, self_edit)  # LoRA SFT
        performance = evaluate(adapted_model, task)
        reward = compute_reward(performance)
    
    # Outer loop: update policy using ReSTEM
    update_policy(self_edits, rewards)
```

### ReSTEM Optimizer

ReSTEM (Rejection Sampling + Supervised Fine-Tuning) provides stable training:

1. **E-step**: Sample self-edits from current model policy
2. **M-step**: Fine-tune only on self-edits with positive rewards

This implements binary reward optimization:
```
r(SE, τ, θ) = 1  if adaptation improves performance
              0  otherwise
```

## Metrics and Analysis

### Training Metrics

- **Reward Statistics**: Mean, std, distribution of rewards
- **Success Rate**: Percentage of positive rewards
- **Performance Improvement**: Before/after adaptation comparison
- **Training Dynamics**: Trends, stability, potential issues

### Domain-Specific Metrics

**Knowledge Incorporation**:
- No-context question answering accuracy
- Baseline vs adapted performance
- Questions per task statistics

**Few-Shot Learning**:
- ARC task accuracy
- Grid-level prediction accuracy
- Augmentation effectiveness

### Visualizations

The framework provides training dynamics analysis:
- Reward trends over time
- Success rate evolution
- Performance improvement distribution
- Training stability metrics

## Extending SEAL

### Adding New Domains

1. Inherit from `BaseDomain`:
```python
class MyDomain(BaseDomain):
    def create_self_edit_prompt(self, context):
        # Generate prompt for self-edit creation
        pass
    
    def evaluate(self, model, tokenizer, task):
        # Evaluate model performance
        pass
    
    def compute_reward(self, performance, task):
        # Convert performance to reward
        pass
```

2. Add data loading and preprocessing support
3. Register domain in the main framework

### Custom Reward Models

Implement different reward strategies:
```python
class MyRewardModel(BaseRewardModel):
    def compute_reward(self, before, after, task):
        # Custom reward computation
        return reward_value
```

### Custom Data Sources

Add support for new datasets:
```python
class MyDataLoader:
    def load_data(self):
        # Load and format data
        return [(context, task), ...]
```

## Examples

See the `experiments/` directory for complete examples:

- `knowledge_incorporation/`: SQuAD training with different prompts
- `few_shot_learning/`: ARC training with various augmentations

## Citation

If you use this implementation, please cite the original SEAL paper:

```bibtex
@article{seal2024,
  title={Self-Adapting LLMs: Improving Performance through Self-Generated Synthetic Data},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `samples_per_iteration` or use smaller model
2. **Low success rates**: Adjust `reward_threshold` or try different prompts
3. **Slow training**: Enable LoRA, reduce `max_length`, use faster model

### Debug Mode

Run with debug logging to see detailed information:
```bash
python train.py --log-level DEBUG --preset test
```

### Performance Tips

- Use GPU when available (`--device cuda`)
- Start with preset configurations
- Monitor success rates - aim for 10-30%
- Use smaller models for initial experiments
- Enable wandb for experiment tracking

## Acknowledgments

This implementation is based on the SEAL paper and incorporates ideas from:
- LoRA: Low-Rank Adaptation of Large Language Models
- ReSTEM: Rejection Sampling for Test-Time Adaptation
- Test-Time Training for ARC tasks