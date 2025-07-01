"""
Few-Shot Learning Domain Implementation

Implements the few-shot learning instantiation of SEAL for the ARC 
(Abstraction and Reasoning Corpus) benchmark using Test-Time Training (TTT).

The domain generates self-edits specifying data augmentations and training
hyperparameters, then evaluates on held-out test inputs.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import json
import random
import numpy as np
from dataclasses import dataclass
from ..core.self_edit import BaseDomain


@dataclass
class ARCExample:
    """Single ARC input-output demonstration"""
    input_grid: List[List[int]]
    output_grid: List[List[int]]


@dataclass
class ARCTask:
    """ARC task with demonstrations and test case"""
    task_id: str
    train_examples: List[ARCExample]
    test_input: List[List[int]]
    test_output: Optional[List[List[int]]] = None  # Ground truth for evaluation


@dataclass
class FewShotLearningContext:
    """Context for few-shot learning (ARC demonstrations)"""
    task: ARCTask
    

class FewShotLearningDomain(BaseDomain):
    """
    Few-Shot Learning domain implementation for ARC tasks.
    
    This domain:
    1. Takes ARC demonstrations and generates self-edits specifying:
       - Data augmentations to apply
       - Training hyperparameters 
    2. Applies the self-edit to create augmented training data
    3. Fine-tunes the model using the specified parameters
    4. Evaluates on the held-out test input
    5. Rewards self-edits that improve test performance
    """
    
    def __init__(
        self,
        max_augmentations: int = 10,
        available_augmentations: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.max_augmentations = max_augmentations
        self.logger = logger or logging.getLogger(__name__)
        
        # Available augmentation tools
        if available_augmentations is None:
            self.available_augmentations = [
                "rotation_90", "rotation_180", "rotation_270",
                "flip_horizontal", "flip_vertical", 
                "transpose", "reflection_diagonal",
                "size_augmentation", "color_permutation",
                "chain_augmentations", "repeat_augmentations"
            ]
        else:
            self.available_augmentations = available_augmentations
            
        # Available hyperparameters
        self.available_hyperparams = {
            "learning_rate": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
            "epochs": [1, 2, 3, 5, 10],
            "loss_strategy": ["all_tokens", "output_only"],
            "batch_size": [1, 2, 4]
        }
        
    def create_self_edit_prompt(self, context: FewShotLearningContext) -> str:
        """
        Create a prompt for generating self-edits (augmentation + hyperparameter specs).
        
        Args:
            context: FewShotLearningContext containing ARC task
            
        Returns:
            Formatted prompt string
        """
        task = context.task
        
        # Format the ARC examples for the prompt
        examples_text = ""
        for i, example in enumerate(task.train_examples):
            examples_text += f"Example {i+1}:\n"
            examples_text += f"Input: {self._format_grid(example.input_grid)}\n"
            examples_text += f"Output: {self._format_grid(example.output_grid)}\n\n"
            
        # Create prompt for self-edit generation
        prompt = f"""Given the following ARC (Abstraction and Reasoning Corpus) task examples, generate a configuration for test-time training that specifies:

1. Data augmentations to apply
2. Training hyperparameters

Task Examples:
{examples_text}

Available augmentations: {', '.join(self.available_augmentations)}
Available learning rates: {self.available_hyperparams['learning_rate']}
Available epochs: {self.available_hyperparams['epochs']}
Available loss strategies: {self.available_hyperparams['loss_strategy']}

Generate a JSON configuration for optimal test-time training:

Configuration:"""
        
        return prompt
    
    def post_process_self_edit(
        self, 
        self_edit: str, 
        context: FewShotLearningContext
    ) -> str:
        """
        Post-process and validate the generated self-edit configuration.
        
        Args:
            self_edit: Raw generated self-edit
            context: Original context
            
        Returns:
            Processed and validated self-edit
        """
        # Clean up the self-edit
        self_edit = self_edit.strip()
        
        # Try to extract JSON from the self-edit
        try:
            # Look for JSON-like content
            start_idx = self_edit.find('{')
            end_idx = self_edit.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = self_edit[start_idx:end_idx]
                config = json.loads(json_str)
            else:
                # If no JSON found, create a default configuration
                config = self._create_default_config()
        
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, create a default configuration
            self.logger.warning("Failed to parse self-edit JSON, using default configuration")
            config = self._create_default_config()
        
        # Validate and fix the configuration
        config = self._validate_config(config)
        
        # Convert back to formatted JSON
        processed_self_edit = json.dumps(config, indent=2)
        
        self.logger.debug(f"Processed self-edit configuration: {config}")
        
        return processed_self_edit
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration"""
        return {
            "basic_augmentations": True,
            "size_augmentations": False,
            "chain_augmentations": False,
            "repeat_augmentations": False,
            "learning_rate": random.choice(self.available_hyperparams["learning_rate"]),
            "epochs": random.choice(self.available_hyperparams["epochs"]),
            "loss_strategy": random.choice(self.available_hyperparams["loss_strategy"]),
            "batch_size": 1
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix configuration parameters"""
        validated_config = {}
        
        # Augmentation flags
        validated_config["basic_augmentations"] = config.get("basic_augmentations", True)
        validated_config["size_augmentations"] = config.get("size_augmentations", False)
        validated_config["chain_augmentations"] = config.get("chain_augmentations", False)
        validated_config["repeat_augmentations"] = config.get("repeat_augmentations", False)
        
        # Learning rate
        lr = config.get("learning_rate", 1e-5)
        if lr not in self.available_hyperparams["learning_rate"]:
            lr = min(self.available_hyperparams["learning_rate"], 
                    key=lambda x: abs(x - lr))
        validated_config["learning_rate"] = lr
        
        # Epochs
        epochs = config.get("epochs", 3)
        if epochs not in self.available_hyperparams["epochs"]:
            epochs = min(self.available_hyperparams["epochs"],
                        key=lambda x: abs(x - epochs))
        validated_config["epochs"] = epochs
        
        # Loss strategy
        loss_strategy = config.get("loss_strategy", "all_tokens")
        if loss_strategy not in self.available_hyperparams["loss_strategy"]:
            loss_strategy = "all_tokens"
        validated_config["loss_strategy"] = loss_strategy
        
        # Batch size
        batch_size = config.get("batch_size", 1)
        if batch_size not in self.available_hyperparams["batch_size"]:
            batch_size = 1
        validated_config["batch_size"] = batch_size
        
        return validated_config
    
    def evaluate(
        self, 
        model: torch.nn.Module, 
        tokenizer: Any, 
        task: ARCTask
    ) -> Dict[str, float]:
        """
        Evaluate model performance on ARC task.
        
        Args:
            model: Language model to evaluate
            tokenizer: Model tokenizer
            task: ARC task
            
        Returns:
            Performance metrics
        """
        model.eval()
        
        # Generate prediction for test input
        predicted_output = self._generate_arc_output(
            model, tokenizer, task.test_input, task.train_examples
        )
        
        # Compute accuracy if ground truth is available
        if task.test_output is not None:
            accuracy = self._compute_grid_accuracy(predicted_output, task.test_output)
        else:
            # If no ground truth, use a heuristic score
            accuracy = self._heuristic_score(predicted_output, task.test_input)
        
        performance = {
            "accuracy": accuracy,
            "predicted_output": predicted_output
        }
        
        self.logger.debug(f"ARC evaluation: {accuracy:.3f} accuracy")
        
        return performance
    
    def compute_reward(
        self, 
        performance: Dict[str, float], 
        task: ARCTask
    ) -> float:
        """
        Compute reward based on performance metrics.
        
        Args:
            performance: Performance metrics from evaluate()
            task: ARC task
            
        Returns:
            Reward value (binary: 1.0 for improvement, 0.0 otherwise)
        """
        accuracy = performance["accuracy"]
        
        # Binary reward: 1.0 if accuracy is above threshold, 0.0 otherwise
        threshold = 0.5  # Can be adjusted based on task difficulty
        binary_reward = 1.0 if accuracy > threshold else 0.0
        
        return binary_reward
    
    def apply_augmentations(
        self,
        examples: List[ARCExample],
        config: Dict[str, Any]
    ) -> List[ARCExample]:
        """
        Apply data augmentations based on configuration.
        
        Args:
            examples: Original examples
            config: Augmentation configuration
            
        Returns:
            Augmented examples
        """
        augmented_examples = list(examples)  # Start with original examples
        
        for example in examples:
            # Basic augmentations (rotations, flips)
            if config.get("basic_augmentations", False):
                augmented_examples.extend(self._apply_basic_augmentations(example))
            
            # Size augmentations
            if config.get("size_augmentations", False):
                augmented_examples.extend(self._apply_size_augmentations(example))
            
            # Chain augmentations (combine multiple transformations)
            if config.get("chain_augmentations", False):
                augmented_examples.extend(self._apply_chain_augmentations(example))
            
            # Repeat augmentations
            if config.get("repeat_augmentations", False):
                augmented_examples.extend(self._apply_repeat_augmentations(example))
        
        self.logger.debug(f"Generated {len(augmented_examples)} augmented examples "
                         f"from {len(examples)} original examples")
        
        return augmented_examples
    
    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for text representation"""
        return str(grid)
    
    def _generate_arc_output(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        test_input: List[List[int]],
        train_examples: List[ARCExample],
        max_length: int = 200
    ) -> List[List[int]]:
        """Generate output for ARC test input using the model"""
        
        # Create prompt with train examples and test input
        prompt = "ARC Task Examples:\n"
        for i, example in enumerate(train_examples):
            prompt += f"Input {i+1}: {self._format_grid(example.input_grid)}\n"
            prompt += f"Output {i+1}: {self._format_grid(example.output_grid)}\n\n"
        
        prompt += f"Test Input: {self._format_grid(test_input)}\n"
        prompt += "Test Output:"
        
        # Tokenize and generate
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        # Decode generated tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Parse the generated text to extract grid
        try:
            # Look for grid-like patterns in the generated text
            predicted_output = self._parse_grid_from_text(generated_text)
        except:
            # If parsing fails, return a default grid
            predicted_output = [[0] * len(test_input[0])] * len(test_input)
            
        return predicted_output
    
    def _parse_grid_from_text(self, text: str) -> List[List[int]]:
        """Parse a grid from generated text"""
        # Simple parsing - look for list-like structures
        import ast
        
        # Try to find and evaluate list structures in the text
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    grid = ast.literal_eval(line)
                    if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                        return grid
                except:
                    continue
        
        # If no valid grid found, return a 3x3 default
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    def _compute_grid_accuracy(
        self, 
        predicted: List[List[int]], 
        ground_truth: List[List[int]]
    ) -> float:
        """Compute accuracy between predicted and ground truth grids"""
        if len(predicted) != len(ground_truth):
            return 0.0
            
        total_cells = 0
        correct_cells = 0
        
        for i in range(len(predicted)):
            if len(predicted[i]) != len(ground_truth[i]):
                continue
                
            for j in range(len(predicted[i])):
                total_cells += 1
                if predicted[i][j] == ground_truth[i][j]:
                    correct_cells += 1
        
        return correct_cells / max(total_cells, 1)
    
    def _heuristic_score(
        self, 
        predicted: List[List[int]], 
        test_input: List[List[int]]
    ) -> float:
        """Compute a heuristic score when ground truth is not available"""
        # Simple heuristic: prefer non-trivial outputs
        total_cells = sum(len(row) for row in predicted)
        non_zero_cells = sum(sum(1 for cell in row if cell != 0) for row in predicted)
        
        if total_cells == 0:
            return 0.0
            
        # Score based on diversity of output
        diversity_score = non_zero_cells / total_cells
        
        # Prefer outputs that are different from input
        if predicted != test_input:
            diversity_score += 0.2
            
        return min(diversity_score, 1.0)
    
    # Augmentation methods
    def _apply_basic_augmentations(self, example: ARCExample) -> List[ARCExample]:
        """Apply basic augmentations (rotations, flips)"""
        augmented = []
        
        # Rotation 90 degrees
        rotated_input = self._rotate_grid_90(example.input_grid)
        rotated_output = self._rotate_grid_90(example.output_grid)
        augmented.append(ARCExample(rotated_input, rotated_output))
        
        # Horizontal flip
        flipped_input = self._flip_grid_horizontal(example.input_grid)
        flipped_output = self._flip_grid_horizontal(example.output_grid)
        augmented.append(ARCExample(flipped_input, flipped_output))
        
        return augmented
    
    def _apply_size_augmentations(self, example: ARCExample) -> List[ARCExample]:
        """Apply size-based augmentations"""
        # For now, return empty list (size augmentation is complex for ARC)
        return []
    
    def _apply_chain_augmentations(self, example: ARCExample) -> List[ARCExample]:
        """Apply chained augmentations"""
        # Combine rotation and flip
        rotated_input = self._rotate_grid_90(example.input_grid)
        rotated_output = self._rotate_grid_90(example.output_grid)
        
        chained_input = self._flip_grid_horizontal(rotated_input)
        chained_output = self._flip_grid_horizontal(rotated_output)
        
        return [ARCExample(chained_input, chained_output)]
    
    def _apply_repeat_augmentations(self, example: ARCExample) -> List[ARCExample]:
        """Apply repeated augmentations"""
        # Apply the same augmentation multiple times with slight variations
        repeated = []
        
        # Multiple rotations
        input_grid = example.input_grid
        output_grid = example.output_grid
        
        for _ in range(2):
            input_grid = self._rotate_grid_90(input_grid)
            output_grid = self._rotate_grid_90(output_grid)
            repeated.append(ARCExample(input_grid, output_grid))
            
        return repeated
    
    # Grid transformation utilities
    def _rotate_grid_90(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 90 degrees clockwise"""
        if not grid or not grid[0]:
            return grid
            
        rows, cols = len(grid), len(grid[0])
        rotated = [[0] * rows for _ in range(cols)]
        
        for i in range(rows):
            for j in range(cols):
                rotated[j][rows - 1 - i] = grid[i][j]
                
        return rotated
    
    def _flip_grid_horizontal(self, grid: List[List[int]]) -> List[List[int]]:
        """Flip grid horizontally"""
        return [row[::-1] for row in grid]
    
    def _flip_grid_vertical(self, grid: List[List[int]]) -> List[List[int]]:
        """Flip grid vertically"""
        return grid[::-1]