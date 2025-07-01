"""
Data Preprocessing Utilities for SEAL

Implements preprocessing functions for different domains and data formats.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import re
import json
from collections import defaultdict

from ..domains.knowledge_incorporation import (
    KnowledgeIncorporationTask, 
    KnowledgeIncorporationContext
)
from ..domains.few_shot_learning import ARCTask, ARCExample


class BasePreprocessor:
    """Base class for data preprocessors"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def preprocess(self, data: Any) -> Any:
        """Preprocess data - to be implemented by subclasses"""
        raise NotImplementedError


class KnowledgeIncorporationPreprocessor(BasePreprocessor):
    """Preprocessor for Knowledge Incorporation domain"""
    
    def __init__(
        self,
        max_passage_length: int = 1000,
        max_question_length: int = 200,
        max_answer_length: int = 100,
        min_passage_length: int = 50,
        filter_duplicates: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.max_passage_length = max_passage_length
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.min_passage_length = min_passage_length
        self.filter_duplicates = filter_duplicates
        
    def preprocess_task(self, task: KnowledgeIncorporationTask) -> Optional[KnowledgeIncorporationTask]:
        """
        Preprocess a knowledge incorporation task.
        
        Args:
            task: Original task
            
        Returns:
            Preprocessed task or None if task should be filtered out
        """
        # Clean passage
        cleaned_passage = self._clean_text(task.passage)
        
        # Filter by length
        if len(cleaned_passage) < self.min_passage_length:
            return None
        if len(cleaned_passage) > self.max_passage_length:
            cleaned_passage = cleaned_passage[:self.max_passage_length]
            
        # Clean title
        cleaned_title = self._clean_text(task.title) if task.title else ""
        
        # Process questions
        cleaned_questions = []
        for qa in task.questions:
            question = self._clean_text(qa["question"])
            answer = self._clean_text(qa["answer"])
            
            # Filter by length
            if len(question) > self.max_question_length:
                question = question[:self.max_question_length]
            if len(answer) > self.max_answer_length:
                answer = answer[:self.max_answer_length]
                
            # Skip empty questions/answers
            if not question.strip() or not answer.strip():
                continue
                
            cleaned_questions.append({
                "question": question,
                "answer": answer
            })
        
        # Filter out tasks with no valid questions
        if not cleaned_questions:
            return None
            
        # Filter duplicates
        if self.filter_duplicates:
            cleaned_questions = self._filter_duplicate_questions(cleaned_questions)
            
        if not cleaned_questions:
            return None
            
        return KnowledgeIncorporationTask(
            passage=cleaned_passage,
            title=cleaned_title,
            questions=cleaned_questions
        )
    
    def preprocess_context(self, context: KnowledgeIncorporationContext) -> KnowledgeIncorporationContext:
        """Preprocess a knowledge incorporation context"""
        cleaned_passage = self._clean_text(context.passage)
        cleaned_title = self._clean_text(context.title) if context.title else ""
        
        # Truncate if too long
        if len(cleaned_passage) > self.max_passage_length:
            cleaned_passage = cleaned_passage[:self.max_passage_length]
            
        return KnowledgeIncorporationContext(
            passage=cleaned_passage,
            title=cleaned_title
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\?\!\,\:\;\-\(\)\[\]\{\}\"\'\/]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _filter_duplicate_questions(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter out duplicate questions"""
        seen_questions = set()
        filtered_questions = []
        
        for qa in questions:
            question_normalized = qa["question"].lower().strip()
            if question_normalized not in seen_questions:
                seen_questions.add(question_normalized)
                filtered_questions.append(qa)
                
        return filtered_questions


class FewShotLearningPreprocessor(BasePreprocessor):
    """Preprocessor for Few-Shot Learning domain (ARC)"""
    
    def __init__(
        self,
        max_grid_size: int = 30,
        min_grid_size: int = 1,
        max_examples_per_task: int = 10,
        min_examples_per_task: int = 1,
        normalize_colors: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.max_grid_size = max_grid_size
        self.min_grid_size = min_grid_size
        self.max_examples_per_task = max_examples_per_task
        self.min_examples_per_task = min_examples_per_task
        self.normalize_colors = normalize_colors
        
    def preprocess_task(self, task: ARCTask) -> Optional[ARCTask]:
        """
        Preprocess an ARC task.
        
        Args:
            task: Original ARC task
            
        Returns:
            Preprocessed task or None if task should be filtered out
        """
        # Process training examples
        processed_examples = []
        for example in task.train_examples:
            processed_example = self._preprocess_example(example)
            if processed_example is not None:
                processed_examples.append(processed_example)
        
        # Filter by number of examples
        if len(processed_examples) < self.min_examples_per_task:
            return None
        if len(processed_examples) > self.max_examples_per_task:
            processed_examples = processed_examples[:self.max_examples_per_task]
            
        # Process test input
        processed_test_input = self._preprocess_grid(task.test_input)
        if processed_test_input is None:
            return None
            
        # Process test output if available
        processed_test_output = None
        if task.test_output is not None:
            processed_test_output = self._preprocess_grid(task.test_output)
            
        return ARCTask(
            task_id=task.task_id,
            train_examples=processed_examples,
            test_input=processed_test_input,
            test_output=processed_test_output
        )
    
    def _preprocess_example(self, example: ARCExample) -> Optional[ARCExample]:
        """Preprocess a single ARC example"""
        input_grid = self._preprocess_grid(example.input_grid)
        output_grid = self._preprocess_grid(example.output_grid)
        
        if input_grid is None or output_grid is None:
            return None
            
        return ARCExample(
            input_grid=input_grid,
            output_grid=output_grid
        )
    
    def _preprocess_grid(self, grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Preprocess a single grid"""
        if not grid or not grid[0]:
            return None
            
        # Check grid dimensions
        height = len(grid)
        width = len(grid[0])
        
        if height < self.min_grid_size or height > self.max_grid_size:
            return None
        if width < self.min_grid_size or width > self.max_grid_size:
            return None
            
        # Ensure all rows have the same length
        for row in grid:
            if len(row) != width:
                return None
                
        # Copy grid to avoid modifying original
        processed_grid = [row[:] for row in grid]
        
        # Normalize colors if requested
        if self.normalize_colors:
            processed_grid = self._normalize_grid_colors(processed_grid)
            
        return processed_grid
    
    def _normalize_grid_colors(self, grid: List[List[int]]) -> List[List[int]]:
        """Normalize colors in a grid to start from 0"""
        # Find unique colors
        unique_colors = set()
        for row in grid:
            unique_colors.update(row)
            
        # Create mapping from old colors to new colors (0, 1, 2, ...)
        color_mapping = {color: i for i, color in enumerate(sorted(unique_colors))}
        
        # Apply mapping
        normalized_grid = []
        for row in grid:
            normalized_row = [color_mapping[color] for color in row]
            normalized_grid.append(normalized_row)
            
        return normalized_grid


class DataPreprocessor:
    """Main preprocessor that handles different domains"""
    
    def __init__(
        self,
        domain: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        self.domain = domain
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize domain-specific preprocessor
        if domain == "knowledge_incorporation":
            self.preprocessor = KnowledgeIncorporationPreprocessor(
                max_passage_length=config.get("max_passage_length", 1000),
                max_question_length=config.get("max_question_length", 200),
                max_answer_length=config.get("max_answer_length", 100),
                min_passage_length=config.get("min_passage_length", 50),
                filter_duplicates=config.get("filter_duplicates", True),
                logger=logger
            )
        elif domain == "few_shot_learning":
            self.preprocessor = FewShotLearningPreprocessor(
                max_grid_size=config.get("max_grid_size", 30),
                min_grid_size=config.get("min_grid_size", 1),
                max_examples_per_task=config.get("max_examples_per_task", 10),
                min_examples_per_task=config.get("min_examples_per_task", 1),
                normalize_colors=config.get("normalize_colors", True),
                logger=logger
            )
        else:
            raise ValueError(f"Unknown domain: {domain}")
    
    def preprocess_data(
        self, 
        data: List[Tuple[Any, Any]]
    ) -> List[Tuple[Any, Any]]:
        """
        Preprocess a list of (context, task) pairs.
        
        Args:
            data: List of (context, task) pairs
            
        Returns:
            List of preprocessed (context, task) pairs
        """
        preprocessed_data = []
        filtered_count = 0
        
        for context, task in data:
            # Preprocess based on domain
            if self.domain == "knowledge_incorporation":
                processed_task = self.preprocessor.preprocess_task(task)
                if processed_task is not None:
                    processed_context = self.preprocessor.preprocess_context(context)
                    preprocessed_data.append((processed_context, processed_task))
                else:
                    filtered_count += 1
                    
            elif self.domain == "few_shot_learning":
                processed_task = self.preprocessor.preprocess_task(task)
                if processed_task is not None:
                    # Context contains the task, so update it
                    processed_context = type(context)(task=processed_task)
                    preprocessed_data.append((processed_context, processed_task))
                else:
                    filtered_count += 1
        
        self.logger.info(f"Preprocessing complete: {len(preprocessed_data)} samples kept, "
                        f"{filtered_count} samples filtered out")
        
        return preprocessed_data
    
    def get_preprocessing_stats(
        self, 
        data: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """Get statistics about the data after preprocessing"""
        if self.domain == "knowledge_incorporation":
            return self._get_knowledge_incorporation_stats(data)
        elif self.domain == "few_shot_learning":
            return self._get_few_shot_learning_stats(data)
        else:
            return {}
    
    def _get_knowledge_incorporation_stats(
        self, 
        data: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """Get statistics for knowledge incorporation data"""
        passage_lengths = []
        questions_per_task = []
        question_lengths = []
        answer_lengths = []
        
        for context, task in data:
            passage_lengths.append(len(context.passage))
            questions_per_task.append(len(task.questions))
            
            for qa in task.questions:
                question_lengths.append(len(qa["question"]))
                answer_lengths.append(len(qa["answer"]))
        
        stats = {
            "num_tasks": len(data),
            "passage_length": {
                "mean": np.mean(passage_lengths),
                "std": np.std(passage_lengths),
                "min": np.min(passage_lengths),
                "max": np.max(passage_lengths)
            },
            "questions_per_task": {
                "mean": np.mean(questions_per_task),
                "std": np.std(questions_per_task),
                "min": np.min(questions_per_task),
                "max": np.max(questions_per_task)
            },
            "question_length": {
                "mean": np.mean(question_lengths),
                "std": np.std(question_lengths),
                "min": np.min(question_lengths),
                "max": np.max(question_lengths)
            },
            "answer_length": {
                "mean": np.mean(answer_lengths),
                "std": np.std(answer_lengths),
                "min": np.min(answer_lengths),
                "max": np.max(answer_lengths)
            }
        }
        
        return stats
    
    def _get_few_shot_learning_stats(
        self, 
        data: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """Get statistics for few-shot learning data"""
        examples_per_task = []
        grid_sizes = []
        color_counts = []
        
        for context, task in data:
            examples_per_task.append(len(task.train_examples))
            
            # Analyze all grids (train inputs/outputs + test input)
            all_grids = []
            for example in task.train_examples:
                all_grids.extend([example.input_grid, example.output_grid])
            all_grids.append(task.test_input)
            if task.test_output:
                all_grids.append(task.test_output)
                
            for grid in all_grids:
                height, width = len(grid), len(grid[0])
                grid_sizes.append(height * width)
                
                # Count unique colors
                unique_colors = set()
                for row in grid:
                    unique_colors.update(row)
                color_counts.append(len(unique_colors))
        
        stats = {
            "num_tasks": len(data),
            "examples_per_task": {
                "mean": np.mean(examples_per_task),
                "std": np.std(examples_per_task),
                "min": np.min(examples_per_task),
                "max": np.max(examples_per_task)
            },
            "grid_size": {
                "mean": np.mean(grid_sizes),
                "std": np.std(grid_sizes),
                "min": np.min(grid_sizes),
                "max": np.max(grid_sizes)
            },
            "color_count": {
                "mean": np.mean(color_counts),
                "std": np.std(color_counts),
                "min": np.min(color_counts),
                "max": np.max(color_counts)
            }
        }
        
        return stats


class DataAugmentor:
    """Data augmentation utilities for SEAL training"""
    
    @staticmethod
    def augment_knowledge_incorporation_data(
        data: List[Tuple[Any, Any]],
        augmentation_factor: int = 2
    ) -> List[Tuple[Any, Any]]:
        """Augment knowledge incorporation data by paraphrasing"""
        # For now, just duplicate the data
        # In a real implementation, you might use paraphrasing models
        augmented_data = list(data)
        
        for _ in range(augmentation_factor - 1):
            augmented_data.extend(data)
            
        return augmented_data
    
    @staticmethod
    def augment_few_shot_learning_data(
        data: List[Tuple[Any, Any]],
        augmentation_types: List[str] = ["rotation", "flip"]
    ) -> List[Tuple[Any, Any]]:
        """Augment few-shot learning data with geometric transformations"""
        augmented_data = list(data)
        
        for context, task in data:
            if "rotation" in augmentation_types:
                # Rotate all grids 90 degrees
                rotated_examples = []
                for example in task.train_examples:
                    rotated_input = DataAugmentor._rotate_grid_90(example.input_grid)
                    rotated_output = DataAugmentor._rotate_grid_90(example.output_grid)
                    rotated_examples.append(ARCExample(rotated_input, rotated_output))
                
                rotated_test_input = DataAugmentor._rotate_grid_90(task.test_input)
                rotated_test_output = None
                if task.test_output:
                    rotated_test_output = DataAugmentor._rotate_grid_90(task.test_output)
                
                rotated_task = ARCTask(
                    task_id=task.task_id + "_rot90",
                    train_examples=rotated_examples,
                    test_input=rotated_test_input,
                    test_output=rotated_test_output
                )
                rotated_context = type(context)(task=rotated_task)
                augmented_data.append((rotated_context, rotated_task))
        
        return augmented_data
    
    @staticmethod
    def _rotate_grid_90(grid: List[List[int]]) -> List[List[int]]:
        """Rotate a grid 90 degrees clockwise"""
        rows, cols = len(grid), len(grid[0])
        rotated = [[0] * rows for _ in range(cols)]
        
        for i in range(rows):
            for j in range(cols):
                rotated[j][rows - 1 - i] = grid[i][j]
                
        return rotated