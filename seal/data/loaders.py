"""
Data Loading Utilities for SEAL

Implements data loaders for different domains (Knowledge Incorporation, Few-Shot Learning)
with support for various datasets like SQuAD, ARC, etc.
"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import logging
import json
import random
from pathlib import Path
import numpy as np

from datasets import load_dataset, Dataset as HFDataset
from ..domains.knowledge_incorporation import (
    KnowledgeIncorporationTask, 
    KnowledgeIncorporationContext,
    KnowledgeIncorporationDataProcessor
)
from ..domains.few_shot_learning import ARCTask, ARCExample, FewShotLearningContext


class SEALDataset(Dataset):
    """Base dataset class for SEAL training"""
    
    def __init__(
        self,
        data: List[Tuple[Any, Any]],  # List of (context, task) pairs
        domain: str,
        logger: Optional[logging.Logger] = None
    ):
        self.data = data
        self.domain = domain
        self.logger = logger or logging.getLogger(__name__)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def shuffle(self):
        """Shuffle the dataset"""
        random.shuffle(self.data)
        
    def get_domain(self):
        return self.domain


class KnowledgeIncorporationDataLoader:
    """Data loader for Knowledge Incorporation domain"""
    
    def __init__(
        self,
        dataset_name: str = "squad",
        split: str = "train",
        max_samples: Optional[int] = None,
        min_questions_per_context: int = 1,
        logger: Optional[logging.Logger] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.min_questions_per_context = min_questions_per_context
        self.logger = logger or logging.getLogger(__name__)
        
    def load_squad_data(self) -> List[Tuple[KnowledgeIncorporationContext, KnowledgeIncorporationTask]]:
        """Load SQuAD dataset for knowledge incorporation"""
        self.logger.info(f"Loading SQuAD dataset: {self.split} split")
        
        # Load SQuAD dataset
        if self.dataset_name == "squad":
            dataset = load_dataset("squad", split=self.split)
        elif self.dataset_name == "squad_v2":
            dataset = load_dataset("squad_v2", split=self.split)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        # Group by context
        context_groups = {}
        for example in dataset:
            context = example['context']
            if context not in context_groups:
                context_groups[context] = {
                    'title': example.get('title', ''),
                    'questions': [],
                    'answers': []
                }
            
            context_groups[context]['questions'].append(example['question'])
            
            # Handle different answer formats
            if 'answers' in example:
                answer_text = example['answers']['text'][0] if example['answers']['text'] else ""
            else:
                answer_text = ""
                
            context_groups[context]['answers'].append(answer_text)
        
        # Convert to SEAL format
        seal_data = []
        for context_text, info in context_groups.items():
            # Skip contexts with too few questions
            if len(info['questions']) < self.min_questions_per_context:
                continue
                
            # Create task
            questions = [
                {"question": q, "answer": a} 
                for q, a in zip(info['questions'], info['answers'])
            ]
            
            task = KnowledgeIncorporationTask(
                passage=context_text,
                title=info['title'],
                questions=questions
            )
            
            context = KnowledgeIncorporationContext(
                passage=context_text,
                title=info['title']
            )
            
            seal_data.append((context, task))
            
            if self.max_samples and len(seal_data) >= self.max_samples:
                break
                
        self.logger.info(f"Loaded {len(seal_data)} knowledge incorporation examples")
        return seal_data
    
    def load_custom_data(self, data_path: str) -> List[Tuple[KnowledgeIncorporationContext, KnowledgeIncorporationTask]]:
        """Load custom knowledge incorporation data from JSON file"""
        self.logger.info(f"Loading custom data from: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        seal_data = []
        for item in data:
            task = KnowledgeIncorporationTask(
                passage=item['passage'],
                title=item.get('title', ''),
                questions=item['questions']
            )
            
            context = KnowledgeIncorporationContext(
                passage=item['passage'],
                title=item.get('title', '')
            )
            
            seal_data.append((context, task))
            
            if self.max_samples and len(seal_data) >= self.max_samples:
                break
                
        self.logger.info(f"Loaded {len(seal_data)} custom knowledge incorporation examples")
        return seal_data


class FewShotLearningDataLoader:
    """Data loader for Few-Shot Learning domain (ARC)"""
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
        min_examples_per_task: int = 1,
        logger: Optional[logging.Logger] = None
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.max_samples = max_samples
        self.min_examples_per_task = min_examples_per_task
        self.logger = logger or logging.getLogger(__name__)
        
    def load_arc_data(self) -> List[Tuple[FewShotLearningContext, ARCTask]]:
        """Load ARC dataset for few-shot learning"""
        self.logger.info(f"Loading ARC dataset: {self.split} split")
        
        if self.dataset_path is None:
            # Try to load from huggingface datasets
            try:
                dataset = load_dataset("arc", split=self.split)
                return self._process_hf_arc_data(dataset)
            except:
                self.logger.warning("Could not load ARC from huggingface, using synthetic data")
                return self._generate_synthetic_arc_data()
        else:
            # Load from local JSON file
            return self._load_arc_from_json(self.dataset_path)
    
    def _load_arc_from_json(self, data_path: str) -> List[Tuple[FewShotLearningContext, ARCTask]]:
        """Load ARC data from JSON file"""
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        seal_data = []
        for task_id, task_data in data.items():
            # Parse training examples
            train_examples = []
            for example in task_data['train']:
                train_examples.append(ARCExample(
                    input_grid=example['input'],
                    output_grid=example['output']
                ))
            
            # Skip tasks with too few examples
            if len(train_examples) < self.min_examples_per_task:
                continue
                
            # Parse test example
            test_example = task_data['test'][0]  # Assume single test example
            
            task = ARCTask(
                task_id=task_id,
                train_examples=train_examples,
                test_input=test_example['input'],
                test_output=test_example.get('output')  # May be None for evaluation
            )
            
            context = FewShotLearningContext(task=task)
            
            seal_data.append((context, task))
            
            if self.max_samples and len(seal_data) >= self.max_samples:
                break
                
        self.logger.info(f"Loaded {len(seal_data)} ARC examples")
        return seal_data
    
    def _process_hf_arc_data(self, dataset) -> List[Tuple[FewShotLearningContext, ARCTask]]:
        """Process ARC data from HuggingFace dataset"""
        seal_data = []
        
        for i, example in enumerate(dataset):
            # Extract training examples
            train_examples = []
            for train_ex in example.get('train', []):
                train_examples.append(ARCExample(
                    input_grid=train_ex['input'],
                    output_grid=train_ex['output']
                ))
            
            if len(train_examples) < self.min_examples_per_task:
                continue
                
            # Extract test example
            test_examples = example.get('test', [])
            if not test_examples:
                continue
                
            test_example = test_examples[0]
            
            task = ARCTask(
                task_id=f"arc_{i}",
                train_examples=train_examples,
                test_input=test_example['input'],
                test_output=test_example.get('output')
            )
            
            context = FewShotLearningContext(task=task)
            seal_data.append((context, task))
            
            if self.max_samples and len(seal_data) >= self.max_samples:
                break
                
        return seal_data
    
    def _generate_synthetic_arc_data(self) -> List[Tuple[FewShotLearningContext, ARCTask]]:
        """Generate synthetic ARC-like data for testing"""
        self.logger.info("Generating synthetic ARC data")
        
        seal_data = []
        num_tasks = self.max_samples or 10
        
        for i in range(num_tasks):
            # Generate simple pattern tasks
            task = self._create_synthetic_arc_task(f"synthetic_{i}")
            context = FewShotLearningContext(task=task)
            seal_data.append((context, task))
            
        return seal_data
    
    def _create_synthetic_arc_task(self, task_id: str) -> ARCTask:
        """Create a synthetic ARC task with simple patterns"""
        # Simple pattern: copy input to output
        train_examples = []
        
        for _ in range(3):  # 3 training examples
            size = random.randint(2, 4)
            input_grid = [[random.randint(0, 5) for _ in range(size)] for _ in range(size)]
            output_grid = [row[:] for row in input_grid]  # Copy input
            
            train_examples.append(ARCExample(
                input_grid=input_grid,
                output_grid=output_grid
            ))
        
        # Test example
        test_size = random.randint(2, 4)
        test_input = [[random.randint(0, 5) for _ in range(test_size)] for _ in range(test_size)]
        test_output = [row[:] for row in test_input]  # Copy input
        
        return ARCTask(
            task_id=task_id,
            train_examples=train_examples,
            test_input=test_input,
            test_output=test_output
        )


class DataLoader:
    """Main data loader for SEAL framework"""
    
    def __init__(
        self,
        domain: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        self.domain = domain
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
    def load_data(self) -> SEALDataset:
        """Load data for the specified domain"""
        if self.domain == "knowledge_incorporation":
            return self._load_knowledge_incorporation_data()
        elif self.domain == "few_shot_learning":
            return self._load_few_shot_learning_data()
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
    
    def _load_knowledge_incorporation_data(self) -> SEALDataset:
        """Load knowledge incorporation data"""
        loader = KnowledgeIncorporationDataLoader(
            dataset_name=self.config.get("dataset_name", "squad"),
            split=self.config.get("split", "train"),
            max_samples=self.config.get("max_samples"),
            min_questions_per_context=self.config.get("min_questions_per_context", 1),
            logger=self.logger
        )
        
        if self.config.get("data_path"):
            data = loader.load_custom_data(self.config["data_path"])
        else:
            data = loader.load_squad_data()
            
        return SEALDataset(data, "knowledge_incorporation", self.logger)
    
    def _load_few_shot_learning_data(self) -> SEALDataset:
        """Load few-shot learning data"""
        loader = FewShotLearningDataLoader(
            dataset_path=self.config.get("dataset_path"),
            split=self.config.get("split", "train"),
            max_samples=self.config.get("max_samples"),
            min_examples_per_task=self.config.get("min_examples_per_task", 1),
            logger=self.logger
        )
        
        data = loader.load_arc_data()
        return SEALDataset(data, "few_shot_learning", self.logger)
    
    def create_torch_dataloader(
        self,
        dataset: SEALDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> TorchDataLoader:
        """Create PyTorch DataLoader for batched processing"""
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for SEAL data"""
        contexts = [item[0] for item in batch]
        tasks = [item[1] for item in batch]
        
        return {
            "contexts": contexts,
            "tasks": tasks
        }


class DataSplitter:
    """Utility for splitting data into train/validation/test sets"""
    
    @staticmethod
    def split_data(
        data: List[Tuple[Any, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
        """Split data into train/validation/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
            
        random.seed(random_seed)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        n_total = len(shuffled_data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = shuffled_data[:n_train]
        val_data = shuffled_data[n_train:n_train + n_val]
        test_data = shuffled_data[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    @staticmethod
    def create_split_datasets(
        domain: str,
        data: List[Tuple[Any, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[SEALDataset, SEALDataset, SEALDataset]:
        """Create train/validation/test datasets"""
        train_data, val_data, test_data = DataSplitter.split_data(
            data, train_ratio, val_ratio, test_ratio, random_seed
        )
        
        logger = logger or logging.getLogger(__name__)
        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return (
            SEALDataset(train_data, domain, logger),
            SEALDataset(val_data, domain, logger),
            SEALDataset(test_data, domain, logger)
        )