"""
Self-Edit Generation Module

Implements the self-edit generation mechanism that creates synthetic data
and optimization parameters for model adaptation.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import logging


class SelfEditGenerator:
    """
    Generates self-edits for model adaptation.
    
    This class coordinates with domain-specific implementations to generate
    appropriate self-edits based on the input context.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        domain: Any,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        max_length: int = 512,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.domain = domain
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length
        self.logger = logger or logging.getLogger(__name__)
        
    def generate(self, context: Any) -> str:
        """
        Generate a self-edit for the given context.
        
        Args:
            context: Domain-specific context (e.g., passage, few-shot examples)
            
        Returns:
            Generated self-edit as a string
        """
        # Create domain-specific prompt
        prompt = self.domain.create_self_edit_prompt(context)
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length // 2
        ).to(self.model.device)
        
        # Generate self-edit
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length // 2,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode generated tokens (exclude input prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        self_edit = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Post-process using domain-specific logic
        self_edit = self.domain.post_process_self_edit(self_edit, context)
        
        self.logger.debug(f"Generated self-edit: {self_edit[:100]}...")
        
        return self_edit
    
    def generate_batch(self, contexts: List[Any]) -> List[str]:
        """
        Generate self-edits for a batch of contexts.
        
        Args:
            contexts: List of domain-specific contexts
            
        Returns:
            List of generated self-edits
        """
        self_edits = []
        
        for context in contexts:
            self_edit = self.generate(context)
            self_edits.append(self_edit)
            
        return self_edits
    
    def set_generation_params(
        self,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_length: Optional[int] = None
    ):
        """Update generation parameters"""
        if temperature is not None:
            self.temperature = temperature
        if top_k is not None:
            self.top_k = top_k
        if top_p is not None:
            self.top_p = top_p
        if max_length is not None:
            self.max_length = max_length


class BaseDomain(ABC):
    """
    Abstract base class for domain-specific implementations.
    
    Each domain (Knowledge Incorporation, Few-Shot Learning) should inherit
    from this class and implement the required methods.
    """
    
    @abstractmethod
    def create_self_edit_prompt(self, context: Any) -> str:
        """
        Create a prompt for self-edit generation based on the context.
        
        Args:
            context: Domain-specific context
            
        Returns:
            Prompt string for the language model
        """
        pass
    
    @abstractmethod
    def post_process_self_edit(self, self_edit: str, context: Any) -> str:
        """
        Post-process the generated self-edit.
        
        Args:
            self_edit: Raw generated self-edit
            context: Original context
            
        Returns:
            Processed self-edit
        """
        pass
    
    @abstractmethod
    def evaluate(
        self, 
        model: torch.nn.Module, 
        tokenizer: Any, 
        task: Any
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a task.
        
        Args:
            model: Language model to evaluate
            tokenizer: Model tokenizer
            task: Domain-specific task
            
        Returns:
            Performance metrics
        """
        pass
    
    @abstractmethod
    def compute_reward(self, performance: Dict[str, float], task: Any) -> float:
        """
        Compute reward based on performance metrics.
        
        Args:
            performance: Performance metrics from evaluate()
            task: Domain-specific task
            
        Returns:
            Reward value (higher is better)
        """
        pass


class SelfEditProcessor:
    """
    Utility class for processing and analyzing self-edits.
    """
    
    @staticmethod
    def analyze_self_edit(self_edit: str) -> Dict[str, Any]:
        """
        Analyze properties of a self-edit.
        
        Args:
            self_edit: Self-edit string
            
        Returns:
            Analysis results
        """
        analysis = {
            "length": len(self_edit),
            "word_count": len(self_edit.split()),
            "line_count": len(self_edit.split('\n')),
            "contains_questions": '?' in self_edit,
            "contains_statements": '.' in self_edit,
            "contains_lists": any(line.strip().startswith(('-', '*', str(i))) 
                                 for i in range(10) for line in self_edit.split('\n')),
        }
        return analysis
    
    @staticmethod
    def filter_self_edits(
        self_edits: List[str], 
        criteria: Dict[str, Any]
    ) -> List[str]:
        """
        Filter self-edits based on criteria.
        
        Args:
            self_edits: List of self-edits
            criteria: Filtering criteria
            
        Returns:
            Filtered self-edits
        """
        filtered = []
        
        for self_edit in self_edits:
            analysis = SelfEditProcessor.analyze_self_edit(self_edit)
            
            # Apply filtering criteria
            if criteria.get("min_length") and analysis["length"] < criteria["min_length"]:
                continue
            if criteria.get("max_length") and analysis["length"] > criteria["max_length"]:
                continue
            if criteria.get("min_words") and analysis["word_count"] < criteria["min_words"]:
                continue
            if criteria.get("max_words") and analysis["word_count"] > criteria["max_words"]:
                continue
                
            filtered.append(self_edit)
            
        return filtered
    
    @staticmethod
    def deduplicate_self_edits(self_edits: List[str], threshold: float = 0.8) -> List[str]:
        """
        Remove duplicate or very similar self-edits.
        
        Args:
            self_edits: List of self-edits
            threshold: Similarity threshold for deduplication
            
        Returns:
            Deduplicated self-edits
        """
        if not self_edits:
            return []
            
        # Simple deduplication based on string similarity
        unique_edits = [self_edits[0]]
        
        for edit in self_edits[1:]:
            is_duplicate = False
            for unique_edit in unique_edits:
                # Calculate simple similarity (Jaccard similarity of words)
                words1 = set(edit.lower().split())
                words2 = set(unique_edit.lower().split())
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                
                if similarity > threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_edits.append(edit)
                
        return unique_edits