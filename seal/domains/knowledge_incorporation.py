"""
Knowledge Incorporation Domain Implementation

Implements the knowledge incorporation instantiation of SEAL where the goal
is to efficiently incorporate information from passages into the model's weights
so it can be recalled without relying on context.

The domain generates "implications" from passages and evaluates on no-context
questions about the passage content.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import re
from dataclasses import dataclass
from ..core.self_edit import BaseDomain


@dataclass
class KnowledgeIncorporationTask:
    """Task structure for knowledge incorporation"""
    passage: str
    title: str
    questions: List[Dict[str, str]]  # List of {question, answer} pairs
    
    
@dataclass 
class KnowledgeIncorporationContext:
    """Context structure for knowledge incorporation"""
    passage: str
    title: str


class KnowledgeIncorporationDomain(BaseDomain):
    """
    Knowledge Incorporation domain implementation.
    
    This domain:
    1. Takes a passage and generates implications from it
    2. Fine-tunes the model on these implications
    3. Evaluates on questions about the passage WITHOUT providing the passage context
    4. Rewards self-edits that improve no-context question answering
    """
    
    def __init__(
        self,
        max_implications: int = 10,
        max_implication_length: int = 200,
        temperature: float = 0.8,
        logger: Optional[logging.Logger] = None
    ):
        self.max_implications = max_implications
        self.max_implication_length = max_implication_length
        self.temperature = temperature
        self.logger = logger or logging.getLogger(__name__)
        
        # Prompts for different self-edit generation strategies
        self.prompt_templates = {
            "implications": """Given the following passage, list several implications derived from the content. These should include inferences, logical consequences, or restatements that capture the key information.

Passage: {passage}

Implications:""",
            
            "questions_answers": """Given the following passage, create question-answer pairs that test understanding of the key information.

Passage: {passage}

Questions and Answers:""",
            
            "rewrite": """Given the following passage, rewrite the information in different ways to help with memorization and understanding.

Passage: {passage}

Rewritten versions:""",
            
            "facts": """Given the following passage, extract the key facts and present them in a clear, structured way.

Passage: {passage}

Key facts:"""
        }
        
        self.current_prompt_type = "implications"
        
    def create_self_edit_prompt(self, context: KnowledgeIncorporationContext) -> str:
        """
        Create a prompt for generating self-edits (implications) from a passage.
        
        Args:
            context: KnowledgeIncorporationContext containing passage and title
            
        Returns:
            Formatted prompt string
        """
        # Add title information if available
        passage_text = context.passage
        if context.title:
            passage_text = f"Title: {context.title}\n\n{passage_text}"
            
        # Use the current prompt template
        prompt = self.prompt_templates[self.current_prompt_type].format(
            passage=passage_text
        )
        
        return prompt
    
    def post_process_self_edit(
        self, 
        self_edit: str, 
        context: KnowledgeIncorporationContext
    ) -> str:
        """
        Post-process the generated self-edit to ensure quality.
        
        Args:
            self_edit: Raw generated self-edit
            context: Original context
            
        Returns:
            Processed self-edit
        """
        # Clean up the self-edit
        self_edit = self_edit.strip()
        
        # Split into individual implications/statements
        lines = [line.strip() for line in self_edit.split('\n') if line.strip()]
        
        # Process each line
        processed_lines = []
        for line in lines:
            # Remove leading numbers, bullets, or dashes
            line = re.sub(r'^[\d\.\-\*\+\s]+', '', line)
            line = line.strip()
            
            # Skip empty lines or very short lines
            if len(line) < 10:
                continue
                
            # Limit length
            if len(line) > self.max_implication_length:
                line = line[:self.max_implication_length].rsplit(' ', 1)[0] + '...'
                
            processed_lines.append(line)
        
        # Limit number of implications
        if len(processed_lines) > self.max_implications:
            processed_lines = processed_lines[:self.max_implications]
            
        # Join back together
        processed_self_edit = '\n'.join(processed_lines)
        
        self.logger.debug(f"Processed self-edit: {len(processed_lines)} implications")
        
        return processed_self_edit
    
    def evaluate(
        self, 
        model: torch.nn.Module, 
        tokenizer: Any, 
        task: KnowledgeIncorporationTask
    ) -> Dict[str, float]:
        """
        Evaluate model performance on knowledge incorporation task.
        
        The model is evaluated on questions about the passage WITHOUT
        providing the passage as context.
        
        Args:
            model: Language model to evaluate
            tokenizer: Model tokenizer
            task: Knowledge incorporation task
            
        Returns:
            Performance metrics
        """
        model.eval()
        
        total_questions = len(task.questions)
        correct_answers = 0
        total_score = 0.0
        answer_scores = []
        
        with torch.no_grad():
            for qa_pair in task.questions:
                question = qa_pair['question']
                correct_answer = qa_pair['answer']
                
                # Generate answer without passage context
                generated_answer = self._generate_answer(
                    model, tokenizer, question
                )
                
                # Score the answer
                score = self._score_answer(generated_answer, correct_answer)
                answer_scores.append(score)
                total_score += score
                
                # Binary correctness (score > 0.5)
                if score > 0.5:
                    correct_answers += 1
                    
        # Compute metrics
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        average_score = total_score / total_questions if total_questions > 0 else 0.0
        
        performance = {
            "accuracy": accuracy,
            "average_score": average_score,
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "answer_scores": answer_scores
        }
        
        self.logger.debug(f"Knowledge evaluation: {accuracy:.3f} accuracy, "
                         f"{average_score:.3f} avg score")
        
        return performance
    
    def compute_reward(
        self, 
        performance: Dict[str, float], 
        task: KnowledgeIncorporationTask
    ) -> float:
        """
        Compute reward based on performance metrics.
        
        Args:
            performance: Performance metrics from evaluate()
            task: Knowledge incorporation task
            
        Returns:
            Reward value (binary: 1.0 for improvement, 0.0 otherwise)
        """
        # Use average score as the main reward signal
        reward = performance["average_score"]
        
        # Binary reward: 1.0 if performance is above threshold, 0.0 otherwise
        threshold = 0.5  # Can be adjusted based on task difficulty
        binary_reward = 1.0 if reward > threshold else 0.0
        
        return binary_reward
    
    def _generate_answer(
        self, 
        model: torch.nn.Module, 
        tokenizer: Any, 
        question: str,
        max_length: int = 100
    ) -> str:
        """Generate an answer to a question using the model"""
        # Format the question as a prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        # Decode generated tokens (exclude input prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up answer
        answer = answer.strip()
        
        # Stop at first sentence if multiple sentences
        sentences = answer.split('.')
        if len(sentences) > 1:
            answer = sentences[0] + '.'
            
        return answer
    
    def _score_answer(self, generated_answer: str, correct_answer: str) -> float:
        """
        Score a generated answer against the correct answer.
        
        Uses multiple scoring methods:
        1. Exact match (case-insensitive)
        2. Token overlap (Jaccard similarity)
        3. Substring match
        
        Args:
            generated_answer: Generated answer
            correct_answer: Correct answer
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Normalize answers
        gen_answer = generated_answer.lower().strip()
        correct = correct_answer.lower().strip()
        
        # Exact match
        if gen_answer == correct:
            return 1.0
            
        # Token overlap (Jaccard similarity)
        gen_tokens = set(gen_answer.split())
        correct_tokens = set(correct.split())
        
        if len(gen_tokens.union(correct_tokens)) == 0:
            jaccard_score = 0.0
        else:
            jaccard_score = len(gen_tokens.intersection(correct_tokens)) / len(gen_tokens.union(correct_tokens))
        
        # Substring match
        substring_score = 0.0
        if correct in gen_answer or gen_answer in correct:
            substring_score = 0.5
            
        # Combine scores
        final_score = max(jaccard_score, substring_score)
        
        return final_score
    
    def set_prompt_type(self, prompt_type: str):
        """Set the type of prompt to use for self-edit generation"""
        if prompt_type in self.prompt_templates:
            self.current_prompt_type = prompt_type
            self.logger.info(f"Switched to prompt type: {prompt_type}")
        else:
            available_types = list(self.prompt_templates.keys())
            raise ValueError(f"Invalid prompt type. Available types: {available_types}")
    
    def evaluate_with_context(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        task: KnowledgeIncorporationTask
    ) -> Dict[str, float]:
        """
        Evaluate model performance WITH passage context (for comparison).
        
        This provides a baseline to compare against no-context performance.
        """
        model.eval()
        
        total_questions = len(task.questions)
        correct_answers = 0
        total_score = 0.0
        
        with torch.no_grad():
            for qa_pair in task.questions:
                question = qa_pair['question']
                correct_answer = qa_pair['answer']
                
                # Generate answer WITH passage context
                prompt = f"Passage: {task.passage}\n\nQuestion: {question}\nAnswer:"
                
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                generated_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Score the answer
                score = self._score_answer(generated_answer, correct_answer)
                total_score += score
                
                if score > 0.5:
                    correct_answers += 1
                    
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        average_score = total_score / total_questions if total_questions > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "average_score": average_score,
            "total_questions": total_questions,
            "correct_answers": correct_answers
        }


class KnowledgeIncorporationDataProcessor:
    """Utility class for processing knowledge incorporation data"""
    
    @staticmethod
    def create_task_from_squad(squad_example: Dict[str, Any]) -> KnowledgeIncorporationTask:
        """
        Create a KnowledgeIncorporationTask from a SQuAD dataset example.
        
        Args:
            squad_example: SQuAD dataset example
            
        Returns:
            KnowledgeIncorporationTask instance
        """
        context = squad_example.get('context', '')
        title = squad_example.get('title', '')
        
        # Extract questions and answers
        questions = []
        if 'questions' in squad_example and 'answers' in squad_example:
            for q, a in zip(squad_example['questions'], squad_example['answers']):
                # Handle different answer formats
                if isinstance(a, dict) and 'text' in a:
                    answer_text = a['text'][0] if isinstance(a['text'], list) else a['text']
                elif isinstance(a, list):
                    answer_text = a[0] if a else ""
                else:
                    answer_text = str(a)
                    
                questions.append({
                    'question': q,
                    'answer': answer_text
                })
                
        return KnowledgeIncorporationTask(
            passage=context,
            title=title,
            questions=questions
        )
    
    @staticmethod
    def create_context_from_task(task: KnowledgeIncorporationTask) -> KnowledgeIncorporationContext:
        """Create context from task"""
        return KnowledgeIncorporationContext(
            passage=task.passage,
            title=task.title
        )