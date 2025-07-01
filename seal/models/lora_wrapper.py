"""
LoRA Wrapper for Efficient Parameter Updates

Implements Low-Rank Adaptation (LoRA) for efficient fine-tuning in SEAL.
Since SEAL requires frequent parameter updates with small amounts of data,
LoRA provides a lightweight alternative to full fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM
import copy


class LoRAWrapper:
    """
    Wrapper class for applying LoRA to language models in SEAL.
    
    This class handles the creation and application of LoRA adapters
    for self-edit based fine-tuning.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.base_model = base_model
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.logger = logger or logging.getLogger(__name__)
        
        # Default target modules for common model architectures
        if target_modules is None:
            self.target_modules = self._get_default_target_modules()
        else:
            self.target_modules = target_modules
            
        # Create LoRA configuration
        self.lora_config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Initialize PEFT model
        self._init_peft_model()
        
        self.logger.info(f"LoRA initialized with r={r}, alpha={alpha}, dropout={dropout}")
        self.logger.info(f"Target modules: {self.target_modules}")
        
    def _get_default_target_modules(self) -> List[str]:
        """Get default target modules based on model architecture"""
        # Common target modules for different architectures
        model_name = getattr(self.base_model.config, 'model_type', '').lower()
        
        if 'gpt' in model_name or 'dialogpt' in model_name:
            return ["c_attn", "c_proj", "c_fc"]
        elif 'llama' in model_name:
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif 'bert' in model_name:
            return ["query", "value", "key", "dense"]
        else:
            # Generic fallback - target attention and feedforward layers
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    def _init_peft_model(self):
        """Initialize PEFT model with LoRA configuration"""
        try:
            self.peft_model = get_peft_model(self.base_model, self.lora_config)
            self.logger.info("PEFT model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize PEFT model: {e}")
            raise
    
    def apply_self_edit(
        self, 
        self_edit: str, 
        context: Any, 
        tokenizer: Any = None,
        max_length: int = 512,
        learning_rate: float = 1e-4,
        num_steps: int = 10
    ) -> nn.Module:
        """
        Apply a self-edit to create an adapted model using LoRA.
        
        Args:
            self_edit: Generated self-edit content
            context: Original context
            tokenizer: Model tokenizer
            max_length: Maximum sequence length
            learning_rate: Learning rate for adaptation
            num_steps: Number of gradient steps
            
        Returns:
            Adapted model with LoRA parameters updated
        """
        # Create a copy of the PEFT model for this adaptation
        adapted_model = copy.deepcopy(self.peft_model)
        
        if tokenizer is None:
            self.logger.warning("No tokenizer provided, skipping adaptation")
            return adapted_model
            
        # Prepare training data from self-edit
        training_text = f"Context: {context}\nSelf-Edit: {self_edit}"
        
        # Tokenize
        inputs = tokenizer(
            training_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(adapted_model.device)
        
        # Setup optimizer for LoRA parameters only
        optimizer = torch.optim.AdamW(adapted_model.parameters(), lr=learning_rate)
        
        # Fine-tune on self-edit
        adapted_model.train()
        for step in range(num_steps):
            optimizer.zero_grad()
            
            outputs = adapted_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"]
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                self.logger.debug(f"Adaptation step {step}, loss: {loss.item():.4f}")
        
        adapted_model.eval()
        return adapted_model
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get information about trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for param in self.peft_model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": 100 * trainable_params / all_params
        }
    
    def save_adapter(self, path: str):
        """Save LoRA adapter weights"""
        self.peft_model.save_pretrained(path)
        self.logger.info(f"LoRA adapter saved to {path}")
    
    def load_adapter(self, path: str):
        """Load LoRA adapter weights"""
        # Load adapter into the base model
        self.peft_model = PeftModel.from_pretrained(self.base_model, path)
        self.logger.info(f"LoRA adapter loaded from {path}")
    
    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights into base model and return unloaded model"""
        merged_model = self.peft_model.merge_and_unload()
        self.logger.info("LoRA weights merged into base model")
        return merged_model
    
    def disable_adapters(self):
        """Disable LoRA adapters (use base model only)"""
        self.peft_model.disable_adapters()
        self.logger.info("LoRA adapters disabled")
    
    def enable_adapters(self):
        """Enable LoRA adapters"""
        self.peft_model.enable_adapters()
        self.logger.info("LoRA adapters enabled")
    
    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict containing only adapter parameters"""
        adapter_state_dict = {}
        for name, param in self.peft_model.named_parameters():
            if 'lora_' in name:
                adapter_state_dict[name] = param.clone()
        return adapter_state_dict
    
    def set_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load adapter parameters from state dict"""
        # Filter for LoRA parameters only
        lora_state_dict = {k: v for k, v in state_dict.items() if 'lora_' in k}
        
        # Load into model
        missing_keys, unexpected_keys = self.peft_model.load_state_dict(
            lora_state_dict, strict=False
        )
        
        if missing_keys:
            self.logger.warning(f"Missing keys when loading adapter: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys when loading adapter: {unexpected_keys}")
    
    def reset_adapters(self):
        """Reset all LoRA adapters to their initial state"""
        for module in self.peft_model.modules():
            if hasattr(module, 'reset_lora_parameters'):
                module.reset_lora_parameters()
        self.logger.info("LoRA adapters reset to initial state")


class LoRAManager:
    """
    Manager class for handling multiple LoRA adapters in SEAL.
    
    This class can manage multiple LoRA configurations and switch between them
    as needed during training.
    """
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.adapters = {}
        self.current_adapter = None
        self.logger = logging.getLogger(__name__)
    
    def create_adapter(
        self,
        name: str,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ) -> LoRAWrapper:
        """Create a new LoRA adapter"""
        adapter = LoRAWrapper(
            self.base_model,
            r=r,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules
        )
        
        self.adapters[name] = adapter
        self.logger.info(f"Created adapter '{name}' with r={r}, alpha={alpha}")
        
        return adapter
    
    def switch_adapter(self, name: str):
        """Switch to a different adapter"""
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not found")
            
        self.current_adapter = name
        self.logger.info(f"Switched to adapter '{name}'")
    
    def get_current_adapter(self) -> Optional[LoRAWrapper]:
        """Get the current active adapter"""
        if self.current_adapter is None:
            return None
        return self.adapters[self.current_adapter]
    
    def list_adapters(self) -> List[str]:
        """List all available adapters"""
        return list(self.adapters.keys())
    
    def remove_adapter(self, name: str):
        """Remove an adapter"""
        if name in self.adapters:
            del self.adapters[name]
            if self.current_adapter == name:
                self.current_adapter = None
            self.logger.info(f"Removed adapter '{name}'")
    
    def get_adapter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all adapters"""
        info = {}
        for name, adapter in self.adapters.items():
            info[name] = {
                "r": adapter.r,
                "alpha": adapter.alpha,
                "dropout": adapter.dropout,
                "target_modules": adapter.target_modules,
                "trainable_params": adapter.get_trainable_parameters()
            }
        return info