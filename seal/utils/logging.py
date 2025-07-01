"""
Logging Setup and Utilities for SEAL

Provides logging configuration and utilities for the SEAL framework,
including integration with Weights & Biases for experiment tracking.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import time
import json


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for SEAL.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional file path to save logs
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in logs
        
    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create custom format
    if format_string is None:
        if include_timestamp:
            format_string = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[]
    )
    
    # Get root logger
    logger = logging.getLogger("seal")
    logger.handlers.clear()  # Clear any existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.setLevel(numeric_level)
    logger.info(f"Logging initialized with level: {level}")
    
    return logger


class WandBLogger:
    """
    Weights & Biases integration for SEAL experiment tracking.
    """
    
    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.wandb_available = False
        self.run = None
        
        # Try to import wandb
        try:
            import wandb
            self.wandb = wandb
            self.wandb_available = True
        except ImportError:
            self.logger.warning("wandb not available. Install with: pip install wandb")
            return
        
        # Initialize wandb run if project specified
        if project:
            self.init_run(project, entity, name, config, tags)
    
    def init_run(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None
    ):
        """Initialize a wandb run"""
        if not self.wandb_available:
            return
            
        try:
            self.run = self.wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                tags=tags,
                reinit=True
            )
            self.logger.info(f"WandB run initialized: {project}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {e}")
            self.wandb_available = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb"""
        if not self.wandb_available or not self.run:
            return
            
        try:
            self.wandb.log(metrics, step=step)
        except Exception as e:
            self.logger.error(f"Failed to log metrics to wandb: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to wandb"""
        if not self.wandb_available or not self.run:
            return
            
        try:
            self.wandb.config.update(config)
        except Exception as e:
            self.logger.error(f"Failed to log config to wandb: {e}")
    
    def log_artifact(self, file_path: str, name: str, type: str = "model"):
        """Log an artifact to wandb"""
        if not self.wandb_available or not self.run:
            return
            
        try:
            artifact = self.wandb.Artifact(name, type=type)
            artifact.add_file(file_path)
            self.wandb.log_artifact(artifact)
            self.logger.info(f"Logged artifact: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log artifact: {e}")
    
    def finish(self):
        """Finish the wandb run"""
        if self.wandb_available and self.run:
            self.wandb.finish()
            self.logger.info("WandB run finished")


class SEALLogger:
    """
    Main logger class for SEAL that combines standard logging with experiment tracking.
    """
    
    def __init__(
        self,
        name: str = "seal",
        level: str = "INFO",
        log_file: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        # Setup standard logging
        self.logger = setup_logging(level, log_file)
        self.logger = logging.getLogger(name)
        
        # Setup wandb if configured
        self.wandb_logger = None
        if wandb_config:
            self.wandb_logger = WandBLogger(
                project=wandb_config.get("project"),
                entity=wandb_config.get("entity"),
                name=wandb_config.get("name"),
                config=wandb_config.get("config"),
                tags=wandb_config.get("tags"),
                logger=self.logger
            )
        
        # Metrics buffer for batched logging
        self.metrics_buffer = {}
        self.buffer_size = 0
        self.max_buffer_size = 10
        
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
        buffer: bool = False
    ):
        """
        Log metrics to both standard logger and wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step/iteration number
            prefix: Prefix to add to metric names
            buffer: Whether to buffer metrics for batched logging
        """
        # Format metrics for standard logging
        metric_strs = []
        for name, value in metrics.items():
            if isinstance(value, float):
                metric_strs.append(f"{prefix}{name}: {value:.4f}")
            else:
                metric_strs.append(f"{prefix}{name}: {value}")
        
        if metric_strs:
            self.logger.info(f"Metrics - {', '.join(metric_strs)}")
        
        # Log to wandb
        if self.wandb_logger:
            wandb_metrics = {}
            for name, value in metrics.items():
                wandb_metrics[f"{prefix}{name}"] = value
                
            if buffer:
                self.metrics_buffer.update(wandb_metrics)
                self.buffer_size += len(wandb_metrics)
                
                if self.buffer_size >= self.max_buffer_size:
                    self.flush_metrics(step)
            else:
                self.wandb_logger.log_metrics(wandb_metrics, step)
    
    def flush_metrics(self, step: Optional[int] = None):
        """Flush buffered metrics to wandb"""
        if self.wandb_logger and self.metrics_buffer:
            self.wandb_logger.log_metrics(self.metrics_buffer, step)
            self.metrics_buffer.clear()
            self.buffer_size = 0
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
            
        if self.wandb_logger:
            self.wandb_logger.log_config(config)
    
    def log_training_start(self, config: Dict[str, Any], dataset_size: int):
        """Log training start information"""
        self.logger.info("=" * 50)
        self.logger.info("SEAL Training Started")
        self.logger.info("=" * 50)
        self.logger.info(f"Dataset size: {dataset_size}")
        self.logger.info(f"Domain: {config.get('domain', 'unknown')}")
        self.logger.info(f"Model: {config.get('model_name', 'unknown')}")
        self.logger.info(f"Max iterations: {config.get('max_outer_iterations', 'unknown')}")
        
        self.log_config(config)
    
    def log_training_end(self, final_metrics: Dict[str, Any], total_time: float):
        """Log training completion information"""
        self.logger.info("=" * 50)
        self.logger.info("SEAL Training Completed")
        self.logger.info("=" * 50)
        self.logger.info(f"Total training time: {total_time:.2f} seconds")
        
        if final_metrics:
            self.logger.info("Final metrics:")
            for name, value in final_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {name}: {value:.4f}")
                else:
                    self.logger.info(f"  {name}: {value}")
    
    def log_iteration(
        self,
        iteration: int,
        metrics: Dict[str, Any],
        iteration_time: float,
        detailed: bool = False
    ):
        """Log iteration information"""
        if detailed or iteration % 10 == 0:
            self.logger.info(f"Iteration {iteration}:")
            self.logger.info(f"  Time: {iteration_time:.2f}s")
            
            for name, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {name}: {value:.4f}")
                elif isinstance(value, (int, str)):
                    self.logger.info(f"  {name}: {value}")
        
        # Always log to wandb
        if self.wandb_logger:
            wandb_metrics = dict(metrics)
            wandb_metrics["iteration_time"] = iteration_time
            self.wandb_logger.log_metrics(wandb_metrics, step=iteration)
    
    def log_checkpoint(self, checkpoint_path: str, iteration: int):
        """Log checkpoint saving"""
        self.logger.info(f"Checkpoint saved: {checkpoint_path} (iteration {iteration})")
        
        if self.wandb_logger:
            self.wandb_logger.log_artifact(checkpoint_path, f"checkpoint_iter_{iteration}", "model")
    
    def log_evaluation(self, eval_metrics: Dict[str, Any], dataset_name: str = "eval"):
        """Log evaluation results"""
        self.logger.info(f"Evaluation on {dataset_name}:")
        for name, value in eval_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {name}: {value:.4f}")
            else:
                self.logger.info(f"  {name}: {value}")
        
        if self.wandb_logger:
            wandb_metrics = {f"eval_{name}": value for name, value in eval_metrics.items()}
            self.wandb_logger.log_metrics(wandb_metrics)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        error_msg = f"Error{' in ' + context if context else ''}: {str(error)}"
        self.logger.error(error_msg)
        
        if self.wandb_logger:
            self.wandb_logger.log_metrics({"error": 1, "error_context": context})
    
    def finish(self):
        """Finish logging (close wandb run)"""
        self.flush_metrics()
        if self.wandb_logger:
            self.wandb_logger.finish()
        
        self.logger.info("Logging finished")


def create_logger(
    name: str = "seal",
    level: str = "INFO", 
    log_file: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None
) -> SEALLogger:
    """
    Create a SEAL logger with standard logging and optional wandb integration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        wandb_project: Optional wandb project name
        wandb_config: Optional wandb configuration
        
    Returns:
        Configured SEAL logger
    """
    wandb_logger_config = None
    if wandb_project:
        wandb_logger_config = {
            "project": wandb_project,
            "config": wandb_config
        }
    
    return SEALLogger(
        name=name,
        level=level,
        log_file=log_file,
        wandb_config=wandb_logger_config
    )