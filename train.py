from detectron2.engine import DefaultTrainer, HookBase
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
import torch
import logging
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict

from data_loader import register_datasets, visualize_dataset_samples
from detectron2_config import setup_cfg

class LossTrackingHook(HookBase):
    """
    Hook to track training losses and learning rate.
    Saves detailed loss components and learning rate history.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Initialize loss history
        self.loss_history = {
            'total_loss': [],
            'loss_cls': [],  # Classification loss
            'loss_box_reg': [],  # Box regression loss
            'loss_rpn_cls': [],  # RPN classification loss
            'loss_rpn_loc': [],  # RPN localization loss
            'learning_rate': [],
            'iteration': [],
            'time_elapsed': [],
        }
        self.start_time = time.time()
        
        # Create output directory for logs
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, 'training_logs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("detectron2.trainer")
        
    def before_train(self):
        """Called before training starts."""
        self.start_time = time.time()
        self.logger.info("Starting training...")
        
    def after_step(self):
        """
        Called after each training step.
        Records losses, learning rate, and time elapsed.
        """
        # Skip if not main process in distributed training
        if not comm.is_main_process():
            return
            
        storage = get_event_storage()
        
        # Get current iteration stats
        iteration = storage.iter
        
        # Get current learning rate
        lr = self.trainer.optimizer.param_groups[0]["lr"]
        
        # Get current losses
        total_loss = float(storage.histories["total_loss"].avg())
        loss_dict = {
            'total_loss': total_loss,
            'loss_cls': float(storage.histories.get("loss_cls", 0).avg()),
            'loss_box_reg': float(storage.histories.get("loss_box_reg", 0).avg()),
            'loss_rpn_cls': float(storage.histories.get("loss_rpn_cls", 0).avg()),
            'loss_rpn_loc': float(storage.histories.get("loss_rpn_loc", 0).avg()),
        }
        
        # Record time elapsed
        time_elapsed = time.time() - self.start_time
        
        # Update history
        self.loss_history['iteration'].append(iteration)
        self.loss_history['time_elapsed'].append(time_elapsed)
        self.loss_history['learning_rate'].append(lr)
        for k, v in loss_dict.items():
            self.loss_history[k].append(v)
        
        # Log progress
        if iteration % 20 == 0:  # Log every 20 iterations
            self.logger.info(
                f"Iteration {iteration}: total_loss: {total_loss:.4f}, "
                f"lr: {lr:.6f}, time: {time_elapsed:.1f}s"
            )
            
        # Save loss history periodically
        if iteration % 100 == 0:  # Save every 100 iterations
            self._save_loss_history()
            
    def after_train(self):
        """Called after training completes."""
        if comm.is_main_process():
            self._save_loss_history()
            self.logger.info("Training completed. Final loss history saved.")
            
    def _save_loss_history(self):
        """Save loss history to JSON file."""
        output_file = os.path.join(
            self.output_dir,
            f'loss_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(output_file, 'w') as f:
            json.dump(self.loss_history, f, indent=2)

class EvaluationHook(HookBase):
    """
    Hook to run evaluation during training.
    Tracks multiple metrics including mAP, precision, recall, and F1-score.
    """
    def __init__(self, cfg, val_dataset_name: str):
        super().__init__()
        self.cfg = cfg
        self.val_dataset_name = val_dataset_name
        self.eval_period = cfg.TEST.EVAL_PERIOD  # Evaluate every N iterations
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, 'eval_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize metric history
        self.metric_history = defaultdict(list)
        self.logger = logging.getLogger("detectron2.trainer")
        
    def _do_evaluation(self):
        """Perform evaluation and return metrics."""
        # Create COCO evaluator
        evaluator = COCOEvaluator(
            self.val_dataset_name,
            self.cfg,
            False,  # Distributed eval set to False
            output_dir=self.output_dir
        )
        
        # Create validation data loader
        val_loader = build_detection_test_loader(
            self.cfg,
            self.val_dataset_name
        )
        
        # Run evaluation
        results = inference_on_dataset(
            self.trainer.model,
            val_loader,
            evaluator
        )
        
        if not results:
            self.logger.warning("Empty evaluation results")
            return None
            
        # Extract metrics from results
        metrics = {
            'iteration': self.trainer.iter,
            'time': time.time(),
            'bbox/AP': results['bbox']['AP'],
            'bbox/AP50': results['bbox']['AP50'],
            'bbox/AP75': results['bbox']['AP75'],
            'bbox/APl': results['bbox']['APl'],
            'bbox/APm': results['bbox']['APm'],
            'bbox/APs': results['bbox']['APs']
        }
        
        # Calculate precision, recall, and F1 for each category
        for category_id, category_results in results['bbox'].items():
            if isinstance(category_id, int):  # Skip summary metrics
                metrics[f'category_{category_id}/precision'] = category_results['precision']
                metrics[f'category_{category_id}/recall'] = category_results['recall']
                metrics[f'category_{category_id}/f1'] = (
                    2 * category_results['precision'] * category_results['recall'] /
                    (category_results['precision'] + category_results['recall'])
                    if category_results['precision'] + category_results['recall'] > 0
                    else 0
                )
        
        return metrics
        
    def after_step(self):
        """Run evaluation periodically during training."""
        next_iter = self.trainer.iter + 1
        if self.eval_period > 0 and next_iter % self.eval_period == 0:
            self.logger.info(f"Starting evaluation at iteration {next_iter}")
            metrics = self._do_evaluation()
            
            if metrics is not None:
                # Update metric history
                for k, v in metrics.items():
                    self.metric_history[k].append(v)
                
                # Log main metrics
                self.logger.info("Evaluation results:")
                self.logger.info(f"  bbox/AP: {metrics['bbox/AP']:.4f}")
                self.logger.info(f"  bbox/AP50: {metrics['bbox/AP50']:.4f}")
                self.logger.info(f"  bbox/AP75: {metrics['bbox/AP75']:.4f}")
                
                # Save metrics
                self._save_metrics()
    
    def _save_metrics(self):
        """Save evaluation metrics to JSON file."""
        output_file = os.path.join(
            self.output_dir,
            f'eval_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(output_file, 'w') as f:
            json.dump(dict(self.metric_history), f, indent=2)

class CustomTrainer(DefaultTrainer):
    """
    Custom trainer with evaluation capabilities.
    """
    def __init__(self, cfg, val_dataset_name: Optional[str] = None):
        """Initialize trainer with validation dataset."""
        super().__init__(cfg)
        self.val_dataset_name = val_dataset_name
        self.checkpointer = DetectionCheckpointer(
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        
    def build_hooks(self):
        """Build training hooks including evaluation."""
        hooks = super().build_hooks()
        hooks.append(LossTrackingHook(self.cfg))
        
        # Add evaluation hook if validation dataset is provided
        if self.val_dataset_name:
            hooks.append(EvaluationHook(self.cfg, self.val_dataset_name))
        
        return hooks
        
    def run_step(self):
        """Implement custom training step with gradient clipping."""
        assert self.model.training, "Model must be in training mode."
        
        # Load next batch of data
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        
        # Forward pass
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        
        # Backward pass
        self.optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping
        if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Record stats
        self._write_metrics(loss_dict, data_time)
        
    def state_dict(self):
        """Get state dict for checkpointing."""
        ret = super().state_dict()
        ret["loss_history"] = self.hooks[-2].loss_history  # Loss tracking hook
        if self.val_dataset_name:
            ret["metric_history"] = self.hooks[-1].metric_history  # Evaluation hook
        return ret
        
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        super().load_state_dict(state_dict)
        if "loss_history" in state_dict:
            self.hooks[-2].loss_history = state_dict["loss_history"]
        if "metric_history" in state_dict and self.val_dataset_name:
            self.hooks[-1].metric_history = state_dict["metric_history"]

def train_model(
    train_json: str,
    train_image_dir: str,
    val_json: Optional[str] = None,
    val_image_dir: Optional[str] = None,
    num_classes: int = 5,
    output_dir: str = "./output",
    resume: bool = False,
    eval_period: int = 1000  # Evaluate every N iterations
):
    """
    Main training function with evaluation support.
    
    Args:
        train_json: Path to training set COCO JSON
        train_image_dir: Path to training images directory
        val_json: Path to validation set COCO JSON (optional)
        val_image_dir: Path to validation images directory (optional)
        num_classes: Number of object classes
        output_dir: Directory to save outputs
        resume: Whether to resume from last checkpoint
        eval_period: Number of iterations between evaluations
    """
    # Set up logging
    logger = setup_logger(output_dir)
    logger.info("Setting up training...")
    
    # Register datasets
    success = register_datasets(
        train_json=train_json,
        train_image_dir=train_image_dir,
        val_json=val_json,
        val_image_dir=val_image_dir
    )
    
    if not success:
        logger.error("Failed to register datasets")
        return
        
    # Create config
    cfg = setup_cfg(
        train_dataset_name="my_dataset_train",
        num_classes=num_classes,
        output_dir=output_dir
    )
    
    # Set evaluation period
    cfg.TEST.EVAL_PERIOD = eval_period
    
    # Create trainer
    trainer = CustomTrainer(
        cfg,
        val_dataset_name="my_dataset_val" if val_json else None
    )
    
    # Resume training if requested
    if resume:
        logger.info("Resuming from last checkpoint...")
        trainer.resume_or_load(resume=True)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()

def main():
    """Example usage of the training script with evaluation."""
    train_model(
        train_json="path/to/train.json",
        train_image_dir="path/to/train/images",
        val_json="path/to/val.json",
        val_image_dir="path/to/val/images",
        num_classes=5,
        output_dir="./output",
        resume=False,
        eval_period=1000  # Evaluate every 1000 iterations
    )

if __name__ == "__main__":
    main() 