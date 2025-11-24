"""
Training script for SmartNotes OCR model.

This script handles the complete training pipeline including:
- Data loading
- Model initialization
- Training loop with validation
- Checkpoint saving
- Metrics tracking and logging
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Setup imports
from smartnotes.paths import setup_imports

setup_imports()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import contextlib

from config import Config, TrainingConfig
from utils import get_logger, get_device, log_config, log_error, ensure_path_exists
from src.dataloader.ocr_dataloader import SmartNotesOCRDataset, collate_fn
from src.model.ocr_model import CRNN

logger = get_logger(__name__)


class WarmupScheduler:
    """
    Learning rate scheduler with warmup.
    
    Gradually increases LR from 0 to base_lr over warmup steps.
    This improves training stability, especially early in training.
    """
    
    def __init__(self, optimizer, base_lr: float, warmup_epochs: int = 2):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            warmup_epochs: Number of epochs for warmup phase
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self, epoch: int) -> float:
        """
        Update learning rate based on epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Current learning rate
        """
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup: gradually increase from 0 to base_lr
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # After warmup, use base learning rate
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class OCRTrainer:
    """
    Trainer class for OCR model.
    
    Handles the complete training process including:
    - Device management
    - Data loading
    - Training and validation loops
    - Checkpoint management
    - Metrics tracking
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        resume_from: Optional[str] = None
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            config: Training configuration object
            resume_from: Path to checkpoint to resume from
        """
        self.config = config or Config.training
        self.device = get_device(
            use_cuda=self.config.USE_CUDA,
            use_mps=self.config.USE_MPS,
            force_cpu=False
        )
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.warmup_scheduler = None
        self.criterion = None
        self.scaler = None
        self.start_epoch = 0
        
        logger.info(f"Training config: {self.config.__dict__}")
    
    def setup(self, num_classes: int) -> None:
        """
        Setup model, optimizer, and loss function.
        
        Args:
            num_classes: Number of output classes
        """
        logger.info(f"Setting up model with {num_classes} classes...")
        
        # Create model
        self.model = CRNN(num_classes=num_classes).to(self.device)
        logger.info(f"Model initialized on {self.device}")
        
        # Loss function
        self.criterion = nn.CTCLoss(
            blank=num_classes,
            zero_infinity=self.config.CTC_ZERO_INFINITY
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Warmup scheduler (linear warmup for first 2 epochs)
        self.warmup_scheduler = WarmupScheduler(
            self.optimizer,
            base_lr=self.config.LEARNING_RATE,
            warmup_epochs=2
        )
        
        # Main learning rate scheduler (kicks in after warmup)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.LR_SCHEDULER_STEP_SIZE,
            gamma=self.config.LR_SCHEDULER_GAMMA
        )
        
        # Mixed precision training (CUDA only)
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info("Setup complete")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Run one training epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()  # type: ignore
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Training",
            total=len(train_loader),
            leave=True
        )
        
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            
            try:
                # Calculate input and target lengths
                input_lengths = torch.full(
                    (imgs.size(0),),
                    imgs.size(3) // 4,
                    dtype=torch.long,
                    device=self.device
                )
                target_lengths = torch.tensor(
                    [torch.count_nonzero(lbl).item() for lbl in labels],
                    dtype=torch.long,
                    device=self.device
                )
                
                self.optimizer.zero_grad()  # type: ignore
                
                # Forward pass
                if self.scaler and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        preds = self.model(imgs)  # type: ignore
                        loss = self.criterion(  # type: ignore
                            preds.log_softmax(2),
                            labels,
                            input_lengths,
                            target_lengths
                        )
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)  # type: ignore
                    self.scaler.update()
                else:
                    preds = self.model(imgs)  # type: ignore
                    loss = self.criterion(  # type: ignore
                        preds.log_softmax(2),
                        labels,
                        input_lengths,
                        target_lengths
                    )
                    loss.backward()
                    self.optimizer.step()  # type: ignore
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            except Exception as e:
                log_error(logger, f"Error in batch {batch_idx}", e)
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Run validation.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average loss for validation
        """
        self.model.eval()  # type: ignore
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation", leave=False):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                try:
                    input_lengths = torch.full(
                        (imgs.size(0),),
                        imgs.size(3) // 4,
                        dtype=torch.long,
                        device=self.device
                    )
                    target_lengths = torch.tensor(
                        [torch.count_nonzero(lbl).item() for lbl in labels],
                        dtype=torch.long,
                        device=self.device
                    )
                    
                    preds = self.model(imgs)  # type: ignore
                    loss = self.criterion(  # type: ignore
                        preds.log_softmax(2),
                        labels,
                        input_lengths,
                        target_lengths
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    log_error(logger, "Error during validation", e)
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        save_dir = Path(self.config.SAVE_DIR)
        ensure_path_exists(str(save_dir))
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),  # type: ignore
            'optimizer_state_dict': self.optimizer.state_dict(),  # type: ignore
            'scheduler_state_dict': self.scheduler.state_dict(),  # type: ignore
        }
        
        # Regular checkpoint
        checkpoint_path = save_dir / f"ocr_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Best checkpoint
        if is_best:
            best_path = save_dir / "ocr_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> None:
        """
        Run complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.start_epoch, num_epochs):
            logger.info(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            
            # Apply warmup for first 2 epochs
            if epoch < 2 and self.warmup_scheduler is not None:
                lr = self.warmup_scheduler.step(epoch)
                logger.info(f"Warmup LR: {lr:.6f}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            val_loss = self.validate(val_loader)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Learning rate step (only after warmup)
            if epoch >= 2 and self.scheduler is not None:
                self.scheduler.step()  # type: ignore
            
            current_lr = self.optimizer.param_groups[0]['lr']  # type: ignore
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_FREQUENCY == 0:
                self.save_checkpoint(epoch + 1)
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, is_best=True)
        
        logger.info("\nTraining complete!")


def main():
    """Main training function."""
    try:
        # Print configuration
        Config.print_config()
        
        # Setup directories
        ensure_path_exists(str(Config.training.SAVE_DIR))
        
        # Load datasets
        logger.info("Loading datasets...")
        train_dataset = SmartNotesOCRDataset(
            root_dir=Config.dataset.ROOT_DIR,
            mode='train'
        )
        val_dataset = SmartNotesOCRDataset(
            root_dir=Config.dataset.ROOT_DIR,
            mode='val'
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.training.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=Config.dataset.NUM_WORKERS
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.training.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=Config.dataset.NUM_WORKERS
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        
        # Initialize trainer
        num_classes = len(train_dataset.tokenizer.chars)
        trainer = OCRTrainer()
        trainer.setup(num_classes)
        
        # Train
        trainer.train(train_loader, val_loader)
        
        logger.info("Training completed successfully!")
    
    except Exception as e:
        log_error(logger, "Fatal error during training", e)
        raise


if __name__ == "__main__":
    main()
