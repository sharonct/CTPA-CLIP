import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import gc
from tqdm import tqdm

# Try to import CUDA Amp for mixed precision training
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Automatic Mixed Precision not available")

# Import local modules
from model_components import CTReportGenerator, RobustVisionFeatureExtractor, CrossAttentionLayer
from data_utils import CTReportDataset, TrainingMetricsTracker

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def train_report_generator(model, train_dataset, optimizer, scheduler, 
                          test_dataset=None, eval_frequency=1, num_epochs=5, 
                          batch_size=4, save_path="./models/ct_report",
                          vision_encoder=None, evaluator=None, 
                          use_mixed_precision=True, gradient_checkpointing=True,
                          accumulation_steps=1, max_grad_norm=1.0,
                          eval_samples=10):
    """
    Train the CT report generator with optimized performance
    
    Args:
        model: CT report generator model
        train_dataset: Dataset for training
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        test_dataset: Dataset for evaluation (optional)
        eval_frequency: Evaluate every N epochs
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        save_path: Path to save the model
        vision_encoder: Vision encoder for evaluation
        evaluator: NLGMetricsEvaluator instance (optional)
        use_mixed_precision: Whether to use mixed precision training
        gradient_checkpointing: Whether to use gradient checkpointing
        accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        eval_samples: Number of samples to use for evaluation
        
    Returns:
        tuple: (best_model_path, metrics_tracker)
    """
    # Enable gradient checkpointing if requested (saves memory)
    if gradient_checkpointing and hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Set up mixed precision training
    scaler = GradScaler() if use_mixed_precision and AMP_AVAILABLE else None
    if scaler:
        logger.info("Using mixed precision training with automatic scaler")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True  # Better for performance
    )
    
    # Create evaluation dataloader if evaluation dataset is provided
    if test_dataset and evaluator is None:
        from evaluation_module import NLGMetricsEvaluator
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        evaluator = NLGMetricsEvaluator(model, vision_encoder, test_dataloader, device=device)
    
    # Initialize metrics tracker
    metrics_tracker = TrainingMetricsTracker(save_dir=os.path.join(save_path, "metrics"))
    
    # Prepare for training
    model.train()
    
    # Track global batch index and best model
    global_batch_idx = 0
    best_model_path = None
    best_val_score = -float('inf')
    best_loss = float('inf')
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting training with {len(train_dataset)} samples for {num_epochs} epochs")
    logger.info(f"Using batch size {batch_size} with {accumulation_steps} gradient accumulation steps")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        
        # Reset gradients at the start of each epoch
        optimizer.zero_grad()
        
        # Progress bar for better visibility
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Prepare labels for causal language modeling (shift right)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore the last token
            
            try:
                # Mixed precision context for forward and backward pass
                with autocast(enabled=use_mixed_precision and AMP_AVAILABLE):
                    # Forward pass
                    logits = model(images, input_ids, attention_mask)
                    
                    # Compute loss
                    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                # Backward pass with gradient scaling
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update parameters after accumulation steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    # Gradient clipping
                    if scaler:
                        scaler.unscale_(optimizer)
                    
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Update with scaler if using mixed precision
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Zero gradients after optimization step
                    optimizer.zero_grad()
                    
                    # Update learning rate scheduler (if it's not epoch-based)
                    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                        scheduler.step()
                
                # Get current learning rate
                current_lr = scheduler.get_last_lr()[0]
                
                # Track metrics (multiply by accumulation_steps to get the real loss)
                actual_loss = loss.item() * accumulation_steps
                metrics_tracker.update_batch_loss(global_batch_idx, actual_loss, current_lr)
                global_batch_idx += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{actual_loss:.4f}", 
                    'lr': f"{current_lr:.6f}"
                })
                
                # Accumulate epoch loss
                epoch_loss += actual_loss
                batch_count += 1
                
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_epoch_loss:.4f}")
        
        # Update epoch metrics
        is_best_loss = metrics_tracker.update_epoch_loss(epoch, avg_epoch_loss)
        
        # Update learning rate scheduler (if it's epoch-based)
        if isinstance(scheduler, (torch.optim.lr_scheduler.StepLR, 
                                 torch.optim.lr_scheduler.MultiStepLR,
                                 torch.optim.lr_scheduler.ExponentialLR,
                                 torch.optim.lr_scheduler.CosineAnnealingLR)):
            scheduler.step()
        
        # Only save model if loss has improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_loss_path = os.path.join(save_path, "best_model_by_loss.pt")
            
            logger.info(f"New best loss: {avg_epoch_loss:.4f}. Saving model...")
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "loss": avg_epoch_loss
            }, best_loss_path)
            
            logger.info(f"Best model (by loss) saved to {best_loss_path}")
            
            # If no validation data, use the best loss model as the overall best
            if test_dataset is None:
                best_model_path = best_loss_path
        
        # Evaluation step (if test_dataset is provided)
        if test_dataset and evaluator and (epoch + 1) % eval_frequency == 0:
            logger.info(f"Running evaluation after epoch {epoch+1}...")
            
            # Switch to evaluation mode
            model.eval()
            
            # Run evaluation with limited samples
            eval_results = evaluator.evaluate(max_samples=eval_samples)
            
            # Log evaluation metrics
            logger.info(f"Evaluation results after epoch {epoch+1}:")
            for metric_name, metric_value in eval_results['metrics'].items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Calculate an average score (using ROUGE-L and BERTScore F1 if available)
            val_score = 0
            metrics_count = 0
            
            if 'rougeL_score' in eval_results['metrics']:
                val_score += eval_results['metrics']['rougeL_score']
                metrics_count += 1
                
            if 'bert_f1' in eval_results['metrics'] and eval_results['metrics']['bert_f1'] > 0:
                val_score += eval_results['metrics']['bert_f1']
                metrics_count += 1
                
            if metrics_count > 0:
                val_score /= metrics_count
            
            # Check if this is the best model by validation score
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_model_path = os.path.join(save_path, "best_model_by_validation.pt")
                
                logger.info(f"New best validation score: {val_score:.4f}. Saving model...")
                
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch,
                    "val_score": val_score,
                    "loss": avg_epoch_loss
                }, best_val_model_path)
                
                logger.info(f"Best model (by validation) saved to {best_val_model_path}")
                
                # Update the best overall model path
                best_model_path = best_val_model_path
            
            # Save evaluation results
            eval_save_path = os.path.join(save_path, "metrics", f"eval_epoch_{epoch+1}.json")
            os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)
            with open(eval_save_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            # Switch back to training mode
            model.train()
            
            # Re-enable gradient checkpointing if it was used
            if gradient_checkpointing and hasattr(model.llm, "gradient_checkpointing_enable"):
                model.llm.gradient_checkpointing_enable()
        
        # Save and visualize metrics after each epoch
        metrics_tracker.save_metrics()
        metrics_tracker.visualize_metrics()
        
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final metrics save and visualization
    metrics_tracker.save_metrics()
    metrics_tracker.visualize_metrics()
    
    logger.info(f"Training completed. Best model saved at: {best_model_path}")
    
    return best_model_path, metrics_tracker


def setup_training(train_data, val_data, save_path, batch_size=4, num_epochs=5, 
                   lr=5e-5, cross_attention_lr=2e-4, lora_r=8, lora_alpha=16,
                   use_mixed_precision=True, gradient_checkpointing=True,
                   accumulation_steps=2, eval_samples=10, cache_dir=None):
    """
    Set up all components for optimized training
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        save_path: Path to save models and metrics
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        lr: Learning rate for LLM
        cross_attention_lr: Learning rate for cross attention
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        use_mixed_precision: Whether to use mixed precision
        gradient_checkpointing: Whether to use gradient checkpointing
        accumulation_steps: Number of steps to accumulate gradients
        eval_samples: Number of samples to use for evaluation
        cache_dir: Directory to cache processed tensors
        
    Returns:
        tuple: (best_model_path, metrics_tracker)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define global dtype - use bfloat16 or float16 for mixed precision
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using {dtype} precision for model components")
    
    # Load Meditron-7B as LLM, with optimizations
    llm_name = "epfl-llm/meditron-7b"
    logger.info(f"Loading {llm_name}...")
    
    # Load model with optimizations
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, 
        torch_dtype=dtype,
        use_auth_token=True,
        device_map="auto",  # Better memory management
        load_in_8bit=False,  # Keep this off for better quality
    )
    
    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        llm.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled on LLM")
    
    # Import CT-CLIP
    try:
        from ct_clip.pretrained_model import ctclip
        vision_encoder = ctclip.visual_transformer
        logger.info("Loaded CT-CLIP vision encoder")
    except ImportError:
        logger.error("CT-CLIP not found in path. Make sure it's properly imported.")
        return None, None
    
    # Create robust vision feature extractor with matching dtype
    vision_feature_extractor = RobustVisionFeatureExtractor(ctclip, device=device, dtype=dtype)
    
    # Convert vision feature extractor projection to appropriate dtype
    vision_feature_extractor.projection = vision_feature_extractor.projection.to(dtype=dtype)
    
    # Apply LoRA to the language model with reduced parameters
    lora_config = LoraConfig(
        r=lora_r,  # Reduced from original
        lora_alpha=lora_alpha,  # Reduced from original
        lora_dropout=0.05,  # Lower dropout for faster convergence
        target_modules=["q_proj", "v_proj"],  # Fewer target modules (removed k_proj and o_proj)
        bias="none"
    )
    llm = get_peft_model(llm, lora_config)
    logger.info(f"Applied LoRA configuration to LLM: r={lora_r}, alpha={lora_alpha}")
    
    # Create cross-attention layer with matching dtype
    cross_attention = CrossAttentionLayer(
        text_dim=llm.config.hidden_size,
        vision_dim=vision_feature_extractor.feature_dim
    ).to(device).to(dtype=dtype)
    
    # Create report generator model
    model = CTReportGenerator(
        llm=llm,
        vision_feature_extractor=vision_feature_extractor,
        cross_attention=cross_attention
    ).to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer
    
    # Log datasets
    logger.info(f"Using training dataset with {len(train_data)} samples")
    logger.info(f"Using validation dataset with {len(val_data)} samples")
    
    # Create optimizer with different learning rates for different components
    optimizer = optim.AdamW(
        [
            {"params": model.llm.parameters(), "lr": lr},
            {"params": model.cross_attention.parameters(), "lr": cross_attention_lr}
        ],
        weight_decay=0.01,
        eps=1e-8  # Slightly higher epsilon for better stability
    )
    
    # Calculate total steps for the scheduler (adjusted for accumulation)
    total_steps = len(train_data) * num_epochs // (batch_size * accumulation_steps)
    
    # Create scheduler (OneCycleLR with warmup and cosine decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr, cross_attention_lr],
        total_steps=total_steps,
        pct_start=0.05,  # 5% of training for warmup (reduced from 10%)
        anneal_strategy='cos',
        div_factor=10.0,  # Initial LR will be max_lr/10 (less aggressive)
        final_div_factor=1000.0  # Final LR will be max_lr/1000
    )
    
    # Create evaluator with limited samples for faster evaluation
    from evaluation_module import NLGMetricsEvaluator
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    evaluator = NLGMetricsEvaluator(model, vision_encoder, val_dataloader, device=device)
    
    logger.info(f"Starting training for {num_epochs} epochs with batch size {batch_size}...")
    logger.info(f"Using mixed precision: {use_mixed_precision}, gradient checkpointing: {gradient_checkpointing}")
    logger.info(f"Gradient accumulation steps: {accumulation_steps}, evaluation samples: {eval_samples}")
    
    # Train model
    best_model_path, metrics_tracker = train_report_generator(
        model=model,
        train_dataset=train_data,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_path=save_path,
        test_dataset=val_data,
        eval_frequency=1,  
        vision_encoder=vision_encoder,
        evaluator=evaluator,
        use_mixed_precision=use_mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
        accumulation_steps=accumulation_steps,
        eval_samples=eval_samples
    )
    
    return best_model_path, metrics_tracker