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


# def train_report_generator(model, dataset, optimizer, scheduler, num_epochs=5, 
#                           batch_size=2, save_path="./models/ct_report",
#                           eval_dataset=None, eval_frequency=1, vision_encoder=None,
#                           evaluator=None):

def train_report_generator(model, train_dataset, optimizer, scheduler, 
                          test_dataset=None, eval_frequency=2, num_epochs=5, 
                          batch_size=2, save_path="./models/ct_report",
                          vision_encoder=None, evaluator=None):
    """
    Train the CT report generator with metrics tracking and periodic evaluation
    
    Args:
        model: CT report generator model
        dataset: Dataset for training
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        save_path: Path to save the model
        eval_dataset: Dataset for evaluation (optional)
        eval_frequency: Evaluate every N epochs
        vision_encoder: Vision encoder for evaluation
        evaluator: NLGMetricsEvaluator instance (optional)
        
    Returns:
        tuple: (best_model_path, metrics_tracker)
    """
    # Create dataloaders
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create evaluation dataloader if evaluation dataset is provided
    # if eval_dataset and evaluator is None:
    #     from evaluation_module import NLGMetricsEvaluator
    #     eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    #     evaluator = NLGMetricsEvaluator(model, vision_encoder, eval_dataloader, device=device)

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
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move data to device
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Prepare labels for causal language modeling
            # Labels are the input_ids shifted right
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore the last token
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass
                logits = model(images, input_ids, attention_mask)
                
                # Compute loss
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                
                # Get current learning rate
                current_lr = scheduler.get_last_lr()[0]
                
                # Update learning rate scheduler (if it's not epoch-based)
                if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step()
                
                # Track metrics
                metrics_tracker.update_batch_loss(global_batch_idx, loss.item(), current_lr)
                global_batch_idx += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, "
                               f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                
                # Accumulate epoch loss
                epoch_loss += loss.item()
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
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "loss": avg_epoch_loss
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # If this is the best model based on training loss, create a best_model.pt link
        if is_best_loss:
            best_model_path = os.path.join(save_path, "best_model_by_loss.pt")
            try:
                # Copy the file (more compatible but uses more disk space)
                import shutil
                shutil.copy2(checkpoint_path, best_model_path)
                logger.info(f"New best model (by loss) saved with loss: {avg_epoch_loss:.4f}")
            except Exception as e:
                logger.error(f"Error copying best model: {e}")
        
        # Evaluation step (if eval_dataset is provided)
        if test_dataset and evaluator and (epoch + 1) % eval_frequency == 0:
            logger.info(f"Running evaluation after epoch {epoch+1}...")
            
            # Switch to evaluation mode
            model.eval()
            
            # Run evaluation
            eval_results = evaluator.evaluate(max_samples=50)  # Limit for faster evaluation
            
            # Log evaluation metrics
            logger.info(f"Evaluation results after epoch {epoch+1}:")
            for metric_name, metric_value in eval_results['metrics'].items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Calculate an average score (using ROUGE-L and BERTScore F1 if available)
            if 'bert_f1' in eval_results['metrics']:
                val_score = (eval_results['metrics']['rougeL_score'] + 
                           eval_results['metrics']['bert_f1']) / 2
            else:
                val_score = eval_results['metrics']['rougeL_score']
            
            # Check if this is the best model by validation score
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_model_path = os.path.join(save_path, "best_model_by_validation.pt")
                
                try:
                    import shutil
                    shutil.copy2(checkpoint_path, best_val_model_path)
                    logger.info(f"New best model (by validation) saved with score: {val_score:.4f}")
                    
                    # Update the best overall model path if this is better than the previous best
                    best_model_path = best_val_model_path
                except Exception as e:
                    logger.error(f"Error copying best validation model: {e}")
            
            # Save evaluation results
            eval_save_path = os.path.join(save_path, "metrics", f"eval_epoch_{epoch+1}.json")
            os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)
            with open(eval_save_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            # Switch back to training mode
            model.train()
        
        # Save and visualize metrics after each epoch
        metrics_tracker.save_metrics()
        metrics_tracker.visualize_metrics()
    
    # Final metrics save and visualization
    metrics_tracker.save_metrics()
    metrics_tracker.visualize_metrics()
    
    logger.info(f"Training completed. Best model saved at: {best_model_path}")
    
    return best_model_path, metrics_tracker


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, is_best=False):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        loss: Current loss
        save_path: Base path to save the model
        is_best: Whether this is the best model so far
    
    Returns:
        str: Path to the saved checkpoint
    """
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss
    }, checkpoint_path)
    
    # If this is the best model, create a copy or symlink
    if is_best:
        best_path = os.path.join(save_path, "best_model.pt")
        try:
            # Copy the file (more compatible but uses more disk space)
            import shutil
            shutil.copy2(checkpoint_path, best_path)
        except Exception as e:
            logger.error(f"Error copying best model: {e}")
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def setup_training(train_data, val_data, save_path, batch_size=2, num_epochs=10, 
                   lr=2e-5, cross_attention_lr=1e-4, lora_r=16, lora_alpha=32):
    """
    Set up all components for training
    
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
    
    # Define global dtype to ensure consistency
    dtype = torch.bfloat16
    logger.info(f"Using {dtype} precision for model components")
    
    # Load Meditron-7B as LLM
    llm_name = "epfl-llm/meditron-7b"
    logger.info(f"Loading {llm_name}...")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, 
        torch_dtype=dtype, 
        use_auth_token=True
    ).to(device)
    
    # Import CT-CLIP
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ct_clip.pretrained_model import ctclip
        vision_encoder = ctclip.visual_transformer
        logger.info("Loaded CT-CLIP vision encoder")
    except ImportError:
        logger.error("CT-CLIP not found in path. Make sure it's properly imported.")
        return None, None
    
    # Create robust vision feature extractor with matching dtype
    vision_feature_extractor = RobustVisionFeatureExtractor(ctclip, device=device,dtype=dtype)
    
    # Convert vision feature extractor projection to bfloat16
    vision_feature_extractor.projection = vision_feature_extractor.projection.to(dtype=dtype)
    
    # Apply LoRA to the language model
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    llm = get_peft_model(llm, lora_config)
    logger.info("Applied LoRA configuration to LLM")
    
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
        weight_decay=0.01
    )
    
    # Calculate total steps for the scheduler
    total_steps = len(train_data) * num_epochs // batch_size
    
    # Create scheduler (OneCycleLR with warmup and cosine decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr, cross_attention_lr],
        total_steps=total_steps,
        pct_start=0.1,  # 10% of training for warmup
        anneal_strategy='cos',
        div_factor=25.0,  # Initial LR will be max_lr/25
        final_div_factor=10000.0  # Final LR will be max_lr/10000
    )
    
    # Create evaluator
    from evaluation_module import NLGMetricsEvaluator
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    evaluator = NLGMetricsEvaluator(model, vision_encoder, val_dataloader, device=device)
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # Make sure the train_report_generator function parameters match what it expects
    best_model_path, metrics_tracker = train_report_generator(
        model=model,
        train_dataset=train_data,  # Changed from "dataset" to "train_dataset"
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_path=save_path,
        test_dataset=val_data,  # Changed from "eval_dataset" to "test_dataset"
        eval_frequency=1,  
        vision_encoder=vision_encoder,
        evaluator=evaluator
    )
    
    return best_model_path, metrics_tracker


# if __name__ == "__main__":
#     # Configure training parameters directly (no argparse)
#     train_data_path = "/path/to/train_dataset.jsonl"
#     val_data_path = "/path/to/val_dataset.jsonl"
#     save_path = "./models/ct_report"
#     batch_size = 2
#     num_epochs = 10
#     learning_rate = 2e-5
#     cross_attention_lr = 1e-4
#     lora_r = 16
#     lora_alpha = 32
    
#     # Run training
#     best_model_path, _ = setup_training(
#         train_data_path=train_data_path,
#         val_data_path=val_data_path,
#         save_path=save_path,
#         batch_size=batch_size,
#         num_epochs=num_epochs,
#         lr=learning_rate,
#         cross_attention_lr=cross_attention_lr,
#         lora_r=lora_r,
#         lora_alpha=lora_alpha
#     )
    
#     # Final message
#     if best_model_path:
#         logger.info(f"CT Report Generation training completed successfully! Best model: {best_model_path}")
#     else:
#         logger.error("Training failed to complete successfully.")