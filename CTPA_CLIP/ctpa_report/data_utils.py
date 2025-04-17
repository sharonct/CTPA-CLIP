import torch
import torch.nn.functional as F
import json
import numpy as np
import logging
import os
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CTReportDataset(Dataset):
    """
    Dataset for CT scan report generation with optimizations
    """
    def __init__(self, jsonl_file, tokenizer, target_size=(112, 224, 224), max_length=512, 
                 cache_dir=None, use_cache=True):
        self.data = []
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Create cache directory if needed and doesn't exist
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load data from JSONL file
        with open(jsonl_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} samples from {jsonl_file}")
    
    def __len__(self):
        return len(self.data)

    def get_cache_path(self, image_path, size_str):
        """Generate a unique cache file path based on the image path and target size"""
        if not self.use_cache or not self.cache_dir:
            return None
            
        # Create a hash of the image path and size for the cache filename
        hash_key = hashlib.md5(f"{image_path}_{size_str}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.pt")
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Load CT scan
            image_path = item["image_path"]
            
            # Create a string representation of the target size for caching
            size_str = f"{self.target_size[0]}_{self.target_size[1]}_{self.target_size[2]}"
            cache_path = self.get_cache_path(image_path, size_str)
            
            # Try to load from cache first
            if cache_path and os.path.exists(cache_path):
                try:
                    image_tensor = torch.load(cache_path, map_location='cpu')
                    logger.debug(f"Loaded cached tensor from {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}. Regenerating.")
                    image_tensor = None
            else:
                image_tensor = None
                
            # If not in cache, load and process
            if image_tensor is None:
                # Load raw data
                image_features = np.load(image_path)["arr_0"]
                
                # Convert to tensor
                image_tensor = torch.tensor(image_features, dtype=torch.float32)
                
                # Ensure correct dimensions [C, D, H, W]
                if image_tensor.ndim == 3:  # [D, H, W]
                    image_tensor = image_tensor.unsqueeze(0)  # Add channel dim
                
                # Resize if needed - use nearest neighbor for speed
                C, D, H, W = image_tensor.shape
                target_D, target_H, target_W = self.target_size
                
                if D != target_D or H != target_H or W != target_W:
                    image_tensor = F.interpolate(
                        image_tensor.unsqueeze(0),
                        size=self.target_size,
                        mode="trilinear",  # Use trilinear interpolation
                        align_corners=False
                    ).squeeze(0)
                
                # Normalize to [0, 1] range for better performance
                if image_tensor.max() > 1.0:
                    image_tensor = image_tensor / image_tensor.max()
                
                # Save to cache if enabled
                if cache_path:
                    try:
                        torch.save(image_tensor, cache_path)
                        logger.debug(f"Cached tensor to {cache_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save to cache: {e}")
            
            # Get prompt and report
            report = item["report"]
            
            # Create a prompt for report generation
            prompt = "Generate a detailed clinical report for this CT scan:"
            
            # Tokenize the full sequence (prompt + report)
            full_text = f"{prompt} {report}"
            encoding = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Extract the relevant tensors
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            
            return {
                "image": image_tensor,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "report": report,
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return dummy data to prevent training crashes
            dummy_tensor = torch.zeros(1, self.target_size[0], self.target_size[1], self.target_size[2])
            dummy_text = "Dummy report for error handling"
            dummy_encoding = self.tokenizer(
                f"Generate a detailed clinical report for this CT scan: {dummy_text}",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "image": dummy_tensor,
                "input_ids": dummy_encoding["input_ids"].squeeze(0),
                "attention_mask": dummy_encoding["attention_mask"].squeeze(0),
                "report": dummy_text,
                "prompt": "Generate a detailed clinical report for this CT scan:"
            }


def create_data_loaders(train_jsonl, val_jsonl, tokenizer, batch_size=4, target_size=(112, 224, 224), 
                        num_workers=4, pin_memory=True, prefetch_factor=2, cache_dir=None):
    """
    Create optimized data loaders for training and validation
    
    Args:
        train_jsonl: Path to training data JSONL file
        val_jsonl: Path to validation data JSONL file
        tokenizer: Tokenizer for text processing
        batch_size: Batch size for training
        target_size: Target size for CT scans (D, H, W)
        num_workers: Number of worker threads for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch
        cache_dir: Directory to cache processed tensors
        
    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    # Create datasets
    train_dataset = CTReportDataset(
        jsonl_file=train_jsonl,
        tokenizer=tokenizer,
        target_size=target_size,
        cache_dir=cache_dir
    )
    
    val_dataset = CTReportDataset(
        jsonl_file=val_jsonl,
        tokenizer=tokenizer,
        target_size=target_size,
        cache_dir=cache_dir
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # Drop last incomplete batch for better performance
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch size 1 for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info(f"Created data loaders with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    logger.info(f"Using {num_workers} workers, batch size {batch_size}, prefetch factor {prefetch_factor}")
    
    return train_loader, val_loader, train_dataset, val_dataset


def load_and_process_dataset(jsonl_file, tokenizer, target_size=(112, 224, 224), cache_dir=None):
    """
    Load and process a dataset from a JSONL file with optimizations
    
    Args:
        jsonl_file: Path to JSONL file
        tokenizer: Tokenizer for text processing
        target_size: Target size for CT scans
        cache_dir: Directory to cache processed tensors
        
    Returns:
        CTReportDataset: Processed dataset
    """
    try:
        # Create dataset with caching
        dataset = CTReportDataset(
            jsonl_file=jsonl_file,
            tokenizer=tokenizer,
            target_size=target_size,
            cache_dir=cache_dir,
            use_cache=cache_dir is not None
        )
        
        logger.info(f"Loaded and processed dataset with {len(dataset)} samples from {jsonl_file}")
        logger.info(f"Using {'cached' if cache_dir else 'non-cached'} preprocessing")
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset from {jsonl_file}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    
class TrainingMetricsTracker:
    """
    Track and visualize training metrics
    """
    def __init__(self, save_dir="./metrics"):
        import os
        self.save_dir = save_dir
        self.metrics = {
            'per_batch_loss': [],
            'epoch_avg_loss': [],
            'learning_rates': [],
            'best_loss': float('inf')
        }
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize visualization
        self.can_visualize = True
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available. Visualization will be disabled.")
            self.can_visualize = False
    
    def update_batch_loss(self, batch_idx, loss, lr=None):
        """Update metrics with batch loss and optional learning rate"""
        self.metrics['per_batch_loss'].append((batch_idx, loss))
        if lr is not None:
            self.metrics['learning_rates'].append((batch_idx, lr))
    
    def update_epoch_loss(self, epoch, avg_loss):
        """Update metrics with epoch average loss"""
        self.metrics['epoch_avg_loss'].append((epoch, avg_loss))
        
        # Track best loss
        if avg_loss < self.metrics['best_loss']:
            self.metrics['best_loss'] = avg_loss
            return True  # Indicate this is a new best model
        return False
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        import os
        import json
        metrics_file = os.path.join(self.save_dir, "training_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def visualize_metrics(self):
        """Generate and save visualizations of tracked metrics"""
        if not self.can_visualize:
            logger.warning("Visualization requested but matplotlib setup failed.")
            return
        
        import matplotlib.pyplot as plt
        import os
        
        # Extract data for plotting
        batches, batch_losses = zip(*self.metrics['per_batch_loss']) if self.metrics['per_batch_loss'] else ([], [])
        epochs, epoch_losses = zip(*self.metrics['epoch_avg_loss']) if self.metrics['epoch_avg_loss'] else ([], [])
        lr_batches, learning_rates = zip(*self.metrics['learning_rates']) if self.metrics['learning_rates'] else ([], [])
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot batch losses
        axes[0].plot(batches, batch_losses)
        axes[0].set_title('Training Loss per Batch')
        axes[0].set_xlabel('Batch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        # Plot epoch average losses
        axes[1].plot(epochs, epoch_losses, 'o-')
        axes[1].set_title('Average Loss per Epoch')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Average Loss')
        axes[1].grid(True)
        
        # Plot learning rate
        if learning_rates:
            axes[2].plot(lr_batches, learning_rates)
            axes[2].set_title('Learning Rate Schedule')
            axes[2].set_xlabel('Batch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].grid(True)
        else:
            axes[2].set_visible(False)
        
        # Adjust layout and save figure
        plt.tight_layout()
        fig_path = os.path.join(self.save_dir, "training_metrics.png")
        plt.savefig(fig_path)
        logger.info(f"Metrics visualization saved to {fig_path}")
        plt.close(fig)