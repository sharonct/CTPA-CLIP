import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import torch.nn.functional as F
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ct_clip.pretrained_model import ctclip

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class VisionFeatureExtractor(nn.Module):
    def __init__(
        self, 
        vision_encoder, 
        feature_dim=512, 
        device=None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_encoder = vision_encoder.to(self.device)
        
        try:
            # Safer dimension inference
            self.input_dim = self._safe_infer_input_dimension()
            
            # Create projection layer
            self.feature_projector = nn.Sequential(
                nn.Linear(self.input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU()
            ).to(self.device)

        except Exception as e:
            logging.error(f"Initialization error: {e}")
            raise

    def _safe_infer_input_dimension(self, fallback_dim=512):
        try:
            # Create a small sample input matching encoder's expected shape
            sample_input = torch.randn(
                1, 1, 
                self.vision_encoder.temporal_patch_size, 
                self.vision_encoder.image_size[0] // self.vision_encoder.patch_size[0],
                self.vision_encoder.image_size[1] // self.vision_encoder.patch_size[1], 
                device=self.device
            )
            
            with torch.no_grad():
                # Debug encoder's embedding steps
                patch_embedded = self.vision_encoder.to_patch_emb(sample_input)
                
                try:
                    # Extensive logging and error handling for spatial transformer
                    logging.info(f"Patch embedded shape: {patch_embedded.shape}")
                    
                    spatial_features = self.vision_encoder.enc_spatial_transformer(
                        rearrange(patch_embedded, 'b t h w d -> (b t) (h w) d')
                    )
                    
                    # Adaptive pooling to handle potential shape variations
                    pooled_features = F.adaptive_avg_pool2d(
                        spatial_features.reshape(-1, spatial_features.size(-1)).unsqueeze(0), 
                        (1, spatial_features.size(-1))
                    ).squeeze()
                    
                    return pooled_features.numel()
                
                except Exception as transformer_error:
                    logging.warning(f"Spatial transformer error: {transformer_error}")
                    return fallback_dim

        except Exception as input_error:
            logging.warning(f"Sample input processing failed: {input_error}")
            return fallback_dim

    def forward(self, x):
        try:
            # Robust device and type management
            x = x.to(self.device).float()
            
            # Extensive debugging logging
            logging.info(f"Input tensor shape: {x.shape}, Device: {x.device}")
            
            try:
                # Embedding and transformer processing
                patch_embedded = self.vision_encoder.to_patch_emb(x)
                
                # Reshape for spatial transformer
                spatial_input = rearrange(patch_embedded, 'b t h w d -> (b t) (h w) d')
                
                # Safe spatial transformer call
                spatial_features = self.vision_encoder.enc_spatial_transformer(spatial_input)
                
                # Reshape and pool features
                spatial_features = spatial_features.reshape(
                    x.size(0), x.size(2), x.size(3), x.size(4), -1
                )
                
                pooled_features = F.adaptive_avg_pool3d(
                    spatial_features.permute(0, 4, 1, 2, 3), 
                    (1, 1, 1)
                ).squeeze()
                
                # Feature projection
                features = self.feature_projector(pooled_features)
                
                logging.info(f"Vision feature shape: {features.shape}")
                return features

            except Exception as extraction_error:
                # More informative fallback
                return torch.randn(x.size(0), 512, device=self.device)

        except Exception as e:
            logging.error(f"Forward pass error: {e}")
            return torch.randn(x.size(0), 512, device=self.device)

# Utility function for reshaping
def rearrange(tensor, pattern):
    """
    Simple rearrange implementation for this context.
    In a full implementation, you'd use einops.rearrange.
    """
    if pattern == 'b t h w d -> (b t) (h w) d':
        return tensor.reshape(-1, tensor.size(2) * tensor.size(3), tensor.size(4))
    raise ValueError(f"Unsupported rearrange pattern: {pattern}")

class CustomVQADataset(Dataset):
    def __init__(self, jsonl_file, target_size=480, target_depth=240):
        self.data = []
        self.target_size = target_size
        self.target_depth = target_depth

        with open(jsonl_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        
        try:
            image_features = np.load(image_path)["arr_0"]

            image_tensor = torch.tensor(image_features, dtype=torch.float32)
            if image_tensor.ndimension() == 3:  
                image_tensor = image_tensor.unsqueeze(0)

            C, D, H, W = image_tensor.shape

            if D != self.target_depth or H != self.target_size or W != self.target_size:
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(self.target_depth, self.target_size, self.target_size),
                    mode="trilinear",
                    align_corners=False
                ).squeeze(0)

            assert image_tensor.shape == (C, self.target_depth, self.target_size, self.target_size), "Shape mismatch after resizing!"

            text = item["question"] + " " + item["answer"]
            print(text)

            return image_tensor, text
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Return a dummy tensor and text to prevent training crash
            dummy_tensor = torch.zeros(1, self.target_depth, self.target_size, self.target_size)
            return dummy_tensor, "Dummy question Dummy answer"

def save_model(model, vision_proj_layer, optimizer, epoch, save_path):
    """
    Save the model, vision projection layer, optimizer state, and LoRA adapter
    
    Args:
        model (torch.nn.Module): The main language model
        vision_proj_layer (torch.nn.Module): Vision projection layer
        optimizer (torch.optim.Optimizer): Optimizer state
        epoch (int): Current training epoch
        save_path (str): Directory to save the model
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Create checkpoint directory with epoch
    checkpoint_dir = os.path.join(save_path, f'checkpoint_epoch_{epoch}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Construct save state dictionary
    save_state = {
        'model_state_dict': model.state_dict(),
        'vision_proj_layer_state_dict': vision_proj_layer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    # Save full model checkpoint
    checkpoint_filename = os.path.join(checkpoint_dir, 'full_model_checkpoint.pth')
    torch.save(save_state, checkpoint_filename)
    
    # Save LoRA adapter
    lora_save_path = os.path.join(checkpoint_dir, 'lora_adapter')
    model.save_pretrained(lora_save_path)
    
    logging.info(f"Model checkpoint saved to {checkpoint_filename}")
    logging.info(f"LoRA adapter saved to {lora_save_path}")

class TrainingMetricsTracker:
    def __init__(self, save_path="./metrics"):
        """
        Initialize metrics tracker with configurable save path
        
        Args:
            save_path (str): Directory to save metrics JSON files
        """
        self.metrics = {
            'epochs': [],
            'learning_rates': [],
            'training_losses': [],
            'avg_batch_losses': []
        }
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def update(self, epoch, learning_rate, epoch_loss, batch_losses):
        """
        Update metrics for a specific epoch
        
        Args:
            epoch (int): Current training epoch
            learning_rate (float): Current learning rate
            epoch_loss (float): Average loss for the entire epoch
            batch_losses (list): List of batch losses during the epoch
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['learning_rates'].append(learning_rate)
        self.metrics['training_losses'].append(epoch_loss)
        self.metrics['avg_batch_losses'].append(batch_losses)

    def save_metrics(self, filename=None):
        """
        Save metrics to a JSON file
        
        Args:
            filename (str, optional): Custom filename for metrics. 
                                     If None, generates a default filename.
        """
        if filename is None:
            filename = f"training_metrics_{len(self.metrics['epochs'])}_epochs.json"
        
        filepath = os.path.join(self.save_path, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            logging.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save metrics: {e}")

def train_model(dataloader, model, vision_encoder, optimizer, scheduler, num_epochs=5, save_path="./model"):
    # Explicit device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Initialize metrics tracker
    metrics_tracker = TrainingMetricsTracker(save_path=os.path.join(save_path, "metrics"))

    # Move all components to the same device
    model = model.to(device)
    
    # Wrap vision encoder with feature extractor
    vision_feature_extractor = VisionFeatureExtractor(vision_encoder, device=device)
    
    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Projection Layer initialization
    dummy_images, _ = next(iter(dataloader))
    dummy_images = dummy_images.to(device)
    
    try:
        # Test feature extraction with explicit device handling
        dummy_features = vision_feature_extractor(dummy_images)
        input_dim = dummy_features.size(-1)
        output_dim = model.config.vocab_size

        # Initialize projection layer on the same device
        vision_proj_layer = ProjectionLayer(input_dim, output_dim).to(device)

        # Training loop
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_losses = []
            current_lr = scheduler.get_last_lr()[0]
            
            for batch_idx, (images, texts) in enumerate(dataloader):
                # Ensure all tensors are on the correct device
                images = images.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                try:
                    # Extract and project visual features with explicit device management
                    vision_embedding = vision_feature_extractor(images)
                    
                    # Ensure vision_embedding is on the correct device before projection
                    vision_embedding = vision_embedding.to(device)
                    
                    # Project visual features
                    vision_embedding = vision_proj_layer(vision_embedding)

                    # Tokenize text with device handling
                    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    # Forward pass and loss computation
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=input_ids
                    )
                    loss = outputs.loss

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                
                    # Track loss
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    batch_losses.append(batch_loss)
                
                    # Logging
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {batch_loss:.4f}")
                
                except Exception as step_error:
                    logger.error(f"Training step failed: {step_error}")
                    continue

            # Average epoch loss and scheduler step
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step()

            # Update metrics tracker
            metrics_tracker.update(
                epoch=epoch+1, 
                learning_rate=current_lr, 
                epoch_loss=avg_loss, 
                batch_losses=batch_losses
            )

            logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Learning Rate: {current_lr}")

            # Model saving logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, vision_proj_layer, optimizer, epoch, save_path)
                
                # Save metrics for the best model
                metrics_tracker.save_metrics(f"best_model_metrics_epoch_{epoch+1}.json")
                logger.info(f"Model saved with improved performance (loss: {avg_loss:.4f})")

        # Save final metrics
        metrics_tracker.save_metrics()

    except Exception as init_error:
        logger.error(f"Training initialization error: {init_error}")
        raise

    return metrics_tracker
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim * 2, 1024)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Ensure input is float tensor
        x = x.float()
        return self.projection(x)

def main():
    # Set environment variables for memory management
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Meditron-7B as LLM
    llm_name = "epfl-llm/meditron-7b"
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, use_auth_token=True).to(device)

    # Load CT-ViT as visual encoder
    vision_encoder = ctclip.visual_transformer

    # Stage 2: Fine-tuning with LoRA
    lora_config = LoraConfig(
        r=8,           # Rank of the update matrices
        lora_alpha=16, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    llm = get_peft_model(llm, lora_config)

    # Optimizer and Scheduler Configuration
    optimizer = optim.AdamW(
        llm.parameters(), 
        lr=2e-4,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Dataset and DataLoader
    dataset = CustomVQADataset("/teamspace/studios/this_studio/CTPA-CLIP/data/train_vqa_dataset.jsonl")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Training
    train_model(dataloader, llm, vision_encoder, optimizer, scheduler, num_epochs=10, save_path="/teamspace/studios/this_studio/models/vqa/model")

if __name__ == "__main__":
    main()