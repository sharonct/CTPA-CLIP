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
from transformers import BertTokenizer, BertModel
from peft import LoraConfig, get_peft_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ct_clip.pretrained_model import tokenizer, text_encoder, ctclip

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class SimpleVisionFeatureExtractor(nn.Module):
    """
    A simplified feature extractor that bypasses the problematic transformer
    components and extracts features directly from patch embeddings.
    """
    def __init__(self, ctclip_model, feature_dim=512, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctclip = ctclip_model.to(self.device)
        self.vision_encoder = self.ctclip.visual_transformer
        
        # Create a projection network from patched embeddings to features
        self.projection = nn.Sequential(
            nn.Linear(512, feature_dim),  # CT-CLIP patch embedding is 512-dim
            nn.LayerNorm(feature_dim),
            nn.GELU()
        ).to(self.device)
        
        logger.info(f"Initialized SimpleVisionFeatureExtractor with output dim={feature_dim}")
    
    def forward(self, x):
        try:
            # Ensure proper device and type
            x = x.to(self.device).float()
            
            with torch.no_grad():
                # Apply patch embedding (this part still works)
                patch_embedded = self.vision_encoder.to_patch_emb(x)
                
                # Log the shape for debugging
                b, t, h, w, c = patch_embedded.shape
                
                # Simple pooling approach (average across spatial and temporal dimensions)
                # First average across spatial dimensions (h, w)
                spatial_pooled = patch_embedded.mean(dim=(2, 3))  # -> [b, t, c]
                
                # Then average across temporal dimension (t)
                temporal_pooled = spatial_pooled.mean(dim=1)  # -> [b, c]
                
                # Project to feature dimension
                features = self.projection(temporal_pooled)
                
                return features
            
        except Exception as e:
            logger.error(f"Error in SimpleVisionFeatureExtractor: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return placeholder features in case of error
            return torch.randn(x.size(0), 512, device=self.device)

class MedicalVQAModel(nn.Module):
    def __init__(self, text_encoder, vision_feature_dim=512, text_feature_dim=768, vocab_size=30522):
        super().__init__()
        
        # Text encoder (BiomedVLP-CXR-BERT)
        self.text_encoder = text_encoder
        
        # Multi-head cross-attention between vision and text
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Vision projection to match text dimension
        self.vision_projection = nn.Linear(vision_feature_dim, text_feature_dim)
        
        # Vision-text fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(vision_feature_dim + text_feature_dim, text_feature_dim),
            nn.LayerNorm(text_feature_dim),
            nn.Dropout(0.1),
            nn.GELU()
        )
        
        # Output generation head
        self.lm_head = nn.Linear(text_feature_dim, vocab_size)
        
    def forward(self, vision_features, input_ids, attention_mask, labels=None):
        # Get text features from BERT
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get full sequence of text features
        text_sequence = text_outputs.last_hidden_state
        
        # Project vision features to match text dimension
        vision_proj = self.vision_projection(vision_features).unsqueeze(1)  # Add sequence dimension
        
        # Apply cross-attention: text features attend to vision features
        attended_features, _ = self.cross_attention(
            query=text_sequence,
            key=vision_proj,
            value=vision_proj
        )
        
        # Get pooled representation (CLS token)
        pooled_features = attended_features[:, 0, :]
        
        # Also concatenate vision features with CLS for direct fusion
        vision_text_combined = torch.cat([vision_features, pooled_features], dim=1)
        fused_features = self.fusion(vision_text_combined)
        
        # Language modeling head for text generation
        # Replicate fused features to match sequence length for text generation
        repeated_features = fused_features.unsqueeze(1).expand(-1, attended_features.size(1), -1)
        
        # Combine with attended features (residual connection)
        final_features = attended_features + repeated_features
        
        # Predict token probabilities 
        logits = self.lm_head(final_features)
        
        # If labels are provided, compute loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # Reshape for loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Calculate loss only on non-padding tokens
            active_loss = shift_labels.ne(self.text_encoder.config.pad_token_id).view(-1)
            active_logits = shift_logits.view(-1, logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]
            loss = loss_fn(active_logits, active_labels)
            
            return logits, loss
        
        return logits

class CustomVQADataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, target_size=480, target_depth=240, max_length=512):
        self.data = []
        self.target_size = target_size
        self.target_depth = target_depth
        self.tokenizer = tokenizer
        self.max_length = max_length

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

            # Prepare question and answer
            question = item["question"]
            answer = item["answer"]
            
            # For text generation, we format the input as a prompt and output
            # Example: "Question: What is visible in this CT scan? Answer: The scan shows..."
            prompt = f"Question: {question} Answer:"
            full_text = f"Question: {question} Answer: {answer}"
            
            # Tokenize the full text for training
            encoding = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

            return {
                "image": image_tensor,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "question": question,
                "answer": answer,
                "prompt": prompt
            }
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Return a dummy tensor and text to prevent training crash
            dummy_tensor = torch.zeros(1, self.target_depth, self.target_size, self.target_size)
            dummy_encoding = self.tokenizer(
                "Question: dummy question Answer: dummy answer",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "image": dummy_tensor,
                "input_ids": dummy_encoding["input_ids"].squeeze(0),
                "attention_mask": dummy_encoding["attention_mask"].squeeze(0),
                "question": "dummy question",
                "answer": "dummy answer",
                "prompt": "Question: dummy question Answer:"
            }

def save_model(model, vision_extractor, optimizer, epoch, save_path):
    """Save the model checkpoints"""
    os.makedirs(save_path, exist_ok=True)
    checkpoint_dir = os.path.join(save_path, f'checkpoint_epoch_{epoch}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_state = {
        'model_state_dict': model.state_dict(),
        'vision_extractor_state_dict': vision_extractor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    checkpoint_filename = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
    torch.save(save_state, checkpoint_filename)
    logger.info(f"Model checkpoint saved to {checkpoint_filename}")

def train_model(dataloader, model, ctclip_model, optimizer, scheduler, num_epochs=5, save_path="./model"):
    # Initialize metrics tracking
    metrics = {
        'epochs': [],
        'training_losses': [],
        'perplexities': []
    }
    
    # Create vision feature extractor
    vision_feature_extractor = SimpleVisionFeatureExtractor(ctclip_model, device=device)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Prepare labels for next token prediction (shift input_ids)
            labels = input_ids.clone()
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Extract visual features
                vision_features = vision_feature_extractor(images)
                
                # Forward pass
                _, loss = model(vision_features, input_ids, attention_mask, labels=labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
                
                # Logging
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1} completed: Avg Loss = {avg_loss:.4f}, Perplexity = {perplexity:.4f}")
        
        # Save metrics
        metrics['epochs'].append(epoch+1)
        metrics['training_losses'].append(avg_loss)
        metrics['perplexities'].append(perplexity)
        
        # Save model if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, vision_feature_extractor, optimizer, epoch+1, save_path)
            logger.info(f"New best model saved with loss: {avg_loss:.4f}")
    
    # Save final metrics
    os.makedirs(os.path.join(save_path, "metrics"), exist_ok=True)
    with open(os.path.join(save_path, "metrics", "training_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def main():
    # Set environment variables for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Initialize the VQA model
    vqa_model = MedicalVQAModel(
        text_encoder=text_encoder,
        vocab_size=tokenizer.vocab_size
    ).to(device)
    
    # Apply LoRA to fine-tune the BERT model efficiently
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value"]  # Target attention modules in BERT
    )
    vqa_model.text_encoder = get_peft_model(vqa_model.text_encoder, lora_config)
    
    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        vqa_model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Dataset and DataLoader
    dataset = CustomVQADataset(
        "/teamspace/studios/this_studio/data/train_vqa_dataset.jsonl",
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training
    train_model(
        dataloader, 
        vqa_model, 
        ctclip,  # Pass the full CTCLIP model
        optimizer, 
        scheduler, 
        num_epochs=10, 
        save_path="/teamspace/studios/this_studio/vqa/model2"
    )

if __name__ == "__main__":
    main()