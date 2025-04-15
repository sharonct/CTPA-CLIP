import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobustVisionFeatureExtractor(nn.Module):
    """
    A robust feature extractor that handles various CT scan formats
    and extracts meaningful features while gracefully handling errors.
    """
    def __init__(self, ctclip_model, feature_dim=512, device=None, dtype=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctclip = ctclip_model.to(self.device)
        self.vision_encoder = self.ctclip.visual_transformer
        self.feature_dim = feature_dim
        self.dtype = dtype
        
        # Create a projection network from patched embeddings to features
        self.projection = nn.Sequential(
            nn.Linear(512, feature_dim),  # CT-CLIP patch embedding is 512-dim
            nn.LayerNorm(feature_dim),
            nn.GELU()
        ).to(self.device)
        
        # Convert to specified dtype if provided
        if dtype is not None:
            self.projection = self.projection.to(dtype)
        
        logger.info(f"Initialized RobustVisionFeatureExtractor with output dim={feature_dim}")
    
    def forward(self, x):
        try:
            # Ensure proper device and type
            x = x.to(self.device).float()  # Keep input as float32 for CT-CLIP
            
            # Log input shape for debugging
            logger.debug(f"Input tensor shape: {x.shape}")
            
            with torch.no_grad():
                # Apply patch embedding (this part still works reliably)
                patch_embedded = self.vision_encoder.to_patch_emb(x)
                
                # Log the shape for debugging
                b, t, h, w, c = patch_embedded.shape
                logger.debug(f"Patch embedded shape: {b}, {t}, {h}, {w}, {c}")
                
                # Simple pooling approach (average across spatial and temporal dimensions)
                # First average across spatial dimensions (h, w)
                spatial_pooled = patch_embedded.mean(dim=(2, 3))  # -> [b, t, c]
                
                # Then average across temporal dimension (t)
                temporal_pooled = spatial_pooled.mean(dim=1)  # -> [b, c]
                
                # Convert the input to match projection layer dtype before passing
                if self.dtype is not None:
                    temporal_pooled = temporal_pooled.to(dtype=self.dtype)
                
                # Project to feature dimension
                features = self.projection(temporal_pooled)
                
                return features
            
        except Exception as e:
            logger.error(f"Error in RobustVisionFeatureExtractor: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return placeholder features in case of error
            return torch.randn(x.size(0), self.feature_dim, device=self.device, 
                              dtype=self.dtype if self.dtype is not None else torch.float32)


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for attending from text to vision features
    """
    def __init__(self, text_dim=768, vision_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(text_dim, text_dim)
        self.key = nn.Linear(vision_dim, text_dim)
        self.value = nn.Linear(vision_dim, text_dim)
        
        self.multihead = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_features, vision_features):
        """
        Args:
            text_features: [batch_size, seq_len, text_dim]
            vision_features: [batch_size, vision_dim]
        """
        # Ensure consistent dtype - convert vision features to match text features
        text_dtype = text_features.dtype
        vision_features = vision_features.to(dtype=text_dtype)
        
        # Make sure all layer parameters match the input dtype
        if self.query.weight.dtype != text_dtype:
            self.query = self.query.to(dtype=text_dtype)
            self.key = self.key.to(dtype=text_dtype)
            self.value = self.value.to(dtype=text_dtype)
            self.multihead = self.multihead.to(dtype=text_dtype)
            self.norm = self.norm.to(dtype=text_dtype)
            
        # Project vision features to match text dimension
        vision_features = vision_features.unsqueeze(1)  # [batch_size, 1, vision_dim]
        
        # Multi-head attention
        queries = self.query(text_features)
        keys = self.key(vision_features)
        values = self.value(vision_features)
        
        # Apply attention
        attn_output, _ = self.multihead(
            query=queries,
            key=keys,
            value=values
        )
        
        # Add residual connection and normalize
        output = self.norm(text_features + self.dropout(attn_output))
        
        return output

class CTReportGenerator(nn.Module):
    """
    End-to-end model for CT scan report generation
    """
    def __init__(self, llm, vision_feature_extractor, cross_attention=None):
        super().__init__()
        self.llm = llm
        self.vision_feature_extractor = vision_feature_extractor
        self.tokenizer = None  # Will be set after initialization
        
        # Determine the LLM dtype for consistency
        self.llm_dtype = next(llm.parameters()).dtype
        logger.info(f"LLM is using dtype: {self.llm_dtype}")
        
        # Convert vision feature extractor to match
        vision_feature_extractor.projection = vision_feature_extractor.projection.to(dtype=self.llm_dtype)
        
        # Create cross-attention if not provided
        if cross_attention is None:
            self.cross_attention = CrossAttentionLayer(
                text_dim=self.llm.config.hidden_size,
                vision_dim=vision_feature_extractor.feature_dim
            ).to(self.llm_dtype)
        else:
            self.cross_attention = cross_attention.to(self.llm_dtype)
    
    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass for training
        """
        # Extract visual features - convert to LLM dtype
        visual_features = self.vision_feature_extractor(images).to(dtype=self.llm_dtype)
        
        # Get LLM hidden states for the prompt
        llm_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Access hidden states
        last_hidden_state = llm_outputs.hidden_states[-1]
        
        # Apply cross-attention between visual features and text features
        cross_attn_output = self.cross_attention(last_hidden_state, visual_features)
        
        # Prepare for generation (replace the original hidden states)
        lm_head_output = self.llm.lm_head(cross_attn_output)
        
        # Return logits for training
        return lm_head_output

    def generate_report(self, images, prompt, max_length=512, temperature=0.7):
        """
        Generate a CT scan report
        
        Args:
            images: CT scan images [batch_size, channels, depth, height, width]
            prompt: Text prompt to condition the generation
            max_length: Maximum length of the generated report
            temperature: Temperature for sampling
            
        Returns:
            str: Generated report
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please set the tokenizer before generation.")
            
        device = images.device
        tokenizer = self.tokenizer
        
        # Tokenize the prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Extract visual features
        with torch.no_grad():
            visual_features = self.vision_feature_extractor(images)
        
        # Generate text with visual conditioning
        generated_ids = []
        
        # Start with the input prompt tokens
        curr_ids = inputs["input_ids"]
        curr_mask = inputs["attention_mask"]
        
        # Auto-regressive generation
        for _ in range(max_length):
            with torch.no_grad():
                # Get LLM hidden states
                llm_outputs = self.llm(
                    input_ids=curr_ids,
                    attention_mask=curr_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get the last hidden state
                last_hidden_state = llm_outputs.hidden_states[-1]
                
                # Apply cross-attention
                cross_attn_output = self.cross_attention(last_hidden_state, visual_features)
                
                # Get next token logits (from the last position)
                next_token_logits = self.llm.lm_head(cross_attn_output)[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the generated token
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            curr_mask = torch.cat([curr_mask, torch.ones_like(next_token)], dim=1)
            
            # Check if we've generated the end token
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(curr_ids[0], skip_special_tokens=True)
        
        # Extract the generated report (remove the prompt)
        report = generated_text[len(prompt):]
        
        return report

def load_model(model_path, device=None):
    """
    Helper function to load a saved model
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model to
        
    Returns:
        CTReportGenerator: Loaded model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import os
    import sys
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load LLM and tokenizer
    llm_name = "epfl-llm/meditron-7b"  # This would need to be passed or retrieved from checkpoint
    
    # Load LLM
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, 
        torch_dtype=torch.bfloat16, 
        use_auth_token=True
    ).to(device)
    
    # Apply LoRA (with the same config as during training)
    lora_config = LoraConfig(
        r=16,  # These values would need to be passed or retrieved from checkpoint
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    llm = get_peft_model(llm, lora_config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Import CT-CLIP
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ct_clip.pretrained_model import ctclip
        
        # Create feature extractor
        vision_feature_extractor = RobustVisionFeatureExtractor(ctclip, device=device)
        
        # Create cross-attention
        cross_attention = CrossAttentionLayer(
            text_dim=llm.config.hidden_size,
            vision_dim=vision_feature_extractor.feature_dim
        ).to(device)
        
        # Create model
        model = CTReportGenerator(
            llm=llm,
            vision_feature_extractor=vision_feature_extractor,
            cross_attention=cross_attention
        ).to(device)
        
        # Set tokenizer
        model.tokenizer = tokenizer
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Model loaded from {model_path}")
        
        return model
        
    except ImportError:
        logger.error("CT-CLIP not found in path. Make sure it's properly imported.")
        return None