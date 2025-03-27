import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Tuple

class AdvancedCTReportGenerator(nn.Module):
    def __init__(self, 
                 ctclip_model, 
                 text_model_name='google/flan-t5-base', 
                 latent_dim=512):
        super().__init__()
        
        # 3D CT Image Encoder (from CT-ViT)
        self.visual_transformer = ctclip_model.visual_transformer
        
        # Feature Projection Layer
        self.visual_projection = nn.Sequential(
            nn.Linear(294912, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # Text Generation Backbone (using T5 instead of BERT)
        self.text_generator = T5ForConditionalGeneration.from_pretrained(text_model_name)
        
        # Adapter to align visual features with text model
        self.visual_text_adapter = nn.Sequential(
            nn.Linear(latent_dim, self.text_generator.config.d_model),
            nn.LayerNorm(self.text_generator.config.d_model),
            nn.GELU()
        )
        
    def extract_visual_features(self, ct_scan):
        """Extract and project visual features from 3D CT scan"""
        # Patch embedding
        tokens = self.visual_transformer.to_patch_emb(ct_scan)
        
        # Spatial and temporal encoding
        visual_features = self.visual_transformer.encode(tokens)
        
        # Average pooling across spatial and temporal dimensions
        visual_features = visual_features.mean(dim=[1, 2, 3])
        
        # Project to latent space
        projected_features = self.visual_projection(visual_features)
        
        return projected_features
    
    def forward(self, 
                ct_scan: torch.Tensor, 
                text_tokens: Optional[dict] = None,
                max_length: int = 150,
                generate: bool = False):
        """
        Forward pass for report generation
        
        Args:
            ct_scan (torch.Tensor): 3D CT scan tensor
            text_tokens (dict, optional): Tokenized text for training
            max_length (int): Maximum generation length
            generate (bool): Whether to generate reports
        
        Returns:
            Depending on mode, returns loss or generated reports
        """
        # Extract visual features
        visual_features = self.extract_visual_features(ct_scan)
        
        # Adapt visual features to text model's dimensions
        encoder_hidden_states = self.visual_text_adapter(visual_features).unsqueeze(1)
        
        # Training mode (with teacher forcing)
        if text_tokens is not None and not generate:
            loss = self.text_generator(
                input_ids=text_tokens['input_ids'],
                attention_mask=text_tokens['attention_mask'],
                encoder_hidden_states=encoder_hidden_states,
                labels=text_tokens['input_ids']
            ).loss
            
            return loss
        
        # Inference mode (generation)
        elif generate:
            batch_size = ct_scan.size(0)
            
            # Prepare initial decoder input
            decoder_input_ids = torch.full(
                (batch_size, 1), 
                self.text_generator.config.decoder_start_token_id, 
                dtype=torch.long, 
                device=ct_scan.device
            )
            
            # Generate reports
            generated_ids = self.text_generator.generate(
                encoder_hidden_states=encoder_hidden_states,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                num_return_sequences=1
            )
            
            # Decode generated tokens to text
            generated_reports = [
                self.text_generator.tokenizer.decode(g, skip_special_tokens=True) 
                for g in generated_ids
            ]
            
            return generated_reports
    
    def generate_report(self, ct_scan):
        """
        Convenience method for generating a single report
        
        Args:
            ct_scan (torch.Tensor): Preprocessed 3D CT scan
        
        Returns:
            str: Generated radiology report
        """
        self.eval()
        with torch.no_grad():
            if ct_scan.dim() == 4:  # Add batch dimension if not present
                ct_scan = ct_scan.unsqueeze(0)
            return self.forward(ct_scan, generate=True)[0]

# Optional: Custom Loss for Report Generation
class RadiologyReportLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cosine_similarity_loss = nn.CosineEmbeddingLoss()
        self.alpha = alpha
    
    def forward(self, generated_reports, ground_truth_reports):
        """
        Combine multiple loss signals for report generation
        
        Args:
            generated_reports (torch.Tensor): Model-generated reports
            ground_truth_reports (torch.Tensor): Reference reports
        
        Returns:
            torch.Tensor: Combined loss
        """
        # Cross-entropy loss for token-level accuracy
        ce_loss = self.cross_entropy_loss(generated_reports, ground_truth_reports)
        
        # Cosine similarity loss for semantic alignment
        similarity_target = torch.ones(generated_reports.size(0)).to(generated_reports.device)
        cosine_loss = self.cosine_similarity_loss(generated_reports, ground_truth_reports, similarity_target)
        
        # Weighted combination of losses
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * cosine_loss
        
        return total_loss