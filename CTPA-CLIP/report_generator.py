import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel, BertLMHeadModel
from ct_clip import CTCLIP
from data import CTReportDataset
from pretrained_model import ctclip



class CTCLIPReportGenerator(nn.Module):
    def __init__(self, ctclip, bert_model_name="microsoft/BiomedVLP-CXR-BERT-specialized"):
        super().__init__()
        # Load the original CTCLIP model

        ctclip_checkpoint = ctclip
        
        # Initialize the visual transformer from the original CTCLIP
        self.visual_transformer = ctclip_checkpoint.visual_transformer

        # Create BERT config for encoder-decoder architecture
        config = BertConfig.from_pretrained(bert_model_name)
        config.is_decoder = True
        config.add_cross_attention = True

        # Initialize tokenizer for generation
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Special tokens for report generation
        special_tokens = {'pad_token': '[PAD]', 
                          'bos_token': '[BOS]', 
                          'eos_token': '[EOS]'}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Initialize the decoder (BERT with language modeling head)
        self.decoder = BertLMHeadModel.from_pretrained(bert_model_name, config=config)
        
        # Resize embedding and output layer for new tokens
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        
        # Visual projection to match BERT hidden size
        self.visual_projection = nn.Linear(512, config.hidden_size)

        state_dict = ctclip_checkpoint.state_dict()
        
        # Load the projection layer weights if they exist in the checkpoint
        if 'to_visual_latent.weight' in state_dict:
            original_proj = state_dict['to_visual_latent.weight']
            if original_proj.shape[1] == 294912:  # Ensure dimensions match
                self.visual_feature_extraction = nn.Linear(294912, 512)

                # Copy weights
                self.visual_feature_extraction.weight.data.copy_(original_proj)

                # Copy bias if available
                if 'to_visual_latent.bias' in state_dict:
                    self.visual_feature_extraction.bias.data.copy_(state_dict['to_visual_latent.bias'])

        # Freeze the visual encoder initially (optional)
        for param in self.visual_transformer.parameters():
            param.requires_grad = False
        
    def forward(self, ct_scans, report_tokens=None, generate=False, max_length=150):
        batch_size = ct_scans.size(0)
        
        # Encode the CT scans
        with torch.no_grad():  # Assuming visual encoder is frozen
            # Extract visual features
            visual_features = self.visual_transformer(ct_scans)
            # If the model uses the feature extraction layer before projection
            if hasattr(self, 'visual_feature_extraction'):
                # Reshape if needed
                visual_features = visual_features.view(batch_size, -1)
                visual_features = self.visual_feature_extraction(visual_features)
                
            # Project to match decoder dimensions
            visual_context = self.visual_projection(visual_features)
        
        # During training
        if report_tokens is not None and not generate:
            # Pass the visual features as encoder_hidden_states for cross-attention
            decoder_outputs = self.decoder(
                input_ids=report_tokens['input_ids'],
                attention_mask=report_tokens['attention_mask'],
                encoder_hidden_states=visual_context.unsqueeze(1),  # Add sequence dimension
                labels=report_tokens['input_ids'],  # For calculating loss
                return_dict=True
            )
            
            return decoder_outputs.loss, decoder_outputs.logits
            
        # During inference
        elif generate:
            # Start with BOS token
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]] * batch_size).to(ct_scans.device)
            
            # Generate text auto-regressively
            generated_ids = self.decoder.generate(
                input_ids=input_ids,
                encoder_hidden_states=visual_context.unsqueeze(1),
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the generated IDs to text
            generated_reports = [self.tokenizer.decode(g, skip_special_tokens=True) 
                               for g in generated_ids]
                
            return generated_reports

    def generate_report(self, ct_scan):
        """Convenience method for report generation from a single CT scan"""
        self.eval()
        with torch.no_grad():
            if ct_scan.dim() == 4:  # Add batch dimension if not present
                ct_scan = ct_scan.unsqueeze(0)
            return self.forward(ct_scan, generate=True)[0]  # Return first report for single image
