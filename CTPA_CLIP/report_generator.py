# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import BertTokenizer, BertConfig, BertModel, BertLMHeadModel
# from ct_clip import CTCLIP


# class CTCLIPReportGenerator(nn.Module):
#     def __init__(self, ctclip, bert_model_name="microsoft/BiomedVLP-CXR-BERT-specialized"):
#         super().__init__()
#         # Load the original CTCLIP model

#         ctclip_checkpoint = ctclip
        
#         # Initialize the visual transformer from the original CTCLIP
#         self.visual_transformer = ctclip_checkpoint.visual_transformer

#         # Create BERT config for encoder-decoder architecture
#         config = BertConfig.from_pretrained(bert_model_name)
#         config.is_decoder = True
#         config.add_cross_attention = True

#         # Initialize tokenizer for generation
#         self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

#         # Special tokens for report generation
#         special_tokens = {'pad_token': '[PAD]', 
#                           'bos_token': '[BOS]', 
#                           'eos_token': '[EOS]'}
#         self.tokenizer.add_special_tokens(special_tokens)
        
#         # Initialize the decoder (BERT with language modeling head)
#         self.decoder = BertLMHeadModel.from_pretrained(bert_model_name, config=config)
        
#         # Resize embedding and output layer for new tokens
#         self.decoder.resize_token_embeddings(len(self.tokenizer))
        
#         # Visual projection to match BERT hidden size
#         self.visual_projection = nn.Linear(512, config.hidden_size)
#         self.visual_projection = nn.Sequential(
#         nn.LayerNorm(512),
#         nn.Linear(512, config.hidden_size)
#     )

#         state_dict = ctclip_checkpoint.state_dict()
        
#         # Load the projection layer weights if they exist in the checkpoint
#         if 'to_visual_latent.weight' in state_dict:
#             original_proj = state_dict['to_visual_latent.weight']
#             if original_proj.shape[1] == 294912:  # Ensure dimensions match
#                 self.visual_feature_extraction = nn.Linear(294912, 512)

#                 # Copy weights
#                 self.visual_feature_extraction.weight.data.copy_(original_proj)

#                 # Copy bias if available
#                 if 'to_visual_latent.bias' in state_dict:
#                     self.visual_feature_extraction.bias.data.copy_(state_dict['to_visual_latent.bias'])

#         # Freeze the visual encoder initially (optional)
#         for param in self.visual_transformer.parameters():
#             param.requires_grad = True
        
#     def forward(self, ct_scans, report_tokens=None, generate=False, max_length=150):
#         batch_size = ct_scans.size(0)
        
#         # Get visual features using only the encoder part
#         # Remove torch.no_grad() to allow potential fine-tuning of visual features
#         # Extract visual features - only use the encode method
#         tokens = self.visual_transformer.to_patch_emb(ct_scans)
#         visual_features = self.visual_transformer.encode(tokens)
        
#         # Get the encoded tokens as a flattened representation
#         # Reshape to what we need - average over spatial and temporal dimensions
#         # visual_features = visual_features.mean(dim=[1, 2, 3])  # Average over t, h, w dimensions
#         visual_features = self.visual_transformer.encode(tokens)[:, 0, :]
        
#         # Project to match decoder dimensions
#         visual_context = self.visual_projection(visual_features)
#         print(visual_context.shape) 
#         print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
#         # During training
#         if report_tokens is not None and not generate:
#             # Pass the visual features as encoder_hidden_states for cross-attention
#             decoder_outputs = self.decoder(
#                 input_ids=report_tokens['input_ids'],
#                 attention_mask=report_tokens['attention_mask'],
#                 encoder_hidden_states=visual_context.unsqueeze(1),  # Add sequence dimension
#                 labels=report_tokens['input_ids'],  # For calculating loss
#                 return_dict=True
#             )
            
#             return decoder_outputs.loss, decoder_outputs.logits
            
#         # During inference
#         elif generate:
#             # Start generation with BOS token
#             device = ct_scans.device
#             input_ids = torch.full(
#                 (batch_size, 1), 
#                 self.tokenizer.bos_token_id, 
#                 dtype=torch.long, 
#                 device=device
#             )
            
#             # More robust generation parameters
#             # generated_ids = self.decoder.generate(
#             #     input_ids=input_ids,
#             #     encoder_hidden_states=visual_context.unsqueeze(1),
#             #     max_length=max_length,
#             #     num_beams=5,  # Increased from 4
#             #     no_repeat_ngram_size=2,  # Slightly increased to reduce repetition
#             #     temperature=0.8,  # Slightly increased for more diversity
#             #     early_stopping=True,
#             #     top_k=60,  # Increased top-k
#             #     top_p=0.9,  # Slightly adjusted top-p
#             #     num_return_sequences=1,  # Ensure single sequence
#             #     pad_token_id=self.tokenizer.pad_token_id,
#             #     eos_token_id=self.tokenizer.eos_token_id,
#             #     do_sample=True  # Enable sampling for more natural text
#             # )

#             generated_ids = self.decoder.generate(
#                 input_ids=input_ids,
#                 encoder_hidden_states=visual_context.unsqueeze(1),
#                 max_length=150,
#                 num_beams=1,  # Force greedy
#                 do_sample=False
#             )

#             # Decode the generated IDs to text
#             generated_reports = [
#                 self.tokenizer.decode(g, skip_special_tokens=True) 
#                 for g in generated_ids
#             ]
            
#             return generated_reports

#     def generate_report(self, ct_scan):
#         self.eval()
#         with torch.no_grad():
#             if ct_scan.dim() == 4:  # Add batch dimension if not present
#                 ct_scan = ct_scan.unsqueeze(0)
#             return self.forward(ct_scan, generate=True)[0]  # Return first report for single image


# import torch
# import torch.nn as nn
# from transformers import T5ForConditionalGeneration

# class CTCLIPReportGenerator(nn.Module):
#     def __init__(self, ctclip):
#         super().__init__()
#         self.visual_transformer = ctclip.visual_transformer  # CTViT (image encoder)
#         self.text_decoder = T5ForConditionalGeneration.from_pretrained("t5-small")

#         # Project visual features to match decoder input size
#         self.visual_projection = nn.Linear(512, self.text_decoder.config.d_model)

#     def forward(self, images, input_ids, attention_mask):
#         visual_features = self.visual_transformer(images)
#         visual_features = visual_features.mean(dim=1)  # Mean pooling over spatial tokens
#         visual_context = self.visual_projection(visual_features)

#         outputs = self.text_decoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             encoder_outputs=(visual_context.unsqueeze(1),),
#             return_dict=True
#         )

#         return outputs

#     def generate(self, images):
#         visual_features = self.visual_transformer(images)
#         visual_features = visual_features.mean(dim=1)
#         visual_context = self.visual_projection(visual_features)

#         input_ids = torch.tensor([[self.text_decoder.config.decoder_start_token_id]]).to(images.device)

#         generated_ids = self.text_decoder.generate(
#             input_ids=input_ids,
#             encoder_outputs=(visual_context.unsqueeze(1),),
#             max_length=150,
#             num_beams=5,
#             no_repeat_ngram_size=3,
#             early_stopping=True
#         )

#         return self.text_decoder.decode(generated_ids[0], skip_special_tokens=True)


import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class CTCLIPReportGenerator(nn.Module):
    def __init__(self, ctclip):
        super().__init__()
        # Load the original CTCLIP model
        self.visual_transformer = ctclip.visual_transformer

        # Initialize T5 for text generation
        self.decoder = T5ForConditionalGeneration.from_pretrained("t5-large")

        # **Add the missing tokenizer**
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")

        # Project visual features to match T5 hidden size
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, self.decoder.config.d_model)
        )

    def forward(self, ct_scans, input_ids=None, attention_mask=None, generate=False):
        batch_size = ct_scans.size(0)

        # Extract visual features using ViT
        tokens = self.visual_transformer.to_patch_emb(ct_scans)
        visual_features = self.visual_transformer.encode(tokens)

        # ✅ FIXED: Use mean pooling to get a proper feature vector
        visual_features = visual_features.mean(dim=1)

        # Project visual features to match T5 input size
        visual_context = self.visual_projection(visual_features)  # Shape: (batch_size, hidden_size)

        # ✅ FIXED: Reshape to match T5 expected format (batch_size, seq_length=1, hidden_size)
        visual_context = visual_context.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)

        if generate:
            return self.generate(visual_context)

        # ✅ FIXED: Correctly pass visual_context to the decoder
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=(visual_context,),  # ✅ Correct shape
            labels=input_ids,  
            return_dict=True
        )

        return decoder_outputs.loss, decoder_outputs.logits



    def generate(self, visual_context):
        """
        Generate a radiology report from visual embeddings.
        """
        batch_size = visual_context.shape[0]
        input_ids = torch.full(
            (batch_size, 1), 
            self.decoder.config.decoder_start_token_id, 
            dtype=torch.long, 
            device=visual_context.device
        )

        # Beam search with top-k & nucleus sampling
        generated_ids = self.decoder.generate(
            input_ids=input_ids,
            encoder_outputs=(visual_context.unsqueeze(1),),
            max_length=150,
            num_beams=5,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            early_stopping=True
        )

        return self.decoder.decode(generated_ids[0], skip_special_tokens=True)

    def generate_report(self, ct_scan):
        """
        Generate a report for a single CT scan.
        """
        self.eval()
        with torch.no_grad():
            if ct_scan.dim() == 4:  # Add batch dimension if not present
                ct_scan = ct_scan.unsqueeze(0)
            return self.forward(ct_scan, generate=True)[0]  # Return first report
