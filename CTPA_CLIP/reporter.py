from torch._tensor import Tensor
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertLMHeadModel
from pretrained_model import image_encoder
from ct_clip import CTCLIP
from transformers import AutoTokenizer



class CTCLIPReportGenerator(nn.Module):
    def __init__(self, ctclip):
        super().__init__()
        
        # Load CTCLIP model (which contains CTViT and BERT-based text encoder)
        self.ctclip = ctclip
        
        # Use the same tokenizer as the CTCLIP text encoder
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True )

        
    def encode_ct_scan(self, ct_scan):
        """Extract multimodal features from 3D CT scan using CTCLIP."""
        visual_features = self.ctclip.visual_transformer.to_patch_emb(ct_scan)
        patch_count = visual_features.shape[1] * visual_features.shape[2] * visual_features.shape[3]
    
        print(f"Generated patch count: {patch_count}")
        visual_features = self.ctclip.visual_transformer.encode(visual_features)
        
        # Debugging: Check feature shape

        # Apply proper 3D Average Pooling to ensure feature size matches `to_visual_latent`
        pooled_features = F.adaptive_avg_pool3d(visual_features, (12, 12, 2))  # Reduce dimensions
        vision_embedding = pooled_features.view(pooled_features.size(0), -1)  # Flatten
        
        # Debugging: Confirm new shape
        print(f"Resized Feature Shape: {vision_embedding.shape}")

        # Ensure the shape matches expected input for `to_visual_latent`
        expected_dim = self.ctclip.to_visual_latent.in_features
        if vision_embedding.shape[1] != expected_dim:
            print(f"Resizing from {vision_embedding.shape[1]} to {expected_dim}")
            vision_embedding = F.adaptive_avg_pool1d(vision_embedding.unsqueeze(1), expected_dim).squeeze(1)

        # Convert to latent space
        vision_embedding = self.ctclip.to_visual_latent(vision_embedding)

        # Ensure vision_embedding is correctly mapped before `to_text_latent`
        if vision_embedding.shape[1] != 768:
            print(f"Resizing vision embedding from {vision_embedding.shape[1]} to 768 for text alignment")
            vision_embedding = F.adaptive_avg_pool1d(vision_embedding.unsqueeze(1), 768).squeeze(1)

        text_embedding = self.ctclip.to_text_latent(vision_embedding)  # Align with text space
        
        return text_embedding

    
    def generate_report(self, ct_scan, temperature=1.2, max_length=128, repetition_penalty=1.2):
        """Generate a radiology report from a 3D CT scan using CTCLIP."""
        text_embedding = self.encode_ct_scan(ct_scan)

        # Ensure input is properly formatted for the text model
        # input_ids = torch.argmax(text_embedding, dim=-1).unsqueeze(0)
        input_ids = torch.zeros((1, 1), dtype=torch.long).to(text_embedding.device)


        # Ensure the text transformer is BertLMHeadModel
        if not isinstance(self.ctclip.text_transformer, BertLMHeadModel):
            raise TypeError("CTCLIP's text transformer must be BertLMHeadModel for text generation.")

        # Generate text with improved sampling
        generated_ids = self.ctclip.text_transformer.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,  # Increase randomness
            top_k=50,  # Limit vocabulary choices
            top_p=0.95,  # Enable nucleus sampling
            repetition_penalty=repetition_penalty,  # Reduce repetition
            do_sample=True  # Ensure diversity in generation
        )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)




def load_preprocessed_scan(file_path):
    data = np.load(file_path)
    
    # Assuming the array is saved as the first (and only) item in the .npz file
    scan_array = data[data.files[0]]
    
    # Convert to torch tensor
    # Add batch and channel dimensions
    tensor = torch.from_numpy(scan_array).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, depth, height, width]
    
    return tensor

def preprocess_ct_scan(ct_scan, target_size=480, target_depth=240):
    # Ensure input is in expected shape
    if ct_scan.dim() == 3:
        ct_scan = ct_scan.unsqueeze(0)  # Add channel dim

    batch, channel, depth, height, width = ct_scan.shape
    
    # Enforce depth divisibility by 10 (to match pt=10 in to_patch_emb)
    target_depth = (target_depth // 10) * 10  
    target_size = (target_size // 20) * 20  # Ensure divisibility for p1=20, p2=20

    # Resize CT scan
    ct_scan_resized = F.interpolate(
        ct_scan.float(),
        size=(target_depth, target_size, target_size),
        mode="trilinear",
        align_corners=False
    )
    
    return ct_scan_resized


text_encoder = BertLMHeadModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

ctclip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_text = 768,
    dim_image = 294912,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = '/teamspace/studios/this_studio/data/test_preprocessed/test_PE/test_PE45278df/PE45278df.npz'

# Load the preprocessed scan
ct_scan = load_preprocessed_scan(file_path)
ct_scan = preprocess_ct_scan(ct_scan)
ct_scan = ct_scan.to(device) 
print("Preprocessed scan shape:", ct_scan.shape)

model = CTCLIPReportGenerator(ctclip).to(device) 


report = model.generate_report(ct_scan)
print(report)