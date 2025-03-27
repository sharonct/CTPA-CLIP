import os
import torch
import numpy as np
import nibabel as nib
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from test import preprocess_ct_scan, load_preprocessed_scan
import nltk
nltk.download('punkt')


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vision_feature_extractor():
    """
    Load vision feature extractor
    
    Returns:
        callable: Vision feature extraction function
    """
    from vqa_meditron import VisionFeatureExtractor
    from pretrained_model import ctclip
    
    vision_encoder = ctclip.visual_transformer
    return VisionFeatureExtractor(vision_encoder)

def prepare_model_for_inference(
    checkpoint_dir, 
    llm_name="epfl-llm/meditron-7b"
):
    """
    Prepare model and tokenizer for inference
    
    Args:
        checkpoint_dir (str): Directory containing model checkpoints
        llm_name (str): Pretrained LLM name
    
    Returns:
        tuple: (model, tokenizer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find the latest checkpoint
    checkpoints = [
        os.path.join(checkpoint_dir, f) 
        for f in os.listdir(checkpoint_dir) 
        if f.endswith('.pth')
    ]
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    logger.info(f"Using checkpoint: {latest_checkpoint}")
    
    # Load base LLM
    model = AutoModelForCausalLM.from_pretrained(
        llm_name, 
        torch_dtype=torch.bfloat16, 
        use_auth_token=True
    ).to(device)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,           # Rank of the update matrices
        lora_alpha=16, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    # Load checkpoint state dict
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_ct_scan_inference(
    model, 
    tokenizer, 
    ct_tensor, 
    vision_feature_extractor, 
    question="What medical condition can be observed in this CT scan?"
):
    """
    Generate inference for a single CT scan
    
    Args:
        model (torch.nn.Module): Trained model
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        ct_tensor (torch.Tensor): Processed CT scan tensor
        vision_feature_extractor (callable): Vision feature extraction function
        question (str): Medical question for the CT scan
    
    Returns:
        str: Model's generated response
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        # Extract visual features
        visual_features = vision_feature_extractor(ct_tensor.to(device))
        
        # Prepare question input
        inputs = tokenizer(
            question, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        # Generate response
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=256,
            num_return_sequences=1,
            temperature=0.7
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def main(ct_scan_path, question=None):
    """
    Main inference pipeline for single CT scan
    
    Args:
        ct_scan_path (str): Path to CT scan .nii file
        question (str, optional): Medical question for the CT scan
    """
    # Configuration
    model_checkpoint_dir = "/teamspace/studios/this_studio/vqa/model"

    ct_scan = load_preprocessed_scan(ct_scan_path)
    ct_scan = preprocess_ct_scan(ct_scan)
        
    # Prepare CT scan    
    # Load vision feature extractor
    vision_feature_extractor = load_vision_feature_extractor()
    
    # Prepare model
    model, tokenizer = prepare_model_for_inference(model_checkpoint_dir)
    
    # Set default question if not provided
    if question is None:
        question = "Analyze this medical CT scan and describe any notable medical conditions or abnormalities."
    
    # Generate inference
    response = generate_ct_scan_inference(
        model, 
        tokenizer, 
        ct_scan, 
        vision_feature_extractor, 
        question
    )
    
    # Print results
    print("Input CT Scan:", ct_scan_path)
    print("Question:", question)
    print("Model Response:", response)

if __name__ == "__main__":
    import sys
    
    ct_scan_path = '/teamspace/studios/this_studio/data/test_preprocessed/test_PE/test_PE4527ded/PE4527ded.npz'
    question = None
    
    main(ct_scan_path, question)