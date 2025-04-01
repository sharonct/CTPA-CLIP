import os
import torch
import json
import logging
import numpy as np
import pandas as pd
import re
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

from vqa_meditron import VisionFeatureExtractor
from ct_clip.pretrained_model import ctclip

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest model checkpoint in the directory
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
    
    Returns:
        str: Path to the latest checkpoint
    """
    # List all files in the directory
    checkpoints = [
        os.path.join(checkpoint_dir, f) 
        for f in os.listdir(checkpoint_dir) 
        if f.endswith('.pth')
    ]
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort checkpoints by modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def prepare_model_for_inference(checkpoint_dir, llm_name="epfl-llm/meditron-7b"):
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
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_responses(model, tokenizer, test_data, vision_feature_extractor):
    """
    Generate model responses for test dataset
    
    Args:
        model (torch.nn.Module): Trained model
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        test_data (list): Test dataset
        vision_feature_extractor (VisionFeatureExtractor): Vision feature extractor
    
    Returns:
        list: Generated responses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    responses = []

    with torch.no_grad():
        for item in test_data:
            try:
                # Load and process image
                image_tensor = torch.tensor(
                    np.load(item["image_path"])["arr_0"], 
                    dtype=torch.float32
                ).unsqueeze(0).to(device)
                
                # Extract visual features
                visual_features = vision_feature_extractor(image_tensor)
                
                # Prepare question input
                question = item["question"]
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
                    max_length=128,  # Adjust as needed
                    num_return_sequences=1,
                    temperature=0.7
                )
                
                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append({
                    "question": question,
                    "ground_truth": item["answer"],
                    "generated_response": response
                })
            
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                responses.append({
                    "question": question,
                    "ground_truth": item.get("answer", "N/A"),
                    "generated_response": "ERROR"
                })
    
    return responses

def load_test_dataset(jsonl_path, target_size=480, target_depth=240):
    """
    Load test dataset
    
    Args:
        jsonl_path (str): Path to test dataset
        target_size (int): Target image size
        target_depth (int): Target image depth
    
    Returns:
        list: Test dataset items
    """
    test_data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            test_data.append(item)
    return test_data

def compute_custom_metrics(responses):
    """
    Compute custom NLG metrics
    
    Args:
        responses (list): Generated responses with ground truth
    
    Returns:
        dict: Evaluation metrics
    """
    # Prepare for metrics calculation
    generated_responses = [resp["generated_response"] for resp in responses]
    ground_truth = [resp["ground_truth"] for resp in responses]
    
    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # Metrics calculation
    metrics = {
        "Perfect Matches": 0,
        "ROUGE-1 Precision": [],
        "ROUGE-1 Recall": [],
        "ROUGE-L Precision": [],
        "ROUGE-L Recall": [],
        "BLEU-1 Scores": [],
        "BLEU-4 Scores": []
    }
    
    for gen_resp, true_resp in zip(generated_responses, ground_truth):
        # Perfect match
        if gen_resp.strip().lower() == true_resp.strip().lower():
            metrics["Perfect Matches"] += 1
        
        # Tokenize responses
        gen_tokens = word_tokenize(gen_resp.lower())
        true_tokens = word_tokenize(true_resp.lower())
        
        # ROUGE Calculation
        rouge_scores = rouge.score(true_resp, gen_resp)
        metrics["ROUGE-1 Precision"].append(rouge_scores['rouge1'].precision)
        metrics["ROUGE-1 Recall"].append(rouge_scores['rouge1'].recall)
        metrics["ROUGE-L Precision"].append(rouge_scores['rougeL'].precision)
        metrics["ROUGE-L Recall"].append(rouge_scores['rougeL'].recall)
        
        # BLEU Calculation
        try:
            bleu1_score = sentence_bleu([true_tokens], gen_tokens, weights=(1, 0, 0, 0))
            bleu4_score = sentence_bleu([true_tokens], gen_tokens)
            metrics["BLEU-1 Scores"].append(bleu1_score)
            metrics["BLEU-4 Scores"].append(bleu4_score)
        except Exception as e:
            logger.warning(f"BLEU score calculation error: {e}")
            metrics["BLEU-1 Scores"].append(0)
            metrics["BLEU-4 Scores"].append(0)
    
    # Compute averages
    metrics["Total Samples"] = len(responses)
    metrics["Perfect Match Percentage"] = (metrics["Perfect Matches"] / len(responses)) * 100
    metrics["Average ROUGE-1 Precision"] = np.mean(metrics["ROUGE-1 Precision"])
    metrics["Average ROUGE-1 Recall"] = np.mean(metrics["ROUGE-1 Recall"])
    metrics["Average ROUGE-L Precision"] = np.mean(metrics["ROUGE-L Precision"])
    metrics["Average ROUGE-L Recall"] = np.mean(metrics["ROUGE-L Recall"])
    metrics["Average BLEU-1 Score"] = np.mean(metrics["BLEU-1 Scores"])
    metrics["Average BLEU-4 Score"] = np.mean(metrics["BLEU-4 Scores"])
    
    return metrics

def save_evaluation_results(responses, metrics, output_path):
    """
    Save evaluation results to JSON and CSV
    
    Args:
        responses (list): Generated responses
        metrics (dict): Evaluation metrics
        output_path (str): Base output path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save detailed responses
    with open(f"{output_path}_responses.json", "w") as f:
        json.dump(responses, f, indent=2)
    
    # Save metrics
    with open(f"{output_path}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create CSV for easier viewing
    df = pd.DataFrame(responses)
    df.to_csv(f"{output_path}_responses.csv", index=False)
    
    logger.info(f"Evaluation results saved to {output_path}")

def main():
    # Configuration
    test_dataset_path = "/teamspace/studios/this_studio/data/vqa_dataset_eval.jsonl"
    model_checkpoint_dir = "/teamspace/studios/this_studio/vqa/model"
    output_path = "/teamspace/studios/this_studio/vqa/evaluation/results"
    
    # Additional imports
    import sys
    sys.path.append("/teamspace/studios/this_studio")
    
    vision_encoder = ctclip.visual_transformer
    vision_feature_extractor = VisionFeatureExtractor(vision_encoder)
    
    # Load test dataset
    test_data = load_test_dataset(test_dataset_path)
    
    # Prepare model for inference
    model, tokenizer = prepare_model_for_inference(model_checkpoint_dir)
    
    # Generate responses
    responses = generate_responses(model, tokenizer, test_data, vision_feature_extractor)
    
    # Evaluate metrics
    metrics = compute_custom_metrics(responses)
    
    # Save results
    save_evaluation_results(responses, metrics, output_path)
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()