import torch
import logging
import json
import numpy as np
import os

import nltk
nltk.download('punkt')
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from vqa_meditron import VisionFeatureExtractor, CustomVQADataset

class NLGMetricsEvaluator:
    def __init__(self, model, vision_encoder, dataloader, device=None):
        """
        Initialize the NLG metrics evaluator for VQA model.
        
        Args:
            model (torch.nn.Module): The trained language model
            vision_encoder (torch.nn.Module): Vision encoder for feature extraction
            dataloader (torch.utils.data.DataLoader): Validation/test dataloader
            device (torch.device, optional): Compute device. Defaults to cuda if available.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.vision_encoder = vision_encoder
        self.dataloader = dataloader
        
        # Initialize metrics calculators
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Vision feature extractor
        self.vision_feature_extractor = VisionFeatureExtractor(vision_encoder, device=self.device)
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def compute_nlg_metrics(self, references, predictions):
        """
        Compute various NLG metrics between reference and predicted answers.
        
        Args:
            references (list): List of ground truth answers
            predictions (list): List of model-generated answers
        
        Returns:
            dict: Computed NLG metrics
        """
        # BLEU Score
        bleu_scores = [
            sentence_bleu([word_tokenize(ref)], word_tokenize(pred)) 
            for ref, pred in zip(references, predictions)
        ]
        
        # ROUGE Scores
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        for ref, pred in zip(references, predictions):
            rouge_result = self.rouge_scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(rouge_result[key].fmeasure)
        
        # BERTScore
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
        
        # Aggregate metrics
        metrics = {
            'bleu_score': np.mean(bleu_scores),
            'rouge1_score': np.mean(rouge_scores['rouge1']),
            'rouge2_score': np.mean(rouge_scores['rouge2']),
            'rougeL_score': np.mean(rouge_scores['rougeL']),
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }
        
        return metrics

    def evaluate(self, max_samples=None):
        """
        Perform model evaluation on the dataset.
        
        Args:
            max_samples (int, optional): Limit number of samples to evaluate
        
        Returns:
            dict: Evaluation results including NLG metrics
        """
        self.model.eval()
        all_references = []
        all_predictions = []
        
        with torch.no_grad():
            for idx, (images, texts) in enumerate(self.dataloader):
                if max_samples and idx >= max_samples:
                    break
                
                # Move images to device
                images = images.to(self.device)
                
                try:
                    # Extract visual features
                    vision_embedding = self.vision_feature_extractor(images)
                    
                    # Tokenize input texts
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    
                    # Generate predictions
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=128,
                        num_return_sequences=1,
                        do_sample=False
                    )
                    
                    # Decode predictions and references
                    predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    references = texts
                    
                    all_references.extend(references)
                    all_predictions.extend(predictions)
                    
                except Exception as e:
                    self.logger.error(f"Evaluation error for batch {idx}: {e}")
                    continue
        
        # Compute NLG metrics
        nlg_metrics = self.compute_nlg_metrics(all_references, all_predictions)
        
        # Save detailed results
        results = {
            'num_samples': len(all_references),
            'metrics': nlg_metrics,
            'predictions': all_predictions,
            'references': all_references
        }
        
        # Optional: Save results to JSON
        self._save_results(results)
        
        return results

    def _save_results(self, results, filename='nlg_evaluation_results.json'):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results (dict): Evaluation results
            filename (str): Output filename
        """
        try:
            output_path = os.path.join("/teamspace/studios/this_studio/vqa/evaluation", filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            self.logger.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

def evaluate_model(model, vision_encoder, dataloader):
    """
    Wrapper function for model evaluation.
    
    Args:
        model (torch.nn.Module): Trained language model
        vision_encoder (torch.nn.Module): Vision encoder
        dataloader (torch.utils.data.DataLoader): Validation/test dataloader
    
    Returns:
        dict: Evaluation metrics
    """
    evaluator = NLGMetricsEvaluator(model, vision_encoder, dataloader)
    return evaluator.evaluate(max_samples=100)  # Limit to 100 samples for demonstration

# Update main function to include evaluation
def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Meditron-7B as LLM
    llm_name = "epfl-llm/meditron-7b"
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, use_auth_token=True).to(device)

    # Load CT-ViT as visual encoder
    from pretrained_model import ctclip
    vision_encoder = ctclip.visual_transformer

    # Stage 2: Fine-tuning with LoRA
    lora_config = LoraConfig(
        r=8,           # Rank of the update matrices
        lora_alpha=16, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    llm = get_peft_model(llm, lora_config)

    dataset = CustomVQADataset("/teamspace/studios/this_studio/data/vqa_dataset_eval.jsonl")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # After training, perform evaluation
    evaluation_results = evaluate_model(llm, vision_encoder, dataloader)
    print("Evaluation Metrics:")
    for metric, value in evaluation_results['metrics'].items():
        print(f"{metric}: {value}")