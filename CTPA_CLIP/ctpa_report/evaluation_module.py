import os
import json
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NLGMetricsEvaluator:
    """
    Optimized evaluator for natural language generation metrics
    """
    def __init__(self, model, vision_encoder=None, dataloader=None, device=None, batch_size=1):
        self.model = model
        self.vision_encoder = vision_encoder
        self.dataloader = dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        
        # Initialize tokenizer
        if model.tokenizer is None:
            from transformers import AutoTokenizer
            model.tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
            model.tokenizer.pad_token = model.tokenizer.eos_token
        
        # Initialize metrics with lazy loading for efficiency
        self.metrics_initialized = False
        self._initialize_metrics_lazy()
    
    def _initialize_metrics_lazy(self):
        """Lazy initialize metrics only when needed"""
        self.sentence_bleu = None
        self.smoothing = None
        self.rouge = None
        self.bert_scorer = None
        self.has_bert_score = False
    
    def _ensure_metrics_initialized(self):
        """Ensure metrics are initialized when needed"""
        if self.metrics_initialized:
            return True
            
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            self.sentence_bleu = sentence_bleu
            self.smoothing = SmoothingFunction().method1
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            from rouge import Rouge
            self.rouge = Rouge()
            
            # Only initialize BERTScore if explicitly requested (it's memory intensive)
            self.has_bert_score = False
            
            self.metrics_initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Error initializing metrics: {e}")
            self.metrics_initialized = False
            return False
    
    def evaluate(self, max_samples=10, temperature=0.7):
        """
        Evaluate the model on the dataloader with optimizations
        
        Args:
            max_samples: Maximum number of samples to evaluate (None for all)
            temperature: Temperature for text generation
            
        Returns:
            dict: Evaluation metrics and sample predictions
        """
        if not self._ensure_metrics_initialized():
            logger.error("Metrics not initialized. Cannot evaluate.")
            return {"error": "Metrics not initialized"}
        
        if self.dataloader is None:
            logger.error("No dataloader provided. Cannot evaluate.")
            return {"error": "No dataloader provided"}
        
        # Set model to evaluation mode
        self.model.eval()
        
        all_references = []
        all_predictions = []
        
        sample_predictions = []
        sample_references = []
        
        # Limit the number of samples if specified
        num_samples = len(self.dataloader) if max_samples is None else min(max_samples, len(self.dataloader))
        
        logger.info(f"Evaluating model on {num_samples} samples...")
        
        # Use a shorter generation length for faster evaluation
        max_gen_length = 256  # Reduced from 512
        
        for i, batch in enumerate(tqdm(self.dataloader, total=num_samples)):
            if i >= num_samples:
                break
            
            try:
                # Move data to device
                images = batch["image"].to(self.device)
                prompt = batch["prompt"][0]  # Assuming batch size 1
                reference = batch["report"][0]  # Assuming batch size 1
                
                # Generate prediction with faster settings
                with torch.no_grad():
                    prediction = self.model.generate_report(
                        images=images,
                        prompt=prompt,
                        max_length=max_gen_length,
                        temperature=temperature
                    )
                
                # Store prediction and reference
                all_predictions.append(prediction)
                all_references.append(reference)
                
                # Store sample predictions for results
                sample_predictions.append(prediction)
                sample_references.append(reference)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
            
            # Explicitly clean up memory after each sample
            if i % 5 == 0:  # Every 5 samples
                torch.cuda.empty_cache()
        
        # Calculate metrics (with more efficient implementation)
        metrics = self._calculate_metrics(all_references, all_predictions)
        
        # Create results dictionary
        results = {
            "num_samples": len(all_references),
            "metrics": metrics,
            "sample_predictions": sample_predictions[:5],  # Limit to 5 samples for brevity
            "sample_references": sample_references[:5]     # Limit to 5 samples for brevity
        }
        
        # Free up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    def _calculate_metrics(self, references, predictions):
        """
        Calculate NLG metrics with optimizations
        
        Args:
            references: List of reference texts
            predictions: List of predicted texts
            
        Returns:
            dict: Metrics dictionary
        """
        metrics = {}
        
        try:
            # ROUGE scores (more memory efficient than BLEU)
            rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
            
            # Process in smaller batches to save memory
            batch_size = 10
            for i in range(0, len(references), batch_size):
                batch_refs = references[i:i+batch_size]
                batch_preds = predictions[i:i+batch_size]
                
                for ref, pred in zip(batch_refs, batch_preds):
                    try:
                        # Skip empty predictions or references
                        if not pred.strip() or not ref.strip():
                            continue
                        
                        # Calculate ROUGE
                        rouge = self.rouge.get_scores(pred, ref)[0]
                        rouge_scores["rouge1"].append(rouge["rouge-1"]["f"])
                        rouge_scores["rouge2"].append(rouge["rouge-2"]["f"])
                        rouge_scores["rougeL"].append(rouge["rouge-l"]["f"])
                    except Exception as e:
                        logger.warning(f"Error calculating ROUGE: {e}")
                        continue
            
            metrics["rouge1_score"] = np.mean(rouge_scores["rouge1"]) if rouge_scores["rouge1"] else 0.0
            metrics["rouge2_score"] = np.mean(rouge_scores["rouge2"]) if rouge_scores["rouge2"] else 0.0
            metrics["rougeL_score"] = np.mean(rouge_scores["rougeL"]) if rouge_scores["rougeL"] else 0.0
            
            # BLEU scores (calculated for a small subset to save time)
            max_bleu_samples = min(20, len(references))
            bleu_scores = []
            
            for i in range(max_bleu_samples):
                try:
                    ref, pred = references[i], predictions[i]
                    
                    # Tokenize
                    ref_tokens = ref.lower().split()
                    pred_tokens = pred.lower().split()
                    
                    # Calculate BLEU
                    bleu = self.sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoothing)
                    bleu_scores.append(bleu)
                except Exception as e:
                    logger.warning(f"Error calculating BLEU: {e}")
                    continue
            
            metrics["bleu_score"] = np.mean(bleu_scores) if bleu_scores else 0.0
            
            # BERTScore only calculated if explicitly enabled and for a small subset
            if self.has_bert_score and hasattr(self, 'bert_scorer') and self.bert_scorer is not None:
                try:
                    # Use at most 10 samples for BERTScore to save memory
                    max_bert_samples = min(10, len(references))
                    valid_pairs = [(predictions[i], references[i]) for i in range(max_bert_samples) 
                                if predictions[i].strip() and references[i].strip()]
                    
                    if valid_pairs:
                        valid_preds, valid_refs = zip(*valid_pairs)
                        
                        # Calculate BERTScore
                        bert_p, bert_r, bert_f1 = self.bert_scorer.score(valid_preds, valid_refs)
                        
                        metrics["bert_precision"] = bert_p.mean().item()
                        metrics["bert_recall"] = bert_r.mean().item()
                        metrics["bert_f1"] = bert_f1.mean().item()
                    else:
                        metrics["bert_precision"] = 0.0
                        metrics["bert_recall"] = 0.0
                        metrics["bert_f1"] = 0.0
                except Exception as e:
                    logger.warning(f"Error calculating BERTScore: {e}")
                    metrics["bert_precision"] = 0.0
                    metrics["bert_recall"] = 0.0
                    metrics["bert_f1"] = 0.0
            
        except Exception as e:
            logger.error(f"Error in _calculate_metrics: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return metrics


def visualize_sample(model, sample, output_dir, sample_idx=0, temperature=0.7):
    """
    Visualize a sample prediction with the CT scan (optimized)
    
    Args:
        model: CT report generator model
        sample: Sample from the dataset
        output_dir: Output directory for visualization
        sample_idx: Sample index for filename
        temperature: Temperature for text generation
        
    Returns:
        dict: Visualization metadata
    """
    try:
        import os
        import matplotlib.pyplot as plt
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Move data to model device
        device = next(model.parameters()).device
        image = sample["image"].to(device)
        
        # Generate prediction (with shorter max length)
        prompt = sample["prompt"]
        reference = sample["report"]
        
        with torch.no_grad():
            prediction = model.generate_report(
                images=image.unsqueeze(0),
                prompt=prompt,
                temperature=temperature,
                max_length=256  # Shorter generation
            )
        
        # Visualize CT scan slices (middle slice for each dimension)
        # Use a more efficient figure size
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Get the image data (assume [C, D, H, W] format)
        img_data = image.cpu().numpy()[0]  # Remove channel dimension
        
        # Get middle slices
        d, h, w = img_data.shape
        slice_d = img_data[d//2, :, :]
        slice_h = img_data[:, h//2, :]
        slice_w = img_data[:, :, w//2]
        
        # Plot slices with reduced DPI for faster rendering
        axes[0].imshow(slice_d, cmap='gray')
        axes[0].set_title(f"Depth={d//2}")
        
        axes[1].imshow(slice_h, cmap='gray')
        axes[1].set_title(f"Height={h//2}")
        
        axes[2].imshow(slice_w, cmap='gray')
        axes[2].set_title(f"Width={w//2}")
        
        # Set title with prediction summary (shorter)
        pred_summary = prediction[:50] + "..." if len(prediction) > 50 else prediction
        fig.suptitle(f"Sample {sample_idx}", fontsize=10)
        
        # Save figure with lower quality for speed
        vis_path = os.path.join(output_dir, f"sample_{sample_idx}_viz.png")
        plt.tight_layout()
        plt.savefig(vis_path, dpi=100)  # Lower DPI
        plt.close(fig)
        
        # Save text
        text_path = os.path.join(output_dir, f"sample_{sample_idx}_text.txt")
        with open(text_path, 'w') as f:
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write(f"REFERENCE:\n{reference}\n\n")
            f.write(f"PREDICTION:\n{prediction}\n")
        
        return {
            "sample_idx": sample_idx,
            "visualization_path": vis_path,
            "text_path": text_path,
            "prediction": prediction,
            "reference": reference
        }
        
    except Exception as e:
        logger.error(f"Error in visualize_sample: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}


def run_full_evaluation(model_path, dataset, output_dir, visualize_samples=True, num_visualization_samples=5):
    """
    Run optimized evaluation on a model
    
    Args:
        model_path: Path to the model checkpoint
        dataset: Dataset for evaluation
        output_dir: Output directory for results
        visualize_samples: Whether to visualize samples
        num_visualization_samples: Number of samples to visualize
        
    Returns:
        dict: Evaluation results
    """
    from model_components import load_model
    from torch.utils.data import DataLoader
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with efficient settings
    model = load_model(model_path, device=device)
    
    if model is None:
        logger.error(f"Failed to load model from {model_path}")
        return {"error": "Failed to load model"}
    
    # Create dataloader with worker settings
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=2,  # Speed up data loading
        pin_memory=True
    )
    
    # Create evaluator
    evaluator = NLGMetricsEvaluator(model, dataloader=dataloader, device=device)
    
    # Run evaluation with limited samples
    max_eval_samples = min(50, len(dataset))  # Limit to 50 samples max
    results = evaluator.evaluate(max_samples=max_eval_samples)
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Visualize fewer samples
    if visualize_samples:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Visualize fewer samples to save time
        num_to_visualize = min(num_visualization_samples, 5)  
        logger.info(f"Visualizing {num_to_visualize} samples...")
        
        for i, sample in enumerate(dataset):
            if i >= num_to_visualize:
                break
            
            # Visualize sample
            vis_result = visualize_sample(model, sample, vis_dir, sample_idx=i)
            
            if "error" in vis_result:
                logger.error(f"Error visualizing sample {i}: {vis_result['error']}")
                
            # Clean memory after each sample
            torch.cuda.empty_cache()
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return results