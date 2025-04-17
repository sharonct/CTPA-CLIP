import os
import logging
import torch
from transformers import AutoTokenizer
from torch.utils.data import random_split
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for CT-CLIP import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from model_components import load_model
from train_module import setup_training
from evaluation_module import run_full_evaluation
from data_utils import load_and_process_dataset, CTReportDataset


def evaluate(config):
    """
    Evaluate a trained model
    
    Args:
        config: Dictionary of configuration parameters
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(config["model_path"], device=device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_and_process_dataset(
        jsonl_file=config["test_data_path"],
        tokenizer=tokenizer,
        target_size=config["target_size"]
    )
    
    # Run evaluation
    results = run_full_evaluation(
        model_path=config["model_path"],
        dataset=dataset,
        output_dir=config["output_dir"],
        visualize_samples=config["visualize_samples"],
        num_visualization_samples=config["num_samples"]
    )
    
    return results

def create_train_val_split(jsonl_file, tokenizer, val_ratio=0.1, target_size=(240, 480, 480)):
    # Load the full dataset
    full_dataset = CTReportDataset(
        jsonl_file=jsonl_file,
        tokenizer=tokenizer,
        target_size=target_size
    )
    
    # Calculate split sizes
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    # Create the split
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    logger.info(f"Split dataset into {train_size} training and {val_size} validation samples")
    
    return train_dataset, val_dataset

def main():
    """
    Main function
    """
    config = {
        # General
        "mode": "train_and_evaluate",  # "train", "evaluate", or "train_and_evaluate"
        "save_path": "/teamspace/studios/this_studio/CTPA-CLIP/models/ct_report",
        "output_dir": "/teamspace/studios/this_studio/CTPA-CLIP/models/evaluation_results",
        
        # Training
        "train_data_path": "/teamspace/studios/this_studio/CTPA-CLIP/data/train_dataset.jsonl",
        "test_data_path": "/teamspace/studios/this_studio/CTPA-CLIP/data/test_dataset.jsonl",
        "batch_size": 4,
        "num_epochs": 5,
        "val_ratio": 0.1,
        "learning_rate": 5e-5,
        "cross_attention_lr": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "accumulation_steps": 2,
        "eval_frequency": 1,
        "eval_samples": 10,  

        # Optimization
        "use_mixed_precision": True,
        "fp16_opt_level": "O2",     

        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,

        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,

        "cache_preprocessed": True,
        "cache_dir": "/teamspace/studios/this_studio/CTPA-CLIP/cache",

        # Evaluation
        "model_path": "/teamspace/studios/this_studio/CTPA-CLIP/models/ct_report/best_model_by_validation.pt",
        "visualize_samples": True,
        "num_samples": 5,
        
        # Data
        "target_size": (240, 480, 480),
        # "target_size": (112, 224, 224),

    }
    
    # Run in the specified mode
    if config["mode"] == "evaluate":
        logger.info("Starting evaluation...")
        results = evaluate(config)
        logger.info(f"Evaluation completed with results: {results}")
        
    elif config["mode"] == "train_and_evaluate":
        logger.info("Starting training with test evaluation...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create train/val split from training data
        train_dataset, val_dataset = create_train_val_split(
            jsonl_file=config["train_data_path"],
            tokenizer=tokenizer,
            val_ratio=config["val_ratio"],
            target_size=config["target_size"]
        )
        
        # Load test dataset
        test_dataset = load_and_process_dataset(
            jsonl_file=config["test_data_path"],
            tokenizer=tokenizer,
            target_size=config["target_size"]
        )
        
        # Setup training with test evaluation
        best_model_path, _ = setup_training(
            train_data=train_dataset,
            val_data=val_dataset,  # Use test dataset for evaluation
            save_path=config["save_path"],
            batch_size=config["batch_size"],
            num_epochs=config["num_epochs"],
            lr=config["learning_rate"],
            cross_attention_lr=config["cross_attention_lr"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"]
        )
        
        logger.info(f"Training completed. Best model: {best_model_path}")
        
        # Run evaluation on the final model
        config["model_path"] = best_model_path
        results = evaluate(config) 
        logger.info(f"Evaluation completed with results: {results}")
    
    else:
        logger.error(f"Invalid mode: {config['mode']}. Must be 'train', 'evaluate', or 'train_and_evaluate'")


if __name__ == "__main__":
    main()
