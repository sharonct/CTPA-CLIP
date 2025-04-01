# CTPA-CLIP

## Overview
CTPA-CLIP is a project aimed at generating detailed reports from CTPA (Computed Tomography Pulmonary Angiography) scans. This tool leverages advanced image processing techniques and machine learning models to analyze scan data and produce comprehensive diagnostic reports.

## Features
- Automated analysis of CTPA scans
- Generation of detailed diagnostic reports
- Integration with existing medical imaging workflows

## Essential Files and Workflow
1. **Generate VQA Dataset**: Prepare the dataset required for Visual Question Answering (VQA) using CTPA scans.
2. **Fine-tune Model**: Fine-tune the pre-trained model using the VQA dataset.
3. **Perform Inference**: Run inference on new data to generate reports.
4. **Evaluate Generated Reports**: Evaluate the quality of the generated reports.
5. **Usage**: Use the fine-tuned model to perform inference on CTPA scans.

### File Details

#### 1. Generate VQA Dataset
The dataset preparation code should be executed to create the required training and evaluation data for VQA. Ensure your data is in the appropriate format as expected by the model.

#### 2. Fine-tune Model: `vqa_meditron.py`
This script is responsible for fine-tuning the pretrained model using VQA dataset.
- **Key Components**:
  - `VisionFeatureExtractor`: Extracts features from CTPA images.
  - `CustomVQADataset`: Custom dataset class for loading and preprocessing the VQA dataset.
  - `train_model`: Function to train the model using the provided dataset.

#### 3. Perform Inference: `vqa_inference.py`
This script is used for running inference on new data to generate reports.
- **Key Components**:
  - `prepare_model_for_inference`: Loads the fine-tuned model and prepares it for inference.
  - `generate_responses`: Generates responses for the test dataset using the trained model.
  - `load_test_dataset`: Loads the test dataset.
  - `compute_custom_metrics`: Computes various Natural Language Generation (NLG) metrics for evaluation.

#### 4. Evaluate Generated Reports: `evaluate_reports.py`
This script evaluates the quality of the generated reports using various NLG metrics.
- **Key Components**:
  - `NLGMetricsEvaluator`: Class for evaluating the model using BLEU, ROUGE, and BERTScore metrics.
  - `evaluate`: Function to perform evaluation on the dataset.

#### 5. Usage: `ct_scan_inference.py`
This script is used to perform inference on new CTPA scans using the fine-tuned model to generate diagnostic reports.