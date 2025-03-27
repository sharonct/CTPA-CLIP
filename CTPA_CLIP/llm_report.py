import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForConditionalGeneration
import numpy as np

class CTPAReportGenerator:
    def __init__(self, ctclip_model, text_tokenizer, text_decoder):
        """
        Initialize the CTPA report generation module for Pulmonary Embolism detection.
        
        Args:
            ctclip_model (CTCLIP): Pre-trained CT-CLIP model
            text_tokenizer (BertTokenizer): Tokenizer for text processing
            text_decoder (BertForConditionalGeneration): Language model for text generation
        """
        self.ctclip = ctclip_model
        self.tokenizer = text_tokenizer
        self.decoder = text_decoder
        self.device = next(ctclip_model.parameters()).device

        # PE-specific findings and characteristics
        self.pe_characteristics = [
            "Pulmonary Embolism",
            "Contrast filling defects",
            "Vascular occlusion",
            "Segmental or subsegmental vessel involvement",
            "Clot location in pulmonary arteries",
            "Perfusion defects",
            "Hampton's hump",
            "Wedge-shaped peripheral opacity",
            "Right ventricular strain",
            "Pleural-based opacity"
        ]

    def extract_visual_features(self, ctpa_scan):
        """
        Extract visual features from the contrast-enhanced CTPA scan.
        
        Args:
            ctpa_scan (torch.Tensor): Input CTPA scan tensor
        
        Returns:
            torch.Tensor: Visual features
        """
        with torch.no_grad():
            visual_features = self.ctclip.image_encoder(ctpa_scan)
        return visual_features

    def generate_pe_prompts(self, ctpa_scan, num_candidates=5):
        """
        Generate specialized prompts for Pulmonary Embolism detection.
        
        Args:
            ctpa_scan (torch.Tensor): Input CTPA scan tensor
            num_candidates (int): Number of candidate prompts to generate
        
        Returns:
            list: Candidate prompts for PE report generation
        """
        candidate_prompts = [
            "Comprehensively analyze this CTPA scan for signs of Pulmonary Embolism.",
            "Describe the vascular characteristics and potential thrombotic findings in this contrast-enhanced CT scan.",
            "Provide a detailed radiological assessment focusing on pulmonary arterial obstruction and perfusion defects.",
            "Evaluate the scan for evidence of acute or chronic pulmonary embolism, including clot location and extent.",
            "Assess the pulmonary vasculature for contrast filling defects and potential thrombotic complications."
        ]

        return candidate_prompts[:num_candidates]

    def score_prompt_relevance(self, ctpa_scan, candidate_prompts):
        """
        Score and select the most relevant prompt for PE report generation.
        
        Args:
            ctpa_scan (torch.Tensor): Input CTPA scan tensor
            candidate_prompts (list): List of candidate text prompts
        
        Returns:
            str: Best prompt for report generation
        """
        visual_features = self.extract_visual_features(ctpa_scan)
        
        prompt_scores = []
        for prompt in candidate_prompts:
            text_tokens = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Compute similarity between visual features and text prompts
            similarity_score = self.ctclip(text_tokens, ctpa_scan).mean().item()
            prompt_scores.append(similarity_score)

        best_prompt_index = np.argmax(prompt_scores)
        return candidate_prompts[best_prompt_index]

    def generate_pe_report(self, ctpa_scan, max_length=300):
        """
        Generate a specialized Pulmonary Embolism detection report.
        
        Args:
            ctpa_scan (torch.Tensor): Input CTPA scan tensor
            max_length (int): Maximum length of the generated report
        
        Returns:
            dict: Comprehensive report with findings and analysis
        """
        # Generate and select the best prompt
        candidate_prompts = self.generate_pe_prompts(ctpa_scan)
        best_prompt = self.score_prompt_relevance(ctpa_scan, candidate_prompts)

        # Tokenize the best prompt
        input_ids = self.tokenizer.encode(best_prompt, return_tensors="pt").to(self.device)

        # Generate report using the decoder
        output = self.decoder.generate(
            input_ids, 
            max_length=max_length, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2,
            temperature=0.7
        )

        generated_report_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Perform additional analysis
        findings = self._analyze_pe_characteristics(ctpa_scan)

        return {
            "report_text": generated_report_text,
            "pe_findings": findings,
            "scan_details": {
                "scan_type": "Contrast-enhanced CTPA",
                "purpose": "Pulmonary Embolism Detection"
            }
        }

    def _analyze_pe_characteristics(self, ctpa_scan):
        """
        Perform detailed analysis of PE-specific characteristics.
        
        Args:
            ctpa_scan (torch.Tensor): Input CTPA scan tensor
        
        Returns:
            dict: Detailed findings for each PE characteristic
        """
        findings = {}
        
        for characteristic in self.pe_characteristics:
            # Simulate characteristic detection
            # In a real implementation, this would use specialized detection algorithms
            text_tokens = self.tokenizer(
                f"Assess the presence of {characteristic}", 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            confidence_score = self.ctclip(text_tokens, ctpa_scan).mean().item()
            
            findings[characteristic] = {
                "detected": confidence_score > 0.5,
                "confidence": confidence_score
            }
        
        return findings

def load_ctpa_report_generator(ctclip_path, device='cuda'):
    """
    Utility function to load the CTPA report generation module.
    
    Args:
        ctclip_path (str): Path to the CT-CLIP model checkpoint
        device (str): Device to load the models on
    
    Returns:
        CTPAReportGenerator: Initialized CTPA report generation module
    """
    # Load CT-CLIP model
    ctclip = CTCLIP(...)  # Use your existing model initialization
    ctclip.load(ctclip_path)
    ctclip.to(device)

    # Load tokenizer and text decoder
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized')
    text_decoder = BertForConditionalGeneration.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized')

    report_generator: CTPAReportGenerator = CTPAReportGenerator(ctclip, tokenizer, text_decoder)
    return report_generator

# Example usage
report_generator = load_ctpa_report_generator('/path/to/CT-CLIP_v2.pt')
ctpa_scan_tensor = ...  # Your preprocessed contrast-enhanced CTPA scan tensor
pe_report = report_generator.generate_pe_report(ctpa_scan_tensor)
print(pe_report['report_text'])
print("\nPE Findings:")
for finding, details in pe_report['pe_findings'].items():
    print(f"{finding}: Detected = {details['detected']}, Confidence = {details['confidence']:.2f}")