from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
from data import CTReportDataset
from pretrained_model import ctclip
from report_generator import CTCLIPReportGenerator

def load_ctclip_for_report_generation(ctclip, bert_model="microsoft/BiomedVLP-CXR-BERT-specialized"):
    model = CTCLIPReportGenerator(ctclip, bert_model_name=bert_model)
    return model

def fine_tune_report_generator(model, train_dataset, valid_dataset, output_dir, 
                              learning_rate=5e-5, epochs=3, batch_size=8):
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    # Prepare optimizer - only train decoder and projection layers
    optimizer = torch.optim.AdamW([
        {'params': model.decoder.parameters()},
        {'params': model.visual_projection.parameters()}
    ], lr=learning_rate)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for ct_scans, reports in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            ct_scans = ct_scans.to(device)
            
            # Tokenize reports
            report_tokens = model.tokenizer(
                reports, 
                padding='max_length',
                truncation=True,
                max_length=150,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass
            loss, _ = model(ct_scans, report_tokens)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for ct_scans, reports in tqdm(valid_loader, desc="Validation"):
                ct_scans = ct_scans.to(device)
                
                # Tokenize reports
                report_tokens = model.tokenizer(
                    reports, 
                    padding='max_length',
                    truncation=True,
                    max_length=150,
                    return_tensors='pt'
                ).to(device)
                
                # Forward pass
                loss, _ = model(ct_scans, report_tokens)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoint if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, f"report_gen_model_epoch_{epoch+1}.pt"))
            print(f"Model checkpoint saved to {output_dir}")

# Create report generation model
report_generator = load_ctclip_for_report_generation(ctclip)

train_dataset = CTReportDataset("/teamspace/studios/this_studio/data/train_preprocessed", "/teamspace/studios/this_studio/data/train_reports.csv")
test_dataset = CTReportDataset("/teamspace/studios/this_studio/data/test_preprocessed", "/teamspace/studios/this_studio/data/test_reports.csv", split = 'test')

fine_tune_report_generator(report_generator, train_dataset, test_dataset, "/teamspace/studios/this_studio/reports/checkpoints/")


