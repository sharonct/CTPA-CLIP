# from report_generator import CTCLIPReportGenerator
# from torch.utils.data import DataLoader
# import os
# import torch
# from tqdm import tqdm
# from data import CTReportDataset
# from pretrained_model import ctclip
# from report_generator import CTCLIPReportGenerator
# from CTCLIP_report import AdvancedCTReportGenerator

# def load_ctclip_for_report_generation(ctclip, bert_model="microsoft/BiomedVLP-CXR-BERT-specialized"):
#     model = CTCLIPReportGenerator(ctclip, bert_model_name=bert_model)
#     return model

# def fine_tune_report_generator(model, train_dataset, valid_dataset, output_dir, 
#                               learning_rate=3e-5, epochs=15, batch_size=2):
    
#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
#     # Prepare optimizer with different learning rates for different components
#     # optimizer = torch.optim.AdamW([
#     #     {'params': model.text_decoder.parameters(), 'lr': learning_rate},
#     #     {'params': model.visual_projector.parameters(), 'lr': learning_rate * 0.1},
#     #     {'params': model.visual_transformer.parameters(), 'lr': learning_rate * 0.01}
#     # ])
#     optimizer = torch.optim.AdamW([
#         {'params': model.decoder.parameters(), 'lr': 2e-5},  # Lower for BERT
#         {'params': model.visual_projection.parameters(), 'lr': 1e-5},  # Visual projection
#         {'params': model.visual_transformer.parameters(), 'lr': 1e-6}  # Much lower for ViT
#     ])
    
#     # Learning rate scheduler
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, 
#         mode='min', 
#         factor=0.5, 
#         patience=3, 
#         verbose=True
#     )
    
#     # Training loop
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
    
#     best_val_loss = float('inf')
    
#     for epoch in range(epochs):
#         # Training
#         model.train()
#         train_loss = 0
        
#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         for ct_scans, reports in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
#             ct_scans = ct_scans.to(device)
            
#             # Tokenize reports
#             report_tokens = model.tokenizer(
#                 reports, 
#                 padding='max_length',
#                 truncation=True,
#                 max_length=150,
#                 return_tensors='pt'
#             ).to(device)

#             # Forward pass
#             loss, _ = model(ct_scans, report_tokens)
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
        
#         avg_train_loss = train_loss / len(train_loader)
        
#         # Validation
#         model.eval()
#         val_loss = 0
        
#         with torch.no_grad():
#             for ct_scans, reports in tqdm(valid_loader, desc="Validation"):
#                 ct_scans = ct_scans.to(device)
                
#                 # Tokenize reports
#                 report_tokens = model.tokenizer(
#                     reports, 
#                     padding='max_length',
#                     truncation=True,
#                     max_length=150,
#                     return_tensors='pt'
#                 ).to(device)
                
#                 # Forward pass
#                 loss, _ = model(ct_scans, report_tokens)
#                 val_loss += loss.item()
        
#         avg_val_loss = val_loss / len(valid_loader)
        
#         print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
#         # Update learning rate
#         scheduler.step(avg_val_loss)
        
#         # Save checkpoint if improved
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             os.makedirs(output_dir, exist_ok=True)
#             torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': best_val_loss,
#                 'epoch': epoch
#             }, os.path.join(output_dir, f"report_gen_model_best.pt"))
#             print(f"Model checkpoint saved to {output_dir}")

# # Create report generation model
# report_generator: CTCLIPReportGenerator = load_ctclip_for_report_generation(ctclip)

# train_dataset = CTReportDataset("/teamspace/studios/this_studio/data/train_preprocessed", "/teamspace/studios/this_studio/data/train_reports.csv")
# test_dataset = CTReportDataset("/teamspace/studios/this_studio/data/test_preprocessed", "/teamspace/studios/this_studio/data/test_reports.csv", split = 'test')

# fine_tune_report_generator(report_generator, train_dataset, test_dataset, "/teamspace/studios/this_studio/reports/checkpoints/")


# # Create report generation model
# # report_generator: EnhancedCTCLIPReportGenerator = load_ctclip_for_enhanced_report_generation(ctclip)

# # train_dataset = CTReportDataset("/teamspace/studios/this_studio/data/train_preprocessed", "/teamspace/studios/this_studio/data/train_reports.csv")
# # test_dataset = CTReportDataset("/teamspace/studios/this_studio/data/test_preprocessed", "/teamspace/studios/this_studio/data/test_reports.csv", split = 'test')

# # fine_tune_report_generator(report_generator, train_dataset, test_dataset, "/teamspace/studios/this_studio/reports1/checkpoints/")


# # report_generator = AdvancedCTReportGenerator(ctclip_model=ctclip)

# # # Generate report for a CT scan
# # report = report_generator.generate_report(preprocessed_ct_scan)


# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from data_loader import CTReportDataset
# from report_generator import CTCLIPReportGenerator
# from pretrained_model import ctclip

# # Paths
# train_image_dir = "/teamspace/studios/this_studio/data/train_preprocessed"
# train_report_csv = "/teamspace/studios/this_studio/data/train_reports.csv"
# test_image_dir = "/teamspace/studios/this_studio/data/test_preprocessed"
# test_report_csv = "/teamspace/studios/this_studio/data/test_reports.csv"

# # Load dataset
# train_dataset = CTReportDataset(train_image_dir, train_report_csv)
# valid_dataset = CTReportDataset(test_image_dir, test_report_csv, split='test')
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=2)

# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CTCLIPReportGenerator(ctclip).to(device)

# # Optimizer & Scheduler
# optimizer = optim.AdamW([
#     {'params': model.text_decoder.parameters(), 'lr': 3e-5},
#     {'params': model.visual_projection.parameters(), 'lr': 3e-5},
#     {'params': model.visual_transformer.parameters(), 'lr': 1e-5}
# ])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# # Training Loop
# for epoch in range(5):
#     model.train()
#     train_loss = 0
#     for images, input_ids, attention_mask in train_loader:
#         images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

#         optimizer.zero_grad()
#         outputs = model(images, input_ids, attention_mask)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
    
#     avg_train_loss = train_loss / len(train_loader)
#     print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

#     scheduler.step(avg_train_loss)

# # Save Model
# torch.save(model.state_dict(), "/teamspace/studios/this_studio/reports/checkpoints/report_gen_model_best.pt")


from report_generator import CTCLIPReportGenerator
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
from data import CTReportDataset
from pretrained_model import ctclip
from report_generator import CTCLIPReportGenerator
from CTCLIP_report import AdvancedCTReportGenerator

def load_ctclip_for_report_generation(ctclip, bert_model="microsoft/BiomedVLP-CXR-BERT-specialized"):
    model = CTCLIPReportGenerator(ctclip, bert_model_name=bert_model)
    return model

def load_ctclip_for_report_generation_2(ctclip):
    model = CTCLIPReportGenerator(ctclip)
    return model


def fine_tune_report_generator(model, train_dataset, valid_dataset, output_dir, 
                              learning_rate=3e-5, epochs=15, batch_size=2):
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    # Prepare optimizer with different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': model.decoder.parameters(), 'lr': 2e-5},  # Lower for T5
        {'params': model.visual_projection.parameters(), 'lr': 1e-5},  # Visual projection
        {'params': model.visual_transformer.parameters(), 'lr': 1e-6}  # Lower for ViT
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
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
            
            # ✅ FIXED: Convert `report_tokens` properly
            report_tokens = model.tokenizer(
                reports, 
                padding='max_length',
                truncation=True,
                max_length=150,
                return_tensors='pt'
            )
            input_ids = report_tokens['input_ids'].to(device)
            attention_mask = report_tokens['attention_mask'].to(device)

            # Forward pass
            loss, _ = model(ct_scans, input_ids=input_ids, attention_mask=attention_mask)
            
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
                
                # ✅ FIXED: Convert `report_tokens` properly in validation
                report_tokens = model.tokenizer(
                    reports, 
                    padding='max_length',
                    truncation=True,
                    max_length=150,
                    return_tensors='pt'
                )
                input_ids = report_tokens['input_ids'].to(device)
                attention_mask = report_tokens['attention_mask'].to(device)

                # Forward pass
                loss, _ = model(ct_scans, input_ids=input_ids, attention_mask=attention_mask)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save checkpoint if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'epoch': epoch
            }, os.path.join(output_dir, f"report_gen_model_best.pt"))
            print(f"Model checkpoint saved to {output_dir}")


# Create report generation model
report_generator: CTCLIPReportGenerator = load_ctclip_for_report_generation_2(ctclip)

train_dataset = CTReportDataset("/teamspace/studios/this_studio/data/train_preprocessed", "/teamspace/studios/this_studio/data/train_reports.csv")
test_dataset = CTReportDataset("/teamspace/studios/this_studio/data/test_preprocessed", "/teamspace/studios/this_studio/data/test_reports.csv", split = 'test')

fine_tune_report_generator(report_generator, train_dataset, test_dataset, "/teamspace/studios/this_studio/reports/checkpoints/")


# Create report generation model
# report_generator: EnhancedCTCLIPReportGenerator = load_ctclip_for_enhanced_report_generation(ctclip)

# train_dataset = CTReportDataset("/teamspace/studios/this_studio/data/train_preprocessed", "/teamspace/studios/this_studio/data/train_reports.csv")
# test_dataset = CTReportDataset("/teamspace/studios/this_studio/data/test_preprocessed", "/teamspace/studios/this_studio/data/test_reports.csv", split = 'test')

# fine_tune_report_generator(report_generator, train_dataset, test_dataset, "/teamspace/studios/this_studio/reports1/checkpoints/")

