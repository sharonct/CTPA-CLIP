import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from data_loader import CTReportDataset
import torch.nn.functional as F

from reporter import model

def collate_fn(batch):
    images, input_ids, attention_masks = zip(*batch)

    # Find max spatial dimensions in the batch
    max_d = max(img.shape[1] for img in images)  # Depth
    max_h = max(img.shape[2] for img in images)  # Height
    max_w = max(img.shape[3] for img in images)  # Width

    # Ensure divisibility for patching
    patch_size = 20  # Change based on model requirements
    max_d = (max_d // patch_size) * patch_size  # Round down to nearest multiple
    max_h = (max_h // patch_size) * patch_size
    max_w = (max_w // patch_size) * patch_size

    padded_images = [
        F.interpolate(img.unsqueeze(0), size=(max_d, max_h, max_w), mode="trilinear", align_corners=False).squeeze(0)
        for img in images
    ]

    return torch.stack(padded_images), torch.stack(input_ids), torch.stack(attention_masks)

# Load dataset and create DataLoader
train_dataset = CTReportDataset(data_folder="/teamspace/studios/this_studio/data/train_preprocessed", csv_file="/teamspace/studios/this_studio/data/train_reports.csv", split='train')
val_dataset = CTReportDataset(data_folder="/teamspace/studios/this_studio/data/test_preprocessed", csv_file="/teamspace/studios/this_studio/data/test_reports.csv", split='test')

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for ct_scans, input_ids, attention_masks in train_loader:
        ct_scans, input_ids, attention_masks = ct_scans.to(device), input_ids.to(device), attention_masks.to(device)

        optimizer.zero_grad()

        # Encode the CT scan
        text_embedding = model.encode_ct_scan(ct_scans)

        # Forward pass through the text model
        outputs = model.ctclip.text_transformer(
            input_ids=input_ids, attention_mask=attention_masks, labels=input_ids
        )
        loss = outputs.loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save fine-tuned model
torch.save(model.state_dict(), "/teamspace/studios/this_studio/reports/fine_tuned_ctclip.pth")
