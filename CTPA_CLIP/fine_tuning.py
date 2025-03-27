from pretrained_model import ctclip
from CTCLIPTrainer import CTClipTrainer


# Freeze all layers except the last few
for param in ctclip.parameters():
    param.requires_grad = False  # Freeze all layers

# Unfreeze only the final layers for adaptation
for param in ctclip.visual_transformer.parameters():
    param.requires_grad = True  # Fine-tune image encoder for contrast-enhanced CT

for param in ctclip.text_transformer.parameters():
    param.requires_grad = True  # Fine-tune text encoder slightly
 

ctclip.train()  # Set to training mode 

trainer = CTClipTrainer(
    ctclip,
    reports_file_train="/teamspace/studios/this_studio/data/train_reports.csv", # Contrast-enhanced CT reports
    reports_file_valid="/teamspace/studios/this_studio/data/test_reports.csv",  # Validation data
    data_train="/teamspace/studios/this_studio/data/train_preprocessed",        # New CT images
    data_valid="/teamspace/studios/this_studio/data/test_preprocessed",         # New CT images
    labels="/teamspace/studios/this_studio/data/inferred_labels.csv",           # Labels for contrast-enhanced CT
    batch_size=2,
    results_folder="/teamspace/studios/this_studio/models",
    num_train_steps=20,  # Reduce steps for fine-tuning
    num_workers=1,
)


trainer.train()