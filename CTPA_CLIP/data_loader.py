import os
import glob
import json
from typing import Any
from torch._tensor import Tensor
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm
from transformers import AutoTokenizer

def resize_array(array, current_spacing, target_spacing):
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    
    resized_array = (
        F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False)
        .cpu()
        .numpy()
    )
    return resized_array

class CTReportDataset(Dataset):
    def __init__(
        self,
        data_folder,
        csv_file,
        tokenizer_name=AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True),
        split='train',
        min_slices=20,
        resize_dim=500,
        force_num_frames=True,
        max_length=512
    ):
        self.split = split
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.tokenizer = tokenizer_name
        self.max_length = max_length
        self.accession_to_text = self.load_accession_text(csv_file)
        self.samples = self.prepare_samples()
        percent = 80
        num_files = int((len(self.samples) * percent) / 100)
        self.samples = self.samples[:num_files]
        print(f"Loaded {len(self.samples)} samples.")

        self.transform = transforms.Compose(
            [transforms.Resize((resize_dim, resize_dim)), transforms.ToTensor()]
        )
        self.npz_to_tensor = partial(self.npz_img_to_tensor, transform=self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row["impression_id"]] = row["impressions"]
        return accession_to_text

    def prepare_samples(self):
        samples = []

        for study_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, "*"))):
            study_id = os.path.basename(study_folder)
            for subfolder in glob.glob(os.path.join(study_folder, "*")):
                if self.split == 'train':
                    filename = os.path.basename(subfolder).replace("train_", "")
                else:
                    filename = os.path.basename(subfolder).replace("test_", "")
                
                npz_file = os.path.join(subfolder, filename + ".npz")
                if not os.path.exists(npz_file):
                    continue
                
                accession_number = os.path.basename(npz_file).replace(".npz", "")
                if accession_number not in self.accession_to_text:
                    continue
                
                impression_text = self.accession_to_text[accession_number]
                input_text_concat = (
                    " ".join(impression_text) if impression_text != "Not given." else ""
                )
                
                samples.append((npz_file, input_text_concat))
        
        return samples

    def __len__(self):
        return len(self.samples)

    def npz_img_to_tensor(self, path, transform):
        data = np.load(path)
        ct_scan = data["arr_0"]
        tensor = torch.tensor(ct_scan).float().unsqueeze(0)

        return tensor


    def tokenize_text(self, text):
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def __getitem__(self, index):
        npz_file, input_text = self.samples[index]
        video_tensor = self.npz_to_tensor(npz_file)
        input_text = input_text.replace('"', "").replace("'", "").replace("(", "").replace(")", "")
        tokenized_text = self.tokenize_text(input_text)

        return (
            video_tensor,
            tokenized_text["input_ids"].squeeze(0),  # Extract `input_ids` tensor
            tokenized_text["attention_mask"].squeeze(0),  # Extract `attention_mask` tensor
        )