import os
from typing import Any
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm

from data_prep import train_df


def read_nii_files(directory):
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii'):
                nii_files.append(os.path.join(root, file))
    return nii_files

def read_nii_data(file_path):
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def resize_array(array, current_spacing, target_spacing):
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

def process_file(file_path):
    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return

    file_name = os.path.basename(file_path)

    row = df[df['VolumeName'] == file_name]
    slope = float(row["RescaleSlope"].iloc[0])
    intercept = float(row["RescaleIntercept"].iloc[0])
    xy_spacing = float(row["XYSpacing"].iloc[0][0])
    z_spacing = float(row["ZSpacing"].iloc[0])


    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = ((img_data / 1000)).astype(np.float32)

    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    resized_array = resize_array(tensor, current, target)
    resized_array = resized_array[0][0]

    save_folder = "/teamspace/studios/this_studio/data/train_df2_preprocessed/"
    file_name_no_ext = file_name.split(".")[0]  
    subfolder = "train_" + file_name_no_ext[:2]
    subsubfolder = "train_" + file_name_no_ext 
    folder_path_new = os.path.join(save_folder, subfolder, subsubfolder)
    os.makedirs(folder_path_new, exist_ok=True)
    
    file_name = file_name.split(".")[0]+".npz"
    save_path = os.path.join(folder_path_new, file_name)
    np.savez(save_path, resized_array)


train_ctpa = '/teamspace/studios/this_studio/test'
nii_files = read_nii_files(train_ctpa)

all_files = sorted(os.listdir(train_ctpa))
split_idx = int(len(all_files) * 0.8)

nii_files = nii_files[:split_idx]

df: Any = train_df

num_workers = 2

# Process files using multiprocessing with tqdm progress bar
with Pool(num_workers) as pool:
    list(tqdm(pool.imap(process_file, nii_files), total=len(nii_files)))
