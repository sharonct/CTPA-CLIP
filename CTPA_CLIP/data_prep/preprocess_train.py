import os
import ast  # For safely evaluating string representations of lists
from typing import Any
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm

def read_nii_files(directory):
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                full_path = os.path.join(root, file)
                normalized_path = os.path.normpath(full_path).replace("\\", "/")
                nii_files.append(normalized_path)
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

def parse_xy_spacing(xy_spacing_value):
    """Parse the XYSpacing value from the CSV, which could be in different formats."""
    if isinstance(xy_spacing_value, list):
        return float(xy_spacing_value[0])
    elif isinstance(xy_spacing_value, (int, float)):
        return float(xy_spacing_value)
    elif isinstance(xy_spacing_value, str):
        try:
            # Safely evaluate the string representation of a list
            xy_list = ast.literal_eval(xy_spacing_value)
            if isinstance(xy_list, list) and len(xy_list) > 0:
                return float(xy_list[0])
        except (ValueError, SyntaxError):
            pass
        # Try direct conversion if it's a simple numeric string
        try:
            return float(xy_spacing_value)
        except ValueError:
            pass
    
    raise ValueError(f"Could not parse XYSpacing value: {xy_spacing_value}")

# Pass the dataframe as an argument to the function
def process_file(args):
    file_path, df = args
    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return

    file_name = os.path.basename(file_path)

    row = df[df['VolumeName'] == file_name]
    if row.empty:
        print(f"No metadata found for {file_name}. Skipping.")
        return
        
    slope = float(row["RescaleSlope"].iloc[0])
    intercept = float(row["RescaleIntercept"].iloc[0])
    
    try:
        xy_spacing = parse_xy_spacing(row["XYSpacing"].iloc[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
    except Exception as e:
        print(f"Error parsing spacing for {file_name}: {e}")
        return

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

    save_folder = "/teamspace/studios/this_studio/CTPA-CLIP/data/train"
    file_name_no_ext = file_name.split(".")[0]  
    subfolder = "train_" + file_name_no_ext[:2]
    subsubfolder = "train_" + file_name_no_ext 
    folder_path_new = os.path.join(save_folder, subfolder, subsubfolder)
    os.makedirs(folder_path_new, exist_ok=True)
    
    output_file_name = file_name_no_ext + ".npz"
    save_path = os.path.join(folder_path_new, output_file_name)
    np.savez(save_path, resized_array)
    
    del img_data, tensor, resized_array
    
    try:
        os.remove(file_path)
        print(f"Successfully processed {file_name} → {output_file_name} and deleted original file")
    except Exception as e:
        print(f"Processed {file_name} → {output_file_name} but failed to delete original file: {e}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # For Windows compatibility

    # Load the dataframe once
    train_df = pd.read_csv("/teamspace/studios/this_studio/CTPA-CLIP/data/train_metadata.csv")

    train_ctpa = '/teamspace/studios/this_studio/CTPA-CLIP/data/inspect'
    nii_files = read_nii_files(train_ctpa)

    # Make sure we're processing files that exist in the metadata
    valid_files = []
    for file_path in nii_files:
        file_name = os.path.basename(file_path)
        if file_name in train_df['VolumeName'].values:
            valid_files.append(file_path)
        else:
            print(f"Warning: {file_name} not found in metadata. Skipping.")
    
    if not valid_files:
        print("No valid files found to process. Check the path and metadata.")
        exit(1)
    
    print(f"Found {len(valid_files)} valid files to process.")
    print("WARNING: Original files will be deleted after processing to save space!")
    
    # Optional confirmation prompt
    confirmation = input("Continue with processing and deletion? (y/n): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        exit(0)
    
    # Create argument tuples with both the file path and the dataframe
    args = [(file_path, train_df) for file_path in valid_files]

    num_workers = os.cpu_count() // 2 or 1  # Use half the available cores or at least 1
    print(f"Using {num_workers} workers for processing.")

    # Process files using multiprocessing with tqdm progress bar
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_file, args), total=len(args)))
        
    print("Processing complete. Original files have been deleted to save space.")
