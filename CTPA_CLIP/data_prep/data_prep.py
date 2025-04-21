import os
import pandas as pd
import numpy as np
import nibabel as nib

def extract_nii_metadata(directory):
    data = []

    for file in os.listdir(directory):
        if file.endswith(".nii") or file.endswith(".nii.gz"):  # Handles compressed .nii files too
            file_path = os.path.join(directory, file)

            try:
                img = nib.load(file_path)
                header = img.header

                # Extract metadata with NaN handling
                slope = header["scl_slope"]
                intercept = header["scl_inter"]

                if np.isnan(slope):
                    slope = 1.0
                if np.isnan(intercept):
                    intercept = 0.0

                # Extract voxel spacing (X, Y, Z)
                voxel_spacing = header["pixdim"][1:4] 
                xy_spacing = [float(v) for v in voxel_spacing[:2]]
                z_spacing = voxel_spacing[2]  # Z 

                # Append to list
                data.append([file, slope, intercept, xy_spacing, z_spacing])

            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(data, columns=["VolumeName", "RescaleSlope", "RescaleIntercept", "XYSpacing", "ZSpacing"])
    
    return df

ct_metadata = extract_nii_metadata('C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/inspect')

split_idx = int(len(ct_metadata) * 0.8)

train_df = ct_metadata
test_df = ct_metadata.iloc[split_idx:]

print(len(train_df))
print(len(test_df))
train_df.to_csv("C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/train_metadata.csv", index=False)
test_df.to_csv("C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/test_metadata.csv", index=False)
