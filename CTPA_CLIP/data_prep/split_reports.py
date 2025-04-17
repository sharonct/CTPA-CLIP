import os
import pandas as pd

reports = pd.read_csv("C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/all_reports.csv")
directory = "C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/inspect"
file_names = [os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

split_idx = int(len(file_names) * 0.8)

train_files = file_names[:split_idx]
test_files = file_names[split_idx:]

all_reports = reports['impression_id'].values.tolist()

train_ids = [item for item in all_reports if f"{item}.nii" in train_files]
test_ids = [item for item in all_reports if f"{item}.nii" in test_files]

train_reports = reports[reports['impression_id'].isin(train_ids)]
test_reports = reports[reports['impression_id'].isin(test_ids)]

# Save train and test reports as CSV
train_reports.to_csv("C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/train_reports.csv", index=False)
test_reports.to_csv("C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/test_reports.csv", index=False)