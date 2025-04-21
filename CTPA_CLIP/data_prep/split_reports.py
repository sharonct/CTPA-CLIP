import os
import pandas as pd

reports = pd.read_csv("C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/all_reports.csv")
directory = "C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/inspect"

all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

file_bases = [os.path.splitext(f)[0] for f in all_files]
file_bases = [os.path.splitext(base)[0] if base.endswith('.nii') else base for base in file_bases]

split_idx = int(len(file_bases) * 0.8)
train_file_bases = file_bases[:split_idx]
test_file_bases = file_bases[split_idx:]

train_ids = []
test_ids = []

for impression_id in reports['impression_id'].values:
    if impression_id in train_file_bases:
        train_ids.append(impression_id)
    elif impression_id in test_file_bases:
        test_ids.append(impression_id)

train_reports = reports[reports['impression_id'].isin(train_ids)]
test_reports = reports[reports['impression_id'].isin(test_ids)]

print(f"Train reports: {len(train_reports)}")
print(f"Test reports: {len(test_reports)}")

train_reports.to_csv("C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/train_reports.csv", index=False)
test_reports.to_csv("C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/test_reports.csv", index=False)