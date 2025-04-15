import os
import pandas as pd

reports = pd.read_csv("/teamspace/studios/this_studio/CTPA-CLIP/data/all_reports.csv")
directory = "/teamspace/studios/this_studio/inspect/inspect2/CTPA"
file_names = [os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

split_idx = int(len(file_names) * 0.8)

train_files = file_names[:split_idx]
test_files = file_names[split_idx:]

all_reports = reports['impression_id'].values.tolist()
print(all_reports[-5:])
print(train_files[-5:])

print("PE4524dcf" in train_files)


train_ids = [item for item in all_reports if item in train_files]
test_ids = [item for item in all_reports if item in test_files]
print(len(train_ids))
# print(file_names)
print(len(train_files))

# train = [item for item in all_reports if item in file_names]
# reports_train = reports[reports['impression_id'].isin(train)]

train_reports = reports[reports['impression_id'].isin(train_ids)]
test_reports = reports[reports['impression_id'].isin(test_ids)]

# Save train and test reports as CSV
train_reports.to_csv("/teamspace/studios/this_studio/CTPA-CLIP/data/train_ctpa_reports.csv", index=False)
test_reports.to_csv("/teamspace/studios/this_studio/CTPA-CLIP/data/test_ctpa_reports.csv", index=False)

# reports_train.to_csv("/teamspace/studios/this_studio/data/reports_train.csv", index=False)
