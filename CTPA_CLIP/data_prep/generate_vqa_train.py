import os
import json
import pandas as pd

# Paths to dataset
image_dir = "/teamspace/studios/this_studio/CTPA-CLIP/data/train/train_PE"
report_csv = "/teamspace/studios/this_studio/CTPA-CLIP/data/all_reports.csv"
output_jsonl = "/teamspace/studios/this_studio/CTPA-CLIP/data/train_dataset.jsonl"

# Load reports
reports_df = pd.read_csv(report_csv)

def create_vqa_jsonl():
    records_count = 0
    missing_count = 0
    
    with open(output_jsonl, "w") as f:
        for _, row in reports_df.iterrows():
            impression_id = row["impression_id"]
            impression_text = row["impressions"].strip()

            # Locate Image File - ensure consistent path format with forward slashes
            image_folder = os.path.join(image_dir, f"train_{impression_id}")
            image_path = os.path.join(image_folder, f"{impression_id}.npz")
            
            # Normalize path to use forward slashes consistentl
            image_path = image_path.replace("\\", "/")
            
            if not os.path.exists(image_path):
                missing_count += 1
                continue  # Skip if image is missing
            
            json_record = {
                "image_id": impression_id,
                "image_path": image_path,
                "report": impression_text
            }
            f.write(json.dumps(json_record) + "\n")  # Write line-by-line
            records_count += 1
    
    return records_count, missing_count

records_count, missing_count = create_vqa_jsonl()
print(f"VQA dataset saved in JSONL format at {output_jsonl}")
print(f"Total records created: {records_count}")
print(f"Missing images skipped: {missing_count}")