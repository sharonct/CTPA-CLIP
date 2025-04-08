import os
import json
import pandas as pd

# Paths to dataset
image_dir = "/teamspace/studios/this_studio/data/train/train_PE"
report_csv = "/teamspace/studios/this_studio/data/reports_train.csv"
output_jsonl = "/teamspace/studios/this_studio/data/train_vqa_dataset.jsonl"

# Load reports
reports_df = pd.read_csv(report_csv)

def generate_questions():
    return [
        "What findings do you observe in this CT scan?",
        "Could you summarize the observations from this CT scan?",
        "What abnormalities are present in this CT scan?",
        "How would you interpret the results of this CT scan?"
    ]

def create_vqa_jsonl():
    with open(output_jsonl, "w") as f:
        for _, row in reports_df.iterrows():
            impression_id = row["impression_id"]
            impression_text = row["impressions"].strip()

            # Locate Image File
            image_folder = os.path.join(image_dir, f"train_{impression_id}")
            image_path = os.path.join(image_folder, f"{impression_id}.npz")
            
            if not os.path.exists(image_path):
                continue  # Skip if image is missing
            
            # Generate QA pairs
            for question in generate_questions():
                json_record = {
                    "image_id": impression_id,
                    "image_path": image_path,
                    "question": question,
                    "answer": impression_text
                }
                f.write(json.dumps(json_record) + "\n")  # Write line-by-line

create_vqa_jsonl()
print(f"VQA dataset saved in JSONL format at {output_jsonl}")
