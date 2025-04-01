import pandas as pd
import re
import numpy as np

def preprocess_impressions(text):
    if not isinstance(text, str) or text.strip() == "":
        return None  # Return None for empty or invalid text
    
    impressions = re.split(r'IMPRESSION:\s*', text, flags=re.IGNORECASE)[1:]

    cleaned_impressions = []
    
    for imp in impressions:
        imp = re.sub(r'END OF IMPRESSION:.*', '', imp, flags=re.IGNORECASE)
        imp = re.sub(r'SUMMARY[:\d-]*\s*', '', imp, flags=re.IGNORECASE)
        imp = imp.strip().lower()
        imp = re.sub(r'\b\d+\.\s*', '', imp)
        imp = re.sub(r'<hcw>', '', imp)
        imp = re.sub(r'\s+', ' ', imp)
        imp = re.sub(r'(\s,)+', '', imp)
        imp = re.sub(r'\s+\.', '.', imp)
        imp = re.sub(r'\b\d+\b(?!\s(months|mm))', '', imp)
        
        if imp:
            cleaned_impressions.append(imp)

    final_text = " ".join(cleaned_impressions).strip()
    
    return final_text if final_text else None  # Return None if empty

reports = pd.read_csv('/teamspace/studios/this_studio/data/Final_Impressions.csv')

reports['impressions'] = reports['impressions'].apply(preprocess_impressions)
reports['impressions'] = reports['impressions'].replace("", np.nan)
reports.dropna(subset=['impressions'], inplace=True)
