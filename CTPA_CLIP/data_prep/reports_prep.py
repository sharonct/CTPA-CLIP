import pandas as pd
import re
import numpy as np

def preprocess_impressions(text):
    if not isinstance(text, str) or text.strip() == "":
        return None  # Return None for empty or invalid text
    
    # Extract impressions section
    impressions = re.split(r'IMPRESSION:\s*', text, flags=re.IGNORECASE)[1:]

    cleaned_impressions = []
    
    for imp in impressions:
        # Remove "END OF IMPRESSION" and summary markers
        imp = re.sub(r'END OF IMPRESSION:.*', '', imp, flags=re.IGNORECASE)
        imp = re.sub(r'SUMMARY[:\d-]*\s*', '', imp, flags=re.IGNORECASE)

        # General cleanup
        imp = imp.strip().lower()
        imp = re.sub(r'\b\d+\.\s*', '', imp)  # Remove numbered points
        imp = re.sub(r'<hcw>', '', imp)  # Remove placeholders like <hcw>
        imp = re.sub(r'\s+', ' ', imp)  # Normalize whitespace
        imp = re.sub(r'(\s,)+', '', imp)  # Remove extra commas
        imp = re.sub(r'\s+\.', '.', imp)  # Fix spacing before periods
        imp = re.sub(r'\b\d+\b(?!\s(months|mm))', '', imp)  # Remove standalone numbers except months/mm

        
        discussion_patterns = [
            r'preliminary findings provided by.*? at .*? to .*?\.', 
            r'changes to the final report regarding impression #-? were added to the final report and reported to .*? on .*? at .*?\.', 
            r'this (was|is) (an? )?(on-call|non-called)?\s*(case|study)?\s*(and )?was discussed with .*? at .*? by .*?\.', 
            r'(preliminary|final)?\s*report (was )?discussed with .*?(by phone|via telephone)? on .*? at .*?\.', 
            r'this (finding|case)? was discussed with .*?(by phone|via telephone)? at .*? on .*?\.', 
            r'(additional|preliminary)?\s*finding[s]? (was|were)? discussed with .*?(by phone|via telephone)? at .*? on .*?\.', 
            r'discussed above findings with .*?(by phone|via telephone)? on .*? at .*?\.', 
            r'the possibility of .*? was discussed with .*?(by phone|via telephone)? on .*? at .*?\.', 
            r'the (final|preliminary)? interpretation (was )?discussed with .*?(by phone|via telephone)? on .*? at .*?\.', 
            r'finding #?\d* (was )?discussed with .*? at approximately .*?\.', 
            r'this finding was discussed with .*? in the emergency department at the time of the examination\.',  
            r'please note this was an? (on-call|non-called) case and was discussed with .*? at pager .*? on .*? at .*? by .*?\.',  
            r'preliminary report was discussed with .*? by the on-call resident on .*? at approximately .*?\.',  
            r'this was discussed with .*? at .*? on .*?\.',  
            r'discussed findings with .*? by telephone on .*? at .*?\.',  
            r'the final interpretation was discussed with .*? by telephone on .*? at .*?\.',  
            r'finding #?\d* was discussed with .*? at approximately .*?\.',  
            r'finding #?\d* was discussed with .*? of obstetrics and gynecology at .*? on .*?\.',  
            r'preliminary findings discussed with .*? at .*? on .*?\.',  
            r'this is an? non-called case and was discussed with .*? at .*? on .*? by .*?\.',  
            r'additional finding of .*? was discussed with .*? at .*? on .*?\.',  
            r'this was discussed with .*? via telephone on .*? at .*?\.',  
            r'this finding was discussed with .*? in the emergency department at the time of the examination\.',  
            r'preliminary report findings were communicated to .*? at .*? on .*?\.',  
            r'changes to the final report were added and communicated to .*? at .*? on .*?\.',  
            r'discussion with .*? regarding .*? occurred at .*? on .*?\.',  
            r'notification of .*? was provided to .*? at .*? on .*?\.',  
            r'phone call made to .*? at .*? on .*?\.',  
            r'patient findings were reviewed with .*? at .*? on .*?\.',  
            r'case was escalated to .*? and discussed at .*? on .*?\.',  
            r'final report was verified and communicated to .*? at .*? on .*?\.',  
            r'communication regarding this case took place with .*? at .*? on .*?\.',  
            r'this case was reviewed and discussed with .*? at .*? on .*?\.',  
            r'findings conveyed to .*? at .*? on .*?\.',  
            r'radiology consultation with .*? was conducted at .*? on .*?\.',  
            r'phone discussion occurred with .*? at .*? on .*?\.',  
            r'consultation summary sent to .*? at .*? on .*?\.',  
            r'follow-up discussion with .*? occurred at .*? on .*?\.',  
            r'urgent findings were relayed to .*? at .*? on .*?\.'
        ]

        for pattern in discussion_patterns:
            imp = re.sub(pattern, '', imp, flags=re.IGNORECASE)

        imp = re.sub(r'<time>', '', imp)
        imp = re.sub(r'<date>', '', imp)
        
        # Extra cleanup for leftover spaces
        imp = re.sub(r'\s+', ' ', imp).strip()

        if imp:
            cleaned_impressions.append(imp)

    final_text = " ".join(cleaned_impressions).strip()
    
    return final_text if final_text else None  


reports = pd.read_csv('C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/Final_Impressions.csv')

reports['impressions'] = reports['impressions'].apply(preprocess_impressions)
reports['impressions'] = reports['impressions'].replace("", np.nan)
reports.dropna(subset=['impressions'], inplace=True)
reports.to_csv("C:/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/data/all_reports.csv")
