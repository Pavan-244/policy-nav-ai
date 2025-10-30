import pandas as pd
import re

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess text for quantum NLP / TF-IDF:
    - combine 'title' + 'full_text'
    - lowercase
    - remove non-alphanumeric characters
    - store in 'text_for_nlp' column
    """
    df = df.copy()
    
    def clean_text(x):
        if not isinstance(x, str):
            x = str(x)
        x = x.lower()
        x = re.sub(r"[^a-z0-9\s]", "", x)
        return x
    
    df["text_for_nlp"] = (df.get("title", "") + " " + df.get("full_text", "")).apply(clean_text)
    
    return df
