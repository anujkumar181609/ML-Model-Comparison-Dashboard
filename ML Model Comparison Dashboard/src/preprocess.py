import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df= df.copy()
    
    #drop duplicates
    df= df.drop_duplicates()
    
    #to remove likely ID columns (unique for each row)
    id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    df = df.drop(columns=id_cols)
    
            
    #to deal with conversions    
    for col in df.columns:
    
        # skip already numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
    
        temp = df[col].astype(str).str.replace(',', '')
        converted = pd.to_numeric(temp, errors='coerce')
    
        # calculate how much data successfully converted
        valid_ratio = converted.notna().mean()
    
        # if mostly numeric → convert
        if valid_ratio > 0.8:
            df[col] = converted
        # else keep as categorical (will encode later)
            
            
            
    #handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
        
                
                
    #encode categorical 
    le= LabelEncoder()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = le.fit_transform(df[col])
    return df

