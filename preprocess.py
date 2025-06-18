import pandas as pd
import re
import numpy as np

def take_column(df, column_name):
    """Extract a single column from the dataframe"""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe")
    return df[column_name]

def concat_columns(df, columns, separator=' '):
    """Concatenate multiple columns with a separator"""
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in dataframe")
    return df[columns].apply(lambda x: separator.join(x.astype(str)), axis=1)

def clean_text(text_series, remove_html=True, remove_urls=True, remove_special_chars=True, lowercase=True):
    """Clean text data with various options"""
    cleaned = text_series.copy()
    
    if remove_html:
        cleaned = cleaned.str.replace(r'<[^>]+>', '', regex=True)
    
    if remove_urls:
        cleaned = cleaned.str.replace(r'http\S+|www.\S+', '', regex=True)
    
    if remove_special_chars:
        cleaned = cleaned.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    
    if lowercase:
        cleaned = cleaned.str.lower()
    
    # Remove extra whitespace
    cleaned = cleaned.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return cleaned

def truncate_text(text_series, max_length=512):
    """Truncate text to maximum length"""
    return text_series.str.slice(0, max_length)

def filter_by_length(df, text_column, min_length=10, max_length=1000):
    """Filter dataframe by text length"""
    text_lengths = df[text_column].str.len()
    mask = (text_lengths >= min_length) & (text_lengths <= max_length)
    return df[mask].reset_index(drop=True)

def sample_data(df, n_samples=1000, random_state=42):
    """Sample n rows from the dataframe"""
    if len(df) <= n_samples:
        return df
    return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

def get_preprocessing_options():
    """Get available preprocessing options"""
    return {
        "take_column": {
            "description": "Use a single column as text",
            "params": ["column_name"]
        },
        "concat_columns": {
            "description": "Concatenate multiple columns",
            "params": ["columns", "separator"]
        },
        "clean_text": {
            "description": "Clean text (remove HTML, URLs, special chars, lowercase)",
            "params": ["remove_html", "remove_urls", "remove_special_chars", "lowercase"]
        },
        "truncate_text": {
            "description": "Truncate text to maximum length",
            "params": ["max_length"]
        },
        "filter_by_length": {
            "description": "Filter by text length",
            "params": ["text_column", "min_length", "max_length"]
        },
        "sample_data": {
            "description": "Randomly sample data",
            "params": ["n_samples", "random_state"]
        }
    }

def apply_preprocessing(df, preprocessing_steps):
    """Apply a series of preprocessing steps to the dataframe"""
    processed_df = df.copy()
    text_column = None
    
    for step in preprocessing_steps:
        step_name = step['name']
        params = step.get('params', {})
        
        if step_name == "take_column":
            text_column = params['column_name']
            processed_text = take_column(processed_df, text_column)
            # Create new dataframe with text column
            processed_df = pd.DataFrame({text_column: processed_text})
            
        elif step_name == "concat_columns":
            columns = params['columns']
            separator = params.get('separator', ' ')
            text_column = 'combined_text'
            processed_df[text_column] = concat_columns(processed_df, columns, separator)
            
        elif step_name == "clean_text":
            if text_column is None:
                raise ValueError("No text column defined. Use 'take_column' or 'concat_columns' first.")
            processed_df[text_column] = clean_text(
                processed_df[text_column],
                remove_html=params.get('remove_html', True),
                remove_urls=params.get('remove_urls', True),
                remove_special_chars=params.get('remove_special_chars', True),
                lowercase=params.get('lowercase', True)
            )
            
        elif step_name == "truncate_text":
            if text_column is None:
                raise ValueError("No text column defined. Use 'take_column' or 'concat_columns' first.")
            processed_df[text_column] = truncate_text(
                processed_df[text_column],
                max_length=params.get('max_length', 512)
            )
            
        elif step_name == "filter_by_length":
            target_column = params.get('text_column', text_column)
            if target_column is None:
                raise ValueError("No text column specified for filtering.")
            processed_df = filter_by_length(
                processed_df,
                target_column,
                min_length=params.get('min_length', 10),
                max_length=params.get('max_length', 1000)
            )
            
        elif step_name == "sample_data":
            processed_df = sample_data(
                processed_df,
                n_samples=params.get('n_samples', 1000),
                random_state=params.get('random_state', 42)
            )
    
    return processed_df, text_column