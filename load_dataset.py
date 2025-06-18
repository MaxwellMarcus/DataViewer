import datasets
import pandas as pd
import json
import os
from urllib.parse import urlparse
import requests

def load_dataset_hf(dataset_name, split=None, streaming=False, config_name=None):
    """Load dataset from Hugging Face Hub"""
    try:
        if split:
            dataset = datasets.load_dataset(dataset_name, config_name, split=split, streaming=streaming)
        else:
            dataset = datasets.load_dataset(dataset_name, config_name, streaming=streaming)
        
        # Convert to pandas if not streaming
        if not streaming:
            if isinstance(dataset, datasets.DatasetDict):
                # Return the first split as default
                first_split = list(dataset.keys())[0]
                return dataset[first_split].to_pandas()
            else:
                return dataset.to_pandas()
        
        return dataset
    except Exception as e:
        print(f"Error loading HuggingFace dataset '{dataset_name}': {e}")
        return None

def load_dataset_as_pandas(file_path):
    """Load local dataset file as pandas DataFrame"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
            if file_path.endswith('.jsonl'):
                # Handle JSONL files
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                return pd.DataFrame(data)
            else:
                return pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        elif file_path.endswith('.tsv'):
            return pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Error loading local dataset '{file_path}': {e}")
        return None

def load_dataset_as_pandas_from_url(url):
    """Load dataset from URL as pandas DataFrame"""
    try:
        # Parse URL to determine file type
        parsed_url = urlparse(url)
        file_extension = os.path.splitext(parsed_url.path)[1].lower()
        
        if file_extension == '.csv':
            return pd.read_csv(url)
        elif file_extension in ['.json', '.jsonl']:
            response = requests.get(url)
            response.raise_for_status()
            
            if file_extension == '.jsonl':
                data = []
                for line in response.text.strip().split('\n'):
                    if line.strip():
                        data.append(json.loads(line))
                return pd.DataFrame(data)
            else:
                return pd.read_json(url)
        elif file_extension == '.parquet':
            return pd.read_parquet(url)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(url)
        elif file_extension == '.tsv':
            return pd.read_csv(url, sep='\t')
        else:
            # Try to read as CSV by default
            return pd.read_csv(url)
    except Exception as e:
        print(f"Error loading dataset from URL '{url}': {e}")
        return None

def load_dataset_as_pandas_from_file(file_path):
    """Load dataset from local file (alias for load_dataset_as_pandas)"""
    return load_dataset_as_pandas(file_path)

def get_dataset_info(source_type, source_path):
    """Get information about a dataset without fully loading it"""
    try:
        if source_type == "huggingface":
            # Get dataset info from HuggingFace
            try:
                dataset_info = datasets.get_dataset_info(source_path)
                return {
                    "description": dataset_info.description,
                    "features": list(dataset_info.features.keys()) if dataset_info.features else [],
                    "splits": list(dataset_info.splits.keys()) if dataset_info.splits else [],
                    "size": sum(split.num_examples for split in dataset_info.splits.values()) if dataset_info.splits else 0
                }
            except:
                return {"error": "Could not fetch dataset info"}
        
        elif source_type in ["file", "url"]:
            # For files, we need to load a sample to get info
            if source_type == "file":
                df = load_dataset_as_pandas(source_path)
            else:
                df = load_dataset_as_pandas_from_url(source_path)
            
            if df is not None:
                return {
                    "columns": list(df.columns),
                    "shape": df.shape,
                    "dtypes": df.dtypes.to_dict(),
                    "sample": df.head(3).to_dict('records')
                }
            else:
                return {"error": "Could not load dataset"}
                
    except Exception as e:
        return {"error": str(e)}

def get_popular_datasets():
    """Get list of popular datasets for quick selection"""
    return {
        "Text Classification": [
            "imdb",
            "sentiment140", 
            "amazon_reviews_multi",
            "yelp_review_full"
        ],
        "Question Answering": [
            "squad",
            "squad_v2",
            "natural_questions"
        ],
        "Text Generation": [
            "wikitext",
            "openwebtext",
            "bookcorpus"
        ],
        "Summarization": [
            "cnn_dailymail",
            "xsum",
            "multi_news"
        ],
        "General Text": [
            "wikipedia",
            "common_crawl",
            "oscar"
        ]
    }

def validate_source(source_type, source_path):
    """Validate if a data source is accessible"""
    try:
        if source_type == "huggingface":
            # Try to get dataset info
            datasets.get_dataset_info(source_path)
            return True, "Dataset found on Hugging Face"
        
        elif source_type == "file":
            if os.path.exists(source_path):
                return True, "File exists"
            else:
                return False, "File not found"
        
        elif source_type == "url":
            response = requests.head(source_path, timeout=10)
            if response.status_code == 200:
                return True, "URL accessible"
            else:
                return False, f"URL returned status code {response.status_code}"
                
    except Exception as e:
        return False, str(e)

def load_dataset_smart(source_type, source_path, **kwargs):
    """Smart dataset loader that determines the best loading method"""
    try:
        if source_type == "huggingface":
            return load_dataset_hf(source_path, **kwargs)
        elif source_type == "file":
            return load_dataset_as_pandas(source_path)
        elif source_type == "url":
            return load_dataset_as_pandas_from_url(source_path)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    except Exception as e:
        print(f"Error in smart dataset loader: {e}")
        return None