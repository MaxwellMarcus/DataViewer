import numpy as np
import os
import torch
from PIL import Image

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

def run_BERT(input_texts, model_name="bert-base-uncased"):
    """Run BERT model to generate embeddings"""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library not available. Install with: pip install transformers")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        embeddings = []
        for text in input_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)
    except Exception as e:
        print(f"Error running BERT: {e}")
        return None

def run_CLIP(input_texts, model_name="ViT-B/32"):
    """Run CLIP model to generate text embeddings"""
    if not HAS_CLIP:
        raise ImportError("CLIP library not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device=device)
        
        embeddings = []
        for text in input_texts:
            text_tokens = clip.tokenize([text]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                embedding = text_features.cpu().numpy().squeeze()
                embeddings.append(embedding)
        
        return np.array(embeddings)
    except Exception as e:
        print(f"Error running CLIP: {e}")
        return None

def run_GEMMA(input_texts, model_name="google/gemma-2b"):
    """Run Gemma model to generate embeddings"""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library not available. Install with: pip install transformers")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        embeddings = []
        for text in input_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)
    except Exception as e:
        print(f"Error running Gemma: {e}")
        return None

def run_OpenAI_Embeddings(input_texts, model_name="text-embedding-ada-002"):
    """Run OpenAI embeddings API"""
    if not HAS_OPENAI:
        raise ImportError("openai library not available. Install with: pip install openai")
    
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not found in environment variables")
            return None
        
        client = openai.OpenAI()
        embeddings = []
        
        for text in input_texts:
            response = client.embeddings.create(
                input=text,
                model=model_name
            )
            embedding = np.array(response.data[0].embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    except Exception as e:
        print(f"Error running OpenAI embeddings: {e}")
        return None

def run_Sentence_Transformer(input_texts, model_name="all-MiniLM-L6-v2"):
    """Run Sentence Transformer model to generate embeddings"""
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers library not available. Install with: pip install sentence-transformers")
    
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(input_texts)
        return embeddings
    except Exception as e:
        print(f"Error running Sentence Transformer: {e}")
        return None

def get_available_models():
    """Get list of available models with descriptions"""
    models = {}
    
    if HAS_SENTENCE_TRANSFORMERS:
        models["Sentence Transformer"] = {
            "models": [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "all-MiniLM-L12-v2",
                "multi-qa-mpnet-base-dot-v1",
                "paraphrase-multilingual-MiniLM-L12-v2"
            ],
            "description": "Fast and efficient sentence embeddings"
        }
    
    if HAS_TRANSFORMERS:
        models["BERT"] = {
            "models": [
                "bert-base-uncased",
                "bert-large-uncased", 
                "distilbert-base-uncased",
                "roberta-base"
            ],
            "description": "Bidirectional transformer embeddings"
        }
        
        models["Gemma"] = {
            "models": [
                "google/gemma-2b",
                "google/gemma-7b"
            ],
            "description": "Google's Gemma language model embeddings"
        }
    
    if HAS_CLIP:
        models["CLIP"] = {
            "models": [
                "ViT-B/32",
                "ViT-B/16", 
                "ViT-L/14"
            ],
            "description": "Vision-language model for text embeddings"
        }
    
    if HAS_OPENAI:
        models["OpenAI"] = {
            "models": [
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large"
            ],
            "description": "High-quality OpenAI embeddings (requires API key)"
        }
    
    # Fallback - always provide at least one option
    if not models:
        models["Simple"] = {
            "models": ["random-embeddings"],
            "description": "Fallback random embeddings (for testing)"
        }
    
    return models

def run_model(model_type, model_name, input_texts):
    """Run the specified model on input texts"""
    if model_type == "Sentence Transformer":
        return run_Sentence_Transformer(input_texts, model_name)
    elif model_type == "BERT":
        return run_BERT(input_texts, model_name)
    elif model_type == "CLIP":
        return run_CLIP(input_texts, model_name)
    elif model_type == "OpenAI":
        return run_OpenAI_Embeddings(input_texts, model_name)
    elif model_type == "Gemma":
        return run_GEMMA(input_texts, model_name)
    elif model_type == "Simple":
        # Fallback for testing
        return np.random.rand(len(input_texts), 384)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

