# Enhanced File Selector Features

The enhanced file selector provides two modes for setting up your data visualization:

## Mode 1: Load Existing Files (Original)
- Select pre-computed embeddings (.npy files)
- Select metadata files (.csv, .json, .parquet, .xlsx)
- Choose primary text column for display
- Works exactly like the original file selector

## Mode 2: Process Dataset with AI Models (New)
This mode allows you to process raw datasets and generate embeddings on-the-fly.

### Supported Data Sources
1. **Hugging Face Datasets**: Load datasets directly from the HF Hub
   - Examples: `imdb`, `squad`, `wikitext`, `amazon_reviews_multi`
   - Automatically handles dataset splits and formats

2. **URLs**: Load datasets from web URLs
   - Supports: CSV, JSON, JSONL, Parquet, Excel files
   - Example: `https://example.com/dataset.csv`

3. **Local Files**: Load datasets from your computer
   - Supports: CSV, JSON, JSONL, Parquet, Excel, TSV files

### Available AI Models

#### Sentence Transformers (Recommended)
- `all-MiniLM-L6-v2` - Fast, lightweight, good quality
- `all-mpnet-base-v2` - Higher quality, slower
- `multi-qa-mpnet-base-dot-v1` - Optimized for Q&A
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

#### BERT Family
- `bert-base-uncased` - Standard BERT
- `bert-large-uncased` - Larger BERT model
- `distilbert-base-uncased` - Faster, smaller BERT
- `roberta-base` - RoBERTa variant

#### OpenAI Models (API Key Required)
- `text-embedding-ada-002` - High quality, cost-effective
- `text-embedding-3-small` - Latest small model
- `text-embedding-3-large` - Latest large model

#### CLIP Models
- `ViT-B/32` - Vision-language model for text
- `ViT-B/16` - Higher resolution
- `ViT-L/14` - Largest CLIP model

#### Google Gemma
- `google/gemma-2b` - Smaller Gemma model
- `google/gemma-7b` - Larger Gemma model

### Processing Options
- **Text Column**: Specify which column contains the text to embed
- **Max Samples**: Limit the number of samples (default: 1000)
- **Clean Text**: Remove HTML, URLs, special characters, convert to lowercase
- **Truncate Text**: Limit text length to 512 characters

### Example Workflows

#### 1. Process IMDb Movie Reviews
1. Select "Process Dataset with AI Model"
2. Data Source: "Hugging Face Dataset"
3. Path: `imdb`
4. Text Column: `text`
5. Model: Sentence Transformer → `all-MiniLM-L6-v2`
6. Click "Process & Generate"

#### 2. Process Custom CSV from URL
1. Select "Process Dataset with AI Model"
2. Data Source: "URL"
3. Path: `https://example.com/my_data.csv`
4. Text Column: `content`
5. Model: OpenAI → `text-embedding-ada-002` (requires API key)
6. Click "Process & Generate"

#### 3. Process Local File
1. Select "Process Dataset with AI Model"
2. Data Source: "Local File"
3. Path: `/path/to/my_dataset.json`
4. Text Column: `description`
5. Model: BERT → `bert-base-uncased`
6. Click "Process & Generate"

## Installation

Install additional dependencies:
```bash
pip install -r requirements_enhanced.txt
```

For OpenAI models, set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Output Files

When using Mode 2, the system generates:
- `generated_embeddings.npy` - The computed embeddings
- `generated_metadata.csv` - Original data plus processed text

These files are automatically loaded into the main visualization application.

## Performance Tips

1. **Start Small**: Use max samples of 1000-5000 for initial exploration
2. **Model Choice**: Sentence Transformers are fastest for most use cases
3. **Text Cleaning**: Enable for noisy web data, disable for clean text
4. **GPU**: Models will automatically use GPU if available (CUDA)
5. **Memory**: Large models may require 8GB+ RAM

## Troubleshooting

### Common Issues
- **Out of Memory**: Reduce max samples or use a smaller model
- **Slow Processing**: Try a Sentence Transformer model instead of BERT/CLIP
- **Dataset Not Found**: Check HuggingFace dataset name spelling
- **URL Errors**: Ensure the URL is accessible and points to a data file
- **API Errors**: Verify OpenAI API key is set correctly

### Model-Specific Notes
- **OpenAI**: Requires internet connection and valid API key
- **CLIP**: Requires additional installation of CLIP library
- **Gemma**: May require special permissions for some model variants
- **Large Models**: BERT-large and similar may be slow without GPU

## Testing

Test the enhanced selector with:
```bash
python test_enhanced_selector.py
```

## 🎯 **NEW FEATURES ADDED**

### ⚡ **Batch Size Selection**
- **Feature**: Configure processing batch size for AI model inference
- **Range**: 10-1000 samples per batch (default: 100)
- **Benefits**: 
  - Memory optimization for large datasets
  - Prevents out-of-memory errors
  - Progress tracking for long-running jobs
  - Resume capability if processing fails
- **Usage**: Set batch size in "Dataset Processing" mode before running models

### 📋 **Column Selection & Filtering**
- **Feature**: Choose which metadata columns to save
- **Benefits**:
  - Reduce file sizes significantly
  - Focus on relevant data only
  - Faster loading and visualization
  - Better memory usage
- **UI**: Interactive checkboxes with column type indicators (text/numeric/boolean)
- **Controls**: "Select All" and "Clear Selection" buttons

### 🤗 **Hugging Face Authentication**
- **Feature**: Login with HF API token for private access
- **Benefits**:
  - Access private datasets and models
  - Use gated models (e.g., Llama, Gemma)
  - Higher rate limits for API calls
  - Access to community contributions
- **UI**: Secure token input with authentication status
- **Auto-detection**: Reads tokens from environment variables (`HUGGINGFACE_HUB_TOKEN`, `HF_TOKEN`)

### 🔧 **Custom Model Weights**
- **Feature**: Use custom fine-tuned models from any HF repository
- **Benefits**:
  - Leverage domain-specific fine-tuned models
  - Use your own trained models
  - Access community fine-tuned models
  - Better performance for specific tasks
- **UI**: Repository validation with model info display
- **Format**: `username/model-name` or organization repositories

## 🚀 **Enhanced File Selector Overview**

The enhanced file selector now provides two distinct modes:

### Mode 1: Load Existing Files
- Load pre-computed embeddings (`.npy`) and metadata (`.csv`, `.json`, `.parquet`, `.xlsx`)
- Automatic data validation and consistency checking
- Primary column selection for visualization

### Mode 2: Process Dataset with AI Models
- **Data Sources**:
  - 🤗 Hugging Face datasets (e.g., `imdb`, `squad`, `tweet_eval`)
  - 🌐 URLs (CSV, JSON, direct downloads)
  - 📁 Local files (multiple formats supported)

- **AI Models** (5 types with fallbacks):
  - **Sentence Transformers**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, etc.
  - **BERT**: `bert-base-uncased`, `distilbert-base-uncased`, etc.
  - **CLIP**: `ViT-B/32`, `ViT-B/16`, etc.
  - **Gemma**: `google/gemma-2b`, `google/gemma-7b`
  - **OpenAI**: `text-embedding-ada-002`, `text-embedding-3-small` (requires API key)
  - **Simple**: Random embeddings fallback

## 🔧 **Processing Pipeline Features**

### Advanced Text Preprocessing
- **Column Selection**: Choose which text column to process
- **Text Cleaning**: Remove special characters, normalize whitespace
- **Text Truncation**: Limit text length (default: 512 tokens)
- **Sampling**: Limit dataset size (default: 1000 samples)

### **NEW: Batch Processing**
- Configurable batch sizes for model inference
- Real-time progress tracking with batch-level updates
- Error handling and recovery for failed batches
- Memory-efficient processing for large datasets

### **NEW: Smart Column Management**
- Preview dataset to see all available columns
- Interactive column selection with type detection
- Memory optimization by filtering early in pipeline
- Only selected columns saved in final metadata

## 📊 **Generated Output**

### Files Created
- `generated_embeddings.npy`: NumPy array of embeddings
- `generated_metadata.csv`: **Filtered metadata with only selected columns**

### **NEW: Enhanced Metadata**
- Only user-selected columns included
- Automatic addition of `processed_text` column
- Optimized file sizes
- Faster loading in visualization

## 🎮 **Usage Workflow**

### Basic Workflow
1. **Launch**: `python app_dearpygui.py`
2. **Select Mode**: Choose "Load Files" or "Process Dataset"
3. **Configure**: Set data source, model, and processing options
4. **NEW: Set Batch Size**: Choose appropriate batch size for your system
5. **NEW: Preview & Select Columns**: Load preview and select relevant columns
6. **Process**: Generate embeddings with progress tracking
7. **Visualize**: Automatic transition to main visualization interface

### **NEW: Column Selection Workflow**
1. Enter data source and click "Preview Dataset"
2. Review available columns with type indicators
3. Select/deselect columns using checkboxes
4. Use "Select All" or "Clear Selection" as needed
5. Verify selection before processing

### **NEW: Batch Size Optimization**
- **Small datasets** (<1K samples): Use default 100
- **Medium datasets** (1K-10K): Use 50-200 based on available memory
- **Large datasets** (>10K): Use 10-50 for memory-constrained systems
- **High-end systems**: Can use up to 1000 for maximum speed

### **NEW: Hugging Face Authentication Workflow**
1. Get your token from https://huggingface.co/settings/tokens
2. Choose "Read" access for datasets/models, or "Write" for uploading
3. Enter token in the "HF Token" field and click "Login"
4. Status shows "✓ Authenticated as [username]" when successful
5. Now access private repositories and gated models

### **NEW: Custom Model Workflow** 
1. Enable "Use Custom Model Weights" checkbox
2. Enter repository name (e.g., `microsoft/DialoGPT-medium`)
3. Click "Validate Repo" to check accessibility
4. Model info displays: type, file count, accessibility status
5. Process with your custom model instead of default weights

## 🛡️ **Error Handling & Reliability**

### Batch Processing Safety
- Individual batch error handling
- Progress tracking with batch-level feedback
- Graceful degradation on model failures
- Memory cleanup between batches

### Data Validation
- Early column filtering for memory efficiency
- Consistency checks between embeddings and metadata
- File format validation
- API key verification for OpenAI models

## 💡 **Performance Tips**

### Memory Optimization
- Use column selection to reduce memory usage
- Adjust batch size based on available RAM
- Filter unnecessary columns early in pipeline
- Monitor system resources during processing

### Speed Optimization
- Larger batch sizes = faster processing (if memory allows)
- Select only needed columns to reduce I/O
- Use local models to avoid API rate limits
- Preview datasets before full processing

## 🔧 **Technical Implementation**

### New Components
- `batch_size` parameter with UI control
- `selected_columns` state management
- `update_column_selection_ui()` method
- Enhanced `process_dataset()` with batch processing
- Column type detection and filtering
- `hf_token` and `hf_authenticated` state management
- `use_custom_model` and `custom_model_repo` for custom weights
- HF authentication methods (`hf_login()`, `check_hf_auth()`)
- Custom model validation (`validate_custom_model()`)

### Enhanced UI Elements
- Batch size input with validation (10-1000 range)
- Column selection window with scrolling
- Progress tracking with batch-level updates
- Memory usage estimation and reporting
- HF authentication section with secure token input
- Custom model toggle with repository validation
- Real-time authentication status display
- Model info display (type, files, accessibility)

## 🎯 **Benefits Summary**

1. **⚡ Faster Processing**: Batch processing optimizes memory usage
2. **💾 Reduced Storage**: Column filtering saves disk space
3. **🎛️ Better Control**: Fine-grained control over processing parameters
4. **📊 Cleaner Data**: Only relevant columns in final output
5. **🔍 Better Preview**: See exactly what data you're working with
6. **⚠️ Error Recovery**: Robust handling of processing failures
7. **📈 Scalability**: Handle larger datasets efficiently
8. **🤗 Private Access**: Use private HF datasets and models
9. **🔧 Custom Models**: Leverage domain-specific fine-tuned models
10. **🚀 Advanced AI**: Access latest and best-performing models

## 🚀 **Demo & Testing**

Run the demo script to explore new features:
```bash
python demo_enhanced_features.py
```

This showcase:
- Interactive column selection
- Batch size configuration
- Memory usage optimization
- Progress tracking
- Error handling

The enhanced file selector transforms the data visualization tool into a complete end-to-end pipeline for processing raw datasets into interactive visualizations with maximum efficiency and control. 