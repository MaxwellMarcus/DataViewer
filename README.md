# Data Visualization Tool

A powerful data visualization application for exploring high-dimensional embeddings with interactive clustering, search, and custom axis creation.

## Getting Started

### File Requirements

The application requires two files to get started:

1. **Embeddings File** (`.npy` or `.npz`)
   - A numpy array containing your embeddings data
   - Should be a 2D array where each row is an embedding vector
   - Common formats: `.npy` (numpy array) or `.npz` (compressed numpy)

2. **Metadata File** (`.csv`, `.json`, `.parquet`, or `.xlsx`)
   - Contains metadata corresponding to each embedding
   - Must have the same number of rows as embeddings
   - Should include a text column for captions/descriptions

### Starting the Application

1. Run the application:
   ```bash
   python app_dearpygui.py
   ```

2. **File Selection Interface** will appear:
   - Select your embeddings file (`.npy` format recommended)
   - Select your metadata file (`.csv` format recommended)
   - Choose which column contains the primary text to display
   - Click "Continue" to load the data

3. **Example Data Button**: If you have files named `embeddings.npy` and `metadata.csv` in the current directory, you can click "Load Example Data" to automatically load them.

## Features

### Visualization Tabs
- **t-SNE**: Interactive t-SNE visualization with customizable perplexity
- **PCA**: Principal Component Analysis with explained variance
- **UMAP**: UMAP dimensionality reduction
- **Vector Projection**: Custom semantic axes using text prompts

### Clustering
- **HDBSCAN**: Density-based clustering with adjustable parameters
- **Metadata Clustering**: Group by metadata values
- **Text Search Clustering**: Create clusters from search results
- **Lasso Selection**: Draw custom selections to create clusters

### Interactive Features
- **Point Selection**: Click points to create custom axes
- **Search & Highlight**: Find and highlight specific text
- **Similarity Search**: Find semantically similar items
- **Hover Information**: Detailed info on mouse hover
- **Custom Axes**: Create semantic direction vectors

### Advanced Features
- **Saved Axes**: Save and reuse custom axes across sessions
- **Plane Projection**: Orthogonal projection for multiple axes
- **Real-time Updates**: Dynamic clustering and visualization updates
- **Export**: Save visualizations and clustering results

## File Format Examples

### Embeddings File (`embeddings.npy`)
```python
import numpy as np

# Example: 1000 items with 384-dimensional embeddings
embeddings = np.random.randn(1000, 384)
np.save('embeddings.npy', embeddings)
```

### Metadata File (`metadata.csv`)
```csv
title,description,category,date
"Sample Title 1","Description of item 1","news","2024-01-01"
"Sample Title 2","Description of item 2","tech","2024-01-02"
```

## Usage Tips

1. **Start Small**: Begin with a subset of your data (1000-5000 items) to ensure smooth performance
2. **Choose Good Text Columns**: Select metadata columns with meaningful text for the best visualization experience
3. **Experiment with Parameters**: Try different clustering parameters and dimensionality reduction settings
4. **Save Useful Axes**: Create and save semantic axes that work well for your data domain
5. **Use Multiple Views**: Create multiple tabs with different visualizations for comprehensive analysis

## Troubleshooting

### Common Issues

**"Embeddings and metadata have different lengths"**
- Ensure your embeddings array and metadata file have the same number of rows

**"Column not found in metadata"**
- Check that the selected primary text column exists in your metadata file

**"Error loading data"**
- Verify file formats are supported (.npy for embeddings, .csv/.json/.parquet for metadata)
- Check that files are not corrupted and can be loaded with pandas/numpy

**Performance Issues**
- Try reducing the dataset size or adjusting visualization parameters
- Consider using UMAP instead of t-SNE for large datasets

## Dependencies

Required Python packages:
- `dearpygui`
- `numpy`
- `pandas`
- `scikit-learn`
- `umap-learn`
- `hdbscan`
- `sentence-transformers` (for text encoding)

Install with:
```bash
pip install dearpygui numpy pandas scikit-learn umap-learn hdbscan sentence-transformers
```

## License

This project is open source. Choose an appropriate license for your use case. 