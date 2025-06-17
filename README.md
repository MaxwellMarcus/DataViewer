# t-SNE Visualization Desktop Applications

This repository contains two versions of a desktop application that displays an interactive t-SNE (t-Distributed Stochastic Neighbor Embedding) visualization.

## Available Versions

### 1. PyQt5 Version (`app.py`)
- Uses PyQt5 with embedded Plotly visualization
- Web-based interactive plot with full Plotly features
- Professional Qt-based interface

### 2. Dear PyGui Version (`app_dearpygui.py`)
- Uses Dear PyGui with native plotting
- High-performance rendering
- Modern, game-engine-like interface
- Built-in regenerate data functionality

## Features

- Interactive t-SNE plot with zoom and pan capabilities
- Color-coded clusters
- Hover information for data points
- Native desktop interface
- Modern and clean design

## Requirements

- Python 3.7 or higher
- PyQt5
- Dear PyGui
- Plotly
- NumPy
- scikit-learn
- pandas

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Individual Versions
**PyQt5 version:**
```bash
python app.py
```

**Dear PyGui version:**
```bash
python app_dearpygui.py
```

### Option 2: Use the Comparison Script
```bash
python run_comparison.py
```
This will give you a menu to choose which version to run.

## Customizing the Data

To use your own data, modify the `create_tsne_plot` method in the `TSNEVisualizer` class in `app.py`. Replace the sample data generation with your own data loading and preprocessing steps.

## Note

The current version uses randomly generated sample data. You can modify the code to load your own dataset by replacing the data generation part in the `create_tsne_plot` method. 