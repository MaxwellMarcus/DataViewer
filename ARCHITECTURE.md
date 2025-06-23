# Data Visualization Tool - Architecture Documentation

## Overview

The Data Visualization Tool is a Python-based desktop application built with DearPyGui for interactive exploration of high-dimensional embeddings. The application provides a comprehensive platform for visualizing, clustering, and analyzing embedding data with rich metadata support.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  File Selector  │  │   Main App      │  │  Font Config    │ │
│  │  (Enhanced)     │  │  (DearPyGui)    │  │  (Arial)        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    Management Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Data Manager   │  │   Tab Manager   │  │ Visualization   │ │
│  │  (Core Data)    │  │  (UI Tabs)      │  │ Components      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    Visualization Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Scatter Tabs   │  │  Dimensionality │  │  Custom Axes    │ │
│  │  (Plot Types)   │  │  Reduction      │  │  (KVP/Metadata)│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  ML Models      │  │  Preprocessing  │  │  Data Loading   │ │
│  │  (Embeddings)   │  │  (Text/Data)    │  │  (Multiple)     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Application Entry Point (`app_dearpygui.py`)

**Purpose**: Main application controller and DearPyGui interface coordinator

**Key Responsibilities**:
- Application lifecycle management
- Two-phase initialization (file selection → main application)
- UI orchestration and event handling
- Integration of all major components

**Architecture Pattern**: MVC Controller + Coordinator

**Key Classes**:
- `VisualizerDPG`: Main application class
  - Manages application flow
  - Coordinates between DataManager and TabManager
  - Handles clustering operations
  - Manages search and selection functionality

**Data Flow**:
1. File selection phase using `EnhancedFileSelector`
2. Data validation and loading
3. DataManager initialization
4. TabManager setup and UI creation
5. Event-driven user interactions

### 2. Data Management Layer (`data_manager.py`)

**Purpose**: Central data management, clustering, and state coordination

**Key Responsibilities**:
- Embeddings and metadata management
- Clustering operations (HDBSCAN, custom clusters)
- Color management for visualization
- Lasso selection handling
- Custom axes management
- Data export functionality

**Architecture Pattern**: Data Access Object (DAO) + State Manager

**Key Classes**:
- `DataManager`: Core data management class
  - Stores embeddings (`self.X`) and metadata (`self.metadata`)
  - Manages cluster labels (`self.labels`) and colors (`self.cluster_colors`)
  - Provides clustering methods (HDBSCAN, metadata-based, search-based)
  - Handles saved axes and point selections

**Data Structures**:
```python
# Core data
self.X: np.ndarray           # Embeddings (n_points, embedding_dim)
self.metadata: pd.DataFrame  # Metadata with arbitrary columns
self.labels: np.ndarray      # Cluster assignments (n_points,)

# Preprocessing results
self.umap_data: np.ndarray   # UMAP 2D projection
self.pca_data: np.ndarray    # PCA projection
self.pca: PCA               # PCA transformer

# State management
self.cluster_colors: dict    # {cluster_id: [r,g,b]}
self.cluster_history: list   # History of clustering operations
self.saved_axes: list       # Saved custom axes
self.lasso_selection: np.ndarray  # Selected point indices
```

### 3. Tab Management System (`tab_manager.py`)

**Purpose**: Dynamic tab creation and management for different visualization types

**Key Responsibilities**:
- Tab lifecycle management (creation, deletion, refresh)
- Dimensionality reduction tab creation (t-SNE, PCA, UMAP)
- Custom vector projection (KVP) tab creation
- UI state coordination between tabs
- Axis management and preview functionality

**Architecture Pattern**: Factory + Manager Pattern

**Key Classes**:
- `TabManager`: Central tab coordination
  - Manages tab instances and lifecycle
  - Provides factory methods for different tab types
  - Coordinates tab UI and data synchronization

**Tab Types Supported**:
- Standard tabs: UMAP, PCA, NGon (default geometric)
- Dynamic t-SNE tabs with custom parameters
- Dynamic PCA tabs with custom components
- Dynamic UMAP tabs with custom parameters
- Custom KVP (Key-Value Projection) tabs
- Metadata-based projection tabs

### 4. Visualization Components (`scatter_tabs.py`)

**Purpose**: Individual visualization tab implementations with interactive features

**Key Responsibilities**:
- Plot rendering and data visualization
- User interaction handling (hover, click, selection)
- Lasso selection implementation
- Custom legend management
- Point highlighting and clustering visualization

**Architecture Pattern**: Template Method + Observer Pattern

**Key Classes**:

#### Base Class: `ScatterTab`
- Abstract base for all visualization types
- Common functionality: hover, selection, lasso, clustering
- Template methods for tab-specific customization

#### Specialized Implementations:
- `PCATab`: Principal Component Analysis visualization
- `TSNETab`: t-SNE visualization with parameter controls
- `UMAPTab`: UMAP visualization with parameter controls
- `KVPTab`: Custom vector projection visualization
- `NGonTab`: Geometric N-gon visualization
- `MetadataTab`: Metadata-based 2D projections
- `CustomEncodedTab`: User-defined encoded projections

**Interactive Features**:
- Mouse hover with debounced information display
- Point selection for custom axis creation
- Lasso selection (rectangular and polygonal)
- Cluster visibility toggling
- Focus mode for individual clusters
- Custom legend with cluster management

### 5. File Management System

#### Enhanced File Selector (`enhanced_file_selector.py`)

**Purpose**: Advanced file selection with dataset processing capabilities

**Key Responsibilities**:
- Tab-based interface for file vs. dataset modes
- File validation and preview
- AI model integration for dataset processing
- Hugging Face authentication management
- Column selection and data preview

**Architecture Pattern**: Strategy Pattern + State Machine

**Features**:
- File mode: Load existing embeddings + metadata
- Dataset mode: Process raw data with AI models
- Support for multiple file formats (CSV, JSON, Parquet, Excel)
- Real-time validation and preview
- Batch processing with progress tracking

#### Legacy File Selector (`file_selector.py`)

**Purpose**: Simple file selection interface (maintained for compatibility)

**Key Responsibilities**:
- Basic embeddings and metadata file selection
- File validation
- Column selection for primary text field

### 6. AI Model Integration (`sample_models.py`)

**Purpose**: AI model abstraction layer for embedding generation

**Key Responsibilities**:
- Model availability detection
- Unified interface for different model types
- Fallback mechanisms for missing dependencies

**Supported Models**:
- **Sentence Transformers**: Fast, efficient sentence embeddings
- **BERT**: Bidirectional transformer embeddings
- **CLIP**: Vision-language model for text embeddings
- **Gemma**: Google's language model embeddings
- **OpenAI**: High-quality embeddings via API
- **Fallback**: Random embeddings for testing

**Architecture Pattern**: Strategy Pattern + Factory Pattern

### 7. Data Processing Pipeline

#### Dataset Loading (`load_dataset.py`)

**Purpose**: Smart dataset loading from various sources

**Key Responsibilities**:
- Hugging Face dataset integration
- URL-based data loading
- Local file processing
- Data format detection and conversion

#### Preprocessing (`preprocess.py`)

**Purpose**: Text preprocessing and data cleaning

**Key Responsibilities**:
- Text normalization and cleaning
- Data validation and consistency checks
- Format standardization

### 8. Supporting Infrastructure

#### Font Configuration (`font_config.py`)

**Purpose**: Cross-platform font management

**Key Responsibilities**:
- Arial font detection across operating systems
- Font registry management
- Global font application
- Fallback to default fonts when needed

**Cross-Platform Support**:
- macOS: System font locations
- Windows: Windows Fonts directory
- Linux: Common font directories with fallbacks

#### Utility Modules

- `projection.py`: Vector projection utilities
- `encode.py`: Text encoding utilities
- `ball_projection.py`: Geometric projection methods
- `check_setup.py`: Environment validation

## Data Flow Architecture

### 1. Application Startup Flow

```
User Launch → Enhanced File Selector → File Validation → Data Loading → Main Application
```

**Detailed Steps**:
1. `VisualizerDPG.__init__()` creates application instance
2. `show_file_selector()` displays enhanced file selector
3. User selects files or configures dataset processing
4. File validation and data loading
5. `start_main_application()` initializes main interface
6. DataManager and TabManager initialization
7. Default tab creation and UI setup

### 2. Data Processing Flow

```
Raw Data → Validation → DataManager → Preprocessing → Visualization → User Interaction
```

**Components Involved**:
- File validation in enhanced file selector
- Data loading and consistency checks
- UMAP/PCA preprocessing in DataManager
- Tab-specific data transformations
- Real-time clustering and visualization updates

### 3. User Interaction Flow

```
User Input → Event Handler → DataManager Update → Tab Refresh → UI Update
```

**Interaction Types**:
- **Clustering**: User parameters → DataManager.cluster_* → All tabs refresh
- **Selection**: Mouse interaction → Point selection → Highlight update
- **Search**: Text query → Match finding → Highlight and cluster creation
- **Tab Creation**: Parameters → TabManager factory → New tab instance

## State Management

### Global State Components

1. **Data State** (DataManager):
   - Embeddings and metadata
   - Current cluster assignments
   - Saved axes and selections
   - Color mappings

2. **UI State** (TabManager):
   - Active tabs and their configurations
   - Current tab selection
   - Tab-specific parameters

3. **Selection State** (Distributed):
   - Lasso selections (per tab)
   - Point selections (per tab)
   - Search results (global)

### State Synchronization

The application maintains state consistency through:
- **Observer Pattern**: DataManager notifies all tabs of cluster changes
- **Central Coordination**: TabManager coordinates tab-specific state
- **Event-Driven Updates**: UI events trigger appropriate state updates

## Extension Points

### Adding New Visualization Types

1. **Inherit from ScatterTab**: Implement abstract methods
2. **Register with TabManager**: Add factory method
3. **Implement UI Controls**: Override `setup_axis_controls()`
4. **Add Data Transformation**: Implement transformation logic

### Adding New Data Sources

1. **Extend EnhancedFileSelector**: Add new mode tab
2. **Implement Loader**: Add to data loading pipeline
3. **Add Validation**: Implement format-specific validation

### Adding New AI Models

1. **Implement Model Interface**: Follow pattern in `sample_models.py`
2. **Add to Model Registry**: Update `get_available_models()`
3. **Handle Dependencies**: Add optional import with fallback

## Performance Considerations

### Optimization Strategies

1. **Lazy Loading**: Tab data computed on-demand
2. **Efficient Updates**: Only refresh affected components
3. **Memory Management**: Careful handling of large embeddings
4. **Debounced Interactions**: Smooth hover and selection experiences

### Scalability Limits

- **Data Size**: Optimized for 1K-10K points; larger datasets may need subsampling
- **Embedding Dimensions**: Handles high-dimensional embeddings efficiently
- **Tab Count**: UI can handle multiple tabs with independent state

## Security and Dependencies

### Dependency Management

The application has tiered dependencies:
- **Core**: DearPyGui, NumPy, Pandas, Scikit-learn (required)
- **Enhanced**: Sentence Transformers, UMAP, HDBSCAN (recommended)
- **Optional**: OpenAI, CLIP, Transformers (for AI features)

### Data Security

- Local processing (no data sent to external servers by default)
- Optional OpenAI integration requires API key
- Hugging Face authentication for private datasets

## Testing and Validation

### Built-in Validation

- File format validation
- Data consistency checks
- Model availability detection
- Environment setup verification (`check_setup.py`)

### Example Data

- `get_test_embeddings.py`: Generate test data
- Example data loading functionality
- Built-in fallbacks for missing models

## Configuration and Customization

### Configuration Files

- `requirements.txt`: Core dependencies
- `requirements_enhanced.txt`: Extended dependencies
- `.gitignore`: Version control exclusions

### Customization Points

- Color schemes and themes
- Font configurations
- Default parameters for algorithms
- UI layout and sizing

This architecture provides a robust, extensible platform for embedding visualization with clear separation of concerns and well-defined interfaces between components. 