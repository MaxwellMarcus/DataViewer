"""
Shared pytest fixtures for the data visualization test suite.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os
from pathlib import Path

# Test data constants
TEST_DATA_POINTS = 100
TEST_EMBEDDING_DIM = 384
TEST_CLUSTERS = 4

@pytest.fixture
def sample_embeddings():
    """Create sample embedding data for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.randn(TEST_DATA_POINTS, TEST_EMBEDDING_DIM)

@pytest.fixture
def sample_metadata():
    """Create sample metadata DataFrame for testing."""
    np.random.seed(42)
    
    # Generate diverse metadata
    captions = [f"Sample caption {i} with some meaningful text" for i in range(TEST_DATA_POINTS)]
    categories = np.random.choice(['A', 'B', 'C', 'D'], TEST_DATA_POINTS)
    numerical_values = np.random.uniform(0, 100, TEST_DATA_POINTS)
    boolean_values = np.random.choice([True, False], TEST_DATA_POINTS)
    
    return pd.DataFrame({
        'caption': captions,
        'category': categories,
        'score': numerical_values,
        'is_valid': boolean_values,
        'index': range(TEST_DATA_POINTS)
    })

@pytest.fixture
def sample_labels():
    """Create sample cluster labels for testing."""
    np.random.seed(42)
    return np.random.randint(0, TEST_CLUSTERS, TEST_DATA_POINTS)

@pytest.fixture
def mock_dpg():
    """Mock DearPyGUI to avoid GUI dependencies in tests."""
    with patch('dearpygui.dearpygui') as mock:
        # Setup common mock returns
        mock.does_item_exist.return_value = True
        mock.get_mouse_pos.return_value = (100, 100)
        mock.get_plot_mouse_pos.return_value = (50, 50)
        mock.is_item_hovered.return_value = False
        mock.get_value.return_value = "test_value"
        
        # Mock context managers
        mock.theme.return_value.__enter__ = Mock(return_value=mock)
        mock.theme.return_value.__exit__ = Mock(return_value=None)
        mock.theme_component.return_value.__enter__ = Mock(return_value=mock)
        mock.theme_component.return_value.__exit__ = Mock(return_value=None)
        mock.plot.return_value.__enter__ = Mock(return_value=mock)
        mock.plot.return_value.__exit__ = Mock(return_value=None)
        mock.plot_axis.return_value.__enter__ = Mock(return_value=mock)
        mock.plot_axis.return_value.__exit__ = Mock(return_value=None)
        mock.group.return_value.__enter__ = Mock(return_value=mock)
        mock.group.return_value.__exit__ = Mock(return_value=None)
        mock.child_window.return_value.__enter__ = Mock(return_value=mock)
        mock.child_window.return_value.__exit__ = Mock(return_value=None)
        mock.item_handler_registry.return_value.__enter__ = Mock(return_value=mock)
        mock.item_handler_registry.return_value.__exit__ = Mock(return_value=None)
        mock.handler_registry.return_value.__enter__ = Mock(return_value=mock)
        mock.handler_registry.return_value.__exit__ = Mock(return_value=None)
        mock.draw_layer.return_value.__enter__ = Mock(return_value=mock)
        mock.draw_layer.return_value.__exit__ = Mock(return_value=None)
        mock.drawlist.return_value.__enter__ = Mock(return_value=mock)
        mock.drawlist.return_value.__exit__ = Mock(return_value=None)
        
        yield mock

@pytest.fixture
def temp_files():
    """Create temporary files for testing file operations."""
    files = {}
    temp_dir = tempfile.mkdtemp()
    
    # Create sample embeddings file
    embeddings = np.random.randn(TEST_DATA_POINTS, TEST_EMBEDDING_DIM)
    embeddings_path = os.path.join(temp_dir, 'test_embeddings.npy')
    np.save(embeddings_path, embeddings)
    files['embeddings'] = embeddings_path
    
    # Create sample metadata file
    metadata = pd.DataFrame({
        'caption': [f"Test caption {i}" for i in range(TEST_DATA_POINTS)],
        'category': np.random.choice(['A', 'B', 'C'], TEST_DATA_POINTS),
        'value': np.random.randn(TEST_DATA_POINTS)
    })
    metadata_path = os.path.join(temp_dir, 'test_metadata.csv')
    metadata.to_csv(metadata_path, index=False)
    files['metadata'] = metadata_path
    
    yield files
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_sklearn():
    """Mock scikit-learn components for faster testing."""
    with patch('sklearn.manifold.TSNE') as mock_tsne, \
         patch('sklearn.decomposition.PCA') as mock_pca, \
         patch('umap.UMAP') as mock_umap, \
         patch('hdbscan.HDBSCAN') as mock_hdbscan:
        
        # Setup TSNE mock
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.randn(TEST_DATA_POINTS, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        # Setup PCA mock
        mock_pca_instance = Mock()
        mock_pca_instance.fit_transform.return_value = np.random.randn(TEST_DATA_POINTS, TEST_EMBEDDING_DIM)
        mock_pca_instance.explained_variance_ratio_ = np.random.rand(TEST_EMBEDDING_DIM)
        mock_pca.return_value = mock_pca_instance
        
        # Setup UMAP mock
        mock_umap_instance = Mock()
        mock_umap_instance.fit_transform.return_value = np.random.randn(TEST_DATA_POINTS, 2)
        mock_umap.return_value = mock_umap_instance
        
        # Setup HDBSCAN mock
        mock_hdbscan_instance = Mock()
        mock_hdbscan_instance.fit_predict.return_value = np.random.randint(0, TEST_CLUSTERS, TEST_DATA_POINTS)
        mock_hdbscan.return_value = mock_hdbscan_instance
        
        yield {
            'tsne': mock_tsne,
            'pca': mock_pca,
            'umap': mock_umap,
            'hdbscan': mock_hdbscan
        }

@pytest.fixture
def data_manager_instance(sample_embeddings, sample_metadata, mock_sklearn):
    """Create a DataManager instance with test data."""
    # Import here to avoid import issues with mocking
    from data_manager import DataManager
    
    # Create instance without GUI-dependent clustering
    dm = DataManager(
        embeddings=sample_embeddings,
        metadata=sample_metadata,
        umap_hbdscan_clustering=False,  # Disable to avoid GUI dependencies
        primary_metadata_column='caption'
    )
    
    return dm

@pytest.fixture
def scatter_tab_instance(data_manager_instance, mock_dpg):
    """Create a ScatterTab instance with test data."""
    from scatter_tabs import ScatterTab
    
    tab = ScatterTab(
        name="test_tab",
        data_manager=data_manager_instance,
        transformation=None
    )
    
    return tab

@pytest.fixture(scope="session")
def integration_data():
    """Create larger dataset for integration tests."""
    np.random.seed(123)
    n_points = 1000
    n_dims = 512
    
    embeddings = np.random.randn(n_points, n_dims)
    metadata = pd.DataFrame({
        'caption': [f"Integration test item {i}" for i in range(n_points)],
        'group': np.random.choice(['Group1', 'Group2', 'Group3', 'Group4'], n_points),
        'score': np.random.uniform(0, 1, n_points),
        'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='1H')
    })
    
    return {
        'embeddings': embeddings,
        'metadata': metadata,
        'expected_clusters': 4
    }

class MockApp:
    """Mock application class for testing app-level functionality."""
    def __init__(self):
        self.data_manager = None
        self.tab_manager = None
        self.selected_embeddings = None
        self.selected_metadata = None
        
    def initialize_data_manager(self, embeddings, metadata):
        """Mock initialization method."""
        pass
        
    def show_main_interface(self):
        """Mock main interface method."""
        pass

@pytest.fixture
def mock_app():
    """Create a mock application instance."""
    return MockApp()

@pytest.fixture
def performance_data():
    """Create performance test data with various sizes."""
    sizes = [100, 500, 1000, 5000]
    datasets = {}
    
    for size in sizes:
        np.random.seed(42)
        datasets[size] = {
            'embeddings': np.random.randn(size, 128),
            'metadata': pd.DataFrame({
                'caption': [f"Item {i}" for i in range(size)],
                'category': np.random.choice(['A', 'B', 'C'], size)
            })
        }
    
    return datasets

# Custom assertions for testing
def assert_embeddings_shape(embeddings, expected_points, expected_dims):
    """Assert embeddings have expected shape."""
    assert isinstance(embeddings, np.ndarray), "Embeddings should be numpy array"
    assert embeddings.shape == (expected_points, expected_dims), \
        f"Expected shape ({expected_points}, {expected_dims}), got {embeddings.shape}"

def assert_metadata_valid(metadata, expected_points):
    """Assert metadata is valid DataFrame."""
    assert isinstance(metadata, pd.DataFrame), "Metadata should be pandas DataFrame"
    assert len(metadata) == expected_points, \
        f"Expected {expected_points} rows, got {len(metadata)}"
    assert not metadata.empty, "Metadata should not be empty"

def assert_labels_valid(labels, expected_points, min_clusters=1):
    """Assert cluster labels are valid."""
    assert isinstance(labels, np.ndarray), "Labels should be numpy array"
    assert len(labels) == expected_points, \
        f"Expected {expected_points} labels, got {len(labels)}"
    assert len(np.unique(labels)) >= min_clusters, \
        f"Expected at least {min_clusters} clusters"

# Export custom assertions
pytest.assert_embeddings_shape = assert_embeddings_shape
pytest.assert_metadata_valid = assert_metadata_valid
pytest.assert_labels_valid = assert_labels_valid 