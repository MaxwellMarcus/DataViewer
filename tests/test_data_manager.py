"""
Unit tests for DataManager class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from data_manager import DataManager
from tests.conftest import TEST_DATA_POINTS, TEST_EMBEDDING_DIM, TEST_CLUSTERS

class TestDataManagerInitialization:
    """Test DataManager initialization and basic setup."""
    
    def test_init_basic(self, sample_embeddings, sample_metadata, mock_sklearn):
        """Test basic initialization."""
        dm = DataManager(
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            umap_hbdscan_clustering=False,
            primary_metadata_column='caption'
        )
        
        assert dm.n_points == TEST_DATA_POINTS
        assert dm.X.shape == (TEST_DATA_POINTS, TEST_EMBEDDING_DIM)
        assert isinstance(dm.metadata, pd.DataFrame)
        assert dm.primary_metadata_column == 'caption'
        assert dm.labels is not None
        assert len(dm.labels) == TEST_DATA_POINTS
    
    def test_init_with_umap_clustering(self, sample_embeddings, sample_metadata, mock_sklearn):
        """Test initialization with UMAP and HDBSCAN clustering."""
        dm = DataManager(
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            umap_hbdscan_clustering=True,
            primary_metadata_column='caption'
        )
        
        assert dm.umap_data is not None
        assert dm.umap is not None
        assert dm.hbdscan_labels is not None
        
        # Verify mock calls
        mock_sklearn['umap'].assert_called_once()
        mock_sklearn['hdbscan'].assert_called_once()
    
    def test_init_invalid_data(self):
        """Test initialization with invalid data."""
        with pytest.raises((ValueError, TypeError)):
            DataManager(
                embeddings="invalid_data",
                metadata=None
            )
    
    def test_pca_initialization(self, sample_embeddings, sample_metadata, mock_sklearn):
        """Test PCA initialization and variance calculation."""
        dm = DataManager(
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            umap_hbdscan_clustering=False
        )
        
        assert dm.pca is not None
        assert dm.pca_data is not None
        assert dm.pca_variance_75 >= 0
        # PCA data should have same number of points, but may have fewer dimensions
        assert dm.pca_data.shape[0] == TEST_DATA_POINTS
        assert dm.pca_data.shape[1] <= TEST_EMBEDDING_DIM

class TestClusteringOperations:
    """Test clustering functionality."""
    
    def test_hdbscan_clustering(self, data_manager_instance, mock_sklearn):
        """Test HDBSCAN clustering."""
        result = data_manager_instance.cluster_by_hdbscan(
            min_cluster_size=10,
            min_samples=5
        )
        
        assert result is True
        assert data_manager_instance.labels is not None
        assert len(data_manager_instance.labels) == TEST_DATA_POINTS
        
        # Check clustering history
        history = data_manager_instance.get_clustering_history()
        assert len(history) > 0
        assert history[-1]['operation'] == 'HDBSCAN'
    
    def test_metadata_value_clustering(self, data_manager_instance):
        """Test clustering by metadata values."""
        result = data_manager_instance.cluster_by_metadata_value('category')
        
        assert result is True
        assert data_manager_instance.labels is not None
        
        # Check that all unique categories are represented
        unique_categories = data_manager_instance.metadata['category'].unique()
        unique_labels = np.unique(data_manager_instance.labels)
        assert len(unique_labels) == len(unique_categories)
    
    def test_substring_match_clustering(self, data_manager_instance):
        """Test clustering by substring matching."""
        result = data_manager_instance.cluster_by_substring_match('caption', 'Sample')
        
        assert result is True
        assert data_manager_instance.labels is not None
        
        # Should create binary clusters (0 and 1)
        unique_labels = np.unique(data_manager_instance.labels)
        assert len(unique_labels) <= 2
        assert 0 in unique_labels or 1 in unique_labels
    
    def test_reset_to_original_clusters(self, data_manager_instance, mock_sklearn):
        """Test resetting to original clusters."""
        # First perform clustering to have something to reset from
        data_manager_instance.cluster_by_metadata_value('category')
        original_labels = data_manager_instance.labels.copy()
        
        # Set some initial labels to reset to
        data_manager_instance.hbdscan_labels = np.zeros(TEST_DATA_POINTS, dtype=int)
        
        result = data_manager_instance.reset_to_original_clusters()
        assert result is True
        
        # Labels should be different from the metadata clustering
        assert not np.array_equal(data_manager_instance.labels, original_labels)
    
    def test_clear_all_clusters(self, data_manager_instance):
        """Test clearing all clusters."""
        # First create some clusters
        data_manager_instance.cluster_by_metadata_value('category')
        
        result = data_manager_instance.clear_all_clusters()
        assert result is True
        
        # All points should be in cluster 0
        assert np.all(data_manager_instance.labels == 0)
        assert len(np.unique(data_manager_instance.labels)) == 1
    
    def test_clustering_invalid_column(self, data_manager_instance):
        """Test clustering with invalid column."""
        result = data_manager_instance.cluster_by_metadata_value('nonexistent_column')
        assert result is False
    
    def test_clustering_empty_substring(self, data_manager_instance):
        """Test clustering with empty substring."""
        result = data_manager_instance.cluster_by_substring_match('caption', '')
        assert result is False

class TestColorManagement:
    """Test cluster color management."""
    
    def test_initialize_cluster_colors(self, data_manager_instance):
        """Test cluster color initialization."""
        data_manager_instance.initialize_cluster_colors()
        
        unique_clusters = np.unique(data_manager_instance.labels)
        for cluster_id in unique_clusters:
            assert cluster_id in data_manager_instance.cluster_colors
            color = data_manager_instance.cluster_colors[cluster_id]
            assert len(color) == 3  # RGB
            assert all(0 <= c <= 255 for c in color)
    
    def test_get_cluster_color(self, data_manager_instance):
        """Test getting cluster colors."""
        cluster_id = 0
        color = data_manager_instance.get_cluster_color(cluster_id)
        
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
    
    def test_get_cluster_color_nonexistent(self, data_manager_instance):
        """Test getting color for nonexistent cluster."""
        color = data_manager_instance.get_cluster_color(999)
        assert color == [128, 128, 128]  # Default gray
    
    def test_update_cluster_colors(self, data_manager_instance):
        """Test updating cluster colors after clustering."""
        old_colors = data_manager_instance.cluster_colors.copy()
        
        # Change clustering
        data_manager_instance.cluster_by_metadata_value('category')
        data_manager_instance.update_cluster_colors()
        
        # Colors should be updated for new clusters
        new_unique_clusters = np.unique(data_manager_instance.labels)
        for cluster_id in new_unique_clusters:
            assert cluster_id in data_manager_instance.cluster_colors

class TestMetadataManagement:
    """Test metadata operations."""
    
    def test_add_metadata_valid(self, data_manager_instance):
        """Test adding valid metadata."""
        new_metadata = {
            'new_column': list(range(TEST_DATA_POINTS)),
            'text_column': [f"text_{i}" for i in range(TEST_DATA_POINTS)]
        }
        
        result = data_manager_instance.add_metadata(new_metadata)
        assert result is True
        
        assert 'new_column' in data_manager_instance.metadata.columns
        assert 'text_column' in data_manager_instance.metadata.columns
    
    def test_add_metadata_invalid_length(self, data_manager_instance):
        """Test adding metadata with wrong length."""
        new_metadata = {
            'short_column': [1, 2, 3]  # Too short
        }
        
        result = data_manager_instance.add_metadata(new_metadata)
        assert result is False
    
    def test_get_available_columns(self, data_manager_instance):
        """Test getting available columns."""
        columns = data_manager_instance.get_available_columns()
        assert isinstance(columns, list)
        assert 'caption' in columns
        assert 'category' in columns

class TestPointInformation:
    """Test point information retrieval."""
    
    def test_get_point_info_valid(self, data_manager_instance):
        """Test getting valid point information."""
        point_info = data_manager_instance.get_point_info(0)
        
        assert 'index' in point_info
        assert 'cluster' in point_info
        assert 'metadata' in point_info
        assert point_info['index'] == 0
        assert 'caption' in point_info['metadata']
    
    def test_get_point_info_invalid_index(self, data_manager_instance):
        """Test getting point info with invalid index."""
        point_info = data_manager_instance.get_point_info(-1)
        assert 'error' in point_info
        
        point_info = data_manager_instance.get_point_info(TEST_DATA_POINTS + 1)
        assert 'error' in point_info
    
    def test_get_cluster_info(self, data_manager_instance):
        """Test getting cluster information."""
        cluster_info = data_manager_instance.get_cluster_info()
        
        assert 'n_clusters' in cluster_info
        assert 'cluster_ids' in cluster_info
        assert 'cluster_sizes' in cluster_info
        assert 'total_points' in cluster_info
        assert cluster_info['total_points'] == TEST_DATA_POINTS

class TestLassoSelection:
    """Test lasso selection functionality."""
    
    @patch('dearpygui.dearpygui')
    def test_add_to_lasso_selection(self, mock_dpg, data_manager_instance):
        """Test adding points to lasso selection."""
        # Mock DPG calls that happen in _update_lasso_count
        mock_dpg.does_item_exist.return_value = False
        mock_dpg.set_value.return_value = None
        
        indices = [0, 1, 2, 3, 4]
        data_manager_instance.add_to_lasso_selection(indices)
        
        selection = data_manager_instance.get_lasso_selection()
        assert len(selection) == len(indices)
        assert all(idx in selection for idx in indices)
    
    @patch('dearpygui.dearpygui')
    def test_clear_lasso_selection(self, mock_dpg, data_manager_instance):
        """Test clearing lasso selection."""
        # Mock DPG calls
        mock_dpg.does_item_exist.return_value = False
        mock_dpg.set_value.return_value = None
        
        # First add some selections
        data_manager_instance.add_to_lasso_selection([0, 1, 2])
        assert len(data_manager_instance.get_lasso_selection()) > 0
        
        # Then clear
        data_manager_instance.clear_lasso_selection()
        assert len(data_manager_instance.get_lasso_selection()) == 0
    
    @patch('dearpygui.dearpygui')
    def test_handle_lasso_selection(self, mock_dpg, data_manager_instance):
        """Test handling lasso selection from tabs."""
        # Mock DPG calls
        mock_dpg.does_item_exist.return_value = False
        mock_dpg.set_value.return_value = None
        
        indices = [5, 6, 7, 8]
        data_manager_instance.handle_lasso_selection(indices)
        
        selection = data_manager_instance.get_lasso_selection()
        assert all(idx in selection for idx in indices)
    
    @patch('dearpygui.dearpygui')
    def test_duplicate_lasso_selection(self, mock_dpg, data_manager_instance):
        """Test adding duplicate indices to lasso selection."""
        # Mock DPG calls
        mock_dpg.does_item_exist.return_value = False
        mock_dpg.set_value.return_value = None
        
        indices = [0, 1, 2]
        
        # Add first time
        data_manager_instance.add_to_lasso_selection(indices)
        first_count = len(data_manager_instance.get_lasso_selection())
        
        # Add again (should not duplicate)
        data_manager_instance.add_to_lasso_selection(indices)
        second_count = len(data_manager_instance.get_lasso_selection())
        
        assert first_count == second_count

class TestSavedAxes:
    """Test saved axes functionality."""
    
    def test_add_saved_axis_valid(self, data_manager_instance):
        """Test adding a valid saved axis."""
        axis_info = {
            'name': 'test_axis',
            'type': 'difference',
            'vector': np.random.randn(TEST_EMBEDDING_DIM),
            'description': 'Test axis'
        }
        
        result = data_manager_instance.add_saved_axis(axis_info)
        assert result is True
        
        saved_axes = data_manager_instance.get_saved_axes()
        assert len(saved_axes) == 1
        assert saved_axes[0]['name'] == 'test_axis'
    
    def test_add_saved_axis_duplicate_name(self, data_manager_instance):
        """Test adding axis with duplicate name."""
        axis_info = {
            'name': 'duplicate_axis',
            'type': 'difference',
            'vector': np.random.randn(TEST_EMBEDDING_DIM)
        }
        
        # Add first time
        result1 = data_manager_instance.add_saved_axis(axis_info)
        assert result1 is True
        
        # Add second time with same name
        result2 = data_manager_instance.add_saved_axis(axis_info)
        assert result2 is False
    
    def test_add_saved_axis_missing_fields(self, data_manager_instance):
        """Test adding axis with missing required fields."""
        axis_info = {
            'name': 'incomplete_axis'
            # Missing 'type' and 'vector'
        }
        
        result = data_manager_instance.add_saved_axis(axis_info)
        assert result is False
    
    def test_remove_saved_axis(self, data_manager_instance):
        """Test removing a saved axis."""
        # First add an axis
        axis_info = {
            'name': 'removable_axis',
            'type': 'difference',
            'vector': np.random.randn(TEST_EMBEDDING_DIM)
        }
        data_manager_instance.add_saved_axis(axis_info)
        
        # Then remove it
        result = data_manager_instance.remove_saved_axis('removable_axis')
        assert result is True
        
        # Should be gone
        saved_axes = data_manager_instance.get_saved_axes()
        assert len(saved_axes) == 0
    
    def test_remove_nonexistent_axis(self, data_manager_instance):
        """Test removing nonexistent axis."""
        result = data_manager_instance.remove_saved_axis('nonexistent_axis')
        assert result is False
    
    def test_get_saved_axis_by_name(self, data_manager_instance):
        """Test getting saved axis by name."""
        axis_info = {
            'name': 'findable_axis',
            'type': 'difference',
            'vector': np.random.randn(TEST_EMBEDDING_DIM)
        }
        data_manager_instance.add_saved_axis(axis_info)
        
        retrieved_axis = data_manager_instance.get_saved_axis_by_name('findable_axis')
        assert retrieved_axis is not None
        assert retrieved_axis['name'] == 'findable_axis'
        
        # Test nonexistent
        nonexistent = data_manager_instance.get_saved_axis_by_name('nonexistent')
        assert nonexistent is None
    
    def test_get_saved_axis_names(self, data_manager_instance):
        """Test getting saved axis names."""
        # Add multiple axes
        for i in range(3):
            axis_info = {
                'name': f'axis_{i}',
                'type': 'difference',
                'vector': np.random.randn(TEST_EMBEDDING_DIM)
            }
            data_manager_instance.add_saved_axis(axis_info)
        
        names = data_manager_instance.get_saved_axis_names()
        assert len(names) == 3
        assert all(f'axis_{i}' in names for i in range(3))
    
    def test_get_saved_axis_data(self, data_manager_instance):
        """Test getting projection data for saved axis."""
        # Create a simple axis vector
        axis_vector = np.ones(TEST_EMBEDDING_DIM)
        axis_info = {
            'name': 'projection_axis',
            'type': 'difference',
            'vector': axis_vector
        }
        data_manager_instance.add_saved_axis(axis_info)
        
        projection_data = data_manager_instance.get_saved_axis_data('projection_axis')
        assert projection_data is not None
        assert len(projection_data) == TEST_DATA_POINTS
        
        # Test nonexistent axis
        nonexistent_data = data_manager_instance.get_saved_axis_data('nonexistent')
        assert nonexistent_data is None

class TestAxisCreation:
    """Test axis creation from points."""
    
    def test_create_axis_from_points(self, data_manager_instance):
        """Test creating axis from multiple points."""
        point_indices = [0, 1, 2, 3]
        axis_info = data_manager_instance.create_axis_from_points(
            point_indices, 
            axis_name='multi_point_axis'
        )
        
        assert axis_info is not None
        assert axis_info['name'] == 'multi_point_axis'
        assert axis_info['type'] == 'points_selection'
        assert 'vector' in axis_info
        assert len(axis_info['vector']) == TEST_EMBEDDING_DIM
        assert axis_info['source_points'] == point_indices
    
    def test_create_axis_from_single_point(self, data_manager_instance):
        """Test creating axis from single point."""
        point_index = 5
        axis_info = data_manager_instance.create_axis_from_single_point(
            point_index,
            axis_name='single_point_axis'
        )
        
        assert axis_info is not None
        assert axis_info['name'] == 'single_point_axis'
        assert axis_info['type'] == 'single_point'
        assert 'vector' in axis_info
        assert axis_info['source_point'] == point_index
    
    def test_create_axis_insufficient_points(self, data_manager_instance):
        """Test creating axis with insufficient points."""
        axis_info = data_manager_instance.create_axis_from_points([0])  # Only one point
        assert axis_info is None
    
    def test_create_axis_invalid_point_index(self, data_manager_instance):
        """Test creating axis with invalid point index."""
        axis_info = data_manager_instance.create_axis_from_single_point(-1)
        assert axis_info is None
        
        axis_info = data_manager_instance.create_axis_from_single_point(TEST_DATA_POINTS + 1)
        assert axis_info is None

class TestPointSelection:
    """Test point selection functionality."""
    
    @patch('dearpygui.dearpygui')
    def test_handle_single_point_selection(self, mock_dpg, data_manager_instance):
        """Test handling single point selection."""
        # Mock DPG functions that are called in handle_single_point_selection
        mock_dpg.does_item_exist.return_value = True
        mock_dpg.set_value.return_value = None
        mock_dpg.configure_item.return_value = None
        
        point_index = 10
        data_manager_instance.handle_single_point_selection(point_index)
        
        assert data_manager_instance.selected_point_index == point_index
    
    @patch('dearpygui.dearpygui')
    def test_create_and_save_axis_from_selected_point(self, mock_dpg, data_manager_instance):
        """Test creating and saving axis from selected point."""
        # First select a point
        point_index = 15
        data_manager_instance.selected_point_index = point_index
        
        # Mock GUI elements
        mock_dpg.does_item_exist.return_value = True
        
        result = data_manager_instance.create_and_save_axis_from_selected_point('test_selected_axis')
        assert result is True
        
        # Check that axis was saved
        saved_axes = data_manager_instance.get_saved_axes()
        assert len(saved_axes) == 1
        assert saved_axes[0]['name'] == 'test_selected_axis'
        
        # Selection should be cleared
        assert data_manager_instance.selected_point_index is None
    
    @patch('dearpygui.dearpygui')
    def test_create_axis_no_selected_point(self, mock_dpg, data_manager_instance):
        """Test creating axis when no point is selected."""
        # Mock DPG functions
        mock_dpg.does_item_exist.return_value = True
        mock_dpg.configure_item.return_value = None
        mock_dpg.set_value.return_value = None
        
        data_manager_instance.selected_point_index = None
        
        result = data_manager_instance.create_and_save_axis_from_selected_point()
        assert result is False

class TestDataExport:
    """Test data export functionality."""
    
    def test_export_data_without_embeddings(self, data_manager_instance):
        """Test exporting data without embeddings."""
        export_data = data_manager_instance.export_data(include_embeddings=False)
        
        assert 'metadata' in export_data
        assert 'labels' in export_data
        assert 'n_points' in export_data
        assert 'cluster_history' in export_data
        assert 'embeddings' not in export_data
        
        assert export_data['n_points'] == TEST_DATA_POINTS
    
    def test_export_data_with_embeddings(self, data_manager_instance):
        """Test exporting data with embeddings."""
        export_data = data_manager_instance.export_data(include_embeddings=True)
        
        assert 'embeddings' in export_data
        assert len(export_data['embeddings']) == TEST_DATA_POINTS
        assert len(export_data['embeddings'][0]) == TEST_EMBEDDING_DIM

class TestClusteringHistory:
    """Test clustering operation history."""
    
    def test_clustering_history_recording(self, data_manager_instance):
        """Test that clustering operations are recorded."""
        initial_history_length = len(data_manager_instance.get_clustering_history())
        
        # Perform clustering operation
        data_manager_instance.cluster_by_metadata_value('category')
        
        history = data_manager_instance.get_clustering_history()
        assert len(history) == initial_history_length + 1
        
        latest_operation = history[-1]
        assert latest_operation['operation'] == 'Metadata Value'
        assert 'timestamp' in latest_operation
        assert 'parameters' in latest_operation
    
    def test_multiple_clustering_operations_history(self, data_manager_instance):
        """Test history with multiple operations."""
        initial_length = len(data_manager_instance.get_clustering_history())
        
        # Perform multiple operations
        data_manager_instance.cluster_by_metadata_value('category')
        data_manager_instance.cluster_by_substring_match('caption', 'Sample')
        data_manager_instance.clear_all_clusters()
        
        history = data_manager_instance.get_clustering_history()
        assert len(history) == initial_length + 3
        
        operations = [op['operation'] for op in history[-3:]]
        assert 'Metadata Value' in operations
        assert 'Substring Match' in operations
        assert 'Clear All' in operations

@pytest.mark.integration
class TestDataManagerIntegration:
    """Integration tests for DataManager."""
    
    @patch('dearpygui.dearpygui')
    def test_full_workflow(self, mock_dpg, sample_embeddings, sample_metadata, mock_sklearn):
        """Test complete workflow from initialization to axis creation."""
        # Mock DPG functions used in lasso selection
        mock_dpg.does_item_exist.return_value = True
        mock_dpg.set_value.return_value = None
        
        # Initialize
        dm = DataManager(
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            umap_hbdscan_clustering=False,
            primary_metadata_column='caption'
        )
        
        # Perform clustering
        dm.cluster_by_metadata_value('category')
        
        # Select points and create axis
        points = [0, 1, 2]
        axis_info = dm.create_axis_from_points(points, 'integration_axis')
        assert axis_info is not None
        
        # Save axis
        result = dm.add_saved_axis(axis_info)
        assert result is True
        
        # Test lasso selection
        dm.handle_lasso_selection([5, 6, 7])
        assert len(dm.get_lasso_selection()) == 3
        
        # Export data
        export_data = dm.export_data()
        assert export_data['n_points'] == TEST_DATA_POINTS
        
        # Verify state consistency
        cluster_info = dm.get_cluster_info()
        assert cluster_info['total_points'] == TEST_DATA_POINTS

@pytest.mark.performance
class TestDataManagerPerformance:
    """Performance tests for DataManager."""
    
    def test_large_dataset_clustering(self, performance_data):
        """Test clustering performance with larger datasets."""
        for size, data in performance_data.items():
            if size > 1000:  # Only test larger sizes for performance
                dm = DataManager(
                    embeddings=data['embeddings'],
                    metadata=data['metadata'],
                    umap_hbdscan_clustering=False
                )
                
                # Test clustering performance
                import time
                start_time = time.time()
                dm.cluster_by_metadata_value('category')
                clustering_time = time.time() - start_time
                
                # Should complete in reasonable time (adjust threshold as needed)
                assert clustering_time < 5.0, f"Clustering took too long: {clustering_time:.2f}s for {size} points"

def test_data_manager_placeholder():
    """Placeholder test to ensure test structure works."""
    assert True 