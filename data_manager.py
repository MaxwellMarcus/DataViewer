#!/usr/bin/env python3
"""
DataManager class for handling data, metadata, clustering operations, color management, and control UI
"""

import numpy as np
import pandas as pd
import hdbscan
from typing import Dict, List, Optional, Any, Tuple
import umap
import dearpygui.dearpygui as dpg
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from projection import get_projection
from encode import encode
import re
from sklearn.metrics.pairwise import euclidean_distances
import time


class DataManager:
    """
    Manages data, metadata, clustering operations, and color management for the visualization application.
    """
    
    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame, umap_hbdscan_clustering: bool = True, auto_cluster: bool = True, primary_metadata_column = None):
        """
        Initialize the DataManager with embeddings and metadata.
        
        Args:
            embeddings (np.ndarray): The embedding vectors
            metadata (pd.DataFrame): DataFrame containing metadata including captions
            umap_hbdscan_clustering (bool): Whether to perform UMAP and HDBSCAN clustering
        """
        self.X = embeddings
        self.n_points = len(embeddings)
        self.n_neighbors = 15
        self.min_dist = 0.1
        self.umap_data = None
        self.primary_metadata_column = primary_metadata_column
        
        if umap_hbdscan_clustering:
            print("Performing UMAP dimensionality reduction...")
            reducer = umap.UMAP(n_components=2, n_neighbors=self.n_neighbors, min_dist=self.min_dist, unique=True)
            self.umap_data = reducer.fit_transform(embeddings)

        self.pca = PCA()
        self.pca_data = self.pca.fit_transform(self.X)
        self.pca_variance_75 = 0
        while self.pca.explained_variance_ratio_[ :self.pca_variance_75 ].sum() < 0.75:
            self.pca_variance_75 += 1
            if self.pca_variance_75 > self.X.shape[1]:
                self.pca_variance_75 = self.X.shape[1]
                break

        # Create main DataFrame with metadata
        self.metadata = metadata
        
        # Cluster management
        self.labels = np.zeros(self.n_points, dtype=int)
        self.hbdscan_labels = None
        self.cluster_history = []  # History of clustering operations
        
        # Color management
        self.cluster_colors = {}
        self.base_colors = [
            [255, 89, 94],    # Coral red
            [138, 201, 38],   # Lime green
            [25, 130, 196],   # Ocean blue 
            [255, 202, 58],   # Marigold yellow
            [106, 76, 147],   # Royal purple
            [255, 157, 0],    # Orange
            [57, 181, 174],   # Turquoise
            [229, 80, 157],   # Hot pink
            [141, 153, 174],  # Cool gray
            [86, 204, 242]    # Sky blue
        ]
        
        # Initialize cluster labels and colors
        self.original_labels = self.labels.copy()
        self.n_clusters = len(np.unique(self.labels))

        if auto_cluster:
            self._perform_initial_clustering()
        
        # Initialize colors for clusters
        self.initialize_cluster_colors()
        
        self.lasso_selection = np.array([], dtype=np.int32)
        
        # Initialize saved axes and point selection
        self.saved_axes = []
        self.selected_point_index = None
        
        print(f"DataManager initialized with {self.n_points} data points")
        print(f"Data shape: {self.X.shape}")
        if self.metadata is not None:
            print(f"DataFrame columns: {self.metadata.columns.tolist()}")
        else:
            print("No metadata provided")
    
    def _perform_initial_clustering(self):
        """Perform initial HDBSCAN clustering"""
        try:
            if self.umap_data is None:
                print("No UMAP data available, skipping initial clustering")
                return
                
            print("Performing initial HDBSCAN clustering...")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=10)
            self.hbdscan_labels = clusterer.fit_predict(self.umap_data)
            self.labels = self.hbdscan_labels.copy()
            
            # Record initial clustering
            self._record_clustering_operation("Initial HDBSCAN", {
                'method': 'HDBSCAN',
                'min_cluster_size': 500,
                'min_samples': 10
            })
            
            unique_clusters = np.unique(self.labels)
            print(f"Initial clustering created {len(unique_clusters)} clusters")
            for cluster_id in unique_clusters:
                cluster_size = np.sum(self.labels == cluster_id)
                print(f"Cluster {cluster_id}: {cluster_size} points")
                
        except Exception as e:
            print(f"Error in initial clustering: {str(e)}")
            # Fallback to single cluster
            self.labels = np.zeros(self.n_points, dtype=int)
            self.hbdscan_labels = self.labels.copy()

    def initialize_cluster_colors(self):
        """Initialize consistent colors for all clusters"""
        if self.labels is None:
            return
            
        unique_clusters = np.unique(self.labels)
        for cluster_id in unique_clusters:
            if cluster_id not in self.cluster_colors:
                # Use base colors first, then generate random ones
                if cluster_id < len(self.base_colors) and cluster_id >= 0:
                    self.cluster_colors[cluster_id] = self.base_colors[cluster_id]
                else:
                    # Generate a consistent random color based on cluster_id
                    np.random.seed(abs(cluster_id) + 1000)  # Ensure consistency
                    self.cluster_colors[cluster_id] = [
                        np.random.randint(50, 255),
                        np.random.randint(50, 255),
                        np.random.randint(50, 255)
                    ]
        
        print(f"Initialized colors for {len(unique_clusters)} clusters")

    def get_cluster_color(self, cluster_id):
        """Get color for a specific cluster"""
        if cluster_id not in self.cluster_colors:
            self.initialize_cluster_colors()
        return self.cluster_colors.get(cluster_id, [128, 128, 128])  # Default gray

    def update_cluster_colors(self):
        """Update colors when clusters change"""
        self.initialize_cluster_colors()
    
    def add_metadata(self, metadata_dict: Dict[str, List[Any]]) -> bool:
        """
        Add metadata columns to the DataFrame.
        
        Args:
            metadata_dict (Dict[str, List[Any]]): Dictionary where keys are column names
                                                 and values are lists of metadata values
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if len(metadata_dict) == 0:
                print("No metadata provided")
                return False
                
            # Verify all metadata lists have the same length as the data
            for col, values in metadata_dict.items():
                if len(values) != self.n_points:
                    print(f"Error: Metadata column '{col}' has length {len(values)} but expected {self.n_points}")
                    return False
                    
            # Add new columns to DataFrame
            for col, values in metadata_dict.items():
                self.metadata[col] = values
                    
            print(f"Added metadata columns: {', '.join(metadata_dict.keys())}")
            print(f"DataFrame now has columns: {self.metadata.columns.tolist()}")
            return True
            
        except Exception as e:
            print(f"Error adding metadata: {str(e)}")
            return False
    
    def get_available_columns(self) -> List[str]:
        """Get list of available columns for clustering"""
        if self.metadata is not None:
            return self.metadata.columns.tolist()
        else:
            return []
    
    def cluster_by_hdbscan(self, min_cluster_size: int = 500, min_samples: int = 10) -> bool:
        """
        Perform HDBSCAN clustering.
        
        Args:
            min_cluster_size (int): Minimum cluster size
            min_samples (int): Minimum samples
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Performing HDBSCAN clustering with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            
            # Use UMAP data if available, otherwise use original embeddings
            data_for_clustering = self.umap_data if self.umap_data is not None else self.X
            
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            new_labels = clusterer.fit_predict(data_for_clustering)
            
            # Update labels
            self.labels = new_labels
            
            # Update colors for new clusters
            self.update_cluster_colors()
            
            # Record clustering operation
            self._record_clustering_operation("HDBSCAN", {
                'method': 'HDBSCAN',
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples
            })
            
            # Debug results
            unique_clusters = np.unique(new_labels)
            print(f"HDBSCAN created {len(unique_clusters)} clusters")
            for cluster_id in unique_clusters:
                cluster_size = np.sum(new_labels == cluster_id)
                print(f"Cluster {cluster_id}: {cluster_size} points")
            
            return True
            
        except Exception as e:
            print(f"Error in HDBSCAN clustering: {str(e)}")
            return False
    
    def cluster_by_metadata_value(self, column: str) -> bool:
        """
        Create clusters based on unique values in a metadata column.
        
        Args:
            column (str): Column name to cluster by
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if column not in self.metadata.columns:
                print(f"Error: Column '{column}' not found in data")
                return False
            
            print(f"Creating clusters based on column: {column}")
            
            # Get unique values and create mapping to cluster IDs
            unique_values = self.metadata[column].unique()
            value_to_cluster = {val: i for i, val in enumerate(unique_values)}
            
            print(f"Found {len(unique_values)} unique values in column '{column}'")
            for val, cluster_id in value_to_cluster.items():
                print(f"Value '{val}' -> Cluster {cluster_id}")
            
            # Create new labels based on values
            new_labels = np.array([value_to_cluster[val] for val in self.metadata[column]])
            
            # Update labels
            self.labels = new_labels
            
            # Update colors for new clusters
            self.update_cluster_colors()
            
            # Record clustering operation
            self._record_clustering_operation("Metadata Value", {
                'method': 'Metadata Value',
                'column': column,
                'unique_values': unique_values.tolist()
            })
            
            # Debug results
            unique_clusters = np.unique(new_labels)
            print(f"Created {len(unique_clusters)} clusters from metadata")
            for cluster_id in unique_clusters:
                cluster_size = np.sum(new_labels == cluster_id)
                value = list(value_to_cluster.keys())[list(value_to_cluster.values()).index(cluster_id)]
                print(f"Cluster {cluster_id} ('{value}'): {cluster_size} points")
            
            return True
            
        except Exception as e:
            print(f"Error in metadata value clustering: {str(e)}")
            return False
    
    def cluster_by_substring_match(self, column: str, substring: str) -> bool:
        """
        Create binary clusters based on substring matching.
        
        Args:
            column (str): Column name to search in
            substring (str): Substring to search for
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if column not in self.metadata.columns:
                print(f"Error: Column '{column}' not found in data")
                return False
            
            if not substring:
                print("Error: Empty substring provided")
                return False
            
            print(f"Creating clusters based on substring '{substring}' in column '{column}'")
            
            # Create binary labels: 1 for matches, 0 for non-matches
            new_labels = np.array([
                1 if substring.lower() in str(val).lower() else 0 
                for val in self.metadata[column]
            ])
            
            # Update labels
            self.labels = new_labels
            
            # Update colors for new clusters
            self.update_cluster_colors()
            
            # Record clustering operation
            self._record_clustering_operation("Substring Match", {
                'method': 'Substring Match',
                'column': column,
                'substring': substring
            })
            
            # Debug results
            matches = np.sum(new_labels == 1)
            non_matches = np.sum(new_labels == 0)
            print(f"Substring matching results:")
            print(f"- Matches: {matches} points")
            print(f"- Non-matches: {non_matches} points")
            
            # Show some example matches
            if matches > 0:
                print("\nExample matches:")
                match_indices = np.where(new_labels == 1)[0]
                for idx in match_indices[:5]:  # Show first 5 matches
                    value = self.metadata.iloc[idx][column]
                    print(f"- Point {idx}: {value}")
            
            return True
            
        except Exception as e:
            print(f"Error in substring match clustering: {str(e)}")
            return False
    
    def reset_to_original_clusters(self) -> bool:
        """
        Reset clusters to the original HDBSCAN clustering.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("Resetting to original clusters...")
            
            if self.hbdscan_labels is not None:
                self.labels = self.hbdscan_labels.copy()
                
                # Update colors for original clusters
                self.update_cluster_colors()
                
                # Record operation
                self._record_clustering_operation("Reset to Original", {
                    'method': 'Reset',
                    'target': 'original'
                })
                
                unique_clusters = np.unique(self.labels)
                print(f"Reset to original clustering with {len(unique_clusters)} clusters")
                return True
            else:
                print("No original clusters available")
                return False
                
        except Exception as e:
            print(f"Error resetting to original clusters: {str(e)}")
            return False
    
    def clear_all_clusters(self) -> bool:
        """
        Clear all clusters and create a single cluster containing all points.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("Clearing all clusters...")
            
            # Create single cluster with all points
            self.labels = np.zeros(self.n_points, dtype=int)
            
            # Update colors for single cluster
            self.update_cluster_colors()
            
            # Record operation
            self._record_clustering_operation("Clear All", {
                'method': 'Clear All',
                'result': 'Single cluster with all points'
            })
            
            print(f"All points assigned to single cluster (Cluster 0): {self.n_points} points")
            return True
            
        except Exception as e:
            print(f"Error clearing all clusters: {str(e)}")
            return False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get information about current clusters.
        
        Returns:
            Dict[str, Any]: Cluster information
        """
        if self.labels is None:
            return {"error": "No clustering performed"}
        
        unique_clusters = np.unique(self.labels)
        cluster_info = {
            "n_clusters": len(unique_clusters),
            "cluster_ids": unique_clusters.tolist(),
            "cluster_sizes": {},
            "cluster_colors": {},
            "total_points": self.n_points
        }
        
        for cluster_id in unique_clusters:
            cluster_info["cluster_sizes"][int(cluster_id)] = int(np.sum(self.labels == cluster_id))
            cluster_info["cluster_colors"][int(cluster_id)] = self.get_cluster_color(cluster_id)
        
        return cluster_info
    
    def get_point_info(self, point_idx: int) -> Dict[str, Any]:
        """
        Get information about a specific point.
        
        Args:
            point_idx (int): Index of the point
            
        Returns:
            Dict[str, Any]: Point information
        """
        if point_idx < 0 or point_idx >= self.n_points:
            return {"error": f"Invalid point index: {point_idx}"}
        
        point_info = {
            "index": point_idx,
            "cluster": int(self.labels[point_idx]) if self.labels is not None else None,
            "cluster_color": self.get_cluster_color(self.labels[point_idx]) if self.labels is not None else None,
            "metadata": {}
        }
        
        # Add all metadata if available
        if self.metadata is not None:
            for col in self.metadata.columns:
                point_info["metadata"][col] = self.metadata.iloc[point_idx][col]
        
        return point_info
    
    def _record_clustering_operation(self, operation_name: str, parameters: Dict[str, Any]):
        """Record a clustering operation in the history"""
        operation = {
            "timestamp": pd.Timestamp.now(),
            "operation": operation_name,
            "parameters": parameters,
            "n_clusters": len(np.unique(self.labels)) if self.labels is not None else 0
        }
        self.cluster_history.append(operation)
    
    def get_clustering_history(self) -> List[Dict[str, Any]]:
        """Get the history of clustering operations"""
        return self.cluster_history.copy()
    
    def get_lasso_selection(self):
        """Get the current lasso selection as a list"""
        return list(self.lasso_selection)
    
    def add_to_lasso_selection(self, indices):
        """Add indices to the lasso selection"""
        self.lasso_selection = np.concatenate([self.lasso_selection, np.array(indices, dtype=np.int32)[~np.isin(indices, self.lasso_selection)]])
        
        # Update UI counter
        self._update_lasso_count()

    def clear_lasso_selection(self):
        """Clear the lasso selection"""
        self.lasso_selection = np.array([], dtype=np.int32)
        
        # Update UI counter
        self._update_lasso_count()
    
    def _update_lasso_count(self):
        """Update the lasso selection count in the UI"""
        try:
            import dearpygui.dearpygui as dpg
            count = len(self.lasso_selection)
            print(f"[DEBUG] Updating lasso count UI: {count} points")
            if dpg.does_item_exist("lasso_selection_count"):
                dpg.set_value("lasso_selection_count", f"Selected Points: {count}")
                print(f"[DEBUG] Updated lasso_selection_count UI element")
            else:
                print(f"[DEBUG] lasso_selection_count UI element does not exist")
        except Exception as e:
            print(f"[ERROR] Error updating lasso count: {e}")
    
    def handle_lasso_selection(self, selected_indices):
        """Handle lasso selection from a specific tab"""
        print(f"[DEBUG] Received {len(selected_indices)} selected indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
        
        # Add to global selection
        old_count = len(self.lasso_selection)
        self.add_to_lasso_selection(selected_indices)
        new_count = len(self.lasso_selection)
        print(f"[DEBUG] Global lasso selection updated: {old_count} -> {new_count}")

    def export_data(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """
        Export current state as a dictionary.
        
        Args:
            include_embeddings (bool): Whether to include the full embeddings
            
        Returns:
            Dict containing the current state
        """
        export_dict = {
            'metadata': self.metadata.to_dict('records'),
            'labels': self.labels.tolist(),
            'n_points': self.n_points,
            'cluster_history': self.cluster_history.copy()
        }
        
        if include_embeddings:
            export_dict['embeddings'] = self.X.tolist()
            if self.umap_data is not None:
                export_dict['umap_data'] = self.umap_data.tolist()
        
        return export_dict

    # ===== SAVED AXES FUNCTIONALITY =====
    
    def add_saved_axis(self, axis_info: Dict[str, Any]) -> bool:
        """
        Add a saved axis to the collection.
        
        Args:
            axis_info (Dict[str, Any]): Axis information including name, type, vector, etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if axis with same name already exists
            existing_names = [axis['name'] for axis in self.saved_axes]
            if axis_info['name'] in existing_names:
                print(f"Axis with name '{axis_info['name']}' already exists")
                return False
            
            # Ensure required fields are present
            required_fields = ['name', 'type', 'vector']
            for field in required_fields:
                if field not in axis_info:
                    print(f"Missing required field '{field}' in axis info")
                    return False
            
            # Add to saved axes
            self.saved_axes.append(axis_info.copy())
            print(f"Added saved axis: {axis_info['name']} (type: {axis_info['type']})")
            return True
            
        except Exception as e:
            print(f"Error adding saved axis: {e}")
            return False
    
    def remove_saved_axis(self, axis_name: str) -> bool:
        """
        Remove a saved axis by name.
        
        Args:
            axis_name (str): Name of the axis to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            original_count = len(self.saved_axes)
            self.saved_axes = [axis for axis in self.saved_axes if axis['name'] != axis_name]
            
            if len(self.saved_axes) < original_count:
                print(f"Removed saved axis: {axis_name}")
                return True
            else:
                print(f"Axis '{axis_name}' not found")
                return False
                
        except Exception as e:
            print(f"Error removing saved axis: {e}")
            return False
    
    def get_saved_axes(self) -> List[Dict[str, Any]]:
        """
        Get all saved axes.
        
        Returns:
            List[Dict[str, Any]]: List of saved axis information
        """
        return self.saved_axes.copy()
    
    def get_saved_axis_names(self) -> List[str]:
        """
        Get names of all saved axes.
        
        Returns:
            List[str]: List of saved axis names
        """
        return [axis['name'] for axis in self.saved_axes]
    
    def get_saved_axis_by_name(self, name: str) -> Dict[str, Any]:
        """
        Get a saved axis by name.
        
        Args:
            name (str): Name of the axis
            
        Returns:
            Dict[str, Any]: Axis information or None if not found
        """
        for axis in self.saved_axes:
            if axis['name'] == name:
                return axis.copy()
        return None
    
    def get_saved_axis_data(self, axis_name: str) -> np.ndarray:
        """
        Get projection data for a saved axis.
        
        Args:
            axis_name (str): Name of the saved axis
            
        Returns:
            np.ndarray: Projected data or None if axis not found
        """
        try:
            axis_info = self.get_saved_axis_by_name(axis_name)
            if axis_info is None:
                print(f"Saved axis '{axis_name}' not found")
                return None
            
            vector = axis_info.get('vector', None)
            if vector is None:
                print(f"No vector found for saved axis '{axis_name}'")
                return None
            
            # Project the data using the saved vector
            return np.dot(self.X, vector)
            
        except Exception as e:
            print(f"Error getting saved axis data for '{axis_name}': {e}")
            return None
    
    def create_axis_from_points(self, point_indices: List[int], axis_name: str = None) -> Dict[str, Any]:
        """
        Create a custom axis from selected points by computing the direction 
        from the centroid of the points to each individual point.
        
        Args:
            point_indices (List[int]): Indices of selected points
            axis_name (str): Optional name for the axis
            
        Returns:
            Dict[str, Any]: Created axis information or None if failed
        """
        try:
            if len(point_indices) < 2:
                print("Need at least 2 points to create an axis")
                return None
            
            # Get embeddings for selected points
            selected_embeddings = self.X[point_indices]
            
            # Compute centroid
            centroid = np.mean(selected_embeddings, axis=0)
            
            # Compute direction vectors from centroid to each point
            direction_vectors = selected_embeddings - centroid
            
            # Average the direction vectors to get the main axis direction
            main_direction = np.mean(direction_vectors, axis=0)
            
            # Normalize the vector
            norm = np.linalg.norm(main_direction)
            if norm > 0:
                main_direction = main_direction / norm
            else:
                print("Cannot create axis: zero vector")
                return None
            
            # Generate axis name if not provided
            if axis_name is None:
                axis_name = f"Points_Axis_{len(self.saved_axes) + 1}"
            
            # Create axis info
            axis_info = {
                'name': axis_name,
                'type': 'points_selection',
                'vector': main_direction,
                'source_points': point_indices,
                'centroid': centroid,
                'n_points': len(point_indices)
            }
            
            print(f"Created axis from {len(point_indices)} points: {axis_name}")
            return axis_info
            
        except Exception as e:
            print(f"Error creating axis from points: {e}")
            return None
    
    def create_axis_from_single_point(self, point_index: int, reference_point_index: int = None, axis_name: str = None) -> Dict[str, Any]:
        """
        Create a custom axis from a single point by computing the direction 
        from a reference point (or centroid) to the selected point.
        
        Args:
            point_index (int): Index of the selected point
            reference_point_index (int): Optional reference point index (uses centroid if None)
            axis_name (str): Optional name for the axis
            
        Returns:
            Dict[str, Any]: Created axis information or None if failed
        """
        try:
            if point_index < 0 or point_index >= self.n_points:
                print(f"Invalid point index: {point_index}")
                return None
            
            # Get the selected point embedding
            selected_embedding = self.X[point_index]
            
            # Determine reference point
            if reference_point_index is not None:
                if reference_point_index < 0 or reference_point_index >= self.n_points:
                    print(f"Invalid reference point index: {reference_point_index}")
                    return None
                reference_embedding = self.X[reference_point_index]
            else:
                # Use centroid of all data as reference
                reference_embedding = np.mean(self.X, axis=0)
            
            # Compute direction vector
            direction_vector = selected_embedding - reference_embedding
            
            # Normalize the vector
            norm = np.linalg.norm(direction_vector)
            if norm > 0:
                direction_vector = direction_vector / norm
            else:
                print("Cannot create axis: zero vector")
                return None
            
            # Generate axis name if not provided
            if axis_name is None:
                point_info = self.get_point_info(point_index)
                if 'caption' in point_info['metadata']:
                    # Use first few words of caption
                    caption = str(point_info['metadata']['caption'])
                    words = caption.split()[:3]
                    axis_name = f"Point_{point_index}_{'_'.join(words)}"
                else:
                    axis_name = f"Point_{point_index}_Axis"
            
            # Create axis info
            axis_info = {
                'name': axis_name,
                'type': 'single_point',
                'vector': direction_vector,
                'source_point': point_index,
                'reference_point': reference_point_index,
                'reference_type': 'point' if reference_point_index is not None else 'centroid'
            }
            
            print(f"Created axis from point {point_index}: {axis_name}")
            return axis_info
            
        except Exception as e:
            print(f"Error creating axis from single point: {e}")
            return None
    
    # ===== POINT SELECTION FUNCTIONALITY =====
    
    def handle_single_point_selection(self, point_index: int):
        """
        Handle single point selection for axis creation.
        
        Args:
            point_index (int): Index of the selected point
        """
        try:
            print(f"Single point selected: {point_index}")
            
            # Store the selected point for potential axis creation
            self.selected_point_index = point_index
            
            # Get point information
            point_info = self.get_point_info(point_index)
            print(f"Selected point info: Cluster {point_info['cluster']}")
            
            # Update UI if available
            try:
                import dearpygui.dearpygui as dpg
                if dpg.does_item_exist("selected_point_info"):
                    info_text = f"Selected Point: {point_index}\n"
                    info_text += f"Cluster: {point_info['cluster']}\n"
                    if 'caption' in point_info['metadata']:
                        caption = str(point_info['metadata']['caption'])[:100]
                        info_text += f"Caption: {caption}{'...' if len(str(point_info['metadata']['caption'])) > 100 else ''}"
                    dpg.set_value("selected_point_info", info_text)
                
                # Show axis creation controls
                if dpg.does_item_exist("point_axis_controls"):
                    dpg.configure_item("point_axis_controls", show=True)
                    
            except Exception as ui_error:
                print(f"UI update error: {ui_error}")
            
        except Exception as e:
            print(f"Error handling single point selection: {e}")
    
    def create_and_save_axis_from_selected_point(self, axis_name: str = None) -> bool:
        """
        Create and save an axis from the currently selected point.
        
        Args:
            axis_name (str): Optional name for the axis
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not hasattr(self, 'selected_point_index') or self.selected_point_index is None:
                print("No point selected")
                return False
            
            # Create axis from selected point
            axis_info = self.create_axis_from_single_point(self.selected_point_index, axis_name=axis_name)
            
            if axis_info is None:
                return False
            
            # Add to saved axes
            success = self.add_saved_axis(axis_info)
            
            if success:
                # Clear selection
                self.selected_point_index = None
                
                # Update UI
                try:
                    import dearpygui.dearpygui as dpg
                    if dpg.does_item_exist("point_axis_controls"):
                        dpg.configure_item("point_axis_controls", show=False)
                    if dpg.does_item_exist("selected_point_info"):
                        dpg.set_value("selected_point_info", "No point selected")
                except:
                    pass
            
            return success
            
        except Exception as e:
            print(f"Error creating and saving axis from selected point: {e}")
            return False
