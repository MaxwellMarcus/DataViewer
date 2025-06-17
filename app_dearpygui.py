import dearpygui.dearpygui as dpg
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from projection import get_projection
from encode import encode
import re
from sklearn.metrics.pairwise import euclidean_distances
import time
from datasets import load_dataset
import hdbscan
import pandas as pd
from data_manager import DataManager
from tab_manager import TabManager
import umap
from scatter_tabs import *


class VisualizerDPG:
    def __init__(self):
        """Initialize the application"""
        dpg.create_context()

        print("Loading data...")
        embeddings = np.load("./tech_news_embeddings.npy")
        metadata = pd.read_csv("./metadata.csv" )
            
        # Create DataFrame with captions
        # metadata = pd.DataFrame({'caption': captions})
            
        # Initialize DataManager
        self.data_manager = DataManager(embeddings, metadata, primary_metadata_column="title")
        

        
        # Initialize TabManager
        self.tab_manager = TabManager(self.data_manager)

        # Initialize tabs through TabManager
        self.tab_manager.initialize_tabs()

        # Initialize cluster tracking
        self.current_search_matches = []
        self.current_similarity_matches = []
        self.custom_clusters = {}  # {cluster_id: {"name": "cluster_name", "indices": [list_of_indices]}}
        self.next_cluster_id = 4  # Start after the original 4 clusters (0, 1, 2, 3)
        

        
        # Setup UI
        self.setup_ui()

    def get_cluster_color(self, cluster_id):
        """Get color for a specific cluster using DataManager"""
        return self.data_manager.get_cluster_color(cluster_id)

    def add_metadata(self, metadata_dict):
        """Add metadata using the DataManager"""
        return self.data_manager.add_metadata(metadata_dict)
        
    def get_metadata(self):
        """Get metadata using the DataManager"""
        return self.data_manager.metadata

    @property
    def labels(self):
        """Get labels from DataManager"""
        return self.data_manager.labels
    
    @property 
    def clusters(self):
        """Get clusters through TabManager"""
        return self.tab_manager.get_clusters()

    def get_current_axes_info(self):
        """Get current axes information through TabManager"""
        return self.tab_manager.get_current_axes_info()
    
    def update_axes_display(self):
        """Update the axes information display"""
        if not dpg.does_item_exist("current_tab_type"):
            return
            
        axes_info = self.get_current_axes_info()
        
        # Update text displays
        dpg.set_value("current_tab_type", f"Tab Type: {axes_info['type']}")
        dpg.set_value("current_x_axis", f"X-Axis: {axes_info['x_axis']}")
        dpg.set_value("current_y_axis", f"Y-Axis: {axes_info['y_axis']}")
        
        # Show/hide controls based on changeability
        if dpg.does_item_exist("pca_controls"):
            dpg.configure_item("pca_controls", show=(axes_info['type'] == 'PCA'))
        if dpg.does_item_exist("kvp_controls"):
            dpg.configure_item("kvp_controls", show=(axes_info['type'] == 'Vector Projection'))
        
        # Update PCA specific information
        if axes_info['type'] == 'PCA' and dpg.does_item_exist("pca_variance_info"):
            variance = axes_info['parameters'].get('explained_variance', [0.0, 0.0])
            dpg.set_value("pca_variance_info", 
                         f"Explained Variance: PC1={variance[0]:.3f}, PC2={variance[1]:.3f}")
    
    def on_tab_change(self, sender=None, app_data=None):
        """Handle tab change through TabManager"""
        self.tab_manager.on_tab_change(sender, app_data)
        self.update_axes_display()
        
    def setup_ui(self):
        """Setup the main UI with tabbed controls at bottom"""

        with dpg.window(label="Main Window", width=1200, height=800, pos=(0, 0), tag="main_window", no_resize=True, no_title_bar=True, no_collapse=True, no_scrollbar=True):
            # Create main window for plots
            with dpg.child_window(label="Visualization", tag="visualization_window", height=-250):
                self.tab_manager.setup_main_tab_ui()
            
            # Create bottom control panel with tabs
            with dpg.child_window(label="Controls", tag="controls_window", height=250):
                with dpg.tab_bar():
                    with dpg.tab(label="Data"):
                        with dpg.tab_bar():
                            # Cluster Controls Tab
                            with dpg.tab(label="Cluster Controls"):
                                self.setup_cluster_controls()
                            
                            # Search Controls Tab  
                            with dpg.tab(label="Search"):
                                self.setup_search_controls()
                    with dpg.tab(label="Tab Creation"):
                         with dpg.tab_bar():
                             # Vector Projection Tab Creation
                             with dpg.tab(label="Vector Projection"):
                                 self.tab_manager.setup_kvp_creation_controls()
                             
                             # Dimensionality Reduction Tab Creation
                             with dpg.tab(label="Dimensionality Reduction"):
                                 self.tab_manager.setup_dimred_creation_controls()

        dpg.create_viewport(title="Data Visualization", width=1200, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        
        print("Application UI setup complete. Starting main loop...")
        print("Lasso functionality should now be available:")
        print("1. Hold Shift and drag on the plot to select rectangular areas")
        print("2. Hold Ctrl and click to create polygon selections")
        print("3. Double-click to clear lasso selection")
        print("4. Use 'Create Cluster from Selection' to create clusters from selected points")
        print("\nPress Ctrl+C or close the window to exit...")
        
        dpg.start_dearpygui()
        dpg.destroy_context()

    def setup_cluster_controls(self):
        """Setup cluster creation and management controls"""
        dpg.add_text("CLUSTER CONTROLS", color=[255, 200, 100])
        dpg.add_separator()
        
        # Cluster creation section
        dpg.add_text("Create New Cluster", color=[200, 200, 200])
        
        # Cluster creation method selection
        with dpg.group(horizontal=True):
            dpg.add_text("Method:")
            dpg.add_combo(
                items=["HDBSCAN", "Metadata Value", "Substring Match"],
                default_value="HDBSCAN",
                width=150,
                tag="cluster_method"
            )
        
        # HDBSCAN parameters (shown by default)
        with dpg.group(tag="hdbscan_params", show=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Min Cluster Size:")
                dpg.add_slider_int(
                    min_value=10,
                    max_value=1000,
                    default_value=500,
                    width=150,
                    tag="min_cluster_size"
                )
            
            with dpg.group(horizontal=True):
                dpg.add_text("Min Samples:")
                dpg.add_slider_int(
                    min_value=5,
                    max_value=50,
                    default_value=10,
                    width=150,
                    tag="min_samples"
                )
        
        # Metadata value clustering parameters
        with dpg.group(tag="metadata_params", show=False):
            with dpg.group(horizontal=True):
                dpg.add_text("Column:")
                dpg.add_combo(
                    items=self.data_manager.get_available_columns(),
                    width=150,
                    tag="metadata_column"
                )
        
        # Substring matching parameters
        with dpg.group(tag="substring_params", show=False):
            with dpg.group(horizontal=True):
                dpg.add_text("Column:")
                dpg.add_combo(
                    items=self.data_manager.get_available_columns(),
                    width=150,
                    tag="substring_column"
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Substring:")
                dpg.add_input_text(
                    width=150,
                    tag="substring_value"
                )
        
        dpg.add_button(label="Create Cluster", callback=self.create_new_cluster_callback, width=-1)
        
        # Update parameter visibility based on selected method
        dpg.set_item_callback("cluster_method", self.update_cluster_params_callback)
        
        dpg.add_separator()
        
        # Cluster management
        dpg.add_text("Manage Clusters", color=[200, 200, 200])
        dpg.add_button(label="Reset to Original Clusters", callback=self.reset_clusters_callback, width=-1)
        dpg.add_button(label="Clear All Clusters", callback=self.clear_all_clusters_callback, width=-1)
        
        dpg.add_separator()
        
        # Cluster information
        dpg.add_text("Cluster Information", color=[200, 200, 200])
        dpg.add_button(label="Show Cluster Info", callback=self.show_cluster_info_callback, width=-1)
        dpg.add_button(label="Show Clustering History", callback=self.show_clustering_history_callback, width=-1)
        
        dpg.add_separator()
        
        # Point selection and axis creation
        dpg.add_text("POINT SELECTION", color=[255, 200, 100])
        dpg.add_separator()
        
        dpg.add_text("Click on a point in the plot to select it for axis creation", color=[200, 200, 200])
        
        # Selected point info
        dpg.add_text("Selected Point Info:", color=[180, 180, 180])
        dpg.add_text("No point selected", tag="selected_point_info", wrap=400, color=[150, 150, 150])
        
        # Axis creation controls (hidden by default)
        with dpg.group(tag="point_axis_controls", show=False):
            dpg.add_separator()
            dpg.add_text("Create Axis from Selected Point", color=[255, 255, 100])
            
            with dpg.group(horizontal=True):
                dpg.add_text("Axis Name:")
                dpg.add_input_text(hint="Optional custom name", width=200, tag="point_axis_name")
            
            dpg.add_button(label="Create and Save Axis", callback=self.create_axis_from_point_callback, width=300)
            dpg.add_text("Creates an axis from centroid to selected point", color=[150, 150, 150], wrap=300)
        
        dpg.add_separator()
        
        # Lasso selection controls
        dpg.add_text("LASSO SELECTION", color=[255, 200, 100])
        with dpg.group(horizontal=True):
            dpg.add_button(label="Create Cluster from Selection", callback=self.create_cluster_from_lasso_callback, width=200)
            dpg.add_button(label="Clear Selection", callback=self.clear_lasso_selection_callback, width=100)
        
        dpg.add_text("Selected Points: 0", tag="lasso_selection_count", color=[180, 180, 180])
        
        dpg.add_separator()
        
        # Status/Response area
        dpg.add_text("Status Messages", color=[200, 200, 200])
        dpg.add_input_text(
            label="",
            default_value="Ready for clustering operations...",
            multiline=True,
            readonly=True,
            height=60,
            width=-1,
            tag="response_text"
        )

    def setup_search_controls(self):
        """Setup search and similarity search controls"""
        dpg.add_text("SEARCH CONTROLS", color=[255, 200, 100])
        dpg.add_separator()
        
        # Text search section
        dpg.add_text("Search in Captions", color=[200, 200, 200])
        with dpg.group(horizontal=True):
            dpg.add_input_text(hint="Enter search term...", width=200, tag="search_input")
            dpg.add_button(label="Search", callback=self.search_captions_callback)
        
        # Search results and highlighting
        with dpg.group(horizontal=True):
            dpg.add_button(label="Highlight Results", callback=self.search_and_highlight_callback)
            dpg.add_button(label="Clear Highlights", callback=self.clear_highlights_callback)
            dpg.add_button(label="Create Cluster", callback=self.create_cluster_from_search_callback)
        
        dpg.add_separator()
        
        # Similarity search section
        dpg.add_text("Similarity Search", color=[200, 200, 200])
        with dpg.group(horizontal=True):
            dpg.add_input_text(hint="Enter text for similarity...", width=200, tag="similarity_input")
            dpg.add_button(label="Find Similar", callback=self.similarity_search_callback)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Threshold:")
            dpg.add_slider_float(
                min_value=0.0,
                max_value=2.0,
                default_value=0.5,
                width=100,
                tag="similarity_threshold"
            )
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Clear Similar", callback=self.clear_similarity_callback)
            dpg.add_button(label="Create Cluster", callback=self.create_cluster_from_similarity_callback)

    # =============================================================================
    # WRAPPER CALLBACK METHODS
    # =============================================================================
    
    def create_new_cluster_callback(self, sender=None, app_data=None):
        """Wrapper for creating a new cluster"""
        method = dpg.get_value("cluster_method")
        
        try:
            success = False
            if method == "HDBSCAN":
                min_cluster_size = dpg.get_value("min_cluster_size")
                min_samples = dpg.get_value("min_samples")
                success = self.data_manager.cluster_by_hdbscan(min_cluster_size, min_samples)
                
            elif method == "Metadata Value":
                column = dpg.get_value("metadata_column")
                if not column:
                    dpg.set_value("response_text", "Error: Please select a metadata column")
                    return
                success = self.data_manager.cluster_by_metadata_value(column)
                
            elif method == "Substring Match":
                column = dpg.get_value("substring_column")
                substring = dpg.get_value("substring_value")
                if not column or not substring:
                    dpg.set_value("response_text", "Error: Please select a column and enter a substring")
                    return
                success = self.data_manager.cluster_by_substring_match(column, substring)
                
            if success:
                self.sync_clusters_to_all_tabs()
                self.refresh_all_plots()
                dpg.set_value("response_text", f"Successfully created clusters using {method}")
            else:
                dpg.set_value("response_text", f"Failed to create clusters using {method}")
                
        except Exception as e:
            dpg.set_value("response_text", f"Error creating clusters: {str(e)}")
            print(f"Error in create_new_cluster: {str(e)}")

    def reset_clusters_callback(self, sender=None, app_data=None):
        """Wrapper for resetting clusters"""
        try:
            success = self.data_manager.reset_to_original_clusters()
            if success:
                self.sync_clusters_to_all_tabs()
                self.refresh_all_plots()
                dpg.set_value("response_text", "Successfully reset to original clusters")
            else:
                dpg.set_value("response_text", "Failed to reset clusters")
        except Exception as e:
            dpg.set_value("response_text", f"Error resetting clusters: {str(e)}")
            print(f"Error in reset_clusters: {str(e)}")

    def clear_all_clusters_callback(self, sender=None, app_data=None):
        """Wrapper for clearing all clusters"""
        try:
            success = self.data_manager.clear_all_clusters()
            if success:
                self.sync_clusters_to_all_tabs()
                self.refresh_all_plots()
                dpg.set_value("response_text", "Successfully cleared all clusters")
            else:
                dpg.set_value("response_text", "Failed to clear clusters")
        except Exception as e:
            dpg.set_value("response_text", f"Error clearing clusters: {str(e)}")
            print(f"Error in clear_all_clusters: {str(e)}")

    def show_cluster_info_callback(self, sender=None, app_data=None):
        """Wrapper for showing cluster information"""
        try:
            cluster_info = self.data_manager.get_cluster_info()
            
            info_text = f"Total clusters: {cluster_info['n_clusters']}\n"
            info_text += f"Total points: {cluster_info['total_points']}\n\n"
            info_text += "Cluster breakdown:\n"
            
            for cluster_id, count in cluster_info['cluster_sizes'].items():
                if cluster_id == -1:
                    info_text += f"Noise: {count} points\n"
                else:
                    info_text += f"Cluster {cluster_id}: {count} points\n"
            
            dpg.set_value("response_text", info_text)
            
        except Exception as e:
            dpg.set_value("response_text", f"Error showing cluster info: {str(e)}")
            print(f"Error in show_cluster_info: {str(e)}")

    def show_clustering_history_callback(self, sender=None, app_data=None):
        """Wrapper for showing clustering history"""
        try:
            history = self.data_manager.get_clustering_history()
            
            if not history:
                dpg.set_value("response_text", "No clustering operations recorded.")
                return
        
            history_text = "Clustering History:\n\n"
            for i, operation in enumerate(history, 1):
                history_text += f"{i}. {operation['operation']} at {operation['timestamp']}\n"
                history_text += f"   Parameters: {operation['parameters']}\n\n"
            
            dpg.set_value("response_text", history_text)
            
        except Exception as e:
            dpg.set_value("response_text", f"Error showing clustering history: {str(e)}")
            print(f"Error in show_clustering_history: {str(e)}")

    def update_cluster_params_callback(self, sender, app_data):
        """Update parameter visibility based on selected clustering method"""
        method = dpg.get_value("cluster_method")
        
        # Hide all parameter groups
        dpg.configure_item("hdbscan_params", show=False)
        dpg.configure_item("metadata_params", show=False) 
        dpg.configure_item("substring_params", show=False)
        
        # Show relevant parameter group
        if method == "HDBSCAN":
            dpg.configure_item("hdbscan_params", show=True)
        elif method == "Metadata Value":
            dpg.configure_item("metadata_params", show=True)
        elif method == "Substring Match":
            dpg.configure_item("substring_params", show=True)

    def search_captions_callback(self, sender=None, app_data=None):
        """Wrapper for caption search"""
        try:
            search_term = dpg.get_value("search_input")
            if not search_term:
                dpg.set_value("response_text", "Error: Please enter a search term")
                return
            
            if self.data_manager.metadata is None or 'caption' not in self.data_manager.metadata.columns:
                dpg.set_value("response_text", "Error: No 'caption' column found in metadata")
                return
        
            # Perform case-insensitive search
            matches = self.data_manager.metadata[
                self.data_manager.metadata['caption'].str.contains(search_term, case=False, na=False)
            ]
            
            # Store search results
            self.current_search_matches = matches.index.tolist()
            
            dpg.set_value("response_text", f"Found {len(self.current_search_matches)} matches for '{search_term}'")
            
        except Exception as e:
            dpg.set_value("response_text", f"Error searching captions: {str(e)}")
            print(f"Error in search_captions: {str(e)}")

    def search_and_highlight_callback(self, sender=None, app_data=None):
        """Wrapper for highlighting search results"""
        try:
            if not hasattr(self.data_manager, 'search_results') or not self.data_manager.search_results:
                dpg.set_value("response_text", "No search results to highlight. Please search first.")
                return
                
            # Highlight points on all tabs
            for tab in self.tab_manager.tabs:
                if hasattr(tab, 'highlight_search_results'):
                    tab.highlight_search_results(self.data_manager.search_results)
            
            dpg.set_value("response_text", f"Highlighted {len(self.data_manager.search_results)} search results")
            
        except Exception as e:
            dpg.set_value("response_text", f"Error highlighting search results: {str(e)}")
            print(f"Error in search_and_highlight: {str(e)}")

    def clear_highlights_callback(self, sender=None, app_data=None):
        """Wrapper for clearing highlights"""
        try:
            # Clear highlights on all tabs
            for tab in self.tab_manager.tabs:
                if hasattr(tab, 'clear_search_highlights'):
                    tab.clear_search_highlights()
            
            dpg.set_value("response_text", "Cleared all highlights")
            
        except Exception as e:
            dpg.set_value("response_text", f"Error clearing highlights: {str(e)}")
            print(f"Error in clear_highlights: {str(e)}")

    def create_cluster_from_search_callback(self, sender=None, app_data=None):
        """Wrapper for creating cluster from search results"""
        if not self.current_search_matches:
            dpg.set_value("search_results", "No search results to create cluster from.")
            return
        
        # Get the cluster name from input
        cluster_name = dpg.get_value("cluster_name_input").strip()
        if not cluster_name:
            cluster_name = f"Custom Cluster {self.next_cluster_id}"
        
        try:
            # Store custom cluster info
            self.custom_clusters[self.next_cluster_id] = {
                "name": cluster_name,
                "indices": self.current_search_matches.copy()
            }
            
            # Add cluster name to the main clusters list
            self.clusters.append(cluster_name)
            
            # Use centralized cluster update method
            self.update_cluster_labels(self.current_search_matches, self.next_cluster_id)
            
            # Update search results
            result_text = f"Created cluster '{cluster_name}' with {len(self.current_search_matches)} datapoints.\n\n"
            result_text += f"Points moved to cluster {self.next_cluster_id}:\n"
            for i, idx in enumerate(self.current_search_matches[:5]):  # Show first 5
                caption = self.data_manager.data.iloc[idx]['caption'] if idx < len(self.data_manager.data) else "Unknown"
                result_text += f"• Point {idx}: {caption[:50]}{'...' if len(caption) > 50 else ''}\n"
            if len(self.current_search_matches) > 5:
                result_text += f"• ...and {len(self.current_search_matches) - 5} more points"
            
            dpg.set_value("search_results", result_text)
            
            # Clear input and disable button
            dpg.set_value("cluster_name_input", "")
            dpg.configure_item("create_cluster_btn", enabled=False)
            
            # Increment cluster ID for next cluster
            self.next_cluster_id += 1
            self.current_search_matches = []
            
        except Exception as e:
            dpg.set_value("search_results", f"Error creating cluster: {str(e)}")
            print(f"Error in create_cluster_from_search: {str(e)}")
    
    def similarity_search_callback(self, sender=None, app_data=None):
        """Wrapper for similarity search"""
        try:
            search_text = dpg.get_value("similarity_input")
            threshold = dpg.get_value("similarity_threshold")
            
            if not search_text:
                dpg.set_value("response_text", "Error: Please enter text for similarity search")
                return
            
            # Encode the search text
            search_embedding = encode([search_text])
            
            # Calculate distances to all points
            distances = euclidean_distances(search_embedding, self.data_manager.X).flatten()
            
            # Find points within threshold
            similar_indices = np.where(distances <= threshold)[0]
            
            # Store similarity results
            self.data_manager.similarity_results = similar_indices.tolist()
            self.data_manager.similarity_distances = distances[similar_indices]
            
            dpg.set_value("response_text", f"Found {len(self.data_manager.similarity_results)} similar points to '{search_text}' (threshold: {threshold:.2f})")
            
        except Exception as e:
            dpg.set_value("response_text", f"Error in similarity search: {str(e)}")
            print(f"Error in similarity_search: {str(e)}")
    
    def clear_similarity_callback(self, sender=None, app_data=None):
        """Wrapper for clearing similarity results"""
        try:
            # Clear similarity highlights on all tabs
            for tab in self.tab_manager.tabs:
                if hasattr(tab, 'clear_similarity_highlights'):
                    tab.clear_similarity_highlights()
            
            dpg.set_value("response_text", "Cleared similarity highlights")
            
        except Exception as e:
            dpg.set_value("response_text", f"Error clearing similarity: {str(e)}")
            print(f"Error in clear_similarity: {str(e)}")
    
    def create_cluster_from_similarity_callback(self, sender=None, app_data=None):
        """Wrapper for creating cluster from similarity results"""
        try:
            if not hasattr(self.data_manager, 'similarity_results') or not self.data_manager.similarity_results:
                dpg.set_value("response_text", "No similarity results to cluster. Please search first.")
            return
        
            # Create new cluster from similarity results
            new_cluster_id = self.data_manager.n_clusters
            
            # Update labels for similarity result indices
            for idx in self.data_manager.similarity_results:
                self.data_manager.labels[idx] = new_cluster_id
            
            # Update cluster count and colors
            self.data_manager.n_clusters += 1
            self.data_manager.update_cluster_colors()
            
            # Sync and refresh
            self.sync_clusters_to_all_tabs()
            self.refresh_all_plots()
            
            # Record clustering operation
            self.data_manager._record_clustering_operation("Similarity-based clustering", {
                'similarity_results_count': len(self.data_manager.similarity_results),
                'new_cluster_id': new_cluster_id
            })
            
            dpg.set_value("response_text", f"Created cluster {new_cluster_id} from {len(self.data_manager.similarity_results)} similar points")
            
        except Exception as e:
            dpg.set_value("response_text", f"Error creating cluster from similarity: {str(e)}")
            print(f"Error in create_cluster_from_similarity: {str(e)}")
    
    def create_cluster_from_lasso_callback(self, sender=None, app_data=None):
        """Wrapper for creating cluster from lasso selection"""
        try:
            selected_indices = self.data_manager.lasso_selection
            
            if len(selected_indices) == 0:
                dpg.set_value("response_text", "No points selected. Use lasso tool to select points first.")
                return
                                     
            # Create new cluster ID
            new_cluster_id = self.data_manager.n_clusters
            
            # Update labels for selected indices
            for idx in selected_indices:
                self.data_manager.labels[idx] = new_cluster_id
            
            # Update cluster count
            self.data_manager.n_clusters += 1
            
            # Update colors and sync
            self.data_manager.update_cluster_colors()
            
            # Sync and refresh
            self.sync_clusters_to_all_tabs()
            self.refresh_all_plots()
            
            # Record clustering operation
            self.data_manager._record_clustering_operation("Lasso selection clustering", {
                'selected_points_count': len(selected_indices),
                'new_cluster_id': new_cluster_id
            })
            
            dpg.set_value("response_text", f"Created cluster {new_cluster_id} from {len(selected_indices)} selected points")
            
            # Clear selection after creating cluster
            self.data_manager.clear_lasso_selection()
            dpg.set_value("lasso_selection_count", "Selected Points: 0")
                
        except Exception as e:
            dpg.set_value("response_text", f"Error creating cluster from lasso: {str(e)}")
            print(f"Error in create_cluster_from_lasso: {str(e)}")
    
    def clear_lasso_selection_callback(self, sender=None, app_data=None):
        """Wrapper for clearing lasso selection"""
        try:
            self.data_manager.clear_lasso_selection()
            print("Lasso selection cleared")
        except Exception as e:
            print(f"Error clearing lasso selection: {e}")

    def create_axis_from_point_callback(self, sender=None, app_data=None):
        """Create and save an axis from the selected point"""
        try:
            axis_name = dpg.get_value("point_axis_name")
            if not axis_name.strip():
                axis_name = None
            
            success = self.data_manager.create_and_save_axis_from_selected_point(axis_name)
            
            if success:
                # Clear the input field
                dpg.set_value("point_axis_name", "")
                
                # Update any UI that shows saved axes
                self.tab_manager.refresh_all_lists()
                
                print("Successfully created and saved axis from selected point")
            else:
                print("Failed to create axis from selected point")
                
        except Exception as e:
            print(f"Error creating axis from point: {e}")

    def sync_clusters_to_all_tabs(self):
        """Synchronize cluster labels to all tabs through TabManager"""
        self.tab_manager.sync_clusters_to_all_tabs()
    
    def refresh_all_plots(self):
        """Refresh all plot data through TabManager"""
        self.tab_manager.refresh_all_plots()
    
    def update_cluster_labels(self, indices, new_cluster_id):
        """Update cluster labels through TabManager"""
        self.tab_manager.update_cluster_labels(indices, new_cluster_id)

    def create_highlight_theme(self, color):
        """Create a theme for highlighted points"""
        with dpg.theme() as theme_id:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [255, 255, 255, 255], category=dpg.mvThemeCat_Plots)  # White outline
                # Set marker style - larger and more prominent
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 10, category=dpg.mvThemeCat_Plots)  # Larger size
        return theme_id

    def create_similarity_highlight_theme(self, color, distance):
        """Create a theme for similarity highlighted points"""
        # Size based on similarity (smaller distance = larger point)
        size = max(8, int(15 - distance * 2))  # Scale size inversely with distance
        
        with dpg.theme() as theme_id:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [255, 255, 255, 255], category=dpg.mvThemeCat_Plots)  # White outline
                # Set marker style - size varies with similarity
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, size, category=dpg.mvThemeCat_Plots)
        return theme_id

    def highlight_points(self, match_indices):
        """Highlight specific datapoints in all plots"""
        if not match_indices:
            return
        
        # Define highlight colors (brighter versions)
        highlight_colors = [
            [255, 150, 150, 255],    # Bright Red
            [150, 200, 255, 255],    # Bright Blue
            [150, 255, 150, 255],    # Bright Green  
            [255, 200, 255, 255]     # Bright Purple
        ]
        
        # Update all tab plots
        for tab in self.tabs:
            # Ensure tab has synchronized labels
            tab.labels = self.labels.copy()
            
            # Refresh the plot data first 
            tab.add_scatter_data(self.clusters)
            
            # Then add highlighted points on top
            match_indices_set = set(match_indices)
            
            for i in range(len(self.clusters)):  # For each cluster
                # Get all points in this cluster that match
                cluster_mask = tab.labels == i
                cluster_indices = np.where(cluster_mask)[0]
                highlight_mask = np.array([idx in match_indices_set for idx in cluster_indices])
                
                if np.any(highlight_mask):
                    # Get the matching points in this cluster
                    matching_cluster_indices = cluster_indices[highlight_mask]
                    x_data = tab.X[matching_cluster_indices, 0].tolist()
                    y_data = tab.X[matching_cluster_indices, 1].tolist()
                    
                    # Add highlighted series with safe tag checking
                    highlight_tag = f"{tab.name}_highlight_{i}"
                    
                    dpg.add_scatter_series(
                        x_data, y_data,
                        label=f"Highlighted Cluster {i} ({len(x_data)} matches)",
                        parent=f"{tab.name}_y_axis",
                        tag=highlight_tag
                    )
                    
                    # Apply highlight theme
                    color_idx = i if i < len(highlight_colors) else -1
                    highlight_theme = self.create_highlight_theme(highlight_colors[color_idx])
                    dpg.bind_item_theme(highlight_tag, highlight_theme)

    def add_saved_axis(self, axis_info):
        """Add a saved axis to the global list"""
        try:
            # Add unique ID if not present
            if 'id' not in axis_info:
                axis_info['id'] = len(self.saved_axes)
            self.saved_axes.append(axis_info)
            print(f"Added saved axis: {axis_info['name']}")
        except Exception as e:
            print(f"Error adding saved axis: {e}")

    def remove_saved_axis(self, axis_id):
        """Remove a saved axis by ID"""
        self.saved_axes = [axis for axis in self.saved_axes if axis['id'] != axis_id]
        self.refresh_saved_axes_ui()
    
    def get_saved_axes(self):
        """Get all saved axes"""
        return self.saved_axes
    
    def create_custom_projection_tab(self, axis1_info, axis2_info):
        """Create a new tab with custom axis projection"""
        try:
            # Create axes array for KVPTab
            axes = np.array([axis1_info['vector'], axis2_info['vector']])
            
            # Create tab name
            tab_name = f"Custom: {axis1_info['name'][:20]}... vs {axis2_info['name'][:20]}..."
            
            # Create new KVPTab with custom axes
            custom_tab = KVPTab(tab_name, self.data_manager.X, self.data_manager.data, self.data_manager.labels, axes, self.data_manager)
            
            # Add to tabs list
            self.tabs.append(custom_tab)
            
            # Add the tab to the UI
            with dpg.mutex():
                # Get the tab bar
                if dpg.does_item_exist("tab_bar"):
                    dpg.push_container_stack("tab_bar")
                    custom_tab.setup_ui(self.clusters)
                    dpg.pop_container_stack()
            
            print(f"Created custom projection tab: {tab_name}")
            return custom_tab
            
        except Exception as e:
            print(f"Error creating custom projection tab: {e}")
            return None
    
    def refresh_saved_axes_ui(self):
        """Refresh the saved axes UI display"""
        # This will be called to update any UI that displays saved axes
        saved_axes_tag = "saved_axes_list"
        if dpg.does_item_exist(saved_axes_tag):
            # Clear existing items
            dpg.delete_item(saved_axes_tag, children_only=True)
            
            # Add current saved axes
            with dpg.mutex():
                dpg.push_container_stack(saved_axes_tag)
                
                if len(self.saved_axes) == 0:
                    dpg.add_text("No saved axes yet", color=[128, 128, 128])
                else:
                    for axis_info in self.saved_axes:
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"• {axis_info['name'][:40]}{'...' if len(axis_info['name']) > 40 else ''}")
                            dpg.add_button(
                                label="X",
                                width=20,
                                height=20,
                                callback=lambda s, a, u=axis_info['id']: self.remove_saved_axis(u)
                            )
                
                dpg.pop_container_stack()
        
        # Update combo boxes
        self.update_axis_combos()
    
    def update_axis_combos(self):
        """Update the axis selection combo boxes"""
        if len(self.saved_axes) == 0:
            items = ["Select saved axis..."]
        else:
            items = ["Select saved axis..."] + [axis['name'] for axis in self.saved_axes]
        
        # Update combo boxes if they exist
        for combo_tag in ["x_axis_combo", "y_axis_combo"]:
            if dpg.does_item_exist(combo_tag):
                dpg.configure_item(combo_tag, items=items)
        
        # Enable/disable create button based on selections
        self.check_projection_button_state()
    
    def check_projection_button_state(self):
        """Check if projection button should be enabled"""
        axis1_name = dpg.get_value("axis1_combo")
        axis2_name = dpg.get_value("axis2_combo")
        
        # Enable if both axes are selected and different
        enable = (axis1_name != "Select axis..." and 
                 axis2_name != "Select axis..." and 
                 axis1_name != axis2_name)
        
        if dpg.does_item_exist("create_projection_tab_btn"):
            dpg.configure_item("create_projection_tab_btn", enabled=enable)
    
    def _compute_vector_projection(self, from_text, to_text):
        """Compute vector projection from text pair (difference vector)"""
        try:
            # Generate embeddings for the words
            embeddings = encode([from_text, to_text])
            
            # Create difference vector (semantic direction)
            vector = embeddings[1] - embeddings[0]
            
            # Ensure it's 1D
            if vector.ndim > 1:
                vector = vector.flatten()
                
            return vector
            
        except Exception as e:
            print(f"Error computing vector projection for '{from_text}' → '{to_text}': {e}")
            return None
    
    def _compute_single_word_projection(self, text):
        """Compute single word projection"""
        try:
            # Generate embedding for the text
            vector = encode(text)
            
            # Ensure it's 1D
            if vector.ndim > 1:
                vector = vector.flatten()
                
            return vector
            
        except Exception as e:
            print(f"Error computing single word projection for '{text}': {e}")
            return None
    
    def get_saved_axis_data(self, axis_name):
        """Get projection data for a saved axis"""
        try:
            # Find the saved axis
            saved_axis = None
            for axis in self.saved_axes:
                if axis['name'] == axis_name:
                    saved_axis = axis
                    break
            
            if saved_axis is None:
                print(f"Saved axis '{axis_name}' not found")
                return None
            
            # Compute projection based on axis type
            if saved_axis['type'] == 'vector_projection':
                # Difference vector
                vector = self._compute_vector_projection(
                    saved_axis['from_text'], 
                    saved_axis['to_text']
                )
                if vector is not None:
                    return np.dot(self.data_manager.X, vector)
                    
            elif saved_axis['type'] == 'single_word':
                # Single word projection
                vector = self._compute_single_word_projection(saved_axis['word'])
                if vector is not None:
                    return np.dot(self.data_manager.X, vector)
            
            return None
            
        except Exception as e:
            print(f"Error getting saved axis data for '{axis_name}': {e}")
            return None
    
    def _get_metadata_column(self, column_name):
        """Get metadata column data"""
        try:
            if hasattr(self.data_manager, 'metadata') and self.data_manager.metadata is not None:
                if column_name in self.data_manager.metadata.columns:
                    return self.data_manager.metadata[column_name]
            return None
        except Exception as e:
            print(f"Error getting metadata column '{column_name}': {e}")
            return None

def main():
    """Main function to run the Dear PyGui t-SNE visualizer"""
    app = VisualizerDPG()
    # The app starts automatically in __init__ via setup_ui()

if __name__ == '__main__':
    main() 
