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
from enhanced_file_selector import EnhancedFileSelector
import umap
from scatter_tabs import *


class VisualizerDPG:
    def __init__(self):
        """Initialize the application"""
        # Initialize basic attributes
        self.data_manager = None
        self.tab_manager = None
        self.current_search_matches = []
        self.current_similarity_matches = []
        self.custom_clusters = {}
        self.next_cluster_id = 4
        
        # Storage for selected files
        self.selected_embeddings = None
        self.selected_metadata = None
        self.selected_primary_column = None
        
        # Start with file selector
        self.run_application()

    def run_application(self):
        """Run the complete application flow"""
        # Step 1: Show file selector
        if not self.show_file_selector():
            return  # User cancelled or error occurred
            
        # Step 2: Start main application with selected data
        self.start_main_application()

    def show_file_selector(self):
        """Show enhanced file selector and return True if files were selected"""
        dpg.create_context()
        
        file_selector = EnhancedFileSelector(callback=self.on_files_selected)
        file_selector.show_file_selector()
        
        # Create viewport and start DearPyGui with just file selector
        dpg.create_viewport(title="Data Visualization Setup", width=1000, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("enhanced_file_selector", True)
        dpg.start_dearpygui()
        dpg.destroy_context()
        
        # Return True if files were selected
        return self.selected_embeddings is not None
        
    def on_files_selected(self, embeddings, metadata, primary_column):
        """Handle when files are selected and validated"""
        print("Files selected successfully!")
        
        # Store the selected data
        self.selected_embeddings = embeddings
        self.selected_metadata = metadata
        self.selected_primary_column = primary_column
        
        # Close the file selector
        dpg.stop_dearpygui()
        
    def start_main_application(self):
        """Start the main application with selected data"""
        print("Loading selected data...")
                
        # Create new context for main application
        dpg.create_context()
        
        # Load the actual data from file paths
        try:
            # Load embeddings data
            if self.selected_embeddings.endswith('.npy'):
                embeddings_data = np.load(self.selected_embeddings)
                print(f"Loaded embeddings from {self.selected_embeddings}: shape {embeddings_data.shape}")
            else:
                raise ValueError(f"Unsupported embeddings file format: {self.selected_embeddings}")
            
            # Load metadata
            if self.selected_metadata.endswith('.csv'):
                metadata_data = pd.read_csv(self.selected_metadata)
            elif self.selected_metadata.endswith('.json'):
                metadata_data = pd.read_json(self.selected_metadata)
            elif self.selected_metadata.endswith('.parquet'):
                metadata_data = pd.read_parquet(self.selected_metadata)
            elif self.selected_metadata.endswith(('.xlsx', '.xls')):
                metadata_data = pd.read_excel(self.selected_metadata)
            else:
                raise ValueError(f"Unsupported metadata file format: {self.selected_metadata}")
            
            print(f"Loaded metadata from {self.selected_metadata}: shape {metadata_data.shape}")
            print(f"Metadata columns: {list(metadata_data.columns)}")
            
            # Validate data consistency
            if len(embeddings_data) != len(metadata_data):
                raise ValueError(f"Data mismatch: {len(embeddings_data)} embeddings vs {len(metadata_data)} metadata rows")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create error dialog
            with dpg.window(label="Error Loading Data", modal=True, width=400, height=200):
                dpg.add_text(f"Failed to load data files:")
                dpg.add_text(f"Error: {str(e)}")
                dpg.add_separator()
                dpg.add_button(label="OK", callback=lambda: dpg.stop_dearpygui())
            dpg.setup_dearpygui()
            dpg.show_viewport()
            dpg.start_dearpygui()
            dpg.destroy_context()
            return
        
        # Initialize DataManager with actual loaded data
        self.data_manager = DataManager(
            embeddings_data, 
            metadata_data, 
            primary_metadata_column=self.selected_primary_column
        )
        
        # Initialize TabManager
        self.tab_manager = TabManager(self.data_manager)
        
        # Initialize tabs through TabManager
        self.tab_manager.initialize_tabs()

        # Setup main UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main application UI"""
        with dpg.window(label="Data Visualization Tool", width=1200, height=800, pos=(0, 0), 
                       tag="main_window", no_resize=True, no_title_bar=True, no_collapse=True, no_scrollbar=True):
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

        dpg.create_viewport(title="Data Visualization Tool", width=1200, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        print("Application loaded successfully!")
        print("Lasso functionality available:")
        print("1. Hold Shift and drag on the plot to select rectangular areas")
        print("2. Hold Ctrl and click to create polygon selections")
        print("3. Double-click to clear lasso selection")
        print("4. Use 'Create Cluster from Selection' to create clusters from selected points")
        print("\nPress Ctrl+C or close the window to exit...")
        
        dpg.start_dearpygui()
        dpg.destroy_context()

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
                dpg.add_input_text(width=150, tag="substring_value")
        
        # Set callback for method change
        dpg.set_item_callback("cluster_method", self.update_cluster_params_callback)
        
        # Cluster action buttons
        with dpg.group(horizontal=True):
            dpg.add_button(label="Create Cluster", callback=self.create_new_cluster_callback)
            dpg.add_button(label="Reset Clusters", callback=self.reset_clusters_callback)
            dpg.add_button(label="Clear All", callback=self.clear_all_clusters_callback)
        
        dpg.add_separator()
        
        # Information and management section
        dpg.add_text("Cluster Information", color=[200, 200, 200])
        with dpg.group(horizontal=True):
            dpg.add_button(label="Show Info", callback=self.show_cluster_info_callback)
            dpg.add_button(label="Show History", callback=self.show_clustering_history_callback)
        
        dpg.add_separator()
        
        # Lasso selection section
        dpg.add_text("LASSO SELECTION", color=[255, 200, 100])
        dpg.add_text("Use mouse to create selections on plots", color=[200, 200, 200])
        with dpg.group(horizontal=True):
            dpg.add_button(label="Create Cluster from Selection", callback=self.create_cluster_from_lasso_callback)
            dpg.add_button(label="Clear Selection", callback=self.clear_lasso_selection_callback)
        
        dpg.add_separator()
        
        # Point selection section for axis creation
        dpg.add_text("POINT SELECTION", color=[255, 200, 100])
        dpg.add_text("Click on points to create custom axes", color=[200, 200, 200])
        
        # Selected point info
        dpg.add_text("Selected Point: None", tag="selected_point_info", color=[150, 150, 150])
        
        # Axis creation controls
        with dpg.group(horizontal=True):
            dpg.add_input_text(hint="Custom axis name...", width=200, tag="axis_name_input")
            dpg.add_button(label="Create Axis", callback=self.create_axis_from_point_callback, enabled=False, tag="create_axis_btn")
        
        dpg.add_separator()
        
        # Response text area
        dpg.add_text("Response:", color=[255, 255, 100])
        dpg.add_text("Ready to create clusters...", tag="response_text", wrap=400, color=[200, 200, 200])

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
                
            self.highlight_points(self.data_manager.search_results)
            dpg.set_value("response_text", f"Highlighted {len(self.data_manager.search_results)} search results")
            
        except Exception as e:
            dpg.set_value("response_text", f"Error highlighting search results: {str(e)}")
            print(f"Error in search_and_highlight: {str(e)}")

    def clear_highlights_callback(self, sender=None, app_data=None):
        """Wrapper for clearing highlights"""
        try:
            # Clear highlights on all tabs
            for tab in self.tab_manager.tabs:
                tab.clear_highlights()
            
            dpg.set_value("response_text", "Cleared all highlights")
            
        except Exception as e:
            dpg.set_value("response_text", f"Error clearing highlights: {str(e)}")
            print(f"Error in clear_highlights: {str(e)}")

    def create_cluster_from_search_callback(self, sender=None, app_data=None):
        """Wrapper for creating cluster from search results"""
        try:
            if not self.current_search_matches:
                dpg.set_value("response_text", "No search results to create cluster from. Please search first.")
                return
            
            # Create custom cluster from search matches
            cluster_id = self.next_cluster_id
            cluster_name = f"Search_Cluster_{cluster_id}"
            
            # Update labels for the matched indices
            self.update_cluster_labels(self.current_search_matches, cluster_id)
            
            # Store custom cluster info
            self.custom_clusters[cluster_id] = {
                "name": cluster_name,
                "indices": self.current_search_matches.copy()
            }
            
            # Increment next cluster ID
            self.next_cluster_id += 1
            
            # Update all visualizations
            self.sync_clusters_to_all_tabs()
            self.refresh_all_plots()
            
            dpg.set_value("response_text", f"Created cluster '{cluster_name}' with {len(self.current_search_matches)} points")
            
            # Clear search matches
            self.current_search_matches = []
            
        except Exception as e:
            dpg.set_value("response_text", f"Error creating cluster from search: {str(e)}")
            print(f"Error in create_cluster_from_search: {str(e)}")

    def similarity_search_callback(self, sender=None, app_data=None):
        """Wrapper for similarity search"""
        try:
            query_text = dpg.get_value("similarity_input")
            threshold = dpg.get_value("similarity_threshold")
            
            if not query_text:
                dpg.set_value("response_text", "Error: Please enter text for similarity search")
                return
            
            # Perform similarity search through DataManager
            results = self.data_manager.similarity_search(query_text, threshold)
            
            if results is not None:
                self.current_similarity_matches = results
                dpg.set_value("response_text", f"Found {len(results)} similar items with threshold {threshold}")
            else:
                dpg.set_value("response_text", "Error performing similarity search")
                
        except Exception as e:
            dpg.set_value("response_text", f"Error in similarity search: {str(e)}")
            print(f"Error in similarity_search: {str(e)}")

    def clear_similarity_callback(self, sender=None, app_data=None):
        """Wrapper for clearing similarity highlights"""
        try:
            # Clear similarity highlights on all tabs
            for tab in self.tab_manager.tabs:
                tab.clear_similarity_highlights()
            
            self.current_similarity_matches = []
            dpg.set_value("response_text", "Cleared similarity highlights")
            
        except Exception as e:
            dpg.set_value("response_text", f"Error clearing similarity highlights: {str(e)}")
            print(f"Error in clear_similarity: {str(e)}")

    def create_cluster_from_similarity_callback(self, sender=None, app_data=None):
        """Wrapper for creating cluster from similarity results"""
        try:
            if not self.current_similarity_matches:
                dpg.set_value("response_text", "No similarity results to create cluster from. Please search first.")
                return
            
            # Create custom cluster from similarity matches
            cluster_id = self.next_cluster_id
            cluster_name = f"Similarity_Cluster_{cluster_id}"
            
            # Update labels for the matched indices
            self.update_cluster_labels(self.current_similarity_matches, cluster_id)
            
            # Store custom cluster info
            self.custom_clusters[cluster_id] = {
                "name": cluster_name,
                "indices": self.current_similarity_matches.copy()
            }
            
            # Increment next cluster ID
            self.next_cluster_id += 1
            
            # Update all visualizations
            self.sync_clusters_to_all_tabs()
            self.refresh_all_plots()
            
            dpg.set_value("response_text", f"Created cluster '{cluster_name}' with {len(self.current_similarity_matches)} points")
            
            # Clear similarity matches
            self.current_similarity_matches = []
            
        except Exception as e:
            dpg.set_value("response_text", f"Error creating cluster from similarity: {str(e)}")
            print(f"Error in create_cluster_from_similarity: {str(e)}")

    def create_cluster_from_lasso_callback(self, sender=None, app_data=None):
        """Wrapper for creating cluster from lasso selection"""
        try:
            # Get current active tab
            current_tab = self.tab_manager.get_current_tab()
            if not current_tab:
                dpg.set_value("response_text", "Error: No active tab found")
                return
            
            # Get lasso selection from current tab
            selected_indices = current_tab.get_lasso_selection()
            if not selected_indices:
                dpg.set_value("response_text", "No lasso selection found. Please make a selection first.")
                return
            
            # Create custom cluster from lasso selection
            cluster_id = self.next_cluster_id
            cluster_name = f"Lasso_Cluster_{cluster_id}"
            
            # Update labels for the selected indices
            self.update_cluster_labels(selected_indices, cluster_id)
            
            # Store custom cluster info
            self.custom_clusters[cluster_id] = {
                "name": cluster_name,
                "indices": selected_indices.copy()
            }
            
            # Increment next cluster ID
            self.next_cluster_id += 1
            
            # Update all visualizations
            self.sync_clusters_to_all_tabs()
            self.refresh_all_plots()
            
            dpg.set_value("response_text", f"Created cluster '{cluster_name}' with {len(selected_indices)} points")
            
            # Clear lasso selection
            current_tab.clear_lasso_selection()
            
        except Exception as e:
            dpg.set_value("response_text", f"Error creating cluster from lasso: {str(e)}")
            print(f"Error in create_cluster_from_lasso: {str(e)}")

    def clear_lasso_selection_callback(self, sender=None, app_data=None):
        """Wrapper for clearing lasso selection"""
        try:
            current_tab = self.tab_manager.get_current_tab()
            if current_tab:
                current_tab.clear_lasso_selection()
                dpg.set_value("response_text", "Cleared lasso selection")
        except Exception as e:
            dpg.set_value("response_text", f"Error clearing lasso selection: {str(e)}")

    def create_axis_from_point_callback(self, sender=None, app_data=None):
        """Wrapper for creating axis from selected point"""
        try:
            # Get the custom axis name
            axis_name = dpg.get_value("axis_name_input")
            if not axis_name:
                dpg.set_value("response_text", "Please enter a name for the custom axis")
                return
            
            # Use DataManager to create and save the axis
            success = self.data_manager.create_and_save_axis_from_selected_point(axis_name)
            
            if success:
                dpg.set_value("response_text", f"Created and saved axis: {axis_name}")
                # Clear the input and disable button
                dpg.set_value("axis_name_input", "")
                dpg.configure_item("create_axis_btn", enabled=False)
                dpg.set_value("selected_point_info", "Selected Point: None")
            else:
                dpg.set_value("response_text", "Error: No point selected or failed to create axis")
                
        except Exception as e:
            dpg.set_value("response_text", f"Error creating axis from point: {str(e)}")
            print(f"Error in create_axis_from_point: {str(e)}")

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def sync_clusters_to_all_tabs(self):
        """Sync cluster information to all tabs"""
        self.tab_manager.sync_clusters_to_all_tabs()

    def refresh_all_plots(self):
        """Refresh all plot visualizations"""
        self.tab_manager.refresh_all_plots()

    def update_cluster_labels(self, indices, new_cluster_id):
        """Update cluster labels for specified indices"""
        self.data_manager.update_cluster_labels(indices, new_cluster_id)

    def create_highlight_theme(self, color):
        """Create a highlight theme for points"""
        with dpg.theme() as highlight_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 8, category=dpg.mvThemeCat_Plots)
        return highlight_theme

    def create_similarity_highlight_theme(self, color, distance):
        """Create a similarity highlight theme with distance-based sizing"""
        # Scale marker size based on distance (closer = larger)
        max_distance = 2.0  # Maximum expected distance
        marker_size = max(4, int(12 * (1 - min(distance, max_distance) / max_distance)))
        
        with dpg.theme() as highlight_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [255, 255, 255, 255], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, marker_size, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerWeight, 2, category=dpg.mvThemeCat_Plots)
        return highlight_theme

    def highlight_points(self, match_indices):
        """Highlight points across all tabs"""
        try:
            highlight_color = [255, 255, 0, 200]  # Bright yellow
            highlight_theme = self.create_highlight_theme(highlight_color)
            
            for tab in self.tab_manager.tabs:
                # Clear existing highlights
                tab.clear_highlights()
                
                # Add new highlights
                for i, idx in enumerate(match_indices):
                    if idx < len(tab.x_data) and idx < len(tab.y_data):
                        highlight_tag = f"{tab.name}_highlight_{i}"
                        
                        # Add highlight point
                        dpg.add_scatter_series(
                            [tab.x_data[idx]], 
                            [tab.y_data[idx]], 
                            parent=f"{tab.name}_y_axis",
                            tag=highlight_tag
                        )
                        
                        # Apply highlight theme
                        dpg.bind_item_theme(highlight_tag, highlight_theme)
                        
        except Exception as e:
            print(f"Error highlighting points: {e}")

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

if __name__ == '__main__':
    main() 
