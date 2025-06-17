#!/usr/bin/env python3
"""
TabManager class for handling tab creation and management
"""

import numpy as np
import dearpygui.dearpygui as dpg
from projection import get_projection
from encode import encode
import time
from scatter_tabs import *
import matplotlib.path as mpath
import pandas as pd

class TabManager:
    """
    Manages tab creation and dynamic tab functionality for the visualization application.
    """
    
    def __init__(self, data_manager):
        """
        Initialize the TabManager.
        
        Args:
            data_manager: Reference to the DataManager instance
        """
        self.data_manager = data_manager
        
        # Track all tabs (both initial and created)
        self.tabs = []
        self.created_tabs = []
        self.tab_counter = 0
        self.current_tab_index = 0
        
        # Tab ID mapping for UI
        self.tab_id_map = {}
        
        # Lasso selection state
        self.lasso_enabled = False
        self.lasso_selection = set()  # Set of selected point indices
        
        print("TabManager initialized")
    
    def initialize_tabs(self):
        """Initialize the standard tabs"""
        
        print("Generating UMAP...")
        umap_tab = UMAPTab("UMAP", self.data_manager, self.data_manager.umap_data)
        
        print("Creating tabs...")
        self.tabs = [
            umap_tab,
            PCATab("PCA", self.data_manager),
            NGonTab("NGON", self.data_manager),
        ]
        
        print(f"Initialized {len(self.tabs)} tabs")
        return self.tabs
    
    def add_tab(self, tab):
        """Add a tab to the tabs list"""
        self.tabs.append(tab)
        return len(self.tabs) - 1
    
    def get_tabs(self):
        """Get all tabs"""
        return self.tabs
    
    def get_current_tab(self):
        """Get the current active tab"""
        if 0 <= self.current_tab_index < len(self.tabs):
            return self.tabs[self.current_tab_index]
        return None
    
    def set_current_tab_index(self, index):
        """Set the current active tab index"""
        if 0 <= index < len(self.tabs):
            self.current_tab_index = index
            return True
        return False
    
    def get_current_tab_index(self):
        """Get the current tab index"""
        return self.current_tab_index
    
    def sync_clusters_to_all_tabs(self):
        """Synchronize cluster labels to all tabs"""
        for tab in self.tabs:
            if hasattr(tab, 'update_labels'):
                tab.update_labels(self.data_manager.labels)
    
    def refresh_all_plots(self):
        """Refresh all plot data across all tabs"""
        try:
            for tab in self.tabs:
                if hasattr(tab, 'add_scatter_data'):
                    tab.add_scatter_data(self.get_clusters())
                # Also refresh the legend to reflect current cluster state
                if hasattr(tab, 'refresh_custom_legend'):
                    tab.refresh_custom_legend()
        except Exception as e:
            print(f"Error refreshing plots: {e}")
    
    def get_clusters(self):
        """Get clusters - for now return a simple list based on unique cluster IDs"""
        unique_clusters = np.unique(self.data_manager.labels)
        return [f"Cluster {i}" for i in range(len(unique_clusters))]
    
    def setup_kvp_creation_controls(self):
        """Setup comprehensive custom axes tab creation controls UI"""
        self.setup_custom_axes_creation_controls()
        
        dpg.add_separator()
        
        # Tab Management Section
        dpg.add_text("TAB MANAGEMENT", color=[255, 200, 100])
        dpg.add_separator()
        
        # Show created tabs
        dpg.add_text("Created Tabs:", color=[180, 180, 180])
        dpg.add_text("None", tag="created_tabs_list", wrap=580)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Refresh Tab List", callback=self.refresh_tab_list, width=150)
            dpg.add_button(label="Clear All Custom Tabs", callback=self.clear_custom_tabs, width=150)
        
        dpg.add_separator()
        
        # Status text
        dpg.add_text("Status:", color=[200, 200, 200])
        dpg.add_text("Ready to create new tabs", tag="tab_status_text", wrap=580)

    def setup_dimred_creation_controls(self):
        """Setup dimensionality reduction tab creation controls UI"""
        dpg.add_text("DIMENSIONALITY REDUCTION TABS", color=[255, 200, 100])
        dpg.add_separator()
        
        # Tab type selection
        dpg.add_text("Select Tab Type:", color=[200, 200, 200])
        dpg.add_combo(
            items=["t-SNE", "PCA", "UMAP"],
            default_value="t-SNE",
            width=200,
            tag="dimred_tab_type"
        )
        
        dpg.add_separator()
        
        # t-SNE Parameters
        with dpg.collapsing_header(label="t-SNE Parameters", default_open=True, tag="tsne_params"):
            dpg.add_text("Perplexity:")
            dpg.add_slider_int(
                min_value=5,
                max_value=100,
                default_value=30,
                width=200,
                tag="tsne_perplexity"
            )
            dpg.add_text("Learning Rate:")
            dpg.add_slider_float(
                min_value=10.0,
                max_value=1000.0,
                default_value=200.0,
                width=200,
                tag="tsne_learning_rate"
            )
            dpg.add_text("Max Iterations:")
            dpg.add_slider_int(
                min_value=250,
                max_value=2000,
                default_value=1000,
                width=200,
                tag="tsne_max_iter"
            )
        
        # PCA Parameters
        with dpg.collapsing_header(label="PCA Parameters", default_open=False, tag="pca_params"):
            dpg.add_text("Number of Components:")
            dpg.add_slider_int(
                min_value=2,
                max_value=10,
                default_value=2,
                width=200,
                tag="pca_n_components"
            )
            dpg.add_checkbox(label="Whiten", tag="pca_whiten", default_value=False)
        
        # UMAP Parameters
        with dpg.collapsing_header(label="UMAP Parameters", default_open=False, tag="umap_params"):
            dpg.add_text("Number of Neighbors:")
            dpg.add_slider_int(
                min_value=2,
                max_value=100,
                default_value=15,
                width=200,
                tag="umap_n_neighbors"
            )
            dpg.add_text("Minimum Distance:")
            dpg.add_slider_float(
                min_value=0.0,
                max_value=1.0,
                default_value=0.1,
                width=200,
                tag="umap_min_dist"
            )
            dpg.add_text("Metric:")
            dpg.add_combo(
                items=["euclidean", "manhattan", "cosine", "correlation"],
                default_value="euclidean",
                width=200,
                tag="umap_metric"
            )
        
        dpg.add_separator()
        
        # Create button
        dpg.add_button(
            label="Create Dimensionality Reduction Tab",
            callback=self.create_dimred_tab,
            width=300
        )
        
        dpg.add_separator()
        
        # Status text for dimred tabs
        dpg.add_text("Status:", color=[200, 200, 200])
        dpg.add_text("Ready to create dimensionality reduction tabs", tag="dimred_status_text", wrap=580)

    def create_kvp_tab_difference(self, sender=None, app_data=None):
        """Create KVP tab using difference vectors (from → to)"""
        try:
            # Get input values
            axis_1_from = dpg.get_value("tab_axis_1_from")
            axis_1_to = dpg.get_value("tab_axis_1_to")
            axis_2_from = dpg.get_value("tab_axis_2_from")
            axis_2_to = dpg.get_value("tab_axis_2_to")
            
            if not all([axis_1_from, axis_1_to, axis_2_from, axis_2_to]):
                dpg.set_value("tab_status_text", "Error: Please fill in all text fields")
                return
            
            dpg.set_value("tab_status_text", f"Creating projection for:\nAxis 1: {axis_1_from} → {axis_1_to}\nAxis 2: {axis_2_from} → {axis_2_to}")
            
            # Generate embeddings for the words
            print("Encoding text for custom axes...")
            embeddings = encode([axis_1_from, axis_1_to, axis_2_from, axis_2_to])
            
            # Create difference vectors (semantic directions)
            axis_1_vector = embeddings[1] - embeddings[0]
            axis_2_vector = embeddings[3] - embeddings[2]


            # Create the tab
            tab_name = f"KVP_{axis_1_from}-{axis_1_to}_{axis_2_from}-{axis_2_to}"
            tab_info = {
                'name': tab_name,
                'type': 'KVP_Difference',
                'axes': [axis_1_vector, axis_2_vector],
                'axis_labels': [f"{axis_1_from} → {axis_1_to}", f"{axis_2_from} → {axis_2_to}"],
                'timestamp': time.time()
            }
            
            success = self._create_kvp_tab(tab_info)
            
            if success:
                # Clear input fields
                dpg.set_value("tab_axis_1_from", "")
                dpg.set_value("tab_axis_1_to", "")
                dpg.set_value("tab_axis_2_from", "")
                dpg.set_value("tab_axis_2_to", "")
                
                dpg.set_value("tab_status_text", f"Successfully created KVP tab: {tab_name}")
                self.refresh_tab_list()
            else:
                dpg.set_value("tab_status_text", "Error: Failed to create tab")
                
        except None as e:
            dpg.set_value("tab_status_text", f"Error creating KVP tab: {str(e)}")
            print(f"Error in create_kvp_tab_difference: {str(e)}")

    def create_kvp_tab_single(self, sender=None, app_data=None):
        """Create KVP tab using single word axes"""
        try:
            # Get input values
            axis_1_word = dpg.get_value("tab_axis_1_single")
            axis_2_word = dpg.get_value("tab_axis_2_single")
            
            if not all([axis_1_word, axis_2_word]):
                dpg.set_value("tab_status_text", "Error: Please fill in both axis words")
                return
            
            dpg.set_value("tab_status_text", f"Creating projection for:\nAxis 1: {axis_1_word}\nAxis 2: {axis_2_word}")
            
            # Generate embeddings for the words
            print("Encoding text for single word axes...")
            axes = encode([axis_1_word, axis_2_word])
            
            
            # Create the tab
            tab_name = f"KVP_{axis_1_word}_{axis_2_word}"
            tab_info = {
                'name': tab_name,
                'type': 'KVP_Single',
                'axis_labels': [axis_1_word, axis_2_word],
                'axes': axes,
                'timestamp': time.time()
            }
            
            success = self._create_kvp_tab(tab_info)
            
            if success:
                # Clear input fields
                dpg.set_value("tab_axis_1_single", "")
                dpg.set_value("tab_axis_2_single", "")
                
                dpg.set_value("tab_status_text", f"Successfully created KVP tab: {tab_name}")
                self.refresh_tab_list()
            else:
                dpg.set_value("tab_status_text", "Error: Failed to create tab")
                
        except None as e:
            dpg.set_value("tab_status_text", f"Error creating KVP tab: {str(e)}")
            print(f"Error in create_kvp_tab_single: {str(e)}")

    def create_dimred_tab(self, sender=None, app_data=None):
        """Create a dimensionality reduction tab (t-SNE, PCA, or UMAP)"""
        try:
            # Get selected tab type
            tab_type = dpg.get_value("dimred_tab_type")
            
            dpg.set_value("dimred_status_text", f"Creating {tab_type} tab...")
            
            if tab_type == "t-SNE":
                self._create_tsne_tab()
            elif tab_type == "PCA":
                self._create_pca_tab()
            elif tab_type == "UMAP":
                self._create_umap_tab()
            else:
                dpg.set_value("dimred_status_text", f"Error: Unknown tab type {tab_type}")
                
        except None as e:
            dpg.set_value("dimred_status_text", f"Error creating {tab_type} tab: {str(e)}")
            print(f"Error in create_dimred_tab: {str(e)}")

    def _create_tsne_tab(self):
        """Create a t-SNE tab with custom parameters"""
        try:
            from sklearn.manifold import TSNE
            
            # Get parameters
            perplexity = dpg.get_value("tsne_perplexity")
            learning_rate = dpg.get_value("tsne_learning_rate")
            max_iter = dpg.get_value("tsne_max_iter")
            
            # Create unique tab name
            tab_name = f"t-SNE_p{perplexity}_lr{learning_rate:.0f}_i{max_iter}"
            
            # Create t-SNE transformation
            print(f"Computing t-SNE with perplexity={perplexity}, learning_rate={learning_rate}, max_iter={max_iter}")
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=42
            )
            tsne_data = tsne.fit_transform(self.data_manager.X)
            
            # Create new t-SNE tab
            new_tab = TSNETab(
                name=tab_name,
                data_manager=self.data_manager,
                tsne_data=tsne_data,
                tsne_params={
                    'perplexity': perplexity,
                    'learning_rate': learning_rate,
                    'max_iter': max_iter
                }
            )
            
            # Add to tabs list
            tab_index = self.add_tab(new_tab)
            
            # Add tab to main interface
            with dpg.tab(label=new_tab.name, tag=f"{new_tab.name}_tab", parent="main_tab_bar"):
                new_tab.setup_content(self.get_clusters())
            
            # Update tab mapping
            self.tab_id_map[f"{new_tab.name}_tab"] = tab_index
            
            dpg.set_value("dimred_status_text", f"Successfully created t-SNE tab: {tab_name}")
            print(f"Successfully created t-SNE tab: {tab_name}")
            
        except Exception as e:
            dpg.set_value("dimred_status_text", f"Error creating t-SNE tab: {str(e)}")
            print(f"Error in _create_tsne_tab: {str(e)}")

    def _create_pca_tab(self):
        """Create a PCA tab with custom parameters"""
        try:
            from sklearn.decomposition import PCA
            
            # Get parameters
            n_components = dpg.get_value("pca_n_components")
            whiten = dpg.get_value("pca_whiten")
            
            # Create unique tab name
            tab_name = f"PCA_{n_components}comp{'_whiten' if whiten else ''}"
            
            # Create PCA transformation
            print(f"Computing PCA with n_components={n_components}, whiten={whiten}")
            pca = PCA(
                n_components=n_components,
                whiten=whiten,
                random_state=42
            )
            pca_data = pca.fit_transform(self.data_manager.X)
            
            # Create new PCA tab
            new_tab = PCATab(
                name=tab_name,
                data_manager=self.data_manager,
                pca_data=pca_data[:, :2],  # Use first 2 components for visualization
                pca_model=pca
            )
            
            # Add to tabs list
            tab_index = self.add_tab(new_tab)
            
            # Add tab to main interface
            with dpg.tab(label=new_tab.name, tag=f"{new_tab.name}_tab", parent="main_tab_bar"):
                new_tab.setup_content(self.get_clusters())
            
            # Update tab mapping
            self.tab_id_map[f"{new_tab.name}_tab"] = tab_index
            
            dpg.set_value("dimred_status_text", f"Successfully created PCA tab: {tab_name}")
            print(f"Successfully created PCA tab: {tab_name}")
            
        except Exception as e:
            dpg.set_value("dimred_status_text", f"Error creating PCA tab: {str(e)}")
            print(f"Error in _create_pca_tab: {str(e)}")

    def _create_umap_tab(self):
        """Create a UMAP tab with custom parameters"""
        try:
            import umap
            
            # Get parameters
            n_neighbors = dpg.get_value("umap_n_neighbors")
            min_dist = dpg.get_value("umap_min_dist")
            metric = dpg.get_value("umap_metric")
            
            # Create unique tab name
            tab_name = f"UMAP_n{n_neighbors}_d{min_dist:.2f}_{metric}"
            
            # Create UMAP transformation
            print(f"Computing UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=42
            )
            umap_data = reducer.fit_transform(self.data_manager.X)
            
            # Create new UMAP tab
            new_tab = UMAPTab(
                name=tab_name,
                data_manager=self.data_manager,
                umap_data=umap_data,
                umap_params={
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    'metric': metric
                }
            )
            
            # Add to tabs list
            tab_index = self.add_tab(new_tab)
            
            # Add tab to main interface
            with dpg.tab(label=new_tab.name, tag=f"{new_tab.name}_tab", parent="main_tab_bar"):
                new_tab.setup_content(self.get_clusters())
            
            # Update tab mapping
            self.tab_id_map[f"{new_tab.name}_tab"] = tab_index
            
            dpg.set_value("dimred_status_text", f"Successfully created UMAP tab: {tab_name}")
            print(f"Successfully created UMAP tab: {tab_name}")
            
        except Exception as e:
            dpg.set_value("dimred_status_text", f"Error creating UMAP tab: {str(e)}")
            print(f"Error in _create_umap_tab: {str(e)}")

    def _create_kvp_tab(self, tab_info):
        """Internal method to create a tab from tab_info"""
        try:

            # Create new KVP tab with the projection
            new_tab = KVPTab(
                name=tab_info['name'],
                data_manager=self.data_manager,
                axes=tab_info['axes'],
                axis_labels=tab_info.get('axis_labels', [])
            )
            
            # Add to tabs list
            tab_index = self.add_tab(new_tab)
            
            # Add tab to main interface
            with dpg.tab(label=new_tab.name, tag=f"{new_tab.name}_tab", parent="main_tab_bar"):
                new_tab.setup_content(self.get_clusters())
            
            # Update tab mapping
            self.tab_id_map[f"{new_tab.name}_tab"] = tab_index
            
            # Store tab info
            self.created_tabs.append(tab_info)
            self.tab_counter += 1
            
            print(f"Successfully created tab: {tab_info['name']}")
            return True
            
        except None as e:
            print(f"Error creating tab: {str(e)}")
            return False

    def refresh_tab_list(self, sender=None, app_data=None):
        """Refresh the display of created tabs"""
        try:
            if not self.created_tabs:
                dpg.set_value("created_tabs_list", "None")
            else:
                tab_list = ""
                for i, tab_info in enumerate(self.created_tabs):
                    tab_list += f"{i+1}. {tab_info['name']} ({tab_info['type']})\n"
                dpg.set_value("created_tabs_list", tab_list.strip())
                
        except Exception as e:
            print(f"Error refreshing tab list: {str(e)}")

    def clear_custom_tabs(self, sender=None, app_data=None):
        """Clear all custom tabs (this would need integration with the main UI)"""
        try:
            # Note: This is a placeholder - full implementation would need to
            # remove tabs from the UI and clean up references
            self.created_tabs.clear()
            self.refresh_tab_list()
            dpg.set_value("tab_status_text", "Custom tabs cleared (UI removal not implemented)")
            
        except Exception as e:
            dpg.set_value("tab_status_text", f"Error clearing tabs: {str(e)}")
            print(f"Error in clear_custom_tabs: {str(e)}")

    def get_created_tabs(self):
        """Get list of created tabs"""
        return self.created_tabs.copy()

    def get_tab_count(self):
        """Get number of created tabs"""
        return len(self.created_tabs)
    
    def setup_main_tab_ui(self):
        """Setup the main tab bar UI with all tabs"""
        
        with dpg.tab_bar(tag="main_tab_bar"):
            for tab in self.tabs:
                tab.setup_ui(self.get_clusters())
    
    def on_tab_change(self, sender=None, app_data=None):
        """Handle tab change events"""
        try:
            # Check if we have the tab ID in our mapping
            if app_data in self.tab_id_map:
                self.current_tab_index = self.tab_id_map[app_data]
                print(f"Tab changed to index: {self.current_tab_index} ({self.tabs[self.current_tab_index].name})")
                return True
            else:
                print(f"Tab ID {app_data} not found in mapping, current_tab_index: {self.current_tab_index}")
                return False
        except Exception as e:
            print(f"Error in tab change detection: {e}")
            return False
    
    def get_current_axes_info(self):
        """Get current axes information for the active tab"""
        current_tab = self.get_current_tab()
        if not current_tab:
            return {
                "type": "Unknown",
                "x_axis": "Unknown X",
                "y_axis": "Unknown Y",
                "changeable": False,
                "parameters": {}
            }
        
        tab_type = type(current_tab).__name__
        
        if tab_type == "PCATab":
            return {
                "type": "PCA",
                "x_axis": "First Principal Component",
                "y_axis": "Second Principal Component",
                "changeable": True,
                "parameters": {
                    "components": 2,
                    "explained_variance": getattr(current_tab.pca, 'explained_variance_ratio_', [0.0, 0.0])
                }
            }
        elif tab_type == "TSNETab":
            return {
                "type": "t-SNE",
                "x_axis": "t-SNE Dimension 1",
                "y_axis": "t-SNE Dimension 2",
                "changeable": False,
                "parameters": {
                    "perplexity": 30,
                    "note": "t-SNE axes are not interchangeable"
                }
            }
        elif tab_type == "UMAPTab":
            return {
                "type": "UMAP",
                "x_axis": "UMAP Dimension 1", 
                "y_axis": "UMAP Dimension 2",
                "changeable": False,
                "parameters": {
                    "n_neighbors": 15,
                    "min_dist": 0.1,
                    "note": "UMAP axes are not interchangeable"
                }
            }
        elif tab_type == "KVPTab":
            return {
                "type": "Vector Projection",
                "x_axis": "Custom Axis 1",
                "y_axis": "Custom Axis 2", 
                "changeable": True,
                "parameters": {
                    "axes": getattr(current_tab, 'axes', None)
                }
            }
        
        return {
            "type": "Unknown",
            "x_axis": "Unknown X",
            "y_axis": "Unknown Y",
            "changeable": False,
            "parameters": {}
        }
    
    def update_cluster_labels(self, indices, new_cluster_id):
        """Update cluster labels for specific indices and sync across all tabs"""
        for idx in indices:
            if idx < len(self.data_manager.labels):
                self.data_manager.labels[idx] = new_cluster_id
        
        # Update colors for new clusters
        self.data_manager.update_cluster_colors()
        
        # Sync to all tabs
        self.sync_clusters_to_all_tabs()
        
                # Refresh all plots
        self.refresh_all_plots()
    
    def setup_custom_axes_creation_controls(self):
        """Setup streamlined custom axes creation controls UI"""
        dpg.add_text("CUSTOM AXES TABS", color=[255, 200, 100])
        dpg.add_separator()
        
        dpg.add_text("Create custom visualization by selecting axis types and sources", color=[200, 200, 200])
        dpg.add_separator()
        
        # X-Axis Configuration
        dpg.add_text("X-Axis Configuration", color=[255, 255, 100])
        with dpg.group(horizontal=True):
            dpg.add_text("Type:")
            dpg.add_combo(
                items=["Metadata Column", "Saved Axis", "Custom Axis"],
                default_value="Metadata Column",
                width=150,
                tag="x_axis_type",
                callback=self.update_axis_options
            )
        
        # X-Axis specific options (will be shown/hidden based on type)
        with dpg.group(tag="x_metadata_group", show=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Column:")
                dpg.add_combo(
                    items=self.get_numeric_metadata_columns(),
                    width=200,
                    tag="x_metadata_column"
                )
        
        with dpg.group(tag="x_saved_group", show=False):
            with dpg.group(horizontal=True):
                dpg.add_text("Saved Axis:")
                dpg.add_combo(
                    items=self.get_saved_axes_list(),
                    width=200,
                    tag="x_saved_axis"
                )
        
        with dpg.group(tag="x_custom_group", show=False):
            dpg.add_text("Custom Axis Type:")
            dpg.add_combo(
                items=["Difference Vector (A → B)", "Single Word", "Custom Phrase"],
                default_value="Difference Vector (A → B)",
                width=200,
                tag="x_custom_type",
                callback=self.update_custom_inputs
            )
            
            with dpg.group(tag="x_difference_inputs", show=True):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(hint="From (e.g., 'happy')", width=100, tag="x_from")
                    dpg.add_text("→")
                    dpg.add_input_text(hint="To (e.g., 'sad')", width=100, tag="x_to")
            
            with dpg.group(tag="x_word_inputs", show=False):
                dpg.add_input_text(hint="Word (e.g., 'emotion')", width=200, tag="x_word")
            
            with dpg.group(tag="x_phrase_inputs", show=False):
                dpg.add_input_text(hint="Phrase/sentence", width=200, tag="x_phrase")
        
        dpg.add_separator()
        
        # Y-Axis Configuration
        dpg.add_text("Y-Axis Configuration", color=[255, 255, 100])
        with dpg.group(horizontal=True):
            dpg.add_text("Type:")
            dpg.add_combo(
                items=["Metadata Column", "Saved Axis", "Custom Axis"],
                default_value="Metadata Column",
                width=150,
                tag="y_axis_type",
                callback=self.update_axis_options
            )
        
        # Y-Axis specific options
        with dpg.group(tag="y_metadata_group", show=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Column:")
                dpg.add_combo(
                    items=self.get_numeric_metadata_columns(),
                    width=200,
                    tag="y_metadata_column"
                )
        
        with dpg.group(tag="y_saved_group", show=False):
            with dpg.group(horizontal=True):
                dpg.add_text("Saved Axis:")
                dpg.add_combo(
                    items=self.get_saved_axes_list(),
                    width=200,
                    tag="y_saved_axis"
                )
        
        with dpg.group(tag="y_custom_group", show=False):
            dpg.add_text("Custom Axis Type:")
            dpg.add_combo(
                items=["Difference Vector (A → B)", "Single Word", "Custom Phrase"],
                default_value="Difference Vector (A → B)",
                width=200,
                tag="y_custom_type",
                callback=self.update_custom_inputs
            )
            
            with dpg.group(tag="y_difference_inputs", show=True):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(hint="From (e.g., 'young')", width=100, tag="y_from")
                    dpg.add_text("→")
                    dpg.add_input_text(hint="To (e.g., 'old')", width=100, tag="y_to")
            
            with dpg.group(tag="y_word_inputs", show=False):
                dpg.add_input_text(hint="Word (e.g., 'technology')", width=200, tag="y_word")
            
            with dpg.group(tag="y_phrase_inputs", show=False):
                dpg.add_input_text(hint="Phrase/sentence", width=200, tag="y_phrase")
        
        dpg.add_separator()
        
        # Projection option (only shown when both axes are custom/saved)
        dpg.add_checkbox(
            label="Use plane projection (orthogonalize axes)", 
            tag="use_plane_projection",
            default_value=False,
            show=False
        )
        dpg.add_text(
            "Plane projection creates orthogonal axes for better separation", 
            tag="projection_help", 
            wrap=400, 
            color=[150, 150, 150],
            show=False
        )
        
        dpg.add_separator()
        
        # Tab creation controls
        dpg.add_input_text(hint="Tab name (optional)", width=300, tag="unified_tab_name")
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Create Custom Tab", callback=self.create_unified_custom_tab, width=200)
            dpg.add_button(label="Save Custom Axes", callback=self.save_current_custom_axes, width=150)
            dpg.add_button(label="Refresh Lists", callback=self.refresh_all_lists, width=100)
        
        dpg.add_separator()
        
        # Preview/Status
        dpg.add_text("Tab Preview:", color=[180, 180, 180])
        dpg.add_text("Configure axes above to see preview", tag="unified_preview", wrap=560, color=[150, 150, 150])

    def _refresh_main_tab_ui(self):
        """Refresh the main tab UI to include new tabs"""
        try:
            # Get the main tab bar
            if dpg.does_item_exist("main_tab_bar"):
                # Add the new tab to the existing tab bar
                new_tab = self.tabs[-1]  # Get the most recently added tab
                
                with dpg.mutex():
                    dpg.push_container_stack("main_tab_bar")
                    new_tab.setup_ui(self.get_clusters())
                    dpg.pop_container_stack()
                
                print(f"Added new tab '{new_tab.name}' to main UI")
            else:
                print("Main tab bar not found - cannot add new tab to UI")
                
        except Exception as e:
            print(f"Error refreshing main tab UI: {e}")
            import traceback
            traceback.print_exc()
    
    def update_axis_options(self, sender=None, app_data=None):
        """Update axis-specific options based on selected type"""
        try:
            # Determine which axis was changed
            axis = "x" if sender == "x_axis_type" else "y"
            axis_type = dpg.get_value(f"{axis}_axis_type")
            
            # Hide all option groups for this axis
            dpg.configure_item(f"{axis}_metadata_group", show=False)
            dpg.configure_item(f"{axis}_saved_group", show=False)
            dpg.configure_item(f"{axis}_custom_group", show=False)
            
            # Show the appropriate group
            if axis_type == "Metadata Column":
                dpg.configure_item(f"{axis}_metadata_group", show=True)
            elif axis_type == "Saved Axis":
                dpg.configure_item(f"{axis}_saved_group", show=True)
            elif axis_type == "Custom Axis":
                dpg.configure_item(f"{axis}_custom_group", show=True)
            
            # Update projection option visibility
            self.update_projection_option_visibility()
            self.update_preview()
            
        except Exception as e:
            print(f"Error updating axis options: {e}")
    
    def update_custom_inputs(self, sender=None, app_data=None):
        """Update custom input fields based on selected custom type"""
        try:
            # Determine which axis was changed
            axis = "x" if "x_custom_type" in sender else "y"
            custom_type = dpg.get_value(f"{axis}_custom_type")
            
            # Hide all input groups for this axis
            dpg.configure_item(f"{axis}_difference_inputs", show=False)
            dpg.configure_item(f"{axis}_word_inputs", show=False)
            dpg.configure_item(f"{axis}_phrase_inputs", show=False)
            
            # Show the appropriate input group
            if custom_type == "Difference Vector (A → B)":
                dpg.configure_item(f"{axis}_difference_inputs", show=True)
            elif custom_type == "Single Word":
                dpg.configure_item(f"{axis}_word_inputs", show=True)
            elif custom_type == "Custom Phrase":
                dpg.configure_item(f"{axis}_phrase_inputs", show=True)
            
            self.update_preview()
            
        except Exception as e:
            print(f"Error updating custom inputs: {e}")
    
    def update_projection_option_visibility(self):
        """Show/hide projection option based on axis types"""
        try:
            x_type = dpg.get_value("x_axis_type")
            y_type = dpg.get_value("y_axis_type")
            
            # Show projection option only when both axes are Custom or Saved
            show_projection = (x_type in ["Custom Axis", "Saved Axis"] and 
                             y_type in ["Custom Axis", "Saved Axis"])
            
            dpg.configure_item("use_plane_projection", show=show_projection)
            dpg.configure_item("projection_help", show=show_projection)
            
        except Exception as e:
            print(f"Error updating projection option visibility: {e}")
    
    def update_preview(self):
        """Update the tab preview text"""
        try:
            x_type = dpg.get_value("x_axis_type")
            y_type = dpg.get_value("y_axis_type")
            
            x_desc = self.get_axis_description("x", x_type)
            y_desc = self.get_axis_description("y", y_type)
            
            use_projection = dpg.get_value("use_plane_projection") if dpg.does_item_exist("use_plane_projection") else False
            projection_text = " (with plane projection)" if use_projection else ""
            
            preview_text = f"X-Axis: {x_desc}\nY-Axis: {y_desc}{projection_text}"
            dpg.set_value("unified_preview", preview_text)
            
        except Exception as e:
            print(f"Error updating preview: {e}")
    
    def get_axis_description(self, axis, axis_type):
        """Get description for an axis based on its type and current values"""
        try:
            if axis_type == "Metadata Column":
                column = dpg.get_value(f"{axis}_metadata_column")
                return f"Metadata: {column}"
            
            elif axis_type == "Saved Axis":
                saved_axis = dpg.get_value(f"{axis}_saved_axis")
                return f"Saved: {saved_axis}"
            
            elif axis_type == "Custom Axis":
                custom_type = dpg.get_value(f"{axis}_custom_type")
                
                if custom_type == "Difference Vector (A → B)":
                    from_val = dpg.get_value(f"{axis}_from")
                    to_val = dpg.get_value(f"{axis}_to")
                    if from_val and to_val:
                        return f"Custom: {from_val} → {to_val}"
                    else:
                        return "Custom: Difference Vector (incomplete)"
                
                elif custom_type == "Single Word":
                    word = dpg.get_value(f"{axis}_word")
                    if word:
                        return f"Custom: {word}"
                    else:
                        return "Custom: Single Word (incomplete)"
                
                elif custom_type == "Custom Phrase":
                    phrase = dpg.get_value(f"{axis}_phrase")
                    if phrase:
                        return f"Custom: {phrase[:30]}{'...' if len(phrase) > 30 else ''}"
                    else:
                        return "Custom: Phrase (incomplete)"
            
            return "Not configured"
            
        except Exception as e:
            print(f"Error getting axis description: {e}")
            return "Error"
    
    def create_unified_custom_tab(self, sender=None, app_data=None):
        """Create a custom tab using the unified interface"""
        try:
            # Get axis configurations
            x_type = dpg.get_value("x_axis_type")
            y_type = dpg.get_value("y_axis_type")
            use_projection = dpg.get_value("use_plane_projection") if dpg.does_item_exist("use_plane_projection") else False
            
            # Get axis data
            x_data, x_vector, x_label = self.get_axis_data("x", x_type)
            y_data, y_vector, y_label = self.get_axis_data("y", y_type)
            
            if x_data is None or y_data is None:
                print("Error: Could not retrieve axis data")
                return
            
            # Generate tab name
            tab_name = dpg.get_value("unified_tab_name")
            if not tab_name.strip():
                tab_name = f"Custom_{x_label.replace(' ', '_')}_vs_{y_label.replace(' ', '_')}"
            
            # Use plane projection if requested and both axes have vectors
            if use_projection and x_vector is not None and y_vector is not None:
                projection_data = get_projection(self.data_manager.X, x_vector, y_vector)
                x_data = projection_data[:, 0]
                y_data = projection_data[:, 1]
                tab_name += "_projected"
            
            # Create the tab
            new_tab = CustomEncodedTab(
                tab_name,
                self.data_manager,
                x_data,
                y_data,
                axis_labels=[x_label, y_label]
            )
            
            # Add to tabs
            self.add_tab(new_tab)
            self.created_tabs.append({
                'name': tab_name,
                'type': f'Unified Custom ({x_type} + {y_type})',
                'x_axis': x_label,
                'y_axis': y_label,
                'projection': use_projection
            })
            
            # Refresh UI
            self._refresh_main_tab_ui()
            self.refresh_tab_list()
            
            print(f"Created unified custom tab: {tab_name}")
            
            # Clear inputs
            self.clear_unified_inputs()
            
        except Exception as e:
            print(f"Error creating unified custom tab: {e}")
            import traceback
            traceback.print_exc()
    
    def get_axis_data(self, axis, axis_type):
        """Get data, vector, and label for an axis"""
        try:
            if axis_type == "Metadata Column":
                column = dpg.get_value(f"{axis}_metadata_column")
                data = self._get_metadata_column_data(column.replace(" (categorical)", ""))
                return data, None, column
            
            elif axis_type == "Saved Axis":
                saved_axis = dpg.get_value(f"{axis}_saved_axis")
                data = self._get_saved_axis_data(saved_axis)
                # Get the vector for projection if needed
                vector = self.get_saved_axis_vector(saved_axis)
                return data, vector, saved_axis
            
            elif axis_type == "Custom Axis":
                custom_type = dpg.get_value(f"{axis}_custom_type")
                
                if custom_type == "Difference Vector (A → B)":
                    from_val = dpg.get_value(f"{axis}_from")
                    to_val = dpg.get_value(f"{axis}_to")
                    if not from_val or not to_val:
                        return None, None, None
                    
                    vector = self._compute_difference_vector(from_val, to_val)
                    if vector is not None:
                        data = np.dot(self.data_manager.X, vector)
                        return data, vector, f"{from_val} → {to_val}"
                
                elif custom_type == "Single Word":
                    word = dpg.get_value(f"{axis}_word")
                    if not word:
                        return None, None, None
                    
                    vector = self._compute_single_vector(word)
                    if vector is not None:
                        data = np.dot(self.data_manager.X, vector)
                        return data, vector, word
                
                elif custom_type == "Custom Phrase":
                    phrase = dpg.get_value(f"{axis}_phrase")
                    if not phrase:
                        return None, None, None
                    
                    vector = self._compute_single_vector(phrase)
                    if vector is not None:
                        data = np.dot(self.data_manager.X, vector)
                        label = phrase[:30] + "..." if len(phrase) > 30 else phrase
                        return data, vector, label
            
            return None, None, None
            
        except Exception as e:
            print(f"Error getting axis data for {axis}: {e}")
            return None, None, None
    
    def get_saved_axis_vector(self, axis_name):
        """Get the vector for a saved axis"""
        try:
            axis_info = self.data_manager.get_saved_axis_by_name(axis_name)
            if axis_info:
                return axis_info.get('vector', None)
            return None
        except Exception as e:
            print(f"Error getting saved axis vector: {e}")
            return None
    
    def save_current_custom_axes(self, sender=None, app_data=None):
        """Save the current custom axes configuration with vectors"""
        try:
            saved_count = 0
            
            for axis in ["x", "y"]:
                axis_type = dpg.get_value(f"{axis}_axis_type")
                
                if axis_type == "Custom Axis":
                    custom_type = dpg.get_value(f"{axis}_custom_type")
                    
                    if custom_type == "Difference Vector (A → B)":
                        from_val = dpg.get_value(f"{axis}_from")
                        to_val = dpg.get_value(f"{axis}_to")
                        
                        if from_val and to_val:
                            # Compute and save the vector
                            vector = self._compute_difference_vector(from_val, to_val)
                            if vector is not None:
                                axis_data = {
                                    'name': f"{from_val}_to_{to_val}",
                                    'type': 'difference_vector',
                                    'from_text': from_val,
                                    'to_text': to_val,
                                    'vector': vector
                                }
                                if self.data_manager.add_saved_axis(axis_data):
                                    saved_count += 1
                    
                    elif custom_type == "Single Word":
                        word = dpg.get_value(f"{axis}_word")
                        
                        if word:
                            # Compute and save the vector
                            vector = self._compute_single_vector(word)
                            if vector is not None:
                                axis_data = {
                                    'name': word,
                                    'type': 'single_word',
                                    'word': word,
                                    'vector': vector
                                }
                                if self.data_manager.add_saved_axis(axis_data):
                                    saved_count += 1
                    
                    elif custom_type == "Custom Phrase":
                        phrase = dpg.get_value(f"{axis}_phrase")
                        
                        if phrase:
                            # Compute and save the vector
                            vector = self._compute_single_vector(phrase)
                            if vector is not None:
                                phrase_name = phrase[:20] + "..." if len(phrase) > 20 else phrase
                                axis_data = {
                                    'name': phrase_name,
                                    'type': 'custom_phrase',
                                    'phrase': phrase,
                                    'vector': vector
                                }
                                if self.data_manager.add_saved_axis(axis_data):
                                    saved_count += 1
            
            if saved_count > 0:
                self.refresh_all_lists()
                print(f"Saved {saved_count} custom axes with vectors")
            else:
                print("No custom axes to save")
            
        except Exception as e:
            print(f"Error saving current custom axes: {e}")
    
    def refresh_all_lists(self, sender=None, app_data=None):
        """Refresh all dropdown lists"""
        try:
            # Refresh metadata columns
            metadata_columns = self.get_numeric_metadata_columns()
            for combo_tag in ["x_metadata_column", "y_metadata_column"]:
                if dpg.does_item_exist(combo_tag):
                    dpg.configure_item(combo_tag, items=metadata_columns)
                    if metadata_columns and metadata_columns[0] != "No numeric columns available":
                        dpg.set_value(combo_tag, metadata_columns[0])
            
            # Refresh saved axes
            saved_axes = self.get_saved_axes_list()
            for combo_tag in ["x_saved_axis", "y_saved_axis"]:
                if dpg.does_item_exist(combo_tag):
                    dpg.configure_item(combo_tag, items=saved_axes)
                    if saved_axes and saved_axes[0] != "No saved axes available":
                        dpg.set_value(combo_tag, saved_axes[0])
            
            print("Refreshed all dropdown lists")
            
        except Exception as e:
            print(f"Error refreshing lists: {e}")
    
    def clear_unified_inputs(self):
        """Clear all input fields in the unified interface"""
        try:
            # Clear text inputs
            input_tags = [
                "x_from", "x_to", "x_word", "x_phrase",
                "y_from", "y_to", "y_word", "y_phrase",
                "unified_tab_name"
            ]
            
            for tag in input_tags:
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, "")
            
            # Reset checkboxes
            if dpg.does_item_exist("use_plane_projection"):
                dpg.set_value("use_plane_projection", False)
            
            # Update preview
            self.update_preview()
            
        except Exception as e:
            print(f"Error clearing unified inputs: {e}")

    def get_numeric_metadata_columns(self):
        """Get list of numeric metadata columns available for axes"""
        try:
            if hasattr(self.data_manager, 'metadata') and self.data_manager.metadata is not None:
                numeric_columns = []
                for col in self.data_manager.metadata.columns:
                    if col != 'text':  # Skip the text column
                        # Check if column is numeric or can be converted
                        try:
                            pd.to_numeric(self.data_manager.metadata[col], errors='raise')
                            numeric_columns.append(col)
                        except (ValueError, TypeError):
                            # Try to see if it's categorical with reasonable number of categories
                            unique_vals = self.data_manager.metadata[col].nunique()
                            if unique_vals <= 50:  # Reasonable for categorical encoding
                                numeric_columns.append(f"{col} (categorical)")
                
                return numeric_columns if numeric_columns else ["No numeric columns available"]
            else:
                return ["No metadata available"]
        except Exception as e:
            print(f"Error getting numeric metadata columns: {e}")
            return ["Error loading columns"]
    
    def get_saved_axes_list(self):
        """Get list of saved custom axes"""
        try:
            saved_axes_names = self.data_manager.get_saved_axis_names()
            if saved_axes_names:
                return saved_axes_names
            else:
                return ["No saved axes available"]
        except Exception as e:
            print(f"Error getting saved axes: {e}")
            return ["Error loading saved axes"]

    def _compute_difference_vector(self, from_text, to_text):
        """Compute difference vector from text pair"""
        try:
            # Encode both texts
            from_embedding = encode(from_text)
            to_embedding = encode(to_text)
            
            # Handle single text vs batch
            if from_embedding.ndim > 1:
                from_embedding = from_embedding.flatten()
            if to_embedding.ndim > 1:
                to_embedding = to_embedding.flatten()
            
            # Compute difference vector (to - from)
            difference_vector = to_embedding - from_embedding
            
            # Normalize the vector
            norm = np.linalg.norm(difference_vector)
            if norm > 0:
                difference_vector = difference_vector / norm
            
            return difference_vector
        except Exception as e:
            print(f"Error computing difference vector: {e}")
            return None
    
    def _compute_single_vector(self, text):
        """Compute single vector from text"""
        try:
            # Encode the text
            embedding = encode(text)
            
            # Handle single text vs batch
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # Normalize the vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            print(f"Error computing single vector: {e}")
            return None
    
    def _get_metadata_column_data(self, column_name):
        """Get processed data from a metadata column"""
        try:
            if not hasattr(self.data_manager, 'metadata') or self.data_manager.metadata is None:
                return None
            
            if column_name not in self.data_manager.metadata.columns:
                print(f"Column '{column_name}' not found in metadata")
                return None
            
            column_data = self.data_manager.metadata[column_name]
            
            # Try to convert to numeric
            try:
                numeric_data = pd.to_numeric(column_data, errors='raise')
                return numeric_data.values
            except (ValueError, TypeError):
                # Handle categorical data
                unique_vals = column_data.nunique()
                if unique_vals <= 50:  # Reasonable for categorical
                    # Create numeric encoding
                    categories = column_data.unique()
                    category_map = {cat: i for i, cat in enumerate(categories)}
                    encoded_data = column_data.map(category_map)
                    print(f"Encoded categorical column '{column_name}' with {len(categories)} categories")
                    return encoded_data.values
                else:
                    print(f"Column '{column_name}' has too many unique values ({unique_vals}) for categorical encoding")
                    return None
                    
        except Exception as e:
            print(f"Error processing metadata column '{column_name}': {e}")
            return None
    
    def _get_saved_axis_data(self, axis_name):
        """Get projection data from a saved axis"""
        try:
            return self.data_manager.get_saved_axis_data(axis_name)
        except Exception as e:
            print(f"Error getting saved axis data for '{axis_name}': {e}")
            return None

    