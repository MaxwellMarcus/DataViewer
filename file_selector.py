import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import os
from pathlib import Path

class FileSelector:
    def __init__(self, callback=None):
        """
        Initialize the file selector interface.
        
        Args:
            callback: Function to call when files are selected and validated
        """
        self.callback = callback
        self.embeddings_file = None
        self.metadata_file = None
        self.primary_metadata_column = "title"  # Default
        self.metadata_columns = []
        
    def show_file_selector(self):
        """Show the file selection dialog"""
        print( "showing file selector")
        with dpg.window(
            label="Data Visualization - File Selection", 
            tag="file_selector_window",
            no_close=True,
            no_collapse=True,
            no_resize=True,
            no_title_bar=True,
            no_move=True
        ):
            dpg.add_text("Welcome to Data Visualization Tool", color=[255, 255, 100])
            dpg.add_separator()
            
            # Instructions
            dpg.add_text("Please select your data files to get started:", wrap=550)
            dpg.add_spacer(height=10)
            
            # Embeddings file selection
            dpg.add_text("1. Embeddings File (.npy)", color=[200, 200, 255])
            dpg.add_text("   Select a numpy file containing your embeddings data", 
                        color=[150, 150, 150], wrap=550)
            
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    default_value=self.embeddings_file,
                    tag="embeddings_path",
                    width=400,
                    readonly=True,
                    hint="No file selected"
                )
                dpg.add_button(
                    label="Browse...",
                    callback=self.select_embeddings_file,
                    width=80
                )
            dpg.add_text("", tag="embeddings_status", color=[150, 150, 150])
            
            dpg.add_spacer(height=15)
            
            # Metadata file selection
            dpg.add_text("2. Metadata File (.csv, .json, .parquet)", color=[200, 255, 200])
            dpg.add_text("   Select a file containing metadata for your embeddings", 
                        color=[150, 150, 150], wrap=550)
            
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    default_value=self.metadata_file,
                    tag="metadata_path",
                    width=400,
                    readonly=True,
                    hint="No file selected"
                )
                dpg.add_button(
                    label="Browse...",
                    callback=self.select_metadata_file,
                    width=80
                )
            dpg.add_text("", tag="metadata_status", color=[150, 150, 150])
            
            dpg.add_spacer(height=15)
            
            # Primary metadata column selection
            dpg.add_text("3. Primary Text Column", color=[255, 200, 200])
            dpg.add_text("   Select which column contains the main text to display", 
                        color=[150, 150, 150], wrap=550)
            
            dpg.add_combo(
                default_value=self.primary_metadata_column,
                items=self.metadata_columns,
                tag="primary_column_combo",
                width=200,
                enabled=False,
                callback=self.on_primary_column_changed
            )
            
            dpg.add_spacer(height=20)
            
            # Status and validation
            dpg.add_text("Status:", color=[255, 255, 100])
            dpg.add_text("Please select both files to continue", 
                        tag="status_text", color=[255, 100, 100])
            
            dpg.add_spacer(height=20)
            dpg.add_separator()
            
            # Action buttons
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=300)
                dpg.add_button(
                    label="Load Example Data",
                    callback=self.load_example_data,
                    width=120,
                    tag="example_btn"
                )
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label="Continue",
                    callback=self.validate_and_continue,
                    width=100,
                    enabled=False,
                    tag="continue_btn"
                )
    
    def select_embeddings_file(self):
        """Open file dialog for embeddings file"""
        # Delete any existing dialog first
        if dpg.does_item_exist("embeddings_file_dialog"):
            dpg.delete_item("embeddings_file_dialog")
            
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.embeddings_file_callback,
            tag="embeddings_file_dialog",
            width=700,
            height=400,
            modal=True
        ):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".npy", color=[255, 255, 0, 255])
            dpg.add_file_extension(".npz", color=[255, 255, 0, 255])
        
        dpg.show_item("embeddings_file_dialog")
    
    def select_metadata_file(self):
        """Open file dialog for metadata file"""
        # Delete any existing dialog first
        if dpg.does_item_exist("metadata_file_dialog"):
            dpg.delete_item("metadata_file_dialog")
            
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.metadata_file_callback,
            tag="metadata_file_dialog",
            width=700,
            height=400,
            modal=True
        ):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".csv", color=[0, 255, 0, 255])
            dpg.add_file_extension(".json", color=[0, 255, 255, 255])
            dpg.add_file_extension(".parquet", color=[255, 0, 255, 255])
            dpg.add_file_extension(".xlsx", color=[0, 255, 0, 255])
        
        dpg.show_item("metadata_file_dialog")
    
    def embeddings_file_callback(self, sender, app_data):
        """Handle embeddings file selection"""
        print( app_data )
        file_path = list( app_data['selections'].values() )[ 0 ]
        self.embeddings_file = file_path
        
        # Update UI
        dpg.set_value("embeddings_path", file_path)
        
        # Validate file
        if self.validate_embeddings_file(file_path):
            dpg.set_value("embeddings_status", "✓ Valid embeddings file")
            dpg.configure_item("embeddings_status", color=[100, 255, 100])
        else:
            dpg.set_value("embeddings_status", "✗ Invalid embeddings file")
            dpg.configure_item("embeddings_status", color=[255, 100, 100])
            self.embeddings_file = None
        
        # Hide dialog instead of deleting
        if dpg.does_item_exist("embeddings_file_dialog"):
            dpg.hide_item("embeddings_file_dialog")
        
        self.update_status()
    
    def metadata_file_callback(self, sender, app_data):
        """Handle metadata file selection"""
        file_path = list( app_data['selections'].values() )[ 0 ]
        self.metadata_file = file_path
        
        # Update UI
        dpg.set_value("metadata_path", file_path)
        
        # Validate file and load column names
        if self.validate_metadata_file(file_path):
            dpg.set_value("metadata_status", "✓ Valid metadata file")
            dpg.configure_item("metadata_status", color=[100, 255, 100])
            self.load_metadata_columns(file_path)
        else:
            dpg.set_value("metadata_status", "✗ Invalid metadata file")
            dpg.configure_item("metadata_status", color=[255, 100, 100])
            self.metadata_file = None
            dpg.configure_item("primary_column_combo", items=[], enabled=False)
        
        # Hide dialog instead of deleting
        if dpg.does_item_exist("metadata_file_dialog"):
            dpg.hide_item("metadata_file_dialog")
            
        self.update_status()
    
    def validate_embeddings_file(self, file_path):
        """Validate that the embeddings file is valid"""
        try:
            if not os.path.exists(file_path):
                return False
            
            # Try to load the numpy file
            if file_path.endswith('.npy'):
                data = np.load(file_path)
                if len(data.shape) != 2:
                    print(f"Warning: Expected 2D array, got {data.shape}")
                    return False
                print(f"Embeddings shape: {data.shape}")
                return True
            elif file_path.endswith('.npz'):
                data = np.load(file_path)
                # For npz files, we'll need to handle this differently
                print(f"NPZ file contains: {list(data.keys())}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error validating embeddings file: {e}")
            return False
    
    def validate_metadata_file(self, file_path):
        """Validate that the metadata file is valid"""
        try:
            if not os.path.exists(file_path):
                return False
            
            # Try to load the metadata file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=5)  # Just read first few rows for validation
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, nrows=5)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                df = df.head(5)  # Limit to first few rows
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, nrows=5)
            else:
                return False
            
            print(f"Metadata columns: {df.columns.tolist()}")
            return True
            
        except Exception as e:
            print(f"Error validating metadata file: {e}")
            return False
    
    def load_metadata_columns(self, file_path):
        """Load column names from metadata file"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=1)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, nrows=1)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                df = df.head(1)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, nrows=1)
            else:
                return
            
            self.metadata_columns = df.columns.tolist()
            
            # Update combo box
            dpg.configure_item("primary_column_combo", 
                             items=self.metadata_columns, 
                             enabled=True)
            
            # Try to auto-select a good default column
            default_columns = ['title', 'text', 'content', 'description', 'caption']
            for col in default_columns:
                if col in self.metadata_columns:
                    dpg.set_value("primary_column_combo", col)
                    self.primary_metadata_column = col
                    break
            else:
                # If no good default found, select the first column
                if self.metadata_columns:
                    dpg.set_value("primary_column_combo", self.metadata_columns[0])
                    self.primary_metadata_column = self.metadata_columns[0]
                    
        except Exception as e:
            print(f"Error loading metadata columns: {e}")
    
    def on_primary_column_changed(self, sender, app_data):
        """Handle primary column selection change"""
        self.primary_metadata_column = app_data
        self.update_status()
    
    def update_status(self):

        """Update the status text and enable/disable continue button"""
        if self.embeddings_file and self.metadata_file and self.primary_metadata_column:
            dpg.set_value("status_text", "✓ Ready to load data")
            dpg.configure_item("status_text", color=[100, 255, 100])
            dpg.configure_item("continue_btn", enabled=True)
        elif self.embeddings_file and self.metadata_file:
            dpg.set_value("status_text", "Please select a primary text column")
            dpg.configure_item("status_text", color=[255, 255, 100])
            dpg.configure_item("continue_btn", enabled=False)
        else:
            missing = []
            if not self.embeddings_file:
                missing.append("embeddings file")
            if not self.metadata_file:
                missing.append("metadata file")
            
            dpg.set_value("status_text", f"Please select: {', '.join(missing)}")
            dpg.configure_item("status_text", color=[255, 100, 100])
            dpg.configure_item("continue_btn", enabled=False)
    
    def load_example_data(self):
        """Load example data if available"""
        # Look for example files in the current directory
        example_embeddings = None
        example_metadata = None
        
        # Common example file names
        embeddings_patterns = ['embeddings.npy', 'data.npy', 'vectors.npy', 'example_embeddings.npy']
        metadata_patterns = ['metadata.csv', 'data.csv', 'captions.csv', 'example_metadata.csv']
        
        for pattern in embeddings_patterns:
            if os.path.exists(pattern):
                example_embeddings = pattern
                break
        
        for pattern in metadata_patterns:
            if os.path.exists(pattern):
                example_metadata = pattern
                break
        
        if example_embeddings and example_metadata:
            self.embeddings_file = os.path.abspath(example_embeddings)
            self.metadata_file = os.path.abspath(example_metadata)
            
            dpg.set_value("embeddings_path", self.embeddings_file)
            dpg.set_value("metadata_path", self.metadata_file)
            
            # Validate and load
            if self.validate_embeddings_file(self.embeddings_file):
                dpg.set_value("embeddings_status", "✓ Valid embeddings file")
                dpg.configure_item("embeddings_status", color=[100, 255, 100])
            if self.validate_metadata_file(self.metadata_file):
                dpg.set_value("metadata_status", "✓ Valid metadata file")
                dpg.configure_item("metadata_status", color=[100, 255, 100])
                self.load_metadata_columns(self.metadata_file)
            
            self.update_status()
        else:
            dpg.set_value("status_text", "No example data found in current directory")
            dpg.configure_item("status_text", color=[255, 100, 100])
    
    def validate_and_continue(self):
        """Validate selections and continue to main app"""
        if not self.embeddings_file or not self.metadata_file or not self.primary_metadata_column:
            return
        
        # Final validation
        try:
            # Load and validate embeddings
            if self.embeddings_file.endswith('.npy'):
                embeddings = np.load(self.embeddings_file)
            elif self.embeddings_file.endswith('.npz'):
                npz_data = np.load(self.embeddings_file)
                # Try to find the embeddings array
                if 'embeddings' in npz_data:
                    embeddings = npz_data['embeddings']
                elif 'data' in npz_data:
                    embeddings = npz_data['data']
                else:
                    # Use the first array
                    embeddings = npz_data[list(npz_data.keys())[0]]
            
            # Load metadata
            if self.metadata_file.endswith('.csv'):
                metadata = pd.read_csv(self.metadata_file)
            elif self.metadata_file.endswith('.json'):
                metadata = pd.read_json(self.metadata_file)
            elif self.metadata_file.endswith('.parquet'):
                metadata = pd.read_parquet(self.metadata_file)
            elif self.metadata_file.endswith('.xlsx'):
                metadata = pd.read_excel(self.metadata_file)
            
            # Validate dimensions match
            if len(embeddings) != len(metadata):
                dpg.set_value("status_text", 
                             f"Error: Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) have different lengths")
                dpg.configure_item("status_text", color=[255, 100, 100])
                return
            
            # Validate primary column exists
            if self.primary_metadata_column not in metadata.columns:
                dpg.set_value("status_text", 
                             f"Error: Column '{self.primary_metadata_column}' not found in metadata")
                dpg.configure_item("status_text", color=[255, 100, 100])
                return
            
            # Success! Call callback and hide window
            if self.callback:
                self.callback(embeddings, metadata, self.primary_metadata_column)
            
            # Hide the file selector window instead of deleting it
            if dpg.does_item_exist("file_selector_window"):
                dpg.hide_item("file_selector_window")
            
        except Exception as e:
            dpg.set_value("status_text", f"Error loading data: {str(e)}")
            dpg.configure_item("status_text", color=[255, 100, 100])
            print(f"Error in validate_and_continue: {e}") 