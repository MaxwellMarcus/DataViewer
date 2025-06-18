import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import os
import threading
from sample_models import get_available_models, run_model
from preprocess import apply_preprocessing
from load_dataset import load_dataset_smart, validate_source

class EnhancedFileSelector:
    def __init__(self, callback=None):
        self.callback = callback
        self.mode = "file_selection"
        
        # File mode
        self.embeddings_file = None
        self.metadata_file = None
        self.primary_metadata_column = "title"
        self.metadata_columns = []
        
        # Dataset mode
        self.source_type = "huggingface"
        self.source_path = ""
        self.selected_model_type = "Simple"  # Default fallback
        self.selected_model_name = "random-embeddings"
        
        # New: Column selection state
        self.available_columns = []
        self.selected_columns = []
        self.dataset_preview = None
        
        # New: Batch processing state
        self.batch_size = 100  # Default batch size
        
        # New: Hugging Face authentication state
        self.hf_token = ""
        self.hf_authenticated = False
        
        # New: Custom model weights
        self.use_custom_model = False
        self.custom_model_repo = ""
        
    def show_file_selector(self):
        """Show the enhanced file selector"""
        # Check for existing HF authentication
        self.check_existing_hf_auth()
        
        with dpg.window(
            label="Data Visualization Setup",
            tag="enhanced_file_selector",
            width=800,
            height=700,
            no_resize=True,
            modal=True
        ):
            # Title
            dpg.add_text("Data Visualization Setup", color=[70, 130, 180])
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Mode selection
            dpg.add_text("Choose setup method:")
            dpg.add_radio_button(
                items=["Load Existing Files", "Process Dataset with AI"],
                default_value=0,
                callback=self.on_mode_changed,
                horizontal=False,
                tag="mode_selector"
            )
            
            dpg.add_separator()
            dpg.add_spacer(height=20)
            
            # Content area
            with dpg.child_window(height=450, tag="content_area"):
                self.create_file_mode_ui()
    
    def on_mode_changed(self, sender, app_data):
        """Handle mode change"""
        self.mode = "file_selection" if app_data == 0 else "dataset_processing"
        
        # Clear content and rebuild with proper parent context
        dpg.delete_item("content_area", children_only=True)
        
        # Push the content_area as parent context before adding items
        dpg.push_container_stack("content_area")
        
        try:
            if self.mode == "file_selection":
                self.create_file_mode_ui()
            else:
                self.create_dataset_mode_ui()
        finally:
            # Always pop the container stack
            dpg.pop_container_stack()
    
    def create_file_mode_ui(self):
        """Create file selection UI"""
        dpg.add_text("Load Existing Files")
        dpg.add_separator()
        dpg.add_spacer(height=10)
        
        # Embeddings file
        dpg.add_text("1. Embeddings File (.npy)")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="embeddings_path", width=400, readonly=True)
            dpg.add_button(label="Browse", callback=self.select_embeddings_file)
        dpg.add_text("", tag="embeddings_status")
        
        dpg.add_spacer(height=15)
        
        # Metadata file
        dpg.add_text("2. Metadata File (.csv, .json, etc.)")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="metadata_path", width=400, readonly=True)
            dpg.add_button(label="Browse", callback=self.select_metadata_file)
        dpg.add_text("", tag="metadata_status")
        
        dpg.add_spacer(height=15)
        
        # Column selection
        dpg.add_text("3. Primary Text Column")
        dpg.add_combo(
            tag="primary_column_combo",
            items=self.metadata_columns,
            width=200,
            enabled=False,
            callback=self.on_primary_column_changed
        )
        
        dpg.add_spacer(height=30)
        
        # Buttons
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Load Example Data",
                callback=self.load_example_data
            )
            dpg.add_spacer(width=200)
            dpg.add_button(
                label="Continue",
                callback=self.continue_file_mode,
                enabled=False,
                tag="continue_file_btn"
            )
    
    def create_dataset_mode_ui(self):
        """Create dataset processing UI"""
        dpg.add_text("Process Dataset with AI Model")
        dpg.add_separator()
        dpg.add_spacer(height=10)
        
        # Data source
        dpg.add_text("Data Source:")
        dpg.add_combo(
            items=["Hugging Face Dataset", "URL", "Local File"],
            default_value="Hugging Face Dataset",
            callback=self.on_source_type_changed,
            tag="source_type"
        )
        
        dpg.add_text("Path/URL:")
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="source_path",
                width=400,
                hint="e.g., imdb, squad",
                callback=self.on_source_path_changed
            )
            dpg.add_button(label="Validate", callback=self.validate_source)
        dpg.add_text("", tag="source_status")
        
        dpg.add_spacer(height=15)
        
        # Hugging Face Authentication Section
        dpg.add_text("🤗 Hugging Face Authentication:", color=[255, 200, 100])
        with dpg.group(horizontal=True):
            dpg.add_text("HF Token:")
            dpg.add_input_text(tag="hf_token", width=200, password=True, hint="hf_...", callback=self.on_hf_token_changed)
            dpg.add_button(label="Login", callback=self.hf_login)
            dpg.add_button(label="Check Status", callback=self.check_hf_auth)
        dpg.add_text("Not authenticated", tag="hf_status", color=[255, 100, 100])
        
        dpg.add_spacer(height=15)
        
        # Model selection
        dpg.add_text("AI Model:")
        available_models = get_available_models()
        
        # Ensure we have valid defaults
        if available_models:
            first_model_type = list(available_models.keys())[0]
            self.selected_model_type = first_model_type
            self.selected_model_name = available_models[first_model_type]["models"][0]
        
        dpg.add_combo(
            items=list(available_models.keys()),
            default_value=self.selected_model_type,
            callback=self.on_model_type_changed,
            tag="model_type"
        )
        
        dpg.add_combo(
            items=available_models[self.selected_model_type]["models"],
            default_value=self.selected_model_name,
            callback=self.on_model_name_changed,
            tag="model_name"
        )
        
        # Custom Model Weights Section
        dpg.add_spacer(height=10)
        dpg.add_checkbox(label="Use Custom Model Weights", tag="use_custom_model", callback=self.on_custom_model_toggle)
        
        with dpg.group(tag="custom_model_group", show=False):
            dpg.add_text("Custom Model Repository:", color=[200, 200, 200])
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="custom_model_repo", 
                    width=300, 
                    hint="e.g., username/my-fine-tuned-model",
                    callback=self.on_custom_model_repo_changed
                )
                dpg.add_button(label="Validate Repo", callback=self.validate_custom_model)
            dpg.add_text("", tag="custom_model_status")
        
        dpg.add_spacer(height=15)
        
        # Processing options
        dpg.add_text("Processing Options:")
        with dpg.group(horizontal=True):
            dpg.add_text("Text Column:")
            dpg.add_input_text(tag="text_column", width=150, hint="text")
            
            dpg.add_text("Max Samples:")
            dpg.add_input_int(tag="max_samples", default_value=1000, width=100)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Batch Size:")
            dpg.add_input_int(tag="batch_size", default_value=100, width=100, min_value=10, max_value=1000, callback=self.on_batch_size_changed)
            dpg.add_text("(for model processing)")
        
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Clean Text", default_value=True, tag="clean_text")
            dpg.add_checkbox(label="Truncate Text", default_value=True, tag="truncate_text")
        
        dpg.add_spacer(height=15)
        
        # Column selection section
        dpg.add_text("Metadata Columns:")
        dpg.add_text("Select which columns to save in metadata:", color=[180, 180, 180])
        
        with dpg.child_window(width=400, height=120, tag="column_selection_window"):
            dpg.add_text("Load dataset preview to see available columns", tag="column_selection_placeholder")
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select All Columns", callback=self.select_all_columns)
            dpg.add_button(label="Clear Selection", callback=self.clear_column_selection)
        
        dpg.add_spacer(height=20)
        
        # Buttons
        with dpg.group(horizontal=True):
            dpg.add_button(label="Preview Dataset", callback=self.preview_dataset)
            dpg.add_spacer(width=100)
            dpg.add_button(label="Process & Generate", callback=self.process_dataset)
        
        dpg.add_spacer(height=10)
        dpg.add_text("", tag="processing_status")
        dpg.add_progress_bar(tag="progress_bar", width=400, show=False)
    
    # Event handlers
    def on_source_type_changed(self, sender, app_data):
        type_map = {"Hugging Face Dataset": "huggingface", "URL": "url", "Local File": "file"}
        self.source_type = type_map[app_data]
    
    def on_source_path_changed(self, sender, app_data):
        self.source_path = app_data
    
    def on_model_type_changed(self, sender, app_data):
        self.selected_model_type = app_data
        available_models = get_available_models()
        models = available_models[app_data]["models"]
        
        # Check if the combo still exists before updating
        if dpg.does_item_exist("model_name"):
            dpg.configure_item("model_name", items=models, default_value=models[0])
            self.selected_model_name = models[0]
    
    def on_model_name_changed(self, sender, app_data):
        self.selected_model_name = app_data
    
    def on_batch_size_changed(self, sender, app_data):
        self.batch_size = app_data
    
    def on_hf_token_changed(self, sender, app_data):
        """Handle HF token input change"""
        self.hf_token = app_data
    
    def on_custom_model_toggle(self, sender, app_data):
        """Handle custom model checkbox toggle"""
        self.use_custom_model = app_data
        if dpg.does_item_exist("custom_model_group"):
            dpg.configure_item("custom_model_group", show=app_data)
    
    def on_custom_model_repo_changed(self, sender, app_data):
        """Handle custom model repo input change"""
        self.custom_model_repo = app_data
    
    def validate_source(self):
        """Validate data source"""
        if not self.source_path:
            if dpg.does_item_exist("source_status"):
                dpg.set_value("source_status", "Please enter a data source")
            return
        
        if dpg.does_item_exist("source_status"):
            dpg.set_value("source_status", "Validating...")
        
        def validate():
            try:
                is_valid, message = validate_source(self.source_type, self.source_path)
                if dpg.does_item_exist("source_status"):
                    dpg.set_value("source_status", message)
            except Exception as e:
                if dpg.does_item_exist("source_status"):
                    dpg.set_value("source_status", f"Error: {str(e)}")
        
        threading.Thread(target=validate, daemon=True).start()
    
    def hf_login(self):
        """Login to Hugging Face using token"""
        if not self.hf_token:
            if dpg.does_item_exist("hf_status"):
                dpg.set_value("hf_status", "Please enter HF token")
                dpg.configure_item("hf_status", color=[255, 100, 100])
            return
        
        if dpg.does_item_exist("hf_status"):
            dpg.set_value("hf_status", "Authenticating...")
            dpg.configure_item("hf_status", color=[255, 255, 100])
        
        def authenticate():
            try:
                # Try to import huggingface_hub
                try:
                    from huggingface_hub import login, whoami
                    
                    # Attempt login
                    login(token=self.hf_token)
                    
                    # Verify authentication
                    user_info = whoami(token=self.hf_token)
                    username = user_info.get('name', 'Unknown')
                    
                    self.hf_authenticated = True
                    if dpg.does_item_exist("hf_status"):
                        dpg.set_value("hf_status", f"✓ Authenticated as {username}")
                        dpg.configure_item("hf_status", color=[100, 255, 100])
                    
                    # Set environment variable for other libraries
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_token
                    
                except ImportError:
                    if dpg.does_item_exist("hf_status"):
                        dpg.set_value("hf_status", "huggingface_hub not installed")
                        dpg.configure_item("hf_status", color=[255, 100, 100])
                    
            except Exception as e:
                self.hf_authenticated = False
                if dpg.does_item_exist("hf_status"):
                    dpg.set_value("hf_status", f"Auth failed: {str(e)[:50]}")
                    dpg.configure_item("hf_status", color=[255, 100, 100])
        
        threading.Thread(target=authenticate, daemon=True).start()
    
    def check_hf_auth(self):
        """Check current HF authentication status"""
        def check_status():
            try:
                from huggingface_hub import whoami
                
                # Check if token is set in environment or provided
                token = self.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                
                if token:
                    user_info = whoami(token=token)
                    username = user_info.get('name', 'Unknown')
                    self.hf_authenticated = True
                    if dpg.does_item_exist("hf_status"):
                        dpg.set_value("hf_status", f"✓ Authenticated as {username}")
                        dpg.configure_item("hf_status", color=[100, 255, 100])
                else:
                    self.hf_authenticated = False
                    if dpg.does_item_exist("hf_status"):
                        dpg.set_value("hf_status", "Not authenticated")
                        dpg.configure_item("hf_status", color=[255, 100, 100])
                        
            except ImportError:
                if dpg.does_item_exist("hf_status"):
                    dpg.set_value("hf_status", "huggingface_hub not installed")
                    dpg.configure_item("hf_status", color=[255, 100, 100])
            except Exception as e:
                self.hf_authenticated = False
                if dpg.does_item_exist("hf_status"):
                    dpg.set_value("hf_status", f"Check failed: {str(e)[:50]}")
                    dpg.configure_item("hf_status", color=[255, 100, 100])
        
        threading.Thread(target=check_status, daemon=True).start()
    
    def validate_custom_model(self):
        """Validate custom model repository"""
        if not self.custom_model_repo:
            if dpg.does_item_exist("custom_model_status"):
                dpg.set_value("custom_model_status", "Please enter model repository")
            return
        
        if dpg.does_item_exist("custom_model_status"):
            dpg.set_value("custom_model_status", "Validating...")
        
        def validate():
            try:
                from huggingface_hub import model_info
                
                # Get token if available
                token = self.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                
                # Check if model exists and is accessible
                info = model_info(self.custom_model_repo, token=token)
                
                model_type = getattr(info, 'pipeline_tag', 'unknown')
                model_size = getattr(info, 'siblings', [])
                
                if dpg.does_item_exist("custom_model_status"):
                    dpg.set_value("custom_model_status", 
                                f"✓ Valid model ({model_type}, {len(model_size)} files)")
                    
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg:
                    status = "Model not found or private"
                elif "401" in error_msg or "403" in error_msg:
                    status = "Authentication required"
                else:
                    status = f"Error: {error_msg[:30]}..."
                
                if dpg.does_item_exist("custom_model_status"):
                    dpg.set_value("custom_model_status", status)
        
        threading.Thread(target=validate, daemon=True).start()
    
    def check_existing_hf_auth(self):
        """Check for existing HF authentication on startup"""
        # Check environment variable first
        env_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if env_token:
            self.hf_token = env_token
            # Trigger auth check in background
            threading.Thread(target=self._silent_auth_check, daemon=True).start()
    
    def _silent_auth_check(self):
        """Silent authentication check for startup"""
        try:
            from huggingface_hub import whoami
            user_info = whoami(token=self.hf_token)
            username = user_info.get('name', 'Unknown')
            self.hf_authenticated = True
            # Update UI if it exists
            if dpg.does_item_exist("hf_status"):
                dpg.set_value("hf_status", f"✓ Authenticated as {username}")
                dpg.configure_item("hf_status", color=[100, 255, 100])
            if dpg.does_item_exist("hf_token"):
                dpg.set_value("hf_token", self.hf_token)
        except:
            # Silent fail - user can manually authenticate if needed
            pass
    
    def preview_dataset(self):
        """Preview dataset"""
        if not self.source_path:
            if dpg.does_item_exist("processing_status"):
                dpg.set_value("processing_status", "Please enter a data source")
            return
        
        if dpg.does_item_exist("processing_status"):
            dpg.set_value("processing_status", "Loading preview...")
        
        def load_preview():
            try:
                df = load_dataset_smart(self.source_type, self.source_path)
                if df is not None:
                    self.dataset_preview = df
                    self.available_columns = list(df.columns)
                    self.selected_columns = list(df.columns)  # Default: select all columns
                    
                    if dpg.does_item_exist("processing_status"):
                        dpg.set_value("processing_status", f"Preview: {len(df)} rows, {len(df.columns)} columns loaded")
                    
                    # Update column selection UI
                    self.update_column_selection_ui()
                else:
                    if dpg.does_item_exist("processing_status"):
                        dpg.set_value("processing_status", "Failed to load dataset")
            except Exception as e:
                if dpg.does_item_exist("processing_status"):
                    dpg.set_value("processing_status", f"Error: {str(e)}")
        
        threading.Thread(target=load_preview, daemon=True).start()
    
    def update_column_selection_ui(self):
        """Update the column selection UI with available columns"""
        if not dpg.does_item_exist("column_selection_window"):
            return
        
        # Clear existing column selection
        if dpg.does_item_exist("column_selection_placeholder"):
            dpg.delete_item("column_selection_placeholder")
        
        # Remove existing column checkboxes
        for col in self.available_columns:
            if dpg.does_item_exist(f"col_checkbox_{col}"):
                dpg.delete_item(f"col_checkbox_{col}")
        
        # Add column checkboxes
        with dpg.group(parent="column_selection_window"):
            for col in self.available_columns:
                is_selected = col in self.selected_columns
                dpg.add_checkbox(
                    label=f"{col} ({self.get_column_info(col)})",
                    default_value=is_selected,
                    tag=f"col_checkbox_{col}",
                    callback=lambda s, a, c=col: self.on_column_selection_changed(c, a)
                )
    
    def get_column_info(self, column_name):
        """Get brief info about a column"""
        if self.dataset_preview is None:
            return "unknown"
        
        col_data = self.dataset_preview[column_name]
        dtype = str(col_data.dtype)
        
        if dtype.startswith('object') or dtype.startswith('string'):
            return "text"
        elif dtype.startswith('int') or dtype.startswith('float'):
            return "numeric"
        elif dtype.startswith('bool'):
            return "boolean"
        else:
            return dtype
    
    def on_column_selection_changed(self, column_name, is_selected):
        """Handle column selection change"""
        if is_selected and column_name not in self.selected_columns:
            self.selected_columns.append(column_name)
        elif not is_selected and column_name in self.selected_columns:
            self.selected_columns.remove(column_name)
    
    def select_all_columns(self):
        """Select all columns"""
        self.selected_columns = list(self.available_columns)
        for col in self.available_columns:
            if dpg.does_item_exist(f"col_checkbox_{col}"):
                dpg.set_value(f"col_checkbox_{col}", True)
    
    def clear_column_selection(self):
        """Clear all column selections"""
        self.selected_columns = []
        for col in self.available_columns:
            if dpg.does_item_exist(f"col_checkbox_{col}"):
                dpg.set_value(f"col_checkbox_{col}", False)
    
    def process_dataset(self):
        """Process dataset and generate embeddings with batch processing and column filtering"""
        text_column = dpg.get_value("text_column") if dpg.does_item_exist("text_column") else ""
        if not text_column:
            if dpg.does_item_exist("processing_status"):
                dpg.set_value("processing_status", "Please specify text column")
            return
        
        # Validate column selection
        if not self.selected_columns:
            if dpg.does_item_exist("processing_status"):
                dpg.set_value("processing_status", "Please select at least one metadata column")
            return
        
        if dpg.does_item_exist("processing_status"):
            dpg.set_value("processing_status", "Processing...")
        if dpg.does_item_exist("progress_bar"):
            dpg.configure_item("progress_bar", show=True)
            dpg.set_value("progress_bar", 0.0)
        
        def process():
            try:
                # Load dataset
                if dpg.does_item_exist("processing_status"):
                    dpg.set_value("processing_status", "Loading dataset...")
                if dpg.does_item_exist("progress_bar"):
                    dpg.set_value("progress_bar", 0.1)
                
                df = load_dataset_smart(self.source_type, self.source_path)
                
                # Sample data
                max_samples = dpg.get_value("max_samples") if dpg.does_item_exist("max_samples") else 1000
                if len(df) > max_samples:
                    df = df.sample(n=max_samples, random_state=42)
                
                # Filter columns early to save memory
                all_needed_columns = list(set(self.selected_columns + [text_column]))
                df = df[all_needed_columns]
                
                # Preprocess
                if dpg.does_item_exist("processing_status"):
                    dpg.set_value("processing_status", "Preprocessing text...")
                if dpg.does_item_exist("progress_bar"):
                    dpg.set_value("progress_bar", 0.2)
                
                steps = [{"name": "take_column", "params": {"column_name": text_column}}]
                
                clean_text_val = dpg.get_value("clean_text") if dpg.does_item_exist("clean_text") else True
                truncate_val = dpg.get_value("truncate_text") if dpg.does_item_exist("truncate_text") else True
                
                if clean_text_val:
                    steps.append({"name": "clean_text", "params": {}})
                
                if truncate_val:
                    steps.append({"name": "truncate_text", "params": {"max_length": 512}})
                
                processed_df, final_column = apply_preprocessing(df, steps)
                texts = processed_df[final_column].tolist()
                
                # Generate embeddings with batch processing
                batch_size = dpg.get_value("batch_size") if dpg.does_item_exist("batch_size") else self.batch_size
                total_texts = len(texts)
                all_embeddings = []
                
                if dpg.does_item_exist("processing_status"):
                    dpg.set_value("processing_status", f"Generating embeddings (batch size: {batch_size})...")
                
                for i in range(0, total_texts, batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_progress = 0.3 + (i / total_texts) * 0.4  # Progress from 30% to 70%
                    
                    if dpg.does_item_exist("progress_bar"):
                        dpg.set_value("progress_bar", batch_progress)
                    if dpg.does_item_exist("processing_status"):
                        dpg.set_value("processing_status", f"Processing batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}...")
                    
                    try:
                        # Determine model to use
                        model_type = self.selected_model_type
                        if self.use_custom_model and self.custom_model_repo:
                            model_name = self.custom_model_repo
                        else:
                            model_name = self.selected_model_name
                        
                        # Set HF token if available
                        if self.hf_token:
                            os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_token
                        
                        batch_embeddings = run_model(model_type, model_name, batch_texts)
                        if batch_embeddings is not None:
                            all_embeddings.append(batch_embeddings)
                        else:
                            raise Exception(f"Model failed on batch {i//batch_size + 1}")
                    except Exception as e:
                        if dpg.does_item_exist("processing_status"):
                            dpg.set_value("processing_status", f"Error in batch {i//batch_size + 1}: {str(e)}")
                        return
                
                # Combine all embeddings
                if dpg.does_item_exist("processing_status"):
                    dpg.set_value("processing_status", "Combining embeddings...")
                if dpg.does_item_exist("progress_bar"):
                    dpg.set_value("progress_bar", 0.8)
                
                embeddings = np.vstack(all_embeddings)
                
                # Save files with only selected columns
                if dpg.does_item_exist("processing_status"):
                    dpg.set_value("processing_status", "Saving files...")
                if dpg.does_item_exist("progress_bar"):
                    dpg.set_value("progress_bar", 0.9)
                
                np.save("generated_embeddings.npy", embeddings)
                
                # Create metadata with only selected columns plus processed text
                metadata_df = df[self.selected_columns].copy()
                metadata_df['processed_text'] = texts
                metadata_df.to_csv("generated_metadata.csv", index=False)
                
                if dpg.does_item_exist("progress_bar"):
                    dpg.set_value("progress_bar", 1.0)
                if dpg.does_item_exist("processing_status"):
                    dpg.set_value("processing_status", f"Complete! Generated {len(embeddings)} embeddings with {len(metadata_df.columns)} metadata columns.")
                
                # Continue to app
                if self.callback:
                    dpg.delete_item("enhanced_file_selector")
                    self.callback("generated_embeddings.npy", "generated_metadata.csv", "processed_text")
                
            except Exception as e:
                if dpg.does_item_exist("processing_status"):
                    dpg.set_value("processing_status", f"Error: {str(e)}")
                if dpg.does_item_exist("progress_bar"):
                    dpg.configure_item("progress_bar", show=False)
        
        threading.Thread(target=process, daemon=True).start()
    
    # File selection methods
    def select_embeddings_file(self):
        if dpg.does_item_exist("embeddings_dialog"):
            dpg.hide_item("embeddings_dialog")
        
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.embeddings_callback,
            tag="embeddings_dialog"
        ):
            dpg.add_file_extension(".npy")
        
        dpg.show_item("embeddings_dialog")
    
    def select_metadata_file(self):
        if dpg.does_item_exist("metadata_dialog"):
            dpg.hide_item("metadata_dialog")
        
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.metadata_callback,
            tag="metadata_dialog"
        ):
            dpg.add_file_extension(".csv")
            dpg.add_file_extension(".json")
            dpg.add_file_extension(".parquet")
            dpg.add_file_extension(".xlsx")
        
        dpg.show_item("metadata_dialog")
    
    def embeddings_callback(self, sender, app_data):
        file_path = list(app_data['selections'].values())[0]
        self.embeddings_file = file_path
        if dpg.does_item_exist("embeddings_path"):
            dpg.set_value("embeddings_path", file_path)
        if dpg.does_item_exist("embeddings_status"):
            dpg.set_value("embeddings_status", "✓ Valid file")
        dpg.hide_item("embeddings_dialog")
        self.update_continue_button()
    
    def metadata_callback(self, sender, app_data):
        file_path = list(app_data['selections'].values())[0]
        self.metadata_file = file_path
        if dpg.does_item_exist("metadata_path"):
            dpg.set_value("metadata_path", file_path)
        
        # Load columns
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=1)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, nrows=1)
            else:
                df = pd.read_excel(file_path, nrows=1)
            
            self.metadata_columns = list(df.columns)
            if dpg.does_item_exist("primary_column_combo"):
                dpg.configure_item("primary_column_combo", items=self.metadata_columns, enabled=True)
            
            # Smart default
            for col in ['title', 'text', 'content', 'processed_text']:
                if col in self.metadata_columns:
                    if dpg.does_item_exist("primary_column_combo"):
                        dpg.set_value("primary_column_combo", col)
                    self.primary_metadata_column = col
                    break
            
            if dpg.does_item_exist("metadata_status"):
                dpg.set_value("metadata_status", "✓ Valid file")
        except:
            if dpg.does_item_exist("metadata_status"):
                dpg.set_value("metadata_status", "✗ Invalid file")
        
        dpg.hide_item("metadata_dialog")
        self.update_continue_button()
    
    def on_primary_column_changed(self, sender, app_data):
        self.primary_metadata_column = app_data
    
    def update_continue_button(self):
        if self.embeddings_file and self.metadata_file:
            if dpg.does_item_exist("continue_file_btn"):
                dpg.configure_item("continue_file_btn", enabled=True)
    
    def load_example_data(self):
        """Load example data"""
        embeddings_files = [f for f in os.listdir('.') if f.endswith('.npy')]
        metadata_files = [f for f in os.listdir('.') if f.endswith(('.csv', '.json'))]
        
        if embeddings_files and metadata_files:
            self.embeddings_file = embeddings_files[0]
            self.metadata_file = metadata_files[0]
            
            if dpg.does_item_exist("embeddings_path"):
                dpg.set_value("embeddings_path", self.embeddings_file)
            if dpg.does_item_exist("metadata_path"):
                dpg.set_value("metadata_path", self.metadata_file)
            if dpg.does_item_exist("embeddings_status"):
                dpg.set_value("embeddings_status", "✓ Example loaded")
            if dpg.does_item_exist("metadata_status"):
                dpg.set_value("metadata_status", "✓ Example loaded")
            
            self.update_continue_button()
    
    def continue_file_mode(self):
        """Continue with file mode"""
        if self.callback and self.embeddings_file and self.metadata_file:
            dpg.delete_item("enhanced_file_selector")
            self.callback(self.embeddings_file, self.metadata_file, self.primary_metadata_column) 