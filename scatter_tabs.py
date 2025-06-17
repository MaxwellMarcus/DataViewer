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
import umap
import matplotlib.path as mpath



class ScatterTab:
    def __init__(self, name, data_manager, transformation=None):
        self.name = name

        self.transformation = transformation
        self.data_manager = data_manager  # Reference to shared data manager for color management

        # Mouse click state for selection
        self.mouse_down_pos = None
        self.mouse_up_pos = None
        self.mouse_down_time = None

        # Hover state tracking to prevent flickering
        self.current_hover_point = None
        self.last_hover_time = 0
        self.hover_debounce_time = 0.1  # 100ms debounce for smoother hover

        # Point selection state for axis creation
        self.selected_points = []  # List to store selected point indices
        self.selection_highlight_themes = []  # Themes for highlighting selected points
        
        # Theme tracking
        self.theme_ids = []
        # Create theme for hover area visualization
        with dpg.theme() as hover_area_theme:
            with dpg.theme_component(dpg.mvAreaSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, [255, 0, 255, 200], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, [255, 0, 255, 255], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, [255, 0, 255, 255], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
        self.hover_area_theme = hover_area_theme
        with dpg.theme() as hover_area_mean_theme:
            with dpg.theme_component(dpg.mvAreaSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, [0, 255, 150, 200], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, [0, 255, 255, 255], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
        self.hover_area_mean_theme = hover_area_mean_theme
        
        # Cluster visibility tracking
        self.cluster_visibility = {}  # {cluster_id: True/False}
        self.focus_mode = False
        self.focused_cluster = None

        self.double_click = False

        # Lasso selection state
        self.lasso_enabled = False
        self.lasso_selection = np.array([], dtype=np.int32)
        self.lasso_path = []  # For polygon lasso
        self.lasso_start_pos = None  # For rectangle lasso
        self.lasso_drawing = False
        self.shift_pressed = False
        self.control_pressed = False
        self.lasso_highlight_themes = []


        if self.transformation is not None:
            self.X = self.transformation(self.X)

        # Initialize visibility for all clusters
        unique_clusters = np.unique(self.data_manager.labels)
        for cluster_id in unique_clusters:
            self.cluster_visibility[cluster_id] = True

    def create_scatter_theme(self, color):
        """Create a theme for scatter plot points"""
        # Convert RGB to RGBA format for theme
        color_rgba = color + [255] if len(color) == 3 else color
        
        with dpg.theme() as theme_id:
            with dpg.theme_component(dpg.mvScatterSeries):
                # Set both line and fill colors for consistency
                dpg.add_theme_color(dpg.mvPlotCol_Line, color_rgba, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color_rgba, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color_rgba, category=dpg.mvThemeCat_Plots)
                # Also set the legend color to match
                dpg.add_theme_color(dpg.mvPlotCol_LegendText, color_rgba, category=dpg.mvThemeCat_Plots)
                # Set marker style
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 1, category=dpg.mvThemeCat_Plots)
        return theme_id

    def update_labels( self, labels ):
        self.clear_all_series()

        # Reset cluster visibility for new cluster set
        self.cluster_visibility = {}
        self.focus_mode = False
        self.focused_cluster = None
        
        unique_clusters = np.unique(self.data_manager.labels)
        for cluster_id in unique_clusters:
            self.cluster_visibility[cluster_id] = True
        
        # Refresh the legend to show new clusters
        self.refresh_custom_legend()

    def setup_ui(self, clusters):
        with dpg.tab(label=self.name, tag=f"{self.name}_tab"):
           self.setup_content(clusters)
    
    def setup_axis_controls(self):
        """Base axis controls - can be overridden by subclasses"""
        dpg.add_text("AXIS INFORMATION", color=[255, 255, 100])
        dpg.add_separator()
        dpg.add_text(f"Visualization Type: {self.name}")
        dpg.add_text("X-Axis: Component 1")
        dpg.add_text("Y-Axis: Component 2")
        dpg.add_separator()
        
        dpg.add_separator()
        dpg.add_text("No specific controls available for this visualization type.", wrap=280)

    def setup_content(self, clusters):
        """Setup UI content without creating a tab wrapper (for dynamic tabs)"""
         # Create horizontal layout with controls on left, plot in center, legend on right
        with dpg.group(horizontal=True, tag=f"{self.name}_content"):
            # Left side - axis controls (will be overridden by subclasses)
            with dpg.child_window(width=300, height=-1, border=True, tag=f"{self.name}_controls"):
    
                # Text box for displaying hovered point data
                dpg.add_input_text(
                    label="",
                    default_value="Hover over a point to see its information...",
                    multiline=True,
                    readonly=True,
                    height=150,
                    width=280,
                    tag=f"{self.name}_point_info_text"
                )
                dpg.add_separator()
                with dpg.plot(width=-20, no_title=True, no_menus=True, no_box_select=True, no_mouse_pos=True, crosshairs=False, equal_aspects=True, no_inputs=True, no_frame=True ):
                    dpg.add_plot_axis( dpg.mvXAxis, no_label=True, tag=f"{self.name}_hover_axis_x" )
                    with dpg.plot_axis( dpg.mvYAxis, no_label=True, tag=f"{self.name}_hover_axis" ):
                        # dpg.add_line_series( [ 0, 100, 100, 0 ], [ 0, 0, 100, 100], color=[255, 0, 255, 50], tag=f"{self.name}_hover_lines", loop=True )
                        dpg.add_area_series( [ 0, 100, 100, 0 ], [ 0, 0, 100, 100 ], tag=f"{self.name}_hover_area" )
                        # Apply theme to the initial hover area
                        dpg.bind_item_theme(f"{self.name}_hover_area", self.hover_area_theme)

                # Point selection for custom axis creation
                dpg.add_text("CUSTOM AXIS CREATION", color=[255, 255, 100])
                dpg.add_separator()
                
                dpg.add_text("Select two points to create a custom axis", wrap=280, color=[200, 200, 200])
                
                # Selected points display
                dpg.add_text("Selected Points:", color=[255, 255, 100])
                dpg.add_text("None", tag=f"{self.name}_selected_points_text", wrap=280)
                
                # Control buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Save Axis",
                        callback=self.save_custom_axis,
                        width=100,
                        enabled=False,
                        tag=f"{self.name}_save_axis_btn"
                    )

                dpg.add_separator()

                self.setup_axis_controls()

            # Center - the plot
            with dpg.child_window(width=-200, height=-1, border=False):
                with dpg.plot(label=self.name, width=-1, height=-1, tag=f"{self.name}_plot"):
                    # Configure plot (no legend)
                    dpg.add_plot_axis(dpg.mvXAxis, label=f"{self.name} Component 1", tag=f"{self.name}_x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, label=f"{self.name} Component 2", tag=f"{self.name}_y_axis")
                    
                    # Add scatter series for each cluster
                    self.add_scatter_data(clusters)
            
            # Right side - custom legend
            with dpg.child_window(width=180, height=-1, border=True, tag=f"{self.name}_legend"):
                self.setup_custom_legend()

        # Add item handler for the plot
        with dpg.item_handler_registry(tag=f"{self.name}_plot_handler"):
            dpg.add_item_hover_handler(callback=self.hover_callback)
            dpg.add_item_clicked_handler(callback=self.click_callback, button=dpg.mvMouseButton_Left)
            dpg.add_item_double_clicked_handler(callback=self.double_click_callback, button=dpg.mvMouseButton_Left)
        dpg.bind_item_handler_registry(f"{self.name}_plot", f"{self.name}_plot_handler")

        # Add global mouse handlers for robust click-vs-drag selection
        with dpg.handler_registry(tag=f"{self.name}_mouse_handler"):
            dpg.add_mouse_click_handler(callback=self.mouse_down_callback, button=dpg.mvMouseButton_Left)
            dpg.add_mouse_release_handler(callback=self.mouse_up_callback, button=dpg.mvMouseButton_Left)
            dpg.add_mouse_drag_handler(callback=self.mouse_drag_callback, button=dpg.mvMouseButton_Left)
            dpg.add_key_press_handler(callback=self.key_press_callback)
            dpg.add_key_release_handler(callback=self.key_release_callback)

    def set_lasso_mode(self, enabled):
        """Enable or disable lasso selection mode"""
        self.lasso_enabled = enabled
        print(f"[DEBUG] Lasso mode {'enabled' if enabled else 'disabled'} on tab {self.name}")
        print(f"[DEBUG] Tab {self.name} lasso_enabled state: {self.lasso_enabled}")
            
        # Configure plot pan/zoom behavior
        plot_tag = f"{self.name}_plot"
        if dpg.does_item_exist(plot_tag):
            # When lasso is enabled, disable pan and box select (zoom) to avoid conflicts
            if enabled:
                dpg.configure_item(plot_tag, 
                    pan_button=dpg.mvMouseButton_Middle,  # Pan only with middle mouse
                    box_select_button=-1  # Disable box select (zoom) completely
                )
                print(f"[DEBUG] Plot pan behavior updated for {self.name}: pan_button=Middle, box_select disabled")
            else:
                dpg.configure_item(plot_tag, 
                    pan_button=dpg.mvMouseButton_Left,  # Restore normal pan
                    box_select_button=dpg.mvMouseButton_Right  # Restore box select
                )
                dpg.delete_item(f"{self.name}_lasso_annotation")
                print(f"[DEBUG] Plot pan behavior updated for {self.name}: pan_button=Left, box_select=Right")

    def key_press_callback(self, sender, app_data):
        """Handle key press events"""

        if app_data == dpg.mvKey_LShift or app_data == dpg.mvKey_RShift:
            self.shift_pressed = True
            self.set_lasso_mode( True )
            print(f"[DEBUG] Shift pressed on {self.name} - shift_pressed: {self.shift_pressed}")
        if app_data == dpg.mvKey_LControl or app_data == dpg.mvKey_RControl:
            self.control_pressed = True
            self.set_lasso_mode( True )
            print(f"[DEBUG] Control pressed on {self.name} - control_pressed: {self.control_pressed}")

    def key_release_callback(self, sender, app_data):
        """Handle key release events"""
        if app_data == dpg.mvKey_LShift or app_data == dpg.mvKey_RShift:
            self.shift_pressed = False
            self.set_lasso_mode( False )
            print(f"[DEBUG] Shift released on {self.name} - shift_pressed: {self.shift_pressed}")
        if app_data == dpg.mvKey_LControl or app_data == dpg.mvKey_RControl:
            self.control_pressed = False
            self.set_lasso_mode( False )
            print(f"[DEBUG] Control released on {self.name} - control_pressed: {self.control_pressed}")

    def mouse_drag_callback(self, sender, app_data):
        """Handle mouse drag events for lasso selection"""
        if not self.lasso_enabled or ( not self.shift_pressed and not self.control_pressed ):
            return
            
        plot_tag = f"{self.name}_plot"
        plot_pos = None
        if dpg.does_item_exist(plot_tag) and dpg.is_item_hovered(plot_tag):
            plot_pos = dpg.get_plot_mouse_pos()
        if not plot_pos:
            return
            
        x_mouse, y_mouse = plot_pos
        print(f"Lasso drag at ({x_mouse:.2f}, {y_mouse:.2f}) on {self.name}")
        
        # Get selection mode from UI
        selection_mode = "Rectangle" if self.shift_pressed else ("Polygon" if self.control_pressed else "None")
        
        lasso_annotation_tag = f"{self.name}_lasso_annotation"
        if dpg.does_item_exist(lasso_annotation_tag):
            dpg.delete_item(lasso_annotation_tag)
                
        if selection_mode == "Rectangle":
            # For rectangle, we just track the current position
            # The actual selection happens on mouse up
            x_start, y_start = self.lasso_start_pos
            x_end, y_end = x_mouse, y_mouse
            with dpg.draw_layer(parent=f"{self.name}_plot", tag=lasso_annotation_tag): 
                dpg.draw_rectangle(
                    [x_start, y_start],
                    [x_end, y_end],
                    color=[255, 0, 255, 100],  # Magenta with transparency
                    thickness=0,
                    fill=[255, 0, 255, 50]  # Filled with more transparent magenta
                )
            print(f"Drew rectangle from ({x_start:.2f}, {y_start:.2f}) to ({x_end:.2f}, {y_end:.2f})")
        elif selection_mode == "Polygon":
            # For polygon, add points to the path during drag
            if self.lasso_drawing:
                self.lasso_path.append([x_mouse, y_mouse])
                print(f"Added point to lasso path: {len(self.lasso_path)} points")
                with dpg.draw_layer(parent=f"{self.name}_plot", tag=lasso_annotation_tag):
                    dpg.draw_polygon(
                        self.lasso_path + [self.lasso_path[0]],
                        color=[255, 0, 255, 100],  # Semi-transparent magenta
                        thickness=1,
                        fill=[255, 0, 255]  # More transparent magenta fill
                    )

    def hover_callback(self, sender, app_data):
        """Handle when mouse hovers over the projection plot area"""
        self.update_hover_annotation()
    
    def click_callback(self, sender, app_data):
        # No-op: selection is now handled in mouse_up_callback
        pass

    def double_click_callback(self, sender, app_data):
        # Handle lasso polygon completion for double-click
        if self.shift_pressed or self.control_pressed:
            # Clear lasso selection when double-clicking
            self.clear_lasso_selection()
        else:
            self.clear_point_selection()
            self.refresh_plot_visibility()
            self.double_click = True

    def mouse_down_callback(self, sender, app_data):
        plot_tag = f"{self.name}_plot"
        if dpg.does_item_exist(plot_tag) and dpg.is_item_hovered(plot_tag):
            abs_pos = dpg.get_mouse_pos()
            plot_pos = dpg.get_plot_mouse_pos()
            
            print(f"[DEBUG] Mouse down on {self.name}: abs_pos={abs_pos}, plot_pos={plot_pos}")
            print(f"[DEBUG] Lasso enabled: {self.lasso_enabled}, Shift pressed: {self.shift_pressed}")
            
            if abs_pos:
                self.mouse_down_pos = abs_pos
            else:
                self.mouse_down_pos = None
                
            # Handle lasso selection start
            if self.lasso_enabled and ( self.shift_pressed or self.control_pressed ) and plot_pos:
                x_mouse, y_mouse = plot_pos
                print(f"[DEBUG] Lasso selection started at ({x_mouse:.2f}, {y_mouse:.2f}) on {self.name}")
                
                # Get selection mode
                selection_mode = "Rectangle" if self.shift_pressed else ("Polygon" if self.control_pressed else "None")
                    
                if selection_mode == "Rectangle":
                    self.lasso_start_pos = [x_mouse, y_mouse]
                    self.lasso_drawing = True
                    print(f"[DEBUG] Started rectangle lasso at {self.lasso_start_pos}")
                elif selection_mode == "Polygon":
                    if not self.lasso_drawing:
                        # Start new polygon
                        self.lasso_path = [[x_mouse, y_mouse]]
                        self.lasso_drawing = True
                        print(f"[DEBUG] Started polygon lasso on {self.name}")
                    else:
                        # Add point to existing polygon
                        self.lasso_path.append([x_mouse, y_mouse])
                        print(f"[DEBUG] Added point to polygon: {len(self.lasso_path)} points")
            else:
                if not self.lasso_enabled:
                    print(f"[DEBUG] Lasso not enabled on {self.name}")
                if not self.shift_pressed:
                    print(f"[DEBUG] Shift not pressed on {self.name}")
                if not self.control_pressed:
                    print(f"[DEBUG] Control not pressed on {self.name}")
                if not plot_pos:
                    print(f"[DEBUG] No plot position available on {self.name}")
        else:
            self.mouse_down_pos = None
            if not dpg.does_item_exist(plot_tag):
                print(f"[DEBUG] Plot {plot_tag} does not exist")
            if not dpg.is_item_hovered(plot_tag):
                print(f"[DEBUG] Plot {plot_tag} not hovered")

    def mouse_up_callback(self, sender, app_data):
        if self.double_click:
            self.double_click = False
            return
        
        plot_tag = f"{self.name}_plot"
        if dpg.does_item_exist(plot_tag) and dpg.is_item_hovered(plot_tag):
            abs_pos = dpg.get_mouse_pos()
            plot_pos = dpg.get_plot_mouse_pos()
            
            print(f"[DEBUG] Mouse up on {self.name}: abs_pos={abs_pos}, plot_pos={plot_pos}")
            print(f"[DEBUG] Lasso enabled: {self.lasso_enabled}, Shift pressed: {self.shift_pressed}, Drawing: {self.lasso_drawing}")
            
            # Handle lasso selection completion
            if self.lasso_enabled and ( self.shift_pressed or self.control_pressed ) and self.lasso_drawing and plot_pos:
                x_mouse, y_mouse = plot_pos
                print(f"[DEBUG] Lasso selection ended at ({x_mouse:.2f}, {y_mouse:.2f}) on {self.name}")
                
                # Get selection mode
                selection_mode = "Rectangle" if self.shift_pressed else ("Polygon" if self.control_pressed else "None")
                    
                if selection_mode == "Rectangle" and self.lasso_start_pos:
                    # Complete rectangle selection
                    end_pos = [x_mouse, y_mouse]
                    print(f"[DEBUG] Completing rectangle selection from {self.lasso_start_pos} to {end_pos}")
                    selected_indices = self.select_points_in_rectangle(self.lasso_start_pos, end_pos)
                    print(f"[DEBUG] Rectangle selection found {len(selected_indices)} points: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
                    self.handle_lasso_selection(selected_indices)
                    self.lasso_drawing = False
                    self.lasso_start_pos = None
                # Polygon selection is completed on double-click
                elif selection_mode == "Polygon":
                    self.complete_lasso_selection()
                
                return
            
            # Handle regular point selection (non-lasso)
            if abs_pos and self.mouse_down_pos and not (self.lasso_enabled and ( self.shift_pressed or self.control_pressed ) ):
                dx = self.mouse_down_pos[0] - abs_pos[0]
                dy = self.mouse_down_pos[1] - abs_pos[1]
                dist = (dx**2 + dy**2)**0.5
                print(f"[DEBUG] Regular click distance: {dist:.2f}")
                if dist < 3.0:
                    # Treat as click
                    if plot_pos:
                        x_mouse, y_mouse = plot_pos
                        closest_point_info = self.find_closest_point(x_mouse, y_mouse, self.X, self.data_manager.labels)
                        if closest_point_info:
                            point_idx, cluster, distance, x_coord, y_coord = closest_point_info
                            print(f"[DEBUG] Found closest point {point_idx} at distance {distance:.2f}")
                            if distance < 3.0:
                                self.select_point(point_idx)
                                self.lasso_selection = np.array([], dtype=np.int32)
                            else:
                                self.clear_point_selection()
                else:
                    self.clear_point_selection()
            self.mouse_down_pos = None
        else:
            self.mouse_down_pos = None

    def complete_lasso_selection(self):
        """Complete polygon lasso selection"""
        if len(self.lasso_path) >= 3:
            selected_indices = self.select_points_in_polygon(self.lasso_path)
            self.handle_lasso_selection(selected_indices)
            print(f"Completed polygon lasso selection: {len(selected_indices)} points selected")
        
        # Reset lasso state
        self.lasso_drawing = False
        self.lasso_path = []

    def select_points_in_rectangle(self, start_pos, end_pos):
        """Select points within a rectangular area"""
        try:
            x1, y1 = start_pos
            x2, y2 = end_pos
            
            # Ensure proper ordering
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            print(f"[DEBUG] Rectangle selection bounds: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
            print(f"[DEBUG] Data shape: {self.X.shape}, checking {len(self.X)} points")
            
            # Find points within the rectangle
            selected_indices = []
            for i, (x, y) in enumerate(self.X):
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    selected_indices.append(i)
            
            print(f"[DEBUG] Rectangle selection found {len(selected_indices)} points")
            if len(selected_indices) > 0:
                print(f"[DEBUG] First 10 selected indices: {selected_indices[:10]}")
                print(f"[DEBUG] Sample coordinates: {[(self.X[i, 0], self.X[i, 1]) for i in selected_indices[:5]]}")
            
            return selected_indices
            
        except Exception as e:
            print(f"[ERROR] Error in rectangle selection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def select_points_in_polygon(self, polygon_points):
        """Select points within a polygon area"""
        try:
            if len(polygon_points) < 3:
                print("Polygon needs at least 3 points")
                return []
            
            # Convert to numpy arrays
            polygon_points = np.array(polygon_points)
            data_points = np.array(self.X)
            
            print(f"Polygon selection with {len(polygon_points)} points, checking {len(data_points)} data points")
            
            # Create matplotlib path
            path = mpath.Path(polygon_points)
            
            # Check which points are inside
            inside = path.contains_points(data_points)
            selected_indices = np.where(inside)[0].tolist()
            
            print(f"Found {len(selected_indices)} points in polygon")
            return selected_indices
            
        except Exception as e:
            print(f"Error in polygon selection: {e}")
            return []

    def handle_lasso_selection(self, selected_indices):
        """Handle the result of lasso selection"""
        print(f"[DEBUG] handle_lasso_selection called with {len(selected_indices)} indices")
        
        if not selected_indices:
            print("[DEBUG] No points selected")
            return

            
        # Add new selection
        old_total = len(self.lasso_selection)
        self.lasso_selection = np.concatenate([self.lasso_selection, np.array(selected_indices, dtype=np.int32)])
        new_total = len(self.lasso_selection)
        
        print(f"[DEBUG] Added {len(selected_indices)} new points, total selection: {old_total} -> {new_total}")
        print(f"[DEBUG] Current lasso selection: {list(self.lasso_selection)[:10]}{'...' if len(self.lasso_selection) > 10 else ''}")
        
        # Notify tab manager if available
        self.data_manager.handle_lasso_selection( selected_indices )

        
        # Update visual highlights
        print(f"[DEBUG] Updating visual highlights for {len(self.lasso_selection)} points")
        self.refresh_plot_visibility()

    def clear_lasso_selection(self):
        """Clear lasso selection"""
        self.lasso_selection = np.array([], dtype=np.int32)
        self.refresh_plot_visibility()
        self.lasso_drawing = False
        self.lasso_path = []
        self.lasso_start_pos = None
        print(f"Cleared lasso selection on {self.name}")

    def highlight_lasso_selection(self, selected_indices):
        """Highlight lasso selection from external source"""
        self.lasso_selection = np.array(selected_indices, dtype=np.int32)
        self.refresh_plot_visibility()

    def select_point(self, point_idx):
        """Select a point for axis creation"""
        if point_idx in self.selected_points:
            # Deselect if already selected
            self.selected_points.remove(point_idx)
            # Clear selection in data manager
            self.data_manager.selected_point_index = None
        else:
            # Add to selection (max 2 points)
            if len(self.selected_points) < 2:
                self.selected_points.append(point_idx)
            else:
                # Replace the first selected point with the new one
                old_point = self.selected_points.pop(0)
                self.selected_points.append(point_idx)
            
            # Notify data manager of single point selection
            self.data_manager.handle_single_point_selection(point_idx)
        
        self.update_selection_display()
        self.refresh_plot_visibility()
    
    def add_point_highlight(self, point_idx):
        """Add a highlight to a selected point"""
        try:
            # Create a highlight theme for the selected point
            highlight_color = [255, 255, 0, 255]  # Yellow highlight
            
            with dpg.theme() as highlight_theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, highlight_color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, highlight_color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [0, 0, 0, 255], category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 8, category=dpg.mvThemeCat_Plots)
            
            # Add the highlighted point as a separate series
            highlight_tag = f"{self.name}_highlight_{point_idx}"
            x_coord = self.X[point_idx, 0]
            y_coord = self.X[point_idx, 1]
            
            if dpg.does_item_exist(highlight_tag):
                dpg.delete_item(highlight_tag)
            
            dpg.add_scatter_series(
                [x_coord], [y_coord],
                label=f"Selected Point {point_idx}",
                parent=f"{self.name}_y_axis",
                tag=highlight_tag
            )
            
            dpg.bind_item_theme(highlight_tag, highlight_theme)
            self.selection_highlight_themes.append((highlight_theme, highlight_tag))
            
        except Exception as e:
            print(f"Error highlighting point {point_idx}: {e}")
    
    def remove_point_highlight(self, point_idx):
        """Remove highlight from a deselected point"""
        try:
            highlight_tag = f"{self.name}_highlight_{point_idx}"
            if dpg.does_item_exist(highlight_tag):
                dpg.delete_item(highlight_tag)
            
            # Remove associated theme
            self.selection_highlight_themes = [
                (theme, tag) for theme, tag in self.selection_highlight_themes 
                if tag != highlight_tag
            ]
        except Exception as e:
            print(f"Error removing highlight for point {point_idx}: {e}")
    
    def clear_point_selection(self, sender=None, app_data=None):
        """Clear all selected points"""
        self.selected_points.clear()
        self.update_selection_display()
        self.refresh_plot_visibility()
    
    def update_selection_display(self):
        """Update the UI to show current selection status"""
        text_tag = f"{self.name}_selected_points_text"
        save_btn_tag = f"{self.name}_save_axis_btn"
        
        if dpg.does_item_exist(text_tag):
            if len(self.selected_points) == 0:
                dpg.set_value(text_tag, "None")
            elif len(self.selected_points) == 1:
                point_idx = self.selected_points[0]
                caption = self.data_manager.metadata.iloc[point_idx][ self.data_manager.primary_metadata_column ]
                dpg.set_value(text_tag, f"Point 1: {point_idx}\n{caption}")
            else:
                point1_idx, point2_idx = self.selected_points
                caption1 = self.data_manager.metadata.iloc[point1_idx][ self.data_manager.primary_metadata_column ]
                caption2 = self.data_manager.metadata.iloc[point2_idx][ self.data_manager.primary_metadata_column ]
                dpg.set_value(text_tag, f"Point 1: {point1_idx}\n{caption1}\n\nPoint 2: {point2_idx}\n{caption2}")
        
        # Enable/disable save button
        if dpg.does_item_exist(save_btn_tag):
            dpg.configure_item(save_btn_tag, enabled=len(self.selected_points) > 0)
    
    def save_custom_axis(self, sender=None, app_data=None):
        """Save the difference between two selected points as a custom axis"""
        if len(self.selected_points) == 0:
            print("Must select at least one point to create an axis")
            return
        
        try:
            if len(self.selected_points) == 1:
                axis_vector = self.data_manager.X[self.selected_points[0]]
            else:
                point1_idx, point2_idx = self.selected_points
                
                # Calculate axis vector from original high-dimensional data
                axis_vector = self.data_manager.X[point2_idx] - self.data_manager.X[point1_idx]
            
            # Normalize the axis vector
            axis_vector = axis_vector / np.linalg.norm(axis_vector)
            
            # Create axis metadata
            axis_info = {
                'vector': axis_vector,
                'name': f"From Point {point1_idx} to Point {point2_idx}" if len(self.selected_points) == 2 else f"Point {point1_idx}" ,
                'point1_idx': point1_idx,
                'point2_idx': point2_idx if len(self.selected_points) == 2 else None,
                'point1_caption': self.data_manager.metadata.iloc[point1_idx][ self.data_manager.primary_metadata_column ],
                'point2_caption': self.data_manager.metadata.iloc[point2_idx][ self.data_manager.primary_metadata_column ] if len(self.selected_points) == 2 else None,
                'type': 'difference' if len(self.selected_points) == 2 else 'single'
            }
            
            # Add to global saved axes through the data manager
            if hasattr(self.data_manager, 'add_saved_axis'):
                self.data_manager.add_saved_axis(axis_info)
                print(f"Saved custom axis: {axis_info['name']}")
            else:
                print("Could not save axis - no axis manager available")
            
        except Exception as e:
            print(f"Error saving custom axis: {e}")

    def update_hover_annotation(self):
        """Update hover annotation that appears next to datapoints"""
        try:
            current_time = time.time()
            
            # Debounce rapid calls
            if current_time - self.last_hover_time < self.hover_debounce_time:
                return
            
            # Select plot and data based on type
            plot_tag = f"{self.name}_plot"
            
            # Check if plot exists and try to get plot mouse position
            if dpg.does_item_exist(plot_tag) and dpg.is_item_hovered(plot_tag):
                plot_mouse_pos = dpg.get_plot_mouse_pos()
                if plot_mouse_pos:
                    x_mouse, y_mouse = plot_mouse_pos
                    
                    # Find closest point
                    closest_point_info = self.find_closest_point(x_mouse, y_mouse, self.X, self.data_manager.labels)
                    if closest_point_info:
                        point_idx, cluster, distance, x_coord, y_coord = closest_point_info
                        
                        # Only show annotation if close enough to a point
                        if distance < 3.0:  # Reduced distance for more precise hovering
                            # Check if this is the same point we're already hovering over
                            current_hover_key = (self.name, point_idx)
                            if self.current_hover_point != current_hover_key:
                                self.current_hover_point = current_hover_key
                                self.show_hover_annotation(point_idx, cluster, x_coord, y_coord, plot_tag)
                                self.last_hover_time = current_time
                        else:
                            if self.current_hover_point is not None:
                                self.current_hover_point = None
                                self.hide_hover_annotation()
                                self.last_hover_time = current_time
                    else:
                        if self.current_hover_point is not None:
                            self.current_hover_point = None
                            self.hide_hover_annotation()
                            self.last_hover_time = current_time
                else:
                    if self.current_hover_point is not None:
                        self.current_hover_point = None
                        self.hide_hover_annotation()
                        self.last_hover_time = current_time
            else:
                if self.current_hover_point is not None:
                    self.current_hover_point = None
                    self.hide_hover_annotation()
                    self.last_hover_time = current_time
        except Exception as e:
            print(f"Hover error: {e}")
            if self.current_hover_point is not None:
                self.current_hover_point = None
                self.hide_hover_annotation()
    
    def show_hover_annotation(self, point_idx, cluster, x_coord, y_coord, plot_tag):
        """Display point information in the left panel text box"""
        # Create detailed information text
        cluster_colors = ["Red", "Blue", "Green", "Other"]
        cluster_name = cluster_colors[cluster] if cluster < len(cluster_colors) else f"Cluster {cluster}"
        
        # Format the information display
        info_text = f"Point Index: {point_idx}\n"
        info_text += f"Cluster: {cluster_name}\n"
        info_text += f"Coordinates: ({x_coord:.3f}, {y_coord:.3f})\n\n"
        
        # Add all metadata including caption
        if point_idx < len(self.data_manager.metadata):
            info_text += "Metadata:\n"
            info_text += f"{'-' * 15}\n"
            for col in self.data_manager.metadata.columns:
                value = self.data_manager.metadata.iloc[point_idx][col]
                # Wrap long text values
                if isinstance(value, str) and len(value) > 40:
                    words = value.split()
                    lines = []
                    current_line = []
                    line_length = 0
                    
                    for word in words:
                        if line_length + len(word) + 1 <= 40:  # +1 for space
                            current_line.append(word)
                            line_length += len(word) + 1
                        else:
                            lines.append(" ".join(current_line))
                            current_line = [word]
                            line_length = len(word)
                            
                    if current_line:
                        lines.append(" ".join(current_line))
                    value = "\n    " + "\n    ".join(lines)
                info_text += f"{col}: {value}\n"

        offset = 1
        points = self.data_manager.pca_variance_75
        datapoint = self.data_manager.pca_data[point_idx][ :points ]
        print( self.data_manager.pca.explained_variance_ratio_[ :points ].sum(), self.data_manager.pca_variance_75 )
        xpoints = []
        ypoints = []
        for i in range(len(datapoint)):
            theta = i * 2 * np.pi / len( datapoint )
            length = datapoint[i] + offset
            xpoints.append( length * np.cos(theta) )
            ypoints.append( length * np.sin(theta) )

        if dpg.does_item_exist(f"{self.name}_hover_area"):
            dpg.delete_item(f"{self.name}_hover_area")

        dpg.add_area_series( xpoints, ypoints, tag=f"{self.name}_hover_area", parent=f"{self.name}_hover_axis", fill=[255, 0, 255, 200] )
        dpg.bind_item_theme(f"{self.name}_hover_area", self.hover_area_theme)

        cluster_mean = self.data_manager.pca_data[self.data_manager.labels == cluster].mean(axis=0)[ :points ]
        for i in range(len(datapoint)):
            theta = i * 2 * np.pi / len( datapoint )
            length = cluster_mean[i] + offset
            xpoints.append( length * np.cos(theta) )
            ypoints.append( length * np.sin(theta) )

        if dpg.does_item_exist(f"{self.name}_hover_area_mean"):
            dpg.delete_item(f"{self.name}_hover_area_mean")
        dpg.add_area_series( xpoints, ypoints, tag=f"{self.name}_hover_area_mean", parent=f"{self.name}_hover_axis", fill=[0, 255, 150, 200] )
        # dpg.bind_item_theme(f"{self.name}_hover_area_mean", self.hover_area_mean_theme)

        dpg.fit_axis_data(f"{self.name}_hover_axis")
        dpg.fit_axis_data(f"{self.name}_hover_axis_x")
                
        # Update the text box in the left panel
        text_tag = f"{self.name}_point_info_text"
        if dpg.does_item_exist(text_tag):
            dpg.set_value(text_tag, info_text)

    def hide_hover_annotation(self):
        """Clear the point information text box"""
        text_tag = f"{self.name}_point_info_text"
        if dpg.does_item_exist(text_tag):
            dpg.set_value(text_tag, "Hover over a point to see its information...")
    
    def setup_custom_legend(self):
        """Setup custom legend with visibility controls"""
        dpg.add_text("CLUSTER LEGEND", color=[255, 255, 100])
        dpg.add_separator()
        
        # Get unique clusters
        unique_clusters = np.unique(self.data_manager.labels)
        
        # Add controls for each cluster
        for cluster_id in sorted(unique_clusters):
            # Get consistent color from data manager
            if self.data_manager:
                color = self.data_manager.get_cluster_color(cluster_id)
            else:
                # Fallback to local colors if no data manager
                fallback_colors = [
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
                color = fallback_colors[cluster_id % len(fallback_colors)]
            
            cluster_name = f"Cluster {cluster_id}"
            
            # Create clickable legend entry for this cluster
            with dpg.group(horizontal=True, tag=f"{self.name}_cluster_{cluster_id}_group"):
                # Clickable color indicator
                with dpg.drawlist(width=15, height=15, tag=f"{self.name}_cluster_{cluster_id}_color"):
                    dpg.draw_rectangle((0, 0), (15, 15), color=color, fill=color, tag=f"{self.name}_cluster_{cluster_id}_rect")
                
                # Clickable cluster name
                is_visible = self.is_cluster_visible(cluster_id)
                text_color = [255, 255, 255] if is_visible else [128, 128, 128]
                dpg.add_text(cluster_name, color=text_color, tag=f"{self.name}_cluster_{cluster_id}_text")
            
            # Add click handlers for both the color box and text (with error handling)
            try:
                with dpg.item_handler_registry(tag=f"{self.name}_cluster_{cluster_id}_color_handler"):
                    dpg.add_item_clicked_handler(callback=self.toggle_cluster_visibility, user_data=cluster_id, button=dpg.mvMouseButton_Right)
                    dpg.add_item_clicked_handler(callback=self.focus_cluster, user_data=cluster_id, button=dpg.mvMouseButton_Left)
                    dpg.add_item_hover_handler(callback=self.focus_cluster_hover, user_data=cluster_id)
                    
                
                with dpg.item_handler_registry(tag=f"{self.name}_cluster_{cluster_id}_text_handler"):
                    dpg.add_item_clicked_handler(callback=self.toggle_cluster_visibility, user_data=cluster_id, button=dpg.mvMouseButton_Right)
                    dpg.add_item_clicked_handler(callback=self.focus_cluster, user_data=cluster_id, button=dpg.mvMouseButton_Left)
                    dpg.add_item_hover_handler(callback=self.focus_cluster_hover, user_data=cluster_id)

                # Bind handlers
                dpg.bind_item_handler_registry(f"{self.name}_cluster_{cluster_id}_color", f"{self.name}_cluster_{cluster_id}_color_handler")
                dpg.bind_item_handler_registry(f"{self.name}_cluster_{cluster_id}_text", f"{self.name}_cluster_{cluster_id}_text_handler")
            except Exception as e:
                # Skip handler creation if it fails, legend will still show but without interactivity
                print(f"Warning: Could not create handlers for cluster {cluster_id} in {self.name}: {e}")
            
            dpg.add_separator()

        # Add legend hover handler with error handling
        try:
            with dpg.item_handler_registry(tag=f"{self.name}_legend_handler"):
                dpg.add_item_hover_handler( callback=self.check_legend_hover )

            dpg.bind_item_handler_registry(f"{self.name}_legend", f"{self.name}_legend_handler")
        except Exception as e:
            # Skip legend hover handler if it fails
            print(f"Warning: Could not create legend hover handler for {self.name}: {e}")
        
        # Add instructions and controls at the bottom
        dpg.add_text("CONTROLS", color=[255, 255, 100])
        dpg.add_separator()
        dpg.add_text("• Left click: Hide/Show", color=[200, 200, 200], wrap=160)
        dpg.add_text("• Right click: Focus", color=[200, 200, 200], wrap=160)
        dpg.add_separator()
        dpg.add_button(
            label="Show All",
            width=-1,
            callback=self.show_all_clusters
        )
    
    def check_legend_hover(self, sender, app_data):
        """Check if the legend is being hovered over"""
        try:
            unique_clusters = np.unique(self.data_manager.labels)

            for cluster_id in unique_clusters:
                text_tag = f"{self.name}_cluster_{cluster_id}_text"
                color_tag = f"{self.name}_cluster_{cluster_id}_color"
                
                # Check if items exist before checking hover state
                text_hovered = dpg.does_item_exist(text_tag) and dpg.is_item_hovered(text_tag)
                color_hovered = dpg.does_item_exist(color_tag) and dpg.is_item_hovered(color_tag)
                
                if text_hovered or color_hovered:
                    if self.cluster_visibility.get(cluster_id, True):
                        return
                    else:
                        break

            if self.focus_mode == "hovered":
                self.focus_mode = False
                self.focused_cluster = None
                for cid in self.cluster_visibility:
                    if self.cluster_visibility[cid]:
                        self.cluster_visibility[cid] = True

                self.refresh_plot_visibility()
        except Exception as e:
            # Silently handle errors to prevent spam in console
            # This can happen during legend refresh when items are being recreated
            pass
    
    def toggle_cluster_visibility(self, sender, app_data, user_data):
        """Toggle visibility of a specific cluster"""
        cluster_id = user_data
        # Toggle the current visibility state
        current_visibility = self.cluster_visibility.get(cluster_id, True)
        new_visibility = not current_visibility
        self.cluster_visibility[cluster_id] = new_visibility

        self.focus_mode = False
        self.focused_cluster = None

        for cid in self.cluster_visibility:
            if self.cluster_visibility[cid] == "hovered" or self.cluster_visibility[cid] == "unhovered":
                self.cluster_visibility[cid] = True
        
        # Update visual feedback in the legend
        self.update_legend_item_appearance(cluster_id, new_visibility)
        
        # If we're in focus mode and this cluster is being hidden, exit focus mode
        if self.focus_mode and self.focused_cluster == cluster_id:
            self.show_all_clusters()
        else:
            self.refresh_plot_visibility()

    def update_legend_item_appearance(self, cluster_id, is_visible):
        """Update the visual appearance of a legend item based on visibility"""
        try:
            text_tag = f"{self.name}_cluster_{cluster_id}_text"
            color_tag = f"{self.name}_cluster_{cluster_id}_color"
            rect_tag = f"{self.name}_cluster_{cluster_id}_rect"
            
            if dpg.does_item_exist(text_tag):
                # Update text color - full color if visible, grayed out if hidden
                text_color = ([200, 200, 200] if is_visible == "unhovered" else [255, 255, 255]) if is_visible else [128, 128, 128]
                dpg.configure_item(text_tag, color=text_color)
            
            if dpg.does_item_exist(rect_tag):
                # Update color box - full color if visible, grayed out if hidden
                if self.data_manager:
                    base_color = self.data_manager.get_cluster_color(cluster_id)
                else:
                    # Fallback color
                    base_color = [255, 255, 255]
                
                if is_visible == "unhovered":
                    display_color = [int(c * 0.8) for c in base_color]
                elif is_visible:
                    display_color = base_color
                else:
                    # Make color grayed out (reduce brightness)
                    display_color = [int(c * 0.5) for c in base_color]
                
                # Update the rectangle color directly
                dpg.configure_item(rect_tag, color=display_color, fill=display_color)
        except Exception as e:
            # Silently handle errors during legend updates
            # This can happen when items are being recreated during refresh
            pass
    
    def focus_cluster(self, sender, app_data, user_data):
        """Focus on a specific cluster (hide all others)"""
        cluster_id = user_data

        if self.focus_mode and not self.focus_mode == "hovered":
            self.show_all_clusters()
            return
        
        # Set focus mode
        self.focus_mode = True
        self.focused_cluster = cluster_id
        
        # Hide all clusters except the focused one
        for cid in self.cluster_visibility:
            if cid == cluster_id:
                self.cluster_visibility[cid] = True
                self.update_legend_item_appearance(cid, True)
            else:
                self.cluster_visibility[cid] = False
                self.update_legend_item_appearance(cid, False)
        
        self.refresh_plot_visibility()

    def focus_cluster_hover(self, sender, app_data, user_data):
        """Focus on a specific cluster (hide all others)"""
        cluster_id = user_data
        if not self.is_cluster_visible(cluster_id) or self.focus_mode or self.focused_cluster == cluster_id:
            return

        # Set focus mode
        self.focus_mode = "hovered"
        self.focused_cluster = cluster_id
        
        # Hide all clusters except the focused one
        for cid in self.cluster_visibility:
            if cid == cluster_id:
                self.cluster_visibility[cid] = "hovered"
                self.update_legend_item_appearance(cid, True)
            elif self.cluster_visibility[cid]:
                self.cluster_visibility[cid] = "unhovered"
                self.update_legend_item_appearance(cid, "unhovered")
        
        self.refresh_plot_visibility()
    
    def show_all_clusters(self, sender=None, app_data=None):
        """Show all clusters and exit focus mode"""
        self.focus_mode = False
        self.focused_cluster = None
        
        # Show all clusters
        for cluster_id in self.cluster_visibility:
            self.cluster_visibility[cluster_id] = True
            self.update_legend_item_appearance(cluster_id, True)
        
        self.refresh_plot_visibility()
    
    def refresh_plot_visibility(self):
        """Refresh the plot to show/hide clusters based on visibility settings"""
        try:
            # Clear existing series and themes
            self.clear_all_series()
            
            # Re-add only visible clusters
            self.add_scatter_data([])
        except Exception as e:
            print(f"Error refreshing plot visibility: {e}")
            # Try to clear everything and re-add
            try:
                # Force clear everything
                unique_clusters = np.unique(self.data_manager.labels)
                for cluster_id in unique_clusters:
                    cluster_tag = f"{self.name}_cluster_{cluster_id}"
                    if dpg.does_item_exist(cluster_tag):
                        dpg.delete_item(cluster_tag)
                
                # Clear and recreate themes
                for theme_id in self.theme_ids:
                    if dpg.does_item_exist(theme_id):
                        dpg.delete_item(theme_id)
                self.theme_ids.clear()
                
                # Try again
                self.add_scatter_data([])
            except Exception as e2:
                print(f"Failed to recover from plot refresh error: {e2}")
    
    def is_cluster_visible(self, cluster_id):
        """Check if a cluster should be visible"""
        return self.cluster_visibility.get(cluster_id, True)
    
    def refresh_custom_legend(self):
        """Refresh the custom legend panel when clusters change"""
        # Skip if legend has been disabled due to persistent errors
        if hasattr(self, '_legend_disabled') and self._legend_disabled:
            return
            
        legend_tag = f"{self.name}_legend"
        if not dpg.does_item_exist(legend_tag):
            return
            
        # try:
        # More careful cleanup approach
        self._cleanup_legend_handlers()
        
        # Clear the legend panel completely
        dpg.delete_item(legend_tag)
        print( "Checking item existance: ", dpg.does_item_exist( f"{self.name}_cluster_0_group" ))
        
        # Re-setup the legend with current clusters
        with dpg.child_window(width=180, height=650, border=True, tag=f"{self.name}_legend", parent=f"{self.name}_content"):
            self.setup_custom_legend()
                    
        # except Exception as e:
        #     print(f"Error refreshing legend for {self.name}: {e}")
        #     # Try simplified recreation without complex cleanup
        #     try:
        #         # Just clear content and recreate
        #         if dpg.does_item_exist(legend_tag):
        #             dpg.delete_item(legend_tag, children_only=True)
        #         self._simple_legend_setup()
                        
        #     except Exception as e2:
        #         print(f"Failed to recover legend for {self.name}: {e2}")
        #         # Mark this tab as having legend issues
        #         self._legend_disabled = True
    
    def _cleanup_legend_handlers(self):
        """Clean up legend handlers before refresh"""
        try:
            unique_clusters = np.unique(self.data_manager.labels)
            for cluster_id in unique_clusters:
                handler_tags = [
                    f"{self.name}_cluster_{cluster_id}_color_handler",
                    f"{self.name}_cluster_{cluster_id}_text_handler"
                ]
                for handler_tag in handler_tags:
                    if dpg.does_item_exist(handler_tag):
                        dpg.delete_item(handler_tag)
            
            # Clear legend handler
            legend_handler_tag = f"{self.name}_legend_handler"
            if dpg.does_item_exist(legend_handler_tag):
                dpg.delete_item(legend_handler_tag)
        except:
            # Ignore cleanup errors
            pass
    
    def _simple_legend_setup(self):
        """Simplified legend setup without complex handlers"""
        try:
            dpg.add_text("CLUSTER LEGEND", color=[255, 255, 100])
            dpg.add_separator()
            
            # Get unique clusters
            unique_clusters = np.unique(self.data_manager.labels)
            
            # Add simple controls for each cluster (without hover handlers)
            for cluster_id in sorted(unique_clusters):
                if self.data_manager:
                    color = self.data_manager.get_cluster_color(cluster_id)
                else:
                    color = [255, 255, 255]
                
                cluster_name = f"Cluster {cluster_id}"
                
                # Simple group without complex handlers
                with dpg.group(horizontal=True):
                    # Simple color indicator
                    with dpg.drawlist(width=15, height=15):
                        dpg.draw_rectangle((0, 0), (15, 15), color=color, fill=color)
                    
                    # Simple cluster name
                    dpg.add_text(cluster_name)
                
                dpg.add_separator()
            
            # Simple show all button
            dpg.add_text("CONTROLS", color=[255, 255, 100])
            dpg.add_separator()
            dpg.add_text("Legend refresh disabled due to UI conflicts", color=[255, 100, 100])
            
        except Exception as e:
            print(f"Even simple legend setup failed for {self.name}: {e}")

    def find_closest_point(self, x_mouse, y_mouse, data, labels):
        """Find the closest data point to mouse coordinates"""
        if len(data) == 0:
            return None
        
        # Calculate distances from mouse position to all points
        mouse_pos = np.array([x_mouse, y_mouse])
        distances = np.sqrt(np.sum((data - mouse_pos) ** 2, axis=1))
        
        # Find the closest point
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # Get cluster for this point
        cluster = labels[closest_idx] if labels is not None else 0
        
        # Get coordinates
        x_coord, y_coord = data[closest_idx]
        
        return closest_idx, cluster, closest_distance, x_coord, y_coord

    def add_scatter_data(self, clusters):
        """Add scatter plot data for each cluster"""
        # Clear existing scatter series first to avoid tag conflicts
        self.clear_all_series()
        
        # Clear existing themes
        for theme_id in self.theme_ids:
            if dpg.does_item_exist(theme_id):
                dpg.delete_item(theme_id)
        self.theme_ids.clear()
        
        # Group points by cluster
        unique_clusters = np.unique(self.data_manager.labels)

        points_selected = len(self.selected_points) > 0
        
        for cluster_id in unique_clusters:
            # Skip cluster if it's not visible
            if not self.is_cluster_visible(cluster_id):
                continue
            
            # Get indices for this cluster
            cluster_indices = np.where(self.data_manager.labels == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Get data for this cluster
                cluster_data = self.X[cluster_indices]
                lasso_indices = self.lasso_selection[ np.isin( self.lasso_selection, cluster_indices ) ]
                lasso_data = self.X[ lasso_indices ] if len( lasso_indices ) > 0 else np.array([[]])

                x_data = cluster_data[:, 0].tolist()
                y_data = cluster_data[:, 1].tolist()

                lasso_x_data = lasso_data[:, 0].tolist() if len( lasso_indices) > 0 else []
                lasso_y_data = lasso_data[:, 1].tolist() if len( lasso_indices ) > 0 else []

                if points_selected:
                    mask = np.isin(self.selected_points, cluster_indices)
                    selected_in_cluster = np.array(self.selected_points)[mask]
                    if len(selected_in_cluster) > 0:
                        selected_data = self.X[selected_in_cluster]
                        selected_x_data = selected_data[:, 0].tolist()
                        selected_y_data = selected_data[:, 1].tolist()
                
                # Determine cluster name and color using DataManager
                cluster_name = f"Cluster {cluster_id}"
                base_color = self.data_manager.get_cluster_color(cluster_id)
                color = base_color
                
                if self.is_cluster_visible(cluster_id) == "unhovered" or len( self.lasso_selection ) or points_selected:
                    color = [int(c * 0.3) for c in color]
                
                # Create scatter series with unique tag
                cluster_tag = f"{self.name}_cluster_{cluster_id}"
                
                # Make sure tag doesn't already exist
                if dpg.does_item_exist(cluster_tag):
                    dpg.delete_item(cluster_tag)
                
                try:
                    dpg.add_scatter_series(
                        x_data, 
                        y_data,
                        label=cluster_name,
                        parent=f"{self.name}_y_axis",
                        tag=cluster_tag
                    )

                    if points_selected and len(selected_in_cluster) > 0:
                        dpg.add_scatter_series(
                            selected_x_data,
                            selected_y_data,
                            label=f"selected_{cluster_name}",
                            parent=f"{self.name}_y_axis",
                            tag=f"{cluster_tag}_selected"
                        )
                    
                    elif len( lasso_x_data ):
                        dpg.add_scatter_series(
                            lasso_x_data,
                            lasso_y_data,
                            label=f"lasso_{cluster_name}",
                            parent=f"{self.name}_y_axis",
                            tag=f"{cluster_tag}_lasso"
                        )
                except Exception as e:
                    print(f"Error creating scatter series for cluster {cluster_id}: {e}")
                    # Try without tag
                    try:
                        dpg.add_scatter_series(
                            x_data, 
                            y_data,
                            label=cluster_name,
                            parent=f"{self.name}_y_axis"
                        )
                        if points_selected and len(selected_in_cluster) > 0:
                            dpg.add_scatter_series(
                                selected_x_data,
                                selected_y_data,
                                label=f"selected_{cluster_name}",
                                parent=f"{self.name}_y_axis",
                            )
                        elif len( lasso_x_data ):
                            dpg.add_scatter_series(
                                lasso_x_data,
                                lasso_y_data,
                                label=f"lasso_{cluster_name}",
                                parent=f"{self.name}_y_axis",
                            )
                    except Exception as e2:
                        print(f"Failed to create scatter series even without tag: {e2}")
                        continue
                
                # Apply color theme to ensure both points and legend match
                theme_id = self.create_scatter_theme(color)
                dpg.bind_item_theme(cluster_tag, theme_id)
                self.theme_ids.append(theme_id)

                if points_selected and len(selected_in_cluster) > 0:
                    theme_id = self.create_scatter_theme(base_color)
                    dpg.bind_item_theme(f"{cluster_tag}_selected", theme_id)
                    self.theme_ids.append(theme_id)
                
                elif len( lasso_x_data ):
                    theme_id = self.create_scatter_theme(base_color)
                    dpg.bind_item_theme(f"{cluster_tag}_lasso", theme_id)
                    self.theme_ids.append(theme_id)

    def clear_all_series(self):
        """Clear all scatter series and highlights for this tab"""
        # Clear themes first
        for theme_id in self.theme_ids:
            if dpg.does_item_exist(theme_id):
                dpg.delete_item(theme_id)
        self.theme_ids.clear()
        
        # Clear main cluster series - use actual cluster IDs instead of range
        unique_clusters = np.unique(self.data_manager.labels)
        for cluster_id in unique_clusters:
            cluster_tag = f"{self.name}_cluster_{cluster_id}"
            if dpg.does_item_exist(cluster_tag):
                dpg.delete_item(cluster_tag)
                
            # Clear highlight series
            highlight_tag = f"{self.name}_cluster_{cluster_id}_selected"
            if dpg.does_item_exist(highlight_tag):
                dpg.delete_item(highlight_tag)
                
            lasso_tag = f"{self.name}_cluster_{cluster_id}_lasso"
            if dpg.does_item_exist(lasso_tag):
                dpg.delete_item(lasso_tag)
                
            # Clear similarity series
            similarity_tag = f"{self.name}_similarity_{cluster_id}"
            if dpg.does_item_exist(similarity_tag):
                dpg.delete_item(similarity_tag)
        
        # Clear individual similarity points
        for i in range(100):  # Support up to 100 similarity points
            similarity_tag = f"{self.name}_similarity_{i}"
            if dpg.does_item_exist(similarity_tag):
                dpg.delete_item(similarity_tag)

class PCATab(ScatterTab):
    def __init__(self, name, data_manager, pca_data=None):
        super().__init__(name, data_manager)

        self.pca_data = pca_data
        if self.pca_data is None:
            self.pca = PCA(n_components=5, random_state=42)  # Compute more components for flexibility
            self.pca_data = self.pca.fit_transform(self.data_manager.X)

        self.X = self.pca_data[:, :2]
    
    def setup_axis_controls(self):
        """PCA-specific axis controls"""
        dpg.add_text("PCA CONTROLS", color=[100, 255, 255])
        dpg.add_separator()
        
        # Display explained variance
        variance_ratios = self.pca.explained_variance_ratio_
        dpg.add_text("Explained Variance:")
        for i, var in enumerate(variance_ratios):
            dpg.add_text(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)", color=[200, 200, 200])
        
        dpg.add_separator()
        dpg.add_text("COMPONENT SELECTION", color=[255, 200, 100])
        
        # X-axis component selection
        with dpg.group(horizontal=True):
            dpg.add_text("X-Axis:")
            dpg.add_combo(
                items=[f"PC{i+1}" for i in range(5)],
                default_value="PC1",
                width=80,
                tag=f"{self.name}_x_component",
                callback=self.update_pca_projection
            )
        
        # Y-axis component selection  
        with dpg.group(horizontal=True):
            dpg.add_text("Y-Axis:")
            dpg.add_combo(
                items=[f"PC{i+1}" for i in range(5)],
                default_value="PC2",
                width=80,
                tag=f"{self.name}_y_component", 
                callback=self.update_pca_projection
            )
        
        dpg.add_button(label="Apply Changes", callback=self.update_pca_projection, width=-1)
        
        dpg.add_separator()
        dpg.add_text("INFORMATION", color=[255, 200, 100])
        dpg.add_text("• Select different principal components to explore", wrap=280)
        dpg.add_text("• Higher PC numbers capture less variance", wrap=280)
        dpg.add_text("• Changes update the visualization in real-time", wrap=280)
    
    def update_pca_projection(self, sender=None, app_data=None):
        """Update PCA projection with selected components"""
        try:
            x_comp_str = dpg.get_value(f"{self.name}_x_component")
            y_comp_str = dpg.get_value(f"{self.name}_y_component")
            
            # Extract component numbers (PC1 -> 0, PC2 -> 1, etc.)
            x_comp = int(x_comp_str.replace("PC", "")) - 1
            y_comp = int(y_comp_str.replace("PC", "")) - 1
            
            if x_comp == y_comp:
                print("Warning: X and Y components cannot be the same")
                return
            
            # Get the full PCA transformation and select the desired components
            self.X = self.transformed_data[:, [x_comp, y_comp]]
            
            # Update axis labels
            dpg.set_item_label(f"{self.name}_x_axis", f"PC{x_comp+1}")
            dpg.set_item_label(f"{self.name}_y_axis", f"PC{y_comp+1}")
            
            # Refresh the plot
            self.add_scatter_data([])
            
            # Refresh the custom legend
            self.refresh_custom_legend()
            
            print(f"Updated PCA to PC{x_comp+1} vs PC{y_comp+1}")
            
        except Exception as e:
            print(f"Error updating PCA projection: {e}")

class KVPTab(ScatterTab):
    def __init__(self, name, data_manager, axes, axis_labels=["axis 1", "axis 2"] ):
        super().__init__(name, data_manager)

        self.axis_labels = axis_labels
        self.axes = axes

        proj = get_projection(self.data_manager.X, axes[0], axes[1])
        self.X = proj
    
    def setup_axis_controls(self):
        print( "setup_axis_controls" )
        """Vector Projection specific axis controls"""
        dpg.add_text("VECTOR PROJECTION", color=[255, 100, 255])
        dpg.add_separator()
        
        dpg.add_text("Current Projection:")
        dpg.add_text("• Semantic axis-based projection", color=[200, 200, 200])
        dpg.add_text("• Each axis represents a concept direction", color=[200, 200, 200])
        
        dpg.add_separator()
        dpg.add_text("CREATE NEW PROJECTION", color=[255, 200, 100])
        
        dpg.add_text("Axis 1 (From → To):")
        with dpg.group(horizontal=True):
            dpg.add_input_text(hint="From text", width=130, tag=f"{self.name}_axis1_from")
            dpg.add_input_text(hint="To text", width=130, tag=f"{self.name}_axis1_to")
            dpg.add_button(label="Update Axis 1", callback=self.update_projection, user_data="axis1", width=130)
        
        dpg.add_text("Axis 2 (From → To):")
        with dpg.group(horizontal=True):
            dpg.add_input_text(hint="From text", width=130, tag=f"{self.name}_axis2_from")
            dpg.add_input_text(hint="To text", width=130, tag=f"{self.name}_axis2_to")
            dpg.add_button(label="Update Axis 2", callback=self.update_projection, user_data="axis2", width=130)
        
        dpg.add_button(label="Update Both", callback=self.update_projection, user_data="all", width=-1)
        
        dpg.add_separator()
        dpg.add_text("EXAMPLES", color=[100, 255, 100])
        dpg.add_text("• negative → positive", wrap=280, color=[150, 150, 150])
        dpg.add_text("• small → large", wrap=280, color=[150, 150, 150])
        dpg.add_text("• past → future", wrap=280, color=[150, 150, 150])
        
        dpg.add_separator()
        dpg.add_text("INFORMATION", color=[255, 200, 100])
        dpg.add_text("• Creates semantic axes from text concepts", wrap=280)
        dpg.add_text("• Projects data onto these conceptual directions", wrap=280)
        dpg.add_text("• Updates visualization in real-time", wrap=280)
    
    def update_projection(self, sender=None, app_data=None, user_data=None):
        """Update the vector projection with new axes"""
        try:
            # Get input values
            texts = []
            if user_data == "axis1" or user_data == "all":
                from_1 = dpg.get_value(f"{self.name}_axis1_from").strip()
                to_1 = dpg.get_value(f"{self.name}_axis1_to").strip()   
                texts.append(from_1)
                texts.append(to_1)
                self.axis_labels[0] = f"{from_1} → {to_1}"
                if not from_1 or not to_1:
                    print("Error: All axis fields must be filled")
                    return
            if user_data == "axis2" or user_data == "all":
                from_2 = dpg.get_value(f"{self.name}_axis2_from").strip()
                to_2 = dpg.get_value(f"{self.name}_axis2_to").strip()
                texts.append(from_2)
                texts.append(to_2)
                self.axis_labels[1] = f"{from_2} → {to_2}"
                if not from_2 or not to_2:
                    print("Error: All axis fields must be filled")
                    return
            # Get embeddings for the text
            embeddings = encode(texts)
            
            # Create new axes
            new_axes = [
                embeddings[1] - embeddings[0] if user_data == "axis1" or user_data == "all" else self.axes[0],  # to_1 - from_1
                embeddings[3] - embeddings[2] if user_data == "all" else ( embeddings[1] - embeddings[0] if user_data == "axis2" else self.axes[1] )   # to_2 - from_2
            ]
            
            # Update the tab
            self.axes = new_axes
            proj = get_projection(self.data_manager.X, new_axes[0], new_axes[1])
            self.X = proj
            
            # Update axis labels
            if user_data == "axis1" or user_data == "all":
                dpg.set_item_label(f"{self.name}_x_axis", f"{from_1} → {to_1}")
            if user_data == "axis2" or user_data == "all":
                dpg.set_item_label(f"{self.name}_y_axis", f"{from_2} → {to_2}")
            
            # Refresh the plot
            self.add_scatter_data([])
            
            # Clear inputs
            dpg.set_value(f"{self.name}_axis1_from", "")
            dpg.set_value(f"{self.name}_axis1_to", "")
            dpg.set_value(f"{self.name}_axis2_from", "")
            dpg.set_value(f"{self.name}_axis2_to", "")
            
            print(f"Updated projection")
            
        except Exception as e:
            print(f"Error updating projection: {e}")

class TSNETab(ScatterTab):
    def __init__(self, name, data_manager, perplexity=30):
        
        super().__init__(name, data_manager)

        self.perplexity = min(30, len(self.data_manager.X)-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=self.perplexity)
        transformed_data = tsne.fit_transform(self.data_manager.X)
        self.X = transformed_data
    
    def setup_axis_controls(self):
        """t-SNE specific axis controls"""
        dpg.add_text("t-SNE CONTROLS", color=[255, 150, 100])
        dpg.add_separator()
        
        dpg.add_text("Current Parameters:")
        dpg.add_text(f"• Perplexity: {self.perplexity}", color=[200, 200, 200])
        dpg.add_text("• Random State: 42", color=[200, 200, 200])
        dpg.add_text("• Components: 2", color=[200, 200, 200])
        
        dpg.add_separator()
        dpg.add_text("PARAMETER ADJUSTMENT", color=[255, 200, 100])
        
        with dpg.group(horizontal=True):
            dpg.add_text("Perplexity:")
            dpg.add_slider_int(
                min_value=5,
                max_value=min(100, len(self.data_manager.X)//4),
                default_value=self.perplexity,
                width=150,
                tag=f"{self.name}_perplexity"
            )
        
        dpg.add_button(label="Recompute t-SNE", callback=self.recompute_tsne, width=-1)
        
        dpg.add_separator()
        dpg.add_text("INFORMATION", color=[255, 200, 100])
        dpg.add_text("• t-SNE preserves local structure", wrap=280)
        dpg.add_text("• Perplexity controls local vs global structure", wrap=280)
        dpg.add_text("• Lower values focus on local details", wrap=280)
        dpg.add_text("• Higher values preserve global structure", wrap=280)
        dpg.add_text("• Axes are not directly interpretable", wrap=280)
    
    def recompute_tsne(self, sender=None, app_data=None):
        """Recompute t-SNE with new parameters"""
        try:
            new_perplexity = dpg.get_value(f"{self.name}_perplexity")
            
            print(f"Recomputing t-SNE with perplexity={new_perplexity}...")
            
            # Recompute t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=new_perplexity)
            self.X = tsne.fit_transform(self.data_manager.X)
            self.perplexity = new_perplexity
            
            # Update the display
            dpg.set_value(f"{self.name}_perplexity", new_perplexity)
            
            # Refresh the plot
            self.add_scatter_data([])
            
            # Refresh the custom legend
            self.refresh_custom_legend()
            
            print(f"t-SNE recomputed with perplexity={new_perplexity}")
            
        except Exception as e:
            print(f"Error recomputing t-SNE: {e}")

class UMAPTab(ScatterTab):
    def __init__(self, name, data_manager, umap_data=None):
        self.n_neighbors = 15
        self.min_dist = 0.1
        
        super().__init__(name, data_manager)

        if umap_data is None:
            reducer = umap.UMAP(n_components=2, n_neighbors=self.n_neighbors, min_dist=self.min_dist, unique=True)
            umap_data = reducer.fit_transform(data_manager.X)

        self.X = umap_data
    def setup_axis_controls(self):
        """UMAP specific axis controls"""

        dpg.add_text("UMAP CONTROLS", color=[100, 255, 150])
        dpg.add_separator()
        
        dpg.add_text("Current Parameters:")
        dpg.add_text(f"• n_neighbors: {self.n_neighbors}", color=[200, 200, 200])
        dpg.add_text(f"• min_dist: {self.min_dist}", color=[200, 200, 200])
        dpg.add_text("• Random State: 42", color=[200, 200, 200])
        
        dpg.add_separator()
        dpg.add_text("PARAMETER ADJUSTMENT", color=[255, 200, 100])
        
        with dpg.group(horizontal=True):
            dpg.add_text("n_neighbors:")
            dpg.add_slider_int(
                min_value=2,
                max_value=min(200, len(self.data_manager.X)//4),
                default_value=self.n_neighbors,
                width=150,
                tag=f"{self.name}_n_neighbors"
            )
        
        with dpg.group(horizontal=True):
            dpg.add_text("min_dist:")
            dpg.add_slider_float(
                min_value=0.0,
                max_value=1.0,
                default_value=self.min_dist,
                width=150,
                tag=f"{self.name}_min_dist"
            )
        
        dpg.add_button(label="Recompute UMAP", callback=self.recompute_umap, width=-1)
        
        dpg.add_separator()
        dpg.add_text("INFORMATION", color=[255, 200, 100])
        dpg.add_text("• UMAP preserves both local and global structure", wrap=280)
        dpg.add_text("• n_neighbors: controls local vs global structure", wrap=280)
        dpg.add_text("• min_dist: controls how tightly points cluster", wrap=280)
        dpg.add_text("• Lower min_dist = tighter clusters", wrap=280)
    
    def recompute_umap(self, sender=None, app_data=None):
        """Recompute UMAP with new parameters"""

        try:
            new_n_neighbors = dpg.get_value(f"{self.name}_n_neighbors")
            new_min_dist = dpg.get_value(f"{self.name}_min_dist")
            
            print(f"Recomputing UMAP with n_neighbors={new_n_neighbors}, min_dist={new_min_dist}...")
            
            # Recompute UMAP
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=new_n_neighbors, min_dist=new_min_dist)
            self.X = reducer.fit_transform(self.data_manager.X)
            self.n_neighbors = new_n_neighbors
            self.min_dist = new_min_dist
            
            # Refresh the plot
            self.add_scatter_data([])
            
            # Refresh the custom legend
            self.refresh_custom_legend()
            
            print(f"UMAP recomputed with n_neighbors={new_n_neighbors}, min_dist={new_min_dist}")
            
        except Exception as e:
            print(f"Error recomputing UMAP: {e}")

class NGonTab(ScatterTab):
    def __init__(self, name, data_manager):
        super().__init__(name, data_manager)

        # self.pca = PCA(n_components=self.data_manager.X.shape[1], random_state=42)  # Compute more components for flexibility
        # self.pca_data = self.pca.fit_transform(self.data_manager.X)
        self.X = np.array([[]])

        self.n_gons = np.zeros( ( self.data_manager.X.shape[ 0 ], 2, self.data_manager.X.shape[ 1 ] ) )
        for i in range(self.data_manager.X.shape[1]):
            angle = i * 2 * np.pi / self.data_manager.X.shape[1]
            length = self.data_manager.X[ :, i ] + 0.2
            self.n_gons[ :, 0, i ] = length * np.cos(angle)
            self.n_gons[ :, 1, i ] = length * np.sin(angle)


    def setup_axis_controls(self):
        """NGon specific axis controls"""

        dpg.add_text("NGON CONTROLS", color=[100, 255, 150])

    def add_scatter_data( self, clusters ):
        """Add scatter plot data for each cluster"""
        # Clear existing scatter series first to avoid tag conflicts
        self.clear_all_series()
        
        # Clear existing themes
        for theme_id in self.theme_ids:
            if dpg.does_item_exist(theme_id):
                dpg.delete_item(theme_id)
        self.theme_ids.clear()
        
        # Group points by cluster
        unique_clusters = np.unique(self.data_manager.labels)

        points_selected = len(self.selected_points) > 0
        
        for cluster_id in unique_clusters:
            # Skip cluster if it's not visible
            if not self.is_cluster_visible(cluster_id):
                continue
            
            # Get indices for this cluster
            cluster_indices = np.where(self.data_manager.labels == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Get data for this cluster
                cluster_data = np.mean( self.n_gons[cluster_indices], axis=0 )
                lasso_indices = self.lasso_selection[ np.isin( self.lasso_selection, cluster_indices ) ]
                lasso_data = np.mean( self.n_gons[lasso_indices], axis=0 ) if len( lasso_indices ) > 0 else np.array([[]])

                print( cluster_data.shape )

                x_data = cluster_data[ 0, :].tolist()
                y_data = cluster_data[ 1, :].tolist()

                lasso_x_data = lasso_data[ 0, :].tolist() if len( lasso_indices) > 0 else []
                lasso_y_data = lasso_data[ 1, :].tolist() if len( lasso_indices ) > 0 else []

                if points_selected:
                    mask = np.isin(self.selected_points, cluster_indices)
                    selected_in_cluster = np.array(self.selected_points)[mask]
                    if len(selected_in_cluster) > 0:
                        selected_data = np.mean( self.n_gons[selected_in_cluster], axis=0 )
                        selected_x_data = selected_data[ 0, :].tolist()
                        selected_y_data = selected_data[ 1, :].tolist()
                
                # Determine cluster name and color using DataManager
                cluster_name = f"Cluster {cluster_id}"
                base_color = self.data_manager.get_cluster_color(cluster_id)
                color = base_color
                
                if self.is_cluster_visible(cluster_id) == "unhovered" or len( self.lasso_selection ) or points_selected:
                    color = [int(c * 0.3) for c in color]
                
                # Create scatter series with unique tag
                cluster_tag = f"{self.name}_cluster_{cluster_id}"
                
                # Make sure tag doesn't already exist
                if dpg.does_item_exist(cluster_tag):
                    dpg.delete_item(cluster_tag)
                
                try:
                    dpg.add_area_series(
                        x_data, 
                        y_data,
                        label=cluster_name,
                        parent=f"{self.name}_y_axis",
                        tag=cluster_tag
                    )

                    if points_selected and len(selected_in_cluster) > 0:
                        dpg.add_area_series(
                            selected_x_data,
                            selected_y_data,
                            label=f"selected_{cluster_name}",
                            parent=f"{self.name}_y_axis",
                            tag=f"{cluster_tag}_selected"
                        )
                    
                    elif len( lasso_x_data ):
                        dpg.add_area_series(
                            lasso_x_data,
                            lasso_y_data,
                            label=f"lasso_{cluster_name}",
                            parent=f"{self.name}_y_axis",
                            tag=f"{cluster_tag}_lasso"
                        )
                except Exception as e:
                    print(f"Error creating scatter series for cluster {cluster_id}: {e}")
                    # Try without tag
                    try:
                        dpg.add_area_series(
                            x_data, 
                            y_data,
                            label=cluster_name,
                            parent=f"{self.name}_y_axis"
                        )
                        if points_selected and len(selected_in_cluster) > 0:
                            dpg.add_area_series(
                                selected_x_data,
                                selected_y_data,
                                label=f"selected_{cluster_name}",
                                parent=f"{self.name}_y_axis",
                            )
                        elif len( lasso_x_data ):
                            dpg.add_area_series(
                                lasso_x_data,
                                lasso_y_data,
                                label=f"lasso_{cluster_name}",
                                parent=f"{self.name}_y_axis",
                            )
                    except Exception as e2:
                        print(f"Failed to create scatter series even without tag: {e2}")
                        continue
                
                # Apply color theme to ensure both points and legend match
                theme_id = self.create_scatter_theme(color)
                dpg.bind_item_theme(cluster_tag, theme_id)
                self.theme_ids.append(theme_id)

                if points_selected and len(selected_in_cluster) > 0:
                    theme_id = self.create_scatter_theme(base_color)
                    dpg.bind_item_theme(f"{cluster_tag}_selected", theme_id)
                    self.theme_ids.append(theme_id)
                
                elif len( lasso_x_data ):
                    theme_id = self.create_scatter_theme(base_color)
                    dpg.bind_item_theme(f"{cluster_tag}_lasso", theme_id)
                    self.theme_ids.append(theme_id)

    def clear_all_series(self):
        """Clear all scatter series and highlights for this tab"""
        # Clear themes first
        for theme_id in self.theme_ids:
            if dpg.does_item_exist(theme_id):
                dpg.delete_item(theme_id)
        self.theme_ids.clear()
        
        # Clear main cluster series - use actual cluster IDs instead of range
        unique_clusters = np.unique(self.data_manager.labels)
        for cluster_id in unique_clusters:
            cluster_tag = f"{self.name}_cluster_{cluster_id}"
            if dpg.does_item_exist(cluster_tag):
                dpg.delete_item(cluster_tag)
                
            # Clear highlight series
            highlight_tag = f"{self.name}_cluster_{cluster_id}_selected"
            if dpg.does_item_exist(highlight_tag):
                dpg.delete_item(highlight_tag)
                
            lasso_tag = f"{self.name}_cluster_{cluster_id}_lasso"
            if dpg.does_item_exist(lasso_tag):
                dpg.delete_item(lasso_tag)
                
            # Clear similarity series
            similarity_tag = f"{self.name}_similarity_{cluster_id}"
            if dpg.does_item_exist(similarity_tag):
                dpg.delete_item(similarity_tag)
        
        # Clear individual similarity points
        for i in range(100):  # Support up to 100 similarity points
            similarity_tag = f"{self.name}_similarity_{i}"
            if dpg.does_item_exist(similarity_tag):
                dpg.delete_item(similarity_tag)

class MetadataTab(ScatterTab):
    """Tab for visualizing metadata columns as X/Y axes"""
    def __init__(self, name, data_manager, x_data, y_data, axis_labels=["X-Axis", "Y-Axis"]):
        # Initialize data first
        self.x_data = x_data
        self.y_data = y_data
        self.axis_labels = axis_labels
        
        # Create 2D projection from the data
        self.X = np.column_stack([x_data, y_data])
        
        # Call parent constructor without transformation
        super().__init__(name, data_manager, transformation=None)
    
    def setup_axis_controls(self):
        """Setup axis controls specific to metadata visualization"""
        dpg.add_text("METADATA VISUALIZATION", color=[255, 255, 100])
        dpg.add_separator()
        dpg.add_text(f"X-Axis: {self.axis_labels[0]}")
        dpg.add_text(f"Y-Axis: {self.axis_labels[1]}")
        dpg.add_separator()
        
        # Statistics
        dpg.add_text("AXIS STATISTICS", color=[200, 200, 100])
        dpg.add_text(f"X-Axis Range: {self.x_data.min():.2f} to {self.x_data.max():.2f}")
        dpg.add_text(f"Y-Axis Range: {self.y_data.min():.2f} to {self.y_data.max():.2f}")
        dpg.add_text(f"X-Axis Mean: {self.x_data.mean():.2f}")
        dpg.add_text(f"Y-Axis Mean: {self.y_data.mean():.2f}")
        dpg.add_separator()
        
        dpg.add_text("This visualization shows the relationship between two metadata columns.", wrap=280, color=[180, 180, 180])


class CustomEncodedTab(ScatterTab):
    """Tab for visualizing custom encoded axes (text-based projections)"""
    def __init__(self, name, data_manager, x_data, y_data, axis_labels=["X-Axis", "Y-Axis"]):
        # Initialize data first
        self.x_data = x_data
        self.y_data = y_data
        self.axis_labels = axis_labels
        
        # Create 2D projection from the data
        self.X = np.column_stack([x_data, y_data])
        
        # Call parent constructor without transformation
        super().__init__(name, data_manager, transformation=None)
    
    def setup_axis_controls(self):
        """Setup axis controls specific to custom encoded visualization"""
        dpg.add_text("CUSTOM ENCODED AXES", color=[255, 255, 100])
        dpg.add_separator()
        dpg.add_text(f"X-Axis: {self.axis_labels[0]}")
        dpg.add_text(f"Y-Axis: {self.axis_labels[1]}")
        dpg.add_separator()
        
        # Statistics
        dpg.add_text("PROJECTION STATISTICS", color=[200, 200, 100])
        dpg.add_text(f"X-Axis Range: {self.x_data.min():.2f} to {self.x_data.max():.2f}")
        dpg.add_text(f"Y-Axis Range: {self.y_data.min():.2f} to {self.y_data.max():.2f}")
        dpg.add_text(f"X-Axis Mean: {self.x_data.mean():.2f}")
        dpg.add_text(f"Y-Axis Mean: {self.y_data.mean():.2f}")
        dpg.add_text(f"X-Axis Std: {self.x_data.std():.2f}")
        dpg.add_text(f"Y-Axis Std: {self.y_data.std():.2f}")
        dpg.add_separator()
        
        # Correlation info
        correlation = np.corrcoef(self.x_data, self.y_data)[0, 1]
        dpg.add_text(f"Axis Correlation: {correlation:.3f}")
        dpg.add_separator()
        
        dpg.add_text("This visualization shows projections of the data onto custom semantic axes created from text encoding.", wrap=280, color=[180, 180, 180])