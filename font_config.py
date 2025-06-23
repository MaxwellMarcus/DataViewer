#!/usr/bin/env python3
"""
Font configuration for Dear PyGui application
Handles Arial font loading and setup
"""

import dearpygui.dearpygui as dpg
import os
import platform


class FontManager:
    """Manages font loading and configuration for Dear PyGui"""
    
    def __init__(self):
        self.font_registry = None
        self.default_font = None
        self.arial_font_path = None
        self.font_size = 14
        
    def find_arial_font(self):
        """Find Arial font file based on the operating system"""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            possible_paths = [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "/Library/Fonts/Arial.ttf"
            ]
        elif system == "Windows":
            possible_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/Arial.ttf"
            ]
        elif system == "Linux":
            possible_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Fallback
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Fallback
                "/usr/share/fonts/TTF/arial.ttf",
                "/usr/share/fonts/arial.ttf",
                "/System/Library/Fonts/Arial.ttf"
            ]
        else:
            possible_paths = []
        
        # Find the first existing font file
        for path in possible_paths:
            if os.path.exists(path):
                self.arial_font_path = path
                print(f"Found font: {path}")
                return path
        
        print("Warning: Arial font not found, using default Dear PyGui font")
        return None
    
    def setup_fonts(self, font_size=14):
        """Setup fonts for Dear PyGui application"""
        self.font_size = font_size
        
        # Find Arial font
        font_path = self.find_arial_font()
        if not font_path:
            print("Using default Dear PyGui font")
            return None
        
        try:            
            # Create font registry
            with dpg.font_registry() as font_registry:
                self.font_registry = font_registry
                
                # Add Arial font
                self.default_font = dpg.add_font(font_path, font_size * 2 )                
                # Add different sizes if needed
                self.large_font = dpg.add_font(font_path, font_size * 2 + 4)
                self.small_font = dpg.add_font(font_path, font_size * 2 - 2)

            dpg.set_global_font_scale( 0.5 )
                
            print(f"Successfully loaded Arial font from: {font_path}")
            return self.default_font
            
        except Exception as e:
            print(f"Error loading Arial font: {e}")
            print("Using default Dear PyGui font")
            return None
    
    def apply_font(self):
        """Apply the default font to Dear PyGui"""
        if self.default_font:
            dpg.bind_font(self.default_font)
            print("Applied Arial font to application")
        else:
            print("No custom font to apply")
    
    def get_font(self, size="default"):
        """Get font by size"""
        if size == "large":
            return getattr(self, 'large_font', self.default_font)
        elif size == "small":
            return getattr(self, 'small_font', self.default_font)
        else:
            return self.default_font


# Global font manager instance
font_manager = FontManager()


def setup_fonts(font_size=14):
    """Convenience function to setup fonts"""
    return font_manager.setup_fonts(font_size)


def apply_fonts():
    """Convenience function to apply fonts"""
    font_manager.apply_font()


def get_font(size="default"):
    """Convenience function to get font"""
    return font_manager.get_font(size) 