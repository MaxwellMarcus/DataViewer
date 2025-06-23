#!/usr/bin/env python3
"""
Test runner script for the DataVisualization project.

This script provides convenient ways to run different types of tests:
- Unit tests
- Integration tests  
- Performance tests
- GUI tests (optional)
- Coverage reports

Usage:
    python run_tests.py                    # Run all unit tests
    python run_tests.py --all              # Run all tests including integration/performance
    python run_tests.py --integration      # Run integration tests only
    python run_tests.py --performance      # Run performance tests only
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --gui              # Include GUI tests
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --module data_manager  # Test specific module
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False

def check_dependencies():
    """Check if required testing dependencies are installed."""
    try:
        import pytest
        import pytest_cov
        import pytest_mock
        print("✅ All testing dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing testing dependency: {e}")
        print("Install testing dependencies with: pip install -r requirements.txt")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run tests for DataVisualization project")
    
    # Test type selection
    parser.add_argument('--all', action='store_true', 
                       help='Run all tests including integration and performance')
    parser.add_argument('--unit', action='store_true', default=True,
                       help='Run unit tests only (default)')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests only')
    parser.add_argument('--gui', action='store_true',
                       help='Include GUI tests (normally skipped)')
    
    # Test filtering
    parser.add_argument('--module', type=str,
                       help='Run tests for specific module (e.g., data_manager, scatter_tabs)')
    parser.add_argument('--test', type=str,
                       help='Run specific test function or class')
    parser.add_argument('--fast', action='store_true',
                       help='Skip slow tests (marked with @pytest.mark.slow)')
    
    # Output options
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--html-cov', action='store_true',
                       help='Generate HTML coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output')
    
    # Development options
    parser.add_argument('--pdb', action='store_true',
                       help='Drop into debugger on failures')
    parser.add_argument('--lf', action='store_true',
                       help='Run only last failed tests')
    parser.add_argument('--ff', action='store_true',
                       help='Run failures first')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Add test path
    test_path = 'tests/'
    
    # Handle module-specific testing
    if args.module:
        test_path = f'tests/test_{args.module}.py'
        if not os.path.exists(test_path):
            print(f"❌ Test file not found: {test_path}")
            print("Available test modules:")
            test_files = list(Path('tests').glob('test_*.py'))
            for test_file in test_files:
                module_name = test_file.stem.replace('test_', '')
                print(f"  - {module_name}")
            return 1
    
    # Handle specific test
    if args.test:
        if args.module:
            test_path = f'{test_path}::{args.test}'
        else:
            test_path = f'tests/ -k {args.test}'
    
    cmd.append(test_path)
    
    # Add test type markers
    test_markers = []
    
    if args.all:
        # Include all tests
        pass
    elif args.integration:
        test_markers.append('integration')
    elif args.performance:
        test_markers.append('performance')
    elif args.gui:
        test_markers.append('gui')
    else:
        # Default: unit tests only
        test_markers.append('unit or not integration and not performance and not gui')
    
    if args.fast:
        test_markers.append('not slow')
    
    if not args.gui and 'gui' not in test_markers:
        test_markers.append('not gui')
    
    if test_markers:
        cmd.extend(['-m', ' and '.join(test_markers)])
    
    # Add output options
    if args.verbose:
        cmd.append('-v')
    elif args.quiet:
        cmd.append('-q')
    else:
        cmd.append('-v')  # Default to verbose
    
    # Add coverage options
    if args.coverage or args.html_cov:
        cmd.extend(['--cov=.'])
        cmd.extend(['--cov-report=term-missing'])
        
        if args.html_cov:
            cmd.extend(['--cov-report=html'])
    
    # Add development options
    if args.pdb:
        cmd.append('--pdb')
    if args.lf:
        cmd.append('--lf')
    if args.ff:
        cmd.append('--ff')
    
    # Additional pytest options for better output
    cmd.extend([
        '--tb=short',           # Shorter traceback format
        '--disable-warnings',   # Disable warnings for cleaner output
    ])
    
    # Run tests
    success = run_command(cmd, "Running tests")
    
    if success:
        print(f"\n🎉 Tests completed successfully!")
        
        if args.coverage or args.html_cov:
            print("\n📊 Coverage report generated")
            if args.html_cov:
                print("📋 HTML coverage report: htmlcov/index.html")
        
        # Show additional information
        print("\n📝 Test Summary:")
        print(f"   Test path: {test_path}")
        if test_markers:
            print(f"   Test markers: {' and '.join(test_markers)}")
        
        return 0
    else:
        print(f"\n💥 Tests failed!")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if all dependencies are installed: pip install -r requirements.txt")
        print("   2. Make sure you're in the project root directory")
        print("   3. Try running individual test modules: python run_tests.py --module data_manager")
        print("   4. Use --pdb to debug failing tests")
        
        return 1

def show_test_info():
    """Show information about available tests."""
    print("📋 Available Test Modules:")
    print("=" * 40)
    
    test_files = list(Path('tests').glob('test_*.py'))
    for test_file in test_files:
        module_name = test_file.stem.replace('test_', '')
        print(f"  🧪 {module_name:<15} - {test_file}")
    
    print(f"\n📊 Test Statistics:")
    print("=" * 40)
    
    # Count tests
    try:
        result = subprocess.run(['python', '-m', 'pytest', '--collect-only', '-q'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:  # Show last few lines which contain summary
                if 'test' in line:
                    print(f"  {line}")
    except:
        print("  Could not collect test statistics")
    
    print(f"\n🏷️  Test Markers:")
    print("=" * 40)
    print("  🔹 unit        - Fast unit tests")
    print("  🔹 integration - Integration tests")  
    print("  🔹 performance - Performance benchmarks")
    print("  🔹 gui         - GUI component tests")
    print("  🔹 slow        - Tests that take longer to run")

if __name__ == '__main__':
    # Show test info if no arguments provided
    if len(sys.argv) == 1:
        show_test_info()
        print(f"\n🚀 Run tests with: python run_tests.py --help")
        sys.exit(0)
    
    sys.exit(main()) 