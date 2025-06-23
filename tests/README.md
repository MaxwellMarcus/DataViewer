# DataVisualization Testing Suite

This directory contains comprehensive unit tests for the DataVisualization project. The testing suite is designed to ensure code quality, catch regressions, and provide confidence when making changes.

## 🚀 Quick Start

```bash
# Run all unit tests
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run specific module tests
python run_tests.py --module data_manager

# Run all tests including integration and performance
python run_tests.py --all
```

## 📁 Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── test_data_manager.py     # Tests for DataManager class
├── test_scatter_tabs.py     # Tests for ScatterTab classes
├── test_tab_manager.py      # Tests for TabManager class
├── test_app.py             # Tests for main application
├── test_utilities.py       # Tests for utility functions
└── README.md               # This file
```

## 🧪 Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **Purpose**: Test individual functions and classes in isolation
- **Speed**: Fast (< 1 second each)
- **Scope**: Single components with mocked dependencies
- **Examples**: DataManager methods, utility functions, point selection logic

### Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test interaction between multiple components
- **Speed**: Medium (1-10 seconds each)
- **Scope**: Multiple classes working together
- **Examples**: Complete data loading pipeline, tab coordination

### Performance Tests (`@pytest.mark.performance`)
- **Purpose**: Ensure acceptable performance with large datasets
- **Speed**: Slow (10+ seconds each)
- **Scope**: Scalability and efficiency testing
- **Examples**: Large dataset processing, clustering performance

### GUI Tests (`@pytest.mark.gui`)
- **Purpose**: Test GUI components (normally skipped)
- **Speed**: Variable
- **Scope**: DearPyGUI interface components
- **Examples**: UI creation, event handling

## 🔧 Running Tests

### Basic Commands

```bash
# Show available tests and info
python run_tests.py

# Run unit tests only (default)
python run_tests.py --unit

# Run all test types
python run_tests.py --all

# Run specific test categories
python run_tests.py --integration
python run_tests.py --performance
python run_tests.py --gui
```

### Filtering Tests

```bash
# Test specific module
python run_tests.py --module data_manager
python run_tests.py --module scatter_tabs

# Test specific function or class
python run_tests.py --test test_point_selection
python run_tests.py --test TestDataManagerInitialization

# Skip slow tests
python run_tests.py --fast
```

### Coverage Reports

```bash
# Terminal coverage report
python run_tests.py --coverage

# HTML coverage report
python run_tests.py --html-cov
# Open htmlcov/index.html in browser
```

### Development Options

```bash
# Debug failing tests
python run_tests.py --pdb

# Run only last failed tests
python run_tests.py --lf

# Run failures first
python run_tests.py --ff

# Verbose output
python run_tests.py --verbose
```

## 🏗️ Test Architecture

### Fixtures (conftest.py)

The testing suite uses pytest fixtures to provide reusable test data:

- **`sample_embeddings`**: Reproducible embedding data (100x384)
- **`sample_metadata`**: Diverse metadata DataFrame
- **`sample_labels`**: Cluster labels for testing
- **`mock_dpg`**: Mocked DearPyGUI to avoid GUI dependencies
- **`mock_sklearn`**: Mocked scikit-learn for faster tests
- **`data_manager_instance`**: Ready-to-use DataManager instance
- **`scatter_tab_instance`**: Ready-to-use ScatterTab instance
- **`temp_files`**: Temporary test files (auto-cleanup)
- **`performance_data`**: Large datasets for performance testing

### Mocking Strategy

Tests extensively use mocking to:
- **Avoid GUI dependencies**: Mock DearPyGUI calls
- **Speed up tests**: Mock slow ML operations (UMAP, t-SNE, HDBSCAN)
- **Isolate components**: Test individual classes without dependencies
- **Control randomness**: Use fixed seeds for reproducible tests

### Test Data

All test data is generated programmatically to ensure:
- **Reproducibility**: Fixed random seeds
- **Variety**: Different data types and edge cases
- **Scalability**: Multiple dataset sizes for performance testing
- **Realism**: Data that mimics real-world usage patterns

## 🎯 Test Coverage Goals

The testing suite aims for:
- **Line Coverage**: >85% of non-GUI code
- **Branch Coverage**: >80% of decision points
- **Function Coverage**: >90% of public methods
- **Class Coverage**: 100% of main classes

### Current Coverage Areas

✅ **Well Covered**:
- DataManager core functionality
- Clustering operations
- Point selection logic
- File loading and validation
- Mathematical utilities

🔄 **Partially Covered**:
- GUI event handling (mocked)
- Complex user interactions
- Error recovery scenarios

⚠️ **Needs Improvement**:
- Edge cases with corrupted data
- Network-dependent operations
- Very large dataset scenarios

## 📝 Writing New Tests

### Test Naming Conventions

```python
class TestClassName:
    """Test ClassName functionality."""
    
    def test_method_name_behavior(self, fixtures):
        """Test that method_name does specific_behavior."""
        # Arrange
        setup_data = create_test_data()
        
        # Act
        result = object_under_test.method_name(setup_data)
        
        # Assert
        assert result.expected_property == expected_value
```

### Test Organization

1. **Group by functionality**: Related tests in the same class
2. **Descriptive names**: Clear test purpose from the name
3. **AAA pattern**: Arrange, Act, Assert structure
4. **Single responsibility**: One concept per test
5. **Independent tests**: No dependencies between tests

### Using Fixtures

```python
def test_data_manager_clustering(self, data_manager_instance):
    """Test clustering with pre-configured DataManager."""
    result = data_manager_instance.cluster_by_hdbscan(min_cluster_size=10)
    assert result is True

def test_with_custom_data(self, sample_embeddings, sample_metadata):
    """Test with custom configuration."""
    dm = DataManager(sample_embeddings, sample_metadata)
    # ... test implementation
```

### Mocking Guidelines

```python
# Mock external dependencies
@patch('module.external_dependency')
def test_with_mocked_dependency(self, mock_dependency):
    mock_dependency.return_value = expected_result
    # ... test implementation

# Use fixture mocks for consistency
def test_with_dpg_mock(self, mock_dpg):
    mock_dpg.get_value.return_value = "test_value"
    # ... test implementation
```

### Performance Test Guidelines

```python
@pytest.mark.performance
def test_large_dataset_performance(self, performance_data):
    """Test performance with large dataset."""
    large_data = performance_data[5000]  # 5000 data points
    
    import time
    start_time = time.time()
    
    # Operation under test
    result = process_large_dataset(large_data)
    
    processing_time = time.time() - start_time
    
    # Performance assertion
    assert processing_time < 2.0, f"Processing took too long: {processing_time:.2f}s"
    
    # Correctness assertion
    assert result.is_valid()
```

## 🐛 Debugging Tests

### Common Issues and Solutions

**1. Import Errors**
```bash
# Make sure you're in the project root
cd /path/to/DataVisualization

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

**2. Mock Issues**
```python
# Check mock is being called
mock_function.assert_called_once()

# Check mock return value
mock_function.return_value = expected_value

# Reset mocks between tests
mock_function.reset_mock()
```

**3. Fixture Problems**
```python
# Check fixture scope
@pytest.fixture(scope="function")  # Reset for each test
@pytest.fixture(scope="session")   # Share across session

# Debug fixture values
def test_debug_fixture(self, sample_data):
    print(f"Sample data shape: {sample_data.shape}")
    assert False  # Will show the print output
```

**4. GUI Test Issues**
```bash
# Skip GUI tests if causing problems
python run_tests.py --fast

# Run with GUI tests explicitly
python run_tests.py --gui
```

### Debugging Commands

```bash
# Drop into debugger on failure
python run_tests.py --pdb

# Run single test with maximum verbosity
python run_tests.py --test test_specific_function -v

# Show local variables in tracebacks
python run_tests.py --tb=long

# Capture print statements
python run_tests.py -s
```

## 📊 Continuous Integration

The test suite is designed to work in CI/CD environments:

### GitHub Actions Configuration

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - run: pip install -r requirements.txt
      - run: python run_tests.py --coverage
```

### Local Pre-commit Hook

```bash
# Install pre-commit hook
echo "python run_tests.py --fast" > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## 🔄 Maintenance

### Regular Tasks

1. **Update test data**: Keep fixtures current with real data patterns
2. **Review coverage**: Identify and test uncovered code paths  
3. **Performance benchmarks**: Monitor performance regressions
4. **Dependency updates**: Keep testing libraries current
5. **Documentation**: Update this README with new patterns

### Adding New Test Categories

1. **Create marker**: Add to `pytest.ini`
2. **Update runner**: Modify `run_tests.py`
3. **Document usage**: Update this README
4. **Add examples**: Create sample tests

## 🤝 Contributing

When contributing new features:

1. **Write tests first**: TDD approach preferred
2. **Maintain coverage**: Don't decrease coverage percentage  
3. **Add integration tests**: For multi-component features
4. **Update docs**: Include any new testing patterns
5. **Run full suite**: Ensure no regressions

### Pull Request Checklist

- [ ] All tests pass: `python run_tests.py --all`
- [ ] Coverage maintained: `python run_tests.py --coverage`
- [ ] New tests added for new functionality
- [ ] Performance tests for data-intensive features
- [ ] Documentation updated if needed

## 📚 Resources

- **pytest Documentation**: https://docs.pytest.org/
- **pytest-cov**: https://pytest-cov.readthedocs.io/
- **Mock/MagicMock**: https://docs.python.org/3/library/unittest.mock.html
- **Hypothesis (Property Testing)**: https://hypothesis.readthedocs.io/

---

For questions about the testing suite, please check existing tests for examples or create an issue with specific questions. 