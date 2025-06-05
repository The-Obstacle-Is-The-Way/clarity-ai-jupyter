# Clarity EEG Analysis Test Suite

This directory contains the test suite for the Clarity EEG Analysis project. The tests are organized into unit tests and integration tests following best practices for Python testing.

## Test Structure

- **Unit Tests**: Located in `tests/unit/`, these test individual components in isolation
  - `data/` - Tests for data loading and preprocessing functions
  - `features/` - Tests for feature extraction methods
  - `models/` - Tests for neural network model architectures
  - `training/` - Tests for dataset handling and training functions

- **Integration Tests**: Located in `tests/integration/`, these test the interaction between components
  - `test_pipeline.py` - End-to-end tests of the data processing and model training pipeline

## Running Tests

### Basic Usage

To run all tests:

```bash
pytest
```

To run unit tests only:

```bash
pytest tests/unit
```

To run specific test files:

```bash
pytest tests/unit/models/test_models.py
```

### Running with Coverage

For test coverage reporting:

```bash
# Run tests with coverage report
./run_tests.sh

# Or manually:
pytest --cov=src/clarity --cov-report=term-missing tests/
```

The HTML coverage report will be available in `./coverage_report/`.

### Test Configuration

The test configuration is defined in `pytest.ini` in the project root.

## Fixtures

Test fixtures are defined in `tests/conftest.py` and include:

- `seed`: Reproducible random seed
- `sample_eeg_data`: Synthetic EEG data for testing
- `sample_epochs`: Segmented EEG epochs for testing
- `subject_labels`: Sample subject ID to label mapping

## Guidelines for Writing Tests

1. **Test Naming**: Follow the pattern `test_*` for test functions and files
2. **Isolation**: Unit tests should not rely on external data or services
3. **Fixtures**: Use fixtures to share setup code between tests
4. **Mocking**: Use `pytest-mock` for external dependencies
5. **Coverage**: Aim for high test coverage of core functionality
6. **Performance**: Keep tests fast - mark slow tests with `@pytest.mark.slow`
7. **Documentation**: Each test should have a docstring explaining what it's testing

## Continuous Integration

These tests are designed to run in CI environments. The GitHub Actions workflow will:

1. Run all tests with coverage reporting
2. Run static type checking with pyright
3. Run linting with ruff