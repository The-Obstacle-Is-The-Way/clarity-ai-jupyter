#!/bin/bash
# Run tests with coverage reporting

# Exit on error
set -e

# Source virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install test dependencies if not already installed
pip install -r requirements-dev.txt

# Run tests with coverage
pytest tests/unit -v --cov=src/clarity

# Generate HTML coverage report
python -m pytest --cov=src/clarity --cov-report=html:coverage_report tests/

echo "✅ Tests completed - HTML coverage report available in ./coverage_report/"