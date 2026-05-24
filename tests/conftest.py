import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that download model weights and run the full pipeline",
    )
