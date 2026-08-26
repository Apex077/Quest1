"""Shared fixtures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import app as flask_app_module


@pytest.fixture(autouse=True)
def clean_tasks():
    """Wipe module-level task state between tests."""
    flask_app_module._tasks.clear()
    yield
    flask_app_module._tasks.clear()


@pytest.fixture
def client():
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as c:
        yield c
