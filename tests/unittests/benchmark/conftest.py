import shutil
import tempfile

import pytest


@pytest.fixture(scope="function")
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)
