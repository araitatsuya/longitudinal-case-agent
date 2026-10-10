import json
import os
import tempfile
from pathlib import Path

import pytest

# app.main builds a module-level app on import; keep its data out of the repo.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="case-agent-test-"))
os.environ.setdefault("PLANNER", "mock")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.planner import MockPlanner  # noqa: E402

INBOX = Path(__file__).resolve().parent.parent / "inbox"


def load_event(name: str) -> dict:
    return json.loads((INBOX / name).read_text())


@pytest.fixture
def day1():
    return load_event("day1_admit.json")


@pytest.fixture
def day2():
    return load_event("day2_labs.json")


@pytest.fixture
def day3():
    return load_event("day3_culture.json")


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def client(data_dir):
    return TestClient(create_app(data_dir=data_dir, planner=MockPlanner()))
