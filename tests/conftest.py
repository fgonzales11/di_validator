import pytest

from di_validator import store


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DI_WORKSPACE", str(tmp_path / "workspace"))
    store.initialize()
    return tmp_path
