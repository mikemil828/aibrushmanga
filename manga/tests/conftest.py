import pytest

from manga import settings


@pytest.fixture(autouse=True)
def shipped_settings():
    """Every test runs on the shipped defaults, whatever settings.local.toml or MANGA_SETTINGS hold."""
    settings.load(argv=False, user=False)
    yield
    settings.load(argv=False, user=False)
