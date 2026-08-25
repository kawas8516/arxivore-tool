from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app import budget


@pytest.fixture(autouse=True)
def _clean_ledger():
    budget.reset()
    yield
    budget.reset()


def test_add_accumulates_both_directions():
    budget.add(100, 50)
    budget.add(10, 5)
    assert budget.spent() == 165


def test_negative_values_are_ignored():
    """A provider reporting nonsense must not credit the ledger."""
    budget.add(-500, -500)
    assert budget.spent() == 0


def test_remaining_tracks_configured_ceiling():
    ceiling = budget.get_settings().daily_token_budget
    budget.add(1000, 0)
    assert budget.remaining() == ceiling - 1000


def test_not_exhausted_below_ceiling():
    budget.add(1, 1)
    assert budget.exhausted() is False


def test_exhausted_at_ceiling():
    budget.add(budget.get_settings().daily_token_budget, 0)
    assert budget.exhausted() is True
    assert budget.remaining() == 0


def test_ledger_rolls_over_on_new_utc_day():
    budget.add(999, 1)
    assert budget.spent() == 1000

    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)

    class _FakeDatetime:
        @staticmethod
        def now(tz=None):
            return tomorrow

    with patch("app.budget.datetime", _FakeDatetime):
        assert budget.spent() == 0


def test_rollover_does_not_reset_within_same_day():
    budget.add(500, 0)
    assert budget.spent() == 500
    assert budget.spent() == 500
