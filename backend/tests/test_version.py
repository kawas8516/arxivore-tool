"""Keeps the project's version consistent across every place it is stated.

`FastAPI(version=...)` sat hardcoded at "0.1.0" through four releases, so
/openapi.json and /docs told every reader the project predated almost everything
in the repo. Nobody noticed because nothing checked.

The fix is one constant (`app.__version__`) plus these tests, so the number can
only be wrong in one place at a time — and the suite says so immediately.

Releasing: bump `app.__version__`, add the matching `# ... — vX.Y.Z` heading to
RELEASE.md, then tag. These tests fail until the first two agree.
"""

import re
import subprocess
from pathlib import Path

import pytest

from app import __version__
from app.main import app

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _release_versions() -> list[str]:
    """Every version RELEASE.md declares, in document order (newest first)."""
    text = (_REPO_ROOT / "RELEASE.md").read_text(encoding="utf-8")
    return re.findall(r"^#+ .*?v(\d+\.\d+\.\d+)\s*$", text, re.M)


def test_version_is_valid_semver():
    assert _SEMVER.match(__version__), f"{__version__!r} is not MAJOR.MINOR.PATCH"


def test_openapi_advertises_the_real_version():
    """The regression this file exists for: /docs must not lie about the version."""
    assert app.version == __version__


def test_release_notes_document_the_current_version():
    """A release without notes is a release nobody can read."""
    versions = _release_versions()
    assert versions, "no version headings found in RELEASE.md"
    newest = max(versions, key=lambda v: tuple(int(p) for p in v.split(".")))
    assert __version__ in versions, (
        f"app.__version__ is {__version__} but RELEASE.md has no heading for it. "
        f"Newest documented: {newest}. Add the section, or fix the constant."
    )


def test_current_version_is_the_newest_in_release_notes():
    """Guards the other direction: notes written for a version never shipped.

    Uses max() rather than the first entry, so this stays correct regardless of
    how the file is ordered.
    """
    versions = _release_versions()
    newest = max(versions, key=lambda v: tuple(int(p) for p in v.split(".")))
    assert __version__ == newest, (
        f"app.__version__ is {__version__} but RELEASE.md's newest section is "
        f"{newest} — one of the two was updated without the other."
    )


def test_release_notes_are_newest_first():
    """Newest release at the top, descending — what a reader wants first.

    Guards the ordering itself, so a section appended to the bottom out of habit
    fails here instead of quietly burying the latest release under the history.
    """
    versions = [tuple(int(p) for p in v.split(".")) for v in _release_versions()]
    assert versions == sorted(versions, reverse=True), (
        "RELEASE.md sections are out of order (expected newest first): "
        + " -> ".join(".".join(map(str, v)) for v in versions)
    )


@pytest.mark.live_models
def test_git_tag_matches_the_declared_version():
    """Once tagged, the tag and the constant must agree.

    Marked live_models only because it shells out to git and needs tags present —
    a shallow CI clone has none, and that should not fail the default suite.
    """
    result = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    tags = [t.strip().lstrip("v") for t in result.stdout.split() if t.strip()]
    if not tags:
        pytest.skip("HEAD is not tagged — nothing to compare")
    assert __version__ in tags, (
        f"HEAD is tagged {tags} but app.__version__ is {__version__}"
    )
