"""Arxivore backend.

`__version__` is the single source of truth for the project's version. It is what
`FastAPI(version=...)` advertises at /openapi.json and /docs, and what
tests/test_version.py checks against the newest heading in RELEASE.md.

It sat at "0.1.0" hardcoded in main.py through four releases, so the API docs
told every reader the project predated almost everything in the repo. Bump this
constant and the RELEASE.md heading together; the test fails if they disagree.
"""

__version__ = "0.5.1"
