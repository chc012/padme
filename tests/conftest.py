"""Anchor the suite to the repo root.

Nine tests read repo-relative paths (`docs/REPRODUCE.md`, `tests/fixtures/...`, `config/...`),
and one does it at *module import* time, so collection itself -- not just assertions -- depends
on the working directory. Without this, running `pytest` from anywhere but the repo root
produced 12 failures that look like real breakage and are not.

`pyproject.toml`'s `pythonpath = ["."]` already assumes the rootdir is this directory, so this
makes that assumption uniform rather than introducing a new one.
"""

import os
import pathlib

os.chdir(pathlib.Path(__file__).resolve().parents[1])
