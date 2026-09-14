"""Repo-relative path constants (single source of truth for this repo).

src/ and scripts/ each keep a byte-identical copy; both directories sit one
level under the repo root, so the same computation works from either.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO_ROOT, "data", "malecns")       # raw downloads land here
PROCESSED = os.path.join(DATA, "processed")             # adjacency caches (committed)
CKPT = os.path.join(DATA, "ckpts")                      # checkpoints (gitignored)
RESULTS = os.path.join(REPO_ROOT, "results")
SRC = os.path.join(REPO_ROOT, "src")
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
CORPUS = os.path.join(REPO_ROOT, "data_provenance", "tinyshakespeare_input.txt")
