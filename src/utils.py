"""
Utility functions for the project, common I/O helpers
"""

import json
import pickle
from __future__ import annotations

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f) # noqa: S301
    
def dump_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
