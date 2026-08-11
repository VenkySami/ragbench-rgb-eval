#!/usr/bin/env python3
"""Downloads the 3 required English RGB data files (master branch, since `main`
does not exist on the upstream repo)."""
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/chen700564/RGB/master/data"
FILES = ["en_refine.json", "en_int.json", "en_fact.json"]


def main() -> int:
    for fname in FILES:
        url = f"{BASE_URL}/{fname}"
        dest = DATA_DIR / fname
        print(f"Downloading {url} -> {dest}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  {len(resp.content):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
