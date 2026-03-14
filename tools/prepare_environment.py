#!/usr/bin/env python3
"""Minimal environment check for Kaggle runtime."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch


def main() -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "workspace": str(Path.cwd()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
