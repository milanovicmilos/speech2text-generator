#!/usr/bin/env python3
"""Package Kaggle run artifacts into a single zip for upload/download."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip artifacts directory")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory to archive")
    parser.add_argument("--zip_prefix", type=str, default="asr_artifacts", help="Output zip name prefix")
    args = parser.parse_args()

    source = Path(args.source_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    output_base = source.parent / args.zip_prefix
    zip_path = shutil.make_archive(
        str(output_base),
        "zip",
        root_dir=str(source.parent),
        base_dir=source.name,
    )
    print(f"[DONE] Created archive: {zip_path}")


if __name__ == "__main__":
    main()
