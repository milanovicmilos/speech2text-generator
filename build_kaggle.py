#!/usr/bin/env python3
"""Build a reproducible Kaggle package from canonical source folders."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent


def _copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".ipynb_checkpoints",
        ),
    )


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _entries(bundle_name: str) -> List[Tuple[Path, Path]]:
    bundle_root = ROOT / "dist" / bundle_name
    return [
        (ROOT / "src", bundle_root / "src"),
        (ROOT / "cli", bundle_root / "cli"),
        (ROOT / "tools", bundle_root / "tools"),
        (ROOT / "configs", bundle_root / "configs"),
        (ROOT / "kaggle", bundle_root / "kaggle"),
        (ROOT / "notebooks" / "Kaggle_Full_ASR_Run.ipynb", bundle_root / "kaggle" / "Kaggle_Full_ASR_Run.ipynb"),
        (ROOT / "requirements.txt", bundle_root / "requirements.txt"),
        (ROOT / "README.md", bundle_root / "README.md"),
    ]


def build(bundle_name: str, zip_output: bool) -> Path:
    bundle_root = ROOT / "dist" / bundle_name
    bundle_root.mkdir(parents=True, exist_ok=True)

    for src, dst in _entries(bundle_name):
        if not src.exists():
            raise FileNotFoundError(f"Missing required source path: {src}")
        if src.is_dir():
            _copy_dir(src, dst)
        else:
            _copy_file(src, dst)

    if zip_output:
        archive_base = str(bundle_root)
        shutil.make_archive(archive_base, "zip", root_dir=bundle_root.parent, base_dir=bundle_root.name)

    return bundle_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kaggle-ready package from central source folders")
    parser.add_argument("--bundle_name", type=str, default="kaggle_bundle", help="Output folder name under dist/")
    parser.add_argument(
        "--zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also generate dist/<bundle_name>.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = build(bundle_name=args.bundle_name, zip_output=args.zip)
    print(f"Built bundle at: {bundle_root}")
    if args.zip:
        print(f"Built zip at: {bundle_root}.zip")


if __name__ == "__main__":
    main()
