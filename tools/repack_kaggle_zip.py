#!/usr/bin/env python3
"""Build a single Kaggle bundle zip from canonical source folders."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def _iter_files(folder: Path, output_zip_name: str):
    for path in folder.rglob("*"):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix == ".pyc":
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue
        if path.name == output_zip_name:
            continue
        if path.suffix.lower() == ".zip":
            continue
        yield path


def _validate_archive_layout(output_zip: Path, bundle_name: str) -> None:
    with zipfile.ZipFile(output_zip, mode="r") as archive:
        names = archive.namelist()

    duplicate_prefix = f"{bundle_name}/{bundle_name}/"
    duplicated = [name for name in names if name.startswith(duplicate_prefix)]
    if duplicated:
        raise RuntimeError(
            "Invalid archive layout: duplicated top-level bundle folder detected: "
            f"{duplicated[0]}"
        )

    invalid_roots = [name for name in names if not name.startswith(f"{bundle_name}/")]
    if invalid_roots:
        raise RuntimeError(
            "Invalid archive layout: entries escaped the expected top-level bundle folder: "
            f"{invalid_roots[0]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack Kaggle code zip from project root")
    parser.add_argument("--bundle_name", type=str, default="kaggle_bundle", help="Top-level folder name inside zip")
    parser.add_argument(
        "--zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate tools/<bundle_name>.zip",
    )
    args = parser.parse_args()

    if not args.zip:
        print("--no-zip selected; nothing to build.")
        return

    root = Path(__file__).resolve().parents[1]
    tools_dir = Path(__file__).resolve().parent
    output_zip = tools_dir / f"{args.bundle_name}.zip"

    source_dirs = [
        root / "src",
        root / "tools",
        root / "cli",
        root / "configs",
    ]
    source_files = [
        root / "notebooks" / "Kaggle_Full_ASR_Run.ipynb",
        root / "requirements.txt",
    ]

    for source in source_dirs:
        if not source.exists():
            raise FileNotFoundError(f"Missing required source folder: {source}")
    for source in source_files:
        if not source.exists():
            raise FileNotFoundError(f"Missing required source file: {source}")

    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in source_dirs:
            base_name = source.name
            for file_path in _iter_files(source, output_zip.name):
                relative = file_path.relative_to(source)
                arcname = Path(args.bundle_name) / base_name / relative
                archive.write(file_path, arcname.as_posix())

        for source in source_files:
            if source.parent.name == "notebooks":
                arcname = Path(args.bundle_name) / "notebooks" / source.name
            else:
                arcname = Path(args.bundle_name) / source.name
            archive.write(source, arcname.as_posix())

    _validate_archive_layout(output_zip, args.bundle_name)

    print(f"Built zip at: {output_zip}")


if __name__ == "__main__":
    main()
