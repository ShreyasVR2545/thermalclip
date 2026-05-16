"""
FLIR ADAS v2 Dataset Downloader
=================================
Downloads and prepares the Teledyne FLIR ADAS v2 thermal dataset from Kaggle.

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Set KAGGLE_API_TOKEN env var, or save token to ~/.kaggle/access_token,
       or place legacy kaggle.json in ~/.kaggle/
    3. Accept the dataset's terms on Kaggle first

The dataset is ~11.8 GB compressed, ~14 GB extracted.
"""

import os
import sys
import json
import shutil
from pathlib import Path


def check_kaggle_credentials():
    """
    Verify Kaggle API credentials exist.

    Supports three auth methods (checked in order):
        1. KAGGLE_API_TOKEN env var (new-style token, e.g. KGAT_...)
        2. ~/.kaggle/access_token file (new-style, saved to disk)
        3. ~/.kaggle/kaggle.json (legacy username+key JSON)
    """
    kaggle_dir = Path.home() / ".kaggle"

    # Method 1: environment variable (new Kaggle API tokens)
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("[✓] Kaggle auth: KAGGLE_API_TOKEN environment variable")
        return

    # Method 2: access_token file (new Kaggle API tokens saved to disk)
    access_token_file = kaggle_dir / "access_token"
    if access_token_file.exists():
        print(f"[✓] Kaggle auth: {access_token_file}")
        return

    # Method 3: legacy kaggle.json
    cred_file = kaggle_dir / "kaggle.json"
    if cred_file.exists():
        try:
            os.chmod(str(cred_file), 0o600)
        except OSError:
            pass  # Windows doesn't support chmod the same way
        print(f"[✓] Kaggle auth: {cred_file}")
        return

    # None found
    print("ERROR: Kaggle credentials not found.")
    print("Set up using ONE of these methods:\n")
    print("  Option A — Environment variable (recommended):")
    print("    Set KAGGLE_API_TOKEN to your token from kaggle.com → Settings → API Tokens")
    print("    PowerShell:  $env:KAGGLE_API_TOKEN = \"KGAT_your_token_here\"")
    print("    Linux/Mac:   export KAGGLE_API_TOKEN=KGAT_your_token_here\n")
    print("  Option B — Save token to file:")
    print(f"    Save your token string to {access_token_file}\n")
    print("  Option C — Legacy kaggle.json:")
    print(f"    Place kaggle.json at {cred_file}")
    sys.exit(1)


def download_dataset(data_dir: str = "data/flir_adas_v2",
                     kaggle_dataset: str = "samdazel/teledyne-flir-adas-thermal-dataset-v2"):
    """Download and extract FLIR ADAS v2 from Kaggle."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    marker = data_dir / ".download_complete"
    if marker.exists():
        print(f"[✓] Dataset already downloaded at {data_dir}")
        return

    check_kaggle_credentials()

    print(f"[↓] Downloading {kaggle_dataset}...")
    print("    This is ~11.8 GB — may take a while on slow connections.")

    try:
        import kaggle
        kaggle.api.dataset_download_files(
            kaggle_dataset,
            path=str(data_dir),
            unzip=True,
        )
    except Exception as e:
        # Fallback to CLI
        print(f"  Python API failed ({e}), trying CLI...")
        ret = os.system(
            f"kaggle datasets download -d {kaggle_dataset} "
            f"-p {data_dir} --unzip"
        )
        if ret != 0:
            print("ERROR: kaggle CLI failed. Install with: pip install kaggle")
            sys.exit(1)

    marker.touch()
    print(f"[✓] Dataset downloaded and extracted to {data_dir}")


def validate_pairs(data_dir: str = "data/flir_adas_v2"):
    """
    Validate thermal–RGB pairing and report dataset statistics.

    Checks:
        1. rgb_to_thermal_vid_map.json exists and is valid
        2. All referenced image files exist on disk
        3. Reports number of valid pairs per split
    """
    data_dir = Path(data_dir)

    print("\n[Validate] Checking dataset structure...")

    # Check mapping file
    map_file = data_dir / "rgb_to_thermal_vid_map.json"
    if map_file.exists():
        with open(map_file) as f:
            mapping = json.load(f)
        print(f"  [✓] Mapping file: {len(mapping)} entries")
    else:
        print("  [!] rgb_to_thermal_vid_map.json not found — will use filename matching")
        mapping = None

    # Check each split
    for split in ["train", "val"]:
        rgb_dirs = [
            data_dir / f"images_rgb_{split}" / "data",
            data_dir / f"images_rgb_{split}",
        ]
        thermal_dirs = [
            data_dir / f"images_thermal_{split}" / "data",
            data_dir / f"images_thermal_{split}",
        ]

        rgb_dir = next((d for d in rgb_dirs if d.exists()), None)
        thermal_dir = next((d for d in thermal_dirs if d.exists()), None)

        if rgb_dir is None:
            print(f"  [✗] {split}: RGB directory not found")
            continue
        if thermal_dir is None:
            print(f"  [✗] {split}: Thermal directory not found")
            continue

        rgb_files = list(rgb_dir.glob("*.[jp][pn][g]"))
        thermal_files = list(thermal_dir.glob("*.[jp][pn][g]"))

        # Count valid pairs
        if mapping:
            valid = sum(
                1 for r, t in mapping.items()
                if (rgb_dir / r).exists() and (thermal_dir / t).exists()
            )
        else:
            rgb_stems = {p.stem for p in rgb_files}
            thermal_stems = {p.stem for p in thermal_files}
            valid = len(rgb_stems & thermal_stems)

        print(f"  [✓] {split}: {len(rgb_files)} RGB, {len(thermal_files)} thermal, "
              f"{valid} valid pairs")

    # Check annotations
    for split in ["train", "val"]:
        ann_candidates = [
            data_dir / f"images_thermal_{split}" / "coco.json",
            data_dir / f"images_thermal_{split}" / "data" / "coco.json",
        ]
        ann_file = next((f for f in ann_candidates if f.exists()), None)
        if ann_file:
            with open(ann_file) as f:
                coco = json.load(f)
            n_cats = len(coco.get("categories", []))
            n_anns = len(coco.get("annotations", []))
            cat_names = [c["name"] for c in coco.get("categories", [])]
            print(f"  [✓] {split} annotations: {n_anns} boxes across {n_cats} classes")
            print(f"      Classes: {', '.join(cat_names[:10])}{'...' if n_cats > 10 else ''}")
        else:
            print(f"  [!] {split}: No COCO annotation file found")

    print("\n[Validate] Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download FLIR ADAS v2 dataset")
    parser.add_argument("--data_dir", type=str, default="data/flir_adas_v2",
                        help="Directory to download data to")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate existing download, don't download")
    args = parser.parse_args()

    if not args.validate_only:
        download_dataset(args.data_dir)

    validate_pairs(args.data_dir)
