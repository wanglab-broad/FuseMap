#!/usr/bin/env python
"""
Download missing data files for FuseMap from Google Drive.

This script downloads:
1. molCCF/ folder (pretrained model weights)
2. ad_cell.h5ad (atlas reference data)

Usage:
    uv run python scripts/download_data.py
"""

import os
import sys
from pathlib import Path

def download_data():
    """Download missing data files from Google Drive."""
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown is not installed. Please run: uv sync")
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print(f"Project root: {project_root}")
    print()

    # Google Drive file/folder IDs
    MOLCCF_FOLDER_ID = "1auybpmekWuW_G-7YPloJr-B96qiT1nFS"
    AD_CELL_FILE_ID = "15LIkQTridS_ATwDy6dejIdzbMm39sEv3"

    # Target paths
    molccf_path = project_root / "molCCF"
    atlas_data_dir = project_root / "agent_setup" / "atlas_data"
    ad_cell_path = atlas_data_dir / "ad_cell.h5ad"

    # Create directories if they don't exist
    atlas_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FuseMap Data Download")
    print("=" * 70)
    print()

    # Download molCCF folder
    required_molccf_files = [
        molccf_path / "trained_model" / "FuseMap_final_model_final.pt",
        molccf_path / "ad_embed_single_cell.h5ad",
        molccf_path / "ad_embed_spatial.h5ad",
    ]

    if molccf_path.exists() and all(f.exists() for f in required_molccf_files):
        print(f"✓ molCCF/ folder is complete at {molccf_path}")
    else:
        print(f"📥 Downloading molCCF/ folder...")
        print(f"   Target: {molccf_path}")
        try:
            gdown.download_folder(
                id=MOLCCF_FOLDER_ID,
                output=str(molccf_path),
                quiet=False,
                use_cookies=False
            )
            print(f"✓ molCCF/ folder downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download molCCF/: {e}")
            print(f"  You can manually download from:")
            print(f"  https://drive.google.com/drive/u/2/folders/{MOLCCF_FOLDER_ID}")
            return False

    print()

    # Download ad_cell.h5ad
    if ad_cell_path.exists():
        print(f"✓ ad_cell.h5ad already exists at {ad_cell_path}")
    else:
        print(f"📥 Downloading ad_cell.h5ad...")
        print(f"   Target: {ad_cell_path}")
        try:
            gdown.download(
                id=AD_CELL_FILE_ID,
                output=str(ad_cell_path),
                quiet=False,
                use_cookies=False
            )
            print(f"✓ ad_cell.h5ad downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download ad_cell.h5ad: {e}")
            print(f"  You can manually download from:")
            print(f"  https://drive.google.com/file/d/{AD_CELL_FILE_ID}/view")
            return False

    print()
    print("=" * 70)
    print("✓ All data files are ready!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = download_data()
    sys.exit(0 if success else 1)
