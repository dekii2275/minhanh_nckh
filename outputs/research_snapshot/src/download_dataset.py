"""
Module: download_dataset.py
Purpose: Verification, metadata logging, and extraction of the Penn Action dataset.
"""

import hashlib
import json
import os
import tarfile
import time
from datetime import datetime


def get_file_checksum(filepath, chunk_size=65536):
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_and_extract_dataset(archive_path="data/raw/Penn_Action.tar.gz", extract_to="data/raw"):
    """
    Verify archive integrity and extract Penn_Action.tar.gz.
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    file_size_bytes = os.path.getsize(archive_path)
    file_size_gb = file_size_bytes / (1024**3)
    print(f"Archive found: {archive_path}")
    print(f"File size: {file_size_bytes} bytes ({file_size_gb:.2f} GB)")

    print("Verifying archive integrity with tarfile...")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        print(f"Total entries in archive: {len(members)}")

        # Check sample member paths
        frame_members = [m.name for m in members if "frames" in m.name and m.name.endswith(".jpg")]
        label_members = [m.name for m in members if "labels" in m.name and m.name.endswith(".mat")]
        print(
            f"Found {len(frame_members)} jpg frame files and {len(label_members)} .mat label files."
        )

    print(f"Extracting archive to {extract_to}...")
    start_time = time.time()
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.1f}s.")

    # Record download & verification metadata
    metadata = {
        "dataset_name": "Penn Action Dataset",
        "official_source": "https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz",
        "download_timestamp": datetime.now().isoformat(),
        "archive_path": archive_path,
        "archive_size_bytes": file_size_bytes,
        "archive_size_gb": file_size_gb,
        "total_frames_found": len(frame_members),
        "total_labels_found": len(label_members),
        "extracted_path": os.path.join(extract_to, "Penn_Action"),
    }

    meta_path = "data/raw/download_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved download metadata to {meta_path}")
    return metadata


if __name__ == "__main__":
    archive = "data/raw/Penn_Action.tar.gz"
    if os.path.exists(archive):
        verify_and_extract_dataset(archive, "data/raw")
    else:
        print(f"Archive {archive} does not exist yet. Please wait for download to finish.")
