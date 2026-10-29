"""
Download and unpack the J-HMDB dataset (joint positions, splits, and videos).
Official Source: Max Planck Institute for Intelligent Systems (https://files.is.tue.mpg.de/jhmdb/)
"""

import os
import urllib.request
import zipfile

from tqdm import tqdm

DATA_DIR = os.path.abspath("data/raw/jhmdb")
os.makedirs(DATA_DIR, exist_ok=True)

URLS = {
    "joint_positions.zip": "https://files.is.tue.mpg.de/jhmdb/joint_positions.zip",
    "splits.zip": "https://files.is.tue.mpg.de/jhmdb/splits.zip",
    "JHMDB_video.zip": "https://files.is.tue.mpg.de/jhmdb/JHMDB_video.zip",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract():
    for fname, url in URLS.items():
        zip_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(zip_path):
            print(f"Downloading {fname} from {url}...")
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=fname) as t:
                urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
            print(f"Downloaded {fname} successfully.")
        else:
            print(f"{fname} already exists at {zip_path}.")

        # Extract
        target_extract = os.path.join(DATA_DIR, fname.replace(".zip", ""))
        if not os.path.exists(target_extract):
            print(f"Extracting {fname} to {target_extract}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print(f"Extracted {fname} successfully.")
        else:
            print(f"Directory {target_extract} already extracted.")


if __name__ == "__main__":
    download_and_extract()
