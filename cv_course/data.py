import os
import requests
import zipfile
import tarfile
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def get_data_path(name: str) -> Path:
    return DATA_ROOT / name


def download_dataset(
    name: str,
    url: str,
    extract: bool = True,
    archive_type: str = "zip",
) -> Path:
    dest = get_data_path(name)
    os.makedirs(DATA_ROOT, exist_ok=True)

    archive_path = DATA_ROOT / f"{name}.{archive_type}"
    if archive_path.exists():
        print(f"[{name}] archive already cached.")
    else:
        print(f"[{name}] downloading from {url} ...")
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[{name}] downloaded ({archive_path.stat().st_size / 1e6:.1f} MB)")

    if extract and not dest.exists():
        print(f"[{name}] extracting ...")
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(DATA_ROOT)
        elif archive_type in ("tar", "gz"):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(DATA_ROOT)
        print(f"[{name}] extracted to {dest}")
    elif not extract:
        return archive_path

    return dest
