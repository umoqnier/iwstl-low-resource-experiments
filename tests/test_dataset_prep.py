from pathlib import Path

import pytest
from rich.progress import Progress

from utils.downloader import download_dataset
from utils.extraction import extract_dataset

# Replace these with your actual Nextcloud public links
SIMPLE_REGISTRY = {
    "test_lang": {
        "urls": [
            (
                "https://nextcloud.tepezil.net/public.php/dav/files/2nz9zQWqmJx63x3/?accept=zip",
                "test_simple.tgz",
            ),
        ],
        "patterned_urls": [],
        "dest_dir": Path("test_data"),
    },
}

MULTI_REGISTRY = {
    "test_lang": {
        "urls": [
            (
                "https://nextcloud.tepezil.net/public.php/dav/files/TMreCYX9xz65NKj/?accept=zip",
                "test_multi.tgz.part00",
            ),
            (
                "https://nextcloud.tepezil.net/public.php/dav/files/f3KMcgSPmn6XCqm/?accept=zip",
                "test_multi.tgz.part01",
            ),
        ],
        "patterned_urls": [],
        "dest_dir": Path("test_data"),
    },
}


def test_simple_download_extract(tmp_path):
    """Test downloading and extracting a simple .tgz archive."""
    test_dest_dir = tmp_path / "simple_datasets"
    test_dest_dir.mkdir()
    local_registry = {
        "test_lang": {
            **SIMPLE_REGISTRY["test_lang"],
            "dest_dir": test_dest_dir,
        }
    }

    progress = Progress()

    # 1. Download
    download_dataset("test_lang", local_registry, progress)
    assert (test_dest_dir / "test_simple.tgz").exists()

    # 2. Extract
    extract_dataset("test_lang", local_registry, progress)

    # Debug: list files if assertion fails
    if not (test_dest_dir / "movilidad_urbana.md").exists():
        print(
            f"\nFiles found in {test_dest_dir}: {[p.name for p in test_dest_dir.rglob('*')]}"
        )

    assert (test_dest_dir / "movilidad_urbana.md").exists(), "Simple extraction failed"


def test_multi_download_extract(tmp_path):
    """Test downloading and extracting a multipart .tgz archive."""
    test_dest_dir = tmp_path
    local_registry = {
        "test_lang": {
            **MULTI_REGISTRY["test_lang"],
            "dest_dir": test_dest_dir,
        }
    }

    progress = Progress()

    # 1. Download
    download_dataset("test_lang", local_registry, progress)
    assert (test_dest_dir / "test_multi.tgz.part00").exists()
    assert (test_dest_dir / "test_multi.tgz.part01").exists()

    # 2. Extract
    extract_dataset("test_lang", local_registry, progress)
    assert (test_dest_dir / "movilidad_urbana.md").exists(), (
        "Multipart extraction failed"
    )
