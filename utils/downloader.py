import os
from pathlib import Path

import requests
from rich.progress import Progress
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class DownloadError(Exception):
    """Custom exception for download failures."""

    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (requests.exceptions.RequestException, DownloadError)
    ),
    reraise=True,
)
def download_file(url: str, dest_path: Path, progress: Progress, task_id: int):
    """
    Downloads a file from a URL to a destination path with streaming and retries.
    """
    # Check if file already exists and has size > 0 to avoid redundant downloads
    if dest_path.exists() and dest_path.stat().st_size > 0:
        progress.update(
            task_id,
            advance=0,
            description=f"[green]Skipping {dest_path.name} (already exists)[/green]",
        )
        return

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))

            # Update progress bar total if known
            if total_size > 0:
                progress.update(task_id, total=total_size)

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
    except requests.exceptions.RequestException as e:
        # Clean up partially downloaded file on failure to avoid corrupted files on retry
        if dest_path.exists():
            dest_path.unlink()
        raise e


def download_dataset(lang_code: str, registry: dict, progress: Progress):
    """
    Orchestrates the download of all files for a given language based on the registry.
    """
    if lang_code not in registry:
        return

    config = registry[lang_code]
    dest_dir = Path(config["dest_dir"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    urls_to_download = []

    # Collect static URLs
    for item in config.get("urls", []):
        if isinstance(item, tuple):
            url, filename = item
        else:
            url = item
            filename = url.split("/")[-1]
        urls_to_download.append((url, filename))

    # Collect patterned URLs
    for item in config.get("patterned_urls", []):
        if len(item) == 3:
            url_pattern, filename_pattern, range_obj = item
            for i in range_obj:
                urls_to_download.append(
                    (url_pattern.format(i), filename_pattern.format(i))
                )
        elif len(item) == 2:
            url_pattern, range_obj = item
            for i in range_obj:
                url = url_pattern.format(i)
                filename = url.split("/")[-1]
                urls_to_download.append((url, filename))

    for url, filename in urls_to_download:
        dest_path = dest_dir / filename

        task_id = progress.add_task(
            description=f"Downloading {filename}...", total=None
        )
        try:
            download_file(url, dest_path, progress, task_id)
        finally:
            progress.remove_task(task_id)
