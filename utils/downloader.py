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
    Downloads a file from a URL to a destination path with streaming, retries, and RESUME support.
    """
    # Get existing size for resume
    existing_size = dest_path.stat().st_size if dest_path.exists() else 0

    try:
        # 1. Initial request to check server support and total size
        head = requests.head(url, timeout=30)
        total_size = int(head.headers.get("content-length", 0))

        if total_size == 0:
            # Fallback if HEAD fails to provide content-length
            with requests.get(url, stream=True, timeout=30) as r:
                total_size = int(r.headers.get("content-length", 0))

        # 2. Determine if we can/should resume
        # If file exists and is exactly the total size, skip it
        if existing_size == total_size and total_size > 0:
            progress.update(
                task_id,
                advance=0,
                description=f"[green]Skipping {dest_path.name} (complete)[/green]",
            )
            return

        # 3. Perform the download (with potential range)
        headers = {}
        if 0 < existing_size < total_size:
            headers["Range"] = f"bytes={existing_size}-"

        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            # If we asked for a range but got 200 instead of 206, the server doesn't support resume
            if r.status_code == 200 and existing_size > 0:
                # Restart from 0
                mode = "wb"
                current_pos = 0
            elif r.status_code == 206:
                # Success: Server is sending the remaining part
                mode = "ab"
                current_pos = existing_size
            else:
                r.raise_for_status()
                mode = "wb"
                current_pos = 0

            if total_size > 0:
                progress.update(task_id, total=total_size, completed=current_pos)

            with open(dest_path, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

    except requests.exceptions.RequestException as e:
        # We NO LONGER unlink on failure because we WANT to resume from where we stopped
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
