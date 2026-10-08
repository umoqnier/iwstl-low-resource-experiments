import tarfile
from pathlib import Path

from rich.progress import Progress


def extract_tgz(file_path: Path, dest_dir: Path, progress: Progress, task_id: int):
    """
    Extracts a .tgz file to the destination directory.
    """
    try:
        progress.update(task_id, description=f"Extracting {file_path.name}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=dest_dir, filter="data")
        progress.update(
            task_id, description=f"[green]Extracted {file_path.name}[/green]"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract {file_path}: {e}")


def extract_dataset(lang_code: str, registry: dict, progress: Progress):
    """
    Handles extraction of all datasets for a given language, including multipart archives.
    """
    if lang_code not in registry:
        return

    config = registry[lang_code]
    dest_dir = Path(config["dest_dir"])

    # 1. Find all files in the destination directory
    all_files = list(dest_dir.glob("*"))

    # 2. Identify multipart archives by looking for .part00, .part01, etc.
    # We group files that share the same base name before the .partXX suffix
    multipart_groups = {}
    for file in all_files:
        if ".tgz.part" in file.name:
            base_name = file.name.split(".tgz.part")[0] + ".tgz"
            multipart_groups.setdefault(base_name, []).append(file)

    # 3. Concatenate and extract multipart archives
    for base_filename, parts in multipart_groups.items():
        combined_tgz = dest_dir / base_filename
        task_id = progress.add_task(
            description=f"Processing multipart {base_filename}...", total=None
        )
        try:
            progress.update(
                task_id, description=f"Concatenating parts for {base_filename}..."
            )
            # Sort parts to ensure they are joined in the correct order (.part00, .part01...)
            parts.sort()

            with open(combined_tgz, "wb") as outfile:
                for part_path in parts:
                    with open(part_path, "rb") as infile:
                        while True:
                            chunk = infile.read(1024 * 1024)
                            if not chunk:
                                break
                            outfile.write(chunk)

            extract_tgz(combined_tgz, dest_dir, progress, task_id)
            combined_tgz.unlink()
        finally:
            progress.remove_task(task_id)

    # 4. Extract any other standalone archives (that aren't parts)
    for file in all_files:
        # Check for .tgz or .tar.gz extension
        if (
            file.name.endswith(".tgz") or file.name.endswith(".tar.gz")
        ) and ".part" not in file.name:
            task_id = progress.add_task(
                description=f"Extracting {file.name}...", total=None
            )
            try:
                extract_tgz(file, dest_dir, progress, task_id)
            finally:
                progress.remove_task(task_id)
