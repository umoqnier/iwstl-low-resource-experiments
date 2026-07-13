import random
from pathlib import Path

DATASETS_PATH = Path("datasets")
CANARY_MODEL_ID = "nvidia/canary-1b-v2"
CANARY_FLASH_MODEL_ID = "nvidia/canary-1b-v2-flash"
MODELS_PATH = Path("models")
MANIFESTS_PATH = Path("manifests")
QUECHUA_PATH = DATASETS_PATH / Path("quechua")
# Hugginface ID
MAPUCHE_ID = "mengct00/Mapudungun_iwslt26"
NAHUATL_PATH = DATASETS_PATH / Path("nahuatl")


def generate_split_mapping(
    root_path: Path, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, seed=42
):
    """
    Automatically generates a split mapping by partitioning subdirectories
    of the root_path.
    """
    if not root_path.exists():
        return {"train": [], "dev": [], "test": []}

    folders = [f.name for f in root_path.iterdir() if f.is_dir()]

    random.seed(seed)
    random.shuffle(folders)

    total = len(folders)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    return {
        "train": folders[:train_end],
        "dev": folders[train_end:dev_end],
        "test": folders[dev_end:],
    }


# Generate the Nahuatl splits automatically
# We target the 'SpeechTranslation' subdirectory specifically
NAHUATL_SPLITS = generate_split_mapping(NAHUATL_PATH / "SpeechTranslation")
