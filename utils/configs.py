from pathlib import Path

DATASETS_PATH = Path("datasets")
CANARY_MODEL_ID = "nvidia/canary-1b-v2"
CANARY_FLASH_MODEL_ID = "nvidia/canary-1b-flash"
MODELS_PATH = Path("models")
MANIFESTS_PATH = Path("manifests")
QUECHUA_PATH = DATASETS_PATH / Path("quechua")
# Hugginface ID
MAPUCHE_ID = "mengct00/Mapudungun_iwslt26"
MAPUCHE_DATASET_PATH = DATASETS_PATH / Path("mapuche")
NAHUATL_PATH = DATASETS_PATH / Path("nahuatl")
# Audios with translation are Botanica only
NAHUATL_AUDIOS_PATH = NAHUATL_PATH / Path("Sound-files-Puebla-Nahuatl")
NAHUATL_TRANSLATIONS_PATH = (
    NAHUATL_PATH
    / Path("Puebla-Nahuatl-Manifest")
    / Path("ELAN-files-Final-proofed-and-most-translated")
)
NAHUATL_TRANSCRIPTIONS_PATH = (
    NAHUATL_PATH / Path("Puebla-Nahuatl-Manifest") / Path("ELAN-files-First-draft-only")
)
SPLITS_RATIOS = {"train": 0.8, "test": 0.2, "validation": 0.1}
