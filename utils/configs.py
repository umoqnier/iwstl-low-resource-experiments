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
    / Path("Pueble-Nahuatl-Manifest")
    / Path("ELAN-files-Final-proofed-and-most-translated")
)
NAHUATL_TRANSCRIPTIONS_PATH = (
    NAHUATL_PATH / Path("Pueble-Nahuatl-Manifest") / Path("ELAN-files-First-draft-only")
)
SPLITS_RATIOS = {"train": 0.8, "test": 0.2, "validation": 0.1}

# Registry mapping language codes to their download resources
DATASET_DOWNLOAD_REGISTRY = {
    "azz": {
        "urls": [
            "https://openslr.trmal.net/resources/92/Puebla-Nahuatl-Manifest.tgz",
            "https://openslr.trmal.net/resources/92/SpeechTranslation_Nahuatl_Manifest.tgz",
        ],
        "patterned_urls": [
            (
                "https://openslr.trmal.net/resources/92/Sound-Files-Puebla-Nahuatl.tgz.part{:02d}",
                range(10),
            ),
        ],
        "dest_dir": NAHUATL_PATH,
    },
}
