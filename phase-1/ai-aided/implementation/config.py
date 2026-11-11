"""Model, data paths, and analysis groupings for SigLIP 2 on XM3600."""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
XM3600_DIR = ROOT / "xm3600"                  # captions.jsonl + the 3600 <key>.jpg files
CAPTIONS_JSONL = XM3600_DIR / "captions.jsonl"
IMAGES_DIR = XM3600_DIR                       # images sit directly next to captions.jsonl
# everything the pipeline writes; on Colab set VIZ4AI_PROCESSED to a Google Drive folder
PROCESSED_DIR = Path(os.environ.get("VIZ4AI_PROCESSED", ROOT / "processed"))

# Precomputed files (written by the pipeline, read by the app)
CAPTIONS_TABLE = PROCESSED_DIR / "captions.parquet"        # one row per caption
IMAGES_TABLE = PROCESSED_DIR / "images.parquet"            # one row per image (+ resolution)
IMAGE_EMB = PROCESSED_DIR / "image_embeddings.npy"         # (n_images, dim), row order = images.parquet
CAPTION_EMB = PROCESSED_DIR / "caption_embeddings.npy"     # (n_captions, dim), row order = captions.parquet
CAPTION_SCORES = PROCESSED_DIR / "caption_scores.parquet"  # captions + cosine + margin
IMAGE_LANG_TABLE = PROCESSED_DIR / "image_language.parquet"  # mean per (image, caption language)
MATRICES_FILE = PROCESSED_DIR / "matrices.parquet"         # long format: state, caption_lang, locale, value
EMBEDDING_METADATA = PROCESSED_DIR / "embedding_metadata.json"
SCORE_METADATA = PROCESSED_DIR / "score_metadata.json"

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
MODEL_NAME = "google/siglip2-base-patch16-256"
MODEL_REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
EMBEDDING_DIM = 768
IMAGE_SIZE = 256              # fixed-resolution checkpoint; processor performs resizing
TEXT_MAX_LENGTH = 64          # lowercase, pad and truncate using the pinned processor
EMBED_BATCH_SIZE_TEXT = 128
EMBED_BATCH_SIZE_IMAGE = 16
# Captions per (image, caption language). None = all (~2 on average, ~261k captions).
# 1 = only the first caption. Changing this requires regenerating the input tables
# and embeddings with the same setting.
MAX_CAPTIONS_PER_LANG = None

# --------------------------------------------------------------------------
# Languages
# --------------------------------------------------------------------------
# The 36 XM3600 languages (code -> English name). Image locales use the same codes.
LANGUAGES = {
    "ar": "Arabic", "bn": "Bengali", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian", "fi": "Finnish",
    "fil": "Filipino", "fr": "French", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "mi": "Maori", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "quz": "Cusco Quechua", "ro": "Romanian", "ru": "Russian", "sv": "Swedish", "sw": "Swahili",
    "te": "Telugu", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese",
    "zh": "Chinese",
}

# Optional language exclusions apply to both heatmap axes.
UNSUPPORTED_LANGUAGES: list[str] = []

# Dropped locales' images are also removed from the distractor set of the margin,
# so the analysis set is consistent. Set False to keep them as distractors only.
DROP_IMAGES_FROM_DISTRACTORS = True

# Known caveats about where a locale's images come from (Thapliyal et al., 2022).
LOCALE_NOTES = {
    "hi": "Hindi-locale images were selected at the global level (in-region images went to Bengali and Telugu).",
    "fa": "14 of the Persian-locale images are continent-level, not from Iran.",
}

# --------------------------------------------------------------------------
# Grouping 1: language family (groups kept separate, not merged)
# Order of the dict = order of the groups on the axes; alphabetical by name within a group.
# --------------------------------------------------------------------------
LANGUAGE_FAMILIES = {
    "Germanic": ["da", "de", "en", "nl", "no", "sv"],
    "Romance": ["es", "fr", "it", "pt", "ro"],
    "Slavic": ["cs", "hr", "pl", "ru", "uk"],
    "Hellenic": ["el"],
    "Indo-Iranian": ["bn", "fa", "hi"],
    "Uralic": ["fi", "hu"],
    "Turkic": ["tr"],
    "Semitic": ["ar", "he"],
    "Austronesian": ["fil", "id", "mi"],
    "Sino-Tibetan": ["zh"],
    "Japonic": ["ja"],
    "Koreanic": ["ko"],
    "Kra-Dai": ["th"],
    "Austroasiatic": ["vi"],
    "Dravidian": ["te"],
    "Bantu": ["sw"],
    "Quechuan": ["quz"],
}

# --------------------------------------------------------------------------
# Grouping 2: Inglehart-Welzel cultural map zones (World Values Survey)
# Exploratory locale-level assignments, not per-image cultural annotations.
# Geographic ambiguity and overlap with language families limit interpretation.
# --------------------------------------------------------------------------
CULTURAL_ZONES = {
    "Protestant Europe": ["da", "de", "fi", "nl", "no", "sv"],
    "English-speaking": ["en", "mi"],
    "Catholic Europe": ["cs", "es", "fr", "hr", "hu", "it", "pl", "pt"],
    "Orthodox Europe": ["el", "ro", "ru", "uk"],
    "Latin America": ["quz", "fil"],
    "Confucian": ["ja", "ko", "zh"],
    "West & South Asia": ["he", "hi", "te", "th", "vi"],
    "African-Islamic": ["ar", "bn", "fa", "id", "sw", "tr"],
}

GROUPINGS = {"family": LANGUAGE_FAMILIES, "culture": CULTURAL_ZONES}


def active_languages() -> list[str]:
    """Dataset languages kept in the analysis."""
    return [c for c in LANGUAGES if c not in UNSUPPORTED_LANGUAGES]


def ordered_languages(grouping: str) -> list[tuple[str, str]]:
    """(group, code) pairs in axis order: groups in dict order, alphabetical by name inside."""
    keep = set(active_languages())
    out = []
    for group, codes in GROUPINGS[grouping].items():
        for code in sorted((c for c in codes if c in keep), key=lambda c: LANGUAGES[c]):
            out.append((group, code))
    return out


def _check_groupings() -> None:
    for name, groups in GROUPINGS.items():
        codes = [c for cs in groups.values() for c in cs]
        missing = set(LANGUAGES) - set(codes)
        dupes = {c for c in codes if codes.count(c) > 1}
        assert not missing and not dupes, f"{name}: missing={missing} duplicates={dupes}"


_check_groupings()
