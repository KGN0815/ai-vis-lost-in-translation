"""Identify the checkpoint, preprocessing and row order behind generated results."""
import hashlib
import json
from pathlib import Path

from . import config


def model_settings() -> dict:
    return {
        "model": config.MODEL_NAME,
        "revision": config.MODEL_REVISION,
        "dimension": config.EMBEDDING_DIM,
        "image_size": config.IMAGE_SIZE,
        "text_max_length": config.TEXT_MAX_LENGTH,
        "lowercase": True,
        "use_fast_processor": False,
        "normalized": True,
        "inference_dtype": "float32",
        "embedding_dtype": "float16",
        "max_captions_per_lang": config.MAX_CAPTIONS_PER_LANG,
    }


def record_fingerprint(images, captions) -> str:
    """Hash image and caption records in the exact order used by the pipeline."""
    digest = hashlib.sha256()
    for records in (images, captions):
        digest.update(b"records\n")
        for row in records:
            digest.update(json.dumps(tuple(row), ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def run_metadata(images, captions) -> dict:
    return {
        **model_settings(),
        "n_images": len(images),
        "n_captions": len(captions),
        "records_sha256": record_fingerprint(images, captions),
    }


def score_settings() -> dict:
    return {
        "analysis_languages": config.active_languages(),
        "drop_images_from_distractors": config.DROP_IMAGES_FROM_DISTRACTORS,
    }


def write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_metadata(path: Path, command: str, expected: dict | None = None) -> dict:
    """Reject missing or incompatible results instead of silently relabelling them."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"No valid {path.name}. Run `python {command}` first; see README.md.") from exc
    settings = {**model_settings(), **(expected or {})}
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in settings.items()):
        raise RuntimeError(f"{path.name} does not match the configured SigLIP 2 run. "
                           f"Regenerate it with `python {command}`.")
    return metadata
