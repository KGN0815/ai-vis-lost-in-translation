"""
Pure-Python reader for XM3600 (no pandas / pyarrow), shared by prepare_data.py and embed.py.

Both scripts MUST produce rows in the same order, so the row order of the embedding files
matches processed/images.parquet and processed/captions.parquet. That order is defined here:
  images  : order of the lines in captions.jsonl
  captions: per image line, languages in config.LANGUAGES order, captions in file order
"""
import json

from . import config


def read_records():
    """Returns (images, captions):
    images   = [(image_key, locale), ...]
    captions = [(image_key, locale, caption_lang, caption_idx, text), ...]  # list index = caption_id
    """
    images, captions = [], []
    if not config.CAPTIONS_JSONL.is_file():
        raise FileNotFoundError(f"Missing {config.CAPTIONS_JSONL}. "
                                "Place the XM3600 captions and JPGs in xm3600/; see README.md.")
    with open(config.CAPTIONS_JSONL, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            key, locale = item["image/key"], item["image/locale"]
            images.append((key, locale))
            for lang in config.LANGUAGES:
                entry = item.get(lang)
                if entry is None:
                    continue
                texts = entry["caption"]
                limit = config.MAX_CAPTIONS_PER_LANG
                if limit is not None:
                    texts = texts[:limit]
                for idx, text in enumerate(texts):
                    captions.append((key, locale, lang, idx, text))
    return images, captions
