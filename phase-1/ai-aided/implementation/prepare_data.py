"""
Step 1: read XM3600 and write the caption and image tables.

Input : xm3600/captions.jsonl, xm3600/<key>.jpg
Output: processed/captions.parquet, processed/images.parquet
Also prints source image resolutions and the fixed SigLIP 2 input size.

Expected captions.jsonl line (one per image):
  {"image/key": "...", "image/locale": "de",
   "ar": {"caption": ["...", "..."], ...}, "bn": {...}, ...}
"""
import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .xm3600_reader import read_records


def load_captions() -> tuple[pd.DataFrame, pd.DataFrame]:
    image_rows, caption_rows = read_records()   # same row order as embed.py uses
    images = pd.DataFrame(image_rows, columns=["image_key", "locale"])
    captions = pd.DataFrame(caption_rows,
                            columns=["image_key", "locale", "caption_lang", "caption_idx", "text"])
    captions.insert(0, "caption_id", np.arange(len(captions)))
    return images, captions


def add_resolutions(images: pd.DataFrame) -> pd.DataFrame:
    widths, heights = [], []
    for key in images["image_key"]:
        with Image.open(config.IMAGES_DIR / f"{key}.jpg") as im:
            w, h = im.size
        widths.append(w)
        heights.append(h)
    images = images.assign(width=widths, height=heights)
    images["pixels"] = images["width"] * images["height"]
    return images


def print_summary(images: pd.DataFrame, captions: pd.DataFrame) -> None:
    print(f"images: {len(images)}   captions: {len(captions)}")
    print("images per locale (should be 100):",
          images["locale"].value_counts().describe()[["min", "max"]].to_dict())
    per = captions.groupby(["image_key", "caption_lang"]).size()
    print("captions per (image, language): "
          f"mean {per.mean():.2f}, min {per.min()}, max {per.max()}")
    unknown = set(images["locale"]) - set(config.LANGUAGES)
    if unknown:
        print("WARNING: locales not in config.LANGUAGES:", unknown)

    px = images["pixels"]
    print("\nResolution (pixels per image):")
    for q in [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 1]:
        print(f"  {int(q * 100):>3}th pct: {int(px.quantile(q)):>9,d}")
    print("most common sizes:")
    print((images["width"].astype(str) + "x" + images["height"].astype(str))
          .value_counts().head(8).to_string())
    print(f"\nSigLIP 2's processor resizes images to "
          f"{config.IMAGE_SIZE} x {config.IMAGE_SIZE}; source aspect ratios may change.")


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    images, captions = load_captions()
    images = add_resolutions(images)
    config.EMBEDDING_METADATA.unlink(missing_ok=True)
    config.SCORE_METADATA.unlink(missing_ok=True)
    images.to_parquet(config.IMAGES_TABLE, index=False)
    captions.to_parquet(config.CAPTIONS_TABLE, index=False)
    print_summary(images, captions)


if __name__ == "__main__":
    main()
