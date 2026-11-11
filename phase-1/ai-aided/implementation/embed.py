"""Step 2: encode XM3600 with the pinned SigLIP 2 checkpoint.

Images use the checkpoint's 256 x 256 processor. Text is lowercased, padded and
truncated to 64 tokens. Inference uses float32;
L2-normalized vectors are stored as float16 in the caption/image table row order.

  python -m implementation.embed --test          # up to 10 locales, no output files
  python -m implementation.embed                 # full, resumable run
  python -m implementation.embed --part 0/2      # two workers: 0/2 also encodes images, 1/2 texts only
  python -m implementation.embed --merge         # merge finished chunks without loading the model
"""
import argparse
from collections import defaultdict
import hashlib
import json
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from . import config
from .artifacts import run_metadata, write_metadata
from .xm3600_reader import read_records

CHUNK = 8192


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "xpu" and not torch.xpu.is_available():
            raise RuntimeError("Intel GPU is unavailable to PyTorch. Use an XPU-enabled PyTorch "
                               "build and Intel compute runtime, and check /dev/dri access. "
                               "See the Intel GPU instructions in README.md.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SiglipEmbedder:
    def __init__(self, model, processor, device):
        self.model = model.to(device).float().eval()
        self.processor = processor
        self.device = device

    @torch.inference_mode()
    def encode(self, items: list, modality: str, batch_size: int) -> np.ndarray:
        if not items or batch_size < 1:
            raise ValueError("Encoding needs nonempty inputs and a positive batch size.")
        parts = []
        started = time.time()
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            if modality == "image":
                photos = []
                try:
                    for path in batch:
                        with Image.open(path) as photo:
                            photos.append(photo.convert("RGB"))
                    inputs = self.processor(images=photos, return_tensors="pt").to(self.device)
                finally:
                    for photo in photos:
                        photo.close()
                features = self.model.get_image_features(**inputs)
            elif modality == "text":
                inputs = self.processor(
                    text=[text.lower() for text in batch], padding="max_length",
                    max_length=config.TEXT_MAX_LENGTH, truncation=True, return_tensors="pt",
                ).to(self.device)
                features = self.model.get_text_features(**inputs)
            else:
                raise ValueError(f"Unknown modality: {modality}")

            features = features.float()
            if features.shape != (len(batch), config.EMBEDDING_DIM):
                raise ValueError(f"Unexpected SigLIP 2 feature shape: {tuple(features.shape)}")
            if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
                raise ValueError("The model returned nonfinite or zero feature vectors.")
            vectors = F.normalize(features, dim=-1)
            parts.append(vectors.cpu().numpy().astype(np.float16))
            done = min(start + batch_size, len(items))
            print(f"  {done}/{len(items)} ({done / (time.time() - started):.1f}/s)", flush=True)
        return np.concatenate(parts)

    def images(self, keys) -> np.ndarray:
        paths = [config.IMAGES_DIR / f"{key}.jpg" for key in keys]
        return self.encode(paths, "image", config.EMBED_BATCH_SIZE_IMAGE)

    def captions(self, texts) -> np.ndarray:
        return self.encode(list(texts), "text", config.EMBED_BATCH_SIZE_TEXT)


def load_model(device: torch.device) -> SiglipEmbedder:
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(config.MODEL_NAME, revision=config.MODEL_REVISION)
    processor = AutoProcessor.from_pretrained(
        config.MODEL_NAME, revision=config.MODEL_REVISION, use_fast=False,
    )
    embedder = SiglipEmbedder(model, processor, device)
    print(f"{config.MODEL_NAME} @ {config.MODEL_REVISION} on {device}, float32", flush=True)
    if device.type == "xpu":
        print(f"Intel GPU: {torch.xpu.get_device_name(device)}", flush=True)
    return embedder


def run_test(model, images, captions) -> None:
    seen, sample = set(), []
    for key, locale in images:
        if locale not in seen:
            seen.add(locale)
            sample.append(key)
        if len(sample) == 10:
            break
    if len(sample) < 2:
        raise ValueError("The sanity check needs images from at least two locales.")
    positions = {key: i for i, key in enumerate(sample)}
    caps = [caption for caption in captions if caption[0] in positions]
    print(f"sanity check: {len(sample)} images, {len(caps)} captions", flush=True)
    image_emb = model.images(sample).astype(np.float32)
    caption_emb = model.captions([caption[4] for caption in caps]).astype(np.float32)
    print(f"shapes: images {image_emb.shape}, captions {caption_emb.shape}")
    print(f"first image norms: {np.linalg.norm(image_emb, axis=1)[:3]}")
    print(f"first caption norms: {np.linalg.norm(caption_emb, axis=1)[:3]}")
    similarities = caption_emb @ image_emb.T
    true = np.array([positions[caption[0]] for caption in caps])
    own = similarities[np.arange(len(true)), true]
    other = (similarities.sum(axis=1) - own) / (len(sample) - 1)
    hits = similarities.argmax(axis=1) == true
    print(f"mean cosine: own image {own.mean():.3f}, other images {other.mean():.3f}")
    print(f"top-1: {hits.mean():.1%} (chance {1 / len(sample):.1%})")
    by_language = defaultdict(list)
    for caption, hit in zip(caps, hits):
        by_language[caption[2]].append(hit)
    worst = sorted((np.mean(values), lang) for lang, values in by_language.items())[:8]
    print("lowest top-1 languages:", ", ".join(f"{lang}: {rate:.2f}" for rate, lang in worst))


def chunk_dir_for(metadata: dict):
    # Checkpoint, preprocessing, caption selection AND record order identify a run.
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    return config.PROCESSED_DIR / f"chunks_siglip2_{signature}"


def check_array(path, rows: int) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (rows, config.EMBEDDING_DIM) or array.dtype != np.float16:
        raise ValueError(f"{path}: expected {(rows, config.EMBEDDING_DIM)} float16, "
                         f"got {array.shape} {array.dtype}. Remove this incomplete chunk and rerun.")
    return array


def save_array(path, array) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(path)


def merge_chunks(metadata: dict) -> bool:
    directory = chunk_dir_for(metadata)
    n_captions = metadata["n_captions"]
    n_chunks = (n_captions + CHUNK - 1) // CHUNK
    paths = [directory / f"captions_{i:04d}.npy" for i in range(n_chunks)]
    image_path = directory / "image_embeddings.npy"
    missing = [path.name for path in [image_path, *paths] if not path.is_file()]
    if missing:
        print(f"{len(missing)} files still missing. Finish all workers, then run "
              "`python -m implementation.embed --merge`.")
        return False

    image_emb = check_array(image_path, metadata["n_images"])
    for i, path in enumerate(paths):
        check_array(path, min(CHUNK, n_captions - i * CHUNK))
    config.EMBEDDING_METADATA.unlink(missing_ok=True)
    config.SCORE_METADATA.unlink(missing_ok=True)
    temporary = config.CAPTION_EMB.with_suffix(".tmp.npy")
    merged = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(n_captions, config.EMBEDDING_DIM),
    )
    for i, path in enumerate(paths):
        chunk = np.load(path, mmap_mode="r", allow_pickle=False)
        merged[i * CHUNK:i * CHUNK + len(chunk)] = chunk
    merged.flush()
    del merged
    temporary.replace(config.CAPTION_EMB)
    save_array(config.IMAGE_EMB, image_emb)
    write_metadata(config.EMBEDDING_METADATA, metadata)
    print(f"written: {metadata['n_images']} image and {n_captions} caption embeddings "
          f"({config.EMBEDDING_DIM} dimensions), with checkpoint and row-order metadata")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true", help="small sanity check, writes nothing")
    mode.add_argument("--merge", action="store_true", help="merge finished chunks, no model loaded")
    parser.add_argument("--device", default="auto", help="auto, cuda, xpu, mps or cpu")
    parser.add_argument("--part", default="0/1", help="worker i/n (0/2 and 1/2 for two workers)")
    args = parser.parse_args()
    try:
        part_i, part_n = (int(value) for value in args.part.split("/"))
        if not 0 <= part_i < part_n:
            raise ValueError
    except ValueError:
        parser.error("--part must be i/n with n > 0 and 0 <= i < n")

    images, captions = read_records()
    if not images or not captions:
        parser.error("The dataset must contain images and captions.")
    metadata = run_metadata(images, captions)
    if args.merge:
        if not merge_chunks(metadata):
            raise SystemExit(1)
        return

    model = load_model(pick_device(args.device))
    if args.test:
        run_test(model, images, captions)
        return

    directory = chunk_dir_for(metadata)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / "image_embeddings.npy"
    if part_i == 0:
        if image_path.exists():
            check_array(image_path, len(images))
        else:
            print("Embedding images ...", flush=True)
            save_array(image_path, model.images([key for key, _ in images]))

    texts = [caption[4] for caption in captions]
    n_chunks = (len(texts) + CHUNK - 1) // CHUNK
    for i in range(part_i, n_chunks, part_n):
        path = directory / f"captions_{i:04d}.npy"
        chunk_texts = texts[i * CHUNK:(i + 1) * CHUNK]
        if path.exists():
            check_array(path, len(chunk_texts))
            continue
        print(f"Embedding captions, chunk {i + 1}/{n_chunks} (worker {args.part})", flush=True)
        save_array(path, model.captions(chunk_texts))
    if part_n == 1:
        merge_chunks(metadata)
    else:
        # Workers never race to replace the same merged output files.
        print(f"Worker {args.part} finished. Once all workers finish, "
              "run `python -m implementation.embed --merge`.")


if __name__ == "__main__":
    main()
