"""Prepare and score a balanced English-to-German caption control.

prepare exports source captions before examining scores. Fill translation,
translator and review_status, then score using only new text embeddings.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .analyze import SEED, analysis_path, cached_snapshot, rank_similarities, validate_inputs
from .artifacts import write_metadata

DEFAULT_INPUT = config.ROOT / "annotations" / "translations_en_de.csv"


def prepare(path, per_locale=36, seed=SEED):
    if path.exists():
        raise ValueError(f"Refusing to overwrite translations: {path}")
    captions = pd.read_parquet(config.CAPTIONS_TABLE)
    english = (captions[captions.caption_lang == "en"]
               .sort_values(["image_key", "caption_idx"]).drop_duplicates("image_key"))
    frames = []
    for locale in ["en", "de"]:
        candidates = english[english.locale == locale].sort_values("image_key")
        if len(candidates) < per_locale:
            raise ValueError(f"Need {per_locale} English-captioned images in locale {locale}.")
        frames.append(candidates.sample(n=per_locale, random_state=seed))
    sample = pd.concat(frames).sort_values(["locale", "image_key"])
    output = sample[["caption_id", "image_key", "locale", "caption_lang", "text"]].rename(
        columns={"caption_id": "source_caption_id", "caption_lang": "source_lang", "text": "source_text"})
    output.insert(0, "pair_id", [f"en-de-{i:03}" for i in range(len(output))])
    output["target_lang"] = "de"
    for column in ["translation", "translator", "review_status", "review_notes"]:
        output[column] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(f"Exported {len(output)} image pairs to {path}; seed {seed}; no model scores used.")


def validate_translations(frame, captions):
    required = {"pair_id", "source_caption_id", "image_key", "locale", "source_lang", "source_text",
                "target_lang", "translation", "translator", "review_status"}
    if not required.issubset(frame):
        raise ValueError(f"Missing translation columns: {sorted(required - set(frame.columns))}")
    if frame.empty or frame.pair_id.duplicated().any() or frame.source_caption_id.duplicated().any():
        raise ValueError("Translation pairs and source caption IDs must be nonempty and unique.")
    source = captions.set_index("caption_id")
    for row in frame.itertuples():
        if row.source_caption_id not in source.index:
            raise ValueError("Unknown source caption ID.")
        original = source.loc[row.source_caption_id]
        if (row.image_key, row.locale, row.source_lang, row.source_text) != (
                original.image_key, original.locale, original.caption_lang, original.text):
            raise ValueError("Translation source text or image does not match the original caption ID.")
        if row.target_lang not in config.active_languages() or row.target_lang == row.source_lang:
            raise ValueError("Target must be a different active language.")
        if row.review_status not in {"ai_generated", "human_reviewed", "rejected"}:
            raise ValueError("review_status must be ai_generated, human_reviewed, or rejected.")
        if row.review_status != "rejected" and (not str(row.translation).strip() or not str(row.translator).strip()):
            raise ValueError("Each included translation needs text and translator provenance.")
    return frame[frame.review_status != "rejected"].copy()


def paired_summary(pairs, bootstrap=2000, seed=SEED):
    rows = []
    rng = np.random.default_rng(seed)
    for locale in ["all", *sorted(pairs.locale.unique())]:
        group = pairs if locale == "all" else pairs[pairs.locale == locale]
        for metric in ["margin", "recall_at_1"]:
            for name, a, b in [
                ("translation_minus_source", f"translated_{metric}", f"source_{metric}"),
                ("human_target_minus_translation", f"human_target_{metric}", f"translated_{metric}"),
            ]:
                delta = (group[a] - group[b]).to_numpy(dtype=float)
                # Resample whole image pairs; overall estimate preserves locale balance.
                estimates = np.zeros(bootstrap)
                for loc in sorted(group.locale.unique()):
                    values = delta[group.locale.to_numpy() == loc]
                    draws = rng.integers(0, len(values), (bootstrap, len(values)))
                    estimates += values[draws].mean(axis=1) * len(values) / len(delta)
                low, high = np.quantile(estimates, [0.025, 0.975])
                rows.append({"locale": locale, "metric": metric, "comparison": name,
                             "effect": delta.mean(), "ci_low": low, "ci_high": high,
                             "n_images": len(group),
                             "n_human_reviewed": int((group.review_status == "human_reviewed").sum())})
    return pd.DataFrame(rows)


def score(path, device="cpu"):
    from . import embed
    from transformers import AutoModel, AutoProcessor
    metadata, images, captions, iv, _ = validate_inputs()
    frame = validate_translations(pd.read_csv(path, keep_default_na=False), captions)
    if frame.empty:
        raise ValueError("No non-rejected translation pairs.")
    original = pd.read_parquet(analysis_path("retrieval.parquet")).set_index("caption_id")
    analysis_path("translation_metadata.json").unlink(missing_ok=True)
    snapshot = cached_snapshot()
    encoder = embed.SiglipEmbedder(
        AutoModel.from_pretrained(snapshot, local_files_only=True),
        AutoProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False),
        embed.pick_device(device),
    )
    vectors = encoder.captions(frame.translation.tolist())
    mask = images.locale.isin(config.active_languages()) if config.DROP_IMAGES_FROM_DISTRACTORS else np.ones(len(images), bool)
    candidates = images[mask].reset_index(drop=True)
    image_vectors = np.asarray(iv[np.flatnonzero(mask)], dtype=np.float32)
    lookup = {key: i for i, key in enumerate(candidates.image_key)}
    similarities = vectors.astype(np.float32) @ image_vectors.T
    result = rank_similarities(similarities, frame.image_key.map(lookup).to_numpy())
    output = frame.copy().reset_index(drop=True)
    true = result["cosine"]
    output["translated_cosine"] = true
    output["translated_margin"] = true - (similarities.sum(axis=1) - true) / (len(candidates) - 1)
    output["translated_rank"] = result["rank"]
    output["translated_recall_at_1"] = (result["rank"] == 1).astype(float)
    output["translated_top_image_key"] = candidates.image_key.to_numpy()[result["top_indices"][:, 0]]
    human = original.groupby(["image_key", "caption_lang"])[["cosine", "margin", "rank", "recall_at_1"]].mean()
    for metric in ["cosine", "margin", "rank", "recall_at_1"]:
        output[f"source_{metric}"] = original.loc[output.source_caption_id, metric].to_numpy()
        output[f"human_target_{metric}"] = [human.loc[(r.image_key, r.target_lang), metric] for r in output.itertuples()]
    output.to_parquet(analysis_path("translation_pairs.parquet"), index=False)
    paired_summary(output).to_parquet(analysis_path("translation_effects.parquet"), index=False)
    np.save(analysis_path("translation_embeddings.npy"), vectors)
    input_file = str(path.resolve().relative_to(config.ROOT)) if path.resolve().is_relative_to(config.ROOT) else str(path.resolve())
    write_metadata(analysis_path("translation_metadata.json"), {
        **metadata, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_file": input_file, "n_pairs": len(output),
        "n_human_reviewed": int((output.review_status == "human_reviewed").sum()),
        "source_languages": sorted(set(output.source_lang)), "target_languages": sorted(set(output.target_lang)),
        "bootstrap": 2000, "seed": SEED,
    })
    print(f"Scored {len(output)} translated captions; reused every image embedding and original-caption result.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--per-locale", type=int, default=36)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.per_locale < 2:
            parser.error("Use at least two images per locale.")
        prepare(args.input, args.per_locale)
    else:
        score(args.input, args.device)


if __name__ == "__main__":
    main()
