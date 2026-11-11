"""Phase I output analysis. Reuses stored embeddings; never runs the image model.

python -m implementation.analyze --bootstrap 2000 --permutations 4999
"""
import argparse
import numpy as np
import pandas as pd

from . import config
from .artifacts import read_metadata, record_fingerprint, score_settings, write_metadata

VERSION = 1
TOP_K = 5
SEED = 20261111


def analysis_path(name):
    return config.PROCESSED_DIR / name


def cached_snapshot():
    """Find the exact checkpoint in the configured or project-local cache offline."""
    from huggingface_hub import snapshot_download, constants
    from huggingface_hub.errors import LocalEntryNotFoundError
    for cache in [constants.HF_HUB_CACHE, str(config.ROOT / "data" / "huggingface" / "hub")]:
        try:
            return snapshot_download(config.MODEL_NAME, revision=config.MODEL_REVISION,
                                     cache_dir=cache, local_files_only=True)
        except LocalEntryNotFoundError:
            continue
    raise RuntimeError("Pinned SigLIP 2 checkpoint is not cached. Cache it first; "
                       "token diagnostics can be omitted with python -m implementation.analyze --skip-tokenizer.")


def validate_inputs():
    metadata = read_metadata(config.SCORE_METADATA, "-m implementation.compute_scores", score_settings())
    embedding = read_metadata(config.EMBEDDING_METADATA, "-m implementation.embed")
    if any(metadata.get(k) != v for k, v in embedding.items()):
        raise ValueError("Score and embedding metadata identify different runs.")
    captions = pd.read_parquet(config.CAPTIONS_TABLE)
    images = pd.read_parquet(config.IMAGES_TABLE)
    fingerprint = record_fingerprint(
        images[["image_key", "locale"]].itertuples(index=False, name=None),
        captions[["image_key", "locale", "caption_lang", "caption_idx", "text"]]
        .itertuples(index=False, name=None),
    )
    if fingerprint != metadata["records_sha256"]:
        raise ValueError("Caption/image tables differ from the recorded embedding order.")
    if not np.array_equal(captions.caption_id, np.arange(len(captions))):
        raise ValueError("Caption IDs must identify the original embedding rows.")
    image_vectors = np.load(config.IMAGE_EMB, mmap_mode="r", allow_pickle=False)
    text_vectors = np.load(config.CAPTION_EMB, mmap_mode="r", allow_pickle=False)
    if (image_vectors.shape != (len(images), config.EMBEDDING_DIM)
            or text_vectors.shape != (len(captions), config.EMBEDDING_DIM)):
        raise ValueError("Embedding dimensions or row counts do not match the tables.")
    return metadata, images, captions, image_vectors, text_vectors


def rank_similarities(similarities, owners, top_k=TOP_K):
    """Exact rank, stable top-k, and best incorrect candidate; ties use image order.

    Returned indices address columns of similarities. A positive hard margin means
    the true image strictly beats every incorrect image. It is not a probability.
    """
    sims = np.asarray(similarities)
    owners = np.asarray(owners, dtype=int)
    if sims.ndim != 2 or sims.shape[1] < 2 or len(owners) != len(sims) or top_k < 1:
        raise ValueError("Retrieval requires rows of scores against at least two images.")
    if not np.isfinite(sims).all() or (owners < 0).any() or (owners >= sims.shape[1]).any():
        raise ValueError("Invalid similarities or true-image indices.")
    positions = np.arange(sims.shape[1])
    true = sims[np.arange(len(sims)), owners]
    rank = 1 + (sims > true[:, None]).sum(axis=1)
    rank += ((sims == true[:, None]) & (positions[None, :] < owners[:, None])).sum(axis=1)
    k = min(top_k, sims.shape[1])
    candidates = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(sims, candidates, axis=1)
    # Resolve ties at the partition boundary too, not just within the selected k.
    boundary = candidate_scores.min(axis=1)
    for row in range(len(sims)):
        above = np.flatnonzero(sims[row] > boundary[row])
        equal = np.flatnonzero(sims[row] == boundary[row])[:k - len(above)]
        selected = np.concatenate([above, equal])
        candidates[row] = selected[np.lexsort((selected, -sims[row, selected]))]
    top_scores = np.take_along_axis(sims, candidates, axis=1)
    wrong = sims.copy()
    wrong[np.arange(len(sims)), owners] = -np.inf
    wrong_index = wrong.argmax(axis=1)
    wrong_score = wrong[np.arange(len(sims)), wrong_index]
    return {
        "rank": rank.astype(np.int32), "top_indices": candidates.astype(np.int32),
        "top_scores": top_scores.astype(np.float32), "wrong_index": wrong_index,
        "hard_margin": true - wrong_score, "cosine": true,
    }


def compute_retrieval(images, captions, image_vectors, text_vectors, chunk=512):
    langs = config.active_languages()
    mask = images.locale.isin(langs) if config.DROP_IMAGES_FROM_DISTRACTORS else np.ones(len(images), bool)
    original_indices = np.flatnonzero(mask)
    candidates = images.iloc[original_indices].reset_index(drop=True)
    lookup = {key: i for i, key in enumerate(candidates.image_key)}
    selected = captions[captions.caption_lang.isin(langs) & captions.image_key.isin(lookup)].copy()
    vectors = np.asarray(image_vectors[original_indices], dtype=np.float32)
    frames, neighbors, neighbor_scores = [], [], []
    for start in range(0, len(selected), chunk):
        batch = selected.iloc[start:start + chunk]
        similarities = np.asarray(text_vectors[batch.caption_id.to_numpy()], dtype=np.float32) @ vectors.T
        owners = batch.image_key.map(lookup).to_numpy()
        result = rank_similarities(similarities, owners)
        frame = batch.copy()
        frame["rank"] = result["rank"]
        frame["cosine"] = result["cosine"]
        frame["hard_margin"] = result["hard_margin"]
        frame["top_image_key"] = candidates.image_key.to_numpy()[result["top_indices"][:, 0]]
        frame["retrieved_locale"] = candidates.locale.to_numpy()[result["top_indices"][:, 0]]
        frame["wrong_image_key"] = candidates.image_key.to_numpy()[result["wrong_index"]]
        frame["reciprocal_rank"] = 1.0 / result["rank"]
        for k in [1, 5, 10]:
            frame[f"recall_at_{k}"] = (result["rank"] <= k).astype(float)
        # Preserve the established mean-distractor margin alongside the hard margin.
        true = result["cosine"]
        frame["margin"] = true - (similarities.sum(axis=1) - true) / (len(candidates) - 1)
        frames.append(frame)
        neighbors.append(original_indices[result["top_indices"]])
        neighbor_scores.append(result["top_scores"])
        if start % (chunk * 50) == 0:
            print(f"Retrieval: {start:,}/{len(selected):,}", flush=True)
    return pd.concat(frames, ignore_index=True), np.concatenate(neighbors), np.concatenate(neighbor_scores)


# These are observable caption cues, not asserted image labels or cultural labels.
TOPICS = {
    "Animals": r"\b(dog|dogs|cat|cats|bird|birds|horse|horses|cow|cows|sheep|chicken|rooster|hen|elephant|monkey|fish|insect|butterfly|duck|ducks)\b",
    "Vehicles": r"\b(car|cars|truck|trucks|bus|buses|train|trains|motorcycle|bicycle|bike|boat|boats|ship|plane|airplane|vehicle|vehicles)\b",
    "Food": r"\b(food|meal|dish|bread|rice|cake|cakes|pizza|fruit|vegetable|vegetables|restaurant|cooking|cooked|eating|salad|meat|soup|sandwich)\b",
    "Buildings": r"\b(building|buildings|house|houses|church|temple|mosque|castle|palace|cathedral|architecture|tower|bridge|statue|monument)\b",
    "Nature": r"\b(mountain|mountains|forest|trees|tree|river|lake|sea|ocean|beach|flower|flowers|waterfall|landscape|sunset|garden|field)\b",
    "People": r"\b(person|people|man|woman|men|women|child|children|boy|girl|crowd|family|player|players)\b",
    "Text cue": r"\b(sign|signs|signage|text|letter|letters|written|writing|label|logo|number|numbers|billboard|poster)\b",
}


def caption_features(captions, tokenizer=None):
    output = captions[["caption_id", "image_key", "caption_lang"]].copy()
    output["characters"] = captions.text.str.len()
    output["has_digit"] = captions.text.str.contains(r"\d", regex=True)
    if tokenizer is not None:
        counts = []
        for start in range(0, len(captions), 4096):
            encoded = tokenizer(captions.text.iloc[start:start + 4096].str.lower().tolist(),
                                padding=False, truncation=False, add_special_tokens=True)
            counts.extend(map(len, encoded["input_ids"]))
        output["tokens"] = np.array(counts, dtype=np.int32)
        output["truncated"] = output.tokens > config.TEXT_MAX_LENGTH
    english = captions[captions.caption_lang == "en"].groupby("image_key").text.agg(" ".join)
    for name, pattern in TOPICS.items():
        flags = english.str.contains(pattern.replace("(", "(?:"), case=False, regex=True)
        output[f"topic_{name}"] = captions.image_key.map(flags).fillna(False).astype(bool)
    return output


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    values = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(values, 1)
    return result


def matching_statistics(per_image, metric="margin", bootstrap=2000, permutations=4999, seed=SEED):
    """Image-level, locale-stratified bootstrap and common image-label permutations.

    Captions are averaged before this function. A bootstrap sample keeps the
    entire language vector of each sampled image together. Every permutation
    shuffles one shared locale vector, preserving locale counts and dependence.
    """
    table = per_image.pivot(index="image_key", columns="caption_lang", values=metric)
    locales = per_image.drop_duplicates("image_key").set_index("image_key").locale.reindex(table.index)
    languages = [c for c in config.active_languages() if c in table and c in set(locales)]
    table = table[languages]
    if table.isna().any().any() or len(languages) < 2:
        raise ValueError("Matching inference needs complete image/language vectors and at least two locales.")
    y = table.to_numpy(dtype=float)
    labels = np.array([languages.index(c) if c in languages else -1 for c in locales])
    keep = labels >= 0
    y, labels = y[keep], labels[keep]
    n, l = y.shape
    sizes = np.bincount(labels, minlength=l)
    if (sizes < 2).any():
        raise ValueError("Each locale needs at least two images for bootstrap uncertainty.")
    means = np.stack([y[labels == j].mean(axis=0) for j in range(l)])
    weights = sizes / n

    def effects(m, adjusted):
        if adjusted:
            m = m - m.mean(axis=-1, keepdims=True)
        matched = np.diagonal(m, axis1=-2, axis2=-1)
        total = (m * weights[:, None]).sum(axis=-2)
        other = (total - matched * weights) / (1 - weights)
        return matched - other

    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap, l, l))
    for j in range(l):
        group = y[labels == j]
        # Shared multinomial image counts preserve paired language measurements.
        counts = rng.multinomial(len(group), np.full(len(group), 1 / len(group)), size=bootstrap)
        boot[:, j] = counts @ group / len(group)
    rows = []
    global_results = {}
    for adjusted in [False, True]:
        observed = effects(means, adjusted)
        low, high = np.quantile(effects(boot, adjusted), [0.025, 0.975], axis=0)
        data = y - y.mean(axis=1, keepdims=True) if adjusted else y
        totals = data.sum(axis=0)
        null = np.empty((permutations, l))
        # Reset so both views use exactly the same randomized locale assignments.
        permutation_rng = np.random.default_rng(seed + 1)
        for b in range(permutations):
            shuffled = permutation_rng.permutation(labels)
            matched_sum = np.bincount(shuffled, weights=data[np.arange(n), shuffled], minlength=l)
            null[b] = matched_sum / sizes - (totals - matched_sum) / (n - sizes)
        p = (1 + (np.abs(null) >= np.abs(observed)).sum(axis=0)) / (permutations + 1)
        q = bh_adjust(p)
        view = "image_centered" if adjusted else "raw"
        global_results[view] = {
            "effect": float(observed.mean()),
            "p": float((1 + (np.abs(null.mean(axis=1)) >= abs(observed.mean())).sum()) / (permutations + 1)),
        }
        for j, lang in enumerate(languages):
            rows.append({"caption_lang": lang, "metric": metric, "view": view,
                         "effect": observed[j], "ci_low": low[j], "ci_high": high[j],
                         "p": p[j], "q_bh": q[j], "n_matched": int(sizes[j]),
                         "n_other": int(n - sizes[j])})
    return pd.DataFrame(rows), global_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip token counts explicitly; never substitutes character counts.")
    args = parser.parse_args(argv)
    if args.bootstrap < 100 or args.permutations < 99:
        parser.error("Use at least 100 bootstrap samples and 99 permutations.")
    metadata, images, captions, iv, tv = validate_inputs()
    target = analysis_path("analysis_metadata.json")
    target.unlink(missing_ok=True)
    retrieval, neighbors, neighbor_scores = compute_retrieval(images, captions, iv, tv)
    retrieval.to_parquet(analysis_path("retrieval.parquet"), index=False)
    np.save(analysis_path("retrieval_indices.npy"), neighbors.astype(np.int32))
    np.save(analysis_path("retrieval_cosines.npy"), neighbor_scores)
    tokenizer = None
    if not args.skip_tokenizer:
        from transformers import AutoTokenizer
        snapshot = cached_snapshot()
        # A local path also avoids Transformers' optional remote model-info probe.
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    features = caption_features(captions, tokenizer)
    features.to_parquet(analysis_path("caption_features.parquet"), index=False)
    metric_columns = ["cosine", "margin", "hard_margin", "reciprocal_rank", "recall_at_1", "recall_at_5", "recall_at_10"]
    per_image = retrieval.groupby(["image_key", "locale", "caption_lang"], as_index=False)[metric_columns].mean()
    per_image.to_parquet(analysis_path("retrieval_image_language.parquet"), index=False)
    results, global_results = [], {}
    for subset, frame in [("all", per_image), ("without_hi", per_image[per_image.locale != "hi"])]:
        for metric in ["margin", "recall_at_1"]:
            print(f"Inference: {subset}, {metric}", flush=True)
            stats, global_stats = matching_statistics(frame, metric, args.bootstrap, args.permutations, args.seed)
            results.append(stats.assign(subset=subset))
            global_results[f"{subset}/{metric}"] = global_stats
    pd.concat(results).to_parquet(analysis_path("matching_effects.parquet"), index=False)
    # A balanced, score-independent review sample includes successes and failures.
    sample = (retrieval.groupby(["caption_lang", "locale"], group_keys=False)
              .sample(n=1, random_state=args.seed)[["caption_id", "image_key", "caption_lang", "text", "rank", "top_image_key"]])
    for column in ["error_type", "visible_text", "specific_name", "notes", "reviewer"]:
        sample[column] = ""
    sample.to_csv(analysis_path("error_review_template.csv"), index=False)
    write_metadata(target, {**metadata, "analysis_version": VERSION, "bootstrap": args.bootstrap,
                           "permutations": args.permutations, "seed": args.seed, "top_k": TOP_K,
                           "token_counts_available": tokenizer is not None,
                           "n_retrieval_captions": len(retrieval), "global_tests": global_results})
    print(f"Analysis ready: {len(retrieval):,} caption queries; {len(per_image):,} image/language units.")


if __name__ == "__main__":
    main()
