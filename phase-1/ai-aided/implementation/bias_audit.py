"""Controlled multilingual retrieval audit using existing FairFace annotations.

Run from ai-aided/: python -m implementation.bias_audit prepare|score
No demographic attributes are predicted from images. Skin tone is not measured.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .analyze import cached_snapshot, SEED
from .artifacts import model_settings, read_metadata, write_metadata

VERSION = 1
DATASET = "HuggingFaceM4/FairFace"
REVISION = "54d573cdb8b5af490ba8da9da2799628f6e5c496"
SOURCE_URL = (f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
              "0.25/validation-00000-of-00001-951dbd63c8724ee1.parquet")
SOURCE_SHA256 = "d0cfac2888356b3d3fa3d837bc30961d8c860934e4c5df862e8cb87812bfa137"
AGES = ["20-29", "30-39"]
RACES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
GENDERS = ["Male", "Female"]
LANGUAGES = ["en", "de", "fr", "es", "ja", "sw"]
AXES = ["race", "gender", "age"]
KS = (1, 10, 50)
PROMPTS = config.ROOT / "annotations" / "bias_prompts.csv"


def output(name):
    return config.PROCESSED_DIR / f"bias_{name}"


def data_dir():
    return config.ROOT / "data" / "fairface"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def sample_candidates(labels, per_stratum=16, seed=SEED):
    """Equal counts in every dataset race × gender × adult-age stratum."""
    if per_stratum < 2:
        raise ValueError("At least two candidates per stratum are required.")
    if labels.source_row.duplicated().any():
        raise ValueError("Source rows must identify distinct images.")
    frames = []
    rng = np.random.default_rng(seed)
    for race in RACES:
        for gender in GENDERS:
            for age in AGES:
                group = labels[(labels.race == race) & (labels.gender == gender) & (labels.age == age)]
                if len(group) < per_stratum:
                    raise ValueError(f"Insufficient candidates in {race}/{gender}/{age}: {len(group)}.")
                frames.append(group.iloc[rng.choice(len(group), per_stratum, replace=False)])
    # Randomize the common tie-breaking order independently of demographic groups.
    return pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)


def prepare(per_stratum=16, seed=SEED):
    if per_stratum < 3:
        raise ValueError("Use at least three images per stratum for top-50 subset checks.")
    import pyarrow.parquet as pq
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "validation-0.25.parquet"
    if not source.exists():
        temporary = source.with_suffix(".download")
        with urllib.request.urlopen(SOURCE_URL, timeout=90) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024): stream.write(block)
        temporary.replace(source)
    if digest(source) != SOURCE_SHA256:
        raise ValueError("FairFace parquet hash does not match the pinned source.")
    table = pq.read_table(source)
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    frame = table.select(["race", "gender", "age"]).to_pandas()
    for name in AXES:
        names = features[name]["names"]
        frame[name] = frame[name].map(dict(enumerate(names)))
    frame["source_row"] = np.arange(len(frame))
    selected = sample_candidates(frame, per_stratum, seed)
    images = table.column("image")
    records = []
    (root / "images").mkdir(exist_ok=True)
    for row in selected.itertuples():
        item = images[int(row.source_row)].as_py()
        raw = item["bytes"]
        with Image.open(BytesIO(raw)) as photo:
            photo.verify()
        key = f"ff-val-{row.source_row:05d}"
        filename = root / "images" / f"{key}.jpg"
        if not filename.exists() or digest(filename) != hashlib.sha256(raw).hexdigest():
            filename.write_bytes(raw)
        records.append(dict(image_key=key, source_row=row.source_row, source_path=item["path"],
                            race=row.race, gender=row.gender, age=row.age,
                            image_path=str(filename.relative_to(config.ROOT)),
                            image_sha256=hashlib.sha256(raw).hexdigest()))
    output("metadata.json").unlink(missing_ok=True)
    output("candidates.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output("candidates.csv"), index=False)
    write_metadata(output("candidate_metadata.json"), {
        "dataset": DATASET, "dataset_revision": REVISION, "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256, "source_split": "validation", "crop_padding": 0.25,
        "license": "CC BY 4.0", "per_stratum": per_stratum, "seed": seed,
        "n_candidates": len(records), "adult_age_bands": AGES,
        "manifest_sha256": digest(output("candidates.csv")),
        "labels": "Dataset-provided perceived categories; not inferred identities or skin-tone labels.",
    })
    print(f"Prepared {len(records)} candidates: 7 groups × 2 gender labels × 2 age bands × {per_stratum}.", flush=True)


def load_candidates(verify_images=False):
    metadata = json.loads(output("candidate_metadata.json").read_text())
    if (metadata.get("dataset_revision") != REVISION or metadata.get("source_sha256") != SOURCE_SHA256
            or digest(output("candidates.csv")) != metadata.get("manifest_sha256")):
        raise ValueError("Candidate manifest/source changed; rerun bias_audit prepare and score.")
    candidates = pd.read_csv(output("candidates.csv"), keep_default_na=False)
    counts = candidates.groupby(AXES).size()
    if (len(candidates) != metadata["n_candidates"] or candidates.image_key.duplicated().any()
            or len(counts) != len(RACES) * len(GENDERS) * len(AGES)
            or not counts.eq(metadata["per_stratum"]).all()
            or set(candidates.race) != set(RACES) or set(candidates.gender) != set(GENDERS)
            or set(candidates.age) != set(AGES)):
        raise ValueError("Candidate pool is not the recorded complete balanced design.")
    if verify_images:
        for row in candidates.itertuples():
            path = config.ROOT / row.image_path
            if not path.exists() or digest(path) != row.image_sha256:
                raise ValueError(f"Missing or changed candidate image: {row.image_key}")
    return candidates, metadata


def validate_prompts(prompts):
    needed = {"prompt_id", "language", "family", "kind", "text", "source_text", "review_status", "translator"}
    if not needed.issubset(prompts) or prompts.empty or prompts.prompt_id.duplicated().any():
        raise ValueError("Prompt table needs all columns and unique prompt IDs.")
    if not set(prompts.language).issubset(config.LANGUAGES) or "en" not in set(prompts.language):
        raise ValueError("Use known languages and an English reference.")
    if not set(prompts.kind).issubset({"neutral", "control"}):
        raise ValueError("Prompts must be neutral queries or object controls.")
    allowed = {"source_original", "ai_generated", "human_reviewed"}
    if not prompts.review_status.isin(allowed).all():
        raise ValueError("Every prompt needs source_original, ai_generated or human_reviewed provenance.")
    if ((prompts.language != "en") & (prompts.review_status == "source_original")).any():
        raise ValueError("Translations cannot be marked as original English source prompts.")
    if prompts[["text", "source_text", "translator"]].apply(lambda s: s.astype(str).str.strip().eq("")).any().any():
        raise ValueError("Prompt text and provenance may not be empty.")
    for family, group in prompts.groupby("family"):
        if (set(group.language) != set(prompts.language) or group.language.duplicated().any()
                or group.kind.nunique() != 1):
            raise ValueError(f"Family {family} must have one equivalent prompt per language.")
        source = group[group.language == "en"].iloc[0]
        if source.text != source.source_text or not group.source_text.eq(source.text).all():
            raise ValueError("Translation source text differs from the family English prompt.")
    if prompts[prompts.kind == "neutral"].family.nunique() < 3 or prompts[prompts.kind == "control"].family.nunique() < 2:
        raise ValueError("Use at least three neutral paraphrases and two object controls.")
    return prompts.reset_index(drop=True)


def group_exposure(scores, labels, k):
    """Per-query top-k shares, DCG-weighted shares and whole-pool group ranks."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.ndim != 2 or scores.shape[1] != len(labels) or not 1 <= k <= len(labels):
        raise ValueError("Scores, labels and top-k dimensions do not agree.")
    if not np.isfinite(scores).all(): raise ValueError("Nonfinite similarity scores.")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, len(labels) + 1), order.shape), axis=1)
    weights = 1 / np.log2(np.arange(2, k + 2))
    rows = []
    for group in sorted(set(labels)):
        indicator = labels[order[:, :k]] == group
        rows.append(dict(group=group, baseline=float((labels == group).mean()),
                         share=indicator.mean(axis=1),
                         discounted_share=(indicator * weights).sum(axis=1) / weights.sum(),
                         mean_rank=ranks[:, labels == group].mean(axis=1)))
    return rows


def exposure_tables(scores, prompts, candidates, ks=KS):
    rows = []
    neutral = prompts.kind.eq("neutral").to_numpy()
    query = prompts.loc[neutral].reset_index(drop=True)
    for axis in AXES:
        for k in ks:
            for result in group_exposure(scores[neutral], candidates[axis], k):
                for i, prompt in enumerate(query.itertuples()):
                    rows.append(dict(prompt_id=prompt.prompt_id, language=prompt.language, family=prompt.family,
                                     axis=axis, group=result["group"], k=k, baseline=result["baseline"],
                                     share=result["share"][i], discounted_share=result["discounted_share"][i],
                                     mean_rank=result["mean_rank"][i]))
    per_prompt = pd.DataFrame(rows)
    aggregate = per_prompt.groupby(["language", "axis", "group", "k"], as_index=False).agg(
        share=("share", "mean"), baseline=("baseline", "first"),
        discounted_share=("discounted_share", "mean"), mean_rank=("mean_rank", "mean"),
        n_prompts=("prompt_id", "size"))
    aggregate["excess_pp"] = 100 * (aggregate.share - aggregate.baseline)
    reference = aggregate[aggregate.language == "en"][["axis", "group", "k", "share"]].rename(columns={"share": "english_share"})
    aggregate = aggregate.merge(reference, on=["axis", "group", "k"], validate="many_to_one")
    aggregate["delta_english_pp"] = 100 * (aggregate.share - aggregate.english_share)
    return per_prompt, aggregate


def subsample_stability(scores, prompts, candidates, repetitions=500, seed=SEED, fraction=0.75):
    """Paired balanced subsets; ranges describe sensitivity, not population CIs.

    Every language sees identical subsets in each repetition. Prompt families
    stay fixed and are not treated as independent human subjects.
    """
    if not 0 < fraction < 1 or repetitions < 2:
        raise ValueError("Use repeated proper subsets.")
    strata = [group.index.to_numpy() for _, group in candidates.groupby(AXES, sort=True)]
    take = int(np.floor(min(map(len, strata)) * fraction))
    if take < 1: raise ValueError("Not enough images for balanced subsets.")
    if take * len(strata) < max(KS):
        raise ValueError("Balanced subsets must contain at least 50 images.")
    rng = np.random.default_rng(seed)
    samples = []
    for b in range(repetitions):
        indices = np.sort(np.concatenate([rng.choice(group, take, replace=False) for group in strata]))
        subset = candidates.iloc[indices].reset_index(drop=True)
        _, summary = exposure_tables(scores[:, indices], prompts, subset)
        samples.append(summary[["language", "axis", "group", "k", "excess_pp", "delta_english_pp"]].assign(repetition=b))
    frame = pd.concat(samples, ignore_index=True)
    ranges = frame.groupby(["language", "axis", "group", "k"]).agg(
        stability_low=("excess_pp", lambda x: x.quantile(.025)),
        stability_high=("excess_pp", lambda x: x.quantile(.975)),
        delta_stability_low=("delta_english_pp", lambda x: x.quantile(.025)),
        delta_stability_high=("delta_english_pp", lambda x: x.quantile(.975)),
    ).reset_index()
    return ranges, take * len(strata)


def sanity_table(scores, prompts, candidates):
    rows = []
    for language, group in prompts.groupby("language", sort=False):
        neutral_indices = group.index[group.kind == "neutral"]
        control_indices = group.index[group.kind == "control"]
        positive = scores[neutral_indices].mean(axis=0)
        negative = scores[control_indices].max(axis=0)
        top10 = np.argsort(-scores[neutral_indices], axis=1, kind="stable")[:, :min(10, len(candidates))]
        rows.append(dict(language=language, n_portraits=len(candidates),
                         person_over_object_rate=float((positive > negative).mean()),
                         person_over_object_margin=float((positive - negative).mean()),
                         mean_query_score_std=float(scores[neutral_indices].std(axis=1).mean()),
                         unique_top10_images=int(len(np.unique(top10)))))
    return pd.DataFrame(rows)


def score(device="cpu", repetitions=500):
    from transformers import AutoModel, AutoProcessor
    from .embed import SiglipEmbedder, pick_device
    candidates, candidate_metadata = load_candidates(verify_images=True)
    prompts = validate_prompts(pd.read_csv(PROMPTS, keep_default_na=False))
    output("metadata.json").unlink(missing_ok=True)
    snapshot = cached_snapshot()
    processor = AutoProcessor.from_pretrained(snapshot, use_fast=False, local_files_only=True)
    tokens = processor.tokenizer(prompts.text.str.lower().tolist(), truncation=False, padding=False, add_special_tokens=True)
    prompts["tokens"] = [len(ids) for ids in tokens["input_ids"]]
    if (prompts.tokens > config.TEXT_MAX_LENGTH).any():
        raise ValueError("Audit prompts must fit the text window without truncation.")
    encoder = SiglipEmbedder(AutoModel.from_pretrained(snapshot, local_files_only=True), processor, pick_device(device))
    cache = output("image_embeddings.npy")
    cache_metadata = output("image_embedding_metadata.json")
    expected = {**model_settings(), "manifest_sha256": candidate_metadata["manifest_sha256"]}
    image_vectors = None
    if cache.exists() and cache_metadata.exists():
        previous = json.loads(cache_metadata.read_text())
        if previous == expected:
            image_vectors = np.load(cache, allow_pickle=False)
            if image_vectors.shape != (len(candidates), config.EMBEDDING_DIM) or not np.isfinite(image_vectors).all():
                raise ValueError("Invalid cached audit image embeddings.")
    if image_vectors is None:
        print(f"Encoding {len(candidates)} FairFace portraits on {device}.", flush=True)
        image_vectors = encoder.encode([config.ROOT / p for p in candidates.image_path], "image", config.EMBED_BATCH_SIZE_IMAGE)
        np.save(cache, image_vectors)
        write_metadata(cache_metadata, expected)
    print(f"Encoding {len(prompts)} audit prompts.", flush=True)
    text_vectors = encoder.captions(prompts.text.tolist())
    np.save(output("text_embeddings.npy"), text_vectors)
    similarities = text_vectors.astype(np.float32) @ image_vectors.astype(np.float32).T
    per_prompt, metrics = exposure_tables(similarities, prompts, candidates)
    print(f"Computing {repetitions} paired balanced-subsample repetitions.", flush=True)
    stability, subset_n = subsample_stability(similarities, prompts, candidates, repetitions)
    metrics = metrics.merge(stability, on=["language", "axis", "group", "k"], validate="one_to_one")
    prompts.to_parquet(output("queries.parquet"), index=False)
    per_prompt.to_parquet(output("per_prompt.parquet"), index=False)
    metrics.to_parquet(output("exposure.parquet"), index=False)
    sanity_table(similarities, prompts, candidates).to_parquet(output("sanity.parquet"), index=False)
    np.save(output("similarities.npy"), similarities)
    write_metadata(output("metadata.json"), {
        **expected, **candidate_metadata, "audit_version": VERSION,
        "prompts_sha256": digest(PROMPTS), "n_prompts": len(prompts),
        "n_neutral_families": prompts[prompts.kind == "neutral"].family.nunique(),
        "languages": list(prompts.language.unique()), "top_k": list(KS),
        "subsample_repetitions": repetitions, "subsample_candidates": subset_n,
        "subsample_fraction": 0.75,
        "n_unreviewed_translations": int(((prompts.language != "en") & (prompts.review_status != "human_reviewed")).sum()),
        "uncertainty": "Central 95% balanced-subset stability ranges, not population confidence intervals or significance tests.",
    })
    print(f"Audit complete: {len(candidates)} portraits, {len(prompts)} prompts, {len(metrics)} exposure summaries.", flush=True)


def load_results():
    metadata = read_metadata(output("metadata.json"), "-m implementation.bias_audit score", {"audit_version": VERSION})
    candidates, candidate_metadata = load_candidates()
    if (metadata["manifest_sha256"] != candidate_metadata["manifest_sha256"]
            or metadata["prompts_sha256"] != digest(PROMPTS)):
        raise ValueError("Audit candidates/prompts/review status changed. Rescore the audit.")
    queries = pd.read_parquet(output("queries.parquet"))
    matrix = np.load(output("similarities.npy"), allow_pickle=False)
    if (len(queries) != metadata["n_prompts"] or matrix.shape != (len(queries), len(candidates))
            or not np.isfinite(matrix).all()):
        raise ValueError("Audit results do not match their metadata.")
    return dict(metadata=metadata, candidates=candidates, queries=queries, similarities=matrix,
                exposure=pd.read_parquet(output("exposure.parquet")),
                per_prompt=pd.read_parquet(output("per_prompt.parquet")),
                sanity=pd.read_parquet(output("sanity.parquet")))


def top_candidates(results, language, family, k=10):
    query = results["queries"]
    selected = query[(query.language == language) & (query.family == family)]
    if len(selected) != 1: raise ValueError("Choose one available language and prompt family.")
    position = selected.index[0]
    similarities = results["similarities"][position]
    indices = np.argsort(-similarities, kind="stable")[:k]
    frame = results["candidates"].iloc[indices].copy()
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame["cosine"] = similarities[indices]
    return frame.reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("--per-stratum", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args(argv)
    if args.command == "prepare": prepare(args.per_stratum)
    else:
        if args.repetitions < 100: parser.error("Use at least 100 subsample repetitions.")
        score(args.device, args.repetitions)


if __name__ == "__main__": main()
