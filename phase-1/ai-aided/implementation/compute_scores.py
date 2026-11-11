"""
Step 3: margins, aggregated tables and the heatmap matrices.

Score per caption c (language L, true image i):
    margin(c) = cos(c, i) - mean_{j != i} cos(c, j)        j = all other images in the analysis set
Per (image, caption language): mean over all available captions.
Heatmap cell (caption language L, image locale K): mean over the images of locale K.

Three normalisation states of the matrix M (rows = caption language, columns = image locale):
    raw    : M
    row    : M - row mean                                   (removes the caption-language baseline)
    double : M - row mean - column mean + grand mean        (also removes the locale baseline)

Output: caption_scores.parquet, image_language.parquet, matrices.parquet
"""
import numpy as np
import pandas as pd

from . import config
from .artifacts import read_metadata, record_fingerprint, score_settings, write_metadata

CHUNK = 8192


def caption_margins(captions: pd.DataFrame, image_emb: np.ndarray, image_index: dict) -> pd.DataFrame:
    cap_emb = np.load(config.CAPTION_EMB, mmap_mode="r")
    img = image_emb.astype(np.float32)
    n_img = len(img)
    true_idx = captions["image_key"].map(image_index).to_numpy()
    rows = captions["caption_id"].to_numpy()  # row in caption_embeddings.npy

    cos_true = np.empty(len(captions), dtype=np.float32)
    mean_other = np.empty(len(captions), dtype=np.float32)
    for start in range(0, len(captions), CHUNK):
        sl = slice(start, start + CHUNK)
        sims = np.asarray(cap_emb[rows[sl]], dtype=np.float32) @ img.T     # (chunk, n_img)
        t = sims[np.arange(sims.shape[0]), true_idx[sl]]
        cos_true[sl] = t
        mean_other[sl] = (sims.sum(axis=1) - t) / (n_img - 1)
    return captions.assign(cosine=cos_true, mean_distractor=mean_other,
                           margin=cos_true - mean_other)


def build_matrices(image_lang: pd.DataFrame, langs: list[str]) -> pd.DataFrame:
    grouped = image_lang.groupby(["caption_lang", "locale"])
    M = grouped["margin"].mean().unstack().reindex(index=langs, columns=langs)
    n_images = grouped.size().unstack().reindex(index=langs, columns=langs)
    n_captions = grouped["n_captions"].sum().unstack().reindex(index=langs, columns=langs)

    row_mean = M.mean(axis=1)
    col_mean = M.mean(axis=0)
    grand = M.to_numpy().mean()
    states = {
        "raw": M,
        "row": M.sub(row_mean, axis=0),
        "double": M.sub(row_mean, axis=0).sub(col_mean, axis=1) + grand,
    }

    frames = []
    for state, mat in states.items():
        long = mat.stack().rename("value").reset_index()
        long.columns = ["caption_lang", "locale", "value"]
        # rank of each caption language within its column (1 = describes this locale best)
        long["rank_in_column"] = long.groupby("locale")["value"].rank(ascending=False, method="min").astype(int)
        long["state"] = state
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    counts = pd.DataFrame({"n_images": n_images.stack(), "n_captions": n_captions.stack()}).reset_index()
    counts.columns = ["caption_lang", "locale", "n_images", "n_captions"]
    return out.merge(counts, on=["caption_lang", "locale"])


def print_checks(scores: pd.DataFrame, image_lang: pd.DataFrame, matrices: pd.DataFrame) -> None:
    print("\nNegative-score check (raw values):")
    print(f"  captions with margin < 0        : {(scores['margin'] < 0).mean():.2%}")
    print(f"  (image, language) mean margin <0: {(image_lang['margin'] < 0).mean():.2%}")
    raw = matrices[matrices["state"] == "raw"]["value"]
    print(f"  heatmap cells (raw) < 0         : {(raw < 0).mean():.2%}")
    print(f"  captions with cosine < 0        : {(scores['cosine'] < 0).mean():.2%}")

    diag = matrices[(matrices["state"] == "row") & (matrices["caption_lang"] == matrices["locale"])]
    best = (diag["rank_in_column"] == 1).sum()
    print(f"\nMatched language is the best describer for {best}/{len(diag)} locales (row-centred).")


def main() -> None:
    metadata = read_metadata(config.EMBEDDING_METADATA, "-m implementation.embed")
    langs = config.active_languages()
    images = pd.read_parquet(config.IMAGES_TABLE)
    captions = pd.read_parquet(config.CAPTIONS_TABLE)
    image_emb = np.load(config.IMAGE_EMB, allow_pickle=False)
    cap_emb = np.load(config.CAPTION_EMB, mmap_mode="r", allow_pickle=False)
    if (image_emb.shape != (len(images), config.EMBEDDING_DIM)
            or cap_emb.shape != (len(captions), config.EMBEDDING_DIM)
            or metadata.get("n_images") != len(images)
            or metadata.get("n_captions") != len(captions)):
        raise ValueError("Embeddings do not match the SigLIP 2 dimensions or table sizes. "
                         "Run python -m implementation.prepare_data, then python -m implementation.embed.")
    fingerprint = record_fingerprint(
        images[["image_key", "locale"]].itertuples(index=False, name=None),
        captions[["image_key", "locale", "caption_lang", "caption_idx", "text"]]
        .itertuples(index=False, name=None),
    )
    if fingerprint != metadata.get("records_sha256"):
        raise ValueError("Table records or row order differ from the embedding run. Run python -m implementation.embed.")
    if not np.array_equal(captions["caption_id"].to_numpy(), np.arange(len(captions))):
        raise ValueError("Caption IDs no longer identify embedding rows. Run python -m implementation.prepare_data and python -m implementation.embed.")

    if config.DROP_IMAGES_FROM_DISTRACTORS:
        keep_img = images["locale"].isin(langs)
    else:
        keep_img = pd.Series(True, index=images.index)
    image_emb = image_emb[keep_img.to_numpy()]
    images = images[keep_img].reset_index(drop=True)
    image_index = {k: i for i, k in enumerate(images["image_key"])}
    if len(images) < 2:
        raise ValueError("Margins require at least two images in the analysis set.")

    captions = captions[captions["caption_lang"].isin(langs) & captions["image_key"].isin(image_index)]
    print(f"analysis set: {len(langs)} languages, {len(images)} images, {len(captions)} captions")

    config.SCORE_METADATA.unlink(missing_ok=True)
    scores = caption_margins(captions, image_emb, image_index)
    scores.to_parquet(config.CAPTION_SCORES, index=False)

    image_lang = (scores.groupby(["image_key", "locale", "caption_lang"])
                  .agg(cosine=("cosine", "mean"), margin=("margin", "mean"),
                       n_captions=("caption_id", "size"))
                  .reset_index())
    image_lang = image_lang[image_lang["locale"].isin(langs)]  # drop columns of unsupported locales
    image_lang.to_parquet(config.IMAGE_LANG_TABLE, index=False)

    matrices = build_matrices(image_lang, langs)
    matrices.to_parquet(config.MATRICES_FILE, index=False)
    print_checks(scores, image_lang, matrices)
    write_metadata(config.SCORE_METADATA, {**metadata, **score_settings()})
    print("\nwritten:", config.CAPTION_SCORES.name, config.IMAGE_LANG_TABLE.name, config.MATRICES_FILE.name)


if __name__ == "__main__":
    main()
