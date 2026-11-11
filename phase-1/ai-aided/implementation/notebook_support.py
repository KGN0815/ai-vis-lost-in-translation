"""Validated result loading and figure layout for Phase_I_Experiments.ipynb."""
import hashlib
from math import ceil
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .analyze import VERSION, analysis_path
from .artifacts import read_metadata, score_settings

COLORS = {"en": "#c57a37", "de": "#247b8b", "ja": "#7963a8", "te": "#b85e5e", "fr": "#5f8560"}
NEUTRAL = "#bac9d0"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 115, "savefig.dpi": 150, "font.size": 10,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c1cdd3", "axes.labelcolor": "#234252",
        "text.color": "#234252", "xtick.color": "#526874", "ytick.color": "#526874",
        "grid.color": "#e8edf0", "figure.facecolor": "white", "axes.facecolor": "white",
        "figure.constrained_layout.use": True,
    })


def show_figure(figure):
    from IPython.display import display
    display(figure)
    plt.close(figure)


def load_results():
    """Read completed output tables and verify their common run provenance."""
    score = read_metadata(config.SCORE_METADATA, "-m implementation.compute_scores", score_settings())
    analysis = read_metadata(analysis_path("analysis_metadata.json"), "-m implementation.analyze",
                             {**score_settings(), "analysis_version": VERSION})
    if score["records_sha256"] != analysis["records_sha256"]:
        raise ValueError("Analysis and scores identify different input records. Regenerate the analysis.")
    tables = {
        "queries": "retrieval.parquet",
        "effects": "matching_effects.parquet", "features": "caption_features.parquet",
        "matrices": "matrices.parquet", "images": "images.parquet",
        "translation_pairs": "translation_pairs.parquet", "translation_effects": "translation_effects.parquet",
    }
    data = {key: pd.read_parquet(analysis_path(filename)) for key, filename in tables.items()}
    if len(data["queries"]) != analysis["n_retrieval_captions"] or len(data["images"]) != score["n_images"]:
        raise ValueError("Output counts differ from their metadata.")
    translation = read_metadata(analysis_path("translation_metadata.json"), "-m implementation.translation_control score", score_settings())
    if translation["records_sha256"] != score["records_sha256"]:
        raise ValueError("Translation pilot belongs to a different run.")
    csv_path = Path(translation["input_file"])
    if not csv_path.is_absolute(): csv_path = config.ROOT / csv_path
    if not csv_path.exists() or hashlib.sha256(csv_path.read_bytes()).hexdigest() != translation["input_sha256"]:
        raise ValueError("Pilot translation/review data changed. Rescore before using these results.")
    data.update(score_metadata=score, analysis_metadata=analysis, translation_metadata=translation)
    return data


def plot_language_retrieval(summary):
    values = summary.sort_values("Recall@1 (%)")
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = [COLORS.get(code, NEUTRAL) for code in values.index]
    bars = ax.barh(values.index.map(config.LANGUAGES), values["Recall@1 (%)"], color=colors)
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
    ax.set(xlim=(0, 100), xlabel="Image-balanced Recall@1 (%)", title="Retrieval differs strongly between caption languages")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    return fig


def plot_heatmap(matrix):
    order = [code for _, code in config.ordered_languages("family") if code in matrix.index]
    values = matrix.reindex(index=order, columns=order).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 9))
    limit = np.abs(values).max()
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="equal")
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(order)), [f"{config.LANGUAGES[c]} ({c})" for c in order], fontsize=8)
    for i in range(len(order)):
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor="#233d4b", linewidth=.7))
    ax.set(xlabel="Image locale (sampling label)", ylabel="Caption language",
           title="Language × locale: double-centered similarity margins")
    fig.colorbar(im, ax=ax, shrink=.75, label="Margin residual after removing language and locale means")
    return fig


def plot_matching_effects(raw, centered):
    raw = raw.sort_values("effect")
    centered = centered.set_index("caption_lang").loc[raw.caption_lang]
    fig, ax = plt.subplots(figsize=(10, 9))
    positions = np.arange(len(raw))
    for frame, y, name, color in [
        (raw, positions - .15, "Unadjusted", "#c57a37"),
        (centered, positions + .15, "Image-centered", "#247b8b"),
    ]:
        point = frame.effect.to_numpy() * 100
        errors = np.maximum(0, np.vstack([frame.effect - frame.ci_low, frame.ci_high - frame.effect]) * 100)
        ax.errorbar(point, y, xerr=errors, fmt="o", markersize=4, capsize=2,
                    color=color, alpha=.9, linewidth=1, label=name)
    ax.axvline(0, color="#71858f", linestyle="--", linewidth=1)
    ax.set_yticks(positions, raw.caption_lang.map(config.LANGUAGES), fontsize=8)
    ax.set(xlabel="Matched-locale minus other-locale Recall@1 (percentage points)",
           title="Matching advantage: image-level estimates and pointwise 95% intervals")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    return fig


def plot_margin_vs_retrieval(comparison):
    fig, ax = plt.subplots(figsize=(8, 5))
    for code, row in comparison.iterrows():
        ax.scatter(row["Overall Recall@1 (%)"], row["Matching margin difference"], s=45,
                   color=COLORS.get(code, NEUTRAL), zorder=3)
        if code in COLORS:
            ax.annotate(config.LANGUAGES[code], (row["Overall Recall@1 (%)"], row["Matching margin difference"]),
                        xytext=(5, 7 if code != "fr" else -15), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#94a5ae", linestyle="--", linewidth=1)
    ax.set(xlabel="Overall image-balanced Recall@1 (%)", ylabel="Matched − other mean-distractor margin",
           title="A large local margin advantage need not mean reliable retrieval")
    ax.grid(alpha=.6)
    return fig


def plot_translation_pairs(pairs):
    fig, ax = plt.subplots(figsize=(7, 6))
    for locale, group in pairs.groupby("locale"):
        ax.scatter(group.source_margin, group.translated_margin, color=COLORS.get(locale, NEUTRAL),
                   label=f"{config.LANGUAGES[locale]} locale (n={len(group)})", alpha=.8, s=35)
    low = min(pairs.source_margin.min(), pairs.translated_margin.min()) - .01
    high = max(pairs.source_margin.max(), pairs.translated_margin.max()) + .01
    ax.plot([low, high], [low, high], color="#94a5ae", linestyle="--", linewidth=1)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Original English: margin", ylabel="Translated German: margin",
           title="Same intended content, different caption language")
    ax.set_aspect("equal", adjustable="box"); ax.legend(frameon=False); ax.grid(alpha=.6)
    return fig


def plot_translation_effects(effects):
    frame = effects[effects.metric == "margin"]
    fig, ax = plt.subplots(figsize=(9, 4))
    locales = [c for c in ["all", "de", "en"] if c in set(frame.locale)]
    for name, offset, color, label in [
        ("translation_minus_source", -.12, "#c57a37", "Translated German − English source"),
        ("human_target_minus_translation", .12, "#247b8b", "Independent German − translated German"),
    ]:
        group = frame[frame.comparison == name].set_index("locale").loc[locales]
        errors = np.maximum(0, np.vstack([group.effect - group.ci_low, group.ci_high - group.effect]))
        ax.errorbar(group.effect, np.arange(len(locales)) + offset, xerr=errors, fmt="o", color=color,
                    capsize=3, label=label, markersize=6)
    ax.axvline(0, color="#94a5ae", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(locales)), ["Both locales" if c == "all" else config.LANGUAGES[c] + " locale" for c in locales])
    ax.set(xlabel="Paired margin difference with pointwise 95% bootstrap interval",
           title="Translation control: language change versus description change")
    ax.invert_yaxis(); ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), frameon=False, fontsize=9)
    return fig


def image_gallery(keys, titles, columns=3):
    rows = ceil(len(keys) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False)
    for ax in axes.flat: ax.set_axis_off()
    for ax, key, title in zip(axes.flat, keys, titles):
        path = config.IMAGES_DIR / f"{key}.jpg"
        if path.exists():
            with Image.open(path) as photo: ax.imshow(np.asarray(photo.convert("RGB")))
        else:
            ax.text(.5, .5, "Image unavailable locally", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(textwrap.fill(title, 48) + f"\n{key}", fontsize=9, fontweight="normal")
    return fig


def retrieval_gallery(caption_id, results):
    queries = results["queries"]
    positions = np.flatnonzero(queries.caption_id.to_numpy() == caption_id)
    if len(positions) != 1: raise ValueError("Choose one valid original caption ID.")
    position = int(positions[0]); query = queries.iloc[position]
    indices = np.load(analysis_path("retrieval_indices.npy"), mmap_mode="r", allow_pickle=False)[position]
    cosines = np.load(analysis_path("retrieval_cosines.npy"), mmap_mode="r", allow_pickle=False)[position]
    keys = [query.image_key, *results["images"].iloc[indices].image_key.tolist()]
    titles = [f"Intended image · rank {query['rank']}"]
    titles.extend(f"Retrieved #{i}: cosine {value:.4f}" for i, value in enumerate(cosines, 1))
    return image_gallery(keys, titles)


def plot_cue_summary(summary, language):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh(summary.index, summary["Recall@1 (%)"], color="#247b8b")
    axes[0].set(xlim=(0, 100), xlabel="Image-balanced Recall@1 (%)", title="Retrieval by caption cue")
    axes[1].barh(summary.index, summary["Matching margin difference"], color="#c57a37")
    axes[1].axvline(0, color="#94a5ae", linewidth=1)
    axes[1].set(xlabel="Matched − other image-centered margin", title="Descriptive matching differences")
    for ax in axes: ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.suptitle(f"{config.LANGUAGES[language]} · English-caption keyword cues (overlapping groups)", fontsize=13)
    return fig


def plot_truncation(rates):
    frame = rates.sort_values("Truncated (%)").tail(12)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(frame.index.map(config.LANGUAGES), frame["Truncated (%)"], color="#247b8b")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    ax.set(xlabel="Captions longer than 64 tokens (%)", title="Truncation is concentrated in a few languages")
    ax.set_xlim(0, max(frame["Truncated (%)"].max() * 1.2, 1))
    ax.grid(axis="x"); ax.set_axisbelow(True)
    return fig
