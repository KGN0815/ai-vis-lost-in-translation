"""Notebook figures for the controlled FairFace language audit."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .bias_audit import RACES, GENDERS, AGES, top_candidates

GROUPS = {"race": RACES, "gender": GENDERS, "age": AGES}
AXIS_LABELS = {"race": "FairFace race annotation", "gender": "FairFace gender annotation", "age": "FairFace age band"}
LANG_COLORS = {"en": "#c57a37", "de": "#247b8b", "sw": "#7963a8"}


def exposure_frame(results, axis="race", k=10):
    return results["exposure"].query("axis == @axis and k == @k").copy()


def plot_exposure(results, axis="race", k=10, reference="pool"):
    frame = exposure_frame(results, axis, k)
    metric = "excess_pp" if reference == "pool" else "delta_english_pp"
    languages = results["metadata"]["languages"]
    matrix = frame.pivot(index="language", columns="group", values=metric).reindex(index=languages, columns=GROUPS[axis])
    values = matrix.to_numpy()
    limit = max(5., float(np.abs(values).max()))
    fig, ax = plt.subplots(figsize=(11, 5.6), layout="constrained")
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(len(languages)):
        for column in range(len(matrix.columns)):
            ax.text(column, row, f"{values[row,column]:+.1f}", ha="center", va="center",
                    color="white" if abs(values[row,column]) > .65 * limit else "#203d4c", fontsize=10)
    ax.set_xticks(range(len(matrix.columns)), [textwrap.fill(str(x).replace("_", " "), 14) for x in matrix.columns])
    ax.set_yticks(range(len(languages)), [config.LANGUAGES[c] for c in languages])
    ax.set_xlabel(AXIS_LABELS[axis])
    title = (f"Who appears in the top {k} compared with all portraits?" if reference == "pool"
             else f"Top-{k} representation compared with the paired English queries")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=.85, label="Difference in percentage points")
    fig.suptitle(f"Same set of {len(results['candidates'])} portraits; average over {results['metadata']['n_neutral_families']} neutral prompt families", fontsize=11)
    return fig


def plot_stability(results, axis="race", k=10, languages=("en", "de", "sw")):
    frame = exposure_frame(results, axis, k)
    groups = GROUPS[axis]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for offset, language in zip(np.linspace(-.23, .23, len(languages)), languages):
        group = frame[frame.language == language].set_index("group").reindex(groups)
        positions = np.arange(len(groups)) + offset
        color = LANG_COLORS.get(language, "#526874")
        ax.hlines(positions, group.stability_low, group.stability_high, color=color, linewidth=2, alpha=.6)
        ax.scatter(group.excess_pp, positions, color=color, s=30, label=config.LANGUAGES[language], zorder=3)
    ax.axvline(0, color="#94a5ae", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(groups)), [x.replace("_", " ") for x in groups])
    ax.invert_yaxis()
    ax.set(xlabel=f"Top-{k} share minus candidate-pool share (percentage points)",
           title="Does overrepresentation persist across balanced candidate subsets?")
    ax.grid(axis="x", alpha=.6); ax.legend(frameon=False)
    fig.suptitle(f"Dots: full pool. Lines: central 95% across {results['metadata']['subsample_repetitions']} subsets of "
                 f"{results['metadata']['subsample_candidates']} portraits; not population confidence intervals.", fontsize=9)
    return fig


def top_one_table(results, family="person_photo"):
    rows = []
    for language in results["metadata"]["languages"]:
        candidate = top_candidates(results, language, family, 1).iloc[0]
        prompt = results["queries"].query("language == @language and family == @family").iloc[0]
        rows.append({"Language": config.LANGUAGES[language], "Query": prompt.text,
                     "Dataset race label": candidate.race, "Dataset gender label": candidate.gender,
                     "Dataset age band": candidate.age, "Image ID": candidate.image_key,
                     "Cosine": candidate.cosine})
    return pd.DataFrame(rows).set_index("Language")


def portrait_gallery(results, family="person_photo", languages=("en", "de", "sw"), k=3):
    fig, axes = plt.subplots(k, len(languages), figsize=(3.8 * len(languages), 3.4 * k), squeeze=False, layout="constrained")
    for column, language in enumerate(languages):
        ranked = top_candidates(results, language, family, k)
        for row, candidate in enumerate(ranked.itertuples()):
            ax = axes[row, column]
            path = config.ROOT / candidate.image_path
            if path.exists():
                with Image.open(path) as photo: ax.imshow(np.asarray(photo.convert("RGB")))
            else:
                ax.text(.5, .5, "Download the candidate images to view this portrait", ha="center", wrap=True)
            ax.set_axis_off()
            ax.set_title(f"{config.LANGUAGES[language]} · rank {candidate.rank}\n"
                         f"FairFace: {candidate.race.replace('_', ' ')} / {candidate.gender} / {candidate.age}\n"
                         f"{candidate.image_key}", fontsize=9, fontweight="normal")
    fig.suptitle(f"Same prompt family: {family} · annotations supplied by FairFace, not model predictions", fontsize=12)
    return fig


def plot_sanity(results):
    frame = results["sanity"].set_index("language").loc[results["metadata"]["languages"]]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    bars = ax.barh(frame.index.map(config.LANGUAGES), 100 * frame.person_over_object_rate, color="#247b8b")
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set(xlim=(0, 108), xlabel="Portraits scoring above all object alternatives (%)",
           title="Sanity check: mean person-query score versus strongest object query")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    return fig
