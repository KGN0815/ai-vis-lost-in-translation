"""Linked Phase I views for output-based investigation and local human review."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, dash_table

from . import config
from .analyze import VERSION, TOPICS, analysis_path
from .artifacts import read_metadata, score_settings

REVIEWS = config.ROOT / "annotations" / "error_reviews.csv"
ERROR_TYPES = {
    "plausible_alternative": "Plausible alternative image",
    "same_category_wrong_instance": "Same category, different instance",
    "wrong_attributes": "Wrong attributes / relationship",
    "background_or_context": "Similar background / context",
    "unrelated": "Unrelated content",
    "correct": "Correct retrieval",
    "unclear": "Unclear",
}
METRICS = {"margin": "Mean-distractor margin", "recall_at_1": "Recall@1"}
COLORS = {"en": "#cc7b38", "de": "#26798b"}
_DATA = None


def get_data():
    global _DATA
    if _DATA is None:
        metadata = read_metadata(analysis_path("analysis_metadata.json"), "-m implementation.analyze",
                                 {**score_settings(), "analysis_version": VERSION})
        score = read_metadata(config.SCORE_METADATA, "-m implementation.compute_scores", score_settings())
        if metadata["records_sha256"] != score["records_sha256"]:
            raise RuntimeError("Analysis belongs to a different embedding run. Run python -m implementation.analyze.")
        queries = pd.read_parquet(analysis_path("retrieval.parquet"))
        features = pd.read_parquet(analysis_path("caption_features.parquet"))
        queries = queries.merge(features.drop(columns=["image_key", "caption_lang"]), on="caption_id", validate="one_to_one")
        images = pd.read_parquet(config.IMAGES_TABLE)
        english = queries[queries.caption_lang == "en"].groupby("image_key").text.first().to_dict()
        _DATA = dict(metadata=metadata, queries=queries, images=images, english=english,
                     neighbors=np.load(analysis_path("retrieval_indices.npy"), mmap_mode="r"),
                     cosines=np.load(analysis_path("retrieval_cosines.npy"), mmap_mode="r"),
                     effects=pd.read_parquet(analysis_path("matching_effects.parquet")))
        _DATA["image_baseline"] = queries.groupby(["image_key", "caption_lang"]).margin.mean().groupby("image_key").mean()
    return _DATA


def empty_figure(message):
    return go.Figure().update_layout(template="plotly_white", height=300,
        annotations=[dict(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)])


def style_figure(fig, title=None, height=440):
    fig.update_layout(template="plotly_white", title=title, height=height,
                      font=dict(family="Arial, sans-serif", size=12),
                      margin=dict(l=90, r=30, t=55, b=55), paper_bgcolor="white")
    return fig


def control(title, component):
    return html.Label([html.Span(title, className="control-label"), component], className="control")


def section(title, description, children):
    return html.Section([html.H3(title), html.P(description, className="muted"), *children], className="analysis-panel")


def layout():
    from .bias_dashboard import audit_tab
    try:
        data = get_data()
    except (RuntimeError, FileNotFoundError) as exc:
        return html.Div(f"Extended analysis is not ready: {exc}", className="notice")
    langs = [{"label": name, "value": code} for code, name in config.LANGUAGES.items() if code in config.active_languages()]
    query = data["queries"]
    overall = query.groupby(["image_key", "caption_lang"]).recall_at_1.mean().mean()
    return html.Div([
        html.Div([
            html.Div([html.Strong(f"{len(data['images']):,}"), html.Span("images")]),
            html.Div([html.Strong(f"{len(query):,}"), html.Span("caption queries")]),
            html.Div([html.Strong(f"{overall:.1%}"), html.Span("image-balanced multilingual Recall@1")]),
        ], className="summary-strip"),
        html.P("Investigate a pattern, inspect the retrieved images, then test an explanation. "
               "Locale is a sampling label. Similarity and retrieval differences alone do not establish cultural bias.", className="notice"),
        dcc.Tabs(id="investigation-tab", value="retrieval", children=[
            dcc.Tab(label="Matching effects", value="matching", children=[section(
                "Does matching language and locale help?",
                "Captions are averaged within each image first. Intervals resample whole images within locales. "
                "Image-centering subtracts each image’s mean across the included languages to account for general image difficulty.", [
                    html.Div([
                        control("Outcome", dcc.Dropdown(id="effect-metric", options=[{"label": v, "value": k} for k, v in METRICS.items()], value="margin", clearable=False)),
                        control("Comparison", dcc.Dropdown(id="effect-view", options=[{"label": "Image-centered", "value": "image_centered"}, {"label": "Unadjusted", "value": "raw"}], value="image_centered", clearable=False)),
                        control("Sensitivity", dcc.Checklist(id="omit-hi", options=[{"label": "Omit Hindi locale and language", "value": "hi"}], value=[])),
                    ], className="control-row"),
                    html.Div(id="effect-summary", className="notice"), dcc.Graph(id="effect-plot"),
                    html.P("Permutation tests shuffle one common locale-label vector across images, preserving locale counts. "
                           "p-values assume exchangeability. They do not establish a causal language effect. q-values use Benjamini–Hochberg "
                           "within each displayed set of languages. Bootstrap intervals are pointwise 95% intervals.", className="muted"),
                ])]),
            dcc.Tab(label="Retrieval errors", value="retrieval", children=[section(
                "What does the model retrieve instead?",
                "Filter a cohort, select a caption, and compare its intended image with the top five candidates. "
                "A different image can still be a valid interpretation of a generic caption. Review the content before calling it a semantic error.", [
                    html.Div([
                        control("Caption language", dcc.Dropdown(id="query-language", options=langs, value="de", clearable=False)),
                        control("Image locale", dcc.Dropdown(id="query-locale", options=[{"label": "All locales", "value": "all"}, *langs], value="all", clearable=False)),
                        control("English caption cue", dcc.Dropdown(id="query-topic", options=[{"label": "All captions", "value": "all"}, *[{"label": k, "value": k} for k in TOPICS]], value="all", clearable=False)),
                        control("Result", dcc.Dropdown(id="query-result", options=[{"label": "Wrong image at rank 1", "value": "errors"}, {"label": "Correct image at rank 1", "value": "correct"}, {"label": "All queries", "value": "all"}], value="errors", clearable=False)),
                        control("Order", dcc.Dropdown(id="query-sort", options=[{"label": "Largest wrong-image lead", "value": "hard_margin"}, {"label": "Worst true-image rank", "value": "rank"}, {"label": "Reproducible random sample", "value": "sample"}], value="hard_margin", clearable=False)),
                    ], className="control-row"),
                    html.Div(id="query-summary", className="notice"),
                    dash_table.DataTable(id="query-table", columns=[
                        {"name": "ID", "id": "caption_id"}, {"name": "Caption", "id": "text"},
                        {"name": "Locale", "id": "locale"}, {"name": "True-image rank", "id": "rank"},
                        {"name": "True − best wrong", "id": "hard_margin"},
                    ], data=[], row_selectable="single", selected_rows=[0], page_size=8,
                        style_table={"overflowX": "auto"}, style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto", "padding": "10px", "maxWidth": "650px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#edf3f5"}),
                    html.Div(id="retrieval-detail"),
                    html.Details([html.Summary("Record a human interpretation"),
                        html.Div([
                            control("Error type", dcc.Dropdown(id="review-type", options=[{"label": v, "value": k} for k, v in ERROR_TYPES.items()], value="unclear", clearable=False)),
                            control("Visible text in intended image", dcc.Dropdown(id="review-text", options=["yes", "no", "unclear"], value="unclear", clearable=False)),
                            control("Reviewer", dcc.Input(id="reviewer", placeholder="Your name", type="text")),
                            control("Notes", dcc.Textarea(id="review-notes", placeholder="Evidence for the interpretation")),
                        ], className="control-row"),
                        html.Button("Save review locally", id="save-review", n_clicks=0), html.Span(id="review-status"),
                    ]),
                    html.Button("Download balanced review template", id="download-review", n_clicks=0), dcc.Download(id="review-download"),
                    html.Div(id="review-summary", children=review_summary()),
                ])]),
            dcc.Tab(label="Caption cues", value="cues", children=[section(
                "Which inputs are associated with the behavior?",
                "Topic cues are transparent keyword matches in the original English captions, applied to all languages of the same image. "
                "They can overlap and are not verified image labels. Length diagnostics compare captions within a selected language.", [
                    control("Caption language", dcc.Dropdown(id="cue-language", options=langs, value="de", clearable=False)),
                    dcc.Graph(id="cue-plot"), dcc.Graph(id="cue-matching"),
                    html.P("Cue-specific matching effects use image-centered margins. Pointwise 95% intervals resample images "
                           "within matched/other cohorts (1,000 draws). Cues with fewer than three images in either cohort are omitted. "
                           "These overlapping, exploratory comparisons have no multiple-testing correction.", className="muted"),
                    dcc.Graph(id="length-plot"), html.Div(id="token-summary", className="notice"),
                ])]),
            dcc.Tab(label="Translation control", value="translation", children=[section(
                "Language or descriptive content? English → German pilot",
                "72 images were sampled before looking at scores: 36 English-locale and 36 German-locale images. "
                "Each image has its first original English caption, an AI-generated German translation, and independently written German captions. "
                "The same image candidates and preprocessing are used throughout.", [
                    html.Div(id="translation-status", className="notice"),
                    control("Translation review", dcc.RadioItems(id="translation-review", options=[{"label": "All non-rejected translations", "value": "all"}, {"label": "Human-reviewed only", "value": "reviewed"}], value="all", inline=True)),
                    dcc.Graph(id="translation-scatter"), dcc.Graph(id="translation-effects"), html.Div(id="translation-detail"),
                    html.P("Translated German − source English approximately controls content. Independent German − translated German "
                           "holds language fixed while changing content and writing style. Neither difference is a pure causal effect. "
                           "Intervals resample image pairs within locales. This pilot covers two languages and does not generalize to all 36.", className="muted"),
                ])]),
            audit_tab(),
        ]),
    ], className="investigation")


def effect_figure(metric, view, omit_hi):
    data = get_data()
    subset = "without_hi" if omit_hi else "all"
    df = data["effects"].query("metric == @metric and view == @view and subset == @subset").sort_values("effect")
    scale = 100 if metric == "recall_at_1" else 1
    fig = go.Figure(go.Scatter(x=df.effect * scale, y=df.caption_lang.map(config.LANGUAGES), mode="markers",
        marker=dict(size=8, color="#26798b"), error_x=dict(type="data", symmetric=False,
            array=(df.ci_high - df.effect) * scale, arrayminus=(df.effect - df.ci_low) * scale),
        customdata=df[["p", "q_bh", "n_matched", "n_other"]],
        hovertemplate="%{y}<br>Effect: %{x:.4f}<br>p=%{customdata[0]:.4f}<br>q=%{customdata[1]:.4f}<br>Images: %{customdata[2]} matched / %{customdata[3]} other<extra></extra>"))
    fig.add_vline(x=0, line_color="#999", line_dash="dot")
    fig.update_xaxes(title="Matched − other (percentage points)" if scale == 100 else "Matched − other margin")
    style_figure(fig, "Matching advantage with 95% bootstrap intervals", height=850)
    global_test = data["metadata"]["global_tests"][f"{subset}/{metric}"][view]
    summary = (f"Mean language effect: {global_test['effect'] * scale:+.4f}{' pp' if scale == 100 else ''}. "
               f"Global two-sided permutation p = {global_test['p']:.4f}. "
               f"{data['metadata']['bootstrap']:,} bootstrap samples and {data['metadata']['permutations']:,} permutations. "
               "Hindi images were sampled globally. Compare the sensitivity view.")
    return fig, summary


def filtered_queries(lang, locale, topic, result, sort):
    df = get_data()["queries"]
    mask = df.caption_lang == lang
    if locale != "all": mask &= df.locale == locale
    if topic != "all": mask &= df[f"topic_{topic}"]
    cohort = df.loc[mask]
    baseline = cohort.groupby("image_key").recall_at_1.mean().mean()
    if result == "errors": mask &= df["rank"] > 1
    if result == "correct": mask &= df["rank"] == 1
    filtered = df.loc[mask]
    if sort == "sample":
        shown = filtered.sample(n=min(60, len(filtered)), random_state=20261111)
    else:
        shown = filtered.sort_values([sort, "caption_id"], ascending=[sort != "rank", True]).head(60)
    columns = ["caption_id", "text", "locale", "rank", "hard_margin"]
    records = shown[columns].copy()
    records["hard_margin"] = records.hard_margin.map(lambda value: f"{value:+.5f}")
    summary = (f"{len(filtered):,} captions from {filtered.image_key.nunique():,} images in this filtered cohort. "
               f"Before filtering by success/failure: {len(cohort):,} captions, image-balanced Recall@1 {baseline:.1%}. "
               "Showing up to 60 queries.") if len(filtered) else "No captions match these filters."
    return records.to_dict("records"), summary, [0] if len(records) else []


def picture(key, title, caption=""):
    return html.Div([html.Strong(title), html.Img(src=f"/img/{key}", alt=title),
                     html.Small(key), html.P(caption)], className="candidate-card")


def retrieval_detail(records, selected):
    if not records or not selected or selected[0] >= len(records):
        return html.P("Select a caption to inspect its retrievals.")
    data = get_data()
    caption_id = int(records[selected[0]]["caption_id"])
    positions = np.flatnonzero(data["queries"].caption_id.to_numpy() == caption_id)
    position = positions[0]
    row = data["queries"].iloc[position]
    cards = [picture(row.image_key, f"Intended image · rank {row['rank']}", data["english"].get(row.image_key, ""))]
    for k, (index, cosine) in enumerate(zip(data["neighbors"][position], data["cosines"][position]), 1):
        image = data["images"].iloc[int(index)]
        title = f"#{k} · cosine {cosine:.4f}" + (" · intended image" if image.image_key == row.image_key else "")
        cards.append(picture(image.image_key, title, data["english"].get(image.image_key, "")))
    same = data["queries"][data["queries"].image_key == row.image_key]
    means = same.groupby("caption_lang")[["margin", "recall_at_1"]].mean().sort_values("margin")
    fig = go.Figure(go.Bar(x=means.margin, y=means.index.map(config.LANGUAGES), orientation="h",
        marker_color=[COLORS.get(c, "#c4d0d5") for c in means.index],
        customdata=means.recall_at_1, hovertemplate="%{y}<br>Margin %{x:.4f}<br>Recall@1 %{customdata:.0%}<extra></extra>"))
    style_figure(fig, "Same image, all caption languages (independent descriptions)", height=700)
    selected_languages = list(dict.fromkeys([row.caption_lang, "en", row.locale]))
    descriptions = [html.Div([html.B(config.LANGUAGES.get(lang, lang)), html.Ul([
        html.Li(f"{r.text} · rank {r.rank}") for r in same[same.caption_lang == lang].itertuples()])])
        for lang in selected_languages]
    return html.Div([html.H4(f"Caption {caption_id}: {row.text}"), html.Div(cards, className="candidate-grid"),
                     html.Div(descriptions, className="caption-comparison"), dcc.Graph(figure=fig)])


def cue_figures(lang):
    df = get_data()["queries"].query("caption_lang == @lang")
    rows = []
    for topic in TOPICS:
        group = df[df[f"topic_{topic}"]]
        balanced = group.groupby("image_key").recall_at_1.mean()
        if len(balanced): rows.append((topic, balanced.mean() * 100, len(balanced)))
    cue = pd.DataFrame(rows, columns=["topic", "recall", "n"])
    fig = go.Figure(go.Bar(x=cue.topic, y=cue.recall, text=cue.n.map(lambda n: f"n={n}"),
                           marker_color="#26798b", hovertemplate="%{x}<br>Recall@1 %{y:.1f}%<br>%{text}<extra></extra>"))
    fig.update_yaxes(title="Image-balanced Recall@1 (%)", range=[0, 100])
    style_figure(fig, f"{config.LANGUAGES[lang]}: retrieval by English-caption cue (overlapping cohorts)")
    if "tokens" not in df:
        return fig, empty_figure("Run python -m implementation.analyze with the cached tokenizer to compute token lengths."), "Token counts unavailable."
    bins = pd.cut(df.tokens, [0, 16, 32, 48, 64, np.inf], labels=["1–16", "17–32", "33–48", "49–64", ">64"])
    grouped = df.assign(length_bin=bins).groupby(["length_bin", "image_key"], observed=True).recall_at_1.mean()
    summary = grouped.groupby(level=0, observed=True).agg(["mean", "size"])
    length = go.Figure(go.Bar(x=summary.index.astype(str), y=summary["mean"] * 100,
                             text=summary["size"].map(lambda n: f"n={n}"), marker_color="#cc7b38"))
    length.update_xaxes(title="Tokens before truncation (including special tokens)")
    length.update_yaxes(title="Image-balanced Recall@1 within bin (%)", range=[0, 100])
    style_figure(length, "Does retrieval vary with caption length?")
    return fig, length, (f"{df.truncated.mean():.2%} of {config.LANGUAGES[lang]} captions exceed {config.TEXT_MAX_LENGTH} tokens. "
                          "Length is correlated with content. These descriptive comparisons do not isolate a truncation effect. "
                          "An image may occur in multiple length bins when its captions differ in length.")


def cue_matching_figure(lang):
    data = get_data()
    queries = data["queries"]
    baseline = data.get("image_baseline")
    if baseline is None:
        baseline = queries.groupby(["image_key", "caption_lang"]).margin.mean().groupby("image_key").mean()
    df = queries[queries.caption_lang == lang]
    aggregation = {"margin": "mean", "locale": "first", **{f"topic_{topic}": "first" for topic in TOPICS}}
    images = df.groupby("image_key").agg(aggregation)
    images["adjusted"] = images.margin - baseline.reindex(images.index)
    rng = np.random.default_rng(20261111)
    rows = []
    for topic in TOPICS:
        group = images[images[f"topic_{topic}"]]
        matched = group[group.locale == lang].adjusted.to_numpy()
        other = group[group.locale != lang].adjusted.to_numpy()
        if min(len(matched), len(other)) < 3: continue
        differences = matched[rng.integers(0, len(matched), (1000, len(matched)))].mean(axis=1)
        differences -= other[rng.integers(0, len(other), (1000, len(other)))].mean(axis=1)
        low, high = np.quantile(differences, [.025, .975])
        rows.append(dict(topic=topic, effect=matched.mean() - other.mean(), low=low, high=high,
                         matched=len(matched), other=len(other)))
    if not rows: return empty_figure("Too few matched-locale images in these cue cohorts.")
    frame = pd.DataFrame(rows).sort_values("effect")
    fig = go.Figure(go.Scatter(x=frame.effect, y=frame.topic, mode="markers", marker=dict(size=9, color="#26798b"),
        error_x=dict(type="data", symmetric=False, array=np.maximum(0, frame.high - frame.effect), arrayminus=np.maximum(0, frame.effect - frame.low)),
        customdata=frame[["matched", "other"]], hovertemplate="%{y}<br>Matched − other: %{x:.4f}<br>Images: %{customdata[0]} matched / %{customdata[1]} other<extra></extra>"))
    fig.add_vline(x=0, line_color="#999", line_dash="dot")
    fig.update_xaxes(title="Matched − other image-centered margin")
    return style_figure(fig, "Which caption cues accompany a matching advantage?")


def translation_data():
    metadata = read_metadata(analysis_path("translation_metadata.json"), "-m implementation.translation_control score", score_settings())
    if metadata["records_sha256"] != get_data()["metadata"]["records_sha256"]:
        raise RuntimeError("Translation results belong to a different embedding run.")
    source = Path(metadata["input_file"])
    if not source.is_absolute(): source = config.ROOT / source
    if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != metadata["input_sha256"]:
        raise RuntimeError("Translations/review status changed. Run python -m implementation.translation_control score.")
    return metadata, pd.read_parquet(analysis_path("translation_pairs.parquet"))


def translation_figures(review):
    try:
        _, pairs = translation_data()
    except (RuntimeError, FileNotFoundError) as exc:
        empty = empty_figure("Translation results are not ready.")
        return empty, empty, str(exc)
    if review == "reviewed": pairs = pairs[pairs.review_status == "human_reviewed"]
    if pairs.empty:
        empty = empty_figure("No independently reviewed translations yet.")
        return empty, empty, "The pilot is AI-generated. Review translations in annotations/translations_en_de.csv and rerun scoring."
    scatter = go.Figure()
    for locale, frame in pairs.groupby("locale"):
        scatter.add_trace(go.Scatter(x=frame.source_margin, y=frame.translated_margin, mode="markers",
            name=f"{config.LANGUAGES[locale]} locale", customdata=frame[["pair_id", "source_text", "translation"]],
            marker=dict(color=COLORS.get(locale, "#888"), size=9),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<br>English %{x:.4f}<br>Translated German %{y:.4f}<extra></extra>"))
    low = min(pairs.source_margin.min(), pairs.translated_margin.min())
    high = max(pairs.source_margin.max(), pairs.translated_margin.max())
    scatter.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", line=dict(color="#999", dash="dot"), hoverinfo="skip", showlegend=False))
    scatter.update_xaxes(title="Original English margin")
    scatter.update_yaxes(title="Equivalent German translation margin")
    style_figure(scatter, "Same intended content, two languages · select a point")
    from .translation_control import paired_summary
    summary = paired_summary(pairs)
    summary = summary[summary.metric == "margin"]
    effect = go.Figure()
    labels = {"translation_minus_source": "Translated German − English source", "human_target_minus_translation": "Independent German − translated German"}
    for comparison, frame in summary.groupby("comparison"):
        effect.add_trace(go.Scatter(x=frame.effect, y=frame.locale.map(lambda c: "Both locales" if c == "all" else config.LANGUAGES[c]),
            name=labels[comparison], mode="markers", marker=dict(size=9),
            error_x=dict(type="data", symmetric=False, array=frame.ci_high - frame.effect, arrayminus=frame.effect - frame.ci_low),
            customdata=frame.n_images, hovertemplate="%{y}<br>Difference %{x:.4f}<br>n=%{customdata} image pairs<extra></extra>"))
    effect.add_vline(x=0, line_color="#999", line_dash="dot")
    effect.update_xaxes(title="Paired margin difference with 95% bootstrap interval")
    style_figure(effect, "Separate a language change from a same-language description change")
    effect.update_layout(legend=dict(orientation="h", y=-0.3), margin=dict(b=110))
    status = (f"Showing {len(pairs)} pairs, with {(pairs.review_status == 'human_reviewed').sum()} independently reviewed. "
              "AI-generated translations can introduce meaning changes. Treat these results as an exploratory pilot.")
    return scatter, effect, status


def translation_detail(click):
    if not click or not click.get("points") or not click["points"][0].get("customdata"):
        return "Select a translation point to read the source, translation, and independent German descriptions."
    pair_id = click["points"][0]["customdata"][0]
    _, pairs = translation_data()
    match = pairs[pairs.pair_id == pair_id]
    if match.empty: return "No translation pair selected."
    row = match.iloc[0]
    q = get_data()["queries"]
    human = q[(q.image_key == row.image_key) & (q.caption_lang == row.target_lang)]
    return html.Div([
        picture(row.image_key, f"{row.pair_id} · {config.LANGUAGES[row.locale]} locale"),
        html.P(f"English source: {row.source_text} · rank {row.source_rank:g}"),
        html.P(f"German translation: {row.translation} · rank {row.translated_rank:g}"),
        html.B("Independent German descriptions"), html.Ul([html.Li(f"{r.text} · rank {r.rank}") for r in human.itertuples()]),
        html.Small(f"Provenance: {row.translator}. Review: {row.review_status}. {row.get('review_notes', '')}"),
    ])


def save_review(records, selected, error_type, visible_text, reviewer, notes):
    if not records or not selected or selected[0] >= len(records) or not str(reviewer or "").strip():
        return "Select a caption and enter a reviewer name."
    if error_type not in ERROR_TYPES or visible_text not in {"yes", "no", "unclear"}:
        return "Choose valid review categories."
    caption_id = int(records[selected[0]]["caption_id"])
    row = get_data()["queries"].set_index("caption_id").loc[caption_id]
    record = dict(caption_id=caption_id, image_key=row.image_key, top_image_key=row.top_image_key,
                  caption_lang=row.caption_lang, error_type=error_type, visible_text=visible_text,
                  reviewer=reviewer.strip(), notes=notes or "",
                  records_sha256=get_data()["metadata"]["records_sha256"])
    previous = pd.read_csv(REVIEWS, keep_default_na=False) if REVIEWS.exists() else pd.DataFrame()
    if len(previous): previous = previous[previous.caption_id != caption_id]
    updated = pd.concat([previous, pd.DataFrame([record])], ignore_index=True)
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    temporary = REVIEWS.with_suffix(".tmp.csv")
    updated.to_csv(temporary, index=False)
    temporary.replace(REVIEWS)
    return f" Saved interpretation for caption {caption_id} in annotations/error_reviews.csv."


def review_summary():
    if not REVIEWS.exists():
        return html.P("No human error interpretations recorded yet. Categories are never inferred from locale alone.", className="muted")
    reviews = pd.read_csv(REVIEWS, keep_default_na=False)
    reviews = reviews[reviews.records_sha256 == get_data()["metadata"]["records_sha256"]]
    fig = go.Figure()
    for error_type, group in reviews.groupby("error_type"):
        counts = group.caption_lang.value_counts()
        fig.add_trace(go.Bar(x=counts.index.map(config.LANGUAGES), y=counts.values, name=ERROR_TYPES.get(error_type, error_type)))
    style_figure(fig, f"Human interpretations · {len(reviews)} reviewed queries")
    fig.update_layout(barmode="stack")
    return html.Div([dcc.Graph(figure=fig), html.P("Counts describe the reviewed sample, which may be selectively inspected. "
                      "They do not estimate population error frequencies without a sampling design.", className="muted")])


def register_callbacks(app):
    from .bias_dashboard import register_callbacks as register_bias_callbacks
    register_bias_callbacks(app)
    @app.callback(Output("effect-plot", "figure"), Output("effect-summary", "children"), Input("effect-metric", "value"), Input("effect-view", "value"), Input("omit-hi", "value"))
    def update_effect(metric, view, omit): return effect_figure(metric, view, bool(omit))

    @app.callback(Output("query-language", "value"), Output("query-locale", "value"), Input("selected-cell", "data"), prevent_initial_call=True)
    def follow_cell(cell):
        from dash import no_update
        return (cell["caption_lang"], cell["locale"]) if cell else (no_update, no_update)

    @app.callback(Output("query-table", "data"), Output("query-summary", "children"), Output("query-table", "selected_rows"),
                  Input("query-language", "value"), Input("query-locale", "value"), Input("query-topic", "value"), Input("query-result", "value"), Input("query-sort", "value"))
    def update_queries(*args): return filtered_queries(*args)

    @app.callback(Output("retrieval-detail", "children"), Input("query-table", "data"), Input("query-table", "selected_rows"))
    def update_detail(records, selected): return retrieval_detail(records, selected)

    @app.callback(Output("cue-plot", "figure"), Output("cue-matching", "figure"), Output("length-plot", "figure"), Output("token-summary", "children"), Input("cue-language", "value"))
    def update_cues(lang):
        cue, length, summary = cue_figures(lang)
        return cue, cue_matching_figure(lang), length, summary

    @app.callback(Output("translation-scatter", "figure"), Output("translation-effects", "figure"), Output("translation-status", "children"), Input("translation-review", "value"))
    def update_translations(review): return translation_figures(review)

    @app.callback(Output("translation-detail", "children"), Input("translation-scatter", "clickData"))
    def show_translation(click): return translation_detail(click)

    @app.callback(Output("review-status", "children"), Output("review-summary", "children"), Input("save-review", "n_clicks"), State("query-table", "data"), State("query-table", "selected_rows"),
                  State("review-type", "value"), State("review-text", "value"), State("reviewer", "value"), State("review-notes", "value"), prevent_initial_call=True)
    def record_review(clicks, *args): return save_review(*args), review_summary()

    @app.callback(Output("review-download", "data"), Input("download-review", "n_clicks"), prevent_initial_call=True)
    def download_reviews(clicks): return dcc.send_file(str(analysis_path("error_review_template.csv")))
