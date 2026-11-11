"""Optional dashboard tab for the same bias-audit outputs used by the notebook."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html, dash_table

from . import config
from .bias_audit import load_results, top_candidates, data_dir
from .bias_plots import GROUPS, AXIS_LABELS


def audit_tab():
    try:
        data = load_results()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return dcc.Tab(label="FairFace extension", value="bias", children=html.Div([
            html.H3("Additional experiment: FairFace portrait retrieval"),
            html.P(f"Audit not ready: {exc}"),
            html.Code("python -m implementation.bias_audit prepare; python -m implementation.bias_audit score"),
        ], className="analysis-panel"))
    families = data["queries"].query("kind == 'neutral'").family.unique()
    meta = data["metadata"]
    return dcc.Tab(label="FairFace extension", value="bias", children=html.Section([
        html.H3("Additional experiment: FairFace portrait retrieval"),
        html.P("This extension follows the original XM3600 analysis of caption language and image locale. It uses the same SigLIP 2 checkpoint "
               "with a separate portrait dataset to ask whether changing a neutral query's language changes which groups rank highest.", className="muted"),
        html.P(f"{meta['n_candidates']} FairFace adult portraits, balanced across seven dataset race annotations, "
               "two gender annotations and age bands 20–29 / 30–39. Every language searches the same pool. "
               "These annotations come from the benchmark; the audit does not infer identities or measure skin tone.", className="muted"),
        html.Div(f"Exploratory pilot: {meta['n_neutral_families']} neutral prompt families in {len(meta['languages'])} languages. "
                 f"{meta['n_unreviewed_translations']} translated neutral and control prompts await independent review. "
                 "Overrepresentation is relative to candidate availability and does not by itself establish unfairness or a causal language effect.", className="notice"),
        html.Div([
            html.Label([html.Span("Dataset annotation", className="control-label"),
                dcc.Dropdown(id="bias-axis", options=[{"label": v, "value": k} for k, v in AXIS_LABELS.items()], value="race", clearable=False)], className="control"),
            html.Label([html.Span("Ranking depth", className="control-label"),
                dcc.Dropdown(id="bias-k", options=[1, 10, 50], value=10, clearable=False)], className="control"),
            html.Label([html.Span("Reference", className="control-label"),
                dcc.Dropdown(id="bias-reference", options=[{"label": "Balanced candidate pool", "value": "pool"}, {"label": "Same prompt families in English", "value": "english"}], value="pool", clearable=False)], className="control"),
        ], className="control-row"),
        dcc.Graph(id="bias-heatmap"),
        html.P("Cells average five prompt results. At depth 10, the 50 retrieval slots may repeat portraits across prompts. "
               f"They are not 50 independent people. Stability ranges use {meta['subsample_repetitions']} common, balanced "
               f"subsets of {meta['subsample_candidates']} portraits. These ranges describe sensitivity to the pool rather than population confidence intervals.", className="muted"),
        html.Div([
            html.Label([html.Span("Inspect language", className="control-label"),
                dcc.Dropdown(id="bias-language", options=[{"label": config.LANGUAGES[c], "value": c} for c in meta["languages"]], value="de", clearable=False)], className="control"),
            html.Label([html.Span("Equivalent prompt family", className="control-label"),
                dcc.Dropdown(id="bias-family", options=list(families), value="person_photo", clearable=False)], className="control"),
        ], className="control-row"),
        html.Div(id="bias-query-text", className="notice"),
        html.P("The gallery shows one selected prompt family (up to five portraits per language). "
               "The table and heatmap average all five families.", className="muted"),
        html.Div(id="bias-gallery"),
        dash_table.DataTable(id="bias-exposure-table", columns=[
            {"name": "Dataset label", "id": "group"}, {"name": "Pool %", "id": "baseline"},
            {"name": "Top-k %", "id": "share"}, {"name": "Excess (pp)", "id": "excess_pp"},
            {"name": "Difference from English (pp)", "id": "delta_english_pp"},
            {"name": "Subset low (pp)", "id": "stability_low"}, {"name": "Subset high (pp)", "id": "stability_high"},
        ], data=[], style_table={"overflowX": "auto"}, style_cell={"textAlign": "left", "padding": "9px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#edf3f5"}),
        dcc.Graph(id="bias-sanity"),
        html.P("Sanity check: the average score for neutral person prompts must exceed the strongest score for car, building, tree or food prompts "
               "for a known portrait. This is a diagnostic, not a validated language-fluency measure. Pose, expression and image quality "
               "are not matched. Benchmark labels and translations can be imperfect.", className="muted"),
        html.P(["Images/annotations: FairFace, Kärkkäinen & Joo (WACV 2021), CC BY 4.0. ",
                html.A("Dataset and citation", href="https://github.com/joojs/fairface", target="_blank")], className="muted"),
    ], className="analysis-panel"))


def heatmap(axis, k, reference):
    data = load_results()
    column = "excess_pp" if reference == "pool" else "delta_english_pp"
    frame = data["exposure"].query("axis == @axis and k == @k")
    matrix = frame.pivot(index="language", columns="group", values=column).reindex(index=data["metadata"]["languages"], columns=GROUPS[axis])
    limit = max(5., float(np.abs(matrix.to_numpy()).max()))
    figure = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=matrix.columns.tolist(),
        y=[config.LANGUAGES[c] for c in matrix.index], colorscale="RdBu_r", zmin=-limit, zmax=limit, zmid=0,
        texttemplate="%{z:+.1f}", hovertemplate="%{y}<br>Dataset label %{x}<br>Difference %{z:+.1f} percentage points<extra></extra>",
        colorbar=dict(title="Difference<br>(pp)")))
    figure.update_layout(template="plotly_white", height=450, margin=dict(l=100, r=30, t=55, b=90),
                         title=f"Top-{k}: representation relative to {'candidate availability' if reference == 'pool' else 'English'}")
    return figure


def detail(language, family, axis, k):
    data = load_results()
    cards = []
    shown = list(dict.fromkeys(["en", language]))
    for lang in shown:
        ranked = top_candidates(data, lang, family, min(int(k), 5))
        portraits = []
        for row in ranked.itertuples():
            portraits.append(html.Div([
                html.Strong(f"#{row.rank} · {row.cosine:.4f}"),
                html.Img(src=f"/fairface/{row.image_key}", alt=f"FairFace portrait {row.image_key}"),
                html.Small(row.image_key), html.P(f"FairFace: {row.race} / {row.gender} / {row.age}"),
            ], className="candidate-card"))
        cards.append(html.Div([html.H4(config.LANGUAGES[lang]), html.Div(portraits, className="candidate-grid")]))
    query = data["queries"].query("language == @language and family == @family").iloc[0]
    frame = data["exposure"].query("language == @language and axis == @axis and k == @k").copy()
    frame[["baseline", "share"]] *= 100
    columns = ["group", "baseline", "share", "excess_pp", "delta_english_pp", "stability_low", "stability_high"]
    return cards, frame[columns].round(2).to_dict("records"), f"{query.text} · English source: {query.source_text} · Review: {query.review_status}"


def sanity_figure():
    frame = load_results()["sanity"]
    figure = go.Figure(go.Bar(x=frame.language.map(config.LANGUAGES), y=frame.person_over_object_rate * 100,
                              text=(frame.person_over_object_rate * 100).round(1), marker_color="#247b8b"))
    figure.update_layout(template="plotly_white", height=350, title="Person-versus-object prompt sanity check",
                         yaxis=dict(title="Portraits passing check (%)", range=[0, 105]))
    return figure


def register_callbacks(app):
    @app.server.route("/fairface/<key>")
    def serve_portrait(key):
        from flask import send_from_directory
        return send_from_directory(data_dir() / "images", f"{key}.jpg")

    @app.callback(Output("bias-heatmap", "figure"), Input("bias-axis", "value"), Input("bias-k", "value"), Input("bias-reference", "value"))
    def update_heatmap(axis, k, reference): return heatmap(axis, k, reference)

    @app.callback(Output("bias-gallery", "children"), Output("bias-exposure-table", "data"), Output("bias-query-text", "children"),
                  Input("bias-language", "value"), Input("bias-family", "value"), Input("bias-axis", "value"), Input("bias-k", "value"))
    def update_detail(language, family, axis, k): return detail(language, family, axis, k)

    @app.callback(Output("bias-sanity", "figure"), Input("bias-language", "value"))
    def update_sanity(language): return sanity_figure()
