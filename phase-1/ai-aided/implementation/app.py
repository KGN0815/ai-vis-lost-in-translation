"""
Phase I heatmap: caption language x image locale (Dash).

Run:  python -m implementation.app   then open http://127.0.0.1:8050

Controls
  Normalisation : raw margin / row-centred / double-centred
  Grouping      : language family / Inglehart-Welzel cultural zone
Marks
  outline = matched cell (caption language == image locale)
  dot     = best-describing caption language for each locale (column maximum)
Heatmap selection links the language and locale to the example and retrieval views.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from . import config
from .artifacts import read_metadata, score_settings

OUTLINE_COLOR = "black"
DOT_COLOR = "#404040"   # dark gray
SEPARATOR_COLOR = "#9a9a9a"
CELL_PX = 18

STATE_LABELS = {
    "raw": "Raw margin",
    "row": "Row-centred",
    "double": "Double-centred",
}
COLORBAR_TITLES = {
    "raw": "Margin",
    "row": "Margin −<br>language mean",
    "double": "Residual<br>(language &<br>locale removed)",
}
GROUPING_LABELS = {"family": "Language family", "culture": "Cultural zone (Inglehart–Welzel)"}

read_metadata(config.SCORE_METADATA, "-m implementation.compute_scores", score_settings())
MATRICES = pd.read_parquet(config.MATRICES_FILE)


def label(code: str) -> str:
    star = "*" if code in config.LOCALE_NOTES else ""
    return f"{config.LANGUAGES[code]} ({code}){star}"


def color_settings(values: np.ndarray, state: str) -> dict:
    """Sequential 0..98th percentile for raw; symmetric RdBu around 0 otherwise."""
    if state == "raw":
        return dict(colorscale="Viridis", zmin=0.0, zmax=float(np.nanpercentile(values, 98)))
    a = float(np.nanpercentile(np.abs(values), 98))
    return dict(colorscale="RdBu_r", zmin=-a, zmax=a, zmid=0.0)


def build_figure(state: str, grouping: str) -> go.Figure:
    order = config.ordered_languages(grouping)
    codes = [c for _, c in order]
    groups = [g for g, _ in order]
    n = len(codes)

    df = MATRICES[MATRICES["state"] == state]
    z = df.pivot(index="caption_lang", columns="locale", values="value").reindex(index=codes, columns=codes)
    rank = df.pivot(index="caption_lang", columns="locale", values="rank_in_column").reindex(index=codes, columns=codes)
    n_img = df.pivot(index="caption_lang", columns="locale", values="n_images").reindex(index=codes, columns=codes)
    n_cap = df.pivot(index="caption_lang", columns="locale", values="n_captions").reindex(index=codes, columns=codes)
    Z = z.to_numpy()

    # Hover data: language, locale, rank, image count, caption count, sampling note.
    custom = np.empty((n, n, 6), dtype=object)
    for i, cl in enumerate(codes):
        for j, loc in enumerate(codes):
            note = config.LOCALE_NOTES.get(loc, "")
            custom[i, j] = [config.LANGUAGES[cl], config.LANGUAGES[loc], int(rank.iat[i, j]),
                            int(n_img.iat[i, j]), int(n_cap.iat[i, j]),
                            f"<br><i>{note}</i>" if note else ""]

    fig = go.Figure(go.Heatmap(
        z=Z, x=list(range(n)), y=list(range(n)), customdata=custom,
        colorbar=dict(title=dict(text=COLORBAR_TITLES[state]), thickness=14, len=0.6,
                      x=1.0, xref="paper", xpad=150),
        hovertemplate=(
            "Caption language: <b>%{customdata[0]}</b><br>"
            "Image locale: <b>%{customdata[1]}</b><br>"
            "Value: %{z:.4f}<br>"
            "Rank in column: %{customdata[2]} / " + str(n) + "<br>"
            "Images: %{customdata[3]}, captions: %{customdata[4]}"
            "%{customdata[5]}<extra></extra>"
        ),
        xgap=1, ygap=1,
        **color_settings(Z, state),
    ))

    # dot: best-describing caption language per locale (column maximum)
    best_rows = np.nanargmax(Z, axis=0)
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=best_rows.tolist(), mode="markers", hoverinfo="skip",
        marker=dict(color=DOT_COLOR, size=7, line=dict(color="white", width=1)),
        name="Best describer", showlegend=False,
    ))

    shapes = []
    # group separators (drawn first so the matched-cell outlines stay on top)
    starts = [0] + [k for k in range(1, n) if groups[k] != groups[k - 1]]
    bounds = starts + [n]
    for b in starts[1:]:
        common = dict(type="line", line=dict(color=SEPARATOR_COLOR, width=1.5), layer="above")
        shapes.append(dict(common, x0=b - 0.5, x1=b - 0.5, y0=-0.5, y1=n - 0.5))
        shapes.append(dict(common, y0=b - 0.5, y1=b - 0.5, x0=-0.5, x1=n - 0.5))
    # outline around matched cells
    for i in range(n):
        shapes.append(dict(type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                           line=dict(color=OUTLINE_COLOR, width=2), layer="above"))
    annotations = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        mid = (s + e - 1) / 2
        g = groups[s]
        annotations.append(dict(x=mid, y=1.0, xref="x", yref="paper", yanchor="bottom",
                                text=g, textangle=-60, showarrow=False, font=dict(size=10),
                                xanchor="left"))
        annotations.append(dict(x=1.0, y=mid, xref="paper", yref="y", xanchor="left",
                                xshift=6, text=g, showarrow=False, font=dict(size=10)))

    side = n * CELL_PX
    ticks = [label(c) for c in codes]
    fig.update_layout(
        shapes=shapes, annotations=annotations,
        width=side + 520, height=side + 330,
        margin=dict(l=170, r=330, t=120, b=170),
        plot_bgcolor="white",
        xaxis=dict(tickmode="array", tickvals=list(range(n)), ticktext=ticks, tickangle=-90,
                   title="Image locale", side="bottom", showgrid=False, zeroline=False,
                   range=[-0.6, n - 0.4], constrain="domain"),
        yaxis=dict(tickmode="array", tickvals=list(range(n)), ticktext=ticks,
                   title="Caption language", range=[n - 0.4, -0.6], showgrid=False,
                   zeroline=False, scaleanchor="x", constrain="domain"),
        font=dict(size=11),
    )
    return fig


app = Dash(__name__, assets_folder=str(config.ROOT / "implementation" / "assets"), suppress_callback_exceptions=True)
app.title = "Phase I heatmap"
CENTER = {"display": "flex", "justifyContent": "center"}

app.layout = html.Div(style={"fontFamily": "sans-serif", "padding": "12px", "textAlign": "center"}, children=[
    html.H3("Caption language × image locale (SigLIP 2, XM3600)"),
    html.Div(style={**CENTER, "gap": "40px"}, children=[
        html.Div([html.B("Normalisation"), dcc.RadioItems(
            id="state", value="double", inline=True,
            options=[{"label": v, "value": k} for k, v in STATE_LABELS.items()])]),
        html.Div([html.B("Grouping"), dcc.RadioItems(
            id="grouping", value="family", inline=True,
            options=[{"label": v, "value": k} for k, v in GROUPING_LABELS.items()])]),
    ]),
    html.Div("Outline = matched cell (caption language = image locale). "
             "Dot = best-describing caption language for that locale. "
             "* = locale with a known caveat about where its images come from (see hover).",
             style={"fontSize": "13px", "color": "#555", "margin": "6px 0"}),
    html.Div(style={"overflowX": "auto", "width": "100%"}, children=[
        dcc.Graph(id="heatmap", config={"displaylogo": False}, style={"width": "max-content", "margin": "auto"})]),
    dcc.Store(id="selected-cell"),
    html.Div(id="selection-info", style={"fontSize": "14px", "fontWeight": "bold"}),
    html.Div(id="examples", style={"maxWidth": "1400px", "margin": "16px auto", "textAlign": "left"}),
])


@app.callback(Output("heatmap", "figure"), Input("state", "value"), Input("grouping", "value"))
def update_heatmap(state, grouping):
    return build_figure(state, grouping)


@app.callback(Output("selected-cell", "data"), Output("selection-info", "children"),
              Input("heatmap", "clickData"), State("grouping", "value"))
def select_cell(click, grouping):
    if not click or not click.get("points"):
        return None, "Click a cell to select it."
    p = click["points"][0]
    # x/y are the column/row indices of the clicked cell (also when a dot is clicked);
    # map them back to language codes with the axis order of the current grouping
    codes = [c for _, c in config.ordered_languages(grouping)]
    try:
        cell = {"caption_lang": codes[int(round(p["y"]))], "locale": codes[int(round(p["x"]))]}
    except (KeyError, TypeError, ValueError, IndexError):
        return None, f"Could not read the click: {p}"
    return cell, (f"Selected: captions in {config.LANGUAGES[cell['caption_lang']]} "
                  f"× images from the {config.LANGUAGES[cell['locale']]} locale")


# --------------------------------------------------------------------------
# Example view: images of the selected cell, sorted by margin
# --------------------------------------------------------------------------
IMAGE_LANG = pd.read_parquet(config.IMAGE_LANG_TABLE)          # (image, caption language) means
CAPTIONS = pd.read_parquet(config.CAPTION_SCORES,                # caption level, with margins
                           columns=["image_key", "caption_lang", "text", "margin"])
ENGLISH = (pd.read_parquet(config.CAPTIONS_TABLE, columns=["image_key", "caption_lang", "text"])
           .query("caption_lang == 'en'").groupby("image_key")["text"].first())
LANG_MEAN = IMAGE_LANG.groupby("caption_lang")["margin"].mean()   # language baseline
N_EACH = 5   # best / median / worst images shown


@app.server.route("/img/<key>")
def serve_image(key):
    from flask import send_from_directory
    return send_from_directory(config.IMAGES_DIR, f"{key}.jpg")


def image_card(row, lang, locale):
    key = row["image_key"]
    caps = CAPTIONS[(CAPTIONS.image_key == key) & (CAPTIONS.caption_lang == lang)]
    matched = IMAGE_LANG[(IMAGE_LANG.image_key == key) & (IMAGE_LANG.caption_lang == locale)]["margin"]
    matched_txt = (f"{config.LANGUAGES[locale]} (matched) on this image: {matched.iloc[0]:.3f}"
                   if len(matched) and lang != locale else "")
    return html.Div(style={"width": "250px", "border": "1px solid #ddd", "borderRadius": "6px",
                           "padding": "6px", "fontSize": "12px", "background": "white"}, children=[
        html.Img(src=f"/img/{key}", style={"width": "100%", "height": "170px", "objectFit": "cover",
                                            "borderRadius": "4px"}),
        html.Div([html.B(f"margin {row['margin']:.3f}"), f"  ({key})"], style={"margin": "4px 0"}),
        html.Ul([html.Li(f"{t}  [{m:.3f}]") for t, m in zip(caps["text"], caps["margin"])],
                style={"paddingLeft": "16px", "margin": "2px 0"}),
        html.Div(f"English reference caption: {ENGLISH.get(key, '–')}",
                 style={"color": "#666", "fontStyle": "italic", "marginTop": "4px"}),
        html.Div(matched_txt, style={"color": "#666", "marginTop": "2px"}),
    ])


def card_row(title, df, lang, locale):
    return html.Div([
        html.H4(title, style={"margin": "14px 0 6px"}),
        html.Div([image_card(r, lang, locale) for _, r in df.iterrows()],
                 style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
    ])


@app.callback(Output("examples", "children"), Input("selected-cell", "data"))
def show_examples(cell):
    if not cell:
        return ""
    lang, locale = cell["caption_lang"], cell["locale"]
    sub = (IMAGE_LANG[(IMAGE_LANG.caption_lang == lang) & (IMAGE_LANG.locale == locale)]
           .sort_values("margin", ascending=False).reset_index(drop=True))
    if sub.empty:
        return html.Div("No data for this cell.")
    mid = len(sub) // 2
    note = config.LOCALE_NOTES.get(locale)
    summary = (f"{len(sub)} images from the {config.LANGUAGES[locale]} locale, captions in "
               f"{config.LANGUAGES[lang]}: mean margin {sub['margin'].mean():.3f} "
               f"(this language over all locales: {LANG_MEAN[lang]:.3f}, "
               f"difference {sub['margin'].mean() - LANG_MEAN[lang]:+.3f}). "
               "Numbers in [ ] are the margins of the individual captions.")
    return html.Div([
        html.Div(summary, style={"fontSize": "14px"}),
        html.Div(note, style={"fontSize": "13px", "color": "#a33"}) if note else None,
        card_row(f"Best {N_EACH} images", sub.head(N_EACH), lang, locale),
        card_row(f"Median {N_EACH} images", sub.iloc[max(mid - N_EACH // 2, 0):mid - N_EACH // 2 + N_EACH], lang, locale),
        card_row(f"Worst {N_EACH} images", sub.tail(N_EACH).iloc[::-1], lang, locale),
    ])


from .explorer import layout as investigation_layout, register_callbacks

app.layout.children.append(investigation_layout())
register_callbacks(app)
app.title = "Lost in Translation · Phase I"

if __name__ == "__main__":
    app.run(debug=False)
