# Phase I: AI-aided experiments

Start with **[Phase_I_Experiments.ipynb](Phase_I_Experiments.ipynb)** for the
research questions, analysis, saved plots, image examples, and findings.

The analysis covers 3,600 XM3600 images and 261,375 captions using SigLIP 2.
A translation pilot using 72 images explores the roles of language and content
by translating English captions into German. The translations still need
independent human review.

**Appendix A adds a separate FairFace experiment** to the original XM3600
analysis. It asks whether the language of a neutral person query changes which
portraits rank highest. The same SigLIP 2 checkpoint searches 448 balanced
FairFace portraits using five prompt families in English, German, French,
Spanish, Japanese and Swahili. The audit uses FairFace's existing annotations.
Its translated prompts also need independent review.

## Contents

| File or folder | Purpose |
|---|---|
| `Phase_I_Experiments.ipynb` | Main experiments and results |
| `supplementary/` | Exploratory language-family/cultural-zone comparison |
| `implementation/` | Data preparation, inference, analysis, plotting, tests, and optional dashboard |
| `processed/` | Scores, derived tables, run metadata, and local embeddings |
| `annotations/` | Translation inputs, representation-audit prompts, provenance, and review records |
| `xm3600/` | Local raw captions and images, excluded from Git |
| `data/` | Local model cache, FairFace source/images, and HTML exports, excluded from Git |

## Read or run the notebook

Saved outputs are included. Running the notebook uses the tables in `processed/`
and local images for examples. It requires neither a GPU nor a webserver. Raw
images and most embedding arrays are ignored by Git and must be downloaded or
regenerated for workflows that require them. See the implementation reference.
Sections 1 to 7 cover XM3600, and Section 8 summarizes those findings.
The FairFace extension follows in Appendix A, with comparisons against the pool
and English queries, subset checks and portrait examples. Its tables work
without downloading FairFace. Regenerating the gallery requires the local images.

From this folder, set up the environment if needed:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --sys-prefix --name ai-vis-phase1 --display-name "AI Vis Phase I (.venv)"
```

In VS Code, enable the Python and Jupyter extensions, open the notebook, select
this folder's `.venv/bin/python` as the kernel, and choose **Run All**.
The notebook also supports execution from the repository root.

To create an executed HTML copy:

```bash
.venv/bin/python -m nbconvert --to html --execute Phase_I_Experiments.ipynb --ExecutePreprocessor.timeout=300 --output-dir data/notebook-runs
```

For raw-data setup, recomputing results, or the optional interactive dashboard,
see [implementation/README.md](implementation/README.md).
