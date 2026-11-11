# Implementation reference

Supporting code for [Phase_I_Experiments.ipynb](../Phase_I_Experiments.ipynb).
Run all commands below from **`phase-1/ai-aided/`**, the parent of this folder.
Notebook setup is documented in the [main README](../README.md).

## Modules

| Module | Purpose |
|---|---|
| `prepare_data`, `xm3600_reader` | Read XM3600 and construct aligned image/caption tables |
| `embed` | Encode images and captions, resume work, and merge worker chunks |
| `compute_scores` | Compute cosine margins and language-by-locale matrices |
| `analyze` | Retrieval ranks, matching-effect inference, caption cues, and token diagnostics |
| `translation_control` | Prepare and score the paired English-to-German pilot |
| `bias_audit` | Prepare a balanced FairFace pool and score multilingual neutral person prompts |
| `bias_plots`, `bias_dashboard` | Notebook figures and optional representation-audit tab |
| `notebook_support` | Load validated results and render notebook figures |
| `app`, `explorer`, `assets/` | Optional Dash explorer and human-review interface |
| `config`, `artifacts` | Model settings, paths, groupings, and provenance validation |
| `tests/` | Numerical, pipeline, and dashboard checks |

## Data and model

Download [XM3600](https://google.github.io/crossmodal-3600/) and place
`captions.jsonl` and the `<image-key>.jpg` files together in `xm3600/`.
Raw data and model weights are local files, excluded from Git.

The pipeline uses `google/siglip2-base-patch16-256` at revision
`3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`. The first embedding run downloads
the pinned checkpoint and processor. Install a PyTorch build suitable for the
inference device when setting up the environment.

## Reproduce the analysis

```bash
.venv/bin/python -m implementation.prepare_data
.venv/bin/python -m implementation.embed --test
.venv/bin/python -m implementation.embed
.venv/bin/python -m implementation.compute_scores
.venv/bin/python -m implementation.analyze
.venv/bin/python -m implementation.translation_control score --device cpu
```

Start at `analyze` when embeddings and scores are already available. It performs
no model inference but uses the pinned tokenizer for length diagnostics.
`--skip-tokenizer` explicitly omits those diagnostics. Translation scoring
embeds the translated texts and reuses original image/caption results.

For a new pilot, run `python -m implementation.translation_control prepare
--input annotations/new_pilot.csv --per-locale 36`, fill its translation and
review fields, then score that file with `score --input annotations/new_pilot.csv`.
Set `review_status` to `human_reviewed` only after independent review, or
`rejected` to exclude a pair. Changes to the CSV require rescoring.

Inference accepts `--device cuda`, `xpu`, `mps`, or `cpu`. The default is `auto`.
For two workers, run `python -m implementation.embed --part 0/2` and
`python -m implementation.embed --part 1/2` in separate sessions. Part 0 also
encodes images. After both finish, run `python -m implementation.embed --merge`.
Single-worker runs also resume completed chunks.

`VIZ4AI_PROCESSED` selects an alternative output directory. Use the same setting
for every worker and downstream command.

## Additional experiment: FairFace

The FairFace audit extends the original XM3600 investigation with a separate
question about query language and portrait retrieval. It uses the same model
checkpoint and a new candidate pool. FairFace supplies race, gender and age
annotations for the group comparisons. The results are presented in Appendix A
of the notebook.

```bash
.venv/bin/python -m implementation.bias_audit prepare
.venv/bin/python -m implementation.bias_audit score --device cpu
```

`prepare` downloads the pinned FairFace validation parquet (about 63 MB), verifies
its SHA-256, and deterministically selects 448 adult portraits. It reproduces
the checked-in manifest and restores the local gallery images. `score` uses the
already-cached pinned SigLIP 2 checkpoint, encodes the new portraits and prompts,
and computes top-1/10/50 group shares plus 500 paired balanced-subset checks.
Existing audit image embeddings are reused when their manifest and model match.
XM3600 embeddings are not recomputed. If setting up a fresh machine, download
the checkpoint with the normal embedding setup first.

Edit `annotations/bias_prompts.csv` to review translations. Set `human_reviewed`
only after independent review and record the reviewer in `translator`. Rerun
`score` after any edit. Stale inputs are rejected by the notebook and dashboard.
The current 45 translated prompts are marked `ai_generated`. Model settings,
source revision, sampling seed, candidate hashes, token counts and translation
provenance are saved alongside `processed/bias_*` results. Raw portraits and
embeddings stay local. Derived score tables and metadata can be shared.

See the [experiment notebook](../Phase_I_Experiments.ipynb) for the design and interpretation limits. Candidate-subset stability ranges are not population confidence intervals.

## Optional dashboard

```bash
.venv/bin/python -m implementation.app
```

Open <http://127.0.0.1:8050>. Select a heatmap cell to explore its captions,
intended image, and retrieved candidates. Additional tabs show matching effects,
caption cues, token lengths, translation controls, and the **FairFace extension**.
The representation tab switches annotation axis, ranking depth and reference,
and compares a selected language's portraits with English. Raw JPGs are needed for
image galleries. Completed-result views require no GPU.

The retrieval review form saves interpretations to `annotations/error_reviews.csv`,
including reviewer and run provenance. Its counts describe the reviewed sample.
`processed/error_review_template.csv` provides a fixed random query per
language/locale cell. A completed worksheet is not imported automatically.

## Intel GPU inference

```bash
.venv/bin/python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/xpu
./implementation/run_xpu.sh --test
./implementation/run_xpu.sh
```

The wrapper uses `.venv/bin/python`, selects XPU, and caches weights under
`data/huggingface/`. It needs a compatible Intel Level Zero runtime and access
to the GPU render device. If `.venv/intel-runtime/usr/lib64` exists, the wrapper
loads that local runtime. The environment and runtime are excluded from Git.

## Scores and provenance

Inference runs in evaluation mode with float32 features. The pinned processor
converts RGB images to 256 × 256 inputs. Captions are lowercased, padded to
64 tokens, and truncated. L2-normalized 768-dimensional embeddings are stored
as float16. Cosine scoring uses float32.

A caption's margin is its cosine similarity to the intended image minus its
mean cosine similarity to other images. Heatmaps average captions within each
image and language, then average images within each locale. Raw, row-centered,
and double-centered margins are available.
See the [experiment notebook](../Phase_I_Experiments.ipynb) for inference and interpretation.

Metadata records model settings and the ordered-input fingerprint. Downstream
readers validate these before displaying results. Preparing data or merging
embeddings invalidates downstream metadata. Rerun scoring and analysis afterward.

## Verification

```bash
.venv/bin/python -m unittest discover -s implementation/tests -v
```

Tests use synthetic data and a tiny randomly initialized SigLIP model without
model downloads. They cover encoding, batching/merging, row alignment, exact
ranks and margins, statistical resampling, translation provenance, and dashboard
components. These checks verify implementation rather than real retrieval quality.

## Sources and reproducibility

- [XM3600 dataset and annotation documentation](https://google.github.io/crossmodal-3600/). Images in the notebook's retrieval examples are from XM3600. Image IDs link them to dataset attribution records.
- [SigLIP 2 checkpoint](https://huggingface.co/google/siglip2-base-patch16-256), pinned revision `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`.
- [FairFace, Kärkkäinen & Joo, WACV 2021](https://github.com/joojs/fairface), CC BY 4.0. The [Hugging Face mirror](https://huggingface.co/datasets/HuggingFaceM4/FairFace) provides the validation split with crop padding 0.25 at revision `54d573cdb8b5af490ba8da9da2799628f6e5c496`. Candidate IDs, hashes and seed are recorded in `processed/bias_candidates.csv` and its metadata.
- [AI interaction records](../../../ai-interactions/README.md) and the [supplementary family/culture experiment](../supplementary/family_vs_culture_analysis.ipynb).

The XM3600 analysis contains 3,600 images, 261,375 caption queries and 129,600 image/language units. Its input fingerprint is `ec7f2b7c2dca8e7e4d86d520290d3d038cb4cc45f3fe9be1c860acf01a619ddc`. The translation pilot selects 36 images from each of the English and German locales before scoring, using seed `20261111`. The FairFace pool of 448 portraits and the 500 balanced subset checks use the same seed. Candidate order is shuffled once so ties follow a fixed order. Five neutral prompt families and four object controls are used in each of six languages. All 45 translated prompts await independent review.

- Retrieval ranks and ties in the top five use the original image table order. Results for each caption, candidate indices and scores are stored in `processed/retrieval.parquet`, `processed/retrieval_indices.npy` and `processed/retrieval_cosines.npy`. Indices refer to the complete original image table. The result loader checks model settings, counts, input fingerprints and translation provenance.
- Matching intervals are percentile bootstraps with seed `20261111`. Stored Benjamini–Hochberg q-values are calculated separately within each metric, view and sensitivity set. The intervals in the plots are pointwise. The Hindi sensitivity analysis does not recompute retrieval ranks.
- Caption cues come from the keyword rules in `implementation.analyze.TOPICS`. They can overlap and miss synonyms. The notebook's cue comparison requires at least three images in each cohort and displays point estimates without adjusted significance tests.
- The FairFace source parquet SHA-256 is `d0cfac2888356b3d3fa3d837bc30961d8c860934e4c5df862e8cb87812bfa137`. Selection is without replacement within strata. The image bytes were checked for duplicates, but distinct identities were not established. Rank-discounted shares and mean ranks across the whole pool are available in the stored audit tables.
- Both experiments use the pinned checkpoint revision recorded above, normalized 768-dimensional embeddings, lowercased text and a 64-token limit. The audit uses normal checkpoint image preprocessing.

Matching inference uses 2,000 locale-stratified image bootstraps and 4,999 shared, two-sided locale-label permutations with a plus-one correction. All languages use the same shuffle and locale counts are preserved. The global statistic averages language effects. The translation pilot uses 2,000 paired image resamples within locales.
