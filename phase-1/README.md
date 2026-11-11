# Phase I: How does the model behave?

**Research question:** Does a caption's language match the image's locale, and
does that improve SigLIP 2's similarity score?

**Status:** The AI-aided full SigLIP 2 run completed on the local Intel GPU on
11 November 2026, generating embeddings for 3,600 images and 261,375 captions,
plus margin scores and heatmap matrices. The heatmap and example gallery use
these results. The human-only notebook uses the same checkpoint. The
AI-aided analysis includes uncertainty estimates and permutation tests for
matching effects, retrieval examples, caption keywords and length diagnostics.
A translation pilot covers 72 images with English captions translated into
German. The translations still need independent review.

The AI-aided pair later added a **FairFace extension**, presented in Appendix A
of the notebook. It asks whether neutral person queries in six languages
retrieve different demographic groups from the same 448 portraits. This is a
separate experiment using the same checkpoint. XM3600 remains the dataset for
the original question about language and locale. The audit uses FairFace's annotations.
Its translations need independent review, and the comparison of person and object prompts suggests
weaker semantic alignment for Swahili.

Start with the [AI-aided experiment notebook](ai-aided/Phase_I_Experiments.ipynb)
and its [setup and run instructions](ai-aided/README.md). Supporting Python code
is in `ai-aided/implementation/`. The family/culture notebook is supplementary.

Both pairs study the inputs and outputs of `google/siglip2-base-patch16-256`
as a black box, using Crossmodal 3600. Model and dataset links are in the
[project overview](../README.md).

- [AI-aided pair](ai-aided/): Klaus Niederberger (K12342339) and Markus Brunner (K12141831).
- [Human-only pair](human-only/): Jakob Schwarzmaier (K12325928) and William Johann Strouhal (K12240477).

## Analysis direction

1. **Language × locale heatmap:** compare all 36 caption languages with all
   36 image locale labels. Normalize each language row to reveal patterns
   beyond language baselines. Document the score definition and normalization,
   and keep the underlying scores available for inspection.
2. **Matched versus mismatched comparison:** for each caption language, compare
   scores on images whose locale label matches that language with scores on
   images whose locale label differs. Show the estimated difference with
   uncertainty and a permutation test. Document the sampling and test design.
3. **Interactive examples:** clicking a heatmap cell should show its images,
   captions, and scores. Investigate concrete cases where a matching advantage
   appears, disappears, or reverses, and connect them to systematic patterns.

The [AI-aided notebook](ai-aided/Phase_I_Experiments.ipynb) follows the heatmap
patterns into retrieval errors and caption-content comparisons. The human-only
pair's notebook develops its own analysis of the shared question.

## Dataset interpretation

Use **image locale** for the sampling label, rather than treating it as a
guaranteed region of origin. The dataset uses geographic sampling with fallback
rules. All 100 images in its Hindi subset were sampled globally. Captions are
written independently across languages and can describe different details.
See the [dataset paper, Sections 2.2–2.3](https://arxiv.org/abs/2205.12522).

Consequently, interpret score differences as associations involving language,
caption content, and image selection. These comparisons alone do not isolate
a causal language effect or establish cultural bias.

## Pair coordination

- Before starting: agree on the same task and model, questions, dataset, and
  initial visualization, then conduct the investigations separately.
- At the end: compare findings, visualization choices, breadth and depth, time
  and verification effort, and confidence supported by evidence. Identify
  behaviors to investigate further in Phase II.

Save influential AI interactions in [../ai-interactions/](../ai-interactions/)
and complete [the midpoint reflection](../ai_reflection_midpoint.md), including
the Phase I AI-aided pair's names and student IDs at the beginning.

## Midpoint submission

- Public GitHub repository link: record it in the [project overview](../README.md).
- Phase I [AI interaction records](../ai-interactions/).
- [GenAI reflection](../ai_reflection_midpoint.md), kept at the repository root.
- Progress video (up to 5 minutes): not yet recorded.

Keep supporting slides and presentation files directly in this folder. The video
introduces the problem, summarizes each pair's work and findings, and shows key
visualizations.
