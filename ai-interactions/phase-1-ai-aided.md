# Phase I: AI-aided team interaction record

**Dates:** 10–11 November 2026  
**Participants and tools:** AI-aided team members working with Claude Opus 5.5 (claude.ai agent mode) and the Codex assistant (GPT-6 family).  
**Goal:** Investigate how caption language and image locale relate to cross-modal retrieval on Crossmodal-3600 (XM3600), and document decisions and limitations of AI assistance.

## Concept and analysis design

- The team specified `google/siglip2-base-patch16-256`, a 36 × 36 caption-language by image-locale heatmap, matched versus mismatched comparisons with uncertainty and permutation testing, and interactive examples. The assistant corrected documentation after the user clarified that this was initially a plan, not completed analysis. The shared Phase II concept was to compare LIME and Integrated Gradients on a balanced set of the same images with captions in different languages.
- Asked to brainstorm before implementation, the AI suggested comparing a caption's image similarity with its mean similarity to other images, and using double-centering to separate language and locale baselines. The team adopted these ideas, added raw and row-centered views, and made its own choices about grouping, visualization and candidate images.
- The AI suggested a cultural grouping with a source. The team checked that source and corrected the draft assignments. Locale is an imperfect proxy for culture, especially for languages spoken across countries. The dataset's unusual Hindi and Persian image sampling also requires caution.

## Implementation and verification

- Codex implemented the SigLIP 2 pipeline and helped run it on an Intel GPU. The GPU initially failed because of a missing OpenCL configuration. A local runtime configuration resolved it. CPU and GPU outputs agreed within the stated tolerance. The completed run produced embeddings for 3,600 images and 261,375 captions. Scores were checked against direct calculations and the dashboard views were tested.
- At the user's request, the assistant organized experiments and results in `phase-1/ai-aided/Phase_I_Experiments.ipynb`, with supporting code in `implementation/`. Notebook execution and regression tests verified the reorganization and later additions.

## Investigation and interpretation

- Following discussion of the lab slides, the team added retrieval examples, analyses of caption keywords and length, uncertainty estimates based on images, and a pilot translating English captions into German. The 72 pilot images were selected without using scores. The translations were AI-generated and not independently reviewed. Independent German captions outperformed the translations in this pilot. Differences in content, style and translation quality remain possible explanations. The result does not establish a causal effect or generalize to all 36 languages.
- At the team's request, Codex added the FairFace audit after the original XM3600 investigation. This extension uses the same model with a separate portrait dataset, supplied annotations and neutral queries in six languages. German and Japanese queries showed different group shares in the top results. Swahili also showed a shift, alongside weaker separation between person and object prompts. The results depend on the fixed pool and unreviewed translations.
- The team documented the remaining limits: locale sampling, unreviewed translations, benchmark category limitations and the absence of a systematic human review of retrieval errors.

## Lessons from the collaboration

- The team adopted the suggested margin and centering methods, then checked the assumptions behind the comparisons. Source checks led to corrections in the AI-generated group assignments.
- Hardware checks and independent numerical comparisons verified GPU execution.
- A browser check exposed a dashboard issue, and a hand-calculated test expectation needed correction. Checking the tests was part of checking the generated code.
