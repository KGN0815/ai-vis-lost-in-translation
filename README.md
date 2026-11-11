# AI-Vis: Lost in Translation

## Project details

- **Task:** Investigate how caption language and image locale relate to SigLIP 2's image-text similarity and retrieval results.
- **Model:** [google/siglip2-base-patch16-256](https://huggingface.co/google/siglip2-base-patch16-256), a pretrained SigLIP 2 vision-language embedding model.
- **Dataset:** [Crossmodal 3600 (XM3600)](https://google.github.io/crossmodal-3600/), with 3,600 images and human-written captions in 36 languages.
- **Dataset for the additional experiment:** [FairFace](https://github.com/joojs/fairface), with 448 adult portraits for a separate multilingual retrieval audit (CC BY 4.0).
- **Repository:** [KGN0815/ai-vis-lost-in-translation](https://github.com/KGN0815/ai-vis-lost-in-translation)
- **Team members:** [team-info.json](team-info.json)
- **Workload:** [workload.json](workload.json), with equal contribution shares of 0.25 (25%) per member.

## Project status

Both pairs investigate the relationship between caption language and image
locale on XM3600. The full AI-aided embedding run completed on the local Intel
GPU on 11 November 2026, covering 3,600 images and 261,375 captions in 36 languages.

The [AI-aided notebook](phase-1/ai-aided/Phase_I_Experiments.ipynb) contains the
heatmap, matching comparisons, retrieval examples, caption diagnostics and a
pilot translating English captions into German for 72 images. The translations
still need independent human review. The
[human-only notebook](phase-1/human-only/Phase1_non_ai_team.ipynb) presents the
other pair's analysis of the same model and dataset.

Supporting code and the optional Dash explorer are in
`phase-1/ai-aided/implementation/`.
See the [setup instructions](phase-1/ai-aided/README.md) to run them.
Phase II is planned.

## Additional experiment: FairFace

After the original XM3600 investigation, the AI-aided pair added a separate
question: does the language of a neutral person query change which portraits
rank highest? This extension uses the same SigLIP 2 checkpoint with a balanced
pool of 448 FairFace portraits and queries in six languages. It appears in
**Appendix A** of the AI-aided notebook.

FairFace supplies the demographic annotations used to compare the results.
The audit examines group shares in the retrieved portraits, their sensitivity
to the candidate pool, and how well the model separates person and object prompts.
Its translated prompts still need independent review. The original findings
about caption language and image locale are based on XM3600.

## Research plan

1. **Phase I:** Does a caption's language match the image's locale, and does
   that improve SigLIP 2's similarity score? Examine a 36 × 36
   language-by-locale heatmap, compare matched and mismatched locales with
   uncertainty and a permutation test, and inspect concrete examples.
2. **Phase II:** Which image regions support the score differences between
   languages found in Phase I? Plan to compare LIME and Integrated Gradients
   on the same images with captions in different languages, using a balanced
   set of cases.

Use **image locale** for the dataset's sampling label. Caveats about locale
and caption content are documented in the [Phase I plan](phase-1/README.md).

See the [Phase I](phase-1/README.md) and [Phase II](phase-2/README.md) plans for
the analysis direction.

## Repository structure

```text
phase-1/                    # How does the model behave?
├── ai-aided/
└── human-only/
phase-2/                    # What did the model learn?
├── ai-aided/
└── human-only/
shared/                     # Common data, model setup, and helper code
ai-interactions/            # GenAI interaction records for both phases
ai_reflection_midpoint.md   # Phase I GenAI reflection
README.md                   # Task, model(s), dataset(s), and project overview
team-info.json              # Group members
workload.json               # Contribution shares
```

Each pair's work is stored in its phase and role folder. The pairs share a task,
model and starting question, then investigate separately. They switch AI-aided
and human-only roles after Phase I. Membership is listed in each phase's README.

The [AI interaction record](ai-interactions/) and
[midpoint reflection](ai_reflection_midpoint.md) document the Phase I use of
GenAI and the comparison between the pairs. The progress video is still to be
recorded. Phase II code, interaction records, the final video and slides are
also pending.

See [GeneralInformation.md](GeneralInformation.md) for assignment requirements.
