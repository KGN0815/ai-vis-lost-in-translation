# Phase II: What did the model learn?

**Research question:** Which image regions support the score differences
between languages found in Phase I?

**Status:** Concept only. Both LIME and Integrated Gradients remain to be
implemented. There are no explanation results yet.

Investigate the same SigLIP 2 model and Crossmodal 3600 images used in Phase I.

- [AI-aided pair](ai-aided/): Jakob Schwarzmaier (K12325928) and William Johann Strouhal (K12240477), the Phase I human-only pair.
- [Human-only pair](human-only/): Klaus Niederberger (K12342339) and Markus Brunner (K12141831), the Phase I AI-aided pair.

Place each pair's code, notebooks, results, and notes in its role folder.

## Planned analysis

1. Apply **LIME and Integrated Gradients** to the same images with captions in
   different languages. Both pairs plan to use both techniques independently,
   meeting the assignment's expectation of at least two techniques per pair.
2. Compare whether the explanations highlight similar objects or different
   background regions when the caption language changes. Explain the same
   similarity score with both methods and record the method settings.
3. Use a balanced set of cases across languages and image locales, including
   cases where Phase I finds an advantage, no clear advantage, or a reversal.
   Document how cases were selected.
4. Compare where the methods agree and disagree across these cases, and relate
   the explanations to Phase I's systematic patterns.

Keep each image fixed when comparing captions in different languages. Account
for the [caption-content and locale caveats](../phase-1/README.md#dataset-interpretation)
when interpreting differences in highlighted regions.

## Pair coordination

- Before starting: agree on the questions, data, initial visualization, and
  Phase I findings to investigate. Both pairs keep the same task and model.
- During analysis: each pair uses at least two explanation or representation
  analysis techniques and compares or combines their insights to explain
  behaviors or differences between input groups.
- At the end: compare findings, visualization choices, breadth and depth, time
  and verification effort, and confidence supported by evidence. Reflect on
  the role switch.

Save influential AI interactions in [../ai-interactions/](../ai-interactions/).

## Final submission

- Complete code from both pairs for Phase I and Phase II.
- [AI interaction records](../ai-interactions/) from both phases.
- Recorded presentation (10 minutes): not yet recorded.
- Slide deck: not yet prepared.

Keep the slide deck and supporting presentation files directly in this folder.
The presentation covers the problem and motivation, methods and findings from
both pairs for Phase I then Phase II, AI4Vis reflection, and conclusions.
