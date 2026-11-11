# Phase I GenAI reflection

| AI-aided pair: first name | Last name | Student ID |
| --- | --- | --- |
| Klaus | Niederberger | K12342339 |
| Markus | Brunner | K12141831 |

## How AI influenced the work

We used Claude and Codex to discuss the analysis, write and debug code, and
check the results. Two suggestions shaped the XM3600 investigation: comparing
the similarity to the intended image with the average similarity to other images,
and removing language and locale means from the heatmap. We adopted both,
chose the final views, and checked the suggested cultural groupings against
their source. Some group assignments needed correction.

The work grew from the heatmap into retrieval examples, uncertainty estimates
and a pilot translating English captions into German. We later added the
FairFace audit as a separate experiment using the same model. It asks how query language
changes the portraits retrieved from a fixed pool, while the original
investigation asks about caption language and image locale on XM3600.
Both sets of AI-generated translations still need independent review.

## Comparison with the human-only pair

The human-only notebook used the same model and dataset, cosine similarity,
retrieval rank, a heatmap normalized by language, and a concrete Japanese car example.
It reported higher cosine for matching locales in 34 of 36 languages. The AI-aided
analysis likewise found positive margin differences for matching locales in 34 of 36
languages, but the two measures differ and their numerical estimates should not
be treated as interchangeable. The AI-aided work added more controls and
interactive views. We did not record either pair's time or verification effort
consistently, so we cannot tell how much AI changed the time needed.

## Team communication

One of our main lessons from Phase I is that we need more discussion before
implementation. AI makes it easy to turn a new idea into code and results
quickly, before everyone has had time to discuss the direction. We would have
benefited from agreeing more clearly on the main question, priorities and
which experiments were additions to the original work.

For Phase II, we would discuss changes of direction with the whole team before
expanding the work. Each pair should still develop its own analysis while
keeping everyone informed about what it is testing and why. We need to give
those conversations time, even when the code is quick to write.

## Failures and practices

The code and its checks both needed review. The GPU setup initially failed
because of the OpenCL configuration, a dashboard issue appeared during a live
view check, and a hand-calculated test expectation was wrong. We checked model
settings and sources, compared GPU output with a CPU reference, recomputed
sample scores, ran tests and inspected the visualizations.

For future work, independently review the pilot and audit translations, inspect
semantic retrieval errors, and compare the pairs' effort and confidence directly.
The [interaction record](ai-interactions/phase-1-ai-aided.md) summarizes the
influential decisions and checks.
