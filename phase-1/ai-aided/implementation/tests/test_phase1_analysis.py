"""Numerical and provenance checks for the research comparisons."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from implementation import analyze
from implementation import config
from implementation import translation_control as translation


class RetrievalTests(unittest.TestCase):
    def test_ties_and_wrong_candidate_match_full_stable_sort(self):
        sims = np.array([[.8, .8, .8, .2], [.1, .4, .3, .9], [.5, .5, .5, .5]])
        owners = np.array([2, 3, 1])
        result = analyze.rank_similarities(sims, owners, top_k=2)
        expected = np.argsort(-sims, axis=1, kind="stable")
        np.testing.assert_array_equal(result["top_indices"], expected[:, :2])
        np.testing.assert_array_equal(result["rank"], [3, 1, 2])
        np.testing.assert_allclose(result["hard_margin"], [0, .5, 0])
        self.assertTrue(np.all(result["wrong_index"] != owners))

    def test_streamed_rows_and_original_neighbor_indices_with_exclusions(self):
        images = pd.DataFrame({"image_key": ["a", "b", "c"], "locale": ["en", "fr", "de"]})
        captions = pd.DataFrame({"caption_id": [0, 1, 2], "image_key": ["a", "b", "c"],
                                 "caption_lang": ["en", "fr", "de"], "locale": ["en", "fr", "de"]})
        vectors = np.array([[1., 0.], [1., 1.], [0., 1.]])
        with patch.object(config, "UNSUPPORTED_LANGUAGES", ["fr"]):
            table, neighbors, scores = analyze.compute_retrieval(images, captions, vectors, vectors, chunk=1)
        np.testing.assert_array_equal(table.caption_id, [0, 2])
        np.testing.assert_array_equal(table["rank"], [1, 1])
        np.testing.assert_array_equal(neighbors[:, 0], [0, 2])
        self.assertEqual(scores.shape, (2, 2))
        np.testing.assert_allclose(table.margin, 1)

    def test_nonfinite_and_invalid_owners_are_rejected(self):
        with self.assertRaises(ValueError): analyze.rank_similarities([[np.nan, 0]], [0])
        with self.assertRaises(ValueError): analyze.rank_similarities([[1, 0]], [2])


class InferenceTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for locale, base in [("en", 9.), ("de", 1.)]:
            for i in range(4):
                for lang in ["en", "de"]:
                    rows.append(dict(image_key=f"{locale}{i}", locale=locale, caption_lang=lang,
                                     margin=base + .8 * (locale == lang)))
        return pd.DataFrame(rows)

    def test_image_centering_removes_shared_image_difficulty(self):
        results, global_tests = analyze.matching_statistics(self.fixture(), bootstrap=200, permutations=499)
        adjusted = results[results["view"] == "image_centered"]
        np.testing.assert_allclose(adjusted.effect, .8)
        np.testing.assert_allclose(adjusted.ci_low, .8)
        raw = results[results["view"] == "raw"].set_index("caption_lang")
        self.assertGreater(raw.loc["en", "effect"], 8)
        self.assertLess(raw.loc["de", "effect"], -7)
        self.assertLess(global_tests["image_centered"]["p"], .1)
        self.assertTrue(((results.p > 0) & (results.p <= 1)).all())
        self.assertTrue((results.q_bh >= results.p - 1e-12).all())

    def test_results_are_reproducible_and_ignore_input_row_order(self):
        a, g1 = analyze.matching_statistics(self.fixture(), bootstrap=100, permutations=99)
        b, g2 = analyze.matching_statistics(self.fixture().sample(frac=1, random_state=9), bootstrap=100, permutations=99)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(g1, g2)

    def test_incomplete_image_vectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            analyze.matching_statistics(self.fixture().iloc[1:], bootstrap=100, permutations=99)

    def test_multiple_testing_correction_known_values(self):
        np.testing.assert_allclose(analyze.bh_adjust([.01, .04, .03]), [.03, .04, .04])


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.captions = pd.DataFrame([dict(caption_id=7, image_key="a", locale="de", caption_lang="en", text="A white car.")])
        self.frame = pd.DataFrame([dict(pair_id="one", source_caption_id=7, image_key="a", locale="de",
            source_lang="en", source_text="A white car.", target_lang="de", translation="Ein weißes Auto.",
            translator="AI test fixture", review_status="ai_generated")])

    def test_source_alignment_and_review_provenance(self):
        self.assertEqual(len(translation.validate_translations(self.frame, self.captions)), 1)
        changed = self.frame.assign(source_text="An unrelated caption.")
        with self.assertRaisesRegex(ValueError, "does not match"):
            translation.validate_translations(changed, self.captions)
        with self.assertRaisesRegex(ValueError, "unique"):
            translation.validate_translations(pd.concat([self.frame, self.frame]), self.captions)
        with self.assertRaisesRegex(ValueError, "review_status"):
            translation.validate_translations(self.frame.assign(review_status=""), self.captions)
        self.assertTrue(translation.validate_translations(self.frame.assign(review_status="rejected"), self.captions).empty)

    def test_preparation_is_balanced_and_does_not_overwrite(self):
        source = pd.concat([self.captions.assign(caption_id=i, image_key=f"{locale}{i}", locale=locale, caption_idx=0)
                            for locale in ["en", "de"] for i in range(4)], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "captions.parquet"
            source.to_parquet(table)
            path = Path(directory) / "translations.csv"
            with patch.object(config, "CAPTIONS_TABLE", table): translation.prepare(path, per_locale=2)
            sample = pd.read_csv(path, keep_default_na=False)
            self.assertEqual(sample.locale.value_counts().to_dict(), {"de": 2, "en": 2})
            self.assertTrue((sample.translation == "").all())
            with self.assertRaisesRegex(ValueError, "overwrite"): translation.prepare(path, per_locale=2)

    def test_bootstrap_keeps_image_pairs_together(self):
        pairs = pd.DataFrame({"locale": ["de", "de", "en", "en"], "review_status": ["ai_generated"] * 4})
        for metric in ["margin", "recall_at_1"]:
            pairs[f"source_{metric}"] = [0., .5, .2, .7]
            pairs[f"translated_{metric}"] = pairs[f"source_{metric}"] + .1
            pairs[f"human_target_{metric}"] = pairs[f"translated_{metric}"] + .2
        summary = translation.paired_summary(pairs, bootstrap=100)
        row = summary.query("locale == 'all' and metric == 'margin' and comparison == 'translation_minus_source'").iloc[0]
        np.testing.assert_allclose([row.effect, row.ci_low, row.ci_high], .1)
        self.assertEqual(row.n_images, 4)
        self.assertEqual(row.n_human_reviewed, 0)


class ExplorerTests(unittest.TestCase):
    def setUp(self):
        from implementation import explorer
        self.explorer = explorer
        rows = []
        for i, (key, locale) in enumerate([("a", "en"), ("b", "de")]):
            for lang in ["en", "de"]:
                row = dict(caption_id=len(rows), image_key=key, locale=locale, caption_lang=lang,
                           text="A white car." if lang == "en" else "Ein weißes Auto.",
                           rank=1 if lang == locale else 2, margin=.2, hard_margin=.1,
                           recall_at_1=float(lang == locale), tokens=8, truncated=False,
                           top_image_key=key if lang == locale else ("b" if key == "a" else "a"))
                row.update({f"topic_{topic}": topic == "Vehicles" for topic in analyze.TOPICS})
                rows.append(row)
        self.data = dict(queries=pd.DataFrame(rows), images=pd.DataFrame({"image_key": ["a", "b"]}),
                         english={"a": "A white car.", "b": "Another white car."},
                         neighbors=np.tile([0, 1], (4, 1)), cosines=np.tile([.2, .1], (4, 1)),
                         metadata={"records_sha256": "fixture"})
        patcher = patch.object(explorer, "_DATA", self.data)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_filter_and_gallery_render_real_dash_components(self):
        records, summary, selected = self.explorer.filtered_queries("de", "all", "Vehicles", "errors", "hard_margin")
        self.assertEqual(len(records), 1)
        self.assertIn("Before filtering by success/failure", summary)
        detail = self.explorer.retrieval_detail(records, selected)
        self.assertTrue(detail.children)
        self.assertIsInstance(records[0]["hard_margin"], str)
        a, b, text = self.explorer.cue_figures("de")
        self.assertTrue(a.to_json() and b.to_json())
        self.assertIn("0.00%", text)

    def test_reviews_are_persistent_upserts_with_run_provenance(self):
        records, _, selected = self.explorer.filtered_queries("de", "all", "all", "errors", "sample")
        with tempfile.TemporaryDirectory() as directory, patch.object(self.explorer, "REVIEWS", Path(directory) / "reviews.csv"):
            first = self.explorer.save_review(records, selected, "plausible_alternative", "no", "Test reviewer", "Generic caption")
            self.assertIn("Saved", first)
            self.explorer.save_review(records, selected, "unclear", "unclear", "Test reviewer", "Revised interpretation")
            table = pd.read_csv(self.explorer.REVIEWS)
            self.assertEqual(len(table), 1)
            self.assertEqual(table.error_type.iloc[0], "unclear")
            self.assertEqual(table.records_sha256.iloc[0], "fixture")
            self.assertTrue(self.explorer.review_summary().children)


if __name__ == "__main__": unittest.main()
