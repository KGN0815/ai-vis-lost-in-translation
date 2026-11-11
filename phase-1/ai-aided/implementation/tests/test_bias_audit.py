"""Offline tests of balanced sampling, paired exposure and audit provenance."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from implementation import bias_audit as audit
from implementation import config


def label_fixture(per_stratum=5):
    rows = []
    for race in audit.RACES:
        for gender in audit.GENDERS:
            for age in audit.AGES:
                for _ in range(per_stratum):
                    rows.append(dict(source_row=len(rows), image_key=f"image-{len(rows)}", race=race, gender=gender, age=age))
    return pd.DataFrame(rows)


def prompt_fixture():
    rows = []
    for kind, families in [("neutral", ["person", "portrait", "face"]), ("control", ["car", "tree"])]:
        for family in families:
            source = f"A photo of {family}."
            for language in ["en", "de"]:
                rows.append(dict(prompt_id=f"{language}-{family}", family=family, kind=kind,
                                 language=language, text=source if language == "en" else f"Ein Foto: {family}.",
                                 source_text=source, review_status="source_original" if language == "en" else "ai_generated",
                                 translator="Test fixture"))
    return pd.DataFrame(rows)


class CandidateTests(unittest.TestCase):
    def test_balanced_reproducible_sampling_and_randomized_common_order(self):
        labels = label_fixture()
        selected = audit.sample_candidates(labels, per_stratum=4)
        self.assertEqual(len(selected), 112)
        self.assertTrue(selected.groupby(audit.AXES).size().eq(4).all())
        self.assertFalse(selected.source_row.duplicated().any())
        pd.testing.assert_frame_equal(selected, audit.sample_candidates(labels, per_stratum=4))
        self.assertFalse(selected.source_row.is_monotonic_increasing)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            audit.sample_candidates(labels[labels.race != "Black"], per_stratum=4)
        with self.assertRaisesRegex(ValueError, "distinct"):
            audit.sample_candidates(pd.concat([labels, labels.iloc[:1]]), per_stratum=4)

    def test_changed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "PROCESSED_DIR", Path(directory)):
            manifest = audit.output("candidates.csv")
            manifest.write_text("tampered\n")
            audit.output("candidate_metadata.json").write_text(json.dumps({
                "dataset_revision": audit.REVISION, "source_sha256": audit.SOURCE_SHA256,
                "manifest_sha256": "different",
            }))
            with self.assertRaisesRegex(ValueError, "changed"):
                audit.load_candidates()

    def test_changed_prompt_review_file_requires_rescoring(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "PROCESSED_DIR", Path(directory)):
            prompt_path = Path(directory) / "prompts.csv"
            prompt_path.write_text("Edited translation/review record\n")
            audit.output("metadata.json").write_text(json.dumps({
                **audit.model_settings(), "audit_version": audit.VERSION,
                "manifest_sha256": "matching", "prompts_sha256": "previous",
            }))
            with patch.object(audit, "PROMPTS", prompt_path), patch.object(
                    audit, "load_candidates", return_value=(pd.DataFrame(), {"manifest_sha256": "matching"})):
                with self.assertRaisesRegex(ValueError, "Rescore"):
                    audit.load_results()


class ExposureTests(unittest.TestCase):
    def test_share_baseline_discount_and_full_rank_have_known_values(self):
        scores = np.array([[1., 2., 3., 6., 5., 4.], [6., 1., 2., 5., 4., 3.]])
        values = {row["group"]: row for row in audit.group_exposure(scores, np.array(["A"] * 3 + ["B"] * 3), 2)}
        np.testing.assert_allclose(values["B"]["share"], [1., .5])
        self.assertEqual(values["A"]["baseline"], .5)
        np.testing.assert_allclose(values["A"]["mean_rank"], [5., 4.])
        self.assertAlmostEqual(values["B"]["discounted_share"][1], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
        with self.assertRaises(ValueError): audit.group_exposure(scores, ["A"] * 6, 7)
        with self.assertRaises(ValueError): audit.group_exposure(np.array([[np.nan, 0]]), ["A", "B"], 1)

    def test_identical_language_scores_have_zero_paired_difference(self):
        candidates = audit.sample_candidates(label_fixture(), 4)
        prompts = audit.validate_prompts(prompt_fixture())
        rng = np.random.default_rng(1)
        scores = np.empty((len(prompts), len(candidates)))
        for family, group in prompts.groupby("family"):
            scores[group.index] = rng.normal(size=len(candidates))
        per_prompt, means = audit.exposure_tables(scores, prompts, candidates)
        np.testing.assert_allclose(means.delta_english_pp, 0)
        np.testing.assert_allclose(means.groupby(["language", "axis", "k"]).share.sum(), 1)
        np.testing.assert_allclose(means.groupby(["language", "axis", "k"]).excess_pp.sum(), 0, atol=1e-12)
        a, n = audit.subsample_stability(scores, prompts, candidates, repetitions=4)
        b, _ = audit.subsample_stability(scores, prompts, candidates, repetitions=4)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(n, 84)
        np.testing.assert_allclose(a[["delta_stability_low", "delta_stability_high"]], 0)

    def test_stable_ties_follow_shared_candidate_order(self):
        result = audit.group_exposure(np.zeros((2, 4)), ["A", "B", "B", "A"], 1)
        a = next(row for row in result if row["group"] == "A")
        np.testing.assert_allclose(a["share"], 1)


class PromptTests(unittest.TestCase):
    def test_complete_equivalent_prompt_grid_and_provenance_required(self):
        prompts = prompt_fixture()
        self.assertEqual(len(audit.validate_prompts(prompts)), 10)
        with self.assertRaisesRegex(ValueError, "per language"):
            audit.validate_prompts(prompts.iloc[1:])
        with self.assertRaisesRegex(ValueError, "source text"):
            audit.validate_prompts(prompts.assign(source_text="Changed source"))
        with self.assertRaisesRegex(ValueError, "provenance"):
            audit.validate_prompts(prompts.assign(review_status=""))
        with self.assertRaisesRegex(ValueError, "original English"):
            audit.validate_prompts(prompts.assign(review_status="source_original"))


class DashboardTests(unittest.TestCase):
    def test_tab_controls_and_portrait_route_use_audit_outputs(self):
        from dash import Dash
        from implementation import bias_dashboard as dashboard
        candidates = audit.sample_candidates(label_fixture(), 4)
        prompts = audit.validate_prompts(prompt_fixture())
        scores = np.random.default_rng(10).normal(size=(len(prompts), len(candidates)))
        per_prompt, exposure = audit.exposure_tables(scores, prompts, candidates)
        stability, subset_n = audit.subsample_stability(scores, prompts, candidates, repetitions=2)
        exposure = exposure.merge(stability, on=["language", "axis", "group", "k"])
        data = dict(candidates=candidates, queries=prompts, similarities=scores,
                    per_prompt=per_prompt, exposure=exposure, sanity=audit.sanity_table(scores, prompts, candidates),
                    metadata=dict(n_candidates=len(candidates), n_neutral_families=3, languages=["en", "de"],
                                  n_unreviewed_translations=5, subsample_repetitions=2, subsample_candidates=subset_n))
        with patch.object(dashboard, "load_results", return_value=data):
            app = Dash(__name__, suppress_callback_exceptions=True)
            app.layout = dashboard.audit_tab()
            dashboard.register_callbacks(app)
            for axis in audit.AXES:
                for k in audit.KS:
                    for reference in ["pool", "english"]:
                        figure = dashboard.heatmap(axis, k, reference)
                        self.assertEqual(np.asarray(figure.data[0].z).shape, (2, candidates[axis].nunique()))
                        self.assertTrue(np.isfinite(np.asarray(figure.data[0].z)).all())
            cards, records, text = dashboard.detail("de", "person", "race", 10)
            self.assertEqual(len(cards), 2)
            self.assertEqual(len(records), 7)
            self.assertIn("ai_generated", text)
            self.assertEqual(len(dashboard.sanity_figure().data[0].y), 2)
            client = app.server.test_client()
            self.assertEqual(client.get("/_dash-layout").status_code, 200)
            with tempfile.TemporaryDirectory() as directory, patch.object(dashboard, "data_dir", return_value=Path(directory)):
                images = Path(directory) / "images"
                images.mkdir()
                (images / "test.jpg").write_bytes(b"portrait route fixture")
                with client.get("/fairface/test") as response:
                    self.assertEqual(response.data, b"portrait route fixture")
                self.assertEqual(client.get("/fairface/missing").status_code, 404)
                (Path(directory) / "outside.jpg").write_bytes(b"outside image must not be served")
                with client.get("/fairface/../outside") as response:
                    # Dash's catch-all can return its HTML shell for an unknown URL.
                    self.assertNotIn(b"outside image must not be served", response.data)


if __name__ == "__main__": unittest.main()
