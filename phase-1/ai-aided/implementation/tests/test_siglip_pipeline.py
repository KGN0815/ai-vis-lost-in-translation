"""Offline integration checks using a tiny, randomly initialized SigLIP model.

These exercise real Transformers APIs and synthetic data, not research results.
Run from ai-aided/: python -m unittest discover -s tests -v
"""
from contextlib import redirect_stderr, redirect_stdout
import importlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import sentencepiece as spm
import torch
import torch.nn.functional as F
from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipProcessor, SiglipTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from implementation import artifacts
from implementation import compute_scores
from implementation import config
from implementation import embed
from implementation import prepare_data
from implementation.xm3600_reader import read_records


class SiglipPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        cls.assets = tempfile.TemporaryDirectory()
        asset_dir = Path(cls.assets.name)
        corpus = asset_dir / "corpus.txt"
        corpus.write_text("A DOG ON A BEACH\na dog on a beach\nein hund am strand\n"
                          "un chien sur la plage\nein bild und noch ein bild\n", encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(asset_dir / "tokenizer"), vocab_size=64,
            hard_vocab_limit=False, bos_id=-1, eos_id=1, unk_id=0, minloglevel=2,
        )
        tokenizer = SiglipTokenizer(vocab_file=str(asset_dir / "tokenizer.model"))
        cls.processor = SiglipProcessor(
            image_processor=SiglipImageProcessor(size={"height": 256, "width": 256}),
            tokenizer=tokenizer,
        )
        cls.model = SiglipModel(SiglipConfig(
            text_config={"vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 64},
            vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                           "num_attention_heads": 2, "image_size": 256, "patch_size": 16},
        )).eval()

    @classmethod
    def tearDownClass(cls):
        cls.assets.cleanup()

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        root = Path(self.workspace.name)
        self.raw = root / "xm3600"
        self.raw.mkdir()
        processed = root / "processed"
        paths = {
            "PROCESSED_DIR": processed, "XM3600_DIR": self.raw, "IMAGES_DIR": self.raw,
            "CAPTIONS_JSONL": self.raw / "captions.jsonl", "EMBEDDING_DIM": 16,
            "LANGUAGES": {"de": "German", "en": "English", "fr": "French"},
        }
        for name, filename in {
            "CAPTIONS_TABLE": "captions.parquet", "IMAGES_TABLE": "images.parquet",
            "IMAGE_EMB": "image_embeddings.npy", "CAPTION_EMB": "caption_embeddings.npy",
            "CAPTION_SCORES": "caption_scores.parquet", "IMAGE_LANG_TABLE": "image_language.parquet",
            "MATRICES_FILE": "matrices.parquet", "EMBEDDING_METADATA": "embedding_metadata.json",
            "SCORE_METADATA": "score_metadata.json",
        }.items():
            paths[name] = processed / filename
        config_patch = patch.multiple(config, **paths)
        config_patch.start()
        self.addCleanup(config_patch.stop)
        chunk_patch = patch.object(embed, "CHUNK", 5)
        chunk_patch.start()
        self.addCleanup(chunk_patch.stop)
        records = []
        for i, locale in enumerate(config.LANGUAGES):
            key = f"image_{i}"
            photo = Image.new("L" if i == 0 else "RGB", (30 + i, 20 + i))
            photo.save(self.raw / f"{key}.jpg")
            photo.close()
            records.append({
                "image/key": key, "image/locale": locale,
                "de": {"caption": ["EIN HUND AM STRAND", "Noch ein Bild"]},
                "en": {"caption": ["A DOG ON A BEACH"]},
                "fr": {"caption": ["Un chien sur la plage"]},
            })
        config.CAPTIONS_JSONL.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
        )
        self.addCleanup(lambda: sys.modules.pop("implementation.app", None))

    def test_encoding_matches_direct_siglip_features(self):
        encoder = embed.SiglipEmbedder(self.model, self.processor, torch.device("cpu"))
        texts = ["A DOG ON A BEACH", "UN CHIEN " * 100]
        with redirect_stdout(StringIO()):
            vectors = encoder.captions(texts)
            images = encoder.images(["image_0", "image_1"])
        inputs = self.processor(text=[text.lower() for text in texts], padding="max_length",
                                max_length=64, truncation=True, return_tensors="pt")
        self.assertEqual(inputs["input_ids"].shape, (2, 64))
        with torch.no_grad():
            expected = F.normalize(self.model.get_text_features(**inputs), dim=-1).numpy()
        np.testing.assert_allclose(vectors, expected, atol=5e-4)
        self.assertEqual(vectors.dtype, np.float16)
        self.assertEqual(images.shape, (2, 16))
        np.testing.assert_allclose(np.linalg.norm(images.astype(np.float32), axis=1), 1, atol=5e-4)
        self.assertTrue(np.isfinite(vectors).all())

    def test_pipeline_workers_scores_and_dashboard(self):
        with self.assertRaisesRegex(RuntimeError, "embedding_metadata"):
            compute_scores.main()
        with redirect_stdout(StringIO()):
            prepare_data.main()
        initial_files = sorted(config.PROCESSED_DIR.iterdir())
        with patch("transformers.AutoModel.from_pretrained", return_value=self.model) as model_loader, \
             patch("transformers.AutoProcessor.from_pretrained", return_value=self.processor) as processor_loader, \
             redirect_stdout(StringIO()):
            with patch.object(sys, "argv", ["embed.py", "--test", "--device", "cpu"]):
                embed.main()
            self.assertEqual(sorted(config.PROCESSED_DIR.iterdir()), initial_files)
            with patch.object(sys, "argv", ["embed.py", "--part", "0/2", "--device", "cpu"]):
                embed.main()
            self.assertFalse(config.EMBEDDING_METADATA.exists())
            with patch.object(sys, "argv", ["embed.py", "--part", "1/2", "--device", "cpu"]):
                embed.main()
            self.assertFalse(config.EMBEDDING_METADATA.exists())
            model_loader.assert_called_with(
                "google/siglip2-base-patch16-256", revision="3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab",
            )
            processor_loader.assert_called_with(
                "google/siglip2-base-patch16-256", revision="3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab",
                use_fast=False,
            )
            # A complete run can resume solely from its chunks, without loading a model.
            before_merge = model_loader.call_count
            with patch.object(sys, "argv", ["embed.py", "--merge"]):
                embed.main()
            self.assertEqual(model_loader.call_count, before_merge)
            compute_scores.main()

        captions = pd.read_parquet(config.CAPTIONS_TABLE)
        images = pd.read_parquet(config.IMAGES_TABLE)
        scores = pd.read_parquet(config.CAPTION_SCORES)
        image_vectors = np.load(config.IMAGE_EMB).astype(np.float32)
        caption_vectors = np.load(config.CAPTION_EMB).astype(np.float32)
        self.assertEqual(caption_vectors.shape, (12, 16))
        owners = images.set_index("image_key").index.get_indexer(captions.image_key)
        for i, owner in enumerate(owners):
            similarities = image_vectors @ caption_vectors[i]
            margin = similarities[owner] - np.delete(similarities, owner).mean()
            self.assertAlmostEqual(float(scores.margin.iloc[i]), float(margin), places=6)
        matrices = pd.read_parquet(config.MATRICES_FILE)
        self.assertEqual(len(matrices), 27)
        double = matrices[matrices.state == "double"].pivot(index="caption_lang", columns="locale", values="value")
        np.testing.assert_allclose(double.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(double.mean(axis=1), 0, atol=1e-7)

        dashboard = importlib.import_module("implementation.app")
        client = dashboard.app.server.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        with client.get("/img/image_0") as response:
            self.assertEqual(response.status_code, 200)
        for state in ["raw", "row", "double"]:
            for grouping in ["family", "culture"]:
                figure = dashboard.build_figure(state, grouping)
                self.assertEqual(np.asarray(figure.data[0].z).shape, (3, 3))
                self.assertTrue(figure.to_json())
        cell, _ = dashboard.select_cell({"points": [{"x": 0, "y": 0}]}, "family")
        self.assertEqual(cell, {"caption_lang": "en", "locale": "en"})
        self.assertTrue(dashboard.show_examples(cell).children)

        # Same-sized tables with a different order must not silently score wrong pairs.
        captions.iloc[::-1].to_parquet(config.CAPTIONS_TABLE, index=False)
        with self.assertRaisesRegex(ValueError, "row order"):
            compute_scores.main()

    def test_incompatible_artifacts_and_changed_run_cannot_be_reused(self):
        images, captions = read_records()
        metadata = artifacts.run_metadata(images, captions)
        other_order = artifacts.run_metadata(images, list(reversed(captions)))
        self.assertNotEqual(embed.chunk_dir_for(metadata), embed.chunk_dir_for(other_order))
        with patch.object(config, "MODEL_REVISION", "another-revision"):
            other_checkpoint = artifacts.run_metadata(images, captions)
        self.assertNotEqual(embed.chunk_dir_for(metadata), embed.chunk_dir_for(other_checkpoint))
        artifacts.write_metadata(config.EMBEDDING_METADATA, {**metadata, "model": "another-model"})
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            compute_scores.main()
        wrong = Path(self.workspace.name) / "wrong.npy"
        np.save(wrong, np.zeros((3, 2048), dtype=np.float16))
        with self.assertRaisesRegex(ValueError, "expected"):
            embed.check_array(wrong, 3)

    def test_invalid_worker_is_rejected(self):
        with patch.object(sys, "argv", ["embed.py", "--part", "2/2"]), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as caught:
                embed.main()
        self.assertEqual(caught.exception.code, 2)

    def test_explicit_intel_gpu_request_never_falls_back_to_cpu(self):
        with patch.object(torch.xpu, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Intel GPU is unavailable"):
                embed.pick_device("xpu")


if __name__ == "__main__":
    unittest.main()
