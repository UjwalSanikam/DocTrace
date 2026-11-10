"""Lightweight provenance and integrity verification tests."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import torch
from finalize_release import evaluate, ProvenanceError, read
from export_checkpoint import main as export_main


class ProvenanceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="doctrace_prov_test_"))
        self.run_dir = self.temp_dir / "run"
        self.run_dir.mkdir(parents=True)
        self.output_dir = self.temp_dir / "eval_output"

        # Create a valid dummy checkpoint
        self.checkpoint_path = self.run_dir / "ckpt.pt"
        self.dummy_state = {
            "model": {},
            "config": {"model": "small", "batch_size": 4, "split_limitation": "test"},
            "epoch": 8,
        }
        torch.save(self.dummy_state, self.checkpoint_path)
        self.ckpt_hash = sha256(self.checkpoint_path.read_bytes()).hexdigest()

        # Validation indices
        self.val_indices = [10, 20, 30]
        self.val_indices_hash = sha256(json.dumps(self.val_indices).encode()).hexdigest()
        (self.run_dir / "validation_indices.json").write_text(json.dumps(self.val_indices), encoding="utf-8")

        # Create matching summary.json
        self.output_dir.mkdir(parents=True)
        self.valid_summary = {
            "suite": "validation",
            "checkpoint_epoch": 8,
            "checkpoint_sha256": self.ckpt_hash,
            "index_manifest_sha256": self.val_indices_hash,
            "threshold": 0.5,
            "batch_size": 4,
            "protocol": (
                "Frozen weights, native 512x512 images, no tuning. "
                "JPEG uses Pillow quality and 4:2:0 subsampling; "
                "resize is bicubic down/up to original grid; "
                "blur radius 0.7. Masks unchanged."
            ),
            "conditions": {
                "clean": {"macro_dice": 0.5, "count": 3},
                "jpeg90": {"macro_dice": 0.4, "count": 3},
            },
        }
        (self.output_dir / "summary.json").write_text(json.dumps(self.valid_summary), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evaluate_passes_on_valid_provenance(self):
        report = evaluate(
            self.run_dir,
            "ckpt.pt",
            self.output_dir,
            suite="validation",
            conditions=["clean", "jpeg90"],
            threshold=0.5,
        )
        self.assertEqual(report["checkpoint_sha256"], self.ckpt_hash)

    def test_evaluate_detects_corrupted_checkpoint_hash(self):
        # Tamper with checkpoint file
        self.checkpoint_path.write_bytes(b"tampered content")
        with self.assertRaises(ProvenanceError) as ctx:
            evaluate(
                self.run_dir,
                "ckpt.pt",
                self.output_dir,
                suite="validation",
                conditions=["clean", "jpeg90"],
                threshold=0.5,
            )
        self.assertIn("Checkpoint SHA-256 mismatch", str(ctx.exception))

    def test_evaluate_detects_changed_threshold(self):
        self.valid_summary["threshold"] = 0.75
        (self.output_dir / "summary.json").write_text(json.dumps(self.valid_summary), encoding="utf-8")
        with self.assertRaises(ProvenanceError) as ctx:
            evaluate(
                self.run_dir,
                "ckpt.pt",
                self.output_dir,
                suite="validation",
                conditions=["clean", "jpeg90"],
                threshold=0.5,
            )
        self.assertIn("Threshold mismatch", str(ctx.exception))

    def test_evaluate_detects_stale_manifest_hash(self):
        # Update validation indices on disk without updating summary
        (self.run_dir / "validation_indices.json").write_text(json.dumps([1, 2, 3, 4]), encoding="utf-8")
        with self.assertRaises(ProvenanceError) as ctx:
            evaluate(
                self.run_dir,
                "ckpt.pt",
                self.output_dir,
                suite="validation",
                conditions=["clean", "jpeg90"],
                threshold=0.5,
            )
        self.assertIn("Index manifest hash mismatch", str(ctx.exception))

    def test_evaluate_detects_sample_count_mismatch(self):
        self.valid_summary["conditions"]["clean"]["count"] = 999
        (self.output_dir / "summary.json").write_text(json.dumps(self.valid_summary), encoding="utf-8")
        with self.assertRaises(ProvenanceError) as ctx:
            evaluate(
                self.run_dir,
                "ckpt.pt",
                self.output_dir,
                suite="validation",
                conditions=["clean", "jpeg90"],
                threshold=0.5,
            )
        self.assertIn("Sample count mismatch", str(ctx.exception))

    def test_evaluate_detects_missing_checkpoint(self):
        self.checkpoint_path.unlink()
        with self.assertRaises(ProvenanceError) as ctx:
            evaluate(
                self.run_dir,
                "ckpt.pt",
                self.output_dir,
                suite="validation",
                conditions=["clean", "jpeg90"],
                threshold=0.5,
            )
        self.assertIn("Candidate checkpoint missing", str(ctx.exception))

    def test_evaluate_detects_missing_validation_manifest(self):
        (self.run_dir / "validation_indices.json").unlink()
        with self.assertRaises(ProvenanceError) as ctx:
            evaluate(
                self.run_dir,
                "ckpt.pt",
                self.output_dir,
                suite="validation",
                conditions=["clean", "jpeg90"],
                threshold=0.5,
            )
        self.assertIn("Required validation manifest missing", str(ctx.exception))

    def test_evaluate_detects_altered_protocol_parameters(self):
        self.valid_summary["protocol"] = "Altered JPEG quality 85, 4:4:4 subsampling"
        (self.output_dir / "summary.json").write_text(json.dumps(self.valid_summary), encoding="utf-8")
        with self.assertRaises(ProvenanceError) as ctx:
            evaluate(
                self.run_dir,
                "ckpt.pt",
                self.output_dir,
                suite="validation",
                conditions=["clean", "jpeg90"],
                threshold=0.5,
            )
        self.assertIn("Protocol/transformation parameter mismatch", str(ctx.exception))

    def test_evaluate_detects_altered_batch_size(self):
        self.valid_summary["batch_size"] = 16
        (self.output_dir / "summary.json").write_text(json.dumps(self.valid_summary), encoding="utf-8")
        with self.assertRaises(ProvenanceError) as ctx:
            evaluate(
                self.run_dir,
                "ckpt.pt",
                self.output_dir,
                suite="validation",
                conditions=["clean", "jpeg90"],
                threshold=0.5,
            )
        self.assertIn("Batch size mismatch", str(ctx.exception))

    def test_evaluate_detects_suite_mismatch(self):
        self.valid_summary["suite"] = "fcd"
        (self.output_dir / "summary.json").write_text(json.dumps(self.valid_summary), encoding="utf-8")
        with self.assertRaises(ProvenanceError) as ctx:
            evaluate(
                self.run_dir,
                "ckpt.pt",
                self.output_dir,
                suite="validation",
                conditions=["clean", "jpeg90"],
                threshold=0.5,
            )
        self.assertIn("Suite mismatch", str(ctx.exception))

    def test_export_checkpoint_validates_required_keys(self):
        bad_ckpt = self.temp_dir / "bad.pt"
        torch.save({"only_model": {}}, bad_ckpt)
        out_dir = self.temp_dir / "bad_export"
        import sys
        old_argv = sys.argv
        sys.argv = ["export_checkpoint.py", str(bad_ckpt), "--output", str(out_dir)]
        try:
            with self.assertRaises(ValueError) as ctx:
                export_main()
            self.assertIn("missing required key", str(ctx.exception))
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
