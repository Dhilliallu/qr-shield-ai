import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import torch
from config import IMAGE_SIZE
from src.model import QRClassifier
from src.evaluate_external import (
    check_no_training_overlap,
    evaluate_external_dataset,
    image_hashes,
    validate_external_root,
)


class ValidateExternalRootTests(unittest.TestCase):
    def test_requires_fake_and_real_class_folders(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "fake").mkdir()
            (root / "real").mkdir()

            self.assertEqual(validate_external_root(root), root.resolve())

    def test_rejects_project_dataset_subdirectory(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_external_root(Path(__file__).resolve().parents[1] / "dataset" / "test")

    def test_rejects_missing_class_folder(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "fake").mkdir()

            with self.assertRaisesRegex(ValueError, "exactly"):
                validate_external_root(root)


class ExternalImageHashTests(unittest.TestCase):
    def test_detects_duplicate_images_inside_external_dataset(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "fake").mkdir()
            (root / "real").mkdir()
            (root / "fake" / "one.png").write_bytes(b"identical image bytes")
            (root / "real" / "two.png").write_bytes(b"identical image bytes")

            with self.assertRaisesRegex(ValueError, "Exact duplicate"):
                image_hashes(root)

    def test_rejects_exact_overlap_with_project_images(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            external_image = root / "external" / "fake" / "external.png"
            project_image = root / "project" / "fake" / "project.png"
            external_image.parent.mkdir(parents=True)
            project_image.parent.mkdir(parents=True)
            external_image.write_bytes(b"same image data")
            project_image.write_bytes(b"same image data")

            with self.assertRaisesRegex(ValueError, "overlap"):
                check_no_training_overlap(
                    image_hashes(external_image.parents[1]),
                    comparison_dirs=(project_image.parents[1],),
                )


class EvaluateExternalDatasetTests(unittest.TestCase):
    def test_evaluates_class_folders_without_training(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "external"
            checkpoint_path = root / "baseline.pth"
            for class_name, payload in (
                ("fake", "https://fake.example/"),
                ("real", "https://real.example/"),
            ):
                class_dir = dataset_root / class_name
                class_dir.mkdir(parents=True)
                image = cv2.QRCodeEncoder_create().encode(payload)
                self.assertTrue(cv2.imwrite(str(class_dir / "sample.png"), image))

            model = QRClassifier(num_classes=2)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": {"fake": 0, "real": 1},
                    "image_size": IMAGE_SIZE,
                },
                checkpoint_path,
            )

            report = evaluate_external_dataset(dataset_root, checkpoint_path)

            self.assertEqual(report["sample_count"], 2)
            self.assertEqual(report["class_counts"], {"fake": 1, "real": 1})
            self.assertEqual(len(report["predictions"]), 2)
            self.assertIn("accuracy", report)
            self.assertTrue(
                report["independence_checks"]["no_exact_hash_overlap_with_project_images"]
            )
            self.assertFalse(
                report["independence_checks"][
                    "external_provenance_or_label_correctness_verified"
                ]
            )


if __name__ == "__main__":
    unittest.main()
