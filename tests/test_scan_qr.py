import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import torch

from config import IMAGE_SIZE
from src.model import QRClassifier
from src.scan_qr import build_scan_report


class BuildScanReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.image_path = self.directory / "qr.png"
        self.checkpoint_path = self.directory / "baseline.pth"
        qr_image = cv2.QRCodeEncoder_create().encode("https://example.com/check")
        self.assertTrue(cv2.imwrite(str(self.image_path), qr_image))

        model = QRClassifier(num_classes=2)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": {"fake": 0, "real": 1},
                "image_size": IMAGE_SIZE,
            },
            self.checkpoint_path,
        )

    def test_builds_report_without_exposing_payload_by_default(self) -> None:
        report = build_scan_report(self.image_path, self.checkpoint_path)

        self.assertTrue(report["qr"]["decoded"])
        self.assertNotIn("payload", report["qr"])
        self.assertEqual(
            report["qr"]["url_assessment"]["host"],
            "example.com",
        )
        self.assertIn(
            report["visual_classification"]["predicted_label"],
            {"fake", "real"},
        )
        self.assertEqual(
            set(report["visual_classification"]["scores"]),
            {"fake", "real"},
        )
        self.assertIn("not a security verdict", " ".join(report["limitations"]))
        json.dumps(report)

    def test_can_include_payload_when_explicitly_requested(self) -> None:
        report = build_scan_report(
            self.image_path,
            self.checkpoint_path,
            show_payload=True,
        )

        self.assertEqual(report["qr"]["payload"], "https://example.com/check")

    def test_classifier_report_is_returned_when_qr_cannot_be_decoded(self) -> None:
        blank_image = self.directory / "blank.png"
        self.assertTrue(
            cv2.imwrite(str(blank_image), 255 * torch.ones((224, 224, 3), dtype=torch.uint8).numpy())
        )

        report = build_scan_report(blank_image, self.checkpoint_path)

        self.assertFalse(report["qr"]["decoded"])
        self.assertIn("error", report["qr"])
        self.assertIsNotNone(report["visual_classification"])

    def test_missing_image_fails_clearly(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_scan_report(self.directory / "missing.png", self.checkpoint_path)


if __name__ == "__main__":
    unittest.main()
