import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from src.audit_dataset import audit_dataset, decode_qr


class AuditDatasetTests(unittest.TestCase):
    def test_audits_labels_splits_and_duplicate_payloads(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for source, label in (
                ("benign_qr_images_500", "real"),
                ("malicious_qr_images_500", "fake"),
            ):
                source_dir = root / source
                source_dir.mkdir()
                qr = cv2.QRCodeEncoder_create().encode("https://example.com/")
                self.assertTrue(cv2.imwrite(str(source_dir / f"{label}.png"), qr))

            for split in ("train", "val", "test"):
                for label in ("real", "fake"):
                    split_dir = root / split / label
                    split_dir.mkdir(parents=True)
                    qr = cv2.QRCodeEncoder_create().encode("https://example.com/")
                    image = cv2.resize(qr, (224, 224), interpolation=cv2.INTER_NEAREST)
                    self.assertTrue(cv2.imwrite(str(split_dir / f"{label}.png"), image))

            report = audit_dataset(root)

            self.assertEqual(report["source_totals"]["image_count"], 2)
            self.assertEqual(report["source_totals"]["decoded_qr_count"], 2)
            self.assertEqual(report["source_totals"]["undecodable_qr_count"], 0)
            self.assertEqual(
                len(report["duplicate_checks"]["cross_label_identical_payloads"]),
                1,
            )
            self.assertEqual(
                len(
                    report["duplicate_checks"][
                        "repeated_payloads_across_prepared_splits"
                    ]
                ),
                3,
            )
            self.assertFalse(report["source_folder_labels_independently_verified"])
            self.assertFalse(report["network_requests_made"])

    def test_decode_returns_none_for_non_image(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "not-an-image.png"
            image_path.write_text("not an image", encoding="utf-8")

            self.assertIsNone(decode_qr(image_path))

    def test_audit_fails_when_a_source_class_is_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "benign_qr_images_500"
            source_dir.mkdir()
            qr = cv2.QRCodeEncoder_create().encode("https://example.com/")
            self.assertTrue(cv2.imwrite(str(source_dir / "sample.png"), qr))

            with self.assertRaisesRegex(FileNotFoundError, "malicious_qr_images_500"):
                audit_dataset(root)


if __name__ == "__main__":
    unittest.main()
