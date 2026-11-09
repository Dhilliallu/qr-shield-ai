import unittest

import torch

from src.model import QRClassifier
from src.train import calculate_metrics


class QRClassifierTests(unittest.TestCase):
    def test_outputs_one_score_per_class(self) -> None:
        model = QRClassifier(num_classes=2)
        images = torch.randn(4, 3, 224, 224)

        output = model(images)

        self.assertEqual(tuple(output.shape), (4, 2))
        self.assertTrue(torch.isfinite(output).all().item())


class CalculateMetricsTests(unittest.TestCase):
    def test_calculates_accuracy_and_class_metrics(self) -> None:
        confusion = torch.tensor([[8, 2], [1, 9]])
        class_to_idx = {"fake": 0, "real": 1}

        metrics = calculate_metrics(confusion, class_to_idx)

        self.assertEqual(metrics["accuracy"], 0.85)
        self.assertAlmostEqual(metrics["per_class"]["fake"]["precision"], 8 / 9)
        self.assertEqual(metrics["per_class"]["real"]["recall"], 0.9)

    def test_rejects_empty_confusion_matrix(self) -> None:
        with self.assertRaises(ValueError):
            calculate_metrics(torch.zeros((2, 2), dtype=torch.int64), {"fake": 0, "real": 1})


if __name__ == "__main__":
    unittest.main()
