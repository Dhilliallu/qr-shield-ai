import unittest
from pathlib import Path

from src.analyze_errors import EXPECTED_CLASSES, REPORT_PATH


class ErrorAnalysisConfigurationTests(unittest.TestCase):
    def test_analysis_uses_expected_classes(self) -> None:
        self.assertEqual(EXPECTED_CLASSES, {"fake", "real"})

    def test_report_target_is_under_reports_directory(self) -> None:
        self.assertEqual(REPORT_PATH.name, "test_error_analysis.csv")
        self.assertEqual(REPORT_PATH.parent.name, "reports")
        self.assertEqual(REPORT_PATH.parent.parent, Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    unittest.main()
