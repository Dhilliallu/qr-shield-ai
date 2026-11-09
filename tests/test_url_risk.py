import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from src.scan_qr import decode_qr_image
from src.url_risk import analyze_url


class DecodeQrImageTests(unittest.TestCase):
    def test_decodes_small_local_qr_image(self) -> None:
        payload = "https://example.com/check"
        qr_image = cv2.QRCodeEncoder_create().encode(payload)

        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "qr.png"
            self.assertTrue(cv2.imwrite(str(image_path), qr_image))

            self.assertEqual(decode_qr_image(image_path), payload)

    def test_missing_image_raises_a_clear_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            decode_qr_image(Path("missing-qr-image.png"))


class AnalyzeUrlTests(unittest.TestCase):
    def test_https_domain_without_warning_signals(self) -> None:
        result = analyze_url("https://example.com/path")

        self.assertTrue(result.is_web_url)
        self.assertEqual(result.host, "example.com")
        self.assertEqual(result.warnings, ())
        self.assertIn("not verified safe", result.status)

    def test_http_and_embedded_credentials_are_reported(self) -> None:
        result = analyze_url("http://user:password@example.com/login")

        self.assertEqual(
            result.warnings,
            (
                "URL contains embedded username or password information.",
                "Connection uses HTTP rather than encrypted HTTPS.",
            ),
        )
        self.assertEqual(result.status, "Review recommended")

    def test_non_public_ip_and_nonstandard_port_are_reported(self) -> None:
        result = analyze_url("http://192.168.1.10:8080/")

        self.assertIn(
            "Destination uses an IP address instead of a domain name.",
            result.warnings,
        )
        self.assertIn("IP address is non-public or reserved.", result.warnings)
        self.assertIn("URL uses a non-standard web port (8080).", result.warnings)

    def test_domain_without_scheme_is_parsed_as_https(self) -> None:
        result = analyze_url("www.example.com/path")

        self.assertTrue(result.is_web_url)
        self.assertEqual(result.host, "www.example.com")
        self.assertEqual(result.warnings, ())

    def test_plain_text_is_not_mistaken_for_a_web_url(self) -> None:
        result = analyze_url("QR code information")

        self.assertFalse(result.is_web_url)
        self.assertEqual(result.status, "Not a web URL")

    def test_non_web_uri_is_not_treated_as_an_http_url(self) -> None:
        result = analyze_url("javascript:alert(1)")

        self.assertFalse(result.is_web_url)
        self.assertEqual(
            result.status,
            "Non-web URI; do not open automatically",
        )


if __name__ == "__main__":
    unittest.main()
