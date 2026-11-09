import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from config import DEVICE, IMAGE_SIZE, MODEL_DIR
from src.model import QRClassifier
from src.url_risk import analyze_url


DEFAULT_CHECKPOINT = MODEL_DIR / "baseline_cnn.pth"
EXPECTED_CLASSES = {"fake", "real"}


def decode_qr_image(image_path: Path) -> str:
    """Decode one QR code from a local image without opening its destination."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")

    detector = cv2.QRCodeDetector()
    decoded_text, _points, _straight_qr = detector.detectAndDecode(image)
    if decoded_text:
        return decoded_text

    height, width = image.shape[:2]
    for scale in (2, 4):
        if max(height, width) * scale > 4096:
            break
        enlarged = cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        decoded_text, _points, _straight_qr = detector.detectAndDecode(enlarged)
        if decoded_text:
            return decoded_text

    raise ValueError(f"No decodable QR code found in: {image_path}")


def load_classifier(checkpoint_path: Path) -> tuple[QRClassifier, dict[str, int]]:
    """Load the trained model and its saved label mapping."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    class_to_idx = checkpoint.get("class_to_idx")
    if (
        not isinstance(class_to_idx, dict)
        or set(class_to_idx) != EXPECTED_CLASSES
        or set(class_to_idx.values()) != set(range(len(EXPECTED_CLASSES)))
    ):
        raise ValueError(
            "Checkpoint must contain a valid fake/real class mapping; "
            f"found {class_to_idx!r}"
        )
    if tuple(checkpoint.get("image_size", ())) != tuple(IMAGE_SIZE):
        raise ValueError(
            f"Checkpoint image size {checkpoint.get('image_size')!r} "
            f"does not match configured size {IMAGE_SIZE!r}."
        )

    model = QRClassifier(num_classes=len(class_to_idx)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_to_idx


def classify_image(
    image_path: Path,
    model: QRClassifier,
    class_to_idx: dict[str, int],
) -> dict[str, Any]:
    """Return visual CNN scores for one image; these are not calibrated."""
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            tensor = transform(image).unsqueeze(0).to(DEVICE)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read image file: {image_path}") from error

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

    index_to_class = {index: name for name, index in class_to_idx.items()}
    scores = {
        index_to_class[index]: float(probabilities[index])
        for index in range(len(index_to_class))
    }
    predicted_label = max(scores, key=scores.__getitem__)
    return {
        "predicted_label": predicted_label,
        "scores": scores,
        "score_note": "Softmax scores are uncalibrated and are not safety probabilities.",
    }


def build_scan_report(
    image_path: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    show_payload: bool = False,
) -> dict[str, Any]:
    """Combine visual image classification with local QR and URL checks."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model, class_to_idx = load_classifier(checkpoint_path)
    return _build_scan_report_with_classifier(
        image_path,
        model,
        class_to_idx,
        show_payload,
    )


def build_scan_reports(
    image_paths: list[Path],
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    show_payload: bool = False,
) -> list[dict[str, Any]]:
    """Scan multiple images while loading the classifier only once."""
    if not image_paths:
        raise ValueError("At least one image is required for a batch scan.")
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

    model, class_to_idx = load_classifier(checkpoint_path)
    return [
        _build_scan_report_with_classifier(
            image_path,
            model,
            class_to_idx,
            show_payload,
        )
        for image_path in image_paths
    ]


def _build_scan_report_with_classifier(
    image_path: Path,
    model: QRClassifier,
    class_to_idx: dict[str, int],
    show_payload: bool,
) -> dict[str, Any]:
    visual_classification = classify_image(image_path, model, class_to_idx)
    try:
        payload = decode_qr_image(image_path)
    except ValueError as error:
        qr_result: dict[str, Any] = {
            "decoded": False,
            "error": str(error),
            "url_assessment": None,
        }
    else:
        url_assessment = analyze_url(payload)
        qr_result = {
            "decoded": True,
            "payload_length": len(payload),
            "url_assessment": {
                "is_web_url": url_assessment.is_web_url,
                "host": url_assessment.host,
                "status": url_assessment.status,
                "warnings": list(url_assessment.warnings),
            },
        }
        if show_payload:
            qr_result["payload"] = payload

    return {
        "image": str(image_path.resolve()),
        "visual_classification": visual_classification,
        "qr": qr_result,
        "limitations": [
            "The visual classifier is a dataset-trained baseline, not a security verdict.",
            "A QR image prediction does not validate its encoded destination.",
            "URL checks are heuristic only; no destination was opened or contacted.",
            "No heuristic warnings do not mean a URL is safe.",
        ],
    }


def build_batch_csv(reports: list[dict[str, Any]]) -> str:
    """Create a payload-free CSV summary of scan reports."""
    fieldnames = (
        "image",
        "cnn_prediction",
        "fake_score",
        "real_score",
        "qr_decoded",
        "url_review",
        "host",
        "warning_count",
        "warnings",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for report in reports:
        visual = report["visual_classification"]
        qr = report["qr"]
        assessment = qr.get("url_assessment") or {}
        warnings = assessment.get("warnings", [])
        writer.writerow(
            {
                "image": report.get("image", ""),
                "cnn_prediction": visual["predicted_label"],
                "fake_score": f"{visual['scores'].get('fake', 0.0):.6f}",
                "real_score": f"{visual['scores'].get('real', 0.0):.6f}",
                "qr_decoded": qr["decoded"],
                "url_review": assessment.get("status", "Not decoded"),
                "host": assessment.get("host") or "",
                "warning_count": len(warnings),
                "warnings": "; ".join(warnings),
            }
        )
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a QR image and locally inspect its decoded URL."
    )
    parser.add_argument("image", type=Path, help="Path to a local QR image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the JSON report",
    )
    parser.add_argument(
        "--show-payload",
        action="store_true",
        help="Include the full decoded QR contents in the report",
    )
    args = parser.parse_args()

    report = build_scan_report(args.image, args.checkpoint, args.show_payload)
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")
        print(f"JSON scan report saved to: {args.output}")
    print(report_json)


if __name__ == "__main__":
    main()
