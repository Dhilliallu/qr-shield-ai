import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATASET_DIR, DEVICE, IMAGE_SIZE, MODEL_DIR
from src.model import QRClassifier
from src.train import calculate_metrics


TEST_DIR = DATASET_DIR / "test"
CHECKPOINT_PATH = MODEL_DIR / "baseline_cnn.pth"
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "test_error_analysis.csv"
EXPECTED_CLASSES = {"fake", "real"}


def analyze_test_errors() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {CHECKPOINT_PATH}")

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    if not test_dataset.samples:
        raise ValueError(f"No test images found in {TEST_DIR}")
    if set(test_dataset.classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected test classes {sorted(EXPECTED_CLASSES)}, "
            f"found {test_dataset.classes}"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    class_to_idx = test_dataset.class_to_idx
    if checkpoint.get("class_to_idx") != class_to_idx:
        raise ValueError(
            "The checkpoint class mapping does not match the test dataset: "
            f"{checkpoint.get('class_to_idx')} != {class_to_idx}"
        )

    model = QRClassifier(num_classes=len(class_to_idx)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    index_to_class = {index: name for name, index in class_to_idx.items()}
    confusion = torch.zeros((len(class_to_idx), len(class_to_idx)), dtype=torch.int64)
    rows: list[dict[str, object]] = []
    sample_offset = 0

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1).cpu()
            predictions = probabilities.argmax(dim=1)

            for batch_index, (true_index, predicted_index) in enumerate(
                zip(labels.tolist(), predictions.tolist())
            ):
                sample_path, _ = test_dataset.samples[sample_offset + batch_index]
                probabilities_for_image = probabilities[batch_index]
                is_correct = true_index == predicted_index
                confusion[true_index, predicted_index] += 1
                rows.append(
                    {
                        "image_path": Path(sample_path).relative_to(
                            Path(__file__).resolve().parent.parent
                        ),
                        "true_label": index_to_class[true_index],
                        "predicted_label": index_to_class[predicted_index],
                        "correct": is_correct,
                        "predicted_confidence": float(
                            probabilities_for_image[predicted_index]
                        ),
                        "fake_probability": float(
                            probabilities_for_image[class_to_idx["fake"]]
                        ),
                        "real_probability": float(
                            probabilities_for_image[class_to_idx["real"]]
                        ),
                    }
                )
            sample_offset += len(labels)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "true_label",
        "predicted_label",
        "correct",
        "predicted_confidence",
        "fake_probability",
        "real_probability",
    ]
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = calculate_metrics(confusion, class_to_idx)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test images analyzed: {len(rows)}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print("Confusion matrix (rows=true, columns=predicted; class index order):")
    print(f"  {class_to_idx}")
    print(confusion.tolist())
    print("Errors by direction:")
    for true_label, predicted_label in (("fake", "real"), ("real", "fake")):
        errors = [
            row
            for row in rows
            if row["true_label"] == true_label
            and row["predicted_label"] == predicted_label
        ]
        print(f"  {true_label} predicted as {predicted_label}: {len(errors)}")
        for row in sorted(errors, key=lambda item: float(item["predicted_confidence"])):
            print(
                f"    {row['image_path']} "
                f"(predicted confidence {float(row['predicted_confidence']):.1%})"
            )
    print(f"Per-image predictions saved to: {REPORT_PATH}")


if __name__ == "__main__":
    analyze_test_errors()
