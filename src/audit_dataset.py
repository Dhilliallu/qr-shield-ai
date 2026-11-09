import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from config import DATASET_DIR, IMAGE_SIZE


SOURCE_CLASSES = {
    "benign_qr_images_500": "real",
    "malicious_qr_images_500": "fake",
}
SPLITS = ("train", "val", "test")
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "dataset_audit.json"


def decode_qr(image_path: Path) -> str | None:
    """Decode a QR locally, without following or contacting its payload."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    detector = cv2.QRCodeDetector()
    payload, _points, _straight_qr = detector.detectAndDecode(image)
    if payload:
        return payload

    height, width = image.shape[:2]
    for scale in (2, 4):
        if max(height, width) * scale > 4096:
            break
        enlarged = cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        payload, _points, _straight_qr = detector.detectAndDecode(enlarged)
        if payload:
            return payload
    return None


def image_fingerprints(image_path: Path) -> tuple[str, str] | None:
    """Return file and decoded-pixel hashes for a readable image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    file_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    pixel_hash = hashlib.sha256(
        repr(image.shape).encode("ascii") + image.tobytes()
    ).hexdigest()
    return file_hash, pixel_hash


def find_duplicate_groups(
    paths_by_key: dict[str, list[str]],
) -> list[dict[str, Any]]:
    return [
        {"count": len(paths), "images": sorted(paths)}
        for paths in paths_by_key.values()
        if len(paths) > 1
    ]


def audit_dataset(dataset_dir: Path = DATASET_DIR) -> dict[str, Any]:
    """Summarize QR decoding, duplicate, payload-overlap, and size risks."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    source_paths: dict[str, list[Path]] = {}
    for source_dir, label in SOURCE_CLASSES.items():
        folder = dataset_dir / source_dir
        if not folder.is_dir():
            raise FileNotFoundError(f"Source class folder not found: {folder}")
        source_paths[label] = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
        )
        if not source_paths[label]:
            raise ValueError(f"No supported images found in {folder}")

    label_stats: dict[str, dict[str, Any]] = {}
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    pixel_hashes: dict[str, list[str]] = defaultdict(list)
    payload_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    split_payloads: dict[str, set[str]] = {split: set() for split in SPLITS}
    sizes: dict[str, set[tuple[int, int]]] = defaultdict(set)
    unreadable: list[str] = []
    undecodable: list[str] = []
    prepared_counts: dict[str, dict[str, int]] = {}
    prepared_invalid_size: list[str] = []
    payload_count = 0

    for label, paths in source_paths.items():
        decoded_count = 0
        web_payload_count = 0
        unreadable_label_count = 0
        for path in paths:
            relative_path = path.relative_to(dataset_dir).as_posix()
            fingerprints = image_fingerprints(path)
            if fingerprints is None:
                unreadable.append(relative_path)
                unreadable_label_count += 1
                continue
            exact_hash, pixel_hash = fingerprints
            exact_hashes[exact_hash].append(relative_path)
            pixel_hashes[pixel_hash].append(relative_path)

            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                height, width = image.shape[:2]
                sizes[label].add((width, height))

            payload = decode_qr(path)
            if payload is None:
                undecodable.append(relative_path)
                continue

            decoded_count += 1
            payload_count += 1
            payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            payload_records[payload_hash].append({"label": label, "image": relative_path})
            if payload.lstrip().lower().startswith(("http://", "https://", "www.")):
                web_payload_count += 1

            for split in SPLITS:
                prepared_path = dataset_dir / split / label / f"{path.stem}.png"
                if prepared_path.is_file():
                    split_payloads[split].add(payload_hash)

        label_stats[label] = {
            "source_image_count": len(paths),
            "decoded_qr_count": decoded_count,
            "undecodable_qr_count": len(paths) - decoded_count - unreadable_label_count,
            "unreadable_image_count": unreadable_label_count,
            "web_like_payload_count": web_payload_count,
            "source_dimensions": [
                {"width": width, "height": height}
                for width, height in sorted(sizes[label])
            ],
        }

    duplicates = find_duplicate_groups(exact_hashes)
    pixel_duplicates = find_duplicate_groups(pixel_hashes)
    cross_label_payloads = []
    for payload_hash, records in payload_records.items():
        labels = sorted({record["label"] for record in records})
        if len(labels) > 1:
            cross_label_payloads.append(
                {
                    "payload_sha256": payload_hash,
                    "labels": labels,
                    "images": sorted(record["image"] for record in records),
                }
            )

    repeated_payloads_across_splits = []
    for first_index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[first_index + 1 :]:
            shared = split_payloads[first_split] & split_payloads[second_split]
            if shared:
                repeated_payloads_across_splits.append(
                    {
                        "first_split": first_split,
                        "second_split": second_split,
                        "shared_payload_count": len(shared),
                        "payload_sha256": sorted(shared),
                    }
                )

    for split in SPLITS:
        prepared_counts[split] = {}
        for label in SOURCE_CLASSES.values():
            folder = dataset_dir / split / label
            files = sorted(folder.glob("*")) if folder.is_dir() else []
            prepared_counts[split][label] = len(
                [path for path in files if path.is_file()]
            )
            for path in files:
                if not path.is_file():
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or tuple(image.shape[:2][::-1]) != tuple(IMAGE_SIZE):
                    prepared_invalid_size.append(path.relative_to(dataset_dir).as_posix())

    source_classes_match_target = set(SOURCE_CLASSES.values()) == {"fake", "real"}
    report: dict[str, Any] = {
        "dataset_directory": str(dataset_dir),
        "source_label_mapping": SOURCE_CLASSES,
        "source_folder_labels_independently_verified": False,
        "network_requests_made": False,
        "source_totals": {
            "image_count": sum(len(paths) for paths in source_paths.values()),
            "decoded_qr_count": payload_count,
            "unreadable_image_count": len(unreadable),
            "undecodable_qr_count": len(undecodable),
        },
        "by_label": label_stats,
        "duplicate_checks": {
            "exact_file_duplicate_groups": duplicates,
            "pixel_identical_duplicate_groups": pixel_duplicates,
            "cross_label_identical_payloads": cross_label_payloads,
            "repeated_payloads_across_prepared_splits": repeated_payloads_across_splits,
        },
        "prepared_split_counts": prepared_counts,
        "prepared_images_with_wrong_size_or_read_error": prepared_invalid_size,
        "unreadable_source_images": unreadable,
        "undecodable_qr_images": undecodable,
        "limitations": [
            "Folder names were treated as labels; payload decoding cannot prove that a destination is benign or malicious.",
            "No QR destination was opened, resolved, or contacted.",
            "Exact file and decoded-pixel hashes do not detect resized, recompressed, or otherwise modified duplicates.",
            "A repeated decoded payload does not by itself establish that the source labels are wrong.",
        ],
    }

    expected_source_counts = {"fake": 500, "real": 500}
    report["source_counts_match_expected"] = (
        source_classes_match_target
        and {
            label: len(paths) for label, paths in source_paths.items()
        } == expected_source_counts
    )
    return report


def main() -> None:
    report = audit_dataset()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Dataset audit saved to: {REPORT_PATH}")
    print(f"Source image count: {report['source_totals']['image_count']}")
    print(
        "QR decoded: "
        f"{report['source_totals']['decoded_qr_count']} / "
        f"{report['source_totals']['image_count']}"
    )
    print(
        "Unreadable source images: "
        f"{report['source_totals']['unreadable_image_count']}"
    )
    print(
        "Exact duplicate groups: "
        f"{len(report['duplicate_checks']['exact_file_duplicate_groups'])}"
    )
    print(
        "Pixel-identical duplicate groups: "
        f"{len(report['duplicate_checks']['pixel_identical_duplicate_groups'])}"
    )
    print(
        "Cross-label identical QR payloads: "
        f"{len(report['duplicate_checks']['cross_label_identical_payloads'])}"
    )
    print(
        "Repeated payloads across prepared splits: "
        f"{len(report['duplicate_checks']['repeated_payloads_across_prepared_splits'])}"
    )
    print("Source labels independently verified: No")
    print("Network requests made: No")


if __name__ == "__main__":
    main()
