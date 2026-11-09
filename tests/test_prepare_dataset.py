import unittest
from pathlib import Path

from src.prepare_dataset import split_items


class SplitItemsTests(unittest.TestCase):
    def test_splits_500_images_into_expected_counts(self) -> None:
        items = [Path(f"image_{index}.png") for index in range(500)]

        splits = split_items(items)

        self.assertEqual(
            {name: len(paths) for name, paths in splits.items()},
            {"train": 350, "val": 75, "test": 75},
        )

    def test_split_is_reproducible(self) -> None:
        items = [Path(f"image_{index}.png") for index in range(20)]

        self.assertEqual(split_items(items), split_items(items))

    def test_splits_do_not_lose_or_repeat_items(self) -> None:
        items = [Path(f"image_{index}.png") for index in range(50)]

        splits = split_items(items)
        combined = splits["train"] + splits["val"] + splits["test"]

        self.assertCountEqual(combined, items)
        self.assertEqual(len(combined), len(set(combined)))


if __name__ == "__main__":
    unittest.main()
