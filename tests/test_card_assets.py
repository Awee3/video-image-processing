import unittest
from pathlib import Path


class CardAssetsTest(unittest.TestCase):
    def test_model_labels_match_training_folders_and_game_artwork(self):
        root = Path(__file__).resolve().parents[1]
        labels = (root / "model" / "class_mapping.txt").read_text().splitlines()
        folders = sorted(p.name for p in (root / "dataset_kaggle").iterdir() if p.is_dir())
        faces = {p.stem for p in (root / "assets" / "cards").glob("*.jpg")}

        self.assertEqual(len(folders), 52)
        self.assertEqual(labels, [f"{i}: {name}" for i, name in enumerate(folders)])
        self.assertEqual(faces, set(folders))
