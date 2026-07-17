import unittest
import json
import os
import tempfile

import numpy as np

from src.data import (
    FoldPreprocessor,
    apply_eeg_control,
    fold_indices,
    load_multimodal_data,
)
from src.features import N_FAMILIES, feature_names


class DataTests(unittest.TestCase):
    def test_sentence_folds_are_disjoint_and_complete(self):
        labels = np.tile(np.arange(3), 20)
        seen = []
        for train, val, test in fold_indices(labels, n_folds=5, val_size=0.2, seed=42):
            self.assertFalse(set(train) & set(val))
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(val) & set(test))
            seen.extend(test.tolist())
        self.assertEqual(sorted(seen), list(range(len(labels))))

    def test_preprocessor_uses_training_rows_only(self):
        eeg = np.array(
            [
                [[1.0, np.nan]],
                [[3.0, 5.0]],
                [[1000.0, 1000.0]],
            ],
            dtype=np.float32,
        )
        mask = np.ones((3, 1), dtype=bool)
        processor = FoldPreprocessor().fit(eeg, mask, np.array([0, 1]))
        self.assertAlmostEqual(float(processor.median[0]), 2.0)
        self.assertAlmostEqual(float(processor.median[1]), 5.0)

    def test_preprocessor_zeros_features_missing_from_training_fold(self):
        eeg = np.array(
            [
                [[1.0, np.nan]],
                [[3.0, np.nan]],
                [[5.0, 900.0]],
            ],
            dtype=np.float32,
        )
        mask = np.ones((3, 1), dtype=bool)
        processor = FoldPreprocessor().fit(eeg, mask, np.array([0, 1]))
        transformed = processor.transform(eeg, mask)
        self.assertEqual(processor.n_all_missing_features, 1)
        np.testing.assert_array_equal(transformed[..., 1], 0.0)

    def test_shuffle_stays_inside_each_split(self):
        eeg = np.arange(12, dtype=np.float32).reshape(6, 1, 2)
        mask = np.ones((6, 1), dtype=bool)
        splits = (np.array([0, 1, 2]), np.array([3]), np.array([4, 5]))
        shuffled, shuffled_mask = apply_eeg_control(eeg, mask, splits, "shuffled", 42)
        for split in splits:
            before = sorted(eeg[split, 0, 0].tolist())
            after = sorted(shuffled[split, 0, 0].tolist())
            self.assertEqual(before, after)
        np.testing.assert_array_equal(mask, shuffled_mask)

    def test_subject_caches_align_by_sentence_id(self):
        with tempfile.TemporaryDirectory() as directory:
            labels_path = os.path.join(directory, "labels.csv")
            with open(labels_path, "w") as handle:
                handle.write(
                    "sentence_id,sentence,sentiment_label\n"
                    "10,negative sentence,-1\n"
                    "20,neutral sentence,0\n"
                    "30,positive sentence,1\n"
                )
            names = feature_names(105)
            with open(os.path.join(directory, "feature_names.json"), "w") as handle:
                json.dump(names, handle)
            width = 105 * N_FAMILIES
            features = np.ones((3, width), dtype=np.float32)
            features[:, 2] = np.nan
            np.savez_compressed(
                os.path.join(directory, "S1.npz"),
                X=features,
                sentence_id=np.array([30, 10, 20]),
                label=np.array([1, -1, 0]),
            )
            data = load_multimodal_data(labels_path, directory)
            self.assertEqual(data.labels.tolist(), [0, 1, 2])
            self.assertEqual(tuple(data.eeg.shape), (3, 1, 104 * N_FAMILIES))
            self.assertEqual(data.n_channels, 104)
            self.assertTrue(data.subject_mask.all())
            self.assertEqual(
                data.summary()["dropped_reference_channel"],
                {"index": 104, "label": "Cz", "n_features": N_FAMILIES},
            )
            self.assertEqual(data.summary()["n_globally_all_missing_features"], 1)
            self.assertEqual(
                data.summary()["globally_all_missing_features"], [names[2]]
            )


if __name__ == "__main__":
    unittest.main()
