import tempfile
import unittest
from pathlib import Path

import numpy as np

from deeplob_data import CachedArrayView, SlidingWindowDataset, cache_text_dataset


def _write_matrix(path, matrix):
    with open(path, "w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(" ".join(str(float(value)) for value in row))
            handle.write("\n")


class DeepLOBDataTests(unittest.TestCase):
    def test_cache_text_dataset_creates_float32_memmap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            matrix = np.arange(24, dtype=np.float64).reshape(6, 4)
            txt_path = tmp_path / "sample.txt"
            cache_path = tmp_path / "sample.npy"
            _write_matrix(txt_path, matrix)

            cache_info = cache_text_dataset(txt_path, cache_path, dtype=np.float32)

            self.assertEqual(cache_info.shape, matrix.shape)
            self.assertEqual(cache_info.dtype, np.dtype(np.float32))
            loaded = np.load(cache_info.cache_path, mmap_mode="r")
            self.assertEqual(loaded.dtype, np.float32)
            np.testing.assert_allclose(loaded, matrix.astype(np.float32))

    def test_sliding_window_dataset_reads_views_without_precomputing_all_windows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rows = 45
            cols = 8
            matrix = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
            txt_path = tmp_path / "lob.txt"
            cache_path = tmp_path / "lob.npy"
            _write_matrix(txt_path, matrix)
            cache_text_dataset(txt_path, cache_path)

            source = np.load(cache_path, mmap_mode="r")
            view = CachedArrayView(source, start_col=1, end_col=7)
            dataset = SlidingWindowDataset(view=view, k=2, num_classes=3, T=3)

            self.assertEqual(len(dataset), 4)

            sample_x, sample_y = dataset[0]
            self.assertEqual(sample_x.shape, (1, 3, 40))
            sample_x_dtype = getattr(sample_x.dtype, "name", str(sample_x.dtype))
            sample_y_dtype = getattr(sample_y.dtype, "name", str(sample_y.dtype))
            self.assertIn("float32", sample_x_dtype)
            self.assertIn("int64", sample_y_dtype)

            expected_x = matrix[:40, 1:4].T.astype(np.float32)
            np.testing.assert_allclose(sample_x[0], expected_x)

            expected_y = int(matrix[-5 + 2, 3] - 1)
            self.assertEqual(sample_y.item(), expected_y)


if __name__ == "__main__":
    unittest.main()
