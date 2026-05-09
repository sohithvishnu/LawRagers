"""Unit tests for CAP ingest batching helpers."""

from scripts.ingest_cap import batched


class TestBatched:
    def test_groups_items_by_requested_batch_size(self):
        groups = list(batched([1, 2, 3, 4, 5], size=2))

        assert groups == [[1, 2], [3, 4], [5]]

    def test_rejects_non_positive_batch_size(self):
        try:
            list(batched([1, 2], size=0))
        except ValueError as exc:
            assert "positive" in str(exc)
        else:
            raise AssertionError("Expected ValueError for non-positive batch size")
