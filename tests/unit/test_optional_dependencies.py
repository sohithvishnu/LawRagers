"""Unit tests for optional dependency checks."""

import importlib

from retriever_service.main import warn_if_missing_pysbd


class TestOptionalDependencies:
    def test_warn_if_missing_pysbd_no_warning_when_installed(self, recwarn):
        warn_if_missing_pysbd(import_module=importlib.import_module)

        assert not recwarn

    def test_pysbd_is_installed(self):
        import pysbd  # noqa: F401
