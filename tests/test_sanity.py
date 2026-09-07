"""Sanity test to confirm the package imports and pytest collects successfully."""

import hgc


def test_package_imports_with_version():
    assert hgc.__version__ == "0.1.0"
