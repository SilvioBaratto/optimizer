"""Smoke: the package imports under the workspace env."""


def test_package_imports() -> None:
    import portopt_db

    assert portopt_db.__version__
