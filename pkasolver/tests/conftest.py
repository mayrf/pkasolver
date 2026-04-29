"""
Pytest configuration and shared fixtures for pkasolver tests.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--num-models",
        action="store",
        default=None,
        type=int,
        help="Number of ensemble models to load in QueryModel (overrides default of 25). "
        "Use 1 for fast CI runs.",
    )


@pytest.fixture(scope="session")
def num_models(request):
    """Number of ensemble models to use in tests.

    Defaults to 1 in CI environments (via --num-models 1 or CI=true env var),
    and to the full 25 otherwise.
    """
    import os

    cli_val = request.config.getoption("--num-models")
    if cli_val is not None:
        return cli_val
    if os.getenv("CI") == "true":
        return 1
    return 25


@pytest.fixture(scope="session")
def query_model(num_models):
    """Shared QueryModel instance for the test session."""
    from pkasolver.query import QueryModel

    return QueryModel(num_models=num_models)
