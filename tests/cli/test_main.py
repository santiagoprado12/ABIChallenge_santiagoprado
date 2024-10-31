"""Module to test the CLI."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_train_model():
    """Fixture to mock the train function in the train module."""
    with patch("src.ml_core.train.train") as mock_train:
        yield mock_train


@pytest.fixture
def mock_validation_model():
    """Fixture to mock the validate function in the validation module."""
    with patch("src.ml_core.validation.validate") as mock_validate:
        yield mock_validate


@pytest.fixture
def mock_run_makefile():
    """Fixture to mock the run_makefile function in the main CLI module."""
    with patch("src.cli.main.run_makefile") as mock_run_makefile:
        yield mock_run_makefile


@pytest.fixture
def mock_postgresql_manager():
    """Fixture to mock the PostgreSQLManager class in the main CLI module."""
    with patch("src.cli.main.PostgreSQLManager") as mock_db_manager:
        yield mock_db_manager


def test_train_command(mock_train_model):
    """Test the 'train' CLI command to verify it invokes the train function.

    Args:
        mock_train_model: Mocked train function.
    """
    result = runner.invoke(
        app, ["train", "--model", "random_forest", "--acc-threshold", "0.8"]
    )

    # Verify that the train function was called with the correct parameters
    mock_train_model.assert_called_once_with(["random_forest"], 0.8)
    assert result.exit_code == 0


def test_validation_command_with_threshold(mock_validation_model, mock_run_makefile):
    """Test the 'validation' CLI command to check threshold-based validation.

    Args:
        mock_validation_model: Mocked validation function.
        mock_run_makefile: Mocked run_makefile function.
    """
    mock_validation_model.return_value = 0.5

    result = runner.invoke(app, ["validation", "--acc-threshold", 0.7])

    # Verify the validation function call and conditional training invocation
    mock_validation_model.assert_called_once()
    mock_run_makefile.assert_called_once_with("train")
    assert result.exit_code == 0


def test_test_command_without_coverage(mock_run_makefile):
    """Test the 'test' CLI command to verify it runs tests without coverage.

    Args:
        mock_run_makefile: Mocked run_makefile function.
    """
    result = runner.invoke(app, ["test"])

    # Verify that run_makefile was called with 'test'
    mock_run_makefile.assert_called_once_with("test")
    assert result.exit_code == 0


def test_test_command_with_coverage(mock_run_makefile):
    """Test the 'test' CLI command with the --coverage option.

    Args:
        mock_run_makefile: Mocked run_makefile function.
    """
    result = runner.invoke(app, ["test", "--coverage"])

    # Verify that run_makefile was called with 'test-coverage'
    mock_run_makefile.assert_called_once_with("test-coverage")
    assert result.exit_code == 0


def test_run_sql_command(mock_postgresql_manager):
    """Test the 'run-sql' CLI command to verify SQL file execution.

    Args:
        mock_postgresql_manager: Mocked PostgreSQLManager class.
    """
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True

        result = runner.invoke(
            app, ["run-sql", "--sql-file", "tests/cli/moked_sql/test_sql.sql"]
        )

        assert result.exit_code == 0


def test_run_sql_command_file_not_found():
    """Test the 'run-sql' CLI command with a non-existent file path."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False

        result = runner.invoke(app, ["run-sql", "--sql-file", "nonexistent.sql"])

        # Verify that the command exits with an error message
        assert "Error: File 'nonexistent.sql' does not exist." in result.output
        assert result.exit_code == 0
