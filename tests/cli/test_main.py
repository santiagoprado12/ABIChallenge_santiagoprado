import os
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_train_model():
    with patch("src.ml_core.train.train") as mock_train:
        yield mock_train


@pytest.fixture
def mock_validation_model():
    with patch("src.ml_core.validation.validate") as mock_validate:
        yield mock_validate


@pytest.fixture
def mock_run_makefile():
    with patch("src.cli.main.run_makefile") as mock_run_makefile:
        yield mock_run_makefile


@pytest.fixture
def mock_postgresql_manager():
    with patch(
        "src.cli.main.PostgreSQLManager"
    ) as mock_db_manager:
        yield mock_db_manager


def test_train_command(mock_train_model):
    # Simulate calling the 'train' command
    result = runner.invoke(
        app, ["train", "--model", "random_forest", "--acc-threshold", "0.8"]
    )

    # Verify that the train_model.train function was called correctly
    mock_train_model.assert_called_once_with(["random_forest"], 0.8)
    assert result.exit_code == 0


def test_validation_command_with_threshold(mock_validation_model, mock_run_makefile):
    # Mock the validate function to return a specific score
    mock_validation_model.return_value = 0.5

    # Simulate calling the 'validation' command
    result = runner.invoke(app, ["validation", "--acc-threshold", 0.7])

    # Verify that the validate function was called
    mock_validation_model.assert_called_once()

    mock_run_makefile.assert_called_once_with("train")
    assert result.exit_code == 0


def test_test_command_without_coverage(mock_run_makefile):
    # Simulate calling the 'test' command without coverage
    result = runner.invoke(app, ["test"])

    # Verify that run_makefile was called with 'test'
    mock_run_makefile.assert_called_once_with("test")
    assert result.exit_code == 0


def test_test_command_with_coverage(mock_run_makefile):
    # Simulate calling the 'test' command with --coverage
    result = runner.invoke(app, ["test", "--coverage"])

    # Verify that run_makefile was called with 'test-coverage'
    mock_run_makefile.assert_called_once_with("test-coverage")
    assert result.exit_code == 0


def test_run_sql_command(mock_postgresql_manager):
    # Mock the file system check for file existence
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True

        # Simulate calling the 'run-sql' command
        result = runner.invoke(
            app, ["run-sql", "--sql-file", "tests/cli/moked_sql/test_sql.sql"]
        )

        assert result.exit_code == 0


def test_run_sql_command_file_not_found():
    # Mock the file system check for file existence
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False

        # Simulate calling the 'run-sql' command with a non-existent file
        result = runner.invoke(app, ["run-sql", "--sql-file", "nonexistent.sql"])

        # Verify that the command exits with an error message
        assert "Error: File 'nonexistent.sql' does not exist." in result.output
        assert result.exit_code == 0
