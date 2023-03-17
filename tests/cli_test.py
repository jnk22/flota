"""Tests for CLI."""

import re

from typer.testing import CliRunner

from flota.cli import cli

runner = CliRunner()


def test_cli_tokenize() -> None:
    """Test `tokenize` command."""
    result = runner.invoke(cli, ["tokenize", "bert-base-uncased", "visualization"])
    assert result.exit_code == 0
    assert "vis ##ua ##lization\n" in result.stdout


def test_cli_encode() -> None:
    """Test `encode` command."""
    result = runner.invoke(cli, ["encode", "bert-base-uncased", "visualization"])
    assert result.exit_code == 0
    assert "101 25292 6692 22731 102\n" in result.stdout


def test_cli_print_version() -> None:
    """Test version argument."""
    result = runner.invoke(cli, ["--version"])
    version_regex = r"\d+\.\d+\.\d+"  # x.y.z version format
    assert result.exit_code == 0
    assert "FLOTA CLI version: " in result.stdout
    assert re.search(version_regex, result.stdout) is not None
