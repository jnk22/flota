"""Tests for FLOTA module for generated test files."""

import json
from pathlib import Path  # noqa: TCH003

import pytest

from flota import FlotaMode, FlotaTokenizer

from .utils.file_based_test import FileBasedTest, FunctionType

TEST_MODELS = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "xlnet-base-cased",
    "gpt2",
]
TEST_MODES = [FlotaMode.FLOTA_DP_FRONT, FlotaMode.FLOTA_DP_BACK]
TEST_K = range(5)
TEST_DATA = ["cs", "maths", "physics"]


@pytest.mark.parametrize("data", TEST_DATA)
@pytest.mark.parametrize("strict", [False])
@pytest.mark.parametrize("k", TEST_K)
@pytest.mark.parametrize("mode", TEST_MODES, ids=lambda x: x.value)
@pytest.mark.parametrize("model", TEST_MODELS)
class TestGeneratedFiles(FileBasedTest):
    """Regression tests based on generated FLOTA tokenization output."""

    @pytest.mark.parametrize(
        "function_type", [FunctionType.TOKENIZE], ids=lambda x: x.value
    )
    def test_generated_tokenize(
        self,
        flota_tokenizer: FlotaTokenizer,
        input_file: Path,
        output_file: Path,
    ) -> None:
        """Test tokenize method to be equal to generated output."""
        expected = json.load(output_file.open("rb"))

        test_input = input_file.read_text().splitlines()
        actual = [flota_tokenizer.tokenize(test_line) for test_line in test_input]

        assert actual == expected
