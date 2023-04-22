"""Tests for FLOTA module based on original implementation."""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

import pytest
import torch

from flota import FlotaMode, FlotaTokenizer

from .utils.file_based_test import FileBasedTest, FunctionType

if TYPE_CHECKING:
    from pathlib import Path


TEST_MODELS_NOT_STRICT = ["bert-base-uncased", "xlnet-base-cased", "gpt2"]
TEST_MODELS_STRICT = ["bert-base-uncased"]
TEST_MODELS = [(model, True) for model in TEST_MODELS_STRICT] + [
    (model, False) for model in TEST_MODELS_NOT_STRICT
]
TEST_MODES = [FlotaMode.FLOTA, FlotaMode.FIRST, FlotaMode.LONGEST]
TEST_K = range(1, 5)
TEST_DATA = ["cs", "maths", "physics"]

assert_equal_tensors = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
"""Custom `assert_equal` function for tensor comparisons. """


@pytest.mark.parametrize("data", TEST_DATA)
@pytest.mark.parametrize("k", TEST_K)
@pytest.mark.parametrize("mode", TEST_MODES, ids=lambda x: x.value)
class TestOriginalFiles(FileBasedTest):
    """Regression tests based on original FLOTA tokenization output.

    This class tests the following methods:
    - FlotaTokenizer.tokenize()
    - FlotaTokenizer.encode()
    - FlotaTokenizer.__call__()

    All test data files have been generated with the original
    implementation of the FLOTA tokenization.

    If any of these tests fail, it is likely that the implementation
    differs from the original implementation.
    """

    @pytest.mark.parametrize(("model", "strict"), TEST_MODELS)
    @pytest.mark.parametrize(
        "function_type", [FunctionType.TOKENIZE], ids=lambda x: x.value
    )
    def test_original_tokenize(
        self,
        flota_tokenizer: FlotaTokenizer,
        input_file: Path,
        output_file: Path,
    ) -> None:
        """Test tokenize method to be equal to original output."""
        expected = json.load(output_file.open("rb"))

        test_input = input_file.read_text().splitlines()
        actual = [flota_tokenizer.tokenize(test_line) for test_line in test_input]

        assert actual == expected

    @pytest.mark.parametrize(("model", "strict"), TEST_MODELS)
    @pytest.mark.parametrize(
        "function_type", [FunctionType.ENCODE], ids=lambda x: x.value
    )
    def test_original_encode(
        self,
        flota_tokenizer: FlotaTokenizer,
        input_file: Path,
        output_file: Path,
    ) -> None:
        """Test encode method to be equal to original output."""
        expected = json.load(output_file.open("rb"))

        test_input = input_file.read_text().splitlines()
        actual = [flota_tokenizer.encode(test_line) for test_line in test_input]

        assert actual == expected

    @pytest.mark.parametrize(("model", "strict"), TEST_MODELS)
    @pytest.mark.parametrize(
        "function_type", [FunctionType.CALL], ids=lambda x: x.value
    )
    def test_original_call(
        self,
        flota_tokenizer: FlotaTokenizer,
        input_file: Path,
        output_file: Path,
    ) -> None:
        """Test __call__ method to be equal to original output."""
        expected = torch.load(output_file.open("rb"))

        test_input = json.load(input_file.open("rb"))
        actual = [flota_tokenizer(test_line) for test_line in test_input]

        assert len(actual) == len(expected)

        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert actual_item.keys() == expected_item.keys()
            assert_equal_tensors(actual_item["input_ids"], expected_item["input_ids"])
            assert_equal_tensors(
                actual_item["attention_mask"], expected_item["attention_mask"]
            )
