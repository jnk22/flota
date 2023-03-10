"""Module providing base class for FLOTA I/O based tests."""

import os
import sys
from enum import Enum
from pathlib import Path

import pytest

from flota import FlotaMode, FlotaTokenizer


class FunctionType(Enum):
    """Enum for tested function type, used for file paths."""

    TOKENIZE = "tokenize"  # noqa: S105
    ENCODE = "encode"
    CALL = "call"


class FileBasedTest:
    """Base class for I/O based file tests."""

    @property
    def _test_data_path(self) -> Path:
        # Get the path of the module that contains the test class.
        # The test directory must be named after that file.
        module_path = sys.modules[type(self).__module__].__file__
        assert module_path is not None

        return Path(os.path.relpath(module_path)).with_suffix("")

    @pytest.fixture()
    @staticmethod
    def flota_tokenizer(
        model: str, mode: FlotaMode, k: int, *, strict: bool
    ) -> FlotaTokenizer:
        """Return FlotaTokenizer based on input parameters."""
        return FlotaTokenizer.from_pretrained(model, mode, k=k, strict=strict)

    @pytest.fixture()
    def input_file(self, function_type: FunctionType, data: str) -> Path:
        """Input file path based on tested function and dataset."""
        file_suffix = ".json" if function_type == FunctionType.CALL else ".txt"
        file_name = f"{data}{file_suffix}"

        return self._build_file("input", function_type.value, file_name)

    @pytest.fixture()
    def output_file(  # noqa: PLR0913
        self,
        function_type: FunctionType,
        model: str,
        mode: FlotaMode,
        k: int,
        data: str,
        *,
        strict: bool,
    ) -> Path:
        """Output file path based on tested function, dataset and tokenizer."""
        file_strict = "_strict" if strict else ""
        file_suffix = ".pt" if function_type == FunctionType.CALL else ".json"
        file_name = f"{model}_{mode.value}_{k}{file_strict}_{data}{file_suffix}"

        return self._build_file("output", function_type.value, file_name)

    def _build_file(self, io_type: str, function_type: str, file_name: str) -> Path:
        file = Path(f"{self._test_data_path}/{io_type}/{function_type}/{file_name}")

        assert file.exists()

        return file
