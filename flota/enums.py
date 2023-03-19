"""Data helpers for FLOTA application."""

from enum import Enum


class FlotaMode(Enum):
    """An enum representing the different modes of the FLOTA application."""

    FLOTA = "flota"
    FLOTA_DP = "flota-dp"
    FIRST = "first"
    LONGEST = "longest"


class TokenizeMode(Enum):
    """Wrapper for `FlotaMode` to be used in FLOTA CLI."""

    BASE = "base"
    FLOTA = FlotaMode.FLOTA.value
    FLOTA_DP = FlotaMode.FLOTA_DP.value
    FIRST = FlotaMode.FIRST.value
    LONGEST = FlotaMode.LONGEST.value


class ResultFileExistsMode(Enum):
    """Mode for handling existing results file."""

    SKIP = "skip"
    APPEND = "append"
    OVERWRITE = "overwrite"


class NoiseType(Enum):
    """An enum representing different types of noise."""

    NONE = "none"
    TEST = "test"
    TRAIN = "train"


class RunType(Enum):
    """An enum representing different run types."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
