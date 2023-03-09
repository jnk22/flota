"""Data helpers for FLOTA application."""

from enum import Enum


class FlotaMode(Enum):
    """An enum representing the different modes of the FLOTA application."""

    FLOTA = "flota"
    FLOTA_DP_FRONT = "flota-dp-front"
    FLOTA_DP_BACK = "flota-dp-back"
    FIRST = "first"
    LONGEST = "longest"


class TokenizeMode(Enum):
    """Wrapper for `FlotaMode` to be used in FLOTA CLI."""

    BASE = "base"
    FLOTA = FlotaMode.FLOTA.value
    FLOTA_DP_FRONT = FlotaMode.FLOTA_DP_FRONT.value
    FLOTA_DP_BACK = FlotaMode.FLOTA_DP_BACK.value
    FIRST = FlotaMode.FIRST.value
    LONGEST = FlotaMode.LONGEST.value


class NoiseType(Enum):
    """An enum representing different types of noise."""

    NONE = "none"
    TEST = "test"
    TRAIN = "train"

    @property
    def filename_extension(self) -> str:
        """The extension to add to a filename based on the noise type."""
        return f"_noise_{self.value}" if self != NoiseType.NONE else ""


class RunType(Enum):
    """An enum representing different run types."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
