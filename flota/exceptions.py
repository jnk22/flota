"""FLOTA tokenizer specific exceptions."""


class UnsupportedModelError(ValueError):
    """Raised when an unsupported pre-trained model is specified."""


class PretrainedTokenizerLoadError(ValueError):
    """Raised when the tokenizer could not be loaded from the pre-trained model."""
