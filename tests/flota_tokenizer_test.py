"""Tests for FLOTA module."""

import itertools

import pytest

from flota import CacheInfo, FlotaMode, FlotaTokenizer
from flota.exceptions import PretrainedTokenizerLoadError, UnsupportedModelError


@pytest.mark.parametrize("cache_size", [-128, -1, 0])
def test_flota_tokenizer_cache_disabled(cache_size: int) -> None:
    """Test various cache sizes for FLOTA tokenizer."""
    flota_tokenizer = FlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA, cache_size=cache_size
    )

    disabled_cache = CacheInfo(0, 0, 0, 0)
    assert flota_tokenizer.cache_info == disabled_cache

    for _ in range(10):
        flota_tokenizer.tokenize("test")

    disabled_cache = CacheInfo(0, 10, 0, 0)
    assert flota_tokenizer.cache_info == disabled_cache


@pytest.mark.parametrize("cache_size", [10, 50, 100])
def test_flota_tokenizer_cache_fixed_without_replacement(cache_size: int) -> None:
    """Test various cache sizes for FLOTA tokenizer."""
    flota_tokenizer = FlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA, cache_size=cache_size
    )

    words = ("test", "visualization", "cachesizetest")

    new_cache = CacheInfo(0, 0, cache_size, 0)
    assert new_cache == flota_tokenizer.cache_info

    first_miss_cache = CacheInfo(0, len(words), cache_size, len(words))
    for word in words:
        flota_tokenizer.tokenize(word)

    assert first_miss_cache == flota_tokenizer.cache_info

    retry_hits_cache = CacheInfo(500 * len(words), len(words), cache_size, len(words))
    for word, _ in itertools.product(words, range(500)):
        flota_tokenizer.tokenize(word)

    assert retry_hits_cache == flota_tokenizer.cache_info


def test_flota_tokenizer_cache_size_unlimited() -> None:
    """Test various cache sizes for FLOTA tokenizer."""
    flota_tokenizer = FlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA, cache_size=None
    )

    unlimited_cache = CacheInfo(0, 0, None, 0)

    assert unlimited_cache == flota_tokenizer.cache_info


def test_flota_dp_words() -> None:
    """Test various words for FLOTA dynamic programming mode."""
    test_word = "visualization"

    expected = ["visual", "##ization"]

    flota_tokenizer = FlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA_DP_BACK, k=3
    )
    actual = flota_tokenizer.tokenize(test_word)

    assert actual == expected


def test_prefix_vocab_order() -> None:
    """Ensure that word order in prefix vocab does not matter."""
    prefix_vocab = ["anti", "a"]
    prefix_vocab_reversed = ["a", "anti"]

    # Assume that prefix 'anti' will be split as prefix regardless of
    # input order of prefix_vocab.
    # If the order of prefixes in prefix_vocab matters, then the output
    # would start with ["a", ...], ignoring 'anti'.
    expected = ["anti", "##pas", "##ti"]

    for vocab in [prefix_vocab, prefix_vocab_reversed]:
        tokenizer = FlotaTokenizer.from_pretrained(
            "distilbert-base-uncased", FlotaMode.FLOTA, k=3, prefix_vocab=vocab
        )

        actual = tokenizer.tokenize("antipasti")

        assert actual == expected


@pytest.mark.parametrize("model", ["bert", "roberta", "xlnet"])
def test_unsupported_pretrained_model_raises_error(model: str) -> None:
    """Raise error when pretrained model does not exist."""
    expected_error_msg = f"Could not load tokenizer from pretrained model '{model}'"

    with pytest.raises(PretrainedTokenizerLoadError, match=expected_error_msg):
        FlotaTokenizer.from_pretrained(model, FlotaMode.FLOTA)


@pytest.mark.parametrize("model", ["xlm-mlm-en-2048", "ctrl", "unknown-model-name"])
def test_unsupported_model_name_raises_error(model: str) -> None:
    """Raise error when unsupported model is specified."""
    expected_error_msg = f"Model '{model}' is not supported by FlotaTokenizer"

    with pytest.raises(UnsupportedModelError, match=expected_error_msg):
        FlotaTokenizer.from_pretrained(model, FlotaMode.FLOTA)


def test_longest_sort_order() -> None:
    """Special test for FLOTA 'longest' mode to ensure sort order."""
    flota_tokenizer = FlotaTokenizer.from_pretrained(
        "bert-base-uncased", FlotaMode.LONGEST, k=4
    )
    assert flota_tokenizer.tokenize("ws1s") == ["w", "##s", "##1", "##s"]


@pytest.mark.parametrize(
    ("input_word", "expected"),
    [
        ("hopeful_men", ["Ġhopeful", "_-", "men"]),
        ("square_omega", ["Ġsquare", "_-", "Ġomega"]),
    ],
)
def test_flota_tokenizer_gpt2_includes_hyphen(
    input_word: str, expected: list[str]
) -> None:
    """Verify that hyphen is included in FLOTA's GPT2 tokenization.

    This case has been found during regression tests with the full
    dataset (arxiv_cs_1e+03), based on the original implementation.
    """
    flota_tokenizer = FlotaTokenizer.from_pretrained("gpt2", FlotaMode.FLOTA, k=4)
    assert flota_tokenizer.tokenize(input_word) == expected
