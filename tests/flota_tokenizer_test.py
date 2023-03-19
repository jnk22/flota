"""Tests for FLOTA module."""

import itertools

import pytest

from flota import AutoFlotaTokenizer, CacheInfo, FlotaMode
from flota.exceptions import PretrainedTokenizerLoadError, UnsupportedModelError


@pytest.mark.parametrize("cache_size", [-128, -1, 0])
def test_flota_tokenizer_cache_disabled(cache_size: int) -> None:
    """Test various cache sizes for FLOTA tokenizer."""
    flota_tokenizer = AutoFlotaTokenizer.from_pretrained(
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
    flota_tokenizer = AutoFlotaTokenizer.from_pretrained(
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
    flota_tokenizer = AutoFlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA, cache_size=None
    )

    unlimited_cache = CacheInfo(0, 0, None, 0)

    assert unlimited_cache == flota_tokenizer.cache_info


def test_flota_dp_words() -> None:
    """Test various words for FLOTA dynamic programming mode."""
    test_word = "visualization"

    expected = ["visual", "##ization"]

    flota_tokenizer = AutoFlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA_DP
    )
    actual = flota_tokenizer.tokenize(test_word)

    assert actual == expected


@pytest.mark.parametrize(
    ("word", "k", "prefixes", "expected"),
    [
        ("simplify", 1, ["sim"], ["sim", "##plify"]),
        ("undesirable", 1, ["un"], ["un", "desirable"]),
        ("unbelievable", 1, ["de", "un"], ["unbelievable"]),  # exists in model's vocab
        ("deescalation", 1, ["de", "un"], ["de", "##lation"]),
        ("deescalation", 3, ["de", "un"], ["de", "##es", "##ca", "##lation"]),
    ],
)
def test_prefixes_used(
    word: str, k: int, prefixes: list[str], expected: list[str]
) -> None:
    """Ensure that prefixes are used."""
    tokenizer = AutoFlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA, k=k, prefix_vocab=prefixes
    )

    assert tokenizer.tokenize(word) == expected


@pytest.mark.parametrize(
    ("word", "k", "suffixes", "expected"),
    [
        ("simplify", 1, ["plify"], ["sim", "##plify"]),
        ("simplify", 1, ["ify"], ["sim", "##ify"]),
        ("undesirable", 1, ["able"], ["und", "##able"]),
        ("visualization", 1, ["zation"], ["visual", "##zation"]),
        ("undesirable", 3, ["able"], ["und", "##e", "##sir", "##able"]),
    ],
)
def test_suffixes_used(
    word: str, k: int, suffixes: list[str], expected: list[str]
) -> None:
    """Ensure that suffixes are used."""
    tokenizer = AutoFlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA, k=k, suffix_vocab=suffixes
    )

    assert tokenizer.tokenize(word) == expected


@pytest.mark.parametrize(
    ("word", "k", "prefixes", "suffixes"),
    [
        ("unbelievable", 3, ["de"], ["able"]),
        ("desirable", 3, ["de", "un"], ["able"]),
        ("company", 2, [], ["y"]),
    ],
)
def test_affixes_skipped(
    word: str, k: int, prefixes: list[str], suffixes: list[str]
) -> None:
    """Tested words are already present in model's vocab.

    Usually, affixes are cut before tokenizing the rest of the word.
    However, if the word is already present in the model's vocab,
    the tokenizer will not cut the affixes and return the full word
    instead.
    """
    tokenizer = AutoFlotaTokenizer.from_pretrained(
        "distilbert-base-uncased",
        FlotaMode.FLOTA,
        k=k,
        prefix_vocab=prefixes,
        suffix_vocab=suffixes,
    )

    assert tokenizer.tokenize(word) == [word]


@pytest.mark.parametrize(
    ("word", "k", "prefixes", "expected"),
    [
        ("simplify", 1, ["un", "de"], ["##plify"]),
        ("undesirable", 1, ["sim"], ["desirable"]),
        ("deescalation", 2, ["non"], ["dee", "##lation"]),
    ],
)
def test_prefixes_not_found_in_word(
    word: str, k: int, prefixes: list[str], expected: list[str]
) -> None:
    """Ensure that prefixes are skipped when not present in input word."""
    tokenizer = AutoFlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA, k=k, prefix_vocab=prefixes
    )

    assert tokenizer.tokenize(word) == expected


@pytest.mark.parametrize(
    ("word", "k", "suffixes", "expected"),
    [
        ("simplify", 1, ["able", "zation"], ["##plify"]),
        ("undesirable", 1, ["ly"], ["desirable"]),
        ("deescalation", 2, ["ify"], ["dee", "##lation"]),
    ],
)
def test_suffixes_not_found_in_word(
    word: str, k: int, suffixes: list[str], expected: list[str]
) -> None:
    """Ensure that suffixes are skipped when not present in input word."""
    tokenizer = AutoFlotaTokenizer.from_pretrained(
        "distilbert-base-uncased", FlotaMode.FLOTA, k=k, suffix_vocab=suffixes
    )

    assert tokenizer.tokenize(word) == expected


def test_prefix_vocab_order() -> None:
    """Ensure that word order in prefix vocab does not matter."""
    prefix_vocab = ["anti", "a"]
    prefix_vocab_reversed = ["a", "anti"]

    # Assume that prefix 'anti' will be split as prefix regardless of
    # input order of prefix_vocab.
    # If the order of prefixes in prefix_vocab matters, then the output
    # would start with ["a", ...], ignoring the larger prefix 'anti'.
    expected = ["anti", "##pas", "##ti"]

    for vocab in [prefix_vocab, prefix_vocab_reversed]:
        tokenizer = AutoFlotaTokenizer.from_pretrained(
            "distilbert-base-uncased", FlotaMode.FLOTA, k=3, prefix_vocab=vocab
        )

        actual = tokenizer.tokenize("antipasti")

        assert actual == expected


def test_suffix_vocab_order() -> None:
    """Ensure that word order in suffix vocab does not matter."""
    suffix_vocab = ["plify", "fy"]
    suffix_vocab_reversed = ["fy", "plify"]

    # Assume that suffix '##plify' will be split as suffix regardless of
    # input order of suffix_vocab.
    # If the order of suffixes in suffix_vocab matters, then the output
    # would end with [..., "y"], ignoring the larger suffix 'plify'.
    expected = ["sim", "##plify"]

    for vocab in [suffix_vocab, suffix_vocab_reversed]:
        tokenizer = AutoFlotaTokenizer.from_pretrained(
            "distilbert-base-uncased", FlotaMode.FLOTA, k=2, suffix_vocab=vocab
        )

        actual = tokenizer.tokenize("simplify")

        assert actual == expected


@pytest.mark.parametrize("model", ["bert", "roberta", "xlnet"])
def test_unsupported_pretrained_model_raises_error(model: str) -> None:
    """Raise error when pretrained model does not exist."""
    expected_error_msg = f"Could not load tokenizer from pretrained model '{model}'"

    with pytest.raises(PretrainedTokenizerLoadError, match=expected_error_msg):
        AutoFlotaTokenizer.from_pretrained(model, FlotaMode.FLOTA)


@pytest.mark.parametrize("model", ["xlm-mlm-en-2048", "ctrl", "unknown-model-name"])
def test_unsupported_model_name_raises_error(model: str) -> None:
    """Raise error when unsupported model is specified."""
    expected_error_msg = f"Model '{model}' is not supported by FlotaTokenizer"

    with pytest.raises(UnsupportedModelError, match=expected_error_msg):
        AutoFlotaTokenizer.from_pretrained(model, FlotaMode.FLOTA)


def test_longest_sort_order() -> None:
    """Special test for FLOTA 'longest' mode to ensure sort order."""
    flota_tokenizer = AutoFlotaTokenizer.from_pretrained(
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
    flota_tokenizer = AutoFlotaTokenizer.from_pretrained("gpt2", FlotaMode.FLOTA, k=4)
    assert flota_tokenizer.tokenize(input_word) == expected
