"""Module for FLOTA tokenization methods."""

from __future__ import annotations

import contextlib
import functools
import re
from abc import ABC, abstractmethod
from itertools import chain
from typing import TYPE_CHECKING, NamedTuple

import torch
from transformers import AutoTokenizer

from .enums import FlotaMode
from .exceptions import PretrainedTokenizerLoadError, UnsupportedModelError
from .flota_dp import DPContainer, DPItem

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from tokenizers import Tokenizer


class CacheInfo(NamedTuple):
    """Representation wrapper for `functools.lru_cache` info."""

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


class FlotaTokenizer(ABC):
    """Abstract base class for FLOTA tokenization methods."""

    __SPECIAL = "-"

    def __init__(  # noqa: PLR0913
        self,
        tokenizer: Tokenizer,
        mode: FlotaMode,
        *,
        k: int | None = None,
        strict: bool = False,
        cache_size: int | None = 128,
        prefixes: Iterable[str] | None = None,
        suffixes: Iterable[str] | None = None,
        **_: Any,  # noqa: ANN401
    ) -> None:
        """Initialize a FlotaTokenizer class instance.

        Parameters
        ----------
        tokenizer
            The pre-trained tokenizer to use for the model.
        mode
            The mode to use for the model.
        k
            The number of clusters to use for the model, by default 3.
        strict
            Use strict mode for tokenization.
        cache_size
            Use cache for tokenization, by default 128. Set to None for
            unlimited cache size, 0 to disable.
        prefixes
            Prefix vocabulary to use for the model.
        suffixes
            Suffix vocabulary to use for the model.

        Returns
        -------
        None
        """
        self._tok: Any = tokenizer
        self._vocab: set[str] = set(self._tok.vocab.keys())
        self.__vocab_max_len: int = max(len(word) for word in self._vocab)
        self.__mode: FlotaMode = mode
        self.__k: int | None = k if k and k > 0 else None
        self.__strict: bool = strict

        self.__prefixes: tuple[str, ...] = (
            self.__build_prefixes(prefixes) if prefixes else ()
        )
        self.__suffixes: tuple[str, ...] = (
            self.__build_suffixes(suffixes) if suffixes else ()
        )

        # Wrap tokenize function with built-in cache.
        # This is a workaround as the decorator approach may leak memory:
        # https://rednafi.github.io/reflections/dont-wrap-instance-methods-with-functoolslru_cache-decorator-in-python.html
        self.tokenize = functools.lru_cache(cache_size)(  # type: ignore[assignment]
            self.tokenize
        )

        # Enable dynamic programming for FLOTA tokenization.
        self.__build_dynamic = functools.cache(  # type: ignore[assignment]
            self.__build_dynamic
        )

    @property
    def cache_info(self) -> CacheInfo:
        """Return cache statistics for `tokenize` method."""
        return self.tokenize.cache_info()  # type: ignore[attr-defined]

    @property
    def _pad_token_id(self) -> int:
        # Return the pad token id of the model.
        return self._tok.pad_token_id

    @property
    def _pre_encode_token_ids(self) -> tuple[int, ...]:
        # Return token ids of special tokens that should be prepended
        # to the tokenized text. If the model does not have
        # a special token, return an empty tuple.
        return ()

    @property
    def _post_encode_token_ids(self) -> tuple[int, ...]:
        # Return token ids of special tokens that should be appended
        # to the tokenized text. If the model does not have
        # a special token, return an empty tuple.
        return ()

    @functools.cached_property
    def __encoded_max_length(self) -> int:
        # Return the maximum length of the encoded text, based on model
        # maximum length and pre/post token ids.
        return (
            self._tok.model_max_length
            - len(self._pre_encode_token_ids)
            + len(self._post_encode_token_ids)
        )

    @property
    @abstractmethod
    def _special_token(self) -> str:
        """Return the special token used by the tokenizer."""

    def _word_combinations(
        self, word: str, *, start: bool, default_vocab: bool
    ) -> tuple[str, ...]:
        # Return word combinations in order including special token.
        # If strict mode is enabled, only return the word itself.
        if self.__strict and not default_vocab and start:
            return (word,)

        combined = self._special_token + word
        return (combined, word) if start else (word, combined)

    def __call__(self, texts: Iterable[str]) -> dict[str, torch.Tensor]:
        """Tokenize and encode a list of texts and return them as torch tensors.

        Parameters
        ----------
        texts
            An iterable of texts to be tokenized and encoded.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary with keys "input_ids" and "attention_mask" and
            values as the corresponding torch tensors.

        Examples
        --------
        >>> tokenizer = AutoFlotaTokenizer.from_pretrained('distilbert-base-cased', FlotaMode.FLOTA, k=5)
        >>> texts = ['Sample text', 'to be tokenized']
        >>> tokenized = tokenizer(texts)
        >>> tokenized['input_ids']
        tensor([[  101,   156, 26671,  3087,   102,     0],
                [  101,  1106,  1129, 22559,  2200,   102]])
        >>> tokenized['attention_mask']
        tensor([[1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1]])
        """  # noqa: E501
        encoded_texts = [self.encode(text) for text in texts]
        batch_size = len(encoded_texts)
        max_len = max(len(text) for text in encoded_texts)

        input_ids = self._pad_token_id * torch.ones((batch_size, max_len)).long()
        attention_mask = torch.zeros_like(input_ids).long()

        for i, text in enumerate(encoded_texts):
            input_ids[i, self._tensor_text_input_index(text)] = torch.tensor(text)
            attention_mask[i, self._tensor_text_input_index(text)] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a list of integers.

        The text is first tokenized using the FLOTA method, then the
        tokenized text is converted into a list of integers using the
        tokenizer's `convert_tokens_to_ids` method. The resulting list
        is then truncated to the maximum length specified for the
        tokenizer.

        Parameters
        ----------
        text
            The input text to be encoded.

        Returns
        -------
        list[int]
            A list of integers representing the encoded input text.

        Examples
        --------
        >>> tokenizer = AutoFlotaTokenizer.from_pretrained('distilbert-base-cased', FlotaMode.FLOTA, k=3)
        >>> text = 'This is a sample text to be tokenized and encoded'
        >>> tokenizer.encode(text)
        [101, 1188, 1110, 170, 6876, 3087, 1106, 1129, 22559, 2200, 1105, 12544, 102]

        >>> text = 'This is another example'
        >>> tokenizer.encode(text)
        [101, 1188, 1110, 1330, 1859, 102]
        """  # noqa: E501
        words = re.findall(r"[\w]+|[^\s\w]", text)
        tokens = (token for word in words for token in self.tokenize(word))
        token_ids = self._tok.convert_tokens_to_ids(tokens)[: self.__encoded_max_length]

        return [*self._pre_encode_token_ids, *token_ids, *self._post_encode_token_ids]

    def tokenize(self, word: str) -> list[str]:
        """Tokenize the input word using the FLOTA method.

        If the word is in the vocabulary, it is returned as is. If the
        word is not in the vocabulary, it is tokenized according to the
        mode specified in the tokenizer's initialization. If the
        tokenizer was initialized with a prefix/suffix vocabulary, the
        word is first split into a prefix and a suffix before being
        tokenized.

        Parameters
        ----------
        word
            The input word to be tokenized.

        Returns
        -------
        list[str]
            A list of strings representing the tokenized word.

        Examples
        --------
        >>> tokenizer = AutoFlotaTokenizer.from_pretrained('distilbert-base-cased', FlotaMode.FLOTA, k=2)
        >>> word = 'example'
        >>> tokenizer.tokenize(word)
        ['example']

        >>> word = 'tokenization'
        >>> tokenizer.tokenize(word)
        ['token', '##ization']

        >>> word = 'mutagenic'
        >>> tokenizer.tokenize(word)
        ['##uta', '##genic']
        """  # noqa: E501
        token, _ = self.__word_in_vocab(word, default_vocab=True)
        if token:
            return [token]

        prefix, word = self.__split_prefix(word) if self.__prefixes else (None, word)
        suffix, word = self.__split_suffix(word) if self.__suffixes else (None, word)
        tokens = self.__tokenize_method(word, start=not prefix)

        return [token for token in (prefix, *tokens, suffix) if token]

    @staticmethod
    def _tensor_text_input_index(text: list[int]) -> slice:
        # Default index for tensor text input.
        # This defines the range of the input_ids and attention_mask.
        return slice(-len(text), None)

    def _tokenize_flota(self, word: str, *, start: bool) -> list[str]:
        # FLOTA tokenization method, returns a list of subwords based
        # on max_subword_split.
        flota_dict = self.__build_flota_dict(word, self.__k, start=start)
        return [token for _, token in sorted(flota_dict.items())]

    def _tokenize_flota_dp(self, word: str, *, start: bool) -> list[str]:
        # FLOTA dynamic tokenization method using memoization.
        tokens = self.__build_dynamic(word, start=start).tokens
        self.__build_dynamic.cache_clear()  # type: ignore[attr-defined]

        return tokens

    def _tokenize_first(self, word: str, **_: bool) -> list[str]:
        # FLOTA first method, returns the first 'k' subwords.
        return self._tok.tokenize(word)[: self.__k]

    def _tokenize_longest(self, word: str, **_: bool) -> list[str]:
        # FLOTA longest method, returns the 'k' longest subwords.
        topk_tokens = sorted(
            enumerate(self._tok.tokenize(word)),
            key=lambda x: len(x[1].lstrip(self._special_token)),
            reverse=True,
        )[: self.__k]

        return [token for _, token in sorted(topk_tokens, key=lambda x: x[0])]

    def __split_prefix(self, word: str) -> tuple[str | None, str]:
        # Split provided word into prefix and stem if possible.
        with contextlib.suppress(StopIteration):
            prefix = next(px for px in self.__prefixes if word.startswith(px))
            return prefix, f"{self._special_token}{word[len(prefix):]}"

        return None, word

    def __split_suffix(self, word: str) -> tuple[str | None, str]:
        # Split provided word into suffix and stem if possible.
        with contextlib.suppress(StopIteration):
            suffix = next(sx for sx in self.__suffixes if word.endswith(sx))
            return f"{self._special_token}{suffix}", word[: -len(suffix)]

        return None, word

    def __tokenize_method(self, word: str, **kwargs: bool) -> list[str]:
        # Tokenize based on the tokenizer's mode.
        match self.__mode:
            case FlotaMode.FLOTA:
                return self._tokenize_flota(word, **kwargs)
            case FlotaMode.FLOTA_DP:
                return self._tokenize_flota_dp(word, **kwargs)
            case FlotaMode.FIRST:
                return self._tokenize_first(word, **kwargs)
            case FlotaMode.LONGEST:
                return self._tokenize_longest(word, **kwargs)
            case _:
                return NotImplemented

    def __build_flota_dict(
        self, word: str, recursive_k: int | None, *, start: bool
    ) -> dict[int, str]:
        # Return a dictionary of subword indices and subwords.
        finished_word = len(word) * self.__SPECIAL
        if (recursive_k is not None and recursive_k < 1) or finished_word == word:
            return {}

        flota_dict, next_word = self.__max_subword_split(word, start=start)

        if next_word:
            next_k = recursive_k - 1 if recursive_k is not None else None
            return flota_dict | self.__build_flota_dict(next_word, next_k, start=start)

        return flota_dict

    def __max_subword_split(
        self, word: str, *, start: bool
    ) -> tuple[dict[int, str], str | None]:
        # Max subword split function.
        # Find the longest subword in the word that is in the vocabulary.
        for j in range(min(len(word), self.__vocab_max_len), 0, -1):
            for i in range(len(word) - j + 1):
                if word[i] == self.__SPECIAL:
                    continue

                subword = word[i : i + j]
                token, _ = self.__word_in_vocab(subword, start=(start and i == 0))
                if token:
                    return {i: token}, word[:i] + j * self.__SPECIAL + word[i + j :]

        return {}, None

    def __build_dynamic(self, word: str, index: int = 0, *, start: bool) -> DPContainer:
        # Dynamic programming method for FLOTA tokenization.
        if not word:
            return DPContainer()

        # Add current token token to results if whole word is in vocabulary.
        token, first_match = self.__word_in_vocab(word, start=start)
        if token:
            return DPContainer({DPItem(token, word, index, first_match=first_match)})

        # Generate all possible subword pairs with their respective scores.
        token_candidates = []
        split_words = ((word[:i], word[i:], i) for i in range(1, len(word)))
        for sw_left, sw_right, i in split_words:
            dp_left = self.__build_dynamic(sw_left, index, start=True and start)
            dp_right = self.__build_dynamic(sw_right, index + i, start=False)
            token_candidates.append(DPContainer.from_structs(dp_left, dp_right))

        # Reverse to keep longer parts of the beginning of the word.
        return max(reversed(token_candidates), default=DPContainer())

    def __word_in_vocab(
        self, word: str, *, start: bool = True, default_vocab: bool = False
    ) -> tuple[str | None, bool]:
        # Return the first token in the vocabulary, None otherwise.
        test_words = self._word_combinations(
            word, start=start, default_vocab=default_vocab
        )

        return next(
            ((w, i == 0) for i, w in enumerate(test_words) if w in self._vocab),
            (None, False),
        )

    def __build_prefixes(self, prefixes: Iterable[str]) -> tuple[str, ...]:
        # Build prefix vocabulary for the model, sorted by length in descending order.
        prefixes = chain.from_iterable({(px.title(), px.lower()) for px in prefixes})
        filtered = (word for word in prefixes if word in self._vocab)
        return tuple(sorted(filtered, key=len, reverse=True))

    def __build_suffixes(self, suffixes: Iterable[str]) -> tuple[str, ...]:
        # Build suffix vocabulary for the model, sorted by length in descending order.
        special = self._special_token
        filtered = (
            word for word in suffixes if f"{special}{word.lower()}" in self._vocab
        )
        return tuple(sorted(filtered, key=len, reverse=True))


class BertFlotaTokenizer(FlotaTokenizer):
    """FLOTA tokenizer for BERT models."""

    @property
    def _special_token(self) -> str:
        return self._tok.decoder.prefix

    @property
    def _pre_encode_token_ids(self) -> tuple[int, ...]:
        return (self._tok.cls_token_id,)

    @property
    def _post_encode_token_ids(self) -> tuple[int, ...]:
        return (self._tok.sep_token_id,)

    def _word_combinations(self, word: str, **kwargs: bool) -> tuple[str, ...]:
        # Special case for BERT models: Use reversed default.
        return tuple(reversed(super()._word_combinations(word, **kwargs)))

    @staticmethod
    def _tensor_text_input_index(text: list[int]) -> slice:
        return slice(None, len(text))


class XLNetFlotaTokenizer(FlotaTokenizer):
    """FLOTA tokenizer for XLNet models."""

    @property
    def _special_token(self) -> str:
        return self._tok.decoder.replacement

    @property
    def _post_encode_token_ids(self) -> tuple[int, ...]:
        return self._tok.sep_token_id, self._tok.cls_token_id


class GPT2FlotaTokenizer(FlotaTokenizer):
    """FLOTA tokenizer for GPT-2 models."""

    @property
    def _special_token(self) -> str:
        return "\u0120"

    @property
    def _pad_token_id(self) -> int:
        return self._tok.eos_token_id

    def _tokenize_first(self, word: str, **kwargs: bool) -> list[str]:
        return super()._tokenize_first(f" {word}", **kwargs)

    def _tokenize_longest(self, word: str, **kwargs: bool) -> list[str]:
        return super()._tokenize_longest(f" {word}", **kwargs)


class AutoFlotaTokenizer:
    """Class for creating model specific `FlotaTokenizer`."""

    __MAPPING: dict[str, type[FlotaTokenizer]] = {
        "bert": BertFlotaTokenizer,
        "xlnet": XLNetFlotaTokenizer,
        "gpt2": GPT2FlotaTokenizer,
    }

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        mode: FlotaMode,
        *,
        k: int | None = None,
        model_max_length: int = 512,
        **kwargs: int | bool | Iterable[str] | None,
    ) -> FlotaTokenizer:
        """Create a FlotaTokenizer class instance from a pre-trained model.

        Parameters
        ----------
        model
            The name of the pre-trained model to load.
        mode
            The mode to use for the model.
        k
            The number of clusters to use for the model.
        model_max_length
            Maximum length of the encoded text.
        **kwargs
            Additional keyword arguments to pass to the tokenizer.

        Returns
        -------
        FlotaTokenizer
            A class instance of the FlotaTokenizer that matches the
            specified pre-trained model.

        Raises
        ------
        UnsupportedModelError
            If the specified model is not supported by FlotaTokenizer.
        PretrainedTokenizerLoadError
            If the tokenizer could not be loaded from the pre-trained model.
        """
        try:
            # Find matching FlotaTokenizer class for the specified model.
            flota_tokenizer_class = next(
                ft_class
                for ft_key, ft_class in cls.__MAPPING.items()
                if ft_key in model
            )

            pretrained_tokenizer = AutoTokenizer.from_pretrained(
                model, model_max_length=model_max_length
            )

        except StopIteration as exc:
            err_msg = f"Model '{model}' is not supported by FlotaTokenizer"
            raise UnsupportedModelError(err_msg) from exc

        except (OSError, ValueError) as exc:
            err_msg = f"Could not load tokenizer from pretrained model '{model}'"
            raise PretrainedTokenizerLoadError(err_msg) from exc

        else:
            return flota_tokenizer_class(
                pretrained_tokenizer, mode, k=k, **kwargs  # type: ignore[arg-type]
            )
