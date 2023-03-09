"""Utility classes for FLOTA dynamic programming method."""

from __future__ import annotations

import functools
import heapq
from dataclasses import KW_ONLY, InitVar, dataclass, field
from itertools import chain


@dataclass(order=True, frozen=True)
class DPItem:
    """A dynamic programming item that represents a token.

    The DPItem class is used by the dynamic programming algorithm in the
    FlotaTokenizer class to represent a token in a sequence of tokens.

    Parameters
    ----------
    token
        The token represented by this DPItem.
    index
        The index of this DPItem in the sequence of tokens.
    word
        The word that this DPItem's token represents.
    first_match
        A flag indicating whether the represented word is the first in a
        checked sequence of words. If True, then the DPItem's score is
        increased.
    prefer_front
        A flag indicating whether this DPItem should be sorted before
        other DPItems with the same score. If True, then DPItems with
        lower indices will be sorted before DPItems with higher
        indices.

    Attributes
    ----------
    score
        The score assigned to this DPItem by the dynamic programming
        algorithm. The score is calculated based on the length of the
        word represented by the token and whether the DPItem is the
        first in the sequence to match its word.
    sort_index
        The index used to sort DPItems with the same score. If
        prefer_front is True, then the sort_index is negative.
        Otherwise, the sort_index is positive.
    token_index
        The index of this DPItem in the sequence of tokens.

    Returns
    -------
    DPItem:
        An instance of the DPItem class.
    """

    token: str = field(compare=False)
    score: int = field(init=False)
    token_index: int = field(compare=False, init=False)
    sort_index: int = field(init=False)

    index: InitVar[int]
    word: InitVar[str]
    _: KW_ONLY
    first_match: InitVar[bool]
    prefer_front: InitVar[bool] = True

    def __post_init__(
        self,
        index: int,
        word: str,
        first_match: bool,  # noqa: FBT001
        prefer_front: bool,  # noqa: FBT001
    ) -> None:
        """Initialize the DPItem object with the provided parameters."""
        object.__setattr__(self, "token_index", index)
        object.__setattr__(self, "sort_index", (1 - 2 * prefer_front) * index)
        object.__setattr__(self, "score", len(word) ** 3 + int(first_match))


@functools.total_ordering
@dataclass
class DPContainer:
    """A container for dynamic programming items.

    The DPContainer class is used by the dynamic programming algorithm
    in the FlotaTokenizer class to store the DPItems that are being
    considered.

    Parameters
    ----------
    items
        The set of DPItem objects to store in this container. Defaults to
        an empty set.

    Attributes
    ----------
    score
        The score of this container, which is the sum of the scores of all
        DPItem objects in the set.
    tokens
        A list of tokens in this container, sorted by token index.

    Notes
    -----
    This class is decorated with the functools.total_ordering decorator
    to provide comparison operators. DPContainers can be compared to
    each other based on their scores.

    Examples
    --------
    Creating a DPContainer with a set of DPItem objects:

    >>> dp_items = {DPItem('token', 0, 'token', first_match=True), DPItem('ization', 5, 'ization', first_match=False)}
    >>> dp_container = DPContainer(dp_items)
    >>> dp_container.score
    469
    >>> dp_container.tokens
    ['token', 'ization']

    Creating a DPContainer from multiple DPContainers:

    >>> dp_container1 = DPContainer({DPItem('vis', 0, 'vis', first_match=True), DPItem('##ua', 3, 'ua', first_match=True)})
    >>> dp_container2 = DPContainer({DPItem('##li', 5, 'li', first_match=True), DPItem('##zation', 7, 'zation', first_match=True)})
    >>> dp_container = DPContainer.from_structs(3, dp_container1, dp_container2)
    >>> dp_container.score
    254
    >>> dp_container.tokens
    ['vis', '##ua', '##zation']
    """  # noqa: E501

    items: set[DPItem] = field(default_factory=set)

    @classmethod
    def from_structs(cls, k: int | None, *dp_structs: DPContainer) -> DPContainer:
        """Create a new DPContainer from multiple DPContainers.

        This method creates a new DPContainer by combining DPContainers from
        multiple sources. If the parameter k is not None, the set of DPItems
        with the top k scores is returned. Otherwise, all DPItems are included.

        Parameters
        ----------
        k : int or None
            The maximum number of DPItems to include in the new DPContainer.
            If k is None, all DPItems will be included.
        *dp_structs : DPContainer
            A variable number of DPContainers to combine.

        Returns
        -------
        DPContainer
            A new DPContainer containing DPItems from all DPContainers passed as
            arguments.
        """
        items = chain.from_iterable(dps.items for dps in dp_structs)
        return cls(set(heapq.nlargest(k, items) if k else items))

    def __eq__(self, other: object) -> bool:
        """Check if the current container's score equals another's score."""
        if isinstance(other, DPContainer):
            return self.score == other.score

        return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Check if the current container's score is less than another's score."""
        if isinstance(other, DPContainer):
            return self.score < other.score

        return NotImplemented

    @property
    def score(self) -> int:
        """Get the score of the container.

        The score of a container is the sum of the scores of all token
        items in the container.
        """
        return sum(item.score for item in self.items)

    @property
    def tokens(self) -> list[str]:
        """Return all tokens in a container in original word's order."""
        return [item.token for item in sorted(self.items, key=lambda i: i.token_index)]
