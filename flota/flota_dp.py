"""Utility classes for FLOTA dynamic programming method."""

from __future__ import annotations

import functools
from dataclasses import KW_ONLY, dataclass, field
from itertools import chain


@dataclass(frozen=True, slots=True)
class DPItem:
    """An item representing a token and its score.

    The DPItem class is used by the dynamic programming algorithm in the
    FlotaTokenizer class to represent a token in a sequence of tokens.

    Parameters
    ----------
    token
        The token represented by this DPItem.
    word
        The word that this DPItem's token represents.
    index
        The index of this DPItem in the sequence of tokens.
    first_match
        A flag indicating whether the represented word is the first in a
        checked sequence of words.

    Attributes
    ----------
    score

    Returns
    -------
    DPItem:
        An instance of the DPItem class.
    """

    token: str
    word: str
    index: int
    _: KW_ONLY
    first_match: bool

    @property
    def score(self) -> tuple[int, bool, int]:
        """Score tuple.

        The score consists of three elements:
            1. length of word squared
            2. first_match flag
            3. index of the DPItem in the sequence of tokens
        """
        return len(self.word) ** 2, self.first_match, self.index


@functools.total_ordering
@dataclass(frozen=True, slots=True)
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

    Notes
    -----
    This class is decorated with the functools.total_ordering decorator
    to provide comparison operators. DPContainers can be compared to
    each other based on their scores.

    Examples
    --------
    Creating a DPContainer with a set of DPItem objects:

    >>> dp_items = {DPItem('token', 'token', 0, first_match=True), DPItem('ization', 'ization', 5, first_match=False)}
    >>> dp_container = DPContainer(dp_items)
    >>> dp_container.score
    (74, 1, 5)
    >>> dp_container.tokens
    ['token', 'ization']

    Creating a DPContainer from multiple DPContainers:

    >>> dp_container1 = DPContainer({DPItem('vis', 'vis', 0, first_match=True), DPItem('##ua', 'ua', 3, first_match=True)})
    >>> dp_container2 = DPContainer({DPItem('##li', 'li', 5, first_match=True), DPItem('##zation', 'zation', 7, first_match=True)})
    >>> dp_container = DPContainer.from_structs(dp_container1, dp_container2)
    >>> dp_container.score
    (53, 4, 15)
    >>> dp_container.tokens
    ['vis', '##ua', '##li', '##zation']
    """  # noqa: E501

    items: set[DPItem] = field(default_factory=set)

    @classmethod
    def from_containers(cls, *dp_containers: DPContainer) -> DPContainer:
        """Create a new DPContainer from multiple DPContainers.

        This method creates a new DPContainer by combining DPContainers from
        multiple sources. If the parameter k is not None, the set of DPItems
        with the top k scores is returned. Otherwise, all DPItems are included.

        Parameters
        ----------
        *dp_structs : DPContainer
            A variable number of DPContainers to combine.

        Returns
        -------
        DPContainer
            A new DPContainer containing DPItems from all DPContainers passed as
            arguments.
        """
        return cls(set(chain.from_iterable(dp_c.items for dp_c in dp_containers)))

    def __eq__(self, other: object) -> bool:
        """Check if the current container's score equals another's score."""
        return self.score == other.score if isinstance(other, DPContainer) else False

    def __lt__(self, other: object) -> bool:
        """Check if the current container's score is less than another's score."""
        return self.score < other.score if isinstance(other, DPContainer) else False

    @property
    def score(self) -> tuple[int, ...]:
        """Get the score of the container.

        The score of a container is the sum of the scores of all token
        items in the container.
        """
        scores = (item.score for item in self.items)
        return tuple(sum(score) for score in zip(*scores, strict=True))

    @property
    def tokens(self) -> list[str]:
        """Return all tokens in a container in original word's order."""
        return [item.token for item in sorted(self.items, key=lambda i: i.index)]
