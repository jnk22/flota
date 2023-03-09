"""FLOTA backend server.

This module provides a backend server for FLOTA. It is implemented using
FastAPI and Typer.

The API can be used for tokenizing and encoding input and is meant for
demo purposes.
"""

import fastapi
import typer
import uvicorn
from fastapi import Depends, Query
from pydantic import BaseModel, Field

from flota import FlotaMode, FlotaTokenizer

app = fastapi.FastAPI()
cli = typer.Typer()


class FlotaTokenizerInput(BaseModel):
    """Input fields for FlotaTokenizer.

    model
        Pretrained tokenizer to use. Default is "bert-base-uncased".
    mode
        Flota mode to use. Default is FlotaMode.FLOTA.
    k
        Value of k used in the algorithm. Default is None.
    strict
        Whether to use strict mode for BERT models. Default is False.
    """

    model: str = Field(default="bert-base-uncased")
    mode: FlotaMode = Field(default=FlotaMode.FLOTA)
    k: int | None = Field(default=None)
    strict: bool = Field(default=False)


@app.get("/tokenize")
async def tokenize(
    word: str,
    flota_tokenizer_input: FlotaTokenizerInput = Depends(),
    prefix_vocab: list[str] | None = Query(default=None),
    suffix_vocab: list[str] | None = Query(default=None),
) -> list[str]:
    """Tokenize input word using FlotaTokenizer.

    Parameters
    ----------
    word
        Input word to tokenize.
    flota_tokenizer_input
        Input fields for FlotaTokenizer.
    prefix_vocab
        List of prefix vocabulary.
    suffix_vocab
        List of suffix vocabulary.

    Returns
    -------
    list[str]
        Tokenized input word.
    """
    flota_tokenizer = FlotaTokenizer.from_pretrained(
        flota_tokenizer_input.model,
        flota_tokenizer_input.mode,
        k=flota_tokenizer_input.k,
        strict=flota_tokenizer_input.strict,
        prefix_vocab=prefix_vocab,
        suffix_vocab=suffix_vocab,
    )
    return flota_tokenizer.tokenize(word)


@app.get("/encode")
async def encode(
    text: str,
    flota_tokenizer_input: FlotaTokenizerInput = Depends(),
    prefix_vocab: list[str] | None = Query(default=None),
    suffix_vocab: list[str] | None = Query(default=None),
) -> list[int]:
    """Encode input text using FlotaTokenizer.

    Parameters
    ----------
    text
        Input text to encode.
    flota_tokenizer_input: FlotaTokenizerInput
        Input fields for FlotaTokenizer.
    prefix_vocab
        List of prefix vocabulary.
    suffix_vocab
        List of suffix vocabulary.

    Returns
    -------
    list[int]
        Encoded input text.
    """
    flota_tokenizer = FlotaTokenizer.from_pretrained(
        flota_tokenizer_input.model,
        flota_tokenizer_input.mode,
        k=flota_tokenizer_input.k,
        strict=flota_tokenizer_input.strict,
        prefix_vocab=prefix_vocab,
        suffix_vocab=suffix_vocab,
    )
    return flota_tokenizer.encode(text)


@cli.command()
def main(
    host: str = typer.Option("127.0.0.1", help="Listen host."),
    port: int = typer.Option(8000, help="Listen port."),
) -> None:
    """Run FLOTA API backend server."""
    uvicorn.run(app, host=host, port=port)
