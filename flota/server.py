"""FLOTA backend server.

This module provides a backend server for FLOTA. It is implemented using
FastAPI and Typer.

The API can be used for tokenizing and encoding input and is meant for
demo purposes.
"""

import fastapi
from fastapi import Depends, Query
from pydantic import BaseModel, Field

from flota import AutoFlotaTokenizer, FlotaMode

app = fastapi.FastAPI()


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
    prefixes: list[str] | None = Query(default=None),
    suffixes: list[str] | None = Query(default=None),
) -> list[str]:
    """Tokenize input word using FlotaTokenizer."""
    flota_tokenizer = AutoFlotaTokenizer.from_pretrained(
        flota_tokenizer_input.model,
        flota_tokenizer_input.mode,
        k=flota_tokenizer_input.k,
        strict=flota_tokenizer_input.strict,
        prefixes=prefixes,
        suffixes=suffixes,
    )
    return flota_tokenizer.tokenize(word)


@app.get("/encode")
async def encode(
    text: str,
    flota_tokenizer_input: FlotaTokenizerInput = Depends(),
    prefixes: list[str] | None = Query(default=None),
    suffixes: list[str] | None = Query(default=None),
) -> list[int]:
    """Encode input text using FlotaTokenizer."""
    flota_tokenizer = AutoFlotaTokenizer.from_pretrained(
        flota_tokenizer_input.model,
        flota_tokenizer_input.mode,
        k=flota_tokenizer_input.k,
        strict=flota_tokenizer_input.strict,
        prefixes=prefixes,
        suffixes=suffixes,
    )
    return flota_tokenizer.encode(text)


def run(host: str, port: int) -> None:
    """Run FLOTA API backend server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
