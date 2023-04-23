"""Module for running FLOTA tokenization."""

from pathlib import Path
from typing import Optional

from typer import Argument, Exit, Option, Typer

from .enums import FlotaMode, NoiseType, ResultFileExistsMode, TokenizeMode

IntOrNone = Optional[int]
PathOrNone = Optional[Path]
BoolOrNone = Optional[bool]


def version_callback(*, value: bool) -> None:
    """Print version and exit."""
    if value:
        from .cli import print_version

        print_version()
        raise Exit


CLI_MODEL_NAME = Argument(..., help="Name of model")
CLI_DATASET = Argument(
    ..., help="CSV data file path without suffixes '[_train|_dev|_test].csv'"
)
CLI_WORDS_ENCODE = Argument(..., help="Words to encode")
CLI_WORDS_TOKENIZE = Argument(..., help="Words to tokenize")
CLI_LEARNING_RATE = Option(1e-5, min=1e-12, help="Learning rate")
CLI_THETA = Option(0.3, min=0, max=1, help="Theta value for enabled noise")
CLI_BATCH_SIZE = Option(64, min=1, help="Batch size")
CLI_EPOCHS = Option(20, min=1, help="Number of epochs")
CLI_K = Option(
    None,
    min=0,
    help="Number of maximum subwords, excluding prefix/suffix (0/None=unlimited)",
)
CLI_CACHE_SIZE = Option(
    default=None, help="FLOTA internal cache size (0=disable, None=unlimited)"
)
CLI_RANDOM_SEED = Option(None, help="Random seed")
CLI_CUDA_DEVICE = Option("0", help="Selected CUDA device")
CLI_PREFIXES = Option(None, exists=True, help="Prefix vocabulary path")
CLI_SUFFIXES = Option(None, exists=True, help="Suffix vocabulary path")
CLI_OUTPUT = Option("results", help="Output directory for results")
CLI_NOISE = Option(
    NoiseType.NONE.value, case_sensitive=False, help="Noise for input data"
)
CLI_MODE = Option(
    TokenizeMode.FLOTA.value, case_sensitive=False, help="FLOTA mode or base"
)
CLI_RESULTS_EXIST = Option(
    default=ResultFileExistsMode.APPEND.value,
    help="Overwrite, append or skip if results file already exists",
)
CLI_STRICT = Option(default=False, help="Use strict mode")
CLI_HOST = Option("127.0.0.1", help="Listen host.")
CLI_PORT = Option(8000, help="Listen port.")
CLI_VERSION = Option(None, "--version", callback=version_callback, is_eager=True)

cli = Typer()


@cli.command()
def run(  # noqa: PLR0913
    model_name: str = CLI_MODEL_NAME,
    dataset: str = CLI_DATASET,
    *,
    learning_rate: float = CLI_LEARNING_RATE,
    theta: float = CLI_THETA,
    batch_size: int = CLI_BATCH_SIZE,
    epochs: int = CLI_EPOCHS,
    k: IntOrNone = CLI_K,
    cache_size: IntOrNone = CLI_CACHE_SIZE,
    random_seed: IntOrNone = CLI_RANDOM_SEED,
    cuda_device: str = CLI_CUDA_DEVICE,
    prefixes: PathOrNone = CLI_PREFIXES,
    suffixes: PathOrNone = CLI_SUFFIXES,
    output: Path = CLI_OUTPUT,
    noise: NoiseType = CLI_NOISE,
    mode: TokenizeMode = CLI_MODE,
    results_exist: ResultFileExistsMode = CLI_RESULTS_EXIST,
    strict: bool = CLI_STRICT,
) -> None:
    """Run FLOTA tokenization."""
    from .cli import run

    run(
        model_name,
        dataset,
        learning_rate=learning_rate,
        theta=theta,
        batch_size=batch_size,
        epochs=epochs,
        k=k,
        cache_size=cache_size,
        random_seed=random_seed,
        cuda_device=cuda_device,
        prefixes=prefixes,
        suffixes=suffixes,
        output=output,
        noise=noise,
        mode=mode,
        results_exist=results_exist,
        strict=strict,
    )


@cli.command()
def encode(  # noqa: PLR0913
    model_name: str = CLI_MODEL_NAME,
    words: list[str] = CLI_WORDS_ENCODE,
    *,
    k: IntOrNone = CLI_K,
    cache_size: IntOrNone = CLI_CACHE_SIZE,
    prefixes: PathOrNone = CLI_PREFIXES,
    suffixes: PathOrNone = CLI_SUFFIXES,
    mode: FlotaMode = CLI_MODE,
    strict: bool = CLI_STRICT,
) -> None:
    """Encode input words.

    Output will be separated into a single line for each word.
    """
    from .cli import encode

    encode(
        model_name,
        words,
        k=k,
        cache_size=cache_size,
        prefixes=prefixes,
        suffixes=suffixes,
        mode=mode,
        strict=strict,
    )


@cli.command()
def tokenize(  # noqa: PLR0913
    model_name: str = CLI_MODEL_NAME,
    words: list[str] = CLI_WORDS_TOKENIZE,
    *,
    k: IntOrNone = CLI_K,
    cache_size: IntOrNone = CLI_CACHE_SIZE,
    prefixes: PathOrNone = CLI_PREFIXES,
    suffixes: PathOrNone = CLI_SUFFIXES,
    mode: FlotaMode = CLI_MODE,
    strict: bool = CLI_STRICT,
) -> None:
    """Tokenize input words.

    Output will be separated into a single line for each word.
    """
    from .cli import tokenize

    tokenize(
        model_name,
        words,
        k=k,
        cache_size=cache_size,
        prefixes=prefixes,
        suffixes=suffixes,
        mode=mode,
        strict=strict,
    )


@cli.command()
def server(host: str = CLI_HOST, port: int = CLI_PORT) -> None:
    """Run FLOTA API backend server."""
    from .server import run

    run(host, port)


@cli.callback()
def callback(*, value: bool = CLI_VERSION) -> None:  # noqa: ARG001
    """Run application to print version."""
