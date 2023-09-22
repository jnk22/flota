# FLOTA

[![Lint & Test](https://github.com/jnk22/flota/actions/workflows/ci.yml/badge.svg)](https://github.com/jnk22/flota/actions/workflows/ci.yml)
[![Python version](https://img.shields.io/badge/python-3.10-blue)](./pyproject.toml)
[![codecov](https://codecov.io/github/jnk22/flota/branch/main/graph/badge.svg?token=Q5F44R4TTQ)](https://codecov.io/github/jnk22/flota)
[![Maintainability](https://api.codeclimate.com/v1/badges/b39bcc206b0667d336c3/maintainability)](https://codeclimate.com/github/jnk22/flota/maintainability)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d5ecd4974eca494f82201976f424c2ba)](https://www.codacy.com/gh/jnk22/flota/dashboard?utm_source=github.com&utm_medium=referral&utm_content=jnk22/flota&utm_campaign=Badge_Grade)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/jnk22/flota/main.svg)](https://results.pre-commit.ci/latest/github/jnk22/flota/main)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code and data for the ACL paper
[An Embarrassingly Simple Method to Mitigate Undesirable Properties of
Pretrained Language Model Tokenizers](https://aclanthology.org/2022.acl-short.43.pdf).
The paper introduces FLOTA (Few Longest Token Approximation), a simple yet
effective method to improve the tokenization of pretrained language models.

---

Additionally, this repository contains:

- Updated code structure
- Alternative tokenization method, based on the original FLOTA idea
- Additional passing of **prefixes** and **suffixes** to tokenizer
- Simple HTTP API for tokenizing/encoding words
- Improved CLI

## Dependencies

All dependencies are defined in [pyproject.toml](pyproject.toml).

## Installation

Using **pipx**:

```bash
pipx install "git+https://github.com/jnk22/flota.git#egg=flota[api,cli]"
```

Using **poetry**:

```bash
git clone https://github.com/jnk22/flota
cd flota
poetry install --extras "api cli"
```

_The extra packages **api** and **cli** are only required for the CLI
application and HTTP server. These can be omitted if only used as
library._

## Usage

### Python library

```python
from flota import AutoFlotaTokenizer, FlotaMode

# Original mode: FLOTA, k=3
flota = AutoFlotaTokenizer.from_pretrained("bert-base-uncased", FlotaMode.FLOTA, k=3)
print(flota.tokenize("visualization"))  # ['vis', '#ua', '##lization']

# Additional mode: FLOTA-DP
flota = AutoFlotaTokenizer.from_pretrained("bert-base-uncased", FlotaMode.FLOTA_DP)
print(flota.tokenize("visualization"))  # ['visual', '##ization']
```

### CLI application

_This requires the installation of extra packages **CLI**!_

#### Run performance tests

```bash
flota run bert-base-uncased data/arxiv_cs_1e+02
```

#### Tokenize words

```bash
flota tokenize bert-base-uncased this is an example text to be tokenized
```

#### Encode words

```bash
flota encode bert-base-uncased this is an example text to be encoded
```

### HTTP server

The FLOTA server is a demo backend that serves an HTTP API for demo purposes.

```bash
flota server --host 127.0.0.1 --port 8000

# In another terminal:
curl -X 'GET' 'http://127.0.0.1:8000/tokenize?word=visualization&model=bert-base-uncased&mode=flota'
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) or
[http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) for OpenAPI
visualizations.

## Data

- **arXiv Dataset** _(English)_
- **[Ten Thousand German News Articles Dataset](https://tblock.github.io/10kGNAD/)** _(German)_

All datasets are available in [`data`](./data/).

## Citation

If you use the code or data in this repository, please cite the following paper:

```bib
@inproceedings{hofmann2022flota,
    title = {An Embarrassingly Simple Method to Mitigate Undesirable Properties of Pretrained Language Model Tokenizers},
    author = {Hofmann, Valentin and Sch{\"u}tze, Hinrich and Pierrehumbert, Janet},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
    year = {2022}
}
```
