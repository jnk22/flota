# FLOTA

[![Lint & Test](https://github.com/jnk22/flota/actions/workflows/ci.yml/badge.svg)](https://github.com/jnk22/flota/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/jnk22/flota/branch/main/graph/badge.svg?token=Q5F44R4TTQ)](https://codecov.io/github/jnk22/flota)
[![Maintainability](https://api.codeclimate.com/v1/badges/b39bcc206b0667d336c3/maintainability)](https://codeclimate.com/github/jnk22/flota/maintainability)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d5ecd4974eca494f82201976f424c2ba)](https://www.codacy.com/gh/jnk22/flota/dashboard?utm_source=github.com&utm_medium=referral&utm_content=jnk22/flota&utm_campaign=Badge_Grade)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/jnk22/flota/main.svg)](https://results.pre-commit.ci/latest/github/jnk22/flota/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code and data for the ACL paper
[An Embarrassingly Simple Method to Mitigate Undesirable Properties of
Pretrained Language Model Tokenizers](https://aclanthology.org/2022.acl-short.43.pdf).
The paper introduces FLOTA (Few Longest Token Approximation), a simple yet
effective method to improve the tokenization of pretrained language models.

---

Additionally, this repository contains:

- Code structure updates
- Alternative tokenization method, based on the original FLOTA idea
- Additional passing of **prefix** and **suffix** vocabs to tokenizer
- Simple HTTP API for tokenizing/encoding words
- Improved CLI

## Dependencies

All dependencies are defined in [pyproject.toml](pyproject.toml).

Python 3.8+ is required.

## Installation

Using **pipx**:

```bash
pipx install git+https://github.com/jnk22/flota.git#egg=flota[backend]
```

Using **poetry**:

```bash
git clone https://github.com/jnk22/flota
cd flota
poetry install --extras presentation
```

_The extra package **presentation** installs the HTTP API backend.
This can be omitted if not required._

## Usage

### CLI

```bash
flota bert-base-uncased data/arxiv_cs_1e+02
```

### HTTP API

The FLOTA API is a demo backend that serves an HTTP API for demo purposes.

```bash
flota-api --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) or
[http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

### Predefined Scripts

_TODO_

## Data

- **arXiv Dataset** _(English)_
- **[Ten Thousand German News Articles Dataset](https://tblock.github.io/10kGNAD/)** _(German)_

All datasets can be found in `data`.

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
