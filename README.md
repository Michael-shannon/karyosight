# karyosight

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/michael-shannon/karyosight/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/michael-shannon/karyosight/actions/workflows/tests.yml
[linting-badge]:            https://github.com/michael-shannon/karyosight/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/michael-shannon/karyosight/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/michael-shannon/karyosight/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/michael-shannon/karyosight/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-GPLv3-blue.svg
<!-- prettier-ignore-end -->

ploidy prediction

This project is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University
College London.

## About

### Project Team

Michael Shannon ([m.j.shannon@pm.me](mailto:m.j.shannon@pm.me))

<!-- TODO: how do we have an array of collaborators ? -->

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Built With

<!-- TODO: can cookiecutter make a list of frameworks? -->

- [Framework 1](https://something.com)
- [Framework 2](https://something.com)
- [Framework 3](https://something.com)

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`karyosight` requires Python 3.13.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using
a environment management tool such as
[Conda](https://docs.conda.io/projects/conda/en/stable/). To install the latest
development version of `karyosight` using `pip` in the currently active
environment run

```sh
pip install git+https://github.com/michael-shannon/karyosight.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/michael-shannon/karyosight.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running Locally

How to run the application on your local system.

### Running Tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments
using [`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

### Building Documentation

The MkDocs HTML documentation can be built locally by running

```sh
tox -e docs
```

from the root of the repository. The built documentation will be written to
`site`.

Alternatively to build and preview the documentation locally, in a Python
environment with the optional `docs` dependencies installed, run

```sh
mkdocs serve
```

## Roadmap

- [x] Initial Research
- [ ] Minimum viable product <-- You are Here
- [ ] Alpha Release
- [ ] Feature-Complete Release

## Acknowledgements

This work was funded by The Black Family Foundation.
