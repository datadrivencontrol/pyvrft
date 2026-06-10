# pyvrft

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20602578-blue)](https://doi.org/10.5281/zenodo.20602578)
[![PyPI](https://img.shields.io/pypi/v/pyvrft.svg)](https://pypi.org/project/pyvrft/)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Virtual Reference Feedback Tuning Toolbox

## Description

This Python toolbox provides commands to design feedback controllers using Virtual Reference Feedback Tuning.
The toolbox implements SISO and MIMO controller design using standard least-squares estimation and instrumental variables.

## Documentation

Documentation is available at <https://pyvrft.net/>.

## Install

Use PIP to install:

```bash
pip install pyvrft
```

The package name on PyPI is `pyvrft`, but the Python import package is `vrft`.

## Use

Please check the *example* folder. Basic use:

```Python
p = vrft.design(u, y, y, Td, C, L)
```
where *u* and *y* are input/output data, *Td* is the reference model, *C* describes the controller structure and *L* is a pre-filter.

## Citation

The archived software release is available at <https://doi.org/10.5281/zenodo.20602578>.
Related SoftwareX article: <https://doi.org/10.1016/j.softx.2019.100383>.
For citation guidance, see <https://pyvrft.net/citation/>.

## Contributors

Diego Eckhard - diegoeck@ufrgs.br - @diegoeck

Emerson Christ Boeira - emerson.boeira@ufrgs.br - @emersonboeira

Lara Colognese de Almeida - lara.almeida@ufrgs.br - @lara-ca
