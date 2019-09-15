# pyvrft

Virtual Reference Feedback Tuning

## Description

This Python Toolbox provides commands to design feedback controllers using the method Virtual Reference Feedback Tuning.
The toolbox implements both SISO and MIMO controllers, using standard least-squares implementation and instrumental variables.

## Install

Use PIP to install:

```bash
pip install pyvrft
```

## Use

Please check the *example* folder. Basic use:

```Python
p = vrft.design(u, y, y, Td, C, L)
```
where *u* and *y* are input/output data, *Td* is the reference mode, *C* describes the controller structure and *L* is a pre-filter.

## Contributors

Diego Eckhard - diegoeck@ufrgs.br - @diegoeck

Emerson Christ Boeira - emerson.boeira@ufrgs.br - @emersonboeira
