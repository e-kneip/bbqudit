# BBQudit

BBQudit is a Python package that constructs and simulates qudit bivariate bicycle codes for any given bivariate polynomials over $GF(p)$. In particular, it provides the following features:
- **Construction** - given 2 bivariate polynomials over $GF(p)$, initialise a qudit bivariate bicycle code, supplying parity check matrices, $H_X, H_Z$, logicals operators, and dictionaries with qudit connectivity.
- **Visualisation** - draw the Tanner graph of a qudit bivariate bicycle code.
- **Simulation** - simulate a qudit bivariate bicycle code, under either code capacity (depolarising) or circuit-level noise models, and using a simple qudit decoder (Dijkstra + OSD).

# Demonstrations

Here are some simple examples.

Start by importing the package

*TBD*

## Initialisation

*TBD*

## Visualisation

*TBD*

## Simulation

*TBD*

Find more examples in *TBD*

# Installation

BBQudit is hosted on [PyPI](https://pypi.org/project/bbqudit/), simply use pip to install it:

```
pip install bbqudit
```

Alternatively, clone the repository and build from source:

```
   git clone https://github.com/e-kneip/bbqudit.git
   cd qudit-bivariate-bicycle
   pip install -e .
```

