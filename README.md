# BBQudit

BBQudit is a Python package that constructs and simulates qudit bivariate bicycle codes for any given bivariate polynomials over $GF(p)$. In particular, it provides the following features:
- **Construction** - given two bivariate polynomials over $GF(p)$, initialise a qudit bivariate bicycle code, supplying parity check matrices, $H_X, H_Z$, logicals operators, and dictionaries with qudit connectivity.
- **Visualisation** - draw the Tanner graph of a qudit bivariate bicycle code.
- **Simulation** - simulate a qudit bivariate bicycle code, under either code capacity (depolarising) or circuit-level noise models, and using a simple qudit decoder (Dijkstra + OSD).

# Demonstrations

Here are some simple examples.

Start by importing the package,

```
import numpy as np
from bbq.polynomial import Polynomial
from bbq.bbq_code import BivariateBicycle
```

## Initialisation

To construct a bivariate bicycle code, first initialise two bivariate polynomials over a field $p$. For the toric code, the relevant polynomials are:

```
a = Polynomial(3, np.array([[1, 0], [-1, 0]]))
b = Polynomial(3, np.array([[1, -1], [0, 0]]))
```

Then the bivariate bicycle code is defined for parameters $l, m, k$, where $l, m$ determine the number of physical qudits, i.e. $x=S_l\otimes I_m$ and $y=I_l\otimes S_m$ for cyclic shift matrix $S$, and $k$ is the CSS parameter defining the parity check matrices, i.e.

$$\qquad H_X=(a, b) \qquad H_Z=(qb^T, (p-q)a^T)$$.


```
bb = BivariateBicycle(a, b, 5, 5, 2, 'Qudit Toric Code')
```

From here we can find various properties of the code, for example
- *parity check matrices* given by ```bb.hx``` and ```bb.hz```.
- *logical operators* given by ```bb.x_logicals``` and ```bb.z_logicals```.
- *code connectivity mapping* given by ```bb.qubits_dict```.

For more see the [source code](bbq/bbq_code.py)

## Visualisation

To display the Tanner graph of the bivariate bicycle code, simply use

```
bb.draw()
```

Here are some examples for the toric code with varying qudit dimension, $p$.

![image](https://github.com/user-attachments/assets/4a51c541-a9aa-48bb-a3cb-3146a5ec1d7b)


## Simulation

Simulations can be carried out under both code capacity (depolarising) and circuit-level noise. Implementation is still under develpoment, see [here](bbq/simulation.py) for more details, but we illustarte some initial results for the toric code under code capacity below.

![image](https://github.com/user-attachments/assets/08d71b46-3137-4c91-8283-bf689e10f986)

<br/>

Find more examples of code constructions in the [examples](bbq/examples).

# Installation

BBQudit is hosted on [PyPI](https://pypi.org/project/bbqudit/), simply use pip to install it:

```
pip install bbqudit
```

Alternatively, clone the repository and build from source:

```
   git clone https://github.com/e-kneip/bbqudit.git
   cd bbqudit
   pip install -e .
```

