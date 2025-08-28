# BBQudit  <img width="30" height="30" alt="image" src="https://github.com/user-attachments/assets/95c90d0b-46ee-4520-ac65-14247ae958ca" />


BBQudit is a Python package that constructs and simulates qudit bivariate bicycle codes for any given bivariate polynomials over $GF(p)$. In particular, it provides the following features:
- **Construction** - given two bivariate polynomials over $GF(p)$, initialise a qudit bivariate bicycle code, supplying parity check matrices, $H_X, H_Z$, logicals operators, and qudit connectivity.
- **Visualisation** - draw the Tanner graph of a qudit bivariate bicycle code.
- **Simulation** - simulate a qudit bivariate bicycle code, under either code capacity (depolarising) or circuit-level noise models, using a generalised qudit BP+OSD decoder.
# Demonstrations

Here are some simple examples. For more examples, see the notebooks in the [examples folder](bbq/examples).

Start by importing the package,

```
from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
```

## Initialisation

To construct a bivariate bicycle code, first initialise the field, $GF(p)$ and two bivariate polynomials. For the qutrit toric code, the relevant field and polynomials are:

```
field = Field(3)
x, y = Monomial(field, 'x'), Monomial(field, 'y')

a = 1 - x
b = 1 - y
```

Then the bivariate bicycle code is defined for parameters $l, m, q$, where $l, m$ determine the number of physical qudits, i.e. $x=S_l\otimes I_m$ and $y=I_l\otimes S_m$ for cyclic shift matrix $S$, and $q$ is the CSS parameter defining the parity check matrices, i.e.

$$\qquad H_X=(a, b) \qquad H_Z=(qb^T, (p-q)a^T)$$.


```
bb = BivariateBicycle(a, b, 5, 5, 2, 'Qudit Toric Code')
```

From here we can find various properties of the code, for example
- *parity check matrices* given by ```bb.hx``` and ```bb.hz```.
- *logical operators* given by ```bb.x_logicals``` and ```bb.z_logicals```.
- *code parameters* given by ```bb.parameters```.
- *code connectivity mapping* given by ```bb.qubits_dict```.

For more see the [source code](bbq/bbq_code.py).

## Visualisation

To display the Tanner graph of the bivariate bicycle code, simply use

```
bb.draw()
```

Here are some examples for the toric code with varying qudit dimension, $p$.

![image](https://github.com/user-attachments/assets/4a51c541-a9aa-48bb-a3cb-3146a5ec1d7b)


## Simulation

Simulations are currently only implemented for code capacity (or depolarising) noise, with the relevant code in the experimental notebooks, but we illustrate some initial results for the qudit toric code under code capacity below.

<img width="1127" height="867" alt="image" src="https://github.com/user-attachments/assets/0d11d958-e3a1-431d-b83a-e3b84b919451" />

<br/>

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

