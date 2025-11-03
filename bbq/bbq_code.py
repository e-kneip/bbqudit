"""Implementation of the BivariateBicycle class for qudits."""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.patches as mpatches
import galois

from scipy.sparse import coo_matrix, hstack
from matplotlib.patches import Rectangle, Circle
from sympy import isprime
from bbq.polynomial import Polynomial


class ValueWarning(UserWarning):
    pass


class BivariateBicycle:
    """Implementation of the Bivariate Bicycle code on qudits.

    Parameters
    ----------
    a : Polynomial
        Polynomial a over the finite field.
    b : Polynomial
        Polynomial b over the finite field.
    l : int
        Dimension of left cyclic shift matrix.
    m : int
        Dimension of right cyclic shift matrix.
    q : int
        Defines CSS code construction H_x=(A|B) and H_y=(qB^T|(a.field.p-q)A^T).
    name : str
        Name of the code.
    """

    def __init__(
        self, a: Polynomial, b: Polynomial, l: int, m: int, q: int, name: str = None
    ):
        if not isinstance(a, Polynomial):
            raise TypeError("a must be a Polynomial")
        if not isinstance(b, Polynomial):
            raise TypeError("b must be a Polynomial")
        if not isinstance(l, int):
            raise TypeError("l must be an integer")
        if not isinstance(m, int):
            raise TypeError("m must be an integer")
        if not isinstance(q, int):
            raise TypeError("q must be an integer")
        if not 0 < q or not q < a.field.p:
            raise ValueError(
                "q must be a positive integer less than the field of the polynomials"
            )
        if a.field.p != b.field.p:
            raise ValueError("Polynomials a and b must be over the same field")
        if not isprime(a.field.p):
            print("Warning: Field is not prime.")
            warnings.warn("Field is not prime.", ValueWarning)
        if not (isinstance(name, str) or name == None):
            raise TypeError("name must be a string")
        self.a, self.b = a, b
        self.field = a.field
        self.l, self.m, self.q = l, m, q
        self.hx = np.hstack((a(l, m), b(l, m)))
        self.hz = (
            np.hstack(
                (q * b(l, m).transpose(), (self.field.p - q) * a(l, m).transpose())
            )
            % self.field.p
        )
        self.A, self.B = self._monomials()
        self.qudits_dict, self.data_qudits, self.Xchecks, self.Zchecks = (
            self._qudits()
        )
        self.edges = self._edges()
        self.name = name
        self.x_logicals, self.z_logicals = self._compute_logicals()
        self.distance = None

        self.parameters = [self.hx.shape[1], len(self.x_logicals), self.distance]

        if not self.x_logicals:
            print("Warning: No X logicals found for these parameters.")
            warnings.warn("No X logicals found for these parameters.", ValueWarning)
        if not self.z_logicals:
            print("Warning: No Z logicals found for these parameters.")
            warnings.warn("No Z logicals found for these parameters.", ValueWarning)

    def __str__(self):
        """String representation of BivariateBicycle."""
        return f"Bivariate Bicycle code for\na(x, y) = {self.a}\nb(x, y) = {self.b}"

    def __repr__(self):
        """Canonical string epresentation of BivariateBicycle."""
        return f"BivariateBicycle({self.a.__repr__()}, {self.b.__repr__()})"

    def _monomials(self):
        """Construct monomials for the Bivariate Bicycle code."""
        a, b = self.a, self.b
        l, m = self.l, self.m
        A, B = [], []
        row, col = np.nonzero(a.coefficients)
        for i in range(len(row)):
            poly_coef = np.zeros((a.coefficients.shape), dtype=int)
            poly_coef[row[i], col[i]] = a.coefficients[row[i], col[i]]
            poly = Polynomial(a.field, poly_coef)
            A.append(poly(l, m))
        row, col = np.nonzero(b.coefficients)
        for i in range(len(row)):
            poly_coef = np.zeros((b.coefficients.shape), dtype=int)
            poly_coef[row[i], col[i]] = b.coefficients[row[i], col[i]]
            poly = Polynomial(b.field, poly_coef)
            B.append(poly(l, m))
        return A, B

    def _qudits(self):
        """Give names to each qudit and store in a dictionary: (qudit_type, qudit_type_number) : qudit_index"""
        l, m = self.l, self.m
        qudits_dict = {}
        data_qudits, Xchecks, Zchecks = [], [], []
        for i in range(l * m):
            # X checks
            node_name = ("Xcheck", i)
            Xchecks.append(node_name)
            qudits_dict[node_name] = i

        for i in range(l * m):
            # Left data qudits
            node_name = ("data_left", i)
            data_qudits.append(node_name)
            qudits_dict[node_name] = l * m + i

        for i in range(l * m):
            # Right data qudits
            node_name = ("data_right", i)
            data_qudits.append(node_name)
            qudits_dict[node_name] = 2 * l * m + i

        for i in range(l * m):
            # Z checks
            node_name = ("Zcheck", i)
            Zchecks.append(node_name)
            qudits_dict[node_name] = 3 * l * m + i
        return qudits_dict, data_qudits, Xchecks, Zchecks

    def _edges(self):
        """Set up edges connecting data and measurement qudits in a dictionary: ((check_qudit_type, check_type_number), monomial_index/direction) : (qudit_type, qudit_number)"""
        l, m = self.l, self.m
        q = self.q
        field = self.field
        A, B = self.A, self.B
        edges = {}
        for i in range(l * m):
            # X checks
            check_name = ("Xcheck", i)
            # Left data qudits
            for j in range(len(A)):
                y = int(np.nonzero(A[j][i, :])[0][0])
                edges[(check_name, j)] = (("data_left", y), int(A[j][i, y]))
            # Right data qudits
            for j in range(len(B)):
                y = int(np.nonzero(B[j][i, :])[0][0])
                edges[(check_name, len(A) + j)] = (("data_right", y), int(B[j][i, y]))

        for i in range(l * m):
            # Z checks
            check_name = ("Zcheck", i)
            # Left data qudits
            for j in range(len(B)):
                y = int(np.nonzero(B[j][:, i])[0][0])
                edges[(check_name, j)] = (
                    ("data_left", y),
                    (q * int(B[j][y, i])) % field.p,
                )
            # Right data qudits
            for j in range(len(A)):
                y = int(np.nonzero(A[j][:, i])[0][0])
                edges[(check_name, len(A) + j)] = (
                    ("data_right", y),
                    ((field.p - q) * int(A[j][y, i])) % field.p,
                )
        return edges

    def _compute_logicals(self):
        """Compute logical operators for the code."""
        hx, hz = self.hx, self.hz
        field = self.field

        # Set up Galois field array
        GF = galois.GF(field.p)
        Hx_gal, Hz_gal = GF(hx), GF(hz)
        x_logicals, z_logicals = [], []
        Xcheck, Zcheck = Hx_gal, Hz_gal

        # X logicals must be in the kernel of Hz and not the image of Hx^T
        ker_hz = Hz_gal.null_space()
        rank = np.linalg.matrix_rank(Hx_gal)
        for vec in ker_hz:
            Xcheck = GF(np.vstack((Xcheck, vec)))
            if np.linalg.matrix_rank(Xcheck) > rank:
                x_logicals.append(vec)
                rank += 1
            else:
                np.delete(Xcheck, -1, axis=0)

        # Z logicals must be in the kernel of Hx and not the image of Hz^T
        ker_hx = Hx_gal.null_space()
        rank = np.linalg.matrix_rank(Hz_gal)
        for vec in ker_hx:
            Zcheck = GF(np.vstack((Zcheck, vec)))
            if np.linalg.matrix_rank(Zcheck) > rank:
                z_logicals.append(vec)
                rank += 1
            else:
                np.delete(Zcheck, -1, axis=0)

        # Check correct number of logicals found: k = n - m
        assert len(x_logicals) == len(z_logicals)
        m = np.linalg.matrix_rank(Hx_gal) + np.linalg.matrix_rank(Hz_gal)
        n = self.hx.shape[1]
        if not len(x_logicals) == n - m:
            raise ValueError("Incorrect number of logical operators found.")

        return [x_log.__array__(dtype=int) for x_log in x_logicals], [
            z_log.__array__(dtype=int) for z_log in z_logicals
        ]

    def draw(self):
        """Draw the Bivariate Bicycle code Tanner graph."""
        # Define parameters
        hx, hz = self.hx, self.hz
        m, n = hx.shape
        a_coefficients, b_coefficients = self.a.coefficients, self.b.coefficients
        a_factors_min, a_factors_max = self.a.factor()
        b_factors_min, b_factors_max = self.b.factor()
        x_max = max(a_factors_max[0], b_factors_max[0])
        y_max = max(a_factors_max[1], b_factors_max[1])
        name = self.name

        # Set up plot
        fig, ax = plt.subplots()
        ax.set_xlim(-0.3, (n // 2) // self.l - 0.2)
        ax.set_ylim(-0.3, m // self.m - 0.2)
        ax.set_aspect("equal", adjustable="box")

        # Define nodes
        def x_stabiliser(x, y):
            return Rectangle(
                (x, y),
                width=0.1,
                height=0.1,
                edgecolor="lightcoral",
                facecolor="lightcoral",
                zorder=3,
            )

        def z_stabiliser(x, y):
            return Rectangle(
                (x, y),
                width=0.1,
                height=0.1,
                edgecolor="lightseagreen",
                facecolor="lightseagreen",
                zorder=3,
            )

        def l_data(x, y):
            return Circle(
                (x, y),
                radius=0.06,
                edgecolor="royalblue",
                facecolor="royalblue",
                zorder=3,
            )

        def r_data(x, y):
            return Circle(
                (x, y), radius=0.06, edgecolor="gold", facecolor="gold", zorder=3
            )

        # Draw nodes
        for i in np.arange(0, (n // 2) // self.l, 1):
            for j in np.arange(0, m // self.m, 1):
                ax.add_patch(x_stabiliser(i + 0.45, j - 0.05))
                ax.add_patch(z_stabiliser(i - 0.05, j + 0.45))
                ax.add_patch(l_data(i + 0.5, j + 0.5))
                ax.add_patch(r_data(i, j))

        for i in range(-x_max, x_max + m // self.m):
            for j in range(-y_max, y_max + (n // 2) // self.l):
                for k in range(a_coefficients.shape[0]):
                    for l in range(a_coefficients.shape[1]):
                        # Draw x stabiliser edges
                        if a_coefficients[k, l]:
                            div = a_coefficients[k, l]
                            if div == 1:
                                ax.plot(
                                    [0.5 + j, k + j - a_factors_min[0]],
                                    [i, -l + i + a_factors_min[1]],
                                    color="slategray",
                                )
                            else:
                                (line,) = ax.plot(
                                    [0.5 + j, k + j - a_factors_min[0]],
                                    [i, -l + i + a_factors_min[1]],
                                    color="slategray",
                                )
                                line.set_dashes([16 / div**2, 2, 16 / div**2, 2])
                                line.set_dash_capstyle("round")

                        # Draw z stabiliser edges
                        if a_coefficients[k, l]:
                            div = (self.q * a_coefficients[k, l]) % self.field.p
                            if div == 1:
                                ax.plot(
                                    [j, 0.5 - k + j + a_factors_min[0]],
                                    [0.5 + i, 0.5 + l + i - a_factors_min[1]],
                                    color="darkgray",
                                )
                            else:
                                (line,) = ax.plot(
                                    [j, 0.5 - k + j + a_factors_min[0]],
                                    [0.5 + i, 0.5 + l + i - a_factors_min[1]],
                                    color="darkgray",
                                )
                                line.set_dashes([16 / div**2, 2, 16 / div**2, 2])
                                line.set_dash_capstyle("round")

                for k in range(b_coefficients.shape[0]):
                    for l in range(b_coefficients.shape[1]):
                        # Draw x stabiliser edges
                        if b_coefficients[k, l]:
                            div = b_coefficients[k, l]
                            if div == 1:
                                ax.plot(
                                    [0.5 + j, 0.5 + k + j - b_factors_min[0]],
                                    [i, 0.5 - l + i + b_factors_min[1]],
                                    color="slategray",
                                )
                            else:
                                (line,) = ax.plot(
                                    [0.5 + j, 0.5 + k + j - b_factors_min[0]],
                                    [i, 0.5 - l + i + b_factors_min[1]],
                                    color="slategray",
                                )
                                line.set_dashes([16 / div**2, 2, 16 / div**2, 2])
                                line.set_dash_capstyle("round")

                        # Draw z stabiliser edges
                        if b_coefficients[k, l]:
                            div = (
                                (self.field.p - self.q) * b_coefficients[k, l]
                            ) % self.field.p
                            if div == 1:
                                ax.plot(
                                    [j, -k + j + b_factors_min[0]],
                                    [0.5 + i, l + i - b_factors_min[1]],
                                    color="darkgray",
                                )
                            else:
                                (line,) = ax.plot(
                                    [j, -k + j + b_factors_min[0]],
                                    [0.5 + i, l + i - b_factors_min[1]],
                                    color="darkgray",
                                )
                                line.set_dashes([16 / div**2, 2, 16 / div**2, 2])
                                line.set_dash_capstyle("round")

        # Draw boundary
        ax.plot(
            [-0.25, -0.25], [-0.25, m // self.m - 0.25], color="black", linewidth=0.7
        )
        ax.arrow(
            -0.25,
            -0.25,
            0,
            m // self.m / 2,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.plot(
            [-0.25, (n // 2) // self.l - 0.25],
            [-0.25, -0.25],
            color="black",
            linewidth=0.7,
        )
        ax.arrow(
            -0.25,
            -0.25,
            ((n // 2) // self.l) / 2 - 0.05,
            0,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.arrow(
            -0.25,
            -0.25,
            ((n // 2) // self.l) / 2 + 0.05,
            0,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.plot(
            [-0.25, (n // 2) // self.l - 0.25],
            [m // self.m - 0.25, m // self.m - 0.25],
            color="black",
            linewidth=0.7,
        )
        ax.arrow(
            -0.25,
            m // self.m - 0.25,
            (m // self.l) / 2 - 0.05,
            0,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.arrow(
            -0.25,
            m // self.m - 0.25,
            (m // self.l) / 2 + 0.05,
            0,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.plot(
            [(n // 2) // self.l - 0.25, (n // 2) // self.l - 0.25],
            [-0.25, m // self.m - 0.25],
            color="black",
            linewidth=0.7,
        )
        ax.arrow(
            (n // 2) // self.l - 0.25,
            -0.25,
            0,
            m // self.m / 2,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )

        # Make plot look nice
        ax.set_axis_off()
        if name:
            ax.set_title(f"Tanner Graph of {name}")
        else:
            ax.set_title("Tanner Graph")

        # Add legend
        handles = ["X stabiliser", "Z stabiliser", "Left data", "Right data"]
        lines = []
        patch_colours = ["lightcoral", "lightseagreen", "royalblue", "gold"]
        for i in range(4):
            lines.append(mpatches.Patch(color=patch_colours[i]))
        for i in range(1, self.field.p):
            (xline,) = ax.plot([0], [0], color="slategray")
            (zline,) = ax.plot([0], [0], color="darkgray")
            xline.set_dashes([16 / i**2, 2, 16 / i**2, 2])
            zline.set_dashes([16 / i**2, 2, 16 / i**2, 2])
            xline.set_dash_capstyle("round")
            zline.set_dash_capstyle("round")
            lines.append(xline)
            lines.append(zline)
            if i == 1:
                handles.append("X")
                handles.append("Z")
            else:
                handles.append(f"X^{i}")
                handles.append(f"Z^{i}")
        ax.legend(
            lines, handles, loc="upper left", bbox_to_anchor=(1, 1), handlelength=2.4
        )
