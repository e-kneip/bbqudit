"""Implementation of the BivariateBicycle class for qudits."""

from polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

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
        Defines CSS code construction H_x=(A|B) and H_y=(qB^T|(q-1)A^T).
    """

    def __init__(self, a : Polynomial, b : Polynomial, l : int, m : int, q : int):
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
        if not 0 < q or not q < a.field:
            raise ValueError("q must be a positive integer less than the field of the polynomials")
        if a.field != b.field:
            raise ValueError("Polynomials a and b must be over the same field")
        self.a, self.b = a, b
        self.field = a.field
        self.l, self.m, self.q = l, m, q
        self.hx = np.hstack((a(l, m), b(l, m)))
        self.hy = np.hstack((q * b(l, m).transpose(), (self.field-q) * a(l, m).transpose())) % self.field

    def __str__(self):
        """String representation of BivariateBicycle."""
        return f"Bivariate Bicycle code for\na = {self.a}\nb = {self.b}"

    def __repr__(self):
        """Canonical string epresentation of BivariateBicycle."""
        return f"BivariateBicycle({self.a.__repr__()}, {self.b.__repr__()})"

    def draw(self):
        """Draw the Bivariate Bicycle code Tanner graph."""
        # Define parameters
        hx, hy = self.hx, self.hy
        m, n = hx.shape
        a_coefficients = self.a.coefficients
        b_coefficients = self.b.coefficients

        # Set up plot
        fig, ax = plt.subplots()
        ax.set_xlim(0.5, (n//2)//self.l+0.5)
        ax.set_ylim(0.5, m//self.m+0.5)
        ax.set_aspect('equal', adjustable='box')

        # Define nodes
        def x_stabiliser(x, y):
            return Rectangle((x, y), width=0.05, height=0.05, 
                        edgecolor='lightcoral', facecolor='lightcoral', zorder=3)
        def z_stabiliser(x, y):
            return Rectangle((x, y), width=0.05, height=0.05, 
                        edgecolor='lightseagreen', facecolor='lightseagreen', zorder=3)
        def l_data(x, y):
            return Circle((x, y), radius=0.03, edgecolor='royalblue', facecolor='royalblue', zorder=3)
        def r_data(x, y):
            return Circle((x, y), radius=0.03, edgecolor='gold', facecolor='gold', zorder=3)

        # Draw nodes
        for i in np.arange(0.75, (n//2)//self.l+0.5, 1):
            for j in np.arange(0.75, m//self.m+0.5, 1):
                ax.add_patch(x_stabiliser(i+0.475, j-0.025))
                ax.add_patch(z_stabiliser(i-0.025, j+0.475))
                ax.add_patch(l_data(i+0.5, j+0.5))
                ax.add_patch(r_data(i, j))

        # Draw x stabiliser edges
        for i in range(m):
            for j in range(n//2):
                for k in range(a_coefficients.shape[0]):
                    for l in range(a_coefficients.shape[1]):
                        if a_coefficients[k, l]:
                            ax.plot([0.75+j, 0.25+k+j], [0.25+i, 0.25+l+i], color='slategray')

        for i in range(m):
            for j in range(n//2):
                for k in range(b_coefficients.shape[0]):
                    for l in range(b_coefficients.shape[1]):
                        if b_coefficients[k, l]:
                            ax.plot([0.75+j, 0.75+k+j], [0.25+i, 0.75+l+i], color='slategray')  

        # Draw z stabiliser edges
        for i in range(-1, m):
            for j in range(-1, n//2):
                for k in range(a_coefficients.shape[0]):
                    for l in range(a_coefficients.shape[1]):
                        if a_coefficients[k, l]:
                            ax.plot([0.25+j, 0.75-k+j], [0.75+i, 0.75+l+i], color='darkgray')
        for i in range(-1, m):
            for j in range(-1, n//2):
                for k in range(b_coefficients.shape[0]):
                    for l in range(b_coefficients.shape[1]):
                        if b_coefficients[k, l]:
                            ax.plot([0.25+j, 0.25+k+j], [0.75+i, 0.25-l+i], color='darkgray')


        # Make plot look nice
        ax.set_axis_off()

        ax.legend(['X stabiliser', 'Z stabiliser', 'Left data', 'Right data'], loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title('Tanner Graph');
