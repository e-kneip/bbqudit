"""Implementation of the BivariateBicycle class for qudits."""

from polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from sympy import isprime
import warnings
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


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
        Defines CSS code construction H_x=(A|B) and H_y=(qB^T|(a.field-q)A^T).
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
        if not isprime(self.field):
            warnings.warn("Field is not prime.", ValueWarning)

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
        a_factors = self.a.factor()
        b_factors = self.b.factor()

        # Set up plot
        fig, ax = plt.subplots()
        ax.set_xlim(-0.3, (n//2)//self.l-0.2)
        ax.set_ylim(-0.3, m//self.m-0.2)
        ax.set_aspect('equal', adjustable='box')

        # Define nodes
        def x_stabiliser(x, y):
            return Rectangle((x, y), width=0.1, height=0.1, 
                        edgecolor='lightcoral', facecolor='lightcoral', zorder=3)
        def z_stabiliser(x, y):
            return Rectangle((x, y), width=0.1, height=0.1, 
                        edgecolor='lightseagreen', facecolor='lightseagreen', zorder=3)
        def l_data(x, y):
            return Circle((x, y), radius=0.06, edgecolor='royalblue', facecolor='royalblue', zorder=3)
        def r_data(x, y):
            return Circle((x, y), radius=0.06, edgecolor='gold', facecolor='gold', zorder=3)

        # Draw nodes
        for i in np.arange(0, (n//2)//self.l, 1):
            for j in np.arange(0, m//self.m, 1):
                ax.add_patch(x_stabiliser(i+0.45, j-0.05))
                ax.add_patch(z_stabiliser(i-0.05, j+0.45))
                ax.add_patch(l_data(i+0.5, j+0.5))
                ax.add_patch(r_data(i, j))

        # Draw x stabiliser edges
        for i in range(m//self.m):
            for j in range((n//2)//self.l):
                for k in range(a_coefficients.shape[0]):
                    for l in range(a_coefficients.shape[1]):
                        if a_coefficients[k, l]:
                            div = a_coefficients[k, l]
                            if div == 1:
                                ax.plot([0.5+j, k+j-a_factors[0]], [i, -l+i+a_factors[1]], color='slategray')
                            else:
                                line, = ax.plot([0.5+j, k+j-a_factors[0]], [i, -l+i+a_factors[1]], color='slategray') 
                                line.set_dashes([16/div**2, 2, 16/div**2, 2])
                                line.set_dash_capstyle('round')

        for i in range(m//self.m):
            for j in range((n//2)//self.l):
                for k in range(b_coefficients.shape[0]):
                    for l in range(b_coefficients.shape[1]):
                        if b_coefficients[k, l]:
                            div = b_coefficients[k, l]
                            if div == 1:
                                ax.plot([0.5+j, 0.5+k+j-b_factors[0]], [i, 0.5-l+i+b_factors[1]], color='slategray')
                            else:
                                line, = ax.plot([0.5+j, 0.5+k+j-b_factors[0]], [i, 0.5-l+i+b_factors[1]], color='slategray')  
                                line.set_dashes([16/div**2, 2, 16/div**2, 2])
                                line.set_dash_capstyle('round')

        # Draw z stabiliser edges
        for i in range(m//self.m):
            for j in range((n//2)//self.l):
                for k in range(a_coefficients.shape[0]):
                    for l in range(a_coefficients.shape[1]):
                        if a_coefficients[k, l]:
                            div = (self.q * a_coefficients[k, l]) % self.field
                            if div == 1:
                                ax.plot([j, 0.5-k+j+a_factors[0]], [0.5+i, 0.5+l+i-a_factors[1]], color='darkgray')
                            else:
                                line, = ax.plot([j, 0.5-k+j+a_factors[0]], [0.5+i, 0.5+l+i-a_factors[1]], color='darkgray')
                                line.set_dashes([16/div**2, 2, 16/div**2, 2])
                                line.set_dash_capstyle('round')

        for i in range(m//self.m):
            for j in range((n//2)//self.l):
                for k in range(b_coefficients.shape[0]):
                    for l in range(b_coefficients.shape[1]):
                        if b_coefficients[k, l]:
                            div = ((self.field-self.q) * b_coefficients[k, l]) % self.field
                            if div == 1:
                                ax.plot([j, -k+j+b_factors[0]], [0.5+i, l+i-b_factors[1]], color='darkgray')
                            else:
                                line, = ax.plot([j, -k+j+b_factors[0]], [0.5+i, l+i-b_factors[1]], color='darkgray') 
                                line.set_dashes([16/div**2, 2, 16/div**2, 2])
                                line.set_dash_capstyle('round')

        # Draw boundary
        ax.plot([-0.25, -0.25], [-0.25, m//self.m-0.25], color='black', linewidth=0.7)
        ax.arrow(-0.25, -0.25, 0, m//self.m/2, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.plot([-0.25, (n//2)//self.l-0.25], [-0.25, -0.25], color='black', linewidth=0.7)
        ax.arrow(-0.25, -0.25, ((n//2)//self.l)/2-0.05, 0, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.arrow(-0.25, -0.25, ((n//2)//self.l)/2+0.05, 0, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.plot([-0.25, (n//2)//self.l-0.25], [m//self.m-0.25, m//self.m-0.25], color='black', linewidth=0.7)
        ax.arrow(-0.25, m//self.m-0.25, (m//self.l)/2-0.05, 0, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.arrow(-0.25, m//self.m-0.25, (m//self.l)/2+0.05, 0, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.plot([(n//2)//self.l-0.25, (n//2)//self.l-0.25], [-0.25, m//self.m-0.25], color='black', linewidth=0.7)
        ax.arrow((n//2)//self.l-0.25, -0.25, 0, m//self.m/2, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)

        # Make plot look nice
        ax.set_axis_off()
        ax.set_title('Tanner Graph')

        # Add legend
        handles = ['X stabiliser', 'Z stabiliser', 'Left data', 'Right data']
        lines = []
        patch_colours = ['lightcoral', 'lightseagreen', 'royalblue', 'gold']
        for i in range(4):
            lines.append(mpatches.Patch(color=patch_colours[i]))
        for i in range(1, self.field):
            xline, = ax.plot([0], [0], color='slategray')
            zline, = ax.plot([0], [0], color='darkgray')
            xline.set_dashes([16/i**2, 2, 16/i**2, 2])
            zline.set_dashes([16/i**2, 2, 16/i**2, 2])
            xline.set_dash_capstyle('round')
            zline.set_dash_capstyle('round')
            lines.append(xline)
            lines.append(zline)
            if i==1:
                handles.append('X')
                handles.append('Z')
            else:
                handles.append(f'X^{i}')
                handles.append(f'Z^{i}')
        ax.legend(lines, handles, loc='upper left', bbox_to_anchor=(1, 1), handlelength=2.4);
