"""Implementation of the Polynomial class over finite fields."""

class Polynomial:
    """Polynomial class over finite fields."""

    def __init__(self, field, coefficients):
        if type(field) is not int:
            raise TypeError("field must be an integer")
        if type(coefficients) is not list:
            raise TypeError("coefficients must be a list")
        for i in coefficients:
            if type(i) is not int:
                raise TypeError("Coefficients must be a list of integers")
        self.field = field
        self.coefficients = [i % field for i in coefficients]

    def __str__(self):
        """String representation of Polynomial."""
        return " + ".join([f"{self.coefficients[i]}x^{i}" for i in range(len(self.coefficients))])

    def __repr__(self):
        """Canonical string representation of Polynomial."""
        return f"Polynomial({self.field}, {self.coefficients})"

    def __add__(self, other):
        """Add two Polynomials."""
        if self.field != other.field:
            raise ValueError("Fields must be the same")
        common = min(len(self.coefficients), len(other.coefficients))
        result = [self.coefficients[i] + other.coefficients[i] for i in range(common)]
        result += self.coefficients[common:] + other.coefficients[common:]
        return Polynomial(self.field, result)

    def __sub__(self, other):
        """Subtract two Polynomials."""
        if self.field != other.field:
            raise ValueError("Fields must be the same")
        common = min(len(self.coefficients), len(other.coefficients))
        result = [self.coefficients[i] - other.coefficients[i] for i in range(common)]
        result += self.coefficients[common:] + [-i for i in other.coefficients[common:]]
        return Polynomial(self.field, result)
