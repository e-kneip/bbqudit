"""Implementation of a qudit CSS error correction code."""

import numpy as np
from bbq.field import Field
from bbq.parameters import logicals, bp_distance


class QuditCode:
    """Base class for a qudit CSS error correction code.

    Parameters
    ----------
    field : Field
        The finite field the code is defined over.
    hx : np.ndarray
        The X parity check matrix.
    hz : np.ndarray
        The Z parity check matrix.
    distance : tuple[int] | bool
        The distance of the code (dx, dz): can be preset as ints or if True, will estimate distance, if False, will rough upper bound by logicals.
    name : str
        The name of the code.
    """

    def __init__(
        self,
        field: Field,
        hx: np.ndarray,
        hz: np.ndarray,
        distance: tuple[int] | bool = False,
        name: str = "Qudit Code",
    ):
        if not hx.shape[1] == hz.shape[1]:
            raise ValueError(
                "The number of columns in hx must equal the number of columns in hz."
            )

        self.field = field
        self.hx, self.hz = hx, hz
        self.name = name
        self.distance = distance

        # Compute logicals and parameters
        self.x_logicals, self.z_logicals = logicals(field, hx, hz)
        self.n, self.k = self.hx.shape[1], len(self.x_logicals)
        self.dx, self.dz, self.d = self._compute_distance()

        self.parameters = (self.n, self.k, self.d)

    def __repr__(self):
        """Return canonical string representation of the code."""
        return f"QuditCode({self.field.__repr__()}, {self.hx.__repr__()}, {self.hz.__repr__()}, {self.distance.__repr__()}, {self.name.__repr__()})"

    def __str__(self):
        """Return string representation of the code."""
        if self.distance is False:
            dist = f"<{self.d}"
        else:
            dist = f"{self.d}"
        return f"{self.name}: [[{self.n}, {self.k}, {dist}]]_{self.field.p} \n Hx = {self.hx} \n Hz = {self.hz}"

    def _compute_distance(self):
        """Return distances dx, dz and d of the code."""
        # Rough upper bound from weight of logicals
        if self.distance is False:
            self.dx = min(
                len(np.nonzero(self.x_logicals[i])[0])
                for i in range(len(self.x_logicals))
            )
            self.dz = min(
                len(np.nonzero(self.z_logicals[i])[0])
                for i in range(len(self.z_logicals))
            )
            self.d = min(self.dx, self.dz)

        # Tighter bound from BP distance
        elif self.distance is True:
            self.dx, self.dz = (
                bp_distance(self.field, self.hx, self.x_logicals),
                bp_distance(self.field, self.hz, self.z_logicals),
            )
            self.d = min(self.dx, self.dz)

        # Preset distances
        else:
            self.dx, self.dz = self.distance
            self.d = min(self.dx, self.dz)

        return self.dx, self.dz, self.d
