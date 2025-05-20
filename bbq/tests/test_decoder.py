import numpy as np
import pytest

from bbq.decoder import dijkstra


@pytest.mark.parametrize(
    "h, syndrome, distances",
    [
        [
            np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]),
            np.array([0, 0, 1]),
            np.array([5, 3.0, 1.0, 1.0]),
        ],
    ],
)
def test_dijkstra(h, syndrome, distances):
    assert np.allclose(dijkstra(h, syndrome), distances)
