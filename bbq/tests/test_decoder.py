import numpy as np
import pytest

from bbq.decoder import dijkstra, d_osd, bp_osd


@pytest.mark.parametrize(
    "h, syndrome, distances",
    [
        [
            np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]),
            np.array([0, 0, 1]),
            np.array([5.0, 3.0, 1.0, 1.0]),
        ],
    ],
)
def test_dijkstra(h, syndrome, distances):
    assert np.allclose(dijkstra(h, syndrome), distances)


@pytest.mark.parametrize(
    "field, h_eff, syndrome, prior, order, debug, error,",
    [
        [
            2,
            np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]]),
            np.array([0, 0, 1]),
            np.array([[0.9, 0.1], [0.9, 0.1], [0.8, 0.2], [0.8, 0.2]]),
            0,
            False,
            np.array([0, 0, 1, 1]),
        ],
    ],
)
def test_d_osd(field, h_eff, syndrome, prior, order, debug, error):
    predicted_error, success = d_osd(field, h_eff, syndrome, prior, order, debug)
    assert success
    assert np.allclose(predicted_error, error)


@pytest.mark.parametrize(
    "field, h_eff, syndrome, prior, max_iter, order, debug, error,",
    [
        [
            2,
            np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]]),
            np.array([0, 0, 1]),
            np.array([[0.9, 0.1], [0.9, 0.1], [0.8, 0.2], [0.8, 0.2]]),
            1000,
            0,
            False,
            np.array([0, 0, 1, 1]),
        ],
    ],
)
def test_bp_osd(field, h_eff, syndrome, prior, max_iter, order, debug, error):
    predicted_error, success = bp_osd(
        field, h_eff, syndrome, prior, max_iter, order, debug
    )
    assert success
    assert np.allclose(predicted_error, error)
