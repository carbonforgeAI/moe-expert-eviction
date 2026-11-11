import numpy as np
import pytest

from moe_eviction.cache import NormalizedExponentialRecency


def test_recency_decays_and_receives_impulses() -> None:
    trace = NormalizedExponentialRecency(num_experts=4, capacity=10.0, gamma=0.5)

    first = trace.update((1,))
    second = trace.update((1,))
    third = trace.update((2,))

    np.testing.assert_allclose(first, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(second, [0.0, 1.5, 0.0, 0.0])
    np.testing.assert_allclose(third, [0.0, 0.75, 1.0, 0.0])


def test_recency_scales_down_to_capacity() -> None:
    trace = NormalizedExponentialRecency(num_experts=4, capacity=1.0, gamma=0.5)
    state = trace.update((0, 1))
    assert state.sum() == pytest.approx(1.0)
