import jax.numpy as jnp
import pytest
from scipy.integrate import simpson as scipy_simpson

from ..simpson import simpson


@pytest.mark.parametrize("N", [2, 128, 129])
def test_simpson(N):
    x = jnp.linspace(0., 1., N)
    y = jnp.sin(x)

    result = simpson(y, x)
    expected = scipy_simpson(y, x)

    assert jnp.allclose(result, expected)
