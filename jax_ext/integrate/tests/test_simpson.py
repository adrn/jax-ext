import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import simpson as scipy_simpson

from ..simpson import ln_simpson, simpson

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("N", [2, 128, 129])
def test_simpson(N):
    x = jnp.linspace(0., 1., N)
    y = jnp.sin(x)

    result = simpson(y, x)
    expected = scipy_simpson(y, x)

    assert jnp.allclose(result, expected, atol=1e-16)


@pytest.mark.parametrize("N", [2, 128, 129])
def test_ln_simpson(N):
    x = jnp.linspace(0., 1., N)
    y = jnp.sin(x)**2 + 0.341  # random positive function

    result = ln_simpson(jnp.log(y), x)
    expected = jnp.log(scipy_simpson(y, x))

    assert jnp.allclose(result, expected, atol=1e-15)
